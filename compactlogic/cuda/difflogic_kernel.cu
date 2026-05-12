#include "utils.cuh"

#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <vector>

// This file implements the train/eval CUDA kernels for the dense LogicLayer.
//
// Layout conventions:
//   - x / grad_x: [in_dim, batch]
//   - y / grad_y: [out_dim, batch]
//   - a / b:      [out_dim, num_gates]
//   - w:          [out_dim, num_gates]
//
// Dense-layer backward_x uses precomputed inverse connectivity metadata:
//   given_x_indices_of_y_start / given_x_indices_of_y / given_x_indices_of_gate.
// This removes atomic accumulation in the optimized regular-layer path.

#define BACKWARD_W_BATCH_THREADS 1
#define THREADS_PER_BLOCK 512
#define BACKWARD_W_MAX_THREADS 512

namespace {

inline int choose_dense_backward_w_threads(const int64_t batch_size) {
    int threads = 1;
    while (threads < batch_size && threads * 2 <= BACKWARD_W_MAX_THREADS) {
        threads *= 2;
    }
    return threads;
}

} // namespace

// Forward pass: one thread computes one (out neuron, batch item) output by looping over candidate gates.
template <typename scalar_t>
__global__ void logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> gate_seq
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {
            scalar_t y_ = static_cast<scalar_t>(0);
            for (  // num_gates dim
                auto idx_gate = 0;
                idx_gate < w.size(1);
                ++idx_gate
            ){

                const auto idx_a = a[col][idx_gate];
                const auto idx_b = b[col][idx_gate];
                const auto a_ = x[idx_a][row];
                const auto b_ = x[idx_b][row];

                const auto w_ = w[col][idx_gate];

                y_ += w_ * bin_op<scalar_t>(a_, b_, gate_seq[col][idx_gate]);
            }
            y[col][row] = y_;
        }
    }
}


// Weight-gradient pass: one block reduces over the batch dimension for one (out neuron, gate) pair.
template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_w
) {
    auto col = blockIdx.x;
    auto idx_gate = blockIdx.y;
    const auto tid = threadIdx.x;
    float sum = 0.0f;

    const auto idx_a = a[col][idx_gate];
    const auto idx_b = b[col][idx_gate];
    const auto gate_type = gate_seq[col][idx_gate];
    for (
        auto row = tid;
        row < x.size(1);
        row += blockDim.x
    ) {
        const auto grad_y_ = grad_y[col][row];
        const auto a_ = x[idx_a][row];
        const auto b_ = x[idx_b][row];
        sum += static_cast<float>(grad_y_ * bin_op<scalar_t>(a_, b_, gate_type));
    }

    const auto lane = tid & 31;
    const auto warp_id = tid >> 5;
    const auto warp_count = (blockDim.x + 31) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    extern __shared__ unsigned char shared_mem[];
    float* warp_sums = reinterpret_cast<float*>(shared_mem);
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    if (tid == 0) {
        grad_w[col][idx_gate] = static_cast<scalar_t>(block_sum);
    }
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y,
    torch::PackedTensorAccessor64<int8_t, 1, torch::RestrictPtrTraits> given_x_indices_of_gate
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);
            col += blockDim.y * gridDim.y
        ) {

            scalar_t grad_x_ = static_cast<scalar_t>(0);

            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];

            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                const auto gate_code = given_x_indices_of_gate[cur];
                const bool idx_is_a = gate_code >= 0;
                const auto idx_gate = idx_is_a ? gate_code : static_cast<int8_t>(-gate_code - 1);

                const auto grad_y_ = grad_y[idx_y][row];

                // compute grad_x
                if (idx_is_a) {
                    const auto idx_b = b[idx_y][idx_gate];
                    const auto b_ = x[idx_b][row];
                    const auto dy_dx = w[idx_y][idx_gate] * bin_op_grad_a<scalar_t>(b_, gate_seq[idx_y][idx_gate]);
                    grad_x_ += dy_dx * grad_y_;
                } else {
                    const auto idx_a = a[idx_y][idx_gate];
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = w[idx_y][idx_gate] * bin_op_grad_b<scalar_t>(a_, gate_seq[idx_y][idx_gate]);
                    grad_x_ += dy_dx * grad_y_;
                }
            }
            grad_x[col][row] = grad_x_;
    }}
}


torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);

    const auto batch_size = x.size(1);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_forward", ([&] {
                           logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               gate_seq.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());

    return y;
}


torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor gate_seq,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(gate_seq);
    CHECK_INPUT(grad_y);


    const auto batch_size = x.size(1);
    const auto out_size = grad_y.size(0);
    const auto num_gates = a.size(1);

    auto grad_w = torch::empty({out_size, num_gates}, torch::dtype(x.dtype()).device(x.device()));

    const int threads = choose_dense_backward_w_threads(batch_size);
    dim3 threads_per_block(threads);

    const dim3 blocks_per_grid(
        static_cast<int64_t>(out_size),
        static_cast<int64_t>(num_gates)
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_backward_w", ([&] {
                           size_t shared_mem_bytes = ((threads_per_block.x + 31) / 32) * sizeof(float);
                           logic_layer_cuda_backward_w_kernel<scalar_t>
                           <<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
                               x.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                               gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                       }));

    gpuErrchk(cudaPeekAtLastError());

    return grad_w;
}


torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y,
    torch::Tensor given_x_indices_of_gate
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);
    CHECK_INPUT(given_x_indices_of_gate);

    auto grad_x = torch::empty_like(x);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(  // batch dim, in dim
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_backward_x", ([&] {
                           logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               gate_seq.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y_start.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               given_x_indices_of_gate.packed_accessor64<int8_t, 1, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());

    return grad_x;
}


/**********************************************************************************************************************/
/**  INFERENCE MODE  **************************************************************************************************/
/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {
    for (  // neuron dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(0);
        row += blockDim.x * gridDim.x
    ) {
        const auto w_ = w[row];
        const auto idx_a = a[row][w_];
        const auto idx_b = b[row][w_];
        const auto gate_type = gate_seq[row][w_];
        for (  // batch dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(1);
            col += blockDim.y * gridDim.y
        ) {
            const auto a_ = x[idx_a][col];
            const auto b_ = x[idx_b][col];
            y[row][col] = bin_op(a_, b_, gate_type);
        }
    }
}

torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    auto y = torch::zeros({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(16, 16);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(y.size(0), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(y.size(1), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_eval_kernel", ([&] {
                                   logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                       x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                                       b.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                                       w.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>(),
                                       gate_seq.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
                                       y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                                   );
                               }));

    gpuErrchk(cudaPeekAtLastError());

    return y;
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void tensor_packbits_cuda_kernel(
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b
) {

    for (  // neuron in b and t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in b
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < b.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            } val;
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            val.signed_scalar = b[row][col];
            for (unsigned int i = 0; i < bit_count; ++i) {
                const auto t_col = bit_count * col + i;
                if (t_col < t.size(1)) {    
                    const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(t[row][t_col]) << i;
                    val.unsigned_scalar = val.unsigned_scalar | bit_mask;
                }
            }
            b[row][col] = val.signed_scalar;
        }
    }
}

std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
) {
    CHECK_INPUT(t);

    const auto batch_in_size = t.size(1);
    const auto batch_out_size = ceil_div(batch_in_size, static_cast<int64_t>(bit_count));
    const auto out_size = t.size(0);
    const auto pad_len = (bit_count - batch_in_size % bit_count) % bit_count;

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto dispatch_type = [bit_count]() {
        switch (bit_count) {
        case 8:
            return torch::kInt8;
        case 16:
            return torch::kInt16;
        case 32:
            return torch::kInt32;
        case 64:
            return torch::kInt64;
        default:
            throw std::invalid_argument("`bit_count` has to be in { 8, 16, 32, 64 }");
        }
    }();
    auto b = torch::zeros({out_size, batch_out_size}, torch::dtype(dispatch_type).device(t.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.scalar_type(), "tensor_packbits_cuda_kernel", ([&] {
                                   tensor_packbits_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(t.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                                                                                                                            b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));
    gpuErrchk(cudaPeekAtLastError());

    return {b, pad_len};
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> t
) {

    for (  // class in t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in t
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < t.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union scalar_t_ {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            };
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            int res = 0;
            const auto class_size = b.size(0) / t.size(0);
            for (int i = 0; i < class_size; ++i) {
                const scalar_t_ val = {.signed_scalar = b[row * class_size + i][col / bit_count]};
                const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);
                res += !!(val.unsigned_scalar & bit_mask);
            }
            t[row][col] = res;
        }
    }
}

torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
) {
    CHECK_INPUT(b);

    const int bit_count = 8 * b.element_size();

    const auto batch_in_size = b.size(1);
    const auto in_size = b.size(0);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);
    assert(in_size % k == 0);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(torch::kInt32).device(b.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.scalar_type(), "groupbitsum_kernel", ([&] {
                                   groupbitsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                        b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                        t.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
                                        );
                               }));
    gpuErrchk(cudaPeekAtLastError());

    return t.transpose(0, 1).contiguous();
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_x_kernel_atomicAdd(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_x
) {

    const auto neuron_idx = blockIdx.x;
    const auto gate_idx = blockIdx.z;
    const auto idx_a = a[neuron_idx][gate_idx];
    const auto idx_b = b[neuron_idx][gate_idx];
    const auto gate_type = gate_seq[neuron_idx][gate_idx];
    const auto w_ = w[neuron_idx][gate_idx];

    for (  // batch dim
        auto batch_idx = blockIdx.y * blockDim.x + threadIdx.x;
        batch_idx < x.size(1);
        batch_idx += blockDim.x * gridDim.y
    ) {

        const auto grad_y_ = grad_y[neuron_idx][batch_idx];
        const auto a_ = x[idx_a][batch_idx];
        const auto b_ = x[idx_b][batch_idx];
        const auto t = w_ * grad_y_;

        auto dy_dx_0 = t * bin_op_grad_a<scalar_t>(b_, gate_type);
        atomicAdd(&grad_x[idx_a][batch_idx], dy_dx_0);

        auto dy_dx_1 = t * bin_op_grad_b<scalar_t>(a_, gate_type);
        atomicAdd(&grad_x[idx_b][batch_idx], dy_dx_1);
    }
}


torch::Tensor logic_layer_cuda_backward_x_atomicAdd(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);
    CHECK_INPUT(grad_y);

    auto grad_x = torch::zeros_like(x);
    const auto batch_size = x.size(1);

    dim3 threads_per_block(min(static_cast<int64_t>(THREADS_PER_BLOCK), static_cast<int64_t>(batch_size)));

    // check the validity of blockDim.y and blockDim.z
    const dim3 blocks_per_grid(  // out dim, batch dim, gate_idx dim
        grad_y.size(0),
        ceil_div(static_cast<int64_t>(batch_size), static_cast<int64_t>(threads_per_block.x)),
        a.size(1)
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "logic_layer_cuda_backward_x_atomicAdd", ([&] {
           logic_layer_cuda_backward_x_kernel_atomicAdd<scalar_t><<<blocks_per_grid, threads_per_block>>>(
               x.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               a.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
               b.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
               w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
               grad_y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               grad_x.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
           );
       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}
