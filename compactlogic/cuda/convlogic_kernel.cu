#include "utils.cuh"
#include <torch/extension.h>

// This file implements the train-time and eval-time CUDA kernels for ConvLayer.
//
// High-level tensor layout conventions:
//   - x / grad_x: [batch, c_in, h_in, w_in]
//   - y / grad_y: [batch, c_out, h_out, w_out]
//   - offsets_*:  [operand(2), c_out, num_gates]
//   - w:          [c_out, num_gates]
//
// Parallelization strategy:
//   - forward:     one thread owns one (batch, c_out, out_y, out_x) output value and loops over gates
//   - backward_x:  one thread owns one (batch, c_out, out_y, out_x, gate) contribution and atomically scatters
//   - backward_w:  one block owns one (c_out, gate) weight gradient and reduces over batch * spatial positions
//
// These mappings match the current mathematical formulation and are kept explicit for maintainability.

#define THREADS_PER_BLOCK 256

namespace {

struct ConvForwardLaunch {
    dim3 blocks;
    dim3 threads;
};

inline ConvForwardLaunch make_conv_forward_launch(
    const int64_t batch_size,
    const int64_t c_out,
    const int64_t spatial_size
) {
    int threads_c = 1;
    int threads_spatial = min(spatial_size, static_cast<int64_t>(THREADS_PER_BLOCK));
    if (threads_spatial * 2 <= THREADS_PER_BLOCK) {
        threads_c = min(static_cast<int64_t>(THREADS_PER_BLOCK / threads_spatial), c_out);
    }

    return ConvForwardLaunch{
        dim3(
            batch_size,
            ceil_div(c_out, static_cast<int64_t>(threads_c)),
            ceil_div(spatial_size, static_cast<int64_t>(threads_spatial))
        ),
        dim3(threads_c, threads_spatial)
    };
}

struct ConvBackwardXLaunch {
    dim3 blocks;
    dim3 threads;
};

inline ConvBackwardXLaunch make_conv_backward_x_launch(
    const int64_t batch_size,
    const int64_t c_out,
    const int64_t spatial_size,
    const int64_t num_gates
) {
    int threads_c = 1;
    int threads_spatial = min(spatial_size, static_cast<int64_t>(THREADS_PER_BLOCK / num_gates));
    if (threads_spatial * num_gates * 2 <= THREADS_PER_BLOCK) {
        threads_c = min(static_cast<int64_t>(THREADS_PER_BLOCK / (threads_spatial * num_gates)), c_out);
    }

    return ConvBackwardXLaunch{
        dim3(
            batch_size,
            ceil_div(c_out, static_cast<int64_t>(threads_c)),
            ceil_div(spatial_size, static_cast<int64_t>(threads_spatial))
        ),
        dim3(threads_c, threads_spatial, num_gates)
    };
}

inline int choose_conv_backward_w_threads(const int64_t reduction_size) {
    int threads = 1;
    while (threads < reduction_size && threads * 2 <= THREADS_PER_BLOCK) {
        threads *= 2;
    }
    return threads;
}

} // namespace

// Forward pass: one thread computes one output value and accumulates over candidate gates.
template <typename scalar_t>
__global__ void conv_layer_cuda_forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> offsets_ch,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_h,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_w,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> y,
    int stride, int pad, int h_out, int w_out
) {
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y * blockDim.x + threadIdx.x;
    const auto flat_idx = blockIdx.z * blockDim.y + threadIdx.y;
    if (flat_idx >= h_out * w_out || c_out >= w.size(0))
        return;
    const int out_y = flat_idx / w_out;
    const int out_x = flat_idx % w_out;

    const auto h_in = x.size(2);
    const auto w_in = x.size(3);
    const auto num_gates = offsets_ch.size(2);

    auto y_ = static_cast<scalar_t>(0);

    for (int idx_gate = 0; idx_gate < num_gates; ++idx_gate) {
        const auto gate_type = gate_seq[c_out][idx_gate];
        const auto w_ = w[c_out][idx_gate];
        const auto a_ch = offsets_ch[0][c_out][idx_gate];
        const auto a_h = offsets_h[0][c_out][idx_gate];
        const auto a_w = offsets_w[0][c_out][idx_gate];
        const auto b_ch = offsets_ch[1][c_out][idx_gate];
        const auto b_h = offsets_h[1][c_out][idx_gate];
        const auto b_w = offsets_w[1][c_out][idx_gate];

        const int a_x = out_x * stride + a_w;
        const int b_x = out_x * stride + b_w;
        const int a_y = out_y * stride + a_h;
        const int b_y = out_y * stride + b_h;

        scalar_t a_val = 0, b_val = 0;

        if (a_x >= 0 && a_x < w_in && a_y >= 0 && a_y < h_in)
            a_val = x[n][a_ch][a_y][a_x];

        if (b_x >= 0 && b_x < w_in && b_y >= 0 && b_y < h_in)
            b_val = x[n][b_ch][b_y][b_x];

        y_ += bin_op<scalar_t>(a_val, b_val, gate_type) * w_;
    }
    y[n][c_out][out_y][out_x] = y_;
}

torch::Tensor conv_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    int stride, int pad, int ks
) {
    CHECK_INPUT(x);
    CHECK_INPUT(offsets_ch);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_w);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);

    const auto N = x.size(0);
    const auto h_in = x.size(2);
    const auto w_in = x.size(3);
    const auto c_out = w.size(0);
    const auto h_out = (h_in + 2 * pad - ks) / stride + 1;
    const auto w_out = (w_in + 2 * pad - ks) / stride + 1;
    const auto num_gates = w.size(1);

    TORCH_CHECK(h_out > 0 && w_out > 0, "Invalid output shape: h_out=", h_out, ", w_out=", w_out);

    auto y = torch::empty({N, c_out, h_out, w_out}, x.options());

    const auto launch = make_conv_forward_launch(N, c_out, h_out * w_out);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_layer_cuda_forward_kernel",
                               (
                                   [&]
                                   {
                                       conv_layer_cuda_forward_kernel<scalar_t>
                                           <<<launch.blocks, launch.threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               offsets_ch.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_h.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_w.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                                               y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               stride, pad, h_out, w_out);
                                   }));
    gpuErrchk(cudaPeekAtLastError());

    return y;
}

// Input-gradient pass: one thread computes one gate-specific output contribution and scatters it to grad_x.
template <typename scalar_t>
__global__ void conv_layer_cuda_backward_x_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> offsets_ch,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_h,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_w,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x,
    int stride, int pad
) {
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y * blockDim.x + threadIdx.x;
    const auto h_out = grad_y.size(2);
    const auto w_out = grad_y.size(3);
    const auto flat_idx = blockIdx.z * blockDim.y + threadIdx.y;
    const auto idx_gate = threadIdx.z;
    if (flat_idx >= h_out * w_out || c_out >= w.size(0))
        return;

    const auto h_in = x.size(2);
    const auto w_in = x.size(3);
    const int out_y = flat_idx / w_out;
    const int out_x = flat_idx % w_out;

    const auto a_ch = offsets_ch[0][c_out][idx_gate];
    const auto a_h = offsets_h[0][c_out][idx_gate];
    const auto a_w = offsets_w[0][c_out][idx_gate];
    const auto b_ch = offsets_ch[1][c_out][idx_gate];
    const auto b_h = offsets_h[1][c_out][idx_gate];
    const auto b_w = offsets_w[1][c_out][idx_gate];

    const scalar_t w_ = w[c_out][idx_gate];
    const uint8_t gate_type = gate_seq[c_out][idx_gate];

    const int a_x = out_x * stride + a_w;
    const int a_y = out_y * stride + a_h;
    const int b_x = out_x * stride + b_w;
    const int b_y = out_y * stride + b_h;

    scalar_t grad_y_ = grad_y[n][c_out][out_y][out_x];
    bool valid_0 = a_x >= 0 && a_x < w_in && a_y >= 0 && a_y < h_in;
    scalar_t a_val = (valid_0) ? x[n][a_ch][a_y][a_x] : 0;
    bool valid_1 = b_x >= 0 && b_x < w_in && b_y >= 0 && b_y < h_in;
    scalar_t b_val = (valid_1) ? x[n][b_ch][b_y][b_x] : 0;

    if (valid_0) {
        const auto dy_dx_0 = w_ * bin_op_grad_a<scalar_t>(b_val, gate_type);
        atomicAdd(&grad_x[n][a_ch][a_y][a_x], dy_dx_0 * grad_y_);
    }
    if (valid_1) {
        const auto dy_dx_1 = w_ * bin_op_grad_b<scalar_t>(a_val, gate_type);
        atomicAdd(&grad_x[n][b_ch][b_y][b_x], dy_dx_1 * grad_y_);
    }
}

torch::Tensor conv_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    int stride, int pad
) {
    CHECK_INPUT(x);
    CHECK_INPUT(offsets_ch);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_w);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);
    CHECK_INPUT(grad_y);
    const auto N = x.size(0);
    const auto c_out = grad_y.size(1);
    const auto h_out = grad_y.size(2);
    const auto w_out = grad_y.size(3);
    const auto num_gates = offsets_ch.size(2);
    TORCH_CHECK(num_gates <= THREADS_PER_BLOCK, "num_gates should not exceed THREADS_PER_BLOCK");

    auto grad_x = torch::zeros_like(x);
    const auto launch = make_conv_backward_x_launch(N, c_out, h_out * w_out, num_gates);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_layer_cuda_backward_x_kernel",
                               (
                                   [&]
                                   {
                                       conv_layer_cuda_backward_x_kernel<scalar_t>
                                           <<<launch.blocks, launch.threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               offsets_ch.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_h.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_w.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                                               grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               stride, pad);
                                   }));
    gpuErrchk(cudaPeekAtLastError());
    return grad_x;
}

// Weight-gradient pass: one block reduces over all batch/spatial applications for one (c_out, gate) pair.
template <typename scalar_t>
__global__ void conv_layer_cuda_backward_w_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> offsets_ch,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_h,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_w,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_w,
    int stride, int h_out, int w_out
) {
    const auto c_out = blockIdx.x;
    const auto k = blockIdx.y;
    const auto tid = threadIdx.x;

    const auto N = x.size(0);
    const auto h_in = x.size(2);
    const auto w_in = x.size(3);

    const auto num_gates = offsets_ch.size(2);
    if (k >= num_gates) return;

    const auto gate_type = gate_seq[c_out][k];

    const auto a_ch = offsets_ch[0][c_out][k];
    const auto b_ch = offsets_ch[1][c_out][k];

    const auto a_h = offsets_h[0][c_out][k];
    const auto a_w = offsets_w[0][c_out][k];
    const auto b_h = offsets_h[1][c_out][k];
    const auto b_w = offsets_w[1][c_out][k];

    // 1D reduction over all applications of this gate across batch and spatial positions.
    const int total = N * h_out * w_out;

    float local_sum = 0.0f;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        const int n = idx / (h_out * w_out);
        const int rem = idx - n * (h_out * w_out);
        const int out_y = rem / w_out;
        const int out_x = rem - out_y * w_out;

        const int a_x = out_x * stride + a_w;
        const int a_y = out_y * stride + a_h;
        const int b_x = out_x * stride + b_w;
        const int b_y = out_y * stride + b_h;

        scalar_t a_val = static_cast<scalar_t>(0);
        scalar_t b_val = static_cast<scalar_t>(0);

        if (a_x >= 0 && a_x < w_in && a_y >= 0 && a_y < h_in)
            a_val = x[n][a_ch][a_y][a_x];

        if (b_x >= 0 && b_x < w_in && b_y >= 0 && b_y < h_in)
            b_val = x[n][b_ch][b_y][b_x];

        const scalar_t gate_out = bin_op<scalar_t>(a_val, b_val, gate_type);
        local_sum += static_cast<float>(gate_out * grad_y[n][c_out][out_y][out_x]);
    }

    const auto lane = tid & 31;
    const auto warp_id = tid >> 5;
    const auto warp_count = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    extern __shared__ unsigned char smem[];
    float* warp_sums = reinterpret_cast<float*>(smem);
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
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
        grad_w[c_out][k] = static_cast<scalar_t>(block_sum);
    }
}


torch::Tensor conv_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    int stride
) {
    CHECK_INPUT(x);
    CHECK_INPUT(offsets_ch);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_w);
    CHECK_INPUT(gate_seq);
    CHECK_INPUT(grad_y);

    const auto c_out = grad_y.size(1);
    const auto h_out = grad_y.size(2);
    const auto w_out = grad_y.size(3);
    const auto num_gates = offsets_ch.size(2);

    auto grad_w = torch::empty({c_out, num_gates}, x.options());

    dim3 blocks((unsigned)c_out, (unsigned)num_gates);
    const int total = x.size(0) * h_out * w_out;
    const int threads = choose_conv_backward_w_threads(total);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_layer_cuda_backward_w_kernel", ([&] {
        const size_t shmem_bytes = ((threads + 31) / 32) * sizeof(float);

        conv_layer_cuda_backward_w_kernel<scalar_t>
            <<<blocks, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                offsets_ch.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                offsets_h.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                offsets_w.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                stride, h_out, w_out
            );
    }));

    gpuErrchk(cudaPeekAtLastError());
    return grad_w;
}


template <typename scalar_t>
__global__ void conv_layer_cuda_forward_eval_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> offsets_ch,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_h,
    const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> offsets_w,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> gate_seq,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> y,
    int stride, int pad, int h_out, int w_out
) {
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y;

    const auto h_in = x.size(2);
    const auto w_in = x.size(3);

    const auto idx_gate = w[c_out];
    const auto gate_type = gate_seq[c_out][idx_gate];
    const auto a_ch = offsets_ch[0][c_out][idx_gate];
    const auto a_h = offsets_h[0][c_out][idx_gate];
    const auto a_w = offsets_w[0][c_out][idx_gate];
    const auto b_ch = offsets_ch[1][c_out][idx_gate];
    const auto b_h = offsets_h[1][c_out][idx_gate];
    const auto b_w = offsets_w[1][c_out][idx_gate];
    for (int out_x = threadIdx.x; out_x < w_out; out_x += blockDim.x) {
        const int a_x = out_x * stride + a_w;
        const int b_x = out_x * stride + b_w;
        for (int out_y = threadIdx.y; out_y < h_out; out_y += blockDim.y){
            const int a_y = out_y * stride + a_h;
            const int b_y = out_y * stride + b_h;

            scalar_t a_val = 0, b_val = 0;

            if (a_x >= 0 && a_x < w_in && a_y >= 0 && a_y < h_in)
                a_val = x[n][a_ch][a_y][a_x];

            if (b_x >= 0 && b_x < w_in && b_y >= 0 && b_y < h_in)
                b_val = x[n][b_ch][b_y][b_x];

            y[n][c_out][out_y][out_x] = bin_op<scalar_t>(a_val, b_val, gate_type);
        }
    }
}

// Eval pass: use the discretized winning gate only; no softmax-weighted gate loop.
torch::Tensor
conv_layer_cuda_forward_eval(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    int stride, int pad, int ks
) {
    CHECK_INPUT(x);
    CHECK_INPUT(offsets_ch);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_w);
    CHECK_INPUT(w);
    CHECK_INPUT(gate_seq);

    const auto N = x.size(0);
    const auto h_in = x.size(2);
    const auto w_in = x.size(3);
    const auto c_out = w.size(0);
    const auto h_out = (h_in + 2 * pad - ks) / stride + 1;
    const auto w_out = (w_in + 2 * pad - ks) / stride + 1;

    TORCH_CHECK(h_out > 0 && w_out > 0, "Invalid output shape: h_out=", h_out, ", w_out=", w_out);

    auto y = torch::empty({N, c_out, h_out, w_out}, x.options());

    dim3 blocks_per_grid(N, c_out);
    int threads_x = min(static_cast<int64_t>(w_out), static_cast<int64_t>(32));
    int threads_y = min(static_cast<int64_t>(h_out), static_cast<int64_t>(32));
    dim3 threads_per_block(threads_x, threads_y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_layer_cuda_forward_eval_kernel",
                               (
                                   [&]
                                   {
                                       conv_layer_cuda_forward_eval_kernel<scalar_t>
                                           <<<blocks_per_grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               w.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
                                               offsets_ch.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_h.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               offsets_w.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
                                               gate_seq.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                                               y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               stride, pad, h_out, w_out);
                                   }));
    gpuErrchk(cudaPeekAtLastError());

    return y;
}
