#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

// Thin pybind wrapper for dense LogicLayer CUDA kernels.
// Keep these bindings aligned with difflogic_kernel.cu and the Python autograd wrapper.

namespace py = pybind11;

torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq
);
torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor gate_seq,
    torch::Tensor grad_y
);
torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y,
    torch::Tensor given_x_types_of_gate
);
torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq
);
std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
);
torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
);
torch::Tensor logic_layer_cuda_backward_x_atomicAdd(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor gate_seq) {
            return logic_layer_cuda_forward(x, a, b, w, gate_seq);
        },
        "logic layer forward (CUDA)");
    m.def(
        "backward_w", [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor gate_seq, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_w(x, a, b, gate_seq, grad_y);
        },
        "logic layer backward w (CUDA)");
    m.def(
        "backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor gate_seq, torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y, torch::Tensor given_x_types_of_gate) {
            return logic_layer_cuda_backward_x(x, a, b, w, gate_seq, grad_y, given_x_indices_of_y_start, given_x_indices_of_y, given_x_types_of_gate);
        },
        "logic layer backward x (CUDA)");
    m.def(
        "eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor gate_seq) {
            return logic_layer_cuda_eval(x, a, b, w, gate_seq);
        },
        "logic layer eval (CUDA)");
    m.def(
        "tensor_packbits_cuda",
        [](torch::Tensor t, const int bit_count) {
            return tensor_packbits_cuda(t, bit_count);
        },
        "ltensor_packbits_cuda (CUDA)");
    m.def(
        "groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k) {
            if (b.size(0) % k != 0) {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) + ") has to be divisible by k (" + std::to_string(k) + ") but it is not");
            }
            return groupbitsum(b, pad_len, k);
        },
        "groupbitsum (CUDA)");
    m.def(
        "backward_x_atomicAdd",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor gate_seq, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_x_atomicAdd(x, a, b, w, gate_seq, grad_y);
        },
        "logic layer backward x (CUDA)");
}
