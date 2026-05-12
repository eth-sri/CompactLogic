#include <torch/extension.h>
#include <vector>

// Thin pybind wrapper for convolutional CUDA kernels.
// Keep signatures here in sync with convlogic_kernel.cu and with ConvLayerCudaFunction in Python.

torch::Tensor conv_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    int stride, int pad, int ks
);

torch::Tensor conv_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    int stride, int pad
);

torch::Tensor conv_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor gate_seq,
    torch::Tensor grad_y,
    int stride
);

torch::Tensor
conv_layer_cuda_forward_eval(
    torch::Tensor x,
    torch::Tensor offsets_ch,
    torch::Tensor offsets_h,
    torch::Tensor offsets_w,
    torch::Tensor w,
    torch::Tensor gate_seq,
    int stride, int pad, int ks
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](
            torch::Tensor x,
            torch::Tensor offsets_ch,
            torch::Tensor offsets_h,
            torch::Tensor offsets_w,
            torch::Tensor w,
            torch::Tensor gate_seq,
            int stride, int pad, int ks
        ) {
            return conv_layer_cuda_forward(x, offsets_ch, offsets_h, offsets_w, w, gate_seq, stride, pad, ks);
        },
        "conv logic forward (CUDA)");

    m.def(
        "backward_x",
        [](
            torch::Tensor x,
            torch::Tensor offsets_ch,
            torch::Tensor offsets_h,
            torch::Tensor offsets_w,
            torch::Tensor w,
            torch::Tensor gate_seq,
            torch::Tensor grad_y,
            int stride, int pad
        ) {
            return conv_layer_cuda_backward_x(x, offsets_ch, offsets_h, offsets_w, w, gate_seq, grad_y, stride, pad);
        },
        "conv logic backward x (CUDA)");

    m.def(
        "backward_w",
        [](
            torch::Tensor x,
            torch::Tensor offsets_ch,
            torch::Tensor offsets_h,
            torch::Tensor offsets_w,
            torch::Tensor gate_seq,
            torch::Tensor grad_y,
            int stride
        ) {
            return conv_layer_cuda_backward_w(x, offsets_ch, offsets_h, offsets_w, gate_seq, grad_y, stride);
        },
        "conv logic backward w (CUDA)");

    m.def(
        "eval",
        [](
            torch::Tensor x,
            torch::Tensor offsets_ch,
            torch::Tensor offsets_h,
            torch::Tensor offsets_w,
            torch::Tensor w,
            torch::Tensor gate_seq,
            int stride, int pad, int ks
        ) {
            return conv_layer_cuda_forward_eval(x, offsets_ch, offsets_h, offsets_w, w, gate_seq, stride, pad, ks);
        },
        "conv logic forward (CUDA)");
}
