# CUDA kernel developer guide

This directory contains the custom CUDA implementation for the two train-time kernel families used by `compactlogic`:

- `difflogic_*`: dense `LogicLayer`
- `convlogic_*`: convolutional `ConvLayer`

The goal of this note is to help future developers understand:

1. where the important entry points are,
2. what tensor layouts each kernel expects,
3. which kernels are currently performance-critical,
4. how to change the code safely.

---

## 1. File map

### `utils.cuh`
Shared CUDA helpers:

- tensor checks (`CHECK_INPUT`, etc.)
- CUDA error checking
- `ceil_div`
- atomic helpers
- Boolean operator implementation:
  - `bin_op`
  - `bin_op_grad_a`
  - `bin_op_grad_b`

If the Boolean algebra changes, update it **here first**.

### `difflogic_kernel.cu`
CUDA kernels for the dense `LogicLayer`.

Main entry points:

- `logic_layer_cuda_forward`
- `logic_layer_cuda_backward_x`
- `logic_layer_cuda_backward_w`
- `logic_layer_cuda_eval`

### `convlogic_kernel.cu`
CUDA kernels for the convolutional `ConvLayer`.

Main entry points:

- `conv_layer_cuda_forward`
- `conv_layer_cuda_backward_x`
- `conv_layer_cuda_backward_w`
- `conv_layer_cuda_forward_eval`

### `difflogic.cpp` / `convlogic.cpp`
PyBind wrapper files. These are intentionally thin. They should only expose the CUDA entry points to Python and keep signatures synchronized with:

- the corresponding `*_kernel.cu` file, and
- the Python autograd wrappers in `compactlogic.py`.

---

## 2. Tensor layout conventions

### Dense `LogicLayer`

- `x`: `[in_dim, batch]`
- `y`: `[out_dim, batch]`
- `a`, `b`: `[out_dim, num_gates]`
- `w`: `[out_dim, num_gates]`
- `gate_seq`: `[out_dim, num_gates]`

Important note:
- Python transposes the logical input from `[batch, in_dim]` to `[in_dim, batch]` before dispatching into CUDA.

### Convolutional `ConvLayer`

- `x`: `[batch, c_in, h_in, w_in]`
- `y`: `[batch, c_out, h_out, w_out]`
- `offsets_ch`, `offsets_h`, `offsets_w`: `[2, c_out, num_gates]`
- `w`: `[c_out, num_gates]`
- `gate_seq`: `[c_out, num_gates]`

The leading dimension of the offsets selects the first or second operand of the Boolean gate.

---

## 3. Current kernel structure

## 3.1 Dense kernels (`difflogic_kernel.cu`)

### Forward
One thread computes one `(out neuron, batch item)` output and loops over the gate candidates.

### `backward_x`
This uses the **inverse-connectivity** formulation:

- `given_x_indices_of_y_start`
- `given_x_indices_of_y`
- `given_x_indices_of_gate`

Each thread owns one `(input neuron, batch item)` gradient and gathers all downstream uses.

This path replaced the older atomic-add implementation for the regular layer because it was a measurable win.

### `backward_w`
One block owns one `(out neuron, gate)` pair and reduces over the batch dimension.

This kernel currently uses:

- dynamic power-of-two thread count up to `BACKWARD_W_MAX_THREADS`
- warp-level reduction
- one shared float per warp

---

## 3.2 Convolutional kernels (`convlogic_kernel.cu`)

### Forward
One thread computes one `(batch, c_out, out_y, out_x)` output and loops over all candidate gates.

### `backward_x`
This is still the **atomic scatter** version.

One thread owns one `(batch, c_out, out_y, out_x, gate)` contribution and atomically accumulates into `grad_x`.

Why it still exists:

- a gather-style inverse reformulation was tested,
- it was correct,
- but it regressed end-to-end speed in the current workload.

So future developers should treat the conv `backward_x` formulation carefully: the most obvious “regular-layer style” rewrite already failed experimentally.

### `backward_w`
One block owns one `(c_out, gate)` pair and reduces over:

- batch
- output height
- output width

The current version uses:

- dynamic power-of-two block size up to `THREADS_PER_BLOCK`
- warp-level reduction
- one shared float per warp

This improved readability and gave a very small speedup, but it was not a major optimization.

### Eval
The eval kernel runs the discretized winner gate only, with no softmax-weighted loop over all gates.

---

## 4. What has already been tried

These results matter because they prevent future developers from repeating expensive dead ends.

### Dense layer
Successful:

- removed forced synchronizations in hot paths
- switched regular-layer `backward_x` away from atomic-add to inverse connectivity
- minor cleanup of `backward_w`

Outcome:

- meaningful but limited end-to-end gain
- remaining slowdown appears mostly tied to the richer relaxation itself

### Convolutional layer
Tried and rejected:

- gather-style non-atomic `backward_x`

Outcome:

- numerically correct
- slower end-to-end

Tried and kept:

- local `backward_w` reduction cleanup

Outcome:

- very small speedup only

Current conclusion:

- no clearly convincing large kernel-level conv optimization remains without a larger structural redesign

---

## 5. Where the bottlenecks currently are

As of the latest measurements:

- dense models: backward dominates, especially `grad_w`
- conv models: backward dominates strongly (~80% of total train-step time)

For conv specifically:

- `backward_x` remains atomic and expensive
- `backward_w` is also large in absolute time
- but obvious local tweaks have not moved the needle much

---

## 6. Safe workflow for future changes

Before changing kernels:

1. update or add tests in `tests_opt/`
2. run correctness tests
3. run the relevant benchmark
4. only then decide whether to keep the change

Recommended workflow:

1. activate the project environment,
2. apply whatever local GPU / CPU pinning policy is appropriate for your machine,
3. then run the relevant tests and benchmarks.

Typical commands:

```bash
python -m unittest discover -s tests_opt -v
python tests_opt/bench_conv_time_breakdown.py --variant conv_cifar_S --batch-size 128 --warmup 2 --steps 5 --repeats 3
python tests_opt/bench_conv_models.py --variant conv_cifar_S --batch-size 128 --warmup 2 --steps 5 --repeats 3
```

For dense kernels, use the corresponding dense benchmarks under `tests_opt/`.

---

## 7. Practical coding guidance

- Prefer **shared helpers in `utils.cuh`** over duplicating Boolean-op logic.
- Keep pybind wrappers thin.
- Document the thread/block ownership of every kernel.
- If a launch heuristic is nontrivial, hide it behind a small helper function.
- Avoid “clever” rewrites unless benchmarks justify them.
- Preserve mathematical equivalence unless an explicit algorithmic change is intended.

---

## 8. If you want to optimize further

The next meaningful advances are likely to require **structural reformulation**, not micro-tuning.

Examples of higher-risk directions:

- a new exact mathematical reorganization of the relaxed computation,
- a more global work decomposition for conv backward,
- or a formulation that changes which intermediate quantities are shared and reused.

Those should be treated as research changes, not routine kernel cleanup.
