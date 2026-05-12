# Compiler

This folder contains the code that turns a trained `compactlogic` checkpoint into a fixed hardware-style model.
Its supported scope is the minimal pipeline needed to extract a circuit, prune/reindex it, emit the preferred Verilog backend, and support the report generator.

## Most users only need one command

If your goal is:
> “I trained a checkpoint and now I want compiler-level stats and an FPGA-oriented report,”

run:

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt>
```

Example:

```bash
python -m simulation.paper_stats \
  --checkpoint results/2026-04-04/210049/ckpt_best_resample.pt
```

That command will:
1. load the checkpoint,
2. compile the discretized Boolean network,
3. compare the compiled circuit semantics to the original model on the full test set,
   and record the report reference mode explicitly,
4. write a paper-style report into the checkpoint directory,
5. emit the preferred compiled Verilog model.

You do **not** need to call the lower-level compiler modules directly unless you want to debug or develop the compiler itself.

## What the compiler currently supports

Current scope:
- **regular** MNIST models
- **conv_spm** MNIST models
- **MNIST** full-test-set paper report
- **regular** thresholded CIFAR-10 models
- **conv_spm** thresholded CIFAR-10 models
- **thresholded CIFAR-10** full-test-set paper report
- preferred backend:
  - naive-pruned
  - reindexed
  - balanced output reduction
  - one-cycle clocked wrapper

## Environment

Recommended conda environment for the current compiler/report flow:
- Python 3.11
- `torch`
- `torchvision`
- `pyyaml`

Use the same project environment as the rest of the repository; no separate compiler-only environment is required.

## Main compiler stages

The compiler internally does the following:

1. **extract** a discrete circuit from the checkpoint
2. **naively prune** unused or trivial logic
3. **reindex** the remaining live nodes
4. **emit Verilog**
5. optionally wrap the circuit in a **clocked** interface

For most users, these stages are already packaged into `simulation.paper_stats`.

## What the preferred final backend means

The preferred emitted model is:
- **naive-pruned**: dead or trivial logic removed,
- **reindexed**: live node ids compacted,
- **balanced**: class-count reduction implemented as a balanced tree,
- **clocked**: wrapped for one-sample-per-cycle hardware-style timing.

This is the backend used by the paper-level report.
