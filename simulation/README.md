# Simulation and reporting

This folder contains the easiest user-facing entry points for validating compiled models and generating FPGA-oriented reports.
The supported scope is intentionally narrow: generate a compiler report and emit the final Verilog model next to a provided checkpoint.

## Quick start: generate a paper-level report for a checkpoint

If you have a trained checkpoint and want the compiler results without learning the compiler internals, run:

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt>
```

Example:

```bash
python -m simulation.paper_stats \
  --checkpoint results/2026-04-04/210049/ckpt_best_resample.pt
```

If the checkpoint directory contains `meta_data.json`, that is usually enough. The script will infer it automatically.

## Conda environment setup

Set up the reporting environment from the repository root before running any simulation/report commands.
Use the same global/project conda environment that you use for `compactlogic`, then install the simulation extras into that environment.
You do **not** need to recreate the environment if it already exists.

Recommended steps:

```bash
conda activate compactlogic

pip install -e .
pip install -r experiments/requirements.txt
pip install -r simulation/requirements.txt
```

Notes:
- `pip install -e .` installs the local `compactlogic` package from this repository into the global/project environment.
- `experiments/requirements.txt` installs the standard repository Python dependencies into that same environment.
- `simulation/requirements.txt` adds the extra Python packages used by the simulation/reporting flow into that same environment.
- If you do not already have the project environment, create it first, e.g. `conda create -n compactlogic python=3.11 -y`.
- If you need CUDA-backed PyTorch, install the correct `torch`/`torchvision` build for your system before or instead of the generic `pip install -r simulation/requirements.txt` step.

## What this command writes

By default, it writes into the **checkpoint directory**:

- `compiler_paper_stats.json`
- `compiler_paper_stats.md`
- the preferred compiled Verilog model

## What is inside the report

The paper-level report focuses on compiler questions:

### 1. Full-test-set semantic agreement
It checks whether the compiled **circuit semantics** match the original model on the dataset test split.

Reported fields include:
- original accuracy
- compiled accuracy
- exact argmax match rate
- exact class-count match rate
- mismatch counts
- max/mean output differences

### 2. Logical gate counts
It reports:
- raw logical gates
- pruned logical gates
- final compiled Boolean-core logical gates
- logical-gate reduction ratio

### 3. Compiled artifact size
It reports:
- compiled Verilog path
- module name
- Verilog size in MB
- Verilog line count

### 4. FPGA-oriented performance estimate
It reports:
- estimated Fmax
- estimated sample time
- estimated throughput
- latency cycles
- initiation interval
- max Boolean depth

## Current support

The current paper-level reporting path supports:
- **regular** MNIST models
- **conv_spm** MNIST models
- **MNIST**
- **regular** thresholded CIFAR-10 models
- **conv_spm** thresholded CIFAR-10 models
- **thresholded CIFAR-10**

Important:
- the full-test-set comparison is performed with the compiled circuit evaluator,
- the report explicitly records whether the reference was the original compactlogic model or the extracted discrete circuit,
- the full test set is **not** run through external Verilog simulation in the report path.

## Environment

Recommended environment for the reporting flow:
- Python 3.11 in the same global/project conda environment used for the repository
- `torch`
- `torchvision`
- `pyyaml`

Current default FPGA target for heuristic estimates:
- **AMD/Xilinx ZCU104**
- part: **`xczu7ev-ffvc1156-2-e`**

## Common options

### Use a custom metadata path

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt> \
  --metadata <path_to_meta_data.json>
```

### Use a custom config path

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt> \
  --config <path_to_yaml>
```

### Write outputs somewhere else

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt> \
  --out-dir <output_directory>
```

### Change evaluation batch size

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt> \
  --eval-batch-size 512
```

## Important note on timing numbers

The default paper-level timing numbers are currently **heuristic CPU-only FPGA estimates**.
They are useful for early reporting and model comparison, but they are **not** vendor timing-closure numbers.
