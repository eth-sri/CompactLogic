# Compact Boolean Networks

This repository contains the official implementation for the paper **Learning Compact Boolean Networks**.  
Paper: [arXiv:2602.05830](https://arxiv.org/abs/2602.05830)

`compactlogic` is a Python package for building, training, and deploying **logic gate neural networks** with
gradient-based optimization in PyTorch.

This codebase is based on **Deep Differentiable Logic Gate Networks** (NeurIPS 2022).  
Paper: [arXiv:2210.08277](https://arxiv.org/abs/2210.08277) · 
Reference repo: [difflogic](https://github.com/Felix-Petersen/difflogic/)

## If you only want a compiler report for a trained checkpoint

You do **not** need to understand the compiler internals.

For a CompactLogic checkpoint, run:

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt>
```

If the checkpoint has a sibling `meta_data.json` file, that is usually enough.

Example:

```bash
python -m simulation.paper_stats \
  --checkpoint results/example_run/ckpt_best_resample.pt
```

For a ConvLogic checkpoint, run:

```bash
python -m third_party.convlogic.paper_stats \
  --checkpoint <path_to_convlogic_checkpoint.ckpt>
```

Example:

```bash
python -m third_party.convlogic.paper_stats \
  --checkpoint checkpoints/convlogic/conv_mnist_S/best.ckpt
```

Pretrained checkpoint data for the convolutional models can be found here:
[Mega link](https://mega.nz/file/O05ykbiT#8e1QHADmP-ECqL0DK53_MdAmR6SyItmrtZ3RXM6TAzE). 
The shared files contain the `.pt` files for the trained models and their FPGA simulation results.

By default, this writes the following files into the **checkpoint directory**:

- `compiler_paper_stats.json`
- `compiler_paper_stats.md`
- the compiled Verilog model, e.g. `compactlogic_*_balanced_clocked.v`

For ConvLogic, the filenames are:

- `convlogic_compiler_paper_stats.json`
- `convlogic_compiler_paper_stats.md`
- the compiled Verilog model, e.g. `convlogic_*_balanced_clocked.v`

The report includes:
- full-test-set match between compiled circuit semantics and the original model,
- the explicit report reference mode (`original_compactlogic_model` or `extracted_discrete_circuit`),
- raw vs pruned logical gate counts,
- compiled artifact size,
- estimated FPGA sample time and throughput.

Current paper-level report support:
- **regular** MNIST checkpoints
- **conv_spm** MNIST checkpoints
- **regular** thresholded CIFAR-10 checkpoints
- **conv_spm** thresholded CIFAR-10 checkpoints

See [simulation/README.md](simulation/README.md) for the simplest usage guide.

## Requirements

### Core package
- Python 3.6+
- PyTorch (CUDA-enabled)
- NVIDIA CUDA toolkit for building the custom CUDA extensions

> ⚠️ CUDA is required for the training/inference package itself. Ensure your installed `torch` version matches your CUDA runtime/toolkit.

### Compiler/report path
For the current compiler/report workflow we recommend a conda environment with:
- Python 3.11
- `torch`
- `torchvision`
- `pyyaml`

If you already use a global/project conda environment for this repository, install the reporting extras into that same environment:

```bash
conda activate compactlogic
pip install -r experiments/requirements.txt
pip install -r simulation/requirements.txt
```

You do **not** need to recreate the environment if it already exists.

## Installation

This repository is maintained as an installable Python package for local/development use, but is not distributed via PyPI.

### Install from source

```bash
git clone https://github.com/ShengpuWang1/compact-logic.git
cd compact-logic
pip install -e .
```

If you run into environment/setup issues, see [INSTALLATION_SUPPORT.md](INSTALLATION_SUPPORT.md).

### Optional: install experiment dependencies

```bash
pip install -r experiments/requirements.txt
```

## Quick start for training

The training entry point is:

```bash
python experiments/experiment_script.py --config <path_to_yaml>
```

Example:

```bash
python experiments/experiment_script.py --config experiments/configs/regular_mnist_small.yaml
```

## Core modules

### `LogicLayer`

`LogicLayer` learns:
- which binary logic gate to apply per output neuron,
- and which input pair to connect to each candidate gate.

Important parameters:
- `in_dim`, `out_dim`: input/output widths
- `num_gates`: number of candidate gates per output neuron

Example:

```python
from compactlogic import LogicLayer

layer = LogicLayer(
    in_dim=784,
    out_dim=12_000,
    num_gates=16,
)
```

### `ConvLayer`

`ConvLayer` extends logic-gate computation to image-like tensors by applying logic operations over local neighborhoods
(convolution-style receptive fields).

Example:

```python
from compactlogic import ConvLayer

conv = ConvLayer(
    in_shape=(1, 28, 28),
    c_out=128,
    ks=3,
    stride=2,
    padding=1,
    num_gates=16,
)
```

### `GroupSum`

`GroupSum(k, tau)` aggregates final logic activations into class logits, where:
- `k`: number of output classes
- `tau`: temperature parameter used in the reduction

## Training utilities

### `EntropyRowResampler`

`EntropyRowResampler` monitors the gate-selection distribution of each neuron and selectively refreshes neurons that are
stably diverged or dominated.

In YAML configs, this utility is controlled by:
- `resample`
- `patience`
- `max_iters`

### `SequentialEntropyFreezer`

`SequentialEntropyFreezer` progressively converts layers from soft/differentiable gate mixtures to
**discrete argmax gates** during training.

In YAML configs, this utility is controlled by:
- `freeze`
- `min_iters`
- `freeze_num_layers`

## Project structure

```text
compactlogic/                 # Package and CUDA kernels
experiments/                  # Training scripts and configs
compiler/                     # Checkpoint -> circuit/Verilog compiler
simulation/                   # Reports, consistency checks, FPGA-oriented estimation
tests/                        # Fast default tests + optional slow integration tests
```

## Citation

If you use this repository, please cite:

- **Learning Compact Boolean Networks** (arXiv:2602.05830)
