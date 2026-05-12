# CompactLogic

Official implementation for **Learning Compact Boolean Networks**.

- Paper: [arXiv:2602.05830](https://arxiv.org/abs/2602.05830)
- Repository: <https://github.com/eth-sri/CompactLogic>
- Checkpoint archive: [Mega](https://mega.nz/file/O05ykbiT#8e1QHADmP-ECqL0DK53_MdAmR6SyItmrtZ3RXM6TAzE)
- License: [MIT](LICENSE)

CompactLogic is a PyTorch package for learning compact Boolean networks with differentiable training, discretizing them into logic-gate networks, and compiling trained models into FPGA-oriented circuit artifacts.

## Highlights

- **Differentiable logic-gate layers** for training Boolean-network models with gradient-based optimization.
- **Compact gate selection** through candidate-gate learning and entropy-based training utilities.
- **Convolution-style logic layers** for image-like inputs.
- **Checkpoint-to-circuit compiler** for extracting discrete Boolean circuits and emitting Verilog.
- **Reporting tools** for semantic consistency checks, gate-count summaries, artifact-size summaries, and FPGA-oriented throughput estimates.

## Repository layout

```text
compactlogic/     # Python package and CUDA extension sources
experiments/      # Training entry point, dataset helpers, and experiment configs
compiler/         # Discrete circuit extraction, pruning, scheduling, and Verilog generation
simulation/       # Compiler/reporting scripts and consistency checks
third_party/      # Reference material, including ConvLogic-related material
```

## Installation

This repository is intended to be installed from source.

```bash
git clone https://github.com/eth-sri/CompactLogic.git
cd CompactLogic
pip install -e .
```

The core package builds custom CUDA extensions and requires a CUDA-enabled PyTorch installation and a matching NVIDIA CUDA toolkit.

For experiment and reporting utilities, install the additional requirements:

```bash
pip install -r experiments/requirements.txt
pip install -r simulation/requirements.txt
```

If you run into setup issues, see [INSTALLATION_SUPPORT.md](INSTALLATION_SUPPORT.md).

## Quick start: train a model

Training is configured with YAML files under `experiments/configs/`.

```bash
python experiments/experiment_script.py --config experiments/configs/regular_mnist_small.yaml
```

Other example configs include compact regular models, convolution-style models, MNIST, thresholded CIFAR-10, ECG, and tabular/segmentation settings.

## Quick start: compile and report a checkpoint

To generate compiler statistics and FPGA-oriented estimates for a trained CompactLogic checkpoint:

```bash
python -m simulation.paper_stats \
  --checkpoint <path_to_checkpoint.pt>
```

Example:

```bash
python -m simulation.paper_stats \
  --checkpoint results/example_run/ckpt_best_resample.pt
```

If the checkpoint directory contains a sibling `meta_data.json`, the script will usually infer the required metadata automatically.

By default, the report is written next to the checkpoint and includes:

- `compiler_paper_stats.json`
- `compiler_paper_stats.md`
- a compiled Verilog model such as `compactlogic_*_balanced_clocked.v`

The report summarizes:

- semantic agreement between the compiled circuit and the trained model,
- raw, pruned, and compiled Boolean gate counts,
- compiled Verilog size,
- heuristic FPGA sample time and throughput estimates.

See [simulation/README.md](simulation/README.md) for the full reporting guide and available options.

## Python API sketch

```python
from compactlogic import LogicLayer, ConvLayer, GroupSum

logic = LogicLayer(
    in_dim=784,
    out_dim=12_000,
    num_gates=16,
)

conv = ConvLayer(
    in_shape=(1, 28, 28),
    c_out=128,
    ks=3,
    stride=2,
    padding=1,
    num_gates=16,
)

head = GroupSum(k=10, tau=10.0)
```

The main package exports:

- `LogicLayer`: learns input connections and Boolean gate choices for dense logic layers.
- `ConvLayer`: applies logic-gate computation over local receptive fields.
- `GroupSum`: aggregates final logic activations into class logits.
- `EntropyRowResampler`: refreshes unstable or dominated gate-selection rows during training.
- `SequentialEntropyFreezer`: progressively discretizes soft gate mixtures into argmax gates.

## Compiler and simulation workflow

The compiler path extracts a discrete circuit from a trained checkpoint, optionally prunes redundant structure, schedules the Boolean network, and emits Verilog. The simulation/reporting path then checks the compiled semantics against the model and reports circuit-level statistics.

Current paper-level reporting support includes:

- regular MNIST checkpoints,
- convolution-style MNIST checkpoints,
- regular thresholded CIFAR-10 checkpoints,
- convolution-style thresholded CIFAR-10 checkpoints.

The default FPGA timing numbers are heuristic CPU-only estimates for early comparison and reporting. They are not vendor timing-closure results.

## Citation

If you use CompactLogic, please cite:

```bibtex
@misc{wang2026compactlogic,
  title        = {Learning Compact Boolean Networks},
  author       = {Wang, Shengpu and Mao, Yuhao and Zhang, Yani and Vechev, Martin},
  year         = {2026},
  eprint       = {2602.05830},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2602.05830}
}
```

Please check the arXiv page for the most up-to-date citation metadata.
