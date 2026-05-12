# Installation Support

`compactlogic` is a Python 3.6+ package for PyTorch-based logic-gate networks. The original setup notes target PyTorch 1.9.0+ for CUDA builds. The package builds custom CUDA extensions, so the main installation requirement is a working CUDA-enabled PyTorch environment with a compatible local CUDA Toolkit.

## Requirements

- Python 3.6+
- PyTorch (CUDA-enabled)
- NVIDIA CUDA toolkit for building the custom CUDA extensions

> ⚠️ CUDA is required. Ensure your installed `torch` version matches your CUDA runtime/toolkit.

### Tested Environments

- PyTorch: 1.13, 2.5, 2.11
- CUDA: 12.2, 12.8, 13.0
- NVCC: 11.8, 13.2

## Install from source

This repository is intended to be installed from source:

```shell
git clone https://github.com/eth-sri/CompactLogic.git
cd CompactLogic
pip install -e .
```

For the experiment scripts, install the experiment dependencies in the same Python environment:

```shell
pip install -r experiments/requirements.txt
```

For compiler/reporting utilities, also install:

```shell
pip install -r simulation/requirements.txt
```

## CUDA and PyTorch compatibility

The CUDA version used by your installed PyTorch package must be compatible with the CUDA Toolkit available on the machine that builds the extensions.

You can check the locally available CUDA driver/runtime information with:

```shell
nvidia-smi
```

You can install PyTorch and torchvision builds for a specific CUDA version, for example:

```shell
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130  # CUDA 13.0
```

If you see an error like this while building the CUDA extension:

```text
Failed to build difflogic

...

RuntimeError:
    The detected CUDA version (11.2) mismatches the version that was used to compile
    PyTorch (11.7). Please make sure to use the same CUDA versions.
```

then install a PyTorch build whose CUDA version matches your local CUDA Toolkit more closely, or use a machine/environment with a matching CUDA Toolkit. In some cases, trying a nearby older PyTorch/CUDA build can resolve mismatches even when the advertised versions appear compatible.

[//]: # (## Tested versions)

[//]: # ()
[//]: # (`compactlogic` has been tested with PyTorch 1.13 and 2.5.)
