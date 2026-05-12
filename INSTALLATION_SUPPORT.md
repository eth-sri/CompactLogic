# Installation Support

`compactlogic` is a Python 3.6+ and PyTorch 1.9.0+ based library.
The library can be installed with: (TODO)
```shell
pip install compactlogic
```
> ⚠️ Note that `difflogic` requires CUDA, the CUDA Toolkit (for compilation), and `torch>=1.9.0` (matching the CUDA version).

**It is very important that the installed version of PyTorch was compiled with a CUDA version that is compatible with 
the CUDA version of the locally installed CUDA Toolkit.**

You can check your CUDA version by running `nvidia-smi`.

You can install PyTorch and torchvision of a specific version, e.g., via 

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html  # CUDA version 11.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html  # CUDA version 11.3
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html  # CUDA version 11.7
```

If you get the following error message:

```
Failed to build difflogic
 
...

RuntimeError:
    The detected CUDA version (11.2) mismatches the version that was used to compile
    PyTorch (11.7). Please make sure to use the same CUDA versions.
```

You need to make sure that the versions match, typically by installing a different PyTorch version.
Note that there are some versions of PyTorch that have been compiled with CUDA versions different from the advertised
versions, so in case it should match but doesn't, a quick fix can be to try some other (e.g., older) PyTorch versions.

---

`compactlogic` has been tested with PyTorch versions 1.13 and 2.5.

---

For the experiments, please make sure all dependencies in `experiments/requirements.txt` are installed in the Python 
environment.

