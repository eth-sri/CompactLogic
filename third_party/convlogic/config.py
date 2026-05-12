"""Checkpoint-configuration inference for the minimal vendored ConvLogic adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ConvLogicConfig:
    variant_name: str
    dataset_name: str
    k: int
    tau: float
    input_channels: int
    input_size: int
    num_classes: int = 10
    num_chn: int = 2
    conv_layers: int = 3


_MNIST_TAU_BY_K = {
    16: 6.5,
    64: 28.0,
    1024: 35.0,
}

_CIFAR_CONFIG_BY_K = {
    32: ('cifar10-3', 20.0),
    256: ('cifar10-3', 40.0),
    512: ('cifar10-31', 280.0),
    1024: ('cifar10-31', 340.0),
}


def _infer_threshold_levels(dataset_name: str) -> int:
    if '-' not in dataset_name:
        return 1
    try:
        return int(dataset_name.split('-')[-1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f'Unsupported ConvLogic dataset name: {dataset_name!r}') from exc


def infer_config_from_checkpoint(checkpoint_path: str | Path) -> ConvLogicConfig:
    """Infer the supported runtime/extraction config directly from a ConvLogic checkpoint."""
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    variant_name = path.parent.name.lower()
    family = 'mnist' if 'mnist' in variant_name else 'cifar10' if 'cifar' in variant_name else None
    if family is None:
        raise ValueError(f'Unable to infer dataset family from checkpoint path: {path}')

    conv_prefixes = sorted({key.rsplit('.', 1)[0] for key in state_dict if key.endswith('.selection')})
    if not conv_prefixes:
        raise ValueError(f'No ConvLogic convolutional layers found in checkpoint: {path}')
    k = int(state_dict[f'{conv_prefixes[0]}.selection'].shape[0])

    if family == 'mnist':
        if k not in _MNIST_TAU_BY_K:
            raise ValueError(f'Unsupported ConvLogic MNIST checkpoint scale k={k} for {path}')
        tau = _MNIST_TAU_BY_K[k]
        dataset_name = 'mnist'
        input_channels = 1
        input_size = 28
        num_classes = 10
    else:
        if k not in _CIFAR_CONFIG_BY_K:
            raise ValueError(f'Unsupported ConvLogic CIFAR checkpoint scale k={k} for {path}')
        dataset_name, tau = _CIFAR_CONFIG_BY_K[k]
        threshold_levels = _infer_threshold_levels(dataset_name)
        input_channels = 3 * threshold_levels
        input_size = 32
        num_classes = 10

    return ConvLogicConfig(
        variant_name=variant_name,
        dataset_name=dataset_name,
        k=k,
        tau=float(tau),
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        num_chn=2,
        conv_layers=len(conv_prefixes),
    )
