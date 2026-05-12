"""Native runtime evaluation utilities for supported vendored ConvLogic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from compiler.gates import eval_gate_torch
from compiler.ir import RegularCircuitIR

from .config import ConvLogicConfig, infer_config_from_checkpoint
from .extract import _conv_hyperparams, extract_circuit_from_checkpoint


_MNIST_DATA_ROOT = './data-mnist'
_CIFAR10_DATA_ROOT = './data-cifar10'


@dataclass(frozen=True)
class NativeConvLayer:
    gate_ids: torch.Tensor  # [out_channels, 7]
    channels: torch.Tensor  # [out_channels, 8]
    rows: torch.Tensor      # [out_channels, 8]
    cols: torch.Tensor      # [out_channels, 8]
    kernel: int
    padding: int


@dataclass(frozen=True)
class NativeDenseLayer:
    gate_ids: torch.Tensor  # [out_dim]
    indices_0: torch.Tensor
    indices_1: torch.Tensor


@dataclass(frozen=True)
class NativeConvLogicRuntime:
    config: ConvLogicConfig
    conv_layers: tuple[NativeConvLayer, ...]
    dense_layers: tuple[NativeDenseLayer, ...]
    device: torch.device


@torch.no_grad()
def _apply_per_channel_gates(gate_ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for gate_id in range(16):
        mask = gate_ids == gate_id
        if bool(mask.any()):
            out[:, mask] = eval_gate_torch(gate_id, a[:, mask], b[:, mask])
    return out


@torch.no_grad()
def _gather_selected(
    x_padded: torch.Tensor,
    channels: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    h_out: int,
    w_out: int,
) -> torch.Tensor:
    batch_size = x_padded.shape[0]
    out_channels = channels.shape[0]
    device = x_padded.device
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
    channel_index = channels.view(1, out_channels, 1, 1)
    row_index = torch.arange(h_out, device=device).view(1, 1, h_out, 1) + rows.view(1, out_channels, 1, 1)
    col_index = torch.arange(w_out, device=device).view(1, 1, 1, w_out) + cols.view(1, out_channels, 1, 1)
    return x_padded[batch_index, channel_index, row_index, col_index]


@torch.no_grad()
def _evaluate_native_conv_layer(x: torch.Tensor, layer: NativeConvLayer) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    pad = layer.padding
    kernel = layer.kernel
    h_out = height + 2 * pad - kernel + 1
    w_out = width + 2 * pad - kernel + 1

    x_padded = torch.zeros(
        (batch_size, channels, height + 2 * pad, width + 2 * pad),
        dtype=torch.bool,
        device=x.device,
    )
    x_padded[:, :, pad : pad + height, pad : pad + width] = x

    stage1_outputs: list[torch.Tensor] = []
    for pair in range(4):
        a = _gather_selected(
            x_padded,
            layer.channels[:, 2 * pair],
            layer.rows[:, 2 * pair],
            layer.cols[:, 2 * pair],
            h_out=h_out,
            w_out=w_out,
        )
        b = _gather_selected(
            x_padded,
            layer.channels[:, 2 * pair + 1],
            layer.rows[:, 2 * pair + 1],
            layer.cols[:, 2 * pair + 1],
            h_out=h_out,
            w_out=w_out,
        )
        stage1_outputs.append(_apply_per_channel_gates(layer.gate_ids[:, pair], a, b))

    stage2_left = _apply_per_channel_gates(layer.gate_ids[:, 4], stage1_outputs[0], stage1_outputs[1])
    stage2_right = _apply_per_channel_gates(layer.gate_ids[:, 5], stage1_outputs[2], stage1_outputs[3])
    stage3 = _apply_per_channel_gates(layer.gate_ids[:, 6], stage2_left, stage2_right)

    pooled = (
        stage3[:, :, 0::2, 0::2]
        | stage3[:, :, 0::2, 1::2]
        | stage3[:, :, 1::2, 0::2]
        | stage3[:, :, 1::2, 1::2]
    )
    return pooled


@torch.no_grad()
def _evaluate_native_dense_layer(x: torch.Tensor, layer: NativeDenseLayer) -> torch.Tensor:
    a = x[:, layer.indices_0]
    b = x[:, layer.indices_1]
    return _apply_per_channel_gates(layer.gate_ids, a, b)


@torch.no_grad()
def load_native_runtime(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = 'cpu',
) -> NativeConvLogicRuntime:
    """Reconstruct the native ConvLogic runtime directly from a checkpoint state dict."""
    checkpoint_path = Path(checkpoint_path)
    config = infer_config_from_checkpoint(checkpoint_path)
    target_device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    conv_prefixes = sorted({key.rsplit('.', 1)[0] for key in state_dict if key.endswith('.selection')})
    conv_hparams = _conv_hyperparams(config)
    conv_layers: list[NativeConvLayer] = []
    for prefix, (kernel, padding) in zip(conv_prefixes, conv_hparams, strict=True):
        weights = state_dict[f'{prefix}.weights'].argmax(dim=-1).to(device=target_device, dtype=torch.long)
        selection = state_dict[f'{prefix}.selection'].to(torch.int64)
        conv_layers.append(
            NativeConvLayer(
                gate_ids=weights,
                channels=((selection >> 16) & 0xFFFF).to(device=target_device, dtype=torch.long),
                rows=((selection >> 8) & 0xFF).to(device=target_device, dtype=torch.long),
                cols=(selection & 0xFF).to(device=target_device, dtype=torch.long),
                kernel=kernel,
                padding=padding,
            )
        )

    dense_prefixes = sorted({key.rsplit('.', 1)[0] for key in state_dict if key.endswith('.indices_0')})
    dense_layers: list[NativeDenseLayer] = []
    for prefix in dense_prefixes:
        dense_layers.append(
            NativeDenseLayer(
                gate_ids=state_dict[f'{prefix}.weights'].argmax(dim=-1).to(device=target_device, dtype=torch.long),
                indices_0=state_dict[f'{prefix}.indices_0'].to(device=target_device, dtype=torch.long),
                indices_1=state_dict[f'{prefix}.indices_1'].to(device=target_device, dtype=torch.long),
            )
        )

    return NativeConvLogicRuntime(
        config=config,
        conv_layers=tuple(conv_layers),
        dense_layers=tuple(dense_layers),
        device=target_device,
    )


@torch.no_grad()
def evaluate_native_runtime(runtime: NativeConvLogicRuntime, batch_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the reconstructed native ConvLogic runtime on a batch of booleanized inputs."""
    x = batch_inputs.to(device=runtime.device)
    if x.ndim == 2:
        x = x.reshape(x.shape[0], runtime.config.input_channels, runtime.config.input_size, runtime.config.input_size)
    x = x.to(dtype=torch.bool)

    features = x
    for layer in runtime.conv_layers:
        features = _evaluate_native_conv_layer(features, layer)

    flat = features.reshape(features.shape[0], -1)
    for layer in runtime.dense_layers:
        flat = _evaluate_native_dense_layer(flat, layer)

    class_counts = flat.to(torch.int64).reshape(flat.shape[0], runtime.config.num_classes, -1).sum(dim=-1)
    logits = class_counts.to(torch.float32) / float(runtime.config.tau)
    return {'class_counts': class_counts.cpu(), 'logits': logits.cpu()}


@torch.no_grad()
def evaluate_checkpoint_native(
    checkpoint_path: str | Path,
    batch_inputs: torch.Tensor,
    *,
    device: str | torch.device = 'cpu',
) -> dict[str, torch.Tensor]:
    """Convenience wrapper: load a checkpoint runtime and evaluate one input batch."""
    runtime = load_native_runtime(checkpoint_path, device=device)
    return evaluate_native_runtime(runtime, batch_inputs)


@torch.no_grad()
def evaluate_circuit_torch(
    circuit: RegularCircuitIR,
    batch_inputs: torch.Tensor,
    *,
    device: str | torch.device = 'cpu',
) -> dict[str, torch.Tensor]:
    """Evaluate an extracted ConvLogic circuit through the common torch circuit interpreter."""
    target_device = torch.device(device)
    x = batch_inputs.to(device=target_device, dtype=torch.bool)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] != circuit.input_dim:
        raise ValueError(f'Input shape mismatch: got {tuple(x.shape)}, expected second dim {circuit.input_dim}.')

    total_signals = circuit.total_signal_count
    signal_values = torch.zeros((x.shape[0], total_signals), dtype=torch.bool, device=target_device)
    signal_values[:, : circuit.input_dim] = x

    for layer in circuit.layers:
        out_ids = torch.tensor([node.node_id for node in layer.nodes], dtype=torch.long, device=target_device)
        gate_ids = torch.tensor([node.gate_id for node in layer.nodes], dtype=torch.long, device=target_device)

        input_a_ids = []
        input_a_has_signal = []
        input_a_const = []
        input_b_ids = []
        input_b_has_signal = []
        input_b_const = []
        for node in layer.nodes:
            for source, ids, has_signal, const_bits in [
                (node.input_a, input_a_ids, input_a_has_signal, input_a_const),
                (node.input_b, input_b_ids, input_b_has_signal, input_b_const),
            ]:
                if source.signal_id is not None:
                    ids.append(source.signal_id)
                    has_signal.append(True)
                    const_bits.append(False)
                else:
                    ids.append(0)
                    has_signal.append(False)
                    const_bits.append(bool(source.const_value))

        input_a_ids = torch.tensor(input_a_ids, dtype=torch.long, device=target_device)
        input_a_has_signal = torch.tensor(input_a_has_signal, dtype=torch.bool, device=target_device)
        input_a_const = torch.tensor(input_a_const, dtype=torch.bool, device=target_device)
        input_b_ids = torch.tensor(input_b_ids, dtype=torch.long, device=target_device)
        input_b_has_signal = torch.tensor(input_b_has_signal, dtype=torch.bool, device=target_device)
        input_b_const = torch.tensor(input_b_const, dtype=torch.bool, device=target_device)

        for gate_id in range(16):
            mask = gate_ids == gate_id
            if not bool(mask.any()):
                continue
            a = torch.zeros((x.shape[0], int(mask.sum().item())), dtype=torch.bool, device=target_device)
            b = torch.zeros_like(a)
            if bool(input_a_has_signal[mask].any()):
                a[:, input_a_has_signal[mask]] = signal_values[:, input_a_ids[mask][input_a_has_signal[mask]]]
            if bool(input_b_has_signal[mask].any()):
                b[:, input_b_has_signal[mask]] = signal_values[:, input_b_ids[mask][input_b_has_signal[mask]]]
            if bool(input_a_const[mask].any()):
                a[:, input_a_const[mask]] = True
            if bool(input_b_const[mask].any()):
                b[:, input_b_const[mask]] = True
            signal_values[:, out_ids[mask]] = eval_gate_torch(gate_id, a, b)

    final_bits = signal_values[:, circuit.output.input_start : circuit.output.input_start + circuit.output.input_dim].to(torch.int64)
    group_size = circuit.output.group_size
    pad = group_size * circuit.output.num_classes - final_bits.shape[1]
    if pad > 0:
        final_bits = torch.cat((final_bits, torch.zeros((final_bits.shape[0], pad), dtype=torch.int64, device=target_device)), dim=1)
    class_counts = final_bits.view(final_bits.shape[0], circuit.output.num_classes, group_size).sum(dim=-1)
    logits = class_counts.to(torch.float32) / float(circuit.output.tau)
    return {'class_counts': class_counts.cpu(), 'logits': logits.cpu()}


def _uniform_thresholds(levels: int) -> torch.Tensor:
    return torch.linspace(1, levels, levels, dtype=torch.float32) / (levels + 1)


def _threshold_transform(levels: int):
    thresholds = _uniform_thresholds(levels)

    def _apply_thresholds(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([(x > t).float() for t in thresholds], dim=0)

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(_apply_thresholds),
    ])


def _binarize_loaded_batch(batch_inputs: torch.Tensor) -> torch.Tensor:
    return (batch_inputs >= 0.5).to(torch.bool)


def make_test_loader(checkpoint_path: str | Path, *, batch_size: int, download: bool = True) -> DataLoader:
    """Build the full test loader for a supported ConvLogic checkpoint."""
    config = infer_config_from_checkpoint(checkpoint_path)
    if config.dataset_name == 'mnist':
        ds = torchvision.datasets.MNIST(
            root=_MNIST_DATA_ROOT,
            train=False,
            download=download,
            transform=torchvision.transforms.ToTensor(),
        )
    elif config.dataset_name.startswith('cifar10'):
        threshold_levels = config.input_channels // 3
        ds = torchvision.datasets.CIFAR10(
            root=_CIFAR10_DATA_ROOT,
            train=False,
            download=download,
            transform=_threshold_transform(threshold_levels),
        )
    else:
        raise ValueError(f'Unsupported dataset for ConvLogic adapter: {config.dataset_name!r}')
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


def evaluate_extracted_checkpoint(
    checkpoint_path: str | Path,
    batch_inputs: torch.Tensor,
    *,
    device: str | torch.device = 'cpu',
) -> dict[str, torch.Tensor]:
    """Evaluate the extracted common-IR circuit for a ConvLogic checkpoint."""
    circuit = extract_circuit_from_checkpoint(checkpoint_path)
    return evaluate_circuit_torch(circuit, batch_inputs, device=device)
