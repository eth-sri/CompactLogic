"""Circuit extraction from supported vendored ConvLogic checkpoints into the common IR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from compiler.ir import GateInput, GateNode, GroupSumIR, LogicLayerIR, RegularCircuitIR

from .config import ConvLogicConfig, infer_config_from_checkpoint

OR_GATE_ID = 7


@dataclass(frozen=True)
class _Shape3D:
    channels: int
    height: int
    width: int

    @property
    def dim(self) -> int:
        return self.channels * self.height * self.width


def _flat_index(shape: _Shape3D, channel: int, row: int, col: int) -> int:
    return (channel * shape.height + row) * shape.width + col


def _signal_from_feature(shape: _Shape3D, input_start: int, channel: int, row: int, col: int) -> GateInput:
    return GateInput(signal_id=input_start + _flat_index(shape, channel, row, col))


def _unpack_selection_word(word: int) -> tuple[int, int, int]:
    channel = (word >> 16) & 0xFFFF
    row = (word >> 8) & 0xFF
    col = word & 0xFF
    return int(channel), int(row), int(col)


def _make_layer(
    *,
    layer_index: int,
    input_start: int,
    in_dim: int,
    next_node_id: int,
    gate_specs: list[tuple[int, GateInput, GateInput]],
) -> LogicLayerIR:
    nodes = tuple(
        GateNode(
            node_id=next_node_id + local_index,
            layer_index=layer_index,
            local_index=local_index,
            gate_id=gate_id,
            input_a=input_a,
            input_b=input_b,
        )
        for local_index, (gate_id, input_a, input_b) in enumerate(gate_specs)
    )
    return LogicLayerIR(
        layer_index=layer_index,
        in_dim=in_dim,
        out_dim=len(nodes),
        input_start=input_start,
        output_start=next_node_id,
        nodes=nodes,
    )


def _conv_hyperparams(config: ConvLogicConfig) -> list[tuple[int, int]]:
    if config.dataset_name == 'mnist':
        return [(5, 0), (3, 1), (3, 1)]
    if config.dataset_name.startswith('cifar10'):
        return [(3, 1)] * config.conv_layers
    raise ValueError(f'Unsupported ConvLogic dataset for extraction: {config.dataset_name!r}')


def _sorted_prefixes(state_dict: dict[str, torch.Tensor], suffix: str) -> list[str]:
    return sorted({key.rsplit('.', 1)[0] for key in state_dict if key.endswith(suffix)})


def _extract_complete_conv_layer(
    *,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    layer_index_start: int,
    input_shape: _Shape3D,
    input_start: int,
    next_node_id: int,
    kernel: int,
    padding: int,
) -> tuple[list[LogicLayerIR], _Shape3D, int]:
    weights = state_dict[f'{prefix}.weights']
    selection = state_dict[f'{prefix}.selection'].to(torch.int64)

    out_channels, gate_count, truth_table_count = weights.shape
    if gate_count != 7 or truth_table_count != 16:
        raise ValueError(f'Unexpected ConvLogic layer shape for {prefix}: {tuple(weights.shape)}')

    gate_ids = weights.argmax(dim=-1).to(torch.long)

    h_stage1 = input_shape.height + 2 * padding - kernel + 1
    w_stage1 = input_shape.width + 2 * padding - kernel + 1
    if h_stage1 % 2 != 0 or w_stage1 % 2 != 0:
        raise ValueError(f'ConvLogic pooling requires even spatial shape after conv; got {(h_stage1, w_stage1)} for {prefix}')
    pool_shape = _Shape3D(out_channels, h_stage1 // 2, w_stage1 // 2)

    stage1_specs: list[tuple[int, GateInput, GateInput]] = []
    for out_c in range(out_channels):
        decoded = [_unpack_selection_word(int(v.item())) for v in selection[out_c]]
        for out_r in range(h_stage1):
            for out_w in range(w_stage1):
                for pair in range(4):
                    ch_a, off_r_a, off_c_a = decoded[2 * pair]
                    ch_b, off_r_b, off_c_b = decoded[2 * pair + 1]

                    in_r_a = out_r + off_r_a - padding
                    in_c_a = out_w + off_c_a - padding
                    if 0 <= in_r_a < input_shape.height and 0 <= in_c_a < input_shape.width:
                        input_a = _signal_from_feature(input_shape, input_start, ch_a, in_r_a, in_c_a)
                    else:
                        input_a = GateInput(const_value=0)

                    in_r_b = out_r + off_r_b - padding
                    in_c_b = out_w + off_c_b - padding
                    if 0 <= in_r_b < input_shape.height and 0 <= in_c_b < input_shape.width:
                        input_b = _signal_from_feature(input_shape, input_start, ch_b, in_r_b, in_c_b)
                    else:
                        input_b = GateInput(const_value=0)

                    stage1_specs.append((int(gate_ids[out_c, pair].item()), input_a, input_b))

    layer_stage1 = _make_layer(
        layer_index=layer_index_start,
        input_start=input_start,
        in_dim=input_shape.dim,
        next_node_id=next_node_id,
        gate_specs=stage1_specs,
    )
    stage1_base = layer_stage1.output_start

    stage2_specs: list[tuple[int, GateInput, GateInput]] = []
    for out_c in range(out_channels):
        for out_r in range(h_stage1):
            for out_w in range(w_stage1):
                base = (((out_c * h_stage1) + out_r) * w_stage1 + out_w) * 4
                stage2_specs.append((int(gate_ids[out_c, 4].item()), GateInput(signal_id=stage1_base + base + 0), GateInput(signal_id=stage1_base + base + 1)))
                stage2_specs.append((int(gate_ids[out_c, 5].item()), GateInput(signal_id=stage1_base + base + 2), GateInput(signal_id=stage1_base + base + 3)))

    layer_stage2 = _make_layer(
        layer_index=layer_index_start + 1,
        input_start=layer_stage1.output_start,
        in_dim=layer_stage1.out_dim,
        next_node_id=layer_stage1.output_stop,
        gate_specs=stage2_specs,
    )
    stage2_base = layer_stage2.output_start

    stage3_specs: list[tuple[int, GateInput, GateInput]] = []
    for out_c in range(out_channels):
        for out_r in range(h_stage1):
            for out_w in range(w_stage1):
                base = (((out_c * h_stage1) + out_r) * w_stage1 + out_w) * 2
                stage3_specs.append((int(gate_ids[out_c, 6].item()), GateInput(signal_id=stage2_base + base + 0), GateInput(signal_id=stage2_base + base + 1)))

    layer_stage3 = _make_layer(
        layer_index=layer_index_start + 2,
        input_start=layer_stage2.output_start,
        in_dim=layer_stage2.out_dim,
        next_node_id=layer_stage2.output_stop,
        gate_specs=stage3_specs,
    )
    stage3_base = layer_stage3.output_start

    pool_stage1_specs: list[tuple[int, GateInput, GateInput]] = []
    for out_c in range(out_channels):
        for pool_r in range(pool_shape.height):
            for pool_c in range(pool_shape.width):
                src_r = pool_r * 2
                src_c = pool_c * 2
                idx00 = ((out_c * h_stage1 + src_r) * w_stage1 + src_c)
                idx01 = ((out_c * h_stage1 + src_r) * w_stage1 + (src_c + 1))
                idx10 = ((out_c * h_stage1 + (src_r + 1)) * w_stage1 + src_c)
                idx11 = ((out_c * h_stage1 + (src_r + 1)) * w_stage1 + (src_c + 1))
                pool_stage1_specs.append((OR_GATE_ID, GateInput(signal_id=stage3_base + idx00), GateInput(signal_id=stage3_base + idx01)))
                pool_stage1_specs.append((OR_GATE_ID, GateInput(signal_id=stage3_base + idx10), GateInput(signal_id=stage3_base + idx11)))

    layer_pool1 = _make_layer(
        layer_index=layer_index_start + 3,
        input_start=layer_stage3.output_start,
        in_dim=layer_stage3.out_dim,
        next_node_id=layer_stage3.output_stop,
        gate_specs=pool_stage1_specs,
    )
    pool1_base = layer_pool1.output_start

    pool_stage2_specs: list[tuple[int, GateInput, GateInput]] = []
    for out_c in range(out_channels):
        for pool_r in range(pool_shape.height):
            for pool_c in range(pool_shape.width):
                base = (((out_c * pool_shape.height) + pool_r) * pool_shape.width + pool_c) * 2
                pool_stage2_specs.append((OR_GATE_ID, GateInput(signal_id=pool1_base + base + 0), GateInput(signal_id=pool1_base + base + 1)))

    layer_pool2 = _make_layer(
        layer_index=layer_index_start + 4,
        input_start=layer_pool1.output_start,
        in_dim=layer_pool1.out_dim,
        next_node_id=layer_pool1.output_stop,
        gate_specs=pool_stage2_specs,
    )

    return [layer_stage1, layer_stage2, layer_stage3, layer_pool1, layer_pool2], pool_shape, layer_pool2.output_stop


def _extract_dense_layer(
    *,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    layer_index: int,
    input_start: int,
    input_dim: int,
    next_node_id: int,
) -> LogicLayerIR:
    weights = state_dict[f'{prefix}.weights']
    indices_0 = state_dict[f'{prefix}.indices_0'].to(torch.long)
    indices_1 = state_dict[f'{prefix}.indices_1'].to(torch.long)
    gate_ids = weights.argmax(dim=-1).to(torch.long)
    gate_specs = [
        (
            int(gate_ids[i].item()),
            GateInput(signal_id=input_start + int(indices_0[i].item())),
            GateInput(signal_id=input_start + int(indices_1[i].item())),
        )
        for i in range(weights.shape[0])
    ]
    return _make_layer(
        layer_index=layer_index,
        input_start=input_start,
        in_dim=input_dim,
        next_node_id=next_node_id,
        gate_specs=gate_specs,
    )


def extract_circuit_from_checkpoint(checkpoint_path: str | Path) -> RegularCircuitIR:
    """Extract a supported ConvLogic checkpoint into this repository's common circuit IR."""
    checkpoint_path = Path(checkpoint_path)
    config = infer_config_from_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    layers: list[LogicLayerIR] = []
    next_node_id = config.input_channels * config.input_size * config.input_size
    layer_index = 0

    input_shape = _Shape3D(config.input_channels, config.input_size, config.input_size)
    input_start = 0

    conv_prefixes = _sorted_prefixes(state_dict, '.selection')
    conv_hparams = _conv_hyperparams(config)
    if len(conv_prefixes) != len(conv_hparams):
        raise ValueError(f'Expected {len(conv_hparams)} convolutional layers for {config.variant_name}, found {len(conv_prefixes)}')

    for prefix, (kernel, padding) in zip(conv_prefixes, conv_hparams, strict=True):
        extracted_layers, input_shape, next_node_id = _extract_complete_conv_layer(
            state_dict=state_dict,
            prefix=prefix,
            layer_index_start=layer_index,
            input_shape=input_shape,
            input_start=input_start,
            next_node_id=next_node_id,
            kernel=kernel,
            padding=padding,
        )
        layers.extend(extracted_layers)
        layer_index += len(extracted_layers)
        input_start = extracted_layers[-1].output_start

    dense_input_dim = input_shape.dim
    dense_prefixes = _sorted_prefixes(state_dict, '.indices_0')
    for prefix in dense_prefixes:
        layer = _extract_dense_layer(
            state_dict=state_dict,
            prefix=prefix,
            layer_index=layer_index,
            input_start=input_start,
            input_dim=dense_input_dim,
            next_node_id=next_node_id,
        )
        layers.append(layer)
        layer_index += 1
        input_start = layer.output_start
        dense_input_dim = layer.out_dim
        next_node_id = layer.output_stop

    output = GroupSumIR(
        num_classes=config.num_classes,
        tau=config.tau,
        input_start=input_start,
        input_dim=dense_input_dim,
    )
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'metadata_path': str(checkpoint_path),
        'dataset': config.dataset_name,
        'struct': 'convlogic_complete',
        'experiment_id': checkpoint_path.parent.name,
        'model_scale': config.k,
        'tau': config.tau,
        'source': 'third_party.convlogic',
        'variant_name': config.variant_name,
    }
    return RegularCircuitIR(
        input_dim=config.input_channels * config.input_size * config.input_size,
        layers=tuple(layers),
        output=output,
        metadata=metadata,
    )
