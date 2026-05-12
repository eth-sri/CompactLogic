from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from compiler import extract_circuit_from_checkpoint
from compiler.gates import eval_gate_torch
from compiler.ir import GateInput, GateNode
from compiler.model_builder import build_compactlogic_model, input_shape_of_dataset
from compiler.prune import NaivePrunedCircuit


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _infer_input_shape(dataset: str) -> tuple[int, ...]:
    return input_shape_of_dataset(dataset)


def _build_regular_model(args: dict[str, Any]) -> torch.nn.Sequential:
    return build_compactlogic_model(args)


def _prepare_bool_inputs(batch_inputs: torch.Tensor, *, input_dim: int, device: torch.device) -> torch.Tensor:
    x = batch_inputs.to(device=device, dtype=torch.bool)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] != input_dim:
        raise ValueError(f'Input shape mismatch: got {tuple(x.shape)}, expected second dim {input_dim}.')
    return x


def _resolve_gate_inputs(
    nodes: list[GateNode] | tuple[GateNode, ...],
    *,
    attr: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signal_ids: list[int] = []
    has_signal: list[bool] = []
    const_bits: list[bool] = []
    for node in nodes:
        source: GateInput = getattr(node, attr)
        if source.signal_id is not None:
            signal_ids.append(source.signal_id)
            has_signal.append(True)
            const_bits.append(False)
        else:
            signal_ids.append(0)
            has_signal.append(False)
            const_bits.append(bool(source.const_value))
    return (
        torch.tensor(signal_ids, dtype=torch.long, device=device),
        torch.tensor(has_signal, dtype=torch.bool, device=device),
        torch.tensor(const_bits, dtype=torch.bool, device=device),
    )


def _gather_gate_values(
    signal_values: torch.Tensor,
    signal_ids: torch.Tensor,
    has_signal: torch.Tensor,
    const_bits: torch.Tensor,
) -> torch.Tensor:
    gathered = torch.zeros((signal_values.shape[0], signal_ids.numel()), dtype=torch.bool, device=signal_values.device)
    if bool(has_signal.any()):
        gathered[:, has_signal] = signal_values[:, signal_ids[has_signal]]
    if bool(const_bits.any()):
        gathered[:, const_bits] = True
    return gathered


@torch.no_grad()
def evaluate_extracted_circuit_torch(
    checkpoint_path: str | Path,
    batch_inputs: torch.Tensor,
    metadata_path: str | Path | None = None,
    *,
    device: str | torch.device = 'cpu',
) -> dict[str, torch.Tensor]:
    circuit = extract_circuit_from_checkpoint(checkpoint_path, metadata_path)
    target_device = torch.device(device)
    x = _prepare_bool_inputs(batch_inputs, input_dim=circuit.input_dim, device=target_device)

    total_signals = circuit.total_signal_count
    signal_values = torch.zeros((x.shape[0], total_signals), dtype=torch.bool, device=target_device)
    signal_values[:, : circuit.input_dim] = x

    for layer in circuit.layers:
        gate_ids = torch.tensor([node.gate_id for node in layer.nodes], dtype=torch.long, device=target_device)
        out_ids = torch.tensor([node.node_id for node in layer.nodes], dtype=torch.long, device=target_device)
        input_a_ids, input_a_has_signal, input_a_const = _resolve_gate_inputs(layer.nodes, attr='input_a', device=target_device)
        input_b_ids, input_b_has_signal, input_b_const = _resolve_gate_inputs(layer.nodes, attr='input_b', device=target_device)

        for gate_id in range(16):
            mask = gate_ids == gate_id
            if not bool(mask.any()):
                continue
            a = _gather_gate_values(
                signal_values,
                input_a_ids[mask],
                input_a_has_signal[mask],
                input_a_const[mask],
            )
            b = _gather_gate_values(
                signal_values,
                input_b_ids[mask],
                input_b_has_signal[mask],
                input_b_const[mask],
            )
            signal_values[:, out_ids[mask]] = eval_gate_torch(gate_id, a, b)

    final_bits = signal_values[:, circuit.output.input_start : circuit.output.input_start + circuit.output.input_dim].to(torch.int64)
    group_size = circuit.output.group_size
    pad = group_size * circuit.output.num_classes - final_bits.shape[1]
    if pad > 0:
        final_bits = torch.cat((final_bits, torch.zeros((final_bits.shape[0], pad), dtype=torch.int64, device=target_device)), dim=1)
    class_counts = final_bits.view(final_bits.shape[0], circuit.output.num_classes, group_size).sum(dim=-1)
    logits = class_counts.to(torch.float32) / float(circuit.output.tau)
    return {
        'class_counts': class_counts.cpu(),
        'logits': logits.cpu(),
    }


@torch.no_grad()
def evaluate_regular_circuit_cpu(
    checkpoint_path: str | Path,
    batch_inputs: torch.Tensor,
    metadata_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    return evaluate_extracted_circuit_torch(
        checkpoint_path,
        batch_inputs,
        metadata_path,
        device='cpu',
    )


@torch.no_grad()
def evaluate_naive_pruned_circuit(
    pruned: NaivePrunedCircuit,
    batch_inputs: torch.Tensor,
    *,
    device: str | torch.device = 'cpu',
) -> dict[str, torch.Tensor]:
    target_device = torch.device(device)
    x = _prepare_bool_inputs(batch_inputs, input_dim=pruned.input_dim, device=target_device)

    max_signal_id = max((node.node_id for node in pruned.nodes), default=pruned.input_dim - 1)
    signal_values = torch.zeros((x.shape[0], max_signal_id + 1), dtype=torch.bool, device=target_device)
    signal_values[:, : pruned.input_dim] = x

    nodes_by_layer: dict[int, list[Any]] = defaultdict(list)
    for node in pruned.nodes:
        nodes_by_layer[node.layer_index].append(node)

    for layer_index in sorted(nodes_by_layer):
        layer_nodes = nodes_by_layer[layer_index]
        gate_ids = torch.tensor([node.gate_id for node in layer_nodes], dtype=torch.long, device=target_device)
        out_ids = torch.tensor([node.node_id for node in layer_nodes], dtype=torch.long, device=target_device)
        input_a_ids, input_a_has_signal, input_a_const = _resolve_gate_inputs(layer_nodes, attr='input_a', device=target_device)
        input_b_ids, input_b_has_signal, input_b_const = _resolve_gate_inputs(layer_nodes, attr='input_b', device=target_device)

        for gate_id in range(16):
            mask = gate_ids == gate_id
            if not bool(mask.any()):
                continue
            a = _gather_gate_values(
                signal_values,
                input_a_ids[mask],
                input_a_has_signal[mask],
                input_a_const[mask],
            )
            b = _gather_gate_values(
                signal_values,
                input_b_ids[mask],
                input_b_has_signal[mask],
                input_b_const[mask],
            )
            signal_values[:, out_ids[mask]] = eval_gate_torch(gate_id, a, b)

    counts = []
    for group in pruned.class_groups:
        dynamic_ids = [source.signal_id for source in group if source.signal_id is not None]
        inverted = torch.tensor([source.inverted for source in group if source.signal_id is not None], dtype=torch.bool, device=target_device)
        const_bias = sum(int(source.const_value) for source in group if source.const_value is not None)

        if dynamic_ids:
            bits = signal_values[:, torch.tensor(dynamic_ids, dtype=torch.long, device=target_device)]
            if bool(inverted.any()):
                bits = torch.where(inverted.unsqueeze(0), ~bits, bits)
            acc = bits.to(torch.int64).sum(dim=1)
        else:
            acc = torch.zeros((x.shape[0],), dtype=torch.int64, device=target_device)
        if const_bias:
            acc = acc + const_bias
        counts.append(acc)

    class_counts = torch.stack(counts, dim=1)
    tau = float(pruned.metadata.get('tau', 1.0))
    logits = class_counts.to(torch.float32) / tau
    return {
        'class_counts': class_counts.cpu(),
        'logits': logits.cpu(),
    }


@torch.no_grad()
def evaluate_naive_pruned_circuit_cpu(pruned: NaivePrunedCircuit, batch_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    return evaluate_naive_pruned_circuit(pruned, batch_inputs, device='cpu')


@torch.no_grad()
def evaluate_original_model_cuda(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    batch_inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; original compactlogic model cannot be evaluated here.')

    metadata = _load_json(Path(metadata_path))
    args = metadata['args']
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = _build_regular_model(args)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model = model.to('cuda')
    model.eval()

    logits = model(batch_inputs.to(torch.float32).to('cuda')).detach().cpu()
    tau = float(args['tau'])
    class_counts = torch.round(logits * tau).to(torch.int64)
    return {
        'logits': logits,
        'class_counts': class_counts,
    }


def run_consistency_check(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    batch_size: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    metadata = _load_json(Path(metadata_path))
    dataset = metadata['args']['dataset']
    input_shape = _infer_input_shape(dataset)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    batch_inputs = torch.randint(0, 2, (batch_size, *input_shape), generator=generator, dtype=torch.int64).to(torch.bool)

    compiled = evaluate_regular_circuit_cpu(checkpoint_path, batch_inputs, metadata_path)

    result: dict[str, Any] = {
        'batch_size': batch_size,
        'seed': seed,
        'dataset': dataset,
        'compiled_class_counts': compiled['class_counts'].tolist(),
        'compiled_logits': compiled['logits'].tolist(),
        'checked_against_original_model': False,
        'reference_mode': 'extracted_discrete_circuit',
    }

    if torch.cuda.is_available():
        original = evaluate_original_model_cuda(checkpoint_path, metadata_path, batch_inputs)
        same_counts = torch.equal(compiled['class_counts'], original['class_counts'])
        max_logit_diff = (compiled['logits'] - original['logits']).abs().max().item()
        result.update(
            {
                'checked_against_original_model': True,
                'original_class_counts': original['class_counts'].tolist(),
                'original_logits': original['logits'].tolist(),
                'same_class_counts': bool(same_counts),
                'same_argmax': bool(torch.equal(compiled['class_counts'].argmax(dim=1), original['class_counts'].argmax(dim=1))),
                'max_logit_abs_diff': float(max_logit_diff),
                'reference_mode': 'original_compactlogic_model',
            }
        )
    else:
        result['note'] = 'CUDA unavailable; original model comparison skipped.'

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Check consistency between compiled compactlogic IR and the original compactlogic model.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    result = run_consistency_check(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
