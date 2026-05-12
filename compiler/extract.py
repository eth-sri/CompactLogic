from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from compactlogic import ConvLayer, LogicLayer

from .ir import CompiledCircuitIR, GateInput, GateNode, GroupSumIR, LogicLayerIR, RegularCircuitIR
from .model_builder import build_compactlogic_model, input_dim_of_dataset, num_classes_of_dataset


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_metadata_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name("meta_data.json")


def load_checkpoint_artifacts(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and the sibling experiment metadata JSON."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else infer_metadata_path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = _load_json(metadata_path)
    return checkpoint, metadata


def _logic_layer_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    prefixes: list[str] = []
    for key in state_dict:
        prefix, _, suffix = key.partition(".")
        if suffix == "weights" and prefix.isdigit():
            prefixes.append(prefix)
    return sorted(prefixes, key=int)


def _extract_dense_layer_ir(
    *,
    prefix: str,
    logical_layer_index: int,
    state_dict: dict[str, torch.Tensor],
    input_start: int,
    in_dim: int,
    next_node_id: int,
) -> LogicLayerIR:
    weights = state_dict[f"{prefix}.weights"]
    indices = state_dict[f"{prefix}.indices"]
    gate_sequence = state_dict[f"{prefix}.gate_sequence"]

    out_dim, num_gates = weights.shape
    if indices.shape != (2, out_dim, num_gates):
        raise ValueError(
            f"Layer {prefix} has incompatible indices shape {tuple(indices.shape)} for weights shape {tuple(weights.shape)}"
        )
    if gate_sequence.shape != (out_dim, num_gates):
        raise ValueError(
            f"Layer {prefix} has incompatible gate_sequence shape {tuple(gate_sequence.shape)} for weights shape {tuple(weights.shape)}"
        )

    winners = weights.argmax(dim=1)
    row_ids = torch.arange(out_dim, dtype=torch.long)

    chosen_a = indices[0, row_ids, winners].to(torch.long)
    chosen_b = indices[1, row_ids, winners].to(torch.long)
    chosen_gate = gate_sequence[row_ids, winners].to(torch.long)

    nodes = tuple(
        GateNode(
            node_id=next_node_id + local_index,
            layer_index=logical_layer_index,
            local_index=local_index,
            gate_id=int(chosen_gate[local_index].item()),
            input_a=GateInput(signal_id=input_start + int(chosen_a[local_index].item())),
            input_b=GateInput(signal_id=input_start + int(chosen_b[local_index].item())),
        )
        for local_index in range(out_dim)
    )

    return LogicLayerIR(
        layer_index=logical_layer_index,
        in_dim=in_dim,
        out_dim=out_dim,
        input_start=input_start,
        output_start=next_node_id,
        nodes=nodes,
    )


def _flatten_nchw_index(c: int, h: int, w: int, h_in: int, w_in: int) -> int:
    return (c * h_in + h) * w_in + w


def _extract_conv_layer_ir(
    *,
    prefix: str,
    logical_layer_index: int,
    module: ConvLayer,
    state_dict: dict[str, torch.Tensor],
    input_start: int,
    next_node_id: int,
) -> LogicLayerIR:
    weights = state_dict[f"{prefix}.weights"]
    gate_sequence = state_dict[f"{prefix}.gate_sequence"]
    offsets_ch = state_dict[f"{prefix}.offsets_ch"].to(torch.long)
    offsets_h = state_dict[f"{prefix}.offsets_h"].to(torch.long)
    offsets_w = state_dict[f"{prefix}.offsets_w"].to(torch.long)

    c_out, num_gates = weights.shape
    expected_offsets_shape = (2, c_out, num_gates)
    if gate_sequence.shape != (c_out, num_gates):
        raise ValueError(
            f"Layer {prefix} has incompatible gate_sequence shape {tuple(gate_sequence.shape)} for weights shape {tuple(weights.shape)}"
        )
    if offsets_ch.shape != expected_offsets_shape or offsets_h.shape != expected_offsets_shape or offsets_w.shape != expected_offsets_shape:
        raise ValueError(
            f"Layer {prefix} has incompatible conv offset shapes: ch={tuple(offsets_ch.shape)}, h={tuple(offsets_h.shape)}, w={tuple(offsets_w.shape)}"
        )

    winners = weights.argmax(dim=1)
    row_ids = torch.arange(c_out, dtype=torch.long)
    chosen_gate = gate_sequence[row_ids, winners].to(torch.long)
    chosen_ch_a = offsets_ch[0, row_ids, winners]
    chosen_ch_b = offsets_ch[1, row_ids, winners]
    chosen_h_a = offsets_h[0, row_ids, winners]
    chosen_h_b = offsets_h[1, row_ids, winners]
    chosen_w_a = offsets_w[0, row_ids, winners]
    chosen_w_b = offsets_w[1, row_ids, winners]

    c_in, h_in, w_in = module.in_shape
    _, h_out, w_out = module.out_shape
    out_dim = int(module.out_dim)
    stride = int(module.stride)

    nodes: list[GateNode] = []
    for out_c in range(c_out):
        gate_id = int(chosen_gate[out_c].item())
        in_ch_a = int(chosen_ch_a[out_c].item())
        in_ch_b = int(chosen_ch_b[out_c].item())
        off_h_a = int(chosen_h_a[out_c].item())
        off_h_b = int(chosen_h_b[out_c].item())
        off_w_a = int(chosen_w_a[out_c].item())
        off_w_b = int(chosen_w_b[out_c].item())

        for out_h in range(h_out):
            base_h = out_h * stride
            for out_w in range(w_out):
                base_w = out_w * stride
                local_index = (out_c * h_out + out_h) * w_out + out_w

                in_h_a = base_h + off_h_a
                in_w_a = base_w + off_w_a
                if 0 <= in_h_a < h_in and 0 <= in_w_a < w_in:
                    input_a = GateInput(signal_id=input_start + _flatten_nchw_index(in_ch_a, in_h_a, in_w_a, h_in, w_in))
                else:
                    input_a = GateInput(const_value=0)

                in_h_b = base_h + off_h_b
                in_w_b = base_w + off_w_b
                if 0 <= in_h_b < h_in and 0 <= in_w_b < w_in:
                    input_b = GateInput(signal_id=input_start + _flatten_nchw_index(in_ch_b, in_h_b, in_w_b, h_in, w_in))
                else:
                    input_b = GateInput(const_value=0)

                nodes.append(
                    GateNode(
                        node_id=next_node_id + local_index,
                        layer_index=logical_layer_index,
                        local_index=local_index,
                        gate_id=gate_id,
                        input_a=input_a,
                        input_b=input_b,
                    )
                )

    if len(nodes) != out_dim:
        raise ValueError(f"Conv layer {prefix} extracted {len(nodes)} nodes, expected {out_dim}.")

    return LogicLayerIR(
        layer_index=logical_layer_index,
        in_dim=int(module.in_dim),
        out_dim=out_dim,
        input_start=input_start,
        output_start=next_node_id,
        nodes=tuple(nodes),
    )


def extract_circuit_from_state(
    state_dict: dict[str, torch.Tensor],
    *,
    args: dict[str, Any] | None = None,
    num_classes: int,
    tau: float,
    metadata: dict[str, Any] | None = None,
) -> CompiledCircuitIR:
    """Extract a discretized compactlogic circuit IR from a checkpoint state dict.

    The historical function name is kept for compatibility, but the extractor now
    supports both regular and conv-based compactlogic models whose layers ultimately
    discretize into 2-input Boolean gates.
    """
    layer_prefixes = _logic_layer_prefixes(state_dict)
    if not layer_prefixes:
        raise ValueError("No LogicLayer/ConvLayer weights found in state dict.")

    args = args or {}
    dataset = str(args.get("dataset", metadata.get("dataset") if metadata is not None else ""))
    input_dim = input_dim_of_dataset(dataset)
    model = build_compactlogic_model(args) if args else None

    layers: list[LogicLayerIR] = []
    prev_output_start = 0
    prev_out_dim = input_dim
    next_node_id = input_dim

    for logical_layer_index, prefix in enumerate(layer_prefixes):
        module = model[int(prefix)] if model is not None else None
        if isinstance(module, ConvLayer):
            layer_ir = _extract_conv_layer_ir(
                prefix=prefix,
                logical_layer_index=logical_layer_index,
                module=module,
                state_dict=state_dict,
                input_start=prev_output_start,
                next_node_id=next_node_id,
            )
        else:
            if not isinstance(module, LogicLayer) and f"{prefix}.indices" not in state_dict:
                raise ValueError(f"Unsupported or unclassified logic layer at prefix {prefix}.")
            layer_ir = _extract_dense_layer_ir(
                prefix=prefix,
                logical_layer_index=logical_layer_index,
                state_dict=state_dict,
                input_start=prev_output_start,
                in_dim=prev_out_dim,
                next_node_id=next_node_id,
            )

        layers.append(layer_ir)
        prev_output_start = layer_ir.output_start
        prev_out_dim = layer_ir.out_dim
        next_node_id = layer_ir.output_stop

    output = GroupSumIR(
        num_classes=int(num_classes),
        tau=float(tau),
        input_start=prev_output_start,
        input_dim=prev_out_dim,
    )

    return RegularCircuitIR(
        input_dim=input_dim,
        layers=tuple(layers),
        output=output,
        metadata=metadata or {},
    )


def extract_circuit_from_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
) -> CompiledCircuitIR:
    """Load the checkpoint and metadata, then extract the discretized circuit IR."""
    checkpoint, metadata = load_checkpoint_artifacts(checkpoint_path, metadata_path)
    args = metadata.get("args", {})
    state_dict = checkpoint["model_state"]

    num_classes = _infer_num_classes(args)
    tau = float(args["tau"])

    compiler_metadata = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "metadata_path": str(Path(metadata_path) if metadata_path is not None else infer_metadata_path(checkpoint_path)),
        "dataset": args.get("dataset"),
        "struct": args.get("struct"),
        "experiment_id": args.get("experiment_id"),
        "model_scale": args.get("model_scale"),
        "num_layers": args.get("num_layers"),
        "num_gates": args.get("num_gates"),
        "tau": tau,
        "checkpoint_step": checkpoint.get("step"),
    }

    return extract_circuit_from_state(
        state_dict,
        args=args,
        num_classes=num_classes,
        tau=tau,
        metadata=compiler_metadata,
    )


def _infer_num_classes(args: dict[str, Any]) -> int:
    return num_classes_of_dataset(str(args.get("dataset")))


def extract_regular_circuit_from_state(
    state_dict: dict[str, torch.Tensor],
    *,
    args: dict[str, Any] | None = None,
    num_classes: int,
    tau: float,
    metadata: dict[str, Any] | None = None,
) -> RegularCircuitIR:
    return extract_circuit_from_state(
        state_dict,
        args=args,
        num_classes=num_classes,
        tau=tau,
        metadata=metadata,
    )


def extract_regular_circuit_from_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
) -> RegularCircuitIR:
    return extract_circuit_from_checkpoint(checkpoint_path, metadata_path)


def _preview_nodes(circuit: RegularCircuitIR, per_layer: int = 3) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for layer in circuit.layers:
        for node in layer.nodes[:per_layer]:
            preview.append(
                {
                    "layer_index": node.layer_index,
                    "local_index": node.local_index,
                    "node_id": node.node_id,
                    "gate_id": node.gate_id,
                    "gate_name": node.gate_name,
                    "input_a": asdict(node.input_a),
                    "input_b": asdict(node.input_b),
                }
            )
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a compactlogic checkpoint into compiler IR.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ckpt_best_*.pt")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to sibling meta_data.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full IR as JSON. By default, print a compact summary.",
    )
    args = parser.parse_args()

    circuit = extract_regular_circuit_from_checkpoint(args.checkpoint, args.metadata)
    if args.json:
        print(json.dumps(circuit.to_dict(), indent=2))
        return

    payload = circuit.summary()
    payload["preview_nodes"] = _preview_nodes(circuit)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
