"""Narrow top-level compiler API for report generation and final Verilog emission."""

from .ir import CompiledCircuitIR, GateNode, GroupSumIR, LogicLayerIR, RegularCircuitIR
from .prune import BoolSource, NaivePrunedCircuit

__all__ = [
    "GateNode",
    "GroupSumIR",
    "LogicLayerIR",
    "CompiledCircuitIR",
    "RegularCircuitIR",
    "BoolSource",
    "NaivePrunedCircuit",
    "extract_circuit_from_checkpoint",
    "extract_regular_circuit_from_checkpoint",
    "emit_regular_verilog_from_checkpoint",
    "naive_prune_circuit",
    "naive_prune_regular_circuit",
    "reindex_pruned_circuit",
]


def __getattr__(name: str):
    if name in {"extract_circuit_from_checkpoint", "extract_regular_circuit_from_checkpoint"}:
        from .extract import (
            extract_circuit_from_checkpoint,
            extract_regular_circuit_from_checkpoint,
        )

        return {
            "extract_circuit_from_checkpoint": extract_circuit_from_checkpoint,
            "extract_regular_circuit_from_checkpoint": extract_regular_circuit_from_checkpoint,
        }[name]
    if name == "emit_regular_verilog_from_checkpoint":
        from .verilog import emit_regular_verilog_from_checkpoint

        return {
            "emit_regular_verilog_from_checkpoint": emit_regular_verilog_from_checkpoint,
        }[name]
    if name in {
        "naive_prune_circuit",
        "naive_prune_regular_circuit",
        "reindex_pruned_circuit",
    }:
        from .prune import naive_prune_circuit, naive_prune_regular_circuit, reindex_pruned_circuit

        return {
            "naive_prune_circuit": naive_prune_circuit,
            "naive_prune_regular_circuit": naive_prune_regular_circuit,
            "reindex_pruned_circuit": reindex_pruned_circuit,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
