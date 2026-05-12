from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compiler import extract_circuit_from_checkpoint, naive_prune_regular_circuit, reindex_pruned_circuit
from compiler.ir import GateInput
from compiler.prune import BoolSource, NaivePrunedCircuit


@dataclass(frozen=True)
class TimingProfile:
    name: str
    register_overhead_ns: float
    boolean_level_ns: float
    adder_level_ns: float
    notes: str


DEFAULT_TIMING_PROFILES: dict[str, TimingProfile] = {
    'optimistic': TimingProfile(
        name='optimistic',
        register_overhead_ns=0.80,
        boolean_level_ns=0.12,
        adder_level_ns=0.20,
        notes='Loose optimistic placeholder for a well-mapped/pipelined FPGA datapath.',
    ),
    'nominal': TimingProfile(
        name='nominal',
        register_overhead_ns=1.20,
        boolean_level_ns=0.18,
        adder_level_ns=0.30,
        notes='Default placeholder profile for CPU-only early estimation; not a synthesis report.',
    ),
    'pessimistic': TimingProfile(
        name='pessimistic',
        register_overhead_ns=1.80,
        boolean_level_ns=0.25,
        adder_level_ns=0.45,
        notes='Conservative placeholder profile for a routing-heavy unoptimized implementation.',
    ),
}


@dataclass(frozen=True)
class OutputReductionEstimate:
    implementation: str
    dynamic_terms_max: int
    reduction_levels: int
    critical_path_ns: float
    sample_time_ns: float
    estimated_fmax_mhz: float
    latency_cycles: int
    initiation_interval: int
    throughput_samples_per_sec: float


@dataclass(frozen=True)
class FpgaEstimateReport:
    target_fpga: str
    profile: TimingProfile
    pruning_stage: str
    reindexed: bool
    input_dim: int
    bop_count: int
    num_classes: int
    count_width: int
    max_boolean_depth: int
    per_class_dynamic_terms: tuple[int, ...]
    per_class_const_bias: tuple[int, ...]
    current_emitter: OutputReductionEstimate
    recommended_balanced_tree: OutputReductionEstimate
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'target_fpga': self.target_fpga,
            'profile': asdict(self.profile),
            'pruning_stage': self.pruning_stage,
            'reindexed': self.reindexed,
            'input_dim': self.input_dim,
            'bop_count': self.bop_count,
            'num_classes': self.num_classes,
            'count_width': self.count_width,
            'max_boolean_depth': self.max_boolean_depth,
            'per_class_dynamic_terms': list(self.per_class_dynamic_terms),
            'per_class_const_bias': list(self.per_class_const_bias),
            'current_emitter': asdict(self.current_emitter),
            'recommended_balanced_tree': asdict(self.recommended_balanced_tree),
            'metadata': self.metadata,
        }


def _source_dynamic_and_bias(source: BoolSource) -> tuple[int, int]:
    if source.signal_id is not None:
        return 1, 0
    assert source.const_value is not None
    return 0, int(source.const_value)


def _input_depth(source: GateInput, input_dim: int, node_depths: dict[int, int]) -> int:
    if source.const_value is not None:
        return 0
    assert source.signal_id is not None
    if source.signal_id < input_dim:
        return 0
    return node_depths[source.signal_id]


def _compute_node_depths(pruned: NaivePrunedCircuit) -> dict[int, int]:
    node_depths: dict[int, int] = {}
    for node in pruned.nodes:
        depth_a = _input_depth(node.input_a, pruned.input_dim, node_depths)
        depth_b = _input_depth(node.input_b, pruned.input_dim, node_depths)
        node_depths[node.node_id] = 1 + max(depth_a, depth_b)
    return node_depths


def _reduction_levels(term_count: int, *, implementation: str) -> int:
    if term_count <= 1:
        return 0
    if implementation == 'linear_sum_chain':
        return term_count - 1
    if implementation == 'balanced_sum_tree':
        return math.ceil(math.log2(term_count))
    raise ValueError(f'Unsupported implementation: {implementation!r}')


def _estimate_output_impl(
    *,
    implementation: str,
    max_dynamic_terms: int,
    max_boolean_depth: int,
    profile: TimingProfile,
) -> OutputReductionEstimate:
    reduction_levels = _reduction_levels(max_dynamic_terms, implementation=implementation)
    critical_path_ns = (
        profile.register_overhead_ns
        + max_boolean_depth * profile.boolean_level_ns
        + reduction_levels * profile.adder_level_ns
    )
    estimated_fmax_mhz = 1000.0 / critical_path_ns
    throughput_samples_per_sec = estimated_fmax_mhz * 1_000_000.0
    sample_time_ns = 1_000.0 / estimated_fmax_mhz
    return OutputReductionEstimate(
        implementation=implementation,
        dynamic_terms_max=max_dynamic_terms,
        reduction_levels=reduction_levels,
        critical_path_ns=critical_path_ns,
        sample_time_ns=sample_time_ns,
        estimated_fmax_mhz=estimated_fmax_mhz,
        latency_cycles=1,
        initiation_interval=1,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )


def estimate_fpga_from_pruned_circuit(
    pruned: NaivePrunedCircuit,
    *,
    target_fpga: str = 'AMD/Xilinx ZCU104 (heuristic only; no synthesis)',
    profile_name: str = 'nominal',
) -> FpgaEstimateReport:
    profile = DEFAULT_TIMING_PROFILES[profile_name]
    node_depths = _compute_node_depths(pruned)

    per_class_dynamic_terms: list[int] = []
    per_class_const_bias: list[int] = []
    max_boolean_depth = 0
    for class_sources in pruned.class_groups:
        dynamic_terms = 0
        const_bias = 0
        for source in class_sources:
            d, bias = _source_dynamic_and_bias(source)
            dynamic_terms += d
            const_bias += bias
            if source.signal_id is not None:
                if source.signal_id < pruned.input_dim:
                    signal_depth = 0
                else:
                    signal_depth = node_depths[source.signal_id]
                max_boolean_depth = max(max_boolean_depth, signal_depth)
        per_class_dynamic_terms.append(dynamic_terms)
        per_class_const_bias.append(const_bias)

    max_dynamic_terms = max(per_class_dynamic_terms, default=0)
    count_width = max(1, (len(pruned.class_groups[0]) + 1).bit_length()) if pruned.class_groups else 1

    current_emitter = _estimate_output_impl(
        implementation='linear_sum_chain',
        max_dynamic_terms=max_dynamic_terms,
        max_boolean_depth=max_boolean_depth,
        profile=profile,
    )
    recommended_balanced_tree = _estimate_output_impl(
        implementation='balanced_sum_tree',
        max_dynamic_terms=max_dynamic_terms,
        max_boolean_depth=max_boolean_depth,
        profile=profile,
    )

    metadata = dict(pruned.metadata)
    metadata.update(
        {
            'estimation_method': 'heuristic_structural_depth',
            'timing_profile': profile.name,
            'timing_profile_notes': profile.notes,
            'warning': 'This is a CPU-only structural estimate, not a synthesis/place-and-route timing report.',
        }
    )

    return FpgaEstimateReport(
        target_fpga=target_fpga,
        profile=profile,
        pruning_stage=str(pruned.metadata.get('pruning_stage', 'unknown')),
        reindexed=bool(pruned.metadata.get('reindexed', False)),
        input_dim=pruned.input_dim,
        bop_count=pruned.bop_count,
        num_classes=pruned.num_classes,
        count_width=count_width,
        max_boolean_depth=max_boolean_depth,
        per_class_dynamic_terms=tuple(per_class_dynamic_terms),
        per_class_const_bias=tuple(per_class_const_bias),
        current_emitter=current_emitter,
        recommended_balanced_tree=recommended_balanced_tree,
        metadata=metadata,
    )


def estimate_fpga_from_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    prune: str = 'naive',
    reindex: bool = True,
    target_fpga: str = 'AMD/Xilinx ZCU104 (heuristic only; no synthesis)',
    profile_name: str = 'nominal',
) -> FpgaEstimateReport:
    if prune != 'naive':
        raise ValueError("Only prune='naive' is supported by the current estimator.")

    circuit = extract_circuit_from_checkpoint(checkpoint_path, metadata_path)
    pruned = naive_prune_regular_circuit(circuit)
    if reindex:
        pruned = reindex_pruned_circuit(pruned)
    return estimate_fpga_from_pruned_circuit(pruned, target_fpga=target_fpga, profile_name=profile_name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Estimate FPGA timing/throughput heuristically from a compactlogic checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, default=None)
    parser.add_argument('--prune', choices=('naive',), default='naive')
    parser.add_argument('--reindex', action='store_true')
    parser.add_argument('--profile', choices=tuple(DEFAULT_TIMING_PROFILES), default='nominal')
    parser.add_argument('--target-fpga', type=str, default='AMD/Xilinx ZCU104 (heuristic only; no synthesis)')
    args = parser.parse_args()

    report = estimate_fpga_from_checkpoint(
        args.checkpoint,
        args.metadata,
        prune=args.prune,
        reindex=args.reindex,
        target_fpga=args.target_fpga,
        profile_name=args.profile,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == '__main__':
    main()
