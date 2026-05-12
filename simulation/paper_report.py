"""Shared report dataclasses and Markdown rendering for compiler-style summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from compiler.prune import NaivePrunedCircuit
from simulation.estimate_fpga import FpgaEstimateReport


def _metadata_string(metadata: dict[str, object], key: str, default: str) -> str:
    return str(metadata.get(key, default))


@dataclass(frozen=True)
class SemanticMatchStats:
    dataset: str
    split: str
    reference_mode: str
    num_samples: int
    original_accuracy: float
    compiled_accuracy: float
    exact_argmax_match_rate: float
    exact_class_count_match_rate: float
    argmax_mismatch_count: int
    class_count_mismatch_count: int
    max_abs_count_diff: int
    mean_abs_count_diff: float
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    first_argmax_mismatch_indices: tuple[int, ...]
    first_class_count_mismatch_indices: tuple[int, ...]


@dataclass(frozen=True)
class LogicalGateCountStats:
    raw_logical_gates: int
    pruned_logical_gates: int
    final_compiled_boolean_core_gates: int
    logical_gate_reduction_ratio: float


@dataclass(frozen=True)
class CompiledArtifactStats:
    compiled_verilog_path: str
    module_name: str
    reduction_style: str
    clocked_wrapper: bool
    compiled_verilog_size_mb: float
    compiled_verilog_line_count: int


@dataclass(frozen=True)
class PerformanceEstimateStats:
    target_fpga: str
    timing_profile_name: str
    estimated_fmax_mhz: float
    estimated_sample_time_ns: float
    estimated_throughput_samples_per_sec: float
    latency_cycles: int
    initiation_interval: int
    count_width: int
    max_boolean_depth: int
    per_class_dynamic_terms_max: int


@dataclass(frozen=True)
class PaperCompilerStats:
    checkpoint_path: str
    metadata_path: str
    config_path: str | None
    semantic_match: SemanticMatchStats
    logical_gate_counts: LogicalGateCountStats
    compiled_artifact: CompiledArtifactStats
    performance_estimate: PerformanceEstimateStats
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_logical_gate_stats(raw_gate_count: int, pruned: NaivePrunedCircuit) -> LogicalGateCountStats:
    """Build raw/pruned logical-gate summary statistics."""
    return LogicalGateCountStats(
        raw_logical_gates=raw_gate_count,
        pruned_logical_gates=pruned.bop_count,
        final_compiled_boolean_core_gates=pruned.bop_count,
        logical_gate_reduction_ratio=pruned.bop_count / raw_gate_count,
    )


def build_artifact_stats(verilog_path: Path, module_name: str) -> CompiledArtifactStats:
    """Build emitted-Verilog artifact statistics for the report."""
    verilog_text = verilog_path.read_text(encoding='utf-8')
    return CompiledArtifactStats(
        compiled_verilog_path=str(verilog_path),
        module_name=module_name,
        reduction_style='balanced',
        clocked_wrapper=True,
        compiled_verilog_size_mb=verilog_path.stat().st_size / 1_000_000.0,
        compiled_verilog_line_count=verilog_text.count('\n') + (0 if verilog_text.endswith('\n') else 1),
    )


def build_performance_stats(report: FpgaEstimateReport) -> PerformanceEstimateStats:
    """Convert the FPGA-estimator output into the compact report schema."""
    preferred = report.recommended_balanced_tree
    return PerformanceEstimateStats(
        target_fpga=report.target_fpga,
        timing_profile_name=report.profile.name,
        estimated_fmax_mhz=preferred.estimated_fmax_mhz,
        estimated_sample_time_ns=preferred.sample_time_ns,
        estimated_throughput_samples_per_sec=preferred.throughput_samples_per_sec,
        latency_cycles=preferred.latency_cycles,
        initiation_interval=preferred.initiation_interval,
        count_width=report.count_width,
        max_boolean_depth=report.max_boolean_depth,
        per_class_dynamic_terms_max=max(report.per_class_dynamic_terms, default=0),
    )


def render_markdown(report: PaperCompilerStats) -> str:
    """Render a human-readable Markdown summary of a compiler report."""
    semantic = report.semantic_match
    gates = report.logical_gate_counts
    artifact = report.compiled_artifact
    perf = report.performance_estimate
    method_name = _metadata_string(report.metadata, 'method_name', 'compiled model')
    reference_description = _metadata_string(
        report.metadata,
        'reference_description',
        'the native checkpoint/runtime semantics used as the reference for this backend',
    )
    reference_fallback_note = _metadata_string(
        report.metadata,
        'reference_fallback_note',
        'The exact reference implementation is backend-specific and is recorded in the reference mode field above.',
    )
    return "\n".join(
        [
            '# Compiler Paper-Level Stats',
            '',
            f'- Checkpoint: `{report.checkpoint_path}`',
            f'- Metadata: `{report.metadata_path}`',
            f'- Config: `{report.config_path}`' if report.config_path is not None else '- Config: `None`',
            '',
            '## Semantic equivalence on full test set',
            '',
            f'- This section validates the compiled circuit semantics against {reference_description}.',
            f'- Method/backend: `{method_name}`',
            f'- Dataset: `{semantic.dataset}` ({semantic.split})',
            f'- Reference mode: `{semantic.reference_mode}`',
            f'- Samples: **{semantic.num_samples}**',
            f'- Original accuracy: **{semantic.original_accuracy:.6f}**',
            f'- Compiled accuracy: **{semantic.compiled_accuracy:.6f}**',
            f'- Exact argmax match rate: **{semantic.exact_argmax_match_rate:.6f}**',
            f'- Exact class-count match rate: **{semantic.exact_class_count_match_rate:.6f}**',
            f'- Argmax mismatches: **{semantic.argmax_mismatch_count}**',
            f'- Class-count mismatches: **{semantic.class_count_mismatch_count}**',
            f'- Max abs count diff: **{semantic.max_abs_count_diff}**',
            f'- Mean abs count diff: **{semantic.mean_abs_count_diff:.6f}**',
            f'- Max abs logit diff: **{semantic.max_abs_logit_diff:.6e}**',
            f'- Mean abs logit diff: **{semantic.mean_abs_logit_diff:.6e}**',
            '',
            '## Logical gate counts',
            '',
            f'- Raw logical gates: **{gates.raw_logical_gates}**',
            f'- Pruned logical gates: **{gates.pruned_logical_gates}**',
            f'- Final compiled Boolean-core logical gates: **{gates.final_compiled_boolean_core_gates}**',
            f'- Logical-gate reduction ratio: **{gates.logical_gate_reduction_ratio:.6f}**',
            '',
            '## Compiled artifact',
            '',
            f'- Module name: `{artifact.module_name}`',
            f'- Verilog path: `{artifact.compiled_verilog_path}`',
            f'- Verilog size (MB): **{artifact.compiled_verilog_size_mb:.6f}**',
            f'- Verilog line count: **{artifact.compiled_verilog_line_count}**',
            '',
            '## FPGA-oriented estimate',
            '',
            f'- Target FPGA: `{perf.target_fpga}`',
            f'- Timing profile: `{perf.timing_profile_name}`',
            f'- Estimated Fmax (MHz): **{perf.estimated_fmax_mhz:.6f}**',
            f'- Estimated sample time (ns): **{perf.estimated_sample_time_ns:.6f}**',
            f'- Estimated throughput (samples/s): **{perf.estimated_throughput_samples_per_sec:.6f}**',
            f'- Latency cycles: **{perf.latency_cycles}**',
            f'- Initiation interval: **{perf.initiation_interval}**',
            f'- Count width: **{perf.count_width}**',
            f'- Max Boolean depth: **{perf.max_boolean_depth}**',
            f'- Max per-class dynamic terms: **{perf.per_class_dynamic_terms_max}**',
            '',
            '## Notes',
            '',
            '- The semantic comparison evaluates the exact compiled Boolean-core semantics on the full test set.',
            f'- {reference_fallback_note}',
            '- The emitted balanced clocked Verilog is generated from that compiled core, but the full test set is not executed through Verilog simulation in this report.',
            '- The timing numbers are heuristic CPU-only FPGA estimates unless replaced by synthesis reports.',
            '- Verilog file size is a compiler artifact-size metric, not an FPGA LUT/BRAM area measurement.',
            '',
        ]
    )
