"""Shared assembly/writer utilities for compiler-style report generation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from compiler.prune import NaivePrunedCircuit
from simulation.estimate_fpga import FpgaEstimateReport
from simulation.paper_report import (
    PaperCompilerStats,
    SemanticMatchStats,
    build_artifact_stats,
    build_logical_gate_stats,
    build_performance_stats,
    render_markdown,
)


def _resolved_output_dir(out_dir: str | Path) -> Path:
    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    return resolved_out_dir


def write_paper_compiler_report(
    report: PaperCompilerStats,
    *,
    out_dir: str | Path,
    json_filename: str,
    markdown_filename: str,
) -> PaperCompilerStats:
    """Write a populated compiler report as JSON plus Markdown."""
    resolved_out_dir = _resolved_output_dir(out_dir)
    (resolved_out_dir / json_filename).write_text(json.dumps(report.to_dict(), indent=2), encoding='utf-8')
    (resolved_out_dir / markdown_filename).write_text(render_markdown(report), encoding='utf-8')
    return report


def build_and_write_paper_compiler_report(
    *,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    config_path: str | Path | None,
    out_dir: str | Path,
    raw_gate_count: int,
    pruned: NaivePrunedCircuit,
    module_name: str,
    emit_verilog: Callable[[Path], None],
    evaluate_semantics: Callable[[], SemanticMatchStats],
    estimate_performance: Callable[[], FpgaEstimateReport],
    report_metadata: dict[str, object],
    json_filename: str,
    markdown_filename: str,
) -> PaperCompilerStats:
    """Emit Verilog, run semantic/performance evaluation, and write the final report."""
    resolved_out_dir = _resolved_output_dir(out_dir)

    compiled_verilog_path = resolved_out_dir / f'{module_name}.v'
    emit_start = time.perf_counter()
    emit_verilog(compiled_verilog_path)
    verilog_emit_time_seconds = time.perf_counter() - emit_start

    semantic = evaluate_semantics()
    fpga_report = estimate_performance()

    metadata = dict(report_metadata)
    metadata.setdefault('semantic_validation_scope', semantic.split)
    metadata.setdefault('reference_mode', semantic.reference_mode)
    metadata['verilog_emit_time_seconds'] = verilog_emit_time_seconds

    report = PaperCompilerStats(
        checkpoint_path=str(checkpoint_path),
        metadata_path=str(metadata_path),
        config_path=None if config_path is None else str(config_path),
        semantic_match=semantic,
        logical_gate_counts=build_logical_gate_stats(raw_gate_count, pruned),
        compiled_artifact=build_artifact_stats(compiled_verilog_path, module_name),
        performance_estimate=build_performance_stats(fpga_report),
        metadata=metadata,
    )

    return write_paper_compiler_report(
        report,
        out_dir=resolved_out_dir,
        json_filename=json_filename,
        markdown_filename=markdown_filename,
    )
