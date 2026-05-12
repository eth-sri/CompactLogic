"""Compiler-style paper-report generation for supported vendored ConvLogic checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compiler.clocked import emit_clocked_wrapper_verilog
from compiler.prune import naive_prune_circuit, reindex_pruned_circuit
from compiler.verilog import emit_naive_pruned_verilog
from simulation.estimate_fpga import estimate_fpga_from_pruned_circuit
from simulation.paper_report import (
    SemanticMatchStats,
    PaperCompilerStats,
)
from simulation.report_pipeline import build_and_write_paper_compiler_report
from simulation.semantic_eval import (
    evaluate_semantic_match,
    expected_test_samples,
    validate_full_test_semantic_scope,
)

from .config import ConvLogicConfig, infer_config_from_checkpoint
from .evaluate import _binarize_loaded_batch, evaluate_native_runtime, load_native_runtime, make_test_loader
from .extract import extract_circuit_from_checkpoint


def _resolved_torch_device(device: str | None) -> torch.device:
    return torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))


def _default_module_name(config: ConvLogicConfig, reduction_style: str) -> str:
    suffix = 'naive_pruned_balanced' if reduction_style == 'balanced' else 'naive_pruned'
    dataset = config.dataset_name.replace('-', '_')
    return f'convlogic_{config.variant_name}_{dataset}_{suffix}'


def _validate_full_report_request(*, consistency_max_batches: int | None, allow_partial_report: bool) -> None:
    if consistency_max_batches is not None and not allow_partial_report:
        raise ValueError(
            'Standard ConvLogic compiler reports must validate compiled semantics on the full test set. '
            'Pass allow_partial_report=True only for debugging runs.'
        )


def evaluate_reference_accuracy(
    checkpoint_path: str | Path,
    *,
    batch_size: int = 256,
    download: bool = True,
    device: str | None = None,
) -> tuple[float, int]:
    """Measure full-test native ConvLogic accuracy for a supported checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    resolved_device = _resolved_torch_device(device)
    runtime = load_native_runtime(checkpoint_path, device=resolved_device)
    loader = make_test_loader(checkpoint_path, batch_size=batch_size, download=download)

    num_samples = 0
    correct = 0
    for batch_inputs, batch_labels in loader:
        batch_inputs = _binarize_loaded_batch(batch_inputs)
        batch_labels = batch_labels.to(torch.int64)
        outputs = evaluate_native_runtime(runtime, batch_inputs)
        preds = outputs['class_counts'].argmax(dim=1)
        correct += int((preds == batch_labels).sum().item())
        num_samples += int(batch_inputs.shape[0])
    return correct / num_samples, num_samples


def evaluate_compiled_consistency(
    checkpoint_path: str | Path,
    *,
    batch_size: int = 256,
    download: bool = True,
    device: str | None = None,
    mismatch_cap: int = 20,
    max_batches: int | None = None,
) -> SemanticMatchStats:
    """Compare compiled ConvLogic circuit semantics against the native ConvLogic runtime."""
    checkpoint_path = Path(checkpoint_path)
    config = infer_config_from_checkpoint(checkpoint_path)
    resolved_device = _resolved_torch_device(device)

    circuit = extract_circuit_from_checkpoint(checkpoint_path)
    pruned = reindex_pruned_circuit(naive_prune_circuit(circuit))
    runtime = load_native_runtime(checkpoint_path, device=resolved_device)
    loader = make_test_loader(checkpoint_path, batch_size=batch_size, download=download)

    from simulation.consistency import evaluate_naive_pruned_circuit

    stats = evaluate_semantic_match(
        dataset=config.dataset_name,
        reference_mode='native_structured_convlogic_checkpoint',
        data_loader=loader,
        compiled_batch_fn=lambda batch_inputs: evaluate_naive_pruned_circuit(pruned, batch_inputs, device=resolved_device),
        reference_batch_fn=lambda batch_inputs: evaluate_native_runtime(runtime, batch_inputs),
        input_transform=_binarize_loaded_batch,
        label_transform=lambda batch_labels: batch_labels.to(torch.int64),
        mismatch_cap=mismatch_cap,
        max_batches=max_batches,
    )
    if max_batches is None:
        validate_full_test_semantic_scope(stats, expected_num_samples=expected_test_samples(loader))
    return stats


def generate_convlogic_paper_stats(
    checkpoint_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    eval_batch_size: int = 256,
    download_dataset: bool = True,
    device: str | None = None,
    timing_profile: str = 'nominal',
    consistency_max_batches: int | None = None,
    allow_partial_report: bool = False,
) -> PaperCompilerStats:
    """Generate a compiler-style report for a supported ConvLogic checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    _validate_full_report_request(
        consistency_max_batches=consistency_max_batches,
        allow_partial_report=allow_partial_report,
    )
    config = infer_config_from_checkpoint(checkpoint_path)
    resolved_out_dir = Path(out_dir) if out_dir is not None else checkpoint_path.parent

    circuit = extract_circuit_from_checkpoint(checkpoint_path)
    pruned = reindex_pruned_circuit(naive_prune_circuit(circuit))
    inner_module_name = _default_module_name(config, 'balanced')
    wrapper_module_name = f'{inner_module_name}_clocked'
    full_reference_accuracy, reference_num_samples = evaluate_reference_accuracy(
        checkpoint_path,
        batch_size=eval_batch_size,
        download=download_dataset,
        device=device,
    )
    semantic = evaluate_compiled_consistency(
        checkpoint_path,
        batch_size=eval_batch_size,
        download=download_dataset,
        device=device,
        max_batches=consistency_max_batches,
    )
    return build_and_write_paper_compiler_report(
        checkpoint_path=checkpoint_path,
        metadata_path=checkpoint_path,
        config_path=None,
        out_dir=resolved_out_dir,
        raw_gate_count=circuit.node_count,
        pruned=pruned,
        module_name=wrapper_module_name,
        emit_verilog=lambda target_path: target_path.write_text(
            emit_naive_pruned_verilog(pruned, module_name=inner_module_name, reduction_style='balanced').rstrip()
            + '\n\n'
            + emit_clocked_wrapper_verilog(
                inner_module_name,
                input_width=pruned.input_dim,
                output_width=pruned.num_classes * max(1, (len(pruned.class_groups[0]) + 1).bit_length()),
                wrapper_module_name=wrapper_module_name,
            ),
            encoding='utf-8',
        ),
        evaluate_semantics=lambda: semantic,
        estimate_performance=lambda: estimate_fpga_from_pruned_circuit(pruned, profile_name=timing_profile),
        report_metadata={
            'compiled_backend': 'convlogic_naive_pruned_reindexed_balanced_clocked',
            'method_name': 'convlogic',
            'dataset': config.dataset_name,
            'variant_name': config.variant_name,
            'timing_profile': timing_profile,
            'reference_description': 'the native ConvLogic runtime for this checkpoint',
            'reference_fallback_note': 'The reference is the native structured ConvLogic runtime for the same checkpoint.',
            'full_test_reference_accuracy': full_reference_accuracy,
            'full_test_reference_num_samples': reference_num_samples,
            'consistency_max_batches': consistency_max_batches,
            'warning': 'Performance numbers are heuristic unless external synthesis reports are attached.',
        },
        json_filename='convlogic_compiler_paper_stats.json',
        markdown_filename='convlogic_compiler_paper_stats.md',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate compiler-style paper stats for a ConvLogic checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--timing-profile', choices=('optimistic', 'nominal', 'pessimistic'), default='nominal')
    parser.add_argument('--max-batches', type=int, default=None, help='Limit compiled consistency validation to the first N batches (debug only; requires --allow-partial-report).')
    parser.add_argument('--allow-partial-report', action='store_true', help='Allow writing a non-full-test ConvLogic report. Intended only for debugging.')
    parser.add_argument('--no-download-dataset', action='store_true')
    args = parser.parse_args()

    report = generate_convlogic_paper_stats(
        args.checkpoint,
        out_dir=args.out_dir,
        eval_batch_size=args.eval_batch_size,
        download_dataset=not args.no_download_dataset,
        device=args.device,
        timing_profile=args.timing_profile,
        consistency_max_batches=args.max_batches,
        allow_partial_report=args.allow_partial_report,
    )
    print(json.dumps(report.to_dict(), indent=2))
    target_dir = Path(args.out_dir) if args.out_dir is not None else args.checkpoint.parent
    print(f'Wrote {target_dir / "convlogic_compiler_paper_stats.json"}')
    print(f'Wrote {target_dir / "convlogic_compiler_paper_stats.md"}')


if __name__ == '__main__':
    main()
