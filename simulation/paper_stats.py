"""User-facing paper-report generation for CompactLogic checkpoints."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from compiler import extract_circuit_from_checkpoint, naive_prune_regular_circuit, reindex_pruned_circuit
from compiler.clocked import emit_naive_pruned_reindexed_clocked_bundle_from_checkpoint
from compiler.model_builder import build_compactlogic_model
from compiler.verilog import _default_pruned_module_name
from simulation.consistency import evaluate_extracted_circuit_torch, evaluate_naive_pruned_circuit
from simulation.context import dataset_name as _dataset_name
from simulation.context import resolve_context
from simulation.estimate_fpga import estimate_fpga_from_checkpoint
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


class _PostTransformDataset(Dataset):
    def __init__(self, base_ds: Dataset, post):
        self.base_ds = base_ds
        self.post = post

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        image, target = self.base_ds[i]
        return self.post(image), target


@torch.no_grad()
def _evaluate_original_model_batch(
    model: torch.nn.Module,
    *,
    batch_inputs: torch.Tensor,
    tau: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    logits = model(batch_inputs.to(torch.float32).to(device)).detach().cpu()
    class_counts = torch.round(logits * tau).to(torch.int64)
    return {
        'logits': logits,
        'class_counts': class_counts,
    }


@torch.no_grad()
def _evaluate_reference_discrete_batch(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    batch_inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return evaluate_extracted_circuit_torch(
        checkpoint_path,
        batch_inputs,
        metadata_path,
        device=device,
    )


def _make_mnist_test_loader(*, batch_size: int, download: bool, data_root: Path) -> DataLoader:
    dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=False,
        download=download,
        transform=torchvision.transforms.ToTensor(),
    )
    post = lambda t: (t >= 0.5).to(torch.bool)
    wrapped = _PostTransformDataset(dataset, post)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


def _make_cifar10_test_loader(dataset: str, *, batch_size: int, download: bool, data_root: Path) -> DataLoader:
    num_thd = int(dataset.split('-')[2])

    def binarize(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([(x > (i + 1) / (num_thd + 1)).to(torch.bool) for i in range(num_thd)], dim=0)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(binarize),
        ]
    )
    ds = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


def _make_test_loader(dataset: str, *, batch_size: int, download: bool) -> DataLoader:
    if dataset == 'mnist':
        return _make_mnist_test_loader(batch_size=batch_size, download=download, data_root=Path('./data-mnist'))
    if dataset.startswith('cifar-10-') and dataset.endswith('-thresholds'):
        return _make_cifar10_test_loader(dataset, batch_size=batch_size, download=download, data_root=Path('./data-cifar'))
    raise ValueError(
        'Full-test-set paper stats currently support only MNIST and thresholded CIFAR-10 datasets, '
        f'got {dataset!r}.'
    )


def evaluate_full_test_set_semantics(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    *,
    config_path: str | Path | None = None,
    batch_size: int = 256,
    download: bool = True,
    device: str | None = None,
    mismatch_cap: int = 20,
) -> SemanticMatchStats:
    """Evaluate compiled CompactLogic circuit semantics against the checkpoint reference on the full test set."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path)

    from simulation.context import load_json, load_yaml

    metadata = load_json(metadata_path)
    config = load_yaml(Path(config_path)) if config_path is not None and Path(config_path).exists() else {}
    dataset = _dataset_name(metadata, config)
    test_loader = _make_test_loader(dataset, batch_size=batch_size, download=download)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = metadata['args']
    tau = float(args['tau'])

    resolved_device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    original_model: torch.nn.Module | None = None
    reference_mode = 'extracted_discrete_circuit'
    if resolved_device.type == 'cuda':
        candidate_model = build_compactlogic_model(args)
        try:
            candidate_model.load_state_dict(checkpoint['model_state'], strict=True)
        except RuntimeError as exc:
            warnings.warn(
                'Falling back to extracted discrete circuit semantics because the original model '
                f'could not be reconstructed from {checkpoint_path}: {exc}',
                stacklevel=2,
            )
        else:
            original_model = candidate_model.to(resolved_device)
            original_model.eval()
            reference_mode = 'original_compactlogic_model'

    circuit = extract_circuit_from_checkpoint(checkpoint_path, metadata_path)
    pruned = reindex_pruned_circuit(naive_prune_regular_circuit(circuit))

    if original_model is not None:
        reference_batch_fn = lambda batch_inputs: _evaluate_original_model_batch(
            original_model,
            batch_inputs=batch_inputs,
            tau=tau,
            device=resolved_device,
        )
    else:
        reference_batch_fn = lambda batch_inputs: _evaluate_reference_discrete_batch(
            checkpoint_path,
            metadata_path,
            batch_inputs=batch_inputs,
            device=resolved_device,
        )

    stats = evaluate_semantic_match(
        dataset=dataset,
        reference_mode=reference_mode,
        data_loader=test_loader,
        compiled_batch_fn=lambda batch_inputs: evaluate_naive_pruned_circuit(pruned, batch_inputs, device=resolved_device),
        reference_batch_fn=reference_batch_fn,
        input_transform=lambda batch_inputs: batch_inputs.to(torch.bool),
        label_transform=lambda batch_labels: batch_labels.to(torch.int64),
        mismatch_cap=mismatch_cap,
    )
    validate_full_test_semantic_scope(stats, expected_num_samples=expected_test_samples(test_loader))
    return stats


def generate_paper_compiler_stats(
    checkpoint_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    eval_batch_size: int = 256,
    download_dataset: bool = True,
    device: str | None = None,
    timing_profile: str = 'nominal',
) -> PaperCompilerStats:
    """Generate a compiler-style paper report for a supported CompactLogic checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint path not found: {checkpoint_path}')

    resolved_metadata_path, metadata, resolved_config_path, config = resolve_context(
        checkpoint_path,
        metadata_path,
        config_path,
    )
    dataset = _dataset_name(metadata, config)
    if metadata.get('args', {}).get('struct') not in {None, 'regular', 'conv_spm'}:
        raise ValueError('Paper compiler stats currently support only regular and conv_spm models.')
    if dataset != 'mnist' and not (dataset.startswith('cifar-10-') and dataset.endswith('-thresholds')):
        raise ValueError(
            'Paper compiler stats currently support only MNIST and thresholded CIFAR-10 datasets, '
            f'got {dataset!r}.'
        )

    resolved_out_dir = Path(out_dir) if out_dir is not None else checkpoint_path.parent

    circuit = extract_circuit_from_checkpoint(checkpoint_path, resolved_metadata_path)
    pruned = reindex_pruned_circuit(naive_prune_regular_circuit(circuit))
    inner_module_name = _default_pruned_module_name(pruned.metadata, 'balanced')
    module_name = f'{inner_module_name}_clocked'
    return build_and_write_paper_compiler_report(
        checkpoint_path=checkpoint_path,
        metadata_path=resolved_metadata_path,
        config_path=resolved_config_path,
        out_dir=resolved_out_dir,
        raw_gate_count=circuit.node_count,
        pruned=pruned,
        module_name=module_name,
        emit_verilog=lambda target_path: target_path.write_text(
            emit_naive_pruned_reindexed_clocked_bundle_from_checkpoint(
                checkpoint_path,
                resolved_metadata_path,
                reduction_style='balanced',
            ),
            encoding='utf-8',
        ),
        evaluate_semantics=lambda: evaluate_full_test_set_semantics(
            checkpoint_path,
            resolved_metadata_path,
            config_path=resolved_config_path,
            batch_size=eval_batch_size,
            download=download_dataset,
            device=device,
        ),
        estimate_performance=lambda: estimate_fpga_from_checkpoint(
            checkpoint_path,
            resolved_metadata_path,
            prune='naive',
            reindex=True,
            profile_name=timing_profile,
        ),
        report_metadata={
            'compiled_backend': 'naive_pruned_reindexed_balanced_clocked',
            'method_name': 'compactlogic',
            'dataset': dataset,
            'timing_profile': timing_profile,
            'reference_description': 'the compactlogic checkpoint semantics for this model',
            'reference_fallback_note': (
                'When CUDA is available, the reference is the original compactlogic model; '
                'otherwise the report falls back to the extracted discrete checkpoint semantics.'
            ),
            'warning': 'Performance numbers are heuristic unless external synthesis reports are attached.',
        },
        json_filename='compiler_paper_stats.json',
        markdown_filename='compiler_paper_stats.md',
    )


def _prompt_path(prompt: str) -> str:
    value = input(prompt).strip()
    if value == '':
        raise ValueError('Expected a non-empty path.')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate paper-level compiler stats for a checkpoint and write them into the checkpoint directory.')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Path to ckpt_best_*.pt. If omitted, the script prompts for it.')
    parser.add_argument('--metadata', type=Path, default=None, help='Optional path to meta_data.json. Defaults to the sibling file next to the checkpoint.')
    parser.add_argument('--config', type=Path, default=None, help='Optional config YAML path. Defaults to metadata["args"]["config"] when available.')
    parser.add_argument('--out-dir', type=Path, default=None, help='Optional output directory. Defaults to the checkpoint directory.')
    parser.add_argument('--eval-batch-size', type=int, default=256, help='Batch size for full-test-set semantic evaluation.')
    parser.add_argument('--device', type=str, default=None, help='Optional device override for the original model, e.g. cpu or cuda.')
    parser.add_argument('--timing-profile', choices=('optimistic', 'nominal', 'pessimistic'), default='nominal')
    parser.add_argument('--no-download-dataset', action='store_true', help='Do not download the dataset if it is missing locally.')
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint is not None else Path(_prompt_path('Checkpoint path: '))

    report = generate_paper_compiler_stats(
        checkpoint,
        metadata_path=args.metadata,
        config_path=args.config,
        out_dir=args.out_dir,
        eval_batch_size=args.eval_batch_size,
        download_dataset=not args.no_download_dataset,
        device=args.device,
        timing_profile=args.timing_profile,
    )
    print(json.dumps(report.to_dict(), indent=2))
    target_dir = Path(args.out_dir) if args.out_dir is not None else checkpoint.parent
    print(f'Wrote {target_dir / "compiler_paper_stats.json"}')
    print(f'Wrote {target_dir / "compiler_paper_stats.md"}')


if __name__ == '__main__':
    main()
