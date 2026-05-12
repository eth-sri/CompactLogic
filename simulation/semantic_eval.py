"""Shared semantic-evaluation helpers for compiler/report pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.utils.data import DataLoader

from simulation.paper_report import SemanticMatchStats


BatchEvalFn = Callable[[torch.Tensor], dict[str, torch.Tensor]]
TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def expected_test_samples(data_loader: DataLoader) -> int | None:
    """Return the dataset size when the loader exposes one, else ``None``."""
    dataset = getattr(data_loader, 'dataset', None)
    try:
        return None if dataset is None else int(len(dataset))
    except TypeError:
        return None


def validate_full_test_semantic_scope(
    stats: SemanticMatchStats,
    *,
    expected_num_samples: int | None = None,
) -> None:
    """Reject semantic-validation results that should not be treated as full-test reports."""
    if stats.split != 'test':
        raise ValueError(f'Expected full-test semantic validation (`split=test`), got {stats.split!r}.')
    if expected_num_samples is not None and stats.num_samples != expected_num_samples:
        raise ValueError(
            f'Expected semantic validation over {expected_num_samples} samples, got {stats.num_samples}. '
            'Refusing to treat this as a full-test result.'
        )


@torch.no_grad()
def evaluate_semantic_match(
    *,
    dataset: str,
    reference_mode: str,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    compiled_batch_fn: BatchEvalFn,
    reference_batch_fn: BatchEvalFn,
    input_transform: TensorTransform | None = None,
    label_transform: TensorTransform | None = None,
    mismatch_cap: int = 20,
    max_batches: int | None = None,
) -> SemanticMatchStats:
    """Compare compiled outputs against a reference backend over a dataset iterator."""
    num_samples = 0
    original_correct = 0
    compiled_correct = 0
    argmax_match = 0
    count_match = 0
    total_count_abs_diff = 0.0
    total_logit_abs_diff = 0.0
    total_output_entries = 0
    max_count_abs_diff = 0
    max_logit_abs_diff = 0.0
    argmax_mismatch_indices: list[int] = []
    class_count_mismatch_indices: list[int] = []
    sample_offset = 0
    processed_batches = 0

    for batch_inputs, batch_labels in data_loader:
        if max_batches is not None and processed_batches >= max_batches:
            break

        if input_transform is not None:
            batch_inputs = input_transform(batch_inputs)
        if label_transform is not None:
            batch_labels = label_transform(batch_labels)

        batch_size_actual = batch_inputs.shape[0]
        compiled = compiled_batch_fn(batch_inputs)
        reference = reference_batch_fn(batch_inputs)

        compiled_counts = compiled['class_counts']
        reference_counts = reference['class_counts']
        compiled_logits = compiled['logits']
        reference_logits = reference['logits']

        compiled_preds = compiled_counts.argmax(dim=1)
        reference_preds = reference_counts.argmax(dim=1)

        exact_count_mask = (compiled_counts == reference_counts).all(dim=1)
        exact_argmax_mask = compiled_preds == reference_preds

        original_correct += int((reference_preds == batch_labels).sum().item())
        compiled_correct += int((compiled_preds == batch_labels).sum().item())
        count_match += int(exact_count_mask.sum().item())
        argmax_match += int(exact_argmax_mask.sum().item())
        num_samples += batch_size_actual
        processed_batches += 1

        count_abs_diff = (compiled_counts - reference_counts).abs()
        logit_abs_diff = (compiled_logits - reference_logits).abs()
        max_count_abs_diff = max(max_count_abs_diff, int(count_abs_diff.max().item()))
        max_logit_abs_diff = max(max_logit_abs_diff, float(logit_abs_diff.max().item()))
        total_count_abs_diff += float(count_abs_diff.sum().item())
        total_logit_abs_diff += float(logit_abs_diff.sum().item())
        total_output_entries += int(count_abs_diff.numel())

        if len(argmax_mismatch_indices) < mismatch_cap:
            local = (~exact_argmax_mask).nonzero(as_tuple=False).flatten().tolist()
            remaining = mismatch_cap - len(argmax_mismatch_indices)
            argmax_mismatch_indices.extend(sample_offset + idx for idx in local[:remaining])
        if len(class_count_mismatch_indices) < mismatch_cap:
            local = (~exact_count_mask).nonzero(as_tuple=False).flatten().tolist()
            remaining = mismatch_cap - len(class_count_mismatch_indices)
            class_count_mismatch_indices.extend(sample_offset + idx for idx in local[:remaining])
        sample_offset += batch_size_actual

    split_name = 'test' if max_batches is None else f'test_first_{processed_batches}_batches'
    return SemanticMatchStats(
        dataset=dataset,
        split=split_name,
        reference_mode=reference_mode,
        num_samples=num_samples,
        original_accuracy=original_correct / max(1, num_samples),
        compiled_accuracy=compiled_correct / max(1, num_samples),
        exact_argmax_match_rate=argmax_match / max(1, num_samples),
        exact_class_count_match_rate=count_match / max(1, num_samples),
        argmax_mismatch_count=num_samples - argmax_match,
        class_count_mismatch_count=num_samples - count_match,
        max_abs_count_diff=max_count_abs_diff,
        mean_abs_count_diff=total_count_abs_diff / max(1, total_output_entries),
        max_abs_logit_diff=max_logit_abs_diff,
        mean_abs_logit_diff=total_logit_abs_diff / max(1, total_output_entries),
        first_argmax_mismatch_indices=tuple(argmax_mismatch_indices),
        first_class_count_mismatch_indices=tuple(class_count_mismatch_indices),
    )
