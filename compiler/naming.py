from __future__ import annotations


def module_name_from_metadata(metadata: dict[str, object], suffix: str = "") -> str:
    dataset = str(metadata.get("dataset", "unknown")).replace("-", "_")
    struct = str(metadata.get("struct", "regular")).replace("-", "_")
    experiment_id = str(metadata.get("experiment_id", "model"))
    base = f"compactlogic_{struct}_{dataset}_{experiment_id}"
    return f"{base}_{suffix}" if suffix else base


def default_pruned_module_name(metadata: dict[str, object], reduction_style: str) -> str:
    suffix = "naive_pruned_balanced" if reduction_style == "balanced" else "naive_pruned"
    return module_name_from_metadata(metadata, suffix=suffix)
