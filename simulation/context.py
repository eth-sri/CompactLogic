from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_metadata_path(checkpoint_path: Path, metadata_path: str | Path | None) -> Path:
    if metadata_path is not None:
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f'Metadata path not found: {path}')
        return path
    sibling = checkpoint_path.with_name('meta_data.json')
    if sibling.exists():
        return sibling
    raise FileNotFoundError('Could not infer sibling meta_data.json; pass --metadata explicitly.')


def resolve_context(
    checkpoint_path: Path,
    metadata_path: str | Path | None,
    config_path: str | Path | None,
) -> tuple[Path, dict[str, Any], Path | None, dict[str, Any]]:
    resolved_metadata_path = resolve_metadata_path(checkpoint_path, metadata_path)
    metadata = load_json(resolved_metadata_path)

    resolved_config_path: Path | None = None
    config: dict[str, Any] = {}
    config_candidate = config_path or metadata.get('args', {}).get('config')
    if config_candidate is not None:
        candidate_path = Path(config_candidate)
        if candidate_path.exists():
            resolved_config_path = candidate_path
            config = load_yaml(candidate_path)
    return resolved_metadata_path, metadata, resolved_config_path, config


def dataset_name(metadata: dict[str, Any], config: dict[str, Any]) -> str:
    dataset = metadata.get('args', {}).get('dataset') or config.get('dataset')
    if not isinstance(dataset, str) or dataset == '':
        raise ValueError('Could not determine dataset from metadata/config.')
    return dataset
