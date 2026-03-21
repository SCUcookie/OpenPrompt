from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from openprompt_rs.utils.io import load_yaml


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = load_yaml(path)
    base_entries = config.pop("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]

    merged: dict[str, Any] = {}
    for entry in base_entries:
        merged = deep_merge_dict(merged, load_config(path.parent / entry))
    merged = deep_merge_dict(merged, config)
    return merged

