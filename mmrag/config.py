from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the default config, optionally merged with another YAML file and a dict of overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
