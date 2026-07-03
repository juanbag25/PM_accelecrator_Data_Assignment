"""Configuration loader.

The single source of truth for project conventions is ``config.yaml`` at the
repository root. Everything (target, horizon, seed, split quantiles, paths) is
read from there — never hard-code these values elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the ``src`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the YAML configuration as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str | Path) -> Path:
    """Resolve a repo-relative path (as stored in config.yaml) to absolute."""
    return (REPO_ROOT / rel_path).resolve()
