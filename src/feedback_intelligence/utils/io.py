"""I/O utilities for artifacts and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_joblib(path: Path, payload: Any) -> None:
    """Write a Python object artifact with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_joblib(path: Path) -> Any:
    """Load a Python object artifact saved with joblib."""
    return joblib.load(path)
