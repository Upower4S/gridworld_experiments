"""Small IO helpers for conceptual belief space scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_parquet_checked(
    path: str | Path,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a parquet file and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"{path} is missing required columns: {missing_text}")
    return df


def write_json(path: str | Path, obj: object) -> Path:
    """Write JSON with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path


def read_json(path: str | Path) -> object:
    """Read a JSON file."""
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def save_markdown(path: str | Path, text: str) -> Path:
    """Write markdown text, ensuring a trailing newline."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n")
    return path


def find_existing_file(candidates: Iterable[str | Path]) -> Path | None:
    """Return the first existing file from an ordered candidate list."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    return None
