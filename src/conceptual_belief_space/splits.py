"""Grouped split helpers for leakage-resistant analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import ensure_dir


def make_decision_id(df: pd.DataFrame) -> pd.Series:
    """Return canonical decision ids, constructing them when needed."""
    if "decision_id" in df.columns:
        decision_id = df["decision_id"].astype("string")
        if decision_id.notna().all():
            return decision_id.astype(str)
    if {"trajectory_id", "step_id"}.issubset(df.columns):
        return df["trajectory_id"].astype(str) + "::" + df["step_id"].astype(str)
    raise ValueError("cannot construct decision_id without decision_id or trajectory_id + step_id")


def grouped_train_test_splits(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Create deterministic K-fold splits where groups never cross folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if group_col not in df.columns:
        raise ValueError(f"missing group column: {group_col}")
    groups = pd.Series(df[group_col].dropna().unique()).to_numpy()
    if len(groups) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds number of groups={len(groups)}")
    rng = np.random.default_rng(seed)
    groups = groups.copy()
    rng.shuffle(groups)

    positions = np.arange(len(df))
    folds = np.array_split(groups, n_splits)
    splits: list[dict[str, list[int]]] = []
    for fold_id, test_groups in enumerate(folds):
        test_group_set = set(test_groups.tolist())
        test_mask = df[group_col].isin(test_group_set).to_numpy()
        train_idx = positions[~test_mask].astype(int).tolist()
        test_idx = positions[test_mask].astype(int).tolist()
        splits.append(
            {
                "fold": int(fold_id),
                "group_col": group_col,
                "train_indices": train_idx,
                "test_indices": test_idx,
                "test_groups": [str(group) for group in test_groups.tolist()],
            }
        )
    return splits


def _json_ready(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    return obj


def save_splits_json(path: str | Path, splits: Iterable[object]) -> Path:
    """Save split definitions as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_ready(list(splits)), indent=2, sort_keys=True) + "\n")
    return path
