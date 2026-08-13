"""Action belief vector transforms."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS, EPS


def action_probability_columns() -> list[str]:
    """Return canonical action probability column names."""
    return [f"q_{action}" for action in ACTIONS]


def stable_softmax(
    logits: Sequence[float] | np.ndarray,
    *,
    axis: int = -1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute a numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = np.asarray(logits, dtype=float) / temperature
    max_x = np.nanmax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    denom = np.nansum(exp_x, axis=axis, keepdims=True)
    return exp_x / np.clip(denom, EPS, None)


def _as_probability_matrix(q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(q, pd.DataFrame):
        cols = action_probability_columns()
        missing = [col for col in cols if col not in q.columns]
        if missing:
            raise ValueError(f"missing action probability columns: {missing}")
        arr = q[cols].to_numpy(dtype=float)
    else:
        arr = np.asarray(q, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != len(ACTIONS):
        raise ValueError(f"expected last dimension of size {len(ACTIONS)}, got {arr.shape}")
    return arr


def normalize_action_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize q_UP/q_DOWN/q_LEFT/q_RIGHT columns row-wise."""
    cols = action_probability_columns()
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing action probability columns: {missing}")
    out = df.copy()
    q = out[cols].to_numpy(dtype=float)
    q = np.clip(q, 0.0, None)
    denom = q.sum(axis=1, keepdims=True)
    valid = denom[:, 0] > 0
    q_norm = np.full_like(q, 1.0 / len(ACTIONS), dtype=float)
    q_norm[valid] = q[valid] / denom[valid]
    out.loc[:, cols] = q_norm
    return out


def log_q_transform(q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], eps: float = EPS) -> np.ndarray:
    """Return log probabilities after clipping."""
    arr = _as_probability_matrix(q)
    return np.log(np.clip(arr, eps, None))


def clr_transform(q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], eps: float = EPS) -> np.ndarray:
    """Return centered log-ratio transformed action beliefs."""
    log_q = log_q_transform(q, eps=eps)
    return log_q - log_q.mean(axis=-1, keepdims=True)


def sqrt_q_transform(q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return square-root transformed probabilities."""
    arr = _as_probability_matrix(q)
    return np.sqrt(np.clip(arr, 0.0, None))


def compute_entropy(q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], eps: float = EPS) -> np.ndarray:
    """Compute categorical entropy for action probabilities."""
    arr = _as_probability_matrix(q)
    arr = np.clip(arr, eps, None)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return -(arr * np.log(arr)).sum(axis=1)


def compute_commitment_to_action(
    q: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    actions: str | Sequence[str],
) -> np.ndarray:
    """Return q_action for each row.

    `actions` may be a single action label or one label per row.
    """
    arr = _as_probability_matrix(q)
    if isinstance(actions, str):
        labels = [actions] * len(arr)
    else:
        labels = list(actions)
    if len(labels) != len(arr):
        raise ValueError(f"expected {len(arr)} action labels, got {len(labels)}")
    action_to_col = {action: i for i, action in enumerate(ACTIONS)}
    out = np.empty(len(arr), dtype=float)
    for i, action in enumerate(labels):
        if action not in action_to_col:
            raise ValueError(f"unknown action label: {action}")
        out[i] = arr[i, action_to_col[action]]
    return out
