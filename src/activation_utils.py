"""Shared utilities for activation-side gridworld commitment analyses.

This module intentionally does not extract model activations. It only provides
small, reusable helpers for downstream scripts/notebooks that consume already
extracted activation tensors and behavior-side probability targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DEFAULT_ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}


def make_decision_id(trajectory_id: object, step_id: object) -> str:
    """Create the canonical decision identifier used across tables."""
    return f"{trajectory_id}::{step_id}"


def stable_softmax(
    logits: Sequence[float] | np.ndarray,
    *,
    axis: int = -1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Numerically stable softmax over an array-like input."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = np.asarray(logits, dtype=float) / temperature
    max_x = np.nanmax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    denom = np.nansum(exp_x, axis=axis, keepdims=True)
    return exp_x / denom


def clr_transform(
    q_UP: Sequence[float] | np.ndarray | float,
    q_DOWN: Sequence[float] | np.ndarray | float,
    q_LEFT: Sequence[float] | np.ndarray | float,
    q_RIGHT: Sequence[float] | np.ndarray | float,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Centered log-ratio transform for the four action probabilities."""
    q = np.stack([q_UP, q_DOWN, q_LEFT, q_RIGHT], axis=-1).astype(float)
    q = np.clip(q, eps, None)
    log_q = np.log(q)
    return log_q - log_q.mean(axis=-1, keepdims=True)


def standardize_matrix(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize a 2D matrix and return standardized values plus parameters."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {x.shape}")
    if mean is None:
        mean = np.nanmean(x, axis=0)
    if scale is None:
        scale = np.nanstd(x, axis=0)
    scale = np.where(np.abs(scale) < eps, 1.0, scale)
    return (x - mean) / scale, mean, scale


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_parquet(path: str | Path, *, required: bool = False) -> pd.DataFrame | None:
    """Read parquet if it exists; return None for missing optional files."""
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        print(f"WARNING: optional parquet not found: {path}")
        return None
    return pd.read_parquet(path)


def _normalize_test_size(test_size: float | int, n_groups: int) -> int:
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("float test_size must be between 0 and 1")
        n_test = int(np.ceil(n_groups * test_size))
    else:
        n_test = int(test_size)
    if n_groups <= 1:
        return 0
    return min(max(n_test, 1), n_groups - 1)


def groupwise_train_test_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    test_size: float | int = 0.2,
    random_state: int = 0,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by whole groups so no group appears in both train and test."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    groups = pd.Series(df[group_col].dropna().unique())
    if groups.empty:
        raise ValueError(f"No non-null groups found in {group_col}")
    group_values = groups.to_numpy()
    if shuffle:
        rng = np.random.default_rng(random_state)
        group_values = group_values.copy()
        rng.shuffle(group_values)
    n_test = _normalize_test_size(test_size, len(group_values))
    test_groups = set(group_values[-n_test:]) if n_test else set()
    test_mask = df[group_col].isin(test_groups)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def split_by_decision_id(
    df: pd.DataFrame,
    *,
    test_size: float | int = 0.2,
    random_state: int = 0,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groupwise split using `decision_id`."""
    return groupwise_train_test_split(
        df,
        group_col="decision_id",
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )


def split_by_trajectory_id(
    df: pd.DataFrame,
    *,
    test_size: float | int = 0.2,
    random_state: int = 0,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groupwise split using `trajectory_id`."""
    return groupwise_train_test_split(
        df,
        group_col="trajectory_id",
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )


def groupwise_kfold_indices(
    df: pd.DataFrame,
    *,
    group_col: str,
    n_splits: int = 5,
    random_state: int = 0,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test positional indices for group-preserving K folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    groups = pd.Series(df[group_col].dropna().unique()).to_numpy()
    if len(groups) < n_splits:
        raise ValueError("n_splits cannot exceed the number of groups")
    if shuffle:
        rng = np.random.default_rng(random_state)
        groups = groups.copy()
        rng.shuffle(groups)

    folds = np.array_split(groups, n_splits)
    row_positions = np.arange(len(df))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_groups in folds:
        test_group_set = set(fold_groups)
        test_mask = df[group_col].isin(test_group_set).to_numpy()
        splits.append((row_positions[~test_mask], row_positions[test_mask]))
    return splits


def get_action_colors(
    actions: Iterable[str] = ACTIONS,
    *,
    colors: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a color mapping for action labels."""
    base = dict(DEFAULT_ACTION_COLORS)
    if colors is not None:
        base.update(colors)
    return {action: base.get(action, "#666666") for action in actions}


def save_figure(fig, path: str | Path, *, dpi: int = 200, close: bool = True) -> Path:
    """Save a matplotlib figure with parent directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path


def plot_action_bars(
    values: Mapping[str, float] | pd.Series,
    *,
    ax=None,
    colors: Mapping[str, str] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
):
    """Plot one bar per action using matplotlib only."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    series = pd.Series(values).reindex(ACTIONS)
    color_map = get_action_colors(colors=colors)
    ax.bar(ACTIONS, series.to_numpy(dtype=float), color=[color_map[a] for a in ACTIONS])
    ax.set_xlabel("action")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def plot_curve_with_sem(
    x: Sequence[float],
    y: Sequence[float],
    sem: Sequence[float] | None = None,
    *,
    ax=None,
    label: str | None = None,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
):
    """Plot a mean curve with an optional SEM band."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ax.plot(x_arr, y_arr, marker="o", label=label, color=color)
    if sem is not None:
        sem_arr = np.asarray(sem, dtype=float)
        ax.fill_between(x_arr, y_arr - sem_arr, y_arr + sem_arr, alpha=0.2, color=color)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    return ax


def plot_action_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    action_col: str = "agent_action",
    ax=None,
    colors: Mapping[str, str] | None = None,
    alpha: float = 0.7,
    size: float = 12.0,
    title: str | None = None,
):
    """Scatter plot colored by action label using matplotlib only."""
    import matplotlib.pyplot as plt

    missing = [col for col in [x, y, action_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing plotting column(s): {missing}")
    if ax is None:
        _, ax = plt.subplots()
    color_map = get_action_colors(colors=colors)
    for action, sub in df.groupby(action_col, dropna=False):
        action_label = str(action)
        ax.scatter(
            sub[x],
            sub[y],
            s=size,
            alpha=alpha,
            label=action_label,
            color=color_map.get(action_label, "#666666"),
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend(title=action_col, markerscale=1.5)
    return ax
