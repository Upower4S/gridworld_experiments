"""Trajectory helpers for conceptual belief space analyses."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS


def path_length(coords: np.ndarray) -> float:
    """Sum Euclidean distances between consecutive coordinate points."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def total_variation(values: np.ndarray) -> float:
    """Sum L1 distance between consecutive vector values."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.abs(np.diff(values, axis=0)).sum())


def ordered_centroid_coords(centroids: pd.DataFrame, coord_cols: Sequence[str]) -> np.ndarray:
    """Return centroid coordinates ordered by ACTIONS."""
    rows: list[np.ndarray] = []
    for action in ACTIONS:
        part = centroids.loc[centroids["action"].astype(str) == action]
        if part.empty:
            raise ValueError(f"missing centroid for action {action}")
        rows.append(part.iloc[0][list(coord_cols)].to_numpy(dtype=float))
    return np.vstack(rows)


def distances_to_centroids(coords: np.ndarray, centroid_coords: np.ndarray) -> np.ndarray:
    """Return distance from each point to each centroid."""
    coords = np.asarray(coords, dtype=float)
    centroid_coords = np.asarray(centroid_coords, dtype=float)
    diff = coords[:, None, :] - centroid_coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def nearest_centroid_actions(coords: np.ndarray, centroid_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest action label and distance for each point."""
    distances = distances_to_centroids(coords, centroid_coords)
    idx = np.argmin(distances, axis=1)
    return np.asarray(ACTIONS, dtype=object)[idx], distances[np.arange(len(distances)), idx]


def distance_to_action_centroid(
    coords: np.ndarray,
    centroid_coords: np.ndarray,
    final_actions: Sequence[str],
) -> np.ndarray:
    """Return distance from each point to its final-action centroid."""
    action_to_idx = {action: i for i, action in enumerate(ACTIONS)}
    distances = distances_to_centroids(coords, centroid_coords)
    out = np.full(len(coords), np.nan, dtype=float)
    for i, action in enumerate(final_actions):
        idx = action_to_idx.get(str(action))
        if idx is not None:
            out[i] = distances[i, idx]
    return out


def switch_count(labels: Sequence[str]) -> int:
    """Count label changes over an ordered trajectory."""
    labels = list(map(str, labels))
    if len(labels) < 2:
        return 0
    return int(sum(a != b for a, b in zip(labels[:-1], labels[1:])))


def first_lock_in_progress(
    progress: Sequence[float],
    nearest_actions: Sequence[str],
    final_action: str,
) -> float | None:
    """First progress where nearest centroid equals final action."""
    for p, action in zip(progress, nearest_actions):
        if str(action) == str(final_action):
            return float(p)
    return None


def permanent_lock_in_progress(
    progress: Sequence[float],
    nearest_actions: Sequence[str],
    final_action: str,
) -> float | None:
    """Earliest progress where all later nearest centroids equal final action."""
    progress_arr = np.asarray(progress, dtype=float)
    actions = np.asarray(list(map(str, nearest_actions)), dtype=object)
    final = str(final_action)
    for i in range(len(actions)):
        if np.all(actions[i:] == final):
            return float(progress_arr[i])
    return None


def selected_decision_ids(
    df: pd.DataFrame,
    *,
    max_decisions: int = 8,
    action_col: str = "agent_action",
) -> list[str]:
    """Pick several decisions with long trajectories, balanced by action where possible."""
    counts = (
        df.groupby([action_col, "decision_id"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values([action_col, "n"], ascending=[True, False])
    )
    selected: list[str] = []
    for action in ACTIONS:
        part = counts.loc[counts[action_col].astype(str) == action]
        if not part.empty:
            selected.append(str(part["decision_id"].iloc[0]))
    for decision_id in counts.sort_values("n", ascending=False)["decision_id"].astype(str):
        if decision_id not in selected:
            selected.append(decision_id)
        if len(selected) >= max_decisions:
            break
    return selected[:max_decisions]
