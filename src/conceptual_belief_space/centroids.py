"""Centroid helpers for conceptual belief space analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ACTIONS


def compute_action_centroids(
    features: np.ndarray,
    coords_2d: np.ndarray,
    actions: pd.Series,
    *,
    feature_columns: list[str],
    coord_prefix: str = "beh",
) -> pd.DataFrame:
    """Compute high-dimensional and 2D centroids by action."""
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape (n, 2)")
    if len(features) != len(coords_2d) or len(features) != len(actions):
        raise ValueError("features, coords_2d, and actions must have the same row count")
    if features.shape[1] != len(feature_columns):
        raise ValueError("feature_columns length must match feature dimension")

    action_values = actions.astype(str).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for action in ACTIONS:
        mask = action_values == action
        if not mask.any():
            continue
        feature_centroid = features[mask].mean(axis=0)
        coord_centroid = coords_2d[mask].mean(axis=0)
        row: dict[str, float | int | str] = {
            "action": action,
            "n_extreme_examples": int(mask.sum()),
            f"{coord_prefix}_umap1": float(coord_centroid[0]),
            f"{coord_prefix}_umap2": float(coord_centroid[1]),
        }
        for col, value in zip(feature_columns, feature_centroid):
            row[f"centroid_{col}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_action_centroids_2d(
    coords_2d: np.ndarray,
    actions: pd.Series,
    *,
    coord_prefix: str,
) -> pd.DataFrame:
    """Compute 2D centroids by action."""
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape (n, 2)")
    if len(coords_2d) != len(actions):
        raise ValueError("coords_2d and actions must have the same row count")
    action_values = actions.astype(str).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for action in ACTIONS:
        mask = action_values == action
        if not mask.any():
            continue
        centroid = coords_2d[mask].mean(axis=0)
        rows.append(
            {
                "action": action,
                "n_extreme_examples": int(mask.sum()),
                f"{coord_prefix}_umap1": float(centroid[0]),
                f"{coord_prefix}_umap2": float(centroid[1]),
            }
        )
    return pd.DataFrame(rows)


def compute_highdim_action_centroids(features: np.ndarray, actions: pd.Series) -> dict[str, np.ndarray]:
    """Compute high-dimensional centroids by action."""
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if len(features) != len(actions):
        raise ValueError("features and actions must have the same row count")
    action_values = actions.astype(str).to_numpy()
    out: dict[str, np.ndarray] = {}
    for action in ACTIONS:
        mask = action_values == action
        if mask.any():
            out[action] = np.asarray(features[mask].mean(axis=0), dtype=np.float32)
    return out


def pairwise_centroid_distances(centroids: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute pairwise Euclidean distances between action centroids."""
    centroid_cols = [f"centroid_{col}" for col in feature_columns]
    rows: list[dict[str, float | str]] = []
    for _, left in centroids.iterrows():
        for _, right in centroids.iterrows():
            diff = left[centroid_cols].to_numpy(dtype=float) - right[centroid_cols].to_numpy(dtype=float)
            rows.append(
                {
                    "action_i": str(left["action"]),
                    "action_j": str(right["action"]),
                    "centroid_distance": float(np.linalg.norm(diff)),
                }
            )
    return pd.DataFrame(rows)
