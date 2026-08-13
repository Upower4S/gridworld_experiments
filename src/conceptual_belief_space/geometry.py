"""Geometry utilities for centroid distance-matrix alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ACTIONS
from .manifold import slugify


@dataclass(frozen=True)
class HighdimActivationCentroid:
    split_id: int
    subset: str
    layer: int
    representation: str
    action: str
    vector: np.ndarray


HIGHDIM_KEY_RE = re.compile(
    r"^split_(?P<split_id>\d+)__subset_(?P<subset>.+)__layer_(?P<layer>\d+)"
    r"__repr_(?P<representation>.+)__action_(?P<action>UP|DOWN|LEFT|RIGHT)$"
)


def ordered_action_matrix(df: pd.DataFrame, value_cols: list[str]) -> np.ndarray:
    """Return rows ordered as ACTIONS for the requested columns."""
    missing_cols = [col for col in value_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing centroid value columns: {missing_cols}")
    rows: list[np.ndarray] = []
    for action in ACTIONS:
        part = df.loc[df["action"].astype(str) == action]
        if part.empty:
            raise ValueError(f"missing centroid for action {action}")
        rows.append(part.iloc[0][value_cols].to_numpy(dtype=float))
    return np.vstack(rows)


def euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    x = np.asarray(x, dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def cosine_distance_matrix(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine distance matrix."""
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    x_norm = x / norms
    sim = np.clip(x_norm @ x_norm.T, -1.0, 1.0)
    return 1.0 - sim


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle values excluding the diagonal."""
    matrix = np.asarray(matrix, dtype=float)
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def distance_matrix_long(matrix: np.ndarray, *, value_name: str) -> pd.DataFrame:
    """Convert an action distance matrix to long form."""
    rows: list[dict[str, float | str]] = []
    for i, action_i in enumerate(ACTIONS):
        for j, action_j in enumerate(ACTIONS):
            rows.append({"action_i": action_i, "action_j": action_j, value_name: float(matrix[i, j])})
    return pd.DataFrame(rows)


def parse_activation_highdim_centroids(npz) -> list[HighdimActivationCentroid]:
    """Parse Step 5 high-dimensional activation centroid arrays."""
    records: list[HighdimActivationCentroid] = []
    for key in npz.files:
        match = HIGHDIM_KEY_RE.match(key)
        if not match:
            continue
        records.append(
            HighdimActivationCentroid(
                split_id=int(match.group("split_id")),
                subset=match.group("subset"),
                layer=int(match.group("layer")),
                representation=match.group("representation"),
                action=match.group("action"),
                vector=np.asarray(npz[key], dtype=np.float32),
            )
        )
    return records


def highdim_centroid_lookup(records: Iterable[HighdimActivationCentroid]) -> dict[tuple[int, str, int, str], dict[str, np.ndarray]]:
    """Build a lookup keyed by split/subset/layer/representation."""
    lookup: dict[tuple[int, str, int, str], dict[str, np.ndarray]] = {}
    for record in records:
        key = (record.split_id, record.subset, record.layer, record.representation)
        lookup.setdefault(key, {})[record.action] = record.vector
    return lookup


def highdim_matrix_for_actions(action_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Return an ACTIONS-ordered high-dimensional centroid matrix."""
    missing = [action for action in ACTIONS if action not in action_vectors]
    if missing:
        raise ValueError(f"missing high-dimensional centroids for actions: {missing}")
    return np.vstack([np.asarray(action_vectors[action], dtype=float) for action in ACTIONS])


def matrix_npz_key(
    *,
    kind: str,
    split_id: int,
    subset: str,
    representation: str,
    distance_kind: str,
    layer: int | None = None,
) -> str:
    """Build a stable NPZ key for a distance matrix."""
    parts = [
        slugify(kind),
        f"split_{int(split_id)}",
        f"subset_{slugify(subset)}",
    ]
    if layer is not None:
        parts.append(f"layer_{int(layer):02d}")
    parts.extend([f"repr_{slugify(representation)}", slugify(distance_kind)])
    return "__".join(parts)
