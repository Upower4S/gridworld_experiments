"""Activation feature loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def parse_layers_arg(layers_arg: str, available_layers: Sequence[int]) -> list[int]:
    """Parse `all` or a comma-separated layer list."""
    available = sorted(int(layer) for layer in available_layers)
    if layers_arg.strip().lower() == "all":
        return available
    requested = sorted({int(part.strip()) for part in layers_arg.split(",") if part.strip()})
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(f"requested layers are not available: {missing}; available={available}")
    return requested


def parse_representations_arg(
    representations_arg: str,
    available_representations: Sequence[str],
) -> list[str]:
    """Parse a comma-separated activation representation list."""
    available = set(str(rep) for rep in available_representations)
    requested = [part.strip() for part in representations_arg.split(",") if part.strip()]
    if not requested:
        raise ValueError("at least one activation representation is required")
    missing = [rep for rep in requested if rep not in available]
    if missing:
        raise ValueError(f"requested representations are not available: {missing}; available={sorted(available)}")
    return requested


def activation_feature_path(feature_dir: str | Path, layer: int, representation: str) -> Path:
    """Return the deterministic activation feature file path."""
    return Path(feature_dir) / f"layer_{int(layer):02d}_{representation}.npy"


def load_activation_memmap(path: str | Path, expected_rows: int | None = None) -> np.ndarray:
    """Open an activation feature matrix as a read-only NumPy memmap."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D feature matrix, got shape {arr.shape}")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ValueError(f"{path} row count {arr.shape[0]} does not match expected {expected_rows}")
    return arr


def fit_train_standardizer(
    features: np.ndarray,
    train_indices: Sequence[int],
    *,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Fit StandardScaler-equivalent parameters on train rows only."""
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError("cannot fit standardizer with zero train rows")
    train_x = np.asarray(features[train_indices], dtype=np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def transform_with_standardizer(
    features: np.ndarray,
    standardizer: dict[str, np.ndarray],
    *,
    l2_normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply train-fitted standardization and optional row L2 normalization."""
    x = np.asarray(features, dtype=np.float32)
    out = (x - standardizer["mean"]) / standardizer["scale"]
    out = np.asarray(out, dtype=np.float32)
    if l2_normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms).astype(np.float32)
        out = out / norms
    return np.asarray(out, dtype=np.float32)


def transform_indices_with_standardizer(
    features: np.ndarray,
    indices: Sequence[int],
    standardizer: dict[str, np.ndarray],
    *,
    l2_normalize: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Load selected rows and apply train-fitted standardization."""
    indices = np.asarray(indices, dtype=np.int64)
    selected = np.asarray(features[indices], dtype=np.float32)
    return transform_with_standardizer(
        selected,
        standardizer,
        l2_normalize=l2_normalize,
        eps=eps,
    )


def standardize_features_for_split(
    features: np.ndarray,
    train_indices: Sequence[int],
    *,
    l2_normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit preprocessing on train rows and transform all rows."""
    standardizer = fit_train_standardizer(features, train_indices, eps=eps)
    transformed = transform_with_standardizer(
        features,
        standardizer,
        l2_normalize=l2_normalize,
        eps=eps,
    )
    return transformed, standardizer
