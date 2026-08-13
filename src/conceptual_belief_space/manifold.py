"""Manifold helpers for conceptual belief space analyses."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np

from .config import ACTIONS, DEFAULT_SEED

Representation = Literal["clr", "sqrt_q"]


def representation_columns(representation: str) -> list[str]:
    """Return feature columns for a behavior belief representation."""
    if representation == "clr":
        return [f"clr_{action}" for action in ACTIONS]
    if representation == "sqrt_q":
        return [f"sqrt_q_{action}" for action in ACTIONS]
    raise ValueError("representation must be one of: clr, sqrt_q")


def require_umap():
    """Import umap-learn or fail with the installation command."""
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "UMAP is required for the conceptual belief manifold. Install it with:\n"
            "  pip install umap-learn"
        ) from exc
    return umap


def has_cuml_umap() -> bool:
    """Return whether RAPIDS cuML UMAP appears importable."""
    try:
        import cuml.manifold  # type: ignore  # noqa: F401
        import cupy  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_umap_backend(backend: str) -> str:
    """Resolve `auto` to the fastest available UMAP backend."""
    backend = str(backend).lower()
    if backend not in {"auto", "cpu", "cuml"}:
        raise ValueError("umap backend must be one of: auto, cpu, cuml")
    if backend == "auto":
        return "cuml" if has_cuml_umap() else "cpu"
    if backend == "cuml" and not has_cuml_umap():
        raise ImportError(
            "The cuML UMAP backend requires a CUDA/RAPIDS environment with `cuml` and `cupy` installed. "
            "On non-CUDA machines, use `--umap_backend cpu`."
        )
    return backend


def fit_umap_transform(
    train_x: np.ndarray,
    all_x: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = DEFAULT_SEED,
    backend: str = "cpu",
    n_jobs: int | None = None,
) -> tuple[object, np.ndarray]:
    """Fit UMAP on train anchors and transform all rows."""
    if train_x.ndim != 2 or all_x.ndim != 2:
        raise ValueError("train_x and all_x must both be 2D")
    if train_x.shape[1] != all_x.shape[1]:
        raise ValueError(f"feature dimension mismatch: train={train_x.shape}, all={all_x.shape}")
    if len(train_x) < 2:
        raise ValueError("need at least two train anchor examples to fit UMAP")

    resolved_backend = resolve_umap_backend(backend)
    if resolved_backend == "cuml":
        import cupy as cp  # type: ignore
        from cuml.manifold import UMAP as CuMLUMAP  # type: ignore

        reducer = CuMLUMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        train_gpu = cp.asarray(np.asarray(train_x, dtype=np.float32))
        all_gpu = cp.asarray(np.asarray(all_x, dtype=np.float32))
        reducer.fit(train_gpu)
        coords = cp.asnumpy(reducer.transform(all_gpu))
        return reducer, np.asarray(coords, dtype=np.float32)

    umap = require_umap()
    kwargs: dict[str, object] = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }
    if n_jobs is not None:
        kwargs["n_jobs"] = int(n_jobs)
    reducer = umap.UMAP(**kwargs)
    reducer.fit(train_x)
    coords = reducer.transform(all_x)
    return reducer, np.asarray(coords, dtype=np.float32)


def fit_debug_pca_transform(train_x: np.ndarray, all_x: np.ndarray) -> tuple[object, np.ndarray]:
    """Diagnostic PCA fallback, only for explicit debug use."""
    from sklearn.decomposition import PCA

    model = PCA(n_components=2, random_state=DEFAULT_SEED)
    model.fit(train_x)
    coords = model.transform(all_x)
    return model, np.asarray(coords, dtype=np.float32)


def slugify(value: object) -> str:
    """Make a filesystem-safe token."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def model_path(
    output_dir: str | Path,
    *,
    split_id: int,
    subset: str,
    representation: str,
    method: str = "umap",
) -> Path:
    """Return the deterministic model path for a behavior manifold model."""
    output_dir = Path(output_dir)
    return output_dir / f"behavior_{method}_{split_id}_{slugify(subset)}_{slugify(representation)}.joblib"


def activation_model_path(
    output_dir: str | Path,
    *,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    method: str = "umap",
) -> Path:
    """Return the deterministic model path for an activation manifold model."""
    output_dir = Path(output_dir)
    return (
        output_dir
        / f"activation_{method}_{split_id}_{slugify(subset)}_layer_{int(layer):02d}_{slugify(representation)}.joblib"
    )
