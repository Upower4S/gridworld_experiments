"""Diagnostics for train-only action-extreme activation UMAPs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS, DEFAULT_SEED
from .io import ensure_dir
from .plotting import action_color_map, savefig


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated string list."""
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_splits(path: str | Path) -> list[dict[str, Any]]:
    """Load grouped split records."""
    import json

    with Path(path).open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of split records")
    return data


def train_row_ids_for_split(beliefs: pd.DataFrame, splits: list[dict[str, Any]], split_id: int) -> np.ndarray:
    """Return train row_id values for a split."""
    split_lookup = {int(split["split_id"]): split for split in splits}
    if int(split_id) not in split_lookup:
        raise ValueError(f"Unknown split_id {split_id}; available={sorted(split_lookup)}")
    train_indices = np.asarray(split_lookup[int(split_id)]["train_indices"], dtype=np.int64)
    if "row_id" not in beliefs.columns:
        raise ValueError("beliefs must contain row_id")
    if train_indices.min(initial=0) < 0 or train_indices.max(initial=-1) >= len(beliefs):
        raise ValueError(f"train_indices for split {split_id} are outside beliefs row bounds")
    return beliefs.iloc[train_indices]["row_id"].to_numpy(dtype=np.int64)


def select_extreme_rows(
    *,
    beliefs: pd.DataFrame,
    splits: list[dict[str, Any]],
    extremes: pd.DataFrame,
    split_id: int,
    subset: str,
    top_k_per_action: int | None,
    max_rows_per_action: int | None,
    deduplicate_row_ids: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select train-only action-extreme rows and merge belief metadata."""
    required_extreme = {"split_id", "subset", "action", "row_id", "q_action", "rank_within_action"}
    missing = sorted(required_extreme - set(extremes.columns))
    if missing:
        raise ValueError(f"extremes missing required columns: {missing}")
    if "row_id" not in beliefs.columns:
        raise ValueError("beliefs missing required column row_id")
    if not beliefs["row_id"].is_unique:
        raise ValueError("beliefs row_id must be unique")

    selected = extremes[
        extremes["split_id"].astype(int).eq(int(split_id))
        & extremes["subset"].astype(str).eq(str(subset))
    ].copy()
    if selected.empty:
        available = (
            extremes[["split_id", "subset"]]
            .drop_duplicates()
            .sort_values(["split_id", "subset"])
            .to_dict("records")
        )
        raise ValueError(f"No extreme rows for split_id={split_id}, subset={subset!r}; available={available[:20]}")

    train_row_ids = set(train_row_ids_for_split(beliefs, splits, split_id).tolist())
    before_train_filter = len(selected)
    if "split_role" in selected.columns:
        selected = selected[selected["split_role"].astype(str).str.lower().eq("train")].copy()
    selected = selected[selected["row_id"].astype(int).isin(train_row_ids)].copy()
    train_filtered = before_train_filter - len(selected)
    if selected.empty:
        raise ValueError(f"No train extreme rows remain for split_id={split_id}, subset={subset!r}")

    selected["row_id"] = selected["row_id"].astype(np.int64)
    selected["q_action"] = pd.to_numeric(selected["q_action"], errors="coerce")
    selected["rank_within_action"] = pd.to_numeric(selected["rank_within_action"], errors="coerce")
    selected = selected[selected["action"].astype(str).isin(ACTIONS)].copy()
    if selected.empty:
        raise ValueError("No selected extreme rows have valid action labels")

    duplicate_pairs_before = len(selected)
    duplicates_seen = int(selected["row_id"].duplicated(keep=False).sum())
    if deduplicate_row_ids == "keep_highest_q_action":
        selected = (
            selected.sort_values(["row_id", "q_action", "rank_within_action"], ascending=[True, False, True])
            .drop_duplicates("row_id", keep="first")
            .copy()
        )
    elif deduplicate_row_ids != "none":
        raise ValueError("--deduplicate_row_ids must be keep_highest_q_action or none")
    duplicates_removed = duplicate_pairs_before - len(selected)

    selected = selected.sort_values(["action", "q_action", "rank_within_action"], ascending=[True, False, True])
    if top_k_per_action is not None:
        selected = selected.groupby("action", group_keys=False).head(int(top_k_per_action)).copy()

    if max_rows_per_action is not None:
        sampled_parts: list[pd.DataFrame] = []
        rng = np.random.default_rng(random_state)
        for action in ACTIONS:
            part = selected[selected["action"].astype(str).eq(action)].copy()
            if len(part) > int(max_rows_per_action):
                # pandas sample needs an int seed, so derive a stable per-action seed.
                seed = int(rng.integers(0, 2**31 - 1))
                part = part.sample(n=int(max_rows_per_action), random_state=seed)
            sampled_parts.append(part)
        selected = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else selected.iloc[0:0].copy()

    selected = selected.rename(columns={"action": "extreme_action"})
    metadata_cols = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
        "entropy_commitment",
        "q_UP",
        "q_DOWN",
        "q_LEFT",
        "q_RIGHT",
    ]
    available_metadata = [col for col in metadata_cols if col in beliefs.columns]
    merged = selected.merge(
        beliefs[available_metadata],
        on="row_id",
        how="left",
        suffixes=("", "_belief"),
        validate="many_to_one",
    )
    for col in ["decision_id", "trajectory_id", "prefix_token_index", "progress", "agent_action", "top_action"]:
        belief_col = f"{col}_belief"
        if belief_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[belief_col]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[belief_col])
            merged = merged.drop(columns=[belief_col])
    if merged["decision_id"].isna().any() if "decision_id" in merged.columns else True:
        raise ValueError("Failed to merge selected extreme rows with belief metadata")

    counts = merged["extreme_action"].value_counts().reindex(ACTIONS, fill_value=0).astype(int).to_dict()
    summary = {
        "split_id": int(split_id),
        "subset": str(subset),
        "rows_before_train_filter": int(before_train_filter),
        "rows_removed_by_train_filter": int(train_filtered),
        "duplicates_seen": int(duplicates_seen),
        "duplicates_removed": int(duplicates_removed),
        "top_k_per_action": None if top_k_per_action is None else int(top_k_per_action),
        "max_rows_per_action": None if max_rows_per_action is None else int(max_rows_per_action),
        "n_rows": int(len(merged)),
        "counts_by_extreme_action": counts,
    }
    return merged.reset_index(drop=True), summary


def resolve_feature_row_indices(
    selected: pd.DataFrame,
    *,
    feature_rows: int,
    activation_metadata: pd.DataFrame | None = None,
) -> np.ndarray:
    """Resolve selected rows to feature matrix row indices."""
    row_ids = selected["row_id"].to_numpy(dtype=np.int64)
    if len(row_ids) and row_ids.min() >= 0 and row_ids.max() < feature_rows:
        return row_ids

    if activation_metadata is None:
        raise ValueError(
            "row_id values are not direct feature row indices and activation metadata was not provided"
        )

    if {"row_id", "row_index"}.issubset(activation_metadata.columns):
        mapping = activation_metadata.set_index("row_id")["row_index"].to_dict()
        out = selected["row_id"].map(mapping)
        if out.notna().all():
            return out.to_numpy(dtype=np.int64)

    if {"prefix_id", "row_index"}.issubset(activation_metadata.columns) and "prefix_id" in selected.columns:
        mapping = activation_metadata.set_index("prefix_id")["row_index"].to_dict()
        out = selected["prefix_id"].map(mapping)
        if out.notna().all():
            return out.to_numpy(dtype=np.int64)

    raise ValueError(
        "Could not map selected rows to activation feature rows. "
        "Expected row_id as direct index, or activation metadata with row_id/prefix_id and row_index."
    )


def load_feature_memmap(feature_dir: str | Path, layer: int, representation: str) -> np.ndarray:
    """Open a layer/representation activation feature file as a read-only memmap."""
    path = Path(feature_dir) / f"layer_{int(layer):02d}_{representation}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {arr.shape}")
    return arr


def fit_standard_scaler_chunked(
    features: np.ndarray,
    train_indices: Sequence[int],
    *,
    chunk_size: int = 2048,
):
    """Fit sklearn StandardScaler on train rows in chunks."""
    from sklearn.preprocessing import StandardScaler

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError("Cannot fit scaler with zero train rows")
    if train_indices.min(initial=0) < 0 or train_indices.max(initial=-1) >= features.shape[0]:
        raise ValueError("Train indices are outside feature row bounds")
    scaler = StandardScaler(copy=True)
    for start in range(0, len(train_indices), chunk_size):
        idx = train_indices[start : start + chunk_size]
        scaler.partial_fit(np.asarray(features[idx], dtype=np.float32))
    return scaler


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return np.asarray(x / norms, dtype=np.float32)


def preprocess_selected_features(
    features: np.ndarray,
    selected_indices: np.ndarray,
    train_indices: Sequence[int],
    *,
    fit_scaler_on: str,
    l2_normalize: bool,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale selected features using train-fitted parameters and optional L2 normalization."""
    if selected_indices.min(initial=0) < 0 or selected_indices.max(initial=-1) >= features.shape[0]:
        raise ValueError("Selected feature row indices are outside feature row bounds")

    selected_x = np.asarray(features[selected_indices], dtype=np.float32)
    if fit_scaler_on == "train":
        scaler = fit_standard_scaler_chunked(features, train_indices, chunk_size=chunk_size)
        selected_x = np.asarray(scaler.transform(selected_x), dtype=np.float32)
        scaler_summary = {
            "fit_scaler_on": "train",
            "n_scaler_rows": int(len(train_indices)),
        }
    elif fit_scaler_on == "selected":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        selected_x = np.asarray(scaler.fit_transform(selected_x), dtype=np.float32)
        scaler_summary = {
            "fit_scaler_on": "selected",
            "n_scaler_rows": int(len(selected_indices)),
        }
    elif fit_scaler_on == "none":
        scaler_summary = {
            "fit_scaler_on": "none",
            "n_scaler_rows": 0,
        }
    else:
        raise ValueError("--fit_scaler_on must be train, selected, or none")

    if not np.isfinite(selected_x).all():
        raise ValueError("Selected preprocessed features contain NaN or infinity")
    if l2_normalize:
        selected_x = l2_normalize_rows(selected_x)
    return selected_x, scaler_summary


def fit_umap(
    x: np.ndarray,
    *,
    backend: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int | None,
    n_jobs: int,
) -> np.ndarray:
    """Fit 2D UMAP on selected rows and return coordinates for those same rows."""
    if x.ndim != 2:
        raise ValueError("UMAP input must be 2D")
    if len(x) < 3:
        raise ValueError("Need at least three selected rows for UMAP")
    effective_neighbors = min(int(n_neighbors), max(2, len(x) - 1))
    backend = str(backend).lower()
    if backend in {"sklearn", "cpu", "umap"}:
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError("UMAP requires `umap-learn`. Install with `pip install umap-learn`.") from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric=metric,
            random_state=random_state,
            n_jobs=int(n_jobs),
        )
        coords = reducer.fit_transform(x)
        return np.asarray(coords, dtype=np.float32)

    if backend == "cuml":
        try:
            import cupy as cp  # type: ignore
            from cuml.manifold import UMAP as CuMLUMAP  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The cuml backend requires RAPIDS/cuML and cupy in a CUDA environment. "
                "Use --umap_backend sklearn on this machine."
            ) from exc

        reducer = CuMLUMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(cp.asarray(np.asarray(x, dtype=np.float32)))
        return np.asarray(cp.asnumpy(coords), dtype=np.float32)

    raise ValueError("--umap_backend must be sklearn, cpu, umap, or cuml")


def _class_centroids(x: np.ndarray, labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    actions = [action for action in ACTIONS if np.any(labels == action)]
    centroids = np.vstack([x[labels == action].mean(axis=0) for action in actions]).astype(np.float32)
    return actions, centroids


def _pairwise_centroid_distance_mean(centroids: np.ndarray, *, metric: str) -> float:
    if len(centroids) < 2:
        return float("nan")
    values: list[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if metric == "cosine":
                denom = np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                value = 1.0 - float(np.dot(centroids[i], centroids[j]) / denom) if denom > 0 else float("nan")
            else:
                value = float(np.linalg.norm(centroids[i] - centroids[j]))
            values.append(value)
    return float(np.nanmean(values)) if values else float("nan")


def nearest_centroid_metrics(x: np.ndarray, labels: np.ndarray, *, metric: str) -> dict[str, float]:
    """Nearest-centroid accuracy and margin."""
    actions, centroids = _class_centroids(x, labels)
    if len(actions) < 2:
        return {"accuracy": float("nan"), "mean_margin": float("nan")}

    if metric == "cosine":
        x_norm = l2_normalize_rows(np.asarray(x, dtype=np.float32))
        c_norm = l2_normalize_rows(np.asarray(centroids, dtype=np.float32))
        scores = x_norm @ c_norm.T
        order = np.argsort(scores, axis=1)[:, ::-1]
        pred = np.asarray(actions, dtype=object)[order[:, 0]]
        margins = scores[np.arange(len(x)), order[:, 0]] - scores[np.arange(len(x)), order[:, 1]]
    else:
        diffs = x[:, None, :] - centroids[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        order = np.argsort(distances, axis=1)
        pred = np.asarray(actions, dtype=object)[order[:, 0]]
        margins = distances[np.arange(len(x)), order[:, 1]] - distances[np.arange(len(x)), order[:, 0]]

    return {
        "accuracy": float(np.mean(pred == labels)),
        "mean_margin": float(np.mean(margins)),
    }


def within_action_distance_mean(x: np.ndarray, labels: np.ndarray, *, metric: str) -> float:
    """Mean point-to-own-centroid distance."""
    actions, centroids = _class_centroids(x, labels)
    if len(actions) < 1:
        return float("nan")
    values: list[float] = []
    for i, action in enumerate(actions):
        part = x[labels == action]
        centroid = centroids[i]
        if metric == "cosine":
            part_norm = l2_normalize_rows(np.asarray(part, dtype=np.float32))
            centroid_norm = l2_normalize_rows(np.asarray(centroid[None, :], dtype=np.float32))[0]
            distances = 1.0 - part_norm @ centroid_norm
        else:
            distances = np.linalg.norm(part - centroid, axis=1)
        values.extend(float(v) for v in distances)
    return float(np.nanmean(values)) if values else float("nan")


def safe_silhouette(x: np.ndarray, labels: np.ndarray, *, metric: str) -> float:
    """Return silhouette score or NaN if not defined."""
    from sklearn.metrics import silhouette_score

    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(x, labels, metric=metric))
    except Exception:
        return float("nan")


def compute_metrics_for_config(
    *,
    layer: int,
    representation: str,
    selected: pd.DataFrame,
    highdim_x: np.ndarray,
    coords: np.ndarray,
    l2_normalize: bool,
) -> dict[str, float | int | str]:
    """Compute separation metrics for one layer/representation."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    labels = selected["extreme_action"].astype(str).to_numpy()
    counts = selected["extreme_action"].value_counts().reindex(ACTIONS, fill_value=0).astype(int)
    highdim_metric = "cosine" if l2_normalize else "euclidean"

    umap_centroid = nearest_centroid_metrics(coords, labels, metric="euclidean")
    highdim_centroid = nearest_centroid_metrics(highdim_x, labels, metric=highdim_metric)
    _, umap_centroids = _class_centroids(coords, labels)
    _, highdim_centroids = _class_centroids(highdim_x, labels)
    between_umap = _pairwise_centroid_distance_mean(umap_centroids, metric="euclidean")
    within_umap = within_action_distance_mean(coords, labels, metric="euclidean")
    between_highdim = _pairwise_centroid_distance_mean(highdim_centroids, metric=highdim_metric)
    within_highdim = within_action_distance_mean(highdim_x, labels, metric=highdim_metric)

    try:
        db = float(davies_bouldin_score(coords, labels)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        db = float("nan")
    try:
        ch = float(calinski_harabasz_score(coords, labels)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        ch = float("nan")

    return {
        "layer": int(layer),
        "representation": representation,
        "n_rows": int(len(selected)),
        "n_rows_UP": int(counts["UP"]),
        "n_rows_DOWN": int(counts["DOWN"]),
        "n_rows_LEFT": int(counts["LEFT"]),
        "n_rows_RIGHT": int(counts["RIGHT"]),
        "silhouette_score_umap_by_extreme_action": safe_silhouette(coords, labels, metric="euclidean"),
        "silhouette_score_highdim_by_extreme_action": safe_silhouette(highdim_x, labels, metric=highdim_metric),
        "nearest_centroid_accuracy_umap": umap_centroid["accuracy"],
        "nearest_centroid_accuracy_highdim": highdim_centroid["accuracy"],
        "mean_nearest_centroid_margin_umap": umap_centroid["mean_margin"],
        "mean_nearest_centroid_margin_highdim": highdim_centroid["mean_margin"],
        "between_centroid_distance_mean_umap": between_umap,
        "within_action_distance_mean_umap": within_umap,
        "between_within_ratio_umap": between_umap / within_umap if within_umap and math.isfinite(within_umap) else float("nan"),
        "between_centroid_distance_mean_highdim": between_highdim,
        "within_action_distance_mean_highdim": within_highdim,
        "between_within_ratio_highdim": (
            between_highdim / within_highdim if within_highdim and math.isfinite(within_highdim) else float("nan")
        ),
        "davies_bouldin_score_umap": db,
        "calinski_harabasz_score_umap": ch,
    }


def coordinate_frame(
    *,
    selected: pd.DataFrame,
    coords: np.ndarray,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
) -> pd.DataFrame:
    """Build coordinate output frame for one config."""
    cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "prefix_token_index",
        "progress",
        "extreme_action",
        "agent_action",
        "top_action",
        "q_action",
        "commitment_to_final",
        "entropy_commitment",
    ]
    out = selected[[col for col in cols if col in selected.columns]].copy()
    out.insert(0, "representation", representation)
    out.insert(0, "layer", int(layer))
    out.insert(0, "subset", str(subset))
    out.insert(0, "split_id", int(split_id))
    out["umap1"] = coords[:, 0].astype(np.float32)
    out["umap2"] = coords[:, 1].astype(np.float32)
    return out


def plot_individual_umap(
    coords: pd.DataFrame,
    *,
    path: str | Path,
    title: str,
    s: float = 16,
) -> Path:
    """Plot one UMAP scatter with action centroids."""
    import matplotlib.pyplot as plt

    colors = action_color_map()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for action in ACTIONS:
        part = coords[coords["extreme_action"].astype(str).eq(action)]
        if part.empty:
            continue
        ax.scatter(
            part["umap1"],
            part["umap2"],
            s=s,
            alpha=0.78,
            color=colors[action],
            label=action,
            edgecolors="none",
        )
        centroid = part[["umap1", "umap2"]].mean()
        ax.scatter(
            [centroid["umap1"]],
            [centroid["umap2"]],
            s=150,
            marker="X",
            color=colors[action],
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(
            float(centroid["umap1"]),
            float(centroid["umap2"]),
            f" {action}",
            fontsize=9,
            weight="bold",
            va="center",
        )
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.18, linewidth=0.5)
    return savefig(path, dpi=200, fig=fig)


def plot_grid_by_representation(
    coords_all: pd.DataFrame,
    *,
    representation: str,
    layers: Sequence[int],
    path: str | Path,
) -> Path:
    """Create a grid over layers for one representation."""
    import matplotlib.pyplot as plt

    layers = list(layers)
    ncols = min(4, max(1, len(layers)))
    nrows = int(math.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    colors = action_color_map()
    for ax in axes.ravel():
        ax.axis("off")
    for i, layer in enumerate(layers):
        ax = axes.ravel()[i]
        ax.axis("on")
        part = coords_all[
            coords_all["representation"].astype(str).eq(representation)
            & coords_all["layer"].astype(int).eq(int(layer))
        ]
        for action in ACTIONS:
            action_part = part[part["extreme_action"].astype(str).eq(action)]
            if not action_part.empty:
                ax.scatter(
                    action_part["umap1"],
                    action_part["umap2"],
                    s=8,
                    alpha=0.75,
                    color=colors[action],
                    label=action if i == 0 else None,
                    edgecolors="none",
                )
        ax.set_title(f"layer {int(layer):02d}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=len(ACTIONS), loc="upper center")
    fig.suptitle(f"Extreme activation UMAP grid: {representation}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return savefig(path, dpi=200, fig=fig)


def plot_layer_representation_grid(
    coords_all: pd.DataFrame,
    *,
    layers: Sequence[int],
    representations: Sequence[str],
    path: str | Path,
) -> Path:
    """Create a layer by representation UMAP grid."""
    import matplotlib.pyplot as plt

    layers = list(layers)
    representations = list(representations)
    fig, axes = plt.subplots(
        len(layers),
        len(representations),
        figsize=(4.0 * len(representations), 2.9 * len(layers)),
        squeeze=False,
    )
    colors = action_color_map()
    for row, layer in enumerate(layers):
        for col, representation in enumerate(representations):
            ax = axes[row, col]
            part = coords_all[
                coords_all["representation"].astype(str).eq(representation)
                & coords_all["layer"].astype(int).eq(int(layer))
            ]
            for action in ACTIONS:
                action_part = part[part["extreme_action"].astype(str).eq(action)]
                if not action_part.empty:
                    ax.scatter(
                        action_part["umap1"],
                        action_part["umap2"],
                        s=6,
                        alpha=0.72,
                        color=colors[action],
                        label=action if row == 0 and col == 0 else None,
                        edgecolors="none",
                    )
            if row == 0:
                ax.set_title(str(representation), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{int(layer):02d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=len(ACTIONS), loc="upper center")
    fig.suptitle("Extreme activation UMAPs by layer and representation", y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return savefig(path, dpi=200, fig=fig)


def rank_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by the requested diagnostic criteria."""
    criteria = [
        "silhouette_score_highdim_by_extreme_action",
        "nearest_centroid_accuracy_highdim",
        "silhouette_score_umap_by_extreme_action",
        "nearest_centroid_accuracy_umap",
        "between_within_ratio_highdim",
    ]
    ranked = metrics.copy()
    for col in criteria:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked = ranked.sort_values(criteria, ascending=[False, False, False, False, False], na_position="last")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    return ranked


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Render a small dataframe as markdown without requiring tabulate."""
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "NA" if pd.isna(v) else f"{float(v):.4f}")
    columns = list(shown.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def direct_answer(metrics: pd.DataFrame) -> tuple[str, str]:
    """Classify whether the diagnostic shows clear action separation."""
    if metrics.empty:
        return "no", "No metric rows were produced."
    best = rank_metric_table(metrics).iloc[0]
    sil = float(best.get("silhouette_score_highdim_by_extreme_action", float("nan")))
    acc = float(best.get("nearest_centroid_accuracy_highdim", float("nan")))
    if (math.isfinite(sil) and sil >= 0.35) or (math.isfinite(acc) and acc >= 0.85):
        verdict = "yes"
    elif (math.isfinite(sil) and sil >= 0.15) or (math.isfinite(acc) and acc >= 0.60):
        verdict = "weakly"
    else:
        verdict = "no"
    reason = (
        f"Best config is layer {int(best['layer'])} / {best['representation']} with "
        f"high-dimensional silhouette={sil:.4f} and nearest-centroid accuracy={acc:.4f}."
    )
    return verdict, reason

