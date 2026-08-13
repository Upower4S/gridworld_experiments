"""Strict utilities for probe-guided activation PCA/UMAP embeddings.

Central invariant: PCA/UMAP coordinates are computed from activation arrays only.
Probe/logit/score/q/metadata columns may be labels or plot metadata, but are not
embedding input features.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{a}" for a in ACTIONS]
EPS = 1e-12
FORBIDDEN_INPUT_COLUMNS = {
    "q_UP", "q_DOWN", "q_LEFT", "q_RIGHT", "log_q_UP", "log_q_DOWN", "log_q_LEFT", "log_q_RIGHT",
    "clr_UP", "clr_DOWN", "clr_LEFT", "clr_RIGHT", "b_clr_UP", "b_clr_DOWN", "b_clr_LEFT", "b_clr_RIGHT",
    "sqrt_q_UP", "sqrt_q_DOWN", "sqrt_q_LEFT", "sqrt_q_RIGHT", "agent_action", "top_action",
    "true_q_top_action", "commitment_to_final", "entropy_commitment", "margin_commitment", "progress",
    "prefix_token_index", "trajectory_id", "decision_id", "row_id", "row_index", "prefix_id", "step_id",
    "stacker_score_UP", "stacker_score_DOWN", "stacker_score_LEFT", "stacker_score_RIGHT",
    "score_UP", "score_DOWN", "score_LEFT", "score_RIGHT",
}
REQUIRED_CONTROLS = ["all_dims", "random_top_K", "shuffled_probe_top_K", "progress_probe_top_K"]
PROBE_FAMILIES = ["probe_top_K", "probe_weighted_top_K"]
ALL_FAMILIES = ["all_dims", "probe_top_K", "probe_weighted_top_K", "random_top_K", "shuffled_probe_top_K", "progress_probe_top_K"]
META_KEEP_COLS = [
    "row_id", "row_index", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index",
    "progress", "agent_action", "top_action", "true_q_top_action", *Q_COLS, "entropy_commitment",
    "commitment_to_final", "margin_commitment", "is_before_first_action_mention", "is_before_first_final_action_mention",
    "strict_before_any_action_mention", "strict_before_final_action_mention",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_markdown(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n")


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def action_indices(values: Iterable[str]) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    s = pd.Series(values).astype(str)
    out = s.map(mapping)
    if out.isna().any():
        raise ValueError(f"Unknown action labels: {sorted(s[out.isna()].unique())}")
    return out.to_numpy(dtype=np.int64)


def compute_true_q_top(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in Q_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing q columns: {missing}")
    q = df[Q_COLS].to_numpy(dtype=np.float64)
    q = q / np.clip(q.sum(axis=1, keepdims=True), EPS, None)
    return pd.Series(np.asarray(ACTIONS, dtype=object)[np.argmax(q, axis=1)], index=df.index)


def load_activation_metadata(path: str | Path) -> pd.DataFrame:
    df = read_parquet(path)
    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = np.arange(len(df), dtype=np.int64)
    if "row_id" not in df.columns:
        df["row_id"] = df["row_index"].astype(np.int64)
    required = ["row_index", "prefix_id", "decision_id", "trajectory_id", "prefix_token_index", "progress", *Q_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Activation metadata missing columns: {missing}")
    q = df[Q_COLS].to_numpy(dtype=np.float64)
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("q columns are not normalized within tolerance")
    df = df.reset_index(drop=True).copy()
    df["true_q_top_action"] = compute_true_q_top(df)
    if "top_action" not in df.columns:
        df["top_action"] = df["true_q_top_action"]
    if "strict_before_any_action_mention" not in df.columns and "is_before_first_action_mention" in df.columns:
        df["strict_before_any_action_mention"] = df["is_before_first_action_mention"].fillna(False).astype(bool)
    if "strict_before_final_action_mention" not in df.columns and "is_before_first_final_action_mention" in df.columns:
        df["strict_before_final_action_mention"] = df["is_before_first_final_action_mention"].fillna(False).astype(bool)
    return df


def feature_path(features_dir: str | Path, layer: int, representation: str) -> Path:
    path = Path(features_dir) / f"layer_{int(layer):02d}_{representation}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def inspect_feature_array(path: str | Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activation array: {path}; got {arr.shape}")
    return {"path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype), "size_bytes": int(Path(path).stat().st_size)}


def discover_feature_arrays(features_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(features_dir).glob("layer_*_*.npy")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        representation = "_".join(parts[2:])
        info = inspect_feature_array(path)
        rows.append({"layer": layer, "representation": representation, **info})
    return rows


def load_activation_matrix(metadata: pd.DataFrame, features_dir: str | Path, layer: int, representation: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = feature_path(features_dir, layer, representation)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activation array: {path}; got {arr.shape}")
    rows = metadata["row_index"].to_numpy(dtype=np.int64)
    if rows.min(initial=0) < 0 or rows.max(initial=0) >= arr.shape[0]:
        raise ValueError(f"row_index out of bounds for {path}: row_index range=({rows.min()}, {rows.max()}), array rows={arr.shape[0]}")
    x = np.asarray(arr[rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Activation matrix contains NaN or inf: {path}")
    info = inspect_feature_array(path)
    info.update({"layer": int(layer), "representation": representation})
    return x, info


def load_splits(path: str | Path, n_rows: int, group_col: str, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Split file must contain a list: {path}")
    splits: list[dict[str, Any]] = []
    for rec in raw:
        sid = int(rec.get("split_id", rec.get("fold", len(splits))))
        train = np.asarray(rec.get("train_indices", rec.get("train_idx")), dtype=np.int64)
        test = np.asarray(rec.get("test_indices", rec.get("test_idx")), dtype=np.int64)
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Split {sid} has empty train/test indices")
        if train.min() < 0 or test.min() < 0 or train.max() >= n_rows or test.max() >= n_rows:
            raise ValueError(f"Split {sid} indices are out of bounds for n_rows={n_rows}")
        train_groups = set(metadata.iloc[train][group_col].astype(str))
        test_groups = set(metadata.iloc[test][group_col].astype(str))
        if train_groups & test_groups:
            raise ValueError(f"Trajectory leakage in split {sid}: {sorted(train_groups & test_groups)[:5]}")
        splits.append({"split_id": sid, "train_idx": train, "test_idx": test, "n_train_groups": len(train_groups), "n_test_groups": len(test_groups)})
    return splits


def fit_standardizer(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32, copy=False)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(len(ACTIONS)):
        mask = y_true == i
        if mask.any():
            vals.append(float((y_pred[mask] == i).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for c in range(len(ACTIONS)):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def fit_action_probe(xz: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, dict[str, float]]:
    from sklearn.linear_model import RidgeClassifier

    model = RidgeClassifier(alpha=float(alpha), class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(xz[train_idx], y[train_idx])
        pred_train = model.predict(xz[train_idx])
        pred_test = model.predict(xz[test_idx])
    coef = np.asarray(model.coef_, dtype=np.float32)
    importance = np.linalg.norm(coef, axis=0).astype(np.float32)
    metrics = {
        "train_accuracy": float((pred_train == y[train_idx]).mean()),
        "train_balanced_accuracy": balanced_accuracy(y[train_idx], pred_train),
        "train_macro_f1": macro_f1(y[train_idx], pred_train),
        "test_accuracy": float((pred_test == y[test_idx]).mean()),
        "test_balanced_accuracy": balanced_accuracy(y[test_idx], pred_test),
        "test_macro_f1": macro_f1(y[test_idx], pred_test),
    }
    return importance, metrics


def fit_progress_importance(xz: np.ndarray, progress: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr

    model = Ridge(alpha=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(xz[train_idx], progress[train_idx])
        pred = model.predict(xz[train_idx])
    coef = np.abs(np.asarray(model.coef_, dtype=np.float32))
    try:
        corr = float(pearsonr(progress[train_idx], pred).statistic)
    except Exception:
        corr = float("nan")
    return coef, {"train_progress_pearson": corr}


def topk(importance: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), int(importance.shape[0]))
    return np.argsort(-importance)[:k].astype(np.int64)


def make_feature_view(xz: np.ndarray, family: str, selected: np.ndarray | None, importance: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if family == "all_dims":
        return xz, {"n_input_dimensions": int(xz.shape[1]), "selected_dimensions": "all", "weighted": False, "input_feature_kind": "activation_vectors_only"}
    if selected is None:
        raise ValueError(f"{family} requires selected dimensions")
    view = xz[:, selected]
    weighted = family == "probe_weighted_top_K"
    info: dict[str, Any] = {"n_input_dimensions": int(view.shape[1]), "selected_dimensions": selected.astype(int).tolist(), "weighted": False, "input_feature_kind": "activation_vectors_only"}
    if weighted:
        if importance is None:
            raise ValueError("probe_weighted_top_K requires importance")
        weights = importance[selected].astype(np.float32)
        weights = weights / np.clip(float(weights.mean()), EPS, None)
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        view = view * weights[None, :]
        info.update({"weighted": True, "weight_mean": float(weights.mean()), "weight_min": float(weights.min()), "weight_max": float(weights.max())})
    return view.astype(np.float32, copy=False), info


def fit_pca(x_view: np.ndarray, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, svd_solver="randomized", random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca.fit(x_view[train_idx])
        coords = pca.transform(x_view).astype(np.float32)
    return coords, {
        "embedding_method": "pca",
        "fit_rows": int(len(train_idx)),
        "explained_variance_ratio_1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_2": float(pca.explained_variance_ratio_[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_[:2].sum()),
    }


def fit_umap(x_view: np.ndarray, train_idx: np.ndarray, seed: int, max_train_rows: int = 8000) -> tuple[np.ndarray, dict[str, Any]]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
    import umap

    rng = np.random.default_rng(int(seed))
    fit_idx = np.asarray(train_idx, dtype=np.int64)
    if len(fit_idx) > int(max_train_rows):
        fit_idx = rng.choice(fit_idx, size=int(max_train_rows), replace=False)
        fit_idx.sort()
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reducer.fit(x_view[fit_idx])
        coords = reducer.transform(x_view).astype(np.float32)
    return coords, {"embedding_method": "umap", "fit_rows": int(len(fit_idx)), "umap_n_neighbors": 15, "umap_min_dist": 0.1, "umap_metric": "cosine"}


def centroid_separation_ratio(coords: np.ndarray, labels: np.ndarray) -> float:
    centroids = []
    within = []
    for action in ACTIONS:
        mask = labels == action
        if mask.sum() < 2:
            continue
        arr = coords[mask]
        centroid = arr.mean(axis=0)
        centroids.append(centroid)
        within.extend(np.linalg.norm(arr - centroid, axis=1).tolist())
    if len(centroids) < 2 or not within:
        return float("nan")
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(between) / np.clip(float(np.mean(within)), EPS, None))


def embedding_quality(coords: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, strict_mask: np.ndarray | None = None, high_mask: np.ndarray | None = None) -> dict[str, float]:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier

    labels = np.asarray(ACTIONS, dtype=object)[y]
    test_coords = coords[test_idx]
    test_labels = labels[test_idx]
    out: dict[str, float] = {}
    if len(np.unique(test_labels)) > 1 and len(test_idx) >= 10:
        sample = np.arange(len(test_idx))
        if len(sample) > 10000:
            sample = np.random.default_rng(seed).choice(sample, size=10000, replace=False)
        try:
            out["test_silhouette_true_q_top"] = float(silhouette_score(test_coords[sample], test_labels[sample]))
        except Exception:
            out["test_silhouette_true_q_top"] = float("nan")
    out["test_centroid_separation_ratio"] = centroid_separation_ratio(test_coords, test_labels)
    try:
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(coords[train_idx], y[train_idx])
        pred = knn.predict(test_coords)
        out["test_knn_accuracy"] = float((pred == y[test_idx]).mean())
        out["test_knn_balanced_accuracy"] = balanced_accuracy(y[test_idx], pred)
        out["test_knn_macro_f1"] = macro_f1(y[test_idx], pred)
    except Exception:
        out["test_knn_accuracy"] = float("nan")
        out["test_knn_balanced_accuracy"] = float("nan")
        out["test_knn_macro_f1"] = float("nan")
    for subset_name, mask in [("strict_pre_action", strict_mask), ("high_commitment", high_mask)]:
        if mask is None:
            continue
        local = mask[test_idx].astype(bool)
        out[f"{subset_name}_n_test_rows"] = int(local.sum())
        if local.sum() >= 10 and len(np.unique(test_labels[local])) > 1:
            try:
                out[f"{subset_name}_test_silhouette_true_q_top"] = float(silhouette_score(test_coords[local], test_labels[local]))
            except Exception:
                out[f"{subset_name}_test_silhouette_true_q_top"] = float("nan")
    return out


def metadata_for_output(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    cols = [c for c in META_KEEP_COLS if c in df.columns]
    return df.iloc[indices][cols].copy()


def validate_no_forbidden_feature_names(names: Iterable[str]) -> None:
    bad = sorted(set(map(str, names)) & FORBIDDEN_INPUT_COLUMNS)
    if bad:
        raise ValueError(f"Forbidden columns listed as embedding input features: {bad}")


def available_layers_representations(feature_infos: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    return sorted({int(x["layer"]) for x in feature_infos}), sorted({str(x["representation"]) for x in feature_infos})
