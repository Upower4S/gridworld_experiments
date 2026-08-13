"""Utilities for probe-guided activation PCA/UMAP embeddings.

The functions in this module enforce the central guardrail for this experiment:
linear probes may guide activation dimensions, but embedding coordinates must be
fit from activation feature matrices, never from probe logits, source scores,
stacker scores, q values, or metadata columns.
"""

from __future__ import annotations

import json
import math
import re
import random
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{a}" for a in ACTIONS]
DEFAULT_SEED = 42
EPS = 1e-12

FORBIDDEN_SCORE_RE = re.compile(
    r"^(score|stacker_score|source_score|mean100_score|prefixmean_score|segment_score|pairwise_score)_(UP|DOWN|LEFT|RIGHT)$"
)
FORBIDDEN_WILDCARD_RE = re.compile(r"(^|_)(score|logit|prob|q)_(UP|DOWN|LEFT|RIGHT)$")

METADATA_KEEP_COLS = [
    "row_id",
    "row_index",
    "prefix_id",
    "decision_id",
    "trajectory_id",
    "step_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    "top_action",
    "true_q_top_action",
    *Q_COLS,
    "entropy",
    "entropy_commitment",
    "commitment_to_final",
    "margin_commitment",
    "num_cot_tokens",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
    "first_action_mention_token_index",
    "first_final_action_mention_token_index",
    "split_role",
    "split_id",
]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_markdown(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n")


def read_parquet_robust(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as first_exc:
        try:
            return pd.read_parquet(path, columns=columns, engine="fastparquet")
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not read parquet {path} with pyarrow or fastparquet. "
                f"pyarrow error: {first_exc!r}; fastparquet error: {second_exc!r}"
            ) from second_exc


def parse_csv_ints(text: str) -> list[int]:
    if str(text).strip().lower() == "all":
        return discover_layers(Path("outputs/activation/features"))
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def discover_layers(features_dir: Path, representation: str | None = None) -> list[int]:
    pattern = f"layer_*_{representation}.npy" if representation else "layer_*_*.npy"
    layers = set()
    for path in Path(features_dir).glob(pattern):
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                layers.add(int(parts[1]))
            except ValueError:
                pass
    return sorted(layers)


def forbidden_score_columns(columns: Iterable[str]) -> list[str]:
    bad = []
    for col in columns:
        if FORBIDDEN_SCORE_RE.match(str(col)):
            bad.append(str(col))
    return bad


def assert_no_score_columns_as_features(feature_columns: Iterable[str]) -> None:
    bad = forbidden_score_columns(feature_columns)
    if bad:
        raise ValueError(
            "Forbidden score/probe columns were requested as embedding input features: "
            f"{bad}. PCA/UMAP inputs must be activation-derived arrays only."
        )


def action_indices(values: Iterable[str]) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    series = pd.Series(values).astype(str)
    out = series.map(mapping)
    if out.isna().any():
        bad = sorted(series[out.isna()].unique())
        raise ValueError(f"Unknown action labels: {bad}")
    return out.to_numpy(dtype=np.int64)


def compute_true_q_top(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in Q_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing q columns: {missing}")
    q = df[Q_COLS].to_numpy(dtype=np.float64)
    q = q / np.clip(q.sum(axis=1, keepdims=True), EPS, None)
    return pd.Series(np.asarray(ACTIONS, dtype=object)[np.argmax(q, axis=1)], index=df.index)


def merge_behavior_columns(metadata: pd.DataFrame, beliefs_path: Path | None) -> pd.DataFrame:
    df = metadata.copy()
    if beliefs_path is None or not Path(beliefs_path).exists():
        if "true_q_top_action" not in df.columns:
            df["true_q_top_action"] = compute_true_q_top(df)
        return df

    belief_cols = [
        "row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress",
        "agent_action", "top_action", *Q_COLS, "entropy", "entropy_commitment", "commitment_to_final",
        "margin_commitment", "num_cot_tokens", "strict_before_any_action_mention",
        "strict_before_final_action_mention", "first_action_mention_token_index",
        "first_final_action_mention_token_index", "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    beliefs_all = read_parquet_robust(beliefs_path)
    available = [c for c in belief_cols if c in beliefs_all.columns]
    beliefs = beliefs_all[available].copy()
    if "prefix_id" in df.columns and "prefix_id" in beliefs.columns:
        keys = ["prefix_id"]
    elif all(c in df.columns for c in ["decision_id", "prefix_token_index"]) and all(c in beliefs.columns for c in ["decision_id", "prefix_token_index"]):
        keys = ["decision_id", "prefix_token_index"]
    else:
        raise ValueError("Could not find stable keys to join behavior beliefs to activation metadata")
    beliefs = beliefs.drop_duplicates(keys)
    merged = df.merge(beliefs, on=keys, how="left", suffixes=("", "__belief"))
    for col in belief_cols:
        bcol = f"{col}__belief"
        if bcol in merged.columns:
            if col in merged.columns:
                merged[col] = merged[bcol].combine_first(merged[col])
            else:
                merged[col] = merged[bcol]
            merged = merged.drop(columns=[bcol])
    if "row_id" not in merged.columns:
        if "row_index" in merged.columns:
            merged.insert(0, "row_id", merged["row_index"].to_numpy(dtype=np.int64))
        else:
            merged.insert(0, "row_id", np.arange(len(merged), dtype=np.int64))
    merged["true_q_top_action"] = compute_true_q_top(merged)
    if "top_action" not in merged.columns:
        merged["top_action"] = merged["true_q_top_action"]
    return merged


def load_metadata(metadata_path: Path, beliefs_path: Path | None = None) -> pd.DataFrame:
    df = read_parquet_robust(metadata_path)
    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = np.arange(len(df), dtype=np.int64)
    df = merge_behavior_columns(df, beliefs_path)
    required = ["row_index", "decision_id", "trajectory_id", "prefix_token_index", "progress", *Q_COLS, "true_q_top_action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Activation metadata after behavior merge is missing required columns: {missing}")
    return df.reset_index(drop=True)


def make_group_splits(df: pd.DataFrame, n_splits: int, seed: int, group_col: str = "trajectory_id") -> list[dict[str, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    groups = df[group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Need at least two groups for grouped split")
    if len(unique) < n_splits:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return [{"split_id": 0, "train_idx": tr, "test_idx": te} for tr, te in splitter.split(df, groups=groups)]
    splitter = GroupKFold(n_splits=n_splits)
    return [
        {"split_id": i, "train_idx": tr, "test_idx": te}
        for i, (tr, te) in enumerate(splitter.split(df, groups=groups))
    ]


def add_split_role(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, split_id: int) -> pd.DataFrame:
    out = df.copy()
    role = np.full(len(out), "unused", dtype=object)
    role[train_idx] = "train"
    role[test_idx] = "test"
    out["split_role"] = role
    out["split_id"] = int(split_id)
    return out


def activation_feature_path(features_dir: Path, layer: int, representation: str) -> Path:
    candidates = [
        Path(features_dir) / f"layer_{int(layer):02d}_{representation}.npy",
        Path(features_dir) / f"layer_{int(layer)}_{representation}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No activation feature file found for layer={layer} representation={representation} under {features_dir}")


def estimate_feature_gb(n_rows: int, hidden_dim: int, n_layers: int) -> float:
    return float(n_rows) * float(hidden_dim) * float(n_layers) * 4.0 / (1024.0 ** 3)


def load_activation_features(
    metadata: pd.DataFrame,
    features_dir: Path,
    layers: list[int],
    representation: str,
    *,
    max_feature_gb: float = 8.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not layers:
        raise ValueError("No layers requested")
    row_idx = metadata["row_index"].to_numpy(dtype=np.int64)
    arrays = []
    infos = []
    for layer in layers:
        path = activation_feature_path(features_dir, layer, representation)
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D activation matrix for {path}, got shape {arr.shape}")
        if row_idx.max(initial=0) >= arr.shape[0] or row_idx.min(initial=0) < 0:
            raise ValueError(f"row_index values are outside activation matrix bounds for {path}: shape={arr.shape}")
        infos.append({"layer": int(layer), "representation": representation, "path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype)})
        arrays.append(arr)
    hidden = int(arrays[0].shape[1])
    for arr in arrays:
        if int(arr.shape[1]) != hidden:
            raise ValueError("All selected activation layers must have the same hidden dimension")
    est = estimate_feature_gb(len(metadata), hidden, len(arrays))
    if est > float(max_feature_gb):
        raise MemoryError(f"Requested activation matrix is estimated at {est:.2f} GiB; increase --max_feature_gb or use fewer layers")
    parts = [np.asarray(arr[row_idx], dtype=np.float32) for arr in arrays]
    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False) if len(parts) > 1 else parts[0].astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Activation feature matrix contains NaN or infinity")
    return x, infos


def standardize_from_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    xz = ((x - mean) / scale).astype(np.float32, copy=False)
    return xz, mean, scale


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


def fit_linear_probe(xz: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, *, alpha: float = 1.0) -> tuple[Any, np.ndarray, dict[str, float]]:
    from sklearn.linear_model import RidgeClassifier

    model = RidgeClassifier(alpha=float(alpha), class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(xz[train_idx], y[train_idx])
        pred_train = model.predict(xz[train_idx])
        pred_test = model.predict(xz[test_idx])
    coef = np.asarray(model.coef_, dtype=np.float32)
    if coef.ndim != 2 or coef.shape[1] != xz.shape[1]:
        raise RuntimeError(f"Unexpected probe coefficient shape {coef.shape}; expected (*, {xz.shape[1]})")
    importance = np.linalg.norm(coef, axis=0).astype(np.float32)
    metrics = {
        "probe_train_accuracy": float((pred_train == y[train_idx]).mean()),
        "probe_train_balanced_accuracy": balanced_accuracy(y[train_idx], pred_train),
        "probe_train_macro_f1": macro_f1(y[train_idx], pred_train),
        "probe_test_accuracy": float((pred_test == y[test_idx]).mean()),
        "probe_test_balanced_accuracy": balanced_accuracy(y[test_idx], pred_test),
        "probe_test_macro_f1": macro_f1(y[test_idx], pred_test),
    }
    return model, importance, metrics


def make_guided_view(xz: np.ndarray, importance: np.ndarray, method: str, top_k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = str(method)
    if method == "plain_activation_baseline":
        selected = np.arange(xz.shape[1], dtype=np.int64)
        return xz, selected, {"guidance": "none", "note": "baseline PCA/UMAP on standardized activation vectors"}
    order = np.argsort(-importance)
    if method == "probe_select_topk":
        k = min(int(top_k), xz.shape[1])
        selected = order[:k].astype(np.int64)
        return xz[:, selected].astype(np.float32, copy=False), selected, {"guidance": "probe_coefficient_topk", "top_k": int(k)}
    if method == "probe_weighted_all":
        weights = importance.astype(np.float32).copy()
        if not np.isfinite(weights).all() or float(weights.max(initial=0.0)) <= 0:
            raise ValueError("Probe importances are invalid for weighted view")
        weights = weights / np.clip(float(np.mean(weights)), EPS, None)
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        selected = np.arange(xz.shape[1], dtype=np.int64)
        return (xz * weights[None, :]).astype(np.float32, copy=False), selected, {"guidance": "probe_coefficient_weighted_all", "weight_clip_max": 10.0}
    if method == "probe_weighted_topk":
        k = min(int(top_k), xz.shape[1])
        selected = order[:k].astype(np.int64)
        weights = importance[selected].astype(np.float32)
        weights = weights / np.clip(float(np.mean(weights)), EPS, None)
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        return (xz[:, selected] * weights[None, :]).astype(np.float32, copy=False), selected, {"guidance": "probe_coefficient_weighted_topk", "top_k": int(k), "weight_clip_max": 10.0}
    raise ValueError(f"Unknown guidance method: {method}")


def fit_pca_embedding(x_view: np.ndarray, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any], Any]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, svd_solver="randomized", random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca.fit(x_view[train_idx])
        coords = pca.transform(x_view).astype(np.float32)
    info = {
        "embedding_method": "pca",
        "fit_rows": int(len(train_idx)),
        "explained_variance_ratio_1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_2": float(pca.explained_variance_ratio_[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_[:2].sum()),
    }
    return coords, info, pca


def fit_umap_embedding(x_view: np.ndarray, train_idx: np.ndarray, seed: int, *, max_train_rows: int, n_neighbors: int, min_dist: float, metric: str) -> tuple[np.ndarray, dict[str, Any], Any]:
    try:
        import umap
    except Exception as exc:
        raise RuntimeError("UMAP was requested but umap-learn is not installed. Install with: pip install umap-learn") from exc
    rng = np.random.default_rng(int(seed))
    fit_idx = np.asarray(train_idx, dtype=np.int64)
    if len(fit_idx) > int(max_train_rows):
        fit_idx = rng.choice(fit_idx, size=int(max_train_rows), replace=False)
        fit_idx.sort()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
    )
    reducer.fit(x_view[fit_idx])
    coords = reducer.transform(x_view).astype(np.float32)
    info = {"embedding_method": "umap", "fit_rows": int(len(fit_idx)), "n_neighbors": int(n_neighbors), "min_dist": float(min_dist), "metric": str(metric)}
    return coords, info, reducer


def metric_rows_for_embedding(df: pd.DataFrame, x_col: str, y_col: str, group_col: str = "true_q_top_action") -> dict[str, float]:
    from sklearn.metrics import silhouette_score

    out: dict[str, float] = {}
    coords = df[[x_col, y_col]].to_numpy(dtype=np.float32)
    labels = df[group_col].astype(str).to_numpy() if group_col in df.columns else None
    if labels is not None and len(np.unique(labels)) > 1 and len(df) >= 10:
        sample = np.arange(len(df))
        if len(sample) > 10000:
            rng = np.random.default_rng(DEFAULT_SEED)
            sample = rng.choice(sample, size=10000, replace=False)
        try:
            out[f"silhouette_{group_col}"] = float(silhouette_score(coords[sample], labels[sample]))
        except Exception:
            out[f"silhouette_{group_col}"] = float("nan")
    return out


def safe_metadata_for_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METADATA_KEEP_COLS if c in df.columns]
    return df[cols].copy()


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, part in df.groupby("split_role", dropna=False):
        rows.append({
            "split_role": str(role),
            "n_rows": int(len(part)),
            "n_trajectories": int(part["trajectory_id"].nunique()) if "trajectory_id" in part.columns else 0,
            "n_decisions": int(part["decision_id"].nunique()) if "decision_id" in part.columns else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compatibility API for scripts 60-66.
# These wrappers keep the exact names requested by the probe-guided embedding
# pipeline while reusing the stricter helpers above.


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def compute_q_labels(df: pd.DataFrame) -> pd.Series:
    return compute_true_q_top(df).rename("q_top_action")


def make_grouped_folds(df: pd.DataFrame, group_col: str = "trajectory_id", n_splits: int = 5, seed: int = DEFAULT_SEED) -> list[dict[str, np.ndarray]]:
    return make_group_splits(df, n_splits=n_splits, seed=seed, group_col=group_col)


def load_activation_view(metadata: pd.DataFrame, features_dir: str | Path, layer: int, representation: str) -> tuple[np.ndarray, dict[str, Any]]:
    x, infos = load_activation_features(metadata, Path(features_dir), [int(layer)], representation, max_feature_gb=32.0)
    return x, infos[0]


def fit_linear_action_probe(xz: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, *, alpha: float = 1.0):
    return fit_linear_probe(xz, y, train_idx, test_idx, alpha=alpha)


def extract_probe_importance(model: Any, method: str = "l2") -> np.ndarray:
    coef = np.asarray(getattr(model, "coef_", None), dtype=np.float32)
    if coef.ndim != 2:
        raise ValueError("Probe model does not expose a 2D coef_ matrix")
    method = str(method)
    if method == "l2":
        return np.linalg.norm(coef, axis=0).astype(np.float32)
    if method == "coef_std":
        return np.std(coef, axis=0).astype(np.float32)
    if method == "pairwise_contrast":
        vals = []
        for i in range(coef.shape[0]):
            for j in range(i + 1, coef.shape[0]):
                vals.append(np.abs(coef[i] - coef[j]))
        return np.mean(np.vstack(vals), axis=0).astype(np.float32) if vals else np.abs(coef[0]).astype(np.float32)
    raise ValueError(f"Unknown importance method: {method}")


def select_topk_features(importance: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(importance, dtype=np.float32))[: min(int(k), len(importance))].astype(np.int64)


def apply_feature_set(
    xz: np.ndarray,
    feature_set: str,
    *,
    selected: np.ndarray | None = None,
    importance: np.ndarray | None = None,
    gamma: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_set = str(feature_set)
    if feature_set == "all_dims":
        return xz.astype(np.float32, copy=False), {"n_input_dimensions": int(xz.shape[1]), "input_feature_kind": "activation_vectors_only"}
    if selected is None:
        raise ValueError(f"Feature set {feature_set} requires selected dimensions")
    selected = np.asarray(selected, dtype=np.int64)
    x_view = xz[:, selected].astype(np.float32, copy=False)
    info = {"n_input_dimensions": int(x_view.shape[1]), "input_feature_kind": "activation_vectors_only"}
    if feature_set.startswith("probe_weighted"):
        if importance is None:
            raise ValueError(f"Feature set {feature_set} requires importance weights")
        weights = np.asarray(importance, dtype=np.float32)[selected]
        weights = np.power(np.clip(weights, 0.0, None), float(gamma))
        weights = weights / np.clip(float(np.mean(weights)), EPS, None)
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        x_view = (x_view * weights[None, :]).astype(np.float32, copy=False)
        info["weight_gamma"] = float(gamma)
    return x_view, info


def residualize_progress(
    x: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    mode: str = "none",
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode)
    if mode == "none":
        return x, {"residualize": "none"}
    if mode != "progress_prefix":
        raise ValueError(f"Unknown residualization mode: {mode}")
    cov_cols = [c for c in ["progress", "prefix_token_index", "num_cot_tokens"] if c in metadata.columns]
    if not cov_cols:
        return x, {"residualize": mode, "covariates": []}
    c = metadata[cov_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    c = np.column_stack([np.ones(len(c), dtype=np.float64), c])
    coef, *_ = np.linalg.lstsq(c[np.asarray(train_idx, dtype=np.int64)], x[np.asarray(train_idx, dtype=np.int64)].astype(np.float64), rcond=None)
    resid = x.astype(np.float64) - c @ coef
    return resid.astype(np.float32), {"residualize": mode, "covariates": cov_cols}


def run_pca_embedding(x_view: np.ndarray, train_idx: np.ndarray, seed: int = DEFAULT_SEED):
    coords, info, model = fit_pca_embedding(x_view, train_idx, seed)
    info["embedding_coordinate_source"] = "PCA_on_activation_features"
    return coords, info, model


def run_umap_embedding(
    x_view: np.ndarray,
    train_idx: np.ndarray,
    seed: int = DEFAULT_SEED,
    *,
    max_train_rows: int = 20000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
):
    coords, info, model = fit_umap_embedding(
        x_view,
        train_idx,
        seed,
        max_train_rows=max_train_rows,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    info["embedding_coordinate_source"] = "UMAP_on_activation_features"
    return coords, info, model


def _centroid_separation(coords: np.ndarray, y: np.ndarray) -> float:
    centroids = []
    within = []
    for c in range(len(ACTIONS)):
        part = coords[y == c]
        if len(part) == 0:
            continue
        mu = part.mean(axis=0)
        centroids.append(mu)
        within.extend(np.linalg.norm(part - mu, axis=1).tolist())
    if len(centroids) < 2 or not within:
        return float("nan")
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(dists) / np.clip(np.mean(within), EPS, None))


def compute_embedding_metrics(
    coords: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: pd.DataFrame | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from scipy.stats import spearmanr

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    out: dict[str, float] = {}
    x_test = coords[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    if len(np.unique(y_test)) > 1 and len(x_test) > 10:
        sample = np.arange(len(x_test))
        if len(sample) > 10000:
            sample = np.random.default_rng(seed).choice(sample, size=10000, replace=False)
        try:
            out["silhouette_by_true_q_top_action"] = float(silhouette_score(x_test[sample], y_test[sample]))
        except Exception:
            out["silhouette_by_true_q_top_action"] = float("nan")
    else:
        out["silhouette_by_true_q_top_action"] = float("nan")
    out["branch_separation_ratio"] = _centroid_separation(x_test, y_test)
    centroids = {}
    for c in range(len(ACTIONS)):
        part = coords[train_idx][y_train == c]
        if len(part):
            centroids[c] = part.mean(axis=0)
    if centroids:
        keys = sorted(centroids)
        mat = np.vstack([centroids[k] for k in keys])
        pred = np.asarray(keys)[np.argmin(((x_test[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2), axis=1)]
        out["nearest_centroid_balanced_accuracy"] = float(balanced_accuracy_score(y_test, pred))
    else:
        out["nearest_centroid_balanced_accuracy"] = float("nan")
    try:
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(coords[train_idx], y_train)
        out["knn_2d_balanced_accuracy"] = float(balanced_accuracy_score(y_test, knn.predict(x_test)))
    except Exception:
        out["knn_2d_balanced_accuracy"] = float("nan")
    try:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(coords[train_idx], y_train)
        out["logistic_2d_balanced_accuracy"] = float(balanced_accuracy_score(y_test, lr.predict(x_test)))
    except Exception:
        out["logistic_2d_balanced_accuracy"] = float("nan")
    if metadata is not None and "progress" in metadata.columns:
        progress = pd.to_numeric(metadata["progress"], errors="coerce").fillna(0).to_numpy(dtype=float)
        try:
            out["spearman_coord_x_progress"] = float(spearmanr(coords[test_idx, 0], progress[test_idx]).statistic)
            out["spearman_coord_y_progress"] = float(spearmanr(coords[test_idx, 1], progress[test_idx]).statistic)
        except Exception:
            out["spearman_coord_x_progress"] = float("nan")
            out["spearman_coord_y_progress"] = float("nan")
        try:
            reg = LinearRegression().fit(coords[train_idx], progress[train_idx])
            out["r2_predicting_progress_from_2d_embedding"] = float(reg.score(coords[test_idx], progress[test_idx]))
        except Exception:
            out["r2_predicting_progress_from_2d_embedding"] = float("nan")
    return out


def save_scatter(path: str | Path, df: pd.DataFrame, x: str, y: str, color: str, title: str) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 7))
    if color in df.columns and df[color].dtype == object:
        for val, part in df.groupby(color, dropna=False):
            ax.scatter(part[x], part[y], s=8, alpha=0.55, label=str(val), edgecolors="none", rasterized=True)
        ax.legend(frameon=True, fontsize=8)
    else:
        sc = ax.scatter(df[x], df[y], c=df[color] if color in df.columns else None, s=8, alpha=0.55, cmap="viridis", edgecolors="none", rasterized=True)
        if color in df.columns:
            fig.colorbar(sc, ax=ax, label=color)
    ax.set_xlabel("dimension 1 from PCA/UMAP on activation features")
    ax.set_ylabel("dimension 2 from PCA/UMAP on activation features")
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
