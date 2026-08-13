#!/usr/bin/env python3
"""Quantify alignment between activation geometry and behavior commitment geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "spar_cache"
    _xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg_cache)
if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "spar_matplotlib"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: scipy. Install scipy for RSA and rank correlations.") from exc


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}
B_CLR_COLS = [f"b_clr_{action}" for action in ACTIONS]
Q_COLS = [f"q_{action}" for action in ACTIONS]
SUBSET_NAMES = [
    "all",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "high_commitment",
    "low_commitment",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_repo_root(path: Path) -> Path:
    path = path.resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "outputs").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd().resolve()


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def feature_file(features_dir: Path, layer: int, representation: str) -> Path:
    return features_dir / f"layer_{layer:02d}_{representation}.npy"


def discover_layers(features_dir: Path, representation: str) -> list[int]:
    layers: set[int] = set()
    for path in features_dir.glob(f"layer_*_{representation}.npy"):
        parts = path.stem.split("_")
        if len(parts) >= 3 and parts[0] == "layer":
            try:
                layers.add(int(parts[1]))
            except ValueError:
                continue
    return sorted(layers)


def recursively_find_layer(obj: Any) -> int | None:
    if isinstance(obj, dict):
        if "layer" in obj and obj["layer"] is not None:
            try:
                return int(obj["layer"])
            except (TypeError, ValueError):
                pass
        preferred = [
            "best_layer_behavior_clr_prediction",
            "best_layer_commitment_prediction",
            "best_layer_strict_pre_action_decoding",
            "best_representation",
        ]
        for key in preferred:
            if key in obj:
                found = recursively_find_layer(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = recursively_find_layer(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursively_find_layer(value)
            if found is not None:
                return found
    return None


def selected_layer_from_activation_manifold(path: Path, representation: str) -> int | None:
    if not path.exists():
        return None
    try:
        cols = pd.read_parquet(path, columns=["layer", "representation"])
    except Exception:
        try:
            cols = pd.read_parquet(path, columns=["layer"])
        except Exception:
            return None
    if "representation" in cols.columns:
        cols = cols[cols["representation"].astype(str).eq(representation)]
    values = sorted(pd.Series(cols["layer"]).dropna().astype(int).unique().tolist())
    if len(values) == 1:
        return values[0]
    return None


def select_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    activation_manifold_path: Path,
    features_dir: Path,
    representation: str,
) -> tuple[int, str]:
    available_layers = discover_layers(features_dir, representation)
    if not available_layers:
        raise SystemExit(f"No feature arrays found for representation {representation!r} in {features_dir}")
    if requested_layer is not None:
        if requested_layer not in available_layers:
            raise SystemExit(f"Requested layer {requested_layer} unavailable. Available layers: {available_layers}")
        return requested_layer, "explicit --layer"
    if config_path is not None and config_path.exists():
        try:
            layer = recursively_find_layer(json.loads(config_path.read_text()))
            if layer in available_layers:
                return int(layer), f"best_activation_config: {config_path}"
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; trying activation manifold layer.")
    layer = selected_layer_from_activation_manifold(activation_manifold_path, representation)
    if layer in available_layers:
        return int(layer), f"activation manifold layer from {activation_manifold_path}"
    index = int(round(0.75 * (len(available_layers) - 1)))
    return available_layers[index], "mid-late fallback from available feature layers"


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    result = spearmanr(x[mask], y[mask])
    return float(result.correlation) if result.correlation is not None else float("nan")


def group_splits(df: pd.DataFrame, group_col: str, n_splits: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    groups = df[group_col].to_numpy()
    unique_groups = pd.Series(groups).dropna().unique()
    if len(unique_groups) < 2:
        return [], "fewer than two groups"
    if len(unique_groups) <= 5:
        splitter = LeaveOneGroupOut()
        return list(splitter.split(df, groups=groups)), "LeaveOneGroupOut"
    splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
    return list(splitter.split(df, groups=groups)), f"GroupKFold({min(n_splits, len(unique_groups))})"


def subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
    if "is_before_first_action_mention" in df.columns:
        masks["strict_before_any_action_mention"] = df["is_before_first_action_mention"].fillna(True).astype(bool).to_numpy()
    else:
        masks["strict_before_any_action_mention"] = np.ones(n, dtype=bool)
    if "is_before_first_final_action_mention" in df.columns:
        masks["strict_before_final_action_mention"] = df["is_before_first_final_action_mention"].fillna(True).astype(bool).to_numpy()
    else:
        masks["strict_before_final_action_mention"] = np.ones(n, dtype=bool)
    if "commitment_to_final" in df.columns:
        c = pd.to_numeric(df["commitment_to_final"], errors="coerce").to_numpy(dtype=float)
        masks["high_commitment"] = np.isfinite(c) & (c >= 0.8)
        masks["low_commitment"] = np.isfinite(c) & (c <= 0.4)
    else:
        masks["high_commitment"] = np.zeros(n, dtype=bool)
        masks["low_commitment"] = np.zeros(n, dtype=bool)
    return masks


def valid_base_mask(df: pd.DataFrame, group_col: str) -> np.ndarray:
    mask = np.asarray(df[group_col].notna().to_numpy(), dtype=bool).copy()
    for col in B_CLR_COLS:
        mask &= np.asarray(pd.to_numeric(df[col], errors="coerce").notna().to_numpy(), dtype=bool)
    return mask


def true_commitment_from_q(df: pd.DataFrame) -> np.ndarray:
    values = np.full(len(df), np.nan, dtype=float)
    for idx, action in enumerate(ACTIONS):
        mask = df["agent_action"].astype(str).eq(action).to_numpy() if "agent_action" in df.columns else np.zeros(len(df), dtype=bool)
        col = f"q_{action}"
        if col in df.columns:
            values[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce")
    if "commitment_to_final" in df.columns:
        fallback = pd.to_numeric(df["commitment_to_final"], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, fallback)
    return values


def commitment_from_pred_q(df: pd.DataFrame, q: np.ndarray) -> np.ndarray:
    values = np.full(len(df), np.nan, dtype=float)
    if "agent_action" not in df.columns:
        return values
    actions = df["agent_action"].astype(str).to_numpy()
    for idx, action in enumerate(ACTIONS):
        values[actions == action] = q[actions == action, idx]
    return values


def top_action_from_q(q: np.ndarray) -> np.ndarray:
    return np.asarray(ACTIONS, dtype=object)[np.argmax(q, axis=1)]


def ridge_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    group_col: str,
    n_splits: int,
    alpha: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    splits, split_name = group_splits(df, group_col, n_splits)
    if not splits:
        return None, {"status": "skipped", "skip_reason": split_name, "n_folds": 0}
    preds = np.full(y.shape, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        if np.asarray(fold_pred).ndim == 1:
            fold_pred = np.asarray(fold_pred).reshape(-1, 1)
        preds[test_idx] = fold_pred
    return preds, {"status": "ok", "cv_method": split_name, "n_folds": len(splits)}


def ridge_metrics(
    *,
    subset: str,
    layer: int,
    representation: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_info: dict[str, Any],
) -> dict[str, Any]:
    pred_q = stable_softmax(y_pred)
    true_commit = true_commitment_from_q(df)
    pred_commit = commitment_from_pred_q(df, pred_q)
    pred_top = top_action_from_q(pred_q)
    r2_dims = r2_score(y_true, y_pred, multioutput="raw_values", force_finite=True)
    row: dict[str, Any] = {
        "analysis": "ridge_activation_to_behavior_clr",
        "subset": subset,
        "layer": layer,
        "representation": representation,
        "status": "ok",
        "skip_reason": "",
        "n_rows": len(df),
        "n_groups": int(df["_group"].nunique()),
        "n_folds": cv_info.get("n_folds", 0),
        "cv_method": cv_info.get("cv_method", ""),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2_mean": float(np.mean(r2_dims)),
        "predicted_commitment_pearson": pearson_corr(true_commit, pred_commit),
        "predicted_commitment_spearman": spearman_corr(true_commit, pred_commit),
        "pred_top_action_accuracy_vs_top_action": float(accuracy_score(df["top_action"].astype(str), pred_top)) if "top_action" in df.columns else float("nan"),
        "pred_top_action_accuracy_vs_agent_action": float(accuracy_score(df["agent_action"].astype(str), pred_top)) if "agent_action" in df.columns else float("nan"),
    }
    for i, action in enumerate(ACTIONS):
        row[f"r2_{action}"] = float(r2_dims[i])
        row[f"pearson_{action}"] = pearson_corr(y_true[:, i], y_pred[:, i])
    return row


def fit_latent_model(method: str, n_components: int, max_iter: int) -> Any:
    if method == "cca":
        return CCA(n_components=n_components, max_iter=max_iter, scale=False)
    return PLSRegression(n_components=n_components, max_iter=max_iter, scale=False)


def latent_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    subset: str,
    layer: int,
    representation: str,
    method: str,
    n_components: int,
    group_col: str,
    n_splits: int,
    max_iter: int,
) -> dict[str, Any]:
    splits, split_name = group_splits(df, group_col, n_splits)
    if not splits:
        return {
            "analysis": f"{method}_activation_behavior",
            "subset": subset,
            "layer": layer,
            "representation": representation,
            "status": "skipped",
            "skip_reason": split_name,
            "n_rows": len(df),
            "n_groups": int(df[group_col].nunique()),
        }
    preds = np.full(y.shape, np.nan, dtype=float)
    x_scores = np.full((len(df), n_components), np.nan, dtype=float)
    y_scores = np.full((len(df), n_components), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_x.fit_transform(X[train_idx])
        y_train = scaler_y.fit_transform(y[train_idx])
        X_test = scaler_x.transform(X[test_idx])
        model = fit_latent_model(method, n_components, max_iter)
        model.fit(X_train, y_train)
        pred_scaled = model.predict(X_test)
        preds[test_idx] = scaler_y.inverse_transform(pred_scaled)
        transformed = model.transform(X_test, scaler_y.transform(y[test_idx]))
        if isinstance(transformed, tuple):
            x_s, y_s = transformed
        else:
            x_s = transformed
            y_s = np.full_like(x_s, np.nan)
        x_scores[test_idx] = x_s[:, :n_components]
        y_scores[test_idx] = y_s[:, :n_components]
    r2_dims = r2_score(y, preds, multioutput="raw_values", force_finite=True)
    row: dict[str, Any] = {
        "analysis": f"{method}_activation_behavior",
        "subset": subset,
        "layer": layer,
        "representation": representation,
        "status": "ok",
        "skip_reason": "",
        "n_rows": len(df),
        "n_groups": int(df[group_col].nunique()),
        "n_folds": len(splits),
        "cv_method": split_name,
        "mse": float(mean_squared_error(y, preds)),
        "r2_mean": float(np.mean(r2_dims)),
    }
    for i, action in enumerate(ACTIONS):
        row[f"r2_{action}"] = float(r2_dims[i])
    for comp in range(n_components):
        row[f"latent_component_{comp + 1}_corr"] = pearson_corr(x_scores[:, comp], y_scores[:, comp])
    return row


def fit_full_latent_coordinates(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_components: int,
    max_iter: int,
) -> np.ndarray:
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    model = fit_latent_model(method, n_components, max_iter)
    model.fit(X_scaled, y_scaled)
    transformed = model.transform(X_scaled, y_scaled)
    if isinstance(transformed, tuple):
        x_scores = transformed[0]
    else:
        x_scores = transformed
    return np.asarray(x_scores[:, :n_components], dtype=float)


def affine_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.column_stack([train_x, np.ones(len(train_x))])
    coef, *_ = np.linalg.lstsq(x_aug, train_y, rcond=None)
    test_aug = np.column_stack([test_x, np.ones(len(test_x))])
    return test_aug @ coef, coef


def procrustes_cv(
    joined: pd.DataFrame,
    *,
    subset: str,
    layer: int,
    representation: str,
    group_col: str,
    n_splits: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = ["act_pca1", "act_pca2", "beh_pca1", "beh_pca2", group_col]
    missing = [col for col in required if col not in joined.columns]
    if missing:
        return ({"analysis": "procrustes_pca_alignment", "subset": subset, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": f"missing {missing}"}, pd.DataFrame())
    work = joined.dropna(subset=required).copy().reset_index(drop=True)
    if len(work) < 4:
        return ({"analysis": "procrustes_pca_alignment", "subset": subset, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": "not enough rows"}, pd.DataFrame())
    splits, split_name = group_splits(work, group_col, n_splits)
    if not splits:
        return ({"analysis": "procrustes_pca_alignment", "subset": subset, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": split_name}, pd.DataFrame())
    act_xy = work[["act_pca1", "act_pca2"]].to_numpy(dtype=float)
    beh_xy = work[["beh_pca1", "beh_pca2"]].to_numpy(dtype=float)
    pred = np.full_like(beh_xy, np.nan, dtype=float)
    nn_matches: list[bool] = []
    for train_idx, test_idx in splits:
        fold_pred, _ = affine_fit_predict(act_xy[train_idx], beh_xy[train_idx], act_xy[test_idx])
        pred[test_idx] = fold_pred
        if "agent_action" in work.columns:
            train_beh = beh_xy[train_idx]
            train_actions = work.iloc[train_idx]["agent_action"].astype(str).to_numpy()
            test_actions = work.iloc[test_idx]["agent_action"].astype(str).to_numpy()
            for p, true_action in zip(fold_pred, test_actions):
                nearest = int(np.argmin(np.linalg.norm(train_beh - p, axis=1)))
                nn_matches.append(train_actions[nearest] == true_action)
    ss_err = float(np.sum((pred - beh_xy) ** 2))
    ss_ref = float(np.sum((beh_xy - beh_xy.mean(axis=0, keepdims=True)) ** 2))
    out = work[[col for col in ["prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index"] if col in work.columns]].copy()
    out["aligned_act_beh_pca1"] = pred[:, 0]
    out["aligned_act_beh_pca2"] = pred[:, 1]
    row = {
        "analysis": "procrustes_pca_alignment",
        "subset": subset,
        "layer": layer,
        "representation": representation,
        "status": "ok",
        "skip_reason": "",
        "n_rows": len(work),
        "n_groups": int(work[group_col].nunique()),
        "n_folds": len(splits),
        "cv_method": split_name,
        "coordinate_corr_flat": pearson_corr(beh_xy.reshape(-1), pred.reshape(-1)),
        "coordinate_corr_dim1": pearson_corr(beh_xy[:, 0], pred[:, 0]),
        "coordinate_corr_dim2": pearson_corr(beh_xy[:, 1], pred[:, 1]),
        "procrustes_disparity": ss_err / ss_ref if ss_ref else float("nan"),
        "nearest_neighbor_action_consistency": float(np.mean(nn_matches)) if nn_matches else float("nan"),
    }
    return row, out


def rsa_metrics(
    X: np.ndarray,
    df: pd.DataFrame,
    *,
    subset: str,
    layer: int,
    representation: str,
    sample_size: int,
    random_state: int,
    max_scatter_pairs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(df) < 4:
        return ({"analysis": "rsa_pairwise_distances", "subset": subset, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": "not enough rows"}, pd.DataFrame())
    n = min(sample_size, len(df))
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(len(df), size=n, replace=False)) if n < len(df) else np.arange(len(df))
    X_sample = np.asarray(X[sample_idx], dtype=np.float32)
    y_sample = df.iloc[sample_idx][B_CLR_COLS].to_numpy(dtype=float)
    act_dist = pdist(X_sample, metric="euclidean")
    beh_dist = pdist(y_sample, metric="euclidean")
    if "progress" in df.columns:
        progress = pd.to_numeric(df.iloc[sample_idx]["progress"], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
        progress_dist = pdist(progress, metric="euclidean")
    else:
        progress_dist = np.full_like(act_dist, np.nan)
    if "agent_action" in df.columns:
        actions = df.iloc[sample_idx]["agent_action"].astype(str).to_numpy()
        same_action = np.asarray([a == b for i, a in enumerate(actions[:-1]) for b in actions[i + 1 :]], dtype=float)
    else:
        same_action = np.full_like(act_dist, np.nan)
    row = {
        "analysis": "rsa_pairwise_distances",
        "subset": subset,
        "layer": layer,
        "representation": representation,
        "status": "ok",
        "skip_reason": "",
        "n_rows": int(n),
        "n_groups": int(df["_group"].nunique()) if "_group" in df.columns else float("nan"),
        "n_pairs": int(len(act_dist)),
        "rsa_activation_behavior_spearman": spearman_corr(act_dist, beh_dist),
        "rsa_activation_progress_spearman": spearman_corr(act_dist, progress_dist),
        "rsa_activation_same_action_spearman": spearman_corr(act_dist, same_action),
    }
    if len(act_dist) > max_scatter_pairs:
        pair_idx = rng.choice(len(act_dist), size=max_scatter_pairs, replace=False)
    else:
        pair_idx = np.arange(len(act_dist))
    scatter = pd.DataFrame(
        {
            "activation_distance": act_dist[pair_idx],
            "behavior_distance": beh_dist[pair_idx],
            "progress_distance": progress_dist[pair_idx],
            "same_action": same_action[pair_idx],
            "subset": subset,
        }
    )
    return row, scatter


def base_prediction_frame(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cols = [
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
        "margin_commitment",
        "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    out = df[[col for col in cols if col in df.columns]].copy()
    for i, action in enumerate(ACTIONS):
        out[f"true_b_clr_{action}"] = y_true[:, i]
        out[f"pred_b_clr_{action}"] = y_pred[:, i]
    pred_q = stable_softmax(y_pred)
    for i, action in enumerate(ACTIONS):
        out[f"pred_q_{action}"] = pred_q[:, i]
    out["pred_commitment_to_final"] = commitment_from_pred_q(df, pred_q)
    out["pred_top_action"] = top_action_from_q(pred_q)
    out["true_commitment_to_final"] = true_commitment_from_q(df)
    return out


def plot_true_vs_pred_commitment(pred: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred["true_commitment_to_final"], pred["pred_commitment_to_final"], s=10, alpha=0.45, linewidths=0)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("True commitment_to_final")
    ax.set_ylabel("Predicted commitment_to_final")
    ax.set_title("Activation-predicted vs true commitment")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_behavior_pca(pred: pd.DataFrame, path: Path) -> None:
    y_pred = pred[[f"pred_b_clr_{action}" for action in ACTIONS]].to_numpy(dtype=float)
    coords = PCA(n_components=2, random_state=0).fit_transform(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    for action in ACTIONS:
        mask = pred["agent_action"].astype(str).eq(action).to_numpy() if "agent_action" in pred.columns else np.zeros(len(pred), dtype=bool)
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, color=ACTION_COLORS[action], label=action, linewidths=0)
    ax.set_xlabel("pred_beh_pca1")
    ax.set_ylabel("pred_beh_pca2")
    ax.set_title("Predicted behavior CLR from activations")
    ax.legend(title="agent_action")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latent_by_action(pred: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for action in ACTIONS:
        mask = pred["agent_action"].astype(str).eq(action).to_numpy() if "agent_action" in pred.columns else np.zeros(len(pred), dtype=bool)
        if mask.any():
            ax.scatter(pred.loc[mask, "act_beh_cca1"], pred.loc[mask, "act_beh_cca2"], s=10, alpha=0.6, color=ACTION_COLORS[action], label=action, linewidths=0)
    ax.set_xlabel("act_beh_cca1")
    ax.set_ylabel("act_beh_cca2")
    ax.set_title("Activation-behavior latent embedding by action")
    ax.legend(title="agent_action")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latent_by_commitment(pred: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(pred["act_beh_cca1"], pred["act_beh_cca2"], c=pred["commitment_to_final"], s=10, alpha=0.75, cmap="viridis", linewidths=0)
    fig.colorbar(scatter, ax=ax, label="commitment_to_final")
    ax.set_xlabel("act_beh_cca1")
    ax.set_ylabel("act_beh_cca2")
    ax.set_title("Activation-behavior latent embedding by commitment")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_procrustes_alignment(pred: pd.DataFrame, path: Path, max_lines: int, random_state: int) -> None:
    required = ["beh_pca1", "beh_pca2", "aligned_act_beh_pca1", "aligned_act_beh_pca2"]
    fig, ax = plt.subplots(figsize=(7, 6))
    if any(col not in pred.columns for col in required):
        ax.text(0.5, 0.5, "Procrustes coordinates unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        rng = np.random.default_rng(random_state)
        n = len(pred)
        sample = rng.choice(n, size=min(max_lines, n), replace=False) if n else []
        ax.scatter(pred["beh_pca1"], pred["beh_pca2"], s=8, alpha=0.25, color="#4C78A8", label="behavior PCA", linewidths=0)
        ax.scatter(pred["aligned_act_beh_pca1"], pred["aligned_act_beh_pca2"], s=8, alpha=0.25, color="#F58518", label="aligned activation PCA", linewidths=0)
        for idx in sample:
            ax.plot(
                [pred.iloc[idx]["beh_pca1"], pred.iloc[idx]["aligned_act_beh_pca1"]],
                [pred.iloc[idx]["beh_pca2"], pred.iloc[idx]["aligned_act_beh_pca2"]],
                color="black",
                alpha=0.12,
                linewidth=0.5,
            )
        ax.legend()
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.set_title("Behavior PCA vs aligned activation PCA")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rsa(scatter: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if scatter.empty:
        ax.text(0.5, 0.5, "RSA scatter unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(scatter["activation_distance"], scatter["behavior_distance"], s=7, alpha=0.25, linewidths=0)
    ax.set_xlabel("Activation pairwise distance")
    ax.set_ylabel("Behavior CLR pairwise distance")
    ax.set_title("RSA: activation vs behavior distance")
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    layer: int,
    representation: str,
    layer_source: str,
    feature_path: Path,
    behavior_manifold_rows: int,
    activation_manifold_rows: int,
    figures: list[Path],
    latent_method: str,
) -> None:
    lines: list[str] = []
    lines.append("# Activation-Behavior Alignment Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Prediction rows: {len(predictions):,}")
    lines.append(f"- Decisions: {predictions['decision_id'].nunique() if 'decision_id' in predictions.columns else 'NA'}")
    lines.append(f"- Trajectories: {predictions['trajectory_id'].nunique() if 'trajectory_id' in predictions.columns else 'NA'}")
    lines.append(f"- Activation manifold rows: {activation_manifold_rows:,}")
    lines.append(f"- Behavior manifold rows: {behavior_manifold_rows:,}")
    lines.append(f"- Selected layer: {layer}")
    lines.append(f"- Selected representation: `{representation}`")
    lines.append(f"- Layer source: {layer_source}")
    lines.append(f"- Feature file: `{feature_path}`")
    lines.append(f"- Latent method: `{latent_method}`")
    lines.append("")

    lines.append("## Key Metrics: All Prefixes\n")
    all_rows = metrics[(metrics["subset"].eq("all")) & (metrics["status"].eq("ok"))]
    for analysis in ["ridge_activation_to_behavior_clr", f"{latent_method}_activation_behavior", "procrustes_pca_alignment", "rsa_pairwise_distances"]:
        row_df = all_rows[all_rows["analysis"].eq(analysis)]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        lines.append(f"### {analysis}\n")
        for col in [
            "r2_mean",
            "r2_UP",
            "r2_DOWN",
            "r2_LEFT",
            "r2_RIGHT",
            "mse",
            "predicted_commitment_pearson",
            "pred_top_action_accuracy_vs_top_action",
            "pred_top_action_accuracy_vs_agent_action",
            "latent_component_1_corr",
            "latent_component_2_corr",
            "coordinate_corr_flat",
            "procrustes_disparity",
            "nearest_neighbor_action_consistency",
            "rsa_activation_behavior_spearman",
            "rsa_activation_progress_spearman",
            "rsa_activation_same_action_spearman",
        ]:
            if col in row and pd.notna(row[col]):
                lines.append(f"- `{col}`: {fmt(row[col])}")
        lines.append("")

    lines.append("## Subset Metrics\n")
    display_cols = [
        "analysis",
        "subset",
        "status",
        "n_rows",
        "n_groups",
        "r2_mean",
        "predicted_commitment_pearson",
        "pred_top_action_accuracy_vs_top_action",
        "coordinate_corr_flat",
        "procrustes_disparity",
        "rsa_activation_behavior_spearman",
    ]
    display_cols = [col for col in display_cols if col in metrics.columns]
    lines.append(metrics[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation Caveats\n")
    lines.append("- Ridge and PLS/CCA metrics use grouped splits, so prefixes from the same group are not split across train and test folds.")
    lines.append("- Procrustes alignment evaluates whether the existing 2D unsupervised PCA views can be linearly aligned; it is not equivalent to high-dimensional decoding.")
    lines.append("- RSA is computed on a sampled subset of pairs to keep pairwise distances tractable.")
    lines.append("- Behavior CLR vectors come from model action scores, not from one-hot labels.")
    lines.append("")

    lines.append("## Figures\n")
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.append("")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--activation_manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--behavior_manifold", default="outputs/behavior_manifold.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--output", default="outputs/activation/activation_behavior_alignment_metrics.csv")
    parser.add_argument("--predictions_output", default="outputs/activation/activation_behavior_alignment_predictions.parquet")
    parser.add_argument("--report", default="outputs/activation/reports/activation_behavior_alignment_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/alignment")
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--group_by", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--latent_method", choices=["pls", "cca"], default="pls")
    parser.add_argument("--latent_components", type=int, default=2)
    parser.add_argument("--latent_max_iter", type=int, default=500)
    parser.add_argument("--rsa_sample_size", type=int, default=1500)
    parser.add_argument("--rsa_max_scatter_pairs", type=int, default=50000)
    parser.add_argument("--procrustes_plot_pairs", type=int, default=250)
    parser.add_argument("--random_state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    repo_root = infer_repo_root(metadata_path)
    metadata_path = resolve_path(metadata_path, repo_root)
    activation_manifold_path = resolve_path(args.activation_manifold, repo_root)
    behavior_manifold_path = resolve_path(args.behavior_manifold, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root) if args.config else None
    output_path = resolve_path(args.output, repo_root)
    predictions_path = resolve_path(args.predictions_output, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    meta = pd.read_parquet(metadata_path)
    activation_manifold = pd.read_parquet(activation_manifold_path)
    behavior_manifold = pd.read_parquet(behavior_manifold_path)
    if args.group_by not in meta.columns:
        fallback = "decision_id" if args.group_by == "trajectory_id" and "decision_id" in meta.columns else None
        if fallback is None:
            raise SystemExit(f"Missing group column {args.group_by!r}")
        print(f"WARNING: missing {args.group_by}; using {fallback}")
        group_col = fallback
    else:
        group_col = args.group_by

    missing_behavior = [col for col in B_CLR_COLS if col not in meta.columns]
    if missing_behavior:
        raise SystemExit(f"Metadata missing behavior CLR columns: {missing_behavior}")

    layer, layer_source = select_layer(
        requested_layer=args.layer,
        config_path=config_path,
        activation_manifold_path=activation_manifold_path,
        features_dir=features_dir,
        representation=args.representation,
    )
    selected_feature_path = feature_file(features_dir, layer, args.representation)
    X_mmap = np.load(selected_feature_path, mmap_mode="r")
    if X_mmap.shape[0] != len(meta):
        raise SystemExit(f"Feature row mismatch: {X_mmap.shape[0]} rows but metadata has {len(meta)}")

    finite_x = np.isfinite(np.asarray(X_mmap[: min(10, len(meta))])).all()
    if not finite_x:
        print("WARNING: non-finite values detected in initial feature sample; full finite filtering is not implemented here.")

    base_mask = valid_base_mask(meta, group_col)
    if "alignment_status" in meta.columns:
        base_mask &= ~meta["alignment_status"].fillna("").astype(str).str.startswith("failed").to_numpy()
    work_meta = meta.loc[base_mask].copy().reset_index(drop=True)
    X_all = np.asarray(X_mmap[base_mask], dtype=np.float32)
    y_all = work_meta[B_CLR_COLS].to_numpy(dtype=float)
    work_meta["_group"] = work_meta[group_col].astype(str)

    masks = subset_masks(work_meta)
    metrics: list[dict[str, Any]] = []
    all_predictions: pd.DataFrame | None = None
    rsa_scatter_all = pd.DataFrame()

    for subset in SUBSET_NAMES:
        subset_mask = masks[subset]
        if subset_mask.sum() < 4:
            metrics.append({"analysis": "ridge_activation_to_behavior_clr", "subset": subset, "layer": layer, "representation": args.representation, "status": "skipped", "skip_reason": "not enough rows", "n_rows": int(subset_mask.sum())})
            metrics.append({"analysis": f"{args.latent_method}_activation_behavior", "subset": subset, "layer": layer, "representation": args.representation, "status": "skipped", "skip_reason": "not enough rows", "n_rows": int(subset_mask.sum())})
            metrics.append({"analysis": "rsa_pairwise_distances", "subset": subset, "layer": layer, "representation": args.representation, "status": "skipped", "skip_reason": "not enough rows", "n_rows": int(subset_mask.sum())})
            continue
        df_subset = work_meta.loc[subset_mask].copy().reset_index(drop=True)
        X_subset = X_all[subset_mask]
        y_subset = y_all[subset_mask]
        print(f"Running ridge and {args.latent_method.upper()} alignment for subset {subset} ({len(df_subset)} rows)...")
        ridge_pred, cv_info = ridge_cv_predict(
            X_subset,
            y_subset,
            df_subset.rename(columns={group_col: "_split_group"}) if False else df_subset,
            group_col="_group",
            n_splits=args.n_splits,
            alpha=args.ridge_alpha,
        )
        if ridge_pred is None:
            metrics.append({"analysis": "ridge_activation_to_behavior_clr", "subset": subset, "layer": layer, "representation": args.representation, "status": "skipped", "skip_reason": cv_info.get("skip_reason", "fit failed"), "n_rows": len(df_subset), "n_groups": int(df_subset["_group"].nunique())})
        else:
            metrics.append(ridge_metrics(subset=subset, layer=layer, representation=args.representation, df=df_subset, y_true=y_subset, y_pred=ridge_pred, cv_info=cv_info))
            if subset == "all":
                all_predictions = base_prediction_frame(df_subset, y_subset, ridge_pred)

        metrics.append(
            latent_cv_metrics(
                X_subset,
                y_subset,
                df_subset,
                subset=subset,
                layer=layer,
                representation=args.representation,
                method=args.latent_method,
                n_components=args.latent_components,
                group_col="_group",
                n_splits=args.n_splits,
                max_iter=args.latent_max_iter,
            )
        )

        rsa_row, rsa_scatter = rsa_metrics(
            X_subset,
            df_subset,
            subset=subset,
            layer=layer,
            representation=args.representation,
            sample_size=args.rsa_sample_size,
            random_state=args.random_state,
            max_scatter_pairs=args.rsa_max_scatter_pairs,
        )
        metrics.append(rsa_row)
        if subset == "all":
            rsa_scatter_all = rsa_scatter

    if all_predictions is None:
        raise SystemExit("All-prefix ridge prediction failed; cannot write predictions/figures.")

    print("Fitting full latent coordinates for prediction output...")
    latent_coords = fit_full_latent_coordinates(
        X_all,
        y_all,
        method=args.latent_method,
        n_components=args.latent_components,
        max_iter=args.latent_max_iter,
    )
    all_predictions["act_beh_cca1"] = latent_coords[:, 0]
    all_predictions["act_beh_cca2"] = latent_coords[:, 1] if latent_coords.shape[1] > 1 else np.nan

    act_join_cols = ["prefix_id", "act_pca1", "act_pca2"]
    for col in ["layer", "representation"]:
        if col in activation_manifold.columns:
            act_join_cols.append(col)
    act_join = activation_manifold[act_join_cols].drop_duplicates("prefix_id")
    beh_join = behavior_manifold[["prefix_id", "beh_pca1", "beh_pca2"]].drop_duplicates("prefix_id")
    manifold_join = work_meta.merge(act_join, on="prefix_id", how="left", suffixes=("", "_actman"))
    manifold_join = manifold_join.merge(beh_join, on="prefix_id", how="left")

    procrustes_prediction_all: pd.DataFrame | None = None
    for subset in SUBSET_NAMES:
        subset_mask = masks[subset]
        joined_subset = manifold_join.loc[subset_mask].copy().reset_index(drop=True)
        row, proc_pred = procrustes_cv(joined_subset, subset=subset, layer=layer, representation=args.representation, group_col="_group", n_splits=args.n_splits)
        metrics.append(row)
        if subset == "all" and not proc_pred.empty:
            procrustes_prediction_all = proc_pred

    all_predictions = all_predictions.merge(
        manifold_join[["prefix_id", "act_pca1", "act_pca2", "beh_pca1", "beh_pca2"]].drop_duplicates("prefix_id"),
        on="prefix_id",
        how="left",
    )
    if procrustes_prediction_all is not None:
        all_predictions = all_predictions.merge(
            procrustes_prediction_all[["prefix_id", "aligned_act_beh_pca1", "aligned_act_beh_pca2"]].drop_duplicates("prefix_id"),
            on="prefix_id",
            how="left",
        )

    metrics_df = pd.DataFrame(metrics)
    ensure_dir(output_path.parent)
    metrics_df.to_csv(output_path, index=False)
    ensure_dir(predictions_path.parent)
    all_predictions.to_parquet(predictions_path, index=False)

    figures: list[Path] = []
    fig = figures_dir / "predicted_vs_true_commitment.png"
    plot_true_vs_pred_commitment(all_predictions, fig)
    figures.append(fig)
    fig = figures_dir / "predicted_behavior_from_activation_pca.png"
    plot_predicted_behavior_pca(all_predictions, fig)
    figures.append(fig)
    fig = figures_dir / "cca_by_agent_action.png"
    plot_latent_by_action(all_predictions, fig)
    figures.append(fig)
    fig = figures_dir / "cca_by_commitment.png"
    plot_latent_by_commitment(all_predictions, fig)
    figures.append(fig)
    fig = figures_dir / "behavior_vs_activation_pca_alignment.png"
    plot_procrustes_alignment(all_predictions, fig, max_lines=args.procrustes_plot_pairs, random_state=args.random_state)
    figures.append(fig)
    fig = figures_dir / "rsa_activation_vs_behavior_distance.png"
    plot_rsa(rsa_scatter_all, fig)
    figures.append(fig)

    write_report(
        report_path,
        metrics=metrics_df,
        predictions=all_predictions,
        layer=layer,
        representation=args.representation,
        layer_source=layer_source,
        feature_path=selected_feature_path,
        behavior_manifold_rows=len(behavior_manifold),
        activation_manifold_rows=len(activation_manifold),
        figures=figures,
        latent_method=args.latent_method,
    )

    print(f"Selected layer {layer} ({layer_source}), representation {args.representation}")
    print(f"Wrote {output_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(figures)} figures under {figures_dir}")


if __name__ == "__main__":
    main()
