#!/usr/bin/env python3
"""Robustness controls for activation-side gridworld commitment analyses."""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}
B_CLR_COLS = [f"b_clr_{action}" for action in ACTIONS]
Q_COLS = [f"q_{action}" for action in ACTIONS]
PROGRESS_COVARIATES = ["progress", "prefix_token_index", "num_cot_tokens"]


def parse_int_list(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    return path if path.is_absolute() else repo_root / path


def make_decision_id(df: pd.DataFrame) -> pd.DataFrame:
    if "decision_id" not in df.columns and {"trajectory_id", "step_id"}.issubset(df.columns):
        out = df.copy()
        out["decision_id"] = out["trajectory_id"].astype(str) + "::" + out["step_id"].astype(str)
        return out
    return df


def read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional table not found: {path}")
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"WARNING: failed to read optional table {path}: {exc}")
        return None


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
        for key in [
            "best_layer_behavior_clr_prediction",
            "best_layer_commitment_prediction",
            "best_layer_strict_pre_action_decoding",
            "best_representation",
        ]:
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


def unique_nonnull_value(df: pd.DataFrame | None, col: str) -> Any | None:
    if df is None or col not in df.columns:
        return None
    values = pd.Series(df[col]).dropna().unique().tolist()
    return values[0] if len(values) == 1 else None


def select_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    manifold: pd.DataFrame | None,
    centroid_scores: pd.DataFrame | None,
    features_dir: Path,
    representation: str,
) -> tuple[int, str]:
    available = discover_layers(features_dir, representation)
    if not available:
        raise SystemExit(f"No feature arrays found for representation {representation!r} in {features_dir}")
    if requested_layer is not None:
        if requested_layer not in available:
            raise SystemExit(f"Requested layer {requested_layer} unavailable. Available: {available}")
        return requested_layer, "explicit --layer"
    value = unique_nonnull_value(centroid_scores, "layer")
    if value is not None and int(value) in available:
        return int(value), "centroid_scores layer"
    value = unique_nonnull_value(manifold, "layer")
    if value is not None and int(value) in available:
        return int(value), "activation_manifold layer"
    if config_path is not None and config_path.exists():
        try:
            layer = recursively_find_layer(json.loads(config_path.read_text()))
            if layer in available:
                return int(layer), f"best_activation_config: {config_path}"
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; using fallback layer")
    index = int(round(0.75 * (len(available) - 1)))
    return available[index], "mid-late fallback layer"


def load_feature_rows(meta: pd.DataFrame, features_dir: Path, layer: int, representation: str, repo_root: Path) -> np.ndarray:
    path = feature_file(features_dir, layer, representation)
    if not path.exists():
        col = f"feature_file_layer_{layer:02d}_{representation}"
        if col in meta.columns:
            for raw in pd.Series(meta[col]).dropna().astype(str).unique().tolist():
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = repo_root / candidate
                if candidate.exists():
                    path = candidate
                    break
    if not path.exists():
        raise SystemExit(f"Feature file not found for layer={layer}, representation={representation}: {path}")
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise SystemExit(f"Expected 2D feature matrix, got {matrix.shape}: {path}")
    if "row_index" in meta.columns:
        row_idx = pd.to_numeric(meta["row_index"], errors="coerce")
        if row_idx.notna().all():
            rows = row_idx.astype(int).to_numpy()
            if rows.min(initial=0) >= 0 and rows.max(initial=-1) < matrix.shape[0]:
                return np.asarray(matrix[rows], dtype=np.float32)
    if len(meta) != matrix.shape[0]:
        raise SystemExit(f"Cannot align feature rows: table has {len(meta)} rows, matrix has {matrix.shape[0]}")
    return np.asarray(matrix[:], dtype=np.float32)


def add_num_cot_tokens(meta: pd.DataFrame) -> pd.DataFrame:
    out = make_decision_id(meta).copy()
    if "num_cot_tokens" in out.columns:
        out["num_cot_tokens"] = pd.to_numeric(out["num_cot_tokens"], errors="coerce")
        return out
    if {"decision_id", "prefix_token_index"}.issubset(out.columns):
        prefix = pd.to_numeric(out["prefix_token_index"], errors="coerce")
        out["num_cot_tokens"] = prefix.groupby(out["decision_id"]).transform("max")
    else:
        out["num_cot_tokens"] = np.nan
    return out


def merge_optional_columns(base: pd.DataFrame, other: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    other = make_decision_id(other)
    out = base.copy()
    join_cols = ["prefix_id"] if "prefix_id" in out.columns and "prefix_id" in other.columns else ["decision_id", "prefix_token_index"]
    if not set(join_cols).issubset(other.columns):
        return out
    payload_cols = [col for col in columns if col in other.columns and col not in out.columns]
    if not payload_cols:
        return out
    add_cols = [*join_cols, *payload_cols]
    before = len(out)
    out = out.merge(other[add_cols].drop_duplicates(join_cols), on=join_cols, how="left")
    if len(out) != before:
        raise RuntimeError("Optional metadata merge changed row count")
    return out


def custom_group_splits(groups: np.ndarray, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    series = pd.Series(groups)
    unique = series.dropna().unique().tolist()
    if len(unique) < 2:
        return []
    rng = np.random.default_rng(seed)
    unique = np.asarray(unique, dtype=object)
    rng.shuffle(unique)
    n_splits = min(max(2, n_splits), len(unique))
    folds = np.array_split(unique, n_splits)
    row_positions = np.arange(len(groups))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        test_groups = set(fold.tolist())
        test_mask = series.isin(test_groups).to_numpy()
        train_idx = row_positions[~test_mask]
        test_idx = row_positions[test_mask]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, test_idx))
    return splits


def finite_target_mask(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        mask &= pd.to_numeric(df[col], errors="coerce").notna().to_numpy()
    return mask


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def classification_metrics(
    X: np.ndarray,
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str,
    n_splits: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    mask = df[label_col].notna().to_numpy() & df[group_col].notna().to_numpy()
    y_raw = df.loc[mask, label_col].astype(str).to_numpy()
    groups = df.loc[mask, group_col].to_numpy()
    X_sub = np.asarray(X[mask], dtype=np.float32)
    n_groups = len(pd.Series(groups).dropna().unique())
    n_classes = len(pd.Series(y_raw).dropna().unique())
    if len(y_raw) < 8 or n_groups < 2 or n_classes < 2:
        return {"status": "skipped", "skip_reason": "not enough rows/groups/classes", "n_rows": int(len(y_raw)), "n_groups": n_groups}
    splits = custom_group_splits(groups, n_splits=n_splits, seed=seed)
    if not splits:
        return {"status": "skipped", "skip_reason": "no valid grouped folds", "n_rows": int(len(y_raw)), "n_groups": n_groups}
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    pred = np.full_like(y, fill_value=-1)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="lsqr"))
        model.fit(X_sub[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X_sub[test_idx])
    valid = pred >= 0
    if valid.sum() == 0:
        return {"status": "skipped", "skip_reason": "no valid predictions", "n_rows": int(len(y_raw)), "n_groups": n_groups}
    return {
        "status": "ok",
        "n_rows": int(valid.sum()),
        "n_groups": n_groups,
        "n_folds": len(splits),
        "accuracy": float(accuracy_score(y[valid], pred[valid])),
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], pred[valid])),
        "macro_f1": float(f1_score(y[valid], pred[valid], average="macro", zero_division=0)),
    }


def regression_metrics(
    X: np.ndarray,
    df: pd.DataFrame,
    *,
    target_cols: list[str],
    group_col: str,
    n_splits: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    if not set(target_cols).issubset(df.columns):
        return {"status": "skipped", "skip_reason": "missing target columns", "n_rows": 0, "n_groups": 0}
    mask = finite_target_mask(df, target_cols) & df[group_col].notna().to_numpy()
    y = df.loc[mask, target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = df.loc[mask, group_col].to_numpy()
    X_sub = np.asarray(X[mask], dtype=np.float32)
    n_groups = len(pd.Series(groups).dropna().unique())
    if len(y) < 8 or n_groups < 2:
        return {"status": "skipped", "skip_reason": "not enough rows/groups", "n_rows": int(len(y)), "n_groups": n_groups}
    splits = custom_group_splits(groups, n_splits=n_splits, seed=seed)
    if not splits:
        return {"status": "skipped", "skip_reason": "no valid grouped folds", "n_rows": int(len(y)), "n_groups": n_groups}
    pred = np.full_like(y, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(X_sub[train_idx], y[train_idx])
        fold_pred = model.predict(X_sub[test_idx])
        if fold_pred.ndim == 1:
            fold_pred = fold_pred.reshape(-1, 1)
        pred[test_idx] = fold_pred
    valid = np.isfinite(pred).all(axis=1)
    if valid.sum() == 0:
        return {"status": "skipped", "skip_reason": "no valid predictions", "n_rows": int(len(y)), "n_groups": n_groups}
    r2_values = []
    pearson_values = []
    result: dict[str, Any] = {
        "status": "ok",
        "n_rows": int(valid.sum()),
        "n_groups": n_groups,
        "n_folds": len(splits),
        "mse": float(mean_squared_error(y[valid], pred[valid])),
    }
    for idx, col in enumerate(target_cols):
        r2 = float(r2_score(y[valid, idx], pred[valid, idx]))
        corr = pearson_corr(y[valid, idx], pred[valid, idx])
        result[f"r2_{col}"] = r2
        result[f"pearson_{col}"] = corr
        r2_values.append(r2)
        pearson_values.append(corr)
    result["mean_r2"] = float(np.nanmean(r2_values))
    result["mean_pearson"] = float(np.nanmean(pearson_values))
    return result


def metric_rows_from_result(
    result: dict[str, Any],
    *,
    control: str,
    subset: str,
    metric_prefix: str,
    layer: int | float,
    representation: str,
    group_by: str,
    label: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extra = extra or {}
    base = {
        "control": control,
        "subset": subset,
        "layer": layer,
        "representation": representation,
        "group_by": group_by,
        "target": label,
        "status": result.get("status", "ok"),
        "skip_reason": result.get("skip_reason", ""),
        "n_rows": result.get("n_rows", np.nan),
        "n_groups": result.get("n_groups", np.nan),
        "n_folds": result.get("n_folds", np.nan),
        **extra,
    }
    rows = []
    for key, value in result.items():
        if key in {"status", "skip_reason", "n_rows", "n_groups", "n_folds"}:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            rows.append({**base, "metric": f"{metric_prefix}_{key}", "value": float(value)})
    if not rows and result.get("status") != "ok":
        rows.append({**base, "metric": metric_prefix, "value": np.nan})
    return rows


def shuffle_labels_by_decision(df: pd.DataFrame, *, seed: int) -> pd.Series:
    decisions = df[["decision_id", "agent_action"]].drop_duplicates("decision_id")
    rng = np.random.default_rng(seed)
    shuffled = decisions["agent_action"].to_numpy().copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(decisions["decision_id"].astype(str), shuffled))
    return df["decision_id"].astype(str).map(mapping)


def residualize_progress(X: np.ndarray, df: pd.DataFrame, covariates: list[str] = PROGRESS_COVARIATES) -> tuple[np.ndarray, dict[str, Any]]:
    present = [col for col in covariates if col in df.columns]
    if not present:
        return X.copy(), {"enabled": False, "columns": []}
    cov = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_cols = np.isfinite(cov).all(axis=0)
    present = [col for col, ok in zip(present, finite_cols) if ok]
    cov = cov[:, finite_cols]
    if cov.shape[1] == 0:
        return X.copy(), {"enabled": False, "columns": []}
    means = cov.mean(axis=0)
    scales = cov.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    cov_z = (cov - means) / scales
    design = np.column_stack([np.ones(len(cov_z)), cov_z])
    beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    residual = X - design @ beta
    return residual.astype(np.float32, copy=False), {
        "enabled": True,
        "columns": present,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "beta_shape": list(beta.shape),
    }


def add_cot_length_strata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    length_by_decision = out[["decision_id", "num_cot_tokens"]].drop_duplicates("decision_id").copy()
    length_by_decision["num_cot_tokens"] = pd.to_numeric(length_by_decision["num_cot_tokens"], errors="coerce")
    valid = length_by_decision["num_cot_tokens"].notna()
    if valid.sum() < 3:
        out["cot_length_stratum"] = "unknown"
        return out
    try:
        length_by_decision.loc[valid, "cot_length_stratum"] = pd.qcut(
            length_by_decision.loc[valid, "num_cot_tokens"],
            q=3,
            labels=["short", "medium", "long"],
            duplicates="drop",
        ).astype(str)
    except Exception:
        ranks = length_by_decision.loc[valid, "num_cot_tokens"].rank(method="first")
        length_by_decision.loc[valid, "cot_length_stratum"] = pd.qcut(
            ranks,
            q=3,
            labels=["short", "medium", "long"],
        ).astype(str)
    length_by_decision["cot_length_stratum"] = length_by_decision["cot_length_stratum"].fillna("unknown")
    return out.merge(length_by_decision[["decision_id", "cot_length_stratum"]], on="decision_id", how="left")


def balanced_decision_mask(df: pd.DataFrame, *, seed: int) -> np.ndarray:
    decisions = df[["decision_id", "agent_action"]].drop_duplicates("decision_id").dropna()
    counts = decisions["agent_action"].value_counts()
    if counts.empty:
        return np.zeros(len(df), dtype=bool)
    n_per = int(counts.min())
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for action in ACTIONS:
        vals = decisions.loc[decisions["agent_action"].astype(str).eq(action), "decision_id"].astype(str).to_numpy()
        if len(vals) == 0:
            continue
        rng.shuffle(vals)
        selected.extend(vals[:n_per].tolist())
    return df["decision_id"].astype(str).isin(set(selected)).to_numpy()


def strict_before_any_mask(df: pd.DataFrame) -> np.ndarray:
    if "is_before_first_action_mention" in df.columns:
        return df["is_before_first_action_mention"].fillna(True).astype(bool).to_numpy()
    return np.ones(len(df), dtype=bool)


def basin_accuracy(df: pd.DataFrame, *, label_col: str = "activation_basin_matches_final") -> dict[str, Any]:
    if label_col not in df.columns:
        return {"status": "skipped", "skip_reason": f"missing {label_col}", "n_rows": len(df), "n_groups": df.get("decision_id", pd.Series()).nunique()}
    vals = df[label_col]
    mask = vals.notna()
    if mask.sum() == 0:
        return {"status": "skipped", "skip_reason": "no non-null basin labels", "n_rows": 0, "n_groups": 0}
    return {
        "status": "ok",
        "n_rows": int(mask.sum()),
        "n_groups": int(df.loc[mask, "decision_id"].nunique()) if "decision_id" in df.columns else np.nan,
        "accuracy": float(vals.loc[mask].astype(bool).mean()),
    }


def shuffled_basin_accuracy(df: pd.DataFrame, *, shuffled_col: str) -> dict[str, Any]:
    if "activation_basin_action" not in df.columns or shuffled_col not in df.columns:
        return {"status": "skipped", "skip_reason": "missing basin action or shuffled label", "n_rows": len(df), "n_groups": np.nan}
    mask = df["activation_basin_action"].notna() & df[shuffled_col].notna()
    if mask.sum() == 0:
        return {"status": "skipped", "skip_reason": "no comparable rows", "n_rows": 0, "n_groups": 0}
    return {
        "status": "ok",
        "n_rows": int(mask.sum()),
        "n_groups": int(df.loc[mask, "decision_id"].nunique()),
        "accuracy": float(df.loc[mask, "activation_basin_action"].astype(str).eq(df.loc[mask, shuffled_col].astype(str)).mean()),
    }


def fit_pca2(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    X_z = StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_z)
    return coords, pca


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_true_vs_shuffled(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sub = metrics[
        metrics["control"].isin(["true_label_decoding_by_layer", "shuffled_label_decoding_by_layer"])
        & metrics["metric"].eq("agent_action_balanced_accuracy")
        & metrics["status"].eq("ok")
    ]
    if sub.empty:
        ax.text(0.5, 0.5, "No decoding metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        for control, label, color in [
            ("true_label_decoding_by_layer", "true labels", "#4C78A8"),
            ("shuffled_label_decoding_by_layer", "shuffled labels", "#F58518"),
        ]:
            s = sub[sub["control"].eq(control)].sort_values("layer")
            ax.plot(s["layer"], s["value"], marker="o", linewidth=1.8, label=label, color=color)
        ax.axhline(0.25, color="#777777", linestyle="--", linewidth=1, label="chance")
        ax.legend(frameon=False)
    ax.set_xlabel("layer")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("True vs shuffled agent-action decoding")
    ax.grid(alpha=0.18)
    save_figure(fig, path)


def plot_residualized_pca(df: pd.DataFrame, residual_coords: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    if {"act_pca1", "act_pca2"}.issubset(df.columns):
        coords_list = [(df[["act_pca1", "act_pca2"]].to_numpy(dtype=float), "Original activation PCA")]
    else:
        coords_list = [(None, "Original activation PCA unavailable")]
    coords_list.append((residual_coords, "Progress-residualized activation PCA"))
    for ax, (coords, title) in zip(axes, coords_list):
        if coords is None or len(coords) == 0:
            ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for action in ACTIONS:
                mask = df["agent_action"].astype(str).eq(action).to_numpy()
                ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.45, color=ACTION_COLORS[action], label=action)
            ax.legend(frameon=False, markerscale=2)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(alpha=0.12)
    save_figure(fig, path)


def plot_action_balanced(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics[
        metrics["control"].isin(["selected_layer_true_decoding", "action_balanced_subsample"])
        & metrics["metric"].isin([
            "agent_action_balanced_accuracy",
            "behavior_clr_mean_r2",
            "basin_accuracy_accuracy",
            "strict_pre_action_balanced_accuracy",
        ])
        & metrics["status"].eq("ok")
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if sub.empty:
        ax.text(0.5, 0.5, "No action-balanced metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = ["agent_action_balanced_accuracy", "behavior_clr_mean_r2", "basin_accuracy_accuracy", "strict_pre_action_balanced_accuracy"]
        x = np.arange(len(labels))
        width = 0.35
        for i, (control, name, color) in enumerate([
            ("selected_layer_true_decoding", "all decisions", "#4C78A8"),
            ("action_balanced_subsample", "action-balanced", "#54A24B"),
        ]):
            vals = []
            for metric in labels:
                row = sub[sub["control"].eq(control) & sub["metric"].eq(metric)]
                vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=name, color=color, alpha=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels(["action acc", "behavior R2", "basin acc", "strict acc"], rotation=20, ha="right")
        ax.legend(frameon=False)
    ax.set_ylabel("metric value")
    ax.set_title("Action-balanced robustness metrics")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, path)


def plot_cot_length(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics[
        metrics["control"].eq("cot_length_stratified")
        & metrics["metric"].eq("basin_accuracy_accuracy")
        & metrics["status"].eq("ok")
    ].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    order = ["short", "medium", "long"]
    if sub.empty:
        ax.text(0.5, 0.5, "No CoT-length metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        sub["subset"] = pd.Categorical(sub["subset"], categories=order, ordered=True)
        sub = sub.sort_values("subset")
        ax.bar(sub["subset"].astype(str), sub["value"], color="#4C78A8", alpha=0.78)
        for i, row in enumerate(sub.itertuples()):
            ax.text(i, row.value + 0.01, f"n={int(row.n_groups)}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("CoT length stratum")
    ax.set_ylabel("activation basin accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("CoT-length stratified activation basin accuracy")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, path)


def plot_group_split(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics[
        metrics["control"].eq("group_split_sensitivity")
        & metrics["metric"].isin(["agent_action_balanced_accuracy", "behavior_clr_mean_r2"])
        & metrics["status"].eq("ok")
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if sub.empty:
        ax.text(0.5, 0.5, "No group split metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        metrics_order = ["agent_action_balanced_accuracy", "behavior_clr_mean_r2"]
        groups = ["trajectory_id", "decision_id"]
        x = np.arange(len(metrics_order))
        width = 0.35
        for i, group_by in enumerate(groups):
            vals = []
            for metric in metrics_order:
                row = sub[sub["group_by"].eq(group_by) & sub["metric"].eq(metric)]
                vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=group_by, alpha=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels(["action acc", "behavior R2"])
        ax.legend(frameon=False)
    ax.set_ylabel("metric value")
    ax.set_title("Group split sensitivity")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, path)


def write_report(path: Path, metrics: pd.DataFrame, *, layer: int, representation: str, layer_source: str, residual_info: dict[str, Any]) -> None:
    def val(control: str, metric: str, subset: str = "all", group_by: str | None = None) -> float:
        sub = metrics[metrics["control"].eq(control) & metrics["metric"].eq(metric) & metrics["subset"].eq(subset)]
        if group_by is not None:
            sub = sub[sub["group_by"].eq(group_by)]
        sub = sub[sub["status"].eq("ok")]
        return float(sub["value"].iloc[0]) if len(sub) else float("nan")

    true_best = metrics[
        metrics["control"].eq("true_label_decoding_by_layer")
        & metrics["metric"].eq("agent_action_balanced_accuracy")
        & metrics["status"].eq("ok")
    ]
    shuffled_best = metrics[
        metrics["control"].eq("shuffled_label_decoding_by_layer")
        & metrics["metric"].eq("agent_action_balanced_accuracy")
        & metrics["status"].eq("ok")
    ]
    true_row = true_best.sort_values("value", ascending=False).iloc[0] if len(true_best) else None
    shuf_row = shuffled_best.sort_values("value", ascending=False).iloc[0] if len(shuffled_best) else None

    true_acc = val("selected_layer_true_decoding", "agent_action_balanced_accuracy")
    residual_acc = val("progress_residualized", "agent_action_balanced_accuracy")
    true_r2 = val("selected_layer_true_decoding", "behavior_clr_mean_r2")
    residual_r2 = val("progress_residualized", "behavior_clr_mean_r2")
    balanced_acc = val("action_balanced_subsample", "agent_action_balanced_accuracy")
    balanced_r2 = val("action_balanced_subsample", "behavior_clr_mean_r2")
    traj_acc = val("group_split_sensitivity", "agent_action_balanced_accuracy", group_by="trajectory_id")
    decision_acc = val("group_split_sensitivity", "agent_action_balanced_accuracy", group_by="decision_id")
    basin_true = val("selected_layer_true_decoding", "basin_accuracy_accuracy")
    basin_shuf = val("shuffled_label_centroid", "basin_accuracy_accuracy")

    cot = metrics[
        metrics["control"].eq("cot_length_stratified")
        & metrics["metric"].eq("basin_accuracy_accuracy")
        & metrics["status"].eq("ok")
    ]
    cot_cov_acc = val("cot_length_covariate_added", "agent_action_balanced_accuracy")
    cot_cov_r2 = val("cot_length_covariate_added", "behavior_clr_mean_r2")

    lines = [
        "# Activation Robustness Controls Report",
        "",
        "## Summary",
        "",
        f"- Selected layer: {layer}",
        f"- Selected representation: `{representation}`",
        f"- Layer source: {layer_source}",
        f"- Metrics rows: {len(metrics):,}",
        f"- Progress residualization enabled: {residual_info.get('enabled', False)}",
        f"- Progress residualization covariates: `{residual_info.get('columns', [])}`",
        "",
        "## Shuffled-label Control",
        "",
    ]
    if true_row is not None:
        lines.append(f"- Best true-label layer decoding: layer {int(true_row['layer'])}, balanced accuracy {float(true_row['value']):.4f}")
    if shuf_row is not None:
        lines.append(f"- Best shuffled-label layer decoding: layer {int(shuf_row['layer'])}, balanced accuracy {float(shuf_row['value']):.4f}")
    lines.extend(
        [
            f"- Selected-layer true centroid basin accuracy: {basin_true:.4f}",
            f"- Selected-layer shuffled-label centroid basin accuracy: {basin_shuf:.4f}",
            "- Interpretation: shuffled-label decoding should be near chance if action labels are the source of the signal rather than group or progress artifacts.",
            "",
            "## Progress Residualization",
            "",
            f"- Selected-layer original action decoding balanced accuracy: {true_acc:.4f}",
            f"- Selected-layer residualized action decoding balanced accuracy: {residual_acc:.4f}",
            f"- Selected-layer original behavior CLR mean R2: {true_r2:.4f}",
            f"- Selected-layer residualized behavior CLR mean R2: {residual_r2:.4f}",
            "- Interpretation: remaining separation after residualization suggests action geometry is not explained solely by token position or CoT length.",
            "",
            "## Action-balanced Subsampling",
            "",
            f"- Action-balanced action decoding balanced accuracy: {balanced_acc:.4f}",
            f"- Action-balanced behavior CLR mean R2: {balanced_r2:.4f}",
            "- Interpretation: compare these values with the all-decision selected-layer metrics to assess action imbalance sensitivity.",
            "",
            "## CoT-length Control",
            "",
        ]
    )
    if len(cot):
        for row in cot.sort_values("subset").itertuples():
            lines.append(f"- {row.subset}: activation basin accuracy {row.value:.4f} over {int(row.n_groups)} decisions")
    else:
        lines.append("- CoT-length stratified basin accuracy was unavailable.")
    lines.extend(
        [
            f"- With `num_cot_tokens` appended as a covariate, action decoding balanced accuracy is {cot_cov_acc:.4f} and behavior CLR mean R2 is {cot_cov_r2:.4f}.",
            "- Interpretation: large monotonic differences by CoT length would suggest long traces dominate activation geometry.",
            "",
            "## Group-split Sensitivity",
            "",
            f"- Trajectory-level split action decoding balanced accuracy: {traj_acc:.4f}",
            f"- Decision-level split action decoding balanced accuracy: {decision_acc:.4f}",
            "- Interpretation: trajectory-level splits are usually harder because the model must generalize across trajectory files rather than within them.",
            "",
            "## Required Answers",
            "",
            f"- Does action decoding survive shuffled-label control? True best={float(true_row['value']) if true_row is not None else float('nan'):.4f}, shuffled best={float(shuf_row['value']) if shuf_row is not None else float('nan'):.4f}.",
            f"- Does action geometry survive progress residualization? Selected-layer action balanced accuracy changes from {true_acc:.4f} to {residual_acc:.4f}; behavior CLR mean R2 changes from {true_r2:.4f} to {residual_r2:.4f}.",
            f"- Are results robust to action imbalance? Action-balanced action accuracy is {balanced_acc:.4f} and behavior CLR mean R2 is {balanced_r2:.4f}.",
            "- Do long CoTs dominate the activation manifold? Inspect the CoT-length stratified basin figure and per-stratum rows above.",
            f"- Is trajectory-level split harder than decision-level split? Trajectory split accuracy {traj_acc:.4f}; decision split accuracy {decision_acc:.4f}.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--layer_sweep_metrics", default="outputs/activation/layer_sweep_metrics.csv")
    parser.add_argument("--manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--centroid_scores", default="outputs/activation/activation_centroid_scores.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--output", default="outputs/activation/activation_control_metrics.csv")
    parser.add_argument("--report", default="outputs/activation/reports/activation_controls_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/controls")
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layers", default="all", help="all or comma-separated layer list for layer-wise true/shuffled decoding")
    parser.add_argument("--group_by", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = infer_repo_root(Path(args.metadata))
    metadata_path = resolve_path(args.metadata, repo_root)
    layer_sweep_path = resolve_path(args.layer_sweep_metrics, repo_root)
    manifold_path = resolve_path(args.manifold, repo_root)
    centroid_path = resolve_path(args.centroid_scores, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root)
    output_path = resolve_path(args.output, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    metadata = add_num_cot_tokens(pd.read_parquet(metadata_path))
    manifold = read_optional_table(manifold_path)
    if manifold is not None:
        manifold = make_decision_id(manifold)
    centroid_scores = read_optional_table(centroid_path)
    if centroid_scores is not None:
        centroid_scores = make_decision_id(centroid_scores)
    _layer_sweep = read_optional_table(layer_sweep_path)

    base = metadata.copy()
    base = merge_optional_columns(
        base,
        centroid_scores,
        [
            "activation_basin_action",
            "activation_basin_matches_final",
            "activation_margin",
            "act_sim_final",
            "act_sim_UP",
            "act_sim_DOWN",
            "act_sim_LEFT",
            "act_sim_RIGHT",
            "layer",
            "representation",
            "lock_margin_threshold",
        ],
    )
    base = merge_optional_columns(base, manifold, ["act_pca1", "act_pca2", "layer", "representation"])
    if args.group_by not in base.columns:
        raise SystemExit(f"Missing group column: {args.group_by}")

    selected_layer, layer_source = select_layer(
        requested_layer=args.layer,
        config_path=config_path,
        manifold=manifold,
        centroid_scores=centroid_scores,
        features_dir=features_dir,
        representation=args.representation,
    )
    layer_list = parse_int_list(args.layers)
    if layer_list is None:
        layer_list = discover_layers(features_dir, args.representation)

    metrics: list[dict[str, Any]] = []
    base["agent_action_shuffled"] = shuffle_labels_by_decision(base, seed=args.seed)

    print(f"Selected layer {selected_layer} ({layer_source}), representation {args.representation}")
    print(f"Layer-wise shuffled-label control over layers: {layer_list}")

    for layer in layer_list:
        print(f"Decoding true/shuffled labels for layer {layer}...")
        X_layer = load_feature_rows(base, features_dir, layer, args.representation, repo_root)
        true_res = classification_metrics(
            X_layer,
            base,
            label_col="agent_action",
            group_col=args.group_by,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=args.ridge_alpha,
        )
        metrics.extend(
            metric_rows_from_result(
                true_res,
                control="true_label_decoding_by_layer",
                subset="all",
                metric_prefix="agent_action",
                layer=layer,
                representation=args.representation,
                group_by=args.group_by,
                label="agent_action",
            )
        )
        shuf_res = classification_metrics(
            X_layer,
            base,
            label_col="agent_action_shuffled",
            group_col=args.group_by,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=args.ridge_alpha,
        )
        metrics.extend(
            metric_rows_from_result(
                shuf_res,
                control="shuffled_label_decoding_by_layer",
                subset="all",
                metric_prefix="agent_action",
                layer=layer,
                representation=args.representation,
                group_by=args.group_by,
                label="agent_action_shuffled",
            )
        )

    X = load_feature_rows(base, features_dir, selected_layer, args.representation, repo_root)
    true_cls = classification_metrics(
        X,
        base,
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            true_cls,
            control="selected_layer_true_decoding",
            subset="all",
            metric_prefix="agent_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
        )
    )
    true_reg = regression_metrics(
        X,
        base,
        target_cols=B_CLR_COLS,
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            true_reg,
            control="selected_layer_true_decoding",
            subset="all",
            metric_prefix="behavior_clr",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="behavior_clr",
        )
    )
    metrics.extend(
        metric_rows_from_result(
            basin_accuracy(base),
            control="selected_layer_true_decoding",
            subset="all",
            metric_prefix="basin_accuracy",
            layer=selected_layer,
            representation=args.representation,
            group_by="none",
            label="activation_basin_matches_final",
        )
    )
    metrics.extend(
        metric_rows_from_result(
            shuffled_basin_accuracy(base, shuffled_col="agent_action_shuffled"),
            control="shuffled_label_centroid",
            subset="all",
            metric_prefix="basin_accuracy",
            layer=selected_layer,
            representation=args.representation,
            group_by="none",
            label="activation_basin_action_vs_shuffled_agent_action",
        )
    )

    strict_mask = strict_before_any_mask(base)
    strict_df = base.loc[strict_mask].reset_index(drop=True)
    strict_X = X[strict_mask]
    strict_cls = classification_metrics(
        strict_X,
        strict_df,
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            strict_cls,
            control="selected_layer_true_decoding",
            subset="strict_before_any_action_mention",
            metric_prefix="strict_pre_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
        )
    )

    print("Running progress residualization control...")
    X_resid, residual_info = residualize_progress(X, base)
    resid_cls = classification_metrics(
        X_resid,
        base,
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            resid_cls,
            control="progress_residualized",
            subset="all",
            metric_prefix="agent_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
            extra={"covariates": json.dumps(residual_info.get("columns", []))},
        )
    )
    resid_reg = regression_metrics(
        X_resid,
        base,
        target_cols=B_CLR_COLS,
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            resid_reg,
            control="progress_residualized",
            subset="all",
            metric_prefix="behavior_clr",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="behavior_clr",
            extra={"covariates": json.dumps(residual_info.get("columns", []))},
        )
    )
    residual_coords, residual_pca = fit_pca2(X_resid)
    metrics.append(
        {
            "control": "progress_residualized_pca",
            "subset": "all",
            "layer": selected_layer,
            "representation": args.representation,
            "group_by": "none",
            "target": "activation_features",
            "metric": "pca_explained_variance_ratio_pc1",
            "value": float(residual_pca.explained_variance_ratio_[0]),
            "status": "ok",
            "skip_reason": "",
            "n_rows": len(base),
            "n_groups": base["decision_id"].nunique(),
            "n_folds": np.nan,
            "covariates": json.dumps(residual_info.get("columns", [])),
        }
    )
    metrics.append(
        {
            "control": "progress_residualized_pca",
            "subset": "all",
            "layer": selected_layer,
            "representation": args.representation,
            "group_by": "none",
            "target": "activation_features",
            "metric": "pca_explained_variance_ratio_pc2",
            "value": float(residual_pca.explained_variance_ratio_[1]),
            "status": "ok",
            "skip_reason": "",
            "n_rows": len(base),
            "n_groups": base["decision_id"].nunique(),
            "n_folds": np.nan,
            "covariates": json.dumps(residual_info.get("columns", [])),
        }
    )

    print("Running action-balanced subsampling control...")
    balanced_mask = balanced_decision_mask(base, seed=args.seed)
    balanced_df = base.loc[balanced_mask].reset_index(drop=True)
    balanced_X = X[balanced_mask]
    bal_cls = classification_metrics(
        balanced_X,
        balanced_df,
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            bal_cls,
            control="action_balanced_subsample",
            subset="all",
            metric_prefix="agent_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
        )
    )
    bal_reg = regression_metrics(
        balanced_X,
        balanced_df,
        target_cols=B_CLR_COLS,
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            bal_reg,
            control="action_balanced_subsample",
            subset="all",
            metric_prefix="behavior_clr",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="behavior_clr",
        )
    )
    metrics.extend(
        metric_rows_from_result(
            basin_accuracy(balanced_df),
            control="action_balanced_subsample",
            subset="all",
            metric_prefix="basin_accuracy",
            layer=selected_layer,
            representation=args.representation,
            group_by="none",
            label="activation_basin_matches_final",
        )
    )
    balanced_strict_mask = strict_before_any_mask(balanced_df)
    bal_strict_cls = classification_metrics(
        balanced_X[balanced_strict_mask],
        balanced_df.loc[balanced_strict_mask].reset_index(drop=True),
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            bal_strict_cls,
            control="action_balanced_subsample",
            subset="strict_before_any_action_mention",
            metric_prefix="strict_pre_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
        )
    )

    print("Running CoT-length controls...")
    stratified = add_cot_length_strata(base)
    for stratum in ["short", "medium", "long"]:
        mask = stratified["cot_length_stratum"].astype(str).eq(stratum).to_numpy()
        sub_df = stratified.loc[mask].reset_index(drop=True)
        if len(sub_df) == 0:
            continue
        metrics.extend(
            metric_rows_from_result(
                basin_accuracy(sub_df),
                control="cot_length_stratified",
                subset=stratum,
                metric_prefix="basin_accuracy",
                layer=selected_layer,
                representation=args.representation,
                group_by="none",
                label="activation_basin_matches_final",
            )
        )
        cls = classification_metrics(
            X[mask],
            sub_df,
            label_col="agent_action",
            group_col=args.group_by,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=args.ridge_alpha,
        )
        metrics.extend(
            metric_rows_from_result(
                cls,
                control="cot_length_stratified",
                subset=stratum,
                metric_prefix="agent_action",
                layer=selected_layer,
                representation=args.representation,
                group_by=args.group_by,
                label="agent_action",
            )
        )
    X_plus_len = np.column_stack([X, pd.to_numeric(base["num_cot_tokens"], errors="coerce").fillna(0).to_numpy(dtype=float)])
    cov_cls = classification_metrics(
        X_plus_len,
        base,
        label_col="agent_action",
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            cov_cls,
            control="cot_length_covariate_added",
            subset="all",
            metric_prefix="agent_action",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="agent_action",
            extra={"covariates": json.dumps(["num_cot_tokens"])},
        )
    )
    cov_reg = regression_metrics(
        X_plus_len,
        base,
        target_cols=B_CLR_COLS,
        group_col=args.group_by,
        n_splits=args.n_splits,
        seed=args.seed,
        alpha=args.ridge_alpha,
    )
    metrics.extend(
        metric_rows_from_result(
            cov_reg,
            control="cot_length_covariate_added",
            subset="all",
            metric_prefix="behavior_clr",
            layer=selected_layer,
            representation=args.representation,
            group_by=args.group_by,
            label="behavior_clr",
            extra={"covariates": json.dumps(["num_cot_tokens"])},
        )
    )

    print("Running group split sensitivity control...")
    for group_col in ["trajectory_id", "decision_id"]:
        if group_col not in base.columns:
            continue
        cls = classification_metrics(
            X,
            base,
            label_col="agent_action",
            group_col=group_col,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=args.ridge_alpha,
        )
        metrics.extend(
            metric_rows_from_result(
                cls,
                control="group_split_sensitivity",
                subset="all",
                metric_prefix="agent_action",
                layer=selected_layer,
                representation=args.representation,
                group_by=group_col,
                label="agent_action",
            )
        )
        reg = regression_metrics(
            X,
            base,
            target_cols=B_CLR_COLS,
            group_col=group_col,
            n_splits=args.n_splits,
            seed=args.seed,
            alpha=args.ridge_alpha,
        )
        metrics.extend(
            metric_rows_from_result(
                reg,
                control="group_split_sensitivity",
                subset="all",
                metric_prefix="behavior_clr",
                layer=selected_layer,
                representation=args.representation,
                group_by=group_col,
                label="behavior_clr",
            )
        )

    metrics_df = pd.DataFrame(metrics)
    ensure_dir(output_path.parent)
    ensure_dir(report_path.parent)
    ensure_dir(figures_dir)
    metrics_df.to_csv(output_path, index=False)

    plot_true_vs_shuffled(metrics_df, figures_dir / "true_vs_shuffled_decoding.png")
    plot_residualized_pca(base, residual_coords, figures_dir / "progress_residualized_pca_by_action.png")
    plot_action_balanced(metrics_df, figures_dir / "action_balanced_decoding.png")
    plot_cot_length(metrics_df, figures_dir / "cot_length_stratified_basin_accuracy.png")
    plot_group_split(metrics_df, figures_dir / "group_split_sensitivity.png")

    write_report(
        report_path,
        metrics_df,
        layer=selected_layer,
        representation=args.representation,
        layer_source=layer_source,
        residual_info=residual_info,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote 5 figures under {figures_dir}")


if __name__ == "__main__":
    main()
