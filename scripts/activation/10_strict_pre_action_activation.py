#!/usr/bin/env python3
"""Strict pre-action activation analyses before explicit action verbalization."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

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
STRICT_SUBSETS = [
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "buffered_before_any_action_mention",
    "buffered_before_final_action_mention",
]
ALL_SUBSETS = ["all_prefixes", *STRICT_SUBSETS]


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
    if not layers:
        raise SystemExit(f"No feature arrays found for representation {representation!r} in {features_dir}")
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


def selected_layer_from_manifold(path: Path, representation: str) -> int | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["layer", "representation"])
    except Exception:
        try:
            df = pd.read_parquet(path, columns=["layer"])
        except Exception:
            return None
    if "representation" in df.columns:
        df = df[df["representation"].astype(str).eq(representation)]
    values = sorted(pd.Series(df["layer"]).dropna().astype(int).unique().tolist())
    return values[0] if len(values) == 1 else None


def select_plot_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    manifold_path: Path,
    available_layers: list[int],
    representation: str,
) -> tuple[int, str]:
    if requested_layer is not None:
        if requested_layer not in available_layers:
            raise SystemExit(f"Requested plot layer {requested_layer} unavailable. Available: {available_layers}")
        return requested_layer, "explicit --plot_layer"
    if config_path.exists():
        try:
            layer = recursively_find_layer(json.loads(config_path.read_text()))
            if layer in available_layers:
                return int(layer), f"best_activation_config: {config_path}"
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; trying manifold layer")
    layer = selected_layer_from_manifold(manifold_path, representation)
    if layer in available_layers:
        return int(layer), f"activation manifold layer from {manifold_path}"
    index = int(round(0.75 * (len(available_layers) - 1)))
    return available_layers[index], "mid-late fallback from available feature layers"


def make_decision_id(df: pd.DataFrame) -> pd.DataFrame:
    if "decision_id" not in df.columns and {"trajectory_id", "step_id"}.issubset(df.columns):
        out = df.copy()
        out["decision_id"] = out["trajectory_id"].astype(str) + "::" + out["step_id"].astype(str)
        return out
    return df


def merge_mention_indices(meta: pd.DataFrame, prefixes_path: Path) -> pd.DataFrame:
    out = make_decision_id(meta).copy()
    needed = ["first_action_mention_token_index", "first_final_action_mention_token_index"]
    if all(col in out.columns for col in needed):
        return out
    if not prefixes_path.exists():
        raise SystemExit(
            f"Strict mention index columns are missing and prefixes file was not found: {prefixes_path}"
        )
    prefixes_cols = ["prefix_id", "trajectory_id", "step_id", "prefix_token_index", *needed]
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(prefixes_path).schema_arrow.names)
        read_cols = [col for col in prefixes_cols if col in available]
        prefixes = pd.read_parquet(prefixes_path, columns=read_cols)
    except Exception as exc:
        print(f"WARNING: failed to read prefixes schema from {prefixes_path}: {exc}; reading full file")
        prefixes = pd.read_parquet(prefixes_path)
    if not set(needed).issubset(prefixes.columns):
        prefixes = pd.read_parquet(prefixes_path)
    prefixes = make_decision_id(prefixes)
    join_cols = ["prefix_id"] if "prefix_id" in out.columns and "prefix_id" in prefixes.columns else ["decision_id", "prefix_token_index"]
    add_cols = [col for col in [*join_cols, *needed] if col in prefixes.columns]
    prefixes_small = prefixes[add_cols].drop_duplicates(join_cols)
    before = len(out)
    out = out.merge(prefixes_small, on=join_cols, how="left", suffixes=("", "_from_prefixes"))
    if len(out) != before:
        raise RuntimeError("Merging mention indices changed row count")
    for col in needed:
        alt = f"{col}_from_prefixes"
        if alt in out.columns:
            out[col] = out[col].combine_first(out[alt]) if col in out.columns else out[alt]
            out = out.drop(columns=[alt])
    missing = [col for col in needed if col not in out.columns or out[col].isna().all()]
    if missing:
        raise SystemExit(f"Could not recover strict mention columns: {missing}")
    return out


def add_strict_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prefix = pd.to_numeric(out["prefix_token_index"], errors="coerce")
    first_any = pd.to_numeric(out["first_action_mention_token_index"], errors="coerce")
    first_final = pd.to_numeric(out["first_final_action_mention_token_index"], errors="coerce")
    out["strict_before_any_action_mention"] = first_any.isna() | (prefix < first_any)
    out["strict_before_final_action_mention"] = first_final.isna() | (prefix < first_final)
    out["buffered_before_any_action_mention"] = first_any.isna() | (prefix < (first_any - 5))
    out["buffered_before_final_action_mention"] = first_final.isna() | (prefix < (first_final - 5))
    out["normalized_progress_to_first_action_mention"] = np.where(
        first_any.to_numpy(dtype=float) > 0,
        prefix.to_numpy(dtype=float) / first_any.to_numpy(dtype=float),
        np.nan,
    )
    out["normalized_progress_to_first_final_action_mention"] = np.where(
        first_final.to_numpy(dtype=float) > 0,
        prefix.to_numpy(dtype=float) / first_final.to_numpy(dtype=float),
        np.nan,
    )
    return out


def group_splits(df: pd.DataFrame, group_col: str, n_splits: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    groups = df[group_col].to_numpy()
    unique = pd.Series(groups).dropna().unique()
    if len(unique) < 2:
        return [], "fewer than two groups"
    if len(unique) <= 5:
        splitter = LeaveOneGroupOut()
        return list(splitter.split(df, groups=groups)), "LeaveOneGroupOut"
    folds = min(n_splits, len(unique))
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(df, groups=groups)), f"GroupKFold({folds})"


def fit_classifier(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, group_col: str, n_splits: int, max_iter: int) -> dict[str, Any]:
    mask = pd.Series(y).notna().to_numpy() & df[group_col].notna().to_numpy()
    X = X[mask]
    y = np.asarray(y, dtype=object)[mask]
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 4 or len(pd.Series(y).dropna().unique()) < 2:
        return {"status": "skipped", "skip_reason": "not enough rows or classes", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    splits, cv_name = group_splits(sub, group_col, n_splits)
    if not splits:
        return {"status": "skipped", "skip_reason": cv_name, "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    pred = np.full(len(sub), None, dtype=object)
    for train_idx, test_idx in splits:
        if len(pd.Series(y[train_idx]).unique()) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=max_iter))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    valid = pd.Series(pred).notna().to_numpy()
    if not valid.any():
        return {"status": "skipped", "skip_reason": "no valid folds", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    return {
        "status": "ok",
        "skip_reason": "",
        "n_rows": int(valid.sum()),
        "n_groups": int(sub[group_col].nunique()),
        "n_folds": len(splits),
        "cv_method": cv_name,
        "accuracy": float(accuracy_score(y[valid], pred[valid])),
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], pred[valid])),
        "macro_f1": float(f1_score(y[valid], pred[valid], average="macro", zero_division=0)),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if spearmanr is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    result = spearmanr(x[mask], y[mask])
    return float(result.correlation) if result.correlation is not None else float("nan")


def fit_ridge(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, group_col: str, n_splits: int, alpha: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        finite_y = np.isfinite(y)
    else:
        finite_y = np.isfinite(y).all(axis=1)
    mask = finite_y & df[group_col].notna().to_numpy()
    X = X[mask]
    y = y[mask]
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 4:
        return {"status": "skipped", "skip_reason": "not enough rows", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    splits, cv_name = group_splits(sub, group_col, n_splits)
    if not splits:
        return {"status": "skipped", "skip_reason": cv_name, "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    pred = np.full(y.shape, np.nan, dtype=float)
    for train_idx, test_idx in splits:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        if pred.ndim == 2 and np.asarray(fold_pred).ndim == 1:
            fold_pred = np.asarray(fold_pred).reshape(-1, 1)
        pred[test_idx] = fold_pred
    if y.ndim == 2:
        r2_dims = r2_score(y, pred, multioutput="raw_values", force_finite=True)
        out = {
            "status": "ok",
            "skip_reason": "",
            "n_rows": len(sub),
            "n_groups": int(sub[group_col].nunique()),
            "n_folds": len(splits),
            "cv_method": cv_name,
            "r2_mean": float(np.mean(r2_dims)),
            "mse": float(mean_squared_error(y, pred)),
        }
        for idx, action in enumerate(ACTIONS):
            out[f"r2_{action}"] = float(r2_dims[idx])
            out[f"pearson_{action}"] = pearson(y[:, idx], pred[:, idx])
        return out
    return {
        "status": "ok",
        "skip_reason": "",
        "n_rows": len(sub),
        "n_groups": int(sub[group_col].nunique()),
        "n_folds": len(splits),
        "cv_method": cv_name,
        "r2": float(r2_score(y, pred, force_finite=True)),
        "mse": float(mean_squared_error(y, pred)),
        "pearson": pearson(y, pred),
        "spearman": spearman(y, pred),
    }


def evaluate_layer_subset(X: np.ndarray, df: pd.DataFrame, *, layer: int, subset: str, group_col: str, n_splits: int, max_iter: int, ridge_alpha: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cls = fit_classifier(X, df["agent_action"].to_numpy(), df, group_col, n_splits, max_iter)
    cls.update({"layer": layer, "subset": subset, "analysis": "agent_action_decoding", "target": "agent_action"})
    rows.append(cls)
    if all(col in df.columns for col in B_CLR_COLS):
        reg = fit_ridge(X, df[B_CLR_COLS].to_numpy(dtype=float), df, group_col, n_splits, ridge_alpha)
        reg.update({"layer": layer, "subset": subset, "analysis": "behavior_clr_decoding", "target": "behavior_clr"})
        rows.append(reg)
    if "commitment_to_final" in df.columns:
        reg = fit_ridge(X, df["commitment_to_final"].to_numpy(dtype=float), df, group_col, n_splits, ridge_alpha)
        reg.update({"layer": layer, "subset": subset, "analysis": "commitment_decoding", "target": "commitment_to_final"})
        rows.append(reg)
    return rows


def centroid_metrics(scores: pd.DataFrame, subset_col: str, subset_name: str, layer: int, representation: str) -> dict[str, Any]:
    if subset_col not in scores.columns:
        return {"analysis": "centroid_basin", "subset": subset_name, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": f"missing {subset_col}"}
    sub = scores[scores[subset_col].fillna(False).astype(bool)].copy()
    if sub.empty:
        return {"analysis": "centroid_basin", "subset": subset_name, "layer": layer, "representation": representation, "status": "skipped", "skip_reason": "empty subset", "n_rows": 0}
    row = {
        "analysis": "centroid_basin",
        "subset": subset_name,
        "layer": layer,
        "representation": representation,
        "status": "ok",
        "skip_reason": "",
        "n_rows": len(sub),
        "n_decisions": int(sub["decision_id"].nunique()) if "decision_id" in sub.columns else np.nan,
    }
    if "activation_basin_matches_final" in sub.columns:
        row["activation_basin_accuracy"] = float(sub["activation_basin_matches_final"].mean())
    if "act_sim_final" in sub.columns:
        row["mean_act_sim_final"] = float(sub["act_sim_final"].mean())
    if {"act_sim_final", "commitment_to_final"}.issubset(sub.columns):
        row["act_sim_final_commitment_pearson"] = pearson(sub["act_sim_final"], sub["commitment_to_final"])
        row["act_sim_final_commitment_spearman"] = spearman(sub["act_sim_final"], sub["commitment_to_final"])
    return row


def scatter_pca_by_action(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty or not {"act_pca1", "act_pca2", "agent_action"}.issubset(df.columns):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        for action in ACTIONS:
            sub = df[df["agent_action"].astype(str).eq(action)]
            if not sub.empty:
                ax.scatter(sub["act_pca1"], sub["act_pca2"], s=10, alpha=0.65, color=ACTION_COLORS[action], label=action, linewidths=0)
        ax.legend(title="agent_action")
    ax.set_xlabel("act_pca1")
    ax.set_ylabel("act_pca2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_curve_bins(df: pd.DataFrame, norm_col: str, n_bins: int) -> pd.DataFrame:
    work = df.copy()
    values = pd.to_numeric(work[norm_col], errors="coerce")
    work = work[np.isfinite(values) & (values >= 0) & (values < 1)].copy()
    bins = np.linspace(0, 1, n_bins + 1)
    work["normalized_bin"] = pd.cut(
        work[norm_col],
        bins=bins,
        include_lowest=True,
        right=False,
        labels=[f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)],
    )
    work["normalized_bin_mid"] = work["normalized_bin"].map({f"{bins[i]:.2f}-{bins[i+1]:.2f}": (bins[i] + bins[i + 1]) / 2 for i in range(n_bins)})
    return work


def build_curves(scores: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        ("to_first_action_mention", "normalized_progress_to_first_action_mention", "strict_before_any_action_mention"),
        ("to_first_final_action_mention", "normalized_progress_to_first_final_action_mention", "strict_before_final_action_mention"),
    ]
    for curve_name, norm_col, strict_col in specs:
        if norm_col not in scores.columns or strict_col not in scores.columns:
            continue
        work = add_curve_bins(scores[scores[strict_col].fillna(False).astype(bool)], norm_col, n_bins)
        if work.empty:
            continue
        agg = work.groupby("normalized_bin", observed=False).agg(
            normalized_bin_mid=("normalized_bin_mid", "first"),
            prefix_points=("prefix_id", "count") if "prefix_id" in work.columns else (norm_col, "count"),
            decision_steps=("decision_id", "nunique") if "decision_id" in work.columns else (norm_col, "count"),
            mean_act_sim_final=("act_sim_final", "mean") if "act_sim_final" in work.columns else (norm_col, "count"),
            sem_act_sim_final=("act_sim_final", "sem") if "act_sim_final" in work.columns else (norm_col, "count"),
            activation_basin_accuracy=("activation_basin_matches_final", "mean") if "activation_basin_matches_final" in work.columns else (norm_col, "count"),
            mean_commitment_to_final=("commitment_to_final", "mean") if "commitment_to_final" in work.columns else (norm_col, "count"),
        ).reset_index()
        agg["curve"] = curve_name
        agg["norm_col"] = norm_col
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_curve(curves: pd.DataFrame, curve: str, y_col: str, path: Path, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = curves[curves["curve"].eq(curve)].copy()
    if sub.empty or y_col not in sub.columns:
        ax.text(0.5, 0.5, "No curve data", ha="center", va="center", transform=ax.transAxes)
    else:
        x = pd.to_numeric(sub["normalized_bin_mid"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8)
        if y_col == "mean_act_sim_final" and "sem_act_sim_final" in sub.columns:
            sem = pd.to_numeric(sub["sem_act_sim_final"], errors="coerce").fillna(0).to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, alpha=0.2)
    ax.set_xlabel("Normalized progress to mention")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "accuracy" in y_col:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_commitment_vs_similarity(scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if {"strict_before_any_action_mention", "commitment_to_final", "act_sim_final"}.issubset(scores.columns):
        sub = scores[scores["strict_before_any_action_mention"].fillna(False).astype(bool)]
        ax.scatter(sub["commitment_to_final"], sub["act_sim_final"], s=10, alpha=0.35, linewidths=0)
    else:
        ax.text(0.5, 0.5, "Centroid/commitment data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Behavior commitment_to_final")
    ax.set_ylabel("Activation similarity to final centroid")
    ax.set_title("Behavior commitment vs activation similarity before action mention")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for curve, label in [("to_first_action_mention", "Any action"), ("to_first_final_action_mention", "Final action")]:
        sub = curves[curves["curve"].eq(curve)].copy()
        if sub.empty:
            continue
        x = pd.to_numeric(sub["normalized_bin_mid"], errors="coerce").to_numpy(dtype=float)
        axes[0].plot(x, sub["prefix_points"], marker="o", linewidth=1.5, label=label)
        axes[1].plot(x, sub["decision_steps"], marker="o", linewidth=1.5, label=label)
    axes[0].set_ylabel("Prefix points")
    axes[1].set_ylabel("Decision steps")
    for ax in axes:
        ax.set_xlabel("Normalized progress to mention")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Strict pre-action coverage by normalized bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decoding(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    df = metrics[(metrics["analysis"].eq("agent_action_decoding")) & (metrics["status"].eq("ok"))].copy()
    plot_subsets = [
        "strict_before_any_action_mention",
        "strict_before_final_action_mention",
        "buffered_before_any_action_mention",
        "buffered_before_final_action_mention",
    ]
    for subset in plot_subsets:
        sub = df[df["subset"].eq(subset)].sort_values("layer")
        if not sub.empty:
            ax.plot(sub["layer"], sub["balanced_accuracy"], marker="o", linewidth=1.6, label=subset.replace("_", " "))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Strict pre-action agent-action decoding by layer")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
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


def write_report(path: Path, *, metrics: pd.DataFrame, curves: pd.DataFrame, meta: pd.DataFrame, scores: pd.DataFrame | None, figures: list[Path], layers: list[int], plot_layer: int, plot_layer_source: str) -> None:
    lines: list[str] = []
    lines.append("# Strict Pre-action Activation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Prefix rows: {len(meta):,}")
    lines.append(f"- Decisions: {meta['decision_id'].nunique() if 'decision_id' in meta.columns else 'NA'}")
    lines.append(f"- Trajectories: {meta['trajectory_id'].nunique() if 'trajectory_id' in meta.columns else 'NA'}")
    lines.append(f"- Layers decoded: `{layers}`")
    lines.append(f"- Plot/manifold layer: {plot_layer}")
    lines.append(f"- Plot/manifold layer source: {plot_layer_source}")
    lines.append("")

    lines.append("## Strict Coverage\n")
    for subset in STRICT_SUBSETS:
        if subset in meta.columns:
            sub = meta[meta[subset].fillna(False).astype(bool)]
            lines.append(f"- `{subset}`: {len(sub):,} prefixes, {sub['decision_id'].nunique() if 'decision_id' in sub.columns else 'NA'} decisions")
    lines.append("")

    lines.append("## Decoding Highlights\n")
    ok = metrics[(metrics["status"].eq("ok")) & (metrics["analysis"].eq("agent_action_decoding"))]
    for subset in STRICT_SUBSETS:
        sub = ok[ok["subset"].eq(subset)].copy()
        if not sub.empty:
            best = sub.sort_values("balanced_accuracy", ascending=False).iloc[0]
            lines.append(f"- Best `{subset}` action decoding: layer {int(best['layer'])}, balanced accuracy {fmt(best['balanced_accuracy'])}")
    bclr = metrics[(metrics["status"].eq("ok")) & (metrics["analysis"].eq("behavior_clr_decoding"))]
    for subset in ["strict_before_any_action_mention", "strict_before_final_action_mention"]:
        sub = bclr[bclr["subset"].eq(subset)].copy()
        if not sub.empty:
            best = sub.sort_values("r2_mean", ascending=False).iloc[0]
            lines.append(f"- Best `{subset}` behavior CLR R2: layer {int(best['layer'])}, mean R2 {fmt(best['r2_mean'])}")
    commit = metrics[(metrics["status"].eq("ok")) & (metrics["analysis"].eq("commitment_decoding"))]
    for subset in ["strict_before_any_action_mention", "strict_before_final_action_mention"]:
        sub = commit[commit["subset"].eq(subset)].copy()
        if not sub.empty:
            best = sub.sort_values("r2", ascending=False).iloc[0]
            lines.append(f"- Best `{subset}` commitment R2: layer {int(best['layer'])}, R2 {fmt(best['r2'])}")
    lines.append("")

    if scores is not None and not scores.empty:
        lines.append("## Centroid/Basin Strict Metrics\n")
        for subset in STRICT_SUBSETS:
            if subset in scores.columns:
                sub = scores[scores[subset].fillna(False).astype(bool)]
                lines.append(
                    f"- `{subset}`: basin accuracy {fmt(sub['activation_basin_matches_final'].mean() if 'activation_basin_matches_final' in sub else np.nan)}, "
                    f"mean final-centroid similarity {fmt(sub['act_sim_final'].mean() if 'act_sim_final' in sub else np.nan)}, "
                    f"similarity/commitment Pearson {fmt(pearson(sub['act_sim_final'], sub['commitment_to_final']) if {'act_sim_final','commitment_to_final'}.issubset(sub.columns) else np.nan)}"
                )
        lines.append("")

    lines.append("## Interpretation\n")
    best_any = ok[ok["subset"].eq("strict_before_any_action_mention")]
    best_final = ok[ok["subset"].eq("strict_before_final_action_mention")]
    if not best_any.empty:
        lines.append(f"- Before any explicit action token, best linear action decoding balanced accuracy is {fmt(best_any['balanced_accuracy'].max())}.")
    if not best_final.empty:
        lines.append(f"- Before explicit final-action token, best linear action decoding balanced accuracy is {fmt(best_final['balanced_accuracy'].max())}.")
    if scores is not None and not scores.empty and "act_sim_final" in scores.columns:
        any_curve = curves[curves["curve"].eq("to_first_action_mention")]
        if not any_curve.empty:
            first = any_curve["mean_act_sim_final"].dropna().iloc[0] if any_curve["mean_act_sim_final"].notna().any() else np.nan
            last = any_curve["mean_act_sim_final"].dropna().iloc[-1] if any_curve["mean_act_sim_final"].notna().any() else np.nan
            lines.append(f"- Mean final-centroid similarity changes from {fmt(first)} early to {fmt(last)} near first action mention.")
    if not curves.empty:
        late = curves[pd.to_numeric(curves["normalized_bin_mid"], errors="coerce").ge(0.85)]
        if not late.empty:
            min_decisions = int(late["decision_steps"].min())
            lines.append(f"- Late normalized bins have at least {min_decisions} decision steps in the current curves.")
    lines.append("- Caveat: pre-action means no explicit lexical action token in the generated reasoning, not no implicit coordinate/path evidence.")
    lines.append("- PCA manifold separation is descriptive; decoding metrics provide the linear readout test.")
    lines.append("")

    lines.append("## Metrics Table\n")
    display_cols = ["analysis", "subset", "layer", "status", "n_rows", "n_groups", "balanced_accuracy", "macro_f1", "r2_mean", "r2", "pearson", "activation_basin_accuracy", "mean_act_sim_final", "act_sim_final_commitment_pearson"]
    display_cols = [c for c in display_cols if c in metrics.columns]
    lines.append(metrics[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Figures\n")
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--centroid_scores", default="outputs/activation/activation_centroid_scores.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet", help="Optional source for first action mention token indices.")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--output", default="outputs/activation/strict_pre_action_activation_metrics.csv")
    parser.add_argument("--curves_output", default="outputs/activation/strict_pre_action_activation_curves.csv")
    parser.add_argument("--report", default="outputs/activation/reports/strict_pre_action_activation_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/strict_pre_action")
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--plot_layer", type=int, default=None)
    parser.add_argument("--group_by", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--curve_bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    repo_root = infer_repo_root(metadata_path)
    metadata_path = resolve_path(metadata_path, repo_root)
    manifold_path = resolve_path(args.manifold, repo_root)
    centroid_path = resolve_path(args.centroid_scores, repo_root)
    prefixes_path = resolve_path(args.prefixes, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root)
    output_path = resolve_path(args.output, repo_root)
    curves_output_path = resolve_path(args.curves_output, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    meta = pd.read_parquet(metadata_path)
    meta = merge_mention_indices(meta, prefixes_path)
    meta = add_strict_columns(meta)
    if args.group_by not in meta.columns:
        if args.group_by == "trajectory_id" and "decision_id" in meta.columns:
            group_col = "decision_id"
            print("WARNING: trajectory_id unavailable; using decision_id")
        else:
            raise SystemExit(f"Missing group column {args.group_by}")
    else:
        group_col = args.group_by

    available_layers = discover_layers(features_dir, args.representation)
    layer_filter = parse_int_list(args.layers)
    layers = available_layers if layer_filter is None else [layer for layer in available_layers if layer in set(layer_filter)]
    if not layers:
        raise SystemExit("No selected layers have feature arrays")
    plot_layer, plot_layer_source = select_plot_layer(
        requested_layer=args.plot_layer,
        config_path=config_path,
        manifold_path=manifold_path,
        available_layers=available_layers,
        representation=args.representation,
    )

    metrics: list[dict[str, Any]] = []
    for layer in layers:
        path = feature_file(features_dir, layer, args.representation)
        X = np.load(path, mmap_mode="r")
        if X.shape[0] != len(meta):
            raise SystemExit(f"Feature row mismatch for {path}: {X.shape[0]} != {len(meta)}")
        print(f"Evaluating strict pre-action decoding for layer {layer}...")
        for subset in ALL_SUBSETS:
            if subset == "all_prefixes":
                mask = np.ones(len(meta), dtype=bool)
            else:
                mask = meta[subset].fillna(False).astype(bool).to_numpy()
            sub = meta.loc[mask].copy().reset_index(drop=True)
            if sub.empty:
                metrics.append({"analysis": "agent_action_decoding", "subset": subset, "layer": layer, "status": "skipped", "skip_reason": "empty subset", "n_rows": 0})
                continue
            layer_metrics = evaluate_layer_subset(
                np.asarray(X[mask], dtype=np.float32),
                sub,
                layer=layer,
                subset=subset,
                group_col=group_col,
                n_splits=args.n_splits,
                max_iter=args.max_iter,
                ridge_alpha=args.ridge_alpha,
            )
            for row in layer_metrics:
                row["representation"] = args.representation
            metrics.extend(layer_metrics)

    scores = None
    if centroid_path.exists():
        scores = pd.read_parquet(centroid_path)
        scores = merge_mention_indices(scores, prefixes_path)
        scores = add_strict_columns(scores)
        if "representation" in scores.columns:
            scores = scores[scores["representation"].astype(str).eq(args.representation)].copy()
        if "layer" in scores.columns and plot_layer in set(scores["layer"].dropna().astype(int)):
            scores = scores[scores["layer"].astype(int).eq(plot_layer)].copy()
        for subset in STRICT_SUBSETS:
            metrics.append(centroid_metrics(scores, subset, subset, plot_layer, args.representation))
    else:
        print(f"WARNING: centroid scores not found: {centroid_path}")

    metrics_df = pd.DataFrame(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

    curves = build_curves(scores, args.curve_bins) if scores is not None and not scores.empty else pd.DataFrame()
    curves_output_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curves_output_path, index=False)

    manifold = pd.read_parquet(manifold_path) if manifold_path.exists() else pd.DataFrame()
    if not manifold.empty:
        manifold = merge_mention_indices(manifold, prefixes_path)
        manifold = add_strict_columns(manifold)
        if "representation" in manifold.columns:
            manifold = manifold[manifold["representation"].astype(str).eq(args.representation)].copy()
        if "layer" in manifold.columns:
            manifold = manifold[manifold["layer"].astype(int).eq(plot_layer)].copy()

    figures: list[Path] = []
    fig = figures_dir / "strict_pre_action_decoding_by_layer.png"
    plot_decoding(metrics_df, fig)
    figures.append(fig)
    fig = figures_dir / "pca_strict_before_any_action_by_agent_action.png"
    scatter_pca_by_action(
        manifold[manifold["strict_before_any_action_mention"].fillna(False).astype(bool)] if not manifold.empty else pd.DataFrame(),
        fig,
        "Activation PCA before any action mention",
    )
    figures.append(fig)
    fig = figures_dir / "pca_strict_before_final_action_by_agent_action.png"
    scatter_pca_by_action(
        manifold[manifold["strict_before_final_action_mention"].fillna(False).astype(bool)] if not manifold.empty else pd.DataFrame(),
        fig,
        "Activation PCA before final-action mention",
    )
    figures.append(fig)
    fig = figures_dir / "final_centroid_similarity_to_first_action_mention.png"
    plot_curve(curves, "to_first_action_mention", "mean_act_sim_final", fig, "Mean similarity to final centroid", "Final-centroid similarity before first action mention")
    figures.append(fig)
    fig = figures_dir / "final_centroid_similarity_to_first_final_action_mention.png"
    plot_curve(curves, "to_first_final_action_mention", "mean_act_sim_final", fig, "Mean similarity to final centroid", "Final-centroid similarity before first final-action mention")
    figures.append(fig)
    fig = figures_dir / "basin_accuracy_to_first_action_mention.png"
    plot_curve(curves, "to_first_action_mention", "activation_basin_accuracy", fig, "Basin accuracy", "Activation basin accuracy before first action mention")
    figures.append(fig)
    fig = figures_dir / "behavior_commitment_vs_activation_similarity_pre_action.png"
    plot_commitment_vs_similarity(scores if scores is not None else pd.DataFrame(), fig)
    figures.append(fig)
    fig = figures_dir / "strict_pre_action_coverage.png"
    plot_coverage(curves, fig)
    figures.append(fig)

    write_report(
        report_path,
        metrics=metrics_df,
        curves=curves,
        meta=meta,
        scores=scores,
        figures=figures,
        layers=layers,
        plot_layer=plot_layer,
        plot_layer_source=plot_layer_source,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {curves_output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(figures)} figures under {figures_dir}")


if __name__ == "__main__":
    main()
