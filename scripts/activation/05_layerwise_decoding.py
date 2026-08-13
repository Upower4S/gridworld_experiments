#!/usr/bin/env python3
"""Layer-wise linear decoding for activation-side commitment analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from sklearn.svm import LinearSVC
except Exception:  # pragma: no cover - optional sklearn component
    LinearSVC = None

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - scipy is optional
    spearmanr = None


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
BEHAVIOR_CLR_COLS = [f"b_clr_{action}" for action in ACTIONS]
SCALAR_TARGETS = ["commitment_to_final", "entropy_commitment", "margin_commitment"]
CLASS_TARGETS = ["agent_action", "top_action", "is_top_action_final"]
DEFAULT_SUBSETS = [
    "all",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "post_mention",
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


def parse_int_list(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_repo_root(path: Path) -> Path:
    path = path.resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "scripts").exists() and (parent / "outputs").exists():
            return parent
    return Path.cwd().resolve()


def feature_file(features_dir: Path, layer: int, representation: str) -> Path:
    return features_dir / f"layer_{layer:02d}_{representation}.npy"


def discover_layers(features_dir: Path, representations: list[str]) -> list[int]:
    layers: set[int] = set()
    for rep in representations:
        for path in features_dir.glob(f"layer_*_{rep}.npy"):
            parts = path.stem.split("_")
            if len(parts) >= 3 and parts[0] == "layer":
                try:
                    layers.add(int(parts[1]))
                except ValueError:
                    continue
    if not layers:
        raise FileNotFoundError(f"No feature arrays found in {features_dir} for {representations}")
    return sorted(layers)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if spearmanr is None:
        return float("nan")
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    result = spearmanr(x[mask], y[mask])
    return float(result.correlation) if result.correlation is not None else float("nan")


def cv_splitter(groups: np.ndarray, *, n_splits: int, logo_threshold: int):
    unique_groups = pd.Series(groups).dropna().unique()
    n_groups = len(unique_groups)
    if n_groups < 2:
        return None
    if n_groups <= logo_threshold:
        return LeaveOneGroupOut(), n_groups
    return GroupKFold(n_splits=min(n_splits, n_groups)), min(n_splits, n_groups)


def subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}

    if "is_before_first_action_mention" in df.columns:
        before_any = df["is_before_first_action_mention"].fillna(True).astype(bool).to_numpy()
    else:
        before_any = np.ones(n, dtype=bool)
    if "first_action_mention_token_index" in df.columns and "prefix_token_index" in df.columns:
        first_any = pd.to_numeric(df["first_action_mention_token_index"], errors="coerce")
        prefix = pd.to_numeric(df["prefix_token_index"], errors="coerce")
        before_any = before_any & (first_any.isna().to_numpy() | (prefix < first_any).to_numpy())
    masks["strict_before_any_action_mention"] = before_any

    if "is_before_first_final_action_mention" in df.columns:
        before_final = df["is_before_first_final_action_mention"].fillna(True).astype(bool).to_numpy()
    else:
        before_final = np.ones(n, dtype=bool)
    if "first_final_action_mention_token_index" in df.columns and "prefix_token_index" in df.columns:
        first_final = pd.to_numeric(df["first_final_action_mention_token_index"], errors="coerce")
        prefix = pd.to_numeric(df["prefix_token_index"], errors="coerce")
        before_final = before_final & (first_final.isna().to_numpy() | (prefix < first_final).to_numpy())
    masks["strict_before_final_action_mention"] = before_final

    masks["post_mention"] = ~before_any

    if "commitment_to_final" in df.columns:
        commitment = pd.to_numeric(df["commitment_to_final"], errors="coerce").to_numpy(dtype=float)
        masks["high_commitment"] = np.isfinite(commitment) & (commitment >= 0.8)
        masks["low_commitment"] = np.isfinite(commitment) & (commitment <= 0.4)
    else:
        masks["high_commitment"] = np.zeros(n, dtype=bool)
        masks["low_commitment"] = np.zeros(n, dtype=bool)
    return masks


def valid_regression_mask(df: pd.DataFrame, cols: list[str], base_mask: np.ndarray, group_col: str) -> np.ndarray:
    mask = base_mask.copy()
    for col in cols:
        mask &= pd.to_numeric(df[col], errors="coerce").notna().to_numpy()
    mask &= df[group_col].notna().to_numpy()
    return mask


def valid_classification_mask(df: pd.DataFrame, col: str, base_mask: np.ndarray, group_col: str) -> np.ndarray:
    return base_mask & df[col].notna().to_numpy() & df[group_col].notna().to_numpy()


def empty_metric_row(
    *,
    layer: int,
    representation: str,
    subset: str,
    target_kind: str,
    target_name: str,
    n_rows: int,
    n_groups: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "representation": representation,
        "subset": subset,
        "target_kind": target_kind,
        "target_name": target_name,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "n_folds": 0,
        "status": "skipped",
        "skip_reason": reason,
    }


def fit_predict_ridge_multi(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    n_splits: int,
    logo_threshold: int,
) -> tuple[np.ndarray | None, int, str | None]:
    splitter_info = cv_splitter(groups, n_splits=n_splits, logo_threshold=logo_threshold)
    if splitter_info is None:
        return None, 0, "fewer than 2 groups"
    splitter, n_folds = splitter_info
    preds = np.full(y.shape, np.nan, dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        if preds.ndim == 2 and np.asarray(fold_pred).ndim == 1:
            fold_pred = np.asarray(fold_pred).reshape(-1, 1)
        preds[test_idx] = fold_pred
    return preds, n_folds, None


def fit_predict_classifier(
    X: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    c_value: float,
    n_splits: int,
    logo_threshold: int,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray | None, int, str | None, list[Any]]:
    labels = pd.Series(y_raw).dropna().unique().tolist()
    if len(labels) < 2:
        return None, 0, "fewer than 2 target classes", labels
    splitter_info = cv_splitter(groups, n_splits=n_splits, logo_threshold=logo_threshold)
    if splitter_info is None:
        return None, 0, "fewer than 2 groups", labels
    splitter, n_folds = splitter_info

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    pred_codes = np.full(y.shape, -1, dtype=int)

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_classes = np.unique(y[train_idx])
        if len(train_classes) < 2:
            continue
        if model_name == "linear_svm":
            if LinearSVC is None:
                return None, 0, "LinearSVC is unavailable", labels
            clf = LinearSVC(C=c_value, random_state=random_state, max_iter=max_iter, dual="auto")
        else:
            clf = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
        model = make_pipeline(StandardScaler(), clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X[train_idx], y[train_idx])
        pred_codes[test_idx] = model.predict(X[test_idx])

    valid = pred_codes >= 0
    if not valid.any():
        return None, n_folds, "no valid classification folds", labels
    preds = np.full(y_raw.shape, None, dtype=object)
    preds[valid] = encoder.inverse_transform(pred_codes[valid])
    return preds, n_folds, None, encoder.classes_.tolist()


def regression_rows(
    *,
    layer: int,
    representation: str,
    subset: str,
    target_kind: str,
    target_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_rows: int,
    n_groups: int,
    n_folds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mse = float(mean_squared_error(y_true, y_pred))
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        r2_dims = r2_score(y_true, y_pred, multioutput="raw_values", force_finite=True)
        r2_mean = float(np.mean(r2_dims))
        row: dict[str, Any] = {
            "layer": layer,
            "representation": representation,
            "subset": subset,
            "target_kind": target_kind,
            "target_name": "behavior_clr",
            "n_rows": n_rows,
            "n_groups": n_groups,
            "n_folds": n_folds,
            "status": "ok",
            "skip_reason": "",
            "r2": r2_mean,
            "mse": mse,
        }
        for idx, name in enumerate(target_names):
            suffix = name.replace("b_clr_", "")
            row[f"r2_{suffix}"] = float(r2_dims[idx])
            row[f"pearson_{suffix}"] = pearson_corr(y_true[:, idx], y_pred[:, idx])
        rows.append(row)
        return rows

    y_t = y_true.reshape(-1)
    y_p = y_pred.reshape(-1)
    rows.append(
        {
            "layer": layer,
            "representation": representation,
            "subset": subset,
            "target_kind": target_kind,
            "target_name": target_names[0],
            "n_rows": n_rows,
            "n_groups": n_groups,
            "n_folds": n_folds,
            "status": "ok",
            "skip_reason": "",
            "r2": float(r2_score(y_t, y_p, force_finite=True)),
            "mse": float(mean_squared_error(y_t, y_p)),
            "pearson": pearson_corr(y_t, y_p),
            "spearman": spearman_corr(y_t, y_p),
        }
    )
    return rows


def classification_row(
    *,
    layer: int,
    representation: str,
    subset: str,
    target_kind: str,
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    n_rows: int,
    n_groups: int,
    n_folds: int,
) -> dict[str, Any]:
    valid = pd.Series(y_pred).notna().to_numpy()
    y_t = y_true[valid]
    y_p = y_pred[valid]
    label_list = sorted(pd.Series(labels).dropna().tolist(), key=lambda x: str(x))
    matrix = confusion_matrix(y_t, y_p, labels=label_list)
    return {
        "layer": layer,
        "representation": representation,
        "subset": subset,
        "target_kind": target_kind,
        "target_name": target_name,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "n_folds": n_folds,
        "status": "ok",
        "skip_reason": "",
        "accuracy": float(accuracy_score(y_t, y_p)),
        "balanced_accuracy": float(balanced_accuracy_score(y_t, y_p)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "confusion_labels_json": json.dumps([str(x) for x in label_list]),
        "confusion_matrix_json": json.dumps(matrix.astype(int).tolist()),
        "n_predicted": int(valid.sum()),
    }


def prediction_frame(
    meta: pd.DataFrame,
    row_indices: np.ndarray,
    *,
    layer: int,
    representation: str,
    subset: str,
    target_kind: str,
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    base_cols = [
        "row_index",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
    ]
    base_cols = [col for col in base_cols if col in meta.columns]
    out = meta.iloc[row_indices][base_cols].copy()
    out["layer"] = layer
    out["representation"] = representation
    out["subset"] = subset
    out["target_kind"] = target_kind
    out["target_name"] = target_name
    if y_true.ndim == 2:
        names = target_names or [f"dim_{i}" for i in range(y_true.shape[1])]
        for idx, name in enumerate(names):
            out[f"true_{name}"] = y_true[:, idx]
            out[f"pred_{name}"] = y_pred[:, idx]
    else:
        out["true_value"] = y_true
        out["pred_value"] = y_pred
    return out


def evaluate_layer_representation(
    meta: pd.DataFrame,
    X_all: np.ndarray,
    *,
    layer: int,
    representation: str,
    subsets: dict[str, np.ndarray],
    group_col: str,
    ridge_alpha: float,
    logistic_c: float,
    n_splits: int,
    logo_threshold: int,
    random_state: int,
    max_iter: int,
    run_svm: bool,
    save_predictions: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for subset_name in DEFAULT_SUBSETS:
        base_mask = subsets[subset_name]

        behavior_cols = [col for col in BEHAVIOR_CLR_COLS if col in meta.columns]
        if len(behavior_cols) == 4:
            mask = valid_regression_mask(meta, behavior_cols, base_mask, group_col)
            row_indices = np.flatnonzero(mask)
            n_groups = int(meta.loc[mask, group_col].nunique())
            if len(row_indices) < 4 or n_groups < 2:
                metrics.append(
                    empty_metric_row(
                        layer=layer,
                        representation=representation,
                        subset=subset_name,
                        target_kind="regression",
                        target_name="behavior_clr",
                        n_rows=len(row_indices),
                        n_groups=n_groups,
                        reason="not enough rows or groups",
                    )
                )
            else:
                X = np.asarray(X_all[row_indices], dtype=np.float32)
                y = meta.loc[mask, behavior_cols].to_numpy(dtype=float)
                groups = meta.loc[mask, group_col].to_numpy()
                y_pred, folds, reason = fit_predict_ridge_multi(
                    X,
                    y,
                    groups,
                    alpha=ridge_alpha,
                    n_splits=n_splits,
                    logo_threshold=logo_threshold,
                )
                if y_pred is None:
                    metrics.append(
                        empty_metric_row(
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind="regression",
                            target_name="behavior_clr",
                            n_rows=len(row_indices),
                            n_groups=n_groups,
                            reason=reason or "fit failed",
                        )
                    )
                else:
                    metrics.extend(
                        regression_rows(
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind="regression",
                            target_names=behavior_cols,
                            y_true=y,
                            y_pred=y_pred,
                            n_rows=len(row_indices),
                            n_groups=n_groups,
                            n_folds=folds,
                        )
                    )
                    if save_predictions:
                        predictions.append(
                            prediction_frame(
                                meta,
                                row_indices,
                                layer=layer,
                                representation=representation,
                                subset=subset_name,
                                target_kind="regression",
                                target_name="behavior_clr",
                                y_true=y,
                                y_pred=y_pred,
                                target_names=behavior_cols,
                            )
                        )

        for target in SCALAR_TARGETS:
            if target not in meta.columns:
                continue
            mask = valid_regression_mask(meta, [target], base_mask, group_col)
            row_indices = np.flatnonzero(mask)
            n_groups = int(meta.loc[mask, group_col].nunique())
            if len(row_indices) < 4 or n_groups < 2:
                metrics.append(
                    empty_metric_row(
                        layer=layer,
                        representation=representation,
                        subset=subset_name,
                        target_kind="regression",
                        target_name=target,
                        n_rows=len(row_indices),
                        n_groups=n_groups,
                        reason="not enough rows or groups",
                    )
                )
                continue
            X = np.asarray(X_all[row_indices], dtype=np.float32)
            y = meta.loc[mask, target].to_numpy(dtype=float).reshape(-1, 1)
            groups = meta.loc[mask, group_col].to_numpy()
            y_pred, folds, reason = fit_predict_ridge_multi(
                X,
                y,
                groups,
                alpha=ridge_alpha,
                n_splits=n_splits,
                logo_threshold=logo_threshold,
            )
            if y_pred is None:
                metrics.append(
                    empty_metric_row(
                        layer=layer,
                        representation=representation,
                        subset=subset_name,
                        target_kind="regression",
                        target_name=target,
                        n_rows=len(row_indices),
                        n_groups=n_groups,
                        reason=reason or "fit failed",
                    )
                )
            else:
                metrics.extend(
                    regression_rows(
                        layer=layer,
                        representation=representation,
                        subset=subset_name,
                        target_kind="regression",
                        target_names=[target],
                        y_true=y,
                        y_pred=y_pred,
                        n_rows=len(row_indices),
                        n_groups=n_groups,
                        n_folds=folds,
                    )
                )
                if save_predictions:
                    predictions.append(
                        prediction_frame(
                            meta,
                            row_indices,
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind="regression",
                            target_name=target,
                            y_true=y.reshape(-1),
                            y_pred=y_pred.reshape(-1),
                        )
                    )

        for target in CLASS_TARGETS:
            if target not in meta.columns:
                continue
            mask = valid_classification_mask(meta, target, base_mask, group_col)
            row_indices = np.flatnonzero(mask)
            n_groups = int(meta.loc[mask, group_col].nunique())
            y_raw = meta.loc[mask, target].to_numpy()
            model_specs = [("logistic_regression", "classification")]
            if run_svm:
                model_specs.append(("linear_svm", "classification_svm"))
            for model_name, target_kind in model_specs:
                if len(row_indices) < 4 or n_groups < 2 or len(pd.Series(y_raw).dropna().unique()) < 2:
                    metrics.append(
                        empty_metric_row(
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind=target_kind,
                            target_name=target,
                            n_rows=len(row_indices),
                            n_groups=n_groups,
                            reason="not enough rows, groups, or classes",
                        )
                    )
                    continue
                X = np.asarray(X_all[row_indices], dtype=np.float32)
                groups = meta.loc[mask, group_col].to_numpy()
                y_pred, folds, reason, labels = fit_predict_classifier(
                    X,
                    y_raw,
                    groups,
                    model_name=model_name,
                    c_value=logistic_c,
                    n_splits=n_splits,
                    logo_threshold=logo_threshold,
                    random_state=random_state,
                    max_iter=max_iter,
                )
                if y_pred is None:
                    metrics.append(
                        empty_metric_row(
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind=target_kind,
                            target_name=target,
                            n_rows=len(row_indices),
                            n_groups=n_groups,
                            reason=reason or "fit failed",
                        )
                    )
                else:
                    metrics.append(
                        classification_row(
                            layer=layer,
                            representation=representation,
                            subset=subset_name,
                            target_kind=target_kind,
                            target_name=target,
                            y_true=y_raw,
                            y_pred=y_pred,
                            labels=labels,
                            n_rows=len(row_indices),
                            n_groups=n_groups,
                            n_folds=folds,
                        )
                    )
                    if save_predictions and target_kind == "classification":
                        predictions.append(
                            prediction_frame(
                                meta,
                                row_indices,
                                layer=layer,
                                representation=representation,
                                subset=subset_name,
                                target_kind=target_kind,
                                target_name=target,
                                y_true=y_raw,
                                y_pred=y_pred,
                            )
                        )

    return metrics, predictions


def best_row(
    metrics: pd.DataFrame,
    *,
    subset: str,
    target_name: str,
    metric_col: str,
    representations: list[str] | None = None,
) -> dict[str, Any] | None:
    df = metrics[
        metrics["status"].eq("ok")
        & metrics["subset"].eq(subset)
        & metrics["target_name"].eq(target_name)
        & metrics[metric_col].notna()
    ].copy()
    if representations is not None:
        df = df[df["representation"].isin(representations)]
    if df.empty:
        return None
    idx = df[metric_col].astype(float).idxmax()
    row = df.loc[idx].to_dict()
    return {
        "layer": int(row["layer"]),
        "representation": row["representation"],
        "subset": subset,
        "target_name": target_name,
        "metric": metric_col,
        "value": float(row[metric_col]),
        "n_rows": int(row["n_rows"]),
        "n_groups": int(row["n_groups"]),
    }


def plot_metric_by_layer(
    metrics: pd.DataFrame,
    *,
    representation: str,
    subset: str,
    target_name: str,
    metric_col: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    df = metrics[
        metrics["status"].eq("ok")
        & metrics["representation"].eq(representation)
        & metrics["subset"].eq(subset)
        & metrics["target_name"].eq(target_name)
        & metrics[metric_col].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    if not df.empty:
        df = df.sort_values("layer")
        ax.plot(df["layer"], df[metric_col], marker="o", linewidth=1.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_representation_comparison(
    metrics: pd.DataFrame,
    *,
    representations: list[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rep in representations:
        df = metrics[
            metrics["status"].eq("ok")
            & metrics["representation"].eq(rep)
            & metrics["subset"].eq("all")
            & metrics["target_name"].eq("behavior_clr")
            & metrics["r2"].notna()
        ].copy()
        if df.empty:
            continue
        df = df.sort_values("layer")
        ax.plot(df["layer"], df["r2"], marker="o", linewidth=1.8, label=rep)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-validated R2")
    ax.set_title("Representation comparison: behavior CLR decoding")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    *,
    meta: pd.DataFrame,
    metrics: pd.DataFrame,
    recommendations: dict[str, Any],
    layers: list[int],
    representations: list[str],
    group_col: str,
) -> None:
    ok = metrics[metrics["status"].eq("ok")]
    lines: list[str] = []
    lines.append("# Layer-wise Activation Decoding Report\n")
    lines.append("## Inputs\n")
    lines.append(f"- Prefix rows: {len(meta):,}")
    lines.append(f"- Decision steps: {meta['decision_id'].nunique() if 'decision_id' in meta.columns else 'NA'}")
    lines.append(f"- Trajectory IDs: {meta['trajectory_id'].nunique() if 'trajectory_id' in meta.columns else 'NA'}")
    lines.append(f"- Group column: `{group_col}`")
    lines.append(f"- Layers: `{layers}`")
    lines.append(f"- Representations: `{representations}`")
    lines.append("")

    lines.append("## Completed Fits\n")
    lines.append(f"- Metric rows: {len(metrics):,}")
    lines.append(f"- Successful rows: {len(ok):,}")
    lines.append(f"- Skipped rows: {int(metrics['status'].eq('skipped').sum()) if 'status' in metrics.columns else 0:,}")
    lines.append("")

    lines.append("## Recommendations\n")
    for key, value in recommendations.items():
        lines.append(f"- `{key}`: `{json.dumps(value)}`")
    lines.append("")

    def add_top_table(title: str, target_name: str, metric_col: str, subset: str = "all", n: int = 8) -> None:
        df = ok[
            ok["target_name"].eq(target_name)
            & ok["subset"].eq(subset)
            & ok[metric_col].notna()
        ].copy()
        lines.append(f"## {title}\n")
        if df.empty:
            lines.append("No successful fits.\n")
            return
        cols = ["layer", "representation", "subset", "target_name", metric_col, "n_rows", "n_groups", "n_folds"]
        table = df.sort_values(metric_col, ascending=False)[cols].head(n)
        lines.append(table.to_markdown(index=False))
        lines.append("")

    add_top_table("Top Behavior CLR Fits", "behavior_clr", "r2")
    add_top_table("Top Commitment Fits", "commitment_to_final", "r2")
    add_top_table("Top Strict Pre-action Behavior CLR Fits", "behavior_clr", "r2", subset="strict_before_any_action_mention")
    add_top_table("Top Agent Action Classifiers", "agent_action", "balanced_accuracy")

    skipped = metrics[metrics["status"].eq("skipped")].copy()
    if not skipped.empty:
        lines.append("## Common Skip Reasons\n")
        counts = skipped["skip_reason"].fillna("").value_counts().head(10)
        lines.append(counts.to_markdown())
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_recommendations(path: Path, recommendations: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recommendations, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--representation", default="delta_pre", help="Primary representation, comma list, or all")
    parser.add_argument(
        "--comparison_representations",
        default="raw,delta_pre,velocity",
        help="Representations included for comparison plots/metrics if files exist.",
    )
    parser.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    parser.add_argument("--group_by", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--output", default="outputs/activation/layer_sweep_metrics.csv")
    parser.add_argument("--predictions_output", default="outputs/activation/layer_sweep_predictions.parquet")
    parser.add_argument("--save_predictions", type=parse_bool, default=False)
    parser.add_argument("--figures_dir", default="figures/activation/layer_sweep")
    parser.add_argument("--report", default="outputs/activation/reports/layer_sweep_report.md")
    parser.add_argument("--best_config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logistic_c", type=float, default=1.0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--leave_one_group_out_threshold", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--run_svm", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    repo_root = infer_repo_root(metadata_path)
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = repo_root / features_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = repo_root / figures_dir
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    best_config_path = Path(args.best_config)
    if not best_config_path.is_absolute():
        best_config_path = repo_root / best_config_path
    predictions_path = Path(args.predictions_output)
    if not predictions_path.is_absolute():
        predictions_path = repo_root / predictions_path

    meta = pd.read_parquet(metadata_path)
    if "row_index" not in meta.columns:
        meta = meta.copy()
        meta.insert(0, "row_index", np.arange(len(meta), dtype=np.int64))
    if args.group_by not in meta.columns:
        fallback = "decision_id" if args.group_by == "trajectory_id" and "decision_id" in meta.columns else None
        if fallback is None:
            raise KeyError(f"Missing group column {args.group_by!r}")
        print(f"WARNING: missing {args.group_by}; falling back to {fallback}")
        group_col = fallback
    else:
        group_col = args.group_by

    if args.representation.lower() == "all":
        requested_reps = ["raw", "delta_pre", "velocity", "unit_delta_pre"]
    else:
        requested_reps = parse_str_list(args.representation)
    comparison_reps = parse_str_list(args.comparison_representations)
    representations = []
    for rep in [*requested_reps, *comparison_reps]:
        if rep not in representations:
            representations.append(rep)

    layer_filter = parse_int_list(args.layers)
    layers = discover_layers(features_dir, representations)
    if layer_filter is not None:
        layers = [layer for layer in layers if layer in set(layer_filter)]
    if not layers:
        raise FileNotFoundError("No matching layer feature arrays found")

    existing_reps = []
    for rep in representations:
        if any(feature_file(features_dir, layer, rep).exists() for layer in layers):
            existing_reps.append(rep)
        else:
            print(f"WARNING: no feature files found for representation {rep!r}; skipping")
    representations = existing_reps
    if not representations:
        raise FileNotFoundError("No requested representation feature arrays found")

    subsets = subset_masks(meta)
    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for rep in representations:
        for layer in layers:
            path = feature_file(features_dir, layer, rep)
            if not path.exists():
                print(f"WARNING: missing feature array {path}; skipping")
                continue
            print(f"Evaluating layer {layer}, representation {rep}...")
            X = np.load(path, mmap_mode="r")
            if X.shape[0] != len(meta):
                raise ValueError(f"Feature row count mismatch for {path}: {X.shape[0]} != {len(meta)}")
            layer_metrics, layer_predictions = evaluate_layer_representation(
                meta,
                X,
                layer=layer,
                representation=rep,
                subsets=subsets,
                group_col=group_col,
                ridge_alpha=args.ridge_alpha,
                logistic_c=args.logistic_c,
                n_splits=args.n_splits,
                logo_threshold=args.leave_one_group_out_threshold,
                random_state=args.random_state,
                max_iter=args.max_iter,
                run_svm=args.run_svm,
                save_predictions=args.save_predictions,
            )
            metrics.extend(layer_metrics)
            prediction_frames.extend(layer_predictions)

    metrics_df = pd.DataFrame(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

    if args.save_predictions:
        if prediction_frames:
            pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
        else:
            pd.DataFrame().to_parquet(predictions_path, index=False)

    primary_rep = requested_reps[0] if requested_reps else representations[0]
    if primary_rep not in representations:
        primary_rep = representations[0]

    plot_metric_by_layer(
        metrics_df,
        representation=primary_rep,
        subset="all",
        target_name="behavior_clr",
        metric_col="r2",
        ylabel="Cross-validated R2",
        title=f"Behavior CLR decoding ({primary_rep})",
        path=figures_dir / "r2_behavior_clr_by_layer.png",
    )
    plot_metric_by_layer(
        metrics_df,
        representation=primary_rep,
        subset="all",
        target_name="commitment_to_final",
        metric_col="r2",
        ylabel="Cross-validated R2",
        title=f"Commitment decoding ({primary_rep})",
        path=figures_dir / "r2_commitment_by_layer.png",
    )
    plot_metric_by_layer(
        metrics_df,
        representation=primary_rep,
        subset="all",
        target_name="agent_action",
        metric_col="balanced_accuracy",
        ylabel="Balanced accuracy",
        title=f"Agent action decoding ({primary_rep})",
        path=figures_dir / "agent_action_accuracy_by_layer.png",
    )
    plot_metric_by_layer(
        metrics_df,
        representation=primary_rep,
        subset="strict_before_any_action_mention",
        target_name="behavior_clr",
        metric_col="r2",
        ylabel="Cross-validated R2",
        title=f"Strict pre-action behavior CLR decoding ({primary_rep})",
        path=figures_dir / "strict_pre_action_r2_by_layer.png",
    )
    plot_representation_comparison(
        metrics_df,
        representations=[rep for rep in ["raw", "delta_pre", "velocity"] if rep in representations],
        path=figures_dir / "representation_comparison.png",
    )

    recommendations = {
        "best_layer_behavior_clr_prediction": best_row(
            metrics_df,
            subset="all",
            target_name="behavior_clr",
            metric_col="r2",
        ),
        "best_layer_commitment_prediction": best_row(
            metrics_df,
            subset="all",
            target_name="commitment_to_final",
            metric_col="r2",
        ),
        "best_layer_strict_pre_action_decoding": best_row(
            metrics_df,
            subset="strict_before_any_action_mention",
            target_name="behavior_clr",
            metric_col="r2",
        ),
        "best_representation": best_row(
            metrics_df,
            subset="all",
            target_name="behavior_clr",
            metric_col="r2",
        ),
    }
    write_recommendations(best_config_path, recommendations)
    write_report(
        report_path,
        meta=meta,
        metrics=metrics_df,
        recommendations=recommendations,
        layers=layers,
        representations=representations,
        group_col=group_col,
    )

    print(f"Wrote {output_path}")
    if args.save_predictions:
        print(f"Wrote {predictions_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {best_config_path}")
    print(f"Wrote figures under {figures_dir}")


if __name__ == "__main__":
    main()
