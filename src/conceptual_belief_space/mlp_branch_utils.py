"""Utilities for supervised MLP bottleneck branch probes.

This module intentionally treats the 2D coordinates as a supervised
nonlinear readout, not as an unsupervised activation manifold.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS, EPS
from .io import ensure_dir, read_json, write_json
from .metrics import balanced_accuracy_np, macro_f1_np, r2_score_np

Q_COLS = [f"q_{action}" for action in ACTIONS]
CLR_COLS = [f"clr_{action}" for action in ACTIONS]
ACTION_TO_INDEX = {action: i for i, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {i: action for action, i in ACTION_TO_INDEX.items()}
ACTION_COLORS = {"UP": "C0", "DOWN": "C1", "LEFT": "C2", "RIGHT": "C3"}
ACTION_VERTICES = {
    "UP": np.asarray([1.0, 1.0], dtype=np.float32),
    "DOWN": np.asarray([1.0, -1.0], dtype=np.float32),
    "LEFT": np.asarray([-1.0, 1.0], dtype=np.float32),
    "RIGHT": np.asarray([-1.0, -1.0], dtype=np.float32),
}


def now_run_id(prefix: str, **parts: object) -> str:
    body = "_".join(str(v).replace("/", "-") for v in parts.values())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{body}_{stamp}" if body else f"{prefix}_{stamp}"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def json_ready(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.nanmax(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32)


def normalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    denom = q.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= EPS
    denom = np.where(denom <= EPS, 1.0, denom)
    out = q / denom
    if bad.any():
        out[bad] = 1.0 / len(ACTIONS)
    return out.astype(np.float32)


def entropy_commitment(q: np.ndarray) -> np.ndarray:
    q = normalize_q(q)
    ent = -np.sum(q * np.log(np.clip(q, EPS, None)), axis=1) / np.log(len(ACTIONS))
    return (1.0 - ent).astype(np.float32)


def action_labels_to_indices(labels: Sequence[object]) -> np.ndarray:
    values = []
    for label in labels:
        text = str(label)
        if text not in ACTION_TO_INDEX:
            raise ValueError(f"invalid action label: {text!r}")
        values.append(ACTION_TO_INDEX[text])
    return np.asarray(values, dtype=np.int64)


def action_indices_to_labels(indices: Sequence[int]) -> np.ndarray:
    return np.asarray([INDEX_TO_ACTION[int(i)] for i in indices], dtype=object)


def top_actions_from_q(q: np.ndarray) -> np.ndarray:
    return action_indices_to_labels(np.argmax(q, axis=1))


def q_top_action_coverage(
    q_pred: np.ndarray,
    *,
    true_top_action: Sequence[object] | None = None,
    min_count: int = 1,
) -> dict[str, Any]:
    """Summarize whether argmax(q_pred) covers the actions present in truth."""

    q_p = normalize_q(q_pred)
    pred_top = np.asarray([str(v) for v in top_actions_from_q(q_p)], dtype=object)
    pred_counts = {action: int(np.sum(pred_top == action)) for action in ACTIONS}
    if true_top_action is None:
        true_counts = {action: 0 for action in ACTIONS}
        present_actions = ACTIONS
    else:
        true_top = np.asarray([str(v) for v in true_top_action], dtype=object)
        true_counts = {action: int(np.sum(true_top == action)) for action in ACTIONS}
        present_actions = [action for action in ACTIONS if true_counts[action] > 0]
        if not present_actions:
            present_actions = ACTIONS
    missing = [action for action in present_actions if pred_counts[action] < int(min_count)]
    return {
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "present_true_actions": present_actions,
        "missing_actions": missing,
        "num_predicted_actions": int(sum(count > 0 for count in pred_counts.values())),
        "min_predicted_count_on_present_true_actions": int(min(pred_counts[action] for action in present_actions)) if present_actions else 0,
        "has_action_collapse": bool(missing),
    }


def commitment_to_final(q: np.ndarray, agent_actions: Sequence[object]) -> np.ndarray:
    q = normalize_q(q)
    idx = action_labels_to_indices(agent_actions)
    return q[np.arange(len(q)), idx].astype(np.float32)


def branch_targets_from_q(q: np.ndarray) -> np.ndarray:
    vertices = np.stack([ACTION_VERTICES[action] for action in ACTIONS], axis=0)
    return normalize_q(q) @ vertices


def nearest_vertex(z_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.stack([ACTION_VERTICES[action] for action in ACTIONS], axis=0)
    z = np.asarray(z_plot, dtype=np.float32)
    distances = np.linalg.norm(z[:, None, :] - vertices[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    nearest = order[:, 0]
    margin = distances[np.arange(len(z)), order[:, 1]] - distances[np.arange(len(z)), order[:, 0]]
    return action_indices_to_labels(nearest), distances[np.arange(len(z)), nearest], margin.astype(np.float32)


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.std(x_arr) <= EPS or np.std(y_arr) <= EPS:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        val = spearmanr(x_arr, y_arr, nan_policy="omit").correlation
        return float(val) if np.isfinite(val) else float("nan")
    except Exception:
        xr = pd.Series(x_arr).rank(method="average").to_numpy()
        yr = pd.Series(y_arr).rank(method="average").to_numpy()
        return safe_pearson(xr, yr)


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, s))
    except Exception:
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"r2": float("nan"), "pearson": float("nan"), "spearman": float("nan"), "mse": float("nan"), "mae": float("nan")}
    return {
        "r2": r2_score_np(y, p),
        "pearson": safe_pearson(y, p),
        "spearman": safe_spearman(y, p),
        "mse": float(np.mean((y - p) ** 2)),
        "mae": float(np.mean(np.abs(y - p))),
    }


def classification_summary(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, Any]:
    true = np.asarray([str(v) for v in y_true], dtype=object)
    pred = np.asarray([str(v) for v in y_pred], dtype=object)
    matrix: dict[str, dict[str, int]] = {}
    for actual in ACTIONS:
        matrix[actual] = {}
        for predicted in ACTIONS:
            matrix[actual][predicted] = int(np.sum((true == actual) & (pred == predicted)))
    return {
        "accuracy": float(np.mean(true == pred)) if len(true) else float("nan"),
        "balanced_accuracy": balanced_accuracy_np(true, pred),
        "macro_f1": macro_f1_np(true, pred),
        "confusion_matrix_json": json.dumps(matrix, sort_keys=True),
    }


def q_metrics(q_true: np.ndarray, q_pred: np.ndarray, *, top_action: Sequence[object], agent_action: Sequence[object]) -> dict[str, float]:
    q_t = normalize_q(q_true)
    q_p = normalize_q(q_pred)
    pred_top = top_actions_from_q(q_p)
    true_top = np.asarray([str(v) for v in top_action], dtype=object)
    agent = np.asarray([str(v) for v in agent_action], dtype=object)
    kl = q_t * (np.log(np.clip(q_t, EPS, None)) - np.log(np.clip(q_p, EPS, None)))
    commit_true = commitment_to_final(q_t, agent)
    commit_pred = commitment_to_final(q_p, agent)
    ent_true = entropy_commitment(q_t)
    ent_pred = entropy_commitment(q_p)
    return {
        "q_kl": float(np.mean(np.sum(kl, axis=1))),
        "q_mse": float(np.mean((q_t - q_p) ** 2)),
        "q_top_action_accuracy": float(np.mean(pred_top == true_top)),
        "q_top_action_balanced_accuracy": balanced_accuracy_np(true_top, pred_top),
        "q_agent_action_accuracy": float(np.mean(pred_top == agent)),
        "q_agent_action_balanced_accuracy": balanced_accuracy_np(agent, pred_top),
        "commitment_to_final_pearson_from_q": safe_pearson(commit_true, commit_pred),
        "commitment_to_final_spearman_from_q": safe_spearman(commit_true, commit_pred),
        "entropy_commitment_pearson_from_q": safe_pearson(ent_true, ent_pred),
        "entropy_commitment_spearman_from_q": safe_spearman(ent_true, ent_pred),
    }


def bottleneck_metrics(
    z_plot: np.ndarray,
    *,
    q_true: np.ndarray,
    top_action: Sequence[object],
    agent_action: Sequence[object],
    entropy_commitment_values: Sequence[float],
    top_commitment_values: Sequence[float],
    progress: Sequence[float] | None = None,
) -> dict[str, float]:
    nearest, distances, margins = nearest_vertex(z_plot)
    true_top = np.asarray([str(v) for v in top_action], dtype=object)
    agent = np.asarray([str(v) for v in agent_action], dtype=object)
    radius = np.linalg.norm(np.asarray(z_plot, dtype=float), axis=1)
    q_targets = branch_targets_from_q(q_true)
    target_dist = np.linalg.norm(np.asarray(z_plot, dtype=float) - q_targets, axis=1)
    top_idx = action_labels_to_indices(true_top)
    vertices = np.stack([ACTION_VERTICES[action] for action in ACTIONS], axis=0)
    dist_to_top_vertex = np.linalg.norm(np.asarray(z_plot) - vertices[top_idx], axis=1)
    out = {
        "vertex_top_action_accuracy": float(np.mean(nearest == true_top)),
        "vertex_top_action_balanced_accuracy": balanced_accuracy_np(true_top, nearest),
        "vertex_agent_action_accuracy": float(np.mean(nearest == agent)),
        "vertex_agent_action_balanced_accuracy": balanced_accuracy_np(agent, nearest),
        "radius_commitment_spearman": safe_spearman(radius, entropy_commitment_values),
        "radius_progress_spearman": safe_spearman(radius, progress) if progress is not None else float("nan"),
        "distance_to_top_vertex_spearman_1_minus_top_commitment": safe_spearman(dist_to_top_vertex, 1.0 - np.asarray(top_commitment_values, dtype=float)),
        "anchor_target_mse": float(np.mean(target_dist**2)),
        "vertex_margin_mean": float(np.nanmean(margins)) if len(margins) else float("nan"),
        "branch_separation_ratio_top_action": branch_separation_ratio(z_plot, true_top),
        "branch_separation_ratio_agent_action": branch_separation_ratio(z_plot, agent),
    }
    try:
        from sklearn.metrics import silhouette_score

        if len(np.unique(true_top)) > 1 and len(z_plot) > len(np.unique(true_top)):
            out["silhouette_top_action"] = float(silhouette_score(z_plot, true_top))
        else:
            out["silhouette_top_action"] = float("nan")
        if len(np.unique(agent)) > 1 and len(z_plot) > len(np.unique(agent)):
            out["silhouette_agent_action"] = float(silhouette_score(z_plot, agent))
        else:
            out["silhouette_agent_action"] = float("nan")
    except Exception:
        out["silhouette_top_action"] = float("nan")
        out["silhouette_agent_action"] = float("nan")
    return out


def branch_separation_ratio(z: np.ndarray, labels: Sequence[object]) -> float:
    z_arr = np.asarray(z, dtype=float)
    labels_arr = np.asarray([str(v) for v in labels], dtype=object)
    centroids = []
    within = []
    for action in ACTIONS:
        mask = labels_arr == action
        if mask.any():
            centroid = z_arr[mask].mean(axis=0)
            centroids.append(centroid)
            within.extend(np.linalg.norm(z_arr[mask] - centroid, axis=1).tolist())
    if len(centroids) < 2 or not within:
        return float("nan")
    c = np.stack(centroids)
    dists = []
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            dists.append(float(np.linalg.norm(c[i] - c[j])))
    denom = float(np.mean(within))
    return float(np.mean(dists) / denom) if denom > EPS else float("nan")


def make_behavior_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    parquet_errors: list[str] = []
    for engine in ("pyarrow", "fastparquet"):
        try:
            df = pd.read_parquet(path, engine=engine).copy()
            break
        except Exception as exc:
            parquet_errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    else:
        try:
            import duckdb

            df = duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
        except Exception as exc:
            joined = "; ".join(parquet_errors)
            raise RuntimeError(
                f"failed to read behavior parquet {path}; pandas errors: {joined}; "
                f"duckdb: {type(exc).__name__}: {exc}"
            ) from exc
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["row_id"] = df["row_id"].astype(int)
    if "decision_id" not in df.columns:
        if {"trajectory_id", "step_id"}.issubset(df.columns):
            df["decision_id"] = df["trajectory_id"].astype(str) + "::" + df["step_id"].astype(str)
        else:
            raise ValueError("behavior table needs decision_id or trajectory_id + step_id")
    missing = [col for col in ["trajectory_id", "prefix_token_index", "progress", "agent_action", *Q_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"behavior table missing required columns: {missing}")
    q = normalize_q(df[Q_COLS].to_numpy())
    df[Q_COLS] = q
    if "top_action" not in df.columns:
        df["top_action"] = top_actions_from_q(q)
    if "top_commitment" not in df.columns:
        df["top_commitment"] = q.max(axis=1)
    if "entropy_commitment" not in df.columns:
        df["entropy_commitment"] = entropy_commitment(q)
    if "commitment_to_final" not in df.columns:
        df["commitment_to_final"] = commitment_to_final(q, df["agent_action"])
    if not set(CLR_COLS).issubset(df.columns):
        log_q = np.log(np.clip(q, EPS, None))
        clr = log_q - log_q.mean(axis=1, keepdims=True)
        for i, action in enumerate(ACTIONS):
            df[f"clr_{action}"] = clr[:, i]
    if "strict_before_any_action_mention" not in df.columns and "first_action_mention_token_index" in df.columns:
        df["strict_before_any_action_mention"] = df["prefix_token_index"] < df["first_action_mention_token_index"]
    if "strict_before_final_action_mention" not in df.columns and "first_final_action_mention_token_index" in df.columns:
        df["strict_before_final_action_mention"] = df["prefix_token_index"] < df["first_final_action_mention_token_index"]
    if "num_cot_tokens" not in df.columns:
        df["num_cot_tokens"] = np.nan
    return df.reset_index(drop=True)


def subset_mask(df: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all_prefixes":
        return np.ones(len(df), dtype=bool)
    if subset in {"post_final_action_mention", "after_first_final_action_mention"}:
        if "strict_before_final_action_mention" not in df.columns:
            raise ValueError("post_final_action_mention requires strict_before_final_action_mention")
        return ~df["strict_before_final_action_mention"].fillna(False).astype(bool).to_numpy()
    if subset in {"post_any_action_mention", "after_first_action_mention"}:
        if "strict_before_any_action_mention" not in df.columns:
            raise ValueError("post_any_action_mention requires strict_before_any_action_mention")
        return ~df["strict_before_any_action_mention"].fillna(False).astype(bool).to_numpy()
    if subset in df.columns:
        return df[subset].fillna(False).astype(bool).to_numpy()
    raise ValueError(f"subset {subset!r} is unavailable")


@dataclass
class SplitDef:
    split_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: list[str]
    test_groups: list[str]
    group_col: str


def load_or_create_splits(path: str | Path, df: pd.DataFrame, group_col: str, seed: int = 42, n_splits: int = 5) -> list[SplitDef]:
    path = Path(path)
    splits: list[SplitDef] = []
    if path.exists():
        obj = read_json(path)
        for i, item in enumerate(obj):
            split_id = int(item.get("split_id", item.get("fold", i)))
            if "train_indices" in item and "test_indices" in item:
                train = np.asarray(item["train_indices"], dtype=np.int64)
                test = np.asarray(item["test_indices"], dtype=np.int64)
            elif "train_row_ids" in item and "test_row_ids" in item:
                row_to_pos = {int(row_id): pos for pos, row_id in enumerate(df["row_id"].astype(int).tolist())}
                train = np.asarray([row_to_pos[int(v)] for v in item["train_row_ids"] if int(v) in row_to_pos], dtype=np.int64)
                test = np.asarray([row_to_pos[int(v)] for v in item["test_row_ids"] if int(v) in row_to_pos], dtype=np.int64)
            else:
                item_group_col = str(item.get("group_col", group_col))
                train_groups = item.get("train_groups", item.get(f"train_{item_group_col}s", []))
                test_groups = item.get("test_groups", item.get(f"test_{item_group_col}s", []))
                train = np.flatnonzero(df[item_group_col].astype(str).isin([str(v) for v in train_groups]).to_numpy())
                test = np.flatnonzero(df[item_group_col].astype(str).isin([str(v) for v in test_groups]).to_numpy())
            item_group_col = str(item.get("group_col", group_col))
            train_groups = sorted(df.iloc[train][item_group_col].astype(str).dropna().unique().tolist()) if len(train) else []
            test_groups = sorted(df.iloc[test][item_group_col].astype(str).dropna().unique().tolist()) if len(test) else []
            splits.append(SplitDef(split_id, train, test, train_groups, test_groups, item_group_col))
        return splits
    from sklearn.model_selection import GroupKFold

    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(max(2, n_splits), len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    for split_id, (train, test) in enumerate(gkf.split(np.arange(len(df)), groups=groups)):
        splits.append(
            SplitDef(
                split_id=split_id,
                train_indices=train.astype(np.int64),
                test_indices=test.astype(np.int64),
                train_groups=sorted(np.unique(groups[train]).tolist()),
                test_groups=sorted(np.unique(groups[test]).tolist()),
                group_col=group_col,
            )
        )
    return splits


def make_train_val_split(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    group_col: str,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    groups = df.iloc[train_indices][group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return train_indices, train_indices[:0]
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    val_groups = set(shuffled[:n_val].tolist())
    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    val = train_indices[val_mask]
    train = train_indices[~val_mask]
    return train, val


def class_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(ACTIONS)).astype(np.float32)
    weights = np.zeros(len(ACTIONS), dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = float(counts[nonzero].sum()) / (float(nonzero.sum()) * counts[nonzero])
    weights[~nonzero] = 0.0
    return weights


def fit_standardizer(features: np.ndarray, indices: np.ndarray, standardize: bool = True, eps: float = 1e-8) -> dict[str, np.ndarray]:
    if not standardize:
        return {"mean": np.zeros(features.shape[1], dtype=np.float32), "scale": np.ones(features.shape[1], dtype=np.float32)}
    x = np.asarray(features[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def transform_features(features: np.ndarray, indices: np.ndarray, scaler: dict[str, np.ndarray], l2_normalize: bool = False) -> np.ndarray:
    x = np.asarray(features[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    x = (x - scaler["mean"]) / scaler["scale"]
    if l2_normalize:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norm < EPS, 1.0, norm)
    return np.asarray(x, dtype=np.float32)


def progress_features(df: pd.DataFrame, indices: np.ndarray, scaler: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cols = ["progress", "prefix_token_index", "num_cot_tokens"]
    x = df.iloc[np.asarray(indices, dtype=np.int64)][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale < EPS, 1.0, scale).astype(np.float32)
        scaler = {"mean": mean, "scale": scale}
    x = (x - scaler["mean"]) / scaler["scale"]
    return x.astype(np.float32), scaler


def prediction_frame(
    base: pd.DataFrame,
    indices: np.ndarray,
    *,
    run_id: str,
    split_id: int,
    layer: int,
    representation: str,
    subset: str,
    model_type: str,
    split_role: str,
    q_pred: np.ndarray,
    commitment_pred: np.ndarray,
    z_raw: np.ndarray,
    z_plot: np.ndarray,
) -> pd.DataFrame:
    idx = np.asarray(indices, dtype=np.int64)
    cols = [
        "row_id",
        "trajectory_id",
        "decision_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        *Q_COLS,
        "entropy_commitment",
        "commitment_to_final",
        "top_commitment",
        "num_cot_tokens",
    ]
    existing = [col for col in cols if col in base.columns]
    out = base.iloc[idx][existing].copy()
    out.insert(0, "model_type", model_type)
    out.insert(0, "subset", subset)
    out.insert(0, "representation", representation)
    out.insert(0, "layer", int(layer))
    out.insert(0, "split_id", int(split_id))
    out.insert(0, "run_id", run_id)
    out["split_role"] = split_role
    q_norm = normalize_q(q_pred)
    for j, action in enumerate(ACTIONS):
        out[f"q_pred_{action}"] = q_norm[:, j]
    out["entropy_commitment_pred"] = np.asarray(commitment_pred, dtype=np.float32).reshape(-1)
    out["z1_raw"] = z_raw[:, 0]
    out["z2_raw"] = z_raw[:, 1]
    out["z1_plot"] = z_plot[:, 0]
    out["z2_plot"] = z_plot[:, 1]
    out["z_radius"] = np.linalg.norm(z_plot, axis=1)
    nearest, distance, margin = nearest_vertex(z_plot)
    out["nearest_vertex_action"] = nearest
    out["nearest_vertex_distance"] = distance
    out["vertex_margin"] = margin
    return out


def metric_row_from_predictions(
    pred: pd.DataFrame,
    *,
    run_id: str,
    split_id: int,
    layer: int,
    representation: str,
    subset: str,
    model_type: str,
    train_n: int,
    val_n: int,
    test_n: int,
    n_train_trajectories: int,
    n_val_trajectories: int,
    n_test_trajectories: int,
    args_dict: dict[str, Any],
    best_epoch: int | None,
    validation_composite: float | None,
    status: str = "ok",
    warnings: str = "",
) -> dict[str, Any]:
    test = pred[pred["split_role"].astype(str).eq("test")].copy()
    row: dict[str, Any] = {
        "run_id": run_id,
        "split_id": int(split_id),
        "layer": int(layer),
        "representation": representation,
        "subset": subset,
        "model_type": model_type,
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(test_n),
        "n_train_trajectories": int(n_train_trajectories),
        "n_val_trajectories": int(n_val_trajectories),
        "n_test_trajectories": int(n_test_trajectories),
        "hidden_dim": int(args_dict.get("hidden_dim", 0) or 0),
        "hidden_dim2": int(args_dict.get("hidden_dim2", 0) or 0),
        "hidden_dim_mlp": int(args_dict.get("hidden_dim", 0) or 0),
        "bottleneck_dim": int(args_dict.get("bottleneck_dim", 2) or 2),
        "lambda_q": float(args_dict.get("lambda_q", 0.0) or 0.0),
        "lambda_top_action": float(args_dict.get("lambda_top_action", 0.0) or 0.0),
        "lambda_agent_action": float(args_dict.get("lambda_agent_action", 0.0) or 0.0),
        "lambda_commitment": float(args_dict.get("lambda_commitment", 0.0) or 0.0),
        "lambda_anchor": float(args_dict.get("lambda_anchor", 0.0) or 0.0),
        "lambda_smooth": float(args_dict.get("lambda_smooth", 0.0) or 0.0),
        "lambda_radius": float(args_dict.get("lambda_radius", 0.0) or 0.0),
        "best_epoch": int(best_epoch) if best_epoch is not None else -1,
        "validation_composite": float(validation_composite) if validation_composite is not None else float("nan"),
        "status": status,
        "warnings": warnings,
    }
    if test.empty:
        return row
    q_true = test[Q_COLS].to_numpy(dtype=np.float32)
    q_pred = test[[f"q_pred_{a}" for a in ACTIONS]].to_numpy(dtype=np.float32)
    q_m = q_metrics(q_true, q_pred, top_action=test["top_action"], agent_action=test["agent_action"])
    q_coverage = q_top_action_coverage(q_pred, true_top_action=test["top_action"], min_count=1)
    commit_m = scalar_metrics(test["entropy_commitment"].to_numpy(), test["entropy_commitment_pred"].to_numpy())
    top_q = top_actions_from_q(q_pred)
    top_head = test["nearest_vertex_action"].to_numpy() if "top_head_pred" not in test.columns else test["top_head_pred"].to_numpy()
    agent_head = test["nearest_vertex_action"].to_numpy() if "agent_head_pred" not in test.columns else test["agent_head_pred"].to_numpy()
    top_head_m = classification_summary(test["top_action"], top_head)
    agent_head_m = classification_summary(test["agent_action"], agent_head)
    bottleneck_m = bottleneck_metrics(
        test[["z1_plot", "z2_plot"]].to_numpy(dtype=np.float32),
        q_true=q_true,
        top_action=test["top_action"],
        agent_action=test["agent_action"],
        entropy_commitment_values=test["entropy_commitment"],
        top_commitment_values=test["top_commitment"],
        progress=test["progress"],
    )
    high = test["entropy_commitment"].to_numpy(dtype=float) >= 0.7
    low = test["entropy_commitment"].to_numpy(dtype=float) <= 0.3
    mask = high | low
    auc = binary_auc(high[mask].astype(int), test.loc[mask, "entropy_commitment_pred"].to_numpy(dtype=float)) if mask.any() else float("nan")
    row.update(
        {
            "test_q_kl": q_m["q_kl"],
            "test_q_mse": q_m["q_mse"],
            "test_q_top_action_accuracy": q_m["q_top_action_accuracy"],
            "test_q_top_action_balanced_accuracy": q_m["q_top_action_balanced_accuracy"],
            "test_q_agent_action_accuracy": q_m["q_agent_action_accuracy"],
            "test_q_agent_action_balanced_accuracy": q_m["q_agent_action_balanced_accuracy"],
            "test_q_pred_top_num_actions": q_coverage["num_predicted_actions"],
            "test_q_pred_top_min_count_on_present_true_actions": q_coverage["min_predicted_count_on_present_true_actions"],
            "test_q_pred_top_action_collapse": q_coverage["has_action_collapse"],
            "test_q_pred_top_missing_actions_json": json.dumps(q_coverage["missing_actions"]),
            "test_top_head_accuracy": top_head_m["accuracy"],
            "test_top_head_balanced_accuracy": top_head_m["balanced_accuracy"],
            "test_top_head_macro_f1": top_head_m["macro_f1"],
            "test_agent_head_accuracy": agent_head_m["accuracy"],
            "test_agent_head_balanced_accuracy": agent_head_m["balanced_accuracy"],
            "test_agent_head_macro_f1": agent_head_m["macro_f1"],
            "test_commitment_spearman": commit_m["spearman"],
            "test_commitment_pearson": commit_m["pearson"],
            "test_commitment_r2": commit_m["r2"],
            "test_commitment_mse": commit_m["mse"],
            "test_commitment_mae": commit_m["mae"],
            "test_commitment_auc_high_low": auc,
            "test_radius_commitment_spearman": bottleneck_m["radius_commitment_spearman"],
            "test_radius_progress_spearman": bottleneck_m["radius_progress_spearman"],
            "test_vertex_top_action_accuracy": bottleneck_m["vertex_top_action_accuracy"],
            "test_vertex_top_action_balanced_accuracy": bottleneck_m["vertex_top_action_balanced_accuracy"],
            "test_vertex_agent_action_accuracy": bottleneck_m["vertex_agent_action_accuracy"],
            "test_vertex_agent_action_balanced_accuracy": bottleneck_m["vertex_agent_action_balanced_accuracy"],
            "test_silhouette_top_action": bottleneck_m["silhouette_top_action"],
            "test_silhouette_agent_action": bottleneck_m["silhouette_agent_action"],
            "test_branch_separation_ratio_top_action": bottleneck_m["branch_separation_ratio_top_action"],
            "test_branch_separation_ratio_agent_action": bottleneck_m["branch_separation_ratio_agent_action"],
            "test_anchor_target_mse": bottleneck_m["anchor_target_mse"],
            "top_head_confusion_matrix_json": top_head_m["confusion_matrix_json"],
            "agent_head_confusion_matrix_json": agent_head_m["confusion_matrix_json"],
        }
    )
    for action in ACTIONS:
        row[f"test_true_top_count_{action}"] = q_coverage["true_counts"][action]
        row[f"test_q_pred_top_count_{action}"] = q_coverage["pred_counts"][action]
        action_values = q_pred[:, ACTION_TO_INDEX[action]]
        row[f"test_mean_q_pred_{action}"] = float(np.mean(action_values))
        row[f"test_max_q_pred_{action}"] = float(np.max(action_values))
    return row


def save_best_configs(metrics: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    candidates = metrics[metrics["status"].astype(str).eq("ok")].copy() if "status" in metrics.columns else metrics.copy()
    main = candidates[candidates["model_type"].astype(str).eq("mlp_branch")].copy()
    criteria = {
        "best_q_top_action_balanced_accuracy": ("test_q_top_action_balanced_accuracy", False),
        "best_commitment_spearman": ("test_commitment_spearman", False),
        "best_radius_commitment_spearman": ("test_radius_commitment_spearman", False),
        "best_vertex_top_action_balanced_accuracy": ("test_vertex_top_action_balanced_accuracy", False),
        "strict_pre_action_q_top_action_accuracy": ("test_q_top_action_balanced_accuracy", False),
    }
    for name, (col, ascending) in criteria.items():
        frame = main
        if name.startswith("strict"):
            frame = frame[frame["subset"].astype(str).str.contains("strict", na=False)]
        if "q_top_action" in col and "test_q_pred_top_action_collapse" in frame.columns:
            noncollapsed = frame[~frame["test_q_pred_top_action_collapse"].fillna(False).astype(bool)]
            if not noncollapsed.empty:
                frame = noncollapsed
        if frame.empty or col not in frame.columns:
            continue
        frame = frame[np.isfinite(pd.to_numeric(frame[col], errors="coerce"))]
        if frame.empty:
            continue
        best = frame.sort_values(col, ascending=ascending).iloc[0].to_dict()
        configs[name] = json_ready(best)
    for baseline, name in [("linear_baseline", "best_mlp_vs_linear_improvement"), ("progress_only", "best_mlp_vs_progress_only_improvement")]:
        if main.empty:
            continue
        rows = []
        base = candidates[candidates["model_type"].astype(str).eq(baseline)]
        keys = ["split_id", "layer", "representation", "subset"]
        for _, mlp in main.iterrows():
            match = base
            for key in keys:
                match = match[match[key].astype(str).eq(str(mlp[key]))]
            if match.empty:
                continue
            delta = float(mlp.get("test_q_top_action_balanced_accuracy", np.nan)) - float(match.iloc[0].get("test_q_top_action_balanced_accuracy", np.nan))
            item = mlp.to_dict()
            item["delta_vs_" + baseline] = delta
            rows.append(item)
        if rows:
            best = pd.DataFrame(rows).sort_values("delta_vs_" + baseline, ascending=False).iloc[0].to_dict()
            configs[name] = json_ready(best)
    write_json(path, configs)
    return configs
