#!/usr/bin/env python3
"""Pre-reasoning vs post-reasoning activation analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}
Q_COLS = [f"q_{action}" for action in ACTIONS]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_layers(value: str, available_layers: list[int]) -> list[int]:
    if value.lower() == "all":
        return available_layers
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    missing = [layer for layer in parsed if layer not in available_layers]
    if missing:
        raise SystemExit(f"Requested unavailable layer(s): {missing}. Available: {available_layers}")
    return parsed


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


def selected_layer_from_activation_manifold(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["layer"])
    except Exception:
        return None
    values = sorted(pd.Series(df["layer"]).dropna().astype(int).unique().tolist())
    return values[0] if len(values) == 1 else None


def select_plot_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    activation_manifold_path: Path | None,
    available_layers: list[int],
) -> tuple[int, str]:
    if requested_layer is not None:
        if requested_layer not in available_layers:
            raise SystemExit(f"Requested plot layer {requested_layer} unavailable. Available: {available_layers}")
        return requested_layer, "explicit --plot_layer"
    if config_path is not None and config_path.exists():
        try:
            layer = recursively_find_layer(json.loads(config_path.read_text()))
            if layer in available_layers:
                return int(layer), f"best_activation_config: {config_path}"
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; trying activation manifold layer.")
    if activation_manifold_path is not None:
        layer = selected_layer_from_activation_manifold(activation_manifold_path)
        if layer in available_layers:
            return int(layer), f"activation manifold layer from {activation_manifold_path}"
    index = int(round(0.75 * (len(available_layers) - 1)))
    return available_layers[index], "mid-late fallback from available layers"


def path_for_layer(path_value: Any, layer: int) -> Path:
    text = str(path_value)
    if "{layer}" in text:
        return Path(text.format(layer=layer))
    return Path(re.sub(r"/layer_\d+/", f"/layer_{layer}/", text))


def bf16_uint16_to_float32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


class ActivationRowLoader:
    def __init__(self, *, max_open: int = 128):
        self.max_open = max_open
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def _get_bf16_memmap(self, path: Path, hidden_dim: int) -> np.memmap:
        key = (str(path), hidden_dim)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        size_bytes = path.stat().st_size
        if size_bytes % (hidden_dim * 2) != 0:
            raise ValueError(f"File size is not divisible by hidden_dim*2 for {path}")
        rows = size_bytes // (hidden_dim * 2)
        mmap = np.memmap(path, mode="r", dtype="<u2", shape=(rows, hidden_dim))
        self._cache[key] = mmap
        if len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return mmap

    def load_row(self, row: pd.Series, *, layer: int) -> np.ndarray:
        path = path_for_layer(row["activation_file"], layer)
        row_index = int(row["activation_row_index"])
        hidden_dim = int(row["hidden_dim"])
        fmt = str(row.get("activation_format", ""))
        suffix = path.suffix.lower()
        if suffix == ".bf16" or fmt == "packed_raw_bf16":
            mmap = self._get_bf16_memmap(path, hidden_dim)
            return bf16_uint16_to_float32(np.asarray(mmap[row_index]))
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            return np.asarray(arr[row_index], dtype=np.float32)
        if suffix == ".npz":
            key = row.get("activation_key") if row.get("activation_key") and row.get("activation_key") != "row" else None
            with np.load(path, allow_pickle=False) as data:
                if key is None:
                    key = data.files[0]
                return np.asarray(data[key][row_index], dtype=np.float32)
        if suffix in {".pt", ".pth"}:
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError("torch is required to read .pt/.pth activation files") from exc
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                key = row.get("activation_key") if row.get("activation_key") in obj else next(iter(obj))
                obj = obj[key]
            if hasattr(obj, "detach"):
                obj = obj.detach().cpu().numpy()
            return np.asarray(obj[row_index], dtype=np.float32)
        raise ValueError(f"Unsupported activation file format for {path}")


def available_layers_from_index(index: pd.DataFrame) -> list[int]:
    layers: set[int] = set()
    for raw in index["available_layers_json"].dropna().head(1000):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            layers.update(int(x) for x in parsed)
    if not layers:
        raise SystemExit("Could not infer available layers from activation index")
    return sorted(layers)


def behavior_decision_summary(prefix_metrics: pd.DataFrame) -> pd.DataFrame:
    prefix_metrics = make_decision_id(prefix_metrics)
    if "decision_id" not in prefix_metrics.columns:
        raise SystemExit("prefix commitment metrics must contain decision_id or trajectory_id + step_id")
    sort_cols = [col for col in ["decision_id", "prefix_token_index", "progress", "prefix_id"] if col in prefix_metrics.columns]
    ordered = prefix_metrics.sort_values(sort_cols).copy()
    rows: list[dict[str, Any]] = []
    for decision_id, group in ordered.groupby("decision_id", sort=False):
        first = group.iloc[0]
        final = group.iloc[-1]
        row = {
            "decision_id": decision_id,
            "trajectory_id": final.get("trajectory_id", pd.NA),
            "step_id": final.get("step_id", pd.NA),
            "agent_action": final.get("agent_action", pd.NA),
            "num_prefixes": int(len(group)),
            "prompt_only_prefix_id": first.get("prefix_id", pd.NA),
            "prompt_only_progress": float(first.get("progress", np.nan)),
            "prompt_only_commitment_to_final": float(first.get("commitment_to_final", np.nan)),
            "prompt_only_top_action": first.get("top_action", pd.NA),
            "final_prefix_id": final.get("prefix_id", pd.NA),
            "final_prefix_progress": float(final.get("progress", np.nan)),
            "final_prefix_commitment_to_final": float(final.get("commitment_to_final", np.nan)),
            "final_prefix_top_action": final.get("top_action", pd.NA),
            "final_prefix_entropy_commitment": float(final.get("entropy_commitment", np.nan)),
        }
        for action in ACTIONS:
            col = f"q_{action}"
            if col in first:
                row[f"prompt_only_q_{action}"] = float(first[col])
            if col in final:
                row[f"final_q_{action}"] = float(final[col])
        rows.append(row)
    return pd.DataFrame(rows)


def merge_permanent_summary(summary: pd.DataFrame, permanent_path: Path) -> pd.DataFrame:
    if not permanent_path.exists():
        print(f"WARNING: optional permanent lock-in summary not found: {permanent_path}")
        return summary
    permanent = make_decision_id(pd.read_parquet(permanent_path))
    keep = [
        col
        for col in [
            "decision_id",
            "permanent_lock_in_progress",
            "permanent_top_action_lock_progress",
            "switch_count_total",
            "final_prefix_is_top_action_final",
        ]
        if col in permanent.columns
    ]
    if "decision_id" not in keep:
        return summary
    return summary.merge(permanent[keep].drop_duplicates("decision_id"), on="decision_id", how="left")


def pre_post_pointer_table(index: pd.DataFrame, decisions: set[str]) -> pd.DataFrame:
    subset = index[index["decision_id"].isin(decisions) & index["token_type"].isin(["pre_reasoning", "post_reasoning"])].copy()
    pre = subset[subset["token_type"].eq("pre_reasoning")].sort_values(["decision_id", "activation_id"]).drop_duplicates("decision_id")
    post = subset[subset["token_type"].eq("post_reasoning")].sort_values(["decision_id", "activation_id"]).drop_duplicates("decision_id")
    rows: list[dict[str, Any]] = []
    common = sorted(set(pre["decision_id"]) & set(post["decision_id"]))
    pre_by = pre.set_index("decision_id")
    post_by = post.set_index("decision_id")
    for decision_id in common:
        p = pre_by.loc[decision_id]
        q = post_by.loc[decision_id]
        rows.append(
            {
                "decision_id": decision_id,
                "trajectory_id": p.get("trajectory_id", pd.NA),
                "step_id": p.get("step_id", pd.NA),
                "num_cot_tokens": p.get("num_cot_tokens", pd.NA),
                "pre_activation_id": p.get("activation_id", pd.NA),
                "pre_activation_file": p.get("activation_file", pd.NA),
                "pre_activation_row_index": p.get("activation_row_index", pd.NA),
                "post_activation_id": q.get("activation_id", pd.NA),
                "post_activation_file": q.get("activation_file", pd.NA),
                "post_activation_row_index": q.get("activation_row_index", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms < eps, 1.0, norms)


def build_raw_centroids(
    meta: pd.DataFrame,
    features_dir: Path,
    *,
    layer: int,
    threshold: float,
    min_progress: float | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    raw_path = features_dir / f"layer_{layer:02d}_raw.npy"
    if not raw_path.exists():
        return None, {"available": False, "reason": f"missing {raw_path}"}
    if "commitment_to_final" not in meta.columns or "agent_action" not in meta.columns:
        return None, {"available": False, "reason": "metadata lacks commitment_to_final or agent_action"}
    X_raw = np.load(raw_path, mmap_mode="r")
    if X_raw.shape[0] != len(meta):
        return None, {"available": False, "reason": "raw feature row count mismatch"}
    base = pd.to_numeric(meta["commitment_to_final"], errors="coerce").ge(threshold).to_numpy()
    if min_progress is not None:
        base &= pd.to_numeric(meta["progress"], errors="coerce").ge(min_progress).to_numpy()
    centroids = []
    counts: dict[str, int] = {}
    for action in ACTIONS:
        mask = base & meta["agent_action"].astype(str).eq(action).to_numpy()
        counts[action] = int(mask.sum())
        if counts[action] == 0:
            return None, {"available": False, "reason": f"no centroid rows for {action}", "counts": counts}
        centroids.append(np.asarray(X_raw[mask], dtype=np.float32).mean(axis=0))
    return np.vstack(centroids).astype(np.float32), {
        "available": True,
        "source": "raw_high_commitment_prefixes",
        "threshold": threshold,
        "min_progress": min_progress,
        "counts": counts,
    }


def load_pre_post_for_layer(pointer_behavior: pd.DataFrame, index: pd.DataFrame, *, layer: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    pre_rows = index[index["token_type"].eq("pre_reasoning")].drop_duplicates("decision_id").set_index("decision_id")
    post_rows = index[index["token_type"].eq("post_reasoning")].drop_duplicates("decision_id").set_index("decision_id")
    loader = ActivationRowLoader(max_open=256)
    rows: list[dict[str, Any]] = []
    pre_vectors: list[np.ndarray] = []
    post_vectors: list[np.ndarray] = []
    for _, base in pointer_behavior.iterrows():
        decision_id = base["decision_id"]
        if decision_id not in pre_rows.index or decision_id not in post_rows.index:
            continue
        pre_row = pre_rows.loc[decision_id]
        post_row = post_rows.loc[decision_id]
        h_pre = loader.load_row(pre_row, layer=layer).astype(np.float32, copy=False)
        h_post = loader.load_row(post_row, layer=layer).astype(np.float32, copy=False)
        pre_vectors.append(h_pre)
        post_vectors.append(h_post)
        row = base.to_dict()
        row.update(
            {
                "layer": int(layer),
                "pre_activation_file": str(path_for_layer(pre_row["activation_file"], layer)),
                "pre_activation_row_index": int(pre_row["activation_row_index"]),
                "post_activation_file": str(path_for_layer(post_row["activation_file"], layer)),
                "post_activation_row_index": int(post_row["activation_row_index"]),
                "hidden_dim": int(pre_row["hidden_dim"]),
            }
        )
        rows.append(row)
    if not rows:
        raise SystemExit(f"No pre/post rows could be loaded for layer {layer}")
    pre = np.vstack(pre_vectors).astype(np.float32)
    post = np.vstack(post_vectors).astype(np.float32)
    delta = post - pre
    return pd.DataFrame(rows), pre, post, delta


def add_geometry_features(df: pd.DataFrame, pre: np.ndarray, post: np.ndarray, delta: np.ndarray, centroids: np.ndarray | None) -> pd.DataFrame:
    out = df.copy()
    out["delta_post_pre_l2"] = np.linalg.norm(delta, axis=1)
    out["pre_post_cosine"] = np.asarray([cosine(a, b) for a, b in zip(pre, post)], dtype=float)
    out["pre_l2"] = np.linalg.norm(pre, axis=1)
    out["post_l2"] = np.linalg.norm(post, axis=1)
    if centroids is not None:
        centroid_norm = row_normalize(centroids)
        pre_sim = row_normalize(pre) @ centroid_norm.T
        post_sim = row_normalize(post) @ centroid_norm.T
        for idx, action in enumerate(ACTIONS):
            out[f"pre_sim_{action}"] = pre_sim[:, idx]
            out[f"post_sim_{action}"] = post_sim[:, idx]
        final_pre = np.full(len(out), np.nan, dtype=float)
        final_post = np.full(len(out), np.nan, dtype=float)
        for idx, action in enumerate(ACTIONS):
            mask = out["agent_action"].astype(str).eq(action).to_numpy()
            final_pre[mask] = pre_sim[mask, idx]
            final_post[mask] = post_sim[mask, idx]
        out["pre_sim_final_centroid"] = final_pre
        out["post_sim_final_centroid"] = final_post
        out["pre_centroid_basin_action"] = np.asarray(ACTIONS, dtype=object)[np.argmax(pre_sim, axis=1)]
        out["post_centroid_basin_action"] = np.asarray(ACTIONS, dtype=object)[np.argmax(post_sim, axis=1)]
        out["pre_centroid_basin_matches_final"] = out["pre_centroid_basin_action"].astype(str).eq(out["agent_action"].astype(str))
        out["post_centroid_basin_matches_final"] = out["post_centroid_basin_action"].astype(str).eq(out["agent_action"].astype(str))
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


def fit_classification_cv(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, group_col: str, n_splits: int, max_iter: int) -> tuple[dict[str, Any], np.ndarray | None]:
    mask = pd.Series(y).notna().to_numpy() & df[group_col].notna().to_numpy()
    X = X[mask]
    y = np.asarray(y, dtype=object)[mask]
    sub = df.loc[mask].reset_index(drop=True)
    labels = sorted(pd.Series(y).dropna().unique().tolist(), key=str)
    if len(labels) < 2 or len(sub) < 4:
        return {"status": "skipped", "skip_reason": "not enough classes or rows", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}, None
    splits, cv_name = group_splits(sub, group_col, n_splits)
    if not splits:
        return {"status": "skipped", "skip_reason": cv_name, "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}, None
    pred = np.full(len(sub), None, dtype=object)
    for train_idx, test_idx in splits:
        if len(pd.Series(y[train_idx]).unique()) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=max_iter),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    valid = pd.Series(pred).notna().to_numpy()
    if not valid.any():
        return {"status": "skipped", "skip_reason": "no valid folds", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}, None
    metrics = {
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
    return metrics, pred


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def fit_regression_cv(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, group_col: str, n_splits: int, alpha: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & df[group_col].notna().to_numpy()
    X = X[mask]
    y = y[mask]
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 4:
        return {"status": "skipped", "skip_reason": "not enough rows", "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    splits, cv_name = group_splits(sub, group_col, n_splits)
    if not splits:
        return {"status": "skipped", "skip_reason": cv_name, "n_rows": len(sub), "n_groups": int(sub[group_col].nunique())}
    pred = np.full(len(sub), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    valid = np.isfinite(pred)
    return {
        "status": "ok",
        "skip_reason": "",
        "n_rows": int(valid.sum()),
        "n_groups": int(sub[group_col].nunique()),
        "n_folds": len(splits),
        "cv_method": cv_name,
        "r2": float(r2_score(y[valid], pred[valid], force_finite=True)),
        "mse": float(mean_squared_error(y[valid], pred[valid])),
        "pearson": pearson(y[valid], pred[valid]),
    }


def decoding_metrics_for_layer(
    layer: int,
    features: pd.DataFrame,
    pre: np.ndarray,
    post: np.ndarray,
    delta: np.ndarray,
    *,
    group_col: str,
    n_splits: int,
    max_iter: int,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, X in [("pre", pre), ("post", post), ("delta_post_pre", delta)]:
        metrics, _ = fit_classification_cv(X, features["agent_action"].to_numpy(), features, group_col, n_splits, max_iter)
        metrics.update({"layer": layer, "input_state": name, "target": "agent_action", "model": "logistic_regression"})
        rows.append(metrics)
    metrics = fit_regression_cv(
        pre,
        features["prompt_only_commitment_to_final"].to_numpy(dtype=float),
        features,
        group_col,
        n_splits,
        ridge_alpha,
    )
    metrics.update({"layer": layer, "input_state": "pre", "target": "prompt_only_commitment_to_final", "model": "ridge"})
    rows.append(metrics)
    metrics = fit_regression_cv(
        post,
        features["final_prefix_commitment_to_final"].to_numpy(dtype=float),
        features,
        group_col,
        n_splits,
        ridge_alpha,
    )
    metrics.update({"layer": layer, "input_state": "post", "target": "final_prefix_commitment_to_final", "model": "ridge"})
    rows.append(metrics)
    return rows


def plot_action_decoding(metrics: pd.DataFrame, path: Path) -> None:
    df = metrics[(metrics["target"].eq("agent_action")) & (metrics["status"].eq("ok"))].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for state, label in [("pre", "pre"), ("post", "post"), ("delta_post_pre", "post-minus-pre")]:
        sub = df[df["input_state"].eq(state)].sort_values("layer")
        if not sub.empty:
            ax.plot(sub["layer"], sub["balanced_accuracy"], marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Pre vs post action decoding by layer")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_norm_by_action(features: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [features.loc[features["agent_action"].astype(str).eq(action), "delta_post_pre_l2"].dropna().to_numpy() for action in ACTIONS]
    ax.boxplot(data, tick_labels=ACTIONS, patch_artist=True)
    ax.set_xlabel("Final action")
    ax.set_ylabel("||h_post - h_pre||")
    ax.set_title("Post-minus-pre norm by final action")
    ax.grid(True, axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pre_post_centroid_similarity(features: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if {"pre_sim_final_centroid", "post_sim_final_centroid"}.issubset(features.columns):
        data = [features["pre_sim_final_centroid"].dropna().to_numpy(), features["post_sim_final_centroid"].dropna().to_numpy()]
        ax.boxplot(data, tick_labels=["pre", "post"], patch_artist=True)
    else:
        ax.text(0.5, 0.5, "Centroid similarities unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Similarity to final-action centroid")
    ax.set_title("Pre vs post final-centroid similarity")
    ax.grid(True, axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(features: pd.DataFrame, x_col: str, y_col: str, path: Path, *, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if {x_col, y_col}.issubset(features.columns):
        ax.scatter(features[x_col], features[y_col], s=24, alpha=0.7, linewidths=0)
    else:
        ax.text(0.5, 0.5, "Required columns unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pre_post_pca(features: pd.DataFrame, pre: np.ndarray, post: np.ndarray, path: Path, *, max_lines: int, random_state: int) -> None:
    stacked = np.vstack([pre, post]).astype(np.float32)
    coords = PCA(n_components=2, random_state=random_state).fit_transform(StandardScaler().fit_transform(stacked))
    n = len(features)
    pre_xy = coords[:n]
    post_xy = coords[n:]
    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(random_state)
    line_indices = rng.choice(n, size=min(max_lines, n), replace=False)
    for idx in line_indices:
        ax.plot([pre_xy[idx, 0], post_xy[idx, 0]], [pre_xy[idx, 1], post_xy[idx, 1]], color="black", alpha=0.15, linewidth=0.7)
    for action in ACTIONS:
        mask = features["agent_action"].astype(str).eq(action).to_numpy()
        if mask.any():
            ax.scatter(pre_xy[mask, 0], pre_xy[mask, 1], s=22, marker="o", color=ACTION_COLORS[action], alpha=0.65, label=f"pre {action}", linewidths=0)
            ax.scatter(post_xy[mask, 0], post_xy[mask, 1], s=32, marker="x", color=ACTION_COLORS[action], alpha=0.85, label=f"post {action}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of pre and post reasoning states")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
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
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    layers: list[int],
    plot_layer: int,
    plot_layer_source: str,
    centroid_info_by_layer: dict[int, dict[str, Any]],
    figures: list[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Pre/Post Reasoning Activation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Feature rows: {len(features):,}")
    lines.append(f"- Decisions: {features['decision_id'].nunique() if 'decision_id' in features.columns else 'NA'}")
    lines.append(f"- Trajectories: {features['trajectory_id'].nunique() if 'trajectory_id' in features.columns else 'NA'}")
    lines.append(f"- Layers analyzed: `{layers}`")
    lines.append(f"- Plot/reference layer: {plot_layer}")
    lines.append(f"- Plot/reference layer source: {plot_layer_source}")
    lines.append("")

    selected = features[features["layer"].eq(plot_layer)]
    lines.append("## Geometry at Reference Layer\n")
    lines.append(f"- Mean ||h_post - h_pre||: {fmt(selected['delta_post_pre_l2'].mean())}")
    lines.append(f"- Median ||h_post - h_pre||: {fmt(selected['delta_post_pre_l2'].median())}")
    lines.append(f"- Mean cos(h_pre, h_post): {fmt(selected['pre_post_cosine'].mean())}")
    if "pre_sim_final_centroid" in selected.columns:
        lines.append(f"- Mean pre similarity to final-action centroid: {fmt(selected['pre_sim_final_centroid'].mean())}")
        lines.append(f"- Mean post similarity to final-action centroid: {fmt(selected['post_sim_final_centroid'].mean())}")
    lines.append("")

    lines.append("## Decoding Highlights\n")
    ok = metrics[metrics["status"].eq("ok")]
    for state in ["pre", "post", "delta_post_pre"]:
        sub = ok[(ok["target"].eq("agent_action")) & (ok["input_state"].eq(state))].copy()
        if not sub.empty:
            best = sub.sort_values("balanced_accuracy", ascending=False).iloc[0]
            lines.append(
                f"- Best `{state} -> agent_action`: layer {int(best['layer'])}, "
                f"balanced accuracy {fmt(best['balanced_accuracy'])}"
            )
    for target in ["prompt_only_commitment_to_final", "final_prefix_commitment_to_final"]:
        sub = ok[ok["target"].eq(target)].copy()
        if not sub.empty:
            best = sub.sort_values("r2", ascending=False).iloc[0]
            lines.append(f"- Best `{target}` regression: layer {int(best['layer'])}, R2 {fmt(best['r2'])}, Pearson {fmt(best['pearson'])}")
    lines.append("")

    lines.append("## Interpretation\n")
    pre_best = ok[(ok["target"].eq("agent_action")) & (ok["input_state"].eq("pre"))]
    post_best = ok[(ok["target"].eq("agent_action")) & (ok["input_state"].eq("post"))]
    delta_best = ok[(ok["target"].eq("agent_action")) & (ok["input_state"].eq("delta_post_pre"))]
    if not pre_best.empty:
        lines.append("- If `h_pre` decodes action above chance, the prompt state carries an internal action prior.")
    if not pre_best.empty and not post_best.empty:
        delta_acc = post_best["balanced_accuracy"].max() - pre_best["balanced_accuracy"].max()
        lines.append(f"- Best post-minus-pre improvement in action balanced accuracy: {fmt(delta_acc)}.")
    if not delta_best.empty:
        lines.append("- If `delta_post_pre` decodes action, the reasoning-induced state change is action-specific.")
    lines.append("")

    lines.append("## Centroid Availability\n")
    for layer, info in centroid_info_by_layer.items():
        lines.append(f"- Layer {layer}: `{json.dumps(info, default=str)}`")
    lines.append("")

    lines.append("## Metrics Table\n")
    display_cols = [
        "layer",
        "input_state",
        "target",
        "model",
        "status",
        "n_rows",
        "n_groups",
        "balanced_accuracy",
        "macro_f1",
        "r2",
        "pearson",
    ]
    display_cols = [col for col in display_cols if col in metrics.columns]
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
    parser.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--activation_manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--prefix_commitment_metrics", default="outputs/prefix_commitment_metrics.parquet")
    parser.add_argument("--permanent_lockin_summary", default="outputs/followups/permanent_lockin_summary.parquet")
    parser.add_argument("--output", default="outputs/activation/pre_post_reasoning_features.parquet")
    parser.add_argument("--metrics_output", default="outputs/activation/pre_post_decoding_metrics.csv")
    parser.add_argument("--report", default="outputs/activation/reports/pre_post_reasoning_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/pre_post")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--plot_layer", type=int, default=None)
    parser.add_argument("--group_by", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--centroid_commitment_threshold", type=float, default=0.9)
    parser.add_argument("--centroid_min_progress", type=float, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--pca_max_lines", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation_index_path = Path(args.activation_index)
    repo_root = infer_repo_root(activation_index_path)
    activation_index_path = resolve_path(activation_index_path, repo_root)
    metadata_path = resolve_path(args.metadata, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root) if args.config else None
    activation_manifold_path = resolve_path(args.activation_manifold, repo_root) if args.activation_manifold else None
    prefix_metrics_path = resolve_path(args.prefix_commitment_metrics, repo_root)
    permanent_path = resolve_path(args.permanent_lockin_summary, repo_root)
    output_path = resolve_path(args.output, repo_root)
    metrics_output_path = resolve_path(args.metrics_output, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    activation_index = pd.read_parquet(activation_index_path)
    activation_index = make_decision_id(activation_index)
    available_layers = available_layers_from_index(activation_index)
    layers = parse_layers(args.layers, available_layers)
    plot_layer, plot_layer_source = select_plot_layer(
        requested_layer=args.plot_layer,
        config_path=config_path,
        activation_manifold_path=activation_manifold_path,
        available_layers=available_layers,
    )
    if plot_layer not in layers:
        layers = sorted(set(layers + [plot_layer]))

    metadata = make_decision_id(pd.read_parquet(metadata_path))
    prefix_metrics = pd.read_parquet(prefix_metrics_path)
    behavior = behavior_decision_summary(prefix_metrics)
    behavior = merge_permanent_summary(behavior, permanent_path)

    pointers = pre_post_pointer_table(activation_index, set(behavior["decision_id"]))
    pointer_behavior = behavior.merge(pointers, on=["decision_id", "trajectory_id", "step_id"], how="inner")
    if pointer_behavior.empty:
        raise SystemExit("No behavior decisions had matching pre/post activation pointers")
    if args.group_by not in pointer_behavior.columns:
        if args.group_by == "trajectory_id" and "decision_id" in pointer_behavior.columns:
            group_col = "decision_id"
            print("WARNING: trajectory_id unavailable; using decision_id groups")
        else:
            raise SystemExit(f"Missing group column {args.group_by}")
    else:
        group_col = args.group_by

    all_feature_rows: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, Any]] = []
    centroid_info_by_layer: dict[int, dict[str, Any]] = {}
    plot_layer_payload: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray] | None = None

    for layer in layers:
        print(f"Loading pre/post activations and fitting decoders for layer {layer}...")
        layer_df, pre, post, delta = load_pre_post_for_layer(pointer_behavior, activation_index, layer=layer)
        centroids, centroid_info = build_raw_centroids(
            metadata,
            features_dir,
            layer=layer,
            threshold=args.centroid_commitment_threshold,
            min_progress=args.centroid_min_progress,
        )
        centroid_info_by_layer[layer] = centroid_info
        layer_features = add_geometry_features(layer_df, pre, post, delta, centroids)
        all_feature_rows.append(layer_features)
        all_metric_rows.extend(
            decoding_metrics_for_layer(
                layer,
                layer_features,
                pre,
                post,
                delta,
                group_col=group_col,
                n_splits=args.n_splits,
                max_iter=args.max_iter,
                ridge_alpha=args.ridge_alpha,
            )
        )
        if layer == plot_layer:
            plot_layer_payload = (layer_features, pre, post, delta)

    features = pd.concat(all_feature_rows, ignore_index=True)
    metrics = pd.DataFrame(all_metric_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_output_path, index=False)

    if plot_layer_payload is None:
        raise RuntimeError("plot layer payload missing after analysis")
    plot_features, plot_pre, plot_post, plot_delta = plot_layer_payload
    figures: list[Path] = []
    fig = figures_dir / "pre_vs_post_action_decoding_by_layer.png"
    plot_action_decoding(metrics, fig)
    figures.append(fig)
    fig = figures_dir / "post_minus_pre_norm_by_action.png"
    plot_delta_norm_by_action(plot_features, fig)
    figures.append(fig)
    fig = figures_dir / "pre_vs_post_final_centroid_similarity.png"
    plot_pre_post_centroid_similarity(plot_features, fig)
    figures.append(fig)
    fig = figures_dir / "prompt_only_commitment_vs_pre_activation_similarity.png"
    scatter_plot(
        plot_features,
        "prompt_only_commitment_to_final",
        "pre_sim_final_centroid",
        fig,
        xlabel="Prompt-only behavior commitment",
        ylabel="Pre activation final-centroid similarity",
        title="Prompt-only commitment vs pre activation similarity",
    )
    figures.append(fig)
    fig = figures_dir / "final_commitment_vs_post_activation_similarity.png"
    scatter_plot(
        plot_features,
        "final_prefix_commitment_to_final",
        "post_sim_final_centroid",
        fig,
        xlabel="Final-prefix behavior commitment",
        ylabel="Post activation final-centroid similarity",
        title="Final commitment vs post activation similarity",
    )
    figures.append(fig)
    fig = figures_dir / "pre_post_pca_trajectories.png"
    plot_pre_post_pca(plot_features, plot_pre, plot_post, fig, max_lines=args.pca_max_lines, random_state=args.random_state)
    figures.append(fig)

    write_report(
        report_path,
        features=features,
        metrics=metrics,
        layers=layers,
        plot_layer=plot_layer,
        plot_layer_source=plot_layer_source,
        centroid_info_by_layer=centroid_info_by_layer,
        figures=figures,
    )

    print(f"Analyzed {len(pointer_behavior)} decisions across {len(layers)} layer(s): {layers}")
    print(f"Reference plot layer {plot_layer} ({plot_layer_source})")
    print(f"Wrote {output_path}")
    print(f"Wrote {metrics_output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(figures)} figures under {figures_dir}")


if __name__ == "__main__":
    main()
