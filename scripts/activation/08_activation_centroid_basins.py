#!/usr/bin/env python3
"""Activation action-centroid and basin analysis for gridworld commitment."""

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


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}
PROGRESS_BINS = np.linspace(0.0, 1.0, 21)


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


def selected_layer_from_activation_manifold(path: Path, representation: str) -> int | None:
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


def select_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    activation_manifold_path: Path | None,
    features_dir: Path,
    representation: str,
) -> tuple[int, str]:
    available = discover_layers(features_dir, representation)
    if not available:
        raise SystemExit(f"No feature arrays found for representation {representation!r} in {features_dir}")
    if requested_layer is not None:
        if requested_layer not in available:
            raise SystemExit(f"Requested layer {requested_layer} unavailable. Available layers: {available}")
        return requested_layer, "explicit --layer"
    if config_path is not None and config_path.exists():
        try:
            layer = recursively_find_layer(json.loads(config_path.read_text()))
            if layer in available:
                return int(layer), f"best_activation_config: {config_path}"
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; trying activation manifold layer.")
    if activation_manifold_path is not None:
        layer = selected_layer_from_activation_manifold(activation_manifold_path, representation)
        if layer in available:
            return int(layer), f"activation manifold layer from {activation_manifold_path}"
    index = int(round(0.75 * (len(available) - 1)))
    return available[index], "mid-late fallback from available feature layers"


def make_decision_id(df: pd.DataFrame) -> pd.DataFrame:
    if "decision_id" not in df.columns and {"trajectory_id", "step_id"}.issubset(df.columns):
        out = df.copy()
        out["decision_id"] = out["trajectory_id"].astype(str) + "::" + out["step_id"].astype(str)
        return out
    return df


def row_l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms < eps, 1.0, norms)


def stable_softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    y = np.asarray(x, dtype=float) / temperature
    y = y - np.max(y, axis=1, keepdims=True)
    exp_y = np.exp(y)
    return exp_y / exp_y.sum(axis=1, keepdims=True)


def final_prefix_mask(meta: pd.DataFrame) -> np.ndarray:
    sort_cols = [col for col in ["decision_id", "prefix_token_index", "progress", "prefix_id"] if col in meta.columns]
    if "decision_id" not in meta.columns:
        raise SystemExit("final_prefixes centroid source requires decision_id")
    ordered = meta.reset_index().sort_values(sort_cols)
    final_indices = ordered.groupby("decision_id", sort=False)["index"].tail(1).to_numpy()
    mask = np.zeros(len(meta), dtype=bool)
    mask[final_indices] = True
    return mask


def post_reasoning_mask(meta: pd.DataFrame) -> np.ndarray:
    for col in ["token_type", "activation_token_type"]:
        if col in meta.columns:
            return meta[col].astype(str).eq("post_reasoning").to_numpy()
    return np.zeros(len(meta), dtype=bool)


def centroid_source_mask(
    meta: pd.DataFrame,
    *,
    source: str,
    threshold: float,
    min_progress: float | None,
) -> np.ndarray:
    if source == "high_commitment_prefixes":
        if "commitment_to_final" not in meta.columns:
            raise SystemExit("high_commitment_prefixes requires commitment_to_final")
        mask = pd.to_numeric(meta["commitment_to_final"], errors="coerce").ge(threshold).to_numpy()
        if min_progress is not None:
            if "progress" not in meta.columns:
                raise SystemExit("--centroid_min_progress requires progress")
            mask &= pd.to_numeric(meta["progress"], errors="coerce").ge(min_progress).to_numpy()
        return mask
    if source == "final_prefixes":
        return final_prefix_mask(meta)
    if source == "post_reasoning":
        return post_reasoning_mask(meta)
    raise ValueError(f"Unknown centroid source: {source}")


def build_centroids(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    source: str,
    threshold: float,
    min_progress: float | None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    source_mask = centroid_source_mask(meta, source=source, threshold=threshold, min_progress=min_progress)
    centroids: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for action in ACTIONS:
        mask = source_mask & meta["agent_action"].astype(str).eq(action).to_numpy()
        counts[action] = int(mask.sum())
        if counts[action] == 0:
            raise SystemExit(
                f"No centroid rows for action {action}. Try --centroid_source final_prefixes, "
                "lower --centroid_commitment_threshold, or remove --centroid_min_progress."
            )
        centroids.append(np.asarray(X[mask], dtype=np.float32).mean(axis=0))
    centroid_matrix = np.vstack(centroids).astype(np.float32)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1)
    metadata = {
        "source": source,
        "centroid_commitment_threshold": threshold,
        "centroid_min_progress": min_progress,
        "counts_by_action": counts,
        "centroid_l2_norms": {action: float(norm) for action, norm in zip(ACTIONS, centroid_norms)},
        "total_source_rows": int(source_mask.sum()),
    }
    return centroid_matrix, metadata, source_mask


def compute_scores(
    X: np.ndarray,
    meta: pd.DataFrame,
    centroids: np.ndarray,
    *,
    commitment_mode: str,
    softmax_temperature: float,
) -> pd.DataFrame:
    X_norm = row_l2_normalize(np.asarray(X, dtype=np.float32))
    C_norm = row_l2_normalize(np.asarray(centroids, dtype=np.float32))
    sims = X_norm @ C_norm.T
    sim_df = pd.DataFrame({f"act_sim_{action}": sims[:, idx] for idx, action in enumerate(ACTIONS)})
    top_idx = np.argmax(sims, axis=1)
    sorted_sims = np.sort(sims, axis=1)
    margin = sorted_sims[:, -1] - sorted_sims[:, -2]
    softmax_sims = stable_softmax(sims, softmax_temperature)

    out = meta.copy()
    for col in sim_df.columns:
        out[col] = sim_df[col].to_numpy(dtype=float)
    for idx, action in enumerate(ACTIONS):
        out[f"act_sim_softmax_{action}"] = softmax_sims[:, idx]
    out["activation_basin_action"] = np.asarray(ACTIONS, dtype=object)[top_idx]
    out["activation_margin"] = margin

    action_to_index = {action: idx for idx, action in enumerate(ACTIONS)}
    final_sim = np.full(len(out), np.nan, dtype=float)
    final_prob = np.full(len(out), np.nan, dtype=float)
    for action, idx in action_to_index.items():
        mask = out["agent_action"].astype(str).eq(action).to_numpy()
        final_sim[mask] = sims[mask, idx]
        final_prob[mask] = softmax_sims[mask, idx]
    out["act_sim_final"] = final_sim
    out["activation_commitment_to_final_softmax"] = final_prob
    out["activation_commitment_to_final"] = final_prob if commitment_mode == "softmax" else final_sim
    out["activation_basin_matches_final"] = out["activation_basin_action"].astype(str).eq(out["agent_action"].astype(str))
    if "top_action" in out.columns:
        out["activation_basin_matches_behavior_top"] = out["activation_basin_action"].astype(str).eq(out["top_action"].astype(str))
    else:
        out["activation_basin_matches_behavior_top"] = pd.NA
    return out


def first_consecutive_lock(group: pd.DataFrame, *, margin_threshold: float, consecutive: int) -> float:
    cond = (
        group["activation_basin_matches_final"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(group["activation_margin"], errors="coerce").ge(margin_threshold).to_numpy()
    )
    progress = pd.to_numeric(group["progress"], errors="coerce").to_numpy(dtype=float)
    if len(cond) < consecutive:
        return float("nan")
    for start in range(0, len(cond) - consecutive + 1):
        if cond[start : start + consecutive].all():
            return float(progress[start])
    return float("nan")


def permanent_lock(group: pd.DataFrame, *, margin_threshold: float) -> float:
    cond = (
        group["activation_basin_matches_final"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(group["activation_margin"], errors="coerce").ge(margin_threshold).to_numpy()
    )
    progress = pd.to_numeric(group["progress"], errors="coerce").to_numpy(dtype=float)
    for start in range(len(cond)):
        if cond[start:].all():
            return float(progress[start])
    return float("nan")


def switch_count(values: pd.Series) -> int:
    vals = values.dropna().astype(str).to_numpy()
    if len(vals) <= 1:
        return 0
    return int(np.sum(vals[1:] != vals[:-1]))


def build_summary(scores: pd.DataFrame, *, margin_threshold: float, consecutive: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_cols = [col for col in ["decision_id", "prefix_token_index", "progress", "prefix_id"] if col in scores.columns]
    ordered = scores.sort_values(sort_cols).copy()
    for decision_id, group in ordered.groupby("decision_id", sort=False):
        final = group.iloc[-1]
        rows.append(
            {
                "decision_id": decision_id,
                "trajectory_id": final.get("trajectory_id", pd.NA),
                "step_id": final.get("step_id", pd.NA),
                "agent_action": final.get("agent_action", pd.NA),
                "num_prefixes": int(len(group)),
                "final_prefix_progress": float(final.get("progress", np.nan)),
                "final_activation_basin_action": final.get("activation_basin_action", pd.NA),
                "final_activation_basin_matches_final": bool(final.get("activation_basin_matches_final", False)),
                "final_activation_margin": float(final.get("activation_margin", np.nan)),
                "final_act_sim_final": float(final.get("act_sim_final", np.nan)),
                "mean_act_sim_final": float(pd.to_numeric(group["act_sim_final"], errors="coerce").mean()),
                "mean_activation_basin_matches_final": float(group["activation_basin_matches_final"].astype(float).mean()),
                "first_activation_basin_lock_progress": first_consecutive_lock(
                    group, margin_threshold=margin_threshold, consecutive=consecutive
                ),
                "permanent_activation_basin_lock_progress": permanent_lock(
                    group, margin_threshold=margin_threshold
                ),
                "activation_switch_count": switch_count(group["activation_basin_action"]),
            }
        )
    return pd.DataFrame(rows)


def read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional parquet not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"WARNING: failed to read optional parquet {path}: {exc}")
        return None


def merge_behavior_summaries(summary: pd.DataFrame, step_summary_path: Path, permanent_path: Path) -> pd.DataFrame:
    out = summary.copy()
    step = read_optional_parquet(step_summary_path)
    if step is not None and not step.empty:
        step = make_decision_id(step)
        keep = [col for col in ["decision_id", "lock_in_progress", "switch_count"] if col in step.columns]
        if "decision_id" in keep:
            renamed = step[keep].drop_duplicates("decision_id").rename(
                columns={
                    "lock_in_progress": "first_behavior_lock_in_progress",
                    "switch_count": "behavior_switch_count",
                }
            )
            out = out.merge(renamed, on="decision_id", how="left")
    permanent = read_optional_parquet(permanent_path)
    if permanent is not None and not permanent.empty:
        permanent = make_decision_id(permanent)
        keep = [
            col
            for col in [
                "decision_id",
                "first_lock_in_progress_existing",
                "permanent_lock_in_progress",
                "switch_count_total",
            ]
            if col in permanent.columns
        ]
        if "decision_id" in keep:
            renamed = permanent[keep].drop_duplicates("decision_id").rename(
                columns={
                    "first_lock_in_progress_existing": "first_behavior_lock_in_progress_from_permanent_summary",
                    "permanent_lock_in_progress": "permanent_behavior_lock_progress",
                    "switch_count_total": "behavior_switch_count_from_permanent_summary",
                }
            )
            out = out.merge(renamed, on="decision_id", how="left")
    if "first_behavior_lock_in_progress" not in out.columns and "first_behavior_lock_in_progress_from_permanent_summary" in out.columns:
        out["first_behavior_lock_in_progress"] = out["first_behavior_lock_in_progress_from_permanent_summary"]
    if "behavior_switch_count" not in out.columns and "behavior_switch_count_from_permanent_summary" in out.columns:
        out["behavior_switch_count"] = out["behavior_switch_count_from_permanent_summary"]
    if "permanent_behavior_lock_progress" in out.columns:
        out["activation_minus_behavior_permanent_lock_progress"] = (
            pd.to_numeric(out["permanent_activation_basin_lock_progress"], errors="coerce")
            - pd.to_numeric(out["permanent_behavior_lock_progress"], errors="coerce")
        )
    if "first_behavior_lock_in_progress" in out.columns:
        out["activation_minus_behavior_first_lock_progress"] = (
            pd.to_numeric(out["first_activation_basin_lock_progress"], errors="coerce")
            - pd.to_numeric(out["first_behavior_lock_in_progress"], errors="coerce")
        )
    return out


def add_progress_bin(df: pd.DataFrame, bins: np.ndarray = PROGRESS_BINS) -> pd.DataFrame:
    out = df.copy()
    out["progress_bin"] = pd.cut(
        pd.to_numeric(out["progress"], errors="coerce"),
        bins=bins,
        include_lowest=True,
        labels=[f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)],
    )
    return out


def plot_similarity_over_progress(scores: pd.DataFrame, path: Path) -> None:
    work = add_progress_bin(scores)
    summary = work.groupby("progress_bin", observed=False).agg(mean_sim=("act_sim_final", "mean"), sem_sim=("act_sim_final", "sem")).reset_index()
    x = np.arange(len(summary))
    y = summary["mean_sim"].to_numpy(dtype=float)
    sem = summary["sem_sim"].fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.fill_between(x, y - sem, y + sem, alpha=0.2)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(summary["progress_bin"].astype(str).to_numpy()[::2], rotation=35, ha="right")
    ax.set_xlabel("Progress bin")
    ax.set_ylabel("Mean similarity to final-action centroid")
    ax.set_title("Activation similarity to final centroid over progress")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_over_progress(scores: pd.DataFrame, path: Path, *, strict_pre_action: bool = False) -> None:
    work = scores.copy()
    if strict_pre_action and "is_before_first_action_mention" in work.columns:
        work = work[work["is_before_first_action_mention"].fillna(True).astype(bool)]
    work = add_progress_bin(work)
    summary = work.groupby("progress_bin", observed=False).agg(acc=("activation_basin_matches_final", "mean"), n=("activation_basin_matches_final", "size")).reset_index()
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, summary["acc"].to_numpy(dtype=float), marker="o", linewidth=1.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(summary["progress_bin"].astype(str).to_numpy()[::2], rotation=35, ha="right")
    ax.set_xlabel("Progress bin")
    ax.set_ylabel("Fraction basin == final action")
    title = "Strict pre-action activation basin accuracy" if strict_pre_action else "Activation basin accuracy over progress"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_over_progress(scores: pd.DataFrame, path: Path) -> None:
    work = add_progress_bin(scores)
    counts = work.groupby(["progress_bin", "activation_basin_action"], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(columns=ACTIONS, fill_value=0)
    frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    x = np.arange(len(frac))
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(frac))
    for action in ACTIONS:
        values = frac[action].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=ACTION_COLORS[action], label=action, width=0.9)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(frac.index.astype(str).to_numpy()[::2], rotation=35, ha="right")
    ax.set_xlabel("Progress bin")
    ax.set_ylabel("Activation basin action fraction")
    ax.set_title("Activation basin distribution over progress")
    ax.legend(title="basin")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_vs_activation_similarity(scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores["commitment_to_final"], scores["act_sim_final"], s=10, alpha=0.35, linewidths=0)
    ax.set_xlabel("Behavior commitment_to_final")
    ax.set_ylabel("Activation similarity to final centroid")
    ax.set_title("Behavior commitment vs activation centroid similarity")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lockin_comparison(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if "permanent_behavior_lock_progress" not in summary.columns:
        ax.text(0.5, 0.5, "Behavior permanent lock-in unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        x = pd.to_numeric(summary["permanent_behavior_lock_progress"], errors="coerce")
        y = pd.to_numeric(summary["permanent_activation_basin_lock_progress"], errors="coerce")
        ax.scatter(x, y, s=28, alpha=0.7, linewidths=0)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.65)
    ax.set_xlabel("Permanent behavior lock-in progress")
    ax.set_ylabel("Permanent activation basin lock-in progress")
    ax.set_title("Behavior vs activation permanent lock-in")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_examples(scores: pd.DataFrame, max_examples: int) -> list[str]:
    switches = scores.groupby("decision_id")["activation_basin_action"].apply(switch_count).sort_values(ascending=False)
    selected: list[str] = []
    for decision_id in switches.index:
        if decision_id not in selected:
            selected.append(str(decision_id))
        if len(selected) >= max_examples:
            return selected
    return selected


def plot_example_trajectories(scores: pd.DataFrame, path: Path, max_examples: int) -> None:
    examples = choose_examples(scores, max_examples)
    n = max(1, min(len(examples), max_examples))
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(3.2, 3.0 * nrows)), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, decision_id in zip(axes.flat, examples):
        ax.set_visible(True)
        group = scores[scores["decision_id"].astype(str).eq(decision_id)].sort_values(["prefix_token_index", "progress"])
        x = group["prefix_token_index"] if "prefix_token_index" in group.columns else group["progress"]
        for action in ACTIONS:
            ax.plot(x, group[f"act_sim_{action}"], label=action, color=ACTION_COLORS[action], linewidth=1.5)
        final_action = str(group["agent_action"].iloc[-1]) if "agent_action" in group.columns else "NA"
        ax.set_title(f"{decision_id} final={final_action}", fontsize=8)
        ax.set_xlabel("prefix_token_index")
        ax.set_ylabel("cosine similarity")
        ax.grid(True, alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    centroid_metadata: dict[str, Any],
    layer: int,
    layer_source: str,
    representation: str,
    feature_path: Path,
    margin_threshold: float,
    consecutive: int,
    figures: list[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Activation Centroid Basin Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Prefix rows: {len(scores):,}")
    lines.append(f"- Decisions: {summary['decision_id'].nunique() if 'decision_id' in summary.columns else 'NA'}")
    lines.append(f"- Trajectories: {scores['trajectory_id'].nunique() if 'trajectory_id' in scores.columns else 'NA'}")
    lines.append(f"- Selected layer: {layer}")
    lines.append(f"- Selected representation: `{representation}`")
    lines.append(f"- Layer source: {layer_source}")
    lines.append(f"- Feature file: `{feature_path}`")
    lines.append(f"- Centroid source: `{centroid_metadata['source']}`")
    lines.append(f"- Centroid commitment threshold: {centroid_metadata['centroid_commitment_threshold']}")
    lines.append(f"- Centroid min progress: {centroid_metadata['centroid_min_progress']}")
    lines.append(f"- Lock margin threshold: {margin_threshold}")
    lines.append(f"- Consecutive prefixes for first lock: {consecutive}")
    lines.append("")

    lines.append("## Centroid Rows\n")
    counts = pd.Series(centroid_metadata["counts_by_action"], name="rows")
    lines.append(counts.to_markdown())
    lines.append("")

    lines.append("## Prefix-Level Basin Metrics\n")
    lines.append(f"- Mean similarity to final centroid: {fmt(scores['act_sim_final'].mean())}")
    lines.append(f"- Mean activation margin: {fmt(scores['activation_margin'].mean())}")
    lines.append(f"- Basin matches final action: {fmt(scores['activation_basin_matches_final'].mean())}")
    if "activation_basin_matches_behavior_top" in scores.columns:
        lines.append(f"- Basin matches behavior top_action: {fmt(scores['activation_basin_matches_behavior_top'].astype(float).mean())}")
    if "is_before_first_action_mention" in scores.columns:
        strict = scores[scores["is_before_first_action_mention"].fillna(True).astype(bool)]
        lines.append(f"- Strict pre-action basin matches final action: {fmt(strict['activation_basin_matches_final'].mean())}")
    lines.append("")

    lines.append("## Decision-Level Lock-In Metrics\n")
    for col in [
        "first_activation_basin_lock_progress",
        "permanent_activation_basin_lock_progress",
        "activation_switch_count",
        "first_behavior_lock_in_progress",
        "permanent_behavior_lock_progress",
        "behavior_switch_count",
        "activation_minus_behavior_permanent_lock_progress",
    ]:
        if col in summary.columns:
            lines.append(f"- Median `{col}`: {fmt(pd.to_numeric(summary[col], errors='coerce').median())}")
    lines.append("")

    lines.append("## Basin Action Counts\n")
    lines.append(scores["activation_basin_action"].value_counts(dropna=False).sort_index().to_markdown())
    lines.append("")

    lines.append("## Caveats\n")
    lines.append("- Centroid basins are descriptive nearest-centroid geometry, not a supervised decoder.")
    lines.append("- Cosine similarities depend on the selected layer, representation, and centroid source.")
    lines.append("- `post_reasoning` centroids require post-reasoning feature rows; the current prefix feature table may not contain them.")
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
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--activation_manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--step_summary", default="outputs/step_commitment_summary.parquet")
    parser.add_argument("--permanent_lockin_summary", default="outputs/followups/permanent_lockin_summary.parquet")
    parser.add_argument("--output_scores", default="outputs/activation/activation_centroid_scores.parquet")
    parser.add_argument("--output_summary", default="outputs/activation/activation_basin_summary.parquet")
    parser.add_argument("--report", default="outputs/activation/reports/activation_centroid_basin_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/centroids")
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--centroid_source",
        choices=["high_commitment_prefixes", "final_prefixes", "post_reasoning"],
        default="high_commitment_prefixes",
    )
    parser.add_argument("--centroid_commitment_threshold", type=float, default=0.9)
    parser.add_argument("--centroid_min_progress", type=float, default=None)
    parser.add_argument("--activation_commitment_mode", choices=["similarity", "softmax"], default="similarity")
    parser.add_argument("--softmax_temperature", type=float, default=0.1)
    parser.add_argument("--lock_margin_threshold", type=float, default=0.005)
    parser.add_argument("--consecutive_prefixes", type=int, default=3)
    parser.add_argument("--num_examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    repo_root = infer_repo_root(metadata_path)
    metadata_path = resolve_path(metadata_path, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root) if args.config else None
    activation_manifold_path = resolve_path(args.activation_manifold, repo_root) if args.activation_manifold else None
    step_summary_path = resolve_path(args.step_summary, repo_root)
    permanent_path = resolve_path(args.permanent_lockin_summary, repo_root)
    output_scores_path = resolve_path(args.output_scores, repo_root)
    output_summary_path = resolve_path(args.output_summary, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    meta = make_decision_id(pd.read_parquet(metadata_path))
    if "decision_id" not in meta.columns:
        raise SystemExit("Metadata must contain decision_id or trajectory_id + step_id")
    if "agent_action" not in meta.columns:
        raise SystemExit("Metadata must contain agent_action")

    layer, layer_source = select_layer(
        requested_layer=args.layer,
        config_path=config_path,
        activation_manifold_path=activation_manifold_path,
        features_dir=features_dir,
        representation=args.representation,
    )
    selected_feature_path = feature_file(features_dir, layer, args.representation)
    X = np.load(selected_feature_path, mmap_mode="r")
    if X.shape[0] != len(meta):
        raise SystemExit(f"Feature row mismatch: {X.shape[0]} feature rows but {len(meta)} metadata rows")

    print(f"Building centroids from layer {layer}, representation {args.representation}...")
    X_float = np.asarray(X, dtype=np.float32)
    centroids, centroid_metadata, source_mask = build_centroids(
        X_float,
        meta,
        source=args.centroid_source,
        threshold=args.centroid_commitment_threshold,
        min_progress=args.centroid_min_progress,
    )
    meta = meta.copy()
    meta["used_for_activation_centroid"] = source_mask
    scores = compute_scores(
        X_float,
        meta,
        centroids,
        commitment_mode=args.activation_commitment_mode,
        softmax_temperature=args.softmax_temperature,
    )
    scores["layer"] = int(layer)
    scores["representation"] = args.representation
    scores["centroid_source"] = args.centroid_source
    scores["centroid_commitment_threshold"] = args.centroid_commitment_threshold
    scores["centroid_min_progress"] = args.centroid_min_progress
    scores["lock_margin_threshold"] = args.lock_margin_threshold

    summary = build_summary(scores, margin_threshold=args.lock_margin_threshold, consecutive=args.consecutive_prefixes)
    summary = merge_behavior_summaries(summary, step_summary_path, permanent_path)
    summary["layer"] = int(layer)
    summary["representation"] = args.representation
    summary["centroid_source"] = args.centroid_source
    summary["lock_margin_threshold"] = args.lock_margin_threshold

    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_scores_path, index=False)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_summary_path, index=False)

    figures: list[Path] = []
    fig = figures_dir / "activation_similarity_to_final_centroid_over_progress.png"
    plot_similarity_over_progress(scores, fig)
    figures.append(fig)
    fig = figures_dir / "activation_basin_accuracy_over_progress.png"
    plot_accuracy_over_progress(scores, fig)
    figures.append(fig)
    fig = figures_dir / "activation_basin_distribution_over_progress.png"
    plot_distribution_over_progress(scores, fig)
    figures.append(fig)
    fig = figures_dir / "behavior_commitment_vs_activation_similarity.png"
    plot_behavior_vs_activation_similarity(scores, fig)
    figures.append(fig)
    fig = figures_dir / "behavior_vs_activation_permanent_lockin.png"
    plot_lockin_comparison(summary, fig)
    figures.append(fig)
    fig = figures_dir / "example_activation_basin_trajectories.png"
    plot_example_trajectories(scores, fig, max_examples=args.num_examples)
    figures.append(fig)
    fig = figures_dir / "strict_pre_action_activation_basin_accuracy.png"
    plot_accuracy_over_progress(scores, fig, strict_pre_action=True)
    figures.append(fig)

    write_report(
        report_path,
        scores=scores,
        summary=summary,
        centroid_metadata=centroid_metadata,
        layer=layer,
        layer_source=layer_source,
        representation=args.representation,
        feature_path=selected_feature_path,
        margin_threshold=args.lock_margin_threshold,
        consecutive=args.consecutive_prefixes,
        figures=figures,
    )

    print(f"Selected layer {layer} ({layer_source}), representation {args.representation}")
    print(f"Centroid counts: {centroid_metadata['counts_by_action']}")
    print(f"Wrote {output_scores_path}")
    print(f"Wrote {output_summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(figures)} figures under {figures_dir}")


if __name__ == "__main__":
    main()
