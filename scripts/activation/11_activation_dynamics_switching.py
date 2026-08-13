#!/usr/bin/env python3
"""Activation trajectory dynamics and switching analysis for gridworld commitment."""

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
Q_COLS = [f"q_{action}" for action in ACTIONS]
ACT_SIM_COLS = [f"act_sim_{action}" for action in ACTIONS]
PROGRESS_BINS = np.linspace(0.0, 1.0, 21)


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


def read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional parquet not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"WARNING: failed to read optional parquet {path}: {exc}")
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


def unique_nonnull_value(df: pd.DataFrame | None, column: str) -> Any | None:
    if df is None or column not in df.columns:
        return None
    values = pd.Series(df[column]).dropna().unique().tolist()
    return values[0] if len(values) == 1 else None


def select_layer_and_representation(
    *,
    requested_layer: int | None,
    requested_representation: str | None,
    features_dir: Path,
    centroid_scores: pd.DataFrame,
    manifold: pd.DataFrame,
) -> tuple[int, str, str]:
    representation = requested_representation
    source_bits: list[str] = []
    if representation is None:
        rep = unique_nonnull_value(centroid_scores, "representation")
        if rep is not None:
            representation = str(rep)
            source_bits.append("centroid_scores representation")
        else:
            rep = unique_nonnull_value(manifold, "representation")
            if rep is not None:
                representation = str(rep)
                source_bits.append("activation_manifold representation")
    if representation is None:
        representation = "delta_pre"
        source_bits.append("default representation")

    available_layers = discover_layers(features_dir, representation)
    if not available_layers:
        raise SystemExit(f"No feature arrays found for representation {representation!r} in {features_dir}")

    layer = requested_layer
    if layer is not None:
        if layer not in available_layers:
            raise SystemExit(f"Requested layer {layer} unavailable for {representation}. Available: {available_layers}")
        source_bits.append("explicit --layer")
    else:
        value = unique_nonnull_value(centroid_scores, "layer")
        if value is not None and int(value) in available_layers:
            layer = int(value)
            source_bits.append("centroid_scores layer")
        else:
            value = unique_nonnull_value(manifold, "layer")
            if value is not None and int(value) in available_layers:
                layer = int(value)
                source_bits.append("activation_manifold layer")
        if layer is None:
            index = int(round(0.75 * (len(available_layers) - 1)))
            layer = available_layers[index]
            source_bits.append("mid-late fallback layer")
    return int(layer), representation, "; ".join(source_bits)


def merge_manifold(base: pd.DataFrame, manifold: pd.DataFrame) -> pd.DataFrame:
    if manifold.empty:
        return base
    manifold = make_decision_id(manifold)
    out = base.copy()
    join_cols = ["prefix_id"] if "prefix_id" in out.columns and "prefix_id" in manifold.columns else ["decision_id", "prefix_token_index"]
    add_cols = [col for col in [*join_cols, "act_pca1", "act_pca2", "layer", "representation"] if col in manifold.columns]
    if not set(join_cols).issubset(add_cols):
        return out
    small = manifold[add_cols].drop_duplicates(join_cols).rename(
        columns={"layer": "manifold_layer", "representation": "manifold_representation"}
    )
    before = len(out)
    out = out.merge(small, on=join_cols, how="left", suffixes=("", "_from_manifold"))
    if len(out) != before:
        raise RuntimeError("Merging activation manifold changed row count")
    for col in ["act_pca1", "act_pca2"]:
        alt = f"{col}_from_manifold"
        if alt in out.columns:
            out[col] = out[col].combine_first(out[alt]) if col in out.columns else out[alt]
            out = out.drop(columns=[alt])
    return out


def merge_missing_metadata(base: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = make_decision_id(metadata)
    out = base.copy()
    missing_cols = [col for col in metadata.columns if col not in out.columns]
    if not missing_cols:
        return out
    join_cols = ["prefix_id"] if "prefix_id" in out.columns and "prefix_id" in metadata.columns else ["decision_id", "prefix_token_index"]
    add_cols = [col for col in [*join_cols, *missing_cols] if col in metadata.columns]
    if not set(join_cols).issubset(add_cols):
        return out
    small = metadata[add_cols].drop_duplicates(join_cols)
    before = len(out)
    out = out.merge(small, on=join_cols, how="left")
    if len(out) != before:
        raise RuntimeError("Merging activation metadata changed row count")
    return out


def resolve_feature_path(df: pd.DataFrame, features_dir: Path, repo_root: Path, layer: int, representation: str) -> Path:
    direct = feature_file(features_dir, layer, representation)
    if direct.exists():
        return direct
    col = f"feature_file_layer_{layer:02d}_{representation}"
    if col in df.columns:
        for value in pd.Series(df[col]).dropna().astype(str).unique().tolist():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.exists():
                return candidate
    raise SystemExit(f"Feature array not found for layer {layer}, representation {representation}: {direct}")


def load_feature_rows(df: pd.DataFrame, feature_path: Path) -> np.ndarray:
    matrix = np.load(feature_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise SystemExit(f"Expected a 2D feature matrix, got shape {matrix.shape}: {feature_path}")
    if "row_index" in df.columns:
        rows = pd.to_numeric(df["row_index"], errors="coerce")
        if rows.notna().all():
            row_idx = rows.astype(int).to_numpy()
            if row_idx.min(initial=0) >= 0 and row_idx.max(initial=-1) < matrix.shape[0]:
                return np.asarray(matrix[row_idx], dtype=np.float32)
    if len(df) != matrix.shape[0]:
        raise SystemExit(
            f"Cannot align feature rows: table has {len(df)} rows but feature matrix has {matrix.shape[0]} rows, "
            "and row_index is unavailable/invalid."
        )
    return np.asarray(matrix[:], dtype=np.float32)


def switch_count(values: pd.Series) -> int:
    vals = values.dropna().astype(str).to_numpy()
    if len(vals) <= 1:
        return 0
    return int(np.sum(vals[1:] != vals[:-1]))


def l1_total_variation(group: pd.DataFrame, cols: list[str]) -> float:
    if not set(cols).issubset(group.columns) or len(group) <= 1:
        return float("nan")
    q = group[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(q).all():
        return float("nan")
    return float(np.abs(np.diff(q, axis=0)).sum(axis=1).sum())


def cosine_distance_consecutive(vecs: np.ndarray) -> np.ndarray:
    out = np.full(len(vecs), np.nan, dtype=float)
    if len(vecs) <= 1:
        return out
    prev = vecs[:-1]
    curr = vecs[1:]
    numer = np.einsum("ij,ij->i", prev, curr)
    denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1)
    cos = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 1e-12)
    out[1:] = 1.0 - np.clip(cos, -1.0, 1.0)
    return out


def curvature_angles(vecs: np.ndarray) -> np.ndarray:
    out = np.full(len(vecs), np.nan, dtype=float)
    if len(vecs) <= 2:
        return out
    seg = np.diff(vecs, axis=0)
    prev = seg[:-1]
    nxt = seg[1:]
    numer = np.einsum("ij,ij->i", prev, nxt)
    denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
    cos = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 1e-12)
    out[1:-1] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def first_consecutive_lock(group: pd.DataFrame, *, margin_threshold: float, consecutive: int) -> float:
    if not {"activation_basin_matches_final", "activation_margin", "progress"}.issubset(group.columns):
        return float("nan")
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


def first_consecutive_lock_index(group: pd.DataFrame, *, margin_threshold: float, consecutive: int) -> int | None:
    if not {"activation_basin_matches_final", "activation_margin"}.issubset(group.columns):
        return None
    cond = (
        group["activation_basin_matches_final"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(group["activation_margin"], errors="coerce").ge(margin_threshold).to_numpy()
    )
    if len(cond) < consecutive:
        return None
    for start in range(0, len(cond) - consecutive + 1):
        if cond[start : start + consecutive].all():
            return start
    return None


def permanent_lock(group: pd.DataFrame, *, margin_threshold: float) -> float:
    if not {"activation_basin_matches_final", "activation_margin", "progress"}.issubset(group.columns):
        return float("nan")
    cond = (
        group["activation_basin_matches_final"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(group["activation_margin"], errors="coerce").ge(margin_threshold).to_numpy()
    )
    progress = pd.to_numeric(group["progress"], errors="coerce").to_numpy(dtype=float)
    for start in range(len(cond)):
        if cond[start:].all():
            return float(progress[start])
    return float("nan")


def permanent_lock_index(group: pd.DataFrame, *, margin_threshold: float) -> int | None:
    if not {"activation_basin_matches_final", "activation_margin"}.issubset(group.columns):
        return None
    cond = (
        group["activation_basin_matches_final"].fillna(False).to_numpy(dtype=bool)
        & pd.to_numeric(group["activation_margin"], errors="coerce").ge(margin_threshold).to_numpy()
    )
    for start in range(len(cond)):
        if cond[start:].all():
            return start
    return None


def count_switches_after(values: pd.Series, start_index: int | None) -> float:
    if start_index is None:
        return float("nan")
    vals = values.astype(str).to_numpy()[start_index:]
    if len(vals) <= 1:
        return 0.0
    return float(np.sum(vals[1:] != vals[:-1]))


def build_prefix_and_summary(
    base: pd.DataFrame,
    X: np.ndarray,
    *,
    layer: int,
    representation: str,
    margin_threshold: float,
    consecutive_prefixes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_cols = [col for col in ["decision_id", "prefix_token_index", "progress", "prefix_id"] if col in base.columns]
    ordered = base.sort_values(sort_cols).reset_index(drop=True).copy()
    X = np.asarray(X, dtype=np.float32)
    if len(ordered) != X.shape[0]:
        raise RuntimeError(f"Feature matrix row count mismatch: {X.shape[0]} vs table {len(ordered)}")

    for col in ["activation_step_distance", "activation_cosine_step_distance", "activation_curvature"]:
        ordered[col] = np.nan
    ordered["activation_cumulative_path_length"] = np.nan
    ordered["activation_manifold_step_distance"] = np.nan
    ordered["activation_manifold_cumulative_path_length"] = np.nan

    if {"act_pca1", "act_pca2"}.issubset(ordered.columns):
        pca = ordered[["act_pca1", "act_pca2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        center = np.nanmean(pca, axis=0)
        ordered["activation_radius"] = np.sqrt(np.nansum((pca - center) ** 2, axis=1))
    else:
        pca = None
        center = np.array([np.nan, np.nan])
        ordered["activation_radius"] = np.nan
    ordered["activation_radius_delta"] = np.nan

    summary_rows: list[dict[str, Any]] = []
    for decision_id, group in ordered.groupby("decision_id", sort=False):
        idx = group.index.to_numpy()
        vecs = X[idx]
        diffs = np.diff(vecs, axis=0) if len(vecs) > 1 else np.empty((0, vecs.shape[1]), dtype=np.float32)
        step_distance = np.full(len(group), np.nan, dtype=float)
        if len(group) > 1:
            step_distance[1:] = np.linalg.norm(diffs, axis=1)
        cosine_step = cosine_distance_consecutive(vecs)
        curvature = curvature_angles(vecs)
        cumulative = np.nancumsum(np.nan_to_num(step_distance, nan=0.0))

        ordered.loc[idx, "activation_step_distance"] = step_distance
        ordered.loc[idx, "activation_cosine_step_distance"] = cosine_step
        ordered.loc[idx, "activation_curvature"] = curvature
        ordered.loc[idx, "activation_cumulative_path_length"] = cumulative

        manifold_path = float("nan")
        if pca is not None:
            p = pca[idx]
            pca_step = np.full(len(group), np.nan, dtype=float)
            if len(group) > 1 and np.isfinite(p).all():
                pca_step[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1)
                manifold_path = float(np.nansum(pca_step))
            ordered.loc[idx, "activation_manifold_step_distance"] = pca_step
            ordered.loc[idx, "activation_manifold_cumulative_path_length"] = np.nancumsum(
                np.nan_to_num(pca_step, nan=0.0)
            )
        radius = pd.to_numeric(ordered.loc[idx, "activation_radius"], errors="coerce").to_numpy(dtype=float)
        radius_delta = np.full(len(group), np.nan, dtype=float)
        if len(group) > 1:
            radius_delta[1:] = np.diff(radius)
        ordered.loc[idx, "activation_radius_delta"] = radius_delta

        first_idx = first_consecutive_lock_index(group, margin_threshold=margin_threshold, consecutive=consecutive_prefixes)
        permanent_idx = permanent_lock_index(group, margin_threshold=margin_threshold)
        activation_switches = switch_count(group.get("activation_basin_action", pd.Series(dtype=object)))
        behavior_switches_computed = switch_count(group.get("top_action", pd.Series(dtype=object)))
        behavior_tv_computed = l1_total_variation(group, Q_COLS)
        final = group.iloc[-1]
        summary_rows.append(
            {
                "decision_id": decision_id,
                "trajectory_id": final.get("trajectory_id", pd.NA),
                "step_id": final.get("step_id", pd.NA),
                "agent_action": final.get("agent_action", pd.NA),
                "num_prefixes": int(len(group)),
                "final_prefix_progress": float(pd.to_numeric(pd.Series([final.get("progress", np.nan)]), errors="coerce").iloc[0]),
                "layer": layer,
                "representation": representation,
                "activation_total_variation": float(np.nansum(step_distance)),
                "activation_mean_step_distance": float(np.nanmean(step_distance)) if np.isfinite(step_distance).any() else float("nan"),
                "activation_median_step_distance": float(np.nanmedian(step_distance)) if np.isfinite(step_distance).any() else float("nan"),
                "activation_mean_cosine_step_distance": float(np.nanmean(cosine_step)) if np.isfinite(cosine_step).any() else float("nan"),
                "activation_manifold_path_length": manifold_path,
                "activation_mean_radius": float(np.nanmean(radius)) if np.isfinite(radius).any() else float("nan"),
                "activation_final_radius": float(radius[-1]) if len(radius) and np.isfinite(radius[-1]) else float("nan"),
                "activation_radius_change": float(radius[-1] - radius[0]) if len(radius) > 1 and np.isfinite(radius[[0, -1]]).all() else float("nan"),
                "activation_mean_curvature": float(np.nanmean(curvature)) if np.isfinite(curvature).any() else float("nan"),
                "activation_basin_switch_count": activation_switches,
                "first_activation_basin_lock_progress": first_consecutive_lock(
                    group, margin_threshold=margin_threshold, consecutive=consecutive_prefixes
                ),
                "permanent_activation_basin_lock_progress": permanent_lock(group, margin_threshold=margin_threshold),
                "switch_count_after_first_activation_lock": count_switches_after(
                    group.get("activation_basin_action", pd.Series(dtype=object)), first_idx
                ),
                "switch_count_after_permanent_activation_lock": count_switches_after(
                    group.get("activation_basin_action", pd.Series(dtype=object)), permanent_idx
                ),
                "behavior_switch_count_computed": behavior_switches_computed,
                "behavior_total_variation_computed": behavior_tv_computed,
                "final_commitment_to_final": float(final.get("commitment_to_final", np.nan)),
                "mean_commitment_to_final": float(pd.to_numeric(group.get("commitment_to_final", pd.Series(dtype=float)), errors="coerce").mean()),
                "final_entropy_commitment": float(final.get("entropy_commitment", np.nan)),
                "mean_entropy_commitment": float(pd.to_numeric(group.get("entropy_commitment", pd.Series(dtype=float)), errors="coerce").mean()),
                "final_activation_basin_action": final.get("activation_basin_action", pd.NA),
                "final_activation_basin_matches_final": final.get("activation_basin_matches_final", pd.NA),
                "final_act_sim_final": float(final.get("act_sim_final", np.nan)),
            }
        )

    ordered["layer"] = layer
    ordered["representation"] = representation
    return ordered, pd.DataFrame(summary_rows)


def merge_behavior_summaries(summary: pd.DataFrame, step_summary_path: Path, permanent_path: Path) -> pd.DataFrame:
    out = summary.copy()
    step = read_optional_parquet(step_summary_path)
    if step is not None and not step.empty:
        step = make_decision_id(step)
        keep_cols = [
            col
            for col in [
                "decision_id",
                "source_file",
                "lock_in_progress",
                "switch_count",
                "total_variation",
                "final_prefix_commitment_to_final",
                "final_prefix_top_action",
                "final_prefix_is_top_action_final",
            ]
            if col in step.columns
        ]
        if "decision_id" in keep_cols:
            renamed = step[keep_cols].drop_duplicates("decision_id").rename(
                columns={
                    "lock_in_progress": "first_behavior_lock_in_progress",
                    "switch_count": "behavior_switch_count_from_step_summary",
                    "total_variation": "behavior_total_variation_from_step_summary",
                }
            )
            out = out.merge(renamed, on="decision_id", how="left")

    permanent = read_optional_parquet(permanent_path)
    if permanent is not None and not permanent.empty:
        permanent = make_decision_id(permanent)
        keep_cols = [
            col
            for col in [
                "decision_id",
                "first_lock_in_progress_existing",
                "permanent_lock_in_progress",
                "permanent_top_action_lock_progress",
                "switch_count_total",
                "switch_count_after_first_lock",
                "switch_count_after_permanent_lock",
            ]
            if col in permanent.columns
        ]
        if "decision_id" in keep_cols:
            renamed = permanent[keep_cols].drop_duplicates("decision_id").rename(
                columns={
                    "first_lock_in_progress_existing": "first_behavior_lock_in_progress_from_permanent_summary",
                    "permanent_lock_in_progress": "permanent_behavior_lock_progress",
                    "permanent_top_action_lock_progress": "permanent_behavior_top_action_lock_progress",
                    "switch_count_total": "behavior_switch_count_from_permanent_summary",
                    "switch_count_after_first_lock": "behavior_switch_count_after_first_lock",
                    "switch_count_after_permanent_lock": "behavior_switch_count_after_permanent_lock",
                }
            )
            out = out.merge(renamed, on="decision_id", how="left")

    if "behavior_switch_count_from_step_summary" in out.columns:
        out["behavior_switch_count"] = out["behavior_switch_count_from_step_summary"]
    elif "behavior_switch_count_from_permanent_summary" in out.columns:
        out["behavior_switch_count"] = out["behavior_switch_count_from_permanent_summary"]
    else:
        out["behavior_switch_count"] = out["behavior_switch_count_computed"]

    if "behavior_total_variation_from_step_summary" in out.columns:
        out["behavior_total_variation"] = out["behavior_total_variation_from_step_summary"]
    else:
        out["behavior_total_variation"] = out["behavior_total_variation_computed"]

    if "first_behavior_lock_in_progress" not in out.columns:
        out["first_behavior_lock_in_progress"] = np.nan
    if "first_behavior_lock_in_progress_from_permanent_summary" in out.columns:
        out["first_behavior_lock_in_progress"] = out["first_behavior_lock_in_progress"].combine_first(
            out["first_behavior_lock_in_progress_from_permanent_summary"]
        )
    if "permanent_behavior_lock_progress" not in out.columns:
        out["permanent_behavior_lock_progress"] = np.nan

    out["activation_minus_behavior_first_lock_progress"] = (
        pd.to_numeric(out["first_activation_basin_lock_progress"], errors="coerce")
        - pd.to_numeric(out["first_behavior_lock_in_progress"], errors="coerce")
    )
    out["activation_minus_behavior_permanent_lock_progress"] = (
        pd.to_numeric(out["permanent_activation_basin_lock_progress"], errors="coerce")
        - pd.to_numeric(out["permanent_behavior_lock_progress"], errors="coerce")
    )
    return out


def corr_pair(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[int, float, float]:
    if x_col not in df.columns or y_col not in df.columns:
        return 0, float("nan"), float("nan")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return int(mask.sum()), float("nan"), float("nan")
    pearson = float(x[mask].corr(y[mask], method="pearson"))
    if spearmanr is not None:
        spearman = float(spearmanr(x[mask], y[mask], nan_policy="omit").correlation)
    else:
        spearman = float(x[mask].corr(y[mask], method="spearman"))
    return int(mask.sum()), pearson, spearman


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_plot(summary: pd.DataFrame, x_col: str, y_col: str, path: Path, *, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if x_col not in summary.columns or y_col not in summary.columns:
        plot_empty(ax, "Required columns unavailable")
    else:
        x = pd.to_numeric(summary[x_col], errors="coerce")
        y = pd.to_numeric(summary[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) == 0:
            plot_empty(ax, "No paired values")
        else:
            colors = [ACTION_COLORS.get(str(a), "#666666") for a in summary.loc[mask, "agent_action"]]
            ax.scatter(x[mask], y[mask], s=36, alpha=0.78, c=colors, edgecolor="white", linewidth=0.4)
            min_v = float(np.nanmin([x[mask].min(), y[mask].min()]))
            max_v = float(np.nanmax([x[mask].max(), y[mask].max()]))
            if np.isfinite(min_v) and np.isfinite(max_v) and min_v != max_v:
                ax.plot([min_v, max_v], [min_v, max_v], color="#888888", linewidth=1, linestyle="--")
            n, pearson, spearman = corr_pair(summary.loc[mask], x_col, y_col)
            ax.text(
                0.02,
                0.98,
                f"n={n}\nPearson={pearson:.3f}\nSpearman={spearman:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.18)
    save_figure(fig, path)


def plot_radius_over_progress(prefix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if "activation_radius" not in prefix.columns:
        plot_empty(ax, "activation_radius unavailable")
    else:
        tmp = prefix.copy()
        tmp["progress_bin"] = pd.cut(pd.to_numeric(tmp["progress"], errors="coerce"), PROGRESS_BINS, include_lowest=True)
        plotted = False
        for action in ACTIONS:
            sub = tmp[tmp["agent_action"].astype(str).eq(action)]
            if sub.empty:
                continue
            grouped = sub.groupby("progress_bin", observed=True)["activation_radius"]
            mean = grouped.mean()
            count = grouped.count()
            sem = grouped.sem().fillna(0.0)
            mids = np.array([interval.mid for interval in mean.index], dtype=float)
            if len(mids) == 0:
                continue
            plotted = True
            ax.plot(mids, mean.to_numpy(dtype=float), marker="o", linewidth=1.8, label=f"{action} (n={int(count.sum())})", color=ACTION_COLORS[action])
            ax.fill_between(mids, mean.to_numpy(dtype=float) - sem.to_numpy(dtype=float), mean.to_numpy(dtype=float) + sem.to_numpy(dtype=float), color=ACTION_COLORS[action], alpha=0.15, linewidth=0)
        if not plotted:
            plot_empty(ax, "No radius data")
        else:
            ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("reasoning progress")
    ax.set_ylabel("activation PCA radius")
    ax.set_title("Activation radius over progress by final action")
    ax.grid(alpha=0.18)
    save_figure(fig, path)


def plot_path_length_by_action(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    data = []
    labels = []
    for action in ACTIONS:
        vals = pd.to_numeric(summary.loc[summary["agent_action"].astype(str).eq(action), "activation_manifold_path_length"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals):
            data.append(vals)
            labels.append(action)
    if not data:
        plot_empty(ax, "No path length data")
    else:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, action in zip(bp["boxes"], labels):
            patch.set_facecolor(ACTION_COLORS.get(action, "#777777"))
            patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, (vals, action) in enumerate(zip(data, labels), start=1):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.45, color=ACTION_COLORS.get(action, "#777777"), edgecolor="none")
    ax.set_xlabel("final action")
    ax.set_ylabel("activation PCA path length")
    ax.set_title("Activation manifold path length by final action")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, path)


def plot_example_high_switch(prefix: pd.DataFrame, summary: pd.DataFrame, path: Path, *, max_examples: int) -> list[str]:
    candidates = summary.sort_values(["activation_basin_switch_count", "activation_total_variation"], ascending=False).head(max_examples)
    example_ids = candidates["decision_id"].astype(str).tolist()
    n = max(1, len(example_ids))
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    if not example_ids or not {"act_pca1", "act_pca2"}.issubset(prefix.columns):
        ax = axes.ravel()[0]
        ax.set_visible(True)
        plot_empty(ax, "No PCA trajectory examples")
    else:
        for ax, decision_id in zip(axes.ravel(), example_ids):
            ax.set_visible(True)
            g = prefix[prefix["decision_id"].astype(str).eq(decision_id)].sort_values("prefix_token_index")
            action = str(g["agent_action"].iloc[-1]) if len(g) else ""
            color = ACTION_COLORS.get(action, "#666666")
            x = pd.to_numeric(g["act_pca1"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(g["act_pca2"], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=1.4, alpha=0.85)
            ax.scatter(x, y, c=pd.to_numeric(g["progress"], errors="coerce"), cmap="viridis", s=16, alpha=0.75)
            if len(x):
                ax.scatter([x[0]], [y[0]], marker="o", s=70, color="white", edgecolor=color, linewidth=1.6, label="start")
                ax.scatter([x[-1]], [y[-1]], marker="s", s=70, color=color, edgecolor="black", linewidth=0.5, label="end")
            switches = int(candidates.loc[candidates["decision_id"].astype(str).eq(decision_id), "activation_basin_switch_count"].iloc[0])
            ax.set_title(f"{decision_id}\n{action}, activation switches={switches}", fontsize=9)
            ax.set_xlabel("act_pca1")
            ax.set_ylabel("act_pca2")
            ax.grid(alpha=0.15)
    fig.suptitle("High-switch activation manifold trajectories", y=1.02)
    save_figure(fig, path)
    return example_ids


def plot_tentative_permanent(prefix: pd.DataFrame, summary: pd.DataFrame, path: Path, *, max_examples: int) -> list[str]:
    mask = (
        pd.to_numeric(summary["activation_basin_switch_count"], errors="coerce").gt(0)
        & pd.to_numeric(summary["permanent_activation_basin_lock_progress"], errors="coerce").notna()
    )
    candidates = summary[mask].sort_values(["activation_basin_switch_count", "permanent_activation_basin_lock_progress"], ascending=[False, True]).head(max_examples)
    example_ids = candidates["decision_id"].astype(str).tolist()
    n = max(1, len(example_ids))
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 3.2 * n), squeeze=False)
    if not example_ids or not set(ACT_SIM_COLS).issubset(prefix.columns):
        plot_empty(axes[0, 0], "No tentative-then-permanent examples")
    else:
        for ax, decision_id in zip(axes.ravel(), example_ids):
            g = prefix[prefix["decision_id"].astype(str).eq(decision_id)].sort_values("prefix_token_index")
            x = pd.to_numeric(g["prefix_token_index"], errors="coerce").to_numpy(dtype=float)
            for action in ACTIONS:
                ax.plot(x, pd.to_numeric(g[f"act_sim_{action}"], errors="coerce"), label=action, color=ACTION_COLORS[action], linewidth=1.3)
            if "commitment_to_final" in g.columns:
                ax.plot(x, pd.to_numeric(g["commitment_to_final"], errors="coerce"), label="behavior q_final", color="#222222", linewidth=1.5, linestyle="--")
            row = candidates[candidates["decision_id"].astype(str).eq(decision_id)].iloc[0]
            lock = row.get("permanent_activation_basin_lock_progress", np.nan)
            if pd.notna(lock) and "progress" in g.columns:
                progress = pd.to_numeric(g["progress"], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(progress).any():
                    nearest = int(np.nanargmin(np.abs(progress - float(lock))))
                    if 0 <= nearest < len(x):
                        ax.axvline(x[nearest], color="#111111", linestyle=":", linewidth=1.2, label="activation permanent lock")
            action = str(g["agent_action"].iloc[-1]) if len(g) else ""
            switches = int(row.get("activation_basin_switch_count", 0))
            ax.set_title(f"{decision_id} ({action}), activation switches={switches}", fontsize=9)
            ax.set_xlabel("prefix token index")
            ax.set_ylabel("centroid similarity / behavior q")
            ax.grid(alpha=0.16)
            ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Tentative-then-permanent activation commitment examples", y=1.01)
    save_figure(fig, path)
    return example_ids


def write_report(
    path: Path,
    *,
    prefix: pd.DataFrame,
    summary: pd.DataFrame,
    layer: int,
    representation: str,
    layer_source: str,
    feature_path: Path,
    margin_threshold: float,
    consecutive_prefixes: int,
    high_switch_examples: list[str],
    tentative_examples: list[str],
) -> None:
    n_switch, switch_r, switch_s = corr_pair(summary, "behavior_switch_count", "activation_basin_switch_count")
    n_lock, lock_r, lock_s = corr_pair(summary, "permanent_behavior_lock_progress", "permanent_activation_basin_lock_progress")
    n_tv, tv_r, tv_s = corr_pair(summary, "behavior_total_variation", "activation_total_variation")

    lock_diff = pd.to_numeric(summary.get("activation_minus_behavior_permanent_lock_progress", pd.Series(dtype=float)), errors="coerce")
    valid_lock = lock_diff.dropna()
    tol = 0.05
    lock_before = int((valid_lock < -tol).sum())
    lock_after = int((valid_lock > tol).sum())
    lock_around = int((valid_lock.abs() <= tol).sum())
    median_lock_diff = float(valid_lock.median()) if len(valid_lock) else float("nan")

    switches = pd.to_numeric(summary["activation_basin_switch_count"], errors="coerce")
    high_threshold = float(switches.quantile(0.75)) if switches.notna().any() else float("nan")
    low_threshold = float(switches.quantile(0.25)) if switches.notna().any() else float("nan")
    high_mask = switches.ge(high_threshold) if np.isfinite(high_threshold) else pd.Series(False, index=summary.index)
    low_mask = switches.le(low_threshold) if np.isfinite(low_threshold) else pd.Series(False, index=summary.index)
    high_n = int(high_mask.sum())
    low_n = int(low_mask.sum())
    high_prefixes = float(pd.to_numeric(summary.loc[high_mask, "num_prefixes"], errors="coerce").mean()) if high_n else float("nan")
    low_prefixes = float(pd.to_numeric(summary.loc[low_mask, "num_prefixes"], errors="coerce").mean()) if low_n else float("nan")
    high_tv = float(pd.to_numeric(summary.loc[high_mask, "activation_total_variation"], errors="coerce").mean()) if high_n else float("nan")
    low_tv = float(pd.to_numeric(summary.loc[low_mask, "activation_total_variation"], errors="coerce").mean()) if low_n else float("nan")
    high_path = float(pd.to_numeric(summary.loc[high_mask, "activation_manifold_path_length"], errors="coerce").mean()) if high_n else float("nan")
    low_path = float(pd.to_numeric(summary.loc[low_mask, "activation_manifold_path_length"], errors="coerce").mean()) if low_n else float("nan")

    tentative_mask = switches.gt(0) & pd.to_numeric(summary["permanent_activation_basin_lock_progress"], errors="coerce").notna()
    tentative_n = int(tentative_mask.sum())
    post_first_switches = pd.to_numeric(summary.get("switch_count_after_first_activation_lock", pd.Series(dtype=float)), errors="coerce")
    mean_post_first = float(post_first_switches.dropna().mean()) if post_first_switches.notna().any() else float("nan")

    lines = [
        "# Activation Dynamics and Switching Report",
        "",
        "## Summary",
        "",
        f"- Prefix rows: {len(prefix):,}",
        f"- Decisions: {summary['decision_id'].nunique():,}",
        f"- Trajectories: {summary['trajectory_id'].nunique():,}" if "trajectory_id" in summary.columns else "- Trajectories: unavailable",
        f"- Selected layer: {layer}",
        f"- Selected representation: `{representation}`",
        f"- Layer/representation source: {layer_source}",
        f"- Feature file: `{feature_path}`",
        f"- Activation lock margin threshold: {margin_threshold}",
        f"- Consecutive prefixes for first activation lock: {consecutive_prefixes}",
        "",
        "## Dynamics Metrics",
        "",
        f"- Mean activation basin switch count: {pd.to_numeric(summary['activation_basin_switch_count'], errors='coerce').mean():.4f}",
        f"- Mean behavior switch count: {pd.to_numeric(summary['behavior_switch_count'], errors='coerce').mean():.4f}",
        f"- Mean activation total variation: {pd.to_numeric(summary['activation_total_variation'], errors='coerce').mean():.4f}",
        f"- Mean behavior total variation: {pd.to_numeric(summary['behavior_total_variation'], errors='coerce').mean():.4f}",
        f"- Decisions with activation permanent lock: {int(pd.to_numeric(summary['permanent_activation_basin_lock_progress'], errors='coerce').notna().sum())}",
        f"- Decisions with behavior permanent lock: {int(pd.to_numeric(summary['permanent_behavior_lock_progress'], errors='coerce').notna().sum())}",
        "",
        "## Behavior vs Activation Alignment",
        "",
        f"- Switch-count correlation: n={n_switch}, Pearson={switch_r:.4f}, Spearman={switch_s:.4f}",
        f"- Permanent-lock progress correlation: n={n_lock}, Pearson={lock_r:.4f}, Spearman={lock_s:.4f}",
        f"- Total-variation correlation: n={n_tv}, Pearson={tv_r:.4f}, Spearman={tv_s:.4f}",
        "",
        "## Lock Timing",
        "",
        f"- Median activation-minus-behavior permanent lock progress: {median_lock_diff:.4f}",
        f"- Activation permanent lock earlier than behavior by > {tol}: {lock_before}",
        f"- Activation permanent lock around behavior within +/- {tol}: {lock_around}",
        f"- Activation permanent lock later than behavior by > {tol}: {lock_after}",
        "",
        "## High-switch Complexity",
        "",
        f"- High-switch threshold used: activation_basin_switch_count >= {high_threshold:.2f}",
        f"- Low-switch threshold used: activation_basin_switch_count <= {low_threshold:.2f}",
        f"- High-switch decisions: {high_n}; low-switch decisions: {low_n}",
        f"- Mean num_prefixes high-switch vs low-switch: {high_prefixes:.2f} vs {low_prefixes:.2f}",
        f"- Mean activation total variation high-switch vs low-switch: {high_tv:.4f} vs {low_tv:.4f}",
        f"- Mean activation manifold path length high-switch vs low-switch: {high_path:.4f} vs {low_path:.4f}",
        "",
        "## Tentative Commitment",
        "",
        f"- Decisions with activation switching followed by permanent activation lock: {tentative_n}",
        f"- Mean activation switches after first activation lock: {mean_post_first:.4f}",
        f"- High-switch example decisions: {', '.join(high_switch_examples) if high_switch_examples else 'none'}",
        f"- Tentative-then-permanent example decisions: {', '.join(tentative_examples) if tentative_examples else 'none'}",
        "",
        "## Answers",
        "",
        f"- Do behavior switches correspond to activation basin switches? Switch-count Pearson={switch_r:.4f} and Spearman={switch_s:.4f}; inspect the switch-count scatter for outliers and nonlinearity.",
        f"- Does activation permanent lock-in occur before, after, or around behavior lock-in? Among paired decisions, earlier={lock_before}, around={lock_around}, later={lock_after}, with median activation-minus-behavior progress {median_lock_diff:.4f}.",
        f"- Are high-switch decisions longer or more geometrically complex? High-switch decisions have mean prefixes {high_prefixes:.2f} vs {low_prefixes:.2f} for low-switch decisions, and mean activation total variation {high_tv:.4f} vs {low_tv:.4f}.",
        f"- Is there evidence for tentative commitment followed by stable commitment? {tentative_n} decisions show at least one activation basin switch and later permanent activation lock; example trajectories are plotted in the tentative/permanent figure.",
        "",
        "## Caveats",
        "",
        "- Activation distances are computed in the selected feature representation, while manifold path length and radius are computed in the 2D PCA manifold.",
        "- Activation basin switches depend on the centroid construction and lock margin inherited from the centroid score table.",
        "- A basin switch is not identical to a behavior-side probability switch; it is a nearest-centroid transition in activation space.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--centroid_scores", default="outputs/activation/activation_centroid_scores.parquet")
    parser.add_argument("--manifold", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--permanent_lockin_summary", default="outputs/followups/permanent_lockin_summary.parquet")
    parser.add_argument("--step_summary", default="outputs/step_commitment_summary.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--representation", default=None)
    parser.add_argument("--output", default="outputs/activation/activation_dynamics_summary.parquet")
    parser.add_argument("--prefix_output", default="outputs/activation/activation_dynamics_prefix_metrics.parquet")
    parser.add_argument("--report", default="outputs/activation/reports/activation_dynamics_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/dynamics")
    parser.add_argument("--lock_margin_threshold", type=float, default=None)
    parser.add_argument("--consecutive_prefixes", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=6)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = infer_repo_root(Path(args.centroid_scores))
    metadata_path = resolve_path(args.metadata, repo_root)
    centroid_path = resolve_path(args.centroid_scores, repo_root)
    manifold_path = resolve_path(args.manifold, repo_root)
    permanent_path = resolve_path(args.permanent_lockin_summary, repo_root)
    step_summary_path = resolve_path(args.step_summary, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    output_path = resolve_path(args.output, repo_root)
    prefix_output_path = resolve_path(args.prefix_output, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    metadata = make_decision_id(pd.read_parquet(metadata_path))
    centroid_scores = make_decision_id(pd.read_parquet(centroid_path))
    manifold = make_decision_id(pd.read_parquet(manifold_path))

    layer, representation, layer_source = select_layer_and_representation(
        requested_layer=args.layer,
        requested_representation=args.representation,
        features_dir=features_dir,
        centroid_scores=centroid_scores,
        manifold=manifold,
    )
    margin_threshold = args.lock_margin_threshold
    if margin_threshold is None:
        value = unique_nonnull_value(centroid_scores, "lock_margin_threshold")
        margin_threshold = float(value) if value is not None else 0.005

    base = merge_missing_metadata(centroid_scores, metadata)
    base = merge_manifold(base, manifold)
    base = make_decision_id(base)
    feature_path = resolve_feature_path(base, features_dir, repo_root, layer, representation)

    sort_cols = [col for col in ["decision_id", "prefix_token_index", "progress", "prefix_id"] if col in base.columns]
    base_sorted = base.sort_values(sort_cols).reset_index(drop=True).copy()
    X = load_feature_rows(base_sorted, feature_path)

    prefix_metrics, summary = build_prefix_and_summary(
        base_sorted,
        X,
        layer=layer,
        representation=representation,
        margin_threshold=margin_threshold,
        consecutive_prefixes=args.consecutive_prefixes,
    )
    summary = merge_behavior_summaries(summary, step_summary_path, permanent_path)

    ensure_dir(output_path.parent)
    ensure_dir(prefix_output_path.parent)
    ensure_dir(report_path.parent)
    ensure_dir(figures_dir)
    prefix_metrics.to_parquet(prefix_output_path, index=False)
    summary.to_parquet(output_path, index=False)

    scatter_plot(
        summary,
        "behavior_switch_count",
        "activation_basin_switch_count",
        figures_dir / "behavior_vs_activation_switch_count.png",
        xlabel="behavior switch count",
        ylabel="activation basin switch count",
        title="Behavior vs activation basin switch count",
    )
    scatter_plot(
        summary,
        "permanent_behavior_lock_progress",
        "permanent_activation_basin_lock_progress",
        figures_dir / "behavior_vs_activation_permanent_lockin.png",
        xlabel="behavior permanent lock-in progress",
        ylabel="activation permanent lock-in progress",
        title="Behavior vs activation permanent lock-in",
    )
    scatter_plot(
        summary,
        "behavior_total_variation",
        "activation_total_variation",
        figures_dir / "behavior_vs_activation_total_variation.png",
        xlabel="behavior total variation",
        ylabel="activation total variation",
        title="Behavior vs activation total variation",
    )
    plot_radius_over_progress(prefix_metrics, figures_dir / "activation_radius_over_progress_by_action.png")
    plot_path_length_by_action(summary, figures_dir / "activation_path_length_by_action.png")
    high_switch_examples = plot_example_high_switch(
        prefix_metrics,
        summary,
        figures_dir / "example_high_switch_activation_trajectories.png",
        max_examples=args.max_examples,
    )
    tentative_examples = plot_tentative_permanent(
        prefix_metrics,
        summary,
        figures_dir / "example_tentative_permanent_trajectories.png",
        max_examples=min(args.max_examples, 4),
    )

    write_report(
        report_path,
        prefix=prefix_metrics,
        summary=summary,
        layer=layer,
        representation=representation,
        layer_source=layer_source,
        feature_path=feature_path,
        margin_threshold=margin_threshold,
        consecutive_prefixes=args.consecutive_prefixes,
        high_switch_examples=high_switch_examples,
        tentative_examples=tentative_examples,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {prefix_output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote 7 figures under {figures_dir}")


if __name__ == "__main__":
    main()
