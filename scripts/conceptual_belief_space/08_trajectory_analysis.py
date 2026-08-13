#!/usr/bin/env python3
"""Step 8: trajectory analysis in behavior and activation conceptual spaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.plotting import (  # noqa: E402
    plot_activation_trajectories,
    plot_behavior_trajectories,
    savefig,
)
from src.conceptual_belief_space.stats import pearson_corr, spearman_corr  # noqa: E402
from src.conceptual_belief_space.trajectories import (  # noqa: E402
    distance_to_action_centroid,
    first_lock_in_progress,
    nearest_centroid_actions,
    ordered_centroid_coords,
    path_length,
    permanent_lock_in_progress,
    selected_decision_ids,
    switch_count,
    total_variation,
)


Q_COLS = [f"q_{action}" for action in ACTIONS]
CLR_COLS = [f"clr_{action}" for action in ACTIONS]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Step 8 requires conceptual UMAP outputs from Steps 4 and 5. "
            "Install UMAP with `pip install umap-learn`, rerun Steps 4 and 5, then rerun Step 8."
        )


def validate_inputs(
    beliefs: pd.DataFrame,
    behavior_coords: pd.DataFrame,
    behavior_centroids: pd.DataFrame,
    activation_coords: pd.DataFrame,
    activation_centroids: pd.DataFrame,
) -> None:
    missing_belief = [col for col in ["row_id", "decision_id", "agent_action", *Q_COLS, *CLR_COLS] if col not in beliefs.columns]
    if missing_belief:
        raise ValueError(f"belief table missing required columns: {missing_belief}")
    missing_behavior = [
        col
        for col in ["split_id", "subset", "representation", "row_id", "decision_id", "beh_umap1", "beh_umap2"]
        if col not in behavior_coords.columns
    ]
    if missing_behavior:
        raise ValueError(f"behavior coordinates missing columns: {missing_behavior}")
    missing_behavior_centroids = [
        col
        for col in ["split_id", "subset", "representation", "action", "beh_umap1", "beh_umap2"]
        if col not in behavior_centroids.columns
    ]
    if missing_behavior_centroids:
        raise ValueError(f"behavior centroids missing columns: {missing_behavior_centroids}")
    missing_activation = [
        col
        for col in [
            "split_id",
            "subset",
            "layer",
            "activation_representation",
            "row_id",
            "decision_id",
            "act_umap1",
            "act_umap2",
        ]
        if col not in activation_coords.columns
    ]
    if missing_activation:
        raise ValueError(f"activation coordinates missing columns: {missing_activation}")
    missing_activation_centroids = [
        col
        for col in ["split_id", "subset", "layer", "activation_representation", "action", "act_umap1", "act_umap2"]
        if col not in activation_centroids.columns
    ]
    if missing_activation_centroids:
        raise ValueError(f"activation centroids missing columns: {missing_activation_centroids}")


def add_belief_vectors(coords: pd.DataFrame, beliefs: pd.DataFrame) -> pd.DataFrame:
    keep = ["row_id", *Q_COLS, *CLR_COLS]
    return coords.merge(beliefs[keep], how="left", on="row_id", validate="many_to_one")


def behavior_prefix_scores_for_block(
    block: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    representation: str,
) -> pd.DataFrame:
    block = block.sort_values(["decision_id", "prefix_token_index", "row_id"]).copy()
    centroid_coords = ordered_centroid_coords(centroids, ["beh_umap1", "beh_umap2"])
    coords = block[["beh_umap1", "beh_umap2"]].to_numpy(dtype=float)
    nearest, nearest_distance = nearest_centroid_actions(coords, centroid_coords)
    final_distance = distance_to_action_centroid(coords, centroid_coords, block["agent_action"].astype(str))

    out_cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
        "beh_umap1",
        "beh_umap2",
    ]
    out_cols = [col for col in out_cols if col in block.columns]
    out = block[out_cols].copy()
    out.insert(0, "space", "behavior")
    out.insert(1, "split_id", int(split_id))
    out.insert(2, "subset", subset)
    out.insert(3, "behavior_representation", representation)
    out["layer"] = np.nan
    out["activation_representation"] = ""
    out["distance_to_final_centroid"] = final_distance
    out["nearest_centroid_action"] = nearest
    out["nearest_centroid_distance"] = nearest_distance
    return out


def activation_prefix_scores_for_block(
    block: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
) -> pd.DataFrame:
    block = block.sort_values(["decision_id", "prefix_token_index", "row_id"]).copy()
    centroid_coords = ordered_centroid_coords(centroids, ["act_umap1", "act_umap2"])
    coords = block[["act_umap1", "act_umap2"]].to_numpy(dtype=float)
    nearest, nearest_distance = nearest_centroid_actions(coords, centroid_coords)
    final_distance = distance_to_action_centroid(coords, centroid_coords, block["agent_action"].astype(str))

    out_cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
        "act_umap1",
        "act_umap2",
    ]
    out_cols = [col for col in out_cols if col in block.columns]
    out = block[out_cols].copy()
    out.insert(0, "space", "activation")
    out.insert(1, "split_id", int(split_id))
    out.insert(2, "subset", subset)
    out.insert(3, "behavior_representation", "")
    out["layer"] = int(layer)
    out["activation_representation"] = activation_representation
    out["distance_to_final_centroid"] = final_distance
    out["nearest_centroid_action"] = nearest
    out["nearest_centroid_distance"] = nearest_distance
    return out


def behavior_decision_metrics(block: pd.DataFrame, prefix_scores: pd.DataFrame, *, split_id: int, subset: str, representation: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision_id, part in block.sort_values(["decision_id", "prefix_token_index", "row_id"]).groupby("decision_id", sort=False):
        scores = prefix_scores.loc[prefix_scores["decision_id"] == decision_id].sort_values(["prefix_token_index", "row_id"])
        final_action = str(part["agent_action"].iloc[0])
        progress = part["progress"].to_numpy(dtype=float)
        rows.append(
            {
                "metric_scope": "decision",
                "space": "behavior",
                "split_id": int(split_id),
                "subset": subset,
                "behavior_representation": representation,
                "layer": np.nan,
                "activation_representation": "",
                "decision_id": decision_id,
                "trajectory_id": part["trajectory_id"].iloc[0] if "trajectory_id" in part.columns else "",
                "agent_action": final_action,
                "n_prefixes": int(len(part)),
                "behavior_path_length_umap": path_length(part[["beh_umap1", "beh_umap2"]].to_numpy(dtype=float)),
                "behavior_total_variation_q": total_variation(part[Q_COLS].to_numpy(dtype=float)),
                "behavior_total_variation_clr": total_variation(part[CLR_COLS].to_numpy(dtype=float)),
                "behavior_centroid_switch_count": switch_count(scores["nearest_centroid_action"]),
                "first_behavior_lock_in_progress": first_lock_in_progress(
                    progress,
                    scores["nearest_centroid_action"],
                    final_action,
                ),
                "permanent_behavior_lock_in_progress": permanent_lock_in_progress(
                    progress,
                    scores["nearest_centroid_action"],
                    final_action,
                ),
                "mean_behavior_distance_to_final_centroid": float(scores["distance_to_final_centroid"].mean()),
                "final_behavior_distance_to_final_centroid": float(scores["distance_to_final_centroid"].iloc[-1]),
            }
        )
    return rows


def activation_decision_metrics(block: pd.DataFrame, prefix_scores: pd.DataFrame, *, split_id: int, subset: str, layer: int, activation_representation: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision_id, part in block.sort_values(["decision_id", "prefix_token_index", "row_id"]).groupby("decision_id", sort=False):
        scores = prefix_scores.loc[prefix_scores["decision_id"] == decision_id].sort_values(["prefix_token_index", "row_id"])
        final_action = str(part["agent_action"].iloc[0])
        progress = part["progress"].to_numpy(dtype=float)
        rows.append(
            {
                "metric_scope": "decision",
                "space": "activation",
                "split_id": int(split_id),
                "subset": subset,
                "behavior_representation": "",
                "layer": int(layer),
                "activation_representation": activation_representation,
                "decision_id": decision_id,
                "trajectory_id": part["trajectory_id"].iloc[0] if "trajectory_id" in part.columns else "",
                "agent_action": final_action,
                "n_prefixes": int(len(part)),
                "activation_path_length_umap": path_length(part[["act_umap1", "act_umap2"]].to_numpy(dtype=float)),
                "activation_centroid_switch_count": switch_count(scores["nearest_centroid_action"]),
                "first_activation_lock_in_progress": first_lock_in_progress(
                    progress,
                    scores["nearest_centroid_action"],
                    final_action,
                ),
                "permanent_activation_lock_in_progress": permanent_lock_in_progress(
                    progress,
                    scores["nearest_centroid_action"],
                    final_action,
                ),
                "mean_activation_distance_to_final_centroid": float(scores["distance_to_final_centroid"].mean()),
                "final_activation_distance_to_final_centroid": float(scores["distance_to_final_centroid"].iloc[-1]),
            }
        )
    return rows


def compute_behavior_outputs(
    behavior_coords: pd.DataFrame,
    behavior_centroids: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = add_belief_vectors(behavior_coords, beliefs)
    prefix_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    for (split_id, subset, representation), block in coords.groupby(["split_id", "subset", "representation"], sort=True):
        centroids = behavior_centroids.loc[
            (behavior_centroids["split_id"] == split_id)
            & (behavior_centroids["subset"].astype(str) == str(subset))
            & (behavior_centroids["representation"].astype(str) == str(representation))
        ]
        if centroids.empty:
            continue
        prefix_scores = behavior_prefix_scores_for_block(
            block,
            centroids,
            split_id=int(split_id),
            subset=str(subset),
            representation=str(representation),
        )
        prefix_parts.append(prefix_scores)
        metrics_rows.extend(
            behavior_decision_metrics(
                block,
                prefix_scores,
                split_id=int(split_id),
                subset=str(subset),
                representation=str(representation),
            )
        )
    return pd.concat(prefix_parts, ignore_index=True), pd.DataFrame(metrics_rows)


def compute_activation_outputs(
    activation_coords: pd.DataFrame,
    activation_centroids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    for (split_id, subset, layer, representation), block in activation_coords.groupby(
        ["split_id", "subset", "layer", "activation_representation"],
        sort=True,
    ):
        centroids = activation_centroids.loc[
            (activation_centroids["split_id"] == split_id)
            & (activation_centroids["subset"].astype(str) == str(subset))
            & (activation_centroids["layer"] == layer)
            & (activation_centroids["activation_representation"].astype(str) == str(representation))
        ]
        if centroids.empty:
            continue
        prefix_scores = activation_prefix_scores_for_block(
            block,
            centroids,
            split_id=int(split_id),
            subset=str(subset),
            layer=int(layer),
            activation_representation=str(representation),
        )
        prefix_parts.append(prefix_scores)
        metrics_rows.extend(
            activation_decision_metrics(
                block,
                prefix_scores,
                split_id=int(split_id),
                subset=str(subset),
                layer=int(layer),
                activation_representation=str(representation),
            )
        )
    return pd.concat(prefix_parts, ignore_index=True), pd.DataFrame(metrics_rows)


def comparison_metrics(behavior_metrics: pd.DataFrame, activation_metrics: pd.DataFrame) -> pd.DataFrame:
    if behavior_metrics.empty or activation_metrics.empty:
        return pd.DataFrame()
    behavior_subset = behavior_metrics.loc[behavior_metrics["space"] == "behavior"].copy()
    activation_subset = activation_metrics.loc[activation_metrics["space"] == "activation"].copy()
    rows: list[dict[str, Any]] = []
    for (split_id, subset, behavior_representation), beh_block in behavior_subset.groupby(
        ["split_id", "subset", "behavior_representation"],
        sort=True,
    ):
        act_blocks = activation_subset.loc[
            (activation_subset["split_id"] == split_id)
            & (activation_subset["subset"].astype(str) == str(subset))
        ]
        for (layer, activation_representation), act_block in act_blocks.groupby(["layer", "activation_representation"], sort=True):
            merged = beh_block.merge(
                act_block,
                how="inner",
                on="decision_id",
                suffixes=("_behavior", "_activation"),
            )
            if merged.empty:
                continue
            pairs = [
                (
                    "path_length",
                    "behavior_path_length_umap",
                    "activation_path_length_umap",
                ),
                (
                    "switch_count",
                    "behavior_centroid_switch_count",
                    "activation_centroid_switch_count",
                ),
                (
                    "first_lock_in_progress",
                    "first_behavior_lock_in_progress",
                    "first_activation_lock_in_progress",
                ),
                (
                    "permanent_lock_in_progress",
                    "permanent_behavior_lock_in_progress",
                    "permanent_activation_lock_in_progress",
                ),
                (
                    "mean_distance_to_final_centroid",
                    "mean_behavior_distance_to_final_centroid",
                    "mean_activation_distance_to_final_centroid",
                ),
            ]
            for metric_name, beh_col, act_col in pairs:
                rows.append(
                    {
                        "metric_scope": "comparison",
                        "space": "behavior_vs_activation",
                        "split_id": int(split_id),
                        "subset": str(subset),
                        "behavior_representation": str(behavior_representation),
                        "layer": int(layer),
                        "activation_representation": str(activation_representation),
                        "comparison_metric": metric_name,
                        "n_decisions": int(len(merged)),
                        "pearson": pearson_corr(merged[beh_col], merged[act_col]),
                        "spearman": spearman_corr(merged[beh_col], merged[act_col]),
                    }
                )
    return pd.DataFrame(rows)


def load_geometry_metrics(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def select_default_blocks(
    prefix_scores: pd.DataFrame,
    geometry_metrics: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior = prefix_scores.loc[prefix_scores["space"] == "behavior"].copy()
    activation = prefix_scores.loc[prefix_scores["space"] == "activation"].copy()
    if behavior.empty or activation.empty:
        return behavior, activation
    if geometry_metrics is not None and not geometry_metrics.empty and "is_primary" in geometry_metrics.columns:
        primary = geometry_metrics.loc[geometry_metrics["is_primary"].astype(bool)].copy()
        primary = primary.loc[np.isfinite(pd.to_numeric(primary.get("spearman"), errors="coerce"))]
        if not primary.empty:
            best = primary.sort_values(["spearman", "pearson"], ascending=False).iloc[0]
            split_id = int(best["split_id"])
            subset = str(best["subset"])
            layer = int(best["activation_layer"])
            activation_representation = str(best["activation_representation"])
            behavior_representation = str(best["behavior_representation"])
            return (
                behavior.loc[
                    (behavior["split_id"] == split_id)
                    & (behavior["subset"].astype(str) == subset)
                    & (behavior["behavior_representation"].astype(str) == behavior_representation)
                ].copy(),
                activation.loc[
                    (activation["split_id"] == split_id)
                    & (activation["subset"].astype(str) == subset)
                    & (activation["layer"] == layer)
                    & (activation["activation_representation"].astype(str) == activation_representation)
                ].copy(),
            )

    beh_first = behavior[["split_id", "subset", "behavior_representation"]].drop_duplicates().iloc[0]
    act_match = activation.loc[
        (activation["split_id"] == beh_first["split_id"])
        & (activation["subset"].astype(str) == str(beh_first["subset"]))
    ]
    if act_match.empty:
        act_first = activation[["split_id", "subset", "layer", "activation_representation"]].drop_duplicates().iloc[0]
    else:
        act_first = act_match[["split_id", "subset", "layer", "activation_representation"]].drop_duplicates().iloc[0]
    return (
        behavior.loc[
            (behavior["split_id"] == beh_first["split_id"])
            & (behavior["subset"].astype(str) == str(beh_first["subset"]))
            & (behavior["behavior_representation"].astype(str) == str(beh_first["behavior_representation"]))
        ].copy(),
        activation.loc[
            (activation["split_id"] == act_first["split_id"])
            & (activation["subset"].astype(str) == str(act_first["subset"]))
            & (activation["layer"] == act_first["layer"])
            & (activation["activation_representation"].astype(str) == str(act_first["activation_representation"]))
        ].copy(),
    )


def make_figures(
    prefix_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    geometry_metrics: pd.DataFrame | None,
    figure_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []
    behavior_block, activation_block = select_default_blocks(prefix_scores, geometry_metrics)
    if not behavior_block.empty:
        decisions = selected_decision_ids(behavior_block)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_behavior_trajectories(
            behavior_block,
            decision_ids=decisions,
            ax=ax,
            title="Selected behavior trajectories in conceptual UMAP space",
        )
        path = figure_dir / "selected_behavior_trajectories.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Selected behavior trajectories"})

    if not activation_block.empty:
        decisions = selected_decision_ids(activation_block)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_activation_trajectories(
            activation_block,
            decision_ids=decisions,
            ax=ax,
            title="Selected activation trajectories in conceptual UMAP space",
        )
        path = figure_dir / "selected_activation_trajectories.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Selected activation trajectories"})

    if not behavior_block.empty or not activation_block.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if not behavior_block.empty:
            binned = behavior_block.copy()
            binned["progress_bin"] = pd.cut(binned["progress"], bins=np.linspace(0, 1, 11), include_lowest=True)
            curve = binned.groupby("progress_bin", observed=True)["distance_to_final_centroid"].mean()
            x = [interval.mid for interval in curve.index]
            ax.plot(x, curve.to_numpy(dtype=float), marker="o", label="behavior")
        if not activation_block.empty:
            binned = activation_block.copy()
            binned["progress_bin"] = pd.cut(binned["progress"], bins=np.linspace(0, 1, 11), include_lowest=True)
            curve = binned.groupby("progress_bin", observed=True)["distance_to_final_centroid"].mean()
            x = [interval.mid for interval in curve.index]
            ax.plot(x, curve.to_numpy(dtype=float), marker="o", label="activation")
        ax.set_xlabel("reasoning progress")
        ax.set_ylabel("mean distance to final-action centroid")
        ax.set_title("Distance to final centroid over progress")
        ax.legend(frameon=False)
        path = figure_dir / "distance_to_final_centroid_over_progress.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Distance to final centroid over progress"})

    comparison = metrics.loc[metrics["metric_scope"] == "comparison"].copy() if "metric_scope" in metrics.columns else pd.DataFrame()
    if not comparison.empty:
        for metric_name, filename, title in [
            ("path_length", "behavior_vs_activation_path_length.png", "Behavior vs activation path length correlation"),
            ("switch_count", "behavior_vs_activation_switch_count.png", "Behavior vs activation switch count correlation"),
            ("first_lock_in_progress", "behavior_vs_activation_lock_in_progress.png", "Behavior vs activation lock-in correlation"),
        ]:
            part = comparison.loc[comparison["comparison_metric"] == metric_name].copy()
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(part["layer"], part["spearman"], s=24, alpha=0.75)
            ax.axhline(0, color="#888888", lw=0.8)
            ax.set_xlabel("activation layer")
            ax.set_ylabel("Spearman correlation across decisions")
            ax.set_title(title)
            path = figure_dir / filename
            savefig(path, fig=fig)
            figures.append({"path": rel(path), "description": title})
    return figures


def write_report(
    path: Path,
    *,
    beliefs_path: Path,
    behavior_coords_path: Path,
    behavior_centroids_path: Path,
    activation_coords_path: Path,
    activation_centroids_path: Path,
    geometry_metrics_path: Path,
    metrics_output: Path,
    prefix_scores_output: Path,
    metrics: pd.DataFrame,
    prefix_scores: pd.DataFrame,
    figures: list[dict[str, str]],
) -> None:
    decision_rows = metrics.loc[metrics["metric_scope"] == "decision"] if "metric_scope" in metrics.columns else pd.DataFrame()
    comparison_rows = metrics.loc[metrics["metric_scope"] == "comparison"] if "metric_scope" in metrics.columns else pd.DataFrame()
    behavior_decisions = decision_rows.loc[decision_rows["space"] == "behavior"] if not decision_rows.empty else pd.DataFrame()
    activation_decisions = decision_rows.loc[decision_rows["space"] == "activation"] if not decision_rows.empty else pd.DataFrame()
    lines = [
        "# Trajectory Analysis",
        "",
        "Step 8 analyzes decision-level trajectories through behavior and activation conceptual UMAP spaces.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Beliefs: `{rel(beliefs_path)}`",
        f"- Behavior coordinates: `{rel(behavior_coords_path)}`",
        f"- Behavior centroids: `{rel(behavior_centroids_path)}`",
        f"- Activation coordinates: `{rel(activation_coords_path)}`",
        f"- Activation 2D centroids: `{rel(activation_centroids_path)}`",
        f"- Geometry alignment metrics: `{rel(geometry_metrics_path)}`",
        f"- Trajectory metrics: `{rel(metrics_output)}`",
        f"- Prefix trajectory scores: `{rel(prefix_scores_output)}`",
        "",
        "## Method",
        "",
        "- Uses conceptual UMAP coordinates from Steps 4 and 5.",
        "- Does not use old blind PCA coordinates.",
        "- Nearest-centroid basin labels are descriptive and should not be interpreted as a causal mechanism.",
        "- Behavior total variation is computed in both q and CLR(q) spaces.",
        "- Lock-in is defined descriptively as nearest centroid matching the final action; permanent lock-in requires all later prefixes to remain matched.",
        "",
        "## Counts",
        "",
        f"- Prefix score rows: `{len(prefix_scores)}`",
        f"- Decision metric rows: `{len(decision_rows)}`",
        f"- Behavior decision metric rows: `{len(behavior_decisions)}`",
        f"- Activation decision metric rows: `{len(activation_decisions)}`",
        f"- Comparison rows: `{len(comparison_rows)}`",
    ]
    if not behavior_decisions.empty:
        lines.extend(
            [
                "",
                "## Behavior Summary",
                "",
                f"- Mean behavior path length: `{float(behavior_decisions['behavior_path_length_umap'].mean())}`",
                f"- Mean behavior q total variation: `{float(behavior_decisions['behavior_total_variation_q'].mean())}`",
                f"- Mean behavior CLR total variation: `{float(behavior_decisions['behavior_total_variation_clr'].mean())}`",
                f"- Mean behavior centroid switch count: `{float(behavior_decisions['behavior_centroid_switch_count'].mean())}`",
            ]
        )
    if not activation_decisions.empty:
        lines.extend(
            [
                "",
                "## Activation Summary",
                "",
                f"- Mean activation path length: `{float(activation_decisions['activation_path_length_umap'].mean())}`",
                f"- Mean activation centroid switch count: `{float(activation_decisions['activation_centroid_switch_count'].mean())}`",
            ]
        )
    if not comparison_rows.empty:
        top = comparison_rows.sort_values("spearman", ascending=False).head(10)
        lines.extend(["", "## Top Comparison Correlations", ""])
        for row in top.itertuples():
            lines.append(
                f"- `{row.comparison_metric}`, split `{row.split_id}`, subset `{row.subset}`, "
                f"layer `{int(row.layer)}`, repr `{row.activation_representation}`: "
                f"Spearman `{row.spearman}`, Pearson `{row.pearson}`"
            )
    lines.extend(["", "## Figures", ""])
    if figures:
        for fig in figures:
            lines.append(f"- `{fig['path']}`: {fig['description']}")
    else:
        lines.append("- No figures generated.")
    save_markdown(path, "\n".join(lines))


def run(
    *,
    beliefs_path: Path,
    behavior_coords_path: Path,
    behavior_centroids_path: Path,
    activation_coords_path: Path,
    activation_centroids_path: Path,
    geometry_metrics_path: Path,
    metrics_output: Path,
    prefix_scores_output: Path,
    report_output: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for input_path, label in [
        (beliefs_path, "action belief vectors"),
        (behavior_coords_path, "behavior UMAP coordinates"),
        (behavior_centroids_path, "behavior centroids"),
        (activation_coords_path, "activation UMAP coordinates"),
        (activation_centroids_path, "activation 2D centroids"),
    ]:
        require_existing(input_path, label)

    beliefs = read_parquet_checked(beliefs_path)
    behavior_coords = read_parquet_checked(behavior_coords_path)
    behavior_centroids = read_parquet_checked(behavior_centroids_path)
    activation_coords = read_parquet_checked(activation_coords_path)
    activation_centroids = read_parquet_checked(activation_centroids_path)
    validate_inputs(beliefs, behavior_coords, behavior_centroids, activation_coords, activation_centroids)

    geometry_metrics = load_geometry_metrics(geometry_metrics_path)
    behavior_prefix_scores, behavior_metrics = compute_behavior_outputs(
        behavior_coords,
        behavior_centroids,
        beliefs,
    )
    activation_prefix_scores, activation_metrics = compute_activation_outputs(
        activation_coords,
        activation_centroids,
    )
    prefix_scores = pd.concat([behavior_prefix_scores, activation_prefix_scores], ignore_index=True)
    comparisons = comparison_metrics(behavior_metrics, activation_metrics)
    metrics = pd.concat([behavior_metrics, activation_metrics, comparisons], ignore_index=True, sort=False)

    ensure_dir(prefix_scores_output.parent)
    prefix_scores.to_parquet(prefix_scores_output, index=False)
    ensure_dir(metrics_output.parent)
    metrics.to_csv(metrics_output, index=False)
    figures = make_figures(prefix_scores, metrics, geometry_metrics, figure_dir)
    write_report(
        report_output,
        beliefs_path=beliefs_path,
        behavior_coords_path=behavior_coords_path,
        behavior_centroids_path=behavior_centroids_path,
        activation_coords_path=activation_coords_path,
        activation_centroids_path=activation_centroids_path,
        geometry_metrics_path=geometry_metrics_path,
        metrics_output=metrics_output,
        prefix_scores_output=prefix_scores_output,
        metrics=metrics,
        prefix_scores=prefix_scores,
        figures=figures,
    )
    return metrics, prefix_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--behavior_coords", default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_coords.parquet")
    parser.add_argument("--behavior_centroids", default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_centroids.parquet")
    parser.add_argument("--activation_coords", default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_coords.parquet")
    parser.add_argument(
        "--activation_centroids_2d",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_2d.parquet",
    )
    parser.add_argument(
        "--geometry_metrics",
        default=DEFAULT_OUTPUT_DIR / "geometry_alignment/distance_matrix_metrics.csv",
    )
    parser.add_argument(
        "--metrics_output",
        default=DEFAULT_OUTPUT_DIR / "trajectories/trajectory_metrics.csv",
    )
    parser.add_argument(
        "--prefix_scores_output",
        default=DEFAULT_OUTPUT_DIR / "trajectories/prefix_trajectory_scores.parquet",
    )
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "trajectories/08_trajectory_report.md")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "trajectories")
    args = parser.parse_args()

    metrics, prefix_scores = run(
        beliefs_path=resolve_repo_path(args.beliefs),
        behavior_coords_path=resolve_repo_path(args.behavior_coords),
        behavior_centroids_path=resolve_repo_path(args.behavior_centroids),
        activation_coords_path=resolve_repo_path(args.activation_coords),
        activation_centroids_path=resolve_repo_path(args.activation_centroids_2d),
        geometry_metrics_path=resolve_repo_path(args.geometry_metrics),
        metrics_output=resolve_repo_path(args.metrics_output),
        prefix_scores_output=resolve_repo_path(args.prefix_scores_output),
        report_output=resolve_repo_path(args.report),
        figure_dir=resolve_repo_path(args.figure_dir),
    )
    print(f"Wrote {rel(args.metrics_output)} with {len(metrics)} rows")
    print(f"Wrote {rel(args.prefix_scores_output)} with {len(prefix_scores)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
