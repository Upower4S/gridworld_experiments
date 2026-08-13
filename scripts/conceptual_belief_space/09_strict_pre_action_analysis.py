#!/usr/bin/env python3
"""Step 9: strict pre-action conceptual belief analysis."""

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
from src.conceptual_belief_space.metrics import commitment_from_q  # noqa: E402
from src.conceptual_belief_space.stats import pearson_corr  # noqa: E402
from src.conceptual_belief_space.trajectories import (  # noqa: E402
    distance_to_action_centroid,
    distances_to_centroids,
    nearest_centroid_actions,
    ordered_centroid_coords,
)
from src.conceptual_belief_space.plotting import savefig  # noqa: E402


Q_COLS = [f"q_{action}" for action in ACTIONS]
STRICT_SUBSETS = [
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "buffered_before_any_action_mention",
    "buffered_before_final_action_mention",
]
BOUNDARY_INDEX = {
    "any_action": "first_action_mention_token_index",
    "final_action": "first_final_action_mention_token_index",
}
SUBSET_BOUNDARY = {
    "strict_before_any_action_mention": "any_action",
    "buffered_before_any_action_mention": "any_action",
    "strict_before_final_action_mention": "final_action",
    "buffered_before_final_action_mention": "final_action",
}
BASE_COLS = [
    "row_id",
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
    "first_action_mention_token_index",
    "first_final_action_mention_token_index",
]


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


def optional_read_parquet(path: Path, label: str, notes: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        notes.append(f"Optional input unavailable: {label} at `{rel(path)}`")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        notes.append(f"Optional input failed to read: {label} at `{rel(path)}`: {type(exc).__name__}: {exc}")
        return None


def optional_read_table(path: Path, label: str, notes: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        notes.append(f"Optional input unavailable: {label} at `{rel(path)}`")
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as exc:
        notes.append(f"Optional input failed to read: {label} at `{rel(path)}`: {type(exc).__name__}: {exc}")
        return None


def validate_beliefs(df: pd.DataFrame) -> None:
    missing = [col for col in ["row_id", "decision_id", "prefix_token_index", "agent_action", "top_action", *Q_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"belief table missing required columns: {missing}")
    if not df["row_id"].is_unique:
        raise ValueError("belief row_id must be unique")
    invalid = sorted(set(df["agent_action"].dropna().astype(str)) - set(ACTIONS))
    if invalid:
        raise ValueError(f"invalid agent_action labels: {invalid}")


def normalized_progress(df: pd.DataFrame, boundary: str) -> pd.Series:
    index_col = BOUNDARY_INDEX[boundary]
    if index_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    denom = pd.to_numeric(df[index_col], errors="coerce")
    prefix = pd.to_numeric(df["prefix_token_index"], errors="coerce")
    out = prefix / denom.replace(0, np.nan)
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(float)


def normalized_bin(values: pd.Series, n_bins: int) -> pd.Series:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cut = pd.cut(values.clip(lower=0.0, upper=1.0), bins=bins, include_lowest=True)
    return cut.astype(str)


def strict_base_rows(beliefs: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    keep_cols = [col for col in BASE_COLS if col in beliefs.columns]
    for subset in STRICT_SUBSETS:
        if subset not in beliefs.columns:
            skipped.append({"subset": subset, "reason": "missing subset column"})
            continue
        boundary = SUBSET_BOUNDARY[subset]
        index_col = BOUNDARY_INDEX[boundary]
        if index_col not in beliefs.columns:
            skipped.append({"subset": subset, "reason": f"missing {index_col}"})
            continue
        mask = beliefs[subset].fillna(False).astype(bool)
        part = beliefs.loc[mask, keep_cols].copy()
        if part.empty:
            skipped.append({"subset": subset, "reason": "no rows"})
            continue
        part["strict_subset"] = subset
        part["boundary_type"] = boundary
        part["boundary_token_index"] = pd.to_numeric(part[index_col], errors="coerce")
        part["normalized_progress_to_mention"] = normalized_progress(part, boundary)
        part = part.loc[part["normalized_progress_to_mention"].notna()].copy()
        if part.empty:
            skipped.append({"subset": subset, "reason": "no rows with finite normalized progress"})
            continue
        part["normalized_bin"] = normalized_bin(part["normalized_progress_to_mention"], n_bins)
        part["top_action_is_final"] = part["top_action"].astype(str) == part["agent_action"].astype(str)
        rows.append(part)
    if not rows:
        return pd.DataFrame(), skipped
    return pd.concat(rows, ignore_index=True), skipped


def behavior_prefix_scores(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out.insert(0, "source", "behavior")
    out["split_id"] = np.nan
    out["layer"] = np.nan
    out["activation_representation"] = ""
    out["probe_model"] = ""
    out["control_type"] = ""
    out["activation_distance_to_final_centroid"] = np.nan
    out["activation_similarity_to_final_centroid"] = np.nan
    out["activation_nearest_centroid_action"] = ""
    out["activation_basin_accuracy"] = np.nan
    out["activation_centroid_margin"] = np.nan
    out["probe_predicted_commitment_to_final"] = np.nan
    out["probe_predicted_top_action"] = ""
    out["probe_top_action_accuracy"] = np.nan
    return out


def activation_scores_from_coords(
    base: pd.DataFrame,
    activation_coords: pd.DataFrame | None,
    activation_centroids: pd.DataFrame | None,
    prefix_trajectory_scores: pd.DataFrame | None,
    notes: list[str],
) -> pd.DataFrame:
    if activation_coords is not None and activation_centroids is not None:
        required_coords = [
            "split_id",
            "subset",
            "layer",
            "activation_representation",
            "row_id",
            "decision_id",
            "agent_action",
            "act_umap1",
            "act_umap2",
        ]
        required_centroids = ["split_id", "subset", "layer", "activation_representation", "action", "act_umap1", "act_umap2"]
        if any(col not in activation_coords.columns for col in required_coords) or any(col not in activation_centroids.columns for col in required_centroids):
            notes.append("Activation coordinate/centroid inputs were present but missing required columns; activation analysis skipped.")
        else:
            parts: list[pd.DataFrame] = []
            for (split_id, subset, layer, representation), coords_block in activation_coords.groupby(
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
                merged = base.merge(
                    coords_block[["row_id", "act_umap1", "act_umap2"]],
                    how="inner",
                    on="row_id",
                    validate="many_to_one",
                )
                if merged.empty:
                    continue
                centroid_coords = ordered_centroid_coords(centroids, ["act_umap1", "act_umap2"])
                coords = merged[["act_umap1", "act_umap2"]].to_numpy(dtype=float)
                distances = distances_to_centroids(coords, centroid_coords)
                nearest, nearest_dist = nearest_centroid_actions(coords, centroid_coords)
                final_dist = distance_to_action_centroid(coords, centroid_coords, merged["agent_action"].astype(str))
                sorted_dist = np.sort(distances, axis=1)
                margin = sorted_dist[:, 1] - sorted_dist[:, 0]
                part = merged.copy()
                part.insert(0, "source", "activation")
                part["split_id"] = int(split_id)
                part["layer"] = int(layer)
                part["activation_representation"] = str(representation)
                part["probe_model"] = ""
                part["control_type"] = ""
                part["activation_distance_to_final_centroid"] = final_dist
                part["activation_similarity_to_final_centroid"] = -final_dist
                part["activation_nearest_centroid_action"] = nearest
                part["activation_nearest_centroid_distance"] = nearest_dist
                part["activation_centroid_margin"] = margin
                part["activation_basin_accuracy"] = nearest == part["agent_action"].astype(str).to_numpy()
                part["probe_predicted_commitment_to_final"] = np.nan
                part["probe_predicted_top_action"] = ""
                part["probe_top_action_accuracy"] = np.nan
                parts.append(part.drop(columns=["act_umap1", "act_umap2"]))
            if parts:
                return pd.concat(parts, ignore_index=True)
            notes.append("Activation coordinate inputs were present but no matching strict pre-action rows were found.")

    if prefix_trajectory_scores is not None and "space" in prefix_trajectory_scores.columns:
        act = prefix_trajectory_scores.loc[prefix_trajectory_scores["space"].astype(str) == "activation"].copy()
        required = ["row_id", "split_id", "layer", "activation_representation", "distance_to_final_centroid", "nearest_centroid_action"]
        if not act.empty and all(col in act.columns for col in required):
            merged = base.merge(
                act[required + [col for col in ["nearest_centroid_distance"] if col in act.columns]],
                how="inner",
                on="row_id",
            )
            if not merged.empty:
                out = merged.copy()
                out.insert(0, "source", "activation")
                out["probe_model"] = ""
                out["control_type"] = ""
                out["activation_distance_to_final_centroid"] = out["distance_to_final_centroid"]
                out["activation_similarity_to_final_centroid"] = -out["distance_to_final_centroid"]
                out["activation_nearest_centroid_action"] = out["nearest_centroid_action"].astype(str)
                out["activation_centroid_margin"] = np.nan
                out["activation_basin_accuracy"] = out["activation_nearest_centroid_action"] == out["agent_action"].astype(str)
                out["probe_predicted_commitment_to_final"] = np.nan
                out["probe_predicted_top_action"] = ""
                out["probe_top_action_accuracy"] = np.nan
                return out.drop(columns=[c for c in ["distance_to_final_centroid", "nearest_centroid_action"] if c in out.columns])
        notes.append("Prefix trajectory scores were present but did not contain usable activation rows.")
    else:
        notes.append("Activation conceptual similarity unavailable because activation coordinates/centroids or trajectory prefix scores are missing.")

    return pd.DataFrame()


def probe_q_columns(preds: pd.DataFrame, prefix: str) -> list[str]:
    return [f"{prefix}_q_{action}" for action in ACTIONS]


def available_probe_prefixes(preds: pd.DataFrame) -> list[str]:
    candidates = ["ridge_clr", "ridge_q", "elasticnet_clr"]
    return [prefix for prefix in candidates if all(col in preds.columns for col in probe_q_columns(preds, prefix))]


def probe_scores(base: pd.DataFrame, probe_predictions: pd.DataFrame | None, notes: list[str]) -> pd.DataFrame:
    if probe_predictions is None:
        notes.append("Probe-predicted belief unavailable because probe predictions are missing.")
        return pd.DataFrame()
    prefixes = available_probe_prefixes(probe_predictions)
    if not prefixes:
        notes.append("Probe predictions were present but no recognized predicted q columns were found.")
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    meta_cols = [col for col in ["row_id", "split_id", "subset", "layer", "activation_representation"] if col in probe_predictions.columns]
    for prefix in prefixes:
        q_cols = probe_q_columns(probe_predictions, prefix)
        top_col = f"{prefix}_pred_top_action"
        pred_cols = meta_cols + q_cols + ([top_col] if top_col in probe_predictions.columns else [])
        merged = base.merge(probe_predictions[pred_cols], how="inner", on="row_id")
        if merged.empty:
            continue
        q = merged[q_cols].to_numpy(dtype=float)
        pred_commitment = commitment_from_q(q, merged["agent_action"].astype(str))
        out = merged.copy()
        out.insert(0, "source", "probe")
        out["probe_model"] = prefix
        out["control_type"] = "true"
        if "split_id" not in out.columns:
            out["split_id"] = np.nan
        if "layer" not in out.columns:
            out["layer"] = np.nan
        if "activation_representation" not in out.columns:
            out["activation_representation"] = ""
        out["activation_distance_to_final_centroid"] = np.nan
        out["activation_similarity_to_final_centroid"] = np.nan
        out["activation_nearest_centroid_action"] = ""
        out["activation_basin_accuracy"] = np.nan
        out["activation_centroid_margin"] = np.nan
        out["probe_predicted_commitment_to_final"] = pred_commitment
        if top_col in out.columns:
            out["probe_predicted_top_action"] = out[top_col].astype(str)
            out["probe_top_action_accuracy"] = out["probe_predicted_top_action"] == out["agent_action"].astype(str)
        else:
            out["probe_predicted_top_action"] = ""
            out["probe_top_action_accuracy"] = np.nan
        parts.append(out.drop(columns=[col for col in q_cols + [top_col] if col in out.columns]))
    if parts:
        return pd.concat(parts, ignore_index=True)
    notes.append("Probe predictions were present but no strict pre-action rows matched.")
    return pd.DataFrame()


def metric_rows_for_group(group: pd.DataFrame, base_context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_prefix = int(len(group))
    n_decisions = int(group["decision_id"].nunique())
    action_counts = group["agent_action"].value_counts().to_dict()
    common = {
        **base_context,
        "n_prefix_rows": n_prefix,
        "n_decisions": n_decisions,
    }
    rows.append({**common, "metric": "prefix_count", "value": float(n_prefix), "action": ""})
    rows.append({**common, "metric": "decision_count", "value": float(n_decisions), "action": ""})
    for action in ACTIONS:
        rows.append({**common, "metric": "action_prefix_count", "value": float(action_counts.get(action, 0)), "action": action})

    numeric_metrics = [
        "commitment_to_final",
        "entropy_commitment",
        "top_action_is_final",
        "activation_similarity_to_final_centroid",
        "activation_distance_to_final_centroid",
        "activation_basin_accuracy",
        "activation_centroid_margin",
        "probe_predicted_commitment_to_final",
        "probe_top_action_accuracy",
    ]
    for metric in numeric_metrics:
        if metric in group.columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not values.empty:
                rows.append({**common, "metric": f"mean_{metric}", "value": float(values.mean()), "action": ""})
    if {"commitment_to_final", "probe_predicted_commitment_to_final"}.issubset(group.columns):
        values = group[["commitment_to_final", "probe_predicted_commitment_to_final"]].dropna()
        if len(values) >= 2:
            rows.append(
                {
                    **common,
                    "metric": "pearson_true_vs_probe_predicted_commitment",
                    "value": pearson_corr(values["commitment_to_final"], values["probe_predicted_commitment_to_final"]),
                    "action": "",
                }
            )
    return rows


def aggregate_metrics(prefix_scores: pd.DataFrame) -> pd.DataFrame:
    if prefix_scores.empty:
        return pd.DataFrame()
    group_cols = [
        "source",
        "strict_subset",
        "boundary_type",
        "normalized_bin",
        "split_id",
        "layer",
        "activation_representation",
        "probe_model",
        "control_type",
    ]
    for col in group_cols:
        if col not in prefix_scores.columns:
            prefix_scores[col] = ""
    rows: list[dict[str, Any]] = []
    for key, group in prefix_scores.groupby(group_cols, dropna=False, sort=True):
        context = dict(zip(group_cols, key))
        rows.extend(metric_rows_for_group(group, context))
    return pd.DataFrame(rows)


def probe_control_summary(probe_metrics: pd.DataFrame | None) -> pd.DataFrame:
    if probe_metrics is None or probe_metrics.empty or "control_type" not in probe_metrics.columns:
        return pd.DataFrame()
    metric_candidates = [col for col in ["mean_r2", "r2", "balanced_accuracy", "top_action_accuracy_vs_true_top_action"] if col in probe_metrics.columns]
    rows: list[dict[str, Any]] = []
    for target, target_group in probe_metrics.groupby("target", dropna=False):
        for metric in metric_candidates:
            values = target_group[["control_type", metric]].dropna()
            if values.empty:
                continue
            for control_type, part in values.groupby("control_type"):
                rows.append(
                    {
                        "source": "probe_control_summary",
                        "strict_subset": "all_probe_metrics",
                        "boundary_type": "",
                        "normalized_bin": "",
                        "split_id": np.nan,
                        "layer": np.nan,
                        "activation_representation": "",
                        "probe_model": "",
                        "control_type": control_type,
                        "n_prefix_rows": np.nan,
                        "n_decisions": np.nan,
                        "metric": f"mean_probe_metric_{target}_{metric}",
                        "value": float(part[metric].mean()),
                        "action": "",
                    }
                )
    return pd.DataFrame(rows)


def make_figures(metrics: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []

    def metric_curve(metric: str, source: str | None = None) -> pd.DataFrame:
        part = metrics.loc[metrics["metric"] == metric].copy()
        if source is not None:
            part = part.loc[part["source"] == source]
        if part.empty:
            return part
        grouped = (
            part.groupby(["strict_subset", "normalized_bin"], dropna=False)["value"]
            .mean()
            .reset_index()
            .sort_values(["strict_subset", "normalized_bin"])
        )
        return grouped

    coverage = metric_curve("prefix_count", source="behavior")
    if not coverage.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for subset, part in coverage.groupby("strict_subset"):
            ax.plot(range(len(part)), part["value"], marker="o", label=str(subset))
        ax.set_xlabel("normalized progress bin")
        ax.set_ylabel("prefix rows")
        ax.set_title("Strict pre-action coverage by normalized bin")
        ax.legend(frameon=False, fontsize=8)
        path = figure_dir / "coverage_by_normalized_bin.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Coverage by normalized bin"})

    behavior = metric_curve("mean_commitment_to_final", source="behavior")
    if not behavior.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for subset, part in behavior.groupby("strict_subset"):
            ax.plot(range(len(part)), part["value"], marker="o", label=str(subset))
        ax.set_xlabel("normalized progress bin")
        ax.set_ylabel("mean q(final action)")
        ax.set_title("Behavior commitment before action mention")
        ax.legend(frameon=False, fontsize=8)
        path = figure_dir / "behavior_commitment_before_mention.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Behavior commitment before mention"})

    activation_similarity = metric_curve("mean_activation_similarity_to_final_centroid", source="activation")
    if not activation_similarity.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for subset, part in activation_similarity.groupby("strict_subset"):
            ax.plot(range(len(part)), part["value"], marker="o", label=str(subset))
        ax.set_xlabel("normalized progress bin")
        ax.set_ylabel("mean negative distance to final centroid")
        ax.set_title("Activation final-centroid similarity before action mention")
        ax.legend(frameon=False, fontsize=8)
        path = figure_dir / "activation_final_centroid_similarity_before_mention.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Activation final-centroid similarity before mention"})

    probe_commitment = metric_curve("mean_probe_predicted_commitment_to_final", source="probe")
    if not probe_commitment.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for subset, part in probe_commitment.groupby("strict_subset"):
            ax.plot(range(len(part)), part["value"], marker="o", label=str(subset))
        ax.set_xlabel("normalized progress bin")
        ax.set_ylabel("mean predicted q(final action)")
        ax.set_title("Probe-predicted commitment before action mention")
        ax.legend(frameon=False, fontsize=8)
        path = figure_dir / "probe_predicted_commitment_before_mention.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Probe-predicted commitment before mention"})

    basin = metric_curve("mean_activation_basin_accuracy", source="activation")
    if not basin.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for subset, part in basin.groupby("strict_subset"):
            ax.plot(range(len(part)), part["value"], marker="o", label=str(subset))
        ax.set_xlabel("normalized progress bin")
        ax.set_ylabel("activation basin accuracy")
        ax.set_title("Activation basin accuracy before action mention")
        ax.legend(frameon=False, fontsize=8)
        path = figure_dir / "activation_basin_accuracy_before_mention.png"
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": "Activation basin accuracy before mention"})

    return figures


def write_report(
    path: Path,
    *,
    beliefs_path: Path,
    behavior_coords_path: Path,
    activation_coords_path: Path,
    probe_predictions_path: Path,
    prefix_trajectory_scores_path: Path,
    metrics_path: Path,
    prefix_scores_path: Path,
    metrics: pd.DataFrame,
    prefix_scores: pd.DataFrame,
    skipped: list[dict[str, str]],
    notes: list[str],
    figures: list[dict[str, str]],
) -> None:
    lines = [
        "# Strict Pre-action Conceptual Belief Analysis",
        "",
        "Step 9 tests whether behavior and activation commitment is visible before explicit action tokens appear in the generated CoT.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Beliefs: `{rel(beliefs_path)}`",
        f"- Behavior coordinates: `{rel(behavior_coords_path)}`",
        f"- Activation coordinates: `{rel(activation_coords_path)}`",
        f"- Probe predictions: `{rel(probe_predictions_path)}`",
        f"- Prefix trajectory scores: `{rel(prefix_trajectory_scores_path)}`",
        f"- Metrics CSV: `{rel(metrics_path)}`",
        f"- Prefix scores parquet: `{rel(prefix_scores_path)}`",
        "",
        "## Counts",
        "",
        f"- Prefix score rows: `{len(prefix_scores)}`",
        f"- Metric rows: `{len(metrics)}`",
        f"- Sources present: `{sorted(prefix_scores['source'].dropna().unique().tolist()) if not prefix_scores.empty else []}`",
    ]

    if not metrics.empty:
        behavior_commit = metrics.loc[
            (metrics["source"] == "behavior")
            & (metrics["metric"] == "mean_commitment_to_final")
        ]
        if not behavior_commit.empty:
            lines.extend(
                [
                    "",
                    "## Behavior Commitment",
                    "",
                    f"- Mean across strict bins: `{float(behavior_commit['value'].mean())}`",
                    f"- Max bin mean: `{float(behavior_commit['value'].max())}`",
                    f"- Min bin mean: `{float(behavior_commit['value'].min())}`",
                ]
            )
        activation_basin = metrics.loc[
            (metrics["source"] == "activation")
            & (metrics["metric"] == "mean_activation_basin_accuracy")
        ]
        if not activation_basin.empty:
            lines.extend(
                [
                    "",
                    "## Activation Basin Accuracy",
                    "",
                    f"- Mean across strict bins: `{float(activation_basin['value'].mean())}`",
                    f"- Max bin mean: `{float(activation_basin['value'].max())}`",
                ]
            )
        probe_commit = metrics.loc[
            (metrics["source"] == "probe")
            & (metrics["metric"] == "mean_probe_predicted_commitment_to_final")
        ]
        if not probe_commit.empty:
            lines.extend(
                [
                    "",
                    "## Probe-predicted Commitment",
                    "",
                    f"- Mean across strict bins: `{float(probe_commit['value'].mean())}`",
                    f"- Max bin mean: `{float(probe_commit['value'].max())}`",
                ]
            )
        probe_controls = metrics.loc[metrics["source"] == "probe_control_summary"]
        if not probe_controls.empty:
            lines.extend(["", "## Probe True-vs-shuffled Controls", ""])
            for row in probe_controls.head(12).itertuples(index=False):
                lines.append(
                    f"- `{row.control_type}` `{row.metric}`: `{float(row.value)}`"
                )

    lines.extend(["", "## Any-action vs Final-action Boundaries", ""])
    if not metrics.empty:
        boundary_summary = (
            metrics.loc[(metrics["source"] == "behavior") & (metrics["metric"] == "mean_commitment_to_final")]
            .groupby("boundary_type")["value"]
            .mean()
        )
        if not boundary_summary.empty:
            for boundary, value in boundary_summary.items():
                lines.append(f"- `{boundary}` mean behavior commitment: `{float(value)}`")
        else:
            lines.append("- Boundary comparison unavailable.")
    else:
        lines.append("- Boundary comparison unavailable.")

    lines.extend(["", "## Skipped Strict Subsets", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- `{item['subset']}`: {item['reason']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Optional Input Notes", ""])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- All optional inputs used successfully.")

    lines.extend(["", "## Figures", ""])
    if figures:
        for fig in figures:
            lines.append(f"- `{fig['path']}`: {fig['description']}")
    else:
        lines.append("- No figures generated.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Pre-action means no explicit lexical action token, not no implicit coordinate or path evidence.",
            "- Current activations are CoT-token activations, not necessarily final JSON decision-readout activations.",
            "- Observational curves do not prove causality.",
        ]
    )
    save_markdown(path, "\n".join(lines))


def run(
    *,
    beliefs_path: Path,
    behavior_coords_path: Path,
    activation_coords_path: Path,
    activation_centroids_path: Path,
    probe_predictions_path: Path,
    probe_metrics_path: Path,
    prefix_trajectory_scores_path: Path,
    metrics_output: Path,
    prefix_scores_output: Path,
    report_output: Path,
    figure_dir: Path,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beliefs = read_parquet_checked(beliefs_path)
    validate_beliefs(beliefs)
    notes: list[str] = []
    behavior_coords = optional_read_parquet(behavior_coords_path, "behavior coordinates", notes)
    if behavior_coords is not None:
        notes.append("Behavior coordinates were found but behavior pre-action curves use action belief values directly.")
    activation_coords = optional_read_parquet(activation_coords_path, "activation coordinates", notes)
    activation_centroids = optional_read_parquet(activation_centroids_path, "activation 2D centroids", notes)
    probe_predictions = optional_read_parquet(probe_predictions_path, "probe predictions", notes)
    probe_metrics = optional_read_table(probe_metrics_path, "probe metrics", notes)
    prefix_trajectory_scores = optional_read_parquet(prefix_trajectory_scores_path, "prefix trajectory scores", notes)

    base, skipped = strict_base_rows(beliefs, n_bins=n_bins)
    if base.empty:
        raise SystemExit("No strict pre-action rows were available for analysis.")

    parts = [behavior_prefix_scores(base)]
    activation = activation_scores_from_coords(
        base,
        activation_coords=activation_coords,
        activation_centroids=activation_centroids,
        prefix_trajectory_scores=prefix_trajectory_scores,
        notes=notes,
    )
    if not activation.empty:
        parts.append(activation)
    probes = probe_scores(base, probe_predictions, notes)
    if not probes.empty:
        parts.append(probes)

    prefix_scores = pd.concat(parts, ignore_index=True, sort=False)
    metrics = aggregate_metrics(prefix_scores)
    controls = probe_control_summary(probe_metrics)
    if not controls.empty:
        metrics = pd.concat([metrics, controls], ignore_index=True, sort=False)

    ensure_dir(prefix_scores_output.parent)
    prefix_scores.to_parquet(prefix_scores_output, index=False)
    ensure_dir(metrics_output.parent)
    metrics.to_csv(metrics_output, index=False)
    figures = make_figures(metrics, figure_dir)
    write_report(
        report_output,
        beliefs_path=beliefs_path,
        behavior_coords_path=behavior_coords_path,
        activation_coords_path=activation_coords_path,
        probe_predictions_path=probe_predictions_path,
        prefix_trajectory_scores_path=prefix_trajectory_scores_path,
        metrics_path=metrics_output,
        prefix_scores_path=prefix_scores_output,
        metrics=metrics,
        prefix_scores=prefix_scores,
        skipped=skipped,
        notes=notes,
        figures=figures,
    )
    return metrics, prefix_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--behavior_coords", default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_coords.parquet")
    parser.add_argument("--activation_coords", default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_coords.parquet")
    parser.add_argument(
        "--activation_centroids_2d",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_2d.parquet",
    )
    parser.add_argument("--probe_predictions", default=DEFAULT_OUTPUT_DIR / "probes/probe_predictions.parquet")
    parser.add_argument("--probe_metrics", default=DEFAULT_OUTPUT_DIR / "probes/probe_metrics.csv")
    parser.add_argument(
        "--prefix_trajectory_scores",
        default=DEFAULT_OUTPUT_DIR / "trajectories/prefix_trajectory_scores.parquet",
    )
    parser.add_argument(
        "--metrics_output",
        default=DEFAULT_OUTPUT_DIR / "strict_pre_action/strict_pre_action_metrics.csv",
    )
    parser.add_argument(
        "--prefix_scores_output",
        default=DEFAULT_OUTPUT_DIR / "strict_pre_action/strict_pre_action_prefix_scores.parquet",
    )
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "strict_pre_action/09_strict_pre_action_report.md")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "strict_pre_action")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    metrics, prefix_scores = run(
        beliefs_path=resolve_repo_path(args.beliefs),
        behavior_coords_path=resolve_repo_path(args.behavior_coords),
        activation_coords_path=resolve_repo_path(args.activation_coords),
        activation_centroids_path=resolve_repo_path(args.activation_centroids_2d),
        probe_predictions_path=resolve_repo_path(args.probe_predictions),
        probe_metrics_path=resolve_repo_path(args.probe_metrics),
        prefix_trajectory_scores_path=resolve_repo_path(args.prefix_trajectory_scores),
        metrics_output=resolve_repo_path(args.metrics_output),
        prefix_scores_output=resolve_repo_path(args.prefix_scores_output),
        report_output=resolve_repo_path(args.report),
        figure_dir=resolve_repo_path(args.figure_dir),
        n_bins=args.n_bins,
    )
    print(f"Wrote {rel(args.metrics_output)} with {len(metrics)} rows")
    print(f"Wrote {rel(args.prefix_scores_output)} with {len(prefix_scores)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
