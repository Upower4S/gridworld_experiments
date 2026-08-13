#!/usr/bin/env python3
"""Step 10: robustness controls and ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import (  # noqa: E402
    ACTIONS,
    DEFAULT_FIGURE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
)
from src.conceptual_belief_space.controls import (  # noqa: E402
    action_balanced_subset,
    assign_cot_length_bins,
    behavior_metric_summary,
    extreme_ablation_counts,
    q_columns,
    random_prefix_split_indices,
    shuffle_agent_action_across_decisions,
    shuffle_q_vectors_across_decisions,
    split_leakage_counts,
)
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.plotting import (  # noqa: E402
    metric_heatmap_from_table,
    savefig,
    simple_bar,
    simple_lines,
)
from src.conceptual_belief_space.splits import grouped_train_test_splits  # noqa: E402


STRICT_SUBSETS = [
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "buffered_before_any_action_mention",
    "buffered_before_final_action_mention",
]
METRIC_COLUMNS = [
    "mean_commitment_to_final",
    "mean_entropy_commitment",
    "mean_margin_commitment",
    "top_action_accuracy",
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


def metric_row(
    *,
    control_family: str,
    control_name: str,
    metric: str,
    value: float | int | bool | None,
    subset: str = "all_prefixes",
    status: str = "ok",
    note: str = "",
    split_id: int | float | None = np.nan,
    layer: int | float | None = np.nan,
    representation: str = "",
    target: str = "",
    model: str = "",
    n_rows: int | float | None = np.nan,
    n_decisions: int | float | None = np.nan,
) -> dict[str, Any]:
    if isinstance(value, bool):
        numeric_value = float(value)
        text_value = str(value)
    elif value is None:
        numeric_value = np.nan
        text_value = ""
    else:
        try:
            numeric_value = float(value)
            text_value = ""
        except Exception:
            numeric_value = np.nan
            text_value = str(value)
    return {
        "control_family": control_family,
        "control_name": control_name,
        "subset": subset,
        "split_id": split_id,
        "layer": layer,
        "representation": representation,
        "target": target,
        "model": model,
        "metric": metric,
        "value": numeric_value,
        "text_value": text_value,
        "n_rows": n_rows,
        "n_decisions": n_decisions,
        "status": status,
        "note": note,
    }


def append_behavior_summary(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    control_family: str,
    control_name: str,
    subset: str,
    q_cols: list[str] | None = None,
    action_col: str = "agent_action",
    top_action_col: str | None = None,
    status: str = "ok",
    note: str = "",
) -> None:
    q_cols = q_cols or q_columns()
    summary = behavior_metric_summary(df, q_cols=q_cols, action_col=action_col, top_action_col=top_action_col)
    for metric in METRIC_COLUMNS:
        rows.append(
            metric_row(
                control_family=control_family,
                control_name=control_name,
                subset=subset,
                metric=metric,
                value=summary[metric],
                status=status,
                note=note,
                n_rows=summary["n_rows"],
                n_decisions=summary["n_decisions"],
            )
        )


def subset_frames(beliefs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {"all_prefixes": beliefs.copy()}
    for subset in STRICT_SUBSETS:
        if subset in beliefs.columns:
            frames[subset] = beliefs.loc[beliefs[subset].fillna(False).astype(bool)].copy()
    return frames


def compute_belief_level_controls(beliefs: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = subset_frames(beliefs)
    shuffled_labels = shuffle_agent_action_across_decisions(beliefs, seed=seed)
    shuffled_q = shuffle_q_vectors_across_decisions(beliefs, seed=seed + 1)
    shuffled_q_cols = [f"shuffled_q_{action}" for action in ACTIONS]

    for subset, part in frames.items():
        append_behavior_summary(
            rows,
            part,
            control_family="behavior_baseline",
            control_name="original",
            subset=subset,
            top_action_col="top_action" if "top_action" in part.columns else None,
        )
        label_part = shuffled_labels.loc[part.index].copy()
        append_behavior_summary(
            rows,
            label_part,
            control_family="shuffled_labels",
            control_name="agent_action_shuffled_across_decision_id",
            subset=subset,
            action_col="agent_action_shuffled",
            top_action_col="top_action" if "top_action" in label_part.columns else None,
            note="Final-action labels are shuffled at decision level and broadcast to all prefixes.",
        )
        q_part = shuffled_q.loc[part.index].copy()
        append_behavior_summary(
            rows,
            q_part,
            control_family="shuffled_behavior_q",
            control_name="q_trajectory_shuffled_across_decision_id",
            subset=subset,
            q_cols=shuffled_q_cols,
            note="Behavior q trajectories are shuffled across decision ids, with prefixes matched by nearest progress.",
        )

    balanced = action_balanced_subset(beliefs, seed=seed)
    append_behavior_summary(
        rows,
        beliefs,
        control_family="action_balanced_subsampling",
        control_name="all_decisions",
        subset="all_prefixes",
    )
    append_behavior_summary(
        rows,
        balanced,
        control_family="action_balanced_subsampling",
        control_name="balanced_decisions",
        subset="all_prefixes",
        note="Equal number of decision ids sampled per final action.",
    )

    with_bins = assign_cot_length_bins(beliefs)
    for cot_bin, part in with_bins.groupby("cot_length_bin", dropna=False):
        append_behavior_summary(
            rows,
            part,
            control_family="cot_length_stratification",
            control_name=str(cot_bin),
            subset="all_prefixes",
        )

    return pd.DataFrame(rows)


def load_splits(path: Path, notes: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        notes.append(f"Optional input unavailable: splits at `{rel(path)}`")
        return []
    try:
        obj = read_json(path)
        if not isinstance(obj, list):
            notes.append(f"Splits file `{rel(path)}` did not contain a list.")
            return []
        return [item for item in obj if isinstance(item, dict)]
    except Exception as exc:
        notes.append(f"Splits file failed to read: `{rel(path)}`: {type(exc).__name__}: {exc}")
        return []


def compute_split_sensitivity(beliefs: pd.DataFrame, splits: list[dict[str, Any]], *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        split_id = split.get("fold", split.get("split_id", np.nan))
        group_col = str(split.get("group_col", "trajectory_id"))
        if group_col not in beliefs.columns:
            continue
        counts = split_leakage_counts(
            beliefs,
            train_indices=split.get("train_indices", []),
            test_indices=split.get("test_indices", []),
            group_col=group_col,
        )
        for metric, value in counts.items():
            rows.append(
                metric_row(
                    control_family="split_sensitivity",
                    control_name=f"grouped_by_{group_col}",
                    split_id=split_id,
                    metric=metric,
                    value=value,
                    status="ok",
                    note="Grouped split loaded from group_splits.json.",
                )
            )

    for group_col in ["trajectory_id", "decision_id"]:
        if group_col not in beliefs.columns or beliefs[group_col].nunique() < 2:
            continue
        n_splits = min(5, int(beliefs[group_col].nunique()))
        if n_splits < 2:
            continue
        try:
            derived = grouped_train_test_splits(beliefs, group_col=group_col, n_splits=n_splits, seed=seed)
        except Exception:
            continue
        for split in derived:
            counts = split_leakage_counts(
                beliefs,
                train_indices=split["train_indices"],
                test_indices=split["test_indices"],
                group_col=group_col,
            )
            for metric, value in counts.items():
                rows.append(
                    metric_row(
                        control_family="split_sensitivity",
                        control_name=f"derived_grouped_by_{group_col}",
                        split_id=split["fold"],
                        metric=metric,
                        value=value,
                        status="ok",
                        note="Derived grouped split for sensitivity; no model refit is performed here.",
                    )
                )

    train, test = random_prefix_split_indices(len(beliefs), seed=seed)
    for group_col in ["trajectory_id", "decision_id"]:
        if group_col not in beliefs.columns:
            continue
        counts = split_leakage_counts(beliefs, train_indices=train, test_indices=test, group_col=group_col)
        for metric, value in counts.items():
            rows.append(
                metric_row(
                    control_family="split_sensitivity",
                    control_name="random_prefix_split_invalid_leakage_diagnostic",
                    metric=metric,
                    value=value,
                    status="invalid_for_main_claims",
                    note="Random prefix split intentionally leaks correlated prefixes; included only as a leakage diagnostic.",
                )
            )
    return pd.DataFrame(rows)


def summarize_probe_metrics(probe_metrics: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    if probe_metrics is None or probe_metrics.empty:
        rows.append(
            metric_row(
                control_family="probe_metrics",
                control_name="unavailable",
                metric="status",
                value=None,
                status="missing_input",
                note="Probe metrics are missing; Step 7 must run before probe controls can be summarized.",
            )
        )
        return pd.DataFrame(rows), notes

    candidate_metrics = [
        "mean_r2",
        "r2",
        "mse",
        "pearson",
        "balanced_accuracy",
        "macro_f1",
        "top_action_accuracy_vs_true_top_action",
        "top_action_accuracy_vs_agent_action",
        "predicted_commitment_pearson",
        "kl_divergence",
    ]
    metric_cols = [col for col in candidate_metrics if col in probe_metrics.columns]
    if not metric_cols:
        notes.append("Probe metrics table was present but no recognized metric columns were found.")
        return pd.DataFrame(rows), notes

    context_cols = {
        "subset": "all_prefixes",
        "split_id": np.nan,
        "layer": np.nan,
        "activation_representation": "",
        "target": "",
        "model": "",
        "control_type": "unknown",
    }
    for _, row in probe_metrics.iterrows():
        for metric in metric_cols:
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            rows.append(
                metric_row(
                    control_family="probe_metrics",
                    control_name=str(row.get("control_type", context_cols["control_type"])),
                    subset=str(row.get("subset", context_cols["subset"])),
                    split_id=row.get("split_id", context_cols["split_id"]),
                    layer=row.get("layer", context_cols["layer"]),
                    representation=str(row.get("activation_representation", context_cols["activation_representation"])),
                    target=str(row.get("target", context_cols["target"])),
                    model=str(row.get("model", context_cols["model"])),
                    metric=metric,
                    value=float(value),
                    status="ok",
                    note="Summarized from Step 7 probe_metrics.csv; negative R2 is preserved.",
                )
            )

    has_true = "control_type" in probe_metrics.columns and (probe_metrics["control_type"].astype(str) == "true").any()
    has_shuffled = "control_type" in probe_metrics.columns and (probe_metrics["control_type"].astype(str) != "true").any()
    if not has_true or not has_shuffled:
        rows.append(
            metric_row(
                control_family="shuffled_labels",
                control_name="probe_true_vs_shuffled",
                metric="status",
                value=None,
                status="not_available",
                note="Probe table lacks true or shuffled-control rows.",
            )
        )
    return pd.DataFrame(rows), notes


def summarize_geometry_metrics(geometry_metrics: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    if geometry_metrics is None or geometry_metrics.empty:
        rows.append(
            metric_row(
                control_family="geometry_alignment",
                control_name="unavailable",
                metric="status",
                value=None,
                status="missing_input",
                note="Geometry metrics are missing; Step 6 must run before centroid alignment controls can be summarized.",
            )
        )
        return pd.DataFrame(rows), notes

    metric_cols = [
        col
        for col in [
            "pearson",
            "spearman",
            "kendall_tau",
            "permutation_spearman_observed",
            "permutation_p_value",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
        if col in geometry_metrics.columns
    ]
    for _, row in geometry_metrics.iterrows():
        control_name = "primary" if bool(row.get("is_primary", False)) else "secondary"
        for metric in metric_cols:
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            rows.append(
                metric_row(
                    control_family="geometry_alignment",
                    control_name=control_name,
                    subset=str(row.get("subset", "all_prefixes")),
                    split_id=row.get("split_id", np.nan),
                    layer=row.get("activation_layer", np.nan),
                    representation=str(row.get("activation_representation", "")),
                    metric=metric,
                    value=float(value),
                    status="ok",
                    note="Summarized from Step 6 distance_matrix_metrics.csv.",
                )
            )
    return pd.DataFrame(rows), notes


def summarize_representation_ablation(probe_metrics: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_representations = ["raw", "delta_pre", "unit_delta_pre", "velocity"]
    if probe_metrics is None or probe_metrics.empty:
        for representation in expected_representations:
            rows.append(
                metric_row(
                    control_family="representation_ablation",
                    control_name=representation,
                    representation=representation,
                    metric="status",
                    value=None,
                    status="missing_probe_metrics",
                    note="Requires Step 7 probe metrics to compare activation representations.",
                )
            )
        return pd.DataFrame(rows)

    if "activation_representation" not in probe_metrics.columns:
        rows.append(
            metric_row(
                control_family="representation_ablation",
                control_name="unavailable",
                metric="status",
                value=None,
                status="missing_activation_representation_column",
                note="Probe metrics are present but do not expose activation_representation.",
            )
        )
        return pd.DataFrame(rows)

    metric_cols = [col for col in ["mean_r2", "r2", "balanced_accuracy", "top_action_accuracy_vs_true_top_action"] if col in probe_metrics.columns]
    true = probe_metrics.copy()
    if "control_type" in true.columns:
        true = true.loc[true["control_type"].astype(str) == "true"].copy()
    for representation in expected_representations:
        part = true.loc[true["activation_representation"].astype(str) == representation]
        if part.empty:
            rows.append(
                metric_row(
                    control_family="representation_ablation",
                    control_name=representation,
                    representation=representation,
                    metric="status",
                    value=None,
                    status="representation_missing",
                    note="No true probe metric rows found for this representation.",
                )
            )
            continue
        for metric in metric_cols:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                metric_row(
                    control_family="representation_ablation",
                    control_name=representation,
                    representation=representation,
                    metric=f"mean_{metric}",
                    value=float(values.mean()),
                    status="ok",
                    note="Mean over available true probe metric rows for this representation.",
                )
            )
    return pd.DataFrame(rows)


def compute_extreme_ablation_metrics(
    beliefs: pd.DataFrame,
    splits: list[dict[str, Any]],
) -> pd.DataFrame:
    train_indices = splits[0].get("train_indices") if splits else None
    subset_names = ["all_prefixes", *[subset for subset in STRICT_SUBSETS if subset in beliefs.columns]]
    counts = extreme_ablation_counts(
        beliefs,
        train_indices=train_indices,
        thresholds=(0.8, 0.9, 0.95),
        top_ks=(100, 250, 500),
        max_per_decisions=(1, 3, 5),
        subset_names=subset_names,
    )
    rows: list[dict[str, Any]] = []
    for _, row in counts.iterrows():
        control_name = f"threshold_{row['threshold']}_topk_{row['top_k']}_maxperdecision_{row['max_per_decision']}"
        for metric in ["balanced_count_per_action", "total_selected_balanced", *[f"selected_{a}" for a in ACTIONS]]:
            rows.append(
                metric_row(
                    control_family="extreme_selection_ablation",
                    control_name=control_name,
                    subset=str(row["subset"]),
                    metric=metric,
                    value=row.get(metric),
                    status=str(row["status"]),
                    note="Counts train-only extreme-example availability under alternative selection hyperparameters.",
                )
            )
    return pd.DataFrame(rows)


def add_unavailable_heavy_controls(rows: list[pd.DataFrame], probe_metrics: pd.DataFrame | None, geometry_metrics: pd.DataFrame | None) -> None:
    unavailable: list[dict[str, Any]] = []
    if probe_metrics is None:
        unavailable.append(
            metric_row(
                control_family="progress_residualization",
                control_name="activation_feature_residualized_probe_rerun",
                metric="status",
                value=None,
                status="not_run_missing_probe_artifacts",
                note="Requires activation feature reruns and Step 7-style probe fitting; not recomputed from missing probe artifacts.",
            )
        )
    else:
        residual_rows = probe_metrics[
            probe_metrics.astype(str).apply(lambda col: col.str.contains("residual", case=False, regex=False)).any(axis=1)
        ]
        if residual_rows.empty:
            unavailable.append(
                metric_row(
                    control_family="progress_residualization",
                    control_name="activation_feature_residualized_probe_rerun",
                    metric="status",
                    value=None,
                    status="not_available_in_probe_metrics",
                    note="No progress-residualized probe rows were found in Step 7 metrics.",
                )
            )
    if geometry_metrics is None:
        unavailable.append(
            metric_row(
                control_family="progress_residualization",
                control_name="activation_feature_residualized_centroid_rerun",
                metric="status",
                value=None,
                status="not_run_missing_geometry_artifacts",
                note="Requires Step 5/6 reruns on residualized activation features.",
            )
        )
    rows.append(pd.DataFrame(unavailable))


def make_figures(metrics: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []

    shuffled = metrics.loc[
        (metrics["metric"] == "top_action_accuracy")
        & (metrics["subset"] == "all_prefixes")
        & (metrics["control_family"].isin(["behavior_baseline", "shuffled_labels", "shuffled_behavior_q"]))
    ].copy()
    if not shuffled.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        simple_bar(
            shuffled["control_name"],
            shuffled["value"],
            ax=ax,
            title="True vs shuffled belief-level action accuracy",
            ylabel="top-action accuracy",
            rotation=25,
        )
        out = figure_dir / "true_vs_shuffled_probe_metrics.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "True vs shuffled belief-level metrics; probe controls appear here when available."})

    geometry = metrics.loc[
        (metrics["control_family"] == "geometry_alignment")
        & (metrics["metric"].isin(["spearman", "permutation_spearman_observed"]))
        & (metrics["status"] == "ok")
    ].copy()
    if not geometry.empty:
        grouped = (
            geometry.groupby(["control_name", "layer"], dropna=False)["value"]
            .mean()
            .reset_index()
            .sort_values("layer")
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        simple_lines(
            grouped,
            x_col="layer",
            y_col="value",
            group_col="control_name",
            ax=ax,
            title="True vs shuffled centroid distance correlations",
            ylabel="mean distance correlation",
        )
        out = figure_dir / "true_vs_shuffled_centroid_distance_correlations.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Centroid distance correlations and permutation-control summaries."})

    residual = metrics.loc[
        (metrics["control_family"] == "progress_residualization")
        & (metrics["metric"].isin(["mean_r2", "r2", "spearman"]))
        & (metrics["status"] == "ok")
    ].copy()
    if not residual.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        simple_lines(
            residual,
            x_col="layer",
            y_col="value",
            group_col="control_name",
            ax=ax,
            title="Original vs progress-residualized metrics",
            ylabel="metric value",
        )
        out = figure_dir / "original_vs_progress_residualized_metrics.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Original vs progress-residualized metrics."})

    balanced = metrics.loc[
        (metrics["control_family"] == "action_balanced_subsampling")
        & (metrics["metric"].isin(["mean_commitment_to_final", "top_action_accuracy"]))
    ].copy()
    if not balanced.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = balanced["control_name"] + " / " + balanced["metric"]
        simple_bar(
            labels,
            balanced["value"],
            ax=ax,
            title="Action-balanced vs all metrics",
            ylabel="metric value",
            rotation=35,
        )
        out = figure_dir / "action_balanced_vs_all_metrics.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Action-balanced vs all metrics."})

    cot = metrics.loc[
        (metrics["control_family"] == "cot_length_stratification")
        & (metrics["metric"] == "mean_commitment_to_final")
    ].copy()
    if not cot.empty:
        order = {"short": 0, "medium": 1, "long": 2, "unknown": 3}
        cot["order"] = cot["control_name"].map(order).fillna(99)
        cot = cot.sort_values("order")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        simple_bar(
            cot["control_name"],
            cot["value"],
            ax=ax,
            title="CoT-length stratified commitment",
            ylabel="mean commitment_to_final",
            rotation=0,
        )
        out = figure_dir / "cot_length_stratified_metrics.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "CoT-length stratified metrics."})

    probe = metrics.loc[
        (metrics["control_family"] == "probe_metrics")
        & (metrics["control_name"] == "true")
        & (metrics["metric"].isin(["mean_r2", "balanced_accuracy"]))
        & (metrics["status"] == "ok")
    ].copy()
    if not probe.empty and {"representation", "layer", "value"}.issubset(probe.columns):
        heatmap_data = probe.dropna(subset=["layer"])
        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            metric_heatmap_from_table(
                heatmap_data,
                row_col="representation",
                col_col="layer",
                value_col="value",
                ax=ax,
                title="Representation-by-layer probe metric heatmap",
            )
            out = figure_dir / "representation_by_layer_heatmap.png"
            savefig(out, fig=fig)
            figures.append({"path": rel(out), "description": "Representation-by-layer heatmap."})

    return figures


def summarize_robustness(metrics: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    original = metrics.loc[
        (metrics["control_family"] == "behavior_baseline")
        & (metrics["control_name"] == "original")
        & (metrics["subset"] == "all_prefixes")
        & (metrics["metric"] == "top_action_accuracy")
    ]
    shuffled = metrics.loc[
        (metrics["control_family"] == "shuffled_labels")
        & (metrics["subset"] == "all_prefixes")
        & (metrics["metric"] == "top_action_accuracy")
    ]
    if not original.empty and not shuffled.empty:
        delta = float(original["value"].mean() - shuffled["value"].mean())
        if delta > 0.05:
            lines.append(f"- Behavior top-action accuracy is above decision-label shuffle by `{delta}`.")
        else:
            lines.append(f"- Behavior top-action accuracy is weak relative to decision-label shuffle; delta `{delta}`.")

    probe = metrics.loc[
        (metrics["control_family"] == "probe_metrics")
        & (metrics["metric"].isin(["mean_r2", "r2", "balanced_accuracy"]))
        & (metrics["status"] == "ok")
    ]
    if probe.empty:
        lines.append("- Probe robustness cannot be assessed yet because probe metrics are unavailable.")
    else:
        true = probe.loc[probe["control_name"] == "true"]
        control = probe.loc[probe["control_name"] != "true"]
        if not true.empty:
            min_r2 = true.loc[true["metric"].isin(["mean_r2", "r2"]), "value"].min()
            if np.isfinite(min_r2):
                lines.append(f"- Probe metrics include negative R2 honestly; minimum true R2 is `{float(min_r2)}`.")
        if not true.empty and not control.empty:
            lines.append("- Probe true-vs-shuffled controls are present; inspect `control_metrics.csv` by target/layer.")
        else:
            lines.append("- Probe shuffled controls are missing or incomplete.")

    geometry = metrics.loc[(metrics["control_family"] == "geometry_alignment") & (metrics["status"] == "ok")]
    if geometry.empty:
        lines.append("- Centroid distance alignment controls cannot be assessed yet because geometry metrics are unavailable.")
    else:
        primary = geometry.loc[geometry["control_name"] == "primary"]
        if not primary.empty:
            lines.append("- Geometry alignment metrics are present; permutation p-values should be interpreted with the four-action/six-distance caveat.")

    residual_status = metrics.loc[metrics["control_family"] == "progress_residualization"]
    if residual_status.empty or (residual_status["status"].astype(str) != "ok").all():
        lines.append("- Progress residualization is not available from current artifacts; it requires activation-feature reruns.")
    return lines


def write_report(
    path: Path,
    *,
    beliefs_path: Path,
    splits_path: Path,
    probe_metrics_path: Path,
    geometry_metrics_path: Path,
    output_path: Path,
    metrics: pd.DataFrame,
    notes: list[str],
    figures: list[dict[str, str]],
) -> None:
    lines = [
        "# Controls and Ablations",
        "",
        "Step 10 summarizes robustness controls and ablations for the conceptual belief space pipeline.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Beliefs: `{rel(beliefs_path)}`",
        f"- Splits: `{rel(splits_path)}`",
        f"- Probe metrics: `{rel(probe_metrics_path)}`",
        f"- Geometry metrics: `{rel(geometry_metrics_path)}`",
        f"- Control metrics: `{rel(output_path)}`",
        "",
        "## Counts",
        "",
        f"- Metric rows: `{len(metrics)}`",
        f"- Control families: `{sorted(metrics['control_family'].dropna().unique().tolist()) if not metrics.empty else []}`",
        "",
        "## Robustness Summary",
        "",
        *summarize_robustness(metrics),
        "",
        "## Availability Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- All requested optional metric inputs were available.")

    unavailable = metrics.loc[metrics["status"].astype(str).str.contains("missing|not_available|not_run", regex=True)]
    if not unavailable.empty:
        lines.extend(["", "## Weak or Not Yet Robust", ""])
        for row in unavailable.head(40).itertuples(index=False):
            lines.append(f"- `{row.control_family}` / `{row.control_name}`: {row.status}. {row.note}")
        if len(unavailable) > 40:
            lines.append(f"- ... {len(unavailable) - 40} additional unavailable rows omitted.")

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Negative R2 values are retained and should be read as worse than predicting the test-set mean.",
            "- Random prefix splits are included only as an invalid leakage diagnostic and must not support main claims.",
            "- Shuffled controls and action-balanced summaries are robustness diagnostics, not causal evidence.",
            "- Do not overclaim if true metrics are close to shuffled controls, unstable across representations, or unavailable after residualization.",
            "",
            "## Figures",
            "",
        ]
    )
    if figures:
        lines.extend(f"- `{fig['path']}`: {fig['description']}" for fig in figures)
    else:
        lines.append("- No figures generated.")
    save_markdown(path, "\n".join(lines))


def run(
    *,
    beliefs_path: Path,
    splits_path: Path,
    probe_metrics_path: Path,
    geometry_metrics_path: Path,
    output_path: Path,
    report_path: Path,
    figure_dir: Path,
    seed: int,
) -> pd.DataFrame:
    beliefs = read_parquet_checked(
        beliefs_path,
        required_columns=["decision_id", "trajectory_id", "agent_action", *q_columns()],
    ).reset_index(drop=True)
    notes: list[str] = []
    splits = load_splits(splits_path, notes)
    probe_metrics = optional_read_table(probe_metrics_path, "probe metrics", notes)
    geometry_metrics = optional_read_table(geometry_metrics_path, "geometry metrics", notes)

    parts: list[pd.DataFrame] = [
        compute_belief_level_controls(beliefs, seed=seed),
        compute_split_sensitivity(beliefs, splits, seed=seed),
        compute_extreme_ablation_metrics(beliefs, splits),
    ]
    probe_summary, probe_notes = summarize_probe_metrics(probe_metrics)
    geometry_summary, geometry_notes = summarize_geometry_metrics(geometry_metrics)
    notes.extend(probe_notes)
    notes.extend(geometry_notes)
    parts.extend([probe_summary, geometry_summary, summarize_representation_ablation(probe_metrics)])
    add_unavailable_heavy_controls(parts, probe_metrics, geometry_metrics)

    metrics = pd.concat([part for part in parts if part is not None and not part.empty], ignore_index=True, sort=False)
    ensure_dir(output_path.parent)
    metrics.to_csv(output_path, index=False)
    figures = make_figures(metrics, figure_dir)
    write_report(
        report_path,
        beliefs_path=beliefs_path,
        splits_path=splits_path,
        probe_metrics_path=probe_metrics_path,
        geometry_metrics_path=geometry_metrics_path,
        output_path=output_path,
        metrics=metrics,
        notes=notes,
        figures=figures,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--probe_metrics", default=DEFAULT_OUTPUT_DIR / "probes/probe_metrics.csv")
    parser.add_argument("--geometry_metrics", default=DEFAULT_OUTPUT_DIR / "geometry_alignment/distance_matrix_metrics.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR / "controls/control_metrics.csv")
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "controls/10_controls_report.md")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "controls")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    metrics = run(
        beliefs_path=resolve_repo_path(args.beliefs),
        splits_path=resolve_repo_path(args.splits),
        probe_metrics_path=resolve_repo_path(args.probe_metrics),
        geometry_metrics_path=resolve_repo_path(args.geometry_metrics),
        output_path=resolve_repo_path(args.output),
        report_path=resolve_repo_path(args.report),
        figure_dir=resolve_repo_path(args.figure_dir),
        seed=args.seed,
    )
    print(f"Wrote {rel(args.output)} with {len(metrics)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
