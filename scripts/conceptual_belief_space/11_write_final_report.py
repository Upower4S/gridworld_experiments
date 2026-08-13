#!/usr/bin/env python3
"""Step 11: write the final conceptual belief space report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, save_markdown  # noqa: E402


REPORT_INPUTS = [
    ("IO audit", DEFAULT_OUTPUT_DIR / "01_io_audit.md"),
    ("Action belief vectors", DEFAULT_OUTPUT_DIR / "02_action_belief_vectors_report.md"),
    ("Train-only extreme examples", DEFAULT_OUTPUT_DIR / "03_extreme_examples_report.md"),
    ("Behavior conceptual manifold", DEFAULT_OUTPUT_DIR / "04_behavior_manifold_report.md"),
    ("Activation conceptual manifold", DEFAULT_OUTPUT_DIR / "05_activation_manifold_report.md"),
    ("Centroid distance matrix alignment", DEFAULT_OUTPUT_DIR / "geometry_alignment/06_geometry_alignment_report.md"),
    ("Sparse probes", DEFAULT_OUTPUT_DIR / "probes/07_probe_report.md"),
    ("Trajectory analysis", DEFAULT_OUTPUT_DIR / "trajectories/08_trajectory_report.md"),
    ("Strict pre-action analysis", DEFAULT_OUTPUT_DIR / "strict_pre_action/09_strict_pre_action_report.md"),
    ("Robustness controls", DEFAULT_OUTPUT_DIR / "controls/10_controls_report.md"),
    ("Action-centroid contrast projection", DEFAULT_OUTPUT_DIR / "action_centroid_contrast/12_action_centroid_contrast_report.md"),
]
SECTION_ORDER = [
    "Motivation: Why Blind PCA Was Insufficient",
    "Dataset and Inputs",
    "Behavior Action-belief Construction",
    "Train-only Extreme Examples",
    "Behavior Conceptual Manifold",
    "Activation Conceptual Manifold",
    "Action-centroid Contrast Projection",
    "Centroid Distance Matrix Alignment",
    "Sparse Probes",
    "Trajectory Analysis",
    "Strict Pre-action Analysis",
    "Robustness Controls",
    "What Changed Relative to the Previous PCA Analysis",
    "Interpretation and Limitations",
    "Next Experiments",
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


def markdown_rel_from_report(path: str | Path, report_path: Path) -> str:
    path = Path(path).resolve()
    return os.path.relpath(path, start=report_path.parent.resolve())


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text()
    except Exception as exc:
        return f"Could not read `{rel(path)}`: {type(exc).__name__}: {exc}"


def optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def compact_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def report_availability() -> pd.DataFrame:
    rows = []
    for label, path in REPORT_INPUTS:
        full = resolve_repo_path(path)
        rows.append(
            {
                "section": label,
                "path": rel(full),
                "available": full.exists(),
                "size_bytes": int(full.stat().st_size) if full.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    part = df.copy()
    if columns is not None:
        part = part[[col for col in columns if col in part.columns]]
    part = part.head(max_rows)
    return part.to_markdown(index=False)


def action_belief_summary(output_dir: Path) -> list[str]:
    path = output_dir / "action_belief_vectors.parquet"
    df = optional_table(path)
    if df is None or df.empty:
        return ["- Action belief vector table is unavailable."]
    lines = [
        f"- Prefix rows: `{len(df):,}`",
        f"- Decision steps: `{df['decision_id'].nunique():,}`" if "decision_id" in df.columns else "- Decision steps: `NA`",
        f"- Trajectories: `{df['trajectory_id'].nunique():,}`" if "trajectory_id" in df.columns else "- Trajectories: `NA`",
    ]
    if "commitment_to_final" in df.columns:
        lines.append(f"- Mean commitment_to_final: `{compact_number(df['commitment_to_final'].mean())}`")
    if "top_action" in df.columns and "agent_action" in df.columns:
        lines.append(f"- Top-action equals final action: `{compact_number((df['top_action'].astype(str) == df['agent_action'].astype(str)).mean())}`")
    if "agent_action" in df.columns:
        counts = df.drop_duplicates("decision_id")["agent_action"].value_counts().sort_index() if "decision_id" in df.columns else df["agent_action"].value_counts().sort_index()
        lines.append("- Final-action decision counts: " + ", ".join(f"`{k}`={int(v)}" for k, v in counts.items()))
    return lines


def extremes_summary(output_dir: Path) -> list[str]:
    path = output_dir / "extreme_examples/action_extreme_examples.parquet"
    df = optional_table(path)
    if df is None or df.empty:
        return ["- Extreme-example table is unavailable."]
    lines = [
        f"- Extreme rows: `{len(df):,}`",
        f"- Splits represented: `{df['split_id'].nunique():,}`" if "split_id" in df.columns else "- Splits represented: `NA`",
        f"- Subsets represented: `{df['subset'].nunique():,}`" if "subset" in df.columns else "- Subsets represented: `NA`",
    ]
    if {"action", "subset"}.issubset(df.columns):
        counts = df.groupby(["subset", "action"]).size().rename("n").reset_index().head(16)
        lines.extend(["", markdown_table(counts)])
    return lines


def strict_summary(output_dir: Path) -> list[str]:
    metrics = optional_table(output_dir / "strict_pre_action/strict_pre_action_metrics.csv")
    if metrics is None or metrics.empty:
        return ["- Strict pre-action metrics are unavailable."]
    lines = [f"- Strict pre-action metric rows: `{len(metrics):,}`"]
    behavior = metrics.loc[(metrics.get("source") == "behavior") & (metrics.get("metric") == "mean_commitment_to_final")] if "source" in metrics.columns else pd.DataFrame()
    if not behavior.empty:
        lines.append(f"- Mean behavior commitment across strict bins: `{compact_number(behavior['value'].mean())}`")
        boundary = behavior.groupby("boundary_type")["value"].mean().reset_index()
        lines.append("")
        lines.append(markdown_table(boundary))
    return lines


def controls_summary(output_dir: Path) -> list[str]:
    metrics = optional_table(output_dir / "controls/control_metrics.csv")
    if metrics is None or metrics.empty:
        return ["- Control metrics are unavailable."]
    lines = [
        f"- Control metric rows: `{len(metrics):,}`",
        "- Control families: " + ", ".join(f"`{name}`" for name in sorted(metrics["control_family"].dropna().unique())),
    ]
    key = metrics.loc[
        (metrics["subset"] == "all_prefixes")
        & (metrics["metric"] == "top_action_accuracy")
        & (metrics["control_family"].isin(["behavior_baseline", "shuffled_labels", "shuffled_behavior_q"]))
    ][["control_family", "control_name", "value", "status"]]
    if not key.empty:
        lines.extend(["", markdown_table(key)])
    unavailable = metrics.loc[metrics["status"].astype(str).str.contains("missing|not_available|not_run", regex=True)]
    if not unavailable.empty:
        lines.extend(["", "- Unavailable or not-yet-run controls are recorded explicitly; top examples:", markdown_table(unavailable[["control_family", "control_name", "status"]].drop_duplicates(), max_rows=12)])
    return lines


def probe_summary(output_dir: Path) -> list[str]:
    metrics = optional_table(output_dir / "probes/probe_metrics.csv")
    if metrics is None or metrics.empty:
        return ["- Sparse probe metrics are unavailable. Do not claim activation-to-belief alignment from probes yet."]
    lines = [f"- Probe metric rows: `{len(metrics):,}`"]
    for metric in ["mean_r2", "r2", "balanced_accuracy", "top_action_accuracy_vs_true_top_action"]:
        if metric in metrics.columns:
            true = metrics.loc[metrics.get("control_type", "true").astype(str) == "true", metric]
            true = pd.to_numeric(true, errors="coerce").dropna()
            if not true.empty:
                lines.append(f"- True `{metric}` mean: `{compact_number(true.mean())}`, min: `{compact_number(true.min())}`, max: `{compact_number(true.max())}`")
    return lines


def geometry_summary(output_dir: Path) -> list[str]:
    metrics = optional_table(output_dir / "geometry_alignment/distance_matrix_metrics.csv")
    if metrics is None or metrics.empty:
        return ["- Geometry alignment metrics are unavailable. Do not claim behavior-activation manifold alignment yet."]
    primary = metrics.loc[metrics["is_primary"].fillna(False)] if "is_primary" in metrics.columns else metrics
    lines = [f"- Geometry metric rows: `{len(metrics):,}`", f"- Primary metric rows: `{len(primary):,}`"]
    if not primary.empty and "spearman" in primary.columns:
        lines.append(f"- Mean primary Spearman: `{compact_number(primary['spearman'].mean())}`")
        best = primary.sort_values("spearman", ascending=False).head(5)
        cols = [col for col in ["subset", "activation_layer", "activation_representation", "spearman", "permutation_p_value"] if col in best.columns]
        lines.extend(["", markdown_table(best[cols])])
    lines.append("- Caveat: four actions imply only six pairwise centroid distances per matrix comparison.")
    return lines


def action_contrast_summary(output_dir: Path) -> list[str]:
    metrics = optional_table(output_dir / "action_centroid_contrast/action_contrast_metrics.csv")
    if metrics is None or metrics.empty:
        return ["- Action-centroid contrast metrics are unavailable."]
    lines = [
        f"- Action-centroid contrast metric rows: `{len(metrics):,}`",
        f"- Splits: `{metrics['split_id'].nunique():,}`" if "split_id" in metrics.columns else "- Splits: `NA`",
        f"- Layers: `{metrics['layer'].nunique():,}`" if "layer" in metrics.columns else "- Layers: `NA`",
        "- Representations: "
        + ", ".join(f"`{name}`" for name in sorted(metrics["representation"].dropna().astype(str).unique()))
        if "representation" in metrics.columns
        else "- Representations: `NA`",
    ]
    score_col = "test_rows_balanced_accuracy_vs_agent_action_contrast"
    if score_col in metrics.columns:
        best = metrics.sort_values(score_col, ascending=False).head(8)
        cols = [
            "split_id",
            "layer",
            "representation",
            "subset",
            "test_rows_balanced_accuracy_vs_agent_action_contrast",
            "test_rows_balanced_accuracy_vs_agent_action_highdim",
            "train_extreme_balanced_accuracy_vs_extreme_action_contrast",
            "centroid_pair_spearman_highdim_cosine_vs_contrast_euclidean",
        ]
        lines.extend(["", "- Best blocks by held-out/test balanced accuracy:", "", markdown_table(best, cols, max_rows=8)])
        top = best.iloc[0]
        lines.extend(
            [
                "",
                f"- Best held-out/test contrast balanced accuracy: `{compact_number(top.get(score_col))}`.",
                "- Chance for four balanced actions is `0.25`; this result should be interpreted as weak held-out separation unless supported by stronger probe/control evidence.",
            ]
        )
    if {"energy_ratio_1", "energy_ratio_2", "energy_ratio_3"}.issubset(metrics.columns):
        top_energy = metrics[["energy_ratio_1", "energy_ratio_2", "energy_ratio_3"]].mean()
        lines.append(
            "- Mean contrast-axis energy ratios: "
            + ", ".join(f"`axis {i + 1}`={compact_number(value)}" for i, value in enumerate(top_energy.tolist()))
        )
    lines.extend(
        [
            "- Method caveat: this projection is anchored, not blind PCA/UMAP. The SVD is only on four train action-centroid contrast vectors.",
            "- Leakage caveat: preprocessing, centroids, and axes are fit on train rows/train extremes only; test/all-row metrics are more informative than train-extreme separation.",
            "- Geometry caveat: four actions give only six centroid-pair distances.",
        ]
    )
    return lines


def figures_catalog(figure_dir: Path, report_path: Path) -> list[dict[str, str]]:
    paths = sorted(figure_dir.rglob("*.png")) if figure_dir.exists() else []
    return [figure_explanation(path, report_path) for path in paths]


def figure_explanation(path: Path, report_path: Path) -> dict[str, str]:
    name = path.name
    parent = path.parent.name
    defaults = {
        "x_axis": "See the figure title and source script for the plotted horizontal variable.",
        "y_axis": "See the figure title and source script for the plotted vertical variable.",
        "color": "Color encodes the group, metric value, or category named in the figure title when present.",
        "marks": "Each point, line, or bar is an aggregate or prefix-level item generated by the source analysis.",
        "leakage": "This figure reads post-hoc outputs and does not fit a new model.",
        "caveat": "Interpret as descriptive unless paired with held-out grouped metrics and controls.",
    }
    mapping: dict[str, dict[str, str]] = {
        "coverage_by_normalized_bin.png": {
            "x_axis": "Normalized progress bin before the explicit action mention.",
            "y_axis": "Number of prefix rows in each bin.",
            "color": "Line color distinguishes strict/buffered pre-action subset definitions.",
            "marks": "Each line is one strict pre-action subset; each point is a bin count.",
            "leakage": "No model fitting is performed; this is a coverage diagnostic.",
            "caveat": "Coverage can shrink near the mention boundary, so late-bin trends can be noisy.",
        },
        "behavior_commitment_before_mention.png": {
            "x_axis": "Normalized progress bin before the explicit action mention.",
            "y_axis": "Mean behavior q(final action).",
            "color": "Line color distinguishes strict/buffered pre-action subset definitions.",
            "marks": "Each point is a bin-level mean over prefix belief vectors.",
            "leakage": "Uses behavior probabilities already scored for prefixes; no train/test fit occurs in this plot.",
            "caveat": "Pre-action means no lexical action token, not absence of implicit path evidence.",
        },
        "true_vs_shuffled_probe_metrics.png": {
            "x_axis": "Control condition.",
            "y_axis": "Top-action accuracy for belief-level original/shuffled controls.",
            "color": "Bars use a single color; categories are on the x-axis.",
            "marks": "Each bar is an aggregate over prefix rows.",
            "leakage": "This current run is a belief-level shuffle diagnostic; probe controls require Step 7 outputs.",
            "caveat": "This is not evidence of activation-to-belief alignment without probe metrics.",
        },
        "action_balanced_vs_all_metrics.png": {
            "x_axis": "All decisions versus action-balanced subsample, crossed with metric name.",
            "y_axis": "Metric value.",
            "color": "Bars use a single color; categories are on the x-axis.",
            "marks": "Each bar is an aggregate over the selected decision set.",
            "leakage": "No model fitting is performed; balanced sampling is at decision level.",
            "caveat": "Balanced subsets reduce sample size and can increase variance.",
        },
        "cot_length_stratified_metrics.png": {
            "x_axis": "CoT length stratum: short, medium, or long.",
            "y_axis": "Mean commitment_to_final.",
            "color": "Bars use a single color; strata are on the x-axis.",
            "marks": "Each bar is an aggregate over prefixes in decisions from that length stratum.",
            "leakage": "No model fitting is performed.",
            "caveat": "CoT length is correlated with task/trajectory properties and is not causal by itself.",
        },
    }
    info = {**defaults, **mapping.get(name, {})}
    if parent == "action_centroid_contrast":
        if "grid" in name:
            info.update(
                {
                    "x_axis": "Action contrast component 1, derived from SVD of train-only action-centroid contrast vectors.",
                    "y_axis": "Action contrast component 2 from the same anchored contrast basis.",
                    "color": "Color encodes the action variable named in the filename and title: extreme action, agent action, commitment, or progress.",
                    "marks": "Each point is one prefix activation projected into the anchored contrast space; X markers are train-only action centroids.",
                    "leakage": "Centroids and contrast axes are fit from split-train action extremes only; test/all rows are projected afterward.",
                    "caveat": "This is not PCA/UMAP. Strong train-extreme separation can be inflated because the axes are defined from those anchors.",
                }
            )
        elif name.startswith("best_action_contrast_projection"):
            info.update(
                {
                    "x_axis": "Action contrast component 1 from the best block selected by held-out/test balanced accuracy.",
                    "y_axis": "Action contrast component 2 from the same anchored contrast basis.",
                    "color": "Color encodes the variable in the title, such as agent_action, top_action, commitment_to_final, or train extreme action.",
                    "marks": "Each point is a projected prefix row; X markers are train-only action centroids.",
                    "leakage": "The best block uses metrics computed after fitting axes on train-only extremes; selection itself is post-hoc and should be reported as exploratory.",
                    "caveat": "Held-out/test balanced accuracy is close to chance in the current run, so visual overlap should be interpreted conservatively.",
                }
            )
        elif name.startswith("contrast_"):
            info.update(
                {
                    "x_axis": "Activation layer index.",
                    "y_axis": "The metric named in the figure title.",
                    "color": "Line color distinguishes activation representation.",
                    "marks": "Each point is one split/layer/representation aggregate metric.",
                    "leakage": "Metrics are computed from train-only anchored axes and held-out/all-row projections as labeled.",
                    "caveat": "These are descriptive diagnostics; chance for four-action balanced accuracy is 0.25.",
                }
            )
    info["path"] = markdown_rel_from_report(path, report_path)
    info["repo_path"] = rel(path)
    info["section"] = parent
    info["name"] = name
    return info


def figure_markdown(figures: list[dict[str, str]]) -> str:
    if not figures:
        return "_No conceptual belief space figures are available yet._"
    lines: list[str] = []
    for fig in figures:
        lines.extend(
            [
                f"### {fig['name']}",
                "",
                f"![{fig['name']}]({fig['path']})",
                "",
                f"- x-axis: {fig['x_axis']}",
                f"- y-axis: {fig['y_axis']}",
                f"- color: {fig['color']}",
                f"- points/lines/bars: {fig['marks']}",
                f"- train/test leakage: {fig['leakage']}",
                f"- caveat: {fig['caveat']}",
                "",
            ]
        )
    return "\n".join(lines)


def snippets_markdown(output_dir: Path) -> str:
    lines = ["## Source Report Snippets", ""]
    for label, rel_path in REPORT_INPUTS:
        path = resolve_repo_path(rel_path)
        text = read_text(path)
        lines.extend([f"### {label}", "", f"Source: `{rel(path)}`", ""])
        if text is None:
            lines.append("_Missing. This upstream step has not produced its report in the current artifact set._")
        else:
            body = text.strip()
            if len(body) > 5000:
                body = body[:5000].rstrip() + "\n\n_Excerpt truncated by final report compiler._"
            lines.append(body)
        lines.append("")
    return "\n".join(lines)


def final_report_text(output_dir: Path, figure_dir: Path, report_path: Path) -> str:
    availability = report_availability()
    figures = figures_catalog(figure_dir, report_path)
    missing_sections = availability.loc[~availability["available"], "section"].tolist()
    lines: list[str] = [
        "# Conceptual Belief Space Final Report",
        "",
        "This report compiles the paper-style conceptual belief space analysis for the gridworld action commitment experiment.",
        "",
        "## Executive Summary",
        "",
        "- Blind PCA is treated as diagnostic only; it is not the main evidence for an activation manifold.",
        "- The intended main pipeline uses behavior action-belief vectors, train-only extreme examples, anchored behavior/activation conceptual manifolds, centroid distance matrices, sparse probes, trajectory analysis, strict pre-action subsets, and robustness controls.",
        "- In the current artifact set, behavior belief construction, extreme selection, strict pre-action behavior analysis, and controls are available.",
        "- Behavior/activation UMAP manifolds, centroid distance alignment, sparse probes, and trajectory analysis outputs are not yet available, so behavior-activation alignment should not be claimed from this run.",
        "- Four action concepts yield only six pairwise centroid distances, so any centroid-distance correlation must be reported as low-powered geometry evidence.",
        "",
        "## Analysis Availability",
        "",
        markdown_table(availability),
        "",
    ]

    section_bodies = {
        "Motivation: Why Blind PCA Was Insufficient": [
            "Blind PCA can be useful for exploratory visualization, but it is not anchored to the action concepts. A PCA plot may separate variance caused by progress, token position, trajectory length, or unrelated activation directions. The replacement pipeline anchors both behavior and activation spaces with train-only extreme examples for UP, DOWN, LEFT, and RIGHT, then evaluates geometry and probes under grouped splits.",
            "",
            "Main rule: do not claim that blind PCA shows the activation belief manifold. At most, blind PCA is a diagnostic visualization.",
        ],
        "Dataset and Inputs": [
            *action_belief_summary(output_dir),
            "",
            "The behavior-side rows are prefix-level action beliefs. Activation-side analyses should align CoT-token activations to these prefix indices, with separate care for pre_reasoning and post_reasoning states.",
        ],
        "Behavior Action-belief Construction": [
            "The behavior target is a normalized distribution over `UP`, `DOWN`, `LEFT`, and `RIGHT`, plus log, CLR, and square-root transforms.",
            "",
            *action_belief_summary(output_dir),
        ],
        "Train-only Extreme Examples": [
            "Extreme examples are selected from train rows only under grouped splits. They anchor the conceptual manifolds and centroids.",
            "",
            *extremes_summary(output_dir),
        ],
        "Behavior Conceptual Manifold": [
            "The intended behavior manifold is UMAP fit only on train extreme examples, then used to transform all prefix belief vectors.",
            "",
            "_Current status_: behavior manifold coordinates are unavailable in this artifact set. Rerun Step 4 after installing `umap-learn` to generate the primary behavior manifold outputs.",
        ],
        "Activation Conceptual Manifold": [
            "The intended activation manifold uses the same train extreme row ids as the behavior manifold. Activation extremes must not be selected independently.",
            "",
            "_Current status_: activation manifold coordinates and centroids are unavailable in this artifact set. Rerun Step 5 after Step 4.",
        ],
        "Action-centroid Contrast Projection": [
            "This anchored activation projection directly asks whether activation states separate along action-concept axes defined by train-only action-extreme examples.",
            "",
            "It does not use blind PCA or UMAP. The only dimensionality reduction is SVD on the four centered train action-centroid contrast vectors.",
            "",
            *action_contrast_summary(output_dir),
        ],
        "Centroid Distance Matrix Alignment": [
            *geometry_summary(output_dir),
            "",
            "Alignment should be claimed only if distance-matrix correlations and probe metrics both support it. If alignment is weak, report it as weak.",
        ],
        "Sparse Probes": [
            *probe_summary(output_dir),
            "",
            "Probe inputs must be high-dimensional activation features, not UMAP or PCA coordinates. Negative R2 values should be reported honestly.",
        ],
        "Trajectory Analysis": [
            "The intended trajectory analysis compares paths through behavior and activation conceptual spaces, plus centroid basin switching and lock-in.",
            "",
            "_Current status_: trajectory outputs are unavailable because the conceptual behavior/activation manifold artifacts are unavailable.",
        ],
        "Strict Pre-action Analysis": [
            *strict_summary(output_dir),
            "",
            "Pre-action means no explicit lexical action token has appeared in the generated CoT. It does not mean the prompt lacks coordinate, path, or implicit action evidence.",
        ],
        "Robustness Controls": [
            *controls_summary(output_dir),
        ],
        "What Changed Relative to the Previous PCA Analysis": [
            "- Blind PCA is no longer the primary manifold method.",
            "- Conceptual spaces are anchored by train-only extreme examples instead of unsupervised variance directions.",
            "- Behavior and activation spaces are compared with action centroid distance matrices and sparse probes, not visual similarity alone.",
            "- Added action-centroid contrast projection as a focused anchored diagnostic for whether activations separate along train-defined UP/DOWN/LEFT/RIGHT axes.",
            "- Strict pre-action analyses are separated from all-prefix analyses.",
            "- Grouped train/test splits are required for fitting and evaluation to avoid trajectory/decision leakage.",
            "- The report now distinguishes CoT-token activations from final JSON decision-readout activations.",
        ],
        "Interpretation and Limitations": [
            "- Do not claim behavior-activation alignment unless both centroid distance correlations and held-out sparse probe metrics support it.",
            "- If alignment metrics are missing or weak, say they are missing or weak.",
            "- Four actions give only six pairwise centroid distances, so geometry-correlation statistics are low-powered.",
            "- CoT-token activations are not necessarily the same state used for the final JSON action readout.",
            "- Strict pre-action means no explicit action token, not no implicit evidence.",
            "- All current analyses are observational; they do not prove causality.",
            "- Random prefix splits are invalid for main claims and only useful as leakage diagnostics.",
        ],
        "Next Experiments": [
            "1. Install `umap-learn` and rerun Steps 4 and 5 to produce anchored behavior and activation manifolds.",
            "2. Rerun Step 6 for centroid distance matrix alignment and report the four-action/six-distance caveat.",
            "3. Rerun Step 7 for grouped sparse probes across layers and representations.",
            "4. Rerun Steps 8-10 after probe and geometry outputs exist so trajectory and control figures include activation-side evidence.",
            "5. Add final JSON decision-readout activations and compare them with CoT-token activations.",
            "6. Run causal interventions only after descriptive geometry/probe evidence is stable.",
        ],
    }

    for title in SECTION_ORDER:
        lines.extend([f"## {title}", ""])
        body = section_bodies[title]
        lines.extend(body if isinstance(body, list) else [str(body)])
        lines.append("")

    lines.extend(["## Figure Guide", "", figure_markdown(figures), ""])
    if missing_sections:
        lines.extend(["## Missing Upstream Reports", ""])
        lines.extend(f"- {section}" for section in missing_sections)
        lines.append("")
    lines.append(snippets_markdown(output_dir))
    return "\n".join(str(line) for line in lines)


def paper_section_text(output_dir: Path) -> str:
    strict = strict_summary(output_dir)
    controls = controls_summary(output_dir)
    probe = probe_summary(output_dir)
    geometry = geometry_summary(output_dir)
    contrast = action_contrast_summary(output_dir)
    return "\n".join(
        [
            "# Paper Section: Activation Belief Space Analysis",
            "",
            "We replaced blind PCA with a conceptual belief space pipeline. The behavior side represents each prefix by an action-belief distribution over UP, DOWN, LEFT, and RIGHT, with log, centered-log-ratio, and square-root transforms. Train-only action-extreme examples are used to anchor the behavior and activation manifolds, preventing test examples from defining the conceptual axes.",
            "",
            "In the current artifact set, the behavior belief table and strict pre-action behavior analysis are available, but the anchored behavior/activation UMAPs, centroid distance alignment, and sparse probe outputs are not yet available. Therefore we do not claim behavior-activation alignment from the current run.",
            "",
            "Strict pre-action results:",
            *strict,
            "",
            "Robustness controls:",
            *controls[:8],
            "",
            "Geometry alignment status:",
            *geometry,
            "",
            "Action-centroid contrast projection:",
            *contrast,
            "",
            "Sparse probe status:",
            *probe,
            "",
            "Interpretation caveats: blind PCA is diagnostic only; four actions provide only six pairwise centroid distances; CoT-token activations are distinct from final JSON decision-readout activations; and all reported curves are observational rather than causal.",
        ]
    )


def run(*, output: Path, paper_output: Path, output_dir: Path, figure_dir: Path) -> None:
    ensure_dir(output.parent)
    save_markdown(output, final_report_text(output_dir, figure_dir, output))
    ensure_dir(paper_output.parent)
    save_markdown(paper_output, paper_section_text(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR / "conceptual_belief_space_final_report.md")
    parser.add_argument("--paper_output", default=DEFAULT_OUTPUT_DIR / "paper_section_activation_belief_space.md")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    output = resolve_repo_path(args.output)
    paper_output = resolve_repo_path(args.paper_output)
    run(
        output=output,
        paper_output=paper_output,
        output_dir=resolve_repo_path(args.output_dir),
        figure_dir=resolve_repo_path(args.figure_dir),
    )
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(paper_output)}")


if __name__ == "__main__":
    main()
