#!/usr/bin/env python3
"""Write the final commitment-only activation contrast report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_OUTPUT_DIR
from src.conceptual_belief_space.io import ensure_dir, save_markdown


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "commitment_contrast"
FIGURE_DIR = Path("figures/conceptual_belief_space/commitment_contrast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR / "commitment_contrast_report.md")
    parser.add_argument("--metrics", default=OUTPUT_DIR / "commitment_projection_metrics.csv")
    parser.add_argument("--control_summary", default=OUTPUT_DIR / "controls/commitment_control_summary.csv")
    parser.add_argument("--best_axes", default=OUTPUT_DIR / "controls/best_commitment_axes.csv")
    parser.add_argument("--failure_modes", default=OUTPUT_DIR / "controls/failure_modes.csv")
    parser.add_argument("--plot_manifest", default=OUTPUT_DIR / "plot_manifest.json")
    parser.add_argument("--run_manifest", default=OUTPUT_DIR / "run_manifest.json")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
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


def read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def ok_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    if "status" in metrics.columns:
        return metrics[metrics["status"].astype(str).eq("ok")].copy()
    return metrics.copy()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def best_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    part = metrics.copy()
    part["_auc"] = numeric(part, "auc_high_vs_low_test")
    part["_partial"] = numeric(part, "partial_spearman_score_target_resid_progress_test")
    cols = [
        "target",
        "subset",
        "representation",
        "layer",
        "axis_variant",
        "auc_high_vs_low_test",
        "balanced_accuracy_high_vs_low_test",
        "spearman_score_target_test",
        "partial_spearman_score_target_resid_progress_test",
        "spearman_score_target_within_progress_bin_mean_test",
        "d_prime_high_vs_low_test",
        "n_low_test",
        "n_high_test",
    ]
    cols = [col for col in cols if col in part.columns]
    return part.sort_values(["_auc", "_partial"], ascending=False)[cols].head(20)


def pick_figure(plot_manifest: dict, contains: list[str]) -> str | None:
    figures = plot_manifest.get("created_figures", [])
    for fig in figures:
        if all(token in fig for token in contains):
            return fig
    return None


def summarize_control_result(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- Control summary is missing; run script 14."]
    lines = []
    original = summary[summary["axis_variant"].astype(str).eq("original_action_balanced")]
    shuffled = summary[summary["axis_variant"].astype(str).eq("shuffled_commitment_control")]
    progress = summary[summary["axis_variant"].astype(str).eq("progress_matched_action_balanced")]
    residual = summary[summary["axis_variant"].astype(str).eq("progress_residualized_feature_control")]
    if not original.empty:
        lines.append(f"- Original axes mean held-out AUC: {numeric(original, 'auc_high_vs_low_test').mean():.3f}.")
        lines.append(
            f"- Original axes mean partial Spearman after progress covariates: "
            f"{numeric(original, 'partial_spearman_score_target_resid_progress_test').mean():.3f}."
        )
    if not progress.empty:
        lines.append(f"- Progress-matched axes mean held-out AUC: {numeric(progress, 'auc_high_vs_low_test').mean():.3f}.")
    if not shuffled.empty:
        lines.append(f"- Shuffled-commitment control mean held-out AUC: {numeric(shuffled, 'auc_high_vs_low_test').mean():.3f}.")
    if not residual.empty:
        lines.append(f"- Progress-residualized feature control mean held-out AUC: {numeric(residual, 'auc_high_vs_low_test').mean():.3f}.")
    survived = summary.get("survives_shuffled_control")
    if survived is not None:
        try:
            lines.append(f"- Fraction marked as surviving shuffled control: {float(pd.Series(survived).fillna(False).mean()):.3f}.")
        except Exception:
            pass
    return lines


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    ensure_dir(output_dir)
    metrics = ok_metrics(read_csv_optional(resolve(args.metrics)))
    controls = read_csv_optional(resolve(args.control_summary))
    best_axes = read_csv_optional(resolve(args.best_axes))
    failures = read_csv_optional(resolve(args.failure_modes))
    plot_manifest = read_json_optional(resolve(args.plot_manifest))
    run_manifest = read_json_optional(resolve(args.run_manifest))

    best_table = best_results_table(metrics)
    best_row = best_table.iloc[0].to_dict() if not best_table.empty else {}
    figure_candidates = {
        "best_projection_grid": pick_figure(
            plot_manifest,
            [
                "projection_grids/grid",
                str(best_row.get("target", "")),
                str(best_row.get("subset", "")),
                str(best_row.get("axis_variant", "")),
                str(best_row.get("representation", "")),
                "by_commitment.png",
            ],
        ),
        "best_commitment_bin_grid": pick_figure(
            plot_manifest,
            [
                "projection_grids/grid",
                str(best_row.get("target", "")),
                str(best_row.get("subset", "")),
                str(best_row.get("axis_variant", "")),
                str(best_row.get("representation", "")),
                "by_commitment_bin.png",
            ],
        ),
        "best_histogram": pick_figure(plot_manifest, ["metrics/hist_test_score", str(best_row.get("target", ""))]),
        "best_trajectory": pick_figure(plot_manifest, ["trajectories/selected_trajectories_phase_plane", str(best_row.get("target", ""))]),
        "control_summary": "figures/conceptual_belief_space/commitment_contrast/metrics/control_summary.png"
        if resolve("figures/conceptual_belief_space/commitment_contrast/metrics/control_summary.png").exists()
        else None,
    }

    n_rows = run_manifest.get("n_behavior_rows", "unknown")
    layers = run_manifest.get("layers", sorted(metrics["layer"].dropna().astype(int).unique().tolist()) if not metrics.empty and "layer" in metrics.columns else [])
    representations = run_manifest.get("representations", sorted(metrics["representation"].dropna().astype(str).unique().tolist()) if not metrics.empty and "representation" in metrics.columns else [])
    targets = run_manifest.get("targets", sorted(metrics["target"].dropna().astype(str).unique().tolist()) if not metrics.empty and "target" in metrics.columns else [])

    lines = [
        "# Commitment-only Activation Contrast Report",
        "",
        "## Executive Summary",
        "",
        "This analysis tests whether CoT-token activation states contain an action-agnostic low-to-high commitment direction.",
        "PCA and UMAP are not used as the primary projection. The primary projection axis is the train-only high-dimensional centroid contrast `c_high - c_low` for a commitment target.",
        "Action labels are used only for balancing and diagnostics, not as the main target.",
        "",
    ]
    if best_row:
        lines.extend(
            [
                f"- Best observed held-out row: target `{best_row.get('target')}`, subset `{best_row.get('subset')}`, representation `{best_row.get('representation')}`, layer `{best_row.get('layer')}`, variant `{best_row.get('axis_variant')}`.",
                f"- Held-out AUC: {float(best_row.get('auc_high_vs_low_test', np.nan)):.3f}.",
                f"- Held-out balanced accuracy: {float(best_row.get('balanced_accuracy_high_vs_low_test', np.nan)):.3f}.",
                f"- Partial Spearman after progress covariates: {float(best_row.get('partial_spearman_score_target_resid_progress_test', np.nan)):.3f}.",
            ]
        )
    lines.extend(summarize_control_result(controls))
    lines.extend(
        [
            "",
            "If the progress-matched, progress-residualized, strict pre-action, and shuffled controls are weak, the result should be reported as weak rather than as evidence for a robust internal commitment axis.",
            "",
            "## Why this analysis",
            "",
            "Four-action branch plots can fail if action identity is weak, mixed with progress, or not linearly organized by UP/DOWN/LEFT/RIGHT.",
            "The commitment-only analysis asks a more action-agnostic question: is the model internally more or less committed, regardless of which action is favored?",
            "",
            "## Data and features",
            "",
            f"- Prefix rows: {n_rows}",
            f"- Layers: {layers}",
            f"- Representations: {representations}",
            "- Feature path pattern: `outputs/activation/features/layer_{LL}_{repr}.npy`",
            f"- Targets: {targets}",
            "",
            "## Target definitions",
            "",
            "- `top_commitment = max(q_UP, q_DOWN, q_LEFT, q_RIGHT)`",
            "- `entropy_commitment = 1 - H(q) / log(4)`",
            "- `commitment_to_final = q(agent_action)`",
            "",
            "`entropy_commitment` and `top_commitment` are action-agnostic. `commitment_to_final` is hindsight-aligned to the final action label.",
            "",
            "## Method",
            "",
            "1. Use grouped train/test splits by trajectory or decision to avoid prefix leakage.",
            "2. Select train rows with low or high target commitment.",
            "3. Balance fitting rows by action; for the progress-matched control, balance by action and progress bin.",
            "4. Fit the activation scaler on train rows only.",
            "5. Compute `c_low = mean(X_low_train)` and `c_high = mean(X_high_train)`.",
            "6. Define `u_commit = normalize(c_high - c_low)`.",
            "7. Score every selected row with `score_commit(x) = <standardize(x) - c_mid, u_commit>`.",
            "8. Define a second visualization coordinate from an orthogonal train-fitted progress contrast, or use residual norm when the progress contrast degenerates.",
            "9. Evaluate held-out low/high separation and rank correlations against the continuous target.",
            "",
            "No PCA/UMAP coordinates are used for the primary projection or the primary evidence.",
            "",
            "## Main figures",
            "",
        ]
    )
    for label, path in figure_candidates.items():
        if path:
            lines.append(f"### {label.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{label}]({path})")
            lines.append("")
            lines.append(
                "Each point is one prefix-level CoT-token activation state. The x-axis is the train-fitted low-to-high commitment contrast score. "
                "The y-axis is the train-fitted orthogonal progress contrast or residual norm. Color denotes commitment value or commitment bin. "
                "Train/test leakage is avoided because axes and preprocessing are fit only on train rows. "
                "Caveat: this is observational and uses CoT-token activations, not necessarily final JSON decision-readout activations."
            )
            lines.append("")
    lines.extend(["## Metrics table", ""])
    if not best_table.empty:
        lines.append(best_table.to_markdown(index=False))
    else:
        lines.append("No successful metrics were found.")
    lines.extend(["", "## Controls", ""])
    lines.extend(summarize_control_result(controls))
    if not controls.empty:
        lines.extend(["", "### Control Summary Table", "", controls.head(30).to_markdown(index=False)])
    if not failures.empty:
        lines.extend(["", "### Failure Modes", "", failures.head(30).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Supported interpretation, if metrics pass controls: activation states contain a train-defined low-to-high commitment direction that generalizes to held-out trajectories.",
            "Unsupported interpretation, if controls fail or metrics are weak: the current CoT-token activation readout does not robustly expose an action-agnostic commitment axis under these controls.",
            "",
            "Do not claim a causal effect, final JSON decision-readout state, PCA/UMAP manifold evidence, four action branches, or that pre-action means absence of implicit coordinate/path evidence.",
            "",
            "## Next steps",
            "",
            "- Run the same commitment contrast on final JSON decision-readout activations if available.",
            "- Try attention-weighted pooling over recent CoT activations.",
            "- Try query-token or action-token readout activations.",
            "- Compare single-token last-prefix state with pooled prefix state.",
            "- Add more trajectories and seeds.",
        ]
    )
    output = resolve(args.output)
    save_markdown(output, "\n".join(lines))
    print(json.dumps({"report": rel(output), "metrics_rows": int(len(metrics)), "control_rows": int(len(controls))}, indent=2))


if __name__ == "__main__":
    main()
