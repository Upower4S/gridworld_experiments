#!/usr/bin/env python
"""Write the final supervised MLP branch probe report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.io import read_json, save_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def maybe_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return "_Requested columns unavailable._"
    return df[existing].head(n).to_markdown(index=False)


def figure_link(path: str | Path, caption: str) -> str:
    path = Path(path)
    if path.exists():
        return f"![{caption}]({path})\n\n{caption}"
    return f"_Missing figure: `{path}`_"


def best_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frame = metrics[
        metrics["model_type"].astype(str).eq("mlp_branch")
        & metrics["status"].astype(str).eq("ok")
    ].copy()
    if frame.empty:
        return frame
    frame["_score"] = pd.to_numeric(frame.get("test_q_top_action_balanced_accuracy", 0), errors="coerce").fillna(0)
    frame["_score"] += 0.25 * pd.to_numeric(frame.get("test_commitment_spearman", 0), errors="coerce").fillna(0)
    return frame.sort_values("_score", ascending=False)


def bool_statement(condition: bool, positive: str, negative: str) -> str:
    return positive if condition else negative


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    metrics_path = output_dir / "mlp_branch_metrics.csv"
    preds_path = output_dir / "predictions/mlp_branch_predictions.parquet"
    configs_path = output_dir / "best_mlp_branch_configs.json"
    controls_path = output_dir / "controls/mlp_branch_control_summary.csv"
    failures_path = output_dir / "controls/mlp_branch_failure_modes.csv"
    manifest_path = output_dir / "plot_manifest.json"
    output = Path(args.output) if args.output else output_dir / "reports/mlp_branch_report.md"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    controls = pd.read_csv(controls_path) if controls_path.exists() else pd.DataFrame()
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    configs = read_json(configs_path) if configs_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    best = best_rows(metrics)
    best_row = best.iloc[0] if not best.empty else None
    mlp = metrics[metrics["model_type"].astype(str).eq("mlp_branch")] if not metrics.empty and "model_type" in metrics else pd.DataFrame()
    linear = metrics[metrics["model_type"].astype(str).eq("linear_baseline")] if not metrics.empty and "model_type" in metrics else pd.DataFrame()
    shuffled = metrics[metrics["model_type"].astype(str).eq("mlp_shuffled_target")] if not metrics.empty and "model_type" in metrics else pd.DataFrame()
    progress = metrics[metrics["model_type"].astype(str).eq("progress_only")] if not metrics.empty and "model_type" in metrics else pd.DataFrame()

    if not mlp.empty:
        mean_mlp_acc = pd.to_numeric(mlp.get("test_q_top_action_balanced_accuracy"), errors="coerce").mean()
        mean_commit = pd.to_numeric(mlp.get("test_commitment_spearman"), errors="coerce").mean()
    else:
        mean_mlp_acc = float("nan")
        mean_commit = float("nan")
    linear_mean = pd.to_numeric(linear.get("test_q_top_action_balanced_accuracy"), errors="coerce").mean() if not linear.empty else float("nan")
    shuffled_mean = pd.to_numeric(shuffled.get("test_q_top_action_balanced_accuracy"), errors="coerce").mean() if not shuffled.empty else float("nan")
    progress_mean = pd.to_numeric(progress.get("test_q_top_action_balanced_accuracy"), errors="coerce").mean() if not progress.empty else float("nan")
    strict = mlp[mlp["subset"].astype(str).str.contains("strict", na=False)] if not mlp.empty else pd.DataFrame()
    strict_mean = pd.to_numeric(strict.get("test_q_top_action_balanced_accuracy"), errors="coerce").mean() if not strict.empty else float("nan")
    likely_beats_linear = np.isfinite(mean_mlp_acc) and np.isfinite(linear_mean) and mean_mlp_acc > linear_mean
    likely_beats_shuffled = np.isfinite(mean_mlp_acc) and np.isfinite(shuffled_mean) and mean_mlp_acc > shuffled_mean + 0.05
    likely_beats_progress = np.isfinite(mean_mlp_acc) and np.isfinite(progress_mean) and mean_mlp_acc > progress_mean + 0.03

    lines = [
        "# MLP Branch Probe Report",
        "",
        "## Executive Summary",
        "",
        "This analysis trains a supervised nonlinear MLP probe with a 2D bottleneck. The resulting coordinates are a supervised action-belief readout, not an unsupervised activation manifold and not raw PCA/UMAP.",
        "",
        f"- Mean MLP q top-action balanced accuracy: {mean_mlp_acc:.3f}" if np.isfinite(mean_mlp_acc) else "- Mean MLP q top-action balanced accuracy: unavailable",
        f"- Mean MLP commitment Spearman: {mean_commit:.3f}" if np.isfinite(mean_commit) else "- Mean MLP commitment Spearman: unavailable",
        f"- Linear baseline mean top-action balanced accuracy: {linear_mean:.3f}" if np.isfinite(linear_mean) else "- Linear baseline mean top-action balanced accuracy: unavailable",
        f"- Shuffled-target mean top-action balanced accuracy: {shuffled_mean:.3f}" if np.isfinite(shuffled_mean) else "- Shuffled-target mean top-action balanced accuracy: unavailable",
        f"- Progress-only mean top-action balanced accuracy: {progress_mean:.3f}" if np.isfinite(progress_mean) else "- Progress-only mean top-action balanced accuracy: unavailable",
        f"- Strict pre-action mean top-action balanced accuracy: {strict_mean:.3f}" if np.isfinite(strict_mean) else "- Strict pre-action mean top-action balanced accuracy: unavailable",
        f"- MLP beats linear baseline on mean top-action balanced accuracy: {likely_beats_linear}",
        f"- MLP beats shuffled control by margin: {likely_beats_shuffled}",
        f"- MLP beats progress-only control by margin: {likely_beats_progress}",
        "",
        "The correct positive interpretation, if the metrics and controls support it, is: a supervised MLP readout can recover a four-branch action-belief embedding from CoT-token activations. It is not evidence that raw CoT-token activations naturally form four branches.",
        "",
        "## Motivation",
        "",
        "Previous linear probes and centroid projections are the interpretability baseline. The MLP branch probe tests nonlinear recoverability and deliberately creates a readable supervised 2D bottleneck. This makes the visualization useful as a diagnostic, but weaker as evidence about raw geometry.",
        "",
        "## Method",
        "",
        "- Input: high-dimensional activation features aligned to behavior prefix rows.",
        "- Split: trajectory-grouped train/validation/test split; random prefix splits are not used for main results.",
        "- Preprocessing: scaler fitted on train rows only.",
        "- Model: activation h -> hidden layers -> 2D bottleneck z -> output heads.",
        "- Constraint: by default all heads depend only on the 2D bottleneck.",
        "- Targets: behavior q, top_action, agent_action, entropy_commitment, and a light branch anchor.",
        "- Fixed action vertices: UP=(1,1), DOWN=(1,-1), LEFT=(-1,1), RIGHT=(-1,-1).",
        "",
        "Formulas:",
        "",
        "```text",
        "q_pred = softmax(W_q z)",
        "v_q = sum_a q_a v_a",
        "z_plot = tanh(z)",
        "L_anchor = ||z_plot - v_q||^2",
        "```",
        "",
        "## Best Branch Configs",
        "",
        "```json",
        json.dumps(configs, indent=2)[:6000],
        "```",
        "",
        "## Results",
        "",
        "### Best MLP Rows",
        "",
        maybe_table(
            best,
            [
                "run_id",
                "split_id",
                "layer",
                "representation",
                "subset",
                "test_q_top_action_balanced_accuracy",
                "test_commitment_spearman",
                "test_q_kl",
                "test_vertex_top_action_balanced_accuracy",
                "test_radius_commitment_spearman",
            ],
            n=12,
        ),
        "",
        "### MLP vs Controls",
        "",
        maybe_table(
            controls.sort_values("mlp_top_action_bal_acc", ascending=False) if not controls.empty and "mlp_top_action_bal_acc" in controls else controls,
            [
                "run_id",
                "layer",
                "representation",
                "subset",
                "mlp_top_action_bal_acc",
                "delta_top_action_bal_acc_vs_linear_baseline",
                "delta_top_action_bal_acc_vs_mlp_shuffled_target",
                "delta_top_action_bal_acc_vs_progress_only",
                "control_flags",
            ],
            n=12,
        ),
        "",
        "## Figures",
        "",
        figure_link(manifest.get("best_branch_by_top_action", figure_dir / "branch_manifold/best_branch_by_top_action.png"), "Supervised MLP bottleneck embedding by top_action. x/y are tanh bottleneck coordinates; color is top_action; points are prefix rows; star markers are fixed action vertices."),
        "",
        figure_link(manifest.get("best_branch_by_agent_action", figure_dir / "branch_manifold/best_branch_by_agent_action.png"), "Supervised MLP bottleneck embedding by agent_action. x/y are tanh bottleneck coordinates; color is final agent_action."),
        "",
        figure_link(manifest.get("best_branch_by_entropy_commitment", figure_dir / "branch_manifold/best_branch_by_entropy_commitment.png"), "Supervised MLP bottleneck embedding by entropy_commitment. x/y are tanh bottleneck coordinates; color is behavior commitment."),
        "",
        figure_link(manifest.get("selected_branch_trajectories", figure_dir / "trajectories/selected_branch_trajectories.png"), "Selected supervised bottleneck trajectories. Lines connect prefixes of the same decision; open circle marks start and X marks final measured prefix."),
        "",
        figure_link(manifest.get("radius_over_progress", figure_dir / "trajectories/radius_over_progress.png"), "Supervised bottleneck radius over progress. x-axis is progress bin; y-axis is mean radius or commitment."),
        "",
        figure_link(manifest.get("mlp_branch_vs_baselines", figure_dir / "metrics/mlp_branch_vs_baselines.png"), "MLP branch probe compared with baselines. Bars show metrics for matching layer, representation, subset, and split."),
        "",
        figure_link(manifest.get("mlp_branch_metric_heatmap_by_layer_representation", figure_dir / "metrics/mlp_branch_metric_heatmap_by_layer_representation.png"), "Layer/representation heatmap for supervised MLP bottleneck performance."),
        "",
        "## Interpretation",
        "",
        bool_statement(
            likely_beats_shuffled,
            "The MLP is above the shuffled-target control on average, supporting nontrivial recoverability.",
            "The MLP does not clearly beat the shuffled-target control on average; treat branch structure as weak or unsupported.",
        ),
        "",
        bool_statement(
            likely_beats_progress,
            "The MLP is above the progress-only baseline on average, reducing but not eliminating progress-confound concern.",
            "The progress-only baseline is close to the MLP; part of the branch or radius structure may be progress-driven.",
        ),
        "",
        "If the branch plots show four separated arms, the claim should be limited to supervised nonlinear recoverability: a trained bottleneck readout can arrange activations in an action-belief coordinate system.",
        "",
        "## Limitations",
        "",
        "- This is supervised embedding, not an unsupervised raw activation manifold.",
        "- CoT-token activations are analyzed, not final JSON decision-readout activations.",
        "- The four-action setup has limited conceptual richness.",
        "- These are observational diagnostics and do not establish causality.",
        "- Progress, CoT length, and prompt priors may confound parts of the signal.",
        "- Linear probes remain the interpretability baseline; nonlinear MLP success is weaker evidence.",
        "- Bad metrics, weak controls, and negative results should be reported honestly rather than hidden.",
        "",
        "## Next Steps",
        "",
        "- Run the same probe on final JSON decision-readout activations.",
        "- Add attention-pooled CoT states and post_reasoning deltas.",
        "- Run on all 100 trajectories if the current outputs are from a subset.",
        "- Add richer gridworld task concepts beyond four actions.",
    ]
    if not failures.empty:
        lines.extend(["", "## Control Flags", "", maybe_table(failures, list(failures.columns), n=30)])
    save_markdown(output, "\n".join(lines))
    print(json.dumps({"report": str(output), "metrics_rows": int(len(metrics)), "control_rows": int(len(controls))}, indent=2))


if __name__ == "__main__":
    main()
