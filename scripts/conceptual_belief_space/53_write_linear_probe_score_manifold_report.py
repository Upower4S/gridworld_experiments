#!/usr/bin/env python3
"""Write the final report for the supervised linear-probe score manifold."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def fmt(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def figure_table(figure_dir: Path) -> str:
    rows = []
    descriptions = {
        "linear_score_plane_by_true_q_top.png": "Main linear x/y score plane colored by true q-top action; points are held-out prefix rows.",
        "linear_score_plane_by_pred_q_top.png": "Same score plane colored by the raw-only stacker predicted q-top action.",
        "linear_score_plane_by_entropy_commitment.png": "Score plane colored by true q entropy commitment.",
        "linear_score_plane_by_progress.png": "Score plane colored by reasoning progress.",
        "high_commitment_linear_score_plane_by_true_q_top.png": "High-commitment subset, where four-branch structure is most meaningful.",
        "low_vs_high_commitment_comparison.png": "Side-by-side comparison of low-commitment and high-commitment rows.",
        "linear_score_trajectories.png": "Selected decision trajectories through the score plane, ordered by prefix token index.",
        "mean_trajectory_by_action.png": "Median trajectory by true q-top action over progress bins.",
        "strict_pre_action_linear_score_plane.png": "Strict pre-action subset colored by true q-top action.",
        "strict_pre_action_linear_score_plane_by_entropy_commitment.png": "Strict pre-action subset colored by entropy commitment.",
        "score_radius_vs_commitment.png": "Score-plane radius compared with true commitment.",
        "score_radius_vs_progress.png": "Score-plane radius compared with progress.",
        "partial_radius_commitment_control.png": "Correlation and partial-correlation controls for radius, commitment, and progress.",
        "optional_3d_centered_score_simplex.png": "Centered four-score vector shown in 3D Helmert coordinates.",
    }
    for path in sorted(figure_dir.glob("*.png")):
        rows.append({"figure": path.name, "description": descriptions.get(path.name, "Generated score-manifold diagnostic figure.")})
    return pd.DataFrame(rows).to_markdown(index=False) if rows else "No figures found."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifold", default="outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold.parquet")
    p.add_argument("--metrics", default="outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold_metrics.csv")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/linear_probe_score_manifold")
    p.add_argument("--output", default="outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold_report.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifold_path = Path(args.manifold)
    metrics_path = Path(args.metrics)
    figure_dir = Path(args.figure_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    df = pd.read_parquet(manifold_path, columns=[c for c in ["in_sample", "oof_score", "trajectory_id", "decision_id", "true_q_top_action", "pred_q_top", "top_commitment"] if c])
    n_in_sample = int(df.get("in_sample", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()) if "in_sample" in df.columns else 0
    oof_text = "yes" if n_in_sample == 0 else f"mixed; {n_in_sample} in-sample rows are present and should be excluded from main evidence"
    rows = pd.DataFrame([metrics]).T.reset_index().rename(columns={"index": "metric", 0: "value"})
    lines = [
        "# Linear Probe-Induced Action-Evidence Manifold",
        "",
        "## Summary",
        "",
        "This analysis turns the validated raw-only nested OOF source-score stacker into a low-dimensional supervised action-evidence manifold. The manifold is induced by a linear probe; it is not PCA, not UMAP, and not an unsupervised visualization of raw activations.",
        "",
        "The main coordinates are linear contrasts of the four final stacker scores:",
        "",
        "- `linear_x = centered_score_RIGHT - centered_score_LEFT`",
        "- `linear_y = centered_score_UP - centered_score_DOWN`",
        "",
        "Here `centered_score_ACTION = s_ACTION - mean(s_UP, s_DOWN, s_LEFT, s_RIGHT)`. Because the preferred raw-only stacker uses raw activation-derived source scores, this preserves the strongest linearity claim among the stacker variants.",
        "",
        "## Leakage Status",
        "",
        f"- Main rows are out-of-fold held-out rows: {oof_text}.",
        "- If any in-sample rows are present, the scripts keep the flag and plotting excludes them by default.",
        "- The intended evidence is the nested OOF held-out score plane, not a full-data fitted descriptive plot.",
        "",
        "## Metrics",
        "",
        rows.to_markdown(index=False),
        "",
        "Key values:",
        "",
        f"- balanced accuracy vs true q-top action: {fmt(metrics.get('balanced_accuracy_pred_q_top'))}",
        f"- overall accuracy vs true q-top action: {fmt(metrics.get('accuracy_pred_q_top'))}",
        f"- macro F1: {fmt(metrics.get('macro_f1_pred_q_top'))}",
        f"- pairwise q-order accuracy: {fmt(metrics.get('pairwise_q_order_accuracy'))}",
        f"- branch separation ratio: {fmt(metrics.get('branch_separation_ratio'))}",
        f"- Spearman(score_radius, entropy_commitment): {fmt(metrics.get('spearman_score_radius_entropy_commitment'))}",
        f"- partial Spearman(score_radius, entropy_commitment | progress): {fmt(metrics.get('partial_spearman_score_radius_entropy_commitment_given_progress'))}",
        "",
        "## Figures",
        "",
        figure_table(figure_dir),
        "",
        "## Interpretation Rules",
        "",
        "1. This is a supervised linear probe-induced manifold, not an unsupervised activation manifold.",
        "2. The main plots use nested OOF held-out scores when the score export was generated by script 50 with `--save_predictions`.",
        "3. Four-branch structure should be judged primarily on high-commitment rows; low-commitment rows are expected to be mixed or central.",
        "4. A radius-progress correlation means progress and commitment are partly entangled.",
        "5. Strict pre-action separation means action evidence is recoverable before explicit lexical action tokens, not that the representation causally commits the model.",
        "6. The saved raw-only nested validation crosses 0.7 aggregate balanced accuracy, but not every outer fold crosses 0.7.",
        "",
        "## Files",
        "",
        f"- manifold parquet: `{manifold_path}`",
        f"- metrics CSV: `{metrics_path}`",
        f"- figure directory: `{figure_dir}`",
    ]
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
