#!/usr/bin/env python3
"""Three clocks per decision step: probe-decoded vs forced-answer vs verbalization (D3).

For fold-0 held-out trajectories, the trained probe's per-prefix predictions
(`pred_commitment`, `pred_q_*`) define an activation-side clock: when does the DECODED
state settle? Compare, per decision step, against the forced-answer behavioral clock
(argmax stabilization of extracted q) and the verbalization clock (first action mention).
All three clocks are measured on the same held-out steps; the probe never saw these
trajectories during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PRED_Q = [f"pred_q_{action}" for action in ("UP", "DOWN", "LEFT", "RIGHT")]

INK = "#333333"
GRID = "#e6e6e6"
CLOCK_COLORS = {
    "probe argmax stabilization": "#4a3aa7",
    "forced-answer argmax stabilization": "#2a78d6",
    "first verbal action mention": "#1baf7a",
    "probe commitment >= 0.8 (permanent)": "#eb6834",
}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=base / "fold0_20260721/fold_artifacts/split0/predictions_mean_layers_mlp.parquet")
    parser.add_argument("--method", type=str, default=None,
                        help="If the predictions file pools methods (oof_predictions), keep only this one.")
    parser.add_argument("--split_ids", type=str, default=None,
                        help="Comma-separated split_ids to keep (e.g. '0,1,4' for folds with a live q head).")
    parser.add_argument("--behavior_steps", type=Path,
                        default=REPO_ROOT / "outputs/commitment_boundary_steps.parquet")
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/three_clocks")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    return parser.parse_args()


def permanent_cross(values: np.ndarray, progress: np.ndarray, threshold: float) -> float | None:
    suffix_min = np.minimum.accumulate(values[::-1])[::-1]
    hits = np.nonzero(suffix_min >= threshold)[0]
    return float(progress[hits[0]]) if len(hits) else None


def probe_clocks(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = predictions.sort_values(["decision_id", "prefix_token_index", "prefix_id"]) \
        .groupby("decision_id", sort=False)
    for decision_id, group in grouped:
        progress = group["progress"].to_numpy(dtype=float)
        pred_q = group[PRED_Q].to_numpy(dtype=float)
        pred_top = pred_q.argmax(axis=1)
        final_pred_top = pred_top[-1]
        differs = pred_top != final_pred_top
        last_differ = int(np.max(np.nonzero(differs)[0])) if differs.any() else -1
        stab_index = last_differ + 1
        # Decoder noise biases raw stabilization late; a centered rolling mean is the
        # standard mitigation and is applied to predictions only, never to the truth.
        pred_q_smooth = pd.DataFrame(pred_q).rolling(5, center=True, min_periods=1).mean().to_numpy()
        smooth_top = pred_q_smooth.argmax(axis=1)
        smooth_differs = smooth_top != smooth_top[-1]
        smooth_last = int(np.max(np.nonzero(smooth_differs)[0])) if smooth_differs.any() else -1
        pred_commitment = group["pred_commitment"].to_numpy(dtype=float)
        pred_commitment_smooth = pd.Series(pred_commitment).rolling(
            5, center=True, min_periods=1).mean().to_numpy()
        true_commitment = group["entropy_commitment"].to_numpy(dtype=float)
        trajectory_id = str(group["trajectory_id"].iloc[0])
        step_id = int(str(decision_id).rsplit("::", 1)[1])
        step_spearman = (pd.Series(pred_commitment).corr(pd.Series(true_commitment), method="spearman")
                         if len(group) >= 3 else np.nan)
        rows.append({
            "trajectory_id": trajectory_id,
            "step_id": step_id,
            "n_prefixes": int(len(group)),
            "probe_stabilization_progress": float(progress[stab_index]),
            "probe_stabilization_smooth_progress": float(progress[smooth_last + 1]),
            "probe_permanent_0_8_progress": permanent_cross(pred_commitment, progress, 0.8),
            "probe_permanent_0_8_smooth_progress": permanent_cross(pred_commitment_smooth, progress, 0.8),
            "probe_final_top_matches_truth": bool(
                final_pred_top == group[[c.replace("pred_", "") for c in PRED_Q]]
                .to_numpy(dtype=float)[-1].argmax()),
            "step_commitment_spearman": float(step_spearman) if pd.notna(step_spearman) else np.nan,
        })
    return pd.DataFrame(rows)


def quantile_line(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "NA"
    return (f"median {clean.median():.4f}, IQR [{clean.quantile(0.25):.4f}, {clean.quantile(0.75):.4f}], "
            f"mean {clean.mean():.4f}, n={len(clean)}")


def main() -> int:
    args = parse_args()
    from scipy.stats import spearmanr

    predictions = pd.read_parquet(args.predictions)
    if args.method is not None:
        predictions = predictions[predictions["method"] == args.method]
    if args.split_ids is not None:
        keep = {int(token) for token in args.split_ids.split(",") if token.strip()}
        predictions = predictions[predictions["split_id"].astype(int).isin(keep)]
    if predictions.empty:
        raise SystemExit("no prediction rows left after method/split filtering")
    behavior = pd.read_parquet(args.behavior_steps)
    probe = probe_clocks(predictions)
    merged = probe.merge(
        behavior[["trajectory_id", "step_id", "stabilization_progress", "permanent_cross_0_8_progress",
                  "first_mention_progress", "num_cot_tokens", "agent_action"]],
        on=["trajectory_id", "step_id"], how="inner", validate="one_to_one")
    if len(merged) != len(probe):
        raise SystemExit(f"join lost steps: {len(probe)} probe vs {len(merged)} merged")

    merged["probe_minus_behavior_stab"] = (merged["probe_stabilization_progress"]
                                           - merged["stabilization_progress"])
    merged["probe_minus_behavior_stab_smooth"] = (merged["probe_stabilization_smooth_progress"]
                                                  - merged["stabilization_progress"])
    merged["probe_stab_before_mention"] = (merged["probe_stabilization_progress"]
                                           < merged["first_mention_progress"])
    merged["behavior_stab_before_mention"] = (merged["stabilization_progress"]
                                              < merged["first_mention_progress"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out_dir / "three_clocks_steps.parquet", engine="pyarrow", index=False)

    clock_corr = spearmanr(merged["probe_stabilization_progress"],
                           merged["stabilization_progress"]).statistic
    clock_corr_perm = spearmanr(merged["probe_permanent_0_8_progress"].fillna(1.0),
                                merged["permanent_cross_0_8_progress"].fillna(1.0)).statistic

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    clock_values = {
        "probe argmax stabilization": merged["probe_stabilization_progress"].dropna(),
        "forced-answer argmax stabilization": merged["stabilization_progress"].dropna(),
        "first verbal action mention": merged["first_mention_progress"].dropna(),
        "probe commitment >= 0.8 (permanent)": merged["probe_permanent_0_8_progress"].dropna(),
    }
    for label, values in clock_values.items():
        ordered = np.sort(values.to_numpy())
        cdf = np.arange(1, len(ordered) + 1) / len(merged)
        axes[0].plot(ordered, cdf, color=CLOCK_COLORS[label], linewidth=2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("progress through CoT", color=INK)
    axes[0].set_ylabel("fraction of held-out decision steps", color=INK)
    axes[0].set_title("Clock CDFs on fold-0 held-out steps", color=INK, fontsize=11)
    axes[0].legend(handles=[plt.Line2D([], [], color=color, linewidth=2, label=label)
                            for label, color in CLOCK_COLORS.items()],
                   loc="lower right", frameon=False, fontsize=8, labelcolor=INK)

    axes[1].scatter(merged["stabilization_progress"], merged["probe_stabilization_progress"],
                    s=18, color="#2a78d6", alpha=0.55, edgecolors="white", linewidths=0.5)
    axes[1].plot([0, 1], [0, 1], color=INK, linewidth=1.2, linestyle=":")
    axes[1].set_xlabel("forced-answer stabilization progress", color=INK)
    axes[1].set_ylabel("probe stabilization progress", color=INK)
    axes[1].set_title(f"Per-step agreement (Spearman {clock_corr:.3f})", color=INK, fontsize=11)
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure_path = args.figure_dir / "three_clocks_lead_lag.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    lines = [
        "# Three Clocks: probe-decoded vs forced-answer vs verbalization (fold-0 held-out)",
        "",
        f"- Probe: `{args.predictions.name}` (mean_layers_mlp, trained on folds 1-4 trajectories only).",
        f"- Steps: {len(merged)} over {merged.trajectory_id.nunique()} held-out trajectories.",
        "",
        "## Clock distributions (progress units)",
        "",
        f"- probe argmax stabilization: {quantile_line(merged['probe_stabilization_progress'])}",
        f"- probe argmax stabilization (smoothed, window 5): "
        f"{quantile_line(merged['probe_stabilization_smooth_progress'])}",
        f"- forced-answer argmax stabilization: {quantile_line(merged['stabilization_progress'])}",
        f"- probe permanent commitment >= 0.8: {quantile_line(merged['probe_permanent_0_8_progress'])}",
        f"- probe permanent commitment >= 0.8 (smoothed): "
        f"{quantile_line(merged['probe_permanent_0_8_smooth_progress'])}",
        f"- forced-answer permanent commitment >= 0.8: {quantile_line(merged['permanent_cross_0_8_progress'])}",
        f"- first verbal action mention: {quantile_line(merged['first_mention_progress'])}",
        "",
        "## Lead / lag",
        "",
        f"- probe minus forced-answer stabilization: {quantile_line(merged['probe_minus_behavior_stab'])}",
        f"- probe minus forced-answer stabilization (smoothed): "
        f"{quantile_line(merged['probe_minus_behavior_stab_smooth'])}",
        f"- fraction probe stabilizes BEFORE forced answer: "
        f"{(merged['probe_minus_behavior_stab'] < 0).mean():.4f} "
        f"(smoothed: {(merged['probe_minus_behavior_stab_smooth'] < 0).mean():.4f})",
        f"- fraction probe stabilizes before first mention: {merged['probe_stab_before_mention'].mean():.4f} "
        f"(forced answer: {merged['behavior_stab_before_mention'].mean():.4f})",
        f"- Spearman(probe stabilization, forced-answer stabilization) across steps: {clock_corr:.4f}",
        f"- Spearman of permanent-0.8 clocks across steps: {clock_corr_perm:.4f}",
        "",
        "## Probe quality context on these steps",
        "",
        f"- probe final-prefix top action matches truth: {merged['probe_final_top_matches_truth'].mean():.4f}",
        f"- per-step commitment-curve Spearman: {quantile_line(merged['step_commitment_spearman'])}",
        "",
        "## Figure",
        "",
        f"- `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'three_clocks_steps.parquet'} ({len(merged)} steps)")
    print(f"Wrote {args.out_dir / 'report.md'} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
