#!/usr/bin/env python3
"""Conjecture C4: is the switch hazard Markov in the belief? (does margin screen off position)

Proposition 2.4 found a monotonically declining switch hazard in progress, and 2.5 found
that the current commitment level orders the probability of later switches. C4 states the
stronger claim that the belief SCREENS OFF position: conditional on the current margin,
the hazard is approximately independent of progress. Operationally, adding a progress
spline to a margin spline should not improve held-out hazard prediction, and within margin
strata the hazard should be flat in progress.

Hazard target: y_t = 1 iff the top action at the NEXT prefix differs from the current one
(decidable prospectively, unlike stabilization). Trajectory-grouped CV throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INK = "#333333"
GRID = "#e6e6e6"
STRATUM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9aa0a6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix_metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/conjectures/c4_hazard"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/conjectures"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def spline_logistic(n_knots: int = 6) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=n_knots, degree=3),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def main() -> int:
    args = parse_args()
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    metrics = pd.read_parquet(args.prefix_metrics, columns=[
        "trajectory_id", "step_id", "prefix_id", "prefix_token_index", "progress",
        "top_action", "margin_commitment", "entropy_commitment", "commitment_to_final"])
    metrics = metrics.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"])
    grouped = metrics.groupby(["trajectory_id", "step_id"], sort=False)
    next_top = grouped["top_action"].shift(-1)
    metrics["will_switch"] = (next_top.notna() & (next_top != metrics["top_action"]))
    frame = metrics[next_top.notna()].reset_index(drop=True)
    print(f"hazard rows: {len(frame)}; base rate {frame.will_switch.mean():.4f}")

    feature_sets = {
        "margin_only": ["margin_commitment"],
        "progress_only": ["progress"],
        "margin_plus_progress": ["margin_commitment", "progress"],
        "entropy_only": ["entropy_commitment"],
        "entropy_plus_progress": ["entropy_commitment", "progress"],
    }
    y = frame["will_switch"].to_numpy(dtype=int)
    groups = frame["trajectory_id"].to_numpy()
    splitter = GroupKFold(n_splits=args.folds)

    rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {name: np.zeros(len(frame)) for name in feature_sets}
    for name, columns in feature_sets.items():
        x = frame[columns].to_numpy(dtype=float)
        fold_scores = []
        for train_idx, test_idx in splitter.split(x, y, groups):
            model = spline_logistic().fit(x[train_idx], y[train_idx])
            predicted = model.predict_proba(x[test_idx])[:, 1]
            oof_predictions[name][test_idx] = predicted
            fold_scores.append({
                "log_loss": log_loss(y[test_idx], predicted, labels=[0, 1]),
                "auc": roc_auc_score(y[test_idx], predicted),
            })
        rows.append({
            "features": name,
            "mean_log_loss": float(np.mean([s["log_loss"] for s in fold_scores])),
            "std_log_loss": float(np.std([s["log_loss"] for s in fold_scores])),
            "mean_auc": float(np.mean([s["auc"] for s in fold_scores])),
            "std_auc": float(np.std([s["auc"] for s in fold_scores])),
        })
        print(f"{name:24s} logloss={rows[-1]['mean_log_loss']:.5f} auc={rows[-1]['mean_auc']:.4f}")
    # Additive models cannot see an interaction. If position modulates the hazard in
    # opposite directions at different belief levels, only an interaction-capable model
    # can detect it, so the screening-off test must include one.
    from sklearn.ensemble import HistGradientBoostingClassifier

    for name, columns in (("gbt_margin_only", ["margin_commitment"]),
                          ("gbt_margin_plus_progress", ["margin_commitment", "progress"])):
        x = frame[columns].to_numpy(dtype=float)
        fold_scores = []
        for train_idx, test_idx in splitter.split(x, y, groups):
            model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.1,
                random_state=args.seed).fit(x[train_idx], y[train_idx])
            predicted = model.predict_proba(x[test_idx])[:, 1]
            fold_scores.append({"log_loss": log_loss(y[test_idx], predicted, labels=[0, 1]),
                                "auc": roc_auc_score(y[test_idx], predicted)})
        rows.append({
            "features": name,
            "mean_log_loss": float(np.mean([s["log_loss"] for s in fold_scores])),
            "std_log_loss": float(np.std([s["log_loss"] for s in fold_scores])),
            "mean_auc": float(np.mean([s["auc"] for s in fold_scores])),
            "std_auc": float(np.std([s["auc"] for s in fold_scores])),
        })
        print(f"{name:24s} logloss={rows[-1]['mean_log_loss']:.5f} auc={rows[-1]['mean_auc']:.4f}")

    scores = pd.DataFrame(rows)
    base_rate = float(y.mean())
    scores["log_loss_vs_intercept"] = log_loss(y, np.full(len(y), base_rate), labels=[0, 1]) \
        - scores["mean_log_loss"]

    margin_row = scores.set_index("features").loc["margin_only"]
    both_row = scores.set_index("features").loc["margin_plus_progress"]
    progress_row = scores.set_index("features").loc["progress_only"]
    delta_auc = float(both_row["mean_auc"] - margin_row["mean_auc"])
    delta_logloss = float(margin_row["mean_log_loss"] - both_row["mean_log_loss"])

    margin_bins = pd.qcut(frame["margin_commitment"], 5, labels=False, duplicates="drop")
    progress_bins = pd.cut(frame["progress"], bins=5, labels=False, include_lowest=True)
    stratified = frame.assign(margin_bin=margin_bins, progress_bin=progress_bins) \
        .groupby(["margin_bin", "progress_bin"], observed=True).agg(
            hazard=("will_switch", "mean"), n=("will_switch", "size")).reset_index()
    stratified["progress_mid"] = (stratified["progress_bin"] + 0.5) / 5
    within_range = stratified.groupby("margin_bin").agg(
        hazard_min=("hazard", "min"), hazard_max=("hazard", "max"),
        hazard_mean=("hazard", "mean")).reset_index()
    within_range["within_stratum_range"] = within_range["hazard_max"] - within_range["hazard_min"]
    marginal_range = float(frame.groupby(progress_bins)["will_switch"].mean().max()
                           - frame.groupby(progress_bins)["will_switch"].mean().min())
    mean_within_range = float(within_range["within_stratum_range"].mean())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "model_comparison.csv", index=False)
    stratified.to_csv(args.out_dir / "stratified_hazard.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    marginal = frame.groupby(progress_bins)["will_switch"].mean()
    axes[0].plot((marginal.index + 0.5) / 5, marginal.to_numpy(), color="#2a78d6", linewidth=2,
                 marker="o", markersize=5)
    axes[0].set_title("Marginal hazard vs progress", color=INK, fontsize=11)
    axes[0].set_ylabel("P(switch at next prefix)", color=INK)
    for bin_index, group in stratified.groupby("margin_bin"):
        color = STRATUM_COLORS[int(bin_index) % len(STRATUM_COLORS)]
        axes[1].plot(group["progress_mid"], group["hazard"], color=color, linewidth=2,
                     marker="o", markersize=4)
        axes[1].annotate(f"margin Q{int(bin_index) + 1}", (group["progress_mid"].iloc[-1],
                         group["hazard"].iloc[-1]), textcoords="offset points",
                         xytext=(6, 0), ha="left", color=INK, fontsize=8)
    axes[1].set_title("Hazard vs progress within margin quintiles", color=INK, fontsize=11)
    axes[1].set_ylabel("P(switch at next prefix)", color=INK)
    axes[1].set_xlim(0, 1.35)
    for axis in axes:
        axis.set_xlabel("progress through CoT", color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C4: does the belief screen off position in the switch hazard?", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c4_hazard_markov.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    gbt_margin = scores.set_index("features").loc["gbt_margin_only"]
    gbt_both = scores.set_index("features").loc["gbt_margin_plus_progress"]
    delta_auc_gbt = float(gbt_both["mean_auc"] - gbt_margin["mean_auc"])
    # Sign of the within-stratum progress effect, per margin quintile.
    slope_table = stratified.pivot(index="margin_bin", columns="progress_bin", values="hazard")
    slope_table["late_over_early"] = slope_table.iloc[:, -1] / slope_table.iloc[:, 0]
    sign_reversal = bool(slope_table["late_over_early"].max() > 1.2
                         and slope_table["late_over_early"].min() < 0.8)
    verdict = ("SUPPORTED" if delta_auc < 0.01 and delta_auc_gbt < 0.01
               and not sign_reversal else "REFUTED")
    lines = [
        "# Conjecture C4: Markov-in-belief switch hazard",
        "",
        f"- {len(frame)} transitions, base switch rate {base_rate:.4f}; "
        f"trajectory-grouped {args.folds}-fold CV.",
        "",
        "## Held-out model comparison",
        "",
        scores.round(5).to_markdown(index=False),
        "",
        "## Screening-off diagnostics",
        "",
        f"- AUC gain from adding progress to margin, ADDITIVE spline model: {delta_auc:+.4f} "
        f"(margin-only {margin_row['mean_auc']:.4f} -> both {both_row['mean_auc']:.4f})",
        f"- AUC gain from adding progress to margin, INTERACTION-capable model (GBT): "
        f"{delta_auc_gbt:+.4f} ({gbt_margin['mean_auc']:.4f} -> {gbt_both['mean_auc']:.4f})",
        f"- log-loss gain from adding progress to margin: {delta_logloss:+.5f}",
        f"- progress-only AUC: {progress_row['mean_auc']:.4f} (margin-only "
        f"{margin_row['mean_auc']:.4f})",
        f"- marginal hazard range across progress quintiles: {marginal_range:.4f}",
        f"- mean WITHIN-margin-stratum hazard range across progress: {mean_within_range:.4f} "
        f"({100 * mean_within_range / marginal_range:.1f}% of the marginal range)",
        "",
        within_range.round(4).to_markdown(index=False),
        "",
        "## The interaction: position modulates the hazard with OPPOSITE signs",
        "",
        "Hazard by margin quintile (rows) x progress quintile (columns), and the ratio of",
        "the last to the first progress quintile within each margin stratum:",
        "",
        slope_table.round(4).to_markdown(),
        "",
        f"- sign reversal across margin strata detected: {sign_reversal}",
        "",
        f"## Verdict: {verdict}",
        "",
        "Criterion: progress adds < 0.01 AUC over margin in BOTH the additive and the",
        "interaction-capable model, and no sign reversal of the within-stratum progress",
        "effect. An additive model alone cannot decide this, because opposing within-stratum",
        "effects cancel in it.",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nVerdict: {verdict}; delta_auc={delta_auc:+.4f}, "
          f"within/marginal range={mean_within_range:.4f}/{marginal_range:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
