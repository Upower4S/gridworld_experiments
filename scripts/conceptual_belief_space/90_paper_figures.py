#!/usr/bin/env python3
"""Paper-ready versions of four figures.

The experiment scripts title their figures with internal conjecture identifiers ("C4:",
"C5:", "C8:") and pack two panels into plots that are then printed at half column width.
Neither is appropriate in a paper, where the caption carries the message and the panel has
to be legible at print size. This regenerates those four from the same stored artifacts
with captions removed, fonts sized for print, and one panel per point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
NEUTRAL, INK, MUTED, GRID = "#9aa0a6", "#222222", "#5f6368", "#e0e0e0"
QUINTILE_COLORS = [ORANGE, PURPLE, NEUTRAL, GREEN, BLUE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "paper/figures")
    return parser.parse_args()


def style(axis, *, xlabel="", ylabel="", title=""):
    if xlabel:
        axis.set_xlabel(xlabel, color=INK, fontsize=11)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK, fontsize=11)
    if title:
        axis.set_title(title, color=INK, fontsize=11.5)
    axis.tick_params(colors=INK, labelsize=10)
    axis.grid(True, color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def anchors_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    transitions = pd.read_parquet(REPO_ROOT / "outputs/belief_update_anchors.parquet",
                                  columns=["tv", "transition_progress", "is_anchor"])
    ordered = np.sort(transitions["tv"].to_numpy())
    survival = 1.0 - np.arange(1, len(ordered) + 1) / len(ordered)
    threshold = float(transitions.loc[transitions.is_anchor, "tv"].min())

    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    axis.plot(ordered, np.maximum(survival, 1e-6), color=BLUE, linewidth=2)
    axis.set_yscale("log")
    axis.axvline(threshold, color=INK, linewidth=1.2, linestyle=":")
    axis.annotate(f"top 5% of transitions\n(carry 37% of all belief change)",
                  (threshold, 2e-2), xytext=(8, 0), textcoords="offset points",
                  color=INK, fontsize=9.5, va="center")
    style(axis, xlabel="belief change at one transition (total variation)",
          ylabel="fraction of transitions exceeding")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def hazard_interaction_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    stratified = pd.read_csv(REPO_ROOT / "outputs/conjectures/c4_hazard/stratified_hazard.csv")
    figure, axis = plt.subplots(figsize=(5.8, 3.9))
    for quintile, group in stratified.groupby("margin_bin"):
        group = group.sort_values("progress_mid")
        color = QUINTILE_COLORS[int(quintile) % len(QUINTILE_COLORS)]
        axis.plot(group["progress_mid"], group["hazard"], color=color, linewidth=2,
                  marker="o", markersize=4.5)
        label = ("weakest belief (Q1)" if quintile == 0
                 else "strongest belief (Q5)" if quintile == 4 else f"Q{int(quintile) + 1}")
        axis.annotate(label, (group["progress_mid"].iloc[-1], group["hazard"].iloc[-1]),
                      xytext=(7, 0), textcoords="offset points", color=INK, fontsize=9.5,
                      va="center")
    axis.set_xlim(0.05, 1.42)
    style(axis, xlabel="progress through the chain of thought",
          ylabel="P(top action changes at next cut)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def tail_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    exits = pd.read_csv(REPO_ROOT / "outputs/conjectures/c5_tail/early_exit.csv")
    lexicon = pd.read_csv(REPO_ROOT / "outputs/conjectures/c5_tail/head_tail_lexicon.csv")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    axes[0].scatter(exits.loc[~exits.on_frontier, "mean_token_savings"],
                    exits.loc[~exits.on_frontier, "accuracy"],
                    s=20, color=NEUTRAL, alpha=0.55, edgecolors="white", linewidths=0.4)
    frontier = exits[exits.on_frontier].sort_values("mean_token_savings")
    axes[0].plot(frontier["mean_token_savings"], frontier["accuracy"], color=BLUE,
                 linewidth=2, marker="o", markersize=5)
    axes[0].axhline(0.95, color=INK, linewidth=1.2, linestyle=":")
    axes[0].annotate("95% agreement (never reached)", (0.30, 0.955), color=INK, fontsize=9.5)
    best = frontier.iloc[frontier["accuracy"].argmax()]
    axes[0].annotate(f"best online rule:\n{best.accuracy:.3f} at {best.mean_token_savings:.0%} saved",
                     (best.mean_token_savings, best.accuracy), xytext=(14, -30),
                     textcoords="offset points", color=INK, fontsize=9.5,
                     arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    axes[0].set_ylim(0.38, 1.0)
    style(axes[0], xlabel="fraction of tokens saved",
          ylabel="agreement with full-chain action")

    order = ["verification", "action_word", "plan", "correction", "spatial"]
    lexicon = lexicon.set_index("lexicon").loc[order]
    positions = np.arange(len(order))
    width = 0.38
    axes[1].bar(positions - width / 2, lexicon["head_density"], width, color=NEUTRAL,
                label="before the answer settles")
    axes[1].bar(positions + width / 2, lexicon["tail_density"], width, color=BLUE,
                label="after the answer settles")
    for position, row in zip(positions, lexicon.itertuples()):
        axes[1].annotate(f"{row.tail_over_head:.2f}x", (position + width / 2, row.tail_density),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         color=INK, fontsize=9)
    axes[1].set_xticks(positions, [name.replace("_", " ") for name in order],
                       rotation=18, ha="right", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9.5, labelcolor=INK)
    style(axes[1], ylabel="marker density (per 1000 chars)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def transfer_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    results = pd.read_csv(REPO_ROOT / "outputs/conjectures/c8_transfer/transfer.csv")
    bar = float(results["position_bar_spearman"].iloc[0])
    figure, axis = plt.subplots(figsize=(5.8, 3.9))
    axis.plot(results["layer"], results["task_refit_ceiling_spearman"], color=NEUTRAL,
              linewidth=2, marker="o", markersize=5, linestyle="--")
    axis.plot(results["layer"], results["frozen_transfer_spearman"], color=BLUE,
              linewidth=2.2, marker="o", markersize=5.5)
    axis.axhline(bar, color=INK, linewidth=1.2, linestyle=":")
    axis.axhline(0.0, color=MUTED, linewidth=0.9)
    axis.annotate("refit on the new task (ceiling)", (results["layer"].iloc[2],
                  results["task_refit_ceiling_spearman"].iloc[2]), xytext=(-4, 12),
                  textcoords="offset points", color=MUTED, fontsize=9.5)
    axis.annotate("frozen gridworld direction", (results["layer"].iloc[2],
                  results["frozen_transfer_spearman"].iloc[2]), xytext=(-2, -20),
                  textcoords="offset points", color=BLUE, fontsize=9.5)
    axis.annotate("position-only baseline", (results["layer"].iloc[-1], bar),
                  xytext=(-2, 6), textcoords="offset points", ha="right",
                  color=INK, fontsize=9.5)
    style(axis, xlabel="layer (block output)",
          ylabel="Spearman with new-task commitment")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_figure(args.out_dir / "anchors.png")
    hazard_interaction_figure(args.out_dir / "hazard_interaction.png")
    tail_figure(args.out_dir / "tail.png")
    transfer_figure(args.out_dir / "transfer.png")
    for name in ("anchors", "hazard_interaction", "tail", "transfer"):
        print(f"Wrote {args.out_dir / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
