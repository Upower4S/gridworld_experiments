#!/usr/bin/env python3
"""Paper figure: intervention levels (scripts 20, 25, 28).

Left: flip-to-runner-up rate by commitment bin for text injections at two levels — conclusion
statements (original counter template + two alternative wordings) and state facts naming no
action (false wall fact, open-cell fact, true-fact control) — plus the neutral sentence.
Right: flip rate for the sentence-activation transplant by what is written (layers/positions),
against the counter text and the neutral recipient. Bars: Wilson 95% intervals.

Usage: interp/.venv/bin/python scripts/29_plot_intervention_levels.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BINS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95-1"]
BIN_X = [0.15, 0.4, 0.6, 0.775, 0.9, 0.975]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def flips(frame: pd.DataFrame, target: str) -> pd.Series:
    return frame["top_after"] == frame[target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", default="figures/followups/intervention_levels.png")
    args = parser.parse_args()
    inj = pd.read_parquet(REPO_ROOT / "outputs/followups/commitment_gated_injection.parquet")
    bel = pd.read_parquet(REPO_ROOT / "outputs/followups/belief_level_injection.parquet")
    act = pd.read_parquet(REPO_ROOT / "outputs/followups/activation_sentence_transplant.parquet")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
    series = [
        (inj[inj.condition == "counter"], "target_action", "conclusion: original", "#b7352d", "-", "o"),
        (bel[bel.condition == "counter_alt1"], "runner_up", "conclusion: wording 2", "#d9776f", "-", "s"),
        (bel[bel.condition == "counter_alt2"], "runner_up", "conclusion: wording 3", "#e8a9a3", "-", "^"),
        (bel[bel.condition == "belief_block_top"], "runner_up", "state fact: top blocked (false)", "#215caf", "--", "o"),
        (bel[bel.condition == "belief_open_runnerup"], "runner_up", "state fact: runner-up open", "#5b8fd6", "--", "s"),
        (bel[bel.condition == "belief_true_top"], "runner_up", "state fact: true control", "#9dbde8", "--", "^"),
        (inj[inj.condition == "neutral"], "target_action", "neutral sentence", "#777777", ":", "d"),
    ]
    for frame, target, label, color, dash, marker in series:
        ys, lo, hi = [], [], []
        for b in BINS:
            sub = frame[frame["bin"] == b]
            k, n = int(flips(sub, target).sum()), len(sub)
            ys.append(k / n)
            a, c = wilson(k, n)
            lo.append(ys[-1] - a)
            hi.append(c - ys[-1])
        left.errorbar(BIN_X, ys, yerr=[lo, hi], color=color, linestyle=dash, marker=marker, markersize=5,
                      linewidth=1.8, capsize=2.5, label=label)
    left.set_xlabel("commitment to top action at the cut")
    left.set_ylabel("P(forced top action = injected runner-up)")
    left.set_title("Text injections: conclusions vs. state facts")
    left.set_ylim(-0.02, 1.05)
    left.set_xlim(0.05, 1.03)
    left.legend(fontsize=8, frameon=False, loc="center left")
    for ax in (left, right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.7)

    order = [("donor_text", "counter sentence as text"), ("all_layers_full", "all layers, whole sentence"),
             ("all_layers_last6", "all layers, last 6 tokens"), ("all_layers_last1", "all layers, last token"),
             ("layer_10", "layer 10 only"), ("layer_15", "layer 15 only"), ("layer_18", "layer 18 only"),
             ("late_16-23", "layers 16-23"), ("recipient_text", "neutral sentence (recipient)")]
    ys, err, labels, colors = [], [], [], []
    for cond, label in order:
        sub = act[act.condition == cond]
        k, n = int(flips(sub, "runner_up").sum()), len(sub)
        p = k / n
        a, c = wilson(k, n)
        ys.append(p)
        err.append([p - a, c - p])
        labels.append(label)
        colors.append("#b7352d" if cond == "donor_text" else "#777777" if cond == "recipient_text" else "#215caf")
    y = np.arange(len(order))[::-1]
    right.barh(y, ys, xerr=np.array(err).T, color=colors, alpha=0.85, capsize=2.5, height=0.65)
    right.set_yticks(y)
    right.set_yticklabels(labels, fontsize=8.5)
    right.set_xlim(0, 1.0)
    right.set_xlabel("P(forced top action = runner-up), 240 cuts")
    right.set_title("Sentence-activation transplant into a neutral run")
    fig.tight_layout()
    out = REPO_ROOT / args.figure
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
