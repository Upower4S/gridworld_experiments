"""Redraw the coarse-4 sentence-function figure from the saved statistics (script 22).

The original `figures/c5_corrected/coarse4_functions.png` (generating script lost) was
judged cryptic by a reviewer: unlabelled arrows, an x-axis that did not correspond to the
annotated values. This version uses only the artifacts of the coarse-4 run:
  outputs/conjectures/c5_sentence_functions_coarse4/function_stats.csv   (head/tail shares, diff CI)
  outputs/conjectures/c5_sentence_functions_coarse4/event_aligned.csv    (share by 400-char bin)
  outputs/conjectures/c5_sentence_functions_coarse4/placebo_jumps.csv    (jump vs placebo)

Usage (repo root): interp/.venv/bin/python scripts/22_redraw_coarse4_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402  (repo convention)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["world-reading", "solving", "meta-checking", "concluding/answering"]
COLORS = {"world-reading": "#d95f02", "solving": "#1f77b4",
          "meta-checking": "#7f7f7f", "concluding/answering": "#2ca02c"}
BIN_CHARS = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_dir", default="outputs/conjectures/c5_sentence_functions_coarse4")
    parser.add_argument("--figure", default="figures/c5_corrected/coarse4_functions_v2.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = REPO_ROOT / args.stats_dir
    stats = pd.read_csv(d / "function_stats.csv").set_index("function").loc[ORDER]
    aligned = pd.read_csv(d / "event_aligned.csv")
    placebo = pd.read_csv(d / "placebo_jumps.csv").set_index("function").loc[ORDER]

    fig = plt.figure(figsize=(13.5, 7.8), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.6, wspace=0.42)

    # ---- top: paired before/after bars
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(ORDER))
    w = 0.36
    for i, f in enumerate(ORDER):
        c = COLORS[f]
        hatch = "//" if f == "meta-checking" else None
        ax.bar(x[i] - w / 2, stats.loc[f, "head_share"], w, color=c, alpha=0.45,
               edgecolor=c, hatch=hatch, label="before commitment" if i == 0 else None)
        ax.bar(x[i] + w / 2, stats.loc[f, "tail_share"], w, color=c, alpha=0.95,
               edgecolor=c, hatch=hatch, label="after commitment" if i == 0 else None)
        top = max(stats.loc[f, "head_share"], stats.loc[f, "tail_share"])
        ax.text(x[i], top + 0.03,
                f"{stats.loc[f, 'diff']:+.2f}  [{stats.loc[f, 'ci_lo']:+.2f}, {stats.loc[f, 'ci_hi']:+.2f}]",
                ha="center", va="bottom", fontsize=9.5)
    labels = [f + ("\n(exploratory, precision 0.32)" if f == "meta-checking" else "") for f in ORDER]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("share of sentences")
    ax.set_title("Share of each sentence function before vs after the commitment prefix "
                 "(difference with 95% trajectory-bootstrap interval)", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- bottom: event-aligned shares
    for i, f in enumerate(ORDER):
        axi = fig.add_subplot(gs[1, i])
        sub = aligned[aligned["function"] == f].sort_values("offset")
        centers = (sub["offset"].to_numpy() + 0.5) * BIN_CHARS
        c = COLORS[f]
        axi.fill_between(centers, sub["ci_lo"], sub["ci_hi"], color=c, alpha=0.18, linewidth=0)
        axi.plot(centers, sub["mean_share"], "-o", color=c, linewidth=2, markersize=4)
        axi.axvline(0, color="#333333", linestyle="--", linewidth=1)
        pj = placebo.loc[f]
        axi.set_title(f"{f}\nchange at commitment {pj['real_jump']:+.2f}\n"
                      f"placebo [{pj['placebo_lo']:+.2f}, {pj['placebo_hi']:+.2f}], p = {pj['p_two_sided']:.2f}",
                      fontsize=8.5)
        if i == 1:
            axi.set_xlabel("characters relative to the commitment prefix (400-character bins)", fontsize=9.5, x=1.1)
        if i == 0:
            axi.set_ylabel("share of sentences")
        axi.grid(True, alpha=0.3)
        for s in ("top", "right"):
            axi.spines[s].set_visible(False)

    out = REPO_ROOT / args.figure
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
