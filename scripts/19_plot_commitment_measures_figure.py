"""Paper figure: commitment measures along the CoT, on faithful extracted labels.

Two panels from `outputs/prefix_commitment_metrics_extracted.parquet` (the canonical
post-rebase table; never point this at stored stage-03 labels):

  left  - mean commitment_to_final and mean entropy_commitment vs progress, for all
          prefixes (solid) and the strict pre-verbal subset (dashed), with empty-prefix
          (prompt-only) baselines marked at progress 0 and the uniform prior at 0.25;
  right - per-eventual-action mean commitment_to_final vs progress, with empty-prefix
          per-action baselines.

Bands are 95% trajectory-clustered bootstrap intervals (resampling source_file).

Usage (repo root):
    interp/.venv/bin/python scripts/19_plot_commitment_measures_figure.py
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

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {"UP": "#1f77b4", "DOWN": "#ff7f0e", "LEFT": "#2ca02c", "RIGHT": "#5e3c99"}
COMMIT_COLOR = "#215caf"
ENTROPY_COLOR = "#b7352d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/prefix_commitment_metrics_extracted.parquet",
        help="canonical extracted-label prefix metrics parquet",
    )
    parser.add_argument(
        "--figure",
        default="figures/behavior_extracted/commitment_measures_panel.png",
        help="output PNG path",
    )
    parser.add_argument(
        "--report",
        default="outputs/commitment_measures_figure_report.md",
        help="markdown report with the plotted values",
    )
    parser.add_argument(
        "--preverbal_mask",
        default=None,
        help="optional parquet with prefix_id + a boolean column selecting the pre-verbal "
             "subset (default: the table's strict_before_any_action_mention flag)",
    )
    parser.add_argument("--preverbal_column", default="preverbal_decisional")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def binned_cluster_stats(
    frame: pd.DataFrame,
    value_column: str,
    n_bins: int,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean with a 95% cluster bootstrap over trajectories (source_file)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(frame["progress"].to_numpy(), edges) - 1, 0, n_bins - 1)
    codes, _ = pd.factorize(frame["source_file"])
    n_clusters = codes.max() + 1
    values = frame[value_column].to_numpy(dtype=np.float64)

    flat = codes * n_bins + bin_index
    size = n_clusters * n_bins
    sums = np.bincount(flat, weights=values, minlength=size).reshape(n_clusters, n_bins)
    counts = np.bincount(flat, minlength=size).reshape(n_clusters, n_bins)

    total_counts = counts.sum(axis=0)
    means = np.divide(
        sums.sum(axis=0), total_counts, out=np.full(n_bins, np.nan), where=total_counts > 0
    )

    boot = np.full((n_boot, n_bins), np.nan)
    for rep in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        c = counts[draw].sum(axis=0)
        boot[rep] = np.divide(
            sums[draw].sum(axis=0), c, out=np.full(n_bins, np.nan), where=c > 0
        )
    low = np.nanpercentile(boot, 2.5, axis=0)
    high = np.nanpercentile(boot, 97.5, axis=0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, means, low, high


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, alpha=0.3, linewidth=0.7)
    axis.set_xlim(-0.045, 1.0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    table = pd.read_parquet(REPO_ROOT / args.input)
    if not {"commitment_to_final", "entropy_commitment", "progress", "source_file"} <= set(
        table.columns
    ):
        raise SystemExit("input is missing required columns; is this the extracted table?")

    if args.preverbal_mask:
        mask = pd.read_parquet(REPO_ROOT / args.preverbal_mask, columns=["prefix_id", args.preverbal_column])
        table = table.merge(mask, on="prefix_id", how="left")
        strict = table[table[args.preverbal_column].fillna(False).astype(bool)]
    else:
        strict = table[table["strict_before_any_action_mention"]]
    first = table.sort_values("prefix_token_index").groupby(
        ["source_file", "step_id"], as_index=False
    ).first()

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=200)
    report_lines = [
        "# Commitment measures figure (script 19)",
        "",
        f"- Input: `{args.input}`; {table['source_file'].nunique()} trajectory files, "
        f"{len(first)} steps, {len(table)} prefixes; strict pre-verbal rows: {len(strict)}.",
        f"- Bins: {args.bins}; bootstrap: {args.bootstrap} trajectory resamples; "
        f"seed {args.seed}.",
        "",
    ]

    series = [
        (table, "commitment_to_final", COMMIT_COLOR, "-", "commitment to eventual action"),
        (table, "entropy_commitment", ENTROPY_COLOR, "-", "entropy commitment"),
        (strict, "commitment_to_final", COMMIT_COLOR, "--", "commitment, pre-verbal only"),
        (strict, "entropy_commitment", ENTROPY_COLOR, "--", "entropy comm., pre-verbal only"),
    ]
    for frame, column, color, dash, label in series:
        centers, means, low, high = binned_cluster_stats(
            frame, column, args.bins, rng, args.bootstrap
        )
        left.fill_between(centers, low, high, color=color, alpha=0.15, linewidth=0)
        left.plot(centers, means, dash, color=color, linewidth=2.0, label=label)
        report_lines.append(
            f"- {label}: " + ", ".join(f"{m:.3f}" for m in means)
        )

    prompt_commit = float(first["commitment_to_final"].mean())
    prompt_entropy = float(first["entropy_commitment"].mean())
    left.plot([0.0], [prompt_commit], "s", color=COMMIT_COLOR, markersize=7,
              markerfacecolor="white", markeredgewidth=1.8, zorder=5)
    left.plot([0.0], [prompt_entropy], "s", color=ENTROPY_COLOR, markersize=7,
              markerfacecolor="white", markeredgewidth=1.8, zorder=5)
    left.annotate("prompt only", (0.0, prompt_entropy), textcoords="offset points",
                  xytext=(8, -27), fontsize=9, color="#444444")
    left.axhline(0.25, color="#777777", linestyle=":", linewidth=1.2)
    left.annotate("uniform prior 0.25", (0.995, 0.25), textcoords="offset points",
                  xytext=(-2, -13), fontsize=9, color="#777777", ha="right")
    left.set_ylim(0.0, 1.02)
    left.set_xlabel("progress through CoT")
    left.set_ylabel("mean value")
    left.set_title("Commitment measures along the CoT")
    left.legend(loc="upper left", fontsize=9, frameon=False)
    style_axis(left)

    report_lines += [
        "",
        f"- Empty-prefix (prompt-only) means: commitment {prompt_commit:.4f}, "
        f"entropy commitment {prompt_entropy:.4f}.",
        "",
        "## Per-action (right panel)",
        "",
    ]

    for action in ACTIONS:
        frame = table[table["agent_action"] == action]
        centers, means, low, high = binned_cluster_stats(
            frame, "commitment_to_final", args.bins, rng, args.bootstrap
        )
        color = ACTION_COLORS[action]
        right.fill_between(centers, low, high, color=color, alpha=0.13, linewidth=0)
        right.plot(centers, means, "-", color=color, linewidth=2.0, label=action)
        baseline = float(
            first.loc[first["agent_action"] == action, "commitment_to_final"].mean()
        )
        # UP and DOWN baselines coincide at 0.144; nudge them apart horizontally
        offset = {"UP": -0.012, "DOWN": 0.012}.get(action, 0.0)
        right.plot([offset], [baseline], "s", color=color, markersize=6,
                   markerfacecolor="white", markeredgewidth=1.6, zorder=5)
        report_lines.append(
            f"- {action}: prompt-only {baseline:.3f}; bins "
            + ", ".join(f"{m:.3f}" for m in means)
        )

    right.axhline(0.25, color="#777777", linestyle=":", linewidth=1.2)
    right.set_ylim(0.0, 1.02)
    right.set_xlabel("progress through CoT")
    right.set_ylabel("mean commitment to eventual action")
    right.set_title("Per-action commitment along the CoT")
    right.legend(loc="lower right", fontsize=9, frameon=False)
    style_axis(right)

    figure.tight_layout()
    figure_path = REPO_ROOT / args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, bbox_inches="tight")
    (REPO_ROOT / args.report).write_text("\n".join(report_lines) + "\n")
    print(f"wrote {figure_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
