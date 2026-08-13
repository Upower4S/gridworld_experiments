#!/usr/bin/env python3
"""Commitment-boundary phenomenology of the agent's chain of thought (D1).

On faithful extracted labels, characterize per decision step WHEN the forced-answer
belief settles: the retrospective argmax stabilization point (first prefix after which
the top action never changes again), permanent threshold crossings of commitment, the
post-commitment tail (fraction and tokens of CoT generated after the answer is already
stable), and how these clocks relate to the first verbal action mention. Also the
switch hazard along the CoT and the irreversibility of commitment levels.

Agentic multi-class analogue of the "commitment boundary" phenomenology reported for
binary QA (arXiv 2605.06723, 2606.13603).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

INK = "#333333"
GRID = "#e6e6e6"
CLOCK_COLORS = {
    "argmax stabilization": "#2a78d6",
    "permanent commitment >= 0.8": "#eb6834",
    "first verbal action mention": "#1baf7a",
}


def require_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}.") from exc
    return pd, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix_metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    parser.add_argument("--step_output", type=Path,
                        default=Path("outputs/commitment_boundary_steps.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/commitment_boundary_phenomenology.md"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/commitment_boundary"))
    parser.add_argument("--num_bins", type=int, default=10)
    return parser.parse_args()


def per_step_clocks(metrics: Any, pd: Any, np: Any) -> Any:
    rows: list[dict[str, Any]] = []
    grouped = metrics.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"]) \
        .groupby(["trajectory_id", "step_id"], sort=False)
    for (trajectory_id, step_id), group in grouped:
        top = group["top_action"].to_numpy()
        progress = group["progress"].to_numpy(dtype=float)
        tokens = group["prefix_token_index"].to_numpy(dtype=float)
        commitment = group["commitment_to_final"].to_numpy(dtype=float)
        final_top = top[-1]
        num_cot = float(group["num_analysis_tokens"].iloc[0])

        differs = top != final_top
        last_differ = int(np.max(np.nonzero(differs)[0])) if differs.any() else -1
        stab_index = last_differ + 1
        stabilization_progress = float(progress[stab_index])
        stabilization_token = float(tokens[stab_index])

        def permanent_cross(threshold: float) -> float | None:
            suffix_min = np.minimum.accumulate(commitment[::-1])[::-1]
            hits = np.nonzero(suffix_min >= threshold)[0]
            return float(progress[hits[0]]) if len(hits) else None

        mention_index = group["first_action_mention_token_index"].iloc[0]
        mention_progress = (float(mention_index) / num_cot
                            if pd.notna(mention_index) and num_cot > 0 else None)

        rows.append({
            "trajectory_id": trajectory_id,
            "step_id": int(step_id),
            "agent_action": group["agent_action"].iloc[0],
            "num_cot_tokens": num_cot,
            "num_prefixes": int(len(group)),
            "stabilization_progress": stabilization_progress,
            "stabilization_token": stabilization_token,
            "post_stabilization_tail_fraction": 1.0 - stabilization_progress,
            "post_stabilization_tail_tokens": num_cot - stabilization_token,
            "permanent_cross_0_5_progress": permanent_cross(0.5),
            "permanent_cross_0_8_progress": permanent_cross(0.8),
            "first_mention_progress": mention_progress,
            "stabilized_before_mention": (stabilization_progress < mention_progress
                                          if mention_progress is not None else None),
            "mention_minus_stabilization": (mention_progress - stabilization_progress
                                            if mention_progress is not None else None),
            "switch_count": int((top[1:] != top[:-1]).sum()),
            "prompt_only_commitment": float(commitment[0]),
        })
    return pd.DataFrame(rows)


def hazard_curves(metrics: Any, num_bins: int, pd: Any, np: Any) -> tuple[Any, Any]:
    frame = metrics.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"]).copy()
    grouped = frame.groupby(["trajectory_id", "step_id"], sort=False)
    changed = frame["top_action_changed"].fillna(False).astype(bool).to_numpy()
    future = np.zeros(len(frame), dtype=bool)
    for _, group in grouped:
        positions = group.index.to_numpy()
        local_changed = changed[frame.index.get_indexer(positions)]
        suffix = np.cumsum(local_changed[::-1])[::-1]
        future[frame.index.get_indexer(positions)] = (suffix - local_changed) > 0
    frame["has_future_switch"] = future

    progress_bin = pd.cut(frame["progress"], bins=num_bins, labels=False, include_lowest=True)
    hazard = frame.groupby(progress_bin).agg(
        switch_rate=("top_action_changed", "mean"),
        n=("top_action_changed", "size")).reset_index(names="bin")
    hazard["progress_mid"] = (hazard["bin"] + 0.5) / num_bins

    commitment_bin = pd.cut(frame["commitment_to_final"], bins=num_bins, labels=False, include_lowest=True)
    irreversibility = frame.groupby(commitment_bin).agg(
        future_switch_rate=("has_future_switch", "mean"),
        n=("has_future_switch", "size")).reset_index(names="bin")
    irreversibility["commitment_mid"] = (irreversibility["bin"] + 0.5) / num_bins
    return hazard, irreversibility


def draw_figures(steps: Any, hazard: Any, irreversibility: Any, figure_dir: Path, np: Any) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def style(axis: Any, xlabel: str, ylabel: str, title: str) -> None:
        axis.set_xlabel(xlabel, color=INK)
        axis.set_ylabel(ylabel, color=INK)
        axis.set_title(title, color=INK, fontsize=11)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    clock_values = {
        "argmax stabilization": steps["stabilization_progress"].dropna(),
        "permanent commitment >= 0.8": steps["permanent_cross_0_8_progress"].dropna(),
        "first verbal action mention": steps["first_mention_progress"].dropna(),
    }
    for label, values in clock_values.items():
        ordered = np.sort(values.to_numpy())
        cdf = np.arange(1, len(ordered) + 1) / len(steps)
        axis.plot(ordered, cdf, color=CLOCK_COLORS[label], linewidth=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    style(axis, "progress through CoT", "fraction of decision steps",
          "When does the belief settle? Three clocks per decision step (CDF)")
    axis.legend(handles=[plt.Line2D([], [], color=color, linewidth=2, label=label)
                         for label, color in CLOCK_COLORS.items()],
                loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout()
    path = figure_dir / "three_clocks_cdf.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(hazard["progress_mid"], hazard["switch_rate"], color="#2a78d6", linewidth=2)
    style(axes[0], "progress through CoT", "P(top action switches at this prefix)",
          "Switch hazard along the CoT")
    axes[1].plot(irreversibility["commitment_mid"], irreversibility["future_switch_rate"],
                 color="#eb6834", linewidth=2)
    style(axes[1], "current commitment to final action", "P(any later top-action switch)",
          "Irreversibility: does commitment level predict no more switches?")
    axes[1].set_xlim(0, 1)
    figure.tight_layout()
    path = figure_dir / "switch_hazard_irreversibility.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


def quantile_line(series: Any) -> str:
    clean = series.dropna()
    if clean.empty:
        return "NA"
    return (f"median {clean.median():.4f}, IQR [{clean.quantile(0.25):.4f}, {clean.quantile(0.75):.4f}], "
            f"mean {clean.mean():.4f}, n={len(clean)}")


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()

    metrics = pd.read_parquet(args.prefix_metrics)
    steps = per_step_clocks(metrics, pd, np)
    hazard, irreversibility = hazard_curves(metrics, args.num_bins, pd, np)

    args.step_output.parent.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(args.step_output, engine="pyarrow", index=False)
    figures = draw_figures(steps, hazard, irreversibility, args.figure_dir, np)

    with_mention = steps[steps["first_mention_progress"].notna()]
    per_action = steps.groupby("agent_action").agg(
        median_stabilization=("stabilization_progress", "median"),
        mean_tail_fraction=("post_stabilization_tail_fraction", "mean"),
        median_tail_tokens=("post_stabilization_tail_tokens", "median"),
        mean_switches=("switch_count", "mean"),
        n=("step_id", "size")).round(4)
    length_quartile = pd.qcut(steps["num_cot_tokens"], 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    by_length = steps.groupby(length_quartile, observed=True).agg(
        median_stabilization=("stabilization_progress", "median"),
        mean_tail_fraction=("post_stabilization_tail_fraction", "mean"),
        median_cot_tokens=("num_cot_tokens", "median"),
        n=("step_id", "size")).round(4)

    lines = [
        "# Commitment-Boundary Phenomenology (faithful labels)",
        "",
        f"- Input: `{args.prefix_metrics}`; {len(steps)} decision steps, "
        f"{steps.trajectory_id.nunique()} trajectories.",
        "- Clock definitions: argmax stabilization = first prefix after which the forced top action",
        "  never changes again; permanent cross = first prefix after which commitment stays >= the",
        "  threshold; mention = `first_action_mention_token_index / num_cot_tokens`.",
        "",
        "## The three clocks (progress units)",
        "",
        f"- argmax stabilization: {quantile_line(steps['stabilization_progress'])}",
        f"- permanent commitment >= 0.5: {quantile_line(steps['permanent_cross_0_5_progress'])}",
        f"- permanent commitment >= 0.8: {quantile_line(steps['permanent_cross_0_8_progress'])}",
        f"- first verbal action mention: {quantile_line(steps['first_mention_progress'])}",
        "",
        "## The post-commitment tail",
        "",
        f"- tail fraction (CoT after argmax stabilization): {quantile_line(steps['post_stabilization_tail_fraction'])}",
        f"- tail tokens: {quantile_line(steps['post_stabilization_tail_tokens'])}",
        "",
        "## Stabilization vs verbalization",
        "",
        f"- steps with any action mention: {len(with_mention)} / {len(steps)}",
        f"- fraction stabilized BEFORE first mention: {with_mention['stabilized_before_mention'].mean():.4f}",
        f"- mention minus stabilization (progress lead when positive): "
        f"{quantile_line(with_mention['mention_minus_stabilization'])}",
        "",
        "## Per-action clocks",
        "",
        per_action.to_markdown(),
        "",
        "## By CoT length quartile",
        "",
        by_length.to_markdown(),
        "",
        "## Switch hazard and irreversibility",
        "",
        hazard[["progress_mid", "switch_rate", "n"]].round(4).to_markdown(index=False),
        "",
        irreversibility[["commitment_mid", "future_switch_rate", "n"]].round(4).to_markdown(index=False),
        "",
        "## Figures",
        "",
        *[f"- `{path}`" for path in figures],
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.step_output} ({len(steps)} steps)")
    print(f"Wrote {args.report}")
    for path in figures:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
