#!/usr/bin/env python3
"""Belief-update anchors: which CoT chunks move the forced-answer belief? (D5)

Consecutive prefixes differ by one generation chunk (~15-25 tokens), so the per-transition
total-variation distance of the belief, tv_t = 0.5 * ||q_{t+1} - q_t||_1, attributes belief
change to concrete spans of reasoning text. Transitions in the top 5% of tv are "anchors"
(cheap, observational analogue of Thought-Anchors resampling, arXiv 2506.19143). The added
text of each transition is categorized lexically to ask what KIND of content moves beliefs,
and per-step update concentration is related to when the step's answer stabilizes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLS = [f"q_{action}" for action in ACTIONS]

CATEGORY_PATTERNS = {
    "action_word": re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE),
    "correction": re.compile(r"\b(wait|but|actually|however|hmm|oops|mistake|wrong|re-?check|instead)\b",
                             re.IGNORECASE),
    "spatial": re.compile(r"\b(key|door|wall|grid|row|col(?:umn)?|room|cell|position|coordinate)\b"
                          r"|\(\s*\d+\s*,\s*\d+\s*\)", re.IGNORECASE),
    "plan": re.compile(r"\b(plan|path|route|then|first|next|need to|go to|move)\b", re.IGNORECASE),
}

INK = "#333333"
GRID = "#e6e6e6"
CATEGORY_COLORS = {"action_word": "#2a78d6", "correction": "#eb6834", "spatial": "#1baf7a",
                   "plan": "#4a3aa7", "none": "#9aa0a6"}


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
    parser.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"),
                        help="Supplies analysis_prefix_text for chunk diffing.")
    parser.add_argument("--boundary_steps", type=Path,
                        default=Path("outputs/commitment_boundary_steps.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/belief_update_anchors.parquet"))
    parser.add_argument("--report", type=Path, default=Path("outputs/belief_update_anchors.md"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/belief_anchors"))
    parser.add_argument("--anchor_quantile", type=float, default=0.95)
    return parser.parse_args()


def build_transitions(metrics: Any, texts: Any, pd: Any, np: Any) -> Any:
    frame = metrics.merge(texts, on="prefix_id", validate="one_to_one")
    frame = frame.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"])
    rows: list[dict[str, Any]] = []
    for (trajectory_id, step_id), group in frame.groupby(["trajectory_id", "step_id"], sort=False):
        q_matrix = group[Q_COLS].to_numpy(dtype=float)
        commitment = group["commitment_to_final"].to_numpy(dtype=float)
        progress = group["progress"].to_numpy(dtype=float)
        top = group["top_action"].to_numpy()
        text = ["" if value is None or value != value else str(value)
                for value in group["analysis_prefix_text"]]
        for index in range(1, len(group)):
            added = text[index][len(text[index - 1]):]
            rows.append({
                "trajectory_id": trajectory_id,
                "step_id": int(step_id),
                "transition_progress": float(progress[index]),
                "tv": float(0.5 * np.abs(q_matrix[index] - q_matrix[index - 1]).sum()),
                "commitment_delta": float(commitment[index] - commitment[index - 1]),
                "top_changed": bool(top[index] != top[index - 1]),
                "added_text": added,
                "added_chars": len(added),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()
    from scipy.stats import spearmanr

    metrics = pd.read_parquet(args.prefix_metrics,
                              columns=["prefix_id", "trajectory_id", "step_id", "prefix_token_index",
                                       "progress", "top_action", "commitment_to_final", *Q_COLS])
    texts = pd.read_parquet(args.prefixes, columns=["prefix_id", "analysis_prefix_text"])
    transitions = build_transitions(metrics, texts, pd, np)

    for name, pattern in CATEGORY_PATTERNS.items():
        transitions[f"cat_{name}"] = transitions["added_text"].str.contains(pattern, regex=True).fillna(False)
    category_columns = [f"cat_{name}" for name in CATEGORY_PATTERNS]
    transitions["cat_none"] = ~transitions[category_columns].any(axis=1)

    threshold = float(transitions["tv"].quantile(args.anchor_quantile))
    transitions["is_anchor"] = transitions["tv"] >= threshold
    anchors = transitions[transitions.is_anchor]

    total_tv = float(transitions["tv"].sum())
    anchor_tv_share = float(anchors["tv"].sum() / total_tv)

    def top3_share(group: Any) -> float:
        values = np.sort(group["tv"].to_numpy())[::-1]
        return float(values[:3].sum() / values.sum()) if values.sum() > 0 else np.nan

    per_step = transitions.groupby(["trajectory_id", "step_id"]).apply(top3_share, include_groups=False) \
        .rename("top3_tv_share").reset_index()
    boundary = pd.read_parquet(args.boundary_steps,
                               columns=["trajectory_id", "step_id", "stabilization_progress",
                                        "num_cot_tokens"])
    per_step = per_step.merge(boundary, on=["trajectory_id", "step_id"], validate="one_to_one")
    concentration_rho = spearmanr(per_step["top3_tv_share"], per_step["stabilization_progress"]).statistic
    length_rho = spearmanr(per_step["top3_tv_share"], per_step["num_cot_tokens"]).statistic

    enrichment_rows = []
    for name in [*CATEGORY_PATTERNS, "none"]:
        column = f"cat_{name}"
        base_rate = float(transitions[column].mean())
        anchor_rate = float(anchors[column].mean())
        enrichment_rows.append({"category": name, "baseline_rate": base_rate,
                                "anchor_rate": anchor_rate,
                                "enrichment": anchor_rate / base_rate if base_rate > 0 else np.nan})
    enrichment = pd.DataFrame(enrichment_rows)

    decile = pd.cut(transitions["transition_progress"], bins=10, labels=False, include_lowest=True)
    position_profile = transitions.groupby(decile).agg(
        anchor_rate=("is_anchor", "mean"), mean_tv=("tv", "mean"), n=("tv", "size")).reset_index(names="decile")
    position_profile["progress_mid"] = (position_profile["decile"] + 0.5) / 10

    args.output.parent.mkdir(parents=True, exist_ok=True)
    transitions.drop(columns=["added_text"]).to_parquet(args.output, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    def style(axis: Any, xlabel: str, ylabel: str, title: str) -> None:
        axis.set_xlabel(xlabel, color=INK)
        axis.set_ylabel(ylabel, color=INK)
        axis.set_title(title, color=INK, fontsize=11)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ordered = np.sort(transitions["tv"].to_numpy())
    ccdf = 1.0 - np.arange(1, len(ordered) + 1) / len(ordered)
    axes[0].plot(ordered, np.maximum(ccdf, 1e-6), color="#2a78d6", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].axvline(threshold, color=INK, linewidth=1.2, linestyle=":")
    axes[0].annotate(f"anchor threshold {threshold:.3f}", (threshold, 0.5),
                     textcoords="offset points", xytext=(6, 0), color=INK, fontsize=9)
    style(axes[0], "per-transition belief total variation", "CCDF (log scale)",
          "Belief updates are heavy-tailed")
    axes[1].plot(position_profile["progress_mid"], position_profile["anchor_rate"],
                 color="#eb6834", linewidth=2)
    axes[1].axhline(1 - args.anchor_quantile, color=INK, linewidth=1.2, linestyle=":")
    axes[1].annotate("uniform rate", (0.05, 1 - args.anchor_quantile), textcoords="offset points",
                     xytext=(0, 5), color=INK, fontsize=9)
    style(axes[1], "progress through CoT", "anchor rate in decile",
          "Where anchors happen")
    figure.tight_layout()
    figure_a = args.figure_dir / "anchor_distribution.png"
    figure.savefig(figure_a, dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    x_positions = np.arange(len(enrichment))
    bars = axis.bar(x_positions, enrichment["enrichment"],
                    color=[CATEGORY_COLORS[name] for name in enrichment["category"]], width=0.62)
    axis.bar_label(bars, fmt="%.2f", color=INK, fontsize=9, padding=2)
    axis.axhline(1.0, color=INK, linewidth=1.2, linestyle=":")
    axis.set_xticks(x_positions, enrichment["category"].tolist())
    style(axis, "added-chunk content category", "anchor rate / baseline rate",
          "What kind of content moves the belief (enrichment among anchors)")
    figure.tight_layout()
    figure_b = args.figure_dir / "anchor_category_enrichment.png"
    figure.savefig(figure_b, dpi=200)
    plt.close(figure)

    lines = [
        "# Belief-Update Anchors (D5)",
        "",
        f"- {len(transitions)} transitions over {per_step.shape[0]} steps; anchor = top "
        f"{100 * (1 - args.anchor_quantile):.0f}% of per-transition total variation "
        f"(threshold {threshold:.4f}).",
        "",
        "## Concentration",
        "",
        f"- The top 5% of transitions carry {100 * anchor_tv_share:.1f}% of all belief change.",
        f"- Median per-step share of belief change in its top-3 transitions: "
        f"{per_step['top3_tv_share'].median():.4f}",
        f"- Spearman(top-3 share, stabilization progress): {concentration_rho:.4f}",
        f"- Spearman(top-3 share, CoT length): {length_rho:.4f}",
        "",
        "## Anchor character",
        "",
        f"- fraction of anchors where the top action changed: {anchors['top_changed'].mean():.4f} "
        f"(baseline over all transitions: {transitions['top_changed'].mean():.4f})",
        f"- fraction of anchors that DECREASE commitment to the final action: "
        f"{(anchors['commitment_delta'] < 0).mean():.4f}",
        f"- median added-chunk length at anchors: {anchors['added_chars'].median():.0f} chars "
        f"(all transitions: {transitions['added_chars'].median():.0f})",
        "",
        "## Content-category enrichment",
        "",
        enrichment.round(4).to_markdown(index=False),
        "",
        "## Anchor position profile",
        "",
        position_profile[["progress_mid", "anchor_rate", "mean_tv", "n"]].round(4).to_markdown(index=False),
        "",
        "## Figures",
        "",
        f"- `{figure_a}`",
        f"- `{figure_b}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(transitions)} transitions)")
    print(f"Wrote {args.report}, {figure_a}, {figure_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
