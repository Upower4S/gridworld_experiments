#!/usr/bin/env python3
"""Synthesis figures for the two cross-cutting claims, plus a readable C2 panel.

The claims document states two lessons that no single figure carries:

  A. The belief is easy to READ and hard to WRITE. One decodability number stands against
     four independent intervention families.
  B. Belief STRENGTH and belief CONTENT are separable channels, shown here by two pieces
     of evidence that do not depend on the layer geometry of Theorem 3.

It also inherits a hard-to-read C2 panel (two overlapping 14-point line series). The same
data reads far better as a per-layer paired comparison, which is what the question
("is layer 10-11 special?") actually asks.

Every number is recomputed from the stored artifacts; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_MODE = False

# Validated categorical palette (node scripts/validate_palette.js, light mode: ALL CHECKS
# PASS for these four; the greens' 2.74:1 contrast is relieved by direct labels).
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
NEUTRAL = "#9aa0a6"   # deliberate neutral for baselines/controls, never a series
INK = "#333333"
MUTED_INK = "#6b7075"
GRID = "#e6e6e6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--paper", action="store_true",
                        help="Omit figure suptitles; in a paper the caption carries the message.")
    parser.add_argument("--summary_csv", type=Path, default=REPO_ROOT
                        / "outputs/conceptual_belief_space/cot_window_layer_transition_probe/fold0_20260721/summary.csv")
    return parser.parse_args()


def style(axis: Any, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    if xlabel:
        axis.set_xlabel(xlabel, color=INK)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK)
    if title:
        axis.set_title(title, color=INK, fontsize=11)
    axis.tick_params(colors=INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def gather_numbers(args: argparse.Namespace) -> dict[str, Any]:
    numbers: dict[str, Any] = {}

    summary = pd.read_csv(args.summary_csv).set_index("method")
    numbers["read"] = [
        ("activation probe", float(summary.loc["mean_layers_mlp", "mean_spearman"]), BLUE),
        ("length-oracle baseline", float(summary.loc["progress_oracle", "mean_spearman"]), NEUTRAL),
        ("fair positional bar", float(summary.loc["position_index_fair", "mean_spearman"]), NEUTRAL),
    ]

    # Rank-1 steering: best over layer/strength, plus its norm-matched random control.
    steering = pd.read_parquet(REPO_ROOT
                               / "outputs/conceptual_belief_space/layer_steering/steering_full_n28.parquet")
    # Collapse action_UP/DOWN/LEFT/RIGHT into one family: each is applied only to the
    # prefixes whose runner-up is that action, so grouping them separately leaves cells of
    # ~7 rows whose maximum is dominated by sampling noise.
    steering = steering.assign(family=np.where(steering.direction.str.startswith("action_"),
                                               "action", steering.direction))
    cells = steering.groupby(["family", "alpha", "layer"])[["flipped", "flipped_to_target"]].mean()
    real = cells.drop(index="random", level="family")
    steer_any = float(real["flipped"].max())
    steer_target = float(real["flipped_to_target"].max())
    steer_control = float(cells.xs("random", level="family")["flipped"].max())

    ablation = pd.read_csv(REPO_ROOT / "outputs/conjectures/c2_subblock/summary.csv")
    ablate_any = float(ablation["flip_rate"].max())

    patching = pd.read_csv(REPO_ROOT / "outputs/conjectures/c6_patching/summary.csv")
    full_window = patching[patching["mode"] == "full_window"]
    patch_any = float(full_window["any_change_rate"].max())
    patch_target = float(full_window["flip_to_target_rate"].max())

    # Instruction: did the disavowal change the answer at all, and toward the prompt action?
    disavowal = pd.read_parquet(REPO_ROOT / "outputs/conjectures/c7_disavowal/scores_disavowal.parquet")
    baseline = pd.read_parquet(REPO_ROOT
                               / "outputs/followups/mismatched_prompt_cot_scores_extracted.parquet")
    joined = baseline[["row_key", "condition", "top_action", "prompt_agent_action"]].merge(
        disavowal[["row_key", "top_action"]], on="row_key", suffixes=("_base", "_dis"))
    joined = joined[joined.condition == "mismatched"]
    instruct_any = float((joined.top_action_base != joined.top_action_dis).mean())
    gained = ((joined.top_action_dis == joined.prompt_agent_action)
              & (joined.top_action_base != joined.prompt_agent_action)).mean()
    instruct_target = float(gained)

    numbers["write"] = [
        ("rank-1 steering\n(2x residual RMS)", steer_any, steer_target, steer_control),
        ("delete a whole\nattention or MLP block", ablate_any, np.nan, np.nan),
        ("transplant the entire\n12-token window", patch_any, patch_target, np.nan),
        ("instruct the model\nto ignore the reasoning", instruct_any, instruct_target, np.nan),
    ]

    c1 = pd.read_csv(REPO_ROOT / "outputs/conjectures/c1_gain_direction/model_comparison.csv")
    c1_mean = c1[c1.representation == "all_layer_mean"].set_index("model")
    numbers["content_decoders"] = [
        ("additive\ncontent decoder", float(c1_mean.loc["additive", "commitment_spearman"]), NEUTRAL),
        ("content decoder\n+ learned gain", float(c1_mean.loc["gain", "commitment_spearman"]), ORANGE),
        ("dedicated\ncommitment probe", float(summary.loc["mean_layers_mlp", "mean_spearman"]), BLUE),
    ]

    c10 = pd.read_csv(REPO_ROOT / "outputs/conjectures/c10_prompt_prior/correlations.csv").iloc[0]
    numbers["prior"] = [
        ("prior STRENGTH\n(how hard it leans)", abs(float(c10["true_prompt_commitment_vs_stabilization"])), BLUE),
        ("prior CONTENT\n(which way it leans)", abs(float(c10["predicted_q_RIGHT_vs_stabilization"])), NEUTRAL),
    ]
    numbers["ablation_table"] = ablation
    return numbers


def figure_read_write(numbers: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                               gridspec_kw={"width_ratios": [1.0, 1.45]})

    labels = [entry[0] for entry in numbers["read"]]
    values = [entry[1] for entry in numbers["read"]]
    colors = [entry[2] for entry in numbers["read"]]
    positions = np.arange(len(labels))[::-1]
    axes[0].barh(positions, values, color=colors, height=0.55)
    for position, value in zip(positions, values):
        axes[0].text(value + 0.015, position, f"{value:.3f}", va="center",
                     color=INK, fontsize=10)
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlim(0, 1.0)
    style(axes[0], xlabel="held-out Spearman with true commitment",
          title="Reading the belief")
    axes[0].grid(True, axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)

    entries = numbers["write"]
    positions = np.arange(len(entries))[::-1]
    height = 0.34
    for offset, (position, entry) in enumerate(zip(positions, entries)):
        label, any_change, to_target, control = entry
        axes[1].barh(position + height / 2, any_change, color=ORANGE, height=height)
        axes[1].text(any_change + 0.006, position + height / 2, f"{any_change:.3f}",
                     va="center", color=INK, fontsize=9)
        if np.isfinite(to_target):
            axes[1].barh(position - height / 2, to_target, color=BLUE, height=height)
            axes[1].text(to_target + 0.006, position - height / 2, f"{to_target:.3f}",
                         va="center", color=INK, fontsize=9)
        else:
            axes[1].text(0.006, position - height / 2, "no intended target",
                         va="center", color=MUTED_INK, fontsize=8, style="italic")
        if np.isfinite(control):
            axes[1].plot([control, control], [position - height, position + height],
                         color=INK, linewidth=1.4, linestyle=":")
            axes[1].text(control + 0.006, position + height * 1.15,
                         "random-direction control", color=MUTED_INK, fontsize=8)
    axes[1].set_yticks(positions, [entry[0] for entry in entries], fontsize=9)
    axes[1].set_xlim(0, 0.42)
    style(axes[1], xlabel="fraction of cases", title="Moving the belief")
    axes[1].grid(True, axis="x", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="changed the answer at all"),
                            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="changed it to the intended answer")],
                   loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    if not PAPER_MODE:

        figure.suptitle("The forced-answer belief is easy to read and hard to write",
                    color=INK, fontsize=12.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def figure_strength_content(numbers: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    entries = numbers["content_decoders"]
    positions = np.arange(len(entries))
    axes[0].bar(positions, [entry[1] for entry in entries],
                color=[entry[2] for entry in entries], width=0.6)
    for position, entry in zip(positions, entries):
        offset = 0.02 if entry[1] >= 0 else -0.05
        axes[0].text(position, entry[1] + offset, f"{entry[1]:+.3f}", ha="center",
                     color=INK, fontsize=10)
    axes[0].axhline(0.0, color=INK, linewidth=1.0)
    axes[0].set_xticks(positions, [entry[0] for entry in entries], fontsize=9)
    axes[0].set_ylim(-0.15, 0.92)
    style(axes[0], ylabel="Spearman with true commitment",
          title="Decoding the belief's content\ndoes not deliver its strength")
    axes[0].grid(True, axis="y", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)

    entries = numbers["prior"]
    positions = np.arange(len(entries))
    axes[1].bar(positions, [entry[1] for entry in entries],
                color=[entry[2] for entry in entries], width=0.5)
    for position, entry in zip(positions, entries):
        axes[1].text(position, entry[1] + 0.008, f"{entry[1]:.3f}", ha="center",
                     color=INK, fontsize=10)
    axes[1].set_xticks(positions, [entry[0] for entry in entries], fontsize=9)
    axes[1].set_ylim(0, 0.42)
    style(axes[1], ylabel="|Spearman| with settling time",
          title="Before any reasoning, the prior's strength\npredicts settling; its direction does not")
    axes[1].grid(True, axis="y", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)

    if not PAPER_MODE:

        figure.suptitle("Belief strength and belief content are separable channels",
                    color=INK, fontsize=12.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def figure_c2_paired(numbers: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    ablation = numbers["ablation_table"]
    pivot = ablation.pivot(index="layer", columns="part", values="mean_kl").sort_index()
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    positions = np.arange(len(pivot))[::-1]

    for position, (layer, row) in zip(positions, pivot.iterrows()):
        highlight = layer in (10, 11)
        axis.plot([row["attn"], row["mlp"]], [position, position],
                  color=INK if highlight else GRID, linewidth=2.0 if highlight else 1.4,
                  zorder=1, solid_capstyle="round")
        axis.scatter([row["attn"]], [position], s=64, color=BLUE, zorder=3,
                     edgecolors="white", linewidths=1.0)
        axis.scatter([row["mlp"]], [position], s=64, color=ORANGE, zorder=3,
                     edgecolors="white", linewidths=1.0)

    labels = [f"layer {layer}" + ("  <-- predicted bottleneck" if layer in (10, 11) else "")
              for layer in pivot.index]
    axis.set_yticks(positions, labels, fontsize=9)
    for tick, layer in zip(axis.get_yticklabels(), pivot.index):
        tick.set_color(INK if layer in (10, 11) else MUTED_INK)
    axis.set_xlim(0, max(pivot.max()) * 1.18)
    style(axis, xlabel="mean KL(unablated belief || belief after deleting the sub-block)",
          title="Deleting a whole sub-block barely moves the belief anywhere,\n"
                "and layers 10-11 are not exceptional")
    axis.grid(True, axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                                    color=BLUE, label="attention deleted"),
                         plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                                    color=ORANGE, label="MLP deleted")],
                loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    global PAPER_MODE
    PAPER_MODE = bool(args.paper)
    numbers = gather_numbers(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "read_write": args.out_dir / "synthesis_read_write_asymmetry.png",
        "strength_content": args.out_dir / "synthesis_strength_vs_content.png",
        "c2_paired": args.out_dir / "c2_subblock_paired.png",
    }
    figure_read_write(numbers, paths["read_write"])
    figure_strength_content(numbers, paths["strength_content"])
    figure_c2_paired(numbers, paths["c2_paired"])

    print("Reading:")
    for label, value, _ in numbers["read"]:
        print(f"  {label:26s} {value:.4f}")
    print("Moving (any change / toward target / control):")
    for label, any_change, to_target, control in numbers["write"]:
        flat = label.replace("\n", " ")
        print(f"  {flat:44s} {any_change:.4f} / "
              f"{'n/a' if not np.isfinite(to_target) else f'{to_target:.4f}'} / "
              f"{'n/a' if not np.isfinite(control) else f'{control:.4f}'}")
    print("Content decoders vs commitment probe:")
    for label, value, _ in numbers["content_decoders"]:
        print(f"  {label.replace(chr(10), ' '):32s} {value:+.4f}")
    print("Prior:")
    for label, value, _ in numbers["prior"]:
        print(f"  {label.replace(chr(10), ' '):36s} {value:.4f}")
    for name, path in paths.items():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
