#!/usr/bin/env python3
"""Plot probe-guided activation PCA/UMAP embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_guided_embedding_lib as lib

COLORS = {"UP": "#4C78A8", "DOWN": "#F58518", "LEFT": "#54A24B", "RIGHT": "#B279A2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/probe_guided_activation_embeddings.parquet")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--max_points", type=int, default=60000)
    p.add_argument("--seed", type=int, default=lib.DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sample(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=seed)


def setup(ax, title: str):
    ax.set_xlabel("embedding dimension 1 from PCA/UMAP on activation vectors")
    ax.set_ylabel("embedding dimension 2 from PCA/UMAP on activation vectors")
    ax.set_title(title)


def scatter_action(df: pd.DataFrame, color_col: str, path: Path, title: str, max_points: int, seed: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = sample(df, max_points, seed)
    for action in lib.ACTIONS:
        part = plot_df[plot_df[color_col].astype(str).eq(action)]
        if not part.empty:
            ax.scatter(part["embedding_x"], part["embedding_y"], s=8, alpha=0.55, color=COLORS[action], label=action, edgecolors="none", rasterized=True)
    setup(ax, title)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_numeric(df: pd.DataFrame, color_col: str, path: Path, title: str, max_points: int, seed: int):
    import matplotlib.pyplot as plt

    if color_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = sample(df.dropna(subset=[color_col]), max_points, seed)
    sc = ax.scatter(plot_df["embedding_x"], plot_df["embedding_y"], c=plot_df[color_col], s=8, alpha=0.55, cmap="viridis", edgecolors="none", rasterized=True)
    setup(ax, title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_col)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_split_role(df: pd.DataFrame, path: Path, title: str, max_points: int, seed: int):
    import matplotlib.pyplot as plt

    if "split_role" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = sample(df, max_points, seed)
    colors = {"train": "#4C78A8", "test": "#E45756", "unused": "#999999"}
    for role, part in plot_df.groupby("split_role"):
        ax.scatter(part["embedding_x"], part["embedding_y"], s=8, alpha=0.5, color=colors.get(str(role), "#777777"), label=str(role), edgecolors="none", rasterized=True)
    setup(ax, title)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(df: pd.DataFrame, path: Path, max_decisions: int = 40):
    import matplotlib.pyplot as plt

    if "decision_id" not in df.columns:
        return
    last = df.sort_values(["decision_id", "prefix_token_index"]).groupby("decision_id", as_index=False).tail(1)
    chosen = []
    for action in lib.ACTIONS:
        part = last[last["true_q_top_action"].astype(str).eq(action)].sort_values("commitment_to_final", ascending=False).head(max_decisions // 4 + 1)
        chosen.extend(part["decision_id"].astype(str).tolist())
    chosen = chosen[:max_decisions]
    fig, ax = plt.subplots(figsize=(8, 7))
    for decision_id in chosen:
        part = df[df["decision_id"].astype(str).eq(decision_id)].sort_values("prefix_token_index")
        if part.empty:
            continue
        action = str(part["true_q_top_action"].iloc[-1])
        color = COLORS.get(action, "#777777")
        ax.plot(part["embedding_x"], part["embedding_y"], lw=1.0, alpha=0.45, color=color)
        ax.scatter(part["embedding_x"].iloc[0], part["embedding_y"].iloc[0], marker="o", facecolors="none", edgecolors=color, s=34)
        ax.scatter(part["embedding_x"].iloc[-1], part["embedding_y"].iloc[-1], marker="x", color=color, s=44)
    setup(ax, "Representative trajectories in probe-guided activation embedding")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    figure_dir = Path(args.figure_dir)
    if figure_dir.exists() and list(figure_dir.glob("*.png")) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing figures in {figure_dir}; pass --overwrite")
    lib.ensure_dir(figure_dir)
    df = lib.read_parquet_robust(args.embeddings)
    required = ["embedding_x", "embedding_y", "embedding_method", "probe_guidance_method"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Embedding parquet missing required columns: {missing}")
    manifest_rows = []
    for (method, guidance), part in df.groupby(["embedding_method", "probe_guidance_method"], sort=True):
        prefix = f"{method}_{guidance}"
        scatter_action(part, "true_q_top_action", figure_dir / f"{prefix}_by_true_q_top_action.png", f"{method.upper()} on probe-guided activations by true q-top action ({guidance})", args.max_points, args.seed)
        manifest_rows.append({"figure": f"{prefix}_by_true_q_top_action.png", "description": "PCA/UMAP activation coordinates colored by true q-top action."})
        if "agent_action" in part.columns:
            scatter_action(part, "agent_action", figure_dir / f"{prefix}_by_agent_action.png", f"{method.upper()} on probe-guided activations by agent action ({guidance})", args.max_points, args.seed)
            manifest_rows.append({"figure": f"{prefix}_by_agent_action.png", "description": "PCA/UMAP activation coordinates colored by final agent action."})
        for numeric in ["commitment_to_final", "entropy_commitment", "progress"]:
            if numeric in part.columns:
                scatter_numeric(part, numeric, figure_dir / f"{prefix}_by_{numeric}.png", f"{method.upper()} on probe-guided activations by {numeric} ({guidance})", args.max_points, args.seed)
                manifest_rows.append({"figure": f"{prefix}_by_{numeric}.png", "description": f"PCA/UMAP activation coordinates colored by {numeric}."})
        scatter_split_role(part, figure_dir / f"{prefix}_by_split_role.png", f"{method.upper()} on probe-guided activations by split role ({guidance})", args.max_points, args.seed)
        manifest_rows.append({"figure": f"{prefix}_by_split_role.png", "description": "Train/test split role check."})
        if "strict_before_any_action_mention" in part.columns:
            strict = part[part["strict_before_any_action_mention"].fillna(False).astype(bool)]
            if not strict.empty:
                scatter_action(strict, "true_q_top_action", figure_dir / f"{prefix}_strict_pre_action_by_true_q_top_action.png", f"Strict pre-action {method.upper()} activation embedding ({guidance})", args.max_points, args.seed)
                manifest_rows.append({"figure": f"{prefix}_strict_pre_action_by_true_q_top_action.png", "description": "Strict pre-action subset colored by true q-top action."})
        plot_trajectories(part, figure_dir / f"{prefix}_trajectories.png")
        manifest_rows.append({"figure": f"{prefix}_trajectories.png", "description": "Representative decision trajectories connected in activation PCA/UMAP space."})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(figure_dir / "plot_manifest.csv", index=False)
    lib.save_markdown(
        figure_dir / "plot_report.md",
        "# Probe-Guided Activation Embedding Plots\n\n"
        "All x/y coordinates in these figures come from PCA/UMAP applied to activation vectors after probe-guided dimension selection or weighting. They are not probe-score coordinates.\n\n"
        + (manifest.to_markdown(index=False) if not manifest.empty else "No figures generated."),
    )
    lib.log(f"Wrote figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
