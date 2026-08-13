#!/usr/bin/env python3
"""Plot strict probe-guided activation embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from probe_guided_activation_embedding_lib import ACTIONS, ensure_dir, save_markdown

COLORS = {"UP": "#4C78A8", "DOWN": "#F58518", "LEFT": "#54A24B", "RIGHT": "#B279A2"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/embeddings/probe_guided_activation_embeddings.parquet")
    p.add_argument("--control_comparison", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/embedding_control_comparison.csv")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--max_points", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def scatter_action(df: pd.DataFrame, path: Path, title: str, color_col: str = "true_q_top_action") -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    for action in ACTIONS:
        part = df[df[color_col].astype(str).eq(action)]
        if not part.empty:
            ax.scatter(part["embedding_x"], part["embedding_y"], s=8, alpha=0.55, color=COLORS[action], label=action, edgecolors="none", rasterized=True)
    ax.set_xlabel("PCA/UMAP dimension 1 from activation vectors")
    ax.set_ylabel("PCA/UMAP dimension 2 from activation vectors")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_numeric(df: pd.DataFrame, path: Path, title: str, col: str) -> None:
    import matplotlib.pyplot as plt
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["embedding_x"], df["embedding_y"], c=df[col], s=8, alpha=0.55, cmap="viridis", edgecolors="none", rasterized=True)
    ax.set_xlabel("PCA/UMAP dimension 1 from activation vectors")
    ax.set_ylabel("PCA/UMAP dimension 2 from activation vectors")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(col)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def bar_metric(control: pd.DataFrame, path: Path, metric: str) -> None:
    import matplotlib.pyplot as plt
    data = control.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["feature_set_family"], data[metric], color="#4C78A8")
    ax.set_ylabel(metric)
    ax.set_title(f"Control comparison: {metric}")
    ax.tick_params(axis="x", rotation=25)
    for i, (_, row) in enumerate(data.iterrows()):
        val = row[metric]
        if pd.notna(val):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    figure_dir = ensure_dir(args.figure_dir)
    emb = pd.read_parquet(args.embeddings)
    control = pd.read_csv(args.control_comparison)
    plot_rows = []
    # Plot split 0 for each family so every plot is one consistent train-fit coordinate frame.
    families = ["probe_weighted_top_K", "probe_top_K", "all_dims", "random_top_K", "shuffled_probe_top_K", "progress_probe_top_K"]
    for method in sorted(emb["embedding_method"].unique()):
        for fam in families:
            part = emb[(emb["embedding_method"].eq(method)) & (emb["feature_set_family"].eq(fam)) & (emb["split_id"].eq(0))]
            if part.empty:
                continue
            part = sample(part, args.max_points, args.seed)
            name = f"{method}_split0_{fam}_by_true_q_top_action.png"
            scatter_action(part, figure_dir / name, f"{method.upper()} activation embedding split 0: {fam} by true q-top")
            plot_rows.append({"figure": name, "embedding_method": method, "feature_set_family": fam, "split_id": 0, "color": "true_q_top_action", "coordinate_source": "PCA/UMAP on activation vectors", "caveat": "split-specific held-out coordinate frame"})
            if fam == "probe_weighted_top_K":
                for col in ["agent_action", "progress", "commitment_to_final", "entropy_commitment"]:
                    name = f"{method}_split0_{fam}_by_{col}.png"
                    if col in ["agent_action"]:
                        scatter_action(part, figure_dir / name, f"{method.upper()} activation embedding split 0: {fam} by {col}", color_col=col)
                    else:
                        scatter_numeric(part, figure_dir / name, f"{method.upper()} activation embedding split 0: {fam} by {col}", col)
                    plot_rows.append({"figure": name, "embedding_method": method, "feature_set_family": fam, "split_id": 0, "color": col, "coordinate_source": "PCA/UMAP on activation vectors", "caveat": "color is metadata, not input feature"})
                if "strict_before_any_action_mention" in part.columns:
                    strict = part[part["strict_before_any_action_mention"].fillna(False).astype(bool)]
                    if not strict.empty:
                        name = f"{method}_split0_{fam}_strict_pre_action_by_true_q_top_action.png"
                        scatter_action(strict, figure_dir / name, f"Strict pre-action {method.upper()} activation embedding: {fam}")
                        plot_rows.append({"figure": name, "embedding_method": method, "feature_set_family": fam, "split_id": 0, "color": "true_q_top_action", "coordinate_source": "PCA/UMAP on activation vectors", "caveat": "strict subset excludes post-action verbalization rows"})
    for metric in ["test_knn_balanced_accuracy", "test_silhouette_true_q_top", "test_centroid_separation_ratio"]:
        name = f"control_comparison_{metric}.png"
        bar_metric(control, figure_dir / name, metric)
        plot_rows.append({"figure": name, "embedding_method": "pca", "feature_set_family": "all", "split_id": "mean_over_splits", "color": metric, "coordinate_source": "metrics from PCA on activation vectors", "caveat": "higher is not necessarily causal evidence"})
    plot_index = pd.DataFrame(plot_rows)
    plot_index.to_csv(figure_dir / "plot_index.csv", index=False)
    save_markdown(figure_dir / "plot_report.md", "# Probe-Guided Activation Embedding Plots\n\n" + plot_index.to_markdown(index=False))
    print(f"Wrote {figure_dir / 'plot_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
