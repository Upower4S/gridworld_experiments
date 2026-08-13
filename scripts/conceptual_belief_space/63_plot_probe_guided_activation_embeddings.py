#!/usr/bin/env python3
"""Plot probe-guided activation PCA/UMAP embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from probe_guided_embedding_lib import ACTIONS, ensure_dir, save_markdown


COLORS = {"UP": "#4C78A8", "DOWN": "#F58518", "LEFT": "#54A24B", "RIGHT": "#B279A2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/embeddings/probe_guided_activation_embeddings.parquet")
    p.add_argument("--metrics", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/embedding_control_comparison.csv")
    p.add_argument("--stability", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/feature_importance/feature_importance_stability.csv")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--max_points", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def choose_primary(emb: pd.DataFrame) -> pd.DataFrame:
    df = emb.copy()
    filters = [
        ("embedding_method", "pca"),
        ("feature_set", "probe_top_512"),
        ("residualize", "none"),
    ]
    for col, val in filters:
        if col in df.columns and val in set(df[col].astype(str)):
            df = df[df[col].astype(str).eq(val)]
    if "split_id" in df.columns:
        df = df[df["split_id"].eq(sorted(df["split_id"].unique())[0])]
    if "view_name" in df.columns:
        df = df[df["view_name"].eq(sorted(df["view_name"].astype(str).unique())[0])]
    if "split_role" in df.columns and "test" in set(df["split_role"].astype(str)):
        df = df[df["split_role"].astype(str).eq("test")]
    return df.reset_index(drop=True)


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def scatter(df: pd.DataFrame, path: Path, color: str, title: str, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    d = sample(df, max_points, seed)
    fig, ax = plt.subplots(figsize=(8, 7))
    if color in d.columns and (d[color].dtype == object or d[color].astype(str).isin(ACTIONS).any()):
        for action in ACTIONS:
            part = d[d[color].astype(str).eq(action)]
            if not part.empty:
                ax.scatter(part["coord_x"], part["coord_y"], s=8, alpha=0.55, color=COLORS[action], label=action, edgecolors="none", rasterized=True)
        ax.legend(frameon=True, fontsize=8)
    else:
        values = pd.to_numeric(d[color], errors="coerce") if color in d.columns else None
        sc = ax.scatter(d["coord_x"], d["coord_y"], c=values, s=8, alpha=0.55, cmap="viridis", edgecolors="none", rasterized=True)
        if color in d.columns:
            fig.colorbar(sc, ax=ax, label=color)
    ax.set_xlabel("PCA/UMAP dimension 1 from activation features")
    ax.set_ylabel("PCA/UMAP dimension 2 from activation features")
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def low_high(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    specs = [("low commitment < 0.5", df[df["top_commitment"] < 0.5]), ("high commitment >= 0.75", df[df["top_commitment"] >= 0.75])]
    for ax, (title, part) in zip(axes, specs):
        for action in ACTIONS:
            p = part[part["q_top_action"].astype(str).eq(action)]
            ax.scatter(p["coord_x"], p["coord_y"], s=8, alpha=0.55, color=COLORS[action], label=action, edgecolors="none", rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("dimension 1")
        ax.set_ylabel("dimension 2")
    axes[1].legend(frameon=True, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def median_trajectory(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    d = df.copy()
    d["progress_bin"] = pd.cut(pd.to_numeric(d["progress"], errors="coerce"), bins=np.linspace(0, 1, 11), include_lowest=True, labels=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    for action in ACTIONS:
        part = d[d["q_top_action"].astype(str).eq(action)]
        med = part.groupby("progress_bin", as_index=False).agg(coord_x=("coord_x", "median"), coord_y=("coord_y", "median"), progress=("progress", "median"))
        if len(med) >= 2:
            ax.plot(med["coord_x"], med["coord_y"], marker="o", color=COLORS[action], label=action)
            ax.annotate("", xy=(med["coord_x"].iloc[-1], med["coord_y"].iloc[-1]), xytext=(med["coord_x"].iloc[-2], med["coord_y"].iloc[-2]), arrowprops={"arrowstyle": "->", "color": COLORS[action]})
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.set_title("Median trajectory by true q-top action")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def individual_trajectories(df: pd.DataFrame, path: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    decisions = np.asarray(sorted(df["decision_id"].dropna().astype(str).unique()))
    if len(decisions) > 40:
        decisions = rng.choice(decisions, size=40, replace=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    for decision in decisions:
        part = df[df["decision_id"].astype(str).eq(str(decision))].sort_values("prefix_token_index")
        if len(part) < 2:
            continue
        action = str(part["q_top_action"].iloc[-1])
        color = COLORS.get(action, "#999999")
        ax.plot(part["coord_x"], part["coord_y"], color=color, alpha=0.35, linewidth=1)
        ax.scatter(part["coord_x"].iloc[0], part["coord_y"].iloc[0], facecolors="none", edgecolors=color, s=30)
        ax.scatter(part["coord_x"].iloc[-1], part["coord_y"].iloc[-1], marker="x", color=color, s=35)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.set_title("Individual trajectory sample")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def control_grid(metrics_path: Path, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if not metrics_path.exists():
        return False
    control = pd.read_csv(metrics_path)
    cols = [c for c in ["knn_2d_balanced_accuracy", "silhouette_by_true_q_top_action", "branch_separation_ratio"] if c in control.columns]
    if not cols:
        return False
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = control.sort_values(col, ascending=False)
        ax.bar(data["feature_set"], data[col], color="#4C78A8")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def centroid_plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(8, 7))
    for action in ACTIONS:
        part = df[df["q_top_action"].astype(str).eq(action)]
        if part.empty:
            continue
        ax.scatter(part["coord_x"], part["coord_y"], s=6, alpha=0.25, color=COLORS[action], edgecolors="none", rasterized=True)
        xy = part[["coord_x", "coord_y"]].to_numpy(dtype=float)
        mu = xy.mean(axis=0)
        ax.scatter([mu[0]], [mu[1]], marker="*", s=180, color=COLORS[action], edgecolors="black", label=action)
        if len(xy) >= 3:
            cov = np.cov(xy.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse(mu, width=2 * np.sqrt(max(vals[0], 0)), height=2 * np.sqrt(max(vals[1], 0)), angle=angle, fill=False, color=COLORS[action], linewidth=2)
            ax.add_patch(ell)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.set_title("Action centroids and one-sigma ellipses")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stability_grid(stability_path: Path, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if not stability_path.exists():
        return False
    stab = pd.read_csv(stability_path)
    if stab.empty or "mean_pairwise_jaccard" not in stab.columns:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = stab["view_name"].astype(str)
    ax.bar(labels, stab["mean_pairwise_jaccard"], color="#54A24B")
    ax.set_ylabel("mean split-pair top-512 Jaccard")
    ax.set_title("Feature importance stability")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    figure_dir = ensure_dir(args.figure_dir)
    plot_columns = [
        "row_id", "trajectory_id", "decision_id", "progress", "prefix_token_index", "q_top_action",
        "top_commitment", "entropy_commitment", "agent_action", "strict_pre_action", "split_id",
        "split_role", "view_name", "layer", "representation", "feature_set", "embedding_method",
        "residualize", "coord_x", "coord_y", "embedding_coordinate_source",
    ]
    emb = pd.read_parquet(args.embeddings, columns=[c for c in plot_columns if c])
    primary = choose_primary(emb)
    if primary.empty:
        raise RuntimeError("No embedding rows available for plotting")
    plot_rows = []

    for name, col, title in [
        ("by_true_q_top_action.png", "q_top_action", "Activation embedding by true q-top action"),
        ("by_entropy_commitment.png", "entropy_commitment", "Activation embedding by entropy commitment"),
        ("by_top_commitment.png", "top_commitment", "Activation embedding by top commitment"),
        ("by_progress.png", "progress", "Activation embedding by progress"),
    ]:
        scatter(primary, figure_dir / name, col, title, args.max_points, args.seed)
        plot_rows.append({"figure": name, "color": col, "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    low_high(primary, figure_dir / "low_vs_high_commitment.png")
    plot_rows.append({"figure": "low_vs_high_commitment.png", "color": "q_top_action", "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    high = primary[primary["top_commitment"] >= 0.75]
    if not high.empty:
        scatter(high, figure_dir / "high_commitment_by_true_q_top_action.png", "q_top_action", "High-commitment rows by true q-top action", args.max_points, args.seed)
        plot_rows.append({"figure": "high_commitment_by_true_q_top_action.png", "color": "q_top_action", "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    if "strict_pre_action" in primary.columns and primary["strict_pre_action"].fillna(False).astype(bool).any():
        strict = primary[primary["strict_pre_action"].fillna(False).astype(bool)]
        scatter(strict, figure_dir / "strict_pre_action_by_true_q_top_action.png", "q_top_action", "Strict pre-action rows by true q-top action", args.max_points, args.seed)
        plot_rows.append({"figure": "strict_pre_action_by_true_q_top_action.png", "color": "q_top_action", "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    median_trajectory(primary, figure_dir / "median_trajectory_by_true_q_top_action.png")
    plot_rows.append({"figure": "median_trajectory_by_true_q_top_action.png", "color": "q_top_action", "source": "median coord_x/coord_y from activation embedding"})
    individual_trajectories(primary, figure_dir / "individual_trajectory_sample.png", args.seed)
    plot_rows.append({"figure": "individual_trajectory_sample.png", "color": "q_top_action", "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    if control_grid(Path(args.metrics), figure_dir / "control_comparison_grid.png"):
        plot_rows.append({"figure": "control_comparison_grid.png", "color": "metrics", "source": "held-out embedding quality metrics"})
    centroid_plot(primary, figure_dir / "centroid_ellipse_plot.png")
    plot_rows.append({"figure": "centroid_ellipse_plot.png", "color": "q_top_action", "source": "coord_x/coord_y from PCA/UMAP on activation features"})
    if stability_grid(Path(args.stability), figure_dir / "stability_grid.png"):
        plot_rows.append({"figure": "stability_grid.png", "color": "stability", "source": "feature importance stability across grouped splits"})
    plot_index = pd.DataFrame(plot_rows)
    plot_index.to_csv(figure_dir / "plot_index.csv", index=False)
    save_markdown(figure_dir / "plot_report.md", "# Probe-Guided Activation Embedding Plot Index\n\n" + plot_index.to_markdown(index=False))
    print(f"Wrote {figure_dir / 'plot_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
