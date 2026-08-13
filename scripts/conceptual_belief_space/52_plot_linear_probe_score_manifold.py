#!/usr/bin/env python3
"""Plot the supervised linear-probe score manifold."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS, EPS
from src.conceptual_belief_space.plotting import action_color_map, savefig

VERTICES = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x[mask], y[mask]).statistic)
    except Exception:
        return float(pd.Series(x[mask]).rank().corr(pd.Series(y[mask]).rank()))


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: pd.DataFrame) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    cov = covariates.copy()
    for col in cov.columns:
        cov[col] = pd.to_numeric(cov[col], errors="coerce")
        mask &= np.isfinite(cov[col].to_numpy(dtype=float))
    if mask.sum() < len(cov.columns) + 4:
        return float("nan")
    rx = pd.Series(x[mask]).rank().to_numpy(dtype=float)
    ry = pd.Series(y[mask]).rank().to_numpy(dtype=float)
    ranked_cov = np.column_stack([pd.Series(cov[col].to_numpy(dtype=float)[mask]).rank().to_numpy(dtype=float) for col in cov.columns])
    rx = (rx - rx.mean()) / np.clip(rx.std(), EPS, None)
    ry = (ry - ry.mean()) / np.clip(ry.std(), EPS, None)
    ranked_cov = (ranked_cov - ranked_cov.mean(axis=0, keepdims=True)) / np.clip(ranked_cov.std(axis=0, keepdims=True), EPS, None)

    def residualize_vector(values: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        residual = values.astype(float).copy()
        # Repeated one-dimensional projections are stable for the tiny covariate
        # sets used here and avoid ill-conditioned matrix inversions.
        for _ in range(2):
            for col_idx in range(cov_matrix.shape[1]):
                cov_vec = cov_matrix[:, col_idx]
                denom = float(np.dot(cov_vec, cov_vec))
                if denom > EPS:
                    residual = residual - (float(np.dot(residual, cov_vec)) / denom) * cov_vec
            residual = residual - residual.mean()
        return residual

    res_x = residualize_vector(rx, ranked_cov)
    res_y = residualize_vector(ry, ranked_cov)
    if np.std(res_x) < EPS or np.std(res_y) < EPS:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def sample_for_plot(df: pd.DataFrame, max_points: int = 60000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=seed)


def setup_ax(ax, title: str):
    ax.axhline(0, color="#999999", lw=0.8, alpha=0.45)
    ax.axvline(0, color="#999999", lw=0.8, alpha=0.45)
    ax.set_xlabel("linear_x = centered RIGHT score - centered LEFT score")
    ax.set_ylabel("linear_y = centered UP score - centered DOWN score")
    ax.set_title(title)


def overlay_anchors(ax):
    for action, (x, y) in VERTICES.items():
        ax.scatter([x], [y], marker="*", s=260, color=action_color_map()[action], edgecolor="black", linewidth=1.1, zorder=5)
        ax.text(x + 0.035, y + 0.035, action, fontsize=10, weight="bold")


def categorical_plane(df: pd.DataFrame, color_col: str, path: Path, title: str, *, alpha: float = 0.5):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = action_color_map()
    plot_df = sample_for_plot(df)
    for action in ACTIONS:
        part = plot_df[plot_df[color_col].astype(str).eq(action)]
        if part.empty:
            continue
        ax.scatter(part["linear_x"], part["linear_y"], s=8, alpha=alpha, color=colors[action], label=action, edgecolors="none", rasterized=True)
    overlay_anchors(ax)
    setup_ax(ax, title)
    ax.legend(frameon=True, fontsize=9)
    savefig(path, fig=fig)


def numeric_plane(df: pd.DataFrame, value_col: str, path: Path, title: str, colorbar_label: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = sample_for_plot(df)
    sc = ax.scatter(plot_df["linear_x"], plot_df["linear_y"], c=plot_df[value_col], s=8, alpha=0.55, cmap="viridis", edgecolors="none", rasterized=True)
    overlay_anchors(ax)
    setup_ax(ax, title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label)
    savefig(path, fig=fig)


def low_high_comparison(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    colors = action_color_map()
    panels = [("top_commitment < 0.5", df[df["top_commitment"] < 0.5]), ("top_commitment >= 0.75", df[df["top_commitment"] >= 0.75])]
    for ax, (title, part) in zip(axes, panels):
        plot_df = sample_for_plot(part, max_points=30000)
        for action in ACTIONS:
            sub = plot_df[plot_df["true_q_top_action"].astype(str).eq(action)]
            if not sub.empty:
                ax.scatter(sub["linear_x"], sub["linear_y"], s=8, alpha=0.5, color=colors[action], label=action, edgecolors="none", rasterized=True)
        overlay_anchors(ax)
        setup_ax(ax, title)
    axes[1].legend(frameon=True, fontsize=9)
    savefig(path, fig=fig)


def selected_trajectories(df: pd.DataFrame, path: Path, max_decisions: int = 40):
    import matplotlib.pyplot as plt

    if "decision_id" not in df.columns:
        return
    last = df.sort_values(["decision_id", "prefix_token_index"]).groupby("decision_id", as_index=False).tail(1)
    chosen = []
    for action in ACTIONS:
        part = last[last["true_q_top_action"].astype(str).eq(action)].sort_values("top_commitment", ascending=False).head(max_decisions // len(ACTIONS) + 1)
        chosen.extend(part["decision_id"].astype(str).tolist())
    chosen = chosen[:max_decisions]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = action_color_map()
    for decision_id in chosen:
        part = df[df["decision_id"].astype(str).eq(decision_id)].sort_values("prefix_token_index")
        if part.empty:
            continue
        action = str(part["true_q_top_action"].iloc[-1])
        color = colors.get(action, "#666666")
        ax.plot(part["linear_x"], part["linear_y"], lw=1.0, alpha=0.55, color=color)
        ax.scatter(part["linear_x"].iloc[0], part["linear_y"].iloc[0], marker="o", facecolors="none", edgecolors=color, s=36)
        ax.scatter(part["linear_x"].iloc[-1], part["linear_y"].iloc[-1], marker="x", color=color, s=44)
    overlay_anchors(ax)
    setup_ax(ax, "Linear score trajectories through held-out score plane")
    savefig(path, fig=fig)


def mean_trajectory_by_action(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    if "progress" not in df.columns:
        return
    tmp = df.copy()
    tmp["progress_bin"] = pd.cut(tmp["progress"].clip(0, 1), bins=np.linspace(0, 1, 11), include_lowest=True, labels=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = action_color_map()
    for action in ACTIONS:
        part = tmp[tmp["true_q_top_action"].astype(str).eq(action)]
        med = part.groupby("progress_bin", as_index=False).agg(linear_x=("linear_x", "median"), linear_y=("linear_y", "median"))
        med = med.dropna()
        if len(med) < 2:
            continue
        ax.plot(med["linear_x"], med["linear_y"], marker="o", lw=2.0, color=colors[action], label=action)
        for i in range(len(med) - 1):
            ax.annotate("", xy=(med["linear_x"].iloc[i + 1], med["linear_y"].iloc[i + 1]), xytext=(med["linear_x"].iloc[i], med["linear_y"].iloc[i]), arrowprops={"arrowstyle": "->", "color": colors[action], "lw": 1.2})
    overlay_anchors(ax)
    setup_ax(ax, "Median score-plane trajectory by true q-top action")
    ax.legend(frameon=True, fontsize=9)
    savefig(path, fig=fig)


def radius_scatter(df: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str):
    import matplotlib.pyplot as plt

    plot_df = sample_for_plot(df, max_points=60000)
    rho = spearman(plot_df[x_col].to_numpy(float), plot_df[y_col].to_numpy(float))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(plot_df[x_col], plot_df[y_col], s=7, alpha=0.35, color="#4C78A8", edgecolors="none", rasterized=True)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\nSpearman rho = {rho:.3f}")
    savefig(path, fig=fig)


def partial_control(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    values = {
        "radius~entropy_commitment": spearman(df["score_radius"].to_numpy(float), df["entropy_commitment"].to_numpy(float)),
        "radius~progress": spearman(df["score_radius"].to_numpy(float), df["progress"].to_numpy(float)) if "progress" in df.columns else float("nan"),
    }
    if "progress" in df.columns:
        values["radius~entropy | progress"] = partial_spearman(df["score_radius"].to_numpy(float), df["entropy_commitment"].to_numpy(float), df[["progress"]])
        cov_cols = [c for c in ["progress", "prefix_token_index", "num_cot_tokens"] if c in df.columns]
        if cov_cols:
            values["radius~entropy | progress/prefix/len"] = partial_spearman(df["score_radius"].to_numpy(float), df["entropy_commitment"].to_numpy(float), df[cov_cols])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(values)
    vals = [values[k] for k in labels]
    ax.bar(np.arange(len(labels)), vals, color="#4C78A8")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Spearman / partial Spearman")
    ax.set_title("Score radius, commitment, and progress controls")
    for i, val in enumerate(vals):
        if np.isfinite(val):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    savefig(path, fig=fig)


def simplex_3d(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    plot_df = sample_for_plot(df, max_points=30000)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = action_color_map()
    for action in ACTIONS:
        part = plot_df[plot_df["true_q_top_action"].astype(str).eq(action)]
        if part.empty:
            continue
        ax.scatter(part["h1"], part["h2"], part["h3"], s=6, alpha=0.35, color=colors[action], label=action, rasterized=True)
    ax.set_xlabel("h1")
    ax.set_ylabel("h2")
    ax.set_zlabel("h3")
    ax.set_title("Centered stacker scores in 3D Helmert coordinates")
    ax.legend(frameon=True, fontsize=9)
    savefig(path, fig=fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifold", default="outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold.parquet")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/linear_probe_score_manifold")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.manifold)
    if "in_sample" in df.columns:
        main_df = df[~df["in_sample"].fillna(False).astype(bool)].copy()
        if main_df.empty:
            main_df = df.copy()
    else:
        main_df = df.copy()
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    categorical_plane(main_df, "true_q_top_action", figure_dir / "linear_score_plane_by_true_q_top.png", "Cross-fitted linear probe score manifold by true q-top action")
    categorical_plane(main_df, "pred_q_top", figure_dir / "linear_score_plane_by_pred_q_top.png", "Cross-fitted linear probe score manifold by predicted q-top action")
    numeric_plane(main_df, "entropy_commitment", figure_dir / "linear_score_plane_by_entropy_commitment.png", "Cross-fitted score plane by true entropy commitment", "entropy_commitment")
    if "progress" in main_df.columns:
        numeric_plane(main_df, "progress", figure_dir / "linear_score_plane_by_progress.png", "Cross-fitted score plane by reasoning progress", "progress")
    high = main_df[main_df["top_commitment"] >= 0.75]
    if not high.empty:
        categorical_plane(high, "true_q_top_action", figure_dir / "high_commitment_linear_score_plane_by_true_q_top.png", "High-commitment held-out score plane by true q-top action")
    low_high_comparison(main_df, figure_dir / "low_vs_high_commitment_comparison.png")
    selected_trajectories(main_df, figure_dir / "linear_score_trajectories.png")
    mean_trajectory_by_action(main_df, figure_dir / "mean_trajectory_by_action.png")
    if "strict_before_any_action_mention" in main_df.columns:
        strict = main_df[main_df["strict_before_any_action_mention"].fillna(False).astype(bool)]
        if not strict.empty:
            categorical_plane(strict, "true_q_top_action", figure_dir / "strict_pre_action_linear_score_plane.png", "Strict pre-action held-out score plane by true q-top action")
            numeric_plane(strict, "entropy_commitment", figure_dir / "strict_pre_action_linear_score_plane_by_entropy_commitment.png", "Strict pre-action held-out score plane by entropy commitment", "entropy_commitment")
    radius_scatter(main_df, "score_radius", "entropy_commitment", figure_dir / "score_radius_vs_commitment.png", "Score radius vs entropy commitment")
    if "progress" in main_df.columns:
        radius_scatter(main_df, "score_radius", "progress", figure_dir / "score_radius_vs_progress.png", "Score radius vs reasoning progress")
    partial_control(main_df, figure_dir / "partial_radius_commitment_control.png")
    simplex_3d(main_df, figure_dir / "optional_3d_centered_score_simplex.png")
    (figure_dir / "plot_manifest.txt").write_text("\n".join(str(p.name) for p in sorted(figure_dir.glob("*.png"))) + "\n")
    print(f"Wrote figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
