#!/usr/bin/env python
"""Plot supervised MLP bottleneck branch embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.mlp_branch_utils import ACTION_COLORS, ACTION_VERTICES, Q_COLS, normalize_q, top_actions_from_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--best_configs", default=None)
    parser.add_argument("--max_points", type=int, default=12000)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def overlay_vertices(ax) -> None:
    for action in ACTIONS:
        x, y = ACTION_VERTICES[action]
        ax.scatter([x], [y], marker="*", s=180, color=ACTION_COLORS[action], edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(float(x) + 0.03, float(y) + 0.03, action, fontsize=9, weight="bold")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("z1_plot = tanh(supervised bottleneck dim 1)")
    ax.set_ylabel("z2_plot = tanh(supervised bottleneck dim 2)")


def sample_rows(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=seed)


def scatter_action(df: pd.DataFrame, color_col: str, path: Path, title: str, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    data = sample_rows(df, max_points, seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    for action in ACTIONS:
        part = data[data[color_col].astype(str).eq(action)]
        if not part.empty:
            ax.scatter(part["z1_plot"], part["z2_plot"], s=10, alpha=0.55, color=ACTION_COLORS[action], label=action, linewidths=0)
    overlay_vertices(ax)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    savefig(path)


def scatter_numeric(df: pd.DataFrame, color_col: str, path: Path, title: str, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    data = sample_rows(df, max_points, seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(data["z1_plot"], data["z2_plot"], c=data[color_col], s=10, alpha=0.65, cmap="viridis", linewidths=0)
    overlay_vertices(ax)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=color_col)
    savefig(path)


def pick_best(metrics: pd.DataFrame, subset_contains: str | None = None) -> pd.Series | None:
    frame = metrics[
        metrics["model_type"].astype(str).eq("mlp_branch")
        & metrics["status"].astype(str).eq("ok")
    ].copy()
    if subset_contains:
        frame = frame[frame["subset"].astype(str).str.contains(subset_contains, na=False)]
    if frame.empty:
        return None
    metric_cols = ["test_q_top_action_balanced_accuracy", "test_commitment_spearman", "test_vertex_top_action_balanced_accuracy"]
    for col in metric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["_score"] = frame.get("test_q_top_action_balanced_accuracy", 0).fillna(0) + 0.25 * frame.get("test_commitment_spearman", 0).fillna(0)
    return frame.sort_values("_score", ascending=False).iloc[0]


def pick_best_q_prediction_run(metrics: pd.DataFrame, preds: pd.DataFrame, min_predicted_count: int = 1) -> pd.Series | None:
    """Pick a run for q_pred-top-action plots without allowing action collapse.

    The generic plot-best score mixes q accuracy and commitment correlation. That is
    useful for the overall branch visualization, but it can choose a run whose q head
    collapses one action. For the predicted-q-top-action plot, prefer q accuracy and
    require coverage for every action that appears as a true top_action in the test
    split.
    """

    frame = metrics[
        metrics["model_type"].astype(str).eq("mlp_branch")
        & metrics["status"].astype(str).eq("ok")
    ].copy()
    if frame.empty:
        return None
    for col in ["test_q_top_action_balanced_accuracy", "test_q_top_action_accuracy", "test_commitment_spearman"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    q_cols = [f"q_pred_{action}" for action in ACTIONS]
    keep_rows: list[pd.Series] = []
    fallback_rows: list[pd.Series] = []
    for _, row in frame.iterrows():
        run_preds = rows_for_run(preds, row, "test")
        if run_preds.empty or any(col not in run_preds.columns for col in q_cols):
            continue
        predicted = pd.Series(top_actions_from_q(run_preds[q_cols].to_numpy(dtype=np.float32)))
        true_actions = [
            action
            for action in ACTIONS
            if "top_action" in run_preds.columns and int(run_preds["top_action"].astype(str).eq(action).sum()) > 0
        ]
        if not true_actions:
            true_actions = ACTIONS
        counts = predicted.value_counts().to_dict()
        row = row.copy()
        row["_min_predicted_action_count"] = min(int(counts.get(action, 0)) for action in true_actions)
        row["_num_predicted_actions"] = int(sum(int(counts.get(action, 0)) > 0 for action in ACTIONS))
        row["_q_plot_score"] = (
            float(row.get("test_q_top_action_balanced_accuracy", 0) or 0)
            + 0.05 * float(row.get("test_q_top_action_accuracy", 0) or 0)
            + 0.02 * float(row.get("test_commitment_spearman", 0) or 0)
        )
        fallback_rows.append(row)
        if row["_min_predicted_action_count"] >= min_predicted_count:
            keep_rows.append(row)

    candidates = keep_rows if keep_rows else fallback_rows
    if not candidates:
        return None
    candidate_frame = pd.DataFrame(candidates)
    return candidate_frame.sort_values(["_q_plot_score", "_min_predicted_action_count"], ascending=False).iloc[0]


def rows_for_run(preds: pd.DataFrame, row: pd.Series | None, role: str = "test") -> pd.DataFrame:
    if row is None:
        return pd.DataFrame()
    data = preds[preds["run_id"].astype(str).eq(str(row["run_id"]))].copy()
    if role:
        data = data[data["split_role"].astype(str).eq(role)]
    return data


def train_val_test_comparison(preds: pd.DataFrame, row: pd.Series, path: Path, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    data = preds[preds["run_id"].astype(str).eq(str(row["run_id"]))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, role in zip(axes, ["train", "val", "test"]):
        part = sample_rows(data[data["split_role"].astype(str).eq(role)], max_points // 3, seed)
        for action in ACTIONS:
            sub = part[part["top_action"].astype(str).eq(action)]
            if not sub.empty:
                ax.scatter(sub["z1_plot"], sub["z2_plot"], s=8, alpha=0.5, color=ACTION_COLORS[action], label=action, linewidths=0)
        overlay_vertices(ax)
        ax.set_title(f"Supervised MLP bottleneck: {role}")
    axes[-1].legend(loc="upper right", fontsize=8)
    savefig(path)


def trajectory_plot(df: pd.DataFrame, path: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    rng = np.random.default_rng(seed)
    decision_stats = (
        df.groupby("decision_id")
        .agg(n=("row_id", "size"), gain=("entropy_commitment", lambda s: float(s.max() - s.min())), agent_action=("agent_action", "last"))
        .reset_index()
    )
    selected = []
    selected.extend(decision_stats.sort_values("gain", ascending=False).head(16)["decision_id"].tolist())
    for action in ACTIONS:
        candidates = decision_stats[decision_stats["agent_action"].astype(str).eq(action)]["decision_id"].tolist()
        if candidates:
            selected.append(candidates[0])
    remaining = [d for d in decision_stats["decision_id"].tolist() if d not in selected]
    if remaining:
        selected.extend(rng.choice(remaining, size=min(20, len(remaining)), replace=False).tolist())
    selected = list(dict.fromkeys(selected))[:50]
    fig, ax = plt.subplots(figsize=(8, 7))
    for decision_id in selected:
        part = df[df["decision_id"].astype(str).eq(str(decision_id))].sort_values("prefix_token_index")
        if len(part) < 2:
            continue
        color = ACTION_COLORS.get(str(part["agent_action"].iloc[-1]), "0.5")
        ax.plot(part["z1_plot"], part["z2_plot"], color=color, alpha=0.35, linewidth=1.0)
        ax.scatter(part["z1_plot"].iloc[0], part["z2_plot"].iloc[0], facecolors="none", edgecolors=color, s=40, linewidth=1.0)
        ax.scatter(part["z1_plot"].iloc[-1], part["z2_plot"].iloc[-1], marker="x", color=color, s=45, linewidth=1.2)
    overlay_vertices(ax)
    ax.set_title("Supervised MLP bottleneck selected activation trajectories")
    savefig(path)


def radius_over_progress(preds: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = preds[preds["model_type"].astype(str).eq("mlp_branch") & preds["split_role"].astype(str).eq("test")].copy()
    if data.empty:
        return
    data["progress_bin"] = pd.cut(data["progress"], bins=np.linspace(0, 1, 11), include_lowest=True)
    summary = data.groupby("progress_bin", observed=False).agg(radius=("z_radius", "mean"), commitment=("entropy_commitment", "mean")).reset_index()
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, summary["radius"], marker="o", label="mean z_radius")
    ax.plot(x, summary["commitment"], marker="s", label="mean entropy_commitment")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in summary["progress_bin"]], rotation=45, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Supervised MLP bottleneck radius over reasoning progress")
    ax.legend()
    savefig(path)


def q_simplex(preds: pd.DataFrame, row: pd.Series, path: Path, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    data = rows_for_run(preds, row, "test")
    if data.empty:
        return
    q_cols = [f"q_pred_{a}" for a in ACTIONS]
    q = normalize_q(sample_rows(data, max_points, seed)[q_cols].to_numpy(dtype=np.float32))
    vertices = np.stack([ACTION_VERTICES[a] for a in ACTIONS], axis=0)
    z = q @ vertices
    top = top_actions_from_q(q)
    fig, ax = plt.subplots(figsize=(7, 6))
    for action in ACTIONS:
        mask = top == action
        ax.scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.55, color=ACTION_COLORS[action], label=action, linewidths=0)
    overlay_vertices(ax)
    ax.set_title("Supervised MLP bottleneck predicted q projected to action vertices")
    ax.legend(loc="upper right", fontsize=8)
    savefig(path)


def baseline_bars(metrics: pd.DataFrame, row: pd.Series, path: Path) -> None:
    import matplotlib.pyplot as plt

    keys = ["split_id", "layer", "representation", "subset"]
    frame = metrics.copy()
    for key in keys:
        frame = frame[frame[key].astype(str).eq(str(row[key]))]
    if frame.empty:
        return
    order = ["mlp_branch", "linear_baseline", "mlp_shuffled_target", "progress_only", "label_permutation"]
    frame["model_type"] = pd.Categorical(frame["model_type"], categories=order, ordered=True)
    frame = frame.sort_values("model_type")
    metrics_to_plot = [
        ("test_q_top_action_balanced_accuracy", "top-action bal. acc."),
        ("test_commitment_spearman", "commitment Spearman"),
        ("test_q_kl", "q KL (lower better)"),
        ("test_vertex_top_action_balanced_accuracy", "vertex bal. acc."),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4))
    for ax, (col, label) in zip(axes, metrics_to_plot):
        if col not in frame.columns:
            continue
        ax.bar(frame["model_type"].astype(str), pd.to_numeric(frame[col], errors="coerce"))
        ax.set_title(f"Supervised MLP bottleneck {label}")
        ax.tick_params(axis="x", rotation=45)
    savefig(path)


def heatmap(metrics: pd.DataFrame, path: Path, metric_col: str = "test_q_top_action_balanced_accuracy") -> None:
    import matplotlib.pyplot as plt

    frame = metrics[metrics["model_type"].astype(str).eq("mlp_branch") & metrics["status"].astype(str).eq("ok")].copy()
    if frame.empty or metric_col not in frame.columns:
        return
    pivot = frame.groupby(["representation", "layer"], observed=False)[metric_col].mean().unstack("layer")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Representation")
    ax.set_title(f"Supervised MLP bottleneck heatmap: {metric_col}")
    fig.colorbar(im, ax=ax)
    savefig(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    metrics_path = Path(args.metrics) if args.metrics else output_dir / "mlp_branch_metrics.csv"
    predictions_path = Path(args.predictions) if args.predictions else output_dir / "predictions/mlp_branch_predictions.parquet"
    configs_path = Path(args.best_configs) if args.best_configs else output_dir / "best_mlp_branch_configs.json"
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_parquet(predictions_path)
    if preds.empty:
        raise ValueError("prediction table is empty")
    best = pick_best(metrics)
    if best is None:
        raise ValueError("no successful mlp_branch metric rows found")
    test = rows_for_run(preds, best, "test")
    if test.empty:
        raise ValueError("best run has no test predictions")
    q_plot_best = pick_best_q_prediction_run(metrics, preds)
    q_plot_test = rows_for_run(preds, q_plot_best, "test")
    if q_plot_test.empty:
        q_plot_best = best
        q_plot_test = test

    branch_dir = ensure_dir(figure_dir / "branch_manifold")
    traj_dir = ensure_dir(figure_dir / "trajectories")
    metric_dir = ensure_dir(figure_dir / "metrics")
    manifest: dict[str, str] = {}
    items = [
        ("best_branch_by_top_action", lambda p: scatter_action(test, "top_action", p, "Supervised MLP bottleneck action-belief embedding by top_action — test rows", args.max_points, args.random_seed)),
        ("best_branch_by_agent_action", lambda p: scatter_action(test, "agent_action", p, "Supervised MLP bottleneck action-belief embedding by agent_action — test rows", args.max_points, args.random_seed)),
        ("best_branch_by_entropy_commitment", lambda p: scatter_numeric(test, "entropy_commitment", p, "Supervised MLP bottleneck action-belief embedding by entropy_commitment — test rows", args.max_points, args.random_seed)),
        ("best_branch_by_progress", lambda p: scatter_numeric(test, "progress", p, "Supervised MLP bottleneck action-belief embedding by progress — test rows", args.max_points, args.random_seed)),
        ("best_branch_by_predicted_top_action", lambda p: scatter_action(q_plot_test.assign(predicted_top_action=top_actions_from_q(q_plot_test[[f'q_pred_{a}' for a in ACTIONS]].to_numpy(dtype=np.float32))), "predicted_top_action", p, "Supervised MLP bottleneck by predicted q top action — q-selected test rows", args.max_points, args.random_seed)),
        ("branch_train_val_test_comparison", lambda p: train_val_test_comparison(preds, best, p, args.max_points, args.random_seed)),
    ]
    for name, fn in items:
        path = branch_dir / f"{name}.png"
        fn(path)
        manifest[name] = str(path)

    strict_best = pick_best(metrics, subset_contains="strict")
    strict_test = rows_for_run(preds, strict_best, "test")
    if not strict_test.empty:
        path = branch_dir / "strict_pre_action_branch_by_top_action.png"
        scatter_action(strict_test, "top_action", path, "Supervised MLP bottleneck strict pre-action branch by top_action", args.max_points, args.random_seed)
        manifest["strict_pre_action_branch_by_top_action"] = str(path)
        path = branch_dir / "strict_pre_action_branch_by_commitment.png"
        scatter_numeric(strict_test, "entropy_commitment", path, "Supervised MLP bottleneck strict pre-action branch by entropy_commitment", args.max_points, args.random_seed)
        manifest["strict_pre_action_branch_by_commitment"] = str(path)

    path = traj_dir / "selected_branch_trajectories.png"
    trajectory_plot(test, path, args.random_seed)
    manifest["selected_branch_trajectories"] = str(path)
    path = traj_dir / "radius_over_progress.png"
    radius_over_progress(preds, path)
    manifest["radius_over_progress"] = str(path)
    path = traj_dir / "predicted_q_simplex_trajectory.png"
    q_simplex(preds, best, path, args.max_points, args.random_seed)
    manifest["predicted_q_simplex_trajectory"] = str(path)
    path = metric_dir / "mlp_branch_vs_baselines.png"
    baseline_bars(metrics, best, path)
    manifest["mlp_branch_vs_baselines"] = str(path)
    path = metric_dir / "mlp_branch_metric_heatmap_by_layer_representation.png"
    heatmap(metrics, path)
    manifest["mlp_branch_metric_heatmap_by_layer_representation"] = str(path)

    manifest_path = output_dir / "plot_manifest.json"
    write_json(manifest_path, manifest)
    report = [
        "# MLP Branch Plot Report",
        "",
        "All figures show a supervised MLP bottleneck embedding. They are not unsupervised raw activation manifolds.",
        "",
        f"- best run_id: `{best['run_id']}`",
        f"- predicted-q plot run_id: `{q_plot_best['run_id'] if q_plot_best is not None else best['run_id']}`",
        f"- figures created: {len(manifest)}",
        f"- manifest: `{manifest_path}`",
    ]
    save_markdown(output_dir / "reports/17_mlp_branch_plot_report.md", "\n".join(report))
    print(json.dumps({"figures": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
