#!/usr/bin/env python3
"""Evaluate action structure and controls for probe-guided activation embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from probe_guided_embedding_lib import ACTIONS, action_indices, compute_embedding_metrics, ensure_dir, save_markdown


REQUIRED_COMPARE = ["probe_top_512", "all_dims", "random_top_512", "shuffled_probe_top_512", "progress_probe_top_512"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/embeddings/probe_guided_activation_embeddings.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def centroid_distance_stats(df: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    coords = df[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)
    y = action_indices(df["q_top_action"])
    centroids = {}
    for c in range(len(ACTIONS)):
        part = coords[train][y[train] == c]
        if len(part):
            centroids[c] = part.mean(axis=0)
    dists = np.full(len(test), np.nan, dtype=np.float32)
    for n, idx in enumerate(test):
        c = int(y[idx])
        if c in centroids:
            dists[n] = float(np.linalg.norm(coords[idx] - centroids[c]))
    out = {}
    if "entropy_commitment" in df.columns:
        try:
            out["distance_to_true_action_centroid_entropy_commitment_spearman"] = float(spearmanr(dists, pd.to_numeric(df.iloc[test]["entropy_commitment"], errors="coerce")).statistic)
        except Exception:
            out["distance_to_true_action_centroid_entropy_commitment_spearman"] = float("nan")
    reductions = []
    if "decision_id" in df.columns and "prefix_token_index" in df.columns:
        test_df = df.iloc[test].copy()
        test_df["_dist"] = dists
        for _decision, part in test_df.sort_values("prefix_token_index").groupby("decision_id", dropna=False):
            vals = part["_dist"].dropna().to_numpy(dtype=float)
            if len(vals) >= 2:
                reductions.append(vals[0] - vals[-1])
    out["trajectory_movement_toward_true_action_centroid"] = float(np.nanmean(reductions)) if reductions else float("nan")
    return out


def plot_control_bar(control: pd.DataFrame, figure_dir: Path, metric: str) -> Path:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    data = control.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["feature_set"], data[metric], color="#4C78A8")
    ax.set_ylabel(metric)
    ax.set_title(f"Probe-guided embedding control comparison: {metric}")
    ax.tick_params(axis="x", rotation=25)
    path = figure_dir / f"metric_{metric}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    fig = ensure_dir(args.figure_dir)
    precomputed = out / "embeddings" / "embedding_quality_metrics_precomputed.csv"
    if precomputed.exists():
        metrics = pd.read_csv(precomputed)
    else:
        emb = pd.read_parquet(args.embeddings)
        required = ["coord_x", "coord_y", "q_top_action", "split_id", "split_role", "feature_set", "embedding_method", "residualize", "layer", "representation"]
        missing = [c for c in required if c not in emb.columns]
        if missing:
            raise ValueError(f"Embedding table missing required columns: {missing}")
        if emb["forbidden_coordinate_source_used"].fillna(True).astype(bool).any():
            raise RuntimeError("Embedding table marks forbidden coordinate sources as used")
        if not emb["embedding_coordinate_source"].isin(["PCA_on_activation_features", "UMAP_on_activation_features"]).all():
            raise RuntimeError("Unexpected embedding_coordinate_source values")

        metric_rows = []
        group_cols = ["split_id", "view_name", "layer", "representation", "feature_set", "embedding_method", "residualize"]
        for key, part in emb.groupby(group_cols, dropna=False):
            part = part.reset_index(drop=True)
            train = np.flatnonzero(part["split_role"].astype(str).eq("train").to_numpy())
            test = np.flatnonzero(part["split_role"].astype(str).eq("test").to_numpy())
            if len(train) == 0 or len(test) == 0:
                continue
            labels = action_indices(part["q_top_action"])
            coords = part[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)
            metrics_one = compute_embedding_metrics(coords, labels, train, test, part, seed=args.seed + int(part["split_id"].iloc[0]))
            metrics_one.update(centroid_distance_stats(part, train, test))
            row = dict(zip(group_cols, key))
            row.update({"n_train_rows": int(len(train)), "n_test_rows": int(len(test)), **metrics_one})
            metric_rows.append(row)
        metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "embedding_quality_metrics.csv", index=False)
    pca = metrics[metrics["embedding_method"].eq("pca") & metrics["residualize"].eq("none")]
    metric_cols = [
        "silhouette_by_true_q_top_action",
        "branch_separation_ratio",
        "nearest_centroid_balanced_accuracy",
        "knn_2d_balanced_accuracy",
        "logistic_2d_balanced_accuracy",
        "spearman_coord_x_progress",
        "spearman_coord_y_progress",
        "r2_predicting_progress_from_2d_embedding",
        "distance_to_true_action_centroid_entropy_commitment_spearman",
        "trajectory_movement_toward_true_action_centroid",
    ]
    present_metric_cols = [c for c in metric_cols if c in pca.columns]
    control = pca.groupby("feature_set", as_index=False)[present_metric_cols].mean()
    control["is_required_comparison"] = control["feature_set"].isin(REQUIRED_COMPARE)
    best_probe = control[control["feature_set"].isin(["probe_top_512", "probe_weighted_gamma_0.5"])].sort_values("knn_2d_balanced_accuracy", ascending=False).head(1)
    best_control = control[control["feature_set"].isin(["all_dims", "random_top_512", "shuffled_probe_top_512", "progress_probe_top_512"])].sort_values("knn_2d_balanced_accuracy", ascending=False).head(1)
    probe_score = float(best_probe["knn_2d_balanced_accuracy"].iloc[0]) if not best_probe.empty else float("nan")
    control_score = float(best_control["knn_2d_balanced_accuracy"].iloc[0]) if not best_control.empty else float("nan")
    control["best_probe_feature_set"] = None if best_probe.empty else str(best_probe["feature_set"].iloc[0])
    control["best_required_control_feature_set"] = None if best_control.empty else str(best_control["feature_set"].iloc[0])
    control["best_probe_knn_2d_balanced_accuracy"] = probe_score
    control["best_required_control_knn_2d_balanced_accuracy"] = control_score
    control["probe_guided_improves_over_best_required_control"] = bool(np.isfinite(probe_score) and np.isfinite(control_score) and probe_score > control_score)
    control.to_csv(out / "embedding_control_comparison.csv", index=False)

    plot_rows = []
    for metric in ["knn_2d_balanced_accuracy", "silhouette_by_true_q_top_action", "branch_separation_ratio"]:
        if metric in control.columns:
            path = plot_control_bar(control, fig, metric)
            plot_rows.append({"figure": path.name, "metric": metric, "description": "control comparison metric"})
    pd.DataFrame(plot_rows).to_csv(fig / "metric_plot_index.csv", index=False)
    save_markdown(out / "embedding_evaluation_report.md", "# Embedding Evaluation Report\n\n" + control.to_markdown(index=False) + "\n")
    print(f"Wrote {out / 'embedding_quality_metrics.csv'}")
    print(control.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
