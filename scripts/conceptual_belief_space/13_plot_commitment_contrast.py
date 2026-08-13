#!/usr/bin/env python3
"""Plot commitment-only activation contrast projections.

The plotted coordinates come from train-fitted high-dimensional commitment
contrast axes.  No PCA or UMAP is used by this plotting script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_OUTPUT_DIR
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.plotting import savefig


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "commitment_contrast"
FIGURE_DIR = Path("figures/conceptual_belief_space/commitment_contrast")
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
COMMITMENT_BIN_COLORS = {"low": "#4C78A8", "mid": "#B9B9B9", "high": "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--figure_dir", default=FIGURE_DIR)
    parser.add_argument("--metrics", default=OUTPUT_DIR / "commitment_projection_metrics.csv")
    parser.add_argument("--axis_fit_counts", default=OUTPUT_DIR / "axis_fit_counts.csv")
    parser.add_argument("--max_points_per_panel", type=int, default=6000)
    parser.add_argument("--max_histograms", type=int, default=36)
    parser.add_argument("--max_trajectory_figures", type=int, default=18)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def finite_metric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def ok_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "status" not in metrics.columns:
        return metrics.copy()
    return metrics[metrics["status"].astype(str).eq("ok")].copy()


def projection_path(row: pd.Series) -> Path | None:
    value = row.get("projection_file")
    if not isinstance(value, str) or not value:
        return None
    path = resolve(value)
    return path if path.exists() else None


def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    keep_mask = df.get("selected_for_axis_fit", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    keep_mask |= ~df.get("is_train", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    keep = df.loc[keep_mask].copy()
    remaining = max(0, max_points - len(keep))
    rest = df.loc[~keep_mask]
    if remaining > 0 and len(rest) > 0:
        keep = pd.concat([keep, rest.sample(n=min(remaining, len(rest)), random_state=seed)], ignore_index=True)
    if len(keep) > max_points:
        keep = keep.sample(n=max_points, random_state=seed)
    return keep


def y_column(df: pd.DataFrame) -> tuple[str, str]:
    col = "z_score_progress_train_scaled"
    if col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum() > 0:
        return col, "train-fitted orthogonal progress contrast"
    return "z_residual_norm_train_scaled", "residual norm from commitment axis"


def centroid_markers(df: pd.DataFrame, y_col: str) -> dict[str, tuple[float, float]]:
    markers = {}
    for name, flag in [("LOW", "selected_low_fit"), ("HIGH", "selected_high_fit")]:
        if flag in df.columns:
            part = df[df[flag].fillna(False).astype(bool)]
            if not part.empty:
                markers[name] = (
                    float(pd.to_numeric(part["z_score_commit_train_scaled"], errors="coerce").mean()),
                    float(pd.to_numeric(part[y_col], errors="coerce").mean()),
                )
    return markers


def plot_grid(
    metric_rows: pd.DataFrame,
    *,
    target: str,
    subset: str,
    axis_variant: str,
    representation: str,
    color_mode: str,
    output_path: Path,
    max_points: int,
    seed: int,
) -> Path | None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    vmin = vmax = None
    if color_mode == "commitment":
        vals = []
        for _, row in metric_rows.iterrows():
            path = projection_path(row)
            if path is None:
                continue
            part = pd.read_parquet(path, columns=["target_value"])
            vals.append(part["target_value"].to_numpy(float))
        if vals:
            all_vals = np.concatenate(vals)
            vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    any_panel = False
    mappable = None
    for ax, layer in zip(axes.ravel(), LAYERS):
        ax.set_title(f"L{layer:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        row_part = metric_rows[metric_rows["layer"].astype(int).eq(layer)]
        if row_part.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            continue
        path = projection_path(row_part.iloc[0])
        if path is None:
            ax.text(0.5, 0.5, "missing file", ha="center", va="center", transform=ax.transAxes)
            continue
        df = pd.read_parquet(path)
        df = sample_for_plot(df, max_points=max_points, seed=seed + int(layer))
        y_col, _ = y_column(df)
        x = pd.to_numeric(df["z_score_commit_train_scaled"], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        if color_mode == "commitment":
            color = pd.to_numeric(df["target_value"], errors="coerce")
            mappable = ax.scatter(x, y, c=color, s=4, alpha=0.25, cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True, edgecolors="none")
        else:
            for label in ["low", "mid", "high"]:
                part = df[df["commitment_bin"].astype(str).eq(label)]
                if not part.empty:
                    ax.scatter(
                        part["z_score_commit_train_scaled"],
                        part[y_col],
                        s=4,
                        alpha=0.28 if label == "mid" else 0.45,
                        color=COMMITMENT_BIN_COLORS[label],
                        label=label,
                        rasterized=True,
                        edgecolors="none",
                    )
        for label, (cx, cy) in centroid_markers(df, y_col).items():
            ax.scatter([cx], [cy], marker="X", s=70, color="#111111" if label == "HIGH" else "#FFFFFF", edgecolors="#111111", linewidths=0.9)
            ax.text(cx, cy, f" {label}", fontsize=7, weight="bold")
        ax.axvline(0, color="#999999", lw=0.5, alpha=0.5)
        any_panel = True
    for ax in axes.ravel()[len(LAYERS):]:
        ax.axis("off")
    if color_mode == "commitment_bin":
        handles = [
            mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COMMITMENT_BIN_COLORS[label], label=label)
            for label in ["low", "mid", "high"]
        ]
        fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    title = (
        "No PCA/UMAP: high-dimensional commitment contrast projection\n"
        f"target={target}, subset={subset}, variant={axis_variant}, repr={representation}, color={color_mode}\n"
        "x = train-fitted low-to-high commitment contrast score; "
        "y = train-fitted orthogonal progress contrast or residual norm"
    )
    fig.suptitle(title, fontsize=10, y=0.99)
    bottom = 0.055 if color_mode == "commitment_bin" else 0.035
    right = 0.88 if color_mode == "commitment" else 0.985
    fig.subplots_adjust(top=0.88, bottom=bottom, left=0.035, right=right, wspace=0.07, hspace=0.28)
    if color_mode == "commitment" and mappable is not None:
        cax = fig.add_axes([0.905, 0.16, 0.014, 0.62])
        fig.colorbar(mappable, cax=cax, label=target)
    if not any_panel:
        plt.close(fig)
        return None
    return savefig(output_path, dpi=200, fig=fig)


def plot_histogram(row: pd.Series, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    path = projection_path(row)
    if path is None:
        return None
    df = pd.read_parquet(path)
    test = df[~df["is_train"].fillna(True).astype(bool)].copy()
    low = test[test["target_value"] <= float(row.get("low_threshold", 0.4))]["z_score_commit_train_scaled"]
    high = test[test["target_value"] >= float(row.get("high_threshold", 0.9))]["z_score_commit_train_scaled"]
    if low.empty or high.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(low, bins=35, alpha=0.65, density=True, label="test low", color=COMMITMENT_BIN_COLORS["low"])
    ax.hist(high, bins=35, alpha=0.65, density=True, label="test high", color=COMMITMENT_BIN_COLORS["high"])
    # Threshold was fit in raw score space. For this histogram, show the split in z units if possible.
    train_scores = df.loc[df["is_train"].fillna(False).astype(bool), "score_commit"].astype(float)
    z_threshold = (float(row.get("threshold_used", np.nan)) - float(train_scores.mean())) / (float(train_scores.std()) or 1.0)
    if np.isfinite(z_threshold):
        ax.axvline(z_threshold, color="black", lw=1.2, linestyle="--", label="train threshold")
    ax.set_xlabel("z-scored low-to-high commitment contrast score")
    ax.set_ylabel("density")
    ax.set_title(
        "No PCA/UMAP: held-out commitment score separation\n"
        f"{row['target']} {row['subset']} {row['axis_variant']} {row['representation']} L{int(row['layer']):02d}\n"
        f"AUC={float(row.get('auc_high_vs_low_test', np.nan)):.3f}, "
        f"bal acc={float(row.get('balanced_accuracy_high_vs_low_test', np.nan)):.3f}, "
        f"d'={float(row.get('d_prime_high_vs_low_test', np.nan)):.3f}",
        fontsize=9,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def plot_score_over_progress(row: pd.Series, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    path = projection_path(row)
    if path is None:
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    grouped = df.groupby("progress_bin").agg(
        progress=("progress", "mean"),
        score_mean=("z_score_commit_train_scaled", "mean"),
        score_sem=("z_score_commit_train_scaled", lambda x: float(np.nanstd(x) / math.sqrt(max(1, len(x))))),
        target_mean=("target_value", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(grouped["progress"], grouped["score_mean"], yerr=grouped["score_sem"], marker="o", lw=1.5, label="activation commitment score")
    ax.set_xlabel("reasoning progress")
    ax.set_ylabel("mean z-scored commitment contrast score")
    ax2 = ax.twinx()
    ax2.plot(grouped["progress"], grouped["target_mean"], color="#E45756", marker="s", lw=1.2, label="mean target")
    ax2.set_ylabel(f"mean {row['target']}")
    ax.set_title(
        "No PCA/UMAP: commitment score over reasoning progress\n"
        f"{row['target']} {row['subset']} {row['axis_variant']} {row['representation']} L{int(row['layer']):02d}",
        fontsize=9,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def select_trajectory_ids(df: pd.DataFrame, max_decisions: int, seed: int) -> list[str]:
    summaries = []
    for decision_id, part in df.groupby("decision_id"):
        part = part.sort_values("prefix_token_index")
        if len(part) < 2:
            continue
        delta = float(part["z_score_commit_train_scaled"].iloc[-1] - part["z_score_commit_train_scaled"].iloc[0])
        final = float(part["target_value"].iloc[-1])
        monotonic = float(part["z_score_commit_train_scaled"].diff().fillna(0).sum())
        summaries.append((decision_id, delta, final, monotonic, len(part)))
    if not summaries:
        return []
    s = pd.DataFrame(summaries, columns=["decision_id", "delta", "final", "monotonic", "n"])
    chosen = []
    chosen.extend(s.sort_values("delta", ascending=False).head(max_decisions // 3)["decision_id"].tolist())
    chosen.extend(s.sort_values("final", ascending=False).head(max_decisions // 3)["decision_id"].tolist())
    rest = s[~s["decision_id"].isin(chosen)]
    if len(rest):
        chosen.extend(rest.sample(n=min(max_decisions - len(chosen), len(rest)), random_state=seed)["decision_id"].tolist())
    return list(dict.fromkeys(map(str, chosen)))[:max_decisions]


def plot_trajectory_phase(row: pd.Series, output_path: Path, seed: int) -> Path | None:
    import matplotlib.pyplot as plt

    path = projection_path(row)
    if path is None:
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    y_col, y_label = y_column(df)
    decision_ids = select_trajectory_ids(df, 32, seed)
    if not decision_ids:
        return None
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("viridis")
    for decision_id in decision_ids:
        part = df[df["decision_id"].astype(str).eq(str(decision_id))].sort_values("prefix_token_index")
        final_target = float(part["target_value"].iloc[-1])
        color = cmap(np.clip(final_target, 0, 1))
        ax.plot(part["z_score_commit_train_scaled"], part[y_col], lw=1.0, alpha=0.75, color=color)
        ax.scatter(part["z_score_commit_train_scaled"].iloc[0], part[y_col].iloc[0], marker="o", facecolors="none", edgecolors=[color], s=36)
        ax.scatter(part["z_score_commit_train_scaled"].iloc[-1], part[y_col].iloc[-1], marker="x", color=color, s=42)
    ax.set_xlabel("train-fitted low-to-high commitment contrast score")
    ax.set_ylabel(y_label)
    ax.set_title(
        "No PCA/UMAP: selected reasoning trajectories in commitment contrast plane\n"
        f"{row['target']} {row['subset']} {row['axis_variant']} {row['representation']} L{int(row['layer']):02d}",
        fontsize=9,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="final prefix target value")
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def plot_metric_heatmap(metrics: pd.DataFrame, *, target: str, subset: str, axis_variant: str, metric_col: str, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    part = metrics[
        metrics["target"].astype(str).eq(target)
        & metrics["subset"].astype(str).eq(subset)
        & metrics["axis_variant"].astype(str).eq(axis_variant)
    ].copy()
    if part.empty or metric_col not in part.columns:
        return None
    pivot = part.pivot_table(index="representation", columns="layer", values=metric_col, aggfunc="mean")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = pivot.to_numpy(float)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"L{int(c):02d}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_title(f"No PCA/UMAP: {metric_col}\n{target}, {subset}, {axis_variant}", fontsize=9)
    fig.colorbar(im, ax=ax, label=metric_col)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if data[i, j] > np.nanmean(data) else "black")
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def plot_control_bars(metrics: pd.DataFrame, *, target: str, subset: str, representation: str, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    part = metrics[
        metrics["target"].astype(str).eq(target)
        & metrics["subset"].astype(str).eq(subset)
        & metrics["representation"].astype(str).eq(representation)
    ].copy()
    if part.empty:
        return None
    # Use each variant's best layer by AUC for a compact control comparison.
    rows = []
    for variant, group in part.groupby("axis_variant"):
        group = group.copy()
        group["_auc"] = finite_metric(group, "auc_high_vs_low_test")
        best = group.sort_values("_auc", ascending=False).head(1)
        if not best.empty:
            rows.append(best.iloc[0])
    if not rows:
        return None
    best = pd.DataFrame(rows)
    metrics_to_plot = [
        ("auc_high_vs_low_test", "AUC"),
        ("balanced_accuracy_high_vs_low_test", "balanced acc"),
        ("partial_spearman_score_target_resid_progress_test", "partial Spearman"),
    ]
    x = np.arange(len(best))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (col, label) in enumerate(metrics_to_plot):
        ax.bar(x + (i - 1) * width, finite_metric(best, col), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", "\n") for v in best["axis_variant"]], fontsize=8)
    ax.axhline(0.5, color="#999999", lw=0.8, linestyle="--", alpha=0.7, label="AUC chance")
    ax.set_ylim(min(0, np.nanmin([finite_metric(best, c).min() for c, _ in metrics_to_plot]) - 0.05), 1.0)
    ax.set_title(
        "No PCA/UMAP: control comparison for commitment contrast axes\n"
        f"{target}, {subset}, {representation}. Action labels used only for balancing/diagnostics.",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    figure_dir = resolve(args.figure_dir)
    for path in [
        figure_dir,
        figure_dir / "projection_grids",
        figure_dir / "trajectories",
        figure_dir / "metrics",
        figure_dir / "strict_pre_action",
    ]:
        ensure_dir(path)

    metrics_path = resolve(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = ok_metrics(pd.read_csv(metrics_path))
    if metrics.empty:
        raise ValueError("No successful commitment projection metrics found.")
    metrics["layer"] = metrics["layer"].astype(int)
    created: list[str] = []
    skipped: list[str] = []

    # Projection grids by layer.
    group_cols = ["target", "subset", "axis_variant", "representation"]
    for keys, group in metrics.groupby(group_cols):
        target, subset, axis_variant, representation = map(str, keys)
        group = group[group["layer"].isin(LAYERS)].copy()
        if group.empty:
            continue
        for color_mode, suffix in [("commitment", "by_commitment"), ("commitment_bin", "by_commitment_bin")]:
            out = figure_dir / "projection_grids" / f"grid_{target}_{subset}_{axis_variant}_{representation}_{suffix}.png"
            try:
                path = plot_grid(
                    group,
                    target=target,
                    subset=subset,
                    axis_variant=axis_variant,
                    representation=representation,
                    color_mode=color_mode,
                    output_path=out,
                    max_points=args.max_points_per_panel,
                    seed=args.random_seed,
                )
                if path:
                    created.append(rel(path))
            except Exception as exc:
                skipped.append(f"{rel(out)}: {type(exc).__name__}: {exc}")

    # Best-row histograms and trajectory figures.
    best = metrics.copy()
    best["_auc"] = finite_metric(best, "auc_high_vs_low_test")
    best["_partial"] = finite_metric(best, "partial_spearman_score_target_resid_progress_test")
    best = best.sort_values(["_auc", "_partial"], ascending=False)
    for _, row in best.head(args.max_histograms).iterrows():
        out = (
            figure_dir
            / "metrics"
            / f"hist_test_score_{row['target']}_{row['subset']}_{row['axis_variant']}_{row['representation']}_L{int(row['layer']):02d}.png"
        )
        try:
            path = plot_histogram(row, out)
            if path:
                created.append(rel(path))
        except Exception as exc:
            skipped.append(f"{rel(out)}: {type(exc).__name__}: {exc}")
        out2 = (
            figure_dir
            / "trajectories"
            / f"score_over_progress_{row['target']}_{row['subset']}_{row['axis_variant']}_{row['representation']}_L{int(row['layer']):02d}.png"
        )
        try:
            path = plot_score_over_progress(row, out2)
            if path:
                created.append(rel(path))
        except Exception as exc:
            skipped.append(f"{rel(out2)}: {type(exc).__name__}: {exc}")

    for _, row in best.head(args.max_trajectory_figures).iterrows():
        out = (
            figure_dir
            / "trajectories"
            / f"selected_trajectories_phase_plane_{row['target']}_{row['subset']}_{row['axis_variant']}_{row['representation']}_L{int(row['layer']):02d}.png"
        )
        try:
            path = plot_trajectory_phase(row, out, args.random_seed)
            if path:
                created.append(rel(path))
        except Exception as exc:
            skipped.append(f"{rel(out)}: {type(exc).__name__}: {exc}")

    for target in sorted(metrics["target"].astype(str).unique()):
        for subset in sorted(metrics["subset"].astype(str).unique()):
            for axis_variant in sorted(metrics["axis_variant"].astype(str).unique()):
                for metric_col, name in [
                    ("auc_high_vs_low_test", "heatmap_auc"),
                    ("spearman_score_target_test", "heatmap_spearman"),
                    ("partial_spearman_score_target_resid_progress_test", "heatmap_partial_spearman"),
                ]:
                    out = figure_dir / "metrics" / f"{name}_{target}_{subset}_{axis_variant}.png"
                    try:
                        path = plot_metric_heatmap(metrics, target=target, subset=subset, axis_variant=axis_variant, metric_col=metric_col, output_path=out)
                        if path:
                            created.append(rel(path))
                    except Exception as exc:
                        skipped.append(f"{rel(out)}: {type(exc).__name__}: {exc}")
            for representation in sorted(metrics["representation"].astype(str).unique()):
                out = figure_dir / "metrics" / f"control_comparison_{target}_{subset}_{representation}.png"
                try:
                    path = plot_control_bars(metrics, target=target, subset=subset, representation=representation, output_path=out)
                    if path:
                        created.append(rel(path))
                except Exception as exc:
                    skipped.append(f"{rel(out)}: {type(exc).__name__}: {exc}")

    manifest = {"created_figures": created, "skipped": skipped, "n_created": len(created), "n_skipped": len(skipped)}
    write_json(output_dir / "plot_manifest.json", manifest)
    report = [
        "# Commitment Contrast Plot Report",
        "",
        "All figures use train-fitted high-dimensional low-to-high commitment contrast projections.",
        "",
        "- No PCA/UMAP is used as the primary projection.",
        "- x-axis: train-fitted low-to-high commitment contrast score.",
        "- y-axis: train-fitted orthogonal progress contrast or residual norm.",
        "- Action labels are used only for balancing and diagnostics, not as the plotted target.",
        "",
        f"- Created figures: {len(created)}",
        f"- Skipped figures: {len(skipped)}",
    ]
    if skipped:
        report.extend(["", "## Skipped", ""])
        report.extend([f"- {item}" for item in skipped[:100]])
    save_markdown(output_dir / "13_commitment_contrast_plot_report.md", "\n".join(report))
    print(json.dumps({"created": len(created), "skipped": len(skipped), "manifest": rel(output_dir / "plot_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
