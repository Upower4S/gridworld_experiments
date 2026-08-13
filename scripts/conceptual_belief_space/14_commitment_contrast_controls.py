#!/usr/bin/env python3
"""Summarize controls for commitment-only activation contrast axes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--metrics", default=OUTPUT_DIR / "commitment_projection_metrics.csv")
    parser.add_argument("--axis_fit_counts", default=OUTPUT_DIR / "axis_fit_counts.csv")
    parser.add_argument("--max_projection_files_for_diagnostics", type=int, default=80)
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


def metric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def ok_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "status" not in metrics.columns:
        return metrics.copy()
    return metrics[metrics["status"].astype(str).eq("ok")].copy()


def best_by(group: pd.DataFrame, col: str) -> pd.Series | None:
    if group.empty or col not in group.columns:
        return None
    part = group.copy()
    part["_metric"] = metric(part, col)
    part = part[np.isfinite(part["_metric"])]
    if part.empty:
        return None
    return part.sort_values("_metric", ascending=False).iloc[0]


def projection_path(row: pd.Series) -> Path | None:
    value = row.get("projection_file")
    if not isinstance(value, str) or not value:
        return None
    path = resolve(value)
    return path if path.exists() else None


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    mask = np.isfinite(score)
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], score[mask]))


def action_leakage_diagnostic(row: pd.Series) -> dict[str, Any]:
    path = projection_path(row)
    base = {"action_between_variance_ratio": np.nan, "progress_only_auc_high_vs_low": np.nan, "mean_score_by_action_json": "{}"}
    if path is None:
        return base
    try:
        df = pd.read_parquet(path)
    except Exception:
        return base
    if df.empty:
        return base
    means = df.groupby("agent_action")["score_commit"].mean().to_dict() if "agent_action" in df.columns else {}
    overall_var = float(np.nanvar(df["score_commit"].to_numpy(float)))
    between = float(np.nanvar(list(means.values()))) if means else np.nan
    ratio = between / overall_var if overall_var > 1e-12 and np.isfinite(between) else np.nan
    low_thr = float(row.get("low_threshold", 0.4))
    high_thr = float(row.get("high_threshold", 0.9))
    test = df[~df["is_train"].fillna(True).astype(bool)].copy()
    low = test["target_value"].to_numpy(float) <= low_thr
    high = test["target_value"].to_numpy(float) >= high_thr
    lh = low | high
    progress_auc = auc_score(high[lh].astype(int), test.loc[lh, "progress"].to_numpy(float)) if lh.any() else np.nan
    return {
        "action_between_variance_ratio": ratio,
        "progress_only_auc_high_vs_low": progress_auc,
        "mean_score_by_action_json": json.dumps({str(k): float(v) for k, v in means.items()}, sort_keys=True),
    }


def build_best_axes(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original = metrics[metrics["axis_variant"].astype(str).eq("original_action_balanced")].copy()
    for keys, group in original.groupby(["target", "subset", "representation"]):
        target, subset, representation = keys
        for criterion, col in [
            ("best_test_auc", "auc_high_vs_low_test"),
            ("best_partial_spearman", "partial_spearman_score_target_resid_progress_test"),
            ("best_within_progress_spearman", "spearman_score_target_within_progress_bin_mean_test"),
        ]:
            best = best_by(group, col)
            if best is None:
                continue
            rows.append(
                {
                    "criterion": criterion,
                    "target": target,
                    "subset": subset,
                    "representation": representation,
                    "layer": int(best["layer"]),
                    "axis_variant": best["axis_variant"],
                    "auc_high_vs_low_test": float(best.get("auc_high_vs_low_test", np.nan)),
                    "balanced_accuracy_high_vs_low_test": float(best.get("balanced_accuracy_high_vs_low_test", np.nan)),
                    "partial_spearman_score_target_resid_progress_test": float(best.get("partial_spearman_score_target_resid_progress_test", np.nan)),
                    "spearman_score_target_within_progress_bin_mean_test": float(best.get("spearman_score_target_within_progress_bin_mean_test", np.nan)),
                    "projection_file": best.get("projection_file", ""),
                }
            )
    return pd.DataFrame(rows)


def compare_controls(metrics: pd.DataFrame, best_axes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "auc_high_vs_low_test",
        "balanced_accuracy_high_vs_low_test",
        "partial_spearman_score_target_resid_progress_test",
        "spearman_score_target_within_progress_bin_mean_test",
    ]
    key_cols = ["target", "subset", "representation", "layer"]
    for _, best in best_axes.iterrows():
        if best["criterion"] != "best_test_auc":
            continue
        selector = np.ones(len(metrics), dtype=bool)
        for col in key_cols:
            selector &= metrics[col].astype(str).eq(str(best[col]))
        group = metrics[selector].copy()
        original = group[group["axis_variant"].astype(str).eq("original_action_balanced")]
        if original.empty:
            continue
        original = original.iloc[0]
        diag = action_leakage_diagnostic(original)
        for variant in [
            "original_action_balanced",
            "progress_matched_action_balanced",
            "shuffled_commitment_control",
            "progress_residualized_feature_control",
        ]:
            part = group[group["axis_variant"].astype(str).eq(variant)]
            if part.empty:
                continue
            row = part.iloc[0]
            out = {
                "target": best["target"],
                "subset": best["subset"],
                "representation": best["representation"],
                "layer": int(best["layer"]),
                "axis_variant": variant,
                "reference_original_auc": float(original.get("auc_high_vs_low_test", np.nan)),
                "reference_original_partial_spearman": float(original.get("partial_spearman_score_target_resid_progress_test", np.nan)),
                "action_between_variance_ratio": diag["action_between_variance_ratio"] if variant == "original_action_balanced" else np.nan,
                "progress_only_auc_high_vs_low": diag["progress_only_auc_high_vs_low"] if variant == "original_action_balanced" else np.nan,
                "mean_score_by_action_json": diag["mean_score_by_action_json"] if variant == "original_action_balanced" else "{}",
            }
            for col in metric_cols:
                out[col] = float(row.get(col, np.nan))
                out[f"delta_{col}_vs_original"] = float(row.get(col, np.nan)) - float(original.get(col, np.nan))
            shuffled = group[group["axis_variant"].astype(str).eq("shuffled_commitment_control")]
            if not shuffled.empty:
                out["auc_above_shuffled"] = float(row.get("auc_high_vs_low_test", np.nan)) - float(shuffled.iloc[0].get("auc_high_vs_low_test", np.nan))
                out["partial_spearman_above_shuffled"] = float(row.get("partial_spearman_score_target_resid_progress_test", np.nan)) - float(shuffled.iloc[0].get("partial_spearman_score_target_resid_progress_test", np.nan))
            else:
                out["auc_above_shuffled"] = np.nan
                out["partial_spearman_above_shuffled"] = np.nan
            out["survives_shuffled_control"] = bool(
                np.isfinite(out["auc_above_shuffled"])
                and out["auc_above_shuffled"] > 0.03
                and np.isfinite(out["partial_spearman_above_shuffled"])
                and out["partial_spearman_above_shuffled"] > 0.02
            )
            out["progress_leakage_warning"] = bool(
                np.isfinite(out.get("progress_only_auc_high_vs_low", np.nan))
                and np.isfinite(out.get("auc_high_vs_low_test", np.nan))
                and out["progress_only_auc_high_vs_low"] >= out["auc_high_vs_low_test"] - 0.02
            )
            rows.append(out)
    return pd.DataFrame(rows)


def strict_survival(metrics: pd.DataFrame, best_axes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_best = best_axes[(best_axes["criterion"].eq("best_test_auc")) & (best_axes["subset"].eq("all_prefixes"))]
    strict_subsets = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
    for _, best in all_best.iterrows():
        original = metrics[
            metrics["target"].astype(str).eq(str(best["target"]))
            & metrics["representation"].astype(str).eq(str(best["representation"]))
            & metrics["layer"].astype(int).eq(int(best["layer"]))
            & metrics["axis_variant"].astype(str).eq("original_action_balanced")
        ]
        all_row = original[original["subset"].astype(str).eq("all_prefixes")]
        if all_row.empty:
            continue
        all_row = all_row.iloc[0]
        for subset in strict_subsets:
            strict = original[original["subset"].astype(str).eq(subset)]
            if strict.empty:
                continue
            strict = strict.iloc[0]
            rows.append(
                {
                    "target": best["target"],
                    "representation": best["representation"],
                    "layer": int(best["layer"]),
                    "strict_subset": subset,
                    "all_auc": float(all_row.get("auc_high_vs_low_test", np.nan)),
                    "strict_auc": float(strict.get("auc_high_vs_low_test", np.nan)),
                    "all_partial_spearman": float(all_row.get("partial_spearman_score_target_resid_progress_test", np.nan)),
                    "strict_partial_spearman": float(strict.get("partial_spearman_score_target_resid_progress_test", np.nan)),
                    "same_sign_partial": bool(
                        np.sign(float(all_row.get("partial_spearman_score_target_resid_progress_test", np.nan)))
                        == np.sign(float(strict.get("partial_spearman_score_target_resid_progress_test", np.nan)))
                    ),
                    "strict_nontrivial": bool(
                        float(strict.get("auc_high_vs_low_test", np.nan)) > 0.55
                        or abs(float(strict.get("partial_spearman_score_target_resid_progress_test", np.nan))) > 0.05
                    ),
                }
            )
    return pd.DataFrame(rows)


def failure_modes(summary: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        reasons = []
        if row.get("axis_variant") == "original_action_balanced":
            auc = float(row.get("auc_high_vs_low_test", np.nan))
            partial = float(row.get("partial_spearman_score_target_resid_progress_test", np.nan))
            if not np.isfinite(auc) or auc < 0.55:
                reasons.append("held-out_auc_weak")
            if not np.isfinite(partial) or abs(partial) < 0.05:
                reasons.append("partial_spearman_weak")
            if row.get("progress_leakage_warning", False):
                reasons.append("progress_only_baseline_competitive")
        if row.get("axis_variant") == "shuffled_commitment_control":
            auc = float(row.get("auc_high_vs_low_test", np.nan))
            if np.isfinite(auc) and auc > 0.55:
                reasons.append("shuffled_control_high")
        if reasons:
            rows.append({**row.to_dict(), "failure_reasons": ";".join(reasons)})
    if not strict.empty:
        for _, row in strict.iterrows():
            if not row.get("strict_nontrivial", False):
                rows.append({**row.to_dict(), "failure_reasons": "strict_pre_action_weak"})
    return pd.DataFrame(rows)


def plot_control_summary(summary: pd.DataFrame, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return None
    part = summary[summary["axis_variant"].isin(["original_action_balanced", "progress_matched_action_balanced", "shuffled_commitment_control", "progress_residualized_feature_control"])].copy()
    if part.empty:
        return None
    grouped = part.groupby("axis_variant").agg(
        auc=("auc_high_vs_low_test", "mean"),
        partial=("partial_spearman_score_target_resid_progress_test", "mean"),
        bal_acc=("balanced_accuracy_high_vs_low_test", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(grouped))
    width = 0.25
    for i, col in enumerate(["auc", "partial", "bal_acc"]):
        ax.bar(x + (i - 1) * width, grouped[col].to_numpy(float), width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([idx.replace("_", "\n") for idx in grouped.index], fontsize=8)
    ax.axhline(0.5, color="#999999", linestyle="--", lw=0.8, label="AUC/bal acc chance")
    ax.set_title("Commitment contrast control summary\nNo PCA/UMAP; action labels used only for balancing/diagnostics", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    controls_dir = output_dir / "controls"
    figure_dir = resolve(FIGURE_DIR) / "metrics"
    ensure_dir(controls_dir)
    ensure_dir(figure_dir)

    metrics_path = resolve(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = ok_metrics(pd.read_csv(metrics_path))
    if metrics.empty:
        raise ValueError("No successful commitment projection metrics found.")
    metrics["layer"] = metrics["layer"].astype(int)

    best_axes = build_best_axes(metrics)
    summary = compare_controls(metrics, best_axes)
    strict = strict_survival(metrics, best_axes)
    failures = failure_modes(summary, strict)
    if not strict.empty:
        strict_tagged = strict.copy()
        strict_tagged["axis_variant"] = "strict_pre_action_survival"
        summary = pd.concat([summary, strict_tagged], ignore_index=True, sort=False)

    summary_path = controls_dir / "commitment_control_summary.csv"
    best_path = controls_dir / "best_commitment_axes.csv"
    failure_path = controls_dir / "failure_modes.csv"
    summary.to_csv(summary_path, index=False)
    best_axes.to_csv(best_path, index=False)
    failures.to_csv(failure_path, index=False)
    fig_path = plot_control_summary(summary, figure_dir / "control_summary.png")

    report_lines = [
        "# Commitment Contrast Controls Report",
        "",
        "- No PCA/UMAP is used as the primary projection.",
        "- Axes are train-only.",
        "- Action labels are used only for balancing and diagnostics.",
        "- Strict pre-action means no explicit lexical action token, not no implicit coordinate/path evidence.",
        "",
        f"- successful metric rows: {len(metrics)}",
        f"- best-axis rows: {len(best_axes)}",
        f"- control summary rows: {len(summary)}",
        f"- failure mode rows: {len(failures)}",
    ]
    if fig_path:
        report_lines.extend(["", f"![control summary]({rel(fig_path)})"])
    if not best_axes.empty:
        report_lines.extend(["", "## Best Axes", "", best_axes.head(20).to_markdown(index=False)])
    if not failures.empty:
        report_lines.extend(["", "## Failure Modes", "", failures.head(30).to_markdown(index=False)])
    save_markdown(controls_dir / "14_commitment_contrast_controls_report.md", "\n".join(report_lines))
    write_json(
        controls_dir / "controls_manifest.json",
        {
            "summary": rel(summary_path),
            "best_axes": rel(best_path),
            "failure_modes": rel(failure_path),
            "control_summary_figure": rel(fig_path) if fig_path else None,
        },
    )
    print(json.dumps({"summary": rel(summary_path), "best_axes": rel(best_path), "failures": rel(failure_path)}, indent=2))


if __name__ == "__main__":
    main()
