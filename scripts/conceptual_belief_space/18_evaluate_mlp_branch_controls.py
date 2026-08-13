#!/usr/bin/env python
"""Evaluate controls for supervised MLP bottleneck branch probes."""

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

from src.conceptual_belief_space.io import ensure_dir, save_markdown
from src.conceptual_belief_space.mlp_branch_utils import ACTIONS, classification_summary, nearest_vertex, safe_spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/controls")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def match_baseline(metrics: pd.DataFrame, mlp_row: pd.Series, model_type: str) -> pd.Series | None:
    frame = metrics[metrics["model_type"].astype(str).eq(model_type)].copy()
    for key in ["split_id", "layer", "representation", "subset"]:
        frame = frame[frame[key].astype(str).eq(str(mlp_row[key]))]
    if frame.empty:
        return None
    return frame.iloc[0]


def partial_spearman(df: pd.DataFrame, x_col: str, y_col: str, covariates: list[str]) -> float:
    cols = [x_col, y_col] + [c for c in covariates if c in df.columns]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5 or len(cols) <= 2:
        return safe_spearman(df[x_col], df[y_col])
    design = np.column_stack([np.ones(len(data)), data[cols[2:]].to_numpy(dtype=float)])
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    bx, *_ = np.linalg.lstsq(design, x, rcond=None)
    by, *_ = np.linalg.lstsq(design, y, rcond=None)
    return safe_spearman(x - design @ bx, y - design @ by)


def build_summary(metrics: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = metrics[metrics["status"].astype(str).eq("ok")].copy() if "status" in metrics.columns else metrics.copy()
    mlp = ok[ok["model_type"].astype(str).eq("mlp_branch")].copy()
    rows = []
    failures = []
    for _, row in mlp.iterrows():
        item = {
            "run_id": row["run_id"],
            "split_id": row["split_id"],
            "layer": row["layer"],
            "representation": row["representation"],
            "subset": row["subset"],
            "mlp_top_action_bal_acc": row.get("test_q_top_action_balanced_accuracy", np.nan),
            "mlp_commitment_spearman": row.get("test_commitment_spearman", np.nan),
            "mlp_q_kl": row.get("test_q_kl", np.nan),
            "mlp_vertex_top_action_bal_acc": row.get("test_vertex_top_action_balanced_accuracy", np.nan),
            "mlp_radius_commitment_spearman": row.get("test_radius_commitment_spearman", np.nan),
            "mlp_radius_progress_spearman": row.get("test_radius_progress_spearman", np.nan),
        }
        for baseline in ["linear_baseline", "mlp_shuffled_target", "progress_only", "label_permutation"]:
            base = match_baseline(ok, row, baseline)
            if base is None:
                continue
            item[f"{baseline}_top_action_bal_acc"] = base.get("test_q_top_action_balanced_accuracy", np.nan)
            item[f"{baseline}_commitment_spearman"] = base.get("test_commitment_spearman", np.nan)
            item[f"delta_top_action_bal_acc_vs_{baseline}"] = float(row.get("test_q_top_action_balanced_accuracy", np.nan)) - float(base.get("test_q_top_action_balanced_accuracy", np.nan))
            item[f"delta_commitment_spearman_vs_{baseline}"] = float(row.get("test_commitment_spearman", np.nan)) - float(base.get("test_commitment_spearman", np.nan))
        run_preds = preds[preds["run_id"].astype(str).eq(str(row["run_id"])) & preds["split_role"].astype(str).eq("test")].copy()
        if not run_preds.empty:
            item["radius_commitment_partial_spearman_progress"] = partial_spearman(
                run_preds,
                "z_radius",
                "entropy_commitment",
                ["progress", "prefix_token_index", "num_cot_tokens"],
            )
            nearest, _, _ = nearest_vertex(run_preds[["z1_plot", "z2_plot"]].to_numpy(dtype=np.float32))
            item["nearest_vertex_top_confusion_json"] = classification_summary(run_preds["top_action"], nearest)["confusion_matrix_json"]
            item["nearest_vertex_agent_confusion_json"] = classification_summary(run_preds["agent_action"], nearest)["confusion_matrix_json"]
        flags = []
        if item.get("delta_top_action_bal_acc_vs_mlp_shuffled_target", 1.0) <= 0.05:
            flags.append("failed_shuffled_control")
        if item.get("delta_top_action_bal_acc_vs_progress_only", 1.0) <= 0.03:
            flags.append("possibly_progress_driven")
        if str(row["subset"]).startswith("strict") and item.get("delta_top_action_bal_acc_vs_mlp_shuffled_target", 0.0) > 0.05:
            flags.append("strict_signal_present")
        if float(item.get("mlp_q_kl", np.nan)) > 1.0 and float(item.get("mlp_top_action_bal_acc", 0.0)) > 0.5:
            flags.append("classification_without_full_belief_reconstruction")
        if float(item.get("mlp_commitment_spearman", 0.0)) > 0.3:
            flags.append("commitment_recoverable")
        item["control_flags"] = ";".join(flags)
        rows.append(item)
        for flag in flags:
            failures.append(
                {
                    "run_id": row["run_id"],
                    "split_id": row["split_id"],
                    "layer": row["layer"],
                    "representation": row["representation"],
                    "subset": row["subset"],
                    "flag": flag,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def plot_delta(summary: pd.DataFrame, col: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if summary.empty or col not in summary.columns:
        return
    data = summary.copy()
    data["layer"] = pd.to_numeric(data["layer"], errors="coerce")
    grouped = data.groupby(["representation", "layer"], observed=False)[col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for rep, part in grouped.groupby("representation"):
        ax.plot(part["layer"], part[col], marker="o", label=str(rep))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.legend(fontsize=8)
    savefig(path)


def strict_plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    data = summary.copy()
    data["is_strict"] = data["subset"].astype(str).str.contains("strict")
    grouped = data.groupby("is_strict", observed=False)[["mlp_top_action_bal_acc", "mlp_commitment_spearman"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Supervised MLP bottleneck strict pre-action control")
    ax.set_xlabel("Strict pre-action subset")
    savefig(path)


def radius_plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(summary["mlp_radius_progress_spearman"], summary.get("radius_commitment_partial_spearman_progress", np.nan), s=28, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman(z_radius, progress)")
    ax.set_ylabel("Partial Spearman(z_radius, commitment | progress)")
    ax.set_title("Supervised MLP bottleneck radius control")
    savefig(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    controls_dir = ensure_dir(output_dir / "controls")
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = Path(args.metrics) if args.metrics else output_dir / "mlp_branch_metrics.csv"
    predictions_path = Path(args.predictions) if args.predictions else output_dir / "predictions/mlp_branch_predictions.parquet"
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_parquet(predictions_path)
    summary, failures = build_summary(metrics, preds)
    summary_path = controls_dir / "mlp_branch_control_summary.csv"
    failures_path = controls_dir / "mlp_branch_failure_modes.csv"
    summary.to_csv(summary_path, index=False)
    failures.to_csv(failures_path, index=False)

    plot_delta(summary, "delta_top_action_bal_acc_vs_linear_baseline", figure_dir / "mlp_vs_linear_delta.png", "Supervised MLP bottleneck vs linear baseline")
    plot_delta(summary, "delta_top_action_bal_acc_vs_mlp_shuffled_target", figure_dir / "mlp_vs_shuffled.png", "Supervised MLP bottleneck vs shuffled target")
    plot_delta(summary, "delta_top_action_bal_acc_vs_progress_only", figure_dir / "mlp_vs_progress_only.png", "Supervised MLP bottleneck vs progress-only baseline")
    strict_plot(summary, figure_dir / "strict_pre_action_control.png")
    radius_plot(summary, figure_dir / "radius_commitment_vs_progress_partial.png")

    report = [
        "# MLP Branch Control Report",
        "",
        "These controls evaluate a supervised MLP bottleneck embedding. They do not validate an unsupervised raw activation manifold.",
        "",
        f"- summary rows: {len(summary)}",
        f"- flagged rows: {len(failures)}",
        f"- summary: `{summary_path}`",
        f"- failure modes: `{failures_path}`",
    ]
    if not failures.empty:
        report.extend(["", "## Failure / Diagnostic Flags", ""])
        counts = failures["flag"].value_counts()
        report.extend(f"- {flag}: {count}" for flag, count in counts.items())
    save_markdown(output_dir / "reports/18_mlp_branch_controls_report.md", "\n".join(report))
    print(json.dumps({"summary": str(summary_path), "failure_modes": str(failures_path), "rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()

