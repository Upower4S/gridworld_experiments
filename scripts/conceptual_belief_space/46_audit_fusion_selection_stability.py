#!/usr/bin/env python3
"""Audit stability of label-free activation-score fusion choices.

This script does not retrain base activation probes.  It checks whether the best
fusion found in saved OOF score space is stable under leave-one-split-out fusion
selection.  Because the base OOF score files were produced by models whose train
sets overlap other folds, this is a diagnostic stability audit, not a fully
nested no-overfit proof.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def label_indices(values: pd.Series) -> np.ndarray:
    return values.astype(str).map(ACTION_TO_IDX).to_numpy(dtype=np.int64)


def metric_dict(df: pd.DataFrame) -> dict[str, Any]:
    y = label_indices(df["top_action"])
    pred = label_indices(df["predicted_top_action"])
    out = {
        "n_rows": int(len(df)),
        "accuracy": float((pred == y).mean()) if len(df) else float("nan"),
    }
    recalls = []
    for idx, action in enumerate(ACTIONS):
        mask = y == idx
        recall = float((pred[mask] == idx).mean()) if mask.any() else float("nan")
        out[f"recall_{action}"] = recall
        if np.isfinite(recall):
            recalls.append(recall)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion_predictions", default="outputs/conceptual_belief_space/activation_linear_score_fusion_with_segments/linear_score_fusion_predictions.parquet")
    p.add_argument("--split_source", default="outputs/conceptual_belief_space/activation_segment_combo_linear_full/segment_combo_predictions.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_linear_score_fusion_with_segments")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(args.fusion_predictions)
    if "split_id" not in pred.columns:
        split = pd.read_parquet(args.split_source, columns=["row_id", "split_id"])
        pred = pred.merge(split, on="row_id", how="left")
    if pred["split_id"].isna().any():
        raise ValueError("Some fusion prediction rows are missing split_id after merge")

    fold_rows = []
    for (fusion_name, split_id), group in pred.groupby(["fusion_name", "split_id"], sort=False):
        row = {"fusion_name": fusion_name, "split_id": int(split_id)}
        row.update(metric_dict(group))
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fusion_fold_metrics.csv", index=False)

    selection_rows = []
    for heldout in sorted(pred["split_id"].dropna().unique()):
        train = pred[pred["split_id"].ne(heldout)]
        test = pred[pred["split_id"].eq(heldout)]
        train_scores = []
        for fusion_name, group in train.groupby("fusion_name", sort=False):
            m = metric_dict(group)
            train_scores.append((m["balanced_accuracy"], m["accuracy"], fusion_name))
        train_scores.sort(reverse=True)
        best_name = train_scores[0][2]
        test_metrics = metric_dict(test[test["fusion_name"].eq(best_name)])
        selection_rows.append({
            "heldout_split": int(heldout),
            "selected_fusion": best_name,
            "train4_balanced_accuracy": float(train_scores[0][0]),
            "train4_accuracy": float(train_scores[0][1]),
            **{f"heldout_{k}": v for k, v in test_metrics.items()},
        })
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output_dir / "fusion_selection_stability.csv", index=False)

    all_rows = []
    for fusion_name, group in pred.groupby("fusion_name", sort=False):
        row = {"fusion_name": fusion_name}
        row.update(metric_dict(group))
        per_fold = fold_metrics[fold_metrics["fusion_name"].eq(fusion_name)]
        row["mean_fold_balanced_accuracy"] = float(per_fold["balanced_accuracy"].mean())
        row["min_fold_balanced_accuracy"] = float(per_fold["balanced_accuracy"].min())
        all_rows.append(row)
    global_metrics = pd.DataFrame(all_rows).sort_values("balanced_accuracy", ascending=False)
    global_metrics.to_csv(output_dir / "fusion_global_metrics.csv", index=False)

    best = global_metrics.iloc[0]
    selected_counts = selection["selected_fusion"].value_counts().to_dict()
    report = [
        "# Fusion Selection Stability Audit",
        "",
        "## Scope",
        "",
        "- Input: saved OOF action-score fusion predictions.",
        "- This audit does not retrain base activation probes.",
        "- It is a stability diagnostic, not a fully nested no-overfit proof.",
        "- The fully nested version would retrain every component source inside each held-out split before source selection.",
        "",
        "## Best Global Fusion",
        "",
        f"- fusion: `{best['fusion_name']}`",
        f"- global balanced accuracy: `{best['balanced_accuracy']:.6f}`",
        f"- global accuracy: `{best['accuracy']:.6f}`",
        f"- mean fold balanced accuracy: `{best['mean_fold_balanced_accuracy']:.6f}`",
        f"- minimum fold balanced accuracy: `{best['min_fold_balanced_accuracy']:.6f}`",
        "",
        "## Leave-One-Split-Out Fusion Selection",
        "",
        selection.to_markdown(index=False),
        "",
        f"Selected fusion counts: `{selected_counts}`",
        "",
        "## Interpretation",
        "",
        "If the same fusion is selected for every held-out split, the exploratory choice is stable in saved score space. "
        "However, because the component OOF models for the training folds may have trained on the held-out fold's labels, "
        "this audit cannot by itself prove no-overfit performance.",
    ]
    (output_dir / "fusion_selection_stability_report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {output_dir / 'fusion_selection_stability.csv'}")
    print(f"Wrote {output_dir / 'fusion_fold_metrics.csv'}")
    print(f"Wrote {output_dir / 'fusion_selection_stability_report.md'}")
    print(selection.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
