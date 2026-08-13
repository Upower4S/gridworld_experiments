#!/usr/bin/env python3
"""Audit inputs for the strict probe-guided activation embedding pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from probe_guided_activation_embedding_lib import (
    Q_COLS, available_layers_representations, discover_feature_arrays, ensure_dir,
    load_activation_metadata, load_splits, read_json, save_markdown, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--behavior", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--stacker_dir", default="outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3_with_predictions")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--group_col", default="trajectory_id")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    metadata = load_activation_metadata(args.metadata)
    feature_infos = discover_feature_arrays(args.features_dir)
    if not feature_infos:
        raise FileNotFoundError(f"No activation feature arrays found under {args.features_dir}")
    layers, reps = available_layers_representations(feature_infos)
    splits = load_splits(args.splits, len(metadata), args.group_col, metadata)
    row_count_mismatches = [info for info in feature_infos if int(info["shape"][0]) <= int(metadata["row_index"].max())]
    if row_count_mismatches:
        raise ValueError(f"Some feature arrays do not cover metadata row_index values: {row_count_mismatches[:3]}")
    behavior_info = None
    if Path(args.behavior).exists():
        b = pd.read_parquet(args.behavior)
        behavior_info = {"path": args.behavior, "shape": list(b.shape), "columns": list(b.columns), "n_trajectories": int(b["trajectory_id"].nunique()) if "trajectory_id" in b.columns else None}
    stacker_info = {"path": args.stacker_dir, "exists": Path(args.stacker_dir).exists()}
    pred = Path(args.stacker_dir) / "nested_oof_source_stacker_predictions.parquet"
    if pred.exists():
        pdf = pd.read_parquet(pred, columns=["row_id", "trajectory_id", "decision_id", "split_id", "in_sample", "oof_score"])
        stacker_info.update({"prediction_path": str(pred), "prediction_shape": [int(len(pdf)), int(len(pdf.columns))], "n_in_sample": int(pdf["in_sample"].fillna(False).sum()) if "in_sample" in pdf.columns else None})
    q = metadata[Q_COLS].to_numpy(float)
    manifest = {
        "metadata": {"path": args.metadata, "shape": list(metadata.shape), "columns": list(metadata.columns), "n_trajectories": int(metadata[args.group_col].nunique()), "n_decisions": int(metadata["decision_id"].nunique())},
        "behavior": behavior_info,
        "features": {"features_dir": args.features_dir, "n_arrays": len(feature_infos), "available_layers": layers, "available_representations": reps, "arrays": feature_infos},
        "splits": [{"split_id": int(s["split_id"]), "n_train_rows": int(len(s["train_idx"])), "n_test_rows": int(len(s["test_idx"])), "n_train_groups": int(s["n_train_groups"]), "n_test_groups": int(s["n_test_groups"])} for s in splits],
        "stacker_context_not_coordinate_input": stacker_info,
        "strict_columns_found": [c for c in ["strict_before_any_action_mention", "strict_before_final_action_mention", "is_before_first_action_mention", "is_before_first_final_action_mention"] if c in metadata.columns],
        "q_sum_min": float(q.sum(axis=1).min()),
        "q_sum_max": float(q.sum(axis=1).max()),
        "guardrail": "PCA/UMAP inputs must be activation arrays only. Stacker/source/probe scores and behavior q/progress/action metadata are not input features.",
    }
    write_json(output_dir / "io_manifest.json", manifest)
    report = [
        "# Probe-Guided Activation Embedding Input Audit", "",
        f"- activation metadata: `{args.metadata}` shape `{metadata.shape}`",
        f"- activation feature arrays: `{len(feature_infos)}` arrays under `{args.features_dir}`",
        f"- layers: `{layers}`",
        f"- representations: `{reps}`",
        f"- splits: `{args.splits}` with `{len(splits)}` trajectory-grouped folds",
        f"- strict columns found: `{manifest['strict_columns_found']}`",
        f"- q sum min/max: `{manifest['q_sum_min']:.6f}` / `{manifest['q_sum_max']:.6f}`",
        "", "The score-stacker outputs are recorded only as context. They are not valid PCA/UMAP coordinate inputs for this pipeline.",
    ]
    save_markdown(output_dir / "input_audit.md", "\n".join(report))
    print(f"Wrote {output_dir / 'io_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
