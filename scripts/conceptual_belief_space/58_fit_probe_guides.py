#!/usr/bin/env python3
"""Fit train-only linear probe guides for activation dimension selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from probe_guided_activation_embedding_lib import (
    Q_COLS, action_indices, ensure_dir, fit_action_probe, fit_progress_importance,
    fit_standardizer, load_activation_matrix, load_activation_metadata, load_splits,
    parse_csv_ints, parse_csv_strings, save_markdown, standardize, topk, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--layers", default="14,18,22")
    p.add_argument("--representations", default="delta_pre,raw,unit_delta_pre")
    p.add_argument("--top_k", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    imp_dir = ensure_dir(out / "feature_importance")
    set_dir = ensure_dir(out / "feature_sets")
    metadata = load_activation_metadata(args.metadata)
    splits = load_splits(args.splits, len(metadata), args.group_col, metadata)
    y = action_indices(metadata["true_q_top_action"])
    progress = pd.to_numeric(metadata["progress"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    layers = parse_csv_ints(args.layers)
    reps = parse_csv_strings(args.representations)
    rng = np.random.default_rng(args.seed)
    metrics_rows = []

    for layer in layers:
        for rep in reps:
            print(f"Loading layer={layer} representation={rep}", flush=True)
            x, _info = load_activation_matrix(metadata, args.features_dir, layer, rep)
            for split in splits:
                sid = int(split["split_id"])
                train = np.asarray(split["train_idx"], dtype=np.int64)
                test = np.asarray(split["test_idx"], dtype=np.int64)
                mean, scale = fit_standardizer(x, train)
                xz = standardize(x, mean, scale)
                action_imp, action_metrics = fit_action_probe(xz, y, train, test, alpha=args.alpha)
                shuffled_y = y.copy()
                shuffled_train = y[train].copy()
                np.random.default_rng(args.seed + 1000 + sid + layer).shuffle(shuffled_train)
                shuffled_y[train] = shuffled_train
                shuffled_imp, shuffled_metrics = fit_action_probe(xz, shuffled_y, train, test, alpha=args.alpha)
                progress_imp, progress_metrics = fit_progress_importance(xz, progress, train)
                imp = pd.DataFrame({
                    "dimension": np.arange(xz.shape[1], dtype=np.int64),
                    "action_probe_importance": action_imp,
                    "shuffled_probe_importance": shuffled_imp,
                    "progress_probe_importance": progress_imp,
                }).sort_values("action_probe_importance", ascending=False)
                imp_path = imp_dir / f"split_{sid}_layer_{layer:02d}_{rep}_importance.csv"
                imp.to_csv(imp_path, index=False)
                for stage, m in [("action_probe", action_metrics), ("shuffled_label_probe", shuffled_metrics), ("progress_probe", progress_metrics)]:
                    row = {"stage": stage, "split_id": sid, "layer": layer, "representation": rep, "target": "true_q_top_action" if stage == "action_probe" else stage, "n_train_rows": len(train), "n_test_rows": len(test), **m}
                    metrics_rows.append(row)
                feature_sets = {
                    "split_id": sid,
                    "layer": layer,
                    "representation": rep,
                    "top_k": args.top_k,
                    "input_feature_kind": "activation_dimensions_only",
                    "forbidden_features_used": False,
                    "probe_top_K": topk(action_imp, args.top_k).astype(int).tolist(),
                    "probe_weighted_top_K": topk(action_imp, args.top_k).astype(int).tolist(),
                    "random_top_K": np.sort(rng.choice(np.arange(xz.shape[1]), size=min(args.top_k, xz.shape[1]), replace=False)).astype(int).tolist(),
                    "shuffled_probe_top_K": topk(shuffled_imp, args.top_k).astype(int).tolist(),
                    "progress_probe_top_K": topk(progress_imp, args.top_k).astype(int).tolist(),
                }
                write_json(set_dir / f"split_{sid}_layer_{layer:02d}_{rep}_feature_sets.json", feature_sets)
                print(f"  split={sid} action_probe_bal={action_metrics['test_balanced_accuracy']:.4f}", flush=True)
                del xz
            del x
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "probe_metrics.csv", index=False)
    action = metrics[metrics["stage"].eq("action_probe")].copy()
    summary = action.groupby(["layer", "representation"], as_index=False).agg(mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"), mean_test_accuracy=("test_accuracy", "mean"), splits=("split_id", "nunique"))
    summary = summary.sort_values("mean_test_balanced_accuracy", ascending=False)
    summary.to_csv(out / "probe_layer_representation_summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    best["top_k"] = int(args.top_k)
    best["selection_rule"] = "highest mean trajectory-grouped held-out action-probe balanced accuracy"
    write_json(out / "selected_probe_guidance_config.json", best)
    save_markdown(out / "probe_guidance_report.md", "# Probe Guidance Report\n\n" + summary.to_markdown(index=False) + "\n")
    print(summary.head().to_string(index=False))
    print(f"Wrote {out / 'probe_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
