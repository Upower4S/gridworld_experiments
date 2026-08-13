#!/usr/bin/env python3
"""Fit trajectory-grouped linear probes to get activation-dimension importance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from probe_guided_embedding_lib import (
    ACTIONS,
    Q_COLS,
    action_indices,
    compute_q_labels,
    ensure_dir,
    extract_probe_importance,
    fit_linear_action_probe,
    load_activation_view,
    load_metadata,
    make_grouped_folds,
    parse_csv_ints,
    parse_csv_strings,
    save_markdown,
    select_topk_features,
    standardize_from_train,
    write_json,
)


DEFAULT_TOPKS = [64, 128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project_root", default=".")
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--behavior", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--layers", default="8,10,12,14,16,18,20,22")
    p.add_argument("--representations", default="raw,delta_pre,unit_delta_pre")
    p.add_argument("--topk", default="64,128,256,512,1024")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def rel(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_or_make_splits(path: Path, metadata: pd.DataFrame, group_col: str, n_splits: int, seed: int) -> list[dict]:
    if path.exists():
        raw = json.loads(path.read_text())
        return [
            {
                "split_id": int(rec.get("split_id", rec.get("fold", i))),
                "train_idx": np.asarray(rec.get("train_indices", rec.get("train_idx")), dtype=np.int64),
                "test_idx": np.asarray(rec.get("test_indices", rec.get("test_idx")), dtype=np.int64),
            }
            for i, rec in enumerate(raw)
        ]
    return make_grouped_folds(metadata, group_col=group_col, n_splits=n_splits, seed=seed)


def fit_progress_importance(xz: np.ndarray, progress: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr

    model = Ridge(alpha=10.0)
    model.fit(xz[train_idx], progress[train_idx])
    pred = model.predict(xz[train_idx])
    coef = np.abs(np.asarray(model.coef_, dtype=np.float32))
    try:
        corr = float(pearsonr(progress[train_idx], pred).statistic)
    except Exception:
        corr = float("nan")
    return coef, corr


def jaccard(a: set[int], b: set[int]) -> float:
    return float(len(a & b) / len(a | b)) if a or b else float("nan")


def write_feature_set(path: Path, view_name: str, feature_set: str, split_to_features: dict[int, list[int]], seed: int) -> None:
    write_json(
        path,
        {
            "view_name": view_name,
            "feature_set": feature_set,
            "input_feature_kind": "activation_dimensions_only",
            "forbidden_features_used": False,
            "seed": seed,
            "splits": {str(k): v for k, v in split_to_features.items()},
        },
    )


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = ensure_dir(rel(root, args.output_dir))
    imp_dir = ensure_dir(out / "feature_importance")
    set_dir = ensure_dir(out / "feature_sets")
    metadata = load_metadata(rel(root, args.metadata), rel(root, args.behavior))
    if "q_top_action" not in metadata.columns:
        metadata["q_top_action"] = compute_q_labels(metadata)
    y = action_indices(metadata["q_top_action"])
    progress = pd.to_numeric(metadata["progress"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    splits = load_or_make_splits(rel(root, args.splits), metadata, args.group_col, args.n_splits, args.seed)
    layers = parse_csv_ints(args.layers)
    reps = parse_csv_strings(args.representations)
    topks = parse_csv_ints(args.topk)
    metrics_rows: list[dict] = []
    stability_rows: list[dict] = []

    for layer in layers:
        for rep in reps:
            view_name = f"layer_{layer:02d}_{rep}"
            print(f"Fitting probes for {view_name}", flush=True)
            x, info = load_activation_view(metadata, rel(root, args.features_dir), layer, rep)
            rows = []
            split_feature_sets: dict[str, dict[int, list[int]]] = {}
            max_topk = min(max(topks + [512]), x.shape[1])
            for split in splits:
                sid = int(split["split_id"])
                train = np.asarray(split["train_idx"], dtype=np.int64)
                test = np.asarray(split["test_idx"], dtype=np.int64)
                xz, _mean, _scale = standardize_from_train(x, train)
                model, _importance, probe_metrics = fit_linear_action_probe(xz, y, train, test, alpha=args.alpha)
                imp_coef_std = extract_probe_importance(model, "coef_std")
                imp_pairwise = extract_probe_importance(model, "pairwise_contrast")
                imp_l2 = extract_probe_importance(model, "l2")

                shuffled_y = y.copy()
                shuffled_train = y[train].copy()
                np.random.default_rng(args.seed + 1000 + sid + layer).shuffle(shuffled_train)
                shuffled_y[train] = shuffled_train
                shuffled_model, _shuf_imp, shuffled_metrics = fit_linear_action_probe(xz, shuffled_y, train, test, alpha=args.alpha)
                shuffled_l2 = extract_probe_importance(shuffled_model, "l2")
                progress_l2, progress_corr = fit_progress_importance(xz, progress, train)

                rows.append(pd.DataFrame({
                    "view_name": view_name,
                    "split_id": sid,
                    "layer": layer,
                    "representation": rep,
                    "dimension": np.arange(x.shape[1], dtype=np.int64),
                    "coef_std": imp_coef_std,
                    "pairwise_contrast": imp_pairwise,
                    "l2": imp_l2,
                    "shuffled_l2": shuffled_l2,
                    "progress_l2": progress_l2,
                }))
                for stage, metrics in [("action_probe", probe_metrics), ("shuffled_label_probe", shuffled_metrics)]:
                    metrics_rows.append({
                        "stage": stage,
                        "view_name": view_name,
                        "split_id": sid,
                        "layer": layer,
                        "representation": rep,
                        "target": "q_top_action" if stage == "action_probe" else "shuffled_q_top_action",
                        "n_train_rows": len(train),
                        "n_test_rows": len(test),
                        **metrics,
                    })
                metrics_rows.append({
                    "stage": "progress_probe",
                    "view_name": view_name,
                    "split_id": sid,
                    "layer": layer,
                    "representation": rep,
                    "target": "progress",
                    "n_train_rows": len(train),
                    "n_test_rows": len(test),
                    "train_progress_pearson": progress_corr,
                })

                for k in topks:
                    split_feature_sets.setdefault(f"probe_top_{k}", {})[sid] = select_topk_features(imp_l2, k).astype(int).tolist()
                split_feature_sets.setdefault("random_top_512", {})[sid] = np.sort(np.random.default_rng(args.seed + sid + layer).choice(np.arange(x.shape[1]), size=min(512, x.shape[1]), replace=False)).astype(int).tolist()
                split_feature_sets.setdefault("shuffled_probe_top_512", {})[sid] = select_topk_features(shuffled_l2, 512).astype(int).tolist()
                split_feature_sets.setdefault("progress_probe_top_512", {})[sid] = select_topk_features(progress_l2, 512).astype(int).tolist()
                print(f"  split={sid} test_bal={probe_metrics['probe_test_balanced_accuracy']:.4f}", flush=True)
                del xz

            imp = pd.concat(rows, ignore_index=True)
            imp.to_parquet(imp_dir / f"{view_name}_feature_importance.parquet", index=False)
            for name, split_map in split_feature_sets.items():
                write_feature_set(set_dir / f"{view_name}_{name}.json", view_name, name, split_map, args.seed)
            if "probe_top_512" in split_feature_sets:
                split_sets = [set(v) for v in split_feature_sets["probe_top_512"].values()]
                vals = [jaccard(split_sets[i], split_sets[j]) for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))]
                stability_rows.append({"view_name": view_name, "layer": layer, "representation": rep, "feature_set": "probe_top_512", "mean_pairwise_jaccard": float(np.nanmean(vals)) if vals else float("nan")})
            del x

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "probe_metrics.csv", index=False)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(imp_dir / "feature_importance_stability.csv", index=False)
    action = metrics[metrics["stage"].eq("action_probe")]
    summary = action.groupby(["view_name", "layer", "representation"], as_index=False).agg(
        mean_probe_test_balanced_accuracy=("probe_test_balanced_accuracy", "mean"),
        mean_probe_test_accuracy=("probe_test_accuracy", "mean"),
        splits=("split_id", "nunique"),
    ).sort_values("mean_probe_test_balanced_accuracy", ascending=False)
    summary.to_csv(out / "probe_view_summary.csv", index=False)
    if not summary.empty:
        best = summary.iloc[0].to_dict()
        best["selection_rule"] = "highest mean trajectory-grouped held-out q_top_action balanced accuracy"
        best["topks_available"] = topks
        write_json(out / "selected_probe_guidance_config.json", best)
    save_markdown(out / "probe_feature_importance_report.md", "# Probe Feature Importance Report\n\n" + summary.to_markdown(index=False) + "\n")
    print(f"Wrote {out / 'probe_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
