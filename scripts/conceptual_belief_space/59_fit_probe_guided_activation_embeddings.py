#!/usr/bin/env python3
"""Fit PCA/UMAP embeddings from activation vectors using train-only probe guidance and controls."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from probe_guided_activation_embedding_lib import (
    ACTIONS, ALL_FAMILIES, PROBE_FAMILIES, REQUIRED_CONTROLS, embedding_quality, ensure_dir,
    fit_pca, fit_standardizer, fit_umap, load_activation_matrix, load_activation_metadata, load_splits,
    make_feature_view, metadata_for_output, read_json, save_markdown, standardize, write_json, action_indices,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--probe_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--embedding_methods", default="pca,umap")
    p.add_argument("--umap_split_ids", default="0")
    p.add_argument("--max_umap_train_rows", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group_col", default="trajectory_id")
    return p.parse_args()


def parse_methods(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_split_ids(text: str) -> set[int]:
    if text.strip().lower() == "all":
        return set(range(999))
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def load_importances(probe_dir: Path, sid: int, layer: int, rep: str) -> pd.DataFrame:
    path = probe_dir / "feature_importance" / f"split_{sid}_layer_{layer:02d}_{rep}_importance.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    emb_dir = ensure_dir(out / "embeddings")
    metadata = load_activation_metadata(args.metadata)
    splits = load_splits(args.splits, len(metadata), args.group_col, metadata)
    config = read_json(Path(args.probe_dir) / "selected_probe_guidance_config.json")
    layer = int(config["layer"])
    rep = str(config["representation"])
    top_k = int(config.get("top_k", 256))
    y = action_indices(metadata["true_q_top_action"])
    strict_mask = metadata.get("strict_before_any_action_mention", pd.Series(False, index=metadata.index)).fillna(False).to_numpy(dtype=bool)
    high_mask = (metadata.get("commitment_to_final", pd.Series(np.nan, index=metadata.index)).to_numpy(dtype=float) >= 0.75)
    x, info = load_activation_matrix(metadata, args.features_dir, layer, rep)
    methods = parse_methods(args.embedding_methods)
    umap_split_ids = parse_split_ids(args.umap_split_ids)
    frames = []
    metric_rows = []
    feature_index_rows = []

    for split in splits:
        sid = int(split["split_id"])
        train = np.asarray(split["train_idx"], dtype=np.int64)
        test = np.asarray(split["test_idx"], dtype=np.int64)
        mean, scale = fit_standardizer(x, train)
        xz = standardize(x, mean, scale)
        imp = load_importances(Path(args.probe_dir), sid, layer, rep)
        action_imp = imp.sort_values("dimension")["action_probe_importance"].to_numpy(dtype=np.float32)
        shuffled_imp = imp.sort_values("dimension")["shuffled_probe_importance"].to_numpy(dtype=np.float32)
        progress_imp = imp.sort_values("dimension")["progress_probe_importance"].to_numpy(dtype=np.float32)
        sets = read_json(Path(args.probe_dir) / "feature_sets" / f"split_{sid}_layer_{layer:02d}_{rep}_feature_sets.json")
        selected_map = {family: (None if family == "all_dims" else np.asarray(sets[family], dtype=np.int64)) for family in ALL_FAMILIES}
        importance_map = {"probe_top_K": action_imp, "probe_weighted_top_K": action_imp, "shuffled_probe_top_K": shuffled_imp, "progress_probe_top_K": progress_imp, "random_top_K": None, "all_dims": None}
        for family in ALL_FAMILIES:
            x_view, view_info = make_feature_view(xz, family, selected_map[family], importance_map[family])
            feature_index_rows.append({"split_id": sid, "feature_set_family": family, "layer": layer, "representation": rep, "top_k": top_k, "n_input_dimensions": int(x_view.shape[1]), "input_feature_kind": "activation_vectors_only", "forbidden_features_used": False})
            for method in methods:
                if method == "umap" and sid not in umap_split_ids:
                    continue
                print(f"Embedding split={sid} family={family} method={method} dims={x_view.shape[1]}", flush=True)
                if method == "pca":
                    coords, einfo = fit_pca(x_view, train, args.seed + sid)
                elif method == "umap":
                    coords, einfo = fit_umap(x_view, train, args.seed + sid, max_train_rows=args.max_umap_train_rows)
                else:
                    raise ValueError(f"Unknown embedding method: {method}")
                q = embedding_quality(coords, y, train, test, args.seed + sid, strict_mask=strict_mask, high_mask=high_mask)
                metric_rows.append({"split_id": sid, "embedding_method": method, "feature_set_family": family, "layer": layer, "representation": rep, "top_k": top_k, "n_train_rows": len(train), "n_test_rows": len(test), "n_input_dimensions": int(x_view.shape[1]), **einfo, **q})
                meta = metadata_for_output(metadata, test)
                meta["split_id"] = sid
                meta["split_role"] = "test"
                meta["embedding_method"] = method
                meta["feature_set_family"] = family
                meta["layer"] = layer
                meta["representation"] = rep
                meta["top_k"] = top_k
                meta["embedding_x"] = coords[test, 0]
                meta["embedding_y"] = coords[test, 1]
                meta["coordinate_source"] = f"{method.upper()} fit on train activation vectors and transformed held-out activation vectors"
                meta["pca_umap_input_feature_kind"] = "activation_vectors_only"
                meta["forbidden_coordinate_source_used"] = False
                meta["coordinate_frame_id"] = f"split_{sid}_{method}_{family}"
                frames.append(meta)
            del x_view
        del xz
    embeddings = pd.concat(frames, ignore_index=True)
    quality = pd.DataFrame(metric_rows)
    feature_index = pd.DataFrame(feature_index_rows)
    embeddings.to_parquet(emb_dir / "probe_guided_activation_embeddings.parquet", index=False)
    quality.to_csv(out / "embedding_quality_metrics.csv", index=False)
    feature_index.to_csv(out / "feature_sets" / "feature_sets_index.csv", index=False)

    ok = quality.copy()
    pca = ok[ok["embedding_method"].eq("pca")]
    metric_cols = ["test_silhouette_true_q_top", "test_centroid_separation_ratio", "test_knn_balanced_accuracy", "test_knn_accuracy", "strict_pre_action_test_silhouette_true_q_top", "high_commitment_test_silhouette_true_q_top"]
    control = pca.groupby("feature_set_family", as_index=False)[metric_cols].mean()
    control["is_required_control"] = control["feature_set_family"].isin(REQUIRED_CONTROLS)
    best_probe = control[control["feature_set_family"].isin(PROBE_FAMILIES)].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    best_control = control[control["feature_set_family"].isin(REQUIRED_CONTROLS)].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    best_probe_metric = float(best_probe["test_knn_balanced_accuracy"].iloc[0]) if not best_probe.empty else float("nan")
    best_control_metric = float(best_control["test_knn_balanced_accuracy"].iloc[0]) if not best_control.empty else float("nan")
    control["best_probe_family"] = str(best_probe["feature_set_family"].iloc[0]) if not best_probe.empty else None
    control["best_required_control_family"] = str(best_control["feature_set_family"].iloc[0]) if not best_control.empty else None
    control["best_probe_test_knn_balanced_accuracy"] = best_probe_metric
    control["best_required_control_test_knn_balanced_accuracy"] = best_control_metric
    control["probe_improves_over_best_required_control"] = bool(np.isfinite(best_probe_metric) and np.isfinite(best_control_metric) and best_probe_metric > best_control_metric)
    control.to_csv(out / "embedding_control_comparison.csv", index=False)
    manifest = {"selected_layer": layer, "selected_representation": rep, "top_k": top_k, "feature_info": info, "embedding_methods": methods, "umap_split_ids": sorted(umap_split_ids), "guardrail": "coordinates are PCA/UMAP on activation vectors only", "required_controls": REQUIRED_CONTROLS}
    write_json(out / "embedding_manifest.json", manifest)
    save_markdown(out / "embedding_fit_report.md", "# Embedding Fit Report\n\n" + control.to_markdown(index=False) + "\n")
    print(control.to_string(index=False))
    print(f"Wrote {emb_dir / 'probe_guided_activation_embeddings.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
