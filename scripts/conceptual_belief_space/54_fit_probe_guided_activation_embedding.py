#!/usr/bin/env python3
"""Fit probe-guided activation PCA/UMAP embeddings.

The linear probe is used only to guide activation dimensions by coefficient
importance.  PCA/UMAP coordinates are fit from activation feature vectors.  This
script never uses probe scores, stacker scores, source scores, q values, or
metadata columns as embedding input features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_guided_embedding_lib as lib

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--layers", default="20", help="Comma-separated layers or all")
    p.add_argument("--representation", default="raw")
    p.add_argument("--guidance_methods", default="probe_select_topk,probe_weighted_topk", help="probe_select_topk,probe_weighted_topk,probe_weighted_all,plain_activation_baseline")
    p.add_argument("--embedding_methods", default="pca", help="pca,umap, or both comma-separated")
    p.add_argument("--top_k", type=int, default=256)
    p.add_argument("--target", choices=["true_q_top_action", "agent_action"], default="true_q_top_action")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--probe_alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=lib.DEFAULT_SEED)
    p.add_argument("--max_feature_gb", type=float, default=8.0)
    p.add_argument("--max_umap_train_rows", type=int, default=15000)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", default="cosine")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_output_paths(output_dir: Path, overwrite: bool) -> None:
    targets = [
        output_dir / "probe_guided_activation_embeddings.parquet",
        output_dir / "probe_guided_embedding_metrics.csv",
        output_dir / "probe_guided_embedding_manifest.json",
        output_dir / "probe_guided_embedding_report.md",
    ]
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs without --overwrite: {[str(p) for p in existing]}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = lib.ensure_dir(args.output_dir)
    validate_output_paths(output_dir, args.overwrite)

    layers = lib.parse_csv_ints(args.layers) if str(args.layers).lower() != "all" else lib.discover_layers(Path(args.features_dir), args.representation)
    guidance_methods = lib.parse_csv_strings(args.guidance_methods)
    embedding_methods = lib.parse_csv_strings(args.embedding_methods)
    if not layers:
        raise ValueError("No layers selected")
    bad_embed = [m for m in embedding_methods if m not in {"pca", "umap"}]
    if bad_embed:
        raise ValueError(f"Unsupported embedding methods: {bad_embed}")

    lib.log(f"Loading activation metadata: {args.metadata}")
    df = lib.load_metadata(Path(args.metadata), Path(args.beliefs) if args.beliefs else None)
    split_list = lib.make_group_splits(df, args.n_splits, args.seed, group_col=args.group_col)
    split_ids = [int(s["split_id"]) for s in split_list]
    if args.split_id not in split_ids:
        raise ValueError(f"Requested split_id={args.split_id}, available={split_ids}")
    split = [s for s in split_list if int(s["split_id"]) == int(args.split_id)][0]
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    if set(df.iloc[train_idx][args.group_col].astype(str)) & set(df.iloc[test_idx][args.group_col].astype(str)):
        raise RuntimeError("Group leakage detected between train and test rows")
    df = lib.add_split_role(df, train_idx, test_idx, args.split_id)

    lib.log(f"Loading activation features: layers={layers} representation={args.representation}")
    x, feature_infos = lib.load_activation_features(df, Path(args.features_dir), layers, args.representation, max_feature_gb=args.max_feature_gb)
    lib.log(f"Activation matrix shape: {x.shape}; source arrays={len(feature_infos)}")
    lib.assert_no_score_columns_as_features([])

    y = lib.action_indices(df[args.target])
    lib.log(f"Standardizing activation features from train rows only: n_train={len(train_idx)} n_test={len(test_idx)}")
    xz, mean, scale = lib.standardize_from_train(x, train_idx)
    del x

    lib.log(f"Fitting linear probe guide target={args.target} alpha={args.probe_alpha}")
    probe, importance, probe_metrics = lib.fit_linear_probe(xz, y, train_idx, test_idx, alpha=args.probe_alpha)
    top_order = np.argsort(-importance)
    lib.log(f"Probe test balanced accuracy: {probe_metrics['probe_test_balanced_accuracy']:.4f}")

    metadata_out = lib.safe_metadata_for_output(df)
    frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    feature_selection: dict[str, Any] = {}

    for guidance in guidance_methods:
        lib.log(f"Building guided activation view: {guidance}")
        x_view, selected_dims, guidance_info = lib.make_guided_view(xz, importance, guidance, args.top_k)
        feature_selection[guidance] = {
            **guidance_info,
            "n_input_dimensions": int(x_view.shape[1]),
            "selected_dimension_indices_first_50": selected_dims[:50].astype(int).tolist(),
            "selected_dimension_importances_first_50": importance[selected_dims[:50]].astype(float).tolist(),
        }
        for method in embedding_methods:
            lib.log(f"Fitting {method.upper()} on activation view={guidance}; coordinates come from activation vectors")
            if method == "pca":
                coords, emb_info, _model = lib.fit_pca_embedding(x_view, train_idx, args.seed)
            else:
                coords, emb_info, _model = lib.fit_umap_embedding(
                    x_view,
                    train_idx,
                    args.seed,
                    max_train_rows=args.max_umap_train_rows,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                )
            out = metadata_out.copy()
            out["embedding_x"] = coords[:, 0]
            out["embedding_y"] = coords[:, 1]
            out["embedding_method"] = method
            out["probe_guidance_method"] = guidance
            out["layer_spec"] = ",".join(str(x) for x in layers)
            out["representation"] = args.representation
            out["embedding_input_kind"] = "activation_features_probe_guided"
            out["coordinate_source"] = "PCA/UMAP fit on activation vectors after probe-guided dimension selection/weighting"
            out["forbidden_coordinate_source"] = "not probe logits, not stacker/source scores, not q values"
            frames.append(out)
            metric = {
                "embedding_method": method,
                "probe_guidance_method": guidance,
                "layer_spec": ",".join(str(x) for x in layers),
                "representation": args.representation,
                "split_id": int(args.split_id),
                "n_rows": int(len(out)),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_embedding_input_dimensions": int(x_view.shape[1]),
                **probe_metrics,
                **emb_info,
            }
            metric.update(lib.metric_rows_for_embedding(out, "embedding_x", "embedding_y", "true_q_top_action"))
            metric_rows.append(metric)

    embeddings = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    embeddings_path = output_dir / "probe_guided_activation_embeddings.parquet"
    metrics_path = output_dir / "probe_guided_embedding_metrics.csv"
    embeddings.to_parquet(embeddings_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    np.savez_compressed(
        output_dir / "probe_guided_feature_importance.npz",
        importance=importance.astype(np.float32),
        top_dimension_indices=top_order.astype(np.int64),
        scaler_mean=mean.astype(np.float32),
        scaler_scale=scale.astype(np.float32),
        layers=np.asarray(layers, dtype=np.int64),
    )
    manifest = {
        "method": "probe-guided activation PCA/UMAP embedding",
        "guardrail": "embedding coordinates are PCA/UMAP applied to activation vectors only; probe scores/logits/q values are not coordinate inputs",
        "metadata_path": str(args.metadata),
        "beliefs_path": str(args.beliefs),
        "features_dir": str(args.features_dir),
        "layers": layers,
        "representation": args.representation,
        "guidance_methods": guidance_methods,
        "embedding_methods": embedding_methods,
        "top_k": int(args.top_k),
        "target": args.target,
        "split_id": int(args.split_id),
        "group_col": args.group_col,
        "feature_sources": feature_infos,
        "feature_selection": feature_selection,
        "probe_metrics": probe_metrics,
        "output_files": {
            "embeddings": str(embeddings_path),
            "metrics": str(metrics_path),
            "feature_importance": str(output_dir / "probe_guided_feature_importance.npz"),
        },
    }
    lib.write_json(output_dir / "probe_guided_embedding_manifest.json", manifest)
    report = [
        "# Probe-Guided Activation Embedding",
        "",
        "## Guardrail",
        "",
        "The linear probe was used only to select or weight activation dimensions. The plotted coordinates are PCA/UMAP coordinates fit on activation feature vectors. They are not probe logits, source scores, stacker scores, probabilities, or q values.",
        "",
        "## Configuration",
        "",
        f"- layers: `{','.join(str(x) for x in layers)}`",
        f"- representation: `{args.representation}`",
        f"- guidance methods: `{','.join(guidance_methods)}`",
        f"- embedding methods: `{','.join(embedding_methods)}`",
        f"- target used for probe guidance: `{args.target}`",
        f"- split: `{args.split_id}` grouped by `{args.group_col}`",
        "",
        "## Probe Guide Metrics",
        "",
        pd.DataFrame([probe_metrics]).to_markdown(index=False),
        "",
        "## Embedding Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- embeddings: `{embeddings_path}`",
        f"- metrics: `{metrics_path}`",
        f"- manifest: `{output_dir / 'probe_guided_embedding_manifest.json'}`",
        f"- feature importance: `{output_dir / 'probe_guided_feature_importance.npz'}`",
    ]
    lib.save_markdown(output_dir / "probe_guided_embedding_report.md", "\n".join(report))
    lib.log(f"Wrote {embeddings_path}")
    lib.log(f"Wrote {metrics_path}")
    lib.log(f"Wrote {output_dir / 'probe_guided_embedding_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
