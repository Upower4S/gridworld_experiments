#!/usr/bin/env python3
"""Build PCA/UMAP embeddings from activation vectors using probe-guided feature sets."""
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
    apply_feature_set,
    compute_embedding_metrics,
    compute_q_labels,
    ensure_dir,
    load_activation_view,
    load_metadata,
    make_grouped_folds,
    parse_csv_ints,
    parse_csv_strings,
    residualize_progress,
    run_pca_embedding,
    run_umap_embedding,
    save_markdown,
    standardize_from_train,
    write_json,
)


def write_parquet_chunk(path: Path, frame: pd.DataFrame, writer_state: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer_state.get("writer") is None:
        writer_state["writer"] = pq.ParquetWriter(path, table.schema, compression="snappy")
    writer_state["writer"].write_table(table)


def centroid_distance_stats(df: pd.DataFrame, coords: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    centroids = {}
    for c in range(len(ACTIONS)):
        part = coords[train][labels[train] == c]
        if len(part):
            centroids[c] = part.mean(axis=0)
    dists = np.full(len(test), np.nan, dtype=np.float32)
    for n, idx in enumerate(test):
        c = int(labels[idx])
        if c in centroids:
            dists[n] = float(np.linalg.norm(coords[idx] - centroids[c]))
    out: dict[str, float] = {}
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


META_COLS = [
    "row_id",
    "row_index",
    "prefix_id",
    "trajectory_id",
    "decision_id",
    "step_id",
    "progress",
    "prefix_token_index",
    *Q_COLS,
    "q_top_action",
    "top_commitment",
    "entropy_commitment",
    "agent_action",
    "strict_pre_action",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project_root", default=".")
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--behavior", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--probe_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--layers", default="18")
    p.add_argument("--representations", default="raw")
    p.add_argument("--feature_sets", default="all_dims,probe_top_128,probe_top_512,random_top_512,shuffled_probe_top_512,progress_probe_top_512,probe_weighted_gamma_0.5")
    p.add_argument("--embeddings", default="pca,umap")
    p.add_argument("--residualize", default="none")
    p.add_argument("--max_umap_rows", type=int, default=20000)
    p.add_argument("--umap_splits", default="0")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--include_train_rows", action="store_true", help="Write train rows as well as held-out test rows. Defaults to false to keep full sweeps tractable.")
    return p.parse_args()


def rel(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_or_make_splits(path: Path, metadata: pd.DataFrame, group_col: str, n_splits: int, seed: int) -> list[dict]:
    if path.exists():
        raw = json.loads(path.read_text())
        splits = []
        for i, rec in enumerate(raw):
            train = np.asarray(rec.get("train_indices", rec.get("train_idx")), dtype=np.int64)
            test = np.asarray(rec.get("test_indices", rec.get("test_idx")), dtype=np.int64)
            train_groups = set(metadata.iloc[train][group_col].astype(str))
            test_groups = set(metadata.iloc[test][group_col].astype(str))
            if train_groups & test_groups:
                raise ValueError(f"Group leakage in split {i}: {sorted(train_groups & test_groups)[:5]}")
            splits.append({"split_id": int(rec.get("split_id", rec.get("fold", i))), "train_idx": train, "test_idx": test})
        return splits
    return make_grouped_folds(metadata, group_col=group_col, n_splits=n_splits, seed=seed)


def parse_umap_splits(text: str) -> set[int]:
    text = str(text).strip().lower()
    if text == "all":
        return set(range(1000))
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def load_feature_json(probe_dir: Path, view_name: str, feature_set: str) -> dict[int, list[int]]:
    path = probe_dir / "feature_sets" / f"{view_name}_{feature_set}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text())
    return {int(k): [int(vv) for vv in v] for k, v in obj["splits"].items()}


def feature_set_base(feature_set: str) -> str:
    if feature_set.startswith("probe_weighted_gamma_"):
        return "probe_top_512"
    return feature_set


def feature_set_gamma(feature_set: str) -> float:
    if feature_set.startswith("probe_weighted_gamma_"):
        return float(feature_set.split("_")[-1])
    return 0.5


def metadata_for_rows(metadata: pd.DataFrame, split_id: int, split_role: str) -> pd.DataFrame:
    cols = [c for c in META_COLS if c in metadata.columns]
    out = metadata[cols].copy()
    out["split_id"] = split_id
    out["split_role"] = split_role
    return out


def add_behavior_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()
    if "q_top_action" not in df.columns:
        df["q_top_action"] = compute_q_labels(df)
    q = df[Q_COLS].to_numpy(dtype=np.float64)
    df["top_commitment"] = np.max(q, axis=1)
    if "entropy_commitment" not in df.columns:
        entropy = -(q * np.log(np.clip(q, 1e-12, None))).sum(axis=1) / np.log(4)
        df["entropy_commitment"] = 1.0 - entropy
    if "strict_pre_action" not in df.columns:
        if "strict_before_any_action_mention" in df.columns:
            df["strict_pre_action"] = df["strict_before_any_action_mention"].fillna(False).astype(bool)
        elif "is_before_first_action_mention" in df.columns:
            df["strict_pre_action"] = df["is_before_first_action_mention"].fillna(False).astype(bool)
        else:
            df["strict_pre_action"] = False
    return df


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = ensure_dir(rel(root, args.output_dir))
    emb_dir = ensure_dir(out / "embeddings")
    emb_path = emb_dir / "probe_guided_activation_embeddings.parquet"
    if emb_path.exists():
        emb_path.unlink()
    writer_state: dict = {"writer": None}
    probe_dir = rel(root, args.probe_dir)
    metadata = add_behavior_columns(load_metadata(rel(root, args.metadata), rel(root, args.behavior)))
    splits = load_or_make_splits(rel(root, args.splits), metadata, args.group_col, args.n_splits, args.seed)
    layers = parse_csv_ints(args.layers)
    reps = parse_csv_strings(args.representations)
    feature_sets = parse_csv_strings(args.feature_sets)
    methods = parse_csv_strings(args.embeddings)
    residual_modes = parse_csv_strings(args.residualize)
    umap_splits = parse_umap_splits(args.umap_splits)
    meta_rows: list[dict] = []
    metric_rows: list[dict] = []
    labels = action_indices(metadata["q_top_action"])

    for layer in layers:
        for rep in reps:
            view_name = f"layer_{layer:02d}_{rep}"
            print(f"Loading activation view {view_name}", flush=True)
            x, info = load_activation_view(metadata, rel(root, args.features_dir), layer, rep)
            imp_path = probe_dir / "feature_importance" / f"{view_name}_feature_importance.parquet"
            if not imp_path.exists() and any(fs != "all_dims" for fs in feature_sets):
                raise FileNotFoundError(f"Missing feature importance for {view_name}: {imp_path}. Run 61_fit_probe_feature_importance.py first.")
            imp = pd.read_parquet(imp_path) if imp_path.exists() else pd.DataFrame()
            feature_maps = {fs: None for fs in feature_sets}
            for fs in feature_sets:
                if fs == "all_dims":
                    continue
                feature_maps[fs] = load_feature_json(probe_dir, view_name, feature_set_base(fs))
            for split in splits:
                sid = int(split["split_id"])
                train = np.asarray(split["train_idx"], dtype=np.int64)
                test = np.asarray(split["test_idx"], dtype=np.int64)
                xz, _mean, _scale = standardize_from_train(x, train)
                role = np.full(len(metadata), "unused", dtype=object)
                role[train] = "train"
                role[test] = "test"
                base_meta = metadata_for_rows(metadata, sid, "unused")
                base_meta["split_role"] = role
                for residual_mode in residual_modes:
                    xr, residual_info = residualize_progress(xz, metadata, train, residual_mode)
                    for fs in feature_sets:
                        selected = None if fs == "all_dims" else np.asarray(feature_maps[fs][sid], dtype=np.int64)
                        importance = None
                        if fs.startswith("probe_weighted"):
                            importance = imp[imp["split_id"].eq(sid)].sort_values("dimension")["l2"].to_numpy(dtype=np.float32)
                        x_view, view_info = apply_feature_set(xr, fs, selected=selected, importance=importance, gamma=feature_set_gamma(fs))
                        for method in methods:
                            if method == "umap" and sid not in umap_splits:
                                continue
                            print(f"Embedding split={sid} view={view_name} feature_set={fs} method={method} residual={residual_mode} dims={x_view.shape[1]}", flush=True)
                            if method == "pca":
                                coords, emb_info, _model = run_pca_embedding(x_view, train, seed=args.seed + sid)
                            elif method == "umap":
                                coords, emb_info, _model = run_umap_embedding(x_view, train, seed=args.seed + sid, max_train_rows=args.max_umap_rows)
                            else:
                                raise ValueError(f"Unknown embedding method: {method}")
                            metric = compute_embedding_metrics(coords, labels, train, test, metadata, seed=args.seed + sid)
                            metric.update(centroid_distance_stats(metadata, coords, labels, train, test))
                            metric_rows.append({
                                "split_id": sid,
                                "view_name": view_name,
                                "layer": layer,
                                "representation": rep,
                                "feature_set": fs,
                                "embedding_method": method,
                                "residualize": residual_mode,
                                "n_train_rows": len(train),
                                "n_test_rows": len(test),
                                **view_info,
                                **residual_info,
                                **emb_info,
                                **metric,
                            })
                            keep_idx = np.arange(len(metadata), dtype=np.int64) if args.include_train_rows else test
                            frame = base_meta.iloc[keep_idx].copy()
                            frame["representation"] = rep
                            frame["layer"] = layer
                            frame["view_name"] = view_name
                            frame["feature_set"] = fs
                            frame["embedding_method"] = method
                            frame["residualize"] = residual_mode
                            frame["coord_x"] = coords[keep_idx, 0]
                            frame["coord_y"] = coords[keep_idx, 1]
                            frame["embedding_x"] = frame["coord_x"]
                            frame["embedding_y"] = frame["coord_y"]
                            frame["embedding_coordinate_source"] = emb_info["embedding_coordinate_source"]
                            frame["pca_umap_input_feature_kind"] = "activation_vectors_only"
                            frame["forbidden_coordinate_source_used"] = False
                            frame["coordinate_frame_id"] = f"split_{sid}_{view_name}_{fs}_{method}_{residual_mode}"
                            write_parquet_chunk(emb_path, frame, writer_state)
                            meta_rows.append({
                                "split_id": sid,
                                "view_name": view_name,
                                "layer": layer,
                                "representation": rep,
                                "feature_set": fs,
                                "embedding_method": method,
                                "residualize": residual_mode,
                                "n_train_rows": len(train),
                                "n_test_rows": len(test),
                                **view_info,
                                **residual_info,
                                **emb_info,
                                "activation_input_path": info["path"],
                            })
                        del x_view
                    del xr
                del xz
            del x

    if writer_state.get("writer") is not None:
        writer_state["writer"].close()
    pd.DataFrame(meta_rows).to_csv(emb_dir / "embedding_metadata.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(emb_dir / "embedding_quality_metrics_precomputed.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "embedding_quality_metrics.csv", index=False)
    write_json(out / "embedding_build_manifest.json", {
        "layers": layers,
        "representations": reps,
        "feature_sets": feature_sets,
        "embedding_methods": methods,
        "residualize": residual_modes,
        "guardrail": "coord_x/coord_y are PCA/UMAP coordinates computed from activation features only",
    })
    save_markdown(out / "embedding_build_report.md", f"# Embedding Build Report\n\nWrote `{emb_path}`. Full sweeps write held-out rows only by default; held-out metrics are precomputed during embedding construction.\n")
    print(f"Wrote {emb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
