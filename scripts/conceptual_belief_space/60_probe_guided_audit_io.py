#!/usr/bin/env python3
"""Audit inputs for the probe-guided activation PCA/UMAP pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from probe_guided_embedding_lib import ACTIONS, Q_COLS, ensure_dir, save_markdown, write_json, read_parquet_robust


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project_root", default=".")
    p.add_argument("--behavior", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def rel(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def inspect_feature(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    parts = path.stem.split("_")
    layer = int(parts[1])
    representation = "_".join(parts[2:])
    return {
        "path": str(path),
        "layer": layer,
        "representation": representation,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": int(path.stat().st_size),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = ensure_dir(rel(root, args.output_dir))
    behavior_path = rel(root, args.behavior)
    metadata_path = rel(root, args.metadata)
    features_dir = rel(root, args.features_dir)
    split_path = rel(root, args.splits)
    warnings: list[str] = []

    if not behavior_path.exists():
        warnings.append(f"Behavior table not found: {behavior_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Activation metadata not found: {metadata_path}")
    if not features_dir.exists():
        raise FileNotFoundError(f"Activation feature directory not found: {features_dir}")
    if not split_path.exists():
        warnings.append(f"Split file not found: {split_path}; downstream scripts can create grouped folds")

    meta = read_parquet_robust(metadata_path)
    behavior = read_parquet_robust(behavior_path) if behavior_path.exists() else pd.DataFrame()
    feature_rows = [inspect_feature(p) for p in sorted(features_dir.glob("layer_*_*.npy"))]
    if not feature_rows:
        raise FileNotFoundError(f"No activation .npy feature files found under {features_dir}")

    layers = sorted({r["layer"] for r in feature_rows})
    reps = sorted({r["representation"] for r in feature_rows})
    strict_cols = [
        c for c in [
            "strict_before_any_action_mention",
            "strict_before_final_action_mention",
            "is_before_first_action_mention",
            "is_before_first_final_action_mention",
            "first_action_mention_token_index",
            "first_final_action_mention_token_index",
        ]
        if c in meta.columns or c in behavior.columns
    ]
    alignment_keys = []
    for keys in [["row_index"], ["row_id"], ["prefix_id"], ["decision_id", "prefix_token_index"]]:
        if all(k in meta.columns for k in keys) and (behavior.empty or all(k in behavior.columns for k in keys)):
            alignment_keys.append(keys)
    q_found = [c for c in Q_COLS if c in meta.columns or c in behavior.columns]
    missing_q = [c for c in Q_COLS if c not in q_found]
    if missing_q:
        warnings.append(f"Missing q columns: {missing_q}")
    feature_shape_mismatches = [r for r in feature_rows if int(r["shape"][0]) != len(meta)]
    if feature_shape_mismatches:
        warnings.append("Some activation arrays do not have the same row count as activation metadata")

    split_info = None
    if split_path.exists():
        split_raw = json.loads(split_path.read_text())
        split_info = {
            "path": str(split_path),
            "n_splits": len(split_raw) if isinstance(split_raw, list) else None,
            "group_col": split_raw[0].get("group_col") if isinstance(split_raw, list) and split_raw else None,
        }

    manifest = {
        "behavior_table_path": str(behavior_path) if behavior_path.exists() else None,
        "activation_metadata_path": str(metadata_path),
        "activation_feature_dir": str(features_dir),
        "split_file_path": str(split_path) if split_path.exists() else None,
        "activation_feature_paths": feature_rows,
        "available_layers": layers,
        "available_representations": reps,
        "available_strict_pre_action_columns": strict_cols,
        "row_alignment_keys": alignment_keys,
        "metadata_shape": list(meta.shape),
        "behavior_shape": list(behavior.shape) if not behavior.empty else None,
        "actions": ACTIONS,
        "q_columns_found": q_found,
        "split_info": split_info,
        "warnings": warnings,
        "guardrail": "PCA/UMAP coordinates must be computed from activation vectors only; q and metadata columns are labels/metadata only.",
    }
    write_json(out / "io_manifest.json", manifest)

    lines = [
        "# Probe-Guided Activation Embedding IO Audit",
        "",
        f"- behavior table: `{manifest['behavior_table_path']}` shape `{manifest['behavior_shape']}`",
        f"- activation metadata: `{manifest['activation_metadata_path']}` shape `{manifest['metadata_shape']}`",
        f"- activation feature directory: `{manifest['activation_feature_dir']}`",
        f"- split file: `{manifest['split_file_path']}`",
        f"- available layers: `{layers}`",
        f"- available representations: `{reps}`",
        f"- strict pre-action columns: `{strict_cols}`",
        f"- row alignment keys: `{alignment_keys}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- none"])
    lines.extend(["", "## Feature Files", "", pd.DataFrame(feature_rows).to_markdown(index=False)])
    save_markdown(out / "io_audit_report.md", "\n".join(lines))
    print(f"Wrote {out / 'io_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
