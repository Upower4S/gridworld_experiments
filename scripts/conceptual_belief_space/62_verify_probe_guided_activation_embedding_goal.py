#!/usr/bin/env python3
"""Verify strict probe-guided activation embedding outputs and guardrails."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from probe_guided_activation_embedding_lib import REQUIRED_CONTROLS, ensure_dir, read_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    fig = Path(args.figure_dir)
    required = [
        out / "io_manifest.json",
        out / "probe_metrics.csv",
        out / "feature_importance",
        out / "feature_sets",
        out / "embeddings" / "probe_guided_activation_embeddings.parquet",
        out / "embedding_quality_metrics.csv",
        out / "embedding_control_comparison.csv",
        fig / "plot_index.csv",
        out / "probe_guided_activation_embedding_report.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")
    emb = pd.read_parquet(out / "embeddings" / "probe_guided_activation_embeddings.parquet")
    controls = pd.read_csv(out / "embedding_control_comparison.csv")
    report = (out / "probe_guided_activation_embedding_report.md").read_text()
    present_controls = sorted(set(controls["feature_set_family"]) & set(REQUIRED_CONTROLS))
    if set(present_controls) != set(REQUIRED_CONTROLS):
        raise RuntimeError(f"Missing required controls. present={present_controls}, required={REQUIRED_CONTROLS}")
    if "forbidden_coordinate_source_used" not in emb.columns or emb["forbidden_coordinate_source_used"].fillna(True).astype(bool).any():
        raise RuntimeError("Embedding parquet indicates forbidden coordinate sources were used")
    if "pca_umap_input_feature_kind" not in emb.columns or not emb["pca_umap_input_feature_kind"].eq("activation_vectors_only").all():
        raise RuntimeError("Embedding input kind is not activation_vectors_only for all rows")
    if not emb["coordinate_source"].astype(str).str.contains("activation vectors", case=False, regex=False).all():
        raise RuntimeError("Coordinate source does not explicitly state activation vectors")
    if "probe-guided activation manifold" not in report.lower():
        raise RuntimeError("Report does not clearly call this a probe-guided activation manifold")
    if "not a probe-score manifold" not in report.lower():
        raise RuntimeError("Report does not distinguish against probe-score manifold")
    if "No causal claim" not in report and "no causal" not in report.lower():
        raise RuntimeError("Report does not include causal caveat")
    best_probe = controls[controls["feature_set_family"].isin(["probe_top_K", "probe_weighted_top_K"])].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    best_control = controls[controls["feature_set_family"].isin(REQUIRED_CONTROLS)].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    improves = bool((not best_probe.empty) and (not best_control.empty) and float(best_probe["test_knn_balanced_accuracy"].iloc[0]) > float(best_control["test_knn_balanced_accuracy"].iloc[0]))
    passed = {
        "passed": True,
        "required_outputs_present": True,
        "required_controls_present": present_controls,
        "forbidden_coordinate_source_used": False,
        "pca_umap_input_feature_kind": "activation_vectors_only",
        "embedding_methods": sorted(emb["embedding_method"].unique().tolist()),
        "feature_set_families": sorted(emb["feature_set_family"].unique().tolist()),
        "best_probe_family": None if best_probe.empty else str(best_probe["feature_set_family"].iloc[0]),
        "best_probe_test_knn_balanced_accuracy": None if best_probe.empty else float(best_probe["test_knn_balanced_accuracy"].iloc[0]),
        "best_required_control_family": None if best_control.empty else str(best_control["feature_set_family"].iloc[0]),
        "best_required_control_test_knn_balanced_accuracy": None if best_control.empty else float(best_control["test_knn_balanced_accuracy"].iloc[0]),
        "probe_guided_improves_over_best_required_control": improves,
        "note": "GOAL_PASSED verifies guardrails and required comparisons. It does not imply causal evidence.",
    }
    write_json(out / "GOAL_PASSED.json", passed)
    print(f"Wrote {out / 'GOAL_PASSED.json'}")
    print(passed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
