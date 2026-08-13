#!/usr/bin/env python3
"""Run the full probe-guided activation embedding pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project_root", default=".")
    p.add_argument("--behavior", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    p.add_argument("--features_dir", default="outputs/activation/features")
    p.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    p.add_argument("--views", default="raw,delta_pre,unit_delta_pre", help="Comma-separated activation representations")
    p.add_argument("--layers", default="8,10,12,14,16,18,20,22")
    p.add_argument("--topk", default="64,128,256,512,1024")
    p.add_argument("--feature_sets", default="all_dims,probe_top_128,probe_top_512,random_top_512,shuffled_probe_top_512,progress_probe_top_512,probe_weighted_gamma_0.5")
    p.add_argument("--embeddings", default="pca,umap")
    p.add_argument("--residualize", default="none")
    p.add_argument("--max_umap_rows", type=int, default=20000)
    p.add_argument("--umap_splits", default="0")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--output_root", default=None, help="Alias for --output_dir")
    p.add_argument("--figure_root", default=None, help="Alias for --figure_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", type=parse_bool, default=False, help="Accepts true/false. Smoke mode uses the supplied narrow layer/view/feature-set arguments and prints an explicit summary.")
    return p.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    args = parse_args()
    if args.output_root is not None:
        args.output_dir = args.output_root
    if args.figure_root is not None:
        args.figure_dir = args.figure_root
    root = Path(args.project_root).resolve()
    script_dir = root / "scripts" / "conceptual_belief_space"
    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/codex_pycache")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    env.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")

    if args.smoke_test:
        print("Smoke test mode enabled.", flush=True)
        print(f"  views={args.views}", flush=True)
        print(f"  layers={args.layers}", flush=True)
        print(f"  feature_sets={args.feature_sets}", flush=True)
        print(f"  embeddings={args.embeddings}", flush=True)
        print(f"  max_umap_rows={args.max_umap_rows}", flush=True)
        print(f"  seed={args.seed}", flush=True)

    common = ["--project_root", str(root), "--behavior", args.behavior, "--metadata", args.metadata, "--features_dir", args.features_dir, "--splits", args.splits, "--output_dir", args.output_dir]
    run([py, str(script_dir / "60_probe_guided_audit_io.py"), *common], root, env)
    run([
        py,
        str(script_dir / "61_fit_probe_feature_importance.py"),
        *common,
        "--layers",
        args.layers,
        "--representations",
        args.views,
        "--topk",
        args.topk,
        "--seed",
        str(args.seed),
    ], root, env)
    run([
        py,
        str(script_dir / "62_build_probe_guided_activation_embeddings.py"),
        *common,
        "--probe_dir",
        args.output_dir,
        "--layers",
        args.layers,
        "--representations",
        args.views,
        "--feature_sets",
        args.feature_sets,
        "--embeddings",
        args.embeddings,
        "--residualize",
        args.residualize,
        "--max_umap_rows",
        str(args.max_umap_rows),
        "--umap_splits",
        args.umap_splits,
        "--seed",
        str(args.seed),
    ], root, env)
    run([
        py,
        str(script_dir / "64_evaluate_probe_guided_embeddings.py"),
        "--embeddings",
        str(root / args.output_dir / "embeddings" / "probe_guided_activation_embeddings.parquet"),
        "--output_dir",
        args.output_dir,
        "--figure_dir",
        args.figure_dir,
        "--seed",
        str(args.seed),
    ], root, env)
    run([
        py,
        str(script_dir / "63_plot_probe_guided_activation_embeddings.py"),
        "--embeddings",
        str(root / args.output_dir / "embeddings" / "probe_guided_activation_embeddings.parquet"),
        "--metrics",
        str(root / args.output_dir / "embedding_control_comparison.csv"),
        "--stability",
        str(root / args.output_dir / "feature_importance" / "feature_importance_stability.csv"),
        "--figure_dir",
        args.figure_dir,
        "--seed",
        str(args.seed),
    ], root, env)
    run([
        py,
        str(script_dir / "65_write_probe_guided_embedding_report.py"),
        "--output_dir",
        args.output_dir,
        "--figure_dir",
        args.figure_dir,
    ], root, env)
    print("\nProbe-guided activation embedding pipeline completed.", flush=True)
    print(f"Smoke test: {args.smoke_test}", flush=True)
    print(f"Report: {root / args.output_dir / 'probe_guided_activation_embedding_report.md'}", flush=True)
    print(f"Metrics: {root / args.output_dir / 'embedding_quality_metrics.csv'}", flush=True)
    print(f"Control comparison: {root / args.output_dir / 'embedding_control_comparison.csv'}", flush=True)
    print(f"Figures: {root / args.figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
