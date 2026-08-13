#!/usr/bin/env python3
"""Write a markdown report for probe-guided activation embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_guided_embedding_lib as lib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--output", default="outputs/conceptual_belief_space/probe_guided_activation_embedding/probe_guided_activation_embedding_final_report.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    manifest_path = output_dir / "probe_guided_embedding_manifest.json"
    metrics_path = output_dir / "probe_guided_embedding_metrics.csv"
    embeddings_path = output_dir / "probe_guided_activation_embeddings.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    manifest = json.loads(manifest_path.read_text())
    metrics = pd.read_csv(metrics_path)
    fig_manifest = figure_dir / "plot_manifest.csv"
    figures = pd.read_csv(fig_manifest) if fig_manifest.exists() else pd.DataFrame(columns=["figure", "description"])
    lines = [
        "# Probe-Guided Activation PCA/UMAP Report",
        "",
        "## Scientific Guardrail",
        "",
        "This experiment is a probe-guided activation manifold. It is not a probe-score manifold. The final x/y coordinates are produced by PCA or UMAP applied directly to activation feature vectors after train-only probe-guided dimension selection or weighting.",
        "",
        "Forbidden coordinate sources for this report: probe logits, probe probabilities, stacker scores, source scores, q values, and manually defined score contrasts such as `RIGHT - LEFT` or `UP - DOWN`.",
        "",
        "## Configuration",
        "",
        f"- layers: `{','.join(str(x) for x in manifest.get('layers', []))}`",
        f"- representation: `{manifest.get('representation')}`",
        f"- probe guidance methods: `{','.join(manifest.get('guidance_methods', []))}`",
        f"- embedding methods: `{','.join(manifest.get('embedding_methods', []))}`",
        f"- target used only for probe guidance: `{manifest.get('target')}`",
        f"- grouped split: split `{manifest.get('split_id')}` by `{manifest.get('group_col')}`",
        "",
        "## Probe Guide Metrics",
        "",
        pd.DataFrame([manifest.get("probe_metrics", {})]).to_markdown(index=False),
        "",
        "## Embedding Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Figures",
        "",
        figures.to_markdown(index=False) if not figures.empty else "No plot manifest found.",
        "",
        "## Output Files",
        "",
        f"- embeddings: `{embeddings_path}`",
        f"- metrics: `{metrics_path}`",
        f"- manifest: `{manifest_path}`",
        f"- figures: `{figure_dir}`",
        "",
        "## Interpretation",
        "",
        "A separation in these figures means that the selected or weighted activation subspace contains action-relevant variation that PCA/UMAP can expose. It does not mean that the probe scores themselves were used as coordinates. Because the probe guide is trained on the training split only, test points provide a leakage check for whether the activation view generalizes.",
    ]
    lib.save_markdown(args.output, "\n".join(lines))
    lib.log(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
