#!/usr/bin/env python3
"""Write final report for strict probe-guided activation embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from probe_guided_activation_embedding_lib import REQUIRED_CONTROLS, ensure_dir, read_json, save_markdown


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    fig = Path(args.figure_dir)
    io = read_json(out / "io_manifest.json")
    config = read_json(out / "selected_probe_guidance_config.json")
    probe = pd.read_csv(out / "probe_metrics.csv")
    quality = pd.read_csv(out / "embedding_quality_metrics.csv")
    controls = pd.read_csv(out / "embedding_control_comparison.csv")
    plots = pd.read_csv(fig / "plot_index.csv") if (fig / "plot_index.csv").exists() else pd.DataFrame()
    best_probe = controls[controls["feature_set_family"].isin(["probe_top_K", "probe_weighted_top_K"])].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    best_control = controls[controls["feature_set_family"].isin(REQUIRED_CONTROLS)].sort_values("test_knn_balanced_accuracy", ascending=False).head(1)
    improves = False
    if not best_probe.empty and not best_control.empty:
        improves = float(best_probe["test_knn_balanced_accuracy"].iloc[0]) > float(best_control["test_knn_balanced_accuracy"].iloc[0])
    interpretation = "improves over the best required control" if improves else "does not improve over the best required control under the primary kNN balanced-accuracy metric"
    lines = [
        "# Probe-Guided Activation PCA/UMAP Report", "",
        "## Scientific Guardrail", "",
        "This is a probe-guided activation manifold. It is not a probe-score manifold and not an unsupervised intrinsic activation manifold.",
        "The final x/y coordinates are produced by PCA or UMAP applied directly to activation vectors, after train-only probe-guided dimension selection or weighting.",
        "Probe logits, probe probabilities, source scores, stacker scores, q labels, commitment labels, progress, token index, trajectory_id, decision_id, and action labels are not PCA/UMAP input features.",
        "No causal claim is made.", "",
        "## Inputs", "",
        f"- metadata: `{io['metadata']['path']}` shape `{io['metadata']['shape']}`",
        f"- features: `{io['features']['features_dir']}` arrays `{io['features']['n_arrays']}`",
        f"- available layers: `{io['features']['available_layers']}`",
        f"- available representations: `{io['features']['available_representations']}`",
        f"- splits: `{len(io['splits'])}` trajectory-grouped folds", "",
        "## Selected Probe Guidance", "",
        f"- selected layer: `{int(config['layer'])}`",
        f"- selected representation: `{config['representation']}`",
        f"- selection rule: `{config.get('selection_rule')}`", "",
        "## Probe Metrics", "",
        probe.to_markdown(index=False), "",
        "## Embedding Quality Metrics", "",
        quality.to_markdown(index=False), "",
        "## Required Control Comparison", "",
        controls.to_markdown(index=False), "",
        "## Conclusion", "",
        f"The best probe-guided embedding `{best_probe['feature_set_family'].iloc[0] if not best_probe.empty else 'NA'}` {interpretation}.",
        "Weak, negative, or failed metrics are reported as-is above.", "",
        "## Figures", "",
        plots.to_markdown(index=False) if not plots.empty else "No plot index found.", "",
        "## Output Files", "",
        f"- `io_manifest.json`",
        f"- `probe_metrics.csv`",
        f"- `feature_importance/`",
        f"- `feature_sets/`",
        f"- `embeddings/probe_guided_activation_embeddings.parquet`",
        f"- `embedding_quality_metrics.csv`",
        f"- `embedding_control_comparison.csv`",
        f"- `{fig / 'plot_index.csv'}`",
    ]
    save_markdown(out / "probe_guided_activation_embedding_report.md", "\n".join(lines))
    print(f"Wrote {out / 'probe_guided_activation_embedding_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
