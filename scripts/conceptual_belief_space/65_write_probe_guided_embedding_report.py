#!/usr/bin/env python3
"""Write the final report for probe-guided activation PCA/UMAP embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from probe_guided_embedding_lib import load_manifest, save_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/probe_guided_activation_embedding")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/probe_guided_activation_embedding")
    return p.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    fig = Path(args.figure_dir)
    io = load_manifest(out / "io_manifest.json") if (out / "io_manifest.json").exists() else {}
    probe = read_csv_if_exists(out / "probe_metrics.csv")
    quality = read_csv_if_exists(out / "embedding_quality_metrics.csv")
    controls = read_csv_if_exists(out / "embedding_control_comparison.csv")
    plots = read_csv_if_exists(fig / "plot_index.csv")
    stability = read_csv_if_exists(out / "feature_importance" / "feature_importance_stability.csv")

    improves_text = "could not be determined because control metrics were missing"
    if not controls.empty and "probe_guided_improves_over_best_required_control" in controls.columns:
        improves = bool(controls["probe_guided_improves_over_best_required_control"].dropna().iloc[0]) if controls["probe_guided_improves_over_best_required_control"].notna().any() else False
        if improves:
            improves_text = "improves over the best required control under the primary held-out kNN balanced-accuracy metric"
        else:
            improves_text = "does not improve over the best required control under the primary held-out kNN balanced-accuracy metric"

    lines = [
        "# Probe-Guided Activation PCA/UMAP Report",
        "",
        "## Required Scientific Caveats",
        "",
        "1. Coordinates are PCA/UMAP coordinates computed from activation vectors.",
        "2. Probe scores are not used as coordinates.",
        "3. The linear probe is used only for feature selection, feature weighting, or view selection.",
        "4. This is a probe-guided activation manifold, not a probe-score manifold.",
        "5. It is not fully unsupervised because feature selection is supervised.",
        "6. Results are compared against all_dims, random_top_K, shuffled_probe_top_K, and progress_probe_top_K controls.",
        "7. Weak or negative results are reported honestly. No causal claim is made.",
        "",
        "## Inputs",
        "",
        f"- behavior table: `{io.get('behavior_table_path')}`",
        f"- activation metadata: `{io.get('activation_metadata_path')}`",
        f"- activation features: `{io.get('activation_feature_dir')}`",
        f"- split file: `{io.get('split_file_path')}`",
        f"- layers: `{io.get('available_layers')}`",
        f"- representations: `{io.get('available_representations')}`",
        f"- strict pre-action columns: `{io.get('available_strict_pre_action_columns')}`",
        "",
        "## Probe Feature Selection",
        "",
        probe.to_markdown(index=False) if not probe.empty else "Probe metrics are missing.",
        "",
        "## Feature Importance Stability",
        "",
        stability.to_markdown(index=False) if not stability.empty else "Feature stability metrics are missing.",
        "",
        "## Embedding Quality Metrics",
        "",
        quality.to_markdown(index=False) if not quality.empty else "Embedding quality metrics are missing.",
        "",
        "## Control Comparison",
        "",
        controls.to_markdown(index=False) if not controls.empty else "Control comparison metrics are missing.",
        "",
        "## Main Result",
        "",
        f"The best probe-guided activation embedding {improves_text}.",
        "This statement is based on trajectory-grouped held-out splits and the 2D embedding metrics, not on in-sample visualization.",
        "",
        "## Figures",
        "",
        plots.to_markdown(index=False) if not plots.empty else "Plot index is missing.",
        "",
        "## Output Contract",
        "",
        "- `io_manifest.json`",
        "- `probe_metrics.csv`",
        "- `feature_importance/`",
        "- `feature_sets/`",
        "- `embeddings/probe_guided_activation_embeddings.parquet`",
        "- `embeddings/embedding_metadata.csv`",
        "- `embedding_quality_metrics.csv`",
        "- `embedding_control_comparison.csv`",
        "- `figures/conceptual_belief_space/probe_guided_activation_embedding/plot_index.csv`",
    ]
    save_markdown(out / "probe_guided_activation_embedding_report.md", "\n".join(lines))
    print(f"Wrote {out / 'probe_guided_activation_embedding_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
