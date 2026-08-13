#!/usr/bin/env python3
"""Step 6: centroid distance matrix alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED  # noqa: E402
from src.conceptual_belief_space.geometry import (  # noqa: E402
    cosine_distance_matrix,
    euclidean_distance_matrix,
    highdim_centroid_lookup,
    highdim_matrix_for_actions,
    matrix_npz_key,
    ordered_action_matrix,
    parse_activation_highdim_centroids,
    upper_triangle_values,
)
from src.conceptual_belief_space.io import ensure_dir, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.manifold import slugify  # noqa: E402
from src.conceptual_belief_space.plotting import distance_heatmap, savefig  # noqa: E402
from src.conceptual_belief_space.stats import (  # noqa: E402
    bootstrap_corr_ci,
    correlation_summary,
    permutation_test_spearman,
)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Run Steps 4 and 5 after installing UMAP with:\n"
            "  pip install umap-learn"
        )


def behavior_feature_cols(group: pd.DataFrame) -> list[str]:
    cols = [col for col in group.columns if col.startswith("centroid_")]
    if not cols:
        raise ValueError("behavior centroids do not contain high-dimensional centroid_* columns")
    return cols


def validate_centroid_tables(behavior: pd.DataFrame, activation_2d: pd.DataFrame) -> None:
    required_behavior = ["split_id", "subset", "representation", "action", "beh_umap1", "beh_umap2"]
    missing_behavior = [col for col in required_behavior if col not in behavior.columns]
    if missing_behavior:
        raise ValueError(f"behavior centroids missing columns: {missing_behavior}")
    required_activation = [
        "split_id",
        "subset",
        "layer",
        "activation_representation",
        "action",
        "act_umap1",
        "act_umap2",
    ]
    missing_activation = [col for col in required_activation if col not in activation_2d.columns]
    if missing_activation:
        raise ValueError(f"activation 2D centroids missing columns: {missing_activation}")


def add_matrix(matrix_store: dict[str, np.ndarray], key: str, matrix: np.ndarray) -> str:
    matrix_store[key] = np.asarray(matrix, dtype=np.float32)
    return key


def compare_matrices(
    *,
    behavior_matrix: np.ndarray,
    activation_matrix: np.ndarray,
    split_id: int,
    subset: str,
    behavior_representation: str,
    activation_layer: int,
    activation_representation: str,
    behavior_distance_kind: str,
    activation_distance_kind: str,
    behavior_matrix_key: str,
    activation_matrix_key: str,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    behavior_values = upper_triangle_values(behavior_matrix)
    activation_values = upper_triangle_values(activation_matrix)
    correlations = correlation_summary(behavior_values, activation_values)
    permutation = permutation_test_spearman(
        behavior_matrix,
        activation_matrix,
        n_permutations=n_permutations,
        seed=seed,
        upper_triangle_fn=upper_triangle_values,
    )
    bootstrap = bootstrap_corr_ci(
        behavior_values,
        activation_values,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "split_id": int(split_id),
        "subset": subset,
        "behavior_representation": behavior_representation,
        "activation_layer": int(activation_layer),
        "activation_representation": activation_representation,
        "behavior_distance_kind": behavior_distance_kind,
        "activation_distance_kind": activation_distance_kind,
        "is_primary": bool(
            behavior_distance_kind == "behavior_highdim_euclidean"
            and activation_distance_kind == "activation_highdim_cosine"
        ),
        "n_pairwise_distances": int(len(behavior_values)),
        **correlations,
        **permutation,
        **bootstrap,
        "behavior_matrix_key": behavior_matrix_key,
        "activation_matrix_key": activation_matrix_key,
    }


def build_behavior_matrices(
    behavior_group: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    behavior_representation: str,
    matrix_store: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, str]]:
    feature_cols = behavior_feature_cols(behavior_group)
    highdim = ordered_action_matrix(behavior_group, feature_cols)
    umap2d = ordered_action_matrix(behavior_group, ["beh_umap1", "beh_umap2"])

    highdim_matrix = euclidean_distance_matrix(highdim)
    umap_matrix = euclidean_distance_matrix(umap2d)
    highdim_key = add_matrix(
        matrix_store,
        matrix_npz_key(
            kind="behavior",
            split_id=split_id,
            subset=subset,
            representation=behavior_representation,
            distance_kind="highdim_euclidean",
        ),
        highdim_matrix,
    )
    umap_key = add_matrix(
        matrix_store,
        matrix_npz_key(
            kind="behavior",
            split_id=split_id,
            subset=subset,
            representation=behavior_representation,
            distance_kind="umap2d_euclidean",
        ),
        umap_matrix,
    )
    return {
        "behavior_highdim_euclidean": (highdim_matrix, highdim_key),
        "behavior_umap2d_euclidean": (umap_matrix, umap_key),
    }


def build_activation_matrices(
    activation_group: pd.DataFrame,
    highdim_lookup: dict[tuple[int, str, int, str], dict[str, np.ndarray]],
    *,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
    matrix_store: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, str]]:
    matrices: dict[str, tuple[np.ndarray, str]] = {}
    subset_key = slugify(subset)
    repr_key = slugify(activation_representation)
    action_vectors = highdim_lookup.get((split_id, subset_key, layer, repr_key))
    if action_vectors is None:
        action_vectors = highdim_lookup.get((split_id, subset, layer, activation_representation))
    if action_vectors is not None:
        highdim = highdim_matrix_for_actions(action_vectors)
        cosine_matrix = cosine_distance_matrix(highdim)
        euclidean_matrix = euclidean_distance_matrix(highdim)
        cosine_key = add_matrix(
            matrix_store,
            matrix_npz_key(
                kind="activation",
                split_id=split_id,
                subset=subset,
                layer=layer,
                representation=activation_representation,
                distance_kind="highdim_cosine",
            ),
            cosine_matrix,
        )
        euclidean_key = add_matrix(
            matrix_store,
            matrix_npz_key(
                kind="activation",
                split_id=split_id,
                subset=subset,
                layer=layer,
                representation=activation_representation,
                distance_kind="highdim_euclidean",
            ),
            euclidean_matrix,
        )
        matrices["activation_highdim_cosine"] = (cosine_matrix, cosine_key)
        matrices["activation_highdim_euclidean"] = (euclidean_matrix, euclidean_key)

    umap2d = ordered_action_matrix(activation_group, ["act_umap1", "act_umap2"])
    umap_matrix = euclidean_distance_matrix(umap2d)
    umap_key = add_matrix(
        matrix_store,
        matrix_npz_key(
            kind="activation",
            split_id=split_id,
            subset=subset,
            layer=layer,
            representation=activation_representation,
            distance_kind="umap2d_euclidean",
        ),
        umap_matrix,
    )
    matrices["activation_umap2d_euclidean"] = (umap_matrix, umap_key)
    return matrices


def compute_metrics(
    behavior: pd.DataFrame,
    activation_2d: pd.DataFrame,
    activation_highdim_npz: np.lib.npyio.NpzFile,
    *,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    matrix_store: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    highdim_records = parse_activation_highdim_centroids(activation_highdim_npz)
    highdim_lookup = highdim_centroid_lookup(highdim_records)

    activation_groups = {
        (int(split_id), str(subset), int(layer), str(representation)): group.copy()
        for (split_id, subset, layer, representation), group in activation_2d.groupby(
            ["split_id", "subset", "layer", "activation_representation"], sort=True
        )
    }

    rows: list[dict[str, Any]] = []
    for (split_id_raw, subset_raw, behavior_representation_raw), behavior_group in behavior.groupby(
        ["split_id", "subset", "representation"], sort=True
    ):
        split_id = int(split_id_raw)
        subset = str(subset_raw)
        behavior_representation = str(behavior_representation_raw)
        try:
            behavior_matrices = build_behavior_matrices(
                behavior_group,
                split_id=split_id,
                subset=subset,
                behavior_representation=behavior_representation,
                matrix_store=matrix_store,
            )
        except Exception as exc:
            skipped.append(
                f"behavior split={split_id}, subset={subset}, representation={behavior_representation}: {exc}"
            )
            continue

        matching_activation_keys = [
            key for key in activation_groups if key[0] == split_id and key[1] == subset
        ]
        if not matching_activation_keys:
            skipped.append(f"no activation centroids for split={split_id}, subset={subset}")
            continue

        for key in matching_activation_keys:
            _, _, layer, activation_representation = key
            activation_group = activation_groups[key]
            try:
                activation_matrices = build_activation_matrices(
                    activation_group,
                    highdim_lookup,
                    split_id=split_id,
                    subset=subset,
                    layer=layer,
                    activation_representation=activation_representation,
                    matrix_store=matrix_store,
                )
            except Exception as exc:
                skipped.append(
                    f"activation split={split_id}, subset={subset}, layer={layer}, "
                    f"representation={activation_representation}: {exc}"
                )
                continue

            for behavior_kind, (behavior_matrix, behavior_key) in behavior_matrices.items():
                for activation_kind, (activation_matrix, activation_key) in activation_matrices.items():
                    rows.append(
                        compare_matrices(
                            behavior_matrix=behavior_matrix,
                            activation_matrix=activation_matrix,
                            split_id=split_id,
                            subset=subset,
                            behavior_representation=behavior_representation,
                            activation_layer=layer,
                            activation_representation=activation_representation,
                            behavior_distance_kind=behavior_kind,
                            activation_distance_kind=activation_kind,
                            behavior_matrix_key=behavior_key,
                            activation_matrix_key=activation_key,
                            n_permutations=n_permutations,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                    )

    return pd.DataFrame(rows), matrix_store, skipped


def best_primary(metrics: pd.DataFrame) -> pd.Series | None:
    if metrics.empty or "is_primary" not in metrics.columns:
        return None
    primary = metrics.loc[metrics["is_primary"]].copy()
    primary = primary.loc[np.isfinite(primary["spearman"].to_numpy(dtype=float))]
    if primary.empty:
        return None
    return primary.sort_values(["spearman", "pearson"], ascending=False).iloc[0]


def plot_line_by_layer(metrics: pd.DataFrame, path: Path, *, subset_filter: str | None = None) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = metrics.loc[metrics["is_primary"]].copy()
    if subset_filter is not None:
        primary = primary.loc[primary["subset"].astype(str).str.contains(subset_filter, regex=False)]
    if primary.empty:
        return False
    grouped = (
        primary.groupby(["activation_representation", "activation_layer"])["spearman"]
        .mean()
        .reset_index()
        .sort_values("activation_layer")
    )
    if grouped.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for representation, part in grouped.groupby("activation_representation"):
        ax.plot(part["activation_layer"], part["spearman"], marker="o", label=str(representation))
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("activation layer")
    ax.set_ylabel("mean Spearman distance correlation")
    title = "Primary geometry alignment by layer"
    if subset_filter is not None:
        title = f"Primary geometry alignment by layer ({subset_filter})"
    ax.set_title(title)
    ax.legend(frameon=False)
    savefig(path, fig=fig)
    return True


def plot_by_representation(metrics: pd.DataFrame, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = metrics.loc[metrics["is_primary"]].copy()
    if primary.empty:
        return False
    grouped = primary.groupby("activation_representation")["spearman"].mean().sort_values(ascending=False)
    if grouped.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grouped.index.astype(str), grouped.to_numpy(dtype=float), color="#4C78A8")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("activation representation")
    ax.set_ylabel("mean Spearman distance correlation")
    ax.set_title("Primary geometry alignment by activation representation")
    savefig(path, fig=fig)
    return True


def make_figures(
    metrics: pd.DataFrame,
    matrix_store: dict[str, np.ndarray],
    figure_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []
    best = best_primary(metrics)
    if best is not None:
        behavior_key = str(best["behavior_matrix_key"])
        activation_key = str(best["activation_matrix_key"])
        if behavior_key in matrix_store:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            distance_heatmap(
                matrix_store[behavior_key],
                ax=ax,
                title="Behavior centroid distance heatmap (primary high-dimensional)",
            )
            out = figure_dir / "behavior_centroid_distance_heatmap.png"
            savefig(out, fig=fig)
            figures.append({"path": rel(out), "description": "Behavior centroid distance heatmap"})
        if activation_key in matrix_store:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            distance_heatmap(
                matrix_store[activation_key],
                ax=ax,
                title=(
                    "Activation centroid distance heatmap "
                    f"(layer {int(best['activation_layer'])}, {best['activation_representation']})"
                ),
            )
            out = figure_dir / "activation_centroid_distance_heatmap_best.png"
            savefig(out, fig=fig)
            figures.append({"path": rel(out), "description": "Best activation centroid distance heatmap"})

    out = figure_dir / "distance_correlation_by_layer.png"
    if plot_line_by_layer(metrics, out):
        figures.append({"path": rel(out), "description": "Primary distance correlation by layer"})

    out = figure_dir / "distance_correlation_by_representation.png"
    if plot_by_representation(metrics, out):
        figures.append({"path": rel(out), "description": "Primary distance correlation by representation"})

    out = figure_dir / "strict_pre_action_distance_correlation_by_layer.png"
    if plot_line_by_layer(metrics, out, subset_filter="strict_before"):
        figures.append({"path": rel(out), "description": "Strict pre-action primary distance correlation by layer"})
    return figures


def write_report(
    path: Path,
    *,
    behavior_centroids: Path,
    activation_centroids_2d: Path,
    activation_centroids_highdim: Path,
    metrics_path: Path,
    matrices_path: Path,
    metrics: pd.DataFrame,
    skipped: list[str],
    figures: list[dict[str, str]],
    n_permutations: int,
    n_bootstrap: int,
) -> None:
    primary = metrics.loc[metrics["is_primary"]].copy() if not metrics.empty else pd.DataFrame()
    best = best_primary(metrics)
    lines = [
        "# Geometry Alignment",
        "",
        "Step 6 compares behavior and activation conceptual geometry using action centroid distance matrices.",
        "",
        "## Required Caveat",
        "",
        "There are only four action concepts, so each distance-matrix comparison has only six pairwise distances. These correlations are useful paper-style geometry diagnostics, but they should not be overinterpreted as high-powered statistical evidence.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Behavior centroids: `{rel(behavior_centroids)}`",
        f"- Activation 2D centroids: `{rel(activation_centroids_2d)}`",
        f"- Activation high-dimensional centroids: `{rel(activation_centroids_highdim)}`",
        f"- Metrics CSV: `{rel(metrics_path)}`",
        f"- Distance matrices NPZ: `{rel(matrices_path)}`",
        "",
        "## Methods",
        "",
        "- Primary behavior distance: high-dimensional behavior centroid Euclidean distance.",
        "- Secondary behavior distance: 2D behavior UMAP centroid Euclidean distance.",
        "- Primary activation distance: high-dimensional activation centroid cosine distance.",
        "- Secondary activation distances: high-dimensional Euclidean and 2D UMAP Euclidean.",
        "- Matrix comparison uses the six upper-triangle action-pair distances.",
        f"- Permutation test requested permutations: `{n_permutations}`; with four actions, exact 24 action-label permutations are used when possible.",
        f"- Bootstrap requested resamples: `{n_bootstrap}`.",
        "- Bootstrap CI is over paired action-pair distances because centroid-only inputs do not contain decision_id or trajectory_id membership for decision-level bootstrap.",
        "",
        "## Counts",
        "",
        f"- Metric rows: `{len(metrics)}`",
        f"- Primary metric rows: `{len(primary)}`",
        f"- Skipped blocks: `{len(skipped)}`",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                "## Best Primary Alignment",
                "",
                f"- Split: `{int(best['split_id'])}`",
                f"- Subset: `{best['subset']}`",
                f"- Behavior representation: `{best['behavior_representation']}`",
                f"- Activation layer: `{int(best['activation_layer'])}`",
                f"- Activation representation: `{best['activation_representation']}`",
                f"- Pearson: `{best['pearson']}`",
                f"- Spearman: `{best['spearman']}`",
                f"- Kendall tau: `{best['kendall_tau']}`",
                f"- Spearman permutation p-value: `{best['permutation_p_value']}`",
                f"- Spearman bootstrap 95% CI: `[{best['bootstrap_ci_low']}, {best['bootstrap_ci_high']}]`",
            ]
        )

    if not primary.empty:
        summary = (
            primary.groupby(["activation_representation", "activation_layer"])["spearman"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )
        lines.extend(["", "## Top Mean Primary Spearman by Layer", ""])
        for (representation, layer), value in summary.items():
            lines.append(f"- `{representation}`, layer `{int(layer)}`: `{float(value)}`")

    lines.extend(["", "## Figures", ""])
    if figures:
        for fig in figures:
            lines.append(f"- `{fig['path']}`: {fig['description']}")
    else:
        lines.append("- No figures generated.")

    lines.extend(["", "## Skipped Blocks", ""])
    if skipped:
        for item in skipped[:80]:
            lines.append(f"- {item}")
        if len(skipped) > 80:
            lines.append(f"- ... {len(skipped) - 80} additional skipped blocks omitted.")
    else:
        lines.append("- None.")
    save_markdown(path, "\n".join(lines))


def run(
    *,
    behavior_centroids_path: Path,
    activation_centroids_2d_path: Path,
    activation_centroids_highdim_path: Path,
    metrics_output: Path,
    matrices_output: Path,
    report_output: Path,
    figure_dir: Path,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    require_existing_file(behavior_centroids_path, "behavior centroids")
    require_existing_file(activation_centroids_2d_path, "activation 2D centroids")
    require_existing_file(activation_centroids_highdim_path, "activation high-dimensional centroids")

    behavior = read_parquet_checked(behavior_centroids_path)
    activation_2d = read_parquet_checked(activation_centroids_2d_path)
    validate_centroid_tables(behavior, activation_2d)
    with np.load(activation_centroids_highdim_path) as activation_highdim:
        metrics, matrix_store, skipped = compute_metrics(
            behavior,
            activation_2d,
            activation_highdim,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    if metrics.empty:
        raise SystemExit("No compatible behavior/activation centroid blocks were found for alignment.")

    ensure_dir(metrics_output.parent)
    metrics.to_csv(metrics_output, index=False)
    ensure_dir(matrices_output.parent)
    np.savez_compressed(matrices_output, **matrix_store)
    figures = make_figures(metrics, matrix_store, figure_dir)
    write_report(
        report_output,
        behavior_centroids=behavior_centroids_path,
        activation_centroids_2d=activation_centroids_2d_path,
        activation_centroids_highdim=activation_centroids_highdim_path,
        metrics_path=metrics_output,
        matrices_path=matrices_output,
        metrics=metrics,
        skipped=skipped,
        figures=figures,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior_centroids",
        default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_centroids.parquet",
    )
    parser.add_argument(
        "--activation_centroids_2d",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_2d.parquet",
    )
    parser.add_argument(
        "--activation_centroids_highdim",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_highdim.npz",
    )
    parser.add_argument(
        "--metrics_output",
        default=DEFAULT_OUTPUT_DIR / "geometry_alignment/distance_matrix_metrics.csv",
    )
    parser.add_argument(
        "--matrices_output",
        default=DEFAULT_OUTPUT_DIR / "geometry_alignment/distance_matrices.npz",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_OUTPUT_DIR / "geometry_alignment/06_geometry_alignment_report.md",
    )
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "geometry_alignment")
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    metrics = run(
        behavior_centroids_path=resolve_repo_path(args.behavior_centroids),
        activation_centroids_2d_path=resolve_repo_path(args.activation_centroids_2d),
        activation_centroids_highdim_path=resolve_repo_path(args.activation_centroids_highdim),
        metrics_output=resolve_repo_path(args.metrics_output),
        matrices_output=resolve_repo_path(args.matrices_output),
        report_output=resolve_repo_path(args.report),
        figure_dir=resolve_repo_path(args.figure_dir),
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(f"Wrote {rel(args.metrics_output)} with {len(metrics)} rows")
    print(f"Wrote {rel(args.matrices_output)}")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
