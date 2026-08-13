#!/usr/bin/env python3
"""Diagnostic high-commitment train action-extreme activation PCAs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.conceptual_belief_space.extreme_activation_plots import (
    compute_metrics_for_config,
    dataframe_to_markdown,
    load_feature_memmap,
    load_splits,
    parse_int_list,
    parse_str_list,
    preprocess_selected_features,
    rank_metric_table,
    resolve_feature_row_indices,
    select_extreme_rows,
    train_row_ids_for_split,
)
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown, write_json
from src.conceptual_belief_space.plotting import action_color_map, savefig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_map", default=DEFAULT_OUTPUT_DIR / "io_map.json")
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--extremes", default=DEFAULT_OUTPUT_DIR / "extreme_examples/action_extreme_examples.parquet")
    parser.add_argument("--feature_dir", default=Path("outputs/activation/features"))
    parser.add_argument("--activation_metadata", default=Path("outputs/activation/activation_feature_metadata.parquet"))
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR / "activation_extremes/pca")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "activation_extremes/pca")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--subset", default="high_commitment")
    parser.add_argument("--layers", default="0,2,4,6,8,10,12,14,16,18,20,22")
    parser.add_argument("--representations", default="raw,delta_pre,unit_delta_pre")
    parser.add_argument("--top_k_per_action", type=int, default=None)
    parser.add_argument("--max_rows_per_action", type=int, default=None)
    parser.add_argument(
        "--deduplicate_row_ids",
        choices=["keep_highest_q_action", "none"],
        default="keep_highest_q_action",
    )
    parser.add_argument("--fit_scaler_on", choices=["train", "selected", "none"], default="train")
    parser.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save_individual_figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_grid_figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scaler_chunk_size", type=int, default=2048)
    return parser.parse_args()


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


def safe_subset_name(subset: str) -> str:
    return str(subset).replace("/", "_")


def check_output_paths(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        text = "\n".join(f"- {rel(path)}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing PCA diagnostic outputs without --force:\n"
            f"{text}\n"
            "Pass --force to regenerate these activation-extreme PCA diagnostics."
        )


def action_count_text(selected: pd.DataFrame) -> str:
    counts = selected["extreme_action"].value_counts().reindex(ACTIONS, fill_value=0).astype(int)
    return ", ".join(f"{action}={int(counts[action])}" for action in ACTIONS)


def fit_pca(x: np.ndarray, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    if x.ndim != 2:
        raise ValueError("PCA input must be 2D")
    if len(x) < 3:
        raise ValueError("Need at least three selected rows for PCA")
    model = PCA(n_components=2, random_state=random_state)
    coords = model.fit_transform(x)
    return np.asarray(coords, dtype=np.float32), np.asarray(model.explained_variance_ratio_, dtype=np.float32)


def coordinate_frame(
    *,
    selected: pd.DataFrame,
    coords: np.ndarray,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    explained_variance_ratio: np.ndarray,
) -> pd.DataFrame:
    cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "prefix_token_index",
        "progress",
        "extreme_action",
        "agent_action",
        "top_action",
        "q_action",
        "commitment_to_final",
        "entropy_commitment",
    ]
    out = selected[[col for col in cols if col in selected.columns]].copy()
    out.insert(0, "representation", representation)
    out.insert(0, "layer", int(layer))
    out.insert(0, "subset", str(subset))
    out.insert(0, "split_id", int(split_id))
    out["pca1"] = coords[:, 0].astype(np.float32)
    out["pca2"] = coords[:, 1].astype(np.float32)
    out["pca_explained_variance_ratio_1"] = float(explained_variance_ratio[0])
    out["pca_explained_variance_ratio_2"] = float(explained_variance_ratio[1])
    return out


def pca_metrics(
    *,
    layer: int,
    representation: str,
    selected: pd.DataFrame,
    highdim_x: np.ndarray,
    coords: np.ndarray,
    l2_normalize: bool,
    explained_variance_ratio: np.ndarray,
) -> dict[str, Any]:
    raw = compute_metrics_for_config(
        layer=layer,
        representation=representation,
        selected=selected,
        highdim_x=highdim_x,
        coords=coords,
        l2_normalize=l2_normalize,
    )
    rename = {
        "silhouette_score_umap_by_extreme_action": "silhouette_score_pca_by_extreme_action",
        "nearest_centroid_accuracy_umap": "nearest_centroid_accuracy_pca",
        "mean_nearest_centroid_margin_umap": "mean_nearest_centroid_margin_pca",
        "between_centroid_distance_mean_umap": "between_centroid_distance_mean_pca",
        "within_action_distance_mean_umap": "within_action_distance_mean_pca",
        "between_within_ratio_umap": "between_within_ratio_pca",
        "davies_bouldin_score_umap": "davies_bouldin_score_pca",
        "calinski_harabasz_score_umap": "calinski_harabasz_score_pca",
    }
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[rename.get(key, key)] = value
    out["pca_explained_variance_ratio_1"] = float(explained_variance_ratio[0])
    out["pca_explained_variance_ratio_2"] = float(explained_variance_ratio[1])
    out["pca_explained_variance_ratio_sum"] = float(explained_variance_ratio.sum())
    return out


def plot_individual_pca(coords: pd.DataFrame, *, path: str | Path, title: str, s: float = 16) -> Path:
    import matplotlib.pyplot as plt

    colors = action_color_map()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for action in ACTIONS:
        part = coords[coords["extreme_action"].astype(str).eq(action)]
        if part.empty:
            continue
        ax.scatter(
            part["pca1"],
            part["pca2"],
            s=s,
            alpha=0.78,
            color=colors[action],
            label=action,
            edgecolors="none",
        )
        centroid = part[["pca1", "pca2"]].mean()
        ax.scatter(
            [centroid["pca1"]],
            [centroid["pca2"]],
            s=150,
            marker="X",
            color=colors[action],
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(float(centroid["pca1"]), float(centroid["pca2"]), f" {action}", fontsize=9, weight="bold", va="center")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.18, linewidth=0.5)
    return savefig(path, dpi=200, fig=fig)


def plot_grid_by_representation(
    coords_all: pd.DataFrame,
    *,
    representation: str,
    layers: Sequence[int],
    path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    layers = list(layers)
    ncols = min(4, max(1, len(layers)))
    nrows = int(math.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    colors = action_color_map()
    for ax in axes.ravel():
        ax.axis("off")
    for i, layer in enumerate(layers):
        ax = axes.ravel()[i]
        ax.axis("on")
        part = coords_all[
            coords_all["representation"].astype(str).eq(representation)
            & coords_all["layer"].astype(int).eq(int(layer))
        ]
        for action in ACTIONS:
            action_part = part[part["extreme_action"].astype(str).eq(action)]
            if not action_part.empty:
                ax.scatter(
                    action_part["pca1"],
                    action_part["pca2"],
                    s=8,
                    alpha=0.75,
                    color=colors[action],
                    label=action if i == 0 else None,
                    edgecolors="none",
                )
        ax.set_title(f"layer {int(layer):02d}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=len(ACTIONS), loc="upper center")
    fig.suptitle(f"Extreme activation PCA grid: {representation}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return savefig(path, dpi=200, fig=fig)


def plot_layer_representation_grid(
    coords_all: pd.DataFrame,
    *,
    layers: Sequence[int],
    representations: Sequence[str],
    path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    layers = list(layers)
    representations = list(representations)
    fig, axes = plt.subplots(
        len(layers),
        len(representations),
        figsize=(4.0 * len(representations), 2.9 * len(layers)),
        squeeze=False,
    )
    colors = action_color_map()
    for row, layer in enumerate(layers):
        for col, representation in enumerate(representations):
            ax = axes[row, col]
            part = coords_all[
                coords_all["representation"].astype(str).eq(representation)
                & coords_all["layer"].astype(int).eq(int(layer))
            ]
            for action in ACTIONS:
                action_part = part[part["extreme_action"].astype(str).eq(action)]
                if not action_part.empty:
                    ax.scatter(
                        action_part["pca1"],
                        action_part["pca2"],
                        s=6,
                        alpha=0.72,
                        color=colors[action],
                        label=action if row == 0 and col == 0 else None,
                        edgecolors="none",
                    )
            if row == 0:
                ax.set_title(str(representation), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{int(layer):02d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=len(ACTIONS), loc="upper center")
    fig.suptitle("Extreme activation PCAs by layer and representation", y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return savefig(path, dpi=200, fig=fig)


def individual_figure_path(figure_dir: Path, *, layer: int, representation: str, split_id: int, subset: str) -> Path:
    return figure_dir / f"layer_{int(layer):02d}_{representation}_extreme_pca_split{int(split_id)}_{safe_subset_name(subset)}.png"


def grid_figure_path(figure_dir: Path, *, representation: str, split_id: int, subset: str) -> Path:
    return figure_dir / f"extreme_pca_grid_{representation}_split{int(split_id)}_{safe_subset_name(subset)}.png"


def layer_rep_grid_path(figure_dir: Path, *, split_id: int, subset: str) -> Path:
    return figure_dir / f"extreme_pca_layer_representation_grid_split{int(split_id)}_{safe_subset_name(subset)}.png"


def rank_pca_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    criteria = [
        "silhouette_score_highdim_by_extreme_action",
        "nearest_centroid_accuracy_highdim",
        "silhouette_score_pca_by_extreme_action",
        "nearest_centroid_accuracy_pca",
        "between_within_ratio_highdim",
    ]
    for col in criteria:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked = ranked.sort_values(criteria, ascending=[False, False, False, False, False], na_position="last")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    return ranked


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    selected_path: Path,
    coords_path: Path,
    metrics_path: Path,
    best_configs_path: Path,
    selected_summary: dict[str, Any],
    metrics: pd.DataFrame,
    figures: list[Path],
) -> None:
    ranked = rank_pca_metrics(metrics)
    cols = [
        "rank",
        "layer",
        "representation",
        "n_rows",
        "silhouette_score_highdim_by_extreme_action",
        "nearest_centroid_accuracy_highdim",
        "silhouette_score_pca_by_extreme_action",
        "nearest_centroid_accuracy_pca",
        "pca_explained_variance_ratio_sum",
    ]
    top_highdim = ranked.sort_values(
        "silhouette_score_highdim_by_extreme_action",
        ascending=False,
        na_position="last",
    )[cols]
    top_pca = ranked.sort_values(
        "silhouette_score_pca_by_extreme_action",
        ascending=False,
        na_position="last",
    )[cols]
    best = ranked.iloc[0] if not ranked.empty else None
    if best is None:
        verdict = "no"
        basis = "No metric rows were produced."
        recommendation = "No recommendation."
    else:
        sil = float(best.get("silhouette_score_highdim_by_extreme_action", float("nan")))
        acc = float(best.get("nearest_centroid_accuracy_highdim", float("nan")))
        if (math.isfinite(sil) and sil >= 0.35) or (math.isfinite(acc) and acc >= 0.85):
            verdict = "yes"
        elif (math.isfinite(sil) and sil >= 0.15) or (math.isfinite(acc) and acc >= 0.60):
            verdict = "weakly"
        else:
            verdict = "no"
        basis = (
            f"Best config is layer {int(best['layer'])} / {best['representation']} with "
            f"high-dimensional silhouette={sil:.4f}, nearest-centroid accuracy={acc:.4f}, "
            f"and PCA silhouette={float(best.get('silhouette_score_pca_by_extreme_action', float('nan'))):.4f}."
        )
        recommendation = f"Best ranked PCA diagnostic config: layer `{int(best['layer'])}` / `{best['representation']}`."

    figure_lines = [f"- `{rel(path)}`" for path in figures[:80]]
    lines = [
        "# Extreme Activation PCA Diagnostic",
        "",
        "## Goal And Caveat",
        "",
        "This repeats the high-commitment train action-extreme diagnostic with 2D PCA instead of UMAP. "
        "It is a train-only positive control and not held-out evidence.",
        "",
        "## Inputs And Outputs",
        "",
        f"- Selected rows: `{rel(selected_path)}`",
        f"- Coordinates: `{rel(coords_path)}`",
        f"- Metrics: `{rel(metrics_path)}`",
        f"- Best configs: `{rel(best_configs_path)}`",
        "",
        "## Row Selection",
        "",
        f"- split_id: `{args.split_id}`",
        f"- subset: `{args.subset}`",
        f"- final rows: `{selected_summary['n_rows']}`",
        f"- duplicates removed: `{selected_summary['duplicates_removed']}`",
        "",
        "Rows per extreme action:",
        "",
        *[f"- {action}: {selected_summary['counts_by_extreme_action'].get(action, 0)}" for action in ACTIONS],
        "",
        "## PCA Settings",
        "",
        f"- scaler fit on: `{args.fit_scaler_on}`",
        f"- L2 normalize after scaling: `{args.l2_normalize}`",
        f"- random_state: `{args.random_state}`",
        "",
        "## Top 10 By High-Dimensional Silhouette",
        "",
        dataframe_to_markdown(top_highdim, max_rows=10),
        "",
        "## Top 10 By PCA Silhouette",
        "",
        dataframe_to_markdown(top_pca, max_rows=10),
        "",
        "## Best Layer / Representation",
        "",
        recommendation,
        "",
        "## Direct Answer",
        "",
        f"- Do high-commitment train extremes show four clear action branches under PCA? `{verdict}`.",
        f"- Basis: {basis}",
        "",
        "## Figures",
        "",
        *figure_lines,
    ]
    save_markdown(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    io_map_path = resolve_repo_path(args.io_map)
    beliefs_path = resolve_repo_path(args.beliefs)
    splits_path = resolve_repo_path(args.splits)
    extremes_path = resolve_repo_path(args.extremes)
    feature_dir = resolve_repo_path(args.feature_dir)
    activation_metadata_path = resolve_repo_path(args.activation_metadata)
    output_dir = resolve_repo_path(args.output_dir)
    figure_dir = resolve_repo_path(args.figure_dir)
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    selected_path = output_dir / f"selected_extreme_rows_split{int(args.split_id)}_{safe_subset_name(args.subset)}.parquet"
    coords_path = output_dir / "extreme_pca_coords.parquet"
    metrics_path = output_dir / "extreme_pca_metrics.csv"
    best_configs_path = output_dir / "extreme_pca_best_configs.json"
    report_path = output_dir / "12_extreme_activation_pca_report.md"
    check_output_paths([selected_path, coords_path, metrics_path, best_configs_path, report_path], force=args.force)

    if io_map_path.exists():
        io_map = read_json(io_map_path)
        if isinstance(io_map, dict) and not feature_dir.exists() and io_map.get("activation_feature_dir"):
            feature_dir = resolve_repo_path(str(io_map["activation_feature_dir"]))

    beliefs = read_parquet_checked(beliefs_path)
    splits = load_splits(splits_path)
    extremes = read_parquet_checked(extremes_path)
    activation_metadata = read_parquet_checked(activation_metadata_path) if activation_metadata_path.exists() else None
    layers = parse_int_list(args.layers)
    representations = parse_str_list(args.representations)

    selected, selected_summary = select_extreme_rows(
        beliefs=beliefs,
        splits=splits,
        extremes=extremes,
        split_id=args.split_id,
        subset=args.subset,
        top_k_per_action=args.top_k_per_action,
        max_rows_per_action=args.max_rows_per_action,
        deduplicate_row_ids=args.deduplicate_row_ids,
        random_state=args.random_state,
    )
    selected.to_parquet(selected_path, index=False)
    print(f"Wrote {rel(selected_path)} with {len(selected)} selected rows")
    print(f"Rows per extreme action: {', '.join(f'{a}={selected_summary['counts_by_extreme_action'].get(a, 0)}' for a in ACTIONS)}")

    train_row_frame = pd.DataFrame({"row_id": train_row_ids_for_split(beliefs, splits, args.split_id)})
    coord_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    figures: list[Path] = []

    for representation in representations:
        for layer in layers:
            print(f"Fitting extreme PCA for layer {layer}, representation {representation}...")
            features = load_feature_memmap(feature_dir, layer, representation)
            selected_indices = resolve_feature_row_indices(
                selected,
                feature_rows=features.shape[0],
                activation_metadata=activation_metadata,
            )
            train_indices = resolve_feature_row_indices(
                train_row_frame,
                feature_rows=features.shape[0],
                activation_metadata=activation_metadata,
            )
            highdim_x, scaler_summary = preprocess_selected_features(
                features,
                selected_indices,
                train_indices,
                fit_scaler_on=args.fit_scaler_on,
                l2_normalize=bool(args.l2_normalize),
                chunk_size=int(args.scaler_chunk_size),
            )
            coords, evr = fit_pca(highdim_x, random_state=int(args.random_state))
            coord_df = coordinate_frame(
                selected=selected,
                coords=coords,
                split_id=args.split_id,
                subset=args.subset,
                layer=layer,
                representation=representation,
                explained_variance_ratio=evr,
            )
            coord_parts.append(coord_df)
            metrics = pca_metrics(
                layer=layer,
                representation=representation,
                selected=selected,
                highdim_x=highdim_x,
                coords=coords,
                l2_normalize=bool(args.l2_normalize),
                explained_variance_ratio=evr,
            )
            metrics.update(
                {
                    "split_id": int(args.split_id),
                    "subset": str(args.subset),
                    "fit_scaler_on": scaler_summary["fit_scaler_on"],
                    "n_scaler_rows": int(scaler_summary["n_scaler_rows"]),
                    "l2_normalize": bool(args.l2_normalize),
                    "method": "pca",
                }
            )
            metric_rows.append(metrics)
            if args.save_individual_figures:
                title = (
                    f"Extreme activation PCA | L{int(layer):02d} {representation} | "
                    f"split {args.split_id} {args.subset} | n={len(selected)} | {action_count_text(selected)}"
                )
                fig_path = individual_figure_path(
                    figure_dir,
                    layer=layer,
                    representation=representation,
                    split_id=args.split_id,
                    subset=args.subset,
                )
                figures.append(plot_individual_pca(coord_df, path=fig_path, title=title))

    coords_all = pd.concat(coord_parts, ignore_index=True) if coord_parts else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    coords_all.to_parquet(coords_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {rel(coords_path)} with {len(coords_all)} rows")
    print(f"Wrote {rel(metrics_path)} with {len(metrics_df)} rows")

    ranked = rank_pca_metrics(metrics_df)
    write_json(
        best_configs_path,
        {
            "ranking_criteria": [
                "silhouette_score_highdim_by_extreme_action",
                "nearest_centroid_accuracy_highdim",
                "silhouette_score_pca_by_extreme_action",
                "nearest_centroid_accuracy_pca",
                "between_within_ratio_highdim",
            ],
            "split_id": int(args.split_id),
            "subset": str(args.subset),
            "top_configs": ranked.head(20).replace({np.nan: None}).to_dict("records"),
        },
    )
    print(f"Wrote {rel(best_configs_path)}")

    if args.save_grid_figures and not coords_all.empty:
        for representation in representations:
            fig_path = grid_figure_path(
                figure_dir,
                representation=representation,
                split_id=args.split_id,
                subset=args.subset,
            )
            figures.append(
                plot_grid_by_representation(
                    coords_all,
                    representation=representation,
                    layers=layers,
                    path=fig_path,
                )
            )
        if len(representations) > 1:
            fig_path = layer_rep_grid_path(figure_dir, split_id=args.split_id, subset=args.subset)
            figures.append(
                plot_layer_representation_grid(
                    coords_all,
                    layers=layers,
                    representations=representations,
                    path=fig_path,
                )
            )

    write_report(
        report_path,
        args=args,
        selected_path=selected_path,
        coords_path=coords_path,
        metrics_path=metrics_path,
        best_configs_path=best_configs_path,
        selected_summary=selected_summary,
        metrics=metrics_df,
        figures=figures,
    )
    print(f"Wrote {rel(report_path)}")
    print(f"Wrote {len(figures)} figures under {rel(figure_dir)}")


if __name__ == "__main__":
    main()

