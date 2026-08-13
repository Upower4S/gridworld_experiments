#!/usr/bin/env python3
"""Diagnostic high-commitment train action-extreme activation UMAPs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.conceptual_belief_space.extreme_activation_plots import (
    compute_metrics_for_config,
    coordinate_frame,
    dataframe_to_markdown,
    direct_answer,
    fit_umap,
    load_feature_memmap,
    load_splits,
    parse_int_list,
    parse_str_list,
    plot_grid_by_representation,
    plot_individual_umap,
    plot_layer_representation_grid,
    preprocess_selected_features,
    rank_metric_table,
    resolve_feature_row_indices,
    select_extreme_rows,
    train_row_ids_for_split,
)
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_map", default=DEFAULT_OUTPUT_DIR / "io_map.json")
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--extremes", default=DEFAULT_OUTPUT_DIR / "extreme_examples/action_extreme_examples.parquet")
    parser.add_argument("--feature_dir", default=Path("outputs/activation/features"))
    parser.add_argument("--activation_metadata", default=Path("outputs/activation/activation_feature_metadata.parquet"))
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR / "activation_extremes")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "activation_extremes")
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
    parser.add_argument("--umap_backend", choices=["sklearn", "cpu", "umap", "cuml"], default="sklearn")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--fast_nondeterministic_umap", action=argparse.BooleanOptionalAction, default=False)
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


def check_output_paths(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        text = "\n".join(f"- {rel(path)}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing diagnostic outputs without --force:\n"
            f"{text}\n"
            "Pass --force to regenerate these activation-extreme diagnostics."
        )


def action_count_text(selected: pd.DataFrame) -> str:
    counts = selected["extreme_action"].value_counts().reindex(ACTIONS, fill_value=0).astype(int)
    return ", ".join(f"{action}={int(counts[action])}" for action in ACTIONS)


def selected_output_name(split_id: int, subset: str) -> str:
    safe_subset = str(subset).replace("/", "_")
    return f"selected_extreme_rows_split{int(split_id)}_{safe_subset}.parquet"


def individual_figure_path(figure_dir: Path, *, layer: int, representation: str, split_id: int, subset: str) -> Path:
    safe_subset = str(subset).replace("/", "_")
    return figure_dir / f"layer_{int(layer):02d}_{representation}_extreme_umap_split{int(split_id)}_{safe_subset}.png"


def grid_figure_path(figure_dir: Path, *, representation: str, split_id: int, subset: str) -> Path:
    safe_subset = str(subset).replace("/", "_")
    return figure_dir / f"extreme_umap_grid_{representation}_split{int(split_id)}_{safe_subset}.png"


def layer_rep_grid_path(figure_dir: Path, *, split_id: int, subset: str) -> Path:
    safe_subset = str(subset).replace("/", "_")
    return figure_dir / f"extreme_umap_layer_representation_grid_split{int(split_id)}_{safe_subset}.png"


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
    ranked = rank_metric_table(metrics)
    verdict, reason = direct_answer(metrics)

    cols = [
        "rank",
        "layer",
        "representation",
        "n_rows",
        "silhouette_score_highdim_by_extreme_action",
        "nearest_centroid_accuracy_highdim",
        "silhouette_score_umap_by_extreme_action",
        "nearest_centroid_accuracy_umap",
        "between_within_ratio_highdim",
    ]
    highdim_sil = ranked.sort_values(
        "silhouette_score_highdim_by_extreme_action",
        ascending=False,
        na_position="last",
    )[cols]
    highdim_acc = ranked.sort_values(
        "nearest_centroid_accuracy_highdim",
        ascending=False,
        na_position="last",
    )[cols]
    umap_sil = ranked.sort_values(
        "silhouette_score_umap_by_extreme_action",
        ascending=False,
        na_position="last",
    )[cols]

    best = ranked.iloc[0] if not ranked.empty else None
    if best is not None:
        recommendation = (
            f"Layer `{int(best['layer'])}` with representation `{best['representation']}` ranks best by the "
            "combined high-dimensional and UMAP separation criteria."
        )
    else:
        recommendation = "No recommendation because no metrics were produced."

    count_lines = [
        f"- {action}: {selected_summary['counts_by_extreme_action'].get(action, 0)}"
        for action in ACTIONS
    ]
    figure_lines = [f"- `{rel(path)}`" for path in figures[:80]]
    if len(figures) > 80:
        figure_lines.append(f"- ... {len(figures) - 80} more figures")

    lines = [
        "# Extreme Activation UMAP Diagnostic",
        "",
        "## Goal And Caveat",
        "",
        "This is a train-only positive-control visualization. It tests whether the easiest/highest-confidence "
        "action-extreme prefixes separate in activation space. It is not held-out evidence and does not replace "
        "the main conceptual belief space pipeline.",
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
        f"- rows before train filter: `{selected_summary['rows_before_train_filter']}`",
        f"- rows removed by train filter: `{selected_summary['rows_removed_by_train_filter']}`",
        f"- duplicates seen: `{selected_summary['duplicates_seen']}`",
        f"- duplicates removed: `{selected_summary['duplicates_removed']}`",
        f"- final rows: `{selected_summary['n_rows']}`",
        "",
        "Rows per extreme action:",
        "",
        *count_lines,
        "",
        "## Representations",
        "",
        "- `raw` = absolute CoT-token hidden state.",
        "- `delta_pre` = `h_k - h_pre`, reasoning-induced movement from the pre-reasoning baseline.",
        "- `unit_delta_pre` = direction-only normalized `delta_pre`.",
        "",
        "## UMAP Settings",
        "",
        f"- backend: `{args.umap_backend}`",
        f"- n_neighbors: `{args.n_neighbors}`",
        f"- min_dist: `{args.min_dist}`",
        f"- metric: `{args.metric}`",
        f"- random_state: `{None if args.fast_nondeterministic_umap else args.random_state}`",
        f"- n_jobs: `{args.n_jobs}`",
        f"- scaler fit on: `{args.fit_scaler_on}`",
        f"- L2 normalize after scaling: `{args.l2_normalize}`",
        "",
        "## Top 10 By High-Dimensional Silhouette",
        "",
        dataframe_to_markdown(highdim_sil, max_rows=10),
        "",
        "## Top 10 By High-Dimensional Nearest-Centroid Accuracy",
        "",
        dataframe_to_markdown(highdim_acc, max_rows=10),
        "",
        "## Top 10 By UMAP Silhouette",
        "",
        dataframe_to_markdown(umap_sil, max_rows=10),
        "",
        "## Best Layer / Representation",
        "",
        recommendation,
        "",
        "## Direct Answer",
        "",
        f"- Do high-commitment train extremes show four clear action branches? `{verdict}`.",
        f"- Basis: {reason}",
        "",
        "## Interpretation Rules",
        "",
        "- If `raw` separates better than `delta_pre`, action structure may already be present in absolute prompt/trajectory state.",
        "- If `delta_pre` separates better than `raw`, reasoning-induced movement is action-specific.",
        "- If `unit_delta_pre` separates better than `delta_pre`, action signal is mainly directional rather than norm-based.",
        "",
        "## Figures",
        "",
        *figure_lines,
    ]
    save_markdown(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.n_neighbors < 2:
        raise SystemExit("--n_neighbors must be >= 2")
    if args.max_rows_per_action is not None and args.max_rows_per_action < 1:
        raise SystemExit("--max_rows_per_action must be >= 1 when provided")
    if args.top_k_per_action is not None and args.top_k_per_action < 1:
        raise SystemExit("--top_k_per_action must be >= 1 when provided")

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
    selected_path = output_dir / selected_output_name(args.split_id, args.subset)
    coords_path = output_dir / "extreme_umap_coords.parquet"
    metrics_path = output_dir / "extreme_umap_metrics.csv"
    best_configs_path = output_dir / "extreme_umap_best_configs.json"
    report_path = output_dir / "12_extreme_activation_umap_report.md"
    check_output_paths([selected_path, coords_path, metrics_path, best_configs_path, report_path], force=args.force)

    if io_map_path.exists():
        io_map = read_json(io_map_path)
        if isinstance(io_map, dict) and not feature_dir.exists() and io_map.get("activation_feature_dir"):
            feature_dir = resolve_repo_path(str(io_map["activation_feature_dir"]))

    beliefs = read_parquet_checked(beliefs_path)
    splits = load_splits(splits_path)
    extremes = read_parquet_checked(extremes_path)
    activation_metadata = (
        read_parquet_checked(activation_metadata_path)
        if activation_metadata_path.exists()
        else None
    )
    layers = parse_int_list(args.layers)
    representations = parse_str_list(args.representations)
    if not layers:
        raise SystemExit("No layers requested")
    if not representations:
        raise SystemExit("No representations requested")

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
    print(f"Rows per extreme action: {action_count_text(selected)}")

    train_row_ids = train_row_ids_for_split(beliefs, splits, args.split_id)
    train_row_frame = pd.DataFrame({"row_id": train_row_ids})

    coord_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    effective_random_state = None if args.fast_nondeterministic_umap else int(args.random_state)

    for representation in representations:
        for layer in layers:
            print(f"Fitting extreme UMAP for layer {layer}, representation {representation}...")
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
            coords = fit_umap(
                highdim_x,
                backend=args.umap_backend,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                metric=args.metric,
                random_state=effective_random_state,
                n_jobs=args.n_jobs,
            )
            coord_df = coordinate_frame(
                selected=selected,
                coords=coords,
                split_id=args.split_id,
                subset=args.subset,
                layer=layer,
                representation=representation,
            )
            coord_parts.append(coord_df)
            metrics = compute_metrics_for_config(
                layer=layer,
                representation=representation,
                selected=selected,
                highdim_x=highdim_x,
                coords=coords,
                l2_normalize=bool(args.l2_normalize),
            )
            metrics.update(
                {
                    "split_id": int(args.split_id),
                    "subset": str(args.subset),
                    "fit_scaler_on": scaler_summary["fit_scaler_on"],
                    "n_scaler_rows": int(scaler_summary["n_scaler_rows"]),
                    "l2_normalize": bool(args.l2_normalize),
                    "umap_backend": args.umap_backend,
                    "umap_metric": args.metric,
                    "umap_random_state": effective_random_state,
                    "umap_n_jobs": int(args.n_jobs),
                }
            )
            metric_rows.append(metrics)

            if args.save_individual_figures:
                counts = action_count_text(selected)
                title = (
                    f"Extreme activation UMAP | L{int(layer):02d} {representation} | "
                    f"split {args.split_id} {args.subset} | n={len(selected)} | {counts}"
                )
                fig_path = individual_figure_path(
                    figure_dir,
                    layer=layer,
                    representation=representation,
                    split_id=args.split_id,
                    subset=args.subset,
                )
                figures.append(plot_individual_umap(coord_df, path=fig_path, title=title))

    coords_all = pd.concat(coord_parts, ignore_index=True) if coord_parts else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    coords_all.to_parquet(coords_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {rel(coords_path)} with {len(coords_all)} rows")
    print(f"Wrote {rel(metrics_path)} with {len(metrics_df)} rows")

    ranked = rank_metric_table(metrics_df)
    best_payload = {
        "ranking_criteria": [
            "silhouette_score_highdim_by_extreme_action",
            "nearest_centroid_accuracy_highdim",
            "silhouette_score_umap_by_extreme_action",
            "nearest_centroid_accuracy_umap",
            "between_within_ratio_highdim",
        ],
        "split_id": int(args.split_id),
        "subset": str(args.subset),
        "top_configs": ranked.head(20).replace({np.nan: None}).to_dict("records"),
    }
    write_json(best_configs_path, best_payload)
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

