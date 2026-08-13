#!/usr/bin/env python3
"""Step 5: fit activation conceptual manifolds anchored by the same train extremes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.activation_features import (  # noqa: E402
    activation_feature_path,
    load_activation_memmap,
    parse_layers_arg,
    parse_representations_arg,
    standardize_features_for_split,
)
from src.conceptual_belief_space.centroids import (  # noqa: E402
    compute_action_centroids_2d,
    compute_highdim_action_centroids,
)
from src.conceptual_belief_space.config import DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.manifold import (  # noqa: E402
    activation_model_path,
    fit_debug_pca_transform,
    fit_umap_transform,
    resolve_umap_backend,
    slugify,
)
from src.conceptual_belief_space.plotting import activation_umap_scatter, savefig  # noqa: E402


META_COLS = [
    "row_id",
    "decision_id",
    "trajectory_id",
    "step_id",
    "prefix_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    "top_action",
    "commitment_to_final",
    "entropy_commitment",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "buffered_before_any_action_mention",
    "buffered_before_final_action_mention",
]


def parse_subsets_arg(subsets_arg: str, available_subsets: list[str]) -> list[str]:
    """Parse `all` or a comma-separated subset list against available extremes."""
    available = sorted(set(map(str, available_subsets)))
    if subsets_arg.strip().lower() == "all":
        return available
    requested = [part.strip() for part in subsets_arg.split(",") if part.strip()]
    if not requested:
        raise ValueError("at least one subset is required")
    missing = [subset for subset in requested if subset not in set(available)]
    if missing:
        raise ValueError(f"requested subsets are not available: {missing}; available={available}")
    return requested


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


def load_splits(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of split records")
    return data


def validate_inputs(beliefs: pd.DataFrame, splits: list[dict[str, Any]], extremes: pd.DataFrame) -> None:
    required_beliefs = ["row_id", "decision_id", "trajectory_id", "progress", "agent_action", "top_action"]
    missing = [col for col in required_beliefs if col not in beliefs.columns]
    if missing:
        raise ValueError(f"belief table missing required columns: {missing}")
    if not beliefs["row_id"].is_unique:
        raise ValueError("belief row_id must be unique")
    expected = np.arange(len(beliefs), dtype=np.int64)
    if not np.array_equal(beliefs["row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("belief row_id must match row order")

    required_extremes = ["split_id", "subset", "action", "row_id", "split_role"]
    missing_extremes = [col for col in required_extremes if col not in extremes.columns]
    if missing_extremes:
        raise ValueError(f"extreme table missing required columns: {missing_extremes}")
    if (extremes["split_role"].astype(str) != "train").any():
        raise ValueError("activation manifold must use train-only extremes; found non-train split_role")
    split_ids = {int(split["split_id"]) for split in splits}
    unknown = sorted(set(extremes["split_id"].astype(int)) - split_ids)
    if unknown:
        raise ValueError(f"extremes reference unknown split ids: {unknown}")


def assert_train_extremes(extreme_part: pd.DataFrame, split: dict[str, Any], beliefs: pd.DataFrame) -> None:
    train_row_ids = set(beliefs.iloc[split["train_indices"]]["row_id"].astype(int).tolist())
    test_row_ids = set(beliefs.iloc[split["test_indices"]]["row_id"].astype(int).tolist())
    selected = set(extreme_part["row_id"].astype(int).tolist())
    if selected & test_row_ids:
        raise ValueError(f"test rows found in activation extremes for split {split['split_id']}")
    if not selected <= train_row_ids:
        raise ValueError(f"selected extreme rows outside training rows for split {split['split_id']}")


class ParquetBlockWriter:
    """Append pandas blocks to a parquet file without keeping all blocks in RAM."""

    def __init__(self, path: Path):
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        self.path = path
        ensure_dir(path.parent)
        self.pa = pa
        self.pq = pq
        self.writer = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        table = self.pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def make_coord_block(
    beliefs: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
    coords: np.ndarray,
    method: str,
    l2_normalize: bool,
) -> pd.DataFrame:
    block = beliefs[[col for col in META_COLS if col in beliefs.columns]].copy()
    block.insert(0, "activation_representation", activation_representation)
    block.insert(0, "layer", int(layer))
    block.insert(0, "subset", subset)
    block.insert(0, "split_id", int(split_id))
    block["manifold_method"] = "umap" if method == "umap" else "debug_pca_not_primary"
    block["l2_normalize"] = bool(l2_normalize)
    block["act_umap1"] = coords[:, 0].astype(np.float32)
    block["act_umap2"] = coords[:, 1].astype(np.float32)
    return block


def highdim_key(
    *,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    action: str,
) -> str:
    return (
        f"split_{int(split_id)}__subset_{slugify(subset)}__layer_{int(layer):02d}"
        f"__repr_{slugify(representation)}__action_{slugify(action)}"
    )


def fit_one_activation_block(
    *,
    beliefs: pd.DataFrame,
    scaled_all: np.ndarray,
    split: dict[str, Any],
    extreme_part: pd.DataFrame,
    layer: int,
    activation_representation: str,
    models_dir: Path,
    standardizer: dict[str, np.ndarray],
    l2_normalize: bool,
    random_state: int | None,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    umap_backend: str,
    n_jobs: int | None,
    debug_pca: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    split_id = int(split["split_id"])
    subset = str(extreme_part["subset"].iloc[0])
    assert_train_extremes(extreme_part, split, beliefs)

    extreme_row_ids = extreme_part["row_id"].astype(int).to_numpy()
    train_extreme_x = scaled_all[extreme_row_ids]
    if debug_pca:
        reducer, coords = fit_debug_pca_transform(train_extreme_x, scaled_all)
        method = "debug_pca"
    else:
        reducer, coords = fit_umap_transform(
            train_extreme_x,
            scaled_all,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            backend=umap_backend,
            n_jobs=n_jobs,
        )
        method = "umap"

    this_model_path = activation_model_path(
        models_dir,
        split_id=split_id,
        subset=subset,
        layer=layer,
        representation=activation_representation,
        method=method,
    )
    ensure_dir(this_model_path.parent)
    joblib.dump(
        {
            "model": reducer,
            "method": method,
            "split_id": split_id,
            "subset": subset,
            "layer": int(layer),
            "activation_representation": activation_representation,
            "train_extreme_row_ids": extreme_row_ids.astype(int).tolist(),
            "standardizer_mean": standardizer["mean"],
            "standardizer_scale": standardizer["scale"],
            "l2_normalize": bool(l2_normalize),
            "umap_params": {
                "n_components": 2,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
                "backend": umap_backend,
                "n_jobs": n_jobs,
            },
        },
        this_model_path,
    )

    coord_block = make_coord_block(
        beliefs,
        split_id=split_id,
        subset=subset,
        layer=layer,
        activation_representation=activation_representation,
        coords=coords,
        method=method,
        l2_normalize=l2_normalize,
    )

    anchor_coords = coords[extreme_row_ids]
    centroids_2d = compute_action_centroids_2d(
        anchor_coords,
        extreme_part["action"].astype(str).reset_index(drop=True),
        coord_prefix="act",
    )
    centroids_2d.insert(0, "activation_representation", activation_representation)
    centroids_2d.insert(0, "layer", int(layer))
    centroids_2d.insert(0, "subset", subset)
    centroids_2d.insert(0, "split_id", split_id)
    centroids_2d["manifold_method"] = "umap" if method == "umap" else "debug_pca_not_primary"
    centroids_2d["l2_normalize"] = bool(l2_normalize)

    highdim = compute_highdim_action_centroids(
        train_extreme_x,
        extreme_part["action"].astype(str).reset_index(drop=True),
    )
    highdim_named = {
        highdim_key(
            split_id=split_id,
            subset=subset,
            layer=layer,
            representation=activation_representation,
            action=action,
        ): value
        for action, value in highdim.items()
    }

    summary = {
        "split_id": split_id,
        "subset": subset,
        "layer": int(layer),
        "activation_representation": activation_representation,
        "method": method,
        "umap_backend": umap_backend,
        "n_train_extreme_rows": int(len(extreme_part)),
        "n_unique_train_extreme_rows": int(extreme_part["row_id"].nunique()),
        "n_all_rows_transformed": int(len(coord_block)),
        "model_path": rel(this_model_path),
        "centroid_counts": {
            str(row.action): int(row.n_extreme_examples) for row in centroids_2d.itertuples()
        },
    }
    return coord_block, centroids_2d, highdim_named, summary


def plot_configurations(layers: list[int], representations: list[str]) -> set[tuple[int, str]]:
    """Choose a small default set of activation manifolds to plot."""
    if 12 in layers:
        plot_layer = 12
    else:
        plot_layer = layers[len(layers) // 2]
    if "delta_pre" in representations:
        plot_rep = "delta_pre"
    else:
        plot_rep = representations[0]
    return {(plot_layer, plot_rep)}


def make_figures(plot_blocks: dict[tuple[int, str], pd.DataFrame], figure_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []
    for (layer, representation), block in sorted(plot_blocks.items()):
        if block.empty:
            continue
        prefix = f"layer_{int(layer):02d}_{representation}"
        specs = [
            ("agent_action", f"activation_umap_{prefix}_by_agent_action.png", "agent_action"),
            ("top_action", f"activation_umap_{prefix}_by_top_action.png", "top_action"),
            (
                "commitment_to_final",
                f"activation_umap_{prefix}_by_commitment_to_final.png",
                "commitment_to_final",
            ),
            ("progress", f"activation_umap_{prefix}_by_progress.png", "progress"),
        ]
        for color_col, filename, label in specs:
            fig, ax = plt.subplots(figsize=(7, 5))
            activation_umap_scatter(
                block,
                color_col=color_col,
                ax=ax,
                title=(
                    f"Activation UMAP {prefix} by {label}; x=activation UMAP dimension 1, "
                    "y=activation UMAP dimension 2, point=prefix activation state"
                ),
            )
            path = figure_dir / filename
            savefig(path, fig=fig)
            figures.append({"path": rel(path), "description": f"Activation UMAP by {label}"})

        if "strict_before_any_action_mention" in block.columns:
            strict = block.loc[block["strict_before_any_action_mention"].astype(bool)].copy()
            if not strict.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                activation_umap_scatter(
                    strict,
                    color_col="agent_action",
                    ax=ax,
                    title=(
                        f"Strict pre-action activation UMAP {prefix} by agent_action; "
                        "x=activation UMAP dimension 1, y=activation UMAP dimension 2, "
                        "point=prefix activation state before first action mention"
                    ),
                )
                path = figure_dir / f"activation_umap_{prefix}_strict_pre_action_by_agent_action.png"
                savefig(path, fig=fig)
                figures.append(
                    {"path": rel(path), "description": "Strict pre-action activation UMAP by agent_action"}
                )
    return figures


def write_report(
    path: Path,
    *,
    io_map_path: Path,
    beliefs_path: Path,
    splits_path: Path,
    extremes_path: Path,
    coords_path: Path,
    centroids_2d_path: Path,
    centroids_highdim_path: Path,
    feature_dir: Path,
    layers: list[int],
    representations: list[str],
    subsets: list[str],
    l2_normalize: bool,
    summaries: list[dict[str, Any]],
    figures: list[dict[str, str]],
    debug_pca: bool,
    umap_backend: str,
    n_jobs: int | None,
    random_state: int | None,
    coords_rows: int,
    centroid_rows: int,
    highdim_keys: int,
) -> None:
    lines = [
        "# Activation Conceptual Manifold",
        "",
        "Step 5 fits activation manifolds anchored by the same train action-extreme examples used for behavior manifolds.",
        "",
        "## Inputs and Outputs",
        "",
        f"- IO map: `{rel(io_map_path)}`",
        f"- Belief input: `{rel(beliefs_path)}`",
        f"- Splits input: `{rel(splits_path)}`",
        f"- Extreme examples input: `{rel(extremes_path)}`",
        f"- Activation feature directory: `{rel(feature_dir)}`",
        f"- Coordinates output: `{rel(coords_path)}`",
        f"- 2D centroids output: `{rel(centroids_2d_path)}`",
        f"- High-dimensional centroids output: `{rel(centroids_highdim_path)}`",
        "",
        "## Method",
        "",
        f"- Layers: `{layers}`",
        f"- Activation representations: `{representations}`",
        f"- Subsets: `{subsets}`",
        f"- L2 normalize after train-fitted scaling: `{l2_normalize}`",
        "- Standardization is fit on split training rows only.",
        "- UMAP is fit on train action-extreme row_ids only.",
        "- Extreme examples are reused from Step 3; activation extremes are not selected independently.",
        (
            "- UMAP parameters: `n_components=2`, `n_neighbors=15`, `min_dist=0.1`, "
            f"`metric=cosine`, `random_state={random_state}`, `backend={umap_backend}`, `n_jobs={n_jobs}`"
        ),
    ]
    if debug_pca:
        lines.append("- Diagnostic mode used: `--debug_pca`; these outputs are not primary UMAP evidence.")
    else:
        lines.append("- Primary manifold method: `UMAP`.")

    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Fitted blocks: `{len(summaries)}`",
            f"- Coordinate rows: `{coords_rows}`",
            f"- 2D centroid rows: `{centroid_rows}`",
            f"- High-dimensional centroid arrays: `{highdim_keys}`",
            "",
            "## Fitted Blocks",
            "",
        ]
    )
    for item in summaries[:80]:
        lines.append(
            f"- split `{item['split_id']}`, subset `{item['subset']}`, layer `{item['layer']}`, "
            f"repr `{item['activation_representation']}`: train_extreme_rows={item['n_train_extreme_rows']}, "
            f"unique_train_rows={item['n_unique_train_extreme_rows']}, "
            f"centroid_counts={item['centroid_counts']}, model=`{item['model_path']}`"
        )
    if len(summaries) > 80:
        lines.append(f"- ... {len(summaries) - 80} additional blocks omitted from report listing.")

    lines.extend(["", "## Figures", ""])
    if figures:
        for fig in figures:
            lines.append(f"- `{fig['path']}`: {fig['description']}")
    else:
        lines.append("- No figures generated.")
    lines.extend(
        [
            "",
            "Figure interpretation:",
            "",
            "- x-axis = activation UMAP dimension 1.",
            "- y-axis = activation UMAP dimension 2.",
            "- each point = one prefix-level activation state aligned to a behavior belief vector.",
            "- color variable is listed in each plot title.",
        ]
    )
    save_markdown(path, "\n".join(lines))


def run(
    *,
    io_map_path: Path,
    beliefs_path: Path,
    splits_path: Path,
    extremes_path: Path,
    coords_output: Path,
    centroids_2d_output: Path,
    centroids_highdim_output: Path,
    models_dir: Path,
    figure_dir: Path,
    report_output: Path,
    layers_arg: str,
    representations_arg: str,
    subsets_arg: str,
    l2_normalize: bool,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    umap_backend: str,
    n_jobs: int | None,
    deterministic: bool,
    debug_pca: bool,
) -> dict[str, Any]:
    resolved_umap_backend = "debug_pca"
    effective_random_state: int | None = random_state
    if not debug_pca:
        resolved_umap_backend = resolve_umap_backend(umap_backend)
        effective_random_state = random_state if deterministic else None

    io_map = read_json(io_map_path)
    if not isinstance(io_map, dict):
        raise ValueError(f"{io_map_path} must contain a JSON object")
    feature_dir = resolve_repo_path(str(io_map["activation_feature_dir"]))
    layers = parse_layers_arg(layers_arg, io_map.get("available_layers", []))
    representations = parse_representations_arg(
        representations_arg,
        io_map.get("available_representations", []),
    )

    beliefs = read_parquet_checked(beliefs_path)
    splits = load_splits(splits_path)
    extremes = read_parquet_checked(extremes_path)
    validate_inputs(beliefs, splits, extremes)
    subsets = parse_subsets_arg(subsets_arg, sorted(extremes["subset"].dropna().astype(str).unique()))
    split_lookup = {int(split["split_id"]): split for split in splits}
    extreme_groups = {
        (int(split_id), str(subset)): part.reset_index(drop=True)
        for (split_id, subset), part in extremes.groupby(["split_id", "subset"], sort=True)
        if str(subset) in set(subsets)
    }

    plot_configs = plot_configurations(layers, representations)
    plot_blocks: dict[tuple[int, str], pd.DataFrame] = {}
    centroid_parts: list[pd.DataFrame] = []
    highdim_centroids: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []

    writer = ParquetBlockWriter(coords_output)
    try:
        for layer in layers:
            for representation in representations:
                feature_path = activation_feature_path(feature_dir, layer, representation)
                features = load_activation_memmap(feature_path, expected_rows=len(beliefs))
                for split in splits:
                    split_id = int(split["split_id"])
                    train_indices = [int(i) for i in split["train_indices"]]
                    scaled_all, standardizer = standardize_features_for_split(
                        features,
                        train_indices,
                        l2_normalize=l2_normalize,
                    )
                    for (group_split_id, subset), extreme_part in extreme_groups.items():
                        if group_split_id != split_id:
                            continue
                        coord_block, centroids_2d, highdim, summary = fit_one_activation_block(
                            beliefs=beliefs,
                            scaled_all=scaled_all,
                            split=split_lookup[split_id],
                            extreme_part=extreme_part,
                            layer=layer,
                            activation_representation=representation,
                            models_dir=models_dir,
                            standardizer=standardizer,
                            l2_normalize=l2_normalize,
                            random_state=effective_random_state,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            umap_backend=resolved_umap_backend,
                            n_jobs=n_jobs,
                            debug_pca=debug_pca,
                        )
                        writer.write(coord_block)
                        centroid_parts.append(centroids_2d)
                        highdim_centroids.update(highdim)
                        summaries.append(summary)

                        if (
                            split_id == 0
                            and subset == "all_prefixes"
                            and (layer, representation) in plot_configs
                        ):
                            plot_blocks[(layer, representation)] = coord_block.copy()
    finally:
        writer.close()

    centroids_2d_all = pd.concat(centroid_parts, ignore_index=True) if centroid_parts else pd.DataFrame()
    ensure_dir(centroids_2d_output.parent)
    centroids_2d_all.to_parquet(centroids_2d_output, index=False)
    ensure_dir(centroids_highdim_output.parent)
    np.savez_compressed(centroids_highdim_output, **highdim_centroids)
    figures = make_figures(plot_blocks, figure_dir)
    write_report(
        report_output,
        io_map_path=io_map_path,
        beliefs_path=beliefs_path,
        splits_path=splits_path,
        extremes_path=extremes_path,
        coords_path=coords_output,
        centroids_2d_path=centroids_2d_output,
        centroids_highdim_path=centroids_highdim_output,
        feature_dir=feature_dir,
        layers=layers,
        representations=representations,
        subsets=subsets,
        l2_normalize=l2_normalize,
        summaries=summaries,
        figures=figures,
        debug_pca=debug_pca,
        umap_backend=resolved_umap_backend,
        n_jobs=n_jobs,
        random_state=effective_random_state,
        coords_rows=writer.rows_written,
        centroid_rows=len(centroids_2d_all),
        highdim_keys=len(highdim_centroids),
    )
    return {
        "coords_rows": writer.rows_written,
        "centroid_rows": int(len(centroids_2d_all)),
        "highdim_keys": int(len(highdim_centroids)),
        "num_blocks": int(len(summaries)),
        "layers": layers,
        "representations": representations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_map", default=DEFAULT_OUTPUT_DIR / "io_map.json")
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--extremes", default=DEFAULT_OUTPUT_DIR / "extreme_examples/action_extreme_examples.parquet")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--representations", default="raw,delta_pre")
    parser.add_argument("--subsets", default="all", help="`all` or comma-separated subset names from Step 3 extremes.")
    parser.add_argument(
        "--l2_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize rows after train-fitted standardization.",
    )
    parser.add_argument(
        "--coords_output",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_coords.parquet",
    )
    parser.add_argument(
        "--centroids_2d_output",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_2d.parquet",
    )
    parser.add_argument(
        "--centroids_highdim_output",
        default=DEFAULT_OUTPUT_DIR / "activation_manifold/activation_centroids_highdim.npz",
    )
    parser.add_argument("--models_dir", default=DEFAULT_OUTPUT_DIR / "models")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "activation")
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "05_activation_manifold_report.md")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--umap_backend",
        choices=["auto", "cpu", "cuml"],
        default="auto",
        help="Use `cuml` for CUDA/RAPIDS GPU UMAP when available; `auto` falls back to CPU.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="CPU umap-learn worker count. For real parallel UMAP, also pass --no-deterministic.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep fixed UMAP random_state. Disable to let CPU UMAP use n_jobs parallelism.",
    )
    parser.add_argument(
        "--debug_pca",
        action="store_true",
        help="Diagnostic only. Fit PCA instead of UMAP and label outputs as non-primary.",
    )
    args = parser.parse_args()

    try:
        result = run(
            io_map_path=resolve_repo_path(args.io_map),
            beliefs_path=resolve_repo_path(args.beliefs),
            splits_path=resolve_repo_path(args.splits),
            extremes_path=resolve_repo_path(args.extremes),
            coords_output=resolve_repo_path(args.coords_output),
            centroids_2d_output=resolve_repo_path(args.centroids_2d_output),
            centroids_highdim_output=resolve_repo_path(args.centroids_highdim_output),
            models_dir=resolve_repo_path(args.models_dir),
            figure_dir=resolve_repo_path(args.figure_dir),
            report_output=resolve_repo_path(args.report),
            layers_arg=args.layers,
            representations_arg=args.representations,
            subsets_arg=args.subsets,
            l2_normalize=args.l2_normalize,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            umap_backend=args.umap_backend,
            n_jobs=args.n_jobs,
            deterministic=args.deterministic,
            debug_pca=args.debug_pca,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {rel(args.coords_output)} with {result['coords_rows']} rows")
    print(f"Wrote {rel(args.centroids_2d_output)} with {result['centroid_rows']} rows")
    print(f"Wrote {rel(args.centroids_highdim_output)} with {result['highdim_keys']} centroid arrays")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
