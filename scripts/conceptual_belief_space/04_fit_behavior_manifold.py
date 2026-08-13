#!/usr/bin/env python3
"""Step 4: fit behavior conceptual belief manifolds anchored by train extremes."""

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

from src.conceptual_belief_space.centroids import compute_action_centroids  # noqa: E402
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.manifold import (  # noqa: E402
    fit_debug_pca_transform,
    fit_umap_transform,
    model_path,
    representation_columns,
    require_umap,
)
from src.conceptual_belief_space.plotting import (  # noqa: E402
    behavior_umap_scatter,
    plot_behavior_trajectories,
    savefig,
)


METADATA_COLS = [
    "row_id",
    "decision_id",
    "trajectory_id",
    "progress",
    "agent_action",
    "top_action",
    "commitment_to_final",
    "prefix_token_index",
]


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


def validate_inputs(
    beliefs: pd.DataFrame,
    splits: list[dict[str, Any]],
    extremes: pd.DataFrame,
    representation: str,
) -> list[str]:
    feature_cols = representation_columns(representation)
    required_belief = ["row_id", "decision_id", "trajectory_id", *feature_cols]
    missing_belief = [col for col in required_belief if col not in beliefs.columns]
    if missing_belief:
        raise ValueError(f"belief table missing columns: {missing_belief}")
    required_extreme = ["split_id", "subset", "action", "row_id", "split_role"]
    missing_extreme = [col for col in required_extreme if col not in extremes.columns]
    if missing_extreme:
        raise ValueError(f"extreme examples missing columns: {missing_extreme}")
    if not beliefs["row_id"].is_unique:
        raise ValueError("belief row_id must be unique")
    expected = np.arange(len(beliefs), dtype=np.int64)
    if not np.array_equal(beliefs["row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("belief row_id must match row order")
    invalid_actions = sorted(set(extremes["action"].dropna().astype(str)) - set(ACTIONS))
    if invalid_actions:
        raise ValueError(f"invalid extreme action labels: {invalid_actions}")
    split_ids = {int(split["split_id"]) for split in splits}
    unknown_split_ids = sorted(set(extremes["split_id"].astype(int)) - split_ids)
    if unknown_split_ids:
        raise ValueError(f"extremes reference split ids missing from split JSON: {unknown_split_ids}")
    if (extremes["split_role"].astype(str) != "train").any():
        raise ValueError("all extreme examples must have split_role == train")
    return feature_cols


def assert_train_only(extreme_part: pd.DataFrame, split: dict[str, Any], beliefs: pd.DataFrame) -> None:
    train_row_ids = set(beliefs.iloc[split["train_indices"]]["row_id"].astype(int).tolist())
    test_row_ids = set(beliefs.iloc[split["test_indices"]]["row_id"].astype(int).tolist())
    selected = set(extreme_part["row_id"].astype(int).tolist())
    if selected & test_row_ids:
        raise ValueError(f"test rows found in extremes for split {split['split_id']}")
    if not selected <= train_row_ids:
        raise ValueError(f"extreme rows outside training rows for split {split['split_id']}")


def fit_one_block(
    beliefs: pd.DataFrame,
    all_x: np.ndarray,
    feature_cols: list[str],
    split: dict[str, Any],
    extreme_part: pd.DataFrame,
    representation: str,
    models_dir: Path,
    *,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    debug_pca: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    split_id = int(split["split_id"])
    subset = str(extreme_part["subset"].iloc[0])
    assert_train_only(extreme_part, split, beliefs)

    train_row_ids = extreme_part["row_id"].astype(int).to_numpy()
    train_x = all_x[train_row_ids]
    if debug_pca:
        reducer, coords = fit_debug_pca_transform(train_x, all_x)
        method = "debug_pca"
    else:
        reducer, coords = fit_umap_transform(
            train_x,
            all_x,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        method = "umap"

    this_model_path = model_path(
        models_dir,
        split_id=split_id,
        subset=subset,
        representation=representation,
        method=method,
    )
    ensure_dir(this_model_path.parent)
    joblib.dump(
        {
            "model": reducer,
            "method": method,
            "split_id": split_id,
            "subset": subset,
            "representation": representation,
            "feature_columns": feature_cols,
            "train_extreme_row_ids": train_row_ids.astype(int).tolist(),
            "umap_params": {
                "n_components": 2,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
            },
        },
        this_model_path,
    )

    coords_df = beliefs[[col for col in METADATA_COLS if col in beliefs.columns]].copy()
    coords_df.insert(0, "representation", representation)
    coords_df.insert(0, "subset", subset)
    coords_df.insert(0, "split_id", split_id)
    coords_df["beh_umap1"] = coords[:, 0]
    coords_df["beh_umap2"] = coords[:, 1]
    if debug_pca:
        coords_df["diagnostic_method"] = "debug_pca_not_primary_umap"

    anchor_coords = coords[train_row_ids]
    anchor_features = all_x[train_row_ids]
    centroids = compute_action_centroids(
        features=anchor_features,
        coords_2d=anchor_coords,
        actions=extreme_part["action"].astype(str).reset_index(drop=True),
        feature_columns=feature_cols,
    )
    centroids.insert(0, "representation", representation)
    centroids.insert(0, "subset", subset)
    centroids.insert(0, "split_id", split_id)
    if debug_pca:
        centroids["diagnostic_method"] = "debug_pca_not_primary_umap"

    summary = {
        "split_id": split_id,
        "subset": subset,
        "representation": representation,
        "method": method,
        "n_train_extreme_rows": int(len(extreme_part)),
        "n_unique_train_extreme_rows": int(extreme_part["row_id"].nunique()),
        "n_all_rows_transformed": int(len(coords_df)),
        "model_path": rel(this_model_path),
        "centroid_counts": {
            str(row.action): int(row.n_extreme_examples) for row in centroids.itertuples()
        },
    }
    return coords_df, centroids, this_model_path, summary


def select_plot_block(coords: pd.DataFrame) -> pd.DataFrame:
    preferred = coords.loc[(coords["split_id"] == 0) & (coords["subset"] == "all_prefixes")]
    if not preferred.empty:
        return preferred.copy()
    first = coords[["split_id", "subset"]].drop_duplicates().iloc[0]
    return coords.loc[(coords["split_id"] == first["split_id"]) & (coords["subset"] == first["subset"])].copy()


def selected_decisions_for_trajectories(coords: pd.DataFrame, max_decisions: int = 8) -> list[str]:
    decision_counts = (
        coords.groupby(["agent_action", "decision_id"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["agent_action", "n"], ascending=[True, False])
    )
    selected: list[str] = []
    for action in ACTIONS:
        part = decision_counts.loc[decision_counts["agent_action"] == action]
        if not part.empty:
            selected.append(str(part["decision_id"].iloc[0]))
    if len(selected) < max_decisions:
        for decision_id in decision_counts.sort_values("n", ascending=False)["decision_id"].astype(str):
            if decision_id not in selected:
                selected.append(decision_id)
            if len(selected) >= max_decisions:
                break
    return selected[:max_decisions]


def make_figures(coords: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    block = select_plot_block(coords)
    split_id = int(block["split_id"].iloc[0])
    subset = str(block["subset"].iloc[0])
    prefix = f"split_{split_id}_{subset}"
    figures: list[dict[str, str]] = []

    plot_specs = [
        ("agent_action", "behavior_umap_by_agent_action.png", "Behavior UMAP by agent_action"),
        ("top_action", "behavior_umap_by_top_action.png", "Behavior UMAP by top_action"),
        (
            "commitment_to_final",
            "behavior_umap_by_commitment_to_final.png",
            "Behavior UMAP by commitment_to_final",
        ),
        ("progress", "behavior_umap_by_progress.png", "Behavior UMAP by progress"),
    ]
    for color_col, filename, title in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        behavior_umap_scatter(
            block,
            color_col=color_col,
            ax=ax,
            title=(
                f"{title} ({prefix}); x=behavior UMAP dimension 1, "
                "y=behavior UMAP dimension 2, point=prefix belief vector"
            ),
        )
        path = figure_dir / filename
        savefig(path, fig=fig)
        figures.append({"path": rel(path), "description": title})

    decisions = selected_decisions_for_trajectories(block)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_behavior_trajectories(
        block,
        decision_ids=decisions,
        ax=ax,
        title=(
            f"Selected behavior trajectories ({prefix}); x=behavior UMAP dimension 1, "
            "y=behavior UMAP dimension 2, connected points=prefixes from one decision"
        ),
    )
    path = figure_dir / "selected_behavior_trajectories.png"
    savefig(path, fig=fig)
    figures.append({"path": rel(path), "description": "Selected behavior trajectories"})
    return figures


def write_report(
    path: Path,
    *,
    beliefs_path: Path,
    splits_path: Path,
    extremes_path: Path,
    coords_path: Path,
    centroids_path: Path,
    representation: str,
    feature_cols: list[str],
    summaries: list[dict[str, Any]],
    figures: list[dict[str, str]],
    debug_pca: bool,
) -> None:
    lines = [
        "# Behavior Conceptual Belief Manifold",
        "",
        "Step 4 fits behavior belief manifolds anchored by train-only action-extreme examples.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Belief input: `{rel(beliefs_path)}`",
        f"- Splits input: `{rel(splits_path)}`",
        f"- Extreme examples input: `{rel(extremes_path)}`",
        f"- Coordinates output: `{rel(coords_path)}`",
        f"- Centroids output: `{rel(centroids_path)}`",
        "",
        "## Method",
        "",
        f"- Representation: `{representation}`",
        f"- Feature columns: `{feature_cols}`",
        f"- Primary method: `UMAP`",
        "- UMAP parameters: `n_components=2`, `n_neighbors=15`, `min_dist=0.1`, `metric=euclidean`, `random_state=42`",
        "- Each UMAP model is fit only on train-side action-extreme examples for one `(split_id, subset)` block.",
        "- All prefix-level belief rows are transformed after fitting.",
    ]
    if debug_pca:
        lines.append("- Diagnostic mode used: `--debug_pca`; these outputs are not the primary UMAP manifold.")

    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Fitted blocks: `{len(summaries)}`",
            f"- Coordinate rows: `{sum(item['n_all_rows_transformed'] for item in summaries)}`",
            f"- Models saved: `{len(summaries)}`",
            "",
            "## Fitted Blocks",
            "",
        ]
    )
    for item in summaries:
        lines.append(
            f"- split `{item['split_id']}`, subset `{item['subset']}`: "
            f"train_extreme_rows={item['n_train_extreme_rows']}, "
            f"unique_train_rows={item['n_unique_train_extreme_rows']}, "
            f"centroid_counts={item['centroid_counts']}, model=`{item['model_path']}`"
        )

    lines.extend(["", "## Figures", ""])
    for fig in figures:
        lines.append(f"- `{fig['path']}`: {fig['description']}")
    lines.extend(
        [
            "",
            "Figure interpretation:",
            "",
            "- x-axis = behavior UMAP dimension 1.",
            "- y-axis = behavior UMAP dimension 2.",
            "- each point = one prefix-level action belief vector.",
            "- color variable is listed in each plot title.",
        ]
    )
    save_markdown(path, "\n".join(lines))


def run(
    beliefs_path: Path,
    splits_path: Path,
    extremes_path: Path,
    coords_output: Path,
    centroids_output: Path,
    models_dir: Path,
    figure_dir: Path,
    report_output: Path,
    representation: str,
    *,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    debug_pca: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not debug_pca:
        require_umap()

    beliefs = read_parquet_checked(beliefs_path)
    splits = load_splits(splits_path)
    extremes = read_parquet_checked(extremes_path)
    feature_cols = validate_inputs(beliefs, splits, extremes, representation)
    all_x = beliefs[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(all_x).all():
        raise ValueError(f"non-finite values found in representation columns: {feature_cols}")

    split_lookup = {int(split["split_id"]): split for split in splits}
    coord_parts: list[pd.DataFrame] = []
    centroid_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for (split_id, subset), extreme_part in extremes.groupby(["split_id", "subset"], sort=True):
        coords, centroids, _path, summary = fit_one_block(
            beliefs=beliefs,
            all_x=all_x,
            feature_cols=feature_cols,
            split=split_lookup[int(split_id)],
            extreme_part=extreme_part.reset_index(drop=True),
            representation=representation,
            models_dir=models_dir,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            debug_pca=debug_pca,
        )
        coord_parts.append(coords)
        centroid_parts.append(centroids)
        summaries.append(summary)

    coords_all = pd.concat(coord_parts, ignore_index=True)
    centroids_all = pd.concat(centroid_parts, ignore_index=True)
    ensure_dir(coords_output.parent)
    coords_all.to_parquet(coords_output, index=False)
    ensure_dir(centroids_output.parent)
    centroids_all.to_parquet(centroids_output, index=False)
    figures = make_figures(coords_all, figure_dir)
    write_report(
        report_output,
        beliefs_path=beliefs_path,
        splits_path=splits_path,
        extremes_path=extremes_path,
        coords_path=coords_output,
        centroids_path=centroids_output,
        representation=representation,
        feature_cols=feature_cols,
        summaries=summaries,
        figures=figures,
        debug_pca=debug_pca,
    )
    return coords_all, centroids_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument(
        "--extremes",
        default=DEFAULT_OUTPUT_DIR / "extreme_examples/action_extreme_examples.parquet",
    )
    parser.add_argument(
        "--coords_output",
        default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_coords.parquet",
    )
    parser.add_argument(
        "--centroids_output",
        default=DEFAULT_OUTPUT_DIR / "behavior_manifold/behavior_centroids.parquet",
    )
    parser.add_argument("--models_dir", default=DEFAULT_OUTPUT_DIR / "models")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "behavior")
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "04_behavior_manifold_report.md")
    parser.add_argument("--representation", choices=["clr", "sqrt_q"], default="clr")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--debug_pca",
        action="store_true",
        help="Diagnostic only. Fit PCA instead of UMAP and label outputs as non-primary.",
    )
    args = parser.parse_args()

    try:
        coords, centroids = run(
            beliefs_path=resolve_repo_path(args.beliefs),
            splits_path=resolve_repo_path(args.splits),
            extremes_path=resolve_repo_path(args.extremes),
            coords_output=resolve_repo_path(args.coords_output),
            centroids_output=resolve_repo_path(args.centroids_output),
            models_dir=resolve_repo_path(args.models_dir),
            figure_dir=resolve_repo_path(args.figure_dir),
            report_output=resolve_repo_path(args.report),
            representation=args.representation,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            debug_pca=args.debug_pca,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {rel(args.coords_output)} with {len(coords)} rows")
    print(f"Wrote {rel(args.centroids_output)} with {len(centroids)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
