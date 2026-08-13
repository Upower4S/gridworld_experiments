#!/usr/bin/env python3
"""Embed behavior vectors into a low-dimensional behavior manifold."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


DEFAULT_FEATURES = ("b_clr_UP", "b_clr_DOWN", "b_clr_LEFT", "b_clr_RIGHT")
STEP_KEYS = ("trajectory_id", "source_file", "step_id")
PROGRESS_BIN_EDGES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
PROGRESS_BIN_LABELS = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]


def require_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
        from sklearn.decomposition import PCA
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas, pyarrow, scikit-learn, and joblib."
        ) from exc
    return pd, np, PCA, joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/behavior_vectors.parquet"),
        help="Behavior-vector parquet input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/behavior_manifold.parquet"),
        help="Behavior manifold parquet output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/behavior_manifold_report.md"),
        help="Markdown report output.",
    )
    parser.add_argument(
        "--pca_model_output",
        type=Path,
        default=Path("outputs/models/behavior_pca.joblib"),
        help="Fitted PCA model output.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Numeric behavior vector columns to embed.",
    )
    parser.add_argument(
        "--use_umap",
        action="store_true",
        help="Also fit a 2D UMAP embedding if umap-learn is installed.",
    )
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=0)
    return parser.parse_args()


def validate_input(df: Any, features: list[str]) -> None:
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing behavior feature columns: {missing}")
    if len(features) < 2:
        raise SystemExit("At least two feature columns are required for 2D PCA.")
    if len(df) < 2:
        raise SystemExit("At least two rows are required for 2D PCA.")


def prepare_feature_matrix(df: Any, features: list[str], np: Any) -> Any:
    matrix = df[features].astype(float).to_numpy()
    finite_mask = np.isfinite(matrix).all(axis=1)
    if not finite_mask.all():
        bad_count = int((~finite_mask).sum())
        raise SystemExit(f"Found {bad_count} row(s) with non-finite behavior feature values.")
    return matrix


def fit_pca(df: Any, features: list[str], PCA: Any, joblib: Any, output_path: Path) -> tuple[Any, Any]:
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(df[features].astype(float))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    return pca, embedding


def fit_umap_if_requested(
    *,
    matrix: Any,
    use_umap: bool,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[Any | None, Any | None]:
    if not use_umap:
        return None, None
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "UMAP requested with --use_umap, but umap-learn is not installed. "
            "Install umap-learn or omit --use_umap."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(matrix)
    return reducer, embedding


def step_count(df: Any) -> int:
    if all(column in df.columns for column in STEP_KEYS):
        return int(df[list(STEP_KEYS)].drop_duplicates().shape[0])
    if "step_id" in df.columns:
        return int(df["step_id"].nunique())
    return 0


def trajectory_file_count(df: Any) -> int:
    if "source_file" in df.columns:
        return int(df["source_file"].nunique())
    if "trajectory_id" in df.columns:
        return int(df["trajectory_id"].nunique())
    return 0


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def action_counts(df: Any) -> dict[str, int]:
    if "top_action" not in df.columns or df.empty:
        return {}
    return {str(k): int(v) for k, v in df["top_action"].value_counts(dropna=False).sort_index().items()}


def distribution_line(title: str, distribution: dict[str, int]) -> str:
    if not distribution:
        return f"- {title}: unavailable"
    parts = [f"{key}={value}" for key, value in distribution.items()]
    return f"- {title}: " + ", ".join(parts)


def progress_bin_summary(df: Any, pd: Any) -> Any:
    if df.empty or "progress" not in df.columns:
        return pd.DataFrame(
            columns=[
                "progress_bin",
                "prefix_points",
                "mean_commitment_to_final",
                "mean_entropy_commitment",
            ]
        )

    work = df.copy()
    work["progress_bin"] = pd.cut(
        work["progress"].astype(float),
        bins=PROGRESS_BIN_EDGES,
        labels=PROGRESS_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    value_columns = [
        column
        for column in ("commitment_to_final", "entropy_commitment")
        if column in work.columns
    ]
    if not value_columns:
        summary = (
            work.groupby("progress_bin", observed=False)
            .size()
            .reset_index(name="prefix_points")
        )
        summary["mean_commitment_to_final"] = float("nan")
        summary["mean_entropy_commitment"] = float("nan")
        return summary

    summary = (
        work.groupby("progress_bin", observed=False)
        .agg(
            prefix_points=("progress_bin", "size"),
            mean_commitment_to_final=(
                "commitment_to_final",
                "mean",
            )
            if "commitment_to_final" in value_columns
            else ("progress", "count"),
            mean_entropy_commitment=(
                "entropy_commitment",
                "mean",
            )
            if "entropy_commitment" in value_columns
            else ("progress", "count"),
        )
        .reset_index()
    )
    if "commitment_to_final" not in value_columns:
        summary["mean_commitment_to_final"] = float("nan")
    if "entropy_commitment" not in value_columns:
        summary["mean_entropy_commitment"] = float("nan")
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def progress_summary_table(summary: Any) -> str:
    rows = []
    for row in summary.to_dict("records"):
        rows.append(
            [
                str(row["progress_bin"]),
                str(int(row["prefix_points"])),
                fmt_number(row["mean_commitment_to_final"]),
                fmt_number(row["mean_entropy_commitment"]),
            ]
        )
    return markdown_table(
        [
            "progress_bin",
            "prefix_points",
            "mean_commitment_to_final",
            "mean_entropy_commitment",
        ],
        rows,
    )


def pca_components_table(pca: Any, features: list[str]) -> str:
    rows = []
    for component_index, component in enumerate(pca.components_, start=1):
        rows.append(
            [
                f"PC{component_index}",
                *[fmt_number(weight, digits=6) for weight in component],
            ]
        )
    return markdown_table(["component", *features], rows)


def before_first_action_mask(df: Any) -> Any | None:
    if "is_before_first_action_mention" not in df.columns:
        return None
    return df["is_before_first_action_mention"].fillna(True).astype(bool)


def build_report(
    *,
    df: Any,
    pca: Any,
    features: list[str],
    pd: Any,
    input_path: Path,
    output_path: Path,
    pca_model_path: Path,
    used_umap: bool,
) -> str:
    before_mask = before_first_action_mask(df)
    before_df = df[before_mask].copy() if before_mask is not None else df.iloc[0:0].copy()

    lines = [
        "# Behavior Manifold Report",
        "",
        f"- Input file: `{input_path}`",
        f"- Output file: `{output_path}`",
        f"- PCA model: `{pca_model_path}`",
        f"- Number of prefix points: {len(df)}",
        f"- Number of steps: {step_count(df)}",
        f"- Number of trajectory files: {trajectory_file_count(df)}",
        f"- Embedding features: {', '.join(features)}",
        f"- UMAP fitted: {'yes' if used_umap else 'no'}",
        f"- PCA explained variance ratio: "
        f"{', '.join(fmt_number(value, digits=6) for value in pca.explained_variance_ratio_)}",
        "",
        "## PCA Components",
        "",
        pca_components_table(pca, features),
        "",
        "## Action Counts",
        "",
        distribution_line("top_action", action_counts(df)),
        "",
        "## Progress Bins",
        "",
        progress_summary_table(progress_bin_summary(df, pd)),
        "",
        "## Before First Action Mention",
        "",
    ]

    if before_mask is None:
        lines.append("The input does not include `is_before_first_action_mention`, so this split is unavailable.")
    else:
        lines.extend(
            [
                f"- Prefix points: {len(before_df)}",
                f"- Steps: {step_count(before_df)}",
                f"- Trajectory files: {trajectory_file_count(before_df)}",
                distribution_line("top_action", action_counts(before_df)),
                "",
                progress_summary_table(progress_bin_summary(before_df, pd)),
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    pd, np, PCA, joblib = require_dependencies()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    behavior = pd.read_parquet(args.input)
    validate_input(behavior, args.features)
    matrix = prepare_feature_matrix(behavior, args.features, np)

    pca, pca_embedding = fit_pca(behavior, args.features, PCA, joblib, args.pca_model_output)
    output = behavior.copy()
    output["beh_pca1"] = pca_embedding[:, 0]
    output["beh_pca2"] = pca_embedding[:, 1]

    _umap_model, umap_embedding = fit_umap_if_requested(
        matrix=matrix,
        use_umap=args.use_umap,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )
    if umap_embedding is not None:
        output["beh_umap1"] = umap_embedding[:, 0]
        output["beh_umap2"] = umap_embedding[:, 1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, engine="pyarrow", index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(
            df=output,
            pca=pca,
            features=args.features,
            pd=pd,
            input_path=args.input,
            output_path=args.output,
            pca_model_path=args.pca_model_output,
            used_umap=umap_embedding is not None,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {args.output} with {len(output)} rows")
    print(f"Wrote PCA model to {args.pca_model_output}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
