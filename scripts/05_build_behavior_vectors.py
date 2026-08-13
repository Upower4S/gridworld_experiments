#!/usr/bin/env python3
"""Build behavior vectors from prefix action probabilities."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLUMNS = tuple(f"q_{action}" for action in ACTIONS)
B_LOG_COLUMNS = tuple(f"b_log_{action}" for action in ACTIONS)
B_CLR_COLUMNS = tuple(f"b_clr_{action}" for action in ACTIONS)
STEP_KEYS = ("trajectory_id", "source_file", "step_id")


def require_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas, numpy, and pyarrow before running this script."
        ) from exc
    return pd, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/prefix_commitment_metrics.parquet"),
        help="Prefix commitment metrics parquet input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/behavior_vectors.parquet"),
        help="Behavior-vector parquet output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/behavior_vectors_report.md"),
        help="Markdown report output.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Minimum action probability before taking log.",
    )
    return parser.parse_args()


def validate_input(df: Any) -> None:
    required = [
        "prefix_id",
        *STEP_KEYS,
        "prefix_token_index",
        "progress",
        "agent_action",
        *Q_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}")


def action_distribution(values: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).sort_index().items()}


def distribution_line(title: str, distribution: dict[str, int]) -> str:
    if not distribution:
        return f"- {title}: unavailable"
    return f"- {title}: " + ", ".join(f"{key}={value}" for key, value in distribution.items())


def fmt_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def build_behavior_vectors(df: Any, np: Any, epsilon: float) -> tuple[Any, dict[str, float]]:
    if epsilon <= 0.0:
        raise SystemExit("--epsilon must be positive")

    output = df.copy()
    q_values = output[list(Q_COLUMNS)].astype(float).to_numpy()
    finite_mask = np.isfinite(q_values).all(axis=1)
    if not finite_mask.all():
        bad_count = int((~finite_mask).sum())
        raise SystemExit(f"Found {bad_count} row(s) with non-finite q values.")

    q_sums = q_values.sum(axis=1)
    clipped = np.clip(q_values, epsilon, 1.0)
    log_q = np.log(clipped)
    clr = log_q - log_q.mean(axis=1, keepdims=True)

    for index, column in enumerate(B_LOG_COLUMNS):
        output[column] = log_q[:, index]
    for index, column in enumerate(B_CLR_COLUMNS):
        output[column] = clr[:, index]

    output["behavior_vector_transform"] = "centered_log_ratio"
    output["behavior_vector_epsilon"] = float(epsilon)

    diagnostics = {
        "max_abs_q_sum_error": float(np.max(np.abs(q_sums - 1.0))) if len(q_sums) else float("nan"),
        "max_abs_clr_sum": float(np.max(np.abs(clr.sum(axis=1)))) if len(clr) else float("nan"),
        "min_q": float(np.min(q_values)) if len(q_values) else float("nan"),
        "max_q": float(np.max(q_values)) if len(q_values) else float("nan"),
    }
    return output, diagnostics


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def feature_summary_table(df: Any) -> str:
    rows: list[list[str]] = []
    for column in B_CLR_COLUMNS:
        values = df[column].astype(float)
        rows.append(
            [
                column,
                fmt_number(values.mean()),
                fmt_number(values.std()),
                fmt_number(values.min()),
                fmt_number(values.max()),
            ]
        )
    return markdown_table(["feature", "mean", "std", "min", "max"], rows)


def build_report(
    *,
    df: Any,
    diagnostics: dict[str, float],
    input_path: Path,
    output_path: Path,
    epsilon: float,
) -> str:
    step_count = int(df[list(STEP_KEYS)].drop_duplicates().shape[0]) if all(c in df.columns for c in STEP_KEYS) else 0
    trajectory_count = int(df["source_file"].nunique()) if "source_file" in df.columns else 0
    lines = [
        "# Behavior Vectors Report",
        "",
        f"- Input file: `{input_path}`",
        f"- Output file: `{output_path}`",
        f"- Number of prefix points: {len(df)}",
        f"- Number of steps: {step_count}",
        f"- Number of trajectory files: {trajectory_count}",
        f"- Transform: centered log-ratio over q_UP, q_DOWN, q_LEFT, q_RIGHT",
        f"- Epsilon: {fmt_number(epsilon)}",
        f"- Max absolute q-sum error: {fmt_number(diagnostics['max_abs_q_sum_error'])}",
        f"- Max absolute CLR row-sum: {fmt_number(diagnostics['max_abs_clr_sum'])}",
        f"- Minimum q value: {fmt_number(diagnostics['min_q'])}",
        f"- Maximum q value: {fmt_number(diagnostics['max_q'])}",
        "",
        "## Action Distributions",
        "",
        distribution_line("agent_action", action_distribution(df["agent_action"])) if "agent_action" in df.columns else "- agent_action: unavailable",
        distribution_line("top_action", action_distribution(df["top_action"])) if "top_action" in df.columns else "- top_action: unavailable",
        "",
        "## CLR Feature Summary",
        "",
        feature_summary_table(df),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    prefix_metrics = pd.read_parquet(args.input)
    validate_input(prefix_metrics)
    behavior_vectors, diagnostics = build_behavior_vectors(prefix_metrics, np, args.epsilon)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    behavior_vectors.to_parquet(args.output, engine="pyarrow", index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(
            df=behavior_vectors,
            diagnostics=diagnostics,
            input_path=args.input,
            output_path=args.output,
            epsilon=args.epsilon,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {args.output} with {len(behavior_vectors)} rows")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
