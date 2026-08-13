#!/usr/bin/env python3
"""Compute commitment metrics from prefix action score probabilities."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLUMNS = tuple(f"q_{action}" for action in ACTIONS)
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
        default=Path("outputs/prefix_action_scores.parquet"),
        help="Prefix action score parquet input.",
    )
    parser.add_argument(
        "--prefix_output",
        type=Path,
        default=Path("outputs/prefix_commitment_metrics.parquet"),
        help="Prefix-level commitment metrics output.",
    )
    parser.add_argument(
        "--step_output",
        type=Path,
        default=Path("outputs/step_commitment_summary.parquet"),
        help="Step-level commitment summary output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/commitment_report.md"),
        help="Markdown report output.",
    )
    return parser.parse_args()


def validate_input_columns(df: Any) -> None:
    required = [
        *STEP_KEYS,
        "prefix_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        *Q_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}")


def sort_prefixes(df: Any) -> Any:
    return df.sort_values([*STEP_KEYS, "prefix_token_index", "prefix_id"]).reset_index(drop=True)


def compute_prefix_metrics(df: Any, np: Any) -> Any:
    metrics = sort_prefixes(df.copy())
    q_values = metrics[list(Q_COLUMNS)].astype(float).to_numpy()
    if q_values.ndim != 2 or q_values.shape[1] != len(ACTIONS):
        raise ValueError("Expected q matrix with four action columns")

    top_indices = np.argmax(q_values, axis=1)
    sorted_probabilities = np.sort(q_values, axis=1)
    top_prob = sorted_probabilities[:, -1]
    second_prob = sorted_probabilities[:, -2]

    metrics["top_action"] = [ACTIONS[index] for index in top_indices]
    metrics["top_prob"] = top_prob
    metrics["second_prob"] = second_prob

    action_to_index = {action: index for index, action in enumerate(ACTIONS)}
    commitment_to_final: list[float] = []
    for row_index, action in enumerate(metrics["agent_action"].astype(str)):
        action_index = action_to_index.get(action)
        if action_index is None:
            commitment_to_final.append(float("nan"))
        else:
            commitment_to_final.append(float(q_values[row_index, action_index]))
    metrics["commitment_to_final"] = commitment_to_final

    clipped_q = np.clip(q_values, 1e-300, 1.0)
    entropy = -(q_values * np.log(clipped_q)).sum(axis=1)
    normalized_entropy = entropy / math.log(len(ACTIONS))
    metrics["entropy"] = entropy
    metrics["normalized_entropy"] = normalized_entropy
    metrics["entropy_commitment"] = 1.0 - normalized_entropy
    metrics["margin_commitment"] = top_prob - second_prob
    metrics["is_top_action_final"] = metrics["top_action"] == metrics["agent_action"].astype(str)

    grouped = metrics.groupby(list(STEP_KEYS), sort=False, dropna=False)
    previous_top_action = grouped["top_action"].shift()
    metrics["top_action_changed"] = (
        previous_top_action.notna() & (metrics["top_action"] != previous_top_action)
    )
    metrics["commitment_to_final_delta"] = grouped["commitment_to_final"].diff()
    metrics["entropy_commitment_delta"] = grouped["entropy_commitment"].diff()
    return metrics


def first_cross_progress(group: Any, column: str, threshold: float) -> float | None:
    matches = group[group[column] >= threshold]
    if matches.empty:
        return None
    return float(matches.iloc[0]["progress"])


def lock_in_progress(group: Any) -> float | None:
    condition = (
        (group["top_action"] == group["agent_action"].astype(str))
        & (group["margin_commitment"] >= 0.2)
    ).tolist()
    progresses = group["progress"].astype(float).tolist()
    for index in range(0, max(0, len(condition) - 2)):
        if condition[index] and condition[index + 1] and condition[index + 2]:
            return float(progresses[index])
    return None


def total_variation(group: Any, np: Any) -> float:
    if len(group) < 2:
        return 0.0
    q_values = group[list(Q_COLUMNS)].astype(float).to_numpy()
    return float(np.abs(np.diff(q_values, axis=0)).sum(axis=1).sum())


def compute_step_summary(prefix_metrics: Any, np: Any) -> Any:
    rows: list[dict[str, Any]] = []
    if prefix_metrics.empty:
        import pandas as pd

        return pd.DataFrame(rows)

    for key, group in prefix_metrics.groupby(list(STEP_KEYS), sort=False, dropna=False):
        trajectory_id, source_file, step_id = key
        group = group.sort_values(["prefix_token_index", "prefix_id"])
        final_prefix = group.iloc[-1]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "source_file": source_file,
                "step_id": int(step_id),
                "agent_action": final_prefix["agent_action"],
                "num_prefixes": int(len(group)),
                "final_prefix_id": int(final_prefix["prefix_id"]),
                "final_prefix_progress": float(final_prefix["progress"]),
                "final_prefix_commitment_to_final": float(final_prefix["commitment_to_final"]),
                "final_prefix_top_action": final_prefix["top_action"],
                "final_prefix_is_top_action_final": bool(final_prefix["is_top_action_final"]),
                "max_commitment_to_final": float(group["commitment_to_final"].max()),
                "mean_commitment_to_final": float(group["commitment_to_final"].mean()),
                "first_cross_0_5_progress": first_cross_progress(group, "commitment_to_final", 0.5),
                "first_cross_0_7_progress": first_cross_progress(group, "commitment_to_final", 0.7),
                "lock_in_progress": lock_in_progress(group),
                "switch_count": int(group["top_action_changed"].sum()),
                "total_variation": total_variation(group, np),
            }
        )

    import pandas as pd

    return pd.DataFrame(rows)


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if math.isnan(float(value)):
            return "NA"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "0"
    try:
        if math.isnan(float(value)):
            return "0"
    except (TypeError, ValueError):
        return str(value)
    return str(int(value))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def action_distribution(values: Any) -> dict[str, int]:
    counts = values.value_counts(dropna=False).to_dict()
    return {str(action): int(count) for action, count in counts.items()}


def distribution_lines(title: str, distribution: dict[str, int]) -> list[str]:
    if not distribution:
        return [f"- {title}: none"]
    parts = [f"{action}={count}" for action, count in sorted(distribution.items())]
    return [f"- {title}: " + ", ".join(parts)]


def subset_masks(prefix_metrics: Any) -> list[tuple[str, Any]]:
    masks = [("all prefixes", prefix_metrics.index == prefix_metrics.index)]
    if "is_before_first_action_mention" in prefix_metrics.columns:
        before_action = prefix_metrics["is_before_first_action_mention"].fillna(True).astype(bool)
        masks.append(("before first action mention", before_action))
        masks.append(("after first action mention", ~before_action))
    if "is_before_first_final_action_mention" in prefix_metrics.columns:
        before_final = (
            prefix_metrics["is_before_first_final_action_mention"].fillna(True).astype(bool)
        )
        masks.append(("before first final-action mention", before_final))
    return masks


def summarize_subset(prefix_metrics: Any, mask: Any, np: Any) -> dict[str, Any]:
    subset = prefix_metrics[mask].copy()
    step_summary = compute_step_summary(subset, np)
    return {
        "prefix_points": int(len(subset)),
        "steps": int(len(step_summary)),
        "mean_prefix_commitment_to_final": subset["commitment_to_final"].mean() if not subset.empty else None,
        "fraction_prefix_top_action_final": subset["is_top_action_final"].mean() if not subset.empty else None,
        "mean_entropy_commitment": subset["entropy_commitment"].mean() if not subset.empty else None,
        "mean_margin_commitment": subset["margin_commitment"].mean() if not subset.empty else None,
        "mean_final_prefix_commitment_to_final": (
            step_summary["final_prefix_commitment_to_final"].mean()
            if not step_summary.empty
            else None
        ),
        "fraction_final_prefix_top_action_final": (
            step_summary["final_prefix_is_top_action_final"].mean()
            if not step_summary.empty
            else None
        ),
        "median_first_cross_0_5_progress": (
            step_summary["first_cross_0_5_progress"].median()
            if not step_summary.empty
            else None
        ),
        "median_first_cross_0_7_progress": (
            step_summary["first_cross_0_7_progress"].median()
            if not step_summary.empty
            else None
        ),
        "median_lock_in_progress": (
            step_summary["lock_in_progress"].median()
            if not step_summary.empty
            else None
        ),
        "mean_switch_count": (
            step_summary["switch_count"].mean()
            if not step_summary.empty
            else None
        ),
    }


def report_text(prefix_metrics: Any, step_summary: Any, np: Any, input_path: Path) -> str:
    trajectory_files = int(prefix_metrics["source_file"].nunique())
    step_count = int(len(step_summary))
    prefix_points = int(len(prefix_metrics))

    mean_final_commitment = (
        step_summary["final_prefix_commitment_to_final"].mean() if not step_summary.empty else None
    )
    final_top_final_fraction = (
        step_summary["final_prefix_is_top_action_final"].mean() if not step_summary.empty else None
    )
    median_cross_0_5 = (
        step_summary["first_cross_0_5_progress"].median() if not step_summary.empty else None
    )
    median_cross_0_7 = (
        step_summary["first_cross_0_7_progress"].median() if not step_summary.empty else None
    )
    median_lock = step_summary["lock_in_progress"].median() if not step_summary.empty else None
    mean_switch_count = step_summary["switch_count"].mean() if not step_summary.empty else None

    subset_rows = []
    before_action_summary: dict[str, Any] | None = None
    for name, mask in subset_masks(prefix_metrics):
        summary = summarize_subset(prefix_metrics, mask, np)
        if name == "before first action mention":
            before_action_summary = summary
        subset_rows.append(
            [
                name,
                fmt_int(summary["prefix_points"]),
                fmt_int(summary["steps"]),
                fmt_number(summary["mean_prefix_commitment_to_final"]),
                fmt_number(summary["fraction_prefix_top_action_final"]),
                fmt_number(summary["mean_entropy_commitment"]),
                fmt_number(summary["mean_margin_commitment"]),
                fmt_number(summary["mean_final_prefix_commitment_to_final"]),
                fmt_number(summary["fraction_final_prefix_top_action_final"]),
            ]
        )

    lines = [
        "# Commitment Metrics Report",
        "",
        f"- Input score file: `{input_path}`",
        f"- Number of trajectory files: {trajectory_files}",
        f"- Number of steps: {step_count}",
        f"- Number of prefix points: {prefix_points}",
        f"- Mean final-prefix commitment_to_final: {fmt_number(mean_final_commitment)}",
        f"- Fraction of final prefixes where top_action == agent_action: {fmt_number(final_top_final_fraction)}",
        f"- Median first_cross_0_5_progress: {fmt_number(median_cross_0_5)}",
        f"- Median first_cross_0_7_progress: {fmt_number(median_cross_0_7)}",
        f"- Median lock_in_progress: {fmt_number(median_lock)}",
        f"- Mean switch_count: {fmt_number(mean_switch_count)}",
        "",
        "## Action Distributions",
        "",
        *distribution_lines(
            "agent_action distribution by step",
            action_distribution(step_summary["agent_action"]) if not step_summary.empty else {},
        ),
        *distribution_lines(
            "final-prefix top_action distribution by step",
            action_distribution(step_summary["final_prefix_top_action"]) if not step_summary.empty else {},
        ),
        "",
        "## Action-Mention Split",
        "",
        markdown_table(
            [
                "subset",
                "prefix_points",
                "steps",
                "mean_prefix_commitment",
                "frac_prefix_top_final",
                "mean_entropy_commitment",
                "mean_margin_commitment",
                "mean_final_prefix_commitment",
                "frac_final_prefix_top_final",
            ],
            subset_rows,
        ),
        "",
        "## Before First Action Mention Summary",
        "",
    ]

    if before_action_summary is None:
        lines.append("The input does not include `is_before_first_action_mention`, so this split is unavailable.")
    else:
        lines.extend(
            [
                f"- Prefix points: {fmt_int(before_action_summary['prefix_points'])}",
                f"- Steps represented: {fmt_int(before_action_summary['steps'])}",
                f"- Mean final-prefix commitment_to_final within subset: "
                f"{fmt_number(before_action_summary['mean_final_prefix_commitment_to_final'])}",
                f"- Fraction of subset final prefixes where top_action == agent_action: "
                f"{fmt_number(before_action_summary['fraction_final_prefix_top_action_final'])}",
                f"- Median first_cross_0_5_progress within subset: "
                f"{fmt_number(before_action_summary['median_first_cross_0_5_progress'])}",
                f"- Median first_cross_0_7_progress within subset: "
                f"{fmt_number(before_action_summary['median_first_cross_0_7_progress'])}",
                f"- Median lock_in_progress within subset: "
                f"{fmt_number(before_action_summary['median_lock_in_progress'])}",
                f"- Mean switch_count within subset: "
                f"{fmt_number(before_action_summary['mean_switch_count'])}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()

    scores = pd.read_parquet(args.input)
    validate_input_columns(scores)

    prefix_metrics = compute_prefix_metrics(scores, np)
    step_summary = compute_step_summary(prefix_metrics, np)

    args.prefix_output.parent.mkdir(parents=True, exist_ok=True)
    prefix_metrics.to_parquet(args.prefix_output, engine="pyarrow", index=False)
    args.step_output.parent.mkdir(parents=True, exist_ok=True)
    step_summary.to_parquet(args.step_output, engine="pyarrow", index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        report_text(prefix_metrics, step_summary, np, args.input),
        encoding="utf-8",
    )

    print(f"Wrote {args.prefix_output} with {len(prefix_metrics)} rows")
    print(f"Wrote {args.step_output} with {len(step_summary)} rows")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
