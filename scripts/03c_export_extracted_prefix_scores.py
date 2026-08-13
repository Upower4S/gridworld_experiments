#!/usr/bin/env python3
"""Export the faithful CoT-window extraction beliefs as a stage-03 prefix score table.

The 2026-07-21 label-integrity audit quarantined the stored stage-03 MLX scores
(93.5% q-mismatch on re-extraction). This script materializes the adjudicated-faithful
labels from the CoT-window extraction (`extracted_beliefs.parquet`) in the exact output
contract of `03_score_prefix_actions_*`, so the downstream behavioral pipeline
(04 -> 05 -> 06 -> 07) can be re-run unchanged on faithful labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLUMNS = tuple(f"q_{action}" for action in ACTIONS)

BELIEF_COLUMNS = [
    "prefix_id",
    "trajectory_id",
    "step_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    *Q_COLUMNS,
    "num_cot_tokens",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
]

PREFIX_COLUMNS = [
    "prefix_id",
    "trajectory_id",
    "source_file",
    "step_id",
    "prefix_token_index",
    "num_analysis_tokens",
    "progress",
    "agent_action",
    "first_action_mention_token_index",
    "first_final_action_mention_token_index",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
]

SHARED_KEYS = ["trajectory_id", "step_id", "prefix_token_index", "agent_action"]


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
        "--beliefs",
        type=Path,
        default=Path(
            "outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations/extracted_beliefs.parquet"
        ),
        help="Faithful extracted beliefs parquet (canonical labels).",
    )
    parser.add_argument(
        "--prefixes",
        type=Path,
        default=Path("outputs/prefixes.parquet"),
        help="Canonical prefix table supplying source_file and mention metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prefix_action_scores_extracted.parquet"),
        help="Stage-03-contract score table output.",
    )
    parser.add_argument(
        "--progress_tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed |progress| disagreement between the two sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()

    beliefs = pd.read_parquet(args.beliefs, columns=BELIEF_COLUMNS)
    prefixes = pd.read_parquet(args.prefixes, columns=PREFIX_COLUMNS)

    merged = beliefs.merge(
        prefixes,
        on="prefix_id",
        how="inner",
        suffixes=("", "_prefixes"),
        validate="one_to_one",
    )
    missing_from_prefixes = set(beliefs.prefix_id) - set(merged.prefix_id)
    if missing_from_prefixes:
        raise SystemExit(f"{len(missing_from_prefixes)} belief rows have no prefix metadata: "
                         f"{sorted(missing_from_prefixes)[:5]}")
    dropped = sorted(set(prefixes.prefix_id) - set(merged.prefix_id))
    print(f"beliefs rows: {len(beliefs)}; prefixes rows: {len(prefixes)}; "
          f"joined: {len(merged)}; prefix_ids without faithful labels: {dropped[:10]}")

    for key in SHARED_KEYS:
        left = merged[key]
        right = merged[f"{key}_prefixes"]
        mismatch = (left.astype(str) != right.astype(str)).sum()
        if mismatch:
            raise SystemExit(f"column {key!r} disagrees between beliefs and prefixes on {mismatch} rows")
    progress_gap = (merged["progress"] - merged["progress_prefixes"]).abs().max()
    if progress_gap > args.progress_tolerance:
        raise SystemExit(f"progress disagrees by up to {progress_gap:.3e} (> {args.progress_tolerance:.1e})")

    strict_vs_mention = (
        merged["strict_before_any_action_mention"].astype(bool)
        != merged["is_before_first_action_mention"].fillna(True).astype(bool)
    ).sum()
    print(f"strict_before_any_action_mention vs is_before_first_action_mention disagreements: {strict_vs_mention}")

    q_values = merged[list(Q_COLUMNS)].astype(float).to_numpy()
    if (q_values < 0).any():
        raise SystemExit("negative action probabilities in beliefs input")
    row_sums = q_values.sum(axis=1)
    if (row_sums <= 0).any():
        raise SystemExit("zero-sum action probability rows in beliefs input")
    max_sum_deviation = float(np.abs(row_sums - 1.0).max())
    q_values = q_values / row_sums[:, None]
    print(f"max |sum(q) - 1| before renormalization: {max_sum_deviation:.3e}")

    out = pd.DataFrame(
        {
            "prefix_id": merged["prefix_id"],
            "trajectory_id": merged["trajectory_id"],
            "source_file": merged["source_file"],
            "step_id": merged["step_id"],
            "prefix_token_index": merged["prefix_token_index"],
            "num_analysis_tokens": merged["num_analysis_tokens"],
            "num_cot_tokens": merged["num_cot_tokens"],
            "progress": merged["progress"],
            "agent_action": merged["agent_action"],
            **{column: q_values[:, index] for index, column in enumerate(Q_COLUMNS)},
            "top_action": [ACTIONS[index] for index in np.argmax(q_values, axis=1)],
            "scoring_backend": "cot_window_extraction",
            "first_action_mention_token_index": merged["first_action_mention_token_index"],
            "first_final_action_mention_token_index": merged["first_final_action_mention_token_index"],
            "is_before_first_action_mention": merged["is_before_first_action_mention"],
            "is_before_first_final_action_mention": merged["is_before_first_final_action_mention"],
            "strict_before_any_action_mention": merged["strict_before_any_action_mention"],
            "strict_before_final_action_mention": merged["strict_before_final_action_mention"],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, engine="pyarrow", index=False)
    print(f"Wrote {args.output} with {len(out)} rows over "
          f"{out.trajectory_id.nunique()} trajectories / {out.groupby(['trajectory_id','step_id']).ngroups} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
