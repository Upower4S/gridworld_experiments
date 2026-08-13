#!/usr/bin/env python3
"""Step 3: grouped splits and train-only action-extreme example selection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED  # noqa: E402
from src.conceptual_belief_space.extremes import select_action_extremes  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.splits import (  # noqa: E402
    grouped_train_test_splits,
    make_decision_id,
    save_splits_json,
)


SUBSET_SPECS = [
    ("all_prefixes", None, "all rows"),
    (
        "strict_before_any_action_mention",
        "strict_before_any_action_mention",
        "rows before first explicit action mention",
    ),
    (
        "strict_before_final_action_mention",
        "strict_before_final_action_mention",
        "rows before first explicit final-action mention",
    ),
    (
        "buffered_before_any_action_mention",
        "buffered_before_any_action_mention",
        "rows at least buffer tokens before first explicit action mention",
    ),
    (
        "buffered_before_final_action_mention",
        "buffered_before_final_action_mention",
        "rows at least buffer tokens before first explicit final-action mention",
    ),
    ("high_commitment", "commitment_to_final", "commitment_to_final >= threshold"),
    ("low_commitment", "commitment_to_final", "commitment_to_final <= threshold"),
]


@dataclass
class SelectionResult:
    rows: pd.DataFrame
    fallback_used: bool
    skipped_reason: str | None
    threshold_counts: dict[str, int]
    final_counts: dict[str, int]


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


def validate_belief_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    required = ["row_id", "trajectory_id", "prefix_token_index", "agent_action", *[f"q_{a}" for a in ACTIONS]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"belief table missing required columns: {missing}")
    if group_col not in df.columns:
        raise ValueError(f"group_col `{group_col}` is not present in belief table")
    invalid_actions = sorted(set(df["agent_action"].dropna().astype(str)) - set(ACTIONS))
    if invalid_actions:
        raise ValueError(f"invalid agent_action labels: {invalid_actions}")
    if not df["row_id"].is_unique:
        raise ValueError("row_id must be unique")
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(df["row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("row_id must match row order for split/index validation")

    out = df.copy()
    if "decision_id" not in out.columns or out["decision_id"].isna().any():
        out["decision_id"] = make_decision_id(out)
    return out


def effective_n_splits(df: pd.DataFrame, group_col: str, requested: int) -> tuple[int, int]:
    n_groups = int(df[group_col].nunique(dropna=True))
    if n_groups < 2:
        raise ValueError(f"need at least 2 groups for grouped train/test splits, got {n_groups}")
    if requested < 2:
        raise ValueError("--n_splits must be at least 2")
    return min(requested, n_groups), n_groups


def enrich_splits(
    df: pd.DataFrame,
    raw_splits: list[dict[str, Any]],
    group_col: str,
    n_groups: int,
    requested_n_splits: int,
    effective_splits: int,
    seed: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for split in raw_splits:
        split_id = int(split["fold"])
        train_indices = [int(i) for i in split["train_indices"]]
        test_indices = [int(i) for i in split["test_indices"]]
        train_groups = sorted(map(str, df.iloc[train_indices][group_col].dropna().unique().tolist()))
        test_groups = sorted(map(str, df.iloc[test_indices][group_col].dropna().unique().tolist()))
        if set(train_groups) & set(test_groups):
            raise ValueError(f"group leakage in split {split_id}")
        enriched.append(
            {
                "split_id": split_id,
                "fold": split_id,
                "group_col": group_col,
                "seed": int(seed),
                "requested_n_splits": int(requested_n_splits),
                "n_splits": int(effective_splits),
                "n_groups": int(n_groups),
                "n_train_rows": int(len(train_indices)),
                "n_test_rows": int(len(test_indices)),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                "train_groups": train_groups,
                "test_groups": test_groups,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )
    return enriched


def subset_mask(df: pd.DataFrame, subset_name: str, subset_col: str | None) -> tuple[pd.Series | None, str | None]:
    if subset_name == "all_prefixes":
        return pd.Series(True, index=df.index), None
    if subset_name == "high_commitment":
        if "commitment_to_final" not in df.columns:
            return None, "missing commitment_to_final"
        return df["commitment_to_final"] >= 0.9, None
    if subset_name == "low_commitment":
        if "commitment_to_final" not in df.columns:
            return None, "missing commitment_to_final"
        return df["commitment_to_final"] <= 0.4, None
    if subset_col is None or subset_col not in df.columns:
        return None, f"missing {subset_col}"
    return df[subset_col].fillna(False).astype(bool), None


def select_for_action(
    subset_df: pd.DataFrame,
    action: str,
    top_k: int,
    threshold: float | None,
    max_per_decision: int,
) -> pd.DataFrame:
    selected = select_action_extremes(
        df=subset_df,
        action=action,
        split_train_indices=None,
        top_k=top_k,
        threshold=threshold,
        max_per_decision=max_per_decision,
    )
    if selected.empty:
        return selected
    selected = selected.sort_values(
        [f"q_{action}", "commitment_to_final", "progress", "row_id"],
        ascending=[False, False, False, True],
    ).copy()
    selected["rank_within_action"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    selected["q_action"] = selected[f"q_{action}"].astype(float)
    return selected


def select_balanced_extremes(
    train_subset: pd.DataFrame,
    split_id: int,
    subset_name: str,
    top_k: int,
    threshold: float,
    max_per_decision: int,
    min_per_action: int,
) -> SelectionResult:
    threshold_parts: dict[str, pd.DataFrame] = {
        action: select_for_action(
            train_subset,
            action=action,
            top_k=top_k,
            threshold=threshold,
            max_per_decision=max_per_decision,
        )
        for action in ACTIONS
    }
    threshold_counts = {action: int(len(part)) for action, part in threshold_parts.items()}
    fallback_used = min(threshold_counts.values() or [0]) < min_per_action

    if fallback_used:
        parts = {
            action: select_for_action(
                train_subset,
                action=action,
                top_k=top_k,
                threshold=None,
                max_per_decision=max_per_decision,
            )
            for action in ACTIONS
        }
    else:
        parts = threshold_parts

    final_counts_before_balance = {action: int(len(part)) for action, part in parts.items()}
    balance_n = min(final_counts_before_balance.values() or [0])
    if balance_n <= 0:
        return SelectionResult(
            rows=pd.DataFrame(),
            fallback_used=fallback_used,
            skipped_reason="at least one action has zero candidates after fallback",
            threshold_counts=threshold_counts,
            final_counts=final_counts_before_balance,
        )

    balanced_parts: list[pd.DataFrame] = []
    for action in ACTIONS:
        part = parts[action].head(balance_n).copy()
        part["split_id"] = split_id
        part["subset"] = subset_name
        part["action"] = action
        part["rank_within_action"] = np.arange(1, len(part) + 1, dtype=np.int64)
        part["selected_by_threshold"] = bool(not fallback_used)
        part["selected_by_topk"] = True
        part["split_role"] = "train"
        part["top_k_used"] = int(top_k)
        part["threshold_used"] = float(threshold) if not fallback_used else np.nan
        part["fallback_used"] = bool(fallback_used)
        balanced_parts.append(part)

    rows = pd.concat(balanced_parts, ignore_index=True)
    final_counts = {action: int((rows["action"] == action).sum()) for action in ACTIONS}
    return SelectionResult(
        rows=rows,
        fallback_used=fallback_used,
        skipped_reason=None,
        threshold_counts=threshold_counts,
        final_counts=final_counts,
    )


def build_extremes(
    df: pd.DataFrame,
    splits: list[dict[str, Any]],
    top_k: int,
    threshold: float,
    max_per_decision: int,
    min_per_action: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    selection_summaries: list[dict[str, Any]] = []

    for split in splits:
        split_id = int(split["split_id"])
        train_indices = split["train_indices"]
        train_mask = df.index.isin(train_indices)
        for subset_name, subset_col, _description in SUBSET_SPECS:
            mask, reason = subset_mask(df, subset_name, subset_col)
            if reason is not None or mask is None:
                skipped.append({"split_id": split_id, "subset": subset_name, "reason": reason})
                continue
            train_subset = df.loc[train_mask & mask].copy()
            if train_subset.empty:
                skipped.append({"split_id": split_id, "subset": subset_name, "reason": "no train rows"})
                continue

            result = select_balanced_extremes(
                train_subset=train_subset,
                split_id=split_id,
                subset_name=subset_name,
                top_k=top_k,
                threshold=threshold,
                max_per_decision=max_per_decision,
                min_per_action=min_per_action,
            )
            summary = {
                "split_id": split_id,
                "subset": subset_name,
                "train_subset_rows": int(len(train_subset)),
                "fallback_used": bool(result.fallback_used),
                "threshold_counts": result.threshold_counts,
                "final_counts": result.final_counts,
                "skipped_reason": result.skipped_reason,
            }
            selection_summaries.append(summary)
            if result.skipped_reason is not None:
                skipped.append(
                    {"split_id": split_id, "subset": subset_name, "reason": result.skipped_reason}
                )
                continue
            rows.append(result.rows)

    if not rows:
        return pd.DataFrame(), skipped, selection_summaries

    extremes = pd.concat(rows, ignore_index=True)
    output_cols = [
        "split_id",
        "subset",
        "action",
        "row_id",
        "decision_id",
        "trajectory_id",
        "q_action",
        "rank_within_action",
        "selected_by_threshold",
        "selected_by_topk",
        "split_role",
        "top_k_used",
        "threshold_used",
        "fallback_used",
        "prefix_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
    ]
    output_cols = [col for col in output_cols if col in extremes.columns]
    return extremes[output_cols].copy(), skipped, selection_summaries


def validate_outputs(df: pd.DataFrame, splits: list[dict[str, Any]], extremes: pd.DataFrame) -> dict[str, Any]:
    leakage_ok = True
    split_lookup = {int(split["split_id"]): split for split in splits}
    for split in splits:
        train_groups = set(split["train_groups"])
        test_groups = set(split["test_groups"])
        if train_groups & test_groups:
            raise ValueError(f"group leakage detected in split {split['split_id']}")

    if not extremes.empty:
        for split_id, part in extremes.groupby("split_id"):
            split = split_lookup[int(split_id)]
            train_ids = set(df.iloc[split["train_indices"]]["row_id"].astype(int).tolist())
            test_ids = set(df.iloc[split["test_indices"]]["row_id"].astype(int).tolist())
            selected_ids = set(part["row_id"].astype(int).tolist())
            if selected_ids & test_ids:
                leakage_ok = False
                raise ValueError(f"test rows selected in split {split_id}")
            if not selected_ids <= train_ids:
                leakage_ok = False
                raise ValueError(f"selected rows outside train rows in split {split_id}")

        counts = extremes.groupby(["split_id", "subset", "action"]).size().rename("n").reset_index()
        for (split_id, subset), part in counts.groupby(["split_id", "subset"]):
            if set(part["action"]) != set(ACTIONS):
                raise ValueError(f"missing actions in selected extremes for split {split_id}, subset {subset}")
            if part["n"].nunique() != 1:
                raise ValueError(f"unbalanced selected counts for split {split_id}, subset {subset}")

    return {
        "group_leakage_free": leakage_ok,
        "num_extreme_rows": int(len(extremes)),
        "num_split_subset_blocks": int(extremes.groupby(["split_id", "subset"]).ngroups)
        if not extremes.empty
        else 0,
    }


def counts_table(extremes: pd.DataFrame) -> str:
    if extremes.empty:
        return "- No extreme rows selected.\n"
    counts = (
        extremes.groupby(["split_id", "subset", "action"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["split_id", "subset", "action"])
    )
    lines = []
    for (split_id, subset), part in counts.groupby(["split_id", "subset"]):
        action_counts = ", ".join(f"{row.action}={int(row.n)}" for row in part.itertuples())
        lines.append(f"- split `{split_id}`, `{subset}`: {action_counts}")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    beliefs_path: Path,
    splits_path: Path,
    extremes_path: Path,
    df: pd.DataFrame,
    splits: list[dict[str, Any]],
    extremes: pd.DataFrame,
    skipped: list[dict[str, Any]],
    selection_summaries: list[dict[str, Any]],
    validation: dict[str, Any],
    group_col: str,
    requested_n_splits: int,
    effective_splits: int,
    n_groups: int,
    seed: int,
    top_k: int,
    threshold: float,
    max_per_decision: int,
    min_per_action: int,
) -> None:
    fallback_blocks = [
        item for item in selection_summaries if item.get("fallback_used") and not item.get("skipped_reason")
    ]
    selected_blocks = [
        item for item in selection_summaries if item.get("skipped_reason") is None
    ]
    sample_cols = [
        "split_id",
        "subset",
        "action",
        "row_id",
        "decision_id",
        "trajectory_id",
        "q_action",
        "rank_within_action",
        "fallback_used",
    ]
    sample_rows = (
        extremes[[col for col in sample_cols if col in extremes.columns]]
        .head(8)
        .to_dict(orient="records")
        if not extremes.empty
        else []
    )

    lines = [
        "# Grouped Splits and Action Extremes",
        "",
        "Step 3 creates grouped train/test splits and selects action-extreme examples from training rows only.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Belief input: `{rel(beliefs_path)}`",
        f"- Splits output: `{rel(splits_path)}`",
        f"- Extremes output: `{rel(extremes_path)}`",
        "",
        "## Split Configuration",
        "",
        f"- Group column: `{group_col}`",
        f"- Groups: `{n_groups}`",
        f"- Requested folds: `{requested_n_splits}`",
        f"- Effective folds: `{effective_splits}`",
        f"- Seed: `{seed}`",
        f"- Group leakage free: `{validation['group_leakage_free']}`",
        "",
        "## Extreme Selection Configuration",
        "",
        f"- Actions: `{ACTIONS}`",
        f"- Threshold: `{threshold}`",
        f"- Top-k per action before balancing: `{top_k}`",
        f"- Max per decision: `{max_per_decision}`",
        f"- Minimum threshold candidates per action before fallback: `{min_per_action}`",
        "",
        "## Dataset Counts",
        "",
        f"- Belief rows: `{len(df)}`",
        f"- Decisions: `{df['decision_id'].nunique()}`",
        f"- Trajectories: `{df['trajectory_id'].nunique() if 'trajectory_id' in df.columns else 'NA'}`",
        f"- Extreme rows selected: `{len(extremes)}`",
        f"- Selected split/subset blocks: `{validation['num_split_subset_blocks']}`",
        f"- Fallback-selected blocks: `{len(fallback_blocks)}`",
        f"- Skipped blocks: `{len(skipped)}`",
        "",
        "## Selected Counts",
        "",
        counts_table(extremes).rstrip(),
        "",
        "## Skipped Subsets",
        "",
    ]
    if skipped:
        for item in skipped:
            lines.append(f"- split `{item['split_id']}`, `{item['subset']}`: {item['reason']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Fallback Blocks", ""])
    if fallback_blocks:
        for item in fallback_blocks:
            lines.append(
                f"- split `{item['split_id']}`, `{item['subset']}`: "
                f"threshold_counts={item['threshold_counts']}, final_counts={item['final_counts']}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Fold Summary", ""])
    for split in splits:
        lines.append(
            f"- split `{split['split_id']}`: train_rows={split['n_train_rows']}, "
            f"test_rows={split['n_test_rows']}, train_groups={split['n_train_groups']}, "
            f"test_groups={split['n_test_groups']}"
        )

    lines.extend(["", "## Sample Extreme Rows", "", "```json"])
    lines.append(json.dumps(sample_rows, indent=2, default=str))
    lines.append("```")
    save_markdown(path, "\n".join(lines))


def run(
    beliefs_path: Path,
    splits_output: Path,
    extremes_output: Path,
    report_output: Path,
    group_col: str,
    requested_n_splits: int,
    seed: int,
    top_k: int,
    threshold: float,
    max_per_decision: int,
    min_per_action: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    df = read_parquet_checked(beliefs_path)
    df = validate_belief_table(df, group_col=group_col)
    effective_splits, n_groups = effective_n_splits(df, group_col, requested_n_splits)
    raw_splits = grouped_train_test_splits(
        df=df,
        group_col=group_col,
        n_splits=effective_splits,
        seed=seed,
    )
    splits = enrich_splits(
        df=df,
        raw_splits=raw_splits,
        group_col=group_col,
        n_groups=n_groups,
        requested_n_splits=requested_n_splits,
        effective_splits=effective_splits,
        seed=seed,
    )

    extremes, skipped, selection_summaries = build_extremes(
        df=df,
        splits=splits,
        top_k=top_k,
        threshold=threshold,
        max_per_decision=max_per_decision,
        min_per_action=min_per_action,
    )
    validation = validate_outputs(df, splits, extremes)

    save_splits_json(splits_output, splits)
    ensure_dir(extremes_output.parent)
    extremes.to_parquet(extremes_output, index=False)
    write_report(
        path=report_output,
        beliefs_path=beliefs_path,
        splits_path=splits_output,
        extremes_path=extremes_output,
        df=df,
        splits=splits,
        extremes=extremes,
        skipped=skipped,
        selection_summaries=selection_summaries,
        validation=validation,
        group_col=group_col,
        requested_n_splits=requested_n_splits,
        effective_splits=effective_splits,
        n_groups=n_groups,
        seed=seed,
        top_k=top_k,
        threshold=threshold,
        max_per_decision=max_per_decision,
        min_per_action=min_per_action,
    )
    return splits, extremes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beliefs",
        default="outputs/conceptual_belief_space/action_belief_vectors.parquet",
    )
    parser.add_argument(
        "--splits_output",
        default="outputs/conceptual_belief_space/splits/group_splits.json",
    )
    parser.add_argument(
        "--extremes_output",
        default="outputs/conceptual_belief_space/extreme_examples/action_extreme_examples.parquet",
    )
    parser.add_argument(
        "--report",
        default="outputs/conceptual_belief_space/03_extreme_examples_report.md",
    )
    parser.add_argument("--group_col", choices=["trajectory_id", "decision_id"], default="trajectory_id")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--max_per_decision", type=int, default=3)
    parser.add_argument(
        "--min_per_action",
        type=int,
        default=1,
        help="Fallback to unthresholded top-k if any action has fewer threshold candidates.",
    )
    args = parser.parse_args()

    splits, extremes = run(
        beliefs_path=resolve_repo_path(args.beliefs),
        splits_output=resolve_repo_path(args.splits_output),
        extremes_output=resolve_repo_path(args.extremes_output),
        report_output=resolve_repo_path(args.report),
        group_col=args.group_col,
        requested_n_splits=args.n_splits,
        seed=args.seed,
        top_k=args.top_k,
        threshold=args.threshold,
        max_per_decision=args.max_per_decision,
        min_per_action=args.min_per_action,
    )
    print(f"Wrote {rel(args.splits_output)} with {len(splits)} splits")
    print(f"Wrote {rel(args.extremes_output)} with {len(extremes)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
