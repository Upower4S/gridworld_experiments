#!/usr/bin/env python3
"""Align activation timepoints to behavior-side prefix metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{action}" for action in ACTIONS]


def make_decision_id(trajectory_id: object, step_id: object) -> str:
    return f"{trajectory_id}::{step_id}"


def ensure_decision_id(df: pd.DataFrame) -> pd.DataFrame:
    if "decision_id" not in df.columns and {"trajectory_id", "step_id"}.issubset(df.columns):
        df = df.copy()
        df["decision_id"] = df["trajectory_id"].astype(str) + "::" + df["step_id"].astype(str)
    return df


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional parquet not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"WARNING: failed to read optional parquet {path}: {exc}")
        return None


def merge_optional_columns(base: pd.DataFrame, optional: pd.DataFrame | None, label: str) -> pd.DataFrame:
    if optional is None or optional.empty:
        return base
    optional = ensure_decision_id(optional)
    if "prefix_id" in base.columns and "prefix_id" in optional.columns:
        keys = ["prefix_id"]
    elif {"decision_id", "prefix_token_index"}.issubset(base.columns) and {"decision_id", "prefix_token_index"}.issubset(optional.columns):
        keys = ["decision_id", "prefix_token_index"]
    elif "decision_id" in base.columns and "decision_id" in optional.columns:
        keys = ["decision_id"]
    else:
        print(f"WARNING: cannot merge optional {label}; no compatible key")
        return base

    add_cols = [col for col in optional.columns if col not in base.columns and col not in keys]
    if not add_cols:
        return base

    optional_small = optional[keys + add_cols].drop_duplicates(keys)
    before = len(base)
    out = base.merge(optional_small, on=keys, how="left")
    if len(out) != before:
        raise RuntimeError(f"Optional merge {label} changed row count from {before} to {len(out)}")
    print(f"Merged {len(add_cols)} columns from {label}: {add_cols[:12]}{'...' if len(add_cols) > 12 else ''}")
    return out


def add_behavior_clr(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    missing = [col for col in Q_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing q columns for CLR transform: {missing}")
    q = df[Q_COLS].astype(float).clip(lower=eps).to_numpy()
    log_q = np.log(q)
    clr = log_q - log_q.mean(axis=1, keepdims=True)
    out = df.copy()
    for idx, action in enumerate(ACTIONS):
        out[f"b_clr_{action}"] = clr[:, idx]
    return out


def read_behavior_with_optional_inputs(
    behavior_path: Path,
    step_summary_path: Path,
    behavior_vectors_path: Path,
    behavior_manifold_path: Path,
) -> pd.DataFrame:
    behavior = pd.read_parquet(behavior_path)
    behavior = ensure_decision_id(behavior)
    if "decision_id" not in behavior.columns:
        raise KeyError("Behavior table must contain decision_id or trajectory_id + step_id")
    if "prefix_token_index" not in behavior.columns:
        raise KeyError("Behavior table must contain prefix_token_index")

    behavior = merge_optional_columns(behavior, read_optional_parquet(step_summary_path), "step_commitment_summary")
    behavior = merge_optional_columns(behavior, read_optional_parquet(behavior_vectors_path), "behavior_vectors")
    behavior = merge_optional_columns(behavior, read_optional_parquet(behavior_manifold_path), "behavior_manifold")
    behavior = add_behavior_clr(behavior)
    return behavior


def read_activation_index_for_decisions(path: Path, decisions: set[str]) -> pd.DataFrame:
    required_cols = [
        "activation_id",
        "decision_id",
        "token_type",
        "cot_token_index",
        "num_cot_tokens",
        "progress",
        "activation_file",
        "activation_format",
        "activation_key",
        "activation_row_index",
        "available_layers_json",
        "hidden_dim",
        "dtype",
    ]
    available = parquet_columns(path)
    selected = [col for col in required_cols if col in available]
    missing = [col for col in required_cols if col not in available]
    if missing:
        raise KeyError(f"Activation index missing required columns: {missing}")
    activation = pd.read_parquet(path, columns=selected)
    activation = activation[activation["decision_id"].isin(decisions)].copy()
    return activation


def pointer_table(
    activation: pd.DataFrame,
    *,
    token_type: str,
    prefix: str,
) -> pd.DataFrame:
    subset = activation[activation["token_type"].eq(token_type)].copy()
    subset = subset.sort_values(["decision_id", "cot_token_index", "activation_id"]).drop_duplicates("decision_id")
    keep = ["decision_id", "activation_id", "activation_file", "activation_key", "activation_row_index"]
    rename = {
        "activation_id": f"{prefix}_activation_id",
        "activation_file": f"{prefix}_activation_file",
        "activation_key": f"{prefix}_activation_key",
        "activation_row_index": f"{prefix}_activation_row_index",
    }
    return subset[keep].rename(columns=rename)


def main_activation_table(activation: pd.DataFrame, token_type: str) -> pd.DataFrame:
    subset = activation[activation["token_type"].eq(token_type)].copy()
    subset = subset.rename(
        columns={
            "token_type": "activation_token_type",
            "cot_token_index": "activation_cot_token_index",
            "progress": "activation_progress",
        }
    )
    return subset[
        [
            "decision_id",
            "activation_id",
            "activation_token_type",
            "activation_cot_token_index",
            "activation_progress",
            "activation_file",
            "activation_format",
            "activation_key",
            "activation_row_index",
            "available_layers_json",
            "hidden_dim",
            "dtype",
        ]
    ]


def nearest_cot_matches(
    missing: pd.DataFrame,
    cot_main: pd.DataFrame,
    *,
    max_token_tolerance: int,
) -> pd.DataFrame:
    if missing.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    cot_by_decision = {
        decision_id: group.sort_values("activation_cot_token_index").reset_index(drop=True)
        for decision_id, group in cot_main.groupby("decision_id", sort=False)
    }

    for _, row in missing.iterrows():
        decision_id = row["decision_id"]
        target = row["target_cot_token_index"]
        group = cot_by_decision.get(decision_id)
        result = {"row_id": row["row_id"]}
        if group is None or group.empty or pd.isna(target):
            result.update(
                {
                    "alignment_status": "failed_missing_cot",
                    "alignment_token_distance": pd.NA,
                }
            )
            results.append(result)
            continue

        indices = group["activation_cot_token_index"].astype(int).to_numpy()
        target_int = int(target)
        pos = int(np.searchsorted(indices, target_int))
        candidate_positions = [p for p in [pos - 1, pos] if 0 <= p < len(indices)]
        best_pos = min(candidate_positions, key=lambda p: abs(indices[p] - target_int))
        distance = int(abs(indices[best_pos] - target_int))
        if distance > max_token_tolerance:
            result.update(
                {
                    "alignment_status": "failed_outside_tolerance",
                    "alignment_token_distance": distance,
                }
            )
            results.append(result)
            continue

        match = group.iloc[best_pos].to_dict()
        result.update(match)
        result["alignment_status"] = "nearest"
        result["alignment_token_distance"] = distance
        results.append(result)

    return pd.DataFrame(results)


def align_behavior_to_activation(
    behavior: pd.DataFrame,
    activation: pd.DataFrame,
    *,
    max_token_tolerance: int,
) -> pd.DataFrame:
    behavior = behavior.copy().reset_index(drop=True)
    behavior["row_id"] = np.arange(len(behavior))
    behavior["prefix_token_index"] = pd.to_numeric(behavior["prefix_token_index"], errors="coerce")

    num_cot_by_decision = (
        activation[["decision_id", "num_cot_tokens"]]
        .dropna()
        .drop_duplicates("decision_id")
        .set_index("decision_id")["num_cot_tokens"]
        .to_dict()
    )
    behavior["activation_num_cot_tokens"] = behavior["decision_id"].map(num_cot_by_decision)

    pre_main = main_activation_table(activation, "pre_reasoning")
    cot_main = main_activation_table(activation, "cot")
    pre_pointers = pointer_table(activation, token_type="pre_reasoning", prefix="pre")
    post_pointers = pointer_table(activation, token_type="post_reasoning", prefix="post")

    pre_rows = behavior[behavior["prefix_token_index"].eq(0)].copy()
    pre_matches = pre_rows[["row_id", "decision_id"]].merge(pre_main, on="decision_id", how="left")
    pre_matches["alignment_status"] = np.where(pre_matches["activation_id"].notna(), "exact_pre_reasoning", "failed_missing_pre_reasoning")
    pre_matches["alignment_token_distance"] = np.where(pre_matches["activation_id"].notna(), 0, pd.NA)

    cot_rows = behavior[behavior["prefix_token_index"].gt(0)].copy()
    cot_rows["target_cot_token_index"] = cot_rows["prefix_token_index"].round().astype("Int64")
    final_mask = (
        cot_rows["activation_num_cot_tokens"].notna()
        & cot_rows["prefix_token_index"].eq(cot_rows["activation_num_cot_tokens"])
    )
    cot_rows.loc[final_mask, "target_cot_token_index"] = cot_rows.loc[final_mask, "activation_num_cot_tokens"].astype("Int64")

    cot_exact = cot_rows[["row_id", "decision_id", "target_cot_token_index"]].merge(
        cot_main,
        left_on=["decision_id", "target_cot_token_index"],
        right_on=["decision_id", "activation_cot_token_index"],
        how="left",
    )
    cot_exact["alignment_status"] = np.where(cot_exact["activation_id"].notna(), "exact", "missing_exact")
    cot_exact["alignment_token_distance"] = np.where(cot_exact["activation_id"].notna(), 0, pd.NA)

    missing_exact = cot_exact[cot_exact["activation_id"].isna()][["row_id"]].merge(
        cot_rows[["row_id", "decision_id", "target_cot_token_index"]],
        on="row_id",
        how="left",
    )
    nearest = nearest_cot_matches(
        missing_exact,
        cot_main,
        max_token_tolerance=max_token_tolerance,
    )
    cot_matches = cot_exact[cot_exact["activation_id"].notna()].copy()
    if not nearest.empty:
        cot_matches = pd.concat([cot_matches, nearest], ignore_index=True, sort=False)

    matches = pd.concat([pre_matches, cot_matches], ignore_index=True, sort=False)
    match_cols = [
        "row_id",
        "activation_id",
        "activation_token_type",
        "activation_cot_token_index",
        "activation_progress",
        "activation_file",
        "activation_format",
        "activation_key",
        "activation_row_index",
        "available_layers_json",
        "hidden_dim",
        "dtype",
        "alignment_token_distance",
        "alignment_status",
    ]
    for col in match_cols:
        if col not in matches.columns:
            matches[col] = pd.NA
    aligned = behavior.merge(matches[match_cols], on="row_id", how="left")
    aligned["alignment_status"] = aligned["alignment_status"].fillna("failed_no_activation_match")

    aligned = aligned.merge(pre_pointers, on="decision_id", how="left")
    aligned = aligned.merge(post_pointers, on="decision_id", how="left")
    aligned = aligned.drop(columns=["row_id"])
    return aligned


def write_report(path: Path, aligned: pd.DataFrame, *, behavior_rows: int, max_token_tolerance: int) -> None:
    status_counts = aligned["alignment_status"].value_counts(dropna=False).to_dict()
    successful = aligned["activation_id"].notna()
    exact = aligned["alignment_status"].isin(["exact", "exact_pre_reasoning"])
    nearest = aligned["alignment_status"].eq("nearest")
    failed = ~successful
    distance_counts = (
        aligned["alignment_token_distance"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    action_counts = aligned["agent_action"].value_counts().to_dict() if "agent_action" in aligned.columns else {}
    before_action = (
        int(aligned["is_before_first_action_mention"].fillna(False).sum())
        if "is_before_first_action_mention" in aligned.columns
        else None
    )
    before_final_action = (
        int(aligned["is_before_first_final_action_mention"].fillna(False).sum())
        if "is_before_first_final_action_mention" in aligned.columns
        else None
    )
    examples = aligned[successful].head(5)

    lines = [
        "# Activation-Behavior Alignment Report",
        "",
        "## Summary",
        "",
        f"- Behavior prefix rows: {behavior_rows:,}",
        f"- Successfully aligned: {int(successful.sum()):,}",
        f"- Aligned exactly: {int(exact.sum()):,}",
        f"- Aligned by nearest token: {int(nearest.sum()):,}",
        f"- Failed: {int(failed.sum()):,}",
        f"- Max token tolerance: {max_token_tolerance}",
        "",
        "## Alignment Status Counts",
        "",
    ]
    for status, count in status_counts.items():
        lines.append(f"- `{status}`: {count:,}")

    lines.extend(["", "## Token Alignment Distance Distribution", ""])
    if distance_counts:
        for distance, count in distance_counts.items():
            lines.append(f"- `{distance}`: {count:,}")
    else:
        lines.append("- No token distances recorded.")

    lines.extend(["", "## Action Distribution", ""])
    if action_counts:
        for action, count in action_counts.items():
            lines.append(f"- `{action}`: {count:,}")
    else:
        lines.append("- `agent_action` column not available.")

    lines.extend(["", "## Strict Pre-Action Prefix Counts", ""])
    lines.append(f"- Prefixes before first action mention: {before_action if before_action is not None else 'NA'}")
    lines.append(f"- Prefixes before first final-action mention: {before_final_action if before_final_action is not None else 'NA'}")

    lines.extend(["", "## Examples", ""])
    if examples.empty:
        lines.append("- No aligned examples available.")
    else:
        cols = [
            "decision_id",
            "prefix_token_index",
            "token_type",
            "activation_id",
            "activation_token_type",
            "activation_cot_token_index",
            "alignment_status",
            "alignment_token_distance",
            "agent_action",
            "top_action",
            "commitment_to_final",
        ]
        cols = [col for col in cols if col in examples.columns]
        for _, row in examples[cols].iterrows():
            payload = {col: row[col] for col in cols}
            lines.append(f"- `{row['decision_id']}`: `{json.dumps(payload, default=str)}`")

    lines.extend(["", "## Notes", ""])
    lines.append("- `prefix_token_index == 0` is aligned to `pre_reasoning`.")
    lines.append("- `prefix_token_index > 0` is aligned to CoT `cot_token_index == prefix_token_index`, with nearest-token fallback inside tolerance.")
    lines.append("- `pre_*` and `post_*` columns provide same-decision pointers regardless of the main aligned activation row.")
    lines.append("- Activation vectors are not embedded in the aligned parquet; the table stores file and row pointers.")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    parser.add_argument("--behavior", default="outputs/prefix_commitment_metrics.parquet")
    parser.add_argument("--step_summary", default="outputs/step_commitment_summary.parquet")
    parser.add_argument("--behavior_vectors", default="outputs/behavior_vectors.parquet")
    parser.add_argument("--behavior_manifold", default="outputs/behavior_manifold.parquet")
    parser.add_argument("--output", default="outputs/activation/activation_behavior_aligned.parquet")
    parser.add_argument("--max_token_tolerance", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    activation_index_path = Path(args.activation_index)
    behavior_path = Path(args.behavior)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.parent / "alignment_report.md"

    behavior = read_behavior_with_optional_inputs(
        behavior_path,
        Path(args.step_summary),
        Path(args.behavior_vectors),
        Path(args.behavior_manifold),
    )
    behavior_rows = len(behavior)
    decisions = set(behavior["decision_id"].astype(str).unique())
    print(f"Loaded {behavior_rows:,} behavior rows across {len(decisions):,} decisions")

    activation = read_activation_index_for_decisions(activation_index_path, decisions)
    print(f"Loaded {len(activation):,} activation-index rows for behavior decisions")

    aligned = align_behavior_to_activation(
        behavior,
        activation,
        max_token_tolerance=args.max_token_tolerance,
    )
    aligned.to_parquet(output_path, index=False)
    write_report(report_path, aligned, behavior_rows=behavior_rows, max_token_tolerance=args.max_token_tolerance)

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(aligned["alignment_status"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
