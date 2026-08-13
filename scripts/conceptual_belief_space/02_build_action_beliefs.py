#!/usr/bin/env python3
"""Step 2: build prefix-level action belief vectors."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.beliefs import (  # noqa: E402
    action_probability_columns,
    clr_transform,
    compute_commitment_to_action,
    compute_entropy,
    log_q_transform,
    normalize_action_probs,
    sqrt_q_transform,
    stable_softmax,
)
from src.conceptual_belief_space.config import ACTIONS, EPS  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown  # noqa: E402


Q_COLS = action_probability_columns()
LOGSCORE_COLS = [f"logscore_{action}" for action in ACTIONS]
LOG_Q_COLS = [f"log_q_{action}" for action in ACTIONS]
CLR_COLS = [f"clr_{action}" for action in ACTIONS]
SQRT_Q_COLS = [f"sqrt_q_{action}" for action in ACTIONS]
FIRST_ANY_COL = "first_action_mention_token_index"
FIRST_FINAL_COL = "first_final_action_mention_token_index"
STRICT_ANY_COL = "strict_before_any_action_mention"
STRICT_FINAL_COL = "strict_before_final_action_mention"
BUFFERED_ANY_COL = "buffered_before_any_action_mention"
BUFFERED_FINAL_COL = "buffered_before_final_action_mention"


def resolve_repo_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def rel(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def make_decision_id(df: pd.DataFrame) -> pd.Series:
    if "decision_id" in df.columns and df["decision_id"].notna().all():
        return df["decision_id"].astype(str)
    if {"trajectory_id", "step_id"}.issubset(df.columns):
        return df["trajectory_id"].astype(str) + "::" + df["step_id"].astype(str)
    raise ValueError("behavior table needs decision_id or trajectory_id + step_id")


def construct_probabilities(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create normalized q columns, preferring existing probabilities."""
    has_q = all(col in df.columns for col in Q_COLS)
    has_logscores = all(col in df.columns for col in LOGSCORE_COLS)
    diagnostics: dict[str, Any] = {
        "used_source": None,
        "has_q_columns": has_q,
        "has_logscore_columns": has_logscores,
        "max_abs_q_vs_logscore_softmax_diff": None,
        "mean_abs_q_vs_logscore_softmax_diff": None,
    }

    if not has_q and not has_logscores:
        raise ValueError("behavior table must contain q_* columns or logscore_* columns")

    out = df.copy()
    if has_q:
        q_raw = out[Q_COLS].to_numpy(dtype=float)
        if not np.isfinite(q_raw).all():
            raise ValueError("q columns contain NaN or inf")
        if (q_raw < -1e-9).any():
            raise ValueError("q columns contain negative values")
        row_sums = q_raw.sum(axis=1)
        if (row_sums <= 0).any():
            raise ValueError("q columns contain rows with non-positive probability sum")
        out = normalize_action_probs(out)
        diagnostics["used_source"] = "q_columns"
    else:
        log_scores = out[LOGSCORE_COLS].to_numpy(dtype=float)
        if not np.isfinite(log_scores).all():
            raise ValueError("logscore columns contain NaN or inf")
        q = stable_softmax(log_scores, axis=1)
        for i, col in enumerate(Q_COLS):
            out[col] = q[:, i]
        diagnostics["used_source"] = "logscore_columns"

    if has_logscores:
        log_scores = out[LOGSCORE_COLS].to_numpy(dtype=float)
        if np.isfinite(log_scores).all():
            q_from_logscores = stable_softmax(log_scores, axis=1)
            q_preferred = out[Q_COLS].to_numpy(dtype=float)
            diff = np.abs(q_preferred - q_from_logscores)
            diagnostics["max_abs_q_vs_logscore_softmax_diff"] = float(diff.max())
            diagnostics["mean_abs_q_vs_logscore_softmax_diff"] = float(diff.mean())

    return out, diagnostics


def merge_mention_indices(df: pd.DataFrame, io_map: dict[str, Any], root: Path) -> tuple[pd.DataFrame, str | None]:
    """Add first mention token indices from a compatible table when possible."""
    needed = [col for col in [FIRST_ANY_COL, FIRST_FINAL_COL] if col not in df.columns]
    if not needed:
        return df, None

    candidate_paths: list[str] = []
    strict_map = io_map.get("strict_columns_found", {})
    for col in [FIRST_ANY_COL, FIRST_FINAL_COL]:
        for path in strict_map.get(col, []):
            if path not in candidate_paths:
                candidate_paths.append(path)

    behavior_path = io_map.get("behavior_table_path")
    scored_candidates: list[tuple[int, str]] = []
    for path in candidate_paths:
        if path == behavior_path:
            continue
        lower = path.lower()
        score = 0
        if "prefixes_10_trajectories" in lower:
            score += 100
        if "prefixes.parquet" in lower:
            score += 80
        if "diagnostics" in lower or "followups" in lower:
            score -= 30
        scored_candidates.append((score, path))

    for _, path in sorted(scored_candidates, reverse=True):
        full_path = resolve_repo_path(path, root)
        if not full_path.exists() or full_path.suffix.lower() not in {".parquet", ".csv"}:
            continue
        try:
            candidate = (
                pd.read_parquet(full_path)
                if full_path.suffix.lower() == ".parquet"
                else pd.read_csv(full_path)
            )
        except Exception:
            continue

        exact_cols = [col for col in [FIRST_ANY_COL, FIRST_FINAL_COL] if col in candidate.columns]
        if not exact_cols:
            continue

        if "prefix_id" in df.columns and "prefix_id" in candidate.columns:
            keys = ["prefix_id"]
        elif {"trajectory_id", "step_id", "prefix_token_index"}.issubset(df.columns) and {
            "trajectory_id",
            "step_id",
            "prefix_token_index",
        }.issubset(candidate.columns):
            keys = ["trajectory_id", "step_id", "prefix_token_index"]
        else:
            continue

        merge_cols = keys + exact_cols
        candidate_small = candidate[merge_cols].drop_duplicates(keys)
        before = len(df)
        merged = df.merge(candidate_small, how="left", on=keys, suffixes=("", "_from_mentions"))
        if len(merged) != before:
            continue
        filled = sum(int(merged[col].notna().sum()) for col in exact_cols)
        if filled > 0:
            return merged, path

    return df, None


def strict_before(prefix_index: pd.Series, first_index: pd.Series) -> pd.Series:
    known = first_index.notna()
    return pd.Series(np.where(known, prefix_index < first_index, True), index=prefix_index.index).astype(bool)


def add_strict_columns(df: pd.DataFrame, buffer_tokens: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    diagnostics: dict[str, Any] = {
        "buffer_tokens": buffer_tokens,
        "strict_source": {},
    }
    prefix = pd.to_numeric(out["prefix_token_index"], errors="coerce")

    if STRICT_ANY_COL in out.columns:
        out[STRICT_ANY_COL] = out[STRICT_ANY_COL].astype(bool)
        diagnostics["strict_source"][STRICT_ANY_COL] = "existing"
    elif FIRST_ANY_COL in out.columns:
        first_any = pd.to_numeric(out[FIRST_ANY_COL], errors="coerce")
        out[STRICT_ANY_COL] = strict_before(prefix, first_any)
        diagnostics["strict_source"][STRICT_ANY_COL] = FIRST_ANY_COL
    elif "is_before_first_action_mention" in out.columns:
        out[STRICT_ANY_COL] = out["is_before_first_action_mention"].astype(bool)
        diagnostics["strict_source"][STRICT_ANY_COL] = "is_before_first_action_mention"

    if STRICT_FINAL_COL in out.columns:
        out[STRICT_FINAL_COL] = out[STRICT_FINAL_COL].astype(bool)
        diagnostics["strict_source"][STRICT_FINAL_COL] = "existing"
    elif FIRST_FINAL_COL in out.columns:
        first_final = pd.to_numeric(out[FIRST_FINAL_COL], errors="coerce")
        out[STRICT_FINAL_COL] = strict_before(prefix, first_final)
        diagnostics["strict_source"][STRICT_FINAL_COL] = FIRST_FINAL_COL
    elif "is_before_first_final_action_mention" in out.columns:
        out[STRICT_FINAL_COL] = out["is_before_first_final_action_mention"].astype(bool)
        diagnostics["strict_source"][STRICT_FINAL_COL] = "is_before_first_final_action_mention"

    if FIRST_ANY_COL in out.columns:
        first_any = pd.to_numeric(out[FIRST_ANY_COL], errors="coerce")
        known = first_any.notna()
        out[BUFFERED_ANY_COL] = pd.Series(
            np.where(known, prefix < first_any - buffer_tokens, True),
            index=out.index,
        ).astype(bool)
        diagnostics["strict_source"][BUFFERED_ANY_COL] = f"{FIRST_ANY_COL} - {buffer_tokens}"

    if FIRST_FINAL_COL in out.columns:
        first_final = pd.to_numeric(out[FIRST_FINAL_COL], errors="coerce")
        known = first_final.notna()
        out[BUFFERED_FINAL_COL] = pd.Series(
            np.where(known, prefix < first_final - buffer_tokens, True),
            index=out.index,
        ).astype(bool)
        diagnostics["strict_source"][BUFFERED_FINAL_COL] = f"{FIRST_FINAL_COL} - {buffer_tokens}"

    return out, diagnostics


def add_belief_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q = out[Q_COLS]
    log_q = log_q_transform(q, eps=EPS)
    clr = clr_transform(q, eps=EPS)
    sqrt_q = sqrt_q_transform(q)
    raw_entropy = compute_entropy(q, eps=EPS)
    normalized_entropy = raw_entropy / math.log(len(ACTIONS))

    for i, action in enumerate(ACTIONS):
        out[f"log_q_{action}"] = log_q[:, i]
        out[f"clr_{action}"] = clr[:, i]
        out[f"sqrt_q_{action}"] = sqrt_q[:, i]

    out["entropy"] = normalized_entropy
    out["entropy_commitment"] = 1.0 - normalized_entropy
    out["top_action"] = np.asarray(ACTIONS, dtype=object)[np.argmax(out[Q_COLS].to_numpy(dtype=float), axis=1)]
    sorted_q = np.sort(out[Q_COLS].to_numpy(dtype=float), axis=1)
    out["margin_commitment"] = sorted_q[:, -1] - sorted_q[:, -2]
    out["commitment_to_final"] = compute_commitment_to_action(out[Q_COLS], out["agent_action"].astype(str))
    return out


def validate_output(df: pd.DataFrame, tolerance: float) -> None:
    q = df[Q_COLS].to_numpy(dtype=float)
    sums = q.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"q rows do not sum to 1 within tolerance {tolerance}; max error={max_error}")

    numeric_cols = Q_COLS + LOG_Q_COLS + CLR_COLS + SQRT_Q_COLS + [
        "entropy",
        "entropy_commitment",
        "commitment_to_final",
        "margin_commitment",
    ]
    values = df[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_cols = [col for col in numeric_cols if not np.isfinite(df[col].to_numpy(dtype=float)).all()]
        raise ValueError(f"non-finite values found in output columns: {bad_cols}")

    invalid_actions = sorted(set(df["agent_action"].dropna().astype(str)) - set(ACTIONS))
    if invalid_actions:
        raise ValueError(f"invalid agent_action labels: {invalid_actions}")

    expected_row_id = np.arange(len(df))
    if not np.array_equal(df["row_id"].to_numpy(dtype=int), expected_row_id):
        raise ValueError("row_id does not match row order")


def final_prefix_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["decision_id", "prefix_token_index", "row_id"])
        .groupby("decision_id", as_index=False, group_keys=False)
        .tail(1)
    )


def series_summary(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def truncate(value: Any, limit: int = 80) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_report(
    df: pd.DataFrame,
    output_path: Path,
    behavior_path: Path,
    probability_diagnostics: dict[str, Any],
    strict_diagnostics: dict[str, Any],
    mention_source: str | None,
) -> str:
    n_rows = len(df)
    n_decisions = int(df["decision_id"].nunique())
    n_trajectories = int(df["trajectory_id"].nunique()) if "trajectory_id" in df.columns else 0
    decision_actions = df.drop_duplicates("decision_id")["agent_action"]
    prefix_actions = df["agent_action"]
    prompt_rows = df.loc[df["prefix_token_index"] == 0]
    final_rows = final_prefix_rows(df)

    sample_cols = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "q_UP",
        "q_DOWN",
        "q_LEFT",
        "q_RIGHT",
        "commitment_to_final",
        "entropy_commitment",
    ]
    sample_cols = [col for col in sample_cols if col in df.columns]
    sample_rows = []
    for row in df[sample_cols].head(5).to_dict(orient="records"):
        sample_rows.append({key: truncate(value) for key, value in row.items()})

    lines = [
        "# Action Belief Vectors",
        "",
        "Step 2 builds a clean prefix-level action belief table over `UP`, `DOWN`, `LEFT`, and `RIGHT`.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Behavior input: `{rel(behavior_path, REPO_ROOT)}`",
        f"- Output parquet: `{rel(output_path, REPO_ROOT)}`",
        f"- Probability source: `{probability_diagnostics['used_source']}`",
        f"- Mention index source: `{mention_source or 'none'}`",
        "",
        "## Counts",
        "",
        f"- Rows: `{n_rows}`",
        f"- Decisions: `{n_decisions}`",
        f"- Trajectories: `{n_trajectories}`",
        "",
        "## Probability Validation",
        "",
        f"- Existing q columns: `{probability_diagnostics['has_q_columns']}`",
        f"- Existing logscore columns: `{probability_diagnostics['has_logscore_columns']}`",
        f"- Max abs q vs logscore-softmax diff: `{probability_diagnostics['max_abs_q_vs_logscore_softmax_diff']}`",
        f"- Mean abs q vs logscore-softmax diff: `{probability_diagnostics['mean_abs_q_vs_logscore_softmax_diff']}`",
        "",
        "## Action Counts by Decision",
        "",
    ]
    for action, count in value_counts_dict(decision_actions).items():
        lines.append(f"- `{action}`: `{count}`")

    lines.extend(["", "## Action Counts by Prefix", ""])
    for action, count in value_counts_dict(prefix_actions).items():
        lines.append(f"- `{action}`: `{count}`")

    prompt_mean = None if prompt_rows.empty else float(prompt_rows["commitment_to_final"].mean())
    final_summary = series_summary(final_rows["commitment_to_final"])
    final_top_match = float((final_rows["top_action"] == final_rows["agent_action"]).mean()) if len(final_rows) else float("nan")

    lines.extend(
        [
            "",
            "## Commitment Summaries",
            "",
            f"- Mean prompt-only commitment (`prefix_token_index == 0`): `{prompt_mean}`",
            f"- Final-prefix commitment count: `{final_summary['count']}`",
            f"- Final-prefix commitment mean: `{final_summary['mean']}`",
            f"- Final-prefix commitment median: `{final_summary['median']}`",
            f"- Final-prefix commitment min: `{final_summary['min']}`",
            f"- Final-prefix commitment max: `{final_summary['max']}`",
            f"- Final-prefix top action matches agent action: `{final_top_match}`",
            "",
            "## Strict Pre-action Counts",
            "",
        ]
    )
    for col in [STRICT_ANY_COL, STRICT_FINAL_COL, BUFFERED_ANY_COL, BUFFERED_FINAL_COL]:
        if col in df.columns:
            lines.append(f"- `{col}`: `{int(df[col].sum())}` rows")
        else:
            lines.append(f"- `{col}`: not computed")

    lines.extend(["", "Strict column sources:", ""])
    for col, source in strict_diagnostics.get("strict_source", {}).items():
        lines.append(f"- `{col}`: `{source}`")

    lines.extend(["", "## Sample Rows", "", "```json"])
    import json

    lines.append(json.dumps(sample_rows, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def build_action_beliefs(
    io_map_path: Path,
    output_path: Path,
    report_path: Path,
    buffer_tokens: int,
    tolerance: float,
) -> pd.DataFrame:
    io_map = read_json(io_map_path)
    if not isinstance(io_map, dict):
        raise ValueError(f"{io_map_path} did not contain a JSON object")

    behavior_path = resolve_repo_path(str(io_map["behavior_table_path"]), REPO_ROOT)
    df = read_parquet_checked(behavior_path)
    df, probability_diagnostics = construct_probabilities(df)
    df, mention_source = merge_mention_indices(df, io_map, REPO_ROOT)

    invalid_actions = sorted(set(df["agent_action"].dropna().astype(str)) - set(ACTIONS))
    if invalid_actions:
        raise ValueError(f"invalid agent_action labels before transform: {invalid_actions}")

    df["decision_id"] = make_decision_id(df)
    df, strict_diagnostics = add_strict_columns(df, buffer_tokens=buffer_tokens)
    df = add_belief_transforms(df)

    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    if "num_cot_tokens" not in df.columns and "num_analysis_tokens" in df.columns:
        df["num_cot_tokens"] = df["num_analysis_tokens"]

    output_cols = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        *Q_COLS,
        *LOG_Q_COLS,
        *CLR_COLS,
        *SQRT_Q_COLS,
        "entropy",
        "entropy_commitment",
        "commitment_to_final",
        "margin_commitment",
        "num_cot_tokens",
        FIRST_ANY_COL,
        FIRST_FINAL_COL,
        STRICT_ANY_COL,
        STRICT_FINAL_COL,
        BUFFERED_ANY_COL,
        BUFFERED_FINAL_COL,
    ]
    output_cols = [col for col in output_cols if col in df.columns]
    out = df[output_cols].copy()
    validate_output(out, tolerance=tolerance)

    ensure_dir(output_path.parent)
    out.to_parquet(output_path, index=False)
    report = build_report(
        out,
        output_path=output_path,
        behavior_path=behavior_path,
        probability_diagnostics=probability_diagnostics,
        strict_diagnostics=strict_diagnostics,
        mention_source=mention_source,
    )
    save_markdown(report_path, report)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_map", default="outputs/conceptual_belief_space/io_map.json")
    parser.add_argument("--output", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument(
        "--report",
        default="outputs/conceptual_belief_space/02_action_belief_vectors_report.md",
    )
    parser.add_argument("--buffer_tokens", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    io_map_path = resolve_repo_path(args.io_map, REPO_ROOT)
    output_path = resolve_repo_path(args.output, REPO_ROOT)
    report_path = resolve_repo_path(args.report, REPO_ROOT)
    out = build_action_beliefs(
        io_map_path=io_map_path,
        output_path=output_path,
        report_path=report_path,
        buffer_tokens=args.buffer_tokens,
        tolerance=args.tolerance,
    )
    print(f"Wrote {rel(output_path, REPO_ROOT)} with {len(out)} rows")
    print(f"Wrote {rel(report_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
