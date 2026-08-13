#!/usr/bin/env python3
"""Optional Step 12: build richer gridworld task concepts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_OUTPUT_DIR  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_parquet_checked, save_markdown  # noqa: E402
from src.conceptual_belief_space.task_concepts import (  # noqa: E402
    ACTION_DERIVED_CONCEPTS,
    STATE_CONCEPTS,
    action_mask_columns,
    build_step_concept_table,
    concept_summary,
    join_concepts_to_beliefs,
    model_belief_columns,
)


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


def optional_read_parquet(path: Path, notes: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        notes.append(f"Optional canonical step table missing at `{rel(path)}`; only action-category concepts can be built.")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        notes.append(f"Failed to read canonical step table `{rel(path)}`: {type(exc).__name__}: {exc}")
        return None


def compact(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4g}"


def select_output_columns(df: pd.DataFrame) -> list[str]:
    metadata = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "canonical_agent_action",
        "q_UP",
        "q_DOWN",
        "q_LEFT",
        "q_RIGHT",
    ]
    state = [
        "has_grid_state",
        "parse_status",
        "carrying_key",
        "grid_height",
        "grid_width",
        "agent_row",
        "agent_col",
        "goal_row",
        "goal_col",
        "key_row",
        "key_col",
        "door_row",
        "door_col",
        "door_present",
        "key_present",
        "goal_reachable_without_key",
        "door_requires_key",
        "key_required",
        "door_relevant",
        "shortest_path_length",
        "shortest_path_actions_json",
        "num_shortest_path_actions",
    ]
    action_masks = action_mask_columns()
    selected = []
    for prefix in ["env_agent_action_", "env_top_action_", "model_belief_"]:
        for concept in ACTION_DERIVED_CONCEPTS:
            selected.append(prefix + concept)
    strict_cols = [
        "num_cot_tokens",
        "first_action_mention_token_index",
        "first_final_action_mention_token_index",
        "strict_before_any_action_mention",
        "strict_before_final_action_mention",
        "buffered_before_any_action_mention",
        "buffered_before_final_action_mention",
    ]
    ordered = metadata + state + action_masks + selected + strict_cols
    return [col for col in ordered if col in df.columns]


def summarize_scores(scores: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(scores)),
        "decisions": int(scores["decision_id"].nunique()) if "decision_id" in scores else 0,
        "trajectories": int(scores["trajectory_id"].nunique()) if "trajectory_id" in scores else 0,
        "grid_rows": int(scores["has_grid_state"].fillna(False).sum()) if "has_grid_state" in scores else 0,
    }
    if "env_agent_action_shortest_path_action" in scores.columns:
        out["agent_action_shortest_path_rate"] = float(
            pd.to_numeric(scores["env_agent_action_shortest_path_action"], errors="coerce").mean()
        )
    if "env_top_action_shortest_path_action" in scores.columns:
        out["top_action_shortest_path_rate"] = float(
            pd.to_numeric(scores["env_top_action_shortest_path_action"], errors="coerce").mean()
        )
    if "model_belief_shortest_path_action" in scores.columns:
        out["mean_model_belief_shortest_path_action"] = float(
            pd.to_numeric(scores["model_belief_shortest_path_action"], errors="coerce").mean()
        )
    return out


def write_report(
    path: Path,
    *,
    beliefs_path: Path,
    canonical_steps_path: Path,
    output_path: Path,
    beliefs_rows: int,
    step_rows: int,
    step_concept_rows: int,
    scores: pd.DataFrame,
    notes: list[str],
) -> None:
    summary = summarize_scores(scores)
    concept_stats = concept_summary(scores)
    parse_counts = (
        scores[["decision_id", "parse_status"]]
        .drop_duplicates("decision_id")
        .groupby("parse_status")
        .size()
        .rename("decisions")
        .reset_index()
        if {"decision_id", "parse_status"}.issubset(scores.columns)
        else pd.DataFrame()
    )
    state_counts = []
    for concept in STATE_CONCEPTS:
        if concept in scores.columns:
            values = scores[["decision_id", concept]].drop_duplicates("decision_id")
            state_counts.append(
                {
                    "concept": concept,
                    "mean_over_decisions": float(pd.to_numeric(values[concept], errors="coerce").mean()),
                    "true_decisions": int(pd.to_numeric(values[concept], errors="coerce").fillna(0).sum()),
                }
            )
    state_counts_df = pd.DataFrame(state_counts)

    lines = [
        "# Task Concepts",
        "",
        "Optional Step 12 builds richer gridworld task concepts beyond the four action labels.",
        "",
        "## Important Distinction",
        "",
        "- Deterministic environment concepts are computed from the parsed grid, action validity, key/door state, and shortest-path search.",
        "- Model belief concepts are q-weighted sums over action-derived deterministic masks, for example `model_belief_shortest_path_action = sum_a q_a * env_shortest_path_action_a`.",
        "- These two concept types are related but not interchangeable. Reports should not mix them without stating which type is being used.",
        "- The four-action analysis remains the base result. These richer concepts are optional geometry features to reduce dependence on only four action labels.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Belief table: `{rel(beliefs_path)}`",
        f"- Canonical step table: `{rel(canonical_steps_path)}`",
        f"- Output task concept scores: `{rel(output_path)}`",
        "",
        "## Counts",
        "",
        f"- Input belief rows: `{beliefs_rows}`",
        f"- Canonical step rows loaded: `{step_rows}`",
        f"- Deterministic step concept rows: `{step_concept_rows}`",
        f"- Output prefix rows: `{summary['rows']}`",
        f"- Decisions: `{summary['decisions']}`",
        f"- Trajectories: `{summary['trajectories']}`",
        f"- Prefix rows with parsed grid state: `{summary['grid_rows']}`",
        "",
        "## Concept Sets",
        "",
        "- Action-derived concepts: " + ", ".join(f"`{c}`" for c in ACTION_DERIVED_CONCEPTS),
        "- Deterministic state concepts: " + ", ".join(f"`{c}`" for c in STATE_CONCEPTS),
        "",
        "## Shortest-path Diagnostics",
        "",
        f"- Agent action is a shortest-path action, mean over prefixes: `{compact(summary.get('agent_action_shortest_path_rate'))}`",
        f"- Current model top action is a shortest-path action, mean over prefixes: `{compact(summary.get('top_action_shortest_path_rate'))}`",
        f"- Mean q-weighted belief in shortest-path actions: `{compact(summary.get('mean_model_belief_shortest_path_action'))}`",
        "",
        "## Parse Status by Decision",
        "",
        parse_counts.to_markdown(index=False) if not parse_counts.empty else "_No parse status table available._",
        "",
        "## State Concept Counts by Decision",
        "",
        state_counts_df.to_markdown(index=False) if not state_counts_df.empty else "_No state concept counts available._",
        "",
        "## Concept Summary",
        "",
        concept_stats.to_markdown(index=False),
        "",
        "## Example Rows",
        "",
        scores[
            [
                col
                for col in [
                    "decision_id",
                    "prefix_token_index",
                    "agent_action",
                    "top_action",
                    "carrying_key",
                    "door_relevant",
                    "key_required",
                    "goal_reachable_without_key",
                    "shortest_path_actions_json",
                    "model_belief_shortest_path_action",
                    "model_belief_move_toward_goal",
                    "model_belief_valid_action",
                ]
                if col in scores.columns
            ]
        ]
        .head(8)
        .to_markdown(index=False),
        "",
        "## Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No warnings.")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The shortest-path search is an environment approximation based on the parsed grid and known key/door rules.",
            "- Door passability is modeled as possible after the key is carried. This captures the task rule at the level needed for first-action concepts, but it is not a full simulator trace.",
            "- Concept beliefs are induced from action probabilities; they are not independent model readouts.",
            "- Richer task concepts increase the number of geometry targets, but they do not replace the base four-action analysis.",
        ]
    )
    save_markdown(path, "\n".join(lines))


def run(
    *,
    beliefs_path: Path,
    canonical_steps_path: Path,
    output_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    beliefs = read_parquet_checked(
        beliefs_path,
        required_columns=["trajectory_id", "step_id", "agent_action", "q_UP", "q_DOWN", "q_LEFT", "q_RIGHT"],
    ).reset_index(drop=True)
    notes: list[str] = []
    steps = optional_read_parquet(canonical_steps_path, notes)
    step_concepts = None
    if steps is not None:
        required = {"trajectory_id", "step_id", "grid_state_text", "carrying_key"}
        missing = sorted(required - set(steps.columns))
        if missing:
            notes.append(f"Canonical step table is missing columns {missing}; grid concepts are unavailable.")
        else:
            step_concepts = build_step_concept_table(steps)

    scores = join_concepts_to_beliefs(beliefs, step_concepts)
    output_cols = select_output_columns(scores)
    output_scores = scores[output_cols].copy()
    ensure_dir(output_path.parent)
    output_scores.to_parquet(output_path, index=False)
    write_report(
        report_path,
        beliefs_path=beliefs_path,
        canonical_steps_path=canonical_steps_path,
        output_path=output_path,
        beliefs_rows=len(beliefs),
        step_rows=0 if steps is None else len(steps),
        step_concept_rows=0 if step_concepts is None else len(step_concepts),
        scores=output_scores,
        notes=notes,
    )
    return output_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--canonical_steps", default=Path("outputs/canonical_steps.parquet"))
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR / "task_concepts/task_concept_scores.parquet")
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "task_concepts/12_task_concept_report.md")
    args = parser.parse_args()

    scores = run(
        beliefs_path=resolve_repo_path(args.beliefs),
        canonical_steps_path=resolve_repo_path(args.canonical_steps),
        output_path=resolve_repo_path(args.output),
        report_path=resolve_repo_path(args.report),
    )
    print(f"Wrote {rel(args.output)} with {len(scores)} rows")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
