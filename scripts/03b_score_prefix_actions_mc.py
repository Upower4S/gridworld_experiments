#!/usr/bin/env python3
"""Optional multiple-choice interruption action scorer for analysis prefixes."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.probing import (  # noqa: E402
    ACTIONS,
    MC_SCORING_MODE,
    OPTION_LETTERS,
    average_letter_scores_to_action_space,
    deterministic_option_permutations,
    load_hf_backend,
    load_mlx_backend,
    multiple_choice_interruption_context,
    score_contexts_hf_candidates,
    score_contexts_mlx_candidates,
)


LETTER_CANDIDATES = {letter: letter for letter in OPTION_LETTERS}

OUTPUT_COLUMNS = [
    "prefix_id",
    "trajectory_id",
    "source_file",
    "step_id",
    "prefix_token_index",
    "num_analysis_tokens",
    "progress",
    "agent_action",
    "q_UP",
    "q_DOWN",
    "q_LEFT",
    "q_RIGHT",
    "logscore_UP",
    "logscore_DOWN",
    "logscore_LEFT",
    "logscore_RIGHT",
    "top_action",
    "candidate_mode",
    "scoring_mode",
    "scoring_backend",
    "option_permutation_count",
    "option_permutation_seed",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
]


def require_table_dependencies() -> Any:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas and pyarrow before running this script."
        ) from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prefix_action_scores_mc.parquet"),
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--backend", choices=["hf", "litellm", "mlx"], default="hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output parquet by skipping already scored prefix_id rows.",
    )
    parser.add_argument(
        "--num_option_permutations",
        type=int,
        default=4,
        help="Number of deterministic A/B/C/D option permutations to score and average.",
    )
    parser.add_argument(
        "--option_permutation_seed",
        type=int,
        default=0,
        help="Seed for deterministic non-identity option permutation ordering.",
    )
    parser.add_argument(
        "--include_identity_permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include A=UP, B=DOWN, C=LEFT, D=RIGHT as the first permutation.",
    )
    parser.add_argument(
        "--experts_implementation",
        choices=["auto", "eager", "grouped_mm", "batched_mm"],
        default="eager",
        help="MoE experts implementation passed to Transformers for --backend hf.",
    )
    return parser.parse_args()


def atomic_write_parquet(pd: Any, rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_existing_scores(pd: Any, output_path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    existing = pd.read_parquet(output_path)
    missing = [column for column in OUTPUT_COLUMNS if column not in existing.columns]
    if missing:
        raise SystemExit(
            f"Cannot resume from {output_path}: missing expected columns {missing}. "
            "Use --no-resume to overwrite."
        )
    return existing[OUTPUT_COLUMNS].to_dict("records")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def required_input_columns() -> list[str]:
    return [
        "prefix_id",
        "trajectory_id",
        "source_file",
        "step_id",
        "prefix_token_index",
        "num_analysis_tokens",
        "progress",
        "prompt_text",
        "analysis_prefix_text",
        "agent_action",
        "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]


def output_row_from_mc_scores(
    *,
    prefix_row: dict[str, Any],
    action_probabilities: dict[str, float],
    action_logscores: dict[str, float],
    backend: str,
    permutation_count: int,
    permutation_seed: int,
) -> dict[str, Any]:
    top_action = max(ACTIONS, key=lambda action: action_probabilities[action])
    return {
        "prefix_id": int(prefix_row["prefix_id"]),
        "trajectory_id": prefix_row["trajectory_id"],
        "source_file": prefix_row["source_file"],
        "step_id": int(prefix_row["step_id"]),
        "prefix_token_index": int(prefix_row["prefix_token_index"]),
        "num_analysis_tokens": int(prefix_row["num_analysis_tokens"]),
        "progress": float(prefix_row["progress"]),
        "agent_action": prefix_row["agent_action"],
        "q_UP": action_probabilities["UP"],
        "q_DOWN": action_probabilities["DOWN"],
        "q_LEFT": action_probabilities["LEFT"],
        "q_RIGHT": action_probabilities["RIGHT"],
        "logscore_UP": action_logscores["UP"],
        "logscore_DOWN": action_logscores["DOWN"],
        "logscore_LEFT": action_logscores["LEFT"],
        "logscore_RIGHT": action_logscores["RIGHT"],
        "top_action": top_action,
        "candidate_mode": "letter",
        "scoring_mode": MC_SCORING_MODE,
        "scoring_backend": backend,
        "option_permutation_count": permutation_count,
        "option_permutation_seed": permutation_seed,
        "is_before_first_action_mention": bool(prefix_row["is_before_first_action_mention"]),
        "is_before_first_final_action_mention": bool(prefix_row["is_before_first_final_action_mention"]),
    }


def score_letter_contexts(
    *,
    backend: str,
    tokenizer: Any,
    model: Any,
    runtime_a: Any,
    runtime_b: Any,
    contexts: list[str],
) -> list[dict[str, float]]:
    if backend == "hf":
        return score_contexts_hf_candidates(
            tokenizer=tokenizer,
            model=model,
            torch=runtime_a,
            device=runtime_b,
            contexts=contexts,
            candidates=LETTER_CANDIDATES,
        )
    if backend == "mlx":
        return score_contexts_mlx_candidates(
            tokenizer=tokenizer,
            model=model,
            mx=runtime_a,
            nn=runtime_b,
            contexts=contexts,
            candidates=LETTER_CANDIDATES,
        )
    raise SystemExit(f"Unsupported backend: {backend}")


def main() -> int:
    args = parse_args()
    if args.backend == "litellm":
        raise SystemExit(
            "LiteLLM backend is not implemented for teacher-forced multiple-choice "
            "logprob scoring here. Use --backend hf or --backend mlx."
        )
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided")
    if args.save_every < 1:
        raise SystemExit("--save_every must be >= 1")

    option_permutations = deterministic_option_permutations(
        num_permutations=args.num_option_permutations,
        seed=args.option_permutation_seed,
        include_identity=args.include_identity_permutation,
    )

    pd = require_table_dependencies()
    prefixes = pd.read_parquet(args.input)
    if args.limit is not None:
        prefixes = prefixes.head(args.limit)

    missing_columns = [
        column for column in required_input_columns()
        if column not in prefixes.columns
    ]
    if missing_columns:
        raise SystemExit(f"Input is missing expected columns: {missing_columns}")

    existing_rows = read_existing_scores(pd, args.output, args.resume)
    done_prefix_ids = {int(row["prefix_id"]) for row in existing_rows}
    rows_to_score = prefixes[~prefixes["prefix_id"].isin(done_prefix_ids)]
    rows_to_score = rows_to_score.sort_values("prefix_id")

    if args.backend == "hf":
        tokenizer, model, runtime_a, runtime_b = load_hf_backend(
            model_name_or_path=args.model_name_or_path,
            device_arg=args.device,
            dtype_arg=args.dtype,
            experts_implementation=args.experts_implementation,
        )
    elif args.backend == "mlx":
        tokenizer, model, runtime_a, runtime_b = load_mlx_backend(args.model_name_or_path)
    else:
        raise SystemExit(f"Unsupported backend: {args.backend}")

    scored_rows = list(existing_rows)
    pending_since_save = 0
    prefix_records = rows_to_score.to_dict("records")
    total = len(prefix_records)
    print(f"Input rows: {len(prefixes)}")
    print(f"Already scored: {len(existing_rows)}")
    print(f"Rows to score: {total}")
    print(f"Option permutations: {len(option_permutations)}")

    for start in range(0, total, args.batch_size):
        batch_records = prefix_records[start : start + args.batch_size]
        contexts: list[str] = []
        for record in batch_records:
            for option_actions in option_permutations:
                contexts.append(
                    multiple_choice_interruption_context(
                        prompt_text=clean_text(record["prompt_text"]),
                        analysis_prefix_text=clean_text(record["analysis_prefix_text"]),
                        option_actions=option_actions,
                    )
                )

        letter_logscore_rows = score_letter_contexts(
            backend=args.backend,
            tokenizer=tokenizer,
            model=model,
            runtime_a=runtime_a,
            runtime_b=runtime_b,
            contexts=contexts,
        )

        for index, record in enumerate(batch_records):
            begin = index * len(option_permutations)
            end = begin + len(option_permutations)
            action_probabilities, action_logscores = average_letter_scores_to_action_space(
                letter_logscore_rows=letter_logscore_rows[begin:end],
                option_permutations=option_permutations,
                temperature=args.temperature,
            )
            scored_rows.append(
                output_row_from_mc_scores(
                    prefix_row=record,
                    action_probabilities=action_probabilities,
                    action_logscores=action_logscores,
                    backend=args.backend,
                    permutation_count=len(option_permutations),
                    permutation_seed=args.option_permutation_seed,
                )
            )
            pending_since_save += 1

        if pending_since_save >= args.save_every:
            scored_rows.sort(key=lambda row: int(row["prefix_id"]))
            atomic_write_parquet(pd, scored_rows, args.output)
            print(f"Saved {len(scored_rows)} rows")
            pending_since_save = 0

    scored_rows.sort(key=lambda row: int(row["prefix_id"]))
    atomic_write_parquet(pd, scored_rows, args.output)
    print(f"Wrote {args.output} with {len(scored_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
