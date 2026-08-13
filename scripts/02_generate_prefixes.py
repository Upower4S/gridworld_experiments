#!/usr/bin/env python3
"""Generate token-prefix rows from canonical trajectory steps."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


VALID_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas and pyarrow before running this script."
        ) from exc
    return pd, pa, pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/canonical_steps.parquet"),
        help="Canonical step-level parquet input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prefixes.parquet"),
        help="Output parquet file for generated prefixes.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/prefix_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--stride", type=int, default=5, help="Token stride between prefixes.")
    parser.add_argument(
        "--min_prefix_tokens",
        type=int,
        default=1,
        help="Minimum number of analysis tokens in a generated prefix.",
    )
    parser.add_argument(
        "--max_prefixes_per_step",
        type=int,
        default=80,
        help="Maximum generated prefixes per step. First and final prefixes are preserved.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Optional Hugging Face tokenizer name or local path for decoding token ids.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Number of prefix rows to buffer before writing a parquet row group.",
    )
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_tokenizer(tokenizer_name_or_path: str | None) -> Any | None:
    if not tokenizer_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install transformers or omit --tokenizer_name_or_path."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def token_groups(token: dict[str, Any]) -> set[str]:
    groups = token.get("token_groups", [])
    if not isinstance(groups, list):
        return set()
    return {str(group) for group in groups}


def parse_analysis_tokens(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, str) or not raw_value:
        return []
    try:
        tokens = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    if not isinstance(tokens, list):
        return []
    return [
        token
        for token in tokens
        if isinstance(token, dict) and "analysis" in token_groups(token)
    ]


def token_id(token: dict[str, Any]) -> int | None:
    value = token.get("token_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def token_ids(tokens: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for token in tokens:
        value = token_id(token)
        if value is not None:
            ids.append(value)
    return ids


def fallback_decode(tokens: list[dict[str, Any]]) -> str:
    text = "".join(str(token.get("token", "")) for token in tokens)
    return text.replace("Ġ", " ").replace("Ċ", "\n").strip()


def decode_prefix(tokens: list[dict[str, Any]], tokenizer: Any | None) -> str:
    if tokenizer is not None:
        ids = token_ids(tokens)
        if ids:
            try:
                return tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                pass
    return fallback_decode(tokens)


def normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in VALID_ACTIONS else None


def normalized_token_letters(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("Ġ", " ").replace("Ċ", "\n").upper()
    return re.sub(r"[^A-Z]", "", text)


def token_text_matches_action(token: dict[str, Any], action: str | None) -> bool:
    if action is None:
        return False
    return normalized_token_letters(token.get("token")) == action


def first_action_indices(tokens: list[dict[str, Any]], agent_action: str | None) -> tuple[int | None, int | None]:
    first_action_index: int | None = None
    first_final_action_index: int | None = None

    for index, token in enumerate(tokens):
        groups = token_groups(token)
        if "action" not in groups:
            continue
        if first_action_index is None:
            first_action_index = index
        if first_final_action_index is None and token_text_matches_action(token, agent_action):
            first_final_action_index = index
        if first_action_index is not None and first_final_action_index is not None:
            break

    return first_action_index, first_final_action_index


def evenly_thin(values: list[int], max_count: int) -> list[int]:
    if len(values) <= max_count:
        return values
    if max_count < 2:
        raise ValueError("max_prefixes_per_step must be at least 2 to preserve first and final prefixes")

    last_position = len(values) - 1
    selected_positions = {
        round(i * last_position / (max_count - 1))
        for i in range(max_count)
    }

    # Rounding can collapse nearby positions. Fill deterministically from the left.
    if len(selected_positions) < max_count:
        for position in range(len(values)):
            selected_positions.add(position)
            if len(selected_positions) == max_count:
                break

    return [values[position] for position in sorted(selected_positions)]


def prefix_lengths(num_tokens: int, stride: int, min_prefix_tokens: int, max_prefixes_per_step: int) -> list[int]:
    if num_tokens <= 0:
        return []
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if min_prefix_tokens < 1:
        raise ValueError("--min_prefix_tokens must be >= 1")
    if max_prefixes_per_step < 2:
        raise ValueError("--max_prefixes_per_step must be >= 2")

    first_length = min(min_prefix_tokens, num_tokens)
    values = list(range(first_length, num_tokens + 1, stride))
    values.extend([first_length, num_tokens])
    unique_values = sorted(set(values))
    return evenly_thin(unique_values, max_prefixes_per_step)


def build_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("prefix_id", pa.int64()),
            ("trajectory_id", pa.string()),
            ("source_file", pa.string()),
            ("step_id", pa.int64()),
            ("prefix_token_index", pa.int64()),
            ("num_analysis_tokens", pa.int64()),
            ("progress", pa.float64()),
            ("prompt_text", pa.string()),
            ("analysis_prefix_text", pa.string()),
            ("analysis_prefix_token_ids_json", pa.string()),
            ("agent_action", pa.string()),
            ("first_action_mention_token_index", pa.int64()),
            ("first_final_action_mention_token_index", pa.int64()),
            ("is_before_first_action_mention", pa.bool_()),
            ("is_before_first_final_action_mention", pa.bool_()),
        ]
    )


def write_report(
    *,
    report_path: Path,
    total_prefix_rows: int,
    num_steps: int,
    skipped_steps: int,
    action_counts: Counter[str],
    before_first_action_count: int,
    first_action_progresses: list[float],
) -> None:
    mean_prefixes_per_step = total_prefix_rows / num_steps if num_steps else 0.0
    fraction_before = before_first_action_count / total_prefix_rows if total_prefix_rows else 0.0
    median_first_action_progress = median(first_action_progresses) if first_action_progresses else None

    lines = [
        "# Prefix Report",
        "",
        f"- Total prefix rows: `{total_prefix_rows}`",
        f"- Mean prefixes per step: `{mean_prefixes_per_step:.2f}`",
        f"- Number of steps: `{num_steps}`",
        f"- Steps skipped for empty/malformed analysis tokens: `{skipped_steps}`",
        "",
        "## Action Distribution",
        "",
    ]
    for action in VALID_ACTIONS:
        lines.append(f"- `{action}`: `{action_counts.get(action, 0)}`")

    lines.extend(
        [
            "",
            "## Action Mention Prefixes",
            "",
            f"- Fraction of prefixes before first action mention: `{fraction_before:.4f}`",
        ]
    )

    if median_first_action_progress is None:
        lines.append("- Median first action mention progress: `n/a`")
    else:
        lines.append(f"- Median first action mention progress: `{median_first_action_progress:.4f}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    pd, pa, pq = require_dependencies()

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.min_prefix_tokens < 1:
        raise SystemExit("--min_prefix_tokens must be >= 1")
    if args.max_prefixes_per_step < 2:
        raise SystemExit("--max_prefixes_per_step must be >= 2")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    dataframe = pd.read_parquet(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    schema = build_schema(pa)
    writer = pq.ParquetWriter(args.output, schema=schema)

    prefix_id = 0
    batch: list[dict[str, Any]] = []
    total_prefix_rows = 0
    skipped_steps = 0
    before_first_action_count = 0
    first_action_progresses: list[float] = []
    action_counts: Counter[str] = Counter()

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        writer.write_table(table)
        batch = []

    try:
        for row in dataframe.itertuples(index=False):
            row_dict = row._asdict()
            agent_action = normalize_action(row_dict.get("agent_action"))
            action_counts.update([agent_action or ""])

            tokens = parse_analysis_tokens(row_dict.get("analysis_tokens_json"))
            num_tokens = len(tokens)
            if num_tokens == 0:
                skipped_steps += 1
                continue

            first_action_index, first_final_action_index = first_action_indices(tokens, agent_action)
            if first_action_index is not None:
                first_action_progresses.append((first_action_index + 1) / num_tokens)

            ids = token_ids(tokens)
            lengths = prefix_lengths(
                num_tokens=num_tokens,
                stride=args.stride,
                min_prefix_tokens=args.min_prefix_tokens,
                max_prefixes_per_step=args.max_prefixes_per_step,
            )

            for length in lengths:
                prefix_index = length - 1
                is_before_first_action = (
                    True if first_action_index is None else prefix_index < first_action_index
                )
                is_before_first_final_action = (
                    True if first_final_action_index is None else prefix_index < first_final_action_index
                )
                before_first_action_count += int(is_before_first_action)

                prefix_tokens = tokens[:length]
                batch.append(
                    {
                        "prefix_id": prefix_id,
                        "trajectory_id": row_dict.get("trajectory_id"),
                        "source_file": row_dict.get("source_file"),
                        "step_id": int(row_dict.get("step_id")),
                        "prefix_token_index": prefix_index,
                        "num_analysis_tokens": num_tokens,
                        "progress": length / num_tokens,
                        "prompt_text": row_dict.get("prompt_text"),
                        "analysis_prefix_text": decode_prefix(prefix_tokens, tokenizer),
                        "analysis_prefix_token_ids_json": json_dumps(ids[:length]),
                        "agent_action": agent_action,
                        "first_action_mention_token_index": first_action_index,
                        "first_final_action_mention_token_index": first_final_action_index,
                        "is_before_first_action_mention": is_before_first_action,
                        "is_before_first_final_action_mention": is_before_first_final_action,
                    }
                )
                prefix_id += 1
                total_prefix_rows += 1

                if len(batch) >= args.batch_size:
                    flush_batch()

        flush_batch()
    finally:
        writer.close()

    write_report(
        report_path=args.report,
        total_prefix_rows=total_prefix_rows,
        num_steps=len(dataframe),
        skipped_steps=skipped_steps,
        action_counts=action_counts,
        before_first_action_count=before_first_action_count,
        first_action_progresses=first_action_progresses,
    )

    print(f"Read steps: {len(dataframe)}")
    print(f"Prefix rows: {total_prefix_rows}")
    print(f"Skipped steps: {skipped_steps}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

