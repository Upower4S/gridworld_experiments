#!/usr/bin/env python3
"""Fast local MLX scoring for final-JSON action candidates using KV cache.

This is a local-only alternative to the Together scorer.  It scores the same
forced final-JSON action candidates as ``03_score_prefix_actions_json_final.py``
but reuses a KV cache within each decision step:

    prompt + analysis header + generated analysis prefix

For each prefix, the script clones that cache, appends the final JSON prefix,
and scores UP/DOWN/LEFT/RIGHT.  By default it tokenizes the same prefix text as
the reference MLX scorer, then uses common-prefix cache reuse plus cache trim to
avoid recomputing the whole prompt.  It never sends data off-machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FINAL_JSON_PREFIX = '<|end|><|start|>assistant<|channel|>final<|message|>{\n  "action": "'

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
    "scoring_backend",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/prefixes.parquet"))
    p.add_argument("--output", type=Path, default=Path("outputs/prefix_action_scores_mlx_full.parquet"))
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--candidate_mode", choices=["bare_action", "full_json"], default="bare_action")
    p.add_argument(
        "--prefix_tokenization",
        choices=["text", "analysis_token_ids"],
        default="text",
        help="Use text to match 03_score_prefix_actions_json_final.py exactly; analysis_token_ids is faster but may differ at BPE boundaries.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_decisions", type=int, default=None)
    return p.parse_args()


def require_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc
    return pd, mx, nn, load, make_prompt_cache, pyarrow


def encode_without_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def candidate_completion(action: str, candidate_mode: str) -> str:
    if candidate_mode == "bare_action":
        return action
    if candidate_mode == "full_json":
        return f'{action}"\n}}'
    raise ValueError(candidate_mode)


def stable_softmax(logscores: dict[str, float], temperature: float) -> dict[str, float]:
    vals = np.asarray([logscores[a] for a in ACTIONS], dtype=np.float64) / max(float(temperature), 1e-12)
    vals = vals - np.max(vals)
    probs = np.exp(vals)
    probs = probs / probs.sum()
    return {a: float(probs[i]) for i, a in enumerate(ACTIONS)}


def atomic_write_parquet(pd: Any, rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=str(output_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_existing(pd: Any, output_path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    existing = pd.read_parquet(output_path)
    missing = [c for c in OUTPUT_COLUMNS if c not in existing.columns]
    if missing:
        raise SystemExit(f"Cannot resume from {output_path}: missing columns {missing}")
    return existing[OUTPUT_COLUMNS].to_dict("records")


def clone_cache(cache: list[Any]) -> list[Any]:
    return [c.from_state(c.state, c.meta_state) for c in cache]


def trim_cache(cache: list[Any], n: int) -> None:
    if n <= 0:
        return
    for c in cache:
        c.trim(n)


def feed_ids(model: Any, mx: Any, cache: list[Any], ids: list[int]) -> Any:
    if not ids:
        return None
    logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    return logits


def score_single_token_candidates_from_logits(
    *,
    mx: Any,
    nn: Any,
    logits: Any,
    encoded_candidates: dict[str, list[int]],
) -> dict[str, float]:
    log_probs = nn.log_softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
    scores = {action: float(log_probs[ids[0]].item()) for action, ids in encoded_candidates.items()}
    mx.eval(log_probs)
    return scores


def score_sequence_candidate(
    *,
    model: Any,
    mx: Any,
    nn: Any,
    cache: list[Any],
    context_logits: Any,
    candidate_ids: list[int],
) -> float:
    local_cache = clone_cache(cache)
    logits = context_logits
    total = 0.0
    for i, token_id in enumerate(candidate_ids):
        log_probs = nn.log_softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        total += float(log_probs[token_id].item())
        mx.eval(log_probs)
        if i != len(candidate_ids) - 1:
            logits = feed_ids(model, mx, local_cache, [token_id])
    return total


def output_row(prefix_row: dict[str, Any], logscores: dict[str, float], candidate_mode: str, temperature: float) -> dict[str, Any]:
    probs = stable_softmax(logscores, temperature)
    top_action = max(ACTIONS, key=lambda a: probs[a])
    return {
        "prefix_id": int(prefix_row["prefix_id"]),
        "trajectory_id": prefix_row["trajectory_id"],
        "source_file": prefix_row["source_file"],
        "step_id": int(prefix_row["step_id"]),
        "prefix_token_index": int(prefix_row["prefix_token_index"]),
        "num_analysis_tokens": int(prefix_row["num_analysis_tokens"]),
        "progress": float(prefix_row["progress"]),
        "agent_action": prefix_row["agent_action"],
        "q_UP": probs["UP"],
        "q_DOWN": probs["DOWN"],
        "q_LEFT": probs["LEFT"],
        "q_RIGHT": probs["RIGHT"],
        "logscore_UP": float(logscores["UP"]),
        "logscore_DOWN": float(logscores["DOWN"]),
        "logscore_LEFT": float(logscores["LEFT"]),
        "logscore_RIGHT": float(logscores["RIGHT"]),
        "top_action": top_action,
        "candidate_mode": candidate_mode,
        "scoring_backend": "mlx_cached",
        "is_before_first_action_mention": bool(prefix_row["is_before_first_action_mention"]),
        "is_before_first_final_action_mention": bool(prefix_row["is_before_first_final_action_mention"]),
    }


def analysis_ids_from_row(row: dict[str, Any]) -> list[int]:
    raw = row.get("analysis_prefix_token_ids_json", "[]")
    if raw is None:
        return []
    ids = json.loads(raw)
    return [int(x) for x in ids]


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> int:
    args = parse_args()
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if args.save_every < 1:
        raise SystemExit("--save_every must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    pd, mx, nn, load, make_prompt_cache, _ = require_dependencies()
    prefixes = pd.read_parquet(args.input)
    if args.limit is not None:
        prefixes = prefixes.head(args.limit)
    required = [
        "prefix_id",
        "trajectory_id",
        "source_file",
        "step_id",
        "prefix_token_index",
        "num_analysis_tokens",
        "progress",
        "prompt_text",
        "analysis_prefix_text",
        "analysis_prefix_token_ids_json",
        "agent_action",
        "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    missing = [c for c in required if c not in prefixes.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}")

    existing_rows = read_existing(pd, args.output, args.resume)
    done = {int(r["prefix_id"]) for r in existing_rows}
    todo = prefixes[~prefixes["prefix_id"].isin(done)].copy()
    todo = todo.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"])
    if args.max_decisions is not None:
        keys = todo[["trajectory_id", "step_id"]].drop_duplicates().head(args.max_decisions)
        todo = todo.merge(keys, on=["trajectory_id", "step_id"], how="inner")

    print(f"Input rows: {len(prefixes)}")
    print(f"Already scored: {len(existing_rows)}")
    print(f"Rows to score: {len(todo)}")

    model, tokenizer = load(args.model_name_or_path)
    encoded_candidates = {
        action: encode_without_special(tokenizer, candidate_completion(action, args.candidate_mode))
        for action in ACTIONS
    }
    if any(not ids for ids in encoded_candidates.values()):
        raise SystemExit("A candidate encoded to zero tokens")
    single_token_candidates = all(len(ids) == 1 for ids in encoded_candidates.values())

    scored_rows = list(existing_rows)
    pending = 0
    completed = 0
    for (trajectory_id, step_id), group in todo.groupby(["trajectory_id", "step_id"], sort=False):
        group_records = group.to_dict("records")
        if not group_records:
            continue
        base_text = str(group_records[0]["prompt_text"]).rstrip() + ANALYSIS_HEADER
        base_ids = encode_without_special(tokenizer, base_text)
        prefix_cache = make_prompt_cache(model)
        cached_context_ids: list[int] = []
        context_logits = None
        for record in group_records:
            if args.prefix_tokenization == "analysis_token_ids":
                target_context_ids = base_ids + analysis_ids_from_row(record) + encode_without_special(tokenizer, FINAL_JSON_PREFIX)
            else:
                analysis_text = "" if record.get("analysis_prefix_text") is None else str(record["analysis_prefix_text"])
                target_context_ids = encode_without_special(tokenizer, base_text + analysis_text + FINAL_JSON_PREFIX)

            lcp = longest_common_prefix_len(cached_context_ids, target_context_ids)
            if lcp < len(cached_context_ids):
                trim_cache(prefix_cache, len(cached_context_ids) - lcp)
                cached_context_ids = cached_context_ids[:lcp]
                context_logits = None
            suffix = target_context_ids[len(cached_context_ids) :]
            new_logits = feed_ids(model, mx, prefix_cache, suffix)
            if new_logits is not None:
                context_logits = new_logits
            if context_logits is None:
                raise RuntimeError("No logits available for context; duplicate or empty context encountered before initialization")
            cached_context_ids = target_context_ids

            if single_token_candidates:
                logscores = score_single_token_candidates_from_logits(
                    mx=mx,
                    nn=nn,
                    logits=context_logits,
                    encoded_candidates=encoded_candidates,
                )
            else:
                logscores = {
                    action: score_sequence_candidate(
                        model=model,
                        mx=mx,
                        nn=nn,
                        cache=prefix_cache,
                        context_logits=context_logits,
                        candidate_ids=encoded_candidates[action],
                    )
                    for action in ACTIONS
                }
            scored_rows.append(output_row(record, logscores, args.candidate_mode, args.temperature))
            pending += 1
            completed += 1
            if pending >= args.save_every:
                scored_rows.sort(key=lambda r: int(r["prefix_id"]))
                atomic_write_parquet(pd, scored_rows, args.output)
                print(f"Saved {len(scored_rows)} rows")
                pending = 0

    scored_rows.sort(key=lambda r: int(r["prefix_id"]))
    atomic_write_parquet(pd, scored_rows, args.output)
    print(f"Wrote {args.output} with {len(scored_rows)} rows; newly scored {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
