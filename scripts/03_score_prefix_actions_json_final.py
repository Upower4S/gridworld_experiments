#!/usr/bin/env python3
"""Teacher-force score final-JSON action candidates for each analysis prefix."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Any


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


def require_table_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas and pyarrow before running this script."
        ) from exc
    return pd, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prefix_action_scores.parquet"))
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
    parser.add_argument("--candidate_mode", choices=["bare_action", "full_json"], default="bare_action")
    parser.add_argument(
        "--experts_implementation",
        choices=["auto", "eager", "grouped_mm", "batched_mm"],
        default="eager",
        help=(
            "MoE experts implementation passed to Transformers. GPT-OSS needs eager on this "
            "machine because grouped_mm is unavailable on CPU/MPS."
        ),
    )
    return parser.parse_args()


def candidate_completion(action: str, candidate_mode: str) -> str:
    if candidate_mode == "bare_action":
        return action
    if candidate_mode == "full_json":
        return f'{action}"\n}}'
    raise ValueError(f"Unsupported candidate mode: {candidate_mode}")


def scoring_context(prompt_text: str, analysis_prefix_text: str) -> str:
    return prompt_text + ANALYSIS_HEADER + analysis_prefix_text + FINAL_JSON_PREFIX


def stable_softmax(logscores: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    scaled = {action: score / temperature for action, score in logscores.items()}
    max_score = max(scaled.values())
    exp_values = {action: math.exp(score - max_score) for action, score in scaled.items()}
    denom = sum(exp_values.values())
    return {action: value / denom for action, value in exp_values.items()}


def validate_probabilities(probabilities: dict[str, float], tolerance: float = 1e-5) -> None:
    total = sum(probabilities.values())
    if not math.isfinite(total) or abs(total - 1.0) > tolerance:
        raise ValueError(f"Action probabilities do not sum to 1: {total}")


def choose_device(device_arg: str, torch: Any) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(dtype_arg: str, device: str, torch: Any) -> Any | None:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return None


def load_hf_backend(
    model_name_or_path: str,
    device_arg: str,
    dtype_arg: str,
    experts_implementation: str,
) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install transformers and torch to use --backend hf."
        ) from exc

    device = choose_device(device_arg, torch)
    torch_dtype = choose_dtype(dtype_arg, device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if experts_implementation != "auto":
        model_kwargs["experts_implementation"] = experts_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def score_contexts_hf(
    *,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    contexts: list[str],
    candidate_mode: str,
) -> list[dict[str, float]]:
    if not contexts:
        return []

    encoded_contexts = [encode_without_special_tokens(tokenizer, context) for context in contexts]
    encoded_candidates = {
        action: encode_without_special_tokens(tokenizer, candidate_completion(action, candidate_mode))
        for action in ACTIONS
    }
    for action, ids in encoded_candidates.items():
        if not ids:
            raise ValueError(f"Candidate {action} encoded to zero tokens")

    expanded_ids: list[list[int]] = []
    candidate_lengths: list[int] = []
    for context_ids in encoded_contexts:
        for action in ACTIONS:
            candidate_ids = encoded_candidates[action]
            expanded_ids.append(context_ids + candidate_ids)
            candidate_lengths.append(len(candidate_ids))

    max_len = max(len(ids) for ids in expanded_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    for ids in expanded_ids:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    row_scores: list[dict[str, float]] = []
    expanded_index = 0
    for _context_idx in range(len(contexts)):
        scores: dict[str, float] = {}
        for action in ACTIONS:
            ids = expanded_ids[expanded_index]
            candidate_len = candidate_lengths[expanded_index]
            start = len(ids) - candidate_len
            score = 0.0
            for position in range(start, len(ids)):
                token = ids[position]
                score += float(log_probs[expanded_index, position - 1, token].item())
            scores[action] = score
            expanded_index += 1
        row_scores.append(scores)

    return row_scores


def score_contexts_litellm(*_: Any, **__: Any) -> list[dict[str, float]]:
    raise SystemExit(
        "LiteLLM backend is not implemented for teacher-forced logprob scoring here. "
        "Use --backend hf or --backend mlx."
    )


def load_mlx_backend(model_name_or_path: str) -> tuple[Any, Any, Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install mlx-lm to use --backend mlx."
        ) from exc

    model, tokenizer = load(model_name_or_path)
    model.eval()
    return tokenizer, model, mx, nn


def _to_token_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def encode_mlx_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return _to_token_id_list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return _to_token_id_list(tokenizer.encode(text))


def score_one_sequence_mlx(
    *,
    model: Any,
    mx: Any,
    nn: Any,
    token_ids: list[int],
    candidate_len: int,
) -> float:
    if candidate_len < 1:
        raise ValueError("Candidate completion encoded to zero tokens")
    if len(token_ids) <= candidate_len:
        raise ValueError("Scoring sequence must include context tokens before candidate tokens")

    input_tensor = mx.array([token_ids], dtype=mx.int32)
    start = len(token_ids) - candidate_len
    logits = model(input_tensor)
    candidate_logits = logits[0, start - 1 : len(token_ids) - 1, :].astype(mx.float32)
    log_probs = nn.log_softmax(candidate_logits, axis=-1)
    score = 0.0
    for offset, position in enumerate(range(start, len(token_ids))):
        score += float(log_probs[offset, token_ids[position]].item())
    mx.eval(log_probs)
    return score


def score_single_token_candidates_mlx(
    *,
    model: Any,
    mx: Any,
    nn: Any,
    context_ids: list[int],
    encoded_candidates: dict[str, list[int]],
) -> dict[str, float]:
    if not context_ids:
        raise ValueError("Scoring context encoded to zero tokens")

    input_tensor = mx.array([context_ids], dtype=mx.int32)
    logits = model(input_tensor)
    next_token_log_probs = nn.log_softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
    scores = {
        action: float(next_token_log_probs[candidate_ids[0]].item())
        for action, candidate_ids in encoded_candidates.items()
    }
    mx.eval(next_token_log_probs)
    return scores


def score_contexts_mlx(
    *,
    tokenizer: Any,
    model: Any,
    mx: Any,
    nn: Any,
    contexts: list[str],
    candidate_mode: str,
) -> list[dict[str, float]]:
    if not contexts:
        return []

    encoded_candidates = {
        action: encode_mlx_without_special_tokens(
            tokenizer,
            candidate_completion(action, candidate_mode),
        )
        for action in ACTIONS
    }
    for action, ids in encoded_candidates.items():
        if not ids:
            raise ValueError(f"Candidate {action} encoded to zero tokens")

    candidates_are_single_token = all(len(ids) == 1 for ids in encoded_candidates.values())
    row_scores: list[dict[str, float]] = []
    for context in contexts:
        context_ids = encode_mlx_without_special_tokens(tokenizer, context)
        if candidates_are_single_token:
            row_scores.append(
                score_single_token_candidates_mlx(
                    model=model,
                    mx=mx,
                    nn=nn,
                    context_ids=context_ids,
                    encoded_candidates=encoded_candidates,
                )
            )
            continue
        scores: dict[str, float] = {}
        for action in ACTIONS:
            candidate_ids = encoded_candidates[action]
            scores[action] = score_one_sequence_mlx(
                model=model,
                mx=mx,
                nn=nn,
                token_ids=context_ids + candidate_ids,
                candidate_len=len(candidate_ids),
            )
        row_scores.append(scores)
    return row_scores


def output_row_from_scores(
    *,
    prefix_row: dict[str, Any],
    logscores: dict[str, float],
    candidate_mode: str,
    scoring_backend: str,
    temperature: float,
) -> dict[str, Any]:
    probabilities = stable_softmax(logscores, temperature)
    validate_probabilities(probabilities)
    top_action = max(ACTIONS, key=lambda action: probabilities[action])

    return {
        "prefix_id": int(prefix_row["prefix_id"]),
        "trajectory_id": prefix_row["trajectory_id"],
        "source_file": prefix_row["source_file"],
        "step_id": int(prefix_row["step_id"]),
        "prefix_token_index": int(prefix_row["prefix_token_index"]),
        "num_analysis_tokens": int(prefix_row["num_analysis_tokens"]),
        "progress": float(prefix_row["progress"]),
        "agent_action": prefix_row["agent_action"],
        "q_UP": probabilities["UP"],
        "q_DOWN": probabilities["DOWN"],
        "q_LEFT": probabilities["LEFT"],
        "q_RIGHT": probabilities["RIGHT"],
        "logscore_UP": logscores["UP"],
        "logscore_DOWN": logscores["DOWN"],
        "logscore_LEFT": logscores["LEFT"],
        "logscore_RIGHT": logscores["RIGHT"],
        "top_action": top_action,
        "candidate_mode": candidate_mode,
        "scoring_backend": scoring_backend,
        "is_before_first_action_mention": bool(prefix_row["is_before_first_action_mention"]),
        "is_before_first_final_action_mention": bool(prefix_row["is_before_first_final_action_mention"]),
    }


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


def main() -> int:
    args = parse_args()
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided")
    if args.save_every < 1:
        raise SystemExit("--save_every must be >= 1")

    if args.backend == "litellm":
        score_contexts_litellm()

    pd, _ = require_table_dependencies()
    prefixes = pd.read_parquet(args.input)
    if args.limit is not None:
        prefixes = prefixes.head(args.limit)

    missing_columns = [
        column
        for column in [
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
        if column not in prefixes.columns
    ]
    if missing_columns:
        raise SystemExit(f"Input is missing expected columns: {missing_columns}")

    existing_rows = read_existing_scores(pd, args.output, args.resume)
    done_prefix_ids = {int(row["prefix_id"]) for row in existing_rows}
    rows_to_score = prefixes[~prefixes["prefix_id"].isin(done_prefix_ids)]
    rows_to_score = rows_to_score.sort_values("prefix_id")

    if args.backend == "hf":
        tokenizer, model, torch, device = load_hf_backend(
            args.model_name_or_path,
            args.device,
            args.dtype,
            args.experts_implementation,
        )
        mx = None
        nn = None
    elif args.backend == "mlx":
        tokenizer, model, mx, nn = load_mlx_backend(args.model_name_or_path)
        torch = None
        device = "mlx"
    else:
        raise SystemExit(f"Unsupported backend: {args.backend}")

    scored_rows = list(existing_rows)
    pending_since_save = 0

    prefix_records = rows_to_score.to_dict("records")
    total = len(prefix_records)
    print(f"Input rows: {len(prefixes)}")
    print(f"Already scored: {len(existing_rows)}")
    print(f"Rows to score: {total}")

    for start in range(0, total, args.batch_size):
        batch_records = prefix_records[start : start + args.batch_size]
        contexts = [
            scoring_context(
                str(record["prompt_text"]),
                "" if record["analysis_prefix_text"] is None else str(record["analysis_prefix_text"]),
            )
            for record in batch_records
        ]
        if args.backend == "hf":
            logscore_rows = score_contexts_hf(
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                device=device,
                contexts=contexts,
                candidate_mode=args.candidate_mode,
            )
        else:
            logscore_rows = score_contexts_mlx(
                tokenizer=tokenizer,
                model=model,
                mx=mx,
                nn=nn,
                contexts=contexts,
                candidate_mode=args.candidate_mode,
            )

        for record, logscores in zip(batch_records, logscore_rows, strict=True):
            scored_rows.append(
                output_row_from_scores(
                    prefix_row=record,
                    logscores=logscores,
                    candidate_mode=args.candidate_mode,
                    scoring_backend=args.backend,
                    temperature=args.temperature,
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
