#!/usr/bin/env python3
"""Score prefix actions with Together Completions API teacher-forced logprobs."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
import random
import tempfile
import threading
import time
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
    "top_prob",
    "second_prob",
    "scoring_backend",
    "scoring_method",
    "candidate_mode",
    "model_name",
    "logprobs",
    "missing_action_logprob",
    "failed",
    "failure_reason",
    "together_request_id",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "candidate_debug_json",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
]


def require_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas, pyarrow, and tqdm before running."
        ) from exc
    return pd, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prefix_action_scores.parquet"))
    parser.add_argument("--model_name", default="openai/gpt-oss-20b")
    parser.add_argument("--api_key_env", default="TOGETHER_AI_API_KEY")
    parser.add_argument(
        "--env_file",
        type=Path,
        default=Path("interp/reveng/.env"),
        help="Optional .env file to load before reading --api_key_env.",
    )
    parser.add_argument(
        "--scoring_method",
        choices=["echo_teacher_forced", "next_token_toplogprobs"],
        default="echo_teacher_forced",
    )
    parser.add_argument("--candidate_mode", choices=["bare_action", "full_json"], default="bare_action")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output parquet by skipping already scored prefix_id rows.",
    )
    parser.add_argument(
        "--retry_failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, drop failed existing rows so their prefix_id values are scored again.",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of prefixes to score concurrently. Each prefix still scores actions sequentially.",
    )
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff_seconds", type=float, default=2.0)
    parser.add_argument("--fallback_to_echo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail_fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--failures_output",
        type=Path,
        default=Path("outputs/scoring_failures_together.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/together_scoring_report.md"),
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def require_api_key(api_key_env: str, env_file: Path) -> str:
    load_env_file(env_file)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {api_key_env} or put it in {env_file}. "
            "This script never hard-codes API keys."
        )
    return api_key


def load_together_client(api_key: str) -> Any:
    try:
        from together import Together
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: together. Install it with `pip install together`.") from exc
    return Together(api_key=api_key)


_THREAD_LOCAL = threading.local()


def thread_together_client(api_key: str) -> Any:
    """Create one Together client per worker thread."""
    client = getattr(_THREAD_LOCAL, "together_client", None)
    if client is None:
        client = load_together_client(api_key)
        _THREAD_LOCAL.together_client = client
    return client


def candidate_completion(action: str, candidate_mode: str) -> str:
    if candidate_mode == "bare_action":
        return action
    if candidate_mode == "full_json":
        return f'{action}"\n}}'
    raise ValueError(f"Unsupported candidate mode: {candidate_mode}")


def scoring_context(prompt_text: Any, analysis_prefix_text: Any) -> str:
    analysis = "" if analysis_prefix_text is None else str(analysis_prefix_text)
    return str(prompt_text).rstrip() + ANALYSIS_HEADER + analysis + FINAL_JSON_PREFIX


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_plain_data(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_plain_data(item) for item in obj]
    if isinstance(obj, tuple):
        return [to_plain_data(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_plain_data(value) for key, value in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return to_plain_data(obj.model_dump())
        except Exception:
            pass
    return str(obj)


def first_choice(response: Any) -> Any | None:
    choices = get_value(response, "choices", []) or []
    return choices[0] if choices else None


def response_id(response: Any) -> str | None:
    value = get_value(response, "id")
    return None if value is None else str(value)


def response_usage(response: Any) -> tuple[int | None, int | None]:
    usage = get_value(response, "usage")
    prompt_tokens = get_value(usage, "prompt_tokens")
    completion_tokens = get_value(usage, "completion_tokens")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


def prompt_logprobs_from_response(response: Any) -> dict[str, list[Any]]:
    prompt_items = get_value(response, "prompt", []) or []
    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    token_ids: list[Any] = []

    for item in prompt_items:
        logprobs = get_value(item, "logprobs")
        if logprobs is None:
            continue
        item_tokens = get_value(logprobs, "tokens", []) or []
        item_logprobs = get_value(logprobs, "token_logprobs", []) or []
        item_token_ids = get_value(logprobs, "token_ids", []) or []
        tokens.extend(str(token) for token in item_tokens)
        token_logprobs.extend(item_logprobs)
        token_ids.extend(item_token_ids)

    if not tokens:
        choice = first_choice(response)
        logprobs = get_value(choice, "logprobs")
        item_tokens = get_value(logprobs, "tokens", []) or []
        item_logprobs = get_value(logprobs, "token_logprobs", []) or []
        item_token_ids = get_value(logprobs, "token_ids", []) or []
        tokens.extend(str(token) for token in item_tokens)
        token_logprobs.extend(item_logprobs)
        token_ids.extend(item_token_ids)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "token_ids": token_ids,
    }


def normalize_token_text(text: str) -> str:
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("<0x0A>", "\n")
    )


def candidate_suffix_start(tokens: list[str], candidate: str) -> int | None:
    max_tokens_to_check = min(len(tokens), max(16, len(candidate) + 8))
    for start in range(len(tokens) - 1, len(tokens) - max_tokens_to_check - 1, -1):
        suffix_tokens = tokens[start:]
        raw_joined = "".join(suffix_tokens)
        normalized_joined = "".join(normalize_token_text(token) for token in suffix_tokens)
        if raw_joined == candidate or normalized_joined == candidate:
            return start
        if raw_joined.endswith(candidate) or normalized_joined.endswith(candidate):
            return start
    return None


def stable_softmax(logscores: dict[str, float]) -> dict[str, float]:
    max_score = max(logscores.values())
    exp_values = {action: math.exp(score - max_score) for action, score in logscores.items()}
    denom = sum(exp_values.values())
    return {action: value / denom for action, value in exp_values.items()}


def probabilities_from_logscores(logscores: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(score) for score in logscores.values()):
        return {action: float("nan") for action in ACTIONS}
    probabilities = stable_softmax(logscores)
    total = sum(probabilities.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-5:
        raise ValueError(f"Action probabilities do not sum to 1: {total}")
    return probabilities


def is_transient_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in ["timeout", "temporar", "rate limit", "429", "503"])


def is_credit_limit_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 402 or "credit limit" in text or "add credits" in text


def together_completion_with_retry(
    *,
    client: Any,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
    **kwargs: Any,
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            return client.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries or not is_transient_error(exc):
                raise
            delay = retry_backoff_seconds * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    raise RuntimeError("Together request failed") from last_exc


def score_candidate_echo(
    *,
    client: Any,
    model_name: str,
    context: str,
    action: str,
    candidate_mode: str,
    temperature: float,
    top_p: float,
    logprobs: int,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
) -> tuple[float, dict[str, Any]]:
    completion = candidate_completion(action, candidate_mode)
    response = together_completion_with_retry(
        client=client,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        sleep_seconds=sleep_seconds,
        model=model_name,
        prompt=context + completion,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        echo=True,
    )

    prompt_logprobs = prompt_logprobs_from_response(response)
    tokens = prompt_logprobs["tokens"]
    token_logprobs = prompt_logprobs["token_logprobs"]
    prompt_tokens, completion_tokens = response_usage(response)
    score_tokens = tokens
    score_token_logprobs = token_logprobs
    if prompt_tokens is not None and 0 < prompt_tokens <= len(tokens):
        score_tokens = tokens[:prompt_tokens]
        score_token_logprobs = token_logprobs[:prompt_tokens]

    start = candidate_suffix_start(score_tokens, completion)
    if start is None:
        raise ValueError(
            f"Could not identify candidate token boundary for {action}. "
            f"Last tokens: {tokens[-12:]}"
        )

    candidate_tokens = score_tokens[start:]
    candidate_logprobs = score_token_logprobs[start:]
    if len(candidate_tokens) != len(candidate_logprobs):
        raise ValueError("Candidate token/logprob length mismatch")
    if any(value is None for value in candidate_logprobs):
        raise ValueError(f"Candidate logprob includes null value for {action}: {candidate_logprobs}")

    return (
        float(sum(float(value) for value in candidate_logprobs)),
        {
            "candidate_completion": completion,
            "candidate_token_texts": candidate_tokens,
            "candidate_token_logprobs": [float(value) for value in candidate_logprobs],
            "together_request_id": response_id(response),
            "usage_prompt_tokens": prompt_tokens,
            "usage_completion_tokens": completion_tokens,
        },
    )


def score_prefix_echo(
    *,
    client: Any,
    model_name: str,
    context: str,
    candidate_mode: str,
    temperature: float,
    top_p: float,
    logprobs: int,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
) -> dict[str, Any]:
    logscores: dict[str, float] = {}
    debug: dict[str, Any] = {}
    request_ids: list[str] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for action in ACTIONS:
        logscore, action_debug = score_candidate_echo(
            client=client,
            model_name=model_name,
            context=context,
            action=action,
            candidate_mode=candidate_mode,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep_seconds=sleep_seconds,
        )
        logscores[action] = logscore
        debug[action] = {
            "candidate_completion": action_debug["candidate_completion"],
            "candidate_token_texts": action_debug["candidate_token_texts"],
            "candidate_token_logprobs": action_debug["candidate_token_logprobs"],
        }
        if action_debug.get("together_request_id"):
            request_ids.append(str(action_debug["together_request_id"]))
        if action_debug.get("usage_prompt_tokens") is not None:
            prompt_tokens_total += int(action_debug["usage_prompt_tokens"])
        if action_debug.get("usage_completion_tokens") is not None:
            completion_tokens_total += int(action_debug["usage_completion_tokens"])

    return {
        "logscores": logscores,
        "missing_action_logprob": False,
        "candidate_debug": debug,
        "together_request_id": ",".join(request_ids) if request_ids else None,
        "usage_prompt_tokens": prompt_tokens_total if prompt_tokens_total else None,
        "usage_completion_tokens": completion_tokens_total if completion_tokens_total else None,
    }


def top_logprobs_from_response(response: Any) -> dict[str, float]:
    choice = first_choice(response)
    logprobs_obj = get_value(choice, "logprobs")
    top = get_value(logprobs_obj, "top_logprobs")
    if top is None:
        return {}
    if isinstance(top, dict):
        return {str(key): float(value) for key, value in top.items()}
    if isinstance(top, list) and top:
        first = top[0]
        if isinstance(first, dict):
            return {str(key): float(value) for key, value in first.items()}
    return {}


def normalized_action_token(token: str) -> str:
    normalized = normalize_token_text(str(token)).strip()
    return normalized.upper()


def score_prefix_next_token(
    *,
    client: Any,
    model_name: str,
    context: str,
    temperature: float,
    top_p: float,
    logprobs: int,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
) -> dict[str, Any]:
    response = together_completion_with_retry(
        client=client,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        sleep_seconds=sleep_seconds,
        model=model_name,
        prompt=context,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
    )
    top_logprobs = top_logprobs_from_response(response)
    by_action: dict[str, float] = {}
    for token, logprob in top_logprobs.items():
        normalized = normalized_action_token(token)
        if normalized in ACTIONS and normalized not in by_action:
            by_action[normalized] = float(logprob)

    missing = [action for action in ACTIONS if action not in by_action]
    logscores = {
        action: by_action.get(action, float("nan"))
        for action in ACTIONS
    }
    prompt_tokens, completion_tokens = response_usage(response)
    return {
        "logscores": logscores,
        "missing_action_logprob": bool(missing),
        "candidate_debug": {
            "top_logprobs": top_logprobs,
            "missing_actions": missing,
        },
        "together_request_id": response_id(response),
        "usage_prompt_tokens": prompt_tokens,
        "usage_completion_tokens": completion_tokens,
    }


def q_for_agent_action(row: dict[str, Any]) -> float | None:
    action = str(row.get("agent_action"))
    value = row.get(f"q_{action}")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def empty_failed_score_result(reason: str) -> dict[str, Any]:
    return {
        "logscores": {action: float("nan") for action in ACTIONS},
        "probabilities": {action: float("nan") for action in ACTIONS},
        "top_action": None,
        "top_prob": float("nan"),
        "second_prob": float("nan"),
        "missing_action_logprob": False,
        "failed": True,
        "failure_reason": reason,
        "candidate_debug": {},
        "together_request_id": None,
        "usage_prompt_tokens": None,
        "usage_completion_tokens": None,
    }


def result_from_logscores(score_payload: dict[str, Any]) -> dict[str, Any]:
    logscores = score_payload["logscores"]
    probabilities = probabilities_from_logscores(logscores)
    sorted_probs = sorted(probabilities.values(), reverse=True)
    top_action = max(ACTIONS, key=lambda action: probabilities[action])
    return {
        **score_payload,
        "probabilities": probabilities,
        "top_action": top_action,
        "top_prob": sorted_probs[0],
        "second_prob": sorted_probs[1],
        "failed": False,
        "failure_reason": None,
    }


def output_row(prefix_row: dict[str, Any], score_result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    logscores = score_result["logscores"]
    probabilities = score_result["probabilities"]
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
        "top_action": score_result["top_action"],
        "top_prob": score_result["top_prob"],
        "second_prob": score_result["second_prob"],
        "scoring_backend": "together",
        "scoring_method": args.scoring_method,
        "candidate_mode": args.candidate_mode,
        "model_name": args.model_name,
        "logprobs": int(args.logprobs),
        "missing_action_logprob": bool(score_result["missing_action_logprob"]),
        "failed": bool(score_result["failed"]),
        "failure_reason": score_result["failure_reason"],
        "together_request_id": score_result["together_request_id"],
        "usage_prompt_tokens": score_result["usage_prompt_tokens"],
        "usage_completion_tokens": score_result["usage_completion_tokens"],
        "candidate_debug_json": json.dumps(to_plain_data(score_result["candidate_debug"]), ensure_ascii=False),
        "is_before_first_action_mention": bool(prefix_row["is_before_first_action_mention"]),
        "is_before_first_final_action_mention": bool(prefix_row["is_before_first_final_action_mention"]),
    }


def append_failure(path: Path, row: dict[str, Any], reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prefix_id": int(row["prefix_id"]),
        "trajectory_id": row.get("trajectory_id"),
        "source_file": row.get("source_file"),
        "step_id": int(row["step_id"]),
        "failure_reason": reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def read_existing_scores(pd: Any, output_path: Path, resume: bool, retry_failed: bool) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    existing = pd.read_parquet(output_path)
    missing = [column for column in OUTPUT_COLUMNS if column not in existing.columns]
    if missing:
        raise SystemExit(
            f"Cannot resume from {output_path}: missing expected Together columns {missing}. "
            "Use --no-resume to overwrite, or choose a different --output."
        )
    if retry_failed and "failed" in existing.columns:
        existing = existing[~existing["failed"].fillna(False).astype(bool)].copy()
    return existing[OUTPUT_COLUMNS].to_dict("records")


def validate_input_columns(prefixes: Any) -> None:
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
        "agent_action",
        "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    missing = [column for column in required if column not in prefixes.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}")


def write_report(pd: Any, rows: list[dict[str, Any]], report_path: Path, requested: int) -> None:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    successful = df[~df["failed"].fillna(False).astype(bool)]
    failed = df[df["failed"].fillna(False).astype(bool)]
    missing_rate = (
        float(df["missing_action_logprob"].fillna(False).mean())
        if len(df)
        else float("nan")
    )
    mean_final_q: float | None = None
    if not successful.empty:
        final_rows = (
            successful.sort_values(["trajectory_id", "source_file", "step_id", "prefix_token_index", "prefix_id"])
            .groupby(["trajectory_id", "source_file", "step_id"], sort=False, dropna=False)
            .tail(1)
        )
        q_values = [q_for_agent_action(row) for row in final_rows.to_dict("records")]
        q_values = [value for value in q_values if value is not None]
        if q_values:
            mean_final_q = sum(q_values) / len(q_values)

    lines = [
        "# Together Scoring Report",
        "",
        f"- Number of prefixes requested: {requested}",
        f"- Number successfully scored: {len(successful)}",
        f"- Number failed: {len(failed)}",
        f"- Missing action logprob rate: {missing_rate:.4f}" if math.isfinite(missing_rate) else "- Missing action logprob rate: NA",
        (
            f"- Mean q assigned to agent_action at final prefix: {mean_final_q:.4f}"
            if mean_final_q is not None
            else "- Mean q assigned to agent_action at final prefix: NA"
        ),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_one_prefix(
    *,
    client: Any,
    prefix_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = scoring_context(prefix_row["prompt_text"], prefix_row["analysis_prefix_text"])
    try:
        if args.scoring_method == "echo_teacher_forced":
            score_payload = score_prefix_echo(
                client=client,
                model_name=args.model_name,
                context=context,
                candidate_mode=args.candidate_mode,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=args.logprobs,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                sleep_seconds=args.sleep_seconds,
            )
        else:
            score_payload = score_prefix_next_token(
                client=client,
                model_name=args.model_name,
                context=context,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=args.logprobs,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                sleep_seconds=args.sleep_seconds,
            )
            if score_payload["missing_action_logprob"] and args.fallback_to_echo:
                score_payload = score_prefix_echo(
                    client=client,
                    model_name=args.model_name,
                    context=context,
                    candidate_mode=args.candidate_mode,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs=args.logprobs,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    sleep_seconds=args.sleep_seconds,
                )
        if score_payload["missing_action_logprob"] and args.scoring_method == "next_token_toplogprobs":
            return {
                **empty_failed_score_result("Missing one or more action tokens in top_logprobs."),
                **score_payload,
                "failed": True,
                "failure_reason": "Missing one or more action tokens in top_logprobs.",
            }
        return result_from_logscores(score_payload)
    except Exception as exc:  # noqa: BLE001
        if args.fail_fast or is_credit_limit_error(exc):
            raise
        return empty_failed_score_result(str(exc))


def score_record_with_thread_client(
    *,
    api_key: str,
    prefix_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    client = thread_together_client(api_key)
    score_result = score_one_prefix(client=client, prefix_row=prefix_row, args=args)
    return output_row(prefix_row, score_result, args)


def iter_parallel_scored_rows(
    *,
    api_key: str,
    prefix_records: list[dict[str, Any]],
    args: argparse.Namespace,
    tqdm: Any,
) -> Any:
    """Yield scored rows using a bounded pool of prefix-level worker tasks."""
    max_workers = int(args.num_workers)
    max_pending = max(max_workers * 4, max_workers)
    record_iter = iter(prefix_records)
    pending: dict[Any, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_pending, len(prefix_records))):
            record = next(record_iter)
            future = executor.submit(
                score_record_with_thread_client,
                api_key=api_key,
                prefix_row=record,
                args=args,
            )
            pending[future] = record

        with tqdm(total=len(prefix_records), desc="Together scoring", unit="prefix") as progress:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001
                        if args.fail_fast or is_credit_limit_error(exc):
                            raise
                        row = output_row(record, empty_failed_score_result(str(exc)), args)
                    progress.update(1)
                    yield row, record

                    try:
                        next_record = next(record_iter)
                    except StopIteration:
                        continue
                    next_future = executor.submit(
                        score_record_with_thread_client,
                        api_key=api_key,
                        prefix_row=next_record,
                        args=args,
                    )
                    pending[next_future] = next_record


def main() -> int:
    args = parse_args()
    if args.logprobs < 1 or args.logprobs > 20:
        raise SystemExit("--logprobs must be between 1 and 20")
    if args.max_tokens < 1:
        raise SystemExit("--max_tokens must be >= 1")
    if args.save_every < 1:
        raise SystemExit("--save_every must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided")
    if args.max_retries < 0:
        raise SystemExit("--max_retries must be >= 0")
    if args.num_workers < 1:
        raise SystemExit("--num_workers must be >= 1")

    pd, tqdm = require_dependencies()
    api_key = require_api_key(args.api_key_env, args.env_file)
    client = load_together_client(api_key) if args.num_workers == 1 else None

    prefixes = pd.read_parquet(args.input)
    validate_input_columns(prefixes)
    if args.limit is not None:
        prefixes = prefixes.head(args.limit)

    existing_rows = read_existing_scores(pd, args.output, args.resume, args.retry_failed)
    if args.resume and args.retry_failed and args.output.exists():
        existing_rows.sort(key=lambda saved_row: int(saved_row["prefix_id"]))
        atomic_write_parquet(pd, existing_rows, args.output)
        write_report(pd, existing_rows, args.report, requested=len(prefixes))
    if not args.resume and args.failures_output.exists():
        args.failures_output.write_text("", encoding="utf-8")
    done_prefix_ids = {int(row["prefix_id"]) for row in existing_rows}
    rows_to_score = prefixes[~prefixes["prefix_id"].isin(done_prefix_ids)]
    rows_to_score = rows_to_score.sort_values("prefix_id")
    prefix_records = rows_to_score.to_dict("records")
    total = len(prefix_records)
    scoring_calls = total * 4 if args.scoring_method == "echo_teacher_forced" else total

    context_lengths = [
        len(scoring_context(record["prompt_text"], record["analysis_prefix_text"]))
        for record in prefix_records
    ]
    print(f"Prefixes requested: {len(prefixes)}")
    print(f"Already scored: {len(existing_rows)}")
    print(f"Prefixes to score: {total}")
    print(f"Estimated Together scoring calls: {scoring_calls}")
    print(f"Mean prompt length in chars: {sum(context_lengths) / len(context_lengths):.1f}" if context_lengths else "Mean prompt length in chars: NA")
    print(f"Max prompt length in chars: {max(context_lengths)}" if context_lengths else "Max prompt length in chars: NA")

    scored_rows = list(existing_rows)
    pending_since_save = 0
    if args.num_workers == 1:
        row_iter = (
            (output_row(record, score_one_prefix(client=client, prefix_row=record, args=args), args), record)
            for record in tqdm(prefix_records, desc="Together scoring", unit="prefix")
        )
    else:
        row_iter = iter_parallel_scored_rows(
            api_key=api_key,
            prefix_records=prefix_records,
            args=args,
            tqdm=tqdm,
        )

    try:
        for row, record in row_iter:
            if row["failed"]:
                append_failure(args.failures_output, record, str(row["failure_reason"]))
            scored_rows.append(row)
            pending_since_save += 1

            if pending_since_save >= args.save_every:
                scored_rows.sort(key=lambda saved_row: int(saved_row["prefix_id"]))
                atomic_write_parquet(pd, scored_rows, args.output)
                write_report(pd, scored_rows, args.report, requested=len(prefixes))
                print(f"Saved {len(scored_rows)} rows")
                pending_since_save = 0
    except BaseException:
        if pending_since_save:
            scored_rows.sort(key=lambda saved_row: int(saved_row["prefix_id"]))
            atomic_write_parquet(pd, scored_rows, args.output)
            write_report(pd, scored_rows, args.report, requested=len(prefixes))
            print(f"Stopped early; saved {len(scored_rows)} rows")
        raise

    scored_rows.sort(key=lambda row: int(row["prefix_id"]))
    atomic_write_parquet(pd, scored_rows, args.output)
    write_report(pd, scored_rows, args.report, requested=len(prefixes))
    print(f"Wrote {args.output} with {len(scored_rows)} rows")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
