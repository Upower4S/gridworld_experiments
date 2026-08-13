"""Utilities for teacher-forced action probing."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
OPTION_LETTERS = ("A", "B", "C", "D")
MC_SCORING_MODE = "multiple_choice_interruption"


def stable_softmax(logscores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = {key: value / temperature for key, value in logscores.items()}
    max_score = max(scaled.values())
    exp_values = {key: math.exp(value - max_score) for key, value in scaled.items()}
    denom = sum(exp_values.values())
    return {key: value / denom for key, value in exp_values.items()}


def validate_probabilities(probabilities: dict[str, float], tolerance: float = 1e-5) -> None:
    total = sum(probabilities.values())
    if not math.isfinite(total) or abs(total - 1.0) > tolerance:
        raise ValueError(f"Action probabilities do not sum to 1: {total}")


def multiple_choice_interruption_context(
    *,
    prompt_text: str,
    analysis_prefix_text: str,
    option_actions: tuple[str, str, str, str],
) -> str:
    if len(option_actions) != len(OPTION_LETTERS):
        raise ValueError("option_actions must contain exactly four actions")
    invalid = [action for action in option_actions if action not in ACTIONS]
    if invalid:
        raise ValueError(f"Invalid option action(s): {invalid}")

    options_text = "\n".join(
        f"{letter}. {action}"
        for letter, action in zip(OPTION_LETTERS, option_actions, strict=True)
    )
    return (
        f"{prompt_text}\n\n"
        f"Reasoning so far:\n"
        f"{analysis_prefix_text}\n\n"
        f"[INTERRUPTION]\n"
        f"Based on the reasoning so far, if you had to choose the final move now, "
        f"which action would you choose?\n\n"
        f"Options:\n"
        f"{options_text}\n\n"
        f"Answer with one letter only.\n"
        f"Answer:\n"
    )


def deterministic_option_permutations(
    *,
    num_permutations: int,
    seed: int = 0,
    include_identity: bool = True,
) -> list[tuple[str, str, str, str]]:
    if num_permutations < 1:
        raise ValueError("num_permutations must be >= 1")

    identity = tuple(ACTIONS)
    all_permutations = [tuple(p) for p in itertools.permutations(ACTIONS)]
    rng = random.Random(seed)
    if include_identity:
        balanced_rotations = [
            tuple(ACTIONS[offset:] + ACTIONS[:offset])
            for offset in range(len(ACTIONS))
        ]
        rest = [p for p in all_permutations if p not in balanced_rotations]
        rng.shuffle(rest)
        ordered = [*balanced_rotations, *rest]
    else:
        ordered = all_permutations[:]
        rng.shuffle(ordered)

    if num_permutations > len(ordered):
        raise ValueError(f"num_permutations cannot exceed {len(ordered)}")
    return ordered[:num_permutations]


def average_letter_scores_to_action_space(
    *,
    letter_logscore_rows: list[dict[str, float]],
    option_permutations: list[tuple[str, str, str, str]],
    temperature: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if len(letter_logscore_rows) != len(option_permutations):
        raise ValueError("letter_logscore_rows and option_permutations length mismatch")
    if not letter_logscore_rows:
        raise ValueError("At least one option permutation is required")

    probability_sums = {action: 0.0 for action in ACTIONS}
    logscore_sums = {action: 0.0 for action in ACTIONS}

    for letter_logscores, option_actions in zip(
        letter_logscore_rows,
        option_permutations,
        strict=True,
    ):
        missing = [letter for letter in OPTION_LETTERS if letter not in letter_logscores]
        if missing:
            raise ValueError(f"Missing letter logscore(s): {missing}")
        letter_probabilities = stable_softmax(letter_logscores, temperature)
        for letter, action in zip(OPTION_LETTERS, option_actions, strict=True):
            probability_sums[action] += letter_probabilities[letter]
            logscore_sums[action] += letter_logscores[letter]

    count = float(len(letter_logscore_rows))
    action_probabilities = {
        action: probability_sums[action] / count
        for action in ACTIONS
    }
    action_logscores = {
        action: logscore_sums[action] / count
        for action in ACTIONS
    }
    validate_probabilities(action_probabilities)
    return action_probabilities, action_logscores


def _to_token_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def encode_mlx_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return _to_token_id_list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return _to_token_id_list(tokenizer.encode(text))


def encode_hf_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


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
    *,
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


def score_contexts_hf_candidates(
    *,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    contexts: list[str],
    candidates: dict[str, str],
) -> list[dict[str, float]]:
    if not contexts:
        return []

    encoded_contexts = [encode_hf_without_special_tokens(tokenizer, context) for context in contexts]
    encoded_candidates = {
        label: encode_hf_without_special_tokens(tokenizer, candidate)
        for label, candidate in candidates.items()
    }
    for label, ids in encoded_candidates.items():
        if not ids:
            raise ValueError(f"Candidate {label} encoded to zero tokens")

    expanded_ids: list[list[int]] = []
    candidate_lengths: list[int] = []
    candidate_labels: list[str] = []
    for context_ids in encoded_contexts:
        for label, candidate_ids in encoded_candidates.items():
            expanded_ids.append(context_ids + candidate_ids)
            candidate_lengths.append(len(candidate_ids))
            candidate_labels.append(label)

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
    labels = list(candidates)
    for _context_idx in range(len(contexts)):
        scores: dict[str, float] = {}
        for _label in labels:
            ids = expanded_ids[expanded_index]
            candidate_len = candidate_lengths[expanded_index]
            start = len(ids) - candidate_len
            score = 0.0
            for position in range(start, len(ids)):
                score += float(log_probs[expanded_index, position - 1, ids[position]].item())
            scores[candidate_labels[expanded_index]] = score
            expanded_index += 1
        row_scores.append(scores)

    return row_scores


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
        label: float(next_token_log_probs[candidate_ids[0]].item())
        for label, candidate_ids in encoded_candidates.items()
    }
    mx.eval(next_token_log_probs)
    return scores


def score_contexts_mlx_candidates(
    *,
    tokenizer: Any,
    model: Any,
    mx: Any,
    nn: Any,
    contexts: list[str],
    candidates: dict[str, str],
) -> list[dict[str, float]]:
    if not contexts:
        return []

    encoded_candidates = {
        label: encode_mlx_without_special_tokens(tokenizer, candidate)
        for label, candidate in candidates.items()
    }
    for label, ids in encoded_candidates.items():
        if not ids:
            raise ValueError(f"Candidate {label} encoded to zero tokens")

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
        for label, candidate_ids in encoded_candidates.items():
            scores[label] = score_one_sequence_mlx(
                model=model,
                mx=mx,
                nn=nn,
                token_ids=context_ids + candidate_ids,
                candidate_len=len(candidate_ids),
            )
        row_scores.append(scores)
    return row_scores
