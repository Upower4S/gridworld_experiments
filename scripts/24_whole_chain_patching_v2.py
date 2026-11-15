#!/usr/bin/env python3
"""Whole-chain patching, second pass (script 24): pre-verbal foreign chains + paired controls.

Reuses the 396 (prompt A, chain B) pairs staged for D4 (script 10). New conditions, each an
independent forward pass:

  foreign_preverbal       A's prompt + B's chain cut just BEFORE B's first decisional action
                          mention (script 23), i.e. B's grid parsing with no move named
  foreign_preverbal_half  same, cut at half that length
  own_preverbal           A's prompt + A's own chain cut before A's first decisional mention
  own_preverbal_half      (paired controls for the two above)
  own_same_relative       A's own chain cut at the same relative position as the original
                          D4 foreign cut (paired control for the stored D4 scores)

Question: does the forced belief follow B's action from B's parsing alone (no action words
in the patched text), and how does that compare with A's own pre-verbal chain?
Usage: interp/.venv/bin/python scripts/24_whole_chain_patching_v2.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    ANALYSIS_HEADER,
    FORCED_SUFFIX,
    encode_without_special,
    forced_suffix_token_ids,
    stable_action_probs,
)

Q_COLS = [f"q_{a}" for a in ACTIONS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=Path("outputs/followups/mismatched_prompt_cot_prefixes.parquet"))
    p.add_argument("--d4_scores", type=Path, default=Path("outputs/followups/mismatched_prompt_cot_scores_extracted.parquet"))
    p.add_argument("--steps", type=Path, default=Path("outputs/canonical_steps.parquet"))
    p.add_argument("--mentions", type=Path, default=Path("outputs/action_mentions_context_steps.parquet"))
    p.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    p.add_argument("--output", type=Path, default=Path("outputs/followups/whole_chain_patching_v2.parquet"))
    p.add_argument("--report", type=Path, default=Path("outputs/followups/whole_chain_patching_v2_report.md"))
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_evals", type=int, default=None)
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def key_of(source_file: str, step_id: int) -> str:
    return Path(source_file).name.replace(".json", "") + f"::{step_id}"


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        analyze(pd.read_parquet(args.output), args)
        return 0
    from mlx_lm import load

    pairs = pd.read_parquet(args.pairs)
    steps = pd.read_parquet(args.steps, columns=["source_file", "step_id", "agent_action", "prompt_text",
                                                 "analysis_text", "analysis_token_ids_json"])
    steps["key"] = [key_of(s, i) for s, i in zip(steps["source_file"], steps["step_id"])]
    steps = steps.set_index("key")
    mentions = pd.read_parquet(args.mentions)
    mentions["key"] = [key_of(s, i) for s, i in zip(mentions["source_file"], mentions["step_id"])]
    first_dec = mentions.set_index("key")["first_decisional_any"]

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    import mlx.core as mx

    forced_suffix_token_ids(tokenizer)
    candidate_ids = [encode_without_special(tokenizer, a)[0] for a in ACTIONS]
    cache: dict[int, str] = {}

    def prefix_text(key: str, n_tokens: int) -> str:
        ids = json.loads(steps.loc[key, "analysis_token_ids_json"])[: max(0, int(n_tokens))]
        return "".join(cache.setdefault(i, tokenizer.decode([i])) for i in ids)

    def score(prompt: str, prefix: str) -> tuple[np.ndarray, int]:
        ids = encode_without_special(tokenizer, prompt.rstrip() + ANALYSIS_HEADER + prefix + FORCED_SUFFIX)
        logits = model(mx.array([ids], dtype=mx.int32))
        sel = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(sel)
        return stable_action_probs(np.asarray(sel, dtype=np.float32)), len(ids)

    jobs: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        a_key, b_key = row["prompt_source_decision_id"], row["cot_source_decision_id"]
        fa, fb = first_dec.get(a_key, np.nan), first_dec.get(b_key, np.nan)
        n_a = len(json.loads(steps.loc[a_key, "analysis_token_ids_json"]))
        rel = float(row["prefix_token_index"]) / float(row["num_analysis_tokens"])
        base = {"pair_row": int(row.name), "a_key": a_key, "b_key": b_key, "a_action": row["prompt_agent_action"],
                "b_action": row["cot_agent_action"], "mismatch_position": row["mismatch_position"],
                "b_first_decisional": fb, "a_first_decisional": fa}
        if not np.isnan(fb):
            jobs.append({**base, "condition": "foreign_preverbal", "source": b_key, "n_tokens_prefix": int(fb)})
            jobs.append({**base, "condition": "foreign_preverbal_half", "source": b_key, "n_tokens_prefix": int(fb) // 2})
        if not np.isnan(fa):
            jobs.append({**base, "condition": "own_preverbal", "source": a_key, "n_tokens_prefix": int(fa)})
            jobs.append({**base, "condition": "own_preverbal_half", "source": a_key, "n_tokens_prefix": int(fa) // 2})
        jobs.append({**base, "condition": "own_same_relative", "source": a_key, "n_tokens_prefix": int(round(rel * n_a))})
    # own_* conditions repeat per A across its 3 rows: dedupe by (condition, source, n_tokens_prefix)
    seen: set[tuple] = set()
    unique_jobs = []
    for j in jobs:
        k = (j["condition"], j["source"], j["n_tokens_prefix"], j["pair_row"] if j["condition"].startswith("foreign") or j["condition"] == "own_same_relative" else -1)
        if k in seen:
            continue
        seen.add(k)
        unique_jobs.append(j)
    if args.max_evals:
        unique_jobs = unique_jobs[: args.max_evals]

    checkpoint = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        done, done_keys = prev.to_dict("records"), set(prev["row_key"])
        print(f"resume: {len(done_keys)} scored", flush=True)
    for i, j in enumerate(unique_jobs, 1):
        row_key = f"{j['condition']}:{j['pair_row']}:{j['source']}:{j['n_tokens_prefix']}"
        if row_key in done_keys:
            continue
        t0 = time.time()
        q, n = score(str(steps.loc[j["a_key"], "prompt_text"]), prefix_text(j["source"], j["n_tokens_prefix"]))
        done.append({"row_key": row_key, **j, "n_tokens": n, **{f"q_{a}": float(q[k]) for k, a in enumerate(ACTIONS)},
                     "top_action": ACTIONS[int(np.argmax(q))], "seconds": time.time() - t0})
        done_keys.add(row_key)
        if i % 20 == 0 or i == len(unique_jobs):
            print(f"{i}/{len(unique_jobs)} ({done[-1]['seconds']:.1f}s)", flush=True)
        if len(done) % args.checkpoint_every == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(done).to_parquet(checkpoint, index=False)
    scores = pd.DataFrame(done)
    scores.to_parquet(args.output, index=False)
    if checkpoint.exists():
        checkpoint.unlink()
    analyze(scores, args)
    return 0


def analyze(scores: pd.DataFrame, args: argparse.Namespace) -> None:
    idx = {a: i for i, a in enumerate(ACTIONS)}
    qm = scores[Q_COLS].to_numpy()
    scores = scores.copy()
    scores["p_a"] = [qm[i, idx[a]] for i, a in enumerate(scores["a_action"])]
    scores["p_b"] = [qm[i, idx[a]] for i, a in enumerate(scores["b_action"])]
    scores["top_is_a"] = scores["top_action"] == scores["a_action"]
    scores["top_is_b"] = scores["top_action"] == scores["b_action"]
    summ = scores.groupby("condition").agg(n=("row_key", "size"), p_a=("p_a", "mean"), p_b=("p_b", "mean"),
                                           top_is_a=("top_is_a", "mean"), top_is_b=("top_is_b", "mean"),
                                           median_prefix_tokens=("n_tokens_prefix", "median")).round(3)
    d4 = pd.read_parquet(args.d4_scores)
    d4m = d4[d4.condition == "mismatched"].groupby("cut_bucket", observed=True).agg(
        p_a=("p_prompt_action", "mean"), p_b=("p_cot_action", "mean"), top_is_b=("top_is_cot_action", "mean"), n=("row_key", "size")).round(3)
    # pre-verbal length strata for foreign_preverbal
    fp = scores[scores.condition == "foreign_preverbal"].copy()
    fp["len_stratum"] = pd.qcut(fp["n_tokens_prefix"], 3, labels=["short", "mid", "long"])
    strata = fp.groupby("len_stratum", observed=True).agg(n=("row_key", "size"), median_tokens=("n_tokens_prefix", "median"),
                                                          p_a=("p_a", "mean"), p_b=("p_b", "mean"), top_is_b=("top_is_b", "mean")).round(3)
    lines = ["# Whole-chain patching v2: pre-verbal foreign chains and paired controls (script 24)", "",
             f"- {len(scores)} evaluations; independent forward passes; prefixes cut at the first decisional mention (script 23).", "",
             "## New conditions (p_a = prob. of A's action, p_b = prob. of B's action)", "", summ.to_markdown(), "",
             "## Reference: original D4 mismatched cuts (stored scores)", "", d4m.to_markdown(), "",
             "## foreign_preverbal by pre-verbal prefix length (terciles)", "", strata.to_markdown(), ""]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
