#!/usr/bin/env python3
"""Free-running validation of the forced-answer belief (script 27).

At a subset of the script-20 cuts (10 per commitment bin = 60) we let the model finish its own
reasoning by sampling (temperature --temp) continuations from the cut, --samples per cut, up to
--max_new_tokens tokens, and read the action it then gives (forced answer after its own
<|end|>, which is the model's natural final-channel answer). We compare the empirical
distribution of eventual actions with the forced belief pi_{t,k} at the cut.

Usage: interp/.venv/bin/python scripts/27_free_running_validation.py [--resume]
"""

from __future__ import annotations

import argparse
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
    encode_without_special,
    forced_suffix_token_ids,
    stable_action_probs,
)

Q_COLS = [f"q_{a}" for a in ACTIONS]
BIN_LABELS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95-1"]
STOP_IDS = {200007, 200002, 200012}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injection", type=Path, default=Path("outputs/followups/commitment_gated_injection.parquet"))
    p.add_argument("--metrics", type=Path, default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    p.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    p.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    p.add_argument("--output", type=Path, default=Path("outputs/followups/free_running_validation.parquet"))
    p.add_argument("--report", type=Path, default=Path("outputs/followups/free_running_validation_report.md"))
    p.add_argument("--per_bin", type=int, default=10)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_every", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_evals", type=int, default=None)
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        analyze(pd.read_parquet(args.output), args)
        return 0
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    inj = pd.read_parquet(args.injection)
    inj["prefix_id"] = inj["prefix_id"].astype(str)
    cuts = inj[inj.condition == "counter"].groupby("bin", observed=True).head(args.per_bin).copy()
    metrics = pd.read_parquet(args.metrics, columns=["prefix_id", *Q_COLS])
    metrics["prefix_id"] = metrics["prefix_id"].astype(str)
    cuts = cuts.merge(metrics, on="prefix_id", how="left", validate="one_to_one")
    pref = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    pref["prefix_id"] = pref["prefix_id"].astype(str)
    cuts = cuts.merge(pref, on="prefix_id", how="left", validate="one_to_one")
    assert cuts["q_UP"].notna().all()

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_ids = forced_suffix_token_ids(tokenizer)
    candidate_ids = [encode_without_special(tokenizer, a)[0] for a in ACTIONS]

    def force(ids: list[int]) -> np.ndarray:
        logits = model(mx.array([ids + suffix_ids], dtype=mx.int32))
        sel = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(sel)
        return stable_action_probs(np.asarray(sel, dtype=np.float32))

    jobs = [(row, k) for _, row in cuts.iterrows() for k in range(args.samples)]
    if args.max_evals:
        jobs = jobs[: args.max_evals]
    checkpoint = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        done, done_keys = prev.to_dict("records"), set(prev["row_key"])
        print(f"resume: {len(done_keys)}", flush=True)
    for i, (row, k) in enumerate(jobs, 1):
        key = f"{row['prefix_id']}:{k}"
        if key in done_keys:
            continue
        t0 = time.time()
        prefix = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
        ids = encode_without_special(tokenizer, str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + prefix)
        mx.random.seed(args.seed * 1000 + i)
        sampler = make_sampler(temp=args.temp)
        cont: list[int] = []
        ended = False
        for resp in stream_generate(model, tokenizer, prompt=ids, max_tokens=args.max_new_tokens, sampler=sampler):
            if resp.token in STOP_IDS:
                ended = True
                break
            cont.append(resp.token)
        q_end = force(ids + cont)
        done.append({"row_key": key, "prefix_id": row["prefix_id"], "sample": k, "source_file": row["source_file"],
                     "step_id": int(row["step_id"]), "bin": row["bin"], "progress": float(row["progress"]),
                     "agent_action": row["agent_action"], "top_at_cut": row["top_before"], "commitment_at_cut": float(row["commitment_before"]),
                     **{f"cut_{c}": float(row[c]) for c in Q_COLS}, **{f"end_{c}": float(q_end[j]) for j, c in enumerate(Q_COLS)},
                     "eventual_action": ACTIONS[int(np.argmax(q_end))], "ended_naturally": ended, "n_tokens": len(cont),
                     "continuation_tail": tokenizer.decode(cont[-60:]), "seconds": time.time() - t0})
        done_keys.add(key)
        print(f"{i}/{len(jobs)} bin={row['bin']} top@cut={row['top_before']} eventual={done[-1]['eventual_action']} "
              f"agent={row['agent_action']} tokens={len(cont)} ended={ended} ({done[-1]['seconds']:.1f}s)", flush=True)
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
    s = scores.copy()
    cut = s[[f"cut_{c}" for c in Q_COLS]].to_numpy()
    s["q_cut_of_eventual"] = [cut[i, idx[a]] for i, a in enumerate(s["eventual_action"])]
    s["max_q_cut"] = cut.max(axis=1)
    s["eventual_is_top"] = s["eventual_action"] == s["top_at_cut"]
    s["eventual_is_agent"] = s["eventual_action"] == s["agent_action"]
    by_bin = s.groupby("bin", observed=True).agg(n=("row_key", "size"), ended=("ended_naturally", "mean"),
                                                 mean_max_q_cut=("max_q_cut", "mean"), P_eventual_is_top=("eventual_is_top", "mean"),
                                                 P_eventual_is_agent=("eventual_is_agent", "mean"),
                                                 mean_q_cut_of_eventual=("q_cut_of_eventual", "mean")).round(3)
    ended = s[s.ended_naturally]
    by_bin_ended = ended.groupby("bin", observed=True).agg(n=("row_key", "size"), mean_max_q_cut=("max_q_cut", "mean"),
                                                           P_eventual_is_top=("eventual_is_top", "mean")).round(3)
    # per-cut calibration: fraction of samples choosing the top action vs q_cut(top)
    per_cut = s.groupby("prefix_id").agg(frac_top=("eventual_is_top", "mean"), q_top=("max_q_cut", "first"), n=("row_key", "size"))
    corr = per_cut[["frac_top", "q_top"]].corr().iloc[0, 1]
    lines = ["# Free-running validation (script 27)", "",
             f"- {len(s)} sampled continuations ({s['prefix_id'].nunique()} cuts x {args.samples} samples, temp {args.temp}, max {args.max_new_tokens} new tokens).",
             "- Eventual action taken from the model's own answer after its analysis ends; unfinished samples are forced at the token limit.", "",
             "## By commitment bin (all samples)", "", by_bin.to_markdown(), "",
             "## By commitment bin (naturally ended samples only)", "", by_bin_ended.to_markdown(), "",
             f"## Per-cut calibration: corr(fraction of samples choosing top-at-cut, q_cut(top)) = {corr:.3f}", "",
             f"- Overall mean q_cut(top) = {s['max_q_cut'].mean():.3f}; overall P(eventual == top at cut) = {s['eventual_is_top'].mean():.3f}; "
             f"P(eventual == agent's actual action) = {s['eventual_is_agent'].mean():.3f}", ""]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
