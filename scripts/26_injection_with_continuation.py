#!/usr/bin/env python3
"""Injection with continuation (script 26): does an injected sentence survive the model's own
follow-on reasoning?

Subset of the script-20 cuts (20 per commitment bin = 120). For each cut and each injection
(conclusion-level `counter` naming the runner-up; belief-level `belief_block_top` from script
25) we let the model continue greedily for up to --max_new_tokens tokens after the injection,
stopping at <|end|> (analysis finished), and THEN force the answer. Compared with the
immediate forced answer (no continuation) from scripts 20/25.

Usage: interp/.venv/bin/python scripts/26_injection_with_continuation.py [--resume]
"""

from __future__ import annotations

import argparse
import re
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
STOP_IDS = {200007, 200002, 200012}  # <|end|>, <|return|>, <|call|>
COUNTER = " Wait, I should double-check this. Looking at the grid again, the correct move here is {action}."
REJECT_RE = re.compile(r"\b(actually|wait|no,|not a wall|is open|isn't a wall|is not a wall|mistake|wrong|hmm)\b", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injection", type=Path, default=Path("outputs/followups/commitment_gated_injection.parquet"))
    p.add_argument("--belief", type=Path, default=Path("outputs/followups/belief_level_injection.parquet"))
    p.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    p.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    p.add_argument("--output", type=Path, default=Path("outputs/followups/injection_with_continuation.parquet"))
    p.add_argument("--report", type=Path, default=Path("outputs/followups/injection_with_continuation_report.md"))
    p.add_argument("--per_bin", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--checkpoint_every", type=int, default=5)
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
    import mlx.core as mx

    inj = pd.read_parquet(args.injection)
    inj["prefix_id"] = inj["prefix_id"].astype(str)
    counter = inj[inj.condition == "counter"].groupby("bin", observed=True).head(args.per_bin).copy()
    belief = pd.read_parquet(args.belief)
    belief = belief[belief.condition == "belief_block_top"].set_index("prefix_id")
    pref = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    pref["prefix_id"] = pref["prefix_id"].astype(str)
    counter = counter.merge(pref, on="prefix_id", how="left", validate="one_to_one")

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_ids = forced_suffix_token_ids(tokenizer)
    candidate_ids = [encode_without_special(tokenizer, a)[0] for a in ACTIONS]

    def force(ids: list[int]) -> np.ndarray:
        logits = model(mx.array([ids + suffix_ids], dtype=mx.int32))
        sel = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(sel)
        return stable_action_probs(np.asarray(sel, dtype=np.float32))

    jobs = []
    for _, row in counter.iterrows():
        prefix = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
        base = str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + prefix
        jobs.append(("counter", row, base + COUNTER.format(action=row["target_action"]), float(row["p_top_before"])))
        if row["prefix_id"] in belief.index:
            b = belief.loc[row["prefix_id"]]
            jobs.append(("belief_block_top", row, base + str(b["injection"]), float(row["p_top_before"])))
    if args.max_evals:
        jobs = jobs[: args.max_evals]

    checkpoint = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        done, done_keys = prev.to_dict("records"), set(prev["row_key"])
        print(f"resume: {len(done_keys)}", flush=True)
    for i, (cond, row, text, p_top_before) in enumerate(jobs, 1):
        key = f"{cond}:{row['prefix_id']}"
        if key in done_keys:
            continue
        t0 = time.time()
        ids = encode_without_special(tokenizer, text)
        q_immediate = force(ids)
        cont: list[int] = []
        ended = False
        for resp in stream_generate(model, tokenizer, prompt=ids, max_tokens=args.max_new_tokens):
            if resp.token in STOP_IDS:
                ended = True
                break
            cont.append(resp.token)
        cont_text = tokenizer.decode(cont)
        q_after = force(ids + cont)
        done.append({"row_key": key, "condition": cond, "prefix_id": row["prefix_id"], "source_file": row["source_file"],
                     "step_id": int(row["step_id"]), "bin": row["bin"], "progress": float(row["progress"]),
                     "agent_action": row["agent_action"], "top_before": row["top_before"], "runner_up": row["target_action"],
                     "commitment_before": float(row["commitment_before"]), "p_top_before": p_top_before,
                     "n_continuation_tokens": len(cont), "ended_naturally": ended, "continuation": cont_text,
                     "rejection_cue": bool(REJECT_RE.search(cont_text[:200])),
                     **{f"imm_{c}": float(q_immediate[j]) for j, c in enumerate(Q_COLS)},
                     **{f"after_{c}": float(q_after[j]) for j, c in enumerate(Q_COLS)},
                     "top_immediate": ACTIONS[int(np.argmax(q_immediate))], "top_after": ACTIONS[int(np.argmax(q_after))],
                     "seconds": time.time() - t0})
        done_keys.add(key)
        print(f"{i}/{len(jobs)} {cond} imm={done[-1]['top_immediate']} after={done[-1]['top_after']} "
              f"tokens={len(cont)} ended={ended} ({done[-1]['seconds']:.1f}s)", flush=True)
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
    aft = s[[f"after_{c}" for c in Q_COLS]].to_numpy()
    s["p_top_after"] = [aft[i, idx[a]] for i, a in enumerate(s["top_before"])]
    s["changed_immediate"] = s["top_immediate"] != s["top_before"]
    s["changed_after"] = s["top_after"] != s["top_before"]
    s["flip_runner_immediate"] = s["top_immediate"] == s["runner_up"]
    s["flip_runner_after"] = s["top_after"] == s["runner_up"]
    s["reverted"] = s["changed_immediate"] & ~s["changed_after"]
    s["d_top_after"] = s["p_top_after"] - s["p_top_before"]
    rows = []
    for cond, f in s.groupby("condition"):
        r = {"condition": cond, "n": len(f), "changed immediate": f["changed_immediate"].mean(), "changed after continuation": f["changed_after"].mean(),
             "flip->runner immediate": f["flip_runner_immediate"].mean(), "flip->runner after": f["flip_runner_after"].mean(),
             "reverted (changed then back)": f["reverted"].mean(), "mean dq(top) after": f["d_top_after"].mean(),
             "ended naturally": f["ended_naturally"].mean(), "median cont. tokens": f["n_continuation_tokens"].median(),
             "rejection cue": f["rejection_cue"].mean()}
        for b in BIN_LABELS:
            sub = f[f["bin"] == b]
            r[f"changed after {b}"] = sub["changed_after"].mean() if len(sub) else np.nan
        rows.append(r)
    lines = ["# Injection with continuation (script 26)", "", f"- {len(s)} evaluations; greedy continuation up to {args.max_new_tokens} tokens, then forced answer.", "",
             pd.DataFrame(rows).round(3).to_markdown(index=False), "", "## Example continuations (first 300 chars)", ""]
    for cond, f in s.groupby("condition"):
        for _, r in f.head(3).iterrows():
            lines.append(f"- [{cond}, bin {r['bin']}, imm={r['top_immediate']}, after={r['top_after']}] {r['continuation'][:300]!r}")
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
