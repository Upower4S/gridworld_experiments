#!/usr/bin/env python3
"""Belief-level vs conclusion-level injections, and template robustness (script 25).

Same 240 cuts as script 20 (six bins of c_{t,k}, 40 each). At each cut we append ONE
sentence and force the answer with an independent forward pass (no KV-cache reuse):

  belief_block_top     "the cell immediately <dir of top action> of the agent at (r,c) is a
                       wall (#)."  -> a STATE fact whose implication is 'do not take top';
                       false at ~94% of cuts (truth recorded), names no action as a choice
  belief_open_runnerup "the cell immediately <dir of runner-up> ... is open (_)." -> mostly true
  belief_true_top      the TRUE fact about the top action's neighbour cell (control: a fact
                       sentence per se)
  counter_alt1         conclusion-level, alternative wording ("on reflection X looks like the
                       better move")
  counter_alt2         conclusion-level, observation-like ("the shortest path starts by going x")

Compared against script 20's conclusion-level `counter` and `neutral` rows on the same cuts.
Usage: interp/.venv/bin/python scripts/25_belief_level_injection.py [--resume]
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
    FORCED_SUFFIX,
    encode_without_special,
    forced_suffix_token_ids,
    stable_action_probs,
)

Q_COLS = [f"q_{a}" for a in ACTIONS]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
DIRWORD = {"UP": "above", "DOWN": "below", "LEFT": "to the left of", "RIGHT": "to the right of"}
BIN_LABELS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95-1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injection", type=Path, default=Path("outputs/followups/commitment_gated_injection.parquet"))
    p.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    p.add_argument("--steps", type=Path, default=Path("outputs/canonical_steps.parquet"))
    p.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    p.add_argument("--output", type=Path, default=Path("outputs/followups/belief_level_injection.parquet"))
    p.add_argument("--report", type=Path, default=Path("outputs/followups/belief_level_injection_report.md"))
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_evals", type=int, default=None)
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def parse_grid(grid: str) -> tuple[list[list[str]], tuple[int, int]]:
    rows = [line for line in grid.split("\n") if line.strip()][1:]
    cells = [r.split()[1:] for r in rows]
    pos = next((ri, ci) for ri, row in enumerate(cells) for ci, ch in enumerate(row) if ch == "A")
    return cells, pos


def neighbour(grid: str, action: str) -> tuple[str, tuple[int, int]]:
    cells, (r, c) = parse_grid(grid)
    dr, dc = DELTA[action]
    return cells[r + dr][c + dc], (r, c)


def sentence(condition: str, top: str, runner: str, grid: str) -> tuple[str, bool | None]:
    """Return (injected sentence, is_fact_true or None)."""
    top_cell, (r, c) = neighbour(grid, top)
    run_cell, _ = neighbour(grid, runner)
    if condition == "belief_block_top":
        return (f" Wait, looking at the grid again: the cell immediately {DIRWORD[top]} the agent at ({r},{c}) is a wall (#).",
                top_cell == "#")
    if condition == "belief_open_runnerup":
        return (f" Wait, looking at the grid again: the cell immediately {DIRWORD[runner]} the agent at ({r},{c}) is open (_).",
                run_cell != "#")
    if condition == "belief_true_top":
        desc = "a wall (#)" if top_cell == "#" else ("open (_)" if top_cell == "_" else f"the {'key' if top_cell == 'K' else 'goal' if top_cell == 'G' else 'door'} ({top_cell})")
        return (f" Wait, looking at the grid again: the cell immediately {DIRWORD[top]} the agent at ({r},{c}) is {desc}.", True)
    if condition == "counter_alt1":
        return f" Hmm, on reflection {runner} looks like the better move here.", None
    if condition == "counter_alt2":
        return f" Actually, the shortest path from here starts by going {runner.lower()}.", None
    raise ValueError(condition)


CONDITIONS = ["belief_block_top", "belief_open_runnerup", "belief_true_top", "counter_alt1", "counter_alt2"]


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        analyze(pd.read_parquet(args.output), args)
        return 0
    from mlx_lm import load

    base = pd.read_parquet(args.injection)
    plan = base[base["condition"] == "counter"].copy()
    pref = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    pref["prefix_id"] = pref["prefix_id"].astype(str)
    plan["prefix_id"] = plan["prefix_id"].astype(str)
    plan = plan.merge(pref, on="prefix_id", how="left", validate="one_to_one")
    grids = pd.read_parquet(args.steps, columns=["source_file", "step_id", "grid_state_text"])
    plan = plan.merge(grids, on=["source_file", "step_id"], how="left")
    assert plan["prompt_text"].notna().all() and plan["grid_state_text"].notna().all()

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    import mlx.core as mx

    forced_suffix_token_ids(tokenizer)
    candidate_ids = [encode_without_special(tokenizer, a)[0] for a in ACTIONS]

    checkpoint = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        done, done_keys = prev.to_dict("records"), set(prev["row_key"])
        print(f"resume: {len(done_keys)} scored", flush=True)

    jobs = [(cond, row) for _, row in plan.iterrows() for cond in CONDITIONS]
    if args.max_evals:
        jobs = jobs[: args.max_evals]
    for i, (cond, row) in enumerate(jobs, 1):
        key = f"{cond}:{row['prefix_id']}"
        if key in done_keys:
            continue
        text_inj, fact_true = sentence(cond, row["top_before"], row["target_action"], row["grid_state_text"])
        prefix = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
        ids = encode_without_special(tokenizer, str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + prefix + text_inj + FORCED_SUFFIX)
        t0 = time.time()
        logits = model(mx.array([ids], dtype=mx.int32))
        sel = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(sel)
        q = stable_action_probs(np.asarray(sel, dtype=np.float32))
        done.append({"row_key": key, "condition": cond, "prefix_id": row["prefix_id"], "source_file": row["source_file"],
                     "step_id": int(row["step_id"]), "bin": row["bin"], "progress": float(row["progress"]),
                     "agent_action": row["agent_action"], "top_before": row["top_before"],
                     "runner_up": row["target_action"], "commitment_before": float(row["commitment_before"]),
                     "p_top_before": float(row["p_top_before"]), "p_runner_before": float(row["p_target_before"]) if row["condition"] == "counter" else np.nan,
                     "fact_true": fact_true, "injection": text_inj, "n_tokens": len(ids),
                     **{f"after_{c}": float(q[j]) for j, c in enumerate(Q_COLS)},
                     "top_after": ACTIONS[int(np.argmax(q))], "seconds": time.time() - t0})
        done_keys.add(key)
        if i % 10 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} ({done[-1]['seconds']:.1f}s)", flush=True)
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
    aft = scores[[f"after_{c}" for c in Q_COLS]].to_numpy()
    scores = scores.copy()
    scores["p_top_after"] = [aft[i, idx[a]] for i, a in enumerate(scores["top_before"])]
    scores["p_runner_after"] = [aft[i, idx[a]] for i, a in enumerate(scores["runner_up"])]
    scores["top_changed"] = scores["top_after"] != scores["top_before"]
    scores["flip_to_runner"] = scores["top_after"] == scores["runner_up"]
    scores["d_top"] = scores["p_top_after"] - scores["p_top_before"]
    # script-20 reference rows (counter / neutral) on the same cuts
    ref = pd.read_parquet(args.injection)
    ref_aft = ref[[f"after_{c}" for c in Q_COLS]].to_numpy()
    ref = ref.assign(p_top_after=[ref_aft[i, idx[a]] for i, a in enumerate(ref["top_before"])],
                     top_changed=ref["top_after"] != ref["top_before"],
                     flip_to_runner=ref["top_after"] == ref["target_action"])
    ref["d_top"] = ref["p_top_after"] - ref["p_top_before"]
    rows = []
    for cond, frame in list(scores.groupby("condition")) + [("counter (script 20)", ref[ref.condition == "counter"]),
                                                            ("neutral (script 20)", ref[ref.condition == "neutral"])]:
        r = {"condition": cond, "n": len(frame), "top changed": frame["top_changed"].mean(),
             "flip to runner-up": frame["flip_to_runner"].mean(), "mean dq(top)": frame["d_top"].mean()}
        for b in BIN_LABELS:
            sub = frame[frame["bin"] == b]
            r[f"changed {b}"] = sub["top_changed"].mean() if len(sub) else np.nan
        rows.append(r)
    summary = pd.DataFrame(rows)
    # false vs true fact for belief_block_top
    bb = scores[scores.condition == "belief_block_top"]
    fact = bb.groupby("fact_true").agg(n=("top_changed", "size"), top_changed=("top_changed", "mean"),
                                       flip_to_runner=("flip_to_runner", "mean"), d_top=("d_top", "mean"))
    lines = ["# Belief-level vs conclusion-level injections (script 25)", "",
             f"- {len(scores)} evaluations on the {scores['prefix_id'].nunique()} cuts of script 20; independent forward passes.", "",
             "## Summary by condition (all cuts; per-bin top-changed rates)", "", summary.round(3).to_markdown(index=False), "",
             "## belief_block_top split by whether the injected wall fact is true", "", fact.round(3).to_markdown(), "",
             "## Example injections", ""]
    for cond in CONDITIONS:
        lines.append(f"- {cond}: {scores[scores.condition == cond]['injection'].iloc[0]}")
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
