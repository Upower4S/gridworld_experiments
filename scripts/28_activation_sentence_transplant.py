#!/usr/bin/env python3
"""Bandwidth-matched sentence-activation transplant (script 28).

Same 240 cuts as script 20. Donor run  = prefix + COUNTER sentence (names the runner-up) +
forced suffix; recipient run = prefix + NEUTRAL sentence of the same token length + forced
suffix. Both sequences are identical except the 21 sentence tokens, so positions align. We
capture the donor's residual states at the sentence span after every layer and write them
into the recipient at chosen (layers x positions), then read the forced answer.

Conditions (positions within the 21-token sentence span; layers of the 24-layer model):
  all_layers_full      all 24 layers, whole span  -> positive control: must reproduce the
                       donor's (counter-text) answer if the hooking is correct
  all_layers_last12/6/1  all layers, last k sentence tokens only
  early_0-7 / mid_8-15 / late_16-23  layer thirds, whole span
  layer_10 / layer_15 / layer_18     single layers, whole span (comparable to earlier work)

Usage: interp/.venv/bin/python scripts/28_activation_sentence_transplant.py [--resume]
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
COUNTER = " Wait, I should double-check this. Looking at the grid again, the correct move here is {action}."
NEUTRAL_CANDIDATES = [
    " Wait, I should double-check this. Looking at the grid again carefully before settling on a move here.",
    " Wait, I should double-check this. Looking at the grid again more carefully before settling on a move.",
    " Wait, I should double-check this. Looking at the grid once more before settling on any move here.",
    " Wait, I should double-check this. Looking at the grid again before settling on a move here now.",
    " Wait, I should double-check this. Looking at the grid again, thinking it through before settling on a move.",
    " Wait, I should double-check this. Looking at the grid again before I settle on a move here.",
]
CONDITIONS: dict[str, tuple[list[int], int | None]] = {  # name -> (layers, last_k or None=whole span)
    "all_layers_full": (list(range(24)), None),
    "all_layers_last12": (list(range(24)), 12),
    "all_layers_last6": (list(range(24)), 6),
    "all_layers_last1": (list(range(24)), 1),
    "early_0-7": (list(range(0, 8)), None),
    "mid_8-15": (list(range(8, 16)), None),
    "late_16-23": (list(range(16, 24)), None),
    "layer_10": ([10], None),
    "layer_15": ([15], None),
    "layer_18": ([18], None),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injection", type=Path, default=Path("outputs/followups/commitment_gated_injection.parquet"))
    p.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    p.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    p.add_argument("--output", type=Path, default=Path("outputs/followups/activation_sentence_transplant.parquet"))
    p.add_argument("--report", type=Path, default=Path("outputs/followups/activation_sentence_transplant_report.md"))
    p.add_argument("--checkpoint_every", type=int, default=5, help="in cuts")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_cuts", type=int, default=None)
    p.add_argument("--analyze_only", action="store_true")
    return p.parse_args()


def run_layers(model: Any, mx: Any, create_attention_mask: Any, token_ids: list[int], *,
               capture: dict[int, slice] | None = None, patch: dict[int, tuple[slice, Any]] | None = None) -> tuple[Any, dict[int, Any]]:
    transformer = model.model
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    captured: dict[int, Any] = {}
    for index, (layer, layer_type) in enumerate(zip(transformer.layers, transformer.layer_types, strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, None)
        if patch and index in patch:
            sl, donor = patch[index]
            x[:, sl, :] = donor.astype(x.dtype)
        if capture and index in capture:
            captured[index] = mx.array(x[:, capture[index], :])
    return x, captured


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        analyze(pd.read_parquet(args.output), args)
        return 0
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    import mlx.core as mx

    plan = pd.read_parquet(args.injection)
    plan = plan[plan["condition"] == "counter"].copy()
    pref = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    pref["prefix_id"] = pref["prefix_id"].astype(str)
    plan["prefix_id"] = plan["prefix_id"].astype(str)
    plan = plan.merge(pref, on="prefix_id", how="left", validate="one_to_one")
    if args.max_cuts:
        plan = plan.groupby("bin", observed=True).head(max(1, args.max_cuts // 6))

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_ids = forced_suffix_token_ids(tokenizer)
    candidate_ids = [encode_without_special(tokenizer, a)[0] for a in ACTIONS]
    n_counter = {a: len(encode_without_special(tokenizer, COUNTER.format(action=a))) for a in ACTIONS}
    assert len(set(n_counter.values())) == 1, n_counter
    span_len = next(iter(n_counter.values()))
    neutral = next((s for s in NEUTRAL_CANDIDATES if len(encode_without_special(tokenizer, s)) == span_len), None)
    if neutral is None:
        raise SystemExit(f"no neutral candidate with {span_len} tokens: "
                         f"{[len(encode_without_special(tokenizer, s)) for s in NEUTRAL_CANDIDATES]}")
    print(f"span length {span_len}; neutral = {neutral!r}", flush=True)

    def q_from(x: Any) -> np.ndarray:
        logits = model.lm_head(model.model.norm(x))
        sel = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(sel)
        return stable_action_probs(np.asarray(sel, dtype=np.float32))

    checkpoint = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_cuts: set[str] = set()
    if args.resume and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        done, done_cuts = prev.to_dict("records"), set(prev["prefix_id"])
        print(f"resume: {len(done_cuts)} cuts", flush=True)
    n_cuts_done = 0
    for i, (_, row) in enumerate(plan.iterrows(), 1):
        if row["prefix_id"] in done_cuts:
            continue
        t0 = time.time()
        prefix = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
        base_text = str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + prefix
        donor_ids = encode_without_special(tokenizer, base_text + COUNTER.format(action=row["target_action"])) + suffix_ids
        recip_ids = encode_without_special(tokenizer, base_text + neutral) + suffix_ids
        base_ids = encode_without_special(tokenizer, base_text)
        aligned = (len(donor_ids) == len(recip_ids) == len(base_ids) + span_len + len(suffix_ids)
                   and donor_ids[: len(base_ids)] == base_ids == recip_ids[: len(base_ids)])
        common = {"prefix_id": row["prefix_id"], "source_file": row["source_file"], "step_id": int(row["step_id"]),
                  "bin": row["bin"], "progress": float(row["progress"]), "agent_action": row["agent_action"],
                  "top_before": row["top_before"], "runner_up": row["target_action"],
                  "commitment_before": float(row["commitment_before"]), "p_top_before": float(row["p_top_before"]),
                  "aligned": bool(aligned), "n_tokens": len(recip_ids)}
        if not aligned:
            done.append({**common, "condition": "unaligned", **{f"after_{c}": np.nan for c in Q_COLS}, "top_after": None})
            done_cuts.add(row["prefix_id"])
            continue
        span = slice(len(base_ids), len(base_ids) + span_len)
        x_d, captured = run_layers(model, mx, create_attention_mask, donor_ids, capture={k: span for k in range(24)})
        mx.eval(*captured.values())
        results = {"donor_text": q_from(x_d)}
        x_r, _ = run_layers(model, mx, create_attention_mask, recip_ids)
        results["recipient_text"] = q_from(x_r)
        for name, (layers, last_k) in CONDITIONS.items():
            sl = span if last_k is None else slice(span.stop - last_k, span.stop)
            patch = {k: (sl, captured[k] if last_k is None else captured[k][:, -last_k:, :]) for k in layers}
            x_p, _ = run_layers(model, mx, create_attention_mask, recip_ids, patch=patch)
            results[name] = q_from(x_p)
        for name, q in results.items():
            done.append({**common, "condition": name, **{f"after_{c}": float(q[j]) for j, c in enumerate(Q_COLS)},
                         "top_after": ACTIONS[int(np.argmax(q))]})
        done_cuts.add(row["prefix_id"])
        n_cuts_done += 1
        print(f"cut {i}/{len(plan)} ({time.time() - t0:.1f}s) donor_top={ACTIONS[int(np.argmax(results['donor_text']))]} "
              f"ctrl_top={ACTIONS[int(np.argmax(results['all_layers_full']))]}", flush=True)
        if n_cuts_done % args.checkpoint_every == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(done).to_parquet(checkpoint, index=False)
    scores = pd.DataFrame(done)
    scores.attrs["neutral"] = neutral
    scores.to_parquet(args.output, index=False)
    if checkpoint.exists():
        checkpoint.unlink()
    analyze(scores, args)
    return 0


def analyze(scores: pd.DataFrame, args: argparse.Namespace) -> None:
    idx = {a: i for i, a in enumerate(ACTIONS)}
    s = scores[scores.condition != "unaligned"].copy()
    aft = s[[f"after_{c}" for c in Q_COLS]].to_numpy()
    s["p_top_after"] = [aft[i, idx[a]] for i, a in enumerate(s["top_before"])]
    s["p_runner_after"] = [aft[i, idx[a]] for i, a in enumerate(s["runner_up"])]
    s["top_changed"] = s["top_after"] != s["top_before"]
    s["flip_to_runner"] = s["top_after"] == s["runner_up"]
    s["d_top"] = s["p_top_after"] - s["p_top_before"]
    donor = s[s.condition == "donor_text"].set_index("prefix_id")
    s["matches_donor_top"] = [donor.loc[p, "top_after"] == t for p, t in zip(s["prefix_id"], s["top_after"])]
    order = ["donor_text", "recipient_text"] + list(CONDITIONS)
    rows = []
    for cond in order:
        f = s[s.condition == cond]
        r = {"condition": cond, "n": len(f), "top changed": f["top_changed"].mean(), "flip to runner-up": f["flip_to_runner"].mean(),
             "mean dq(top)": f["d_top"].mean(), "top == donor top": f["matches_donor_top"].mean()}
        for b in BIN_LABELS:
            sub = f[f["bin"] == b]
            r[f"flip {b}"] = sub["flip_to_runner"].mean() if len(sub) else np.nan
        rows.append(r)
    summary = pd.DataFrame(rows)
    n_unaligned = int((scores.condition == "unaligned").sum())
    lines = ["# Sentence-activation transplant (script 28)", "",
             f"- {s['prefix_id'].nunique()} aligned cuts ({n_unaligned} skipped as unaligned); neutral sentence: {scores.attrs.get('neutral', '?')!r}",
             "- donor_text = counter sentence as text; recipient_text = neutral sentence as text; other rows = neutral text with donor states written in.", "",
             summary.round(3).to_markdown(index=False), ""]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
