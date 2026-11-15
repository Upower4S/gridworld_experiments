#!/usr/bin/env python3
"""Commitment-gated counter-evidence injection (text-level causal test, script 20).

Design (fixed before scoring): sample (step, cut) pairs stratified into six bins of
pre-cut commitment_to_final (faithful extracted labels). At each sampled cut, append one
injection sentence to the truncated CoT, force the answer suffix, and read the four
action-token probabilities with a full independent forward pass (no KV-cache reuse
anywhere - the corrupted stage-03 path is never used). Three conditions per cut:

  counter - injection names the runner-up action at that cut (the 2nd-highest q);
  neutral - same sentence shape, names no action (controls for appended text + position);
  support - injection names the current top action (positive control).

Predictions: (1) counter-injection flip-to-target rate declines with pre-injection
commitment; (2) neutral injections rarely change the top action; (3) support injections
increase q(top). The paired quantity for counter-vs-neutral is flip/Delta-q of the SAME
runner-up target at the SAME cut.

Usage (repo root):
    interp/.venv/bin/python scripts/20_commitment_gated_injection.py            # full run
    interp/.venv/bin/python scripts/20_commitment_gated_injection.py --resume   # continue
    interp/.venv/bin/python scripts/20_commitment_gated_injection.py --max_evals 6
"""

from __future__ import annotations

import argparse
import json
import sys
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

Q_COLS = [f"q_{action}" for action in ACTIONS]
BIN_EDGES = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001]
BIN_LABELS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95-1"]

COUNTER_TEMPLATE = (
    " Wait, I should double-check this. Looking at the grid again, "
    "the correct move here is {action}."
)
NEUTRAL_TEXT = (
    " Wait, I should double-check this. Looking at the grid again "
    "before settling on a move."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    parser.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/followups/commitment_gated_injection.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/followups/commitment_gated_injection_report.md"))
    parser.add_argument("--figure", type=Path,
                        default=Path("figures/counterfactual/commitment_gated_injection.png"))
    parser.add_argument("--per_bin", type=int, default=40)
    parser.add_argument("--max_cuts_per_step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="skip row_keys already in the partial checkpoint")
    parser.add_argument("--max_evals", type=int, default=None, help="debug cap")
    parser.add_argument("--analyze_only", action="store_true",
                        help="skip scoring; rebuild report/figure from --output")
    return parser.parse_args()


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def build_plan(args: argparse.Namespace) -> pd.DataFrame:
    metrics = pd.read_parquet(args.metrics)
    prefixes = pd.read_parquet(
        args.prefixes,
        columns=["prefix_id", "prompt_text", "analysis_prefix_text"],
    )
    table = metrics.merge(prefixes, on="prefix_id", how="inner", validate="one_to_one")
    q_matrix = table[Q_COLS].to_numpy(dtype=float)
    order = np.argsort(-q_matrix, axis=1)
    table["runner_up_action"] = [ACTIONS[order[i, 1]] for i in range(len(table))]
    table["p_runner_up_before"] = q_matrix[np.arange(len(table)), order[:, 1]]
    table["bin"] = pd.cut(table["commitment_to_final"], bins=BIN_EDGES,
                          labels=BIN_LABELS, include_lowest=True, right=False)

    rng = np.random.default_rng(args.seed)
    table = table.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
    per_step: dict[tuple[str, int], int] = {}
    chosen: list[pd.Series] = []
    for label in BIN_LABELS:
        taken = 0
        for _, row in table[table["bin"] == label].iterrows():
            key = (row["source_file"], int(row["step_id"]))
            if per_step.get(key, 0) >= args.max_cuts_per_step:
                continue
            per_step[key] = per_step.get(key, 0) + 1
            chosen.append(row)
            taken += 1
            if taken >= args.per_bin:
                break
    plan = pd.DataFrame(chosen).reset_index(drop=True)
    return plan


def eval_rows(plan: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    from mlx_lm import load

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    import mlx.core as mx

    forced_suffix_token_ids(tokenizer)  # validates the 12-token contract on load
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        if len(encoded) != 1:
            raise ValueError(f"action {action} not a single token")
        candidate_ids.append(encoded[0])

    checkpoint_path = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume and checkpoint_path.exists():
        previous = pd.read_parquet(checkpoint_path)
        done = previous.to_dict("records")
        done_keys = set(previous["row_key"])
        print(f"resume: {len(done_keys)} evals already scored", flush=True)

    jobs: list[tuple[str, str, pd.Series]] = []
    for _, row in plan.iterrows():
        for condition in ("counter", "neutral", "support"):
            jobs.append((condition, f"{condition}:{row['prefix_id']}", row))
    if args.max_evals is not None:
        jobs = jobs[: args.max_evals]

    import time

    for index, (condition, row_key, row) in enumerate(jobs, start=1):
        if row_key in done_keys:
            continue
        if condition == "counter":
            target = str(row["runner_up_action"])
            injection = COUNTER_TEMPLATE.format(action=target)
        elif condition == "support":
            target = str(row["top_action"])
            injection = COUNTER_TEMPLATE.format(action=target)
        else:
            target = str(row["runner_up_action"])  # paired drift control
            injection = NEUTRAL_TEXT
        prefix_text = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
        text = (str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + prefix_text
                + injection + FORCED_SUFFIX)
        token_ids = encode_without_special(tokenizer, text)
        started = time.time()
        logits = model(mx.array([token_ids], dtype=mx.int32))
        selected = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(selected)
        q_after = stable_action_probs(np.asarray(selected, dtype=np.float32))
        record = {
            "row_key": row_key,
            "condition": condition,
            "prefix_id": str(row["prefix_id"]),
            "source_file": row["source_file"],
            "step_id": int(row["step_id"]),
            "prefix_token_index": int(row["prefix_token_index"]),
            "progress": float(row["progress"]),
            "bin": str(row["bin"]),
            "agent_action": str(row["agent_action"]),
            "top_before": str(row["top_action"]),
            "commitment_before": float(row["commitment_to_final"]),
            "margin_before": float(row["margin_commitment"]),
            "target_action": target,
            "p_target_before": float(row[f"q_{target}"]),
            "p_top_before": float(row[f"q_{row['top_action']}"]),
            "n_tokens": len(token_ids),
            **{f"after_{col}": float(q_after[i]) for i, col in enumerate(Q_COLS)},
            "top_after": ACTIONS[int(np.argmax(q_after))],
            "seconds": time.time() - started,
        }
        done.append(record)
        done_keys.add(row_key)
        if index % 10 == 0 or index == len(jobs):
            print(f"{index}/{len(jobs)} scored ({record['seconds']:.1f}s, "
                  f"{record['n_tokens']} tokens)", flush=True)
        if len(done) % args.checkpoint_every == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(done).to_parquet(checkpoint_path, index=False)

    scores = pd.DataFrame(done)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.output, index=False)
    return scores


def analyze(scores: pd.DataFrame, args: argparse.Namespace) -> None:
    scores = scores.copy()
    after_matrix = scores[[f"after_{c}" for c in Q_COLS]].to_numpy(dtype=float)
    idx = {action: position for position, action in enumerate(ACTIONS)}
    scores["p_target_after"] = [after_matrix[i, idx[a]]
                                for i, a in enumerate(scores["target_action"])]
    scores["delta_q_target"] = scores["p_target_after"] - scores["p_target_before"]
    scores["flipped_to_target"] = scores["top_after"] == scores["target_action"]
    scores["top_changed"] = scores["top_after"] != scores["top_before"]

    rows = []
    for label in BIN_LABELS:
        cell: dict[str, Any] = {"bin": label}
        for condition in ("counter", "neutral", "support"):
            sub = scores[(scores["bin"] == label) & (scores["condition"] == condition)]
            if condition == "counter":
                k = int(sub["flipped_to_target"].sum())
                lo, hi = wilson(k, len(sub))
                cell.update(n=len(sub), counter_flip=sub["flipped_to_target"].mean(),
                            counter_lo=lo, counter_hi=hi,
                            counter_dq=sub["delta_q_target"].mean())
            elif condition == "neutral":
                cell.update(neutral_flip=sub["flipped_to_target"].mean(),
                            neutral_change=sub["top_changed"].mean(),
                            neutral_dq=sub["delta_q_target"].mean())
            else:
                cell.update(support_dq_top=(sub["p_target_after"]
                                            - sub["p_top_before"]).mean())
        rows.append(cell)
    summary = pd.DataFrame(rows)

    counter = scores[scores["condition"] == "counter"]
    from scipy.stats import spearmanr

    rho, pval = spearmanr(counter["commitment_before"], counter["flipped_to_target"])
    lowest = counter[counter["bin"] == BIN_LABELS[0]]["flipped_to_target"]
    highest = counter[counter["bin"] == BIN_LABELS[-1]]["flipped_to_target"]

    # Robustness: at low commitment the runner-up target can coincide with the action
    # the agent eventually takes, so a "flip" there pushes toward the eventual action
    # anyway. Split by whether the injected target is strictly counterfactual.
    counter = counter.assign(target_is_eventual=counter["target_action"]
                             == counter["agent_action"])
    strict = counter[~counter["target_is_eventual"]]
    rho_strict, pval_strict = spearmanr(strict["commitment_before"],
                                        strict["flipped_to_target"])
    rho_prog, pval_prog = spearmanr(strict["progress"], strict["flipped_to_target"])
    rho_cp = spearmanr(strict["commitment_before"], strict["progress"]).statistic
    from scipy.stats import rankdata

    def _resid(y: pd.Series, x: pd.Series) -> np.ndarray:
        y_rank, x_rank = rankdata(y), rankdata(x)
        slope, intercept = np.polyfit(x_rank, y_rank, 1)
        return y_rank - (slope * x_rank + intercept)

    rho_partial, pval_partial = spearmanr(
        _resid(strict["commitment_before"], strict["progress"]),
        _resid(strict["flipped_to_target"], strict["progress"]))
    strict_bins = (counter.groupby("bin", observed=True)
                   .agg(share_target_eventual=("target_is_eventual", "mean"))
                   .join(strict.groupby("bin", observed=True)
                         .agg(strict_n=("flipped_to_target", "size"),
                              strict_flip=("flipped_to_target", "mean")))
                   .reindex(BIN_LABELS).reset_index(names="bin"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)
    x = np.arange(len(BIN_LABELS))
    left.errorbar(x, summary["counter_flip"],
                  yerr=[summary["counter_flip"] - summary["counter_lo"],
                        summary["counter_hi"] - summary["counter_flip"]],
                  fmt="-o", color="#b7352d", linewidth=2, capsize=3,
                  label="counter-evidence injection: flips to injected action")
    left.plot(x, summary["neutral_flip"], "-s", color="#777777", linewidth=1.8,
              label="neutral injection: drifts to same action")
    left.set_xticks(x, BIN_LABELS)
    left.set_ylim(-0.02, 1.0)
    left.set_xlabel("commitment to eventual action at injection point")
    left.set_ylabel("P(forced top action = injected action)")
    left.set_title("Counter-evidence efficacy vs commitment")
    left.legend(fontsize=8.5, frameon=False)
    right.plot(x, summary["counter_dq"], "-o", color="#b7352d", linewidth=2,
               label="counter: $\\Delta$q(injected action)")
    right.plot(x, summary["neutral_dq"], "-s", color="#777777", linewidth=1.8,
               label="neutral: $\\Delta$q(same action)")
    right.plot(x, summary["support_dq_top"], "-^", color="#215caf", linewidth=1.8,
               label="support: $\\Delta$q(current top)")
    right.axhline(0, color="#bbbbbb", linewidth=1)
    right.set_xticks(x, BIN_LABELS)
    right.set_xlabel("commitment to eventual action at injection point")
    right.set_ylabel("mean probability change")
    right.set_title("Belief movement by condition")
    right.legend(fontsize=8.5, frameon=False)
    for axis in (left, right):
        axis.grid(True, alpha=0.3, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, bbox_inches="tight")

    lines = [
        "# Commitment-gated counter-evidence injection (script 20)",
        "",
        f"- {len(scores)} scored evaluations over {scores['prefix_id'].nunique()} cuts, "
        f"{scores.groupby(['source_file', 'step_id']).ngroups} decisions, "
        f"{scores['source_file'].nunique()} trajectories; full independent forward passes,"
        " no KV-cache reuse; forced suffix validated on load; seed "
        f"{args.seed}.",
        f"- Injection templates: counter/support `{COUNTER_TEMPLATE.strip()}`; "
        f"neutral `{NEUTRAL_TEXT.strip()}`.",
        "",
        "## Per-bin summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Headline tests",
        "",
        f"- Spearman(commitment at injection, flipped-to-target), counter rows: "
        f"rho={rho:.3f}, p={pval:.2e}, n={len(counter)}.",
        f"- Counter flip rate, lowest bin {BIN_LABELS[0]}: {lowest.mean():.3f} "
        f"(n={len(lowest)}); highest bin {BIN_LABELS[-1]}: {highest.mean():.3f} "
        f"(n={len(highest)}).",
        f"- Neutral any-top-change rate overall: "
        f"{scores[scores.condition == 'neutral']['top_changed'].mean():.3f}.",
        f"- Support mean q(top) after: "
        f"{scores[scores.condition == 'support']['p_target_after'].mean():.3f} "
        f"(before {scores[scores.condition == 'support']['p_top_before'].mean():.3f}).",
        "",
        "## Robustness: strictly counterfactual targets only",
        "",
        "Rows where the injected runner-up equals the eventually taken action are"
        " excluded (at low commitment a flip there pushes toward the eventual action"
        " anyway).",
        "",
        strict_bins.round(4).to_markdown(index=False),
        "",
        f"- Spearman on strictly counterfactual rows: rho={rho_strict:.3f}, "
        f"p={pval_strict:.2e}, n={len(strict)}.",
        f"- Position attribution (strict rows): rho(flip, progress)={rho_prog:.3f} "
        f"(p={pval_prog:.2e}); rho(commitment, progress)={rho_cp:.3f}; partial "
        f"rho(flip, commitment | progress)={rho_partial:.3f} (p={pval_partial:.2e}) — "
        "commitment and position co-vary, and only part of the decline is attributable "
        "to commitment as such.",
        "",
        f"- Figure: `{args.figure}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(scores)} rows), {args.report}, {args.figure}")


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        analyze(pd.read_parquet(args.output), args)
        return 0
    plan = build_plan(args)
    counts = plan["bin"].value_counts().reindex(BIN_LABELS)
    print("plan:", dict(counts), f"-> {3 * len(plan)} evaluations", flush=True)
    scores = eval_rows(plan, args)
    analyze(scores, args)
    checkpoint_path = args.output.with_suffix(".partial.parquet")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
