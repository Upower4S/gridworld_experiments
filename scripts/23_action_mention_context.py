"""Context-aware detection of action mentions in the CoT (script 23).

The prefix pipeline (stage 02) defines the "first action word" as the first reasoning
token that the trajectory-generation pipeline tagged as a direction word, with no regard
to context: "set up", "to the left of the door", "cannot go up" all count. This script
classifies every occurrence of up/down/left/right in every reasoning trace by its
context and recomputes the two results that depend on the first mention:

  1. the pre-verbal control (commitment / inverse-entropy rise restricted to prefixes
     before the first *decisional* mention), and
  2. the first-mention timing measure (median position; share of decisions whose
     commitment prefix precedes it), with the three-clocks CDF redrawn.

Occurrence classes (frozen rules, precedence top to bottom):
  idiom        set up / pick up / end up / look up / ... / right now / that's right /
               ", right?" / what's left / N steps left / left over / write down / ...
  descriptive  preceded by "to|on|at|in the" or "top-/bottom-/upper-/lower-", or followed
               by column|row|side|of|from|wall|cell|edge|half|corner|neighbo(u)r|hand|...
  negated      a negation cue within the 5 words before (not, n't, cannot, no, never, avoid,
               without, instead of, rather than, blocked, impossible, invalid, illegal,
               neither, nor, unable) or "is/are/would be blocked|invalid|impossible|a wall"
               within 4 words after
  answer       preceded (<= 12 chars) by answer|action|final|output      -> decisional
  movement     preceded by move|go|going|gone|went|head|step|walk|turn   -> decisional
  uppercase    written in capitals (UP/DOWN/LEFT/RIGHT)                  -> decisional
  bare         anything else (route enumerations such as "left, up, up") -> decisional

Usage (repo root):
    interp/.venv/bin/python scripts/23_action_mention_context.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402  (repo convention)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
WORD = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)

# --- word-specific idioms (checked first) ---------------------------------------------
UP_IDIOM_BEFORE = re.compile(
    r"\b(?:set|sets|setting|pick|picks|picked|picking|end|ends|ended|ending|look|looks|looked|"
    r"looking|wake|sum|summing|give|gives|gave|giving|make|makes|made|show|shows|showed|come|"
    r"comes|came|line|lines|lined|add|adds|added|clean|speed|catch|open|opens|opened|light|"
    r"lights|lit|bring|brings|brought|free|freed|team|split|build|builds|built|hold|holds|"
    r"held|keep|keeps|kept|mix|mixed|use|used|using|back|backs|backed)\s+$", re.IGNORECASE)
UP_IDIOM_AFTER = re.compile(r"^\s*-?to-date\b|^\s+to\s+the\s+(?:max|limit|total|number|full|point)\b|^\s+until\b|^\s+front\b", re.IGNORECASE)
DOWN_IDIOM_BEFORE = re.compile(
    r"\b(?:write|writes|wrote|written|note|notes|noted|narrow|narrows|narrowed|break|breaks|"
    r"broke|broken|boil|boils|slow|slows|settle|settles|calm|track|cut|cuts|scroll|scrolls|"
    r"scrolled|pin|pinned|nail|nailed|jot|jotted|lock|locked|hunker|double|dumb|dumbed|"
    r"list|listed|lay|laid|lie|lies|sit|sits|sat|shut|shuts|turn|turned|turning)\s+$", re.IGNORECASE)
DOWN_IDIOM_AFTER = re.compile(r"^\s+to\s+the\s+(?:point|question|fact|detail|wire)\b", re.IGNORECASE)
LEFT_REMAINING_BEFORE = re.compile(r"\b(?:what'?s|what is|what are|nothing|none|anything|something|how (?:much|many)|only thing|time|room)\s+(?:is\s+|are\s+)?$", re.IGNORECASE)
LEFT_IDIOM_AFTER = re.compile(r"^\s+(?:over|behind|out)\b|^\s*-?over\b|^overs?\b", re.IGNORECASE)
RIGHT_IDIOM_BEFORE = re.compile(r"\b(?:that'?s|that is|it'?s|it is|you'?re|you are|sounds?|seems?|looks?|am|i'?m|alright|all|quite|exactly|absolutely|totally)\s+$", re.IGNORECASE)
CORRECTNESS_AFTER = re.compile(r"^\s+(?:way|thing|answer|choice|idea|call|decision|approach|track)\b", re.IGNORECASE)
RIGHT_IDIOM_AFTER = re.compile(r"^\s*(?:now\b|away\b|-?hand\b|-?click\b|angles?\b)", re.IGNORECASE)

# --- descriptive uses ---------------------------------------------------------------
DESC_BEFORE = re.compile(
    r"\b(?:to|on|at|in|towards?|from|of|along|near|by|beside)\s+(?:the|its|our|my|his|her|their|"
    r"agent'?s|door'?s|key'?s|goal'?s)\s*$|\b(?:top|bottom|upper|lower|far|extreme|immediate)\s*-?\s*$",
    re.IGNORECASE)
DESC_AFTER = re.compile(r"^\s*-?\s*(?:column|col|columns|row|rows|side|sides|of|wall|walls|cell|cells|edge|edges|half|corner|corners|neighbou?rs?|hand|part|region|area|border|boundary|end|portion|quadrant|tile|tiles|square|squares|position|coordinate)\b", re.IGNORECASE)

# --- negation ------------------------------------------------------------------------
NEG_BEFORE = re.compile(
    r"\b(?:not|cannot|can'?t|never|avoid\w*|without|instead of|rather than|neither|nor|"
    r"unable to|don'?t|won'?t|shouldn'?t|isn'?t|aren'?t|wouldn'?t|couldn'?t|doesn'?t|mustn'?t|"
    r"no longer|no)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)
NEG_AFTER = re.compile(
    r"^\s+(?:is|are|was|were|would be|will be|'s|isn'?t|aren'?t|seems|looks)\s+(?:blocked|invalid|"
    r"impossible|illegal|not\s+(?:possible|allowed|an option|valid|available|feasible)|a wall|"
    r"wall|out of bounds|off the grid|unavailable|no good|bad|useless|pointless|a dead end)",
    re.IGNORECASE)

# --- strong decisional cues -------------------------------------------------------------
ANSWER_BEFORE = re.compile(r"(?:answer|action|final|output|decision|choice|choose|chose|pick)\W{0,12}$",
                           re.IGNORECASE)
MOVE_BEFORE = re.compile(
    r"(?:\bmove\w*|\bgo(?:es|ing)?|\bgone|\bwent|\bhead\w*|\bstep\w*|\bwalk\w*|\bturn\w*|"
    r"\bpush\w*|\btravel\w*|\bproceed\w*|\bcontinue\w*|\bshift\w*)"
    r"(?:\s+(?:to\s+the|towards?\s+the|straight|directly|one|two|three|\d+|up|down|left|right))?\s*$",
    re.IGNORECASE)

NON_DECISIONAL = {"idiom", "descriptive", "negated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="outputs/canonical_steps.parquet")
    parser.add_argument("--metrics", default="outputs/prefix_commitment_metrics_extracted.parquet")
    parser.add_argument("--clocks", default="outputs/commitment_boundary_steps.parquet")
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    parser.add_argument("--occurrences", default="outputs/action_mentions_context.parquet")
    parser.add_argument("--step_table", default="outputs/action_mentions_context_steps.parquet")
    parser.add_argument("--mask", default="outputs/preverbal_decisional_mask.parquet")
    parser.add_argument("--report", default="outputs/action_mention_context_report.md")
    parser.add_argument("--figure", default="figures/commitment_boundary/three_clocks_cdf_v2.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def classify(text: str, start: int, end: int) -> tuple[str, bool, str]:
    """Return (class, decisional, rule) for the direction word at text[start:end]."""
    before, after = text[max(0, start - 60):start], text[end:end + 60]
    word = text[start:end]
    lw = word.lower()
    if word.isupper():
        lw = "__upper__"  # capitalised direction words are answer vocabulary, never idioms
    if lw == "up" and (UP_IDIOM_BEFORE.search(before) or UP_IDIOM_AFTER.search(after)):
        return "idiom", False, "idiom:up"
    if lw == "down" and (DOWN_IDIOM_BEFORE.search(before) or DOWN_IDIOM_AFTER.search(after)):
        return "idiom", False, "idiom:down"
    if lw == "left" and (LEFT_REMAINING_BEFORE.search(before) or LEFT_IDIOM_AFTER.search(after)):
        return "idiom", False, "idiom:left"
    if lw == "right" and (RIGHT_IDIOM_BEFORE.search(before) or RIGHT_IDIOM_AFTER.search(after)
                          or (re.search(r"\bthe\s+$", before) and CORRECTNESS_AFTER.search(after))):
        return "idiom", False, "idiom:right"
    if NEG_BEFORE.search(before):
        return "negated", False, "neg:before"
    if NEG_AFTER.search(after):
        return "negated", False, "neg:after"
    if ANSWER_BEFORE.search(before):
        return "answer", True, "answer"
    if MOVE_BEFORE.search(before):
        return "movement", True, "movement"
    if word.isupper():
        return "uppercase", True, "uppercase"
    if DESC_BEFORE.search(before):
        return "descriptive", False, "desc:before"
    if DESC_AFTER.search(after):
        return "descriptive", False, "desc:after"
    return "bare", True, "bare"


def token_offsets(tokenizer, ids: list[int], cache: dict[int, str]) -> tuple[list[int], str]:
    pieces = []
    for i in ids:
        s = cache.get(i)
        if s is None:
            s = cache[i] = tokenizer.decode([i])
        pieces.append(s)
    offsets, pos = [], 0
    for s in pieces:
        offsets.append(pos)
        pos += len(s)
    return offsets, "".join(pieces)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    steps = pd.read_parquet(REPO_ROOT / args.steps,
                            columns=["trajectory_id", "source_file", "step_id", "agent_action",
                                     "analysis_text", "analysis_token_ids_json"])
    cache: dict[int, str] = {}
    occ_rows, step_rows, mismatches = [], [], 0
    for _, row in steps.iterrows():
        ids = json.loads(row.analysis_token_ids_json)
        offsets, rebuilt = token_offsets(tokenizer, ids, cache)
        text = row.analysis_text
        if rebuilt != text:
            mismatches += 1
        first_any = first_final = None
        counts = {k: 0 for k in ["idiom", "descriptive", "negated", "answer", "movement", "uppercase", "bare"]}
        for m in WORD.finditer(text):
            cls, decisional, rule = classify(text, m.start(), m.end())
            counts[cls] += 1
            tok_index = bisect.bisect_right(offsets, m.start()) - 1
            word = m.group(1).upper()
            occ_rows.append({"trajectory_id": row.trajectory_id, "source_file": row.source_file,
                             "step_id": int(row.step_id), "token_index": tok_index, "char_start": m.start(),
                             "word": word, "cls": cls, "rule": rule, "decisional": decisional,
                             "is_final_action": word == row.agent_action,
                             "snippet": text[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")})
            if decisional:
                if first_any is None:
                    first_any = tok_index
                if first_final is None and word == row.agent_action:
                    first_final = tok_index
        step_rows.append({"trajectory_id": row.trajectory_id, "source_file": row.source_file,
                          "step_id": int(row.step_id), "agent_action": row.agent_action,
                          "num_analysis_tokens": len(ids),
                          "first_decisional_any": first_any, "first_decisional_final": first_final,
                          **{f"n_{k}": v for k, v in counts.items()}})
    occ = pd.DataFrame(occ_rows)
    per_step = pd.DataFrame(step_rows)
    occ.to_parquet(REPO_ROOT / args.occurrences, index=False)

    # ---- join with the prefix table; old definition for comparison
    metrics = pd.read_parquet(REPO_ROOT / args.metrics)
    old_first = (metrics.drop_duplicates(["source_file", "step_id"])
                 [["source_file", "step_id", "first_action_mention_token_index", "num_cot_tokens"]])
    per_step = per_step.merge(old_first, on=["source_file", "step_id"], how="left")
    per_step.to_parquet(REPO_ROOT / args.step_table, index=False)

    m = metrics.merge(per_step[["source_file", "step_id", "first_decisional_any", "first_decisional_final"]],
                      on=["source_file", "step_id"], how="left")
    m["preverbal_old"] = m["strict_before_any_action_mention"].astype(bool)
    m["preverbal_decisional"] = m["prefix_token_index"] < m["first_decisional_any"].fillna(np.inf)
    m["preverbal_decisional_final"] = m["prefix_token_index"] < m["first_decisional_final"].fillna(np.inf)
    # sensitivity: negated mentions count as decisional
    neg_first = (occ[occ["decisional"] | (occ["cls"] == "negated")]
                 .groupby(["source_file", "step_id"])["token_index"].min().rename("first_incl_negated"))
    m = m.merge(neg_first, on=["source_file", "step_id"], how="left")
    m["preverbal_incl_negated"] = m["prefix_token_index"] < m["first_incl_negated"].fillna(np.inf)
    m[["prefix_id", "preverbal_old", "preverbal_decisional", "preverbal_decisional_final",
       "preverbal_incl_negated"]].to_parquet(REPO_ROOT / args.mask, index=False)

    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0001]

    def subset_summary(flag: str) -> dict:
        sub = m[m[flag]]
        b = pd.cut(sub["progress"], edges, include_lowest=True, right=False)
        g = sub.groupby(b, observed=True)
        return {"n": len(sub), "share": len(sub) / len(m),
                "commit_first_tenth": g["commitment_to_final"].mean().iloc[0],
                "commit_final_quarter": g["commitment_to_final"].mean().iloc[-1],
                "entropy_first_tenth": g["entropy_commitment"].mean().iloc[0],
                "entropy_final_quarter": g["entropy_commitment"].mean().iloc[-1],
                "top_is_final": sub["is_top_action_final"].mean(),
                "n_final_quarter": int(g.size().iloc[-1])}

    summaries = {name: subset_summary(flag) for name, flag in [
        ("old: any tagged direction token", "preverbal_old"),
        ("new: first decisional mention (any action)", "preverbal_decisional"),
        ("new: first decisional mention of the eventual action", "preverbal_decisional_final"),
        ("sensitivity: negated mentions also count", "preverbal_incl_negated")]}

    # ---- timing measures
    clocks = pd.read_parquet(REPO_ROOT / args.clocks)
    ps = per_step.merge(clocks[["trajectory_id", "step_id", "stabilization_progress",
                                "permanent_cross_0_8_progress", "first_mention_progress"]],
                        on=["trajectory_id", "step_id"], how="left")
    ps["first_decisional_progress"] = ps["first_decisional_any"] / ps["num_cot_tokens"]
    has = ps["first_decisional_progress"].notna()
    timing = {
        "steps with any decisional mention": int(has.sum()),
        "median first mention progress (old)": float(ps["first_mention_progress"].median()),
        "median first decisional mention progress (new)": float(ps.loc[has, "first_decisional_progress"].median()),
        "fraction committing before first mention (old)": float((ps["stabilization_progress"] < ps["first_mention_progress"]).mean()),
        "fraction committing before first decisional mention (new, among steps with one)":
            float((ps.loc[has, "stabilization_progress"] < ps.loc[has, "first_decisional_progress"]).mean()),
        "fraction committing before first decisional mention (new, no-mention steps counted as before)":
            float(((ps["stabilization_progress"] < ps["first_decisional_progress"]) | ~has).mean()),
    }

    # ---- figure: three clocks CDF with the new mention clock
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    series = [("first decisional action mention", ps.loc[has, "first_decisional_progress"], "#2ca02c"),
              ("commitment prefix (argmax stabilisation)", ps["stabilization_progress"].dropna(), "#1f77b4"),
              ("permanent $c_{t,k} \\geq 0.8$", ps["permanent_cross_0_8_progress"].dropna(), "#ff7f0e")]
    for label, values, color in series:
        v = np.sort(values.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, linewidth=2.2, label=label)
    ax.set_xlabel("progress through the reasoning trace")
    ax.set_ylabel("fraction of decisions")
    ax.set_title("When does the preference settle? Three timing measures (CDF)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = REPO_ROOT / args.figure
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")

    # ---- report
    lines = ["# Context-aware action mentions (script 23)", "",
             f"- {len(steps)} decisions; {len(occ)} direction-word occurrences; token-offset "
             f"reconstruction mismatches: {mismatches}; seed {args.seed}.", "",
             "## Occurrence classes", "", occ["cls"].value_counts().rename_axis("class").to_frame("n").to_markdown(), "",
             "Decisional = answer | movement | uppercase | bare; non-decisional = idiom | descriptive | negated.", "",
             "## Occurrences by rule", "", occ["rule"].value_counts().rename_axis("rule").to_frame("n").to_markdown(), "",
             "## Examples per rule (random)", ""]
    for rule in occ["rule"].value_counts().index:
        sub = occ[occ["rule"] == rule]
        take = sub.sample(min(5, len(sub)), random_state=int(rng.integers(0, 2**31)))
        for _, r in take.iterrows():
            lines.append(f"- **{rule}** [{r.word}]: …{r.snippet}…")
    lines += ["", "## Pre-verbal control under each definition", "",
              "| definition | n prefixes | share | commitment first tenth -> final quarter | inverse entropy first tenth -> final quarter | top = final | n in final quarter |",
              "|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['n']:,} | {s['share']:.3f} | {s['commit_first_tenth']:.3f} -> {s['commit_final_quarter']:.3f} "
                     f"| {s['entropy_first_tenth']:.3f} -> {s['entropy_final_quarter']:.3f} | {s['top_is_final']:.3f} | {s['n_final_quarter']} |")
    lines += ["", "## Timing measures", ""] + [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in timing.items()]
    lines += ["", f"- Mask written to `{args.mask}` (columns preverbal_old / preverbal_decisional / "
              f"preverbal_decisional_final / preverbal_incl_negated); figure `{args.figure}`."]
    (REPO_ROOT / args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
