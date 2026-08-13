#!/usr/bin/env python3
"""Conjecture C5: is the post-commitment tail a verification phase, and is it cuttable?

Three tests.

(i) ONLINE EARLY EXIT. The conjecture as originally stated ("force the answer at the
    permanent 0.8 crossing") is vacuous: at that cut commitment_to_final >= 0.8 > 0.5, so
    the argmax equals the taken action by construction, and the crossing time is not
    knowable online. The non-vacuous version uses a PROSPECTIVE stopping rule,
    tau_c = first prefix whose margin >= c, decidable at the moment it fires, and asks how
    often the answer at tau_c equals the action the full CoT produced, and how many tokens
    that saves.
(ii) TAIL LEXICON. Is text generated after argmax stabilization enriched in verification
    vocabulary relative to text before it (paired within step, length-normalized)?
(iii) LATE REVERSALS. Do steps that reverse from a high-commitment state carry more
    correction vocabulary in their pre-stabilization region?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEXICONS = {
    "verification": re.compile(
        r"\b(verify|verif\w+|check|checks|checking|confirm\w*|double-?check|re-?check|ensure|"
        r"valid\w*|consistent|correct\b|makes sense|indeed|so the answer|therefore|thus|hence)\b",
        re.IGNORECASE),
    "correction": re.compile(
        r"\b(wait|but|actually|however|hmm|oops|mistake|wrong|instead|re-?consider|scratch that)\b",
        re.IGNORECASE),
    "action_word": re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE),
    "spatial": re.compile(r"\b(key|door|wall|grid|row|col(?:umn)?|room|cell|position|coordinate)\b"
                          r"|\(\s*\d+\s*,\s*\d+\s*\)", re.IGNORECASE),
    "plan": re.compile(r"\b(plan|path|route|then|first|next|need to|go to|move)\b", re.IGNORECASE),
}

PERSISTENCE = (1, 3, 5)      # consecutive prefixes the margin must stay above the threshold
TOKEN_FLOORS = (0, 200, 500)  # minimum tokens emitted before the rule may fire

INK = "#333333"
GRID = "#e6e6e6"
LEXICON_COLORS = {"verification": "#2a78d6", "correction": "#eb6834", "action_word": "#1baf7a",
                  "spatial": "#4a3aa7", "plan": "#9aa0a6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix_metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    parser.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/conjectures/c5_tail"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/conjectures"))
    return parser.parse_args()


def density(text: str, pattern: re.Pattern) -> float:
    """Marker hits per 1000 characters (length-normalized so head/tail are comparable)."""
    if not text:
        return float("nan")
    return 1000.0 * len(pattern.findall(text)) / len(text)


def main() -> int:
    args = parse_args()
    import pyarrow.parquet as pq
    from scipy.stats import mannwhitneyu, wilcoxon

    metrics = pd.read_parquet(args.prefix_metrics, columns=[
        "trajectory_id", "step_id", "prefix_id", "prefix_token_index", "progress", "top_action",
        "agent_action", "margin_commitment", "commitment_to_final", "num_analysis_tokens"])
    metrics = metrics.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"])

    step_rows: list[dict[str, Any]] = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for (trajectory_id, step_id), group in metrics.groupby(["trajectory_id", "step_id"], sort=False):
        top = group["top_action"].to_numpy()
        margin = group["margin_commitment"].to_numpy(dtype=float)
        commitment = group["commitment_to_final"].to_numpy(dtype=float)
        tokens = group["prefix_token_index"].to_numpy(dtype=float)
        prefix_ids = group["prefix_id"].to_numpy()
        total_tokens = float(group["num_analysis_tokens"].iloc[0])
        agent_action = str(group["agent_action"].iloc[0])
        differs = top != top[-1]
        stab_index = (int(np.max(np.nonzero(differs)[0])) + 1) if differs.any() else 0

        record: dict[str, Any] = {
            "trajectory_id": trajectory_id, "step_id": int(step_id),
            "agent_action": agent_action, "num_tokens": total_tokens,
            "stab_prefix_id": int(prefix_ids[stab_index]),
            "final_prefix_id": int(prefix_ids[-1]),
            "stab_progress": float(tokens[stab_index] / total_tokens),
        }
        # A rule may only use quantities available at the moment it fires: the margin so
        # far, how many tokens have been emitted, and how long the margin has persisted.
        # Total CoT length is NOT available (the remaining tokens do not exist yet).
        above = margin >= 0.0
        for threshold in thresholds:
            above = margin >= threshold
            run_length = np.zeros(len(margin), dtype=int)
            for position in range(len(margin)):
                run_length[position] = (run_length[position - 1] + 1) if (
                    above[position] and position > 0) else int(above[position])
            for persistence in PERSISTENCE:
                for floor in TOKEN_FLOORS:
                    eligible = (run_length >= persistence) & (tokens >= floor)
                    key = f"exit_c{threshold}_k{persistence}_f{floor}"
                    hits = np.nonzero(eligible)[0]
                    if len(hits):
                        stop = int(hits[0])
                        record[f"{key}_correct"] = bool(top[stop] == agent_action)
                        record[f"{key}_savings"] = float(1.0 - tokens[stop] / total_tokens)
                        record[f"{key}_fired"] = True
                    else:
                        record[f"{key}_correct"] = True   # never fires -> full CoT is used
                        record[f"{key}_savings"] = 0.0
                        record[f"{key}_fired"] = False
        # A reversal out of an already-high-commitment state.
        switched = np.zeros(len(group), dtype=bool)
        switched[1:] = top[1:] != top[:-1]
        record["late_reversal"] = bool(np.any(switched[1:] & (commitment[:-1] >= 0.8)))
        step_rows.append(record)
    steps = pd.DataFrame(step_rows)

    needed = sorted(set(steps.stab_prefix_id) | set(steps.final_prefix_id))
    texts = pq.read_table(args.prefixes, columns=["prefix_id", "analysis_prefix_text"],
                          filters=[("prefix_id", "in", needed)]).to_pandas()
    text_of = dict(zip(texts.prefix_id, texts.analysis_prefix_text))

    lexicon_rows: list[dict[str, Any]] = []
    for row in steps.itertuples():
        head = text_of.get(row.stab_prefix_id) or ""
        full = text_of.get(row.final_prefix_id) or ""
        tail = full[len(head):] if full.startswith(head) else full[len(head):]
        entry = {"trajectory_id": row.trajectory_id, "step_id": row.step_id,
                 "late_reversal": row.late_reversal,
                 "head_chars": len(head), "tail_chars": len(tail)}
        for name, pattern in LEXICONS.items():
            entry[f"head_{name}"] = density(head, pattern)
            entry[f"tail_{name}"] = density(tail, pattern)
        lexicon_rows.append(entry)
    lexicon = pd.DataFrame(lexicon_rows)
    usable = lexicon[(lexicon.head_chars > 200) & (lexicon.tail_chars > 200)].copy()

    exit_rows = []
    for threshold in thresholds:
        for persistence in PERSISTENCE:
            for floor in TOKEN_FLOORS:
                key = f"exit_c{threshold}_k{persistence}_f{floor}"
                exit_rows.append({
                    "margin_threshold": threshold, "persistence": persistence,
                    "token_floor": floor,
                    "fire_rate": float(steps[f"{key}_fired"].mean()),
                    "accuracy": float(steps[f"{key}_correct"].mean()),
                    "mean_token_savings": float(steps[f"{key}_savings"].mean()),
                    "median_token_savings": float(steps[f"{key}_savings"].median()),
                })
    exits = pd.DataFrame(exit_rows)
    # Pareto frontier: rules not dominated on both accuracy and savings.
    frontier = []
    for row in exits.itertuples():
        dominated = ((exits.accuracy >= row.accuracy) & (exits.mean_token_savings >= row.mean_token_savings)
                     & ((exits.accuracy > row.accuracy) | (exits.mean_token_savings > row.mean_token_savings))
                     ).any()
        frontier.append(not dominated)
    exits["on_frontier"] = frontier

    comparison_rows = []
    for name in LEXICONS:
        head_values = usable[f"head_{name}"].to_numpy()
        tail_values = usable[f"tail_{name}"].to_numpy()
        stat = wilcoxon(tail_values, head_values)
        comparison_rows.append({
            "lexicon": name,
            "head_density": float(np.mean(head_values)),
            "tail_density": float(np.mean(tail_values)),
            "tail_minus_head": float(np.mean(tail_values - head_values)),
            "tail_over_head": float(np.mean(tail_values) / np.mean(head_values)),
            "frac_steps_tail_higher": float(np.mean(tail_values > head_values)),
            "wilcoxon_p": float(stat.pvalue),
        })
    comparisons = pd.DataFrame(comparison_rows)

    reversal_rows = []
    for name in LEXICONS:
        with_reversal = usable.loc[usable.late_reversal, f"head_{name}"].dropna()
        without = usable.loc[~usable.late_reversal, f"head_{name}"].dropna()
        stat = mannwhitneyu(with_reversal, without)
        reversal_rows.append({
            "lexicon": name,
            "head_density_late_reversal": float(with_reversal.mean()),
            "head_density_other": float(without.mean()),
            "ratio": float(with_reversal.mean() / without.mean()),
            "mannwhitney_p": float(stat.pvalue),
            "n_reversal": int(len(with_reversal)), "n_other": int(len(without)),
        })
    reversals = pd.DataFrame(reversal_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(args.out_dir / "steps.parquet", index=False)
    exits.to_csv(args.out_dir / "early_exit.csv", index=False)
    comparisons.to_csv(args.out_dir / "head_tail_lexicon.csv", index=False)
    reversals.to_csv(args.out_dir / "late_reversal_lexicon.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].scatter(exits.loc[~exits.on_frontier, "mean_token_savings"],
                    exits.loc[~exits.on_frontier, "accuracy"],
                    s=22, color="#9aa0a6", alpha=0.6, edgecolors="white", linewidths=0.4)
    frontier_points = exits[exits.on_frontier].sort_values("mean_token_savings")
    axes[0].plot(frontier_points["mean_token_savings"], frontier_points["accuracy"],
                 color="#2a78d6", linewidth=2, marker="o", markersize=6)
    axes[0].axhline(0.95, color=INK, linewidth=1.2, linestyle=":")
    axes[0].annotate("95% agreement", (0.02, 0.95), textcoords="offset points",
                     xytext=(0, 5), color=INK, fontsize=8)
    axes[0].annotate("blue: Pareto frontier over all online rules\n"
                     "(threshold x persistence x token floor)",
                     (0.03, 0.06), xycoords="axes fraction", color=INK, fontsize=8)
    axes[0].set_xlabel("mean token savings", color=INK)
    axes[0].set_ylabel("P(early answer = full-CoT action)", color=INK)
    axes[0].set_title("Best achievable online early exit", color=INK, fontsize=11)

    x_positions = np.arange(len(comparisons))
    width = 0.38
    axes[1].bar(x_positions - width / 2, comparisons["head_density"], width,
                color="#9aa0a6", label="before stabilization")
    axes[1].bar(x_positions + width / 2, comparisons["tail_density"], width,
                color="#2a78d6", label="after stabilization (tail)")
    axes[1].set_xticks(x_positions, comparisons["lexicon"], rotation=20, ha="right")
    axes[1].set_ylabel("marker hits per 1000 chars", color=INK)
    axes[1].set_title("Lexical content before vs after the belief settles", color=INK, fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK)
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C5: is the post-commitment tail verification, and is it cuttable?", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c5_tail_verification.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    best = exits[exits.accuracy >= 0.95]
    best_row = best.iloc[best["mean_token_savings"].argmax()] if len(best) else None
    best_any = exits.iloc[exits["accuracy"].argmax()]
    verification_row = comparisons.set_index("lexicon").loc["verification"]
    part_i = best_row is not None and best_row["mean_token_savings"] >= 0.20
    part_ii = bool(verification_row["tail_minus_head"] > 0 and verification_row["wilcoxon_p"] < 0.01)
    lines = [
        "# Conjecture C5: the post-commitment tail",
        "",
        f"- {len(steps)} steps; lexicon tests on {len(usable)} steps with >200 chars on both sides.",
        "",
        "## (i) Online early exit (prospective rules: margin threshold x persistence x token floor)",
        "",
        "Pareto-optimal rules only (a rule is dominated if another is at least as accurate",
        "and at least as cheap):",
        "",
        exits[exits.on_frontier].sort_values("mean_token_savings").round(4).to_markdown(index=False),
        "",
        (f"- Best rule reaching 95% agreement: margin >= {best_row['margin_threshold']}, "
         f"persistence {int(best_row['persistence'])}, floor {int(best_row['token_floor'])} tokens, "
         f"mean savings {best_row['mean_token_savings']:.4f}, accuracy {best_row['accuracy']:.4f}"
         if best_row is not None else
         f"- NO online rule reaches 95% agreement. Best accuracy achievable is "
         f"{best_any['accuracy']:.4f} (margin >= {best_any['margin_threshold']}, persistence "
         f"{int(best_any['persistence'])}, floor {int(best_any['token_floor'])}), at mean savings "
         f"{best_any['mean_token_savings']:.4f}."),
        "",
        "## (ii) Head vs tail lexicon (density per 1000 chars, paired within step)",
        "",
        comparisons.round(4).to_markdown(index=False),
        "",
        "## (iii) Pre-stabilization lexicon of steps with a high-commitment reversal",
        "",
        reversals.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"- (i) cuttable at >=20% savings with >=95% agreement: {part_i}",
        f"- (ii) tail enriched in verification vocabulary: {part_ii}",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(exits.round(4).to_string(index=False))
    print()
    print(comparisons.round(4).to_string(index=False))
    print()
    print(reversals.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
