#!/usr/bin/env python3
"""Conjecture C7: is CoT dominance disruptable self-conditioning? (disavowal insert)

Theorem 1 showed the forced-answer belief tracks the reasoning text rather than the
prompt's world state. C7 asks WHY: if the mechanism is an inferential authorship prior
("this reasoning is mine and is about this task"), then explicitly disavowing the CoT
should return probability mass to the prompt's action. If the mechanism is architectural
(recency / attention mass on nearby tokens), the disavowal should do little.

Same 396 mismatched + 396 matched rows as D4, one new condition each: a disavowal
sentence is appended to the analysis channel immediately before the forced suffix.
Paired with the D4 scores row-for-row, so every comparison is within-row.
"""

from __future__ import annotations

import argparse
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

DISAVOWAL = (
    "\n\nNote: the reasoning written above was copied from a different maze and does not "
    "describe the map given in this task. Disregard it entirely and choose the next move "
    "using only the map in the task description."
)

INK = "#333333"
GRID = "#e6e6e6"
SERIES_COLORS = {"p(prompt action), no disavowal": "#eb6834",
                 "p(prompt action), disavowal": "#2a78d6",
                 "p(CoT action), no disavowal": "#9aa0a6",
                 "p(CoT action), disavowal": "#1baf7a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatched", type=Path,
                        default=REPO_ROOT / "outputs/followups/mismatched_prompt_cot_prefixes.parquet")
    parser.add_argument("--matched", type=Path,
                        default=REPO_ROOT / "outputs/followups/matched_prompt_cot_prefixes_subset.parquet")
    parser.add_argument("--baseline_scores", type=Path,
                        default=REPO_ROOT / "outputs/followups/mismatched_prompt_cot_scores_extracted.parquet")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c7_disavowal")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from mlx_lm import load
    import mlx.core as mx

    mismatched = pd.read_parquet(args.mismatched)
    matched = pd.read_parquet(args.matched)
    if "prompt_agent_action" not in matched.columns:
        matched["prompt_agent_action"] = matched["agent_action"]
    if args.max_rows is not None:
        mismatched, matched = mismatched.head(args.max_rows), matched.head(args.max_rows)

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    forced_suffix_token_ids(tokenizer)
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "scores.partial.parquet"
    records: list[dict[str, Any]] = []
    for condition, frame in (("mismatched", mismatched), ("matched", matched)):
        for count, (_, row) in enumerate(frame.iterrows(), start=1):
            analysis = "" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"])
            text = (str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + analysis
                    + DISAVOWAL + FORCED_SUFFIX)
            token_ids = encode_without_special(tokenizer, text)
            logits = model(mx.array([token_ids], dtype=mx.int32))
            selected = logits[0, -1, candidate_ids].astype(mx.float32)
            mx.eval(selected)
            q_values = stable_action_probs(np.asarray(selected, dtype=np.float32))
            records.append({
                "condition": condition,
                "row_key": f"{condition}:{row['decision_id']}:{int(row['prefix_token_index'])}",
                "prompt_agent_action": row.get("prompt_agent_action"),
                "cot_agent_action": row.get("cot_agent_action"),
                "prefix_token_index": int(row["prefix_token_index"]),
                "num_analysis_tokens": int(row["num_analysis_tokens"]),
                "relative_cut": float(row["prefix_token_index"]) / float(row["num_analysis_tokens"]),
                **{column: float(q_values[index]) for index, column in enumerate(Q_COLS)},
                "top_action": ACTIONS[int(np.argmax(q_values))],
            })
            if count % 20 == 0 or count == len(frame):
                print(f"[{condition}] {count}/{len(frame)}", flush=True)
                pd.DataFrame(records).to_parquet(checkpoint, index=False)

    scores = pd.DataFrame(records)
    action_index = {action: position for position, action in enumerate(ACTIONS)}
    q_matrix = scores[Q_COLS].to_numpy(dtype=float)
    scores["p_prompt_action"] = [q_matrix[i, action_index[a]]
                                 for i, a in enumerate(scores["prompt_agent_action"].astype(str))]
    scores["p_cot_action"] = [q_matrix[i, action_index[a]]
                              for i, a in enumerate(scores["cot_agent_action"].astype(str))]
    scores["top_is_prompt_action"] = scores["top_action"] == scores["prompt_agent_action"]
    scores["top_is_cot_action"] = scores["top_action"] == scores["cot_agent_action"]
    scores.to_parquet(args.out_dir / "scores_disavowal.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    baseline = pd.read_parquet(args.baseline_scores)[
        ["row_key", "condition", "cut_bucket", "p_prompt_action", "p_cot_action",
         "top_is_prompt_action", "top_is_cot_action"]]
    paired = baseline.merge(scores, on=["row_key", "condition"], suffixes=("_base", "_dis"),
                            validate="one_to_one")
    paired["d_p_prompt"] = paired["p_prompt_action_dis"] - paired["p_prompt_action_base"]
    paired["d_p_cot"] = paired["p_cot_action_dis"] - paired["p_cot_action_base"]
    paired.to_parquet(args.out_dir / "paired.parquet", index=False)

    summary = paired.groupby(["condition", "cut_bucket"], observed=True).agg(
        p_prompt_base=("p_prompt_action_base", "mean"),
        p_prompt_dis=("p_prompt_action_dis", "mean"),
        d_p_prompt=("d_p_prompt", "mean"),
        p_cot_base=("p_cot_action_base", "mean"),
        p_cot_dis=("p_cot_action_dis", "mean"),
        top_prompt_base=("top_is_prompt_action_base", "mean"),
        top_prompt_dis=("top_is_prompt_action_dis", "mean"),
        n=("row_key", "size")).round(4).reset_index()
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    mism = paired[paired.condition == "mismatched"]
    lost = 1.0 - mism["p_prompt_action_base"].mean()
    recovered = mism["d_p_prompt"].mean()
    recovery_fraction = recovered / lost if lost > 0 else float("nan")
    from scipy.stats import wilcoxon
    stat_p = wilcoxon(mism["d_p_prompt"]).pvalue if len(mism) > 10 else float("nan")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    mism_summary = summary[summary.condition == "mismatched"]
    x_positions = np.arange(len(mism_summary))
    series = {"p(prompt action), no disavowal": mism_summary["p_prompt_base"].to_numpy(),
              "p(prompt action), disavowal": mism_summary["p_prompt_dis"].to_numpy(),
              "p(CoT action), no disavowal": mism_summary["p_cot_base"].to_numpy(),
              "p(CoT action), disavowal": mism_summary["p_cot_dis"].to_numpy()}
    for label, values in series.items():
        axis.plot(x_positions, values, color=SERIES_COLORS[label], linewidth=2,
                  marker="o", markersize=5,
                  linestyle="-" if "disavowal" == label.split(", ")[1] else "--")
        axis.annotate(label, (x_positions[-1], values[-1]), textcoords="offset points",
                      xytext=(6, 0), ha="left", color=INK, fontsize=8)
    axis.set_xticks(x_positions, mism_summary["cut_bucket"].astype(str))
    axis.set_xlim(-0.2, len(x_positions) + 2.4)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("CoT cut position", color=INK)
    axis.set_ylabel("mean forced-answer probability", color=INK)
    axis.set_title("C7: does disavowing the foreign CoT restore the prompt's action?",
                   color=INK, fontsize=11)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure_path = args.figure_dir / "c7_disavowal.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    lines = [
        "# Conjecture C7: disavowal insert",
        "",
        "Prediction: if CoT dominance is an inferential authorship prior, an explicit",
        "disavowal recovers >= half the prompt-action mass lost in the mismatched condition",
        "(p(prompt action) 0.15 -> >= 0.4) while barely moving matched controls.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Verdict statistics (mismatched rows)",
        "",
        f"- mean p(prompt action): {mism['p_prompt_action_base'].mean():.4f} -> "
        f"{mism['p_prompt_action_dis'].mean():.4f} (delta {recovered:+.4f})",
        f"- recovery fraction of the mass not on the prompt action: {recovery_fraction:.4f}",
        f"- fraction of rows where the disavowal increases p(prompt action): "
        f"{(mism['d_p_prompt'] > 0).mean():.4f}",
        f"- Wilcoxon signed-rank p on the paired change: {stat_p:.3e}",
        f"- top action becomes the prompt action in "
        f"{mism['top_is_prompt_action_dis'].mean():.4f} of rows "
        f"(baseline {mism['top_is_prompt_action_base'].mean():.4f})",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir}/report.md and {figure_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
