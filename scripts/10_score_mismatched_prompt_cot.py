#!/usr/bin/env python3
"""Score the staged prompt/CoT mismatch counterfactuals with the trusted MLX path (D4).

Each mismatched row pairs the PROMPT of decision A with a CoT prefix of decision B whose
agent took a different action, cut at ~10%/50%/90% of the CoT. Forcing the answer suffix
and reading the four action-token logits asks: does the model's belief track the prompt's
world state or the reasoning text in front of it? Matched controls (prompt and CoT from
the same decision, same cut positions) are scored through the identical runtime path.

Safety: every row is a full, independent forward pass — no KV-cache reuse anywhere
(cache reuse is the suspect path behind the quarantined stage-03 labels). Matched rows
are additionally compared against `extracted_beliefs.parquet`, which re-validates the
extraction labels through a cache-free route.
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

INK = "#333333"
GRID = "#e6e6e6"
SERIES_COLORS = {
    "mismatched: p(CoT action)": "#2a78d6",
    "mismatched: p(prompt action)": "#eb6834",
    "matched control: p(agent action)": "#1baf7a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatched", type=Path,
                        default=Path("outputs/followups/mismatched_prompt_cot_prefixes.parquet"))
    parser.add_argument("--matched", type=Path,
                        default=Path("outputs/followups/matched_prompt_cot_prefixes_subset.parquet"))
    parser.add_argument("--beliefs", type=Path, default=Path(
        "outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations/extracted_beliefs.parquet"))
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/followups/mismatched_prompt_cot_scores_extracted.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/followups/mismatched_prompt_cot_report.md"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/counterfactual"))
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=None, help="Debug cap on scored rows.")
    return parser.parse_args()


def action_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        if len(encoded) != 1:
            raise ValueError(f"action candidate {action} encoded to {len(encoded)} tokens, expected one")
        ids.append(encoded[0])
    return ids


def score_rows(frame: pd.DataFrame, condition: str, model: Any, tokenizer: Any,
               candidate_ids: list[int], mx: Any, checkpoint_path: Path,
               checkpoint_every: int, done: list[dict[str, Any]]) -> None:
    for count, (_, row) in enumerate(frame.iterrows(), start=1):
        text = (str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER
                + ("" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"]))
                + FORCED_SUFFIX)
        token_ids = encode_without_special(tokenizer, text)
        logits = model(mx.array([token_ids], dtype=mx.int32))
        selected = logits[0, -1, candidate_ids].astype(mx.float32)
        mx.eval(selected)
        q_values = stable_action_probs(np.asarray(selected, dtype=np.float32))
        record = {
            "condition": condition,
            "row_key": f"{condition}:{row['decision_id']}:{int(row['prefix_token_index'])}",
            "decision_id": row["decision_id"],
            "trajectory_id": row["trajectory_id"],
            "step_id": int(row["step_id"]),
            "prefix_id": str(row["prefix_id"]),
            "prefix_token_index": int(row["prefix_token_index"]),
            "num_analysis_tokens": int(row["num_analysis_tokens"]),
            "relative_cut": float(row["prefix_token_index"]) / float(row["num_analysis_tokens"]),
            "prompt_agent_action": row.get("prompt_agent_action"),
            "cot_agent_action": row.get("cot_agent_action"),
            "n_tokens": len(token_ids),
            **{column: float(q_values[index]) for index, column in enumerate(Q_COLS)},
            "top_action": ACTIONS[int(np.argmax(q_values))],
        }
        done.append(record)
        if count % 20 == 0 or count == len(frame):
            print(f"[{condition}] {count}/{len(frame)} scored (last {len(token_ids)} tokens)", flush=True)
        if len(done) % checkpoint_every == 0:
            pd.DataFrame(done).to_parquet(checkpoint_path, index=False)


def quantile(series: pd.Series) -> str:
    clean = series.dropna()
    return f"{clean.mean():.4f} (median {clean.median():.4f}, n={len(clean)})" if len(clean) else "NA"


def main() -> int:
    args = parse_args()
    from mlx_lm import load

    mismatched = pd.read_parquet(args.mismatched)
    matched = pd.read_parquet(args.matched)
    if "prompt_agent_action" not in matched.columns:
        matched["prompt_agent_action"] = matched["agent_action"]
    if args.max_rows is not None:
        mismatched = mismatched.head(args.max_rows)
        matched = matched.head(args.max_rows)

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    import mlx.core as mx

    suffix_ids = forced_suffix_token_ids(tokenizer)
    print(f"forced suffix validated: {len(suffix_ids)} tokens")
    candidate_ids = action_token_ids(tokenizer)

    checkpoint_path = args.output.with_suffix(".partial.parquet")
    done: list[dict[str, Any]] = []
    score_rows(mismatched, "mismatched", model, tokenizer, candidate_ids, mx,
               checkpoint_path, args.checkpoint_every, done)
    score_rows(matched, "matched", model, tokenizer, candidate_ids, mx,
               checkpoint_path, args.checkpoint_every, done)
    scores = pd.DataFrame(done)

    action_index = {action: position for position, action in enumerate(ACTIONS)}
    q_matrix = scores[Q_COLS].to_numpy(dtype=float)
    scores["p_prompt_action"] = [
        q_matrix[i, action_index[a]] if a in action_index else np.nan
        for i, a in enumerate(scores["prompt_agent_action"].astype(str))]
    scores["p_cot_action"] = [
        q_matrix[i, action_index[a]] if a in action_index else np.nan
        for i, a in enumerate(scores["cot_agent_action"].astype(str))]
    scores["top_is_prompt_action"] = scores["top_action"] == scores["prompt_agent_action"]
    scores["top_is_cot_action"] = scores["top_action"] == scores["cot_agent_action"]
    scores["cut_bucket"] = pd.cut(scores["relative_cut"], bins=[0.0, 0.3, 0.7, 1.0],
                                  labels=["early (~10%)", "mid (~50%)", "late (~90%)"],
                                  include_lowest=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.output, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    beliefs = pd.read_parquet(args.beliefs,
                              columns=["trajectory_id", "step_id", "prefix_token_index", *Q_COLS])
    matched_scores = scores[scores.condition == "matched"].merge(
        beliefs, on=["trajectory_id", "step_id", "prefix_token_index"],
        how="left", suffixes=("", "_extracted"))
    extracted_cols = [f"{column}_extracted" for column in Q_COLS]
    have_reference = matched_scores[extracted_cols].notna().all(axis=1)
    if have_reference.any():
        fresh = matched_scores.loc[have_reference, Q_COLS].to_numpy(dtype=float)
        reference = matched_scores.loc[have_reference, extracted_cols].to_numpy(dtype=float)
        revalidation_dq = float(np.abs(fresh - reference).max(axis=1).mean())
        revalidation_top = float((fresh.argmax(axis=1) == reference.argmax(axis=1)).mean())
        revalidation_n = int(have_reference.sum())
    else:
        revalidation_dq, revalidation_top, revalidation_n = float("nan"), float("nan"), 0

    summary = scores.groupby(["condition", "cut_bucket"], observed=True).agg(
        p_prompt_action=("p_prompt_action", "mean"),
        p_cot_action=("p_cot_action", "mean"),
        top_is_prompt=("top_is_prompt_action", "mean"),
        top_is_cot=("top_is_cot_action", "mean"),
        n=("row_key", "size")).round(4).reset_index()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    mismatched_summary = summary[summary.condition == "mismatched"]
    matched_summary = summary[summary.condition == "matched"]
    x_positions = np.arange(len(mismatched_summary))
    series = {
        "mismatched: p(CoT action)": mismatched_summary["p_cot_action"].to_numpy(),
        "mismatched: p(prompt action)": mismatched_summary["p_prompt_action"].to_numpy(),
        "matched control: p(agent action)": matched_summary["p_prompt_action"].to_numpy(),
    }
    for label, values in series.items():
        axis.plot(x_positions, values, color=SERIES_COLORS[label], linewidth=2, marker="o", markersize=5)
        axis.annotate(label, (x_positions[-1], values[-1]), textcoords="offset points",
                      xytext=(6, 0), ha="left", color=INK, fontsize=9)
    axis.set_xticks(x_positions, mismatched_summary["cut_bucket"].astype(str))
    axis.set_xlim(-0.2, len(x_positions) + 1.6)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("CoT cut position", color=INK)
    axis.set_ylabel("mean forced-answer probability", color=INK)
    axis.set_title("Whose action does the belief track: the prompt's or the CoT's?", color=INK, fontsize=11)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure_path = args.figure_dir / "mismatched_prompt_cot_tracking.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    lines = [
        "# Prompt/CoT Mismatch Counterfactuals: scored (D4)",
        "",
        f"- {int((scores.condition == 'mismatched').sum())} mismatched rows + "
        f"{int((scores.condition == 'matched').sum())} matched controls; full independent forward",
        "  passes, no KV-cache reuse; forced suffix validated on load.",
        "",
        "## Belief tracking by cut position",
        "",
        summary.to_markdown(index=False),
        "",
        "## Cache-free re-validation of extraction labels (matched rows)",
        "",
        f"- rows with extracted_beliefs reference: {revalidation_n}",
        f"- mean max|dq| fresh-vs-extracted: {revalidation_dq:.4f}",
        f"- top-action agreement fresh-vs-extracted: {revalidation_top:.4f}",
        "",
        "## Figure",
        "",
        f"- `{figure_path}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(scores)} rows), {args.report}, {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
