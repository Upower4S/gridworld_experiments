#!/usr/bin/env python3
"""Conjecture C3: are belief-update anchors causal forks, or just markers of past updates?

Proposition 2.6 identified anchors observationally: transitions carrying the top 5% of
belief change, 75% of which coincide with an argmax switch. That is compatible with two
readings. (a) CAUSAL: an anchor is a fork -- what gets sampled there determines where the
belief goes. (b) EPIPHENOMENAL: the update had already happened internally and the anchor
chunk merely reports it, in which case resampling that chunk changes little.

Test (Thought-Anchors style, arXiv 2506.19143, adapted to a belief functional): branch at
the prefix BEFORE the transition, resample the next chunk R times at temperature 1, force
the answer after each continuation, and measure how much the resulting beliefs DISPERSE.
Controls are position-matched non-anchor transitions from the same step, so dispersion
differences cannot be attributed to being late or early in the CoT.

Generation uses its own fresh cache per sample; scoring the forced answer is always a
separate cache-free forward.
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
KIND_COLORS = {"anchor": "#eb6834", "control": "#2a78d6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=REPO_ROOT / "outputs/belief_update_anchors.parquet")
    parser.add_argument("--prefix_metrics", type=Path,
                        default=REPO_ROOT / "outputs/prefix_commitment_metrics_extracted.parquet")
    parser.add_argument("--prefixes", type=Path, default=REPO_ROOT / "outputs/prefixes.parquet")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c3_anchors")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--per_kind", type=int, default=2, help="Anchors (and controls) per step.")
    parser.add_argument("--resamples", type=int, default=10)
    parser.add_argument("--gen_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def generate_continuation(model: Any, mx: Any, prompt_ids: list[int], *, max_tokens: int,
                          temperature: float, key: Any) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    generated: list[int] = []
    for _ in range(max_tokens):
        scaled = logits[0, -1].astype(mx.float32) / temperature
        key, subkey = mx.random.split(key)
        token = int(mx.random.categorical(scaled, key=subkey).item())
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
    return generated


def forced_answer(model: Any, mx: Any, token_ids: list[int], candidate_ids: list[int]) -> np.ndarray:
    logits = model(mx.array([token_ids], dtype=mx.int32))
    selected = logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(selected)
    return stable_action_probs(np.asarray(selected, dtype=np.float32))


def mean_pairwise_tv(matrix: np.ndarray) -> float:
    total, count = 0.0, 0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            total += 0.5 * np.abs(matrix[i] - matrix[j]).sum()
            count += 1
    return total / max(count, 1)


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_steps, args.per_kind, args.resamples, args.gen_tokens = 1, 1, 3, 20
    rng = np.random.default_rng(args.seed)

    transitions = pd.read_parquet(args.anchors)
    metrics = pd.read_parquet(args.prefix_metrics, columns=[
        "trajectory_id", "step_id", "prefix_id", "prefix_token_index", "progress"])
    metrics = metrics.sort_values(["trajectory_id", "step_id", "prefix_token_index"])
    # transitions[i] compares prefix i+1 to prefix i within a step: recover the branch prefix.
    metrics["order"] = metrics.groupby(["trajectory_id", "step_id"], sort=False).cumcount()
    transitions["order"] = transitions.groupby(["trajectory_id", "step_id"], sort=False).cumcount()
    branch = metrics.rename(columns={"prefix_id": "branch_prefix_id",
                                     "prefix_token_index": "branch_token_index"})
    transitions = transitions.merge(
        branch[["trajectory_id", "step_id", "order", "branch_prefix_id", "branch_token_index"]],
        on=["trajectory_id", "step_id", "order"], how="inner", validate="one_to_one")

    eligible = transitions.groupby(["trajectory_id", "step_id"]).filter(
        lambda group: group.is_anchor.sum() >= args.per_kind
        and (~group.is_anchor).sum() >= args.per_kind)
    step_keys = eligible[["trajectory_id", "step_id"]].drop_duplicates()
    chosen_steps = step_keys.sample(n=min(args.n_steps, len(step_keys)), random_state=args.seed)
    print(f"[c3] {len(chosen_steps)} steps from {len(step_keys)} eligible")

    selections: list[dict[str, Any]] = []
    for _, key in chosen_steps.iterrows():
        group = eligible[(eligible.trajectory_id == key.trajectory_id)
                         & (eligible.step_id == key.step_id)]
        anchors = group[group.is_anchor].nlargest(args.per_kind, "tv")
        non_anchors = group[~group.is_anchor]
        for _, anchor in anchors.iterrows():
            selections.append({**anchor.to_dict(), "kind": "anchor"})
            # Position-matched control: nearest non-anchor in progress, not already used.
            used = {entry["branch_prefix_id"] for entry in selections}
            available = non_anchors[~non_anchors.branch_prefix_id.isin(used)]
            if available.empty:
                continue
            nearest = available.iloc[
                (available.transition_progress - anchor.transition_progress).abs().argmin()]
            selections.append({**nearest.to_dict(), "kind": "control"})
    selection_frame = pd.DataFrame(selections)
    print(f"[c3] {len(selection_frame)} branch points "
          f"({(selection_frame.kind == 'anchor').sum()} anchors)")

    texts = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    text_of = dict(zip(texts.prefix_id, zip(texts.prompt_text, texts.analysis_prefix_text)))

    from mlx_lm import load
    import mlx.core as mx

    print(f"[c3] loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    forced_suffix_token_ids(tokenizer)
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])
    key = mx.random.key(args.seed)

    records: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "resamples.partial.parquet"
    for number, row in enumerate(selection_frame.itertuples(), start=1):
        prompt_text, analysis = text_of[int(row.branch_prefix_id)]
        branch_text = (str(prompt_text).rstrip() + ANALYSIS_HEADER
                       + ("" if analysis is None else str(analysis)))
        branch_ids = encode_without_special(tokenizer, branch_text)
        suffix_ids = encode_without_special(tokenizer, FORCED_SUFFIX)

        q_values = []
        for replicate in range(args.resamples):
            key, subkey = mx.random.split(key)
            continuation = generate_continuation(
                model, mx, branch_ids, max_tokens=args.gen_tokens,
                temperature=args.temperature, key=subkey)
            q_after = forced_answer(model, mx, branch_ids + continuation + suffix_ids, candidate_ids)
            q_values.append(q_after)
            records.append({
                "trajectory_id": row.trajectory_id, "step_id": int(row.step_id),
                "kind": row.kind, "branch_prefix_id": int(row.branch_prefix_id),
                "transition_progress": float(row.transition_progress),
                "observed_tv": float(row.tv), "replicate": replicate,
                **{c: float(q_after[i]) for i, c in enumerate(Q_COLS)},
                "top_action": ACTIONS[int(q_after.argmax())],
            })
        print(f"[c3] branch {number}/{len(selection_frame)} ({row.kind}) done", flush=True)
        pd.DataFrame(records).to_parquet(checkpoint, index=False)

    frame = pd.DataFrame(records)
    frame.to_parquet(args.out_dir / "resamples.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    branch_rows = []
    for (trajectory_id, step_id, branch_prefix_id), group in frame.groupby(
            ["trajectory_id", "step_id", "branch_prefix_id"], sort=False):
        matrix = group[Q_COLS].to_numpy(dtype=float)
        counts = group["top_action"].value_counts(normalize=True)
        entropy = float(-(counts * np.log(counts)).sum())
        branch_rows.append({
            "trajectory_id": trajectory_id, "step_id": step_id,
            "branch_prefix_id": branch_prefix_id,
            "kind": group["kind"].iloc[0],
            "transition_progress": float(group["transition_progress"].iloc[0]),
            "observed_tv": float(group["observed_tv"].iloc[0]),
            "dispersion_tv": mean_pairwise_tv(matrix),
            "argmax_entropy": entropy,
            "distinct_argmax": int(group["top_action"].nunique()),
            "n": len(group),
        })
    branches = pd.DataFrame(branch_rows)
    branches.to_csv(args.out_dir / "branch_dispersion.csv", index=False)

    from scipy.stats import mannwhitneyu
    anchor_values = branches.loc[branches.kind == "anchor", "dispersion_tv"]
    control_values = branches.loc[branches.kind == "control", "dispersion_tv"]
    test = mannwhitneyu(anchor_values, control_values)
    ratio = float(anchor_values.mean() / control_values.mean()) if len(control_values) else float("nan")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for kind in ("anchor", "control"):
        values = branches.loc[branches.kind == kind, "dispersion_tv"].to_numpy()
        ordered = np.sort(values)
        axes[0].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                     color=KIND_COLORS[kind], linewidth=2)
        axes[1].scatter(branches.loc[branches.kind == kind, "transition_progress"], values,
                        s=26, alpha=0.7, color=KIND_COLORS[kind], edgecolors="white",
                        linewidths=0.5, label=kind)
    axes[0].set_xlabel("dispersion of resampled beliefs (mean pairwise TV)", color=INK)
    axes[0].set_ylabel("fraction of branch points", color=INK)
    axes[0].set_title("Do anchors fork the belief more?", color=INK, fontsize=11)
    axes[0].legend(handles=[plt.Line2D([], [], color=KIND_COLORS[k], linewidth=2, label=k)
                            for k in ("anchor", "control")],
                   frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    axes[1].set_xlabel("progress at branch point", color=INK)
    axes[1].set_ylabel("dispersion (mean pairwise TV)", color=INK)
    axes[1].set_title("Position-matched controls", color=INK, fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK)
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C3: are belief-update anchors causal forks?", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c3_anchor_resampling.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    verdict = "SUPPORTED" if ratio >= 2.0 and test.pvalue < 0.05 else (
        "PARTIAL" if ratio > 1.0 and test.pvalue < 0.05 else "NOT SUPPORTED")
    lines = [
        "# Conjecture C3: anchor causality by resampling",
        "",
        f"- {len(branches)} branch points ({int((branches.kind == 'anchor').sum())} anchors, "
        f"{int((branches.kind == 'control').sum())} position-matched controls) from "
        f"{branches.step_id.nunique()} steps; {args.resamples} resamples each, "
        f"{args.gen_tokens} tokens at temperature {args.temperature}.",
        "",
        branches.groupby("kind")[["dispersion_tv", "argmax_entropy", "distinct_argmax",
                                  "observed_tv", "transition_progress"]].mean().round(4).to_markdown(),
        "",
        "## Verdict",
        "",
        f"- dispersion ratio anchor/control: {ratio:.4f} (criterion for SUPPORTED: >= 2.0)",
        f"- Mann-Whitney p: {test.pvalue:.4g}",
        f"- mean distinct sampled answers: anchors "
        f"{branches.loc[branches.kind == 'anchor', 'distinct_argmax'].mean():.3f} vs controls "
        f"{branches.loc[branches.kind == 'control', 'distinct_argmax'].mean():.3f}",
        f"- **{verdict}**",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(branches.groupby("kind")[["dispersion_tv", "argmax_entropy", "distinct_argmax"]].mean())
    print(f"\nVerdict: {verdict} (ratio {ratio:.4f}, p={test.pvalue:.4g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
