#!/usr/bin/env python3
"""Conjecture C2: is layer 10 a routing bottleneck of the MODEL, not just of the probes?

Proposition 3.3 found that layer 10 is the only layer whose occlusion reliably damages the
trained probes. But occlusion importance is a property of a probe's learned shortcuts, so
it cannot by itself establish that layer 10 matters to the model's own computation. C2 says
it does, and specifically that the ATTENTION sub-block around layers 10-11 is where spatial
evidence is routed into an action preference.

Test: run the model forward manually and skip one sub-block (attention or MLP) at one layer,
then read the forced-answer belief. If C2 holds, attention ablation near layer 10-11 damages
the belief more than at neighbouring layers, and more than MLP ablation there.

Every forward is independent and cache-free. A manual-vs-standard forward equivalence check
runs first, so any deviation is attributable to the ablation and not to the reimplementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
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
STRICT_COL = "strict_before_any_action_mention"

INK = "#333333"
GRID = "#e6e6e6"
PART_COLORS = {"attn": "#2a78d6", "mlp": "#eb6834"}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--prefixes", type=Path, default=REPO_ROOT / "outputs/prefixes.parquet")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c2_subblock")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--n_prefixes", type=int, default=24)
    parser.add_argument("--layers", type=str, default="0,2,4,6,8,9,10,11,12,14,16,18,20,22")
    parser.add_argument("--commitment_band", type=str, default="0.2,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def manual_forward(model: Any, mx: Any, create_attention_mask: Any, token_ids: list[int],
                   candidate_ids: list[int], *, ablate_layer: int | None = None,
                   ablate_part: str | None = None) -> np.ndarray:
    transformer = model.model
    layers = list(transformer.layers)
    layer_types = list(transformer.layer_types)
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types, strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        skip_attention = index == ablate_layer and ablate_part == "attn"
        skip_mlp = index == ablate_layer and ablate_part == "mlp"
        residual = x
        if not skip_attention:
            x = residual + layer.self_attn(layer.input_layernorm(x), mask, None)
        residual = x
        if not skip_mlp:
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
    logits = model.lm_head(transformer.norm(x))
    selected = logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(selected)
    return stable_action_probs(np.asarray(selected, dtype=np.float32))


def entropy_commitment_of(q: np.ndarray) -> float:
    return float(1.0 + (q * np.log(np.clip(q, 1e-12, None))).sum() / np.log(len(ACTIONS)))


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_prefixes, args.layers = 3, "10,11"
    layers = [int(token) for token in args.layers.split(",") if token.strip()]
    low_band, high_band = (float(value) for value in args.commitment_band.split(","))

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == 0)
    strict = table[STRICT_COL].to_numpy(dtype=bool)
    test_idx = split["test_idx"][strict[split["test_idx"]]]

    q_all = table[Q_COLS].to_numpy(dtype=np.float64)
    agent_actions = table["agent_action"].astype(str).to_numpy()
    action_index = {action: position for position, action in enumerate(ACTIONS)}
    commitment_to_final = np.array([q_all[i, action_index[a]] for i, a in enumerate(agent_actions)])
    band = ((commitment_to_final[test_idx] >= low_band) & (commitment_to_final[test_idx] <= high_band))
    candidates = test_idx[band]
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(candidates, size=min(args.n_prefixes, len(candidates)), replace=False))
    print(f"[c2] {len(chosen)} prefixes from {len(candidates)} in-band strict fold-0 test rows")

    texts = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    rows = table.iloc[chosen][["prefix_id"]].merge(texts, on="prefix_id", validate="one_to_one")

    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    import mlx.core as mx

    print(f"[c2] loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    forced_suffix_token_ids(tokenizer)
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])

    first = rows.iloc[0]
    check_text = (str(first["prompt_text"]).rstrip() + ANALYSIS_HEADER
                  + ("" if first["analysis_prefix_text"] is None else str(first["analysis_prefix_text"]))
                  + FORCED_SUFFIX)
    check_ids = encode_without_special(tokenizer, check_text)
    manual_q = manual_forward(model, mx, create_attention_mask, check_ids, candidate_ids)
    standard_logits = model(mx.array([check_ids], dtype=mx.int32))
    standard_selected = standard_logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(standard_selected)
    standard_q = stable_action_probs(np.asarray(standard_selected, dtype=np.float32))
    equivalence = float(np.abs(manual_q - standard_q).max())
    print(f"[c2] manual-vs-standard forward max|dq| = {equivalence:.3e}")
    if equivalence > 1e-3:
        raise SystemExit(f"manual forward does not reproduce the standard forward ({equivalence:.3e})")

    records: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "ablation.partial.parquet"
    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        text = (str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER
                + ("" if row["analysis_prefix_text"] is None else str(row["analysis_prefix_text"]))
                + FORCED_SUFFIX)
        token_ids = encode_without_special(tokenizer, text)
        base_q = manual_forward(model, mx, create_attention_mask, token_ids, candidate_ids)
        base_commitment = entropy_commitment_of(base_q)
        base_top = int(base_q.argmax())
        records.append({"prefix_id": int(row["prefix_id"]), "layer": -1, "part": "baseline",
                        **{c: float(base_q[i]) for i, c in enumerate(Q_COLS)},
                        "entropy_commitment": base_commitment, "top_action": ACTIONS[base_top],
                        "kl_from_base": 0.0, "n_tokens": len(token_ids)})
        for layer in layers:
            for part in ("attn", "mlp"):
                q_values = manual_forward(model, mx, create_attention_mask, token_ids, candidate_ids,
                                          ablate_layer=layer, ablate_part=part)
                kl = float((base_q * (np.log(np.clip(base_q, 1e-12, None))
                                      - np.log(np.clip(q_values, 1e-12, None)))).sum())
                records.append({
                    "prefix_id": int(row["prefix_id"]), "layer": layer, "part": part,
                    **{c: float(q_values[i]) for i, c in enumerate(Q_COLS)},
                    "entropy_commitment": entropy_commitment_of(q_values),
                    "top_action": ACTIONS[int(q_values.argmax())],
                    "kl_from_base": kl, "n_tokens": len(token_ids),
                    "base_top_action": ACTIONS[base_top],
                    "base_entropy_commitment": base_commitment,
                })
        print(f"[c2] prefix {number}/{len(rows)} ({len(records)} records)", flush=True)
        pd.DataFrame(records).to_parquet(checkpoint, index=False)

    frame = pd.DataFrame(records)
    frame.to_parquet(args.out_dir / "ablation.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    ablated = frame[frame.part != "baseline"].copy()
    ablated["flipped"] = ablated["top_action"] != ablated["base_top_action"]
    ablated["d_commitment"] = ablated["entropy_commitment"] - ablated["base_entropy_commitment"]
    summary = ablated.groupby(["part", "layer"]).agg(
        mean_kl=("kl_from_base", "mean"), median_kl=("kl_from_base", "median"),
        flip_rate=("flipped", "mean"), mean_d_commitment=("d_commitment", "mean"),
        n=("prefix_id", "size")).round(4).reset_index()
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for part in ("attn", "mlp"):
        subset = summary[summary.part == part].sort_values("layer")
        axes[0].plot(subset["layer"], subset["mean_kl"], color=PART_COLORS[part],
                     linewidth=2, marker="o", markersize=4)
        axes[1].plot(subset["layer"], subset["flip_rate"], color=PART_COLORS[part],
                     linewidth=2, marker="o", markersize=4)
    axes[0].set_ylabel("mean KL(baseline || ablated)", color=INK)
    axes[0].set_title("Belief damage from skipping a sub-block", color=INK, fontsize=11)
    axes[1].set_ylabel("top-action flip rate", color=INK)
    axes[1].set_title("Argmax flips from skipping a sub-block", color=INK, fontsize=11)
    axes[0].legend(handles=[plt.Line2D([], [], color=PART_COLORS[p], linewidth=2,
                                       label={"attn": "attention", "mlp": "MLP"}[p])
                            for p in ("attn", "mlp")],
                   frameon=False, fontsize=9, labelcolor=INK)
    for axis in axes:
        axis.set_xlabel("ablated layer", color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C2: which sub-blocks does the forced-answer belief depend on?", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c2_subblock_ablation.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    attention = summary[summary.part == "attn"].set_index("layer")
    near = [layer for layer in (10, 11) if layer in attention.index]
    far = [layer for layer in attention.index if layer not in (10, 11)]
    near_kl = float(attention.loc[near, "mean_kl"].mean()) if near else float("nan")
    far_kl = float(attention.loc[far, "mean_kl"].mean())
    peak_layer = int(attention["mean_kl"].idxmax())
    verdict = "SUPPORTED" if near_kl > 2 * far_kl else "NOT SUPPORTED"

    lines = [
        "# Conjecture C2: is layer 10-11 attention a routing bottleneck?",
        "",
        f"- {len(rows)} strict fold-0 test prefixes; layers {layers}; cache-free manual forwards.",
        f"- Manual-vs-standard forward equivalence: max|dq| = {equivalence:.3e}.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"- attention-ablation mean KL at layers 10-11: {near_kl:.4f}; elsewhere: {far_kl:.4f}",
        f"- peak attention-ablation layer: {peak_layer}",
        f"- criterion (10-11 damage > 2x elsewhere): {verdict}",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nVerdict: {verdict} (near {near_kl:.4f} vs far {far_kl:.4f}, peak L{peak_layer})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
