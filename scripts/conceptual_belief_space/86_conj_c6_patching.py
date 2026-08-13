#!/usr/bin/env python3
"""Conjecture C6: how much of the state must be transplanted to transplant the belief?

Proposition 3.5 found that ADDING a rank-1 decoder direction does nothing, even at twice
the local residual RMS. C6 says the belief is writable, but only at higher rank or breadth.
This experiment replaces the additive-steering framing with a nested TRANSPLANT ladder: a
donor prefix whose forced answer is action B is used to overwrite progressively more of a
recipient prefix whose forced answer is action A, always at one layer, always over the
probe's own last-12 CoT window:

    rank1_commitment : replace the component along the commitment direction  (1 of 2880 dims)
    rank1_action     : replace the component along the donor's action direction        (1 dim)
    rank3_action     : replace the component in the 3-d LDA action subspace            (3 dims)
    full_window      : replace the entire window state                              (2880 dims)

Because every mode uses the SAME donor states and differs only in how many dimensions are
transplanted, the ladder isolates rank from every other factor. The prediction is a
monotone increase in transplant success, with full_window flipping the answer to B in most
pairs at some layer.
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
MODES = ("rank1_commitment", "rank1_action", "rank3_action", "full_window")

INK = "#333333"
GRID = "#e6e6e6"
MODE_COLORS = {"rank1_commitment": "#9aa0a6", "rank1_action": "#eb6834",
               "rank3_action": "#1baf7a", "full_window": "#2a78d6"}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--prefixes", type=Path, default=REPO_ROOT / "outputs/prefixes.parquet")
    parser.add_argument("--pooled_cache", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    parser.add_argument("--projection", type=Path, default=base / "activations/orthogonal_projection.npy")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c6_patching")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--n_pairs", type=int, default=24)
    parser.add_argument("--layers", type=str, default="3,10,15,18")
    parser.add_argument("--window", type=int, default=12)
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


def fit_bases(pooled: np.ndarray, table: pd.DataFrame, train_idx: np.ndarray, layers: list[int],
              projection: np.ndarray, seed: int) -> dict[int, dict[str, np.ndarray]]:
    """Per layer: orthonormal model-space bases for the commitment axis and action subspace."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    q_matrix = table[Q_COLS].to_numpy(dtype=np.float64)
    action_codes = q_matrix.argmax(axis=1)
    commitment = table["entropy_commitment"].to_numpy(dtype=np.float64)
    bases: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        x_train = np.asarray(pooled[train_idx, layer], dtype=np.float32)
        scaler = StandardScaler().fit(x_train)
        x_std = scaler.transform(x_train)

        def lift(direction_std: np.ndarray) -> np.ndarray:
            model_space = projection @ (direction_std / scaler.scale_)
            return (model_space / np.linalg.norm(model_space)).astype(np.float32)

        ridge = Ridge(alpha=1.0).fit(x_std, commitment[train_idx])
        logistic = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0,
                                      random_state=seed).fit(x_std, action_codes[train_idx])
        centered = logistic.coef_ - logistic.coef_.mean(axis=0, keepdims=True)
        lda = LinearDiscriminantAnalysis(n_components=3).fit(x_std, action_codes[train_idx])
        lifted = np.stack([projection @ (lda.scalings_[:, k] / scaler.scale_) for k in range(3)], axis=1)
        subspace, _ = np.linalg.qr(lifted)

        entry = {"commitment": lift(ridge.coef_.astype(np.float64)),
                 "subspace": subspace.astype(np.float32)}
        for index, action in enumerate(ACTIONS):
            entry[f"action_{action}"] = lift(centered[index].astype(np.float64))
        bases[layer] = entry
    return bases


def forward_capture(model: Any, mx: Any, create_attention_mask: Any, token_ids: list[int],
                    capture_layers: list[int], window_slice: slice) -> dict[int, Any]:
    transformer = model.model
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    captured: dict[int, Any] = {}
    for index, (layer, layer_type) in enumerate(zip(transformer.layers, transformer.layer_types,
                                                    strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, None)
        if index in capture_layers:
            captured[index] = mx.array(x[:, window_slice, :])
    mx.eval(*captured.values())
    return captured


def forward_patched(model: Any, mx: Any, create_attention_mask: Any, token_ids: list[int],
                    candidate_ids: list[int], *, patch_layer: int | None = None,
                    window_slice: slice | None = None, donor_state: Any = None,
                    basis: np.ndarray | None = None) -> np.ndarray:
    """basis=None with donor_state given -> full replacement; otherwise replace the
    component of the recipient inside `basis` with the donor's component."""
    transformer = model.model
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    for index, (layer, layer_type) in enumerate(zip(transformer.layers, transformer.layer_types,
                                                    strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, None)
        if patch_layer is not None and index == patch_layer:
            window = x[:, window_slice, :].astype(mx.float32)
            if basis is None:
                patched = donor_state.astype(mx.float32)
            else:
                basis_mx = mx.array(basis)
                coefficients_recipient = window @ basis_mx
                coefficients_donor = donor_state.astype(mx.float32) @ basis_mx
                patched = window + (coefficients_donor - coefficients_recipient) @ basis_mx.T
            x[:, window_slice, :] = patched.astype(x.dtype)
    logits = model.lm_head(transformer.norm(x))
    selected = logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(selected)
    return stable_action_probs(np.asarray(selected, dtype=np.float32))


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_pairs, args.layers = 2, "10"
    layers = [int(token) for token in args.layers.split(",") if token.strip()]

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == 0)
    strict = table[STRICT_COL].to_numpy(dtype=bool)
    valid = np.array([int(np.sum(mask)) for mask in table["_input_token_mask"]]) == args.window
    train_idx = split["train_idx"][strict[split["train_idx"]]]
    test_idx = split["test_idx"][strict[split["test_idx"]] & valid[split["test_idx"]]]

    projection = np.load(args.projection).astype(np.float64)
    pooled = np.load(args.pooled_cache, mmap_mode="r")
    cached_ids = np.load(args.pooled_cache.with_suffix(".prefix_ids.npy"))
    if not (cached_ids == table["prefix_id"].to_numpy()).all():
        raise SystemExit("pooled cache does not correspond to this table ordering")
    print(f"[c6] fitting bases on {len(train_idx)} strict train rows, layers {layers}", flush=True)
    bases = fit_bases(pooled, table, train_idx, layers, projection, args.seed)

    q_all = table[Q_COLS].to_numpy(dtype=np.float64)
    top_codes = q_all.argmax(axis=1)
    rng = np.random.default_rng(args.seed)
    pairs: list[tuple[int, int]] = []
    pool = list(test_idx)
    rng.shuffle(pool)
    for recipient in pool:
        if len(pairs) >= args.n_pairs:
            break
        donors = [d for d in pool if top_codes[d] != top_codes[recipient]
                  and q_all[d].max() > 0.5 and d != recipient]
        if donors:
            pairs.append((int(recipient), int(donors[rng.integers(len(donors))])))
    print(f"[c6] {len(pairs)} donor/recipient pairs", flush=True)

    texts = pd.read_parquet(args.prefixes, columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    text_of = dict(zip(texts.prefix_id, zip(texts.prompt_text, texts.analysis_prefix_text)))

    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    import mlx.core as mx

    print(f"[c6] loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_length = len(forced_suffix_token_ids(tokenizer))
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])

    def build_tokens(row_index: int) -> list[int]:
        prefix_id = int(table.iloc[row_index]["prefix_id"])
        prompt_text, analysis = text_of[prefix_id]
        text = (str(prompt_text).rstrip() + ANALYSIS_HEADER
                + ("" if analysis is None else str(analysis)) + FORCED_SUFFIX)
        return encode_without_special(tokenizer, text)

    records: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "patching.partial.parquet"
    for number, (recipient, donor) in enumerate(pairs, start=1):
        recipient_tokens = build_tokens(recipient)
        donor_tokens = build_tokens(donor)
        recipient_slice = slice(len(recipient_tokens) - suffix_length - args.window,
                                len(recipient_tokens) - suffix_length)
        donor_slice = slice(len(donor_tokens) - suffix_length - args.window,
                            len(donor_tokens) - suffix_length)
        donor_states = forward_capture(model, mx, create_attention_mask, donor_tokens,
                                       layers, donor_slice)
        base_q = forward_patched(model, mx, create_attention_mask, recipient_tokens, candidate_ids)
        target = int(top_codes[donor])
        base_top = int(base_q.argmax())
        common = {"pair": number, "recipient_prefix_id": int(table.iloc[recipient]["prefix_id"]),
                  "donor_prefix_id": int(table.iloc[donor]["prefix_id"]),
                  "target_action": ACTIONS[target], "base_top_action": ACTIONS[base_top],
                  "base_p_target": float(base_q[target])}
        records.append({**common, "layer": -1, "mode": "baseline",
                        **{c: float(base_q[i]) for i, c in enumerate(Q_COLS)},
                        "p_target": float(base_q[target]), "top_action": ACTIONS[base_top]})
        for layer in layers:
            for mode in MODES:
                if mode == "full_window":
                    basis = None
                elif mode == "rank3_action":
                    basis = bases[layer]["subspace"]
                elif mode == "rank1_action":
                    basis = bases[layer][f"action_{ACTIONS[target]}"][:, None]
                else:
                    basis = bases[layer]["commitment"][:, None]
                q_values = forward_patched(
                    model, mx, create_attention_mask, recipient_tokens, candidate_ids,
                    patch_layer=layer, window_slice=recipient_slice,
                    donor_state=donor_states[layer], basis=basis)
                records.append({**common, "layer": layer, "mode": mode,
                                **{c: float(q_values[i]) for i, c in enumerate(Q_COLS)},
                                "p_target": float(q_values[target]),
                                "top_action": ACTIONS[int(q_values.argmax())]})
        print(f"[c6] pair {number}/{len(pairs)} ({len(records)} records)", flush=True)
        pd.DataFrame(records).to_parquet(checkpoint, index=False)

    frame = pd.DataFrame(records)
    frame.to_parquet(args.out_dir / "patching.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    patched = frame[frame["mode"] != "baseline"].copy()
    patched["d_p_target"] = patched["p_target"] - patched["base_p_target"]
    patched["flipped_to_target"] = patched["top_action"] == patched["target_action"]
    patched["changed"] = patched["top_action"] != patched["base_top_action"]
    summary = patched.groupby(["mode", "layer"]).agg(
        mean_d_p_target=("d_p_target", "mean"),
        flip_to_target_rate=("flipped_to_target", "mean"),
        any_change_rate=("changed", "mean"),
        n=("pair", "size")).round(4).reset_index()
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for mode in MODES:
        subset = summary[summary["mode"] == mode].sort_values("layer")
        axes[0].plot(subset["layer"], subset["mean_d_p_target"], color=MODE_COLORS[mode],
                     linewidth=2, marker="o", markersize=5)
        axes[1].plot(subset["layer"], subset["flip_to_target_rate"], color=MODE_COLORS[mode],
                     linewidth=2, marker="o", markersize=5)
    axes[0].set_ylabel("mean change in p(donor action)", color=INK)
    axes[0].set_title("Belief movement toward the donor", color=INK, fontsize=11)
    axes[1].set_ylabel("fraction of pairs answering the donor action", color=INK)
    axes[1].set_title("Successful transplants", color=INK, fontsize=11)
    axes[0].legend(handles=[plt.Line2D([], [], color=MODE_COLORS[m], linewidth=2,
                                       label=m.replace("_", " "))
                            for m in MODES], frameon=False, fontsize=8, labelcolor=INK)
    for axis in axes:
        axis.set_xlabel("patched layer", color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C6: how many dimensions must be transplanted to move the belief?", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c6_patching.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    best_full = summary[summary["mode"] == "full_window"]["flip_to_target_rate"].max()
    best_rank3 = summary[summary["mode"] == "rank3_action"]["flip_to_target_rate"].max()
    best_rank1 = summary[summary["mode"] == "rank1_action"]["flip_to_target_rate"].max()
    monotone = bool(best_full >= best_rank3 >= best_rank1)
    verdict = "SUPPORTED" if best_full > 0.5 else "NOT SUPPORTED"
    lines = [
        "# Conjecture C6: rank ladder of belief transplantation",
        "",
        f"- {len(pairs)} donor/recipient pairs with different forced answers; layers {layers};",
        f"  window {args.window} tokens; cache-free forwards; nested modes over identical donors.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"- best flip-to-donor rate: full_window {best_full:.4f}, rank3_action {best_rank3:.4f}, "
        f"rank1_action {best_rank1:.4f}",
        f"- monotone in rank: {monotone}",
        f"- criterion (full-window swap flips > 50% of pairs at some layer): {verdict}",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nVerdict: {verdict} (full {best_full:.4f} / rank3 {best_rank3:.4f} / "
          f"rank1 {best_rank1:.4f}, monotone={monotone})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
