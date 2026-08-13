#!/usr/bin/env python3
"""Layer-targeted directional steering of the forced-answer belief (D6).

Causal companion to the D2 orthogonality result. Decoder-defined directions (the ridge
commitment direction and the balanced-logistic action directions, fitted on fold-0 TRAIN
strict rows in the 512-d projected space) are lifted to the 2880-d residual stream through
the stored orthogonal projection and added to the last W=12 CoT token states at one chosen
layer during a full, cache-free forward. Effects are read from the forced-answer q at the
end of the suffix. A norm-matched random direction is the control.

Predictions if the D2 orthogonality claim is causal: commitment-direction steering moves
entropy commitment with little argmax flipping; action-direction steering flips the argmax
toward the target with little systematic commitment change; random directions do neither.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layer_steering")
    parser.add_argument("--n_prefixes", type=int, default=64)
    parser.add_argument("--layers", type=str, default="3,10,15,18")
    parser.add_argument("--alphas", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--scale_mode", choices=["rms", "token_norm"], default="rms",
                        help="token_norm: ||delta|| = alpha * ||h_token|| (the honest scale).")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--steer_scope", choices=["window", "all_cot"], default="window",
                        help="Steer only the probe's last-W CoT window, or every CoT token.")
    parser.add_argument("--commitment_band", type=str, default="0.2,0.9",
                        help="Baseline commitment_to_final band for sampled prefixes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="8 prefixes, layers 3,15, alpha 0.5 only.")
    parser.add_argument("--tag", type=str, default="run")
    return parser.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_directions(pooled: np.ndarray, table: pd.DataFrame, train_idx: np.ndarray,
                   layers: list[int], projection: np.ndarray, seed: int) -> dict[int, dict[str, np.ndarray]]:
    """Decoder directions per layer, lifted to model space and unit-normalized."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    q_matrix = table[Q_COLS].to_numpy(dtype=np.float64)
    action_codes = q_matrix.argmax(axis=1)
    commitment = table["entropy_commitment"].to_numpy(dtype=np.float64)
    directions: dict[int, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(seed)
    for layer in layers:
        x_train = np.asarray(pooled[train_idx, layer], dtype=np.float32)
        scaler = StandardScaler().fit(x_train)
        x_std = scaler.transform(x_train)
        ridge = Ridge(alpha=1.0).fit(x_std, commitment[train_idx])
        logistic = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0,
                                      random_state=seed).fit(x_std, action_codes[train_idx])

        def lift(direction_std: np.ndarray) -> np.ndarray:
            raw = direction_std / scaler.scale_
            model_space = projection @ raw
            return (model_space / np.linalg.norm(model_space)).astype(np.float32)

        centered = logistic.coef_ - logistic.coef_.mean(axis=0, keepdims=True)
        entry = {"commitment": lift(ridge.coef_.astype(np.float64))}
        for index, action in enumerate(ACTIONS):
            entry[f"action_{action}"] = lift(centered[index].astype(np.float64))
        random_direction = rng.standard_normal(projection.shape[0])
        entry["random"] = (random_direction / np.linalg.norm(random_direction)).astype(np.float32)
        directions[layer] = entry
    return directions


def steered_forward(model: Any, mx: Any, create_attention_mask: Any, token_ids: list[int],
                    candidate_ids: list[int], *, steer_layer: int | None = None,
                    window_slice: slice | None = None, direction: np.ndarray | None = None,
                    alpha: float = 0.0, scale_mode: str = "rms") -> np.ndarray:
    """scale_mode='token_norm' makes alpha the perturbation size RELATIVE TO THE TOKEN
    STATE: ||delta|| = alpha * ||h_token||. The original 'rms' mode multiplies a unit-L2
    direction by the PER-ELEMENT rms, which in d=2880 dimensions is smaller by a factor
    sqrt(d) ~= 53.7 -- so 'rms' alpha=2 perturbs the state by only ~3.7%, far too little
    to support a claim about writability."""
    transformer = model.model
    layers = list(transformer.layers)
    layer_types = list(transformer.layer_types)
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    delta_direction = None
    if steer_layer is not None:
        delta_direction = mx.array(direction[None, None, :])
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types, strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, None)
        if steer_layer is not None and index == steer_layer:
            window = x[:, window_slice, :].astype(mx.float32)
            if scale_mode == "token_norm":
                scale = mx.sqrt(mx.sum(mx.square(window), axis=-1, keepdims=True))
            else:
                scale = mx.sqrt(mx.mean(mx.square(window)))
            steered = window + float(alpha) * scale * delta_direction
            x[:, window_slice, :] = steered.astype(x.dtype)
    logits = model.lm_head(transformer.norm(x))
    selected = logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(selected)
    return stable_action_probs(np.asarray(selected, dtype=np.float32))


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_prefixes, args.layers, args.alphas, args.tag = 8, "3,15", "0.5", "smoke"
    layers = [int(token) for token in args.layers.split(",") if token.strip()]
    alphas = [float(token) for token in args.alphas.split(",") if token.strip()]
    low_band, high_band = (float(v) for v in args.commitment_band.split(","))

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == 0)
    strict_mask = table[STRICT_COL].to_numpy(dtype=bool)
    train_idx = split["train_idx"][strict_mask[split["train_idx"]]]
    test_idx = split["test_idx"][strict_mask[split["test_idx"]]]

    projection = np.load(args.projection).astype(np.float64)  # [hidden, projected]
    pooled = np.load(args.pooled_cache, mmap_mode="r")
    print(f"[steer] fitting decoder directions on {len(train_idx)} strict train rows "
          f"for layers {layers}", flush=True)
    directions = fit_directions(pooled, table, train_idx, layers, projection, args.seed)

    commitment_baseline = table["commitment_to_final"].to_numpy(dtype=np.float64) \
        if "commitment_to_final" in table.columns else None
    if commitment_baseline is None:
        q_matrix = table[Q_COLS].to_numpy(dtype=np.float64)
        agent_actions = table["agent_action"].astype(str).to_numpy()
        index_of = {action: i for i, action in enumerate(ACTIONS)}
        commitment_baseline = np.array([q_matrix[i, index_of[a]] for i, a in enumerate(agent_actions)])
    band = (commitment_baseline[test_idx] >= low_band) & (commitment_baseline[test_idx] <= high_band)
    candidates = test_idx[band]
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(candidates, size=min(args.n_prefixes, len(candidates)), replace=False))
    print(f"[steer] {len(chosen)} prefixes sampled from {len(candidates)} in-band strict test rows",
          flush=True)

    texts = pd.read_parquet(args.prefixes,
                            columns=["prefix_id", "prompt_text", "analysis_prefix_text"])
    rows = table.iloc[chosen][["prefix_id", "_input_token_mask", "entropy_commitment"]].merge(
        texts, on="prefix_id", validate="one_to_one")

    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    import mlx.core as mx

    print(f"[steer] loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_length = len(forced_suffix_token_ids(tokenizer))
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])

    records: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / f"steering_{args.tag}.partial.parquet"
    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        text = (str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER
                + ("" if row["analysis_prefix_text"] is None else str(row["analysis_prefix_text"]))
                + FORCED_SUFFIX)
        token_ids = encode_without_special(tokenizer, text)
        n_valid = int(np.asarray(row["_input_token_mask"], dtype=np.float32).sum())
        if args.steer_scope == "all_cot":
            prompt_ids = encode_without_special(tokenizer, str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER)
            window = len(token_ids) - suffix_length - len(prompt_ids)
        else:
            window = min(args.window, max(n_valid, 1))
        window_slice = slice(len(token_ids) - suffix_length - window, len(token_ids) - suffix_length)

        base_q = steered_forward(model, mx, create_attention_mask, token_ids, candidate_ids)
        base_entropy_commitment = 1.0 + (base_q * np.log(np.clip(base_q, 1e-12, None))).sum() / np.log(4)
        base_top = int(base_q.argmax())
        runner_up = base_q.copy()
        runner_up[base_top] = -np.inf
        target = int(runner_up.argmax())
        records.append({"prefix_id": int(row.prefix_id), "layer": -1, "direction": "baseline",
                        "alpha": 0.0, **{c: float(base_q[i]) for i, c in enumerate(Q_COLS)},
                        "entropy_commitment_q": float(base_entropy_commitment),
                        "top_action": ACTIONS[base_top], "target_action": ACTIONS[target],
                        "n_tokens": len(token_ids), "window": window})

        for layer in layers:
            runs = [("commitment", +1.0), ("commitment", -1.0),
                    (f"action_{ACTIONS[target]}", +1.0),
                    ("random", +1.0), ("random", -1.0)]
            for direction_name, sign in runs:
                for alpha in alphas:
                    q_values = steered_forward(
                        model, mx, create_attention_mask, token_ids, candidate_ids,
                        steer_layer=layer, window_slice=window_slice,
                        direction=directions[layer][direction_name], alpha=sign * alpha,
                        scale_mode=args.scale_mode)
                    entropy_commitment_q = 1.0 + (q_values * np.log(np.clip(q_values, 1e-12, None))
                                                  ).sum() / np.log(4)
                    records.append({
                        "prefix_id": int(row.prefix_id), "layer": layer,
                        "direction": direction_name, "alpha": float(sign * alpha),
                        **{c: float(q_values[i]) for i, c in enumerate(Q_COLS)},
                        "entropy_commitment_q": float(entropy_commitment_q),
                        "top_action": ACTIONS[int(q_values.argmax())],
                        "target_action": ACTIONS[target],
                        "n_tokens": len(token_ids), "window": window,
                    })
        print(f"[steer] prefix {row_number}/{len(rows)} done ({len(records)} records)", flush=True)
        pd.DataFrame(records).to_parquet(checkpoint, index=False)

    frame = pd.DataFrame(records)
    output = args.out_dir / f"steering_{args.tag}.parquet"
    frame.to_parquet(output, index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    base = frame[frame.direction == "baseline"][
        ["prefix_id", "entropy_commitment_q", "top_action"]].rename(
        columns={"entropy_commitment_q": "base_commitment", "top_action": "base_top"})
    steered = frame[frame.direction != "baseline"].merge(base, on="prefix_id")
    steered["d_commitment"] = steered["entropy_commitment_q"] - steered["base_commitment"]
    steered["flipped"] = steered["top_action"] != steered["base_top"]
    steered["flipped_to_target"] = steered["top_action"] == steered["target_action"]
    steered["direction_kind"] = np.where(steered.direction.str.startswith("action_"), "action",
                                         steered.direction)
    summary = steered.groupby(["layer", "direction_kind", "alpha"]).agg(
        mean_d_commitment=("d_commitment", "mean"),
        flip_rate=("flipped", "mean"),
        flip_to_target_rate=("flipped_to_target", "mean"),
        n=("prefix_id", "size")).round(4).reset_index()
    summary.to_csv(args.out_dir / f"steering_summary_{args.tag}.csv", index=False)

    report = [
        f"# Layer-Targeted Steering ({args.tag})",
        "",
        f"- {len(rows)} strict fold-0 test prefixes (baseline commitment in "
        f"[{low_band}, {high_band}]); window last {args.window} CoT tokens; layers {layers}; "
        f"alphas {alphas} in units of window residual RMS; cache-free forwards.",
        "",
        summary.to_markdown(index=False),
    ]
    (args.out_dir / f"steering_report_{args.tag}.md").write_text("\n".join(report) + "\n",
                                                                 encoding="utf-8")
    print(f"Wrote {output} and steering_summary_{args.tag}.csv")
    print(json.dumps(summary.to_dict(orient="records")[:12], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
