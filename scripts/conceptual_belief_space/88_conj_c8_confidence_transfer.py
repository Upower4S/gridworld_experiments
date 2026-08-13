#!/usr/bin/env python3
"""Conjecture C8: is the commitment direction a task-general confidence axis?

Theorem 3 established that commitment occupies a direction essentially orthogonal to action
content. If that direction encodes "how sure am I" rather than "how sure am I about moving
LEFT in a maze", it should transfer to a task with the same answer interface but no spatial
content at all.

Design: arithmetic/logic questions whose four options are LABELLED with the same four
tokens (UP/DOWN/LEFT/RIGHT) and answered through the identical JSON forced-answer format.
The output interface is therefore held exactly fixed while the task content is replaced.
The gridworld commitment direction is frozen (fitted on gridworld fold-0 train rows, with
its standardization) and applied to the new task's window states.

Three references: the frozen-direction transfer, a position-only bar on the same rows, and
a task-refit ceiling (question-grouped CV).
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

PROMPT_TEMPLATE = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-01-22

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|># Instructions

You will be given a question together with four labelled options. Exactly one option is correct. Work the problem out, then report the LABEL of the correct option.

The four labels are UP, DOWN, LEFT and RIGHT. They are arbitrary names for the options and have no other meaning.

Your final answer should be a valid JSON object provided in the following form:

```
{{
  "action": "<UP|DOWN|LEFT|RIGHT>",
}}
```

DO NOT INCLUDE ANY `json` or `jsonb` in your response NOR TICK MARKS LIKE THIS: ```json ```. Start with {{ and end with }} exactly.

# Inputs

Question: {question}

Options:
- UP: {option_up}
- DOWN: {option_down}
- LEFT: {option_left}
- RIGHT: {option_right}<|end|><|start|>assistant"""

QUESTIONS: list[tuple[str, list[str], int]] = [
    ("What is 17 * 23?", ["391", "371", "381", "401"], 0),
    ("A train leaves at 09:47 and arrives at 14:12. How long is the journey?",
     ["4 h 35 min", "4 h 25 min", "5 h 25 min", "3 h 25 min"], 1),
    ("What is the remainder when 2^10 is divided by 7?", ["1", "4", "2", "6"], 3),
    ("If 3x + 7 = 31, what is x?", ["6", "9", "8", "12"], 2),
    ("A shirt costs 40 after a 20% discount. What was the original price?",
     ["48", "50", "52", "45"], 1),
    ("How many distinct ways can 4 people be seated in a row?", ["16", "12", "24", "20"], 2),
    ("What is the sum of the first 20 positive integers?", ["190", "200", "210", "220"], 2),
    ("Alice is twice as old as Bob was 5 years ago. Bob is 15 now. How old is Alice?",
     ["20", "25", "30", "15"], 0),
    ("What is 0.125 expressed as a fraction in lowest terms?",
     ["1/6", "1/4", "1/8", "1/12"], 2),
    ("A bag has 3 red and 5 blue balls. What is the probability a single draw is red?",
     ["3/8", "5/8", "3/5", "1/3"], 0),
    ("If every A is a B, and some B are C, which must be true?",
     ["Some A are C", "All B are A", "Nothing further follows", "No A is C"], 2),
    ("What is the next number: 2, 6, 12, 20, 30, ...?", ["36", "42", "40", "44"], 1),
    ("A rectangle has perimeter 36 and width 7. What is its area?",
     ["77", "91", "84", "63"], 0),
    ("What is 15% of 240?", ["30", "42", "36", "24"], 2),
    ("How many prime numbers are there strictly between 10 and 30?",
     ["5", "6", "7", "8"], 1),
    ("Water freezes at 32 F. What is 50 F in Celsius?", ["12", "8", "18", "10"], 3),
]

INK = "#333333"
GRID = "#e6e6e6"


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--pooled_cache", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    parser.add_argument("--projection", type=Path, default=base / "activations/orthogonal_projection.npy")
    parser.add_argument("--model_name_or_path", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c8_transfer")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--layers", type=str, default="3,10,13,15,18,23")
    parser.add_argument("--gen_tokens", type=int, default=420)
    parser.add_argument("--cuts", type=int, default=20)
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


def fit_frozen_directions(pooled: np.ndarray, table: pd.DataFrame, train_idx: np.ndarray,
                          layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    commitment = table["entropy_commitment"].to_numpy(dtype=np.float64)
    frozen: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        x_train = np.asarray(pooled[train_idx, layer], dtype=np.float32)
        scaler = StandardScaler().fit(x_train)
        ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), commitment[train_idx])
        frozen[layer] = {"mean": scaler.mean_.astype(np.float32),
                         "scale": scaler.scale_.astype(np.float32),
                         "coef": ridge.coef_.astype(np.float64)}
    return frozen


def greedy_generate(model: Any, mx: Any, prompt_ids: list[int], max_tokens: int,
                    stop_ids: set[int]) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    generated: list[int] = []
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        if token in stop_ids:
            break
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
    return generated


def forward_capture_and_answer(model: Any, mx: Any, create_attention_mask: Any,
                               token_ids: list[int], candidate_ids: list[int],
                               window_slice: slice, projection_mx: Any,
                               layers: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    transformer = model.model
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, None)
    sliding_mask = create_attention_mask(x, None, window_size=transformer.window_size)
    captured: dict[int, Any] = {}
    for index, (layer, layer_type) in enumerate(zip(transformer.layers, transformer.layer_types,
                                                    strict=True)):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, None)
        if index in layers:
            captured[index] = (x[0, window_slice, :].astype(mx.float32) @ projection_mx)
    logits = model.lm_head(transformer.norm(x))
    selected = logits[0, -1, candidate_ids].astype(mx.float32)
    mx.eval(selected, *captured.values())
    states = {layer: np.asarray(value, dtype=np.float32).mean(axis=0)
              for layer, value in captured.items()}
    return stable_action_probs(np.asarray(selected, dtype=np.float32)), states


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.cuts, args.gen_tokens = 4, 120
    layers = [int(token) for token in args.layers.split(",") if token.strip()]
    questions = QUESTIONS[:3] if args.smoke else QUESTIONS

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == 0)
    strict = table[STRICT_COL].to_numpy(dtype=bool)
    train_idx = split["train_idx"][strict[split["train_idx"]]]
    pooled = np.load(args.pooled_cache, mmap_mode="r")
    print(f"[c8] freezing commitment directions on {len(train_idx)} gridworld strict train rows",
          flush=True)
    frozen = fit_frozen_directions(pooled, table, train_idx, layers)

    projection = np.load(args.projection).astype(np.float32)

    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    import mlx.core as mx

    print(f"[c8] loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    suffix_ids = forced_suffix_token_ids(tokenizer)
    candidate_ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        assert len(encoded) == 1
        candidate_ids.append(encoded[0])
    projection_mx = mx.array(projection)
    # FORCED_SUFFIX begins with <|end|>, so its first id is the analysis-channel terminator.
    stop_ids = {int(suffix_ids[0])}

    rows: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "transfer.partial.parquet"
    for question_index, (question, options, correct) in enumerate(questions):
        prompt = PROMPT_TEMPLATE.format(
            question=question, option_up=options[0], option_down=options[1],
            option_left=options[2], option_right=options[3])
        prompt_ids = encode_without_special(tokenizer, prompt + ANALYSIS_HEADER)
        cot_ids = greedy_generate(model, mx, prompt_ids, args.gen_tokens, stop_ids)
        if len(cot_ids) < args.window + 2:
            print(f"[c8] question {question_index}: CoT too short ({len(cot_ids)}), skipping")
            continue
        cut_points = np.unique(np.linspace(args.window, len(cot_ids), args.cuts).astype(int))
        for cut in cut_points:
            token_ids = prompt_ids + cot_ids[:int(cut)] + suffix_ids
            window_slice = slice(len(token_ids) - len(suffix_ids) - args.window,
                                 len(token_ids) - len(suffix_ids))
            q_values, states = forward_capture_and_answer(
                model, mx, create_attention_mask, token_ids, candidate_ids,
                window_slice, projection_mx, layers)
            entropy_commitment = float(
                1.0 + (q_values * np.log(np.clip(q_values, 1e-12, None))).sum() / np.log(len(ACTIONS)))
            entry = {
                "question_index": question_index, "cut": int(cut), "n_cot_tokens": len(cot_ids),
                "progress": float(cut) / len(cot_ids), "prefix_token_index": int(cut),
                **{c: float(q_values[i]) for i, c in enumerate(Q_COLS)},
                "entropy_commitment": entropy_commitment,
                "top_action": ACTIONS[int(q_values.argmax())],
                "correct_action": ACTIONS[correct],
            }
            for layer in layers:
                entry[f"state_{layer}"] = states[layer]
            rows.append(entry)
        print(f"[c8] question {question_index + 1}/{len(questions)}: {len(cot_ids)} CoT tokens, "
              f"{len(cut_points)} cuts", flush=True)
        pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("state_")}
                      for row in rows]).to_parquet(checkpoint, index=False)

    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("state_")} for row in rows])
    frame.to_parquet(args.out_dir / "transfer_rows.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    target = frame["entropy_commitment"].to_numpy(dtype=float)
    groups = frame["question_index"].to_numpy()
    position_bar = float(spearmanr(np.log1p(frame["prefix_token_index"].to_numpy(dtype=float)),
                                   target).statistic)
    result_rows = []
    for layer in layers:
        states = np.stack([row[f"state_{layer}"] for row in rows])
        standardized = (states - frozen[layer]["mean"]) / frozen[layer]["scale"]
        transfer_score = standardized @ frozen[layer]["coef"]
        transfer_rho = float(spearmanr(transfer_score, target).statistic)

        oof = np.zeros(len(states))
        n_splits = min(4, len(np.unique(groups)))
        for train_index, test_index in GroupKFold(n_splits=n_splits).split(states, target, groups):
            model_refit = Ridge(alpha=1.0).fit(standardized[train_index], target[train_index])
            oof[test_index] = model_refit.predict(standardized[test_index])
        ceiling_rho = float(spearmanr(oof, target).statistic)
        result_rows.append({"layer": layer, "frozen_transfer_spearman": transfer_rho,
                            "task_refit_ceiling_spearman": ceiling_rho,
                            "position_bar_spearman": position_bar})
        print(f"[c8] layer {layer}: transfer rho={transfer_rho:+.4f} "
              f"ceiling={ceiling_rho:+.4f} bar={position_bar:+.4f}")
    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "transfer.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    axis.plot(results["layer"], results["frozen_transfer_spearman"], color="#2a78d6",
              linewidth=2, marker="o", markersize=5)
    axis.plot(results["layer"], results["task_refit_ceiling_spearman"], color="#1baf7a",
              linewidth=2, marker="o", markersize=5, linestyle="--")
    axis.axhline(abs(position_bar), color=INK, linewidth=1.2, linestyle=":")
    axis.annotate("position-only bar", (results["layer"].iloc[0], abs(position_bar)),
                  textcoords="offset points", xytext=(0, 5), color=INK, fontsize=8)
    axis.annotate("frozen gridworld direction", (results["layer"].iloc[-1],
                  results["frozen_transfer_spearman"].iloc[-1]), textcoords="offset points",
                  xytext=(-6, 8), ha="right", color=INK, fontsize=8)
    axis.annotate("refit on the new task", (results["layer"].iloc[-1],
                  results["task_refit_ceiling_spearman"].iloc[-1]), textcoords="offset points",
                  xytext=(-6, -12), ha="right", color=INK, fontsize=8)
    axis.set_xlabel("layer (block output)", color=INK)
    axis.set_ylabel("Spearman with new-task commitment", color=INK)
    axis.set_title("C8: does the gridworld confidence axis transfer to arithmetic?",
                   color=INK, fontsize=11)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure_path = args.figure_dir / "c8_confidence_transfer.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    best = results.loc[results["frozen_transfer_spearman"].abs().idxmax()]
    verdict = ("SUPPORTED" if abs(best["frozen_transfer_spearman"]) > abs(position_bar)
               and abs(best["frozen_transfer_spearman"]) > 0.2 else "NOT SUPPORTED")
    accuracy = float((frame.groupby("question_index").last()["top_action"]
                      == frame.groupby("question_index").last()["correct_action"]).mean())
    lines = [
        "# Conjecture C8: does the commitment direction transfer to a non-spatial task?",
        "",
        f"- {frame.question_index.nunique()} four-option questions, {len(frame)} prefixes;",
        f"  identical answer interface (UP/DOWN/LEFT/RIGHT labels, same JSON forced suffix),",
        f"  no spatial content. Final-cut answer accuracy on the new task: {accuracy:.3f}.",
        f"- Gridworld commitment directions frozen from fold-0 strict train rows.",
        "",
        results.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"- best frozen transfer: layer {int(best['layer'])}, "
        f"Spearman {best['frozen_transfer_spearman']:+.4f}",
        f"- task-refit ceiling at that layer: {best['task_refit_ceiling_spearman']:+.4f}",
        f"- position-only bar: {position_bar:+.4f}",
        f"- **{verdict}**",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.round(4).to_string(index=False))
    print(f"\nVerdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
