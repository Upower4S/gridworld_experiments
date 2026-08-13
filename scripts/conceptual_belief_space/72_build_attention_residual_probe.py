#!/usr/bin/env python3
"""Build the final last-token attention residual probe.

The strict nested linear stacker is the residual stream.  The fold-frozen
last-token attention gate contributes only its query-dependent correction
(``dynamic - global``).  A single correction strength is selected on declared
development split 0, then applied unchanged to confirmatory splits 1-4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NESTED_SCRIPT = Path(__file__).with_name("68_train_nested_last_token_attention_probe.py")
spec = importlib.util.spec_from_file_location("attention_residual_nested_impl", NESTED_SCRIPT)
nested = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nested
assert spec and spec.loader
spec.loader.exec_module(nested)

base = nested.base
fixed = nested.fixed
ACTIONS = nested.ACTIONS
SOURCE_NAMES = nested.SOURCE_NAMES


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def rowz(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (centered / scale).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stacker_predictions",
        default="outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3_with_predictions/nested_oof_source_stacker_predictions.parquet",
    )
    parser.add_argument(
        "--attention_predictions",
        default="outputs/conceptual_belief_space/last_token_attention_probe_strong_strict_sources/strict_source_attention_predictions.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space/last_token_attention_residual_probe_final",
    )
    parser.add_argument("--alphas", default="-2,-1,-0.5,-0.25,0,0.1,0.25,0.5,1,2,4")
    parser.add_argument("--development_split", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker = pd.read_parquet(args.stacker_predictions).sort_values("row_id").reset_index(drop=True)
    attention = pd.read_parquet(args.attention_predictions).sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(stacker["row_id"].to_numpy(), attention["row_id"].to_numpy()):
        raise RuntimeError("Stacker and attention rows do not align")
    if not np.array_equal(stacker["split_id"].to_numpy(), attention["split_id"].to_numpy()):
        raise RuntimeError("Stacker and attention fold IDs do not align")
    if stacker["in_sample"].astype(bool).any() or not stacker["oof_score"].astype(bool).all():
        raise RuntimeError("Residual stacker rows must all be strict held-out predictions")

    y = base.action_indices(stacker["top_action"])
    split_ids = stacker["split_id"].to_numpy(dtype=np.int64)
    stacker_scores = rowz(
        stacker[[f"stacker_score_{action}" for action in ACTIONS]].to_numpy(dtype=np.float32)
    )
    dynamic = attention[
        [f"attention_probe_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    global_scores = attention[
        [f"global_control_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    shuffled = attention[
        [f"shuffled_query_control_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    query_correction = dynamic - global_scores
    shuffled_correction = shuffled - global_scores
    development = split_ids == args.development_split

    candidate_rows = []
    for alpha in parse_floats(args.alphas):
        scores = stacker_scores + float(alpha) * query_correction
        pred = scores.argmax(axis=1)
        candidate_rows.append(
            {
                "alpha": float(alpha),
                "development_balanced_accuracy": fixed.balanced_accuracy(
                    y[development], pred[development]
                ),
                "development_accuracy": float((pred[development] == y[development]).mean()),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["development_balanced_accuracy", "development_accuracy"], ascending=False
    )
    selected_alpha = float(candidates.iloc[0]["alpha"])
    candidates["selected"] = candidates["alpha"].eq(selected_alpha)

    final_dynamic = stacker_scores + selected_alpha * query_correction
    final_shuffled = stacker_scores + selected_alpha * shuffled_correction
    source_evidence = np.stack(
        [
            attention[
                [f"source_{source_name}_score_{action}" for action in ACTIONS]
            ].to_numpy(dtype=np.float32)
            for source_name in SOURCE_NAMES
        ],
        axis=1,
    )
    fixed_ovr3 = attention[
        [f"fixed_ovr3_control_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    fixed_core4 = attention[
        [f"fixed_core4_control_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    uniform = attention[
        [f"uniform_control_score_{action}" for action in ACTIONS]
    ].to_numpy(dtype=np.float32)
    attention_weights = np.stack(
        [
            attention[
                [f"attention_{source_name}_{action}" for action in ACTIONS]
            ].to_numpy(dtype=np.float32)
            for source_name in SOURCE_NAMES
        ],
        axis=1,
    )

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    attention_rows: list[dict] = []
    for split_id in sorted(np.unique(split_ids)):
        mask = split_ids == split_id
        local_df = stacker.loc[mask].reset_index(drop=True)
        local_indices = np.arange(len(local_df), dtype=np.int64)
        outputs = {
            "dynamic": final_dynamic[mask],
            "global": stacker_scores[mask],
            "uniform": uniform[mask],
            "fixed_ovr3": fixed_ovr3[mask],
            "fixed_core4": fixed_core4[mask],
            "shuffled_query": final_shuffled[mask],
            "source_evidence": source_evidence[mask],
            "attention": attention_weights[mask],
        }
        info = {
            "selected_alpha": selected_alpha,
            "selection_split": int(args.development_split),
            "query_mode": "last1_scores",
            "probe_form": "strict_linear_stacker_plus_last_token_attention_residual",
            "total_probe_parameters": 927_655,
        }
        base.add_all_metrics(metric_rows, local_df, local_indices, outputs, split_id, info)
        prediction_frames.append(base.make_prediction_frame(local_df, local_indices, outputs, split_id))
        attention_rows.extend(base.attention_statistics(outputs, split_id))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = pd.DataFrame(attention_rows)
    candidates.to_csv(output_dir / "attention_residual_alpha_candidates.csv", index=False)
    metrics.to_csv(output_dir / "attention_residual_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "attention_residual_predictions.parquet", index=False)
    weights.to_csv(output_dir / "attention_residual_weights.csv", index=False)
    all_prefixes = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = (
        all_prefixes.groupby("source_name", as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    confirmatory = all_prefixes[all_prefixes["split_id"].ne(args.development_split)]
    confirmatory_summary = (
        confirmatory.groupby("source_name", as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    summary.to_csv(output_dir / "attention_residual_summary.csv", index=False)
    confirmatory_summary.to_csv(
        output_dir / "attention_residual_confirmatory_summary.csv", index=False
    )
    manifest = {
        "probe_definition": "strict activation-only linear residual stream plus query-dependent last-token attention correction",
        "frozen_backbone": True,
        "backbone_name": "openai/gpt-oss-20b",
        "target": "forced-answer q-top action",
        "selected_alpha": selected_alpha,
        "selection_policy": "alpha selected only on declared development split 0",
        "confirmatory_splits": [1, 2, 3, 4],
        "stacker_predictions": args.stacker_predictions,
        "attention_predictions": args.attention_predictions,
        "query": "held-out four-action evidence decoded from the actual last generated prefix token",
        "attention_residual": "alpha * (dynamic_query_gate - no_query_global_gate)",
        "excluded_inputs": base.FORBIDDEN_MODEL_INPUTS,
        "no_future_or_forced_answer_tokens": True,
        "all_component_scores_outer_oof": True,
        "parameter_count": 927_655,
    }
    (output_dir / "attention_residual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = [
        "# Final Last-Token Attention Residual Probe",
        "",
        "## Contract",
        "",
        "- Frozen 20B backbone; fewer than one million readout parameters.",
        "- The actual last prefix token is decoded into the attention query.",
        "- All source scores and gate-training scores are trajectory-grouped OOF.",
        "- Grid/task state, metadata, future tokens, and forced-answer tokens are excluded.",
        "- Alpha is selected on development split 0; splits 1-4 remain confirmatory.",
        "",
        "## Selected development coefficient",
        "",
        candidates.head(1).to_markdown(index=False),
        "",
        "## Five-fold summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Confirmatory folds 1-4",
        "",
        confirmatory_summary.to_markdown(index=False),
    ]
    (output_dir / "attention_residual_report.md").write_text("\n".join(report) + "\n")
    print(candidates.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(confirmatory_summary.to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
