#!/usr/bin/env python3
"""Repeated within-fold permutation audit for the final attention query."""

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

EVAL_SCRIPT = Path(__file__).with_name("71_evaluate_attention_gate_on_strict_sources.py")
spec = importlib.util.spec_from_file_location("query_permutation_eval_impl", EVAL_SCRIPT)
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
assert spec and spec.loader
spec.loader.exec_module(ev)

nested = ev.nested
base = ev.base
fixed = ev.fixed
ACTIONS = ev.ACTIONS
SOURCE_NAMES = ev.SOURCE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_predictions",
        default="outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3_with_predictions/nested_oof_source_stacker_predictions.parquet",
    )
    parser.add_argument(
        "--last1_predictions",
        default="outputs/conceptual_belief_space/activation_window_top_action_linear_full_l10_18_raw/linear_top_action_predictions.parquet",
    )
    parser.add_argument(
        "--development_gate",
        default="outputs/conceptual_belief_space/cached_last_token_attention_gate_screen_split0/selected_cached_last_token_attention_gate.pt",
    )
    parser.add_argument(
        "--confirmatory_gate_dir",
        default="outputs/conceptual_belief_space/nested_last_token_attention_probe_confirmatory_splits1_4",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space/last_token_attention_residual_probe_final",
    )
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    return parser.parse_args()


def rowz(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (centered / scale).astype(np.float32)


def main() -> int:
    import torch

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_df = pd.read_parquet(args.source_predictions)
    last1_df = pd.read_parquet(args.last1_predictions)
    aligned, evidence = ev.build_evidence(source_df, last1_df)
    aligned = aligned.sort_values("row_id").reset_index(drop=True)
    y_all = base.action_indices(aligned["top_action"])
    stacker_all = rowz(
        aligned[[f"stacker_score_{action}" for action in ACTIONS]].to_numpy(dtype=np.float32)
    )
    split_ids = aligned["split_id"].to_numpy(dtype=np.int64)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed + 730_000)
    observed_by_fold: dict[int, float] = {}
    shuffled_by_fold: dict[int, list[float]] = {}

    for split_id in sorted(np.unique(split_ids)):
        mask = split_ids == split_id
        local_evidence = evidence[mask]
        local_stacker = stacker_all[mask]
        local_y = y_all[mask]
        gate_args = argparse.Namespace(
            development_gate=args.development_gate,
            confirmatory_gate_dir=args.confirmatory_gate_dir,
        )
        gate = ev.load_gate_state(int(split_id), gate_args, device)
        tensor = torch.as_tensor(local_evidence, dtype=torch.float32, device=device)
        query = tensor[:, SOURCE_NAMES.index("last1_ovr"), :]
        with torch.no_grad():
            dynamic, _ = gate(tensor, query, mode="dynamic")
            global_scores, _ = gate(tensor, query, mode="global")
        dynamic_np = dynamic.cpu().numpy()
        global_np = global_scores.cpu().numpy()
        observed_scores = local_stacker + float(args.alpha) * (dynamic_np - global_np)
        observed_by_fold[int(split_id)] = fixed.balanced_accuracy(
            local_y, observed_scores.argmax(axis=1)
        )
        shuffled_by_fold[int(split_id)] = []
        for _permutation_id in range(args.permutations):
            order = rng.permutation(len(local_y))
            shuffled_query = query[torch.as_tensor(order, dtype=torch.long, device=device)]
            with torch.no_grad():
                shuffled_dynamic, _ = gate(tensor, shuffled_query, mode="dynamic")
            scores = local_stacker + float(args.alpha) * (
                shuffled_dynamic.cpu().numpy() - global_np
            )
            shuffled_by_fold[int(split_id)].append(
                fixed.balanced_accuracy(local_y, scores.argmax(axis=1))
            )

    rows = []
    for permutation_id in range(args.permutations):
        fold_values = [shuffled_by_fold[split_id][permutation_id] for split_id in sorted(shuffled_by_fold)]
        rows.append(
            {
                "permutation_id": permutation_id,
                "all5_mean_balanced_accuracy": float(np.mean(fold_values)),
                "confirmatory_mean_balanced_accuracy": float(np.mean(fold_values[1:])),
                **{
                    f"split_{split_id}_balanced_accuracy": shuffled_by_fold[split_id][permutation_id]
                    for split_id in sorted(shuffled_by_fold)
                },
            }
        )
    permutations = pd.DataFrame(rows)
    observed_values = [observed_by_fold[split_id] for split_id in sorted(observed_by_fold)]
    observed_all = float(np.mean(observed_values))
    observed_confirm = float(np.mean(observed_values[1:]))
    summary = {
        "permutations": int(args.permutations),
        "alpha": float(args.alpha),
        "observed_all5_mean_balanced_accuracy": observed_all,
        "shuffled_all5_mean": float(permutations["all5_mean_balanced_accuracy"].mean()),
        "shuffled_all5_std": float(permutations["all5_mean_balanced_accuracy"].std(ddof=1)),
        "observed_minus_shuffled_all5_mean": float(
            observed_all - permutations["all5_mean_balanced_accuracy"].mean()
        ),
        "all5_permutation_p_one_sided": float(
            (1 + (permutations["all5_mean_balanced_accuracy"] >= observed_all).sum())
            / (args.permutations + 1)
        ),
        "observed_confirmatory_mean_balanced_accuracy": observed_confirm,
        "shuffled_confirmatory_mean": float(
            permutations["confirmatory_mean_balanced_accuracy"].mean()
        ),
        "shuffled_confirmatory_std": float(
            permutations["confirmatory_mean_balanced_accuracy"].std(ddof=1)
        ),
        "observed_minus_shuffled_confirmatory_mean": float(
            observed_confirm
            - permutations["confirmatory_mean_balanced_accuracy"].mean()
        ),
        "confirmatory_permutation_p_one_sided": float(
            (
                1
                + (
                    permutations["confirmatory_mean_balanced_accuracy"]
                    >= observed_confirm
                ).sum()
            )
            / (args.permutations + 1)
        ),
        "observed_by_fold": observed_by_fold,
    }
    permutations.to_csv(output_dir / "attention_query_permutation_scores.csv", index=False)
    (output_dir / "attention_query_permutation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report = [
        "# Last-Token Query Permutation Audit",
        "",
        f"- Permutations: {args.permutations}",
        f"- Observed five-fold balanced accuracy: {observed_all:.6f}",
        f"- Mean shuffled-query five-fold balanced accuracy: {summary['shuffled_all5_mean']:.6f}",
        f"- Observed minus shuffled: {summary['observed_minus_shuffled_all5_mean']:.6f}",
        f"- One-sided permutation p-value: {summary['all5_permutation_p_one_sided']:.4f}",
        f"- Observed confirmatory balanced accuracy: {observed_confirm:.6f}",
        f"- Mean shuffled-query confirmatory balanced accuracy: {summary['shuffled_confirmatory_mean']:.6f}",
        f"- Confirmatory observed minus shuffled: {summary['observed_minus_shuffled_confirmatory_mean']:.6f}",
        f"- Confirmatory one-sided p-value: {summary['confirmatory_permutation_p_one_sided']:.4f}",
    ]
    (output_dir / "attention_query_permutation_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote permutation audit to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
