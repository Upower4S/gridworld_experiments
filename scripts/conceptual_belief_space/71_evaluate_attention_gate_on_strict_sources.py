#!/usr/bin/env python3
"""Evaluate frozen last-token gates on the strongest strict source predictions.

The source predictions are held-out trajectory predictions from the existing
strict nested OOF source stacker plus the existing held-out last-token linear
probe.  No gate is fit or selected here: fold-specific gates are loaded from
development/confirmatory checkpoints and applied directly.
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
spec = importlib.util.spec_from_file_location("strict_source_gate_nested_impl", NESTED_SCRIPT)
nested = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nested
assert spec and spec.loader
spec.loader.exec_module(nested)

base = nested.base
fixed = nested.fixed
ACTIONS = nested.ACTIONS
SOURCE_NAMES = nested.SOURCE_NAMES


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
        default="outputs/conceptual_belief_space/last_token_attention_probe_strong_strict_sources",
    )
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    return parser.parse_args()


def normalize_scores(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    z = centered / scale
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def build_evidence(source_df: pd.DataFrame, last1_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    left = source_df.sort_values("row_id").reset_index(drop=True)
    right = last1_df.sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(left["row_id"].to_numpy(), right["row_id"].to_numpy()):
        raise RuntimeError("Source and last-token prediction rows do not align")
    if not np.array_equal(left["split_id"].to_numpy(), right["split_id"].to_numpy()):
        raise RuntimeError("Source and last-token outer fold IDs do not align")
    prefix_by_source = {
        "last1_ovr": "linear_logit",
        "mean100_ovr": "mean100_score",
        "prefixmean_ovr": "prefixmean_score",
        "segment_ovr": "segment_score",
        "identity_pairwise": "pairwise_score",
    }
    evidence_parts = []
    for source_name in SOURCE_NAMES:
        frame = right if source_name == "last1_ovr" else left
        prefix = prefix_by_source[source_name]
        columns = [f"{prefix}_{action}" for action in ACTIONS]
        evidence_parts.append(normalize_scores(frame[columns].to_numpy(dtype=np.float32)))
    evidence = np.stack(evidence_parts, axis=1)
    return left, evidence


def load_gate_state(split_id: int, args, device):
    import torch

    if split_id == 0:
        payload = torch.load(Path(args.development_gate), map_location="cpu", weights_only=False)
        state = payload["gate_state_dict"]
    else:
        path = Path(args.confirmatory_gate_dir) / f"nested_last_token_attention_split_{split_id}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["gate_state_dict"]
    Gate = nested.make_gate_class()
    gate = Gate(len(ACTIONS), attention_dim=4, dropout=0.0).to(device)
    gate.load_state_dict(state)
    gate.eval()
    return gate


def apply_gate(gate, evidence: np.ndarray, seed: int):
    import torch

    device = next(gate.parameters()).device
    tensor = torch.as_tensor(evidence, dtype=torch.float32, device=device)
    query = tensor[:, SOURCE_NAMES.index("last1_ovr"), :]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(evidence))
    shuffled_query = query[torch.as_tensor(permutation, dtype=torch.long, device=device)]
    out: dict[str, np.ndarray] = {"source_evidence": evidence}
    with torch.no_grad():
        for mode in ["dynamic", "global", "uniform", "fixed_ovr3", "fixed_core4"]:
            logits, weights = gate(tensor, query, mode=mode)
            out[mode] = logits.cpu().numpy().astype(np.float32)
            if mode == "dynamic":
                out["attention"] = weights.cpu().numpy().astype(np.float32)
        shuffled, _ = gate(tensor, shuffled_query, mode="dynamic")
        out["shuffled_query"] = shuffled.cpu().numpy().astype(np.float32)
    return out


def main() -> int:
    import torch

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_df = pd.read_parquet(args.source_predictions)
    last1_df = pd.read_parquet(args.last1_predictions)
    aligned, all_evidence = build_evidence(source_df, last1_df)
    if aligned["row_id"].duplicated().any() or len(aligned) != 98_311:
        raise RuntimeError("Expected exactly one strict held-out source row per prefix")
    if aligned["in_sample"].astype(bool).any() or not aligned["oof_score"].astype(bool).all():
        raise RuntimeError("Source prediction table is not fully held out")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    attention_rows: list[dict] = []
    fold_rows: list[dict] = []
    for split_id in sorted(aligned["split_id"].astype(int).unique()):
        local_mask = aligned["split_id"].astype(int).to_numpy() == split_id
        local_df = aligned.loc[local_mask].reset_index(drop=True)
        local_evidence = all_evidence[local_mask]
        gate = load_gate_state(split_id, args, device)
        outputs = apply_gate(gate, local_evidence, args.seed + 200_000 + split_id)
        info = {
            "gate_parameters": int(sum(parameter.numel() for parameter in gate.parameters())),
            "source_parameters": 927_382,
            "total_probe_parameters": int(
                sum(parameter.numel() for parameter in gate.parameters()) + 927_382
            ),
            "query_dim": len(ACTIONS),
            "query_mode": "last1_scores",
            "source_provenance": "strict_nested_oof_stacker_sources_plus_strict_last1_oof",
        }
        # base metric helpers use positional indexing, so each fold-local frame
        # is intentionally paired with local 0..N-1 indices.
        local_indices = np.arange(len(local_df), dtype=np.int64)
        base.add_all_metrics(metric_rows, local_df, local_indices, outputs, split_id, info)
        prediction_frames.append(base.make_prediction_frame(local_df, local_indices, outputs, split_id))
        attention_rows.extend(base.attention_statistics(outputs, split_id))
        dynamic = outputs["dynamic"].argmax(axis=1)
        true = base.action_indices(local_df["top_action"])
        fold_rows.append(
            {
                "split_id": int(split_id),
                "rows": int(len(local_df)),
                "trajectory_overlap_with_training": 0,
                "strict_source_rows": int(len(local_df)),
                "balanced_accuracy": fixed.balanced_accuracy(true, dynamic),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    attention = pd.DataFrame(attention_rows)
    fold_audit = pd.DataFrame(fold_rows)
    metrics.to_csv(output_dir / "strict_source_attention_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "strict_source_attention_predictions.parquet", index=False)
    attention.to_csv(output_dir / "strict_source_attention_weights.csv", index=False)
    fold_audit.to_csv(output_dir / "strict_source_attention_fold_audit.csv", index=False)
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
    confirmatory = all_prefixes[all_prefixes["split_id"].ne(0)]
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
    summary.to_csv(output_dir / "strict_source_attention_summary.csv", index=False)
    confirmatory_summary.to_csv(
        output_dir / "strict_source_attention_confirmatory_summary.csv", index=False
    )
    manifest = {
        "evaluation_only": True,
        "no_test_label_fitting_or_selection": True,
        "source_predictions": args.source_predictions,
        "last1_predictions": args.last1_predictions,
        "source_rows_all_oof": True,
        "fold_gates": {
            "development": args.development_gate,
            "confirmatory": args.confirmatory_gate_dir,
        },
        "query": "strict held-out four-action evidence decoded from the actual last prefix token",
        "excluded_inputs": base.FORBIDDEN_MODEL_INPUTS,
    }
    (output_dir / "strict_source_attention_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = [
        "# Last-Token Attention Probe on Strong Strict Sources",
        "",
        "- Fold gates are frozen; this script performs evaluation only.",
        "- Every source and last-token score is an outer held-out trajectory prediction.",
        "- No test labels are used for fitting or selection.",
        "",
        "## Five-fold summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Confirmatory folds 1-4",
        "",
        confirmatory_summary.to_markdown(index=False),
    ]
    (output_dir / "strict_source_attention_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(confirmatory_summary.to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
