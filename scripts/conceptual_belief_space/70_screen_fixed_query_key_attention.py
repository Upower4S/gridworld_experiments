#!/usr/bin/env python3
"""Screen literal last-token query/key attention on cached nested evidence.

The last-token OVR evidence is the query.  Causal summary-expert evidence forms
keys and values.  Attention weights are deterministic query/key similarities;
only a small development grid over similarity, temperature, memory set, and
residual mixing is selected on grouped gate-validation trajectories.
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
spec = importlib.util.spec_from_file_location("fixed_query_key_nested_impl", NESTED_SCRIPT)
nested = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nested
assert spec and spec.loader
spec.loader.exec_module(nested)

base = nested.base
fixed = nested.fixed
seg = nested.seg
ACTIONS = nested.ACTIONS
SOURCE_NAMES = nested.SOURCE_NAMES
Q_COLS = nested.Q_COLS

MEMORY_SETS = {
    "ovr3": ["mean100_ovr", "prefixmean_ovr", "segment_ovr"],
    "core4": ["mean100_ovr", "prefixmean_ovr", "segment_ovr", "identity_pairwise"],
    "all_except_query": ["mean100_ovr", "prefixmean_ovr", "segment_ovr", "identity_pairwise"],
}


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    centered = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(centered)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def attention_scores(
    evidence: np.ndarray,
    *,
    memory_name: str,
    similarity: str,
    beta: float,
    residual_alpha: float,
    query_override: np.ndarray | None = None,
):
    query_id = SOURCE_NAMES.index("last1_ovr")
    memory_ids = [SOURCE_NAMES.index(name) for name in MEMORY_SETS[memory_name]]
    query = evidence[:, query_id, :] if query_override is None else query_override
    memory = evidence[:, memory_ids, :]
    q_center = query - query.mean(axis=1, keepdims=True)
    k_center = memory - memory.mean(axis=2, keepdims=True)

    if similarity == "centered_dot":
        logits = float(beta) * np.einsum("na,nsa->ns", q_center, k_center)
        weights_scalar = softmax(logits, axis=1)
        weights = np.repeat(weights_scalar[:, :, None], len(ACTIONS), axis=2)
    elif similarity == "negative_mse":
        logits = -float(beta) * np.square(memory - query[:, None, :]).mean(axis=2)
        weights_scalar = softmax(logits, axis=1)
        weights = np.repeat(weights_scalar[:, :, None], len(ACTIONS), axis=2)
    elif similarity == "action_product":
        logits = float(beta) * q_center[:, None, :] * k_center
        weights = softmax(logits, axis=1)
    elif similarity == "support_product":
        logits = float(beta) * query[:, None, :] * memory
        weights = softmax(logits, axis=1)
    else:
        raise ValueError(f"Unknown similarity: {similarity}")

    attended = (weights * memory).sum(axis=1)
    fixed_memory = memory.mean(axis=1)
    scores = (1.0 - float(residual_alpha)) * fixed_memory + float(residual_alpha) * attended
    return scores.astype(np.float32), weights.astype(np.float32), memory_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/fixed_query_key_attention_screen")
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--similarities", default="centered_dot,negative_mse,action_product,support_product")
    parser.add_argument("--betas", default="1,2,4,8,16,32")
    parser.add_argument("--residual_alphas", default="0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(args.evidence_dir)
    oof_path = evidence_dir / f"nested_last_token_attention_oof_evidence_split_{args.split_id}.npz"
    test_path = evidence_dir / f"nested_last_token_attention_test_evidence_split_{args.split_id}.npz"
    df = fixed.rank_impl.prepare_belief_rows(
        Path(args.beliefs),
        limit_decisions=None,
        seed=args.seed,
        prefix_path=Path(args.prefixes),
    ).reset_index(drop=True)
    y = base.action_indices(df["top_action"])
    splits = fixed.rank_impl.make_group_splits(df, args.n_splits, args.seed)
    split = next(item for item in splits if int(item["split_id"]) == args.split_id)
    outer_train = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    meta_train, meta_val = seg.ms_impl.train_val_split_by_group(
        df,
        outer_train,
        args.seed + 50_000 + args.split_id,
        0.15,
    )
    base.assert_group_disjoint(df, meta_train, meta_val, "fixed attention meta split")
    with np.load(oof_path) as data:
        cached_rows = data["row_indices"].astype(np.int64)
        cached_evidence = data["source_evidence"].astype(np.float32)
    with np.load(test_path) as data:
        cached_test_rows = data["row_indices"].astype(np.int64)
        test_evidence = data["source_evidence"].astype(np.float32)
    if not np.array_equal(cached_rows, outer_train) or not np.array_equal(cached_test_rows, test_idx):
        raise RuntimeError("Cached evidence rows do not match canonical split rows")
    global_evidence = np.full(
        (len(df), len(SOURCE_NAMES), len(ACTIONS)), np.nan, dtype=np.float32
    )
    global_evidence[outer_train] = cached_evidence

    similarities = [value.strip() for value in args.similarities.split(",") if value.strip()]
    candidate_rows = []
    for memory_name in MEMORY_SETS:
        for similarity in similarities:
            for beta in parse_floats(args.betas):
                for residual_alpha in parse_floats(args.residual_alphas):
                    val_scores, val_weights, _ = attention_scores(
                        global_evidence[meta_val],
                        memory_name=memory_name,
                        similarity=similarity,
                        beta=beta,
                        residual_alpha=residual_alpha,
                    )
                    pred = val_scores.argmax(axis=1)
                    candidate_rows.append(
                        {
                            "memory_name": memory_name,
                            "similarity": similarity,
                            "beta": beta,
                            "residual_alpha": residual_alpha,
                            "meta_val_balanced_accuracy": fixed.balanced_accuracy(y[meta_val], pred),
                            "meta_val_accuracy": float((pred == y[meta_val]).mean()),
                            "mean_attention_std": float(val_weights.std(axis=0).mean()),
                        }
                    )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["meta_val_balanced_accuracy", "meta_val_accuracy"], ascending=False
    )
    selected = candidates.iloc[0].to_dict()
    candidates["selected"] = False
    candidates.loc[candidates.index[0], "selected"] = True

    dynamic, selected_weights, memory_ids = attention_scores(
        test_evidence,
        memory_name=str(selected["memory_name"]),
        similarity=str(selected["similarity"]),
        beta=float(selected["beta"]),
        residual_alpha=float(selected["residual_alpha"]),
    )
    rng = np.random.default_rng(args.seed + 200_000 + args.split_id)
    query_perm = rng.permutation(len(test_idx))
    shuffled_query = test_evidence[
        query_perm, SOURCE_NAMES.index("last1_ovr"), :
    ]
    shuffled, _shuffled_weights, _ = attention_scores(
        test_evidence,
        memory_name=str(selected["memory_name"]),
        similarity=str(selected["similarity"]),
        beta=float(selected["beta"]),
        residual_alpha=float(selected["residual_alpha"]),
        query_override=shuffled_query,
    )
    ovr3_ids = [SOURCE_NAMES.index(name) for name in MEMORY_SETS["ovr3"]]
    core4_ids = [SOURCE_NAMES.index(name) for name in MEMORY_SETS["core4"]]
    fixed_ovr3 = test_evidence[:, ovr3_ids, :].mean(axis=1)
    fixed_core4 = test_evidence[:, core4_ids, :].mean(axis=1)
    uniform = test_evidence.mean(axis=1)
    outputs = {
        "dynamic": dynamic,
        "global": test_evidence[:, memory_ids, :].mean(axis=1),
        "uniform": uniform,
        "fixed_ovr3": fixed_ovr3,
        "fixed_core4": fixed_core4,
        "shuffled_query": shuffled,
        "source_evidence": test_evidence,
        "attention": np.zeros_like(test_evidence),
    }
    for local_id, source_id in enumerate(memory_ids):
        outputs["attention"][:, source_id, :] = selected_weights[:, local_id, :]
    info = {
        "gate_parameters": 0,
        "query_dim": len(ACTIONS),
        "query_mode": "last1_scores",
        "similarity": selected["similarity"],
        "beta": selected["beta"],
        "residual_alpha": selected["residual_alpha"],
        "memory_name": selected["memory_name"],
        "gate_validation_balanced_accuracy": selected["meta_val_balanced_accuracy"],
    }
    metric_rows: list[dict] = []
    base.add_all_metrics(metric_rows, df, test_idx, outputs, args.split_id, info)
    metrics = pd.DataFrame(metric_rows)
    candidates.to_csv(output_dir / "fixed_query_key_attention_candidates.csv", index=False)
    metrics.to_csv(output_dir / "fixed_query_key_attention_metrics.csv", index=False)
    predictions = base.make_prediction_frame(df, test_idx, outputs, args.split_id)
    predictions.to_parquet(output_dir / "fixed_query_key_attention_predictions.parquet", index=False)
    pd.DataFrame(base.attention_statistics(outputs, args.split_id)).to_csv(
        output_dir / "fixed_query_key_attention_weights.csv", index=False
    )
    all_prefix = metrics[metrics["eval_subset"].eq("all_prefixes")][
        ["source_name", "balanced_accuracy", "accuracy", "macro_f1", "pairwise_accuracy"]
    ]
    report = [
        "# Fixed Last-Token Query/Key Attention Screen",
        "",
        "Configuration selection used grouped gate-validation trajectories in development split 0.",
        "",
        "## Selected configuration",
        "",
        pd.DataFrame([selected]).to_markdown(index=False),
        "",
        "## Outer development-fold diagnostics",
        "",
        all_prefix.to_markdown(index=False),
    ]
    (output_dir / "fixed_query_key_attention_report.md").write_text("\n".join(report) + "\n")
    (output_dir / "fixed_query_key_attention_manifest.json").write_text(
        json.dumps(
            {
                "development_only": True,
                "query": "last-token OVR action evidence",
                "keys_and_values": MEMORY_SETS[str(selected["memory_name"])],
                "selection": selected,
                "excluded_inputs": base.FORBIDDEN_MODEL_INPUTS,
                "arguments": vars(args),
            },
            indent=2,
        )
        + "\n"
    )
    print(candidates.head(20).to_string(index=False), flush=True)
    print(all_prefix.to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
