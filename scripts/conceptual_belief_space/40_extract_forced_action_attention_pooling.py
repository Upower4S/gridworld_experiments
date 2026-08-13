#!/usr/bin/env python3
"""Forced-final-context attention pooling for activation-only rank probes.

This optional branch needs a local model/runtime that returns attention tensors.
API scoring backends such as Together cannot provide the query-to-prefix attention
matrix required here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED


RANK_SCRIPT = Path(__file__).with_name("39_train_activation_rank_probes.py")
spec = importlib.util.spec_from_file_location("activation_rank_probe_impl", RANK_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import rank-probe implementation from {RANK_SCRIPT}")
rank_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank_impl
spec.loader.exec_module(rank_impl)

Q_COLS = [f"q_{action}" for action in ACTIONS]
FORCED_FINAL_PREFIX = '<|end|><|start|>assistant<|channel|>final<|message|>{\n  "action": "'
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
POOLING_VARIANTS = [
    "attn_mean_heads_topk",
    "attn_max_heads_topk",
    "attn_low_entropy_head_topk",
    "attn_recency_tau5_topk",
    "attn_minus_mean20_topk",
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def load_prefix_rows(path: Path, beliefs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "prefix_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "prompt_text",
        "analysis_prefix_text",
        "analysis_prefix_token_ids_json",
    ]
    prefixes = rank_impl.read_parquet_robust(path, columns=cols)
    merge_cols = ["trajectory_id", "step_id", "prefix_token_index"]
    if "prefix_id" in beliefs.columns and "prefix_id" in prefixes.columns:
        merge_cols = ["prefix_id"]
    out = beliefs.merge(prefixes, on=merge_cols, how="left", suffixes=("", "_prefix"))
    missing = out["prompt_text"].isna().sum()
    if missing:
        raise ValueError(f"Missing prompt/prefix text for {missing} belief rows after merge with {path}")
    return out


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_forced_input(tokenizer, row: pd.Series) -> tuple[list[int], int, int]:
    prompt_text = str(row["prompt_text"])
    analysis_ids = json.loads(str(row["analysis_prefix_token_ids_json"]))
    if not isinstance(analysis_ids, list):
        raise ValueError("analysis_prefix_token_ids_json did not decode to a list")
    base_ids = encode_no_special(tokenizer, prompt_text + ANALYSIS_HEADER)
    tail_ids = encode_no_special(tokenizer, FORCED_FINAL_PREFIX)
    input_ids = base_ids + [int(x) for x in analysis_ids] + tail_ids
    analysis_start = len(base_ids)
    analysis_end = analysis_start + len(analysis_ids)
    return input_ids, analysis_start, analysis_end


def attention_weights_for_layer(attentions: Any, layer: int, analysis_start: int, analysis_end: int) -> np.ndarray:
    if layer >= len(attentions):
        raise ValueError(f"Requested attention layer {layer}, but model returned {len(attentions)} layers")
    attn = attentions[layer]
    if hasattr(attn, "detach"):
        attn = attn.detach().float().cpu().numpy()
    attn = np.asarray(attn)
    # Shape: batch, heads, query_position, key_position. Query is final token before action completion.
    span = attn[0, :, -1, analysis_start:analysis_end].astype(np.float32)
    if span.ndim != 2:
        raise ValueError(f"Unexpected attention span shape: {span.shape}")
    return span


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights[~np.isfinite(weights)] = 0.0
    total = float(weights.sum())
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float32)
    return weights / total


def topk_weights(base_weights: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(base_weights) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float32)
    k = min(int(top_k), len(base_weights))
    idx = np.argpartition(-base_weights, k - 1)[:k]
    idx = idx[np.argsort(-base_weights[idx])]
    return idx.astype(int), normalize(base_weights[idx])


def low_entropy_head(span: np.ndarray) -> np.ndarray:
    probs = span.copy()
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return span[int(np.argmin(entropy))]


def attention_base_weights(span: np.ndarray, variant: str) -> np.ndarray:
    if variant == "attn_mean_heads_topk" or variant == "attn_minus_mean20_topk":
        return span.mean(axis=0)
    if variant == "attn_max_heads_topk":
        return span.max(axis=0)
    if variant == "attn_low_entropy_head_topk":
        return low_entropy_head(span)
    if variant == "attn_recency_tau5_topk":
        mean = span.mean(axis=0)
        ages = np.arange(len(mean) - 1, -1, -1, dtype=np.float32)
        recency = np.exp(-ages / 5.0)
        return mean * recency
    raise ValueError(f"Unknown attention pooling variant: {variant}")


def pooled_activation_for_row(
    row: pd.Series,
    span: np.ndarray,
    pointer: Any,
    loader: Any,
    *,
    layer: int,
    top_k: int,
    variant: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    analysis_len = span.shape[1]
    if analysis_len <= 0:
        raise ValueError("Cannot pool attention over an empty analysis prefix")
    base = attention_base_weights(span, variant)
    selected_offsets, weights = topk_weights(base, top_k)
    vectors = []
    cot_indices = []
    for offset in selected_offsets:
        cot_index = int(offset) + 1
        if cot_index not in pointer.cot_rows_by_token:
            continue
        vec = loader.load_row(
            rank_impl.path_for_layer(pointer.cot_file, layer),
            pointer.cot_rows_by_token[cot_index],
            pointer.hidden_dim,
        )
        vectors.append(vec.astype(np.float32, copy=False))
        cot_indices.append(cot_index)
    if not vectors:
        raise ValueError("No selected attention key tokens had matching activation rows")
    weights = weights[: len(vectors)]
    weights = normalize(weights)
    pooled = (np.asarray(vectors, dtype=np.float32) * weights[:, None]).sum(axis=0)
    if variant == "attn_minus_mean20_topk":
        prefix_k = int(row["prefix_token_index"])
        start = max(1, prefix_k - 19)
        mean_vecs = []
        for cot_index in range(start, max(prefix_k, 0) + 1):
            if cot_index in pointer.cot_rows_by_token:
                mean_vecs.append(
                    loader.load_row(
                        rank_impl.path_for_layer(pointer.cot_file, layer),
                        pointer.cot_rows_by_token[cot_index],
                        pointer.hidden_dim,
                    )
                )
        if mean_vecs:
            pooled = pooled - np.asarray(mean_vecs, dtype=np.float32).mean(axis=0)
    meta = {
        "selected_cot_indices": cot_indices,
        "selected_attention_weights": weights.tolist(),
        "analysis_prefix_attention_tokens": int(analysis_len),
    }
    return pooled.astype(np.float32, copy=False), meta


def load_model_and_tokenizer(model_name_or_path: str, dtype: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("transformers and torch are required for local attention extraction") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    torch_dtype = dtype_map[dtype]
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_attention_feature_matrix(
    df: pd.DataFrame,
    pointers: dict[str, Any],
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    top_k: int,
    variant: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch

    loader = rank_impl.BF16MatrixCache()
    features = []
    meta_rows = []
    device = next(model.parameters()).device
    for row_idx, row in df.iterrows():
        input_ids, analysis_start, analysis_end = build_forced_input(tokenizer, row)
        if analysis_end <= analysis_start:
            raise ValueError(f"Empty analysis prefix for row {row_idx}")
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_tensor, output_attentions=True, use_cache=False, return_dict=True)
        if out.attentions is None:
            raise RuntimeError("Local model did not return attentions. Use a runtime/model that supports output_attentions=True.")
        span = attention_weights_for_layer(out.attentions, layer, analysis_start, analysis_end)
        pointer = pointers[str(row["decision_id"])]
        pooled, meta = pooled_activation_for_row(row, span, pointer, loader, layer=layer, top_k=top_k, variant=variant)
        features.append(pooled)
        meta.update({
            "row_id": int(row.get("row_id", row_idx)),
            "prefix_id": int(row.get("prefix_id", row_idx)) if pd.notna(row.get("prefix_id", np.nan)) else None,
            "decision_id": str(row["decision_id"]),
            "layer": int(layer),
            "top_k": int(top_k),
            "pooling_variant": variant,
            "attention_keys_exclude_prompt": True,
            "attention_keys_exclude_forced_final_json": True,
        })
        meta_rows.append(meta)
    return np.asarray(features, dtype=np.float32), meta_rows


def append_or_write_metrics(output_dir: Path, new_metrics: pd.DataFrame) -> pd.DataFrame:
    path = output_dir / "rank_probe_metrics.csv"
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        all_metrics = pd.concat([old, new_metrics], ignore_index=True)
    else:
        all_metrics = new_metrics
    all_metrics.to_csv(path, index=False)
    return all_metrics


def append_or_write_predictions(output_dir: Path, new_predictions: pd.DataFrame) -> pd.DataFrame:
    path = output_dir / "rank_probe_predictions.parquet"
    if path.exists() and path.stat().st_size > 0:
        old = rank_impl.read_parquet_robust(path)
        all_predictions = pd.concat([old, new_predictions], ignore_index=True)
    else:
        all_predictions = new_predictions
    all_predictions.to_parquet(path, index=False)
    return all_predictions


def plot_attention_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    attn = metrics[(metrics["feature_family"] == "forced_action_attention_pooling") & (metrics["eval_subset"] == "all_prefixes")]
    if attn.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, sub in attn.groupby("feature_variant"):
        summary = sub.groupby("top_k")["pairwise_ranking_accuracy"].mean().sort_index()
        ax.plot(summary.index, summary.values, marker="o", label=variant)
    ax.set_xlabel("attention top_k")
    ax.set_ylabel("pairwise ranking accuracy")
    ax.set_title("Attention pooling top-k comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(fig_dir / "attention_topk_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, sub in attn.groupby("feature_variant"):
        summary = sub.groupby("layer")["pairwise_ranking_accuracy"].mean().sort_index()
        ax.plot(summary.index, summary.values, marker="o", label=variant)
    ax.set_xlabel("attention/activation layer")
    ax.set_ylabel("pairwise ranking accuracy")
    ax.set_title("Attention pooling layer comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(fig_dir / "attention_layer_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_attention_report(output_dir: Path, manifest: dict[str, Any], metrics: pd.DataFrame) -> None:
    report_path = output_dir / "activation_rank_probe_report.md"
    lines = []
    if report_path.exists():
        lines.append(report_path.read_text().rstrip())
        lines.append("")
    lines.extend([
        "## Forced-Action Attention Pooling Run",
        "",
        f"- Local model: `{manifest.get('model_name_or_path')}`",
        f"- Rows processed: {manifest.get('n_rows'):,}",
        f"- Layers: {manifest.get('layers')}",
        f"- top_k values: {manifest.get('top_k_values')}",
        f"- Pooling variants: {manifest.get('pooling_variants')}",
        "- Attention keys are restricted to generated analysis-prefix tokens.",
        "- Prompt tokens and forced final-JSON tokens are excluded from the attention key span.",
    ])
    attn = metrics[(metrics["feature_family"] == "forced_action_attention_pooling") & (metrics["eval_subset"] == "all_prefixes")]
    if not attn.empty:
        best = attn.sort_values("pairwise_ranking_accuracy", ascending=False).iloc[0]
        lines.extend([
            "",
            "Best attention pooling row:",
            "",
            f"- layer: {best['layer']}",
            f"- top_k: {best.get('top_k')}",
            f"- pooling variant: {best['feature_variant']}",
            f"- pairwise ranking accuracy: {best['pairwise_ranking_accuracy']}",
            f"- top-action accuracy: {best['top_action_accuracy']}",
        ])
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_rank_probes")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/activation_rank_probes")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--layers", default="10,12,14,16,18,20,22")
    parser.add_argument("--top_k", default="1,2,4,8,16,32")
    parser.add_argument("--pooling_variants", default=",".join(POOLING_VARIANTS))
    parser.add_argument("--limit_decisions", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tie_eps", type=float, default=1e-6)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_csv_ints(args.layers)
    top_k_values = parse_csv_ints(args.top_k)
    pooling_variants = parse_csv_strings(args.pooling_variants)
    bad = [v for v in pooling_variants if v not in POOLING_VARIANTS]
    if bad:
        raise ValueError(f"Unsupported pooling variants: {bad}")

    beliefs = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=args.limit_decisions, seed=args.seed)
    beliefs = load_prefix_rows(Path(args.prefixes), beliefs)
    pointers = rank_impl.build_activation_pointers(Path(args.activation_index), set(beliefs["decision_id"].astype(str)), layers=layers)
    splits = rank_impl.make_group_splits(beliefs, args.n_splits, args.seed)
    q_all = beliefs[Q_COLS].to_numpy(dtype=np.float64)
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.dtype, args.device)

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    pool_meta: list[dict[str, Any]] = []
    for layer in layers:
        for top_k in top_k_values:
            for variant in pooling_variants:
                print(f"Extracting attention-pooled features layer={layer} top_k={top_k} variant={variant}")
                x, meta = extract_attention_feature_matrix(beliefs, pointers, model, tokenizer, layer=layer, top_k=top_k, variant=variant)
                pool_meta.extend(meta)
                for split in splits:
                    split_id = int(split["split_id"])
                    train_idx = split["train_idx"]
                    test_idx = split["test_idx"]
                    train_groups = set(beliefs.iloc[train_idx]["trajectory_id"].astype(str))
                    test_groups = set(beliefs.iloc[test_idx]["trajectory_id"].astype(str))
                    leakage = sorted(train_groups.intersection(test_groups))
                    if leakage:
                        raise RuntimeError(f"Trajectory leakage in split {split_id}: {leakage[:5]}")
                    x_train, x_test, _, _ = rank_impl.standardize_train_test(x[train_idx], x[test_idx])
                    _, scores, losses = rank_impl.fit_linear_rank_probe(
                        x_train,
                        q_all[train_idx],
                        x_test,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        seed=args.seed + split_id + layer + int(top_k),
                        tie_eps=args.tie_eps,
                        batch_size=args.batch_size,
                    )
                    base = {
                        "feature_family": "forced_action_attention_pooling",
                        "label_mode": "q_order",
                        "split_id": split_id,
                        "layer": layer,
                        "representation": "raw",
                        "feature_variant": variant,
                        "top_k": int(top_k),
                        "window_size": np.nan,
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "final_train_loss": float(losses[-1]) if losses else float("nan"),
                    }
                    rank_impl.add_metric_rows(
                        metric_rows,
                        base=base,
                        scores=scores,
                        q=q_all[test_idx],
                        test_df=beliefs.iloc[test_idx],
                        tie_eps=args.tie_eps,
                    )
                    pred = rank_impl.make_predictions_frame(
                        beliefs.iloc[test_idx].reset_index(drop=True),
                        scores,
                        split_id=split_id,
                        layer=layer,
                        representation="raw",
                        feature_variant=variant,
                        family="forced_action_attention_pooling",
                    )
                    pred["top_k"] = int(top_k)
                    pred_frames.append(pred)

    new_metrics = pd.DataFrame(metric_rows)
    all_metrics = append_or_write_metrics(output_dir, new_metrics)
    new_predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    append_or_write_predictions(output_dir, new_predictions)
    manifest = {
        "status": "completed",
        "model_name_or_path": args.model_name_or_path,
        "n_rows": int(len(beliefs)),
        "n_decisions": int(beliefs["decision_id"].nunique()),
        "layers": layers,
        "top_k_values": top_k_values,
        "pooling_variants": pooling_variants,
        "forced_context_suffix": FORCED_FINAL_PREFIX,
        "query_token": "final token immediately before action completion",
        "attention_key_span": "analysis_prefix_token_ids_json only",
        "excludes_prompt_tokens": True,
        "excludes_forced_final_json_tokens": True,
        "pooling_metadata_sample": pool_meta[:20],
    }
    rank_impl.write_json(output_dir / "attention_pooling_manifest.json", manifest)
    if pool_meta:
        (output_dir / "attention_pooling_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pool_meta))
    plot_attention_metrics(all_metrics, fig_dir)
    update_attention_report(output_dir, manifest, all_metrics)
    print(f"Wrote attention metrics under {output_dir}")
    print(f"Wrote attention figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
