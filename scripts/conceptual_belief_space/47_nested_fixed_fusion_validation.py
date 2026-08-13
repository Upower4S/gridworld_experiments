#!/usr/bin/env python3
"""Nested validation of a fixed activation-only temporal+pairwise fusion.

The earlier best score-fusion result crossed 0.7 balanced q-top accuracy but was
exploratory because it reused saved OOF component scores.  This script retrains
the component linear probes inside each outer trajectory split, then applies the
pre-specified fusion:

  mean_row_softmax_t1(mean100_ovr, identity_pairwise, prefixmean_ovr, segment_ovr)

All component models are linear in activation-only cached features.  Metadata is
used only for row alignment, grouped splits, residual indexing, labels, and
reporting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

RANK_SCRIPT = Path(__file__).with_name("39_train_activation_rank_probes.py")
SEGMENT_SCRIPT = Path(__file__).with_name("45_train_cached_segment_combo_linear.py")

rank_spec = importlib.util.spec_from_file_location("activation_rank_probe_impl", RANK_SCRIPT)
rank_impl = importlib.util.module_from_spec(rank_spec)
sys.modules[rank_spec.name] = rank_impl
assert rank_spec and rank_spec.loader
rank_spec.loader.exec_module(rank_impl)

seg_spec = importlib.util.spec_from_file_location("segment_combo_impl", SEGMENT_SCRIPT)
seg_impl = importlib.util.module_from_spec(seg_spec)
sys.modules[seg_spec.name] = seg_impl
assert seg_spec and seg_spec.loader
seg_spec.loader.exec_module(seg_impl)

Q_COLS = [f"q_{a}" for a in ACTIONS]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]

SOURCE_CONFIGS = {
    "mean100_ovr": ["raw100"],
    "prefixmean_ovr": ["rawpfx", "residpfx"],
    "segment_ovr": ["rawpfx", "raw20", "raw100", "residpfx", "resid20", "resid100"],
    "identity_pairwise": ["raw20"],
}
FUSION_SOURCES = ["mean100_ovr", "identity_pairwise", "prefixmean_ovr", "segment_ovr"]
TEMPORAL_OVR_FUSION_SOURCES = ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_split_ids(text: str, all_ids: list[int]) -> list[int]:
    if text == "all":
        return all_ids
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def action_indices(values: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


def row_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    z = centered / scale / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def fixed_fusion(source_scores: dict[str, np.ndarray], source_names: list[str] | None = None) -> np.ndarray:
    names = source_names or FUSION_SOURCES
    stacked = np.stack([row_softmax(source_scores[name], temperature=1.0) for name in names], axis=0)
    return stacked.mean(axis=0)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for idx in range(len(ACTIONS)):
        mask = y_true == idx
        if mask.any():
            vals.append(float((y_pred[mask] == idx).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for c in range(len(ACTIONS)):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def pairwise_accuracy(scores: np.ndarray, q: np.ndarray, tie_eps: float = 1e-6) -> float:
    correct = 0
    total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def metric_row(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, *, split_id: int, eval_subset: str, source_name: str) -> dict[str, Any]:
    sub = df.iloc[indices]
    y = action_indices(sub["top_action"])
    pred = scores.argmax(axis=1)
    q = sub[Q_COLS].to_numpy(dtype=np.float64)
    row = {
        "source_name": source_name,
        "split_id": int(split_id),
        "eval_subset": eval_subset,
        "n_rows": int(len(indices)),
        "accuracy": float((pred == y).mean()) if len(indices) else float("nan"),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "pairwise_accuracy": pairwise_accuracy(scores, q),
    }
    for idx, action in enumerate(ACTIONS):
        mask = y == idx
        row[f"recall_{action}"] = float((pred[mask] == idx).mean()) if mask.any() else float("nan")
    return row


def add_subset_metrics(rows: list[dict[str, Any]], df: pd.DataFrame, test_idx: np.ndarray, scores: np.ndarray, *, split_id: int, source_name: str) -> None:
    rows.append(metric_row(df, test_idx, scores, split_id=split_id, eval_subset="all_prefixes", source_name=source_name))
    local = df.iloc[test_idx]
    for col in SUBSET_COLS:
        if col not in local.columns:
            continue
        mask = local[col].fillna(False).astype(bool).to_numpy()
        if mask.any():
            rows.append(metric_row(df, test_idx[mask], scores[mask], split_id=split_id, eval_subset=col, source_name=source_name))


def train_source_ovr(
    arrays: dict[tuple[int, int], np.ndarray],
    terms: list[str],
    layers: list[int],
    df: pd.DataFrame,
    y: np.ndarray,
    q_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    *,
    source_name: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    logits, info = seg_impl.train_model(
        arrays,
        terms,
        layers,
        y,
        q_all,
        train_idx,
        val_idx,
        test_idx,
        feature_rows,
        pre_rows,
        loss_kind="ovr_bce",
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )
    info.update({"source_name": source_name, "terms": terms})
    return logits, info


def rank_targets(q: np.ndarray, tie_eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros((len(q), len(PAIR_INDICES)), dtype=np.float32)
    mask = np.ones_like(y, dtype=np.float32)
    for pair_id, (i, j) in enumerate(PAIR_INDICES):
        diff = q[:, i] - q[:, j]
        y[:, pair_id] = (diff > 0).astype(np.float32)
        mask[:, pair_id] = (np.abs(diff) > tie_eps).astype(np.float32)
    return y, mask


def train_identity_pairwise(
    arrays: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    df: pd.DataFrame,
    q_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    terms = SOURCE_CONFIGS["identity_pairwise"]
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    mean, scale = seg_impl.compute_scaler(arrays, terms, layers, train_idx, feature_rows, pre_rows, batch_size=max(batch_size, 512))
    model = torch.nn.Linear(len(mean), len(PAIR_INDICES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_pair, mask_pair = rank_targets(q_all)
    score_idx = val_idx if len(val_idx) else train_idx
    best_state = None
    best_val = -1.0
    best_epoch = -1
    stale = 0

    def predict_pairs(indices: np.ndarray) -> np.ndarray:
        outs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                xb = (seg_impl.build_batch(arrays, terms, layers, idx, feature_rows, pre_rows) - mean) / scale
                out = model(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
                outs.append(out.astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(PAIR_INDICES)), dtype=np.float32)

    def pair_logits_to_action_scores(pair_logits: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(pair_logits), len(ACTIONS)), dtype=np.float32)
        for pair_id, (i, j) in enumerate(PAIR_INDICES):
            # Margin voting: positive evidence for i over j and negative evidence for j over i.
            scores[:, i] += pair_logits[:, pair_id]
            scores[:, j] -= pair_logits[:, pair_id]
        return scores

    for epoch in range(int(epochs)):
        order = rng.permutation(train_idx)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = (seg_impl.build_batch(arrays, terms, layers, idx, feature_rows, pre_rows) - mean) / scale
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device))
            yb = torch.tensor(y_pair[idx], dtype=torch.float32, device=device)
            mb = torch.tensor(mask_pair[idx], dtype=torch.float32, device=device)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (raw * mb).sum() / mb.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_pair_logits = predict_pairs(score_idx)
        val_scores = pair_logits_to_action_scores(val_pair_logits)
        y_val = q_all[score_idx].argmax(axis=1)
        val_bal = balanced_accuracy(y_val, val_scores.argmax(axis=1))
        if val_bal > best_val + 1e-5:
            best_val = val_bal
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_scores = pair_logits_to_action_scores(predict_pairs(test_idx))
    return test_scores, {"source_name": "identity_pairwise", "best_epoch": int(best_epoch), "validation_balanced_accuracy": float(best_val), "device": str(device), "feature_dim": int(len(mean)), "terms": terms}


def make_predictions(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, *, split_id: int, source_name: str) -> pd.DataFrame:
    sub = df.iloc[indices].copy()
    for idx, action in enumerate(ACTIONS):
        sub[f"score_{action}"] = scores[:, idx]
    sub["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[scores.argmax(axis=1)]
    sub["split_id"] = int(split_id)
    sub["source_name"] = source_name
    keep = [c for c in ["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action", "predicted_top_action", "split_id", "source_name", *Q_COLS, *SUBSET_COLS, *[f"score_{a}" for a in ACTIONS]] if c in sub.columns]
    return sub[keep]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_nested_fixed_fusion_validation")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    df = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=None, seed=args.seed, prefix_path=Path(args.prefixes))
    y = action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    feature_rows = seg_impl.ms_impl.feature_rows_by_position(df)
    pre_rows = seg_impl.ms_impl.pre_feature_rows_by_position(df)
    splits = rank_impl.make_group_splits(df, args.n_splits, args.seed)
    wanted = set(parse_split_ids(args.split_ids, [int(s["split_id"]) for s in splits]))
    splits = [s for s in splits if int(s["split_id"]) in wanted]

    all_terms = sorted({term for terms in SOURCE_CONFIGS.values() for term in terms})
    arrays = seg_impl.load_arrays(all_terms, layers)
    manifest = {
        "activation_only": True,
        "fixed_fusions": {
            "fixed_temporal_identity_pair_fusion": FUSION_SOURCES,
            "fixed_temporal_ovr3_fusion": TEMPORAL_OVR_FUSION_SOURCES,
        },
        "sources": SOURCE_CONFIGS,
        "layers": layers,
        "n_rows": int(len(df)),
        "n_decisions": int(df["decision_id"].nunique()),
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "note": "All component probes are retrained inside each outer split; this is stricter than saved-score fusion.",
    }
    (output_dir / "nested_fixed_fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    source_info: list[dict[str, Any]] = []

    for split in splits:
        split_id = int(split["split_id"])
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        if set(df.iloc[train_idx]["trajectory_id"].astype(str)) & set(df.iloc[test_idx]["trajectory_id"].astype(str)):
            raise RuntimeError(f"Trajectory leakage in split {split_id}")
        subtrain_idx, val_idx = seg_impl.ms_impl.train_val_split_by_group(df, train_idx, args.seed + split_id, args.val_fraction)
        print(f"Outer split {split_id}: train={len(subtrain_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        source_scores: dict[str, np.ndarray] = {}
        for source_name in ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]:
            terms = SOURCE_CONFIGS[source_name]
            print(f"  Training source={source_name} terms={terms}", flush=True)
            scores, info = train_source_ovr(
                arrays,
                terms,
                layers,
                df,
                y,
                q_all,
                subtrain_idx,
                val_idx,
                test_idx,
                feature_rows,
                pre_rows,
                source_name=source_name,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                seed=args.seed + split_id,
            )
            source_scores[source_name] = scores
            source_info.append({"split_id": split_id, **info})
            add_subset_metrics(metric_rows, df, test_idx, scores, split_id=split_id, source_name=source_name)
            if args.save_predictions:
                pred_frames.append(make_predictions(df, test_idx, scores, split_id=split_id, source_name=source_name))
        print("  Training source=identity_pairwise", flush=True)
        pair_scores, info = train_identity_pairwise(
            arrays,
            layers,
            df,
            q_all,
            subtrain_idx,
            val_idx,
            test_idx,
            feature_rows,
            pre_rows,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed + split_id,
        )
        source_scores["identity_pairwise"] = pair_scores
        source_info.append({"split_id": split_id, **info})
        add_subset_metrics(metric_rows, df, test_idx, pair_scores, split_id=split_id, source_name="identity_pairwise")
        if args.save_predictions:
            pred_frames.append(make_predictions(df, test_idx, pair_scores, split_id=split_id, source_name="identity_pairwise"))

        fused = fixed_fusion(source_scores, FUSION_SOURCES)
        add_subset_metrics(metric_rows, df, test_idx, fused, split_id=split_id, source_name="fixed_temporal_identity_pair_fusion")
        if args.save_predictions:
            pred_frames.append(make_predictions(df, test_idx, fused, split_id=split_id, source_name="fixed_temporal_identity_pair_fusion"))
        fused_temporal = fixed_fusion(source_scores, TEMPORAL_OVR_FUSION_SOURCES)
        add_subset_metrics(metric_rows, df, test_idx, fused_temporal, split_id=split_id, source_name="fixed_temporal_ovr3_fusion")
        if args.save_predictions:
            pred_frames.append(make_predictions(df, test_idx, fused_temporal, split_id=split_id, source_name="fixed_temporal_ovr3_fusion"))
        pd.DataFrame(metric_rows).to_csv(output_dir / "nested_fixed_fusion_metrics.partial.csv", index=False)
        pd.DataFrame(source_info).to_csv(output_dir / "nested_fixed_fusion_source_info.partial.csv", index=False)
        print(pd.DataFrame(metric_rows).tail(5).to_string(index=False), flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "nested_fixed_fusion_metrics.csv", index=False)
    pd.DataFrame(source_info).to_csv(output_dir / "nested_fixed_fusion_source_info.csv", index=False)
    if args.save_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "nested_fixed_fusion_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby("source_name", as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "nested_fixed_fusion_summary.csv", index=False)
    report = [
        "# Nested Fixed Fusion Validation",
        "",
        "## Scope",
        "",
        "- Component linear probes are retrained inside each outer trajectory split.",
        "- Fusion rules are fixed before this script runs: row-softmax t=1 averaged across the temporal+pairwise source set and across the temporal-only OVR source set.",
        "- Inputs are activation-only cached features; metadata is used only for indexing, splitting, labels, and reports.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows.",
    ]
    (output_dir / "nested_fixed_fusion_report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {output_dir / 'nested_fixed_fusion_metrics.csv'}")
    print(f"Wrote {output_dir / 'nested_fixed_fusion_summary.csv'}")
    print(summary.to_string(index=False) if not summary.empty else "No summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
