#!/usr/bin/env python3
"""Apply train-prior debiasing to nested activation-only source-score fusions.

Input source scores should be out-of-fold predictions from
47_nested_fixed_fusion_validation.py.  For each outer split, this script computes
a class-prior bias from the other outer splits' labels and adds

    -alpha * log(train_prior(action))

to the fused action log scores.  The model inputs remain activation-only; labels
and metadata are used only for train-prior calculation and evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conceptual_belief_space.config import ACTIONS

Q_COLS = [f"q_{a}" for a in ACTIONS]
SCORE_COLS = [f"score_{a}" for a in ACTIONS]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SOURCE_GROUPS = {
    "ovr3": ["mean100_ovr", "prefixmean_ovr", "segment_ovr"],
    "all4": ["mean100_ovr", "identity_pairwise", "prefixmean_ovr", "segment_ovr"],
}
DEFAULT_CONFIGS = [
    {"config_name": "all4_t1p5_prior_alpha0p1", "group": "all4", "temperature": 1.5, "alpha": 0.1},
    {"config_name": "all4_t0p75_prior_alpha0p25", "group": "all4", "temperature": 0.75, "alpha": 0.25},
    {"config_name": "all4_t1p0_prior_alpha0p1", "group": "all4", "temperature": 1.0, "alpha": 0.1},
    {"config_name": "ovr3_t0p75_prior_alpha0p1", "group": "ovr3", "temperature": 0.75, "alpha": 0.1},
    {"config_name": "ovr3_t1p5_prior_alpha0p1", "group": "ovr3", "temperature": 1.5, "alpha": 0.1},
]


def row_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    z = centered / scale / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def action_indices(series: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return series.astype(str).map(mapping).to_numpy(dtype=np.int64)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(len(ACTIONS)):
        mask = y_true == i
        if mask.any():
            vals.append(float((y_pred[mask] == i).mean()))
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
    correct = total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def metric_row(df: pd.DataFrame, mask: np.ndarray, scores: np.ndarray, *, config: dict[str, Any], split_id: int, eval_subset: str) -> dict[str, Any]:
    sub = df.loc[mask]
    local_scores = scores[mask]
    y = action_indices(sub["top_action"])
    pred = local_scores.argmax(axis=1)
    q = sub[Q_COLS].to_numpy(dtype=np.float64)
    row = {
        "config_name": config["config_name"],
        "group": config["group"],
        "temperature": float(config["temperature"]),
        "alpha": float(config["alpha"]),
        "split_id": int(split_id),
        "eval_subset": eval_subset,
        "n_rows": int(mask.sum()),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "pairwise_accuracy": pairwise_accuracy(local_scores, q),
    }
    for i, action in enumerate(ACTIONS):
        am = y == i
        row[f"recall_{action}"] = float((pred[am] == i).mean()) if am.any() else float("nan")
    return row


def load_source_scores(predictions: pd.DataFrame, source_names: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base = predictions[predictions["source_name"].isin(source_names)].copy()
    if base.empty:
        raise ValueError(f"No requested source rows found: {source_names}")
    meta_cols = [c for c in ["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action", "split_id", *Q_COLS, *SUBSET_COLS] if c in base.columns]
    meta = base[meta_cols].drop_duplicates("row_id").sort_values("row_id").reset_index(drop=True)
    scores = {}
    for source in source_names:
        sub = base[base["source_name"].eq(source)].set_index("row_id").loc[meta["row_id"], SCORE_COLS]
        if len(sub) != len(meta):
            raise ValueError(f"Source {source} is missing aligned rows")
        scores[source] = sub.to_numpy(dtype=np.float64)
    return meta, scores


def fused_scores(source_scores: dict[str, np.ndarray], source_names: list[str], temperature: float) -> np.ndarray:
    probs = np.stack([row_softmax(source_scores[name], temperature) for name in source_names], axis=0).mean(axis=0)
    return np.log(np.clip(probs, 1e-12, None))


def apply_train_prior_bias(meta: pd.DataFrame, base_scores: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[int, list[float]]]:
    y = action_indices(meta["top_action"])
    splits = meta["split_id"].to_numpy()
    out = base_scores.copy()
    bias_by_split: dict[int, list[float]] = {}
    for split_id in sorted(pd.unique(meta["split_id"])):
        train = splits != split_id
        test = splits == split_id
        counts = np.bincount(y[train], minlength=len(ACTIONS)).astype(np.float64)
        prior = counts / np.clip(counts.sum(), 1.0, None)
        bias = -float(alpha) * np.log(np.clip(prior, 1e-12, None))
        bias = bias - bias.mean()
        out[test] = out[test] + bias[None, :]
        bias_by_split[int(split_id)] = bias.tolist()
    return out, bias_by_split


def parse_configs(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return list(DEFAULT_CONFIGS)
    configs = []
    for item in text.split(";"):
        if not item.strip():
            continue
        # name,group,temp,alpha
        name, group, temp, alpha = [x.strip() for x in item.split(",")]
        configs.append({"config_name": name, "group": group, "temperature": float(temp), "alpha": float(alpha)})
    return configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="outputs/conceptual_belief_space/activation_nested_fixed_fusion_validation_fulltrain/nested_fixed_fusion_predictions.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_train_prior_debiased_fusion")
    p.add_argument("--configs", default=None, help="Optional semicolon list: name,group,temp,alpha;...")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    all_source_names = sorted({name for cfg in configs for name in SOURCE_GROUPS[cfg["group"]]})
    predictions = pd.read_parquet(args.predictions)
    meta, source_scores = load_source_scores(predictions, all_source_names)

    metric_rows = []
    pred_frames = []
    manifest = {
        "input_predictions": str(args.predictions),
        "activation_only_source_scores": True,
        "note": "Train-prior bias is computed per split from labels outside the held-out split. Configs are candidate fixed rules, not selected on held-out rows.",
        "configs": configs,
    }
    bias_manifest: dict[str, Any] = {}
    for config in configs:
        names = SOURCE_GROUPS[config["group"]]
        base = fused_scores(source_scores, names, float(config["temperature"]))
        scores, bias_by_split = apply_train_prior_bias(meta, base, float(config["alpha"]))
        bias_manifest[config["config_name"]] = bias_by_split
        split_values = sorted(pd.unique(meta["split_id"]))
        for split_id in split_values:
            split_mask = meta["split_id"].to_numpy() == split_id
            metric_rows.append(metric_row(meta, split_mask, scores, config=config, split_id=int(split_id), eval_subset="all_prefixes"))
            for col in SUBSET_COLS:
                if col in meta.columns:
                    mask = split_mask & meta[col].fillna(False).astype(bool).to_numpy()
                    if mask.any():
                        metric_rows.append(metric_row(meta, mask, scores, config=config, split_id=int(split_id), eval_subset=col))
        pred = meta.copy()
        for i, action in enumerate(ACTIONS):
            pred[f"score_{action}"] = scores[:, i]
        pred["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[scores.argmax(axis=1)]
        pred["config_name"] = config["config_name"]
        pred["group"] = config["group"]
        pred["temperature"] = float(config["temperature"])
        pred["alpha"] = float(config["alpha"])
        pred_frames.append(pred)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "train_prior_debiased_fusion_metrics.csv", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby(["config_name", "group", "temperature", "alpha"], as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "train_prior_debiased_fusion_summary.csv", index=False)

    selection_rows = []
    for held_split in sorted(pd.unique(all_rows["split_id"])):
        train_rows = all_rows[all_rows["split_id"].ne(held_split)]
        means = train_rows.groupby("config_name")["balanced_accuracy"].mean().sort_values(ascending=False)
        if means.empty:
            continue
        selected_name = means.index[0]
        test_row = all_rows[(all_rows["split_id"].eq(held_split)) & (all_rows["config_name"].eq(selected_name))].iloc[0]
        selection_rows.append({
            "held_split": int(held_split),
            "selected_config": selected_name,
            "train_mean_balanced_accuracy": float(means.iloc[0]),
            "test_balanced_accuracy": float(test_row["balanced_accuracy"]),
            "test_accuracy": float(test_row["accuracy"]),
        })
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output_dir / "train_prior_debiased_fusion_leave_one_split_selection.csv", index=False)

    pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "train_prior_debiased_fusion_predictions.parquet", index=False)
    manifest["bias_by_split"] = bias_manifest
    (output_dir / "train_prior_debiased_fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Train-Prior Debiased Activation Fusion",
        "",
        "## Scope",
        "",
        "- Source scores come from nested activation-only linear probes.",
        "- Per-split class-prior bias is computed from non-held-out split labels only.",
        "- No grid/task features or prefix metadata are model inputs.",
        "- These are candidate fixed rules; configs should be treated cautiously if chosen after exploratory inspection.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows.",
        "",
        "## Leave-One-Split Config Selection Audit",
        "",
        selection.to_markdown(index=False) if not selection.empty else "No selection rows.",
    ]
    (output_dir / "train_prior_debiased_fusion_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
