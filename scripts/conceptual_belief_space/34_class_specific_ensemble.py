#!/usr/bin/env python
"""Evaluate class-specific reliability ensembles over top-action base models.

This post-processor uses only validation predictions to estimate how reliable
each base model is for each candidate action. The resulting action-specific
weights are then applied to held-out test predictions. Test labels are never
used to choose weights or Viterbi transition strengths.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown


PROB_COLS = [f"prob_{action}" for action in ACTIONS]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: str
    feature_set: str | None = None


MODEL_SPECS = [
    ModelSpec("top_meta", "top_action_valpred_top_meta_full"),
    ModelSpec("q05_meta", "top_action_valpred_q05_meta_full"),
    ModelSpec("agent03_q03_meta", "top_action_valpred_agent03_q03_meta_full"),
    ModelSpec("mean_all_tfm", "top_action_valpred_mean_all_tfm_full"),
    ModelSpec("mean_all_bigru_h256", "top_action_valpred_mean_all_bigru_full"),
    ModelSpec("mean_late_bigru", "top_action_valpred_mean_late_bigru_full"),
    ModelSpec("mix_all_bigru", "top_action_valpred_mix_all_bigru_full"),
    ModelSpec("mean_all_bigru_h512_raw", "top_action_valpred_mean_all_bigru_h512_full"),
    ModelSpec("mean_all_bigru_h512_delta", "top_action_valpred_mean_all_bigru_h512_delta_pre_full"),
    ModelSpec("mean_all_bigru_h512_qonly", "top_action_valpred_mean_all_bigru_h512_qonly_agent03_full"),
    ModelSpec("layer_10_12_14_16_18_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_10_12_14_16_18"),
    ModelSpec("layer_08_10_12_14_16_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_08_10_12_14_16"),
    ModelSpec("layer_10_12_14_16_18_20_h512_qonly", "top_action_valpred_layer_10_20_h512_qonly_full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_class_specific_ensemble")
    parser.add_argument("--transition_weights", default="0,0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.35,0.5,0.75,1.0")
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def load_predictions(base_dir: Path, spec: ModelSpec, split_role: str) -> pd.DataFrame:
    path = base_dir / spec.directory / f"{split_role}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if spec.feature_set is not None:
        df = df[df["feature_set"] == spec.feature_set].copy()
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[needed].copy()
    if df.empty:
        raise ValueError(f"empty predictions for {spec.name} from {path}")
    df["row_id"] = df["row_id"].astype(np.int64)
    df["split_id"] = df["split_id"].astype(int)
    return df


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    mapping = {action: i for i, action in enumerate(ACTIONS)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def normalize_probs(probs: np.ndarray, eps: float) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, None)
    return probs / probs.sum(axis=1, keepdims=True)


def prediction_frame_for_split(
    model_frames: dict[str, pd.DataFrame],
    split_id: int,
    metadata: pd.DataFrame,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    split_frames = {}
    common: set[int] | None = None
    for name, df in model_frames.items():
        part = df[df["split_id"] == split_id].copy()
        rows = set(part["row_id"].astype(int))
        common = rows if common is None else common & rows
        split_frames[name] = part
    if not common:
        raise ValueError(f"no common rows for split {split_id}")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    rows = pd.DataFrame({"row_id": row_ids}).merge(metadata, on="row_id", how="left", validate="one_to_one")
    if rows[["decision_id", "prefix_token_index", "top_action"]].isna().any().any():
        raise ValueError(f"metadata missing for split {split_id}")
    probs = {}
    true_label = None
    for name, part in split_frames.items():
        part = part[part["row_id"].isin(row_ids)].sort_values("row_id")
        if len(part) != len(row_ids):
            raise ValueError(f"{name} has duplicate/missing rows on split {split_id}")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"{name} row order mismatch on split {split_id}")
        labels = part["true_top_action"].to_numpy()
        if true_label is None:
            true_label = labels
        elif not np.array_equal(true_label, labels):
            raise ValueError(f"true labels mismatch for {name} split {split_id}")
        probs[name] = normalize_probs(part[PROB_COLS].to_numpy(dtype=np.float64), eps)
    rows["true_top_action"] = true_label
    return rows, probs


def confusion_json(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    mat = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return json.dumps(mat.tolist())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_json": confusion_json(y_true, y_pred),
    }


def viterbi_decode(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float, eps: float) -> np.ndarray:
    if transition_weight <= 0:
        return probs.argmax(axis=1)
    ordered = rows[["decision_id", "prefix_token_index"]].copy()
    ordered["_pos"] = np.arange(len(rows), dtype=np.int64)
    ordered = ordered.sort_values(["decision_id", "prefix_token_index", "_pos"])
    pred = np.zeros(len(rows), dtype=np.int64)
    logp = np.log(normalize_probs(probs, eps))
    eye = np.eye(len(ACTIONS), dtype=np.float64)
    for _, group in ordered.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        back = np.zeros_like(emit, dtype=np.int64)
        dp = emit[0].copy()
        for t in range(1, len(pos)):
            score = dp[:, None] + transition_weight * eye
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def class_specific_weights(
    val_probs: dict[str, np.ndarray],
    val_rows: pd.DataFrame,
    names: list[str],
    kind: str,
    eps: float,
    top_k: int | None = None,
) -> np.ndarray:
    y_true = labels_to_idx(val_rows["true_top_action"])
    stack = np.stack([val_probs[name] for name in names], axis=0)
    pred = stack.argmax(axis=2)
    weights = np.ones((len(names), len(ACTIONS)), dtype=np.float64)
    for action_idx in range(len(ACTIONS)):
        true_mask = y_true == action_idx
        for model_idx in range(len(names)):
            if kind == "class_recall_prob":
                weights[model_idx, action_idx] = float(stack[model_idx, true_mask, action_idx].mean()) if true_mask.any() else 1.0
            elif kind == "class_recall_acc":
                weights[model_idx, action_idx] = (float(np.mean(pred[model_idx, true_mask] == action_idx)) if true_mask.any() else 0.0) + eps
            elif kind == "class_precision":
                pred_mask = pred[model_idx] == action_idx
                weights[model_idx, action_idx] = (float(np.mean(y_true[pred_mask] == action_idx)) if pred_mask.any() else 0.0) + eps
            elif kind == "class_f1":
                pred_mask = pred[model_idx] == action_idx
                recall = float(np.mean(pred[model_idx, true_mask] == action_idx)) if true_mask.any() else 0.0
                precision = float(np.mean(y_true[pred_mask] == action_idx)) if pred_mask.any() else 0.0
                weights[model_idx, action_idx] = (2.0 * recall * precision / (recall + precision + eps)) + eps
            elif kind == "class_nll_inv":
                weights[model_idx, action_idx] = 1.0 / (-np.log(np.clip(stack[model_idx, true_mask, action_idx], eps, 1.0)).mean() + eps) if true_mask.any() else 1.0
            else:
                raise ValueError(f"unknown weight kind: {kind}")
    if top_k is not None:
        top_k = max(1, min(int(top_k), len(names)))
        sparse = np.zeros_like(weights)
        for action_idx in range(len(ACTIONS)):
            keep = np.argsort(weights[:, action_idx])[-top_k:]
            sparse[keep, action_idx] = weights[keep, action_idx]
        weights = sparse
    weights = np.clip(weights, eps, None)
    return weights / weights.sum(axis=0, keepdims=True)


def apply_class_weights(probs_by_model: dict[str, np.ndarray], names: list[str], weights: np.ndarray, eps: float) -> np.ndarray:
    stack = np.stack([probs_by_model[name] for name in names], axis=0)
    out = np.sum(stack * weights[:, None, :], axis=0)
    return normalize_probs(out, eps)


def choose_transition_weight(rows: pd.DataFrame, probs: np.ndarray, grid: list[float], eps: float) -> tuple[float, float]:
    y_true = labels_to_idx(rows["true_top_action"])
    best_tw = 0.0
    best_bal = -np.inf
    for tw in grid:
        pred = viterbi_decode(rows, probs, tw, eps)
        bal = float(metrics(y_true, pred)["balanced_accuracy"])
        if bal > best_bal:
            best_bal = bal
            best_tw = float(tw)
    return best_tw, best_bal


def evaluate_probs(
    rows: pd.DataFrame,
    probs: np.ndarray,
    *,
    method: str,
    split_id: int,
    weight_kind: str,
    models: list[str],
    transition_weight: float,
    validation_balanced_accuracy: float,
    weights: np.ndarray,
    eps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_true = labels_to_idx(rows["true_top_action"])
    y_pred = viterbi_decode(rows, probs, transition_weight, eps)
    row = {
        "method": method,
        "split_id": int(split_id),
        "weight_kind": weight_kind,
        "n": int(len(rows)),
        **metrics(y_true, y_pred),
        "validation_balanced_accuracy": float(validation_balanced_accuracy),
        "models_json": json.dumps(models),
        "transition_weight": float(transition_weight),
        "weights_json": json.dumps(weights.tolist()),
    }
    for i, action in enumerate(ACTIONS):
        row[f"true_count_{action}"] = int(np.sum(y_true == i))
        row[f"pred_count_{action}"] = int(np.sum(y_pred == i))
    pred_df = rows[["row_id", "decision_id", "prefix_token_index", "true_top_action"]].copy()
    pred_df.insert(0, "split_id", int(split_id))
    pred_df.insert(0, "method", method)
    pred_df["pred_top_action"] = idx_to_labels(y_pred)
    for i, action in enumerate(ACTIONS):
        pred_df[f"prob_{action}"] = probs[:, i]
    return row, pred_df


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = ensure_dir(args.output_dir)
    transition_grid = [float(x) for x in args.transition_weights.split(",") if x.strip()]
    metadata = pd.read_parquet(args.beliefs, columns=["row_id", "decision_id", "trajectory_id", "prefix_token_index", "top_action"])
    metadata["row_id"] = metadata["row_id"].astype(np.int64)

    test_frames = {spec.name: load_predictions(base_dir, spec, "test") for spec in MODEL_SPECS}
    val_frames = {spec.name: load_predictions(base_dir, spec, "validation") for spec in MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))
    names = [spec.name for spec in MODEL_SPECS]
    method_specs: list[tuple[str, str, int | None]] = [
        ("class_recall_prob_viterbi_val_tw", "class_recall_prob", None),
        ("class_recall_acc_viterbi_val_tw", "class_recall_acc", None),
        ("class_precision_viterbi_val_tw", "class_precision", None),
        ("class_f1_viterbi_val_tw", "class_f1", None),
        ("class_nll_inv_viterbi_val_tw", "class_nll_inv", None),
    ]
    for top_k in [3, 5, 7, 9, 11, 13]:
        method_specs.append((f"class_recall_prob_top{top_k}_viterbi_val_tw", "class_recall_prob", top_k))

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        test_rows, test_probs = prediction_frame_for_split(test_frames, split_id, metadata, args.eps)
        val_rows, val_probs = prediction_frame_for_split(val_frames, split_id, metadata, args.eps)
        for method, kind, top_k in method_specs:
            weights = class_specific_weights(val_probs, val_rows, names, kind, args.eps, top_k=top_k)
            val_weighted = apply_class_weights(val_probs, names, weights, args.eps)
            test_weighted = apply_class_weights(test_probs, names, weights, args.eps)
            tw, val_bal = choose_transition_weight(val_rows, val_weighted, transition_grid, args.eps)
            row, pred = evaluate_probs(
                test_rows,
                test_weighted,
                method=method,
                split_id=split_id,
                weight_kind=kind,
                models=names,
                transition_weight=tw,
                validation_balanced_accuracy=val_bal,
                weights=weights,
                eps=args.eps,
            )
            row["top_k"] = int(top_k) if top_k is not None else np.nan
            metric_rows.append(row)
            pred_frames.append(pred)
            print(f"RESULT split={split_id} method={method} bal={row['balanced_accuracy']:.6f} acc={row['accuracy']:.6f} tw={tw}", flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    summary = (
        metrics_df.groupby("method", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    predictions = pd.concat(pred_frames, ignore_index=True)
    selected_by_validation = summary.sort_values(
        ["mean_validation_balanced_accuracy", "mean_balanced_accuracy"], ascending=[False, False]
    ).head(1)
    metrics_df.to_csv(output_dir / "class_specific_ensemble_metrics.csv", index=False)
    summary.to_csv(output_dir / "class_specific_ensemble_summary.csv", index=False)
    selected_by_validation.to_csv(output_dir / "class_specific_ensemble_validation_selected_summary.csv", index=False)
    predictions.to_parquet(output_dir / "class_specific_ensemble_predictions.parquet", index=False)
    lines = [
        "# Class-Specific Top-Action Ensemble",
        "",
        "Each split estimates model/action reliability weights from validation predictions only, then applies those weights to held-out test predictions.",
        "Viterbi transition weight is also selected on validation predictions only.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Validation-Selected Method",
        "",
        "This table selects the method with highest mean validation balanced accuracy, not the best test score.",
        "",
        selected_by_validation.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics_df.to_markdown(index=False),
    ]
    save_markdown(output_dir / "class_specific_ensemble_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(output_dir / "class_specific_ensemble_summary.csv"), "metrics": str(output_dir / "class_specific_ensemble_metrics.csv"), "predictions": str(output_dir / "class_specific_ensemble_predictions.parquet")}, indent=2))


if __name__ == "__main__":
    main()
