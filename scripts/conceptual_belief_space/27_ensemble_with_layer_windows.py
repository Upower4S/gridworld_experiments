#!/usr/bin/env python
"""Evaluate top-action ensembles after adding layer-window BiGRU models."""

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
    ModelSpec(
        "layer_10_12_14_16_18_h512_qonly",
        "top_action_valpred_layer_windows_h512_qonly_full",
        "mean_10_12_14_16_18",
    ),
    ModelSpec(
        "layer_08_10_12_14_16_h512_qonly",
        "top_action_valpred_layer_windows_h512_qonly_full",
        "mean_08_10_12_14_16",
    ),
    ModelSpec("layer_10_12_14_16_18_20_h512_qonly", "top_action_valpred_layer_10_20_h512_qonly_full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_layer_window_ensemble_update")
    parser.add_argument("--transition_weights", default="0,0.05,0.1,0.15,0.25,0.4,0.6")
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def load_predictions(base_dir: Path, spec: ModelSpec, split_role: str) -> pd.DataFrame:
    path = base_dir / spec.directory / f"{split_role}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if spec.feature_set is not None:
        df = df[df["feature_set"] == spec.feature_set].copy()
    if df.empty:
        raise ValueError(f"no rows for {spec.name} from {path}")
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[needed].copy()
    df["row_id"] = df["row_id"].astype(np.int64)
    df["split_id"] = df["split_id"].astype(int)
    return df


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    mapping = {action: i for i, action in enumerate(ACTIONS)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


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


def prediction_frame_for_split(
    model_frames: dict[str, pd.DataFrame],
    split_id: int,
    metadata: pd.DataFrame,
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
    base = pd.DataFrame({"row_id": row_ids})
    base = base.merge(metadata, on="row_id", how="left", validate="one_to_one")
    if base[["decision_id", "prefix_token_index", "top_action"]].isna().any().any():
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
        probs[name] = part[PROB_COLS].to_numpy(dtype=np.float64)
    base["true_top_action"] = true_label
    return base, probs


def normalize_probs(probs: np.ndarray, eps: float) -> np.ndarray:
    probs = np.clip(probs, eps, None)
    return probs / probs.sum(axis=1, keepdims=True)


def average_probs(probs_by_model: dict[str, np.ndarray], names: list[str], weights: np.ndarray | None, eps: float) -> np.ndarray:
    stack = np.stack([probs_by_model[name] for name in names], axis=0)
    if weights is None:
        out = stack.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(names), dtype=np.float64)
        w = w / w.sum()
        out = np.tensordot(w, stack, axes=(0, 0))
    return normalize_probs(out, eps)


def viterbi_decode(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float, eps: float) -> np.ndarray:
    ordered = rows[["decision_id", "prefix_token_index"]].copy()
    ordered["_pos"] = np.arange(len(rows), dtype=np.int64)
    ordered = ordered.sort_values(["decision_id", "prefix_token_index", "_pos"])
    pred = np.zeros(len(rows), dtype=np.int64)
    logp = np.log(normalize_probs(probs, eps))
    for _, group in ordered.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        back = np.zeros_like(emit, dtype=np.int64)
        dp = emit[0].copy()
        for t in range(1, len(pos)):
            score = dp[:, None] + transition_weight * np.eye(len(ACTIONS), dtype=np.float64)
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def evaluate_probs(
    rows: pd.DataFrame,
    probs: np.ndarray,
    *,
    method: str,
    split_id: int,
    models: list[str],
    transition_weight: float,
    eps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_true = labels_to_idx(rows["true_top_action"])
    if transition_weight > 0:
        y_pred = viterbi_decode(rows, probs, transition_weight, eps)
    else:
        y_pred = probs.argmax(axis=1)
    row = {
        "method": method,
        "split_id": int(split_id),
        "n": int(len(rows)),
        **metrics(y_true, y_pred),
        "models_json": json.dumps(models),
        "transition_weight": float(transition_weight),
    }
    for i, action in enumerate(ACTIONS):
        row[f"true_count_{action}"] = int(np.sum(y_true == i))
        row[f"pred_count_{action}"] = int(np.sum(y_pred == i))
    pred_df = pd.DataFrame(
        {
            "method": method,
            "split_id": int(split_id),
            "row_id": rows["row_id"].to_numpy(dtype=np.int64),
            "true_top_action": rows["true_top_action"].to_numpy(),
            "pred_top_action": idx_to_labels(y_pred),
        }
    )
    for i, action in enumerate(ACTIONS):
        pred_df[f"prob_{action}"] = probs[:, i]
    return row, pred_df


def validation_model_weights(val_probs: dict[str, np.ndarray], val_rows: pd.DataFrame, names: list[str]) -> np.ndarray:
    y_true = labels_to_idx(val_rows["true_top_action"])
    vals = []
    for name in names:
        pred = val_probs[name].argmax(axis=1)
        vals.append(float(metrics(y_true, pred)["balanced_accuracy"]))
    return np.asarray(vals, dtype=np.float64)


def choose_transition_weight(
    val_rows: pd.DataFrame,
    probs: np.ndarray,
    grid: list[float],
    eps: float,
) -> tuple[float, float]:
    y_true = labels_to_idx(val_rows["true_top_action"])
    best_tw = 0.0
    best_bal = -np.inf
    for tw in grid:
        pred = viterbi_decode(val_rows, probs, tw, eps) if tw > 0 else probs.argmax(axis=1)
        bal = float(metrics(y_true, pred)["balanced_accuracy"])
        if bal > best_bal:
            best_bal = bal
            best_tw = float(tw)
    return best_tw, best_bal


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = ensure_dir(args.output_dir)
    transition_grid = [float(x) for x in args.transition_weights.split(",") if x.strip()]
    metadata = pd.read_parquet(
        args.beliefs,
        columns=["row_id", "decision_id", "trajectory_id", "prefix_token_index", "top_action"],
    )
    metadata["row_id"] = metadata["row_id"].astype(np.int64)

    test_frames = {spec.name: load_predictions(base_dir, spec, "test") for spec in MODEL_SPECS}
    val_frames = {spec.name: load_predictions(base_dir, spec, "validation") for spec in MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))
    names13 = [spec.name for spec in MODEL_SPECS]
    names10 = names13[:-3]
    names12 = names13[:-1]
    layer_names = names13[-3:]
    methods = [
        ("uniform10", names10, "uniform", False),
        ("uniform12", names12, "uniform", False),
        ("val_weighted12", names12, "val_weighted", False),
        ("layer_pair", layer_names, "uniform", False),
        ("uniform13", names13, "uniform", False),
        ("val_weighted13", names13, "val_weighted", False),
        ("uniform12_viterbi_val_tw", names12, "uniform", True),
        ("val_weighted12_viterbi_val_tw", names12, "val_weighted", True),
        ("layer_pair_viterbi_val_tw", layer_names, "uniform", True),
        ("uniform13_viterbi_val_tw", names13, "uniform", True),
        ("val_weighted13_viterbi_val_tw", names13, "val_weighted", True),
    ]

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        test_rows, test_probs = prediction_frame_for_split(test_frames, split_id, metadata)
        val_rows, val_probs = prediction_frame_for_split(val_frames, split_id, metadata)
        for method_name, method_models, weight_mode, use_viterbi in methods:
            weights = None
            if weight_mode == "val_weighted":
                weights = validation_model_weights(val_probs, val_rows, method_models)
            test_avg = average_probs(test_probs, method_models, weights, args.eps)
            val_avg = average_probs(val_probs, method_models, weights, args.eps)
            tw = 0.0
            val_bal = np.nan
            if use_viterbi:
                tw, val_bal = choose_transition_weight(val_rows, val_avg, transition_grid, args.eps)
            row, pred_df = evaluate_probs(
                test_rows,
                test_avg,
                method=method_name,
                split_id=split_id,
                models=method_models,
                transition_weight=tw,
                eps=args.eps,
            )
            row["validation_balanced_accuracy"] = float(val_bal) if np.isfinite(val_bal) else np.nan
            row["weights_json"] = json.dumps(weights.tolist()) if weights is not None else ""
            metric_rows.append(row)
            pred_frames.append(pred_df)
            print(
                f"RESULT split={split_id} method={method_name} "
                f"bal={row['balanced_accuracy']:.6f} acc={row['accuracy']:.6f} tw={tw}",
                flush=True,
            )

    metrics_df = pd.DataFrame(metric_rows)
    summary = (
        metrics_df.groupby("method", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics_df.to_csv(out_dir / "ensemble_layer_window_metrics.csv", index=False)
    summary.to_csv(out_dir / "ensemble_layer_window_summary.csv", index=False)
    predictions.to_parquet(out_dir / "ensemble_layer_window_predictions.parquet", index=False)
    lines = [
        "# Layer-Window Ensemble Update",
        "",
        "Validation predictions select Viterbi transition weights; held-out test predictions are used only for final scores.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics_df.to_markdown(index=False),
    ]
    save_markdown(out_dir / "ensemble_layer_window_report.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "summary": str(out_dir / "ensemble_layer_window_summary.csv"),
                "metrics": str(out_dir / "ensemble_layer_window_metrics.csv"),
                "predictions": str(out_dir / "ensemble_layer_window_predictions.parquet"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
