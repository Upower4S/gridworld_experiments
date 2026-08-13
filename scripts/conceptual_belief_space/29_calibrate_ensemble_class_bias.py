#!/usr/bin/env python
"""Validation-only class-bias calibration for top-action ensembles."""

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


PROB_COLS = [f"prob_{a}" for a in ACTIONS]


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_calibrated_class_bias_ensemble")
    p.add_argument("--n_random", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bias_scale", type=float, default=1.5)
    p.add_argument("--temperature_grid", default="0.45,0.6,0.75,0.9,1.0,1.15,1.35,1.6,2.0")
    p.add_argument("--transition_weights", default="0,0.05,0.1,0.15,0.25,0.4,0.6,0.8,1.0")
    return p.parse_args()


def load_predictions(base_dir: Path, spec: ModelSpec, split_role: str) -> pd.DataFrame:
    path = base_dir / spec.directory / f"{split_role}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if spec.feature_set is not None:
        df = df[df["feature_set"] == spec.feature_set].copy()
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[needed].copy()
    df["row_id"] = df["row_id"].astype(np.int64)
    df["split_id"] = df["split_id"].astype(int)
    if df.empty:
        raise ValueError(f"empty predictions for {spec.name}")
    return df


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    m = {a: i for i, a in enumerate(ACTIONS)}
    return np.asarray([m[str(x)] for x in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    mat = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_json": json.dumps(mat.tolist()),
    }


def split_table(
    model_frames: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    split_id: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    common: set[int] | None = None
    parts = {}
    for name, df in model_frames.items():
        part = df[df["split_id"] == split_id].copy()
        rows = set(part["row_id"].astype(int))
        common = rows if common is None else common & rows
        parts[name] = part
    if not common:
        raise ValueError(f"no common rows for split {split_id}")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    rows_df = pd.DataFrame({"row_id": row_ids}).merge(metadata, on="row_id", how="left", validate="one_to_one")
    if rows_df[["decision_id", "prefix_token_index", "top_action"]].isna().any().any():
        raise ValueError("metadata merge failed")
    probs = {}
    true = None
    for name, part in parts.items():
        part = part[part["row_id"].isin(row_ids)].sort_values("row_id")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"row mismatch for {name}")
        labels = part["true_top_action"].to_numpy()
        if true is None:
            true = labels
        elif not np.array_equal(true, labels):
            raise ValueError(f"label mismatch for {name}")
        arr = part[PROB_COLS].to_numpy(dtype=np.float64)
        arr = np.clip(arr, 1e-12, None)
        probs[name] = arr / arr.sum(axis=1, keepdims=True)
    rows_df["true_top_action"] = true
    return rows_df, probs


def average_probs(probs: dict[str, np.ndarray], names: list[str], weights: np.ndarray | None = None) -> np.ndarray:
    stack = np.stack([probs[n] for n in names], axis=0)
    if weights is None:
        out = stack.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
        out = np.tensordot(w, stack, axes=(0, 0))
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def validation_model_weights(probs: dict[str, np.ndarray], y_true: np.ndarray, names: list[str]) -> np.ndarray:
    return np.asarray([metrics_np(y_true, probs[name].argmax(axis=1))["balanced_accuracy"] for name in names], dtype=np.float64)


def calibrate_probs(probs: np.ndarray, bias: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None)) / max(float(temperature), 1e-6)
    logits = logits + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def viterbi(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float) -> np.ndarray:
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    logp = np.log(probs)
    ordered = rows[["decision_id", "prefix_token_index"]].copy()
    ordered["_pos"] = np.arange(len(rows), dtype=np.int64)
    ordered = ordered.sort_values(["decision_id", "prefix_token_index", "_pos"])
    pred = np.zeros(len(rows), dtype=np.int64)
    stay = float(transition_weight) * np.eye(len(ACTIONS), dtype=np.float64)
    for _, group in ordered.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        dp = emit[0].copy()
        back = np.zeros_like(emit, dtype=np.int64)
        for t in range(1, len(pos)):
            score = dp[:, None] + stay
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def best_transition(rows: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, grid: list[float]) -> tuple[float, float]:
    best_tw = 0.0
    best_bal = -np.inf
    for tw in grid:
        pred = probs.argmax(axis=1) if tw == 0 else viterbi(rows, probs, tw)
        bal = metrics_np(y_true, pred)["balanced_accuracy"]
        if bal > best_bal:
            best_bal = float(bal)
            best_tw = float(tw)
    return best_tw, best_bal


def search_calibration(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    rng: np.random.Generator,
    n_random: int,
    bias_scale: float,
    temperatures: list[float],
) -> tuple[np.ndarray, float, float]:
    candidates: list[tuple[np.ndarray, float]] = []
    candidates.append((np.zeros(len(ACTIONS), dtype=np.float64), 1.0))
    # Empirical prior correction is a strong deterministic baseline.
    pred_counts = np.bincount(probs.argmax(axis=1), minlength=len(ACTIONS)).astype(np.float64) + 1.0
    true_counts = np.bincount(y_true, minlength=len(ACTIONS)).astype(np.float64) + 1.0
    prior_bias = np.log(true_counts / true_counts.sum()) - np.log(pred_counts / pred_counts.sum())
    prior_bias = prior_bias - prior_bias.mean()
    for temp in temperatures:
        candidates.append((prior_bias.copy(), temp))
    for _ in range(n_random):
        bias = rng.normal(0.0, bias_scale, size=len(ACTIONS))
        bias = np.clip(bias, -2.5 * bias_scale, 2.5 * bias_scale)
        bias = bias - bias.mean()
        temp = float(rng.choice(temperatures))
        candidates.append((bias.astype(np.float64), temp))
    best_bias = np.zeros(len(ACTIONS), dtype=np.float64)
    best_temp = 1.0
    best_bal = -np.inf
    for bias, temp in candidates:
        pred = calibrate_probs(probs, bias, temp).argmax(axis=1)
        bal = metrics_np(y_true, pred)["balanced_accuracy"]
        if bal > best_bal:
            best_bal = float(bal)
            best_bias = bias.copy()
            best_temp = float(temp)
    # Coordinate refinement around the best random candidate.
    for step in [0.5, 0.25, 0.1, 0.05]:
        improved = True
        while improved:
            improved = False
            for k in range(len(ACTIONS)):
                for sign in [-1.0, 1.0]:
                    bias = best_bias.copy()
                    bias[k] += sign * step
                    bias = bias - bias.mean()
                    pred = calibrate_probs(probs, bias, best_temp).argmax(axis=1)
                    bal = metrics_np(y_true, pred)["balanced_accuracy"]
                    if bal > best_bal:
                        best_bal = float(bal)
                        best_bias = bias
                        improved = True
            for temp in temperatures:
                pred = calibrate_probs(probs, best_bias, temp).argmax(axis=1)
                bal = metrics_np(y_true, pred)["balanced_accuracy"]
                if bal > best_bal:
                    best_bal = float(bal)
                    best_temp = float(temp)
                    improved = True
    return best_bias, best_temp, best_bal


def eval_one(rows: pd.DataFrame, probs: np.ndarray, method: str, split_id: int, models: list[str], transition_weight: float) -> tuple[dict[str, object], pd.DataFrame]:
    y_true = labels_to_idx(rows["true_top_action"])
    pred = probs.argmax(axis=1) if transition_weight == 0 else viterbi(rows, probs, transition_weight)
    m = metrics_np(y_true, pred)
    out = {
        "method": method,
        "split_id": int(split_id),
        "n": int(len(rows)),
        **m,
        "transition_weight": float(transition_weight),
        "models_json": json.dumps(models),
    }
    pred_df = pd.DataFrame(
        {
            "method": method,
            "split_id": int(split_id),
            "row_id": rows["row_id"].to_numpy(dtype=np.int64),
            "true_top_action": rows["true_top_action"].to_numpy(),
            "pred_top_action": idx_to_labels(pred),
        }
    )
    for i, a in enumerate(ACTIONS):
        pred_df[f"prob_{a}"] = probs[:, i]
    return out, pred_df


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    base_dir = Path(args.base_dir)
    metadata = pd.read_parquet(args.beliefs, columns=["row_id", "decision_id", "trajectory_id", "prefix_token_index", "top_action"])
    metadata["row_id"] = metadata["row_id"].astype(np.int64)
    test_frames = {s.name: load_predictions(base_dir, s, "test") for s in MODEL_SPECS}
    val_frames = {s.name: load_predictions(base_dir, s, "validation") for s in MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))
    names = [s.name for s in MODEL_SPECS]
    temps = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    tw_grid = [float(x) for x in args.transition_weights.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        test_rows, test_probs_by_model = split_table(test_frames, metadata, split_id)
        val_rows, val_probs_by_model = split_table(val_frames, metadata, split_id)
        y_val = labels_to_idx(val_rows["true_top_action"])
        for base_method, weights in [
            ("uniform13", None),
            ("val_weighted13", validation_model_weights(val_probs_by_model, y_val, names)),
        ]:
            val_avg = average_probs(val_probs_by_model, names, weights)
            test_avg = average_probs(test_probs_by_model, names, weights)
            bias, temp, val_cal_bal = search_calibration(
                val_rows,
                y_val,
                val_avg,
                rng=rng,
                n_random=args.n_random,
                bias_scale=args.bias_scale,
                temperatures=temps,
            )
            val_cal = calibrate_probs(val_avg, bias, temp)
            test_cal = calibrate_probs(test_avg, bias, temp)
            tw, val_tw_bal = best_transition(val_rows, y_val, val_cal, tw_grid)
            for suffix, probs, tw_used, val_score in [
                ("calibrated", test_cal, 0.0, val_cal_bal),
                ("calibrated_viterbi_val_tw", test_cal, tw, val_tw_bal),
            ]:
                method = f"{base_method}_{suffix}"
                row, pred_df = eval_one(test_rows, probs, method, split_id, names, tw_used)
                row["validation_balanced_accuracy"] = float(val_score)
                row["temperature"] = float(temp)
                row["bias_json"] = json.dumps([float(x) for x in bias])
                row["weights_json"] = "" if weights is None else json.dumps([float(x) for x in weights])
                metric_rows.append(row)
                pred_frames.append(pred_df)
                print(
                    f"RESULT split={split_id} method={method} "
                    f"bal={row['balanced_accuracy']:.6f} acc={row['accuracy']:.6f} "
                    f"val={val_score:.6f} temp={temp:.3f} tw={tw_used}",
                    flush=True,
                )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    metrics.to_csv(out_dir / "calibrated_class_bias_metrics.csv", index=False)
    summary.to_csv(out_dir / "calibrated_class_bias_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(out_dir / "calibrated_class_bias_predictions.parquet", index=False)
    lines = [
        "# Class-Bias Calibrated Ensemble",
        "",
        "Biases and temperatures are selected using validation predictions only. Held-out test predictions are used only for reported scores.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    save_markdown(out_dir / "calibrated_class_bias_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(out_dir / "calibrated_class_bias_summary.csv"), "metrics": str(out_dir / "calibrated_class_bias_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
