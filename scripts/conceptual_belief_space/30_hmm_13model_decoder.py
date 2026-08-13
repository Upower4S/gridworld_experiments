#!/usr/bin/env python
"""13-model ensemble HMM decoder with validation-internal hyperparameter selection."""

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
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_hmm_13model_decoder")
    p.add_argument("--alpha_grid", default="0.1,0.5,1,2,5,10,20,50")
    p.add_argument("--temperature_grid", default="0.5,0.65,0.8,1.0,1.25,1.6,2.0")
    p.add_argument("--transition_scale_grid", default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64)


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


def load_predictions(base_dir: Path, spec: ModelSpec, role: str) -> pd.DataFrame:
    path = base_dir / spec.directory / f"{role}_predictions.parquet"
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
    return df


def split_table(model_frames: dict[str, pd.DataFrame], metadata: pd.DataFrame, split_id: int):
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


def average_probs(probs: dict[str, np.ndarray], names: list[str], weights: np.ndarray | None) -> np.ndarray:
    stack = np.stack([probs[n] for n in names], axis=0)
    if weights is None:
        out = stack.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
        out = np.tensordot(w, stack, axes=(0, 0))
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def validation_model_weights(probs: dict[str, np.ndarray], y_true: np.ndarray, names: list[str]) -> np.ndarray:
    return np.asarray([metrics_np(y_true, probs[name].argmax(axis=1))["balanced_accuracy"] for name in names], dtype=np.float64)


def transition_matrix(rows: pd.DataFrame, y: np.ndarray, alpha: float) -> np.ndarray:
    temp = rows[["decision_id", "prefix_token_index"]].copy()
    temp["_pos"] = np.arange(len(rows), dtype=np.int64)
    temp = temp.sort_values(["decision_id", "prefix_token_index", "_pos"])
    counts = np.full((len(ACTIONS), len(ACTIONS)), float(alpha), dtype=np.float64)
    for _, group in temp.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        seq = y[pos]
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a), int(b)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(np.clip(probs, 1e-12, None))


def decode_hmm(rows: pd.DataFrame, probs: np.ndarray, log_trans: np.ndarray, temperature: float, transition_scale: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None)) / max(float(temperature), 1e-6)
    ordered = rows[["decision_id", "prefix_token_index"]].copy()
    ordered["_pos"] = np.arange(len(rows), dtype=np.int64)
    ordered = ordered.sort_values(["decision_id", "prefix_token_index", "_pos"])
    pred = np.zeros(len(rows), dtype=np.int64)
    trans = float(transition_scale) * log_trans
    for _, group in ordered.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logits[pos]
        dp = emit[0].copy()
        back = np.zeros_like(emit, dtype=np.int64)
        for t in range(1, len(pos)):
            score = dp[:, None] + trans
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def inner_validation_split(rows: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    decisions = np.asarray(sorted(rows["decision_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(decisions)
    n_hold = max(1, int(round(0.35 * len(decisions))))
    hold = set(decisions[:n_hold])
    is_hold = rows["decision_id"].astype(str).isin(hold).to_numpy()
    return np.flatnonzero(~is_hold), np.flatnonzero(is_hold)


def select_hparams(
    val_rows: pd.DataFrame,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    *,
    alphas: list[float],
    temperatures: list[float],
    scales: list[float],
    seed: int,
) -> dict[str, float]:
    inner_train, inner_hold = inner_validation_split(val_rows, seed)
    train_rows = val_rows.iloc[inner_train].reset_index(drop=True)
    hold_rows = val_rows.iloc[inner_hold].reset_index(drop=True)
    y_train = y_val[inner_train]
    y_hold = y_val[inner_hold]
    hold_probs = val_probs[inner_hold]
    best = {"alpha": 1.0, "temperature": 1.0, "transition_scale": 0.0, "inner_balanced_accuracy": -np.inf}
    for alpha in alphas:
        log_trans = transition_matrix(train_rows, y_train, alpha)
        for temp in temperatures:
            for scale in scales:
                pred = decode_hmm(hold_rows, hold_probs, log_trans, temp, scale) if scale > 0 else (
                    np.log(np.clip(hold_probs, 1e-12, None)) / temp
                ).argmax(axis=1)
                bal = metrics_np(y_hold, pred)["balanced_accuracy"]
                if bal > best["inner_balanced_accuracy"]:
                    best = {
                        "alpha": float(alpha),
                        "temperature": float(temp),
                        "transition_scale": float(scale),
                        "inner_balanced_accuracy": float(bal),
                    }
    return best


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
    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    temps = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    scales = [float(x) for x in args.transition_scale_grid.split(",") if x.strip()]
    metric_rows = []
    pred_frames = []
    for split_id in splits:
        test_rows, test_probs_by_model = split_table(test_frames, metadata, split_id)
        val_rows, val_probs_by_model = split_table(val_frames, metadata, split_id)
        y_val = labels_to_idx(val_rows["true_top_action"])
        y_test = labels_to_idx(test_rows["true_top_action"])
        for base_method, weights in [
            ("uniform13", None),
            ("val_weighted13", validation_model_weights(val_probs_by_model, y_val, names)),
        ]:
            val_probs = average_probs(val_probs_by_model, names, weights)
            test_probs = average_probs(test_probs_by_model, names, weights)
            hp = select_hparams(
                val_rows,
                val_probs,
                y_val,
                alphas=alphas,
                temperatures=temps,
                scales=scales,
                seed=args.seed + split_id,
            )
            log_trans = transition_matrix(val_rows.reset_index(drop=True), y_val, hp["alpha"])
            pred = decode_hmm(
                test_rows.reset_index(drop=True),
                test_probs,
                log_trans,
                hp["temperature"],
                hp["transition_scale"],
            )
            m = metrics_np(y_test, pred)
            method = f"{base_method}_hmm_inner_selected"
            row = {
                "method": method,
                "split_id": int(split_id),
                "n": int(len(test_rows)),
                **m,
                **hp,
                "weights_json": "" if weights is None else json.dumps([float(x) for x in weights]),
            }
            metric_rows.append(row)
            pred_df = pd.DataFrame(
                {
                    "method": method,
                    "split_id": int(split_id),
                    "row_id": test_rows["row_id"].to_numpy(dtype=np.int64),
                    "true_top_action": test_rows["true_top_action"].to_numpy(),
                    "pred_top_action": idx_to_labels(pred),
                }
            )
            pred_frames.append(pred_df)
            print(
                f"RESULT split={split_id} method={method} bal={m['balanced_accuracy']:.6f} "
                f"acc={m['accuracy']:.6f} inner={hp['inner_balanced_accuracy']:.6f} "
                f"alpha={hp['alpha']} temp={hp['temperature']} scale={hp['transition_scale']}",
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
    metrics.to_csv(out_dir / "hmm_13model_metrics.csv", index=False)
    summary.to_csv(out_dir / "hmm_13model_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(out_dir / "hmm_13model_predictions.parquet", index=False)
    lines = [
        "# 13-Model HMM Decoder",
        "",
        "Hyperparameters are selected on a validation-internal decision split, then transitions are refit on all validation rows and evaluated on held-out test rows.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    save_markdown(out_dir / "hmm_13model_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(out_dir / "hmm_13model_summary.csv"), "metrics": str(out_dir / "hmm_13model_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
