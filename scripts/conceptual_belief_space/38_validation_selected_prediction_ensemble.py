#!/usr/bin/env python
"""Validation-selected ensemble over saved top-action prediction files."""

from __future__ import annotations

import argparse
import json
import sys
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument(
        "--prediction_dirs",
        required=True,
        help="Comma-separated name=directory entries containing validation_predictions.parquet and test_predictions.parquet.",
    )
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_validation_selected_prediction_ensemble")
    p.add_argument("--weight_grid", default="0,0.1,0.2,0.35,0.5,0.65,0.8,0.9,1.0")
    p.add_argument("--temperature_grid", default="0.65,0.8,1.0,1.25,1.6")
    p.add_argument("--transition_weights", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--bias_search", choices=["basic", "coord_argmax"], default="basic")
    p.add_argument("--coord_steps", default="0.3,0.15,0.075")
    p.add_argument("--eps", type=float, default=1e-12)
    return p.parse_args()


def parse_prediction_dirs(text: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"prediction dir entry must be name=path: {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), Path(path.strip())))
    if len(out) < 1:
        raise ValueError("at least one prediction directory is required")
    names = [name for name, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names: {names}")
    return out


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    mapping = {action: i for i, action in enumerate(ACTIONS)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def normalize_probs(q: np.ndarray, eps: float) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    return q / q.sum(axis=1, keepdims=True)


def calibrate_probs(q: np.ndarray, bias: np.ndarray, temperature: float, eps: float) -> np.ndarray:
    logits = np.log(normalize_probs(q, eps)) / max(float(temperature), eps)
    logits = logits + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    mat = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        mat[int(true), int(pred)] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_json": json.dumps(mat.tolist()),
    }


def viterbi_decode(rows: pd.DataFrame, q: np.ndarray, transition_weight: float, eps: float) -> np.ndarray:
    if transition_weight <= 0:
        return q.argmax(axis=1)
    logp = np.log(normalize_probs(q, eps))
    pred = np.zeros(len(rows), dtype=np.int64)
    eye = np.eye(len(ACTIONS), dtype=np.float64) * float(transition_weight)
    order = rows[["decision_id", "prefix_token_index"]].copy()
    order["_pos"] = np.arange(len(rows), dtype=np.int64)
    order = order.sort_values(["decision_id", "prefix_token_index", "_pos"])
    for _, group in order.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        back = np.zeros_like(emit, dtype=np.int64)
        dp = emit[0].copy()
        for t in range(1, len(pos)):
            score = dp[:, None] + eye
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def load_prediction_dir(path: Path, role: str) -> pd.DataFrame:
    df = pd.read_parquet(path / f"{role}_predictions.parquet")
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} {role} predictions missing columns: {missing}")
    out = df[needed].copy()
    out["split_id"] = out["split_id"].astype(int)
    out["row_id"] = out["row_id"].astype(np.int64)
    return out


def aligned_split(
    frames: dict[str, pd.DataFrame],
    split_id: int,
    metadata: pd.DataFrame,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    common: set[int] | None = None
    split_parts: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        part = df[df["split_id"] == split_id].copy()
        rows = set(part["row_id"].astype(int))
        common = rows if common is None else common & rows
        split_parts[name] = part
    if not common:
        raise ValueError(f"no common rows for split {split_id}")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    rows = pd.DataFrame({"row_id": row_ids}).merge(metadata, on="row_id", how="left", validate="one_to_one")
    if rows[["decision_id", "prefix_token_index", "top_action"]].isna().any().any():
        raise ValueError(f"metadata merge failed for split {split_id}")
    probs: dict[str, np.ndarray] = {}
    labels = None
    for name, part in split_parts.items():
        part = part[part["row_id"].isin(row_ids)].sort_values("row_id")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"row order mismatch for {name} split {split_id}")
        current_labels = part["true_top_action"].to_numpy()
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError(f"label mismatch for {name} split {split_id}")
        probs[name] = normalize_probs(part[PROB_COLS].to_numpy(dtype=np.float64), eps)
    rows["true_top_action"] = labels
    return rows, probs


def candidate_weights(names: list[str], grid: list[float]) -> list[np.ndarray]:
    if len(names) == 1:
        return [np.ones(1, dtype=np.float64)]
    if len(names) == 2:
        return [np.asarray([1.0 - w, w], dtype=np.float64) for w in grid]
    weights = [np.ones(len(names), dtype=np.float64) / len(names)]
    for i in range(len(names)):
        one = np.zeros(len(names), dtype=np.float64)
        one[i] = 1.0
        weights.append(one)
    return weights


def weighted_probs(probs: dict[str, np.ndarray], names: list[str], weights: np.ndarray, eps: float) -> np.ndarray:
    stack = np.stack([probs[name] for name in names], axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(names), dtype=np.float64) / len(names)
    return normalize_probs(np.tensordot(w, stack, axes=(0, 0)), eps)


def bias_candidates(q: np.ndarray, y_true: np.ndarray) -> list[tuple[str, np.ndarray]]:
    pred_counts = np.bincount(q.argmax(axis=1), minlength=len(ACTIONS)).astype(np.float64) + 1.0
    true_counts = np.bincount(y_true, minlength=len(ACTIONS)).astype(np.float64) + 1.0
    prior = np.log(true_counts / true_counts.sum()) - np.log(pred_counts / pred_counts.sum())
    prior = prior - prior.mean()
    return [("none", np.zeros(len(ACTIONS), dtype=np.float64)), ("prior", prior)]


def fit_coord_argmax_bias(
    q: np.ndarray,
    y_true: np.ndarray,
    temperatures: list[float],
    steps: list[float],
) -> tuple[str, np.ndarray, float, float]:
    best_kind = "none"
    best_bias = np.zeros(len(ACTIONS), dtype=np.float64)
    best_temp = 1.0
    best_bal = -np.inf
    for kind, bias in bias_candidates(q, y_true):
        for temp in temperatures:
            pred = calibrate_probs(q, bias, temp, 1e-12).argmax(axis=1)
            bal = metric_dict(y_true, pred)["balanced_accuracy"]
            if float(bal) > best_bal:
                best_kind = kind
                best_bias = bias.copy()
                best_temp = float(temp)
                best_bal = float(bal)
    for step in steps:
        improved = True
        passes = 0
        while improved and passes < 12:
            improved = False
            passes += 1
            for action_idx in range(len(ACTIONS)):
                for sign in (-1.0, 1.0):
                    bias = best_bias.copy()
                    bias[action_idx] += sign * float(step)
                    bias = bias - bias.mean()
                    for temp in temperatures:
                        pred = calibrate_probs(q, bias, temp, 1e-12).argmax(axis=1)
                        bal = metric_dict(y_true, pred)["balanced_accuracy"]
                        if float(bal) > best_bal + 1e-12:
                            best_kind = "coord"
                            best_bias = bias.copy()
                            best_temp = float(temp)
                            best_bal = float(bal)
                            improved = True
    return best_kind, best_bias, best_temp, best_bal


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    specs = parse_prediction_dirs(args.prediction_dirs)
    names = [name for name, _ in specs]
    weight_grid = parse_float_list(args.weight_grid)
    temp_grid = parse_float_list(args.temperature_grid)
    transition_weights = parse_float_list(args.transition_weights)
    coord_steps = parse_float_list(args.coord_steps)
    metadata = pd.read_parquet(args.beliefs, columns=["row_id", "decision_id", "prefix_token_index", "top_action"])
    metadata["row_id"] = metadata["row_id"].astype(np.int64)

    val_frames = {name: load_prediction_dir(path, "validation") for name, path in specs}
    test_frames = {name: load_prediction_dir(path, "test") for name, path in specs}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))
    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        val_rows, val_probs = aligned_split(val_frames, split_id, metadata, args.eps)
        test_rows, test_probs = aligned_split(test_frames, split_id, metadata, args.eps)
        y_val = labels_to_idx(val_rows["true_top_action"])
        y_test = labels_to_idx(test_rows["true_top_action"])
        best: dict[str, object] | None = None
        best_q_test = None
        for weights in candidate_weights(names, weight_grid):
            q_val_base = weighted_probs(val_probs, names, weights, args.eps)
            q_test_base = weighted_probs(test_probs, names, weights, args.eps)
            if args.bias_search == "coord_argmax":
                bias_options = [fit_coord_argmax_bias(q_val_base, y_val, temp_grid, coord_steps)]
            else:
                bias_options = [
                    (bias_kind, bias, temp, float("nan"))
                    for bias_kind, bias in bias_candidates(q_val_base, y_val)
                    for temp in temp_grid
                ]
            for bias_kind, bias, temp, val_argmax_bal in bias_options:
                    q_val = calibrate_probs(q_val_base, bias, temp, args.eps)
                    for tw in transition_weights:
                        pred_val = viterbi_decode(val_rows, q_val, tw, args.eps)
                        val_bal = metric_dict(y_val, pred_val)["balanced_accuracy"]
                        if best is None or float(val_bal) > float(best["validation_balanced_accuracy"]):
                            best = {
                                "split_id": int(split_id),
                                "models_json": json.dumps(names),
                                "weights_json": json.dumps(weights.tolist()),
                                "bias_kind": bias_kind,
                                "bias_json": json.dumps(bias.tolist()),
                                "temperature": float(temp),
                                "transition_weight": float(tw),
                                "bias_search": args.bias_search,
                                "validation_argmax_balanced_accuracy": float(val_argmax_bal),
                                "validation_balanced_accuracy": float(val_bal),
                            }
                            best_q_test = calibrate_probs(q_test_base, bias, temp, args.eps)
        assert best is not None and best_q_test is not None
        pred_test = viterbi_decode(test_rows, best_q_test, float(best["transition_weight"]), args.eps)
        row = {
            **best,
            **metric_dict(y_test, pred_test),
            "n_test": int(len(test_rows)),
            "n_val": int(len(val_rows)),
        }
        metric_rows.append(row)
        pred_df = test_rows[["row_id", "decision_id", "prefix_token_index", "true_top_action"]].copy()
        pred_df.insert(0, "split_id", int(split_id))
        pred_df["pred_top_action"] = idx_to_labels(pred_test)
        for i, action in enumerate(ACTIONS):
            pred_df[f"prob_{action}"] = best_q_test[:, i]
        pred_frames.append(pred_df)
        print(
            f"RESULT split={split_id} val_bal={row['validation_balanced_accuracy']:.6f} "
            f"test_bal={row['balanced_accuracy']:.6f} acc={row['accuracy']:.6f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(
        [
            {
                "folds": int(metrics["split_id"].nunique()),
                "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
                "std_balanced_accuracy": float(metrics["balanced_accuracy"].std()),
                "mean_accuracy": float(metrics["accuracy"].mean()),
                "mean_macro_f1": float(metrics["macro_f1"].mean()),
                "mean_validation_balanced_accuracy": float(metrics["validation_balanced_accuracy"].mean()),
            }
        ]
    )
    metrics.to_csv(output_dir / "validation_selected_ensemble_metrics.csv", index=False)
    summary.to_csv(output_dir / "validation_selected_ensemble_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "validation_selected_ensemble_predictions.parquet", index=False)
    lines = [
        "# Validation-Selected Prediction Ensemble",
        "",
        f"- models: `{', '.join(names)}`",
        "- selection data: validation predictions only",
        "- evaluation data: held-out test predictions only",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Split Metrics",
        "",
        metrics[
            [
                "split_id",
                "validation_balanced_accuracy",
                "balanced_accuracy",
                "accuracy",
                "macro_f1",
                "weights_json",
                "bias_kind",
                "temperature",
                "transition_weight",
            ]
        ].to_markdown(index=False),
    ]
    save_markdown(output_dir / "validation_selected_ensemble_report.md", "\n".join(lines))
    print(json.dumps({"metrics": str(output_dir / "validation_selected_ensemble_metrics.csv"), "summary": str(output_dir / "validation_selected_ensemble_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
