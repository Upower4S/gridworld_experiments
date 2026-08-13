"""Metrics for conceptual belief space probe analyses."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .beliefs import stable_softmax
from .config import ACTIONS, EPS
from .stats import pearson_corr


def mean_squared_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 that honestly reports negative values."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return float("nan")
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    numer = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numer / denom


def regression_vector_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dim_names: Sequence[str],
) -> dict[str, float]:
    """R2, MSE, and Pearson for a multi-output regression target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, float] = {"mse": mean_squared_error_np(y_true, y_pred)}
    r2_values: list[float] = []
    for i, name in enumerate(dim_names):
        r2 = r2_score_np(y_true[:, i], y_pred[:, i])
        corr = pearson_corr(y_true[:, i], y_pred[:, i])
        out[f"r2_{name}"] = r2
        out[f"pearson_{name}"] = corr
        if np.isfinite(r2):
            r2_values.append(float(r2))
    out["mean_r2"] = float(np.mean(r2_values)) if r2_values else float("nan")
    return out


def scalar_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics for scalar regression."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "r2": r2_score_np(y_true, y_pred),
        "mse": mean_squared_error_np(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
    }


def normalize_q_predictions(q_pred: np.ndarray) -> np.ndarray:
    """Clip and row-normalize predicted probabilities."""
    q = np.asarray(q_pred, dtype=float)
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    denom = q.sum(axis=1, keepdims=True)
    invalid = denom[:, 0] <= EPS
    denom = np.where(denom <= EPS, 1.0, denom)
    q = q / denom
    if invalid.any():
        q[invalid] = 1.0 / len(ACTIONS)
    return q


def q_from_clr(clr_pred: np.ndarray) -> np.ndarray:
    """Map predicted CLR-like scores back to an action distribution."""
    return stable_softmax(np.asarray(clr_pred, dtype=float), axis=1)


def action_argmax(q: np.ndarray) -> np.ndarray:
    """Return ACTIONS argmax labels for probability-like rows."""
    idx = np.argmax(np.asarray(q, dtype=float), axis=1)
    return np.asarray(ACTIONS, dtype=object)[idx]


def commitment_from_q(q: np.ndarray, agent_actions: Sequence[str]) -> np.ndarray:
    action_to_idx = {action: i for i, action in enumerate(ACTIONS)}
    out = np.empty(len(q), dtype=float)
    for i, action in enumerate(agent_actions):
        out[i] = q[i, action_to_idx[str(action)]]
    return out


def mean_kl_divergence(q_true: np.ndarray, q_pred: np.ndarray) -> float:
    q_true = normalize_q_predictions(q_true)
    q_pred = normalize_q_predictions(q_pred)
    kl = q_true * (np.log(np.clip(q_true, EPS, None)) - np.log(np.clip(q_pred, EPS, None)))
    return float(np.mean(np.sum(kl, axis=1)))


def q_prediction_metrics(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    *,
    true_top_action: Sequence[str],
    agent_action: Sequence[str],
) -> dict[str, float]:
    q_norm = normalize_q_predictions(q_pred)
    pred_top = action_argmax(q_norm)
    true_top = np.asarray(list(true_top_action), dtype=object)
    agent = np.asarray(list(agent_action), dtype=object)
    return {
        "top_action_accuracy_vs_true_top_action": float(np.mean(pred_top == true_top)),
        "top_action_accuracy_vs_agent_action": float(np.mean(pred_top == agent)),
        "kl_divergence": mean_kl_divergence(q_true, q_norm),
        "predicted_commitment_pearson": pearson_corr(
            commitment_from_q(q_true, agent),
            commitment_from_q(q_norm, agent),
        ),
    }


def balanced_accuracy_np(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = ACTIONS) -> float:
    true = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    recalls: list[float] = []
    for label in labels:
        mask = true == label
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1_np(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = ACTIONS) -> float:
    true = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    scores: list[float] = []
    for label in labels:
        tp = float(np.sum((true == label) & (pred == label)))
        fp = float(np.sum((true != label) & (pred == label)))
        fn = float(np.sum((true == label) & (pred != label)))
        denom = 2 * tp + fp + fn
        if denom > 0:
            scores.append(2 * tp / denom)
    return float(np.mean(scores)) if scores else float("nan")


def confusion_matrix_dict(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = ACTIONS) -> dict[str, dict[str, int]]:
    true = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    matrix: dict[str, dict[str, int]] = {}
    for actual in labels:
        matrix[actual] = {}
        for predicted in labels:
            matrix[actual][predicted] = int(np.sum((true == actual) & (pred == predicted)))
    return matrix


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float | str]:
    matrix = confusion_matrix_dict(y_true, y_pred)
    return {
        "balanced_accuracy": balanced_accuracy_np(y_true, y_pred),
        "macro_f1": macro_f1_np(y_true, y_pred),
        "confusion_matrix_json": json.dumps(matrix, sort_keys=True),
    }
