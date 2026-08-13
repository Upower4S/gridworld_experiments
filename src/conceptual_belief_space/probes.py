"""Probe model helpers for conceptual belief space analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProbeModels:
    ridge_clr: object
    ridge_q: object
    ridge_commitment: object
    ridge_entropy: object
    elasticnet_clr: object | None
    logreg_top_action: object | None
    logreg_agent_action: object | None


def make_ridge(alpha: float):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha)


def make_multitask_elasticnet(alpha: float, l1_ratio: float, max_iter: int):
    from sklearn.linear_model import MultiTaskElasticNet

    return MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=0)


def make_logistic_regression(max_iter: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
    )


def can_fit_classifier(labels: Sequence[str]) -> bool:
    return len(set(map(str, labels))) >= 2


def shuffled_copy(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(y).copy()
    rng.shuffle(out)
    return out


def resolve_torch_device_name(device: str = "auto") -> str:
    """Resolve a requested torch device for GPU-accelerated Ridge solves."""
    import torch

    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested torch device `cuda`, but CUDA is not available.")
    if device == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("Requested torch device `mps`, but Apple MPS is not available.")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("torch device must be one of: auto, cpu, cuda, mps")
    return device


def fit_predict_torch_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    *,
    device: str = "auto",
    batch_size: int = 65536,
) -> np.ndarray:
    """Fit Ridge regression with torch and return NumPy predictions.

    This mirrors scikit-learn Ridge's default intercept behavior by centering
    train features and targets, solving for weights, and adding the target mean
    back at prediction time.
    """
    import torch

    resolved = resolve_torch_device_name(device)
    y_arr = np.array(y_train, dtype=np.float32, copy=True)
    y_was_1d = y_arr.ndim == 1
    if y_was_1d:
        y_arr = y_arr[:, None]
    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    if x_train_arr.ndim != 2 or x_test_arr.ndim != 2:
        raise ValueError("x_train and x_test must be 2D")
    if y_arr.ndim != 2:
        raise ValueError("y_train must be 1D or 2D")
    if x_train_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x_train and y_train row counts differ")

    torch_device = torch.device(resolved)
    x = torch.as_tensor(x_train_arr, dtype=torch.float32, device=torch_device)
    y = torch.as_tensor(y_arr, dtype=torch.float32, device=torch_device)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    xtx = x_centered.T @ x_centered
    eye = torch.eye(xtx.shape[0], dtype=torch.float32, device=torch_device)
    xty = x_centered.T @ y_centered
    weights = torch.linalg.solve(xtx + float(alpha) * eye, xty)

    preds: list[np.ndarray] = []
    batch_size = int(batch_size) if int(batch_size) > 0 else len(x_test_arr)
    with torch.no_grad():
        for start in range(0, len(x_test_arr), batch_size):
            part = torch.as_tensor(
                x_test_arr[start : start + batch_size],
                dtype=torch.float32,
                device=torch_device,
            )
            pred = (part - x_mean) @ weights + y_mean
            preds.append(pred.detach().cpu().numpy())
    out = np.concatenate(preds, axis=0).astype(np.float32, copy=False)
    return out.reshape(-1) if y_was_1d else out
