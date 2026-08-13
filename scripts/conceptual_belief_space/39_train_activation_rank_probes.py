#!/usr/bin/env python3
"""Activation-only rank probes from last-token-window combinations.

This experiment tests whether linear scores over activation-only features can
recover the ordering of q_UP/q_DOWN/q_LEFT/q_RIGHT.  The primary feature family
uses actual per-CoT-token activations from the packed activation store, not
previously measured prefix rows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED


Q_COLS = [f"q_{action}" for action in ACTIONS]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
DEFAULT_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
DEFAULT_REPRESENTATIONS = ["raw", "delta_pre", "unit_delta_pre"]
DEFAULT_VARIANTS = ["last1", "mean20", "exp20_tau2", "exp20_tau5", "exp20_tau10", "concat20"]
STRICT_SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
MENTION_COLS = [
    "first_action_mention_token_index",
    "first_final_action_mention_token_index",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
    "buffered_before_any_action_mention",
    "buffered_before_final_action_mention",
]


def read_parquet_robust(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet, falling back to fastparquet for legacy full-run files."""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as first_exc:
        try:
            return pd.read_parquet(path, columns=columns, engine="fastparquet")
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not read parquet file {path} with pyarrow or fastparquet. "
                f"pyarrow error: {first_exc!r}; fastparquet error: {second_exc!r}"
            ) from second_exc


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_csv_ints(value: str) -> list[int]:
    if value.lower() == "default":
        return list(DEFAULT_LAYERS)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    if value.lower() == "default":
        return list(DEFAULT_REPRESENTATIONS)
    return [x.strip() for x in value.split(",") if x.strip()]


def bf16_uint16_to_float32(values: np.ndarray) -> np.ndarray:
    uint32_values = values.astype(np.uint32) << 16
    return uint32_values.view(np.float32)


def path_for_layer(path_value: str | Path, layer: int) -> Path:
    path = Path(str(path_value))
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part.startswith("layer_"):
            parts[i] = f"layer_{layer}"
            return Path(*parts)
    text = str(path)
    if "{layer}" in text:
        return Path(text.format(layer=layer))
    return path


class BF16MatrixCache:
    """Small LRU cache for packed bf16 activation matrices."""

    def __init__(self, *, max_open: int = 64):
        self.max_open = max_open
        self._cache: OrderedDict[tuple[str, int], np.memmap] = OrderedDict()

    def get(self, path: Path, hidden_dim: int) -> np.memmap:
        key = (str(path), int(hidden_dim))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not path.exists():
            raise FileNotFoundError(path)
        size_bytes = path.stat().st_size
        row_bytes = int(hidden_dim) * 2
        if size_bytes % row_bytes:
            raise ValueError(f"{path} size is not divisible by hidden_dim*2")
        rows = size_bytes // row_bytes
        mmap = np.memmap(path, mode="r", dtype="<u2", shape=(rows, int(hidden_dim)))
        self._cache[key] = mmap
        if len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return mmap

    def load_row(self, path: Path, row_index: int, hidden_dim: int) -> np.ndarray:
        mmap = self.get(path, hidden_dim)
        return bf16_uint16_to_float32(np.asarray(mmap[int(row_index)])).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DecisionActivationPointers:
    decision_id: str
    trajectory_id: str
    step_id: int
    num_cot_tokens: int
    hidden_dim: int
    pre_file: str
    pre_row_index: int
    cot_file: str
    cot_rows_by_token: dict[int, int]
    available_layers: tuple[int, ...]


def select_smoke_decisions(df: pd.DataFrame, limit_decisions: int | None, seed: int) -> set[str] | None:
    if not limit_decisions:
        return None
    decisions = sorted(df["decision_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(decisions)
    return set(decisions[: int(limit_decisions)])


def prepare_belief_rows(
    path: Path,
    *,
    limit_decisions: int | None,
    seed: int,
    prefix_path: Path | None = None,
) -> pd.DataFrame:
    needed = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "num_cot_tokens",
        "activation_window_end_token_index",
        *Q_COLS,
        *MENTION_COLS,
        *STRICT_SUBSET_COLS,
    ]
    df = read_parquet_robust(path)
    missing = [c for c in ["decision_id", "trajectory_id", "prefix_token_index", *Q_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required behavior columns: {missing}")
    if "row_id" not in df.columns:
        df = df.copy()
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    if "top_action" not in df.columns:
        q = df[Q_COLS].to_numpy(dtype=np.float64)
        df["top_action"] = np.asarray(ACTIONS, dtype=object)[np.argmax(q, axis=1)]
    if "num_cot_tokens" not in df.columns:
        df["num_cot_tokens"] = np.nan
    if prefix_path is not None and prefix_path.exists():
        prefix_cols = [
            "prefix_id",
            "trajectory_id",
            "step_id",
            "prefix_token_index",
            "analysis_prefix_token_ids_json",
            "first_action_mention_token_index",
            "first_final_action_mention_token_index",
            "is_before_first_action_mention",
            "is_before_first_final_action_mention",
        ]
        try:
            all_prefix_cols = read_parquet_robust(prefix_path).columns.tolist()
            available_prefix_cols = [c for c in prefix_cols if c in all_prefix_cols]
            prefixes = read_parquet_robust(prefix_path, columns=available_prefix_cols)
            prefixes["analysis_prefix_num_tokens"] = prefixes["analysis_prefix_token_ids_json"].map(
                lambda raw: len(json.loads(str(raw))) if pd.notna(raw) else np.nan
            )
            merge_cols = ["prefix_id"] if "prefix_id" in df.columns else ["trajectory_id", "step_id", "prefix_token_index"]
            prefix_merge_cols = [
                c
                for c in [
                    "analysis_prefix_num_tokens",
                    "first_action_mention_token_index",
                    "first_final_action_mention_token_index",
                    "is_before_first_action_mention",
                    "is_before_first_final_action_mention",
                ]
                if c in prefixes.columns and c not in df.columns
            ]
            df = df.merge(prefixes[merge_cols + prefix_merge_cols], on=merge_cols, how="left")
            df["activation_window_end_token_index"] = np.where(
                df["prefix_token_index"].astype(int).le(0),
                0,
                df["analysis_prefix_num_tokens"].fillna(df["prefix_token_index"]).astype(int),
            )
        except Exception as exc:
            print(f"Warning: could not derive token-count window ends from {prefix_path}: {exc}")
            df["activation_window_end_token_index"] = df["prefix_token_index"].astype(int)
    else:
        df["activation_window_end_token_index"] = df["prefix_token_index"].astype(int)
    if "strict_before_any_action_mention" not in df.columns:
        if "first_action_mention_token_index" in df.columns:
            mention = pd.to_numeric(df["first_action_mention_token_index"], errors="coerce")
            fallback = df["is_before_first_action_mention"] if "is_before_first_action_mention" in df.columns else True
            df["strict_before_any_action_mention"] = np.where(
                mention.notna(),
                df["prefix_token_index"].astype(int) < mention,
                fallback,
            )
        elif "is_before_first_action_mention" in df.columns:
            df["strict_before_any_action_mention"] = df["is_before_first_action_mention"].fillna(True).astype(bool)
    if "strict_before_final_action_mention" not in df.columns:
        if "first_final_action_mention_token_index" in df.columns:
            mention = pd.to_numeric(df["first_final_action_mention_token_index"], errors="coerce")
            fallback = (
                df["is_before_first_final_action_mention"]
                if "is_before_first_final_action_mention" in df.columns
                else True
            )
            df["strict_before_final_action_mention"] = np.where(
                mention.notna(),
                df["prefix_token_index"].astype(int) < mention,
                fallback,
            )
        elif "is_before_first_final_action_mention" in df.columns:
            df["strict_before_final_action_mention"] = df["is_before_first_final_action_mention"].fillna(True).astype(bool)
    for col in STRICT_SUBSET_COLS:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    keep = [c for c in needed if c in df.columns]
    df = df[keep].copy()
    df["decision_id"] = df["decision_id"].astype(str)
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    df["prefix_token_index"] = df["prefix_token_index"].astype(int)
    q = df[Q_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError("q columns contain NaN or inf")
    sums = q.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        df.loc[:, Q_COLS] = q / sums[:, None]
    selected = select_smoke_decisions(df, limit_decisions, seed)
    if selected is not None:
        df = df[df["decision_id"].isin(selected)].reset_index(drop=True)
    return df.reset_index(drop=True)


def build_activation_pointers(
    token_index_path: Path,
    belief_decisions: set[str],
    *,
    layers: Iterable[int],
) -> dict[str, DecisionActivationPointers]:
    required = [
        "decision_id",
        "trajectory_id",
        "step_id",
        "token_type",
        "cot_token_index",
        "num_cot_tokens",
        "activation_file",
        "activation_format",
        "activation_row_index",
        "available_layers_json",
        "hidden_dim",
    ]
    token_df = read_parquet_robust(token_index_path, columns=required)
    token_df = token_df[token_df["decision_id"].astype(str).isin(belief_decisions)].copy()
    if token_df.empty:
        raise ValueError("No activation token-index rows match behavior decision_id values")
    token_df["decision_id"] = token_df["decision_id"].astype(str)
    token_df["token_type"] = token_df["token_type"].astype(str)
    out: dict[str, DecisionActivationPointers] = {}
    requested_layers = set(int(layer) for layer in layers)

    for decision_id, group in token_df.groupby("decision_id", sort=False):
        pre = group[group["token_type"].eq("pre_reasoning")]
        cot = group[group["token_type"].eq("cot")]
        if pre.empty or cot.empty:
            continue
        pre_row = pre.iloc[0]
        cot = cot.dropna(subset=["cot_token_index", "activation_row_index"]).copy()
        cot["cot_token_index"] = cot["cot_token_index"].astype(int)
        cot["activation_row_index"] = cot["activation_row_index"].astype(int)
        cot = cot.sort_values("cot_token_index")
        available_layers = tuple(json.loads(str(pre_row["available_layers_json"])))
        missing_layers = sorted(requested_layers.difference(available_layers))
        if missing_layers:
            raise ValueError(f"Decision {decision_id} lacks requested layers {missing_layers}")
        formats = set(group["activation_format"].astype(str).unique())
        if formats != {"packed_raw_bf16"}:
            raise ValueError(f"Unsupported activation formats for decision {decision_id}: {formats}")
        cot_rows = dict(zip(cot["cot_token_index"].astype(int), cot["activation_row_index"].astype(int)))
        out[decision_id] = DecisionActivationPointers(
            decision_id=decision_id,
            trajectory_id=str(pre_row["trajectory_id"]),
            step_id=int(pre_row["step_id"]),
            num_cot_tokens=int(pre_row["num_cot_tokens"]),
            hidden_dim=int(pre_row["hidden_dim"]),
            pre_file=str(pre_row["activation_file"]),
            pre_row_index=int(pre_row["activation_row_index"]),
            cot_file=str(cot.iloc[0]["activation_file"]),
            cot_rows_by_token=cot_rows,
            available_layers=available_layers,
        )
    missing = sorted(belief_decisions.difference(out))
    if missing:
        sample = missing[:5]
        raise ValueError(f"Activation pointers missing or malformed for {len(missing)} decisions, sample={sample}")
    return out


class Last20FeatureBuilder:
    """Build activation-only features for one layer from actual token activations."""

    def __init__(
        self,
        pointers: dict[str, DecisionActivationPointers],
        *,
        layer: int,
        representation: str,
        window_size: int,
        cache: BF16MatrixCache | None = None,
    ):
        self.pointers = pointers
        self.layer = int(layer)
        self.representation = representation
        self.window_size = int(window_size)
        self.cache = cache or BF16MatrixCache()
        hidden_dims = {p.hidden_dim for p in pointers.values()}
        if len(hidden_dims) != 1:
            raise ValueError(f"Expected one hidden dimension, found {hidden_dims}")
        self.hidden_dim = int(next(iter(hidden_dims)))
        if representation not in {"raw", "delta_pre", "unit_delta_pre"}:
            raise ValueError(f"Unsupported representation: {representation}")

    def _load_pre(self, pointer: DecisionActivationPointers) -> np.ndarray:
        return self.cache.load_row(path_for_layer(pointer.pre_file, self.layer), pointer.pre_row_index, pointer.hidden_dim)

    def _load_cot(self, pointer: DecisionActivationPointers, token_index: int) -> np.ndarray:
        if token_index not in pointer.cot_rows_by_token:
            raise KeyError(f"{pointer.decision_id} missing cot_token_index={token_index}")
        return self.cache.load_row(
            path_for_layer(pointer.cot_file, self.layer),
            pointer.cot_rows_by_token[int(token_index)],
            pointer.hidden_dim,
        )

    def _represent(self, h: np.ndarray, h_pre: np.ndarray) -> np.ndarray:
        if self.representation == "raw":
            return h
        delta = h - h_pre
        if self.representation == "delta_pre":
            return delta
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            return np.zeros_like(delta, dtype=np.float32)
        return (delta / norm).astype(np.float32, copy=False)

    def window(self, decision_id: str, prefix_token_index: int) -> tuple[np.ndarray, int, int]:
        pointer = self.pointers[str(decision_id)]
        h_pre = self._load_pre(pointer)
        k = int(prefix_token_index)
        available: list[np.ndarray] = []
        if k <= 0:
            available.append(self._represent(h_pre, h_pre))
        else:
            start = max(1, k - self.window_size + 1)
            missing = []
            for token_index in range(start, k + 1):
                if token_index not in pointer.cot_rows_by_token:
                    missing.append(token_index)
                    continue
                available.append(self._represent(self._load_cot(pointer, token_index), h_pre))
            if missing:
                raise KeyError(f"{decision_id} missing token activations in requested window: {missing[:10]}")
        n_available = min(len(available), self.window_size)
        out = np.zeros((self.window_size, self.hidden_dim), dtype=np.float32)
        if available:
            values = np.asarray(available[-self.window_size :], dtype=np.float32)
            out[-len(values) :] = values
        return out, n_available, max(0, self.window_size - n_available)


def combine_window(window: np.ndarray, n_available: int, variant: str) -> np.ndarray:
    if variant == "last1":
        return window[-1]
    if variant == "concat20":
        return window.reshape(-1)
    if n_available <= 0:
        return np.zeros(window.shape[1], dtype=np.float32)
    available = window[-n_available:]
    if variant == "mean20":
        return available.mean(axis=0)
    if variant.startswith("exp20_tau"):
        tau = float(variant.replace("exp20_tau", ""))
        ages = np.arange(n_available - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-ages / tau)
        weights /= weights.sum()
        return (available * weights[:, None]).sum(axis=0)
    raise ValueError(f"Unknown feature variant: {variant}")


def combine_window_with_train_pad(
    window: np.ndarray,
    n_available: int,
    variant: str,
    pad_vector: np.ndarray,
) -> np.ndarray:
    """Combine a 20-token window after replacing left pads with a train-fold mean vector."""
    if n_available >= window.shape[0]:
        filled = window
    else:
        filled = window.copy()
        n_pad = int(window.shape[0] - max(n_available, 0))
        if n_pad > 0:
            filled[:n_pad] = pad_vector.astype(np.float32, copy=False)
    if variant == "last1":
        return filled[-1]
    if variant == "concat20":
        return filled.reshape(-1)
    if variant == "mean20":
        return filled.mean(axis=0)
    if variant.startswith("exp20_tau"):
        tau = float(variant.replace("exp20_tau", ""))
        ages = np.arange(filled.shape[0] - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-ages / tau)
        weights /= weights.sum()
        return (filled * weights[:, None]).sum(axis=0)
    raise ValueError(f"Unknown feature variant: {variant}")


def make_group_splits(df: pd.DataFrame, n_splits: int, seed: int) -> list[dict[str, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    groups = df["trajectory_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two trajectory groups for leakage-free evaluation")
    if len(unique_groups) < n_splits:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return [
            {"split_id": 0, "train_idx": train_idx, "test_idx": test_idx}
            for train_idx, test_idx in splitter.split(df, groups=groups)
        ]
    splitter = GroupKFold(n_splits=n_splits)
    return [
        {"split_id": split_id, "train_idx": train_idx, "test_idx": test_idx}
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups))
    ]


def rank_targets(q: np.ndarray, tie_eps: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros((len(q), len(PAIR_INDICES)), dtype=np.float32)
    mask = np.ones_like(y, dtype=np.float32)
    for pair_id, (i, j) in enumerate(PAIR_INDICES):
        diff = q[:, i] - q[:, j]
        ties = np.abs(diff) <= tie_eps
        y[:, pair_id] = (diff > 0).astype(np.float32)
        mask[:, pair_id] = (~ties).astype(np.float32)
    return y, mask


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit_linear_rank_probe(
    x_train: np.ndarray,
    q_train: np.ndarray,
    x_test: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    tie_eps: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(x_train.shape[1], len(ACTIONS), bias=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y, mask = rank_targets(q_train, tie_eps)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    m_tensor = torch.tensor(mask, dtype=torch.float32)
    pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long, device=device)
    pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long, device=device)
    n = len(x_train)
    losses: list[float] = []
    rng = np.random.default_rng(seed)
    for _epoch in range(int(epochs)):
        order = rng.permutation(n)
        total = 0.0
        denom = 0
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            mb = m_tensor[idx].to(device)
            scores = model(xb)
            pair_logits = scores[:, pair_i] - scores[:, pair_j]
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, yb, reduction="none")
            loss = (raw_loss * mb).sum() / mb.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            denom += len(idx)
        losses.append(total / max(denom, 1))
    with torch.no_grad():
        train_scores = model(torch.tensor(x_train, dtype=torch.float32, device=device)).cpu().numpy()
        test_scores = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()
    return train_scores.astype(np.float32), test_scores.astype(np.float32), losses


def pairwise_accuracy(scores: np.ndarray, q: np.ndarray, tie_eps: float) -> float:
    correct = 0
    total = 0
    for i, j in PAIR_INDICES:
        diff = q[:, i] - q[:, j]
        valid = np.abs(diff) > tie_eps
        if not valid.any():
            continue
        pred = scores[:, i] > scores[:, j]
        truth = diff > 0
        correct += int((pred[valid] == truth[valid]).sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def orders_from_values(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, axis=1)


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    for row_i in range(order.shape[0]):
        ranks[row_i, order[row_i]] = np.arange(order.shape[1])
    return ranks


def kendall_tau_4(pred_order: np.ndarray, true_order: np.ndarray) -> np.ndarray:
    pred_rank = ranks_from_order(pred_order)
    true_rank = ranks_from_order(true_order)
    out = np.zeros(len(pred_order), dtype=np.float32)
    for row_i in range(len(pred_order)):
        concordant = 0
        discordant = 0
        for a, b in PAIR_INDICES:
            pred_sign = pred_rank[row_i, a] < pred_rank[row_i, b]
            true_sign = true_rank[row_i, a] < true_rank[row_i, b]
            if pred_sign == true_sign:
                concordant += 1
            else:
                discordant += 1
        out[row_i] = (concordant - discordant) / len(PAIR_INDICES)
    return out


def spearman_rank_4(pred_order: np.ndarray, true_order: np.ndarray) -> np.ndarray:
    pred_rank = ranks_from_order(pred_order).astype(np.float32)
    true_rank = ranks_from_order(true_order).astype(np.float32)
    pred_rank -= pred_rank.mean(axis=1, keepdims=True)
    true_rank -= true_rank.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(pred_rank, axis=1) * np.linalg.norm(true_rank, axis=1)
    denom[denom == 0] = np.nan
    return (pred_rank * true_rank).sum(axis=1) / denom


def metrics_for_rows(scores: np.ndarray, q: np.ndarray, *, tie_eps: float) -> dict[str, float]:
    pred_order = orders_from_values(scores)
    true_order = orders_from_values(q)
    pred_top = pred_order[:, 0]
    true_top = true_order[:, 0]
    pred_top2 = np.sort(pred_order[:, :2], axis=1)
    true_top2 = np.sort(true_order[:, :2], axis=1)
    return {
        "n_rows": int(len(scores)),
        "pairwise_ranking_accuracy": pairwise_accuracy(scores, q, tie_eps),
        "exact_4_action_order_accuracy": float((pred_order == true_order).all(axis=1).mean()),
        "top_action_accuracy": float((pred_top == true_top).mean()),
        "top2_set_accuracy": float((pred_top2 == true_top2).all(axis=1).mean()),
        "mean_kendall_tau": float(np.nanmean(kendall_tau_4(pred_order, true_order))),
        "mean_spearman_rank": float(np.nanmean(spearman_rank_4(pred_order, true_order))),
        **{
            f"top_action_recall_{action}": float(((pred_top == idx) & (true_top == idx)).sum() / max((true_top == idx).sum(), 1))
            for idx, action in enumerate(ACTIONS)
        },
    }


def feature_manifest_from_prefix_lengths(
    df: pd.DataFrame,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
) -> dict[str, Any]:
    end = pd.to_numeric(
        df.get("activation_window_end_token_index", df["prefix_token_index"]),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=np.int64)
    available_counts = np.where(end <= 0, 1, np.minimum(end, builder.window_size)).astype(int)
    pad_counts = np.maximum(builder.window_size - available_counts, 0)
    feature_dim = builder.hidden_dim * builder.window_size if variant == "concat20" else builder.hidden_dim
    return {
        "variant": variant,
        "feature_dim": int(feature_dim),
        "mean_available_tokens": float(np.mean(available_counts)),
        "min_available_tokens": int(np.min(available_counts)),
        "max_available_tokens": int(np.max(available_counts)),
        "mean_left_padding_tokens": float(np.mean(pad_counts)),
        "uses_actual_cot_token_activations": True,
        "uses_previous_measured_prefix_rows": False,
    }


def build_features_for_rows(
    df: pd.DataFrame,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = []
    available_counts = []
    pad_counts = []
    for _, row in df.iterrows():
        end_token_index = int(row.get("activation_window_end_token_index", row["prefix_token_index"]))
        window, n_available, n_pad = builder.window(str(row["decision_id"]), end_token_index)
        rows.append(combine_window(window, n_available, variant))
        available_counts.append(n_available)
        pad_counts.append(n_pad)
    x = np.asarray(rows, dtype=np.float32)
    manifest = {
        "variant": variant,
        "feature_dim": int(x.shape[1]),
        "mean_available_tokens": float(np.mean(available_counts)),
        "min_available_tokens": int(np.min(available_counts)),
        "max_available_tokens": int(np.max(available_counts)),
        "mean_left_padding_tokens": float(np.mean(pad_counts)),
        "uses_actual_cot_token_activations": True,
        "uses_previous_measured_prefix_rows": False,
    }
    return x, manifest


def train_pad_vector_for_indices(
    df: pd.DataFrame,
    indices: np.ndarray,
    builder: Last20FeatureBuilder,
) -> np.ndarray:
    total = np.zeros(builder.hidden_dim, dtype=np.float64)
    count = 0
    for idx in indices:
        row = df.iloc[int(idx)]
        end_token_index = int(row.get("activation_window_end_token_index", row["prefix_token_index"]))
        window, n_available, _ = builder.window(str(row["decision_id"]), end_token_index)
        if n_available <= 0:
            continue
        values = window[-int(n_available):]
        total += values.astype(np.float64, copy=False).sum(axis=0)
        count += int(n_available)
    if count == 0:
        return np.zeros(builder.hidden_dim, dtype=np.float32)
    return (total / count).astype(np.float32)


def build_features_for_indices_with_train_pad(
    df: pd.DataFrame,
    indices: np.ndarray,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
    pad_vector: np.ndarray,
) -> np.ndarray:
    rows = []
    for idx in indices:
        row = df.iloc[int(idx)]
        end_token_index = int(row.get("activation_window_end_token_index", row["prefix_token_index"]))
        window, n_available, _ = builder.window(str(row["decision_id"]), end_token_index)
        rows.append(combine_window_with_train_pad(window, n_available, variant, pad_vector))
    return np.asarray(rows, dtype=np.float32)


def build_features_for_indices(
    df: pd.DataFrame,
    indices: np.ndarray,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
) -> np.ndarray:
    x, _ = build_features_and_padding_mask_for_indices(
        df,
        indices,
        builder,
        variant=variant,
        return_padding_mask=False,
    )
    return x


def build_features_and_padding_mask_for_indices(
    df: pd.DataFrame,
    indices: np.ndarray,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
    return_padding_mask: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    rows = []
    padding_masks = []
    for idx in indices:
        row = df.iloc[int(idx)]
        end_token_index = int(row.get("activation_window_end_token_index", row["prefix_token_index"]))
        window, n_available, n_pad = builder.window(str(row["decision_id"]), end_token_index)
        rows.append(combine_window(window, n_available, variant))
        if return_padding_mask:
            if variant == "concat20":
                mask = np.zeros(builder.window_size * builder.hidden_dim, dtype=bool)
                if n_pad > 0:
                    mask[: int(n_pad) * builder.hidden_dim] = True
            else:
                mask = np.zeros(builder.hidden_dim, dtype=bool)
            padding_masks.append(mask)
    x = np.asarray(rows, dtype=np.float32)
    if return_padding_mask:
        return x, np.asarray(padding_masks, dtype=bool)
    return x, None


def streaming_feature_stats(
    df: pd.DataFrame,
    indices: np.ndarray,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = builder.hidden_dim * builder.window_size if variant == "concat20" else builder.hidden_dim
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    counts = np.zeros(feature_dim, dtype=np.float64)
    for start in range(0, len(indices), int(batch_size)):
        batch_idx = indices[start : start + int(batch_size)]
        xb, pad_mask = build_features_and_padding_mask_for_indices(
            df,
            batch_idx,
            builder,
            variant=variant,
            return_padding_mask=(variant == "concat20"),
        )
        xb = xb.astype(np.float64, copy=False)
        if pad_mask is not None:
            real = ~pad_mask
            total += np.where(real, xb, 0.0).sum(axis=0)
            total_sq += np.where(real, xb * xb, 0.0).sum(axis=0)
            counts += real.sum(axis=0)
        else:
            total += xb.sum(axis=0)
            total_sq += (xb * xb).sum(axis=0)
            counts += len(xb)
    safe_counts = np.maximum(counts, 1.0)
    mean = (total / safe_counts).astype(np.float32)
    var = total_sq / safe_counts - mean.astype(np.float64) ** 2
    var[var < 1e-12] = 1.0
    std = np.sqrt(var).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def fit_linear_rank_probe_streaming(
    df: pd.DataFrame,
    q_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    builder: Last20FeatureBuilder,
    *,
    variant: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    tie_eps: float,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    import torch

    mean, std = streaming_feature_stats(df, train_idx, builder, variant=variant, batch_size=batch_size)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(len(mean), len(ACTIONS), bias=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_all, mask_all = rank_targets(q_all, tie_eps)
    pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long, device=device)
    pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        order = rng.permutation(train_idx)
        total = 0.0
        denom = 0
        for start in range(0, len(order), int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            xb, pad_mask = build_features_and_padding_mask_for_indices(
                df,
                batch_idx,
                builder,
                variant=variant,
                return_padding_mask=(variant == "concat20"),
            )
            xb = (xb - mean) / std
            if pad_mask is not None:
                xb[pad_mask] = 0.0
            yb = y_all[batch_idx]
            mb = mask_all[batch_idx]
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb, dtype=torch.float32, device=device)
            mb_t = torch.tensor(mb, dtype=torch.float32, device=device)
            scores = model(xb_t)
            pair_logits = scores[:, pair_i] - scores[:, pair_j]
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, yb_t, reduction="none")
            loss = (raw_loss * mb_t).sum() / mb_t.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(batch_idx)
            denom += len(batch_idx)
        losses.append(total / max(denom, 1))

    score_batches = []
    with torch.no_grad():
        for start in range(0, len(test_idx), int(batch_size)):
            batch_idx = test_idx[start : start + int(batch_size)]
            xb, pad_mask = build_features_and_padding_mask_for_indices(
                df,
                batch_idx,
                builder,
                variant=variant,
                return_padding_mask=(variant == "concat20"),
            )
            xb = (xb - mean) / std
            if pad_mask is not None:
                xb[pad_mask] = 0.0
            scores = model(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
            score_batches.append(scores.astype(np.float32))
    return np.concatenate(score_batches, axis=0), losses


def add_metric_rows(
    metric_rows: list[dict[str, Any]],
    *,
    base: dict[str, Any],
    scores: np.ndarray,
    q: np.ndarray,
    test_df: pd.DataFrame,
    tie_eps: float,
) -> None:
    subsets: dict[str, np.ndarray] = {"all_prefixes": np.ones(len(test_df), dtype=bool)}
    for col in STRICT_SUBSET_COLS:
        if col in test_df.columns:
            subsets[col] = test_df[col].fillna(False).astype(bool).to_numpy()
    for subset, mask in subsets.items():
        if not mask.any():
            continue
        row = dict(base)
        row["eval_subset"] = subset
        row.update(metrics_for_rows(scores[mask], q[mask], tie_eps=tie_eps))
        metric_rows.append(row)


def primary_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rows that count as primary activation-only evidence."""
    out = metrics.copy()
    if "label_mode" in out.columns:
        out = out[out["label_mode"].eq("q_order")]
    if "feature_family" in out.columns:
        out = out[out["feature_family"].ne("invalid_progress_only_diagnostic")]
    return out


def make_predictions_frame(
    df: pd.DataFrame,
    scores: np.ndarray,
    *,
    split_id: int,
    layer: int,
    representation: str,
    feature_variant: str,
    family: str,
) -> pd.DataFrame:
    pred_order = orders_from_values(scores)
    true_order = orders_from_values(df[Q_COLS].to_numpy(dtype=np.float64))
    pred_top = pred_order[:, 0]
    out = df[
        [
            c
            for c in [
                "row_id",
                "prefix_id",
                "decision_id",
                "trajectory_id",
                "step_id",
                "prefix_token_index",
                "progress",
                "agent_action",
                "top_action",
                *Q_COLS,
                *MENTION_COLS,
                *STRICT_SUBSET_COLS,
            ]
            if c in df.columns
        ]
    ].copy()
    out["split_id"] = split_id
    out["feature_family"] = family
    out["layer"] = layer
    out["representation"] = representation
    out["feature_variant"] = feature_variant
    for idx, action in enumerate(ACTIONS):
        out[f"s_{action}"] = scores[:, idx]
    out["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[pred_top]
    out["predicted_order"] = [" > ".join(np.asarray(ACTIONS, dtype=object)[order]) for order in pred_order]
    out["true_order"] = [" > ".join(np.asarray(ACTIONS, dtype=object)[order]) for order in true_order]
    return out


def plot_metric_lines(metrics: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_metric_rows(metrics)
    all_rows = primary[primary["eval_subset"].eq("all_prefixes")].copy()
    if all_rows.empty:
        return

    def plot_by_layer(metric: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (rep, variant), sub in all_rows.groupby(["representation", "feature_variant"]):
            if sub.empty:
                continue
            summary = sub.groupby("layer")[metric].mean().sort_index()
            ax.plot(summary.index, summary.values, marker="o", label=f"{rep}/{variant}")
        ax.set_xlabel("activation layer")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.savefig(fig_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    plot_by_layer("pairwise_ranking_accuracy", "ranking_accuracy_by_layer.png", "Pairwise ranking accuracy by layer")
    plot_by_layer("top_action_accuracy", "top_action_accuracy_by_layer.png", "Top-action accuracy by layer")

    fig, ax = plt.subplots(figsize=(8, 5))
    combo = all_rows.groupby("feature_variant")["pairwise_ranking_accuracy"].mean().sort_values(ascending=False)
    ax.bar(combo.index.astype(str), combo.values)
    ax.set_ylabel("mean pairwise ranking accuracy")
    ax.set_title("Last-20 activation combination comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(fig_dir / "last20_combination_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    strict = primary[primary["eval_subset"].ne("all_prefixes")].copy()
    if not strict.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for subset, sub in strict.groupby("eval_subset"):
            summary = sub.groupby("layer")["pairwise_ranking_accuracy"].mean().sort_index()
            ax.plot(summary.index, summary.values, marker="o", label=subset)
        ax.set_xlabel("activation layer")
        ax.set_ylabel("pairwise ranking accuracy")
        ax.set_title("Strict pre-action rank decoding")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(fig_dir / "strict_pre_action_rank_decoding.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_confusion(predictions: pd.DataFrame, metrics: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    primary = primary_metric_rows(metrics)
    all_rows = primary[primary["eval_subset"].eq("all_prefixes")].copy()
    if all_rows.empty or predictions.empty:
        return
    best = all_rows.sort_values("pairwise_ranking_accuracy", ascending=False).iloc[0]
    mask = (
        predictions["split_id"].eq(best["split_id"])
        & predictions["layer"].eq(best["layer"])
        & predictions["representation"].eq(best["representation"])
        & predictions["feature_variant"].eq(best["feature_variant"])
    )
    sub = predictions[mask].copy()
    if sub.empty:
        return
    true = sub["top_action"].astype(str).to_numpy()
    pred = sub["predicted_top_action"].astype(str).to_numpy()
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    action_to_idx = {action: idx for idx, action in enumerate(ACTIONS)}
    for t, p in zip(true, pred):
        if t in action_to_idx and p in action_to_idx:
            matrix[action_to_idx[t], action_to_idx[p]] += 1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ACTIONS)), ACTIONS)
    ax.set_yticks(range(len(ACTIONS)), ACTIONS)
    ax.set_xlabel("predicted q-top action")
    ax.set_ylabel("true q-top action")
    ax.set_title("Best rank probe confusion matrix")
    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.savefig(fig_dir / "confusion_matrix_best_rank_probe.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, *, metrics: pd.DataFrame, manifest: dict[str, Any], predictions_rows: int) -> None:
    lines = [
        "# Activation-Only Rank Probe Report",
        "",
        "## Scope",
        "",
        "- Primary model input: activation vectors only.",
        "- Excluded from primary feature matrices: grid/task features, prompt metadata, progress, prefix index, trajectory id, decision id, and action labels.",
        "- Target: ordering of q_UP/q_DOWN/q_LEFT/q_RIGHT, trained with pairwise logistic ranking over the six action pairs.",
        "- Last-20 features are loaded from actual CoT-token activation rows in `activation_token_index.parquet`.",
        "- Prefix 0 uses the pre_reasoning activation and left-padding.",
        "- Non-concat short windows use train-fold mean activation padding before the window is combined; concat padding is zero after train-only standardization.",
        "",
        "## Run Summary",
        "",
        f"- Behavior rows: {manifest.get('n_behavior_rows'):,}",
        f"- Decisions: {manifest.get('n_decisions'):,}",
        f"- Trajectories: {manifest.get('n_trajectories'):,}",
        f"- Layers: {manifest.get('layers')}",
        f"- Representations: {manifest.get('representations')}",
        f"- Feature variants: {manifest.get('feature_variants')}",
        f"- Splits: {manifest.get('n_splits')}",
        f"- Prediction rows saved: {predictions_rows:,}",
        "",
        "## Best Metrics",
        "",
    ]
    if not metrics.empty:
        primary = primary_metric_rows(metrics)
        all_rows = primary[primary["eval_subset"].eq("all_prefixes")]
        if not all_rows.empty:
            best = all_rows.sort_values("pairwise_ranking_accuracy", ascending=False).iloc[0]
            for key in [
                "split_id",
                "layer",
                "representation",
                "feature_variant",
                "pairwise_ranking_accuracy",
                "exact_4_action_order_accuracy",
                "top_action_accuracy",
                "top2_set_accuracy",
                "mean_kendall_tau",
                "mean_spearman_rank",
            ]:
                lines.append(f"- {key}: {best[key]}")
        strict = primary[primary["eval_subset"].ne("all_prefixes")]
        if not strict.empty:
            lines.extend(["", "## Strict Pre-Action Metrics", ""])
            strict_summary = strict.groupby("eval_subset")[
                ["pairwise_ranking_accuracy", "top_action_accuracy", "mean_kendall_tau"]
            ].mean()
            lines.append(strict_summary.to_markdown())
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Grouped split column: trajectory_id.",
            "- No trajectory appears in both train and test within a split.",
            "- Pairwise labels are derived only from q ordering; q ties within the configured tolerance are masked.",
            "- Feature construction uses packed `cot.bf16` rows addressed by `cot_token_index`; it does not reuse previous measured prefix rows.",
            "- For `concat20`, padded positions are excluded from train statistics and set to zero after train-only standardization.",
            "- A shuffled-q diagnostic is supported by `--include_shuffled_baseline`; those rows are labeled `label_mode=shuffled_q`.",
            "",
            "## Attention-Pooled Branch",
            "",
            "- Forced-action attention pooling is implemented in `40_extract_forced_action_attention_pooling.py`.",
            "- It requires a local model/runtime that returns attentions. Together/API scoring cannot provide the needed query-to-prefix attention matrix.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beliefs",
        default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet",
    )
    parser.add_argument(
        "--activation_index",
        default="outputs/activation/activation_token_index.parquet",
    )
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space/activation_rank_probes",
    )
    parser.add_argument(
        "--figure_dir",
        default="figures/conceptual_belief_space/activation_rank_probes",
    )
    parser.add_argument("--layers", default="default")
    parser.add_argument("--representations", default="default")
    parser.add_argument("--feature_variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--limit_decisions", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tie_eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save_predictions", choices=["all", "best", "none"], default="all")
    parser.add_argument("--include_shuffled_baseline", action="store_true")
    parser.add_argument("--include_progress_only_diagnostic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_csv_ints(args.layers)
    representations = parse_csv_strings(args.representations)
    variants = parse_csv_strings(args.feature_variants)
    invalid_variants = [v for v in variants if v not in DEFAULT_VARIANTS]
    if invalid_variants:
        raise ValueError(f"Unsupported feature variants: {invalid_variants}")

    beliefs = prepare_belief_rows(
        Path(args.beliefs),
        limit_decisions=args.limit_decisions,
        seed=args.seed,
        prefix_path=Path(args.prefixes) if args.prefixes else None,
    )
    pointers = build_activation_pointers(
        Path(args.activation_index),
        set(beliefs["decision_id"].astype(str).unique()),
        layers=layers,
    )
    splits = make_group_splits(beliefs, args.n_splits, args.seed)
    q_all = beliefs[Q_COLS].to_numpy(dtype=np.float64)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_manifest: dict[str, Any] = {
        "input_behavior_table": str(Path(args.beliefs)),
        "input_activation_token_index": str(Path(args.activation_index)),
        "n_behavior_rows": int(len(beliefs)),
        "n_decisions": int(beliefs["decision_id"].nunique()),
        "n_trajectories": int(beliefs["trajectory_id"].nunique()),
        "layers": layers,
        "representations": representations,
        "feature_variants": variants,
        "window_size": int(args.window_size),
        "n_splits": int(len(splits)),
        "primary_feature_inputs": "activation_only",
        "excluded_primary_inputs": [
            "grid_state",
            "task_features",
            "progress",
            "prefix_token_index",
            "num_cot_tokens",
            "trajectory_id",
            "decision_id",
            "agent_action",
            "top_action",
        ],
        "feature_checks": [],
        "window_end_policy": "prefix 0 uses pre_reasoning; later prefixes use len(analysis_prefix_token_ids_json) when available, otherwise prefix_token_index",
        "progress_only_diagnostic_included": bool(args.include_progress_only_diagnostic),
        "progress_only_diagnostic_status": "invalid leakage/metadata diagnostic; not activation-only evidence",
    }

    cache = BF16MatrixCache()
    for layer in layers:
        for representation in representations:
            print(f"Layer {layer} representation {representation}")
            builder = Last20FeatureBuilder(
                pointers,
                layer=layer,
                representation=representation,
                window_size=args.window_size,
                cache=cache,
            )
            for variant in variants:
                print(f"  Building {variant}")
                feature_info = feature_manifest_from_prefix_lengths(beliefs, builder, variant=variant)
                if variant == "concat20":
                    feature_info.update({
                        "streaming_training": True,
                        "padding_policy": "padded concat positions are excluded from train statistics and set to zero after train-only standardization",
                        "train_mean_padding": False,
                    })
                    x = None
                else:
                    feature_info.update({
                        "streaming_training": False,
                        "padding_policy": "short windows are left-padded with the train-fold mean activation vector before combination; aggregate features are then train-only standardized",
                        "train_mean_padding": True,
                    })
                    x = None
                feature_info.update({"layer": layer, "representation": representation})
                feature_manifest["feature_checks"].append(feature_info)
                for split in splits:
                    split_id = int(split["split_id"])
                    train_idx = split["train_idx"]
                    test_idx = split["test_idx"]
                    train_groups = set(beliefs.iloc[train_idx]["trajectory_id"].astype(str))
                    test_groups = set(beliefs.iloc[test_idx]["trajectory_id"].astype(str))
                    leakage = sorted(train_groups.intersection(test_groups))
                    if leakage:
                        raise RuntimeError(f"Trajectory leakage in split {split_id}: {leakage[:5]}")
                    q_train = q_all[train_idx]
                    q_test = q_all[test_idx]
                    if variant == "concat20":
                        scores, losses = fit_linear_rank_probe_streaming(
                            beliefs,
                            q_all,
                            train_idx,
                            test_idx,
                            builder,
                            variant=variant,
                            epochs=args.epochs,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            seed=args.seed + split_id + layer,
                            tie_eps=args.tie_eps,
                            batch_size=args.batch_size,
                        )
                    else:
                        pad_vector = train_pad_vector_for_indices(beliefs, train_idx, builder)
                        x_train_raw = build_features_for_indices_with_train_pad(
                            beliefs, train_idx, builder, variant=variant, pad_vector=pad_vector
                        )
                        x_test_raw = build_features_for_indices_with_train_pad(
                            beliefs, test_idx, builder, variant=variant, pad_vector=pad_vector
                        )
                        x_train, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)
                        _, scores, losses = fit_linear_rank_probe(
                            x_train,
                            q_train,
                            x_test,
                            epochs=args.epochs,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            seed=args.seed + split_id + layer,
                            tie_eps=args.tie_eps,
                            batch_size=args.batch_size,
                        )
                    base = {
                        "feature_family": "last20_token_window",
                        "label_mode": "q_order",
                        "split_id": split_id,
                        "layer": layer,
                        "representation": representation,
                        "feature_variant": variant,
                        "window_size": int(args.window_size),
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "final_train_loss": float(losses[-1]) if losses else float("nan"),
                    }
                    add_metric_rows(
                        metric_rows,
                        base=base,
                        scores=scores,
                        q=q_test,
                        test_df=beliefs.iloc[test_idx],
                        tie_eps=args.tie_eps,
                    )
                    if args.save_predictions == "all":
                        prediction_frames.append(
                            make_predictions_frame(
                                beliefs.iloc[test_idx].reset_index(drop=True),
                                scores,
                                split_id=split_id,
                                layer=layer,
                                representation=representation,
                                feature_variant=variant,
                                family="last20_token_window",
                            )
                        )
                    if args.include_shuffled_baseline:
                        rng = np.random.default_rng(args.seed + split_id + 999)
                        shuffled_train_q = q_train.copy()
                        rng.shuffle(shuffled_train_q, axis=0)
                        if variant == "concat20":
                            shuffled_q_all = q_all.copy()
                            shuffled_q_all[train_idx] = shuffled_train_q
                            shuffled_scores, shuffled_losses = fit_linear_rank_probe_streaming(
                                beliefs,
                                shuffled_q_all,
                                train_idx,
                                test_idx,
                                builder,
                                variant=variant,
                                epochs=args.epochs,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                seed=args.seed + split_id + layer + 777,
                                tie_eps=args.tie_eps,
                                batch_size=args.batch_size,
                            )
                        else:
                            _, shuffled_scores, shuffled_losses = fit_linear_rank_probe(
                                x_train,
                                shuffled_train_q,
                                x_test,
                                epochs=args.epochs,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                seed=args.seed + split_id + layer + 777,
                                tie_eps=args.tie_eps,
                                batch_size=args.batch_size,
                            )
                        shuffled_base = dict(base)
                        shuffled_base["label_mode"] = "shuffled_q"
                        shuffled_base["final_train_loss"] = float(shuffled_losses[-1]) if shuffled_losses else float("nan")
                        add_metric_rows(
                            metric_rows,
                            base=shuffled_base,
                            scores=shuffled_scores,
                            q=q_test,
                            test_df=beliefs.iloc[test_idx],
                            tie_eps=args.tie_eps,
                        )


    if args.include_progress_only_diagnostic:
        diag_cols = ["progress", "prefix_token_index", "num_cot_tokens"]
        diag = beliefs[diag_cols].copy()
        diag = diag.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_diag = diag.to_numpy(dtype=np.float32)
        for split in splits:
            split_id = int(split["split_id"])
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            x_train, x_test, _, _ = standardize_train_test(x_diag[train_idx], x_diag[test_idx])
            _, scores, losses = fit_linear_rank_probe(
                x_train,
                q_all[train_idx],
                x_test,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed + split_id + 404,
                tie_eps=args.tie_eps,
                batch_size=args.batch_size,
            )
            base = {
                "feature_family": "invalid_progress_only_diagnostic",
                "label_mode": "q_order",
                "split_id": split_id,
                "layer": -1,
                "representation": "metadata_diagnostic",
                "feature_variant": "progress_prefix_length_only",
                "window_size": 0,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "final_train_loss": float(losses[-1]) if losses else float("nan"),
            }
            add_metric_rows(
                metric_rows,
                base=base,
                scores=scores,
                q=q_all[test_idx],
                test_df=beliefs.iloc[test_idx],
                tie_eps=args.tie_eps,
            )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "rank_probe_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if args.save_predictions == "best" and not metrics.empty:
        # Kept for CLI compatibility. The all-prediction path is preferred for analysis.
        pass
    predictions_path = output_dir / "rank_probe_predictions.parquet"
    if not predictions.empty:
        predictions.to_parquet(predictions_path, index=False)
    else:
        pd.DataFrame().to_parquet(predictions_path, index=False)

    write_json(output_dir / "last20_feature_manifest.json", feature_manifest)
    attention_manifest = {
        "implemented_in": "scripts/conceptual_belief_space/40_extract_forced_action_attention_pooling.py",
        "status": "not_run_by_39_train_activation_rank_probes",
        "requires_local_model_attentions": True,
        "api_backends_supported": False,
    }
    write_json(output_dir / "attention_pooling_manifest.json", attention_manifest)
    plot_metric_lines(metrics, fig_dir)
    plot_confusion(predictions, metrics, fig_dir)
    write_report(output_dir / "activation_rank_probe_report.md", metrics=metrics, manifest=feature_manifest, predictions_rows=len(predictions))

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {output_dir / 'last20_feature_manifest.json'}")
    print(f"Wrote {output_dir / 'activation_rank_probe_report.md'}")
    print(f"Wrote figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
