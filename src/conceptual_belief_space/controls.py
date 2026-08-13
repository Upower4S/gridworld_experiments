"""Robustness-control helpers for conceptual belief space analyses."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS, DEFAULT_SEED, EPS

Q_COLS = [f"q_{action}" for action in ACTIONS]


def q_columns() -> list[str]:
    return list(Q_COLS)


def top_action_from_q(q: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.asarray(q, dtype=float), axis=1)
    return np.asarray(ACTIONS, dtype=object)[idx]


def commitment_from_q_columns(
    df: pd.DataFrame,
    *,
    q_cols: Sequence[str] = Q_COLS,
    action_col: str = "agent_action",
) -> np.ndarray:
    action_to_idx = {action: i for i, action in enumerate(ACTIONS)}
    q = df[list(q_cols)].to_numpy(dtype=float)
    out = np.full(len(df), np.nan, dtype=float)
    for i, action in enumerate(df[action_col].astype(str)):
        idx = action_to_idx.get(action)
        if idx is not None:
            out[i] = q[i, idx]
    return out


def entropy_commitment_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = np.clip(q, EPS, 1.0)
    entropy = -np.sum(q * np.log(q), axis=1) / np.log(len(ACTIONS))
    return 1.0 - entropy


def margin_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    sorted_q = np.sort(q, axis=1)
    return sorted_q[:, -1] - sorted_q[:, -2]


def behavior_metric_summary(
    df: pd.DataFrame,
    *,
    q_cols: Sequence[str] = Q_COLS,
    action_col: str = "agent_action",
    top_action_col: str | None = None,
) -> dict[str, float | int]:
    """Summarize action-belief quality under a chosen label or q-vector control."""
    if df.empty:
        return {
            "n_rows": 0,
            "n_decisions": 0,
            "mean_commitment_to_final": float("nan"),
            "mean_entropy_commitment": float("nan"),
            "mean_margin_commitment": float("nan"),
            "top_action_accuracy": float("nan"),
        }
    q = df[list(q_cols)].to_numpy(dtype=float)
    predicted_top = (
        df[top_action_col].astype(str).to_numpy()
        if top_action_col and top_action_col in df.columns
        else top_action_from_q(q)
    )
    labels = df[action_col].astype(str).to_numpy()
    commitment = commitment_from_q_columns(df, q_cols=q_cols, action_col=action_col)
    return {
        "n_rows": int(len(df)),
        "n_decisions": int(df["decision_id"].nunique()) if "decision_id" in df.columns else 0,
        "mean_commitment_to_final": float(np.nanmean(commitment)),
        "mean_entropy_commitment": float(np.nanmean(entropy_commitment_from_q(q))),
        "mean_margin_commitment": float(np.nanmean(margin_from_q(q))),
        "top_action_accuracy": float(np.mean(predicted_top == labels)),
    }


def shuffle_agent_action_across_decisions(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    decision_col: str = "decision_id",
    action_col: str = "agent_action",
    output_col: str = "agent_action_shuffled",
) -> pd.DataFrame:
    """Shuffle final action labels at decision level and broadcast to prefixes."""
    if decision_col not in df.columns or action_col not in df.columns:
        raise ValueError(f"missing {decision_col} or {action_col}")
    decisions = (
        df[[decision_col, action_col]]
        .drop_duplicates(decision_col)
        .sort_values(decision_col)
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    shuffled = decisions[action_col].astype(str).to_numpy().copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(decisions[decision_col].astype(str), shuffled))
    out = df.copy()
    out[output_col] = out[decision_col].astype(str).map(mapping)
    return out


def shuffle_q_vectors_across_decisions(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    decision_col: str = "decision_id",
    progress_col: str = "progress",
    q_cols: Sequence[str] = Q_COLS,
) -> pd.DataFrame:
    """Shuffle whole belief trajectories across decisions, matching rows by nearest progress."""
    if decision_col not in df.columns:
        raise ValueError(f"missing decision column: {decision_col}")
    missing_q = [col for col in q_cols if col not in df.columns]
    if missing_q:
        raise ValueError(f"missing q columns: {missing_q}")
    if progress_col not in df.columns:
        progress_col = "prefix_token_index"
    decisions = sorted(df[decision_col].astype(str).dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    sources = np.asarray(decisions, dtype=object).copy()
    rng.shuffle(sources)
    source_for_target = dict(zip(decisions, sources))

    groups = {
        str(decision): part.sort_values(progress_col).copy()
        for decision, part in df.groupby(df[decision_col].astype(str), sort=False)
    }
    out = df.copy()
    shuffled_q = np.full((len(out), len(q_cols)), np.nan, dtype=float)
    for target_decision, target_part in out.groupby(out[decision_col].astype(str), sort=False):
        source_decision = source_for_target[str(target_decision)]
        source_part = groups[source_decision]
        source_progress = pd.to_numeric(source_part[progress_col], errors="coerce").to_numpy(dtype=float)
        source_q = source_part[list(q_cols)].to_numpy(dtype=float)
        target_progress = pd.to_numeric(target_part[progress_col], errors="coerce").to_numpy(dtype=float)
        if len(source_progress) == 0:
            continue
        nearest_idx = np.abs(target_progress[:, None] - source_progress[None, :]).argmin(axis=1)
        shuffled_q[target_part.index.to_numpy()] = source_q[nearest_idx]
    for i, col in enumerate(q_cols):
        out[f"shuffled_{col}"] = shuffled_q[:, i]
    return out


def action_balanced_decision_ids(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    decision_col: str = "decision_id",
    action_col: str = "agent_action",
) -> list[str]:
    """Sample an equal number of decision ids per final action."""
    decisions = (
        df[[decision_col, action_col]]
        .drop_duplicates(decision_col)
        .dropna(subset=[decision_col, action_col])
    )
    counts = decisions[action_col].astype(str).value_counts()
    counts = counts.reindex(ACTIONS).dropna()
    if counts.empty:
        return []
    n = int(counts.min())
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for action in ACTIONS:
        ids = decisions.loc[decisions[action_col].astype(str) == action, decision_col].astype(str).to_numpy()
        if len(ids) == 0:
            continue
        ids = ids.copy()
        rng.shuffle(ids)
        selected.extend(ids[:n].tolist())
    return selected


def action_balanced_subset(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    decision_col: str = "decision_id",
    action_col: str = "agent_action",
) -> pd.DataFrame:
    ids = set(action_balanced_decision_ids(df, seed=seed, decision_col=decision_col, action_col=action_col))
    return df.loc[df[decision_col].astype(str).isin(ids)].copy()


def assign_cot_length_bins(
    df: pd.DataFrame,
    *,
    length_col: str = "num_cot_tokens",
    decision_col: str = "decision_id",
    output_col: str = "cot_length_bin",
) -> pd.DataFrame:
    """Assign short/medium/long bins by decision-level CoT length tertiles."""
    out = df.copy()
    if length_col not in out.columns:
        out[output_col] = "unknown"
        return out
    decision_lengths = (
        out[[decision_col, length_col]]
        .drop_duplicates(decision_col)
        .dropna(subset=[length_col])
        .copy()
    )
    if decision_lengths.empty:
        out[output_col] = "unknown"
        return out
    values = pd.to_numeric(decision_lengths[length_col], errors="coerce")
    q1 = float(values.quantile(1.0 / 3.0))
    q2 = float(values.quantile(2.0 / 3.0))

    def label(value: float) -> str:
        if not np.isfinite(value):
            return "unknown"
        if value <= q1:
            return "short"
        if value <= q2:
            return "medium"
        return "long"

    mapping = {
        str(row[decision_col]): label(float(row[length_col]))
        for _, row in decision_lengths.iterrows()
    }
    out[output_col] = out[decision_col].astype(str).map(mapping).fillna("unknown")
    return out


def residualize_features_train_only(
    features: np.ndarray,
    covariates: np.ndarray,
    train_indices: Sequence[int],
) -> np.ndarray:
    """Regress features on covariates using train rows only, then return residuals for all rows."""
    x = np.asarray(features, dtype=np.float32)
    c = np.asarray(covariates, dtype=np.float32)
    if x.ndim != 2 or c.ndim != 2 or len(x) != len(c):
        raise ValueError("features and covariates must be 2D matrices with equal row count")
    train = np.asarray(train_indices, dtype=np.int64)
    design = np.column_stack([np.ones(len(c), dtype=np.float32), c])
    beta, *_ = np.linalg.lstsq(design[train], x[train], rcond=None)
    fitted = design @ beta
    return np.asarray(x - fitted, dtype=np.float32)


def random_prefix_split_indices(
    n_rows: int,
    *,
    test_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Random prefix split for leakage diagnostics only; not valid for main claims."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_test = max(1, int(round(n_rows * test_fraction)))
    test = np.sort(idx[:n_test])
    train = np.sort(idx[n_test:])
    return train, test


def split_leakage_counts(
    df: pd.DataFrame,
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    group_col: str,
) -> dict[str, int | bool]:
    train_groups = set(df.iloc[list(train_indices)][group_col].astype(str))
    test_groups = set(df.iloc[list(test_indices)][group_col].astype(str))
    overlap = train_groups & test_groups
    return {
        "n_train_rows": int(len(train_indices)),
        "n_test_rows": int(len(test_indices)),
        "n_train_groups": int(len(train_groups)),
        "n_test_groups": int(len(test_groups)),
        "n_overlapping_groups": int(len(overlap)),
        "has_group_leakage": bool(overlap),
    }


def extreme_ablation_counts(
    df: pd.DataFrame,
    *,
    train_indices: Sequence[int] | None,
    thresholds: Sequence[float] = (0.8, 0.9, 0.95),
    top_ks: Sequence[int] = (100, 250, 500),
    max_per_decisions: Sequence[int] = (1, 3, 5),
    subset_names: Sequence[str] = ("all_prefixes",),
) -> pd.DataFrame:
    """Count balanced action extremes over a threshold/top-k/max-per-decision grid."""
    from .extremes import select_action_extremes

    train_df = df if train_indices is None else df.iloc[list(train_indices)].copy()
    rows: list[dict[str, object]] = []
    for subset, threshold, top_k, max_per_decision in product(
        subset_names,
        thresholds,
        top_ks,
        max_per_decisions,
    ):
        if subset == "all_prefixes":
            part = train_df
        elif subset in train_df.columns:
            part = train_df.loc[train_df[subset].fillna(False).astype(bool)]
        else:
            rows.append(
                {
                    "subset": subset,
                    "threshold": float(threshold),
                    "top_k": int(top_k),
                    "max_per_decision": int(max_per_decision),
                    "status": "skipped_missing_subset",
                    "balanced_count_per_action": 0,
                    "total_selected_balanced": 0,
                }
            )
            continue

        counts: dict[str, int] = {}
        for action in ACTIONS:
            selected = select_action_extremes(
                part,
                action=action,
                split_train_indices=None,
                top_k=top_k,
                threshold=threshold,
                max_per_decision=max_per_decision,
            )
            counts[action] = int(len(selected))
        balanced = min(counts.values()) if counts else 0
        rows.append(
            {
                "subset": subset,
                "threshold": float(threshold),
                "top_k": int(top_k),
                "max_per_decision": int(max_per_decision),
                "status": "ok" if balanced > 0 else "no_balanced_extremes",
                "balanced_count_per_action": int(balanced),
                "total_selected_balanced": int(balanced * len(ACTIONS)),
                **{f"selected_{action}": counts.get(action, 0) for action in ACTIONS},
            }
        )
    return pd.DataFrame(rows)
