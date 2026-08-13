"""Train-only action extreme example selection."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import ACTIONS


def _train_subset(df: pd.DataFrame, split_train_indices: Sequence[int] | None) -> pd.DataFrame:
    if split_train_indices is None:
        return df.copy()
    return df.iloc[list(split_train_indices)].copy()


def select_action_extremes(
    df: pd.DataFrame,
    action: str,
    split_train_indices: Sequence[int] | None,
    top_k: int = 50,
    threshold: float | None = None,
    max_per_decision: int | None = 1,
) -> pd.DataFrame:
    """Select high-belief examples for one action from train rows only."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    q_col = f"q_{action}"
    if q_col not in df.columns:
        raise ValueError(f"missing action probability column: {q_col}")

    candidates = _train_subset(df, split_train_indices)
    if threshold is not None:
        candidates = candidates.loc[candidates[q_col] >= threshold].copy()
    sort_cols = [q_col]
    ascending = [False]
    if "commitment_to_final" in candidates.columns:
        sort_cols.append("commitment_to_final")
        ascending.append(False)
    if "progress" in candidates.columns:
        sort_cols.append("progress")
        ascending.append(False)
    candidates = candidates.sort_values(sort_cols, ascending=ascending)

    if max_per_decision is not None and "decision_id" in candidates.columns:
        candidates = candidates.groupby("decision_id", group_keys=False).head(max_per_decision)

    if top_k is not None:
        candidates = candidates.head(top_k)

    out = candidates.copy()
    out["extreme_action"] = action
    out["extreme_score"] = out[q_col].astype(float)
    out["extreme_rank"] = range(1, len(out) + 1)
    return out


def select_balanced_extremes_for_actions(
    df: pd.DataFrame,
    split_train_indices: Sequence[int] | None,
    top_k: int = 50,
    threshold: float | None = None,
    max_per_decision: int | None = 1,
    actions: Sequence[str] = ACTIONS,
    balance_to_min: bool = True,
) -> pd.DataFrame:
    """Select balanced train-only extremes for all actions."""
    selected = [
        select_action_extremes(
            df=df,
            action=action,
            split_train_indices=split_train_indices,
            top_k=top_k,
            threshold=threshold,
            max_per_decision=max_per_decision,
        )
        for action in actions
    ]
    if not selected:
        return pd.DataFrame()
    if balance_to_min:
        n = min(len(part) for part in selected)
        selected = [part.head(n).copy() for part in selected]
    return pd.concat(selected, ignore_index=True)
