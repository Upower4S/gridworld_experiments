#!/usr/bin/env python
"""Gridworld state/task-feature baseline for prefix top_action prediction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown


ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

GRID_SYMBOLS = ["#", "_", "A", "K", "D", "G"]
GRID_SYMBOL_NAMES = {"#": "wall", "_": "empty", "A": "agent", "K": "key", "D": "door", "G": "goal", "OUT": "out"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--canonical", default="outputs/canonical_steps.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_gridworld_state_baseline")
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_grid(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(text).splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0].isdigit() and len(parts) > 1 and all(tok.isdigit() for tok in parts[1:]):
            continue
        if parts[0].isdigit():
            parts = parts[1:]
        if parts:
            rows.append(parts)
    return rows


def find_pos(grid: list[list[str]], symbol: str) -> tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == symbol:
                return r, c
    return None


def is_passable(grid: list[list[str]], r: int, c: int, carrying_key: bool) -> bool:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[r]):
        return False
    val = grid[r][c]
    if val == "#":
        return False
    if val == "D" and not carrying_key:
        # Treat closed door as blocked until key is carried.
        return False
    return True


def shortest_first_actions(
    grid: list[list[str]],
    start: tuple[int, int] | None,
    target: tuple[int, int] | None,
    carrying_key: bool,
) -> set[str]:
    if start is None or target is None:
        return set()
    if start == target:
        return set()
    q = deque([(start, [])])
    seen = {start}
    found_len = None
    firsts: set[str] = set()
    while q:
        (r, c), path = q.popleft()
        if found_len is not None and len(path) > found_len:
            break
        if (r, c) == target:
            found_len = len(path)
            if path:
                firsts.add(path[0])
            continue
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen or not is_passable(grid, nr, nc, carrying_key):
                continue
            seen.add((nr, nc))
            q.append(((nr, nc), path + [action]))
    return firsts


def shortest_distance(
    grid: list[list[str]],
    start: tuple[int, int] | None,
    target: tuple[int, int] | None,
    carrying_key: bool,
) -> float:
    if start is None or target is None:
        return 99.0
    if start == target:
        return 0.0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), dist = q.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen or not is_passable(grid, nr, nc, carrying_key):
                continue
            if (nr, nc) == target:
                return float(dist + 1)
            seen.add((nr, nc))
            q.append(((nr, nc), dist + 1))
    return 99.0


def planned_goal_search(
    grid: list[list[str]],
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
    key: tuple[int, int] | None,
    carrying_key: bool,
) -> tuple[float, set[str]]:
    """Shortest path to goal in augmented state (cell, has_key)."""
    if start is None or goal is None:
        return 99.0, set()
    start_has_key = bool(carrying_key or key is None)
    if start == goal:
        return 0.0, set()
    q = deque([((start[0], start[1], start_has_key), 0, [])])
    seen = {(start[0], start[1], start_has_key)}
    found_len: float | None = None
    firsts: set[str] = set()
    while q:
        (r, c, has_key), dist, path = q.popleft()
        if found_len is not None and dist > found_len:
            break
        if (r, c) == goal:
            found_len = float(dist)
            if path:
                firsts.add(path[0])
            continue
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= len(grid) or nc >= len(grid[nr]):
                continue
            val = grid[nr][nc]
            if val == "#":
                continue
            next_has_key = bool(has_key or val == "K")
            if val == "D" and not next_has_key:
                continue
            state = (nr, nc, next_has_key)
            if state in seen:
                continue
            seen.add(state)
            q.append((state, dist + 1, path + [action]))
    return (99.0 if found_len is None else found_len), firsts


def manhattan(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 99.0
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def state_features(row: pd.Series) -> dict[str, float]:
    grid = parse_grid(row["grid_state_text"])
    h = len(grid)
    w = max((len(r) for r in grid), default=1)
    carrying = bool(row.get("carrying_key", False))
    pos = {sym: find_pos(grid, sym) for sym in ["A", "K", "D", "G"]}
    a = pos["A"]
    feats: dict[str, float] = {
        "grid_h": float(h),
        "grid_w": float(w),
        "carrying_key": float(carrying),
        "key_present": float(pos["K"] is not None),
        "door_present": float(pos["D"] is not None),
        "goal_present": float(pos["G"] is not None),
    }
    cell_count = max(h * w, 1)
    flat_counts = {sym: 0 for sym in GRID_SYMBOLS}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val in flat_counts:
                flat_counts[val] += 1
    for sym in GRID_SYMBOLS:
        feats[f"frac_{GRID_SYMBOL_NAMES[sym]}"] = flat_counts[sym] / cell_count
    # The current dataset uses 9x9 grids. Keeping the full cell map gives tree
    # models exact layout information without using labels or future actions.
    for r in range(9):
        for c in range(9):
            val = grid[r][c] if r < h and c < len(grid[r]) else "OUT"
            for sym in GRID_SYMBOLS:
                feats[f"cell_{r}_{c}_{GRID_SYMBOL_NAMES[sym]}"] = float(val == sym)
    for sym, p in pos.items():
        feats[f"{sym}_r"] = -1.0 if p is None else p[0] / max(h - 1, 1)
        feats[f"{sym}_c"] = -1.0 if p is None else p[1] / max(w - 1, 1)
    for target_name, target in [("key", pos["K"]), ("door", pos["D"]), ("goal", pos["G"])]:
        feats[f"dist_{target_name}"] = manhattan(a, target) / max(h + w, 1)
    # If key exists and is not carried, key is the immediate subgoal; otherwise the goal is.
    subgoal = pos["K"] if (pos["K"] is not None and not carrying) else pos["G"]
    feats["dist_subgoal"] = manhattan(a, subgoal) / max(h + w, 1)
    for target_name, target in [("key", pos["K"]), ("door", pos["D"]), ("goal", pos["G"]), ("subgoal", subgoal)]:
        feats[f"bfs_dist_{target_name}"] = shortest_distance(grid, a, target, carrying) / cell_count
    plan_dist_goal, first_to_plan_goal = planned_goal_search(grid, a, pos["G"], pos["K"], carrying)
    feats["plan_dist_goal"] = plan_dist_goal / cell_count
    first_to_goal = shortest_first_actions(grid, a, pos["G"], carrying)
    first_to_key = shortest_first_actions(grid, a, pos["K"], carrying)
    first_to_door = shortest_first_actions(grid, a, pos["D"], carrying)
    first_to_subgoal = shortest_first_actions(grid, a, subgoal, carrying)
    for action, (dr, dc) in ACTION_DELTAS.items():
        if a is None:
            nr, nc = -1, -1
            valid = False
        else:
            nr, nc = a[0] + dr, a[1] + dc
            valid = is_passable(grid, nr, nc, carrying)
        next_pos = None if not valid else (nr, nc)
        feats[f"valid_{action}"] = float(valid)
        if a is None:
            raw_next_symbol = "OUT"
        elif nr < 0 or nc < 0 or nr >= h or nc >= len(grid[nr]):
            raw_next_symbol = "OUT"
        else:
            raw_next_symbol = grid[nr][nc]
        for sym in [*GRID_SYMBOLS, "OUT"]:
            feats[f"{action}_next_{GRID_SYMBOL_NAMES[sym]}"] = float(raw_next_symbol == sym)
        feats[f"{action}_next_r"] = -1.0 if a is None else nr / max(h - 1, 1)
        feats[f"{action}_next_c"] = -1.0 if a is None else nc / max(w - 1, 1)
        for target_name, target in [("key", pos["K"]), ("door", pos["D"]), ("goal", pos["G"]), ("subgoal", subgoal)]:
            before = manhattan(a, target)
            after = manhattan(next_pos, target)
            feats[f"{action}_delta_dist_{target_name}"] = (before - after) / max(h + w, 1)
            before_bfs = shortest_distance(grid, a, target, carrying)
            after_bfs = shortest_distance(grid, next_pos, target, carrying) if valid else 99.0
            feats[f"{action}_delta_bfs_dist_{target_name}"] = (before_bfs - after_bfs) / cell_count
        if valid:
            after_plan_dist, _ = planned_goal_search(grid, next_pos, pos["G"], pos["K"], carrying or raw_next_symbol == "K")
        else:
            after_plan_dist = 99.0
        feats[f"{action}_delta_plan_dist_goal"] = (plan_dist_goal - after_plan_dist) / cell_count
        feats[f"{action}_first_to_plan_goal"] = float(action in first_to_plan_goal)
        feats[f"{action}_first_to_goal"] = float(action in first_to_goal)
        feats[f"{action}_first_to_key"] = float(action in first_to_key)
        feats[f"{action}_first_to_door"] = float(action in first_to_door)
        feats[f"{action}_first_to_subgoal"] = float(action in first_to_subgoal)
    return feats


def labels_to_idx(labels) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def grouped_splits(df: pd.DataFrame, group_col: str, seed: int, n_splits: int = 5):
    from sklearn.model_selection import GroupKFold

    groups = df[group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    k = min(n_splits, len(unique))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    rank = {g: i for i, g in enumerate(shuffled)}
    order = np.argsort([rank[g] for g in groups])
    dummy = np.zeros(len(df), dtype=np.int8)
    for split_id, (tr, te) in enumerate(GroupKFold(n_splits=k).split(dummy[order], groups=groups[order])):
        yield split_id, order[tr], order[te]


def train_eval_models(x: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, args: argparse.Namespace):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        ),
        "extra_trees_800": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "random_forest_600": RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=args.seed,
        ),
    }
    metric_rows = []
    pred_frames = []
    for split_id, train_idx, test_idx in grouped_splits(meta, args.group_col, args.seed):
        for name, model in models.items():
            model.fit(x.iloc[train_idx], y[train_idx])
            pred = model.predict(x.iloc[test_idx])
            m = metrics_np(y[test_idx], pred)
            row = {
                "model": name,
                "split_id": int(split_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **m,
            }
            metric_rows.append(row)
            pred_df = pd.DataFrame(
                {
                    "model": name,
                    "split_id": int(split_id),
                    "row_id": meta.iloc[test_idx]["row_id"].to_numpy(dtype=np.int64),
                    "true_top_action": idx_to_labels(y[test_idx]),
                    "pred_top_action": idx_to_labels(pred),
                }
            )
            pred_frames.append(pred_df)
            print(
                f"RESULT split={split_id} model={name} bal={m['balanced_accuracy']:.6f} acc={m['accuracy']:.6f}",
                flush=True,
            )
    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    beliefs = pd.read_parquet(
        args.beliefs,
        columns=["row_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "top_action"],
    )
    canonical = pd.read_parquet(
        args.canonical,
        columns=["trajectory_id", "step_id", "grid_state_text", "carrying_key"],
    )
    canonical["step_id"] = canonical["step_id"].astype(beliefs["step_id"].dtype)
    df = beliefs.merge(canonical, on=["trajectory_id", "step_id"], how="left", validate="many_to_one")
    if df["grid_state_text"].isna().any():
        raise ValueError("missing grid_state_text after joining canonical steps")
    state_df = pd.DataFrame([state_features(row) for _, row in df.iterrows()])
    prefix = pd.DataFrame(
        {
            "progress": pd.to_numeric(df["progress"], errors="coerce").fillna(0.0),
            "prefix_token_index": pd.to_numeric(df["prefix_token_index"], errors="coerce").fillna(0.0),
        }
    )
    x = pd.concat([state_df, prefix], axis=1).astype(float)
    y = labels_to_idx(df["top_action"])
    metrics, preds = train_eval_models(x, y, df, args)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    metrics.to_csv(out_dir / "gridworld_state_baseline_metrics.csv", index=False)
    summary.to_csv(out_dir / "gridworld_state_baseline_summary.csv", index=False)
    preds.to_parquet(out_dir / "gridworld_state_baseline_predictions.parquet", index=False)
    lines = [
        "# Gridworld State Top-Action Baseline",
        "",
        "Features use parsed grid_state/carrying_key plus prefix progress. They do not include agent_action or behavior q columns.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    save_markdown(out_dir / "gridworld_state_baseline_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(out_dir / "gridworld_state_baseline_summary.csv"), "metrics": str(out_dir / "gridworld_state_baseline_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
