#!/usr/bin/env python
"""Blend activation-ensemble emissions with gridworld task-feature classifiers.

The activation side uses existing validation/test prediction parquet files from
the 13 base models. The grid side is trained inside each outer split, excluding
both the validation rows and held-out test rows used by those activation files.
Blend weight and Viterbi transition weight are selected on validation only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ModelSpec("layer_10_12_14_16_18_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_10_12_14_16_18"),
    ModelSpec("layer_08_10_12_14_16_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_08_10_12_14_16"),
    ModelSpec("layer_10_12_14_16_18_20_h512_qonly", "top_action_valpred_layer_10_20_h512_qonly_full"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--canonical", default="outputs/canonical_steps.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_grid_activation_blend")
    p.add_argument("--topks", default="7,9,11")
    p.add_argument("--grid_weights", default="0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0")
    p.add_argument("--transition_weights", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_grid_helper():
    helper_path = ROOT / "scripts" / "conceptual_belief_space" / "31_gridworld_state_top_action_baseline.py"
    spec = importlib.util.spec_from_file_location("gridworld_state_top_action_baseline", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gridworld_state_top_action_baseline", module)
    spec.loader.exec_module(module)
    return module


def load_prediction(base_dir: Path, spec: ModelSpec, role: str) -> pd.DataFrame:
    path = base_dir / spec.directory / f"{role}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if spec.feature_set is not None:
        df = df[df["feature_set"] == spec.feature_set].copy()
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if df.empty:
        raise ValueError(f"empty predictions for {spec.name}")
    df = df[needed].copy()
    df["split_id"] = df["split_id"].astype(int)
    df["row_id"] = df["row_id"].astype(np.int64)
    return df


def labels_to_idx(labels) -> np.ndarray:
    mapping = {action: i for i, action in enumerate(ACTIONS)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def normalize_probs(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, None)
    return q / q.sum(axis=1, keepdims=True)


def activation_frame(frames: dict[str, pd.DataFrame], split_id: int, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    common: set[int] | None = None
    parts = {}
    for name, df in frames.items():
        part = df[df["split_id"] == split_id].copy()
        rows = set(part["row_id"].astype(int))
        common = rows if common is None else common & rows
        parts[name] = part
    if not common:
        raise ValueError(f"no common rows for split {split_id}")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    rows = pd.DataFrame({"row_id": row_ids}).merge(meta, on="row_id", how="left", validate="one_to_one")
    stack = []
    labels = None
    for name, part in parts.items():
        part = part[part["row_id"].isin(row_ids)].sort_values("row_id")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"row order mismatch for {name} split {split_id}")
        q = normalize_probs(part[PROB_COLS].to_numpy(dtype=np.float64))
        stack.append(q)
        labels = part["true_top_action"].to_numpy() if labels is None else labels
    return row_ids, rows, np.stack(stack, axis=0), labels_to_idx(labels)


def activation_class_weights(stack: np.ndarray, y_true: np.ndarray, topk: int) -> np.ndarray:
    weights = np.zeros((stack.shape[0], len(ACTIONS)), dtype=np.float64)
    for action_idx in range(len(ACTIONS)):
        mask = y_true == action_idx
        rel = np.asarray([stack[m, mask, action_idx].mean() if mask.any() else 0.25 for m in range(stack.shape[0])])
        keep = np.argsort(rel)[-min(int(topk), stack.shape[0]) :]
        weights[keep, action_idx] = rel[keep]
    weights = np.clip(weights, 1e-12, None)
    return weights / weights.sum(axis=0, keepdims=True)


def aggregate_activation(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return normalize_probs(np.sum(stack * weights[:, None, :], axis=0))


def viterbi_decode(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float) -> np.ndarray:
    if transition_weight <= 0:
        return probs.argmax(axis=1)
    logp = np.log(normalize_probs(probs))
    pred = np.zeros(len(rows), dtype=np.int64)
    order = rows[["decision_id", "prefix_token_index"]].copy()
    order["_pos"] = np.arange(len(rows), dtype=np.int64)
    order = order.sort_values(["decision_id", "prefix_token_index", "_pos"])
    eye = np.eye(len(ACTIONS), dtype=np.float64)
    for _, group in order.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        back = np.zeros_like(emit, dtype=np.int64)
        dp = emit[0].copy()
        for t in range(1, len(pos)):
            score = dp[:, None] + transition_weight * eye
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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


def choose_blend(rows: pd.DataFrame, q_act: np.ndarray, q_grid: np.ndarray, y_true: np.ndarray, grid_weights: list[float], transition_weights: list[float]) -> tuple[float, float, float]:
    best_gw = 0.0
    best_tw = 0.0
    best_bal = -np.inf
    for gw in grid_weights:
        q = normalize_probs((1.0 - gw) * q_act + gw * q_grid)
        for tw in transition_weights:
            pred = viterbi_decode(rows, q, tw)
            bal = float(metric_dict(y_true, pred)["balanced_accuracy"])
            if bal > best_bal:
                best_bal = bal
                best_gw = float(gw)
                best_tw = float(tw)
    return best_gw, best_tw, best_bal


def grid_models(seed: int):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "hgbdt": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=31, l2_regularization=0.05, random_state=seed),
        "rf300": RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "et400": ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgb_d3_400": XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=400, max_depth=3, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=2),
        "xgb_d4_500": XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0, min_child_weight=2),
        "xgb_d5_400": XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, min_child_weight=3),
    }


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(ACTIONS)), dtype=np.float64)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, j]
    return normalize_probs(out)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    base_dir = Path(args.base_dir)
    grid_weights = [float(x) for x in args.grid_weights.split(",") if x.strip()]
    transition_weights = [float(x) for x in args.transition_weights.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]

    grid_helper = load_grid_helper()
    beliefs = pd.read_parquet(args.beliefs, columns=["row_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "top_action"])
    canonical = pd.read_parquet(args.canonical, columns=["trajectory_id", "step_id", "grid_state_text", "carrying_key"])
    canonical["step_id"] = canonical["step_id"].astype(beliefs["step_id"].dtype)
    full = beliefs.merge(canonical, on=["trajectory_id", "step_id"], how="left", validate="many_to_one").reset_index(drop=True)
    if full["grid_state_text"].isna().any():
        raise ValueError("missing grid_state_text after canonical join")
    state = pd.DataFrame([grid_helper.state_features(row) for _, row in full.iterrows()])
    prefix = pd.DataFrame({"progress": pd.to_numeric(full["progress"], errors="coerce").fillna(0.0), "prefix_token_index": pd.to_numeric(full["prefix_token_index"], errors="coerce").fillna(0.0)})
    x_grid = pd.concat([state, prefix], axis=1).astype(float)
    y_all = grid_helper.labels_to_idx(full["top_action"])
    row_to_pos = {int(row_id): pos for pos, row_id in enumerate(full["row_id"].astype(int).tolist())}
    meta = full[["row_id", "decision_id", "trajectory_id", "prefix_token_index", "top_action"]].copy()

    val_frames = {spec.name: load_prediction(base_dir, spec, "validation") for spec in MODEL_SPECS}
    test_frames = {spec.name: load_prediction(base_dir, spec, "test") for spec in MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))
    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        val_ids, val_rows, val_stack, y_val = activation_frame(val_frames, split_id, meta)
        test_ids, test_rows, test_stack, y_test = activation_frame(test_frames, split_id, meta)
        val_pos = np.asarray([row_to_pos[int(row_id)] for row_id in val_ids], dtype=np.int64)
        test_pos = np.asarray([row_to_pos[int(row_id)] for row_id in test_ids], dtype=np.int64)
        train_mask = np.ones(len(full), dtype=bool)
        train_mask[val_pos] = False
        train_mask[test_pos] = False
        train_pos = np.flatnonzero(train_mask)
        for topk in topks:
            weights = activation_class_weights(val_stack, y_val, topk)
            q_val_act = aggregate_activation(val_stack, weights)
            q_test_act = aggregate_activation(test_stack, weights)
            for grid_name, model in grid_models(args.seed).items():
                model.fit(x_grid.iloc[train_pos], y_all[train_pos])
                q_val_grid = align_proba(model.predict_proba(x_grid.iloc[val_pos]), model.classes_)
                q_test_grid = align_proba(model.predict_proba(x_grid.iloc[test_pos]), model.classes_)
                gw, tw, val_bal = choose_blend(val_rows, q_val_act, q_val_grid, y_val, grid_weights, transition_weights)
                q_test = normalize_probs((1.0 - gw) * q_test_act + gw * q_test_grid)
                pred = viterbi_decode(test_rows, q_test, tw)
                row = {
                    "split_id": int(split_id),
                    "topk": int(topk),
                    "grid_model": grid_name,
                    "grid_weight": float(gw),
                    "transition_weight": float(tw),
                    "validation_balanced_accuracy": float(val_bal),
                    **metric_dict(y_test, pred),
                    "n_train_grid": int(len(train_pos)),
                    "n_val": int(len(val_ids)),
                    "n_test": int(len(test_ids)),
                }
                metric_rows.append(row)
                pred_df = test_rows[["row_id", "decision_id", "prefix_token_index", "top_action"]].copy()
                pred_df.insert(0, "split_id", int(split_id))
                pred_df.insert(1, "topk", int(topk))
                pred_df.insert(2, "grid_model", grid_name)
                pred_df["true_top_action"] = idx_to_labels(y_test)
                pred_df["pred_top_action"] = idx_to_labels(pred)
                for i, action in enumerate(ACTIONS):
                    pred_df[f"prob_{action}"] = q_test[:, i]
                pred_frames.append(pred_df)
                print(f"RESULT split={split_id} topk={topk} grid={grid_name} bal={row['balanced_accuracy']:.6f} acc={row['accuracy']:.6f} gw={gw} tw={tw}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["topk", "grid_model"], as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
            mean_grid_weight=("grid_weight", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    selected_by_validation = summary.sort_values(["mean_validation_balanced_accuracy", "mean_balanced_accuracy"], ascending=[False, False]).head(1)
    metrics.to_csv(output_dir / "grid_activation_blend_metrics.csv", index=False)
    summary.to_csv(output_dir / "grid_activation_blend_summary.csv", index=False)
    selected_by_validation.to_csv(output_dir / "grid_activation_blend_validation_selected_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "grid_activation_blend_predictions.parquet", index=False)
    lines = [
        "# Grid + Activation Top-Action Blend",
        "",
        "Grid classifiers are trained inside each outer split excluding validation and test rows. Blend and Viterbi weights are selected on validation only.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Validation-Selected Method",
        "",
        selected_by_validation.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    save_markdown(output_dir / "grid_activation_blend_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(output_dir / "grid_activation_blend_summary.csv"), "metrics": str(output_dir / "grid_activation_blend_metrics.csv"), "validation_selected": str(output_dir / "grid_activation_blend_validation_selected_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
