#!/usr/bin/env python
"""Grid+activation blend with a validation-selected final-proxy penalty.

This is a constrained follow-up to ``35_grid_activation_blend.py``.  It targets
the observed failure mode where early prefixes are over-predicted as the final
decision action.  For each decision, the final action proxy is the predicted
top action at the last measured prefix.  On early prefixes, validation may
select a multiplicative penalty for that proxy action before Viterbi decoding.

The penalty, grid blend weight, and Viterbi transition weight are selected on
validation rows only.  Grid models are trained inside each split after excluding
validation and test rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown


def load_blend35():
    path = ROOT / "scripts" / "conceptual_belief_space" / "35_grid_activation_blend.py"
    spec = importlib.util.spec_from_file_location("grid_activation_blend35", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("grid_activation_blend35", module)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--canonical", default="outputs/canonical_steps.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty")
    p.add_argument("--candidate_pairs", default="9:xgb_d5_400,11:xgb_d5_400,9:xgb_d4_500,11:xgb_d4_500,7:xgb_d4_500,11:rf300")
    p.add_argument("--grid_weights", default="0.2,0.3,0.4,0.5")
    p.add_argument("--weight_modes", default="none", help="Comma-separated grid model sample-weight modes: none,balanced,sqrt_balanced")
    p.add_argument("--cutoffs", default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--penalties", default="1.0,0.75,0.5,0.35,0.25,0.15")
    p.add_argument("--transition_weights", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_candidate_pairs(text: str) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for part in text.split(","):
        if not part.strip():
            continue
        topk, name = part.split(":", 1)
        pairs.append((int(topk), name))
    return pairs


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def fit_grid_model(model, x_train: pd.DataFrame, y_train: np.ndarray, weight_mode: str):
    if weight_mode == "none":
        model.fit(x_train, y_train)
        return model
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weight = compute_sample_weight("balanced", y_train)
    if weight_mode == "sqrt_balanced":
        sample_weight = np.sqrt(sample_weight)
    elif weight_mode != "balanced":
        raise ValueError(f"unknown weight_mode: {weight_mode}")
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def final_proxy_actions(rows: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    tmp = rows[["decision_id", "prefix_token_index"]].copy()
    tmp["_pos"] = np.arange(len(rows), dtype=np.int64)
    tmp["_pred"] = probs.argmax(axis=1)
    proxy_by_decision: dict[Any, int] = {}
    for decision_id, group in tmp.sort_values(["decision_id", "prefix_token_index", "_pos"]).groupby("decision_id", sort=False):
        proxy_by_decision[decision_id] = int(group.iloc[-1]["_pred"])
    return np.asarray([proxy_by_decision[d] for d in rows["decision_id"]], dtype=np.int64)


def apply_final_proxy_penalty(
    rows: pd.DataFrame,
    probs: np.ndarray,
    cutoff: float,
    penalty: float,
    only_if_proxy_is_current_top: bool,
    blend35,
) -> np.ndarray:
    if penalty == 1.0:
        return probs
    out = probs.copy()
    proxy = final_proxy_actions(rows, probs)
    progress = pd.to_numeric(rows["progress"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = progress < cutoff
    if only_if_proxy_is_current_top:
        mask &= probs.argmax(axis=1) == proxy
    idx = np.flatnonzero(mask)
    out[idx, proxy[idx]] *= penalty
    return blend35.normalize_probs(out)


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


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def main() -> None:
    args = parse_args()
    blend35 = load_blend35()
    output_dir = ensure_dir(args.output_dir)
    base_dir = Path(args.base_dir)
    candidates = parse_candidate_pairs(args.candidate_pairs)
    grid_weights = parse_float_list(args.grid_weights)
    weight_modes = parse_str_list(args.weight_modes)
    cutoffs = parse_float_list(args.cutoffs)
    penalties = parse_float_list(args.penalties)
    transition_weights = parse_float_list(args.transition_weights)

    grid_helper = blend35.load_grid_helper()
    beliefs = pd.read_parquet(
        args.beliefs,
        columns=["row_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "top_action"],
    )
    canonical = pd.read_parquet(args.canonical, columns=["trajectory_id", "step_id", "grid_state_text", "carrying_key"])
    canonical["step_id"] = canonical["step_id"].astype(beliefs["step_id"].dtype)
    full = beliefs.merge(canonical, on=["trajectory_id", "step_id"], how="left", validate="many_to_one").reset_index(drop=True)
    if full["grid_state_text"].isna().any():
        raise ValueError("missing grid_state_text after canonical join")
    state = pd.DataFrame([grid_helper.state_features(row) for _, row in full.iterrows()])
    prefix = pd.DataFrame(
        {
            "progress": pd.to_numeric(full["progress"], errors="coerce").fillna(0.0),
            "prefix_token_index": pd.to_numeric(full["prefix_token_index"], errors="coerce").fillna(0.0),
        }
    )
    x_grid = pd.concat([state, prefix], axis=1).astype(float)
    y_all = grid_helper.labels_to_idx(full["top_action"])
    row_to_pos = {int(row_id): pos for pos, row_id in enumerate(full["row_id"].astype(int).tolist())}
    meta = full[["row_id", "decision_id", "trajectory_id", "prefix_token_index", "progress", "top_action"]].copy()

    val_frames = {spec.name: blend35.load_prediction(base_dir, spec, "validation") for spec in blend35.MODEL_SPECS}
    test_frames = {spec.name: blend35.load_prediction(base_dir, spec, "test") for spec in blend35.MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        val_ids, val_rows, val_stack, y_val = blend35.activation_frame(val_frames, split_id, meta)
        test_ids, test_rows, test_stack, y_test = blend35.activation_frame(test_frames, split_id, meta)
        val_pos = np.asarray([row_to_pos[int(row_id)] for row_id in val_ids], dtype=np.int64)
        test_pos = np.asarray([row_to_pos[int(row_id)] for row_id in test_ids], dtype=np.int64)
        train_mask = np.ones(len(full), dtype=bool)
        train_mask[val_pos] = False
        train_mask[test_pos] = False
        train_pos = np.flatnonzero(train_mask)

        fitted_models = {}
        for grid_name in sorted({name for _, name in candidates}):
            for weight_mode in weight_modes:
                model = blend35.grid_models(args.seed)[grid_name]
                fitted_models[(grid_name, weight_mode)] = fit_grid_model(model, x_grid.iloc[train_pos], y_all[train_pos], weight_mode)

        for topk, grid_name in candidates:
            weights = blend35.activation_class_weights(val_stack, y_val, topk)
            q_val_act = blend35.aggregate_activation(val_stack, weights)
            q_test_act = blend35.aggregate_activation(test_stack, weights)
            for weight_mode in weight_modes:
                model = fitted_models[(grid_name, weight_mode)]
                q_val_grid = blend35.align_proba(model.predict_proba(x_grid.iloc[val_pos]), model.classes_)
                q_test_grid = blend35.align_proba(model.predict_proba(x_grid.iloc[test_pos]), model.classes_)

                best: tuple[float, float, float, float, bool, float] = (-np.inf, 0.0, 0.1, 1.0, False, 0.0)
                for grid_weight in grid_weights:
                    q_val_base = blend35.normalize_probs((1.0 - grid_weight) * q_val_act + grid_weight * q_val_grid)
                    for cutoff in cutoffs:
                        for penalty in penalties:
                            for only_if_top in [False, True]:
                                q_val = apply_final_proxy_penalty(val_rows, q_val_base, cutoff, penalty, only_if_top, blend35)
                                for transition_weight in transition_weights:
                                    pred_val = blend35.viterbi_decode(val_rows, q_val, transition_weight)
                                    bal = metric_dict(y_val, pred_val)["balanced_accuracy"]
                                    if bal > best[0]:
                                        best = (float(bal), float(grid_weight), float(cutoff), float(penalty), bool(only_if_top), float(transition_weight))

                val_bal, grid_weight, cutoff, penalty, only_if_top, transition_weight = best
                q_test_base = blend35.normalize_probs((1.0 - grid_weight) * q_test_act + grid_weight * q_test_grid)
                q_test = apply_final_proxy_penalty(test_rows, q_test_base, cutoff, penalty, only_if_top, blend35)
                pred = blend35.viterbi_decode(test_rows, q_test, transition_weight)
                row = {
                    "split_id": int(split_id),
                    "topk": int(topk),
                    "grid_model": grid_name,
                    "weight_mode": weight_mode,
                    "grid_weight": float(grid_weight),
                    "cutoff": float(cutoff),
                    "penalty": float(penalty),
                    "only_if_proxy_is_current_top": bool(only_if_top),
                    "transition_weight": float(transition_weight),
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
                pred_df.insert(3, "weight_mode", weight_mode)
                pred_df["true_top_action"] = idx_to_labels(y_test)
                pred_df["pred_top_action"] = idx_to_labels(pred)
                for i, action in enumerate(ACTIONS):
                    pred_df[f"prob_{action}"] = q_test[:, i]
                pred_frames.append(pred_df)
                print(
                    f"RESULT split={split_id} topk={topk} grid={grid_name} weight={weight_mode} bal={row['balanced_accuracy']:.6f} "
                    f"acc={row['accuracy']:.6f} gw={grid_weight} cutoff={cutoff} penalty={penalty} tw={transition_weight}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["topk", "grid_model", "weight_mode"], as_index=False)
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
    selected_by_validation = summary.sort_values(
        ["mean_validation_balanced_accuracy", "mean_balanced_accuracy"], ascending=[False, False]
    ).head(1)
    metrics.to_csv(output_dir / "final_proxy_penalty_metrics.csv", index=False)
    summary.to_csv(output_dir / "final_proxy_penalty_summary.csv", index=False)
    selected_by_validation.to_csv(output_dir / "final_proxy_penalty_validation_selected_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "final_proxy_penalty_predictions.parquet", index=False)
    lines = [
        "# Final-Proxy Penalty Grid + Activation Blend",
        "",
        "Grid classifiers are trained inside each outer split excluding validation and test rows. The final-proxy penalty, blend weight, and Viterbi transition weight are selected on validation only.",
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
    save_markdown(output_dir / "final_proxy_penalty_report.md", "\n".join(lines))
    print(json.dumps({"summary": str(output_dir / "final_proxy_penalty_summary.csv"), "metrics": str(output_dir / "final_proxy_penalty_metrics.csv")}, indent=2))


if __name__ == "__main__":
    main()
