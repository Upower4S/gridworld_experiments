#!/usr/bin/env python
"""Row-wise validation-trained meta stacker for top_action prediction.

For each outer split this script trains a meta-classifier on the split's
validation rows only, using base activation-model probabilities plus optional
gridworld planning features.  Hyperparameters are selected by an inner grouped
decision split of the validation rows, then the selected model is refit on all
validation rows and evaluated on the held-out test rows.

The target is ``top_action``.  Behavior q columns are never used as input
features.
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
    ModelSpec("layer_10_12_14_16_18_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_10_12_14_16_18"),
    ModelSpec("layer_08_10_12_14_16_h512_qonly", "top_action_valpred_layer_windows_h512_qonly_full", "mean_08_10_12_14_16"),
    ModelSpec("layer_10_12_14_16_18_20_h512_qonly", "top_action_valpred_layer_10_20_h512_qonly_full"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--canonical", default="outputs/canonical_steps.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_rowwise_meta_stacker")
    p.add_argument("--feature_sets", default="probs,probs_grid,probs_grid_raw")
    p.add_argument("--models", default="logreg,hgbdt,xgb_d3,xgb_d4,et500")
    p.add_argument("--transition_weights", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--inner_holdout_frac", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def load_grid_helper():
    path = ROOT / "scripts" / "conceptual_belief_space" / "31_gridworld_state_top_action_baseline.py"
    spec = importlib.util.spec_from_file_location("gridworld_state_top_action_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
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
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[needed].copy()
    df["split_id"] = df["split_id"].astype(int)
    df["row_id"] = df["row_id"].astype(np.int64)
    if df.empty:
        raise ValueError(f"empty prediction table for {spec.name} {role}")
    return df


def labels_to_idx(labels) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def normalize_probs(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, None)
    return q / q.sum(axis=1, keepdims=True)


def entropy(q: np.ndarray) -> np.ndarray:
    q = normalize_probs(q)
    return -np.sum(q * np.log(q), axis=1) / np.log(q.shape[1])


def margin(q: np.ndarray) -> np.ndarray:
    q = normalize_probs(q)
    part = np.partition(q, -2, axis=1)
    return part[:, -1] - part[:, -2]


def split_table(frames: dict[str, pd.DataFrame], split_id: int, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    common: set[int] | None = None
    parts: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        part = df[df["split_id"] == split_id].copy()
        rows = set(part["row_id"].astype(int))
        common = rows if common is None else common & rows
        parts[name] = part
    if not common:
        raise ValueError(f"no common rows for split {split_id}")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    rows_df = pd.DataFrame({"row_id": row_ids}).merge(meta, on="row_id", how="left", validate="one_to_one")
    probs = {}
    labels = None
    for name, part in parts.items():
        part = part[part["row_id"].isin(row_ids)].sort_values("row_id")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"row_id order mismatch for {name} split {split_id}")
        arr = normalize_probs(part[PROB_COLS].to_numpy(dtype=np.float64))
        probs[name] = arr
        labels = part["true_top_action"].to_numpy() if labels is None else labels
    return row_ids, rows_df, probs, labels_to_idx(labels)


def activation_summary_features(probs: dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    stack = np.stack([probs[name] for name in probs], axis=0)
    mean_q = normalize_probs(stack.mean(axis=0))
    std_q = stack.std(axis=0)
    max_q = stack.max(axis=0)
    min_q = stack.min(axis=0)
    for name, q in probs.items():
        cols = {f"{name}_p_{a}": q[:, i] for i, a in enumerate(ACTIONS)}
        cols[f"{name}_entropy"] = entropy(q)
        cols[f"{name}_margin"] = margin(q)
        cols[f"{name}_top"] = q.argmax(axis=1).astype(float)
        frames.append(pd.DataFrame(cols))
    frames.append(pd.DataFrame({f"mean_p_{a}": mean_q[:, i] for i, a in enumerate(ACTIONS)}))
    frames.append(pd.DataFrame({f"std_p_{a}": std_q[:, i] for i, a in enumerate(ACTIONS)}))
    frames.append(pd.DataFrame({f"max_p_{a}": max_q[:, i] for i, a in enumerate(ACTIONS)}))
    frames.append(pd.DataFrame({f"min_p_{a}": min_q[:, i] for i, a in enumerate(ACTIONS)}))
    frames.append(pd.DataFrame({"mean_entropy": entropy(mean_q), "mean_margin": margin(mean_q)}))
    return pd.concat(frames, axis=1).astype(float)


def grid_models(seed: int):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "rf300": RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "et400": ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgb_d4_500": XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0, min_child_weight=2),
        "xgb_d5_400": XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, min_child_weight=3),
        "hgbdt": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=250, max_leaf_nodes=31, l2_regularization=0.05, random_state=seed),
    }


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(ACTIONS)), dtype=np.float64)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, j]
    return normalize_probs(out)


def fit_grid_probs(
    x_grid: pd.DataFrame,
    y_all: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_parts = []
    test_parts = []
    for name, model in grid_models(seed).items():
        model.fit(x_grid.iloc[train_pos], y_all[train_pos])
        qv = align_proba(model.predict_proba(x_grid.iloc[val_pos]), model.classes_)
        qt = align_proba(model.predict_proba(x_grid.iloc[test_pos]), model.classes_)
        val_parts.append(pd.DataFrame({f"grid_{name}_p_{a}": qv[:, i] for i, a in enumerate(ACTIONS)}))
        test_parts.append(pd.DataFrame({f"grid_{name}_p_{a}": qt[:, i] for i, a in enumerate(ACTIONS)}))
        val_parts.append(pd.DataFrame({f"grid_{name}_entropy": entropy(qv), f"grid_{name}_margin": margin(qv)}))
        test_parts.append(pd.DataFrame({f"grid_{name}_entropy": entropy(qt), f"grid_{name}_margin": margin(qt)}))
    return pd.concat(val_parts, axis=1).astype(float), pd.concat(test_parts, axis=1).astype(float)


def context_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for col in ["progress", "prefix_token_index", "num_cot_tokens", "first_action_mention_token_index", "first_final_action_mention_token_index"]:
        if col in rows.columns:
            out[col] = pd.to_numeric(rows[col], errors="coerce").fillna(-1.0).astype(float)
    if "prefix_token_index" in rows.columns and "num_cot_tokens" in rows.columns:
        idx = pd.to_numeric(rows["prefix_token_index"], errors="coerce").fillna(0.0)
        n = pd.to_numeric(rows["num_cot_tokens"], errors="coerce").replace(0, np.nan)
        out["remaining_frac"] = ((n - idx) / n).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.reset_index(drop=True)


def make_feature_set(name: str, base: pd.DataFrame, grid_probs: pd.DataFrame, grid_raw: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    parts = [base.reset_index(drop=True), context_features(rows)]
    if name in {"probs_grid", "probs_grid_raw"}:
        parts.append(grid_probs.reset_index(drop=True))
    if name == "probs_grid_raw":
        parts.append(grid_raw.reset_index(drop=True))
    x = pd.concat(parts, axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32)


def meta_model(name: str, seed: int):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"))
    if name == "hgbdt":
        return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=300, max_leaf_nodes=31, l2_regularization=0.03, random_state=seed)
    if name == "xgb_d3":
        return XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=350, max_depth=3, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, reg_lambda=4.0, min_child_weight=2)
    if name == "xgb_d4":
        return XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed, n_estimators=450, max_depth=4, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0, min_child_weight=2)
    if name == "et500":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown meta model {name}")


def predict_proba_aligned(model, x: pd.DataFrame) -> np.ndarray:
    q = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = list(model.named_steps.values())[-1].classes_
    if classes is None:
        return normalize_probs(q)
    return align_proba(q, np.asarray(classes))


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


def viterbi_decode(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float) -> np.ndarray:
    if transition_weight <= 0:
        return probs.argmax(axis=1)
    q = normalize_probs(probs)
    logp = np.log(q)
    pred = np.zeros(len(rows), dtype=np.int64)
    order = rows[["decision_id", "prefix_token_index"]].copy()
    order["_pos"] = np.arange(len(rows), dtype=np.int64)
    order = order.sort_values(["decision_id", "prefix_token_index", "_pos"])
    eye = np.eye(len(ACTIONS), dtype=np.float64)
    for _, group in order.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        dp = emit[0].copy()
        back = np.zeros_like(emit, dtype=np.int64)
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


def inner_decision_split(rows: pd.DataFrame, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    decisions = np.asarray(sorted(rows["decision_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(decisions)
    n_hold = min(len(decisions) - 1, max(1, int(round(len(decisions) * frac))))
    hold = set(decisions[:n_hold])
    is_hold = rows["decision_id"].astype(str).isin(hold).to_numpy()
    return np.flatnonzero(~is_hold), np.flatnonzero(is_hold)


def select_meta_config(
    x_val_sets: dict[str, pd.DataFrame],
    rows_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_sets: list[str],
    model_names: list[str],
    transition_weights: list[float],
    seed: int,
    inner_holdout_frac: float,
) -> dict[str, Any]:
    inner_train, inner_hold = inner_decision_split(rows_val, inner_holdout_frac, seed)
    best: dict[str, Any] = {"inner_balanced_accuracy": -np.inf}
    for feature_set in feature_sets:
        x = x_val_sets[feature_set]
        for model_name in model_names:
            model = meta_model(model_name, seed)
            model.fit(x.iloc[inner_train], y_val[inner_train])
            q_hold = predict_proba_aligned(model, x.iloc[inner_hold])
            for tw in transition_weights:
                pred = viterbi_decode(rows_val.iloc[inner_hold].reset_index(drop=True), q_hold, tw)
                metrics = metric_dict(y_val[inner_hold], pred)
                bal = metrics["balanced_accuracy"]
                if bal > best["inner_balanced_accuracy"]:
                    best = {
                        "feature_set": feature_set,
                        "model": model_name,
                        "transition_weight": float(tw),
                        "inner_balanced_accuracy": float(bal),
                        "inner_accuracy": metrics["accuracy"],
                        "inner_macro_f1": metrics["macro_f1"],
                    }
    return best


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    base_dir = Path(args.base_dir)
    feature_sets = parse_list(args.feature_sets)
    model_names = parse_list(args.models)
    transition_weights = parse_float_list(args.transition_weights)

    grid_helper = load_grid_helper()
    belief_cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "top_action",
        "num_cot_tokens",
        "first_action_mention_token_index",
        "first_final_action_mention_token_index",
    ]
    available = pd.read_parquet(args.beliefs).columns.tolist()
    belief_cols = [c for c in belief_cols if c in available]
    beliefs = pd.read_parquet(args.beliefs, columns=belief_cols).copy()
    beliefs["row_id"] = beliefs["row_id"].astype(np.int64)
    canonical = pd.read_parquet(args.canonical, columns=["trajectory_id", "step_id", "grid_state_text", "carrying_key"])
    canonical["step_id"] = canonical["step_id"].astype(beliefs["step_id"].dtype)
    full = beliefs.merge(canonical, on=["trajectory_id", "step_id"], how="left", validate="many_to_one").reset_index(drop=True)
    if full["grid_state_text"].isna().any():
        raise ValueError("missing grid_state_text after canonical join")
    grid_raw_all = pd.DataFrame([grid_helper.state_features(row) for _, row in full.iterrows()]).astype(np.float32)
    x_grid_all = pd.concat(
        [
            grid_raw_all.reset_index(drop=True),
            full[["progress", "prefix_token_index"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True),
        ],
        axis=1,
    ).astype(np.float32)
    y_all = labels_to_idx(full["top_action"])
    row_to_pos = {int(row_id): pos for pos, row_id in enumerate(full["row_id"].astype(int).tolist())}
    meta = full[[c for c in belief_cols if c in full.columns]].copy()

    val_frames = {spec.name: load_prediction(base_dir, spec, "validation") for spec in MODEL_SPECS}
    test_frames = {spec.name: load_prediction(base_dir, spec, "test") for spec in MODEL_SPECS}
    splits = sorted(set.intersection(*(set(df["split_id"].unique()) for df in test_frames.values())))

    metrics_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for split_id in splits:
        val_ids, val_rows, val_probs, y_val = split_table(val_frames, split_id, meta)
        test_ids, test_rows, test_probs, y_test = split_table(test_frames, split_id, meta)
        val_pos = np.asarray([row_to_pos[int(r)] for r in val_ids], dtype=np.int64)
        test_pos = np.asarray([row_to_pos[int(r)] for r in test_ids], dtype=np.int64)
        train_mask = np.ones(len(full), dtype=bool)
        train_mask[val_pos] = False
        train_mask[test_pos] = False
        train_pos = np.flatnonzero(train_mask)

        base_val = activation_summary_features(val_probs)
        base_test = activation_summary_features(test_probs)
        grid_val, grid_test = fit_grid_probs(x_grid_all, y_all, train_pos, val_pos, test_pos, args.seed + int(split_id))
        raw_val = grid_raw_all.iloc[val_pos].reset_index(drop=True)
        raw_test = grid_raw_all.iloc[test_pos].reset_index(drop=True)
        x_val_sets = {
            fs: make_feature_set(fs, base_val, grid_val, raw_val, val_rows)
            for fs in feature_sets
        }
        x_test_sets = {
            fs: make_feature_set(fs, base_test, grid_test, raw_test, test_rows)
            for fs in feature_sets
        }

        best = select_meta_config(
            x_val_sets,
            val_rows,
            y_val,
            feature_sets,
            model_names,
            transition_weights,
            args.seed + int(split_id) * 17,
            args.inner_holdout_frac,
        )
        model = meta_model(best["model"], args.seed + int(split_id) * 101)
        model.fit(x_val_sets[best["feature_set"]], y_val)
        q_test = predict_proba_aligned(model, x_test_sets[best["feature_set"]])
        pred_test = viterbi_decode(test_rows.reset_index(drop=True), q_test, best["transition_weight"])
        metrics = metric_dict(y_test, pred_test)
        row = {
            "split_id": int(split_id),
            **best,
            **metrics,
            "n_validation_rows": int(len(y_val)),
            "n_test_rows": int(len(y_test)),
        }
        metrics_rows.append(row)
        pred = test_rows.copy()
        pred["split_id"] = int(split_id)
        pred["true_top_action"] = idx_to_labels(y_test)
        pred["pred_top_action"] = idx_to_labels(pred_test)
        for i, action in enumerate(ACTIONS):
            pred[f"prob_{action}"] = q_test[:, i]
        pred["selected_feature_set"] = best["feature_set"]
        pred["selected_model"] = best["model"]
        pred["selected_transition_weight"] = best["transition_weight"]
        pred_frames.append(pred)
        print(
            json.dumps(
                {
                    "split_id": int(split_id),
                    "feature_set": best["feature_set"],
                    "model": best["model"],
                    "tw": best["transition_weight"],
                    "inner_bal": best["inner_balanced_accuracy"],
                    "test_bal": metrics["balanced_accuracy"],
                    "test_acc": metrics["accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    metrics_path = out_dir / "rowwise_meta_stacker_metrics.csv"
    pred_path = out_dir / "rowwise_meta_stacker_predictions.parquet"
    summary_path = out_dir / "rowwise_meta_stacker_summary.csv"
    report_path = out_dir / "37_rowwise_meta_stacker_report.md"
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_parquet(pred_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "source": "rowwise_meta_stacker_validation_selected",
                "method": "inner-validation-selected rowwise meta-classifier",
                "balanced_accuracy": float(metrics_df["balanced_accuracy"].mean()),
                "accuracy": float(metrics_df["accuracy"].mean()),
                "macro_f1": float(metrics_df["macro_f1"].mean()),
                "notes": "Each outer split trains meta model on validation rows only; model/features/transition selected by inner decision-group split.",
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    lines = [
        "# Row-wise Meta Stacker Report",
        "",
        "This run trains a row-wise meta-classifier on each outer split's validation rows only.",
        "Behavior q columns and target-derived fields are not used as inputs.",
        "",
        f"- Mean balanced accuracy: {summary.loc[0, 'balanced_accuracy']:.6f}",
        f"- Mean accuracy: {summary.loc[0, 'accuracy']:.6f}",
        f"- Mean macro F1: {summary.loc[0, 'macro_f1']:.6f}",
        "",
        "## Split Results",
        "",
        metrics_df[["split_id", "feature_set", "model", "transition_weight", "inner_balanced_accuracy", "balanced_accuracy", "accuracy", "macro_f1"]].to_markdown(index=False),
        "",
        "## Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Predictions: `{pred_path}`",
        f"- Summary: `{summary_path}`",
    ]
    save_markdown(report_path, "\n".join(lines))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
