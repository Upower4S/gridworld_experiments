#!/usr/bin/env python3
"""Nested inner-OOF source-score stacker for activation-only linear probes.

This is the strict no-leakage version of the source-score stacker diagnostic.
For each outer trajectory split:

1. Generate source scores for outer-train rows using inner GroupKFold OOF source
   models.  Each outer-train row is scored by a source model that did not train
   on that row.
2. Use only those OOF source scores and outer-train labels to select a small
   linear meta-stacker configuration on a train-side meta-validation split.
3. Refit the selected meta-stacker on all outer-train OOF source scores.
4. Train final source models on all outer-train rows and score held-out rows.
5. Apply the selected meta-stacker to held-out source scores.

Source models are activation-only linear probes over cached activation features.
The meta-stacker input is only source action scores/probabilities produced from
activation-only sources; grid/task features and prefix metadata are excluded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

FIXED_SCRIPT = Path(__file__).with_name("47_nested_fixed_fusion_validation.py")
spec = importlib.util.spec_from_file_location("nested_fixed_fusion_impl", FIXED_SCRIPT)
fixed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixed
assert spec and spec.loader
spec.loader.exec_module(fixed)

Q_COLS = [f"q_{a}" for a in ACTIONS]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
SOURCE_NAMES = ["mean100_ovr", "prefixmean_ovr", "segment_ovr", "identity_pairwise"]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

META_CONFIGS = [
    {"config_name": "logreg_raw_C0p1", "feature_mode": "raw", "model": "logreg", "C": 0.1},
    {"config_name": "logreg_raw_C1", "feature_mode": "raw", "model": "logreg", "C": 1.0},
    {"config_name": "logreg_prob_C0p1", "feature_mode": "prob", "model": "logreg", "C": 0.1},
    {"config_name": "logreg_rowz_C0p1", "feature_mode": "rowz", "model": "logreg", "C": 0.1},
    {"config_name": "logreg_rawprob_C0p1", "feature_mode": "raw_prob", "model": "logreg", "C": 0.1},
    {"config_name": "logreg_rowzprob_C0p1", "feature_mode": "rowz_prob", "model": "logreg", "C": 0.1},
    {"config_name": "ridge_raw_alpha1", "feature_mode": "raw", "model": "ridge", "alpha": 1.0},
]


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_split_ids(text: str, all_ids: list[int]) -> list[int]:
    if text == "all":
        return list(all_ids)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def action_indices(series: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return series.astype(str).map(mapping).to_numpy(dtype=np.int64)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(len(ACTIONS)):
        mask = y_true == i
        if mask.any():
            vals.append(float((y_pred[mask] == i).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for c in range(len(ACTIONS)):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def pairwise_accuracy(scores: np.ndarray, q: np.ndarray, tie_eps: float = 1e-6) -> float:
    correct = total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def row_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    z = centered / scale / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def rowz(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return centered / scale


def stack_source_scores(source_scores: dict[str, np.ndarray], rows: np.ndarray, mode: str) -> np.ndarray:
    parts = []
    for name in SOURCE_NAMES:
        scores = np.asarray(source_scores[name][rows], dtype=np.float64)
        if mode == "raw":
            parts.append(np.clip(scores, -100.0, 100.0))
        elif mode == "rowz":
            parts.append(rowz(scores))
        elif mode == "prob":
            parts.append(row_softmax(scores, 1.0))
        elif mode == "raw_prob":
            parts.append(np.clip(scores, -100.0, 100.0))
            parts.append(row_softmax(scores, 1.0))
        elif mode == "rowz_prob":
            parts.append(rowz(scores))
            parts.append(row_softmax(scores, 1.0))
        else:
            raise ValueError(f"Unknown meta feature mode: {mode}")
    return np.concatenate(parts, axis=1).astype(np.float64, copy=False)


def make_meta_estimator(config: dict[str, Any]):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if config["model"] == "logreg":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", C=float(config.get("C", 1.0)))
    elif config["model"] == "ridge":
        estimator = RidgeClassifier(class_weight="balanced", alpha=float(config.get("alpha", 1.0)))
    else:
        raise ValueError(f"Unknown meta model: {config['model']}")
    return make_pipeline(StandardScaler(), estimator)


def decision_group_splits(df: pd.DataFrame, indices: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    indices = np.asarray(indices, dtype=np.int64)
    groups = df.iloc[indices]["trajectory_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two trajectory groups for grouped splitting")
    if len(unique_groups) < n_splits:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return [(indices[tr], indices[te]) for tr, te in splitter.split(indices, groups=groups)]
    splitter = GroupKFold(n_splits=int(n_splits))
    return [(indices[tr], indices[te]) for tr, te in splitter.split(indices, groups=groups)]


def maybe_source_train_val(df: pd.DataFrame, train_idx: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if fraction <= 0:
        return np.asarray(train_idx, dtype=np.int64), np.asarray([], dtype=np.int64)
    return fixed.seg_impl.ms_impl.train_val_split_by_group(df, train_idx, seed, fraction)


def train_sources_for_score(
    arrays: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    df: pd.DataFrame,
    y: np.ndarray,
    q_all: np.ndarray,
    train_idx: np.ndarray,
    score_idx: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    source_val_fraction: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    out: dict[str, np.ndarray] = {}
    infos: list[dict[str, Any]] = []
    source_train, source_val = maybe_source_train_val(df, np.asarray(train_idx, dtype=np.int64), seed, source_val_fraction)
    for source_name in ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]:
        terms = fixed.SOURCE_CONFIGS[source_name]
        print(f"      source={source_name} train={len(source_train)} val={len(source_val)} score={len(score_idx)}", flush=True)
        scores, info = fixed.train_source_ovr(
            arrays,
            terms,
            layers,
            df,
            y,
            q_all,
            source_train,
            source_val,
            score_idx,
            feature_rows,
            pre_rows,
            source_name=source_name,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
        )
        out[source_name] = scores
        infos.append({"source_name": source_name, **info})
    print(f"      source=identity_pairwise train={len(source_train)} val={len(source_val)} score={len(score_idx)}", flush=True)
    pair_scores, info = fixed.train_identity_pairwise(
        arrays,
        layers,
        df,
        q_all,
        source_train,
        source_val,
        score_idx,
        feature_rows,
        pre_rows,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )
    out["identity_pairwise"] = pair_scores
    infos.append({"source_name": "identity_pairwise", **info})
    return out, infos


def build_inner_oof_scores(
    arrays: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    df: pd.DataFrame,
    y: np.ndarray,
    q_all: np.ndarray,
    outer_train: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    *,
    n_inner_splits: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    source_val_fraction: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(df)
    oof = {name: np.full((n, len(ACTIONS)), np.nan, dtype=np.float32) for name in SOURCE_NAMES}
    infos: list[dict[str, Any]] = []
    inner_splits = decision_group_splits(df, outer_train, n_inner_splits, seed)
    for inner_id, (inner_train, inner_val) in enumerate(inner_splits):
        print(f"    inner {inner_id}: train={len(inner_train)} val={len(inner_val)}", flush=True)
        scores, src_infos = train_sources_for_score(
            arrays,
            layers,
            df,
            y,
            q_all,
            inner_train,
            inner_val,
            feature_rows,
            pre_rows,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + inner_id,
            source_val_fraction=source_val_fraction,
        )
        for name, score in scores.items():
            oof[name][inner_val] = score.astype(np.float32)
        for info in src_infos:
            infos.append({"inner_id": inner_id, **info})
    missing = {name: int(np.isnan(vals[outer_train]).any(axis=1).sum()) for name, vals in oof.items()}
    bad = {k: v for k, v in missing.items() if v}
    if bad:
        raise RuntimeError(f"OOF source scores missing for some outer-train rows: {bad}")
    return oof, infos


def select_meta_config(
    df: pd.DataFrame,
    y: np.ndarray,
    source_scores: dict[str, np.ndarray],
    outer_train: np.ndarray,
    *,
    seed: int,
    meta_val_fraction: float,
    meta_configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if meta_val_fraction > 0:
        meta_train, meta_val = fixed.seg_impl.ms_impl.train_val_split_by_group(df, outer_train, seed, meta_val_fraction)
    else:
        # Selection on all outer-train rows is less strict; retained for debugging only.
        meta_train = outer_train
        meta_val = outer_train
    rows = []
    for config in meta_configs:
        x_train = stack_source_scores(source_scores, meta_train, config["feature_mode"])
        x_val = stack_source_scores(source_scores, meta_val, config["feature_mode"])
        model = make_meta_estimator(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y[meta_train])
        pred = model.predict(x_val)
        rows.append({
            **config,
            "meta_train_rows": int(len(meta_train)),
            "meta_val_rows": int(len(meta_val)),
            "meta_val_accuracy": float((pred == y[meta_val]).mean()),
            "meta_val_balanced_accuracy": balanced_accuracy(y[meta_val], pred),
            "meta_val_macro_f1": macro_f1(y[meta_val], pred),
        })
    candidates = pd.DataFrame(rows).sort_values(["meta_val_balanced_accuracy", "meta_val_accuracy"], ascending=False)
    selected = candidates.iloc[0].to_dict()
    return selected, candidates


def fit_meta_and_decision_scores(config: dict[str, Any], source_scores_train: dict[str, np.ndarray], source_scores_test: dict[str, np.ndarray], train_idx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    # For test score dicts, rows are stored in local order.  Convert to a temporary row index space.
    x_train = stack_source_scores(source_scores_train, train_idx, config["feature_mode"])
    local_rows = np.arange(next(iter(source_scores_test.values())).shape[0], dtype=np.int64)
    x_test = stack_source_scores(source_scores_test, local_rows, config["feature_mode"])
    model = make_meta_estimator(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y[train_idx])
    pred = model.predict(x_test)
    if hasattr(model[-1], "decision_function"):
        scores = model.decision_function(x_test)
    elif hasattr(model[-1], "predict_proba"):
        scores = np.log(np.clip(model.predict_proba(x_test), 1e-12, None))
    else:
        scores = np.zeros((len(pred), len(ACTIONS)), dtype=np.float64)
        scores[np.arange(len(pred)), pred] = 1.0
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        raise RuntimeError("Unexpected binary decision_function output")
    return pred.astype(np.int64), scores, model


def add_metric_rows(rows: list[dict[str, Any]], df: pd.DataFrame, test_idx: np.ndarray, pred: np.ndarray, scores: np.ndarray, *, split_id: int, config: dict[str, Any]) -> None:
    sub = df.iloc[test_idx]
    y_true = action_indices(sub["top_action"])
    q = sub[Q_COLS].to_numpy(dtype=np.float64)
    subsets = {"all_prefixes": np.ones(len(test_idx), dtype=bool)}
    for col in SUBSET_COLS:
        if col in sub.columns:
            subsets[col] = sub[col].fillna(False).astype(bool).to_numpy()
    for subset, mask in subsets.items():
        if not mask.any():
            continue
        row = {
            "source_name": "nested_oof_source_score_stacker",
            "split_id": int(split_id),
            "eval_subset": subset,
            "n_rows": int(mask.sum()),
            "accuracy": float((pred[mask] == y_true[mask]).mean()),
            "balanced_accuracy": balanced_accuracy(y_true[mask], pred[mask]),
            "macro_f1": macro_f1(y_true[mask], pred[mask]),
            "pairwise_accuracy": pairwise_accuracy(scores[mask], q[mask]),
            "selected_config": config["config_name"],
            "selected_feature_mode": config["feature_mode"],
            "selected_model": config["model"],
            "selected_C": config.get("C", np.nan),
            "selected_alpha": config.get("alpha", np.nan),
            "meta_val_balanced_accuracy": config.get("meta_val_balanced_accuracy"),
            "meta_val_accuracy": config.get("meta_val_accuracy"),
        }
        for i, action in enumerate(ACTIONS):
            am = y_true[mask] == i
            row[f"recall_{action}"] = float((pred[mask][am] == i).mean()) if am.any() else float("nan")
        rows.append(row)


def make_predictions(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    split_id: int,
    config: dict[str, Any],
    test_source_scores: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return held-out outer-fold prediction rows with source and stacker scores.

    These rows are suitable for downstream supervised score-manifold plots.  Each
    row is scored only by models trained outside that row's outer trajectory
    group, so ``in_sample`` is always False for this export.
    """
    sub = df.iloc[test_idx].copy()
    stacker_scores = np.asarray(scores, dtype=np.float64)
    for i, action in enumerate(ACTIONS):
        sub[f"stacker_score_{action}"] = stacker_scores[:, i]
        # Backward-compatible alias used by older smoke exports.
        sub[f"score_{action}"] = stacker_scores[:, i]

    score_order = np.argsort(stacker_scores, axis=1)
    top_idx = score_order[:, -1]
    second_idx = score_order[:, -2]
    pred_labels = np.asarray(ACTIONS, dtype=object)[pred]
    sub["predicted_q_top_action"] = pred_labels
    sub["predicted_top_action"] = pred_labels
    sub["score_margin"] = stacker_scores[np.arange(len(sub)), top_idx] - stacker_scores[np.arange(len(sub)), second_idx]
    sub["outer_fold_id"] = int(split_id)
    sub["split_id"] = int(split_id)
    sub["split_role"] = "test"
    sub["in_sample"] = False
    sub["oof_score"] = True
    sub["source_name"] = "nested_oof_source_score_stacker"
    sub["selected_config"] = config["config_name"]
    sub["selected_feature_mode"] = config["feature_mode"]

    if all(c in sub.columns for c in Q_COLS):
        q = sub[Q_COLS].to_numpy(dtype=np.float64)
        true_idx = np.argmax(q, axis=1)
        sub["true_q_top_action"] = np.asarray(ACTIONS, dtype=object)[true_idx]
        correct_count = np.zeros(len(sub), dtype=np.float64)
        valid_count = np.zeros(len(sub), dtype=np.float64)
        for i, j in PAIR_INDICES:
            ai, aj = ACTIONS[i], ACTIONS[j]
            valid = np.abs(q[:, i] - q[:, j]) > 1e-6
            correct = (stacker_scores[:, i] > stacker_scores[:, j]) == (q[:, i] > q[:, j])
            sub[f"pairwise_valid_{ai}_{aj}"] = valid
            sub[f"pairwise_correct_{ai}_{aj}"] = valid & correct
            correct_count += (valid & correct).astype(np.float64)
            valid_count += valid.astype(np.float64)
        sub["pairwise_q_order_accuracy"] = np.divide(correct_count, valid_count, out=np.full(len(sub), np.nan), where=valid_count > 0)

    source_prefixes = {
        "mean100_ovr": "mean100",
        "prefixmean_ovr": "prefixmean",
        "segment_ovr": "segment",
        "identity_pairwise": "pairwise",
    }
    if test_source_scores:
        for source_name, prefix in source_prefixes.items():
            if source_name not in test_source_scores:
                continue
            source_arr = np.asarray(test_source_scores[source_name], dtype=np.float64)
            if source_arr.shape[0] != len(sub):
                raise RuntimeError(f"Source score row count mismatch for {source_name}: {source_arr.shape[0]} != {len(sub)}")
            for i, action in enumerate(ACTIONS):
                sub[f"{prefix}_score_{action}"] = source_arr[:, i]

    metadata_cols = [
        "row_id", "prefix_id", "decision_id", "trajectory_id", "step_id",
        "prefix_token_index", "progress", "num_cot_tokens", "agent_action",
        "top_action", "true_q_top_action", "predicted_q_top_action", "predicted_top_action",
        "outer_fold_id", "split_id", "split_role", "in_sample", "oof_score",
        "source_name", "selected_config", "selected_feature_mode", "score_margin",
        "pairwise_q_order_accuracy", "first_action_mention_token_index",
        "first_final_action_mention_token_index", "strict_before_any_action_mention",
        "strict_before_final_action_mention", "buffered_before_any_action_mention",
        "buffered_before_final_action_mention", "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    score_cols = []
    for prefix in ["stacker", "score", "mean100", "prefixmean", "segment", "pairwise"]:
        score_cols.extend([f"{prefix}_score_{action}" for action in ACTIONS])
    pairwise_cols = [c for c in sub.columns if c.startswith("pairwise_valid_") or c.startswith("pairwise_correct_")]
    keep = [c for c in [*metadata_cols, *Q_COLS, *score_cols, *pairwise_cols] if c in sub.columns]
    return sub[keep]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_nested_oof_source_stacker")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="0")
    p.add_argument("--n_inner_splits", type=int, default=3)
    p.add_argument("--meta_val_fraction", type=float, default=0.15)
    p.add_argument("--source_val_fraction", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--meta_config_names", default="all", help="Comma-separated config_name filter, or all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    df = fixed.rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=None, seed=args.seed, prefix_path=Path(args.prefixes))
    y = action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    feature_rows = fixed.seg_impl.ms_impl.feature_rows_by_position(df)
    pre_rows = fixed.seg_impl.ms_impl.pre_feature_rows_by_position(df)
    splits = fixed.rank_impl.make_group_splits(df, args.n_splits, args.seed)
    wanted = set(parse_split_ids(args.split_ids, [int(s["split_id"]) for s in splits]))
    splits = [s for s in splits if int(s["split_id"]) in wanted]
    all_terms = sorted({term for terms in fixed.SOURCE_CONFIGS.values() for term in terms})
    arrays = fixed.seg_impl.load_arrays(all_terms, layers)
    if args.meta_config_names == "all":
        meta_configs = list(META_CONFIGS)
    else:
        wanted_meta = {x.strip() for x in args.meta_config_names.split(",") if x.strip()}
        meta_configs = [cfg for cfg in META_CONFIGS if cfg["config_name"] in wanted_meta]
        if not meta_configs:
            raise ValueError(f"No meta configs matched {sorted(wanted_meta)}")

    manifest = {
        "activation_only": True,
        "method": "nested inner-OOF source-score linear stacker",
        "layers": layers,
        "source_names": SOURCE_NAMES,
        "meta_configs": meta_configs,
        "n_inner_splits": int(args.n_inner_splits),
        "meta_val_fraction": float(args.meta_val_fraction),
        "source_val_fraction": float(args.source_val_fraction),
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "note": "Meta-stacker selection and fitting use only inner-OOF source scores for outer-train rows.",
    }
    (output_dir / "nested_oof_source_stacker_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    source_info_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    for split in splits:
        split_id = int(split["split_id"])
        outer_train = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        if set(df.iloc[outer_train]["trajectory_id"].astype(str)) & set(df.iloc[test_idx]["trajectory_id"].astype(str)):
            raise RuntimeError(f"Trajectory leakage in outer split {split_id}")
        print(f"Outer split {split_id}: outer_train={len(outer_train)} test={len(test_idx)}", flush=True)
        print("  Building inner OOF source scores", flush=True)
        oof_scores, inner_infos = build_inner_oof_scores(
            arrays,
            layers,
            df,
            y,
            q_all,
            outer_train,
            feature_rows,
            pre_rows,
            n_inner_splits=args.n_inner_splits,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed + split_id * 100,
            source_val_fraction=args.source_val_fraction,
        )
        for info in inner_infos:
            source_info_rows.append({"split_id": split_id, "role": "inner_oof", **info})
        selected, candidates = select_meta_config(df, y, oof_scores, outer_train, seed=args.seed + split_id, meta_val_fraction=args.meta_val_fraction, meta_configs=meta_configs)
        selected["split_id"] = split_id
        selected_rows.append(selected)
        candidates.insert(0, "split_id", split_id)
        candidate_frames.append(candidates)
        print("  Selected meta", json.dumps({k: selected[k] for k in selected if k not in {'config_name'}}, default=str), selected["config_name"], flush=True)
        print("  Training final source models for held-out test", flush=True)
        test_source_scores, final_infos = train_sources_for_score(
            arrays,
            layers,
            df,
            y,
            q_all,
            outer_train,
            test_idx,
            feature_rows,
            pre_rows,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed + 10000 + split_id,
            source_val_fraction=args.source_val_fraction,
        )
        for info in final_infos:
            source_info_rows.append({"split_id": split_id, "role": "final_test", **info})
        pred, scores, _model = fit_meta_and_decision_scores(selected, oof_scores, test_source_scores, outer_train, y)
        add_metric_rows(metric_rows, df, test_idx, pred, scores, split_id=split_id, config=selected)
        if args.save_predictions:
            pred_frames.append(make_predictions(df, test_idx, pred, scores, split_id=split_id, config=selected, test_source_scores=test_source_scores))
        pd.DataFrame(metric_rows).to_csv(output_dir / "nested_oof_source_stacker_metrics.partial.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(output_dir / "nested_oof_source_stacker_selected_configs.partial.csv", index=False)
        pd.concat(candidate_frames, ignore_index=True).to_csv(output_dir / "nested_oof_source_stacker_meta_candidates.partial.csv", index=False)
        pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_oof_source_stacker_source_info.partial.csv", index=False)
        print(pd.DataFrame(metric_rows).tail(3).to_string(index=False), flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "nested_oof_source_stacker_metrics.csv", index=False)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(output_dir / "nested_oof_source_stacker_selected_configs.csv", index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(output_dir / "nested_oof_source_stacker_meta_candidates.csv", index=False)
    pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_oof_source_stacker_source_info.csv", index=False)
    if args.save_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "nested_oof_source_stacker_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby("source_name", as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "nested_oof_source_stacker_summary.csv", index=False)
    report = [
        "# Nested OOF Source-Score Stacker",
        "",
        "## Scope",
        "",
        "- Source models are activation-only linear probes over cached activation combinations.",
        "- Meta-stacker uses only source scores/probabilities, not grid/task features or prefix metadata.",
        "- Inner OOF source scores are generated inside each outer training fold before selecting/fitting the meta-stacker.",
        "- Held-out outer trajectories are not used for source training, meta selection, or meta fitting.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows.",
        "",
        "## Selected Meta Configs",
        "",
        selected_df.to_markdown(index=False) if not selected_df.empty else "No selected configs.",
    ]
    (output_dir / "nested_oof_source_stacker_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False) if not summary.empty else "No summary")
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
