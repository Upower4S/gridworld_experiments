#!/usr/bin/env python
"""Grouped prefix-text classifiers for prefix-level top_action.

This uses prefix text as input, not q/logscore columns. It is a diagnostic
upper-bound style classifier for whether generated reasoning text already
contains enough information to recover the scored top_action.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_prefix_text_classifier")
    p.add_argument("--group_col", default="trajectory_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features_word", type=int, default=180000)
    p.add_argument("--max_features_char", type=int, default=220000)
    p.add_argument("--variants", nargs="+", default=["analysis_word_char", "analysis_tail_word_char", "prompt_analysis_word"])
    return p.parse_args()


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
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_json": json.dumps(mat.tolist()),
    }
    for i, action in enumerate(ACTIONS):
        out[f"recall_{action}"] = float(mat[i, i] / max(mat[i].sum(), 1))
        out[f"pred_count_{action}"] = int((y_pred == i).sum())
        out[f"true_count_{action}"] = int((y_true == i).sum())
    return out


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


def load_table(args: argparse.Namespace) -> pd.DataFrame:
    belief_cols = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "top_action",
        "agent_action",
        "strict_before_any_action_mention",
        "strict_before_final_action_mention",
        "buffered_before_any_action_mention",
        "buffered_before_final_action_mention",
    ]
    beliefs = pd.read_parquet(args.beliefs, columns=belief_cols)
    prefixes = pd.read_parquet(
        args.prefixes,
        columns=[
            "prefix_id",
            "prompt_text",
            "analysis_prefix_text",
            "first_action_mention_token_index",
            "first_final_action_mention_token_index",
        ],
    )
    df = beliefs.merge(prefixes, on="prefix_id", how="left", validate="one_to_one")
    if df["analysis_prefix_text"].isna().any():
        missing = int(df["analysis_prefix_text"].isna().sum())
        raise ValueError(f"missing analysis_prefix_text for {missing} rows")
    df["analysis_prefix_text"] = df["analysis_prefix_text"].fillna("").astype(str)
    df["prompt_text"] = df["prompt_text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def make_text(df: pd.DataFrame, variant: str) -> pd.Series:
    analysis = df["analysis_prefix_text"].fillna("").astype(str)
    if variant == "analysis_word_char":
        return analysis
    if variant == "analysis_tail_word_char":
        return analysis.str[-1200:]
    if variant == "prompt_analysis_word":
        prompt = df["prompt_text"].fillna("").astype(str).str[-1800:]
        return prompt + "\n\nReasoning so far:\n" + analysis.str[-1800:]
    if variant == "analysis_plus_progress":
        progress_bin = pd.cut(df["progress"], bins=np.linspace(0, 1.000001, 11), labels=False, include_lowest=True).fillna(0).astype(int)
        return analysis + "\nPROGRESS_BIN_" + progress_bin.astype(str)
    raise ValueError(f"unknown variant: {variant}")


def build_model(variant: str, args: argparse.Namespace):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.svm import LinearSVC

    if variant == "prompt_analysis_word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=args.max_features_word,
        )
    else:
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=args.max_features_word,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=args.max_features_char,
                    ),
                ),
            ],
            n_jobs=1,
        )
    return make_pipeline(vectorizer, LinearSVC(C=0.7, class_weight="balanced", random_state=args.seed, max_iter=8000))


def evaluate_subsets(df: pd.DataFrame, test_idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    subset_defs = {
        "strict_before_any": "strict_before_any_action_mention",
        "strict_before_final": "strict_before_final_action_mention",
        "buffered_before_any": "buffered_before_any_action_mention",
        "after_any": "strict_before_any_action_mention",
    }
    for name, col in subset_defs.items():
        if col not in df.columns:
            continue
        mask = df.iloc[test_idx][col].fillna(False).astype(bool).to_numpy()
        if name == "after_any":
            mask = ~mask
        if mask.sum() < 20:
            out[f"{name}_n"] = float(mask.sum())
            out[f"{name}_balanced_accuracy"] = float("nan")
            continue
        out[f"{name}_n"] = float(mask.sum())
        out[f"{name}_balanced_accuracy"] = float(metrics_np(y_true[mask], y_pred[mask])["balanced_accuracy"])
    return out


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    df = load_table(args)
    y = labels_to_idx(df["top_action"])
    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "decisions": int(df["decision_id"].nunique()),
                "groups": int(df[args.group_col].nunique()),
                "variants": args.variants,
            },
            indent=2,
        ),
        flush=True,
    )
    metric_rows = []
    pred_frames = []
    for variant in args.variants:
        text = make_text(df, variant)
        for split_id, train_idx, test_idx in grouped_splits(df, args.group_col, args.seed):
            model = build_model(variant, args)
            model.fit(text.iloc[train_idx], y[train_idx])
            pred = model.predict(text.iloc[test_idx])
            m = metrics_np(y[test_idx], pred)
            row = {
                "variant": variant,
                "split_id": int(split_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **m,
                **evaluate_subsets(df, test_idx, y[test_idx], pred),
            }
            metric_rows.append(row)
            pred_frames.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "split_id": int(split_id),
                        "row_id": df.iloc[test_idx]["row_id"].to_numpy(dtype=np.int64),
                        "true_top_action": idx_to_labels(y[test_idx]),
                        "pred_top_action": idx_to_labels(pred),
                    }
                )
            )
            print(
                f"RESULT variant={variant} split={split_id} "
                f"bal={m['balanced_accuracy']:.6f} acc={m['accuracy']:.6f} f1={m['macro_f1']:.6f}",
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("variant", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_strict_before_any_balanced_accuracy=("strict_before_any_balanced_accuracy", "mean"),
            mean_strict_before_final_balanced_accuracy=("strict_before_final_balanced_accuracy", "mean"),
            mean_after_any_balanced_accuracy=("after_any_balanced_accuracy", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    metrics.to_csv(out_dir / "prefix_text_classifier_metrics.csv", index=False)
    summary.to_csv(out_dir / "prefix_text_classifier_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(out_dir / "prefix_text_classifier_predictions.parquet", index=False)
    lines = [
        "# Prefix Text Top-Action Classifier",
        "",
        "Input is generated prefix text. No q/logscore columns, agent_action, or top_action are used as features.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    save_markdown(out_dir / "prefix_text_classifier_report.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "summary": str(out_dir / "prefix_text_classifier_summary.csv"),
                "metrics": str(out_dir / "prefix_text_classifier_metrics.csv"),
                "predictions": str(out_dir / "prefix_text_classifier_predictions.parquet"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
