#!/usr/bin/env python3
"""Build a supervised linear-probe score manifold from nested OOF stacker scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS, EPS

Q_COLS = [f"q_{a}" for a in ACTIONS]
STACKER_COLS = [f"stacker_score_{a}" for a in ACTIONS]
LEGACY_SCORE_COLS = [f"score_{a}" for a in ACTIONS]
SOURCE_PREFIXES = ["mean100", "prefixmean", "segment", "pairwise"]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
VERTICES = {
    "UP": (0.0, 1.0),
    "DOWN": (0.0, -1.0),
    "LEFT": (-1.0, 0.0),
    "RIGHT": (1.0, 0.0),
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)


def find_prediction_file(stacker_dir: Path) -> Path | None:
    candidates = [
        stacker_dir / "nested_oof_source_stacker_predictions.parquet",
        stacker_dir / "nested_oof_source_stacker_detailed_predictions.parquet",
        stacker_dir / "oof_source_stacker_predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(stacker_dir.glob("*predictions*.parquet"))
    return found[0] if found else None


def read_predictions(stacker_dir: Path, require_oof: bool) -> pd.DataFrame:
    pred_path = find_prediction_file(stacker_dir)
    if pred_path is None:
        message = (
            f"No nested OOF prediction parquet found in {stacker_dir}.\n"
            "Re-run script 50 with prediction export, for example:\n"
            "python scripts/conceptual_belief_space/50_nested_oof_source_stacker.py "
            "--output_dir outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3 "
            "--split_ids all --meta_config_names logreg_raw_C0p1,logreg_raw_C1,ridge_raw_alpha1 "
            "--epochs 8 --n_inner_splits 3 --save_predictions\n"
            "The existing raw-only summary has fold-level metrics but not per-row held-out score coordinates."
        )
        if require_oof:
            raise FileNotFoundError(message)
        raise FileNotFoundError(message)
    df = pd.read_parquet(pred_path)
    df.attrs["prediction_path"] = str(pred_path)
    return df


def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not all(c in df.columns for c in STACKER_COLS):
        if all(c in df.columns for c in LEGACY_SCORE_COLS):
            for action in ACTIONS:
                df[f"stacker_score_{action}"] = df[f"score_{action}"]
        else:
            missing = [c for c in STACKER_COLS if c not in df.columns]
            raise ValueError(f"Missing stacker score columns: {missing}")
    return df


def maybe_join_beliefs(df: pd.DataFrame, beliefs_path: Path) -> pd.DataFrame:
    needed = [*Q_COLS, "trajectory_id", "decision_id", "prefix_token_index", "progress", "agent_action"]
    missing = [c for c in needed if c not in df.columns]
    if not missing:
        return df
    if not beliefs_path.exists():
        raise FileNotFoundError(f"Missing columns {missing} and belief table does not exist: {beliefs_path}")
    beliefs = pd.read_parquet(beliefs_path)
    join_keys: list[str]
    if "row_id" in df.columns and "row_id" in beliefs.columns:
        join_keys = ["row_id"]
    elif "prefix_id" in df.columns and "prefix_id" in beliefs.columns:
        join_keys = ["prefix_id"]
    else:
        join_keys = ["decision_id", "prefix_token_index"]
    extra_cols = [c for c in beliefs.columns if c not in df.columns or c in join_keys]
    merged = df.merge(beliefs[extra_cols].drop_duplicates(join_keys), on=join_keys, how="left", suffixes=("", "_belief"))
    still_missing = [c for c in needed if c not in merged.columns]
    if still_missing:
        raise ValueError(f"Could not recover required belief columns after joining {beliefs_path}: {still_missing}")
    return merged


def action_indices(values: Iterable[str]) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return pd.Series(values).astype(str).map(mapping).to_numpy(dtype=np.float64)


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
    correct = 0
    total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x[mask], y[mask]).statistic)
    except Exception:
        return float(pd.Series(x[mask]).rank().corr(pd.Series(y[mask]).rank()))


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: pd.DataFrame) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    cov = covariates.copy()
    for col in cov.columns:
        cov[col] = pd.to_numeric(cov[col], errors="coerce")
        mask &= np.isfinite(cov[col].to_numpy(dtype=float))
    if mask.sum() < len(cov.columns) + 4:
        return float("nan")
    rx = pd.Series(x[mask]).rank().to_numpy(dtype=float)
    ry = pd.Series(y[mask]).rank().to_numpy(dtype=float)
    ranked_cov = np.column_stack([pd.Series(cov[col].to_numpy(dtype=float)[mask]).rank().to_numpy(dtype=float) for col in cov.columns])
    rx = (rx - rx.mean()) / np.clip(rx.std(), EPS, None)
    ry = (ry - ry.mean()) / np.clip(ry.std(), EPS, None)
    ranked_cov = (ranked_cov - ranked_cov.mean(axis=0, keepdims=True)) / np.clip(ranked_cov.std(axis=0, keepdims=True), EPS, None)

    def residualize_vector(values: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        residual = values.astype(float).copy()
        # Repeated one-dimensional projections are stable for the tiny covariate
        # sets used here and avoid ill-conditioned matrix inversions.
        for _ in range(2):
            for col_idx in range(cov_matrix.shape[1]):
                cov_vec = cov_matrix[:, col_idx]
                denom = float(np.dot(cov_vec, cov_vec))
                if denom > EPS:
                    residual = residual - (float(np.dot(residual, cov_vec)) / denom) * cov_vec
            residual = residual - residual.mean()
        return residual

    res_x = residualize_vector(rx, ranked_cov)
    res_y = residualize_vector(ry, ranked_cov)
    if np.std(res_x) < EPS or np.std(res_y) < EPS:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def branch_separation(df: pd.DataFrame) -> float:
    centroids = []
    within = []
    for action in ACTIONS:
        part = df.loc[df["true_q_top_action"].astype(str).eq(action), ["linear_x", "linear_y"]].dropna()
        if len(part) < 2:
            continue
        arr = part.to_numpy(dtype=float)
        centroid = arr.mean(axis=0)
        centroids.append(centroid)
        within.extend(np.linalg.norm(arr - centroid, axis=1).tolist())
    if len(centroids) < 2 or not within:
        return float("nan")
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(dists) / np.clip(np.mean(within), EPS, None))


def silhouette(df: pd.DataFrame, label_col: str, seed: int = 42) -> float:
    if label_col not in df.columns:
        return float("nan")
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return float("nan")
    valid = df[["linear_x", "linear_y", label_col]].dropna()
    if valid[label_col].nunique() < 2 or len(valid) < 10:
        return float("nan")
    if len(valid) > 10000:
        valid = valid.sample(10000, random_state=seed)
    coords = valid[["linear_x", "linear_y"]].to_numpy(dtype=float)
    coords = (coords - coords.mean(axis=0, keepdims=True)) / np.clip(coords.std(axis=0, keepdims=True), EPS, None)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(silhouette_score(coords, valid[label_col].astype(str).to_numpy()))


def compute_manifold(df: pd.DataFrame, stacker_dir: Path) -> pd.DataFrame:
    scores = df[STACKER_COLS].to_numpy(dtype=np.float64)
    centered = scores - scores.mean(axis=1, keepdims=True)
    for i, action in enumerate(ACTIONS):
        df[f"s_{action}"] = scores[:, i]
        df[f"s_{action}_c"] = centered[:, i]
    df["linear_x"] = df["s_RIGHT_c"] - df["s_LEFT_c"]
    df["linear_y"] = df["s_UP_c"] - df["s_DOWN_c"]
    df["score_radius"] = np.sqrt(df["linear_x"].to_numpy(dtype=float) ** 2 + df["linear_y"].to_numpy(dtype=float) ** 2)
    order = np.argsort(scores, axis=1)
    df["score_margin"] = scores[np.arange(len(df)), order[:, -1]] - scores[np.arange(len(df)), order[:, -2]]
    df["pred_q_top"] = np.asarray(ACTIONS, dtype=object)[order[:, -1]]

    probs = stable_softmax(scores)
    for i, action in enumerate(ACTIONS):
        df[f"p_{action}"] = probs[:, i]
    df["simplex_x"] = df["p_RIGHT"] - df["p_LEFT"]
    df["simplex_y"] = df["p_UP"] - df["p_DOWN"]
    df["pred_top_commitment"] = probs.max(axis=1)
    pred_entropy = -np.sum(probs * np.log(np.clip(probs, EPS, None)), axis=1)
    df["pred_entropy_commitment"] = 1.0 - pred_entropy / math.log(4)

    q = df[Q_COLS].to_numpy(dtype=np.float64)
    q = q / np.clip(q.sum(axis=1, keepdims=True), EPS, None)
    true_idx = q.argmax(axis=1)
    df["true_q_top_action"] = np.asarray(ACTIONS, dtype=object)[true_idx]
    df["top_commitment"] = q.max(axis=1)
    entropy = -np.sum(q * np.log(np.clip(q, EPS, None)), axis=1)
    df["entropy"] = entropy
    df["entropy_commitment"] = 1.0 - entropy / math.log(4)
    if "agent_action" in df.columns:
        agent_idx = action_indices(df["agent_action"])
        commitment = np.full(len(df), np.nan)
        valid = np.isfinite(agent_idx)
        commitment[valid] = q[np.arange(len(df))[valid], agent_idx[valid].astype(int)]
        df["commitment_to_final"] = commitment

    df["h1"] = (df["s_UP_c"] - df["s_DOWN_c"]) / math.sqrt(2)
    df["h2"] = (df["s_LEFT_c"] - df["s_RIGHT_c"]) / math.sqrt(2)
    df["h3"] = (df["s_UP_c"] + df["s_DOWN_c"] - df["s_LEFT_c"] - df["s_RIGHT_c"]) / 2.0
    if "oof_score" not in df.columns:
        df["oof_score"] = ~df.get("in_sample", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    if "in_sample" not in df.columns:
        df["in_sample"] = ~df["oof_score"].fillna(False).astype(bool)
    df["score_manifold_source_dir"] = str(stacker_dir)
    df["score_manifold_interpretation"] = "supervised_linear_probe_induced_oof_score_manifold"
    return df


def build_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    main = df.loc[~df["in_sample"].fillna(False).astype(bool)].copy()
    if main.empty:
        main = df.copy()
    y_true = action_indices(main["true_q_top_action"])
    y_pred = action_indices(main["pred_q_top"])
    q = main[Q_COLS].to_numpy(dtype=np.float64)
    scores = main[[f"s_{a}" for a in ACTIONS]].to_numpy(dtype=np.float64)
    metrics: dict[str, float | int | str] = {
        "n_rows": int(len(main)),
        "n_trajectories": int(main["trajectory_id"].nunique()) if "trajectory_id" in main.columns else 0,
        "n_decisions": int(main["decision_id"].nunique()) if "decision_id" in main.columns else 0,
        "balanced_accuracy_pred_q_top": balanced_accuracy(y_true.astype(int), y_pred.astype(int)),
        "accuracy_pred_q_top": float((y_true == y_pred).mean()),
        "macro_f1_pred_q_top": macro_f1(y_true.astype(int), y_pred.astype(int)),
        "pairwise_q_order_accuracy": pairwise_accuracy(scores, q),
        "spearman_score_radius_top_commitment": spearman(main["score_radius"].to_numpy(float), main["top_commitment"].to_numpy(float)),
        "spearman_score_radius_entropy_commitment": spearman(main["score_radius"].to_numpy(float), main["entropy_commitment"].to_numpy(float)),
        "spearman_score_radius_progress": spearman(main["score_radius"].to_numpy(float), main["progress"].to_numpy(float)) if "progress" in main.columns else float("nan"),
        "branch_separation_ratio": branch_separation(main),
        "silhouette_by_true_q_top": silhouette(main, "true_q_top_action"),
        "silhouette_by_pred_q_top": silhouette(main, "pred_q_top"),
    }
    for i, action in enumerate(ACTIONS):
        mask = y_true == i
        metrics[f"per_class_recall_{action}"] = float((y_pred[mask] == i).mean()) if mask.any() else float("nan")
    if "progress" in main.columns:
        metrics["partial_spearman_score_radius_entropy_commitment_given_progress"] = partial_spearman(
            main["score_radius"].to_numpy(float), main["entropy_commitment"].to_numpy(float), main[["progress"]]
        )
        cov_cols = [c for c in ["progress", "prefix_token_index", "num_cot_tokens"] if c in main.columns]
        metrics["partial_spearman_score_radius_entropy_commitment_given_progress_prefix_len"] = partial_spearman(
            main["score_radius"].to_numpy(float), main["entropy_commitment"].to_numpy(float), main[cov_cols]
        ) if cov_cols else float("nan")
    high = main[main["top_commitment"] >= 0.75]
    low = main[main["top_commitment"] < 0.5]
    for name, part in [("high_commitment", high), ("low_commitment", low)]:
        if part.empty:
            metrics[f"{name}_balanced_accuracy"] = float("nan")
            metrics[f"{name}_n_rows"] = 0
            continue
        metrics[f"{name}_n_rows"] = int(len(part))
        metrics[f"{name}_balanced_accuracy"] = balanced_accuracy(action_indices(part["true_q_top_action"]).astype(int), action_indices(part["pred_q_top"]).astype(int))
    if "strict_before_any_action_mention" in main.columns:
        strict = main[main["strict_before_any_action_mention"].fillna(False).astype(bool)]
        metrics["strict_pre_action_n_rows"] = int(len(strict))
        if not strict.empty:
            metrics["strict_pre_action_balanced_accuracy"] = balanced_accuracy(action_indices(strict["true_q_top_action"]).astype(int), action_indices(strict["pred_q_top"]).astype(int))
            metrics["strict_pre_action_radius_commitment_spearman"] = spearman(strict["score_radius"].to_numpy(float), strict["entropy_commitment"].to_numpy(float))
    return metrics


def write_report(output_dir: Path, manifold: pd.DataFrame, metrics: dict[str, float | int | str], prediction_path: str | None) -> None:
    lines = [
        "# Linear Probe Score Manifold",
        "",
        "This manifold is supervised and probe-induced. It is not PCA, not UMAP, and not an unsupervised raw activation manifold.",
        "",
        "## Coordinates",
        "",
        "The main coordinates use nested held-out raw-only stacker scores:",
        "",
        "- `x = (s_RIGHT - mean(s)) - (s_LEFT - mean(s))`",
        "- `y = (s_UP - mean(s)) - (s_DOWN - mean(s))`",
        "- `score_radius = sqrt(x^2 + y^2)`",
        "",
        "Softmax simplex coordinates are included only for visualization; the primary score plane is linear in the stacker scores.",
        "",
        "## Inputs",
        "",
        f"- prediction_path: `{prediction_path or 'unknown'}`",
        f"- rows: {metrics.get('n_rows')}",
        f"- trajectories: {metrics.get('n_trajectories')}",
        f"- decisions: {metrics.get('n_decisions')}",
        "",
        "## Main Metrics",
        "",
        pd.DataFrame([metrics]).T.reset_index().rename(columns={"index": "metric", 0: "value"}).to_markdown(index=False),
        "",
        "## Interpretation Notes",
        "",
        "- The main plots should use rows with `in_sample == False`, i.e. outer-fold held-out scores.",
        "- The raw-only version preserves the strongest linearity claim: source probes and the stacker consume only activation-derived scores, not grid/task features or metadata.",
        "- Four-branch structure should be evaluated mostly on high-commitment rows; low-commitment rows are expected to be mixed and closer to the origin.",
        "- If radius correlates with progress, commitment and generation progress are entangled and should be interpreted cautiously.",
        "- If strict pre-action rows separate, this is pre-action recoverable action evidence, not causal commitment.",
        "- Aggregate nested validation crosses 0.7 balanced accuracy in the saved raw-only run, but not every fold necessarily crosses 0.7.",
    ]
    (output_dir / "linear_probe_score_manifold_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stacker_dir", default="outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3")
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/linear_probe_score_manifold")
    p.add_argument("--require_oof", type=parse_bool, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stacker_dir = Path(args.stacker_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_predictions(stacker_dir, args.require_oof)
    prediction_path = df.attrs.get("prediction_path")
    df = ensure_score_columns(df)
    df = maybe_join_beliefs(df, Path(args.beliefs))
    df = compute_manifold(df, stacker_dir)
    if args.require_oof and df["in_sample"].fillna(False).astype(bool).any():
        in_sample = int(df["in_sample"].fillna(False).astype(bool).sum())
        raise RuntimeError(f"Found {in_sample} in-sample rows while --require_oof is true")
    metrics = build_metrics(df)
    df.to_parquet(output_dir / "linear_probe_score_manifold.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "linear_probe_score_manifold_metrics.csv", index=False)
    meta = {
        "prediction_path": prediction_path,
        "stacker_dir": str(stacker_dir),
        "require_oof": bool(args.require_oof),
        "coordinate_definition": {
            "linear_x": "centered_score_RIGHT - centered_score_LEFT",
            "linear_y": "centered_score_UP - centered_score_DOWN",
            "simplex_x": "softmax_score_RIGHT - softmax_score_LEFT",
            "simplex_y": "softmax_score_UP - softmax_score_DOWN",
        },
    }
    (output_dir / "linear_probe_score_manifold_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_report(output_dir, df, metrics, prediction_path)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
