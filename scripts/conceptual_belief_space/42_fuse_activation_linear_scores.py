#!/usr/bin/env python3
"""Leakage-conservative fusion of activation-only linear score predictions.

This is a diagnostic for the activation-rank goal.  It combines saved OOF scores
from activation-only linear probes without fitting a second-stage model on
labels, so it avoids the nested-stacking leakage issue where meta-train OOF
scores may have been produced by base models trained on the outer test
trajectories.

Primary inputs are only four action scores emitted by existing linear probes.
Metadata is used only for alignment, splits, labels, subsets, and reporting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{a}" for a in ACTIONS]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
META_COLS = [
    "row_id",
    "prefix_id",
    "decision_id",
    "trajectory_id",
    "step_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    "top_action",
    "split_id",
    *Q_COLS,
    *SUBSET_COLS,
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: str
    score_cols: tuple[str, str, str, str]
    filter_col: str | None = None
    filter_value: str | None = None


def default_specs() -> list[SourceSpec]:
    specs = [
        SourceSpec("mean100_raw_ovr", "outputs/conceptual_belief_space/activation_window_mean100_linear_full/mean100_full_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT")),
        SourceSpec("mean50_resid", "outputs/conceptual_belief_space/activation_window_mean50_linear_full/mean50_full_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT"), "mode", "resid_mean50"),
        SourceSpec("mean50_raw_plus_resid", "outputs/conceptual_belief_space/activation_window_mean50_linear_full/mean50_full_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT"), "mode", "raw_plus_resid_mean50"),
        SourceSpec("pairwise_vote_full", "outputs/conceptual_belief_space/activation_pairwise_voting_linear_full/pairwise_voting_linear_full_predictions.parquet", ("vote_score_UP", "vote_score_DOWN", "vote_score_LEFT", "vote_score_RIGHT")),
        SourceSpec("pairwise_vote_cal", "outputs/conceptual_belief_space/activation_pairwise_voting_linear_calibrated/pairwise_voting_calibrated_predictions.parquet", ("calibrated_vote_score_UP", "calibrated_vote_score_DOWN", "calibrated_vote_score_LEFT", "calibrated_vote_score_RIGHT")),
        SourceSpec("ovr_cal", "outputs/conceptual_belief_space/activation_ovr_linear_calibrated/ovr_linear_calibrated_predictions.parquet", ("ovr_score_UP", "ovr_score_DOWN", "ovr_score_LEFT", "ovr_score_RIGHT")),
        SourceSpec("identity_ovr_cal", "outputs/conceptual_belief_space/activation_multilayer_identity_ovr_calibrated/multilayer_identity_ovr_calibrated_predictions.parquet", ("calibrated_ovr_score_UP", "calibrated_ovr_score_DOWN", "calibrated_ovr_score_RIGHT", "calibrated_ovr_score_LEFT")),
        SourceSpec("identity_pairwise", "outputs/conceptual_belief_space/activation_multilayer_identity_linear_full/multilayer_identity_pairwise_predictions.parquet", ("vote_score_UP", "vote_score_DOWN", "vote_score_LEFT", "vote_score_RIGHT")),
        SourceSpec("top_l16_18_delta_mean20", "outputs/conceptual_belief_space/activation_window_top_action_linear_full_l16_18_delta_windows/linear_top_action_predictions.parquet", ("linear_logit_UP", "linear_logit_DOWN", "linear_logit_LEFT", "linear_logit_RIGHT")),
        SourceSpec("top_l10_18_raw_last1", "outputs/conceptual_belief_space/activation_window_top_action_linear_full_l10_18_raw/linear_top_action_predictions.parquet", ("linear_logit_UP", "linear_logit_DOWN", "linear_logit_LEFT", "linear_logit_RIGHT")),
        SourceSpec("rank_long_multipool", "outputs/conceptual_belief_space/activation_window_rank_objective_cached_long_multipool/cached_rank_long_predictions.parquet", ("s_UP", "s_DOWN", "s_LEFT", "s_RIGHT")),
        SourceSpec("rank_concat_views", "outputs/conceptual_belief_space/activation_window_rank_concat_views/concat_views_rank_predictions.parquet", ("s_UP", "s_DOWN", "s_LEFT", "s_RIGHT")),
        SourceSpec("top_l16_18_raw_mean20", "outputs/conceptual_belief_space/activation_window_top_action_linear_full_l16_18_raw_windows/linear_top_action_predictions.parquet", ("linear_logit_UP", "linear_logit_DOWN", "linear_logit_LEFT", "linear_logit_RIGHT"), "pooling", "mean20"),
        SourceSpec("top_l16_18_raw_exp20_tau5", "outputs/conceptual_belief_space/activation_window_top_action_linear_full_l16_18_raw_windows/linear_top_action_predictions.parquet", ("linear_logit_UP", "linear_logit_DOWN", "linear_logit_LEFT", "linear_logit_RIGHT"), "pooling", "exp20_tau5"),
        SourceSpec("cached_multiscale_resid_20_100", "outputs/conceptual_belief_space/activation_cached_multiscale_linear_full/cached_multiscale_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT")),
        SourceSpec("prefixmean_raw_and_resid", "outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear_full/prefixmean_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT")),
        SourceSpec("segment_rawresid_pfx_20_100", "outputs/conceptual_belief_space/activation_segment_combo_linear_full/segment_combo_predictions.parquet", ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT")),
    ]
    # Correct a transposed tuple above defensively before returning.
    fixed = []
    for s in specs:
        if s.name == "identity_ovr_cal":
            fixed.append(SourceSpec(s.name, s.path, ("calibrated_ovr_score_UP", "calibrated_ovr_score_DOWN", "calibrated_ovr_score_LEFT", "calibrated_ovr_score_RIGHT")))
        else:
            fixed.append(s)
    return fixed


def read_columns(path: Path) -> list[str]:
    return pd.read_parquet(path).columns.tolist()


def load_source(spec: SourceSpec, base: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray] | None:
    path = Path(spec.path)
    if not path.exists():
        print(f"Skipping missing source {spec.name}: {path}")
        return None
    columns = read_columns(path)
    required = ["row_id", *spec.score_cols]
    if spec.filter_col:
        required.append(spec.filter_col)
    missing = [c for c in required if c not in columns]
    if missing:
        print(f"Skipping {spec.name}: missing columns {missing}")
        return None
    keep_meta = [c for c in META_COLS if c in columns]
    df = pd.read_parquet(path, columns=list(dict.fromkeys([*keep_meta, *required])))
    if spec.filter_col:
        df = df[df[spec.filter_col].astype(str).eq(str(spec.filter_value))].copy()
    if len(df) != df["row_id"].nunique():
        print(f"Skipping {spec.name}: duplicate row_id after filtering ({len(df)} rows, {df['row_id'].nunique()} unique)")
        return None
    df = df.sort_values("row_id").reset_index(drop=True)
    if base is not None and not np.array_equal(base["row_id"].to_numpy(), df["row_id"].to_numpy()):
        print(f"Skipping {spec.name}: row_id mismatch")
        return None
    scores = df[list(spec.score_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        print(f"Skipping {spec.name}: non-finite scores")
        return None
    return df, scores


def label_indices(values: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for idx in range(len(ACTIONS)):
        mask = y_true == idx
        if mask.any():
            vals.append(float((y_pred[mask] == idx).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for idx in range(len(ACTIONS)):
        tp = float(((y_true == idx) & (y_pred == idx)).sum())
        fp = float(((y_true != idx) & (y_pred == idx)).sum())
        fn = float(((y_true == idx) & (y_pred != idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def exact_order_accuracy(scores: np.ndarray, q: np.ndarray) -> float:
    return float((np.argsort(-scores, axis=1) == np.argsort(-q, axis=1)).all(axis=1).mean())


def top2_set_accuracy(scores: np.ndarray, q: np.ndarray) -> float:
    pred = np.sort(np.argsort(-scores, axis=1)[:, :2], axis=1)
    true = np.sort(np.argsort(-q, axis=1)[:, :2], axis=1)
    return float((pred == true).all(axis=1).mean())


def metrics_for_scores(base: pd.DataFrame, scores: np.ndarray, *, name: str, family: str) -> list[dict[str, Any]]:
    y = label_indices(base["top_action"])
    q = base[Q_COLS].to_numpy(dtype=np.float64)
    out = []
    subsets = {"all_prefixes": np.ones(len(base), dtype=bool)}
    for col in SUBSET_COLS:
        if col in base.columns:
            subsets[col] = base[col].fillna(False).astype(bool).to_numpy()
    pred = scores.argmax(axis=1)
    for subset, mask in subsets.items():
        if not mask.any():
            continue
        row = {
            "fusion_name": name,
            "fusion_family": family,
            "eval_subset": subset,
            "n_rows": int(mask.sum()),
            "accuracy": float((pred[mask] == y[mask]).mean()),
            "balanced_accuracy": balanced_accuracy(y[mask], pred[mask]),
            "macro_f1": macro_f1(y[mask], pred[mask]),
            "pairwise_accuracy": pairwise_accuracy(scores[mask], q[mask]),
            "exact_4_action_order_accuracy": exact_order_accuracy(scores[mask], q[mask]),
            "top2_set_accuracy": top2_set_accuracy(scores[mask], q[mask]),
        }
        for idx, action in enumerate(ACTIONS):
            action_mask = y[mask] == idx
            row[f"recall_{action}"] = float((pred[mask][action_mask] == idx).mean()) if action_mask.any() else float("nan")
        out.append(row)
    return out


def row_z(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=-1, keepdims=True)
    scale = centered.std(axis=-1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return centered / scale


def borda(scores: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(-scores, axis=-1), axis=-1)
    return -ranks.astype(np.float64)


def row_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = row_z(scores) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")].copy()
    all_rows = all_rows.sort_values("balanced_accuracy", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(all_rows["fusion_name"], all_rows["balanced_accuracy"])
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1, label="target 0.7")
    ax.set_xlabel("balanced q-top action accuracy")
    ax.set_title("Activation-only linear score fusion diagnostic")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "linear_score_fusion_balanced_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")].copy().sort_values("pairwise_accuracy", ascending=True).tail(20)
    ax.barh(all_rows["fusion_name"], all_rows["pairwise_accuracy"])
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1, label="0.7")
    ax.set_xlabel("pairwise q-order accuracy")
    ax.set_title("Pairwise rank accuracy from fused activation-only scores")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "linear_score_fusion_pairwise_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")].copy()
    best_bal = all_rows.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_pair = all_rows.sort_values("pairwise_accuracy", ascending=False).iloc[0]
    lines = [
        "# Activation-Only Linear Score Fusion Diagnostic",
        "",
        "## Scope",
        "",
        "- Inputs: saved OOF four-action scores from activation-only linear probes.",
        "- No grid/task features are used.",
        "- Prefix metadata is used only for row alignment, labels, subsets, and reporting.",
        "- No learned second-stage stacker is fitted, because existing OOF score files are not sufficient for leakage-free nested stacking.",
        "- Fusion methods are label-free: row-normalized score averaging and Borda/rank aggregation.",
        "",
        "## Sources",
        "",
    ]
    for source in manifest["loaded_sources"]:
        lines.append(f"- `{source}`")
    if float(best_bal["balanced_accuracy"]) >= 0.7:
        interpretation = (
            "The best label-free score fusion crosses the 0.7 balanced q-top-action threshold. "
            "It is still reported as diagnostic rather than a final no-overfit result: no labels are used to fit fusion weights, "
            "but this fixed source group was identified during exploratory analysis, and a properly nested retraining/selection protocol "
            "would be needed before treating it as the main claim."
        )
    else:
        interpretation = (
            "The best label-free score fusion improves over most individual sources but does not reach the 0.7 balanced q-top-action target. "
            "This should be treated as evidence that simple linear score averaging is not enough; stronger activation-window construction or "
            "properly nested retraining is still needed."
        )
    lines.extend([
        "",
        "## Best All-Prefix Metrics",
        "",
        f"- Best balanced q-top action accuracy: `{best_bal['fusion_name']}` = {best_bal['balanced_accuracy']:.6f}",
        f"- Accuracy: {best_bal['accuracy']:.6f}",
        f"- Macro F1: {best_bal['macro_f1']:.6f}",
        f"- Pairwise q-order accuracy: {best_bal['pairwise_accuracy']:.6f}",
        f"- Best pairwise q-order accuracy: `{best_pair['fusion_name']}` = {best_pair['pairwise_accuracy']:.6f}",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Top Rows",
        "",
        all_rows.sort_values("balanced_accuracy", ascending=False).head(12).to_markdown(index=False),
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_linear_score_fusion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = None
    sources: list[tuple[str, np.ndarray]] = []
    skipped = []
    for spec in default_specs():
        loaded = load_source(spec, base)
        if loaded is None:
            skipped.append(spec.name)
            continue
        df, scores = loaded
        if base is None:
            base = df[[c for c in META_COLS if c in df.columns]].copy()
        sources.append((spec.name, scores))
    if base is None or not sources:
        raise SystemExit("No usable activation-only score sources were loaded")

    metric_rows: list[dict[str, Any]] = []
    pred_frames = []
    for name, scores in sources:
        metric_rows.extend(metrics_for_scores(base, scores, name=name, family="individual_source"))
        pred = base[["row_id", "top_action"]].copy()
        pred["fusion_name"] = name
        for idx, action in enumerate(ACTIONS):
            pred[f"score_{action}"] = scores[:, idx]
        pred["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[scores.argmax(axis=1)]
        pred_frames.append(pred)

    stack = np.stack([scores for _, scores in sources], axis=0)
    fusions: list[tuple[str, str, np.ndarray]] = [
        ("mean_row_z_all", "label_free_score_average", row_z(stack).mean(axis=0)),
        ("mean_row_center_all", "label_free_score_average", (stack - stack.mean(axis=2, keepdims=True)).mean(axis=0)),
        ("borda_all", "label_free_rank_aggregation", borda(stack).mean(axis=0)),
    ]
    # Fixed, documented source groups. These are diagnostics, not selected by fitting labels here.
    groups = {
        "top_action_oriented_core": ["mean100_raw_ovr", "mean50_resid", "mean50_raw_plus_resid", "identity_ovr_cal", "identity_pairwise"],
        "rank_oriented_core": ["mean50_resid", "identity_pairwise", "pairwise_vote_full", "pairwise_vote_cal", "rank_concat_views", "rank_long_multipool"],
        "temporal_segment_core": ["mean100_raw_ovr", "cached_multiscale_resid_20_100", "prefixmean_raw_and_resid", "segment_rawresid_pfx_20_100"],
        "temporal_identity_pair_core": ["mean100_raw_ovr", "identity_pairwise", "prefixmean_raw_and_resid", "segment_rawresid_pfx_20_100"],
        "all_top_temporal_core": ["mean100_raw_ovr", "mean50_resid", "mean50_raw_plus_resid", "identity_ovr_cal", "cached_multiscale_resid_20_100", "prefixmean_raw_and_resid", "segment_rawresid_pfx_20_100"],
    }
    source_map = {name: scores for name, scores in sources}
    for group_name, names in groups.items():
        present = [source_map[n] for n in names if n in source_map]
        if not present:
            continue
        sub = np.stack(present, axis=0)
        fusions.append((f"mean_row_z_{group_name}", "fixed_group_score_average", row_z(sub).mean(axis=0)))
        fusions.append((f"mean_row_softmax_t1_{group_name}", "fixed_group_row_softmax_average", row_softmax(sub, temperature=1.0).mean(axis=0)))
        fusions.append((f"mean_row_softmax_t2_{group_name}", "fixed_group_row_softmax_average", row_softmax(sub, temperature=2.0).mean(axis=0)))
        fusions.append((f"borda_{group_name}", "fixed_group_rank_aggregation", borda(sub).mean(axis=0)))
    for name, family, scores in fusions:
        metric_rows.extend(metrics_for_scores(base, scores, name=name, family=family))
        pred = base[["row_id", "top_action"]].copy()
        pred["fusion_name"] = name
        for idx, action in enumerate(ACTIONS):
            pred[f"score_{action}"] = scores[:, idx]
        pred["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[scores.argmax(axis=1)]
        pred_frames.append(pred)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    manifest = {
        "loaded_sources": [name for name, _ in sources],
        "skipped_sources": skipped,
        "n_rows": int(len(base)),
        "n_sources": int(len(sources)),
        "primary_input": "saved activation-only linear action scores",
        "uses_grid_or_task_features": False,
        "uses_prefix_metadata_as_model_input": False,
        "learned_second_stage": False,
        "leakage_note": "No labels are used to fit fusion weights; learned stacking would require nested retraining of base models.",
    }
    metrics.to_csv(output_dir / "linear_score_fusion_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "linear_score_fusion_predictions.parquet", index=False)
    (output_dir / "linear_score_fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_metrics(metrics, output_dir)
    write_report(output_dir / "linear_score_fusion_report.md", metrics, manifest)
    print(f"Wrote {output_dir / 'linear_score_fusion_metrics.csv'}")
    print(f"Wrote {output_dir / 'linear_score_fusion_predictions.parquet'}")
    print(f"Wrote {output_dir / 'linear_score_fusion_report.md'}")
    best = metrics[metrics.eval_subset.eq("all_prefixes")].sort_values("balanced_accuracy", ascending=False).iloc[0]
    print(f"Best all-prefix balanced accuracy: {best.fusion_name} {best.balanced_accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
