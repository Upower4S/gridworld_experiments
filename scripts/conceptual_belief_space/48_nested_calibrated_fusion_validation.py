#!/usr/bin/env python3
"""Nested train-side calibration for activation-only linear source fusions.

This script is a stricter follow-up to the fixed fusion validator.  For each
outer trajectory split it:

1. Splits the outer training trajectories into a model-training subset and a
   calibration subset.
2. Trains each activation-only linear source on the model-training subset and
   scores the calibration subset.
3. Selects a source group, row-softmax temperature, and class-bias vector using
   only calibration trajectories.
4. Retrains each source on the full outer training trajectories.
5. Applies the selected fusion/calibration to the held-out outer test split.

No grid/task features or prefix metadata are model inputs.  Metadata is used only
for row alignment, grouped splits, residual indexing, label construction, and
reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from itertools import product
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

SOURCE_GROUPS = {
    "ovr3": ["mean100_ovr", "prefixmean_ovr", "segment_ovr"],
    "all4": ["mean100_ovr", "identity_pairwise", "prefixmean_ovr", "segment_ovr"],
    "prefix_segment": ["prefixmean_ovr", "segment_ovr"],
    "mean_segment": ["mean100_ovr", "segment_ovr"],
    "mean_prefix": ["mean100_ovr", "prefixmean_ovr"],
    "segment_only": ["segment_ovr"],
    "prefix_only": ["prefixmean_ovr"],
    "mean100_only": ["mean100_ovr"],
}


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_split_ids(text: str, all_ids: list[int]) -> list[int]:
    if text == "all":
        return list(all_ids)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def row_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    z = centered / scale / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def fuse_to_log_scores(source_scores: dict[str, np.ndarray], source_names: list[str], *, temperature: float) -> np.ndarray:
    probs = np.stack([row_softmax(source_scores[name], temperature) for name in source_names], axis=0).mean(axis=0)
    return np.log(np.clip(probs, 1e-12, None))


def centered_bias_grid(values: list[float]) -> np.ndarray:
    seen: set[tuple[float, ...]] = set()
    rows = []
    for vals in product(values, repeat=len(ACTIONS)):
        arr = np.asarray(vals, dtype=np.float32)
        arr = arr - arr.mean()
        key = tuple(np.round(arr, 6).tolist())
        if key not in seen:
            seen.add(key)
            rows.append(arr)
    return np.asarray(rows, dtype=np.float32)


def y_top(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    return fixed.action_indices(df.iloc[indices]["top_action"])


def eval_bal_acc(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    pred = scores.argmax(axis=1)
    return fixed.balanced_accuracy(y_true, pred), float((pred == y_true).mean())


def select_fusion_config(
    source_scores: dict[str, np.ndarray],
    df: pd.DataFrame,
    cal_idx: np.ndarray,
    *,
    temperatures: list[float],
    bias_values: list[float],
) -> dict[str, Any]:
    y = y_top(df, cal_idx)
    q = df.iloc[cal_idx][Q_COLS].to_numpy(dtype=np.float64)
    biases = centered_bias_grid(bias_values)
    best: dict[str, Any] | None = None
    rows = []
    for group_name, names in SOURCE_GROUPS.items():
        if any(name not in source_scores for name in names):
            continue
        for temp in temperatures:
            base = fuse_to_log_scores(source_scores, names, temperature=temp)
            for bias in biases:
                scores = base + bias[None, :]
                bal, acc = eval_bal_acc(scores, y)
                pairwise = fixed.pairwise_accuracy(scores, q)
                row = {
                    "group_name": group_name,
                    "source_names": names,
                    "temperature": float(temp),
                    "bias": bias.tolist(),
                    "calibration_balanced_accuracy": float(bal),
                    "calibration_accuracy": float(acc),
                    "calibration_pairwise_accuracy": float(pairwise),
                }
                rows.append(row)
                if best is None:
                    best = row
                    continue
                key = (bal, acc, pairwise, -float(np.square(bias).sum()), -len(names))
                best_key = (
                    best["calibration_balanced_accuracy"],
                    best["calibration_accuracy"],
                    best["calibration_pairwise_accuracy"],
                    -float(np.square(np.asarray(best["bias"], dtype=np.float32)).sum()),
                    -len(best["source_names"]),
                )
                if key > best_key:
                    best = row
    if best is None:
        raise RuntimeError("No valid fusion config was selectable")
    best = dict(best)
    best["candidate_count"] = len(rows)
    return best


def apply_config(source_scores: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    base = fuse_to_log_scores(source_scores, list(config["source_names"]), temperature=float(config["temperature"]))
    bias = np.asarray(config["bias"], dtype=np.float32)
    return base + bias[None, :]


def train_all_sources(
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
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    out: dict[str, np.ndarray] = {}
    infos: list[dict[str, Any]] = []
    empty_val = np.asarray([], dtype=np.int64)
    for source_name in ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]:
        terms = fixed.SOURCE_CONFIGS[source_name]
        print(f"    source={source_name} train={len(train_idx)} score={len(score_idx)}", flush=True)
        scores, info = fixed.train_source_ovr(
            arrays,
            terms,
            layers,
            df,
            y,
            q_all,
            train_idx,
            empty_val,
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
    print(f"    source=identity_pairwise train={len(train_idx)} score={len(score_idx)}", flush=True)
    pair_scores, info = fixed.train_identity_pairwise(
        arrays,
        layers,
        df,
        q_all,
        train_idx,
        empty_val,
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


def add_metrics(rows: list[dict[str, Any]], df: pd.DataFrame, test_idx: np.ndarray, scores: np.ndarray, *, split_id: int, source_name: str, config: dict[str, Any]) -> None:
    fixed.add_subset_metrics(rows, df, test_idx, scores, split_id=split_id, source_name=source_name)
    for row in rows:
        if row.get("split_id") == split_id and row.get("source_name") == source_name and "selected_group" not in row:
            row["selected_group"] = config.get("group_name")
            row["selected_temperature"] = config.get("temperature")
            row["selected_bias_json"] = json.dumps(config.get("bias"))
            row["calibration_balanced_accuracy"] = config.get("calibration_balanced_accuracy")
            row["calibration_accuracy"] = config.get("calibration_accuracy")
            row["calibration_pairwise_accuracy"] = config.get("calibration_pairwise_accuracy")
            row["candidate_count"] = config.get("candidate_count")


def make_predictions(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, *, split_id: int, source_name: str, config: dict[str, Any]) -> pd.DataFrame:
    out = fixed.make_predictions(df, indices, scores, split_id=split_id, source_name=source_name)
    out["selected_group"] = config.get("group_name")
    out["selected_temperature"] = config.get("temperature")
    out["selected_bias_json"] = json.dumps(config.get("bias"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_nested_calibrated_fusion_validation")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="0")
    p.add_argument("--cal_fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--temperatures", default="0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--bias_values", default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    temperatures = parse_floats(args.temperatures)
    bias_values = parse_floats(args.bias_values)

    df = fixed.rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=None, seed=args.seed, prefix_path=Path(args.prefixes))
    y = fixed.action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    feature_rows = fixed.seg_impl.ms_impl.feature_rows_by_position(df)
    pre_rows = fixed.seg_impl.ms_impl.pre_feature_rows_by_position(df)
    splits = fixed.rank_impl.make_group_splits(df, args.n_splits, args.seed)
    wanted = set(parse_split_ids(args.split_ids, [int(s["split_id"]) for s in splits]))
    splits = [s for s in splits if int(s["split_id"]) in wanted]
    all_terms = sorted({term for terms in fixed.SOURCE_CONFIGS.values() for term in terms})
    arrays = fixed.seg_impl.load_arrays(all_terms, layers)

    manifest = {
        "activation_only": True,
        "method": "nested train-side source-group, temperature, and class-bias calibration",
        "layers": layers,
        "source_groups": SOURCE_GROUPS,
        "temperatures": temperatures,
        "bias_values": bias_values,
        "cal_fraction": args.cal_fraction,
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "note": "Calibration is selected on outer-train calibration trajectories only; test trajectories are not used for selection.",
    }
    (output_dir / "nested_calibrated_fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    source_info_rows: list[dict[str, Any]] = []

    for split in splits:
        split_id = int(split["split_id"])
        outer_train = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        if set(df.iloc[outer_train]["trajectory_id"].astype(str)) & set(df.iloc[test_idx]["trajectory_id"].astype(str)):
            raise RuntimeError(f"Trajectory leakage in split {split_id}")
        subtrain_idx, cal_idx = fixed.seg_impl.ms_impl.train_val_split_by_group(df, outer_train, args.seed + split_id, args.cal_fraction)
        print(f"Outer split {split_id}: subtrain={len(subtrain_idx)} cal={len(cal_idx)} full_train={len(outer_train)} test={len(test_idx)}", flush=True)
        print("  Training calibration sources", flush=True)
        cal_scores, cal_infos = train_all_sources(
            arrays,
            layers,
            df,
            y,
            q_all,
            subtrain_idx,
            cal_idx,
            feature_rows,
            pre_rows,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed + split_id,
        )
        for info in cal_infos:
            source_info_rows.append({"split_id": split_id, "role": "calibration", **info})
        config = select_fusion_config(cal_scores, df, cal_idx, temperatures=temperatures, bias_values=bias_values)
        config_row = {"split_id": split_id, **config}
        selected_rows.append(config_row)
        print("  Selected", json.dumps(config_row), flush=True)

        print("  Training full outer-train sources", flush=True)
        test_scores, test_infos = train_all_sources(
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
            seed=args.seed + 1000 + split_id,
        )
        for info in test_infos:
            source_info_rows.append({"split_id": split_id, "role": "full_train_test", **info})
        fused = apply_config(test_scores, config)
        add_metrics(metric_rows, df, test_idx, fused, split_id=split_id, source_name="nested_calibrated_fusion", config=config)
        if args.save_predictions:
            pred_frames.append(make_predictions(df, test_idx, fused, split_id=split_id, source_name="nested_calibrated_fusion", config=config))

        # Also record the two fixed baselines from the same full-train source scores for apples-to-apples comparison.
        for baseline_name, names in {
            "fixed_temporal_ovr3_fusion_same_run": fixed.TEMPORAL_OVR_FUSION_SOURCES,
            "fixed_temporal_identity_pair_fusion_same_run": fixed.FUSION_SOURCES,
        }.items():
            baseline_config = {
                "group_name": baseline_name,
                "source_names": names,
                "temperature": 1.0,
                "bias": [0.0, 0.0, 0.0, 0.0],
                "calibration_balanced_accuracy": np.nan,
                "calibration_accuracy": np.nan,
                "calibration_pairwise_accuracy": np.nan,
                "candidate_count": 0,
            }
            baseline_scores = apply_config(test_scores, baseline_config)
            add_metrics(metric_rows, df, test_idx, baseline_scores, split_id=split_id, source_name=baseline_name, config=baseline_config)

        pd.DataFrame(metric_rows).to_csv(output_dir / "nested_calibrated_fusion_metrics.partial.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(output_dir / "nested_calibrated_fusion_selected_configs.partial.csv", index=False)
        pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_calibrated_fusion_source_info.partial.csv", index=False)
        print(pd.DataFrame(metric_rows).tail(9).to_string(index=False), flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "nested_calibrated_fusion_metrics.csv", index=False)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "nested_calibrated_fusion_selected_configs.csv", index=False)
    pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_calibrated_fusion_source_info.csv", index=False)
    if args.save_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "nested_calibrated_fusion_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby("source_name", as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "nested_calibrated_fusion_summary.csv", index=False)
    report = [
        "# Nested Calibrated Fusion Validation",
        "",
        "## Scope",
        "",
        "- Source probes are activation-only linear models.",
        "- Source group, row-softmax temperature, and class bias are selected on calibration trajectories from the outer training split only.",
        "- Held-out outer test trajectories are not used for calibration selection.",
        "- Metadata is used only for indexing, grouping, labels, and reports.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows.",
        "",
        "## Selected Configs",
        "",
        selected.to_markdown(index=False) if not selected.empty else "No selected configs.",
    ]
    (output_dir / "nested_calibrated_fusion_report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {output_dir / 'nested_calibrated_fusion_metrics.csv'}")
    print(f"Wrote {output_dir / 'nested_calibrated_fusion_summary.csv'}")
    print(summary.to_string(index=False) if not summary.empty else "No summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
