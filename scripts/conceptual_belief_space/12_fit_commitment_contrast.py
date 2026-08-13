#!/usr/bin/env python3
"""Fit train-only commitment contrast axes for activation states.

This analysis is intentionally not PCA/UMAP.  For each split, layer,
representation, target, subset, and control variant, it defines a single
high-dimensional low-to-high commitment axis from train-only activation
centroids and projects prefix activations onto that axis.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, EPS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json


ACTION_PROB_COLS = [f"q_{action}" for action in ACTIONS]
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "commitment_contrast"
PROJECTION_DIR = OUTPUT_DIR / "projections"
MODEL_DIR = OUTPUT_DIR / "models"
CONTROL_DIR = OUTPUT_DIR / "controls"
FIGURE_DIR = Path("figures/conceptual_belief_space/commitment_contrast")
AXIS_VARIANTS = [
    "original_action_balanced",
    "progress_matched_action_balanced",
    "shuffled_commitment_control",
    "progress_residualized_feature_control",
]


@dataclass(frozen=True)
class SplitRows:
    split_id: int
    train_row_ids: np.ndarray
    test_row_ids: np.ndarray
    source: str


@dataclass
class AxisFit:
    status: str
    warning: str
    low_indices: np.ndarray
    high_indices: np.ndarray
    low_counts: pd.DataFrame
    high_counts: pd.DataFrame
    shuffled_target: np.ndarray | None = None
    fallback_used: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--fallback_behavior", default=Path("outputs/prefix_commitment_metrics.parquet"))
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--activation_metadata", default=Path("outputs/activation/activation_feature_metadata.parquet"))
    parser.add_argument("--feature_dir", default=Path("outputs/activation/features"))
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--layers", nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--representations", nargs="+", default=["delta_pre", "raw", "unit_delta_pre"])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["entropy_commitment", "top_commitment", "commitment_to_final"],
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all_prefixes", "strict_before_any_action_mention", "strict_before_final_action_mention"],
    )
    parser.add_argument("--split_ids", nargs="+", default=["all"])
    parser.add_argument("--high_threshold", type=float, default=0.9)
    parser.add_argument("--low_threshold", type=float, default=0.4)
    parser.add_argument("--progress_bins", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=3)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_projection_rows_per_plot_hint", type=int, default=50000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sanitize(value: Any) -> str:
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def projection_path(
    projection_dir: Path,
    *,
    split_id: int,
    subset: str,
    target: str,
    axis_variant: str,
    layer: int,
    representation: str,
) -> Path:
    name = (
        f"commitment_projection_split{split_id}_{sanitize(subset)}_{sanitize(target)}_"
        f"{sanitize(axis_variant)}_L{layer:02d}_{sanitize(representation)}.parquet"
    )
    return projection_dir / name


def discover_feature_files(feature_dir: Path) -> dict[tuple[int, str], Path]:
    files: dict[tuple[int, str], Path] = {}
    for path in feature_dir.glob("layer_*_*.npy"):
        rest = path.stem[len("layer_") :] if path.stem.startswith("layer_") else ""
        layer_text, sep, representation = rest.partition("_")
        if sep and layer_text.isdigit() and representation:
            files[(int(layer_text), representation)] = path
    return files


def load_behavior_table(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, list[str]]:
    warnings: list[str] = []
    candidates = [resolve(args.beliefs), resolve(args.fallback_behavior)]
    behavior_path = next((path for path in candidates if path.exists()), None)
    if behavior_path is None:
        raise FileNotFoundError(f"No behavior table found in: {[rel(path) for path in candidates]}")
    df = pd.read_parquet(behavior_path).copy()

    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
        warnings.append("Behavior table had no row_id; created row_id from row order.")
    df["row_id"] = df["row_id"].astype(np.int64)

    if "decision_id" not in df.columns:
        if {"trajectory_id", "step_id"} <= set(df.columns):
            df["decision_id"] = df["trajectory_id"].astype(str) + "::" + df["step_id"].astype(str)
            warnings.append("Created decision_id from trajectory_id and step_id.")
        else:
            raise ValueError("Behavior table needs decision_id or trajectory_id + step_id.")

    missing_q = [col for col in ACTION_PROB_COLS if col not in df.columns]
    if missing_q:
        raise ValueError(f"Behavior table is missing action probability columns: {missing_q}")
    q = df[ACTION_PROB_COLS].astype(float).to_numpy()
    q = np.clip(q, EPS, np.inf)
    sums = q.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(sums[:, 0]) | (sums[:, 0] <= 0)
    if bad.any():
        raise ValueError(f"Found {bad.sum()} rows with invalid q sums.")
    q = q / sums
    df[ACTION_PROB_COLS] = q

    df["top_commitment"] = q.max(axis=1)
    entropy_raw = -np.sum(q * np.log(np.clip(q, EPS, None)), axis=1)
    df["entropy"] = entropy_raw
    df["entropy_commitment"] = 1.0 - entropy_raw / math.log(len(ACTIONS))
    if "agent_action" not in df.columns:
        raise ValueError("Behavior table needs agent_action.")
    agent = df["agent_action"].astype(str).to_numpy()
    final = np.full(len(df), np.nan, dtype=float)
    for i, action in enumerate(ACTIONS):
        final[agent == action] = q[agent == action, i]
    if "commitment_to_final" not in df.columns:
        df["commitment_to_final"] = final
    else:
        df["commitment_to_final"] = pd.to_numeric(df["commitment_to_final"], errors="coerce").fillna(pd.Series(final))

    if "top_action" not in df.columns:
        df["top_action"] = np.asarray(ACTIONS, dtype=object)[np.argmax(q, axis=1)]
    if "progress" not in df.columns:
        raise ValueError("Behavior table needs progress.")
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce").clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(args.progress_bins) + 1)
    progress_bin = pd.cut(df["progress"], bins=edges, labels=False, include_lowest=True)
    df["progress_bin"] = progress_bin.astype("float").fillna(args.progress_bins - 1).astype(int).clip(0, args.progress_bins - 1)

    if "prefix_token_index" not in df.columns:
        df["prefix_token_index"] = 0
        warnings.append("Behavior table had no prefix_token_index; filled with 0.")
    if "num_cot_tokens" not in df.columns:
        by_decision = df.groupby("decision_id")["prefix_token_index"].transform("max")
        df["num_cot_tokens"] = by_decision
        warnings.append("Behavior table had no num_cot_tokens; inferred max prefix_token_index by decision.")

    derive_strict_columns(df, warnings)
    return df, behavior_path, warnings


def derive_strict_columns(df: pd.DataFrame, warnings: list[str]) -> None:
    if "strict_before_any_action_mention" not in df.columns:
        if "first_action_mention_token_index" in df.columns:
            first = pd.to_numeric(df["first_action_mention_token_index"], errors="coerce")
            df["strict_before_any_action_mention"] = (
                pd.to_numeric(df["prefix_token_index"], errors="coerce") < first
            ).fillna(False)
            warnings.append("Derived strict_before_any_action_mention from first_action_mention_token_index.")
        elif "is_before_first_action_mention" in df.columns:
            df["strict_before_any_action_mention"] = df["is_before_first_action_mention"].astype(bool)
            warnings.append("Copied strict_before_any_action_mention from is_before_first_action_mention.")

    if "strict_before_final_action_mention" not in df.columns:
        if "first_final_action_mention_token_index" in df.columns:
            first = pd.to_numeric(df["first_final_action_mention_token_index"], errors="coerce")
            df["strict_before_final_action_mention"] = (
                pd.to_numeric(df["prefix_token_index"], errors="coerce") < first
            ).fillna(False)
            warnings.append(
                "Derived strict_before_final_action_mention from first_final_action_mention_token_index."
            )
        elif "is_before_first_final_action_mention" in df.columns:
            df["strict_before_final_action_mention"] = df["is_before_first_final_action_mention"].astype(bool)
            warnings.append(
                "Copied strict_before_final_action_mention from is_before_first_final_action_mention."
            )


def target_bins(values: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    bins = np.full(len(values), "mid", dtype=object)
    bins[np.asarray(values, dtype=float) <= low_threshold] = "low"
    bins[np.asarray(values, dtype=float) >= high_threshold] = "high"
    return bins


def subset_mask(df: pd.DataFrame, subset: str) -> tuple[np.ndarray | None, str]:
    if subset == "all_prefixes":
        return np.ones(len(df), dtype=bool), ""
    if subset in df.columns:
        return df[subset].fillna(False).astype(bool).to_numpy(), ""
    return None, f"Subset {subset} cannot be derived; column missing."


def integer_like(series: pd.Series) -> bool:
    try:
        values = series.dropna().to_numpy()
        as_int = values.astype(np.int64)
        return np.array_equal(values, as_int)
    except Exception:
        return False


def validate_alignment(
    metadata_path: Path,
    behavior: pd.DataFrame,
    feature_shape_n: int,
    warnings: list[str],
) -> dict[int, int]:
    row_ids = behavior["row_id"].astype(np.int64).to_numpy()
    if metadata_path.exists():
        meta = pd.read_parquet(metadata_path)
        if len(meta) != feature_shape_n:
            raise ValueError(
                f"{rel(metadata_path)} has {len(meta)} rows but feature array has {feature_shape_n} rows."
            )
        key_cols = [col for col in ["trajectory_id", "step_id", "prefix_token_index", "decision_id"] if col in behavior.columns and col in meta.columns]
        if key_cols:
            mismatches: list[str] = []
            for col in key_cols:
                left = behavior[col].astype(str).fillna("<NA>").to_numpy()
                right = meta[col].astype(str).fillna("<NA>").to_numpy()
                if not np.array_equal(left, right):
                    mismatches.append(col)
            if mismatches:
                warnings.append(
                    f"Activation metadata and behavior row order do not exactly match for columns {mismatches}; using row_id/index fallback."
                )
            else:
                warnings.append("Activation metadata row order verified against behavior keys.")
        if "row_id" in meta.columns:
            meta_row_ids = meta["row_id"].astype(np.int64).to_numpy()
            missing = set(row_ids.tolist()) - set(meta_row_ids.tolist())
            if missing:
                raise ValueError(f"Activation metadata missing {len(missing)} behavior row_id values.")
            return {int(row_id): int(i) for i, row_id in enumerate(meta_row_ids)}
        warnings.append("Activation metadata has no row_id; using positional row_id alignment after checks.")

    if len(behavior) == feature_shape_n and integer_like(behavior["row_id"]):
        expected = np.arange(feature_shape_n, dtype=np.int64)
        if np.array_equal(np.sort(row_ids), expected) and np.array_equal(row_ids, expected):
            warnings.append("Using behavior row_id as direct feature row index.")
            return {int(row_id): int(row_id) for row_id in row_ids}
    raise ValueError("Cannot validate activation-feature alignment.")


def read_split_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "splits" in data:
            data = data["splits"]
        else:
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"{rel(path)} must contain a list of splits.")
    return data


def parse_split(split: dict[str, Any], behavior: pd.DataFrame) -> SplitRows:
    split_id = int(split.get("split_id", split.get("fold", 0)))
    if "train_row_ids" in split and "test_row_ids" in split:
        return SplitRows(
            split_id=split_id,
            train_row_ids=np.asarray(split["train_row_ids"], dtype=np.int64),
            test_row_ids=np.asarray(split["test_row_ids"], dtype=np.int64),
            source="train_row_ids",
        )
    if "train_indices" in split and "test_indices" in split:
        return SplitRows(
            split_id=split_id,
            train_row_ids=behavior.iloc[np.asarray(split["train_indices"], dtype=np.int64)]["row_id"].to_numpy(np.int64),
            test_row_ids=behavior.iloc[np.asarray(split["test_indices"], dtype=np.int64)]["row_id"].to_numpy(np.int64),
            source="train_indices",
        )
    if "train_groups" in split and "test_groups" in split and "trajectory_id" in behavior.columns:
        train_groups = set(map(str, split["train_groups"]))
        test_groups = set(map(str, split["test_groups"]))
        train_mask = behavior["trajectory_id"].astype(str).isin(train_groups)
        test_mask = behavior["trajectory_id"].astype(str).isin(test_groups)
        return SplitRows(
            split_id=split_id,
            train_row_ids=behavior.loc[train_mask, "row_id"].to_numpy(np.int64),
            test_row_ids=behavior.loc[test_mask, "row_id"].to_numpy(np.int64),
            source="trajectory_groups",
        )
    if "train_decision_ids" in split and "test_decision_ids" in split and "decision_id" in behavior.columns:
        train_ids = set(map(str, split["train_decision_ids"]))
        test_ids = set(map(str, split["test_decision_ids"]))
        train_mask = behavior["decision_id"].astype(str).isin(train_ids)
        test_mask = behavior["decision_id"].astype(str).isin(test_ids)
        return SplitRows(
            split_id=split_id,
            train_row_ids=behavior.loc[train_mask, "row_id"].to_numpy(np.int64),
            test_row_ids=behavior.loc[test_mask, "row_id"].to_numpy(np.int64),
            source="decision_groups",
        )
    raise ValueError(f"Cannot parse split structure with keys: {sorted(split.keys())}")


def make_generated_split(behavior: pd.DataFrame, output_dir: Path, seed: int) -> SplitRows:
    rng = np.random.default_rng(seed)
    group_col = "trajectory_id" if "trajectory_id" in behavior.columns else "decision_id"
    groups = np.asarray(sorted(behavior[group_col].astype(str).unique()))
    rng.shuffle(groups)
    n_test = max(1, int(round(0.2 * len(groups))))
    test_groups = set(groups[:n_test])
    train_groups = set(groups[n_test:])
    train_mask = behavior[group_col].astype(str).isin(train_groups)
    test_mask = behavior[group_col].astype(str).isin(test_groups)
    split_json = [
        {
            "split_id": 0,
            "group_col": group_col,
            "seed": int(seed),
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
        }
    ]
    write_json(output_dir / "generated_group_splits.json", split_json)
    return SplitRows(
        split_id=0,
        train_row_ids=behavior.loc[train_mask, "row_id"].to_numpy(np.int64),
        test_row_ids=behavior.loc[test_mask, "row_id"].to_numpy(np.int64),
        source="generated_group_split",
    )


def load_splits(args: argparse.Namespace, behavior: pd.DataFrame, output_dir: Path) -> tuple[list[SplitRows], list[str]]:
    warnings: list[str] = []
    split_path = resolve(args.splits)
    if split_path.exists():
        splits = [parse_split(item, behavior) for item in read_split_list(split_path)]
    else:
        splits = [make_generated_split(behavior, output_dir, args.random_seed)]
        warnings.append(f"No existing split file found; generated {rel(output_dir / 'generated_group_splits.json')}.")

    requested = args.split_ids
    if len(requested) == 1 and str(requested[0]).lower() == "all":
        return splits, warnings
    requested_ids = {int(item) for item in requested}
    filtered = [split for split in splits if split.split_id in requested_ids]
    missing = sorted(requested_ids - {split.split_id for split in filtered})
    if missing:
        warnings.append(f"Requested split_ids not available and skipped: {missing}")
    return filtered, warnings


def row_ids_to_positions(behavior: pd.DataFrame, row_ids: np.ndarray) -> np.ndarray:
    mapper = pd.Series(np.arange(len(behavior), dtype=np.int64), index=behavior["row_id"].astype(np.int64))
    out = mapper.reindex(row_ids.astype(np.int64))
    if out.isna().any():
        raise ValueError("Split contains row_ids absent from behavior table.")
    return out.to_numpy(np.int64)


def fit_scaler(features: np.ndarray, fit_indices: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    hidden_dim = int(features.shape[1])
    if not standardize:
        return np.zeros(hidden_dim, dtype=np.float32), np.ones(hidden_dim, dtype=np.float32)
    x = np.asarray(features[fit_indices], dtype=np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def transform_block(x: np.ndarray, mean: np.ndarray, std: np.ndarray, standardize: bool, l2_normalize: bool) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    if standardize:
        out = (out - mean) / std
    out = np.asarray(out, dtype=np.float32)
    if l2_normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        out = np.asarray(out / norms, dtype=np.float32)
    return out


def sample_equal(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or len(indices) == 0:
        return np.asarray([], dtype=np.int64)
    if len(indices) <= n:
        return np.asarray(indices, dtype=np.int64)
    return rng.choice(indices, size=n, replace=False).astype(np.int64)


def action_balanced_fit_indices(
    behavior: pd.DataFrame,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    *,
    min_bin_count: int,
    rng: np.random.Generator,
) -> AxisFit:
    low_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    min_count = math.inf
    cell_indices: dict[tuple[str, str], np.ndarray] = {}
    for action in ACTIONS:
        action_mask = behavior["agent_action"].astype(str).to_numpy() == action
        for bin_name in ["low", "high"]:
            idx = np.flatnonzero(candidate_mask & action_mask & (labels == bin_name))
            cell_indices[(action, bin_name)] = idx
            rows.append({"agent_action": action, "progress_bin": -1, "commitment_bin": bin_name, "count": int(len(idx))})
            min_count = min(min_count, len(idx))
    if not np.isfinite(min_count) or min_count < min_bin_count:
        return AxisFit("too_few_balanced_cells", f"Minimum action x low/high cell count is {min_count}.", np.array([], dtype=np.int64), np.array([], dtype=np.int64), pd.DataFrame(rows), pd.DataFrame())
    for action in ACTIONS:
        low_parts.append(sample_equal(cell_indices[(action, "low")], int(min_count), rng))
        high_parts.append(sample_equal(cell_indices[(action, "high")], int(min_count), rng))
    return AxisFit("ok", "", np.concatenate(low_parts), np.concatenate(high_parts), pd.DataFrame(rows), pd.DataFrame())


def progress_matched_fit_indices(
    behavior: pd.DataFrame,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    *,
    min_bin_count: int,
    rng: np.random.Generator,
) -> AxisFit:
    low_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    progress_bins = sorted(behavior.loc[candidate_mask, "progress_bin"].dropna().astype(int).unique().tolist())
    kept_cells = 0
    for action in ACTIONS:
        action_mask = behavior["agent_action"].astype(str).to_numpy() == action
        for progress_bin in progress_bins:
            pb_mask = behavior["progress_bin"].to_numpy() == int(progress_bin)
            low_idx = np.flatnonzero(candidate_mask & action_mask & pb_mask & (labels == "low"))
            high_idx = np.flatnonzero(candidate_mask & action_mask & pb_mask & (labels == "high"))
            rows.append(
                {
                    "agent_action": action,
                    "progress_bin": int(progress_bin),
                    "commitment_bin": "low",
                    "count": int(len(low_idx)),
                }
            )
            rows.append(
                {
                    "agent_action": action,
                    "progress_bin": int(progress_bin),
                    "commitment_bin": "high",
                    "count": int(len(high_idx)),
                }
            )
            n = min(len(low_idx), len(high_idx))
            if n >= min_bin_count:
                kept_cells += 1
                low_parts.append(sample_equal(low_idx, n, rng))
                high_parts.append(sample_equal(high_idx, n, rng))
    if not low_parts or not high_parts:
        return AxisFit("too_few_progress_matched_cells", "No action x progress_bin cells had enough low/high rows.", np.array([], dtype=np.int64), np.array([], dtype=np.int64), pd.DataFrame(rows), pd.DataFrame())
    return AxisFit("ok", f"Kept {kept_cells} action x progress_bin cells.", np.concatenate(low_parts), np.concatenate(high_parts), pd.DataFrame(rows), pd.DataFrame())


def shuffled_labels(
    behavior: pd.DataFrame,
    train_subset_mask: np.ndarray,
    values: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = values.copy()
    train_idx = np.flatnonzero(train_subset_mask)
    group_cols = ["progress_bin", "agent_action"]
    used = np.zeros(len(values), dtype=bool)
    for _, group in behavior.loc[train_subset_mask].groupby(group_cols, dropna=False):
        idx = group.index.to_numpy(np.int64)
        if len(idx) >= 5:
            shuffled[idx] = rng.permutation(shuffled[idx])
            used[idx] = True
    remaining = train_idx[~used[train_idx]]
    if len(remaining) >= 5:
        for _, group in behavior.loc[remaining].groupby("progress_bin", dropna=False):
            idx = group.index.to_numpy(np.int64)
            if len(idx) >= 2:
                shuffled[idx] = rng.permutation(shuffled[idx])
                used[idx] = True
    remaining = train_idx[~used[train_idx]]
    if len(remaining) >= 2:
        shuffled[remaining] = rng.permutation(shuffled[remaining])
    return target_bins(shuffled, low_threshold, high_threshold)


def design_matrix(behavior: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    cols = ["progress", "prefix_token_index"]
    if "num_cot_tokens" in behavior.columns:
        cols.append("num_cot_tokens")
    data = behavior.iloc[indices][cols].astype(float).to_numpy()
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    stds = np.where(stds < EPS, 1.0, stds)
    scaled = (np.nan_to_num(data, nan=means) - means) / stds
    return np.column_stack([np.ones(len(indices)), scaled]).astype(np.float32)


def fit_residual_beta(x_train: np.ndarray, design_train: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design_train.astype(np.float64), x_train.astype(np.float64), rcond=None)
    return beta.astype(np.float32)


def residualize(x: np.ndarray, design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(x - design @ beta, dtype=np.float32)


def normalize_vector(vec: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < EPS:
        return np.zeros_like(vec, dtype=np.float32), norm
    return np.asarray(vec / norm, dtype=np.float32), norm


def axis_fit_counts_rows(
    *,
    split_id: int,
    subset: str,
    target: str,
    axis_variant: str,
    layer: int,
    representation: str,
    fit: AxisFit,
) -> pd.DataFrame:
    rows = []
    counts = fit.low_counts if fit.low_counts is not None and not fit.low_counts.empty else pd.DataFrame()
    for _, row in counts.iterrows():
        rows.append(
            {
                "split_id": split_id,
                "subset": subset,
                "target": target,
                "axis_variant": axis_variant,
                "layer": layer,
                "representation": representation,
                "agent_action": row.get("agent_action"),
                "progress_bin": int(row.get("progress_bin", -1)),
                "low_high": row.get("commitment_bin"),
                "available_count": int(row.get("count", 0)),
                "selected_count": int(
                    np.sum(
                        (fit.low_indices if row.get("commitment_bin") == "low" else fit.high_indices)
                        >= 0
                    )
                ),
                "status": fit.status,
                "warning": fit.warning,
            }
        )
    if not rows:
        rows.append(
            {
                "split_id": split_id,
                "subset": subset,
                "target": target,
                "axis_variant": axis_variant,
                "layer": layer,
                "representation": representation,
                "agent_action": None,
                "progress_bin": -1,
                "low_high": None,
                "available_count": 0,
                "selected_count": 0,
                "status": fit.status,
                "warning": fit.warning,
            }
        )
    return pd.DataFrame(rows)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) < EPS or np.std(y_arr) < EPS:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan")
    val = spearmanr(x, y, nan_policy="omit").correlation
    return float(val) if val is not None else float("nan")


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    mask = np.isfinite(score)
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], score[mask]))


def balanced_accuracy_binary(y_true: np.ndarray, pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return float("nan")
    mask = np.isfinite(y_true.astype(float))
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true[mask], pred[mask]))


def d_prime(low_scores: np.ndarray, high_scores: np.ndarray) -> float:
    low = low_scores[np.isfinite(low_scores)]
    high = high_scores[np.isfinite(high_scores)]
    if len(low) < 2 or len(high) < 2:
        return float("nan")
    pooled = math.sqrt(0.5 * (float(np.var(low, ddof=1)) + float(np.var(high, ddof=1))))
    if pooled < EPS:
        return float("nan")
    return float((np.mean(high) - np.mean(low)) / pooled)


def partial_spearman(df: pd.DataFrame, score_col: str, target_col: str) -> float:
    cov_cols = [col for col in ["progress", "prefix_token_index", "num_cot_tokens"] if col in df.columns]
    values = df[[score_col, target_col] + cov_cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 5:
        return float("nan")
    y_score = values[score_col].to_numpy()
    y_target = values[target_col].to_numpy()
    design = np.column_stack([np.ones(len(values)), values[cov_cols].to_numpy()])
    beta_score, *_ = np.linalg.lstsq(design, y_score, rcond=None)
    beta_target, *_ = np.linalg.lstsq(design, y_target, rcond=None)
    return spearman(y_score - design @ beta_score, y_target - design @ beta_target)


def within_progress_spearman(df: pd.DataFrame, score_col: str, target_col: str) -> float:
    vals = []
    for _, part in df.groupby("progress_bin", dropna=False):
        if len(part) >= 5:
            vals.append(spearman(part[score_col], part[target_col]))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def metric_scope(df: pd.DataFrame, scope_mask: np.ndarray, *, target: str, low_threshold: float, high_threshold: float, threshold: float) -> dict[str, Any]:
    part = df.loc[scope_mask].copy()
    out: dict[str, Any] = {
        "n_rows": int(len(part)),
        "target_mean": float(part["target_value"].mean()) if len(part) else float("nan"),
        "score_commit_mean": float(part["score_commit"].mean()) if len(part) else float("nan"),
        "spearman_score_target": spearman(part["score_commit"], part["target_value"]) if len(part) else float("nan"),
        "pearson_score_target": pearson(part["score_commit"], part["target_value"]) if len(part) else float("nan"),
        "spearman_score_target_within_progress_bin_mean": within_progress_spearman(part, "score_commit", "target_value") if len(part) else float("nan"),
        "partial_spearman_score_target_resid_progress": partial_spearman(part, "score_commit", "target_value") if len(part) else float("nan"),
        "progress_target_spearman": spearman(part["progress"], part["target_value"]) if len(part) else float("nan"),
        "progress_score_spearman": spearman(part["progress"], part["score_commit"]) if len(part) else float("nan"),
    }
    low_mask = part["target_value"].to_numpy(float) <= low_threshold
    high_mask = part["target_value"].to_numpy(float) >= high_threshold
    lh_mask = low_mask | high_mask
    y = high_mask[lh_mask].astype(int)
    scores = part.loc[lh_mask, "score_commit"].to_numpy(float)
    preds = (scores >= threshold).astype(int)
    out.update(
        {
            "n_low": int(low_mask.sum()),
            "n_high": int(high_mask.sum()),
            "auc_high_vs_low": auc_score(y, scores),
            "balanced_accuracy_high_vs_low": balanced_accuracy_binary(y, preds),
            "d_prime_high_vs_low": d_prime(part.loc[low_mask, "score_commit"].to_numpy(float), part.loc[high_mask, "score_commit"].to_numpy(float)),
            "mean_score_low": float(part.loc[low_mask, "score_commit"].mean()) if low_mask.any() else float("nan"),
            "mean_score_high": float(part.loc[high_mask, "score_commit"].mean()) if high_mask.any() else float("nan"),
        }
    )
    return out


def projection_metrics(
    projection: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    target: str,
    axis_variant: str,
    layer: int,
    representation: str,
    threshold: float,
    low_threshold: float,
    high_threshold: float,
    status: str,
    warning: str,
    action_balance_min_count: int,
    selected_low: int,
    selected_high: int,
    axis_norm: float,
    progress_axis_norm: float,
    projection_file: Path,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "split_id": int(split_id),
        "subset": subset,
        "target": target,
        "axis_variant": axis_variant,
        "layer": int(layer),
        "representation": representation,
        "n_rows": int(len(projection)),
        "n_train": int(projection["is_train"].sum()),
        "n_test": int((~projection["is_train"]).sum()),
        "n_low_test": int(((~projection["is_train"]) & (projection["target_value"] <= low_threshold)).sum()),
        "n_high_test": int(((~projection["is_train"]) & (projection["target_value"] >= high_threshold)).sum()),
        "threshold_used": float(threshold),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "action_balance_min_count": int(action_balance_min_count),
        "selected_low_fit_count": int(selected_low),
        "selected_high_fit_count": int(selected_high),
        "axis_norm": float(axis_norm),
        "progress_axis_norm": float(progress_axis_norm),
        "status": status,
        "warnings": warning,
        "projection_file": rel(projection_file),
    }
    scopes = {
        "test": ~projection["is_train"].to_numpy(bool),
        "train": projection["is_train"].to_numpy(bool),
        "all": np.ones(len(projection), dtype=bool),
    }
    for strict_col in ["strict_before_any_action_mention", "strict_before_final_action_mention"]:
        if strict_col in projection.columns:
            scopes[strict_col] = projection[strict_col].fillna(False).astype(bool).to_numpy()
    for scope, mask in scopes.items():
        vals = metric_scope(
            projection,
            mask,
            target=target,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            threshold=threshold,
        )
        for key, value in vals.items():
            base[f"{scope}_{key}"] = value
    # Compatibility aliases requested in the prompt: use held-out test rows.
    for key in [
        "target_mean",
        "score_commit_mean",
        "spearman_score_target",
        "pearson_score_target",
        "spearman_score_target_within_progress_bin_mean",
        "partial_spearman_score_target_resid_progress",
        "auc_high_vs_low",
        "balanced_accuracy_high_vs_low",
        "d_prime_high_vs_low",
        "mean_score_low",
        "mean_score_high",
        "progress_target_spearman",
        "progress_score_spearman",
    ]:
        base[key if key.endswith("_test") else f"{key}_test"] = base.get(f"test_{key}", float("nan"))
    base["auc_high_vs_low_test"] = base.get("test_auc_high_vs_low", float("nan"))
    base["balanced_accuracy_high_vs_low_test"] = base.get("test_balanced_accuracy_high_vs_low", float("nan"))
    base["d_prime_high_vs_low_test"] = base.get("test_d_prime_high_vs_low", float("nan"))
    base["mean_score_low_test"] = base.get("test_mean_score_low", float("nan"))
    base["mean_score_high_test"] = base.get("test_mean_score_high", float("nan"))
    return base


def compute_progress_axis(
    x_train_subset: np.ndarray,
    train_subset_df: pd.DataFrame,
    u_commit: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    progress = train_subset_df["progress"].to_numpy(float)
    early = progress <= 0.25
    late = progress >= 0.75
    if early.sum() < 3 or late.sum() < 3:
        return np.zeros_like(u_commit, dtype=np.float32), 0.0, "too_few_early_or_late_rows"
    raw = x_train_subset[late].mean(axis=0, dtype=np.float64) - x_train_subset[early].mean(axis=0, dtype=np.float64)
    raw = raw.astype(np.float32)
    orth = raw - float(np.dot(raw, u_commit)) * u_commit
    v, norm = normalize_vector(orth)
    warning = "" if norm >= EPS else "progress axis degenerated after orthogonalization"
    return v, norm, warning


def build_projection_dataframe(
    behavior: pd.DataFrame,
    features: np.ndarray,
    subset_indices: np.ndarray,
    train_positions: set[int],
    fit_low: set[int],
    fit_high: set[int],
    *,
    split_id: int,
    subset: str,
    target: str,
    axis_variant: str,
    layer: int,
    representation: str,
    target_values: np.ndarray,
    commitment_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool,
    l2_normalize: bool,
    u_commit: np.ndarray,
    v_progress: np.ndarray,
    c_mid: np.ndarray,
    score_mean: float,
    score_std: float,
    progress_mean: float,
    progress_std: float,
    residual_mean: float,
    residual_std: float,
    chunk_size: int,
    residual_beta: np.ndarray | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    score_std = score_std if score_std >= EPS else 1.0
    progress_std = progress_std if progress_std >= EPS else 1.0
    residual_std = residual_std if residual_std >= EPS else 1.0

    for start in range(0, len(subset_indices), chunk_size):
        idx = subset_indices[start : start + chunk_size]
        x = transform_block(features[idx], mean, std, standardize, l2_normalize)
        if residual_beta is not None:
            x = residualize(x, design_matrix(behavior, idx), residual_beta)
        centered = x - c_mid
        score = centered @ u_commit
        if np.linalg.norm(v_progress) >= EPS:
            score_progress = centered @ v_progress
        else:
            score_progress = np.full(len(idx), np.nan, dtype=np.float32)
        residual = centered - score[:, None] * u_commit[None, :]
        residual_norm = np.linalg.norm(residual, axis=1).astype(np.float32)
        meta_cols = [
            "row_id",
            "trajectory_id",
            "decision_id",
            "step_id",
            "prefix_token_index",
            "progress",
            "progress_bin",
            "agent_action",
            "top_action",
            *ACTION_PROB_COLS,
            "commitment_to_final",
            "entropy_commitment",
            "top_commitment",
        ]
        optional_cols = [
            "strict_before_any_action_mention",
            "strict_before_final_action_mention",
            "first_action_mention_token_index",
            "first_final_action_mention_token_index",
            "num_cot_tokens",
        ]
        cols = [col for col in meta_cols + optional_cols if col in behavior.columns]
        out = behavior.iloc[idx][cols].copy()
        out.insert(1, "split_id", int(split_id))
        out.insert(2, "subset", subset)
        out.insert(3, "target", target)
        out.insert(4, "axis_variant", axis_variant)
        out.insert(5, "layer", int(layer))
        out.insert(6, "representation", representation)
        out["target_value"] = target_values[idx].astype(np.float32)
        out["commitment_bin"] = commitment_labels[idx]
        out["is_train"] = [int(i) in train_positions for i in idx]
        out["selected_low_fit"] = [int(i) in fit_low for i in idx]
        out["selected_high_fit"] = [int(i) in fit_high for i in idx]
        out["selected_for_axis_fit"] = out["selected_low_fit"] | out["selected_high_fit"]
        out["score_commit"] = score.astype(np.float32)
        out["score_progress_orth"] = score_progress.astype(np.float32)
        out["residual_norm"] = residual_norm.astype(np.float32)
        out["z_score_commit_train_scaled"] = ((score - score_mean) / score_std).astype(np.float32)
        out["z_score_progress_train_scaled"] = ((score_progress - progress_mean) / progress_std).astype(np.float32)
        out["z_residual_norm_train_scaled"] = ((residual_norm - residual_mean) / residual_std).astype(np.float32)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def fit_one_axis(
    behavior: pd.DataFrame,
    features: np.ndarray,
    *,
    split: SplitRows,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    subset: str,
    subset_indices: np.ndarray,
    target: str,
    axis_variant: str,
    layer: int,
    representation: str,
    target_values: np.ndarray,
    commitment_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool,
    l2_normalize: bool,
    chunk_size: int,
    min_bin_count: int,
    low_threshold: float,
    high_threshold: float,
    rng: np.random.Generator,
    projection_file: Path,
    force: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    subset_mask_arr = np.zeros(len(behavior), dtype=bool)
    subset_mask_arr[subset_indices] = True
    train_mask = np.zeros(len(behavior), dtype=bool)
    train_mask[train_positions] = True
    train_subset_mask = subset_mask_arr & train_mask

    labels_for_fit = commitment_labels
    candidate_mask = train_subset_mask
    residual_beta = None
    residual_warning = ""
    if axis_variant == "original_action_balanced":
        fit = action_balanced_fit_indices(behavior, candidate_mask, labels_for_fit, min_bin_count=min_bin_count, rng=rng)
    elif axis_variant == "progress_matched_action_balanced":
        fit = progress_matched_fit_indices(behavior, candidate_mask, labels_for_fit, min_bin_count=min_bin_count, rng=rng)
    elif axis_variant == "shuffled_commitment_control":
        labels_for_fit = shuffled_labels(
            behavior,
            train_subset_mask,
            target_values,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            rng=rng,
        )
        fit = action_balanced_fit_indices(behavior, candidate_mask, labels_for_fit, min_bin_count=min_bin_count, rng=rng)
        fit.shuffled_target = labels_for_fit
    elif axis_variant == "progress_residualized_feature_control":
        fit = action_balanced_fit_indices(behavior, candidate_mask, labels_for_fit, min_bin_count=min_bin_count, rng=rng)
    else:
        raise ValueError(f"Unknown axis variant: {axis_variant}")

    counts = axis_fit_counts_rows(
        split_id=split.split_id,
        subset=subset,
        target=target,
        axis_variant=axis_variant,
        layer=layer,
        representation=representation,
        fit=fit,
    )
    if fit.status != "ok":
        metrics = {
            "split_id": split.split_id,
            "subset": subset,
            "target": target,
            "axis_variant": axis_variant,
            "layer": int(layer),
            "representation": representation,
            "status": fit.status,
            "warnings": fit.warning,
            "projection_file": rel(projection_file),
        }
        return None, metrics, counts, {}

    fit_idx = np.concatenate([fit.low_indices, fit.high_indices]).astype(np.int64)
    x_fit = transform_block(features[fit_idx], mean, std, standardize, l2_normalize)
    x_train_subset = transform_block(features[np.flatnonzero(train_subset_mask)], mean, std, standardize, l2_normalize)
    train_subset_df = behavior.iloc[np.flatnonzero(train_subset_mask)].copy()

    if axis_variant == "progress_residualized_feature_control":
        resid_fit_rows = np.flatnonzero(train_subset_mask)
        x_resid_train = x_train_subset
        design_train = design_matrix(behavior, resid_fit_rows)
        residual_beta = fit_residual_beta(x_resid_train, design_train)
        x_fit = residualize(x_fit, design_matrix(behavior, fit_idx), residual_beta)
        x_train_subset = residualize(x_train_subset, design_train, residual_beta)
        residual_warning = "features residualized against progress/prefix_token_index/num_cot_tokens on train subset"

    low_local = np.arange(len(fit.low_indices))
    high_local = np.arange(len(fit.low_indices), len(fit_idx))
    c_low = x_fit[low_local].mean(axis=0, dtype=np.float64).astype(np.float32)
    c_high = x_fit[high_local].mean(axis=0, dtype=np.float64).astype(np.float32)
    u_commit, axis_norm = normalize_vector(c_high - c_low)
    c_mid = ((c_low + c_high) * 0.5).astype(np.float32)
    raw_low_scores = (x_fit[low_local] - c_mid) @ u_commit
    raw_high_scores = (x_fit[high_local] - c_mid) @ u_commit
    sign_flipped = False
    if np.nanmean(raw_high_scores) < np.nanmean(raw_low_scores):
        u_commit = -u_commit
        raw_low_scores = -raw_low_scores
        raw_high_scores = -raw_high_scores
        sign_flipped = True
    threshold = float(0.5 * (np.nanmean(raw_high_scores) + np.nanmean(raw_low_scores)))

    v_progress, progress_axis_norm, progress_warning = compute_progress_axis(x_train_subset, train_subset_df, u_commit)
    train_centered = x_train_subset - c_mid
    train_scores = train_centered @ u_commit
    if np.linalg.norm(v_progress) >= EPS:
        train_progress_scores = train_centered @ v_progress
    else:
        train_progress_scores = np.full(len(train_centered), np.nan, dtype=np.float32)
    train_residual_norms = np.linalg.norm(train_centered - train_scores[:, None] * u_commit[None, :], axis=1)
    score_mean = float(np.nanmean(train_scores))
    score_std = float(np.nanstd(train_scores))
    progress_mean = float(np.nanmean(train_progress_scores)) if np.isfinite(train_progress_scores).any() else 0.0
    progress_std = float(np.nanstd(train_progress_scores)) if np.isfinite(train_progress_scores).any() else 1.0
    residual_mean = float(np.nanmean(train_residual_norms))
    residual_std = float(np.nanstd(train_residual_norms))

    projection = build_projection_dataframe(
        behavior,
        features,
        subset_indices,
        set(map(int, train_positions)),
        set(map(int, fit.low_indices)),
        set(map(int, fit.high_indices)),
        split_id=split.split_id,
        subset=subset,
        target=target,
        axis_variant=axis_variant,
        layer=layer,
        representation=representation,
        target_values=target_values,
        commitment_labels=commitment_labels,
        mean=mean,
        std=std,
        standardize=standardize,
        l2_normalize=l2_normalize,
        u_commit=u_commit,
        v_progress=v_progress,
        c_mid=c_mid,
        score_mean=score_mean,
        score_std=score_std,
        progress_mean=progress_mean,
        progress_std=progress_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        chunk_size=chunk_size,
        residual_beta=residual_beta,
    )
    ensure_dir(projection_file.parent)
    if projection_file.exists() and not force:
        raise FileExistsError(f"{rel(projection_file)} exists. Pass --force to overwrite.")
    projection.to_parquet(projection_file, index=False)

    warning = "; ".join([w for w in [fit.warning, progress_warning, residual_warning, "sign flipped so high commitment is positive" if sign_flipped else ""] if w])
    metrics = projection_metrics(
        projection,
        split_id=split.split_id,
        subset=subset,
        target=target,
        axis_variant=axis_variant,
        layer=layer,
        representation=representation,
        threshold=threshold,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        status="ok",
        warning=warning,
        action_balance_min_count=int(min(len(fit.low_indices), len(fit.high_indices))),
        selected_low=len(fit.low_indices),
        selected_high=len(fit.high_indices),
        axis_norm=axis_norm,
        progress_axis_norm=progress_axis_norm,
        projection_file=projection_file,
    )
    payload = {
        "u_commit": u_commit.astype(np.float32),
        "v_progress": v_progress.astype(np.float32),
        "c_low": c_low.astype(np.float32),
        "c_high": c_high.astype(np.float32),
        "c_mid": c_mid.astype(np.float32),
        "scaler_mean": mean.astype(np.float32),
        "scaler_std": std.astype(np.float32),
        "residual_beta": np.asarray(residual_beta, dtype=np.float32) if residual_beta is not None else np.asarray([], dtype=np.float32),
        "threshold": np.asarray([threshold], dtype=np.float32),
        "score_train_mean_std": np.asarray([score_mean, score_std], dtype=np.float32),
        "progress_train_mean_std": np.asarray([progress_mean, progress_std], dtype=np.float32),
        "residual_norm_train_mean_std": np.asarray([residual_mean, residual_std], dtype=np.float32),
    }
    return projection, metrics, counts, payload


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    warnings: Sequence[str],
) -> None:
    lines = [
        "# Commitment Contrast Fit Report",
        "",
        "This script fits train-only low-to-high commitment axes in activation space.",
        "",
        "- No PCA/UMAP is used as the primary projection.",
        "- Axes are fit on train trajectories only.",
        "- Action labels are used only for balancing and diagnostics.",
        "- Results are observational, not causal.",
        "",
        "## Run Summary",
        "",
        f"- attempted jobs: {manifest.get('jobs_attempted', 0)}",
        f"- successful jobs: {manifest.get('jobs_successful', 0)}",
        f"- skipped jobs: {manifest.get('jobs_skipped', 0)}",
        f"- behavior rows: {manifest.get('n_behavior_rows', 'NA')}",
        f"- feature rows: {manifest.get('n_feature_rows', 'NA')}",
        "",
    ]
    if not metrics.empty and "auc_high_vs_low_test" in metrics.columns:
        ok = metrics[metrics["status"].astype(str).eq("ok")].copy()
        if not ok.empty:
            top = ok.sort_values("auc_high_vs_low_test", ascending=False).head(10)
            lines.extend(["## Top Held-Out AUC Rows", "", top.to_markdown(index=False), ""])
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
        lines.append("")
    save_markdown(output_dir / "12_commitment_contrast_fit_report.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    projection_dir = output_dir / "projections"
    model_dir = output_dir / "models"
    for path in [
        output_dir,
        projection_dir,
        model_dir,
        output_dir / "controls",
        FIGURE_DIR,
        FIGURE_DIR / "projection_grids",
        FIGURE_DIR / "trajectories",
        FIGURE_DIR / "metrics",
        FIGURE_DIR / "strict_pre_action",
    ]:
        ensure_dir(resolve(path))

    behavior, behavior_path, warnings = load_behavior_table(args)
    feature_dir = resolve(args.feature_dir)
    feature_files = discover_feature_files(feature_dir)
    requested_layers = [int(x) for x in args.layers]
    requested_reps = [str(x) for x in args.representations]
    requested_pairs = [(layer, rep) for layer in requested_layers for rep in requested_reps if (layer, rep) in feature_files]
    missing_pairs = [(layer, rep) for layer in requested_layers for rep in requested_reps if (layer, rep) not in feature_files]
    if missing_pairs:
        warnings.append(f"Skipped unavailable feature files: {missing_pairs[:20]}{'...' if len(missing_pairs) > 20 else ''}")
    if not requested_pairs:
        raise FileNotFoundError("No requested activation feature arrays were found.")

    first_features = np.load(feature_files[requested_pairs[0]], mmap_mode="r")
    row_id_to_feature_idx = validate_alignment(
        resolve(args.activation_metadata),
        behavior,
        int(first_features.shape[0]),
        warnings,
    )
    if len(row_id_to_feature_idx) != len(behavior):
        warnings.append("Row-id map length differs from behavior row count; proceeding with validated subset map.")
    splits, split_warnings = load_splits(args, behavior, output_dir)
    warnings.extend(split_warnings)

    split_ids = [split.split_id for split in splits]
    subsets: list[tuple[str, np.ndarray]] = []
    subset_warnings: list[str] = []
    for subset in args.subsets:
        mask, warning = subset_mask(behavior, subset)
        if mask is None:
            subset_warnings.append(warning)
            continue
        if mask.sum() == 0:
            subset_warnings.append(f"Subset {subset} has zero rows and was skipped.")
            continue
        subsets.append((subset, np.flatnonzero(mask)))
    warnings.extend(subset_warnings)
    targets = [target for target in args.targets if target in behavior.columns]
    missing_targets = [target for target in args.targets if target not in behavior.columns]
    if missing_targets:
        warnings.append(f"Skipped missing targets: {missing_targets}")
    if not targets:
        raise ValueError("No requested targets are available.")

    planned_jobs = len(splits) * len(subsets) * len(requested_pairs) * len(targets) * len(AXIS_VARIANTS)
    manifest: dict[str, Any] = {
        "args": jsonable(vars(args)),
        "python": sys.version,
        "platform": platform.platform(),
        "behavior_table": rel(behavior_path),
        "feature_dir": rel(feature_dir),
        "n_behavior_rows": int(len(behavior)),
        "n_feature_rows": int(first_features.shape[0]),
        "layers": sorted(set(layer for layer, _ in requested_pairs)),
        "representations": sorted(set(rep for _, rep in requested_pairs)),
        "targets": targets,
        "subsets": [name for name, _ in subsets],
        "split_ids": split_ids,
        "axis_variants": AXIS_VARIANTS,
        "planned_jobs": int(planned_jobs),
        "warnings": warnings,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        write_json(output_dir / "dry_run_manifest.json", manifest)
        return

    metrics_rows: list[dict[str, Any]] = []
    count_frames: list[pd.DataFrame] = []
    axis_npz: dict[str, np.ndarray] = {}
    axis_metadata: list[dict[str, Any]] = []
    jobs_attempted = 0
    jobs_successful = 0
    jobs_skipped = 0

    for layer, representation in requested_pairs:
        feature_path = feature_files[(layer, representation)]
        features = np.load(feature_path, mmap_mode="r")
        if features.shape[0] != len(behavior):
            raise ValueError(f"{rel(feature_path)} has {features.shape[0]} rows but behavior has {len(behavior)} rows.")
        print(f"Loaded {rel(feature_path)} shape={features.shape} dtype={features.dtype}")
        for split in splits:
            train_positions = row_ids_to_positions(behavior, split.train_row_ids)
            test_positions = row_ids_to_positions(behavior, split.test_row_ids)
            train_mask = np.zeros(len(behavior), dtype=bool)
            train_mask[train_positions] = True
            for subset, subset_indices in subsets:
                train_subset_indices = np.intersect1d(train_positions, subset_indices, assume_unique=False)
                scaler_fit_indices = train_subset_indices if len(train_subset_indices) >= max(10, args.min_bin_count * len(ACTIONS) * 2) else train_positions
                scaler_fallback = len(scaler_fit_indices) != len(train_subset_indices)
                mean, std = fit_scaler(features, scaler_fit_indices, args.standardize)
                if scaler_fallback:
                    warnings.append(
                        f"Scaler fallback for split={split.split_id}, subset={subset}, layer={layer}, repr={representation}: used all train rows."
                    )
                for target in targets:
                    values = behavior[target].astype(float).to_numpy()
                    labels = target_bins(values, args.low_threshold, args.high_threshold)
                    for axis_variant in AXIS_VARIANTS:
                        jobs_attempted += 1
                        out_path = projection_path(
                            projection_dir,
                            split_id=split.split_id,
                            subset=subset,
                            target=target,
                            axis_variant=axis_variant,
                            layer=layer,
                            representation=representation,
                        )
                        if out_path.exists() and not args.force:
                            jobs_skipped += 1
                            metrics_rows.append(
                                {
                                    "split_id": split.split_id,
                                    "subset": subset,
                                    "target": target,
                                    "axis_variant": axis_variant,
                                    "layer": layer,
                                    "representation": representation,
                                    "status": "skipped_existing",
                                    "warnings": "projection file exists; pass --force to recompute",
                                    "projection_file": rel(out_path),
                                }
                            )
                            continue
                        seed = (
                            int(args.random_seed)
                            + int(split.split_id) * 100000
                            + int(layer) * 1000
                            + hash((representation, subset, target, axis_variant)) % 997
                        )
                        rng = np.random.default_rng(seed)
                        print(
                            f"Processing split={split.split_id} subset={subset} target={target} "
                            f"variant={axis_variant} layer={layer} repr={representation}"
                        )
                        try:
                            projection, metrics, counts, payload = fit_one_axis(
                                behavior,
                                features,
                                split=split,
                                train_positions=train_positions,
                                test_positions=test_positions,
                                subset=subset,
                                subset_indices=subset_indices,
                                target=target,
                                axis_variant=axis_variant,
                                layer=layer,
                                representation=representation,
                                target_values=values,
                                commitment_labels=labels,
                                mean=mean,
                                std=std,
                                standardize=args.standardize,
                                l2_normalize=args.l2_normalize,
                                chunk_size=args.chunk_size,
                                min_bin_count=args.min_bin_count,
                                low_threshold=args.low_threshold,
                                high_threshold=args.high_threshold,
                                rng=rng,
                                projection_file=out_path,
                                force=args.force,
                            )
                            count_frames.append(counts)
                            metrics_rows.append(metrics)
                            if payload:
                                jobs_successful += 1
                                key_base = (
                                    f"s{split.split_id}_{sanitize(subset)}_{sanitize(target)}_"
                                    f"{sanitize(axis_variant)}_L{layer:02d}_{sanitize(representation)}"
                                )
                                for payload_key, arr in payload.items():
                                    axis_npz[f"{key_base}__{payload_key}"] = arr
                                axis_metadata.append(
                                    {
                                        "key_base": key_base,
                                        "split_id": split.split_id,
                                        "subset": subset,
                                        "target": target,
                                        "axis_variant": axis_variant,
                                        "layer": layer,
                                        "representation": representation,
                                        "projection_file": rel(out_path),
                                    }
                                )
                                np.savez_compressed(model_dir / f"{key_base}.npz", **payload)
                            else:
                                jobs_skipped += 1
                        except Exception as exc:  # keep going and report all failures
                            jobs_skipped += 1
                            warning = f"{type(exc).__name__}: {exc}"
                            print(f"WARNING: failed job: {warning}")
                            metrics_rows.append(
                                {
                                    "split_id": split.split_id,
                                    "subset": subset,
                                    "target": target,
                                    "axis_variant": axis_variant,
                                    "layer": layer,
                                    "representation": representation,
                                    "status": "failed",
                                    "warnings": warning,
                                    "projection_file": rel(out_path),
                                }
                            )

    metrics_df = pd.DataFrame(metrics_rows)
    counts_df = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    ensure_dir(output_dir)
    metrics_path = output_dir / "commitment_projection_metrics.csv"
    counts_path = output_dir / "axis_fit_counts.csv"
    metrics_df.to_csv(metrics_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    axis_npz["metadata_json"] = np.asarray(json.dumps(axis_metadata, indent=2), dtype=object)
    if axis_npz:
        np.savez_compressed(output_dir / "commitment_axes.npz", **axis_npz)
    manifest.update(
        {
            "jobs_attempted": int(jobs_attempted),
            "jobs_successful": int(jobs_successful),
            "jobs_skipped": int(jobs_skipped),
            "metrics": rel(metrics_path),
            "axis_fit_counts": rel(counts_path),
            "commitment_axes": rel(output_dir / "commitment_axes.npz"),
            "warnings": warnings,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    write_report(output_dir, manifest, metrics_df, warnings)
    print("Commitment contrast fitting complete.")
    print(json.dumps({k: manifest[k] for k in ["jobs_attempted", "jobs_successful", "jobs_skipped", "metrics", "axis_fit_counts", "commitment_axes"]}, indent=2))


if __name__ == "__main__":
    main()
