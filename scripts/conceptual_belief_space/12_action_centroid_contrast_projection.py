#!/usr/bin/env python3
"""Action-centroid contrast projection for activation states.

This projection is anchored by train-only action-extreme activation centroids.
It does not use PCA or UMAP. The only low-dimensional basis is obtained by SVD
of the four centered action-centroid contrast vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.plotting import action_color_map, savefig


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "action_centroid_contrast"
FIGURE_DIR = DEFAULT_FIGURE_DIR / "action_centroid_contrast"
MODEL_DIR = DEFAULT_OUTPUT_DIR / "models/action_centroid_contrast"
EPS = 1e-8


@dataclass
class SplitRows:
    split_id: int
    train_row_ids: np.ndarray
    test_row_ids: np.ndarray


@dataclass
class BlockResult:
    coords: pd.DataFrame
    centroids: pd.DataFrame
    pairwise: pd.DataFrame
    axes_metadata: pd.DataFrame
    metrics: dict[str, Any]
    model_payload: dict[str, Any]
    centroid_coords: pd.DataFrame


class ParquetBlockWriter:
    """Write parquet blocks incrementally without keeping all rows in RAM."""

    def __init__(self, path: Path):
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        self.path = path
        ensure_dir(path.parent)
        self.pa = pa
        self.pq = pq
        self.writer = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        table = self.pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--extremes", default=DEFAULT_OUTPUT_DIR / "extreme_examples/action_extreme_examples.parquet")
    parser.add_argument("--activation-metadata", default=Path("outputs/activation/activation_feature_metadata.parquet"))
    parser.add_argument("--feature-dir", default=Path("outputs/activation/features"))
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--subset", default="high_commitment")
    parser.add_argument("--split_ids", default="all")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--representations", default="raw,delta_pre,unit_delta_pre")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-split-id", type=int, default=0)
    parser.add_argument("--plot-max-points", type=int, default=12000)
    parser.add_argument("--only-plot-best", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-extreme-per-action", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
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


def parse_int_list(value: str, *, available: Sequence[int] | None = None) -> list[int]:
    if str(value).strip().lower() == "all":
        if available is None:
            raise ValueError("available values are required for `all`")
        return sorted(int(v) for v in available)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_str_list(value: str, *, available: Sequence[str] | None = None) -> list[str]:
    if str(value).strip().lower() == "all":
        if available is None:
            raise ValueError("available values are required for `all`")
        return sorted(str(v) for v in available)
    requested = [part.strip() for part in str(value).split(",") if part.strip()]
    if available is not None:
        missing = [item for item in requested if item not in set(available)]
        if missing:
            print(f"WARNING: requested representations are unavailable and will be skipped: {missing}")
            requested = [item for item in requested if item in set(available)]
    return requested


def discover_feature_files(feature_dir: Path) -> tuple[list[int], list[str], dict[tuple[int, str], Path]]:
    files: dict[tuple[int, str], Path] = {}
    layers: set[int] = set()
    representations: set[str] = set()
    for path in feature_dir.glob("layer_*_*.npy"):
        stem = path.stem
        if not stem.startswith("layer_"):
            continue
        rest = stem[len("layer_") :]
        layer_text, _, representation = rest.partition("_")
        if not layer_text.isdigit() or not representation:
            continue
        layer = int(layer_text)
        layers.add(layer)
        representations.add(representation)
        files[(layer, representation)] = path
    return sorted(layers), sorted(representations), files


def check_overwrite(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        text = "\n".join(f"- {rel(path)}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing action-centroid contrast outputs without --overwrite:\n"
            f"{text}\n"
            "Pass --overwrite to regenerate."
        )


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def integer_like(series: pd.Series) -> bool:
    try:
        values = series.dropna().to_numpy()
        as_int = values.astype(np.int64)
        return np.array_equal(values, as_int)
    except Exception:
        return False


def load_row_index_map(metadata_path: Path, belief_df: pd.DataFrame, feature_shape_n: int) -> dict[int, int]:
    """Return row_id -> npy row index with explicit compatibility checks."""
    if "row_id" not in belief_df.columns:
        raise ValueError("belief table must contain row_id")
    row_ids = belief_df["row_id"].astype(np.int64).to_numpy()

    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        if len(metadata) != feature_shape_n:
            raise ValueError(
                f"{metadata_path} row count {len(metadata)} does not match feature rows {feature_shape_n}"
            )
        if "row_id" in metadata.columns:
            meta_row_ids = metadata["row_id"].astype(np.int64).to_numpy()
            missing = set(row_ids.tolist()) - set(meta_row_ids.tolist())
            if missing:
                raise ValueError(f"Activation metadata missing {len(missing)} belief row_ids")
            return {int(row_id): int(i) for i, row_id in enumerate(meta_row_ids)}

        print(
            f"WARNING: {rel(metadata_path)} exists but has no row_id column. "
            "Falling back only after asserting belief row_id equals feature row index."
        )

    expected = np.arange(feature_shape_n, dtype=np.int64)
    if len(belief_df) == feature_shape_n and integer_like(belief_df["row_id"]):
        sorted_row_ids = np.sort(row_ids)
        if np.array_equal(sorted_row_ids, expected) and np.array_equal(row_ids, expected):
            print("WARNING: using belief row_id as direct npy row index.")
            return {int(row_id): int(row_id) for row_id in row_ids}

    raise ValueError(
        "Cannot infer activation feature row alignment. Need activation metadata with row_id, "
        "or belief row_id exactly equal to [0, n_rows)."
    )


def split_indices_from_row_ids(beliefs: pd.DataFrame, row_ids: np.ndarray) -> np.ndarray:
    row_id_to_pos = pd.Series(np.arange(len(beliefs), dtype=np.int64), index=beliefs["row_id"].astype(np.int64))
    out = row_id_to_pos.reindex(row_ids.astype(np.int64))
    if out.isna().any():
        raise ValueError("Some split row_ids are absent from beliefs")
    return out.to_numpy(dtype=np.int64)


def parse_split_rows(split: dict[str, Any], beliefs: pd.DataFrame) -> SplitRows:
    split_id = int(split.get("split_id", split.get("fold", -1)))
    if split_id < 0:
        raise ValueError(f"Split has no valid split_id: {split.keys()}")

    if "train_row_ids" in split and "test_row_ids" in split:
        return SplitRows(
            split_id=split_id,
            train_row_ids=np.asarray(split["train_row_ids"], dtype=np.int64),
            test_row_ids=np.asarray(split["test_row_ids"], dtype=np.int64),
        )

    if "train_indices" in split and "test_indices" in split:
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        return SplitRows(
            split_id=split_id,
            train_row_ids=beliefs.iloc[train_indices]["row_id"].to_numpy(dtype=np.int64),
            test_row_ids=beliefs.iloc[test_indices]["row_id"].to_numpy(dtype=np.int64),
        )

    if "train_groups" in split and "test_groups" in split and "trajectory_id" in beliefs.columns:
        train_groups = set(map(str, split["train_groups"]))
        test_groups = set(map(str, split["test_groups"]))
        train_mask = beliefs["trajectory_id"].astype(str).isin(train_groups)
        test_mask = beliefs["trajectory_id"].astype(str).isin(test_groups)
        return SplitRows(
            split_id=split_id,
            train_row_ids=beliefs.loc[train_mask, "row_id"].to_numpy(dtype=np.int64),
            test_row_ids=beliefs.loc[test_mask, "row_id"].to_numpy(dtype=np.int64),
        )

    if "train_decision_ids" in split and "test_decision_ids" in split and "decision_id" in beliefs.columns:
        train_ids = set(map(str, split["train_decision_ids"]))
        test_ids = set(map(str, split["test_decision_ids"]))
        train_mask = beliefs["decision_id"].astype(str).isin(train_ids)
        test_mask = beliefs["decision_id"].astype(str).isin(test_ids)
        return SplitRows(
            split_id=split_id,
            train_row_ids=beliefs.loc[train_mask, "row_id"].to_numpy(dtype=np.int64),
            test_row_ids=beliefs.loc[test_mask, "row_id"].to_numpy(dtype=np.int64),
        )

    raise ValueError(
        f"Cannot parse split {split_id}. Expected train_row_ids/test_row_ids, train_indices/test_indices, "
        "train_groups/test_groups with trajectory_id, or train_decision_ids/test_decision_ids with decision_id."
    )


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return np.asarray(x / norms, dtype=np.float32)


def fit_preprocess_stats(
    features: np.ndarray,
    train_indices: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not standardize:
        hidden_dim = int(features.shape[1])
        return np.zeros(hidden_dim, dtype=np.float32), np.ones(hidden_dim, dtype=np.float32)
    train_x = np.asarray(features[train_indices], dtype=np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def transform_rows(
    features: np.ndarray,
    indices: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool,
    l2_normalize: bool,
) -> np.ndarray:
    x = np.asarray(features[indices], dtype=np.float32)
    if standardize:
        x = (x - mean) / std
    x = np.asarray(x, dtype=np.float32)
    if l2_normalize:
        x = l2_normalize_rows(x)
    return x


def compute_axes(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grand_mean = centroids.mean(axis=0, keepdims=True).astype(np.float32)
    contrasts = (centroids - grand_mean).astype(np.float32)
    _, singular_values_full, vt = np.linalg.svd(contrasts, full_matrices=False)
    total_energy = float(np.sum(singular_values_full**2))
    if total_energy > 0:
        energy_full = (singular_values_full**2) / total_energy
    else:
        energy_full = np.zeros_like(singular_values_full)
    axes = np.zeros((3, centroids.shape[1]), dtype=np.float32)
    singular_values = np.zeros(3, dtype=np.float32)
    energy_ratio = np.zeros(3, dtype=np.float32)
    axis_rows: list[dict[str, Any]] = []
    keep = min(3, vt.shape[0])
    axes[:keep] = vt[:keep].astype(np.float32)
    singular_values[:keep] = singular_values_full[:keep].astype(np.float32)
    energy_ratio[:keep] = energy_full[:keep].astype(np.float32)

    for k in range(3):
        if k < keep:
            scores = contrasts @ axes[k]
            j = int(np.argmax(np.abs(scores)))
            if float(scores[j]) < 0:
                axes[k] *= -1.0
                scores = contrasts @ axes[k]
            pos_idx = int(np.argmax(scores))
            neg_idx = int(np.argmin(scores))
            axis_rows.append(
                {
                    "axis": int(k + 1),
                    "singular_value": float(singular_values[k]),
                    "energy_ratio": float(energy_ratio[k]),
                    "dominant_positive_action": ACTIONS[pos_idx],
                    "dominant_negative_action": ACTIONS[neg_idx],
                    "positive_centroid_score": float(scores[pos_idx]),
                    "negative_centroid_score": float(scores[neg_idx]),
                }
            )
        else:
            axis_rows.append(
                {
                    "axis": int(k + 1),
                    "singular_value": 0.0,
                    "energy_ratio": 0.0,
                    "dominant_positive_action": None,
                    "dominant_negative_action": None,
                    "positive_centroid_score": 0.0,
                    "negative_centroid_score": 0.0,
                }
            )
    return axes, singular_values, energy_ratio, contrasts, axis_rows


def nearest_highdim(
    x: np.ndarray,
    centroids: np.ndarray,
    *,
    use_cosine: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_cosine:
        x_norm = l2_normalize_rows(x)
        c_norm = l2_normalize_rows(centroids.astype(np.float32))
        scores = x_norm @ c_norm.T
        order = np.argsort(scores, axis=1)[:, ::-1]
        pred = np.asarray(ACTIONS, dtype=object)[order[:, 0]]
        margins = scores[np.arange(len(x)), order[:, 0]] - scores[np.arange(len(x)), order[:, 1]]
        return pred, margins.astype(np.float32)

    distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    pred = np.asarray(ACTIONS, dtype=object)[order[:, 0]]
    margins = distances[np.arange(len(x)), order[:, 1]] - distances[np.arange(len(x)), order[:, 0]]
    return pred, margins.astype(np.float32)


def nearest_contrast(z: np.ndarray, centroid_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(z[:, None, :] - centroid_coords[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    pred = np.asarray(ACTIONS, dtype=object)[order[:, 0]]
    margins = distances[np.arange(len(z)), order[:, 1]] - distances[np.arange(len(z)), order[:, 0]]
    return pred, margins.astype(np.float32)


def valid_label_mask(labels: pd.Series) -> np.ndarray:
    return labels.notna().to_numpy() & labels.astype(str).isin(ACTIONS).to_numpy()


def accuracy(labels: Sequence[Any], preds: Sequence[Any]) -> float:
    labels_arr = np.asarray(labels, dtype=object)
    preds_arr = np.asarray(preds, dtype=object)
    if len(labels_arr) == 0:
        return float("nan")
    return float(np.mean(labels_arr == preds_arr))


def balanced_accuracy(labels: Sequence[Any], preds: Sequence[Any]) -> float:
    from sklearn.metrics import balanced_accuracy_score

    labels_arr = np.asarray(labels, dtype=object)
    preds_arr = np.asarray(preds, dtype=object)
    if len(labels_arr) == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    try:
        return float(balanced_accuracy_score(labels_arr, preds_arr))
    except Exception:
        return float("nan")


def add_classification_metrics(
    metrics: dict[str, Any],
    *,
    prefix: str,
    labels: pd.Series,
    highdim_pred: np.ndarray,
    contrast_pred: np.ndarray,
    mask: np.ndarray,
) -> None:
    label_mask = mask & valid_label_mask(labels)
    y = labels.loc[label_mask].astype(str).to_numpy()
    metrics[f"{prefix}_accuracy_highdim"] = accuracy(y, highdim_pred[label_mask])
    metrics[f"{prefix}_accuracy_contrast"] = accuracy(y, contrast_pred[label_mask])
    metrics[f"{prefix}_balanced_accuracy_highdim"] = balanced_accuracy(y, highdim_pred[label_mask])
    metrics[f"{prefix}_balanced_accuracy_contrast"] = balanced_accuracy(y, contrast_pred[label_mask])


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("WARNING: scipy unavailable; Spearman correlations will be NaN")
        return float("nan")
    value = spearmanr(x, y, nan_policy="omit").correlation
    return float(value) if value is not None else float("nan")


def pairwise_centroid_distances(
    *,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    centroids: np.ndarray,
    centroid_coords: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    c_norm = l2_normalize_rows(centroids.astype(np.float32))
    for i in range(len(ACTIONS)):
        for j in range(i + 1, len(ACTIONS)):
            contrast_diff = centroid_coords[i] - centroid_coords[j]
            rows.append(
                {
                    "split_id": int(split_id),
                    "subset": subset,
                    "layer": int(layer),
                    "representation": representation,
                    "action_i": ACTIONS[i],
                    "action_j": ACTIONS[j],
                    "highdim_cosine_distance": float(1.0 - np.dot(c_norm[i], c_norm[j])),
                    "highdim_euclidean_distance": float(np.linalg.norm(centroids[i] - centroids[j])),
                    "contrast_euclidean_distance": float(np.linalg.norm(contrast_diff)),
                    "contrast_axis1_absdiff": float(abs(contrast_diff[0])),
                    "contrast_axis2_absdiff": float(abs(contrast_diff[1])),
                    "contrast_axis3_absdiff": float(abs(contrast_diff[2])),
                }
            )
    return pd.DataFrame(rows)


def subset_extremes(extremes: pd.DataFrame, *, split_id: int, subset: str, train_row_ids: set[int]) -> pd.DataFrame:
    required = {"split_id", "subset", "action", "row_id"}
    missing = sorted(required - set(extremes.columns))
    if missing:
        raise ValueError(f"Extreme examples missing required columns: {missing}")
    part = extremes[
        extremes["split_id"].astype(int).eq(int(split_id))
        & extremes["subset"].astype(str).eq(str(subset))
        & extremes["action"].astype(str).isin(ACTIONS)
    ].copy()
    if "split_role" in part.columns:
        part = part[part["split_role"].astype(str).str.lower().eq("train")].copy()
    part["row_id"] = part["row_id"].astype(np.int64)
    if not set(part["row_id"].tolist()) <= train_row_ids:
        leaked = sorted(set(part["row_id"].tolist()) - train_row_ids)
        raise ValueError(f"Leakage error: {len(leaked)} extreme row_ids are not split train rows")
    if "q_action" in part.columns:
        part["q_action"] = pd.to_numeric(part["q_action"], errors="coerce")
        part = part.sort_values(["row_id", "q_action"], ascending=[True, False]).drop_duplicates("row_id")
    else:
        part = part.drop_duplicates("row_id")
    return part


def compute_centroids(
    x_all: np.ndarray,
    all_row_ids: np.ndarray,
    extreme_part: pd.DataFrame,
    *,
    min_extreme_per_action: int,
) -> tuple[np.ndarray, dict[str, int]]:
    row_id_to_pos = {int(row_id): int(i) for i, row_id in enumerate(all_row_ids)}
    centroids: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for action in ACTIONS:
        action_rows = extreme_part[extreme_part["action"].astype(str).eq(action)]["row_id"].astype(int).tolist()
        positions = [row_id_to_pos[row_id] for row_id in action_rows if row_id in row_id_to_pos]
        counts[action] = len(positions)
        if len(positions) < min_extreme_per_action:
            raise ValueError(f"too_few_extremes:{action}:{len(positions)}")
        centroids.append(x_all[np.asarray(positions, dtype=np.int64)].mean(axis=0))
    return np.vstack(centroids).astype(np.float32), counts


def build_coord_frame(
    beliefs: pd.DataFrame,
    *,
    split_rows: SplitRows,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    z: np.ndarray,
    highdim_pred: np.ndarray,
    highdim_margin: np.ndarray,
    contrast_pred: np.ndarray,
    contrast_margin: np.ndarray,
    extreme_part: pd.DataFrame,
) -> pd.DataFrame:
    meta_cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
        "entropy_commitment",
        "q_UP",
        "q_DOWN",
        "q_LEFT",
        "q_RIGHT",
        "strict_before_any_action_mention",
        "strict_before_final_action_mention",
    ]
    out = beliefs[[col for col in meta_cols if col in beliefs.columns]].copy()
    out.insert(0, "representation", representation)
    out.insert(0, "layer", int(layer))
    out.insert(0, "subset", subset)
    out.insert(0, "split_id", int(split_id))
    train_ids = set(split_rows.train_row_ids.astype(int).tolist())
    test_ids = set(split_rows.test_row_ids.astype(int).tolist())
    extreme_map = extreme_part.set_index("row_id")["action"].astype(str).to_dict()
    row_ids = out["row_id"].astype(int)
    out["is_train"] = row_ids.isin(train_ids)
    out["is_test"] = row_ids.isin(test_ids)
    out["extreme_action_for_subset"] = row_ids.map(extreme_map)
    out["is_train_extreme_for_subset"] = out["extreme_action_for_subset"].notna()
    if "strict_before_any_action_mention" in out.columns:
        out["is_strict_before_any_action_mention"] = out["strict_before_any_action_mention"].fillna(False).astype(bool)
    if "strict_before_final_action_mention" in out.columns:
        out["is_strict_before_final_action_mention"] = out["strict_before_final_action_mention"].fillna(False).astype(bool)
    out["action_cc1"] = z[:, 0].astype(np.float32)
    out["action_cc2"] = z[:, 1].astype(np.float32)
    out["action_cc3"] = z[:, 2].astype(np.float32)
    out["nearest_highdim_action"] = highdim_pred
    out["nearest_highdim_margin"] = highdim_margin.astype(np.float32)
    out["nearest_contrast_action"] = contrast_pred
    out["nearest_contrast_margin"] = contrast_margin.astype(np.float32)
    return out


def block_metrics(
    coord_df: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    counts: dict[str, int],
    singular_values: np.ndarray,
    energy_ratio: np.ndarray,
    total_energy: float,
    rank_estimate: int,
    pairwise: pd.DataFrame,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "split_id": int(split_id),
        "subset": subset,
        "layer": int(layer),
        "representation": representation,
        "n_rows_all": int(len(coord_df)),
        "n_train_rows": int(coord_df["is_train"].sum()),
        "n_test_rows": int(coord_df["is_test"].sum()),
        "n_train_extreme_rows": int(coord_df["is_train_extreme_for_subset"].sum()),
        "mean_highdim_margin": float(coord_df["nearest_highdim_margin"].mean()),
        "mean_contrast_margin": float(coord_df["nearest_contrast_margin"].mean()),
        "singular_value_1": float(singular_values[0]),
        "singular_value_2": float(singular_values[1]),
        "singular_value_3": float(singular_values[2]),
        "energy_ratio_1": float(energy_ratio[0]),
        "energy_ratio_2": float(energy_ratio[1]),
        "energy_ratio_3": float(energy_ratio[2]),
        "total_contrast_energy": float(total_energy),
        "rank_estimate": int(rank_estimate),
    }
    for action in ACTIONS:
        metrics[f"n_extreme_{action}"] = int(counts.get(action, 0))

    extreme_mask = coord_df["is_train_extreme_for_subset"].to_numpy()
    add_classification_metrics(
        metrics,
        prefix="train_extreme_accuracy_vs_extreme_action",
        labels=coord_df["extreme_action_for_subset"],
        highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
        contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
        mask=extreme_mask,
    )
    add_classification_metrics(
        metrics,
        prefix="all_rows_accuracy_vs_agent_action",
        labels=coord_df["agent_action"],
        highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
        contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
        mask=np.ones(len(coord_df), dtype=bool),
    )
    if "top_action" in coord_df.columns:
        add_classification_metrics(
            metrics,
            prefix="all_rows_accuracy_vs_top_action",
            labels=coord_df["top_action"],
            highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
            contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
            mask=np.ones(len(coord_df), dtype=bool),
        )
    test_mask = coord_df["is_test"].to_numpy()
    add_classification_metrics(
        metrics,
        prefix="test_rows_accuracy_vs_agent_action",
        labels=coord_df["agent_action"],
        highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
        contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
        mask=test_mask,
    )
    if "is_strict_before_any_action_mention" in coord_df.columns:
        add_classification_metrics(
            metrics,
            prefix="strict_before_any_action_mention_vs_agent_action",
            labels=coord_df["agent_action"],
            highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
            contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
            mask=coord_df["is_strict_before_any_action_mention"].to_numpy(),
        )
    if "is_strict_before_final_action_mention" in coord_df.columns:
        add_classification_metrics(
            metrics,
            prefix="strict_before_final_action_mention_vs_agent_action",
            labels=coord_df["agent_action"],
            highdim_pred=coord_df["nearest_highdim_action"].to_numpy(),
            contrast_pred=coord_df["nearest_contrast_action"].to_numpy(),
            mask=coord_df["is_strict_before_final_action_mention"].to_numpy(),
        )

    high_cos = pairwise["highdim_cosine_distance"].to_numpy(dtype=float)
    high_euc = pairwise["highdim_euclidean_distance"].to_numpy(dtype=float)
    con_euc = pairwise["contrast_euclidean_distance"].to_numpy(dtype=float)
    metrics["centroid_pair_spearman_highdim_cosine_vs_contrast_euclidean"] = spearman(high_cos, con_euc)
    metrics["centroid_pair_pearson_highdim_cosine_vs_contrast_euclidean"] = pearson(high_cos, con_euc)
    metrics["centroid_pair_spearman_highdim_euclidean_vs_contrast_euclidean"] = spearman(high_euc, con_euc)
    metrics["centroid_pair_pearson_highdim_euclidean_vs_contrast_euclidean"] = pearson(high_euc, con_euc)

    # Backwards-compatible aliases matching the requested names exactly.
    metrics["train_extreme_accuracy_vs_extreme_action_highdim"] = metrics[
        "train_extreme_accuracy_vs_extreme_action_accuracy_highdim"
    ]
    metrics["train_extreme_accuracy_vs_extreme_action_contrast"] = metrics[
        "train_extreme_accuracy_vs_extreme_action_accuracy_contrast"
    ]
    metrics["train_extreme_balanced_accuracy_vs_extreme_action_highdim"] = metrics[
        "train_extreme_accuracy_vs_extreme_action_balanced_accuracy_highdim"
    ]
    metrics["train_extreme_balanced_accuracy_vs_extreme_action_contrast"] = metrics[
        "train_extreme_accuracy_vs_extreme_action_balanced_accuracy_contrast"
    ]
    metrics["all_rows_accuracy_vs_agent_action_highdim"] = metrics[
        "all_rows_accuracy_vs_agent_action_accuracy_highdim"
    ]
    metrics["all_rows_accuracy_vs_agent_action_contrast"] = metrics[
        "all_rows_accuracy_vs_agent_action_accuracy_contrast"
    ]
    metrics["all_rows_balanced_accuracy_vs_agent_action_highdim"] = metrics[
        "all_rows_accuracy_vs_agent_action_balanced_accuracy_highdim"
    ]
    metrics["all_rows_balanced_accuracy_vs_agent_action_contrast"] = metrics[
        "all_rows_accuracy_vs_agent_action_balanced_accuracy_contrast"
    ]
    metrics["test_rows_balanced_accuracy_vs_agent_action_highdim"] = metrics[
        "test_rows_accuracy_vs_agent_action_balanced_accuracy_highdim"
    ]
    metrics["test_rows_balanced_accuracy_vs_agent_action_contrast"] = metrics[
        "test_rows_accuracy_vs_agent_action_balanced_accuracy_contrast"
    ]
    if "all_rows_accuracy_vs_top_action_accuracy_highdim" in metrics:
        metrics["all_rows_accuracy_vs_top_action_highdim"] = metrics["all_rows_accuracy_vs_top_action_accuracy_highdim"]
        metrics["all_rows_accuracy_vs_top_action_contrast"] = metrics["all_rows_accuracy_vs_top_action_accuracy_contrast"]
    return metrics


def make_centroid_frame(
    *,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    counts: dict[str, int],
    centroids: np.ndarray,
    contrasts: np.ndarray,
    centroid_coords: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, action in enumerate(ACTIONS):
        rows.append(
            {
                "split_id": int(split_id),
                "subset": subset,
                "layer": int(layer),
                "representation": representation,
                "action": action,
                "n_extreme_rows": int(counts.get(action, 0)),
                "action_cc1": float(centroid_coords[i, 0]),
                "action_cc2": float(centroid_coords[i, 1]),
                "action_cc3": float(centroid_coords[i, 2]),
                "norm_highdim_centroid": float(np.linalg.norm(centroids[i])),
                "norm_highdim_contrast_vector": float(np.linalg.norm(contrasts[i])),
            }
        )
    return pd.DataFrame(rows)


def save_model_npz(
    path: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    axes: np.ndarray,
    singular_values: np.ndarray,
    energy_ratio: np.ndarray,
    centroids: np.ndarray,
    contrasts: np.ndarray,
    centroid_coords: np.ndarray,
    standardize: bool,
    l2_normalize: bool,
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        actions=np.asarray(ACTIONS, dtype=object),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        axes=axes.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        energy_ratio=energy_ratio.astype(np.float32),
        centroids_highdim=centroids.astype(np.float32),
        centroid_contrasts_highdim=contrasts.astype(np.float32),
        centroid_coords=centroid_coords.astype(np.float32),
        standardize=np.asarray([bool(standardize)]),
        l2_normalize=np.asarray([bool(l2_normalize)]),
    )


def run_block(
    *,
    beliefs: pd.DataFrame,
    extremes: pd.DataFrame,
    features: np.ndarray,
    row_index_map: dict[int, int],
    split_rows: SplitRows,
    split_id: int,
    subset: str,
    layer: int,
    representation: str,
    standardize: bool,
    l2_normalize: bool,
    min_extreme_per_action: int,
    model_dir: Path,
) -> BlockResult:
    all_row_ids = beliefs["row_id"].to_numpy(dtype=np.int64)
    all_indices = np.asarray([row_index_map[int(row_id)] for row_id in all_row_ids], dtype=np.int64)
    train_indices = np.asarray([row_index_map[int(row_id)] for row_id in split_rows.train_row_ids], dtype=np.int64)
    train_row_id_set = set(split_rows.train_row_ids.astype(int).tolist())
    test_row_id_set = set(split_rows.test_row_ids.astype(int).tolist())
    if train_row_id_set & test_row_id_set:
        raise ValueError(f"Split {split_id} has overlapping train/test row_ids")

    extreme_part = subset_extremes(extremes, split_id=split_id, subset=subset, train_row_ids=train_row_id_set)
    mean, std = fit_preprocess_stats(features, train_indices, standardize=standardize)
    x_all = transform_rows(
        features,
        all_indices,
        mean=mean,
        std=std,
        standardize=standardize,
        l2_normalize=l2_normalize,
    )
    centroids, counts = compute_centroids(
        x_all,
        all_row_ids,
        extreme_part,
        min_extreme_per_action=min_extreme_per_action,
    )
    axes, singular_values, energy_ratio, contrasts, axis_rows = compute_axes(centroids)
    if axes.shape != (3, features.shape[1]):
        raise ValueError(f"Projection matrix shape should be 3 x D, got {axes.shape}")
    z = x_all @ axes.T
    centroid_coords = contrasts @ axes.T
    if not np.isfinite(z).all():
        raise ValueError("nonfinite_projection")

    highdim_pred, highdim_margin = nearest_highdim(x_all, centroids, use_cosine=l2_normalize)
    contrast_pred, contrast_margin = nearest_contrast(z, centroid_coords)
    coord_df = build_coord_frame(
        beliefs,
        split_rows=split_rows,
        split_id=split_id,
        subset=subset,
        layer=layer,
        representation=representation,
        z=z,
        highdim_pred=highdim_pred,
        highdim_margin=highdim_margin,
        contrast_pred=contrast_pred,
        contrast_margin=contrast_margin,
        extreme_part=extreme_part,
    )
    pairwise = pairwise_centroid_distances(
        split_id=split_id,
        subset=subset,
        layer=layer,
        representation=representation,
        centroids=centroids,
        centroid_coords=centroid_coords,
    )
    rank_estimate = int(np.sum(singular_values > 1e-8))
    total_energy = float(np.sum(singular_values**2))
    metrics = block_metrics(
        coord_df,
        split_id=split_id,
        subset=subset,
        layer=layer,
        representation=representation,
        counts=counts,
        singular_values=singular_values,
        energy_ratio=energy_ratio,
        total_energy=total_energy,
        rank_estimate=rank_estimate,
        pairwise=pairwise,
    )
    axes_metadata = pd.DataFrame(axis_rows)
    axes_metadata.insert(0, "representation", representation)
    axes_metadata.insert(0, "layer", int(layer))
    axes_metadata.insert(0, "subset", subset)
    axes_metadata.insert(0, "split_id", int(split_id))
    axes_metadata["standardize"] = bool(standardize)
    axes_metadata["l2_normalize"] = bool(l2_normalize)
    axes_metadata["n_train_rows"] = int(len(split_rows.train_row_ids))
    axes_metadata["n_train_extreme_rows"] = int(len(extreme_part))
    centroids_df = make_centroid_frame(
        split_id=split_id,
        subset=subset,
        layer=layer,
        representation=representation,
        counts=counts,
        centroids=centroids,
        contrasts=contrasts,
        centroid_coords=centroid_coords,
    )
    model_path = (
        model_dir
        / f"contrast_model_split{int(split_id)}_{subset}_layer{int(layer):02d}_{representation}.npz"
    )
    save_model_npz(
        model_path,
        mean=mean,
        std=std,
        axes=axes,
        singular_values=singular_values,
        energy_ratio=energy_ratio,
        centroids=centroids,
        contrasts=contrasts,
        centroid_coords=centroid_coords,
        standardize=standardize,
        l2_normalize=l2_normalize,
    )
    return BlockResult(
        coords=coord_df,
        centroids=centroids_df,
        pairwise=pairwise,
        axes_metadata=axes_metadata,
        metrics=metrics,
        model_payload={"path": str(model_path)},
        centroid_coords=centroids_df,
    )


def sample_for_plot(df: pd.DataFrame, *, max_points: int, label_col: str, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    if label_col in df.columns:
        parts = []
        per_label = max(1, max_points // max(1, df[label_col].nunique(dropna=True)))
        for _, part in df.groupby(label_col, dropna=False):
            n = min(len(part), per_label)
            parts.append(part.sample(n=n, random_state=seed) if len(part) > n else part)
        out = pd.concat(parts, ignore_index=True)
        if len(out) > max_points:
            out = out.sample(n=max_points, random_state=seed)
        return out
    return df.sample(n=max_points, random_state=seed)


def plot_block_scatter(
    ax,
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    color_col: str,
    continuous: bool,
    title: str,
    alpha: float,
    size: float,
    cmap: str = "viridis",
):
    colors = action_color_map({"UP": "C0", "DOWN": "C1", "LEFT": "C2", "RIGHT": "C3"})
    mappable = None
    if continuous:
        vals = pd.to_numeric(df[color_col], errors="coerce")
        mappable = ax.scatter(
            df["action_cc1"],
            df["action_cc2"],
            c=vals,
            cmap=cmap,
            s=size,
            alpha=alpha,
            edgecolors="none",
        )
    else:
        for action in ACTIONS:
            part = df[df[color_col].astype(str).eq(action)]
            if not part.empty:
                ax.scatter(
                    part["action_cc1"],
                    part["action_cc2"],
                    s=size,
                    alpha=alpha,
                    color=colors[action],
                    label=action,
                    edgecolors="none",
                )
    for _, row in centroids.iterrows():
        action = str(row["action"])
        ax.scatter(
            [row["action_cc1"]],
            [row["action_cc2"]],
            marker="X",
            s=95,
            color=colors.get(action, "black"),
            edgecolors="black",
            linewidths=0.7,
        )
        ax.text(float(row["action_cc1"]), float(row["action_cc2"]), f" {action}", fontsize=8, weight="bold")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return mappable


def plot_grid(
    plot_coords: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    split_id: int,
    subset: str,
    representation: str,
    layers: Sequence[int],
    mode: str,
    output_path: Path,
    plot_max_points: int,
    seed: int,
) -> Path:
    import matplotlib.pyplot as plt

    continuous = mode in {"commitment", "progress"}
    if mode == "extreme_action":
        title_suffix = "by extreme action, train extremes"
        color_col = "extreme_action_for_subset"
        base = plot_coords[plot_coords["is_train_extreme_for_subset"]].copy()
        alpha = 0.8
        size = 8.0
    elif mode == "agent_action":
        title_suffix = "by agent action, all rows"
        color_col = "agent_action"
        base = plot_coords.copy()
        alpha = 0.25
        size = 4.0
    elif mode == "commitment":
        title_suffix = "by commitment_to_final, all rows"
        color_col = "commitment_to_final"
        base = plot_coords.copy()
        alpha = 0.35
        size = 4.0
    elif mode == "progress":
        title_suffix = "by progress, all rows"
        color_col = "progress"
        base = plot_coords.copy()
        alpha = 0.35
        size = 4.0
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

    layers = list(layers)
    ncols = min(4, max(1, len(layers)))
    nrows = int(math.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    mappable = None
    for ax in axes.ravel():
        ax.axis("off")
    for i, layer in enumerate(layers):
        ax = axes.ravel()[i]
        ax.axis("on")
        part = base[
            base["layer"].astype(int).eq(int(layer))
            & base["representation"].astype(str).eq(str(representation))
            & base["split_id"].astype(int).eq(int(split_id))
            & base["subset"].astype(str).eq(str(subset))
        ].copy()
        label_col = "agent_action" if mode != "extreme_action" else "extreme_action_for_subset"
        part = sample_for_plot(part, max_points=plot_max_points, label_col=label_col, seed=seed + int(layer))
        cents = centroids[
            centroids["layer"].astype(int).eq(int(layer))
            & centroids["representation"].astype(str).eq(str(representation))
            & centroids["split_id"].astype(int).eq(int(split_id))
            & centroids["subset"].astype(str).eq(str(subset))
        ]
        if not part.empty:
            mappable = plot_block_scatter(
                ax,
                part,
                cents,
                color_col=color_col,
                continuous=continuous,
                title=f"L{int(layer):02d}",
                alpha=alpha,
                size=size,
            )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles and not continuous:
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=len(ACTIONS),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )
    if continuous and mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.7, pad=0.01)
    fig.suptitle(
        f"Action-centroid contrast projection\n"
        f"split={split_id}, subset={subset}, repr={representation}, {title_suffix}",
        y=0.99,
        fontsize=10,
    )
    bottom = 0.055 if handles and not continuous else 0.035
    right = 0.90 if continuous else 0.985
    fig.subplots_adjust(top=0.89, bottom=bottom, left=0.025, right=right, wspace=0.05, hspace=0.24)
    return savefig(output_path, dpi=200, fig=fig)


def plot_best_single(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    color_col: str,
    continuous: bool,
    title: str,
    output_path: Path,
    plot_max_points: int,
    seed: int,
    train_extremes_only: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    part = df[df["is_train_extreme_for_subset"]].copy() if train_extremes_only else df.copy()
    label_col = "extreme_action_for_subset" if train_extremes_only else color_col
    part = sample_for_plot(part, max_points=plot_max_points, label_col=label_col, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    mappable = plot_block_scatter(
        ax,
        part,
        centroids,
        color_col=label_col if train_extremes_only else color_col,
        continuous=continuous,
        title=title,
        alpha=0.8 if train_extremes_only else 0.35,
        size=10.0 if train_extremes_only else 5.0,
    )
    ax.set_xlabel("action contrast component 1")
    ax.set_ylabel("action contrast component 2")
    handles, labels = ax.get_legend_handles_labels()
    if handles and not continuous:
        ax.legend(handles, labels, frameon=False, fontsize=8)
    if continuous and mappable is not None:
        fig.colorbar(mappable, ax=ax)
    return savefig(output_path, dpi=200, fig=fig)


def line_plot(metrics: pd.DataFrame, *, metric_col: str, title: str, ylabel: str, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for representation, part in metrics.groupby("representation"):
        grouped = part.groupby("layer", as_index=False)[metric_col].mean().sort_values("layer")
        ax.plot(grouped["layer"], grouped[metric_col], marker="o", label=str(representation))
    if "accuracy" in metric_col:
        ax.axhline(0.25, color="#888888", lw=0.8, ls="--", label="chance")
    elif "correlation" in title.lower() or "spearman" in metric_col:
        ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(output_path, dpi=200, fig=fig)


def write_report(
    report_path: Path,
    *,
    output_paths: dict[str, Path],
    metrics: pd.DataFrame,
    skipped: list[dict[str, Any]],
    figures: list[Path],
    layers: list[int],
    representations: list[str],
    split_ids: list[int],
    subset: str,
) -> None:
    def top_table(metric_col: str, n: int = 10) -> str:
        cols = [
            "split_id",
            "layer",
            "representation",
            metric_col,
            "train_extreme_balanced_accuracy_vs_extreme_action_contrast",
            "test_rows_balanced_accuracy_vs_agent_action_contrast",
            "all_rows_balanced_accuracy_vs_agent_action_contrast",
        ]
        available = [col for col in cols if col in metrics.columns]
        part = metrics.sort_values(metric_col, ascending=False, na_position="last")[available].head(n).copy()
        for col in part.columns:
            if pd.api.types.is_float_dtype(part[col]):
                part[col] = part[col].map(lambda value: "NA" if pd.isna(value) else f"{float(value):.4f}")
        if part.empty:
            return "_No rows._"
        lines = [
            "| " + " | ".join(part.columns) + " |",
            "| " + " | ".join(["---"] * len(part.columns)) + " |",
        ]
        for _, row in part.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in part.columns) + " |")
        return "\n".join(lines)

    figure_lines = [f"- `{rel(path)}`" for path in figures[:120]]
    if len(figures) > 120:
        figure_lines.append(f"- ... {len(figures) - 120} more figures")

    lines = [
        "# Action-Centroid Contrast Projection",
        "",
        "## Purpose And Method",
        "",
        "This step asks whether activation states separate along action-concept axes defined by train-only "
        "UP/DOWN/LEFT/RIGHT action-extreme examples.",
        "",
        "**No PCA or UMAP was used.** The only dimensionality reduction is SVD on the four centered train "
        "action-centroid contrast vectors. Because these are four centered action contrasts, the rank is at most 3.",
        "",
        "## Inputs And Outputs",
        "",
        *[f"- {name}: `{rel(path)}`" for name, path in output_paths.items()],
        "",
        "## Run Scope",
        "",
        f"- subset: `{subset}`",
        f"- split_ids: `{split_ids}`",
        f"- layers: `{layers}`",
        f"- representations: `{representations}`",
        "",
        "## Best Blocks",
        "",
        "### Train Extreme Balanced Accuracy vs Extreme Action, Contrast Space",
        "",
        top_table("train_extreme_balanced_accuracy_vs_extreme_action_contrast"),
        "",
        "### Test Rows Balanced Accuracy vs Agent Action, Contrast Space",
        "",
        top_table("test_rows_balanced_accuracy_vs_agent_action_contrast"),
        "",
        "### All Rows Balanced Accuracy vs Agent Action, Contrast Space",
        "",
        top_table("all_rows_balanced_accuracy_vs_agent_action_contrast"),
        "",
        "### Centroid-Pair Spearman: High-D Cosine vs Contrast Euclidean",
        "",
        top_table("centroid_pair_spearman_highdim_cosine_vs_contrast_euclidean"),
        "",
        "## Caveats",
        "",
        "- Four actions imply only six centroid-pair distances, so distance-correlation diagnostics are low-powered.",
        "- Train-extreme separation can be inflated because the axes are built from those extremes; test/all-row metrics are more important.",
        "- This is descriptive geometry, not causal evidence.",
        "- The sign of each SVD axis is a deterministic convention, not an intrinsic semantic direction.",
        "",
        "## Interpretation",
        "",
        "- If train extremes separate but all/test rows do not, action information is present only in selected anchors or is not globally organized as action branches.",
        "- If both train extremes and held-out/test rows separate, that is stronger evidence for an activation action concept subspace.",
        "- If contrast projection does not separate while high-dimensional nearest-centroid does, action information exists in high-D but the first two action contrast axes do not preserve it well.",
        "- If neither high-dimensional nearest-centroid nor contrast projection separates, current CoT-prefix activations do not show clear action geometry under this anchor definition.",
        "",
        "## Skipped Blocks",
        "",
        f"- skipped blocks: `{len(skipped)}`",
    ]
    if skipped:
        for item in skipped[:50]:
            lines.append(
                f"- split={item.get('split_id')} layer={item.get('layer')} representation={item.get('representation')}: {item.get('reason')}"
            )
    lines.extend(["", "## Figures", "", *figure_lines])
    save_markdown(report_path, "\n".join(lines))


def main() -> None:
    if ACTIONS != ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise SystemExit(f"Unexpected ACTIONS order: {ACTIONS}")
    args = parse_args()
    beliefs_path = resolve_repo_path(args.beliefs)
    splits_path = resolve_repo_path(args.splits)
    extremes_path = resolve_repo_path(args.extremes)
    metadata_path = resolve_repo_path(args.activation_metadata)
    feature_dir = resolve_repo_path(args.feature_dir)
    output_dir = resolve_repo_path(args.output_dir)
    figure_dir = resolve_repo_path(args.figure_dir)
    model_dir = resolve_repo_path(args.model_dir)
    ensure_dir(output_dir)
    ensure_dir(figure_dir)
    ensure_dir(model_dir)

    coords_path = output_dir / "action_contrast_coords.parquet"
    centroids_path = output_dir / "action_contrast_centroids.parquet"
    metrics_path = output_dir / "action_contrast_metrics.csv"
    pairwise_path = output_dir / "action_contrast_pairwise_distances.csv"
    axes_path = output_dir / "action_contrast_axes_metadata.csv"
    report_path = output_dir / "12_action_centroid_contrast_report.md"
    check_overwrite(
        [coords_path, centroids_path, metrics_path, pairwise_path, axes_path, report_path],
        overwrite=bool(args.overwrite),
    )

    beliefs = pd.read_parquet(beliefs_path)
    extremes = pd.read_parquet(extremes_path)
    split_records = read_json_list(splits_path)
    detected_layers, detected_reps, feature_files = discover_feature_files(feature_dir)
    layers = parse_int_list(args.layers, available=detected_layers)
    default_reps = [rep for rep in ["raw", "delta_pre", "unit_delta_pre"] if rep in detected_reps]
    representations = parse_str_list(args.representations, available=detected_reps) if args.representations != "all" else default_reps
    split_rows_all = [parse_split_rows(split, beliefs) for split in split_records]
    available_split_ids = [split.split_id for split in split_rows_all]
    split_ids = parse_int_list(args.split_ids, available=available_split_ids)
    split_lookup = {split.split_id: split for split in split_rows_all if split.split_id in set(split_ids)}
    if not layers:
        raise SystemExit("No layers selected")
    if not representations:
        raise SystemExit("No representations selected")
    if not split_lookup:
        raise SystemExit("No splits selected")

    first_feature_path = None
    for layer in layers:
        for representation in representations:
            if (layer, representation) in feature_files:
                first_feature_path = feature_files[(layer, representation)]
                break
        if first_feature_path is not None:
            break
    if first_feature_path is None:
        raise FileNotFoundError("No selected layer/representation feature files exist")
    first_feature = np.load(first_feature_path, mmap_mode="r")
    row_index_map = load_row_index_map(metadata_path, beliefs, int(first_feature.shape[0]))

    coords_writer = ParquetBlockWriter(coords_path)
    centroids_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    pairwise_parts: list[pd.DataFrame] = []
    axes_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    plot_blocks: list[pd.DataFrame] = []
    plot_centroids: list[pd.DataFrame] = []
    best_score = -float("inf")
    best_block: pd.DataFrame | None = None
    best_centroids: pd.DataFrame | None = None
    best_summary: dict[str, Any] | None = None
    attempted = 0
    successful = 0

    try:
        for representation in representations:
            for layer in layers:
                feature_path = feature_files.get((layer, representation))
                if feature_path is None:
                    reason = "missing_feature_file"
                    for split_id in split_ids:
                        skipped.append({"split_id": split_id, "layer": layer, "representation": representation, "reason": reason})
                    print(f"WARNING: missing feature file for layer {layer}, representation {representation}; skipping")
                    continue
                features = np.load(feature_path, mmap_mode="r")
                if int(features.shape[0]) != len(row_index_map):
                    raise ValueError(f"Feature row count mismatch for {feature_path}: {features.shape[0]}")
                for split_id in split_ids:
                    attempted += 1
                    split_rows = split_lookup[int(split_id)]
                    print(f"Processing split={split_id}, layer={layer}, representation={representation}...")
                    try:
                        result = run_block(
                            beliefs=beliefs,
                            extremes=extremes,
                            features=features,
                            row_index_map=row_index_map,
                            split_rows=split_rows,
                            split_id=int(split_id),
                            subset=str(args.subset),
                            layer=int(layer),
                            representation=str(representation),
                            standardize=bool(args.standardize),
                            l2_normalize=bool(args.l2_normalize),
                            min_extreme_per_action=int(args.min_extreme_per_action),
                            model_dir=model_dir,
                        )
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(
                            {
                                "split_id": int(split_id),
                                "layer": int(layer),
                                "representation": str(representation),
                                "reason": str(exc),
                            }
                        )
                        print(f"WARNING: skipped split={split_id}, layer={layer}, representation={representation}: {exc}")
                        continue
                    coords_writer.write(result.coords)
                    centroids_parts.append(result.centroids)
                    metrics_rows.append(result.metrics)
                    pairwise_parts.append(result.pairwise)
                    axes_parts.append(result.axes_metadata)
                    successful += 1
                    if int(split_id) == int(args.plot_split_id):
                        plot_blocks.append(result.coords)
                        plot_centroids.append(result.centroids)
                    score = result.metrics.get("test_rows_balanced_accuracy_vs_agent_action_contrast")
                    if score is None or not math.isfinite(float(score)):
                        score = result.metrics.get("all_rows_balanced_accuracy_vs_agent_action_contrast", -float("inf"))
                    if float(score) > best_score:
                        best_score = float(score)
                        best_block = result.coords.copy()
                        best_centroids = result.centroids.copy()
                        best_summary = result.metrics.copy()
    finally:
        coords_writer.close()

    if coords_writer.rows_written <= 0:
        raise SystemExit("No projection rows were written")

    centroids_df = pd.concat(centroids_parts, ignore_index=True) if centroids_parts else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_rows)
    pairwise_df = pd.concat(pairwise_parts, ignore_index=True) if pairwise_parts else pd.DataFrame()
    axes_df = pd.concat(axes_parts, ignore_index=True) if axes_parts else pd.DataFrame()
    centroids_df.to_parquet(centroids_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    axes_df.to_csv(axes_path, index=False)

    figures: list[Path] = []
    plot_coords = pd.concat(plot_blocks, ignore_index=True) if plot_blocks else pd.DataFrame()
    plot_centroids_df = pd.concat(plot_centroids, ignore_index=True) if plot_centroids else pd.DataFrame()
    if not args.only_plot_best and not plot_coords.empty:
        for representation in representations:
            if representation not in set(plot_coords["representation"].astype(str)):
                continue
            figures.append(
                plot_grid(
                    plot_coords,
                    plot_centroids_df,
                    split_id=int(args.plot_split_id),
                    subset=str(args.subset),
                    representation=representation,
                    layers=layers,
                    mode="extreme_action",
                    output_path=figure_dir
                    / f"action_contrast_grid_split{int(args.plot_split_id)}_{args.subset}_{representation}_by_extreme_action.png",
                    plot_max_points=int(args.plot_max_points),
                    seed=int(args.random_seed),
                )
            )
            figures.append(
                plot_grid(
                    plot_coords,
                    plot_centroids_df,
                    split_id=int(args.plot_split_id),
                    subset=str(args.subset),
                    representation=representation,
                    layers=layers,
                    mode="agent_action",
                    output_path=figure_dir
                    / f"action_contrast_grid_split{int(args.plot_split_id)}_{args.subset}_{representation}_by_agent_action_all_rows.png",
                    plot_max_points=int(args.plot_max_points),
                    seed=int(args.random_seed),
                )
            )
            if "commitment_to_final" in plot_coords.columns:
                figures.append(
                    plot_grid(
                        plot_coords,
                        plot_centroids_df,
                        split_id=int(args.plot_split_id),
                        subset=str(args.subset),
                        representation=representation,
                        layers=layers,
                        mode="commitment",
                        output_path=figure_dir
                        / f"action_contrast_grid_split{int(args.plot_split_id)}_{args.subset}_{representation}_by_commitment_all_rows.png",
                        plot_max_points=int(args.plot_max_points),
                        seed=int(args.random_seed),
                    )
                )
            if "progress" in plot_coords.columns:
                figures.append(
                    plot_grid(
                        plot_coords,
                        plot_centroids_df,
                        split_id=int(args.plot_split_id),
                        subset=str(args.subset),
                        representation=representation,
                        layers=layers,
                        mode="progress",
                        output_path=figure_dir
                        / f"action_contrast_grid_split{int(args.plot_split_id)}_{args.subset}_{representation}_by_progress_all_rows.png",
                        plot_max_points=int(args.plot_max_points),
                        seed=int(args.random_seed),
                    )
                )

    if best_block is not None and best_centroids is not None:
        suffix = (
            f"split={best_summary['split_id']} layer={best_summary['layer']} "
            f"repr={best_summary['representation']} subset={best_summary['subset']}"
            if best_summary
            else "best block"
        )
        figures.append(
            plot_best_single(
                best_block,
                best_centroids,
                color_col="agent_action",
                continuous=False,
                title=f"Best action contrast projection by agent_action | {suffix}",
                output_path=figure_dir / "best_action_contrast_projection_by_agent_action.png",
                plot_max_points=int(args.plot_max_points),
                seed=int(args.random_seed),
            )
        )
        if "top_action" in best_block.columns:
            figures.append(
                plot_best_single(
                    best_block,
                    best_centroids,
                    color_col="top_action",
                    continuous=False,
                    title=f"Best action contrast projection by top_action | {suffix}",
                    output_path=figure_dir / "best_action_contrast_projection_by_top_action.png",
                    plot_max_points=int(args.plot_max_points),
                    seed=int(args.random_seed),
                )
            )
        if "commitment_to_final" in best_block.columns:
            figures.append(
                plot_best_single(
                    best_block,
                    best_centroids,
                    color_col="commitment_to_final",
                    continuous=True,
                    title=f"Best action contrast projection by commitment | {suffix}",
                    output_path=figure_dir / "best_action_contrast_projection_by_commitment.png",
                    plot_max_points=int(args.plot_max_points),
                    seed=int(args.random_seed),
                )
            )
        figures.append(
            plot_best_single(
                best_block,
                best_centroids,
                color_col="extreme_action_for_subset",
                continuous=False,
                title=f"Best action contrast projection: train extremes | {suffix}",
                output_path=figure_dir / "best_action_contrast_projection_train_extremes.png",
                plot_max_points=int(args.plot_max_points),
                seed=int(args.random_seed),
                train_extremes_only=True,
            )
        )

    if not metrics_df.empty:
        metric_for_accuracy = "test_rows_balanced_accuracy_vs_agent_action_contrast"
        if metric_for_accuracy not in metrics_df.columns or metrics_df[metric_for_accuracy].dropna().empty:
            metric_for_accuracy = "all_rows_balanced_accuracy_vs_agent_action_contrast"
        figures.append(
            line_plot(
                metrics_df,
                metric_col=metric_for_accuracy,
                title="Action contrast balanced accuracy by layer",
                ylabel="balanced accuracy",
                output_path=figure_dir / "contrast_accuracy_by_layer.png",
            )
        )
        figures.append(
            line_plot(
                metrics_df,
                metric_col="centroid_pair_spearman_highdim_cosine_vs_contrast_euclidean",
                title="Centroid-pair distance correlation by layer",
                ylabel="Spearman correlation",
                output_path=figure_dir / "contrast_centroid_pair_correlation_by_layer.png",
            )
        )
        metrics_df = metrics_df.copy()
        metrics_df["energy_ratio_1_2"] = metrics_df["energy_ratio_1"] + metrics_df["energy_ratio_2"]
        metrics_df["energy_ratio_1_2_3"] = (
            metrics_df["energy_ratio_1"] + metrics_df["energy_ratio_2"] + metrics_df["energy_ratio_3"]
        )
        figures.append(
            line_plot(
                metrics_df,
                metric_col="energy_ratio_1_2_3",
                title="Action contrast energy captured by three axes",
                ylabel="cumulative energy ratio",
                output_path=figure_dir / "contrast_energy_by_layer.png",
            )
        )

    output_paths = {
        "row-level projection parquet": coords_path,
        "centroid coordinates": centroids_path,
        "metrics": metrics_path,
        "pairwise centroid distances": pairwise_path,
        "axis metadata": axes_path,
        "model directory": model_dir,
    }
    write_report(
        report_path,
        output_paths=output_paths,
        metrics=metrics_df,
        skipped=skipped,
        figures=figures,
        layers=layers,
        representations=representations,
        split_ids=split_ids,
        subset=str(args.subset),
    )

    write_json(
        output_dir / "action_contrast_run_summary.json",
        {
            "blocks_attempted": attempted,
            "blocks_successful": successful,
            "blocks_skipped": len(skipped),
            "skipped": skipped,
            "outputs": {name: rel(path) for name, path in output_paths.items()},
            "report": rel(report_path),
            "figures": [rel(path) for path in figures],
            "best_block": best_summary,
        },
    )

    print("Action-centroid contrast projection complete.")
    print(f"Blocks attempted: {attempted}")
    print(f"Successful blocks: {successful}")
    print(f"Skipped blocks: {len(skipped)}")
    for path in [coords_path, centroids_path, metrics_path, pairwise_path, axes_path, report_path]:
        print(f"Wrote {rel(path)}")
    if best_summary:
        print("Best block summary:")
        print(json.dumps(best_summary, indent=2, sort_keys=True, default=str)[:4000])


if __name__ == "__main__":
    main()
