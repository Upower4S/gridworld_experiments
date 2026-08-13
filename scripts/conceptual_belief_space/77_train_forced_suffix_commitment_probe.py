#!/usr/bin/env python3
"""Train and evaluate the bidirectional last-CoT-window commitment probe."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    ProbeConfig,
    ProjectedShardStore,
    adjacent_layer_delta,
    build_mlp_baseline,
    build_probe,
    entropy_commitment,
)


Q_COLS = [f"q_{action}" for action in ACTIONS]
STRICT_SUBSETS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]

# Positional controls. Commitment rises monotonically with position in the CoT, so a
# probe can score well by decoding position alone. These baselines bound how much of
# any result that explains.
#
# `progress` (= prefix_token_index / num_cot_tokens) and `num_cot_tokens` both depend
# on the total CoT length, which the probe's inputs cannot contain: states at cut t are
# computed from tokens 1..t, and the remaining tokens are not yet generated. Baselines
# using them are *oracle* controls and are not fair comparisons. Only
# `prefix_token_index` is information the probe could plausibly recover on its own, so
# `position_index_fair` is the bar an activation probe is actually required to clear.
POSITIONAL_BASELINES: dict[str, dict[str, Any]] = {
    "position_index_fair": {"features": ["prefix_token_index"], "kind": "fair"},
    "progress_oracle": {"features": ["progress"], "kind": "oracle"},
    "progress_length_spline": {
        "features": ["progress", "prefix_token_index", "num_cot_tokens"],
        "kind": "oracle",
    },
}
# Counts are log1p-compressed before splining; ratios are already bounded.
LOG1P_FEATURES = frozenset({"prefix_token_index", "num_cot_tokens"})
# Fixed edges, so bins are identical across folds and methods and stay comparable.
PROGRESS_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_csv(value: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation_index",
        type=Path,
        default=Path(
            "outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations/activation_index.parquet"
        ),
    )
    parser.add_argument(
        "--beliefs",
        type=Path,
        default=Path("outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("outputs/conceptual_belief_space/splits/group_splits_mlx_full.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/conceptual_belief_space/cot_window_layer_transition_probe"),
    )
    parser.add_argument("--encoder_layers", default="2,4")
    parser.add_argument("--dropouts", default="0.1,0.2")
    parser.add_argument("--lambda_q", default="0,0.1,0.25,0.5")
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baselines", choices=["all", "none"], default="all")
    parser.add_argument("--baseline_max_epochs", type=int, default=30)
    parser.add_argument("--single_layer_stride", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--occlusion_max_rows", type=int, default=4096)
    parser.add_argument("--label_alignment_tolerance", type=float, default=1e-4)
    parser.add_argument("--limit_folds", type=int, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--candidate_limit", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train one candidate for at most two epochs using the first trajectory for train and the second for validation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from atomic per-fold artifacts and model checkpoints. Completed models are loaded "
            "and only missing predictions, diagnostics, or training stages are run."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable live tqdm training progress bars.",
    )
    return parser.parse_args()


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    df.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint_model(path: Path, method: str, device):
    """Restore a trained model, including persistent scalar-normalization buffers."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = dict(payload["config"])
    if "n_input_tokens" not in config_payload and "n_suffix_tokens" in config_payload:
        config_payload["n_input_tokens"] = config_payload.pop("n_suffix_tokens")
    config = ProbeConfig(**config_payload)
    if method == "bidirectional_transformer":
        model = build_probe(config)
    elif method == "transformer_no_deltas":
        config = ProbeConfig(**{**config.__dict__, "use_deltas": False})
        model = build_probe(config)
    elif method == "final_layer_mlp":
        model = build_mlp_baseline(config, mode="final")
    elif method == "mean_layers_mlp":
        model = build_mlp_baseline(config, mode="mean")
    elif method == "concat_layers_mlp":
        model = build_mlp_baseline(config, mode="concat")
    elif method == "best_single_layer_mlp":
        layer = int(payload["selected_physical_layer"])
        model = build_mlp_baseline(config, mode="single", layer_index=layer)
    else:
        raise ValueError(f"cannot restore unknown method {method!r}")
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    model.eval()
    return model, payload, config


def load_or_predict_frame(
    *,
    path: Path,
    model,
    table: pd.DataFrame,
    test_dataset: "ActivationDataset",
    method: str,
    split_id: int,
    batch_size: int,
    device,
) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_parquet(path)
    else:
        prediction = predict_torch_model(model, test_dataset, batch_size=batch_size, device=device)
        frame = predictions_to_frame(table, prediction, method, split_id)
        atomic_parquet(frame, path)
    expected = set(table.iloc[test_dataset.indices]["prefix_id"].astype(str))
    actual = set(frame["prefix_id"].astype(str))
    if actual != expected or frame["prefix_id"].duplicated().any():
        raise RuntimeError(f"{path} does not contain exactly one prediction for every split-{split_id} test row")
    return frame


def prepare_table(
    index_path: Path,
    beliefs_path: Path,
    limit_rows: int | None,
    label_alignment_tolerance: float = 1e-4,
) -> pd.DataFrame:
    index = pd.read_parquet(index_path)
    required_index = [
        "prefix_id",
        "state_path",
        "rms_path",
        "shard_row",
        "shard_n_rows",
        "n_layers",
        "n_input_tokens",
        "projected_dim",
        "input_kind",
        "input_token_mask_json",
    ]
    missing = [column for column in required_index if column not in index.columns]
    if missing:
        raise ValueError(f"activation index is missing columns: {missing}")
    beliefs = pd.read_parquet(beliefs_path)
    required_beliefs = ["prefix_id", "trajectory_id", "decision_id", "progress", "prefix_token_index", *Q_COLS]
    missing = [column for column in required_beliefs if column not in beliefs.columns]
    if missing:
        raise ValueError(f"belief table is missing columns: {missing}")
    keep = list(dict.fromkeys(required_beliefs + ["num_cot_tokens", *STRICT_SUBSETS, "top_action", "agent_action"]))
    keep = [column for column in keep if column in beliefs.columns]
    table = index.merge(beliefs[keep], on="prefix_id", how="inner", validate="one_to_one", suffixes=("_activation", ""))
    if limit_rows is not None:
        table = table.head(int(limit_rows))
    table = table.sort_values("prefix_id").reset_index(drop=True)
    if table.empty:
        raise ValueError("no activation rows align with the belief table")
    input_kinds = table["input_kind"].astype(str).drop_duplicates().tolist()
    if input_kinds != ["cot_last_tokens"]:
        raise ValueError(
            "probe training accepts only input_kind='cot_last_tokens'; forced-suffix states are "
            f"label-leaking inputs and were found in this index: {input_kinds}"
        )
    for dimension in ["n_layers", "n_input_tokens", "projected_dim"]:
        values = table[dimension].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(f"activation index contains multiple {dimension} values: {values}")
    q = table[Q_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("q labels contain invalid values")
    q /= q.sum(axis=1, keepdims=True)
    table.loc[:, Q_COLS] = q
    extracted_cols = [f"extracted_q_{action}" for action in ACTIONS]
    if all(column in table.columns for column in extracted_cols):
        extracted_q = table[extracted_cols].to_numpy(dtype=np.float64)
        extracted_q /= extracted_q.sum(axis=1, keepdims=True)
        alignment_error = np.max(np.abs(q - extracted_q), axis=1)
        table["label_alignment_error"] = alignment_error
        max_error = float(np.max(alignment_error))
        if max_error > float(label_alignment_tolerance):
            raise ValueError(
                f"belief labels disagree with activation-extraction logits by {max_error}, above "
                f"{label_alignment_tolerance}; use the extractor's extracted_beliefs.parquet or reproduce "
                "the original model/runtime revision"
            )
    table["entropy_commitment"] = entropy_commitment(q)
    expected_width = int(table["n_input_tokens"].iloc[0])
    parsed_masks = []
    for prefix_id, raw_mask in zip(
        table["prefix_id"].tolist(), table["input_token_mask_json"].tolist(), strict=True
    ):
        try:
            mask = [int(value) for value in json.loads(str(raw_mask))]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"prefix_id={prefix_id} has a malformed input token mask") from exc
        if len(mask) != expected_width or any(value not in {0, 1} for value in mask):
            raise ValueError(
                f"prefix_id={prefix_id} input token mask must contain {expected_width} binary values"
            )
        if not any(mask):
            raise ValueError(f"prefix_id={prefix_id} has no genuine CoT token in its input window")
        if mask != sorted(mask):
            raise ValueError(f"prefix_id={prefix_id} CoT window mask must be right-aligned")
        parsed_masks.append(mask)
    table["_input_token_mask"] = parsed_masks
    if "num_cot_tokens" not in table.columns:
        table["num_cot_tokens"] = table.groupby("decision_id")["prefix_token_index"].transform("max")
    for subset in STRICT_SUBSETS:
        if subset not in table.columns:
            table[subset] = False
        table[subset] = table[subset].fillna(False).astype(bool)
    return table


def load_outer_splits(path: Path, table: pd.DataFrame) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("splits", payload.get("folds", []))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"could not find split rows in {path}")
    all_rows = np.arange(len(table), dtype=np.int64)
    table_groups = table["trajectory_id"].astype(str).to_numpy()
    splits = []
    for position, item in enumerate(payload):
        split_id = int(item.get("split_id", item.get("fold", position)))
        test_groups = set(str(x) for x in item.get("test_groups", []))
        if not test_groups:
            raise ValueError(f"split {split_id} has no test_groups")
        test_mask = np.asarray([group in test_groups for group in table_groups], dtype=bool)
        train_idx = all_rows[~test_mask]
        test_idx = all_rows[test_mask]
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"split {split_id} becomes empty after activation alignment")
        overlap = set(table.iloc[train_idx]["trajectory_id"].astype(str)) & set(
            table.iloc[test_idx]["trajectory_id"].astype(str)
        )
        if overlap:
            raise RuntimeError(f"trajectory leakage in split {split_id}: {sorted(overlap)[:5]}")
        splits.append({"split_id": split_id, "train_idx": train_idx, "test_idx": test_idx})
    return splits


def inner_group_split(table: pd.DataFrame, outer_train: np.ndarray, fraction: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit

    groups = table.iloc[outer_train]["trajectory_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(fraction), random_state=int(seed))
    local_train, local_val = next(splitter.split(outer_train, groups=groups))
    train_idx = outer_train[local_train]
    val_idx = outer_train[local_val]
    overlap = set(table.iloc[train_idx]["trajectory_id"].astype(str)) & set(
        table.iloc[val_idx]["trajectory_id"].astype(str)
    )
    if overlap:
        raise RuntimeError(f"inner split trajectory leakage: {sorted(overlap)[:5]}")
    return train_idx, val_idx


def decision_weights(table: pd.DataFrame, indices: np.ndarray) -> dict[int, float]:
    decisions = table.iloc[indices]["decision_id"].astype(str)
    counts = decisions.value_counts()
    raw = np.asarray([1.0 / counts[decision] for decision in decisions], dtype=np.float64)
    raw /= raw.mean()
    return {int(index): float(weight) for index, weight in zip(indices, raw, strict=True)}


class ActivationDataset:
    def __init__(self, table: pd.DataFrame, indices: Sequence[int], weights: Mapping[int, float] | None = None):
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = weights or {}
        self.store = ProjectedShardStore(max_open=16)
        record_columns = [
            "state_path",
            "rms_path",
            "shard_row",
            "shard_n_rows",
            "n_layers",
            "n_input_tokens",
            "projected_dim",
        ]
        selected = table.iloc[self.indices]
        self.records = selected[record_columns].to_dict(orient="records")
        self.q = selected[Q_COLS].to_numpy(dtype=np.float32)
        self.commitment = selected["entropy_commitment"].to_numpy(dtype=np.float32)
        self.token_masks = np.asarray(selected["_input_token_mask"].tolist(), dtype=np.float32)
        self.row_weights = np.asarray(
            [float(self.weights.get(int(row_idx), 1.0)) for row_idx in self.indices], dtype=np.float32
        )
        self._positions_by_shard: list[np.ndarray] = []
        shard_positions: dict[str, list[int]] = {}
        for position, record in enumerate(self.records):
            shard_positions.setdefault(str(record["state_path"]), []).append(position)
        self._positions_by_shard = [
            np.asarray(positions, dtype=np.int64) for positions in shard_positions.values()
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def num_batches(self, batch_size: int) -> int:
        width = max(1, int(batch_size))
        return int(sum(math.ceil(len(positions) / width) for positions in self._positions_by_shard))

    def __getitem__(self, item: int):
        import torch

        position = int(item)
        row_idx = int(self.indices[position])
        projected, rms = self.store.load_row(self.records[position])
        return {
            "projected": torch.from_numpy(np.asarray(projected, dtype=np.float32).copy()),
            "rms": torch.from_numpy(np.asarray(rms, dtype=np.float32).copy()),
            "q": torch.from_numpy(self.q[position].copy()),
            "commitment": torch.tensor(float(self.commitment[position]), dtype=torch.float32),
            "weight": torch.tensor(float(self.row_weights[position]), dtype=torch.float32),
            "token_mask": torch.from_numpy(self.token_masks[position].copy()),
            "row_idx": torch.tensor(row_idx, dtype=torch.int64),
        }

    def iter_batches(self, *, batch_size: int, shuffle: bool, seed: int = 0):
        """Yield vectorized shard-local batches in randomized batch order."""
        import torch

        rng = np.random.default_rng(int(seed))
        batches: list[np.ndarray] = []
        for source_positions in self._positions_by_shard:
            positions = source_positions.copy()
            if shuffle:
                rng.shuffle(positions)
            batches.extend(
                positions[start : start + int(batch_size)]
                for start in range(0, len(positions), int(batch_size))
            )
        if shuffle:
            rng.shuffle(batches)
        for positions in batches:
            position_list = positions.tolist()
            records = [self.records[position] for position in position_list]
            projected, rms = self.store.load_rows(records)
            yield {
                "projected": torch.from_numpy(np.asarray(projected, dtype=np.float32)),
                "rms": torch.from_numpy(np.asarray(rms, dtype=np.float32)),
                "q": torch.from_numpy(self.q[positions].copy()),
                "commitment": torch.from_numpy(self.commitment[positions].copy()),
                "weight": torch.from_numpy(self.row_weights[positions].copy()),
                "token_mask": torch.from_numpy(self.token_masks[positions].copy()),
                "row_idx": torch.from_numpy(self.indices[positions].copy()),
            }


def fit_scalar_normalization(table: pd.DataFrame, indices: np.ndarray, chunk_size: int = 32):
    store = ProjectedShardStore(max_open=16)
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), int(chunk_size)):
        projected_rows = []
        rms_rows = []
        token_masks = []
        for row_idx in indices[start : start + int(chunk_size)]:
            projected, rms = store.load_row(table.iloc[int(row_idx)])
            projected_rows.append(projected)
            rms_rows.append(rms)
            token_masks.append(table.iloc[int(row_idx)]["_input_token_mask"])
        projected = np.asarray(projected_rows, dtype=np.float32)
        rms = np.asarray(rms_rows, dtype=np.float32)
        delta = adjacent_layer_delta(projected)
        delta_rms = np.sqrt(np.maximum(np.mean(delta.astype(np.float64) ** 2, axis=-1), 1e-12))
        values_4d = np.stack(
            [np.log(np.maximum(rms, 1e-8)), np.log(np.maximum(delta_rms, 1e-8))], axis=-1
        )
        valid_tokens = np.asarray(token_masks, dtype=bool)[:, None, :]
        valid_tokens = np.broadcast_to(valid_tokens, values_4d.shape[:-1]).reshape(-1)
        values = values_4d.reshape(-1, 2)
        values = values[valid_tokens & np.isfinite(values).all(axis=1)]
        total += values.sum(axis=0)
        total_sq += (values**2).sum(axis=0)
        count += len(values)
    if count == 0:
        raise ValueError("no finite activation scalar features in training rows")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    if valid.sum() < 3 or np.unique(true[valid]).size < 2 or np.unique(pred[valid]).size < 2:
        return float("nan")
    return float(spearmanr(true[valid], pred[valid]).statistic)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    y = frame["entropy_commitment"].to_numpy(dtype=np.float64)
    pred = frame["pred_commitment"].to_numpy(dtype=np.float64)
    q = frame[Q_COLS].to_numpy(dtype=np.float64)
    qhat_cols = [f"pred_q_{action}" for action in ACTIONS]
    qhat = frame[qhat_cols].to_numpy(dtype=np.float64)
    out = {
        "n_rows": int(len(frame)),
        "spearman": safe_spearman(y, pred),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(frame) >= 2 and np.std(pred) > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(frame) >= 2 else float("nan"),
        "q_kl": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-8, None)) - np.log(np.clip(qhat, 1e-8, None))), axis=1))),
        "q_brier": float(np.mean(np.sum((q - qhat) ** 2, axis=1))),
        "top_action_accuracy": float(np.mean(np.argmax(q, axis=1) == np.argmax(qhat, axis=1))),
    }
    extreme = (y <= 0.4) | (y >= 0.9)
    labels = (y[extreme] >= 0.9).astype(int)
    out["high_low_auc"] = (
        float(roc_auc_score(labels, pred[extreme])) if len(np.unique(labels)) == 2 else float("nan")
    )
    return out


def progress_bin_scopes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Position-matched evaluation scopes over fixed progress bins.

    Pooled scores can be inflated by a probe that only decodes where it is in the CoT,
    because commitment climbs with position. Scoring inside a narrow band holds position
    roughly constant: a probe that keeps its correlation within bins is reading something
    other than position, while one that collapses was largely a position decoder.
    """
    if "progress" not in frame.columns:
        return {}
    progress = frame["progress"].to_numpy(dtype=np.float64)
    scopes: dict[str, np.ndarray] = {}
    for lower, upper in zip(PROGRESS_BIN_EDGES[:-1], PROGRESS_BIN_EDGES[1:], strict=True):
        # The last bin is closed so that progress == 1.0 is not dropped.
        if upper >= PROGRESS_BIN_EDGES[-1]:
            mask = (progress >= lower) & (progress <= upper)
        else:
            mask = (progress >= lower) & (progress < upper)
        scopes[f"progress_bin_{lower:.1f}_{upper:.1f}"] = mask
    return scopes


def evaluate_prediction_frame(frame: pd.DataFrame, *, method: str, split_id: int) -> list[dict[str, Any]]:
    rows = []
    scopes = {"all_prefixes": np.ones(len(frame), dtype=bool)}
    for subset in STRICT_SUBSETS:
        if subset in frame.columns:
            scopes[subset] = frame[subset].fillna(False).astype(bool).to_numpy()
    scopes.update(progress_bin_scopes(frame))
    for scope, mask in scopes.items():
        if not mask.any():
            continue
        rows.append({"method": method, "split_id": int(split_id), "subset": scope, **prediction_metrics(frame.loc[mask])})
    return rows


def predict_torch_model(
    model,
    dataset: ActivationDataset,
    *,
    batch_size: int,
    device,
    ablate_layer: int | None = None,
    ablate_token_position: int | None = None,
) -> dict[str, np.ndarray]:
    import torch
    commitments = []
    q_probs = []
    row_indices = []
    model.eval()
    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size=int(batch_size), shuffle=False):
            projected = batch["projected"].to(device)
            rms = batch["rms"].to(device)
            token_mask = batch["token_mask"].to(device)
            layer_mask = None
            if ablate_layer is not None:
                layer_mask = torch.ones((len(projected), projected.shape[1]), dtype=projected.dtype, device=device)
                layer_mask[:, int(ablate_layer)] = 0.0
            if ablate_token_position is not None:
                token_mask = token_mask.clone()
                token_mask[:, int(ablate_token_position)] = 0.0
            output = model(
                projected,
                rms,
                layer_keep_mask=layer_mask,
                token_keep_mask=token_mask,
            )
            commitments.append(output["commitment"].detach().cpu().numpy())
            q_probs.append(torch.softmax(output["q_logits"], dim=-1).detach().cpu().numpy())
            row_indices.append(batch["row_idx"].numpy())
    return {
        "row_idx": np.concatenate(row_indices),
        "commitment": np.concatenate(commitments),
        "q": np.concatenate(q_probs),
    }


def train_torch_model(
    model,
    train_dataset: ActivationDataset,
    val_dataset: ActivationDataset | None,
    *,
    lambda_q: float,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    grad_clip: float,
    device,
    seed: int,
    progress_label: str = "training",
    show_progress: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:  # pragma: no cover - dependency-free fallback
        tqdm = None
    seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_spearman = -float("inf")
    best_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    epoch_range = range(int(max_epochs))
    if tqdm is not None:
        epoch_range = tqdm(
            epoch_range,
            desc=progress_label,
            unit="epoch",
            dynamic_ncols=True,
            disable=not bool(show_progress),
        )
    for epoch in epoch_range:
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        batch_iterator = train_dataset.iter_batches(
            batch_size=int(batch_size), shuffle=True, seed=int(seed) + epoch
        )
        if tqdm is not None:
            batch_iterator = tqdm(
                batch_iterator,
                total=train_dataset.num_batches(batch_size),
                desc=f"{progress_label} e{epoch + 1}/{max_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=not bool(show_progress),
            )
        for batch in batch_iterator:
            projected = batch["projected"].to(device)
            rms = batch["rms"].to(device)
            target_c = batch["commitment"].to(device)
            target_q = batch["q"].to(device)
            weights = batch["weight"].to(device)
            token_mask = batch["token_mask"].to(device)
            output = model(projected, rms, token_keep_mask=token_mask)
            commitment_loss = F.smooth_l1_loss(
                output["commitment"], target_c, beta=0.1, reduction="none"
            )
            log_qhat = F.log_softmax(output["q_logits"], dim=-1)
            q_loss = torch.sum(target_q * (torch.log(target_q.clamp_min(1e-8)) - log_qhat), dim=-1)
            row_loss = commitment_loss + float(lambda_q) * q_loss
            loss = torch.sum(row_loss * weights) / weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            total_loss += float(torch.sum(row_loss.detach() * weights).cpu())
            total_weight += float(weights.sum().detach().cpu())
        epoch_row = {"epoch": epoch, "train_loss": total_loss / max(total_weight, 1e-8)}
        if val_dataset is not None:
            val_pred = predict_torch_model(model, val_dataset, batch_size=batch_size, device=device)
            val_rows = val_dataset.table.iloc[val_pred["row_idx"]]
            val_spearman = safe_spearman(val_rows["entropy_commitment"], val_pred["commitment"])
            val_mae = float(np.mean(np.abs(val_rows["entropy_commitment"].to_numpy() - val_pred["commitment"])))
            epoch_row.update({"val_spearman": val_spearman, "val_mae": val_mae})
            score = val_spearman if np.isfinite(val_spearman) else -float("inf")
            improved = best_state is None or score > best_spearman + 1e-8 or (
                np.isfinite(score)
                and np.isfinite(best_spearman)
                and abs(score - best_spearman) <= 1e-8
                and val_mae < best_mae
            ) or (not np.isfinite(score) and not np.isfinite(best_spearman) and val_mae < best_mae)
            if improved:
                best_spearman = score
                best_mae = val_mae
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    history.append(epoch_row)
                    break
        else:
            best_epoch = epoch
        history.append(epoch_row)
        if tqdm is not None and hasattr(epoch_range, "set_postfix"):
            postfix = {"loss": f"{epoch_row['train_loss']:.4f}"}
            if "val_spearman" in epoch_row:
                postfix.update(
                    {
                        "val_rho": f"{epoch_row['val_spearman']:.4f}",
                        "best_rho": f"{best_spearman:.4f}",
                        "patience": f"{bad_epochs}/{patience}",
                    }
                )
            epoch_range.set_postfix(postfix)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_spearman": float(best_spearman),
        "best_val_mae": float(best_mae),
        "epochs_ran": int(len(history)),
        "history": history,
    }


def predictions_to_frame(table: pd.DataFrame, prediction: dict[str, np.ndarray], method: str, split_id: int):
    columns = [
        "prefix_id",
        "trajectory_id",
        "decision_id",
        "progress",
        "prefix_token_index",
        "num_cot_tokens",
        *STRICT_SUBSETS,
        *Q_COLS,
        "entropy_commitment",
    ]
    columns = [column for column in columns if column in table.columns]
    frame = table.iloc[prediction["row_idx"]][columns].copy().reset_index(drop=True)
    frame.insert(0, "split_id", int(split_id))
    frame.insert(0, "method", method)
    frame["pred_commitment"] = prediction["commitment"].astype(np.float32)
    for action_idx, action in enumerate(ACTIONS):
        frame[f"pred_q_{action}"] = prediction["q"][:, action_idx].astype(np.float32)
    return frame


def fit_selected_torch_model(
    *,
    builder: Callable[[], Any],
    table: pd.DataFrame,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    outer_train: np.ndarray,
    inner_scalar_mean: np.ndarray,
    inner_scalar_std: np.ndarray,
    full_scalar_mean: np.ndarray,
    full_scalar_std: np.ndarray,
    lambda_q: float,
    args: argparse.Namespace,
    device,
    seed: int,
    max_epochs: int,
) -> tuple[Any, dict[str, Any]]:
    weights = decision_weights(table, inner_train)
    train_ds = ActivationDataset(table, inner_train, weights)
    val_ds = ActivationDataset(table, inner_val)
    selection_model = builder()
    selection_model.set_scalar_normalization(inner_scalar_mean, inner_scalar_std)
    selection_model, selection_info = train_torch_model(
        selection_model,
        train_ds,
        val_ds,
        lambda_q=lambda_q,
        max_epochs=max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        device=device,
        seed=seed,
        progress_label="model selection",
        show_progress=not args.no_progress,
    )
    selected_epochs = max(1, int(selection_info["best_epoch"]) + 1)
    full_model = builder()
    full_model.set_scalar_normalization(full_scalar_mean, full_scalar_std)
    full_weights = decision_weights(table, outer_train)
    full_ds = ActivationDataset(table, outer_train, full_weights)
    full_model, full_info = train_torch_model(
        full_model,
        full_ds,
        None,
        lambda_q=lambda_q,
        max_epochs=selected_epochs,
        patience=selected_epochs + 1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        device=device,
        seed=seed,
        progress_label="outer-fold retrain",
        show_progress=not args.no_progress,
    )
    return full_model, {"selection": selection_info, "retrain": full_info, "selected_epochs": selected_epochs}


def fit_positional_baseline(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: Sequence[str],
):
    """Fit a spline-ridge control that sees only positional metadata, no activations."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    feature_cols = list(feature_cols)
    missing = [column for column in feature_cols if column not in table.columns]
    if missing:
        raise ValueError(f"positional baseline is missing columns: {missing}")
    log_columns = np.array([column in LOG1P_FEATURES for column in feature_cols])

    def features(indices):
        x = table.iloc[indices][feature_cols].to_numpy(dtype=np.float64)
        if log_columns.any():
            x[:, log_columns] = np.log1p(np.maximum(x[:, log_columns], 0.0))
        return x

    # The second scaling keeps the spline design matrix well conditioned; LSQR
    # avoids the overflow-prone normal-equation path seen on the full dataset.
    c_model = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3),
        StandardScaler(),
        Ridge(alpha=1.0, solver="lsqr"),
    )
    q_model = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3),
        StandardScaler(),
        Ridge(alpha=1.0, solver="lsqr"),
    )
    c_model.fit(features(train_idx), table.iloc[train_idx]["entropy_commitment"])
    q_model.fit(features(train_idx), table.iloc[train_idx][Q_COLS])
    c_pred = np.clip(c_model.predict(features(test_idx)), 0.0, 1.0)
    q_pred = np.clip(q_model.predict(features(test_idx)), 1e-8, None)
    if not np.isfinite(c_pred).all() or not np.isfinite(q_pred).all():
        raise RuntimeError(f"positional baseline {feature_cols} produced non-finite predictions")
    q_pred /= q_pred.sum(axis=1, keepdims=True)
    return {"row_idx": test_idx, "commitment": c_pred.astype(np.float32), "q": q_pred.astype(np.float32)}


def run_occlusions(model, table, test_idx, config: ProbeConfig, args, device, split_id: int):
    if int(args.occlusion_max_rows) <= 0:
        return []
    rng = np.random.default_rng(args.seed + int(split_id))
    selected = np.asarray(test_idx, dtype=np.int64)
    if len(selected) > int(args.occlusion_max_rows):
        selected = np.sort(rng.choice(selected, size=int(args.occlusion_max_rows), replace=False))
    dataset = ActivationDataset(table, selected)
    base = predict_torch_model(model, dataset, batch_size=args.batch_size, device=device)
    truth = table.iloc[base["row_idx"]]["entropy_commitment"].to_numpy(dtype=np.float64)
    base_spearman = safe_spearman(truth, base["commitment"])
    base_mae = float(np.mean(np.abs(truth - base["commitment"])))
    rows = []
    for layer in range(config.n_layers):
        pred = predict_torch_model(
            model, dataset, batch_size=args.batch_size, device=device, ablate_layer=layer
        )
        rows.append(
            {
                "split_id": split_id,
                "axis": "physical_layer",
                "position": layer,
                "n_rows": len(selected),
                "base_spearman": base_spearman,
                "occluded_spearman": safe_spearman(truth, pred["commitment"]),
                "base_mae": base_mae,
                "occluded_mae": float(np.mean(np.abs(truth - pred["commitment"]))),
            }
        )
    for token_position in range(config.n_input_tokens):
        pred = predict_torch_model(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
            ablate_token_position=token_position,
        )
        rows.append(
            {
                "split_id": split_id,
                "axis": "cot_window_token",
                "position": token_position,
                "relative_to_prefix_end": token_position - (config.n_input_tokens - 1),
                "n_rows": len(selected),
                "base_spearman": base_spearman,
                "occluded_spearman": safe_spearman(truth, pred["commitment"]),
                "base_mae": base_mae,
                "occluded_mae": float(np.mean(np.abs(truth - pred["commitment"]))),
            }
        )
    for row in rows:
        row["spearman_drop"] = row["base_spearman"] - row["occluded_spearman"]
        row["mae_increase"] = row["occluded_mae"] - row["base_mae"]
    return rows


def clustered_bootstrap(predictions: pd.DataFrame, *, main_method: str, samples: int, seed: int):
    if samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows = []
    main = predictions[predictions["method"].eq(main_method)]
    for comparison in sorted(set(predictions["method"]) - {main_method}):
        other = predictions[predictions["method"].eq(comparison)]
        merged = main.merge(
            other[["prefix_id", "pred_commitment"]],
            on="prefix_id",
            how="inner",
            suffixes=("_main", "_other"),
        )
        groups = sorted(merged["trajectory_id"].astype(str).unique())
        if len(groups) < 2:
            continue
        by_group = {group: merged[merged["trajectory_id"].astype(str).eq(group)] for group in groups}
        spearman_delta = []
        mae_delta = []
        for _ in range(int(samples)):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sample = pd.concat([by_group[str(group)] for group in sampled_groups], ignore_index=True)
            truth = sample["entropy_commitment"].to_numpy(dtype=float)
            main_pred = sample["pred_commitment_main"].to_numpy(dtype=float)
            other_pred = sample["pred_commitment_other"].to_numpy(dtype=float)
            spearman_delta.append(safe_spearman(truth, main_pred) - safe_spearman(truth, other_pred))
            mae_delta.append(np.mean(np.abs(truth - other_pred)) - np.mean(np.abs(truth - main_pred)))
        for metric, values in [("spearman_improvement", spearman_delta), ("mae_reduction", mae_delta)]:
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "main_method": main_method,
                    "comparison_method": comparison,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "ci_low": float(np.nanpercentile(values, 2.5)),
                    "ci_high": float(np.nanpercentile(values, 97.5)),
                    "bootstrap_samples": int(samples),
                }
            )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, metrics: pd.DataFrame, selected: list[dict[str, Any]], bootstrap: pd.DataFrame):
    all_rows = metrics[metrics["subset"].eq("all_prefixes")]
    fold_rows = all_rows[all_rows["split_id"].ge(0)]
    summary = (
        fold_rows.groupby("method", as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_spearman=("spearman", "mean"),
            std_spearman=("spearman", "std"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_q_kl=("q_kl", "mean"),
            mean_top_action_accuracy=("top_action_accuracy", "mean"),
        )
        .sort_values(["mean_spearman", "mean_mae"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    lines = [
        "# Bidirectional Last-CoT-Window Commitment Probe",
        "",
        "The primary score is trajectory-grouped held-out Spearman correlation for normalized entropy commitment.",
        "Probe inputs are the last genuine CoT-token states before the forced label suffix. All Transformer attention is bidirectional; layer and token occlusions are predictive feature-importance diagnostics.",
        "",
        "## Outer-Fold Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Pooled Out-of-Fold Scores",
        "",
        all_rows[all_rows["split_id"].eq(-1)].to_markdown(index=False),
        "",
        "## Selected Configurations",
        "",
    ]
    for item in selected:
        lines.append(
            f"- split {item['split_id']}: layers={item['encoder_layers']}, dropout={item['dropout']}, "
            f"lambda_q={item['lambda_q']}, validation Spearman={item['validation_spearman']:.6f}, "
            f"epochs={item['selected_epochs']}"
        )
    if not bootstrap.empty:
        lines.extend(["", "## Clustered Bootstrap Comparisons", "", bootstrap.to_markdown(index=False)])
    bin_rows = metrics[metrics["subset"].str.startswith("progress_bin_") & metrics["split_id"].eq(-1)]
    if not bin_rows.empty:
        pivot = bin_rows.pivot_table(index="method", columns="subset", values="spearman").reset_index()
        lines.extend(
            [
                "",
                "## Position-Matched Scores (pooled out-of-fold Spearman)",
                "",
                "Held-out Spearman inside fixed progress bins. Commitment climbs with position in the "
                "CoT, so a probe that only decodes position scores well pooled and collapses here.",
                "",
                pivot.to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "## Positional Controls",
            "",
            "`position_index_fair` is the bar an activation probe must clear: `prefix_token_index` is "
            "the only positional quantity its inputs could encode. `progress_oracle` and "
            "`progress_length_spline` additionally use the total CoT length, which states at cut t "
            "cannot contain because the remaining tokens are not yet generated; they are oracle "
            "references, not fair comparisons.",
            "",
            "## Interpretation",
            "",
            "This experiment optimizes prediction. Strong performance shows that commitment is encoded in the joint pre-suffix CoT-window layer states; it does not identify a causal layer at which commitment forms. The forced suffix is used only to compute labels and is excluded from probe inputs.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n")


def run_two_trajectory_smoke(
    *,
    table: pd.DataFrame,
    output_dir: Path,
    candidate: tuple[int, float, float],
    args: argparse.Namespace,
    device,
) -> int:
    trajectories = table["trajectory_id"].astype(str).drop_duplicates().tolist()
    if len(trajectories) < 2:
        raise ValueError("--smoke requires activation rows from at least two trajectories")
    train_idx = np.flatnonzero(table["trajectory_id"].astype(str).eq(trajectories[0]).to_numpy())
    val_idx = np.flatnonzero(table["trajectory_id"].astype(str).eq(trajectories[1]).to_numpy())
    mean, std = fit_scalar_normalization(table, train_idx)
    encoder_layers, dropout, lambda_q = candidate
    config = ProbeConfig(
        projected_dim=int(table["projected_dim"].iloc[0]),
        n_layers=int(table["n_layers"].iloc[0]),
        n_input_tokens=int(table["n_input_tokens"].iloc[0]),
        token_dim=args.token_dim,
        d_model=args.d_model,
        encoder_layers=encoder_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=dropout,
        use_deltas=True,
    )
    model = build_probe(config)
    model.set_scalar_normalization(mean, std)
    model, info = train_torch_model(
        model,
        ActivationDataset(table, train_idx, decision_weights(table, train_idx)),
        ActivationDataset(table, val_idx),
        lambda_q=lambda_q,
        max_epochs=min(2, int(args.max_epochs)),
        patience=2,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=min(args.batch_size, max(1, len(train_idx))),
        grad_clip=args.grad_clip,
        device=device,
        seed=args.seed,
        progress_label="two-trajectory smoke",
        show_progress=not args.no_progress,
    )
    prediction = predict_torch_model(
        model,
        ActivationDataset(table, val_idx),
        batch_size=min(args.batch_size, max(1, len(val_idx))),
        device=device,
    )
    frame = predictions_to_frame(table, prediction, "bidirectional_transformer_smoke", 0)
    atomic_parquet(frame, output_dir / "smoke_predictions.parquet")
    metrics = prediction_metrics(frame)
    payload = {
        "status": "complete",
        "train_trajectory": trajectories[0],
        "validation_trajectory": trajectories[1],
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "config": config.__dict__,
        "lambda_q": lambda_q,
        "training": info,
        "metrics": metrics,
        "bidirectional_attention": True,
    }
    atomic_json(output_dir / "smoke_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and not args.force and not args.resume:
        raise SystemExit(
            f"{metrics_path} already exists; pass --resume to reuse checkpoints or --force to replace outputs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_artifact_root = output_dir / "fold_artifacts"
    fold_artifact_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    seed_everything(args.seed)
    table = prepare_table(
        args.activation_index,
        args.beliefs,
        args.limit_rows,
        label_alignment_tolerance=args.label_alignment_tolerance,
    )
    n_layers = int(table["n_layers"].iloc[0])
    n_input_tokens = int(table["n_input_tokens"].iloc[0])
    projected_dim = int(table["projected_dim"].iloc[0])
    candidate_grid = list(
        itertools.product(
            parse_csv(args.encoder_layers, int),
            parse_csv(args.dropouts, float),
            parse_csv(args.lambda_q, float),
        )
    )
    if args.candidate_limit is not None:
        candidate_grid = candidate_grid[: int(args.candidate_limit)]
    if not candidate_grid:
        raise ValueError("empty Transformer candidate grid")
    if args.smoke:
        return run_two_trajectory_smoke(
            table=table,
            output_dir=output_dir,
            candidate=candidate_grid[0],
            args=args,
            device=device,
        )
    outer_splits = load_outer_splits(args.splits, table)
    if args.limit_folds is not None:
        outer_splits = outer_splits[: int(args.limit_folds)]
    print(
        json.dumps(
            {
                "device": str(device),
                "rows": len(table),
                "folds": len(outer_splits),
                "candidate_count": len(candidate_grid),
                "shape": [n_layers, n_input_tokens, projected_dim],
                "input_kind": "cot_last_tokens",
            },
            indent=2,
        ),
        flush=True,
    )

    validation_rows: list[dict[str, Any]] = []
    selected_configs: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    occlusion_rows: list[dict[str, Any]] = []
    for split in outer_splits:
        split_id = int(split["split_id"])
        outer_train = split["train_idx"]
        outer_test = split["test_idx"]
        fold_dir = fold_artifact_root / f"split{split_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_complete_path = fold_dir / "fold_complete.json"
        if args.resume and fold_complete_path.exists():
            complete_payload = json.loads(fold_complete_path.read_text())
            if complete_payload.get("baselines") == args.baselines:
                fold_prediction_paths = sorted(fold_dir.glob("predictions_*.parquet"))
                if not fold_prediction_paths:
                    raise RuntimeError(f"{fold_complete_path} exists without prediction artifacts")
                prediction_frames.extend(pd.read_parquet(path) for path in fold_prediction_paths)
                for validation_name in ["transformer_validation.csv", "single_layer_validation.csv"]:
                    validation_path = fold_dir / validation_name
                    if validation_path.exists():
                        validation_rows.extend(pd.read_csv(validation_path).to_dict(orient="records"))
                selected_payload = json.loads((fold_dir / "selected_config.json").read_text())
                selected_configs.append(
                    {
                        **selected_payload["selection"],
                        "selected_epochs": int(selected_payload["selected_epochs"]),
                        "scalar_mean": selected_payload["scalar_mean"],
                        "scalar_std": selected_payload["scalar_std"],
                        "retrain_epochs": int(selected_payload["selected_epochs"]),
                        "restored_from_fold_artifacts": True,
                    }
                )
                occlusion_path = fold_dir / "occlusions.csv"
                if occlusion_path.exists():
                    occlusion_rows.extend(pd.read_csv(occlusion_path).to_dict(orient="records"))
                print(f"Split {split_id}: restored complete fold artifacts", flush=True)
                continue
        test_dataset = ActivationDataset(table, outer_test)
        inner_train, inner_val = inner_group_split(
            table, outer_train, args.validation_fraction, args.seed + split_id
        )
        scalar_mean, scalar_std = fit_scalar_normalization(table, inner_train)
        transformer_validation_path = fold_dir / "transformer_validation.csv"
        selected_path = fold_dir / "selected_config.json"
        main_checkpoint_path = model_dir / f"bidirectional_probe_split{split_id}.pt"

        if args.resume and main_checkpoint_path.exists():
            main_model, checkpoint, config = load_checkpoint_model(
                main_checkpoint_path, "bidirectional_transformer", device
            )
            if int(checkpoint.get("split_id", split_id)) != split_id:
                raise RuntimeError(f"checkpoint split mismatch in {main_checkpoint_path}")
            selected_epochs = int(checkpoint["selected_epochs"])
            full_scalar_mean = np.asarray(checkpoint["scalar_mean"], dtype=np.float32)
            full_scalar_std = np.asarray(checkpoint["scalar_std"], dtype=np.float32)
            if selected_path.exists():
                selected_payload = json.loads(selected_path.read_text())
                selected = selected_payload["selection"]
            else:
                selected = {
                    "split_id": split_id,
                    "candidate_id": -1,
                    "encoder_layers": int(config.encoder_layers),
                    "dropout": float(config.dropout),
                    "lambda_q": float(checkpoint["lambda_q"]),
                    "validation_spearman": float("nan"),
                    "validation_mae": float("nan"),
                    "best_epoch": selected_epochs - 1,
                    "recovered_from_checkpoint": True,
                }
                atomic_json(
                    selected_path,
                    {
                        "selection": selected,
                        "config": config.__dict__,
                        "selected_epochs": selected_epochs,
                        "scalar_mean": full_scalar_mean.tolist(),
                        "scalar_std": full_scalar_std.tolist(),
                        "recovered_from_checkpoint": True,
                    },
                )
                if not transformer_validation_path.exists():
                    atomic_csv(pd.DataFrame([selected]), transformer_validation_path)
            if not transformer_validation_path.exists():
                atomic_csv(pd.DataFrame([selected]), transformer_validation_path)
            retrain_info = {"epochs_ran": selected_epochs, "recovered_from_checkpoint": True}
            print(f"Split {split_id}: restored selected Transformer checkpoint", flush=True)
        else:
            existing_candidates: dict[int, dict[str, Any]] = {}
            if args.resume and transformer_validation_path.exists():
                existing_frame = pd.read_csv(transformer_validation_path)
                for row in existing_frame.to_dict(orient="records"):
                    candidate_id = int(row["candidate_id"])
                    if candidate_id >= 0:
                        existing_candidates[candidate_id] = row
            best = None
            print(f"Split {split_id}: selecting among {len(candidate_grid)} candidates", flush=True)
            for candidate_id, (encoder_layers, dropout, lambda_q) in enumerate(candidate_grid):
                config = ProbeConfig(
                    projected_dim=projected_dim,
                    n_layers=n_layers,
                    n_input_tokens=n_input_tokens,
                    token_dim=args.token_dim,
                    d_model=args.d_model,
                    encoder_layers=encoder_layers,
                    n_heads=args.n_heads,
                    dim_feedforward=args.dim_feedforward,
                    dropout=dropout,
                    use_deltas=True,
                )
                if candidate_id in existing_candidates:
                    row = existing_candidates[candidate_id]
                else:
                    model = build_probe(config)
                    model.set_scalar_normalization(scalar_mean, scalar_std)
                    weights = decision_weights(table, inner_train)
                    model, info = train_torch_model(
                        model,
                        ActivationDataset(table, inner_train, weights),
                        ActivationDataset(table, inner_val),
                        lambda_q=lambda_q,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        grad_clip=args.grad_clip,
                        device=device,
                        seed=args.seed + 1000 * split_id + candidate_id,
                        progress_label=(
                            f"split {split_id} candidate {candidate_id + 1}/{len(candidate_grid)}"
                        ),
                        show_progress=not args.no_progress,
                    )
                    row = {
                        "split_id": split_id,
                        "candidate_id": candidate_id,
                        "candidate_type": "transformer",
                        "encoder_layers": encoder_layers,
                        "dropout": dropout,
                        "lambda_q": lambda_q,
                        "validation_spearman": info["best_val_spearman"],
                        "validation_mae": info["best_val_mae"],
                        "best_epoch": info["best_epoch"],
                    }
                    existing_candidates[candidate_id] = row
                    ordered = [existing_candidates[key] for key in sorted(existing_candidates)]
                    atomic_csv(pd.DataFrame(ordered), transformer_validation_path)
                score = float(row["validation_spearman"])
                mae = float(row["validation_mae"])
                key = (-(score if np.isfinite(score) else -float("inf")), mae)
                if best is None or key < best["key"]:
                    best = {"key": key, "row": row, "config": config}
            if best is None:
                raise RuntimeError(f"no valid candidate selected for split {split_id}")
            selected = best["row"]
            config = best["config"]
            selected_epochs = max(1, int(selected["best_epoch"]) + 1)
            full_scalar_mean, full_scalar_std = fit_scalar_normalization(table, outer_train)
            atomic_json(
                selected_path,
                {
                    "selection": selected,
                    "config": config.__dict__,
                    "selected_epochs": selected_epochs,
                    "scalar_mean": full_scalar_mean.tolist(),
                    "scalar_std": full_scalar_std.tolist(),
                    "recovered_from_checkpoint": False,
                },
            )
            main_model = build_probe(config)
            main_model.set_scalar_normalization(full_scalar_mean, full_scalar_std)
            main_model, retrain_info = train_torch_model(
                main_model,
                ActivationDataset(table, outer_train, decision_weights(table, outer_train)),
                None,
                lambda_q=float(selected["lambda_q"]),
                max_epochs=selected_epochs,
                patience=selected_epochs + 1,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                grad_clip=args.grad_clip,
                device=device,
                seed=args.seed + 10000 + split_id,
                progress_label=f"split {split_id} selected-model retrain",
                show_progress=not args.no_progress,
            )
            checkpoint = {
                "config": config.__dict__,
                "state_dict": {key: value.detach().cpu() for key, value in main_model.state_dict().items()},
                "scalar_mean": full_scalar_mean,
                "scalar_std": full_scalar_std,
                "lambda_q": float(selected["lambda_q"]),
                "selected_epochs": selected_epochs,
                "selection": selected,
                "retrain_info": retrain_info,
                "split_id": split_id,
            }
            atomic_torch_save(checkpoint, main_checkpoint_path)

        if transformer_validation_path.exists():
            validation_rows.extend(pd.read_csv(transformer_validation_path).to_dict(orient="records"))
        selected_configs.append(
            {
                **selected,
                "selected_epochs": selected_epochs,
                "scalar_mean": full_scalar_mean.tolist(),
                "scalar_std": full_scalar_std.tolist(),
                "retrain_epochs": retrain_info["epochs_ran"],
            }
        )
        main_frame = load_or_predict_frame(
            path=fold_dir / "predictions_bidirectional_transformer.parquet",
            model=main_model,
            table=table,
            test_dataset=test_dataset,
            method="bidirectional_transformer",
            split_id=split_id,
            batch_size=args.batch_size,
            device=device,
        )
        prediction_frames.append(main_frame)
        occlusion_path = fold_dir / "occlusions.csv"
        occlusion_done_path = fold_dir / "occlusions_complete.json"
        if args.resume and occlusion_done_path.exists():
            fold_occlusions = pd.read_csv(occlusion_path).to_dict(orient="records") if occlusion_path.exists() else []
        else:
            fold_occlusions = run_occlusions(main_model, table, outer_test, config, args, device, split_id)
            if fold_occlusions:
                atomic_csv(pd.DataFrame(fold_occlusions), occlusion_path)
            atomic_json(occlusion_done_path, {"complete": True, "rows": len(fold_occlusions)})
        occlusion_rows.extend(fold_occlusions)

        if args.baselines == "all":
            baseline_specs: list[tuple[str, Callable[[], Any]]] = [
                ("final_layer_mlp", lambda cfg=config: build_mlp_baseline(cfg, mode="final")),
                ("mean_layers_mlp", lambda cfg=config: build_mlp_baseline(cfg, mode="mean")),
                ("concat_layers_mlp", lambda cfg=config: build_mlp_baseline(cfg, mode="concat")),
            ]
            no_delta_config = ProbeConfig(**{**config.__dict__, "use_deltas": False})
            baseline_specs.append(("transformer_no_deltas", lambda cfg=no_delta_config: build_probe(cfg)))
            for baseline_id, (method, builder) in enumerate(baseline_specs):
                baseline_checkpoint_path = model_dir / f"{method}_split{split_id}.pt"
                if args.resume and baseline_checkpoint_path.exists():
                    baseline_model, baseline_checkpoint, _ = load_checkpoint_model(
                        baseline_checkpoint_path, method, device
                    )
                    baseline_info = baseline_checkpoint.get("training", {"recovered_from_checkpoint": True})
                    print(f"Split {split_id}: restored {method} checkpoint", flush=True)
                else:
                    baseline_model, baseline_info = fit_selected_torch_model(
                        builder=builder,
                        table=table,
                        inner_train=inner_train,
                        inner_val=inner_val,
                        outer_train=outer_train,
                        inner_scalar_mean=scalar_mean,
                        inner_scalar_std=scalar_std,
                        full_scalar_mean=full_scalar_mean,
                        full_scalar_std=full_scalar_std,
                        lambda_q=float(selected["lambda_q"]),
                        args=args,
                        device=device,
                        seed=args.seed + 20000 + 100 * split_id + baseline_id,
                        max_epochs=args.baseline_max_epochs,
                    )
                    actual_config = no_delta_config if method == "transformer_no_deltas" else config
                    atomic_torch_save(
                        {
                            "config": actual_config.__dict__,
                            "state_dict": {
                                key: value.detach().cpu() for key, value in baseline_model.state_dict().items()
                            },
                            "training": baseline_info,
                            "method": method,
                            "split_id": split_id,
                        },
                        baseline_checkpoint_path,
                    )
                frame = load_or_predict_frame(
                    path=fold_dir / f"predictions_{method}.parquet",
                    model=baseline_model,
                    table=table,
                    test_dataset=test_dataset,
                    method=method,
                    split_id=split_id,
                    batch_size=args.batch_size,
                    device=device,
                )
                prediction_frames.append(frame)

            single_validation_path = fold_dir / "single_layer_validation.csv"
            single_checkpoint_path = model_dir / f"best_single_layer_mlp_split{split_id}.pt"
            existing_single: dict[int, dict[str, Any]] = {}
            if args.resume and single_validation_path.exists():
                for row in pd.read_csv(single_validation_path).to_dict(orient="records"):
                    existing_single[int(row["physical_layer"])] = row
            layers_to_fit = [] if args.resume and single_checkpoint_path.exists() else range(
                0, n_layers, max(1, int(args.single_layer_stride))
            )
            for layer in layers_to_fit:
                if layer not in existing_single:
                    probe = build_mlp_baseline(config, mode="single", layer_index=layer)
                    probe.set_scalar_normalization(scalar_mean, scalar_std)
                    probe, info = train_torch_model(
                        probe,
                        ActivationDataset(table, inner_train, decision_weights(table, inner_train)),
                        ActivationDataset(table, inner_val),
                        lambda_q=float(selected["lambda_q"]),
                        max_epochs=args.baseline_max_epochs,
                        patience=args.patience,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        grad_clip=args.grad_clip,
                        device=device,
                        seed=args.seed + 30000 + 100 * split_id + layer,
                        progress_label=f"split {split_id} single-layer L{layer}",
                        show_progress=not args.no_progress,
                    )
                    existing_single[layer] = {
                        "split_id": split_id,
                        "candidate_type": "single_layer",
                        "physical_layer": layer,
                        "validation_spearman": info["best_val_spearman"],
                        "validation_mae": info["best_val_mae"],
                        "best_epoch": info["best_epoch"],
                    }
                    atomic_csv(
                        pd.DataFrame([existing_single[key] for key in sorted(existing_single)]),
                        single_validation_path,
                    )
            single_rows = [existing_single[key] for key in sorted(existing_single)]
            validation_rows.extend(single_rows)
            best_single = None
            for row in single_rows:
                score = float(row["validation_spearman"])
                mae = float(row["validation_mae"])
                key = (-(score if np.isfinite(score) else -float("inf")), mae)
                if best_single is None or key < best_single["key"]:
                    best_single = {
                        "key": key,
                        "layer": int(row["physical_layer"]),
                        "epochs": max(1, int(row["best_epoch"]) + 1),
                    }
            if args.resume and single_checkpoint_path.exists():
                single_model, single_checkpoint, _ = load_checkpoint_model(
                    single_checkpoint_path, "best_single_layer_mlp", device
                )
                best_single = {
                    "layer": int(single_checkpoint["selected_physical_layer"]),
                    "epochs": int(single_checkpoint["selected_epochs"]),
                }
            elif best_single is not None:
                single_model = build_mlp_baseline(
                    config, mode="single", layer_index=int(best_single["layer"])
                )
                single_model.set_scalar_normalization(full_scalar_mean, full_scalar_std)
                single_model, _ = train_torch_model(
                    single_model,
                    ActivationDataset(table, outer_train, decision_weights(table, outer_train)),
                    None,
                    lambda_q=float(selected["lambda_q"]),
                    max_epochs=int(best_single["epochs"]),
                    patience=int(best_single["epochs"]) + 1,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    grad_clip=args.grad_clip,
                    device=device,
                    seed=args.seed + 40000 + split_id,
                    progress_label=f"split {split_id} best-single-layer retrain",
                    show_progress=not args.no_progress,
                )
                atomic_torch_save(
                    {
                        "config": config.__dict__,
                        "state_dict": {
                            key: value.detach().cpu() for key, value in single_model.state_dict().items()
                        },
                        "method": "best_single_layer_mlp",
                        "selected_physical_layer": int(best_single["layer"]),
                        "selected_epochs": int(best_single["epochs"]),
                        "split_id": split_id,
                    },
                    single_checkpoint_path,
                )
            if best_single is not None:
                frame = load_or_predict_frame(
                    path=fold_dir / "predictions_best_single_layer_mlp.parquet",
                    model=single_model,
                    table=table,
                    test_dataset=test_dataset,
                    method="best_single_layer_mlp",
                    split_id=split_id,
                    batch_size=args.batch_size,
                    device=device,
                )
                frame["selected_physical_layer"] = int(best_single["layer"])
                atomic_parquet(frame, fold_dir / "predictions_best_single_layer_mlp.parquet")
                prediction_frames.append(frame)

            for baseline_name, baseline_spec in POSITIONAL_BASELINES.items():
                baseline_path = fold_dir / f"predictions_{baseline_name}.parquet"
                if args.resume and baseline_path.exists():
                    baseline_frame = pd.read_parquet(baseline_path)
                else:
                    baseline_prediction = fit_positional_baseline(
                        table, outer_train, outer_test, baseline_spec["features"]
                    )
                    baseline_frame = predictions_to_frame(
                        table, baseline_prediction, baseline_name, split_id
                    )
                    atomic_parquet(baseline_frame, baseline_path)
                prediction_frames.append(baseline_frame)
        atomic_json(
            fold_dir / "fold_complete.json",
            {
                "complete": True,
                "split_id": split_id,
                "baselines": args.baselines,
                "prediction_methods": sorted(
                    frame["method"].iloc[0]
                    for frame in prediction_frames
                    if int(frame["split_id"].iloc[0]) == split_id
                ),
            },
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_columns = ["pred_commitment", *[f"pred_q_{action}" for action in ACTIONS]]
    if not np.isfinite(predictions[prediction_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("non-finite out-of-fold predictions detected")
    q_sums = predictions[[f"pred_q_{action}" for action in ACTIONS]].sum(axis=1).to_numpy()
    if not np.allclose(q_sums, 1.0, atol=1e-5):
        raise RuntimeError("predicted action distributions are not normalized")
    main_predictions = predictions[predictions["method"].eq("bidirectional_transformer")]
    expected_main = pd.concat(
        [table.iloc[split["test_idx"]][["prefix_id"]] for split in outer_splits], ignore_index=True
    )
    if (
        main_predictions["prefix_id"].duplicated().any()
        or len(main_predictions) != len(expected_main)
        or set(main_predictions["prefix_id"].astype(str)) != set(expected_main["prefix_id"].astype(str))
    ):
        raise RuntimeError("main model did not produce exactly one out-of-fold prediction per expected prefix")
    expected_prefixes = set(expected_main["prefix_id"].astype(str))
    for method, frame in predictions.groupby("method", sort=True):
        if (
            frame["prefix_id"].duplicated().any()
            or len(frame) != len(expected_main)
            or set(frame["prefix_id"].astype(str)) != expected_prefixes
        ):
            raise RuntimeError(f"{method} did not produce exactly one out-of-fold prediction per expected prefix")
    metric_rows: list[dict[str, Any]] = []
    for (method, split_id), frame in predictions.groupby(["method", "split_id"], sort=True):
        metric_rows.extend(evaluate_prediction_frame(frame, method=str(method), split_id=int(split_id)))
    for method, frame in predictions.groupby("method", sort=True):
        metric_rows.extend(evaluate_prediction_frame(frame, method=str(method), split_id=-1))
    metrics = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(validation_rows)
    occlusions = pd.DataFrame(occlusion_rows)
    bootstrap = clustered_bootstrap(
        predictions,
        main_method="bidirectional_transformer",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    atomic_parquet(predictions, output_dir / "oof_predictions.parquet")
    atomic_csv(metrics, metrics_path)
    atomic_csv(validation, output_dir / "validation_candidates.csv")
    if not occlusions.empty:
        atomic_csv(occlusions, output_dir / "occlusion_metrics.csv")
    if not bootstrap.empty:
        atomic_csv(bootstrap, output_dir / "clustered_bootstrap.csv")
    atomic_json(output_dir / "selected_configs.json", selected_configs)
    manifest = {
        "status": "complete",
        "device": str(device),
        "rows": int(len(table)),
        "outer_folds": int(len(outer_splits)),
        "model_selection_metric": "validation_spearman_then_mae",
        "attention": "bidirectional_unmasked",
        "probe_input_kind": "cot_last_tokens",
        "probe_input_shape": [int(n_layers), int(n_input_tokens), int(projected_dim)],
        "forced_suffix_states_used_as_probe_input": False,
        "forced_suffix_role": "label computation only",
        "token_padding_masked": True,
        "training_progress_bars": not args.no_progress,
        "candidate_grid": [
            {"encoder_layers": layers, "dropout": dropout, "lambda_q": lambda_q}
            for layers, dropout, lambda_q in candidate_grid
        ],
        "activation_index": str(args.activation_index.resolve()),
        "beliefs": str(args.beliefs.resolve()),
        "splits": str(args.splits.resolve()),
        "seed": int(args.seed),
    }
    atomic_json(output_dir / "training_manifest.json", manifest)
    write_report(output_dir, metrics, selected_configs, bootstrap)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
