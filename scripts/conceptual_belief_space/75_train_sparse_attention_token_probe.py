#!/usr/bin/env python3
"""Train a linear probe on native-attention-selected prefix tokens.

The frozen model's final-prefix-token attention row is used only as a sparse,
label-free selector.  The primary probe reads two fixed vectors:

    [current-token residual ; native-attention-weighted top-k prior residuals]

The selector has no trainable parameters.  The only trained component is one
affine four-action readout.  Matched controls replace native selection with
recent, random, or all-prefix pooling while keeping the same readout class.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED


RANK_PATH = Path(__file__).with_name("39_train_activation_rank_probes.py")
rank_spec = importlib.util.spec_from_file_location("sparse_attention_rank_helpers", RANK_PATH)
if rank_spec is None or rank_spec.loader is None:
    raise RuntimeError(f"Could not import rank-probe helpers from {RANK_PATH}")
rank_impl = importlib.util.module_from_spec(rank_spec)
sys.modules[rank_spec.name] = rank_impl
rank_spec.loader.exec_module(rank_impl)

LINEAR_PATH = Path(__file__).with_name("41_train_activation_window_top_action_linear.py")
linear_spec = importlib.util.spec_from_file_location("sparse_attention_linear_helpers", LINEAR_PATH)
if linear_spec is None or linear_spec.loader is None:
    raise RuntimeError(f"Could not import linear-probe helpers from {LINEAR_PATH}")
linear_impl = importlib.util.module_from_spec(linear_spec)
sys.modules[linear_spec.name] = linear_impl
linear_spec.loader.exec_module(linear_impl)


Q_COLS = [f"q_{action}" for action in ACTIONS]
ATTENTION_MODES = {"attention_weighted", "attention_uniform"}
CONTROL_MODES = {"recent_uniform", "random_uniform", "prefix_mean"}
ALL_MEMORY_MODES = ATTENTION_MODES | CONTROL_MODES


def parse_csv_ints(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class ConcatenatedFeatureView:
    """Lazy feature-axis concatenation over row-aligned arrays."""

    def __init__(self, *arrays: Any):
        if not arrays:
            raise ValueError("At least one feature array is required")
        row_counts = {int(array.shape[0]) for array in arrays}
        if len(row_counts) != 1:
            raise ValueError(f"Feature arrays have mismatched row counts: {row_counts}")
        self.arrays = arrays
        self.shape = (next(iter(row_counts)), sum(int(array.shape[1]) for array in arrays))

    def __getitem__(self, index: Any) -> np.ndarray:
        return np.concatenate(
            [np.asarray(array[index], dtype=np.float32) for array in self.arrays],
            axis=-1,
        )


def selector_path(attention_dir: Path, attention_layer: int) -> Path:
    return attention_dir / f"sparse_native_attention_layer_{attention_layer:02d}.npz"


def load_selector(
    attention_dir: Path,
    attention_layer: int,
    beliefs: pd.DataFrame,
) -> dict[str, np.ndarray]:
    path = selector_path(attention_dir, attention_layer)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        selector = {key: np.asarray(data[key]) for key in data.files}
    expected_rows = np.arange(len(beliefs), dtype=np.int64)
    if not np.array_equal(selector["row_position"].astype(np.int64), expected_rows):
        raise RuntimeError(f"{path} is not complete and row-aligned with the belief table")
    if "prefix_id" in beliefs.columns:
        observed = selector["prefix_id"].astype(np.int64)
        expected = beliefs["prefix_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Prefix IDs in {path} do not align with the belief table")
    if not np.array_equal(
        selector["prefix_token_index"].astype(np.int64),
        beliefs["prefix_token_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(f"Prefix token indices in {path} do not align with the belief table")
    return selector


def create_memmap(path: Path, rows: int, hidden_dim: int, dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=(rows, hidden_dim),
    )


def attention_pool(
    cot: np.ndarray,
    selected_indices: np.ndarray,
    selected_weights: np.ndarray,
    *,
    uniform: bool,
) -> np.ndarray:
    valid = selected_indices >= 0
    safe_indices = np.where(valid, selected_indices, 0)
    gathered = cot[safe_indices]
    if uniform:
        weights = valid.astype(np.float32)
    else:
        weights = np.where(valid, selected_weights, 0.0).astype(np.float32)
    normalizer = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        normalizer,
        out=np.zeros_like(weights),
        where=normalizer > 1e-30,
    )
    return np.einsum("bkd,bk->bd", gathered, weights, optimize=True).astype(np.float32)


def recent_pool(cot: np.ndarray, token_indices: np.ndarray, top_k: int) -> np.ndarray:
    output = np.zeros((len(token_indices), cot.shape[1]), dtype=np.float32)
    for row, token_index in enumerate(token_indices):
        end = int(token_index)
        if end <= 0:
            continue
        start = max(0, end - int(top_k))
        output[row] = cot[start:end].mean(axis=0, dtype=np.float32)
    return output


def random_pool(
    cot: np.ndarray,
    token_indices: np.ndarray,
    prefix_ids: np.ndarray,
    top_k: int,
    seed: int,
) -> np.ndarray:
    output = np.zeros((len(token_indices), cot.shape[1]), dtype=np.float32)
    for row, (token_index, prefix_id) in enumerate(zip(token_indices, prefix_ids)):
        count = int(token_index)
        if count <= 0:
            continue
        keep = min(int(top_k), count)
        rng = np.random.default_rng(int(seed) + 1_000_003 * int(prefix_id))
        chosen = rng.choice(count, size=keep, replace=False)
        output[row] = cot[chosen].mean(axis=0, dtype=np.float32)
    return output


def prefix_mean_pool(cot: np.ndarray, token_indices: np.ndarray) -> np.ndarray:
    output = np.zeros((len(token_indices), cot.shape[1]), dtype=np.float32)
    positive = token_indices > 0
    if not positive.any():
        return output
    cumulative = np.cumsum(cot, axis=0, dtype=np.float32)
    for row, token_index in enumerate(token_indices):
        count = int(token_index)
        if count > 0:
            output[row] = cumulative[count - 1] / float(count)
    return output


def build_feature_files(
    beliefs: pd.DataFrame,
    pointers: dict[str, Any],
    selector: dict[str, np.ndarray],
    *,
    activation_layer: int,
    top_ks: list[int],
    memory_mode: str,
    memory_paths: dict[int, Path],
    last_path: Path | None,
    dtype: np.dtype,
    seed: int,
) -> None:
    if memory_mode not in ALL_MEMORY_MODES:
        raise ValueError(f"Unsupported memory mode: {memory_mode}")
    hidden_dim = next(iter(pointers.values())).hidden_dim
    outputs = {
        top_k: create_memmap(path, len(beliefs), hidden_dim, dtype)
        for top_k, path in memory_paths.items()
    }
    last_output = create_memmap(last_path, len(beliefs), hidden_dim, dtype) if last_path else None
    loader = rank_impl.BF16MatrixCache(max_open=64)
    local = beliefs.copy()
    local["_feature_row"] = np.arange(len(local), dtype=np.int64)
    for decision_number, (decision_id, group) in enumerate(local.groupby("decision_id", sort=False), start=1):
        pointer = pointers[str(decision_id)]
        cot_path = rank_impl.path_for_layer(pointer.cot_file, activation_layer)
        cot_mmap = loader.get(cot_path, pointer.hidden_dim)
        cot = rank_impl.bf16_uint16_to_float32(np.asarray(cot_mmap)).astype(np.float32, copy=False)
        rows = group["_feature_row"].to_numpy(dtype=np.int64)
        token_indices = group["prefix_token_index"].to_numpy(dtype=np.int64)
        if np.any(token_indices < 0) or np.any(token_indices >= len(cot)):
            raise IndexError(
                f"Decision {decision_id} has prefix token indices outside [0, {len(cot)})"
            )
        if last_output is not None:
            last_output[rows] = cot[token_indices].astype(dtype, copy=False)

        if memory_mode in ATTENTION_MODES:
            max_top_k = max(top_ks)
            selected_indices = selector["selected_indices"][rows, :max_top_k].astype(np.int64)
            valid = selected_indices >= 0
            safe_indices = np.where(valid, selected_indices, 0)
            gathered = cot[safe_indices]
            if memory_mode == "attention_uniform":
                base_weights = valid.astype(np.float32)
            else:
                selected_weights = selector["selected_weights"][rows, :max_top_k].astype(np.float32)
                base_weights = np.where(valid, selected_weights, 0.0)
            for top_k in top_ks:
                weights = base_weights[:, :top_k]
                normalizer = weights.sum(axis=1, keepdims=True)
                weights = np.divide(
                    weights,
                    normalizer,
                    out=np.zeros_like(weights),
                    where=normalizer > 1e-30,
                )
                pooled = np.einsum(
                    "bkd,bk->bd",
                    gathered[:, :top_k],
                    weights,
                    optimize=True,
                ).astype(np.float32)
                outputs[top_k][rows] = pooled.astype(dtype, copy=False)
        else:
            for top_k in top_ks:
                if memory_mode == "recent_uniform":
                    pooled = recent_pool(cot, token_indices, top_k)
                elif memory_mode == "random_uniform":
                    pooled = random_pool(
                        cot,
                        token_indices,
                        group["prefix_id"].to_numpy(dtype=np.int64),
                        top_k,
                        seed,
                    )
                elif memory_mode == "prefix_mean":
                    pooled = prefix_mean_pool(cot, token_indices)
                else:
                    raise AssertionError(memory_mode)
                outputs[top_k][rows] = pooled.astype(dtype, copy=False)
        if decision_number % 100 == 0:
            print(
                f"  built {memory_mode} layer={activation_layer}: "
                f"{decision_number}/{beliefs['decision_id'].nunique()} decisions",
                flush=True,
            )
    if last_output is not None:
        last_output.flush()
    for output in outputs.values():
        output.flush()


def train_one_split(
    x: Any,
    beliefs: pd.DataFrame,
    y: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    import torch

    split_id = int(split["split_id"])
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    train_groups = set(beliefs.iloc[train_idx]["trajectory_id"].astype(str))
    test_groups = set(beliefs.iloc[test_idx]["trajectory_id"].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Trajectory leakage in split {split_id}: {sorted(overlap)[:5]}")
    subtrain_idx, val_idx = linear_impl.train_val_split_by_group(
        beliefs,
        train_idx,
        args.seed + split_id,
        args.val_fraction,
    )
    model_seed = args.seed + seed_offset + split_id
    torch.manual_seed(model_seed)
    logits, info = linear_impl.fit_linear_classifier(
        x,
        y,
        subtrain_idx,
        val_idx,
        test_idx,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=model_seed,
    )
    return logits, info, test_idx


def development_candidate(
    x: Any,
    beliefs: pd.DataFrame,
    y: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    *,
    attention_layer: int,
    top_k: int,
) -> dict[str, Any]:
    logits, info, test_idx = train_one_split(
        x,
        beliefs,
        y,
        split,
        args,
        seed_offset=10_000,
    )
    y_test = y[test_idx]
    predicted = logits.argmax(axis=1)
    return {
        "attention_layer": int(attention_layer),
        "activation_layer": int(attention_layer - 1),
        "top_k": int(top_k),
        "feature_form": "last_plus_attention_weighted",
        "feature_dim": int(x.shape[1]),
        "parameter_count": int(x.shape[1] * len(ACTIONS) + len(ACTIONS)),
        "selection_metric": "train-only grouped validation balanced accuracy",
        "validation_balanced_accuracy": float(info["validation_balanced_accuracy"]),
        "development_test_balanced_accuracy": linear_impl.balanced_accuracy(y_test, predicted),
        "development_test_accuracy": float((y_test == predicted).mean()),
        "best_epoch": int(info["best_epoch"]),
        "device": str(info["device"]),
    }


def evaluate_configuration(
    name: str,
    x: Any,
    beliefs: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    attention_layer: int,
    top_k: int,
    selector_kind: str,
    token_count_read: str,
    metric_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
) -> None:
    config = {
        "probe_name": name,
        "attention_layer": int(attention_layer),
        "activation_layer": int(attention_layer - 1),
        "top_k": int(top_k),
        "selector_kind": selector_kind,
        "token_count_read": token_count_read,
        "feature_dim": int(x.shape[1]),
        "parameter_count": int(x.shape[1] * len(ACTIONS) + len(ACTIONS)),
        "linear_model": "one affine four-action readout",
    }
    for split in splits:
        split_id = int(split["split_id"])
        print(f"  training {name} split={split_id}", flush=True)
        logits, info, test_idx = train_one_split(
            x,
            beliefs,
            y,
            split,
            args,
            seed_offset=20_000,
        )
        linear_impl.add_metrics(
            metric_rows,
            beliefs,
            test_idx,
            logits,
            split_id,
            config,
            info,
        )
        prediction_frames.append(
            linear_impl.make_predictions(beliefs, test_idx, logits, split_id, config)
        )


def selector_statistics(selector: dict[str, np.ndarray], top_k: int) -> dict[str, Any]:
    indices = selector["selected_indices"][:, :top_k]
    weights = selector["selected_weights"][:, :top_k]
    valid = indices >= 0
    selected_mass = np.where(valid, weights, 0.0).sum(axis=1)
    analysis_mass = selector["analysis_mass"].astype(np.float64)
    fraction = np.divide(
        selected_mass,
        analysis_mass,
        out=np.zeros_like(selected_mass, dtype=np.float64),
        where=analysis_mass > 1e-30,
    )
    token_index = selector["prefix_token_index"].astype(np.int64)
    ages = np.where(valid, token_index[:, None] - indices, np.nan)
    finite_fraction = fraction[np.isfinite(fraction) & (selector["available_keys"] > 0)]
    finite_ages = ages[np.isfinite(ages)]
    return {
        "rows": int(len(indices)),
        "rows_with_prior_keys": int((selector["available_keys"] > 0).sum()),
        "mean_selected_token_count": float(valid.sum(axis=1).mean()),
        "median_selected_fraction_of_analysis_attention_mass": float(np.median(finite_fraction)),
        "mean_selected_fraction_of_analysis_attention_mass": float(np.mean(finite_fraction)),
        "median_selected_token_age": float(np.median(finite_ages)),
        "mean_selected_token_age": float(np.mean(finite_ages)),
        "median_analysis_attention_mass": float(
            np.median(analysis_mass[selector["available_keys"] > 0])
        ),
        "median_normalized_analysis_attention_entropy": float(
            np.nanmedian(selector["normalized_entropy"])
        ),
    }


def write_token_examples(
    path: Path,
    prefixes_path: Path,
    selector: dict[str, np.ndarray],
    top_k: int,
    model_name_or_path: str,
    example_count: int,
) -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(f"Could not load tokenizer: {exc!r}\n")
        return
    prefixes = rank_impl.read_parquet_robust(
        prefixes_path,
        columns=["prefix_id", "analysis_prefix_token_ids_json"],
    ).reset_index(drop=True)
    candidates = np.flatnonzero(selector["prefix_token_index"] > 0)
    if len(candidates) > example_count:
        positions = np.linspace(0, len(candidates) - 1, example_count).round().astype(int)
        candidates = candidates[positions]
    rows: list[dict[str, Any]] = []
    for row_position in candidates:
        token_ids = json.loads(str(prefixes.iloc[int(row_position)]["analysis_prefix_token_ids_json"]))
        query_index = int(selector["prefix_token_index"][row_position])
        selected = selector["selected_indices"][row_position, :top_k]
        weights = selector["selected_weights"][row_position, :top_k]
        valid = selected >= 0
        selected = selected[valid].astype(int)
        weights = weights[valid].astype(float)
        denominator = max(float(weights.sum()), 1e-30)
        rows.append(
            {
                "row_position": int(row_position),
                "prefix_id": int(prefixes.iloc[int(row_position)]["prefix_id"]),
                "query_token_index": query_index,
                "query_token_text": tokenizer.decode([int(token_ids[query_index])]),
                "selected_token_indices_json": json.dumps(selected.tolist()),
                "selected_token_ages_json": json.dumps((query_index - selected).tolist()),
                "selected_token_text_json": json.dumps(
                    [tokenizer.decode([int(token_ids[index])]) for index in selected]
                ),
                "normalized_selected_weights_json": json.dumps((weights / denominator).tolist()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beliefs",
        default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet",
    )
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    parser.add_argument(
        "--attention_dir",
        default="outputs/conceptual_belief_space/sparse_native_attention_selectors",
    )
    parser.add_argument("--attention_layers", default="11,15,19,23")
    parser.add_argument("--top_k_values", default="2,4,8,16")
    parser.add_argument("--selection_split", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    parser.add_argument("--example_count", type=int, default=24)
    parser.add_argument("--temp_dir", default="/private/tmp")
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space/sparse_attention_token_probe",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = Path(args.attention_dir)
    attention_layers = parse_csv_ints(args.attention_layers)
    top_k_values = parse_csv_ints(args.top_k_values)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    beliefs = rank_impl.prepare_belief_rows(
        Path(args.beliefs),
        limit_decisions=None,
        seed=args.seed,
        prefix_path=Path(args.prefixes),
    ).reset_index(drop=True)
    if not np.array_equal(
        beliefs["prefix_id"].to_numpy(dtype=np.int64),
        np.arange(len(beliefs), dtype=np.int64),
    ):
        raise RuntimeError("This strict run requires belief rows to remain aligned to prefix_id")
    activation_layers = [layer - 1 for layer in attention_layers]
    if any(layer < 0 or layer % 2 for layer in activation_layers):
        raise ValueError(
            "Each attention layer must have an available even residual-input layer L-1"
        )
    pointers = rank_impl.build_activation_pointers(
        Path(args.activation_index),
        set(beliefs["decision_id"].astype(str)),
        layers=activation_layers,
    )
    splits = rank_impl.make_group_splits(beliefs, args.n_splits, args.seed)
    split_by_id = {int(split["split_id"]): split for split in splits}
    if args.selection_split not in split_by_id:
        raise ValueError(f"Selection split {args.selection_split} is unavailable")
    y = linear_impl.action_labels_to_indices(beliefs["top_action"])

    candidate_rows: list[dict[str, Any]] = []
    print("Development selection: attention layer and top-k", flush=True)
    for attention_layer in attention_layers:
        activation_layer = attention_layer - 1
        selector = load_selector(attention_dir, attention_layer, beliefs)
        if max(top_k_values) > selector["selected_indices"].shape[1]:
            raise ValueError(
                f"Requested top-k {max(top_k_values)}, but layer {attention_layer} stores only "
                f"{selector['selected_indices'].shape[1]} keys"
            )
        with tempfile.TemporaryDirectory(
            prefix=f"sparse_attn_layer_{attention_layer}_",
            dir=args.temp_dir,
        ) as temp_name:
            temp = Path(temp_name)
            last_path = temp / "last.npy"
            memory_paths = {top_k: temp / f"weighted_k{top_k}.npy" for top_k in top_k_values}
            print(f"Building development features for attention layer {attention_layer}", flush=True)
            build_feature_files(
                beliefs,
                pointers,
                selector,
                activation_layer=activation_layer,
                top_ks=top_k_values,
                memory_mode="attention_weighted",
                memory_paths=memory_paths,
                last_path=last_path,
                dtype=np.dtype(dtype),
                seed=args.seed,
            )
            last = np.load(last_path, mmap_mode="r")
            for top_k in top_k_values:
                memory = np.load(memory_paths[top_k], mmap_mode="r")
                view = ConcatenatedFeatureView(last, memory)
                row = development_candidate(
                    view,
                    beliefs,
                    y,
                    split_by_id[args.selection_split],
                    args,
                    attention_layer=attention_layer,
                    top_k=top_k,
                )
                candidate_rows.append(row)
                print(
                    f"  layer={attention_layer} k={top_k} "
                    f"val_BA={row['validation_balanced_accuracy']:.4f} "
                    f"dev_test_BA={row['development_test_balanced_accuracy']:.4f}",
                    flush=True,
                )
                del memory, view
                gc.collect()
            del last
            gc.collect()
        pd.DataFrame(candidate_rows).to_csv(
            output_dir / "sparse_attention_development_candidates.partial.csv",
            index=False,
        )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["validation_balanced_accuracy", "attention_layer", "top_k"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    candidates.to_csv(output_dir / "sparse_attention_development_candidates.csv", index=False)
    selected = candidates.iloc[0]
    selected_attention_layer = int(selected["attention_layer"])
    selected_activation_layer = selected_attention_layer - 1
    selected_top_k = int(selected["top_k"])
    selector = load_selector(attention_dir, selected_attention_layer, beliefs)
    print(
        f"Selected attention layer {selected_attention_layer}, top-k {selected_top_k} "
        f"from train-only grouped validation",
        flush=True,
    )

    last_path = feature_dir / f"last_token_activation_layer_{selected_activation_layer:02d}.npy"
    weighted_path = feature_dir / (
        f"attention_weighted_memory_attn_layer_{selected_attention_layer:02d}_k{selected_top_k}.npy"
    )
    build_feature_files(
        beliefs,
        pointers,
        selector,
        activation_layer=selected_activation_layer,
        top_ks=[selected_top_k],
        memory_mode="attention_weighted",
        memory_paths={selected_top_k: weighted_path},
        last_path=last_path,
        dtype=np.dtype(dtype),
        seed=args.seed,
    )
    last = np.load(last_path, mmap_mode="r")
    weighted = np.load(weighted_path, mmap_mode="r")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    print("Final grouped evaluation: persistent primary features", flush=True)
    evaluate_configuration(
        "last_only",
        last,
        beliefs,
        y,
        splits,
        args,
        attention_layer=selected_attention_layer,
        top_k=0,
        selector_kind="none",
        token_count_read="1 current token",
        metric_rows=metric_rows,
        prediction_frames=prediction_frames,
    )
    evaluate_configuration(
        "attention_memory_only",
        weighted,
        beliefs,
        y,
        splits,
        args,
        attention_layer=selected_attention_layer,
        top_k=selected_top_k,
        selector_kind="native_head_mean_topk",
        token_count_read=f"at most {selected_top_k} prior tokens",
        metric_rows=metric_rows,
        prediction_frames=prediction_frames,
    )
    primary_view = ConcatenatedFeatureView(last, weighted)
    evaluate_configuration(
        "last_plus_attention_weighted",
        primary_view,
        beliefs,
        y,
        splits,
        args,
        attention_layer=selected_attention_layer,
        top_k=selected_top_k,
        selector_kind="native_head_mean_topk_with_native_weights",
        token_count_read=f"1 current + at most {selected_top_k} prior tokens",
        metric_rows=metric_rows,
        prediction_frames=prediction_frames,
    )

    control_specs = [
        ("last_plus_attention_uniform", "attention_uniform", "native_head_mean_topk_with_uniform_pool"),
        ("last_plus_recent_uniform", "recent_uniform", "matched_k_most_recent"),
        ("last_plus_random_uniform", "random_uniform", "matched_k_deterministic_random"),
        ("last_plus_prefix_mean", "prefix_mean", "all_prior_tokens_blind_mean"),
    ]
    print("Final grouped evaluation: matched selection controls", flush=True)
    with tempfile.TemporaryDirectory(prefix="sparse_attn_controls_", dir=args.temp_dir) as temp_name:
        temp = Path(temp_name)
        for probe_name, memory_mode, selector_kind in control_specs:
            control_path = temp / "control_memory.npy"
            build_feature_files(
                beliefs,
                pointers,
                selector,
                activation_layer=selected_activation_layer,
                top_ks=[selected_top_k],
                memory_mode=memory_mode,
                memory_paths={selected_top_k: control_path},
                last_path=None,
                dtype=np.dtype(dtype),
                seed=args.seed,
            )
            control = np.load(control_path, mmap_mode="r")
            control_view = ConcatenatedFeatureView(last, control)
            evaluate_configuration(
                probe_name,
                control_view,
                beliefs,
                y,
                splits,
                args,
                attention_layer=selected_attention_layer,
                top_k=selected_top_k if memory_mode != "prefix_mean" else -1,
                selector_kind=selector_kind,
                token_count_read=(
                    "1 current + all prior tokens (diagnostic only)"
                    if memory_mode == "prefix_mean"
                    else f"1 current + at most {selected_top_k} prior tokens"
                ),
                metric_rows=metric_rows,
                prediction_frames=prediction_frames,
            )
            del control, control_view
            gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "sparse_attention_probe_metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "sparse_attention_probe_predictions.parquet", index=False)

    all_prefix = metrics[metrics["eval_subset"].eq("all_prefixes")].copy()
    summary = (
        all_prefix.groupby(
            [
                "probe_name",
                "attention_layer",
                "activation_layer",
                "top_k",
                "selector_kind",
                "token_count_read",
                "feature_dim",
                "parameter_count",
            ],
            as_index=False,
        )[["accuracy", "balanced_accuracy", "macro_f1", "pairwise_ranking_accuracy"]]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    summary.to_csv(output_dir / "sparse_attention_probe_summary.csv", index=False)
    confirmatory = (
        all_prefix[all_prefix["split_id"].ne(args.selection_split)]
        .groupby("probe_name", as_index=False)[
            ["accuracy", "balanced_accuracy", "macro_f1", "pairwise_ranking_accuracy"]
        ]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    confirmatory.to_csv(
        output_dir / "sparse_attention_probe_confirmatory_summary.csv",
        index=False,
    )

    statistics = selector_statistics(selector, selected_top_k)
    write_json(output_dir / "sparse_attention_selector_statistics.json", statistics)
    write_token_examples(
        output_dir / "sparse_attention_selected_token_examples.csv",
        Path(args.prefixes),
        selector,
        selected_top_k,
        args.model_name_or_path,
        args.example_count,
    )
    manifest = {
        "definition": (
            "Frozen native last-position attention selects top-k earlier reasoning tokens; "
            "a single affine probe reads current residual plus the weighted selected-token residual mean."
        ),
        "target": "top_action = argmax(q_UP, q_DOWN, q_LEFT, q_RIGHT)",
        "selector_trainable_parameters": 0,
        "probe_trainable_parameters": int(2 * 2880 * len(ACTIONS) + len(ACTIONS)),
        "forced_answer_suffix_appended": False,
        "learned_prefix_attention": False,
        "primary_probe": "last_plus_attention_weighted",
        "selected_attention_layer": selected_attention_layer,
        "selected_activation_layer": selected_activation_layer,
        "selected_top_k": selected_top_k,
        "selection_policy": (
            f"attention layer and k selected by train-only grouped validation inside outer split "
            f"{args.selection_split}; all outer test folds remain evaluation-only"
        ),
        "split_policy": "five trajectory-grouped outer folds; train-only trajectory-group validation for early stopping",
        "excluded_probe_inputs": [
            "grid/task features",
            "progress or prefix position",
            "IDs",
            "agent action",
            "future tokens",
            "forced answer tokens",
            "unselected prior-token activations",
        ],
        "diagnostic_controls": [name for name, _, _ in control_specs] + ["last_only"],
        "interpretation_limit": (
            "Native attention is used as a frozen routing signal, not asserted to be a causal "
            "attribution of token importance."
        ),
        "rows": int(len(beliefs)),
        "decisions": int(beliefs["decision_id"].nunique()),
        "trajectories": int(beliefs["trajectory_id"].nunique()),
        "inputs": {
            "beliefs": args.beliefs,
            "prefixes": args.prefixes,
            "activation_index": args.activation_index,
            "attention_dir": args.attention_dir,
        },
        "selector_statistics": statistics,
    }
    write_json(output_dir / "sparse_attention_probe_manifest.json", manifest)

    primary_summary = summary[summary["probe_name"].eq("last_plus_attention_weighted")]
    primary_confirm = confirmatory[
        confirmatory["probe_name"].eq("last_plus_attention_weighted")
    ]
    lines = [
        "# Sparse Native-Attention Token Probe",
        "",
        "The frozen model's actual prefix-final token is the attention query. Native attention "
        f"selects at most {selected_top_k} earlier reasoning tokens; the probe never reads the other prior-token activations.",
        "",
        f"- selected native attention block: `{selected_attention_layer}` (full attention)",
        f"- matching residual-input layer: `{selected_activation_layer}`",
        f"- selected top-k: `{selected_top_k}`",
        f"- selector parameters: `0`",
        f"- primary linear readout parameters: `{manifest['probe_trainable_parameters']:,}`",
        "- forced-answer suffix: `none`",
        "- learned attention over the prefix: `none`",
        "- interpretation: native attention is a frozen routing signal, not a causal-attribution claim",
        "",
        "## Results",
        "",
        summary.to_markdown(index=False),
        "",
        "## Confirmatory folds",
        "",
        confirmatory.to_markdown(index=False),
        "",
        "## Selector audit",
        "",
        pd.DataFrame([statistics]).to_markdown(index=False),
        "",
        "`last_plus_prefix_mean` is an intentionally blind all-prefix diagnostic, not the proposed probe.",
    ]
    if not primary_summary.empty and not primary_confirm.empty:
        lines.insert(
            17,
            f"Primary mean balanced accuracy: `{float(primary_summary.iloc[0]['balanced_accuracy']):.6f}`; "
            f"confirmatory folds: `{float(primary_confirm.iloc[0]['balanced_accuracy']):.6f}`.",
        )
    (output_dir / "sparse_attention_probe_report.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
