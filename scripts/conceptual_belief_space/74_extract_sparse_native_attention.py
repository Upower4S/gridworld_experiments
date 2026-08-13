#!/usr/bin/env python3
"""Extract sparse native-attention selectors at prefix-final CoT tokens.

For each sampled reasoning prefix, this script uses the frozen GPT-OSS
attention row whose query is the prefix's final reasoning token.  It retains
only the top-k *earlier reasoning tokens* and their native attention weights.
No forced-answer suffix is appended, and no learned attention or label signal
is used during selection.

Only full-attention transformer blocks are accepted by default.  For an odd
GPT-OSS attention block L, the selected keys can later be paired exactly with
the cached residual activations after block L-1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTRACTOR_PATH = ROOT / "interp" / "evaluation_scripts" / "extract_key_door_activations_mlx.py"
spec = importlib.util.spec_from_file_location("keydoor_mlx_activation_extractor", EXTRACTOR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import MLX extraction helpers from {EXTRACTOR_PATH}")
mlx_extract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mlx_extract
spec.loader.exec_module(mlx_extract)


DEFAULT_TRAJECTORIES = "trajectories_key_door_100/trajectories_key_door/*.json"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_LAYERS = "11,15,19,23"


def read_parquet_robust(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path, columns=columns, engine="fastparquet")


def parse_csv_ints(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("Expected at least one attention layer")
    return values


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_step_input(trajectory: dict[str, Any], step: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the unmodified step sequence and absolute analysis-token positions."""
    if mlx_extract._has_stored_token_annotations(trajectory, step):
        prefix_ids = mlx_extract._token_ids(trajectory["prompt"]["prompt_prefix_tokens"])
        grid_ids = mlx_extract._token_ids(step["grid_state_tokens"])
        suffix_ids = mlx_extract._token_ids(mlx_extract._step_suffix_tokens(trajectory, step))
        output_tokens = step["output_tokens"]
        output_ids = mlx_extract._token_ids(output_tokens)
        analysis_offsets = mlx_extract._analysis_token_indices(output_tokens)
    else:
        prefix_ids, grid_ids, suffix_ids = mlx_extract._compact_prompt_token_segments(
            trajectory=trajectory,
            step=step,
            tokenizer=tokenizer,
        )
        output_ids, analysis_offsets = mlx_extract._compact_output_token_ids_and_cot_indices(
            step=step,
            tokenizer=tokenizer,
        )
    output_start = len(prefix_ids) + len(grid_ids) + len(suffix_ids)
    analysis_absolute = [output_start + int(offset) for offset in analysis_offsets]
    return prefix_ids + grid_ids + suffix_ids + output_ids, analysis_absolute


def validate_prefix_queries(
    step_prefixes: pd.DataFrame,
    input_ids: list[int],
    analysis_absolute: list[int],
) -> int:
    mismatches = 0
    for row in step_prefixes.itertuples(index=False):
        token_index = int(row.prefix_token_index)
        if token_index < 0 or token_index >= len(analysis_absolute):
            raise IndexError(
                f"Prefix {row.prefix_id} requests analysis token {token_index}, "
                f"but the step has {len(analysis_absolute)} analysis tokens"
            )
        raw = getattr(row, "analysis_prefix_token_ids_json", None)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        prefix_ids = json.loads(str(raw))
        if not prefix_ids:
            raise ValueError(f"Prefix {row.prefix_id} has no encoded analysis tokens")
        expected = int(input_ids[analysis_absolute[token_index]])
        if int(prefix_ids[-1]) != expected or len(prefix_ids) != token_index + 1:
            mismatches += 1
    return mismatches


def sparse_attention_from_layer_input(
    layer: Any,
    x: Any,
    mask: Any,
    *,
    query_absolute: list[int],
    query_analysis_indices: np.ndarray,
    analysis_absolute: list[int],
    max_k: int,
    query_chunk_size: int,
    exclude_query: bool,
    mx: Any,
) -> dict[str, np.ndarray]:
    """Compute head-mean native attention and retain only top-k analysis keys."""
    if not hasattr(layer, "self_attn") or not hasattr(layer, "input_layernorm"):
        raise TypeError("Selected layer does not expose GPT-OSS self-attention internals")
    attention = layer.self_attn
    normalized = layer.input_layernorm(x)
    batch_size, seq_len, _ = normalized.shape
    if batch_size != 1:
        raise ValueError(f"Expected batch size one, got {batch_size}")
    head_dim = attention.head_dim
    q = attention.q_proj(normalized).reshape(batch_size, seq_len, -1, head_dim).swapaxes(1, 2)
    k = attention.k_proj(normalized).reshape(batch_size, seq_len, -1, head_dim).swapaxes(1, 2)
    q = attention.rope(q)
    k = attention.rope(k)
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError(f"Query heads {q.shape[1]} are not divisible by KV heads {k.shape[1]}")
        k = mx.repeat(k, repeats=q.shape[1] // k.shape[1], axis=1)

    n_queries = len(query_absolute)
    selected_indices = np.full((n_queries, max_k), -1, dtype=np.int32)
    selected_weights = np.zeros((n_queries, max_k), dtype=np.float32)
    analysis_mass = np.zeros(n_queries, dtype=np.float32)
    selected_mass = np.zeros(n_queries, dtype=np.float32)
    selected_fraction = np.zeros(n_queries, dtype=np.float32)
    normalized_entropy = np.full(n_queries, np.nan, dtype=np.float32)
    weighted_age = np.full(n_queries, np.nan, dtype=np.float32)
    available_keys = np.zeros(n_queries, dtype=np.int32)

    transposed_k = k.swapaxes(-1, -2)
    for start in range(0, n_queries, query_chunk_size):
        stop = min(start + query_chunk_size, n_queries)
        chunk_queries = query_absolute[start:stop]
        q_chunk = q[:, :, chunk_queries, :]
        scores = (q_chunk * attention.sm_scale) @ transposed_k
        scores = mlx_extract._apply_selected_attention_mask(scores, mask, chunk_queries, mx)
        probabilities = mlx_extract._softmax_with_optional_sinks(scores, attention.sinks, mx)
        head_mean = probabilities.mean(axis=1)[0].astype(mx.float32)
        mx.eval(head_mean)
        head_mean_np = np.array(head_mean, copy=True)

        for local_row, global_row in enumerate(range(start, stop)):
            query_token = int(query_analysis_indices[global_row])
            key_count = query_token if exclude_query else query_token + 1
            available_keys[global_row] = key_count
            if key_count <= 0:
                continue
            key_absolute = np.asarray(analysis_absolute[:key_count], dtype=np.int64)
            weights = np.asarray(head_mean_np[local_row, key_absolute], dtype=np.float64)
            weights[~np.isfinite(weights)] = 0.0
            weights = np.maximum(weights, 0.0)
            total = float(weights.sum())
            analysis_mass[global_row] = total
            if total <= 1e-30:
                continue
            probabilities_over_analysis = weights / total
            if key_count > 1:
                entropy = -float(
                    np.sum(
                        probabilities_over_analysis
                        * np.log(np.clip(probabilities_over_analysis, 1e-30, None))
                    )
                )
                normalized_entropy[global_row] = entropy / math.log(key_count)
            else:
                normalized_entropy[global_row] = 0.0

            keep = min(max_k, key_count)
            if keep == key_count:
                top = np.argsort(-weights)
            else:
                top = np.argpartition(-weights, keep - 1)[:keep]
                top = top[np.argsort(-weights[top])]
            top_weights = weights[top]
            selected_indices[global_row, :keep] = top.astype(np.int32)
            selected_weights[global_row, :keep] = top_weights.astype(np.float32)
            chosen_mass = float(top_weights.sum())
            selected_mass[global_row] = chosen_mass
            selected_fraction[global_row] = chosen_mass / total
            if chosen_mass > 0:
                ages = query_token - top.astype(np.float64)
                weighted_age[global_row] = float(np.sum(ages * top_weights) / chosen_mass)

    return {
        "selected_indices": selected_indices,
        "selected_weights": selected_weights,
        "analysis_mass": analysis_mass,
        "selected_mass": selected_mass,
        "selected_fraction": selected_fraction,
        "normalized_entropy": normalized_entropy,
        "weighted_age": weighted_age,
        "available_keys": available_keys,
    }


def extract_step(
    model: Any,
    input_ids: list[int],
    analysis_absolute: list[int],
    step_prefixes: pd.DataFrame,
    attention_layers: set[int],
    *,
    max_k: int,
    query_chunk_size: int,
    exclude_query: bool,
    mx: Any,
    create_attention_mask: Any,
) -> dict[int, dict[str, np.ndarray]]:
    query_analysis_indices = step_prefixes["prefix_token_index"].to_numpy(dtype=np.int64)
    query_absolute = [analysis_absolute[int(index)] for index in query_analysis_indices]
    truncated_ids = input_ids[: max(query_absolute) + 1]
    input_array = mx.array([truncated_ids], dtype=mx.int32)
    transformer = mlx_extract._get_transformer(model)
    layers = list(transformer.layers)
    x = mlx_extract._embed_inputs(transformer, input_array)
    masks = mlx_extract._layer_mask_iter(transformer, x, len(layers), create_attention_mask)
    max_selected_layer = max(attention_layers)
    results: dict[int, dict[str, np.ndarray]] = {}
    for layer_index, layer in enumerate(layers):
        if layer_index in attention_layers:
            results[layer_index] = sparse_attention_from_layer_input(
                layer,
                x,
                masks[layer_index],
                query_absolute=query_absolute,
                query_analysis_indices=query_analysis_indices,
                analysis_absolute=analysis_absolute,
                max_k=max_k,
                query_chunk_size=query_chunk_size,
                exclude_query=exclude_query,
                mx=mx,
            )
            if layer_index == max_selected_layer:
                break
        x = mlx_extract._call_layer(layer, x, masks[layer_index], None)
        mx.eval(x)
    return results


def part_path(output_dir: Path, trajectory_id: str, layer: int) -> Path:
    return output_dir / "parts" / trajectory_id / f"attention_layer_{layer:02d}.npz"


def save_trajectory_parts(
    output_dir: Path,
    trajectory_id: str,
    layer_rows: dict[int, list[dict[str, np.ndarray]]],
) -> None:
    for layer, chunks in layer_rows.items():
        if not chunks:
            continue
        path = part_path(output_dir, trajectory_id, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = chunks[0].keys()
        payload = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
        np.savez(path, **payload)


def consolidate_parts(
    output_dir: Path,
    trajectory_ids: list[str],
    attention_layers: list[int],
    *,
    total_prefix_rows: int,
    require_full_coverage: bool,
) -> dict[int, int]:
    coverage: dict[int, int] = {}
    for layer in attention_layers:
        chunks: list[dict[str, np.ndarray]] = []
        for trajectory_id in trajectory_ids:
            path = part_path(output_dir, trajectory_id, layer)
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                chunks.append({key: np.asarray(data[key]) for key in data.files})
        if not chunks:
            raise RuntimeError(f"No extracted parts found for attention layer {layer}")
        keys = chunks[0].keys()
        merged = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
        order = np.argsort(merged["row_position"], kind="stable")
        merged = {key: value[order] for key, value in merged.items()}
        row_positions = merged["row_position"].astype(np.int64)
        if len(np.unique(row_positions)) != len(row_positions):
            raise RuntimeError(f"Duplicate row positions found while consolidating layer {layer}")
        if require_full_coverage:
            expected = np.arange(total_prefix_rows, dtype=np.int64)
            if not np.array_equal(row_positions, expected):
                missing = np.setdiff1d(expected, row_positions)
                raise RuntimeError(
                    f"Layer {layer} covers {len(row_positions)}/{total_prefix_rows} prefixes; "
                    f"first missing rows: {missing[:10].tolist()}"
                )
        np.savez(output_dir / f"sparse_native_attention_layer_{layer:02d}.npz", **merged)
        coverage[layer] = int(len(row_positions))
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--trajectory_patterns", nargs="+", default=[DEFAULT_TRAJECTORIES])
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--attention_layers", default=DEFAULT_LAYERS)
    parser.add_argument("--max_k", type=int, default=16)
    parser.add_argument("--query_chunk_size", type=int, default=4)
    parser.add_argument(
        "--include_query_key",
        action="store_true",
        help="Allow the final query token to select itself. The default selects earlier tokens only.",
    )
    parser.add_argument("--steps", default="all")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_sliding_layers", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space/sparse_native_attention_selectors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_k < 1:
        raise ValueError("--max_k must be positive")
    if args.query_chunk_size < 1:
        raise ValueError("--query_chunk_size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_layers = parse_csv_ints(args.attention_layers)

    prefix_columns = [
        "prefix_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "analysis_prefix_token_ids_json",
    ]
    prefixes = read_parquet_robust(Path(args.prefixes), columns=prefix_columns).reset_index(drop=True)
    prefixes.insert(0, "row_position", np.arange(len(prefixes), dtype=np.int64))
    prefixes["trajectory_id"] = prefixes["trajectory_id"].astype(str)
    prefixes["step_id"] = prefixes["step_id"].astype(int)

    trajectory_paths = mlx_extract._expand_trajectory_paths(
        list(args.trajectory_patterns),
        args.max_trajectories,
    )
    trajectory_ids = [path.stem for path in trajectory_paths]
    selected_prefixes = prefixes[prefixes["trajectory_id"].isin(trajectory_ids)]
    if selected_prefixes.empty:
        raise ValueError("No prefix rows matched the selected trajectories")

    expected_parts = [
        part_path(output_dir, trajectory_id, layer)
        for trajectory_id in trajectory_ids
        for layer in attention_layers
    ]
    all_parts_exist = all(path.exists() for path in expected_parts)
    model_load_seconds = 0.0
    config: dict[str, Any] = {}
    token_mismatches = 0
    processed_steps = 0
    run_start = time.perf_counter()

    if not all_parts_exist:
        mx, create_attention_mask, _ = mlx_extract._import_mlx_runtime()
        model, tokenizer, config, model_load_seconds = mlx_extract._load_mlx_model(
            args.model_name_or_path,
            args.trust_remote_code,
        )
        transformer = mlx_extract._get_transformer(model)
        if max(attention_layers) >= len(transformer.layers):
            raise ValueError(
                f"Requested layer {max(attention_layers)}, but model has {len(transformer.layers)} layers"
            )
        layer_types = list(getattr(transformer, "layer_types", ["unknown"] * len(transformer.layers)))
        sliding = [layer for layer in attention_layers if layer_types[layer] != "full_attention"]
        if sliding and not args.allow_sliding_layers:
            raise ValueError(
                f"Layers {sliding} are not full-attention blocks. Use odd GPT-OSS layers or "
                "pass --allow_sliding_layers explicitly."
            )

        for trajectory_number, trajectory_path in enumerate(trajectory_paths, start=1):
            trajectory_id = trajectory_path.stem
            paths = [part_path(output_dir, trajectory_id, layer) for layer in attention_layers]
            if all(path.exists() for path in paths):
                print(f"[{trajectory_number}/{len(trajectory_paths)}] Reusing {trajectory_id}", flush=True)
                continue
            trajectory = mlx_extract._load_json(trajectory_path)
            trajectory_prefixes = prefixes[prefixes["trajectory_id"].eq(trajectory_id)]
            selected_steps = set(mlx_extract._resolve_step_indices(args.steps, trajectory))
            layer_rows: dict[int, list[dict[str, np.ndarray]]] = defaultdict(list)
            print(
                f"[{trajectory_number}/{len(trajectory_paths)}] Extracting {trajectory_id} "
                f"({len(selected_steps)} steps)",
                flush=True,
            )
            for step_index in sorted(selected_steps):
                step = trajectory["steps"][step_index]
                step_id = int(step.get("step_id", step_index))
                step_prefixes = trajectory_prefixes[
                    trajectory_prefixes["step_id"].eq(step_id)
                ].sort_values("prefix_token_index")
                if step_prefixes.empty:
                    continue
                input_ids, analysis_absolute = build_step_input(trajectory, step, tokenizer)
                step_mismatches = validate_prefix_queries(step_prefixes, input_ids, analysis_absolute)
                token_mismatches += step_mismatches
                if step_mismatches:
                    raise RuntimeError(
                        f"Found {step_mismatches} prefix/token alignment mismatches in "
                        f"{trajectory_id} step {step_id}"
                    )
                results = extract_step(
                    model,
                    input_ids,
                    analysis_absolute,
                    step_prefixes,
                    set(attention_layers),
                    max_k=args.max_k,
                    query_chunk_size=args.query_chunk_size,
                    exclude_query=not args.include_query_key,
                    mx=mx,
                    create_attention_mask=create_attention_mask,
                )
                base = {
                    "row_position": step_prefixes["row_position"].to_numpy(dtype=np.int64),
                    "prefix_id": step_prefixes["prefix_id"].to_numpy(dtype=np.int64),
                    "step_id": step_prefixes["step_id"].to_numpy(dtype=np.int32),
                    "prefix_token_index": step_prefixes["prefix_token_index"].to_numpy(dtype=np.int32),
                }
                for layer, result in results.items():
                    layer_rows[layer].append({**base, **result})
                processed_steps += 1
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            save_trajectory_parts(output_dir, trajectory_id, layer_rows)

    coverage = consolidate_parts(
        output_dir,
        trajectory_ids,
        attention_layers,
        total_prefix_rows=len(prefixes),
        require_full_coverage=not args.allow_partial,
    )
    manifest = {
        "probe_role": "frozen sparse token selector; no labels or trainable selector parameters",
        "model_name_or_path": args.model_name_or_path,
        "model_config_model_type": config.get("model_type"),
        "attention_layers": attention_layers,
        "attention_layer_types": "full_attention required unless explicitly overridden",
        "head_reduction": "arithmetic mean over native query heads",
        "query": "actual final analysis token of each unmodified prefix",
        "key_domain": "earlier generated analysis tokens only"
        if not args.include_query_key
        else "generated analysis tokens through and including the query",
        "forced_answer_suffix_appended": False,
        "learned_attention": False,
        "max_k": int(args.max_k),
        "query_chunk_size": int(args.query_chunk_size),
        "prefix_rows_total": int(len(prefixes)),
        "prefix_rows_selected": int(len(selected_prefixes)),
        "trajectories_selected": int(len(trajectory_ids)),
        "processed_steps_this_run": int(processed_steps),
        "coverage_by_layer": {str(layer): count for layer, count in coverage.items()},
        "token_alignment_mismatches": int(token_mismatches),
        "model_load_seconds": float(model_load_seconds),
        "total_seconds_this_run": float(time.perf_counter() - run_start),
        "inputs": {
            "prefixes": args.prefixes,
            "trajectory_patterns": args.trajectory_patterns,
        },
    }
    write_json(output_dir / "sparse_native_attention_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
