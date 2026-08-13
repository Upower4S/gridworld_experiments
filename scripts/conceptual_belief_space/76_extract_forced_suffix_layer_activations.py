#!/usr/bin/env python3
"""Extract all-layer states for the last genuine CoT-token window.

Run from the repository root with the MLX project environment, for example:

  uv run --project interp --extra mlx python -u \
    scripts/conceptual_belief_space/76_extract_forced_suffix_layer_activations.py

The 12-token forced-final suffix is used only after the input window to compute
the four-action belief label.  Neither suffix nor action-candidate states are
saved as probe inputs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    ANALYSIS_HEADER,
    DEFAULT_COT_WINDOW_TOKENS,
    FORCED_SUFFIX,
    ShardSpec,
    TrajectoryShardWriter,
    array_sha256,
    aligned_cot_window,
    clone_cache,
    encode_without_special,
    forced_suffix_token_ids,
    index_rows_from_shards,
    load_shard_metadata,
    longest_common_prefix_len,
    make_orthogonal_projection,
    shard_slug,
    stable_action_probs,
    trim_cache,
    validate_context_suffix,
)


Q_COLS = [f"q_{action}" for action in ACTIONS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", type=Path, default=Path("outputs/prefixes.parquet"))
    parser.add_argument(
        "--beliefs",
        type=Path,
        default=Path("outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations"),
    )
    parser.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    parser.add_argument("--projected_dim", type=int, default=512)
    parser.add_argument("--projection_seed", type=int, default=42)
    parser.add_argument("--cot_window_tokens", type=int, default=DEFAULT_COT_WINDOW_TOKENS)
    parser.add_argument("--expected_layers", type=int, default=24)
    parser.add_argument("--expected_hidden_dim", type=int, default=2880)
    parser.add_argument("--q_tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--q_mismatch_policy",
        choices=["error", "use_extracted"],
        default="error",
        help=(
            "Default to failing when saved labels do not match this model/runtime. "
            "use_extracted writes a revision-aligned belief table and records the mismatch."
        ),
    )
    parser.add_argument("--verify_standard_rows", type=int, default=2)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--limit_prefixes", type=int, default=None)
    parser.add_argument("--max_prefixes_per_trajectory", type=int, default=None)
    parser.add_argument("--force_trajectory", action="append", default=[])
    return parser.parse_args()


def require_dependencies():
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import make_prompt_cache
        from huggingface_hub import snapshot_download
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency {exc.name!r}. Run with `uv run --project interp --extra mlx python ...`."
        ) from exc
    return pd, mx, load, create_attention_mask, make_prompt_cache, snapshot_download


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    df.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_or_create_projection(path: Path, hidden_dim: int, projected_dim: int, seed: int) -> np.ndarray:
    if path.exists():
        projection = np.load(path)
        if projection.shape != (hidden_dim, projected_dim):
            raise ValueError(f"existing projection shape {projection.shape} is incompatible")
        return np.asarray(projection, dtype=np.float32)
    projection = make_orthogonal_projection(hidden_dim, projected_dim, seed)
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, projection)
    os.replace(temporary, path)
    return projection


def feed_ids(model: Any, mx: Any, cache: list[Any], token_ids: list[int]):
    if not token_ids:
        return None
    logits = model(mx.array([token_ids], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    return logits


def forward_tokens_with_states(
    *,
    model: Any,
    mx: Any,
    create_attention_mask: Any,
    cache: list[Any],
    token_ids: list[int],
    capture_start: int = 0,
    capture_count: int,
    projection_mx: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run cached tokens and capture states for their causal leading segment."""
    if not token_ids:
        raise ValueError("manual state capture requires at least one token")
    capture_start = int(capture_start)
    capture_count = int(capture_count)
    if capture_start < 0 or capture_count <= 0 or capture_start + capture_count > len(token_ids):
        raise ValueError("capture_start/capture_count must select a non-empty token segment")
    transformer = model.model
    layers = list(transformer.layers)
    layer_types = list(transformer.layer_types)
    x = transformer.embed_tokens(mx.array([token_ids], dtype=mx.int32))
    full_mask = create_attention_mask(x, cache[transformer.ga_idx])
    sliding_mask = create_attention_mask(
        x,
        cache[transformer.swa_idx],
        window_size=transformer.window_size,
    )
    projected_layers = []
    rms_layers = []
    for layer, layer_cache, layer_type in zip(layers, cache, layer_types, strict=True):
        mask = full_mask if layer_type == "full_attention" else sliding_mask
        x = layer(x, mask, layer_cache)
        state = x.astype(mx.float32)
        captured = state[0, capture_start : capture_start + capture_count]
        projected_layers.append(captured @ projection_mx)
        rms_layers.append(mx.sqrt(mx.mean(mx.square(captured), axis=-1)))
    normalized = transformer.norm(x)
    logits = model.lm_head(normalized).astype(mx.float32)
    projected = mx.stack(projected_layers, axis=0)
    rms = mx.stack(rms_layers, axis=0)
    mx.eval(projected, rms, logits)
    return (
        np.asarray(projected, dtype=np.float32),
        np.asarray(rms, dtype=np.float32),
        np.asarray(logits[0, -1], dtype=np.float32),
    )


def pad_cot_window(
    projected: np.ndarray,
    rms: np.ndarray,
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad a variable causal CoT window without creating fake token states."""
    n_layers, n_valid, projected_dim = projected.shape
    if n_valid > int(window_size):
        raise ValueError("captured CoT window is wider than its configured storage window")
    states = np.zeros((n_layers, int(window_size), projected_dim), dtype=np.float32)
    scales = np.ones((n_layers, int(window_size)), dtype=np.float32)
    states[:, -n_valid:] = projected
    scales[:, -n_valid:] = rms
    return states, scales


def action_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for action in ACTIONS:
        encoded = encode_without_special(tokenizer, action)
        if len(encoded) != 1:
            raise ValueError(f"action candidate {action} encoded to {len(encoded)} tokens, expected one")
        ids.append(encoded[0])
    return ids


def existing_shard_matches(meta_path: Path, trajectory_rows, spec: ShardSpec) -> bool:
    if not meta_path.exists():
        return False
    metadata = load_shard_metadata(meta_path)
    existing_prefixes = [int(row["prefix_id"]) for row in metadata["rows"]]
    intended_prefixes = [int(value) for value in trajectory_rows["prefix_id"].tolist()]
    stored_spec = metadata.get("spec", {})
    compatible = (
        existing_prefixes == intended_prefixes
        and int(stored_spec.get("n_layers", -1)) == spec.n_layers
        and int(stored_spec.get("n_input_tokens", -1)) == spec.n_input_tokens
        and int(stored_spec.get("projected_dim", -1)) == spec.projected_dim
        and stored_spec.get("projection_sha256") == spec.projection_sha256
        and stored_spec.get("input_kind") == spec.input_kind
        and [int(x) for x in stored_spec.get("label_suffix_token_ids", [])]
        == list(spec.label_suffix_token_ids)
        and stored_spec.get("model_revision") == spec.model_revision
        and stored_spec.get("mlx_lm_version") == spec.mlx_lm_version
    )
    if not compatible:
        raise ValueError(
            f"completed shard {meta_path} does not match current rows/configuration; "
            "use a new output directory or explicitly force this trajectory"
        )
    return True


def main() -> int:
    args = parse_args()
    pd, mx, load, create_attention_mask, make_prompt_cache, snapshot_download = require_dependencies()
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    prefixes = pd.read_parquet(args.prefixes)
    required = [
        "prefix_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "prompt_text",
        "analysis_prefix_text",
        "analysis_prefix_token_ids_json",
        "decision_id",
    ]
    if "decision_id" not in prefixes.columns and {"trajectory_id", "step_id"} <= set(prefixes.columns):
        prefixes = prefixes.copy()
        prefixes["decision_id"] = prefixes["trajectory_id"].astype(str) + "::" + prefixes["step_id"].astype(str)
    missing = [column for column in required if column not in prefixes.columns]
    if missing:
        raise ValueError(f"prefix table is missing columns: {missing}")
    beliefs_full = pd.read_parquet(args.beliefs)
    beliefs = beliefs_full[["prefix_id", *Q_COLS]]
    prefixes = prefixes.merge(beliefs, on="prefix_id", how="inner", validate="one_to_one")
    prefixes = prefixes.sort_values(["trajectory_id", "step_id", "prefix_token_index", "prefix_id"])
    if args.limit_prefixes is not None:
        prefixes = prefixes.head(int(args.limit_prefixes))
    trajectory_ids = prefixes["trajectory_id"].drop_duplicates().astype(str).tolist()
    if args.max_trajectories is not None:
        trajectory_ids = trajectory_ids[: int(args.max_trajectories)]
        prefixes = prefixes[prefixes["trajectory_id"].astype(str).isin(set(trajectory_ids))]
    if args.max_prefixes_per_trajectory is not None:
        prefixes = (
            prefixes.groupby("trajectory_id", sort=False, group_keys=False)
            .head(int(args.max_prefixes_per_trajectory))
            .reset_index(drop=True)
        )
    if prefixes.empty:
        raise ValueError("no aligned prefix/belief rows selected")

    print(f"Loading {args.model_name_or_path}", flush=True)
    model, tokenizer = load(args.model_name_or_path)
    try:
        resolved_snapshot = Path(
            snapshot_download(args.model_name_or_path, local_files_only=True)
        ).resolve()
        model_revision = resolved_snapshot.name if resolved_snapshot.parent.name == "snapshots" else None
    except Exception:
        resolved_snapshot = None
        model_revision = None
    transformer = model.model
    n_layers = len(transformer.layers)
    hidden_dim = int(model.args.hidden_size)
    if n_layers != int(args.expected_layers):
        raise ValueError(f"model has {n_layers} layers, expected {args.expected_layers}")
    if hidden_dim != int(args.expected_hidden_dim):
        raise ValueError(f"model hidden size is {hidden_dim}, expected {args.expected_hidden_dim}")
    suffix_ids = forced_suffix_token_ids(tokenizer)
    candidate_ids = action_token_ids(tokenizer)
    if int(args.cot_window_tokens) <= 0:
        raise ValueError("--cot_window_tokens must be positive")

    projection_path = output_dir / "orthogonal_projection.npy"
    projection = load_or_create_projection(
        projection_path,
        hidden_dim=hidden_dim,
        projected_dim=int(args.projected_dim),
        seed=int(args.projection_seed),
    )
    projection_hash = array_sha256(projection)
    projection_mx = mx.array(projection, dtype=mx.float32)
    spec = ShardSpec(
        n_layers=n_layers,
        n_input_tokens=int(args.cot_window_tokens),
        projected_dim=int(args.projected_dim),
        hidden_dim=hidden_dim,
        projection_seed=int(args.projection_seed),
        projection_sha256=projection_hash,
        input_kind="cot_last_tokens",
        label_suffix_token_ids=tuple(suffix_ids),
        model_name_or_path=str(args.model_name_or_path),
        model_revision=model_revision,
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
    )

    force = set(str(x) for x in args.force_trajectory)
    total_new = 0
    q_errors: list[float] = []
    standard_errors: list[float] = []
    padded_rows = 0
    start_time = time.perf_counter()
    for trajectory_id, trajectory_rows in prefixes.groupby("trajectory_id", sort=False):
        trajectory_id = str(trajectory_id)
        trajectory_rows = trajectory_rows.sort_values(["step_id", "prefix_token_index", "prefix_id"])
        meta_path = shard_dir / f"{shard_slug(trajectory_id)}.json"
        if trajectory_id not in force and existing_shard_matches(meta_path, trajectory_rows, spec):
            print(f"Skipping validated shard {trajectory_id}", flush=True)
            continue
        writer = TrajectoryShardWriter(shard_dir, trajectory_id, spec)
        print(f"Extracting {trajectory_id}: {len(trajectory_rows)} prefixes", flush=True)
        try:
            for (_trajectory, step_id), step_rows in trajectory_rows.groupby(["trajectory_id", "step_id"], sort=False):
                records = step_rows.to_dict("records")
                base_text = str(records[0]["prompt_text"]).rstrip() + ANALYSIS_HEADER
                prefix_cache = make_prompt_cache(model)
                cached_prefix_ids: list[int] = []
                cached_analysis_ids: list[int] = []
                rolling_projected: np.ndarray | None = None
                rolling_rms: np.ndarray | None = None
                for record in records:
                    analysis_text = "" if record.get("analysis_prefix_text") is None else str(record["analysis_prefix_text"])
                    full_ids = encode_without_special(tokenizer, base_text + analysis_text + FORCED_SUFFIX)
                    target_prefix_ids = validate_context_suffix(full_ids, suffix_ids)
                    try:
                        analysis_ids_payload = json.loads(str(record["analysis_prefix_token_ids_json"]))
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} has malformed analysis token IDs"
                        ) from exc
                    if not isinstance(analysis_ids_payload, list):
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} analysis token IDs must be a list"
                        )
                    analysis_ids = [int(token_id) for token_id in analysis_ids_payload]
                    expected_analysis_count = int(record.get("prefix_token_index", -1)) + 1
                    if len(analysis_ids) != expected_analysis_count:
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} stores {len(analysis_ids)} analysis tokens, "
                            f"expected {expected_analysis_count} from prefix_token_index"
                        )
                    window_ids, padded_window_ids, input_token_mask = aligned_cot_window(
                        target_prefix_ids,
                        analysis_ids,
                        window_size=int(args.cot_window_tokens),
                    )
                    padded_rows += int(len(window_ids) < int(args.cot_window_tokens))
                    analysis_lcp = longest_common_prefix_len(cached_analysis_ids, analysis_ids)
                    if analysis_lcp != len(cached_analysis_ids):
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} is not a monotonic extension of the "
                            "preceding CoT prefix"
                        )
                    new_analysis_ids = analysis_ids[analysis_lcp:]
                    if not new_analysis_ids:
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} duplicates the preceding CoT prefix"
                        )
                    prefix_lcp = longest_common_prefix_len(cached_prefix_ids, target_prefix_ids)
                    if prefix_lcp != len(cached_prefix_ids):
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} changes already-cached scoring-context "
                            "tokens; monotonic exact tokenization is required"
                        )
                    new_context_ids = target_prefix_ids[prefix_lcp:]
                    if (
                        len(new_context_ids) < len(new_analysis_ids)
                        or new_context_ids[-len(new_analysis_ids) :] != new_analysis_ids
                    ):
                        raise ValueError(
                            f"prefix_id={record['prefix_id']} new scoring-context tokens do not end "
                            "with the expected new CoT tokens"
                        )
                    verification_cache = (
                        (
                            clone_cache(prefix_cache)
                            if cached_prefix_ids
                            else make_prompt_cache(model)
                        )
                        if len(standard_errors) < int(args.verify_standard_rows)
                        else None
                    )
                    new_projected, new_rms, captured_prefix_logits = forward_tokens_with_states(
                        model=model,
                        mx=mx,
                        create_attention_mask=create_attention_mask,
                        cache=prefix_cache,
                        token_ids=new_context_ids,
                        capture_start=len(new_context_ids) - len(new_analysis_ids),
                        capture_count=len(new_analysis_ids),
                        projection_mx=projection_mx,
                    )
                    cached_prefix_ids = target_prefix_ids
                    cached_analysis_ids = analysis_ids
                    if rolling_projected is None:
                        rolling_projected = new_projected[:, -int(args.cot_window_tokens) :]
                        rolling_rms = new_rms[:, -int(args.cot_window_tokens) :]
                    else:
                        rolling_projected = np.concatenate(
                            [rolling_projected, new_projected], axis=1
                        )[:, -int(args.cot_window_tokens) :]
                        rolling_rms = np.concatenate([rolling_rms, new_rms], axis=1)[
                            :, -int(args.cot_window_tokens) :
                        ]
                    if rolling_projected.shape[1] != len(window_ids):
                        raise RuntimeError("rolling CoT states do not align with the selected token window")
                    projected, rms = pad_cot_window(
                        rolling_projected,
                        rolling_rms,
                        window_size=int(args.cot_window_tokens),
                    )
                    if verification_cache is not None:
                        standard_prefix_logits = feed_ids(
                            model, mx, verification_cache, new_context_ids
                        )
                        selected_standard_mx = standard_prefix_logits[
                            0, -1, candidate_ids
                        ].astype(mx.float32)
                        mx.eval(selected_standard_mx)
                        selected_standard = np.asarray(selected_standard_mx, dtype=np.float32)
                        selected_manual = captured_prefix_logits[candidate_ids]
                        standard_error = float(np.max(np.abs(selected_standard - selected_manual)))
                        standard_errors.append(standard_error)
                        if standard_error > 1e-4:
                            raise RuntimeError(
                                "manual CoT-window forward disagrees with the standard model "
                                f"forward by {standard_error}"
                            )

                    label_cache = clone_cache(prefix_cache)
                    label_logits_mx = feed_ids(model, mx, label_cache, suffix_ids)
                    selected_label_mx = label_logits_mx[0, -1, candidate_ids].astype(mx.float32)
                    mx.eval(selected_label_mx)
                    selected_label_logits = np.asarray(selected_label_mx, dtype=np.float32)

                    extracted_q = stable_action_probs(selected_label_logits)
                    expected_q = np.asarray([record[column] for column in Q_COLS], dtype=np.float64)
                    expected_q /= expected_q.sum()
                    q_error = float(np.max(np.abs(extracted_q - expected_q)))
                    q_errors.append(q_error)
                    if q_error > float(args.q_tolerance) and args.q_mismatch_policy == "error":
                        raise RuntimeError(
                            f"prefix_id={record['prefix_id']} action probabilities disagree by {q_error}, "
                            f"above tolerance {args.q_tolerance}"
                        )
                    row_metadata = {
                        "prefix_id": int(record["prefix_id"]),
                        "trajectory_id": trajectory_id,
                        "decision_id": str(record["decision_id"]),
                        "step_id": int(step_id),
                        "prefix_token_index": int(record.get("prefix_token_index", -1)),
                        "input_kind": "cot_last_tokens",
                        "input_token_ids_json": json.dumps(padded_window_ids),
                        "input_token_mask_json": json.dumps(input_token_mask),
                        "n_valid_input_tokens": int(sum(input_token_mask)),
                        "max_q_error": q_error,
                        **{f"extracted_q_{action}": float(extracted_q[i]) for i, action in enumerate(ACTIONS)},
                    }
                    writer.append(projected, rms, row_metadata)
                    total_new += 1
            writer.finalize()
        except Exception:
            writer.abort()
            raise

    index_rows = index_rows_from_shards(shard_dir)
    index = pd.DataFrame(index_rows).sort_values("prefix_id").reset_index(drop=True)
    if index["prefix_id"].duplicated().any():
        raise RuntimeError("completed activation shards contain duplicate prefix_id values")
    atomic_parquet(index, output_dir / "activation_index.parquet")
    extracted_q_cols = [f"extracted_q_{action}" for action in ACTIONS]
    extracted_labels = beliefs_full[beliefs_full["prefix_id"].isin(set(index["prefix_id"]))].copy()
    extracted_labels = extracted_labels.merge(
        index[["prefix_id", *extracted_q_cols]], on="prefix_id", how="inner", validate="one_to_one"
    )
    for action in ACTIONS:
        extracted_labels[f"q_{action}"] = extracted_labels.pop(f"extracted_q_{action}")
    atomic_parquet(extracted_labels, output_dir / "extracted_beliefs.parquet")
    all_q_errors = index["max_q_error"].to_numpy(dtype=float)
    max_q_error = float(np.max(all_q_errors)) if len(all_q_errors) else float("nan")
    mismatched_rows = int(np.sum(all_q_errors > float(args.q_tolerance)))
    manifest = {
        "status": "complete",
        "model_name_or_path": args.model_name_or_path,
        "resolved_model_snapshot": str(resolved_snapshot) if resolved_snapshot is not None else None,
        "model_revision": model_revision,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": importlib.metadata.version("mlx"),
        "n_rows": int(len(index)),
        "n_trajectories": int(index["trajectory_id"].nunique()),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "projected_dim": int(args.projected_dim),
        "projection_seed": int(args.projection_seed),
        "projection_sha256": projection_hash,
        "projection_path": str(projection_path.resolve()),
        "probe_input_kind": "cot_last_tokens",
        "probe_input_shape": ["rows", n_layers, int(args.cot_window_tokens), int(args.projected_dim)],
        "cot_window_tokens": int(args.cot_window_tokens),
        "cot_window_alignment": "right_aligned; zero state and unit RMS padding masked by input_token_mask",
        "cot_state_chunking": (
            "canonical monotonic text-tokenized prefix cache; newly appended CoT positions are "
            "captured once and retained in the rolling window"
        ),
        "rows_with_left_padding": int(padded_rows),
        "prompt_tokens_in_activation_input": False,
        "forced_suffix_tokens_in_activation_input": False,
        "candidate_tokens_in_activation_input": False,
        "label_source": "four candidate logits after the forced final suffix",
        "state_capture_and_label_use_separate_forwards": True,
        "state_capture_forward_contains_label_suffix": False,
        "label_forced_suffix": FORCED_SUFFIX,
        "label_suffix_token_ids": suffix_ids,
        "label_suffix_token_count": len(suffix_ids),
        "candidate_action_token_ids": dict(zip(ACTIONS, candidate_ids, strict=True)),
        "max_q_error": max_q_error,
        "q_tolerance": float(args.q_tolerance),
        "q_mismatch_policy": args.q_mismatch_policy,
        "q_mismatched_rows": mismatched_rows,
        "saved_beliefs_revision_aligned": mismatched_rows == 0,
        "extracted_beliefs_path": str((output_dir / "extracted_beliefs.parquet").resolve()),
        "max_manual_vs_standard_logit_error": float(max(standard_errors)) if standard_errors else None,
        "new_rows_this_run": int(total_new),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "index_path": str((output_dir / "activation_index.parquet").resolve()),
    }
    atomic_json(output_dir / "extraction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
