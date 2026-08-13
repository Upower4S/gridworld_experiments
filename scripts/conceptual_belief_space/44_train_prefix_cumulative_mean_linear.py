#!/usr/bin/env python3
"""Activation-only linear probes on cumulative prefix-mean CoT activations.

For each prefix k, this builds h_mean_prefix = mean(h_1, ..., h_k) over all
generated reasoning-token activations available so far.  Prefix 0 uses the
pre_reasoning activation.  This tests whether long-range reasoning context that
is discarded by last-100 windows helps linear decoding of the forced q-top
action.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

RANK_SCRIPT = Path(__file__).with_name("39_train_activation_rank_probes.py")
spec = importlib.util.spec_from_file_location("activation_rank_probe_impl", RANK_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {RANK_SCRIPT}")
rank_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank_impl
spec.loader.exec_module(rank_impl)

MULTI_SCRIPT = Path(__file__).with_name("43_train_cached_multiscale_linear.py")
mspec = importlib.util.spec_from_file_location("cached_multiscale_impl", MULTI_SCRIPT)
if mspec is None or mspec.loader is None:
    raise RuntimeError(f"Could not import helpers from {MULTI_SCRIPT}")
ms_impl = importlib.util.module_from_spec(mspec)
sys.modules[mspec.name] = ms_impl
mspec.loader.exec_module(ms_impl)

Q_COLS = [f"q_{a}" for a in ACTIONS]


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_modes(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def terms_for_mode(mode: str) -> list[tuple[str, int]]:
    if mode == "raw_prefixmean":
        return [("raw", -1)]
    if mode == "resid_prefixmean":
        return [("resid", -1)]
    if mode == "raw_and_resid_prefixmean":
        return [("raw", -1), ("resid", -1)]
    raise ValueError(f"Unsupported mode: {mode}")


def build_prefix_mean_cache(
    beliefs: pd.DataFrame,
    pointers: dict[str, Any],
    *,
    layer: int,
    output_path: Path,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        arr = np.load(output_path, mmap_mode="r")
        if arr.shape[0] > int(beliefs["row_id"].max()) and arr.shape[1] == next(iter(pointers.values())).hidden_dim:
            return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = int(beliefs["row_id"].max()) + 1
    hidden = next(iter(pointers.values())).hidden_dim
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(n_rows, hidden))
    loader = rank_impl.BF16MatrixCache(max_open=64)
    df = beliefs.copy()
    df["_pos"] = np.arange(len(df), dtype=np.int64)
    for decision_id, group in df.groupby("decision_id", sort=False):
        pointer = pointers[str(decision_id)]
        pre = loader.load_row(rank_impl.path_for_layer(pointer.pre_file, layer), pointer.pre_row_index, pointer.hidden_dim)
        cot_path = rank_impl.path_for_layer(pointer.cot_file, layer)
        cot_mmap = loader.get(cot_path, pointer.hidden_dim)
        token_items = sorted(pointer.cot_rows_by_token.items())
        token_indices = np.asarray([int(t) for t, _r in token_items], dtype=np.int64)
        row_indices = np.asarray([int(r) for _t, r in token_items], dtype=np.int64)
        cot = rank_impl.bf16_uint16_to_float32(np.asarray(cot_mmap[row_indices])).astype(np.float32, copy=False)
        cumsum = np.cumsum(cot, axis=0, dtype=np.float64)
        for row in group.itertuples(index=False):
            row_id = int(getattr(row, "row_id"))
            k = int(getattr(row, "activation_window_end_token_index", getattr(row, "prefix_token_index")))
            if k <= 0 or len(cot) == 0:
                out[row_id] = pre.astype(np.float32, copy=False)
                continue
            # Token indices are 1-based. Use all available CoT tokens <= k.
            n = int(np.searchsorted(token_indices, k, side="right"))
            if n <= 0:
                out[row_id] = pre.astype(np.float32, copy=False)
            else:
                out[row_id] = (cumsum[n - 1] / float(n)).astype(np.float32, copy=False)
    out.flush()
    return output_path


def load_prefixmean_arrays(cache_dir: Path, layers: list[int]) -> dict[tuple[int, int], np.ndarray]:
    arrays = {}
    for layer in layers:
        path = cache_dir / f"features_layer_{layer:02d}_raw_prefixmean.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[(-1, int(layer))] = np.load(path, mmap_mode="r")
    return arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear")
    p.add_argument("--cache_dir", default=None, help="Optional existing prefixmean feature cache directory")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--modes", default="raw_prefixmean,resid_prefixmean,raw_and_resid_prefixmean")
    p.add_argument("--losses", default="ovr_bce")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=768)
    p.add_argument("--val_fraction", type=float, default=0.0)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    beliefs = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=None, seed=args.seed, prefix_path=Path(args.prefixes))
    layers = parse_layers(args.layers)
    pointers = rank_impl.build_activation_pointers(Path(args.activation_index), set(beliefs["decision_id"].astype(str)), layers=layers)
    manifest = {
        "activation_only": True,
        "feature": "cumulative mean of all CoT token activations up to prefix; prefix0 uses pre_reasoning",
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "decision_id_use": "index activation files, find prefix0 pre_reasoning rows, and grouped splits only",
        "layers": layers,
        "rows": int(len(beliefs)),
        "decisions": int(beliefs["decision_id"].nunique()),
        "trajectories": int(beliefs["trajectory_id"].nunique()),
    }
    (output_dir / "prefixmean_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for layer in layers:
        path = cache_dir / f"features_layer_{layer:02d}_raw_prefixmean.npy"
        print(f"Building/loading prefix cumulative mean layer={layer}", flush=True)
        build_prefix_mean_cache(beliefs, pointers, layer=layer, output_path=path, force=args.force_features)
    arrays = load_prefixmean_arrays(cache_dir, layers)
    splits = rank_impl.make_group_splits(beliefs, args.n_splits, args.seed)
    if args.split_ids != "all":
        want = {int(x) for x in args.split_ids.split(",") if x.strip()}
        splits = [s for s in splits if int(s["split_id"]) in want]
    y = ms_impl.action_indices(beliefs["top_action"])
    feature_rows = ms_impl.feature_rows_by_position(beliefs)
    pre_rows = ms_impl.pre_feature_rows_by_position(beliefs)
    metrics_rows = []
    pred_frames = []
    for mode in parse_modes(args.modes):
        terms = terms_for_mode(mode)
        print(f"Mode {mode} terms={terms} feature_dim={ms_impl.feature_dim(arrays, terms, layers)}", flush=True)
        for loss_kind in ms_impl.parse_losses(args.losses):
            for split in splits:
                split_id = int(split["split_id"])
                train_idx = split["train_idx"]
                test_idx = split["test_idx"]
                if set(beliefs.iloc[train_idx]["trajectory_id"].astype(str)) & set(beliefs.iloc[test_idx]["trajectory_id"].astype(str)):
                    raise RuntimeError(f"Trajectory leakage in split {split_id}")
                subtrain_idx, val_idx = ms_impl.train_val_split_by_group(beliefs, train_idx, args.seed + split_id, args.val_fraction)
                print(f"Training mode={mode} loss={loss_kind} split={split_id} train={len(subtrain_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
                logits, info = ms_impl.fit_model(
                    arrays,
                    terms,
                    layers,
                    y,
                    subtrain_idx,
                    val_idx,
                    test_idx,
                    pre_rows,
                    feature_rows,
                    loss_kind=loss_kind,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    seed=args.seed + split_id,
                )
                ms_impl.add_metric_rows(metrics_rows, beliefs, test_idx, logits, split_id=split_id, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers, info=info)
                if args.save_predictions:
                    pred_frames.append(ms_impl.make_predictions(beliefs, test_idx, logits, split_id=split_id, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers))
                pd.DataFrame(metrics_rows).to_csv(output_dir / "prefixmean_metrics.partial.csv", index=False)
                print(pd.DataFrame(metrics_rows).tail(3).to_string(index=False), flush=True)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "prefixmean_metrics.csv", index=False)
    if args.save_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "prefixmean_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby(["mode", "loss_kind"], as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "prefixmean_summary.csv", index=False)
    report = [
        "# Prefix Cumulative Mean Activation Linear Probe",
        "",
        "- Feature input: activation-only cumulative mean over all reasoning tokens up to each prefix.",
        "- Prefix 0 uses pre_reasoning.",
        "- No grid/task features or prefix metadata are model inputs.",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows.",
    ]
    (output_dir / "prefixmean_report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {output_dir / 'prefixmean_metrics.csv'}")
    print(f"Wrote {output_dir / 'prefixmean_summary.csv'}")
    print(summary.to_string(index=False) if not summary.empty else "No summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
