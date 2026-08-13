#!/usr/bin/env python3
"""Construct activation feature arrays from aligned activation/behavior rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{action}" for action in ACTIONS]
DEFAULT_REPRESENTATIONS = ["raw", "delta_pre", "velocity", "unit_delta_pre"]
SUPPORTED_REPRESENTATIONS = {
    "raw",
    "delta_pre",
    "post_delta_pre",
    "velocity",
    "unit_delta_pre",
    "progress_residualized_delta_pre",
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_layers(value: str, available_layers_json: pd.Series) -> list[int]:
    if value.lower() == "all":
        layers = set()
        for raw in available_layers_json.dropna().unique():
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                layers.update(int(x) for x in parsed)
        if not layers:
            raise ValueError("Could not infer available layers from aligned table")
        return sorted(layers)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_representations(value: str) -> list[str]:
    reps = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [rep for rep in reps if rep not in SUPPORTED_REPRESENTATIONS]
    if invalid:
        raise ValueError(f"Unsupported representation(s): {invalid}")
    return reps


def path_for_layer(path_value: Any, layer: int) -> Path:
    if pd.isna(path_value):
        raise ValueError("activation path is missing")
    text = str(path_value)
    if "{layer}" in text:
        return Path(text.format(layer=layer))
    replaced = re.sub(r"/layer_\d+/", f"/layer_{layer}/", text)
    return Path(replaced)


def bf16_uint16_to_float32(values: np.ndarray) -> np.ndarray:
    """Convert little-endian BF16 uint16 values to float32."""
    uint32_values = values.astype(np.uint32) << 16
    return uint32_values.view(np.float32)


class ActivationRowLoader:
    """Small LRU cache for row-level activation reads."""

    def __init__(self, *, max_open: int = 128):
        self.max_open = max_open
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def _get_bf16_memmap(self, path: Path, hidden_dim: int) -> np.memmap:
        key = (str(path), hidden_dim)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        size_bytes = path.stat().st_size
        itemsize = 2
        if size_bytes % (hidden_dim * itemsize) != 0:
            raise ValueError(f"File size is not divisible by hidden_dim*2 for {path}")
        rows = size_bytes // (hidden_dim * itemsize)
        mmap = np.memmap(path, mode="r", dtype="<u2", shape=(rows, hidden_dim))
        self._cache[key] = mmap
        if len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return mmap

    def load_row(
        self,
        *,
        path: Path,
        row_index: int,
        hidden_dim: int,
        activation_format: str | None,
        activation_key: str | None = None,
    ) -> np.ndarray:
        suffix = path.suffix.lower()
        fmt = activation_format or ""
        if suffix == ".bf16" or fmt == "packed_raw_bf16":
            mmap = self._get_bf16_memmap(path, hidden_dim)
            return bf16_uint16_to_float32(np.asarray(mmap[int(row_index)]))
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            return np.asarray(arr[int(row_index)], dtype=np.float32)
        if suffix == ".npz":
            key = activation_key if activation_key and activation_key != "row" else None
            with np.load(path, allow_pickle=False) as data:
                if key is None:
                    key = data.files[0]
                return np.asarray(data[key][int(row_index)], dtype=np.float32)
        if suffix in {".pt", ".pth"}:
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError("torch is required to read .pt/.pth activation files") from exc
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                key = activation_key if activation_key and activation_key in obj else next(iter(obj))
                obj = obj[key]
            if hasattr(obj, "detach"):
                obj = obj.detach().cpu().numpy()
            return np.asarray(obj[int(row_index)], dtype=np.float32)
        if suffix == ".parquet":
            table = pd.read_parquet(path)
            key = activation_key if activation_key and activation_key in table.columns else None
            if key is None:
                vector_cols = [col for col in table.columns if col.startswith("activation") or col.startswith("hidden")]
                if not vector_cols:
                    raise ValueError(f"No vector-like column found in {path}")
                key = vector_cols[0]
            return np.asarray(table.iloc[int(row_index)][key], dtype=np.float32)
        if suffix in {".h5", ".hdf5"}:
            try:
                import h5py
            except ModuleNotFoundError as exc:
                raise RuntimeError("h5py is required to read HDF5 activation files") from exc
            with h5py.File(path, "r") as handle:
                key = activation_key if activation_key and activation_key in handle else next(iter(handle.keys()))
                return np.asarray(handle[key][int(row_index)], dtype=np.float32)
        if path.name.endswith(".zarr") or suffix == ".zarr":
            try:
                import zarr
            except ModuleNotFoundError as exc:
                raise RuntimeError("zarr is required to read Zarr activation files") from exc
            root = zarr.open(path, mode="r")
            if hasattr(root, "shape"):
                return np.asarray(root[int(row_index)], dtype=np.float32)
            key = activation_key if activation_key and activation_key in root else next(iter(root.keys()))
            return np.asarray(root[key][int(row_index)], dtype=np.float32)
        raise ValueError(f"Unsupported activation file format for {path}")


def open_feature_array(
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: np.dtype,
    memmap: bool,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if memmap:
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return np.empty(shape, dtype=dtype)


def save_feature_array(path: Path, array: np.ndarray, *, memmap: bool) -> None:
    if memmap:
        array.flush()
    else:
        np.save(path, array)


def l2_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector.astype(np.float32)))


def progress_bin(progress: float, n_bins: int = 10) -> int:
    if not np.isfinite(progress):
        return -1
    return int(min(max(math.floor(progress * n_bins), 0), n_bins - 1))


def finite_check(name: str, array: np.ndarray, *, allow_nan: bool) -> int:
    if allow_nan:
        return int(np.isinf(array).sum())
    return int((~np.isfinite(array)).sum())


def metadata_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "commitment_to_final",
        "entropy_commitment",
        "margin_commitment",
        "q_UP",
        "q_DOWN",
        "q_LEFT",
        "q_RIGHT",
        "b_clr_UP",
        "b_clr_DOWN",
        "b_clr_LEFT",
        "b_clr_RIGHT",
        "is_before_first_action_mention",
        "is_before_first_final_action_mention",
    ]
    return [col for col in cols if col in df.columns]


def build_feature_metadata(
    aligned: pd.DataFrame,
    *,
    feature_files: dict[tuple[int, str], Path],
) -> pd.DataFrame:
    meta = aligned[metadata_columns(aligned)].copy()
    meta.insert(0, "row_index", np.arange(len(aligned), dtype=np.int64))
    for (layer, rep), path in sorted(feature_files.items()):
        meta[f"feature_file_layer_{layer:02d}_{rep}"] = str(path)
    return meta


def load_activation_for_row(
    loader: ActivationRowLoader,
    row: pd.Series,
    *,
    layer: int,
    kind: str,
    hidden_dim: int,
) -> np.ndarray:
    if kind == "main":
        file_col = "activation_file"
        row_col = "activation_row_index"
        fmt_col = "activation_format"
        key_col = "activation_key"
    elif kind == "pre":
        file_col = "pre_activation_file"
        row_col = "pre_activation_row_index"
        fmt_col = "activation_format"
        key_col = "pre_activation_key"
    elif kind == "post":
        file_col = "post_activation_file"
        row_col = "post_activation_row_index"
        fmt_col = "activation_format"
        key_col = "post_activation_key"
    else:
        raise ValueError(f"Unknown activation kind: {kind}")

    path = path_for_layer(row[file_col], layer)
    return loader.load_row(
        path=path,
        row_index=int(row[row_col]),
        hidden_dim=hidden_dim,
        activation_format=str(row.get(fmt_col) or ""),
        activation_key=str(row.get(key_col) or ""),
    )


def compute_features_for_layer(
    aligned: pd.DataFrame,
    *,
    layer: int,
    representations: list[str],
    output_dir: Path,
    output_dtype: np.dtype,
    memmap: bool,
    velocity_first: str,
    allow_nan: bool,
    batch_size: int,
    eps: float,
) -> tuple[dict[tuple[int, str], Path], list[dict[str, Any]], list[dict[str, Any]]]:
    n_rows = len(aligned)
    hidden_dims = aligned["hidden_dim"].dropna().astype(int).unique()
    if len(hidden_dims) != 1:
        raise ValueError(f"Expected one hidden_dim, found {hidden_dims}")
    hidden_dim = int(hidden_dims[0])

    feature_files: dict[tuple[int, str], Path] = {}
    arrays: dict[str, np.ndarray] = {}
    for rep in representations:
        path = output_dir / f"layer_{layer:02d}_{rep}.npy"
        arrays[rep] = open_feature_array(
            path,
            shape=(n_rows, hidden_dim),
            dtype=output_dtype,
            memmap=memmap,
        )
        feature_files[(layer, rep)] = path

    loader = ActivationRowLoader()
    validation_rows: list[dict[str, Any]] = []
    norm_acc: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    zero_delta_norms = []

    for decision_id, group in aligned.groupby("decision_id", sort=False):
        ordered = group.sort_values(["prefix_token_index"]).copy()
        h_prev: np.ndarray | None = None

        first_row = ordered.iloc[0]
        h_pre = load_activation_for_row(loader, first_row, layer=layer, kind="pre", hidden_dim=hidden_dim)
        h_post = load_activation_for_row(loader, first_row, layer=layer, kind="post", hidden_dim=hidden_dim)
        post_delta_pre = h_post - h_pre

        for local_pos, (_, row) in enumerate(ordered.iterrows()):
            row_index = int(row["_row_index"])
            h_k = load_activation_for_row(loader, row, layer=layer, kind="main", hidden_dim=hidden_dim)
            delta_pre = h_k - h_pre
            if h_prev is None:
                if velocity_first == "zero":
                    velocity = np.zeros_like(h_k, dtype=np.float32)
                else:
                    velocity = np.full_like(h_k, np.nan, dtype=np.float32)
            else:
                velocity = h_k - h_prev
            h_prev = h_k

            norm_delta = l2_norm(delta_pre)
            if int(row["prefix_token_index"]) == 0:
                zero_delta_norms.append(norm_delta)
            if norm_delta > eps:
                unit_delta = delta_pre / norm_delta
            else:
                unit_delta = np.zeros_like(delta_pre, dtype=np.float32)

            values = {
                "raw": h_k,
                "delta_pre": delta_pre,
                "post_delta_pre": post_delta_pre,
                "velocity": velocity,
                "unit_delta_pre": unit_delta,
            }

            for rep in representations:
                if rep == "progress_residualized_delta_pre":
                    raise NotImplementedError(
                        "progress_residualized_delta_pre is reserved for a later analysis step"
                    )
                vector = values[rep].astype(output_dtype, copy=False)
                invalid = finite_check(rep, vector, allow_nan=allow_nan)
                if invalid:
                    validation_rows.append(
                        {
                            "layer": layer,
                            "representation": rep,
                            "row_index": row_index,
                            "decision_id": decision_id,
                            "invalid_values": invalid,
                        }
                    )
                arrays[rep][row_index] = vector

                if rep in {"raw", "delta_pre", "velocity"}:
                    bin_id = progress_bin(float(row["progress"]))
                    acc = norm_acc[(rep, bin_id)]
                    acc[0] += l2_norm(vector)
                    acc[1] += 1

    for rep, array in arrays.items():
        save_feature_array(feature_files[(layer, rep)], array, memmap=memmap)

    norm_rows = []
    for (rep, bin_id), (total, count) in sorted(norm_acc.items()):
        norm_rows.append(
            {
                "layer": layer,
                "representation": rep,
                "progress_bin": bin_id,
                "mean_l2_norm": total / count if count else np.nan,
                "count": int(count),
            }
        )
    if zero_delta_norms:
        norm_rows.append(
            {
                "layer": layer,
                "representation": "delta_pre_prefix0_validation",
                "progress_bin": -1,
                "mean_l2_norm": float(np.mean(zero_delta_norms)),
                "max_l2_norm": float(np.max(zero_delta_norms)),
                "count": int(len(zero_delta_norms)),
            }
        )
    return feature_files, validation_rows, norm_rows


def write_report(
    path: Path,
    *,
    aligned: pd.DataFrame,
    layers: list[int],
    representations: list[str],
    output_dtype: np.dtype,
    memmap: bool,
    feature_files: dict[tuple[int, str], Path],
    validation_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    velocity_first: str,
    progress_residualize: bool,
) -> None:
    lines = [
        "# Activation Feature Construction Report",
        "",
        "## Summary",
        "",
        f"- Input rows: {len(aligned):,}",
        f"- Decisions: {aligned['decision_id'].nunique():,}",
        f"- Layers: {layers}",
        f"- Representations: {representations}",
        f"- Output dtype: `{np.dtype(output_dtype).name}`",
        f"- Memmap output: {memmap}",
        f"- Velocity first-prefix policy: `{velocity_first}`",
        f"- Progress residualization requested: {progress_residualize}",
        f"- Feature files written: {len(feature_files):,}",
        f"- Validation invalid-value records: {len(validation_rows):,}",
        "",
        "## Feature Files",
        "",
    ]
    for (layer, rep), feature_path in sorted(feature_files.items()):
        size = feature_path.stat().st_size if feature_path.exists() else 0
        lines.append(f"- layer {layer:02d} `{rep}`: `{feature_path}` ({size / (1024 ** 2):.1f} MiB)")

    lines.extend(["", "## L2 Norms By Progress Bin", ""])
    if norm_rows:
        norm_df = pd.DataFrame(norm_rows)
        main_norms = norm_df[norm_df["progress_bin"].ge(0)].copy()
        for (layer, rep), sub in main_norms.groupby(["layer", "representation"]):
            compact = {
                int(row["progress_bin"]): round(float(row["mean_l2_norm"]), 4)
                for _, row in sub.iterrows()
            }
            lines.append(f"- layer {layer:02d} `{rep}`: {compact}")
        validation = norm_df[norm_df["representation"].eq("delta_pre_prefix0_validation")]
        for _, row in validation.iterrows():
            lines.append(
                f"- layer {int(row['layer']):02d} prefix-0 delta_pre mean/max L2: "
                f"{row['mean_l2_norm']:.6g} / {row.get('max_l2_norm', np.nan):.6g} over {int(row['count'])} rows"
            )
    else:
        lines.append("- No norm rows recorded.")

    lines.extend(["", "## Validation", ""])
    if validation_rows:
        lines.append("- Non-finite values were detected:")
        for row in validation_rows[:20]:
            lines.append(f"  - {row}")
    else:
        lines.append("- No non-finite values detected under the configured validation policy.")
    lines.append("- `delta_pre` for `prefix_token_index == 0` was checked through L2 norm summaries above.")
    if progress_residualize:
        lines.append("- Progress residualization is intentionally not implemented yet; requested residualized features were skipped/blocked.")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/activation/activation_behavior_aligned.parquet")
    parser.add_argument("--output_dir", default="outputs/activation/features")
    parser.add_argument("--metadata_output", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--representations", default=",".join(DEFAULT_REPRESENTATIONS))
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memmap", type=parse_bool, default=True)
    parser.add_argument("--velocity_first", choices=["zero", "nan"], default="zero")
    parser.add_argument("--allow_nan", action="store_true")
    parser.add_argument("--progress_residualize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    metadata_output = Path(args.metadata_output)
    report_path = metadata_output.parent / "reports" / "activation_feature_report.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_parquet(input_path).reset_index(drop=True)
    aligned["_row_index"] = np.arange(len(aligned), dtype=np.int64)
    layers = parse_layers(args.layers, aligned["available_layers_json"])
    representations = parse_representations(args.representations)
    if "progress_residualized_delta_pre" in representations and not args.progress_residualize:
        raise SystemExit(
            "Representation progress_residualized_delta_pre requires --progress_residualize; "
            "the option is reserved for later implementation."
        )
    if args.progress_residualize:
        raise SystemExit("Progress residualization option is present but not implemented yet.")

    output_dtype = np.dtype(args.dtype)
    all_feature_files: dict[tuple[int, str], Path] = {}
    all_validation_rows: list[dict[str, Any]] = []
    all_norm_rows: list[dict[str, Any]] = []

    for layer in layers:
        print(f"Building activation features for layer {layer}...")
        feature_files, validation_rows, norm_rows = compute_features_for_layer(
            aligned,
            layer=layer,
            representations=representations,
            output_dir=output_dir,
            output_dtype=output_dtype,
            memmap=args.memmap,
            velocity_first=args.velocity_first,
            allow_nan=args.allow_nan,
            batch_size=args.batch_size,
            eps=1e-12,
        )
        all_feature_files.update(feature_files)
        all_validation_rows.extend(validation_rows)
        all_norm_rows.extend(norm_rows)

    if all_validation_rows and not args.allow_nan:
        sample = all_validation_rows[:5]
        raise RuntimeError(f"Non-finite activation feature values detected: {sample}")

    metadata = build_feature_metadata(aligned, feature_files=all_feature_files)
    metadata.to_parquet(metadata_output, index=False)
    write_report(
        report_path,
        aligned=aligned,
        layers=layers,
        representations=representations,
        output_dtype=output_dtype,
        memmap=args.memmap,
        feature_files=all_feature_files,
        validation_rows=all_validation_rows,
        norm_rows=all_norm_rows,
        velocity_first=args.velocity_first,
        progress_residualize=args.progress_residualize,
    )

    print(f"Wrote {metadata_output}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(all_feature_files)} feature arrays under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
