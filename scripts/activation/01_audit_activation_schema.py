#!/usr/bin/env python3
"""Audit existing activation files without loading large activation arrays.

The script is intentionally schema-tolerant. It recursively inventories common
activation/metadata formats and records whatever shape, dtype, keys, and token
metadata can be inferred cheaply.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOKEN_TYPES = {"pre_reasoning", "cot", "post_reasoning"}
SUPPORTED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".parquet",
    ".json",
    ".jsonl",
    ".h5",
    ".hdf5",
    ".zarr",
    ".bf16",
    ".md",
    ".txt",
}


def json_dumps(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return json.dumps(str(value))


def parse_int_list(text: str) -> list[int] | None:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            return None
    return values


def read_text_head(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        return handle.read(max_bytes).decode("utf-8", errors="replace")


def maybe_load_json(path: Path, max_bytes: int) -> Any | None:
    if path.stat().st_size > max_bytes:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_json_head_metadata(path: Path, max_full_load_bytes: int) -> dict[str, Any]:
    """Extract common JSON metadata, avoiding full loads for large sidecars."""
    data = maybe_load_json(path, max_full_load_bytes)
    if isinstance(data, dict):
        summary = {
            "keys": list(data.keys())[:100],
            "format": data.get("format"),
            "dtype": data.get("dtype"),
            "shape": data.get("shape"),
            "category": data.get("category"),
            "layer_idx": data.get("layer_idx"),
            "step_idx": data.get("step_idx"),
            "byte_order": data.get("byte_order"),
        }
        token_indices = data.get("token_indices")
        if isinstance(token_indices, list):
            summary["token_indices_count"] = len(token_indices)
            summary["token_indices_sample"] = token_indices[:3]
        return summary

    head = read_text_head(path)
    known_keys = [
        "format",
        "dtype",
        "shape",
        "byte_order",
        "category",
        "layer_idx",
        "step_idx",
        "token_indices",
    ]
    summary: dict[str, Any] = {"keys": [key for key in known_keys if f'"{key}"' in head]}
    shape_match = re.search(r'"shape"\s*:\s*\[([^\]]+)\]', head)
    if shape_match:
        summary["shape"] = parse_int_list(shape_match.group(1))
    for key in ["format", "dtype", "byte_order", "category"]:
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', head)
        if match:
            summary[key] = match.group(1)
    for key in ["layer_idx", "step_idx"]:
        match = re.search(rf'"{key}"\s*:\s*(-?\d+)', head)
        if match:
            summary[key] = int(match.group(1))
    if "shape" in summary and isinstance(summary["shape"], list) and summary["shape"]:
        summary["token_indices_count"] = summary["shape"][0]
    return summary


def infer_path_metadata(path: Path, input_dir: Path) -> dict[str, Any]:
    rel = path.relative_to(input_dir) if path.is_relative_to(input_dir) else path
    parts = rel.parts
    metadata: dict[str, Any] = {}
    if len(parts) >= 1:
        metadata["trajectory_id"] = parts[0] if not parts[0].startswith("_") else None
    if len(parts) >= 2:
        metadata["model_id"] = parts[1]
    for part in parts:
        layer_match = re.fullmatch(r"layer_(\d+)", part)
        if layer_match:
            metadata["layer"] = int(layer_match.group(1))
        step_match = re.fullmatch(r"step_(\d+)", part)
        if step_match:
            metadata["step_id"] = int(step_match.group(1))
    if path.stem in TOKEN_TYPES:
        metadata["token_type"] = path.stem
    if metadata.get("trajectory_id") is not None and metadata.get("step_id") is not None:
        metadata["decision_id"] = f"{metadata['trajectory_id']}::{metadata['step_id']}"
    return metadata


def shape_to_hidden_dim(shape: list[int] | tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    if len(shape) >= 1:
        return int(shape[-1])
    return None


def shape_to_rows(shape: list[int] | tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    return int(shape[0])


def infer_storage_layout(
    *,
    path: Path,
    role: str,
    shape: list[int] | None,
    path_meta: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    suffix = path.suffix.lower()
    has_layer_step = path_meta.get("layer") is not None and path_meta.get("step_id") is not None
    token_type = metadata.get("category") or path_meta.get("token_type")
    if suffix == ".bf16" and has_layer_step and token_type in TOKEN_TYPES:
        return "one matrix per trajectory/model/layer/step/token_type with JSON sidecar"
    if suffix == ".json" and metadata.get("format") == "raw_bfloat16":
        return "JSON sidecar for one raw BF16 matrix per trajectory/model/layer/step/token_type"
    if suffix in {".npy", ".pt", ".pth"} and shape is not None:
        if len(shape) == 1:
            return "possibly one tensor per token or one activation vector"
        if len(shape) == 2:
            return "possibly one tensor per decision/layer with rows as tokens"
        if len(shape) >= 3:
            return "possibly batched tensor with layer/token/vector axes"
    if suffix == ".npz":
        return "multi-array archive"
    if suffix == ".parquet":
        return "table, possibly one row per token/prefix with vector columns"
    if suffix in {".h5", ".hdf5"}:
        return "HDF5 group/dataset structure"
    if suffix == ".zarr" or path.name.endswith(".zarr"):
        return "Zarr group/array structure"
    if role == "metadata":
        return "metadata/documentation"
    return "unknown"


def base_row(path: Path, input_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    path_meta = infer_path_metadata(path, input_dir)
    row = {
        "file_path": str(path),
        "relative_path": str(path.relative_to(input_dir) if path.is_relative_to(input_dir) else path),
        "file_type": path.suffix.lower() or ("zarr" if path.name.endswith(".zarr") else ""),
        "size_bytes": int(stat.st_size) if path.is_file() else 0,
        "role": "unknown",
        "keys_json": "",
        "array_shapes_json": "",
        "shape_json": "",
        "dtype": "",
        "likely_num_layers": None,
        "likely_hidden_dim": None,
        "likely_num_tokens_rows": None,
        "trajectory_id": path_meta.get("trajectory_id"),
        "model_id": path_meta.get("model_id"),
        "decision_id": path_meta.get("decision_id"),
        "step_id": path_meta.get("step_id"),
        "token_type": path_meta.get("token_type"),
        "layer": path_meta.get("layer"),
        "storage_inference": "unknown",
        "metadata_summary_json": "",
        "error": "",
    }
    return row


def inspect_raw_bf16(path: Path, input_dir: Path, max_json_bytes: int) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "activation"
    sidecar = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if sidecar.exists():
        metadata = extract_json_head_metadata(sidecar, max_json_bytes)
        shape = metadata.get("shape")
        row["dtype"] = metadata.get("dtype") or "torch.bfloat16"
        row["shape_json"] = json_dumps(shape)
        row["array_shapes_json"] = json_dumps({path.name: shape})
        row["keys_json"] = json_dumps(metadata.get("keys"))
        row["token_type"] = metadata.get("category") or row["token_type"]
        row["layer"] = metadata.get("layer_idx", row["layer"])
        row["step_id"] = metadata.get("step_idx", row["step_id"])
        row["likely_hidden_dim"] = shape_to_hidden_dim(shape)
        row["likely_num_tokens_rows"] = shape_to_rows(shape)
        row["likely_num_layers"] = 1 if row["layer"] is not None else None
        row["metadata_summary_json"] = json_dumps(
            {
                "sidecar": str(sidecar),
                "format": metadata.get("format"),
                "byte_order": metadata.get("byte_order"),
                "token_indices_count": metadata.get("token_indices_count"),
                "expected_payload_bytes": int(math.prod(shape) * 2) if shape else None,
                "payload_size_matches_shape": (
                    int(math.prod(shape) * 2) == row["size_bytes"] if shape else None
                ),
            }
        )
    else:
        shape = None
        row["dtype"] = "torch.bfloat16"
        row["error"] = "missing JSON sidecar for raw bf16 file"
    row["storage_inference"] = infer_storage_layout(
        path=path,
        role=row["role"],
        shape=shape,
        path_meta=infer_path_metadata(path, input_dir),
        metadata=metadata,
    )
    return row


def inspect_json(path: Path, input_dir: Path, max_json_bytes: int) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "metadata"
    metadata = extract_json_head_metadata(path, max_json_bytes)
    shape = metadata.get("shape")
    row["keys_json"] = json_dumps(metadata.get("keys"))
    row["shape_json"] = json_dumps(shape)
    row["array_shapes_json"] = json_dumps({path.name: shape} if shape else None)
    row["dtype"] = metadata.get("dtype") or ""
    row["token_type"] = metadata.get("category") or row["token_type"]
    row["layer"] = metadata.get("layer_idx", row["layer"])
    row["step_id"] = metadata.get("step_idx", row["step_id"])
    row["likely_hidden_dim"] = shape_to_hidden_dim(shape)
    row["likely_num_tokens_rows"] = shape_to_rows(shape)
    row["likely_num_layers"] = 1 if row["layer"] is not None else None
    row["metadata_summary_json"] = json_dumps(metadata)
    row["storage_inference"] = infer_storage_layout(
        path=path,
        role=row["role"],
        shape=shape,
        path_meta=infer_path_metadata(path, input_dir),
        metadata=metadata,
    )
    return row


def inspect_jsonl(path: Path, input_dir: Path, max_lines: int = 20) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "metadata"
    keys: set[str] = set()
    examples = []
    total_lines = 0
    try:
        with path.open() as handle:
            for line in handle:
                total_lines += 1
                if len(examples) < max_lines:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            keys.update(item.keys())
                            examples.append({k: item[k] for k in list(item.keys())[:20]})
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        row["error"] = str(exc)
    row["keys_json"] = json_dumps(sorted(keys))
    row["likely_num_tokens_rows"] = total_lines
    row["metadata_summary_json"] = json_dumps({"sample_rows": examples[:3], "line_count": total_lines})
    row["storage_inference"] = "JSONL metadata/event log"
    return row


def inspect_numpy(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "activation"
    try:
        if path.suffix.lower() == ".npz":
            import zipfile

            shapes = {}
            dtypes = {}
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if not member.endswith(".npy"):
                        continue
                    key = member[:-4]
                    with archive.open(member) as handle:
                        version = np.lib.format.read_magic(handle)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                        else:
                            shape, _, dtype = np.lib.format._read_array_header(
                                handle,
                                version,
                                max_header_size=100000,
                            )
                    shapes[key] = list(shape)
                    dtypes[key] = str(dtype)
            row["keys_json"] = json_dumps(list(shapes.keys()))
            row["array_shapes_json"] = json_dumps(shapes)
            row["dtype"] = json_dumps(dtypes)
            first_shape = next(iter(shapes.values()), None)
            row["likely_hidden_dim"] = shape_to_hidden_dim(first_shape)
            row["likely_num_tokens_rows"] = shape_to_rows(first_shape)
            row["storage_inference"] = "multi-array archive"
        else:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = list(arr.shape)
            row["shape_json"] = json_dumps(shape)
            row["array_shapes_json"] = json_dumps({path.name: shape})
            row["dtype"] = str(arr.dtype)
            row["likely_hidden_dim"] = shape_to_hidden_dim(shape)
            row["likely_num_tokens_rows"] = shape_to_rows(shape)
            row["storage_inference"] = infer_storage_layout(
                path=path,
                role=row["role"],
                shape=shape,
                path_meta=infer_path_metadata(path, input_dir),
                metadata={},
            )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def inspect_parquet(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "table"
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        row["keys_json"] = json_dumps(schema.names)
        row["likely_num_tokens_rows"] = pf.metadata.num_rows
        row["metadata_summary_json"] = json_dumps(
            {
                "num_row_groups": pf.metadata.num_row_groups,
                "schema": str(schema),
            }
        )
        for name in schema.names:
            lower = name.lower()
            if lower in {"layer", "layer_idx"}:
                row["likely_num_layers"] = None
            if lower in {"activation", "hidden_state", "vector"}:
                row["storage_inference"] = "one table with vectors"
        if row["storage_inference"] == "unknown":
            row["storage_inference"] = "table, schema inspected without loading full data"
    except ModuleNotFoundError:
        row["error"] = "pyarrow not installed; parquet metadata inspection unavailable"
    except Exception as exc:
        row["error"] = str(exc)
    return row


def inspect_hdf5(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "activation"
    try:
        import h5py
    except ModuleNotFoundError:
        row["error"] = "h5py not installed; HDF5 inspection unavailable"
        row["storage_inference"] = "HDF5 group/dataset structure"
        return row

    shapes = {}
    dtypes = {}
    try:
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    shapes[name] = list(obj.shape)
                    dtypes[name] = str(obj.dtype)

            handle.visititems(visitor)
            row["keys_json"] = json_dumps(list(shapes.keys())[:200])
            row["array_shapes_json"] = json_dumps(shapes)
            row["dtype"] = json_dumps(dtypes)
            first_shape = next(iter(shapes.values()), None)
            row["likely_hidden_dim"] = shape_to_hidden_dim(first_shape)
            row["likely_num_tokens_rows"] = shape_to_rows(first_shape)
    except Exception as exc:
        row["error"] = str(exc)
    row["storage_inference"] = "HDF5 group/dataset structure"
    return row


def inspect_zarr(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "activation"
    try:
        import zarr
    except ModuleNotFoundError:
        row["error"] = "zarr not installed; Zarr inspection unavailable"
        row["storage_inference"] = "Zarr group/array structure"
        return row

    shapes = {}
    dtypes = {}
    try:
        root = zarr.open(path, mode="r")
        if hasattr(root, "shape"):
            shapes[path.name] = list(root.shape)
            dtypes[path.name] = str(root.dtype)
        else:
            for name, value in root.arrays():
                shapes[name] = list(value.shape)
                dtypes[name] = str(value.dtype)
        row["keys_json"] = json_dumps(list(shapes.keys())[:200])
        row["array_shapes_json"] = json_dumps(shapes)
        row["dtype"] = json_dumps(dtypes)
        first_shape = next(iter(shapes.values()), None)
        row["likely_hidden_dim"] = shape_to_hidden_dim(first_shape)
        row["likely_num_tokens_rows"] = shape_to_rows(first_shape)
    except Exception as exc:
        row["error"] = str(exc)
    row["storage_inference"] = "Zarr group/array structure"
    return row


def inspect_torch(path: Path, input_dir: Path, max_torch_load_bytes: int) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "activation"
    if path.stat().st_size > max_torch_load_bytes:
        row["error"] = (
            f"skipped torch.load for large file > {max_torch_load_bytes} bytes; "
            "shape unavailable without loading"
        )
        row["storage_inference"] = "PyTorch tensor/checkpoint, not loaded because file is large"
        return row
    try:
        import torch
    except ModuleNotFoundError:
        row["error"] = "torch not installed; PyTorch file inspection unavailable"
        row["storage_inference"] = "PyTorch tensor/checkpoint"
        return row
    try:
        obj = torch.load(path, map_location="cpu")
        shapes = {}
        dtypes = {}

        def walk(value: Any, name: str) -> None:
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                shapes[name] = list(value.shape)
                dtypes[name] = str(value.dtype)
            elif isinstance(value, dict):
                for key, child in list(value.items())[:200]:
                    walk(child, f"{name}.{key}" if name else str(key))
            elif isinstance(value, (list, tuple)):
                for idx, child in enumerate(value[:200]):
                    walk(child, f"{name}.{idx}" if name else str(idx))

        walk(obj, "")
        row["keys_json"] = json_dumps(list(shapes.keys())[:200])
        row["array_shapes_json"] = json_dumps(shapes)
        row["dtype"] = json_dumps(dtypes)
        first_shape = next(iter(shapes.values()), None)
        row["likely_hidden_dim"] = shape_to_hidden_dim(first_shape)
        row["likely_num_tokens_rows"] = shape_to_rows(first_shape)
        row["storage_inference"] = infer_storage_layout(
            path=path,
            role=row["role"],
            shape=first_shape,
            path_meta=infer_path_metadata(path, input_dir),
            metadata={},
        )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def inspect_document(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["role"] = "metadata"
    row["keys_json"] = json_dumps(["documentation_text"])
    try:
        text = read_text_head(path, 4096)
        title = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
        row["metadata_summary_json"] = json_dumps({"title_or_first_line": title, "excerpt": text[:1000]})
    except Exception as exc:
        row["error"] = str(exc)
    row["storage_inference"] = "metadata/documentation"
    return row


def inspect_unknown(path: Path, input_dir: Path) -> dict[str, Any]:
    row = base_row(path, input_dir)
    row["error"] = "unsupported or unknown file extension"
    return row


def inspect_path(
    path: Path,
    input_dir: Path,
    *,
    max_json_bytes: int,
    max_torch_load_bytes: int,
) -> dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if path.name.endswith(".zarr") or suffix == ".zarr":
            return inspect_zarr(path, input_dir)
        if suffix == ".bf16":
            return inspect_raw_bf16(path, input_dir, max_json_bytes)
        if suffix == ".json":
            return inspect_json(path, input_dir, max_json_bytes)
        if suffix == ".jsonl":
            return inspect_jsonl(path, input_dir)
        if suffix in {".npy", ".npz"}:
            return inspect_numpy(path, input_dir)
        if suffix == ".parquet":
            return inspect_parquet(path, input_dir)
        if suffix in {".h5", ".hdf5"}:
            return inspect_hdf5(path, input_dir)
        if suffix in {".pt", ".pth"}:
            return inspect_torch(path, input_dir, max_torch_load_bytes)
        if suffix in {".md", ".txt"}:
            return inspect_document(path, input_dir)
        return inspect_unknown(path, input_dir)
    except Exception as exc:
        row = base_row(path, input_dir)
        row["error"] = f"inspection failed: {type(exc).__name__}: {exc}"
        return row


def iter_inventory_paths(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_dir():
            if path.name.endswith(".zarr"):
                paths.append(path)
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def summarize_inventory(df: pd.DataFrame, input_dir: Path) -> dict[str, Any]:
    activation_rows = df[df["role"].eq("activation")]
    layers = sorted(int(x) for x in df["layer"].dropna().unique())
    hidden_dims = sorted(int(x) for x in df["likely_hidden_dim"].dropna().unique())
    token_types = sorted(str(x) for x in df["token_type"].dropna().unique())
    decision_count = df["decision_id"].dropna().nunique()
    trajectory_count = df["trajectory_id"].dropna().nunique()
    storage_counts = df["storage_inference"].value_counts(dropna=False).to_dict()
    return {
        "input_dir": str(input_dir),
        "num_files": int(len(df)),
        "num_activation_files": int(len(activation_rows)),
        "num_metadata_files": int(df["role"].eq("metadata").sum()),
        "total_size_bytes": int(df["size_bytes"].sum()),
        "file_type_counts": df["file_type"].value_counts(dropna=False).to_dict(),
        "role_counts": df["role"].value_counts(dropna=False).to_dict(),
        "storage_inference_counts": storage_counts,
        "trajectory_count": int(trajectory_count),
        "decision_count": int(decision_count),
        "layers": layers,
        "num_layers": len(layers),
        "hidden_dims": hidden_dims,
        "token_types": token_types,
        "has_pre_reasoning": "pre_reasoning" in token_types,
        "has_cot": "cot" in token_types,
        "has_post_reasoning": "post_reasoning" in token_types,
        "error_count": int(df["error"].fillna("").ne("").sum()),
    }


def example_records(df: pd.DataFrame, max_examples: int) -> list[dict[str, Any]]:
    preferred = df.sort_values(["role", "relative_path"]).head(max_examples)
    examples = []
    for _, row in preferred.iterrows():
        examples.append(
            {
                "relative_path": row.get("relative_path"),
                "role": row.get("role"),
                "file_type": row.get("file_type"),
                "size_bytes": int(row.get("size_bytes") or 0),
                "shape": json.loads(row["shape_json"]) if row.get("shape_json") else None,
                "dtype": row.get("dtype"),
                "token_type": row.get("token_type"),
                "layer": row.get("layer"),
                "step_id": row.get("step_id"),
                "storage_inference": row.get("storage_inference"),
                "metadata_summary": (
                    json.loads(row["metadata_summary_json"])
                    if row.get("metadata_summary_json")
                    else None
                ),
                "error": row.get("error"),
            }
        )
    return examples


def write_schema_examples(
    path: Path,
    *,
    summary: dict[str, Any],
    df: pd.DataFrame,
    optional_warnings: list[str],
    max_examples: int,
) -> None:
    token_type_examples = {}
    for token_type in ["pre_reasoning", "cot", "post_reasoning"]:
        subset = df[df["token_type"].eq(token_type)].head(1)
        token_type_examples[token_type] = (
            example_records(subset, 1)[0] if not subset.empty else None
        )

    examples = {
        "summary": summary,
        "optional_package_warnings": optional_warnings,
        "sample_records": example_records(df, max_examples),
        "token_type_examples": token_type_examples,
    }
    path.write_text(json.dumps(examples, indent=2, sort_keys=True))


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def write_report(path: Path, *, summary: dict[str, Any], df: pd.DataFrame) -> None:
    lines = [
        "# Activation Schema Audit Report",
        "",
        "## Inputs",
        "",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Files inventoried: {summary['num_files']:,}",
        f"- Activation files: {summary['num_activation_files']:,}",
        f"- Metadata/documentation files: {summary['num_metadata_files']:,}",
        f"- Total inventoried size: {human_bytes(summary['total_size_bytes'])}",
        f"- Inspection errors/warnings recorded on rows: {summary['error_count']:,}",
        "",
        "## File Types",
        "",
    ]
    for file_type, count in summary["file_type_counts"].items():
        lines.append(f"- `{file_type}`: {count:,}")

    lines.extend(["", "## Inferred Structure", ""])
    lines.append(f"- Trajectories found: {summary['trajectory_count']:,}")
    lines.append(f"- Decision IDs inferred from paths: {summary['decision_count']:,}")
    lines.append(f"- Layers found: {summary['layers']}")
    lines.append(f"- Number of layers found: {summary['num_layers']}")
    lines.append(f"- Hidden dimensions found: {summary['hidden_dims']}")
    lines.append(f"- Token types found: {summary['token_types']}")
    lines.append(f"- Has `pre_reasoning`: {summary['has_pre_reasoning']}")
    lines.append(f"- Has `cot`: {summary['has_cot']}")
    lines.append(f"- Has `post_reasoning`: {summary['has_post_reasoning']}")

    lines.extend(["", "## Storage Layout Inference", ""])
    for layout, count in summary["storage_inference_counts"].items():
        lines.append(f"- {layout}: {count:,}")

    lines.extend(["", "## Shape Summary", ""])
    shape_counts = df["shape_json"].replace("", np.nan).dropna().value_counts().head(20)
    if shape_counts.empty:
        lines.append("- No shapes could be inferred.")
    else:
        for shape, count in shape_counts.items():
            lines.append(f"- `{shape}`: {count:,} files")

    lines.extend(["", "## Token-Type Counts", ""])
    token_counts = df["token_type"].fillna("unknown").value_counts()
    for token_type, count in token_counts.items():
        lines.append(f"- `{token_type}`: {count:,}")

    lines.extend(["", "## Findings", ""])
    if any("raw BF16" in str(x) or "raw_bfloat16" in str(x) for x in df["storage_inference"]):
        lines.append(
            "- A packed raw BF16 layout with JSON sidecars was detected. The raw `.bf16` "
            "payloads should be read only when needed, using sidecar `shape` and `dtype`."
        )
    if summary["has_pre_reasoning"] and summary["has_cot"] and summary["has_post_reasoning"]:
        lines.append(
            "- The expected conceptual token types are present: `pre_reasoning`, `cot`, and `post_reasoning`."
        )
    if summary["hidden_dims"]:
        lines.append(f"- The likely hidden dimension is {summary['hidden_dims'][0]}." if len(summary["hidden_dims"]) == 1 else f"- Multiple hidden dimensions were found: {summary['hidden_dims']}.")
    if summary["layers"]:
        lines.append("- Activations appear to be organized by layer directories or layer metadata.")

    lines.extend(["", "## Recommendations", ""])
    lines.append("- Use JSON sidecars or file metadata to validate shape/dtype before reading activation arrays.")
    lines.append("- For packed BF16 records, compute `delta_h_k = h_cot_token_k - h_pre_reasoning` after loading only the needed trajectory/layer/step records.")
    lines.append("- Align activation CoT token index `k` to behavior `prefix_token_index k`; validate row counts before joining.")
    lines.append("- Use group splits by `trajectory_id` or `decision_id` to avoid leakage between train and test rows.")
    lines.append("- Keep per-layer sweeps streaming/lazy where possible; the raw activation payload is large.")

    lines.extend(["", "## Row-Level Error Samples", ""])
    errors = df[df["error"].fillna("").ne("")][["relative_path", "error"]].head(20)
    if errors.empty:
        lines.append("- No row-level errors recorded.")
    else:
        for _, row in errors.iterrows():
            lines.append(f"- `{row['relative_path']}`: {row['error']}")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/activations", help="Activation root to scan.")
    parser.add_argument("--output_dir", default="outputs/activation", help="Output directory.")
    parser.add_argument(
        "--max_json_full_load_mb",
        type=float,
        default=2.0,
        help="Fully parse JSON files up to this size; larger JSON is inspected from its head only.",
    )
    parser.add_argument(
        "--max_torch_load_mb",
        type=float,
        default=128.0,
        help="Only torch.load .pt/.pth files up to this size.",
    )
    parser.add_argument("--max_examples", type=int, default=12)
    parser.add_argument("--progress_every", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    max_json_bytes = int(args.max_json_full_load_mb * 1024 * 1024)
    max_torch_bytes = int(args.max_torch_load_mb * 1024 * 1024)

    paths = iter_inventory_paths(input_dir)
    print(f"Found {len(paths):,} candidate activation/metadata files under {input_dir}")
    rows = []
    optional_warnings: list[str] = []

    for idx, path in enumerate(paths, start=1):
        if args.progress_every and idx % args.progress_every == 0:
            print(f"Inspected {idx:,}/{len(paths):,} files...")
        row = inspect_path(
            path,
            input_dir,
            max_json_bytes=max_json_bytes,
            max_torch_load_bytes=max_torch_bytes,
        )
        if row.get("error", "").endswith("inspection unavailable"):
            optional_warnings.append(row["error"])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "file_path",
                "relative_path",
                "file_type",
                "size_bytes",
                "role",
                "keys_json",
                "array_shapes_json",
                "shape_json",
                "dtype",
                "likely_num_layers",
                "likely_hidden_dim",
                "likely_num_tokens_rows",
                "trajectory_id",
                "model_id",
                "decision_id",
                "step_id",
                "token_type",
                "layer",
                "storage_inference",
                "metadata_summary_json",
                "error",
            ]
        )

    inventory_path = output_dir / "activation_inventory.parquet"
    report_path = output_dir / "activation_schema_report.md"
    examples_path = output_dir / "schema_examples.json"

    df.to_parquet(inventory_path, index=False)
    summary = summarize_inventory(df, input_dir)
    write_schema_examples(
        examples_path,
        summary=summary,
        df=df,
        optional_warnings=sorted(set(optional_warnings)),
        max_examples=args.max_examples,
    )
    write_report(report_path, summary=summary, df=df)

    print(f"Wrote {inventory_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {examples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
