#!/usr/bin/env python3
"""Build a canonical activation timepoint index without storing vectors.

The index stores pointers into already-extracted activation files. For the
packed raw BF16 layout, each canonical row points to a token row inside a matrix
and records the layers available for that same timepoint.
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


TOKEN_TYPES = ["pre_reasoning", "cot", "post_reasoning"]
TOKEN_TYPE_SET = set(TOKEN_TYPES)


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def parse_int_list(text: str) -> list[int] | None:
    values = []
    for item in text.split(","):
        try:
            values.append(int(item.strip()))
        except ValueError:
            return None
    return values


def read_head(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        return handle.read(max_bytes).decode("utf-8", errors="replace")


def extract_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Read only the JSON head fields needed to index a packed BF16 sidecar."""
    head = read_head(path)
    if '"format"' not in head or "raw_bfloat16" not in head:
        return {}

    meta: dict[str, Any] = {"json_file": str(path)}
    shape_match = re.search(r'"shape"\s*:\s*\[([^\]]+)\]', head)
    if shape_match:
        meta["shape"] = parse_int_list(shape_match.group(1))
    for key in ["format", "dtype", "byte_order", "category"]:
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', head)
        if match:
            meta[key] = match.group(1)
    for key in ["layer_idx", "step_idx"]:
        match = re.search(rf'"{key}"\s*:\s*(-?\d+)', head)
        if match:
            meta[key] = int(match.group(1))
    return meta


def infer_path_metadata(path: Path, input_dir: Path) -> dict[str, Any]:
    rel = path.relative_to(input_dir) if path.is_relative_to(input_dir) else path
    parts = rel.parts
    meta: dict[str, Any] = {"relative_path": str(rel)}
    if len(parts) >= 1 and not parts[0].startswith("_"):
        meta["trajectory_id"] = parts[0]
    if len(parts) >= 2:
        meta["model_id"] = parts[1]
    for part in parts:
        layer_match = re.fullmatch(r"layer_(\d+)", part)
        if layer_match:
            meta["layer"] = int(layer_match.group(1))
        step_match = re.fullmatch(r"step_(\d+)", part)
        if step_match:
            meta["step_id"] = int(step_match.group(1))
    if path.stem in TOKEN_TYPE_SET:
        meta["token_type"] = path.stem
    return meta


def make_decision_id(trajectory_id: object, step_id: object) -> str:
    return f"{trajectory_id}::{step_id}"


def file_template(path: Path) -> str:
    return re.sub(r"/layer_\d+/", "/layer_{layer}/", str(path))


def load_optional_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional parquet not found: {path}")
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        print(f"WARNING: failed to read optional parquet {path}: {exc}")
        return None


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def build_behavior_summary(prefixes_path: Path, commitments_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "prefixes_available": False,
        "commitments_available": False,
        "prefix_decision_count": None,
        "prefix_row_count": None,
        "commitment_decision_count": None,
        "commitment_row_count": None,
    }
    if prefixes_path.exists():
        cols = ["trajectory_id", "step_id", "decision_id", "prefix_token_index", "num_analysis_tokens"]
        available = parquet_columns(prefixes_path)
        selected = [c for c in cols if c in available]
        prefixes = pd.read_parquet(prefixes_path, columns=selected)
        if "decision_id" not in prefixes.columns and {"trajectory_id", "step_id"}.issubset(prefixes.columns):
            prefixes["decision_id"] = prefixes["trajectory_id"].astype(str) + "::" + prefixes["step_id"].astype(str)
        summary.update(
            {
                "prefixes_available": True,
                "prefix_decision_count": int(prefixes["decision_id"].nunique()) if "decision_id" in prefixes.columns else None,
                "prefix_row_count": int(len(prefixes)),
            }
        )
    if commitments_path.exists():
        cols = ["trajectory_id", "step_id", "decision_id", "prefix_token_index"]
        available = parquet_columns(commitments_path)
        selected = [c for c in cols if c in available]
        commits = pd.read_parquet(commitments_path, columns=selected)
        if "decision_id" not in commits.columns and {"trajectory_id", "step_id"}.issubset(commits.columns):
            commits["decision_id"] = commits["trajectory_id"].astype(str) + "::" + commits["step_id"].astype(str)
        summary.update(
            {
                "commitments_available": True,
                "commitment_decision_count": int(commits["decision_id"].nunique()) if "decision_id" in commits.columns else None,
                "commitment_row_count": int(len(commits)),
            }
        )
    return summary


def load_token_metadata(canonical_steps_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-token IDs/text from canonical step rows when available."""
    if not canonical_steps_path.exists():
        print(f"WARNING: optional canonical steps not found: {canonical_steps_path}")
        return {}

    columns = ["trajectory_id", "step_id", "analysis_tokens_json", "analysis_token_ids_json"]
    try:
        df = pd.read_parquet(canonical_steps_path, columns=columns)
    except Exception as exc:
        print(f"WARNING: failed to read token metadata from {canonical_steps_path}: {exc}")
        return {}

    token_map: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        decision_id = make_decision_id(row["trajectory_id"], row["step_id"])
        tokens: list[dict[str, Any]] = []
        raw_tokens = row.get("analysis_tokens_json")
        if isinstance(raw_tokens, str) and raw_tokens:
            try:
                parsed = json.loads(raw_tokens)
                if isinstance(parsed, list):
                    for token in parsed:
                        if isinstance(token, dict):
                            tokens.append(
                                {
                                    "token_id": token.get("token_id"),
                                    "token_text": token.get("token"),
                                }
                            )
            except json.JSONDecodeError:
                tokens = []
        if not tokens:
            raw_ids = row.get("analysis_token_ids_json")
            if isinstance(raw_ids, str) and raw_ids:
                try:
                    parsed_ids = json.loads(raw_ids)
                    if isinstance(parsed_ids, list):
                        tokens = [{"token_id": token_id, "token_text": None} for token_id in parsed_ids]
                except json.JSONDecodeError:
                    pass
        if tokens:
            token_map[decision_id] = tokens
    return token_map


def collect_packed_bf16_records(input_dir: Path) -> tuple[dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    decisions: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    quarantine: list[dict[str, Any]] = []

    for json_path in sorted(input_dir.rglob("*.json")):
        meta = extract_sidecar_metadata(json_path)
        if not meta:
            continue
        path_meta = infer_path_metadata(json_path, input_dir)
        category = meta.get("category") or path_meta.get("token_type")
        trajectory_id = path_meta.get("trajectory_id")
        model_id = path_meta.get("model_id")
        step_id = meta.get("step_idx", path_meta.get("step_id"))
        layer = meta.get("layer_idx", path_meta.get("layer"))
        shape = meta.get("shape")
        bf16_path = json_path.with_suffix(".bf16")

        record = {
            "trajectory_id": trajectory_id,
            "model_id": model_id,
            "step_id": step_id,
            "decision_id": make_decision_id(trajectory_id, step_id) if trajectory_id is not None and step_id is not None else None,
            "token_type": category,
            "layer": layer,
            "shape": shape,
            "dtype": meta.get("dtype"),
            "format": meta.get("format"),
            "activation_file": str(bf16_path),
            "metadata_file": str(json_path),
            "relative_activation_file": str(bf16_path.relative_to(input_dir)) if bf16_path.is_relative_to(input_dir) else str(bf16_path),
            "relative_metadata_file": str(json_path.relative_to(input_dir)) if json_path.is_relative_to(input_dir) else str(json_path),
            "hidden_dim": int(shape[-1]) if shape else None,
            "num_rows": int(shape[0]) if shape else None,
            "bf16_exists": bf16_path.exists(),
        }

        missing = [key for key in ["trajectory_id", "model_id", "step_id", "token_type", "layer", "shape"] if record.get(key) is None]
        if category not in TOKEN_TYPE_SET:
            missing.append("valid_token_type")
        if not bf16_path.exists():
            missing.append("matching_bf16_file")
        if missing:
            quarantine.append({"reason": "malformed_packed_bf16_sidecar", "missing_or_invalid": missing, "record": record})
            continue
        decisions[(str(trajectory_id), str(model_id), int(step_id))][str(category)].append(record)

    return decisions, quarantine


def validate_category_records(records: list[dict[str, Any]], token_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quarantine = []
    if not records:
        return records, quarantine
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)
    valid = []
    for layer, layer_records in sorted(by_layer.items()):
        if len(layer_records) != 1:
            quarantine.append(
                {
                    "reason": "duplicate_layer_category_records",
                    "token_type": token_type,
                    "layer": layer,
                    "records": layer_records,
                }
            )
            valid.append(layer_records[0])
        else:
            valid.append(layer_records[0])
    return valid, quarantine


def category_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    layers = sorted(int(r["layer"]) for r in records)
    shapes = sorted({tuple(r["shape"]) for r in records})
    dtypes = sorted({str(r["dtype"]) for r in records})
    hidden_dims = sorted({int(r["hidden_dim"]) for r in records if r.get("hidden_dim") is not None})
    num_rows = sorted({int(r["num_rows"]) for r in records if r.get("num_rows") is not None})
    first = records[0] if records else {}
    return {
        "layers": layers,
        "shapes": [list(shape) for shape in shapes],
        "dtypes": dtypes,
        "hidden_dims": hidden_dims,
        "num_rows": num_rows,
        "representative_file": first.get("activation_file"),
        "representative_metadata_file": first.get("metadata_file"),
        "activation_file_template": file_template(Path(first["activation_file"])) if first.get("activation_file") else None,
        "metadata_file_template": file_template(Path(first["metadata_file"])) if first.get("metadata_file") else None,
    }


def build_packed_bf16_index(
    decisions: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]],
    *,
    token_metadata: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns: dict[str, list[Any]] = defaultdict(list)
    quarantine: list[dict[str, Any]] = []

    for (trajectory_id, model_id, step_id), categories in sorted(decisions.items()):
        decision_id = make_decision_id(trajectory_id, step_id)
        validated_categories: dict[str, list[dict[str, Any]]] = {}
        malformed = []

        for token_type in TOKEN_TYPES:
            records, category_quarantine = validate_category_records(categories.get(token_type, []), token_type)
            for item in category_quarantine:
                item["decision_id"] = decision_id
                quarantine.append(item)
            validated_categories[token_type] = records
            if not records:
                malformed.append(f"missing_{token_type}")

        if malformed:
            quarantine.append(
                {
                    "reason": "missing_required_token_type",
                    "decision_id": decision_id,
                    "trajectory_id": trajectory_id,
                    "step_id": step_id,
                    "missing": malformed,
                }
            )
            continue

        summaries = {token_type: category_summary(validated_categories[token_type]) for token_type in TOKEN_TYPES}
        cot_rows = summaries["cot"]["num_rows"]
        if len(cot_rows) != 1 or cot_rows[0] < 1:
            quarantine.append(
                {
                    "reason": "ambiguous_or_empty_cot_rows",
                    "decision_id": decision_id,
                    "cot_summary": summaries["cot"],
                }
            )
            continue
        num_cot_tokens = int(cot_rows[0])

        hidden_dims = sorted(
            {
                hidden_dim
                for token_type in TOKEN_TYPES
                for hidden_dim in summaries[token_type]["hidden_dims"]
            }
        )
        dtypes = sorted({dtype for token_type in TOKEN_TYPES for dtype in summaries[token_type]["dtypes"]})
        if len(hidden_dims) != 1:
            quarantine.append({"reason": "inconsistent_hidden_dim", "decision_id": decision_id, "hidden_dims": hidden_dims})
            continue

        pre_rows = summaries["pre_reasoning"]["num_rows"]
        post_rows = summaries["post_reasoning"]["num_rows"]
        pre_selected_row = int(pre_rows[0] - 1) if len(pre_rows) == 1 and pre_rows[0] else None
        post_selected_row = 0 if len(post_rows) == 1 and post_rows[0] else None
        if pre_selected_row is None or post_selected_row is None:
            quarantine.append(
                {
                    "reason": "ambiguous_pre_or_post_rows",
                    "decision_id": decision_id,
                    "pre_rows": pre_rows,
                    "post_rows": post_rows,
                }
            )
            continue

        decision_tokens = token_metadata.get(decision_id, [])
        if decision_tokens and len(decision_tokens) != num_cot_tokens:
            quarantine.append(
                {
                    "reason": "token_metadata_count_mismatch",
                    "decision_id": decision_id,
                    "activation_num_cot_tokens": num_cot_tokens,
                    "token_metadata_count": len(decision_tokens),
                }
            )

        def append_row(
            *,
            token_type: str,
            cot_token_index: int | None,
            progress: float,
            activation_row_index: int | None,
            token_id: Any = None,
            token_text: Any = None,
        ) -> None:
            summary = summaries[token_type]
            source_metadata = {
                "model_id": model_id,
                "storage_layout": "packed_raw_bf16",
                "category_summary": summary,
                "activation_file_template": summary["activation_file_template"],
                "metadata_file_template": summary["metadata_file_template"],
                "pre_post_selection_policy": "pre uses last row; post uses first row",
                "behavior_prefix_token_index": (
                    int(cot_token_index - 1)
                    if token_type == "cot" and cot_token_index is not None
                    else None
                ),
            }
            activation_id = f"{decision_id}::{token_type}::{cot_token_index if cot_token_index is not None else 'null'}"
            columns["activation_id"].append(activation_id)
            columns["decision_id"].append(decision_id)
            columns["trajectory_id"].append(trajectory_id)
            columns["step_id"].append(int(step_id))
            columns["token_type"].append(token_type)
            columns["cot_token_index"].append(cot_token_index)
            columns["num_cot_tokens"].append(num_cot_tokens)
            columns["progress"].append(float(progress))
            columns["token_text"].append(token_text)
            columns["token_id"].append(token_id)
            columns["activation_file"].append(summary["representative_file"])
            columns["activation_format"].append("packed_raw_bf16")
            columns["activation_key"].append("row")
            columns["activation_row_index"].append(activation_row_index)
            columns["token_index_in_tensor"].append(activation_row_index)
            columns["available_layers_json"].append(dumps(summary["layers"]))
            columns["hidden_dim"].append(int(hidden_dims[0]))
            columns["dtype"].append(dtypes[0] if len(dtypes) == 1 else dumps(dtypes))
            columns["source_metadata_json"].append(dumps(source_metadata))

        append_row(
            token_type="pre_reasoning",
            cot_token_index=0,
            progress=0.0,
            activation_row_index=pre_selected_row,
        )

        for row_idx in range(num_cot_tokens):
            cot_token_index = row_idx + 1
            token = decision_tokens[row_idx] if row_idx < len(decision_tokens) else {}
            append_row(
                token_type="cot",
                cot_token_index=cot_token_index,
                progress=cot_token_index / num_cot_tokens,
                activation_row_index=row_idx,
                token_id=token.get("token_id"),
                token_text=token.get("token_text"),
            )

        append_row(
            token_type="post_reasoning",
            cot_token_index=num_cot_tokens + 1,
            progress=1.0,
            activation_row_index=post_selected_row,
        )

    return pd.DataFrame(columns), quarantine


def build_generic_parquet_index(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    quarantine: list[dict[str, Any]] = []
    frames = []
    for path in sorted(input_dir.rglob("*.parquet")):
        try:
            columns = parquet_columns(path)
        except Exception as exc:
            quarantine.append({"reason": "parquet_schema_read_failed", "file": str(path), "error": str(exc)})
            continue
        required_any = {"token_type", "activation_file"}
        if not required_any.intersection(columns):
            quarantine.append({"reason": "parquet_not_activation_index_like", "file": str(path), "columns": columns})
            continue
        vector_like = {"activation", "activation_vector", "hidden_state", "vector", "tensor"}
        selected = [col for col in columns if col not in vector_like]
        try:
            frame = pd.read_parquet(path, columns=selected)
        except Exception as exc:
            quarantine.append({"reason": "parquet_index_read_failed", "file": str(path), "error": str(exc)})
            continue
        frame["source_metadata_json"] = frame.get("source_metadata_json", dumps({"source_parquet": str(path)}))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), quarantine
    return pd.concat(frames, ignore_index=True), quarantine


def validate_index(df: pd.DataFrame) -> list[dict[str, Any]]:
    quarantine = []
    if df.empty:
        quarantine.append({"reason": "empty_activation_token_index"})
        return quarantine
    for decision_id, group in df.groupby("decision_id", dropna=False):
        counts = group["token_type"].value_counts().to_dict()
        if counts.get("pre_reasoning", 0) != 1:
            quarantine.append({"reason": "invalid_pre_reasoning_count", "decision_id": decision_id, "count": counts.get("pre_reasoning", 0)})
        if counts.get("post_reasoning", 0) != 1:
            quarantine.append({"reason": "invalid_post_reasoning_count", "decision_id": decision_id, "count": counts.get("post_reasoning", 0)})
        if counts.get("cot", 0) < 1:
            quarantine.append({"reason": "missing_cot_rows", "decision_id": decision_id})
        cot = group[group["token_type"].eq("cot")].sort_values("cot_token_index")
        if not cot.empty:
            values = cot["cot_token_index"].astype(int).to_numpy()
            if not np.array_equal(values, np.arange(1, len(values) + 1)):
                quarantine.append(
                    {
                        "reason": "non_monotonic_or_non_contiguous_cot_token_index",
                        "decision_id": decision_id,
                        "first_values": values[:10].tolist(),
                        "last_values": values[-10:].tolist(),
                    }
                )
    return quarantine


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_layers(series: pd.Series) -> list[int]:
    layers = set()
    for value in series.dropna().unique():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                layers.update(int(x) for x in parsed)
        except Exception:
            pass
    return sorted(layers)


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    quarantine: list[dict[str, Any]],
    input_dir: Path,
    behavior_summary: dict[str, Any],
) -> None:
    counts = df["token_type"].value_counts().to_dict() if not df.empty else {}
    layers = parse_layers(df["available_layers_json"]) if "available_layers_json" in df.columns else []
    hidden_dims = sorted(int(x) for x in df["hidden_dim"].dropna().unique()) if "hidden_dim" in df.columns and not df.empty else []
    dtype_counts = df["dtype"].value_counts().to_dict() if "dtype" in df.columns and not df.empty else {}
    examples = (
        df.groupby(["decision_id", "trajectory_id", "step_id"], dropna=False)
        .agg(
            rows=("activation_id", "size"),
            cot_rows=("token_type", lambda x: int((x == "cot").sum())),
            num_cot_tokens=("num_cot_tokens", "first"),
        )
        .reset_index()
        .head(5)
        if not df.empty
        else pd.DataFrame()
    )
    invalid_pre = [q for q in quarantine if q.get("reason") in {"invalid_pre_reasoning_count", "invalid_post_reasoning_count"}]

    lines = [
        "# Activation Token Index Report",
        "",
        "## Inputs",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Behavior prefixes available: {behavior_summary['prefixes_available']}",
        f"- Commitment metrics available: {behavior_summary['commitments_available']}",
        f"- Behavior prefix rows / decisions: {behavior_summary['prefix_row_count']} / {behavior_summary['prefix_decision_count']}",
        f"- Commitment rows / decisions: {behavior_summary['commitment_row_count']} / {behavior_summary['commitment_decision_count']}",
        "",
        "## Index Summary",
        "",
        f"- Activation timepoints: {len(df):,}",
        f"- Decision steps: {df['decision_id'].nunique() if not df.empty else 0:,}",
        f"- Trajectory IDs: {df['trajectory_id'].nunique() if not df.empty else 0:,}",
        f"- `pre_reasoning` rows: {counts.get('pre_reasoning', 0):,}",
        f"- `cot` rows: {counts.get('cot', 0):,}",
        f"- `post_reasoning` rows: {counts.get('post_reasoning', 0):,}",
        f"- Available layers: {layers}",
        f"- Hidden dimensions: {hidden_dims}",
        f"- Dtype counts: {dtype_counts}",
        f"- Quarantined records: {len(quarantine):,}",
        "",
        "## Example Decisions",
        "",
    ]
    if examples.empty:
        lines.append("- No decisions indexed.")
    else:
        for _, row in examples.iterrows():
            lines.append(
                f"- `{row['decision_id']}`: rows={int(row['rows'])}, "
                f"cot_rows={int(row['cot_rows'])}, num_cot_tokens={int(row['num_cot_tokens'])}"
            )

    lines.extend(["", "## Missing Or Duplicated Pre/Post Activations", ""])
    if not invalid_pre:
        lines.append("- None detected in the canonical index.")
    else:
        for item in invalid_pre[:20]:
            lines.append(f"- `{item.get('decision_id')}`: {item.get('reason')} count={item.get('count')}")

    lines.extend(["", "## Notes", ""])
    lines.append("- Large activation vectors are not stored in this parquet; rows store file paths, row indices, layers, and metadata pointers.")
    lines.append("- For packed BF16 pre/post records with multiple rows, the index uses the last pre row and first post row as representative timepoints.")
    lines.append("- CoT rows use 1-based `cot_token_index`; the corresponding tensor row is `activation_row_index = cot_token_index - 1`.")
    lines.append("- `source_metadata_json` includes file templates for reconstructing layer-specific paths.")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/activations")
    parser.add_argument("--output", default="outputs/activation/activation_token_index.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--commitments", default="outputs/prefix_commitment_metrics.parquet")
    parser.add_argument("--canonical_steps", default="outputs/canonical_steps.parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.parent / "activation_token_index_report.md"
    quarantine_path = output_path.parent / "quarantined_activation_records.jsonl"

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    print(f"Scanning activation metadata under {input_dir}")
    token_metadata = load_token_metadata(Path(args.canonical_steps))
    behavior_summary = build_behavior_summary(Path(args.prefixes), Path(args.commitments))

    packed_records, quarantine = collect_packed_bf16_records(input_dir)
    if packed_records:
        print(f"Detected packed raw BF16 layout with {len(packed_records):,} decision/layer groups.")
        index_df, build_quarantine = build_packed_bf16_index(packed_records, token_metadata=token_metadata)
        quarantine.extend(build_quarantine)
    else:
        print("No packed raw BF16 sidecars detected; trying generic parquet index layout.")
        index_df, build_quarantine = build_generic_parquet_index(input_dir)
        quarantine.extend(build_quarantine)

    validation_quarantine = validate_index(index_df)
    quarantine.extend(validation_quarantine)

    canonical_columns = [
        "activation_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "token_type",
        "cot_token_index",
        "num_cot_tokens",
        "progress",
        "token_text",
        "token_id",
        "activation_file",
        "activation_format",
        "activation_key",
        "activation_row_index",
        "token_index_in_tensor",
        "available_layers_json",
        "hidden_dim",
        "dtype",
        "source_metadata_json",
    ]
    for column in canonical_columns:
        if column not in index_df.columns:
            index_df[column] = pd.NA
    index_df = index_df[canonical_columns]
    index_df.to_parquet(output_path, index=False)
    write_jsonl(quarantine_path, quarantine)
    write_report(
        report_path,
        df=index_df,
        quarantine=quarantine,
        input_dir=input_dir,
        behavior_summary=behavior_summary,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {quarantine_path}")
    print(f"Indexed {len(index_df):,} activation timepoints across {index_df['decision_id'].nunique() if not index_df.empty else 0:,} decisions.")
    if quarantine:
        print(f"Recorded {len(quarantine):,} quarantined/warning records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
