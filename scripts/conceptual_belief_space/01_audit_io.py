#!/usr/bin/env python3
"""Step 1: IO audit for conceptual belief space analysis.

This script locates existing behavior-side action belief tables and
activation-side feature metadata. It only inspects schemas and feature-array
metadata; it does not run analyses or load large activation arrays into memory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
Q_COLS = [f"q_{action}" for action in ACTIONS]
LOGSCORE_COLS = [f"logscore_{action}" for action in ACTIONS]
B_CLR_COLS = [f"b_clr_{action}" for action in ACTIONS]
METADATA_COLS = [
    "decision_id",
    "trajectory_id",
    "step_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    "top_action",
    "commitment_to_final",
    "entropy_commitment",
]
STRICT_COLS = [
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
    "first_action_mention_token_index",
    "first_final_action_mention_token_index",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
]
SEARCH_ROOTS = ["outputs", "data", "figures", "scripts", "notebooks"]
TABULAR_SUFFIXES = {".parquet", ".csv"}
FEATURE_RE = re.compile(r"^layer_(\d+)_(.+)\.npy$")


def repo_root_from(start: Path) -> Path:
    """Find the repository root from a starting path."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "scripts").exists():
            return candidate
    return Path.cwd().resolve()


def rel(path: Path, root: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_tabular_schema(path: Path) -> tuple[list[str], int | None, str | None]:
    """Read columns and row count cheaply where possible."""
    try:
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                parquet_file = pq.ParquetFile(path)
                rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
                return list(parquet_file.schema_arrow.names), rows, None
            except Exception:
                df = pd.read_parquet(path)
                return list(df.columns), int(len(df)), None
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=0)
            return list(df.columns), None, None
        return [], None, f"unsupported suffix: {path.suffix}"
    except Exception as exc:
        return [], None, f"{type(exc).__name__}: {exc}"


def find_tabular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SEARCH_ROOTS:
        search_root = root / root_name
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
                files.append(path)
    return sorted(files)


def table_infos(root: Path) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for path in find_tabular_files(root):
        columns, rows, error = read_tabular_schema(path)
        infos.append(
            {
                "path": rel(path, root),
                "suffix": path.suffix.lower(),
                "columns": columns,
                "rows": rows,
                "error": error,
            }
        )
    return infos


def behavior_score(path: str, columns: set[str]) -> int:
    """Score whether a table is a prefix-level behavior/action table."""
    lower = path.lower()
    n_q = len(set(Q_COLS) & columns)
    n_logscore = len(set(LOGSCORE_COLS) & columns)
    n_metadata = len(set(METADATA_COLS) & columns)
    score = 0
    score += 30 * n_q
    score += 18 * n_logscore
    score += 8 * len(set(B_CLR_COLS) & columns)
    score += 6 * n_metadata
    if n_q == len(Q_COLS):
        score += 80
    if n_logscore == len(LOGSCORE_COLS):
        score += 35
    if {"trajectory_id", "step_id", "prefix_token_index"}.issubset(columns):
        score += 45
    if {"commitment_to_final", "entropy_commitment"}.issubset(columns):
        score += 35
    if "prefix_commitment_metrics" in lower:
        score += 80
    if "prefix_action_scores" in lower:
        score += 55
    if "behavior_vectors" in lower:
        score += 45
    if "behavior_manifold" in lower:
        score += 15
    if "activation" in lower:
        score -= 120
    if "diagnostics" in lower or "followups" in lower:
        score -= 25
    if "manifold" in lower:
        score -= 20
    return score


def activation_metadata_score(path: str, columns: set[str]) -> int:
    lower = path.lower()
    score = 0
    if "activation_feature_metadata" in lower:
        score += 200
    if "activation" in lower and "metadata" in lower:
        score += 80
    if "feature_file" in " ".join(columns):
        score += 40
    score += 8 * len(set(METADATA_COLS) & columns)
    score += 12 * len(set(Q_COLS) & columns)
    if {"decision_id", "prefix_token_index", "progress"}.issubset(columns):
        score += 30
    if "activation" not in lower:
        score -= 60
    return score


def select_behavior_table(infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for info in infos:
        columns = set(info["columns"])
        score = behavior_score(info["path"], columns)
        has_action_scores = bool(set(Q_COLS) & columns) or bool(set(LOGSCORE_COLS) & columns)
        has_prefix_metadata = "prefix_token_index" in columns and "progress" in columns
        if score > 0 and has_action_scores and has_prefix_metadata:
            candidates.append((score, info))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def select_activation_metadata(infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for info in infos:
        columns = set(info["columns"])
        score = activation_metadata_score(info["path"], columns)
        if score > 0 and ("activation" in info["path"].lower() or "feature_file" in " ".join(columns)):
            candidates.append((score, info))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def behavior_columns_found(behavior_info: dict[str, Any]) -> dict[str, bool]:
    columns = set(behavior_info["columns"])
    found: dict[str, bool] = {}
    for col in METADATA_COLS + Q_COLS + LOGSCORE_COLS:
        found[col] = col in columns
    for action in ACTIONS:
        found[f"action_score_available_{action}"] = (
            f"q_{action}" in columns or f"logscore_{action}" in columns
        )
    found["decision_id_available_or_derivable"] = (
        "decision_id" in columns or {"trajectory_id", "step_id"}.issubset(columns)
    )
    return found


def missing_behavior_columns(behavior_info: dict[str, Any]) -> list[str]:
    columns = set(behavior_info["columns"])
    missing: list[str] = []
    for col in METADATA_COLS:
        if col == "decision_id" and {"trajectory_id", "step_id"}.issubset(columns):
            continue
        if col not in columns:
            missing.append(col)
    for action in ACTIONS:
        if f"q_{action}" not in columns and f"logscore_{action}" not in columns:
            missing.append(f"q_{action} or logscore_{action}")
    return missing


def strict_columns_found(infos: list[dict[str, Any]]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {col: [] for col in STRICT_COLS}
    for info in infos:
        columns = set(info["columns"])
        for col in STRICT_COLS:
            if col in columns:
                found[col].append(info["path"])
    return found


def inspect_npy_feature(path: Path, root: Path) -> dict[str, Any]:
    match = FEATURE_RE.match(path.name)
    layer = int(match.group(1)) if match else None
    representation = match.group(2) if match else None
    info: dict[str, Any] = {
        "path": rel(path, root),
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "layer": layer,
        "representation": representation,
        "shape": None,
        "dtype": None,
        "error": None,
    }
    try:
        arr = np.load(path, mmap_mode="r")
        info["shape"] = [int(x) for x in arr.shape]
        info["dtype"] = str(arr.dtype)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def find_activation_feature_dirs(root: Path) -> list[dict[str, Any]]:
    dirs: dict[Path, list[Path]] = {}
    for root_name in SEARCH_ROOTS:
        search_root = root / root_name
        if not search_root.exists():
            continue
        for path in search_root.rglob("layer_*.npy"):
            if path.is_file() and FEATURE_RE.match(path.name):
                dirs.setdefault(path.parent, []).append(path)

    results: list[dict[str, Any]] = []
    for directory, files in dirs.items():
        feature_infos = [inspect_npy_feature(path, root) for path in sorted(files)]
        layers = sorted({info["layer"] for info in feature_infos if info["layer"] is not None})
        reps = sorted(
            {info["representation"] for info in feature_infos if info["representation"] is not None}
        )
        row_counts = sorted(
            {
                info["shape"][0]
                for info in feature_infos
                if isinstance(info.get("shape"), list) and len(info["shape"]) >= 1
            }
        )
        hidden_dims = sorted(
            {
                info["shape"][1]
                for info in feature_infos
                if isinstance(info.get("shape"), list) and len(info["shape"]) >= 2
            }
        )
        score = len(feature_infos) * 10 + len(layers) * 5 + len(reps) * 5
        if rel(directory, root) == "outputs/activation/features":
            score += 500
        results.append(
            {
                "path": rel(directory, root),
                "score": score,
                "available_layers": layers,
                "available_representations": reps,
                "row_counts": row_counts,
                "hidden_dims": hidden_dims,
                "feature_files": feature_infos,
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)


def candidate_summaries(infos: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in infos:
        columns = set(info["columns"])
        if kind == "behavior":
            score = behavior_score(info["path"], columns)
        else:
            score = activation_metadata_score(info["path"], columns)
        if score <= 0:
            continue
        rows.append(
            {
                "path": info["path"],
                "score": score,
                "rows": info["rows"],
                "error": info["error"],
                "q_columns_found": sorted(set(Q_COLS) & columns),
                "logscore_columns_found": sorted(set(LOGSCORE_COLS) & columns),
                "metadata_columns_found": sorted(set(METADATA_COLS) & columns),
                "strict_columns_found": sorted(set(STRICT_COLS) & columns),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def build_io_map(root: Path) -> dict[str, Any]:
    infos = table_infos(root)
    behavior_info = select_behavior_table(infos)
    activation_info = select_activation_metadata(infos)
    if behavior_info is None:
        raise RuntimeError(
            "No prefix-level behavior/action scoring table found. Expected q_* or "
            "logscore_* action columns plus prefix_token_index/progress metadata."
        )
    if activation_info is None:
        raise RuntimeError(
            "No activation feature metadata table found. Expected something like "
            "outputs/activation/activation_feature_metadata.parquet."
        )

    feature_dirs = find_activation_feature_dirs(root)
    selected_feature_dir = feature_dirs[0] if feature_dirs else None
    behavior_rows = behavior_info.get("rows")
    activation_rows = activation_info.get("rows")
    row_counts_match = (
        behavior_rows is not None and activation_rows is not None and int(behavior_rows) == int(activation_rows)
    )

    io_map: dict[str, Any] = {
        "behavior_table_path": behavior_info["path"],
        "activation_metadata_path": activation_info["path"],
        "activation_feature_dir": selected_feature_dir["path"] if selected_feature_dir else None,
        "available_layers": selected_feature_dir["available_layers"] if selected_feature_dir else [],
        "available_representations": (
            selected_feature_dir["available_representations"] if selected_feature_dir else []
        ),
        "required_behavior_columns_found": behavior_columns_found(behavior_info),
        "missing_behavior_columns": missing_behavior_columns(behavior_info),
        "strict_columns_found": strict_columns_found(infos),
        "recommended_output_dir": "outputs/conceptual_belief_space",
        "recommended_figure_dir": "figures/conceptual_belief_space",
        "row_count_check": {
            "behavior_rows": behavior_rows,
            "activation_metadata_rows": activation_rows,
            "match": row_counts_match,
        },
        "available_activation_feature_files": selected_feature_dir["feature_files"]
        if selected_feature_dir
        else [],
        "candidate_behavior_tables": candidate_summaries(infos, "behavior")[:20],
        "candidate_activation_metadata_tables": candidate_summaries(infos, "activation")[:20],
        "candidate_activation_feature_dirs": feature_dirs,
    }
    return io_map


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def write_markdown(path: Path, io_map: dict[str, Any]) -> None:
    row_check = io_map["row_count_check"]
    lines = [
        "# Conceptual Belief Space IO Audit",
        "",
        "This Step 1 audit locates the existing files needed for conceptual belief space analysis. It inspects schemas and activation feature array metadata only.",
        "",
        "## Found Inputs",
        "",
        f"- Behavior/action table: `{io_map['behavior_table_path']}`",
        f"- Activation feature metadata: `{io_map['activation_metadata_path']}`",
        f"- Activation feature directory: `{io_map['activation_feature_dir']}`",
        f"- Recommended output directory: `{io_map['recommended_output_dir']}`",
        f"- Recommended figure directory: `{io_map['recommended_figure_dir']}`",
        "",
        "## Behavior Columns",
        "",
    ]
    for col, found in io_map["required_behavior_columns_found"].items():
        lines.append(f"- `{col}`: {yes_no(found)}")

    lines.extend(["", "## Missing Behavior Columns", ""])
    if io_map["missing_behavior_columns"]:
        lines.extend(f"- `{col}`" for col in io_map["missing_behavior_columns"])
    else:
        lines.append("- None. `decision_id` may be constructed from `trajectory_id` and `step_id` if absent.")

    lines.extend(
        [
            "",
            "## Strict Pre-action Columns",
            "",
        ]
    )
    for col, paths in io_map["strict_columns_found"].items():
        if paths:
            unique_paths = sorted(set(paths))
            lines.append(f"- `{col}`: found in {', '.join(f'`{p}`' for p in unique_paths[:6])}")
        else:
            lines.append(f"- `{col}`: not found")

    lines.extend(
        [
            "",
            "## Activation Features",
            "",
            f"- Available layers: `{io_map['available_layers']}`",
            f"- Available representations: `{io_map['available_representations']}`",
            f"- Number of feature files inspected: `{len(io_map['available_activation_feature_files'])}`",
        ]
    )
    if io_map["available_activation_feature_files"]:
        lines.append("")
        lines.append("Example feature files:")
        for item in io_map["available_activation_feature_files"][:12]:
            lines.append(
                f"- `{item['path']}`: shape={item['shape']}, dtype={item['dtype']}, size_bytes={item['size_bytes']}"
            )

    lines.extend(
        [
            "",
            "## Row Count Check",
            "",
            f"- Behavior rows: `{row_check['behavior_rows']}`",
            f"- Activation metadata rows: `{row_check['activation_metadata_rows']}`",
            f"- Row counts match: `{yes_no(row_check['match'])}`",
            "",
            "## Candidate Behavior Tables",
            "",
        ]
    )
    for candidate in io_map["candidate_behavior_tables"][:10]:
        lines.append(
            f"- `{candidate['path']}`: score={candidate['score']}, rows={candidate['rows']}, "
            f"q_cols={candidate['q_columns_found']}, logscore_cols={candidate['logscore_columns_found']}"
        )

    lines.extend(["", "## Candidate Activation Metadata Tables", ""])
    for candidate in io_map["candidate_activation_metadata_tables"][:10]:
        lines.append(
            f"- `{candidate['path']}`: score={candidate['score']}, rows={candidate['rows']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=".", help="Repository root or any path inside it.")
    parser.add_argument(
        "--output_dir",
        default="outputs/conceptual_belief_space",
        help="Directory for io_map.json and 01_io_audit.md.",
    )
    args = parser.parse_args()

    root = repo_root_from(Path(args.repo_root))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    try:
        io_map = build_io_map(root)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    json_path = output_dir / "io_map.json"
    md_path = output_dir / "01_io_audit.md"
    write_json(json_path, io_map)
    write_markdown(md_path, io_map)

    print(f"Wrote {rel(json_path, root)}")
    print(f"Wrote {rel(md_path, root)}")
    print(f"Behavior table: {io_map['behavior_table_path']}")
    print(f"Activation metadata: {io_map['activation_metadata_path']}")
    print(f"Activation feature dir: {io_map['activation_feature_dir']}")
    print(f"Row counts match: {yes_no(io_map['row_count_check']['match'])}")


if __name__ == "__main__":
    main()
