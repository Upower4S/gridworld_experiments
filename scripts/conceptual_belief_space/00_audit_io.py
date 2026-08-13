#!/usr/bin/env python3
"""Repository IO audit for conceptual belief-space analysis.

This script only inspects existing files and writes audit artifacts. It does not
run PCA, model inference, activation extraction, or any expensive analysis.
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
Q_COLS = [f"q_{a}" for a in ACTIONS]
LOGSCORE_COLS = [f"logscore_{a}" for a in ACTIONS]
B_CLR_COLS = [f"b_clr_{a}" for a in ACTIONS]
CORE_METADATA_COLS = [
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
]
REQUIRED_CONCEPTUAL_COLUMNS = [*CORE_METADATA_COLS, *Q_COLS]
STRICT_PATTERNS = [
    "strict",
    "first_action_mention",
    "first_final_action_mention",
    "is_before_first_action_mention",
    "is_before_first_final_action_mention",
]
TABULAR_SUFFIXES = {".parquet", ".csv"}
SEARCH_ROOTS = ["outputs", "data", "figures", "scripts", "notebooks"]


def repo_root_from(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "scripts").exists():
            return candidate
    return Path.cwd().resolve()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def schema_columns(path: Path) -> tuple[list[str], int | None, str | None]:
    """Return column names, row count if cheap, and an optional error string."""
    try:
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                pf = pq.ParquetFile(path)
                rows = pf.metadata.num_rows if pf.metadata is not None else None
                return list(pf.schema_arrow.names), rows, None
            except Exception:
                df = pd.read_parquet(path)
                return list(df.columns), int(len(df)), None
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=0)
            return list(df.columns), None, None
        return [], None, f"unsupported suffix {path.suffix}"
    except Exception as exc:  # keep audit robust
        return [], None, f"{type(exc).__name__}: {exc}"


def find_tabular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in SEARCH_ROOTS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
                files.append(path)
    return sorted(files)


def behavior_candidate_score(path: Path, columns: set[str]) -> int:
    lower = str(path).lower()
    score = 0
    score += 10 * len(set(Q_COLS) & columns)
    score += 5 * len(set(LOGSCORE_COLS) & columns)
    score += 4 * len(set(B_CLR_COLS) & columns)
    score += 3 * len(set(CORE_METADATA_COLS) & columns)
    if "prefix_token_index" in columns:
        score += 10
    if "progress" in columns:
        score += 8
    if "commitment_to_final" in columns:
        score += 8
    if "behavior_vectors" in lower:
        score += 20
    elif "behavior" in lower:
        score += 8
    if "commitment" in lower:
        score += 6
    if "prefix" in lower:
        score += 5
    if "score" in lower:
        score += 4
    # Do not use the old PCA manifold as the primary behavior belief source.
    if "manifold" in lower:
        score -= 12
    # Prefer compact belief-vector/scored-prefix tables over large activation metadata.
    if "activation" in lower:
        score -= 15
    if "diagnostics" in lower or "followups" in lower:
        score -= 8
    return score


def find_activation_metadata(root: Path, table_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for info in table_infos:
        cols = set(info["columns"])
        path = info["path"]
        lower = path.lower()
        feature_cols = [c for c in cols if c.startswith("feature_file_layer_")]
        score = 0
        score += 5 * len(feature_cols)
        score += 5 * len(set(REQUIRED_CONCEPTUAL_COLUMNS) & cols)
        if "activation_feature_metadata" in lower:
            score += 100
        elif "activation" in lower and "metadata" in lower:
            score += 40
        if score > 0:
            candidates.append((score, info))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def parse_feature_dir(path: Path) -> tuple[list[int], list[str], list[str]]:
    pattern = re.compile(r"^layer_(\d+)_(.+)\.npy$")
    layers: set[int] = set()
    reps: set[str] = set()
    files: list[str] = []
    if not path.exists():
        return [], [], []
    for file in sorted(path.glob("*.npy")):
        match = pattern.match(file.name)
        if not match:
            continue
        layers.add(int(match.group(1)))
        reps.add(match.group(2))
        files.append(file.name)
    return sorted(layers), sorted(reps), files


def find_feature_dirs(root: Path) -> list[dict[str, Any]]:
    dirs: dict[Path, dict[str, Any]] = {}
    for base_name in SEARCH_ROOTS:
        base = root / base_name
        if not base.exists():
            continue
        for npy in base.rglob("layer_*.npy"):
            layers, reps, files = parse_feature_dir(npy.parent)
            if files:
                dirs[npy.parent] = {"path": npy.parent, "layers": layers, "representations": reps, "files": files}
    return sorted(dirs.values(), key=lambda d: (len(d["files"]), len(d["layers"])), reverse=True)


def strict_columns(table_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in table_infos:
        matched = []
        for col in info["columns"]:
            low = col.lower()
            if any(pattern in low for pattern in STRICT_PATTERNS):
                matched.append(col)
        if matched:
            rows.append({"path": info["path"], "columns": sorted(matched)})
    return rows


def column_locations(columns: list[str], selected_infos: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        paths = [info["path"] for info in selected_infos if col in set(info["columns"])]
        out[col] = {"found": bool(paths), "paths": paths}
    return out


def write_markdown(path: Path, io_map: dict[str, Any]) -> None:
    lines = [
        "# Conceptual Belief Space IO Audit",
        "",
        "This audit locates existing repository inputs for the next paper-style conceptual belief space pipeline. It does not run PCA, probes, activation extraction, or expensive analyses.",
        "",
        "## Recommended Inputs",
        "",
        f"- Behavior belief/action table: `{io_map.get('behavior_table_path')}`",
        f"- Activation feature metadata: `{io_map.get('activation_metadata_path')}`",
        f"- Activation feature directory: `{io_map.get('activation_feature_dir')}`",
        f"- Recommended new output directory: `{io_map.get('recommended_output_dir')}`",
        "",
        "## Available Activation Features",
        "",
        f"- Layers: `{io_map.get('available_layers')}`",
        f"- Representations: `{io_map.get('available_representations')}`",
        "",
        "## Required Columns",
        "",
    ]
    found = io_map.get("required_columns_found", {})
    for col in REQUIRED_CONCEPTUAL_COLUMNS:
        info = found.get(col, {})
        status = "found" if info.get("found") else "missing"
        paths = ", ".join(f"`{p}`" for p in info.get("paths", [])) or "-"
        lines.append(f"- `{col}`: {status}; paths: {paths}")
    lines.extend(["", "## Missing Columns", ""])
    missing = io_map.get("missing_columns", [])
    if missing:
        lines.extend(f"- `{col}`" for col in missing)
    else:
        lines.append("- None across the selected behavior and activation metadata inputs.")
    lines.extend(["", "## Existing Strict Pre-action Columns", ""])
    strict = io_map.get("existing_strict_columns", [])
    if strict:
        for item in strict:
            cols = ", ".join(f"`{c}`" for c in item["columns"])
            lines.append(f"- `{item['path']}`: {cols}")
    else:
        lines.append("- No strict pre-action columns found in scanned tabular files.")
    lines.extend(["", "## Candidate Tables Considered", ""])
    for candidate in io_map.get("candidate_behavior_tables", [])[:10]:
        lines.append(
            f"- `{candidate['path']}`: score={candidate['score']}, rows={candidate.get('rows')}, "
            f"q_cols={candidate['q_columns_found']}, logscore_cols={candidate['logscore_columns_found']}, "
            f"b_clr_cols={candidate['b_clr_columns_found']}"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "Use the behavior table for action belief vectors over UP/DOWN/LEFT/RIGHT and the activation metadata/feature directory for aligned activation rows. The next script should create grouped train/test splits by `trajectory_id`, construct or verify `decision_id`, select action-extreme examples from the training split only, and write new outputs only under `outputs/conceptual_belief_space/`.",
            "",
            "For strict pre-action analyses, use existing boolean columns such as `is_before_first_action_mention` when exact mention indices are not needed. If exact `< first_action_mention_token_index` boundaries are needed, consult tables listed above that contain `first_action_mention_token_index`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_audit(root: Path) -> dict[str, Any]:
    tabular_files = find_tabular_files(root)
    table_infos: list[dict[str, Any]] = []
    for path in tabular_files:
        cols, rows, error = schema_columns(path)
        table_infos.append(
            {
                "path": rel(path, root),
                "suffix": path.suffix.lower(),
                "columns": cols,
                "rows": rows,
                "error": error,
            }
        )

    behavior_candidates = []
    for info in table_infos:
        cols = set(info["columns"])
        score = behavior_candidate_score(root / info["path"], cols)
        if score <= 0:
            continue
        behavior_candidates.append(
            {
                "path": info["path"],
                "score": score,
                "rows": info.get("rows"),
                "q_columns_found": sorted(set(Q_COLS) & cols),
                "logscore_columns_found": sorted(set(LOGSCORE_COLS) & cols),
                "b_clr_columns_found": sorted(set(B_CLR_COLS) & cols),
                "metadata_columns_found": sorted(set(CORE_METADATA_COLS) & cols),
            }
        )
    behavior_candidates = sorted(behavior_candidates, key=lambda x: x["score"], reverse=True)
    behavior_path = behavior_candidates[0]["path"] if behavior_candidates else None
    behavior_info = next((info for info in table_infos if info["path"] == behavior_path), None)

    activation_metadata_info = find_activation_metadata(root, table_infos)
    activation_metadata_path = activation_metadata_info["path"] if activation_metadata_info else None

    feature_dirs = find_feature_dirs(root)
    selected_feature_dir = feature_dirs[0] if feature_dirs else None
    feature_dir_path = rel(selected_feature_dir["path"], root) if selected_feature_dir else None
    available_layers = selected_feature_dir["layers"] if selected_feature_dir else []
    available_representations = selected_feature_dir["representations"] if selected_feature_dir else []

    selected_infos = [info for info in [behavior_info, activation_metadata_info] if info is not None]
    required_found = column_locations(REQUIRED_CONCEPTUAL_COLUMNS, selected_infos)
    missing_columns = [col for col, info in required_found.items() if not info["found"]]

    # Add useful derived notes without pretending the raw behavior table has decision_id.
    derived_columns: dict[str, str] = {}
    if behavior_info is not None:
        behavior_cols = set(behavior_info["columns"])
        if "decision_id" not in behavior_cols and {"trajectory_id", "step_id"}.issubset(behavior_cols):
            derived_columns["decision_id"] = "construct from trajectory_id and step_id in behavior table"

    out_dir = root / "outputs" / "conceptual_belief_space"
    io_map: dict[str, Any] = {
        "behavior_table_path": behavior_path,
        "activation_metadata_path": activation_metadata_path,
        "activation_feature_dir": feature_dir_path,
        "available_layers": available_layers,
        "available_representations": available_representations,
        "required_columns_found": required_found,
        "missing_columns": missing_columns,
        "existing_strict_columns": strict_columns(table_infos),
        "recommended_output_dir": rel(out_dir, root),
        "candidate_behavior_tables": behavior_candidates,
        "candidate_activation_feature_dirs": [
            {
                "path": rel(item["path"], root),
                "available_layers": item["layers"],
                "available_representations": item["representations"],
                "num_feature_files": len(item["files"]),
            }
            for item in feature_dirs
        ],
        "derived_columns_available": derived_columns,
        "search_roots": [name for name in SEARCH_ROOTS if (root / name).exists()],
        "notes": [
            "Use grouped train/test splits by trajectory_id for next conceptual belief-space analyses.",
            "Select action-extreme examples from train split only to avoid leakage.",
            "This audit does not run PCA, probes, nearest-centroid analysis, or activation extraction.",
        ],
    }
    return io_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space")
    args = parser.parse_args()

    root = repo_root_from(Path(args.repo_root))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    io_map = build_audit(root)
    io_map["recommended_output_dir"] = rel(output_dir, root)

    json_path = output_dir / "io_map.json"
    md_path = output_dir / "00_io_audit.md"
    json_path.write_text(json.dumps(io_map, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, io_map)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if io_map["missing_columns"]:
        print("Missing columns:", ", ".join(io_map["missing_columns"]))
    else:
        print("All required columns found across selected behavior and activation metadata inputs.")


if __name__ == "__main__":
    main()
