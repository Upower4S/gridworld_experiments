#!/usr/bin/env python3
"""Prepare a canonical step-level table from nested key-door trajectory JSON files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VALID_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
END_MARKER = "<|end|>"

LONG_FIELD_MAX_CHARS = 240
EXAMPLE_ROW_COUNT = 3


def require_dependencies() -> tuple[Any, Any]:
    """Import runtime dependencies with a clear error for missing parquet support."""
    try:
        import pandas as pd
        import pyarrow as pa
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency: {missing}. Install pandas and pyarrow before running this script."
        ) from exc
    return pd, pa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing trajectory JSON files. Files are discovered recursively.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for canonical_steps.parquet, data_schema_report.md, and quarantined_steps.jsonl.",
    )
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except Exception as exc:  # noqa: BLE001 - report and quarantine malformed files.
        return None, f"{type(exc).__name__}: {exc}"

    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def stable_trajectory_id(payload: dict[str, Any], path: Path) -> str:
    candidates: list[Any] = [
        payload.get("trajectory_id"),
        payload.get("stable_id"),
        payload.get("id"),
        payload.get("trajectory_metadata", {}).get("trajectory_id")
        if isinstance(payload.get("trajectory_metadata"), dict)
        else None,
        payload.get("trajectory_metadata", {}).get("id")
        if isinstance(payload.get("trajectory_metadata"), dict)
        else None,
        payload.get("metadata", {}).get("trajectory_id") if isinstance(payload.get("metadata"), dict) else None,
        payload.get("metadata", {}).get("id") if isinstance(payload.get("metadata"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, int):
            return str(candidate)
    return path.stem


def normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in VALID_ACTIONS else None


def normalize_action_from_token_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = (
        value.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("\\n", "\n")
        .replace('"', " ")
        .replace("'", " ")
        .strip()
        .upper()
    )
    for action in VALID_ACTIONS:
        if re.search(rf"\b{action}\b", text):
            return action
    return None


def token_groups(token: dict[str, Any]) -> set[str]:
    groups = token.get("token_groups", [])
    if not isinstance(groups, list):
        return set()
    return {str(group) for group in groups}


def extract_analysis_tokens(output_tokens: Any) -> list[dict[str, Any]]:
    if not isinstance(output_tokens, list):
        return []
    return [
        token
        for token in output_tokens
        if isinstance(token, dict) and "analysis" in token_groups(token)
    ]


def extract_final_action_from_output(output_tokens: Any) -> str | None:
    if not isinstance(output_tokens, list):
        return None

    final_action_tokens = [
        token
        for token in output_tokens
        if isinstance(token, dict) and {"final", "action"}.issubset(token_groups(token))
    ]

    for token in final_action_tokens:
        action = normalize_action_from_token_text(token.get("token"))
        if action is not None:
            return action

    joined_tokens = "".join(str(token.get("token", "")) for token in final_action_tokens)
    action = normalize_action_from_token_text(joined_tokens)
    if action is not None:
        return action

    for token in final_action_tokens:
        probabilities = token.get("probabilities")
        if not isinstance(probabilities, dict):
            continue
        for candidate in probabilities:
            action = normalize_action_from_token_text(candidate)
            if action is not None:
                return action

    return None


def detokenize_fallback(tokens: list[dict[str, Any]]) -> str:
    pieces = [str(token.get("token", "")) for token in tokens]
    text = "".join(pieces)
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("<|end|>", "")
        .replace("<|return|>", "")
        .strip()
    )


def parse_analysis_text(output_text: Any, analysis_tokens: list[dict[str, Any]]) -> str:
    if isinstance(output_text, str):
        start = output_text.find(ANALYSIS_MARKER)
        if start >= 0:
            content_start = start + len(ANALYSIS_MARKER)
            end = output_text.find(END_MARKER, content_start)
            if end >= 0:
                parsed = output_text[content_start:end].strip()
            else:
                parsed = output_text[content_start:].strip()
            if parsed:
                return parsed

    return detokenize_fallback(analysis_tokens)


def prompt_template(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        return ""
    template = prompt.get("prompt_template")
    return template if isinstance(template, str) else ""


def build_prompt_text(template: str, grid_state_text: str, carrying_key: Any) -> str:
    carrying_key_text = str(bool(carrying_key)) if isinstance(carrying_key, bool) else str(carrying_key)
    return template.replace("{{grid_state}}", grid_state_text).replace(
        "{{carrying_key}}",
        carrying_key_text,
    )


def truncate_value(value: Any, max_chars: int = LONG_FIELD_MAX_CHARS) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    return value


def truncate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: truncate_value(value) for key, value in row.items()}


def output_token_distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    ordered = sorted(values)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(25),
        "median": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def format_distribution(distribution: dict[str, float | int | None]) -> str:
    lines = []
    for key in ("count", "min", "p25", "median", "p75", "p90", "max", "mean"):
        value = distribution.get(key)
        if isinstance(value, float):
            rendered = f"{value:.2f}"
        else:
            rendered = str(value)
        lines.append(f"- `{key}`: {rendered}")
    return "\n".join(lines)


def process_step(
    *,
    path: Path,
    trajectory_id: str,
    step: Any,
    step_index: int,
    template: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_file = str(path)
    if not isinstance(step, dict):
        return None, {
            "trajectory_id": trajectory_id,
            "source_file": source_file,
            "step_id": step_index,
            "reasons": ["step_is_not_object"],
        }

    reasons: list[str] = []
    step_id = step.get("step_id", step_index)

    grid_state = step.get("grid_state")
    if not isinstance(grid_state, list) or not grid_state or not all(isinstance(line, str) for line in grid_state):
        reasons.append("missing_grid_state")
        grid_state_text = ""
    else:
        grid_state_text = "\n".join(grid_state)

    output_tokens = step.get("output_tokens")
    if not isinstance(output_tokens, list) or not output_tokens:
        reasons.append("missing_output_tokens")

    raw_agent_action = step.get("agent_action")
    if raw_agent_action is None:
        reasons.append("missing_agent_action")
    agent_action = normalize_action(raw_agent_action)
    if raw_agent_action is not None and agent_action is None:
        reasons.append("invalid_action")

    analysis_tokens = extract_analysis_tokens(output_tokens)
    analysis_text = parse_analysis_text(step.get("output_text"), analysis_tokens)
    if not analysis_text.strip():
        reasons.append("empty_analysis_text")

    analysis_token_ids = [
        int(token["token_id"])
        for token in analysis_tokens
        if isinstance(token.get("token_id"), int) or (isinstance(token.get("token_id"), str) and token["token_id"].isdigit())
    ]
    final_action_from_output = extract_final_action_from_output(output_tokens)

    row = {
        "trajectory_id": trajectory_id,
        "source_file": source_file,
        "step_id": step_id,
        "grid_state_text": grid_state_text,
        "carrying_key": step.get("carrying_key"),
        "prompt_text": build_prompt_text(template, grid_state_text, step.get("carrying_key")),
        "agent_action": agent_action,
        "analysis_token_ids_json": json_dumps(analysis_token_ids),
        "analysis_tokens_json": json_dumps(analysis_tokens),
        "final_action_from_output": final_action_from_output,
        "final_action_matches_agent_action": bool(final_action_from_output == agent_action)
        if final_action_from_output is not None and agent_action is not None
        else None,
        "analysis_text": analysis_text,
        "output_n_tokens": step.get("output_n_tokens"),
    }

    if reasons:
        quarantine = {
            "trajectory_id": trajectory_id,
            "source_file": source_file,
            "step_id": step_id,
            "reasons": reasons,
            "agent_action": raw_agent_action,
            "output_n_tokens": step.get("output_n_tokens"),
            "step_keys": sorted(step.keys()),
        }
        return None, quarantine

    return row, None


def process_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    payload, error = load_json(path)
    if payload is None:
        quarantine = {
            "trajectory_id": path.stem,
            "source_file": str(path),
            "step_id": None,
            "reasons": ["file_load_error"],
            "error": error,
        }
        return [], [quarantine], False

    trajectory_id = stable_trajectory_id(payload, path)
    steps = payload.get("steps")
    if not isinstance(steps, list):
        quarantine = {
            "trajectory_id": trajectory_id,
            "source_file": str(path),
            "step_id": None,
            "reasons": ["missing_steps"],
        }
        return [], [quarantine], True

    template = prompt_template(payload)
    valid_rows: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        row, quarantine = process_step(
            path=path,
            trajectory_id=trajectory_id,
            step=step,
            step_index=step_index,
            template=template,
        )
        if row is not None:
            valid_rows.append(row)
        if quarantine is not None:
            quarantined_rows.append(quarantine)

    return valid_rows, quarantined_rows, True


def write_quarantine(path: Path, quarantined_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in quarantined_rows:
            f.write(json_dumps(row) + "\n")


def write_report(
    *,
    report_path: Path,
    json_file_count: int,
    loaded_json_file_count: int,
    valid_rows: list[dict[str, Any]],
    quarantined_rows: list[dict[str, Any]],
) -> None:
    action_counts = Counter(row["agent_action"] for row in valid_rows)
    output_tokens = [
        int(row["output_n_tokens"])
        for row in valid_rows
        if isinstance(row.get("output_n_tokens"), int)
        or (isinstance(row.get("output_n_tokens"), str) and str(row["output_n_tokens"]).isdigit())
    ]
    match_values = [
        row["final_action_matches_agent_action"]
        for row in valid_rows
        if row["final_action_matches_agent_action"] is not None
    ]
    match_pct = (sum(bool(value) for value in match_values) / len(match_values) * 100.0) if match_values else None

    examples = [truncate_row(row) for row in valid_rows[:EXAMPLE_ROW_COUNT]]

    lines = [
        "# Data Schema Report",
        "",
        "## Summary",
        "",
        f"- JSON files discovered: `{json_file_count}`",
        f"- JSON files loaded: `{loaded_json_file_count}`",
        f"- Valid steps: `{len(valid_rows)}`",
        f"- Quarantined steps: `{len(quarantined_rows)}`",
        "",
        "## Action Distribution",
        "",
    ]

    if action_counts:
        for action in VALID_ACTIONS:
            lines.append(f"- `{action}`: `{action_counts.get(action, 0)}`")
    else:
        lines.append("- No valid actions found.")

    lines.extend(
        [
            "",
            "## `output_n_tokens` Distribution",
            "",
            format_distribution(output_token_distribution(output_tokens)),
            "",
            "## Final Action Agreement",
            "",
        ]
    )

    if match_pct is None:
        lines.append("- `final_action_from_output` agreement with `agent_action`: n/a")
    else:
        lines.append(
            "- `final_action_from_output` agreement with `agent_action`: "
            f"`{match_pct:.2f}%` over `{len(match_values)}` rows"
        )

    lines.extend(
        [
            "",
            "## Canonical Columns",
            "",
        ]
    )
    columns = list(valid_rows[0].keys()) if valid_rows else []
    if columns:
        for column in columns:
            lines.append(f"- `{column}`")
    else:
        lines.append("- No valid rows produced.")

    lines.extend(["", "## Example Rows", ""])
    if examples:
        lines.append("```json")
        lines.append(json.dumps(examples, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("No valid rows available.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    pd, _pa = require_dependencies()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(path for path in input_dir.rglob("*.json") if path.is_file())

    valid_rows: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []
    loaded_json_count = 0

    for path in json_paths:
        rows, quarantines, loaded = process_file(path)
        valid_rows.extend(rows)
        quarantined_rows.extend(quarantines)
        loaded_json_count += int(loaded)

    canonical_path = output_dir / "canonical_steps.parquet"
    report_path = output_dir / "data_schema_report.md"
    quarantine_path = output_dir / "quarantined_steps.jsonl"

    dataframe = pd.DataFrame(valid_rows)
    dataframe.to_parquet(canonical_path, engine="pyarrow", index=False)
    write_quarantine(quarantine_path, quarantined_rows)
    write_report(
        report_path=report_path,
        json_file_count=len(json_paths),
        loaded_json_file_count=loaded_json_count,
        valid_rows=valid_rows,
        quarantined_rows=quarantined_rows,
    )

    print(f"Discovered JSON files: {len(json_paths)}")
    print(f"Valid steps: {len(valid_rows)}")
    print(f"Quarantined steps: {len(quarantined_rows)}")
    print(f"Wrote {canonical_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {quarantine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
