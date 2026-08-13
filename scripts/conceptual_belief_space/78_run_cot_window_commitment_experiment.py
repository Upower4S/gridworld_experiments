#!/usr/bin/env python3
"""Wait for corrected CoT-window extraction, then run the full probe experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("--extraction_timeout_hours", type=float, default=12.0)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    activation_dir = output_dir / "activations"
    shard_dir = activation_dir / "shards"
    extraction_manifest_path = activation_dir / "extraction_manifest.json"
    training_manifest_path = output_dir / "training_manifest.json"
    lock_path = output_dir / "overnight_runner.lock"
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another overnight runner already holds {lock_path}")

    if training_manifest_path.exists():
        payload = json.loads(training_manifest_path.read_text())
        if payload.get("status") == "complete":
            print(f"[{timestamp()}] Full training is already complete.", flush=True)
            return 0

    deadline = time.monotonic() + float(args.extraction_timeout_hours) * 3600.0
    last_completed = None
    while not extraction_manifest_path.exists():
        completed = len(list(shard_dir.glob("*.json"))) if shard_dir.exists() else 0
        if completed != last_completed:
            print(
                f"[{timestamp()}] Waiting for extraction: {completed}/95 trajectory shards complete",
                flush=True,
            )
            last_completed = completed
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"extraction did not produce {extraction_manifest_path} within "
                f"{args.extraction_timeout_hours} hours"
            )
        time.sleep(max(1.0, float(args.poll_seconds)))

    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    if extraction_manifest.get("status") != "complete":
        raise RuntimeError("extraction manifest exists but is not complete")
    if extraction_manifest.get("probe_input_kind") != "cot_last_tokens":
        raise RuntimeError("refusing to train on an activation dataset that is not cot_last_tokens")
    if extraction_manifest.get("forced_suffix_tokens_in_activation_input") is not False:
        raise RuntimeError("refusing to train because suffix exclusion is not explicitly verified")

    activation_index = activation_dir / "activation_index.parquet"
    beliefs = activation_dir / "extracted_beliefs.parquet"
    for required in [activation_index, beliefs]:
        if not required.exists():
            raise FileNotFoundError(required)

    trainer = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--activation_index",
        str(activation_index),
        "--beliefs",
        str(beliefs),
        "--output_dir",
        str(output_dir),
    ]
    has_partial_training = any((output_dir / "models").glob("*.pt")) or any(
        (output_dir / "fold_artifacts").glob("split*")
    )
    if has_partial_training:
        command.append("--resume")
        mode = "resuming"
    else:
        mode = "starting"
    print(
        f"[{timestamp()}] Extraction complete with {extraction_manifest.get('n_rows')} rows; "
        f"{mode} full training.",
        flush=True,
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"[{timestamp()}] Full corrected experiment complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
