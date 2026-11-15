"""Immutable experiment specifications, atomic records and an exclusive worker lock."""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
GOAL = (
    "Complete a 72-hour local investigation of whether task-relevant internal "
    "representations causally influence GPT-OSS-20B's gridworld actions. Test multiple "
    "intervention families, independently evaluate the strongest candidates on fresh "
    "environments, and deliver reproducible evidence and a proposed causal-section "
    "revision. Distinguish control, semantic specificity, necessity, and unresolved "
    "or negative results. Arbitrary action flips and probe accuracy are insufficient."
)
PAPER = "_IAB_Workshop_2026__Belief_action_gap-4.pdf"
MODEL_SNAPSHOT = "6cee5e81ee83917806bbde320786a8fb61efebee"
DEFAULT_MODEL = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots" / MODEL_SNAPSHOT
DEFAULT_CAMPAIGN = Path("outputs/activation_causality/campaign_20260912")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if fingerprint(read_json(path)) != fingerprint(value):
            raise ValueError(f"refusing to change immutable record: {path}; use a new experiment ID")
        return
    # Atomic publication: never expose a partly written immutable specification.
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        if fingerprint(read_json(path)) != fingerprint(value):
            raise ValueError(f"concurrent conflicting immutable record: {path}")
    finally:
        tmp.unlink(missing_ok=True)


def append_event(campaign: Path, kind: str, **payload: Any) -> None:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / "events.jsonl").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(canonical_json({"time": now_utc(), "kind": kind, **payload}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def worker_lock(campaign: Path, *, lock_root: Path | None = None) -> Iterator[None]:
    # Repo-wide lock: distinct campaigns must not load concurrent 20B model instances.
    root = lock_root if lock_root is not None else Path(__file__).resolve().parents[2] / "outputs/activation_causality"
    lock_path = root / "model_worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another model worker holds {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(canonical_json({"pid": os.getpid(), "campaign": str(campaign.resolve()), "time": now_utc()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def init_campaign(campaign: Path, repo: Path, *, hours: float = 72.0, model_path: Path = DEFAULT_MODEL) -> dict:
    path = campaign / "campaign.json"
    if path.exists():
        existing = read_json(path)
        if existing["budget_hours"] != hours or Path(existing["model_path"]) != model_path.resolve():
            raise ValueError("campaign configuration differs; deadlines cannot silently reset")
        return existing
    start = datetime.now(timezone.utc)
    value = {
        "schema_version": SCHEMA_VERSION,
        "goal": GOAL,
        "started_at": start.isoformat(),
        "deadline": (start + timedelta(hours=hours)).isoformat(),
        "budget_hours": hours,
        "paper": str((repo / PAPER).resolve()),
        "paper_sha256": file_sha256(repo / PAPER),
        "model_path": str(model_path.resolve()),
        "experimental_compute": "local_only",
        "agent_slots": 4,
        "model_workers": 1,
        "minimum_relevant_effect": 0.10,
        "max_confirmation_candidates": 2,
        "confirmation_layout_targets": [128, 64, 32],
        "confirmation_opened": False,
    }
    immutable_json(path, value)
    for name in ("specs", "datasets", "results", "logs", "reviews", "reports", "freezes"):
        (campaign / name).mkdir(parents=True, exist_ok=True)
    append_event(campaign, "campaign_created", deadline=value["deadline"])
    return value


def seconds_remaining(campaign: Path) -> float:
    deadline = datetime.fromisoformat(read_json(campaign / "campaign.json")["deadline"])
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def register_spec(campaign: Path, spec: dict) -> Path:
    experiment_id = spec["experiment_id"]
    if not experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in experiment_id):
        raise ValueError("experiment_id must contain only letters, digits, underscore or hyphen")
    if spec.get("partition") not in {"historical", "discovery", "confirmation"}:
        raise ValueError("partition must be historical, discovery or confirmation")
    if spec.get("partition") == "confirmation" and not spec.get("freeze_id"):
        raise ValueError("confirmation requires a frozen selection record")
    seen = set()
    for job in spec["jobs"]:
        key = job.get("job_id")
        if not key or key in seen:
            raise ValueError("job IDs must be unique and nonempty")
        seen.add(key)
        if job["target_action"] not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("invalid target action")
    path = campaign / "specs" / f"{experiment_id}.json"
    immutable_json(path, spec)
    append_event(campaign, "spec_registered", experiment_id=experiment_id, sha256=file_sha256(path), jobs=len(seen))
    return path


def status(campaign: Path) -> dict:
    config = read_json(campaign / "campaign.json")
    experiments = []
    for path in sorted((campaign / "specs").glob("*.json")):
        spec = read_json(path)
        results = campaign / "results" / spec["experiment_id"]
        records = [read_json(p) for p in sorted(results.glob("*.json"))] if results.exists() else []
        experiments.append({"experiment_id": spec["experiment_id"], "partition": spec["partition"], "retired": (campaign / "reviews" / f"{spec['experiment_id']}_retired.json").exists(), "planned": len(spec["jobs"]), "completed": sum(r.get("status") == "complete" for r in records), "errors": sum(r.get("status") == "error" for r in records)})
    return {"goal": config["goal"], "deadline": config["deadline"], "hours_remaining": max(0, seconds_remaining(campaign) / 3600), "experiments": experiments}
