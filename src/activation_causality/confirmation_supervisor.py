"""Own one confirmation worker process group and enforce its absolute cutoff.

Production has no arbitrary-command/interpreter interface. A separate CPU parent
launches the fixed local campaign entrypoint through the same Python environment.
The invocation preserves its final virtualenv symlink; the resolved executable
binary and environment prefix/configuration are authenticated separately. The worker must call
validate_execution_lease before loading a model and before each job. This module
alone does not retrofit that mandatory check into a legacy worker.

SIGKILL is sent without post-cutoff grace. Actual signal timing is recorded;
normal OS scheduling/suspension is not a hard real-time guarantee. Existing
results are preserved and never reclassified as completed behavioral trials.
The parent must remain running: ordinary termination signals trigger owned
group cleanup, but an uncatchable parent SIGKILL or machine suspension requires
external process supervision for a stronger availability guarantee.

The fixed campaign worker has no child inference processes. Group signalling
is safe only while its leader remains owned and unreaped. This helper does not
claim to supervise detached descendants or descendants left after that leader
exits; it never signals a potentially reused group after reaping the leader.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any
import uuid

from .confirmation_contract import remaining_inference_seconds
from .registry import fingerprint

FORMAT = "supervised_confirmation_execution_lease_v1"
LEASE_PATH_ENV = "ACTIVATION_CONFIRMATION_LEASE_PATH"
LEASE_SHA_ENV = "ACTIVATION_CONFIRMATION_LEASE_SHA256"
REQUIRED_SOURCES = ("src/activation_causality/confirmation_supervisor.py",
                    "src/activation_causality/confirmation_contract.py",
                    "src/activation_causality/campaign.py")


class SupervisorInterrupted(RuntimeError):
    """A normal termination signal must clean up the owned worker first."""


@contextmanager
def _termination_handlers():
    previous = {}
    def interrupt(signum, frame):
        raise SupervisorInterrupted(f"supervisor received {signal.Signals(signum).name}")
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    json.dumps(value, allow_nan=False)
    return value, _sha(raw)


def _exclusive_json(path: Path, value: dict) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # A waiting child sees either no launch record or its complete bytes.
        # Hard-link publication is atomic and refuses to overwrite an old lease.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(raw)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"invalid {label}")
    return value


def _paths(campaign: Path, lease_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[0-9a-f]{32}", lease_id):
        raise ValueError("invalid lease ID")
    return {"lease": campaign / "leases" / (lease_id + ".json"),
            "launch": campaign / "reviews" / "confirmation_supervision" / (lease_id + ".launch.json"),
            "terminal": campaign / "reviews" / "confirmation_supervision" / (lease_id + ".terminal.json"),
            "log": campaign / "logs" / ("confirmation_supervision_" + lease_id + ".log")}


def _check_sources(repo: Path, sources: Any) -> None:
    if not isinstance(sources, dict) or any(name not in sources for name in REQUIRED_SOURCES):
        raise ValueError("supervised confirmation requires the frozen supervisor/contract/worker sources")
    for name, digest in sources.items():
        path = repo / name
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not path.resolve().is_relative_to(repo) or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None or _sha(path.read_bytes()) != digest):
            raise ValueError(f"frozen implementation source changed: {name}")


def _interpreter_binding() -> dict:
    """Keep PEP405 environment discovery at the invoked path, not its symlink target."""
    invoked = Path(sys.executable)
    if not invoked.is_absolute():
        raise ValueError("supervisor requires an absolute Python executable invocation")
    # Canonicalize directories without resolving the final python symlink:
    # launching its base-interpreter target would lose the virtual environment.
    invocation = invoked.parent.resolve(strict=True) / invoked.name
    binary = invocation.resolve(strict=True)
    if not binary.is_file():
        raise ValueError("supervisor Python executable is not a file")
    prefix = Path(sys.prefix).resolve(strict=True)
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    config = prefix / "pyvenv.cfg" if prefix != base_prefix else None
    if config is not None and not config.is_file():
        raise ValueError("virtual environment must retain its pyvenv.cfg")
    return {"interpreter_path": str(invocation), "interpreter_resolved_path": str(binary),
            "interpreter_binary_sha256": _sha(binary.read_bytes()),
            "interpreter_prefix": str(prefix), "interpreter_base_prefix": str(base_prefix),
            "interpreter_pyvenv_cfg_path": str(config) if config is not None else None,
            "interpreter_pyvenv_cfg_sha256": _sha(config.read_bytes()) if config is not None else None}


def _artifacts(campaign: Path, repo: Path, experiment_id: str) -> dict:
    """Read/hash identical bytes and verify freeze/cutoff/entrypoint contracts."""
    experiment_id = _identifier(experiment_id, "experiment ID")
    spec_path = campaign / "specs" / (experiment_id + ".json")
    spec, spec_sha = _read(spec_path)
    if spec.get("experiment_id") != experiment_id or spec.get("partition") != "confirmation":
        raise ValueError("supervisor accepts only an exact registered confirmation specification")
    freeze_id = _identifier(spec.get("freeze_id"), "freeze ID")
    freeze_path = campaign / "freezes" / (freeze_id + ".json")
    freeze, freeze_sha = _read(freeze_path)
    config, campaign_sha = _read(campaign / "campaign.json")
    cutoff = freeze.get("inference_cutoff")
    if not isinstance(cutoff, str):
        raise ValueError("confirmation freeze must contain its absolute inference cutoff")
    remaining_inference_seconds(config["deadline"], cutoff, now=_now())
    sources = freeze.get("implementation")
    _check_sources(repo, sources)
    entrypoint = repo / "scripts/30_activation_causality_campaign.py"
    entrypoint_sha = _sha(entrypoint.read_bytes())
    if freeze.get("worker_entrypoint_sha256") != entrypoint_sha:
        raise ValueError("frozen worker entrypoint hash changed or is missing")
    if not isinstance(spec.get("candidate_id"), str) or not spec["candidate_id"]:
        raise ValueError("confirmation specification requires candidate identity")
    return {"spec": spec, "freeze": freeze, "config": config,
            "binding": {"experiment_id": experiment_id, "freeze_id": freeze_id,
                        "candidate_id": spec["candidate_id"], "cohort_role": spec.get("cohort_role", "primary"),
                        "repo_path": str(repo), "campaign_path": str(campaign), "campaign_sha256": campaign_sha,
                        "spec_path": str(spec_path), "spec_sha256": spec_sha,
                        "freeze_path": str(freeze_path), "freeze_sha256": freeze_sha,
                        "inference_cutoff": cutoff, "worker_entrypoint_path": str(entrypoint),
                        "worker_entrypoint_sha256": entrypoint_sha, "source_hashes": sources,
                        **_interpreter_binding()}}


@contextmanager
def _supervisor_lock(campaign: Path):
    """Separate ownership lock; the child retains its normal model-worker lock."""
    path = campaign / "confirmation_supervisor.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another confirmation supervisor owns this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_execution_lease(campaign: Path, repo: Path, spec: dict) -> dict:
    """Worker-side fail-closed check, before model load and every job.

    The brief launch-record handshake happens before inference. Matching the
    exact child PID and a dedicated session prevents reuse by a different child
    of a still-live supervisor. Terminal leases cannot be resumed; a fresh
    supervisor run always creates a fresh lease while preserving old records.
    """
    campaign, repo = campaign.resolve(), repo.resolve()
    raw_path, expected_sha = os.environ.get(LEASE_PATH_ENV), os.environ.get(LEASE_SHA_ENV)
    if not raw_path or not expected_sha:
        raise ValueError("confirmation inference requires its dedicated supervisor lease")
    path = Path(raw_path).resolve()
    if path.parent != campaign / "leases" or path.suffix != ".json":
        raise ValueError("confirmation lease must be inside the campaign lease directory")
    lease, actual_sha = _read(path)
    if actual_sha != expected_sha or lease.get("format") != FORMAT or path.stem != lease.get("lease_id"):
        raise ValueError("confirmation lease content/hash/identity mismatch")
    paths = _paths(campaign, lease["lease_id"])
    if paths["terminal"].exists():
        raise ValueError("terminal confirmation lease cannot be reused")
    artifacts = _artifacts(campaign, repo, spec.get("experiment_id"))
    if fingerprint(artifacts["spec"]) != fingerprint(spec):
        raise ValueError("worker specification differs from the immutable lease input")
    if any(lease.get(key) != value for key, value in artifacts["binding"].items()):
        raise ValueError("confirmation lease source/freeze/spec/campaign binding changed")
    parent = lease.get("supervisor_pid")
    if type(parent) is not int or parent <= 1 or os.getppid() != parent or os.getsid(0) != os.getpid():
        raise ValueError("worker is not the dedicated child/session of its supervisor")
    try:
        os.kill(parent, 0)
    except OSError as exc:
        raise ValueError("confirmation supervisor is no longer alive") from exc
    handshake_deadline = time.monotonic() + 2
    while not paths["launch"].exists():
        remaining = remaining_inference_seconds(artifacts["config"]["deadline"], lease["inference_cutoff"], now=_now())
        if remaining <= 0 or time.monotonic() >= handshake_deadline or os.getppid() != parent:
            raise ValueError("supervisor launch handshake absent or cutoff reached")
        time.sleep(min(.01, remaining))
    launch, _ = _read(paths["launch"])
    if (launch.get("lease_sha256") != expected_sha or launch.get("worker_pid") != os.getpid()
            or launch.get("worker_process_group") != os.getpid() or launch.get("supervisor_pid") != parent
            or paths["terminal"].exists()):
        raise ValueError("supervisor launch record does not authorize this live child")
    if remaining_inference_seconds(artifacts["config"]["deadline"], lease["inference_cutoff"], now=_now()) <= 0:
        raise ValueError("absolute confirmation inference cutoff reached")
    return {"lease_path": str(path), "lease_sha256": actual_sha, "lease_id": lease["lease_id"],
            "inference_cutoff": lease["inference_cutoff"], "supervisor_pid": parent}


def _owned_kill(process: subprocess.Popen) -> dict:
    """Only a still-owned, unreaped child group can be signalled here."""
    if process.poll() is not None:
        return {"signal_sent": False, "reason": "owned_worker_already_exited"}
    sent = _now()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return {"signal_sent": False, "reason": "owned_process_group_already_exited"}
    return {"signal_sent": True, "signal": "SIGKILL", "signal_sent_at": sent,
            "worker_process_group": process.pid}


def _inventory(campaign: Path, spec: dict) -> dict:
    expected = {job["job_id"] + ".json" for job in spec["jobs"]}
    actual = {path.name for path in (campaign / "results" / spec["experiment_id"]).glob("*.json")}
    return {"planned_jobs": len(expected), "saved_result_files": len(actual & expected),
            "missing_result_files": sorted(expected - actual), "unexpected_result_files": sorted(actual - expected),
            "interpretation": "filename inventory only; existing results preserved, no behavioral completion or outcome inference"}


def run_supervised_confirmation(campaign: Path, repo: Path, experiment_id: str, *, poll_seconds: float = .1) -> dict:
    """Launch exactly one fixed local worker under a newly created cutoff lease.

    No caller-supplied command, interpreter, environment override, or existing
    lease is accepted. Return immutable terminal evidence on normal/nonzero/
    cutoff exits; interruptions preserve evidence then propagate to the caller.
    """
    if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= .25:
        raise ValueError("poll_seconds must be finite and in (0, 0.25]")
    campaign, repo = campaign.resolve(), repo.resolve()
    from .selection import validate_confirmation_ready
    with _supervisor_lock(campaign), _termination_handlers():
        initial = _artifacts(campaign, repo, experiment_id)
        validate_confirmation_ready(campaign, repo, initial["spec"])
        artifacts = _artifacts(campaign, repo, experiment_id)
        if artifacts["binding"] != initial["binding"]:
            raise ValueError("confirmation inputs changed across readiness validation")
        lease_id = uuid.uuid4().hex
        paths = _paths(campaign, lease_id)
        lease = {"format": FORMAT, "lease_id": lease_id, "created_at": _now(),
                 "supervisor_pid": os.getpid(), **artifacts["binding"]}
        lease_sha = _exclusive_json(paths["lease"], lease)
        report = {"format": "supervised_confirmation_terminal_v1", "lease_id": lease_id,
                  "lease_path": str(paths["lease"]), "lease_sha256": lease_sha,
                  "launch_record_path": str(paths["launch"]), "terminal_record_path": str(paths["terminal"]),
                  "log_path": str(paths["log"]), "experiment_id": experiment_id,
                  "inference_cutoff": lease["inference_cutoff"], "started_at": _now(),
                  "process_ownership_scope": "fixed worker and its group only while leader is owned/unreaped; no detached or post-leader-exit descendants",
                  "fixed_worker_assumption": "the registered campaign worker creates no child inference processes",
                  "automated_operational_record_not_scientific_review": True}
        process = None
        failure = None
        remaining = remaining_inference_seconds(artifacts["config"]["deadline"], lease["inference_cutoff"], now=_now())
        monotonic_cutoff = time.monotonic() + remaining
        try:
            if remaining <= 0:
                report.update(status="cutoff_before_launch", worker_exit_code=None, incomplete=True)
            else:
                command = [lease["interpreter_path"], "-I", "-B", lease["worker_entrypoint_path"],
                           "--campaign", str(campaign), "run", "--experiment-id", experiment_id]
                environment = os.environ.copy()
                environment[LEASE_PATH_ENV], environment[LEASE_SHA_ENV] = str(paths["lease"]), lease_sha
                paths["log"].parent.mkdir(parents=True, exist_ok=True)
                with paths["log"].open("xb") as logfile:
                    process = subprocess.Popen(command, cwd=repo, env=environment, stdout=logfile,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    _exclusive_json(paths["launch"], {"format": "supervised_confirmation_launch_v1", "lease_id": lease_id,
                                    "lease_sha256": lease_sha, "supervisor_pid": os.getpid(), "worker_pid": process.pid,
                                    "worker_process_group": process.pid, "command": command, "launched_at": _now()})
                    while process.poll() is None:
                        utc_remaining = remaining_inference_seconds(artifacts["config"]["deadline"], lease["inference_cutoff"], now=_now())
                        remaining = min(utc_remaining, monotonic_cutoff - time.monotonic())
                        if remaining <= 0:
                            report.update(status="cutoff_terminated", incomplete=True, cutoff_action=_owned_kill(process))
                            action = report["cutoff_action"]
                            action.update(utc_remaining_seconds=utc_remaining,
                                          monotonic_remaining_seconds=monotonic_cutoff - time.monotonic())
                            if action.get("signal_sent_at"):
                                action["cutoff_signal_lateness_seconds"] = max(0.0, (
                                    datetime.fromisoformat(action["signal_sent_at"])
                                    - datetime.fromisoformat(lease["inference_cutoff"])).total_seconds())
                            break
                        time.sleep(min(float(poll_seconds), remaining))
                    exit_code = process.wait(timeout=5)
                    report["worker_exit_code"] = exit_code
                    if "status" not in report:
                        report.update(status="worker_exited" if exit_code == 0 else "worker_failed", incomplete=exit_code != 0)
        except BaseException as exc:
            failure = exc
            report.update(status="supervisor_error", incomplete=True, error_type=type(exc).__name__, error=str(exc))
            if process is not None:
                report["cleanup_action"] = _owned_kill(process)
                report["worker_exit_code"] = process.wait(timeout=5)
        finally:
            inventory = _inventory(campaign, artifacts["spec"])
            report.update(finished_at=_now(), current_result_inventory=inventory,
                          incomplete=bool(report.get("incomplete") or inventory["missing_result_files"] or inventory["unexpected_result_files"]),
                          cohort_technical_completion_not_assessed=True,
                          cutoff_clock_policy="minimum of absolute UTC remainder and initial monotonic budget; clock rollback cannot extend inference")
            if paths["launch"].exists():
                report["launch_record_sha256"] = _sha(paths["launch"].read_bytes())
            if paths["log"].exists():
                report["log_sha256"] = _sha(paths["log"].read_bytes())
            report["terminal_record_sha256"] = _exclusive_json(paths["terminal"], report)
        if failure is not None:
            raise failure
        return report
