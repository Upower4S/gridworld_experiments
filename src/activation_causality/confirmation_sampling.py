"""Persisted, bounded CPU-only final confirmation sampling.

This creates a planning manifest, never a registered dataset/specification,
freeze, model result, or readiness approval. The caller authenticates the
production generator and designated adapter; injectable callbacks serve CPU
fixtures too. No fallback transition or new route is implemented here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from pathlib import Path
import re
import signal
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .environments import canonical_layout_fingerprint
from .input_sites import _physical_counterfactual
from .registry import atomic_json, file_sha256, fingerprint, immutable_json, read_json

FAMILIES = ("wall", "key_possession", "door_status")
QUOTAS = {32: (11, 11, 10), 64: (22, 21, 21), 128: (43, 43, 42)}
SALT = "activation_lexical_v1:20260912"
FORMAT = "final_confirmation_sampling_plan_v1"
_CLOCKS: dict[str, tuple[float, float]] = {}


class SamplingDeadlineExceeded(RuntimeError):
    """The shared sampling or absolute inference deadline elapsed."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _utc(value: str) -> datetime:
    _require(isinstance(value, str), "UTC timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), "explicit UTC timestamps required")
    return parsed.astimezone(timezone.utc)


def _now(source: str | Callable[[], str] | None) -> datetime:
    value = source() if callable(source) else source
    return datetime.now(timezone.utc) if value is None else _utc(value)


def _names(values: Any) -> bool:
    return isinstance(values, list) and all(isinstance(value, str) and value for value in values)


def _identity(value: Mapping[str, Any]) -> dict:
    identity = deepcopy(dict(value))
    required = {"policy_sha256", "campaign_deadline_utc", "inference_cutoff_utc", "dataset_seed",
                "generator_sha256", "generator_parameters", "historical_exclusion", "candidates",
                "previously_used_layout_aliases", "previously_used_layout_aliases_sha256",
                "previously_used_layout_aliases_provenance"}
    _require(required <= identity.keys(), "missing sampling identity fields")
    _require(_sha(identity["policy_sha256"]) and _sha(identity["generator_sha256"]), "bind policy and generator fingerprints")
    _require(type(identity["dataset_seed"]) is int and 0 <= identity["dataset_seed"] < 2**32 - 2100, "invalid dataset seed")
    deadline, cutoff = _utc(identity["campaign_deadline_utc"]), _utc(identity["inference_cutoff_utc"])
    _require(deadline - cutoff == timedelta(hours=6), "inference cutoff must preserve exactly six reporting hours")
    previous = identity["previously_used_layout_aliases"]
    _require(_names(previous) and len(set(previous)) == len(previous), "invalid previous layout aliases")
    _require(identity["previously_used_layout_aliases_sha256"] == fingerprint(previous)
             and isinstance(identity["previously_used_layout_aliases_provenance"], dict)
             and bool(identity["previously_used_layout_aliases_provenance"]), "bind previous alias corpus and provenance")
    parameters = identity["generator_parameters"]
    _require(isinstance(parameters, dict) and set(parameters) ==
             {"sizes", "wall_density", "donor_mode", "require_unique_actions", "historical_steps"}, "freeze explicit generator parameters")
    fixed = {"sizes": [7, 9], "wall_density": 0.23, "donor_mode": "matched", "require_unique_actions": True}
    _require(fingerprint({key: parameters[key] for key in fixed}) == fingerprint(fixed), "generator parameters differ from frozen population")
    historical = identity["historical_exclusion"]
    source = parameters["historical_steps"]
    _require(isinstance(source, str) and Path(source).is_absolute(), "final sampling requires a bound historical source path")
    _require(isinstance(historical, dict) and historical.get("status") == "checked" and historical.get("source_path") == source
             and _sha(historical.get("source_sha256")) and isinstance(historical.get("source_exclusion_provenance"), dict)
             and bool(historical["source_exclusion_provenance"]), "final sampling cannot use unchecked historical exclusions")
    candidates = identity["candidates"]
    _require(isinstance(candidates, list) and 1 <= len(candidates) <= 2, "freeze one or two ordered profiles")
    seen = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict), "candidate definition must be an object")
        required_candidate = {"candidate_id", "source_experiment_id", "source_spec_sha256", "configuration",
                              "configuration_sha256", "eligibility_builder_sha256", "eligibility_rule",
                              "families", "family_layout_counts", "require_joint_renderer_eligibility"}
        _require(required_candidate <= candidate.keys(), "missing candidate sampling identity fields")
        for key in ("candidate_id", "source_experiment_id"):
            _require(isinstance(candidate[key], str) and re.fullmatch(r"[A-Za-z0-9_-]+", candidate[key]) is not None,
                     "invalid candidate/source ID")
        _require(candidate["candidate_id"] not in seen, "duplicate candidate ID")
        seen.add(candidate["candidate_id"])
        for key in ("source_spec_sha256", "configuration_sha256", "eligibility_builder_sha256"):
            _require(_sha(candidate[key]), "missing candidate fingerprint " + key)
        config = candidate["configuration"]
        _require(isinstance(config, dict) and fingerprint(config) == candidate["configuration_sha256"], "configuration fingerprint mismatch")
        for key in ("candidate_id", "source_experiment_id", "source_spec_sha256"):
            _require(config.get(key) == candidate[key], "configuration/source identity contradiction")
        families, quotas = candidate["families"], candidate["family_layout_counts"]
        _require(families == list(FAMILIES) or (isinstance(families, list) and len(families) == 1 and families[0] in FAMILIES),
                 "canonical three families or one frozen family required")
        _require(isinstance(quotas, dict) and set(quotas) == set(families)
                 and all(type(n) is int and n > 0 for n in quotas.values()), "invalid family quota map")
        count = sum(quotas.values())
        _require(count in QUOTAS and type(config.get("n_layouts")) is int and config["n_layouts"] == count,
                 "freeze N32/64/128 without changing allocation")
        if len(families) == 3:
            _require(quotas == dict(zip(FAMILIES, QUOTAS[count], strict=True)), "mixed quotas differ from fixed policy")
            _require(config.get("families") == families and config.get("family_layout_counts") == quotas,
                     "mixed configuration quota mismatch")
        else:
            _require(config.get("family") == families[0], "single-family configuration mismatch")
        joint = candidate["require_joint_renderer_eligibility"]
        _require(type(joint) is bool and config.get("require_joint_renderer_eligibility", joint) is joint,
                 "joint renderer requirement must be explicitly frozen")
        inventory = config.get("condition_inventory")
        _require(_names(inventory) and len(inventory) in {14, 15} and len(set(inventory)) == len(inventory),
                 "complete fixed control inventory required")
        rule = candidate["eligibility_rule"]
        _require(isinstance(rule, dict) and set(rule) == {"identifier", "sha256"}
                 and isinstance(rule["identifier"], str) and rule["identifier"] and _sha(rule["sha256"]), "invalid eligibility rule")
    # Full finite JSON fingerprint also covers any additional reviewed identity fields.
    fingerprint(identity)
    return identity


def _verify_historical(identity: dict) -> dict:
    source = Path(identity["generator_parameters"]["historical_steps"])
    before = source.stat()
    digest = file_sha256(source)
    after = source.stat()
    def signature(info):
        return info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev
    _require(signature(before) == signature(after), "historical source changed during verification")
    _require(digest == identity["historical_exclusion"]["source_sha256"], "historical source fingerprint changed")
    return {"source_path": str(source), "source_sha256": digest, "size_bytes": after.st_size,
            "source_exclusion_provenance": deepcopy(identity["historical_exclusion"]["source_exclusion_provenance"])}


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another final sampler holds this plan") from exc
        yield


def _save(path: Path, state: dict) -> None:
    state["revision"] += 1
    state["state_sha256"] = fingerprint({key: value for key, value in state.items() if key != "state_sha256"})
    atomic_json(path, state)


def _load(path: Path) -> dict:
    state = read_json(path)
    _require(state.get("format") == FORMAT and state.get("state_sha256") ==
             fingerprint({key: value for key, value in state.items() if key != "state_sha256"}), "sampling state fingerprint mismatch")
    _require(state["identity_sha256"] == fingerprint(_identity(state["identity"])), "sampling identity fingerprint mismatch")
    return state


def _stop(state: dict, reason: str, **evidence: Any) -> None:
    state.update(status="terminal_incomplete", terminal_reason=reason)
    state["events"].append({"kind": reason, "last_utc": state["last_utc"], **evidence})


def _observe(path: Path, state: dict, now_source: Any) -> datetime:
    current, monotonic = _now(now_source), time.monotonic()
    previous = _utc(state["last_utc"])
    key = str(path.resolve()) + ":" + state["identity_sha256"] + ":" + state["started_utc"]
    anchor = _CLOCKS.setdefault(key, (current.timestamp(), monotonic))
    if current < previous or current.timestamp() + 1.0 < anchor[0] + monotonic - anchor[1]:
        _stop(state, "clock_moved_backwards_or_stalled", observed_utc=current.isoformat(),
              process_monotonic_elapsed_seconds=monotonic - anchor[1])
        raise SamplingDeadlineExceeded("UTC/monotonic clock consistency failed")
    state["last_utc"] = current.isoformat()
    if current >= _utc(state["effective_sampling_cutoff_utc"]):
        _stop(state, "sampling_time_budget_exhausted", observed_utc=current.isoformat())
        raise SamplingDeadlineExceeded("shared two-hour or absolute inference cutoff reached")
    return current


@contextmanager
def _bounded(seconds: float):
    """Interrupt an overlong CPU callback; preserve its reserved batch on error."""
    _require(threading.current_thread() is threading.main_thread(), "bounded final sampler requires the main thread")
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "cannot replace an existing process timer")
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(_signum, _frame):
        raise SamplingDeadlineExceeded("callback exceeded the persisted sampling cutoff")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-6))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def initialize_sampling_plan(path: str | Path, identity: Mapping[str, Any], *, now_utc: Any = None) -> dict:
    """Start the single shared clock, or authenticate and return the existing plan."""
    path, identity = Path(path), _identity(identity)
    with _lock(path):
        historical_audit = _verify_historical(identity)
        if path.exists():
            state = _load(path)
            _require(state["identity_sha256"] == fingerprint(identity), "refusing a changed sampling identity or reset route")
            return deepcopy(state)
        start = _now(now_utc)
        sampling_cutoff = start + timedelta(hours=2)
        effective = min(sampling_cutoff, _utc(identity["inference_cutoff_utc"]))
        state = {"format": FORMAT, "identity": identity, "identity_sha256": fingerprint(identity),
                 "started_utc": start.isoformat(), "last_utc": start.isoformat(),
                 "shared_sampling_deadline_utc": sampling_cutoff.isoformat(), "effective_sampling_cutoff_utc": effective.isoformat(),
                 "revision": 0, "status": "sampling", "terminal_reason": None, "in_progress": None,
                 "attempts": [], "attempted_pairs": [], "events": [],
                 "global_layout_aliases": sorted(identity["previously_used_layout_aliases"]),
                 "candidates": {c["candidate_id"]: {"accepted_pair_ids": [], "accepted_n_layouts_by_family": dict.fromkeys(c["families"], 0),
                     "next_attempt_by_family": dict.fromkeys(c["families"], 0)} for c in identity["candidates"]},
                 "registered_dataset": False, "registered_specification": False, "model_queries": 0,
                 "fallback_transition_supported": False, "clock_override_used": now_utc is not None}
        state["initial_historical_verification"] = historical_audit
        if start >= effective:
            _stop(state, "sampling_time_budget_exhausted")
        state["state_sha256"] = fingerprint(state)
        immutable_json(path, state)
        _CLOCKS[str(path.resolve()) + ":" + state["identity_sha256"] + ":" + state["started_utc"]] = (start.timestamp(), time.monotonic())
        return deepcopy(state)


def _pair_aliases(pair: Mapping[str, Any]) -> set[str]:
    _require(isinstance(pair, Mapping), "generated pair must be an object")
    aliases: set[str] = set()
    for obj in (pair, pair.get("metadata", {})):
        _require(isinstance(obj, Mapping), "invalid physical pair metadata")
        if isinstance(obj.get("layout_id"), str) and obj["layout_id"]:
            aliases.add(obj["layout_id"])
        for key in ("layout_aliases", "variant_layout_ids"):
            values = obj.get(key, [])
            _require(_names(values) and len(values) == len(set(values)), "invalid or duplicate declared pair aliases")
            aliases.update(values)
        for key in ("pair", "source_pair"):
            if key in obj:
                aliases.update(_pair_aliases(obj[key]))
    for role in ("recipient", "counterfactual", "donor"):
        if role in pair:
            aliases.add(canonical_layout_fingerprint(pair[role]))
    return aliases


def _physical(pair: dict, family: str, *, generator_call: dict | None = None) -> set[str]:
    _physical_counterfactual(pair, partition="confirmation")
    _require(pair["family"] == family, "generated pair is outside requested family")
    _require(pair["pair_id"] == fingerprint({key: pair[key] for key in ("layout_id", "recipient", "donor", "family")}),
             "noncanonical generated pair ID")
    if generator_call is not None:
        metadata = pair["metadata"]
        _require(type(metadata.get("seed")) is int and metadata["seed"] == generator_call["seed"], "generated seed differs from reserved call")
        _require(type(metadata.get("candidate_attempt")) is int
                 and 0 <= metadata["candidate_attempt"] < generator_call["max_attempts"], "generated inner attempt outside reserved budget")
        _require(len(pair["recipient"]["grid"]) in generator_call["sizes"], "generated grid size outside frozen population")
    return _pair_aliases(pair)


def _reported_aliases(value: Any) -> set[str]:
    """Retain even rejected declarations; this does not certify their validity."""
    if not isinstance(value, Mapping):
        return set()
    result = {value["layout_id"]} if isinstance(value.get("layout_id"), str) and value["layout_id"] else set()
    for key in ("layout_aliases", "variant_layout_ids"):
        if isinstance(value.get(key), list):
            result.update(item for item in value[key] if isinstance(item, str) and item)
    for key in ("metadata", "pair", "source_pair"):
        result.update(_reported_aliases(value.get(key)))
    return result


def _decision(decision: Any, pair: dict, candidate: dict) -> dict:
    _require(isinstance(decision, dict) and decision.get("pair_id") == pair["pair_id"]
             and decision.get("layout_id") == pair["layout_id"], "missing or mismatched builder decision identity")
    _require(type(decision.get("eligible")) is bool, "explicit builder eligibility required")
    aliases = decision.get("layout_aliases")
    _require(_names(aliases) and len(aliases) == len(set(aliases)) and pair["layout_id"] in aliases,
             "builder decision requires unique complete layout aliases")
    _require(_pair_aliases(pair) <= set(aliases), "builder decision omitted physical pair aliases")
    _require(decision.get("source_pair_sha256") == fingerprint(pair), "builder physical pair fingerprint mismatch")
    evidence = decision.get("evidence")
    _require(isinstance(evidence, dict) and isinstance(evidence.get("adapter_manifest"), dict), "retain full adapter eligibility manifest")
    inventory = candidate["configuration"]["condition_inventory"]
    _require(evidence.get("required_condition_inventory") == inventory, "eligibility inventory differs from frozen profile")
    if decision["eligible"]:
        _require(decision.get("reason") is None and evidence["adapter_manifest"].get("eligible") is True
                 and evidence["adapter_manifest"].get("scheduled_conditions") == inventory, "eligible decision lacks all controls")
        if candidate["require_joint_renderer_eligibility"]:
            joint = evidence.get("joint_renderer_audit")
            _require(isinstance(joint, dict) and joint.get("eligible") is True and joint.get("pair_id") == pair["pair_id"],
                     "eligible decision lacks frozen joint renderer proof")
    else:
        _require(isinstance(decision.get("reason"), str) and decision["reason"], "rejected decision requires a reason")
    fingerprint(decision)
    return deepcopy(decision)


def advance_sampling_plan(
    path: str | Path, *, generator: Callable[..., list[dict]], eligibility_builder: Callable[[dict, list[dict]], dict],
    now_utc: Any = None, expected_identity_sha256: str | None = None,
) -> dict:
    """Reserve/run/persist at most one deterministic batch.

    Generator receives the existing ``generate_pairs`` signature. Builder receives
    (deep-copied frozen candidate definition, exact returned pairs), returning
    ``{eligibility_builder_sha256, decisions:[...]}``. Every decision binds pair
    and layout IDs, source_pair_sha256, eligible, reason, unique complete aliases,
    and evidence={adapter_manifest,required_condition_inventory,joint_renderer_audit}.
    The last field is mandatory for eligible joint-renderer profiles. Root must
    authenticate the designated callback; this checks its declared bindings.

    Generator exhaustion is terminal: the existing generator does not expose
    partially generated objects on exception. Their bytes are explicitly marked
    unavailable. All successfully returned pairs are persisted before eligibility.
    """
    path = Path(path)
    with _lock(path):
        state = _load(path)
        _require((now_utc is not None) == state["clock_override_used"], "sampling clock override mode changed across resume")
        if expected_identity_sha256 is not None:
            _require(expected_identity_sha256 == state["identity_sha256"], "sampling identity differs from caller")
        try:
            historical_audit = _verify_historical(state["identity"])
        except (ValueError, OSError) as exc:
            _stop(state, "historical_source_verification_failed", error_type=type(exc).__name__, error=str(exc))
            _save(path, state)
            return deepcopy(state)
        if state["status"] != "sampling":
            return deepcopy(state)
        if state["in_progress"] is not None:
            _stop(state, "interrupted_reserved_batch_not_retried", batch=deepcopy(state["in_progress"]))
            state["attempts"][state["in_progress"]["attempt_index"]]["status"] = "interrupted_terminal"
            for record in state["attempted_pairs"]:
                if record["batch_id"] == state["in_progress"]["batch_id"] and record["disposition"] == "awaiting_eligibility":
                    record.update(disposition="rejected", reason="interrupted_before_complete_eligibility")
            _save(path, state)
            return deepcopy(state)
        try:
            current = _observe(path, state, now_utc)
        except SamplingDeadlineExceeded:
            _save(path, state)
            return deepcopy(state)
        selected = None
        for candidate in state["identity"]["candidates"]:
            counts = state["candidates"][candidate["candidate_id"]]
            for family in candidate["families"]:
                remaining = candidate["family_layout_counts"][family] - counts["accepted_n_layouts_by_family"][family]
                if remaining:
                    selected = candidate, counts, family, remaining
                    break
            if selected is not None:
                break
        if selected is None:
            state["status"] = "complete"
            _save(path, state)
            return deepcopy(state)
        candidate, counts, family, remaining = selected
        attempt = counts["next_attempt_by_family"][family]
        if attempt >= 100:
            _stop(state, "outer_attempt_budget_exhausted", candidate_id=candidate["candidate_id"], family=family)
            _save(path, state)
            return deepcopy(state)
        call = {"n_layouts": remaining, "seed": state["identity"]["dataset_seed"] + 1000 * FAMILIES.index(family) + attempt,
                "partition": "confirmation", "families": [family], "excluded_layout_ids": state["global_layout_aliases"].copy(),
                "max_attempts": max(1000, 3000 * remaining), **deepcopy(state["identity"]["generator_parameters"])}
        batch_id = fingerprint({"identity_sha256": state["identity_sha256"], "candidate_id": candidate["candidate_id"], "family": family, "attempt": attempt})
        record = {"batch_id": batch_id, "candidate_id": candidate["candidate_id"], "family": family, "attempt": attempt,
                  "started_utc": current.isoformat(), "status": "reserved", "generator_call": call,
                  "generator_call_sha256": fingerprint(call), "candidate_definition_sha256": fingerprint(candidate),
                  "returned_pair_count": None, "partial_pair_bytes_available": None,
                  "historical_verification_before": historical_audit}
        state["in_progress"] = {"batch_id": batch_id, "attempt_index": len(state["attempts"])}
        state["attempts"].append(record)
        counts["next_attempt_by_family"][family] += 1
        _save(path, state)  # Reservation survives an abrupt interruption.
        records = []
        try:
            seconds = (_utc(state["effective_sampling_cutoff_utc"]) - current).total_seconds()
            with _bounded(seconds):
                returned = generator(**deepcopy(call))
            _require(isinstance(returned, list), "generator must return a pair list")
            returned = deepcopy(returned)
            record.update(status="generated", returned_pair_count=len(returned), returned_pairs_sha256=fingerprint(returned),
                          partial_pair_bytes_available=True)
            aliases = set(state["global_layout_aliases"])
            for index, pair in enumerate(returned):
                pair_record = {"batch_id": batch_id, "candidate_id": candidate["candidate_id"], "family": family,
                               "returned_index": index, "pair": pair, "pair_sha256": fingerprint(pair),
                               "disposition": "awaiting_eligibility", "reason": None, "builder_decision": None}
                try:
                    physical_aliases = _physical(pair, family, generator_call=call)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    physical_aliases = _reported_aliases(pair)
                    pair_record.update(disposition="rejected", reason="invalid_generated_physical_pair:" + str(exc))
                pair_record["physical_layout_aliases"] = sorted(physical_aliases)
                pair_record["global_aliases_before_pair"] = sorted(aliases)
                aliases.update(physical_aliases)
                records.append(pair_record)
                state["attempted_pairs"].append(pair_record)
            state["global_layout_aliases"] = sorted(aliases)
            _save(path, state)  # Preserve every returned pair before any callback failure.
            current = _observe(path, state, now_utc)
            record["historical_verification_after_generation"] = _verify_historical(state["identity"])
            _require(len(returned) == remaining, "generator returned a partial or overfull batch; terminal exhaustion")
            _require(all(item["disposition"] == "awaiting_eligibility" for item in records), "generator physical contract failed")
            seconds = (_utc(state["effective_sampling_cutoff_utc"]) - current).total_seconds()
            with _bounded(seconds):
                assessment = eligibility_builder(deepcopy(candidate), deepcopy(returned))
            record["builder_assessment"] = deepcopy(assessment)
            record["builder_assessment_sha256"] = fingerprint(assessment)
            if isinstance(assessment, dict) and isinstance(assessment.get("decisions"), list):
                all_reported = set(state["global_layout_aliases"])
                for decision in assessment["decisions"]:
                    all_reported.update(_reported_aliases(decision))
                state["global_layout_aliases"] = sorted(all_reported)
            _save(path, state)  # Even malformed returned decisions/aliases survive.
            _require(isinstance(assessment, dict) and assessment.get("eligibility_builder_sha256") == candidate["eligibility_builder_sha256"],
                     "eligibility builder fingerprint differs from frozen identity")
            decisions = assessment.get("decisions")
            _require(isinstance(decisions, list) and len(decisions) == len(returned), "one builder decision per returned pair required")
            _require(all(isinstance(item, dict) for item in decisions), "invalid builder decision")
            _require(len({item.get("pair_id") for item in decisions}) == len(decisions), "duplicate builder decision identity")
            decision_map = {item["pair_id"]: item for item in decisions}
            normalized = [_decision(decision_map[pair["pair_id"]], pair, candidate) for pair in returned]
            current = _observe(path, state, now_utc)
            record["historical_verification_after_eligibility"] = _verify_historical(state["identity"])
            # Apply the deterministic returned order; ALL rejected/control aliases
            # remain reserved globally, so another candidate cannot reclaim them.
            running_aliases = set(call["excluded_layout_ids"])
            for pair_record, decision in zip(records, normalized, strict=True):
                aliases = set(decision["layout_aliases"])
                overlapping = aliases & running_aliases
                pair_record.update(builder_decision=decision, builder_decision_sha256=fingerprint(decision),
                                   layout_aliases=sorted(aliases), conflicting_layout_aliases=sorted(overlapping))
                accepted = decision["eligible"] and not overlapping
                pair_record.update(disposition="accepted" if accepted else "rejected",
                                   reason=None if accepted else "previously_used_or_attempted_layout_alias" if overlapping else decision["reason"])
                if accepted:
                    counts["accepted_pair_ids"].append(pair_record["pair"]["pair_id"])
                    counts["accepted_n_layouts_by_family"][family] += 1
                running_aliases.update(aliases)
            state["global_layout_aliases"] = sorted(set(state["global_layout_aliases"]) | running_aliases)
            record.update(status="complete", finished_utc=current.isoformat(),
                          accepted_pair_count=sum(item["disposition"] == "accepted" for item in records),
                          rejected_pair_count=sum(item["disposition"] == "rejected" for item in records))
            state["in_progress"] = None
            if all(state["candidates"][c["candidate_id"]]["accepted_n_layouts_by_family"] == c["family_layout_counts"]
                   for c in state["identity"]["candidates"]):
                state["status"] = "complete"
            elif counts["next_attempt_by_family"][family] >= 100 and counts["accepted_n_layouts_by_family"][family] < candidate["family_layout_counts"][family]:
                _stop(state, "outer_attempt_budget_exhausted", candidate_id=candidate["candidate_id"], family=family)
        except Exception as exc:
            if state["status"] == "sampling":
                _stop(state, "sampling_callback_deadline_exhausted" if isinstance(exc, SamplingDeadlineExceeded) else "sampling_batch_terminal_failure",
                      batch_id=batch_id, error_type=type(exc).__name__, error=str(exc))
            record.update(status="terminal_failure", error_type=type(exc).__name__, error=str(exc))
            if record["returned_pair_count"] is None:
                record["partial_pair_bytes_available"] = False
            for item in records:
                if item["disposition"] == "awaiting_eligibility":
                    item.update(disposition="rejected", reason="terminal_batch_failure_before_complete_eligibility")
            # Keep the reserved batch identity as evidence, not a resumable seed.
        _save(path, state)
        return deepcopy(state)


def select_lexical_subset(pairs: Sequence[Mapping[str, Any]], *, families: Sequence[str]) -> dict:
    """Fixed same-layout 32-case descriptive supplement; never doubles N."""
    pairs, families = deepcopy(list(pairs)), list(families)
    _require(families == list(FAMILIES) or (len(families) == 1 and families[0] in FAMILIES), "invalid fixed lexical families")
    quotas = dict(zip(FAMILIES, QUOTAS[32], strict=True)) if len(families) == 3 else {families[0]: 32}
    seen, grouped = set(), {family: [] for family in families}
    for pair in pairs:
        _require(pair.get("family") in grouped, "lexical source family outside frozen profile")
        aliases = _physical(pair, pair["family"])
        _require(not aliases & seen, "duplicate lexical source layout aliases")
        seen.update(aliases)
        grouped[pair["family"]].append(pair)
    selected = []
    sort_keys = {}
    for family in families:
        _require(len(grouped[family]) >= quotas[family], "insufficient layouts for fixed lexical quota")
        def key(pair):
            layout_id = pair["layout_id"]
            return hashlib.sha256((layout_id + "\n" + SALT).encode()).hexdigest(), layout_id
        ordered = sorted(grouped[family], key=key)
        selected.extend(ordered[:quotas[family]])
        sort_keys.update({pair["layout_id"]: key(pair)[0] for pair in ordered})
    return {"format": "fixed_lexical_same_layout_subset_v1", "selection_salt": SALT,
            "families": families, "family_layout_counts": quotas, "family_weights": {f: quotas[f] / 32 for f in families},
            "source_pairs_sha256": fingerprint(pairs), "selected_pairs_sha256": fingerprint(selected),
            "selected_pairs": selected, "layout_ids": [pair["layout_id"] for pair in selected],
            "pair_ids": [pair["pair_id"] for pair in selected], "layout_sort_hashes": sort_keys,
            "n_layouts": 32, "same_weights_for_both_renderings": True, "additional_independent_layouts": 0,
            "selection_before_model_query": True, "model_queries": 0, "registered_experiment": False}
