#!/usr/bin/env python3
"""Inventory every registered search specification without loading a model.

Usage: python scripts/36_activation_causality_search_ledger.py --campaign DIR
       --output STEM

Writes new, immutable STEM.json and STEM.md. Reads only campaign configuration,
specs, datasets, result/attempt JSON, and retirement markers. A live campaign is
not locked: initial/final file inventories and per-file stat checks bound this
non-transactional observation. No action effects or forced scores are reported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any


OMITTED_JOB_FIELDS = frozenset({"recipient_text", "donor_text", "shared_base_text"})
SOURCE_ID_FIELDS = frozenset({"source_experiment_id", "source_dataset_or_experiment", "source_dataset"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
MANIFEST_DEFINITION = "SHA256 of sorted campaign-relative path:sha256 newline records for successfully read bytes"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def issue(code: str, **detail: Any) -> dict:
    return {"code": code, **detail}


def strict_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def stat_record(path: Path) -> dict:
    value = path.stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino, "device": value.st_dev}


def file_inventory(campaign: Path) -> dict[str, list[str]]:
    patterns = {"configuration": "campaign.json", "events": "events.jsonl", "specs": "specs/*.json",
                "datasets": "datasets/*.json", "results": "results/**/*.json",
                "retirements": "reviews/*_retired.json", "gates": "reviews/engine_validation_*.json"}
    return {kind: sorted(str(path.relative_to(campaign)) for path in campaign.glob(pattern)
                         if path.is_file()) for kind, pattern in patterns.items()}


def read_record(campaign: Path, relative: str) -> tuple[dict, Any]:
    """Hash and parse exactly the same bytes; preserve read/parse/race errors."""
    path = campaign / relative
    record: dict[str, Any] = {"path": relative, "read_started_at": now(), "issues": []}
    data = None
    try:
        record["stat_before_read"] = stat_record(path)
        raw = path.read_bytes()
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        record["size_bytes_read"] = len(raw)
        record["stat_after_read"] = stat_record(path)
        if record["stat_before_read"] != record["stat_after_read"]:
            record["issues"].append(issue("file_changed_during_read"))
        try:
            if relative == "events.jsonl":
                end = raw.rfind(b"\n") + 1
                prefix, trailing = raw[:end], raw[end:]
                record["complete_line_prefix_sha256"] = hashlib.sha256(prefix).hexdigest()
                record["complete_line_prefix_bytes"] = len(prefix)
                record["trailing_bytes_not_parsed"] = len(trailing)
                record["trailing_bytes_sha256"] = hashlib.sha256(trailing).hexdigest()
                data = []
                for line_number, line in enumerate(prefix.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line, object_pairs_hook=strict_object)
                        if not isinstance(event, dict) or not isinstance(event.get("kind"), str):
                            raise ValueError("event must be an object with string kind")
                        canonical(event)  # Reject nonfinite constants.
                        data.append({"line_number": line_number, "event": event})
                    except (ValueError, UnicodeError) as exc:
                        record["issues"].append(issue("invalid_complete_event_line", line_number=line_number, message=str(exc)))
            else:
                data = json.loads(raw, object_pairs_hook=strict_object)
                canonical(data)  # Reject nonfinite constants while retaining the raw-file hash.
        except (ValueError, UnicodeError) as exc:
            record["issues"].append(issue("invalid_json", message=str(exc)))
            data = None
    except OSError as exc:
        record["issues"].append(issue("file_read_error", message=str(exc)))
    record["read_finished_at"] = now()
    return record, data


def manifest(records: list[dict]) -> dict:
    hashed = sorted((record["path"], record["sha256"]) for record in records if "sha256" in record)
    text = "".join(f"{path}:{digest}\n" for path, digest in hashed)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "definition": MANIFEST_DEFINITION,
            "n_listed_files": len(records), "n_hashed_files": len(hashed),
            "unhashed_paths": [record["path"] for record in records if "sha256" not in record]}


def inventory_values(jobs: list[dict], field: str) -> list[dict]:
    counts = Counter(canonical(job.get(field)) for job in jobs)
    return [{"value": json.loads(value), "planned_jobs": count} for value, count in sorted(counts.items())]


def source_references(value: Any, location: str = "") -> list[dict]:
    """Keep explicit lineage, not similarities inferred from names or outcomes."""
    found: dict[tuple[str, str], int] = Counter()

    def walk(item: Any, pointer: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_pointer = f"{pointer}.{key}" if pointer else key
                if key in SOURCE_ID_FIELDS and isinstance(child, str):
                    found[(child, child_pointer)] += 1
                elif key not in {"recipient_text", "donor_text", "shared_base_text"}:
                    walk(child, child_pointer)
        elif isinstance(item, list):
            for child in item:
                walk(child, pointer + "[]")

    walk(value, location)
    return [{"source_id": identifier, "declared_at": pointer, "occurrences": count}
            for (identifier, pointer), count in sorted(found.items())]


def layout_inventory(spec: dict, jobs: list[dict]) -> dict:
    primary = sorted({j["layout_id"] for j in jobs if isinstance(j.get("layout_id"), str)})
    aliases = set(primary)

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if key == "layout_aliases" and isinstance(child_key, str):
                    aliases.add(child_key)
                collect(child, child_key)
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str) and key in {"layout_id", "layout_ids", "variant_layout_ids", "layout_aliases"}:
            aliases.add(value)

    collect(spec.get("layout_aliases", {}), "layout_aliases")
    for job in jobs:
        collect(job.get("layout_aliases", []), "layout_aliases")
        metadata = job.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("pair", "source_pair"):
                collect(metadata.get(key, {}))
    return {"n_unique_job_layout_ids": len(primary), "unique_job_layout_ids": primary,
            "n_recorded_job_layout_ids_and_aliases": len(aliases),
            "recorded_job_layout_ids_and_aliases": sorted(aliases),
            "interpretation": "Recorded identifiers, not independently sampled environments; cuts, endpoints and ranks may reuse layouts."}


def validate_row(row: Any, job: dict | None, spec_hash: str | None, experiment_id: str,
                 filename: str, *, attempt: bool = False) -> list[dict]:
    problems = []
    if not isinstance(row, dict):
        return [issue("result_not_json_object")]
    identifier = row.get("job_id")
    if not isinstance(identifier, str) or not identifier:
        problems.append(issue("missing_or_invalid_result_job_id"))
    if not attempt and filename != identifier:
        problems.append(issue("result_filename_job_id_mismatch", filename_stem=filename, row_job_id=identifier))
    if row.get("experiment_id") != experiment_id:
        problems.append(issue("result_experiment_id_mismatch", expected=experiment_id, actual=row.get("experiment_id")))
    if row.get("spec_sha256") != spec_hash:
        problems.append(issue("result_spec_sha256_mismatch", expected=spec_hash, actual=row.get("spec_sha256")))
    if job is None:
        problems.append(issue("result_job_not_in_registered_spec", job_id=identifier))
    else:
        for key, expected in job.items():
            if key in OMITTED_JOB_FIELDS:
                continue
            if key not in row or canonical(row[key]) != canonical(expected):
                problems.append(issue("frozen_job_field_mismatch", field=key,
                                      expected_value_sha256=fingerprint(expected),
                                      actual_value_sha256=fingerprint(row[key]) if key in row else None))
    if not isinstance(row.get("status"), str) or row["status"] not in {"complete", "error"}:
        problems.append(issue("unknown_technical_status", status=row.get("status")))
    for key in ("engine_key", "execution_key"):
        if not isinstance(row.get(key), str) or not SHA256.fullmatch(row[key]):
            problems.append(issue("missing_or_invalid_execution_identity", field=key))
    if row.get("status") == "complete" and row.get("generation_status") == "technical_error":
        problems.append(issue("technical_complete_conflicts_with_technical_error"))
    return problems


def summarize_spec(campaign: Path, spec_path: str, records: dict, objects: dict,
                   snapshot: dict) -> dict:
    spec_record, spec = records[spec_path], objects[spec_path]
    identifier = Path(spec_path).stem
    entry: dict[str, Any] = {"experiment_id": identifier, "spec_file": spec_record,
                            "spec_sha256": spec_record.get("sha256"), "issues": list(spec_record["issues"])}
    if not isinstance(spec, dict):
        entry["issues"].append(issue("spec_not_json_object"))
        spec = {}
    if spec.get("experiment_id") != identifier:
        entry["issues"].append(issue("spec_filename_experiment_id_mismatch", declared=spec.get("experiment_id")))
    if not IDENTIFIER.fullmatch(identifier):
        entry["issues"].append(issue("invalid_spec_filename_identifier"))
    entry["partition"] = spec.get("partition")
    if not isinstance(entry["partition"], str) or entry["partition"] not in {"historical", "discovery", "confirmation"}:
        entry["issues"].append(issue("invalid_spec_partition"))
    jobs_raw = spec.get("jobs")
    if not isinstance(jobs_raw, list):
        entry["issues"].append(issue("spec_jobs_not_list"))
        jobs_raw = []
    jobs = [job for job in jobs_raw if isinstance(job, dict)]
    job_map = {}
    for index, job in enumerate(jobs_raw):
        if not isinstance(job, dict):
            entry["issues"].append(issue("job_not_object", index=index))
            continue
        job_id = job.get("job_id")
        expected = fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24]
        if job_id != expected:
            entry["issues"].append(issue("noncanonical_frozen_job_id", index=index, job_id=job_id, expected=expected))
        if not isinstance(job_id, str) or not job_id:
            continue
        if job_id in job_map:
            entry["issues"].append(issue("duplicate_frozen_job_id", job_id=job_id))
            continue
        job_map[job_id] = job
        if job.get("experiment_id") != identifier or job.get("partition") != spec.get("partition"):
            entry["issues"].append(issue("frozen_job_experiment_or_partition_mismatch", job_id=job_id))
    entry["inventory"] = {field: inventory_values(jobs, field)
                          for field in ("condition", "family", "seed", "max_new_tokens", "temperature", "natural")}
    entry["layouts"] = layout_inventory(spec, jobs)
    entry["source_references"] = source_references(spec)
    entry["declared_source_provenance"] = {
        key: value for key, value in spec.items()
        if key in {"sources", "dataset_sha256", "source_spec_sha256", "source_native_spec_sha256",
                   "source_spec_fingerprint", "source_pairs_sha256", "source_controlled_spec_fingerprint", "source_offset"}}
    entry["declared_source_provenance_note"] = "Declared fields are preserved verbatim; dataset_sha256 and fingerprints are not assumed to be raw-file SHA256 values."
    retirement_path = f"reviews/{identifier}_retired.json"
    entry["retired"] = retirement_path in snapshot["retirements"]
    entry["retirement"] = {"file": records[retirement_path], "record": objects[retirement_path]} if entry["retired"] else None
    if entry["retired"] and not isinstance(objects[retirement_path], dict):
        entry["issues"].append(issue("retirement_marker_not_json_object", path=retirement_path))
    if entry["retired"] and records[retirement_path]["issues"]:
        entry["issues"].append(issue("retirement_marker_unreadable_or_changed", path=retirement_path))
    entry["scientific_use"] = False if entry["retired"] else None
    root = Path("results") / identifier
    current = [path for path in snapshot["results"] if Path(path).parent == root]
    attempts = [path for path in snapshot["results"] if Path(path).is_relative_to(root / "attempts")]
    other = [path for path in snapshot["results"] if Path(path).is_relative_to(root) and path not in current and path not in attempts]
    row_records, observed_ids, validated_ids, statuses = [], [], [], Counter()
    identities = Counter()
    for path in current:
        record, row = records[path], objects[path]
        row_id = row.get("job_id") if isinstance(row, dict) else None
        job = job_map.get(row_id) if isinstance(row_id, str) else None
        problems = list(record["issues"]) + validate_row(row, job, entry["spec_sha256"], identifier, Path(path).stem)
        if isinstance(row, dict):
            statuses[canonical(row.get("status"))] += 1
            identity = {key: row.get(key) for key in ("engine_key", "execution_key", "inference_configuration")}
            identities[canonical(identity)] += 1
            if isinstance(row_id, str):
                observed_ids.append(row_id)
            if not problems and row_id is not None:
                validated_ids.append(row_id)
        row_records.append({**record, "job_id": row_id, "technical_status": row.get("status") if isinstance(row, dict) else None,
                            "validation_passed": not problems, "issues": problems})
    duplicates = sorted(key for key, count in Counter(observed_ids).items() if count > 1)
    if duplicates:
        entry["issues"].append(issue("duplicate_current_result_job_ids", job_ids=duplicates))
    present_stems = {Path(path).stem for path in current}
    missing = sorted(set(job_map) - present_stems)
    extra = sorted(present_stems - set(job_map))
    if extra:
        entry["issues"].append(issue("unexpected_current_result_filenames", filename_stems=extra))
    if len(identities) > 1:
        entry["issues"].append(issue("mixed_current_execution_identities", n_identities=len(identities)))
    errors = sum(isinstance(objects[path], dict) and (objects[path].get("status") == "error"
                 or objects[path].get("generation_status") == "technical_error") for path in current)
    consistent = not entry["issues"] and all(record["validation_passed"] for record in row_records)
    complete = bool(jobs_raw) and consistent and not missing and not extra and len(current) == len(jobs_raw) and errors == 0 and all(objects[path].get("status") == "complete" for path in current)
    entry["operational_counts"] = {
        "planned_job_definitions": len(jobs_raw), "unique_validly_named_job_ids": len(job_map),
        "current_saved_files": len(current), "current_saved_planned_filename_count": len(present_stems & set(job_map)),
        "current_technical_complete_rows": statuses[canonical("complete")], "current_technical_error_rows": errors,
        "current_unreadable_or_nonobject_rows": sum(not isinstance(objects[path], dict) for path in current),
        "current_consistent_rows": len(validated_ids), "current_inconsistent_rows": sum(not r["validation_passed"] for r in row_records),
        "missing_job_files": len(missing), "unexpected_current_job_files": len(extra),
        "preserved_attempt_files": len(attempts), "other_preserved_nested_result_files": len(other)}
    entry["missing_job_ids"] = missing
    entry["current_result_files"] = row_records
    entry["current_results_manifest"] = manifest(row_records)
    entry["execution_identity_inventory"] = [{**json.loads(value), "current_rows": count} for value, count in sorted(identities.items())]
    entry["current_cohort_consistency_passed"] = consistent
    entry["current_cohort_complete_consistent"] = complete
    entry["operational_state"] = ("retired" if entry["retired"] else "registered_no_current_saved_rows" if not current
                                  else "complete_current_cohort" if complete else "incomplete_or_inconsistent_current_cohort")
    entry["no_saved_rows_interpretation"] = "No current executions are recorded for these job definitions; this does not classify source/preparation roles or imply failure." if not current else None
    entry["completion_status_counts"] = None
    entry["completion_status_by_condition"] = None
    entry["completion_summary_withheld_reason"] = "retired_scientific_use_false" if entry["retired"] else "cohort_not_complete_and_consistent"
    if complete and not entry["retired"]:
        status_counts = Counter()
        condition_counts: dict[str, Counter] = defaultdict(Counter)
        for path in current:
            row = objects[path]
            status = row.get("generation_status")
            expected = {"complete", "truncated", "malformed"} if row.get("natural") is True else {"not_requested"}
            if not isinstance(status, str) or status not in expected:
                entry["issues"].append(issue("invalid_completed_cohort_generation_status", path=path))
            else:
                status_counts[status] += 1
                condition_counts[row["condition"]][status] += 1
        if not entry["issues"]:
            entry["completion_status_counts"] = dict(sorted(status_counts.items()))
            entry["completion_status_by_condition"] = {key: dict(sorted(value.items())) for key, value in sorted(condition_counts.items())}
            entry["completion_summary_withheld_reason"] = None
        else:
            entry["current_cohort_consistency_passed"] = False
            entry["current_cohort_complete_consistent"] = False
            entry["operational_state"] = "incomplete_or_inconsistent_current_cohort"
            entry["completion_summary_withheld_reason"] = "invalid_completion_status"
    attempt_records = []
    for path in attempts + other:
        record, row = records[path], objects[path]
        row_id = row.get("job_id") if isinstance(row, dict) else None
        job = job_map.get(row_id) if isinstance(row_id, str) else None
        problems = list(record["issues"]) + validate_row(row, job, entry["spec_sha256"], identifier, Path(path).stem, attempt=True)
        attempt_records.append({**record, "job_id": row_id, "technical_status": row.get("status") if isinstance(row, dict) else None,
                                "error": row.get("error") if isinstance(row, dict) else None,
                                "category": "preserved_attempt" if path in attempts else "other_nested_result",
                                "execution_key": row.get("execution_key") if isinstance(row, dict) else None,
                                "engine_key": row.get("engine_key") if isinstance(row, dict) else None,
                                "issues": problems})
    entry["preserved_attempts_and_other_nested_results"] = attempt_records
    entry["preserved_attempts_manifest"] = manifest([record for record in attempt_records if record["category"] == "preserved_attempt"])
    return entry


def build_ledger(campaign: Path) -> dict:
    campaign = Path(campaign).resolve()
    started = now()
    initial = file_inventory(campaign)
    if not initial["configuration"] or not (campaign / "specs").is_dir():
        raise ValueError("Require campaign.json and an existing specs directory")
    records, objects = {}, {}
    for relative in sorted({path for paths in initial.values() for path in paths}):
        records[relative], objects[relative] = read_record(campaign, relative)
    experiments = [summarize_spec(campaign, path, records, objects, initial) for path in initial["specs"]]
    registered = {entry["experiment_id"]: entry for entry in experiments}
    for entry in experiments:
        for reference in entry["source_references"]:
            source_id = reference["source_id"]
            candidates = [f"specs/{source_id}.json", f"datasets/{source_id}.json"] if IDENTIFIER.fullmatch(source_id) else []
            reference["resolved_campaign_files"] = [{"path": path, "sha256": records[path].get("sha256")}
                                                    for path in candidates if path in records]
            reference["registered_spec_present"] = source_id in registered
        entry["referenced_by_registered_specs"] = sorted({other["experiment_id"] for other in experiments
            if any(ref["source_id"] == entry["experiment_id"] for ref in other["source_references"])})
    overlaps = []
    for index, left in enumerate(experiments):
        for right in experiments[index + 1:]:
            shared = sorted(set(left["layouts"]["unique_job_layout_ids"]) & set(right["layouts"]["unique_job_layout_ids"]))
            aliases = sorted(set(left["layouts"]["recorded_job_layout_ids_and_aliases"]) & set(right["layouts"]["recorded_job_layout_ids_and_aliases"]))
            if shared or aliases:
                overlaps.append({"left": left["experiment_id"], "right": right["experiment_id"],
                                 "shared_job_layout_ids": shared, "shared_recorded_ids_or_aliases": aliases})
    orphan_paths = [path for path in initial["results"] if len(Path(path).parts) < 3 or Path(path).parts[1] not in registered]
    orphan_records = [{**records[path], "issue": "result_file_has_no_registered_spec",
                       "job_id": objects[path].get("job_id") if isinstance(objects[path], dict) else None,
                       "technical_status": objects[path].get("status") if isinstance(objects[path], dict) else None}
                      for path in orphan_paths]
    final = file_inventory(campaign)
    changes = {kind: {"added": sorted(set(final[kind]) - set(initial[kind])),
                      "removed": sorted(set(initial[kind]) - set(final[kind]))} for kind in initial}
    changed_files = []
    for path, record in records.items():
        try:
            record["stat_at_snapshot_end"] = stat_record(campaign / path)
            if record.get("stat_after_read") != record["stat_at_snapshot_end"]:
                changed_files.append(path)
        except OSError:
            changed_files.append(path)
    unstable = set(changed_files) | {p for record in records.values() if record["issues"] for p in [record["path"]]}
    # A detected result/spec mutation or membership race suppresses even complete
    # cohort completion counts. There is no lock and no point-in-time guarantee.
    for entry in experiments:
        identifier = entry["experiment_id"]
        relevant = {entry["spec_file"]["path"], f"reviews/{identifier}_retired.json"}
        relevant.update(record["path"] for record in entry["current_result_files"])
        membership = [path for kind in ("specs", "results", "retirements") for action in ("added", "removed")
                      for path in changes[kind][action]
                      if path in relevant or Path(path).is_relative_to(Path("results") / identifier)]
        detected = sorted((relevant & unstable) | set(membership))
        entry["snapshot_changes_affecting_cohort"] = detected
        if detected:
            entry["issues"].append(issue("snapshot_changed_during_read", paths=detected))
            entry["completion_status_counts"] = None
            entry["completion_status_by_condition"] = None
            entry["completion_summary_withheld_reason"] = "snapshot_changed_during_read"
            entry["current_cohort_complete_consistent"] = False
            entry["current_cohort_consistency_passed"] = False
            entry["operational_state"] = "retired_snapshot_changed" if entry["retired"] else "snapshot_changed_during_read"
    unregistered_at_end = sorted(set(final["specs"]) - set(initial["specs"]))
    events = objects.get("events.jsonl")
    events = events if isinstance(events, list) else []
    event_inventory = {
        "file": records.get("events.jsonl"), "n_parsed_complete_lines": len(events),
        "kind_counts": dict(sorted(Counter(row["event"]["kind"] for row in events).items())),
        "technical_preflight_gate_stop_or_error_events": [row for row in events if any(
            word in row["event"]["kind"].lower() for word in ("technical", "preflight", "gate", "validation", "stop", "fail", "error"))
            or row["event"].get("error") is not None or row["event"].get("status") in ("error", "failed", "stopped", "technical_error")],
        "interpretation": "Operational events are not behavioral trials. Hash covers captured complete-line prefix; any trailing bytes are hashed separately and not parsed. Appends after capture require a later snapshot."}
    gates = []
    for path in initial["gates"]:
        value = objects[path]
        validation = value.get("validation") if isinstance(value, dict) else None
        gates.append({"file": records[path], "recorded_passed": validation.get("passed") if isinstance(validation, dict) else None,
                      "engine_key": value.get("engine_key") if isinstance(value, dict) else None,
                      "recorded_at": value.get("time") if isinstance(value, dict) else None})
    return {
        "schema_version": 1, "report_kind": "registered_search_inventory", "campaign": str(campaign),
        "snapshot": {"started_at": started, "finished_at": now(), "locking": "none; model worker not interrupted",
                     "atomic_campaign_snapshot": False, "initial_file_inventory": initial,
                     "final_file_inventory": final, "inventory_changes": changes,
                     "read_files_changed_before_snapshot_end": sorted(changed_files),
                     "new_spec_files_not_read": unregistered_at_end,
                     "interpretation": "Counts describe initially listed files read during this interval. Stat/list checks detect ordinary races but cannot prove an atomic snapshot or exclude all concurrent rewrites."},
        "provenance": {"script_path": str(Path(__file__).resolve()),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "python_version": platform.python_version(), "model_loaded": False,
                       "campaign_configuration_file": records["campaign.json"]},
        "definitions": {
            "registry": "Every specs/*.json file present in the initial listing, including retired and unexecuted/source/preparation definitions; prospective plan entries without a registered spec are not specs.",
            "completion_counts": "Only complete, consistent, nonretired, observed-stable cohorts receive generation-status counts. Partial rows receive operational counts only, never native outcomes, forced scores or target effects.",
            "technical_complete": "Raw status=complete is an operational execution flag, not a parsed native action or a claim of scientific validity.",
            "attempts": "Preserved attempts are a separate inventory and never additional trials; unknown nested result files and orphan results remain visible.",
            "frozen_row_validation": "Canonical job ID is SHA256(canonical full job excluding job_id)[:24]. Every frozen field is compared except recipient_text, donor_text and shared_base_text, which the runner omits. Added runtime fields are allowed.",
            "layout_ids": "Recorded unique job layout IDs and aliases do not establish independent samples. Do not pool conditions, cuts, seeds, ranks or endpoint extensions.",
            "overall_completion": "No global planned/saved ratio, pooled target effect, campaign-completion claim or automatic failure label for zero-result source specs is supplied."},
        "registry_counts": {"registered_spec_files_at_start": len(experiments),
                            "retired_spec_files": sum(entry["retired"] for entry in experiments),
                            "spec_files_with_no_current_saved_rows": sum(not entry["current_result_files"] for entry in experiments)},
        "registered_specs": experiments, "recorded_layout_overlaps": overlaps,
        "operational_event_inventory": event_inventory,
        "recorded_engine_gate_inventory": {"files": gates, "interpretation": "Recorded pass flags and file hashes only; this ledger does not independently certify gates or retroactively validate earlier executions."},
        "dataset_file_inventory": [records[path] for path in initial["datasets"]],
        "orphan_result_files": orphan_records,
        "unmatched_retirement_markers": [records[path] for path in initial["retirements"] if Path(path).name.removesuffix("_retired.json") not in registered],
        "all_read_files_manifest": manifest(list(records.values())),
    }


def markdown(ledger: dict) -> str:
    campaign = Path(ledger["campaign"])

    def link(label: str, relative: str) -> str:
        return f"[{label}](<{campaign / relative}>)"

    lines = ["# Registered search inventory", "",
             f"Observed {ledger['snapshot']['started_at']} to {ledger['snapshot']['finished_at']}. "
             "This is an unlocked file snapshot, not a campaign-completion assessment.", "",
             "All registered specs are retained, including retired and unexecuted source/preparation definitions. "
             "Zero saved rows do not imply failure. Layout counts are unique recorded job IDs, not independent environments. "
             "Attempts are separate from current rows. No target effects or forced scores are pooled or reported.", "",
             "| Registered specification | Partition | Operational state | Planned | Saved files | Technical complete | Technical errors | Missing files | Unique layout IDs |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for entry in ledger["registered_specs"]:
        count = entry["operational_counts"]
        lines.append(f"| {link(entry['experiment_id'], entry['spec_file']['path'])} | {entry['partition']} | {entry['operational_state']} | "
                     f"{count['planned_job_definitions']} | {count['current_saved_files']} | {count['current_technical_complete_rows']} | "
                     f"{count['current_technical_error_rows']} | {count['missing_job_files']} | {entry['layouts']['n_unique_job_layout_ids']} |")
    lines += ["", "## Per-specification provenance", ""]
    for entry in ledger["registered_specs"]:
        count = entry["operational_counts"]
        lines += [f"### {entry['experiment_id']}", "", f"Spec SHA256: `{entry['spec_sha256']}`.", ""]
        for field, label in (("condition", "Conditions"), ("family", "Families"), ("seed", "Seeds"), ("max_new_tokens", "Token caps")):
            values = "; ".join(f"`{canonical(item['value'])}` × {item['planned_jobs']} planned jobs" for item in entry["inventory"][field])
            lines.append(f"{label}: {values or 'none' }.")
        sources = sorted({(ref["source_id"], target["path"]) for ref in entry["source_references"] for target in ref["resolved_campaign_files"]})
        unresolved = sorted({ref["source_id"] for ref in entry["source_references"] if not ref["resolved_campaign_files"]})
        lines += ["", "Explicit sources: " + (", ".join(link(identifier, path) for identifier, path in sources) or "none declared as registry IDs") + "."]
        if unresolved:
            lines.append("Unresolved declared source IDs: " + ", ".join(f"`{value}`" for value in unresolved) + ".")
        if entry["referenced_by_registered_specs"]:
            lines.append("Referenced by: " + ", ".join(f"`{value}`" for value in entry["referenced_by_registered_specs"]) + ".")
        if entry["retirement"]:
            lines.append("Retired; scientific use is disabled. " + link("Retirement record", entry["retirement"]["file"]["path"]) + ".")
            retirement = entry["retirement"]["record"]
            lines.append("Retirement reason: " + str(retirement.get("reason", "see record") if isinstance(retirement, dict) else "unreadable or malformed marker; see JSON issue inventory"))
        if entry["no_saved_rows_interpretation"]:
            lines.append(entry["no_saved_rows_interpretation"])
        if entry["completion_status_counts"] is None:
            lines.append(f"Completion outcomes withheld: `{entry['completion_summary_withheld_reason']}`. Operational counts only.")
        else:
            lines.append("Complete-cohort generation status counts: " + canonical(entry["completion_status_counts"]) + ". These are completion statuses, not target effects.")
        lines.append(f"Current-result manifest: `{entry['current_results_manifest']['sha256']}`; "
                     f"{entry['current_results_manifest']['n_hashed_files']} hashed files. "
                     f"Preserved attempts: {count['preserved_attempt_files']}; other nested result files: {count['other_preserved_nested_result_files']}.")
        lines.append("Execution keys: " + (", ".join(f"`{value['execution_key']}` ({value['current_rows']} current rows)" for value in entry["execution_identity_inventory"]) or "none") + ".")
        issue_count = len(entry["issues"]) + sum(len(record["issues"]) for record in entry["current_result_files"])
        lines.append(f"Consistency issues: {issue_count}; inconsistent current rows: {count['current_inconsistent_rows']}. "
                     "The JSON retains each issue, missing job ID, full file hash, layout ID/alias and source-reference field.")
        lines.append("")
    event_inventory = ledger["operational_event_inventory"]
    lines += ["## Operational history and engine gates", "",
              "Operational event kinds: `" + canonical(event_inventory["kind_counts"]) + "`.", "",
              "Technical/preflight/gate/stop/error events are separate from per-job failures:", ""]
    for event in event_inventory["technical_preflight_gate_stop_or_error_events"]:
        lines.append(f"- Captured complete line {event['line_number']}: `{canonical(event['event'])}`")
    if not event_inventory["technical_preflight_gate_stop_or_error_events"]:
        lines.append("None in the captured complete-line prefix; this is not proof that no startup error occurred.")
    if event_inventory["file"]:
        record = event_inventory["file"]
        lines += ["", f"Event prefix SHA256: `{record.get('complete_line_prefix_sha256')}`; "
                  f"trailing bytes not parsed: {record.get('trailing_bytes_not_parsed')}. " + event_inventory["interpretation"], ""]
    for gate in ledger["recorded_engine_gate_inventory"]["files"]:
        lines.append(f"- {link('Recorded gate', gate['file']['path'])}: recorded passed=`{gate['recorded_passed']}`; SHA256 `{gate['file'].get('sha256')}`.")
    lines += ["", ledger["recorded_engine_gate_inventory"]["interpretation"], "", "## Snapshot and interpretation limits", "", MANIFEST_DEFINITION + ".", "",
              ledger["snapshot"]["interpretation"], "",
              "Inventory additions/removals: `" + canonical(ledger["snapshot"]["inventory_changes"]) + "`.", "",
              f"Changed read files: {len(ledger['snapshot']['read_files_changed_before_snapshot_end'])}; "
              f"orphan result files: {len(ledger['orphan_result_files'])}. New specs appearing during the read are listed in JSON and require a new snapshot.", "",
              "Declared source/dataset fingerprints are distinct from actual file SHA256 values. "
              "Recorded overlap is provided in JSON; no independence or global search-completion inference follows from the inventory.", "",
              f"Script SHA256: `{ledger['provenance']['script_sha256']}`.", ""]
    return "\n".join(lines)


def write_outputs(stem: Path, ledger: dict) -> tuple[Path, Path]:
    stem = stem.resolve()
    if stem.suffix in {".json", ".md"}:
        stem = stem.with_suffix("")
    paths = [stem.with_suffix(".json"), stem.with_suffix(".md")]
    stem.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(stem) + ".write.lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = []
    try:
        if any(path.exists() for path in paths):
            raise FileExistsError("Refusing to overwrite a search ledger; choose a new --output stem")
        payloads = [json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n", markdown(ledger)]
        for path, payload in zip(paths, payloads, strict=True):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=stem.parent, prefix=path.name + ".", suffix=".partial", delete=False) as handle:
                temporary.append(Path(handle.name))
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for source, destination in zip(temporary, paths, strict=True):
            os.link(source, destination)
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)
        for path in temporary:
            path.unlink(missing_ok=True)
    return paths[0], paths[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh immutable JSON/Markdown output stem")
    args = parser.parse_args()
    ledger = build_ledger(args.campaign)
    outputs = write_outputs(args.output, ledger)
    print(json.dumps({"json": str(outputs[0]), "markdown": str(outputs[1]),
                      "registered_specs": ledger["registry_counts"]["registered_spec_files_at_start"],
                      "snapshot_started_at": ledger["snapshot"]["started_at"],
                      "snapshot_finished_at": ledger["snapshot"]["finished_at"],
                      "campaign_completion_claim": False}))


if __name__ == "__main__":
    main()
