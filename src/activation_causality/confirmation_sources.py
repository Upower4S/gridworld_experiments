"""Authenticate complete discovery sources for the prospective confirmation.

All requested filename inventories are checked before opening any outcome or
audit. This is a source/authenticity gate, not scientific candidate selection.
It intentionally fails closed if a required source is technically incomplete;
terminal search failures need an explicit separately reviewed closure record.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .registry import file_sha256, fingerprint

CONTEXTUAL_SOURCE = "contextual_input_native_pilot_v1"
SUBSPACE_SOURCES = tuple(f"subspace_{family}_l0_r{rank}_native_v1"
                        for family in ("wall", "key", "door") for rank in (1, 4))
REQUIRED_BATTERY = (CONTEXTUAL_SOURCE, *SUBSPACE_SOURCES)
NATURAL_NORM_ADDENDUM = "outputs/activation_causality/campaign_20260912/reviews/contextual_confirmation_natural_norm_addendum_v1.json"
NATURAL_NORM_ADDENDUM_SHA256 = "134a24bd99adba6df33713c4b7bfe63d8bd911ebb4a2ad744773d14f4ba8b652"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def _bound_path(repo: Path, relative: Any, expected: Any) -> Path:
    _require(isinstance(relative, str) and isinstance(expected, str)
             and re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "artifact requires a path and SHA256")
    path = (repo / relative).resolve()
    _require(path.is_relative_to(repo.resolve()) and path.is_file(), "source artifact is absent or outside the repository")
    _require(file_sha256(path) == expected, "source artifact changed: " + relative)
    return path


def _read_json_bytes(path: Path, expected: str | None = None) -> tuple[dict, str]:
    """Hash the exact bytes subsequently parsed; never authenticate a reopen."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    _require(expected is None or digest == expected, "parsed source artifact fingerprint mismatch: " + str(path))
    value = json.loads(raw)
    _require(isinstance(value, dict), "source artifact must contain a JSON object")
    return value, digest


def _native_inventory(campaign: Path, experiment_id: str) -> tuple[dict, Path, str, dict, list[Path]]:
    from .selection import _source_result_paths

    _require(_name(experiment_id), "invalid source experiment identity")
    path = campaign / "specs" / f"{experiment_id}.json"
    spec, digest = _read_json_bytes(path)
    _require(spec.get("experiment_id") == experiment_id and spec.get("partition") == "discovery",
             "source must be an unchanged registered discovery experiment")
    _require(not (campaign / "reviews" / f"{experiment_id}_retired.json").exists(), "retired source cannot select confirmation")
    jobs = {job["job_id"]: job for job in spec["jobs"]}
    _require(bool(jobs) and len(jobs) == len(spec["jobs"]), "source job identities must be unique and nonempty")
    for job in jobs.values():
        _require(job["job_id"] == fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24]
                 and job.get("experiment_id") == experiment_id and job.get("partition") == "discovery",
                 "source job identity or partition differs from its definition")
        _require(job.get("natural") is True and type(job.get("seed")) is int and job["seed"] == 42
                 and type(job.get("max_new_tokens")) is int and job["max_new_tokens"] == 4096
                 and type(job.get("temperature")) in {int, float} and job["temperature"] == 0.7,
                 "prospective selection requires the fixed native seed/cap/temperature source")
    return spec, path, digest, jobs, _source_result_paths(campaign, experiment_id, jobs)


def _audit_source(repo: Path, campaign: Path, spec: dict, spec_hash: str,
                  result_hashes: dict[str, str], binding: Mapping[str, str]) -> dict[str, Any]:
    _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, "source requires an exact independent-audit binding")
    path = _bound_path(repo, binding["path"], binding["sha256"])
    # Automated recomputations are useful, but they are not a substitute for
    # the independently reviewed named source artifact.
    expected_name = ("contextual_input_native_independent_audit_v1.json" if spec["experiment_id"] == CONTEXTUAL_SOURCE
                     else spec["experiment_id"].removesuffix("_v1") + "_independent_audit_v1.json")
    _require(path == (campaign / "reviews" / expected_name).resolve(), "source audit must be the named independent review artifact")
    audit, _ = _read_json_bytes(path, binding["sha256"])
    n_jobs = len(spec["jobs"])
    expected_raw = {str((campaign / relative).resolve().relative_to(repo.resolve())): digest
                    for relative, digest in result_hashes.items()}
    if spec["experiment_id"] == CONTEXTUAL_SOURCE:
        _require(audit.get("experiment_id") == spec["experiment_id"] and audit.get("spec_sha256") == spec_hash
                 and audit.get("n_results") == n_jobs == 132 and audit.get("technical_status_counts") == {"complete": n_jobs}
                 and audit.get("exact_forced_replay_input_hashes_patch_logs") == n_jobs,
                 "contextual audit does not prove the complete fixed native source")
        _require(audit.get("raw_result_hashes") == expected_raw, "contextual audit result manifest differs from every source control")
        prep = audit.get("preparation_checks", {})
        _require(prep.get("exact_pure_rebuild") is True and prep.get("all_alias_control_and_comparison_inventories_verified") is True,
                 "contextual source reconstruction/control audit is missing")
        identities = audit.get("identities", {})
        _require(set(identities) == {f"layer{layer}_{role}" for layer in (0, 10) for role in ("identity", "restoration_equivalent")}
                 and all(item.get("n") == item.get("entire_generated_ids_exact") == item.get("forced_q_exact") == 6
                         and item.get("max_delta_norm") == 0 for item in identities.values()),
                 "contextual whole-token identity audit is incomplete")
        inputs = expected_raw
    else:
        _require(spec["experiment_id"] in SUBSPACE_SOURCES
                 and audit.get("format") == "subspace_complete_independent_cohort_audit_v1",
                 "unsupported source or independent subspace audit format")
        cohort = audit.get("cohort", {})
        _require(cohort.get("experiment_id") == spec["experiment_id"] and cohort.get("spec_sha256") == spec_hash
                 and cohort.get("state") == "complete" and cohort.get("exact_complete_filename_inventory") is True
                 and cohort.get("planned_jobs") == cohort.get("saved_file_count") == cohort.get("exact_forced_native_replays") == n_jobs == 56
                 and cohort.get("remaining_planned_jobs") == 0 and cohort.get("technical_status_counts") == {"complete": n_jobs}
                 and cohort.get("required_norm_writes_revalidated") == 16,
                 "subspace audit does not prove the complete fixed native source")
        pair_ids = {job["pair_id"] for job in spec["jobs"]}
        identities = cohort.get("identity_checks", [])
        _require(len(identities) == 8 and {(item.get("pair_id"), item.get("condition")) for item in identities}
                 == {(pair, role) for pair in pair_ids for role in ("subspace_identity", "restoration_equivalent")}
                 and all(item.get("exact_entire_generated_identity") is True and item.get("exact_forced_identity") is True
                         for item in identities), "subspace whole-token identities are incomplete")
        norm = cohort.get("paired_applied_norm_checks", [])
        _require(len(norm) == 8 and {(item.get("pair_id"), item.get("dose")) for item in norm}
                 == {(pair, dose) for pair in pair_ids for dose in (0.5, 1.0)}
                 and all(item.get("expected_original_reference_identical") is True for item in norm),
                 "subspace intended/random common norm references are incomplete")
        inputs = audit.get("input_file_sha256")
        _require(isinstance(inputs, dict) and all(inputs.get(path) == digest for path, digest in expected_raw.items()),
                 "subspace audit does not bind every native source control")
    for relative, digest in inputs.items():
        _bound_path(repo, relative, digest)
    return {"path": str(path.relative_to(repo.resolve())), "sha256": binding["sha256"],
            "input_files_sha256": deepcopy(inputs)}


def read_complete_native_sources(campaign: Path, repo: Path, experiment_ids: Sequence[str],
                                 audit_bindings: Mapping[str, Any]) -> dict[str, dict]:
    """Read outcomes only after every requested source has exact filenames."""
    identifiers = list(experiment_ids)
    _require(bool(identifiers) and len(identifiers) == len(set(identifiers))
             and set(identifiers) == set(audit_bindings), "source and audit inventories differ")
    _require(set(identifiers) <= set(REQUIRED_BATTERY), "unsupported prospective native source")
    # All seven inventories precede the first result/audit read in battery mode.
    inventories = {identifier: _native_inventory(campaign, identifier) for identifier in identifiers}
    bundles, all_bound_files = {}, {}
    for identifier in identifiers:
        spec, path, spec_hash, jobs, expected_paths = inventories[identifier]
        rows, result_hashes = [], {}
        for result_path in expected_paths:
            result, result_digest = _read_json_bytes(result_path)
            job_id = result.get("job_id")
            _require(job_id in jobs and result_path.name == f"{job_id}.json", "source result must identify its canonical registered job")
            job = jobs[job_id]
            _require(result.get("spec_sha256") == spec_hash, "source result/spec fingerprint mismatch")
            for key, value in job.items():
                if key not in {"recipient_text", "donor_text"}:
                    _require(key in result and fingerprint(result[key]) == fingerprint(value),
                             "source result immutable job field mismatch: " + key)
            if "recipient_token_ids" in job:
                _require(result.get("input_token_sha256") == fingerprint(job["recipient_token_ids"]),
                         "source result authoritative input-token fingerprint mismatch")
            _require(result.get("status") == "complete", "source requires technical completion of every condition")
            status, action = result.get("generation_status"), result.get("native_action")
            _require(status in {"complete", "truncated", "malformed"}
                     and (action in {"UP", "DOWN", "LEFT", "RIGHT"} if status == "complete" else action is None),
                     "native source action/completion status is inconsistent")
            rows.append(result)
            result_hashes[str(result_path.relative_to(campaign))] = result_digest
        audit = _audit_source(repo, campaign, spec, spec_hash, result_hashes, audit_bindings[identifier])
        durations = [row.get("seconds") for row in rows]
        _require(all(type(value) in {int, float} and math.isfinite(value) and value > 0 for value in durations),
                 "whole-cohort throughput requires every measured native duration")
        bound = {str(path.resolve().relative_to(repo.resolve())): spec_hash,
                 **{str((campaign / relative).resolve().relative_to(repo.resolve())): digest
                    for relative, digest in result_hashes.items()},
                 audit["path"]: audit["sha256"]}
        for relative, digest in audit["input_files_sha256"].items():
            _require(relative not in bound or bound[relative] == digest, "audit and parsed source artifact bindings disagree")
            bound[relative] = digest
        for relative, digest in bound.items():
            _require(relative not in all_bound_files or all_bound_files[relative] == digest,
                     "requested source cohorts bind conflicting artifact bytes")
            all_bound_files[relative] = digest
        bundles[identifier] = {"experiment_id": identifier, "specification": spec, "source_spec_sha256": spec_hash,
                               "rows": rows, "source_result_hashes": result_hashes,
                               "source_result_manifest_sha256": fingerprint(result_hashes), "independent_audit": audit,
                               "n_jobs": len(rows), "mean_seconds_per_job": sum(durations) / len(durations),
                               "generation_status_counts": dict(Counter(row["generation_status"] for row in rows))}
    # Final GLOBAL bracket: an earlier source may change while a later source
    # is being processed. Check all exact inventories and every parsed/supporting
    # artifact after all requested cohorts, including the independent audits.
    for identifier in identifiers:
        _native_inventory(campaign, identifier)
    for relative, digest in all_bound_files.items():
        _bound_path(repo, relative, digest)
    return bundles


def read_closed_native_battery(campaign: Path, repo: Path, audit_bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Complete valid battery only; no early candidate selection or fast subset."""
    _require(set(audit_bindings) == set(REQUIRED_BATTERY), "selection requires contextual and all six native subspace source audits")
    sources = read_complete_native_sources(campaign, repo, REQUIRED_BATTERY, audit_bindings)
    means = {identifier: bundle["mean_seconds_per_job"] for identifier, bundle in sources.items()}
    return {"sources": sources, "complete_cohort_means_seconds": means,
            "minimum_budget_seconds_per_native_job": 1.25 * max(means.values()),
            "throughput_rule": "1.25 times largest complete source cohort mean, including every capped or malformed run"}


def existing_comparison_origin(source: Mapping[str, Any], treatment: str, control: str) -> dict[str, Any]:
    comparisons = [row for row in source["specification"]["comparisons"]
                   if row["treatment"] == treatment and row["control"] == control]
    _require(len(comparisons) == 1 and comparisons[0].get("predicted_direction") == "increase",
             "primary must identify one existing increasing source comparison")
    return {"kind": "existing_source_comparison", "provenance": {
        "source_experiment_id": source["experiment_id"], "source_spec_sha256": source["source_spec_sha256"],
        "source_comparison_id": comparisons[0]["id"],
        "source_result_manifest_sha256": source["source_result_manifest_sha256"],
        "source_independent_audit_path": source["independent_audit"]["path"],
        "source_independent_audit_sha256": source["independent_audit"]["sha256"],
        "discovery_measured": True}}


def prospective_natural_norm_origin(source: Mapping[str, Any], repo: Path, *, layer: int,
                                    adapter_path: str, adapter_sha256: str) -> dict[str, Any]:
    _require(source["experiment_id"] == CONTEXTUAL_SOURCE and type(layer) is int and layer in {0, 10},
             "only the contextual natural-norm random control has a prospective origin exception")
    _require(adapter_path == "src/activation_causality/frozen_contextual_input.py",
             "prospective control requires the explicit frozen contextual adapter")
    _bound_path(repo, NATURAL_NORM_ADDENDUM, NATURAL_NORM_ADDENDUM_SHA256)
    _bound_path(repo, adapter_path, adapter_sha256)
    return {"kind": "prospective_confirmation_control", "provenance": {
        "source_experiment_id": source["experiment_id"], "source_spec_sha256": source["source_spec_sha256"],
        "source_result_manifest_sha256": source["source_result_manifest_sha256"],
        "source_independent_audit_path": source["independent_audit"]["path"],
        "source_independent_audit_sha256": source["independent_audit"]["sha256"],
        "discovery_measured": False, "treatment": f"layer{layer}_natural", "control": f"layer{layer}_natural_norm_random",
        "addendum_path": NATURAL_NORM_ADDENDUM, "addendum_sha256": NATURAL_NORM_ADDENDUM_SHA256,
        "adapter_path": adapter_path, "adapter_sha256": adapter_sha256}}
