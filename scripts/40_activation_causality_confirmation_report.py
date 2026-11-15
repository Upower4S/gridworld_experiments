#!/usr/bin/env python3
"""One frozen global primary analysis plus a descriptive paired lexical report.

Default: refuse incomplete inventories before reading native rows. At cutoff:
retain every frozen slot and denominator without inferring from partial cohorts.
No model, readiness call, registration, or mutation of campaign inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
NATIVE = {"complete", "truncated", "malformed"}
FAMILIES = ("wall", "key_possession", "door_status")
TECHNICAL_EVENTS = {"confirmation_execution_lease_rejected", "model_load_snapshot_gate_failed", "technical_gate_failed",
                    "worker_stopped_full_write_technical_failure", "worker_stopped_repeated_errors"}
STATS = dict(n_bootstrap=10000, n_permutations=10000, seed=42, alpha=.05, minimum_effect=.10)
TEXT_FIELDS = {"recipient_text", "donor_text", "shared_base_text"}
HELPERS = ("analysis", "primary_family", "confirmation_contract", "confirmation_supervisor", "engine", "basis_runtime", "full_write_audit", "subspace_audit")
ANALYSIS_CONFIGURATION = {
    "alpha": .05, "candidate_decision": "both_components_required", "interval_adjustment": "bonferroni_simultaneous",
    "minimum_effect": .10, "missing_or_invalid_primary_p_value": 1.0,
    "mixed_family_constant_guard": "weighted_simultaneous_family_intervals", "mixed_family_constant_p_guard": "two_sided_bounded_layout_hoeffding",
    "mixed_family_resampling": "within_family_layout_bootstrap", "mixed_family_weighting": "frozen_family_layout_counts_over_total",
    "n_bootstrap": 10000, "n_test_bootstrap": 10000, "p_value_adjustment": "holm", "p_value_method": "centered_layout_bootstrap_plus_one",
    "primary_endpoint": "natural_target_action_intention_to_treat", "primary_family": "frozen_global_co_primary_registry",
    "seed": 42, "version": 4, "weighting": "equal_layout_then_equal_pair_seed",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def same(left, right):
    return fingerprint(left) == fingerprint(right)


def strict_json(raw):
    def unique(items):
        answer = {}
        for key, value in items:
            require(key not in answer, "duplicate JSON key")
            answer[key] = value
        return answer
    value = json.loads(raw, object_pairs_hook=unique)
    fingerprint(value)
    return value


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.basis_norm_hashes = {}
        self.lease_records = {}
        self.interpreter_binding = None

    def bytes(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.resolve(strict=True) == path, "canonical absolute input path required")
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "input changed during read")
        digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, "input hash mismatch: " + str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, "input changed between reads")
        self.hashes[str(path)] = digest
        return raw, digest

    def read(self, path, expected=None):
        raw, digest = self.bytes(path, expected)
        return strict_json(raw), digest

    def recheck(self):
        for path, digest in self.hashes.items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "input changed during report: " + path)


def _contract(campaign, freeze_id, inputs):
    from src.activation_causality.confirmation_contract import validate_registry_configurations, validate_primary_spec
    require(re.fullmatch(r"[A-Za-z0-9_-]+", freeze_id) is not None, "invalid freeze ID")
    frozen_path = campaign / "freezes" / (freeze_id + ".json")
    require(set((campaign / "freezes").glob("*.json")) == {frozen_path}, "one complete global freeze required")
    frozen, frozen_sha = inputs.read(frozen_path)
    require(frozen.get("confirmation_mode") == "co_primary_native_v4" and same(frozen.get("schema_version"), 4)
            and frozen.get("freeze_id") == freeze_id, "reviewed co-primary freeze required")
    plan, _ = inputs.read(Path(frozen["plan_path"]), frozen["plan_sha256"])
    require(Path(frozen["plan_path"]).is_relative_to(campaign / "planning"), "plan outside campaign planning")
    review, _ = inputs.read(Path(frozen["review_path"]), frozen["review_sha256"])
    require(review.get("approved") is True and review.get("freeze_id") == freeze_id
            and review.get("plan_sha256") == frozen["plan_sha256"]
            and review.get("plan_fingerprint") == fingerprint(plan), "exact plan approval missing")
    require(all(k in frozen and same(frozen[k], v) for k, v in plan.items()), "freeze shadows reviewed plan")
    if "technical_runtime_amendment" in plan:
        _technical_amendment(plan, inputs)
    config, config_sha = inputs.read(campaign / "campaign.json", frozen["campaign_sha256"])
    from src.activation_causality.basis_runtime import _snapshot
    _snapshot(frozen["model_identity"])
    require(frozen["model_identity"]["path"] == config["model_path"], "frozen model path differs from campaign")
    deadline = datetime.fromisoformat(config["deadline"])
    cutoff = datetime.fromisoformat(frozen["inference_cutoff"])
    require(deadline.tzinfo is not None and cutoff.tzinfo is not None and (deadline - cutoff).total_seconds() == 21600,
            "frozen absolute six-hour reporting reserve differs")
    registry = validate_registry_configurations(frozen["confirmation_primary_family"], frozen["candidate_configurations"])
    require(same(frozen["analysis_configuration"], ANALYSIS_CONFIGURATION), "frozen statistics differ from adopted protocol")
    require(registry["family_size"] == 2 * len(frozen["candidate_configurations"]), "two co-primaries per selected candidate required")
    specs, datasets = {}, {}
    for eid, item in frozen["planned_experiments"].items():
        require(re.fullmatch(r"[A-Za-z0-9_-]+", eid) is not None, "invalid planned experiment ID")
        require(not (campaign / "reviews" / (eid + "_retired.json")).exists(), "retired confirmation cannot be synthesized")
        for kind, folder in (("spec", "specs"), ("dataset", "datasets")):
            planned = Path(item[kind + "_path"])
            require(planned.is_relative_to(campaign / "planning"), "planned input outside planning")
            value, _ = inputs.read(planned, item[kind + "_sha256"])
            registered, _ = inputs.read(campaign / folder / (eid + ".json"), item[kind + "_sha256"])
            require(same(value, registered), "registered/planned input differs")
            require(value.get("experiment_id") == eid and value.get("partition") == "confirmation"
                    and value.get("candidate_id") == item["candidate_id"] and value.get("cohort_role") == item["cohort_role"], "planned input identity differs")
            (specs if kind == "spec" else datasets)[eid] = value
        spec = specs[eid]
        pairs = {p["pair_id"]: p for p in datasets[eid]["pairs"]}
        require(len(pairs) == len(datasets[eid]["pairs"]), "duplicate frozen physical pair")
        require(spec.get("freeze_id") == freeze_id and spec.get("cohort_complete") is True, "frozen cohort incomplete")
        require(len({j["job_id"] for j in spec["jobs"]}) == len(spec["jobs"]), "duplicate planned job")
        for job in spec["jobs"]:
            require(job["job_id"] == fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24], "noncanonical frozen job ID")
            require(job["experiment_id"] == eid and job["partition"] == "confirmation" and job.get("natural") is True
                    and type(job.get("seed")) is int and job["seed"] == 42 and same(job.get("max_new_tokens"), 4096)
                    and same(job.get("temperature"), .7), "frozen native job settings differ")
            require(job["pair_id"] in pairs and same(job["metadata"]["source_pair"], pairs[job["pair_id"]]), "frozen dataset/job physical lineage differs")
        require(set(pairs) == {j["pair_id"] for j in spec["jobs"]}, "unused or absent frozen dataset pair")
        if spec["cohort_role"] == "primary":
            validate_primary_spec(spec, frozen["candidate_configurations"][spec["candidate_id"]], registry)
        else:
            require(spec["cohort_role"] == "lexical_supplement", "unknown cohort role")
    primary = {s["candidate_id"]: eid for eid, s in specs.items() if s["cohort_role"] == "primary"}
    require(set(primary) == set(frozen["candidate_configurations"])
            and len(primary) == sum(s["cohort_role"] == "primary" for s in specs.values()), "one primary per frozen candidate required")
    _validate_supplements(frozen, specs)
    return {"freeze": frozen, "freeze_sha256": frozen_sha, "campaign_configuration": config,
            "campaign_sha256": config_sha, "specs": specs, "datasets": datasets, "primary_ids": primary}


def _technical_amendment(plan, inputs):
    ref = plan["technical_runtime_amendment"]
    amendment, _ = inputs.read(Path(ref["path"]), ref["sha256"])
    require(amendment.get("format") == "prequery_technical_launch_runtime_amendment_v1"
            and amendment.get("confirmation_queries") == 0 and amendment.get("resampling") is False
            and amendment.get("selection_or_experiment_changes") is False, "unsupported technical runtime amendment")
    references = {}
    for name in ("original_plan", "original_plan_scientific_review", "preparation", "runtime_promotion", "sampling_state", "supervisor_fix_review"):
        item = amendment[name]
        references[name], _ = inputs.read(Path(item["path"]), item["sha256"])
    original = references["original_plan"]
    require(fingerprint(original) == amendment["original_plan"]["fingerprint"], "original technical-amendment plan fingerprint differs")
    promotion = references["runtime_promotion"]
    change = promotion["source_change"]
    name = "src/activation_causality/confirmation_supervisor.py"
    require(change["path"] == name and original["implementation"][name] == change["before_sha256"]
            and plan["implementation"][name] == change["after_sha256"], "technical amendment changes unsupported runtime source")
    expected = deepcopy(original)
    expected["implementation"][name] = change["after_sha256"]
    expected["technical_runtime_amendment"] = deepcopy(ref)
    require(same(expected, plan), "technical amendment changed scientific or sampling fields")
    archive = Path(amendment["runtime_promotion"]["path"]).parent
    inputs.bytes(archive / "before" / name, change["before_sha256"])
    required_after = {**plan["implementation"], "scripts/30_activation_causality_campaign.py": plan["worker_entrypoint_sha256"]}
    require(same(promotion["after_runtime_files"], required_after), "technical amendment runtime archive inventory differs")
    for relative, digest in required_after.items():
        inputs.bytes(archive / "after" / relative, digest)
    for item in promotion.get("reviews", {}).values():
        inputs.bytes(Path(item["path"]), item["sha256"])


def _validate_supplements(frozen, specs):
    seen = set()
    for alt in specs.values():
        if alt["cohort_role"] != "lexical_supplement":
            continue
        cid = alt["candidate_id"]
        require(cid not in seen and frozen["supplements"][cid] is True, "unplanned/duplicate lexical supplement")
        seen.add(cid)
        parent = specs[alt["parent_experiment_id"]]
        require(parent["cohort_role"] == "primary" and parent["candidate_id"] == cid
                and alt["parent_spec_fingerprint"] == fingerprint(parent), "lexical parent fingerprint differs")
        subset = alt["lexical_subset"]
        require(alt["lexical_subset_sha256"] == fingerprint(subset), "lexical subset fingerprint differs")
        config = frozen["candidate_configurations"][cid]
        families = list(FAMILIES) if config["protocol_kind"] == "contextual_input" else [config["family"]]
        quotas = dict(zip(FAMILIES, (11, 11, 10), strict=True)) if len(families) == 3 else {families[0]: 32}
        weights = {family: quota / 32 for family, quota in quotas.items()}
        source_pairs = {j["pair_id"]: j["metadata"]["source_pair"] for j in parent["jobs"]}
        selected = []
        for family in families:
            choices = [p for p in source_pairs.values() if p["family"] == family]
            choices.sort(key=lambda p: (hashlib.sha256((p["layout_id"] + "\nactivation_lexical_v1:20260912").encode()).hexdigest(), p["layout_id"]))
            require(len(choices) >= quotas[family], "parent lacks lexical quota")
            selected.extend(choices[:quotas[family]])
        ids = {p["pair_id"] for p in selected}
        require(len(ids) == 32 and set(subset["pair_ids"]) == ids
                and same(alt["family_weights"], weights) and same(alt["family_layout_counts"], quotas), "lexical selection or mixture differs")
        ref = alt["standard_subset_reference"]
        require(ref["parent_spec_fingerprint"] == fingerprint(parent) and ref["experiment_id"] == parent["experiment_id"]
                and same(ref["family_weights"], weights) and ref["execution_jobs_added"] == 0, "standard subset reference differs")
        parent_jobs = {j["job_id"]: j for j in parent["jobs"] if j["pair_id"] in ids}
        require(same(ref["job_fingerprints"], {jid: fingerprint(j) for jid, j in parent_jobs.items()}), "parent job references differ")
        bound = [{k: j[k] for k in ("job_id", "pair_id", "layout_id", "condition", "seed")} for j in parent_jobs.values()]
        require(sorted(ref["job_bindings"], key=lambda j: j["job_id"]) == sorted(bound, key=lambda j: j["job_id"]), "standard reference job inventory differs")
        count = len(config["condition_inventory"])
        require(len(alt["jobs"]) == len(parent_jobs) == 32 * count, "lexical full-control inventory differs")
        actual = Counter((j["pair_id"], j["condition"], j["seed"]) for j in alt["jobs"])
        require(actual == Counter((j["pair_id"], j["condition"], j["seed"]) for j in parent_jobs.values()), "lexical pairs/conditions differ")
        for job in alt["jobs"]:
            original = parent_jobs[job["metadata"]["paired_standard_job_id"]]
            require(job["metadata"]["paired_standard_job_sha256"] == fingerprint(original)
                    and same(job["metadata"]["source_pair"], original["metadata"]["source_pair"])
                    and job["target_action"] == original["target_action"], "lexical physical/job lineage differs")
            require(len(job["selection"]["recipient_positions"]) == len(original["selection"]["recipient_positions"]), "lexical support width differs")
        require(all(c.get("confirmatory") is False and "primary_id" not in c for c in alt["comparisons"] + ref["comparisons"]), "lexical comparison claims a primary slot")
        definitions = {(b["treatment"], b["control"]) for b in config["primary_comparisons"].values()}
        treatment = next(iter(config["primary_comparisons"].values()))["treatment"]
        definitions.update({(treatment, "baseline"), ("counterfactual_text" if len(families) == 3 else "donor_text", "baseline")})
        for comparisons in (alt["comparisons"], ref["comparisons"]):
            require(definitions <= {(c["treatment"], c["control"]) for c in comparisons}, "lexical required descriptive reference missing")
            for comparison in comparisons:
                require(set(comparison["eligible_pair_ids"]) == ids and comparison["expected_n_layouts"] == 32
                        and comparison["expected_n_pairs"] == 32, "lexical comparison cohort differs")
                if len(families) == 3:
                    require(same(comparison["family_weights"], weights) and same(comparison["family_layout_counts"], quotas), "lexical comparison mixture differs")
    require(seen == {cid for cid, enabled in frozen["supplements"].items() if enabled}, "planned lexical supplement missing")


def _module_bindings(frozen, inputs):
    for relative, digest in frozen["implementation"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe frozen implementation path")
        inputs.bytes(REPO_ROOT / relative, digest)
    for name in HELPERS:
        module = importlib.import_module("src.activation_causality." + name)
        path = Path(module.__file__).resolve()
        relative = "src/activation_causality/" + name + ".py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(frozen["implementation"].get(relative) == digest, "loaded reporting dependency differs from freeze: " + relative)
        inputs.hashes[str(path)] = digest
    from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
    for relative in ENGINE_DEPENDENCY_PATHS:
        name = relative.removesuffix("/__init__.py").removesuffix(".py").replace("/", ".")
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == REPO_ROOT / relative, "loaded engine dependency origin differs: " + name)
    import numpy as np
    require(np.__version__ == frozen["activation_representation"]["versions"]["numpy"], "analysis NumPy version differs from frozen runtime")


def _frozen_artifacts(campaign, repo, frozen, inputs):
    """Authenticate approved supporting bytes, without replaying selection/readiness."""
    for ref in [frozen["scientific_selection"], frozen["scientific_selection_independent_review"],
                frozen["technical_preflight"], *frozen["audit_bindings"].values()]:
        path = Path(ref["path"])
        inputs.bytes(path if path.is_absolute() else (repo / path), ref["sha256"])
    for config in frozen["candidate_configurations"].values():
        eid = config["source_experiment_id"]
        require(not (campaign / "reviews" / (eid + "_retired.json")).exists(), "selected discovery source retired")
        inputs.bytes(campaign / "specs" / (eid + ".json"), config["source_spec_sha256"])
        hashes = config["source_result_hashes"]
        require(fingerprint(hashes) == config["source_result_manifest_sha256"], "source result manifest differs")
        for relative, digest in hashes.items():
            path = campaign / relative
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and path.is_relative_to(campaign / "results" / eid), "source result outside its cohort")
            inputs.bytes(path, digest)
        inputs.bytes(repo / config["source_independent_audit_path"], config["source_independent_audit_sha256"])
        for relative, digest in config["source_independent_audit_input_files_sha256"].items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "source audit path must be repository-relative")
            inputs.bytes(repo / relative, digest)
    for manifest in frozen["basis_manifests"].values():
        require(not Path(manifest["source_retirement_guard"]["path"]).exists(), "basis training source retired")
        for path, digest in manifest["artifact_sha256"].items():
            inputs.bytes(Path(path), digest)


def _row_identity(row, job, spec_sha):
    require(row.get("spec_sha256") == spec_sha, "result/spec hash mismatch")
    for key, value in job.items():
        if key not in TEXT_FIELDS:
            require(key in row and same(row[key], value), "result changed frozen field: " + key)
    require(row.get("inference_configuration") == {"head_mode": "full"}, "non-full-head result")
    require(row.get("status") in {"complete", "error"}, "unfinished result object")
    if row["status"] == "error":
        require(row.get("generation_status") == "technical_error" and row.get("native_action") is None, "technical error misclassified as behavior")
        return ["technical_execution_error:" + str(row.get("error_type", "unknown"))]
    from src.activation_causality.analysis import _target_probability
    require(_target_probability(row) is not None and row.get("forced_top_action") == max(ACTIONS, key=lambda a: row["forced_q"][a]), "complete row lacks its frozen forced diagnostic")
    from src.activation_causality.engine import parse_native_action
    require(row.get("input_token_sha256") == fingerprint(job["recipient_token_ids"])
            and row.get("n_input_tokens") == len(job["recipient_token_ids"]), "input token identity differs")
    tokens = row.get("generated_token_ids")
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens)
            and 0 < len(tokens) == row.get("generated_tokens") == row.get("n_new_tokens") <= job["max_new_tokens"], "native token count/IDs invalid")
    parsed = parse_native_action(row["generated_text"])
    require(all(same(row.get(k), v) for k, v in parsed.items()), "native parse differs from recorded continuation")
    reason = row.get("stop_reason")
    require(reason in {"return", "eos", "tool_call", "token_limit"}, "unknown native stop reason")
    complete = reason in {"return", "eos"} and parsed["parse_status"] == "valid"
    status = "complete" if complete else "truncated" if reason == "token_limit" else "malformed"
    require(row.get("generation_status") == status and row.get("native_action") == (parsed["parsed_action"] if complete else None), "native status/action mismatch")
    native_fields = {"action": parsed["parsed_action"] if complete else None, "endpoint": "native_final", "completed": complete,
                     "ended_naturally": reason in {"return", "eos"}, "truncated": reason == "token_limit"}
    require(all(same(row.get(k),v) for k,v in native_fields.items()), "native completion flags differ")
    require(status != "truncated" or len(tokens) == job["max_new_tokens"], "truncation is not at frozen cap")
    return []


def _runtime_evidence(row, job, spec, bundle, inputs):
    frozen, campaign = bundle["freeze"], bundle["campaign"]
    snapshot = frozen["model_identity"]
    require(row.get("model_snapshot_identity_sha256") == snapshot["identity_sha256"], "row model identity differs")
    load, _ = inputs.read(Path(row["model_load_audit_path"]), row["model_load_audit_sha256"])
    require(Path(row["model_load_audit_path"]).is_relative_to(campaign / "reviews"), "model-load audit outside reviews")
    require(load.get("format") == "materialized_model_snapshot_audit_v1" and load.get("passed") is True
            and load.get("experiment_id") == spec["experiment_id"] and load.get("execution_key") == row["execution_key"]
            and same(load.get("before_load"), snapshot) and same(load.get("after_load"), snapshot)
            and same(load.get("activation_representation"), frozen["activation_representation"])
            and load.get("engine_metadata", {}).get("loaded") is True, "model-load audit differs")
    gate, _ = inputs.read(campaign / "reviews" / ("engine_validation_" + row["engine_key"][:16] + ".json"))
    require(gate.get("engine_key") == row["engine_key"] and gate["validation"].get("passed") is True
            and gate["validation"].get("model", {}).get("head_mode") == "full", "engine validation is missing or failed")
    from src.activation_causality.basis_runtime import activation_representation
    require(same(activation_representation(gate["runtime"]), frozen["activation_representation"]), "engine-gate capture representation/runtime differs")
    from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
    sources = frozen["implementation"]
    dependencies = {name: sources[name] for name in ENGINE_DEPENDENCY_PATHS}
    used = {name: value for name, value in sources.items() if name.endswith(("engine.py", "campaign.py"))}
    used.update(dependencies)
    names = {"confirmation_supervisor", "confirmation_contract", "basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit"}
    if spec["protocol_kind"] == "contextual_input":
        names.add("full_write_audit")
    used.update({"src/activation_causality/" + name + ".py": sources["src/activation_causality/" + name + ".py"] for name in names})
    require(same(row.get("inference_source_hashes"), used), "inference source inventory differs from frozen execution")
    for field, names in (("subspace_implementation_sha256", ("basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit")),
                         ("full_write_implementation_sha256", ("full_write_audit", "subspace_audit", "basis_runtime", "model_identity"))):
        if field.startswith("full_write") and spec["protocol_kind"] != "contextual_input":
            continue
        require(same(row.get(field), {"src/activation_causality/"+n+".py":sources["src/activation_causality/"+n+".py"] for n in names}), "cohort helper map differs")
    model_files = {name: snapshot["files"][name] for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json") if name in snapshot["files"]}
    versions = gate["runtime"]["versions"]
    model = bundle["campaign_configuration"]["model_path"]
    execution = {"sources": used, "versions": versions, "model": model, "model_files": model_files, "head_mode": "full",
                 "model_snapshot_identity_sha256": snapshot["identity_sha256"], "confirmation_execution_contract": {
                     "confirmation_mode": frozen["confirmation_mode"], "freeze_id": frozen["freeze_id"],
                     "freeze_sha256": bundle["freeze_sha256"], "inference_cutoff": frozen["inference_cutoff"]}}
    engine = {"engine_sha256": sources["src/activation_causality/engine.py"], "engine_dependency_sources": dependencies,
              "versions": versions, "model": model, "model_files": model_files, "head_mode": "full", "model_snapshot_identity_sha256": snapshot["identity_sha256"]}
    require(row["execution_key"] == fingerprint(execution) and row["engine_key"] == fingerprint(engine), "execution/engine identity does not recompute")
    lease, _ = inputs.read(Path(row["confirmation_lease_path"]), row["confirmation_lease_sha256"])
    require(Path(row["confirmation_lease_path"]) == campaign / "leases" / (row["confirmation_lease_id"] + ".json"), "supervisor lease path differs")
    expected = {"format": "supervised_confirmation_execution_lease_v1", "lease_id": row["confirmation_lease_id"],
                "experiment_id": spec["experiment_id"], "freeze_id": frozen["freeze_id"], "candidate_id": spec["candidate_id"],
                "cohort_role": spec["cohort_role"], "spec_sha256": row["spec_sha256"], "freeze_sha256": bundle["freeze_sha256"],
                "campaign_sha256": bundle["campaign_sha256"], "inference_cutoff": frozen["inference_cutoff"],
                "source_hashes": sources, "worker_entrypoint_sha256": frozen["worker_entrypoint_sha256"]}
    require(all(same(lease.get(k), v) for k, v in expected.items()), "supervisor lease binding differs")
    require(row.get("confirmation_inference_cutoff") == frozen["inference_cutoff"]
            and datetime.fromisoformat(row["started_at"]) < datetime.fromisoformat(frozen["inference_cutoff"]), "job started outside frozen cutoff")
    errors = _lease_evidence(row, lease, bundle, inputs) + _norm_evidence(row, job)
    if job["intervention"].get("audit_projected_norm"):
        config = frozen["candidate_configurations"][spec["candidate_id"]]
        for prepared in row.get("compiled_patch_audits", []):
            require(prepared["projected_norm_audit"]["original_task_basis_float32_sha256"] == _basis_norm_hash(config, inputs), "norm reference uses a different task basis")
    return errors


def _lease_evidence(row, lease, bundle, inputs):
    """Authenticate launch/terminal association without readiness or inference."""
    from src.activation_causality.confirmation_supervisor import _interpreter_binding
    campaign, frozen = bundle["campaign"], bundle["freeze"]
    if inputs.interpreter_binding is None:
        inputs.interpreter_binding = _interpreter_binding()
    expected = {"repo_path": str(REPO_ROOT), "campaign_path": str(campaign),
        "spec_path": str(campaign / "specs" / (lease["experiment_id"]+".json")),
        "freeze_path": str(campaign / "freezes" / (frozen["freeze_id"]+".json")),
        "worker_entrypoint_path": str(REPO_ROOT / "scripts/30_activation_causality_campaign.py"), **inputs.interpreter_binding}
    require(all(same(lease.get(k),v) for k,v in expected.items()), "lease process/path/interpreter binding differs")
    lease_id = row["confirmation_lease_id"]
    root = campaign / "reviews" / "confirmation_supervision"
    launch_path, terminal_path = (root / (lease_id+suffix) for suffix in (".launch.json", ".terminal.json"))
    if not launch_path.exists() or not terminal_path.exists():
        return ["operational_pending:missing_launch_or_terminal:"+lease_id]
    launch, launch_sha = inputs.read(launch_path)
    terminal, terminal_sha = inputs.read(terminal_path)
    command = [lease["interpreter_path"], "-I", "-B", lease["worker_entrypoint_path"], "--campaign", str(campaign), "run", "--experiment-id", lease["experiment_id"]]
    require(launch.get("format") == "supervised_confirmation_launch_v1" and launch.get("lease_id") == lease_id
            and launch.get("lease_sha256") == row["confirmation_lease_sha256"] and same(launch.get("command"), command)
            and type(launch.get("worker_pid")) is int and launch["worker_pid"] > 0
            and launch["worker_pid"] == launch.get("worker_process_group")
            and type(lease.get("supervisor_pid")) is int and lease["supervisor_pid"] > 0
            and launch.get("supervisor_pid") == lease.get("supervisor_pid"), "supervisor launch association differs")
    expected_terminal = {"format": "supervised_confirmation_terminal_v1", "lease_id": lease_id,
        "lease_path": row["confirmation_lease_path"], "lease_sha256": row["confirmation_lease_sha256"],
        "launch_record_path": str(launch_path), "launch_record_sha256": launch_sha, "terminal_record_path": str(terminal_path),
        "experiment_id": lease["experiment_id"], "inference_cutoff": frozen["inference_cutoff"],
        "log_path": str(campaign / "logs" / ("confirmation_supervision_"+lease_id+".log"))}
    require(all(same(terminal.get(k),v) for k,v in expected_terminal.items()), "supervisor terminal association differs")
    inputs.bytes(Path(terminal["log_path"]), terminal["log_sha256"])
    created, launched, started, finished, terminal_end, cutoff = (datetime.fromisoformat(v) for v in
        (lease["created_at"], launch["launched_at"], row["started_at"], row["finished_at"], terminal["finished_at"], frozen["inference_cutoff"]))
    require(all(t.tzinfo is not None for t in (created,launched,started,finished,terminal_end,cutoff))
            and created <= launched <= started <= finished <= terminal_end and finished < cutoff, "row outside authenticated launch/terminal/cutoff window")
    require(terminal.get("status") in {"worker_exited", "worker_failed", "cutoff_terminated", "supervisor_error"}, "unknown supervisor terminal status")
    inputs.lease_records[lease_id] = {"path": str(terminal_path), "sha256": terminal_sha,
        "experiment_id": lease["experiment_id"], "status": terminal["status"], "worker_exit_code": terminal.get("worker_exit_code"),
        "incomplete": terminal.get("incomplete"), "finished_at": terminal["finished_at"],
        "inventory": terminal.get("current_result_inventory")}
    return []


def _supervision_history(campaign, bundle, inputs):
    """Retain earlier leases even when they produced no surviving current row."""
    history = defaultdict(list)
    paths = sorted((campaign / "leases").glob("*.json"))
    frozen = bundle["freeze"]
    for path in paths:
        lease, digest = inputs.read(path)
        eid = lease.get("experiment_id")
        if eid not in bundle["specs"]:
            continue
        require(path.stem == lease.get("lease_id") and lease.get("freeze_sha256") == bundle["freeze_sha256"]
                and lease.get("spec_sha256") == frozen["planned_experiments"][eid]["spec_sha256"]
                and same(lease.get("source_hashes"), frozen["implementation"]), "earlier lease frozen identity differs")
        record = {"lease_path": str(path), "lease_sha256": digest, "created_at": lease["created_at"], "files": {},
                  "status": "completion_evidence_absent", "terminal": None, "launch_evidence_present": False,
                  "scope": "retained operational history; row-bearing leases separately validated in full"}
        root = campaign / "reviews" / "confirmation_supervision"
        launch_path, terminal_path = (root / (path.stem+suffix) for suffix in (".launch.json", ".terminal.json"))
        if launch_path.exists():
            launch, sha = inputs.read(launch_path)
            require(launch.get("lease_id") == path.stem and launch.get("lease_sha256") == digest, "earlier launch association differs")
            record["files"][str(launch_path)] = sha
            record["launch_evidence_present"] = True
        if terminal_path.exists():
            terminal, sha = inputs.read(terminal_path)
            require(terminal.get("lease_id") == path.stem and terminal.get("lease_sha256") == digest
                    and terminal.get("experiment_id") == eid, "earlier terminal association differs")
            if "launch_record_sha256" in terminal and record["launch_evidence_present"]:
                require(terminal["launch_record_sha256"] == record["files"].get(str(launch_path)), "earlier terminal/launch hash differs")
            if "log_sha256" in terminal:
                log = campaign / "logs" / ("confirmation_supervision_"+path.stem+".log")
                require(terminal.get("log_path") == str(log), "earlier supervisor log path differs")
                inputs.bytes(log, terminal["log_sha256"])
                record["files"][str(log)] = terminal["log_sha256"]
            record.update(status=terminal["status"], terminal={k:terminal.get(k) for k in
                ("status", "incomplete", "worker_exit_code", "finished_at", "current_result_inventory", "error_type",
                 "launch_record_sha256", "cleanup_action")})
            record["files"][str(terminal_path)] = sha
        history[eid].append(record)
    require(paths == sorted((campaign / "leases").glob("*.json")), "lease history inventory changed during read")
    return dict(history)


def _historical_pending(history):
    """A later success cannot supply missing evidence for an earlier lease."""
    pending = []
    for record in history:
        terminal = record.get("terminal")
        if terminal is None:
            pending.append({"reason": "historical_lease_missing_terminal", "lease": record["lease_path"]})
            continue
        launched = (terminal.get("status") in {"worker_exited", "worker_failed", "cutoff_terminated"}
                    or terminal.get("launch_record_sha256") is not None
                    or terminal.get("worker_exit_code") is not None or terminal.get("cleanup_action") is not None)
        if record.get("launch_evidence_present") is False and launched:
            pending.append({"reason": "historical_launched_worker_missing_launch", "lease": record["lease_path"]})
    return pending


def _historical_failures(history, attempts, events=()):
    """A later success cannot clear known technical failures; no override API."""
    failures = []
    for record in history:
        terminal = record.get("terminal") or {}
        status = terminal.get("status")
        interrupted = terminal.get("error_type") in {"SupervisorInterrupted", "KeyboardInterrupt"}
        if status == "worker_failed" or status == "supervisor_error" and not interrupted:
            failures.append({"reason": "unresolved_historical_supervised_technical_failure", "lease": record["lease_path"],
                             "status": status, "error_type": terminal.get("error_type")})
    for attempt in attempts:
        require(attempt.get("status") == "error" and attempt.get("generation_status") == "technical_error", "unexpected archived attempt is not an authenticated technical error")
        failures.append({"reason": "unresolved_historical_technical_attempt", "job_id": attempt["job_id"],
                         "path": attempt["path"], "sha256": attempt["sha256"]})
    for record in events:
        event = record["event"]
        if event.get("kind") in TECHNICAL_EVENTS or event.get("kind") == "worker_finished" and event.get("errors", 0) > 0:
            failures.append({"reason": "unresolved_historical_technical_event", "line": record["line"],
                             "event_sha256": record["event_sha256"], "kind": event["kind"]})
    return failures


def _operational_events(campaign, bundle, inputs):
    path = campaign / "events.jsonl"
    if not path.exists():
        return {}
    raw, _ = inputs.bytes(path)
    grouped = defaultdict(list)
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        event = strict_json(line)
        if event.get("experiment_id") in bundle["specs"]:
            grouped[event["experiment_id"]].append({"line": number, "event_sha256": fingerprint(event), "event": event})
    return dict(grouped)


def _basis_norm_hash(config, inputs):
    """The norm-audit and localization array digest conventions are distinct."""
    raw, digest = inputs.bytes(Path(config["basis_path"]), config["basis_sha256"])
    key = fingerprint({"file_sha256": digest, "rank": config["rank"], "localization_array_sha256": config["basis_metadata"]["basis_array_sha256"]})
    if key not in inputs.basis_norm_hashes:
        import numpy as np
        from src.activation_causality.subspace_audit import array_fingerprint
        from src.activation_causality.localization import _array_fingerprint
        basis = np.load(io.BytesIO(raw), allow_pickle=False)
        require(basis.ndim == 2 and basis.shape[1] == config["rank"]
                and _array_fingerprint(basis) == config["basis_metadata"]["basis_array_sha256"], "saved task-basis array identity differs")
        inputs.basis_norm_hashes[key] = array_fingerprint(basis)
    return inputs.basis_norm_hashes[key]


def _norm_evidence(row, job):
    if row["status"] == "error":
        return []  # The genuine technical error already invalidates its whole cohort.
    from src.activation_causality.full_write_audit import validate_full_write_norm_audit
    from src.activation_causality.subspace_audit import validate_applied_norm_audit
    errors, intervention = [], job["intervention"]
    logs = row.get("patch_log", [])
    require(isinstance(logs, list) and len(logs) == len(intervention.get("layers", [])), "actual patch count differs from frozen intervention")
    for item, layer in zip(logs, intervention.get("layers", []), strict=True):
        expected = {"layer": layer, "site": intervention["site"], "positions": job["selection"]["recipient_positions"],
                    "operation": intervention["operation"], "dose": float(intervention.get("dose", 1.0)),
                    "rank": job["metadata"]["rank"] if intervention["operation"] == "subspace" else None}
        require(all(same(item.get(k), v) for k, v in expected.items()), "actual patch geometry differs from frozen intervention")
        # Match the frozen engine's emitted schema; no after-state norm exists.
        for field in ("before_l2", "delta_l2", "relative_delta_l2"):
            values = item.get(field)
            require(isinstance(values, list) and len(values) == len(expected["positions"])
                    and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), "invalid actual patch norm array")
    for flag, prepared_key, applied_key, status_key, validator in (
        ("audit_full_write_norm", "compiled_full_write_audits", "applied_full_write_audits", "full_write_norm_audit_status", validate_full_write_norm_audit),
        ("audit_projected_norm", "compiled_patch_audits", "applied_projected_norm_audits", "projected_norm_audit_status", validate_applied_norm_audit),
    ):
        if not intervention.get(flag):
            continue
        prepared = row.get(prepared_key, [])
        if len(prepared) != 1:
            errors.append("missing_or_multiple_" + prepared_key)
            continue
        audit = prepared[0] if flag == "audit_full_write_norm" else prepared[0]["projected_norm_audit"]
        expected = {"layer": intervention["layers"][0], "site": intervention["site"], "positions": job["selection"]["recipient_positions"], "dose": intervention.get("dose", 1.0)}
        require(all(same(audit.get(k), v) for k, v in expected.items()), "norm audit changes frozen write geometry")
        if flag == "audit_full_write_norm":
            require(same(audit.get("source_identity"), intervention["full_write_source_identity"]), "full-write source identity differs")
            expected_donor = (row["random_orientation_audits"][0]["compiled_donor_float32_sha256"]
                              if intervention.get("random_control") else audit["original_reference_donor_float32_sha256"])
            require(audit.get("compiled_donor_float32_sha256") == expected_donor
                    and isinstance(expected_donor, str) and re.fullmatch(r"[0-9a-f]{64}", expected_donor)
                    and audit.get("inline_definition_sha256") == fingerprint(audit["source_identity"]),
                    "full-write compiled donor/source binding differs")
        else:
            require(audit.get("rank") == job["metadata"]["rank"], "task subspace rank differs")
        actual = validator(audit, logs)
        require(same(row.get(applied_key), [actual]), "stored/recomputed applied norm evidence differs")
        if actual.get("passed") is not True or row.get(status_key) != "passed":
            errors.append("failed_" + status_key)
        if intervention.get("projected_norm_matched_random_basis"):
            helper = prepared[0].get("random_control_audit", {})
            if helper.get("float32", {}).get("all_norms_match_within_tolerance") is not True:
                errors.append("projected_random_float32_sanity_failed")
    if intervention.get("record_random_orientation"):
        records = row.get("random_orientation_audits", [])
        require(len(records) == 1 and records[0].get("seed") == intervention["random_seed"]
                and same(records[0].get("source_identity"), intervention["full_write_source_identity"]), "random orientation source/seed differs")
        import numpy as np
        from src.activation_causality.subspace_audit import array_fingerprint
        record = records[0]
        shape = record.get("shape")
        require(isinstance(shape, list) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
                and shape[0] == len(job["selection"]["recipient_positions"]), "random orientation support differs")
        raw = np.random.default_rng(intervention["random_seed"]).standard_normal(shape).astype(np.float32)
        require(record.get("numpy_version") == np.__version__ and record.get("raw_gaussian_float32_sha256") == array_fingerprint(raw), "fixed Gaussian draw does not reproduce")
        if intervention.get("audit_full_write_norm") and record.get("passed") is not True:
            errors.append("random_orientation_sanity_failed")
    return errors


def _cohort_controls(spec, rows):
    """Engineering identities and shared norm references; no behavioral filter."""
    by_pair = defaultdict(dict)
    jobs = {(j["pair_id"], j["seed"], j["condition"]): j for j in spec["jobs"]}
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = row
    failures = []
    contextual = spec["protocol_kind"] == "contextual_input"
    for pid, group in by_pair.items():
        if contextual:
            layer = spec["frozen_configuration"]["layer"] if "frozen_configuration" in spec else spec["supplement_configuration"]["layer"]
            prefix = f"layer{layer}_"
            identities, baseline = (prefix + "identity", prefix + "restoration_equivalent"), "baseline"
            natural, random, old = (group.get(prefix + name) for name in ("natural", "natural_norm_random", "random"))
            if all(r and r["status"] == "complete" for r in (natural, random, old)):
                # Compare the original reference, retaining every key except
                # the separately bound natural versus synthetic donor hash.
                references = [{k: v for k, v in row["compiled_full_write_audits"][0].items()
                               if k != "compiled_donor_float32_sha256"} for row in (natural, random)]
                if not same(*references):
                    failures.append({"pair_id": pid, "reason": "natural_and_random_norm_references_differ"})
                if old["random_orientation_audits"][0]["raw_gaussian_float32_sha256"] != random["random_orientation_audits"][0]["raw_gaussian_float32_sha256"]:
                    failures.append({"pair_id": pid, "reason": "matched_and_natural_random_orientations_differ"})
        else:
            identities, baseline = ("subspace_identity", "restoration_equivalent"), "baseline"
            for dose in ("0p5", "1"):
                forward = group.get("subspace_forward_" + dose)
                random = group.get("projected_norm_matched_random_subspace_" + dose)
                if all(r and r["status"] == "complete" for r in (forward, random)):
                    a, b = (r["compiled_patch_audits"][0]["projected_norm_audit"] for r in (forward, random))
                    if not same(a, b):
                        failures.append({"pair_id": pid, "reason": "task_and_random_projected_norm_references_differ:" + dose})
        for identity in identities:
            baseline_name = "counterfactual_text" if contextual and identity.endswith("restoration_equivalent") else baseline
            ident, base = group.get(identity), group.get(baseline_name)
            if all(r and r["status"] == "complete" for r in (ident, base)):
                frozen_identity = jobs[pid, ident["seed"], identity]
                frozen_baseline = jobs[pid, ident["seed"], baseline_name]
                require(frozen_identity["metadata"]["paired_baseline_condition"] == baseline_name
                        and same(frozen_identity["recipient_token_ids"], frozen_baseline["recipient_token_ids"]),
                        "identity frozen baseline/input binding differs")
                zero = ident.get("patch_log") and all(v == 0 for item in ident["patch_log"] for v in item["delta_l2"])
                if not zero or ident["generated_token_ids"] != base["generated_token_ids"] or not same(ident.get("forced_q"), base.get("forced_q")):
                    failures.append({"pair_id": pid, "reason": "identity_is_not_exact_replay:" + identity})
    return failures


def load_bundle(campaign, freeze_id, *, at_cutoff=False, repo_root=REPO_ROOT):
    campaign, repo_root = Path(campaign).resolve(), Path(repo_root).resolve()
    inputs = Inputs()
    bundle = _contract(campaign, freeze_id, inputs)
    bundle["campaign"] = campaign
    cutoff = datetime.fromisoformat(bundle["freeze"]["inference_cutoff"])
    require(not at_cutoff or datetime.now(timezone.utc) >= cutoff, "at-cutoff reporting is unavailable before the frozen cutoff")
    inventories = {}
    for eid, spec in bundle["specs"].items():
        expected = {campaign / "results" / eid / (j["job_id"] + ".json") for j in spec["jobs"]}
        actual = set((campaign / "results" / eid).glob("*.json"))
        require(actual <= expected, "unexpected result files: " + eid)
        inventories[eid] = (expected, actual)
    require(at_cutoff or all(expected == actual for expected, actual in inventories.values()),
            "complete-only report requires every planned primary/supplement file; no native outcomes read")
    _module_bindings(bundle["freeze"], inputs)
    _frozen_artifacts(campaign, repo_root, bundle["freeze"], inputs)
    history = _supervision_history(campaign, bundle, inputs)
    events = _operational_events(campaign, bundle, inputs)
    cohorts = {}
    for eid, spec in bundle["specs"].items():
        rows, failures = [], []
        pending = []
        expected, actual = inventories[eid]
        jobs = {j["job_id"]: j for j in spec["jobs"]}
        for path in sorted(actual):
            row, _ = inputs.read(path)
            reasons = _row_identity(row, jobs[path.stem], bundle["freeze"]["planned_experiments"][eid]["spec_sha256"])
            reasons += _runtime_evidence(row, jobs[path.stem], spec, bundle, inputs)
            for reason in reasons:
                (pending if reason.startswith("operational_pending:") else failures).append({"job_id": path.stem, "reason": reason})
            rows.append(row)
        failures.extend(_cohort_controls(spec, rows))
        keys = {row["execution_key"] for row in rows}
        require(len(keys) <= 1, "mixed execution identities within cohort")
        leases = [v for v in inputs.lease_records.values() if v["experiment_id"] == eid]
        historical = history.get(eid, [])
        pending.extend(_historical_pending(historical))
        if historical:
            latest_history = max(historical, key=lambda v: datetime.fromisoformat(v["created_at"]))
            terminal = latest_history["terminal"]
            if terminal is None or terminal["status"] in {"cutoff_before_launch", "cutoff_terminated"}:
                pending.append({"reason": "latest_lease_has_no_complete_execution", "lease": latest_history["lease_path"]})
            elif terminal["status"] in {"worker_failed", "supervisor_error"}:
                if terminal.get("error_type") in {"SupervisorInterrupted", "KeyboardInterrupt"}:
                    pending.append({"reason": "latest_lease_interrupted", "lease": latest_history["lease_path"]})
            elif terminal["status"] != "worker_exited" or terminal["worker_exit_code"] != 0 or terminal["incomplete"] is not False:
                pending.append({"reason": "latest_supervised_execution_incomplete", "lease": latest_history["lease_path"]})
            else:
                inventory = terminal["current_result_inventory"]
                require(inventory.get("planned_jobs") == len(expected) and inventory.get("saved_result_files") == len(actual)
                        and inventory.get("missing_result_files") == sorted(p.name for p in expected - actual)
                        and inventory.get("unexpected_result_files") == [], "completed supervisor inventory differs from current frozen rows")
        attempts, attempt_records = {}, []
        for path in sorted((campaign / "results" / eid / "attempts").glob("*.json")):
            attempt, digest = inputs.read(path)
            require(attempt.get("job_id") in jobs and path.name == attempt["job_id"]+"_"+fingerprint(attempt)[:12]+".json", "archived attempt filename/identity differs")
            _row_identity(attempt, jobs[attempt["job_id"]], bundle["freeze"]["planned_experiments"][eid]["spec_sha256"])
            require(attempt.get("status") == "error", "archived completed behavior cannot be silently replaced")
            _runtime_evidence(attempt, jobs[attempt["job_id"]], spec, bundle, inputs)
            attempts[str(path)] = digest
            attempt_records.append({"path": str(path), "sha256": digest, "job_id": attempt["job_id"],
                                    "status": attempt["status"], "generation_status": attempt["generation_status"]})
        failures.extend(_historical_failures(historical, attempt_records, events.get(eid, [])))
        cohorts[eid] = {"rows": rows, "missing_job_ids": sorted(p.stem for p in expected - actual), "technical_failures": failures,
                        "operational_pending_reasons": pending, "supervision_records": leases,
                        "supervision_history": historical, "prior_attempt_files_sha256": attempts, "prior_attempt_records": attempt_records,
                        "operational_events": events.get(eid, []),
                        "expected_jobs": len(expected), "saved_jobs": len(actual),
                        "technical_status_counts": dict(Counter(row["status"] for row in rows))}
    require(at_cutoff or all(not value["technical_failures"] and not value["operational_pending_reasons"] for value in cohorts.values()), "complete-only report found incomplete/technical-invalid execution evidence")
    for eid, (_, actual) in inventories.items():
        require(set((campaign / "results" / eid).glob("*.json")) == actual, "result inventory changed during read")
    inputs.recheck()
    bundle.update(cohorts=cohorts, mode="at_cutoff" if at_cutoff else "complete_only", authenticated=True,
                  observed_at=datetime.now(timezone.utc).isoformat(), provenance={"input_file_sha256": inputs.hashes,
                    "script_path": str(Path(__file__).resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "independent_final_raw_and_report_audit_required_separately": True})
    bundle.pop("campaign", None)
    return bundle


def _aggregate(values, family_by_layout, weights):
    per_family = {}
    for family, weight in weights.items():
        rows = [value for layout, value in values.items() if family_by_layout[layout] == family]
        require(rows, "missing frozen descriptive family")
        per_family[family] = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
    average = [sum(weights[f] * per_family[f][i] for f in weights) for i in range(len(next(iter(per_family.values()))))]
    return average, per_family


def descriptive_contrast(standard, alternate, treatment, control, pair_ids, weights):
    """Fixed same-layout effects and sharp per-row observed-cohort censor bounds."""
    def keyed(rows):
        selected = [r for r in rows if r["pair_id"] in pair_ids]
        mapping = {(r["pair_id"], r["seed"], r["condition"]): r for r in selected}
        require(len(mapping) == len(selected), "duplicate descriptive observation")
        return mapping
    maps = [keyed(rows) for rows in (standard, alternate)]
    grouped, families = defaultdict(list), {}
    counts = {encoding: {arm: Counter() for arm in ("treatment", "control")} for encoding in ("standard", "alternate")}
    for pid in sorted(pair_ids):
        seeds = sorted({seed for pair, seed, condition in maps[0] if pair == pid and condition == treatment})
        require(seeds == [42], "frozen seed missing in descriptive contrast")
        for seed in seeds:
            records = [mapping[pid, seed, condition] for mapping in maps for condition in (treatment, control)]
            require(all(r["generation_status"] in NATIVE and r["status"] == "complete" for r in records), "partial/technical lexical contrast is not synthesized")
            require(len({(r["layout_id"], r["family"], r["target_action"]) for r in records}) == 1, "paired lexical identities differ")
            layout, family = records[0]["layout_id"], records[0]["family"]
            families[layout] = family
            y = [float(r["generation_status"] == "complete" and r["native_action"] == r["target_action"]) for r in records]
            u = [float(r["generation_status"] != "complete") for r in records]
            ds, da = y[0] - y[1], y[2] - y[3]
            delta = da - ds
            grouped[layout].append([ds, ds-u[1], ds+u[0], da, da-u[3], da+u[2], delta, delta-u[3]-u[0], delta+u[2]+u[1]])
            for index, row in enumerate(records):
                encoding, arm = ("standard" if index < 2 else "alternate"), ("treatment" if index % 2 == 0 else "control")
                category = "unresolved" if row["generation_status"] != "complete" else "target" if y[index] else "other"
                counts[encoding][arm][category] += 1
    require(len(grouped) == 32, "lexical contrast must retain exactly32 layouts")
    quotas = dict(Counter(families.values()))
    require(same(weights, {family: count/32 for family, count in quotas.items()}), "descriptive weights differ from the fixed32 mixture")
    averaged = {layout: [sum(row[i] for row in rows)/len(rows) for i in range(9)] for layout, rows in grouped.items()}
    mean, by_family = _aggregate(averaged, families, weights)
    return {"treatment": treatment, "control": control, "n_layouts": 32, "n_pairs": len(pair_ids), "family_weights": weights,
            "standard": {"itt_effect": mean[0], "completion_bounds": mean[1:3]},
            "alternate": {"itt_effect": mean[3], "completion_bounds": mean[4:6]},
            "encoding_interaction": {"itt_effect": mean[6], "completion_bounds": mean[7:9]},
            "per_family_values": by_family, "counts": counts,
            "inferential_role": "descriptive", "confidence_intervals": None, "p_values": None,
            "bound_scope": "observed-cohort completion sensitivity; not confidence intervals"}


def build_report(authenticated_bundle, *, at_cutoff=False):
    from src.activation_causality.analysis import analyze_results
    from src.activation_causality.confirmation_contract import validate_registry_configurations, validate_primary_spec
    bundle = deepcopy(authenticated_bundle)
    require(bundle.get("authenticated") is True and bundle.get("mode") == ("at_cutoff" if at_cutoff else "complete_only"), "authenticated reporting mode required")
    frozen, specs, cohorts = bundle["freeze"], bundle["specs"], bundle["cohorts"]
    configs = frozen["candidate_configurations"]
    _validate_supplements(frozen, specs)
    registry = validate_registry_configurations(frozen["confirmation_primary_family"], configs)
    require(set(specs) == set(cohorts), "cohort/spec inventory differs")
    for eid, spec in specs.items():
        cohort = cohorts[eid]
        attempts = cohort.get("prior_attempt_records", [])
        require(same(cohort.get("prior_attempt_files_sha256", {}), {a["path"]:a["sha256"] for a in attempts}), "archived attempt provenance inventory differs")
        historical = _historical_failures(cohort.get("supervision_history", []), attempts, cohort.get("operational_events", []))
        existing = {fingerprint(value) for value in cohort["technical_failures"]}
        cohort["technical_failures"].extend(value for value in historical if fingerprint(value) not in existing)
        pending = cohort.setdefault("operational_pending_reasons", [])
        known_pending = {fingerprint(value) for value in pending}
        pending.extend(value for value in _historical_pending(cohort.get("supervision_history", []))
                       if fingerprint(value) not in known_pending)
        jobs = {j["job_id"]: j for j in spec["jobs"]}
        actual = {r["job_id"] for r in cohort["rows"]}
        require(len(actual) == len(cohort["rows"]) and actual <= jobs.keys()
                and cohort["expected_jobs"] == len(jobs) and cohort["saved_jobs"] == len(actual)
                and cohort["missing_job_ids"] == sorted(jobs.keys() - actual), "saved/missing cohort denominator differs")
        require(cohort["technical_failures"] or all(r["status"] != "error" for r in cohort["rows"]), "unreported technical error")
        for row in cohort["rows"]:
            require(all(same(row.get(k), jobs[row["job_id"]].get(k)) for k in
                        ("experiment_id", "pair_id", "layout_id", "family", "condition", "seed", "target_action", "original_action")), "paired observation differs from its frozen job")
    primary = {s["candidate_id"]: s for s in specs.values() if s["cohort_role"] == "primary"}
    require(set(primary) == set(configs), "missing frozen primary specification")
    rows, comparisons, inventories = [], [], {}
    for cid, spec in primary.items():
        validate_primary_spec(spec, configs[cid], registry)
        cohort = cohorts[spec["experiment_id"]]
        missing, errors, pending = cohort["missing_job_ids"], cohort["technical_failures"], cohort.get("operational_pending_reasons", [])
        require(at_cutoff or not missing and not errors and not pending, "complete-only synthesis refuses incomplete/invalid whole candidate")
        state = "invalid" if errors else "pending" if missing or pending else "complete"
        if state == "complete":
            rows.extend(cohort["rows"])
        chosen = [deepcopy(c) for c in spec["comparisons"] if c.get("primary_id") is not None]
        if state == "invalid":
            for comparison in chosen:
                comparison["evidence"] = {**comparison.get("evidence", {}), "technical_valid": False}
        comparisons.extend(chosen)
        inventories[cid] = {"experiment_id": spec["experiment_id"], "whole_cohort_status": state,
            "expected_n_layouts": configs[cid]["n_layouts"], "expected_n_pairs": configs[cid]["n_layouts"],
            **{key: deepcopy(cohort[key]) for key in ("expected_jobs", "saved_jobs", "missing_job_ids", "technical_failures", "technical_status_counts")},
            "operational_pending_reasons": pending, "supervision_records": cohort.get("supervision_records", []),
            "supervision_history": cohort.get("supervision_history", []), "prior_attempt_files_sha256": cohort.get("prior_attempt_files_sha256", {}),
            "prior_attempt_records": cohort.get("prior_attempt_records", []),
            "operational_events": cohort.get("operational_events", []),
            "inferential_input_withheld": state != "complete", "behavioral_failures_imputed_for_missing_or_technical_rows": False}
    combined = analyze_results(rows, comparisons=comparisons, confirmation_primary_family=registry, **STATS)
    for component in combined["primary_components"]:
        cid = component["candidate_id"]
        component["whole_cohort_gate"] = deepcopy(inventories[cid])
    lexical = []
    for alt in specs.values():
        if alt["cohort_role"] != "lexical_supplement":
            continue
        cid, parent_id = alt["candidate_id"], alt["parent_experiment_id"]
        cohort, standard = cohorts[alt["experiment_id"]], cohorts[parent_id]
        available = not cohort["missing_job_ids"] and not cohort["technical_failures"] and not cohort.get("operational_pending_reasons") and inventories[cid]["whole_cohort_status"] == "complete"
        require(at_cutoff or available, "complete-only lexical cohort is incomplete/invalid")
        entry = {"candidate_id": cid, "experiment_id": alt["experiment_id"], "parent_experiment_id": parent_id,
                 "n_layouts": 32, "family_weights": alt["family_weights"], "primary_family_weights": configs[cid].get("family_weights", {configs[cid].get("family"): 1}),
                 "status": "complete" if available else "unmeasured_or_invalid_at_cutoff", "descriptive_transfer_gate_met": False,
                 "missing_job_ids": cohort["missing_job_ids"], "technical_failures": cohort["technical_failures"],
                 "operational_pending_reasons": cohort.get("operational_pending_reasons", []), "contrasts": []}
        entry.update(supervision_history=cohort.get("supervision_history", []), prior_attempt_files_sha256=cohort.get("prior_attempt_files_sha256", {}),
                     prior_attempt_records=cohort.get("prior_attempt_records", []), operational_events=cohort.get("operational_events", []))
        if available:
            ids = set(alt["lexical_subset"]["pair_ids"])
            if configs[cid]["protocol_kind"] == "contextual_input":
                standard_random = {(r["pair_id"], r["condition"]): r for r in standard["rows"] if r["pair_id"] in ids}
                for row in cohort["rows"]:
                    if row["intervention"].get("record_random_orientation"):
                        original = standard_random[row["pair_id"], row["condition"]]
                        require(row["random_orientation_audits"][0]["raw_gaussian_float32_sha256"] == original["random_orientation_audits"][0]["raw_gaussian_float32_sha256"], "lexical random orientation differs across encodings")
            binding_items = list(configs[cid]["primary_comparisons"].items())
            treatment = binding_items[0][1]["treatment"]
            physical = "counterfactual_text" if configs[cid]["protocol_kind"] == "contextual_input" else "donor_text"
            definitions = [(name, binding["treatment"], binding["control"]) for name, binding in binding_items]
            definitions += [("efficacy", treatment, "baseline"), ("physical_counterfactual", physical, "baseline")]
            for name, t, c in definitions:
                value = descriptive_contrast(standard["rows"], cohort["rows"], t, c, ids, alt["family_weights"])
                value["role"] = name
                entry["contrasts"].append(value)
            entry["descriptive_transfer_gate_met"] = all(value["alternate"]["completion_bounds"][0] > 0 and
                (value["role"] == "physical_counterfactual" or value["alternate"]["itt_effect"] >= .10) for value in entry["contrasts"])
        lexical.append(entry)
    condition_counts, display_metadata = {}, {}
    for eid, cohort in cohorts.items():
        spec = specs[eid]
        config = configs[spec["candidate_id"]]
        is_repeat = spec["cohort_role"] == "lexical_supplement"
        planned_pairs = {j["pair_id"]:j for j in spec["jobs"]}
        planned_layouts = {j["layout_id"]:j["family"] for j in spec["jobs"]}
        quotas = dict(Counter(planned_layouts.values()))
        display_metadata[eid] = {"candidate_id":spec["candidate_id"], "cohort_role":spec["cohort_role"],
            "parent_experiment_id":spec.get("parent_experiment_id"), "condition_order":deepcopy(config["condition_inventory"]),
            "expected_n_layouts":len(planned_layouts), "expected_n_pairs":len(planned_pairs), "expected_jobs":len(spec["jobs"]),
            "expected_condition_trials":dict(Counter(j["condition"] for j in spec["jobs"])),
            "renderer_id":spec.get("renderer_id", config.get("renderer_id")), "native_seeds":sorted({j["seed"] for j in spec["jobs"]}),
            "max_new_tokens":4096, "temperatures":sorted({j["temperature"] for j in spec["jobs"]}),
            "family_layout_counts":quotas, "family_weights":{family:n/len(planned_layouts) for family,n in quotas.items()},
            "additional_layouts_beyond_parent":0 if is_repeat else len(planned_layouts),
            "claim_scope":spec.get("claim_scope", config["claim_scope"]), "semantic_headline_eligible":False,
            "layer":config.get("layer"), "rank":config.get("rank"), "selected_dose":config.get("selected_dose", config.get("dose")),
            "basis_id":config.get("basis_id")}
        if cohort["missing_job_ids"] or cohort["technical_failures"] or cohort.get("operational_pending_reasons"):
            condition_counts[eid] = None
            continue
        grouped = defaultdict(list)
        for row in cohort["rows"]:
            grouped[row["condition"]].append(row)
        def counts(values):
            return {"n":len(values), "n_layouts":len({r["layout_id"] for r in values}), "n_pairs":len({r["pair_id"] for r in values}),
                "target":sum(r["native_action"] == r["target_action"] for r in values),
                "original":sum(r["native_action"] == r["original_action"] for r in values),
                "third_action":sum(r["native_action"] in ACTIONS and r["native_action"] not in {r["target_action"],r["original_action"]} for r in values),
                "unresolved":sum(r["generation_status"] != "complete" for r in values),
                "generation_status_counts":dict(Counter(r["generation_status"] for r in values))}
        condition_counts[eid] = {condition:{**counts(values), "by_family":{
            family:counts([r for r in values if r["family"] == family]) for family in quotas}}
            for condition,values in grouped.items()}
    return {"format": "combined_frozen_confirmation_report_v1", "freeze_id": frozen["freeze_id"], "mode": bundle["mode"],
            "observed_at": bundle.get("observed_at"), "provenance": bundle["provenance"], "primary": combined,
            "whole_candidate_inventories": inventories, "lexical_supplements": lexical,
            "complete_cohort_condition_counts": condition_counts,
            "cohort_display_metadata": display_metadata,
            "omitted_lexical_supplements": [cid for cid, enabled in frozen["supplements"].items() if not enabled],
            "independent_final_raw_and_report_audit_required_separately": True,
            "interpretation": ["Native cap/malformed continuations count as zero target success; technical errors and absent files do not become behavioral failures.",
                "All frozen primary slots retain joint multiplicity; candidate support requires both components. An observed effect>=0.10 does not establish a population effect>=0.10.",
                "Lexical results repeat the same32 layouts using the same subset weights in both encodings. Completion bounds are sensitivity bounds, not confidence intervals.",
                "Descriptive lexical transfer does not establish abstract belief, necessity, mediation, semantic invariance or equivalence."]}


def markdown(report):
    def number(value):
        return "—" if value is None else f"{value:.5g}"
    def interval(value):
        return "—" if value is None else "[" + ", ".join(number(v) for v in value) + "]"
    summary = report["primary"]
    lines = ["# Combined frozen confirmation report", "", f"Freeze: `{report['freeze_id']}`. Mode: `{report['mode']}`.", "",
             f"One joint analysis retains all **{summary['confirmation_primary_family']['family_size']}** frozen primary slots. Missing and invalid slots use p=1; candidate support requires both components.", "",
             "| Candidate / primary | State | ITT effect | Simultaneous CI | Completion bounds | Joint Holm p | Verdict |",
             "|---|---|---:|---|---|---:|---|"]
    for row in summary["primary_components"]:
        p = row["primary"]
        bounds = row.get("completion_sensitive_effect_bounds")
        bounds = [bounds["lower"], bounds["upper"]] if bounds else None
        lines.append(f"| {row['candidate_id']} / {row['primary_id']} | {row['component_status']} | {number(p['effect'])} | {interval(p['adjusted_ci'])} | {interval(bounds)} | {number(p['holm_p_value'])} | {row['statistical_verdict']} |")
    lines += ["", "| Candidate | AND state | Frozen layouts | Saved/planned jobs | Missing jobs | Technical findings |", "|---|---|---:|---:|---:|---:|"]
    for candidate in summary["candidate_conjunctions"]:
        item = report["whole_candidate_inventories"][candidate["candidate_id"]]
        lines.append(f"| {candidate['candidate_id']} | {candidate['status']} | {item['expected_n_layouts']} | {item['saved_jobs']}/{item['expected_jobs']} | {len(item['missing_job_ids'])} | {len(item['technical_failures'])} |")
    for entry in report["lexical_supplements"]:
        lines += ["", "## Paired lexical supplement: " + entry["candidate_id"], "", f"Status: `{entry['status']}`. Same32 layouts; subset weights `{json.dumps(entry['family_weights'], sort_keys=True)}`. Primary weights `{json.dumps(entry['primary_family_weights'], sort_keys=True)}`.", "",
                  "| Descriptive contrast | Standard effect [completion bounds] | Alternate effect [completion bounds] | Paired encoding interaction [completion bounds] |", "|---|---|---|---|"]
        for row in entry["contrasts"]:
            cells = [number(row[k]["itt_effect"]) + " " + interval(row[k]["completion_bounds"]) for k in ("standard", "alternate", "encoding_interaction")]
            lines.append("| " + row["role"] + " | " + " | ".join(cells) + " |")
        lines += ["", f"Descriptive transfer gate met: `{entry['descriptive_transfer_gate_met']}`. No lexical significance or equivalence test is performed."]
    lines += ["", *report["interpretation"], "", "An independent final raw-result and report audit is required separately. Exact inputs, failures, denominators and statistical methods are retained in the companion JSON.", ""]
    return "\n".join(lines)


def write_report(report, output_stem):
    stem = Path(output_stem).resolve()
    paths = [Path(str(stem) + suffix) for suffix in (".json", ".md")]
    require(not any(path.exists() for path in paths), "immutable report output already exists")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for path, value in zip(paths, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", markdown(report)), strict=True):
        with path.open("x") as handle:
            handle.write(value)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--at-cutoff", action="store_true")
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    require(not any(output.is_relative_to(campaign / folder) for folder in
                    ("results", "specs", "datasets", "bases", "freezes", "planning", "source_snapshots", "reviews", "leases", "environment")), "output cannot be inside an input tree")
    bundle = load_bundle(campaign, args.freeze_id, at_cutoff=args.at_cutoff)
    report = build_report(bundle, at_cutoff=args.at_cutoff)
    for path, digest in report["provenance"]["input_file_sha256"].items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "input changed during analysis")
    print(json.dumps(write_report(report, output), indent=2))


if __name__ == "__main__":
    main()
