"""Exact prequery plan, publication and readiness for the co-primary campaign.

Planning artifacts remain outside registered specs/datasets. Independent review
binds the complete plan before one immutable freeze is published. Readiness
reconstructs every planned cohort, including the restricted lexical repetition.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .confirmation_contract import inference_cutoff, remaining_inference_seconds, validate_budget, validate_primary_spec, validate_registry_configurations
from .confirmation_profiles import derive_candidate_configuration
from .confirmation_sources import existing_comparison_origin, prospective_natural_norm_origin, read_closed_native_battery
from .discovery_selection import POLICY_SHA256, evaluate_closed_discovery
from .lexical_generalization import CANONICAL, renderer_manifest
from .registry import append_event, file_sha256, fingerprint, immutable_json, now_utc, register_spec

MODE = "co_primary_native_v4"
RULE_SOURCES = ("confirmation_rendering.py", "frozen_contextual_input.py", "frozen_subspace.py",
                "input_sites.py", "contextual_input.py", "lexical_generalization.py", "confirmation_sampling.py")


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _identifier(value: Any) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None, "invalid frozen identity")
    return value


def _read(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate frozen JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    fingerprint(value)
    return value, hashlib.sha256(raw).hexdigest()


def _bound(path: str, digest: str, *, within: Path | None = None) -> Any:
    _require(isinstance(path, str) and Path(path).is_absolute(), "bound artifact needs absolute path")
    p = Path(path)
    _require(str(p.resolve(strict=True)) == path, "planned artifacts require canonical paths")
    if within is not None:
        _require(p.is_relative_to(within.resolve()), "planned artifact outside designated directory")
    value, actual = _read(p)
    _require(actual == digest, "bound artifact changed: " + path)
    return value


def analysis_configuration() -> dict:
    from .selection import ANALYSIS_CONFIG
    return {**ANALYSIS_CONFIG, "version": 4, "primary_family": "frozen_global_co_primary_registry",
            "candidate_decision": "both_components_required", "missing_or_invalid_primary_p_value": 1.0}


def select_budget_allocation(protocols: Mapping[str, str], seconds_per_job: float,
                             remaining_seconds: float, *, lexical_omissions=()) -> dict:
    """Choose the largest common N; omit all optional repeats if minimum cannot fit."""
    from .confirmation_contract import PROTOCOLS
    _require(bool(protocols) and 1 <= len(protocols) <= 2 and len(set(protocols.values())) == len(protocols), "one profile per question required")
    _require(set(lexical_omissions) <= set(protocols), "unknown lexical feasibility omission")
    supplements = {cid: cid not in lexical_omissions for cid in protocols}
    for optional in (supplements, dict.fromkeys(protocols, False)):
        for n in (128, 64, 32):
            configs = {cid: {"protocol_kind": protocol, "n_layouts": n, "generation_seeds": [42]} for cid, protocol in protocols.items()}
            _require(all(protocol in PROTOCOLS for protocol in protocols.values()), "unknown allocation protocol")
            try:
                budget = validate_budget(configs, optional, seconds_per_job, remaining_seconds)
            except ValueError as exc:
                if "exceeds" not in str(exc):
                    raise
                continue
            return {"n_layouts": n, "supplements": optional, "budget": budget,
                    "lexical_omissions_due_to_minimum_budget": sorted(cid for cid in protocols if supplements[cid] and not optional[cid])}
    raise ValueError("qualified profiles cannot fit even standard-only N32 before the inference cutoff")


def eligibility_rule(repo: Path, descriptor: Mapping[str, Any], *, joint: bool) -> dict:
    """Bind the designated adapter code and preselection geometry, never outcomes."""
    _require(type(joint) is bool, "explicit joint-renderer decision required")
    source_hashes = {"src/activation_causality/" + name:
                     file_sha256(repo / "src/activation_causality" / name) for name in RULE_SOURCES}
    value = {"descriptor": dict(descriptor), "require_joint_renderer_eligibility": joint,
             "renderer": renderer_manifest(CANONICAL), "sources": source_hashes,
             "controls": "every fixed condition on every retained layout",
             "aliases": "all physical, donor, control and D4 topology aliases excluded globally",
             "basis": "existing authenticated training basis without refitting"}
    return {"identifier": "complete_frozen_controls_and_canonical_aliases_v4", "sha256": fingerprint(value)}


def candidate_configuration(repo: Path, descriptor: Mapping[str, Any], source: Mapping[str, Any], *,
                            candidate_id: str, n_layouts: int, dataset_seed: int, tokenizer: Any, joint: bool) -> dict:
    layer = descriptor["layer"]
    if descriptor["protocol_kind"] == "contextual_input":
        treatment = f"layer{layer}_natural"
        adapter = "src/activation_causality/frozen_contextual_input.py"
        origins = {
            "contextual_fact_contrast": existing_comparison_origin(source, treatment, f"layer{layer}_same_fact"),
            "natural_norm_direction_contrast": prospective_natural_norm_origin(source, repo, layer=layer,
                adapter_path=adapter, adapter_sha256=file_sha256(repo / adapter))}
    else:
        label = "0p5" if descriptor["selected_dose"] == 0.5 else "1"
        treatment = "subspace_forward_" + label
        origins = {"subspace_efficacy": existing_comparison_origin(source, treatment, "baseline"),
                   "subspace_direction_contrast": existing_comparison_origin(source, treatment, "projected_norm_matched_random_subspace_" + label)}
    return derive_candidate_configuration(descriptor, source, candidate_id=candidate_id, n_layouts=n_layouts,
        dataset_seed=dataset_seed, tokenizer_metadata={"class": type(tokenizer).__name__, "name_or_path": tokenizer.name_or_path,
        "add_special_tokens": False, "context_renderer": "campaign.render_analysis_prompt"},
        renderer_binding=renderer_manifest(CANONICAL), require_joint_renderer_eligibility=joint,
        eligibility_rule=eligibility_rule(repo, descriptor, joint=joint), primary_origins=origins)


def layout_aliases(value: Any) -> set[str]:
    """Collect declared identities and recompute topology for every nested state."""
    from .environments import canonical_layout_fingerprint
    aliases: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("layout_id"), str):
            aliases.add(value["layout_id"])
        for key in ("layout_aliases", "variant_layout_ids", "training_layout_aliases", "training_layout_ids"):
            items = value.get(key, [])
            if isinstance(items, list):
                _require(all(isinstance(item, str) and item for item in items), "invalid nested alias inventory")
                aliases.update(items)
            elif isinstance(items, dict):
                for values in items.values():
                    _require(isinstance(values, list) and all(isinstance(item, str) for item in values), "invalid alias map")
                    aliases.update(values)
        if {"grid", "agent", "has_key"} <= value.keys():
            aliases.add(canonical_layout_fingerprint(value))
        for child in value.values():
            aliases.update(layout_aliases(child))
    elif isinstance(value, list):
        for child in value:
            aliases.update(layout_aliases(child))
    return aliases


def prior_exclusion_snapshot(campaign: Path, *, planned_experiment_ids=()) -> dict:
    """No result files are read. Registered discovery/historical inputs count."""
    allowed = set(planned_experiment_ids)
    paths = sorted(p for folder in ("specs", "datasets") for p in (campaign / folder).glob("*.json") if p.stem not in allowed)
    manifest, aliases = {}, set()
    for path in paths:
        value, digest = _read(path)
        _require(value.get("partition") != "confirmation", "unplanned registered confirmation already exists")
        manifest[str(path.resolve())] = digest
        aliases.update(layout_aliases(value))
    current = sorted(p for folder in ("specs", "datasets") for p in (campaign / folder).glob("*.json") if p.stem not in allowed)
    _require(paths == current and all(file_sha256(Path(p)) == sha for p, sha in manifest.items()), "prior exclusion corpus changed during capture")
    return {"format": "confirmation_prior_registered_aliases_v1", "files_sha256": manifest,
            "layout_aliases": sorted(aliases), "layout_aliases_sha256": fingerprint(sorted(aliases)), "result_files_read": False}


def _sampling_pairs(state: dict, configurations: dict) -> dict[str, list[dict]]:
    from .confirmation_sampling import FORMAT, _identity
    _require(state.get("format") == FORMAT and state.get("state_sha256") == fingerprint({k: v for k, v in state.items() if k != "state_sha256"}), "sampling state fingerprint mismatch")
    identity = _identity(state["identity"])
    _require(state["identity_sha256"] == fingerprint(identity) and state.get("status") == "complete"
             and state.get("in_progress") is None and state.get("clock_override_used") is False
             and state.get("model_queries") == 0 and state.get("registered_specification") is False
             and state.get("registered_dataset") is False, "final plan requires completed unqueried production sampling")
    started, finished = datetime.fromisoformat(state["started_utc"]), datetime.fromisoformat(state["last_utc"])
    deadline = datetime.fromisoformat(state["shared_sampling_deadline_utc"])
    effective = min(deadline, datetime.fromisoformat(identity["inference_cutoff_utc"]))
    _require(deadline == started + timedelta(hours=2) and started <= finished < effective
             and datetime.fromisoformat(state["effective_sampling_cutoff_utc"]) == effective,
             "sampling clock was reset or its deadline exceeded")
    definitions = {row["candidate_id"]: row for row in identity["candidates"]}
    _require(set(definitions) == set(configurations), "sampled candidate set differs from frozen family")
    _validate_sampling_accounting(state)
    pairs = {candidate: [] for candidate in configurations}
    global_aliases = set(identity["previously_used_layout_aliases"])
    seen = set()
    for record in state["attempted_pairs"]:
        pair = record["pair"]
        _require(record["pair_sha256"] == fingerprint(pair), "attempted physical pair changed")
        aliases = layout_aliases(pair) | set(record.get("layout_aliases", record.get("physical_layout_aliases", [])))
        if record["disposition"] != "accepted":
            global_aliases.update(aliases)
            continue
        candidate = record["candidate_id"]
        _require(candidate in pairs and pair["pair_id"] not in seen, "duplicate or unknown accepted pair")
        seen.add(pair["pair_id"])
        decision = record["builder_decision"]
        _require(record["builder_decision_sha256"] == fingerprint(decision) and decision["eligible"] is True
                 and decision["source_pair_sha256"] == fingerprint(pair), "accepted eligibility binding changed")
        _require(layout_aliases(pair) <= set(decision["layout_aliases"]), "accepted decision omitted a physical topology")
        aliases.update(decision["layout_aliases"])
        _require(not aliases & global_aliases, "fresh accepted layouts overlap prior inputs or another candidate")
        global_aliases.update(aliases)
        pairs[candidate].append(pair)
    for candidate, config in configurations.items():
        definition, values = definitions[candidate], pairs[candidate]
        _require(_same(definition["configuration"], config) and definition["configuration_sha256"] == fingerprint(config), "sampled configuration drift")
        _require([p["pair_id"] for p in values] == state["candidates"][candidate]["accepted_pair_ids"]
                 and len(values) == config["n_layouts"], "accepted count/order differs from fixed cohort")
        counts = {family: sum(p["family"] == family for p in values) for family in definition["families"]}
        _require(counts == definition["family_layout_counts"] == state["candidates"][candidate]["accepted_n_layouts_by_family"], "accepted family quotas differ")
    return pairs


def _validate_sampling_accounting(state: dict) -> None:
    """Replay reservations and decisions without drawing new pairs or reading outcomes."""
    from .confirmation_sampling import FAMILIES, _physical, _decision, _reported_aliases
    identity = state["identity"]
    aliases = set(identity["previously_used_layout_aliases"])
    counts = {c["candidate_id"]: {"accepted_pair_ids": [], "accepted_n_layouts_by_family": dict.fromkeys(c["families"], 0),
                               "next_attempt_by_family": dict.fromkeys(c["families"], 0)} for c in identity["candidates"]}
    consumed, last = 0, datetime.fromisoformat(state["started_utc"])
    for attempt in state["attempts"]:
        selected = None
        for candidate in identity["candidates"]:
            current = counts[candidate["candidate_id"]]
            for family in candidate["families"]:
                remaining = candidate["family_layout_counts"][family] - current["accepted_n_layouts_by_family"][family]
                if remaining:
                    selected = candidate, current, family, remaining
                    break
            if selected:
                break
        _require(selected is not None, "sampling added an attempt after all quotas were filled")
        candidate, current, family, remaining = selected
        cid = candidate["candidate_id"]
        index = current["next_attempt_by_family"][family]
        _require(index < 100, "sampling exceeded the fixed family attempt budget")
        expected_call = {"n_layouts": remaining, "seed": identity["dataset_seed"] + 1000 * FAMILIES.index(family) + index,
                         "partition": "confirmation", "families": [family], "excluded_layout_ids": sorted(aliases),
                         "max_attempts": max(1000, 3000 * remaining), **deepcopy(identity["generator_parameters"])}
        batch_id = fingerprint({"identity_sha256": state["identity_sha256"], "candidate_id": cid, "family": family, "attempt": index})
        _require(attempt["batch_id"] == batch_id and attempt["candidate_id"] == cid and attempt["family"] == family
                 and type(attempt["attempt"]) is int and attempt["attempt"] == index and attempt["status"] == "complete"
                 and attempt["candidate_definition_sha256"] == fingerprint(candidate)
                 and _same(attempt["generator_call"], expected_call)
                 and attempt["generator_call_sha256"] == fingerprint(expected_call), "reserved generator call/seed/order/fingerprint changed")
        started, finished = datetime.fromisoformat(attempt["started_utc"]), datetime.fromisoformat(attempt["finished_utc"])
        _require(last <= started <= finished <= datetime.fromisoformat(state["last_utc"]), "sampling attempt timestamps are inconsistent")
        last = finished
        current["next_attempt_by_family"][family] += 1
        records = state["attempted_pairs"][consumed:consumed + remaining]
        _require(len(records) == remaining and type(attempt["returned_pair_count"]) is int
                 and attempt["returned_pair_count"] == remaining and attempt["partial_pair_bytes_available"] is True,
                 "completed generator batch was shortened or expanded")
        physical_aliases = set(aliases)
        for position, record in enumerate(records):
            _require(record["batch_id"] == batch_id and record["candidate_id"] == cid and record["family"] == family
                     and type(record["returned_index"]) is int and record["returned_index"] == position
                     and record["pair_sha256"] == fingerprint(record["pair"]), "returned batch identity/order changed")
            physical = _physical(record["pair"], family, generator_call=expected_call)
            _require(record["physical_layout_aliases"] == sorted(physical)
                     and record["global_aliases_before_pair"] == sorted(physical_aliases), "physical alias reservation changed")
            physical_aliases.update(physical)
        _require(attempt["returned_pairs_sha256"] == fingerprint([r["pair"] for r in records]), "returned batch fingerprint changed")
        assessment = attempt["builder_assessment"]
        _require(attempt["builder_assessment_sha256"] == fingerprint(assessment)
                 and assessment["eligibility_builder_sha256"] == candidate["eligibility_builder_sha256"], "batch eligibility assessment changed")
        decisions = assessment["decisions"]
        _require(len(decisions) == remaining and len({d["pair_id"] for d in decisions}) == remaining, "batch decisions missing or duplicated")
        by_id = {d["pair_id"]: d for d in decisions}
        accepted = 0
        running = set(aliases)
        reported = set()
        for decision in decisions:
            reported.update(_reported_aliases(decision))
        for record in records:
            pair = record["pair"]
            decision = _decision(by_id[pair["pair_id"]], pair, candidate)
            _require(_same(record["builder_decision"], decision)
                     and record["builder_decision_sha256"] == fingerprint(decision), "per-pair decision differs from its batch assessment")
            pair_aliases = set(decision["layout_aliases"])
            overlap = pair_aliases & running
            retain = decision["eligible"] and not overlap
            reason = None if retain else "previously_used_or_attempted_layout_alias" if overlap else decision["reason"]
            _require(record["layout_aliases"] == sorted(pair_aliases) and record["conflicting_layout_aliases"] == sorted(overlap)
                     and record["disposition"] == ("accepted" if retain else "rejected") and record["reason"] == reason,
                     "saved acceptance/disposition or rejected alias accounting differs")
            if retain:
                current["accepted_pair_ids"].append(pair["pair_id"])
                current["accepted_n_layouts_by_family"][family] += 1
                accepted += 1
            running.update(pair_aliases)
        _require(type(attempt["accepted_pair_count"]) is int and attempt["accepted_pair_count"] == accepted
                 and type(attempt["rejected_pair_count"]) is int and attempt["rejected_pair_count"] == remaining - accepted,
                 "batch acceptance denominator changed")
        aliases = physical_aliases | reported | running
        consumed += remaining
    _require(consumed == len(state["attempted_pairs"]) and _same(counts, state["candidates"]), "sampling counter/pair accounting changed")
    _require(state["global_layout_aliases"] == sorted(aliases), "final global alias reservations changed")
    _require(all(counts[c["candidate_id"]]["accepted_n_layouts_by_family"] == c["family_layout_counts"] for c in identity["candidates"]),
             "completed sampling does not fill every fixed quota")


def _external_context(campaign: Path, repo: Path, plan: dict) -> tuple[dict, Any]:
    from .basis_runtime import activation_representation
    from .campaign import runtime_fingerprint
    from .selection import _model_identity, _confirmation_tokenizer
    runtime = runtime_fingerprint(repo)
    _require(_same(runtime["sources"], plan["implementation"]), "frozen implementation changed")
    _require(file_sha256(repo / "scripts/30_activation_causality_campaign.py") == plan["worker_entrypoint_sha256"], "worker entrypoint changed")
    _require(_same(_model_identity(campaign), plan["model_identity"]), "actual model snapshot changed")
    _require(_same(activation_representation(runtime), plan["activation_representation"]), "activation capture representation changed")
    from .confirmation_planning import verify_preflight_prerequisite
    prerequisite = verify_preflight_prerequisite(campaign, repo, runtime)
    _require(_same(prerequisite, plan.get("technical_preflight"))
             and prerequisite["model_snapshot_identity_sha256"] == plan["model_identity"]["identity_sha256"],
             "reviewed real full-write preflight prerequisite is missing or changed")
    battery = read_closed_native_battery(campaign, repo, plan["audit_bindings"])
    actual = evaluate_closed_discovery(campaign, repo, plan["audit_bindings"])
    record = _bound(plan["scientific_selection"]["path"], plan["scientific_selection"]["sha256"])
    _require(_same(record["decision"], actual), "scientific selection differs from complete adopted-policy evaluation")
    _require(actual["policy_sha256"] == POLICY_SHA256, "selection policy differs")
    return {"battery": battery, "decision": actual}, _confirmation_tokenizer(campaign)


def _recheck_external(campaign: Path, repo: Path, plan: dict, context: dict) -> None:
    """Close the global source/model/code bracket after complete adapter replay."""
    from .campaign import runtime_fingerprint
    from .selection import _model_identity, _source_result_paths
    for source in context["battery"]["sources"].values():
        eid = source["experiment_id"]
        jobs = {job["job_id"]: job for job in source["specification"]["jobs"]}
        paths = _source_result_paths(campaign, eid, jobs)
        _require({str(path.relative_to(campaign)) for path in paths} == set(source["source_result_hashes"]), "source inventory changed during plan validation")
        manifest = {str(campaign / "specs" / (eid + ".json")): source["source_spec_sha256"],
                    **{str(campaign / relative): digest for relative, digest in source["source_result_hashes"].items()},
                    str(repo / source["independent_audit"]["path"]): source["independent_audit"]["sha256"],
                    **{str(repo / relative): digest for relative, digest in source["independent_audit"]["input_files_sha256"].items()}}
        _require(all(file_sha256(Path(path)) == digest for path, digest in manifest.items()), "source or independent audit changed during plan validation")
        _require(not (campaign / "reviews" / (eid + "_retired.json")).exists(), "source retired during plan validation")
    runtime = runtime_fingerprint(repo)
    _require(_same(runtime["sources"], plan["implementation"])
             and file_sha256(repo / "scripts/30_activation_causality_campaign.py") == plan["worker_entrypoint_sha256"], "implementation changed during plan validation")
    _require(_same(_model_identity(campaign), plan["model_identity"]), "model snapshot changed during plan validation")
    from .confirmation_planning import verify_preflight_prerequisite
    _require(_same(verify_preflight_prerequisite(campaign, repo, runtime), plan["technical_preflight"]),
             "technical preflight prerequisite changed during plan validation")
    for manifest in plan["basis_manifests"].values():
        _require(all(file_sha256(Path(path)) == digest for path, digest in manifest["artifact_sha256"].items()), "basis changed during adapter replay")
        _require(not Path(manifest["source_retirement_guard"]["path"]).exists(), "basis training source retired during replay")


def validate_confirmation_plan(campaign: Path, repo: Path, plan: dict, *, freeze_time: str | None = None) -> dict:
    """Authenticate complete artifacts and reconstruct every exact planned job."""
    campaign, repo = campaign.resolve(), repo.resolve()
    _require(plan.get("confirmation_mode") == MODE and type(plan.get("schema_version")) is int and plan["schema_version"] == 4,
             "explicit version4 co-primary plan required")
    freeze_id = _identifier(plan["freeze_id"])
    _require(not {"frozen_at", "plan_path", "plan_sha256", "review_path", "review_sha256", "validated_budget"} & plan.keys(),
             "planning document cannot shadow publication metadata")
    config, campaign_sha = _read(campaign / "campaign.json")
    _require(plan["campaign_sha256"] == campaign_sha and plan["inference_cutoff"] == inference_cutoff(config["deadline"]), "campaign/deadline drift")
    _require(_same(plan["analysis_configuration"], analysis_configuration()), "analysis protocol drift")
    configurations = plan["candidate_configurations"]
    registry = validate_registry_configurations(plan["confirmation_primary_family"], configurations)
    experiments = plan["planned_experiments"]
    _require(isinstance(experiments, dict) and experiments, "exact planned specs/datasets required before freezing")
    for identifier in experiments:
        _identifier(identifier)
    prior = _bound(plan["prior_exclusions"]["path"], plan["prior_exclusions"]["sha256"])
    _require(_same(prior_exclusion_snapshot(campaign, planned_experiment_ids=experiments), prior), "prior input/alias corpus changed")
    state_ref = plan["sampling_state"]
    _require(state_ref["path"] == str(campaign / "planning/final_sampling_v1.json"), "one campaign-wide final sampler path is required")
    state = _bound(state_ref["path"], state_ref["sha256"], within=campaign / "planning")
    pairs = _sampling_pairs(state, configurations)
    _require(state["identity"]["policy_sha256"] == POLICY_SHA256
             and state["identity"]["campaign_deadline_utc"] == config["deadline"]
             and state["identity"]["inference_cutoff_utc"] == plan["inference_cutoff"], "sampling campaign/policy differs")
    _require(state["identity"]["generator_sha256"] == file_sha256(repo / "src/activation_causality/environments.py")
             and all(configuration["dataset_seed"] == state["identity"]["dataset_seed"] for configuration in configurations.values()),
             "sampler generator code or shared seed changed")
    _require(state["identity"]["previously_used_layout_aliases"] == prior["layout_aliases"], "sampling omitted prior registered aliases")
    from .environments import load_historical_layout_ids
    historical = state["identity"]["historical_exclusion"]
    _require(file_sha256(Path(historical["source_path"])) == historical["source_sha256"], "historical exclusion file changed")
    historical_aliases, history_audit = load_historical_layout_ids(historical["source_path"])
    _require(_same(history_audit, historical["source_exclusion_provenance"]), "historical topology audit changed")
    _require(not any(layout_aliases(pair) & historical_aliases for values in pairs.values() for pair in values), "confirmation overlaps historical topologies")
    context, tokenizer = _external_context(campaign, repo, plan)
    selected_ids = {value for value in context["decision"]["selected_scientific_profile_ids"].values() if value is not None}
    descriptors = {p["profile_id"]: p["descriptor"] for p in context["decision"]["profiles"]}
    actual_profiles, basis_metadata, basis_manifests = set(), {}, {}
    for cid, configuration in configurations.items():
        descriptor = configuration["source_profile_descriptor"]
        pid = descriptor["profile_id"]
        _require(pid in selected_ids and pid not in actual_profiles and _same(descriptor, descriptors[pid]), "candidate is not the uniquely selected scientific profile")
        actual_profiles.add(pid)
        expected = candidate_configuration(repo, descriptor, context["battery"]["sources"][descriptor["source_experiment_id"]],
            candidate_id=cid, n_layouts=configuration["n_layouts"], dataset_seed=configuration["dataset_seed"], tokenizer=tokenizer,
            joint=configuration["require_joint_renderer_eligibility"])
        _require(_same(expected, configuration), "candidate geometry/origins/eligibility differs from authenticated source derivation")
        if configuration["protocol_kind"] == "input_fact_subspace":
            from .confirmation_basis import validate_confirmation_basis
            basis_metadata[cid], basis_manifests[cid] = validate_confirmation_basis(configuration, configuration["basis_metadata"],
                model_snapshot_identity=plan["model_identity"], activation_representation=plan["activation_representation"])
    omissions = plan.get("qualified_unexecuted_profiles", {})
    _require(set(omissions) == selected_ids - actual_profiles and all(isinstance(reason, dict) and reason.get("reason") for reason in omissions.values()),
             "every omitted qualified profile requires an explicit retained feasibility reason, without backfill")
    _require(_same(basis_manifests, plan["basis_manifests"]), "basis artifact authenticity manifest changed")
    from .confirmation_rendering import build_primary, build_lexical_supplement, eligibility_decisions
    primary_ids = {item["candidate_id"]: eid for eid, item in experiments.items() if item["cohort_role"] == "primary"}
    _require(set(primary_ids) == set(configurations), "each sampled candidate requires a designated primary identity")
    builder_sha = file_sha256(repo / "src/activation_causality/confirmation_rendering.py")
    for definition in state["identity"]["candidates"]:
        cid = definition["candidate_id"]
        _require(definition["eligibility_builder_sha256"] == builder_sha
                 and _same(definition["eligibility_rule"], eligibility_rule(repo, configurations[cid]["source_profile_descriptor"],
                    joint=configurations[cid]["require_joint_renderer_eligibility"])), "sampler eligibility builder/rule changed")
    for attempt in state["attempts"]:
        _require(attempt["status"] == "complete", "complete final sampling contains unfinished attempts")
        cid = attempt["candidate_id"]
        batch_pairs = [record["pair"] for record in state["attempted_pairs"] if record["batch_id"] == attempt["batch_id"]]
        _require(attempt["returned_pairs_sha256"] == fingerprint(batch_pairs), "sampling returned batch changed")
        reconstructed = build_primary(batch_pairs, tokenizer, primary_ids[cid], configurations[cid], freeze_id,
                                      basis_metadata=basis_metadata.get(cid))
        assessment = {"eligibility_builder_sha256": builder_sha, "decisions": eligibility_decisions(reconstructed)}
        _require(_same(assessment, attempt["builder_assessment"])
                 and fingerprint(assessment) == attempt["builder_assessment_sha256"], "sampled eligibility proof differs from designated adapter replay")
    primary_specs, loaded_specs, supplements = {}, {}, {cid: False for cid in configurations}
    for eid, item in experiments.items():
        spec = _bound(item["spec_path"], item["spec_sha256"], within=campaign / "planning")
        dataset = _bound(item["dataset_path"], item["dataset_sha256"], within=campaign / "planning")
        cid, role = spec.get("candidate_id"), spec.get("cohort_role")
        _require(cid in configurations and spec.get("freeze_id") == freeze_id and spec.get("experiment_id") == eid
                 and item["candidate_id"] == cid and item["cohort_role"] == role, "planned artifact identity mismatch")
        _require(dataset.get("experiment_id") == eid and dataset.get("partition") == "confirmation"
                 and dataset.get("candidate_id") == cid and dataset.get("cohort_role") == role, "dataset identity/role mismatch")
        loaded_specs[eid] = spec
        if role == "primary":
            _require(cid not in primary_specs and _same(dataset.get("pairs"), pairs[cid]), "duplicate primary or sampled pair drift")
            expected = build_primary(pairs[cid], tokenizer, eid, configurations[cid], freeze_id, basis_metadata=basis_metadata.get(cid))
            _require(_same(spec, expected), "primary spec differs from exact designated reconstruction")
            validate_primary_spec(spec, configurations[cid], registry)
            primary_specs[cid] = spec
        else:
            _require(role == "lexical_supplement", "unknown confirmation cohort role")
    _require(set(primary_specs) == set(configurations), "every candidate requires exactly one planned primary")
    for eid, spec in loaded_specs.items():
        if spec["cohort_role"] != "lexical_supplement":
            continue
        cid = spec["candidate_id"]
        _require(not supplements[cid], "only one lexical supplement per primary is permitted")
        expected = build_lexical_supplement(primary_specs[cid], tokenizer, eid, configurations[cid], freeze_id,
                                           basis_metadata=basis_metadata.get(cid))
        _require(_same(spec, expected), "lexical supplement differs from exact parent/hash-subset reconstruction")
        dataset = _bound(experiments[eid]["dataset_path"], experiments[eid]["dataset_sha256"], within=campaign / "planning")
        expected_pairs = list({job["pair_id"]: job["metadata"]["source_pair"] for job in expected["jobs"]}.values())
        _require(_same(dataset["pairs"], expected_pairs), "lexical dataset differs from its exact repeated parent subset")
        supplements[cid] = True
    _require(_same(supplements, plan["supplements"])
             and all(configurations[cid]["require_joint_renderer_eligibility"] is enabled for cid, enabled in supplements.items()),
             "planned supplements and whole-primary joint eligibility disagree")
    minimum = context["battery"]["minimum_budget_seconds_per_native_job"]
    _require(plan["budget_seconds_per_native_job"] >= minimum, "budget understates whole-cohort conservative throughput")
    if plan["budget_seconds_per_native_job"] != minimum:
        _require(bool(plan.get("non_outcome_budget_increase_reason")), "budget increase must have a documented non-outcome reason")
    lexical_omissions = plan.get("lexical_feasibility_omissions", {})
    _require(isinstance(lexical_omissions, dict), "lexical omissions must bind retained feasibility records")
    for cid, ref in lexical_omissions.items():
        evidence = _bound(ref["path"], ref["sha256"])
        _require(evidence.get("candidate_id") == cid and evidence.get("renderer_feasible") is False
                 and evidence.get("reason") and evidence.get("model_queries") == 0, "lexical omission requires an unqueried fixed-renderer feasibility failure")
    allocation = select_budget_allocation({cid: c["protocol_kind"] for cid, c in configurations.items()},
        plan["budget_seconds_per_native_job"], remaining_inference_seconds(config["deadline"], plan["inference_cutoff"], now=state["started_utc"]),
        lexical_omissions=lexical_omissions)
    _require(all(c["n_layouts"] == allocation["n_layouts"] for c in configurations.values())
             and _same(supplements, allocation["supplements"]), "sampling did not use the largest common prospectively budgeted allocation")
    budget = validate_budget(configurations, supplements, plan["budget_seconds_per_native_job"],
        remaining_inference_seconds(config["deadline"], plan["inference_cutoff"], now=freeze_time or now_utc()))
    # The global observation bracket includes every planned immutable payload.
    for item in experiments.values():
        for prefix in ("spec", "dataset"):
            _require(file_sha256(Path(item[prefix + "_path"])) == item[prefix + "_sha256"], "planned artifact changed during rebuild")
    for ref in (plan["sampling_state"], plan["scientific_selection"], plan["prior_exclusions"]):
        _require(file_sha256(Path(ref["path"])) == ref["sha256"], "planning input changed during validation")
    _require(file_sha256(Path(historical["source_path"])) == historical["source_sha256"]
             and file_sha256(campaign / "campaign.json") == campaign_sha
             and _same(prior_exclusion_snapshot(campaign, planned_experiment_ids=experiments), prior), "campaign/exclusion inputs changed during rebuild")
    _recheck_external(campaign, repo, plan, context)
    return {"specifications": loaded_specs, "budget": budget}


def freeze_confirmation_plan(campaign: Path, repo: Path, plan_path: Path, review_path: Path) -> dict:
    """Publish only after an independent review binds this exact complete plan."""
    _require(plan_path.resolve().is_relative_to(campaign.resolve() / "planning") and plan_path.resolve() == plan_path.absolute(),
             "reviewed plan must use a canonical path inside campaign/planning")
    plan, digest = _read(plan_path)
    review, review_sha = _read(review_path)
    _require(review.get("approved") is True and review.get("freeze_id") == plan.get("freeze_id")
             and review.get("plan_sha256") == digest and review.get("plan_fingerprint") == fingerprint(plan),
             "independent review must approve the exact prepared plan")
    own = campaign / "freezes" / (_identifier(plan["freeze_id"]) + ".json")
    other = [path for path in (campaign / "freezes").glob("*.json") if path != own]
    _require(not other, "one immutable campaign-wide freeze is permitted")
    if own.exists():
        frozen, _ = _read(own)
        _require(frozen["plan_sha256"] == digest and frozen["review_sha256"] == review_sha, "cannot revise published confirmation freeze")
        return frozen
    _require(not any(_read(path)[0].get("partition") == "confirmation" for path in (campaign / "specs").glob("*.json")),
             "confirmation was registered before its reviewed freeze")
    _require(not any(list((campaign / "results" / eid).glob("*.json")) for eid in plan["planned_experiments"]),
             "planned confirmation outcomes already exist before freeze")
    timestamp = now_utc()
    checked = validate_confirmation_plan(campaign, repo, plan, freeze_time=timestamp)
    _require(file_sha256(plan_path) == digest and file_sha256(review_path) == review_sha, "plan/review changed during validation")
    timestamp = now_utc()
    config, _ = _read(campaign / "campaign.json")
    checked["budget"] = validate_budget(plan["candidate_configurations"], plan["supplements"], plan["budget_seconds_per_native_job"],
        remaining_inference_seconds(config["deadline"], plan["inference_cutoff"], now=timestamp))
    frozen = {**deepcopy(plan), "frozen_at": timestamp, "plan_path": str(plan_path.resolve()), "plan_sha256": digest,
              "review_path": str(review_path.resolve()), "review_sha256": review_sha,
              "validated_budget": checked["budget"]}
    immutable_json(own, frozen)
    append_event(campaign, "co_primary_confirmation_frozen", freeze_id=plan["freeze_id"],
                 plan_sha256=digest, review_sha256=review_sha, primary_family_size=plan["confirmation_primary_family"]["family_size"])
    return frozen


def validate_frozen_confirmation_ready(campaign: Path, repo: Path, spec: dict) -> None:
    freeze_path = campaign / "freezes" / (_identifier(spec["freeze_id"]) + ".json")
    _require(len(list((campaign / "freezes").glob("*.json"))) == 1, "one immutable campaign-wide freeze required")
    frozen, freeze_sha = _read(freeze_path)
    _require(frozen.get("confirmation_mode") == MODE, "new co-primary confirmation cannot use legacy readiness")
    plan = _bound(frozen["plan_path"], frozen["plan_sha256"], within=campaign / "planning")
    review = _bound(frozen["review_path"], frozen["review_sha256"])
    _require(review.get("approved") is True and review.get("plan_sha256") == frozen["plan_sha256"]
             and review.get("plan_fingerprint") == fingerprint(plan) and review.get("freeze_id") == spec["freeze_id"], "exact frozen plan approval changed")
    _require(all(key in frozen and _same(frozen[key], value) for key, value in plan.items()), "freeze shadowed the reviewed plan")
    checked = validate_confirmation_plan(campaign, repo, plan, freeze_time=frozen["frozen_at"])
    _require(_same(checked["budget"], frozen["validated_budget"]), "frozen budget record drift")
    eid = spec["experiment_id"]
    _require(eid in checked["specifications"] and _same(spec, checked["specifications"][eid]), "unplanned or changed registered confirmation")
    for identifier, item in plan["planned_experiments"].items():
        for folder, prefix in (("specs", "spec"), ("datasets", "dataset")):
            registered = campaign / folder / (identifier + ".json")
            if registered.exists():
                _require(file_sha256(registered) == item[prefix + "_sha256"], "registered artifact differs from planned bytes")
    _require(file_sha256(freeze_path) == freeze_sha, "freeze changed during readiness validation")


def register_frozen_plan(campaign: Path, repo: Path, freeze_id: str) -> list[str]:
    frozen, _ = _read(campaign / "freezes" / (_identifier(freeze_id) + ".json"))
    plan = _bound(frozen["plan_path"], frozen["plan_sha256"], within=campaign / "planning")
    checked = validate_confirmation_plan(campaign, repo, plan, freeze_time=frozen["frozen_at"])
    # Validate the independently reviewed frozen document before any publication.
    validate_frozen_confirmation_ready(campaign, repo, next(iter(checked["specifications"].values())))
    for eid, spec in checked["specifications"].items():
        item = plan["planned_experiments"][eid]
        dataset = _bound(item["dataset_path"], item["dataset_sha256"])
        immutable_json(campaign / "datasets" / (eid + ".json"), dataset)
        register_spec(campaign, spec)
        _require(file_sha256(campaign / "specs" / (eid + ".json")) == item["spec_sha256"]
                 and file_sha256(campaign / "datasets" / (eid + ".json")) == item["dataset_sha256"], "registered serialization differs from prepared artifact")
    return list(checked["specifications"])
