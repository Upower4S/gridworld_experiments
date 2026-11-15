"""Forced factual-readout derivatives of physical input-fact discovery cohorts.

This pure adapter never reads results or loads a tokenizer/model. The caller
must verify source registry status and the source file SHA before registration.
The semantic specification fingerprint here is distinct from that file hash.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any

from .environments import prompt_text
from .fact_readout import CLAIM_SCOPE, build_fact_probes
from .input_sites import _physical_counterfactual
from .registry import fingerprint


PATCH_CONDITION = re.compile(r"(identity|factspan|random_factspan|lasttoken)_layer(\d+)")


def _positions(job: dict[str, Any], role: str, *, required: bool) -> None:
    ids = job.get(role + "_token_ids")
    if not required and ids is None:
        return
    if not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Source token IDs must be frozen nonempty integer lists")
    positions = job["selection"].get(role + "_positions")
    if not isinstance(positions, list) or not positions or any(type(value) is not int or not 0 <= value < len(ids) for value in positions):
        raise ValueError("Source explicit positions must be nonempty in-bounds integer lists")
    if len(set(positions)) != len(positions):
        raise ValueError("Source explicit positions must be unique")
    if job.get(role + "_positions", positions) != positions:
        raise ValueError("Source top-level and selection positions disagree")
    audit_positions = job.get("metadata", {}).get(role + "_patch_positions")
    if audit_positions is not None and audit_positions != positions:
        raise ValueError("Source metadata and selection positions disagree")


def _source_pair(job: dict[str, Any]) -> tuple[dict[str, Any], str]:
    metadata = job.get("metadata", {})
    key = "source_pair" if "source_pair" in metadata else "pair"
    pair = metadata.get(key)
    if not isinstance(pair, dict):
        raise ValueError("Source job requires its physical pair metadata")
    _physical_counterfactual(pair)
    for name in ("pair_id", "layout_id", "family", "partition"):
        if job.get(name) != pair.get(name):
            raise ValueError(f"Source job {name} disagrees with physical pair")
    if job.get("target_action") != pair["counterfactual_optimal_actions"][0] or job.get("original_action") != pair["recipient_optimal_actions"][0]:
        raise ValueError("Source frozen action labels disagree with physical pair")
    # Input-site normalization may replace an auxiliary donor, but cannot change
    # the recipient world or its matched physical counterfactual.
    if key == "source_pair" and isinstance(metadata.get("pair"), dict):
        normalized = metadata["pair"]
        for name in ("recipient", "counterfactual", "pair_id", "layout_id", "family", "partition", "intervention"):
            if normalized.get(name) != pair.get(name):
                raise ValueError("Normalized and raw source pairs disagree on physical contrast")
    return deepcopy(pair), "metadata." + key


def build_fact_readout_spec(source_spec: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Clone every input-fact job into an elicited forced factual-report cohort.

    Public API: ``build_fact_readout_spec(source_spec, experiment_id) -> dict``.
    Only registered discovery input-fact specifications with physical R/CF
    prompts and exact token IDs/spans are supported. Every source condition and
    seed is retained; invalid provenance or an incomplete source cohort fails
    the entire build. Auxiliary truths are never reused for other contexts.

    Ordinary ``comparisons`` are empty: ``fact_readout_comparisons`` contains
    the diagnostic endpoints, and ``source_action_comparisons`` preserves the
    original action comparisons for audit. A factual-readout analysis adapter
    is required; these jobs request no new native continuations.
    """
    from .campaign import make_job, render_analysis_prompt

    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Invalid experiment_id")
    if source_spec.get("partition") != "discovery":
        raise ValueError("Fact-readout derivative is discovery-only")
    source_id = source_spec.get("experiment_id")
    if not isinstance(source_id, str) or not source_id or experiment_id == source_id:
        raise ValueError("Derivative requires a distinct valid source experiment ID")
    if not str(source_spec.get("claim_scope", "")).startswith("input_fact_information_routing"):
        raise ValueError("Only physical input-fact information-routing sources are supported")
    source_jobs = source_spec.get("jobs")
    if not isinstance(source_jobs, list) or not source_jobs:
        raise ValueError("Source cohort must contain jobs")
    if len({job.get("job_id") for job in source_jobs}) != len(source_jobs):
        raise ValueError("Duplicate source job IDs")
    source_hash = fingerprint(source_spec)
    jobs, source_map, probe_manifest = [], {}, {}
    inventories: dict[str, set[tuple[str, int]]] = defaultdict(set)
    pair_layout, pair_hashes = {}, {}
    condition_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for source_job in source_jobs:
        values = {key: deepcopy(value) for key, value in source_job.items() if key not in {"job_id", "experiment_id"}}
        expected_id = make_job(source_id, **values)["job_id"]
        if source_job.get("job_id") != expected_id or source_job.get("experiment_id") != source_id:
            raise ValueError("Source job ID or experiment does not match its complete registered definition")
        if source_job.get("partition") != "discovery" or source_job.get("metadata", {}).get("claim_scope") != "input_fact_information_routing":
            raise ValueError("Source job is outside the physical input-fact discovery scope")
        if "fact_probes" in source_job:
            raise ValueError("Source already contains factual probes; use the original input-fact source")
        condition, seed, pair_id = source_job["condition"], source_job["seed"], source_job["pair_id"]
        if type(seed) is not int or seed < 0 or (condition, seed) in inventories[pair_id]:
            raise ValueError("Invalid or duplicate source pair/condition/seed")
        inventories[pair_id].add((condition, seed))
        condition_keys[condition].add((pair_id, seed))
        pair, pair_origin = _source_pair(source_job)
        physical_hash = fingerprint(pair)
        if pair_id in pair_hashes and pair_hashes[pair_id] != physical_hash:
            raise ValueError("Physical pair definitions differ within a source cohort")
        pair_hashes[pair_id], pair_layout[pair_id] = physical_hash, pair["layout_id"]
        texts = {role: render_analysis_prompt(prompt_text(pair[role])) for role in ("recipient", "counterfactual")}
        patch_match = PATCH_CONDITION.fullmatch(condition)
        if condition in {"baseline", "counterfactual_text"}:
            expected_recipient = "recipient" if condition == "baseline" else "counterfactual"
            if source_job["intervention"].get("layers") or source_job.get("donor_text") is not None:
                raise ValueError("Physical source baseline unexpectedly contains an intervention or donor")
            expected_donor = None
        elif patch_match:
            kind, layer = patch_match.group(1), int(patch_match.group(2))
            expected_recipient, expected_donor = "recipient", "recipient" if kind == "identity" else "counterfactual"
            if source_job["intervention"].get("layers") != [layer] or not 0 <= layer < 24:
                raise ValueError("Source condition and intervention layer disagree")
            if bool(source_job["intervention"].get("random_control", False)) != (kind == "random_factspan"):
                raise ValueError("Source random-control flag disagrees with physical condition")
        else:
            raise ValueError("Unsupported source condition or arbitrary context")
        if source_job["recipient_text"] != texts[expected_recipient] or (expected_donor is not None and source_job.get("donor_text") != texts[expected_donor]):
            raise ValueError("Source inputs are not the physical R/CF contexts bound to these truths")
        if source_job.get("selection", {}).get("scope") != "explicit":
            raise ValueError("Source must freeze explicit intervention positions")
        _positions(source_job, "recipient", required=True)
        _positions(source_job, "donor", required=expected_donor is not None)
        if expected_donor is not None and len(source_job["selection"]["recipient_positions"]) != len(source_job["selection"]["donor_positions"]):
            raise ValueError("Source donor and recipient spans have unequal widths")
        probes = build_fact_probes(pair)
        target = next(probe for probe in probes if probe["role"] == "target_fact")
        if not target["available"] or target["recipient_value"] == target["counterfactual_value"]:
            raise ValueError("Target factual probe must change across physical worlds")
        for probe in probes:
            if probe["role"] == "unchanged_fact_control" and probe["available"] and probe["recipient_value"] != probe["counterfactual_value"]:
                raise ValueError("Auxiliary factual control changes across physical worlds")
        source_job_hash = fingerprint(source_job)
        diagnostic = {"claim_scope": CLAIM_SCOPE, "source_experiment_id": source_id,
                      "source_spec_fingerprint": source_hash, "source_job_id": source_job["job_id"],
                      "source_job_fingerprint": source_job_hash, "source_pair_fingerprint": physical_hash,
                      "physical_pair_metadata_origin": pair_origin, "unpatched_physical_context": expected_recipient,
                      "unpatched_context_truths": {probe["id"]: probe[expected_recipient + "_value"] for probe in probes if probe["available"]},
                      "source_natural_generation_requested": source_job.get("natural", False),
                      "native_continuation_requested": False, "source_action_outcomes_not_reestimated": True,
                      "semantic_headline_eligible": False, "latent_decoding": False, "mediation_evidence": False,
                      "selection_uses_model_outputs": False}
        values.update(natural=False, fact_probes=probes, fact_readout=diagnostic,
                      semantic_test=False, semantic_headline_eligible=False, claim_scope=CLAIM_SCOPE)
        values["metadata"]["fact_readout"] = deepcopy(diagnostic)
        derived = make_job(experiment_id, **values)
        jobs.append(derived)
        source_map[source_job["job_id"]] = {"derived_job_id": derived["job_id"], "source_job_fingerprint": source_job_hash}
        probe_manifest.setdefault(pair_id, {"pair_id": pair_id, "layout_id": pair["layout_id"],
                                            "probes": deepcopy(probes), "source_pair_fingerprint": physical_hash})
    expected_inventory = set.union(*inventories.values())
    if any(inventory != expected_inventory for inventory in inventories.values()):
        raise ValueError("Source cohort has incomplete condition/seed inventories across pairs")
    conditions = {condition for condition, _ in expected_inventory}
    if not {"baseline", "counterfactual_text"} <= conditions or not any(PATCH_CONDITION.fullmatch(condition) for condition in conditions):
        raise ValueError("Source cohort requires physical baselines and input-fact interventions")
    source_layers = {int(PATCH_CONDITION.fullmatch(condition).group(2)) for condition in conditions if PATCH_CONDITION.fullmatch(condition)}
    full_conditions = {"baseline", "counterfactual_text"} | {
        f"{kind}_layer{layer}" for layer in source_layers for kind in ("identity", "factspan", "random_factspan", "lasttoken")}
    source_seeds = {seed for _, seed in expected_inventory}
    if expected_inventory != {(condition, seed) for condition in full_conditions for seed in source_seeds}:
        raise ValueError("Source input-fact control matrix or seed inventory is incomplete")
    factual_comparisons = []
    for source_comparison in source_spec.get("comparisons", []):
        common = condition_keys[source_comparison["treatment"]] & condition_keys[source_comparison["control"]]
        if "eligible_pair_ids" in source_comparison:
            common = {key for key in common if key[0] in source_comparison["eligible_pair_ids"]}
        for probe_id in ("target_initial_fact", "unchanged_remote_grid_fact"):
            # Unavailable auxiliary reports remain in the complete cohort and
            # must be reported unavailable by the diagnostic analysis adapter.
            factual_comparisons.append({"id": f"{experiment_id}_{fingerprint(source_comparison)[:12]}_{probe_id}",
                "experiment_id": experiment_id, "partition": "discovery", "treatment": source_comparison["treatment"],
                "control": source_comparison["control"], "probe_id": probe_id,
                "role": "elicited_target_fact_transfer" if probe_id == "target_initial_fact" else "unchanged_fact_report_stability",
                "endpoint": "elicited_forced_fact_report", "probability_key": "counterfactual_truth_q",
                "eligible_pair_ids": sorted({key[0] for key in common}), "expected_n_pairs": len(common),
                "expected_n_layouts": len({pair_layout[key[0]] for key in common}),
                "source_comparison": deepcopy(source_comparison), "claim_scope": CLAIM_SCOPE,
                "semantic_headline_eligible": False, "mediation_evidence": False,
                "outcome_dependent_exclusion_permitted": False})
    result = deepcopy(source_spec)
    result.update(experiment_id=experiment_id, jobs=jobs, comparisons=[], fact_readout_comparisons=factual_comparisons,
                  source_action_comparisons=deepcopy(source_spec.get("comparisons", [])),
                  source_ineligible_action_comparisons=deepcopy(source_spec.get("ineligible_comparisons", [])),
                  ineligible_comparisons=[], source_experiment_id=source_id, source_spec_fingerprint=source_hash,
                  source_job_map=source_map, fact_probe_manifest=list(probe_manifest.values()),
                  n_source_jobs=len(source_jobs), n_derived_jobs=len(jobs), n_source_pairs=len(inventories),
                  complete_source_cohort_preserved=True, semantic_headline_eligible=False, claim_scope=CLAIM_SCOPE,
                  primary_estimand="layout-weighted paired change in elicited forced counterfactual-fact report probability",
                  secondary_estimands=["unchanged-fact report stability", "Boolean candidate probability mass"],
                  hypothesis="An input-fact intervention may change an explicitly elicited factual report; this diagnostic does not establish latent belief decoding, native action mediation, or semantic causality.")
    result.setdefault("protocol", {}).update(natural=False, native_continuation_requested=False,
        claim_scope=CLAIM_SCOPE, semantic_headline_eligible=False, requires_fact_readout_runner=True,
        requires_fact_readout_analysis_adapter=True, source_file_sha256_verification_required=True,
        source_retirement_verification_required=True, original_token_ids_and_positions_preserved=True,
        no_model_based_case_selection=True)
    return result
