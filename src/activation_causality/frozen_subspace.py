"""Pure staged single-family subspace confirmation adapter.

No basis file is read or fitted, no model is loaded, and no freeze is certified.
The caller supplies validated metadata/configuration; independent freeze,
readiness, artifact and loaded-model verification remain mandatory. This first
adapter supports the canonical renderer only. Lexical-supplement glue is pending.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .environments import canonical_layout_fingerprint
from .input_sites import build_input_site_spec
from .registry import SCHEMA_VERSION, fingerprint


PROTOCOL = "input_fact_subspace"
CLAIM_SCOPE = "input_fact_subspace_routing"
ENDPOINT = "natural_target_action_intention_to_treat"
UNRESOLVED_POLICY = "count as zero target success; retain in intention-to-treat denominator; no forced fallback"
ALIGNMENT = "exact_equal_length_common_prefix_and_suffix"
NORM_TOLERANCE = 0.01
FAMILIES = ("wall", "key_possession", "door_status")
CONDITIONS = (
    "baseline", "donor_text", "full_transfer", "subspace_identity", "subspace_removal", "restoration_equivalent",
    "subspace_forward_0p5", "subspace_reverse_0p5", "random_subspace_0p5", "projected_norm_matched_random_subspace_0p5",
    "subspace_forward_1", "subspace_reverse_1", "random_subspace_1", "projected_norm_matched_random_subspace_1",
)
PRIMARY_IDS = ("subspace_efficacy", "subspace_direction_contrast")


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _aliases(pair: Mapping[str, Any]) -> set[str]:
    values = [pair.get("layout_id")]
    for container, field in ((pair, "layout_aliases"), (pair.get("metadata", {}), "variant_layout_ids")):
        aliases = container.get(field, [])
        _require(isinstance(aliases, list) and all(isinstance(x, str) and x for x in aliases), "invalid_layout_aliases")
        values.extend(aliases)
    _require(all(isinstance(value, str) and value for value in values), "invalid_layout_identity")
    for role in ("recipient", "counterfactual", "donor"):
        if role in pair:
            values.append(canonical_layout_fingerprint(pair[role]))
    return set(values)


def _configuration(configuration: Mapping[str, Any], basis_metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from .primary_family import REQUIRED_BINDING_FIELDS, validate_primary_family

    _require(isinstance(configuration, Mapping) and isinstance(basis_metadata, Mapping),
             "configuration and caller-validated basis_metadata must be mappings")
    config, basis = deepcopy(dict(configuration)), deepcopy(dict(basis_metadata))
    required = {"candidate_id", "family", "rank", "layer", "site", "selected_dose", "doses", "generation_seeds",
                "random_basis_seeds", "max_new_tokens", "temperature", "n_layouts", "condition_inventory", "tokenizer",
                "basis_path", "basis_sha256", "basis_metadata_path", "basis_metadata_sha256", "basis_metadata_fingerprint",
                "primary_comparisons", "source_experiment_id", "source_spec_sha256"}
    _require(required <= config.keys(), "missing frozen configuration fields: " + ",".join(sorted(required - config.keys())))
    _require(_name(config["candidate_id"]) and _name(config["source_experiment_id"]), "invalid frozen source/candidate identity")
    _require(config["family"] in FAMILIES and type(config["rank"]) is int and config["rank"] in {1, 4},
             "freeze one supported family and rank 1 or 4")
    _require(type(config["layer"]) is int and config["layer"] == 0 and config["site"] == "resid_post",
             "subspace confirmation requires layer0 resid_post")
    _require(type(config["n_layouts"]) is int and config["n_layouts"] in {32, 64, 128}, "freeze 32,64,128 layouts")
    _require(type(config["selected_dose"]) in {int, float} and config["selected_dose"] in {0.5, 1.0}, "select dose0.5 or1")
    _require(config["doses"] == [0.5, 1.0] and all(type(x) in {int, float} for x in config["doses"]), "retain both source doses")
    for key in ("generation_seeds", "random_basis_seeds"):
        _require(config[key] == [42] and all(type(x) is int for x in config[key]), "freeze the existing seed42 protocol")
    _require(type(config["max_new_tokens"]) is int and config["max_new_tokens"] == 4096
             and type(config["temperature"]) in {int, float} and config["temperature"] == 0.7, "freeze cap4096 temperature0.7")
    _require(config["condition_inventory"] == list(CONDITIONS), "retain the exact ordered fourteen-condition inventory")
    _require(not ({"families", "family_weights", "family_layout_counts", "renderer_override", "prompt_override",
                   "confirmation_family_size", "family_size"} & config.keys()), "mixed families, overrides and local multiplicity sizes are unsupported")
    fixed = {"protocol_kind": PROTOCOL, "natural": True, "operation": "subspace", "layers": [0],
             "renderer_id": "canonical_v1", "input_alignment_rule": ALIGNMENT,
             "selection": {"scope": "explicit", "alignment": ALIGNMENT}, "claim_scope": CLAIM_SCOPE,
             "semantic_headline_eligible": False, "primary_endpoint": ENDPOINT,
             "native_unresolved_policy": UNRESOLVED_POLICY, "conditions_per_layout": 14,
             "projected_norm_relative_tolerance": NORM_TOLERANCE}
    for key, expected in fixed.items():
        _require(key not in config or (config[key] == expected and
                                      (type(expected) is not bool or type(config[key]) is bool)), "contradictory frozen " + key)
    for key, expected in {"dose": config["selected_dose"], "seeds": [42], "cap": 4096,
                          "random_seed": 42, "layout_count": config["n_layouts"]}.items():
        _require(key not in config or (config[key] == expected and type(config[key]) is not bool), "contradictory frozen alias " + key)
    _require(basis.get("format") == "registered_task_basis_v1" and basis.get("model_provenance_version") == 2
             and basis.get("partition") == "discovery" and basis.get("training_only") is True,
             "require existing discovery-trained basis metadata with full snapshot provenance")
    for key in ("family", "rank", "layer", "site"):
        _require(basis.get(key) == config[key] and type(basis.get(key)) is type(config[key]), "basis/configuration mismatch: " + key)
    _require(basis.get("source_condition") == "factspan_layer0" and basis.get("hidden_dim") == 2880,
             "basis must use the registered layer0 input-fact representation")
    _require(_name(basis.get("basis_id")), "missing basis identity")
    for key in ("training_layout_ids", "training_pair_ids"):
        values = basis.get(key)
        _require(isinstance(values, list) and len(values) == len(set(values)) == 4
                 and all(isinstance(value, str) and value for value in values), "basis must retain four independent training layouts/pairs")
    aliases = basis.get("training_layout_aliases")
    _require(isinstance(aliases, list) and bool(aliases) and all(isinstance(x, str) and x for x in aliases)
             and set(basis["training_layout_ids"]) <= set(aliases), "basis training aliases missing or incomplete")
    for key in ("basis_path", "basis_metadata_path"):
        _require(isinstance(config[key], str) and Path(config[key]).is_absolute() and config[key] == basis.get(key),
                 "basis absolute path binding mismatch: " + key)
    for config_key, basis_key in (("basis_sha256", "basis_file_sha256"), ("basis_metadata_sha256", "basis_metadata_sha256")):
        _require(_sha(config[config_key]) and config[config_key] == basis.get(basis_key), "basis file/hash binding mismatch: " + config_key)
    _require(_sha(config["basis_metadata_fingerprint"]) and config["basis_metadata_fingerprint"] == fingerprint(basis),
             "supplied basis metadata differs from the frozen semantic fingerprint")
    for key in ("source_spec_sha256", "request_fingerprint", "model_snapshot_identity_sha256", "model_snapshot_audit_sha256",
                "selected_source_definitions_sha256", "basis_array_sha256"):
        _require(_sha(basis.get(key)), "missing basis provenance digest: " + key)
    _require(_sha(config["source_spec_sha256"]) and isinstance(basis.get("activation_representation"), dict)
             and bool(basis["activation_representation"]) and isinstance(basis.get("source_job_fingerprints"), dict)
             and len(basis["source_job_fingerprints"]) == 4, "missing source/activation provenance")
    _require(isinstance(config["tokenizer"], Mapping) and config["tokenizer"].get("add_special_tokens") is False
             and config["tokenizer"].get("context_renderer") == "campaign.render_analysis_prompt", "missing exact canonical tokenizer contract")
    selected = "0p5" if config["selected_dose"] == 0.5 else "1"
    treatment = "subspace_forward_" + selected
    matched = "projected_norm_matched_random_subspace_" + selected
    expected_intervention = {"layers": [0], "site": "resid_post", "operation": "subspace", "dose": config["selected_dose"],
                             "basis_path": config["basis_path"], "basis_sha256": config["basis_sha256"],
                             "basis_metadata_path": config["basis_metadata_path"], "basis_metadata_sha256": config["basis_metadata_sha256"],
                             "audit_projected_norm": True, "projected_norm_relative_tolerance": NORM_TOLERANCE}
    _require("intervention" not in config or config["intervention"] == expected_intervention,
             "frozen selected intervention differs from intended task-basis transfer")
    primaries = config["primary_comparisons"]
    _require(isinstance(primaries, Mapping) and set(primaries) == set(PRIMARY_IDS), "freeze exactly the two named subspace co-primaries")
    for primary, control in zip(PRIMARY_IDS, ("baseline", matched)):
        binding = primaries[primary]
        _require(isinstance(binding, Mapping) and set(binding) == REQUIRED_BINDING_FIELDS, "primary binding fields differ from the exact scalar-family contract")
        for key, expected in {"treatment": treatment, "control": control, "predicted_direction": "increase",
                              "expected_n_layouts": config["n_layouts"], "expected_n_pairs": config["n_layouts"],
                              "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
                              "target_action_basis": "unique_simulator_counterfactual_optimum",
                              "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False}.items():
            _require(binding.get(key) == expected, "primary binding contradiction: " + primary + ":" + key)
        _require(isinstance(binding["origin"], Mapping) and binding["origin"].get("kind") == "existing_source_comparison",
                 "subspace co-primaries require existing-source origins")
        provenance = binding["origin"].get("provenance")
        _require(isinstance(provenance, Mapping), "missing primary source provenance")
        for key in ("source_experiment_id", "source_spec_sha256"):
            _require(key not in provenance or provenance[key] == config[key], "primary origin contradicts frozen " + key)
    # Local schema validation only, not a global multiplicity-family assignment.
    validate_primary_family({"schema_version": 1, "candidates": {config["candidate_id"]: list(PRIMARY_IDS)},
                             "family_size": 2, "bindings": {config["candidate_id"]: dict(primaries)}})
    config.update(fixed, intervention=expected_intervention)
    return config, basis


def _expand(source: dict[str, Any], metadata: dict[str, Any], experiment_id: str,
            inline_sha: str) -> list[dict[str, Any]]:
    """Condition template copied narrowly from the unchanged discovery builder.

    Parity tests compare all operations/doses/roles/tokens/positions and random
    audit fields; only explicit inline lineage and confirmation scope are new.
    """
    from .campaign import make_job

    rank = metadata["rank"]
    basis_ref = {"basis_path": metadata["basis_path"], "basis_sha256": metadata["basis_file_sha256"],
                 "basis_metadata_path": metadata["basis_metadata_path"],
                 "basis_metadata_sha256": metadata["basis_metadata_sha256"]}
    training_aliases = sorted(set(metadata["training_layout_ids"]) | set(metadata["training_layout_aliases"]))
    recipient_positions = source["selection"]["recipient_positions"]
    donor_positions = source["selection"]["donor_positions"]
    _require(source["recipient_positions"] == recipient_positions and source["donor_positions"] == donor_positions
             and len(recipient_positions) == len(donor_positions) > 0, "source_position_alias_mismatch")
    jobs = []

    def add(condition: str, operation: str | None, dose: float = 1.0, *, reverse: bool = False,
            identity: bool = False, zero: bool = False, random: bool = False,
            equivalent: bool = False, matched: bool = False, audited: bool = False) -> None:
        value = {name: deepcopy(item) for name, item in source.items()
                 if name not in {"job_id", "experiment_id", "condition", "intervention", "metadata"}}
        rpos, dpos = deepcopy((donor_positions, recipient_positions) if reverse else (recipient_positions, donor_positions))
        if reverse:
            for suffix in ("text", "token_ids"):
                value["recipient_" + suffix], value["donor_" + suffix] = source["donor_" + suffix], source["recipient_" + suffix]
        if identity:
            value["donor_text"], value["donor_token_ids"] = value["recipient_text"], deepcopy(value["recipient_token_ids"])
            dpos = deepcopy(rpos)
        if operation is None:
            value["donor_text"], value["donor_token_ids"] = None, None
        value.update(selection={"scope": "explicit", "recipient_positions": rpos, "donor_positions": dpos},
                     recipient_positions=rpos, donor_positions=dpos, semantic_test=False,
                     semantic_headline_eligible=False, claim_scope=CLAIM_SCOPE)
        intervention = {"layers": [] if operation is None else [0], "site": "resid_post",
                        "operation": operation or "replace", "dose": dose}
        if operation == "subspace":
            intervention.update(basis_ref)
            if zero:
                intervention["zero_donor"] = True
            if random:
                intervention.update(random_basis=True, random_seed=42, rank=rank)
            if matched:
                intervention.update(projected_norm_matched_random_basis=True, random_seed=42, rank=rank)
            if audited or matched:
                intervention.update(audit_projected_norm=True, projected_norm_relative_tolerance=NORM_TOLERANCE)
        meta = deepcopy(source["metadata"])
        # The source is an inline definition, never a fictitious registered experiment.
        meta.pop("source_experiment_id", None)
        meta.pop("source_job_id", None)
        meta.update(inline_input_source_definition_sha256=inline_sha,
                    inline_input_source_job_id=source["job_id"], inline_input_source_job_sha256=fingerprint(source),
                    source_lineage_kind="unregistered_inline_input_site_definition",
                    basis_id=metadata["basis_id"], basis_training_layout_ids=training_aliases,
                    basis_training_disjoint=True, rank=rank, semantic_headline_eligible=False,
                    claim_scope=CLAIM_SCOPE, recipient_patch_positions=rpos, donor_patch_positions=dpos,
                    paired_baseline_condition="donor_text" if reverse else "baseline",
                    random_control_matching="rank_and_per_token_applied_norm_against_task_projected_delta" if matched
                    else "rank_only_not_projected_delta_norm" if random else None,
                    projected_norm_audit={"required": True,
                                          "reference": "same recipient/donor task-basis projected delta, scaled by dose",
                                          "relative_tolerance": NORM_TOLERANCE, "zero_reference_requires_exact_zero": True,
                                          "failures_are_technical_errors_without_outcome_based_reselection": True} if audited or matched else None,
                    random_orientation_policy="one predeclared QR orientation per source seed; no outcome-based seed selection" if random or matched else None,
                    random_subspace_note="Per-token scaling of a random projected direction matches the task-projected norm; this is not an unmodified orthogonal-subspace interchange." if matched else None,
                    independent_rescue_evidence=False, engineering_identity_equivalence=identity,
                    restoration_equivalence_only=equivalent,
                    removal_note="Projection-to-zero is an off-manifold intervention; damage is not semantic necessity." if zero else None,
                    restoration_note="Removal followed by replacement in the same projector is algebraically an identity; this job evaluates that identity-equivalent endpoint, not independent rescue." if equivalent else None)
        jobs.append(make_job(experiment_id, condition=condition, intervention=intervention, metadata=meta, **value))

    add("baseline", None)
    add("donor_text", None, reverse=True)
    add("full_transfer", "replace")
    add("subspace_identity", "subspace", identity=True)
    add("subspace_removal", "subspace", reverse=True, zero=True)
    add("restoration_equivalent", "subspace", identity=True, equivalent=True)
    for dose, label in ((0.5, "0p5"), (1.0, "1")):
        add("subspace_forward_" + label, "subspace", dose, audited=True)
        add("subspace_reverse_" + label, "subspace", dose, reverse=True)
        add("random_subspace_" + label, "subspace", dose, random=True)
        add("projected_norm_matched_random_subspace_" + label, "subspace", dose, matched=True)
    return jobs


def build_frozen_subspace_spec(
    pairs: Sequence[Mapping[str, Any]], tokenizer: Any, experiment_id: str,
    configuration: Mapping[str, Any], freeze_id: str, *, basis_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return canonical confirmation jobs and actual complete-control eligibility.

    ``configuration.primary_comparisons`` maps the two PRIMARY_IDS to exact
    frozen bindings. A partial return is a sampler planning artifact: actual
    counts intentionally differ from those bindings until the quota is filled.
    ``basis_metadata`` must already have been validated by the caller; this
    function binds its supplied bytes structurally but never reads artifacts.
    """
    _require(_name(experiment_id) and _name(freeze_id), "valid experiment_id and explicit freeze_id are required")
    config, basis = _configuration(configuration, basis_metadata)
    sources = deepcopy(list(pairs))
    selected, index_map, exclusions = [], [], []
    training_aliases = set(basis["training_layout_ids"]) | set(basis["training_layout_aliases"])
    for index, pair in enumerate(sources):
        _require(isinstance(pair, Mapping), "source pairs must be structured mappings")
        identity = {key: pair.get(key) for key in ("pair_id", "layout_id", "family")}
        reason = None
        if pair.get("family") != config["family"]:
            reason = "frozen_family_mismatch"
        else:
            try:
                if _aliases(pair) & training_aliases:
                    reason = "basis_training_layout_or_alias_overlap"
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                reason = str(exc)
        if reason:
            exclusions.append({"input_index": index, **identity, "reason": reason, "excluded_before_model_query": True})
        else:
            selected.append(pair)
            index_map.append(index)
    # The definition is embedded and hash-bound, explicitly NOT registered.
    # It uses this container ID, with source job IDs scoped only to the inline map.
    inline = build_input_site_spec(selected, tokenizer, experiment_id, layers=(0,), site="resid_post",
                                   natural=True, seeds=(42,), cap=4096, partition="confirmation", freeze_id=freeze_id)
    _require(dict(config["tokenizer"]) == inline["tokenizer"], "tokenizer differs from the frozen configuration")
    inline_sha = fingerprint(inline)
    for row in inline["exclusions"]:
        excluded = deepcopy(row)
        excluded["input_index"] = index_map[row["input_index"]]
        exclusions.append(excluded)
    exclusions.sort(key=lambda row: row["input_index"])
    intended = [job for job in inline["jobs"] if job["condition"] == "factspan_layer0"]
    accepted = deepcopy(inline["accepted_pairs"])
    _require(len(intended) == len(accepted), "inline intended-source inventory differs from accepted pairs")
    # The canonical builder deduplicates its topology/variant aliases. Also
    # retain and deduplicate any explicit aliases of the complete raw source.
    retained, complete_aliases_seen = [], set()
    for source in intended:
        aliases = set(source["layout_aliases"]) | _aliases(source["metadata"]["source_pair"]) | _aliases(source["metadata"]["pair"])
        if aliases & complete_aliases_seen:
            original_index = next(index_map[index] for index, pair in enumerate(selected)
                                  if pair == source["metadata"]["source_pair"])
            exclusions.append({"input_index": original_index,
                               **{key: source[key] for key in ("pair_id", "layout_id", "family")},
                               "reason": "duplicate_complete_source_layout_alias", "excluded_before_model_query": True})
        else:
            complete_aliases_seen.update(aliases)
            retained.append(source)
    intended = retained
    retained_ids = {source["pair_id"] for source in intended}
    accepted = [row for row in accepted if row["pair_id"] in retained_ids]
    exclusions.sort(key=lambda row: row["input_index"])
    pair_ids = sorted(row["pair_id"] for row in accepted)
    n_layouts = len({row["layout_id"] for row in accepted})
    _require(n_layouts == len(pair_ids) == len(set(pair_ids)), "independent layout/pair count differs")
    _require(n_layouts <= config["n_layouts"], "eligible cohort exceeds the frozen layout quota")
    jobs, layout_aliases = [], {}
    for source in intended:
        _require(source["job_id"] == fingerprint({key: value for key, value in source.items() if key != "job_id"})[:24],
                 "noncanonical inline source job identity")
        _require(source["family"] == config["family"] and source["partition"] == "confirmation"
                 and source["seed"] == 42 and source["natural"] is True, "inline source protocol mismatch")
        aliases = set(source["layout_aliases"]) | _aliases(source["metadata"]["source_pair"]) | _aliases(source["metadata"]["pair"])
        _require(not aliases & training_aliases, "basis training alias entered normalized source")
        source = deepcopy(source)
        # Preserve any additional raw-source aliases in every emitted job. The
        # exact inline source job itself remains untouched in the embedded map.
        emitted = _expand(source, basis, experiment_id, inline_sha)
        if aliases != set(source["layout_aliases"]):
            from .campaign import make_job
            revised = []
            for job in emitted:
                fields = {key: value for key, value in job.items() if key not in {"experiment_id", "job_id"}}
                fields["layout_aliases"] = sorted(aliases)
                revised.append(make_job(experiment_id, **fields))
            emitted = revised
        jobs.extend(emitted)
        layout_aliases[source["layout_id"]] = sorted(aliases)
    _require(len(jobs) == 14 * n_layouts and Counter(job["condition"] for job in jobs) ==
             Counter({condition: n_layouts for condition in CONDITIONS}), "incomplete fourteen-condition cohort")
    for job in jobs:
        _require(job["job_id"] == fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24],
                 "noncanonical expanded job identity")
    comparisons = []

    def compare(treatment: str, control: str, direction: str, role: str) -> None:
        comparison = {"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
                      "partition": "confirmation", "candidate_id": config["candidate_id"],
                      "treatment": treatment, "control": control, "predicted_direction": direction, "role": role,
                      "confirmatory": False, "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
                      "target_action_basis": "unique_simulator_counterfactual_optimum", "claim_scope": CLAIM_SCOPE,
                      "semantic_headline_eligible": False, "independent_rescue_evidence": False,
                      "outcome_dependent_exclusion_permitted": False, "expected_n_layouts": n_layouts,
                      "expected_n_pairs": n_layouts, "eligible_pair_ids": pair_ids.copy(),
                      "eligibility_basis": "frozen_simulator_tokenizer_and_basis_aliases_before_model_query",
                      "required_condition_inventory": list(CONDITIONS)}
        for primary_id, binding in config["primary_comparisons"].items():
            if binding["treatment"] == treatment and binding["control"] == control:
                comparison.update(deepcopy(binding))
                comparison.update(primary_id=primary_id, confirmatory=True,
                                  expected_n_layouts=n_layouts, expected_n_pairs=n_layouts)
        comparisons.append(comparison)

    compare("full_transfer", "baseline", "increase", "full_state_reference")
    compare("subspace_identity", "baseline", "zero", "engineering_identity")
    compare("restoration_equivalent", "baseline", "zero", "algebraic_identity_not_rescue")
    compare("subspace_removal", "donor_text", "decrease", "counterfactual_projection_removal_diagnostic")
    for label in ("0p5", "1"):
        compare("subspace_forward_" + label, "baseline", "increase", "held_out_subspace_sufficiency")
        compare("subspace_forward_" + label, "random_subspace_" + label, "increase", "rank_only_random_comparison_not_norm_specificity")
        compare("subspace_forward_" + label, "projected_norm_matched_random_subspace_" + label, "increase", "rank_and_projected_norm_matched_random_comparison")
        compare("subspace_reverse_" + label, "donor_text", "decrease", "conditional_reverse_interchange")
    compare("donor_text", "baseline", "increase", "physical_counterfactual_reference")
    primary = [row for row in comparisons if row["confirmatory"]]
    _require(len(primary) == 2 and {row["primary_id"] for row in primary} == set(PRIMARY_IDS), "exact co-primary binding failed")
    excluded_by_index = {row["input_index"]: row for row in exclusions}
    manifest = [{"input_index": index, **{key: pair.get(key) for key in ("pair_id", "layout_id", "family", "partition")},
                 "eligible": index not in excluded_by_index,
                 "scheduled_conditions": list(CONDITIONS) if index not in excluded_by_index else [],
                 "excluded_before_model_query": index in excluded_by_index, "selection_before_model_query": True,
                 "selection_basis": "simulator_tokenizer_and_training_aliases_only",
                 "reason": excluded_by_index[index]["reason"] if index in excluded_by_index else None}
                for index, pair in enumerate(sources)]
    return {"schema_version": SCHEMA_VERSION, "adapter_version": 1, "protocol_kind": PROTOCOL,
            "experiment_id": experiment_id, "partition": "confirmation", "freeze_id": freeze_id,
            "candidate_id": config["candidate_id"], "family": config["family"],
            "source_experiment_id": config["source_experiment_id"], "source_spec_sha256": config["source_spec_sha256"],
            "jobs": jobs, "comparisons": comparisons, "frozen_configuration": config,
            "configuration_sha256": fingerprint(config), "basis": basis, "basis_file_sha256": config["basis_sha256"],
            "basis_metadata_fingerprint": config["basis_metadata_fingerprint"],
            "inline_input_source": {"format": "unregistered_inline_input_site_definition_v1", "registered": False,
                                    "definition": inline, "definition_sha256": inline_sha,
                                    "job_id_namespace": "inline definition only; not a registered source experiment"},
            "accepted_pairs": accepted, "exclusions": exclusions,
            "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
            "n_source_pairs": len(sources), "n_excluded_pairs": len(exclusions), "n_aligned_pairs": len(accepted),
            "source_pairs_sha256": fingerprint(sources), "tokenizer": deepcopy(inline["tokenizer"]),
            "expected_n_layouts": n_layouts, "expected_n_pairs": n_layouts,
            "requested_n_layouts": config["n_layouts"], "layout_shortfall": config["n_layouts"] - n_layouts,
            "cohort_complete": n_layouts == config["n_layouts"], "eligible_pair_ids": pair_ids,
            "eligibility_manifest": manifest, "layout_aliases": dict(sorted(layout_aliases.items())),
            "primary_comparison_ids": {row["primary_id"]: row["id"] for row in primary},
            "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
            "primary_estimand": "layout-weighted paired native target-action ITT probability difference",
            "target_action_basis": "unique_simulator_counterfactual_optimum", "claim_scope": CLAIM_SCOPE,
            "semantic_headline_eligible": False, "independent_rescue_evidence": False,
            "protocol": {"layer": 0, "layers": [0], "site": "resid_post", "rank": config["rank"],
                         "doses": [0.5, 1.0], "selected_dose": config["selected_dose"], "seeds": [42],
                         "random_basis_seeds": [42], "natural": True, "temperature": 0.7, "max_new_tokens": 4096,
                         "renderer_id": "canonical_v1", "input_alignment_rule": ALIGNMENT,
                         "condition_inventory": list(CONDITIONS), "selected_support_width_may_exceed_one": True,
                         "projected_norm_relative_tolerance": NORM_TOLERANCE, "zero_reference_requires_exact_zero": True,
                         "requires_complete_fourteen_condition_cohort": True,
                         "confirmation_requires_external_freeze": True, "confirmation_requires_external_ready_check": True,
                         "basis_files_read_or_refitted": False, "caller_basis_validation_is_not_certified_here": True,
                         "requires_basis_artifact_and_loaded_model_verification": True,
                         "primary_family_requires_external_exact_binding": True, "lexical_supplement_glue_pending": True,
                         "outcome_dependent_exclusion_permitted": False, "semantic_headline_eligible": False},
            "hypothesis": "One fixed discovery-trained input-fact basis may increase counterfactual-target native actions over baseline and projected-norm-matched random on fresh canonical layouts; this does not establish abstract belief, necessity or mediation."}
