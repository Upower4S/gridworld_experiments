"""Staged fixed contextual confirmation, with complete fifteen-control cohorts.

The public primary adapter accepts the canonical renderer only. The private
structured-state materializer can prepare either reviewed fixed renderer; a
future supplement wrapper must bind parent/subset/role and make every alternate
comparison descriptive. No arbitrary prompt or confirmatory override is exposed.
No models, filesystem operations, registration, or readiness certification occur.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .environments import canonical_layout_fingerprint, optimal_actions, prompt_text, validate_state
from .experiments import build_controlled_state_spec
from .input_sites import _physical_counterfactual, aligned_changed_span
from .lexical_generalization import (
    ALTERNATE, CANONICAL, audit_pair_renderings, render_with_proof, renderer_manifest,
)
from .registry import SCHEMA_VERSION, fingerprint


PROTOCOL = "contextual_input"
CLAIM_SCOPE = "contextual_input_fact_interchange"
ENDPOINT = "natural_target_action_intention_to_treat"
UNRESOLVED_POLICY = "count as zero target success; retain in intention-to-treat denominator; no forced fallback"
ALIGNMENT = "exact_equal_length_common_prefix_and_suffix"
SOURCE_RULE = "within_context_named_variable_toggle"
FAMILIES = ("wall", "key_possession", "door_status")
PRIMARY_IDS = ("contextual_fact_contrast", "natural_norm_direction_contrast")
QUOTAS = {32: {"wall": 11, "key_possession": 11, "door_status": 10},
          64: {"wall": 22, "key_possession": 21, "door_status": 21},
          128: {"wall": 43, "key_possession": 43, "door_status": 42}}
SOURCES = ("recipient", "counterfactual", "donor", "natural_donor", "same_fact_donor", "irrelevant_fact_donor")
REFERENCES = ("baseline", "counterfactual_text", "donor_text", "natural_donor_text", "same_fact_donor_text", "irrelevant_donor_text")
INTERVENTIONS = ("matched", "natural", "same_fact", "irrelevant", "random", "reverse", "identity", "restoration_equivalent", "natural_norm_random")


def condition_inventory(layer: int) -> list[str]:
    return list(REFERENCES) + [f"layer{layer}_{name}" for name in INTERVENTIONS]


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    return {"class": type(tokenizer).__name__, "name_or_path": str(getattr(tokenizer, "name_or_path", "unspecified")),
            "add_special_tokens": False, "context_renderer": "campaign.render_analysis_prompt"}


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    values = list(values.ids if hasattr(values, "ids") else values)
    _require(bool(values) and all(type(value) is int and value >= 0 for value in values), "invalid_tokenizer_output")
    return values


def _configuration(configuration: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    from .primary_family import MIXTURE_FIELDS, REQUIRED_BINDING_FIELDS, validate_primary_family

    _require(isinstance(configuration, Mapping), "configuration must be a mapping")
    config = deepcopy(dict(configuration))
    required = {"candidate_id", "source_experiment_id", "source_spec_sha256", "layer", "site", "families",
                "family_layout_counts", "family_weights", "n_layouts", "generation_seeds", "random_seed",
                "temperature", "max_new_tokens", "condition_inventory", "tokenizer", "renderer_id", "renderer_binding",
                "require_joint_renderer_eligibility", "primary_comparisons"}
    _require(required <= config.keys(), "missing frozen contextual fields: " + ",".join(sorted(required - config.keys())))
    _require(_name(config["candidate_id"]) and _name(config["source_experiment_id"]) and _sha(config["source_spec_sha256"]),
             "invalid frozen source/candidate identity")
    _require(type(config["layer"]) is int and config["layer"] in {0, 10} and config["site"] == "resid_post", "freeze tested layer0 or10 resid_post")
    _require(config["families"] == list(FAMILIES) and "family" not in config, "require exact canonical three-family population")
    count = config["n_layouts"]
    _require(type(count) is int and count in QUOTAS, "freeze32,64,128 independent layouts")
    quotas = QUOTAS[count]
    _require(fingerprint(config["family_layout_counts"]) == fingerprint(quotas),
             "family quotas differ from adopted policy")
    weights = {family: quotas[family] / count for family in FAMILIES}
    _require(fingerprint(config["family_weights"]) == fingerprint(weights),
             "family weights differ from frozen quotas")
    _require(config["generation_seeds"] == [42] and type(config["generation_seeds"][0]) is int
             and type(config["random_seed"]) is int and config["random_seed"] == 42, "freeze generation/random seed42")
    _require(type(config["max_new_tokens"]) is int and config["max_new_tokens"] == 4096
             and type(config["temperature"]) in {int, float} and config["temperature"] == 0.7, "freeze cap4096 temperature0.7")
    _require(config["condition_inventory"] == condition_inventory(config["layer"]), "retain exact ordered fifteen-condition inventory")
    _require(config["renderer_id"] == CANONICAL, "public primary adapter is canonical-only; alternate needs a bound supplement wrapper")
    _require(fingerprint(config["renderer_binding"]) == fingerprint(renderer_manifest(CANONICAL)), "fixed renderer binding mismatch")
    _require(type(config["require_joint_renderer_eligibility"]) is bool, "joint renderer eligibility must be explicitly Boolean")
    _require(fingerprint(config["tokenizer"]) == fingerprint(_tokenizer_metadata(tokenizer)), "tokenizer differs from frozen metadata")
    _require(not ({"prompt_override", "renderer_override", "confirmatory", "family_size", "confirmation_family_size",
                   "cohort_role", "primary_id"} & config.keys()), "unsupported prompt, inferential-role or multiplicity override")
    fixed = {"protocol_kind": PROTOCOL, "layers": [config["layer"]], "natural": True, "operation": "replace", "dose": 1.0,
             "input_alignment_rule": ALIGNMENT, "source_span_localization": SOURCE_RULE,
             "selection": {"scope": "explicit", "alignment": ALIGNMENT, "source_span_localization": SOURCE_RULE},
             "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False, "primary_endpoint": ENDPOINT,
             "native_unresolved_policy": UNRESOLVED_POLICY, "conditions_per_layout": 15,
             "resampling_scheme": "within_family_layout_bootstrap", "stratification_key": "family",
             "full_write_relative_tolerance": 0.01}
    for key, expected in fixed.items():
        _require(key not in config or fingerprint(config[key]) == fingerprint(expected),
                 "contradictory frozen " + key)
    for key, expected in {"cap": 4096, "seeds": [42], "layout_count": count}.items():
        _require(key not in config or fingerprint(config[key]) == fingerprint(expected), "contradictory frozen alias " + key)
    _require("intervention" not in config, "freeze source roles and the complete audited profile, not an arbitrary intervention mapping")
    primaries = config["primary_comparisons"]
    _require(isinstance(primaries, Mapping) and set(primaries) == set(PRIMARY_IDS), "freeze exactly two named contextual co-primaries")
    prefix = f"layer{config['layer']}"
    for primary_id, control, origin in ((PRIMARY_IDS[0], prefix + "_same_fact", "existing_source_comparison"),
                                       (PRIMARY_IDS[1], prefix + "_natural_norm_random", "prospective_confirmation_control")):
        binding = primaries[primary_id]
        _require(isinstance(binding, Mapping) and set(binding) == REQUIRED_BINDING_FIELDS | MIXTURE_FIELDS,
                 "primary fields differ from complete mixed-family binding")
        expected = {"treatment": prefix + "_natural", "control": control, "predicted_direction": "increase",
                    "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
                    "target_action_basis": "unique_simulator_counterfactual_optimum", "claim_scope": CLAIM_SCOPE,
                    "semantic_headline_eligible": False, "expected_n_layouts": count, "expected_n_pairs": count,
                    "families": list(FAMILIES), "family_layout_counts": quotas, "family_weights": weights,
                    "expected_n_layouts_by_family": quotas, "expected_n_pairs_by_family": quotas,
                    "resampling_scheme": "within_family_layout_bootstrap", "stratification_key": "family"}
        _require(fingerprint({key: binding.get(key) for key in expected}) == fingerprint(expected), "contextual primary binding contradiction")
        _require(isinstance(binding["origin"], Mapping) and binding["origin"].get("kind") == origin,
                 "existing and prospective contextual origins must remain distinct")
        provenance = binding["origin"].get("provenance")
        _require(isinstance(provenance, Mapping), "missing primary origin provenance")
        for key in ("source_experiment_id", "source_spec_sha256"):
            _require(key not in provenance or provenance[key] == config[key], "primary origin contradicts " + key)
    validate_primary_family({"schema_version": 1, "candidates": {config["candidate_id"]: list(PRIMARY_IDS)},
                             "family_size": 2, "bindings": {config["candidate_id"]: dict(primaries)}})
    config.update(fixed)
    return config


def _aliases(pair: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for value in (pair, pair.get("metadata", {})):
        if not isinstance(value, Mapping):
            continue
        if isinstance(value.get("layout_id"), str) and value["layout_id"]:
            aliases.add(value["layout_id"])
        for key in ("layout_aliases", "variant_layout_ids"):
            items = value.get(key, [])
            _require(isinstance(items, (list, tuple)) and all(isinstance(item, str) and item for item in items),
                     "invalid_layout_aliases")
            aliases.update(items)
        for key in ("pair", "source_pair"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                aliases.update(_aliases(nested))
    return aliases


def _materialize_pair(
    pair: Mapping[str, Any], tokenizer: Any, experiment_id: str, *, layer: int,
    renderer_id: str, require_joint_renderer_eligibility: bool,
    candidate_source_experiment_id: str, candidate_source_spec_sha256: str,
    freeze_id: str | None = None,
) -> dict[str, Any]:
    """Private fixed-renderer primitive for a future parent-bound supplement.

    Input is a complete canonical physical pair, never edited prompt strings.
    ``renderer_id`` accepts only the two reviewed constants. Confirmation pairs
    require a nonempty freeze ID; discovery pairs remain discovery (for parity
    checks). Output has ``eligible``, ``jobs`` (all fifteen or none), source-span
    proofs, complete aliases and an explicitly unregistered inline definition.
    No comparison or inferential role is assigned here. An alternate wrapper
    must independently bind parent/subset/renderer and mark ALL its comparisons
    descriptive. Candidate source IDs denote protocol lineage, not membership
    of this fresh pair in that discovery experiment.
    """
    from .campaign import make_job, render_analysis_prompt
    from .contextual_input import _toggle_named_fact

    _require(isinstance(pair, Mapping), "physical_pair_must_be_mapping")
    pair = deepcopy(dict(pair))
    _require(_name(experiment_id) and type(layer) is int and layer in {0, 10}, "invalid_materializer_geometry")
    _require(renderer_id in {CANONICAL, ALTERNATE} and type(require_joint_renderer_eligibility) is bool,
             "fixed_renderer_and_explicit_joint_eligibility_required")
    _require(_name(candidate_source_experiment_id) and _sha(candidate_source_spec_sha256), "invalid_candidate_protocol_lineage")
    partition = pair.get("partition")
    _require(partition in {"discovery", "confirmation"}, "invalid_physical_partition")
    _require((partition == "confirmation" and _name(freeze_id)) or (partition == "discovery" and freeze_id is None),
             "confirmation_requires_freeze_id_and_discovery_cannot_be_relabeled")
    _, recipient_user, counterfactual_user, _, _ = _physical_counterfactual(pair, partition=partition)
    validate_state(pair["donor"])
    donor_user = prompt_text(pair["donor"])
    donor_actions = optimal_actions(pair["donor"])
    _require(bool(donor_actions), "unreachable_actual_pair_donor")
    _require(pair.get("donor_prompt_text", donor_user) == donor_user, "donor_prompt_does_not_match_physical_state")
    _require(pair.get("donor_optimal_actions", donor_actions) == donor_actions, "recorded_donor_optimal_actions_disagree_with_simulator")
    _require(pair["pair_id"] == fingerprint({key: pair[key] for key in ("layout_id", "recipient", "donor", "family")}),
             "noncanonical_physical_pair_id")
    canonical_pair = deepcopy(pair)
    canonical_pair.update(recipient_prompt_text=recipient_user, counterfactual_prompt_text=counterfactual_user,
                          donor_prompt_text=donor_user, donor_optimal_actions=donor_actions,
                          recipient_optimal_actions=optimal_actions(pair["recipient"]),
                          counterfactual_optimal_actions=optimal_actions(pair["counterfactual"]))
    inline = build_controlled_state_spec([canonical_pair], experiment_id, layers=(layer,), site="resid_post",
                                         seeds=(42,), max_new_tokens=4096, natural=True)
    inline_sha = fingerprint(inline)
    manifest = deepcopy(inline["eligibility_manifest"][0])
    result = {"eligible": False, "jobs": [], "reason": None, "manifest": manifest,
              "renderer_id": renderer_id, "renderer_binding": renderer_manifest(renderer_id),
              "joint_renderer_audit": None, "source_states": {}, "source_span_audits": {},
              "inline_controlled_source": {"format": "unregistered_inline_controlled_state_definition_v1",
                    "registered": False, "definition": inline, "definition_sha256": inline_sha,
                    "job_id_namespace": "inline definition only; not a registered source experiment"}}
    controls = {row["condition"]: row for row in manifest["controls"]}
    unavailable = [name for name in SOURCES[3:] if name not in controls or controls[name].get("eligible") is not True]
    if unavailable:
        result["reason"] = "complete_control_source_unavailable:" + ",".join(unavailable)
        return result
    states = {name: canonical_pair[name] for name in SOURCES[:3]}
    states.update({name: controls[name]["state"] for name in SOURCES[3:]})
    result["source_states"] = deepcopy(states)
    aliases = _aliases(pair) | set(inline["layout_aliases"][pair["layout_id"]])
    texts, ids, audits, exclusions = {}, {}, {}, []
    for source, state in states.items():
        audit = {"source": source, "eligible": False, "excluded_before_model_query": True,
                 "alignment_basis": "toggle_only_named_variable_in_same_source_context"}
        try:
            validate_state(state)
            _require(bool(optimal_actions(state)), "unreachable_control_source")
            toggled = _toggle_named_fact(state, canonical_pair)
            rendered, toggle_rendered = render_with_proof(state, renderer_id), render_with_proof(toggled, renderer_id)
            text = render_analysis_prompt(rendered["prompt_text"])
            tokens = _encode(tokenizer, text)
            toggle_tokens = _encode(tokenizer, render_analysis_prompt(toggle_rendered["prompt_text"]))
            span = aligned_changed_span(tokens, toggle_tokens)
            if source in controls:
                _require(controls[source]["prompt_text"] == prompt_text(state), "control_prompt_does_not_match_physical_state")
                _require(controls[source]["optimal_actions"] == optimal_actions(state), "recorded_control_optimum_mismatch")
            aliases.update((canonical_layout_fingerprint(state), canonical_layout_fingerprint(toggled)))
            texts[source], ids[source] = text, tokens
            audit.update(eligible=True, reason=None, excluded_before_model_query=False, span=span,
                         source_input_sha256=fingerprint(tokens), toggle_input_sha256=fingerprint(toggle_tokens),
                         source_state_sha256=fingerprint(state), toggled_state_sha256=fingerprint(toggled),
                         context_agent=deepcopy(state["agent"]), toggled_context_agent=deepcopy(toggled["agent"]),
                         agent_location_unchanged_in_span_localization=state["agent"] == toggled["agent"],
                         renderer_proof=rendered["proof"], toggled_renderer_proof=toggle_rendered["proof"])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            audit["reason"] = str(exc)
            exclusions.append({"source": source, "reason": str(exc)})
        audits[source] = audit
    result.update(source_span_audits=audits, layout_aliases=sorted(aliases))
    if require_joint_renderer_eligibility:
        joint = audit_pair_renderings(canonical_pair, tokenizer, profile="contextual")
        result["joint_renderer_audit"] = joint
        if not joint["eligible"]:
            exclusions.extend(deepcopy(joint["exclusions"]))
    for job in inline["jobs"]:
        r, d = (job["metadata"][role + "_source"] for role in ("recipient", "donor"))
        if d is not None and audits[r]["eligible"] and audits[d]["eligible"]:
            for field, reason in (("n_input_tokens", "cross_context_unequal_full_token_lengths"),
                                  ("n_changed_span_tokens", "cross_context_unequal_named_fact_span_widths")):
                if audits[r]["span"][field] != audits[d]["span"][field]:
                    exclusions.append({"condition": job["condition"], "reason": reason})
    if exclusions:
        result.update(reason="complete_control_alignment_failed", alignment_exclusions=exclusions)
        return result
    _require([job["condition"] for job in inline["jobs"]] == condition_inventory(layer)[:-1], "incomplete_or_changed_legacy_inventory")
    templates = deepcopy(inline["jobs"])
    extra = deepcopy(next(job for job in templates if job["condition"] == f"layer{layer}_natural"))
    extra["condition"] = f"layer{layer}_natural_norm_random"
    extra["intervention"].update(random_control=True, random_seed=42, norm_reference="donor_minus_recipient")
    extra["metadata"].update(control_role="natural_norm_matched_random", norm_reference_source="natural_donor",
                             existing_random_orientation_reused=True)
    templates.append(extra)
    jobs = []
    for template in templates:
        r, d = (template["metadata"][role + "_source"] for role in ("recipient", "donor"))
        recipient_positions = audits[r]["span"]["recipient_positions"]
        donor_positions = audits[d]["span"]["recipient_positions"] if d is not None else None
        job = deepcopy(template)
        job.update(recipient_text=texts[r], donor_text=texts[d] if d is not None else None,
                   recipient_token_ids=ids[r], donor_token_ids=ids[d] if d is not None else None,
                   recipient_positions=recipient_positions, donor_positions=donor_positions,
                   selection={"scope": "explicit", "recipient_positions": recipient_positions, "donor_positions": donor_positions},
                   semantic_test=False, semantic_headline_eligible=False, claim_scope=CLAIM_SCOPE,
                   layout_aliases=sorted(aliases))
        job["metadata"].update(
            source_pair=deepcopy(pair), intervention_time="named_fact_input_token", claim_scope=CLAIM_SCOPE,
            representation_kind="observed_fact_input_tokens", abstract_belief_representation_claim=False,
            semantic_headline_eligible=False, input_token_ids_authoritative=True, no_generated_cot_tokens=True,
            site_note="Named-variable changed span localized separately within each fixed source context",
            eligibility_basis="simulator_and_exact_tokenizer_alignment_before_model_query",
            recipient_input_fact_span=deepcopy(audits[r]["span"]),
            donor_input_fact_span=deepcopy(audits[d]["span"]) if d is not None else None,
            recipient_patch_positions=recipient_positions, donor_patch_positions=donor_positions,
            donor_span_not_located_by_recipient_donor_context_difference=True,
            contextual_information_may_be_encoded_in_fact_token_states=True,
            renderer_id=renderer_id, renderer_binding=renderer_manifest(renderer_id),
            recipient_renderer_proof=audits[r]["renderer_proof"],
            donor_renderer_proof=audits[d]["renderer_proof"] if d is not None else None,
            inline_controlled_source_definition_sha256=inline_sha,
            inline_controlled_source_job_id=template["job_id"],
            inline_controlled_source_job_sha256=fingerprint(next(j for j in inline["jobs"] if j["job_id"] == template["job_id"])),
            inline_source_is_unregistered=True, source_pair_sha256=fingerprint(pair),
            candidate_source_experiment_id=candidate_source_experiment_id,
            candidate_source_spec_sha256=candidate_source_spec_sha256,
            same_fact_contrast_is_not_norm_matched=True, existing_random_norm_reference_is_matched_counterfactual=True,
        )
        if freeze_id is not None:
            job["metadata"]["freeze_id"] = freeze_id
        if job["condition"] in {f"layer{layer}_natural", f"layer{layer}_random", f"layer{layer}_natural_norm_random"}:
            is_matched = job["condition"] == f"layer{layer}_random"
            job["intervention"]["full_write_source_identity"] = {
                "lineage_kind": "inline_physical_source_pair", "source_pair_id": pair["pair_id"],
                "source_pair_sha256": fingerprint(pair), "recipient_role": "recipient",
                "donor_role": "matched" if is_matched else "natural", "span_rule": SOURCE_RULE,
                "renderer_id": renderer_id, "recipient_token_sha256": fingerprint(ids[r]),
                "donor_token_sha256": fingerprint(ids[d]), "selection_sha256": fingerprint(job["selection"]),
                "candidate_source_experiment_id": candidate_source_experiment_id,
                "candidate_source_spec_sha256": candidate_source_spec_sha256,
            }
            if not is_matched:
                job["intervention"].update(audit_full_write_norm=True, full_write_relative_tolerance=0.01)
            if job["intervention"].get("random_control"):
                job["intervention"]["record_random_orientation"] = True
        fields = {key: value for key, value in job.items() if key not in {"experiment_id", "job_id"}}
        jobs.append(make_job(experiment_id, **fields))
    result.update(eligible=True, reason=None, jobs=jobs)
    manifest.update(scheduled_conditions=condition_inventory(layer), unscheduled_conditions=[],
                    source_span_audits=audits, selection_basis="simulator_and_tokenizer_only_before_model_query",
                    claim_scope=CLAIM_SCOPE, semantic_headline_eligible=False)
    return result


def build_frozen_contextual_input_spec(
    pairs: Sequence[Mapping[str, Any]], tokenizer: Any, experiment_id: str,
    configuration: Mapping[str, Any], freeze_id: str,
) -> dict[str, Any]:
    """Build canonical primary planning data; external readiness is still required.

    Partial refill batches are returned with truthful actual per-family counts
    and frozen requested quotas. Every accepted layout has all fifteen controls.
    Configuration keeps its complete requested-N co-primary bindings unchanged.
    A jointly eligible alternate rendering is checked before ANY primary query
    only when the explicitly frozen joint-eligibility flag is true.
    """
    config = _configuration(configuration, tokenizer)
    _require(_name(experiment_id) and _name(freeze_id), "require canonical experiment and freeze IDs")
    sources = deepcopy(list(pairs))
    _require(all(isinstance(pair, Mapping) for pair in sources), "physical pair inputs must be mappings")
    jobs, accepted, exclusions, manifest, definitions = [], [], [], [], []
    layout_aliases, seen_aliases, seen_pair_ids = {}, set(), set()
    layer = config["layer"]
    for index, pair in enumerate(sources):
        entry = {"input_index": index, **{key: pair.get(key) for key in ("pair_id", "layout_id", "family", "partition")},
                 "eligible": False, "scheduled_conditions": [], "excluded_before_model_query": True,
                 "selection_before_model_query": True, "selection_basis": "canonical_simulator_and_exact_tokenizer_only"}
        materialized = None
        try:
            _require(pair.get("partition") == "confirmation", "non_confirmation_pair")
            _require(pair.get("family") in FAMILIES, "family_outside_frozen_population")
            _require(pair.get("pair_id") not in seen_pair_ids, "duplicate_pair_id")
            materialized = _materialize_pair(pair, tokenizer, experiment_id, layer=layer, renderer_id=CANONICAL,
                require_joint_renderer_eligibility=config["require_joint_renderer_eligibility"],
                candidate_source_experiment_id=config["source_experiment_id"],
                candidate_source_spec_sha256=config["source_spec_sha256"], freeze_id=freeze_id)
            _require(materialized["eligible"], materialized["reason"] or "complete_control_ineligible")
            aliases = set(materialized["layout_aliases"])
            _require(not aliases & seen_aliases, "duplicate_complete_source_layout_alias")
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            entry["reason"] = str(exc)
            if materialized is not None:
                entry["control_eligibility"] = {key: deepcopy(value) for key, value in materialized.items()
                                                if key not in {"jobs", "inline_controlled_source"}}
            exclusions.append(deepcopy(entry))
        else:
            seen_aliases.update(aliases)
            seen_pair_ids.add(pair["pair_id"])
            jobs.extend(materialized["jobs"])
            layout_aliases[pair["layout_id"]] = sorted(aliases)
            definitions.append({"pair_id": pair["pair_id"], **materialized["inline_controlled_source"]})
            entry.update(eligible=True, reason=None, excluded_before_model_query=False,
                         scheduled_conditions=condition_inventory(layer),
                         control_eligibility={key: deepcopy(value) for key, value in materialized.items()
                                               if key not in {"jobs", "inline_controlled_source"}})
            accepted.append({key: pair[key] for key in ("pair_id", "layout_id", "family", "partition")})
        manifest.append(entry)
    actual = {family: sum(row["family"] == family for row in accepted) for family in FAMILIES}
    quotas = config["family_layout_counts"]
    _require(all(actual[family] <= quotas[family] for family in FAMILIES), "eligible family cohort exceeds frozen quota")
    pair_ids = sorted(row["pair_id"] for row in accepted)
    count = len(accepted)
    _require(count == len(set(row["layout_id"] for row in accepted)), "accepted layouts are not independent")
    _require(len(jobs) == 15 * count and Counter(job["condition"] for job in jobs) ==
             Counter({name: count for name in condition_inventory(layer)}), "incomplete fifteen-condition output")
    comparisons = []
    prefix = f"layer{layer}"
    semantic_controls = {"counterfactual_baseline": "counterfactual_text", "donor_baseline": "natural_donor_text",
                         "same_fact": prefix + "_same_fact", "irrelevant_fact": prefix + "_irrelevant",
                         "donor_mismatch_treatment": prefix + "_natural"}

    def compare(treatment: str, control: str, direction: str, role: str) -> None:
        row = {"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
               "partition": "confirmation", "candidate_id": config["candidate_id"], "treatment": treatment,
               "control": control, "predicted_direction": direction, "role": role, "confirmatory": False,
               "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
               "target_action_basis": "unique_simulator_counterfactual_optimum", "claim_scope": CLAIM_SCOPE,
               "semantic_headline_eligible": False, "independent_rescue_evidence": False,
               "outcome_dependent_exclusion_permitted": False, "expected_n_layouts": count, "expected_n_pairs": count,
               "expected_n_layouts_by_family": actual.copy(), "expected_n_pairs_by_family": actual.copy(),
               "families": list(FAMILIES), "family_layout_counts": deepcopy(quotas), "family_weights": deepcopy(config["family_weights"]),
               "resampling_scheme": "within_family_layout_bootstrap", "stratification_key": "family",
               "eligible_pair_ids": pair_ids.copy(), "required_condition_inventory": condition_inventory(layer),
               "semantic_controls": deepcopy(semantic_controls), "eligibility_requires_all_semantic_controls": True,
               "eligibility_basis": "complete_fifteen_controls_simulator_and_per_source_fact_spans_before_model_query"}
        for primary_id, binding in config["primary_comparisons"].items():
            if (binding["treatment"], binding["control"]) == (treatment, control):
                row.update(deepcopy(binding))
                row.update(primary_id=primary_id, confirmatory=True, expected_n_layouts=count, expected_n_pairs=count,
                           expected_n_layouts_by_family=actual.copy(), expected_n_pairs_by_family=actual.copy())
        comparisons.append(row)

    for treatment, control, direction, role in (
        ("counterfactual_text", "baseline", "increase", "physical_world_sensitivity"),
        (prefix + "_matched", "baseline", "increase", "conditional_sufficiency"),
        (prefix + "_matched", prefix + "_random", "increase", "matched_contrast_norm_random_reference"),
        (prefix + "_natural", "baseline", "increase", "context_generalization"),
        (prefix + "_natural", prefix + "_same_fact", "increase", "contextual_fact_contrast_not_norm_matched"),
        (prefix + "_matched", prefix + "_irrelevant", "increase", "irrelevant_fact_specificity"),
        (prefix + "_reverse", "counterfactual_text", "decrease", "conditional_reverse_effect"),
        (prefix + "_identity", "baseline", "zero", "engineering_identity"),
        (prefix + "_restoration_equivalent", "counterfactual_text", "zero", "engineering_restoration_equivalence"),
        (prefix + "_natural", prefix + "_natural_norm_random", "increase", "prospective_natural_norm_direction_contrast"),
    ):
        compare(treatment, control, direction, role)
    primary = [row for row in comparisons if row["confirmatory"]]
    _require(len(primary) == 2 and {row["primary_id"] for row in primary} == set(PRIMARY_IDS), "exact co-primary binding failed")
    return {"schema_version": SCHEMA_VERSION, "adapter_version": 1, "protocol_kind": PROTOCOL,
            "experiment_id": experiment_id, "partition": "confirmation", "freeze_id": freeze_id,
            "candidate_id": config["candidate_id"], "source_experiment_id": config["source_experiment_id"],
            "source_spec_sha256": config["source_spec_sha256"], "jobs": jobs, "comparisons": comparisons,
            "frozen_configuration": config, "configuration_sha256": fingerprint(config),
            "accepted_pairs": accepted, "exclusions": exclusions, "eligibility_manifest": manifest,
            "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
            "inline_controlled_sources": definitions, "source_pairs_sha256": fingerprint(sources),
            "n_source_pairs": len(sources), "n_excluded_pairs": len(exclusions), "n_aligned_pairs": count,
            "expected_n_layouts": count, "expected_n_pairs": count,
            "expected_n_layouts_by_family": actual.copy(), "expected_n_pairs_by_family": actual.copy(),
            "requested_n_layouts": config["n_layouts"], "requested_n_layouts_by_family": deepcopy(quotas),
            "layout_shortfall": config["n_layouts"] - count,
            "layout_shortfall_by_family": {family: quotas[family] - actual[family] for family in FAMILIES},
            "cohort_complete": actual == quotas, "eligible_pair_ids": pair_ids,
            "layout_aliases": dict(sorted(layout_aliases.items())), "tokenizer": _tokenizer_metadata(tokenizer),
            "renderer_id": CANONICAL, "renderer_binding": renderer_manifest(CANONICAL),
            "primary_comparison_ids": {row["primary_id"]: row["id"] for row in primary},
            "primary_endpoint": ENDPOINT, "native_unresolved_policy": UNRESOLVED_POLICY,
            "primary_estimand": "frozen-family-weighted paired native target-action ITT probability difference",
            "target_action_basis": "unique_simulator_counterfactual_optimum", "claim_scope": CLAIM_SCOPE,
            "semantic_headline_eligible": False, "independent_rescue_evidence": False,
            "protocol": {"layer": layer, "layers": [layer], "site": "resid_post", "seeds": [42], "random_seeds": [42],
                "natural": True, "temperature": 0.7, "max_new_tokens": 4096, "renderer_id": CANONICAL,
                "renderer_binding": renderer_manifest(CANONICAL), "selection_scope": "explicit_named_fact_input_tokens",
                "source_span_localization": SOURCE_RULE, "input_alignment_rule": ALIGNMENT,
                "condition_inventory": condition_inventory(layer), "conditions_per_layout": 15,
                "requires_complete_fifteen_condition_cohort": True,
                "requires_token_id_override_support": True, "requires_explicit_position_support": True,
                "require_joint_renderer_eligibility": config["require_joint_renderer_eligibility"],
                "joint_eligibility_required_condition_inventory": condition_inventory(layer),
                "full_write_relative_tolerance": 0.01, "zero_reference_requires_exact_zero": True,
                "same_fact_contrast_is_not_norm_matched": True,
                "existing_random_reference": "matched_counterfactual", "new_random_reference": "natural_donor",
                "random_orientation_support_and_seed_shared": True, "lexical_supplement_glue_pending": True,
                "confirmation_requires_external_freeze": True, "confirmation_requires_external_ready_check": True,
                "primary_family_requires_external_exact_binding": True, "model_queries": 0,
                "outcome_dependent_exclusion_permitted": False, "semantic_headline_eligible": False},
            "hypothesis": "Named-fact input states may route contextual task information on fresh layouts; both natural-versus-same-fact and natural-versus-norm-random contrasts are required, without an abstract-belief claim."}
