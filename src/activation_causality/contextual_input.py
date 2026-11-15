"""Context-controlled interchange at observed, named-fact input-token spans.

Every source span is located by toggling only the named fact inside that same
source context. Different agent locations therefore do not define the changed
span. Activations at those tokens can still carry contextual information; this
protocol does not identify an abstract belief representation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

from .environments import apply_variable, canonical_layout_fingerprint, prompt_text, validate_state
from .experiments import build_controlled_state_spec
from .input_sites import SITES, _physical_counterfactual, aligned_changed_span
from .registry import SCHEMA_VERSION, fingerprint


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("invalid_input_token_ids")
    return values


def _toggle_named_fact(state: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """Keep all source context, changing exactly the registered binary variable."""
    validate_state(state)
    family, cell = pair["family"], pair["intervention"].get("cell")
    recipient_value = apply_variable(state, pair["recipient"], family, cell)
    counterfactual_value = apply_variable(state, pair["counterfactual"], family, cell)
    if state == recipient_value:
        return counterfactual_value
    if state == counterfactual_value:
        return recipient_value
    raise ValueError("source_named_fact_value_outside_registered_binary_contrast")


def build_contextual_input_spec(
    pairs: Iterable[dict[str, Any]], tokenizer: Any, experiment_id: str,
    layers: Iterable[int] = (0, 10), site: str = "resid_post", natural: bool = True,
    seeds: Iterable[int] = (42,), cap: int = 4096,
) -> dict[str, Any]:
    """Pure discovery builder retaining the complete controlled-state matrix.

    Alignment exclusions precede model queries and apply to all registered
    seeds. Actual donor actions remain unknown until the natural donor baseline
    is observed. Confirmation is intentionally unsupported.
    """
    from .campaign import make_job, render_analysis_prompt

    pairs, layers, seeds = deepcopy(list(pairs)), tuple(layers), tuple(seeds)
    if not experiment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in experiment_id):
        raise ValueError("Invalid experiment_id")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers):
        raise ValueError("Invalid layers")
    if site not in SITES or type(natural) is not bool or type(cap) is not int or cap < 1:
        raise ValueError("Invalid site or generation configuration")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Invalid seeds")
    if not pairs or any(pair.get("partition") != "discovery" for pair in pairs):
        raise ValueError("Contextual input search requires nonempty discovery pairs")
    if len({pair.get("pair_id") for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate pair_id")
    physical_exclusions, valid_pairs = [], []
    for pair in pairs:
        try:
            # Validate the matched counterfactual without replacing the actual
            # dataset donor retained by the existing six-baseline matrix.
            _physical_counterfactual(pair)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            physical_exclusions.append({"pair_id": pair.get("pair_id"), "layout_id": pair.get("layout_id"),
                                        "family": pair.get("family"), "reason": str(exc),
                                        "excluded_before_model_query": True})
            continue
        valid_pairs.append(pair)
    if valid_pairs:
        original = build_controlled_state_spec(valid_pairs, experiment_id, layers=layers, site=site,
                                               seeds=seeds, max_new_tokens=cap, natural=natural)
    else:
        original = {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "discovery",
                    "jobs": [], "comparisons": [], "ineligible_comparisons": [], "eligibility_manifest": [],
                    "layout_aliases": {}, "protocol": {"layers": list(layers), "site": site, "seeds": list(seeds)}}
    spec = deepcopy(original)
    template_by_pair = {pair["pair_id"]: pair for pair in valid_pairs}
    manifest_by_pair = {row["pair_id"]: row for row in spec["eligibility_manifest"]}
    jobs_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in original["jobs"]:
        jobs_by_pair[job["pair_id"]].append(job)
    audits, source_ids = {}, {}
    for pair_id, pair_jobs in jobs_by_pair.items():
        pair, manifest = template_by_pair[pair_id], manifest_by_pair[pair_id]
        states = {name: pair[name] for name in ("recipient", "counterfactual", "donor")}
        states.update({row["condition"]: row["state"] for row in manifest["controls"] if row["eligible"]})
        texts = {}
        for job in pair_jobs:
            for role in ("recipient", "donor"):
                source = job["metadata"][role + "_source"]
                if source is not None:
                    texts[source] = job[role + "_text"]
        source_audits = {}
        aliases = set(spec["layout_aliases"][pair["layout_id"]])
        for source, state in states.items():
            audit = {"source": source, "eligible": False, "excluded_before_model_query": True,
                     "alignment_basis": "toggle_only_named_variable_in_same_source_context"}
            try:
                toggled = _toggle_named_fact(state, pair)
                text = render_analysis_prompt(prompt_text(state))
                if texts.get(source) != text:
                    raise ValueError("source_prompt_does_not_match_physical_state")
                tokens = _encode(tokenizer, text)
                toggle_tokens = _encode(tokenizer, render_analysis_prompt(prompt_text(toggled)))
                span = aligned_changed_span(tokens, toggle_tokens)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                audit["reason"] = str(exc)
            else:
                source_ids[pair_id, source] = tokens
                aliases.update((canonical_layout_fingerprint(state), canonical_layout_fingerprint(toggled)))
                audit.update(eligible=True, reason=None, excluded_before_model_query=False,
                             span=span, source_input_sha256=fingerprint(tokens), toggle_input_sha256=fingerprint(toggle_tokens),
                             source_state_sha256=fingerprint(state), toggled_state_sha256=fingerprint(toggled),
                             context_agent=deepcopy(state["agent"]), toggled_context_agent=deepcopy(toggled["agent"]),
                             agent_location_unchanged_in_span_localization=state["agent"] == toggled["agent"])
            source_audits[source] = audit
            audits[pair_id, source] = audit
        spec["layout_aliases"][pair["layout_id"]] = sorted(aliases)
        manifest.update(source_span_audits=source_audits, scheduled_conditions=[],
                        selection_basis="simulator_and_tokenizer_only_before_model_query",
                        claim_scope="contextual_input_fact_interchange", semantic_headline_eligible=False)
    retained, exclusions, condition_pairs = [], [], defaultdict(set)
    condition_seed_pairs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    excluded_conditions = set()
    for source_job in original["jobs"]:
        pair_id, condition = source_job["pair_id"], source_job["condition"]
        recipient_source, donor_source = (source_job["metadata"][role + "_source"] for role in ("recipient", "donor"))
        recipient_audit = audits[pair_id, recipient_source]
        donor_audit = audits[pair_id, donor_source] if donor_source is not None else None
        reason = None
        if not recipient_audit["eligible"]:
            reason = f"recipient_source_alignment:{recipient_audit['reason']}"
        elif donor_audit is not None and not donor_audit["eligible"]:
            reason = f"donor_source_alignment:{donor_audit['reason']}"
        elif donor_audit is not None:
            if recipient_audit["span"]["n_input_tokens"] != donor_audit["span"]["n_input_tokens"]:
                reason = "cross_context_unequal_full_token_lengths"
            elif recipient_audit["span"]["n_changed_span_tokens"] != donor_audit["span"]["n_changed_span_tokens"]:
                reason = "cross_context_unequal_named_fact_span_widths"
        if reason is not None:
            if (pair_id, condition) not in excluded_conditions:
                excluded_conditions.add((pair_id, condition))
                entry = {"pair_id": pair_id, "layout_id": source_job["layout_id"], "condition": condition,
                         "recipient_source": recipient_source, "donor_source": donor_source,
                         "eligible": False, "reason": reason, "seeds": list(seeds), "excluded_before_model_query": True}
                exclusions.append(entry)
                manifest_by_pair[pair_id]["unscheduled_conditions"].append(deepcopy(entry))
            continue
        recipient_positions = recipient_audit["span"]["recipient_positions"]
        donor_positions = donor_audit["span"]["recipient_positions"] if donor_audit is not None else None
        job = deepcopy(source_job)
        job.update(recipient_token_ids=source_ids[pair_id, recipient_source],
                   donor_token_ids=source_ids[pair_id, donor_source] if donor_source is not None else None,
                   recipient_positions=recipient_positions, donor_positions=donor_positions,
                   selection={"scope": "explicit", "recipient_positions": recipient_positions, "donor_positions": donor_positions},
                   semantic_test=False, semantic_headline_eligible=False,
                   claim_scope="contextual_input_fact_interchange", layout_aliases=spec["layout_aliases"][job["layout_id"]])
        job["metadata"].update(
            intervention_time="named_fact_input_token", claim_scope="contextual_input_fact_interchange",
            representation_kind="observed_fact_input_tokens", abstract_belief_representation_claim=False,
            semantic_headline_eligible=False, input_token_ids_authoritative=True, no_generated_cot_tokens=True,
            site_note="Named-variable changed span localized separately within each fixed source context",
            eligibility_basis="simulator_and_exact_tokenizer_alignment_before_model_query",
            recipient_input_fact_span=deepcopy(recipient_audit["span"]),
            donor_input_fact_span=deepcopy(donor_audit["span"]) if donor_audit is not None else None,
            recipient_patch_positions=recipient_positions, donor_patch_positions=donor_positions,
            donor_span_not_located_by_recipient_donor_context_difference=True,
            contextual_information_may_be_encoded_in_fact_token_states=True,
        )
        job.pop("job_id")
        job.pop("experiment_id")
        retained.append(make_job(experiment_id, **job))
        condition_pairs[condition].add(pair_id)
        condition_seed_pairs[condition].add((pair_id, source_job["seed"]))
        scheduled = manifest_by_pair[pair_id]["scheduled_conditions"]
        if condition not in scheduled:
            scheduled.append(condition)
    comparisons, ineligible_comparisons = [], []
    for original_comparison in original["comparisons"] + original["ineligible_comparisons"]:
        comparison = deepcopy(original_comparison)
        eligible = condition_pairs[comparison["treatment"]] & condition_pairs[comparison["control"]]
        required_conditions = {comparison["treatment"], comparison["control"]}
        if comparison.get("eligibility_requires_all_semantic_controls"):
            required_conditions.update(comparison["semantic_controls"].values())
            for name in required_conditions:
                eligible &= condition_pairs[name]
        complete_seed_pairs = set.intersection(*(condition_seed_pairs[name] for name in required_conditions))
        complete_seed_pairs = {key for key in complete_seed_pairs if key[0] in eligible}
        comparison.update(eligible_pair_ids=sorted(eligible), expected_n_pairs=len(complete_seed_pairs),
                          expected_n_layouts=len({template_by_pair[pair_id]["layout_id"] for pair_id in eligible}),
                          eligibility_basis="frozen_simulator_and_per_source_fact_span_alignment_before_model_query",
                          semantic_headline_eligible=False, claim_scope="contextual_input_fact_interchange")
        comparison.pop("reason", None)
        if eligible:
            comparisons.append(comparison)
        else:
            ineligible_comparisons.append({**comparison, "reason": "no_complete_prespecified_control_cohort_after_alignment"})
    spec.update(jobs=retained, comparisons=comparisons, ineligible_comparisons=ineligible_comparisons,
                physical_pair_exclusions=physical_exclusions, alignment_exclusions=exclusions,
                alignment_exclusion_counts=dict(sorted(Counter(row["reason"] for row in exclusions).items())),
                n_source_pairs=len(pairs), n_physically_valid_pairs=len(valid_pairs),
                n_aligned_recipient_pairs=len(condition_pairs["baseline"] & condition_pairs["counterfactual_text"]),
                source_controlled_spec_fingerprint=fingerprint(original), source_pairs_sha256=fingerprint(pairs),
                claim_scope="contextual_input_fact_interchange", semantic_headline_eligible=False,
                primary_estimand="layout-weighted intention-to-treat paired native target-action probability difference",
                hypothesis="Named-fact observation-token states may route task information across different donor contexts; same-fact and irrelevant-context controls constrain observation specificity, without establishing an abstract belief mechanism.")
    spec["protocol"].pop("last_k", None)
    spec["protocol"].update(selection_scope="explicit_named_fact_input_tokens", requires_explicit_position_support=True,
                            requires_token_id_override_support=True, source_span_localization="within_context_named_variable_toggle",
                            cross_context_equal_full_lengths_required=True, cross_context_equal_span_width_required=True,
                            all_registered_seeds_share_prequery_eligibility=True, semantic_headline_eligible=False,
                            actual_donor_mismatch_requires_natural_donor_baseline_observation=True,
                            claim_scope="contextual_input_fact_interchange")
    return spec
