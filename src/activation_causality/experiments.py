"""Pure builders for controlled state-interchange experiments.

No filesystem access, tokenization, model loading, or result-dependent selection
occurs here. Register the returned specification before querying its jobs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .designs import state_control_specs
from .registry import SCHEMA_VERSION, fingerprint


def build_controlled_state_spec(
    pairs: Iterable[dict[str, Any]], experiment_id: str, *,
    layers: Iterable[int] = (3, 10, 15, 18), site: str = "resid_post",
    last_k: int = 1, seeds: Iterable[int] = (42,), max_new_tokens: int = 2048,
    natural: bool = True,
) -> dict[str, Any]:
    """Build registry-compatible jobs and frozen simulator-only eligibility.

    All contrasts retain the same recipient-counterfactual target action. Reverse
    replacement is compared with ``counterfactual_text``. Identity and restoration-
    equivalent patches are engineering controls, not independent rescue evidence.

    Unavailable controls appear in ``eligibility_manifest``; they are not executed
    as identity substitutes. Every comparison specifies ``eligible_pair_ids``.
    The analyzer must apply that pre-query subset before pairing observations.
    Confirmation remains subject to the registry's existing freeze requirement;
    the wrapper must supply the authorized freeze_id, never fabricate one here.
    """
    # Local import avoids a campaign -> builder -> campaign import cycle.
    from .campaign import make_job, render_analysis_prompt

    pairs = deepcopy(list(pairs))
    layers, seeds = tuple(layers), tuple(seeds)
    if not pairs:
        raise ValueError("At least one pair is required")
    if not experiment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in experiment_id):
        raise ValueError("Invalid experiment_id")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers):
        raise ValueError("layers must be unique indices in 0..23")
    if site not in {"resid_post", "attn_out", "mlp_out"}:
        raise ValueError("Unsupported intervention site")
    if type(last_k) is not int or last_k < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Token counts must be positive integers")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if type(natural) is not bool:
        raise ValueError("natural must be Boolean")
    partitions = {pair["partition"] for pair in pairs}
    if len(partitions) != 1 or not partitions <= {"discovery", "confirmation"}:
        raise ValueError("One discovery or confirmation partition per specification; do not relabel layouts")
    partition = next(iter(partitions))
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate pair_id")

    jobs = []
    eligibility_manifest = []
    condition_pairs: dict[str, set[str]] = {}
    layout_aliases: dict[str, set[str]] = {}
    pair_layout = {}
    for pair in pairs:
        target_set, original_set = pair["counterfactual_optimal_actions"], pair["recipient_optimal_actions"]
        if len(target_set) != 1 or len(original_set) != 1 or target_set == original_set:
            raise ValueError("Controlled states require distinct singleton recipient/counterfactual optimal actions")
        target, original = target_set[0], original_set[0]
        if target not in {"UP", "DOWN", "LEFT", "RIGHT"} or original not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("Invalid simulator action")
        controls = {row["condition"]: row for row in state_control_specs(pair)}
        aliases = {pair["layout_id"], *pair.get("metadata", {}).get("variant_layout_ids", [])}
        for control in controls.values():
            aliases.update(control.get("variant_layout_ids", []))
        layout_aliases.setdefault(pair["layout_id"], set()).update(aliases)
        pair_layout[pair["pair_id"]] = pair["layout_id"]
        text_sources = {"recipient": pair["recipient_prompt_text"], "counterfactual": pair["counterfactual_prompt_text"],
                        "donor": pair["donor_prompt_text"]}
        for key, control in controls.items():
            if control["eligible"]:
                if not isinstance(control.get("prompt_text"), str):
                    raise ValueError("Eligible control is missing its actual prompt")
                text_sources[key] = control["prompt_text"]
        texts = {key: render_analysis_prompt(value) for key, value in text_sources.items()}
        eligibility = {"pair_id": pair["pair_id"], "layout_id": pair["layout_id"], "partition": partition,
                       "selection_basis": "simulator_only_before_model_query", "controls": deepcopy(list(controls.values())),
                       "scheduled_conditions": [], "unscheduled_conditions": []}

        def add(condition: str, recipient_key: str, donor_key: str | None,
                patch_layers: list[int], *, role: str, random_control: bool = False,
                baseline_condition: str = "baseline", donor_observation: str | None = None) -> None:
            if recipient_key not in texts or (donor_key is not None and donor_key not in texts):
                key = recipient_key if recipient_key not in texts else donor_key
                control = controls.get(key, {})
                eligibility["unscheduled_conditions"].append({
                    "condition": condition, "eligible": False, "source": key,
                    "reason": control.get("ineligible_reason", "required_source_unavailable"),
                })
                return
            condition_pairs.setdefault(condition, set()).add(pair["pair_id"])
            eligibility["scheduled_conditions"].append(condition)
            is_identity = role in {"identity", "restoration_equivalent"}
            for generation_seed in seeds:
                intervention = {"layers": patch_layers, "site": site, "operation": "replace", "dose": 1.0}
                if random_control:
                    intervention.update({"random_control": True, "random_seed": generation_seed,
                                         "norm_reference": "donor_minus_recipient"})
                jobs.append(make_job(
                    experiment_id, layout_id=pair["layout_id"], pair_id=pair["pair_id"], family=pair["family"],
                    partition=partition, condition=condition, target_action=target, original_action=original,
                    seed=generation_seed, recipient_text=texts[recipient_key],
                    donor_text=texts[donor_key] if donor_key is not None else None,
                    selection={"scope": "last_tokens", "last_k": last_k}, intervention=intervention,
                    natural=natural, max_new_tokens=max_new_tokens, temperature=0.7,
                    semantic_test=not is_identity, layout_aliases=sorted(aliases),
                    metadata={"pair": pair, "intervention_time": "pre_reasoning",
                              "site_note": "last analysis-header/prompt tokens; no generated action or answer-forcing suffix",
                              "control_role": role, "recipient_source": recipient_key, "donor_source": donor_key,
                              "pre_query_eligible": True, "eligibility_basis": "simulator_only",
                              "paired_baseline_condition": baseline_condition,
                              "donor_observation_condition": donor_observation,
                              "observed_donor_action": None, "observed_donor_action_mismatch": None,
                              "ground_truth_donor_optimal_actions": controls[donor_key]["optimal_actions"] if donor_key in controls else (
                                  pair["counterfactual_optimal_actions"] if donor_key == "counterfactual" else
                                  pair["recipient_optimal_actions"] if donor_key == "recipient" else None),
                              "control_spec": controls.get(donor_key), "engineering_identity_only": is_identity,
                              "independent_rescue_evidence": False,
                              "restoration_note": "Counterfactual self-replacement is identity-equivalent, not removal followed by independent rescue."
                              if role == "restoration_equivalent" else None},
                ))

        add("baseline", "recipient", None, [], role="recipient_baseline")
        add("counterfactual_text", "counterfactual", None, [], role="physical_counterfactual_baseline")
        add("donor_text", "donor", None, [], role="actual_pair_donor_baseline")
        add("natural_donor_text", "natural_donor", None, [], role="actual_context_donor_baseline")
        add("same_fact_donor_text", "same_fact_donor", None, [], role="same_fact_context_donor_baseline")
        add("irrelevant_donor_text", "irrelevant_fact_donor", None, [], role="irrelevant_fact_donor_baseline")
        for layer in layers:
            prefix = f"layer{layer}"
            add(prefix + "_matched", "recipient", "counterfactual", [layer], role="matched_fact_forward",
                donor_observation="counterfactual_text")
            add(prefix + "_natural", "recipient", "natural_donor", [layer], role="context_fact_forward",
                donor_observation="natural_donor_text")
            add(prefix + "_same_fact", "recipient", "same_fact_donor", [layer], role="same_fact_context_control",
                donor_observation="same_fact_donor_text")
            add(prefix + "_irrelevant", "recipient", "irrelevant_fact_donor", [layer], role="irrelevant_fact_control",
                donor_observation="irrelevant_donor_text")
            add(prefix + "_random", "recipient", "counterfactual", [layer], role="norm_matched_random",
                random_control=True, donor_observation="counterfactual_text")
            add(prefix + "_reverse", "counterfactual", "recipient", [layer], role="reverse_replacement",
                baseline_condition="counterfactual_text", donor_observation="baseline")
            add(prefix + "_identity", "recipient", "recipient", [layer], role="identity")
            add(prefix + "_restoration_equivalent", "counterfactual", "counterfactual", [layer],
                role="restoration_equivalent", baseline_condition="counterfactual_text")
        eligibility_manifest.append(eligibility)

    comparisons, ineligible_comparisons = [], []

    def compare(treatment: str, control: str, role: str, direction: str = "increase", *,
                semantic_controls: dict[str, str] | None = None) -> None:
        eligible_set = condition_pairs.get(treatment, set()) & condition_pairs.get(control, set())
        # The mechanism-specific contrast is frozen to cases with the full
        # executable control battery; other contrasts retain their wider cohort.
        if role == "semantic_fact_specificity" and semantic_controls:
            for condition in semantic_controls.values():
                eligible_set &= condition_pairs.get(condition, set())
        eligible = sorted(eligible_set)
        entry = {"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
                 "partition": partition, "treatment": treatment, "control": control,
                 "eligible_pair_ids": eligible, "eligibility_basis": "frozen_simulator_only_before_model_query",
                 "role": role, "predicted_direction": direction, "independent_rescue_evidence": False,
                 "expected_n_layouts": len({pair_layout[pair_id] for pair_id in eligible}),
                 "expected_n_pairs": len(eligible) * len(seeds),
                 "outcome_dependent_exclusion_permitted": False}
        if semantic_controls:
            entry["semantic_controls"] = semantic_controls
            entry["eligibility_requires_all_semantic_controls"] = role == "semantic_fact_specificity"
        if eligible:
            comparisons.append(entry)
        else:
            ineligible_comparisons.append({**entry, "reason": "no_pair_has_both_prespecified_sources"})

    compare("counterfactual_text", "baseline", "physical_world_sensitivity")
    for layer in layers:
        prefix = f"layer{layer}"
        semantic_controls = {"counterfactual_baseline": "counterfactual_text",
                             "donor_baseline": "natural_donor_text", "same_fact": prefix + "_same_fact",
                             "irrelevant_fact": prefix + "_irrelevant", "donor_mismatch_treatment": prefix + "_natural"}
        compare(prefix + "_matched", "baseline", "conditional_sufficiency", semantic_controls=semantic_controls)
        compare(prefix + "_matched", prefix + "_random", "norm_matched_specificity", semantic_controls=semantic_controls)
        compare(prefix + "_natural", "baseline", "context_generalization", semantic_controls=semantic_controls)
        compare(prefix + "_natural", prefix + "_same_fact", "semantic_fact_specificity", semantic_controls=semantic_controls)
        compare(prefix + "_matched", prefix + "_irrelevant", "irrelevant_fact_specificity", semantic_controls=semantic_controls)
        compare(prefix + "_reverse", "counterfactual_text", "conditional_reverse_effect", "decrease")
        compare(prefix + "_identity", "baseline", "engineering_identity", "zero")
        compare(prefix + "_restoration_equivalent", "counterfactual_text", "engineering_restoration_equivalence", "zero")
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": partition,
            "hypothesis": "A named task-state variable causally affects native action selection conditional on the recipient world; semantic specificity requires paired context and irrelevant-fact controls.",
            "primary_estimand": "layout-weighted intention-to-treat paired native target-action probability difference",
            "secondary_estimands": ["forced target probability difference", "forced action logits", "completion-sensitive bounds"],
            "dataset_sha256": fingerprint({"pairs": pairs}), "jobs": jobs, "comparisons": comparisons,
            "eligibility_manifest": eligibility_manifest, "ineligible_comparisons": ineligible_comparisons,
            "layout_aliases": {key: sorted(value) for key, value in sorted(layout_aliases.items())},
            "protocol": {"site": site, "layers": list(layers), "last_k": last_k, "seeds": list(seeds),
                         "natural": natural, "max_new_tokens": max_new_tokens,
                         "selection_basis": "simulator_only_before_model_query", "state_control_version": "semantic_controls_v1",
                         "requires_random_control_support": True, "requires_comparison_eligibility_filter": True,
                         "confirmation_requires_external_freeze": partition == "confirmation",
                         "independent_rescue_evidence": False}}
