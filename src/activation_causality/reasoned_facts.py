"""Authored Boolean facts appended to exact, genuinely generated analysis IDs.

The generated prefix is never re-encoded. The leading-space fact suffix is
encoded separately and concatenated intentionally, so its tokenization need not
equal canonical tokenization of the combined display string.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

from .designs import state_control_specs
from .fact_interventions import ACTION_WORD, LEADS, _clause, _encode, _sentence
from .reasoning_windows import ANALYSIS_BOUNDARIES, POPULATION_SCOPE, _decode, _source_prefix, _valid_ids
from .registry import SCHEMA_VERSION, fingerprint


def _suffixes(tokenizer: Any, sentences: tuple[str, str], last_k: int,
              required_length: int | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Match standalone suffix lengths without touching any generated token."""
    encoded = [_encode(tokenizer, sentence) for sentence in sentences]
    if any(not values or not _valid_ids(values) for values in encoded):
        return None, "invalid_fact_token_ids"
    if len(encoded[0]) != len(encoded[1]):
        return None, "unequal_separately_encoded_fact_lengths"
    width = len(encoded[0])
    if width < last_k:
        return None, "fact_sentence_shorter_than_patch_window"
    if required_length is not None and width != required_length:
        return None, "irrelevant_fact_length_differs_from_relevant_fact"
    decoded = [_decode(tokenizer, values) for values in encoded]
    if any(ACTION_WORD.search(text) or any(marker in text for marker in ANALYSIS_BOUNDARIES)
           for text in (*sentences, *decoded)):
        return None, "fact_contains_action_name_or_channel_boundary"
    return {"sentences": list(sentences), "token_ids": encoded, "token_count": width,
            "decoded_sentences": decoded}, None


def _bound_pair(template: dict[str, Any], source_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    pair = template.get("metadata", {}).get("pair")
    if not isinstance(pair, dict) or not pair:
        raise ValueError("Source baseline requires its authoritative simulator pair metadata")
    pair_id = pair["pair_id"]
    targets, originals = pair["counterfactual_optimal_actions"], pair["recipient_optimal_actions"]
    if pair["partition"] != "discovery" or len(targets) != 1 or len(originals) != 1 or targets == originals:
        raise ValueError("Late-fact inputs require discovery pairs with distinct singleton optimal actions")
    for job in source_jobs:
        meta = job.get("metadata", {})
        if fingerprint(meta.get("pair")) != fingerprint(pair):
            raise ValueError("Simulator pair metadata differs between predeclared source seeds")
        if job["pair_id"] != pair_id and meta.get("source_pair_id") != pair_id:
            raise ValueError("Source pair lineage disagrees with nested simulator pair")
        if job["layout_id"] != pair["layout_id"] or job["family"] != pair["family"]:
            raise ValueError("Source layout or family disagrees with simulator pair")
        if job["target_action"] != targets[0] or job.get("original_action", originals[0]) != originals[0]:
            raise ValueError("Source frozen actions disagree with simulator action sets")
        if meta.get("authored_analysis_sentence"):
            raise ValueError("Source prefix must precede any authored fact intervention")
    return deepcopy(pair)


def build_reasoned_fact_spec(
    source_spec: dict[str, Any], source_rows: Iterable[dict[str, Any]], tokenizer: Any,
    experiment_id: str, cuts: Iterable[int] = (32, 128), layers: Iterable[int] = (10, 15),
    last_k: int = 12, natural: bool = True, cap: int = 4096,
) -> dict[str, Any]:
    """Build late authored-fact discovery jobs using recipient baselines only.

    Each pair/cut requires every planned recipient seed to reach the cut before
    an analysis boundary or standalone direction word. Eligibility never uses
    generated actions or success. The caller must verify registered source-file
    hashes and retirement status before invoking this pure builder.
    """
    from .campaign import make_job

    if source_spec.get("partition") != "discovery":
        raise ValueError("Reasoned-fact search is discovery-only")
    cuts, layers = tuple(cuts), tuple(layers)
    if not cuts or len(set(cuts)) != len(cuts) or any(type(cut) is not int or cut < 1 for cut in cuts):
        raise ValueError("cuts must be unique positive token counts")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers):
        raise ValueError("layers must be unique indices in 0..23")
    if type(last_k) is not int or last_k < 1 or type(cap) is not int or cap < 1 or type(natural) is not bool:
        raise ValueError("Invalid generation or patch-window configuration")
    if not experiment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in experiment_id):
        raise ValueError("Invalid experiment_id")
    source_jobs = {job["job_id"]: job for job in source_spec["jobs"]}
    if len(source_jobs) != len(source_spec["jobs"]):
        raise ValueError("Source specification contains duplicate job IDs")
    baselines: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for job in source_jobs.values():
        if job["condition"] != "baseline":
            continue
        if job.get("experiment_id") != source_spec.get("experiment_id") or job.get("partition") != "discovery":
            raise ValueError("Source job provenance disagrees with discovery specification")
        if type(job["seed"]) is not int or job["seed"] < 0:
            raise ValueError("Source baseline seed must be a nonnegative integer")
        if job["seed"] in baselines[job["pair_id"]]:
            raise ValueError("Source repeats a baseline pair/seed")
        baselines[job["pair_id"]][job["seed"]] = job
    results = {}
    for row in source_rows:
        if row.get("job_id") not in source_jobs:
            continue
        if row["job_id"] in results:
            raise ValueError("Duplicate source result job ID")
        results[row["job_id"]] = row
    jobs, manifest = [], []
    condition_pairs: dict[str, set[str]] = defaultdict(set)
    pair_seeds, pair_layout = {}, {}
    aliases_by_layout: dict[str, set[str]] = defaultdict(set)
    for source_pair_id, by_seed in sorted(baselines.items()):
        seeds = sorted(by_seed)
        template = by_seed[seeds[0]]
        pair = _bound_pair(template, list(by_seed.values()))
        sentences = (_sentence(_clause(pair, "recipient")), _sentence(_clause(pair, "counterfactual")))
        relevant, relevant_reason = _suffixes(tokenizer, sentences, last_k)
        irrelevant_control = next(row for row in state_control_specs(pair) if row["condition"] == "irrelevant_fact_donor")
        irrelevant, irrelevant_reason = None, irrelevant_control.get("ineligible_reason")
        if irrelevant_control["eligible"] and relevant is not None:
            row, col = irrelevant_control["irrelevant_cell"]
            truth = pair["recipient"]["grid"][row][col] == "#"
            for lead in LEADS:
                controls = tuple(_sentence(f"the cell at ({row},{col}) contains a wall: {value}", lead)
                                 for value in (truth, not truth))
                irrelevant, irrelevant_reason = _suffixes(tokenizer, controls, last_k, relevant["token_count"])
                if irrelevant is not None:
                    irrelevant.update({"cell": [row, col], "length_match_lead": lead})
                    break
        aliases = {pair["layout_id"], *pair.get("metadata", {}).get("variant_layout_ids", []),
                   *irrelevant_control.get("variant_layout_ids", [])}
        for job in by_seed.values():
            aliases.update(job.get("layout_aliases", []))
        aliases_by_layout[pair["layout_id"]].update(aliases)
        for cut in cuts:
            pair_id = f"{source_pair_id}::cut{cut}"
            pair_layout[pair_id], pair_seeds[pair_id] = pair["layout_id"], seeds
            # The patch covers authored suffix tokens, so only one generated
            # token is needed for the shared eligibility helper's own window.
            audits = {seed: _source_prefix(by_seed[seed], results.get(by_seed[seed]["job_id"]), tokenizer, cut, 1)
                      for seed in seeds}
            prefix_eligible = all(audit["eligible"] for audit in audits.values())
            eligible = prefix_eligible and relevant is not None
            record = {"pair_id": pair_id, "source_pair_id": pair["pair_id"], "source_baseline_pair_id": source_pair_id,
                      "layout_id": pair["layout_id"], "cut": cut, "seeds": seeds, "eligible": eligible,
                      "reason": None if eligible else ("one_or_more_predeclared_source_seeds_ineligible" if not prefix_eligible else relevant_reason),
                      "recipient_sources": {str(seed): audit for seed, audit in audits.items()},
                      "requires_all_predeclared_seeds": True, "counterfactual_prefix_required": False,
                      "fact_alignment": "separately_encoded_equal_length_suffixes_concatenated_to_exact_prefix_ids",
                      "fact_suffixes": relevant, "fact_exclusion_reason": relevant_reason,
                      "irrelevant_eligible": eligible and irrelevant is not None,
                      "irrelevant_exclusion_reason": irrelevant_reason,
                      "irrelevant_suffixes": irrelevant, "authored_fact_after_generated_reasoning": True,
                      "selection_basis": "unpatched_recipient_trace_boundaries_and_simulator_facts_and_tokenizer_only",
                      "selection_uses_model_action": False, "semantic_headline_eligible": False}
            manifest.append(record)
            if not eligible:
                continue

            def add(condition: str, data: dict[str, Any], recipient_index: int, donor_index: int | None,
                    patch_layers: list[int], *, role: str, baseline: str, full_span: bool = False) -> None:
                condition_pairs[condition].add(pair_id)
                width = data["token_count"] if full_span else last_k
                for seed in seeds:
                    audit, source_job = audits[seed], by_seed[seed]
                    prefix_ids = audit["token_ids"]
                    recipient_ids = prefix_ids + data["token_ids"][recipient_index]
                    donor_ids = prefix_ids + data["token_ids"][donor_index] if donor_index is not None else None
                    recipient_positions = list(range(len(recipient_ids) - width, len(recipient_ids)))
                    donor_positions = list(range(len(donor_ids) - width, len(donor_ids))) if donor_ids is not None else None
                    recipient_sentence = data["sentences"][recipient_index]
                    donor_sentence = data["sentences"][donor_index] if donor_index is not None else None
                    jobs.append(make_job(
                        experiment_id, layout_id=pair["layout_id"], pair_id=pair_id, family=pair["family"],
                        partition="discovery", condition=condition, target_action=pair["counterfactual_optimal_actions"][0],
                        original_action=pair["recipient_optimal_actions"][0], seed=seed,
                        recipient_text=audit["display_text"] + recipient_sentence,
                        donor_text=audit["display_text"] + donor_sentence if donor_sentence is not None else None,
                        recipient_token_ids=recipient_ids, donor_token_ids=donor_ids,
                        selection={"scope": "explicit", "recipient_positions": recipient_positions,
                                   "donor_positions": donor_positions},
                        intervention={"layers": patch_layers, "site": "resid_post", "operation": "replace", "dose": 1.0},
                        natural=natural, max_new_tokens=cap, temperature=source_job.get("temperature", 0.7),
                        semantic_test=False, layout_aliases=sorted(aliases),
                        metadata={"pair": deepcopy(pair), "source_experiment_id": source_spec["experiment_id"],
                                  "source_pair_id": pair["pair_id"], "source_baseline_pair_id": source_pair_id,
                                  "recipient_source_job_id": source_job["job_id"], "source_result_fingerprint": audit["source_result_fingerprint"],
                                  "source_input_token_sha256": audit["source_input_token_sha256"],
                                  "cut": cut, "control_role": role, "paired_baseline_condition": baseline,
                                  "recipient_sentence": recipient_sentence, "donor_sentence": donor_sentence,
                                  "authored_analysis_sentence": True, "authored_fact_after_generated_reasoning": True,
                                  "genuinely_generated_prefix": True, "naturally_generated_fact_sentence": False,
                                  "action_name_free_sentence": True, "recipient_fact_true_in_prompt": recipient_index == 0,
                                  "donor_assertion_is_counterfactual": donor_index == 1,
                                  "base_token_count": audit["base_token_count"], "generated_prefix_token_count": cut,
                                  "shared_prefix_token_count": len(prefix_ids), "fact_start_token": len(prefix_ids),
                                  "fact_end_token_exclusive": len(recipient_ids), "fact_token_count": data["token_count"],
                                  "recipient_patch_positions": recipient_positions, "donor_patch_positions": donor_positions,
                                  "prefix_token_sha256": fingerprint(prefix_ids), "fact_suffix_separately_encoded": True,
                                  "token_concatenation_intentional": True, "full_display_string_canonical_bpe_assumed": False,
                                  "token_ids_authoritative": True, "generated_text_retokenized": False,
                                  "same_unchanged_recipient_prompt_and_reasoning": True,
                                  "irrelevant_control_has_own_baseline": True, "semantic_headline_eligible": False,
                                  "independent_rescue_evidence": False, "population_scope": POPULATION_SCOPE},
                    ))

            baseline, donor = f"baseline_cut{cut}", f"donor_text_cut{cut}"
            add(baseline, relevant, 0, None, [], role="true_fact_baseline", baseline=baseline)
            add(donor, relevant, 1, None, [], role="counterfactual_fact_baseline", baseline=donor)
            add(f"all_layers_full_cut{cut}", relevant, 0, 1, list(range(24)), role="full_span_positive", baseline=baseline, full_span=True)
            if irrelevant is not None:
                add(f"irrelevant_baseline_cut{cut}", irrelevant, 0, None, [], role="irrelevant_true_fact_baseline", baseline=f"irrelevant_baseline_cut{cut}")
                add(f"irrelevant_donor_text_cut{cut}", irrelevant, 1, None, [], role="irrelevant_false_fact_baseline", baseline=f"irrelevant_donor_text_cut{cut}")
            for layer in layers:
                add(f"layer{layer}_cut{cut}", relevant, 0, 1, [layer], role="forward", baseline=baseline)
                add(f"identity_layer{layer}_cut{cut}", relevant, 0, 0, [layer], role="identity", baseline=baseline)
                add(f"reverse_layer{layer}_cut{cut}", relevant, 1, 0, [layer], role="reverse", baseline=donor)
                if irrelevant is not None:
                    add(f"irrelevant_layer{layer}_cut{cut}", irrelevant, 0, 1, [layer], role="irrelevant", baseline=f"irrelevant_baseline_cut{cut}")
    comparisons = []

    def compare(treatment: str, control: str, role: str, direction: str) -> None:
        eligible_ids = sorted(condition_pairs[treatment] & condition_pairs[control])
        if eligible_ids:
            comparisons.append({"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
                                "partition": "discovery", "treatment": treatment, "control": control, "role": role,
                                "eligible_pair_ids": eligible_ids, "expected_n_pairs": sum(len(pair_seeds[pair_id]) for pair_id in eligible_ids),
                                "expected_n_layouts": len({pair_layout[pair_id] for pair_id in eligible_ids}),
                                "predicted_direction": direction, "semantic_headline_eligible": False,
                                "eligibility_basis": "all_predeclared_recipient_seeds_have_premention_prefix_and_aligned_fact_suffix",
                                "exploratory": True})

    for cut in cuts:
        compare(f"donor_text_cut{cut}", f"baseline_cut{cut}", "late_action_free_counterfactual_text", "increase")
        compare(f"all_layers_full_cut{cut}", f"donor_text_cut{cut}", "full_span_positive_control_agreement", "zero")
        for layer in layers:
            compare(f"layer{layer}_cut{cut}", f"baseline_cut{cut}", "late_authored_fact_state_sufficiency", "increase")
            compare(f"identity_layer{layer}_cut{cut}", f"baseline_cut{cut}", "engineering_identity", "zero")
            compare(f"reverse_layer{layer}_cut{cut}", f"donor_text_cut{cut}", "reverse_fact_replacement", "decrease")
            compare(f"irrelevant_layer{layer}_cut{cut}", f"irrelevant_baseline_cut{cut}", "irrelevant_state_fact_sensitivity", "zero")
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "discovery",
            "hypothesis": "An authored action-name-free state fact appended after genuine analysis may redirect continuation; this tests late authored-text sufficiency and does not identify a naturally formed belief mechanism.",
            "primary_estimand": "layout-weighted paired native target-action ITT probability difference",
            "source_experiment_id": source_spec["experiment_id"], "source_spec_fingerprint": fingerprint(source_spec),
            "source_result_fingerprints": {key: fingerprint(value) for key, value in sorted(results.items())},
            "jobs": jobs, "comparisons": comparisons, "eligibility_manifest": manifest,
            "layout_aliases": {key: sorted(value) for key, value in sorted(aliases_by_layout.items())},
            "protocol": {"cuts": list(cuts), "layers": list(layers), "last_k": last_k,
                         "all_predeclared_recipient_seeds_required": True, "counterfactual_prefix_required": False,
                         "fact_suffix_separately_encoded": True, "token_concatenation_intentional": True,
                         "equal_length_fact_suffixes_required": True, "generated_text_retokenized": False,
                         "requires_token_id_override_support": True, "requires_explicit_position_support": True,
                         "authored_fact_after_generated_reasoning": True, "irrelevant_control_own_baseline": True,
                         "semantic_headline_eligible": False, "population_scope": POPULATION_SCOPE}}
