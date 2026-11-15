"""Action-name-free, aligned Boolean fact-sentence transplant specifications.

These are authored analysis sentences appended to an unchanged recipient prompt,
not a claim that the model naturally generated the sentence or the represented fact.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable

from .designs import state_control_specs
from .registry import SCHEMA_VERSION, fingerprint

ACTION_WORD = re.compile(r"(?<!\w)(?:UP|DOWN|LEFT|RIGHT)(?!\w)", re.IGNORECASE)
LEADS = ("", "Currently, ", "For now, ", "At present, ", "For this grid, ",
         "Considering this grid, ", "While examining this grid, ",
         "As I consider the current grid, ", "Taking the current grid into consideration, ")


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _clause(pair: dict[str, Any], source: str) -> str:
    state = pair[source]
    family = pair["family"]
    if family == "key_possession":
        return "the agent carries a key: " + str(state["has_key"])
    r, c = pair["intervention"]["cell"]
    if family == "wall":
        predicate, truth = "a wall", state["grid"][r][c] == "#"
    elif family == "door_status":
        predicate, truth = "a locked door", state["grid"][r][c] == "D"
    else:
        raise ValueError("Unsupported fact family")
    return f"the cell at ({r},{c}) contains {predicate}: {truth}"


def _sentence(clause: str, lead: str = "") -> str:
    text = " " + lead + "I note that " + clause + ". I will use this state fact while considering the grid."
    if ACTION_WORD.search(text):
        raise ValueError("Fact sentence contains a standalone action name")
    return text


def _aligned_pair(tokenizer: Any, base_text: str, base_ids: list[int], sentences: tuple[str, str],
                  required_length: int | None = None) -> tuple[list[int], list[int], int] | None:
    left, right = (_encode(tokenizer, base_text + sentence) for sentence in sentences)
    n_base = len(base_ids)
    if left[:n_base] != base_ids or right[:n_base] != base_ids or len(left) != len(right):
        return None
    count = len(left) - n_base
    if count <= 0 or (required_length is not None and count != required_length):
        return None
    return left, right, count


def build_fact_spec(
    pairs: Iterable[dict[str, Any]], tokenizer: Any, experiment_id: str,
    layers: Iterable[int] = (3, 10, 15, 18), last_k: int = 12, natural: bool = True,
    *, seeds: Iterable[int] = (42,), max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Pure discovery builder with explicit alignment and control eligibility.

    The irrelevant fact uses its *own* true-fact baseline. Its donor flips only an
    irrelevant wall Boolean; comparing it with the relevant true sentence would
    otherwise conflate intervention effects with which fact was mentioned.
    """
    from .campaign import make_job, render_analysis_prompt

    pairs = deepcopy(list(pairs))
    layers, seeds = tuple(layers), tuple(seeds)
    if not pairs or {pair["partition"] for pair in pairs} != {"discovery"}:
        raise ValueError("Fact-sentence search requires nonempty discovery data")
    if not experiment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in experiment_id):
        raise ValueError("Invalid experiment_id")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers):
        raise ValueError("Invalid layers")
    if type(last_k) is not int or last_k < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Token counts must be positive integers")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Invalid seeds")
    if type(natural) is not bool or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Invalid natural flag or duplicate pair IDs")
    jobs, eligibility_manifest = [], []
    condition_pairs: dict[str, set[str]] = {}
    layout_by_pair = {}
    layout_aliases: dict[str, set[str]] = {}
    for pair in pairs:
        targets, originals = pair["counterfactual_optimal_actions"], pair["recipient_optimal_actions"]
        if len(targets) != 1 or len(originals) != 1 or targets == originals:
            raise ValueError("Fact interventions require distinct singleton simulator action sets")
        base_text = render_analysis_prompt(pair["recipient_prompt_text"])
        base_ids = _encode(tokenizer, base_text)
        true_sentence, false_sentence = _sentence(_clause(pair, "recipient")), _sentence(_clause(pair, "counterfactual"))
        aligned = _aligned_pair(tokenizer, base_text, base_ids, (true_sentence, false_sentence))
        record: dict[str, Any] = {"pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
                                  "recipient_sentence": true_sentence, "donor_sentence": false_sentence,
                                  "eligible": aligned is not None, "alignment": "exact_full_context_tokenization",
                                  "selection_basis": "simulator_facts_and_tokenizer_only", "authored_analysis_sentence": True}
        if aligned is None or aligned[2] < last_k:
            record.update({"eligible": False, "reason": "unequal_lengths_or_bpe_prefix_mismatch" if aligned is None else "sentence_shorter_than_window"})
            eligibility_manifest.append(record)
            continue
        recipient_ids, donor_ids, sentence_count = aligned
        controls = {row["condition"]: row for row in state_control_specs(pair)}
        irrelevant = controls["irrelevant_fact_donor"]
        aliases = {pair["layout_id"], *pair.get("metadata", {}).get("variant_layout_ids", []),
                   *irrelevant.get("variant_layout_ids", [])}
        layout_aliases.setdefault(pair["layout_id"], set()).update(aliases)
        layout_by_pair[pair["pair_id"]] = pair["layout_id"]
        record.update({"reason": None, "base_token_count": len(base_ids), "sentence_token_count": sentence_count,
                       "irrelevant_eligible": False, "irrelevant_reason": irrelevant.get("ineligible_reason")})
        irrelevant_data = None
        if irrelevant["eligible"]:
            r, c = irrelevant["irrelevant_cell"]
            truth = pair["recipient"]["grid"][r][c] == "#"
            for lead in LEADS:
                sentences = tuple(_sentence(f"the cell at ({r},{c}) contains a wall: {value}", lead) for value in (truth, not truth))
                match = _aligned_pair(tokenizer, base_text, base_ids, sentences, sentence_count)
                if match is not None:
                    irrelevant_data = sentences, match
                    record.update({"irrelevant_eligible": True, "irrelevant_reason": None, "irrelevant_cell": [r, c],
                                   "irrelevant_sentences": list(sentences), "irrelevant_length_match_lead": lead,
                                   "irrelevant_token_count": match[2]})
                    break
            if irrelevant_data is None:
                record["irrelevant_reason"] = "no_equal_token_length_irrelevant_fact_template"
        eligibility_manifest.append(record)

        def add(condition: str, recipient: list[int], donor: list[int] | None,
                recipient_sentence: str, donor_sentence: str | None, patch_layers: list[int],
                *, role: str, baseline: str = "baseline", entire_sentence: bool = False) -> None:
            condition_pairs.setdefault(condition, set()).add(pair["pair_id"])
            width = sentence_count if entire_sentence else last_k
            for seed in seeds:
                jobs.append(make_job(
                    experiment_id, layout_id=pair["layout_id"], pair_id=pair["pair_id"], family=pair["family"],
                    partition="discovery", condition=condition, target_action=targets[0], original_action=originals[0], seed=seed,
                    recipient_text=base_text + recipient_sentence,
                    donor_text=base_text + donor_sentence if donor_sentence is not None else None,
                    recipient_token_ids=recipient, donor_token_ids=donor, shared_base_text=base_text,
                    selection={"scope": "last_tokens", "last_k": width},
                    intervention={"layers": patch_layers, "site": "resid_post", "operation": "replace", "dose": 1.0},
                    natural=natural, max_new_tokens=max_new_tokens, temperature=0.7,
                    semantic_test=False, layout_aliases=sorted(aliases),
                    metadata={"pair": pair, "control_role": role, "paired_baseline_condition": baseline,
                              "authored_analysis_sentence": True, "naturally_generated_reasoning": False,
                              "action_name_free_sentence": True, "recipient_sentence": recipient_sentence,
                              "donor_sentence": donor_sentence, "base_token_count": len(base_ids),
                              "sentence_token_count": sentence_count, "input_token_ids_authoritative": True,
                              "patch_positions": list(range(len(recipient) - width, len(recipient))),
                              "same_unchanged_recipient_prompt": True,
                              "recipient_fact_true_in_prompt": role not in {"reverse", "counterfactual_fact_baseline", "irrelevant_false_fact_baseline"},
                              "donor_assertion_is_counterfactual": role in {"forward", "irrelevant", "full_span_positive"},
                              "semantic_headline_eligible": False, "independent_rescue_evidence": False,
                              "population_scope": "simulator-selected fresh random-wall grids, not the historical room-layout sampling distribution",
                              "irrelevant_control_has_own_baseline": True},
                ))

        add("baseline", recipient_ids, None, true_sentence, None, [], role="true_fact_baseline")
        add("donor_text", donor_ids, None, false_sentence, None, [], role="counterfactual_fact_baseline", baseline="donor_text")
        add("all_layers_full", recipient_ids, donor_ids, true_sentence, false_sentence, list(range(24)),
            role="full_span_positive", entire_sentence=True)
        if irrelevant_data:
            (irrelevant_true, irrelevant_false), (ir_ids, ir_donor_ids, _) = irrelevant_data
            add("irrelevant_baseline", ir_ids, None, irrelevant_true, None, [], role="irrelevant_true_fact_baseline", baseline="irrelevant_baseline")
            add("irrelevant_donor_text", ir_donor_ids, None, irrelevant_false, None, [], role="irrelevant_false_fact_baseline", baseline="irrelevant_baseline")
        for layer in layers:
            add(f"layer{layer}", recipient_ids, donor_ids, true_sentence, false_sentence, [layer], role="forward")
            add(f"identity_layer{layer}", recipient_ids, recipient_ids, true_sentence, true_sentence, [layer], role="identity")
            add(f"reverse_layer{layer}", donor_ids, recipient_ids, false_sentence, true_sentence, [layer], role="reverse", baseline="donor_text")
            if irrelevant_data:
                add(f"irrelevant_layer{layer}", ir_ids, ir_donor_ids, irrelevant_true, irrelevant_false, [layer],
                    role="irrelevant", baseline="irrelevant_baseline")
    comparisons = []

    def compare(treatment: str, control: str, role: str) -> None:
        eligible = sorted(condition_pairs.get(treatment, set()) & condition_pairs.get(control, set()))
        if eligible:
            comparisons.append({"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
                                "partition": "discovery", "treatment": treatment, "control": control, "role": role,
                                "eligible_pair_ids": eligible, "expected_n_pairs": len(eligible) * len(seeds),
                                "expected_n_layouts": len({layout_by_pair[pair_id] for pair_id in eligible}),
                                "eligibility_basis": "pre_query_simulator_and_exact_token_alignment",
                                "predicted_direction": "decrease" if role == "reverse_fact_replacement" else (
                                    "zero" if role in {"engineering_identity", "full_span_positive_control_agreement"} else "increase"),
                                "semantic_headline_eligible": False, "exploratory": True})

    compare("donor_text", "baseline", "action_free_counterfactual_text")
    compare("all_layers_full", "donor_text", "full_span_positive_control_agreement")
    for layer in layers:
        compare(f"layer{layer}", "baseline", "action_free_sentence_state_sufficiency")
        compare(f"identity_layer{layer}", "baseline", "engineering_identity")
        compare(f"reverse_layer{layer}", "donor_text", "reverse_fact_replacement")
        compare(f"irrelevant_layer{layer}", "irrelevant_baseline", "irrelevant_state_fact_sensitivity")
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "discovery",
            "hypothesis": "Activation states of an action-name-free counterfactual fact sentence may change native action selection; this is authored-text sufficiency, not independent evidence of a naturally formed belief mechanism.",
            "primary_estimand": "layout-weighted paired native target-action ITT probability difference",
            "jobs": jobs, "comparisons": comparisons, "eligibility_manifest": eligibility_manifest,
            "dataset_sha256": fingerprint({"pairs": pairs}),
            "layout_aliases": {key: sorted(value) for key, value in sorted(layout_aliases.items())},
            "protocol": {"layers": list(layers), "last_k": last_k, "seeds": list(seeds),
                         "authored_analysis_sentence": True, "equal_length_full_context_alignment_required": True,
                         "requires_token_id_override_support": True, "action_name_free": True,
                         "irrelevant_control_own_baseline": True, "semantic_headline_eligible": False}}
