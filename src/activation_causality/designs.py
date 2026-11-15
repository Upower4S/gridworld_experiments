"""Preregistered condition descriptions, independent of tokenizer and model.

These are specifications, not scores. The runner must tokenize full contexts,
verify exact alignment, and record actual spans/layers before any intervention.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .environments import (
    DELTAS, apply_variable, canonical_layout_fingerprint, optimal_actions,
    prompt_text, shortest_path,
)

PROTOCOL_VERSION = "belief_to_action_v1"

CONTROL_MATRIX = (
    {"condition": "recipient_baseline", "purpose": "Unedited recipient action distribution", "kind": "baseline"},
    {"condition": "counterfactual_baseline", "purpose": "Behavior under the physically changed recipient world", "kind": "baseline"},
    {"condition": "donor_baseline", "purpose": "Measure actual donor action; do not substitute simulator labels", "kind": "baseline"},
    {"condition": "self_patch", "purpose": "Identity patch must reproduce recipient logits within declared tolerance", "kind": "engineering"},
    {"condition": "full_sentence_positive", "purpose": "Reproduce explicit donor-sentence effect with full aligned states", "kind": "positive"},
    {"condition": "fact_forward", "purpose": "Transfer only the candidate fact representation into recipient", "kind": "sufficiency"},
    {"condition": "fact_reverse", "purpose": "Replace candidate representation in counterfactual with recipient value", "kind": "conditional_necessity"},
    {"condition": "fact_rescue", "purpose": "Restore the candidate route after reverse replacement", "kind": "rescue"},
    {"condition": "same_fact_donor", "purpose": "Control for donor context while holding the fact fixed", "kind": "specificity"},
    {"condition": "irrelevant_fact", "purpose": "Comparable true fact whose interchange leaves optimal actions unchanged", "kind": "specificity"},
    {"condition": "matched_random", "purpose": "Direction/position control matched for rank and intervention norm", "kind": "disruption"},
    {"condition": "native_continuation", "purpose": "Continue from intervention without appending a forced answer", "kind": "behavioral"},
)


def fact_sentence(pair: dict[str, Any], *, source: str = "recipient", template: int = 0) -> str:
    """One state fact, with no named candidate action; truth evaluated explicitly."""
    if source not in {"recipient", "donor", "counterfactual"}:
        raise ValueError("Unknown state source")
    state = pair[source]
    family = pair["family"]
    if family == "key_possession":
        possession = "True" if state["has_key"] else "False"
        if template == 0:
            return f" The agent's carrying-key status is {possession}."
        if template == 1:
            return " The agent currently " + ("has" if state["has_key"] else "lacks") + " the key."
    else:
        r, c = pair["intervention"]["cell"]
        tile = state["grid"][r][c]
        description = {"#": "a wall", "_": "an open space", "D": "a locked door"}[tile]
        if template == 0:
            return f" The cell at row {r}, column {c} contains {description}."
        if template == 1:
            return f" At ({r},{c}), the grid shows {description}."
    raise ValueError("template must be 0 or 1")


def sentence_condition_specs(pair: dict[str, Any], *, template: int = 0) -> list[dict[str, Any]]:
    """Within-recipient-context fact/conclusion controls, prior to tokenization.

    donor_fact is an injected counterfactual assertion in the recipient context;
    it is not described as a correct report of the unchanged recipient world.
    No token-length equality is assumed from character length or word count.
    """
    target_actions = pair["counterfactual_optimal_actions"]
    if len(target_actions) != 1 or len(pair["recipient_optimal_actions"]) != 1:
        raise ValueError("Sentence conclusions require singleton simulator action sets")
    target, original = target_actions[0], pair["recipient_optimal_actions"][0]
    values = [
        ("recipient_fact", fact_sentence(pair, source="recipient", template=template), True, "state_fact"),
        ("donor_fact", fact_sentence(pair, source="donor", template=template), False, "state_fact"),
        ("neutral", " I should examine the current grid carefully before choosing the next move.", None, "neutral"),
        ("target_conclusion", f" The correct next move is {target}.", None, "action_conclusion"),
        ("original_conclusion", f" The correct next move is {original}.", None, "action_conclusion"),
    ]
    return [{"condition": name, "pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
             "partition": pair["partition"], "family": pair["family"], "template": template,
             "prompt_text": pair["recipient_prompt_text"], "analysis_text": sentence,
             "content_type": kind, "fact_true_in_recipient": truth,
             "target_action": target, "original_action": original,
             "target_source": "simulator_counterfactual", "observed_action": None,
             "alignment_status": "requires_tokenizer_validation"}
            for name, sentence, truth, kind in values]


def intervention_design(pair: dict[str, Any]) -> dict[str, Any]:
    """Give the scheduler a complete data/protocol manifest, never token jobs."""
    return {"protocol_version": PROTOCOL_VERSION, "pair_id": pair["pair_id"],
            "layout_id": pair["layout_id"], "partition": pair["partition"], "family": pair["family"],
            "primary_estimand": "intention-to-treat paired difference in native final-action target probability",
            "secondary_estimands": ["forced target-minus-original action logit difference", "forced target action frequency",
                                     "target state-fact readout", "unrelated fact preservation"],
            "predicted_actions": pair["counterfactual_optimal_actions"],
            "donor_actions": pair["donor_optimal_actions"],
            "ground_truth_donor_action_mismatch": pair["metadata"]["ground_truth_donor_action_mismatch"],
            "observed_donor_action_mismatch": None,
            "intervention": pair["intervention"],
            "controls": [dict(condition) for condition in CONTROL_MATRIX],
            "sentence_conditions": sentence_condition_specs(pair),
            "required_before_confirmation": ["freeze selected sites/ranks/layers", "freeze generator and eligibility",
                                             "freeze score and exclusions", "run identity/positive controls"],
            "claim_constraints": ["simulator optimality is not the model's unobserved action",
                                  "conditional rescue is not global necessity",
                                  "forced action effects are not native continuation effects",
                                  "full-state transfer alone does not isolate semantic content",
                                  "confirmation set must remain unqueried during selection"]}


def _control_row(pair: dict[str, Any], condition: str, state: dict[str, Any] | None,
                 *, reason: str | None = None, **metadata: Any) -> dict[str, Any]:
    return {"condition": condition, "pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
            "partition": pair["partition"], "family": pair["family"], "state": state,
            "prompt_text": prompt_text(state) if state is not None else None,
            "optimal_actions": optimal_actions(state) if state is not None else [],
            "eligible": state is not None, "ineligible_reason": reason,
            "observed_action": None, "selection_basis": "simulator_only",
            "variant_layout_ids": [canonical_layout_fingerprint(state)] if state is not None else [],
            **metadata}


def state_control_specs(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """Concrete semantic-specificity source states, without model-dependent selection.

    ``natural_donor`` and ``same_fact_donor`` share a different agent position;
    their only mutual difference is the target variable. The natural donor's
    simulator action differs from the recipient counterfactual's target. Actual
    model donor-action mismatch must be measured separately. If no valid donor
    is available the conditions remain explicitly ineligible, never identity
    duplicates masquerading as context-specificity controls.

    The irrelevant control changes a remote wall cell while preserving BOTH
    recipient and counterfactual shortest distances and all optimal first moves.
    Control wall-topology aliases must join the campaign's exclusion manifest.
    """
    recipient, counterfactual = pair["recipient"], pair["counterfactual"]
    cell = pair["intervention"]["cell"]
    protected = pair["intervention"].get("door_cell") or cell
    target = pair["counterfactual_optimal_actions"]
    natural = same_fact = None
    # Lexicographic simulator-only search is deterministic and independent of model outputs.
    for r, row in enumerate(counterfactual["grid"]):
        for c, tile in enumerate(row):
            if tile != "_" or [r, c] == recipient["agent"] or [r, c] == protected:
                continue
            candidate = deepcopy(counterfactual)
            candidate["agent"] = [r, c]
            if candidate["has_key"] and any(
                0 <= r + dr < len(candidate["grid"]) and 0 <= c + dc < len(row) and
                candidate["grid"][r + dr][c + dc] == "D" for dr, dc in DELTAS.values()
            ):
                continue
            candidate_actions = optimal_actions(candidate)
            if len(candidate_actions) != 1 or not set(candidate_actions).isdisjoint(target):
                continue
            control = apply_variable(candidate, recipient, pair["family"], cell)
            if shortest_path(control)[0] is None:
                continue
            natural, same_fact = candidate, control
            break
        if natural is not None:
            break
    unavailable = "no_reachable_simulator_mismatched_donor_with_same_fact_control"
    rows = [
        _control_row(pair, "natural_donor", natural, reason=unavailable if natural is None else None,
                     ground_truth_donor_action_mismatch=natural is not None,
                     observed_donor_action_mismatch=None, expected_recipient_actions=target),
        _control_row(pair, "same_fact_donor", same_fact, reason=unavailable if same_fact is None else None,
                     paired_condition="natural_donor", expected_recipient_actions=pair["recipient_optimal_actions"]),
    ]
    irrelevant = None
    irrelevant_cell = None
    for r, row in enumerate(recipient["grid"]):
        for c, tile in enumerate(row):
            if r in (0, len(recipient["grid"]) - 1) or c in (0, len(row) - 1):
                continue
            if tile not in "#_" or [r, c] == protected or [r, c] == recipient["agent"]:
                continue
            if sum(abs(a - b) for a, b in zip([r, c], recipient["agent"])) < 2:
                continue
            changed = []
            for original in (recipient, counterfactual):
                candidate = deepcopy(original)
                chars = list(candidate["grid"][r])
                chars[c] = "_" if tile == "#" else "#"
                candidate["grid"][r] = "".join(chars)
                changed.append(candidate)
            if all(shortest_path(new)[0] == shortest_path(old)[0] and optimal_actions(new) == optimal_actions(old)
                   for new, old in zip(changed, (recipient, counterfactual))):
                irrelevant, irrelevant_cell = changed[0], [r, c]
                irrelevant_counterfactual = changed[1]
                break
        if irrelevant is not None:
            break
    irrelevant_row = _control_row(
        pair, "irrelevant_fact_donor", irrelevant,
        reason="no_remote_wall_change_preserving_distances_and_optimal_sets" if irrelevant is None else None,
        irrelevant_cell=irrelevant_cell, expected_recipient_actions=pair["recipient_optimal_actions"],
        relevance_criterion="same shortest distance and optimal action set in both target-variable states",
    )
    if irrelevant is not None:
        irrelevant_row["variant_layout_ids"] = sorted({canonical_layout_fingerprint(irrelevant),
                                                       canonical_layout_fingerprint(irrelevant_counterfactual)})
    rows.append(irrelevant_row)
    return rows
