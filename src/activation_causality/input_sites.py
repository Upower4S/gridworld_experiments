"""Freeze matched physical-fact token interventions without model inference.

These interventions replace states at changed *input fact tokens*. Their
interpretation is information routing from a wall/key/door observation, not
localization of an abstract belief or a generated chain-of-thought state.
Tokenization failures are recorded as exclusions before any model query.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .environments import apply_variable, optimal_actions, prompt_text, validate_state
from .registry import SCHEMA_VERSION, fingerprint


SITES = frozenset({"resid_post", "attn_out", "mlp_out"})
FAMILIES = frozenset({"wall", "key_possession", "door_status"})


def aligned_changed_span(recipient_ids: Sequence[int], donor_ids: Sequence[int]) -> dict[str, Any]:
    """Return the minimal contiguous changed token region with exact alignment.

    A common prefix and nonoverlapping common suffix leave the changed span.
    Equal complete sequence lengths are mandatory; no resampling, truncation,
    padding, or nearest-token matching is performed.
    """
    recipient_ids, donor_ids = list(recipient_ids), list(donor_ids)
    if not recipient_ids or not donor_ids:
        raise ValueError("empty_token_sequence")
    if len(recipient_ids) != len(donor_ids):
        raise ValueError("unequal_full_token_lengths")
    prefix = 0
    while prefix < len(recipient_ids) and recipient_ids[prefix] == donor_ids[prefix]:
        prefix += 1
    if prefix == len(recipient_ids):
        raise ValueError("empty_changed_token_span")
    suffix = 0
    remaining = len(recipient_ids) - prefix
    while suffix < remaining and recipient_ids[-1 - suffix] == donor_ids[-1 - suffix]:
        suffix += 1
    end = len(recipient_ids) - suffix
    positions = list(range(prefix, end))
    if not positions:
        raise ValueError("empty_changed_token_span")
    return {"recipient_positions": positions, "donor_positions": positions.copy(),
            "common_prefix_tokens": prefix, "common_suffix_tokens": suffix,
            "n_input_tokens": len(recipient_ids), "n_changed_span_tokens": len(positions),
            "recipient_changed_token_ids": recipient_ids[prefix:end], "donor_changed_token_ids": donor_ids[prefix:end],
            "alignment": "exact_equal_length_common_prefix_and_suffix"}


def _physical_counterfactual(pair: Mapping[str, Any], partition: str = "discovery") -> tuple[dict, str, str, str, str]:
    """Validate that the physical donor differs in exactly the named variable."""
    if pair.get("partition") != partition:
        raise ValueError(f"non_{partition}_pair")
    if not all(isinstance(pair.get(name), str) and pair[name] for name in ("pair_id", "layout_id")):
        raise ValueError("missing_pair_or_layout_id")
    family = pair.get("family")
    if family not in FAMILIES:
        raise ValueError("unsupported_fact_family")
    recipient, counterfactual = pair.get("recipient"), pair.get("counterfactual")
    if not isinstance(recipient, dict) or not isinstance(counterfactual, dict):
        raise ValueError("missing_physical_states")
    validate_state(recipient)
    validate_state(counterfactual)
    intervention = pair.get("intervention", {})
    if intervention.get("variable") != family:
        raise ValueError("designated_variable_mismatch")
    rebuilt = apply_variable(recipient, counterfactual, family, intervention.get("cell"))
    if rebuilt != counterfactual or recipient == counterfactual:
        raise ValueError("counterfactual_changes_more_than_designated_fact_or_is_identical")
    if partition == "confirmation":
        from .environments import canonical_layout_fingerprint, partition_for_layout
        expected_layout = canonical_layout_fingerprint(
            recipient, mutable_cells=[intervention["cell"]] if family == "wall" else [])
        if pair["layout_id"] != expected_layout or partition_for_layout(expected_layout) != partition:
            raise ValueError("confirmation_layout_reassigned_or_mismatched")
        metadata = pair.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("model_outputs_observed") is not False or metadata.get("confirmation_status") != "unqueried":
            raise ValueError("confirmation_pair_not_declared_unqueried")
        if metadata.get("partition", partition) != partition:
            raise ValueError("confirmation_partition_metadata_conflict")
    if family == "key_possession":
        recipient_value, donor_value = recipient["has_key"], counterfactual["has_key"]
    else:
        r, c = intervention["cell"]
        recipient_value, donor_value = recipient["grid"][r][c], counterfactual["grid"][r][c]
    if intervention.get("recipient_value", recipient_value) != recipient_value or intervention.get("donor_value", donor_value) != donor_value:
        raise ValueError("recorded_fact_values_disagree_with_states")
    recipient_text, counterfactual_text = prompt_text(recipient), prompt_text(counterfactual)
    if pair.get("recipient_prompt_text", recipient_text) != recipient_text:
        raise ValueError("recipient_prompt_does_not_match_physical_state")
    if pair.get("counterfactual_prompt_text", counterfactual_text) != counterfactual_text:
        raise ValueError("counterfactual_prompt_does_not_match_physical_state")
    original_actions, target_actions = optimal_actions(recipient), optimal_actions(counterfactual)
    if len(original_actions) != 1 or len(target_actions) != 1 or original_actions == target_actions:
        raise ValueError("physical_pair_needs_distinct_unique_optimal_actions")
    if pair.get("recipient_optimal_actions", original_actions) != original_actions or pair.get("counterfactual_optimal_actions", target_actions) != target_actions:
        raise ValueError("recorded_optimal_actions_disagree_with_simulator")
    # Always use the matched counterfactual, even when the source dataset also
    # supplies a deliberately different-context donor for another experiment.
    normalized = deepcopy(dict(pair))
    if not isinstance(normalized.get("metadata", {}), dict):
        raise ValueError("invalid_pair_metadata")
    normalized.update(donor=deepcopy(counterfactual), donor_prompt_text=counterfactual_text,
                      donor_optimal_actions=target_actions, recipient_prompt_text=recipient_text,
                      counterfactual_prompt_text=counterfactual_text,
                      recipient_optimal_actions=original_actions, counterfactual_optimal_actions=target_actions)
    normalized.setdefault("metadata", {}).update(
        donor_context="matched_physical_counterfactual", source_pair_sha256=fingerprint(dict(pair)),
        ground_truth_donor_action_mismatch=False, model_outputs_observed=False,
    )
    return normalized, recipient_text, counterfactual_text, original_actions[0], target_actions[0]


def build_input_site_spec(
    pairs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    experiment_id: str,
    layers: Sequence[int] = (0, 3, 10, 15, 18),
    site: str = "resid_post",
    natural: bool = True,
    seeds: Sequence[int] = (42,),
    cap: int = 2048,
    *,
    partition: str = "discovery",
    freeze_id: str | None = None,
) -> dict[str, Any]:
    """Build a discovery spec and explicit pre-query exclusions; never run a model.

    The supplied tokenizer must encode with add_special_tokens=False. Jobs
    freeze full Harmony token IDs and explicit donor/recipient positions.
    Random controls ask the worker for an independent Gaussian direction at
    each position, matched to the real replacement's per-token L2 norm; the
    worker computes those norms from captures at execution time.

    Discovery defaults retain the original specification format. Explicit
    confirmation requires a freeze ID, one layer and native generation; pairs
    retain their actual deterministic partition. Use the frozen adapter for
    primary comparison and cohort accounting, then register before inference.
    """
    # Local import permits the campaign CLI to import this builder without a
    # module-initialization cycle. The renderer performs no IO or model calls.
    from .campaign import make_job, render_analysis_prompt

    if not isinstance(experiment_id, str) or not experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in experiment_id):
        raise ValueError("invalid experiment_id")
    layers, seeds = tuple(layers), tuple(seeds)
    if not layers or len(set(layers)) != len(layers) or any(isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < 24 for layer in layers):
        raise ValueError("layers must be distinct GPT-OSS-20B block indices 0..23")
    if site not in SITES:
        raise ValueError("unsupported intervention site")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be distinct uint32 integers")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if partition not in {"discovery", "confirmation"}:
        raise ValueError("partition must be discovery or confirmation")
    if partition == "confirmation":
        if not isinstance(freeze_id, str) or not freeze_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in freeze_id):
            raise ValueError("confirmation requires an explicit freeze_id")
        if len(layers) != 1 or natural is not True:
            raise ValueError("confirmation requires one layer and native generation")
    elif freeze_id is not None:
        raise ValueError("freeze_id is only valid in explicit confirmation mode")
    jobs, accepted, exclusions = [], [], []
    seen = set()
    seen_confirmation_aliases = set()
    for index, source_pair in enumerate(pairs):
        identity = {name: source_pair.get(name) for name in ("pair_id", "layout_id", "family")}
        try:
            pair, recipient_user, counterfactual_user, original_action, target_action = _physical_counterfactual(source_pair, partition)
            if pair["pair_id"] in seen:
                raise ValueError("duplicate_pair_id")
            seen.add(pair["pair_id"])
            aliases = []
            if partition == "confirmation":
                from .environments import canonical_layout_fingerprint
                declared_aliases = pair.get("metadata", {}).get("variant_layout_ids", [])
                if not isinstance(declared_aliases, list) or any(not isinstance(a, str) or not a for a in declared_aliases):
                    raise ValueError("invalid_confirmation_layout_aliases")
                aliases = sorted({pair["layout_id"], *declared_aliases,
                                  canonical_layout_fingerprint(pair["recipient"]),
                                  canonical_layout_fingerprint(pair["counterfactual"])})
                if set(aliases) & seen_confirmation_aliases:
                    raise ValueError("duplicate_confirmation_layout_or_alias")
            recipient_text = render_analysis_prompt(recipient_user)
            counterfactual_text = render_analysis_prompt(counterfactual_user)
            recipient_ids = [int(value) for value in tokenizer.encode(recipient_text, add_special_tokens=False)]
            donor_ids = [int(value) for value in tokenizer.encode(counterfactual_text, add_special_tokens=False)]
            span = aligned_changed_span(recipient_ids, donor_ids)
            if recipient_ids[span["recipient_positions"][0] : span["recipient_positions"][-1] + 1] == donor_ids[span["donor_positions"][0] : span["donor_positions"][-1] + 1]:
                raise ValueError("empty_changed_token_span")
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            exclusions.append({"input_index": index, **identity, "reason": str(exc), "excluded_before_model_query": True})
            continue
        seen_confirmation_aliases.update(aliases)
        accepted.append({**identity, "source_pair_sha256": fingerprint(dict(source_pair)),
                         "physical_pair_sha256": fingerprint(pair), "span": span,
                         "recipient_input_sha256": fingerprint(recipient_ids), "donor_input_sha256": fingerprint(donor_ids)})
        conditions = [("baseline", "recipient", None, [], span["recipient_positions"], False),
                      ("counterfactual_text", "counterfactual", None, [], span["recipient_positions"], False)]
        for layer in layers:
            conditions.extend([
                (f"identity_layer{layer}", "recipient", "recipient", [layer], span["recipient_positions"], False),
                (f"factspan_layer{layer}", "recipient", "counterfactual", [layer], span["recipient_positions"], False),
                (f"random_factspan_layer{layer}", "recipient", "counterfactual", [layer], span["recipient_positions"], True),
                (f"lasttoken_layer{layer}", "recipient", "counterfactual", [layer], [len(recipient_ids) - 1], False),
            ])
        texts = {"recipient": recipient_text, "counterfactual": counterfactual_text}
        tokens = {"recipient": recipient_ids, "counterfactual": donor_ids}
        for condition, recipient_key, donor_key, condition_layers, positions, random_control in conditions:
            for seed in seeds:
                jobs.append(make_job(
                    experiment_id, layout_id=pair["layout_id"], pair_id=pair["pair_id"], family=pair["family"],
                    partition=partition, condition=condition, target_action=target_action, original_action=original_action,
                    seed=seed, recipient_text=texts[recipient_key], donor_text=texts[donor_key] if donor_key else None,
                    recipient_token_ids=tokens[recipient_key].copy(), donor_token_ids=tokens[donor_key].copy() if donor_key else None,
                    selection={"scope": "explicit", "recipient_positions": list(positions), "donor_positions": list(positions)},
                    recipient_positions=list(positions), donor_positions=list(positions),
                    intervention={"layers": condition_layers, "site": site, "operation": "replace", "dose": 1.0,
                                  "random_control": random_control},
                    natural=bool(natural), max_new_tokens=cap, temperature=0.7, semantic_test=False,
                    metadata={"pair": deepcopy(pair), "input_fact_span": deepcopy(span),
                              "claim_scope": "input_fact_information_routing", "abstract_belief_representation_claim": False,
                              "representation_kind": "input_fact_tokens" if not condition.startswith("lasttoken") else "pre_reasoning_analysis_header_token",
                              "physical_donor": "matched_counterfactual", "changed_variable": pair["intervention"],
                              "no_generated_cot_tokens": True, "model_based_case_selection": False,
                              "random_norm_matching": "per-position L2 of actual donor-minus-recipient at this layer/site" if random_control else None,
                              **({"source_pair": deepcopy(dict(source_pair)), "semantic_headline_eligible": False,
                                  "target_action_basis": "unique_simulator_counterfactual_optimum"}
                                 if partition == "confirmation" else {})},
                    **({"freeze_id": freeze_id, "layout_aliases": aliases,
                        "semantic_headline_eligible": False, "claim_scope": "input_fact_information_routing"}
                       if partition == "confirmation" else {}),
                ))
    comparisons = []
    for layer in layers:
        for name, treatment, control, role in (
            (f"factspan_vs_baseline_L{layer}", f"factspan_layer{layer}", "baseline", "source_fact_transfer"),
            (f"factspan_vs_random_L{layer}", f"factspan_layer{layer}", f"random_factspan_layer{layer}", "matched_norm_specificity"),
            (f"factspan_vs_lasttoken_L{layer}", f"factspan_layer{layer}", f"lasttoken_layer{layer}", "source_vs_readout_site"),
            (f"identity_L{layer}", f"identity_layer{layer}", "baseline", "technical_identity"),
        ):
            comparisons.append({"id": f"{experiment_id}_{name}", "experiment_id": experiment_id,
                                "partition": partition, "treatment": treatment, "control": control, "role": role})
    comparisons.append({"id": f"{experiment_id}_physical_counterfactual", "experiment_id": experiment_id,
                        "partition": partition, "treatment": "counterfactual_text", "control": "baseline", "role": "physical_counterfactual_control"})
    spec = {
        "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": partition,
        "hypothesis": "States at matched wall/key/door input-fact tokens route task information to the native action; compare the same donor at source-fact and final prompt-token sites.",
        "claim_scope": "input_fact_information_routing; not an abstract-belief or CoT localization claim",
        "jobs": jobs, "comparisons": comparisons, "accepted_pairs": accepted, "exclusions": exclusions,
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "n_source_pairs": len(pairs), "n_aligned_pairs": len(accepted), "n_excluded_pairs": len(exclusions),
        "selection_before_model_query": True, "source_pairs_sha256": fingerprint(list(pairs)),
        "tokenizer": {"class": type(tokenizer).__name__, "name_or_path": str(getattr(tokenizer, "name_or_path", "unspecified")),
                      "add_special_tokens": False, "context_renderer": "campaign.render_analysis_prompt"},
    }
    if partition == "confirmation":
        spec.update(freeze_id=freeze_id, semantic_headline_eligible=False,
                    claim_scope="input_fact_information_routing")
        for comparison in comparisons:
            comparison.update(predicted_direction="zero" if comparison["role"] == "technical_identity" else "increase",
                              semantic_headline_eligible=False, claim_scope="input_fact_information_routing")
    return spec
