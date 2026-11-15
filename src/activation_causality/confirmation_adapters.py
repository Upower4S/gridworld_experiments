"""Pure adapters for externally frozen confirmation specifications.

This module does not load a tokenizer or model, read or write files, select a
candidate, or certify an external freeze. The caller supplies a local tokenizer
and an already frozen configuration. Tokenization and simulator eligibility may
exclude candidates before inference; actual eligible counts are never replaced
with the requested cohort size. A partial return is a sampler planning artifact,
not permission to execute an underfilled confirmation cohort.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .input_sites import FAMILIES, SITES, build_input_site_spec
from .registry import fingerprint


INPUT_ALIGNMENT_RULE = "exact_equal_length_common_prefix_and_suffix"
INPUT_CLAIM_SCOPE = "input_fact_information_routing"
NATIVE_ITT_ENDPOINT = "natural_target_action_intention_to_treat"
MIXED_INPUT_FAMILIES = ("wall", "key_possession", "door_status")


def _frozen_input_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize compatible freeze aliases, rejecting unsupported alternatives."""
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration must be a mapping")
    config = deepcopy(dict(configuration))
    layer = config.get("layer")
    layers = config.get("layers")
    if layers is not None:
        if not isinstance(layers, (list, tuple)) or len(layers) != 1:
            raise ValueError("freeze exactly one input-fact layer")
        if layer is not None and layer != layers[0]:
            raise ValueError("layer and layers contradict one another")
        layer = layers[0]
    if type(layer) is not int or not 0 <= layer < 24:
        raise ValueError("frozen layer must be a GPT-OSS block index 0..23")
    site = config.get("site")
    if site not in SITES:
        raise ValueError("freeze one supported intervention site")
    seeds = config.get("generation_seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError("freeze distinct uint32 generation_seeds")
    cap = config.get("max_new_tokens")
    if type(cap) is not int or not 1 <= cap <= 8192:
        raise ValueError("freeze max_new_tokens in 1..8192")
    count = config.get("n_layouts")
    if type(count) is not int or count not in {32, 64, 128}:
        raise ValueError("freeze 32, 64 or 128 independent layouts")
    treatment = config.get("treatment_condition")
    control = config.get("control_condition")
    if treatment != f"factspan_layer{layer}" or control not in {
        "baseline", f"random_factspan_layer{layer}", f"lasttoken_layer{layer}"
    }:
        raise ValueError("primary must compare frozen factspan transfer with baseline, random, or lasttoken")
    inventory = sorted(["baseline", "counterfactual_text", f"identity_layer{layer}",
                        f"factspan_layer{layer}", f"random_factspan_layer{layer}", f"lasttoken_layer{layer}"])
    fixed = {
        "protocol_kind": "input_fact", "natural": True, "temperature": 0.7,
        "operation": "replace", "dose": 1.0, "predicted_direction": "increase",
        "claim_scope": INPUT_CLAIM_SCOPE, "semantic_headline_eligible": False,
        "primary_endpoint": NATIVE_ITT_ENDPOINT,
        "input_alignment_rule": INPUT_ALIGNMENT_RULE,
        "condition_inventory": inventory, "conditions_per_layout_upper_bound": 6,
    }
    for name, expected in fixed.items():
        if name in config and (config[name] != expected or
                               isinstance(expected, bool) and type(config[name]) is not bool):
            raise ValueError(f"unsupported frozen {name}")
    intervention = config.get("intervention")
    expected_intervention = {"layers": [layer], "site": site, "operation": "replace", "dose": 1.0,
                             "random_control": False}
    if intervention is not None:
        if not isinstance(intervention, Mapping) or any(
            key not in expected_intervention or expected_intervention[key] != value
            for key, value in intervention.items()
        ) or any(intervention.get(key) != expected_intervention[key]
                 for key in ("layers", "site", "operation", "dose")):
            raise ValueError("frozen intervention must be the selected full-replacement input-fact treatment")
    selection = config.get("selection")
    expected_selection = {"scope": "explicit", "alignment": INPUT_ALIGNMENT_RULE}
    if selection is not None and selection != expected_selection:
        raise ValueError("freeze the exact input alignment rule, not source-pair token positions")
    if config.get("family") is not None and config["family"] not in FAMILIES:
        raise ValueError("unsupported frozen fact family")
    if "families" in config:
        if "family" in config:
            raise ValueError("mixed families cannot be combined with a scalar family")
        families = config["families"]
        if not isinstance(families, (list, tuple)) or tuple(families) != MIXED_INPUT_FAMILIES:
            raise ValueError("mixed input confirmation requires canonical families order: wall, key_possession, door_status")
        quotas = config.get("family_layout_counts")
        if (not isinstance(quotas, Mapping) or set(quotas) != set(MIXED_INPUT_FAMILIES)
                or any(type(value) is not int or value <= 0 for value in quotas.values())
                or sum(quotas.values()) != count or max(quotas.values()) - min(quotas.values()) > 1):
            raise ValueError("family_layout_counts must be positive balanced quotas summing to n_layouts")
        quotas = {family: quotas[family] for family in MIXED_INPUT_FAMILIES}
        weights = {family: quotas[family] / count for family in MIXED_INPUT_FAMILIES}
        supplied_weights = config.get("family_weights")
        if supplied_weights is not None and (
            not isinstance(supplied_weights, Mapping) or set(supplied_weights) != set(weights)
            or any(type(value) not in (int, float) for value in supplied_weights.values())
            or any(supplied_weights[family] != weights[family] for family in weights)
        ):
            raise ValueError("family_weights must equal frozen family_layout_counts / n_layouts")
        for name, expected in (("resampling_scheme", "within_family_layout_bootstrap"), ("stratification_key", "family")):
            if name in config and config[name] != expected:
                raise ValueError(f"mixed input confirmation requires {name}={expected}")
        config.update(families=list(MIXED_INPUT_FAMILIES), family_layout_counts=quotas, family_weights=weights,
                      resampling_scheme="within_family_layout_bootstrap", stratification_key="family")
    elif "family_layout_counts" in config or "family_weights" in config:
        raise ValueError("family quotas and weights require explicit mixed families")
    # Aliases sometimes carried by callers may not silently disagree with the
    # canonical contract. These do not change the source-derived six controls.
    for alias, expected in (("seeds", list(seeds)), ("cap", cap), ("layout_count", count),
                            ("primary_treatment", treatment), ("primary_control", control),
                            ("conditions_per_layout", 6)):
        if alias in config and config[alias] != expected:
            raise ValueError(f"frozen {alias} contradicts the canonical configuration")
    config.update(fixed)
    config.update(layer=layer, layers=[layer], site=site, generation_seeds=list(seeds),
                  max_new_tokens=cap, n_layouts=count, intervention=expected_intervention,
                  selection=expected_selection)
    return config


def build_frozen_input_spec(
    pairs: Sequence[Mapping[str, Any]], tokenizer: Any, experiment_id: str,
    configuration: Mapping[str, Any], freeze_id: str,
) -> dict[str, Any]:
    """Build one frozen source-fact confirmation, including pre-query exclusions.

    Required configuration keys are ``layer`` (or one-element ``layers``),
    ``site``, ``generation_seeds``, ``max_new_tokens``, ``n_layouts`` and the
    ``treatment_condition``/``control_condition`` pair. The only primary treatment
    is factspan_layerL. Allowed controls are baseline, random_factspan_layerL or
    lasttoken_layerL. Every eligible pair schedules all six source controls.

    A supplied ``family`` excludes other families before tokenization. Optional
    mixed ``families`` must list wall, key_possession, door_status in exactly
    that order, with positive
    balanced ``family_layout_counts`` summing to the requested total. Their
    frozen population weights are quotas divided by that total; exclusions do
    not change those weights. Over-quota eligible family cohorts are rejected.
    Inputs
    must already be unqueried confirmation pairs in their deterministic layout
    partition; neither physical states nor partition labels are rewritten to
    gain eligibility. The target is the simulator's unique physical-counterfactual
    optimum, including for controls. Unresolved native generations count as zero
    target success in the primary intention-to-treat endpoint.

    ``expected_n_layouts`` and ``expected_n_pairs`` always reflect actual eligible
    jobs. ``requested_n_layouts``, ``layout_shortfall`` and ``cohort_complete``
    expose the fill requirement to an external pre-query sampler/ready check.
    Rebuild from job.metadata.source_pair, which preserves the unnormalized pair.
    """
    config = _frozen_input_configuration(configuration)
    mixed = "families" in config
    sources = deepcopy(list(pairs))
    selected, index_map, family_exclusions = [], [], []
    for index, pair in enumerate(sources):
        if not isinstance(pair, Mapping):
            raise ValueError("each source pair must be a mapping")
        if ((config.get("family") is not None and pair.get("family") != config["family"])
                or (mixed and pair.get("family") not in config["families"])):
            family_exclusions.append({"input_index": index, **{k: pair.get(k) for k in ("pair_id", "layout_id", "family")},
                                      "reason": "frozen_family_mismatch", "excluded_before_model_query": True})
        else:
            index_map.append(index)
            selected.append(pair)
    spec = build_input_site_spec(
        selected, tokenizer, experiment_id, layers=config["layers"], site=config["site"],
        seeds=config["generation_seeds"], cap=config["max_new_tokens"], natural=True,
        partition="confirmation", freeze_id=freeze_id,
    )
    if "tokenizer" in config and config["tokenizer"] != spec["tokenizer"]:
        raise ValueError("supplied tokenizer differs from the frozen tokenizer metadata")
    for entry in spec["exclusions"]:
        entry["input_index"] = index_map[entry["input_index"]]
    spec["exclusions"] = sorted(spec["exclusions"] + family_exclusions, key=lambda row: row["input_index"])
    spec["exclusion_counts"] = dict(sorted(Counter(row["reason"] for row in spec["exclusions"]).items()))
    spec["n_source_pairs"] = len(sources)
    spec["n_excluded_pairs"] = len(spec["exclusions"])
    spec["source_pairs_sha256"] = fingerprint(sources)
    by_pair = {entry["pair_id"]: entry for entry in spec["accepted_pairs"]}
    accepted_ids = sorted(by_pair)
    n_layouts = len({entry["layout_id"] for entry in by_pair.values()})
    n_observations = len(accepted_ids) * len(config["generation_seeds"])
    family_layouts, family_observations = {}, {}
    if mixed:
        family_layouts = {family: len({entry["layout_id"] for entry in by_pair.values() if entry["family"] == family})
                          for family in config["families"]}
        family_observations = {family: sum(entry["family"] == family for entry in by_pair.values()) * len(config["generation_seeds"])
                               for family in config["families"]}
        excess = {family: family_layouts[family] - config["family_layout_counts"][family]
                  for family in config["families"] if family_layouts[family] > config["family_layout_counts"][family]}
        if excess:
            raise ValueError(f"eligible mixed-family cohort exceeds frozen family quotas: {excess}")
    layout_aliases: dict[str, list[str]] = {}
    for job in spec["jobs"]:
        if job["partition"] != "confirmation" or job["metadata"]["pair"]["partition"] != "confirmation":
            raise AssertionError("confirmation adapter emitted a non-confirmation job")
        layout_aliases[job["layout_id"]] = job["layout_aliases"]
    primary = []
    for comparison in spec["comparisons"]:
        chosen = (comparison["treatment"] == config["treatment_condition"] and
                  comparison["control"] == config["control_condition"])
        comparison.update(
            confirmatory=chosen, primary_endpoint=NATIVE_ITT_ENDPOINT,
            primary_estimand="layout-weighted intention-to-treat paired native target-action probability difference",
            target_action_basis="unique_simulator_counterfactual_optimum",
            expected_n_layouts=n_layouts, expected_n_pairs=n_observations,
            eligible_pair_ids=accepted_ids.copy(), eligibility_basis="frozen_simulator_and_tokenizer_only_before_model_query",
            outcome_dependent_exclusion_permitted=False, independent_rescue_evidence=False,
            semantic_headline_eligible=False, claim_scope=INPUT_CLAIM_SCOPE,
        )
        if mixed:
            comparison.update(
                resampling_scheme="within_family_layout_bootstrap", stratification_key="family",
                family_layout_counts=deepcopy(config["family_layout_counts"]), family_weights=deepcopy(config["family_weights"]),
                expected_n_layouts_by_family=family_layouts.copy(), expected_n_pairs_by_family=family_observations.copy(),
            )
        if chosen:
            primary.append(comparison)
    if len(primary) != 1:
        raise AssertionError("exactly one frozen primary contrast must exist")
    excluded_by_index = {e["input_index"]: e for e in spec["exclusions"]}
    manifest = []
    for index, pair in enumerate(sources):
        exclusion = excluded_by_index.get(index)
        manifest.append({
            "input_index": index, "pair_id": pair.get("pair_id"), "layout_id": pair.get("layout_id"),
            "partition": pair.get("partition"), "eligible": exclusion is None,
            "scheduled_conditions": config["condition_inventory"].copy() if exclusion is None else [],
            "selection_basis": "simulator_and_tokenizer_only_before_model_query",
            "exclusion_reason": exclusion["reason"] if exclusion else None,
            **({"family": pair.get("family")} if mixed else {}),
        })
    spec.update(
        adapter_version=1, protocol_kind="input_fact", frozen_configuration=config,
        configuration_sha256=fingerprint(config), freeze_id=freeze_id,
        candidate_id=config.get("candidate_id"), source_experiment_id=config.get("source_experiment_id"),
        claim_scope=INPUT_CLAIM_SCOPE, semantic_headline_eligible=False,
        semantic_headline_ineligibility_reason="Source-fact token transfer can test input information routing; it cannot establish an abstract semantic belief representation.",
        primary_endpoint=NATIVE_ITT_ENDPOINT,
        primary_estimand="layout-weighted intention-to-treat paired native target-action probability difference",
        primary_comparison_id=primary[0]["id"], target_action_basis="unique_simulator_counterfactual_optimum",
        secondary_estimands=["forced four-action target probability", "native unresolved fraction"],
        native_unresolved_policy="count as zero target success; retain in intention-to-treat denominator; no forced fallback",
        expected_n_layouts=n_layouts, expected_n_pairs=n_observations,
        requested_n_layouts=config["n_layouts"], layout_shortfall=max(0, config["n_layouts"] - n_layouts),
        layout_excess=max(0, n_layouts - config["n_layouts"]), cohort_complete=n_layouts == config["n_layouts"],
        eligible_pair_ids=accepted_ids, eligibility_manifest=manifest,
        layout_aliases={key: value for key, value in sorted(layout_aliases.items())},
        protocol={"layers": config["layers"], "site": config["site"], "seeds": config["generation_seeds"],
                  "max_new_tokens": config["max_new_tokens"], "natural": True, "temperature": 0.7,
                  "input_alignment_rule": INPUT_ALIGNMENT_RULE, "condition_inventory": config["condition_inventory"],
                  "selection_basis": "simulator_and_tokenizer_only_before_model_query",
                  "confirmation_requires_external_freeze": True, "confirmation_requires_external_ready_check": True,
                  "outcome_dependent_exclusion_permitted": False, "independent_rescue_evidence": False,
                  "semantic_headline_eligible": False},
    )
    if mixed:
        shortfall = {family: config["family_layout_counts"][family] - family_layouts[family] for family in config["families"]}
        spec.update(
            families=config["families"].copy(), requested_n_layouts_by_family=deepcopy(config["family_layout_counts"]),
            expected_n_layouts_by_family=family_layouts.copy(), expected_n_pairs_by_family=family_observations.copy(),
            family_weights=deepcopy(config["family_weights"]), layout_shortfall_by_family=shortfall,
            cohort_complete=n_layouts == config["n_layouts"] and all(value == 0 for value in shortfall.values()),
            resampling_scheme="within_family_layout_bootstrap", stratification_key="family",
        )
        spec["protocol"].update(
            families=config["families"].copy(), family_layout_counts=deepcopy(config["family_layout_counts"]),
            family_weights=deepcopy(config["family_weights"]),
            resampling_scheme="within_family_layout_bootstrap", stratification_key="family",
            family_quotas_must_be_exact_before_execution=True,
        )
    return spec
