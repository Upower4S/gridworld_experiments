"""Optional frozen co-primary registry and analysis, without model or file IO.

This validates an authoritative supplied registry; it does not certify the
selection review, source artifacts, or unqueried confirmation specification.
Those independent freeze/readiness checks remain the caller's responsibility.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping

ENDPOINT = "natural_target_action_intention_to_treat"
UNRESOLVED_POLICY = "count as zero target success; retain in intention-to-treat denominator; no forced fallback"
FAMILIES = ("wall", "key_possession", "door_status")
REQUIRED_BINDING_FIELDS = {
    "treatment", "control", "predicted_direction", "primary_endpoint", "target_action_basis",
    "native_unresolved_policy", "expected_n_layouts", "expected_n_pairs", "eligibility_rule",
    "origin", "claim_scope", "semantic_headline_eligible",
}
MIXTURE_FIELDS = {
    "families", "family_layout_counts", "family_weights", "resampling_scheme", "stratification_key",
    "expected_n_layouts_by_family", "expected_n_pairs_by_family",
}
PENDING_BLOCKERS = {
    "no_complete_execution_pairs", "missing_execution_rows", "frozen_layout_count_not_met",
    "frozen_pair_count_not_met", "fewer_than_32_confirmation_layouts", "frozen_family_counts_not_met",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> str:
    def keys(item):
        if isinstance(item, Mapping):
            _require(all(isinstance(key, str) for key in item), "primary family JSON keys must be strings")
            for child in item.values():
                keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                keys(child)

    keys(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("primary family must contain finite JSON values") from exc


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _binding(binding: Any) -> None:
    _require(isinstance(binding, dict) and REQUIRED_BINDING_FIELDS <= binding.keys(),
             "primary binding is missing required fields")
    for key in ("treatment", "control", "claim_scope"):
        _require(_name(binding[key]), "primary binding requires a nonempty " + key)
    _require(binding["treatment"] != binding["control"], "primary treatment and control must differ")
    for key, expected in {
        "predicted_direction": "increase", "primary_endpoint": ENDPOINT,
        "target_action_basis": "unique_simulator_counterfactual_optimum",
        "native_unresolved_policy": UNRESOLVED_POLICY,
    }.items():
        _require(binding[key] == expected, "unsupported frozen " + key)
    _require(type(binding["semantic_headline_eligible"]) is bool, "semantic eligibility must be Boolean")
    n, pairs = binding["expected_n_layouts"], binding["expected_n_pairs"]
    _require(type(n) is int and n >= 32 and type(pairs) is int and pairs >= n,
             "primary binding requires at least 32 layouts and matching positive pair counts")
    rule = binding["eligibility_rule"]
    _require(isinstance(rule, dict) and set(rule) == {"identifier", "sha256"}
             and _name(rule["identifier"]) and isinstance(rule["sha256"], str)
             and re.fullmatch("[0-9a-f]{64}", rule["sha256"]) is not None,
             "eligibility_rule requires identifier and SHA256")
    origin = binding["origin"]
    _require(isinstance(origin, dict) and set(origin) == {"kind", "provenance"}
             and isinstance(origin["kind"], str)
             and origin["kind"] in {"existing_source_comparison", "prospective_confirmation_control"}
             and isinstance(origin["provenance"], dict) and bool(origin["provenance"]),
             "primary binding requires an explicitly supported origin and provenance")
    if MIXTURE_FIELDS & binding.keys():
        _require(MIXTURE_FIELDS <= binding.keys(), "mixture binding must include every stratification field")
        _require(binding["families"] == list(FAMILIES), "mixture families must use the frozen canonical order")
        counts, pair_counts, weights = (binding[key] for key in
                                        ("family_layout_counts", "expected_n_pairs_by_family", "family_weights"))
        _require(all(isinstance(value, dict) and set(value) == set(FAMILIES)
                     for value in (counts, pair_counts, weights)), "mixture family inventories differ")
        _require(all(type(value) is int and value > 0 for value in counts.values())
                 and sum(counts.values()) == n and max(counts.values()) - min(counts.values()) <= 1
                 and binding["expected_n_layouts_by_family"] == counts,
                 "mixture layout quotas differ from expected counts")
        _require(all(type(pair_counts[f]) is int and pair_counts[f] >= counts[f] for f in FAMILIES)
                 and sum(pair_counts.values()) == pairs, "mixture pair quotas differ from expected counts")
        _require(all(type(weights[f]) in {int, float} and math.isfinite(weights[f])
                     and abs(weights[f] - counts[f] / n) <= 1e-12 for f in FAMILIES),
                 "mixture weights differ from frozen layout quotas")
        _require(binding["resampling_scheme"] == "within_family_layout_bootstrap"
                 and binding["stratification_key"] == "family", "unsupported mixture resampling")


def validate_primary_family(value: Any) -> dict[str, Any]:
    """Return a JSON copy of the complete immutable registry; no slots inferred."""
    _require(isinstance(value, Mapping), "confirmation_primary_family must be a mapping")
    registry = json.loads(canonical(dict(value)))
    _require(set(registry) == {"schema_version", "candidates", "family_size", "bindings"}
             and type(registry["schema_version"]) is int and registry["schema_version"] == 1,
             "primary registry requires schema_version1, candidates, family_size and bindings")
    candidates, bindings = registry["candidates"], registry["bindings"]
    _require(isinstance(candidates, dict) and 1 <= len(candidates) <= 2
             and all(_name(key) for key in candidates), "freeze one or two candidate IDs")
    _require(isinstance(bindings, dict) and set(bindings) == set(candidates),
             "primary binding candidate inventory differs")
    size = 0
    for candidate, identifiers in candidates.items():
        _require(isinstance(identifiers, list) and 1 <= len(identifiers) <= 2
                 and all(_name(identifier) for identifier in identifiers)
                 and len(set(identifiers)) == len(identifiers),
                 "freeze one or two distinct named primaries per candidate")
        _require(isinstance(bindings[candidate], dict) and set(bindings[candidate]) == set(identifiers),
                 "primary binding inventory differs from registered pairs")
        for identifier in identifiers:
            _binding(bindings[candidate][identifier])
        size += len(identifiers)
    _require(type(registry["family_size"]) is int and registry["family_size"] == size <= 4,
             "family_size must equal all frozen candidate/primary pairs")
    return registry


def _component_state(component: dict[str, Any], binding: dict[str, Any]) -> str:
    blockers = set(component["validity_blockers"])
    overflow = (component["n_layouts"] > binding["expected_n_layouts"]
                or component["n_execution_pairs"] > binding["expected_n_pairs"])
    if blockers - PENDING_BLOCKERS or overflow:
        return "invalid"
    if blockers or component["n_pairs"] != binding["expected_n_pairs"]:
        return "pending"
    return "complete"


def _pending(candidate: str, primary: str, binding: dict[str, Any]) -> dict[str, Any]:
    return {"candidate_id": candidate, "primary_id": primary, "comparison_id": None,
            "binding": deepcopy(binding), "component_status": "pending", "statistical_verdict": "pending",
            "primary": {"effect": None, "ci": None, "adjusted_ci": None, "p_value": None,
                        "holm_p_value": 1.0, "multiplicity_input_p_value": 1.0},
            "missing_comparison": True, "native_observations_complete": False,
            "validity_blockers": ["missing_registered_comparison"],
            "eventual_action_robust_to_censoring": False, "headline_semantic_support": False}


def analyze_primary_family(rows, *, comparisons, confirmation_primary_family,
                           confirmation_family_size, **options) -> dict[str, Any]:
    """Reuse existing component statistics, reserving every frozen primary slot."""
    from . import analysis as a

    _require(confirmation_family_size is None,
             "confirmation_primary_family forbids a separate family-size override")
    registry = validate_primary_family(confirmation_primary_family)
    # The empty legacy result supplies the unchanged option validation/config
    # schema without performing a test, inferring comparisons or opening data.
    summary = a.analyze_results([], comparisons=[], **options)
    specs = [dict(spec) for spec in comparisons] if comparisons is not None else []
    assigned = {}
    descriptive = []
    for spec in specs:
        candidate, primary = spec.get("candidate_id"), spec.get("primary_id")
        if primary is None:
            _require(spec.get("confirmatory") is False,
                     "unregistered comparisons must be explicitly descriptive without a primary ID")
            descriptive.append(spec)
            continue
        _require(_name(candidate) and candidate in registry["candidates"]
                 and _name(primary) and primary in registry["candidates"][candidate],
                 "unknown frozen candidate/primary pair")
        key = candidate, primary
        _require(key not in assigned, "duplicate frozen candidate/primary pair")
        _require(spec.get("partition") == "confirmation" and spec.get("confirmatory") is True
                 and not spec.get("inferred", False), "registered primary cannot downgrade its confirmation status")
        binding = registry["bindings"][candidate][primary]
        _require(all(field in spec and canonical(spec[field]) == canonical(expected)
                     for field, expected in binding.items()), "comparison contradicts its exact frozen binding")
        _require(_name(spec.get("experiment_id")) and _name(spec.get("id")),
                 "supplied primary requires its actual experiment and comparison IDs")
        assigned[key] = spec
    records = a.load_results(rows)
    a._validate(records)
    m = registry["family_size"]
    common = {key: options[key] for key in ("n_bootstrap", "n_permutations", "alpha", "seed")}
    components, actual = [], []
    for candidate, primaries in registry["candidates"].items():
        for primary in primaries:
            binding = registry["bindings"][candidate][primary]
            spec = assigned.get((candidate, primary))
            if spec is None:
                components.append(_pending(candidate, primary, binding))
                continue
            result = a._comparison(records, spec, family_size=m, run_inference=False, **common)
            state = _component_state(result, binding)
            if state == "complete":
                result = a._comparison(records, spec, family_size=m, **common)
            else:
                # Keep counts/errors, without inventing inferential quantities
                # from a missing, partial, or technically invalid component.
                result["primary"].update({field: None for field in result["primary"]
                                           if field in {"treatment_rate", "control_rate", "effect", "ci",
                                                        "adjusted_ci", "p_value", "holm_p_value"}})
                result["completion_sensitive_effect_bounds"] = {
                    "lower": None, "upper": None, "adjusted_uncertainty_envelope": None}
            result.update(candidate_id=candidate, primary_id=primary, comparison_id=result["id"],
                          display_id=f"{candidate}/{primary} [{result['id']}]", binding=deepcopy(binding),
                          component_status=state, missing_comparison=False,
                          native_observations_complete=result["n_pairs"] == binding["expected_n_pairs"]
                          and result["n_layouts"] == binding["expected_n_layouts"])
            components.append(result)
            actual.append(result)
    p_values = [component["primary"]["p_value"] if component["component_status"] == "complete"
                and component["primary"]["p_value"] is not None else 1.0 for component in components]
    adjusted = a.holm_adjust(p_values)
    for result, p_value, adjusted_p in zip(components, p_values, adjusted, strict=True):
        result["primary"].update(holm_p_value=adjusted_p, multiplicity_input_p_value=p_value)
        if result["missing_comparison"]:
            continue
        a._interpret_comparison(result, alpha=options["alpha"], minimum_effect=options["minimum_effect"])
        if result["component_status"] != "complete":
            pending = result["component_status"] == "pending"
            result.update(statistical_verdict="pending" if pending else "technically_invalid",
                          report_label="pending" if pending else "technically_invalid",
                          headline_semantic_support=False, eventual_action_robust_to_censoring=False,
                          claim="The frozen component is incomplete; confirmatory inference is pending."
                          if pending else "The frozen component is technically invalid; inference is blocked.")
    for spec in descriptive:
        result = a._comparison(records, spec, family_size=m, **common)
        a._interpret_comparison(result, alpha=options["alpha"], minimum_effect=options["minimum_effect"])
        actual.append(result)
    conjunctions = []
    for candidate, primaries in registry["candidates"].items():
        group = [c for c in components if c["candidate_id"] == candidate]
        invalid = [c["primary_id"] for c in group if c["component_status"] == "invalid"]
        pending = [c["primary_id"] for c in group if c["component_status"] == "pending"]
        supported = all(c["component_status"] == "complete" and c["statistical_verdict"] == "supported" for c in group)
        status = "invalid" if invalid else "pending" if pending else "supported" if supported else "not_supported"
        conjunctions.append({"candidate_id": candidate, "primary_ids": primaries.copy(), "status": status,
                             "supported": status == "supported", "invalid_primary_ids": invalid,
                             "pending_primary_ids": pending,
                             "component_verdicts": {c["primary_id"]: c["statistical_verdict"] for c in group},
                             "eventual_action_robust_to_censoring": supported and all(c["eventual_action_robust_to_censoring"] for c in group),
                             "headline_semantic_support": supported and all(c["headline_semantic_support"] for c in group),
                             "claim": "Every frozen component meets the within-budget effect criterion; stronger claims retain their separate gates."
                             if supported else "The frozen conjunction is not established; inspect each component and its validity separately."})
    pending_keys = [[c["candidate_id"], c["primary_id"]] for c in components if c["component_status"] == "pending"]
    invalid_keys = [[c["candidate_id"], c["primary_id"]] for c in components if c["component_status"] == "invalid"]
    summary.update(schema_version="activation_causality_analysis_v2", n_rows=len(records),
                   n_layouts=len({str(row["layout_id"]) for row in records}), conditions=a._descriptive(records),
                   comparisons=actual, primary_components=components, candidate_conjunctions=conjunctions,
                   confirmation_primary_family=registry,
                   confirmation_primary_family_sha256=hashlib.sha256(canonical(registry).encode()).hexdigest(),
                   missing_or_pending_primary_ids=pending_keys, invalid_primary_ids=invalid_keys,
                   all_primary_components_final=not pending_keys,
                   all_primary_components_have_native_observations=all(c["native_observations_complete"] for c in components),
                   all_primary_components_complete_valid=not pending_keys and not invalid_keys,
                   multiplicity_adjustment_provisional=bool(pending_keys))
    summary["analysis_config"].update(version=4, confirmation_family_size=m,
                                      confirmation_primary_family_size=m,
                                      candidate_decision_rule="all_frozen_primary_components_supported",
                                      component_state_precedence=["invalid", "pending", "complete"])
    summary["cautions"] = [*summary["cautions"],
        "All frozen primary components retain multiplicity slots, including pending and invalid components; candidate support requires every component.",
        "The supplied primary registry is authoritative input, not proof of approved selection or readiness; source/control/basis checks remain external requirements."]
    return summary
