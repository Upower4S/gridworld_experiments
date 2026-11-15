"""Pure contracts shared by prospective freeze, readiness and reporting.

This module does not approve candidates, read outcomes, or authorize a model
query. Source authenticity, exact adapter reconstruction, review and full model
identity remain separate mandatory checks in the caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Mapping

from .primary_family import validate_primary_family
from .registry import fingerprint

FAMILIES = ("wall", "key_possession", "door_status")
PROTOCOLS = {
    "contextual_input": ("contextual_input_fact_interchange", 15,
                         ("contextual_fact_contrast", "natural_norm_direction_contrast")),
    "input_fact_subspace": ("input_fact_subspace_routing", 14,
                            ("subspace_efficacy", "subspace_direction_contrast")),
}
ENDPOINT = "natural_target_action_intention_to_treat"
UNRESOLVED_POLICY = "count as zero target success; retain in intention-to-treat denominator; no forced fallback"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def family_quotas(n_layouts: int) -> dict[str, int]:
    _require(type(n_layouts) is int and n_layouts in {32, 64, 128}, "unsupported frozen cohort size")
    return {family: n_layouts // 3 + (index < n_layouts % 3)
            for index, family in enumerate(FAMILIES)}


def validate_registry_configurations(registry: Mapping[str, Any],
                                    configurations: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact frozen co-primary bindings and one profile per question."""
    family = validate_primary_family(registry)
    _require(isinstance(configurations, Mapping) and set(configurations) == set(family["candidates"]),
             "candidate configurations and primary registry must have the same identities")
    protocols, sizes = [], set()
    for candidate_id, config in configurations.items():
        _require(isinstance(config, Mapping) and config.get("candidate_id") == candidate_id,
                 "candidate identity differs from its configuration key")
        protocol = config.get("protocol_kind")
        _require(protocol in PROTOCOLS, "no approved prospective adapter for candidate protocol")
        protocols.append(protocol)
        scope, n_conditions, primary_ids = PROTOCOLS[protocol]
        _require(family["candidates"][candidate_id] == list(primary_ids),
                 "each question requires its exact ordered two co-primaries")
        _require(fingerprint(config.get("primary_comparisons")) == fingerprint(family["bindings"][candidate_id]),
                 "candidate primary bindings disagree with the frozen registry")
        _require(not {"family_size", "confirmation_family_size", "confirmation_primary_family_size"} & config.keys(),
                 "candidate cannot override global multiplicity")
        n = config.get("n_layouts")
        _require(type(n) is int and n in {32, 64, 128}, "freeze 32, 64 or 128 independent layouts")
        sizes.add(n)
        _require(config.get("generation_seeds") == [42]
                 and all(type(seed) is int for seed in config["generation_seeds"]), "freeze seed42")
        _require(type(config.get("max_new_tokens")) is int and config["max_new_tokens"] == 4096
                 and type(config.get("temperature")) in {int, float} and config["temperature"] == 0.7,
                 "freeze cap4096 and temperature0.7")
        _require(config.get("claim_scope") == scope and config.get("semantic_headline_eligible") is False,
                 "candidate may not promote the input-routing claim scope")
        inventory = config.get("condition_inventory")
        _require(isinstance(inventory, list) and len(inventory) == len(set(inventory)) == n_conditions
                 and all(isinstance(value, str) and value for value in inventory),
                 "candidate must retain its whole control inventory")
        if protocol == "contextual_input":
            counts = family_quotas(n)
            _require("family" not in config and config.get("families") == list(FAMILIES)
                     and config.get("family_layout_counts") == counts
                     and config.get("family_weights") == {key: count / n for key, count in counts.items()},
                     "contextual candidate must retain the fixed balanced family mixture")
        else:
            _require(config.get("family") in FAMILIES
                     and not {"families", "family_layout_counts", "family_weights"} & config.keys(),
                     "subspace candidate must have one fixed family")
        for binding in family["bindings"][candidate_id].values():
            _require(binding["expected_n_layouts"] == n and binding["expected_n_pairs"] == n
                     and binding["claim_scope"] == scope and binding["semantic_headline_eligible"] is False,
                     "primary scope or complete native denominator differs from candidate")
            _require(binding["treatment"] in inventory and binding["control"] in inventory,
                     "primary references a condition outside its complete inventory")
            if protocol == "contextual_input":
                _require(binding.get("families") == config["families"]
                         and binding.get("family_layout_counts") == counts
                         and binding.get("family_weights") == config["family_weights"]
                         and binding.get("expected_n_layouts_by_family") == counts
                         and binding.get("expected_n_pairs_by_family") == counts,
                         "primary mixture differs from the frozen candidate")
            else:
                _require(not {"families", "family_weights", "family_layout_counts"} & binding.keys(),
                         "single-family primary cannot introduce a mixture")
    _require(len(protocols) == len(set(protocols)), "only one candidate per scientific question is permitted")
    _require(len(sizes) == 1, "selected candidates must use the same prospectively budgeted layout count")
    return family


def validate_primary_spec(spec: Mapping[str, Any], config: Mapping[str, Any],
                          registry: Mapping[str, Any]) -> None:
    """Check the complete primary cohort, before reading any outcome.

    Does not replace byte-for-byte reconstruction by the designated adapter.
    In particular, simulator states, tokenizer spans, basis/source artifacts and
    fresh-layout exclusions must be verified by that independent outer gate.
    """
    family = validate_primary_family(registry)
    candidate_id = config["candidate_id"]
    _require(candidate_id in family["candidates"]
             and fingerprint(config["primary_comparisons"]) == fingerprint(family["bindings"][candidate_id]),
             "specification candidate bindings disagree with the primary registry")
    _require(spec.get("partition") == "confirmation" and spec.get("candidate_id") == candidate_id
             and spec.get("protocol_kind") == config["protocol_kind"]
             and spec.get("cohort_role", "primary") == "primary",
             "primary specification role, partition or candidate differs")
    _require(spec.get("claim_scope") == config["claim_scope"] and spec.get("semantic_headline_eligible") is False,
             "primary specification changes the frozen claim scope")
    jobs, comparisons = spec.get("jobs"), spec.get("comparisons")
    _require(isinstance(jobs, list) and bool(jobs) and isinstance(comparisons, list) and bool(comparisons),
             "primary cohort requires execution jobs and comparisons")
    _require(len({row["job_id"] for row in jobs}) == len(jobs), "duplicate primary execution identity")
    inventory = set(config["condition_inventory"])
    per_pair, pair_layout, layout_pair, pair_family = defaultdict(Counter), {}, {}, {}
    for job in jobs:
        _require(job.get("job_id") == fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24],
                 "noncanonical frozen execution identity")
        _require(job.get("experiment_id") == spec.get("experiment_id") and job.get("partition") == "confirmation"
                 and job.get("natural") is True and type(job.get("seed")) is int and job["seed"] == 42
                 and type(job.get("max_new_tokens")) is int and job["max_new_tokens"] == 4096
                 and type(job.get("temperature")) in {int, float} and job["temperature"] == 0.7,
                 "primary execution differs from the frozen native protocol")
        _require(job.get("semantic_headline_eligible") is False and job.get("claim_scope") == config["claim_scope"],
                 "job promotes the frozen claim scope")
        pair, layout, variable = job.get("pair_id"), job.get("layout_id"), job.get("family")
        _require(all(isinstance(value, str) and value for value in (pair, layout, variable)),
                 "primary job requires canonical pair, layout and family identities")
        _require(pair_layout.get(pair, layout) == layout and layout_pair.get(layout, pair) == pair
                 and pair_family.get(pair, variable) == variable, "primary layout/pair/family mapping is inconsistent")
        pair_layout[pair], layout_pair[layout], pair_family[pair] = layout, pair, variable
        per_pair[pair][job.get("condition")] += 1
    n = config["n_layouts"]
    _require(len(pair_layout) == len(layout_pair) == n, "primary cohort does not have every frozen layout")
    _require(all(counts == Counter({condition: 1 for condition in inventory}) for counts in per_pair.values()),
             "every primary layout must retain each planned condition exactly once")
    actual_family_counts = Counter(pair_family.values())
    expected_family_counts = config.get("family_layout_counts", {config.get("family"): n})
    _require(actual_family_counts == Counter(expected_family_counts), "primary cohort family quotas differ")
    _require(len({row["id"] for row in comparisons}) == len(comparisons), "duplicate comparison identity")
    primary_ids = []
    for comparison in comparisons:
        _require(comparison.get("experiment_id") == spec.get("experiment_id")
                 and comparison.get("partition") == "confirmation"
                 and type(comparison.get("confirmatory")) is bool, "comparison requires explicit frozen role")
        _require(comparison.get("treatment") in inventory and comparison.get("control") in inventory,
                 "comparison references a missing control")
        _require(comparison.get("expected_n_layouts") == n and comparison.get("expected_n_pairs") == n
                 and isinstance(comparison.get("eligible_pair_ids"), list)
                 and len(comparison["eligible_pair_ids"]) == n
                 and set(comparison["eligible_pair_ids"]) == set(pair_layout),
                 "comparison may not exclude frozen layouts or seeds")
        if comparison["confirmatory"]:
            primary = comparison.get("primary_id")
            _require(comparison.get("candidate_id") == candidate_id and primary in family["candidates"][candidate_id],
                     "primary comparison has unknown candidate or primary identity")
            binding = family["bindings"][candidate_id][primary]
            _require(all(key in comparison and fingerprint(comparison[key]) == fingerprint(value)
                         for key, value in binding.items()),
                     "primary comparison differs from its exact frozen binding")
            primary_ids.append(primary)
        else:
            _require("primary_id" not in comparison, "descriptive comparison may not claim a primary identity")
    _require(Counter(primary_ids) == Counter(family["candidates"][candidate_id]),
             "primary cohort must contain both frozen primary comparisons exactly once")


def analysis_options(freeze: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Bind reports to the full frozen family, without a local size override."""
    if "confirmation_primary_family" not in freeze:
        _require(not any(config.get("protocol_kind") in PROTOCOLS or "primary_comparisons" in config
                         for config in freeze["candidate_configurations"].values()),
                 "prospective co-primary configurations require their frozen global registry")
        return {"confirmation_family_size": len(freeze["candidate_configurations"])}
    registry = validate_registry_configurations(freeze["confirmation_primary_family"], freeze["candidate_configurations"])
    _require(spec.get("candidate_id") in freeze["candidate_configurations"], "unregistered candidate report")
    _require(spec.get("cohort_role", "primary") == "primary", "lexical supplements require descriptive paired analysis")
    validate_primary_spec(spec, freeze["candidate_configurations"][spec["candidate_id"]], registry)
    return {"confirmation_primary_family": deepcopy(registry)}


def inference_cutoff(deadline: str) -> str:
    """The six-hour reporting reserve is absolute and cannot be reset on resume."""
    value = datetime.fromisoformat(deadline)
    _require(value.tzinfo is not None, "campaign deadline requires an explicit timezone")
    return (value.astimezone(timezone.utc) - timedelta(hours=6)).isoformat()


def remaining_inference_seconds(deadline: str, frozen_cutoff: str, *, now: str) -> float:
    expected = inference_cutoff(deadline)
    _require(frozen_cutoff == expected, "frozen inference cutoff must preserve exactly six reporting hours")
    current = datetime.fromisoformat(now)
    _require(current.tzinfo is not None, "current time requires an explicit timezone")
    return max(0.0, (datetime.fromisoformat(expected) - current.astimezone(timezone.utc)).total_seconds())


def projected_jobs(configurations: Mapping[str, Any], supplements: Mapping[str, bool]) -> int:
    """Count whole control inventories; alternate renderings repeat 32 layouts."""
    _require(set(supplements) == set(configurations) and all(type(value) is bool for value in supplements.values()),
             "budget requires an explicit supplement decision for every candidate")
    total = 0
    for candidate_id, config in configurations.items():
        protocol = config.get("protocol_kind")
        _require(protocol in PROTOCOLS, "unknown budgeted protocol")
        n = config.get("n_layouts")
        _require(type(n) is int and n in {32, 64, 128} and config.get("generation_seeds") == [42],
                 "budget requires fixed full layout count and generation seeds")
        total += (n + (32 if supplements[candidate_id] else 0)) * PROTOCOLS[protocol][1]
    return total


def validate_budget(configurations: Mapping[str, Any], supplements: Mapping[str, bool],
                    seconds_per_job: float, remaining_seconds: float) -> dict[str, Any]:
    for value in (seconds_per_job, remaining_seconds):
        _require(type(value) in {int, float} and math.isfinite(value) and value > 0,
                 "budget requires finite positive time estimates")
    count = projected_jobs(configurations, supplements)
    projected = count * seconds_per_job + 2 * 3600
    _require(projected <= remaining_seconds, "whole confirmation plan exceeds the inference allocation including two-hour overhead")
    return {"jobs": count, "seconds_per_native_job": seconds_per_job, "overhead_seconds": 7200,
            "projected_inference_and_overhead_seconds": projected,
            "remaining_to_absolute_inference_cutoff_seconds": remaining_seconds}
