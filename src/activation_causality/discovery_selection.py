"""Adopted v4 discovery qualification, separate from confirmation inference.

The production entry authenticates the complete seven-source native battery
before evaluating any profile. The pure helper accepts already verified bundles
for CPU tests; it cannot authenticate files or replace that production gate.
No model, sampler, adapter-feasibility or forced-action input is consulted.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .confirmation_sources import (
    CONTEXTUAL_SOURCE,
    REQUIRED_BATTERY,
    read_closed_native_battery,
)
from .registry import fingerprint

POLICY_PATH = "outputs/activation_causality/campaign_20260912/reviews/candidate_selection_policy_proposal_v4.json"
POLICY_SHA256 = "608add8b4ffdb40786f548f6d53578baf3228f1a94e2d28ba9b18be3b62f62d9"
FAMILIES = ("wall", "key_possession", "door_status")
FAMILY_NAMES = {"wall": "wall", "key_possession": "key", "door_status": "door"}
REFERENCES = (
    "baseline",
    "counterfactual_text",
    "donor_text",
    "natural_donor_text",
    "same_fact_donor_text",
    "irrelevant_donor_text",
)
CONTEXTUAL_ROLES = (
    "matched",
    "natural",
    "same_fact",
    "irrelevant",
    "random",
    "reverse",
    "identity",
    "restoration_equivalent",
)
SUBSPACE_CONDITIONS = (
    "baseline",
    "donor_text",
    "full_transfer",
    "subspace_identity",
    "restoration_equivalent",
    "subspace_removal",
    *(
        f"{role}_{dose}"
        for dose in ("0p5", "1")
        for role in (
            "subspace_forward",
            "subspace_reverse",
            "random_subspace",
            "projected_norm_matched_random_subspace",
        )
    ),
)
ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}
RANKING_FIELDS = (
    "negative_minimum_censoring_lower",
    "negative_minimum_itt",
    "negative_efficacy_itt",
    "whole_profile_unresolved_count",
    "patched_layer_count",
    "total_selected_positions",
    "rank_or_zero_for_contextual",
    "dose",
    "negative_contextual_layer_or_zero",
    "whole_profile_seconds",
    "configuration_sha256",
)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mean(values: list[float | int]) -> float:
    return math.fsum(values) / len(values)


def _inventory(source: Mapping[str, Any], experiment_id: str) -> dict:
    """Check the full registered source shape; do not accept selected subsets."""
    spec, rows = source["specification"], source["rows"]
    contextual = experiment_id == CONTEXTUAL_SOURCE
    conditions = (
        set(REFERENCES)
        | {f"layer{layer}_{role}" for layer in (0, 10) for role in CONTEXTUAL_ROLES}
        if contextual
        else set(SUBSPACE_CONDITIONS)
    )
    n = 6 if contextual else 4
    _require(
        source.get("experiment_id") == spec.get("experiment_id") == experiment_id
        and spec.get("partition") == "discovery",
        "source experiment/partition mismatch",
    )
    _require(_hash(source.get("source_spec_sha256")), "source spec hash missing")
    _require(
        isinstance(rows, list)
        and len(rows) == len(spec["jobs"]) == n * len(conditions),
        "complete native source inventory is required",
    )
    jobs = {j["job_id"]: j for j in spec["jobs"]}
    _require(len(jobs) == len(rows), "source job IDs must be unique")
    row_map, layouts = {}, {}
    seen = set()
    for row in rows:
        jid = row.get("job_id")
        _require(jid in jobs and jid not in seen, "unknown or repeated source result")
        seen.add(jid)
        job = jobs[jid]
        _require(
            jid == fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24],
            "source job is not canonical",
        )
        _require(
            row.get("spec_sha256") == source["source_spec_sha256"],
            "source result/spec mismatch",
        )
        for key, value in job.items():
            if key not in {"recipient_text", "donor_text", "shared_base_text"}:
                _require(
                    key in row and fingerprint(row[key]) == fingerprint(value),
                    "immutable source field changed: " + key,
                )
        _require(
            job.get("natural") is True
            and type(job.get("seed")) is int
            and job["seed"] == 42
            and type(job.get("max_new_tokens")) is int
            and job["max_new_tokens"] == 4096
            and type(job.get("temperature")) in (int, float)
            and job["temperature"] == 0.7,
            "selection requires the fixed native source settings",
        )
        _require(
            row.get("status") == "complete", "technical source failures cannot qualify"
        )
        status, action = row.get("generation_status"), row.get("native_action")
        _require(
            status in {"complete", "truncated", "malformed"}
            and (action in ACTIONS if status == "complete" else action is None),
            "native status/action mismatch",
        )
        _require(
            type(row.get("seconds")) in (int, float)
            and math.isfinite(row["seconds"])
            and row["seconds"] > 0,
            "whole-profile cost requires every finite positive native duration",
        )
        _require(
            job.get("target_action") in ACTIONS
            and job.get("original_action") in ACTIONS
            and job["target_action"] != job["original_action"],
            "source target/original action invalid",
        )
        lid, pid, family, condition = (
            job[k] for k in ("layout_id", "pair_id", "family", "condition")
        )
        _require(
            all(isinstance(x, str) and x for x in (lid, pid, family))
            and condition in conditions,
            "unexpected source condition or layout identity",
        )
        lineage = (pid, family, job["target_action"], job["original_action"])
        _require(
            lid not in layouts or layouts[lid] == lineage,
            "layout has inconsistent pairing/target lineage",
        )
        layouts[lid] = lineage
        _require(
            (lid, condition) not in row_map, "duplicate layout/condition observation"
        )
        row_map[lid, condition] = row
    _require(
        len(layouts) == n
        and len({value[0] for value in layouts.values()}) == n
        and set(row_map)
        == {(lid, condition) for lid in layouts for condition in conditions},
        "source must retain one fixed-seed pair per canonical layout and every control",
    )
    families = Counter(value[1] for value in layouts.values())
    if contextual:
        _require(
            families == {family: 2 for family in FAMILIES},
            "contextual source must retain the fixed2/2/2 mixture",
        )
    else:
        family = next(
            f
            for f in FAMILIES
            if experiment_id.startswith("subspace_" + FAMILY_NAMES[f] + "_")
        )
        _require(families == {family: 4}, "subspace source family population changed")
    manifest = source.get("source_result_hashes")
    _require(
        isinstance(manifest, dict)
        and len(manifest) == len(rows)
        and {Path(path).stem for path in manifest} == set(jobs)
        and all(_hash(digest) for digest in manifest.values())
        and fingerprint(manifest) == source.get("source_result_manifest_sha256"),
        "source result manifest incomplete",
    )
    audit = source.get("independent_audit", {})
    _require(
        isinstance(audit.get("path"), str) and _hash(audit.get("sha256")),
        "independent audit binding missing",
    )
    return {
        "source": source,
        "rows": row_map,
        "layouts": layouts,
        "conditions": conditions,
        "family_layout_counts": dict(sorted(families.items())),
    }


def _source_binding(inventory: dict) -> dict:
    source = inventory["source"]
    return {
        "source_experiment_id": source["experiment_id"],
        "source_spec_sha256": source["source_spec_sha256"],
        "source_result_manifest_sha256": source["source_result_manifest_sha256"],
        "source_result_hashes": deepcopy(source["source_result_hashes"]),
        "independent_audit": deepcopy(source["independent_audit"]),
    }


def _contrast(
    inventory: dict,
    treatment: str,
    control: str,
    *,
    roles: list[str],
    registered: bool = True,
) -> dict:
    comparisons = [
        c
        for c in inventory["source"]["specification"]["comparisons"]
        if c["treatment"] == treatment and c["control"] == control
    ]
    _require(
        len(comparisons) == 1 if registered else len(comparisons) <= 1,
        "measured primary/efficacy must have one registered source comparison",
    )
    ids = sorted(value[0] for value in inventory["layouts"].values())
    if comparisons:
        c = comparisons[0]
        _require(
            isinstance(c.get("id"), str) and bool(c["id"]),
            "source comparison ID must be explicit",
        )
        _require(
            c.get("predicted_direction") == "increase",
            "qualification contrast must predict increase",
        )
        _require(
            sorted(c.get("eligible_pair_ids", [])) == ids
            and type(c.get("expected_n_pairs")) is int
            and c["expected_n_pairs"] == len(ids)
            and type(c.get("expected_n_layouts")) is int
            and c["expected_n_layouts"] == len(ids),
            "qualification cannot use a restricted source comparison subset",
        )
    values = []
    for lid, (pid, family, _, _) in sorted(inventory["layouts"].items()):
        a, b = inventory["rows"][lid, treatment], inventory["rows"][lid, control]
        ay, by = (
            int(a["native_action"] == a["target_action"]),
            int(b["native_action"] == b["target_action"]),
        )
        au, bu = (
            int(a["generation_status"] != "complete"),
            int(b["generation_status"] != "complete"),
        )
        values.append(
            {
                "layout_id": lid,
                "pair_id": pid,
                "family": family,
                "treatment_target": ay,
                "control_target": by,
                "treatment_unresolved": au,
                "control_unresolved": bu,
                "itt_delta": ay - by,
                "censoring_lower": ay - by - bu,
                "censoring_upper": ay + au - by,
            }
        )
    return {
        "treatment": treatment,
        "control": control,
        "roles": roles,
        "status": "measured",
        "source_comparison_id": comparisons[0]["id"] if comparisons else None,
        "origin": "existing_source_comparison"
        if comparisons
        else "policy_physical_reference_diagnostic",
        "n_layouts": len(values),
        "native_itt_delta": _mean([v["itt_delta"] for v in values]),
        "completion_sensitive_point_bounds": [
            _mean([v[k] for v in values])
            for k in ("censoring_lower", "censoring_upper")
        ],
        "layout_effects": values,
        "family_effects": {
            family: {
                "n_layouts": sum(v["family"] == family for v in values),
                "native_itt_delta": _mean(
                    [v["itt_delta"] for v in values if v["family"] == family]
                ),
                "completion_sensitive_point_bounds": [
                    _mean([v[k] for v in values if v["family"] == family])
                    for k in ("censoring_lower", "censoring_upper")
                ],
            }
            for family in sorted({v["family"] for v in values})
        },
    }


def _geometry(
    inventory: dict, treatment: str, layer: int, rank: int | None, dose: float
) -> dict:
    profiles, widths = [], {}
    for lid in sorted(inventory["layouts"]):
        row = inventory["rows"][lid, treatment]
        patch, selection = row["intervention"], row["selection"]
        _require(
            patch.get("layers") == [layer]
            and type(patch["layers"][0]) is int
            and patch.get("site") == "resid_post"
            and type(patch.get("dose")) in (int, float)
            and patch["dose"] == dose
            and patch.get("operation") == ("replace" if rank is None else "subspace"),
            "profile geometry differs from the tested fixed intervention",
        )
        _require(
            selection.get("scope") == "explicit",
            "profile requires actual explicit source support",
        )
        rp, dp = selection.get("recipient_positions"), selection.get("donor_positions")
        _require(
            isinstance(rp, list)
            and isinstance(dp, list)
            and len(rp) == len(dp) > 0
            and all(type(p) is int and p >= 0 for p in rp + dp)
            and len(set(rp)) == len(rp)
            and len(set(dp)) == len(dp),
            "source position geometry invalid",
        )
        profiles.append(patch)
        widths[lid] = len(rp)
    _require(
        len({fingerprint(p) for p in profiles}) == 1,
        "selected intervention differs across layouts",
    )
    patch = deepcopy(profiles[0])
    if rank is not None:
        basis = inventory["source"]["specification"].get("basis", {})
        _require(
            type(basis.get("rank")) is int
            and basis["rank"] == rank
            and basis.get("family") == next(iter(inventory["family_layout_counts"]))
            and type(basis.get("layer")) is int
            and basis["layer"] == layer
            and basis.get("site") == "resid_post",
            "source basis rank/family/site mismatch",
        )
        _require(
            patch.get("audit_projected_norm") is True
            and patch.get("projected_norm_relative_tolerance") == 0.01
            and all(
                _hash(patch.get(k)) for k in ("basis_sha256", "basis_metadata_sha256")
            ),
            "selected subspace must retain its audited fixed basis and norm contract",
        )
    return {
        "layer": layer,
        "patched_layer_count": 1,
        "site": "resid_post",
        "rank": rank,
        "dose": dose,
        "operation": patch["operation"],
        "source_intervention": patch,
        "source_position_widths": widths,
        "total_selected_positions": sum(widths.values()),
    }


def _ranking_key(profile: dict) -> list:
    """Ascending lexicographic key, meaningful only within a reviewed group."""
    metrics, geometry = profile["ranking_metrics"], profile["configuration"]["geometry"]
    return [
        -metrics["minimum_censoring_lower"],
        -metrics["minimum_itt"],
        -metrics["efficacy_itt"],
        profile["measured_inventory"]["unresolved_count"],
        geometry["patched_layer_count"],
        geometry["total_selected_positions"],
        geometry["rank"] or 0,
        geometry["dose"],
        -geometry["layer"] if profile["question_slot"] == "contextual_input" else 0,
        profile["measured_inventory"]["total_seconds"],
        profile["configuration_sha256"],
    ]


def _profile(inventory: dict, *, layer: int, rank: int | None, dose: float) -> dict:
    contextual = rank is None
    prefix = f"layer{layer}"
    label = "0p5" if dose == 0.5 else "1"
    treatment = prefix + "_natural" if contextual else "subspace_forward_" + label
    controls = (
        (prefix + "_same_fact",)
        if contextual
        else ("baseline", "projected_norm_matched_random_subspace_" + label)
    )
    conditions = (
        sorted(set(REFERENCES) | {prefix + "_" + role for role in CONTEXTUAL_ROLES})
        if contextual
        else sorted(SUBSPACE_CONDITIONS)
    )
    _require(len(conditions) == 14, "whole measured profile must contain14 conditions")
    measured_pairs: dict[tuple[str, str], list[str]] = {}
    names = (
        ("contextual_fact_contrast",)
        if contextual
        else ("subspace_efficacy", "subspace_direction_contrast")
    )
    for name, control in zip(names, controls, strict=True):
        measured_pairs.setdefault((treatment, control), []).append(name)
    measured_pairs.setdefault((treatment, "baseline"), []).append("efficacy")
    metrics = [
        _contrast(inventory, t, c, roles=roles)
        for (t, c), roles in measured_pairs.items()
    ]
    physical = _contrast(
        inventory,
        "counterfactual_text" if contextual else "donor_text",
        "baseline",
        roles=["physical_counterfactual_reference"],
        registered=contextual,
    )
    failed = []
    for metric in metrics:
        if metric["native_itt_delta"] < 0.10:
            failed.append(
                {
                    "contrast": [metric["treatment"], metric["control"]],
                    "gate": "native_itt_at_least_0.10",
                }
            )
        if metric["completion_sensitive_point_bounds"][0] <= 0:
            failed.append(
                {
                    "contrast": [metric["treatment"], metric["control"]],
                    "gate": "censoring_lower_strictly_positive",
                }
            )
    if (
        physical["native_itt_delta"] <= 0
        or physical["completion_sensitive_point_bounds"][0] <= 0
    ):
        failed.append(
            {
                "contrast": [physical["treatment"], "baseline"],
                "gate": "positive_physical_counterfactual_reference_and_lower_bound",
            }
        )
    geometry = _geometry(inventory, treatment, layer, rank, dose)
    source = inventory["source"]
    configuration = {
        "protocol_kind": "contextual_input" if contextual else "input_fact_subspace",
        "source_experiment_id": source["experiment_id"],
        "source_spec_sha256": source["source_spec_sha256"],
        "geometry": geometry,
        "treatment_condition": treatment,
        "measured_primary_controls": list(controls),
        "measured_conditions": conditions,
        "source_layout_ids": sorted(inventory["layouts"]),
        "family_layout_counts": inventory["family_layout_counts"],
        "family_weights": {
            f: n / len(inventory["layouts"])
            for f, n in inventory["family_layout_counts"].items()
        },
        "generation_seeds": [42],
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "semantic_headline_eligible": False,
    }
    configuration["families" if contextual else "family"] = (
        list(FAMILIES) if contextual else next(iter(inventory["family_layout_counts"]))
    )
    selected_rows = [
        inventory["rows"][lid, condition]
        for lid in sorted(inventory["layouts"])
        for condition in conditions
    ]
    counts = {
        condition: {
            "n": len(inventory["layouts"]),
            "target_count": sum(
                r["native_action"] == r["target_action"]
                for r in selected_rows
                if r["condition"] == condition
            ),
            "generation_status_counts": dict(
                sorted(
                    Counter(
                        r["generation_status"]
                        for r in selected_rows
                        if r["condition"] == condition
                    ).items()
                )
            ),
        }
        for condition in conditions
    }
    family_label = "mixed" if contextual else configuration["family"]
    profile = {
        "profile_id": f"contextual_l{layer}"
        if contextual
        else f"subspace_{family_label}_l{layer}_r{rank}_dose{label}",
        "question_slot": "contextual_input" if contextual else "fixed_family_subspace",
        "configuration": configuration,
        "configuration_sha256": fingerprint(configuration),
        "source_binding": _source_binding(inventory),
        "qualified": not failed,
        "failed_gates": failed,
        "measured_qualification_contrasts": metrics,
        "physical_counterfactual_reference": physical,
        "unmeasured_components": [
            {
                "name": "natural_norm_direction_contrast",
                "treatment": treatment,
                "control": prefix + "_natural_norm_random",
                "status": "unmeasured_prospective_exception",
                "source_comparison_id": None,
                "native_itt_delta": None,
                "completion_sensitive_point_bounds": None,
                "qualifies_in_discovery": None,
                "included_in_ranking": False,
            }
        ]
        if contextual
        else [],
        "measured_inventory": {
            "conditions": conditions,
            "n_conditions": 14,
            "n_layouts": len(inventory["layouts"]),
            "n_rows": len(selected_rows),
            "unresolved_count": sum(
                r["generation_status"] != "complete" for r in selected_rows
            ),
            "total_seconds": math.fsum(r["seconds"] for r in selected_rows),
            "mean_seconds_per_job": _mean([r["seconds"] for r in selected_rows]),
            "condition_counts": counts,
        },
        "ranking_metrics": {
            "minimum_censoring_lower": min(
                m["completion_sensitive_point_bounds"][0] for m in metrics
            ),
            "minimum_itt": min(m["native_itt_delta"] for m in metrics),
            "efficacy_itt": next(
                m["native_itt_delta"] for m in metrics if "efficacy" in m["roles"]
            ),
        },
        "adapter_feasibility_assessed": False,
        "confirmation_opened": False,
    }
    profile["ranking_tuple"] = _ranking_key(profile)
    profile["descriptor"] = {
        "profile_id": profile["profile_id"],
        "protocol_kind": configuration["protocol_kind"],
        "source_experiment_id": source["experiment_id"],
        "source_spec_sha256": source["source_spec_sha256"],
        "layer": layer,
        "site": "resid_post",
        "selected_dose": dose,
        "rank": rank,
        "family": None if contextual else configuration["family"],
    }
    return profile


def evaluate_verified_sources(battery: Mapping[str, Any]) -> dict:
    """Pure policy calculation on a fully authenticated battery, also for tests.

    The fixed sources have one seed/pair per canonical layout. Report every profile
    and its family-stratified descriptive effects, with no family-specific success
    gate added to the adopted whole-mixture contextual question.
    """
    sources = battery.get("sources", {})
    _require(
        isinstance(sources, Mapping) and set(sources) == set(REQUIRED_BATTERY),
        "all seven verified native sources are required",
    )
    inventories = {eid: _inventory(sources[eid], eid) for eid in REQUIRED_BATTERY}
    for family in FAMILIES:
        prefix = "subspace_" + FAMILY_NAMES[family] + "_l0_r"
        a, b = (inventories[prefix + str(rank) + "_native_v1"] for rank in (1, 4))
        _require(
            a["layouts"] == b["layouts"],
            "cross-rank ordering requires identical shared layout/pair/target populations",
        )
    profiles = [
        _profile(inventories[CONTEXTUAL_SOURCE], layer=layer, rank=None, dose=1.0)
        for layer in (0, 10)
    ]
    for family in FAMILIES:
        for rank in (1, 4):
            source = inventories[
                f"subspace_{FAMILY_NAMES[family]}_l0_r{rank}_native_v1"
            ]
            profiles.extend(
                _profile(source, layer=0, rank=rank, dose=dose) for dose in (0.5, 1.0)
            )
    groups = {
        "contextual_input": [
            p for p in profiles if p["question_slot"] == "contextual_input"
        ]
    }
    groups.update(
        {
            family: [p for p in profiles if p["configuration"].get("family") == family]
            for family in FAMILIES
        }
    )
    ranked = {
        key: [
            p["profile_id"]
            for p in sorted(value, key=lambda p: p["ranking_tuple"])
            if p["qualified"]
        ]
        for key, value in groups.items()
    }
    selected = {
        "contextual_input": ranked["contextual_input"][0]
        if ranked["contextual_input"]
        else None,
        "fixed_family_subspace": next(
            (ranked[f][0] for f in FAMILIES if ranked[f]), None
        ),
    }
    return {
        "format": "discovery_scientific_selection_v1",
        "policy_path": POLICY_PATH,
        "policy_sha256": POLICY_SHA256,
        "policy_timing": "data-informed adopted policy after known discovery, before this decision and every confirmation query",
        "source_experiments": list(REQUIRED_BATTERY),
        "all_required_sources_complete": True,
        "profiles": profiles,
        "ranking_field_order": list(RANKING_FIELDS),
        "qualified_rankings_within_population": ranked,
        "subspace_family_priority": list(FAMILIES),
        "selected_scientific_profile_ids": selected,
        "n_selected_scientific_profiles": sum(v is not None for v in selected.values()),
        "question_status": {
            q: "qualified_scientific_profile_selected"
            if p
            else "no_qualifying_candidate"
            for q, p in selected.items()
        },
        "source_audit_bindings": {
            eid: _source_binding(inventory) for eid, inventory in inventories.items()
        },
        "adapter_feasibility_assessed": False,
        "confirmation_opened": False,
        "freeze_created": False,
        "claims": {
            "semantic_headline_eligible": False,
            "population_inference": False,
            "no_qualifier_is_not_a_bounded_population_negative": True,
            "censoring_bounds_are_observed_cohort_point_bounds_not_confidence_intervals": True,
            "reused_cross_rank_layouts_are_not_independent_replications": True,
        },
        "limits": [
            "Scientific selection precedes adapter feasibility; an infeasible selected profile is retained without backfill.",
            "The prospective contextual natural-norm contrast remains unmeasured and must pass its own frozen confirmation component.",
            "No forced-action readout, bootstrap significance, future-adapter availability or confirmation outcome participates in selection.",
        ],
    }


def evaluate_closed_discovery(
    campaign: Path, repo: Path, audit_bindings: Mapping[str, Any]
) -> dict:
    """Production read-only entry: full battery gate precedes any evaluation."""
    battery = read_closed_native_battery(campaign, repo, audit_bindings)
    _require(
        hashlib.sha256((repo / POLICY_PATH).read_bytes()).hexdigest() == POLICY_SHA256,
        "adopted v4 selection policy bytes changed",
    )
    result = evaluate_verified_sources(battery)
    result["production_source_authentication"] = "read_closed_native_battery"
    result["complete_cohort_means_seconds"] = deepcopy(
        battery["complete_cohort_means_seconds"]
    )
    result["minimum_budget_seconds_per_native_job"] = battery[
        "minimum_budget_seconds_per_native_job"
    ]
    return result
