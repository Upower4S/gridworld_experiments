"""Freeze candidate choices and verify confirmation before model queries."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .registry import (
    append_event, file_sha256, fingerprint, immutable_json, now_utc, read_json,
    seconds_remaining,
)

ANALYSIS_CONFIG = {
    "version": 3,
    "alpha": 0.05, "minimum_effect": 0.10, "n_bootstrap": 10000,
    "n_test_bootstrap": 10000, "seed": 42,
    "primary_endpoint": "natural_target_action_intention_to_treat",
    "weighting": "equal_layout_then_equal_pair_seed",
    "p_value_method": "centered_layout_bootstrap_plus_one",
    "interval_adjustment": "bonferroni_simultaneous", "p_value_adjustment": "holm",
    "mixed_family_resampling": "within_family_layout_bootstrap",
    "mixed_family_weighting": "frozen_family_layout_counts_over_total",
    "mixed_family_constant_guard": "weighted_simultaneous_family_intervals",
    "mixed_family_constant_p_guard": "two_sided_bounded_layout_hoeffding",
}

MIXED_FAMILIES = ("wall", "key_possession", "door_status")


def _mixed_family_contract(value: dict[str, Any], total: int) -> dict[str, Any] | None:
    if "families" not in value:
        if "family_layout_counts" in value or "family_weights" in value:
            raise ValueError("family quotas and weights require an explicit mixed families list")
        return None
    families = value["families"]
    if "family" in value or families != list(MIXED_FAMILIES):
        raise ValueError("mixed candidates require exactly the three approved families and no family scalar")
    counts = value.get("family_layout_counts")
    if (not isinstance(counts, dict) or set(counts) != set(MIXED_FAMILIES)
            or any(type(n) is not int or n < 1 for n in counts.values())
            or sum(counts.values()) != total or max(counts.values()) - min(counts.values()) > 1):
        raise ValueError("mixed family quotas must be positive, balanced and sum to n_layouts")
    weights = {family: counts[family] / total for family in MIXED_FAMILIES}
    supplied = value.get("family_weights", weights)
    if (not isinstance(supplied, dict) or set(supplied) != set(weights)
            or any(type(supplied[f]) not in {int, float} or not math.isfinite(supplied[f])
                   or abs(supplied[f] - weights[f]) > 1e-12 for f in weights)):
        raise ValueError("mixed family weights must equal frozen family counts divided by n_layouts")
    return {"families": list(MIXED_FAMILIES),
            "family_layout_counts": {family: counts[family] for family in MIXED_FAMILIES},
            "family_weights": weights}


def excluded_layout_ids(campaign: Path) -> set[str]:
    excluded = set()
    for path in sorted((campaign / "datasets").glob("*.json")):
        dataset = read_json(path)
        for aliases in dataset.get("layout_aliases", {}).values():
            excluded.update(aliases)
        for pair in dataset.get("pairs", []):
            excluded.add(pair["layout_id"])
            excluded.update(pair.get("metadata", {}).get("variant_layout_ids", []))
            for control in pair.get("control_records", []):
                excluded.update(control.get("variant_layout_ids", []))
    for path in sorted((campaign / "specs").glob("*.json")):
        for job in read_json(path).get("jobs", []):
            excluded.add(job["layout_id"])
            excluded.update(job.get("layout_aliases", []))
    return excluded


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _model_identity(campaign: Path) -> dict[str, Any]:
    from .model_identity import snapshot_identity
    config = read_json(campaign / "campaign.json")
    return snapshot_identity(Path(config["model_path"]))


def _confirmation_tokenizer(campaign: Path) -> Any:
    """Load only the local frozen tokenizer; no model or network request."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(read_json(campaign / "campaign.json")["model_path"],
                                         local_files_only=True)


def _pilot(campaign: Path, selection: dict[str, Any]) -> dict[str, Any]:
    identifiers = selection.get("pilot_job_ids")
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("pilot_job_ids must identify completed native executions")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("pilot_job_ids must be unique")
    by_id = {}
    for path in sorted((campaign / "results").glob("*/*.json")):
        result = read_json(path)
        if result.get("job_id") in identifiers:
            if result["job_id"] in by_id:
                raise ValueError("pilot job identity occurs in multiple result paths")
            by_id[result["job_id"]] = (result, path)
    if set(by_id) != set(identifiers):
        raise ValueError("pilot_job_ids include absent execution records")
    durations, caps, hashes, completed = [], [], {}, 0
    for identifier in identifiers:
        result, path = by_id[identifier]
        if result.get("partition") == "confirmation" or (campaign / "reviews" / f"{result['experiment_id']}_retired.json").exists():
            raise ValueError("confirmation or retired outcomes cannot supply the discovery pilot")
        if result.get("status") != "complete" or result.get("natural") is not True or result.get("generation_status") not in {"complete", "truncated", "malformed"}:
            raise ValueError("pilot must contain completed native executions, including recorded non-completions")
        durations.append(_positive(result.get("seconds"), "measured pilot seconds"))
        caps.append(_positive(result.get("max_new_tokens"), "pilot token cap"))
        completed += result["generation_status"] == "complete"
        hashes[str(path.resolve())] = file_sha256(path)
    if not completed:
        raise ValueError("native pilot has no successfully parsed final action")
    measured = sum(durations) / len(durations)
    estimate = _positive(selection.get("pilot_seconds_per_native_job"), "pilot_seconds_per_native_job")
    if estimate + 1e-6 < measured:
        raise ValueError("pilot throughput estimate is faster than its measured mean")
    return {"job_ids": identifiers, "result_hashes": hashes, "mean_seconds": measured,
            "budget_seconds_per_job": estimate, "minimum_token_cap": min(caps),
            "parsed_completions": completed}


def _validated_source_result(path: Path, source_digest: str, jobs: dict[str, dict]) -> dict:
    result = read_json(path)
    identifier = result.get("job_id")
    if identifier not in jobs or path.name != f"{identifier}.json":
        raise ValueError("source result must identify its exact known job and canonical result file")
    if result.get("spec_sha256") != source_digest:
        raise ValueError("source result/spec fingerprint mismatch")
    job = jobs[identifier]
    # The runner omits only display text; authoritative token IDs and every
    # other frozen field must survive unchanged in a selection-source result.
    for key, value in job.items():
        if key not in {"recipient_text", "donor_text"} and (key not in result or result[key] != value):
            raise ValueError(f"source result immutable job field mismatch: {identifier}/{key}")
    if "recipient_token_ids" in job and result.get("input_token_sha256") != fingerprint(job["recipient_token_ids"]):
        raise ValueError("source result authoritative input-token fingerprint mismatch")
    if result.get("natural") is True and result.get("status") == "complete":
        status, action = result.get("generation_status"), result.get("native_action")
        if (status == "complete" and action not in {"UP", "DOWN", "LEFT", "RIGHT"}
                or status in {"truncated", "malformed"} and action is not None):
            raise ValueError("source result native action is inconsistent with its completion status")
    return result


def _source_result_paths(campaign: Path, experiment_id: str, jobs: dict[str, dict]) -> list[Path]:
    """Require the whole registered source cohort before opening any outcome."""
    paths = sorted((campaign / "results" / experiment_id).glob("*.json"))
    expected = {f"{identifier}.json" for identifier in jobs}
    if not expected or {path.name for path in paths if path.is_file()} != expected:
        raise ValueError("selection requires the exact complete source cohort result inventory")
    return paths


def _complete_source_results(campaign: Path, experiment_id: str, source_digest: str,
                             jobs: dict[str, dict]) -> list[tuple[Path, dict]]:
    results = []
    for path in _source_result_paths(campaign, experiment_id, jobs):
        result = _validated_source_result(path, source_digest, jobs)
        if result.get("status") != "complete":
            raise ValueError("selection requires technical completion of every source condition")
        if result.get("natural") is True and result.get("generation_status") not in {"complete", "truncated", "malformed"}:
            raise ValueError("native source conditions must retain a valid completion or unresolved status")
        results.append((path, result))
    return results


def _candidate_configuration(campaign: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    source = campaign / "specs" / f"{candidate['source_experiment_id']}.json"
    spec = read_json(source)
    if spec["partition"] != "discovery":
        raise ValueError("confirmation candidates must originate in discovery")
    if (campaign / "reviews" / f"{candidate['source_experiment_id']}_retired.json").exists():
        raise ValueError("retired experiment cannot select a candidate")
    treatment, control = candidate["treatment_condition"], candidate["control_condition"]
    source_comparisons = [item for item in spec["comparisons"] if item["treatment"] == treatment and item["control"] == control]
    if len(source_comparisons) != 1:
        raise ValueError("selected treatment/control must identify one existing source comparison")
    count = candidate["n_layouts"]
    if type(count) is not int or count not in {32, 64, 128}:
        raise ValueError("freeze 128, 64 or 32 independent layouts")
    mixed = _mixed_family_contract(candidate, count)
    source_families = {job["family"] for job in spec["jobs"] if job["condition"] == treatment}
    family = candidate.get("family", next(iter(source_families)) if len(source_families) == 1 else None)
    selected_families = set(mixed["families"]) if mixed else {family}
    if not selected_families <= source_families:
        raise ValueError("candidate family must identify its source treatment cohort")
    treatment_jobs = [job for job in spec["jobs"] if job["condition"] == treatment and job["family"] in selected_families]
    is_input = str(spec.get("claim_scope", "")).startswith("input_fact_information_routing")
    if mixed and not is_input:
        raise ValueError("mixed-family confirmation is supported only for input_fact")
    protocol_kind = "input_fact" if is_input else "controlled_state"
    profile_keys = ("intervention", "temperature") if is_input else ("intervention", "selection", "temperature")
    profiles = {fingerprint({key: job[key] for key in profile_keys}) for job in treatment_jobs}
    if len(profiles) != 1:
        raise ValueError("source treatment has ambiguous intervention configurations")
    example = treatment_jobs[0]
    intervention, span = example["intervention"], example["selection"]
    if len(intervention["layers"]) != 1 or intervention.get("random_control") or intervention["operation"] != "replace" or intervention["dose"] != 1.0:
        raise ValueError("this confirmation builder supports one frozen full-replacement layer per candidate")
    if is_input:
        from .input_sites import aligned_changed_span
        layer = intervention["layers"][0]
        if treatment != f"factspan_layer{layer}" or control not in {"baseline", f"random_factspan_layer{layer}", f"lasttoken_layer{layer}"}:
            raise ValueError("input confirmation must freeze a source fact-token contrast")
        for job in treatment_jobs:
            alignment = aligned_changed_span(job["recipient_token_ids"], job["donor_token_ids"])
            expected_span = {"scope": "explicit", "recipient_positions": alignment["recipient_positions"],
                             "donor_positions": alignment["donor_positions"]}
            if job["selection"] != expected_span or job.get("metadata", {}).get("representation_kind") != "input_fact_tokens":
                raise ValueError("input source must use its exact aligned changed-fact token span")
            if job.get("metadata", {}).get("claim_scope") != "input_fact_information_routing":
                raise ValueError("input source claim scope is inconsistent")
        if example["temperature"] != 0.7:
            raise ValueError("input confirmation requires the fixed source temperature 0.7")
        tokenizer = spec.get("tokenizer", {})
        if tokenizer.get("add_special_tokens") is not False or tokenizer.get("context_renderer") != "campaign.render_analysis_prompt":
            raise ValueError("input source tokenizer contract is missing")
        if Path(tokenizer.get("name_or_path", "")).resolve() != Path(read_json(campaign / "campaign.json")["model_path"]).resolve():
            raise ValueError("input source tokenizer differs from the frozen local model")
        normalized_span = {"scope": "explicit", "alignment": "exact_equal_length_common_prefix_and_suffix"}
    else:
        if span["scope"] != "last_tokens":
            raise ValueError("this confirmation builder requires the frozen last-token span")
        normalized_span = span
    if source_comparisons[0].get("predicted_direction", "increase") != "increase":
        raise ValueError("primary confirmation must predict increased recipient target-action success")
    seeds = candidate.get("generation_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("freeze unique nonnegative integer generation seeds")
    cap = candidate["max_new_tokens"]
    if type(cap) is not int or not 1 <= cap <= 8192:
        raise ValueError("invalid native-generation token budget")
    dataset_seed = candidate.get("dataset_seed")
    if type(dataset_seed) is not int or dataset_seed < 0:
        raise ValueError("freeze an explicit nonnegative integer dataset_seed")
    inventories = {f: {job["condition"] for job in spec["jobs"] if job["family"] == f
                      and (not job["intervention"]["layers"] or job["intervention"]["layers"] == intervention["layers"])}
                   for f in selected_families}
    conditions = next(iter(inventories.values()))
    if any(inventory != conditions for inventory in inventories.values()):
        raise ValueError("selected families have different source control inventories")
    if control not in conditions:
        raise ValueError("selected control is outside the selected layer's source cohort")
    if is_input and conditions != {"baseline", "counterfactual_text", f"identity_layer{layer}",
                                    f"factspan_layer{layer}", f"random_factspan_layer{layer}", f"lasttoken_layer{layer}"}:
        raise ValueError("input confirmation requires all six source control conditions")
    if "conditions_per_layout" in candidate and candidate["conditions_per_layout"] != len(conditions):
        raise ValueError("conditions_per_layout differs from the source-derived inventory")
    source_digest = file_sha256(source)
    source_jobs = {job["job_id"]: job for job in spec["jobs"]}
    if len(source_jobs) != len(spec["jobs"]):
        raise ValueError("source specification contains duplicate job identities")
    completed = {treatment: set(), control: set()}
    result_paths = {}
    source_results = _complete_source_results(campaign, candidate["source_experiment_id"], source_digest, source_jobs)
    for path, result in source_results:
        if result.get("condition") in completed and result.get("family") in selected_families and result.get("status") == "complete":
            if not is_input or (result.get("natural") is True and result.get("generation_status") in {"complete", "truncated", "malformed"}):
                key = result["family"], result["layout_id"], result["pair_id"], result["seed"]
                if (key, result["condition"]) in result_paths:
                    raise ValueError("source comparison contains duplicate pair/seed/condition executions")
                completed[result["condition"]].add(key)
                result_paths[key, result["condition"]] = path
    paired_source = completed[treatment] & completed[control]
    paired_source_count = len(paired_source)
    if not paired_source_count:
        raise ValueError("selected source comparison has no completed paired development executions")
    observed_by_family = {f: sum(key[0] == f for key in paired_source) for f in selected_families}
    if mixed and any(not n for n in observed_by_family.values()):
        raise ValueError("mixed candidate needs completed paired native source observations in each family")
    configuration = {"candidate_id": candidate["candidate_id"], "source_experiment_id": candidate["source_experiment_id"],
            "source_spec_sha256": source_digest, "source_comparison_id": source_comparisons[0]["id"],
            "observed_source_pairs": paired_source_count,
            "treatment_condition": treatment, "control_condition": control, "family": family,
            "layer": intervention["layers"][0], "layers": list(intervention["layers"]), "site": intervention["site"],
            "protocol_kind": protocol_kind, "intervention": intervention, "selection": normalized_span,
            "temperature": example["temperature"], "n_layouts": count, "generation_seeds": seeds,
            "dataset_seed": dataset_seed,
            "max_new_tokens": cap, "condition_inventory": sorted(conditions),
            "conditions_per_layout_upper_bound": len(conditions), "natural": True}
    configuration["source_result_hashes"] = {str(path.relative_to(campaign)): file_sha256(path)
                                             for path, _ in source_results}
    if is_input:
        configuration.update(claim_scope="input_fact_information_routing", semantic_headline_eligible=False,
                             input_alignment_rule="exact_equal_length_common_prefix_and_suffix", tokenizer=tokenizer)
    else:
        configuration["last_k"] = span["last_k"]
    if mixed:
        configuration.pop("family")
        configuration.update(mixed)
        configuration.update(resampling_scheme="within_family_layout_bootstrap", stratification_key="family")
        configuration["observed_source_pairs_by_family"] = {f: observed_by_family[f] for f in MIXED_FAMILIES}
    return configuration


def freeze_selection(campaign: Path, repo: Path, selection: dict[str, Any]) -> dict:
    """Publish one independently reviewed choice before opening confirmation."""
    freeze_id = selection["freeze_id"]
    if not freeze_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in freeze_id):
        raise ValueError("invalid freeze ID")
    existing = sorted((campaign / "freezes").glob("*.json"))
    own_path = campaign / "freezes" / f"{freeze_id}.json"
    if any(path != own_path for path in existing):
        raise ValueError("one immutable candidate-selection freeze is allowed per campaign")
    if own_path.exists():
        previous = read_json(own_path)
        if previous["selection"] != selection:
            raise ValueError("cannot revise a frozen selection")
        return previous
    if any(read_json(path).get("partition") == "confirmation" for path in (campaign / "specs").glob("*.json")):
        raise ValueError("confirmation already opened before candidate selection")
    candidates = selection["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("freeze one or two candidates")
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs")
    review_path = Path(selection["review_path"])
    if not review_path.is_absolute():
        review_path = repo / review_path
    review = read_json(review_path)
    if review.get("approved") is not True or review.get("freeze_id") != freeze_id or review.get("selection_sha256") != fingerprint(selection):
        raise ValueError("independent review must approve and fingerprint this exact selection")
    if not selection.get("selection_rationale"):
        raise ValueError("record the candidate selection rationale")
    pilot = _pilot(campaign, selection)
    configurations = {candidate["candidate_id"]: _candidate_configuration(campaign, candidate) for candidate in candidates}
    hours = sum(config["n_layouts"] * len(config["generation_seeds"]) * config["conditions_per_layout_upper_bound"]
                * pilot["budget_seconds_per_job"] * max(1.0, config["max_new_tokens"] / pilot["minimum_token_cap"]) / 3600
                for config in configurations.values())
    if hours > max(0, seconds_remaining(campaign) / 3600 - 6):
        raise ValueError("confirmation projection exceeds remaining inference allocation; preserve six hours for reporting")
    from .campaign import runtime_fingerprint
    implementation = runtime_fingerprint(repo)["sources"]
    value = {"schema_version": 2, "frozen_at": now_utc(), "selection": selection,
             "review_sha256": file_sha256(review_path), "candidate_configurations": configurations,
             "pilot": pilot, "analysis_configuration": dict(ANALYSIS_CONFIG),
             "source_specifications": {config["source_experiment_id"]: config["source_spec_sha256"] for config in configurations.values()},
             "source_results": {path: digest for config in configurations.values()
                                for path, digest in config["source_result_hashes"].items()},
             "implementation": implementation,
             "model_identity": _model_identity(campaign), "projected_confirmation_hours": hours,
             "excluded_layout_ids": sorted(excluded_layout_ids(campaign)),
             "claim_scope": "fresh canonical layout families under the frozen local model and native-action protocol"}
    immutable_json(own_path, value)
    append_event(campaign, "candidate_selection_frozen", freeze_id=freeze_id, candidates=len(candidates), projected_hours=hours)
    return value


def validate_confirmation_ready(campaign: Path, repo: Path, spec: dict) -> None:
    """Rebuild the exact frozen job contract before any confirmation execution."""
    if spec["partition"] != "confirmation":
        return
    from .environments import partition_for_layout
    from .experiments import build_controlled_state_spec

    freeze_path = campaign / "freezes" / f"{spec['freeze_id']}.json"
    if len(list((campaign / "freezes").glob("*.json"))) != 1:
        raise ValueError("confirmation requires one campaign-wide immutable freeze")
    freeze = read_json(freeze_path)
    new_mode = (freeze.get("confirmation_mode") == "co_primary_native_v4"
                or "confirmation_primary_family" in freeze
                or any(value.get("protocol_kind") in {"contextual_input", "input_fact_subspace"}
                       or "primary_comparisons" in value for value in freeze.get("candidate_configurations", {}).values()))
    if new_mode:
        from .confirmation_freeze import validate_frozen_confirmation_ready
        validate_frozen_confirmation_ready(campaign, repo, spec)
        return
    from .campaign import runtime_fingerprint
    from .engine import engine_dependency_hashes
    # Legacy/incomplete maps cannot pass by omitting a dependency entirely.
    engine_dependency_hashes({"sources": freeze.get("implementation")})
    current_implementation = runtime_fingerprint(repo)["sources"]
    if set(freeze["implementation"]) != set(current_implementation):
        raise ValueError("implementation source inventory changed after confirmation freeze")
    for relative, digest in freeze["implementation"].items():
        if current_implementation[relative] != digest:
            raise ValueError(f"implementation changed after confirmation freeze: {relative}")
    for identifier, digest in freeze["source_specifications"].items():
        source_path = campaign / "specs" / f"{identifier}.json"
        if file_sha256(source_path) != digest:
            raise ValueError("source specification changed after freeze")
        source = read_json(source_path)
        source_jobs = {job["job_id"]: job for job in source["jobs"]}
        if len(source_jobs) != len(source["jobs"]):
            raise ValueError("source specification contains duplicate job identities")
        source_paths = _source_result_paths(campaign, identifier, source_jobs)
        expected_paths = {str(path.relative_to(campaign)) for path in source_paths}
        for config in freeze["candidate_configurations"].values():
            if config["source_experiment_id"] == identifier and set(config["source_result_hashes"]) != expected_paths:
                raise ValueError("frozen source-result manifest must cover every source condition")
    expected_source_results = {path: digest for config in freeze["candidate_configurations"].values()
                               for path, digest in config["source_result_hashes"].items()}
    if freeze.get("source_results") != expected_source_results:
        raise ValueError("frozen source-result manifests disagree")
    for relative, digest in expected_source_results.items():
        path = campaign / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("source result changed or disappeared after freeze")
    if _model_identity(campaign) != freeze["model_identity"]:
        raise ValueError("local model identity changed after freeze")
    if freeze["analysis_configuration"] != ANALYSIS_CONFIG:
        raise ValueError("statistical configuration differs from the frozen protocol")
    candidate = freeze["candidate_configurations"][spec["candidate_id"]]
    is_input = candidate.get("protocol_kind") == "input_fact"
    mixed = _mixed_family_contract(candidate, candidate["n_layouts"])
    if mixed and not is_input:
        raise ValueError("mixed-family confirmation is supported only for input_fact")
    selected_families = set(mixed["families"]) if mixed else {candidate["family"]}
    if is_input and (candidate.get("claim_scope") != "input_fact_information_routing"
                     or candidate.get("semantic_headline_eligible") is not False):
        raise ValueError("input confirmation cannot promote its frozen claim scope")
    sibling_specs = [read_json(path) for path in (campaign / "specs").glob("*.json")]
    duplicates = [item for item in sibling_specs if item.get("partition") == "confirmation"
                  and item.get("freeze_id") == spec["freeze_id"] and item.get("candidate_id") == spec["candidate_id"]
                  and item["experiment_id"] != spec["experiment_id"]]
    if duplicates:
        raise ValueError("a frozen candidate may be confirmed in only one experiment")
    # A sibling candidate can be registered after the common freeze. Its
    # layouts are therefore absent from freeze-time exclusions but still may
    # not be queried again as independent confirmation evidence.
    later_exclusions = set()
    for item in sibling_specs:
        if item.get("experiment_id") == spec["experiment_id"]:
            continue
        for sibling_job in item.get("jobs", []):
            later_exclusions.add(sibling_job["layout_id"])
            later_exclusions.update(sibling_job.get("layout_aliases", []))
        for aliases in item.get("layout_aliases", {}).values():
            later_exclusions.update(aliases)
    for dataset_path in (campaign / "datasets").glob("*.json"):
        if dataset_path.stem == spec["experiment_id"]:
            continue
        dataset = read_json(dataset_path)
        for pair in dataset.get("pairs", []):
            later_exclusions.add(pair["layout_id"])
            later_exclusions.update(pair.get("metadata", {}).get("variant_layout_ids", []))
        for aliases in dataset.get("layout_aliases", {}).values():
            later_exclusions.update(aliases)
    jobs = spec["jobs"]
    if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("confirmation needs unique nonempty execution jobs")
    pairs = {}
    for job in jobs:
        pair = job.get("metadata", {}).get("source_pair" if is_input else "pair")
        if not isinstance(pair, dict) or pair["family"] not in selected_families or pair["partition"] != "confirmation":
            raise ValueError("confirmation jobs require the frozen family's fresh pair records")
        if partition_for_layout(pair["layout_id"]) != "confirmation":
            raise ValueError("confirmation layout was reassigned from another partition")
        if pair["pair_id"] in pairs and fingerprint(pairs[pair["pair_id"]]) != fingerprint(pair):
            raise ValueError("inconsistent pair metadata within confirmation jobs")
        pairs[pair["pair_id"]] = pair
        if {pair["layout_id"], *job.get("layout_aliases", []), *pair.get("metadata", {}).get("variant_layout_ids", [])} & set(freeze["excluded_layout_ids"]):
            raise ValueError("confirmation layout or one of its variants was previously used")
        if {pair["layout_id"], *job.get("layout_aliases", []), *pair.get("metadata", {}).get("variant_layout_ids", [])} & later_exclusions:
            raise ValueError("confirmation layout or alias overlaps another registered cohort after freeze")
    if len(pairs) != candidate["n_layouts"] or len({pair["layout_id"] for pair in pairs.values()}) != candidate["n_layouts"]:
        raise ValueError("confirmation pair/layout count differs from freeze")
    if mixed:
        counts = {f: sum(pair["family"] == f for pair in pairs.values()) for f in MIXED_FAMILIES}
        if counts != mixed["family_layout_counts"]:
            raise ValueError("confirmation family quotas differ from the frozen balanced cohort")
    if is_input:
        from .confirmation_adapters import build_frozen_input_spec
        expected = build_frozen_input_spec(list(pairs.values()), _confirmation_tokenizer(campaign),
                                           spec["experiment_id"], candidate, spec["freeze_id"])
        if expected.get("tokenizer") != candidate["tokenizer"]:
            raise ValueError("confirmation tokenizer metadata differs from the frozen source")
        if spec.get("claim_scope") != "input_fact_information_routing" or spec.get("semantic_headline_eligible") is not False:
            raise ValueError("input confirmation cannot promote its frozen claim scope")
    else:
        expected = build_controlled_state_spec(
            list(pairs.values()), spec["experiment_id"], layers=candidate["layers"], site=candidate["site"],
            last_k=candidate["last_k"], seeds=candidate["generation_seeds"],
            max_new_tokens=candidate["max_new_tokens"], natural=True)
    actual_jobs = {job["job_id"]: fingerprint(job) for job in jobs}
    expected_jobs = {job["job_id"]: fingerprint(job) for job in expected["jobs"]}
    if actual_jobs != expected_jobs:
        raise ValueError("confirmation jobs differ from the frozen intervention/control/seed contract")
    primary = [item for item in spec["comparisons"] if item.get("confirmatory", True)]
    if len(primary) != 1:
        raise ValueError("each frozen candidate must have exactly one primary comparison")
    chosen = primary[0]
    if chosen["treatment"] != candidate["treatment_condition"] or chosen["control"] != candidate["control_condition"]:
        raise ValueError("confirmation comparison differs from the frozen selection")
    expected_comparisons = {item["id"]: item for item in expected["comparisons"]}
    for comparison in spec["comparisons"]:
        expected_comparison = expected_comparisons.get(comparison["id"])
        if expected_comparison is None or any(comparison.get(key) != value for key, value in expected_comparison.items()):
            raise ValueError("confirmation comparison eligibility or endpoint differs from the rebuilt specification")
    if chosen["expected_n_layouts"] != candidate["n_layouts"] or chosen["expected_n_pairs"] != candidate["n_layouts"] * len(candidate["generation_seeds"]):
        raise ValueError("primary confirmation comparison does not include every frozen layout/seed")
    if mixed:
        expected_pairs_by_family = {f: n * len(candidate["generation_seeds"])
                                    for f, n in mixed["family_layout_counts"].items()}
        if (chosen.get("resampling_scheme") != "within_family_layout_bootstrap"
                or chosen.get("stratification_key") != "family"
                or chosen.get("family_layout_counts") != mixed["family_layout_counts"]
                or chosen.get("family_weights") != mixed["family_weights"]
                or chosen.get("expected_n_layouts_by_family") != mixed["family_layout_counts"]
                or chosen.get("expected_n_pairs_by_family") != expected_pairs_by_family):
            raise ValueError("mixed primary must preserve frozen family quotas, weights and resampling")
