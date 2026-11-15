"""Exploratory interventions on genuine, generated pre-mention reasoning tokens.

Token IDs are authoritative. Decoding locates conservative lexical/analysis
boundaries; generated text is never re-encoded to construct an input.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Iterable

from .registry import SCHEMA_VERSION, fingerprint

ACTION_MENTION = re.compile(r"(?<!\w)(?:UP|DOWN|LEFT|RIGHT)(?!\w)", re.IGNORECASE)
ANALYSIS_BOUNDARIES = ("<|end|>", "<|start|>", "<|channel|>", "<|message|>",
                       "<|fim_suffix|>", "<|endoftext|>", "<|return|>")
SOURCE_CONDITIONS = {"baseline", "counterfactual_text", "natural_donor_text", "donor_text"}
POPULATION_SCOPE = (
    "Fresh simulator-selected random-wall grids (default sizes 7 and 9, wall density 0.23), "
    "not a replication sample from the historical two-by-two room-layout distribution. "
    "Transition mechanics match the original task; prefix eligibility further selects "
    "naturally generated traces long enough to reach the cut before any direction-word mention."
)


def _valid_ids(values: Any) -> bool:
    return isinstance(values, (list, tuple)) and all(type(value) is int and value >= 0 for value in values)


def _encode_base(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    values = list(encoded)
    if not values or not _valid_ids(values):
        raise ValueError("Tokenizer returned invalid base token IDs")
    return values


def _decode(tokenizer: Any, values: list[int]) -> str:
    try:
        return tokenizer.decode(values, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(values)


def _source_prefix(job: dict[str, Any], result: dict[str, Any] | None, tokenizer: Any,
                   cut: int, last_k: int) -> dict[str, Any]:
    """Audit a source output without using its action, probability, or success."""
    audit: dict[str, Any] = {"eligible": False, "source_job_id": job["job_id"], "source_condition": job["condition"],
                             "cut": cut, "last_k": last_k, "selection_uses_model_action": False}

    def reject(reason: str) -> dict[str, Any]:
        return {**audit, "reason": reason}

    if cut < last_k:
        return reject("window_would_include_prompt_tokens")
    if job.get("intervention", {}).get("layers"):
        return reject("source_is_not_unpatched_baseline")
    if not job.get("natural", False):
        return reject("source_did_not_request_natural_generation")
    if result is None:
        return reject("source_result_missing")
    for field in ("experiment_id", "condition", "layout_id", "pair_id", "seed", "target_action"):
        if result.get(field) != job.get(field):
            return reject(f"source_result_provenance_mismatch_{field}")
    if result.get("status") != "complete" or result.get("generation_status") not in {"complete", "truncated", "malformed"}:
        return reject("source_execution_failed_or_has_no_native_generation")
    generated = result.get("generated_token_ids")
    if not _valid_ids(generated):
        return reject("invalid_or_missing_generated_token_ids")
    generated = list(generated)
    audit["source_generated_tokens"] = len(generated)
    if len(generated) < cut:
        return reject("insufficient_generated_tokens")
    if job.get("recipient_token_ids") is not None:
        if not _valid_ids(job["recipient_token_ids"]) or not job["recipient_token_ids"]:
            return reject("invalid_base_token_override")
        base_ids = list(job["recipient_token_ids"])
    else:
        base_ids = _encode_base(tokenizer, job["recipient_text"])
    recorded_hash = result.get("input_token_sha256")
    if recorded_hash is None:
        return reject("missing_source_input_token_fingerprint")
    if fingerprint(base_ids) != recorded_hash:
        return reject("source_input_token_fingerprint_mismatch")
    base_decoded = _decode(tokenizer, base_ids)
    analysis_header = "<|channel|>analysis<|message|>"
    header_position = base_decoded.rfind(analysis_header)
    if header_position < 0 or any(marker in base_decoded[header_position + len(analysis_header):]
                                  for marker in ANALYSIS_BOUNDARIES):
        return reject("source_input_is_not_in_analysis_channel")
    # Complete decoding is only a boundary detector, never input reconstruction.
    full_text = _decode(tokenizer, generated)
    boundary_positions = [(full_text.find(marker), marker) for marker in ANALYSIS_BOUNDARIES if marker in full_text]
    boundary_at, boundary_marker = min(boundary_positions) if boundary_positions else (len(full_text), None)
    analysis_text = full_text[:boundary_at]
    mention = ACTION_MENTION.search(analysis_text)
    mention_at = mention.start() if mention else None
    prefix_ids = generated[:cut]
    prefix_text = _decode(tokenizer, prefix_ids)
    if not full_text.startswith(prefix_text):
        return reject("ambiguous_decoding_boundary")
    audit.update({"analysis_boundary_character": boundary_at, "analysis_boundary_marker": boundary_marker,
                  "first_direction_mention_character": mention_at,
                  "first_direction_mention": mention.group(0) if mention else None,
                  "source_generation_status": result["generation_status"], "base_token_count": len(base_ids),
                  "source_input_token_sha256": recorded_hash, "source_result_fingerprint": fingerprint(result)})
    if len(prefix_text) > boundary_at:
        return reject("cut_reaches_analysis_end_or_channel_marker")
    if mention_at is not None and len(prefix_text) > mention_at:
        return reject("cut_reaches_first_direction_word")
    combined = base_ids + prefix_ids
    positions = list(range(len(combined) - last_k, len(combined)))
    if min(positions) < len(base_ids):
        return reject("window_would_include_prompt_tokens")
    return {**audit, "eligible": True, "reason": None, "token_ids": combined,
            "generated_prefix_token_ids": prefix_ids, "patch_positions": positions,
            "display_text": job["recipient_text"] + prefix_text,
            "token_ids_authoritative": True}


def build_reasoning_window_spec(
    source_spec: dict[str, Any], result_rows: Iterable[dict[str, Any]], tokenizer: Any,
    experiment_id: str, cuts: Iterable[int] = (32, 128), layers: Iterable[int] = (10, 15),
    last_k: int = 12, natural: bool = True, max_new_tokens: int = 2048,
    include_all_layers: bool = True,
) -> dict[str, Any]:
    """Build discovery-only CoT-window jobs from matched baseline outputs.

    A pair/cut/donor cohort is eligible only if every seed predeclared in its
    recipient baseline has both usable source prefixes. Incomplete seed cohorts
    remain in the manifest. There is no pre-reasoning fallback or confirmation
    support, and these full-state transfers carry no semantic-headline claim.
    """
    from .campaign import make_job

    if source_spec.get("partition") != "discovery":
        raise ValueError("Reasoning-window selection is discovery-only; confirmation is prohibited")
    cuts, layers = tuple(cuts), tuple(layers)
    if not cuts or len(set(cuts)) != len(cuts) or any(type(cut) is not int or cut < 1 for cut in cuts):
        raise ValueError("cuts must be unique positive token counts")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers):
        raise ValueError("layers must be unique indices in 0..23")
    if type(last_k) is not int or last_k < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Invalid token counts")
    if type(natural) is not bool:
        raise ValueError("natural must be Boolean")
    if type(include_all_layers) is not bool:
        raise ValueError("include_all_layers must be Boolean")
    if not experiment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in experiment_id):
        raise ValueError("Invalid experiment_id")
    source_jobs = {job["job_id"]: job for job in source_spec["jobs"]}
    if len(source_jobs) != len(source_spec["jobs"]):
        raise ValueError("Source specification contains duplicate job IDs")
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    baseline_seeds: dict[str, set[int]] = defaultdict(set)
    for job in source_jobs.values():
        if job["condition"] not in SOURCE_CONDITIONS:
            continue
        if job["partition"] != "discovery":
            raise ValueError("Source jobs disagree with discovery partition")
        key = str(job["pair_id"]), int(job["seed"]), job["condition"]
        if key in indexed:
            raise ValueError("Source specification repeats a pair/seed/condition")
        indexed[key] = job
        if job["condition"] == "baseline":
            baseline_seeds[key[0]].add(key[1])
    results = {}
    for row in result_rows:
        if row.get("job_id") not in source_jobs:
            continue
        if row["job_id"] in results:
            raise ValueError("Duplicate source result job_id")
        results[row["job_id"]] = row
    jobs, eligibility_manifest = [], []
    condition_pairs: dict[str, set[str]] = defaultdict(set)
    pair_seeds: dict[str, list[int]] = {}
    pair_layout: dict[str, str] = {}
    source_conditions = {job["condition"] for job in indexed.values()}
    matched_source = "counterfactual_text" if "counterfactual_text" in source_conditions else "donor_text"
    donors = [("matched", matched_source)]
    if "natural_donor_text" in source_conditions:
        donors.append(("natural", "natural_donor_text"))
    base_ids_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def source_audit(pair_id: str, seed: int, condition: str, cut: int) -> dict[str, Any]:
        job = indexed.get((pair_id, seed, condition))
        if job is None:
            return {"eligible": False, "reason": "source_job_not_planned", "source_condition": condition, "cut": cut}
        key = job["job_id"], cut
        if key not in base_ids_cache:
            base_ids_cache[key] = _source_prefix(job, results.get(job["job_id"]), tokenizer, cut, last_k)
        return base_ids_cache[key]

    for original_pair_id, seed_set in sorted(baseline_seeds.items()):
        seeds = sorted(seed_set)
        template = indexed[(original_pair_id, seeds[0], "baseline")]
        for cut in cuts:
            pair_id = f"{original_pair_id}::cut{cut}"
            pair_seeds[pair_id] = seeds
            pair_layout[pair_id] = template["layout_id"]
            audits = {seed: source_audit(original_pair_id, seed, "baseline", cut) for seed in seeds}
            donor_audits = {name: {seed: source_audit(original_pair_id, seed, condition, cut) for seed in seeds}
                            for name, condition in donors}
            donor_eligible = {name: all(audits[seed]["eligible"] and values[seed]["eligible"] for seed in seeds)
                              for name, values in donor_audits.items()}
            manifest = {"pair_id": pair_id, "source_pair_id": original_pair_id, "layout_id": template["layout_id"],
                        "cut": cut, "seeds": seeds, "requires_all_predeclared_seeds": True,
                        "recipient_sources": audits, "donor_sources": donor_audits, "eligible_donor_cohorts": donor_eligible,
                        "selection_basis": "baseline_trace_length_and_conservative_lexical_cut_only",
                        "semantic_headline_eligible": False}
            # JSON maps use strings from the outset, preserving round-trip identity.
            manifest["recipient_sources"] = {str(k): v for k, v in audits.items()}
            manifest["donor_sources"] = {name: {str(k): v for k, v in values.items()} for name, values in donor_audits.items()}
            eligibility_manifest.append(manifest)
            if not any(donor_eligible.values()):
                continue

            def add(condition: str, recipient_audits: dict[int, dict[str, Any]],
                    donor_values: dict[int, dict[str, Any]] | None, patch_layers: list[int],
                    source_condition: str, donor_condition: str | None = None) -> None:
                condition_pairs[condition].add(pair_id)
                for seed in seeds:
                    recipient_audit = recipient_audits[seed]
                    donor_audit = donor_values[seed] if donor_values else None
                    source_job = indexed[(original_pair_id, seed, source_condition)]
                    if source_job["target_action"] != template["target_action"]:
                        raise ValueError("Source baseline target actions are inconsistent")
                    if donor_condition is not None:
                        donor_job = indexed[(original_pair_id, seed, donor_condition)]
                        if donor_job["target_action"] != template["target_action"]:
                            raise ValueError("Donor source target action differs from recipient's frozen target")
                    jobs.append(make_job(
                        experiment_id, layout_id=template["layout_id"], pair_id=pair_id, family=template["family"],
                        partition="discovery", condition=condition, target_action=template["target_action"],
                        original_action=template.get("original_action"), seed=seed,
                        recipient_text=recipient_audit["display_text"],
                        donor_text=donor_audit["display_text"] if donor_audit else None,
                        recipient_token_ids=recipient_audit["token_ids"],
                        donor_token_ids=donor_audit["token_ids"] if donor_audit else None,
                        selection={"scope": "last_tokens", "last_k": last_k},
                        intervention={"layers": patch_layers, "site": "resid_post", "operation": "replace", "dose": 1.0},
                        natural=natural, max_new_tokens=max_new_tokens, temperature=source_job.get("temperature", 0.7),
                        semantic_test=False, layout_aliases=deepcopy(template.get("layout_aliases", [template["layout_id"]])),
                        metadata={"source_experiment_id": source_spec["experiment_id"], "source_pair_id": original_pair_id,
                                  "pair": deepcopy(template.get("metadata", {}).get("pair", {})),
                                  "cut": cut, "intervention_time": "generated_reasoning_before_first_lexical_direction_word",
                                  "recipient_source_job_id": recipient_audit["source_job_id"],
                                  "donor_source_job_id": donor_audit["source_job_id"] if donor_audit else None,
                                  "recipient_base_token_count": recipient_audit["base_token_count"],
                                  "donor_base_token_count": donor_audit["base_token_count"] if donor_audit else None,
                                  "recipient_patch_positions": recipient_audit["patch_positions"],
                                  "donor_patch_positions": donor_audit["patch_positions"] if donor_audit else None,
                                  "genuinely_generated_window": True, "token_ids_authoritative": True,
                                  "all_layers_window_transfer": len(patch_layers) == 24,
                                  "full_input_state_replaced": False, "donor_continuation_identity_expected": False,
                                  "generated_text_retokenized": False, "semantic_headline_eligible": False,
                                  "lexical_criterion": "case-insensitive standalone UP/DOWN/LEFT/RIGHT; includes idioms and negation",
                                  "population_scope": POPULATION_SCOPE, "source_baseline_condition": source_condition,
                                  "donor_baseline_condition": donor_condition, "source_success_used_for_selection": False},
                    ))

            add(f"baseline_cut{cut}", audits, None, [], "baseline")
            add(f"identity_cut{cut}", audits, audits, [layers[0]], "baseline", "baseline")
            for donor_name, donor_condition in donors:
                if not donor_eligible[donor_name]:
                    continue
                donor_values = donor_audits[donor_name]
                infix = "" if donor_name == "matched" else "natural_"
                add(f"{infix}donor_cut{cut}", donor_values, None, [], donor_condition)
                for layer in layers:
                    name = f"layer{layer}_{infix}cut{cut}"
                    add(name, audits, donor_values, [layer], "baseline", donor_condition)
                if donor_name == "matched" and include_all_layers:
                    add(f"all_layers_full_cut{cut}", audits, donor_values, list(range(24)), "baseline", donor_condition)
    comparisons = []
    for condition, ids in sorted(condition_pairs.items()):
        if not (condition.startswith("layer") or condition.startswith("identity") or condition.startswith("all_layers_full")):
            continue
        cut = int(condition.rsplit("cut", 1)[1])
        comparisons.append({"id": f"{experiment_id}_{condition}", "experiment_id": experiment_id,
                            "partition": "discovery", "treatment": condition, "control": f"baseline_cut{cut}",
                            "eligible_pair_ids": sorted(ids), "expected_n_layouts": len({pair_layout[key] for key in ids}),
                            "expected_n_pairs": sum(len(pair_seeds[key]) for key in ids),
                            "eligibility_basis": "all_planned_baseline_seeds_have_premention_generated_windows",
                            "predicted_direction": "zero" if condition.startswith("identity") else "increase",
                            "role": "all_layer_generated_window_sufficiency_reference" if condition.startswith("all_layers_full") else "generated_window_intervention",
                            "donor_continuation_identity_expected": False,
                            "semantic_headline_eligible": False, "exploratory": True})
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "discovery",
            "hypothesis": "Exploratory transfer of genuine pre-mention reasoning-window states may redirect continuation; whole-state effects do not isolate a semantic belief mechanism.",
            "primary_estimand": "layout-weighted paired native target-action ITT probability difference",
            "source_experiment_id": source_spec["experiment_id"], "source_spec_fingerprint": fingerprint(source_spec),
            "source_result_fingerprints": {key: fingerprint(value) for key, value in sorted(results.items())},
            "jobs": jobs, "comparisons": comparisons, "eligibility_manifest": eligibility_manifest,
            "protocol": {"cuts": list(cuts), "layers": list(layers), "last_k": last_k,
                         "include_all_layers": include_all_layers,
                         "all_layers_scope": "same last_k generated analysis positions; not the entire differing input context",
                         "all_selected_seeds_required": True, "token_ids_authoritative": True,
                         "requires_token_id_override_support": True, "generated_text_retokenized": False,
                         "analysis_boundary_rule": list(ANALYSIS_BOUNDARIES),
                         "first_mention_rule": ACTION_MENTION.pattern, "case_insensitive": True,
                         "semantic_headline_eligible": False, "population_scope": POPULATION_SCOPE}}
