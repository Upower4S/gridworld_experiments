"""Data-only preparation of controlled discovery and frozen confirmation jobs."""
from __future__ import annotations

from pathlib import Path

from .environments import generate_pairs
from .experiments import build_controlled_state_spec
from .registry import append_event, file_sha256, immutable_json, read_json, register_spec
from .selection import excluded_layout_ids, validate_confirmation_ready


def prepare_controlled(campaign: Path, repo: Path, *, experiment_id: str,
                       source_dataset: str | None = None, n_layouts: int = 12,
                       source_offset: int = 0,
                       seed: int = 20260913, layers: tuple = (3, 10, 15, 18),
                       site: str = "resid_post", last_k: int = 1,
                       seeds: tuple = (42,), natural: bool = True,
                       max_new_tokens: int = 2048) -> dict:
    if source_dataset:
        source = read_json(campaign / "datasets" / f"{source_dataset}.json")
        if source["partition"] != "discovery":
            raise ValueError("controlled discovery cannot reuse confirmation cases")
        if type(source_offset) is not int or source_offset < 0:
            raise ValueError("source_offset must be a nonnegative integer")
        pairs = source["pairs"][source_offset:source_offset + n_layouts]
        if len(pairs) != n_layouts:
            raise ValueError("not enough layouts in source discovery dataset")
    else:
        pairs = generate_pairs(n_layouts, seed, partition="discovery",
                               historical_steps=repo / "outputs/canonical_steps.parquet",
                               excluded_layout_ids=excluded_layout_ids(campaign))
    spec = build_controlled_state_spec(pairs, experiment_id, layers=layers, site=site,
                                      last_k=last_k, seeds=seeds, natural=natural,
                                      max_new_tokens=max_new_tokens)
    spec["source_dataset"] = source_dataset
    spec["source_offset"] = source_offset
    dataset = {"partition": "discovery", "pairs": pairs, "layout_aliases": spec["layout_aliases"],
               "source_dataset": source_dataset, "source_offset": source_offset}
    immutable_json(campaign / "datasets" / f"{experiment_id}.json", dataset)
    register_spec(campaign, spec)
    return spec


def prepare_extension(campaign: Path, *, kind: str, source: str, experiment_id: str,
                      n_layouts: int = 12, layers: tuple = (10, 15), last_k: int = 12,
                      cuts: tuple = (32, 128), natural: bool = True,
                      max_new_tokens: int = 4096) -> dict:
    """Register exact token inputs for a discovery-only intervention family."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(read_json(campaign / "campaign.json")["model_path"],
                                               local_files_only=True)
    if kind in {"reasoning", "reasoned-fact"}:
        from .reasoning_windows import build_reasoning_window_spec
        from .campaign import collect_results
        source_path = campaign / "specs" / f"{source}.json"
        original = read_json(source_path)
        rows = collect_results(campaign, source)
        if any(row.get("spec_sha256") != file_sha256(source_path) for row in rows):
            raise ValueError("reasoning source result/spec provenance mismatch")
        layouts = list(dict.fromkeys(j["layout_id"] for j in original["jobs"]))[:n_layouts]
        selected = {**original, "jobs": [j for j in original["jobs"] if j["layout_id"] in layouts]}
        selected_ids = {j["job_id"] for j in selected["jobs"]}
        selected_rows = [r for r in rows if r["job_id"] in selected_ids]
        if kind == "reasoning":
            spec = build_reasoning_window_spec(selected, selected_rows, tokenizer,
                                                experiment_id, cuts=cuts, layers=layers, last_k=last_k,
                                                natural=natural, max_new_tokens=max_new_tokens)
        else:
            from .reasoned_facts import build_reasoned_fact_spec
            spec = build_reasoned_fact_spec(selected, selected_rows, tokenizer, experiment_id,
                                            cuts=cuts, layers=layers, last_k=last_k,
                                            natural=natural, cap=max_new_tokens)
        spec["source_native_spec_sha256"] = file_sha256(source_path)
        spec["source_layout_selection"] = {"requested": n_layouts, "selected_layout_ids": layouts,
                                            "method": "first whole layout cohorts in registered source order"}
    else:
        original = read_json(campaign / "datasets" / f"{source}.json")
        if original["partition"] != "discovery":
            raise ValueError("extension sweeps can only use discovery layouts")
        pairs = original["pairs"][:n_layouts]
        if len(pairs) != n_layouts:
            raise ValueError("not enough source discovery layouts")
        if kind == "input":
            from .input_sites import build_input_site_spec
            spec = build_input_site_spec(pairs, tokenizer, experiment_id, layers=layers,
                                          natural=natural, cap=max_new_tokens)
        elif kind == "contextual-input":
            from .contextual_input import build_contextual_input_spec
            spec = build_contextual_input_spec(pairs, tokenizer, experiment_id, layers=layers,
                                                natural=natural, cap=max_new_tokens)
        elif kind == "fact":
            from .fact_interventions import build_fact_spec
            spec = build_fact_spec(pairs, tokenizer, experiment_id, layers=layers, last_k=last_k,
                                    natural=natural, max_new_tokens=max_new_tokens)
        else:
            raise ValueError("unsupported extension family")
    spec["source_dataset_or_experiment"] = source
    if not spec["jobs"]:
        immutable_json(campaign / "reviews" / f"{experiment_id}_ineligible.json", spec)
        return spec
    register_spec(campaign, spec)
    return spec


def prepare_confirmation(campaign: Path, repo: Path, *, freeze_id: str,
                         candidate_id: str, experiment_id: str) -> dict:
    """Construct only the frozen configuration, never an open confirmation sweep."""
    freeze = read_json(campaign / "freezes" / f"{freeze_id}.json")
    candidate = freeze["candidate_configurations"][candidate_id]
    for relative, digest in freeze["implementation"].items():
        if file_sha256(repo / relative) != digest:
            raise ValueError(f"implementation changed since freeze: {relative}")
    tokenizer = None
    if candidate.get("protocol_kind") == "input_fact":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(read_json(campaign / "campaign.json")["model_path"],
                                                  local_files_only=True)
    pairs, exclusions = _confirmation_pairs(campaign, repo, candidate, experiment_id,
                                            freeze_id=freeze_id, tokenizer=tokenizer)
    spec = _build_candidate_spec(pairs, candidate, experiment_id, freeze_id, tokenizer)
    spec.update({"freeze_id": freeze_id, "candidate_id": candidate_id,
                 "source_experiment_id": candidate["source_experiment_id"]})
    chosen = 0
    for comparison in spec["comparisons"]:
        primary = comparison["treatment"] == candidate["treatment_condition"] and comparison["control"] == candidate["control_condition"]
        comparison["confirmatory"] = primary
        if primary:
            chosen += 1
            comparison["evidence"] = {"technical_valid": True, "fresh_confirmation": True, "frozen_specification": True,
                                      "semantic_specificity": False, "donor_action_mismatch": False, "independent_replication": False}
            comparison["evidence_note"] = "Semantic specificity and independent replication require a later separate reviewer artifact; the unreviewed result cannot claim semantic support."
    if chosen != 1:
        raise ValueError("frozen primary comparison is absent or ambiguous")
    validate_confirmation_ready(campaign, repo, spec)
    dataset = {"partition": "confirmation", "pairs": pairs, "layout_aliases": spec["layout_aliases"],
               "freeze_id": freeze_id, "candidate_id": candidate_id, "pre_query_exclusions": exclusions}
    immutable_json(campaign / "datasets" / f"{experiment_id}.json", dataset)
    register_spec(campaign, spec)
    append_event(campaign, "confirmation_dataset_prepared_unqueried", freeze_id=freeze_id, candidate_id=candidate_id, experiment_id=experiment_id, layouts=len(pairs))
    return spec


def _build_candidate_spec(pairs: list, candidate: dict, experiment_id: str,
                          freeze_id: str, tokenizer=None) -> dict:
    if candidate.get("protocol_kind", "controlled_state") == "input_fact":
        from .confirmation_adapters import build_frozen_input_spec
        if tokenizer is None:
            raise ValueError("input confirmation requires the frozen local tokenizer")
        return build_frozen_input_spec(pairs, tokenizer, experiment_id, candidate, freeze_id)
    if candidate.get("protocol_kind", "controlled_state") != "controlled_state":
        raise ValueError("no independently reviewed adapter for this confirmation protocol")
    return build_controlled_state_spec(pairs, experiment_id, layers=candidate["layers"],
                                       site=candidate["site"], last_k=candidate["last_k"],
                                       seeds=candidate["generation_seeds"], natural=True,
                                       max_new_tokens=candidate["max_new_tokens"])


def _confirmation_pairs(campaign: Path, repo: Path, candidate: dict, experiment_id: str, *,
                        freeze_id: str = "planning", tokenizer=None) -> tuple[list, list]:
    """Fill the frozen cohort using simulator and token eligibility before querying."""
    if "families" in candidate:
        return _confirmation_pairs_mixed(campaign, repo, candidate, experiment_id,
                                         freeze_id=freeze_id, tokenizer=tokenizer)
    pairs, exclusions = [], []
    excluded = excluded_layout_ids(campaign)
    for attempt in range(100):
        batch = generate_pairs(candidate["n_layouts"] - len(pairs), candidate["dataset_seed"] + attempt,
                               partition="confirmation", families=(candidate["family"],),
                               historical_steps=repo / "outputs/canonical_steps.parquet",
                               excluded_layout_ids=excluded)
        trial = _build_candidate_spec(batch, candidate, experiment_id, freeze_id, tokenizer)
        primary = [c for c in trial["comparisons"] if c["treatment"] == candidate["treatment_condition"]
                   and c["control"] == candidate["control_condition"]]
        eligible = set(primary[0]["eligible_pair_ids"]) if len(primary) == 1 else set()
        manifest = {e["pair_id"]: e for e in trial["eligibility_manifest"]}
        for pair in batch:
            aliases = {pair["layout_id"], *pair.get("metadata", {}).get("variant_layout_ids", []),
                       *trial["layout_aliases"].get(pair["layout_id"], [])}
            collision = sorted(aliases & excluded)
            if pair["pair_id"] in eligible and not collision:
                pairs.append(pair)
            else:
                exclusions.append({"pair": pair, "reason": "layout_or_control_alias_overlap" if collision else (
                    manifest[pair["pair_id"]].get("exclusion_reason") or "frozen_primary_control_unavailable"),
                                   "overlapping_aliases": collision,
                                   "eligibility": manifest[pair["pair_id"]],
                                   "selection_basis": (
                                       "simulator_and_exact_token_alignment_before_model_query"
                                       if candidate.get("protocol_kind") == "input_fact"
                                       else "simulator_only_before_model_query")})
            excluded.update(aliases)
        if len(pairs) == candidate["n_layouts"]:
            return pairs, exclusions
    raise ValueError("unable to fill the frozen simulator-eligible cohort in 100 deterministic batches")


def _confirmation_pairs_mixed(campaign: Path, repo: Path, candidate: dict,
                              experiment_id: str, *, freeze_id: str,
                              tokenizer) -> tuple[list, list]:
    """Refill each frozen family quota with global alias exclusions.

    The seed schedule is dataset_seed + 1000 * canonical family index + attempt
    (0..99). Eligibility uses the simulator and tokenizer only. A missing family
    cannot be replaced by extra cases from another family. Partial adapter calls
    expose actual counts; only the final exact quota cohort may pass readiness.
    """
    families = ["wall", "key_possession", "door_status"]
    counts = candidate.get("family_layout_counts")
    if (candidate.get("protocol_kind") != "input_fact" or candidate["families"] != families
            or candidate.get("family") is not None
            or not isinstance(counts, dict) or set(counts) != set(families)
            or any(type(value) is not int or value < 1 for value in counts.values())
            or candidate.get("n_layouts") not in {32, 64, 128}
            or sum(counts.values()) != candidate["n_layouts"]
            or max(counts.values()) - min(counts.values()) > 1):
        raise ValueError("mixed input confirmation requires exact balanced frozen family quotas")
    weights = {family: counts[family] / candidate["n_layouts"] for family in families}
    if "family_weights" in candidate and candidate["family_weights"] != weights:
        raise ValueError("frozen family weights disagree with layout quotas")
    if type(candidate.get("dataset_seed")) is not int or candidate["dataset_seed"] < 0:
        raise ValueError("mixed confirmation requires a nonnegative integer dataset seed")
    excluded = excluded_layout_ids(campaign)
    pairs, exclusions = [], []
    for family_index, family in enumerate(families):
        accepted = []
        for attempt in range(100):
            remaining = counts[family] - len(accepted)
            seed = candidate["dataset_seed"] + 1000 * family_index + attempt
            batch = generate_pairs(remaining, seed, partition="confirmation", families=(family,),
                                   historical_steps=repo / "outputs/canonical_steps.parquet",
                                   excluded_layout_ids=excluded)
            if len(batch) > remaining or any(pair.get("family") != family for pair in batch):
                raise ValueError("generator changed the requested confirmation family or quota")
            trial = _build_candidate_spec(batch, candidate, experiment_id, freeze_id, tokenizer)
            primary = [comparison for comparison in trial["comparisons"]
                       if comparison["treatment"] == candidate["treatment_condition"]
                       and comparison["control"] == candidate["control_condition"]]
            eligible = set(primary[0]["eligible_pair_ids"]) if len(primary) == 1 else set()
            manifest = {entry["pair_id"]: entry for entry in trial["eligibility_manifest"]}
            for pair in batch:
                aliases = {pair["layout_id"], *pair.get("metadata", {}).get("variant_layout_ids", []),
                           *trial["layout_aliases"].get(pair["layout_id"], [])}
                collision = sorted(aliases & excluded)
                if pair["pair_id"] in eligible and not collision:
                    accepted.append(pair)
                else:
                    exclusions.append({"pair": pair, "requested_family": family,
                        "generation_seed": seed, "refill_attempt": attempt,
                        "reason": "layout_or_control_alias_overlap" if collision else (
                            manifest[pair["pair_id"]].get("exclusion_reason") or "frozen_primary_control_unavailable"),
                        "overlapping_aliases": collision, "eligibility": manifest[pair["pair_id"]],
                        "selection_basis": "simulator_and_exact_token_alignment_before_model_query"})
                excluded.update(aliases)
            if len(accepted) == counts[family]:
                pairs.extend(accepted)
                break
        else:
            raise ValueError(f"unable to fill frozen {family} quota in 100 deterministic batches")
    return pairs, exclusions
