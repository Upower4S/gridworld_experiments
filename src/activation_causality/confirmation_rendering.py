"""Fixed canonical primary and paired alternate lexical-supplement builders.

Pure construction only: no selection, registration, file/model access or basis
refit. Frozen adapters and renderer implementations remain unchanged. Alternate
comparisons are descriptive, and their standard references reuse parent jobs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from . import frozen_contextual_input as contextual
from . import frozen_subspace as subspace
from .confirmation_sampling import select_lexical_subset
from .environments import canonical_layout_fingerprint
from .input_sites import _physical_counterfactual, aligned_changed_span
from .lexical_generalization import ALTERNATE, CANONICAL, audit_pair_renderings, render_with_proof, renderer_manifest
from .registry import fingerprint


VERSION = "fixed_confirmation_rendering_v1"
FAMILIES = contextual.FAMILIES


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def _normalize(configuration: Mapping[str, Any], tokenizer: Any, basis_metadata: Mapping[str, Any] | None) -> tuple[dict, dict | None]:
    _require(isinstance(configuration, Mapping), "frozen configuration required")
    config = deepcopy(dict(configuration))
    _require(_same(config.get("tokenizer"), contextual._tokenizer_metadata(tokenizer)), "frozen tokenizer metadata differs")
    _require(config.get("renderer_id") == CANONICAL and _same(config.get("renderer_binding"), renderer_manifest(CANONICAL)), "primary requires exact canonical renderer binding")
    _require(type(config.get("require_joint_renderer_eligibility")) is bool, "explicit joint renderer requirement required")
    if config.get("protocol_kind") == contextual.PROTOCOL:
        _require(basis_metadata is None, "contextual profile cannot receive a task basis")
        return contextual._configuration(config, tokenizer), None
    _require(config.get("protocol_kind") == subspace.PROTOCOL and isinstance(basis_metadata, Mapping), "subspace profile requires caller-verified existing basis")
    normalized, basis = subspace._configuration(config, basis_metadata)
    if "basis_metadata" in config:
        _require(_same(config["basis_metadata"], basis), "embedded and supplied basis metadata differ")
    return normalized, basis


def _aliases(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("layout_id"), str) and value["layout_id"]:
            result.add(value["layout_id"])
        for key in ("layout_aliases", "variant_layout_ids"):
            items = value.get(key, [])
            if isinstance(items, list):
                result.update(item for item in items if isinstance(item, str) and item)
        for nested in value.values():
            if isinstance(nested, (Mapping, list)):
                result.update(_aliases(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_aliases(nested))
    return result


def _pair_aliases(pair: Mapping[str, Any]) -> set[str]:
    aliases = _aliases(pair)
    for role in ("recipient", "counterfactual", "donor"):
        try:
            aliases.add(canonical_layout_fingerprint(pair[role]))
        except (ValueError, TypeError, KeyError, IndexError):
            pass  # Declared aliases remain available for rejected malformed inputs.
    return aliases


def _subspace_roles(condition: str) -> tuple[str, str | None]:
    if condition == "baseline":
        return "recipient", None
    if condition == "donor_text":
        return "counterfactual", None
    if condition == "subspace_removal" or condition.startswith("subspace_reverse_"):
        return "counterfactual", "recipient"
    if condition in {"subspace_identity", "restoration_equivalent"}:
        return "recipient", "recipient"
    _require(condition in subspace.CONDITIONS, "unknown frozen subspace condition")
    return "recipient", "counterfactual"


def _job(experiment_id: str, value: dict) -> dict:
    from .campaign import make_job
    return make_job(experiment_id, **{key: item for key, item in value.items() if key not in {"experiment_id", "job_id"}})


def _full_manifest(pairs: list[dict], original: list[dict], jobs: list[dict], config: dict,
                   joint_audits: dict[int, dict | None]) -> list[dict]:
    by_index = {entry["input_index"]: deepcopy(entry) for entry in original}
    _require(set(by_index) == set(range(len(pairs))), "adapter manifest does not cover every input pair")
    job_aliases = {}
    for job in jobs:
        job_aliases.setdefault(job["pair_id"], set()).update(job.get("layout_aliases", []))
    entries = []
    for index, pair in enumerate(pairs):
        entry = by_index[index]
        _require(all(entry.get(key) == pair.get(key) for key in ("pair_id", "layout_id", "family")), "adapter source identity mismatch")
        joint = joint_audits.get(index)
        if joint is not None and joint.get("eligible") is not True and entry["eligible"]:
            entry.update(eligible=False, reason="joint_renderer_alignment_failed", excluded_before_model_query=True, scheduled_conditions=[])
        aliases = _pair_aliases(pair) | _aliases(entry) | job_aliases.get(pair.get("pair_id"), set())
        entry.update(source_pair_sha256=fingerprint(pair), layout_aliases=sorted(aliases),
                     required_condition_inventory=deepcopy(config["condition_inventory"]), joint_renderer_audit=deepcopy(joint),
                     renderer_binding=renderer_manifest(CANONICAL),
                     joint_renderer_required=config["require_joint_renderer_eligibility"],
                     selection_before_model_query=True, outcome_dependent_exclusion_permitted=False)
        if entry["eligible"]:
            _require(entry["scheduled_conditions"] == config["condition_inventory"], "accepted pair omits controls")
            if config["require_joint_renderer_eligibility"]:
                _require(joint is not None and joint.get("eligible") is True and joint.get("pair_id") == pair["pair_id"], "accepted primary lacks complete joint proof")
        entries.append(entry)
    return entries


def build_primary(
    pairs: Sequence[Mapping[str, Any]], tokenizer: Any, eid: str, configuration: Mapping[str, Any],
    freeze_id: str, *, basis_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical primary plan with every original input/exclusion retained.

    Subspace joint eligibility uses the unchanged matched-state renderer audit;
    the same rank/dose/basis is subsequently used in both renderings. Partial
    planning cohorts keep actual counts and cannot be treated as ready.
    """
    _require(_name(eid) and _name(freeze_id), "canonical experiment/freeze identities required")
    config, basis = _normalize(configuration, tokenizer, basis_metadata)
    sources = deepcopy(list(pairs))
    _require(all(isinstance(pair, dict) for pair in sources), "structured source pairs required")
    joint_audits: dict[int, dict | None] = {}
    if basis is None:
        spec = contextual.build_frozen_contextual_input_spec(sources, tokenizer, eid, config, freeze_id)
        original_manifest = deepcopy(spec["eligibility_manifest"])
        for entry in original_manifest:
            joint_audits[entry["input_index"]] = entry.get("control_eligibility", {}).get("joint_renderer_audit")
    else:
        initial = subspace.build_frozen_subspace_spec(sources, tokenizer, eid, config, freeze_id, basis_metadata=basis)
        original_manifest = deepcopy(initial["eligibility_manifest"])
        retained = []
        for entry in original_manifest:
            index = entry["input_index"]
            joint = None
            if entry["eligible"] and config["require_joint_renderer_eligibility"]:
                try:
                    joint = audit_pair_renderings(sources[index], tokenizer, profile="matched")
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    joint = {"eligible": False, "pair_id": sources[index].get("pair_id"),
                             "reason": str(exc), "excluded_before_model_query": True}
            joint_audits[index] = joint
            if entry["eligible"] and (joint is None or joint["eligible"]):
                retained.append(sources[index])
        spec = subspace.build_frozen_subspace_spec(retained, tokenizer, eid, config, freeze_id, basis_metadata=basis)
        pair_audits = {sources[index]["pair_id"]: audit for index, audit in joint_audits.items()
                       if audit is not None and audit.get("eligible") is True}
        revised = []
        for job in spec["jobs"]:
            value = deepcopy(job)
            pair = value["metadata"]["source_pair"]
            r, d = _subspace_roles(value["condition"])
            audit = pair_audits.get(value["pair_id"])
            if audit is not None:
                for role, source_role in (("recipient", r), ("donor", d)):
                    if source_role is not None:
                        source_audit = audit["renderings"][CANONICAL][source_role]
                        _require(fingerprint(value[role + "_token_ids"]) == source_audit["input_token_sha256"]
                                 and value["selection"][role + "_positions"] == source_audit["span"]["recipient_positions"],
                                 "canonical adapter geometry differs from joint renderer audit")
            value["metadata"].update(renderer_id=CANONICAL, renderer_binding=renderer_manifest(CANONICAL),
                recipient_renderer_proof=render_with_proof(pair[r], CANONICAL)["proof"],
                donor_renderer_proof=render_with_proof(pair[d], CANONICAL)["proof"] if d else None,
                joint_renderer_audit_sha256=fingerprint(audit) if audit is not None else None,
                canonical_source_pair_preserved=True)
            revised.append(_job(eid, value))
        spec["jobs"] = revised
    manifest = _full_manifest(sources, original_manifest, spec["jobs"], config, joint_audits)
    accepted_indices = [entry["input_index"] for entry in manifest if entry["eligible"]]
    expected_ids = {sources[index]["pair_id"] for index in accepted_indices}
    _require({job["pair_id"] for job in spec["jobs"]} == expected_ids, "rendering eligibility and emitted cohort differ")
    for job in spec["jobs"]:
        indices = [index for index in accepted_indices if sources[index]["pair_id"] == job["pair_id"]]
        _require(len(indices) == 1 and _same(job["metadata"]["source_pair"], sources[indices[0]]), "source_pair lineage drift")
    exclusions = [deepcopy(entry) for entry in manifest if not entry["eligible"]]
    spec.update(wrapper_version=VERSION, cohort_role="primary", wrapper_configuration_sha256=fingerprint(config),
                wrapper_source_pairs=sources, wrapper_source_pairs_sha256=fingerprint(sources),
                source_pairs_sha256=fingerprint(sources), renderer_id=CANONICAL, renderer_binding=renderer_manifest(CANONICAL),
                eligibility_manifest=manifest, exclusions=exclusions,
                n_source_pairs=len(sources), n_excluded_pairs=len(exclusions),
                exclusion_counts=dict(sorted(Counter(row["reason"] for row in exclusions).items())),
                fixed_renderer_wrapper_required_for_rebuild=True)
    return spec


def eligibility_decisions(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose exact per-input decisions for the persisted sampler's callback.

    Caller supplies its authenticated eligibility_builder_sha256 around this
    list. No outcome information is accepted. Rejected aliases are retained.
    """
    _require(spec.get("wrapper_version") == VERSION and spec.get("cohort_role") == "primary", "requires fixed primary wrapper output")
    sources = spec.get("wrapper_source_pairs")
    _require(isinstance(sources, list) and fingerprint(sources) == spec.get("wrapper_source_pairs_sha256"), "source pair manifest fingerprint mismatch")
    manifest = spec.get("eligibility_manifest")
    _require(isinstance(manifest, list) and len(manifest) == len(sources), "one manifest entry per original source required")
    decisions = []
    for index, (pair, entry) in enumerate(zip(sources, manifest, strict=True)):
        _require(entry["input_index"] == index and entry["source_pair_sha256"] == fingerprint(pair), "per-input physical identity mismatch")
        decisions.append({"pair_id": pair.get("pair_id"), "layout_id": pair.get("layout_id"),
            "source_pair_sha256": fingerprint(pair), "eligible": entry["eligible"], "reason": entry["reason"],
            "layout_aliases": deepcopy(entry["layout_aliases"]), "evidence": {
                "adapter_manifest": deepcopy(entry), "required_condition_inventory": deepcopy(entry["required_condition_inventory"]),
                "joint_renderer_audit": deepcopy(entry["joint_renderer_audit"])}})
    return decisions


def _rendered_subspace_source(pair: dict, tokenizer: Any) -> dict:
    from .campaign import render_analysis_prompt
    normalized, _, _, _, _ = _physical_counterfactual(pair, partition="confirmation")
    values = {}
    for role in ("recipient", "counterfactual"):
        rendered = render_with_proof(normalized[role], ALTERNATE)
        text = render_analysis_prompt(rendered["prompt_text"])
        values[role] = {"state": normalized[role], "prompt_text": rendered["prompt_text"], "input_text": text,
                        "input_token_ids": contextual._encode(tokenizer, text), "renderer_proof": rendered["proof"]}
    span = aligned_changed_span(values["recipient"]["input_token_ids"], values["counterfactual"]["input_token_ids"])
    return {"format": "unregistered_inline_rendered_subspace_input_v1", "registered": False,
            "source_pair": deepcopy(pair), "source_pair_sha256": fingerprint(pair), "normalized_pair": normalized,
            "renderer_id": ALTERNATE, "renderer_binding": renderer_manifest(ALTERNATE), "sources": values, "input_fact_span": span}


def _alternate_subspace_job(parent_job: dict, source: dict, eid: str, parent_spec: dict) -> dict:
    value = deepcopy(parent_job)
    r, d = _subspace_roles(value["condition"])
    positions = source["input_fact_span"]["recipient_positions"]
    value.update(recipient_text=source["sources"][r]["input_text"],
                 recipient_token_ids=deepcopy(source["sources"][r]["input_token_ids"]),
                 donor_text=source["sources"][d]["input_text"] if d else None,
                 donor_token_ids=deepcopy(source["sources"][d]["input_token_ids"]) if d else None,
                 recipient_positions=deepcopy(positions), donor_positions=deepcopy(positions),
                 selection={"scope": "explicit", "recipient_positions": deepcopy(positions), "donor_positions": deepcopy(positions)})
    metadata = value["metadata"]
    for key in ("inline_input_source_definition_sha256", "inline_input_source_job_id", "inline_input_source_job_sha256", "source_lineage_kind"):
        metadata.pop(key, None)
    metadata.update(input_fact_span=deepcopy(source["input_fact_span"]), renderer_id=ALTERNATE,
        renderer_binding=renderer_manifest(ALTERNATE), recipient_renderer_proof=source["sources"][r]["renderer_proof"],
        donor_renderer_proof=source["sources"][d]["renderer_proof"] if d else None,
        recipient_patch_positions=deepcopy(positions), donor_patch_positions=deepcopy(positions),
        inline_rendered_source_definition_sha256=fingerprint(source),
        source_lineage_kind="unregistered_inline_fixed_renderer_source_with_bound_parent_job",
        canonical_source_pair_preserved=True, paired_standard_experiment_id=parent_spec["experiment_id"],
        paired_standard_job_id=parent_job["job_id"], paired_standard_job_sha256=fingerprint(parent_job),
        random_orientation_support_and_seed_shared_across_renderings=True)
    return _job(eid, value)


def _descriptive_comparisons(parent: dict, eid: str, subset: dict, *, standard_reference: bool) -> list[dict]:
    result = []
    for original in parent["comparisons"]:
        row = deepcopy(original)
        row.pop("primary_id", None)
        row.update(id=f"{eid}_{'standard_subset_' if standard_reference else ''}{row['treatment']}_vs_{row['control']}",
                   experiment_id=parent["experiment_id"] if standard_reference else eid, confirmatory=False,
                   expected_n_layouts=32, expected_n_pairs=32, eligible_pair_ids=sorted(subset["pair_ids"]),
                   source_parent_comparison_id=original["id"], lexical_same_layout_subset=True,
                   semantic_headline_eligible=False,
                   inferential_role="descriptive_standard_subset_reference" if standard_reference else "descriptive_lexical_supplement")
        if len(subset["families"]) == 3:
            row.update(families=deepcopy(subset["families"]), family_layout_counts=deepcopy(subset["family_layout_counts"]),
                       family_weights=deepcopy(subset["family_weights"]),
                       expected_n_layouts_by_family=deepcopy(subset["family_layout_counts"]),
                       expected_n_pairs_by_family=deepcopy(subset["family_layout_counts"]))
        result.append(row)
    return result


def build_lexical_supplement(
    parent_spec: Mapping[str, Any], tokenizer: Any, eid: str, configuration: Mapping[str, Any],
    freeze_id: str, *, basis_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Alternate-only execution on the exact fixed32 parent subset.

    Standard32 references bind existing parent jobs and receive the SAME subset
    family weights. No primary ID or confirmatory comparison survives here. The
    unchanged parent configuration is explicitly separate from the descriptive
    supplement configuration, so no new primary-family slots are implied.
    """
    _require(isinstance(parent_spec, Mapping) and _name(eid) and _name(freeze_id), "parent and canonical identities required")
    parent = deepcopy(dict(parent_spec))
    config, basis = _normalize(configuration, tokenizer, basis_metadata)
    _require(config["require_joint_renderer_eligibility"] is True, "supplement was not jointly eligible before primary queries")
    _require(parent.get("cohort_role") == "primary" and parent.get("wrapper_version") == VERSION
             and parent.get("cohort_complete") is True and parent.get("freeze_id") == freeze_id
             and parent.get("candidate_id") == config["candidate_id"] and parent.get("experiment_id") != eid,
             "supplement requires complete same-candidate/freeze primary parent and a new ID")
    _require(_same(parent.get("frozen_configuration"), config)
             and parent.get("wrapper_configuration_sha256") == fingerprint(config), "parent configuration differs from frozen argument")
    rebuilt = build_primary(parent["wrapper_source_pairs"], tokenizer, parent["experiment_id"], config, freeze_id, basis_metadata=basis)
    _require(all(key in parent and _same(parent[key], value) for key, value in rebuilt.items()), "parent cannot be reconstructed exactly from canonical sources/configuration")
    accepted = [parent["wrapper_source_pairs"][entry["input_index"]] for entry in parent["eligibility_manifest"] if entry["eligible"]]
    families = list(FAMILIES) if basis is None else [config["family"]]
    subset = select_lexical_subset(accepted, families=families)
    parent_jobs = {(job["pair_id"], job["condition"]): job for job in parent["jobs"]}
    _require(len(parent_jobs) == len(parent["jobs"]), "duplicate parent pair/condition job")
    parent_manifests = {entry["pair_id"]: entry for entry in parent["eligibility_manifest"] if entry["eligible"]}
    jobs, inline, manifests = [], [], []
    for pair in subset["selected_pairs"]:
        pid = pair["pair_id"]
        parent_manifest = parent_manifests[pid]
        if basis is None:
            materialized = contextual._materialize_pair(pair, tokenizer, eid, layer=config["layer"], renderer_id=ALTERNATE,
                require_joint_renderer_eligibility=True, candidate_source_experiment_id=config["source_experiment_id"],
                candidate_source_spec_sha256=config["source_spec_sha256"], freeze_id=freeze_id)
            _require(materialized["eligible"], "previously eligible alternate contextual source changed")
            _require(_same(materialized["joint_renderer_audit"], parent_manifest["joint_renderer_audit"]), "joint renderer evidence changed since parent construction")
            inline.append({"pair_id": pid, **materialized["inline_controlled_source"]})
            emitted = materialized["jobs"]
            for value in emitted:
                standard = parent_jobs[pid, value["condition"]]
                _require(_same({k: v for k, v in value["intervention"].items() if k != "full_write_source_identity"},
                               {k: v for k, v in standard["intervention"].items() if k != "full_write_source_identity"}),
                         "alternate contextual operation/dose/random/audit drift")
                value["metadata"].update(paired_standard_experiment_id=parent["experiment_id"],
                    paired_standard_job_id=standard["job_id"], paired_standard_job_sha256=fingerprint(standard),
                    random_orientation_support_and_seed_shared_across_renderings=True)
        else:
            source = _rendered_subspace_source(pair, tokenizer)
            audit = audit_pair_renderings(pair, tokenizer, profile="matched")
            _require(audit["eligible"] is True and _same(audit, parent_manifest["joint_renderer_audit"]), "joint subspace renderer evidence changed")
            _require(source["input_fact_span"]["n_changed_span_tokens"] == audit["renderings"][CANONICAL]["recipient"]["span"]["n_changed_span_tokens"], "alternate selected support width differs")
            inline.append({"pair_id": pid, "definition": source, "definition_sha256": fingerprint(source), "registered": False})
            emitted = [_alternate_subspace_job(parent_jobs[pid, condition], source, eid, parent) for condition in config["condition_inventory"]]
            for value in emitted:
                _require(_same(value["intervention"], parent_jobs[pid, value["condition"]]["intervention"]), "alternate altered existing basis/rank/dose/random geometry")
        for value in emitted:
            _require(_same(value["metadata"]["source_pair"], pair), "alternate canonical physical source lineage changed")
            standard = parent_jobs[pid, value["condition"]]
            _require(len(value["selection"]["recipient_positions"]) == len(standard["selection"]["recipient_positions"]), "alternate support width differs from parent")
            value["metadata"].update(cohort_role="lexical_supplement", inferential_role="descriptive", semantic_headline_eligible=False)
            value.pop("primary_id", None)
            value["confirmatory"] = False
            jobs.append(_job(eid, value))
        manifests.append({"pair_id": pid, "layout_id": pair["layout_id"], "family": pair["family"], "eligible": True,
            "source_pair_sha256": fingerprint(pair), "layout_aliases": deepcopy(parent_manifest["layout_aliases"]),
            "required_condition_inventory": deepcopy(config["condition_inventory"]),
            "joint_renderer_audit": deepcopy(parent_manifest["joint_renderer_audit"]),
            "scheduled_conditions": deepcopy(config["condition_inventory"]), "selection_before_model_query": True})
    _require(len(jobs) == 32 * len(config["condition_inventory"]), "alternate supplement control inventory incomplete")
    standard_jobs = [parent_jobs[pid, condition] for pid in subset["pair_ids"] for condition in config["condition_inventory"]]
    supplement_config = {key: deepcopy(value) for key, value in config.items() if key != "primary_comparisons"}
    supplement_config.update(n_layouts=32, renderer_id=ALTERNATE, renderer_binding=renderer_manifest(ALTERNATE),
                             cohort_role="lexical_supplement", inferential_role="descriptive")
    if basis is None:
        supplement_config.update(family_layout_counts=deepcopy(subset["family_layout_counts"]), family_weights=deepcopy(subset["family_weights"]))
    return {"schema_version": parent["schema_version"], "wrapper_version": VERSION, "cohort_role": "lexical_supplement",
            "experiment_id": eid, "partition": "confirmation", "freeze_id": freeze_id, "candidate_id": config["candidate_id"],
            "protocol_kind": config["protocol_kind"], "claim_scope": config["claim_scope"], "semantic_headline_eligible": False,
            "parent_experiment_id": parent["experiment_id"], "parent_spec_fingerprint": fingerprint(parent),
            "parent_configuration_sha256": fingerprint(config), "parent_frozen_configuration": config,
            "supplement_configuration": supplement_config, "supplement_configuration_sha256": fingerprint(supplement_config),
            "renderer_id": ALTERNATE, "renderer_binding": renderer_manifest(ALTERNATE), "tokenizer": deepcopy(config["tokenizer"]),
            "jobs": jobs, "comparisons": _descriptive_comparisons(parent, eid, subset, standard_reference=False),
            "lexical_subset": subset, "lexical_subset_sha256": fingerprint(subset), "eligibility_manifest": manifests,
            "expected_n_layouts": 32, "expected_n_pairs": 32, "cohort_complete": True,
            "family_layout_counts": deepcopy(subset["family_layout_counts"]), "family_weights": deepcopy(subset["family_weights"]),
            "layout_aliases": {row["layout_id"]: row["layout_aliases"] for row in manifests},
            "inline_rendered_sources": inline, "independent_rescue_evidence": False,
            "standard_subset_reference": {"experiment_id": parent["experiment_id"], "parent_spec_fingerprint": fingerprint(parent),
                "job_fingerprints": {job["job_id"]: fingerprint(job) for job in standard_jobs},
                "job_bindings": [{"job_id": job["job_id"], "pair_id": job["pair_id"], "layout_id": job["layout_id"],
                                  "condition": job["condition"], "seed": job["seed"]} for job in standard_jobs],
                "comparisons": _descriptive_comparisons(parent, eid, subset, standard_reference=True),
                "n_layouts": 32, "family_layout_counts": deepcopy(subset["family_layout_counts"]),
                "family_weights": deepcopy(subset["family_weights"]), "execution_jobs_added": 0},
            "protocol": {"renderer_id": ALTERNATE, "condition_inventory": deepcopy(config["condition_inventory"]),
                "generation_seeds": [42], "temperature": 0.7, "max_new_tokens": 4096,
                "same_layouts_as_standard_subset": True, "additional_independent_layouts": 0,
                "random_orientation_support_and_seed_shared_across_renderings": True,
                "basis_refitted": False, "new_confirmatory_tests": 0, "model_queries": 0}}
