"""Derive fixed confirmation configurations from source geometry, not outcomes.

Input ``profile`` is the selector's minimal ``profile['descriptor']``. The
source bundle uses confirmation_sources' existing schema. Its rows, timing and
generation summaries are deliberately never accessed or copied. External source
and selection gates authenticate artifacts/qualification; this pure helper does
not open confirmation, freeze, read files, tokenize, fit, or query a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from . import frozen_contextual_input as contextual
from . import frozen_subspace as subspace
from .lexical_generalization import CANONICAL, renderer_manifest
from .registry import fingerprint

DESCRIPTOR_FIELDS = {"profile_id", "protocol_kind", "source_experiment_id", "source_spec_sha256",
                     "layer", "site", "selected_dose", "rank", "family"}
SOURCE_KEYS = ("experiment_id", "specification", "source_spec_sha256", "source_result_hashes",
               "source_result_manifest_sha256", "independent_audit", "n_jobs")
FAMILIES = contextual.FAMILIES


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def _relative(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts


def _profile(value: Mapping[str, Any]) -> dict:
    _require(isinstance(value, Mapping) and set(value) == DESCRIPTOR_FIELDS, "consume exactly the selected geometric descriptor")
    profile = deepcopy(dict(value))
    _require(_name(profile["profile_id"]) and _name(profile["source_experiment_id"]) and _sha(profile["source_spec_sha256"]),
             "invalid profile/source identity")
    _require(profile["site"] == "resid_post" and type(profile["layer"]) is int, "invalid selected source site/layer")
    if profile["protocol_kind"] == contextual.PROTOCOL:
        _require(profile["layer"] in {0, 10} and profile["rank"] is None and profile["family"] is None
                 and _same(profile["selected_dose"], 1.0), "contextual descriptor must preserve mixed-family full transfer")
        _require(profile["source_experiment_id"] == "contextual_input_native_pilot_v1", "unsupported contextual native source")
        _require(profile["profile_id"] == f"contextual_l{profile['layer']}", "contextual profile ID and geometry disagree")
    elif profile["protocol_kind"] == subspace.PROTOCOL:
        _require(profile["layer"] == 0 and type(profile["rank"]) is int and profile["rank"] in {1, 4}
                 and profile["family"] in FAMILIES and type(profile["selected_dose"]) is float
                 and profile["selected_dose"] in {0.5, 1.0}, "invalid fixed subspace descriptor")
        short = {"wall": "wall", "key_possession": "key", "door_status": "door"}[profile["family"]]
        _require(profile["source_experiment_id"] == f"subspace_{short}_l0_r{profile['rank']}_native_v1", "subspace source and geometry disagree")
        label = "0p5" if profile["selected_dose"] == 0.5 else "1"
        _require(profile["profile_id"] == f"subspace_{profile['family']}_l0_r{profile['rank']}_dose{label}", "subspace profile ID and geometry disagree")
    else:
        raise ValueError("unsupported selected profile protocol")
    return profile


def _source(bundle: Mapping[str, Any], profile: dict) -> dict:
    _require(isinstance(bundle, Mapping) and all(key in bundle for key in SOURCE_KEYS), "missing verified source bundle fields")
    # Project before deepcopy: ignored outcome objects need not even be readable.
    source = {key: deepcopy(bundle[key]) for key in SOURCE_KEYS}
    spec = source["specification"]
    _require(isinstance(spec, dict) and spec.get("partition") == "discovery"
             and source["experiment_id"] == spec.get("experiment_id") == profile["source_experiment_id"]
             and source["source_spec_sha256"] == profile["source_spec_sha256"], "source descriptor binding mismatch")
    jobs = spec.get("jobs")
    _require(isinstance(jobs, list) and jobs and all(isinstance(job, dict) for job in jobs), "source jobs required")
    _require(type(source["n_jobs"]) is int and source["n_jobs"] == len(jobs), "source job count mismatch")
    job_ids = [job.get("job_id") for job in jobs]
    _require(len(set(job_ids)) == len(jobs) and all(isinstance(item, str) for item in job_ids), "source job IDs missing/duplicated")
    for job in jobs:
        _require(job["job_id"] == fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24], "noncanonical frozen source job ID")
        expected = {"experiment_id": source["experiment_id"], "partition": "discovery", "natural": True,
                    "seed": 42, "temperature": 0.7, "max_new_tokens": 4096}
        _require(_same({key: job.get(key) for key in expected}, expected), "native source seed/cap/temperature drift")
    manifest = source["source_result_hashes"]
    expected_paths = {f"results/{source['experiment_id']}/{job_id}.json" for job_id in job_ids}
    _require(isinstance(manifest, dict) and set(manifest) == expected_paths and all(_sha(v) for v in manifest.values())
             and source["source_result_manifest_sha256"] == fingerprint(manifest), "source result manifest must bind every exact planned job")
    audit = source["independent_audit"]
    _require(isinstance(audit, dict) and set(audit) == {"path", "sha256", "input_files_sha256"}
             and _relative(audit["path"]) and _sha(audit["sha256"]), "missing independent audit artifact binding")
    inputs = audit["input_files_sha256"]
    _require(isinstance(inputs, dict) and inputs and all(_relative(k) and _sha(v) for k, v in inputs.items()), "invalid audit input file mapping")
    audit_path = PurePosixPath(audit["path"])
    _require(audit_path.parent.name == "reviews", "independent audit must belong to the campaign review directory")
    campaign_prefix = str(audit_path.parent.parent)
    prefix = "" if campaign_prefix == "." else campaign_prefix + "/"
    # Contextual audit producer binds raw files here; it authenticates the spec
    # separately in source_spec_sha256 and its final global source bracket.
    # Do not manufacture a spec entry in that immutable audit mapping.
    spec_path = prefix + f"specs/{source['experiment_id']}.json"
    _require(spec_path not in inputs or inputs[spec_path] == source["source_spec_sha256"], "audit source spec digest contradiction")
    _require(all(inputs.get(prefix + path) == digest for path, digest in manifest.items()), "audit omits or contradicts source result hashes")
    source["source_spec_fingerprint"] = fingerprint(spec)
    return source


def _interventions(profile: dict, basis: dict | None) -> dict[str, dict]:
    base = {"layers": [], "site": "resid_post", "operation": "replace", "dose": 1.0}
    if basis is None:
        expected = {name: deepcopy(base) for name in contextual.REFERENCES}
        for layer in (0, 10):
            for role in contextual.INTERVENTIONS[:-1]:
                intervention = {**base, "layers": [layer]}
                if role == "random":
                    intervention.update(random_control=True, random_seed=42, norm_reference="donor_minus_recipient")
                expected[f"layer{layer}_{role}"] = intervention
        return expected
    ref = {"basis_path": basis["basis_path"], "basis_sha256": basis["basis_file_sha256"],
           "basis_metadata_path": basis["basis_metadata_path"], "basis_metadata_sha256": basis["basis_metadata_sha256"]}
    projection = {"layers": [0], "site": "resid_post", "operation": "subspace", "dose": 1.0, **ref}
    expected = {"baseline": deepcopy(base), "donor_text": deepcopy(base), "full_transfer": {**base, "layers": [0]},
                "subspace_identity": deepcopy(projection), "subspace_removal": {**projection, "zero_donor": True},
                "restoration_equivalent": deepcopy(projection)}
    for dose, label in ((0.5, "0p5"), (1.0, "1")):
        projected = {**projection, "dose": dose}
        audited = {**projected, "audit_projected_norm": True, "projected_norm_relative_tolerance": 0.01}
        expected["subspace_forward_" + label] = audited
        expected["subspace_reverse_" + label] = projected
        expected["random_subspace_" + label] = {**projected, "random_basis": True, "random_seed": 42, "rank": profile["rank"]}
        expected["projected_norm_matched_random_subspace_" + label] = {
            **audited, "projected_norm_matched_random_basis": True, "random_seed": 42, "rank": profile["rank"]}
    return expected


def _geometry(source: dict, profile: dict) -> tuple[dict | None, list[str]]:
    spec = source["specification"]
    is_context = profile["protocol_kind"] == contextual.PROTOCOL
    basis = None if is_context else deepcopy(spec.get("basis"))
    if is_context:
        expected_protocol = {"layers": [0, 10], "site": "resid_post", "seeds": [42], "natural": True,
                             "max_new_tokens": 4096, "selection_scope": "explicit_named_fact_input_tokens",
                             "source_span_localization": "within_context_named_variable_toggle"}
        _require(spec.get("claim_scope") == contextual.CLAIM_SCOPE, "contextual source scope drift")
    else:
        _require(isinstance(basis, dict), "source must contain existing registered basis metadata")
        for key in ("family", "rank", "layer", "site"):
            _require(_same(basis.get(key), profile[key]), "source basis geometry differs from descriptor: " + key)
        _require(spec.get("basis_file_sha256") == basis.get("basis_file_sha256"), "source basis file digest contradiction")
        expected_protocol = {"layer": 0, "rank": profile["rank"], "site": "resid_post", "doses": [0.5, 1.0],
                             "projected_norm_relative_tolerance": 0.01, "zero_reference_requires_exact_zero": True}
    protocol = spec.get("protocol", {})
    _require(_same({key: protocol.get(key) for key in expected_protocol}, expected_protocol), "source geometry/protocol drift")
    expected = _interventions(profile, basis)
    groups = defaultdict(list)
    for job in spec["jobs"]:
        _require(job.get("condition") in expected and _same(job["intervention"], expected[job["condition"]]), "source condition intervention drift")
        _require(isinstance(job.get("pair_id"), str) and job["pair_id"] and isinstance(job.get("layout_id"), str) and job["layout_id"], "source physical identities required")
        groups[job["pair_id"]].append(job)
        selection = job.get("selection", {})
        _require(selection.get("scope") == "explicit", "source span rule drift")
        rids, rpos = job.get("recipient_token_ids"), selection.get("recipient_positions")
        _require(isinstance(rids, list) and rids and all(type(i) is int and i >= 0 for i in rids)
                 and isinstance(rpos, list) and rpos and all(type(p) is int and 0 <= p < len(rids) for p in rpos)
                 and len(set(rpos)) == len(rpos), "source explicit recipient token geometry invalid")
        if "recipient_positions" in job:
            _require(_same(job["recipient_positions"], rpos), "source recipient position aliases disagree")
        if job["intervention"]["layers"]:
            dids, dpos = job.get("donor_token_ids"), selection.get("donor_positions")
            _require(isinstance(dids, list) and len(dids) == len(rids) and all(type(i) is int and i >= 0 for i in dids)
                     and isinstance(dpos, list) and len(dpos) == len(rpos)
                     and all(type(p) is int and 0 <= p < len(dids) for p in dpos) and len(set(dpos)) == len(dpos), "source explicit donor geometry drift")
            if "donor_positions" in job:
                _require(_same(job["donor_positions"], dpos), "source donor position aliases disagree")
        representation = "observed_fact_input_tokens" if is_context else "input_fact_tokens"
        _require(job.get("metadata", {}).get("representation_kind") == representation, "source input-fact representation drift")
    expected_count = 6 if is_context else 4
    _require(len(groups) == expected_count and len(spec["jobs"]) == expected_count * len(expected), "whole source cohort inventory drift")
    families, layouts = Counter(), set()
    for pair_id, jobs in groups.items():
        _require(Counter(j["condition"] for j in jobs) == Counter(dict.fromkeys(expected, 1)), "source pair omits or duplicates controls")
        for key in ("family", "layout_id", "target_action", "original_action"):
            _require(len({job.get(key) for job in jobs}) == 1, "source pair identity/action drift: " + key)
        _require(jobs[0]["target_action"] in {"UP", "DOWN", "LEFT", "RIGHT"}
                 and jobs[0]["original_action"] in {"UP", "DOWN", "LEFT", "RIGHT"}
                 and jobs[0]["target_action"] != jobs[0]["original_action"], "source target must remain fixed and distinct")
        families[jobs[0]["family"]] += 1
        layouts.add(jobs[0]["layout_id"])
    _require(len(layouts) == expected_count, "source pair/layout independence mismatch")
    _require(families == Counter(dict.fromkeys(FAMILIES, 2)) if is_context else families == Counter({profile["family"]: 4}), "source family population drift")
    return basis, sorted(groups)


def _origins(source: dict, primary_origins: Mapping[str, Any], treatments: dict[str, tuple[str, str]],
             pair_ids: list[str], is_context: bool) -> dict:
    _require(isinstance(primary_origins, Mapping) and set(primary_origins) == set(treatments), "exact named primary origins required")
    origins = deepcopy(dict(primary_origins))
    common = {"source_experiment_id": source["experiment_id"], "source_spec_sha256": source["source_spec_sha256"],
              "source_result_manifest_sha256": source["source_result_manifest_sha256"],
              "source_independent_audit_path": source["independent_audit"]["path"],
              "source_independent_audit_sha256": source["independent_audit"]["sha256"]}
    for primary_id, (treatment, control) in treatments.items():
        origin = origins[primary_id]
        prospective = is_context and primary_id == contextual.PRIMARY_IDS[1]
        _require(isinstance(origin, dict) and set(origin) == {"kind", "provenance"} and origin["kind"] ==
                 ("prospective_confirmation_control" if prospective else "existing_source_comparison"), "primary origin kind mismatch")
        provenance = origin["provenance"]
        _require(isinstance(provenance, dict) and _same({key: provenance.get(key) for key in common}, common), "primary origin source/audit bindings disagree")
        if prospective:
            _require(provenance.get("discovery_measured") is False and provenance.get("treatment") == treatment
                     and provenance.get("control") == control and "source_comparison_id" not in provenance,
                     "prospective natural-norm control must remain explicitly unmeasured without an invented source comparison")
            for key in ("addendum", "adapter"):
                _require(_relative(provenance.get(key + "_path")) and _sha(provenance.get(key + "_sha256")), "missing prospective control implementation/review binding")
            _require(provenance["adapter_path"] == "src/activation_causality/frozen_contextual_input.py", "prospective control requires the designated fixed contextual adapter")
            _require(not any(job["condition"] == control for job in source["specification"]["jobs"]), "prospective control already occurs in source")
        else:
            _require(provenance.get("discovery_measured") is True, "existing comparison must be explicitly measured")
            comparisons = [c for c in source["specification"]["comparisons"] if (c.get("treatment"), c.get("control")) == (treatment, control)]
            _require(len(comparisons) == 1, "existing primary must match one unique source comparison")
            comparison = comparisons[0]
            _require(comparison.get("id") == provenance.get("source_comparison_id")
                     and comparison.get("predicted_direction") == "increase"
                     and comparison.get("experiment_id") == source["experiment_id"] and comparison.get("partition") == "discovery"
                     and sorted(comparison.get("eligible_pair_ids", [])) == pair_ids
                     and type(comparison.get("expected_n_layouts")) is int and comparison["expected_n_layouts"] == len(pair_ids)
                     and type(comparison.get("expected_n_pairs")) is int and comparison["expected_n_pairs"] == len(pair_ids), "source comparison ID or whole-cohort eligibility drift")
            if "source_comparison_sha256" in provenance:
                _require(provenance["source_comparison_sha256"] == fingerprint(comparison), "source comparison semantic fingerprint mismatch")
    return origins


def derive_candidate_configuration(
    profile: Mapping[str, Any], source_bundle: Mapping[str, Any], *, candidate_id: str, n_layouts: int,
    dataset_seed: int, tokenizer_metadata: Mapping[str, Any], renderer_binding: Mapping[str, Any],
    require_joint_renderer_eligibility: bool, eligibility_rule: Mapping[str, Any], primary_origins: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the full source-derived configuration accepted by its fixed adapter.

    This accepts only the minimal geometric descriptor, not the selector's full
    metrics/ranking object. Root authenticates the source bundle, chosen profile,
    local tokenizer/renderer, eligibility rule and origin artifacts before use.
    """
    profile = _profile(profile)
    _require(_name(candidate_id) and type(n_layouts) is int and n_layouts in {32, 64, 128}
             and type(dataset_seed) is int and 0 <= dataset_seed < 2**32 - 2100, "invalid frozen allocation/identity/seed")
    _require(type(require_joint_renderer_eligibility) is bool, "joint renderer decision must be explicit")
    _require(isinstance(renderer_binding, Mapping) and _same(dict(renderer_binding), renderer_manifest(CANONICAL)), "only exact reviewed canonical primary renderer binding is allowed")
    _require(isinstance(tokenizer_metadata, Mapping) and set(tokenizer_metadata) ==
             {"class", "name_or_path", "add_special_tokens", "context_renderer"}, "exact tokenizer metadata fields required")
    tokenizer = deepcopy(dict(tokenizer_metadata))
    _require(all(isinstance(tokenizer[key], str) and tokenizer[key] for key in ("class", "name_or_path"))
             and tokenizer["add_special_tokens"] is False and tokenizer["context_renderer"] == "campaign.render_analysis_prompt", "tokenizer metadata contract drift")
    _require(isinstance(eligibility_rule, Mapping) and set(eligibility_rule) == {"identifier", "sha256"}
             and _name(eligibility_rule["identifier"]) and _sha(eligibility_rule["sha256"]), "eligibility rule binding required")
    source = _source(source_bundle, profile)
    basis, pair_ids = _geometry(source, profile)
    is_context = basis is None
    if is_context:
        treatment = f"layer{profile['layer']}_natural"
        treatments = {contextual.PRIMARY_IDS[0]: (treatment, f"layer{profile['layer']}_same_fact"),
                      contextual.PRIMARY_IDS[1]: (treatment, f"layer{profile['layer']}_natural_norm_random")}
    else:
        label = "0p5" if profile["selected_dose"] == 0.5 else "1"
        treatment = "subspace_forward_" + label
        treatments = {subspace.PRIMARY_IDS[0]: (treatment, "baseline"),
                      subspace.PRIMARY_IDS[1]: (treatment, "projected_norm_matched_random_subspace_" + label)}
    origins = _origins(source, primary_origins, treatments, pair_ids, is_context)
    scope = contextual.CLAIM_SCOPE if is_context else subspace.CLAIM_SCOPE
    common_binding = {"predicted_direction": "increase", "primary_endpoint": contextual.ENDPOINT,
                      "target_action_basis": "unique_simulator_counterfactual_optimum",
                      "native_unresolved_policy": contextual.UNRESOLVED_POLICY, "expected_n_layouts": n_layouts,
                      "expected_n_pairs": n_layouts, "eligibility_rule": deepcopy(dict(eligibility_rule)),
                      "claim_scope": scope, "semantic_headline_eligible": False}
    config = {"candidate_id": candidate_id, "source_experiment_id": source["experiment_id"],
              "source_spec_sha256": source["source_spec_sha256"], "source_spec_fingerprint": source["source_spec_fingerprint"],
              "source_result_hashes": source["source_result_hashes"], "source_result_manifest_sha256": source["source_result_manifest_sha256"],
              "source_independent_audit_path": source["independent_audit"]["path"],
              "source_independent_audit_sha256": source["independent_audit"]["sha256"],
              "source_independent_audit_input_files_sha256": source["independent_audit"]["input_files_sha256"],
              "source_n_jobs": source["n_jobs"], "source_profile_descriptor": profile,
              "source_profile_descriptor_sha256": fingerprint(profile), "dataset_seed": dataset_seed,
              "layer": profile["layer"], "site": "resid_post", "n_layouts": n_layouts,
              "generation_seeds": [42], "max_new_tokens": 4096, "temperature": 0.7, "tokenizer": tokenizer,
              "renderer_id": CANONICAL, "renderer_binding": deepcopy(dict(renderer_binding)),
              "require_joint_renderer_eligibility": require_joint_renderer_eligibility,
              "source_authentication_and_scientific_selection_are_external": True,
              "source_outcome_rows_read_by_derivation": False}
    if is_context:
        quotas = contextual.QUOTAS[n_layouts]
        mixture = {"families": list(FAMILIES), "family_layout_counts": deepcopy(quotas),
                   "family_weights": {family: quotas[family] / n_layouts for family in FAMILIES},
                   "resampling_scheme": "within_family_layout_bootstrap", "stratification_key": "family",
                   "expected_n_layouts_by_family": deepcopy(quotas), "expected_n_pairs_by_family": deepcopy(quotas)}
        common_binding.update(mixture)
        config.update(families=list(FAMILIES), family_layout_counts=deepcopy(quotas), family_weights=deepcopy(mixture["family_weights"]),
                      random_seed=42, condition_inventory=contextual.condition_inventory(profile["layer"]))
    else:
        config.update(family=basis["family"], rank=basis["rank"], selected_dose=profile["selected_dose"], doses=[0.5, 1.0],
                      random_basis_seeds=[42], condition_inventory=list(subspace.CONDITIONS), basis_id=basis["basis_id"],
                      basis_path=basis["basis_path"], basis_sha256=basis["basis_file_sha256"],
                      basis_metadata_path=basis["basis_metadata_path"], basis_metadata_sha256=basis["basis_metadata_sha256"],
                      basis_metadata_fingerprint=fingerprint(basis), basis_metadata=deepcopy(basis))
    config["primary_comparisons"] = {primary_id: {**deepcopy(common_binding), "treatment": treatment,
        "control": control, "origin": origins[primary_id]} for primary_id, (treatment, control) in treatments.items()}
    if is_context:
        # Only its class/name are inspected by _configuration; no tokenizer or
        # model is constructed, and no encode method is installed/called.
        metadata_proxy = type(tokenizer["class"], (), {"name_or_path": tokenizer["name_or_path"]})()
        normalized = contextual._configuration(config, metadata_proxy)
    else:
        normalized, returned_basis = subspace._configuration(config, basis)
        _require(_same(returned_basis, basis), "adapter changed existing basis metadata")
    return normalized
