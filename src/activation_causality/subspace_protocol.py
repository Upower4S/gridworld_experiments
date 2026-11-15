"""Locked, provenance-bound basis fitting and held-out discovery specifications."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from . import localization, registry
from .model_identity import snapshot_identity


PROJECTED_NORM_RELATIVE_TOLERANCE = 0.01
CAPTURE_CONVENTION = "full-prefill; exact-input-token-ids; site as declared; float32 capture of model activations"


def _name(value: str) -> None:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Artifact IDs must contain only letters, digits, underscore or hyphen")


def _aliases(job: dict[str, Any]) -> set[str]:
    metadata = job.get("metadata", {})
    aliases = {job["layout_id"], *job.get("layout_aliases", [])}
    for name in ("pair", "source_pair"):
        nested = metadata.get(name, {})
        aliases.update(nested.get("metadata", {}).get("variant_layout_ids", []))
        if isinstance(nested.get("layout_id"), str) and nested["layout_id"]:
            aliases.add(nested["layout_id"])
    return aliases


def _explicit_positions(job: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """Reconcile the compiler, extraction and audit spellings before swapping."""
    resolved = []
    for role in ("recipient", "donor"):
        values = [mapping[key] for mapping, key in (
            (job, role + "_positions"),
            (job.get("selection", {}), role + "_positions"),
            (job.get("metadata", {}), role + "_patch_positions"),
        ) if mapping.get(key) is not None]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError("Conflicting explicit intervention positions")
        positions = values[0] if values else None
        if positions is not None and (not isinstance(positions, list) or not positions
                                      or any(type(i) is not int or i < 0 for i in positions)
                                      or len(set(positions)) != len(positions)):
            raise ValueError("Explicit intervention positions must be nonempty unique integer indices")
        resolved.append(positions)
    if all(value is None for value in resolved):
        return None
    if any(value is None for value in resolved) or len(resolved[0]) != len(resolved[1]):
        raise ValueError("Both explicit intervention spans must have equal lengths")
    return deepcopy((resolved[0], resolved[1]))


def _deadline(campaign: Path) -> None:
    if registry.seconds_remaining(campaign) <= 0:
        raise RuntimeError("Campaign deadline reached; no new model work may start")


def _activation_representation(runtime: dict[str, Any]) -> dict[str, Any]:
    # One producer/consumer contract, including conservative engine/load source
    # provenance; this is not a claim that all dependencies affect capture math.
    from .basis_runtime import activation_representation
    return activation_representation(runtime)


class _DeadlineEngine:
    """Check the budget before each extraction; all other properties delegate."""
    def __init__(self, engine: Any, campaign: Path):
        self.engine, self.campaign = engine, campaign

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def capture(self, *args: Any, **kwargs: Any) -> Any:
        _deadline(self.campaign)
        return self.engine.capture(*args, **kwargs)


def _immutable_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to replace an immutable basis file")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_basis(campaign: Path, repo: Path, source_experiment_id: str, condition: str,
              family: str, layer: int, site: str, rank: int, basis_id: str) -> dict[str, Any]:
    """Fit once under the shared worker lock; never run this on confirmation data.

    Engine construction and validation happen only when this function is called.
    Existing artifacts are returned read-only after request and SHA verification.
    The first seed per pair is selected deterministically before any activations
    are inspected; additional seeds are not counted as independent training pairs.
    """
    from .campaign import make_job, runtime_fingerprint
    from .engine import MLXInterventionEngine

    campaign, repo = Path(campaign), Path(repo)
    _name(source_experiment_id)
    _name(basis_id)
    if type(layer) is not int or not 0 <= layer < 24 or site not in localization.SITES:
        raise ValueError("Invalid layer or site")
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be positive")
    source_path = campaign / "specs" / f"{source_experiment_id}.json"
    source_raw = source_path.read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    source = json.loads(source_raw)
    if source.get("experiment_id") != source_experiment_id:
        raise ValueError("Source specification identity does not match its registry path")
    if source.get("partition") != "discovery":
        raise ValueError("Basis fitting is discovery-only")
    if (campaign / "reviews" / f"{source_experiment_id}_retired.json").exists():
        raise ValueError("Cannot fit a basis from a retired source specification")
    candidates = [job for job in source["jobs"] if job["condition"] == condition and job["family"] == family]
    if not candidates:
        raise ValueError("No source jobs match the single condition and family")
    selected = {}
    for job in sorted(candidates, key=lambda item: (item["pair_id"], item["seed"], item["job_id"])):
        if job.get("experiment_id") != source_experiment_id:
            raise ValueError("Training job experiment does not match its source specification")
        if job["partition"] != "discovery" or not isinstance(job.get("donor_text"), str):
            raise ValueError("Every selected training job needs a discovery donor contrast")
        values = {name: value for name, value in job.items() if name not in {"job_id", "experiment_id"}}
        if make_job(source_experiment_id, **values)["job_id"] != job["job_id"]:
            raise ValueError("Training job ID does not match its immutable source definition")
        intervention = job.get("intervention", {})
        if intervention.get("random_control") or intervention.get("zero_donor") or intervention.get("operation", "replace") != "replace":
            raise ValueError("Select an intended full-replacement contrast, not a synthetic control")
        if intervention.get("layers") != [layer] or intervention.get("site") != site:
            raise ValueError("Requested extraction layer/site must match the registered source job geometry")
        selected.setdefault(job["pair_id"], deepcopy(job))
    jobs = list(selected.values())
    source_job_fingerprints = {job["job_id"]: registry.fingerprint(job) for job in jobs}
    if rank > len(jobs):
        raise ValueError("rank exceeds distinct selected training pairs")
    # Explicit positions are forwarded when supplied in the builder's audit.
    for job in jobs:
        explicit = _explicit_positions(job)
        if explicit is not None:
            job["recipient_positions"], job["donor_positions"] = explicit
    config_path = campaign / "campaign.json"
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    config_sha = hashlib.sha256(config_raw).hexdigest()
    runtime = runtime_fingerprint(repo)
    representation = _activation_representation(runtime)
    definitions_sha = registry.fingerprint(jobs)
    basis_path = (campaign / "bases" / f"{basis_id}.npy").absolute()
    metadata_path = basis_path.with_suffix(".json")

    def existing(request_hash: str) -> dict[str, Any] | None:
        if metadata_path.exists():
            metadata = registry.read_json(metadata_path)
            if metadata.get("request_fingerprint") != request_hash:
                raise ValueError("Basis ID already belongs to another source, runtime, or fit request")
            if not basis_path.exists() or registry.file_sha256(basis_path) != metadata["basis_file_sha256"]:
                raise ValueError("Stored basis file is missing or has changed")
            audit_path = metadata.get("model_snapshot_audit_path")
            if (not audit_path or not Path(audit_path).is_file()
                    or registry.file_sha256(Path(audit_path)) != metadata.get("model_snapshot_audit_sha256")):
                raise ValueError("Stored pre/post model snapshot audit is missing or has changed")
            return {**metadata, "basis_metadata_path": str(metadata_path),
                    "basis_metadata_sha256": registry.file_sha256(metadata_path)}
        if basis_path.exists():
            raise ValueError("Orphan immutable basis file; preserve it and choose a new basis ID")
        return None

    if not metadata_path.exists():
        _deadline(campaign)
    with registry.worker_lock(campaign):
        # Hash payloads under the same lock as the fit, including read-only reuse.
        # No path/mtime cache or index-only identity stands in for actual weights.
        if not metadata_path.exists():
            _deadline(campaign)
        before = snapshot_identity(Path(config["model_path"]))
        model_path = Path(before["path"])
        model_files = before["files"]
        if (registry.file_sha256(source_path) != source_sha
                or registry.file_sha256(config_path) != config_sha
                or (campaign / "reviews" / f"{source_experiment_id}_retired.json").exists()):
            raise ValueError("Source specification or campaign provenance changed before fitting")
        request = {"source_experiment_id": source_experiment_id, "source_spec_sha256": source_sha,
                   "source_spec_path": str(source_path.resolve()),
                   "source_job_fingerprints": source_job_fingerprints,
                   "selected_source_definitions_sha256": definitions_sha,
                   "condition": condition, "family": family, "layer": layer, "site": site, "rank": rank,
                   "selected_job_ids": [job["job_id"] for job in jobs], "model_path": str(model_path),
                   "model_files": model_files, "model_provenance_version": 2,
                   "model_snapshot_identity": before,
                   "model_snapshot_identity_sha256": before["identity_sha256"],
                   "campaign_config_sha256": config_sha, "runtime": runtime,
                   "activation_representation": representation}
        request_hash = registry.fingerprint(request)
        previous = existing(request_hash)
        if previous is not None:
            return previous
        _deadline(campaign)
        registry.immutable_json(campaign / "bases" / f"{basis_id}.request.json", request)
        snapshot_path = (campaign / "reviews" / f"basis_snapshot_{basis_id}.json").absolute()
        if snapshot_path.exists():
            raise ValueError("A prior fit attempt already has an immutable snapshot audit; choose a new basis ID")
        registry.append_event(campaign, "basis_fit_started", basis_id=basis_id, request_fingerprint=request_hash)
        fit_error = None
        try:
            engine = MLXInterventionEngine(model_path)
            validation = engine.validate()
            validation_path = (campaign / "reviews" / f"basis_validation_{basis_id}.json").absolute()
            registry.immutable_json(validation_path, {"request_fingerprint": request_hash, "validation": validation})
            if not validation.get("passed", False):
                raise RuntimeError("Engine validation failed; no basis was fitted")
            _deadline(campaign)
            fitted = localization.extract_and_fit(_DeadlineEngine(engine, campaign), jobs, layer, site, rank)
        except Exception as exc:
            fit_error = exc
        after, snapshot_error = None, None
        try:
            after = snapshot_identity(Path(config["model_path"]))
        except Exception as exc:
            snapshot_error = f"{type(exc).__name__}: {exc}"
        after_representation, representation_error = None, None
        try:
            after_representation = _activation_representation(runtime_fingerprint(repo))
        except Exception as exc:
            representation_error = f"{type(exc).__name__}: {exc}"
        representation_equal = representation == after_representation and representation_error is None
        try:
            source_unchanged = (registry.file_sha256(source_path) == source_sha
                                and registry.file_sha256(config_path) == config_sha
                                and not (campaign / "reviews" / f"{source_experiment_id}_retired.json").exists())
        except OSError:
            source_unchanged = False
        snapshot_equal = before == after and snapshot_error is None
        audit = {"format": "basis_fit_model_snapshot_audit_v1", "basis_id": basis_id,
                 "request_fingerprint": request_hash, "before_fit": before, "after_fit": after,
                 "snapshots_equal": snapshot_equal, "snapshot_error": snapshot_error,
                 "source_and_campaign_unchanged": source_unchanged,
                 "activation_representation_before_fit": representation,
                 "activation_representation_after_fit": after_representation,
                 "activation_representation_unchanged": representation_equal,
                 "activation_representation_error": representation_error,
                 "fit_error": f"{type(fit_error).__name__}: {fit_error}" if fit_error else None,
                 "passed": snapshot_equal and source_unchanged and representation_equal and fit_error is None}
        registry.immutable_json(snapshot_path, audit)
        if not audit["passed"]:
            registry.append_event(campaign, "basis_fit_failed", basis_id=basis_id,
                                  snapshot_audit_path=str(snapshot_path), snapshot_audit_sha256=registry.file_sha256(snapshot_path))
            if fit_error is not None:
                raise fit_error
            raise ValueError("Model snapshot, activation representation, or source provenance changed during fitting; no usable basis published")
        array = np.asarray(fitted["basis"], dtype=np.float32)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        raw = buffer.getvalue()
        file_sha = hashlib.sha256(raw).hexdigest()
        training_aliases = sorted(set().union(*(_aliases(job) for job in jobs)))
        fit_metadata = fitted["metadata"]
        metadata = {"format": "registered_task_basis_v1", "basis_id": basis_id, "partition": "discovery",
                    "training_only": True, "basis_path": str(basis_path), "basis_file_sha256": file_sha,
                    "basis_array_sha256": fit_metadata["basis_sha256"], "request_fingerprint": request_hash,
                    "source_experiment_id": source_experiment_id, "source_condition": condition, "family": family,
                    "source_spec_sha256": source_sha, "source_spec_path": str(source_path.resolve()),
                    "source_job_ids": [job["job_id"] for job in jobs],
                    "source_job_fingerprints": source_job_fingerprints,
                    "selected_source_definitions_sha256": definitions_sha,
                    "seed_selection": "smallest predeclared seed per distinct pair; no outcome-based selection",
                    "layer": layer, "site": site, "rank": rank, "hidden_dim": array.shape[0],
                    "training_layout_ids": fit_metadata["training_layout_ids"],
                    "training_layout_aliases": training_aliases, "training_pair_ids": fit_metadata["training_pair_ids"],
                    "model": deepcopy(engine.metadata), "model_path": str(model_path), "model_files": model_files, "runtime": runtime,
                    "activation_representation": representation,
                    "model_provenance_version": 2, "model_snapshot_identity": before,
                    "model_snapshot_identity_sha256": before["identity_sha256"],
                    "model_snapshot_verified_before_and_after_fit": True,
                    "model_snapshot_audit_path": str(snapshot_path),
                    "model_snapshot_audit_sha256": registry.file_sha256(snapshot_path),
                    "validation_path": str(validation_path), "validation_sha256": registry.file_sha256(validation_path),
                    "fit": fit_metadata, "created_at": registry.now_utc(),
                    "interpretation": "Task-contrast covariance basis; causal and semantic roles remain untested. Input-token fits can capture literal lexical substitutions rather than an abstract belief variable. Rank 4 with four independent one-position training contrasts can span every training vector by construction; training reconstruction is not held-out generalization.",
                    "scientific_limits": {
                        "lexical_substitution_confound_not_excluded": True,
                        "rank_equals_training_position_count": rank == fit_metadata["n_position_rows"],
                        "n_distinct_source_pairs": fit_metadata["n_distinct_source_pairs"],
                        "training_reconstruction_is_generalization_evidence": False,
                        "semantic_headline_eligible": False, "independent_rescue_evidence": False}}
        _immutable_bytes(basis_path, raw)
        registry.immutable_json(metadata_path, metadata)
        registry.append_event(campaign, "basis_fit_completed", basis_id=basis_id, basis_file_sha256=file_sha,
                              training_pairs=len(jobs), rank=rank)
        return {**metadata, "basis_metadata_path": str(metadata_path),
                "basis_metadata_sha256": registry.file_sha256(metadata_path)}


def build_subspace_spec(source_spec: dict[str, Any], basis_metadata: dict[str, Any],
                        basis_path: str | Path, basis_sha256: str, experiment_id: str,
                        natural: bool = True, cap: int = 4096,
                        doses: tuple[float, ...] = (0.5, 1.0)) -> dict[str, Any]:
    """Pure held-out-discovery protocol; loading and SHA checks belong to compiler."""
    from .campaign import make_job

    _name(experiment_id)
    if source_spec.get("partition") != "discovery" or basis_metadata.get("partition") != "discovery":
        raise ValueError("Subspace search and basis training must both be discovery-only")
    if type(natural) is not bool or type(cap) is not int or cap < 1:
        raise ValueError("Invalid natural flag or generation cap")
    doses = tuple(float(dose) for dose in doses)
    if not doses or len(set(doses)) != len(doses) or any(not np.isfinite(dose) or not 0 < dose <= 1 for dose in doses):
        raise ValueError("doses must be unique finite interpolation fractions in (0,1]")
    basis_path = Path(basis_path)
    metadata_path = basis_metadata.get("basis_metadata_path")
    metadata_sha = basis_metadata.get("basis_metadata_sha256")
    if not basis_path.is_absolute() or not metadata_path or not Path(metadata_path).is_absolute():
        raise ValueError("Absolute immutable basis and metadata paths are required")
    if any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in (basis_sha256, metadata_sha)):
        raise ValueError("Both basis and metadata file SHA-256 values are required")
    if basis_metadata.get("basis_file_sha256") != basis_sha256:
        raise ValueError("Basis file hash disagrees with registered metadata")
    layer, site, rank = basis_metadata["layer"], basis_metadata["site"], basis_metadata["rank"]
    if type(layer) is not int or not 0 <= layer < 24 or site not in localization.SITES or type(rank) is not int or rank < 1:
        raise ValueError("Invalid registered basis geometry")
    training_aliases = set(basis_metadata["training_layout_ids"]) | set(basis_metadata.get("training_layout_aliases", []))
    if not training_aliases:
        raise ValueError("Basis training-layout provenance is required")
    intended = [job for job in source_spec["jobs"] if job["condition"] == basis_metadata["source_condition"]
                and job["family"] == basis_metadata["family"]]
    if not intended:
        raise ValueError("No intended held-out replacement jobs match the fitted condition/family")
    if any(_aliases(job) & training_aliases for job in intended):
        raise ValueError("Held-out evaluation overlaps basis-training layout identities or aliases")
    jobs = []
    pair_layout = {}
    seen = set()
    for source in intended:
        if source.get("experiment_id") != source_spec.get("experiment_id"):
            raise ValueError("Held-out job experiment does not match its source specification")
        if source["partition"] != "discovery" or not isinstance(source.get("donor_text"), str):
            raise ValueError("Held-out contrasts need discovery recipient and donor inputs")
        source_intervention = source.get("intervention", {})
        if source_intervention.get("operation", "replace") != "replace" or source_intervention.get("random_control") or source_intervention.get("zero_donor"):
            raise ValueError("Source condition must be an intended full-replacement contrast")
        key = source["pair_id"], source["seed"]
        if key in seen:
            raise ValueError("Duplicate held-out pair/seed source definition")
        seen.add(key)
        pair_layout[source["pair_id"]] = source["layout_id"]
        explicit = _explicit_positions(source)
        overrides = [source.get(role + "_token_ids") for role in ("recipient", "donor")]
        if any(value is not None for value in overrides):
            if any(not isinstance(value, list) or not value or any(type(i) is not int or i < 0 for i in value)
                   for value in overrides):
                raise ValueError("Frozen donor and recipient token IDs must both be valid integer lists")
        basis_ref = {"basis_path": str(basis_path), "basis_sha256": basis_sha256,
                     "basis_metadata_path": str(metadata_path), "basis_metadata_sha256": metadata_sha}

        def add(condition: str, operation: str | None, dose: float = 1.0, *, reverse: bool = False,
                identity: bool = False, zero: bool = False, random: bool = False,
                equivalence: bool = False, projected_norm_matched: bool = False,
                audit_norm: bool = False) -> None:
            value = {name: deepcopy(item) for name, item in source.items()
                     if name not in {"job_id", "experiment_id", "condition", "intervention", "metadata"}}
            value.update({"natural": natural, "max_new_tokens": cap, "semantic_test": False})
            if reverse:
                value["recipient_text"], value["donor_text"] = source["donor_text"], source["recipient_text"]
                if source.get("donor_token_ids") is not None:
                    value["recipient_token_ids"], value["donor_token_ids"] = source["donor_token_ids"], source["recipient_token_ids"]
            if identity:
                value["donor_text"] = value["recipient_text"]
                if value.get("recipient_token_ids") is not None:
                    value["donor_token_ids"] = value["recipient_token_ids"]
            if operation is None:
                value["donor_text"] = None
                value["donor_token_ids"] = None
            intervention = {"layers": [] if operation is None else [layer], "site": site,
                            "operation": operation or "replace", "dose": dose}
            if operation == "subspace":
                intervention.update(basis_ref)
                if zero:
                    intervention["zero_donor"] = True
                if random:
                    intervention.update({"random_basis": True, "random_seed": int(source["seed"]), "rank": rank})
                if projected_norm_matched:
                    intervention.update({"projected_norm_matched_random_basis": True,
                                         "random_seed": int(source["seed"]), "rank": rank})
                if audit_norm or projected_norm_matched:
                    intervention.update({"audit_projected_norm": True,
                                         "projected_norm_relative_tolerance": PROJECTED_NORM_RELATIVE_TOLERANCE})
            meta = deepcopy(source.get("metadata", {}))
            if explicit is not None:
                recipient_positions, donor_positions = deepcopy(explicit[::-1] if reverse else explicit)
                if identity:
                    donor_positions = deepcopy(recipient_positions)
                value["recipient_positions"], value["donor_positions"] = recipient_positions, donor_positions
                value["selection"] = {"scope": "explicit", "recipient_positions": recipient_positions,
                                      "donor_positions": donor_positions}
                meta["recipient_patch_positions"], meta["donor_patch_positions"] = recipient_positions, donor_positions
            meta.update({"source_experiment_id": source_spec["experiment_id"], "source_job_id": source["job_id"],
                         "basis_id": basis_metadata["basis_id"], "basis_training_layout_ids": sorted(training_aliases),
                         "basis_training_disjoint": True, "rank": rank, "semantic_headline_eligible": False,
                         "paired_baseline_condition": "donor_text" if reverse else "baseline",
                         "random_control_matching": (
                             "rank_and_per_token_applied_norm_against_task_projected_delta" if projected_norm_matched
                             else "rank_only_not_projected_delta_norm" if random else None),
                         "projected_norm_audit": {
                             "required": True,
                             "reference": "same recipient/donor task-basis projected delta, scaled by dose",
                             "relative_tolerance": PROJECTED_NORM_RELATIVE_TOLERANCE,
                             "zero_reference_requires_exact_zero": True,
                             "failures_are_technical_errors_without_outcome_based_reselection": True,
                         } if audit_norm or projected_norm_matched else None,
                         "random_orientation_policy": "one predeclared QR orientation per source seed; no outcome-based seed selection" if random or projected_norm_matched else None,
                         "random_subspace_note": "Per-token scaling of a random projected direction matches the task-projected norm; this is not an unmodified orthogonal-subspace interchange." if projected_norm_matched else None,
                         "independent_rescue_evidence": False, "engineering_identity_equivalence": identity,
                         "restoration_equivalence_only": equivalence,
                         "removal_note": "Projection-to-zero is an off-manifold intervention; damage is not semantic necessity." if zero else None,
                         "restoration_note": "Removal followed by replacement in the same projector is algebraically an identity; this job evaluates that identity-equivalent endpoint, not independent rescue." if equivalence else None})
            jobs.append(make_job(experiment_id, condition=condition, intervention=intervention, metadata=meta, **value))

        add("baseline", None)
        add("donor_text", None, reverse=True)
        add("full_transfer", "replace")
        add("subspace_identity", "subspace", identity=True)
        add("subspace_removal", "subspace", reverse=True, zero=True)
        add("restoration_equivalent", "subspace", identity=True, equivalence=True)
        for dose in doses:
            label = format(dose, ".6g").replace(".", "p")
            add(f"subspace_forward_{label}", "subspace", dose, audit_norm=True)
            add(f"subspace_reverse_{label}", "subspace", dose, reverse=True)
            add(f"random_subspace_{label}", "subspace", dose, random=True)
            add(f"projected_norm_matched_random_subspace_{label}", "subspace", dose, projected_norm_matched=True)
    pair_ids = sorted(pair_layout)
    comparisons = []

    def compare(treatment: str, control: str, direction: str, role: str) -> None:
        comparisons.append({"id": f"{experiment_id}_{treatment}_vs_{control}", "experiment_id": experiment_id,
                            "partition": "discovery", "treatment": treatment, "control": control,
                            "eligible_pair_ids": pair_ids, "expected_n_layouts": len(set(pair_layout.values())),
                            "expected_n_pairs": len(intended), "predicted_direction": direction, "role": role,
                            "semantic_headline_eligible": False, "independent_rescue_evidence": False})

    compare("full_transfer", "baseline", "increase", "full_state_reference")
    compare("subspace_identity", "baseline", "zero", "engineering_identity")
    compare("restoration_equivalent", "baseline", "zero", "algebraic_identity_not_rescue")
    compare("subspace_removal", "donor_text", "decrease", "counterfactual_projection_removal_diagnostic")
    for dose in doses:
        label = format(dose, ".6g").replace(".", "p")
        compare(f"subspace_forward_{label}", "baseline", "increase", "held_out_subspace_sufficiency")
        compare(f"subspace_forward_{label}", f"random_subspace_{label}", "increase", "rank_only_random_comparison_not_norm_specificity")
        compare(f"subspace_forward_{label}", f"projected_norm_matched_random_subspace_{label}", "increase", "rank_and_projected_norm_matched_random_comparison")
        compare(f"subspace_reverse_{label}", "donor_text", "decrease", "conditional_reverse_interchange")
    return {"schema_version": registry.SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "discovery",
            "hypothesis": "A discovery-fitted task-contrast subspace may transfer or disrupt action-relevant information on disjoint discovery layouts; covariance and zeroing alone do not establish semantic mediation.",
            "jobs": jobs, "comparisons": comparisons, "basis": deepcopy(basis_metadata),
            "basis_file_sha256": basis_sha256, "source_spec_fingerprint": registry.fingerprint(source_spec),
            "primary_estimand": "layout-weighted paired native target-action ITT probability difference",
            "protocol": {"layer": layer, "site": site, "rank": rank, "doses": list(doses),
                         "held_out_training_alias_disjoint": True, "requires_basis_hash_verification": True,
                         "requires_random_subspace_and_zero_donor_support": True,
                         "requires_projected_norm_matched_random_basis_support": True,
                         "projected_norm_relative_tolerance": PROJECTED_NORM_RELATIVE_TOLERANCE,
                         "projected_norm_audit_applies_to": ["subspace_forward", "projected_norm_matched_random_subspace"],
                         "zero_reference_requires_exact_zero": True,
                         "projected_norm_failure_policy": "Retain technical failures and block complete-cohort claims; no tolerance or seed changes under the registered ID.",
                         "semantic_headline_eligible": False}}
