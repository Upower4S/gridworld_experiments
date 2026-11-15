"""CPU-only authenticity checks before freezing an existing discovery basis.

Caller-supplied model snapshot and activation representation must already be
verified by the outer freeze/readiness gate. No model weights are read, no model
is loaded, and no LoadedModelProof is constructed. Runtime inference retains its
separate mandatory materialized-engine proof.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import basis_runtime
from .localization import _array_fingerprint
from .registry import fingerprint

AUGMENTATION = frozenset({"basis_metadata_path", "basis_metadata_sha256"})
FAMILIES = frozenset({"wall", "key_possession", "door_status"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _equal(left: Any, right: Any) -> bool:
    """Canonical JSON equality preserves numeric and Boolean binding types."""
    return fingerprint(left) == fingerprint(right)


def _object(raw: bytes, label: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant in {label}: {value}")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        _require(isinstance(value, dict), f"{label} must be an object")
        fingerprint(value)  # Also rejects exponent overflow such as 1e999.
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid finite JSON object for {label}: {exc}") from exc
    return value


def validate_confirmation_basis(
    configuration: Mapping[str, Any], basis_metadata: Mapping[str, Any], *,
    model_snapshot_identity: Mapping[str, Any], activation_representation: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Return exact augmented metadata and a complete observed-stable manifest.

    Every payload is read, hashed and parsed from the same captured bytes. The
    unchanged runtime fit/source validators are reused as additional checks;
    canonical semantic comparisons below cover their parsed artifacts too.
    All five artifacts and the absent-retirement guard are rechecked globally
    after validation. This detects observed mutations, not adversarial changes
    reverted between reads; outer immutable storage remains required.
    """
    _require(isinstance(configuration, Mapping) and isinstance(basis_metadata, Mapping),
             "configuration and embedded basis metadata must be mappings")
    config, embedded = deepcopy(dict(configuration)), deepcopy(dict(basis_metadata))
    fingerprint(config)
    fingerprint(embedded)
    _require(isinstance(model_snapshot_identity, Mapping) and isinstance(activation_representation, Mapping),
             "caller-verified model snapshot and representation must be mappings")
    supplied_snapshot = basis_runtime._snapshot(deepcopy(dict(model_snapshot_identity)))
    _require(isinstance(supplied_snapshot.get("path"), str) and Path(supplied_snapshot["path"]).is_absolute(),
             "caller-verified model snapshot requires its absolute path")
    model_root = Path(supplied_snapshot["path"]).resolve()
    _require(basis_runtime._digest(config.get("basis_metadata_sha256")), "basis metadata requires its raw-file SHA256")
    artifacts, payloads, semantic = {}, {}, {}

    def capture(path_value: Any, label: str, expected_sha: str | None = None, *, parse=True):
        _require(isinstance(path_value, str) and Path(path_value).is_absolute(), f"{label} requires an absolute path")
        path = Path(path_value).resolve(strict=True)
        _require(not path.is_relative_to(model_root), "basis audit may not read model-directory artifacts")
        _require(str(path) == path_value, "basis provenance requires canonical paths without symbolic-link aliases")
        _require(path.suffix == (".json" if parse else ".npy"), f"{label} has an unsupported artifact suffix")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None:
            _require(basis_runtime._digest(expected_sha) and digest == expected_sha, f"{label} raw-file fingerprint mismatch")
        key = str(path)
        _require(key not in artifacts, "basis provenance artifacts must have distinct paths")
        artifacts[key], payloads[label] = digest, raw
        if parse:
            value = _object(raw, label)
            semantic[label] = fingerprint(value)
            return path, value
        return path, raw

    metadata_path, metadata = capture(config.get("basis_metadata_path"), "metadata", config.get("basis_metadata_sha256"))
    _require(AUGMENTATION.isdisjoint(metadata), "on-disk metadata must not contain self-referential path/hash augmentation")
    augmented = {**metadata, "basis_metadata_path": config["basis_metadata_path"],
                 "basis_metadata_sha256": config["basis_metadata_sha256"]}
    _require(_equal(embedded, augmented), "embedded metadata must exactly equal on-disk metadata plus the two path/hash fields")
    _require(basis_runtime._digest(config.get("basis_metadata_fingerprint"))
             and config["basis_metadata_fingerprint"] == fingerprint(augmented), "frozen augmented metadata fingerprint differs")
    _require(metadata.get("format") == "registered_task_basis_v1" and type(metadata.get("model_provenance_version")) is int
             and metadata["model_provenance_version"] == 2 and metadata.get("partition") == "discovery"
             and metadata.get("training_only") is True and metadata.get("model_snapshot_verified_before_and_after_fit") is True,
             "confirmation requires version-2 discovery basis provenance")
    _require(config.get("family") in FAMILIES and type(config.get("rank")) is int and config["rank"] in (1, 4)
             and type(config.get("layer")) is int and config["layer"] == 0 and config.get("site") == "resid_post",
             "confirmation basis requires one fixed family, layer0 resid_post and rank1 or4")
    for name in ("family", "rank", "layer", "site"):
        _require(name in metadata and _equal(config[name], metadata[name]), f"basis/configuration {name} differs")
    _require(metadata.get("source_condition") == "factspan_layer0", "confirmation basis requires the fixed input-fact source condition")
    hidden, rank = metadata.get("hidden_dim"), metadata["rank"]
    _require(type(hidden) is int and hidden > 0 and rank <= hidden, "invalid basis hidden dimension/rank")
    for name in ("operation", "layers"):
        if name in config:
            _require(_equal(config[name], "subspace" if name == "operation" else [metadata["layer"]]),
                     "contradictory frozen basis intervention geometry")
    _require(config.get("basis_path") == metadata.get("basis_path")
             and config.get("basis_sha256") == metadata.get("basis_file_sha256")
             and basis_runtime._digest(config.get("basis_sha256")), "basis array file binding differs from metadata")
    if "intervention" in config:
        intervention = config["intervention"]
        fields = {"layers": [metadata["layer"]], "site": metadata["site"], "operation": "subspace",
                  **{key: config[key] for key in ("basis_path", "basis_sha256", "basis_metadata_path", "basis_metadata_sha256")}}
        if "selected_dose" in config:
            fields["dose"] = config["selected_dose"]
        _require(isinstance(intervention, Mapping) and all(key in intervention and _equal(intervention[key], value)
                                                         for key, value in fields.items()),
                 "selected intervention basis binding differs from the frozen configuration")
    if "basis_id" in config:
        _require(_equal(config["basis_id"], metadata.get("basis_id")), "frozen basis identity differs")
    stored_snapshot = basis_runtime._snapshot(metadata.get("model_snapshot_identity"))
    _require(_equal(stored_snapshot, supplied_snapshot)
             and metadata.get("model_snapshot_identity_sha256") == supplied_snapshot["identity_sha256"]
             and _equal(metadata.get("model_files"), supplied_snapshot["files"])
             and metadata.get("model_path") == supplied_snapshot["path"], "basis model snapshot differs from caller-verified snapshot")
    _require(isinstance(activation_representation, Mapping) and bool(activation_representation)
             and _equal(metadata.get("activation_representation"), dict(activation_representation)),
             "basis activation representation differs from caller-verified representation")
    request_path = str(Path(metadata["basis_path"]).with_suffix(".request.json"))
    _, request = capture(request_path, "fit_request")
    _, fit_audit = capture(metadata.get("model_snapshot_audit_path"), "fit_audit", metadata.get("model_snapshot_audit_sha256"))
    source_path, source = capture(metadata.get("source_spec_path"), "source_spec", metadata.get("source_spec_sha256"))
    retirement = source_path.parent.parent / "reviews" / (metadata["source_experiment_id"] + "_retired.json")
    _require(not retirement.exists(), "basis training source is retired")
    _require(fingerprint(request) == metadata.get("request_fingerprint"), "immutable fit request fingerprint differs")
    request_fields = {name: name for name in (
        "source_experiment_id", "source_spec_sha256", "source_spec_path", "source_job_fingerprints",
        "selected_source_definitions_sha256", "family", "layer", "site", "rank", "model_path",
        "model_files", "model_provenance_version", "model_snapshot_identity", "model_snapshot_identity_sha256",
        "activation_representation")}
    request_fields.update(condition="source_condition", selected_job_ids="source_job_ids")
    _require(all(key in request and _equal(request[key], metadata.get(value)) for key, value in request_fields.items()),
             "fit request canonical source/model/geometry bindings differ")
    for name, expected in {
        "before_fit": supplied_snapshot, "after_fit": supplied_snapshot,
        "activation_representation_before_fit": dict(activation_representation),
        "activation_representation_after_fit": dict(activation_representation),
    }.items():
        _require(name in fit_audit and _equal(fit_audit[name], expected), "fit audit canonical model/representation binding differs")
    # These existing validators also enforce all fit success flags, canonical
    # source job IDs, normalized extraction spans, selected IDs and alias sets.
    basis_runtime._verify_fit_audit(metadata)
    basis_runtime._verify_source(metadata)
    selected = {row["job_id"]: row for row in source["jobs"]}
    for job_id in metadata["source_job_ids"]:
        row = selected[job_id]
        _require(_equal(row["intervention"]["layers"], [metadata["layer"]])
                 and _equal(row["intervention"]["site"], metadata["site"]), "source extraction geometry has type/value drift")
    for name in ("training_layout_ids", "training_layout_aliases", "training_pair_ids"):
        values = metadata.get(name)
        _require(isinstance(values, list) and bool(values) and all(isinstance(v, str) and v for v in values)
                 and len(values) == len(set(values)), "training identity inventories must contain unique nonempty strings")
    basis_path, raw_basis = capture(config["basis_path"], "basis_array", config["basis_sha256"], parse=False)
    array = np.load(io.BytesIO(raw_basis), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        if hasattr(array, "close"):
            array.close()
        raise ValueError("basis file must contain one NumPy array")
    _require(array.dtype == np.dtype("float32") and array.shape == (hidden, rank) and np.isfinite(array).all()
             and _array_fingerprint(array) == metadata.get("basis_array_sha256"), "basis array dtype/shape/finite/content binding differs")
    values = array.astype(np.float64)
    gram = np.einsum("hr,hs->rs", values, values, optimize=False)
    error = float(np.max(np.abs(gram - np.eye(rank))))
    _require(error <= 1e-5, "basis columns are not orthonormal")
    # Global bracket includes every payload read by the unchanged validators.
    for path, digest in artifacts.items():
        _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                 "basis provenance artifact changed during validation: " + path)
    _require(not retirement.exists(), "basis source retirement changed during validation")
    manifest = {
        "format": "confirmation_basis_authenticity_v1", "artifact_sha256": artifacts,
        "parsed_semantic_fingerprints": semantic, "basis_array_sha256": _array_fingerprint(array),
        "augmented_metadata_fingerprint": fingerprint(augmented), "basis_id": metadata["basis_id"],
        "basis_path": str(basis_path), "metadata_path": str(metadata_path),
        "family": metadata["family"], "layer": metadata["layer"], "site": metadata["site"],
        "rank": rank, "hidden_dim": hidden, "orthonormality_max_abs_error": error,
        "training_layout_ids": metadata["training_layout_ids"], "training_layout_aliases": metadata["training_layout_aliases"],
        "training_pair_ids": metadata["training_pair_ids"],
        "model_snapshot_identity_sha256": supplied_snapshot["identity_sha256"],
        "activation_representation_sha256": fingerprint(dict(activation_representation)),
        "source_retirement_guard": {"path": str(retirement), "absent_before_and_after": True},
        "all_artifacts_observed_stable": True, "caller_model_and_code_verification_required": True,
        "model_weights_read": False, "model_materialization_claim": False,
    }
    return augmented, manifest
