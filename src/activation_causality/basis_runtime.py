"""Verify registered bases against one materialized local model instance.

The worker hashes weights around actual loading. This module checks that proof
and small basis/source artifacts; it never loads a model or rereads weight shards.
Legacy basis metadata remains readable but cannot authorize new execution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .localization import _array_fingerprint
from .model_identity import FORMAT as SNAPSHOT_FORMAT
from .registry import fingerprint


CAPTURE_CONVENTION = "full-prefill; exact-input-token-ids; site as declared; float32 capture of model activations"


def activation_representation(runtime: dict[str, Any]) -> dict[str, Any]:
    from .engine import engine_dependency_hashes

    engine_sha = runtime.get("sources", {}).get("src/activation_causality/engine.py")
    versions = {name: runtime.get("versions", {}).get(name) for name in ("mlx", "mlx-lm", "numpy")}
    if not _digest(engine_sha) or any(not isinstance(value, str) or not value for value in versions.values()):
        raise ValueError("activation representation requires the engine source hash and numerical runtime versions")
    return {"engine_sha256": engine_sha, "versions": versions, "capture_convention": CAPTURE_CONVENTION,
            "engine_dependency_sources": engine_dependency_hashes(runtime)}


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("a full verified model snapshot is required")
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    shards, files = value.get("indexed_shards"), value.get("files")
    if (value.get("format") != SNAPSHOT_FORMAT or value.get("identity_sha256") != fingerprint(body)
            or value.get("index_inventory_exact") is not True or value.get("stable_during_read") is not True
            or not isinstance(shards, list) or not shards or not isinstance(files, dict)
            or value.get("actual_weight_files") != shards or any(name not in files for name in shards)
            or any(not _digest(digest) for digest in files.values())):
        raise ValueError("model snapshot self-digest or complete weight inventory is invalid")
    return value


@dataclass(frozen=True)
class LoadedModelProof:
    engine_id: int
    snapshot: dict[str, Any]
    representation: dict[str, Any]


def make_loaded_model_proof(engine: Any, before_load: dict[str, Any], after_load: dict[str, Any],
                            runtime: dict[str, Any]) -> LoadedModelProof:
    """Call only after materializing model weights, between the two full hashes."""
    _snapshot(before_load)
    _snapshot(after_load)
    if before_load != after_load:
        raise ValueError("model snapshot changed across actual model loading")
    if engine.metadata.get("loaded") is not True or engine.head_mode != "full":
        raise ValueError("loaded-model proof requires an already materialized full-head engine")
    if str(engine.model_path.resolve()) != before_load.get("path"):
        raise ValueError("loaded-model snapshot belongs to another model path")
    return LoadedModelProof(id(engine), deepcopy(before_load), activation_representation(runtime))


def _read_bound(path_value: Any, expected_sha: Any, label: str) -> tuple[Path, bytes]:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute() or not _digest(expected_sha):
        raise ValueError(f"{label} requires an absolute path and SHA256")
    path = Path(path_value)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"{label} file fingerprint mismatch")
    return path, raw


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _aliases(job: dict[str, Any]) -> set[str]:
    metadata = job.get("metadata", {})
    aliases = {job["layout_id"], *job.get("layout_aliases", [])}
    for name in ("pair", "source_pair"):
        nested = metadata.get(name, {})
        if isinstance(nested.get("layout_id"), str) and nested["layout_id"]:
            aliases.add(nested["layout_id"])
        aliases.update(nested.get("metadata", {}).get("variant_layout_ids", []))
    return aliases


def _verify_fit_audit(metadata: dict[str, Any]) -> None:
    # The audit's request digest must identify the actual immutable request,
    # not merely an arbitrary string repeated in two mutable objects.
    request_path = Path(metadata["basis_path"]).with_suffix(".request.json")
    request = _object(request_path.read_bytes(), "basis fit request")
    if fingerprint(request) != metadata.get("request_fingerprint"):
        raise ValueError("basis fit request fingerprint mismatch")
    fields = {name: name for name in (
        "source_experiment_id", "source_spec_sha256", "source_spec_path", "source_job_fingerprints",
        "selected_source_definitions_sha256", "family", "layer", "site", "rank", "model_path",
        "model_files", "model_provenance_version", "model_snapshot_identity", "model_snapshot_identity_sha256",
        "activation_representation",
    )}
    fields.update(condition="source_condition", selected_job_ids="source_job_ids")
    if any(key not in request or request[key] != metadata.get(value) for key, value in fields.items()):
        raise ValueError("basis fit request and metadata source/model/geometry provenance disagree")
    _, raw = _read_bound(metadata.get("model_snapshot_audit_path"),
                         metadata.get("model_snapshot_audit_sha256"), "basis snapshot audit")
    audit = _object(raw, "basis snapshot audit")
    if (audit.get("format") != "basis_fit_model_snapshot_audit_v1" or audit.get("passed") is not True
            or audit.get("snapshots_equal") is not True or audit.get("source_and_campaign_unchanged") is not True
            or audit.get("snapshot_error") is not None or audit.get("fit_error") is not None
            or audit.get("basis_id") != metadata.get("basis_id")
            or audit.get("request_fingerprint") != metadata.get("request_fingerprint")
            or audit.get("before_fit") != metadata["model_snapshot_identity"]
            or audit.get("after_fit") != metadata["model_snapshot_identity"]
            or audit.get("activation_representation_unchanged") is not True
            or audit.get("activation_representation_error") is not None
            or audit.get("activation_representation_before_fit") != metadata["activation_representation"]
            or audit.get("activation_representation_after_fit") != metadata["activation_representation"]):
        raise ValueError("basis fit snapshot audit does not prove this fit's model and source stability")


def _verify_source(metadata: dict[str, Any]) -> None:
    source_path, raw = _read_bound(metadata.get("source_spec_path"), metadata.get("source_spec_sha256"),
                                  "basis source specification")
    source = _object(raw, "basis source specification")
    source_id = metadata.get("source_experiment_id")
    if (source.get("partition") != "discovery" or source.get("experiment_id") != source_id
            or source_path.name != f"{source_id}.json"
            or (source_path.parent.parent / "reviews" / f"{source_id}_retired.json").exists()):
        raise ValueError("basis source must remain the registered nonretired discovery specification")
    rows = source.get("jobs", [])
    by_id = {row["job_id"]: row for row in rows}
    selected_ids = metadata.get("source_job_ids")
    source_hashes = metadata.get("source_job_fingerprints")
    if (len(by_id) != len(rows) or not isinstance(selected_ids, list) or not selected_ids
            or len(set(selected_ids)) != len(selected_ids) or not isinstance(source_hashes, dict)
            or set(source_hashes) != set(selected_ids) or not set(selected_ids) <= set(by_id)):
        raise ValueError("basis source job lineage is incomplete or duplicated")
    selected = [by_id[identifier] for identifier in selected_ids]
    for row in selected:
        original_definition = {key: value for key, value in row.items() if key != "job_id"}
        if (fingerprint(original_definition)[:24] != row["job_id"] or row.get("experiment_id") != source_id
                or fingerprint(row) != source_hashes[row["job_id"]] or row.get("partition") != "discovery"
                or row.get("condition") != metadata["source_condition"] or row.get("family") != metadata["family"]
                or row.get("intervention", {}).get("layers") != [metadata["layer"]]
                or row.get("intervention", {}).get("site") != metadata["site"]):
            raise ValueError("basis source job fingerprints or extraction geometry disagree")
    # Reconstruct the fit's explicitly normalized source records, preserving
    # their original ordered token IDs and every provenance field.
    normalized = deepcopy(selected)
    for row in normalized:
        resolved = []
        for role in ("recipient", "donor"):
            positions = [mapping[key] for mapping, key in (
                (row, role + "_positions"), (row.get("selection", {}), role + "_positions"),
                (row.get("metadata", {}), role + "_patch_positions"),
            ) if mapping.get(key) is not None]
            if positions and any(value != positions[0] for value in positions[1:]):
                raise ValueError("basis source has conflicting explicit positions")
            resolved.append(positions[0] if positions else None)
        if any(value is not None for value in resolved):
            if any(value is None for value in resolved):
                raise ValueError("basis source must provide both explicit spans")
            row["recipient_positions"], row["donor_positions"] = resolved
    if fingerprint(normalized) != metadata.get("selected_source_definitions_sha256"):
        raise ValueError("basis normalized extraction-source definitions disagree")
    aliases = set().union(*(_aliases(row) for row in selected))
    layouts = {row["layout_id"] for row in selected}
    pairs = {row["pair_id"] for row in selected}
    if (set(metadata.get("training_layout_aliases", [])) != aliases
            or set(metadata.get("training_layout_ids", [])) != layouts
            or set(metadata.get("training_pair_ids", [])) != pairs
            or len(pairs) != len(selected) or metadata["rank"] > len(pairs)):
        raise ValueError("basis training membership or independent rank bound disagrees with its source")


def load_verified_basis(intervention: dict[str, Any], job: dict[str, Any], engine: Any,
                        proof: LoadedModelProof) -> tuple[np.ndarray, dict[str, Any]]:
    """Read/hash exactly the bytes subsequently parsed, without model inference."""
    if not isinstance(proof, LoadedModelProof) or proof.engine_id != id(engine):
        raise ValueError("subspace compilation requires this engine instance's loaded-model proof")
    snapshot = _snapshot(proof.snapshot)
    if (engine.metadata.get("loaded") is not True or engine.head_mode != "full"
            or str(engine.model_path.resolve()) != snapshot.get("path")):
        raise ValueError("loaded-model proof no longer matches this engine")
    _, raw_metadata = _read_bound(intervention.get("basis_metadata_path"),
                                  intervention.get("basis_metadata_sha256"), "subspace metadata")
    metadata = _object(raw_metadata, "subspace metadata")
    if (metadata.get("format") != "registered_task_basis_v1" or metadata.get("model_provenance_version") != 2
            or metadata.get("partition") != "discovery" or metadata.get("training_only") is not True
            or metadata.get("model_snapshot_verified_before_and_after_fit") is not True):
        raise ValueError("new subspace execution requires verified version-2 discovery basis provenance")
    basis_snapshot = _snapshot(metadata.get("model_snapshot_identity"))
    if (basis_snapshot != snapshot or metadata.get("model_snapshot_identity_sha256") != snapshot["identity_sha256"]
            or metadata.get("model_files") != snapshot["files"] or metadata.get("model_path") != snapshot["path"]):
        raise ValueError("basis was fitted with another model snapshot")
    if metadata.get("activation_representation") != proof.representation:
        raise ValueError("basis extraction engine, numerical runtime or capture convention changed")
    if (job.get("family") != metadata.get("family") or intervention.get("operation") != "subspace"
            or intervention.get("site") != metadata.get("site") or intervention.get("layers") != [metadata.get("layer")]):
        raise ValueError("basis family or intervention site/layer mismatch")
    hidden, rank = metadata.get("hidden_dim"), metadata.get("rank")
    if (type(hidden) is not int or hidden != engine.metadata.get("hidden_dim")
            or type(rank) is not int or not 1 <= rank <= hidden
            or ("rank" in intervention and intervention["rank"] != rank)):
        raise ValueError("registered basis geometry differs from the model or intervention")
    training_aliases = set(metadata.get("training_layout_ids", [])) | set(metadata.get("training_layout_aliases", []))
    if not training_aliases or _aliases(job) & training_aliases:
        raise ValueError("basis evaluation overlaps training layout identities or nested aliases")
    _verify_fit_audit(metadata)
    _verify_source(metadata)
    basis_path, raw_basis = _read_bound(intervention.get("basis_path"), intervention.get("basis_sha256"), "subspace basis")
    if (metadata.get("basis_file_sha256") != intervention.get("basis_sha256")
            or metadata.get("basis_path") != str(basis_path)):
        raise ValueError("basis file binding disagrees with registered metadata")
    basis = np.load(io.BytesIO(raw_basis), allow_pickle=False)
    if (not isinstance(basis, np.ndarray) or basis.dtype != np.dtype("float32")
            or basis.shape != (hidden, rank) or not np.isfinite(basis).all()
            or _array_fingerprint(basis) != metadata.get("basis_array_sha256")):
        raise ValueError("basis array format, shape or content fingerprint mismatch")
    gram = np.einsum("hr,hs->rs", basis.astype(np.float64), basis.astype(np.float64), optimize=False)
    if np.max(np.abs(gram - np.eye(rank))) > 1e-5:
        raise ValueError("basis columns are not orthonormal")
    return basis, metadata
