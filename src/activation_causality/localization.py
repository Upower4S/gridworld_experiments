"""Discovery-only PCA of paired task-contrast activations.

This module does not load a model or write files. Fitting uses donor minus
recipient states, preserving token-position rows while giving every distinct
pair equal total weight. These directions describe contrast covariance; they
are not independent proof of semantic specificity, necessity, or mediation.

The intended rank sweep is 1, 4, 16, subject to distinct-pair and numerical-rank
limits. A direction's causal role must be tested separately on held-out layouts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


SITES = frozenset({"resid_post", "attn_out", "mlp_out"})
FORMAT = "task_contrast_pca_v1"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": "float32-le"}, sort_keys=True).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _discovery_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("partition") != "discovery":
        raise ValueError("subspace fitting accepts only explicit partition='discovery' rows")
    for name in ("donor_partition", "recipient_partition", "source_partition"):
        if name in row and row[name] != "discovery":
            raise ValueError(f"{name} must be discovery for a training contrast")
    # Check supplied nested environment provenance rather than trusting a
    # contradictory wrapper label. We cannot infer provenance that is omitted.
    nested = row.get("metadata", {})
    pair = nested.get("pair", {}) if isinstance(nested, Mapping) else {}
    if "source_pair_id" in row and isinstance(nested, Mapping) and "source_pair_id" in nested and row["source_pair_id"] != nested["source_pair_id"]:
        raise ValueError("source_pair_id provenance is contradictory")
    source_pair_id = row.get("source_pair_id", nested.get("source_pair_id") if isinstance(nested, Mapping) else None)
    if source_pair_id is not None and (not isinstance(source_pair_id, str) or not source_pair_id):
        raise ValueError("source_pair_id must be a nonempty string")
    if isinstance(pair, Mapping) and "partition" in pair and pair["partition"] != "discovery":
        raise ValueError("nested pair provenance is not discovery")
    for name in ("pair_id", "layout_id"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"every training row needs a nonempty string {name}")
    if isinstance(pair, Mapping):
        for name in ("pair_id", "layout_id"):
            if name in pair and pair[name] != row[name]:
                derived_pair = name == "pair_id" and source_pair_id == pair[name] and row[name].startswith(source_pair_id + "::")
                if not derived_pair:
                    raise ValueError(f"nested {name} contradicts the training row")
    result = {"pair_id": row["pair_id"], "layout_id": row["layout_id"], "partition": "discovery"}
    if source_pair_id is not None:
        result["source_pair_id"] = source_pair_id
    # Preserve useful JSON-only provenance without retaining complete prompts.
    for name in ("family", "source_experiment_id", "recipient_input_sha256", "donor_input_sha256",
                 "recipient_positions", "donor_positions", "source_definition_sha256", "source_pair_id"):
        if name in row:
            result[name] = json.loads(json.dumps(row[name], allow_nan=False))
    return result


def _rank_argument(rank: int, n_pairs: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or int(rank) < 1:
        raise ValueError("rank must be a positive integer")
    if int(rank) > n_pairs:
        raise ValueError(f"rank {rank} exceeds {n_pairs} distinct discovery training pairs")
    return int(rank)


def fit_task_contrast_subspace(
    contrasts: Sequence[np.ndarray] | np.ndarray,
    pair_metadata: Sequence[Mapping[str, Any]],
    rank: int,
    *,
    layer: int,
    site: str = "resid_post",
    center: bool = False,
) -> dict[str, Any]:
    """Return ``{'basis': float32[hidden, rank], 'metadata': JSON-object}``.

    Supply one [positions, hidden] donor-minus-recipient array per distinct
    pair. A [pairs, hidden] matrix is also accepted (one position per pair).
    Rows within a pair receive weight 1 / number_of_positions, so a longer
    span cannot gain more total weight than a shorter pair. Pair ordering is
    canonicalized before SVD; each returned column's largest entry is positive.

    The default uncentered SVD retains the mean task-contrast direction. Set
    center=True explicitly for weighted centered PCA; this choice is recorded.
    No test, development, historical, or confirmation rows are accepted.
    """
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or layer < 0:
        raise ValueError("layer must be a nonnegative integer")
    if site not in SITES:
        raise ValueError("unsupported activation site")
    rows = [_discovery_metadata(row) for row in pair_metadata]
    if not rows:
        raise ValueError("no discovery contrasts supplied")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("fit requires one contrast array per distinct pair; deduplicate before fitting")
    distinct_source_pairs = {row.get("source_pair_id", row["pair_id"]) for row in rows}
    rank = _rank_argument(rank, len(distinct_source_pairs))
    arrays = list(contrasts)
    if len(arrays) != len(rows):
        raise ValueError("contrast arrays and pair metadata have different lengths")
    prepared = []
    width = None
    for values, row in zip(arrays, rows, strict=True):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("each contrast must be a finite nonempty [positions, hidden] array")
        if width is None:
            width = values.shape[1]
        if values.shape[1] != width:
            raise ValueError("contrast hidden dimensions differ")
        prepared.append((row, values))
    if rank > width:
        raise ValueError("rank exceeds the hidden dimension")
    prepared.sort(key=lambda item: (item[0]["layout_id"], item[0]["pair_id"]))
    rows = [row for row, _ in prepared]
    arrays = [values for _, values in prepared]
    mean = np.mean([values.astype(np.float64).mean(axis=0) for values in arrays], axis=0)
    offset = mean if center else np.zeros(width, dtype=np.float64)
    # Each pair contributes the mean squared projection of its token rows.
    weighted = np.concatenate([(values.astype(np.float64) - offset) / np.sqrt(len(values) * len(arrays))
                               for values in arrays], axis=0)
    _, singular_values, right = np.linalg.svd(weighted, full_matrices=False)
    threshold = float(np.finfo(np.float64).eps * max(weighted.shape) * singular_values[0])
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    if numerical_rank < rank:
        raise ValueError(f"requested rank {rank} exceeds numerical contrast rank {numerical_rank}")
    basis = right[:rank].T.copy()
    for column in range(rank):
        anchor = int(np.argmax(np.abs(basis[:, column])))
        if basis[anchor, column] < 0:
            basis[:, column] *= -1
    basis = np.ascontiguousarray(basis, dtype=np.float32)
    gram = np.einsum("ir,is->rs", basis.astype(np.float64), basis.astype(np.float64), optimize=False)
    orthogonality_error = float(np.max(np.abs(gram - np.eye(rank))))
    if not np.isfinite(orthogonality_error) or orthogonality_error > 1e-5:
        raise ValueError("fitted basis is not orthonormal after float32 conversion")
    pair_ids = [row["pair_id"] for row in rows]
    layout_ids = sorted({row["layout_id"] for row in rows})
    membership = [{"pair_id": row["pair_id"], "layout_id": row["layout_id"]} for row in rows]
    contrast_records = [{"pair_id": row["pair_id"], "positions": len(values), "sha256": _array_fingerprint(values)}
                        for row, values in prepared]
    total_energy = float(np.sum(singular_values * singular_values))
    metadata = {
        "format": FORMAT, "partition": "discovery", "training_only": True,
        "method": "centered_task_contrast_pca" if center else "uncentered_task_contrast_svd",
        "interpretation": "PCA of task contrasts; not independent proof of semantics, necessity, or causal mediation",
        "layer": int(layer), "site": site, "rank": rank, "hidden_dim": int(width),
        "n_training_pairs": len(rows), "n_training_layouts": len(layout_ids),
        "n_distinct_source_pairs": len(distinct_source_pairs),
        "rank_pair_count_basis": "distinct source_pair_id where supplied, otherwise pair_id; cuts cannot inflate pair count",
        "n_position_rows": int(sum(len(values) for values in arrays)),
        "weighting": "equal total weight per pair; equal weight per position within each pair",
        "centered": bool(center), "contrast_mean": mean.tolist(),
        "singular_values": singular_values.tolist(), "numerical_rank": numerical_rank,
        "numerical_rank_threshold": threshold,
        "explained_energy_fraction": ((singular_values[:rank] ** 2) / total_energy).tolist(),
        "captured_energy_fraction": float(np.sum(singular_values[:rank] ** 2) / total_energy),
        "orthogonality_max_abs_error": orthogonality_error,
        "training_pair_ids": pair_ids, "training_layout_ids": layout_ids,
        "training_pair_ids_sha256": _fingerprint(sorted(pair_ids)),
        "training_layout_ids_sha256": _fingerprint(layout_ids),
        "training_membership_sha256": _fingerprint(membership),
        "training_rows": rows, "contrast_records": contrast_records,
        "contrast_sha256": _fingerprint(contrast_records), "basis_sha256": _array_fingerprint(basis),
        "canonical_order": "layout_id then pair_id; largest-absolute component of each basis column positive",
        "numpy_version": np.__version__,
        "family_counts": dict(sorted(Counter(str(row.get("family", "unspecified")) for row in rows).items())),
    }
    # Fail here, before a caller tries to persist unexpectedly non-JSON metadata.
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    return {"basis": basis, "metadata": metadata}


def _positions(values: Any, length: int) -> list[int]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("explicit positions must be a sequence")
    result = list(values)
    if not result or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in result):
        raise ValueError("positions must be nonempty integer indices")
    result = [int(i) for i in result]
    if len(set(result)) != len(result) or min(result) < 0 or max(result) >= length:
        raise ValueError("positions must be distinct and in bounds")
    return result


def extract_and_fit(
    engine: Any,
    jobs: Sequence[Mapping[str, Any]],
    layer: int,
    site: str,
    rank: int,
    *,
    center: bool = False,
) -> dict[str, Any]:
    """Capture contrasts using an already-loaded, caller-owned engine.

    Prefilter jobs to the intended donor/recipient condition. Jobs without a
    donor are errors. Identical repeated seed/layer jobs for a pair are deduped;
    conflicting definitions of the same pair are errors. All partitions are
    checked before any encoding/capture. Explicit recipient_positions and
    donor_positions override selection; otherwise only the root worker's
    ``selection={'scope':'last_tokens','last_k':k}`` contract is supported.

    No engine constructor, loading method, generation, disk IO, or worker-lock
    acquisition is called here. The caller must own the single model worker.
    """
    if not getattr(engine, "metadata", {}).get("loaded", False):
        raise RuntimeError("extract_and_fit requires an already-loaded caller-owned engine")
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or layer < 0 or site not in SITES:
        raise ValueError("invalid extraction layer or site")
    rows = [_discovery_metadata(job) for job in jobs]
    unique: dict[str, tuple[Mapping[str, Any], dict[str, Any], str]] = {}
    for job, row in zip(jobs, rows, strict=True):
        if not isinstance(job.get("recipient_text"), str) or not isinstance(job.get("donor_text"), str):
            raise ValueError("prefilter jobs to explicit donor/recipient text contrasts")
        definition = {"layout_id": row["layout_id"], "recipient_text": job["recipient_text"],
                      "donor_text": job["donor_text"], "selection": job.get("selection"),
                      "recipient_positions": job.get("recipient_positions"), "donor_positions": job.get("donor_positions"),
                      "recipient_token_ids": job.get("recipient_token_ids"), "donor_token_ids": job.get("donor_token_ids")}
        digest = _fingerprint(definition)
        previous = unique.get(row["pair_id"])
        if previous is not None and previous[2] != digest:
            raise ValueError("same training pair has conflicting donor/recipient definitions")
        unique.setdefault(row["pair_id"], (job, row, digest))
    rank = _rank_argument(rank, len({row.get("source_pair_id", row["pair_id"]) for _, row, _ in unique.values()}))
    contrasts, provenance = [], []
    for job, row, digest in sorted(unique.values(), key=lambda item: (item[1]["layout_id"], item[1]["pair_id"])):
        def tokens(which: str) -> list[int]:
            supplied = job.get(which + "_token_ids")
            values = list(engine.encode(job[which + "_text"])) if supplied is None else supplied
            if not isinstance(values, list) or not values or any(type(i) is not int or i < 0 for i in values):
                raise ValueError("token IDs must be nonempty nonnegative integer lists")
            return values
        recipient_ids, donor_ids = tokens("recipient"), tokens("donor")
        if not recipient_ids or not donor_ids:
            raise ValueError("donor and recipient prompts must encode to nonempty sequences")
        selection = job.get("selection", {})
        explicit = "recipient_positions" in job or "donor_positions" in job or selection.get("scope") == "explicit"
        if explicit:
            def chosen_positions(which: str, length: int) -> list[int]:
                name = which + "_positions"
                outer, inner = job.get(name), selection.get(name)
                if outer is not None and inner is not None and list(outer) != list(inner):
                    raise ValueError("top-level and selection positions contradict each other")
                return _positions(outer if outer is not None else inner, length)
            recipient_positions = chosen_positions("recipient", len(recipient_ids))
            donor_positions = chosen_positions("donor", len(donor_ids))
        else:
            if selection.get("scope") != "last_tokens":
                raise ValueError("provide explicit positions or a last_tokens selection")
            count = selection.get("last_k")
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError("last_k must be a positive integer")
            count = min(int(count), len(recipient_ids), len(donor_ids))
            recipient_positions = list(range(len(recipient_ids) - count, len(recipient_ids)))
            donor_positions = list(range(len(donor_ids) - count, len(donor_ids)))
        if len(recipient_positions) != len(donor_positions):
            raise ValueError("donor and recipient position counts must agree")
        recipient = engine.capture(recipient_ids, layers=[int(layer)], positions=recipient_positions, site=site)[int(layer)]
        donor = engine.capture(donor_ids, layers=[int(layer)], positions=donor_positions, site=site)[int(layer)]
        recipient, donor = np.asarray(recipient, dtype=np.float32), np.asarray(donor, dtype=np.float32)
        if recipient.shape != donor.shape or recipient.ndim != 2 or len(recipient) != len(recipient_positions):
            raise ValueError("engine returned incompatible paired activation shapes")
        contrasts.append(donor - recipient)
        provenance.append({**row, "source_experiment_id": job.get("experiment_id"),
                           "source_definition_sha256": digest,
                           "recipient_input_sha256": _fingerprint(recipient_ids), "donor_input_sha256": _fingerprint(donor_ids),
                           "recipient_positions": recipient_positions, "donor_positions": donor_positions})
    result = fit_task_contrast_subspace(contrasts, provenance, rank, layer=int(layer), site=site, center=center)
    result["metadata"]["extraction"] = {
        "input_job_count": len(jobs), "deduplicated_pair_count": len(unique),
        "model": json.loads(json.dumps(engine.metadata, allow_nan=False)),
        "head_mode": getattr(engine, "head_mode", engine.metadata.get("head_mode")),
        "model_loading_performed": False,
    }
    return result
