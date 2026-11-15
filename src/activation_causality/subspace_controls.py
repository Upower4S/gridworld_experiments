"""Pure projected-norm-matched random-subspace controls.

This control is distinct from the existing rank-only random-basis control. It
tests whether direction matters after matching each position's projected update
size; neither control independently establishes an abstract belief mechanism.
No model, files, global RNG, donor selection, or dose is used here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


FORMAT = "projected_norm_matched_random_subspace_v1"
ORTHONORMAL_ATOL = 1e-5
FLOAT32_MATCH_RTOL = 1e-4


def _finite_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    array = array.astype(np.float64)
    if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must be representable as finite float32 values")
    return array


def _norm(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64)
    scale = np.max(np.abs(values), axis=-1)
    normalized = np.divide(values, scale[..., None], out=np.zeros_like(values), where=scale[..., None] != 0)
    return scale * np.sqrt(np.sum(normalized * normalized, axis=-1))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> list[float | None]:
    return [float(abs(a - b) / b) if b > 0 else (0.0 if a == 0 else None)
            for a, b in zip(actual, expected, strict=True)]


def _project64(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    # Explicit float64 sums avoid platform BLAS accumulation differences and
    # spurious floating-point status flags on some Accelerate builds.
    coefficients = np.einsum("ph,hr->pr", values, basis, optimize=False)
    return np.einsum("pr,hr->ph", coefficients, basis, optimize=False)


def _gram_error(basis: np.ndarray) -> float:
    basis64 = basis.astype(np.float64)
    gram = np.einsum("hr,hs->rs", basis64, basis64, optimize=False)
    return float(np.max(np.abs(gram - np.eye(basis.shape[1]))))


def projected_norm_matched_random_control(
    recipient: Any, donor: Any, task_basis: Any, seed: int,
) -> dict[str, Any]:
    """Return ``{'donor': float32 array, 'basis': float32 Q, 'audit': dict}``.

    Recipient and donor must have the same nonempty ``[positions, hidden]``
    shape. Single ``[hidden]`` vectors are also accepted and returned with that
    shape. The supplied task basis has orthonormal columns and shape
    ``[hidden, rank]``. No basis is refitted or silently orthogonalized.

    The random basis uses precisely the current compiler's draw: one
    ``default_rng(seed).standard_normal(task_basis.shape)`` followed by NumPy
    QR, without sign canonicalization or resampling. For each position, project
    the same donor-minus-recipient contrast into both subspaces and scale its
    random projection to the task-projected L2 norm. A zero task projection
    returns the recipient exactly after float32 conversion. A nonzero target
    with a numerically degenerate random projection raises an explicit error.

    Pass the synthetic donor and Q through the ordinary ``subspace`` engine
    operation. The engine applies the requested dose exactly once. The captured
    recipient must match the activation present when this patch is applied;
    preceding interventions may invalidate this assumption.

    Matching is computed in float64. Audit fields also recompute the downstream
    projection in NumPy float32 after donor/Q casting, including cancellation
    and norm errors. These diagnostics do not measure MLX/BF16 arithmetic or
    claim that float32 matching always succeeds. Integration must retain these
    diagnostics and compare the dose-scaled target with the engine's actual
    post-cast ``patch_log.delta_l2``; the helper cannot certify that model write.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    seed = int(seed)
    original = _finite_array(recipient, "recipient")
    source = _finite_array(donor, "donor")
    task = _finite_array(task_basis, "task_basis")
    if original.ndim not in (1, 2) or original.shape != source.shape or min(original.shape, default=0) < 1:
        raise ValueError("recipient/donor must have the same nonempty [positions, hidden] or [hidden] shape")
    vector_input = original.ndim == 1
    original = original[None, :] if vector_input else original
    source = source[None, :] if vector_input else source
    positions, hidden = original.shape
    if task.ndim != 2 or task.shape[0] != hidden or not 1 <= task.shape[1] <= hidden:
        raise ValueError("task_basis must have shape [hidden, rank] with 1 <= rank <= hidden")
    rank = task.shape[1]
    task_gram_error = _gram_error(task)
    if task_gram_error > ORTHONORMAL_ATOL:
        raise ValueError("task_basis columns must be orthonormal; no automatic repair is performed")

    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal(task.shape)
    random_basis, triangular = np.linalg.qr(gaussian)
    relative_degeneracy_threshold = float(64 * np.finfo(np.float64).eps * max(hidden, rank))
    if np.any(np.abs(np.diag(triangular)) <= relative_degeneracy_threshold * np.linalg.norm(gaussian)):
        raise ValueError("single seeded Gaussian basis draw is numerically degenerate; no resampling")
    contrast = source - original
    task_projection = _project64(contrast, task)
    random_projection = _project64(contrast, random_basis)
    contrast_norm = _norm(contrast)
    target_norm = _norm(task_projection)
    random_norm = _norm(random_projection)
    zero_target = target_norm == 0
    threshold = relative_degeneracy_threshold * contrast_norm
    degenerate = (~zero_target) & (random_norm <= threshold)
    if np.any(degenerate):
        indices = np.flatnonzero(degenerate).tolist()
        raise ValueError(f"nonzero task projection but degenerate random projection at positions {indices}; no resampling")
    scales = np.zeros(positions, dtype=np.float64)
    scales[~zero_target] = target_norm[~zero_target] / random_norm[~zero_target]
    scaled = random_projection * scales[:, None]
    # Explicit assignment preserves exact zero identities, including cases
    # where the original contrast has a nonzero random-subspace projection.
    scaled[zero_target] = 0
    synthetic = original + scaled
    if not np.isfinite(synthetic).all() or np.any(np.abs(synthetic) > np.finfo(np.float32).max):
        raise ValueError("scaled synthetic donor is not representable as finite float32 values")
    scaled_norm = _norm(scaled)
    cosine = [None if zero else float(np.clip(np.dot(a / an, b / bn), -1, 1))
              for a, b, an, bn, zero in zip(task_projection, scaled, target_norm, scaled_norm, zero_target, strict=True)]

    recipient32 = original.astype(np.float32)
    donor32 = source.astype(np.float32)
    synthetic32 = np.ascontiguousarray(synthetic, dtype=np.float32)
    random32 = np.ascontiguousarray(random_basis, dtype=np.float32)
    task32 = task.astype(np.float32)
    # Keep the subtraction and both matmuls float32, as in the engine's
    # subspace formula. This is a CPU prediction, not an MLX/BF16 measurement.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        control32 = ((synthetic32 - recipient32) @ random32) @ random32.T
        target32 = ((donor32 - recipient32) @ task32) @ task32.T
        repeated32 = (control32 @ random32) @ random32.T
    if not all(np.isfinite(a).all() for a in (control32, target32, repeated32)):
        raise ValueError("float32 downstream subspace arithmetic is nonfinite")
    control_norm32 = _norm(control32)
    target_norm32 = _norm(target32)
    matches = np.abs(control_norm32 - target_norm) <= FLOAT32_MATCH_RTOL * target_norm
    random_gram_error32 = _gram_error(random32)
    if random_gram_error32 > ORTHONORMAL_ATOL:
        raise ValueError("random basis is not orthonormal after float32 conversion")
    audit = {
        "format": FORMAT, "seed": seed, "rank": rank, "hidden": hidden, "n_positions": positions,
        "input_shape": list(np.shape(recipient)), "matching_unit": "individual_token_position",
        "random_basis_recipe": "numpy.default_rng(seed).standard_normal(task_basis.shape); numpy.linalg.qr; no sign changes",
        "random_draws": 1, "resampled": False, "dose_applied": False,
        "control_kind": "rank_and_task_projected_norm_matched_random_subspace",
        "task_basis_orthonormality_max_abs_error": task_gram_error,
        "orthonormality_absolute_tolerance": ORTHONORMAL_ATOL,
        "random_projection_relative_degeneracy_threshold": relative_degeneracy_threshold,
        "random_projection_absolute_degeneracy_threshold": threshold.tolist(),
        "contrast_l2": contrast_norm.tolist(), "target_projected_l2": target_norm.tolist(),
        "unscaled_random_projected_l2": random_norm.tolist(), "scaled_random_projected_l2": scaled_norm.tolist(),
        "scale_factors": scales.tolist(), "task_random_cosine": cosine, "zero_target_identity": zero_target.tolist(),
        "float32": {
            "random_basis_orthonormality_max_abs_error": random_gram_error32,
            "synthetic_donor_minus_recipient_l2": _norm(synthetic32.astype(np.float64) - recipient32.astype(np.float64)).tolist(),
            "reprojected_control_l2": control_norm32.tolist(), "task_projected_l2": target_norm32.tolist(),
            "reprojected_control_absolute_error_vs_float64_target": np.abs(control_norm32 - target_norm).tolist(),
            "reprojected_control_relative_error_vs_float64_target": _relative_error(control_norm32, target_norm),
            "reprojected_control_relative_error_vs_float32_task": _relative_error(control_norm32, target_norm32),
            "reprojection_idempotence_error_l2": _norm(repeated32.astype(np.float64) - control32.astype(np.float64)).tolist(),
            "norm_match_relative_tolerance": FLOAT32_MATCH_RTOL,
            "norm_matches_float64_target_within_tolerance": matches.tolist(),
            "all_norms_match_within_tolerance": bool(np.all(matches)),
        },
        "engine_applied_norms_measured": False,
        "engine_audit_requirement": "Compare abs(dose) * target_projected_l2 with engine patch_log.delta_l2 after actual model-dtype/BF16 writes; retain float32 errors too.",
        "claim_scope": "direction-versus-size control; not independent semantic-belief evidence",
        "semantic_headline_eligible": False,
    }
    return {"donor": synthetic32[0].copy() if vector_input else synthetic32,
            "basis": random32, "audit": audit}
