"""Prospective checks of subspace write sizes and the captured recipient state.

Preparation uses the ORIGINAL recipient, donor, and task basis for both intended
and projected-norm-matched random writes. It never uses the synthetic donor or
random basis as the target. Validation consumes measurements from the actual
engine write, including its model-dtype rounding, and preserves failed checks.
No model, RNG, file access, or outcome-dependent tolerance selection is used.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

import numpy as np


FORMAT = "subspace_applied_norm_audit_v1"
RELATIVE_TOLERANCE = 0.01
SITES = frozenset({"resid_post", "attn_out", "mlp_out"})


def _array(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if (result.dtype.kind not in "fiu" or result.ndim < 1 or min(result.shape) < 1
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a nonempty finite real numeric array")
    result = result.astype(np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must be representable as finite float32")
    return result


def array_fingerprint(array: Any) -> str:
    """Hash shape plus canonical float32 bytes, retaining exact row order.

    SHA256 input is compact sorted ASCII JSON ``{"dtype":"float32-le",
    "shape":[...]}``, one newline byte, then little-endian float32 values in
    C order. Endianness and array strides do not affect the digest; shape and
    signed zero do. For recipient validation the engine must pass ``before[0]``
    with shape [positions, hidden], after its float32 cast and before the write.
    """
    values = _array(array, "array")
    canonical = np.ascontiguousarray(values, dtype="<f4")
    header = json.dumps({"dtype": "float32-le", "shape": list(canonical.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    scale = np.max(np.abs(values), axis=-1)
    unit = np.divide(values, scale[..., None], out=np.zeros_like(values), where=scale[..., None] != 0)
    return scale * np.sqrt(np.sum(unit * unit, axis=-1))


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positions(values: Any, name: str) -> list[int]:
    if (not isinstance(values, (list, tuple)) or not values
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0 for value in values)):
        raise ValueError(f"{name} must be nonempty nonnegative integer positions")
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique positions in their actual capture order")
    return result


def _layer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("layer must be a nonnegative integer")
    return int(value)


def prepare_projected_norm_audit(
    recipient: Any, donor: Any, basis: Any, layer: int, site: str, positions: list[int],
    dose: float, relative_tolerance: float = RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Prepare JSON expectations before inference, with a fixed relative 1% gate.

    Recipient/donor must have matching [positions, hidden] shapes. A [hidden]
    vector is accepted for one position and canonicalized to [1, hidden]. Basis
    columns must be orthonormal. Projection norms are computed in float64 from
    the supplied ORIGINAL task contrast; expected actual-write norms equal
    ``abs(dose) * norm``. The recipient fingerprint covers its float32 values.

    Exactly zero expected norm requires exactly zero actual norm. No absolute
    tolerance floor is used. Only the prospectively fixed 0.01 relative
    tolerance is accepted; callers cannot relax it after seeing results.
    """
    tolerance = _number(relative_tolerance, "relative_tolerance")
    if tolerance != RELATIVE_TOLERANCE:
        raise ValueError("relative_tolerance is prospectively fixed at 0.01")
    layer = _layer(layer)
    if site not in SITES:
        raise ValueError("unsupported subspace intervention site")
    positions = _positions(positions, "positions")
    dose = _number(dose, "dose")
    original, source, task = (_array(value, name) for value, name in
                              ((recipient, "recipient"), (donor, "donor"), (basis, "basis")))
    if original.shape != source.shape or original.ndim not in (1, 2):
        raise ValueError("recipient/donor must have identical [positions, hidden] or [hidden] shapes")
    if original.ndim == 1:
        original, source = original[None, :], source[None, :]
    count, hidden = original.shape
    if count != len(positions):
        raise ValueError("activation row count must match the ordered positions")
    if task.ndim != 2 or task.shape[0] != hidden or not 1 <= task.shape[1] <= hidden:
        raise ValueError("basis must have shape [hidden, rank] with 1 <= rank <= hidden")
    rank = task.shape[1]
    gram = np.einsum("hr,hs->rs", task, task, optimize=False)
    orthonormality_error = float(np.max(np.abs(gram - np.eye(rank))))
    if orthonormality_error > 1e-5:
        raise ValueError("task basis columns must be orthonormal")
    coefficients = np.einsum("ph,hr->pr", source - original, task, optimize=False)
    projected = np.einsum("pr,hr->ph", coefficients, task, optimize=False)
    target = _norm(projected)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = abs(dose) * target
        upper_bounds = expected * (1 + tolerance)
    if not np.isfinite(expected).all() or not np.isfinite(upper_bounds).all():
        raise ValueError("dose-scaled expected norms must be finite")
    return {
        "format": FORMAT, "layer": layer, "site": site, "positions": positions,
        "operation": "subspace", "dose": dose, "rank": rank, "hidden": hidden,
        "n_positions": count, "recipient_shape": [count, hidden],
        "expected_recipient_float32_sha256": array_fingerprint(original),
        "original_donor_float32_sha256": array_fingerprint(source),
        "original_task_basis_float32_sha256": array_fingerprint(task),
        "recipient_float32_l2": _norm(original.astype(np.float32)).tolist(),
        "expected_task_projected_l2": target.tolist(), "expected_delta_l2": expected.tolist(),
        "calculation_dtype": "float64", "basis_orthonormality_max_abs_error": orthonormality_error,
        "relative_tolerance": tolerance, "absolute_tolerance": 0.0,
        "zero_expected_policy": "actual_delta_l2_must_equal_exactly_zero",
        "tolerance_fixed_before_execution": True,
        "paired_actual_norm_ratio_bounds": [(1 - tolerance) / (1 + tolerance), (1 + tolerance) / (1 - tolerance)],
        "paired_ratio_interpretation": "If intended and matched-random writes both pass against the same nonzero expectation, their actual norm ratio lies in [0.99/1.01, 1.01/0.99], not a pairwise 1% band. Zero/zero ratios are undefined.",
        "target_definition": "L2 of the original donor-minus-recipient contrast projected through the original task basis, then multiplied by abs(dose) exactly once",
        "required_measurement": "Engine delta_l2 after the actual model-dtype write and recipient_float32_sha256 from before[0] before that write",
        "semantic_headline_eligible": False,
    }


def _prepared(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, dict) or audit.get("format") != FORMAT:
        raise ValueError("audit must be a prepared subspace applied-norm audit")
    try:
        result = json.loads(json.dumps(audit, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared audit must contain finite JSON values") from exc
    for field in ("expected_recipient_float32_sha256", "original_donor_float32_sha256", "original_task_basis_float32_sha256"):
        if not isinstance(result.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", result[field]):
            raise ValueError(f"prepared audit has invalid {field}")
    _layer(result.get("layer"))
    if result.get("site") not in SITES or result.get("operation") != "subspace":
        raise ValueError("prepared audit has invalid site/operation")
    positions = _positions(result.get("positions"), "prepared positions")
    count = len(positions)
    hidden, rank = result.get("hidden"), result.get("rank")
    if (type(hidden) is not int or type(rank) is not int or not 1 <= rank <= hidden
            or result.get("recipient_shape") != [count, hidden] or result.get("n_positions") != count
            or type(result.get("n_positions")) is not int
            or any(type(value) is not int for value in result["recipient_shape"])):
        raise ValueError("prepared audit has inconsistent geometry")
    dose = _number(result.get("dose"), "prepared dose")
    if (result.get("relative_tolerance") != RELATIVE_TOLERANCE or result.get("absolute_tolerance") != 0
            or result.get("tolerance_fixed_before_execution") is not True
            or result.get("zero_expected_policy") != "actual_delta_l2_must_equal_exactly_zero"):
        raise ValueError("prepared audit must retain the fixed relative-only 1% and exact-zero policy")
    for field in ("recipient_float32_l2", "expected_task_projected_l2", "expected_delta_l2"):
        values = result.get(field)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"prepared {field} count differs from positions")
        if any(_number(value, field) < 0 for value in values):
            raise ValueError(f"prepared {field} must be nonnegative")
    if result["expected_delta_l2"] != [abs(dose) * value for value in result["expected_task_projected_l2"]]:
        raise ValueError("prepared expected norms do not apply abs(dose) exactly once")
    expected_bounds = [(1 - RELATIVE_TOLERANCE) / (1 + RELATIVE_TOLERANCE),
                       (1 + RELATIVE_TOLERANCE) / (1 - RELATIVE_TOLERANCE)]
    if result.get("paired_actual_norm_ratio_bounds") != expected_bounds:
        raise ValueError("prepared paired-ratio bounds differ from the fixed policy")
    return result


def validate_applied_norm_audit(audit: dict[str, Any], patch_log: Any) -> dict[str, Any]:
    """Return an enriched copy, retaining norm/hash/geometry mismatches as data.

    ``patch_log`` may be an engine's list of entries or one entry. Select by
    layer/site and exact ordered positions; unrelated entries are ignored. A
    single same-site entry with different positions is retained as a failed
    geometry check. Missing/ambiguous logs, missing fingerprints, and malformed
    actual measurements produce ``passed=False`` with explicit errors, so a
    runner can persist diagnostics before raising. Invalid calling types or an
    invalid prepared audit raise ValueError. Inputs are never mutated.
    """
    result = _prepared(audit)
    if isinstance(patch_log, dict):
        entries = [patch_log]
    elif isinstance(patch_log, (list, tuple)):
        entries = list(patch_log)
    else:
        raise ValueError("patch_log must be an entry or a list of entries")
    errors = []
    if any(not isinstance(entry, dict) for entry in entries):
        errors.append("patch_log_contains_non_object_entries")
    candidates = [entry for entry in entries if isinstance(entry, dict)
                  and entry.get("layer") == result["layer"] and entry.get("site") == result["site"]]
    exact = [entry for entry in candidates if entry.get("positions") == result["positions"]]
    selected = exact[0] if len(exact) == 1 else (candidates[0] if not exact and len(candidates) == 1 else None)
    if len(exact) > 1 or (not exact and len(candidates) > 1):
        errors.append("ambiguous_patch_log_for_expected_geometry")
    elif selected is None:
        errors.append("missing_patch_log_for_expected_layer_site")
    actual_hash = selected.get("recipient_float32_sha256") if selected else None
    valid_hash = isinstance(actual_hash, str) and re.fullmatch(r"[0-9a-f]{64}", actual_hash) is not None
    hash_matches = valid_hash and actual_hash == result["expected_recipient_float32_sha256"]
    if not valid_hash:
        errors.append("missing_or_malformed_actual_recipient_float32_sha256")
    elif not hash_matches:
        errors.append("recipient_float32_sha256_mismatch")
    geometry = {}
    for field in ("layer", "site", "positions", "operation", "dose", "rank"):
        actual = selected.get(field) if selected else None
        matches = actual == result[field] and type(actual) is type(result[field])
        # JSON emitters may serialize an integer dose as 1 instead of 1.0.
        if field == "dose":
            try:
                matches = type(actual) in (int, float) and _number(actual, "actual dose") == result[field]
            except ValueError:
                matches = False
        if field == "positions":
            try:
                matches = type(actual) is list and _positions(actual, "actual positions") == result[field]
            except ValueError:
                matches = False
        geometry[field] = bool(matches)
        if not matches:
            errors.append(f"actual_{field}_mismatch")
    count = result["n_positions"]
    actual_norms, actual_before = [], []
    for field, destination in (("delta_l2", actual_norms), ("before_l2", actual_before)):
        values = selected.get(field) if selected else None
        if not isinstance(values, list) or len(values) != count:
            errors.append(f"actual_{field}_missing_or_wrong_count")
            destination.extend([None] * count)
            continue
        for index, value in enumerate(values):
            try:
                valid = type(value) in (int, float) and _number(value, field) >= 0
            except ValueError:
                valid = False
            if not valid:
                errors.append(f"actual_{field}_invalid_at_index_{index}")
                destination.append(None)
            else:
                destination.append(float(value))
    rows = []
    for index, (position, expected, actual) in enumerate(zip(result["positions"], result["expected_delta_l2"], actual_norms, strict=True)):
        lower, upper = expected * (1 - RELATIVE_TOLERANCE), expected * (1 + RELATIVE_TOLERANCE)
        norm_pass = actual is not None and (actual == 0 if expected == 0 else lower <= actual <= upper)
        if not norm_pass:
            errors.append(f"applied_delta_norm_mismatch_at_index_{index}")
        relative_error = abs(actual - expected) / expected if actual is not None and expected > 0 else (0.0 if norm_pass else None)
        relative_error_overflow = relative_error is not None and not math.isfinite(relative_error)
        if relative_error_overflow:
            relative_error = None
        rows.append({"position": position, "expected_delta_l2": expected, "actual_delta_l2": actual,
                     "lower_bound": lower, "upper_bound": upper, "zero_expected": expected == 0,
                     "absolute_error": abs(actual - expected) if actual is not None else None,
                     "relative_error": relative_error, "relative_error_overflow": relative_error_overflow,
                     "norm_passed": bool(norm_pass)})
    validation = {
        "passed": not errors, "errors": errors, "matching_log_count": len(exact),
        "expected_recipient_float32_sha256": result["expected_recipient_float32_sha256"],
        "actual_recipient_float32_sha256": actual_hash if isinstance(actual_hash, str) else None,
        "recipient_hash_matches": bool(hash_matches), "geometry_matches": geometry,
        "actual_delta_l2": actual_norms, "actual_recipient_l2": actual_before,
        "norms_passed": all(row["norm_passed"] for row in rows), "per_position": rows,
        "measurement_scope": "Actual engine post-model-dtype write norms; recipient hash from the pre-write float32 tensor",
    }
    result["applied_validation"] = validation
    result["passed"] = validation["passed"]
    # Return standard JSON even for malformed observed norms: absent/nonfinite
    # values are null, with their positions explicitly identified in errors.
    json.dumps(result, allow_nan=False)
    return deepcopy(result)
