"""Prospective full-replacement norm evidence, separate from subspace audits.

The reference is the ORIGINAL recipient and named physical donor, before any
synthetic random donor. Float32 compiler direction evidence is distinct from
actual model-dtype write norms. Neither certifies a semantic belief mechanism.
This module performs no model, file, or network access.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from .subspace_audit import array_fingerprint

FORMAT = "full_write_norm_audit_v1"
RANDOM_FORMAT = "full_write_random_orientation_v1"
RELATIVE_TOLERANCE = 0.01
FLOAT32_RELATIVE_TOLERANCE = 1e-4
RANDOM_RECIPE = "numpy_default_rng_standard_normal_float32_legacy_row_norm_v1"
ZERO_POLICY = "actual_delta_l2_must_equal_exactly_zero"
SOURCE_HASH_FIELDS = ("candidate_source_spec_sha256", "source_pair_sha256", "recipient_token_sha256", "donor_token_sha256", "selection_sha256")


class FullWritePreparationError(ValueError):
    """Compiler failure carrying finite JSON evidence before any patched prefill."""

    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.full_write_preparation_failure = _json(audit)


def _json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("audit must contain finite JSON values") from exc


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positions(value: Any) -> list[int]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(v) is not int or v < 0 for v in value) or len(set(value)) != len(value)):
        raise ValueError("positions must be ordered unique nonnegative integers")
    return list(value)


def _arrays(recipient: Any, donor: Any) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for value in (recipient, donor):
        array = np.asarray(value)
        if (array.dtype.kind not in "fiu" or array.ndim not in (1, 2) or min(array.shape) < 1
                or not np.isfinite(array).all() or np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max)):
            raise ValueError("captures must be nonempty finite real float32-representable arrays")
        array = np.asarray(array, dtype=np.float32)
        arrays.append(array[None, :] if array.ndim == 1 else array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("recipient/reference donor shapes must match")
    return arrays[0], arrays[1]


def _norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    scale = np.max(np.abs(values), axis=-1)
    unit = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] != 0)
    return scale * np.sqrt(np.sum(unit * unit, axis=-1))


def validate_source_identity(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("full_write_source_identity must be an object")
    result = _json(value)
    for name in SOURCE_HASH_FIELDS:
        if not isinstance(result.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", result[name]) is None:
            raise ValueError(f"full-write source identity requires {name}")
    for name in ("candidate_source_experiment_id", "source_pair_id", "renderer_id"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"full-write source identity requires {name}")
    if (result.get("lineage_kind") != "inline_physical_source_pair"
            or result.get("span_rule") != "within_context_named_variable_toggle"):
        raise ValueError("full-write source lineage/span rule is invalid")
    if result.get("recipient_role") != "recipient" or result.get("donor_role") not in ("natural", "matched", "recipient"):
        raise ValueError("full-write source identity has unsupported physical source roles")
    return result


def prepare_full_write_norm_audit(
    recipient: Any, reference_donor: Any, *, layer: int, site: str,
    recipient_positions: list[int], donor_positions: list[int], dose: float,
    source_identity: dict, relative_tolerance: float = RELATIVE_TOLERANCE,
) -> dict:
    """Fix original full-contrast expectations; dose is applied exactly once.

    The compiler restricts the registered contextual route to resid_post replace
    dose one. This pure helper accepts finite doses to test the arithmetic.
    Exactly zero expectations have no absolute tolerance floor.
    """
    if type(layer) is not int or layer < 0 or site != "resid_post":
        raise ValueError("full-write audit requires a nonnegative layer and resid_post")
    if _number(relative_tolerance, "relative_tolerance") != RELATIVE_TOLERANCE:
        raise ValueError("full-write relative tolerance is prospectively fixed at 0.01")
    dose = _number(dose, "dose")
    rpos, dpos = _positions(recipient_positions), _positions(donor_positions)
    original, reference = _arrays(recipient, reference_donor)
    if original.shape[0] != len(rpos) or len(rpos) != len(dpos):
        raise ValueError("ordered capture positions must match activation rows")
    source = validate_source_identity(source_identity)
    reference_norm = _norm(reference.astype(np.float64) - original.astype(np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        expected = abs(dose) * reference_norm
        upper = (1 + RELATIVE_TOLERANCE) * expected
    if not np.isfinite(expected).all() or not np.isfinite(upper).all():
        raise ValueError("expected norms and fixed tolerance bounds must be finite")
    return {
        "format": FORMAT, "layer": layer, "site": site, "operation": "replace", "dose": dose,
        "positions": rpos, "donor_positions": dpos, "hidden": original.shape[1],
        "n_positions": len(rpos), "recipient_shape": list(original.shape),
        "capture_dtype": "float32", "calculation_dtype": "float64",
        "expected_recipient_float32_sha256": array_fingerprint(original),
        "original_reference_donor_float32_sha256": array_fingerprint(reference),
        "recipient_float32_l2": _norm(original).tolist(),
        "reference_delta_l2": reference_norm.tolist(), "expected_delta_l2": expected.tolist(),
        "source_identity": source, "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": 0.0, "zero_expected_policy": ZERO_POLICY,
        "tolerance_fixed_before_execution": True,
        "paired_actual_norm_ratio_bounds": [0.99 / 1.01, 1.01 / 0.99],
        "paired_ratio_interpretation": "Bounds apply only when both writes pass against the same nonzero original reference; zero/zero is undefined.",
        "reference_donor_hash_evidence": "compiler capture only; the engine does not measure the donor hash",
        "required_measurement": "actual model-dtype delta_l2 and pre-write recipient_float32_sha256",
        "direction_claim": "norm validation does not certify model-dtype direction preservation",
        "semantic_headline_eligible": False,
    }


def _prepared(audit: Any) -> dict:
    if not isinstance(audit, dict) or audit.get("format") != FORMAT:
        raise ValueError("expected a prepared full-write audit")
    result = _json(audit)
    for forbidden in ("basis", "rank", "projector", "expected_task_projected_l2"):
        if forbidden in result:
            raise ValueError("full-write audit cannot contain subspace geometry")
    if (type(result.get("layer")) is not int or result["layer"] < 0
            or result.get("site") != "resid_post" or result.get("operation") != "replace"):
        raise ValueError("prepared full-write geometry is invalid")
    rpos, dpos = _positions(result.get("positions")), _positions(result.get("donor_positions"))
    count, hidden = result.get("n_positions"), result.get("hidden")
    if (type(count) is not int or type(hidden) is not int or hidden < 1
            or count != len(rpos) or count != len(dpos) or result.get("recipient_shape") != [count, hidden]
            or any(type(v) is not int for v in result["recipient_shape"])):
        raise ValueError("prepared full-write capture geometry is inconsistent")
    validate_source_identity(result.get("source_identity"))
    for name in ("expected_recipient_float32_sha256", "original_reference_donor_float32_sha256"):
        if not isinstance(result.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", result[name]) is None:
            raise ValueError(f"prepared full-write audit has invalid {name}")
    if (result.get("relative_tolerance") != RELATIVE_TOLERANCE or result.get("absolute_tolerance") != 0
            or result.get("zero_expected_policy") != ZERO_POLICY or result.get("tolerance_fixed_before_execution") is not True
            or result.get("paired_actual_norm_ratio_bounds") != [0.99 / 1.01, 1.01 / 0.99]):
        raise ValueError("prepared audit must retain fixed 1% relative-only and exact-zero policy")
    dose = _number(result.get("dose"), "prepared dose")
    for name in ("reference_delta_l2", "expected_delta_l2", "recipient_float32_l2"):
        values = result.get(name)
        if not isinstance(values, list) or len(values) != count or any(_number(v, name) < 0 for v in values):
            raise ValueError("prepared norm arrays must be finite, nonnegative and position-aligned")
    if result["expected_delta_l2"] != [abs(dose) * v for v in result["reference_delta_l2"]]:
        raise ValueError("expected norms must apply abs(dose) exactly once")
    if any(not math.isfinite(v * 1.01) for v in result["expected_delta_l2"]):
        raise ValueError("prepared tolerance bounds must remain finite")
    return result


def validate_full_write_norm_audit(prepared: dict, patch_log: Any) -> dict:
    """Return retained mismatch evidence, never promoting invalid writes.

    Malformed calling inputs raise ValueError. Missing/malformed actual engine
    evidence returns passed=False so the runner can persist it before raising.
    """
    result = _prepared(prepared)
    if isinstance(patch_log, dict):
        entries = [patch_log]
    elif isinstance(patch_log, (list, tuple)):
        entries = list(patch_log)
    else:
        raise ValueError("patch_log must be an entry or list of entries")
    errors = []
    if any(not isinstance(entry, dict) for entry in entries):
        errors.append("patch_log_contains_non_object_entries")
    candidates = [entry for entry in entries if isinstance(entry, dict)
                  and entry.get("layer") == result["layer"] and entry.get("site") == result["site"]]
    exact = [entry for entry in candidates if entry.get("positions") == result["positions"]]
    selected = exact[0] if len(exact) == 1 else (candidates[0] if not exact and len(candidates) == 1 else None)
    if selected is None:
        errors.append("missing_or_ambiguous_patch_log")
    geometry = {}
    for name, expected in {**{k: result[k] for k in ("layer", "site", "positions", "operation", "dose")}, "rank": None}.items():
        actual = selected.get(name) if selected else None
        matches = selected is not None and name in selected and type(actual) is type(expected) and actual == expected
        if name == "dose":
            matches = type(actual) in (int, float) and actual == expected
        if name == "positions":
            matches = isinstance(actual, list) and all(type(v) is int for v in actual) and actual == expected
        geometry[name] = matches
        if not matches:
            errors.append(f"actual_{name}_mismatch")
    actual_hash = selected.get("recipient_float32_sha256") if selected else None
    hash_matches = actual_hash == result["expected_recipient_float32_sha256"]
    if not hash_matches:
        errors.append("actual_recipient_hash_missing_or_mismatched")
    values = selected.get("delta_l2") if selected else None
    if not isinstance(values, list) or len(values) != result["n_positions"]:
        errors.append("actual_delta_l2_missing_or_wrong_count")
        values = [None] * result["n_positions"]
    rows = []
    for position, expected, actual in zip(result["positions"], result["expected_delta_l2"], values, strict=True):
        try:
            valid = _number(actual, "actual norm") >= 0
        except ValueError:
            valid = False
        actual = float(actual) if valid else None
        passed = valid and (actual == 0 if expected == 0 else .99 * expected <= actual <= 1.01 * expected)
        if not passed:
            errors.append(f"actual_delta_l2_failed_at_position_{position}")
        relative_error = abs(actual / expected - 1) if expected and actual is not None else None
        relative_overflow = relative_error is not None and not math.isfinite(relative_error)
        rows.append({"position": position, "expected_delta_l2": expected, "actual_delta_l2": actual,
                     "relative_error": None if relative_overflow else relative_error,
                     "relative_error_overflow": relative_overflow,
                     "lower_bound": .99 * expected, "upper_bound": 1.01 * expected, "passed": bool(passed)})
    result.update(actual_recipient_float32_sha256=actual_hash if isinstance(actual_hash, str) else None,
                  recipient_hash_matches=hash_matches, geometry_matches=geometry, positions_audit=rows,
                  actual_delta_l2=[row["actual_delta_l2"] for row in rows], errors=errors, passed=not errors)
    return result


def random_orientation_audit(recipient: Any, reference_donor: Any, raw_draw: Any,
                             scaled_direction: Any, synthetic_donor: Any, *, seed: int,
                             source_identity: dict) -> dict:
    """Inspect the unchanged legacy Gaussian recipe; do not draw or rescale.

    Call immediately after the compiler's original NumPy operations. Recording
    the old matched control adds no new norm gate. Audited new controls require
    this record's fixed float32 sanity checks before patched prefill.
    """
    original, reference = _arrays(recipient, reference_donor)
    raw, scaled = _arrays(raw_draw, scaled_direction)
    synthetic, _ = _arrays(synthetic_donor, synthetic_donor)
    if raw.shape != original.shape or synthetic.shape != original.shape or type(seed) is not int or seed < 0:
        raise ValueError("random orientation geometry/seed is invalid")
    source = validate_source_identity(source_identity)
    target = _norm(reference.astype(np.float64) - original.astype(np.float64))
    raw_norm = _norm(raw)
    intended = scaled.astype(np.float64)
    realized = synthetic.astype(np.float64) - original.astype(np.float64)
    intended_norm, actual = _norm(intended), _norm(realized)
    # These scale factors record the exact legacy float32 norm/division recipe.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        legacy_scale = np.linalg.norm(reference - original, axis=-1) / np.maximum(np.linalg.norm(raw, axis=-1), 1e-12)
    errors, rows = [], []
    for index, (t, r, a, intended_n) in enumerate(zip(target, raw_norm, actual, intended_norm, strict=True)):
        degenerate = bool(t > 0 and r <= 1e-12)
        norm_ok = bool(a == 0 if t == 0 else abs(a / t - 1) <= FLOAT32_RELATIVE_TOLERANCE)
        vector_error = float(_norm((realized[index] - intended[index])[None, :])[0])
        vector_ok = bool(vector_error == 0 if intended_n == 0 else vector_error / intended_n <= FLOAT32_RELATIVE_TOLERANCE)
        cosine = float(np.sum((realized[index] / a) * (intended[index] / intended_n))) if a and intended_n else None
        if degenerate or not norm_ok or not vector_ok or not math.isfinite(float(legacy_scale[index])):
            errors.append(f"float32_random_sanity_failed_at_index_{index}")
        rows.append({"reference_delta_l2": float(t), "raw_gaussian_l2": float(r),
                     "scale_factor": float(legacy_scale[index]) if math.isfinite(float(legacy_scale[index])) else None,
                     "intended_direction_l2": float(intended_n), "realized_delta_l2": float(a),
                     "realized_vs_intended_cosine": cosine, "realized_vs_intended_l2_error": vector_error,
                     "relative_norm_error": float(abs(a / t - 1)) if t else None,
                     "norm_matches": norm_ok, "vector_matches": vector_ok, "degenerate_draw": degenerate})
    return {"format": RANDOM_FORMAT, "recipe": RANDOM_RECIPE, "seed": seed, "shape": list(raw.shape),
            "rng": "numpy.random.default_rng/PCG64", "draw_dtype": "float32", "numpy_version": np.__version__,
            "source_identity": source, "raw_gaussian_float32_sha256": array_fingerprint(raw),
            "reference_donor_float32_sha256": array_fingerprint(reference),
            "recipient_float32_sha256": array_fingerprint(original),
            "scaled_direction_float32_sha256": array_fingerprint(scaled),
            "compiled_donor_float32_sha256": array_fingerprint(synthetic),
            "relative_sanity_tolerance": FLOAT32_RELATIVE_TOLERANCE, "absolute_sanity_tolerance": 0.0,
            "zero_reference_policy": "exact_zero_realized_delta", "positions_audit": rows,
            "errors": errors, "passed": not errors,
            "evidence_scope": "compiler float32 orientation; actual model-dtype direction is not certified"}
