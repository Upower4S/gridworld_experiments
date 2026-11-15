"""Pure full-write geometry and compiler float32 evidence; no model imports."""
from copy import deepcopy
import unittest

import numpy as np

from src.activation_causality.full_write_audit import (
    prepare_full_write_norm_audit, random_orientation_audit, validate_full_write_norm_audit,
)
from src.activation_causality.subspace_audit import array_fingerprint


def identity():
    return {"lineage_kind": "inline_physical_source_pair", "source_pair_id": "fresh-pair",
            "source_pair_sha256": "a" * 64, "candidate_source_experiment_id": "old-protocol",
            "candidate_source_spec_sha256": "b" * 64, "recipient_role": "recipient", "donor_role": "natural",
            "recipient_token_sha256": "c" * 64, "donor_token_sha256": "d" * 64,
            "selection_sha256": "e" * 64, "span_rule": "within_context_named_variable_toggle",
            "renderer_id": "canonical_v1"}


def prepare(recipient=None, donor=None, **kwargs):
    recipient = np.zeros((2, 4), dtype=np.float32) if recipient is None else recipient
    donor = np.asarray([[3, 4, 0, 0], [0, 0, 0, 2]], dtype=np.float32) if donor is None else donor
    args = dict(layer=0, site="resid_post", recipient_positions=[7, 3], donor_positions=[2, 8],
                dose=1, source_identity=identity())
    args.update(kwargs)
    return prepare_full_write_norm_audit(recipient, donor, **args)


def log(audit, **updates):
    row = {key: deepcopy(audit[key]) for key in ("layer", "site", "positions", "dose", "operation")}
    row.update(rank=None, delta_l2=audit["expected_delta_l2"].copy(),
               recipient_float32_sha256=audit["expected_recipient_float32_sha256"])
    row.update(updates)
    return row


class FullWriteAuditTests(unittest.TestCase):
    def test_original_full_difference_not_subspace_and_dose_once(self):
        audit = prepare(dose=-.5)
        self.assertEqual(audit["reference_delta_l2"], [5, 2])
        self.assertEqual(audit["expected_delta_l2"], [2.5, 1])
        self.assertEqual(audit["positions"], [7, 3])
        self.assertEqual(audit["donor_positions"], [2, 8])
        self.assertNotIn("rank", audit)
        self.assertNotIn("basis", audit)
        self.assertTrue(validate_full_write_norm_audit(audit, log(audit))["passed"])
        self.assertEqual(audit["paired_actual_norm_ratio_bounds"], [.99 / 1.01, 1.01 / .99])

    def test_float32_canonicalization_matches_engine_fingerprint(self):
        r = np.array([.1, .2, .3], dtype=np.float64)
        audit = prepare(r, r + 1, recipient_positions=[2], donor_positions=[4])
        self.assertEqual(audit["expected_recipient_float32_sha256"], array_fingerprint(r.astype(np.float32)[None, :]))
        self.assertEqual(audit["recipient_shape"], [1, 3])

    def test_fixed_relative_bounds_and_exact_zero(self):
        audit = prepare()
        self.assertTrue(validate_full_write_norm_audit(audit, log(audit, delta_l2=[4.95, 2.02]))["passed"])
        self.assertFalse(validate_full_write_norm_audit(audit, log(audit, delta_l2=[4.949, 2]))["passed"])
        zero = prepare(np.ones((2, 4)), np.ones((2, 4)))
        self.assertTrue(validate_full_write_norm_audit(zero, log(zero))["passed"])
        self.assertFalse(validate_full_write_norm_audit(zero, log(zero, delta_l2=[1e-100, 0]))["passed"])

    def test_failed_actual_geometry_hash_and_norms_are_retained(self):
        audit = prepare()
        for changes in ({"rank": 4}, {"positions": [3, 7]}, {"operation": "subspace"},
                        {"recipient_float32_sha256": "f" * 64}, {"dose": .5},
                        {"delta_l2": [float("inf"), 1]}, {"delta_l2": [-1, 2]}, {"delta_l2": [True, 2]}):
            with self.subTest(changes=changes):
                result = validate_full_write_norm_audit(audit, log(audit, **changes))
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"])
                self.assertEqual(result["reference_delta_l2"], [5, 2])
        self.assertFalse(validate_full_write_norm_audit(audit, [log(audit), log(audit)])["passed"])
        self.assertFalse(validate_full_write_norm_audit(audit, [])["passed"])
        missing = log(audit)
        missing.pop("rank")
        self.assertFalse(validate_full_write_norm_audit(audit, missing)["passed"])

    def test_invalid_prepared_policy_and_geometry_rejected(self):
        for changes in ({"relative_tolerance": .02}, {"absolute_tolerance": 1e-9}, {"rank": 1},
                        {"expected_delta_l2": [10, 4]}, {"reference_delta_l2": [1e999, 2]},
                        {"positions": [True, 3]}, {"donor_positions": [1, 1]}):
            audit = prepare()
            audit.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_full_write_norm_audit(audit, [])
        for kwargs in ({"relative_tolerance": .02}, {"dose": True}, {"dose": float("inf")},
                       {"layer": True}, {"recipient_positions": [1, 1]}, {"site": "mlp_out"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare(**kwargs)

    def test_scaled_norm_handles_large_float32_values(self):
        maximum = np.finfo(np.float32).max
        audit = prepare(np.array([[maximum, maximum]], np.float32),
                        np.array([[-maximum, -maximum]], np.float32), recipient_positions=[0], donor_positions=[1])
        self.assertTrue(np.isfinite(audit["expected_delta_l2"]).all())
        for bad in (np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1j]]), np.zeros((0, 2))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare(bad, bad, recipient_positions=[0], donor_positions=[0])

    def random(self, r, d, seed=42, raw=None):
        delta = d - r
        raw = np.random.default_rng(seed).standard_normal(delta.shape).astype(np.float32) if raw is None else raw
        direction = raw.copy()
        direction *= np.linalg.norm(delta, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
        synthetic = r + direction
        audit = random_orientation_audit(r, d, raw, direction, synthetic, seed=seed, source_identity=identity())
        return synthetic, audit

    def test_random_recipe_draw_identity_and_per_token_norms(self):
        r = np.zeros((2, 9), dtype=np.float32)
        d = np.arange(18, dtype=np.float32).reshape(2, 9)
        synthetic, audit = self.random(r, d)
        self.assertTrue(audit["passed"])
        other, doubled = self.random(r, d * 2)
        self.assertEqual(audit["raw_gaussian_float32_sha256"], doubled["raw_gaussian_float32_sha256"])
        np.testing.assert_array_equal(other, synthetic * 2)
        _, alternate = self.random(r, d, seed=43)
        self.assertNotEqual(audit["raw_gaussian_float32_sha256"], alternate["raw_gaussian_float32_sha256"])
        self.assertEqual(audit["relative_sanity_tolerance"], 1e-4)

    def test_zero_and_degenerate_random_rows_no_redraw(self):
        r = np.zeros((2, 3), dtype=np.float32)
        _, zero = self.random(r, r)
        self.assertTrue(zero["passed"])
        self.assertTrue(all(row["realized_delta_l2"] == 0 for row in zero["positions_audit"]))
        _, degenerate = self.random(r, r + 1, raw=r.copy())
        self.assertFalse(degenerate["passed"])
        self.assertTrue(all(row["degenerate_draw"] for row in degenerate["positions_audit"]))

    def test_float32_cancellation_is_reported_not_hidden_by_abs_floor(self):
        r = np.full((1, 16), 1e8, dtype=np.float32)
        d = r.copy()
        d[0, 0] += 8
        _, audit = self.random(r, d)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["positions_audit"][0]["vector_matches"])


if __name__ == "__main__":
    unittest.main()
