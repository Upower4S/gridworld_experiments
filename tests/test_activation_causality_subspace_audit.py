"""Model-free tests of prospective write-norm and recipient-identity checks."""

from copy import deepcopy
import hashlib
import json
import unittest

import numpy as np

from src.activation_causality.subspace_audit import (
    array_fingerprint, prepare_projected_norm_audit, validate_applied_norm_audit,
)
from src.activation_causality.subspace_controls import projected_norm_matched_random_control


class SubspaceAppliedAuditTests(unittest.TestCase):
    def setUp(self):
        self.recipient = np.array([[1, 2, 3, 4], [-2, 1, 0, 2]], dtype=np.float32)
        self.donor = self.recipient + np.array([[3, 4, 20, 0], [0, 2, 0, 10]], dtype=np.float32)
        self.basis = np.eye(4, dtype=np.float32)[:, :2]
        self.audit = self.prepare()

    def prepare(self, **changes):
        values = {"recipient": self.recipient, "donor": self.donor, "basis": self.basis,
                  "layer": 10, "site": "resid_post", "positions": [19, 7], "dose": 0.5}
        return prepare_projected_norm_audit(**{**values, **changes})

    def log(self, **changes):
        values = {"layer": 10, "site": "resid_post", "positions": [19, 7], "dose": 0.5,
                  "rank": 2, "operation": "subspace", "delta_l2": [2.5, 1],
                  "before_l2": np.linalg.norm(self.recipient, axis=-1).tolist(),
                  "recipient_float32_sha256": array_fingerprint(self.recipient)}
        return {**values, **changes}

    def test_exact_task_projection_geometry_and_single_dose(self):
        self.assertEqual(self.audit["positions"], [19, 7])
        self.assertEqual(self.audit["expected_task_projected_l2"], [5, 2])
        self.assertEqual(self.audit["expected_delta_l2"], [2.5, 1])
        self.assertEqual(self.audit["recipient_shape"], [2, 4])
        self.assertEqual(self.audit["relative_tolerance"], 0.01)
        self.assertEqual(self.audit["absolute_tolerance"], 0)
        result = validate_applied_norm_audit(self.audit, self.log())
        self.assertTrue(result["passed"])
        self.assertTrue(result["applied_validation"]["recipient_hash_matches"])
        self.assertEqual(result["applied_validation"]["actual_delta_l2"], [2.5, 1])
        self.assertEqual(result["applied_validation"]["errors"], [])
        json.dumps(result, allow_nan=False)

    def test_hash_is_exact_canonical_float32_shape_and_order(self):
        expected_header = b'{"dtype":"float32-le","shape":[2,4]}\n'
        expected = hashlib.sha256(expected_header + self.recipient.astype("<f4").tobytes()).hexdigest()
        self.assertEqual(array_fingerprint(self.recipient), expected)
        self.assertEqual(array_fingerprint(self.recipient.astype(">f4")), expected)
        self.assertEqual(array_fingerprint(np.asfortranarray(self.recipient)), expected)
        self.assertNotEqual(array_fingerprint(self.recipient.reshape(4, 2)), expected)
        self.assertNotEqual(array_fingerprint(self.recipient[::-1]), expected)
        self.assertNotEqual(array_fingerprint(np.array([[0]], dtype=np.float32)),
                            array_fingerprint(np.array([[-0.0]], dtype=np.float32)))

    def test_exact_inclusive_one_percent_limits_and_pairwise_ratio_are_distinct(self):
        lower = [value * 0.99 for value in self.audit["expected_delta_l2"]]
        upper = [value * 1.01 for value in self.audit["expected_delta_l2"]]
        self.assertTrue(validate_applied_norm_audit(self.audit, self.log(delta_l2=lower))["passed"])
        self.assertTrue(validate_applied_norm_audit(self.audit, self.log(delta_l2=upper))["passed"])
        bounds = self.audit["paired_actual_norm_ratio_bounds"]
        self.assertEqual(bounds, [0.99 / 1.01, 1.01 / 0.99])
        self.assertGreater(bounds[1], 1.02)
        outside = [np.nextafter(upper[0], np.inf).item(), upper[1]]
        result = validate_applied_norm_audit(self.audit, self.log(delta_l2=outside))
        self.assertFalse(result["passed"])
        self.assertFalse(result["applied_validation"]["per_position"][0]["norm_passed"])
        self.assertTrue(result["applied_validation"]["per_position"][1]["norm_passed"])

    def test_mismatched_actual_norm_is_retained_not_raised_or_replaced(self):
        result = validate_applied_norm_audit(self.audit, self.log(delta_l2=[1.7, 1]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["expected_delta_l2"], [2.5, 1])
        row = result["applied_validation"]["per_position"][0]
        self.assertEqual(row["actual_delta_l2"], 1.7)
        self.assertAlmostEqual(row["relative_error"], 0.32)
        self.assertIn("applied_delta_norm_mismatch_at_index_0", result["applied_validation"]["errors"])
        json.dumps(result, allow_nan=False)

    def test_zero_target_requires_exact_actual_zero_without_absolute_floor(self):
        audit = self.prepare(donor=self.recipient)
        self.assertEqual(audit["expected_delta_l2"], [0, 0])
        self.assertTrue(validate_applied_norm_audit(audit, self.log(delta_l2=[0, 0]))["passed"])
        result = validate_applied_norm_audit(audit, self.log(delta_l2=[0, 1e-100]))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["applied_validation"]["per_position"][1]["relative_error"])
        self.assertEqual(len(result["applied_validation"]["per_position"]), 2)

    def test_negative_and_zero_dose_use_absolute_norm_once_but_keep_signed_geometry(self):
        for dose, expected in ((-0.5, [2.5, 1]), (0, [0, 0])):
            audit = self.prepare(dose=dose)
            self.assertEqual(audit["expected_delta_l2"], expected)
            self.assertTrue(validate_applied_norm_audit(audit, self.log(dose=dose, delta_l2=expected))["passed"])
        audit = self.prepare(dose=-0.5)
        result = validate_applied_norm_audit(audit, self.log())
        self.assertFalse(result["passed"])
        self.assertIn("actual_dose_mismatch", result["applied_validation"]["errors"])

    def test_recipient_hash_mismatch_fails_even_when_norms_match(self):
        other = self.recipient.copy()
        other[0] *= -1
        np.testing.assert_array_equal(np.linalg.norm(other, axis=-1), np.linalg.norm(self.recipient, axis=-1))
        result = validate_applied_norm_audit(self.audit, self.log(recipient_float32_sha256=array_fingerprint(other)))
        check = result["applied_validation"]
        self.assertFalse(result["passed"])
        self.assertTrue(check["norms_passed"])
        self.assertFalse(check["recipient_hash_matches"])
        self.assertNotEqual(check["actual_recipient_float32_sha256"], check["expected_recipient_float32_sha256"])

    def test_missing_log_hash_and_ambiguous_entries_return_failed_audits(self):
        missing_hash = self.log()
        del missing_hash["recipient_float32_sha256"]
        for logs, expected in (([], "missing_patch_log"), (missing_hash, "missing_or_malformed_actual_recipient"),
                               ([self.log(), self.log()], "ambiguous_patch_log"), ([None], "non_object")):
            with self.subTest(expected=expected):
                result = validate_applied_norm_audit(self.audit, logs)
                self.assertFalse(result["passed"])
                self.assertTrue(any(expected in error for error in result["applied_validation"]["errors"]))
                json.dumps(result, allow_nan=False)

    def test_correct_entry_selected_and_order_site_rank_operation_cannot_be_silently_changed(self):
        unrelated = self.log(layer=3)
        self.assertTrue(validate_applied_norm_audit(self.audit, [unrelated, self.log()])["passed"])
        for change in ({"positions": [7, 19]}, {"positions": [True, 7]}, {"site": "mlp_out"}, {"rank": 1}, {"operation": "replace"}):
            result = validate_applied_norm_audit(self.audit, self.log(**change))
            self.assertFalse(result["passed"])
            self.assertFalse(all(result["applied_validation"]["geometry_matches"].values()))

    def test_invalid_actual_measurements_are_explicit_json_safe_failed_results(self):
        for values in ([2.5], [2.5, -1], [2.5, float("nan")], [True, 1], [10**400, 1], None):
            result = validate_applied_norm_audit(self.audit, self.log(delta_l2=values))
            self.assertFalse(result["passed"])
            self.assertTrue(any("actual_delta_l2" in error for error in result["applied_validation"]["errors"]))
            self.assertEqual(len(result["applied_validation"]["actual_delta_l2"]), 2)
            json.dumps(result, allow_nan=False)

    def test_extreme_relative_mismatch_still_returns_json_safe_audit(self):
        audit = self.prepare(donor=self.recipient, dose=1)
        audit["expected_task_projected_l2"] = [1e-320, 0]
        audit["expected_delta_l2"] = [1e-320, 0]
        result = validate_applied_norm_audit(audit, self.log(dose=1, delta_l2=[1e38, 0]))
        self.assertFalse(result["passed"])
        row = result["applied_validation"]["per_position"][0]
        self.assertIsNone(row["relative_error"])
        self.assertTrue(row["relative_error_overflow"])
        json.dumps(result, allow_nan=False)

    def test_random_control_uses_original_task_expectation_and_actual_write_measurement(self):
        random = projected_norm_matched_random_control(self.recipient, self.donor, self.basis, seed=442)
        actual = 0.5 * (((random["donor"] - self.recipient) @ random["basis"]) @ random["basis"].T)
        norms = np.linalg.norm(actual.astype(np.float64), axis=-1).tolist()
        result = validate_applied_norm_audit(self.audit, self.log(delta_l2=norms))
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_task_projected_l2"], [5, 2])
        # Even a mathematically matched synthetic donor fails if the actual
        # model-dtype write rounds away its update. Never substitute ideal norms.
        rounded_away = validate_applied_norm_audit(self.audit, self.log(delta_l2=[0, 0]))
        self.assertFalse(rounded_away["passed"])

    def test_vector_preparation_hash_matches_actual_engine_before_zero_batch_index(self):
        audit = self.prepare(recipient=self.recipient[0], donor=self.donor[0], positions=[19])
        before = self.recipient[0][None, None, :]
        self.assertEqual(audit["recipient_shape"], [1, 4])
        self.assertEqual(audit["expected_recipient_float32_sha256"], array_fingerprint(before[0]))
        self.assertNotEqual(audit["expected_recipient_float32_sha256"], array_fingerprint(before))

    def test_prepared_audit_and_logs_are_not_mutated(self):
        original = deepcopy(self.audit)
        log = self.log()
        original_log = deepcopy(log)
        result = validate_applied_norm_audit(self.audit, log)
        self.assertEqual(self.audit, original)
        self.assertEqual(log, original_log)
        self.assertEqual(validate_applied_norm_audit(result, log), result)
        result["positions"][0] = 999
        self.assertEqual(self.audit, original)

    def test_invalid_calling_inputs_and_relaxed_tolerance_rejected(self):
        bad = [{"relative_tolerance": 0.02}, {"relative_tolerance": 0}, {"dose": float("nan")},
               {"dose": True}, {"positions": [19]}, {"positions": [19, 19]}, {"positions": [True, 7]},
               {"positions": [-1, 7]}, {"layer": -1}, {"layer": True}, {"site": "other"},
               {"basis": self.basis * 2}, {"basis": np.zeros((4, 0))}, {"donor": self.donor[0]},
               {"recipient": np.zeros((0, 4)), "donor": np.zeros((0, 4))},
               {"recipient": np.full((2, 4), np.nan)}, {"recipient": np.ones((2, 4), dtype=complex)}]
        for changes in bad:
            with self.subTest(fields=list(changes)):
                with self.assertRaises(ValueError):
                    self.prepare(**changes)
        changed = deepcopy(self.audit)
        changed["relative_tolerance"] = 0.02
        with self.assertRaisesRegex(ValueError, "fixed relative"):
            validate_applied_norm_audit(changed, self.log())
        changed = deepcopy(self.audit)
        changed["expected_delta_l2"] = [5, 2]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_applied_norm_audit(changed, self.log())
        with self.assertRaises(ValueError):
            validate_applied_norm_audit(self.audit, "not a log")


if __name__ == "__main__":
    unittest.main()
