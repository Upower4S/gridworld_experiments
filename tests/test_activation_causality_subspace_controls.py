"""CPU-only mathematical tests for projected-size random-subspace controls."""

import json
import unittest

import numpy as np

from src.activation_causality.localization import fit_task_contrast_subspace
from src.activation_causality.subspace_controls import projected_norm_matched_random_control


class ProjectedNormControlTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(517)
        self.recipient = self.rng.normal(size=(5, 32)).astype(np.float32)
        self.delta = (self.rng.normal(size=(5, 32)) * np.array([0.1, 0.3, 1, 3, 10])[:, None]).astype(np.float32)
        self.donor = self.recipient + self.delta
        self.task_basis = np.linalg.qr(self.rng.normal(size=(32, 4)))[0].astype(np.float32)

    def result(self, **overrides):
        arguments = dict(recipient=self.recipient, donor=self.donor, task_basis=self.task_basis, seed=731)
        return projected_norm_matched_random_control(**{**arguments, **overrides})

    def test_each_position_matches_its_task_projected_norm_and_rank(self):
        result = self.result()
        delta64 = self.donor.astype(np.float64) - self.recipient.astype(np.float64)
        task64 = self.task_basis.astype(np.float64)
        expected = np.linalg.norm((delta64 @ task64) @ task64.T, axis=-1)
        audit = result["audit"]
        self.assertEqual(result["basis"].shape, (32, 4))
        self.assertEqual(result["donor"].shape, (5, 32))
        self.assertEqual(result["basis"].dtype, np.float32)
        self.assertEqual(result["donor"].dtype, np.float32)
        self.assertEqual(audit["rank"], 4)
        self.assertEqual(audit["n_positions"], 5)
        np.testing.assert_allclose(audit["target_projected_l2"], expected, rtol=1e-14)
        np.testing.assert_allclose(audit["scaled_random_projected_l2"], expected, rtol=1e-14)
        self.assertGreater(np.ptp(audit["scale_factors"]), 0.1)
        self.assertGreater(np.ptp(expected), 1)
        self.assertTrue(audit["float32"]["all_norms_match_within_tolerance"])
        for cosine in audit["task_random_cosine"]:
            self.assertGreaterEqual(cosine, -1)
            self.assertLessEqual(cosine, 1)
        json.dumps(audit, allow_nan=False)

    def test_exact_compiler_q_draw_reproducibility_without_global_rng_mutation(self):
        before = np.random.get_state()
        first = self.result()
        second = self.result()
        after = np.random.get_state()
        expected, _ = np.linalg.qr(np.random.default_rng(731).standard_normal(self.task_basis.shape))
        np.testing.assert_array_equal(first["basis"], expected.astype(np.float32))
        np.testing.assert_array_equal(first["basis"], second["basis"])
        np.testing.assert_array_equal(first["donor"], second["donor"])
        self.assertEqual(first["audit"], second["audit"])
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(first["audit"]["random_draws"], 1)
        self.assertFalse(first["audit"]["resampled"])

    def test_different_seed_changes_direction_but_preserves_matching(self):
        first, second = self.result(), self.result(seed=732)
        self.assertFalse(np.array_equal(first["basis"], second["basis"]))
        self.assertFalse(np.allclose(first["donor"], second["donor"]))
        np.testing.assert_allclose(first["audit"]["target_projected_l2"], second["audit"]["target_projected_l2"])
        np.testing.assert_allclose(first["audit"]["scaled_random_projected_l2"], second["audit"]["scaled_random_projected_l2"], rtol=1e-14)

    def test_zero_task_projection_is_identity_even_with_nonzero_random_projection(self):
        recipient = np.zeros((3, 4), dtype=np.float32)
        donor = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
        basis = np.eye(4, dtype=np.float32)[:, :1]
        result = self.result(recipient=recipient, donor=donor, task_basis=basis)
        np.testing.assert_array_equal(result["donor"][:2], recipient[:2])
        self.assertEqual(result["audit"]["zero_target_identity"], [True, True, False])
        self.assertGreater(result["audit"]["unscaled_random_projected_l2"][0], 0)
        self.assertEqual(result["audit"]["task_random_cosine"][:2], [None, None])
        self.assertEqual(result["audit"]["scale_factors"][:2], [0, 0])
        self.assertEqual(result["audit"]["float32"]["reprojected_control_l2"][:2], [0, 0])
        self.assertEqual(result["audit"]["n_positions"], 3)

    def test_nonzero_target_with_degenerate_random_projection_fails_without_resampling(self):
        q, _ = np.linalg.qr(np.random.default_rng(731).standard_normal((2, 1)))
        delta = np.array([-q[1, 0], q[0, 0]])
        self.assertGreater(abs(delta[0]), 0.1)
        with self.assertRaisesRegex(ValueError, r"degenerate random projection at positions \[0\]; no resampling"):
            self.result(recipient=np.zeros(2), donor=delta, task_basis=np.eye(2)[:, :1])

    def test_small_nonzero_projection_does_not_silently_become_zero_identity(self):
        result = self.result(recipient=np.zeros(4), donor=np.array([1e-100, 0, 0, 0]), task_basis=np.eye(4)[:, :1])
        self.assertEqual(result["audit"]["zero_target_identity"], [False])
        self.assertGreater(result["audit"]["target_projected_l2"][0], 0)
        self.assertFalse(result["audit"]["float32"]["all_norms_match_within_tolerance"])
        self.assertEqual(result["audit"]["float32"]["reprojected_control_relative_error_vs_float64_target"], [1])
        json.dumps(result["audit"], allow_nan=False)

    def test_float32_reprojection_audit_matches_downstream_formula_and_is_idempotent(self):
        result = self.result()
        q, synthetic = result["basis"], result["donor"]
        applied = ((synthetic - self.recipient) @ q) @ q.T
        reapplied = (applied @ q) @ q.T
        np.testing.assert_allclose(applied, reapplied, atol=3e-6, rtol=3e-6)
        actual = np.linalg.norm(applied.astype(np.float64), axis=-1)
        np.testing.assert_allclose(actual, result["audit"]["float32"]["reprojected_control_l2"], rtol=1e-14)
        np.testing.assert_allclose(q.T @ q, np.eye(4), atol=2e-7)
        self.assertFalse(result["audit"]["engine_applied_norms_measured"])
        self.assertIn("BF16", result["audit"]["engine_audit_requirement"])

    def test_dose_is_applied_once_by_downstream_subspace_operation(self):
        result = self.result()
        q = result["basis"]
        delta = ((result["donor"] - self.recipient) @ q) @ q.T
        for dose in (0, 0.5, 1, -1):
            actual_delta = dose * delta
            np.testing.assert_allclose(np.linalg.norm(actual_delta.astype(np.float64), axis=-1),
                                       abs(dose) * np.array(result["audit"]["target_projected_l2"]), rtol=2e-6, atol=1e-7)
        self.assertFalse(result["audit"]["dose_applied"])

    def test_vector_and_full_rank_geometry(self):
        recipient, donor = self.recipient[0], self.donor[0]
        result = self.result(recipient=recipient, donor=donor, task_basis=np.eye(32))
        self.assertEqual(result["donor"].shape, (32,))
        self.assertEqual(result["basis"].shape, (32, 32))
        np.testing.assert_allclose(result["donor"], donor, rtol=1e-6, atol=2e-7)
        self.assertAlmostEqual(result["audit"]["task_random_cosine"][0], 1)
        np.testing.assert_allclose(result["audit"]["scale_factors"], [1], rtol=1e-14)

    def test_inputs_not_mutated_or_aliased_by_returned_arrays(self):
        original = [a.copy() for a in (self.recipient, self.donor, self.task_basis)]
        result = self.result()
        result["donor"][:] = 0
        result["basis"][:] = 0
        for actual, expected in zip((self.recipient, self.donor, self.task_basis), original, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_real_fitted_discovery_basis_at_ranks_one_and_four(self):
        contrasts = self.rng.normal(size=(8, 3, 12)).astype(np.float32)
        metadata = [{"pair_id": f"pair_{i}", "layout_id": f"layout_{i}", "partition": "discovery"} for i in range(8)]
        recipient, donor = self.rng.normal(size=(2, 3, 12)).astype(np.float32)
        for rank in (1, 4):
            fitted = fit_task_contrast_subspace(contrasts, metadata, rank, layer=10)
            basis = fitted["basis"]
            result = self.result(recipient=recipient, donor=donor, task_basis=basis)
            target = np.linalg.norm(((donor.astype(np.float64) - recipient.astype(np.float64)) @ basis) @ basis.T, axis=-1)
            actual = np.linalg.norm(((result["donor"] - recipient) @ result["basis"]) @ result["basis"].T, axis=-1)
            np.testing.assert_allclose(actual, target, rtol=2e-6, atol=1e-7)
            self.assertEqual(result["audit"]["rank"], rank)

    def test_malformed_shape_nonfinite_basis_and_seed_validation(self):
        cases = [
            {"recipient": np.zeros((0, 32)), "donor": np.zeros((0, 32))},
            {"donor": np.zeros((1, 32))}, {"recipient": np.zeros((1, 5, 32))},
            {"task_basis": np.zeros((31, 4))}, {"task_basis": np.zeros((32, 0))},
            {"task_basis": np.zeros((32, 33))}, {"task_basis": self.task_basis * 2},
            {"task_basis": np.repeat(self.task_basis[:, :1], 4, axis=1)},
            {"task_basis": np.full((32, 4), np.nan)}, {"donor": np.full((5, 32), np.inf)},
            {"donor": np.full((5, 32), 1e100)}, {"donor": self.donor.astype(complex)},
            {"donor": [["text"]]}, {"seed": True}, {"seed": -1}, {"seed": 2**32}, {"seed": 1.5},
        ]
        for override in cases:
            with self.subTest(override=tuple(override)):
                with self.assertRaises(ValueError):
                    self.result(**override)


if __name__ == "__main__":
    unittest.main()
