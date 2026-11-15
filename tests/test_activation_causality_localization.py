"""NumPy-only subspace provenance and extraction tests (unittest compatible)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.activation_causality.localization import extract_and_fit, fit_task_contrast_subspace


def rows(count):
    return [{"pair_id": f"pair{i:03}", "layout_id": f"layout{i:03}", "partition": "discovery", "family": "wall"}
            for i in range(count)]


class FakeEngine:
    def __init__(self, loaded=True):
        self.metadata = {"loaded": loaded, "model_revision": "fake", "head_mode": "full"}
        self.head_mode = "full"
        self.calls = []

    def encode(self, text):
        self.calls.append(("encode", text))
        return list(text.encode())

    def capture(self, token_ids, layers, positions, site):
        self.calls.append(("capture", list(token_ids), layers, positions, site))
        values = np.asarray([[token_ids[p], token_ids[p] * 2, 1, 0] for p in positions], dtype=np.float32)
        return {layers[0]: values}


def jobs(count=2):
    return [{**row, "experiment_id": "discovery_source", "recipient_text": "aabc", "donor_text": "xxyz",
             "selection": {"scope": "last_tokens", "last_k": 2}, "seed": 42}
            for row in rows(count)]


class SubspaceFitTests(unittest.TestCase):
    def test_recovers_known_contrast_direction_and_json_provenance(self):
        contrasts = [np.array([[1, 2, 0, 0], [2, 4, 0, 0]]), np.array([[3, 6, 0, 0]])]
        result = fit_task_contrast_subspace(contrasts, rows(2), 1, layer=10)
        expected = np.array([1, 2, 0, 0]) / np.sqrt(5)
        np.testing.assert_allclose(result["basis"][:, 0], expected, atol=1e-6)
        self.assertEqual(result["basis"].dtype, np.float32)
        metadata = result["metadata"]
        self.assertEqual(metadata["n_training_pairs"], 2)
        self.assertEqual(metadata["n_position_rows"], 3)
        self.assertEqual(metadata["rank"], 1)
        self.assertEqual(len(metadata["training_pair_ids_sha256"]), 64)
        self.assertIn("not independent proof", metadata["interpretation"])
        json.dumps(metadata, allow_nan=False)

    def test_full_hidden_rank_is_orthonormal_and_preserves_all_vectors(self):
        result = fit_task_contrast_subspace(np.eye(4), rows(4), 4, layer=15, site="mlp_out")
        basis = result["basis"]
        np.testing.assert_allclose(np.einsum("ir,is->rs", basis, basis), np.eye(4), atol=1e-6)
        np.testing.assert_allclose(np.einsum("ir,jr->ij", basis, basis), np.eye(4), atol=1e-6)
        self.assertEqual(result["metadata"]["captured_energy_fraction"], 1.0)

    def test_planned_ranks_one_four_sixteen(self):
        matrix = np.diag(np.arange(16, 0, -1)).astype(np.float32)
        for rank in (1, 4, 16):
            result = fit_task_contrast_subspace(matrix, rows(16), rank, layer=10)
            self.assertEqual(result["basis"].shape, (16, rank))
            np.testing.assert_allclose(np.einsum("ir,is->rs", result["basis"], result["basis"]), np.eye(rank), atol=1e-6)

    def test_pair_reordering_does_not_change_basis_or_fingerprints(self):
        rng = np.random.default_rng(61)
        contrasts = [rng.normal(size=(i + 1, 8)).astype(np.float32) for i in range(5)]
        provenance = rows(5)
        first = fit_task_contrast_subspace(contrasts, provenance, 4, layer=10)
        permutation = [4, 1, 3, 0, 2]
        second = fit_task_contrast_subspace([contrasts[i] for i in permutation], [provenance[i] for i in permutation], 4, layer=10)
        np.testing.assert_array_equal(first["basis"], second["basis"])
        for key in ("contrast_sha256", "basis_sha256", "training_membership_sha256", "training_pair_ids_sha256", "training_layout_ids_sha256"):
            self.assertEqual(first["metadata"][key], second["metadata"][key])

    def test_equal_pair_weight_prevents_long_span_dominance(self):
        contrasts = [np.tile([1, 0], (100, 1)), np.array([[0, 1]])]
        result = fit_task_contrast_subspace(contrasts, rows(2), 2, layer=3)
        np.testing.assert_allclose(result["metadata"]["explained_energy_fraction"], [0.5, 0.5], atol=1e-10)

    def test_rank_cannot_be_inflated_by_token_rows_or_duplicates(self):
        with self.assertRaisesRegex(ValueError, "distinct discovery training pairs"):
            fit_task_contrast_subspace([np.eye(4), np.eye(4)], rows(2), 4, layer=10)
        repeated = [rows(1)[0], rows(1)[0]]
        with self.assertRaisesRegex(ValueError, "distinct pair"):
            fit_task_contrast_subspace([np.eye(4), np.eye(4)], repeated, 1, layer=10)

    def test_numerical_rank_and_invalid_data_fail(self):
        with self.assertRaisesRegex(ValueError, "numerical contrast rank"):
            fit_task_contrast_subspace(np.tile([1, 2, 3], (4, 1)), rows(4), 2, layer=10)
        with self.assertRaisesRegex(ValueError, "numerical contrast rank"):
            fit_task_contrast_subspace(np.zeros((4, 3)), rows(4), 1, layer=10)
        for values in ([[np.nan, 1]], [[], []]):
            with self.assertRaises(ValueError):
                fit_task_contrast_subspace([np.asarray(values)], rows(1), 1, layer=10)

    def test_centering_is_explicit_and_removes_constant_contrast(self):
        constant = np.tile([1, 0, 0], (4, 1))
        fit_task_contrast_subspace(constant, rows(4), 1, layer=10)
        with self.assertRaisesRegex(ValueError, "numerical contrast rank"):
            fit_task_contrast_subspace(constant, rows(4), 1, layer=10, center=True)

    def test_rejects_all_non_discovery_and_nested_contradictions(self):
        for partition in ("dev", "confirmation", "historical", "test", None):
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                fit_task_contrast_subspace([[1, 2]], [{**rows(1)[0], "partition": partition}], 1, layer=10)
        for changed in ({"donor_partition": "confirmation"},
                        {"metadata": {"pair": {"partition": "confirmation"}}},
                        {"metadata": {"pair": {"layout_id": "different"}}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                fit_task_contrast_subspace([[1, 2]], [{**rows(1)[0], **changed}], 1, layer=10)


class ExtractionTests(unittest.TestCase):
    def test_extracts_root_last_token_positions_and_deduplicates_seeds(self):
        engine = FakeEngine()
        training_jobs = jobs()
        training_jobs.append({**training_jobs[0], "seed": 123})
        result = extract_and_fit(engine, training_jobs, 10, "resid_post", 1)
        captures = [call for call in engine.calls if call[0] == "capture"]
        self.assertEqual(len(captures), 4)
        self.assertTrue(all(call[2:] == ([10], [2, 3], "resid_post") for call in captures))
        self.assertEqual(result["metadata"]["extraction"]["input_job_count"], 3)
        self.assertEqual(result["metadata"]["n_training_pairs"], 2)
        self.assertFalse(result["metadata"]["extraction"]["model_loading_performed"])
        self.assertEqual(result["metadata"]["training_rows"][0]["recipient_positions"], [2, 3])
        json.dumps(result["metadata"], allow_nan=False)

    def test_rejects_late_leakage_before_any_engine_call(self):
        engine = FakeEngine()
        training_jobs = jobs()
        training_jobs[1]["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            extract_and_fit(engine, training_jobs, 10, "resid_post", 1)
        self.assertEqual(engine.calls, [])

    def test_never_loads_an_unloaded_engine(self):
        engine = FakeEngine(loaded=False)
        with self.assertRaisesRegex(RuntimeError, "already-loaded"):
            extract_and_fit(engine, jobs(), 10, "resid_post", 1)
        self.assertEqual(engine.calls, [])

    def test_conflicting_pair_definitions_and_missing_donors_fail_early(self):
        for training_jobs in ([*jobs(1), {**jobs(1)[0], "donor_text": "other"}],
                              [{**jobs(1)[0], "donor_text": None}]):
            engine = FakeEngine()
            with self.assertRaises(ValueError):
                extract_and_fit(engine, training_jobs, 10, "resid_post", 1)
            self.assertEqual(engine.calls, [])

    def test_rank_limit_is_checked_before_encoding(self):
        engine = FakeEngine()
        with self.assertRaises(ValueError):
            extract_and_fit(engine, jobs(2), 10, "resid_post", 4)
        self.assertEqual(engine.calls, [])

    def test_explicit_position_maps_override_last_token_selection(self):
        engine = FakeEngine()
        training_jobs = [{**jobs(1)[0], "recipient_positions": [0, 2], "donor_positions": [1, 3]}]
        result = extract_and_fit(engine, training_jobs, 15, "attn_out", 1)
        captures = [call for call in engine.calls if call[0] == "capture"]
        self.assertEqual(captures[0][3], [0, 2])
        self.assertEqual(captures[1][3], [1, 3])
        self.assertEqual(result["metadata"]["site"], "attn_out")

    def test_frozen_ids_and_nested_explicit_positions_are_authoritative(self):
        engine = FakeEngine()
        source = {**jobs(1)[0], "recipient_token_ids": [1, 2, 3], "donor_token_ids": [4, 5, 6],
                  "selection": {"scope": "explicit", "recipient_positions": [0, 2], "donor_positions": [1, 2]}}
        result = extract_and_fit(engine, [source], 10, "resid_post", 1)
        self.assertFalse(any(call[0] == "encode" for call in engine.calls))
        captures = [call for call in engine.calls if call[0] == "capture"]
        self.assertEqual(captures[0][1], [1, 2, 3])
        self.assertEqual(captures[1][1], [4, 5, 6])
        self.assertEqual(result["metadata"]["training_rows"][0]["recipient_positions"], [0, 2])
        with self.assertRaisesRegex(ValueError, "contradict"):
            extract_and_fit(FakeEngine(), [{**source, "recipient_positions": [1, 2]}], 10, "resid_post", 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            extract_and_fit(FakeEngine(), [source, {**source, "donor_token_ids": [7, 8, 9]}], 10, "resid_post", 1)

    def test_invalid_frozen_ids_rejected(self):
        for ids in ([], [-1], [True], [1.2]):
            source = {**jobs(1)[0], "recipient_token_ids": ids, "donor_token_ids": [1, 2]}
            engine = FakeEngine()
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                extract_and_fit(engine, [source], 10, "resid_post", 1)
            self.assertFalse(any(call[0] == "capture" for call in engine.calls))

    def test_derived_cut_provenance_requires_matching_original_and_does_not_inflate_rank(self):
        original = jobs(1)[0]
        source = {**original, "pair_id": original["pair_id"] + "::cut64",
                  "metadata": {"source_pair_id": original["pair_id"], "pair": rows(1)[0]}}
        result = extract_and_fit(FakeEngine(), [source], 10, "resid_post", 1)
        self.assertEqual(result["metadata"]["training_rows"][0]["source_pair_id"], original["pair_id"])
        bad = {**source, "metadata": {**source["metadata"], "source_pair_id": "wrong_source"}}
        with self.assertRaisesRegex(ValueError, "contradicts"):
            extract_and_fit(FakeEngine(), [bad], 10, "resid_post", 1)
        next_cut = {**source, "pair_id": original["pair_id"] + "::cut128"}
        with self.assertRaisesRegex(ValueError, "distinct discovery training pairs"):
            extract_and_fit(FakeEngine(), [source, next_cut], 10, "resid_post", 2)


if __name__ == "__main__":
    unittest.main()
