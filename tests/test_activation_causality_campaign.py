"""CPU tests for the job-to-intervention compiler; no model is loaded."""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality.campaign import compile_patch
from src.activation_causality.subspace_audit import array_fingerprint
from src.activation_causality.subspace_controls import projected_norm_matched_random_control


class FakeEngine:
    def __init__(self, model_path="."):
        self.encodes = []
        self.captures = []
        self.model_path = Path(model_path).resolve()

    def encode(self, text):
        self.encodes.append(text)
        return list(text.encode())

    @staticmethod
    def values(ids, layer, positions):
        return np.asarray([[ids[p], 2 * ids[p], layer + 1, p + 1] for p in positions], dtype=np.float32)

    def capture(self, ids, layers, positions, site):
        self.captures.append((ids.copy(), layers.copy(), positions.copy(), site))
        return {layer: self.values(ids, layer, positions) for layer in layers}


def job(**updates):
    value = {"layout_id": "eval_layout", "family": "wall", "seed": 42, "recipient_text": "ignored R", "donor_text": "ignored D",
             "recipient_token_ids": [1, 2, 3, 4], "donor_token_ids": [5, 7, 9, 11],
             "selection": {"scope": "explicit", "recipient_positions": [1, 3], "donor_positions": [1, 3]},
             "intervention": {"layers": [10], "site": "resid_post", "operation": "replace", "dose": 0.5}}
    value.update(updates)
    return value


class CompilerTests(unittest.TestCase):
    def test_authoritative_ids_and_explicit_positions_are_used_without_encoding(self):
        engine, source = FakeEngine(), job()
        ids, patches = compile_patch(engine, source, OrderedDict())
        self.assertEqual(ids, source["recipient_token_ids"])
        self.assertEqual(engine.encodes, [])
        self.assertEqual(engine.captures[0][0], source["donor_token_ids"])
        self.assertEqual(patches[0]["positions"], [1, 3])
        self.assertEqual(patches[0]["dose"], 0.5)

    def test_baseline_requires_no_donor_or_capture(self):
        source = job(donor_text=None, donor_token_ids=None, intervention={"layers": []})
        engine = FakeEngine()
        ids, patches = compile_patch(engine, source, OrderedDict())
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(patches, [])
        self.assertEqual(engine.captures, [])

    def test_invalid_frozen_ids_and_positions_fail(self):
        for bad_ids in ([], [-1], [True], [1.5]):
            with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                compile_patch(FakeEngine(), job(recipient_token_ids=bad_ids), OrderedDict())
        for bad_positions in ([], [-1], [4], [True], [1, 1]):
            selection = {"scope": "explicit", "recipient_positions": bad_positions, "donor_positions": [1, 3]}
            with self.subTest(positions=bad_positions), self.assertRaises(ValueError):
                compile_patch(FakeEngine(), job(selection=selection), OrderedDict())
        with self.assertRaises(ValueError):
            compile_patch(FakeEngine(), job(selection={"scope": "explicit", "recipient_positions": [1], "donor_positions": [1, 3]}), OrderedDict())

    def test_add_compiles_actual_donor_minus_recipient_not_absolute_donor(self):
        source = job()
        source["intervention"]["operation"] = "add"
        _, patches = compile_patch(FakeEngine(), source, OrderedDict())
        expected = FakeEngine.values(source["donor_token_ids"], 10, [1, 3]) - FakeEngine.values(source["recipient_token_ids"], 10, [1, 3])
        np.testing.assert_array_equal(patches[0]["donor"], expected)

    def test_random_replacement_matches_each_token_delta_norm_and_seed(self):
        source = job()
        source["intervention"]["random_control"] = True
        _, patches = compile_patch(FakeEngine(), source, OrderedDict())
        _, repeated = compile_patch(FakeEngine(), source, OrderedDict())
        original = FakeEngine.values(source["recipient_token_ids"], 10, [1, 3])
        donor = FakeEngine.values(source["donor_token_ids"], 10, [1, 3])
        np.testing.assert_allclose(np.linalg.norm(patches[0]["donor"] - original, axis=1), np.linalg.norm(donor - original, axis=1), rtol=1e-6)
        np.testing.assert_array_equal(patches[0]["donor"], repeated[0]["donor"])
        self.assertFalse(np.allclose(patches[0]["donor"], donor))
        source["intervention"]["operation"] = "add"
        _, added = compile_patch(FakeEngine(), source, OrderedDict())
        np.testing.assert_allclose(added[0]["donor"], patches[0]["donor"] - original)

    def test_capture_reuse_does_not_cross_ids_or_sites(self):
        engine, cache, source = FakeEngine(), OrderedDict(), job()
        compile_patch(engine, source, cache)
        compile_patch(engine, source, cache)
        self.assertEqual(len(engine.captures), 1)
        changed = deepcopy(source)
        changed["intervention"]["site"] = "mlp_out"
        compile_patch(engine, changed, cache)
        self.assertEqual(len(engine.captures), 2)

    def subspace_job(self, **controls):
        source = job()
        source["intervention"].update(operation="subspace", basis_path="/tmp/basis.npy", basis_sha256="a" * 64,
                                      basis_metadata_path="/tmp/basis.json", basis_metadata_sha256="b" * 64,
                                      **controls)
        return source

    def verified_loader(self, proof):
        basis = np.eye(4, dtype=np.float32)[:, :2]

        def load(intervention, source, engine, supplied_proof):
            if supplied_proof is not proof:
                raise ValueError("subspace requires this engine's verified loaded-model proof")
            self.assertEqual(intervention["operation"], "subspace")
            self.assertEqual(source["family"], "wall")
            return basis.copy(), {"rank": 2, "basis_id": "basis"}

        return patch("src.activation_causality.basis_runtime.load_verified_basis", side_effect=load)

    def test_subspace_requires_verified_proof_before_capture_and_preserves_zero(self):
        source, proof, engine = self.subspace_job(zero_donor=True), object(), FakeEngine()
        with self.verified_loader(proof) as loader:
            with self.assertRaises(ValueError):
                compile_patch(engine, source, OrderedDict())
            self.assertEqual(engine.captures, [])
            _, patches = compile_patch(engine, source, OrderedDict(), loaded_model_proof=proof)
            self.assertIs(loader.call_args.args[3], proof)
        np.testing.assert_array_equal(patches[0]["basis"], np.eye(4, dtype=np.float32)[:, :2])
        np.testing.assert_array_equal(patches[0]["donor"], np.zeros((2, 4)))

    def test_loader_failure_is_before_any_capture(self):
        engine = FakeEngine()
        with patch("src.activation_causality.basis_runtime.load_verified_basis", side_effect=ValueError("invalid frozen basis")):
            with self.assertRaisesRegex(ValueError, "invalid frozen basis"):
                compile_patch(engine, self.subspace_job(), OrderedDict(), loaded_model_proof=object())
        self.assertEqual(engine.captures, [])

    def test_random_basis_retains_rank_only_deterministic_orthonormal_control(self):
        source, proof = self.subspace_job(random_basis=True, random_seed=731), object()
        with self.verified_loader(proof):
            _, first = compile_patch(FakeEngine(), source, OrderedDict(), loaded_model_proof=proof)
            _, second = compile_patch(FakeEngine(), source, OrderedDict(), loaded_model_proof=proof)
        basis = first[0]["basis"]
        self.assertEqual(basis.shape, (4, 2))
        np.testing.assert_array_equal(basis, second[0]["basis"])
        np.testing.assert_allclose(np.einsum("ir,is->rs", basis, basis), np.eye(2), atol=1e-6)
        self.assertNotIn("projected_norm_audit", first[0])
        self.assertNotIn("expected_recipient_float32_sha256", first[0])

    def test_matched_random_control_and_forward_audit_same_original_task(self):
        proof = object()
        intended = self.subspace_job(audit_projected_norm=True, projected_norm_relative_tolerance=0.01)
        matched = self.subspace_job(projected_norm_matched_random_basis=True, random_seed=731, audit_projected_norm=True, projected_norm_relative_tolerance=0.01)
        original = FakeEngine.values(intended["recipient_token_ids"], 10, [1, 3])
        donor = FakeEngine.values(intended["donor_token_ids"], 10, [1, 3])
        task_basis = np.eye(4, dtype=np.float32)[:, :2]
        expected = projected_norm_matched_random_control(original, donor, task_basis, 731)
        with self.verified_loader(proof):
            intended_engine, matched_engine = FakeEngine(), FakeEngine()
            _, forward = compile_patch(intended_engine, intended, OrderedDict(), loaded_model_proof=proof)
            _, random = compile_patch(matched_engine, matched, OrderedDict(), loaded_model_proof=proof)
        self.assertEqual(len(intended_engine.captures), 2)
        self.assertEqual(len(matched_engine.captures), 2)
        np.testing.assert_array_equal(random[0]["donor"], expected["donor"])
        np.testing.assert_array_equal(random[0]["basis"], expected["basis"])
        np.testing.assert_array_equal(forward[0]["donor"], donor)
        for patch_value in (forward[0], random[0]):
            audit = patch_value["projected_norm_audit"]
            self.assertEqual(audit["relative_tolerance"], 0.01)
            self.assertEqual(audit["original_donor_float32_sha256"], array_fingerprint(donor))
            self.assertEqual(audit["original_task_basis_float32_sha256"], array_fingerprint(task_basis))
            self.assertEqual(patch_value["expected_recipient_float32_sha256"], array_fingerprint(original))
            self.assertEqual(audit["expected_delta_l2"], [0.5 * value for value in audit["expected_task_projected_l2"]])
            self.assertEqual(patch_value["dose"], 0.5)
        self.assertEqual(forward[0]["projected_norm_audit"], random[0]["projected_norm_audit"])

    def test_matched_control_does_not_silently_combine_with_other_controls(self):
        proof = object()
        for flag in ("random_basis", "random_control", "zero_donor"):
            source = self.subspace_job(projected_norm_matched_random_basis=True, audit_projected_norm=True, projected_norm_relative_tolerance=0.01, **{flag: True})
            engine = FakeEngine()
            with self.subTest(flag=flag), self.verified_loader(proof), self.assertRaises(ValueError):
                compile_patch(engine, source, OrderedDict(), loaded_model_proof=proof)
            self.assertEqual(engine.captures, [])
        for value in (0.02, 0.005, True):
            source = self.subspace_job(projected_norm_matched_random_basis=True, audit_projected_norm=True, projected_norm_relative_tolerance=value)
            engine = FakeEngine()
            with self.subTest(tolerance=value), self.verified_loader(proof), self.assertRaises(ValueError):
                compile_patch(engine, source, OrderedDict(), loaded_model_proof=proof)
            self.assertEqual(engine.captures, [])

    def test_matched_control_requires_explicit_frozen_audit_and_tolerance(self):
        proof = object()
        for controls in ({"projected_norm_matched_random_basis": True},
                         {"projected_norm_matched_random_basis": True, "audit_projected_norm": True}):
            engine = FakeEngine()
            with self.subTest(controls=controls), self.verified_loader(proof), self.assertRaises(ValueError):
                compile_patch(engine, self.subspace_job(**controls), OrderedDict(), loaded_model_proof=proof)
            self.assertEqual(engine.captures, [])


if __name__ == "__main__":
    unittest.main()
