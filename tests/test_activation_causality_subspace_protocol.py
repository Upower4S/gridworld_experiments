"""CPU-only basis registration and held-out protocol checks."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality import registry
from src.activation_causality.basis_runtime import load_verified_basis, make_loaded_model_proof
from src.activation_causality.campaign import make_job
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS, engine_dependency_hashes
from src.activation_causality.model_identity import snapshot_identity
from src.activation_causality.subspace_protocol import build_subspace_spec, fit_basis


class FakeEngine:
    constructions = 0
    captures = 0
    validation_passes = True
    on_capture = None

    def __init__(self, model_path):
        type(self).constructions += 1
        self.model_path = Path(model_path)
        self.metadata = {"loaded": True, "model_path": str(model_path), "hidden_dim": 4, "test_double": True}
        self.head_mode = "full"

    def validate(self):
        return {"passed": self.validation_passes, "test_double": True}

    def encode(self, text):
        return [int(text)]

    def capture(self, token_ids, layers, positions, site):
        type(self).captures += 1
        if type(self).on_capture is not None:
            type(self).on_capture()
        value = token_ids[-1]
        row = np.array([value ** power for power in range(1, 5)], dtype=np.float32)
        return {layers[0]: np.tile(row, (len(positions), 1))}


def source_spec(prefix="train", n_pairs=2):
    jobs = []
    for i in range(n_pairs):
        for seed in (42, 43):
            jobs.append(make_job(prefix,
                        **{
                         "pair_id": f"{prefix}pair{i}", "layout_id": f"{prefix}layout{i}",
                         "layout_aliases": [f"{prefix}layout{i}", f"{prefix}variant{i}"],
                         "partition": "discovery", "condition": "layer10", "family": "wall", "seed": seed,
                         "target_action": "RIGHT", "original_action": "LEFT", "recipient_text": str(i + 1),
                         "donor_text": str(i + 2), "recipient_token_ids": [i + 1], "donor_token_ids": [i + 2],
                         "selection": {"scope": "last_tokens", "last_k": 1},
                         "intervention": {"layers": [10], "site": "resid_post", "operation": "replace", "dose": 1.0},
                         "natural": True, "metadata": {}}))
    return {"experiment_id": prefix, "partition": "discovery", "jobs": jobs}


@contextmanager
def fake_lock(campaign):
    yield


class FitBasisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.campaign = self.repo / "campaign"
        self.model = self.repo / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")
        (self.model / "weights-1.safetensors").write_bytes(b"indexed fake weight payload one")
        (self.model / "weights-2.safetensors").write_bytes(b"indexed fake weight payload two")
        registry.atomic_json(self.model / "model.safetensors.index.json", {
            "weight_map": {"layer.a": "weights-1.safetensors", "layer.b": "weights-2.safetensors"}})
        registry.atomic_json(self.campaign / "campaign.json", {"model_path": str(self.model),
                             "deadline": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
        registry.atomic_json(self.campaign / "specs/train.json", source_spec())
        FakeEngine.constructions = FakeEngine.captures = 0
        FakeEngine.validation_passes = True
        FakeEngine.on_capture = None
        self.engine_patch = patch("src.activation_causality.engine.MLXInterventionEngine", FakeEngine)
        self.lock_patch = patch("src.activation_causality.registry.worker_lock", fake_lock)
        self.runtime = {"sources": {"src/activation_causality/engine.py": "e" * 64},
                        "versions": {"mlx": "cpu-test-double", "mlx-lm": "cpu-test-double", "numpy": np.__version__}}
        self.runtime["sources"].update({name: "a" * 64 for name in ENGINE_DEPENDENCY_PATHS})
        self.runtime_patch = patch("src.activation_causality.campaign.runtime_fingerprint", return_value=self.runtime)
        self.engine_patch.start()
        self.lock_patch.start()
        self.runtime_patch.start()

    def tearDown(self):
        self.engine_patch.stop()
        self.lock_patch.stop()
        self.runtime_patch.stop()
        self.temp.cleanup()

    def fit(self, basis_id="basis"):
        return fit_basis(self.campaign, self.repo, "train", "layer10", "wall", 10, "resid_post", 2, basis_id)

    def test_fit_is_validated_deduped_hash_bound_and_idempotent(self):
        metadata = self.fit()
        self.assertEqual(FakeEngine.captures, 4)
        self.assertEqual(len(metadata["training_pair_ids"]), 2)
        self.assertEqual(metadata["model_path"], str(self.model.resolve()))
        self.assertIn("tokenizer.json", metadata["model_files"])
        self.assertIn("weights-1.safetensors", metadata["model_files"])
        self.assertIn("weights-2.safetensors", metadata["model_files"])
        self.assertEqual(metadata["model_provenance_version"], 2)
        identity = metadata["model_snapshot_identity"]
        self.assertEqual(identity["files"], metadata["model_files"])
        self.assertEqual(identity["identity_sha256"], metadata["model_snapshot_identity_sha256"])
        self.assertTrue(metadata["model_snapshot_verified_before_and_after_fit"])
        audit_path = Path(metadata["model_snapshot_audit_path"])
        audit = registry.read_json(audit_path)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["before_fit"], audit["after_fit"])
        self.assertEqual(audit["before_fit"], identity)
        self.assertEqual(registry.file_sha256(audit_path), metadata["model_snapshot_audit_sha256"])
        source = source_spec()
        expected = {job["job_id"]: registry.fingerprint(job) for job in source["jobs"] if job["seed"] == 42}
        self.assertEqual(metadata["source_job_fingerprints"], expected)
        request = registry.read_json(self.campaign / "bases/basis.request.json")
        self.assertEqual(request["model_snapshot_identity"], identity)
        self.assertEqual(request["source_job_fingerprints"], expected)
        self.assertEqual(request["selected_source_definitions_sha256"], metadata["selected_source_definitions_sha256"])
        representation = metadata["activation_representation"]
        self.assertEqual(representation["engine_sha256"], self.runtime["sources"]["src/activation_causality/engine.py"])
        self.assertEqual(representation["engine_dependency_sources"], engine_dependency_hashes(self.runtime))
        self.assertEqual(representation["versions"], self.runtime["versions"])
        self.assertEqual(representation["capture_convention"], "full-prefill; exact-input-token-ids; site as declared; float32 capture of model activations")
        self.assertEqual(request["activation_representation"], representation)
        self.assertTrue(audit["activation_representation_unchanged"])
        self.assertEqual(audit["activation_representation_before_fit"], representation)
        self.assertEqual(audit["activation_representation_after_fit"], representation)
        self.assertEqual(metadata["source_spec_path"], str((self.campaign / "specs/train.json").resolve()))
        self.assertIn("trainvariant0", metadata["training_layout_aliases"])
        path = Path(metadata["basis_path"])
        self.assertEqual(registry.file_sha256(path), metadata["basis_file_sha256"])
        self.assertEqual(registry.file_sha256(Path(metadata["basis_metadata_path"])), metadata["basis_metadata_sha256"])
        np.testing.assert_allclose(np.load(path).T @ np.load(path), np.eye(2), atol=1e-6)
        self.assertEqual(self.fit(), metadata)
        self.assertEqual(FakeEngine.constructions, 1)

    def test_four_vector_rank_four_metadata_does_not_claim_generalization(self):
        registry.atomic_json(self.campaign / "specs/train.json", source_spec(n_pairs=4))
        metadata = fit_basis(self.campaign, self.repo, "train", "layer10", "wall", 10, "resid_post", 4, "rank4")
        self.assertTrue(metadata["scientific_limits"]["rank_equals_training_position_count"])
        self.assertEqual(metadata["scientific_limits"]["n_distinct_source_pairs"], 4)
        self.assertFalse(metadata["scientific_limits"]["training_reconstruction_is_generalization_evidence"])
        self.assertIn("literal lexical substitutions", metadata["interpretation"])
        self.assertIn("Rank 4 with four", metadata["interpretation"])

    def test_index_is_required_even_for_fake_engine(self):
        (self.model / "model.safetensors.index.json").unlink()
        with self.assertRaisesRegex(ValueError, "required"):
            self.fit()
        self.assertEqual(FakeEngine.constructions, 0)

    def test_engine_source_and_runtime_versions_are_required(self):
        self.runtime["versions"]["mlx-lm"] = None
        with self.assertRaisesRegex(ValueError, "engine source hash"):
            self.fit()
        self.assertEqual(FakeEngine.constructions, 0)
        self.runtime["versions"]["mlx-lm"] = "cpu-test-double"
        self.runtime["sources"].clear()
        with self.assertRaisesRegex(ValueError, "engine source hash"):
            self.fit()

    def test_model_or_metadata_change_invalidates_existing_basis_request(self):
        self.fit()
        (self.model / "weights-1.safetensors").write_bytes(b"modified payload")
        with self.assertRaisesRegex(ValueError, "another source"):
            self.fit()
        self.assertEqual(FakeEngine.constructions, 1)
        (self.model / "tokenizer.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "another source"):
            self.fit()
        self.assertEqual(FakeEngine.constructions, 1)

    def test_weight_mutation_with_restored_size_mtime_blocks_publication(self):
        shard = self.model / "weights-1.safetensors"
        before = shard.stat()

        def mutate():
            shard.write_bytes(b"X" * before.st_size)
            os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
            FakeEngine.on_capture = None

        FakeEngine.on_capture = mutate
        with self.assertRaisesRegex(ValueError, "changed during fitting"):
            self.fit("unstable")
        self.assertFalse((self.campaign / "bases/unstable.npy").exists())
        self.assertFalse((self.campaign / "bases/unstable.json").exists())
        audit = registry.read_json(self.campaign / "reviews/basis_snapshot_unstable.json")
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["snapshots_equal"])
        self.assertNotEqual(audit["before_fit"]["files"][shard.name], audit["after_fit"]["files"][shard.name])
        self.assertEqual(audit["before_fit"]["file_metadata"][shard.name]["size_bytes"], before.st_size)

    def test_missing_shard_after_fit_records_unavailable_snapshot_and_no_basis(self):
        def remove():
            (self.model / "weights-2.safetensors").unlink()
            FakeEngine.on_capture = None

        FakeEngine.on_capture = remove
        with self.assertRaisesRegex(ValueError, "changed during fitting"):
            self.fit("missing")
        audit = registry.read_json(self.campaign / "reviews/basis_snapshot_missing.json")
        self.assertIsNone(audit["after_fit"])
        self.assertIn("inventory mismatch", audit["snapshot_error"])
        self.assertFalse((self.campaign / "bases/missing.npy").exists())

    def test_changed_source_or_retirement_during_capture_blocks_publication(self):
        for name in ("source", "retirement"):
            with self.subTest(name=name):
                registry.atomic_json(self.campaign / "specs/train.json", source_spec())

                def mutate():
                    if name == "source":
                        changed = source_spec()
                        changed["new_annotation"] = True
                        registry.atomic_json(self.campaign / "specs/train.json", changed)
                    else:
                        registry.atomic_json(self.campaign / "reviews/train_retired.json", {"retired": True})
                    FakeEngine.on_capture = None

                FakeEngine.on_capture = mutate
                with self.assertRaisesRegex(ValueError, "changed during fitting"):
                    self.fit(name)
                audit = registry.read_json(self.campaign / "reviews" / f"basis_snapshot_{name}.json")
                self.assertFalse(audit["source_and_campaign_unchanged"])
                self.assertFalse((self.campaign / "bases" / f"{name}.npy").exists())

    def test_source_job_id_must_bind_original_frozen_fields(self):
        source = source_spec()
        source["jobs"][0]["recipient_token_ids"] = [100]
        registry.atomic_json(self.campaign / "specs/train.json", source)
        with self.assertRaisesRegex(ValueError, "immutable source definition"):
            self.fit()
        self.assertEqual(FakeEngine.constructions, 0)

    def test_registered_source_geometry_must_match_fit_before_model_work(self):
        for key, value in (("layers", [0]), ("site", "mlp_out")):
            with self.subTest(key=key):
                source = source_spec()
                source["jobs"][0]["intervention"][key] = value
                job = source["jobs"][0]
                source["jobs"][0] = make_job("train", **{name: item for name, item in job.items()
                                                        if name not in {"job_id", "experiment_id"}})
                registry.atomic_json(self.campaign / "specs/train.json", source)
                with self.assertRaisesRegex(ValueError, "source job geometry"):
                    self.fit()
                self.assertFalse((self.campaign / "bases/basis.request.json").exists())
        self.assertEqual(FakeEngine.constructions, 0)

    def test_activation_representation_change_during_fit_blocks_publication(self):
        def mutate():
            self.runtime["sources"]["src/activation_causality/engine.py"] = "f" * 64
            FakeEngine.on_capture = None

        FakeEngine.on_capture = mutate
        with self.assertRaisesRegex(ValueError, "changed during fitting"):
            self.fit("engine_change")
        audit = registry.read_json(self.campaign / "reviews/basis_snapshot_engine_change.json")
        self.assertTrue(audit["snapshots_equal"])
        self.assertFalse(audit["activation_representation_unchanged"])
        self.assertFalse(audit["passed"])
        self.assertFalse((self.campaign / "bases/engine_change.npy").exists())

    def test_each_missing_dependency_blocks_fitting_before_model_construction(self):
        for index, dependency in enumerate(ENGINE_DEPENDENCY_PATHS):
            with self.subTest(dependency=dependency):
                original = self.runtime["sources"].pop(dependency)
                with self.assertRaisesRegex(ValueError, "engine dependency"):
                    self.fit(f"missing_dependency_{index}")
                self.assertEqual(FakeEngine.constructions, 0)
                self.runtime["sources"][dependency] = original

    def test_dependency_change_during_fit_blocks_basis_publication(self):
        for index, dependency in enumerate(ENGINE_DEPENDENCY_PATHS):
            with self.subTest(dependency=dependency):
                identifier = f"dependency_change_{index}"

                def mutate():
                    self.runtime["sources"][dependency] = "f" * 64
                    FakeEngine.on_capture = None

                FakeEngine.on_capture = mutate
                with self.assertRaisesRegex(ValueError, "changed during fitting"):
                    self.fit(identifier)
                audit = registry.read_json(self.campaign / "reviews" / f"basis_snapshot_{identifier}.json")
                self.assertFalse(audit["activation_representation_unchanged"])
                self.assertFalse(audit["passed"])
                self.assertFalse((self.campaign / "bases" / f"{identifier}.npy").exists())
                self.runtime["sources"][dependency] = "a" * 64

    def test_corrupted_snapshot_audit_is_not_reused(self):
        metadata = self.fit()
        Path(metadata["model_snapshot_audit_path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "snapshot audit"):
            self.fit()

    def test_actual_producer_metadata_loads_with_worker_snapshot_proof(self):
        metadata = self.fit()
        before = snapshot_identity(self.model)
        engine = FakeEngine(self.model)
        after = snapshot_identity(self.model)
        proof = make_loaded_model_proof(engine, before, after, self.runtime)
        spec = build_subspace_spec(source_spec("held"), metadata, metadata["basis_path"],
                                   metadata["basis_file_sha256"], "integration", natural=False)
        expected = np.load(metadata["basis_path"], allow_pickle=False)
        reads = []
        original_read = Path.read_bytes

        def record_read(path):
            reads.append(path)
            return original_read(path)

        with patch.object(Path, "read_bytes", record_read), patch.object(engine, "capture", side_effect=AssertionError("loader must not capture")):
            for job in spec["jobs"]:
                if job["intervention"]["operation"] != "subspace":
                    continue
                array, saved = load_verified_basis(job["intervention"], job, engine, proof)
                np.testing.assert_array_equal(array, expected)
                self.assertEqual(saved["model_snapshot_identity"], metadata["model_snapshot_identity"])
                self.assertEqual(saved["activation_representation"], proof.representation)
        self.assertFalse(any(path.suffix == ".safetensors" for path in reads))
        self.assertIn(Path(metadata["source_spec_path"]), reads)
        self.assertIn(Path(metadata["model_snapshot_audit_path"]), reads)

    def test_actual_producer_normalized_spans_and_derived_pair_lineage_load(self):
        source = source_spec()
        for index, job in enumerate(source["jobs"]):
            original_pair_id = job["pair_id"]
            job["pair_id"] += "::cut32"
            job["recipient_token_ids"] = [99, 98, index // 2 + 1]
            job["donor_token_ids"] = [99, 98, 97, index // 2 + 2]
            job["selection"] = {"scope": "explicit", "recipient_positions": [1], "donor_positions": [2]}
            job["metadata"] = {
                "source_pair_id": original_pair_id,
                "pair": {"pair_id": original_pair_id, "layout_id": job["layout_id"], "partition": "discovery",
                         "metadata": {"variant_layout_ids": ["nested_pair_variant"]}},
                "source_pair": {"pair_id": original_pair_id,
                                "layout_id": "nested_source_layout",
                                "metadata": {"variant_layout_ids": ["nested_source_variant"]}},
                "recipient_patch_positions": [1], "donor_patch_positions": [2],
            }
            source["jobs"][index] = make_job("train", **{key: value for key, value in job.items()
                                                        if key not in {"job_id", "experiment_id"}})
        registry.atomic_json(self.campaign / "specs/train.json", source)
        metadata = self.fit("explicit")
        self.assertIn("nested_pair_variant", metadata["training_layout_aliases"])
        self.assertIn("nested_source_variant", metadata["training_layout_aliases"])
        self.assertIn("nested_source_layout", metadata["training_layout_aliases"])
        original_selected = [job for job in source["jobs"] if job["seed"] == 42]
        self.assertNotEqual(registry.fingerprint(original_selected), metadata["selected_source_definitions_sha256"])
        self.assertTrue(all("::cut32" in pair_id for pair_id in metadata["training_pair_ids"]))
        before = snapshot_identity(self.model)
        engine = FakeEngine(self.model)
        proof = make_loaded_model_proof(engine, before, snapshot_identity(self.model), self.runtime)
        spec = build_subspace_spec(source_spec("held"), metadata, metadata["basis_path"],
                                   metadata["basis_file_sha256"], "explicit_integration")
        job = next(job for job in spec["jobs"] if job["condition"] == "subspace_forward_1")
        array, saved = load_verified_basis(job["intervention"], job, engine, proof)
        self.assertEqual(array.shape, (4, 2))
        self.assertEqual(saved["source_job_fingerprints"], metadata["source_job_fingerprints"])

    def test_changed_request_or_corrupted_basis_not_reused(self):
        metadata = self.fit()
        with self.assertRaises(ValueError):
            fit_basis(self.campaign, self.repo, "train", "layer10", "wall", 10, "resid_post", 1, "basis")
        Path(metadata["basis_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.fit()

    def test_confirmation_retirement_deadline_and_validation_block_work(self):
        spec = source_spec()
        spec["partition"] = "confirmation"
        registry.atomic_json(self.campaign / "specs/train.json", spec)
        with self.assertRaises(ValueError):
            self.fit()
        registry.atomic_json(self.campaign / "specs/train.json", source_spec())
        with patch("src.activation_causality.registry.seconds_remaining", return_value=0):
            with self.assertRaises(RuntimeError):
                self.fit()
        self.assertEqual(FakeEngine.constructions, 0)
        FakeEngine.validation_passes = False
        with self.assertRaises(RuntimeError):
            self.fit("validation_fail")
        self.assertEqual(FakeEngine.captures, 0)
        self.assertFalse((self.campaign / "bases/validation_fail.npy").exists())
        audit = registry.read_json(self.campaign / "reviews/basis_snapshot_validation_fail.json")
        self.assertTrue(audit["snapshots_equal"])
        self.assertFalse(audit["passed"])
        self.assertIn("validation failed", audit["fit_error"])
        with self.assertRaisesRegex(ValueError, "prior fit attempt"):
            self.fit("validation_fail")
        self.assertEqual(FakeEngine.constructions, 1)
        registry.atomic_json(self.campaign / "reviews/train_retired.json", {"retired": True})
        with self.assertRaises(ValueError):
            self.fit("retired")


class SubspaceSpecTests(unittest.TestCase):
    def metadata(self):
        return {"basis_id": "basis", "partition": "discovery", "layer": 10, "site": "resid_post", "rank": 2,
                "source_condition": "layer10", "family": "wall", "training_layout_ids": ["trainlayout0"],
                "training_layout_aliases": ["trainlayout0", "trainvariant0"], "basis_file_sha256": "a" * 64,
                "basis_metadata_path": "/tmp/basis.json", "basis_metadata_sha256": "b" * 64}

    def build(self, source=None, metadata=None):
        return build_subspace_spec(source or source_spec("held"), metadata or self.metadata(), "/tmp/basis.npy", "a" * 64, "subspace")

    def test_directions_target_identity_and_counterfactual_removal(self):
        spec = self.build()
        for job in spec["jobs"]:
            self.assertEqual(job["target_action"], "RIGHT")
            self.assertFalse(job["metadata"]["independent_rescue_evidence"])
            if job["condition"] == "subspace_removal":
                self.assertTrue(job["intervention"]["zero_donor"])
                self.assertEqual(job["metadata"]["paired_baseline_condition"], "donor_text")
            if job["condition"] == "restoration_equivalent":
                self.assertEqual(job["recipient_token_ids"], job["donor_token_ids"])
                self.assertTrue(job["metadata"]["restoration_equivalence_only"])
        reverse = [c for c in spec["comparisons"] if c["role"] == "conditional_reverse_interchange"]
        self.assertTrue(all(c["control"] == "donor_text" and c["predicted_direction"] == "decrease" for c in reverse))

    def test_rank_only_random_control_and_hash_paths(self):
        for job in self.build()["jobs"]:
            if job["condition"].startswith("random_subspace"):
                self.assertTrue(job["intervention"]["random_basis"])
                self.assertEqual(job["intervention"]["random_seed"], job["seed"])
                self.assertEqual(job["metadata"]["random_control_matching"], "rank_only_not_projected_delta_norm")
                self.assertNotIn("projected_norm_matched_random_basis", job["intervention"])
                self.assertNotIn("audit_projected_norm", job["intervention"])
            if job["intervention"]["operation"] == "subspace":
                self.assertEqual(job["intervention"]["basis_sha256"], "a" * 64)
                self.assertEqual(job["intervention"]["basis_metadata_sha256"], "b" * 64)

    def test_distinct_norm_matched_control_each_dose_and_joint_fixed_audit(self):
        spec = self.build()
        conditions = {job["condition"] for job in spec["jobs"]}
        self.assertEqual(len(spec["jobs"]), 4 * (6 + 4 * 2))
        for label in ("0p5", "1"):
            self.assertIn(f"random_subspace_{label}", conditions)
            self.assertIn(f"projected_norm_matched_random_subspace_{label}", conditions)
            forward = [job for job in spec["jobs"] if job["condition"] == f"subspace_forward_{label}"]
            control = [job for job in spec["jobs"] if job["condition"] == f"projected_norm_matched_random_subspace_{label}"]
            for treatment, random in zip(forward, control, strict=True):
                for job in (treatment, random):
                    self.assertTrue(job["intervention"]["audit_projected_norm"])
                    self.assertEqual(job["intervention"]["projected_norm_relative_tolerance"], 0.01)
                    self.assertTrue(job["metadata"]["projected_norm_audit"]["zero_reference_requires_exact_zero"])
                self.assertTrue(random["intervention"]["projected_norm_matched_random_basis"])
                self.assertEqual(random["intervention"]["random_seed"], random["seed"])
                self.assertEqual(random["intervention"]["rank"], 2)
                self.assertEqual(random["intervention"]["dose"], treatment["intervention"]["dose"])
                for flag in ("random_basis", "random_control", "zero_donor"):
                    self.assertNotIn(flag, random["intervention"])
                self.assertEqual(random["recipient_token_ids"], treatment["recipient_token_ids"])
                self.assertEqual(random["donor_token_ids"], treatment["donor_token_ids"])
                self.assertEqual(random["metadata"]["paired_baseline_condition"], "baseline")
            matching = [comparison for comparison in spec["comparisons"]
                        if comparison["control"] == f"projected_norm_matched_random_subspace_{label}"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["treatment"], f"subspace_forward_{label}")
            self.assertEqual(matching[0]["role"], "rank_and_projected_norm_matched_random_comparison")
            self.assertFalse(matching[0]["semantic_headline_eligible"])
        self.assertEqual(spec["protocol"]["projected_norm_relative_tolerance"], 0.01)
        self.assertTrue(spec["protocol"]["requires_projected_norm_matched_random_basis_support"])

    def test_training_alias_overlap_and_confirmation_rejected(self):
        source = source_spec("held")
        source["jobs"][0]["layout_aliases"].append("trainvariant0")
        with self.assertRaises(ValueError):
            self.build(source)
        source = source_spec("held")
        source["jobs"][0]["metadata"]["source_pair"] = {"metadata": {"variant_layout_ids": ["trainvariant0"]}}
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.build(source)
        for nested_name in ("pair", "source_pair"):
            source = source_spec("held")
            source["jobs"][0]["metadata"][nested_name] = {"layout_id": "trainvariant0"}
            with self.assertRaisesRegex(ValueError, "overlaps"):
                self.build(source)
        source = source_spec("held")
        source["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            self.build(source)
        metadata = self.metadata()
        metadata["basis_file_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            self.build(metadata=metadata)

    def test_inputs_not_changed_and_counts_match_selected_seeds(self):
        source, metadata = source_spec("held"), self.metadata()
        original = deepcopy((source, metadata))
        spec = self.build(source, metadata)
        self.assertEqual((source, metadata), original)
        self.assertEqual(len({job["job_id"] for job in spec["jobs"]}), len(spec["jobs"]))
        self.assertTrue(all(c["expected_n_pairs"] == 4 and c["expected_n_layouts"] == 2 for c in spec["comparisons"]))

    def test_unequal_context_explicit_positions_reverse_and_identity(self):
        source = source_spec("held")
        for job in source["jobs"]:
            job["recipient_token_ids"], job["donor_token_ids"] = [1, 2, 3], [4, 5, 6, 7]
            job["selection"] = {"scope": "explicit", "recipient_positions": [1, 2], "donor_positions": [2, 3]}
            job["recipient_positions"], job["donor_positions"] = [1, 2], [2, 3]
            job["metadata"] = {"recipient_patch_positions": [1, 2], "donor_patch_positions": [2, 3]}
        for job in self.build(source)["jobs"]:
            reverse = job["condition"].startswith("subspace_reverse") or job["condition"] in {"donor_text", "subspace_removal"}
            identity = job["condition"] in {"subspace_identity", "restoration_equivalent"}
            recipient_positions = [2, 3] if reverse else [1, 2]
            donor_positions = [1, 2] if reverse or identity else [2, 3]
            self.assertEqual(job["recipient_positions"], recipient_positions)
            self.assertEqual(job["donor_positions"], donor_positions)
            self.assertEqual(job["selection"]["recipient_positions"], recipient_positions)
            self.assertEqual(job["selection"]["donor_positions"], donor_positions)
            self.assertEqual(job["metadata"]["recipient_patch_positions"], recipient_positions)
            self.assertEqual(job["metadata"]["donor_patch_positions"], donor_positions)
        source["jobs"][0]["selection"]["donor_positions"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.build(source)

    def test_malformed_source_identity_or_one_sided_tokens_rejected(self):
        source = source_spec("held")
        source["jobs"][0]["experiment_id"] = "other"
        with self.assertRaisesRegex(ValueError, "experiment"):
            self.build(source)
        source = source_spec("held")
        source["jobs"][0]["donor_token_ids"] = None
        with self.assertRaisesRegex(ValueError, "Frozen"):
            self.build(source)


if __name__ == "__main__":
    unittest.main()
