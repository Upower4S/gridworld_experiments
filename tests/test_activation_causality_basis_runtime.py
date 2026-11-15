"""CPU-only tests of basis/loaded-model provenance and immutable byte loading."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality import registry
from src.activation_causality.basis_runtime import (
    activation_representation, load_verified_basis, make_loaded_model_proof,
)
from src.activation_causality.localization import _array_fingerprint
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
from src.activation_causality.model_identity import snapshot_identity


class BasisRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.safetensors").write_bytes(b"tiny indexed payload; no model inference")
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")
        registry.atomic_json(self.model / "model.safetensors.index.json",
                             {"weight_map": {"one.tensor": "weights.safetensors"}})
        self.identity = snapshot_identity(self.model)
        self.runtime = {"sources": {"src/activation_causality/engine.py": "e" * 64},
                        "versions": {"mlx": "1", "mlx-lm": "2", "numpy": "3"}}
        self.runtime["sources"].update({name: "a" * 64 for name in ENGINE_DEPENDENCY_PATHS})
        self.engine = SimpleNamespace(model_path=self.model, head_mode="full",
                                      metadata={"loaded": True, "hidden_dim": 3})
        self.proof = make_loaded_model_proof(self.engine, self.identity, self.identity, self.runtime)
        self.source_job = {"job_id": "source-job", "experiment_id": "source", "partition": "discovery",
                           "pair_id": "train-pair", "layout_id": "train", "layout_aliases": ["train-variant"],
                           "condition": "factspan_layer0", "family": "wall", "seed": 42,
                           "intervention": {"layers": [0], "site": "resid_post"}}
        self.source_job["job_id"] = registry.fingerprint({k: v for k, v in self.source_job.items() if k != "job_id"})[:24]
        self.source = {"experiment_id": "source", "partition": "discovery", "jobs": [self.source_job]}
        self.source_path = self.root / "specs/source.json"
        registry.atomic_json(self.source_path, self.source)
        self.basis_path = self.root / "bases/basis.npy"
        self.basis_path.parent.mkdir()
        self.array = np.eye(3, 1, dtype=np.float32)
        np.save(self.basis_path, self.array)
        self.audit_path = self.root / "reviews/basis_snapshot.json"
        self.audit = {"format": "basis_fit_model_snapshot_audit_v1", "basis_id": "basis",
                      "request_fingerprint": "a" * 64, "before_fit": self.identity, "after_fit": self.identity,
                      "snapshots_equal": True, "snapshot_error": None, "source_and_campaign_unchanged": True,
                      "activation_representation_before_fit": activation_representation(self.runtime),
                      "activation_representation_after_fit": activation_representation(self.runtime),
                      "activation_representation_unchanged": True, "activation_representation_error": None,
                      "fit_error": None, "passed": True}
        registry.atomic_json(self.audit_path, self.audit)
        self.meta_path = self.basis_path.with_suffix(".json")
        self.meta = {
            "format": "registered_task_basis_v1", "model_provenance_version": 2, "partition": "discovery",
            "training_only": True, "basis_id": "basis", "request_fingerprint": "a" * 64,
            "basis_path": str(self.basis_path), "basis_file_sha256": registry.file_sha256(self.basis_path),
            "basis_array_sha256": _array_fingerprint(self.array),
            "model_snapshot_verified_before_and_after_fit": True,
            "model_snapshot_identity": self.identity, "model_snapshot_identity_sha256": self.identity["identity_sha256"],
            "model_files": self.identity["files"], "model_path": self.identity["path"],
            "model_snapshot_audit_path": str(self.audit_path),
            "model_snapshot_audit_sha256": registry.file_sha256(self.audit_path),
            "activation_representation": activation_representation(self.runtime),
            "family": "wall", "layer": 0, "site": "resid_post", "rank": 1, "hidden_dim": 3,
            "training_layout_ids": ["train"], "training_layout_aliases": ["train", "train-variant"],
            "training_pair_ids": ["train-pair"], "source_experiment_id": "source",
            "source_condition": "factspan_layer0", "source_spec_path": str(self.source_path),
            "source_spec_sha256": registry.file_sha256(self.source_path), "source_job_ids": [self.source_job["job_id"]],
            "source_job_fingerprints": {self.source_job["job_id"]: registry.fingerprint(self.source_job)},
            "selected_source_definitions_sha256": registry.fingerprint([self.source_job]),
        }
        self.intervention = {"operation": "subspace", "site": "resid_post", "layers": [0], "rank": 1,
                             "basis_path": str(self.basis_path), "basis_sha256": self.meta["basis_file_sha256"],
                             "basis_metadata_path": str(self.meta_path)}
        self.job = {"family": "wall", "layout_id": "evaluation", "layout_aliases": [], "metadata": {}}
        self.write_request()
        self.write_meta()

    def write_request(self):
        fields = ("source_experiment_id", "source_spec_sha256", "source_spec_path", "source_job_fingerprints",
                  "selected_source_definitions_sha256", "family", "layer", "site", "rank", "model_path",
                  "model_files", "model_provenance_version", "model_snapshot_identity", "model_snapshot_identity_sha256",
                  "activation_representation")
        request = {name: self.meta[name] for name in fields}
        request.update(condition=self.meta["source_condition"], selected_job_ids=self.meta["source_job_ids"])
        registry.atomic_json(self.basis_path.with_suffix(".request.json"), request)
        self.meta["request_fingerprint"] = registry.fingerprint(request)
        self.audit["request_fingerprint"] = self.meta["request_fingerprint"]
        registry.atomic_json(self.audit_path, self.audit)
        self.meta["model_snapshot_audit_sha256"] = registry.file_sha256(self.audit_path)

    def write_meta(self):
        registry.atomic_json(self.meta_path, self.meta)
        self.intervention["basis_metadata_sha256"] = registry.file_sha256(self.meta_path)

    def load(self):
        return load_verified_basis(self.intervention, self.job, self.engine, self.proof)

    def rewrite_audit(self):
        registry.atomic_json(self.audit_path, self.audit)
        self.meta["model_snapshot_audit_sha256"] = registry.file_sha256(self.audit_path)
        self.write_meta()

    def rewrite_array(self, array, *, refresh_array_digest=True):
        np.save(self.basis_path, array)
        self.meta["basis_file_sha256"] = registry.file_sha256(self.basis_path)
        self.intervention["basis_sha256"] = self.meta["basis_file_sha256"]
        if refresh_array_digest:
            self.meta["basis_array_sha256"] = _array_fingerprint(array)
        self.write_meta()

    def test_complete_verified_load_does_not_read_model_shards(self):
        original = Path.read_bytes
        reads = []

        def record(path):
            reads.append(path)
            return original(path)

        with patch.object(Path, "read_bytes", record):
            array, metadata = self.load()
        np.testing.assert_array_equal(array, self.array)
        self.assertEqual(metadata, self.meta)
        self.assertFalse(any(path.suffix == ".safetensors" for path in reads))

    def test_factory_requires_materialized_engine_and_same_full_snapshot(self):
        self.engine.metadata["loaded"] = False
        with self.assertRaisesRegex(ValueError, "materialized"):
            make_loaded_model_proof(self.engine, self.identity, self.identity, self.runtime)
        self.engine.metadata["loaded"] = True
        (self.model / "weights.safetensors").write_bytes(b"different weights")
        after = snapshot_identity(self.model)
        with self.assertRaisesRegex(ValueError, "changed across"):
            make_loaded_model_proof(self.engine, self.identity, after, self.runtime)

    def test_proof_is_engine_instance_bound_and_required(self):
        with self.assertRaisesRegex(ValueError, "engine instance"):
            load_verified_basis(self.intervention, self.job, deepcopy(self.engine), self.proof)
        with self.assertRaisesRegex(ValueError, "engine instance"):
            load_verified_basis(self.intervention, self.job, self.engine, None)

    def test_tampered_snapshot_self_digest_is_rejected(self):
        self.meta["model_snapshot_identity"] = deepcopy(self.identity)
        self.meta["model_snapshot_identity"]["files"]["weights.safetensors"] = "b" * 64
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "self-digest"):
            self.load()

    def test_legacy_basis_cannot_authorize_new_execution(self):
        self.meta.pop("model_provenance_version")
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "version-2"):
            self.load()

    def test_engine_and_capture_runtime_change_rejected_but_reporting_change_ignored(self):
        runtime = deepcopy(self.runtime)
        runtime["sources"]["src/activation_causality/analysis.py"] = "a" * 64
        self.assertEqual(activation_representation(runtime), self.proof.representation)
        for key in ("engine_sha256", "capture_convention", "versions"):
            with self.subTest(key=key):
                old = self.meta["activation_representation"][key]
                self.meta["activation_representation"][key] = "changed"
                self.write_meta()
                with self.assertRaisesRegex(ValueError, "capture convention"):
                    self.load()
                self.meta["activation_representation"][key] = old

    def test_basis_compatibility_pins_each_engine_dependency(self):
        for dependency in ENGINE_DEPENDENCY_PATHS:
            with self.subTest(dependency=dependency):
                changed = deepcopy(self.runtime)
                changed["sources"][dependency] = "f" * 64
                proof = make_loaded_model_proof(self.engine, self.identity, self.identity, changed)
                self.assertNotEqual(proof.representation, self.proof.representation)
                with self.assertRaisesRegex(ValueError, "capture convention"):
                    load_verified_basis(self.intervention, self.job, self.engine, proof)
                changed["sources"].pop(dependency)
                with self.assertRaisesRegex(ValueError, "engine dependency"):
                    activation_representation(changed)

    def test_fit_flag_cannot_replace_connected_successful_audit(self):
        changes = {"passed": False, "source_and_campaign_unchanged": False, "snapshots_equal": False,
                   "request_fingerprint": "c" * 64, "basis_id": "unrelated", "before_fit": {},
                   "after_fit": {}, "fit_error": "failed", "snapshot_error": "changed",
                   "activation_representation_after_fit": {}, "activation_representation_unchanged": False,
                   "activation_representation_error": "changed"}
        for key, value in changes.items():
            with self.subTest(key=key):
                old = self.audit[key]
                self.audit[key] = value
                self.rewrite_audit()
                with self.assertRaisesRegex(ValueError, "snapshot audit"):
                    self.load()
                self.audit[key] = old

    def test_metadata_and_audit_raw_byte_mutations_reject(self):
        self.meta_path.write_bytes(self.meta_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "metadata file fingerprint"):
            self.load()
        self.write_meta()
        self.audit_path.write_bytes(self.audit_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "audit file fingerprint"):
            self.load()

    def test_source_mutation_retirement_and_job_lineage_rejected(self):
        self.source_path.write_bytes(self.source_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source specification file fingerprint"):
            self.load()
        registry.atomic_json(self.source_path, self.source)
        registry.atomic_json(self.root / "reviews/source_retired.json", {"retired": True})
        with self.assertRaisesRegex(ValueError, "nonretired"):
            self.load()
        (self.root / "reviews/source_retired.json").unlink()
        self.source["jobs"][0]["seed"] = 43
        registry.atomic_json(self.source_path, self.source)
        self.meta["source_spec_sha256"] = registry.file_sha256(self.source_path)
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "request|source job fingerprints"):
            self.load()

    def test_nested_alias_overlap_and_family_are_rejected(self):
        for name in ("pair", "source_pair"):
            self.job["metadata"] = {name: {"metadata": {"variant_layout_ids": ["train-variant"]}}}
            with self.assertRaisesRegex(ValueError, "nested aliases"):
                self.load()
            self.job["metadata"] = {name: {"layout_id": "train", "metadata": {}}}
            with self.assertRaisesRegex(ValueError, "nested aliases"):
                self.load()
        self.job["metadata"] = {}
        self.job["family"] = "door_status"
        with self.assertRaisesRegex(ValueError, "family"):
            self.load()

    def test_exact_position_normalization_and_definition_digest(self):
        self.source_job["selection"] = {"scope": "explicit", "recipient_positions": [3], "donor_positions": [4]}
        self.source_job["job_id"] = registry.fingerprint({k: v for k, v in self.source_job.items() if k != "job_id"})[:24]
        registry.atomic_json(self.source_path, self.source)
        self.meta["source_spec_sha256"] = registry.file_sha256(self.source_path)
        self.meta["source_job_ids"] = [self.source_job["job_id"]]
        self.meta["source_job_fingerprints"] = {self.source_job["job_id"]: registry.fingerprint(self.source_job)}
        normalized = {**deepcopy(self.source_job), "recipient_positions": [3], "donor_positions": [4]}
        self.meta["selected_source_definitions_sha256"] = registry.fingerprint([normalized])
        self.write_request()
        self.write_meta()
        self.load()
        self.meta["selected_source_definitions_sha256"] = registry.fingerprint([self.source_job])
        self.write_request()
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "normalized extraction-source"):
            self.load()

    def test_original_job_identity_and_immutable_request_are_independently_checked(self):
        self.source_job["seed"] = 43
        registry.atomic_json(self.source_path, self.source)
        self.meta["source_spec_sha256"] = registry.file_sha256(self.source_path)
        self.meta["source_job_fingerprints"] = {self.source_job["job_id"]: registry.fingerprint(self.source_job)}
        self.meta["selected_source_definitions_sha256"] = registry.fingerprint([self.source_job])
        self.write_request()
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "source job fingerprints"):
            self.load()
        self.basis_path.with_suffix(".request.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "request fingerprint"):
            self.load()

    def test_hidden_rank_site_and_membership_must_agree(self):
        for key, value in (("hidden_dim", 4), ("rank", 2), ("site", "mlp_out"),
                           ("training_layout_aliases", ["train"]), ("training_pair_ids", ["another"])):
            with self.subTest(key=key):
                old = self.meta[key]
                self.meta[key] = value
                self.write_meta()
                with self.assertRaises(ValueError):
                    self.load()
                self.meta[key] = old

    def test_array_file_content_shape_and_orthonormality_checks(self):
        self.basis_path.write_bytes(b"not original bytes")
        with self.assertRaisesRegex(ValueError, "basis file fingerprint"):
            self.load()
        other = np.eye(3, 1, dtype=np.float32)[::-1].copy()
        self.rewrite_array(other, refresh_array_digest=False)
        with self.assertRaisesRegex(ValueError, "array format"):
            self.load()
        self.rewrite_array(np.ones((3, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.load()
        self.rewrite_array(np.ones((2, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "array format"):
            self.load()

    def test_path_swap_after_hash_reads_cannot_change_loaded_array(self):
        original = Path.read_bytes
        replacement = io.BytesIO()
        np.save(replacement, self.array[::-1].copy())

        def swapping_read(path):
            raw = original(path)
            if path == self.basis_path:
                path.write_bytes(replacement.getvalue())
            return raw

        with patch.object(Path, "read_bytes", swapping_read):
            array, _ = self.load()
        np.testing.assert_array_equal(array, self.array)
        self.assertNotEqual(hashlib.sha256(original(self.basis_path)).hexdigest(), self.intervention["basis_sha256"])


if __name__ == "__main__":
    unittest.main()
