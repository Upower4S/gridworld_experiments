"""Basis freeze/readiness checks with synthetic immutable CPU artifacts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality import basis_runtime, registry
from src.activation_causality.confirmation_basis import validate_confirmation_basis
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
from src.activation_causality.localization import _array_fingerprint
from src.activation_causality.model_identity import snapshot_identity


class ConfirmationBasisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.safetensors").write_bytes(b"CPU snapshot fixture, never loaded")
        registry.atomic_json(self.model / "config.json", {"hidden_size": 8})
        registry.atomic_json(self.model / "tokenizer.json", {})
        registry.atomic_json(self.model / "model.safetensors.index.json", {"weight_map": {"tensor": "weights.safetensors"}})
        self.snapshot = snapshot_identity(self.model)
        runtime = {"sources": {"src/activation_causality/engine.py": "a" * 64,
                               **{name: "b" * 64 for name in ENGINE_DEPENDENCY_PATHS}},
                   "versions": {"mlx": "cpu-fixture", "mlx-lm": "cpu-fixture", "numpy": np.__version__}}
        self.representation = basis_runtime.activation_representation(runtime)
        jobs = []
        for index in range(4):
            job = {"experiment_id": "training", "condition": "factspan_layer0", "partition": "discovery",
                   "family": "wall", "pair_id": f"pair{index}", "layout_id": f"layout{index}",
                   "layout_aliases": [f"alias{index}"], "intervention": {"layers": [0], "site": "resid_post"}}
            job["job_id"] = registry.fingerprint(job)[:24]
            jobs.append(job)
        self.source = {"experiment_id": "training", "partition": "discovery", "jobs": jobs}
        self.source_path = self.root / "specs/training.json"
        registry.atomic_json(self.source_path, self.source)
        self.basis_path = self.root / "bases/basis.npy"
        self.basis_path.parent.mkdir()
        self.array = np.eye(8, 1, dtype=np.float32)
        np.save(self.basis_path, self.array)
        self.metadata_path = self.basis_path.with_suffix(".json")
        self.audit_path = self.root / "reviews/basis_fit.json"
        self.metadata = {"format": "registered_task_basis_v1", "model_provenance_version": 2,
                         "partition": "discovery", "training_only": True, "basis_id": "basis",
                         "model_snapshot_verified_before_and_after_fit": True,
                         "family": "wall", "layer": 0, "site": "resid_post", "rank": 1, "hidden_dim": 8,
                         "basis_path": str(self.basis_path), "basis_file_sha256": registry.file_sha256(self.basis_path),
                         "basis_array_sha256": _array_fingerprint(self.array),
                         "source_condition": "factspan_layer0", "source_experiment_id": "training",
                         "source_spec_path": str(self.source_path), "source_spec_sha256": registry.file_sha256(self.source_path),
                         "source_job_ids": [job["job_id"] for job in jobs],
                         "source_job_fingerprints": {job["job_id"]: registry.fingerprint(job) for job in jobs},
                         "selected_source_definitions_sha256": registry.fingerprint(jobs),
                         "training_layout_ids": [f"layout{i}" for i in range(4)],
                         "training_layout_aliases": [f"layout{i}" for i in range(4)] + [f"alias{i}" for i in range(4)],
                         "training_pair_ids": [f"pair{i}" for i in range(4)],
                         "model_snapshot_identity": self.snapshot, "model_snapshot_identity_sha256": self.snapshot["identity_sha256"],
                         "model_files": self.snapshot["files"], "model_path": self.snapshot["path"],
                         "activation_representation": self.representation, "model_snapshot_audit_path": str(self.audit_path)}
        self.refresh_chain()

    def refresh_chain(self):
        fields = ("source_experiment_id", "source_spec_sha256", "source_spec_path", "source_job_fingerprints",
                  "selected_source_definitions_sha256", "family", "layer", "site", "rank", "model_path",
                  "model_files", "model_provenance_version", "model_snapshot_identity", "model_snapshot_identity_sha256", "activation_representation")
        self.request = {name: self.metadata[name] for name in fields}
        self.request.update(condition=self.metadata["source_condition"], selected_job_ids=self.metadata["source_job_ids"])
        registry.atomic_json(self.basis_path.with_suffix(".request.json"), self.request)
        self.metadata["request_fingerprint"] = registry.fingerprint(self.request)
        self.audit = {"format": "basis_fit_model_snapshot_audit_v1", "basis_id": "basis", "passed": True,
                      "request_fingerprint": self.metadata["request_fingerprint"], "snapshots_equal": True,
                      "source_and_campaign_unchanged": True, "snapshot_error": None, "fit_error": None,
                      "before_fit": self.snapshot, "after_fit": self.snapshot,
                      "activation_representation_before_fit": self.representation,
                      "activation_representation_after_fit": self.representation,
                      "activation_representation_unchanged": True, "activation_representation_error": None}
        self.refresh_audit()

    def refresh_audit(self):
        registry.atomic_json(self.audit_path, self.audit)
        self.metadata["model_snapshot_audit_sha256"] = registry.file_sha256(self.audit_path)
        self.refresh_metadata()

    def refresh_metadata(self):
        registry.atomic_json(self.metadata_path, self.metadata)
        self.augmented = {**deepcopy(self.metadata), "basis_metadata_path": str(self.metadata_path),
                          "basis_metadata_sha256": registry.file_sha256(self.metadata_path)}
        self.config = {name: self.metadata[name] for name in ("family", "rank", "layer", "site", "basis_path")}
        self.config.update(basis_sha256=self.metadata["basis_file_sha256"], basis_metadata_path=str(self.metadata_path),
                           basis_metadata_sha256=self.augmented["basis_metadata_sha256"],
                           basis_metadata_fingerprint=registry.fingerprint(self.augmented),
                           source_experiment_id="separately_evaluated_native_candidate", source_spec_sha256="c" * 64)

    def validate(self):
        return validate_confirmation_basis(self.config, self.augmented,
                                           model_snapshot_identity=self.snapshot, activation_representation=self.representation)

    def test_complete_chain_returns_five_raw_hashes_and_no_model_proof(self):
        actual_read = Path.read_bytes
        def reject_model_reads(path):
            self.assertFalse(path.is_relative_to(self.model), path)
            return actual_read(path)
        with patch.object(Path, "read_bytes", reject_model_reads):
            verified, manifest = self.validate()
        self.assertEqual(verified, self.augmented)
        self.assertEqual(len(manifest["artifact_sha256"]), 5)
        self.assertEqual(len(manifest["parsed_semantic_fingerprints"]), 4)
        self.assertFalse(manifest["model_materialization_claim"])
        self.assertFalse(manifest["model_weights_read"])
        self.assertTrue(manifest["all_artifacts_observed_stable"])
        for path, digest in manifest["artifact_sha256"].items():
            self.assertEqual(registry.file_sha256(Path(path)), digest)

    def test_malformed_model_directory_or_weight_path_rejected_before_read(self):
        actual_read = Path.read_bytes
        reads = []
        def track(path):
            reads.append(path)
            return actual_read(path)
        self.config["basis_metadata_path"] = str(self.model / "config.json")
        with patch.object(Path, "read_bytes", side_effect=track, autospec=True), self.assertRaisesRegex(ValueError, "may not read model"):
            self.validate()
        self.assertEqual(reads, [])
        outside = self.root / "wrong.safetensors"
        outside.write_bytes(b"must not read this payload")
        self.config["basis_metadata_path"] = str(outside)
        with patch.object(Path, "read_bytes", side_effect=track, autospec=True), self.assertRaisesRegex(ValueError, "suffix"):
            self.validate()
        self.assertEqual(reads, [])

    def test_symlink_alias_cannot_escape_global_artifact_or_retirement_guard(self):
        alias = self.root / "metadata_alias.json"
        alias.symlink_to(self.metadata_path)
        self.config["basis_metadata_path"] = str(alias)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("alias must fail before payload read")), \
                self.assertRaisesRegex(ValueError, "canonical paths"):
            self.validate()

    def test_embedded_metadata_cannot_add_change_or_drop_fields(self):
        for change in (lambda m: m.update(extra="unbound"), lambda m: m.pop("training_only"),
                       lambda m: m.update(rank=1.0), lambda m: m.update(model_provenance_version=True)):
            old = deepcopy(self.augmented)
            change(self.augmented)
            self.config["basis_metadata_fingerprint"] = registry.fingerprint(self.augmented)
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                self.validate()
            self.augmented = old
        self.refresh_metadata()
        self.metadata["basis_metadata_path"] = str(self.metadata_path)
        self.refresh_metadata()
        with self.assertRaisesRegex(ValueError, "self-referential"):
            self.validate()

    def test_configuration_geometry_and_file_bindings_are_exact(self):
        for key, value in (("rank", True), ("rank", 4), ("layer", 0.0), ("family", "door_status"),
                           ("basis_sha256", "d" * 64), ("basis_metadata_sha256", "d" * 64),
                           ("basis_metadata_fingerprint", "d" * 64), ("basis_path", str(self.root / "other.npy"))):
            old = self.config[key]
            self.config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.config[key] = old
        self.config["intervention"] = {"layers": [0], "site": "resid_post", "operation": "subspace", "basis_path": "/wrong"}
        with self.assertRaisesRegex(ValueError, "selected intervention"):
            self.validate()

    def test_request_numeric_type_drift_rejects_even_when_python_equality_agrees(self):
        self.request["rank"] = 1.0
        registry.atomic_json(self.basis_path.with_suffix(".request.json"), self.request)
        self.metadata["request_fingerprint"] = registry.fingerprint(self.request)
        self.audit["request_fingerprint"] = self.metadata["request_fingerprint"]
        self.refresh_audit()
        with self.assertRaisesRegex(ValueError, "canonical source/model/geometry"):
            self.validate()

    def test_model_snapshot_and_representation_cannot_be_substituted(self):
        changed = deepcopy(self.representation)
        changed["engine_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "representation differs"):
            validate_confirmation_basis(self.config, self.augmented, model_snapshot_identity=self.snapshot, activation_representation=changed)
        changed = deepcopy(self.snapshot)
        changed["path"] = "/different/local/model"
        changed["identity_sha256"] = registry.fingerprint({k: v for k, v in changed.items() if k != "identity_sha256"})
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            validate_confirmation_basis(self.config, self.augmented, model_snapshot_identity=changed, activation_representation=self.representation)

    def test_fit_failure_and_training_alias_omission_are_rejected(self):
        self.audit["passed"] = False
        self.refresh_audit()
        with self.assertRaisesRegex(ValueError, "snapshot audit"):
            self.validate()
        self.audit["passed"] = True
        self.metadata["training_layout_aliases"].pop()
        self.refresh_audit()
        with self.assertRaisesRegex(ValueError, "training membership"):
            self.validate()

    def test_source_job_geometry_numeric_drift_is_rejected(self):
        self.source["jobs"][0]["intervention"]["layers"] = [0.0]
        for row in self.source["jobs"]:
            row["job_id"] = registry.fingerprint({k: v for k, v in row.items() if k != "job_id"})[:24]
        registry.atomic_json(self.source_path, self.source)
        self.metadata.update(source_spec_sha256=registry.file_sha256(self.source_path),
                             source_job_ids=[row["job_id"] for row in self.source["jobs"]],
                             source_job_fingerprints={row["job_id"]: registry.fingerprint(row) for row in self.source["jobs"]},
                             selected_source_definitions_sha256=registry.fingerprint(self.source["jobs"]))
        self.refresh_chain()
        with self.assertRaisesRegex(ValueError, "type/value drift"):
            self.validate()

    def test_array_dtype_content_geometry_and_orthonormality(self):
        for array in (np.eye(8, 1, dtype=np.float64), np.eye(7, 1, dtype=np.float32),
                      np.full((8, 1), np.nan, dtype=np.float32), np.ones((8, 1), dtype=np.float32)):
            np.save(self.basis_path, array)
            self.metadata.update(basis_file_sha256=registry.file_sha256(self.basis_path), basis_array_sha256=_array_fingerprint(array))
            self.refresh_metadata()
            with self.subTest(dtype=array.dtype), self.assertRaises(ValueError):
                self.validate()

    def test_rank_four_orthonormal_basis_and_four_training_pairs(self):
        array = np.eye(8, 4, dtype=np.float32)
        np.save(self.basis_path, array)
        self.metadata.update(rank=4, basis_file_sha256=registry.file_sha256(self.basis_path), basis_array_sha256=_array_fingerprint(array))
        self.refresh_chain()
        _, manifest = self.validate()
        self.assertEqual(manifest["rank"], 4)
        self.assertEqual(manifest["orthonormality_max_abs_error"], 0)

    def test_source_retirement_and_global_mutation_after_private_checks(self):
        retirement = self.root / "reviews/training_retired.json"
        registry.atomic_json(retirement, {"retired": True})
        with self.assertRaisesRegex(ValueError, "retired"):
            self.validate()
        retirement.unlink()
        actual = basis_runtime._verify_source
        def mutate(metadata):
            actual(metadata)
            self.source_path.write_bytes(self.source_path.read_bytes() + b" ")
        with patch.object(basis_runtime, "_verify_source", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed during validation"):
            self.validate()

    def test_strict_json_rejects_duplicate_and_nonfinite_payloads(self):
        raw = self.metadata_path.read_bytes()
        duplicate = b'{"rank":1,' + raw.lstrip()[1:]
        self.metadata_path.write_bytes(duplicate)
        self.config["basis_metadata_sha256"] = registry.file_sha256(self.metadata_path)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.validate()
        self.metadata_path.write_bytes(b'{"rank":1e999}')
        self.config["basis_metadata_sha256"] = registry.file_sha256(self.metadata_path)
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
