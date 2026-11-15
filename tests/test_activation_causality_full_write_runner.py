"""Full-write compiler/runner gates with tiny CPU arrays and fake endpoints."""
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality import campaign as worker
from src.activation_causality import registry
from src.activation_causality.basis_runtime import make_loaded_model_proof
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS, RecipientStateMismatch
from src.activation_causality.model_identity import snapshot_identity
from src.activation_causality.subspace_audit import array_fingerprint


class FakeEngine:
    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        self.metadata = {"loaded": False, "hidden_dim": 8}
        self._head_mode = "full"
        self.captures, self.prefills, self.endpoints = [], [], []
        self.ratio = 1.0
        self.mismatch = False
        self.invalid_capture = False
        self.load_calls = 0

    @property
    def head_mode(self):
        return self._head_mode

    def _ensure_loaded(self):
        self.metadata["loaded"] = True
        self.load_calls += 1

    def validate(self):
        return {"passed": True, "model": {"head_mode": "full"}}

    def encode(self, text):
        raise AssertionError("authoritative IDs must bypass encode")

    def values(self, ids, positions):
        result = np.array([[ids[p] * (h + 1) / 8 for h in range(8)] for p in positions], dtype=np.float32)
        if self.invalid_capture:
            result[0, 0] = np.inf
        return result

    def capture(self, ids, layers, positions, site):
        self.captures.append((ids[:], positions[:]))
        return {layer: self.values(ids, positions) for layer in layers}

    def prefill(self, ids, patches=()):
        self.prefills.append((ids, patches))
        logs = []
        for item in patches:
            original = self.values(ids, item["positions"])
            digest = array_fingerprint(original)
            geometry = {key: item[key] for key in ("layer", "site", "positions", "operation", "dose")}
            if self.mismatch:
                evidence = {**geometry, "rank": None, "write_applied": False,
                            "expected_recipient_float32_sha256": item["expected_recipient_float32_sha256"],
                            "recipient_float32_sha256": "f" * 64}
                raise RecipientStateMismatch(evidence)
            delta = item["donor"] - original
            logs.append({**geometry, "rank": None, "recipient_float32_sha256": digest,
                         "before_l2": np.linalg.norm(original, axis=1).tolist(),
                         "delta_l2": (np.linalg.norm(delta, axis=1) * self.ratio).tolist()})
        return SimpleNamespace(patch_log=logs)

    def forced(self, state):
        self.endpoints.append("forced")
        return {"q": [.1, .2, .3, .4]}

    def generate(self, state, **kwargs):
        self.endpoints.append("native")
        return {"native_action": "RIGHT", "generation_status": "complete", "generated_tokens": 1}


class FullWriteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.root = self.repo / "campaign"
        self.model = self.repo / "model"
        self.model.mkdir()
        (self.model / "weights.safetensors").write_bytes(b"CPU fixture weights, never loaded")
        for name in ("config.json", "tokenizer.json"):
            registry.atomic_json(self.model / name, {})
        registry.atomic_json(self.model / "model.safetensors.index.json", {"weight_map": {"tensor": "weights.safetensors"}})
        registry.atomic_json(self.root / "campaign.json", {"model_path": str(self.model)})
        self.engine = FakeEngine(self.model)
        names = ("engine", "campaign", "basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit", "full_write_audit")
        self.runtime = {"versions": {"numpy": np.__version__, "mlx": "cpu-fixture", "mlx-lm": "cpu-fixture"}, "sources": {f"src/activation_causality/{name}.py": registry.fingerprint(name) for name in names}}
        self.runtime["sources"].update({name: "a" * 64 for name in ENGINE_DEPENDENCY_PATHS})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("builtins.print"))
        self.stack.enter_context(patch.object(worker, "runtime_fingerprint", side_effect=lambda _: deepcopy(self.runtime)))
        self.stack.enter_context(patch.object(worker, "seconds_remaining", return_value=3600))
        self.stack.enter_context(patch.object(worker, "worker_lock", return_value=nullcontext()))
        self.stack.enter_context(patch("src.activation_causality.selection.validate_confirmation_ready"))
        self.constructor = self.stack.enter_context(patch("src.activation_causality.engine.MLXInterventionEngine", return_value=self.engine))

    def job(self, eid="test", *, condition="natural", random=False, recorded_only=False, baseline=False):
        pair = {"pair_id": "fresh-pair", "layout_id": "fresh-layout", "partition": "confirmation"}
        source = {"layout_id": pair["layout_id"], "pair_id": pair["pair_id"], "family": "wall", "partition": "discovery",
                  "condition": condition, "seed": 42, "target_action": "RIGHT", "original_action": "LEFT",
                  "recipient_token_ids": [1, 2, 3, 4], "donor_token_ids": [1, 7, 3, 9],
                  "recipient_text": "unused recipient", "donor_text": "unused donor", "natural": True,
                  "max_new_tokens": 32, "temperature": .7,
                  "selection": {"scope": "explicit", "recipient_positions": [1, 3], "donor_positions": [1, 3]},
                  "intervention": {"layers": [0], "site": "resid_post", "operation": "replace", "dose": 1.0},
                  "metadata": {"source_pair": pair, "renderer_id": "canonical_v1", "recipient_source": "recipient",
                               "donor_source": "counterfactual" if recorded_only else "natural_donor"}}
        if baseline:
            source["intervention"] = {"layers": []}
        else:
            identity = {"lineage_kind": "inline_physical_source_pair", "source_pair_id": pair["pair_id"],
                        "source_pair_sha256": registry.fingerprint(pair), "candidate_source_experiment_id": "discovery-template",
                        "candidate_source_spec_sha256": "b" * 64, "recipient_role": "recipient",
                        "donor_role": "matched" if recorded_only else "natural", "renderer_id": "canonical_v1",
                        "span_rule": "within_context_named_variable_toggle",
                        "recipient_token_sha256": registry.fingerprint(source["recipient_token_ids"]),
                        "donor_token_sha256": registry.fingerprint(source["donor_token_ids"]),
                        "selection_sha256": registry.fingerprint(source["selection"])}
            source["intervention"]["full_write_source_identity"] = identity
            if not recorded_only:
                source["intervention"].update(audit_full_write_norm=True, full_write_relative_tolerance=.01)
            if random or recorded_only:
                source["intervention"].update(random_control=True, random_seed=42, record_random_orientation=True,
                                               norm_reference="donor_minus_recipient")
        return worker.make_job(eid, **source)

    def write_spec(self, jobs):
        eid = jobs[0]["experiment_id"]
        registry.atomic_json(self.root / "specs" / f"{eid}.json", {"experiment_id": eid, "partition": "discovery", "jobs": jobs, "comparisons": []})

    def row(self, job):
        return registry.read_json(self.root / "results" / job["experiment_id"] / (job["job_id"] + ".json"))

    def proof(self):
        self.engine._ensure_loaded()
        snapshot = snapshot_identity(self.model)
        return make_loaded_model_proof(self.engine, snapshot, snapshot, self.runtime)

    def test_compiler_requires_materialized_proof_before_capture(self):
        with self.assertRaisesRegex(ValueError, "loaded-model proof"):
            worker.compile_patch(self.engine, self.job(), OrderedDict())
        self.assertEqual(self.engine.captures, [])
        worker.compile_patch(self.engine, self.job(), OrderedDict(), loaded_model_proof=self.proof())

    def test_source_tokens_pair_roles_and_positions_rejected_before_capture(self):
        mutations = [lambda j: j["recipient_token_ids"].__setitem__(0, 8),
                     lambda j: j["metadata"]["source_pair"].update(layout_id="different"),
                     lambda j: j["metadata"].update(donor_source="counterfactual"),
                     lambda j: j["metadata"].update(renderer_id="alternate"),
                     lambda j: j["selection"].update(recipient_positions=[3, 1]),
                     lambda j: j.update(pair_id="different"),
                     lambda j: j.update(recipient_positions=[3, 1])]
        for mutate in mutations:
            source = self.job()
            mutate(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                worker.compile_patch(self.engine, source, OrderedDict(), loaded_model_proof=self.proof())
        self.assertEqual(self.engine.captures, [])

    def test_malformed_new_flags_and_geometry_fail_before_model_constructor(self):
        for fields in ({"audit_full_write_norm": "yes"}, {"full_write_relative_tolerance": .02},
                       {"layers": [0, 10]}, {"operation": "add"}, {"dose": .5}, {"rank": 1},
                       {"zero_donor": False}, {"random_control": True}):
            job = self.job()
            job["intervention"].update(fields)
            self.write_spec([job])
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                worker.run_experiment(self.root, self.repo, "test")
        self.constructor.assert_not_called()

    def test_intended_and_new_random_prepare_same_reference_then_score(self):
        jobs = [self.job(condition="natural"), self.job(condition="natural_random", random=True)]
        self.write_spec(jobs)
        result = worker.run_experiment(self.root, self.repo, "test")
        self.assertEqual(result["new_results"], 2)
        rows = [self.row(job) for job in jobs]
        for row in rows:
            self.assertEqual(row["full_write_norm_audit_status"], "passed")
            self.assertNotIn("projected_norm_audit_status", row)
            self.assertEqual(row["status"], "complete")
            self.assertTrue(row["applied_full_write_audits"][0]["recipient_hash_matches"])
        self.assertEqual(rows[0]["compiled_full_write_audits"][0]["expected_delta_l2"], rows[1]["compiled_full_write_audits"][0]["expected_delta_l2"])
        self.assertEqual(rows[1]["random_orientation_audits"][0]["source_identity"]["donor_role"], "natural")
        self.assertEqual(self.engine.endpoints, ["forced", "native", "forced", "native"])

    def test_record_only_random_matches_legacy_donor_bytes_and_same_raw_draw(self):
        old = self.job(condition="old", recorded_only=True)
        new = self.job(condition="new", random=True)
        new["donor_token_ids"] = [1, 12, 3, 20]
        new["intervention"]["full_write_source_identity"]["donor_token_sha256"] = registry.fingerprint(new["donor_token_ids"])
        proof = self.proof()
        _, recorded = worker.compile_patch(self.engine, old, OrderedDict(), loaded_model_proof=proof)
        legacy = deepcopy(old)
        for name in ("full_write_source_identity", "record_random_orientation"):
            legacy["intervention"].pop(name)
        _, unrecorded = worker.compile_patch(self.engine, legacy, OrderedDict())
        _, intended = worker.compile_patch(self.engine, new, OrderedDict(), loaded_model_proof=proof)
        np.testing.assert_array_equal(recorded[0]["donor"], unrecorded[0]["donor"])
        self.assertNotIn("full_write_norm_audit", recorded[0])
        self.assertNotIn("expected_recipient_float32_sha256", recorded[0])
        self.assertEqual(recorded[0]["random_orientation_audit"]["raw_gaussian_float32_sha256"], intended[0]["random_orientation_audit"]["raw_gaussian_float32_sha256"])

    def test_actual_norm_failure_persists_before_endpoints_and_stops_first(self):
        jobs = [self.job(), self.job(condition="unexecuted", random=True)]
        self.write_spec(jobs)
        self.engine.ratio = 1.02
        result = worker.run_experiment(self.root, self.repo, "test")
        row = self.row(jobs[0])
        self.assertEqual(result["reason"], "full_write_technical_failure")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(row["full_write_norm_audit_status"], "failed")
        self.assertFalse(row["applied_full_write_audits"][0]["passed"])
        self.assertTrue(row["patch_log"])
        self.assertEqual(self.engine.endpoints, [])
        self.assertFalse((self.root / "results/test" / (jobs[1]["job_id"] + ".json")).exists())

    def test_recipient_guard_failure_preserves_preparation_not_subspace_failure(self):
        job = self.job()
        self.write_spec([job])
        self.engine.mismatch = True
        worker.run_experiment(self.root, self.repo, "test")
        row = self.row(job)
        self.assertEqual(row["full_write_norm_audit_status"], "failed_before_write")
        self.assertFalse(row["prefill_patch_failure"]["write_applied"])
        self.assertTrue(row["compiled_full_write_audits"])
        self.assertNotIn("projected_norm_audit_status", row)
        self.assertEqual(self.engine.endpoints, [])

    def test_nonfinite_capture_preparation_failure_has_evidence_without_prefill(self):
        job = self.job()
        self.write_spec([job])
        self.engine.invalid_capture = True
        worker.run_experiment(self.root, self.repo, "test")
        row = self.row(job)
        self.assertEqual(row["full_write_norm_audit_status"], "failed_preparation")
        self.assertEqual(row["full_write_preparation_failure"]["stage"], "reference_norm_preparation")
        self.assertFalse(row["full_write_preparation_failure"]["write_applied"])
        self.assertEqual(self.engine.prefills, [])
        self.assertEqual(self.engine.endpoints, [])

    def test_failed_float32_sanity_persists_reference_and_draw_without_prefill(self):
        from src.activation_causality.full_write_audit import random_orientation_audit
        job = self.job(random=True)
        self.write_spec([job])
        def fail(*args, **kwargs):
            audit = random_orientation_audit(*args, **kwargs)
            audit.update(passed=False, errors=["synthetic cancellation fixture"])
            return audit
        with patch("src.activation_causality.full_write_audit.random_orientation_audit", side_effect=fail):
            worker.run_experiment(self.root, self.repo, "test")
        row = self.row(job)
        self.assertEqual(row["full_write_norm_audit_status"], "failed_preparation")
        self.assertTrue(row["compiled_full_write_audits"])
        self.assertTrue(row["random_orientation_audits"][0]["raw_gaussian_float32_sha256"])
        self.assertEqual(self.engine.prefills, [])

    def test_record_only_control_does_not_acquire_a_norm_acceptance_gate(self):
        from src.activation_causality.full_write_audit import random_orientation_audit
        job = self.job(recorded_only=True)
        self.write_spec([job])
        def failed_record(*args, **kwargs):
            audit = random_orientation_audit(*args, **kwargs)
            audit.update(passed=False, errors=["record-only diagnostic failure fixture"])
            return audit
        with patch("src.activation_causality.full_write_audit.random_orientation_audit", side_effect=failed_record):
            worker.run_experiment(self.root, self.repo, "test")
        row = self.row(job)
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["full_write_norm_audit_status"], "not_requested")
        self.assertFalse(row["random_orientation_audits"][0]["passed"])
        self.assertEqual(self.engine.endpoints, ["forced", "native"])

    def test_identity_full_write_retains_exact_zero_in_denominator(self):
        job = self.job(condition="identity")
        job["donor_token_ids"] = job["recipient_token_ids"][:]
        job["metadata"]["donor_source"] = "recipient"
        job["intervention"]["full_write_source_identity"].update(
            donor_role="recipient", donor_token_sha256=registry.fingerprint(job["donor_token_ids"]))
        self.write_spec([job])
        worker.run_experiment(self.root, self.repo, "test")
        row = self.row(job)
        self.assertEqual(row["compiled_full_write_audits"][0]["expected_delta_l2"], [0, 0])
        self.assertTrue(row["applied_full_write_audits"][0]["passed"])
        self.assertEqual(self.engine.endpoints, ["forced", "native"])

    def test_entire_spec_helper_and_snapshot_identity_bound_on_baseline_only_resume(self):
        jobs = [self.job(condition="baseline", baseline=True), self.job()]
        self.write_spec(jobs)
        worker.run_experiment(self.root, self.repo, "test", max_jobs=1)
        row = self.row(jobs[0])
        self.assertEqual(row["full_write_norm_audit_status"], "not_requested")
        self.assertIn("model_snapshot_identity_sha256", row)
        self.assertIn("src/activation_causality/full_write_audit.py", row["inference_source_hashes"])
        self.runtime["sources"]["src/activation_causality/full_write_audit.py"] = "f" * 64
        self.constructor.reset_mock()
        with self.assertRaisesRegex(ValueError, "execution code"):
            worker.run_experiment(self.root, self.repo, "test")
        self.constructor.assert_not_called()

    def test_missing_helper_and_changed_weights_reject_before_model(self):
        job = self.job()
        self.write_spec([job])
        digest = self.runtime["sources"].pop("src/activation_causality/full_write_audit.py")
        with self.assertRaisesRegex(ValueError, "fingerprinted"):
            worker.run_experiment(self.root, self.repo, "test")
        self.constructor.assert_not_called()
        self.runtime["sources"]["src/activation_causality/full_write_audit.py"] = digest
        worker.run_experiment(self.root, self.repo, "test")
        (self.model / "weights.safetensors").write_bytes(b"changed")
        self.constructor.reset_mock()
        with self.assertRaisesRegex(ValueError, "execution code"):
            worker.run_experiment(self.root, self.repo, "test")
        self.constructor.assert_not_called()

    def test_absent_flags_preserve_legacy_execution_identity_outputs_and_no_snapshot(self):
        job = self.job(baseline=True)
        self.write_spec([job])
        with patch("src.activation_causality.model_identity.snapshot_identity", side_effect=AssertionError("legacy has no snapshot")):
            worker.run_experiment(self.root, self.repo, "test")
            row = self.row(job)
            for name in ("full_write_audit", "subspace_audit", "basis_runtime", "model_identity"):
                self.runtime["sources"][f"src/activation_causality/{name}.py"] = "f" * 64
            resumed = worker.run_experiment(self.root, self.repo, "test")
        self.assertEqual(resumed["reason"], "no_pending_jobs")
        self.assertNotIn("full_write_norm_audit_status", row)
        self.assertNotIn("model_snapshot_identity_sha256", row)
        expected_sources = {name: digest for name, digest in self.runtime["sources"].items()
                            if name.endswith(("engine.py", "campaign.py")) or name in ENGINE_DEPENDENCY_PATHS}
        expected = registry.fingerprint({"sources": expected_sources, "versions": self.runtime["versions"],
                                         "model": str(self.model), "model_files": {name: registry.file_sha256(self.model / name)
                                         for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")}, "head_mode": "full"})
        self.assertEqual(row["execution_key"], expected)

    def test_archived_legacy_compiler_parity_for_replace_add_random_and_baseline(self):
        archive = Path(__file__).resolve().parent / "fixtures/activation_causality/legacy_campaign_before_full_write.py"
        spec = importlib.util.spec_from_file_location("src.activation_causality._legacy_full_write_test", archive)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for operation, random, baseline in (("replace", False, False), ("replace", True, False),
                                             ("add", False, False), ("add", True, False), ("replace", False, True)):
            job = self.job(baseline=baseline)
            job["intervention"] = {"layers": [] if baseline else [0], "site": "resid_post", "operation": operation, "dose": .5}
            if random:
                job["intervention"].update(random_control=True, random_seed=42)
            left = worker.compile_patch(self.engine, job, OrderedDict())
            right = legacy.compile_patch(self.engine, job, OrderedDict())
            self.assertEqual(left[0], right[0])
            self.assertEqual(len(left[1]), len(right[1]))
            for a, b in zip(left[1], right[1], strict=True):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    if isinstance(a[key], np.ndarray):
                        np.testing.assert_array_equal(a[key], b[key])
                    else:
                        self.assertEqual(a[key], b[key])

    def test_archived_legacy_runner_row_parity_with_fixed_runtime(self):
        archive = Path(__file__).resolve().parent / "fixtures/activation_causality/legacy_campaign_before_full_write.py"
        module_spec = importlib.util.spec_from_file_location("src.activation_causality._legacy_runner_test", archive)
        legacy = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(legacy)
        job = self.job(baseline=True)
        self.write_spec([job])
        previous = self.repo / "legacy_campaign"
        registry.atomic_json(previous / "campaign.json", registry.read_json(self.root / "campaign.json"))
        registry.atomic_json(previous / "specs/test.json", registry.read_json(self.root / "specs/test.json"))
        with patch.object(worker.time, "monotonic", return_value=123), patch.object(worker, "now_utc", return_value="fixed-time"):
            worker.run_experiment(self.root, self.repo, "test")
        with patch.object(legacy, "runtime_fingerprint", return_value=deepcopy(self.runtime)), \
                patch.object(legacy, "worker_lock", return_value=nullcontext()), \
                patch.object(legacy, "seconds_remaining", return_value=3600), \
                patch.object(legacy.time, "monotonic", return_value=123), \
                patch.object(legacy, "now_utc", return_value="fixed-time"):
            legacy.run_experiment(previous, self.repo, "test")
        old = registry.read_json(previous / "results/test" / (job["job_id"] + ".json"))
        self.assertEqual(self.row(job), old)


if __name__ == "__main__":
    unittest.main()
