"""CPU runner gates for loaded-model proofs and retained subspace audits."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.activation_causality import registry
from src.activation_causality.basis_runtime import LoadedModelProof
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
from src.activation_causality.campaign import make_job, run_experiment
from src.activation_causality.engine import RecipientStateMismatch
from src.activation_causality.model_identity import snapshot_identity as actual_snapshot_identity
from src.activation_causality.subspace_audit import array_fingerprint
from src.activation_causality.subspace_controls import projected_norm_matched_random_control as actual_random_control


HELPERS = tuple("src/activation_causality/" + name + ".py" for name in
                ("basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit"))


class FakeSubspaceWorker:
    def __init__(self, model_path, events):
        self.model_path, self.events = Path(model_path).resolve(), events
        self.metadata = {"loaded": False, "hidden_dim": 4}
        self._head_mode = "full"
        self.captures, self.states, self.forced_states, self.native_states = [], [], [], []
        self.failure_mode = None
        self.on_materialize = None

    @property
    def head_mode(self):
        return self._head_mode

    def _ensure_loaded(self):
        self.events.append("materialize")
        self.metadata["loaded"] = True
        if self.on_materialize is not None:
            self.on_materialize()
            self.on_materialize = None

    def validate(self):
        if self.metadata["loaded"] is not True:
            raise AssertionError("subspace worker must materialize inside the snapshot bracket")
        self.events.append("validate")
        return {"passed": True, "model": {"head_mode": "full"}}

    @staticmethod
    def values(ids, layer, positions):
        return np.asarray([[ids[p], 2 * ids[p], layer + 1, p + 1] for p in positions], dtype=np.float32)

    def capture(self, ids, layers, positions, site):
        if self.metadata["loaded"] is not True:
            raise AssertionError("capture preceded verified model materialization")
        self.events.append("capture")
        self.captures.append((ids.copy(), layers.copy(), positions.copy(), site))
        return {layer: self.values(ids, layer, positions) for layer in layers}

    def prefill(self, token_ids, patches=()):
        self.events.append("prefill")
        logs = []
        for value in patches:
            original = self.values(token_ids, value["layer"], value["positions"])
            difference = value["donor"] - original
            if value["operation"] == "subspace":
                difference = (difference @ value["basis"]) @ value["basis"].T
            difference *= value["dose"]
            actual_norms = np.linalg.norm(difference.astype(np.float64), axis=1)
            if self.failure_mode == "norm":
                actual_norms *= 1.1
            recipient_hash = array_fingerprint(original)
            if self.failure_mode == "recipient_prewrite":
                raise RecipientStateMismatch({
                    "status": "recipient_state_mismatch", "write_applied": False,
                    "layer": value["layer"], "site": value["site"], "positions": value["positions"],
                    "operation": value["operation"], "dose": value["dose"], "rank": value["basis"].shape[1],
                    "expected_recipient_float32_sha256": value["expected_recipient_float32_sha256"],
                    "recipient_float32_sha256": "0" * 64,
                    "before_l2": np.linalg.norm(original.astype(np.float64), axis=1).tolist(),
                    "recipient_shape": list(original.shape), "preceding_patch_count": 0,
                })
            if self.failure_mode == "recipient_log":
                recipient_hash = "0" * 64
            logs.append({"layer": value["layer"], "site": value["site"], "positions": value["positions"],
                         "operation": value["operation"], "dose": value["dose"],
                         "rank": value["basis"].shape[1] if "basis" in value else None,
                         "recipient_float32_sha256": recipient_hash,
                         "before_l2": np.linalg.norm(original.astype(np.float64), axis=1).tolist(),
                         "delta_l2": actual_norms.tolist()})
        state = SimpleNamespace(token_ids=tuple(token_ids), patch_log=logs)
        self.states.append(state)
        return state

    def forced(self, state):
        self.events.append("forced")
        self.forced_states.append(state)
        return {"q": [0.1, 0.2, 0.3, 0.4]}

    def generate(self, state, **kwargs):
        self.events.append("native")
        self.native_states.append(state)
        return {"native_action": "RIGHT", "generation_status": "complete", "generated_tokens": 1}


class SubspaceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.campaign = self.repo / "campaign"
        self.model = self.repo / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")
        (self.model / "weights.safetensors").write_bytes(b"strict indexed CPU fixture")
        registry.atomic_json(self.model / "model.safetensors.index.json", {"weight_map": {"one": "weights.safetensors"}})
        registry.atomic_json(self.campaign / "campaign.json", {"model_path": str(self.model)})
        self.events = []
        self.engine = FakeSubspaceWorker(self.model, self.events)
        sources = ("src/activation_causality/engine.py", "src/activation_causality/campaign.py", *HELPERS, *ENGINE_DEPENDENCY_PATHS)
        self.runtime = {"sources": {name: format(index + 1, "064x") for index, name in enumerate(sources)},
                        "versions": {"mlx": "CPU fixture", "mlx-lm": "CPU fixture", "numpy": np.__version__}}
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch("builtins.print"))
        self.stack.enter_context(patch("src.activation_causality.campaign.runtime_fingerprint", side_effect=lambda _: deepcopy(self.runtime)))
        self.stack.enter_context(patch("src.activation_causality.campaign.seconds_remaining", return_value=3600))
        self.stack.enter_context(patch("src.activation_causality.campaign.worker_lock", return_value=nullcontext()))
        self.stack.enter_context(patch("src.activation_causality.selection.validate_confirmation_ready"))
        self.constructor = self.stack.enter_context(patch("src.activation_causality.engine.MLXInterventionEngine", return_value=self.engine))

        def snapshot(path):
            self.events.append("snapshot")
            return actual_snapshot_identity(path)

        self.snapshot = self.stack.enter_context(patch("src.activation_causality.model_identity.snapshot_identity", side_effect=snapshot))

        def load(intervention, job, engine, proof):
            self.events.append("verify_basis")
            self.assertIsInstance(proof, LoadedModelProof)
            self.assertEqual(proof.engine_id, id(engine))
            self.assertEqual(proof.snapshot, actual_snapshot_identity(self.model))
            return np.eye(4, dtype=np.float32)[:, :2], {"rank": 2, "basis_id": "fake_basis"}

        self.loader = self.stack.enter_context(patch("src.activation_causality.basis_runtime.load_verified_basis", side_effect=load))

    def spec(self, identifier, conditions=("baseline", "forward", "matched", "rank_only")):
        jobs = []
        for condition in conditions:
            intervention = {"operation": "subspace", "layers": [10], "site": "resid_post", "dose": 0.5,
                            "basis_path": str(self.repo / "fake.npy"), "basis_sha256": "a" * 64,
                            "basis_metadata_path": str(self.repo / "fake.json"), "basis_metadata_sha256": "b" * 64}
            if condition == "baseline":
                intervention = {"layers": [], "operation": "replace", "site": "resid_post", "dose": 1.0}
            elif condition == "forward":
                intervention.update(audit_projected_norm=True, projected_norm_relative_tolerance=0.01)
            elif condition == "matched":
                intervention.update(projected_norm_matched_random_basis=True, random_seed=731, audit_projected_norm=True, projected_norm_relative_tolerance=0.01)
            elif condition == "rank_only":
                intervention.update(random_basis=True, random_seed=731)
            else:
                raise AssertionError(condition)
            jobs.append(make_job(identifier, condition=condition, family="wall", partition="discovery",
                                 layout_id="evaluation", pair_id="pair", seed=42,
                                 recipient_text="unused", donor_text="unused", recipient_token_ids=[1, 2, 3, 4],
                                 donor_token_ids=[5, 7, 9, 11], selection={"scope": "explicit", "recipient_positions": [1, 3], "donor_positions": [1, 3]},
                                 intervention=intervention, target_action="RIGHT", original_action="LEFT",
                                 natural=True, max_new_tokens=8, temperature=0.7, metadata={}))
        spec = {"experiment_id": identifier, "partition": "discovery", "jobs": jobs, "comparisons": []}
        registry.atomic_json(self.campaign / "specs" / f"{identifier}.json", spec)
        return spec

    def run_campaign(self, identifier, **kwargs):
        return run_experiment(self.campaign, self.repo, identifier, **kwargs)

    def rows(self, identifier):
        return [registry.read_json(path) for path in sorted((self.campaign / "results" / identifier).glob("*.json"))]

    def test_materialized_proof_and_helper_identity_cover_entire_cohort(self):
        self.spec("complete")
        summary = self.run_campaign("complete")
        self.assertEqual(summary["new_results"], 4)
        rows = self.rows("complete")
        self.assertEqual({row["status"] for row in rows}, {"complete"})
        self.assertEqual(len({row["execution_key"] for row in rows}), 1)
        expected_identity = actual_snapshot_identity(self.model)["identity_sha256"]
        expected_helpers = {name: self.runtime["sources"][name] for name in HELPERS}
        for row in rows:
            self.assertEqual(row["model_snapshot_identity_sha256"], expected_identity)
            self.assertEqual(row["subspace_implementation_sha256"], expected_helpers)
            audit_path = Path(row["model_load_audit_path"])
            self.assertEqual(registry.file_sha256(audit_path), row["model_load_audit_sha256"])
            if row["condition"] in {"forward", "matched"}:
                self.assertEqual(row["projected_norm_audit_status"], "passed")
                self.assertEqual(len(row["compiled_patch_audits"]), 1)
                self.assertTrue(row["applied_projected_norm_audits"][0]["passed"])
            else:
                self.assertEqual(row["projected_norm_audit_status"], "not_requested")
        materialize = self.events.index("materialize")
        self.assertIn("snapshot", self.events[:materialize])
        first_query = min(self.events.index("capture"), self.events.index("prefill"))
        self.assertIn("snapshot", self.events[materialize + 1:first_query])
        self.assertLess(self.events.index("verify_basis"), self.events.index("capture"))

    def test_applied_norm_or_recipient_failure_is_retained_before_endpoints(self):
        for mode, condition in (("norm", "matched"), ("recipient_log", "matched"), ("norm", "forward")):
            with self.subTest(mode=mode, condition=condition):
                identifier = mode + "_" + condition
                self.engine.failure_mode = mode
                self.spec(identifier, conditions=(condition,))
                summary = self.run_campaign(identifier)
                row, = self.rows(identifier)
                self.assertEqual(summary["errors"], 1)
                self.assertEqual(row["status"], "error")
                self.assertEqual(row["generation_status"], "technical_error")
                self.assertEqual(row["projected_norm_audit_status"], "failed")
                self.assertEqual(len(row["compiled_patch_audits"]), 1)
                self.assertEqual(row["compiled_patch_audits"][0]["random_control_audit"] is not None, condition == "matched")
                self.assertTrue(row["patch_log"])
                audit, = row["applied_projected_norm_audits"]
                self.assertFalse(audit["passed"])
                self.assertTrue(audit["applied_validation"]["errors"])
                self.assertNotIn("forced_q", row)
        self.assertEqual(self.engine.forced_states, [])
        self.assertEqual(self.engine.native_states, [])

    def test_float32_control_failure_retains_compiled_audit_without_prefill(self):
        self.spec("float32", conditions=("matched",))

        def failed_float32(*args, **kwargs):
            result = actual_random_control(*args, **kwargs)
            result["audit"]["float32"]["all_norms_match_within_tolerance"] = False
            return result

        with patch("src.activation_causality.subspace_controls.projected_norm_matched_random_control", side_effect=failed_float32):
            summary = self.run_campaign("float32")
        row, = self.rows("float32")
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(row["projected_norm_audit_status"], "failed")
        self.assertFalse(row["compiled_patch_audits"][0]["random_control_audit"]["float32"]["all_norms_match_within_tolerance"])
        self.assertEqual(self.engine.states, [])
        self.assertEqual(self.engine.forced_states, [])
        self.assertEqual(self.engine.native_states, [])

    def test_prewrite_recipient_mismatch_retains_structured_failure_without_endpoints(self):
        self.spec("prewrite", conditions=("matched",))
        self.engine.failure_mode = "recipient_prewrite"
        summary = self.run_campaign("prewrite")
        row, = self.rows("prewrite")
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(row["error_type"], "RecipientStateMismatch")
        self.assertEqual(row["projected_norm_audit_status"], "failed")
        self.assertEqual(len(row["compiled_patch_audits"]), 1)
        audit = row["prefill_patch_failure"]
        self.assertFalse(audit["write_applied"])
        self.assertEqual(audit["status"], "recipient_state_mismatch")
        self.assertEqual(audit["recipient_float32_sha256"], "0" * 64)
        self.assertEqual(audit["expected_recipient_float32_sha256"],
                         row["compiled_patch_audits"][0]["projected_norm_audit"]["expected_recipient_float32_sha256"])
        self.assertEqual(self.engine.states, [])
        self.assertEqual(self.engine.forced_states, [])
        self.assertEqual(self.engine.native_states, [])

    def test_loading_weight_change_blocks_all_queries(self):
        self.spec("changed_during_load")
        self.engine.on_materialize = lambda: (self.model / "weights.safetensors").write_bytes(b"changed while model materialized")
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_campaign("changed_during_load")
        self.assertEqual(self.engine.captures, [])
        self.assertEqual(self.engine.states, [])
        self.assertEqual(self.engine.forced_states, [])
        self.assertEqual(self.engine.native_states, [])
        audits = list((self.campaign / "reviews").glob("model_load_changed_during_load_*.json"))
        self.assertEqual(len(audits), 1)
        audit = registry.read_json(audits[0])
        self.assertFalse(audit["passed"])
        self.assertNotEqual(audit["before_load"]["identity_sha256"], audit["after_load"]["identity_sha256"])
        self.assertEqual(audit["error_type"], "ValueError")
        self.assertEqual(self.rows("changed_during_load"), [])

    def test_unchanged_resume_keeps_cohort_key_with_fresh_load_audit(self):
        self.spec("resume_complete")
        first = self.run_campaign("resume_complete", max_jobs=1)
        self.assertEqual(first["new_results"], 1)
        initial, = self.rows("resume_complete")
        self.assertEqual(initial["condition"], "baseline")
        resumed = self.run_campaign("resume_complete")
        self.assertEqual(resumed["new_results"], 3)
        rows = self.rows("resume_complete")
        self.assertEqual(len({row["execution_key"] for row in rows}), 1)
        self.assertEqual(len({row["engine_key"] for row in rows}), 1)
        self.assertEqual(len({row["model_load_audit_path"] for row in rows}), 2)
        self.assertEqual(len({row["model_snapshot_identity_sha256"] for row in rows}), 1)

    def test_every_helper_change_rejects_resume_after_baseline_only_first_batch(self):
        for index, helper in enumerate(HELPERS):
            with self.subTest(helper=helper):
                identifier = f"resume_helper_{index}"
                self.spec(identifier)
                self.run_campaign(identifier, max_jobs=1)
                row, = self.rows(identifier)
                self.assertEqual(row["condition"], "baseline")
                original = self.runtime["sources"][helper]
                self.runtime["sources"][helper] = "f" * 64
                with self.assertRaisesRegex(ValueError, "execution code"):
                    self.run_campaign(identifier)
                self.assertEqual(len(self.rows(identifier)), 1)
                self.runtime["sources"][helper] = original

    def test_actual_weight_payload_change_rejects_resume_even_with_same_size_mtime(self):
        self.spec("weight_resume")
        self.run_campaign("weight_resume", max_jobs=1)
        shard = self.model / "weights.safetensors"
        before = shard.stat()
        shard.write_bytes(b"X" * before.st_size)
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "execution code"):
            self.run_campaign("weight_resume")
        self.assertEqual(len(self.rows("weight_resume")), 1)

    def test_missing_subspace_helper_hash_rejects_before_engine_construction(self):
        self.spec("missing_helper")
        self.runtime["sources"].pop(HELPERS[0])
        with self.assertRaises(ValueError):
            self.run_campaign("missing_helper")
        self.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
