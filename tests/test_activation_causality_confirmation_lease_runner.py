"""Prospective worker lease wiring; fake model, immutable temporary artifacts."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

try:
    from tests import test_activation_causality_full_write_runner as fixtures
except ImportError:
    import test_activation_causality_full_write_runner as fixtures

from src.activation_causality import campaign as worker
from src.activation_causality import registry


class ConfirmationLeaseRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FullWriteRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.repo = self.fixture.root, self.fixture.repo
        self.lease = {"lease_path": "/temporary/lease-a.json", "lease_sha256": "a" * 64,
                      "lease_id": "a" * 32, "inference_cutoff": "2030-01-01T00:00:00+00:00", "supervisor_pid": 1234}
        self.guard = self.fixture.stack.enter_context(patch("src.activation_causality.confirmation_supervisor.validate_execution_lease",
                                                          side_effect=lambda *args: deepcopy(self.lease)))
        for name in ("confirmation_supervisor", "confirmation_contract"):
            self.fixture.runtime["sources"][f"src/activation_causality/{name}.py"] = registry.fingerprint(name)
        self.freeze = {"schema_version": 4, "confirmation_mode": "co_primary_native_v4",
                       "inference_cutoff": self.lease["inference_cutoff"], "confirmation_primary_family": {},
                       "candidate_configurations": {"candidate": {"protocol_kind": "input_fact_subspace"}}}
        self.freeze_path = self.root / "freezes/selection_frozen.json"
        registry.atomic_json(self.freeze_path, self.freeze)

    def spec(self, *, legacy=False, n_jobs=1):
        jobs = []
        for i in range(n_jobs):
            job = self.fixture.job(condition=f"baseline_{i}", baseline=True)
            job["partition"] = "confirmation"
            job["job_id"] = registry.fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24]
            jobs.append(job)
        spec = {"experiment_id": "test", "partition": "confirmation", "freeze_id": "selection_frozen",
                "candidate_id": "candidate", "jobs": jobs, "comparisons": []}
        if not legacy:
            spec["protocol_kind"] = "input_fact_subspace"
        registry.atomic_json(self.root / "specs/test.json", spec)
        return jobs

    def run_worker(self, **kwargs):
        return worker.run_experiment(self.root, self.repo, "test", **kwargs)

    def test_explicit_new_mode_checks_before_hash_load_and_every_job(self):
        jobs = self.spec(n_jobs=2)
        phases = []
        def record(*args):
            phases.append((self.fixture.engine.load_calls, len(self.fixture.engine.prefills)))
            return deepcopy(self.lease)
        self.guard.side_effect = record
        self.run_worker()
        self.assertEqual(phases, [(0, 0), (0, 0), (0, 0), (1, 0), (1, 1)])
        row = self.fixture.row(jobs[0])
        self.assertEqual(row["confirmation_lease_path"], self.lease["lease_path"])
        self.assertEqual(row["confirmation_lease_sha256"], self.lease["lease_sha256"])
        self.assertEqual(row["confirmation_lease_id"], self.lease["lease_id"])
        self.assertEqual(row["confirmation_inference_cutoff"], self.lease["inference_cutoff"])

    def test_unsupervised_new_mode_rejects_before_model_fingerprints_or_constructor(self):
        self.spec()
        self.guard.side_effect = ValueError("no dedicated lease")
        with patch("src.activation_causality.selection.validate_confirmation_ready", side_effect=AssertionError("must not enter readiness")), \
                patch("src.activation_causality.model_identity.snapshot_identity", side_effect=AssertionError("must not hash model")), \
                self.assertRaisesRegex(ValueError, "no dedicated lease"):
            self.run_worker()
        self.fixture.constructor.assert_not_called()
        self.assertEqual(self.fixture.engine.load_calls, 0)
        event = json.loads((self.root / "events.jsonl").read_text().splitlines()[-1])
        self.assertEqual(event["kind"], "confirmation_execution_lease_rejected")
        self.assertEqual(event["phase"], "before_readiness")

    def test_lease_expiry_during_fingerprinting_rejects_before_constructor(self):
        self.spec()
        self.guard.side_effect = [deepcopy(self.lease)] * 2 + [ValueError("cutoff after fingerprinting")]
        with self.assertRaisesRegex(ValueError, "cutoff after"):
            self.run_worker()
        self.fixture.constructor.assert_not_called()
        self.assertEqual(self.fixture.engine.load_calls, 0)

    def test_expiry_during_readiness_rejects_before_model_fingerprint(self):
        self.spec()
        self.guard.side_effect = [deepcopy(self.lease), ValueError("expired during readiness")]
        with patch("src.activation_causality.selection.validate_confirmation_ready") as ready, \
                patch.object(worker, "runtime_fingerprint", side_effect=AssertionError("must not fingerprint")), \
                self.assertRaisesRegex(ValueError, "expired during readiness"):
            self.run_worker()
        ready.assert_called_once()
        self.fixture.constructor.assert_not_called()

    def test_freeze_contract_parses_and_hashes_exactly_the_same_bytes(self):
        self.spec()
        spec = registry.read_json(self.root / "specs/test.json")
        raw = self.freeze_path.read_bytes()
        with patch.object(worker, "file_sha256", side_effect=AssertionError("must not reopen freeze for SHA")), \
                patch.object(type(self.freeze_path), "read_bytes", return_value=raw) as read:
            contract = worker._confirmation_execution_contract(self.root, spec)
        read.assert_called_once_with()
        self.assertEqual(contract["freeze_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(contract["inference_cutoff"], self.freeze["inference_cutoff"])

    def test_between_job_failure_keeps_next_job_unexecuted_and_prior_row_intact(self):
        jobs = self.spec(n_jobs=2)
        self.guard.side_effect = [deepcopy(self.lease)] * 4 + [ValueError("parent gone before next job")]
        with self.assertRaisesRegex(ValueError, "parent gone"):
            self.run_worker()
        self.assertEqual(self.fixture.row(jobs[0])["status"], "complete")
        self.assertFalse((self.root / "results/test" / (jobs[1]["job_id"] + ".json")).exists())
        self.assertEqual(self.fixture.engine.endpoints, ["forced", "native"])
        events = [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["phase"], "before_job")

    def test_fresh_ephemeral_lease_does_not_change_execution_key_on_resume(self):
        jobs = self.spec(n_jobs=2)
        self.run_worker(max_jobs=1)
        first = self.fixture.row(jobs[0])
        self.lease.update(lease_id="b" * 32, lease_sha256="b" * 64, lease_path="/temporary/lease-b.json", supervisor_pid=4321)
        self.run_worker()
        second = self.fixture.row(jobs[1])
        self.assertEqual(first["execution_key"], second["execution_key"])
        self.assertEqual(first["engine_key"], second["engine_key"])
        self.assertNotEqual(first["confirmation_lease_id"], second["confirmation_lease_id"])
        self.assertNotEqual(first["confirmation_lease_sha256"], second["confirmation_lease_sha256"])

    def test_fixed_freeze_contract_and_supervisor_source_mutations_block_resume(self):
        jobs = self.spec(n_jobs=2)
        self.run_worker(max_jobs=1)
        source = "src/activation_causality/confirmation_supervisor.py"
        old = self.fixture.runtime["sources"][source]
        self.fixture.runtime["sources"][source] = "d" * 64
        with self.assertRaisesRegex(ValueError, "execution code"):
            self.run_worker()
        self.fixture.runtime["sources"][source] = old
        self.freeze["inference_cutoff"] = "2030-01-02T00:00:00+00:00"
        registry.atomic_json(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(ValueError, "execution code"):
            self.run_worker()
        self.assertFalse((self.root / "results/test" / (jobs[1]["job_id"] + ".json")).exists())

    def test_missing_marker_never_falls_back_from_new_indicators(self):
        self.spec()
        for field in ("confirmation_primary_family", "candidate_configurations"):
            freeze = {"schema_version": 4, field: self.freeze[field]}
            registry.atomic_json(self.freeze_path, freeze)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "no legacy fallback"):
                self.run_worker()
        self.fixture.constructor.assert_not_called()
        self.guard.assert_not_called()

    def test_legacy_confirmation_does_not_require_lease_or_pin_supervisor_helper(self):
        jobs = self.spec(legacy=True, n_jobs=2)
        registry.atomic_json(self.freeze_path, {"schema_version": 2, "candidate_configurations": {"candidate": {"protocol_kind": "input_fact"}}})
        self.run_worker(max_jobs=1)
        self.guard.assert_not_called()
        self.fixture.runtime["sources"]["src/activation_causality/confirmation_supervisor.py"] = "e" * 64
        self.fixture.runtime["sources"]["src/activation_causality/confirmation_contract.py"] = "e" * 64
        self.run_worker()
        self.guard.assert_not_called()
        first, second = (self.fixture.row(job) for job in jobs)
        self.assertEqual(first["execution_key"], second["execution_key"])
        self.assertNotIn("confirmation_lease_id", first)

    def test_missing_new_helper_hash_rejects_before_constructor(self):
        self.spec()
        self.fixture.runtime["sources"].pop("src/activation_causality/confirmation_supervisor.py")
        with self.assertRaisesRegex(ValueError, "fingerprinted confirmation_supervisor"):
            self.run_worker()
        self.fixture.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
