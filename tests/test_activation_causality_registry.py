from pathlib import Path
import tempfile
import unittest

from src.activation_causality.registry import (
    atomic_json, fingerprint, immutable_json, init_campaign, PAPER,
    register_spec, seconds_remaining, worker_lock,
)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.campaign = self.root / "campaigns" / "first"
        (self.root / PAPER).write_bytes(b"test paper")

    def tearDown(self):
        self.tmp.cleanup()

    def test_immutable_records_and_canonical_ids(self):
        path = self.root / "frozen.json"
        immutable_json(path, {"a": 1, "b": 2})
        immutable_json(path, {"b": 2, "a": 1})
        with self.assertRaises(ValueError):
            immutable_json(path, {"a": 2})
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))

    def test_deadline_does_not_reset_on_resume(self):
        first = init_campaign(self.campaign, self.root)
        second = init_campaign(self.campaign, self.root)
        self.assertEqual(first, second)
        self.assertGreater(seconds_remaining(self.campaign), 71 * 3600)
        with self.assertRaises(ValueError):
            init_campaign(self.campaign, self.root, hours=73)

    def test_model_lock_applies_across_campaigns(self):
        with worker_lock(self.campaign, lock_root=self.root):
            with self.assertRaises(RuntimeError):
                with worker_lock(self.root / "other_parent" / "second", lock_root=self.root):
                    pass
        with worker_lock(self.campaign, lock_root=self.root):
            pass

    def test_confirmation_requires_freeze_and_duplicate_jobs_rejected(self):
        spec = {"experiment_id": "x", "partition": "confirmation", "jobs": []}
        with self.assertRaises(ValueError):
            register_spec(self.campaign, spec)
        spec["partition"] = "discovery"
        spec["jobs"] = [{"job_id": "one", "target_action": "UP"}] * 2
        with self.assertRaises(ValueError):
            register_spec(self.campaign, spec)

    def test_atomic_json_rejects_nan(self):
        with self.assertRaises(ValueError):
            atomic_json(self.root / "result.json", {"x": float("nan")})
        self.assertFalse((self.root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
