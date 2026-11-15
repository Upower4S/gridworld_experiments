"""Dedicated supervisor CPU fixtures: fake worker entrypoints, no model calls."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.activation_causality import confirmation_supervisor as supervisor
from src.activation_causality.confirmation_contract import inference_cutoff


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.campaign = self.repo / "campaign"
        self.campaign.mkdir()
        self.stage = Path(supervisor.__file__).resolve().parents[2]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.ready = self.stack.enter_context(patch("src.activation_causality.selection.validate_confirmation_ready"))
        self.sources = {}
        for name in supervisor.REQUIRED_SOURCES:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# pinned CPU fixture source\n")
            self.sources[name] = supervisor._sha(path.read_bytes())
        self.spec = {"experiment_id": "confirm", "partition": "confirmation", "freeze_id": "frozen",
                     "candidate_id": "candidate", "jobs": [{"job_id": "one"}, {"job_id": "two"}]}
        self.write(self.campaign / "specs/confirm.json", self.spec)
        self.entrypoint = self.repo / "scripts/30_activation_causality_campaign.py"
        self.entrypoint.parent.mkdir()
        self.worker('print("CPU child validated")')
        self.contract(10)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    def worker(self, body):
        # The production supervisor still constructs only its fixed script argv.
        # This temporary entrypoint exercises the real lease validator on CPU.
        script = ("import sys,json,time,os\nfrom pathlib import Path\n"
                  f"sys.path.insert(0,{str(self.stage)!r})\n"
                  "from src.activation_causality.confirmation_supervisor import validate_execution_lease\n"
                  f"campaign=Path({str(self.campaign)!r});repo=Path({str(self.repo)!r})\n"
                  "spec=json.loads((campaign/'specs/confirm.json').read_text())\n"
                  "validation=validate_execution_lease(campaign,repo,spec)\n"
                  "print('LEASE_VALIDATED',flush=True)\n" + body + "\n")
        self.entrypoint.write_text(script)

    def contract(self, seconds):
        deadline = (datetime.now(timezone.utc) + timedelta(hours=6, seconds=seconds)).isoformat()
        self.config = {"deadline": deadline}
        self.freeze = {"inference_cutoff": inference_cutoff(deadline), "implementation": self.sources,
                       "worker_entrypoint_sha256": supervisor._sha(self.entrypoint.read_bytes())}
        self.write(self.campaign / "campaign.json", self.config)
        self.write(self.campaign / "freezes/frozen.json", self.freeze)

    def run_worker(self, **kwargs):
        return supervisor.run_supervised_confirmation(self.campaign, self.repo, "confirm", **kwargs)

    def manual_lease(self):
        artifacts = supervisor._artifacts(self.campaign, self.repo, "confirm")
        lease_id = "a" * 32
        self.paths = supervisor._paths(self.campaign, lease_id)
        lease = {"format": supervisor.FORMAT, "lease_id": lease_id, "created_at": supervisor._now(),
                 "supervisor_pid": 12345, **artifacts["binding"]}
        sha = supervisor._exclusive_json(self.paths["lease"], lease)
        self.launch = {"lease_sha256": sha, "worker_pid": 23456, "worker_process_group": 23456, "supervisor_pid": 12345}
        supervisor._exclusive_json(self.paths["launch"], self.launch)
        self.stack.enter_context(patch.dict(os.environ, {supervisor.LEASE_PATH_ENV: str(self.paths["lease"]), supervisor.LEASE_SHA_ENV: sha}))
        self.stack.enter_context(patch.object(supervisor.os, "getpid", return_value=23456))
        self.stack.enter_context(patch.object(supervisor.os, "getppid", return_value=12345))
        self.stack.enter_context(patch.object(supervisor.os, "getsid", return_value=23456))
        self.stack.enter_context(patch.object(supervisor.os, "kill"))
        return lease

    def test_real_cpu_child_receives_fixed_command_new_lease_and_validates(self):
        report = self.run_worker()
        self.assertEqual(report["status"], "worker_exited")
        self.assertEqual(report["worker_exit_code"], 0)
        self.assertTrue(report["incomplete"])  # A zero exit is not a complete cohort.
        self.assertTrue(report["cohort_technical_completion_not_assessed"])
        self.assertIn("LEASE_VALIDATED", Path(report["log_path"]).read_text())
        launch = json.loads(Path(report["launch_record_path"]).read_text())
        self.assertEqual(launch["command"], [str(Path(sys.executable).parent.resolve() / Path(sys.executable).name), "-I", "-B", str(self.entrypoint),
                                             "--campaign", str(self.campaign), "run", "--experiment-id", "confirm"])
        lease = json.loads(Path(report["lease_path"]).read_text())
        self.assertEqual(lease["source_hashes"], self.sources)
        self.assertEqual(lease["supervisor_pid"], os.getpid())
        self.ready.assert_called_once()
        again = self.run_worker()
        self.assertNotEqual(report["lease_id"], again["lease_id"])
        self.assertTrue(Path(report["terminal_record_path"]).exists())

    def test_existing_venv_child_preserves_prefix_and_dependency_visibility_without_importing_models(self):
        self.assertNotEqual(sys.prefix, sys.base_prefix, "run this integration regression with the existing project virtualenv")
        packages = ("numpy", "mlx_lm", "transformers")
        self.assertTrue(all(importlib.util.find_spec(name) is not None for name in packages))
        self.worker("import importlib.util\n"
                    "evidence={'prefix':str(Path(sys.prefix).resolve()),'base_prefix':str(Path(sys.base_prefix).resolve()),"
                    "'executable':sys.executable,'available':{name:importlib.util.find_spec(name) is not None "
                    "for name in ('numpy','mlx_lm','transformers')},"
                    "'model_packages_imported':[name for name in ('mlx_lm','transformers','mlx.core') if name in sys.modules]}\n"
                    "(campaign/'cpu_environment_probe.json').write_text(json.dumps(evidence))")
        self.contract(10)
        report = self.run_worker()
        self.assertEqual(report["worker_exit_code"], 0, Path(report["log_path"]).read_text())
        evidence = json.loads((self.campaign / "cpu_environment_probe.json").read_text())
        self.assertEqual(evidence["prefix"], str(Path(sys.prefix).resolve()))
        self.assertEqual(evidence["base_prefix"], str(Path(sys.base_prefix).resolve()))
        self.assertEqual(evidence["available"], dict.fromkeys(packages, True))
        self.assertEqual(evidence["model_packages_imported"], [])
        lease = json.loads(Path(report["lease_path"]).read_text())
        binding = supervisor._interpreter_binding()
        self.assertTrue(all(lease[key] == value for key, value in binding.items()))
        self.assertNotEqual(lease["interpreter_path"], lease["interpreter_resolved_path"])
        self.assertEqual(lease["interpreter_binary_sha256"], supervisor._sha(Path(sys.executable).read_bytes()))
        self.assertEqual(lease["interpreter_pyvenv_cfg_sha256"], supervisor._sha((Path(sys.prefix) / "pyvenv.cfg").read_bytes()))
        self.assertEqual(evidence["executable"], lease["interpreter_path"])

    def test_changed_interpreter_binary_or_environment_binding_rejects_lease(self):
        lease = self.manual_lease()
        for field in ("interpreter_binary_sha256", "interpreter_pyvenv_cfg_sha256", "interpreter_prefix"):
            changed = deepcopy(lease)
            changed[field] = "f" * 64
            self.write(self.paths["lease"], changed)
            digest = supervisor._sha(self.paths["lease"].read_bytes())
            with self.subTest(field=field), patch.dict(os.environ, {supervisor.LEASE_SHA_ENV: digest}), \
                    self.assertRaisesRegex(ValueError, "binding changed"):
                supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)

    def test_missing_virtualenv_configuration_rejects_before_launch(self):
        missing = self.repo / "environment_without_config"
        missing.mkdir()
        with patch.object(supervisor.sys, "prefix", str(missing)), \
                patch.object(supervisor.subprocess, "Popen") as launch, \
                self.assertRaisesRegex(ValueError, "pyvenv.cfg"):
            self.run_worker()
        launch.assert_not_called()

    def test_cutoff_kills_owned_sleeping_group_without_post_cutoff_grace(self):
        self.worker("(campaign/'results/confirm').mkdir(parents=True,exist_ok=True)\n"
                    "(campaign/'results/confirm/one.json').write_text('preserved raw bytes')\n"
                    "time.sleep(60)")
        self.contract(.6)
        start = time.monotonic()
        report = self.run_worker(poll_seconds=.01)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(report["status"], "cutoff_terminated")
        self.assertEqual(report["cutoff_action"]["signal"], "SIGKILL")
        self.assertGreaterEqual(report["cutoff_action"]["cutoff_signal_lateness_seconds"], 0)
        self.assertLess(report["worker_exit_code"], 0)
        self.assertTrue(report["incomplete"])
        self.assertEqual((self.campaign / "results/confirm/one.json").read_text(), "preserved raw bytes")
        self.assertEqual(report["current_result_inventory"]["missing_result_files"], ["two.json"])
        self.assertNotIn("native_action", report)

    def test_cutoff_stops_owned_descendant_group_without_touching_other_pid(self):
        heartbeat = self.campaign / "cpu_descendant_heartbeat.txt"
        child = "import time\nfrom pathlib import Path\np=Path(" + repr(str(heartbeat)) + ")\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.01)\n"
        self.worker("import subprocess\nsubprocess.Popen([sys.executable,'-I','-B','-c'," + repr(child) + "])\ntime.sleep(60)")
        self.contract(.6)
        real_killpg = os.killpg
        signalled = []
        def record(group, sig):
            signalled.append(group)
            return real_killpg(group, sig)
        with patch.object(supervisor.os, "killpg", side_effect=record):
            report = self.run_worker(poll_seconds=.01)
        launch = json.loads(Path(report["launch_record_path"]).read_text())
        self.assertEqual(signalled, [launch["worker_pid"]])
        self.assertNotEqual(signalled[0], os.getpgrp())
        before = heartbeat.read_bytes()
        time.sleep(.05)
        self.assertEqual(heartbeat.read_bytes(), before)

    def test_nonzero_cpu_exit_is_operational_failure(self):
        self.worker("raise SystemExit(7)")
        self.contract(10)
        report = self.run_worker()
        self.assertEqual(report["status"], "worker_failed")
        self.assertEqual(report["worker_exit_code"], 7)
        self.assertTrue(report["incomplete"])

    def test_expired_cutoff_never_launches(self):
        self.contract(-1)
        with patch.object(supervisor.subprocess, "Popen") as spawn:
            report = self.run_worker()
        spawn.assert_not_called()
        self.assertEqual(report["status"], "cutoff_before_launch")
        self.assertTrue(Path(report["terminal_record_path"]).exists())

    def test_wall_clock_rollback_cannot_extend_initial_monotonic_budget(self):
        cutoff = datetime.fromisoformat(self.freeze["inference_cutoff"])
        clock = {"wall": cutoff - timedelta(seconds=1), "mono": 0.0, "killed": False}
        process = SimpleNamespace(pid=987654, poll=lambda: -9 if clock["killed"] else None, wait=lambda timeout: -9)
        def start(*args, **kwargs):
            clock["wall"] -= timedelta(minutes=10)
            return process
        def sleep(seconds):
            clock["mono"] += seconds
        def kill(group, sig):
            self.assertEqual(group, process.pid)
            clock["killed"] = True
        with patch.object(supervisor, "_now", side_effect=lambda: clock["wall"].isoformat()), \
                patch.object(supervisor.time, "monotonic", side_effect=lambda: clock["mono"]), \
                patch.object(supervisor.time, "sleep", side_effect=sleep), \
                patch.object(supervisor.subprocess, "Popen", side_effect=start), \
                patch.object(supervisor.os, "killpg", side_effect=kill):
            report = self.run_worker(poll_seconds=.25)
        self.assertEqual(report["status"], "cutoff_terminated")
        self.assertEqual(clock["mono"], 1)
        self.assertGreater(report["cutoff_action"]["utc_remaining_seconds"], 0)
        self.assertLessEqual(report["cutoff_action"]["monotonic_remaining_seconds"], 0)

    def test_second_supervisor_cannot_share_campaign_ownership(self):
        with supervisor._supervisor_lock(self.campaign), patch.object(supervisor.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "another confirmation supervisor"):
                self.run_worker()
        spawn.assert_not_called()

    def test_reaped_leader_group_is_never_signalled_after_ownership_ends(self):
        process = SimpleNamespace(pid=987654, poll=lambda: 0)
        with patch.object(supervisor.os, "killpg") as kill:
            evidence = supervisor._owned_kill(process)
        kill.assert_not_called()
        self.assertFalse(evidence["signal_sent"])
        self.assertEqual(evidence["reason"], "owned_worker_already_exited")

    def test_wrong_cutoff_missing_source_or_entrypoint_pin_reject_before_launch(self):
        for mutation in (lambda f: f.update(inference_cutoff=self.config["deadline"]),
                         lambda f: f["implementation"].pop(supervisor.REQUIRED_SOURCES[0]),
                         lambda f: f.update(worker_entrypoint_sha256="f" * 64)):
            frozen = deepcopy(self.freeze)
            mutation(frozen)
            self.write(self.campaign / "freezes/frozen.json", frozen)
            with self.subTest(frozen=frozen), patch.object(supervisor.subprocess, "Popen") as spawn, self.assertRaises(ValueError):
                self.run_worker()
            spawn.assert_not_called()
            self.write(self.campaign / "freezes/frozen.json", self.freeze)
        self.assertFalse((self.campaign / "leases").exists())

    def test_readiness_failure_or_source_race_creates_no_lease(self):
        self.ready.side_effect = ValueError("not ready")
        with patch.object(supervisor.subprocess, "Popen") as spawn, self.assertRaisesRegex(ValueError, "not ready"):
            self.run_worker()
        spawn.assert_not_called()
        self.assertFalse((self.campaign / "leases").exists())
        def change(*args):
            self.spec["candidate_id"] = "changed"
            self.write(self.campaign / "specs/confirm.json", self.spec)
        self.ready.side_effect = change
        with self.assertRaisesRegex(ValueError, "changed across readiness"):
            self.run_worker()

    def test_startup_failure_has_immutable_terminal_evidence(self):
        with patch.object(supervisor.subprocess, "Popen", side_effect=OSError("CPU fixture failure")), self.assertRaises(OSError):
            self.run_worker()
        terminals = list((self.campaign / "reviews/confirmation_supervision").glob("*.terminal.json"))
        self.assertEqual(len(terminals), 1)
        terminal = json.loads(terminals[0].read_text())
        self.assertEqual(terminal["status"], "supervisor_error")
        self.assertEqual(terminal["error_type"], "OSError")

    def test_termination_signal_restores_handler_and_cleans_owned_worker(self):
        old_handler = signal.getsignal(signal.SIGTERM)
        state = {"interrupted": False, "killed": False}
        def poll():
            if not state["interrupted"]:
                state["interrupted"] = True
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return -9 if state["killed"] else None
        process = SimpleNamespace(pid=987654, poll=poll, wait=lambda timeout: -9)
        def kill(group, sig):
            self.assertEqual(group, process.pid)
            state["killed"] = True
        with patch.object(supervisor.subprocess, "Popen", return_value=process), \
                patch.object(supervisor.os, "killpg", side_effect=kill), \
                self.assertRaises(supervisor.SupervisorInterrupted):
            self.run_worker()
        self.assertIs(signal.getsignal(signal.SIGTERM), old_handler)
        self.assertTrue(state["killed"])
        terminal = json.loads(next((self.campaign / "reviews/confirmation_supervision").glob("*.terminal.json")).read_text())
        self.assertEqual(terminal["error_type"], "SupervisorInterrupted")
        self.assertEqual(terminal["cleanup_action"]["signal"], "SIGKILL")

    def test_worker_rejects_direct_unsupervised_execution(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "dedicated supervisor lease"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)

    def test_worker_requires_exact_live_parent_session_and_launch_pid(self):
        self.manual_lease()
        value = supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)
        self.assertEqual(value["lease_id"], "a" * 32)
        for name, returned in (("getppid", 12346), ("getsid", 111)):
            with self.subTest(name=name), patch.object(supervisor.os, name, return_value=returned), self.assertRaises(ValueError):
                supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)
        self.write(self.paths["launch"], {**self.launch, "worker_pid": 23457})
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)

    def test_worker_rejects_terminal_lease_and_changed_artifact(self):
        self.manual_lease()
        self.write(self.paths["terminal"], {"status": "worker_exited"})
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)
        self.paths["terminal"].unlink()
        self.write(self.campaign / "campaign.json", {**self.config, "changed": True})
        with self.assertRaisesRegex(ValueError, "binding changed"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)

    def test_worker_rejects_parent_death_and_expired_absolute_cutoff(self):
        self.manual_lease()
        with patch.object(supervisor.os, "kill", side_effect=ProcessLookupError()), self.assertRaisesRegex(ValueError, "no longer alive"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)
        future = (datetime.fromisoformat(self.freeze["inference_cutoff"]) + timedelta(seconds=1)).isoformat()
        with patch.object(supervisor, "_now", return_value=future), self.assertRaisesRegex(ValueError, "cutoff reached"):
            supervisor.validate_execution_lease(self.campaign, self.repo, self.spec)

    def test_bad_poll_and_arbitrary_command_are_not_accepted(self):
        for value in (0, -1, True, float("nan"), 30):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_worker(poll_seconds=value)
        with self.assertRaises(TypeError):
            self.run_worker(command=["anything"])

    def test_immutable_records_publish_complete_bytes_atomically_without_overwrite(self):
        path = self.campaign / "atomic_record.json"
        original_link = os.link
        observed = []
        def publish(source, destination):
            self.assertFalse(destination.exists())
            self.assertEqual(json.loads(source.read_bytes()), {"record": "complete"})
            observed.append(source)
            return original_link(source, destination)
        with patch.object(supervisor.os, "link", side_effect=publish):
            digest = supervisor._exclusive_json(path, {"record": "complete"})
        self.assertEqual(digest, supervisor._sha(path.read_bytes()))
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())
        with self.assertRaises(FileExistsError):
            supervisor._exclusive_json(path, {"record": "replacement forbidden"})
        self.assertEqual(json.loads(path.read_bytes()), {"record": "complete"})
        self.assertEqual(list(self.campaign.glob("atomic_record.json.*.partial")), [])


if __name__ == "__main__":
    unittest.main()
