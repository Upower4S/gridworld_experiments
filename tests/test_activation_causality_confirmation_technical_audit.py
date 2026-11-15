"""CPU-only discriminating fixtures for the standalone per-cohort auditor."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "scripts/41_activation_causality_confirmation_technical_audit.py"
SPEC = importlib.util.spec_from_file_location("confirmation_technical_auditor", SOURCE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def canonical_job(**extra):
    job = {"experiment_id": "demo", "partition": "confirmation", "natural": True,
           "pair_id": "pair", "layout_id": "layout", "condition": "baseline", "seed": 42,
           "family": "wall", "recipient_token_ids": [1, 2], "max_new_tokens": 2,
           "intervention": {"layers": [], "site": "resid_post", "operation": "replace", "dose": 1.0},
           "selection": {"scope": "explicit", "recipient_positions": [1], "donor_positions": [1]}, **extra}
    job["job_id"] = mod.fp(job)[:24]
    return job


def ready_fixture(root):
    campaign = root / "campaign"
    job = canonical_job()
    spec = {"experiment_id": "demo", "partition": "confirmation", "freeze_id": "frozen",
            "candidate_id": "candidate", "cohort_role": "primary", "protocol_kind": "contextual_input",
            "semantic_headline_eligible": False, "jobs": [job]}
    spec_sha = write(campaign / "specs/demo.json", spec)
    plan = {"freeze_id": "frozen", "confirmation_mode": "co_primary_native_v4", "schema_version": 4,
            "planned_experiments": {"demo": {"spec_sha256": spec_sha, "candidate_id": "candidate", "cohort_role": "primary"}}}
    plan_path = campaign / "planning/plan.json"
    plan_sha = write(plan_path, plan)
    review = {"approved": True, "freeze_id": "frozen", "plan_sha256": plan_sha, "plan_fingerprint": mod.fp(plan)}
    review_path = campaign / "reviews/approval.json"
    review_sha = write(review_path, review)
    write(campaign / "freezes/frozen.json", {**plan, "plan_path": str(plan_path), "plan_sha256": plan_sha,
                                             "review_path": str(review_path), "review_sha256": review_sha})
    return campaign, spec, job


class FilenameGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.campaign, self.spec, self.job = ready_fixture(self.root)

    def test_partial_malformed_raw_is_never_opened(self):
        path = self.campaign / "results/demo/unexpected.json"
        path.parent.mkdir(parents=True)
        path.write_text("this is not JSON")
        with patch.object(mod.Audit, "authenticate", side_effect=AssertionError("past complete gate")):
            result = mod.Audit(ROOT, self.campaign, "demo").audit()
        self.assertFalse(result["result_contents_read"])
        self.assertEqual(result["state"], "incomplete_filenames_only")
        self.assertEqual(result["readiness"]["unexpected_files"], ["unexpected.json"])
        self.assertNotIn(str(path), result["input_files_sha256"])

    def test_complete_check_ready_is_still_filename_only(self):
        path = self.campaign / "results/demo" / (self.job["job_id"] + ".json")
        path.parent.mkdir(parents=True)
        path.write_text("invalid content deliberately unread")
        with patch.object(mod.Audit, "authenticate", side_effect=AssertionError("past dry-run")):
            result = mod.Audit(ROOT, self.campaign, "demo").audit(check_ready=True)
        self.assertEqual(result["state"], "ready_filenames_only")
        self.assertNotIn(str(path), result["input_files_sha256"])

    def test_complete_last_row_waits_for_terminal_without_read(self):
        path = self.campaign / "results/demo" / (self.job["job_id"] + ".json")
        path.parent.mkdir(parents=True)
        path.write_text("invalid content deliberately unread")
        write(self.campaign / "leases/lease.json", {"lease_id": "lease", "experiment_id": "demo"})
        with patch.object(mod.Audit, "authenticate", side_effect=AssertionError("past terminal gate")):
            result = mod.Audit(ROOT, self.campaign, "demo").audit()
        self.assertEqual(result["state"], "pending_supervision_evidence_no_outcomes_read")
        self.assertNotIn(str(path), result["input_files_sha256"])

    def test_rowless_missing_declared_launch_stays_pending_before_raw(self):
        path = self.campaign / "results/demo" / (self.job["job_id"] + ".json")
        path.parent.mkdir(parents=True)
        path.write_text("unread malformed current result")
        lid = "a" * 32
        write(self.campaign / "leases" / (lid + ".json"), {"lease_id": lid, "experiment_id": "demo"})
        launch = self.campaign / "reviews/confirmation_supervision" / (lid + ".launch.json")
        write(launch.with_name(lid + ".terminal.json"), {"status": "worker_exited", "launch_record_path": str(launch), "launch_record_sha256": "a" * 64})
        with patch.object(mod.Audit, "authenticate", side_effect=AssertionError("past supervision gate")):
            result = mod.Audit(ROOT, self.campaign, "demo").audit()
        self.assertEqual(result["state"], "pending_supervision_evidence_no_outcomes_read")
        self.assertEqual(result["missing_supervision_records"], [str(launch)])
        self.assertNotIn(str(path), result["input_files_sha256"])

    def test_preloaded_real_repo_source_preserves_filename_only_check(self):
        # Report40 imports real src modules before these fixtures in the joint suite.
        loaded = SimpleNamespace(__file__=str(ROOT / "src/__init__.py"))
        with patch.dict(mod.sys.modules, {"src.synthetic_fixture_origin": loaded}):
            result = mod.Audit(ROOT, self.campaign, "demo").audit(check_ready=True)
        self.assertFalse(result["result_contents_read"])
        self.assertEqual(result["state"], "incomplete_filenames_only")

    def test_preloaded_wrong_repo_source_still_rejected(self):
        loaded = SimpleNamespace(__file__=str(self.root / "other_repo/src/__init__.py"))
        with patch.dict(mod.sys.modules, {"src.synthetic_fixture_origin": loaded}):
            with self.assertRaisesRegex(ValueError, "stale implementation import"):
                mod.Audit(ROOT, self.campaign, "demo").audit(check_ready=True)

    def test_changed_frozen_spec_rejected_before_raw(self):
        spec = deepcopy(self.spec)
        spec["jobs"][0]["seed"] = 43
        write(self.campaign / "specs/demo.json", spec)
        with self.assertRaisesRegex(ValueError, "registered spec differs"):
            mod.Audit(ROOT, self.campaign, "demo").audit(True)

    def test_retirement_rejected(self):
        write(self.campaign / "reviews/demo_retired.json", {"scientific_use": False})
        with self.assertRaisesRegex(ValueError, "retired"):
            mod.Audit(ROOT, self.campaign, "demo").audit(True)

    def test_final_lease_terminal_inventory_race_rejected(self):
        a = mod.Audit(ROOT, self.campaign, "demo")
        a.prepare()
        lid = "a" * 32
        write(self.campaign / "leases" / (lid + ".json"), {"lease_id": lid, "experiment_id": "demo"})
        a.terminal_readiness()
        write(self.campaign / "reviews/confirmation_supervision" / (lid + ".terminal.json"), {"lease_id": lid})
        with self.assertRaisesRegex(ValueError, "lease/terminal inventory changed"):
            a.finish({})

    def test_reject_raw_output_and_overwrite(self):
        with self.assertRaisesRegex(ValueError, "input tree"):
            mod.output_allowed(self.campaign / "results/demo/x.json", self.campaign)
        output = self.campaign / "reports/check.json"
        mod.publish(output, {"passed": True})
        with self.assertRaisesRegex(ValueError, "already exists"):
            mod.output_allowed(output, self.campaign)
        with self.assertRaises(FileExistsError):
            mod.publish(output, {"passed": False})
        self.assertEqual(json.loads(output.read_bytes()), {"passed": True})

    def test_strict_json_and_changed_input(self):
        for raw in ('{"x":1,"x":2}', '{"x":1e999}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                mod.strict(raw)
        inputs = mod.Inputs(self.root)
        path = self.root / "data.json"
        write(path, {"x": 1})
        inputs.read(path)
        write(path, {"x": 2})
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            inputs.finish()


class NativeChecks(unittest.TestCase):
    def setUp(self):
        from src.activation_causality.engine import parse_native_action, action_metrics
        from src.conceptual_belief_space.forced_suffix_probe import FORCED_SUFFIX
        self.job = canonical_job()
        self.audit = mod.Audit(ROOT, ROOT / "outputs/activation_causality/campaign_20260912", "demo")
        self.audit.parser = parse_native_action
        self.audit.terminal_ids = {9: "return", 8: "tool_call", 7: "eos"}
        self.text = '<|start|>assistant<|channel|>final<|message|>{"action":"UP"}<|return|>'
        class Tokenizer:
            def decode(inner, ids, skip_special_tokens=False):
                return self.text
        self.audit.tokenizer = Tokenizer()
        metrics = {**action_metrics([3., 2., 1., 0.], [0, 1, 2, 3]), "endpoint": "forced", "suffix": FORCED_SUFFIX}
        self.row = {"input_token_sha256": mod.fp([1, 2]), "n_input_tokens": 2,
                    "forced_q": {a: metrics["q_" + a] for a in mod.ACTIONS}, "forced_metrics": metrics, "forced_top_action": "UP",
                    "generated_token_ids": [4, 9], "generated_tokens": 2, "n_new_tokens": 2,
                    "generated_text": self.text, **parse_native_action(self.text), "stop_reason": "return",
                    "generation_status": "complete", "native_action": "UP", "action": "UP", "completed": True,
                    "ended_naturally": True, "truncated": False, "endpoint": "native_final"}

    def test_complete_native_and_separate_forced_validation(self):
        self.audit.endpoints(self.row, self.job)
        self.row["forced_metrics"]["logit_UP"] += 1
        with self.assertRaisesRegex(ValueError, "logits/conditional"):
            self.audit.endpoints(self.row, self.job)

    def test_tool_call_is_malformed_unresolved_not_retried(self):
        from src.activation_causality.engine import parse_native_action
        self.text = '<|start|>assistant<|channel|>analysis<|message|>tool<|call|>'
        self.row.update(generated_token_ids=[4, 8], generated_text=self.text, **parse_native_action(self.text),
                        stop_reason="tool_call", generation_status="malformed", native_action=None, action=None,
                        completed=False, ended_naturally=False)
        self.audit.endpoints(self.row, self.job)

    def test_cap_preserves_even_parseable_text_as_unresolved(self):
        self.row.update(generated_token_ids=[4, 5], stop_reason="token_limit", generation_status="truncated",
                        native_action=None, action=None, completed=False, ended_naturally=False, truncated=True)
        self.audit.endpoints(self.row, self.job)
        self.row["native_action"] = "UP"
        with self.assertRaisesRegex(ValueError, "status/endpoint"):
            self.audit.endpoints(self.row, self.job)

    def test_token_text_and_terminal_mismatch_rejected(self):
        self.row["generated_text"] += "x"
        with self.assertRaisesRegex(ValueError, "decode"):
            self.audit.endpoints(self.row, self.job)
        self.row["generated_text"] = self.text
        self.row["generated_token_ids"] = [9, 9]
        with self.assertRaisesRegex(ValueError, "continued after"):
            self.audit.endpoints(self.row, self.job)


class PatchChecks(unittest.TestCase):
    def setUp(self):
        from src.activation_causality.full_write_audit import prepare_full_write_norm_audit, validate_full_write_norm_audit
        from src.activation_causality.subspace_audit import array_fingerprint, validate_applied_norm_audit
        pair = {"pair_id": "pair"}
        selection = {"scope": "explicit", "recipient_positions": [1], "donor_positions": [1]}
        identity = {"lineage_kind": "inline_physical_source_pair", "span_rule": "within_context_named_variable_toggle",
                    "source_pair_id": "pair", "source_pair_sha256": mod.fp(pair), "recipient_role": "recipient", "donor_role": "natural",
                    "recipient_token_sha256": mod.fp([1, 2]), "donor_token_sha256": mod.fp([1, 3]),
                    "selection_sha256": mod.fp(selection), "renderer_id": "canonical_v1", "candidate_source_experiment_id": "source",
                    "candidate_source_spec_sha256": "a" * 64}
        self.job = canonical_job(donor_token_ids=[1, 3], selection=selection,
            metadata={"source_pair": pair, "recipient_source": "recipient", "donor_source": "natural_donor", "renderer_id": "canonical_v1"},
            intervention={"layers": [10], "site": "resid_post", "operation": "replace", "dose": 1.0,
                          "audit_full_write_norm": True, "full_write_relative_tolerance": .01, "full_write_source_identity": identity})
        self.r = np.zeros((1, 2880), np.float32)
        self.d = np.zeros_like(self.r)
        self.d[0, 0] = 2
        pre = prepare_full_write_norm_audit(self.r, self.d, layer=10, site="resid_post", recipient_positions=[1], donor_positions=[1],
                                          dose=1., source_identity=identity)
        pre.update(compiled_donor_float32_sha256=array_fingerprint(self.d), inline_definition_sha256=mod.fp(identity))
        # Exercise the actual frozen log emitter using a CPU array shim.
        from src.activation_causality.engine import MLXInterventionEngine
        engine = SimpleNamespace(_mx=SimpleNamespace(float32=np.float32, array=np.array, eval=lambda *args: None))
        logs = []
        MLXInterventionEngine._patch_tensor(engine, np.zeros((1, 2, 2880), np.float32),
            {"layer": 10, "site": "resid_post", "positions": [1], "operation": "replace", "dose": 1.,
             "donor": self.d, "expected_recipient_float32_sha256": array_fingerprint(self.r)}, logs)
        self.row = {"patch_log": logs, "compiled_full_write_audits": [pre],
                    "applied_full_write_audits": [validate_full_write_norm_audit(pre, logs)], "full_write_norm_audit_status": "passed"}
        self.audit = mod.Audit(ROOT, ROOT / "outputs/activation_causality/campaign_20260912", "demo")
        self.audit.full_validator = validate_full_write_norm_audit
        self.audit.projected_validator = validate_applied_norm_audit
        self.audit.array_hash = array_fingerprint
        self.audit.runtime = {"versions": {"numpy": np.__version__}}

    def test_full_write_actual_norm_and_recipient_guard(self):
        self.assertNotIn("after_l2", self.row["patch_log"][0])
        self.audit.patch_evidence(self.row, self.job)
        self.row["patch_log"][0]["recipient_float32_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "replay"):
            self.audit.patch_evidence(self.row, self.job)

    def test_emitted_norm_vectors_reject_missing_nonfinite_or_wrong_shape(self):
        for field in ("before_l2", "delta_l2", "relative_delta_l2"):
            for value in (None, [], [-1.], [float("nan")], [float("inf")], [True]):
                with self.subTest(field=field, value=value):
                    row = deepcopy(self.row)
                    row["patch_log"][0][field] = value
                    with self.assertRaisesRegex(ValueError, "logged norm vectors"):
                        self.audit.patch_evidence(row, self.job)

    def test_compiled_donor_and_inline_source_are_separately_bound(self):
        for field in ("compiled_donor_float32_sha256", "inline_definition_sha256"):
            row = deepcopy(self.row)
            row["compiled_full_write_audits"][0][field] = "b" * 64
            row["applied_full_write_audits"] = [self.audit.full_validator(row["compiled_full_write_audits"][0], row["patch_log"])]
            with self.assertRaisesRegex(ValueError, "compiled donor/source"):
                self.audit.patch_evidence(row, self.job)

    def test_matching_audit_cannot_hide_norm_mismatch(self):
        self.row["patch_log"][0]["delta_l2"] = [2.03]
        pre = self.row["compiled_full_write_audits"][0]
        self.row["applied_full_write_audits"] = [self.audit.full_validator(pre, self.row["patch_log"])]
        with self.assertRaisesRegex(ValueError, "replay"):
            self.audit.patch_evidence(self.row, self.job)

    def test_old_random_failed_record_is_not_a_new_norm_gate(self):
        from src.activation_causality.full_write_audit import random_orientation_audit
        intervention = self.job["intervention"]
        intervention.pop("audit_full_write_norm")
        intervention.pop("full_write_relative_tolerance")
        intervention.update(record_random_orientation=True, random_control=True, random_seed=42, norm_reference="donor_minus_recipient")
        intervention["full_write_source_identity"]["donor_role"] = "matched"
        self.job["metadata"]["donor_source"] = "counterfactual"
        raw = np.random.default_rng(42).standard_normal((1, 2880)).astype(np.float32)
        scaled = raw * (2 / np.linalg.norm(raw, axis=-1))[:, None]
        orientation = random_orientation_audit(self.r, self.d, raw, scaled, self.r + scaled, seed=42,
                                               source_identity=intervention["full_write_source_identity"])
        orientation.update(passed=False, errors=["record_only_test_failure"])
        self.row.update(compiled_full_write_audits=[], applied_full_write_audits=[], full_write_norm_audit_status="not_requested",
                        random_orientation_audits=[orientation])
        self.audit.patch_evidence(self.row, self.job)
        orientation["raw_gaussian_float32_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "seeded random orientation"):
            self.audit.patch_evidence(self.row, self.job)

    def test_exact_zero_rule_no_absolute_floor(self):
        mod.norm_interval([0., 2.], [0., 2.01])
        with self.assertRaisesRegex(ValueError, "exact-zero"):
            mod.norm_interval([0.], [1e-15])

    def test_two_array_hash_conventions_are_distinct(self):
        from src.activation_causality.localization import _array_fingerprint
        from src.activation_causality.subspace_audit import array_fingerprint
        basis = np.eye(4, dtype=np.float32)[:, :1]
        self.assertNotEqual(_array_fingerprint(basis), array_fingerprint(basis))

    def test_real_projected_norm_basis_and_recipient_checks(self):
        from src.activation_causality.subspace_audit import prepare_projected_norm_audit
        basis = np.eye(4, dtype=np.float32)[:, :1]
        r = np.zeros((1, 4), np.float32)
        d = np.array([[2., 3., 4., 5.]], np.float32)
        pre = prepare_projected_norm_audit(r, d, basis, layer=0, site="resid_post", positions=[1], dose=.5)
        job = canonical_job(metadata={"rank": 1}, intervention={"layers": [0], "site": "resid_post", "operation": "subspace",
                            "dose": .5, "audit_projected_norm": True, "projected_norm_relative_tolerance": .01})
        log = {"layer": 0, "site": "resid_post", "operation": "subspace", "positions": [1], "rank": 1,
               "dose": .5, "before_l2": [0.], "relative_delta_l2": [1e12], "delta_l2": [1.],
               "recipient_float32_sha256": self.audit.array_hash(r)}
        row = {"patch_log": [log], "compiled_patch_audits": [{"projected_norm_audit": pre, "random_control_audit": None}],
               "applied_projected_norm_audits": [self.audit.projected_validator(pre, [log])], "projected_norm_audit_status": "passed"}
        self.audit.spec = {"candidate_id": "candidate"}
        self.audit.basis_norm_hashes = {"candidate": self.audit.array_hash(basis)}
        self.audit.patch_evidence(row, job)
        self.audit.basis_norm_hashes["candidate"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "another basis"):
            self.audit.patch_evidence(row, job)


class RuntimeAndControls(unittest.TestCase):
    def setUp(self):
        from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
        from src.activation_causality.basis_runtime import activation_representation
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        c = root / "campaign"
        a = self.audit = mod.Audit(root, c, "demo")
        a.spec_path, a.freeze_path = c / "specs/demo.json", c / "freezes/frozen.json"
        a.spec_sha = "e" * 64
        a.spec = {"protocol_kind": "input_fact_subspace", "candidate_id": "candidate", "cohort_role": "primary"}
        names = {f"src/activation_causality/{n}.py" for n in ("engine", "campaign", "confirmation_supervisor", "confirmation_contract",
                  "basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit")}
        names.update(ENGINE_DEPENDENCY_PATHS)
        sources = {n: mod.fp(n) for n in names}
        a.runtime = {"sources": sources, "versions": {"mlx": ".31", "mlx-lm": ".31", "numpy": np.__version__, "transformers": "test"}}
        a.dependencies = {n: sources[n] for n in ENGINE_DEPENDENCY_PATHS}
        a.config = {"model_path": "/fake-local-model"}
        snapshot = {"identity_sha256": "f" * 64, "path": "/fake-local-model", "files": {n: "1" * 64 for n in
                    ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")}}
        representation = activation_representation(a.runtime)
        a.frozen = {"implementation": sources, "freeze_id": "frozen", "confirmation_mode": "co_primary_native_v4",
                    "campaign_sha256": "c" * 64, "worker_entrypoint_sha256": "d" * 64, "model_identity": snapshot,
                    "activation_representation": representation, "inference_cutoff": "2026-09-15T02:00:00+00:00"}
        freeze_sha = a.inputs.hashes[str(a.freeze_path)] = "b" * 64
        contract = {"confirmation_mode": "co_primary_native_v4", "freeze_id": "frozen", "freeze_sha256": freeze_sha,
                    "inference_cutoff": a.frozen["inference_cutoff"]}
        common = {"versions": a.runtime["versions"], "model": "/fake-local-model", "model_files": snapshot["files"],
                  "head_mode": "full", "model_snapshot_identity_sha256": snapshot["identity_sha256"]}
        execution = mod.fp({"sources": sources, **common, "confirmation_execution_contract": contract})
        engine = mod.fp({"engine_sha256": sources["src/activation_causality/engine.py"], "engine_dependency_sources": a.dependencies, **common})
        def stamp(second):
            return f"2026-09-12T17:00:{second:02d}+00:00"
        lid = "a" * 32
        lease_path = c / "leases" / (lid + ".json")
        a.interpreter_binding = {"interpreter_path": "/fake-venv/bin/python"}
        lease = {"format": "supervised_confirmation_execution_lease_v1", "lease_id": lid, "experiment_id": "demo", "freeze_id": "frozen",
                 "candidate_id": "candidate", "cohort_role": "primary", "repo_path": str(root), "campaign_path": str(c),
                 "campaign_sha256": "c" * 64, "spec_path": str(a.spec_path), "spec_sha256": a.spec_sha,
                 "freeze_path": str(a.freeze_path), "freeze_sha256": freeze_sha, "inference_cutoff": a.frozen["inference_cutoff"],
                 "worker_entrypoint_path": str(root / "scripts/30_activation_causality_campaign.py"), "worker_entrypoint_sha256": "d" * 64,
                 "source_hashes": sources, **a.interpreter_binding, "created_at": stamp(0), "supervisor_pid": 100}
        lease_sha = write(lease_path, lease)
        launch_path = c / "reviews/confirmation_supervision" / (lid + ".launch.json")
        launch = {"format": "supervised_confirmation_launch_v1", "lease_id": lid, "lease_sha256": lease_sha, "supervisor_pid": 100,
                  "worker_pid": 101, "worker_process_group": 101, "launched_at": stamp(1),
                  "command": [lease["interpreter_path"], "-I", "-B", lease["worker_entrypoint_path"], "--campaign", str(c), "run", "--experiment-id", "demo"]}
        launch_sha = write(launch_path, launch)
        logfile = c / "logs" / ("confirmation_supervision_" + lid + ".log")
        logfile.parent.mkdir(parents=True)
        logfile.write_bytes(b"CPU fixture log, no outcomes")
        terminal_path = c / "reviews/confirmation_supervision" / (lid + ".terminal.json")
        terminal = {"format": "supervised_confirmation_terminal_v1", "lease_id": lid, "lease_path": str(lease_path), "lease_sha256": lease_sha,
                    "experiment_id": "demo", "inference_cutoff": a.frozen["inference_cutoff"], "launch_record_path": str(launch_path),
                    "terminal_record_path": str(terminal_path), "launch_record_sha256": launch_sha, "finished_at": stamp(6),
                    "log_path": str(logfile), "log_sha256": hashlib.sha256(logfile.read_bytes()).hexdigest(),
                    "status": "worker_exited", "worker_exit_code": 0, "incomplete": False}
        write(terminal_path, terminal)
        load_path = c / "reviews/load.json"
        load = {"format": "materialized_model_snapshot_audit_v1", "passed": True, "experiment_id": "demo", "execution_key": execution,
                "before_load": snapshot, "after_load": snapshot, "activation_representation": representation, "materialized_engine_id": 12,
                "engine_metadata": {"loaded": True, "head_mode": "full", "hidden_dim": 2880}, "started_at": stamp(2), "finished_at": stamp(3)}
        load_sha = write(load_path, load)
        write(c / "reviews" / ("engine_validation_" + engine[:16] + ".json"), {"engine_key": engine, "runtime": a.runtime,
              "validation": {"passed": True, "model": {"head_mode": "full"}}})
        self.row = {"execution_key": execution, "engine_key": engine, "inference_source_hashes": sources,
                    "subspace_implementation_sha256": {f"src/activation_causality/{n}.py": sources[f"src/activation_causality/{n}.py"] for n in
                        ("basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit")},
                    "model_snapshot_identity_sha256": snapshot["identity_sha256"], "model_load_audit_path": str(load_path),
                    "model_load_audit_sha256": load_sha, "confirmation_lease_id": lid, "confirmation_lease_path": str(lease_path),
                    "confirmation_lease_sha256": lease_sha, "confirmation_inference_cutoff": a.frozen["inference_cutoff"],
                    "started_at": stamp(4), "finished_at": stamp(5)}
        a.leases = {}
        self.launch_path, self.terminal_path = launch_path, terminal_path

    def test_actual_execution_key_and_complete_lease_chain(self):
        self.audit.runtime_evidence(self.row, {})
        self.assertEqual(len(self.audit.leases), 1)
        self.row["execution_key"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "execution key"):
            self.audit.runtime_evidence(self.row, {})

    def test_load_audit_hash_and_row_cutoff(self):
        self.row["model_load_audit_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.audit.runtime_evidence(self.row, {})

    def test_wrong_process_group_cannot_pass_via_coherent_terminal_hash(self):
        launch = json.loads(self.launch_path.read_bytes())
        launch["worker_process_group"] = 102
        sha = write(self.launch_path, launch)
        terminal = json.loads(self.terminal_path.read_bytes())
        terminal["launch_record_sha256"] = sha
        write(self.terminal_path, terminal)
        with self.assertRaisesRegex(ValueError, "launch ownership"):
            self.audit.runtime_evidence(self.row, {})

    def test_existing_rowless_launch_is_independently_authenticated(self):
        a = self.audit
        a.associated_lease_inventory = {self.row["confirmation_lease_path"]: True}
        a.inputs.read(Path(self.row["confirmation_lease_path"]))
        self.assertEqual(len(a.supervision_history()), 1)
        launch = json.loads(self.launch_path.read_bytes())
        launch["command"][-1] = "different_cohort"
        sha = write(self.launch_path, launch)
        terminal = json.loads(self.terminal_path.read_bytes())
        terminal["launch_record_sha256"] = sha
        write(self.terminal_path, terminal)
        a.cache.clear()
        a.inputs.hashes.pop(str(self.launch_path))
        a.inputs.hashes.pop(str(self.terminal_path))
        with self.assertRaisesRegex(ValueError, "historical launch command"):
            a.supervision_history()

    def test_prior_worker_error_cannot_be_cleared_by_later_success(self):
        events = self.audit.campaign / "events.jsonl"
        data = [{"time": "2026-09-12T16:00:00+00:00", "kind": "worker_finished", "experiment_id": "demo", "completed": 0, "errors": 1},
                {"time": "2026-09-12T17:00:05+00:00", "kind": "worker_finished", "experiment_id": "demo", "completed": 1, "errors": 0}]
        events.write_text("".join(json.dumps(v) + "\n" for v in data))
        terminal = json.loads(self.terminal_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "previous worker technical failures"):
            self.audit.operational_events(terminal)

    def test_complete_event_prefix_distinguishes_trailing_append(self):
        events = self.audit.campaign / "events.jsonl"
        value = {"time": "2026-09-12T17:00:05+00:00", "kind": "worker_finished", "experiment_id": "demo", "completed": 1, "errors": 0}
        line = (json.dumps(value) + "\n").encode()
        events.write_bytes(line + b'{"partial":')
        terminal = json.loads(self.terminal_path.read_bytes())
        result = self.audit.operational_events(terminal)
        self.assertEqual(result["complete_prefix_sha256"], hashlib.sha256(line).hexdigest())
        self.assertEqual(result["captured_trailing_bytes"], len(b'{"partial":'))

    def test_rowless_failure_is_not_cleared_by_later_success(self):
        good = {"lease_id": "last", "status": "worker_exited", "worker_exit_code": 0, "incomplete": False}
        failed = {"lease_id": "earlier", "status": "worker_failed", "worker_exit_code": 1, "incomplete": True}
        self.assertEqual(mod.terminal_technical_failures([failed, good]),
                         [{"lease_id": "earlier", "status": "worker_failed", "error_type": None}])
        failed.update(status="supervisor_error", error_type="RuntimeError")
        self.assertEqual(len(mod.terminal_technical_failures([failed, good])), 1)

    def test_missing_only_interruption_remains_separately_reviewable(self):
        interrupted = {"lease_id": "earlier", "status": "supervisor_error", "error_type": "SupervisorInterrupted", "incomplete": True}
        missing = {"lease_id": "missing", "status": "worker_exited", "worker_exit_code": 0, "incomplete": True}
        good = {"lease_id": "last", "status": "worker_exited", "worker_exit_code": 0, "incomplete": False}
        self.assertEqual(mod.terminal_technical_failures([interrupted, missing, good]), [])

    def test_early_gate_failure_event_without_error_row_blocks(self):
        events = self.audit.campaign / "events.jsonl"
        records = [{"time": "2026-09-12T16:00:00+00:00", "kind": "technical_gate_failed", "experiment_id": "demo"},
                   {"time": "2026-09-12T17:00:05+00:00", "kind": "worker_finished", "experiment_id": "demo", "completed": 1, "errors": 0}]
        events.write_text("".join(json.dumps(v) + "\n" for v in records))
        with self.assertRaisesRegex(ValueError, "technical failure events remain unresolved"):
            self.audit.operational_events(json.loads(self.terminal_path.read_bytes()))

    def test_native_identity_requires_entire_sequence(self):
        a = self.audit
        conditions = ["baseline", "subspace_identity", "restoration_equivalent", "subspace_forward_0p5",
                      "projected_norm_matched_random_subspace_0p5", "subspace_forward_1", "projected_norm_matched_random_subspace_1"]
        a.spec = {"protocol_kind": "input_fact_subspace", "jobs": [{"condition": n, "pair_id": "pair", "seed": 42,
                  "recipient_token_ids": [1, 2], "metadata": {"paired_baseline_condition": "baseline"}} for n in conditions]}
        rows = [{"pair_id": "pair", "seed": 42, "condition": n, "patch_log": [{"delta_l2": [0.]}],
                 "forced_q": {"UP": 1.}, "forced_metrics": {}, "generated_token_ids": [1, 2, 3], "generated_text": "same",
                 "generation_status": "complete", "stop_reason": "return", "native_action": "UP",
                 "compiled_patch_audits": [{"projected_norm_audit": {"reference": "same"}}]} for n in conditions]
        self.assertEqual(len(a.controls(rows)), 2)
        rows[1]["generated_token_ids"] = [2, 2, 3]
        with self.assertRaisesRegex(ValueError, "whole-continuation"):
            a.controls(rows)


if __name__ == "__main__":
    unittest.main()
