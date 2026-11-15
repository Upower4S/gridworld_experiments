"""CPU orchestration and prerequisite guards; no production model/source reads."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.activation_causality import confirmation_planning as planning
from src.activation_causality import confirmation_freeze as freeze
from src.activation_causality import confirmation_sampling as sampling
from src.activation_causality.registry import file_sha256, fingerprint, immutable_json, read_json
from test_activation_causality_frozen_contextual_input import Tokenizer, configuration
from src.activation_causality.frozen_contextual_input import _configuration

STAGE = Path(__file__).resolve().parents[1]


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.campaign = Path(self.temp.name).resolve()
        for folder in ("specs", "datasets", "reviews"):
            (self.campaign / folder).mkdir()
        self.deadline = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        immutable_json(self.campaign / "campaign.json", {"deadline": self.deadline})
        self.history = self.campaign / "history.parquet"
        self.history.write_bytes(b"CPU historical reader fixture")
        self.history_audit = {"source": str(self.history), "rows_checked": 0, "layout_fingerprints": 0, "status": "complete"}
        self.decision_path = self.campaign / "reviews/decision.json"
        immutable_json(self.decision_path, {"decision": "fixture"})
        self.prerequisite = self.campaign / "reviews/prerequisite.json"
        immutable_json(self.prerequisite, {"CPU_only": True})
        self.cid = "contextual_l0"
        self.descriptor = {"profile_id": "contextual_l0", "protocol_kind": "contextual_input", "source_experiment_id": "contextual_discovery",
                           "source_spec_sha256": "4" * 64, "layer": 0, "site": "resid_post", "selected_dose": 1.0, "rank": None, "family": None}
        self.tokenizer = Tokenizer()
        self.fixed = {"campaign_sha256": file_sha256(self.campaign / "campaign.json"), "campaign_deadline": self.deadline,
            "inference_cutoff": freeze.inference_cutoff(self.deadline),
            "implementation": {"src/activation_causality/confirmation_planning.py": file_sha256(STAGE / "src/activation_causality/confirmation_planning.py")},
            "activation_representation": {}, "model_identity": {"identity_sha256": "a" * 64},
            "worker_entrypoint_sha256": file_sha256(STAGE / "scripts/30_activation_causality_campaign.py"),
            "scientific_selection": planning._ref(self.decision_path), "scientific_selection_independent_review": planning._ref(self.prerequisite),
            "technical_preflight": planning._ref(self.prerequisite), "audit_bindings": {}, "descriptors": {self.cid: self.descriptor},
            "prior": freeze.prior_exclusion_snapshot(self.campaign),
            "historical_exclusion": {"status": "checked", "source_path": str(self.history), "source_sha256": file_sha256(self.history),
                                     "source_exclusion_provenance": self.history_audit}, "budget_seconds_per_native_job": 295.0}
        self.battery = {"sources": {"contextual_discovery": {}}, "minimum_budget_seconds_per_native_job": 295.0}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.inputs = self.stack.enter_context(patch.object(planning, "_inputs", side_effect=lambda *args: (deepcopy(self.fixed), self.battery, self.tokenizer)))
        self.stack.enter_context(patch.object(freeze, "candidate_configuration", side_effect=self.derive))
        self.stack.enter_context(patch.object(freeze, "_recheck_external"))
        self.stack.enter_context(patch("src.activation_causality.environments.load_historical_layout_ids", return_value=(set(), self.history_audit)))
        self.stack.enter_context(patch.object(freeze, "_external_context", return_value=({"battery": self.battery, "decision": {
            "selected_scientific_profile_ids": {"contextual_input": "contextual_l0", "fixed_family_subspace": None},
            "profiles": [{"profile_id": "contextual_l0", "descriptor": self.descriptor}]}}, self.tokenizer)))

    def derive(self, repo, descriptor, source, *, candidate_id, n_layouts, dataset_seed, tokenizer, joint):
        config = configuration(n_layouts, descriptor["layer"], tokenizer, joint)
        config.update(candidate_id=candidate_id, dataset_seed=dataset_seed, protocol_kind="contextual_input", source_profile_descriptor=deepcopy(descriptor))
        rule = freeze.eligibility_rule(repo, descriptor, joint=joint)
        for binding in config["primary_comparisons"].values():
            binding["eligibility_rule"] = rule
        return _configuration(config, tokenizer)

    def run_plan(self, **kwargs):
        return planning.prepare_plan(self.campaign, STAGE, self.decision_path, freeze_id="fixed", **kwargs)

    def test_real_cpu_sampling_complete_plan_replays_and_resume_preserves_every_byte(self):
        result = self.run_plan()
        self.assertEqual(result["status"], "complete_unregistered", result)
        self.assertEqual(result["planned_experiment_ids"], ["fixed_contextual_primary"])
        plan = read_json(Path(result["plan"]["path"]))
        self.assertEqual(plan["confirmation_primary_family"]["family_size"], 2)
        self.assertEqual(plan["candidate_configurations"][self.cid]["n_layouts"], 32)
        self.assertFalse(plan["supplements"][self.cid])
        self.assertEqual(result["budget"]["jobs"], 480)
        state = read_json(self.campaign / "planning/final_sampling_v1.json")
        self.assertFalse(state["clock_override_used"])
        self.assertEqual(state["identity"]["dataset_seed"], planning.DATASET_SEED)
        self.assertEqual(len(state["candidates"][self.cid]["accepted_pair_ids"]), 32)
        original = {p: p.read_bytes() for p in (self.campaign / "planning").glob("*.json")}
        with patch.object(planning, "generate_pairs", side_effect=AssertionError("must not resample complete state")):
            repeated = self.run_plan()
        self.assertEqual(repeated["plan"], result["plan"])
        self.assertEqual(repeated["sampling_state"], result["sampling_state"])
        self.assertEqual(repeated["budget"]["jobs"], result["budget"]["jobs"])
        self.assertTrue(all(p.read_bytes() == raw for p, raw in original.items()))
        self.assertFalse(list((self.campaign / "specs").iterdir()))
        self.assertFalse(list((self.campaign / "datasets").iterdir()))
        self.assertFalse((self.campaign / "freezes").exists())

    def test_lexical_supplement_repeats_exact_fixed32_parent_subset(self):
        self.fixed["budget_seconds_per_native_job"] = 145.0
        self.battery["minimum_budget_seconds_per_native_job"] = 145.0
        result = self.run_plan()
        self.assertEqual(result["status"], "complete_unregistered", result)
        plan = read_json(Path(result["plan"]["path"]))
        self.assertEqual(plan["candidate_configurations"][self.cid]["n_layouts"], 32)
        self.assertTrue(plan["supplements"][self.cid])
        self.assertEqual(set(result["planned_experiment_ids"]), {"fixed_contextual_primary", "fixed_contextual_alternate"})
        primary = read_json(Path(plan["planned_experiments"]["fixed_contextual_primary"]["spec_path"]))
        alternate = read_json(Path(plan["planned_experiments"]["fixed_contextual_alternate"]["spec_path"]))
        self.assertEqual(len(primary["jobs"]), 480)
        self.assertEqual(len(alternate["jobs"]), 480)
        self.assertEqual({job["pair_id"] for job in primary["jobs"]}, {job["pair_id"] for job in alternate["jobs"]})
        self.assertTrue(all(c["confirmatory"] is False for c in alternate["comparisons"]))

    def test_two_profiles_share_alias_reservations_and_keep_four_primary_slots(self):
        from test_activation_causality_frozen_subspace import basis, configuration as subspace_config
        from src.activation_causality.frozen_subspace import _configuration as normalize_subspace
        from src.activation_causality.lexical_generalization import CANONICAL, renderer_manifest
        metadata = basis()
        descriptor = {"profile_id": "subspace_wall_l0_r1_dose1", "protocol_kind": "input_fact_subspace",
            "source_experiment_id": "native_source", "source_spec_sha256": "4" * 64, "layer": 0,
            "site": "resid_post", "selected_dose": 1.0, "rank": 1, "family": "wall"}
        self.fixed["descriptors"]["wall_subspace"] = descriptor
        self.fixed["budget_seconds_per_native_job"] = 150.0
        self.battery["minimum_budget_seconds_per_native_job"] = 150.0
        self.battery["sources"]["native_source"] = {}
        def derive(repo, selected, source, **kwargs):
            if selected["protocol_kind"] == "contextual_input":
                return self.derive(repo, selected, source, **kwargs)
            config = subspace_config(metadata, kwargs["n_layouts"], 1.0, self.tokenizer)
            config.update(candidate_id=kwargs["candidate_id"], dataset_seed=kwargs["dataset_seed"],
                protocol_kind="input_fact_subspace", source_profile_descriptor=descriptor,
                renderer_id=CANONICAL, renderer_binding=renderer_manifest(CANONICAL),
                require_joint_renderer_eligibility=kwargs["joint"], basis_metadata=metadata)
            rule = freeze.eligibility_rule(repo, selected, joint=kwargs["joint"])
            for binding in config["primary_comparisons"].values():
                binding["eligibility_rule"] = rule
            return normalize_subspace(config, metadata)[0]
        context = {"battery": self.battery, "decision": {
            "selected_scientific_profile_ids": {"contextual_input": "contextual_l0", "fixed_family_subspace": descriptor["profile_id"]},
            "profiles": [{"profile_id": d["profile_id"], "descriptor": d} for d in self.fixed["descriptors"].values()]}}
        with patch.object(freeze, "candidate_configuration", side_effect=derive), \
                patch.object(freeze, "_external_context", return_value=(context, self.tokenizer)), \
                patch("src.activation_causality.confirmation_basis.validate_confirmation_basis", return_value=(metadata, {"CPU_fixture": True})):
            result = self.run_plan()
        self.assertEqual(result["status"], "complete_unregistered", result)
        plan = read_json(Path(result["plan"]["path"]))
        self.assertEqual(plan["confirmation_primary_family"]["family_size"], 4)
        self.assertEqual(result["budget"]["jobs"], 928)
        self.assertEqual(set(result["planned_experiment_ids"]), {"fixed_contextual_primary", "fixed_wall_subspace_primary"})
        values = {}
        for eid, item in plan["planned_experiments"].items():
            spec = read_json(Path(item["spec_path"]))
            values[eid] = set().union(*(freeze.layout_aliases(job) for job in spec["jobs"]))
        left, right = values.values()
        self.assertFalse(left & right)
        state = read_json(self.campaign / "planning/final_sampling_v1.json")
        self.assertEqual(len(state["identity"]["candidates"]), 2)
        self.assertEqual(state["identity"]["dataset_seed"], planning.DATASET_SEED)

    def test_seed_and_public_override_rejected_before_inputs_or_clock(self):
        for seed in (20260913, True, 20260912.0):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "single prospective dataset seed"):
                self.run_plan(dataset_seed=seed)
        with self.assertRaises(TypeError):
            self.run_plan(now_utc="2000-01-01")
        self.inputs.assert_not_called()
        self.assertFalse((self.campaign / "planning/final_sampling_v1.json").exists())

    def test_prerequisite_failure_never_initializes_sampler_and_is_terminal(self):
        self.inputs.side_effect = ValueError("real independent preflight absent")
        with patch.object(sampling, "initialize_sampling_plan", side_effect=AssertionError("must not initialize")):
            result = self.run_plan()
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertEqual(result["phase"], "authenticate_prerequisites")
        self.inputs.reset_mock()
        self.assertEqual(self.run_plan(), result)
        self.inputs.assert_not_called()

    def test_interrupted_reserved_batch_is_preserved_and_never_retried(self):
        with patch.object(planning, "generate_pairs", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_plan()
        path = self.campaign / "planning/final_sampling_v1.json"
        before = read_json(path)
        self.assertIsNotNone(before["in_progress"])
        with patch.object(planning, "generate_pairs", side_effect=AssertionError("cannot rerun reserved batch")):
            result = self.run_plan()
        self.assertEqual(result["status"], "terminal_incomplete")
        after = read_json(path)
        self.assertEqual(before["started_utc"], after["started_utc"])
        self.assertEqual(before["shared_sampling_deadline_utc"], after["shared_sampling_deadline_utc"])
        self.assertEqual(after["terminal_reason"], "interrupted_reserved_batch_not_retried")

    def test_resume_cannot_change_source_or_allocation_and_keeps_clock(self):
        with patch.object(planning, "generate_pairs", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_plan()
        path = self.campaign / "planning/final_sampling_v1.json"
        before = path.read_bytes()
        self.fixed["budget_seconds_per_native_job"] = 1.0
        result = self.run_plan()
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertIn("changed preparation identity", result["error"])
        self.assertEqual(path.read_bytes(), before)

    def test_allocation_boundary_drift_fails_without_generation_or_reset(self):
        first = {"n_layouts": 32, "supplements": {self.cid: False}, "budget": {}, "lexical_omissions_due_to_minimum_budget": [self.cid]}
        second = {**first, "n_layouts": 64}
        with patch.object(planning, "_allocation", side_effect=[first, second]), \
                patch.object(planning, "generate_pairs", side_effect=AssertionError("no generation after budget boundary")):
            result = self.run_plan()
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertIn("allocation boundary", result["error"])
        state = read_json(self.campaign / "planning/final_sampling_v1.json")
        self.assertEqual(state["attempts"], [])
        self.assertFalse(state["clock_override_used"])

    def test_runtime_change_between_init_and_advance_prevents_generation(self):
        actual = planning._unchanged
        calls = []
        def changing(repo, fixed):
            calls.append(None)
            if len(calls) == 4:
                raise ValueError("runtime changed before advance")
            return actual(repo, fixed)
        with patch.object(planning, "_unchanged", side_effect=changing), \
                patch.object(planning, "generate_pairs", side_effect=AssertionError("no generation after runtime drift")):
            result = self.run_plan()
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertIn("runtime changed", result["error"])

    def test_fixed_experiment_ids_and_no_backfill_from_unqualified_profiles(self):
        decision = {"profiles": [{"profile_id": "contextual_l0", "descriptor": self.descriptor},
            {"profile_id": "wall", "descriptor": {"protocol_kind": "input_fact_subspace", "family": "wall"}}],
            "selected_scientific_profile_ids": {"contextual_input": "contextual_l0", "fixed_family_subspace": None}}
        descriptors = planning._descriptors(decision)
        self.assertEqual(set(descriptors), {self.cid})
        self.assertEqual(planning._experiment_ids("fixed", descriptors)[self.cid],
                         {"primary": "fixed_contextual_primary", "lexical_supplement": "fixed_contextual_alternate"})
        decision["selected_scientific_profile_ids"]["contextual_input"] = None
        with self.assertRaisesRegex(ValueError, "no scientifically qualified"):
            planning._descriptors(decision)

class PreflightPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.campaign = self.repo / "campaign"
        self.eid = "contextual_full_write_forced_preflight_v1"
        from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
        names = set(ENGINE_DEPENDENCY_PATHS) | {"src/activation_causality/" + name + ".py" for name in
                ("engine", "campaign", "full_write_audit", "subspace_audit", "basis_runtime", "model_identity")}
        self.runtime = {"sources": {}}
        for name in names:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("CPU-only provenance fixture\n")
            self.runtime["sources"][name] = file_sha256(path)
        self.inference = deepcopy(self.runtime["sources"])
        for name in ("src/activation_causality/analysis.py", "src/activation_causality/selection.py", "scripts/30_activation_causality_campaign.py"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original non-inference CPU source\n")
            self.runtime["sources"][name] = file_sha256(path)
        source_path = self.campaign / "specs/source.json"
        immutable_json(source_path, {"source": "CPU-only fixture"})
        load_path = self.campaign / "reviews/load.json"
        immutable_json(load_path, {"passed": True, "CPU_only": True})
        roles = ("baseline", "matched", "random", "natural", "natural_norm_random", "identity")
        jobs = []
        for index in range(72):
            role = roles[index % 6]
            value = {"experiment_id": self.eid, "condition": f"fixture{index}_{role}", "partition": "discovery",
                     "natural": False, "intervention": {"layers": [] if role == "baseline" else [0]}, "index": index}
            if role in ("matched", "natural", "natural_norm_random", "identity"):
                value["intervention"]["audit_full_write_norm"] = True
            jobs.append({"job_id": fingerprint(value)[:24], **value})
        spec_path = self.campaign / "specs" / (self.eid + ".json")
        immutable_json(spec_path, {"experiment_id": self.eid, "technical_preflight_only": True, "scientific_use": False,
                                  "sources": {str(source_path): file_sha256(source_path)}, "jobs": jobs})
        self.spec_sha = file_sha256(spec_path)
        inputs = {str(p.relative_to(self.repo)): file_sha256(p) for p in (spec_path, source_path, load_path)}
        self.paths = []
        for job in jobs:
            role = roles[job["index"] % 6]
            row = {**job, "status": "complete", "spec_sha256": self.spec_sha, "native_action": None,
                   "generation_status": "not_requested", "inference_source_hashes": self.inference,
                   "model_snapshot_identity_sha256": "a" * 64, "model_load_audit_path": str(load_path),
                   "model_load_audit_sha256": file_sha256(load_path), "full_write_norm_audit_status": "passed",
                   "random_orientation_audits": [{}] if role in ("random", "natural_norm_random") else [],
                   "patch_log": [{"delta_l2": [0.0]}] if role == "identity" else []}
            path = self.campaign / "results" / self.eid / (job["job_id"] + ".json")
            immutable_json(path, row)
            inputs[str(path.relative_to(self.repo))] = file_sha256(path)
            self.paths.append(path)
        self.archive_root = self.campaign / "source_snapshots/technical_preflight_promotion_20260912"
        for name, digest in self.runtime["sources"].items():
            archived = self.archive_root / "after" / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes((self.repo / name).read_bytes())
            inputs[name] = digest
            inputs[str(archived.relative_to(self.repo))] = digest
        self.manifest_path = self.archive_root / "manifest.json"
        immutable_json(self.manifest_path, {"format": "limited_technical_preflight_promotion_v1",
            "after_files": self.runtime["sources"], "registered_preflight_spec_sha256": self.spec_sha})
        inputs[str(self.manifest_path.relative_to(self.repo))] = file_sha256(self.manifest_path)
        self.audit_path = self.campaign / "reviews/contextual_full_write_forced_preflight_independent_audit_v1.json"
        immutable_json(self.audit_path, {"format": "full_write_preflight_independent_audit_v1", "passed": True,
            "experiment_id": self.eid, "spec_sha256": self.spec_sha, "n_results": 72,
            "technical_status_counts": {"complete": 72}, "native_queries": 0, "full_write_audits_passed": 48,
            "random_orientation_records": 24, "identity_rows_exact_zero": 12, "input_files_sha256": inputs,
            "inference_source_hashes": self.inference, "model_snapshot_identity_sha256": "a" * 64})
        for name, value in (("PREFLIGHT_PROMOTION_SHA256", file_sha256(self.manifest_path)),
                            ("PREFLIGHT_AUDIT_SHA256", file_sha256(self.audit_path))):
            patcher = patch.object(planning, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self):
        return planning._preflight(self.campaign, self.repo, self.runtime)

    def test_complete_synthetic_prerequisite_binds_every_source_raw_and_load_file(self):
        ref = self.call()
        self.assertEqual(ref["sha256"], file_sha256(self.audit_path))
        self.assertEqual(ref["model_snapshot_identity_sha256"], "a" * 64)

    def test_non_inference_promotion_uses_only_exact_historical_archive(self):
        before = self.call()
        for name in ("src/activation_causality/analysis.py", "src/activation_causality/selection.py", "scripts/30_activation_causality_campaign.py"):
            (self.repo / name).write_text("prospective reviewed new source bytes\n")
            self.runtime["sources"][name] = file_sha256(self.repo / name)
        after = self.call()
        self.assertEqual(after, before)
        self.assertEqual(after["current_inference_source_hashes"], self.inference)
        self.assertEqual(after["historical_source_archive"]["manifest_sha256"], file_sha256(self.manifest_path))
        archived = self.archive_root / "after/src/activation_causality/analysis.py"
        archived.write_text("changed historical source\n")
        with self.assertRaisesRegex(ValueError, "source archive changed"):
            self.call()

    def test_current_inference_bytes_cannot_hide_behind_unchanged_runtime_or_archive(self):
        (self.repo / "src/activation_causality/engine.py").write_text("changed actual inference source\n")
        with self.assertRaisesRegex(ValueError, "current preflight inference source bytes changed"):
            self.call()

    def test_accepted_audit_manifest_and_fixed_spec_hash_are_pinned(self):
        original = self.audit_path.read_bytes()
        self.audit_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "accepted independent preflight audit bytes"):
            self.call()
        self.audit_path.write_bytes(original)
        manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_path.write_bytes(manifest_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "promotion manifest changed"):
            self.call()
        self.manifest_path.write_bytes(manifest_bytes)
        spec_path = self.campaign / "specs" / (self.eid + ".json")
        spec_path.write_bytes(spec_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "specification differs from accepted promotion"):
            self.call()

    def test_non_source_reference_never_gets_historical_fallback(self):
        load_path = self.campaign / "reviews/load.json"
        load_path.write_text("changed load audit\n")
        with self.assertRaisesRegex(ValueError, "preflight input changed"):
            self.call()

    def test_partial_inventory_fails_before_any_raw_or_audit_content_read(self):
        self.paths[-1].unlink()
        actual = freeze._read
        def guard(path):
            if "results" in path.parts or path == self.audit_path:
                raise AssertionError("partial content must not be read")
            return actual(path)
        with patch.object(freeze, "_read", side_effect=guard), self.assertRaisesRegex(ValueError, "canonical preflight filenames"):
            self.call()

    def test_mutated_row_and_extra_file_reject(self):
        path = self.paths[0]
        row = read_json(path)
        row["native_action"] = "RIGHT"
        path.write_text(__import__("json").dumps(row))
        with self.assertRaisesRegex(ValueError, "raw status/native"):
            self.call()
        extra = path.parent / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "canonical preflight filenames"):
            self.call()

    def test_missing_source_load_hash_or_changed_runtime_reject(self):
        audit = read_json(self.audit_path)
        audit["input_files_sha256"] = {name: digest for name, digest in audit["input_files_sha256"].items() if not name.endswith("/load.json")}
        self.audit_path.write_text(__import__("json").dumps(audit))
        with patch.object(planning, "PREFLIGHT_AUDIT_SHA256", file_sha256(self.audit_path)), \
                self.assertRaisesRegex(ValueError, "omits a spec/raw input"):
            self.call()
        self.runtime["sources"]["src/activation_causality/engine.py"] = "f" * 64
        with patch.object(planning, "PREFLIGHT_AUDIT_SHA256", file_sha256(self.audit_path)), \
                self.assertRaisesRegex(ValueError, "inference runtime differs"):
            self.call()



if __name__ == "__main__":
    unittest.main()
