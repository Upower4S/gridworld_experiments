"""Optional fact endpoint integration using a fake worker and temporary campaign."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.activation_causality.campaign import make_job, run_experiment
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
from src.activation_causality.fact_readout import score_fact_reports as actual_score_fact_reports
from tests.test_activation_causality_fact_readout import BoundaryTokenizer, make_engine as make_small_engine


class FakeWorkerEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.states = []
        self.forced_states = []
        self.native_states = []

    def validate(self):
        return {"passed": True, "model": {"head_mode": "full"}}

    def prefill(self, token_ids, patches=()):
        state = SimpleNamespace(token_ids=tuple(token_ids), patch_log=[], engine_id=id(self))
        self.states.append(state)
        return state

    def forced(self, state):
        self.forced_states.append(state)
        return {"q": [0.1, 0.2, 0.3, 0.4]}

    def generate(self, state, **kwargs):
        self.native_states.append(state)
        return {"native_action": "RIGHT", "generation_status": "complete", "generated_tokens": 1}


class FactRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.campaign = self.repo / "campaign"
        (self.campaign / "specs").mkdir(parents=True)
        model = self.repo / "offline_model"
        model.mkdir()
        (model / "config.json").write_text('{}')
        (self.campaign / "campaign.json").write_text(json.dumps({"model_path": str(model)}))
        self.runtime = {"versions": {"test": "1"}, "sources": {
            "src/activation_causality/engine.py": "engine-stable",
            "src/activation_causality/campaign.py": "worker-stable",
            "src/activation_causality/fact_readout.py": "fact-helper-a"}}
        self.runtime["sources"].update({name: "a" * 64 for name in ENGINE_DEPENDENCY_PATHS})
        self.engine = FakeWorkerEngine(model)
        self.probes = [{"id": "target", "question": "Before any move, is the agent carrying a key?",
                        "recipient_value": False, "counterfactual_value": True}]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("builtins.print"))
        self.stack.enter_context(patch("src.activation_causality.campaign.runtime_fingerprint", side_effect=lambda _: deepcopy(self.runtime)))
        self.stack.enter_context(patch("src.activation_causality.campaign.seconds_remaining", return_value=3600))
        self.stack.enter_context(patch("src.activation_causality.campaign.worker_lock", return_value=nullcontext()))
        self.stack.enter_context(patch("src.activation_causality.selection.validate_confirmation_ready"))
        self.engine_constructor = self.stack.enter_context(patch("src.activation_causality.engine.MLXInterventionEngine", return_value=self.engine))
        self.fact_scorer = self.stack.enter_context(patch("src.activation_causality.fact_readout.score_fact_reports",
                                                       return_value=[{"id": "target", "status": "scored", "source_state_unchanged": True}]))

    def spec(self, experiment_id, *, probes=False, natural=True):
        values = {"layout_id": "layout", "pair_id": "pair", "family": "wall", "partition": "discovery",
                  "condition": "baseline", "target_action": "RIGHT", "original_action": "LEFT", "seed": 42,
                  "recipient_token_ids": [1, 2, 3], "recipient_text": "unused", "donor_text": None,
                  "intervention": {"layers": []}, "natural": natural, "max_new_tokens": 8, "temperature": 0.7}
        if probes is not False:
            values["fact_probes"] = self.probes if probes is True else probes
        job = make_job(experiment_id, **values)
        spec = {"experiment_id": experiment_id, "partition": "discovery", "jobs": [job], "comparisons": []}
        (self.campaign / "specs" / f"{experiment_id}.json").write_text(json.dumps(spec))
        return job

    def run_experiment(self, experiment_id):
        return run_experiment(self.campaign, self.repo, experiment_id)

    def result(self, experiment_id):
        paths = list((self.campaign / "results" / experiment_id).glob("*.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_requested_endpoint_uses_original_state_and_preserves_native_output(self):
        self.spec("with_reports", probes=True)
        result = self.run_experiment("with_reports")
        self.assertEqual(result["new_results"], 1)
        row = self.result("with_reports")
        self.assertEqual(row["fact_readout_status"], "complete")
        self.assertEqual(row["fact_readout_implementation_sha256"], "fact-helper-a")
        self.assertEqual(row["native_action"], "RIGHT")
        self.assertEqual(row["generation_status"], "complete")
        args = self.fact_scorer.call_args.args
        self.assertIs(args[0], self.engine)
        self.assertIs(args[1], self.engine.states[0])
        self.assertEqual(args[2], self.probes)
        self.assertIs(self.engine.forced_states[0], self.engine.states[0])
        self.assertIs(self.engine.native_states[0], self.engine.states[0])

    def test_absent_endpoint_never_calls_helper_or_changes_identity_for_helper_edits(self):
        self.spec("ordinary")
        self.run_experiment("ordinary")
        key = self.result("ordinary")["execution_key"]
        self.assertNotIn("fact_reports", self.result("ordinary"))
        self.runtime["sources"]["src/activation_causality/fact_readout.py"] = "fact-helper-b"
        resumed = self.run_experiment("ordinary")
        self.assertEqual(resumed["new_results"], 0)
        self.assertEqual(self.result("ordinary")["execution_key"], key)
        self.fact_scorer.assert_not_called()

    def test_requested_endpoint_helper_change_rejects_resume(self):
        self.spec("with_reports", probes=True)
        self.run_experiment("with_reports")
        self.runtime["sources"]["src/activation_causality/fact_readout.py"] = "fact-helper-b"
        with self.assertRaisesRegex(ValueError, "execution code"):
            self.run_experiment("with_reports")

    def test_fact_errors_are_explicit_technical_failures(self):
        self.spec("failure", probes=True)
        self.fact_scorer.side_effect = RuntimeError("cache branch proof failed")
        summary = self.run_experiment("failure")
        row = self.result("failure")
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["fact_readout_status"], "error")
        self.assertIn("cache branch proof failed", row["fact_readout_error"]["error"])
        self.assertEqual(row["generation_status"], "technical_error")
        self.assertEqual(self.engine.native_states, [])

    def test_unavailable_probe_is_explicit_not_silently_omitted(self):
        self.spec("unavailable", probes=True, natural=False)
        self.fact_scorer.return_value = [{"id": "target", "status": "unavailable", "unavailable_reason": "BPE boundary",
                                         "source_state_unchanged": True}]
        self.run_experiment("unavailable")
        row = self.result("unavailable")
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["fact_readout_status"], "unavailable")
        self.assertEqual(row["fact_reports"][0]["unavailable_reason"], "BPE boundary")
        self.assertEqual(row["generation_status"], "not_requested")

    def test_helper_may_not_drop_a_requested_probe(self):
        self.spec("missing_report", probes=True)
        self.fact_scorer.return_value = []
        self.run_experiment("missing_report")
        row = self.result("missing_report")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["fact_readout_status"], "error")

    def test_actual_helper_unavailable_boundary_with_small_cached_engine(self):
        class UnsupportedBoolean(BoundaryTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == " true":
                    return [0]
                return super().encode(text, add_special_tokens)
        engine = make_small_engine(UnsupportedBoolean())
        engine.validate = lambda: {"passed": True, "model": {"head_mode": "full"}}
        self.engine_constructor.return_value = engine
        self.fact_scorer.side_effect = actual_score_fact_reports
        self.spec("actual_unavailable", probes=True, natural=False)
        self.run_experiment("actual_unavailable")
        row = self.result("actual_unavailable")
        self.assertEqual(row["fact_readout_status"], "unavailable")
        self.assertTrue(row["fact_reports"][0]["source_state_unchanged"])
        self.assertFalse(row["fact_reports"][0]["query_performed"])
        self.assertEqual(row["generation_status"], "not_requested")

    def test_actual_helper_exception_is_recorded_without_native_continuation(self):
        engine = make_small_engine()
        engine.validate = lambda: {"passed": True, "model": {"head_mode": "full"}}
        advance = engine._advance
        def reject_fact_suffix(state, ids):
            if ids[0] == 300:
                raise RuntimeError("fact suffix backend failure")
            return advance(state, ids)
        engine._advance = reject_fact_suffix
        self.engine_constructor.return_value = engine
        self.fact_scorer.side_effect = actual_score_fact_reports
        self.spec("actual_failure", probes=True)
        self.run_experiment("actual_failure")
        row = self.result("actual_failure")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["generation_status"], "technical_error")
        self.assertEqual(row["fact_readout_status"], "error")
        self.assertIn("fact suffix backend failure", row["fact_readout_error"]["error"])

    def test_empty_or_malformed_optional_field_rejected_before_loading(self):
        for index, value in enumerate(([], None, {}, "target")):
            identifier = f"bad_{index}"
            self.spec(identifier, probes=value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fact_probes"):
                self.run_experiment(identifier)
        self.engine_constructor.assert_not_called()

    def test_dependency_change_binds_execution_resume_and_engine_gate(self):
        for index, dependency in enumerate(ENGINE_DEPENDENCY_PATHS):
            with self.subTest(dependency=dependency):
                identifier = f"dependency_before_{index}"
                self.spec(identifier, probes=False)
                self.run_experiment(identifier)
                before = self.result(identifier)
                self.runtime["sources"][dependency] = "f" * 64
                calls = self.engine_constructor.call_count
                with self.assertRaisesRegex(ValueError, "execution code"):
                    self.run_experiment(identifier)
                self.assertEqual(self.engine_constructor.call_count, calls)
                after_id = f"dependency_after_{index}"
                self.spec(after_id, probes=False)
                self.run_experiment(after_id)
                after = self.result(after_id)
                self.assertNotEqual(before["execution_key"], after["execution_key"])
                self.assertNotEqual(before["engine_key"], after["engine_key"])
                gate = json.loads((self.campaign / "reviews" / f"engine_validation_{after['engine_key'][:16]}.json").read_text())
                self.assertEqual(gate["runtime"]["sources"][dependency], "f" * 64)
                self.runtime["sources"][dependency] = "a" * 64

    def test_missing_dependency_hash_rejects_before_engine_construction(self):
        self.spec("missing_dependency", probes=False)
        self.runtime["sources"].pop(ENGINE_DEPENDENCY_PATHS[-1])
        with self.assertRaisesRegex(ValueError, "engine dependency"):
            self.run_experiment("missing_dependency")
        self.engine_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
