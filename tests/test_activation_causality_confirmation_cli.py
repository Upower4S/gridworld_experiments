"""Exercise the actual staged analyze entrypoint with a four-slot freeze."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(STAGE), str(STAGE / "tests")]
from test_activation_causality_confirmation_contract import fixture  # noqa: E402
from src.activation_causality import analysis, campaign  # noqa: E402


class ConfirmationCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = STAGE / "scripts/30_activation_causality_campaign.py"
        spec = importlib.util.spec_from_file_location("staged_activation_cli", path)
        cls.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.driver)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        one, registry, spec = fixture()
        two, second, _ = fixture("contextual_input")
        for key in ("candidates", "bindings"):
            registry[key].update(second[key])
        registry["family_size"] = 4
        self.freeze = {"candidate_configurations": {one["candidate_id"]: one, two["candidate_id"]: two},
                       "confirmation_primary_family": registry}
        self.spec = {**spec, "freeze_id": "synthetic_freeze"}

    def invoke(self):
        def read(path):
            return self.freeze if path.parent.name == "freezes" else self.spec

        self.export = self.enterContext(patch.object(campaign, "export_results", return_value=self.path / "export.json"))
        self.collect = self.enterContext(patch.object(campaign, "collect_results", return_value=[]))
        self.report = self.enterContext(patch.object(analysis, "write_report", return_value={"synthetic": True}))
        self.enterContext(patch.object(self.driver, "read_json", side_effect=read))
        self.enterContext(patch.object(sys, "argv", ["campaign", "--campaign", str(self.path), "analyze",
                                                   "--experiment-id", self.spec["experiment_id"]]))
        with redirect_stdout(io.StringIO()):
            self.driver.main()

    def test_one_candidate_report_receives_entire_four_component_registry(self):
        self.invoke()
        kwargs = self.report.call_args.kwargs
        self.assertEqual(kwargs["confirmation_primary_family"], self.freeze["confirmation_primary_family"])
        self.assertEqual(kwargs["confirmation_primary_family"]["family_size"], 4)
        self.assertNotIn("confirmation_family_size", kwargs)
        self.assertEqual(kwargs["comparisons"], self.spec["comparisons"])
        self.collect.assert_called_once()

    def test_broken_new_freeze_stops_before_outcome_export_or_collection(self):
        del self.freeze["confirmation_primary_family"]
        with self.assertRaisesRegex(ValueError, "global registry"):
            self.invoke()
        self.export.assert_not_called()
        self.collect.assert_not_called()
        self.report.assert_not_called()

    def test_discovery_and_legacy_confirmation_keep_original_options(self):
        self.spec["partition"] = "discovery"
        self.invoke()
        self.assertNotIn("confirmation_primary_family", self.report.call_args.kwargs)
        self.assertNotIn("confirmation_family_size", self.report.call_args.kwargs)

    def test_genuinely_legacy_freeze_keeps_original_candidate_count(self):
        self.freeze = {"candidate_configurations": {"a": {"protocol_kind": "input_fact"}, "b": {}}}
        self.invoke()
        self.assertEqual(self.report.call_args.kwargs["confirmation_family_size"], 2)
        self.assertNotIn("confirmation_primary_family", self.report.call_args.kwargs)

    def call(self, arguments):
        with patch.object(sys, "argv", ["campaign", "--campaign", str(self.path), *arguments]), redirect_stdout(io.StringIO()):
            self.driver.main()

    def test_direct_run_delegates_lease_and_readiness_order_to_worker(self):
        from src.activation_causality import selection
        with patch.object(selection, "validate_confirmation_ready", side_effect=AssertionError("CLI readiness before worker lease")), \
             patch.object(campaign, "run_experiment", return_value={}) as run:
            self.call(["run", "--experiment-id", "candidate"])
        run.assert_called_once_with(self.path.resolve(), STAGE, "candidate", max_jobs=None, retry_errors=False)

    def test_supervised_queue_stops_at_incomplete_cutoff(self):
        from src.activation_causality import confirmation_supervisor
        with patch.object(confirmation_supervisor, "run_supervised_confirmation", side_effect=[
                {"status": "worker_exited", "incomplete": False}, {"status": "cutoff_terminated", "incomplete": True}]) as run:
            self.call(["run-confirmation", "--experiments", "primary,alternate,later"])
        self.assertEqual([call.args[2] for call in run.call_args_list], ["primary", "alternate"])

    def test_new_freeze_and_registration_have_explicit_routes(self):
        from src.activation_causality import confirmation_freeze
        with patch.object(confirmation_freeze, "freeze_confirmation_plan", return_value={}) as publish:
            self.call(["freeze-co-primary", "--plan", "prepared.json", "--review", "review.json"])
        publish.assert_called_once_with(self.path.resolve(), STAGE, Path("prepared.json"), Path("review.json"))
        with patch.object(confirmation_freeze, "register_frozen_plan", return_value=[]) as register:
            self.call(["register-co-primary", "--freeze-id", "frozen"])
        register.assert_called_once_with(self.path.resolve(), STAGE, "frozen")

    def test_preparation_route_uses_fixed_production_defaults(self):
        from src.activation_causality import confirmation_planning
        with patch.object(confirmation_planning, "prepare_plan", return_value={}) as prepare:
            self.call(["plan-co-primary", "--decision", "decision.json", "--freeze-id", "frozen"])
        prepare.assert_called_once_with(self.path.resolve(), STAGE, Path("decision.json"), freeze_id="frozen")

    def test_supervised_queue_rejects_duplicate_ids_before_launch(self):
        from src.activation_causality import confirmation_supervisor
        with patch.object(confirmation_supervisor, "run_supervised_confirmation") as run, self.assertRaises(ValueError):
            self.call(["run-confirmation", "--experiments", "candidate,candidate"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
