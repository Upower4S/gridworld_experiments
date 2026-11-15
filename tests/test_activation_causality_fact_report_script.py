"""CPU-only guards and weighting checks for the standalone factual reporter."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.activation_causality.environments import generate_pairs
from src.activation_causality.fact_readout import _digest
from src.activation_causality.fact_readout_protocol import build_fact_readout_spec
from src.activation_causality.input_sites import build_input_site_spec
from src.activation_causality.registry import fingerprint
from tests.test_activation_causality_input_sites import FactTokenizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/33_activation_causality_fact_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location("fact_report_script", SCRIPT)
reporter = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(reporter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FactReportScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pair = generate_pairs(1, 20260912, "discovery", families=("key_possession",))[0]
        cls.source_template = build_input_site_spec([pair], FactTokenizer(), "source", layers=(0,), seeds=(42, 43))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.campaign = Path(self.temp.name)
        self.source = deepcopy(self.source_template)
        self.source_path = self.campaign / "specs/source.json"
        source_sha = write_json(self.source_path, self.source)
        self.spec = build_fact_readout_spec(self.source, "reports")
        self.spec["source_spec_sha256"] = source_sha
        self.spec_path = self.campaign / "specs/reports.json"
        spec_sha = write_json(self.spec_path, self.spec)
        self.result_dir = self.campaign / "results/reports"
        self.rows = {}
        for job in self.spec["jobs"]:
            row = {key: deepcopy(value) for key, value in job.items() if key not in reporter.OMITTED_RESULT_FIELDS}
            reports = []
            for probe in job["fact_probes"]:
                if probe["id"] == "target_initial_fact":
                    probability = 0.8 if job["condition"] == "counterfactual_text" else 0.7 if job["condition"] == "factspan_layer0" else 0.3 if job["condition"] == "random_factspan_layer0" else 0.2
                else:
                    probability = 0.9
                truth = str(probe["counterfactual_value"]).lower()
                q = {truth: probability, "false" if truth == "true" else "true": 1 - probability}
                base = {"token_ids_sha256": _digest(job["recipient_token_ids"]), "n_tokens": len(job["recipient_token_ids"]),
                        "state_sha256": "a" * 64, "cache_sha256": "b" * 64, "next_logits_sha256": "c" * 64}
                reports.append({**deepcopy(probe), "probe_sha256": _digest(probe), "status": "scored", "query_performed": True,
                    "source_state_unchanged": True, "base_state": base, "source_state_after": deepcopy(base),
                    "authoritative_prefix_token_ids_sha256": _digest(job["recipient_token_ids"]),
                    "boundary_verified": True, "prefix_reencoded": False, "q": q,
                    "recipient_truth_q": q[str(probe["recipient_value"]).lower()], "counterfactual_truth_q": probability,
                    "candidate_mass": 0.001, "candidate_global_probabilities": {key: value * 0.001 for key, value in q.items()}})
            row.update(status="complete", generation_status="not_requested", native_action=None,
                spec_sha256=spec_sha, engine_key="a" * 64, execution_key="b" * 64,
                fact_readout_implementation_sha256="c" * 64, inference_configuration={"head_mode": "full"},
                input_token_sha256=fingerprint(job["recipient_token_ids"]), fact_probes_sha256=fingerprint(job["fact_probes"]),
                fact_reports=reports, fact_readout_n_requested=2, fact_readout_n_scored=2,
                fact_readout_n_unavailable=0, fact_readout_status="complete")
            self.rows[job["job_id"]] = row
            self.save(row)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, row):
        write_json(self.result_dir / f"{row['job_id']}.json", row)

    def load(self):
        return reporter.load_report(self.campaign, "reports")

    def baseline(self):
        return next(row for row in self.rows.values() if row["condition"] == "baseline")

    def test_complete_report_means_paired_deltas_and_no_native_claim(self):
        report = self.load()
        self.assertEqual(report["n_planned_jobs"], 12)
        self.assertEqual(report["n_scored_probes"], 24)
        self.assertEqual(report["n_unavailable_probes"], 0)
        fact = next(row for row in report["condition_summaries"] if row["condition"] == "factspan_layer0" and row["probe_id"] == "target_initial_fact")
        self.assertAlmostEqual(fact["truth_probability"]["mean"], 0.7)
        self.assertAlmostEqual(fact["candidate_mass"]["mean"], 0.001)
        comparison = next(row for row in report["paired_comparisons"] if row["treatment"] == "factspan_layer0" and row["control"] == "baseline" and row["probe_id"] == "target_initial_fact")
        self.assertAlmostEqual(comparison["overall"]["delta"]["mean"], 0.5)
        self.assertEqual(comparison["overall"]["n_available_pair_seed"], 2)
        self.assertFalse(report["definitions"]["native_action_filters"])
        self.assertFalse(report["definitions"]["semantic_claim"])
        self.assertEqual(len(report["source_result_files"]), 12)

    def test_hierarchical_weighting_is_equal_layout_after_seed_and_pair(self):
        rows = [{"layout_id": "A", "pair_id": "one", "v": 0}, {"layout_id": "A", "pair_id": "one", "v": 1},
                {"layout_id": "A", "pair_id": "two", "v": 0}, {"layout_id": "A", "pair_id": "two", "v": 0},
                {"layout_id": "B", "pair_id": "three", "v": 1}, {"layout_id": "B", "pair_id": "three", "v": None}]
        result = reporter.weighted_summary(rows, "v")
        self.assertAlmostEqual(result["mean"], 0.625)
        self.assertEqual(result["n_available_layouts"], 2)
        self.assertEqual([row["mean"] for row in result["per_layout"]], [0.25, 1.0])

    def test_explicit_unavailable_preserves_denominators_and_paired_unavailability(self):
        row = self.baseline()
        row["fact_reports"][0].update(status="unavailable", unavailable_reason="unsupported_boolean_boundary", query_performed=False)
        row.update(fact_readout_status="partial", fact_readout_n_scored=1, fact_readout_n_unavailable=1)
        self.save(row)
        report = self.load()
        self.assertEqual(report["n_unavailable_probes"], 1)
        baseline = next(value for value in report["condition_summaries"] if value["condition"] == "baseline" and value["probe_id"] == "target_initial_fact")
        self.assertEqual((baseline["n_available"], baseline["n_planned"]), (1, 2))
        comparison = next(value for value in report["paired_comparisons"] if value["treatment"] == "factspan_layer0" and value["control"] == "baseline" and value["probe_id"] == "target_initial_fact")
        self.assertEqual(comparison["overall"]["n_unavailable_pair_seed"], 1)
        self.assertAlmostEqual(comparison["overall"]["delta"]["mean"], 0.5)
        self.assertIsNone(next(value for value in comparison["paired_observations"] if not value["available"])["delta"])

    def test_missing_and_unexpected_results_reject_partial_cohort(self):
        row = self.baseline()
        path = self.result_dir / f"{row['job_id']}.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()
        self.save(row)
        write_json(self.result_dir / "unexpected.json", row)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.load()

    def test_retired_source_or_derivative_is_rejected(self):
        for experiment in ("source", "reports"):
            path = self.campaign / "reviews" / f"{experiment}_retired.json"
            write_json(path, {"retired": True})
            with self.assertRaisesRegex(ValueError, "retired"):
                self.load()
            path.unlink()

    def test_raw_source_file_and_semantic_fingerprints_are_both_checked(self):
        self.source_path.write_text(self.source_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "source file hash"):
            self.load()
        self.spec["source_spec_sha256"] = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self.spec["source_spec_fingerprint"] = "wrong"
        write_json(self.spec_path, self.spec)
        with self.assertRaisesRegex(ValueError, "specification fingerprint"):
            self.load()

    def test_source_job_map_and_registered_job_ids_are_checked(self):
        key = next(iter(self.spec["source_job_map"]))
        self.spec["source_job_map"][key]["source_job_fingerprint"] = "wrong"
        write_json(self.spec_path, self.spec)
        with self.assertRaisesRegex(ValueError, "source_job_map"):
            self.load()
        self.spec["source_job_map"][key]["source_job_fingerprint"] = fingerprint(next(job for job in self.source["jobs"] if job["job_id"] == key))
        self.spec["jobs"][0]["recipient_positions"] = [0]
        write_json(self.spec_path, self.spec)
        with self.assertRaisesRegex(ValueError, "Derivative job ID"):
            self.load()

    def test_result_spec_hash_execution_identity_and_frozen_fields_guard(self):
        original = deepcopy(self.baseline())
        for mutation, message in (({"spec_sha256": "bad"}, "specification hash"),
                                  ({"execution_key": "d" * 64}, "Mixed execution"),
                                  ({"seed": 99}, "frozen job field seed"),
                                  ({"input_token_sha256": "bad"}, "Input-token"),
                                  ({"fact_probes_sha256": "bad"}, "factual-probe")):
            row = {**deepcopy(original), **mutation}
            self.save(row)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                self.load()
            self.save(original)

    def test_errors_missing_probes_and_unproven_state_preservation_rejected(self):
        original = deepcopy(self.baseline())
        mutations = [({"status": "error"}, "complete forced-only"),
                     ({"fact_reports": original["fact_reports"][:1]}, "probe results"),
                     ({"fact_readout_status": "pending"}, "status/count")]
        for mutation, message in mutations:
            self.save({**deepcopy(original), **mutation})
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                self.load()
        row = deepcopy(original)
        row["fact_reports"][0]["source_state_unchanged"] = False
        self.save(row)
        with self.assertRaisesRegex(ValueError, "preservation proof"):
            self.load()

    def test_probe_hash_truth_probability_and_base_token_guards(self):
        original = deepcopy(self.baseline())
        for mutation, message in (({"probe_sha256": "bad"}, "probe hash"),
                                  ({"counterfactual_truth_q": 0.99}, "frozen Boolean truth"),
                                  ({"candidate_mass": 0.9}, "candidate mass"),
                                  ({"base_state": {"token_ids_sha256": "bad", "n_tokens": 1}}, "token provenance"),
                                  ({"status": "unavailable", "query_performed": False}, "explicit reason")):
            row = deepcopy(original)
            row["fact_reports"][0].update(mutation)
            self.save(row)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                self.load()
            self.save(original)

    def test_json_markdown_output_and_cli_auto_selection(self):
        report = reporter.load_report(self.campaign)
        output = self.campaign / "reports" / "diagnostic"
        json_path, markdown_path = reporter.write_report(report, output)
        self.assertEqual(json.loads(json_path.read_text()), report)
        self.assertIn("Available / planned", markdown_path.read_text())
        self.assertIn("do not establish native action", markdown_path.read_text())
        run = subprocess.run([sys.executable, str(SCRIPT), "--campaign", str(self.campaign), "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["jobs"], 12)


if __name__ == "__main__":
    unittest.main()
