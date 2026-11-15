"""CPU-only checks for complete-cohort forced factual-readout derivatives."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from src.activation_causality.campaign import make_job
from src.activation_causality.environments import generate_pairs
from src.activation_causality.fact_readout import CLAIM_SCOPE, build_fact_probes
from src.activation_causality.fact_readout_protocol import build_fact_readout_spec
from src.activation_causality.input_sites import build_input_site_spec
from src.activation_causality.registry import fingerprint
from tests.test_activation_causality_input_sites import FactTokenizer


def rehash(job):
    return make_job(job["experiment_id"], **{key: value for key, value in job.items() if key not in {"job_id", "experiment_id"}})


class FactReadoutProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20260912, "discovery")

    def source(self):
        return build_input_site_spec(self.pairs, FactTokenizer(), "source_input", layers=(0, 10), seeds=(42, 43), cap=256)

    def test_complete_cohort_tokens_positions_and_interventions_preserved(self):
        source = self.source()
        original = deepcopy(source)
        derived = build_fact_readout_spec(source, "fact_reports")
        self.assertEqual(source, original)
        self.assertEqual(len(derived["jobs"]), len(source["jobs"]))
        self.assertEqual(derived["n_source_jobs"], 60)
        self.assertTrue(derived["complete_source_cohort_preserved"])
        for before, after in zip(source["jobs"], derived["jobs"]):
            for key in ("pair_id", "layout_id", "family", "condition", "seed", "recipient_text", "donor_text", "recipient_token_ids", "donor_token_ids",
                        "recipient_positions", "donor_positions", "selection", "intervention", "target_action", "original_action", "temperature", "max_new_tokens"):
                self.assertEqual(before[key], after[key], key)
            self.assertNotEqual(before["job_id"], after["job_id"])
            self.assertEqual(after, rehash(after))
            self.assertFalse(after["natural"])
            self.assertFalse(after["semantic_test"])
            self.assertEqual(after["fact_probes"], build_fact_probes(before["metadata"]["pair"]))
            diagnostic = after["fact_readout"]
            self.assertEqual(diagnostic["source_job_fingerprint"], fingerprint(before))
            self.assertEqual(diagnostic["source_spec_fingerprint"], fingerprint(source))
            self.assertEqual(after["metadata"]["pair"], before["metadata"]["pair"])
            self.assertFalse(diagnostic["native_continuation_requested"])
            self.assertFalse(diagnostic["mediation_evidence"])

    def test_factual_and_source_action_comparisons_are_kept_separate(self):
        source = self.source()
        derived = build_fact_readout_spec(source, "reports")
        self.assertEqual(derived["comparisons"], [])
        self.assertEqual(derived["source_action_comparisons"], source["comparisons"])
        self.assertEqual(len(derived["fact_readout_comparisons"]), 2 * len(source["comparisons"]))
        for comparison in derived["fact_readout_comparisons"]:
            self.assertEqual(comparison["expected_n_pairs"], 6)
            self.assertEqual(comparison["expected_n_layouts"], 3)
            self.assertEqual(comparison["claim_scope"], CLAIM_SCOPE)
            self.assertEqual(comparison["probability_key"], "counterfactual_truth_q")
            self.assertFalse(comparison["semantic_headline_eligible"])

    def test_raw_source_pair_precedence_preserves_original_provenance(self):
        source = self.source()
        originals = {pair["pair_id"]: pair for pair in self.pairs}
        for index, job in enumerate(source["jobs"]):
            job["metadata"]["source_pair"] = deepcopy(originals[job["pair_id"]])
            source["jobs"][index] = rehash(job)
        derived = build_fact_readout_spec(source, "raw_reports")
        for job in derived["jobs"]:
            self.assertEqual(job["fact_readout"]["physical_pair_metadata_origin"], "metadata.source_pair")
            self.assertEqual(job["fact_probes"], build_fact_probes(originals[job["pair_id"]]))
            self.assertNotEqual(job["metadata"]["source_pair"], job["metadata"]["pair"])

    def test_truths_match_each_unpatched_physical_context(self):
        derived = build_fact_readout_spec(self.source(), "truth_reports")
        for job in derived["jobs"]:
            context = "counterfactual" if job["condition"] == "counterfactual_text" else "recipient"
            self.assertEqual(job["fact_readout"]["unpatched_physical_context"], context)
            for probe in job["fact_probes"]:
                if probe["available"]:
                    self.assertEqual(job["fact_readout"]["unpatched_context_truths"][probe["id"]], probe[context + "_value"])
                if probe["role"] == "unchanged_fact_control":
                    self.assertEqual(probe["recipient_value"], probe["counterfactual_value"])

    def test_unavailable_auxiliary_is_retained_without_case_exclusion(self):
        source = self.source()
        original_builder = build_fact_probes
        def no_auxiliary(pair):
            probes = original_builder(pair)
            probes[1].update(available=False, question=None, recipient_value=None, counterfactual_value=None,
                             unavailable_reason="test_no_auxiliary")
            return probes
        with patch("src.activation_causality.fact_readout_protocol.build_fact_probes", side_effect=no_auxiliary):
            derived = build_fact_readout_spec(source, "unavailable_reports")
        self.assertEqual(len(derived["jobs"]), len(source["jobs"]))
        self.assertTrue(all(not job["fact_probes"][1]["available"] for job in derived["jobs"]))
        self.assertTrue(all(comparison["expected_n_pairs"] == 6 for comparison in derived["fact_readout_comparisons"]))

    def test_source_fingerprint_drift_or_incomplete_cohorts_fail_whole_build(self):
        source = self.source()
        source["jobs"][0]["seed"] = 99
        with self.assertRaisesRegex(ValueError, "registered definition"):
            build_fact_readout_spec(source, "drift")
        source = self.source()
        source["jobs"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_fact_readout_spec(source, "incomplete")
        source = self.source()
        source["jobs"] = [job for job in source["jobs"] if job["condition"] != "random_factspan_layer10"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_fact_readout_spec(source, "missing_control")

    def test_arbitrary_contexts_confirmation_and_same_id_rejected(self):
        source = self.source()
        source["claim_scope"] = "contextual_input_fact_interchange"
        with self.assertRaises(ValueError):
            build_fact_readout_spec(source, "context")
        source = self.source()
        source["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            build_fact_readout_spec(source, "confirm")
        source = self.source()
        with self.assertRaises(ValueError):
            build_fact_readout_spec(source, source["experiment_id"])
        source["jobs"][0]["recipient_text"] += " different context"
        source["jobs"][0] = rehash(source["jobs"][0])
        with self.assertRaisesRegex(ValueError, "physical R/CF"):
            build_fact_readout_spec(source, "wrong_context")

    def test_fact_control_truth_tampering_and_position_contradictions_rejected(self):
        original_builder = build_fact_probes
        def wrong_truth(pair):
            probes = original_builder(pair)
            probes[1]["counterfactual_value"] = not probes[1]["recipient_value"]
            return probes
        with patch("src.activation_causality.fact_readout_protocol.build_fact_probes", side_effect=wrong_truth):
            with self.assertRaisesRegex(ValueError, "Auxiliary"):
                build_fact_readout_spec(self.source(), "bad_truth")
        source = self.source()
        source["jobs"][0]["recipient_positions"] = [0]
        source["jobs"][0] = rehash(source["jobs"][0])
        with self.assertRaisesRegex(ValueError, "positions disagree"):
            build_fact_readout_spec(source, "bad_positions")

    def test_derivative_json_is_deterministic_and_contains_no_results(self):
        source = self.source()
        first = build_fact_readout_spec(source, "reports")
        self.assertEqual(first, build_fact_readout_spec(source, "reports"))
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertTrue(first["protocol"]["source_file_sha256_verification_required"])
        self.assertTrue(first["protocol"]["source_retirement_verification_required"])
        self.assertEqual(len(first["source_job_map"]), len(source["jobs"]))
        self.assertTrue(all("native_action" not in job and "fact_reports" not in job for job in first["jobs"]))


if __name__ == "__main__":
    unittest.main()
