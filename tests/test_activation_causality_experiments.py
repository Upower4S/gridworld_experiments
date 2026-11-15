"""Pure job-builder checks; never load a model or read campaign files."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from src.activation_causality.campaign import render_analysis_prompt
from src.activation_causality.environments import generate_pairs
from src.activation_causality.experiments import build_controlled_state_spec


class ControlledStateSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 5821, "discovery")

    def build(self, **kwargs):
        return build_controlled_state_spec(self.pairs, "controlled_test", layers=(10,), seeds=(42, 43), **kwargs)

    def test_deterministic_serializable_and_input_not_mutated(self):
        original = deepcopy(self.pairs)
        spec = self.build()
        self.assertEqual(spec, self.build())
        self.assertEqual(self.pairs, original)
        self.assertEqual(json.loads(json.dumps(spec)), spec)
        self.assertEqual(len({job["job_id"] for job in spec["jobs"]}), len(spec["jobs"]))
        self.assertIn("native", spec["primary_estimand"])

    def test_all_targets_remain_counterfactual_and_reverse_has_correct_baseline(self):
        spec = self.build()
        pairs = {pair["pair_id"]: pair for pair in self.pairs}
        for job in spec["jobs"]:
            pair = pairs[job["pair_id"]]
            self.assertEqual(job["target_action"], pair["counterfactual_optimal_actions"][0])
            self.assertEqual(job["original_action"], pair["recipient_optimal_actions"][0])
            self.assertIsNone(job["metadata"]["observed_donor_action"])
            self.assertIsNone(job["metadata"]["observed_donor_action_mismatch"])
            self.assertFalse(job["metadata"]["independent_rescue_evidence"])
            if job["condition"] == "layer10_reverse":
                self.assertEqual(job["recipient_text"], render_analysis_prompt(pair["counterfactual_prompt_text"]))
                self.assertEqual(job["donor_text"], render_analysis_prompt(pair["recipient_prompt_text"]))
        reverse = next(c for c in spec["comparisons"] if c["role"] == "conditional_reverse_effect")
        self.assertEqual(reverse["control"], "counterfactual_text")
        self.assertEqual(reverse["predicted_direction"], "decrease")

    def test_actual_context_donors_and_specificity_pairs(self):
        spec = self.build()
        for item in spec["eligibility_manifest"]:
            controls = {control["condition"]: control for control in item["controls"]}
            pair_jobs = [job for job in spec["jobs"] if job["pair_id"] == item["pair_id"] and job["seed"] == 42]
            indexed = {job["condition"]: job for job in pair_jobs}
            if controls["natural_donor"]["eligible"]:
                natural = indexed["layer10_natural"]
                same = indexed["layer10_same_fact"]
                self.assertEqual(natural["donor_text"], render_analysis_prompt(controls["natural_donor"]["prompt_text"]))
                self.assertEqual(same["donor_text"], render_analysis_prompt(controls["same_fact_donor"]["prompt_text"]))
                self.assertNotEqual(natural["donor_text"], indexed["layer10_matched"]["donor_text"])
                self.assertEqual(natural["metadata"]["donor_observation_condition"], "natural_donor_text")

    def test_random_controls_declare_actual_delta_norm_and_seed(self):
        spec = self.build()
        for job in spec["jobs"]:
            if job["condition"] != "layer10_random":
                self.assertFalse(job["intervention"].get("random_control", False))
                continue
            self.assertTrue(job["intervention"]["random_control"])
            self.assertEqual(job["intervention"]["norm_reference"], "donor_minus_recipient")
            self.assertEqual(job["intervention"]["random_seed"], job["seed"])

    def test_identity_equivalent_is_not_independent_rescue(self):
        for job in self.build()["jobs"]:
            if job["condition"] in {"layer10_identity", "layer10_restoration_equivalent"}:
                self.assertEqual(job["recipient_text"], job["donor_text"])
                self.assertTrue(job["metadata"]["engineering_identity_only"])
                self.assertFalse(job["semantic_test"])
            if job["condition"] == "layer10_restoration_equivalent":
                self.assertIn("not removal", job["metadata"]["restoration_note"])

    def test_unavailable_controls_recorded_and_comparisons_use_frozen_subset(self):
        from src.activation_causality.designs import state_control_specs
        original = state_control_specs

        def subset_controls(pair):
            rows = original(pair)
            if pair["pair_id"] == self.pairs[0]["pair_id"]:
                for row in rows:
                    if row["condition"] == "irrelevant_fact_donor":
                        row.update({"eligible": False, "state": None, "prompt_text": None,
                                    "ineligible_reason": "test_prespecified_unavailable"})
            return rows

        with patch("src.activation_causality.experiments.state_control_specs", side_effect=subset_controls):
            spec = self.build()
        unavailable = spec["eligibility_manifest"][0]["unscheduled_conditions"]
        self.assertIn("layer10_irrelevant", [row["condition"] for row in unavailable])
        comparison = next(c for c in spec["comparisons"] if c["role"] == "irrelevant_fact_specificity")
        self.assertNotIn(self.pairs[0]["pair_id"], comparison["eligible_pair_ids"])
        self.assertEqual(comparison["expected_n_pairs"], len(comparison["eligible_pair_ids"]) * 2)
        self.assertFalse(comparison["outcome_dependent_exclusion_permitted"])
        semantic = next(c for c in spec["comparisons"] if c["role"] == "semantic_fact_specificity")
        self.assertNotIn(self.pairs[0]["pair_id"], semantic["eligible_pair_ids"])
        self.assertTrue(semantic["eligibility_requires_all_semantic_controls"])
        self.assertEqual(semantic["semantic_controls"]["donor_baseline"], "natural_donor_text")
        self.assertEqual(semantic["semantic_controls"]["donor_mismatch_treatment"], "layer10_natural")

    def test_aliases_and_unique_paired_jobs(self):
        spec = self.build()
        seen = set()
        for job in spec["jobs"]:
            key = job["pair_id"], job["seed"], job["condition"]
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertTrue(set(job["layout_aliases"]) <= set(spec["layout_aliases"][job["layout_id"]]))
        for comparison in spec["comparisons"]:
            for pair_id in comparison["eligible_pair_ids"]:
                for seed in (42, 43):
                    self.assertIn((pair_id, seed, comparison["treatment"]), seen)
                    self.assertIn((pair_id, seed, comparison["control"]), seen)

    def test_arguments_and_partition_not_silently_relabelled(self):
        with self.assertRaises(ValueError):
            build_controlled_state_spec([], "empty")
        for kwargs in ({"layers": (24,)}, {"layers": (10, 10)}, {"last_k": 0},
                       {"seeds": (42, 42)}, {"site": "not_a_site"}, {"max_new_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_controlled_state_spec(self.pairs, "invalid", **kwargs)
        mixed = deepcopy(self.pairs)
        mixed[0]["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            build_controlled_state_spec(mixed, "mixed")
        forced = self.build(natural=False)
        self.assertTrue(all(not job["natural"] for job in forced["jobs"]))


if __name__ == "__main__":
    unittest.main()
