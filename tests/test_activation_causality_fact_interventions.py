"""Pure token alignment, truth, and paired-control checks for Boolean fact edits."""

from copy import deepcopy
import json
import re
import unittest

from src.activation_causality.environments import generate_pairs
from src.activation_causality.fact_interventions import ACTION_WORD, build_fact_spec


class WordTokenizer:
    def __init__(self, *, split_false=False):
        self.vocabulary = {}
        self.split_false = split_false

    def encode(self, text, add_special_tokens=False):
        words = re.findall(r"<\|[^|]+\|>|[A-Za-z_]+|\d+|[^\w\s]", text)
        if self.split_false:
            words = [piece for word in words for piece in (["Fal", "se"] if word == "False" else [word])]
        return [self.vocabulary.setdefault(word, len(self.vocabulary)) for word in words]


class FactInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20261, "discovery")

    def build(self, tokenizer=None):
        return build_fact_spec(self.pairs, tokenizer or WordTokenizer(), "facts", layers=(10,), seeds=(42, 43))

    def test_fact_inputs_have_no_action_words_and_share_true_prompt(self):
        spec = self.build()
        self.assertTrue(spec["jobs"])
        for job in spec["jobs"]:
            meta = job["metadata"]
            self.assertFalse(ACTION_WORD.search(meta["recipient_sentence"]))
            if meta["donor_sentence"]:
                self.assertFalse(ACTION_WORD.search(meta["donor_sentence"]))
            self.assertTrue(job["recipient_text"].startswith(job["shared_base_text"]))
            if job["donor_text"]:
                self.assertTrue(job["donor_text"].startswith(job["shared_base_text"]))
                self.assertEqual(len(job["recipient_token_ids"]), len(job["donor_token_ids"]))
            self.assertFalse(job["semantic_test"])
            self.assertFalse(meta["naturally_generated_reasoning"])
            self.assertGreaterEqual(min(meta["patch_positions"]), meta["base_token_count"])

    def test_reverse_uses_counterfactual_baseline_and_constant_target(self):
        spec = self.build()
        pairs = {pair["pair_id"]: pair for pair in self.pairs}
        for job in spec["jobs"]:
            self.assertEqual(job["target_action"], pairs[job["pair_id"]]["counterfactual_optimal_actions"][0])
            if job["condition"] == "reverse_layer10":
                self.assertEqual(job["metadata"]["paired_baseline_condition"], "donor_text")
                self.assertFalse(job["metadata"]["recipient_fact_true_in_prompt"])
            if job["condition"] in {"donor_text", "irrelevant_donor_text"}:
                self.assertFalse(job["metadata"]["recipient_fact_true_in_prompt"])
        reverse = next(row for row in spec["comparisons"] if row["treatment"] == "reverse_layer10")
        self.assertEqual(reverse["control"], "donor_text")

    def test_irrelevant_edits_have_their_own_aligned_baselines(self):
        spec = self.build()
        irrelevant_comparisons = [row for row in spec["comparisons"] if row["role"] == "irrelevant_state_fact_sensitivity"]
        self.assertTrue(irrelevant_comparisons)
        self.assertEqual(irrelevant_comparisons[0]["control"], "irrelevant_baseline")
        for record in spec["eligibility_manifest"]:
            if record.get("irrelevant_eligible"):
                self.assertEqual(record["sentence_token_count"], record["irrelevant_token_count"])

    def test_unequal_boolean_lengths_never_silently_align(self):
        spec = self.build(WordTokenizer(split_false=True))
        self.assertFalse(spec["jobs"])
        self.assertTrue(all(not row["eligible"] for row in spec["eligibility_manifest"]))

    def test_json_determinism_and_original_inputs_unchanged(self):
        original = deepcopy(self.pairs)
        first, second = self.build(), self.build()
        self.assertEqual(first, second)
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(self.pairs, original)
        self.assertEqual(len(first["jobs"]), len({job["job_id"] for job in first["jobs"]}))

    def test_confirmation_rejected_and_counts_exact(self):
        confirmation = deepcopy(self.pairs)
        for pair in confirmation:
            pair["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            build_fact_spec(confirmation, WordTokenizer(), "confirm")
        spec = self.build()
        for comparison in spec["comparisons"]:
            self.assertEqual(comparison["expected_n_pairs"], 2 * len(comparison["eligible_pair_ids"]))


if __name__ == "__main__":
    unittest.main()
