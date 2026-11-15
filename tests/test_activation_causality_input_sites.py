"""CPU-only physical-pair and tokenizer alignment checks."""

from __future__ import annotations

from copy import deepcopy
import json
import re
import unittest

from src.activation_causality.environments import generate_pairs
from src.activation_causality.input_sites import aligned_changed_span, build_input_site_spec


class FactTokenizer:
    """Both Boolean spellings have one token, as do individual grid symbols."""
    name_or_path = "test-only-fixed-tokenizer"

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("builder must not add tokenizer framing")
        self.calls.append(text)
        return [300 if piece == "False" else 301 if piece == "True" else ord(piece)
                for piece in re.findall(r"False|True|[\s\S]", text)]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class SpanTests(unittest.TestCase):
    def test_minimal_contiguous_span_and_exact_unchanged_flanks(self):
        recipient = [1, 2, 3, 4, 5, 6]
        donor = [1, 2, 9, 4, 8, 6]
        span = aligned_changed_span(recipient, donor)
        self.assertEqual(span["recipient_positions"], [2, 3, 4])
        self.assertEqual(span["donor_positions"], [2, 3, 4])
        self.assertEqual(span["common_prefix_tokens"], 2)
        self.assertEqual(span["common_suffix_tokens"], 1)

    def test_all_changed_and_edge_changes_do_not_overlap_suffix(self):
        self.assertEqual(aligned_changed_span([1, 2], [3, 4])["recipient_positions"], [0, 1])
        self.assertEqual(aligned_changed_span([1, 2], [1, 3])["recipient_positions"], [1])
        self.assertEqual(aligned_changed_span([1, 2], [3, 2])["recipient_positions"], [0])

    def test_empty_unequal_and_identical_sequences_rejected(self):
        for left, right, reason in (([], [], "empty_token_sequence"), ([1], [1, 2], "unequal_full_token_lengths"), ([1, 2], [1, 2], "empty_changed_token_span")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                aligned_changed_span(left, right)


class SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Procedural CPU simulator only; no trajectories, files, or models read.
        cls.pairs = generate_pairs(3, 20260912, partition="discovery", families=("wall", "key_possession", "door_status"))

    def test_all_families_frozen_tokens_conditions_and_counterfactual(self):
        tokenizer = FactTokenizer()
        spec = build_input_site_spec(self.pairs, tokenizer, "input_test", layers=(0, 10), seeds=(42, 43), cap=256)
        self.assertEqual(spec["n_aligned_pairs"], 3)
        self.assertEqual(spec["n_excluded_pairs"], 0)
        self.assertEqual(len(spec["jobs"]), 3 * (2 + 4 * 2) * 2)
        self.assertEqual(len({job["job_id"] for job in spec["jobs"]}), len(spec["jobs"]))
        self.assertTrue(all(not job["semantic_test"] for job in spec["jobs"]))
        for job in spec["jobs"]:
            self.assertEqual(job["selection"]["scope"], "explicit")
            self.assertEqual(job["recipient_positions"], job["selection"]["recipient_positions"])
            self.assertEqual(job["metadata"]["claim_scope"], "input_fact_information_routing")
            self.assertEqual(job["max_new_tokens"], 256)
            if job["donor_token_ids"] is not None:
                self.assertEqual(len(job["recipient_token_ids"]), len(job["donor_token_ids"]))
        for pair in self.pairs:
            group = {job["condition"]: job for job in spec["jobs"] if job["pair_id"] == pair["pair_id"] and job["seed"] == 42}
            baseline, physical, fact = group["baseline"], group["counterfactual_text"], group["factspan_layer10"]
            self.assertEqual(fact["recipient_token_ids"], baseline["recipient_token_ids"])
            self.assertEqual(fact["donor_token_ids"], physical["recipient_token_ids"])
            self.assertEqual(group["identity_layer10"]["recipient_token_ids"], group["identity_layer10"]["donor_token_ids"])
            self.assertTrue(group["random_factspan_layer10"]["intervention"]["random_control"])
            self.assertEqual(group["random_factspan_layer10"]["recipient_positions"], fact["recipient_positions"])
            self.assertLess(max(fact["recipient_positions"]), len(fact["recipient_token_ids"]) - 1)
            self.assertEqual(group["lasttoken_layer10"]["recipient_positions"], [len(fact["recipient_token_ids"]) - 1])
        json.dumps(spec, allow_nan=False)

    def test_unequal_token_lengths_are_explicit_prequery_exclusions(self):
        spec = build_input_site_spec(self.pairs, CharacterTokenizer(), "unequal", layers=(10,))
        self.assertEqual(spec["n_aligned_pairs"], 2)
        self.assertEqual(spec["exclusion_counts"], {"unequal_full_token_lengths": 1})
        self.assertEqual(spec["exclusions"][0]["family"], "key_possession")
        self.assertTrue(spec["exclusions"][0]["excluded_before_model_query"])
        excluded_id = spec["exclusions"][0]["pair_id"]
        self.assertTrue(all(job["pair_id"] != excluded_id for job in spec["jobs"]))

    def test_invalid_physical_changes_are_rejected_before_tokenization(self):
        pair = deepcopy(self.pairs[0])
        pair["counterfactual"]["has_key"] = not pair["recipient"]["has_key"]
        tokenizer = FactTokenizer()
        spec = build_input_site_spec([pair], tokenizer, "two_facts")
        self.assertEqual(spec["jobs"], [])
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(spec["exclusions"][0]["reason"], "counterfactual_changes_more_than_designated_fact_or_is_identical")

    def test_prompt_tampering_and_wrong_declared_values_are_excluded(self):
        for mutation, reason in (({"recipient_prompt_text": "extra instructions"}, "recipient_prompt_does_not_match_physical_state"),
                                 ({"intervention": {**self.pairs[0]["intervention"], "donor_value": "wrong"}}, "recorded_fact_values_disagree_with_states")):
            spec = build_input_site_spec([{**self.pairs[0], **mutation}], FactTokenizer(), "invalid")
            self.assertEqual(spec["jobs"], [])
            self.assertEqual(spec["exclusions"][0]["reason"], reason)

    def test_non_discovery_and_duplicates_are_explicit_exclusions(self):
        confirmation = {**self.pairs[0], "partition": "confirmation"}
        spec = build_input_site_spec([confirmation, self.pairs[1], self.pairs[1]], FactTokenizer(), "duplicates")
        self.assertEqual(spec["n_aligned_pairs"], 1)
        self.assertEqual(spec["exclusion_counts"], {"duplicate_pair_id": 1, "non_discovery_pair": 1})

    def test_mismatched_original_donor_does_not_enter_fact_transfer(self):
        pair = deepcopy(self.pairs[0])
        pair["donor"] = deepcopy(pair["recipient"])
        pair["donor_prompt_text"] = "unrelated donor prompt"
        pair["metadata"]["donor_context"] = "different_agent_position"
        spec = build_input_site_spec([pair], FactTokenizer(), "matched_only", layers=(10,))
        fact = next(job for job in spec["jobs"] if job["condition"] == "factspan_layer10")
        self.assertNotIn("unrelated donor", fact["donor_text"])
        self.assertEqual(fact["metadata"]["pair"]["donor"], pair["counterfactual"])
        self.assertEqual(fact["metadata"]["pair"]["metadata"]["donor_context"], "matched_physical_counterfactual")

    def test_token_aliases_that_hide_the_fact_change_are_excluded(self):
        class ConstantTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]
        spec = build_input_site_spec([self.pairs[0]], ConstantTokenizer(), "aliases")
        self.assertEqual(spec["exclusions"][0]["reason"], "empty_changed_token_span")
        self.assertEqual(spec["jobs"], [])

    def test_spec_is_deterministic_and_does_not_mutate_input_pairs(self):
        original = deepcopy(self.pairs)
        first = build_input_site_spec(self.pairs, FactTokenizer(), "deterministic", layers=(3, 15), natural=False)
        second = build_input_site_spec(self.pairs, FactTokenizer(), "deterministic", layers=(3, 15), natural=False)
        self.assertEqual(first, second)
        self.assertEqual(self.pairs, original)
        self.assertTrue(all(not job["natural"] for job in first["jobs"]))

    def test_invalid_configuration_is_not_silently_excluded(self):
        for kwargs in ({"layers": (24,)}, {"layers": (1, 1)}, {"site": "key_cache"}, {"seeds": (-1,)}, {"cap": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_input_site_spec(self.pairs, FactTokenizer(), "bad_config", **kwargs)


if __name__ == "__main__":
    unittest.main()
