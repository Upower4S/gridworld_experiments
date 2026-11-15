"""CPU checks for named-fact spans in matched and differing donor contexts."""

from copy import deepcopy
import json
import re
import unittest
from unittest.mock import patch

from src.activation_causality.campaign import render_analysis_prompt
from src.activation_causality.contextual_input import build_contextual_input_spec
from src.activation_causality.designs import state_control_specs
from src.activation_causality.environments import apply_variable, generate_pairs, prompt_text
from src.activation_causality.input_sites import aligned_changed_span


class FactTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return [300 if piece == "False" else 301 if piece == "True" else ord(piece)
                for piece in re.findall(r"False|True|[\s\S]", text)]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class ContextTokenizer(FactTokenizer):
    """A context may have different but internally aligned token lengths/spans."""
    def __init__(self, contexts, mode):
        super().__init__()
        self.contexts, self.mode = set(contexts), mode

    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        if text in self.contexts:
            if self.mode == "length":
                ids.insert(0, 900)
            else:
                index = next(i for i, value in enumerate(ids) if value in (300, 301))
                ids[index - 1] = 1000 + ids[index]
        return ids


class ContextualInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20261, "discovery")

    def build(self, pairs=None, tokenizer=None, **kwargs):
        return build_contextual_input_spec(pairs or self.pairs, tokenizer or FactTokenizer(), "context_input",
                                           layers=kwargs.pop("layers", (0, 10)), seeds=kwargs.pop("seeds", (42, 43)), **kwargs)

    def test_complete_matrix_exact_tokens_and_input_only_claims(self):
        spec = self.build()
        self.assertEqual(len(spec["jobs"]), 3 * (6 + 8 * 2) * 2)
        self.assertEqual(spec["n_aligned_recipient_pairs"], 3)
        self.assertFalse(spec["alignment_exclusions"])
        for job in spec["jobs"]:
            self.assertEqual(job["selection"]["scope"], "explicit")
            self.assertEqual(job["recipient_positions"], job["selection"]["recipient_positions"])
            self.assertEqual(job["metadata"]["recipient_patch_positions"], job["recipient_positions"])
            self.assertFalse(job["semantic_test"])
            self.assertFalse(job["semantic_headline_eligible"])
            self.assertEqual(job["claim_scope"], "contextual_input_fact_interchange")
            self.assertIsNone(job["metadata"]["observed_donor_action"])
            self.assertIsNone(job["metadata"]["observed_donor_action_mismatch"])
            if job["donor_token_ids"] is not None:
                self.assertEqual(len(job["recipient_token_ids"]), len(job["donor_token_ids"]))
                self.assertEqual(len(job["recipient_positions"]), len(job["donor_positions"]))
            self.assertLess(max(job["recipient_positions"]), len(job["recipient_token_ids"]) - 1)

    def test_natural_fact_spans_use_same_context_toggle_not_agent_difference(self):
        tokenizer = FactTokenizer()
        spec = self.build(tokenizer=tokenizer, layers=(10,), seeds=(42,))
        for pair in self.pairs:
            controls = {row["condition"]: row for row in state_control_specs(pair)}
            natural, same = controls["natural_donor"], controls["same_fact_donor"]
            self.assertEqual(natural["state"]["agent"], same["state"]["agent"])
            self.assertNotEqual(natural["state"]["agent"], pair["recipient"]["agent"])
            n_ids = tokenizer.encode(render_analysis_prompt(natural["prompt_text"]))
            s_ids = tokenizer.encode(render_analysis_prompt(same["prompt_text"]))
            own_span = aligned_changed_span(n_ids, s_ids)["recipient_positions"]
            group = {job["condition"]: job for job in spec["jobs"] if job["pair_id"] == pair["pair_id"]}
            treatment = group["layer10_natural"]
            self.assertEqual(treatment["donor_positions"], own_span)
            self.assertEqual(treatment["donor_token_ids"], n_ids)
            self.assertEqual(group["layer10_same_fact"]["donor_positions"], own_span)
            self.assertEqual(treatment["metadata"]["donor_observation_condition"], "natural_donor_text")
            self.assertTrue(all(n_ids[index] != ord("A") for index in own_span))
            combined_span = aligned_changed_span(treatment["recipient_token_ids"], n_ids)["recipient_positions"]
            self.assertGreater(len(combined_span), len(own_span))

    def test_irrelevant_context_donor_locates_target_fact_and_keeps_its_value(self):
        tokenizer = FactTokenizer()
        spec = self.build(tokenizer=tokenizer, layers=(10,), seeds=(42,))
        for pair in self.pairs:
            irrelevant = next(row for row in state_control_specs(pair) if row["condition"] == "irrelevant_fact_donor")
            changed = apply_variable(irrelevant["state"], pair["counterfactual"], pair["family"], pair["intervention"]["cell"])
            original_ids = tokenizer.encode(render_analysis_prompt(irrelevant["prompt_text"]))
            toggled_ids = tokenizer.encode(render_analysis_prompt(prompt_text(changed)))
            expected = aligned_changed_span(original_ids, toggled_ids)["recipient_positions"]
            job = next(job for job in spec["jobs"] if job["pair_id"] == pair["pair_id"] and job["condition"] == "layer10_irrelevant")
            self.assertEqual(job["donor_positions"], expected)
            self.assertEqual(job["donor_token_ids"], original_ids)
            self.assertEqual([job["recipient_token_ids"][i] for i in job["recipient_positions"]],
                             [original_ids[i] for i in expected])

    def test_context_length_or_width_mismatch_excludes_only_affected_conditions(self):
        pair = next(pair for pair in self.pairs if pair["family"] == "key_possession")
        controls = state_control_specs(pair)
        contexts = [render_analysis_prompt(row["prompt_text"]) for row in controls if row["condition"] in {"natural_donor", "same_fact_donor"}]
        for mode, reason in (("length", "cross_context_unequal_full_token_lengths"), ("width", "cross_context_unequal_named_fact_span_widths")):
            spec = self.build([pair], ContextTokenizer(contexts, mode), layers=(10,))
            conditions = {job["condition"] for job in spec["jobs"]}
            self.assertIn("layer10_matched", conditions)
            self.assertIn("natural_donor_text", conditions)
            self.assertNotIn("layer10_natural", conditions)
            self.assertNotIn("layer10_same_fact", conditions)
            self.assertIn(reason, spec["alignment_exclusion_counts"])
            excluded = next(row for row in spec["ineligible_comparisons"] if row["role"] == "semantic_fact_specificity")
            self.assertEqual(excluded["eligible_pair_ids"], [])
            self.assertEqual(excluded["expected_n_pairs"], 0)
            self.assertEqual(len(excluded["semantic_controls"]), 5)
            self.assertTrue(excluded["eligibility_requires_all_semantic_controls"])

    def test_within_source_boolean_length_failure_is_explicit_all_seed_exclusion(self):
        spec = self.build(tokenizer=CharacterTokenizer(), layers=(10,))
        self.assertEqual(spec["n_aligned_recipient_pairs"], 2)
        key_pair = next(pair for pair in self.pairs if pair["family"] == "key_possession")
        self.assertFalse(any(job["pair_id"] == key_pair["pair_id"] for job in spec["jobs"]))
        self.assertIn("unequal_full_token_lengths", str(spec["alignment_exclusions"]))
        manifest = next(row for row in spec["eligibility_manifest"] if row["pair_id"] == key_pair["pair_id"])
        self.assertFalse(manifest["scheduled_conditions"])
        self.assertTrue(all(row["seeds"] == [42, 43] for row in spec["alignment_exclusions"]))
        for comparison in spec["comparisons"]:
            self.assertNotIn(key_pair["pair_id"], comparison["eligible_pair_ids"])
            self.assertEqual(comparison["expected_n_pairs"], len(comparison["eligible_pair_ids"]) * 2)

    def test_identity_reverse_random_and_target_actions_are_preserved(self):
        spec = self.build(layers=(10,))
        for job in spec["jobs"]:
            pair = job["metadata"]["pair"]
            self.assertEqual(job["target_action"], pair["counterfactual_optimal_actions"][0])
            if job["condition"].endswith(("_identity", "_restoration_equivalent")):
                self.assertEqual(job["recipient_token_ids"], job["donor_token_ids"])
                self.assertEqual(job["recipient_positions"], job["donor_positions"])
            if job["condition"].endswith("_random"):
                self.assertTrue(job["intervention"]["random_control"])
                self.assertEqual(job["intervention"]["random_seed"], job["seed"])
            if job["condition"].endswith("_reverse"):
                self.assertEqual(job["metadata"]["paired_baseline_condition"], "counterfactual_text")
        reverse = next(row for row in spec["comparisons"] if row["role"] == "conditional_reverse_effect")
        self.assertEqual(reverse["predicted_direction"], "decrease")
        self.assertFalse(reverse["semantic_headline_eligible"])

    def test_physical_prompt_tampering_and_actual_donor_tampering_are_not_silent(self):
        pair = deepcopy(self.pairs[0])
        pair["recipient_prompt_text"] = "wrong prompt"
        tokenizer = FactTokenizer()
        spec = self.build([pair], tokenizer)
        self.assertFalse(spec["jobs"])
        self.assertFalse(tokenizer.calls)
        self.assertIn("recipient_prompt_does_not_match_physical_state", str(spec["physical_pair_exclusions"]))
        pair = deepcopy(self.pairs[0])
        pair["donor_prompt_text"] = "wrong donor"
        spec = self.build([pair], layers=(10,))
        self.assertTrue(any(job["condition"] == "layer10_natural" for job in spec["jobs"]))
        self.assertFalse(any(job["condition"] == "donor_text" for job in spec["jobs"]))
        self.assertIn("source_prompt_does_not_match_physical_state", str(spec["alignment_exclusions"]))

    def test_json_determinism_alias_union_and_no_input_mutation(self):
        before = deepcopy(self.pairs)
        first, second = self.build(), self.build()
        self.assertEqual(first, second)
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(self.pairs, before)
        self.assertEqual(len({job["job_id"] for job in first["jobs"]}), len(first["jobs"]))
        for job in first["jobs"]:
            self.assertEqual(job["layout_aliases"], first["layout_aliases"][job["layout_id"]])

    def test_confirmation_duplicate_and_invalid_config_rejected(self):
        pair = {**self.pairs[0], "partition": "confirmation"}
        with self.assertRaises(ValueError):
            self.build([pair])
        with self.assertRaises(ValueError):
            self.build([self.pairs[0], self.pairs[0]])
        for kwargs in ({"layers": (24,)}, {"seeds": (-1,)}, {"cap": 0}, {"site": "wrong"}, {"natural": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_unavailable_simulator_control_remains_required_after_alignment(self):
        controls = state_control_specs(self.pairs[0])
        irrelevant = next(row for row in controls if row["condition"] == "irrelevant_fact_donor")
        irrelevant.update(eligible=False, state=None, prompt_text=None, ineligible_reason="test_no_irrelevant_source")
        with patch("src.activation_causality.experiments.state_control_specs", return_value=controls):
            spec = self.build([self.pairs[0]], layers=(10,))
        self.assertTrue(any(job["condition"] == "layer10_natural" for job in spec["jobs"]))
        self.assertFalse(any(job["condition"] == "layer10_irrelevant" for job in spec["jobs"]))
        self.assertIn("test_no_irrelevant_source", str(spec["eligibility_manifest"][0]["unscheduled_conditions"]))
        missing = next(row for row in spec["ineligible_comparisons"] if row["role"] == "semantic_fact_specificity")
        self.assertEqual(missing["expected_n_pairs"], 0)
        self.assertEqual(missing["eligible_pair_ids"], [])
        self.assertEqual(missing["semantic_controls"]["irrelevant_fact"], "layer10_irrelevant")


if __name__ == "__main__":
    unittest.main()
