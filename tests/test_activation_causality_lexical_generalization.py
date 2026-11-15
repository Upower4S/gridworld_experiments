"""CPU-only fixed-renderer and span guards; no model/framework imports."""
import copy
import re
import unittest
from unittest.mock import patch

from src.activation_causality.environments import generate_pairs, make_state, prompt_text
from src.activation_causality.lexical_generalization import (
    ALTERNATE, CANONICAL, audit_pair_renderings, parse_prompt, render_prompt,
    render_with_proof, renderer_manifest,
)


class WordTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Unexpected special-token insertion")
        return [self.vocabulary.setdefault(token, len(self.vocabulary))
                for token in re.findall(r"[A-Za-z]+|[0-9]+|[^\s]", text)]


class SplitAlternateTokenizer(WordTokenizer):
    def encode(self, text, add_special_tokens=False):
        return super().encode(text.replace("Yes", "Y es").replace("No", "N o"), add_special_tokens)


class FixedRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(6, 1709, "discovery", families=("wall", "key_possession", "door_status"))

    def state(self, has_key=False):
        return make_state(["#######", "#A_K_G#", "#_D___#", "#######"], has_key=has_key)

    def test_canonical_bytes_and_roundtrip(self):
        for pair in self.pairs:
            for role in ("recipient", "counterfactual", "donor"):
                state = pair[role]
                self.assertEqual(render_prompt(state, CANONICAL), prompt_text(state))
                for renderer in (CANONICAL, ALTERNATE):
                    self.assertEqual(parse_prompt(render_prompt(state, renderer), renderer), state)

    def test_alternate_exact_meaning_and_structural_proof(self):
        for possession in (False, True):
            state = self.state(possession)
            before = copy.deepcopy(state)
            rendered = render_with_proof(state)
            text = rendered["prompt_text"]
            self.assertIn("Legend: W = wall; . = open space; G = goal; A = agent; L = locked door; K = key.", text)
            self.assertTrue(text.endswith("Carrying key: " + ("Yes" if possession else "No")))
            self.assertIn('JSON object of the form {"action": "UP"}', text)
            self.assertEqual(rendered["canonical_state"], before)
            self.assertEqual(state, before)
            self.assertEqual(rendered["proof"]["state_sha256"], rendered["proof"]["reconstructed_state_sha256"])
            self.assertTrue(rendered["proof"]["exact_state_round_trip"])

    def test_shared_mechanics_manifest_and_distinct_encoding(self):
        standard, alternate = renderer_manifest(CANONICAL), renderer_manifest(ALTERNATE)
        self.assertEqual(standard["mechanics_and_action_schema_sha256"], alternate["mechanics_and_action_schema_sha256"])
        self.assertNotEqual(standard["manifest_sha256"], alternate["manifest_sha256"])
        alternate["canonical_to_display"]["#"] = "bad"
        self.assertEqual(renderer_manifest(ALTERNATE)["canonical_to_display"]["#"], "W")

    def test_modified_legend_mechanics_status_and_spacing_rejected(self):
        text = render_prompt(self.state())
        mutations = (text.replace("W = wall", "W = goal"), text.replace("fewest steps", "most steps"),
                     text.replace("Carrying key: No", "Carrying key: False"), text + "\n",
                     text.replace("0 1 2", "0  1 2", 1), text.replace("W W W", "W Q W", 1))
        for changed in mutations:
            self.assertNotEqual(changed, text)
            with self.assertRaises(ValueError):
                parse_prompt(changed, ALTERNATE)

    def test_unknown_renderer_and_noncanonical_state_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_fixed_renderer"):
            render_prompt(self.state(), "custom")
        with self.assertRaisesRegex(ValueError, "exact_canonical_state_fields"):
            render_prompt({**self.state(), "prompt_text": "Injected instruction"})
        with self.assertRaises(ValueError):
            render_prompt({**self.state(), "has_key": "False"})
        with self.assertRaises(ValueError):
            render_prompt({**self.state(), "schema_version": "unknown"})

    def test_goal_hidden_by_agent_rejected(self):
        state = self.state()
        state["agent"] = [1, 5]
        with self.assertRaisesRegex(ValueError, "agent_overlay"):
            render_prompt(state)

    def test_canonical_renderer_drift_fails_closed(self):
        with patch("src.activation_causality.lexical_generalization.prompt_text", return_value="new incompatible grammar"):
            with self.assertRaisesRegex(ValueError, "canonical_renderer_drift"):
                render_prompt(self.state())

    def test_matched_joint_span_geometry_and_target_preservation(self):
        tokenizer = WordTokenizer()
        for pair in self.pairs:
            audit = audit_pair_renderings(pair, tokenizer, profile="matched")
            self.assertTrue(audit["eligible"], audit["exclusions"])
            self.assertEqual(audit["target_action"], pair["counterfactual_optimal_actions"][0])
            self.assertEqual(audit["model_queries"], 0)
            self.assertFalse(audit["registered_experiment"])
            for role in ("recipient", "counterfactual"):
                widths = [audit["renderings"][renderer][role]["span"]["n_changed_span_tokens"]
                          for renderer in (CANONICAL, ALTERNATE)]
                self.assertEqual(widths, [1, 1])

    def test_alternate_support_widening_is_excluded(self):
        pair = next(p for p in self.pairs if p["family"] == "key_possession")
        audit = audit_pair_renderings(pair, SplitAlternateTokenizer(), profile="matched")
        self.assertFalse(audit["eligible"])
        self.assertTrue(any(row["reason"] == "standard_alternate_support_width_mismatch" for row in audit["exclusions"]))

    def test_unequal_full_lengths_are_not_padded(self):
        class BadTokenizer(WordTokenizer):
            def encode(self, text, add_special_tokens=False):
                return super().encode(text.replace("Carrying key: Yes", "Carrying key: affirmative yes"), add_special_tokens)
        pair = next(p for p in self.pairs if p["family"] == "key_possession")
        audit = audit_pair_renderings(pair, BadTokenizer(), profile="matched")
        self.assertFalse(audit["eligible"])
        self.assertTrue(any(row["reason"] == "unequal_full_token_lengths" for row in audit["exclusions"]))

    def test_contextual_requires_every_source_control_before_query(self):
        audit = audit_pair_renderings(self.pairs[0], WordTokenizer(), profile="contextual")
        self.assertTrue(audit["eligible"], audit["exclusions"])
        expected = {"recipient", "counterfactual", "donor", "natural_donor", "same_fact_donor", "irrelevant_fact_donor"}
        self.assertEqual(set(audit["renderings"][ALTERNATE]), expected)
        self.assertIn(["recipient", "natural_donor"], audit["required_source_edges"])
        with patch("src.activation_causality.lexical_generalization.state_control_specs", return_value=[
            {"condition": "natural_donor", "eligible": False, "ineligible_reason": "unavailable"}
        ]):
            unavailable = audit_pair_renderings(self.pairs[0], WordTokenizer(), profile="contextual")
        self.assertFalse(unavailable["eligible"])
        self.assertIn({"source": "natural_donor", "reason": "unavailable"}, unavailable["exclusions"])

    def test_source_state_and_action_mismatch_rejected(self):
        pair = copy.deepcopy(self.pairs[0])
        pair["counterfactual_optimal_actions"] = pair["recipient_optimal_actions"]
        with self.assertRaisesRegex(ValueError, "recorded_simulator_action_mismatch"):
            audit_pair_renderings(pair, WordTokenizer(), profile="matched")
        with self.assertRaisesRegex(ValueError, "unsupported_profile"):
            audit_pair_renderings(self.pairs[0], WordTokenizer(), profile="custom")


if __name__ == "__main__":
    unittest.main()
