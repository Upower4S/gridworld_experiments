"""CPU-only factual truth, exact token boundary and native-cache isolation checks."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from src.activation_causality.engine import MLXInterventionEngine
from src.activation_causality.environments import generate_pairs, make_state
from src.activation_causality.fact_readout import build_fact_probes, score_fact_reports, _tree_fingerprint
from tests.test_activation_causality_engine import FakeCache, FakeModel, NumpyBackend


class BoundaryTokenizer:
    pieces = {300: "<|end|>", 301: "<|start|>", 302: "<|message|>", 303: "<|channel|>",
              304: "<|return|>", 400: " true", 401: " false"}

    def encode(self, text, add_special_tokens=False):
        ids, offset = [], 0
        while offset < len(text):
            match = next(((token, value) for token, value in self.pieces.items() if text.startswith(value, offset)), None)
            if match:
                token, value = match
                ids.append(token)
                offset += len(value)
            else:
                ids.append(ord(text[offset]) + 1)
                offset += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        # Token 450 deliberately has a noncanonical generated spelling: its
        # decoded text re-encodes as three ordinary tokens, never as 450.
        return "".join("abc" if int(i) == 450 else self.pieces.get(int(i), chr(int(i) - 1)) for i in ids)


def make_engine(tokenizer=None):
    model = FakeModel(vocab=512)
    return MLXInterventionEngine.from_components(
        model=model, tokenizer=tokenizer or BoundaryTokenizer(), backend=NumpyBackend,
        make_cache=lambda m: [FakeCache(m.args.hidden_size) for _ in m.model.layers],
        create_attention_mask=lambda x, cache, window_size=None: None,
        candidate_ids=[1, 2, 3, 4], suffix_ids=[5, 6], terminal_ids={304: "return"},
    )


class FactProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20260912, partition="discovery",
                                   families=("wall", "key_possession", "door_status"))

    def test_each_target_truth_and_unchanged_remote_control(self):
        for pair in self.pairs:
            original = deepcopy(pair)
            probes = build_fact_probes(pair)
            target, control = probes
            self.assertTrue(target["available"])
            self.assertNotEqual(target["recipient_value"], target["counterfactual_value"])
            self.assertEqual(target["recipient_value"], pair["family"] != "key_possession")
            self.assertTrue(control["available"])
            self.assertEqual(control["recipient_value"], control["counterfactual_value"])
            r, c = control["predicate"]["cell"]
            self.assertEqual(control["recipient_value"], pair["recipient"]["grid"][r][c] == "#")
            self.assertEqual(pair["recipient"]["grid"][r][c], pair["counterfactual"]["grid"][r][c])
            for probe in probes:
                self.assertIn("before any move", probe["question"])
                self.assertNotRegex(probe["question"].upper(), r"\b(?:UP|DOWN|LEFT|RIGHT)\b")
                self.assertFalse(probe["semantic_headline_eligible"])
                self.assertFalse(probe["latent_decoding"])
                self.assertFalse(probe["mediation_evidence"])
            self.assertEqual(pair, original)

    def test_small_world_surfaces_unavailable_auxiliary(self):
        recipient = make_state(["_G"], [0, 0], has_key=False)
        counterfactual = {**recipient, "has_key": True}
        pair = {"family": "key_possession", "recipient": recipient, "counterfactual": counterfactual,
                "intervention": {"variable": "key_possession", "cell": None}}
        probes = build_fact_probes(pair)
        self.assertFalse(probes[1]["available"])
        self.assertEqual(probes[1]["unavailable_reason"], "no_unchanged_remote_interior_wall_or_floor_cell")

    def test_physical_and_recorded_truth_tampering_rejected(self):
        pair = deepcopy(self.pairs[0])
        pair["counterfactual"]["has_key"] = not pair["recipient"]["has_key"]
        with self.assertRaisesRegex(ValueError, "only the designated"):
            build_fact_probes(pair)
        pair = deepcopy(self.pairs[0])
        pair["intervention"]["donor_value"] = "#"
        with self.assertRaisesRegex(ValueError, "declared fact"):
            build_fact_probes(pair)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.state = self.engine.prefill([98, 99, 100])
        self.probes = [
            {"id": "target", "question": "In the initial state, before any move, is the agent carrying a key?",
             "recipient_value": False, "counterfactual_value": True},
            {"id": "unchanged", "question": "In the initial state, before any move, is row 1, column 1 a wall?",
             "recipient_value": True, "counterfactual_value": True},
        ]

    def test_independent_probes_preserve_native_action_branch_and_rng(self):
        native_before = self.engine.generate(self.state, seed=42, max_new_tokens=5, temp=0)
        forced_before = self.engine.forced(self.state)
        original_ids = self.state.token_ids
        cache_before = [c.values.copy() for c in self.state.cache]
        logits_before = self.state.next_logits.copy()
        initial_rng = np.random.get_state()
        advance = self.engine._advance
        starts = []
        def traced(state, ids):
            starts.append(tuple(state.token_ids))
            advance(state, ids)
        self.engine._advance = traced
        reports = score_fact_reports(self.engine, self.state, self.probes)
        self.engine._advance = advance
        self.assertEqual(starts, [original_ids, original_ids])
        final_rng = np.random.get_state()
        np.testing.assert_array_equal(initial_rng[1], final_rng[1])
        self.assertEqual(initial_rng[2:], final_rng[2:])
        self.assertEqual(self.state.token_ids, original_ids)
        np.testing.assert_array_equal(self.state.next_logits, logits_before)
        for actual, expected in zip(self.state.cache, cache_before):
            np.testing.assert_array_equal(actual.values, expected)
        self.assertEqual(self.engine.generate(self.state, seed=42, max_new_tokens=5, temp=0), native_before)
        self.assertEqual(self.engine.forced(self.state), forced_before)
        for report in reports:
            self.assertEqual(report["status"], "scored")
            self.assertTrue(report["source_state_unchanged"])
            self.assertEqual(report["base_state"], report["source_state_after"])
            self.assertFalse(report["prefix_reencoded"])
            self.assertTrue(report["boundary_verified"])
        json.dumps(reports, allow_nan=False)

    def test_noncanonical_authoritative_prefix_is_not_reencoded(self):
        state = self.engine.prefill([450])
        self.assertNotEqual(self.engine.encode(self.engine.decode(state.token_ids)), list(state.token_ids))
        report = score_fact_reports(self.engine, state, self.probes[:1])[0]
        self.assertEqual(report["status"], "scored")
        self.assertEqual(state.token_ids, (450,))
        self.assertEqual(report["base_state"]["n_tokens"], 1)

    def test_selected_global_logits_mass_and_conditional_scores(self):
        advance = self.engine._advance
        def prescribed(state, ids):
            advance(state, ids)
            state.next_logits = np.zeros(512, dtype=np.float32)
            state.next_logits[[400, 401, 500]] = [5, 4, 6]
        self.engine._advance = prescribed
        report = score_fact_reports(self.engine, self.state, self.probes[:1])[0]
        self.assertAlmostEqual(report["q"]["true"], 1 / (1 + np.exp(-1)))
        self.assertAlmostEqual(report["q"]["true"] + report["q"]["false"], 1)
        expected_mass = (np.exp(5) + np.exp(4)) / (509 + np.exp(5) + np.exp(4) + np.exp(6))
        self.assertAlmostEqual(report["candidate_mass"], expected_mass)
        self.assertAlmostEqual(sum(report["candidate_global_probabilities"].values()), expected_mass)
        self.assertEqual(report["logits"], {"true": 5.0, "false": 4.0})
        self.assertEqual(report["global_top_token_id"], 500)
        self.assertEqual(report["counterfactual_truth_q"], report["q"]["true"])
        self.assertEqual(report["recipient_truth_q"], report["q"]["false"])

    def test_suffix_boundary_failure_is_unavailable_without_query(self):
        class BadBoundary(BoundaryTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                return ids + [400] if text.startswith("<|end|>") and text.endswith(" true") else ids
        engine = make_engine(BadBoundary())
        state = engine.prefill([98])
        engine._advance = lambda *_: self.fail("unsupported BPE boundary must not query the model")
        report = score_fact_reports(engine, state, self.probes[:1])[0]
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["unavailable_reason"], "suffix_Boolean_boundary_not_exact")
        self.assertFalse(report["query_performed"])
        self.assertTrue(report["source_state_unchanged"])

    def test_multitoken_separator_and_invalid_candidate_are_unavailable(self):
        for broken in ("separator", "candidate"):
            class BadTokens(BoundaryTokenizer):
                def encode(self, text, add_special_tokens=False):
                    if broken == "separator" and text == "<|end|>":
                        return [300, 300]
                    if broken == "candidate" and text == " true":
                        return [0]
                    return super().encode(text, add_special_tokens)
            engine = make_engine(BadTokens())
            state = engine.prefill([98])
            engine._advance = lambda *_: self.fail("invalid tokens must not query")
            report = score_fact_reports(engine, state, self.probes[:1])[0]
            self.assertEqual(report["status"], "unavailable")

    def test_unavailable_auxiliary_is_returned_without_query(self):
        self.engine._advance = lambda *_: self.fail("unavailable probe must not query")
        probe = {"id": "absent", "question": None, "recipient_value": None, "counterfactual_value": None,
                 "available": False, "unavailable_reason": "no_remote_cell"}
        report = score_fact_reports(self.engine, self.state, [probe])[0]
        self.assertEqual(report["unavailable_reason"], "no_remote_cell")
        self.assertTrue(report["source_state_unchanged"])

    def test_unloaded_engine_and_invalid_probes_fail_before_any_query(self):
        engine = SimpleNamespace(metadata={"loaded": False})
        with self.assertRaisesRegex(ValueError, "already loaded"):
            score_fact_reports(engine, self.state, self.probes)
        self.engine._advance = lambda *_: self.fail("invalid probes must fail before all queries")
        for mutation in ({"recipient_value": "false"}, {"question": "bad <|end|> control"}):
            with self.assertRaises(ValueError):
                score_fact_reports(self.engine, self.state, [self.probes[0], {**self.probes[1], **mutation}])

    def test_shared_cache_clone_is_refused_before_advance(self):
        self.engine.clone_state = lambda state: state
        self.engine._advance = lambda *_: self.fail("shared source state must never be advanced")
        with self.assertRaisesRegex(RuntimeError, "shared mutable"):
            score_fact_reports(self.engine, self.state, self.probes)

    def test_bf16_cache_proof_converts_before_buffer_access(self):
        class BF16:
            dtype = "bfloat16"
            shape = (2,)
            def astype(self, dtype):
                self.converted = dtype == np.float32
                return np.array([1, 2], dtype=dtype)
            def __array__(self, *args, **kwargs):
                raise RuntimeError("unsupported BF16 buffer")
        value = BF16()
        result = _tree_fingerprint(self.engine, value)
        self.assertTrue(value.converted)
        self.assertEqual(result["dtype"], "bfloat16")
        self.assertEqual(result["shape"], [2])
        self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
