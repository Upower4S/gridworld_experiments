"""CPU-only correctness checks; run with python -m unittest discover -s tests -p '*causality_engine.py'."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import numpy as np

from src.activation_causality.engine import (
    ENGINE_DEPENDENCY_PATHS, MLXInterventionEngine, RecipientStateMismatch,
    action_metrics, engine_dependency_hashes, parse_native_action,
)
from src.activation_causality.subspace_audit import array_fingerprint


class EngineDependencyTests(unittest.TestCase):
    def test_required_closure_rejects_each_missing_or_invalid_hash(self):
        sources = {name: "a" * 64 for name in ENGINE_DEPENDENCY_PATHS}
        self.assertEqual(engine_dependency_hashes({"sources": sources}), sources)
        for name in ENGINE_DEPENDENCY_PATHS:
            for bad in (None, "short", "G" * 64):
                with self.subTest(name=name, bad=bad):
                    changed = dict(sources)
                    if bad is None:
                        changed.pop(name)
                    else:
                        changed[name] = bad
                    with self.assertRaisesRegex(ValueError, "engine dependency"):
                        engine_dependency_hashes({"sources": changed})

    def test_runtime_hashes_actual_dependency_bytes_and_requires_presence(self):
        from src.activation_causality.campaign import runtime_fingerprint
        from src.activation_causality.registry import file_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (*ENGINE_DEPENDENCY_PATHS, "src/activation_causality/engine.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fake source, not imported\n")
            baseline = runtime_fingerprint(root)
            for name in ENGINE_DEPENDENCY_PATHS:
                with self.subTest(name=name):
                    path = root / name
                    path.write_text("# changed dependency, not imported\n")
                    changed = runtime_fingerprint(root)
                    self.assertEqual(changed["sources"][name], file_sha256(path))
                    self.assertNotEqual(changed["sources"][name], baseline["sources"][name])
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, "missing required engine dependency"):
                        runtime_fingerprint(root)
                    path.write_text("# fake source, not imported\n")


class NumpyBackend:
    int32 = np.int32
    float32 = np.float32
    random = np.random

    @staticmethod
    def array(value, dtype=None):
        return np.array(value, dtype=dtype, copy=True)

    @staticmethod
    def eval(*values):
        return None

    @staticmethod
    def logsumexp(value, axis=-1, keepdims=False):
        maximum = np.max(value, axis=axis, keepdims=True)
        result = maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))
        return result if keepdims else np.squeeze(result, axis=axis)


class FakeCache:
    def __init__(self, width):
        self.values = np.empty((0, width), dtype=np.float32)

    @property
    def state(self):
        return [self.values]

    @property
    def meta_state(self):
        return {"offset": len(self.values)}

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls(state[0].shape[-1])
        obj.values = state[0]
        if len(obj.values) != meta_state["offset"]:
            raise ValueError("bad offset")
        return obj


class FakeLayer:
    def __init__(self, width, seed):
        rng = np.random.default_rng(seed)
        self.attention_weight = rng.normal(0, 0.12, (width, width)).astype(np.float32)
        self.mlp_weight = rng.normal(0, 0.15, (width, width)).astype(np.float32)

    @staticmethod
    def input_layernorm(x):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 0.1)

    post_attention_layernorm = input_layernorm

    def self_attn(self, x, mask, cache):
        # A causal attention analogue with real mutable prefix memory. Its
        # chunked and full-sequence outputs must coincide, including under edits.
        old_length = len(cache.values)
        cache.values = np.concatenate([cache.values, x[0]], axis=0)
        running = np.cumsum(cache.values, axis=0)
        average = running / np.arange(1, len(running) + 1, dtype=np.float32)[:, None]
        return (average[old_length:] @ self.attention_weight)[None]

    def mlp(self, x):
        return np.tanh(x @ self.mlp_weight)

    def __call__(self, x, mask, cache):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class FakeModel:
    def __init__(self, width=4, vocab=20):
        rng = np.random.default_rng(91)
        self.embedding = rng.normal(size=(vocab, width)).astype(np.float32)
        self.head = rng.normal(size=(width, vocab)).astype(np.float32)
        self.args = SimpleNamespace(hidden_size=width)
        self.model = SimpleNamespace(
            layers=[FakeLayer(width, 17), FakeLayer(width, 18)],
            layer_types=["sliding_attention", "full_attention"],
            window_size=128, ga_idx=1, swa_idx=0,
            embed_tokens=lambda ids: self.embedding[ids], norm=lambda x: x,
        )

    def lm_head(self, x):
        return np.einsum("...d,dv->...v", x, self.head, optimize=False)

    def __call__(self, ids, cache):
        x = self.model.embed_tokens(ids)
        for layer, entry in zip(self.model.layers, cache):
            x = layer(x, None, entry)
        return self.lm_head(x)


FINAL_HEADER = "<|start|>assistant<|channel|>final<|message|>"


class FakeTokenizer:
    pieces = {
        10: "<|end|>", 11: FINAL_HEADER, 12: '{"action": "LEFT"}',
        13: "<|return|>", 14: "<|call|>",
        15: '{"action": "UP"}', 16: "<|endoftext|>",
    }

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(i, "thinking ") for i in ids)


def make_engine(generation=None):
    model = FakeModel()
    def sampler_factory(temp):
        sequence = iter(generation)
        return lambda logits: np.array([next(sequence)], dtype=np.int32)
    return MLXInterventionEngine.from_components(
        model=model, tokenizer=FakeTokenizer(), backend=NumpyBackend,
        make_cache=lambda m: [FakeCache(m.args.hidden_size) for _ in m.model.layers],
        create_attention_mask=lambda x, cache, window_size=None: None,
        candidate_ids=[1, 2, 3, 4], suffix_ids=[5, 6],
        terminal_ids={13: "return", 14: "tool_call", 16: "eos"},
        sampler_factory=sampler_factory if generation is not None else None,
    )


class EngineTests(unittest.TestCase):
    def test_offline_missing_snapshot_fails_before_mlx_import(self):
        engine = MLXInterventionEngine("/definitely/not/a/model/snapshot")
        self.assertIsNone(engine._model)
        with self.assertRaises(FileNotFoundError):
            engine.prefill([1, 2])

    def test_manual_native_and_patched_cached_reference_parity(self):
        report = make_engine().validate([1, 2, 3, 4], tolerance=2e-6)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["cache_unchanged"])
        self.assertGreater(report["cases"][1]["tokens"], 160)
        self.assertEqual(report["model"]["head_mode"], "full")
        self.assertTrue(report["cases"][1]["screening_multitoken"]["nonvacuous_downstream_logit_effect"])
        self.assertEqual(len(report["cases"][1]["screening_multitoken"]["positions"]), 3)

    def test_bf16_logging_converts_before_numpy_buffer_access(self):
        # Emulate the real MLX BF16 buffer error without importing MLX.
        class FloatValues(np.ndarray):
            def astype(self, dtype, *args, **kwargs):
                if dtype == "bf16-sentinel":
                    return BF16Values(np.asarray(self, dtype=np.float32))
                return np.asarray(self, dtype=dtype).view(FloatValues)
        class BF16Values:
            def __init__(self, values):
                self.values = values
            def astype(self, dtype):
                if dtype != np.float32:
                    raise AssertionError("expected explicit float32 conversion")
                return self.values.view(FloatValues)
            def __array__(self, *args, **kwargs):
                raise RuntimeError("PEP 3118 BF16 format mismatch")
        class Tensor:
            dtype = "bf16-sentinel"
            def __getitem__(self, key):
                return BF16Values(np.ones((1, 1, 4), dtype=np.float32))
            def __setitem__(self, key, value):
                self.updated = value
        engine, logs = make_engine(), []
        engine._patch_tensor(Tensor(), {"layer": 0, "site": "resid_post", "positions": [0],
                                       "operation": "replace", "dose": 1.0, "donor": np.zeros((1, 4))}, logs)
        self.assertEqual(logs[0]["delta_l2"], [2.0])
        self.assertNotIn("recipient_float32_sha256", logs[0])
        guarded_logs = []
        expected = array_fingerprint(np.ones((1, 4), dtype=np.float32))
        engine._patch_tensor(Tensor(), {"layer": 0, "site": "resid_post", "positions": [0],
                                       "operation": "replace", "dose": 1.0, "donor": np.zeros((1, 4)),
                                       "expected_recipient_float32_sha256": expected}, guarded_logs)
        self.assertEqual(guarded_logs[0]["recipient_float32_sha256"], expected)
        self.assertEqual(guarded_logs[0]["delta_l2"], [2.0])

    def test_recipient_guard_uses_actual_prewrite_states_without_changing_patch_numerics(self):
        engine = make_engine()
        ids, positions = [1, 2, 3, 4], [3, 1]
        for site in ("attn_out", "mlp_out", "resid_post"):
            with self.subTest(site=site):
                original = engine.capture(ids, [0], positions, site)[0]
                expected = array_fingerprint(original)
                patch = {"layer": 0, "site": site, "positions": positions,
                         "operation": "subspace", "basis": np.eye(4, dtype=np.float32)[:, :2],
                         "donor": original + 0.25, "dose": 0.5}
                plain = engine.prefill(ids, [patch])
                guarded = engine.prefill(ids, [{**patch, "expected_recipient_float32_sha256": expected.upper()}])
                np.testing.assert_array_equal(plain.next_logits, guarded.next_logits)
                for before, after in zip(plain.cache, guarded.cache, strict=True):
                    np.testing.assert_array_equal(before.values, after.values)
                observed = guarded.patch_log[0].copy()
                self.assertEqual(observed.pop("recipient_float32_sha256"), expected)
                self.assertEqual(observed, plain.patch_log[0])

    def test_same_norm_wrong_recipient_rejected_before_write_even_at_zero_dose(self):
        engine = make_engine()
        original = np.array([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=np.float32)
        wrong = -original[0, [1]]
        np.testing.assert_array_equal(np.linalg.norm(wrong, axis=-1), np.linalg.norm(original[0, [1]], axis=-1))
        for dose in (0, 1):
            tensor, logs = original.copy(), []
            patch = {"layer": 0, "site": "resid_post", "positions": [1], "dose": dose,
                     "operation": "replace", "donor": np.zeros((1, 4), dtype=np.float32),
                     "expected_recipient_float32_sha256": array_fingerprint(wrong)}
            with self.assertRaisesRegex(RecipientStateMismatch, "mismatch before patch write") as caught:
                engine._patch_tensor(tensor, patch, logs)
            np.testing.assert_array_equal(tensor, original)
            self.assertEqual(logs, [])
            diagnostic = caught.exception.patch_audit
            self.assertEqual(diagnostic["recipient_float32_sha256"], array_fingerprint(original[0, [1]]))
            self.assertEqual(diagnostic["expected_recipient_float32_sha256"], array_fingerprint(wrong))
            self.assertEqual(diagnostic["positions"], [1])
            self.assertEqual(diagnostic["dose"], dose)
            self.assertEqual(diagnostic["before_l2"], np.linalg.norm(original[0, [1]], axis=-1).tolist())
            self.assertFalse(diagnostic["write_applied"])
            self.assertEqual(diagnostic["preceding_patch_count"], 0)
            self.assertEqual(diagnostic["recipient_shape"], [1, 4])
            self.assertIsNone(diagnostic["rank"])
            json.dumps(diagnostic, allow_nan=False)

    def test_multiple_disjoint_patches_keep_exact_ordered_recipient_guards(self):
        engine = make_engine()
        ids, site = [1, 2, 3, 4], "resid_post"
        patches = []
        for positions, offset in (([3, 1], 0.5), ([0, 2], -0.25)):
            original = engine.capture(ids, [0], positions, site)[0]
            patches.append({"layer": 0, "site": site, "positions": positions,
                            "operation": "replace", "dose": 1, "donor": original + offset,
                            "expected_recipient_float32_sha256": array_fingerprint(original)})
        guarded = engine.prefill(ids, patches)
        plain = engine.prefill(ids, [{k: v for k, v in patch.items() if k != "expected_recipient_float32_sha256"} for patch in patches])
        np.testing.assert_array_equal(guarded.next_logits, plain.next_logits)
        self.assertEqual([log["positions"] for log in guarded.patch_log], [[3, 1], [0, 2]])
        self.assertEqual([log["recipient_float32_sha256"] for log in guarded.patch_log],
                         [patch["expected_recipient_float32_sha256"] for patch in patches])
        reordered = {**patches[0], "expected_recipient_float32_sha256":
                     array_fingerprint(engine.capture(ids, [0], [1, 3], site)[0])}
        with self.assertRaisesRegex(ValueError, "mismatch before patch write"):
            engine.prefill(ids, [reordered])

    def test_prior_layer_intervention_invalidates_later_original_recipient_hash(self):
        engine = make_engine()
        ids, positions = [1, 2, 3, 4], [2, 3]
        captured = engine.capture(ids, [0, 1], positions)
        patches = [{"layer": layer, "site": "resid_post", "positions": positions,
                    "operation": "replace", "dose": 1, "donor": captured[layer] + 2,
                    "expected_recipient_float32_sha256": array_fingerprint(captured[layer])}
                   for layer in (0, 1)]
        original_patch = engine._patch_tensor
        events = []
        def inspect_write(tensor, patch, logs):
            before = tensor.copy()
            try:
                result = original_patch(tensor, patch, logs)
            except ValueError:
                np.testing.assert_array_equal(tensor, before)
                events.append((patch["layer"], "rejected_before_write", len(logs)))
                raise
            events.append((patch["layer"], "applied", len(logs)))
            return result
        with mock_patch.object(engine, "_patch_tensor", side_effect=inspect_write):
            with self.assertRaisesRegex(RecipientStateMismatch, "mismatch before patch write at layer 1") as caught:
                engine.prefill(ids, patches)
        self.assertEqual(events, [(0, "applied", 1), (1, "rejected_before_write", 1)])
        self.assertEqual(caught.exception.patch_audit["preceding_patch_count"], 1)
        self.assertEqual(caught.exception.patch_audit["layer"], 1)

    def test_malformed_expected_recipient_hash_rejected_before_manual_inference(self):
        engine = make_engine()
        common = {"layer": 0, "positions": [1], "donor": np.zeros((1, 4)), "operation": "replace"}
        for value in (None, True, 7, "", "a" * 63, "g" * 64, "a" * 65, " " + "a" * 64, b"a" * 64):
            with self.subTest(value=value):
                with mock_patch.object(engine, "_manual", side_effect=AssertionError("must validate before inference")):
                    with self.assertRaisesRegex(ValueError, "64-character hexadecimal"):
                        engine.prefill([1, 2, 3], [{**common, "expected_recipient_float32_sha256": value}])

    def test_identity_and_zero_dose_preserve_state(self):
        engine = make_engine()
        ids, positions = [1, 2, 3, 4], [1, 3]
        baseline = engine.prefill(ids)
        for site in ("attn_out", "mlp_out", "resid_post"):
            donor = engine.capture(ids, [0], positions, site)[0]
            for operation, dose, values in (("replace", 1, donor), ("replace", 0, donor + 3), ("add", 0, donor + 3)):
                state = engine.prefill(ids, [{"layer": 0, "site": site, "positions": positions, "donor": values, "operation": operation, "dose": dose}])
                np.testing.assert_array_equal(state.next_logits, baseline.next_logits)
                np.testing.assert_array_equal(state.patch_log[0]["delta_l2"], [0, 0])

    def test_full_basis_equals_replacement_and_partial_dose_interpolates(self):
        engine = make_engine()
        ids, positions = [1, 2, 3], [2]
        donor = engine.capture([4, 5, 6], [0], positions)[0]
        common = {"layer": 0, "positions": positions, "donor": donor, "dose": 0.5}
        replacement = engine.prefill(ids, [{**common, "operation": "replace"}])
        subspace = engine.prefill(ids, [{**common, "operation": "subspace", "basis": np.eye(4)}])
        np.testing.assert_allclose(replacement.next_logits, subspace.next_logits, atol=1e-6)
        baseline = engine.prefill(ids)
        self.assertGreater(np.linalg.norm(replacement.next_logits - baseline.next_logits), 0.01)
        before = engine.capture(ids, [0], positions)[0]
        np.testing.assert_allclose(replacement.patch_log[0]["delta_l2"], np.linalg.norm((donor - before) * 0.5, axis=1), atol=1e-6)

    def test_patched_memory_survives_decoding_and_cache_branches_are_independent(self):
        engine = make_engine()
        ids, positions = [1, 2, 3], [1, 2]
        donor = engine.capture([7, 8, 9], [0], positions)[0]
        patch = {"layer": 0, "positions": positions, "donor": donor, "operation": "replace"}
        state, baseline = engine.prefill(ids, [patch]), engine.prefill(ids)
        state_values = [c.values.copy() for c in state.cache]
        first = engine.forced(state)
        second = engine.forced(state)
        self.assertEqual(first, second)
        self.assertGreater(abs(first["q_UP"] - engine.forced(baseline)["q_UP"]), 1e-5)
        for old, cache in zip(state_values, state.cache):
            np.testing.assert_array_equal(old, cache.values)
        cloned = engine.clone_state(state)
        cloned.cache[0].values[0, 0] += 9
        self.assertNotEqual(cloned.cache[0].values[0, 0], state.cache[0].values[0, 0])
        with self.assertRaises(ValueError):
            make_engine().forced(state)

    def test_invalid_patch_contracts_fail(self):
        engine = make_engine()
        common = {"layer": 0, "positions": [1], "donor": np.zeros((1, 4))}
        for changes in (
            {"layer": 2}, {"site": "kv"}, {"positions": [1, 1]}, {"positions": [-1]},
            {"positions": [3]}, {"donor": np.zeros((2, 4))}, {"dose": np.nan},
            {"operation": "subspace", "basis": np.ones((4, 1))},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                engine.prefill([1, 2, 3], [{**common, **changes}])
        with self.assertRaises(ValueError):
            engine.prefill([1, 2, 3], [common, common])

    def test_generation_crosses_analysis_end_and_reads_native_final(self):
        engine = make_engine([15, 10, 11, 12, 13])
        state = engine.prefill([1, 2, 3])
        initial = [c.values.copy() for c in state.cache]
        result = engine.generate(state, seed=72, max_new_tokens=10)
        self.assertEqual(result["action"], "LEFT")
        self.assertTrue(result["completed"])
        self.assertEqual(result["n_new_tokens"], 5)
        self.assertEqual(result["stop_reason"], "return")
        self.assertEqual(result["generated_token_ids"], [15, 10, 11, 12, 13])
        for old, entry in zip(initial, state.cache):
            np.testing.assert_array_equal(old, entry.values)

    def test_capped_json_is_not_a_completed_natural_action(self):
        engine = make_engine([10, 11, 12, 13])
        result = engine.generate(engine.prefill([1]), seed=1, max_new_tokens=3)
        self.assertEqual(result["parsed_action"], "LEFT")
        self.assertIsNone(result["action"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["completed"])

    def test_tool_call_is_not_natural_action(self):
        engine = make_engine([11, 12, 14])
        result = engine.generate(engine.prefill([1]), seed=1)
        self.assertIsNone(result["action"])
        self.assertFalse(result["ended_naturally"])

    def test_probability_metrics_are_conditional_and_finite(self):
        metrics = action_metrics([1000, 1001, 999, 1002, 9999], [0, 1, 2, 3])
        self.assertAlmostEqual(sum(metrics[f"q_{a}"] for a in ("UP", "DOWN", "LEFT", "RIGHT")), 1)
        self.assertEqual(metrics["top_action"], "RIGHT")
        self.assertEqual(metrics["candidate_mass"], 0)


class NativeParserTests(unittest.TestCase):
    def test_reasoning_json_and_bare_json_are_rejected(self):
        for text in ('{"action":"UP"}', '<|start|>assistant<|channel|>analysis<|message|>{"action":"UP"}<|return|>'):
            self.assertIsNone(parse_native_action(text)["parsed_action"])

    def test_malformed_actions_and_trailing_prose_are_rejected(self):
        for body in ('{"action":"WAIT"}', '{"action":3}', '{"action":"UP"} so done', '{"action":', '["UP"]'):
            with self.subTest(body=body):
                self.assertIsNone(parse_native_action(FINAL_HEADER + body + "<|return|>")["parsed_action"])

    def test_valid_final_does_not_read_an_earlier_reasoning_json(self):
        text = '{"action":"DOWN"}<|end|>' + FINAL_HEADER + '{"action":"RIGHT"}<|return|>'
        self.assertEqual(parse_native_action(text)["parsed_action"], "RIGHT")

    def test_multiple_final_channels_are_explicit(self):
        text = FINAL_HEADER + '{"action":"RIGHT"}<|end|>' + FINAL_HEADER + '{"action":"UP"}<|return|>'
        self.assertEqual(parse_native_action(text)["parse_status"], "multiple_final_messages")


if __name__ == "__main__":
    unittest.main()
