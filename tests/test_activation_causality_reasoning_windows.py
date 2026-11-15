"""Exact-ID and conservative-selection tests for exploratory CoT windows."""

from copy import deepcopy
import json
import unittest

from src.activation_causality.reasoning_windows import build_reasoning_window_spec
from src.activation_causality.registry import fingerprint


class FakeTokenizer:
    """Generated IDs intentionally cannot be recovered by re-encoding their text."""
    def __init__(self):
        self.encode_calls = []
        self.tokens = {1: "BASE-R<|channel|>analysis<|message|>", 2: "BASE-D<|channel|>analysis<|message|>",
                       10: "Think ", 11: "carefully ", 12: "about ",
                       13: "the ", 14: "map. ", 15: "More ", 16: "facts. ", 17: "Done. ",
                       20: "<|end|>", 21: "<|start|>assistant<|channel|>final<|message|>",
                       22: '{"action":"UP"}', 23: "<|fim_suffix|>",
                       30: "up ", 31: "RIGHT ", 32: "upright ", 33: "u", 34: "p ",
                       35: "pick ", 36: "negated ", 37: "upward ", 40: "SHORT"}

    def encode(self, text, add_special_tokens=False):
        self.encode_calls.append(text)
        if text == "BASE-R":
            return [1]
        if text == "BASE-D":
            return [2]
        raise AssertionError("Generated text must never be encoded")

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.tokens[token] for token in ids)


def fixtures(seeds=(42,), *, generated=None):
    tokens = generated if generated is not None else [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23]
    jobs, results = [], []
    for seed in seeds:
        for condition, base in (("baseline", "BASE-R"), ("counterfactual_text", "BASE-D")):
            job_id = f"{condition}-{seed}"
            job = {"job_id": job_id, "experiment_id": "source", "pair_id": "pair", "layout_id": "layout",
                   "condition": condition, "partition": "discovery", "family": "wall", "seed": seed,
                   "recipient_text": base, "target_action": "RIGHT", "original_action": "LEFT",
                   "natural": True, "intervention": {"layers": []}, "metadata": {}, "layout_aliases": ["layout", "wall-variant"]}
            jobs.append(job)
            results.append({**{key: job[key] for key in ("experiment_id", "condition", "layout_id", "pair_id", "seed", "target_action")},
                            "job_id": job_id, "status": "complete", "generation_status": "complete",
                            "generated_token_ids": list(tokens), "generated_text": "UNTRUSTED DISPLAY TEXT",
                            "input_token_sha256": fingerprint([1 if base == "BASE-R" else 2]), "native_action": "UP"})
    return {"experiment_id": "source", "partition": "discovery", "jobs": jobs}, results


class ReasoningWindowTests(unittest.TestCase):
    def build(self, source=None, rows=None, **kwargs):
        if source is None:
            source, rows = fixtures()
        self.tokenizer = FakeTokenizer()
        return build_reasoning_window_spec(source, rows, self.tokenizer, "reasoning_test",
                                           cuts=kwargs.pop("cuts", (4,)), layers=(10,), last_k=2, **kwargs)

    def test_exact_original_and_generated_ids_never_retokenized(self):
        source, rows = fixtures()
        original = deepcopy((source, rows))
        spec = self.build(source, rows)
        self.assertEqual((source, rows), original)
        treated = next(job for job in spec["jobs"] if job["condition"] == "layer10_cut4")
        self.assertEqual(treated["recipient_token_ids"], [1, 10, 11, 12, 13])
        self.assertEqual(treated["donor_token_ids"], [2, 10, 11, 12, 13])
        self.assertEqual(treated["metadata"]["recipient_patch_positions"], [3, 4])
        self.assertEqual(set(self.tokenizer.encode_calls), {"BASE-R", "BASE-D"})
        self.assertEqual(treated["pair_id"], "pair::cut4")
        self.assertFalse(treated["semantic_test"])
        self.assertEqual(json.loads(json.dumps(spec)), spec)

    def test_analysis_boundary_never_included_and_equal_boundary_allowed(self):
        spec = self.build(cuts=(8, 9, 10))
        cuts = {job["metadata"]["cut"] for job in spec["jobs"]}
        self.assertEqual(cuts, {8})
        self.assertIn("cut_reaches_analysis_end_or_channel_marker", str(spec["eligibility_manifest"]))
        self.assertTrue(all(20 not in job["recipient_token_ids"] for job in spec["jobs"]))

    def test_case_insensitive_first_mention_including_partial_word(self):
        for word in (30, 31):
            source, rows = fixtures(generated=[10, 11, word, 14, 15])
            spec = self.build(source, rows, cuts=(2, 3))
            self.assertEqual({job["metadata"]["cut"] for job in spec["jobs"]}, {2})
        source, rows = fixtures(generated=[10, 11, 33, 34, 15])
        spec = self.build(source, rows, cuts=(2, 3))
        self.assertEqual({job["metadata"]["cut"] for job in spec["jobs"]}, {2})

    def test_lexical_rule_is_conservative_about_idioms_but_not_substrings(self):
        source, rows = fixtures(generated=[10, 35, 30, 14, 15])
        self.assertFalse(self.build(source, rows)["jobs"])
        source, rows = fixtures(generated=[10, 32, 37, 14, 15])
        self.assertTrue(self.build(source, rows)["jobs"])

    def test_one_ineligible_seed_rejects_whole_pair_cut(self):
        source, rows = fixtures(seeds=(42, 43))
        rows[-1]["generated_token_ids"] = [10, 30, 14, 15]
        spec = self.build(source, rows)
        self.assertFalse(spec["jobs"])
        self.assertFalse(spec["eligibility_manifest"][0]["eligible_donor_cohorts"]["matched"])
        self.assertEqual(spec["eligibility_manifest"][0]["seeds"], [42, 43])

    def test_missing_result_short_prefix_and_hash_drift_are_explicit(self):
        source, rows = fixtures()
        spec = self.build(source, rows[:-1])
        self.assertIn("source_result_missing", str(spec["eligibility_manifest"]))
        source, rows = fixtures(generated=[10])
        self.assertIn("insufficient_generated_tokens", str(self.build(source, rows)["eligibility_manifest"]))
        source, rows = fixtures()
        rows[0]["input_token_sha256"] = "wrong"
        self.assertIn("source_input_token_fingerprint_mismatch", str(self.build(source, rows)["eligibility_manifest"]))

    def test_truncated_native_sources_are_usable_without_success_filter(self):
        source, rows = fixtures()
        for row in rows:
            row.update({"generation_status": "truncated", "native_action": None})
        self.assertTrue(self.build(source, rows)["jobs"])

    def test_overrides_preserve_original_input_and_never_encode_it(self):
        source, rows = fixtures()
        for job in source["jobs"]:
            job["recipient_token_ids"] = [1 if job["condition"] == "baseline" else 2]
            job["recipient_text"] = "DISPLAY-ONLY"
        spec = self.build(source, rows)
        self.assertTrue(spec["jobs"])
        self.assertFalse(self.tokenizer.encode_calls)

    def test_no_prompt_fallback_and_confirmation_rejected(self):
        self.assertFalse(self.build(cuts=(1,))["jobs"])
        source, rows = fixtures()
        source["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            self.build(source, rows)

    def test_natural_donor_has_separate_condition_and_independent_eligibility(self):
        source, rows = fixtures()
        donor_job, donor_row = deepcopy(source["jobs"][-1]), deepcopy(rows[-1])
        donor_job.update({"condition": "natural_donor_text", "job_id": "natural-42"})
        donor_row["job_id"] = "natural-42"
        donor_row["condition"] = "natural_donor_text"
        source["jobs"].append(donor_job)
        rows.append(donor_row)
        spec = self.build(source, rows)
        self.assertIn("layer10_natural_cut4", {job["condition"] for job in spec["jobs"]})
        self.assertEqual(len([job for job in spec["jobs"] if job["condition"] == "baseline_cut4"]), 1)
        comparison = next(c for c in spec["comparisons"] if c["treatment"] == "layer10_natural_cut4")
        self.assertEqual(comparison["eligible_pair_ids"], ["pair::cut4"])
        self.assertEqual(comparison["expected_n_pairs"], 1)

    def test_final_channel_input_cannot_be_called_reasoning(self):
        source, rows = fixtures()
        tokenizer = FakeTokenizer()
        tokenizer.tokens[1] = "BASE-R<|channel|>final<|message|>"
        spec = build_reasoning_window_spec(source, rows, tokenizer, "wrong_channel", cuts=(4,), last_k=2)
        self.assertFalse(spec["jobs"])
        self.assertIn("source_input_is_not_in_analysis_channel", str(spec["eligibility_manifest"]))

    def test_all_layers_reference_uses_same_window_without_donor_identity_claim(self):
        spec = self.build()
        job = next(job for job in spec["jobs"] if job["condition"] == "all_layers_full_cut4")
        self.assertEqual(job["intervention"]["layers"], list(range(24)))
        self.assertEqual(job["selection"], {"scope": "last_tokens", "last_k": 2})
        self.assertEqual(job["metadata"]["recipient_patch_positions"], [3, 4])
        self.assertTrue(job["metadata"]["all_layers_window_transfer"])
        self.assertFalse(job["metadata"]["full_input_state_replaced"])
        self.assertFalse(job["metadata"]["donor_continuation_identity_expected"])
        comparisons = [row for row in spec["comparisons"] if row["treatment"] == "all_layers_full_cut4"]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["control"], "baseline_cut4")
        self.assertEqual(comparisons[0]["predicted_direction"], "increase")
        self.assertFalse(any(job["condition"].startswith("all_layers") for job in self.build(include_all_layers=False)["jobs"]))


if __name__ == "__main__":
    unittest.main()
