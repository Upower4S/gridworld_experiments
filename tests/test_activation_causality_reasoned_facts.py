"""CPU-only exact-prefix, provenance, and control tests for late facts."""

from copy import deepcopy
import json
import re
import unittest

from src.activation_causality.environments import generate_pairs
from src.activation_causality.fact_interventions import ACTION_WORD
from src.activation_causality.reasoned_facts import build_reasoned_fact_spec
from src.activation_causality.registry import fingerprint


class Tokenizer:
    """Generated tokens have no encoder inverse; facts use a separate vocabulary."""
    def __init__(self, split_false=False):
        self.calls, self.words = [], {}
        self.split_false = split_false
        self.tokens = {1: "BASE<|channel|>analysis<|message|>", 10: "Consider ", 11: "the ",
                       12: "map ", 13: "carefully. ", 14: "Think ", 15: "again. ",
                       20: "<|end|>", 21: "<|start|>assistant<|channel|>final<|message|>",
                       22: '{"action":"UP"}', 23: "up ", 24: "u", 25: "p ", 26: "upright "}

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if text == "BASE":
            return [1]
        if not text.startswith(" "):
            raise AssertionError("Re-encoding a generated prefix or combined display string is forbidden")
        pieces = re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", text)
        if self.split_false:
            pieces = [piece for value in pieces for piece in (["Fal", "se"] if value == "False" else [value])]
        encoded = []
        for piece in pieces:
            token = self.words.setdefault(piece, len(self.words) + 100)
            self.tokens[token] = piece + " "
            encoded.append(token)
        return encoded

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.tokens[token] for token in ids)


def fixture(pair, seeds=(42,), generated=None):
    jobs, rows = [], []
    generated = [10, 11, 12, 13, 14, 15, 20, 21, 22] if generated is None else generated
    for seed in seeds:
        job = {"job_id": f"base-{seed}", "experiment_id": "source", "pair_id": pair["pair_id"],
               "layout_id": pair["layout_id"], "family": pair["family"], "partition": "discovery", "condition": "baseline",
               "seed": seed, "target_action": pair["counterfactual_optimal_actions"][0],
               "original_action": pair["recipient_optimal_actions"][0], "recipient_text": "BASE",
               "natural": True, "intervention": {"layers": []}, "metadata": {"pair": deepcopy(pair)},
               "layout_aliases": [pair["layout_id"], "source-extra-alias"]}
        jobs.append(job)
        rows.append({**{key: job[key] for key in ("job_id", "experiment_id", "condition", "layout_id", "pair_id", "seed", "target_action")},
                     "status": "complete", "generation_status": "complete", "generated_token_ids": list(generated),
                     "generated_text": "DISPLAY STRING MUST NEVER BE USED", "input_token_sha256": fingerprint([1]),
                     "native_action": None})
    return {"experiment_id": "source", "partition": "discovery", "jobs": jobs}, rows


class ReasonedFactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20261, "discovery")

    def build(self, source=None, rows=None, tokenizer=None, **kwargs):
        if source is None:
            source, rows = fixture(self.pairs[0])
        self.tokenizer = tokenizer or Tokenizer()
        return build_reasoned_fact_spec(source, rows, self.tokenizer, "latefacts", cuts=kwargs.pop("cuts", (4,)),
                                        layers=(10,), last_k=kwargs.pop("last_k", 2), **kwargs)

    def test_exact_shared_generated_prefix_separate_fact_encoding_and_no_mutation(self):
        source, rows = fixture(self.pairs[0])
        original = deepcopy((source, rows))
        spec = self.build(source, rows)
        self.assertTrue(spec["jobs"])
        self.assertEqual((source, rows), original)
        self.assertEqual(json.loads(json.dumps(spec)), spec)
        for job in spec["jobs"]:
            self.assertEqual(job["recipient_token_ids"][:5], [1, 10, 11, 12, 13])
            if job["donor_token_ids"] is not None:
                self.assertEqual(job["donor_token_ids"][:5], [1, 10, 11, 12, 13])
            meta = job["metadata"]
            self.assertEqual(meta["fact_start_token"], 5)
            self.assertGreaterEqual(min(job["selection"]["recipient_positions"]), 5)
            self.assertEqual(meta["source_pair_id"], self.pairs[0]["pair_id"])
            self.assertTrue(meta["genuinely_generated_prefix"])
            self.assertFalse(meta["naturally_generated_fact_sentence"])
            self.assertFalse(meta["generated_text_retokenized"])
            self.assertFalse(meta["full_display_string_canonical_bpe_assumed"])
            self.assertFalse(ACTION_WORD.search(meta["recipient_sentence"]))
        self.assertTrue(all(text == "BASE" or text.startswith(" ") for text in self.tokenizer.calls))

    def test_only_recipient_sources_needed_and_all_predeclared_seeds_required(self):
        source, rows = fixture(self.pairs[0], seeds=(42, 43))
        spec = self.build(source, rows)
        self.assertTrue(spec["jobs"])
        self.assertTrue(all(row["expected_n_pairs"] == 2 for row in spec["comparisons"]))
        self.assertFalse(spec["protocol"]["counterfactual_prefix_required"])
        missing = self.build(source, rows[:-1])
        self.assertFalse(missing["jobs"])
        self.assertEqual(missing["eligibility_manifest"][0]["seeds"], [42, 43])
        self.assertIn("source_result_missing", str(missing["eligibility_manifest"]))
        rows[-1]["generated_token_ids"] = [10, 23, 12, 13]
        self.assertFalse(self.build(source, rows)["jobs"])

    def test_analysis_end_direction_and_partial_word_boundaries(self):
        spec = self.build(cuts=(6, 7, 8))
        self.assertEqual({job["metadata"]["cut"] for job in spec["jobs"]}, {6})
        for tokens in ([10, 11, 23, 13], [10, 11, 24, 25, 13]):
            source, rows = fixture(self.pairs[0], generated=tokens)
            spec = self.build(source, rows, cuts=(2, 3))
            self.assertEqual({job["metadata"]["cut"] for job in spec["jobs"]}, {2})
        source, rows = fixture(self.pairs[0], generated=[10, 11, 26, 13])
        self.assertTrue(self.build(source, rows)["jobs"])

    def test_source_result_provenance_hash_and_technical_errors_excluded(self):
        for mutation, reason in (({"target_action": "OTHER"}, "provenance_mismatch_target_action"),
                                 ({"input_token_sha256": "wrong"}, "fingerprint_mismatch"),
                                 ({"status": "error"}, "source_execution_failed")):
            source, rows = fixture(self.pairs[0])
            rows[0].update(mutation)
            spec = self.build(source, rows)
            self.assertFalse(spec["jobs"])
            self.assertIn(reason, str(spec["eligibility_manifest"]))

    def test_truncated_native_prefix_usable_without_action_success_filter(self):
        source, rows = fixture(self.pairs[0])
        rows[0].update(generation_status="truncated", native_action=None)
        self.assertTrue(self.build(source, rows)["jobs"])
        source["jobs"][0]["intervention"]["layers"] = [10]
        self.assertIn("source_is_not_unpatched_baseline", str(self.build(source, rows)["eligibility_manifest"]))

    def test_reverse_identity_and_full_span_controls_are_aligned(self):
        spec = self.build()
        for job in spec["jobs"]:
            meta = job["metadata"]
            self.assertEqual(job["target_action"], self.pairs[0]["counterfactual_optimal_actions"][0])
            if job["condition"].startswith("identity"):
                self.assertEqual(job["recipient_token_ids"], job["donor_token_ids"])
                self.assertEqual(job["selection"]["recipient_positions"], job["selection"]["donor_positions"])
            if job["condition"].startswith("reverse"):
                self.assertFalse(meta["recipient_fact_true_in_prompt"])
                self.assertEqual(meta["paired_baseline_condition"], "donor_text_cut4")
            if job["condition"].startswith("all_layers_full"):
                self.assertEqual(job["intervention"]["layers"], list(range(24)))
                self.assertEqual(min(job["selection"]["recipient_positions"]), meta["fact_start_token"])
        self.assertTrue(all(row["predicted_direction"] == "decrease" for row in spec["comparisons"] if row["role"] == "reverse_fact_replacement"))
        self.assertTrue(all(row["predicted_direction"] == "zero" for row in spec["comparisons"] if row["role"] in {"engineering_identity", "full_span_positive_control_agreement"}))

    def test_irrelevant_facts_have_equal_lengths_and_own_baseline(self):
        spec = self.build()
        record = spec["eligibility_manifest"][0]
        self.assertTrue(record["irrelevant_eligible"])
        self.assertEqual(record["fact_suffixes"]["token_count"], record["irrelevant_suffixes"]["token_count"])
        comparison = next(row for row in spec["comparisons"] if row["role"] == "irrelevant_state_fact_sensitivity")
        self.assertEqual(comparison["control"], "irrelevant_baseline_cut4")
        self.assertEqual(comparison["predicted_direction"], "zero")

    def test_unequal_fact_lengths_and_too_wide_windows_remain_in_manifest(self):
        spec = self.build(tokenizer=Tokenizer(split_false=True))
        self.assertFalse(spec["jobs"])
        self.assertIn("unequal_separately_encoded_fact_lengths", str(spec["eligibility_manifest"]))
        spec = self.build(last_k=1000)
        self.assertFalse(spec["jobs"])
        self.assertIn("fact_sentence_shorter_than_patch_window", str(spec["eligibility_manifest"]))

    def test_base_override_is_authoritative_and_cut_may_be_shorter_than_fact_window(self):
        source, rows = fixture(self.pairs[0])
        source["jobs"][0].update(recipient_text="DISPLAY ONLY", recipient_token_ids=[1])
        spec = self.build(source, rows, cuts=(1,), last_k=12)
        self.assertTrue(spec["jobs"])
        self.assertTrue(all(text.startswith(" ") for text in self.tokenizer.calls))
        self.assertTrue(all(job["recipient_token_ids"][:2] == [1, 10] for job in spec["jobs"]))

    def test_partition_lineage_and_duplicates_fail_closed(self):
        source, rows = fixture(self.pairs[0])
        source["partition"] = "confirmation"
        with self.assertRaises(ValueError):
            self.build(source, rows)
        for field, value in (("experiment_id", "other"), ("layout_id", "other"), ("target_action", "OTHER")):
            source, rows = fixture(self.pairs[0])
            source["jobs"][0][field] = value
            with self.assertRaises(ValueError):
                self.build(source, rows)
        source, rows = fixture(self.pairs[0])
        with self.assertRaises(ValueError):
            self.build(source, rows + rows)
        source["jobs"][0]["metadata"] = {}
        with self.assertRaises(ValueError):
            self.build(source, rows)

    def test_all_three_fact_families_supported_and_aliases_preserved(self):
        for pair in self.pairs:
            source, rows = fixture(pair)
            spec = self.build(source, rows)
            self.assertTrue(spec["jobs"])
            self.assertEqual({job["family"] for job in spec["jobs"]}, {pair["family"]})
            self.assertIn("source-extra-alias", spec["layout_aliases"][pair["layout_id"]])
            self.assertTrue(all(not job["semantic_test"] for job in spec["jobs"]))


if __name__ == "__main__":
    unittest.main()
