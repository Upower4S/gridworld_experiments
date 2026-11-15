"""CPU-only confirmation contract, exclusion accounting and freeze checks."""

from copy import deepcopy
import json
import re
import unittest

from src.activation_causality.confirmation_adapters import (
    INPUT_ALIGNMENT_RULE, MIXED_INPUT_FAMILIES, NATIVE_ITT_ENDPOINT, build_frozen_input_spec,
)
from src.activation_causality.environments import generate_pairs, optimal_actions
from src.activation_causality.input_sites import build_input_site_spec
from src.activation_causality.registry import fingerprint


class Tokenizer:
    name_or_path = "reproduction_fixture"

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("no extra tokenizer framing")
        self.calls.append(text)
        return [300 if p == "False" else 301 if p == "True" else ord(p)
                for p in re.findall(r"False|True|[\s\S]", text)]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class ConfirmationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(3, 20260913, partition="confirmation",
                                   families=("wall", "key_possession", "door_status"))
        cls.discovery = generate_pairs(3, 20260912, partition="discovery",
                                       families=("wall", "key_possession", "door_status"))

    def configuration(self, **updates):
        config = {"layer": 10, "site": "resid_post", "generation_seeds": [42, 43],
                  "max_new_tokens": 4096, "n_layouts": 32,
                  "treatment_condition": "factspan_layer10", "control_condition": "baseline"}
        config.update(updates)
        return config

    def build(self, pairs=None, tokenizer=None, configuration=None, freeze_id="frozen_test"):
        return build_frozen_input_spec(self.pairs if pairs is None else pairs,
                                       Tokenizer() if tokenizer is None else tokenizer,
                                       "input_confirmation_test", configuration or self.configuration(), freeze_id)

    def test_single_layer_six_controls_native_itt_and_actual_counts(self):
        spec = self.build()
        conditions = {"baseline", "counterfactual_text", "identity_layer10", "factspan_layer10",
                      "random_factspan_layer10", "lasttoken_layer10"}
        self.assertEqual(len(spec["jobs"]), 3 * 6 * 2)
        self.assertEqual(spec["expected_n_layouts"], 3)
        self.assertEqual(spec["expected_n_pairs"], 6)
        self.assertEqual(spec["requested_n_layouts"], 32)
        self.assertEqual(spec["layout_shortfall"], 29)
        self.assertFalse(spec["cohort_complete"])
        self.assertEqual(spec["primary_endpoint"], NATIVE_ITT_ENDPOINT)
        self.assertIn("no forced fallback", spec["native_unresolved_policy"])
        self.assertFalse(spec["semantic_headline_eligible"])
        self.assertEqual(spec["claim_scope"], "input_fact_information_routing")
        self.assertEqual(len([c for c in spec["comparisons"] if c["confirmatory"]]), 1)
        for pair in self.pairs:
            jobs = [j for j in spec["jobs"] if j["pair_id"] == pair["pair_id"]]
            self.assertEqual({j["condition"] for j in jobs}, conditions)
            for job in jobs:
                self.assertEqual(job["partition"], "confirmation")
                self.assertEqual(job["metadata"]["pair"]["partition"], "confirmation")
                self.assertEqual(job["metadata"]["source_pair"], pair)
                self.assertEqual(job["target_action"], optimal_actions(pair["counterfactual"])[0])
                self.assertEqual(job["freeze_id"], "frozen_test")
                self.assertTrue(job["natural"])
                self.assertEqual(job["max_new_tokens"], 4096)
                self.assertIn(job["intervention"]["layers"], ([], [10]))
                self.assertFalse(job["semantic_test"])
                self.assertFalse(job["semantic_headline_eligible"])
        for comparison in spec["comparisons"]:
            self.assertEqual(comparison["expected_n_layouts"], 3)
            self.assertEqual(comparison["expected_n_pairs"], 6)
            self.assertEqual(comparison["predicted_direction"],
                             "zero" if comparison["role"] == "technical_identity" else "increase")
            self.assertFalse(comparison["semantic_headline_eligible"])
        json.dumps(spec, allow_nan=False)

    def test_all_frozen_primary_control_choices_preserve_same_jobs(self):
        jobs = None
        for control in ("baseline", "random_factspan_layer10", "lasttoken_layer10"):
            spec = self.build(configuration=self.configuration(control_condition=control))
            chosen = [c for c in spec["comparisons"] if c["confirmatory"]]
            self.assertEqual(chosen[0]["control"], control)
            self.assertEqual(chosen[0]["treatment"], "factspan_layer10")
            if jobs is not None:
                self.assertEqual(spec["jobs"], jobs)
            jobs = spec["jobs"]

    def test_alignment_exclusion_does_not_overwrite_actual_denominator(self):
        spec = self.build(tokenizer=CharacterTokenizer())
        self.assertEqual(spec["n_source_pairs"], 3)
        self.assertEqual(spec["n_aligned_pairs"], 2)
        self.assertEqual(spec["expected_n_layouts"], 2)
        self.assertEqual(spec["expected_n_pairs"], 4)
        self.assertEqual(spec["layout_shortfall"], 30)
        self.assertEqual(spec["exclusion_counts"], {"unequal_full_token_lengths": 1})
        self.assertEqual(spec["exclusions"][0]["family"], "key_possession")
        for c in spec["comparisons"]:
            self.assertEqual(c["expected_n_layouts"], 2)
            self.assertEqual(c["expected_n_pairs"], 4)
        manifest = {e["family"]: e for e in spec["exclusions"]}
        self.assertTrue(manifest["key_possession"]["excluded_before_model_query"])

    def test_partition_is_never_relabelled_and_wrong_sources_never_tokenized(self):
        relabelled = deepcopy(self.discovery[0])
        relabelled["partition"] = "confirmation"
        sources = [self.discovery[1], relabelled]
        original = deepcopy(sources)
        tokenizer = Tokenizer()
        spec = self.build(pairs=sources, tokenizer=tokenizer)
        self.assertEqual(spec["jobs"], [])
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(spec["expected_n_layouts"], 0)
        self.assertEqual(set(spec["exclusion_counts"]),
                         {"non_confirmation_pair", "confirmation_layout_reassigned_or_mismatched"})
        self.assertEqual(sources, original)

    def test_observed_or_unknown_query_provenance_cannot_be_reset(self):
        for mutation in ({"model_outputs_observed": True}, {"confirmation_status": "queried"},
                         {"model_outputs_observed": None}, {"partition": "discovery"}):
            pair = deepcopy(self.pairs[0])
            pair["metadata"].update(mutation)
            tokenizer = Tokenizer()
            with self.subTest(mutation=mutation):
                spec = self.build([pair], tokenizer)
                self.assertEqual(spec["jobs"], [])
                self.assertEqual(tokenizer.calls, [])
                self.assertEqual(spec["n_excluded_pairs"], 1)

    def test_frozen_family_exclusion_indices_stay_in_original_input_order(self):
        # The wall is input0, key input1, door input2. The selected key fails
        # token alignment, so three distinct exclusions retain original indices.
        spec = self.build(tokenizer=CharacterTokenizer(), configuration=self.configuration(family="key_possession"))
        self.assertEqual([e["input_index"] for e in spec["exclusions"]], [0, 1, 2])
        self.assertEqual([e["reason"] for e in spec["exclusions"]],
                         ["frozen_family_mismatch", "unequal_full_token_lengths", "frozen_family_mismatch"])
        self.assertEqual(spec["expected_n_layouts"], 0)
        self.assertTrue(all(not e["eligible"] for e in spec["eligibility_manifest"]))

    def test_duplicate_layout_cannot_inflate_independent_count(self):
        alias = deepcopy(self.pairs[0])
        alias["pair_id"] += "-duplicate"
        spec = self.build([self.pairs[0], alias])
        self.assertEqual(spec["expected_n_layouts"], 1)
        self.assertEqual(spec["n_aligned_pairs"], 1)
        self.assertEqual(spec["exclusion_counts"], {"duplicate_confirmation_layout_or_alias": 1})

    def test_deterministic_rebuild_from_raw_job_pairs_without_mutation(self):
        sources, config = deepcopy(self.pairs), self.configuration()
        source_copy, config_copy = deepcopy(sources), deepcopy(config)
        first = self.build(sources, configuration=config)
        raw_pairs = list({j["pair_id"]: j["metadata"]["source_pair"] for j in first["jobs"]}.values())
        second = self.build(raw_pairs, configuration=first["frozen_configuration"])
        self.assertEqual(first, second)
        self.assertEqual(sources, source_copy)
        self.assertEqual(config, config_copy)

    def test_supported_layer_alias_and_fixed_rule(self):
        config = self.configuration(layers=[10], selection={"scope": "explicit", "alignment": INPUT_ALIGNMENT_RULE})
        del config["layer"]
        spec = self.build(configuration=config)
        self.assertEqual(spec["frozen_configuration"]["layer"], 10)
        self.assertEqual(spec["protocol"]["input_alignment_rule"], INPUT_ALIGNMENT_RULE)

    def test_unsupported_freeze_changes_are_rejected_before_tokenization(self):
        changes = [
            {"layers": [0, 10]}, {"layers": [0]}, {"site": "cache"}, {"natural": False},
            {"semantic_headline_eligible": True}, {"claim_scope": "semantic_belief"},
            {"temperature": 0}, {"dose": 0.5}, {"n_layouts": 6}, {"max_new_tokens": 0},
            {"generation_seeds": [42, 42]}, {"primary_endpoint": "forced_target"},
            {"treatment_condition": "identity_layer10"}, {"control_condition": "counterfactual_text"},
            {"condition_inventory": ["baseline", "factspan_layer10"]},
            {"selection": {"scope": "explicit", "recipient_positions": [100]}},
            {"input_alignment_rule": "nearest_token"}, {"cap": 256},
        ]
        for change in changes:
            tokenizer = Tokenizer()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(tokenizer=tokenizer, configuration=self.configuration(**change))
            self.assertEqual(tokenizer.calls, [])

    def test_missing_freeze_and_unrestricted_confirmation_are_rejected(self):
        for kwargs in ({}, {"freeze_id": "f", "layers": (0, 10)},
                       {"freeze_id": "f", "layers": (10,), "natural": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_input_site_spec(self.pairs, Tokenizer(), "bad_confirm", partition="confirmation", **kwargs)

    def test_frozen_tokenizer_identity_is_enforced(self):
        frozen = self.build()["tokenizer"]
        self.assertEqual(self.build(configuration=self.configuration(tokenizer=frozen))["tokenizer"], frozen)
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            self.build(configuration=self.configuration(tokenizer={**frozen, "name_or_path": "different"}))

    def test_discovery_spec_is_identical_to_pre_adapter_fixture(self):
        spec = build_input_site_spec(self.discovery, Tokenizer(), "reproduction_fixture",
                                    layers=(0, 10), seeds=(42, 43), cap=4096)
        self.assertEqual(fingerprint(spec), "657280fbb545873b2901cb4fb5fa7e648ec4ebeef1f2ef8d52f6aba610780efb")
        self.assertNotIn("freeze_id", spec)
        self.assertTrue(all(j["partition"] == "discovery" for j in spec["jobs"]))

    def mixed_configuration(self, **updates):
        config = self.configuration(families=list(MIXED_INPUT_FAMILIES),
                                    family_layout_counts={"wall": 11, "key_possession": 11, "door_status": 10})
        config.update(updates)
        return config

    def test_mixed_quotas_weights_actual_counts_and_all_six_controls(self):
        config = self.mixed_configuration()
        spec = self.build(configuration=config)
        expected_counts = {family: 1 for family in MIXED_INPUT_FAMILIES}
        expected_pairs = {family: 2 for family in MIXED_INPUT_FAMILIES}
        expected_weights = {family: config["family_layout_counts"][family] / 32 for family in MIXED_INPUT_FAMILIES}
        self.assertEqual(spec["requested_n_layouts_by_family"], config["family_layout_counts"])
        self.assertEqual(spec["expected_n_layouts_by_family"], expected_counts)
        self.assertEqual(spec["expected_n_pairs_by_family"], expected_pairs)
        self.assertEqual(spec["family_weights"], expected_weights)
        self.assertEqual(spec["layout_shortfall_by_family"], {"wall": 10, "key_possession": 10, "door_status": 9})
        self.assertFalse(spec["cohort_complete"])
        self.assertEqual(spec["jobs"], self.build()["jobs"])
        self.assertEqual(len(spec["jobs"]), 3 * 6 * 2)
        for comparison in spec["comparisons"]:
            self.assertEqual(comparison["family_layout_counts"], config["family_layout_counts"])
            self.assertEqual(comparison["family_weights"], expected_weights)
            self.assertEqual(comparison["expected_n_layouts_by_family"], expected_counts)
            self.assertEqual(comparison["expected_n_pairs_by_family"], expected_pairs)
            self.assertEqual(comparison["resampling_scheme"], "within_family_layout_bootstrap")
            self.assertEqual(comparison["stratification_key"], "family")
        self.assertEqual(spec["protocol"]["condition_inventory"], self.build()["protocol"]["condition_inventory"])

    def test_mixed_missing_family_or_alignment_shortfall_never_renormalizes_weights(self):
        config = self.mixed_configuration()
        for pairs, tokenizer in (([self.pairs[0], self.pairs[2]], Tokenizer()), (self.pairs, CharacterTokenizer())):
            spec = self.build(pairs=pairs, tokenizer=tokenizer, configuration=config)
            self.assertEqual(spec["expected_n_layouts_by_family"], {"wall": 1, "key_possession": 0, "door_status": 1})
            self.assertEqual(spec["expected_n_pairs_by_family"], {"wall": 2, "key_possession": 0, "door_status": 2})
            self.assertEqual(spec["layout_shortfall_by_family"]["key_possession"], 11)
            self.assertEqual(spec["family_weights"]["key_possession"], 11 / 32)
            self.assertEqual(spec["expected_n_layouts"], 2)
            self.assertEqual(spec["expected_n_pairs"], 4)
            self.assertFalse(spec["cohort_complete"])
            self.assertEqual(len(spec["jobs"]), 2 * 6 * 2)
            self.assertTrue(all(comparison["expected_n_pairs_by_family"]["key_possession"] == 0 for comparison in spec["comparisons"]))

    def test_mixed_exact_balanced_32_layout_cohort_and_rebuild(self):
        pairs = generate_pairs(32, 20260916, partition="confirmation", families=MIXED_INPUT_FAMILIES)
        spec = self.build(pairs=pairs, configuration=self.mixed_configuration(generation_seeds=[42]))
        self.assertTrue(spec["cohort_complete"])
        self.assertEqual(spec["expected_n_layouts"], 32)
        self.assertEqual(spec["expected_n_layouts_by_family"], {"wall": 11, "key_possession": 11, "door_status": 10})
        self.assertEqual(spec["expected_n_pairs_by_family"], spec["expected_n_layouts_by_family"])
        self.assertEqual(spec["layout_shortfall_by_family"], {family: 0 for family in MIXED_INPUT_FAMILIES})
        self.assertEqual(len(spec["jobs"]), 32 * 6)
        raw_pairs = list({job["pair_id"]: job["metadata"]["source_pair"] for job in spec["jobs"]}.values())
        rebuilt = self.build(pairs=raw_pairs, configuration=spec["frozen_configuration"])
        self.assertEqual(spec, rebuilt)

    def test_mixed_overquota_is_rejected_even_with_overall_shortfall(self):
        pairs = generate_pairs(12, 20260917, partition="confirmation", families=("wall",))
        with self.assertRaisesRegex(ValueError, "exceeds frozen family quotas"):
            self.build(pairs=pairs, configuration=self.mixed_configuration())

    def test_mixed_invalid_family_quota_weight_and_resampling_config_rejected(self):
        changes = [
            {"family": "wall"}, {"family": None}, {"families": ["wall", "key_possession"]},
            {"families": ["wall", "wall", "door_status"]},
            {"family_layout_counts": {"wall": 11, "key_possession": 11}},
            {"family_layout_counts": {"wall": 12, "key_possession": 10, "door_status": 10}},
            {"family_layout_counts": {"wall": 11, "key_possession": 11, "door_status": 11}},
            {"family_layout_counts": {"wall": 0, "key_possession": 16, "door_status": 16}},
            {"family_layout_counts": {"wall": 11.0, "key_possession": 11, "door_status": 10}},
            {"family_weights": {"wall": 1 / 3, "key_possession": 1 / 3, "door_status": 1 / 3}},
            {"resampling_scheme": "pooled_layout_bootstrap"}, {"stratification_key": "pair_id"},
        ]
        for change in changes:
            tokenizer = Tokenizer()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(tokenizer=tokenizer, configuration=self.mixed_configuration(**change))
            self.assertFalse(tokenizer.calls)
        for extra in ({"family_layout_counts": {"wall": 32}}, {"family_weights": {"wall": 1}}):
            with self.assertRaisesRegex(ValueError, "explicit mixed families"):
                self.build(configuration=self.configuration(**extra))

    def test_mixed_noncanonical_order_rejected_valid_large_quotas_and_filtering(self):
        config = self.mixed_configuration(families=["door_status", "wall", "key_possession"])
        tokenizer = Tokenizer()
        with self.assertRaisesRegex(ValueError, "canonical families order"):
            self.build(configuration=config, tokenizer=tokenizer)
        self.assertFalse(tokenizer.calls)
        for count, quotas in ((64, {"wall": 22, "key_possession": 21, "door_status": 21}),
                              (128, {"wall": 43, "key_possession": 43, "door_status": 42})):
            spec = self.build(configuration=self.mixed_configuration(n_layouts=count, family_layout_counts=quotas))
            self.assertEqual(spec["requested_n_layouts_by_family"], quotas)
            self.assertEqual(spec["family_weights"], {family: quotas[family] / count for family in MIXED_INPUT_FAMILIES})
            self.assertFalse(spec["cohort_complete"])
        unknown = {**self.pairs[0], "family": "unapproved"}
        spec = self.build(pairs=[unknown, self.pairs[1]], configuration=self.mixed_configuration())
        self.assertEqual(spec["exclusions"][0]["input_index"], 0)
        self.assertEqual(spec["exclusions"][0]["reason"], "frozen_family_mismatch")
        self.assertEqual(spec["expected_n_layouts_by_family"], {"wall": 0, "key_possession": 1, "door_status": 0})

    def test_single_family_and_legacy_unrestricted_outputs_are_unchanged(self):
        single = self.build(configuration=self.configuration(family="wall"))
        self.assertEqual(fingerprint(single), "3db189fbac210117b560bd14c03ef3ae4250e85e371fabeafca93d1bae61d61f")
        self.assertEqual(fingerprint(self.build()), "139f27f4ede82f5033bb4fd0e36d94c1d1fb936fcc6349f82a1643abba338881")
        self.assertNotIn("family_weights", single)
        self.assertTrue(all("resampling_scheme" not in comparison for comparison in single["comparisons"]))


if __name__ == "__main__":
    unittest.main()
