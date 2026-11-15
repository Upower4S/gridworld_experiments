"""CPU guards for the staged complete contextual profile and fixed renderer."""
from collections import Counter, OrderedDict
from copy import deepcopy
import json
import re
import unittest
from unittest.mock import patch

from src.activation_causality.contextual_input import build_contextual_input_spec
from src.activation_causality.environments import generate_pairs
from src.activation_causality.frozen_contextual_input import (
    ALTERNATE, CANONICAL, CLAIM_SCOPE, ENDPOINT, FAMILIES, PRIMARY_IDS, QUOTAS, UNRESOLVED_POLICY,
    _materialize_pair, _tokenizer_metadata, build_frozen_contextual_input_spec, condition_inventory,
)
from src.activation_causality.lexical_generalization import renderer_manifest
from src.activation_causality.primary_family import validate_primary_family
from src.activation_causality.registry import fingerprint


class Tokenizer:
    name_or_path = "cpu_contextual_fixture"

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("extra framing")
        return [300 if value == "False" else 301 if value == "True" else 302 if value == "No" else
                303 if value == "Yes" else ord(value)
                for value in re.findall(r"False|True|Yes|No|[\s\S]", text)]


def configuration(n=32, layer=0, tokenizer=None, joint=False):
    tokenizer = tokenizer or Tokenizer()
    quota = QUOTAS[n]
    common = {"treatment": f"layer{layer}_natural", "predicted_direction": "increase", "primary_endpoint": ENDPOINT,
              "target_action_basis": "unique_simulator_counterfactual_optimum", "native_unresolved_policy": UNRESOLVED_POLICY,
              "expected_n_layouts": n, "expected_n_pairs": n,
              "eligibility_rule": {"identifier": "complete_contextual_15_v1", "sha256": "3" * 64},
              "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False, "families": list(FAMILIES),
              "family_layout_counts": deepcopy(quota), "family_weights": {f: quota[f] / n for f in FAMILIES},
              "expected_n_layouts_by_family": deepcopy(quota), "expected_n_pairs_by_family": deepcopy(quota),
              "resampling_scheme": "within_family_layout_bootstrap", "stratification_key": "family"}
    return {"candidate_id": "selected_contextual", "source_experiment_id": "contextual_discovery",
            "source_spec_sha256": "4" * 64, "layer": layer, "site": "resid_post", "families": list(FAMILIES),
            "family_layout_counts": deepcopy(quota), "family_weights": {f: quota[f] / n for f in FAMILIES},
            "n_layouts": n, "generation_seeds": [42], "random_seed": 42, "temperature": 0.7,
            "max_new_tokens": 4096, "condition_inventory": condition_inventory(layer), "tokenizer": _tokenizer_metadata(tokenizer),
            "renderer_id": CANONICAL, "renderer_binding": renderer_manifest(CANONICAL),
            "require_joint_renderer_eligibility": joint,
            "primary_comparisons": {pid: {**deepcopy(common), "control": f"layer{layer}_{control}",
                  "origin": {"kind": origin, "provenance": {"source_experiment_id": "contextual_discovery",
                                                           "source_spec_sha256": "4" * 64}}}
                for pid, control, origin in ((PRIMARY_IDS[0], "same_fact", "existing_source_comparison"),
                    (PRIMARY_IDS[1], "natural_norm_random", "prospective_confirmation_control"))}}


class FrozenContextualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(6, 20260919, "confirmation", families=FAMILIES)
        cls.discovery = generate_pairs(3, 20260920, "discovery", families=FAMILIES)

    def build(self, pairs=None, config=None, tokenizer=None):
        tokenizer = tokenizer or Tokenizer()
        return build_frozen_contextual_input_spec(self.pairs if pairs is None else pairs, tokenizer,
            "frozen_contextual_test", config or configuration(tokenizer=tokenizer), "freeze_test")

    def materialize(self, pair=None, renderer=CANONICAL, joint=False, layer=0):
        pair = pair or self.discovery[0]
        return _materialize_pair(pair, Tokenizer(), "frozen_contextual_test", layer=layer, renderer_id=renderer,
            require_joint_renderer_eligibility=joint, candidate_source_experiment_id="contextual_discovery",
            candidate_source_spec_sha256="4" * 64, freeze_id="freeze_test" if pair["partition"] == "confirmation" else None)

    def test_partial_counts_and_full_inventory(self):
        result = self.build()
        self.assertEqual(result["exclusions"], [])
        self.assertEqual(len(result["jobs"]), 90)
        self.assertEqual(result["expected_n_layouts_by_family"], dict.fromkeys(FAMILIES, 2))
        self.assertEqual(result["expected_n_pairs"], 6)
        self.assertEqual(result["layout_shortfall_by_family"], {"wall": 9, "key_possession": 9, "door_status": 8})
        self.assertFalse(result["cohort_complete"])
        for pair in self.pairs:
            rows = [row for row in result["jobs"] if row["pair_id"] == pair["pair_id"]]
            self.assertEqual([row["condition"] for row in rows], condition_inventory(0))
            self.assertEqual({row["target_action"] for row in rows}, set(pair["counterfactual_optimal_actions"]))
            self.assertTrue(all(row["metadata"]["source_pair"] == pair for row in rows))
        json.dumps(result, allow_nan=False)

    def test_both_primaries_and_partial_binding_mismatch(self):
        result = self.build()
        primary = [row for row in result["comparisons"] if row["confirmatory"]]
        self.assertEqual({row["primary_id"] for row in primary}, set(PRIMARY_IDS))
        self.assertEqual(len(result["comparisons"]), 10)
        for row in result["comparisons"]:
            self.assertEqual(row["expected_n_layouts"], 6)
            self.assertEqual(row["expected_n_layouts_by_family"], dict.fromkeys(FAMILIES, 2))
            self.assertEqual(row["family_layout_counts"], QUOTAS[32])
            self.assertEqual(len(row["semantic_controls"]), 5)
            self.assertEqual(row["required_condition_inventory"], condition_inventory(0))
            if not row["confirmatory"]:
                self.assertNotIn("primary_id", row)
        bindings = result["frozen_configuration"]["primary_comparisons"]
        validate_primary_family({"schema_version": 1, "candidates": {"selected_contextual": list(PRIMARY_IDS)},
            "family_size": 2, "bindings": {"selected_contextual": bindings}})
        for row in primary:
            self.assertEqual(bindings[row["primary_id"]]["expected_n_layouts"], 32)
            self.assertNotEqual(row["expected_n_layouts"], bindings[row["primary_id"]]["expected_n_layouts"])
        self.assertEqual(primary[0]["origin"]["kind"], "existing_source_comparison")
        self.assertEqual(primary[1]["origin"]["kind"], "prospective_confirmation_control")

    def test_random_geometry_and_strict_source_identity(self):
        result = self.build()
        for pair in self.pairs:
            rows = {j["condition"]: j for j in result["jobs"] if j["pair_id"] == pair["pair_id"]}
            natural, new, old = (rows["layer0_" + name] for name in ("natural", "natural_norm_random", "random"))
            self.assertEqual(new["donor_token_ids"], natural["donor_token_ids"])
            self.assertEqual(new["selection"], natural["selection"])
            self.assertEqual(old["recipient_token_ids"], new["recipient_token_ids"])
            self.assertEqual(old["recipient_positions"], new["recipient_positions"])
            self.assertEqual(len(old["donor_positions"]), len(new["donor_positions"]))
            self.assertEqual(old["intervention"]["random_seed"], new["intervention"]["random_seed"])
            for job in (new, old):
                self.assertTrue(job["intervention"]["record_random_orientation"])
                self.assertTrue(job["intervention"]["random_control"])
            self.assertNotIn("audit_full_write_norm", old["intervention"])
            for job in (natural, new):
                self.assertTrue(job["intervention"]["audit_full_write_norm"])
                self.assertEqual(job["intervention"]["full_write_relative_tolerance"], 0.01)
            for job, role in ((natural, "natural"), (new, "natural"), (old, "matched")):
                identity = job["intervention"]["full_write_source_identity"]
                self.assertEqual(identity["lineage_kind"], "inline_physical_source_pair")
                self.assertEqual(identity["source_pair_id"], pair["pair_id"])
                self.assertEqual(identity["source_pair_sha256"], fingerprint(pair))
                self.assertEqual(identity["donor_role"], role)
                for key in ("recipient", "donor"):
                    self.assertEqual(identity[key + "_token_sha256"], fingerprint(job[key + "_token_ids"]))
                self.assertEqual(identity["selection_sha256"], fingerprint(job["selection"]))
                self.assertEqual(identity["renderer_id"], job["metadata"]["renderer_id"])
                self.assertNotIn("source_job_id", identity)

    def test_reverse_identity_references_remain_correct(self):
        rows = {j["condition"]: j for j in self.materialize()["jobs"]}
        for role, expected_r, expected_d, reference in (("reverse", "counterfactual_text", "baseline", "counterfactual_text"),
             ("identity", "baseline", "baseline", "baseline"),
             ("restoration_equivalent", "counterfactual_text", "counterfactual_text", "counterfactual_text")):
            job = rows["layer0_" + role]
            self.assertEqual(job["recipient_token_ids"], rows[expected_r]["recipient_token_ids"])
            self.assertEqual(job["donor_token_ids"], rows[expected_d]["recipient_token_ids"])
            self.assertEqual(job["metadata"]["paired_baseline_condition"], reference)
            self.assertFalse(job["metadata"]["independent_rescue_evidence"])

    def test_legacy_fourteen_job_mechanics_exact_for_both_layers(self):
        keys = ("recipient_text", "donor_text", "recipient_token_ids", "donor_token_ids", "selection",
                "recipient_positions", "donor_positions", "target_action", "original_action", "seed", "temperature",
                "natural", "max_new_tokens", "family", "partition", "pair_id", "layout_id")
        audit_keys = {"full_write_source_identity", "record_random_orientation", "audit_full_write_norm", "full_write_relative_tolerance"}
        for layer in (0, 10):
            legacy = build_contextual_input_spec(self.discovery, Tokenizer(), "legacy_test", layers=(layer,))
            for pair in self.discovery:
                new = {j["condition"]: j for j in self.materialize(pair, layer=layer)["jobs"]}
                old = [j for j in legacy["jobs"] if j["pair_id"] == pair["pair_id"]]
                self.assertEqual(len(old), 14)
                for before in old:
                    after = new[before["condition"]]
                    self.assertEqual({key: after[key] for key in keys}, {key: before[key] for key in keys})
                    self.assertEqual({k: v for k, v in after["intervention"].items() if k not in audit_keys}, before["intervention"])
                    for key in before["metadata"]:
                        self.assertEqual(after["metadata"][key], before["metadata"][key])

    def test_inline_source_hash_and_canonical_job_ids(self):
        result = self.build()
        inline = {entry["pair_id"]: entry for entry in result["inline_controlled_sources"]}
        for definition in inline.values():
            self.assertFalse(definition["registered"])
            self.assertEqual(definition["definition_sha256"], fingerprint(definition["definition"]))
        for job in result["jobs"]:
            self.assertEqual(job["job_id"], fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24])
            meta = job["metadata"]
            source = next(row for row in inline[job["pair_id"]]["definition"]["jobs"] if row["job_id"] == meta["inline_controlled_source_job_id"])
            self.assertEqual(fingerprint(source), meta["inline_controlled_source_job_sha256"])
            for obj in (job, meta):
                self.assertNotIn("source_experiment_id", obj)
                self.assertNotIn("source_job_id", obj)
            self.assertIsNone(meta["observed_donor_action"])
            self.assertIsNone(meta["observed_donor_action_mismatch"])
        self.assertEqual(result, self.build())

    def test_unavailable_control_excludes_whole_pair(self):
        from src.activation_causality.experiments import state_control_specs
        def unavailable(pair):
            controls = state_control_specs(pair)
            controls[1].update(eligible=False, ineligible_reason="fixture_missing_control")
            return controls
        with patch("src.activation_causality.experiments.state_control_specs", side_effect=unavailable):
            result = self.build(self.pairs[:1])
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["expected_n_pairs"], 0)
        self.assertIn("complete_control_source_unavailable", result["exclusions"][0]["reason"])
        self.assertFalse(result["cohort_complete"])

    def test_duplicate_aliases_preserve_raw_nested_aliases(self):
        pairs = deepcopy(self.pairs[:2])
        pairs[0]["layout_aliases"] = ["shared_extra"]
        pairs[1]["metadata"]["source_pair"] = {"layout_id": "shared_extra"}
        result = self.build(pairs)
        self.assertEqual(result["expected_n_layouts"], 1)
        self.assertEqual(result["exclusions"][0]["reason"], "duplicate_complete_source_layout_alias")
        self.assertTrue(all("shared_extra" in job["layout_aliases"] for job in result["jobs"]))
        self.assertTrue(all(job["metadata"]["source_pair"] == pairs[0] for job in result["jobs"]))
        duplicate = self.build([self.pairs[0], self.pairs[0]])
        self.assertEqual(duplicate["exclusions"][0]["reason"], "duplicate_pair_id")

    def test_physical_mutations_and_observed_pairs_excluded(self):
        mutations = (lambda p: p.update(partition="discovery"),
                     lambda p: p["metadata"].update(model_outputs_observed=True),
                     lambda p: p.update(layout_id="relabeled"),
                     lambda p: p.update(pair_id="renamed"),
                     lambda p: p.update(recipient_prompt_text="override"),
                     lambda p: p.update(donor_prompt_text="override"),
                     lambda p: p.update(donor_optimal_actions=["invalid"]),
                     lambda p: p.update(family="outside"))
        for mutate in mutations:
            pair = deepcopy(self.pairs[0])
            mutate(pair)
            result = self.build([pair])
            self.assertEqual(result["jobs"], [])
            self.assertEqual(result["n_excluded_pairs"], 1)
            self.assertTrue(result["exclusions"][0]["excluded_before_model_query"])

    def test_configuration_contradictions_rejected(self):
        mutations = (lambda c: c.pop("renderer_binding"), lambda c: c.update(layer=15),
                     lambda c: c.update(families=list(reversed(FAMILIES))), lambda c: c.update(family="wall"),
                     lambda c: c.update(family_layout_counts=dict.fromkeys(FAMILIES, 11)),
                     lambda c: c.update(family_weights=dict.fromkeys(FAMILIES, 1 / 3)),
                     lambda c: c.update(generation_seeds=[43]), lambda c: c.update(random_seed=41),
                     lambda c: c.update(max_new_tokens=2048), lambda c: c.update(temperature=0),
                     lambda c: c.update(condition_inventory=condition_inventory(0)[:-1]),
                     lambda c: c.update(prompt_override="new"), lambda c: c.update(confirmatory=False),
                     lambda c: c.update(full_write_relative_tolerance=0.02),
                     lambda c: c.update(require_joint_renderer_eligibility=1),
                     lambda c: c["primary_comparisons"][PRIMARY_IDS[1]]["origin"].update(kind="existing_source_comparison"),
                     lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(control="baseline"))
        for mutate in mutations:
            config = configuration()
            mutate(config)
            with self.assertRaises(ValueError):
                self.build(config=config)

    def test_only_fixed_renderer_public_canonical_private_alternate(self):
        config = configuration()
        config.update(renderer_id=ALTERNATE, renderer_binding=renderer_manifest(ALTERNATE))
        with self.assertRaisesRegex(ValueError, "canonical-only"):
            self.build(config=config)
        with self.assertRaises(ValueError):
            self.materialize(renderer="arbitrary")
        for pair in self.discovery:
            result = self.materialize(pair, renderer=ALTERNATE, joint=True)
            self.assertTrue(result["eligible"], result)
            self.assertEqual(len(result["jobs"]), 15)
            self.assertTrue(result["joint_renderer_audit"]["eligible"])
            self.assertEqual(set(result["source_states"]), {"recipient", "counterfactual", "donor", "natural_donor", "same_fact_donor", "irrelevant_fact_donor"})
            for job in result["jobs"]:
                self.assertEqual(job["metadata"]["source_pair"], pair)
                self.assertEqual(job["metadata"]["renderer_id"], ALTERNATE)
                self.assertNotIn("primary_id", job)
                self.assertNotIn("confirmatory", job)

    def test_nested_binding_json_types_are_exact(self):
        mutations = (lambda c: c["renderer_binding"].update(custom_renderers_allowed=0),
                     lambda c: c["tokenizer"].update(add_special_tokens=0),
                     lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(semantic_headline_eligible=0),
                     lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(expected_n_layouts=32.0),
                     lambda c: c["primary_comparisons"][PRIMARY_IDS[0]]["family_layout_counts"].update(wall=11.0),
                     lambda c: c.update(dose=1),
                     lambda c: c.update(cap=4096.0))
        for mutate in mutations:
            config = configuration()
            mutate(config)
            with self.assertRaises(ValueError):
                self.build(config=config)

    def test_joint_eligibility_checks_alternate_before_any_primary_query(self):
        class BadAlternate(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                values = super().encode(text, add_special_tokens)
                return values + [999] if "Carrying key: Yes" in text else values
        tokenizer = BadAlternate()
        key = next(pair for pair in self.pairs if pair["family"] == "key_possession")
        canonical = self.build([key], configuration(tokenizer=tokenizer, joint=False), tokenizer)
        self.assertEqual(canonical["expected_n_layouts"], 1)
        joined = self.build([key], configuration(tokenizer=tokenizer, joint=True), tokenizer)
        self.assertEqual(joined["jobs"], [])
        self.assertEqual(joined["exclusions"][0]["reason"], "complete_control_alignment_failed")
        self.assertFalse(joined["exclusions"][0]["control_eligibility"]["joint_renderer_audit"]["eligible"])

    def test_empty_missing_family_and_overquota(self):
        result = self.build([])
        self.assertEqual(result["expected_n_layouts_by_family"], dict.fromkeys(FAMILIES, 0))
        self.assertEqual(result["layout_shortfall_by_family"], QUOTAS[32])
        subset = self.build([p for p in self.pairs if p["family"] == "wall"])
        self.assertEqual(subset["expected_n_layouts_by_family"], {"wall": 2, "key_possession": 0, "door_status": 0})
        pairs = generate_pairs(12, 20260930, "confirmation", families=("wall",))
        with self.assertRaisesRegex(ValueError, "exceeds frozen quota"):
            self.build(pairs)

    def test_joint_equal_lengths_but_different_support_width_excluded(self):
        class WiderAlternate(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                values = super().encode(text, add_special_tokens)
                return [part for value in values for part in
                        ([value, value + 10] if value in {302, 303} else [value])]
        tokenizer = WiderAlternate()
        key = next(pair for pair in self.pairs if pair["family"] == "key_possession")
        result = self.build([key], configuration(tokenizer=tokenizer, joint=True), tokenizer)
        self.assertFalse(result["jobs"])
        audit = result["exclusions"][0]["control_eligibility"]["joint_renderer_audit"]
        self.assertTrue(any(row["reason"] == "standard_alternate_support_width_mismatch" for row in audit["exclusions"]))
        for renderer in (CANONICAL, ALTERNATE):
            self.assertTrue(all(row["eligible"] for row in audit["renderings"][renderer].values()))

    def test_emitted_full_write_contracts_accept_and_detect_token_tampering(self):
        from src.activation_causality.campaign import _full_write_contract
        result = self.build(self.pairs[:1])
        flagged = []
        for job in result["jobs"]:
            if _full_write_contract(job):
                flagged.append(job)
        self.assertEqual([job["condition"] for job in flagged], ["layer0_natural", "layer0_random", "layer0_natural_norm_random"])
        for job in flagged:
            bad = deepcopy(job)
            bad["donor_token_ids"][bad["donor_positions"][0]] += 1
            with self.assertRaisesRegex(ValueError, "token fingerprint mismatch"):
                _full_write_contract(bad)

    def test_adapter_compiler_same_draw_and_natural_reference_with_fake_engine(self):
        import numpy as np
        from test_activation_causality_full_write_runner import FullWriteRunnerTests
        from src.activation_causality.campaign import compile_patch
        from src.activation_causality.full_write_audit import validate_full_write_norm_audit
        fixture = FullWriteRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        proof = fixture.proof()  # Only the fixture's tiny local fake engine.
        def contextual_values(ids, positions):
            context = int(fingerprint(ids)[:8], 16) % 137
            return np.array([[context * (h + 1) / 8 + ids[p] * (h + 2) / 19
                              for h in range(8)] for p in positions], dtype=np.float32)
        fixture.engine.values = contextual_values
        for renderer in (CANONICAL, ALTERNATE):
            jobs = {job["condition"]: job for job in self.materialize(renderer=renderer, joint=True)["jobs"]}
            compiled = {}
            for condition in ("natural", "random", "natural_norm_random"):
                ids, patches = compile_patch(fixture.engine, jobs["layer0_" + condition], OrderedDict(), loaded_model_proof=proof)
                self.assertEqual(len(patches), 1)
                compiled[condition] = patches[0]
                if condition != "random":
                    state = fixture.engine.prefill(ids, patches)
                    self.assertTrue(validate_full_write_norm_audit(patches[0]["full_write_norm_audit"], state.patch_log)["passed"])
            old, new = compiled["random"], compiled["natural_norm_random"]
            self.assertEqual(old["random_orientation_audit"]["raw_gaussian_float32_sha256"],
                             new["random_orientation_audit"]["raw_gaussian_float32_sha256"])
            for key in ("reference_delta_l2", "expected_delta_l2", "expected_recipient_float32_sha256"):
                self.assertEqual(compiled["natural"]["full_write_norm_audit"][key], new["full_write_norm_audit"][key])
            self.assertFalse(np.array_equal(old["donor"], new["donor"]))
        self.assertEqual(fixture.engine.endpoints, [])

    def test_full_quota_counts_all_two_bindings_exact(self):
        pairs = []
        for index, family in enumerate(FAMILIES):
            pairs.extend(generate_pairs(QUOTAS[32][family], 20261001 + index, "confirmation", families=(family,)))
        result = self.build(pairs)
        self.assertEqual(result["exclusions"], [])
        self.assertTrue(result["cohort_complete"])
        self.assertEqual(len(result["jobs"]), 480)
        self.assertEqual(Counter(job["condition"] for job in result["jobs"]), Counter(dict.fromkeys(condition_inventory(0), 32)))
        for row in result["comparisons"]:
            if row["confirmatory"]:
                binding = result["frozen_configuration"]["primary_comparisons"][row["primary_id"]]
                self.assertEqual({key: row[key] for key in binding}, binding)


if __name__ == "__main__":
    unittest.main()
