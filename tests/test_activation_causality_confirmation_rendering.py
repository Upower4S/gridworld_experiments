"""Fixed primary/alternate construction and parent linkage; CPU only."""
from collections import Counter, OrderedDict
from copy import deepcopy
import unittest
from unittest.mock import patch

from src.activation_causality import frozen_contextual_input as contextual
from src.activation_causality import frozen_subspace as subspace
from src.activation_causality.confirmation_rendering import build_primary, build_lexical_supplement, eligibility_decisions
from src.activation_causality.environments import generate_pairs
from src.activation_causality.lexical_generalization import ALTERNATE, CANONICAL, renderer_manifest
from src.activation_causality.registry import fingerprint
from test_activation_causality_frozen_contextual_input import Tokenizer, configuration as contextual_config
from test_activation_causality_frozen_subspace import basis, configuration as subspace_config


def sub_configuration(metadata=None, n=32, tokenizer=None, joint=True):
    metadata, tokenizer = metadata or basis(), tokenizer or Tokenizer()
    result = subspace_config(metadata, n=n, tokenizer=tokenizer)
    result.update(protocol_kind=subspace.PROTOCOL, renderer_id=CANONICAL, renderer_binding=renderer_manifest(CANONICAL),
                  require_joint_renderer_eligibility=joint)
    return result


def context_configuration(n=32, tokenizer=None, joint=True):
    result = contextual_config(n=n, tokenizer=tokenizer or Tokenizer(), joint=joint)
    result["protocol_kind"] = contextual.PROTOCOL
    return result


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mixed = []
        for index, family in enumerate(contextual.FAMILIES):
            cls.mixed.extend(generate_pairs(contextual.QUOTAS[32][family], 20261001 + index, "confirmation", families=(family,)))
        cls.wall = generate_pairs(32, 20261220, "confirmation", families=("wall",))
        cls.key = generate_pairs(2, 20261221, "confirmation", families=("key_possession",))

    def primary(self, pairs=None, config=None, tokenizer=None, metadata=None, eid="primary_fixture"):
        tokenizer = tokenizer or Tokenizer()
        config = config or context_configuration(tokenizer=tokenizer)
        return build_primary(self.mixed if pairs is None else pairs, tokenizer, eid, config, "freeze_fixture", basis_metadata=metadata)

    def test_context_primary_parity_full_origins_and_sampler_decisions(self):
        config = context_configuration()
        parent = self.primary(config=config)
        fixed = contextual.build_frozen_contextual_input_spec(self.mixed, Tokenizer(), "primary_fixture", config, "freeze_fixture")
        self.assertEqual(parent["jobs"], fixed["jobs"])
        self.assertEqual(parent["comparisons"], fixed["comparisons"])
        self.assertTrue(parent["cohort_complete"])
        self.assertEqual(parent["cohort_role"], "primary")
        self.assertEqual(len(parent["jobs"]), 480)
        self.assertEqual(build_primary(self.mixed, Tokenizer(), "primary_fixture", config, "freeze_fixture"), parent)
        decisions = eligibility_decisions(parent)
        self.assertEqual(len(decisions), 32)
        for pair, decision in zip(self.mixed, decisions, strict=True):
            self.assertEqual(decision["source_pair_sha256"], fingerprint(pair))
            self.assertTrue(decision["eligible"])
            self.assertTrue(decision["evidence"]["joint_renderer_audit"]["eligible"])
            self.assertEqual(decision["evidence"]["required_condition_inventory"], config["condition_inventory"])

    def test_subspace_partial_primary_keeps_basis_geometry_and_joint_proof(self):
        metadata = basis()
        config = sub_configuration(metadata)
        parent = self.primary(self.wall[:2], config, metadata=metadata)
        self.assertEqual(len(parent["jobs"]), 28)
        self.assertFalse(parent["cohort_complete"])
        self.assertEqual(parent["expected_n_layouts"], 2)
        self.assertEqual(parent["frozen_configuration"]["primary_comparisons"][subspace.PRIMARY_IDS[0]]["expected_n_layouts"], 32)
        for entry in parent["eligibility_manifest"]:
            self.assertTrue(entry["joint_renderer_audit"]["eligible"])
            self.assertTrue(entry["eligible"])
        for job in parent["jobs"]:
            self.assertEqual(job["metadata"]["renderer_id"], CANONICAL)
            self.assertIn(job["metadata"]["source_pair"], self.wall[:2])
            if job["intervention"]["operation"] == "subspace":
                self.assertEqual(job["intervention"]["basis_sha256"], metadata["basis_file_sha256"])

    def test_subspace_joint_failure_excludes_all_controls_and_rebuilds_original_manifest(self):
        class WiderAlternate(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                original = super().encode(text, add_special_tokens)
                return [part for value in original for part in ([value, value + 10] if value in {302, 303} else [value])]
        tokenizer, metadata = WiderAlternate(), basis("key_possession")
        config = sub_configuration(metadata, tokenizer=tokenizer)
        parent = self.primary(self.key, config, tokenizer, metadata)
        self.assertFalse(parent["jobs"])
        self.assertEqual(parent["n_excluded_pairs"], 2)
        self.assertEqual(parent["wrapper_source_pairs"], self.key)
        self.assertTrue(all(entry["reason"] == "joint_renderer_alignment_failed" for entry in parent["eligibility_manifest"]))
        self.assertTrue(all(entry["joint_renderer_audit"]["exclusions"] for entry in parent["eligibility_manifest"]))
        self.assertEqual(self.primary(self.key, config, tokenizer, metadata), parent)
        standard = self.primary(self.key, sub_configuration(metadata, tokenizer=tokenizer, joint=False), tokenizer, metadata)
        self.assertEqual(len(standard["jobs"]), 28)

    def test_rejected_inputs_and_aliases_preserved(self):
        pairs = deepcopy(self.mixed[:2])
        pairs[1]["metadata"]["source_pair"] = {"layout_id": "extra_rejected_alias"}
        pairs[1]["metadata"]["model_outputs_observed"] = True
        parent = self.primary(pairs)
        self.assertEqual(parent["n_source_pairs"], 2)
        self.assertEqual(parent["n_excluded_pairs"], 1)
        decisions = eligibility_decisions(parent)
        self.assertFalse(decisions[1]["eligible"])
        self.assertIn("extra_rejected_alias", decisions[1]["layout_aliases"])
        self.assertEqual(decisions[1]["source_pair_sha256"], fingerprint(pairs[1]))
        self.assertEqual(parent["wrapper_source_pairs_sha256"], fingerprint(pairs))

    def test_context_supplement_exact32_all_descriptive_parent_unchanged(self):
        config = context_configuration()
        parent = self.primary(config=config)
        before = fingerprint(parent)
        alt = build_lexical_supplement(parent, Tokenizer(), "alternate_fixture", config, "freeze_fixture")
        self.assertEqual(fingerprint(parent), before)
        self.assertEqual(alt["parent_spec_fingerprint"], before)
        self.assertEqual(alt["cohort_role"], "lexical_supplement")
        self.assertEqual(alt["renderer_id"], ALTERNATE)
        self.assertEqual(len(alt["jobs"]), 480)
        self.assertEqual(alt["family_layout_counts"], {"wall": 11, "key_possession": 11, "door_status": 10})
        self.assertEqual(alt["family_weights"], alt["standard_subset_reference"]["family_weights"])
        self.assertEqual(alt["standard_subset_reference"]["execution_jobs_added"], 0)
        self.assertEqual(len(alt["standard_subset_reference"]["job_fingerprints"]), 480)
        self.assertNotIn("primary_comparisons", alt["supplement_configuration"])
        for comparison in alt["comparisons"] + alt["standard_subset_reference"]["comparisons"]:
            self.assertFalse(comparison["confirmatory"])
            self.assertNotIn("primary_id", comparison)
            self.assertEqual(comparison["expected_n_layouts"], 32)
            self.assertEqual(comparison["family_weights"], alt["family_weights"])
        originals = {job["job_id"]: job for job in parent["jobs"]}
        for job in alt["jobs"]:
            self.assertFalse(job["confirmatory"])
            standard = originals[job["metadata"]["paired_standard_job_id"]]
            self.assertEqual(fingerprint(standard), job["metadata"]["paired_standard_job_sha256"])
            self.assertEqual(job["metadata"]["source_pair"], standard["metadata"]["source_pair"])
            self.assertEqual(job["intervention"]["operation"], standard["intervention"]["operation"])
            self.assertEqual(job["target_action"], standard["target_action"])
            self.assertNotEqual(job["recipient_token_ids"], standard["recipient_token_ids"])
            self.assertEqual(len(job["selection"]["recipient_positions"]), len(standard["selection"]["recipient_positions"]))

    def test_subspace_alternate_preserves_all14_basis_doses_reverse_and_identity(self):
        metadata, config = basis(rank=4), sub_configuration(basis(rank=4))
        parent = self.primary(self.wall, config, metadata=metadata)
        self.assertTrue(parent["cohort_complete"])
        alt = build_lexical_supplement(parent, Tokenizer(), "alternate_subspace", config, "freeze_fixture", basis_metadata=metadata)
        self.assertEqual(len(alt["jobs"]), 448)
        self.assertEqual(Counter(job["condition"] for job in alt["jobs"]), Counter(dict.fromkeys(subspace.CONDITIONS, 32)))
        by_parent = {job["job_id"]: job for job in parent["jobs"]}
        for job in alt["jobs"]:
            before = by_parent[job["metadata"]["paired_standard_job_id"]]
            self.assertEqual(job["intervention"], before["intervention"])
            self.assertEqual(job["metadata"]["pair"], before["metadata"]["pair"])
            self.assertEqual(job["metadata"]["source_pair"], before["metadata"]["source_pair"])
            self.assertEqual(job["metadata"]["renderer_id"], ALTERNATE)
            self.assertNotIn("inline_input_source_job_id", job["metadata"])
            self.assertEqual(job["job_id"], fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24])
        first = alt["jobs"][0]["pair_id"]
        rows = {j["condition"]: j for j in alt["jobs"] if j["pair_id"] == first}
        self.assertEqual(rows["subspace_reverse_1"]["recipient_token_ids"], rows["donor_text"]["recipient_token_ids"])
        self.assertEqual(rows["subspace_reverse_1"]["donor_token_ids"], rows["baseline"]["recipient_token_ids"])
        self.assertEqual(rows["subspace_identity"]["recipient_token_ids"], rows["subspace_identity"]["donor_token_ids"])
        self.assertEqual(rows["subspace_removal"]["recipient_token_ids"], rows["donor_text"]["recipient_token_ids"])

    def test_64_layout_primary_has_distinct_fixed32_subset_weights(self):
        pairs = []
        for index, family in enumerate(contextual.FAMILIES):
            pairs.extend(generate_pairs(contextual.QUOTAS[64][family], 20261238 + index, "confirmation", families=(family,)))
        config = context_configuration(n=64)
        parent = self.primary(pairs, config)
        self.assertTrue(parent["cohort_complete"])
        before = fingerprint(parent)
        alt = build_lexical_supplement(parent, Tokenizer(), "alternate_subset32", config, "freeze_fixture")
        self.assertEqual(fingerprint(parent), before)
        expected = {"wall": 11 / 32, "key_possession": 11 / 32, "door_status": 10 / 32}
        self.assertNotEqual(parent["frozen_configuration"]["family_weights"], expected)
        self.assertEqual(alt["family_weights"], expected)
        self.assertEqual(alt["standard_subset_reference"]["family_weights"], expected)
        self.assertEqual(len(alt["jobs"]), 32 * 15)
        self.assertEqual(len(alt["standard_subset_reference"]["job_fingerprints"]), 32 * 15)
        for comparison in alt["comparisons"] + alt["standard_subset_reference"]["comparisons"]:
            self.assertEqual(comparison["family_weights"], expected)
            self.assertEqual(comparison["expected_n_layouts_by_family"], contextual.QUOTAS[32])
            self.assertEqual(comparison["expected_n_pairs_by_family"], contextual.QUOTAS[32])
            self.assertEqual(comparison["expected_n_pairs"], 32)
        reference_ids = {row["pair_id"] for row in alt["standard_subset_reference"]["job_bindings"]}
        self.assertEqual(reference_ids, {row["pair_id"] for row in alt["jobs"]})
        self.assertTrue(reference_ids < {row["pair_id"] for row in parent["jobs"]})

    def test_parent_tamper_configuration_freeze_and_partial_rejected(self):
        config = context_configuration()
        parent = self.primary(config=config)
        for mutate in (lambda p: p["jobs"][0]["recipient_token_ids"].__setitem__(0, 999),
                       lambda p: p["comparisons"][0].update(expected_n_pairs=31),
                       lambda p: p["wrapper_source_pairs"][0]["metadata"].update(extra="changed"),
                       lambda p: p.update(freeze_id="different"),
                       lambda p: p.update(cohort_role="lexical_supplement")):
            changed = deepcopy(parent)
            mutate(changed)
            with self.assertRaises(ValueError):
                build_lexical_supplement(changed, Tokenizer(), "alternate_fixture", config, "freeze_fixture")
        with self.assertRaises(ValueError):
            build_lexical_supplement(self.primary(self.mixed[:2]), Tokenizer(), "alternate_fixture", config, "freeze_fixture")
        standard_config = context_configuration(joint=False)
        with self.assertRaises(ValueError):
            build_lexical_supplement(self.primary(config=standard_config), Tokenizer(), "alternate_fixture", standard_config, "freeze_fixture")

    def test_config_renderer_and_source_hash_exactness(self):
        config = context_configuration()
        config["renderer_binding"]["custom_renderers_allowed"] = 0
        with self.assertRaises(ValueError):
            self.primary(self.mixed[:1], config)
        parent = self.primary(self.mixed[:1])
        parent["wrapper_source_pairs_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            eligibility_decisions(parent)

    def test_alternate_context_compiler_fullwrite_bindings_and_same_orientation(self):
        from test_activation_causality_full_write_runner import FullWriteRunnerTests
        from src.activation_causality.campaign import compile_patch
        fixture = FullWriteRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        proof = fixture.proof()
        parent = self.primary()
        alt = build_lexical_supplement(parent, Tokenizer(), "alternate_compile", context_configuration(), "freeze_fixture")
        pid = alt["jobs"][0]["pair_id"]
        hashes = []
        for spec in (parent, alt):
            for condition in ("layer0_random", "layer0_natural_norm_random"):
                job = next(j for j in spec["jobs"] if j["pair_id"] == pid and j["condition"] == condition)
                _, patches = compile_patch(fixture.engine, job, OrderedDict(), loaded_model_proof=proof)
                hashes.append(patches[0]["random_orientation_audit"]["raw_gaussian_float32_sha256"])
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(fixture.engine.endpoints, [])

    def test_no_file_access_or_basis_refit_during_build(self):
        with patch("builtins.open", side_effect=AssertionError("no IO")):
            parent = self.primary(self.wall[:2], sub_configuration(), metadata=basis())
        self.assertEqual(len(parent["jobs"]), 28)


if __name__ == "__main__":
    unittest.main()
