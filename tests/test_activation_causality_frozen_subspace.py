"""CPU-only frozen-subspace construction, provenance and template checks."""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from src.activation_causality.environments import generate_pairs
from src.activation_causality.frozen_subspace import (
    ALIGNMENT, CLAIM_SCOPE, CONDITIONS, ENDPOINT, PRIMARY_IDS, UNRESOLVED_POLICY,
    _expand, build_frozen_subspace_spec,
)
from src.activation_causality.input_sites import build_input_site_spec
from src.activation_causality.primary_family import validate_primary_family
from src.activation_causality.registry import fingerprint
from src.activation_causality.subspace_protocol import build_subspace_spec


class Tokenizer:
    name_or_path = "cpu_subspace_fixture"

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if add_special_tokens:
            raise AssertionError("No additional tokenizer framing")
        return [300 if value == "False" else 301 if value == "True" else ord(value)
                for value in re.findall(r"False|True|[\s\S]", text)]


def basis(family="wall", rank=1):
    return {"format": "registered_task_basis_v1", "model_provenance_version": 2,
            "partition": "discovery", "training_only": True, "family": family, "rank": rank,
            "layer": 0, "site": "resid_post", "source_condition": "factspan_layer0", "hidden_dim": 2880,
            "basis_id": f"existing_{family}_r{rank}", "basis_path": "/immutable/basis.npy",
            "basis_file_sha256": "a" * 64, "basis_metadata_path": "/immutable/basis.json",
            "basis_metadata_sha256": "b" * 64,
            "training_layout_ids": [f"training_layout_{i}" for i in range(4)],
            "training_pair_ids": [f"training_pair_{i}" for i in range(4)],
            "training_layout_aliases": [f"training_layout_{i}" for i in range(4)] + ["training_extra_alias"],
            "source_spec_sha256": "c" * 64, "request_fingerprint": "d" * 64,
            "model_snapshot_identity_sha256": "e" * 64, "model_snapshot_audit_sha256": "f" * 64,
            "selected_source_definitions_sha256": "0" * 64, "basis_array_sha256": "1" * 64,
            "activation_representation": {"caller_validated_test_fixture": True},
            "source_job_fingerprints": {f"{i:024x}": "2" * 64 for i in range(4)}}


def configuration(metadata, n=32, dose=0.5, tokenizer=None):
    tokenizer = tokenizer or Tokenizer()
    label = "0p5" if dose == 0.5 else "1"
    common = {"treatment": "subspace_forward_" + label, "predicted_direction": "increase",
              "primary_endpoint": ENDPOINT, "target_action_basis": "unique_simulator_counterfactual_optimum",
              "native_unresolved_policy": UNRESOLVED_POLICY, "expected_n_layouts": n, "expected_n_pairs": n,
              "eligibility_rule": {"identifier": "frozen_canonical_subspace_v1", "sha256": "3" * 64},
              "origin": {"kind": "existing_source_comparison", "provenance": {"source_experiment_id": "native_source"}},
              "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False}
    return {"candidate_id": "selected_subspace", "family": metadata["family"], "rank": metadata["rank"],
            "layer": 0, "site": "resid_post", "selected_dose": dose, "doses": [0.5, 1.0],
            "generation_seeds": [42], "random_basis_seeds": [42], "max_new_tokens": 4096,
            "temperature": 0.7, "n_layouts": n, "condition_inventory": list(CONDITIONS),
            "tokenizer": {"class": type(tokenizer).__name__, "name_or_path": tokenizer.name_or_path,
                          "add_special_tokens": False, "context_renderer": "campaign.render_analysis_prompt"},
            "basis_path": metadata["basis_path"], "basis_sha256": metadata["basis_file_sha256"],
            "basis_metadata_path": metadata["basis_metadata_path"], "basis_metadata_sha256": metadata["basis_metadata_sha256"],
            "basis_metadata_fingerprint": fingerprint(metadata), "source_experiment_id": "native_source",
            "source_spec_sha256": "4" * 64,
            "primary_comparisons": {PRIMARY_IDS[0]: {**deepcopy(common), "control": "baseline"},
                                    PRIMARY_IDS[1]: {**deepcopy(common), "control": "projected_norm_matched_random_subspace_" + label}}}


class FrozenSubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(6, 20260919, "confirmation", families=("wall", "key_possession", "door_status"))
        cls.wall = [pair for pair in cls.pairs if pair["family"] == "wall"]
        cls.discovery = generate_pairs(3, 20260920, "discovery", families=("wall", "key_possession", "door_status"))

    def build(self, pairs=None, metadata=None, config=None, tokenizer=None):
        metadata, tokenizer = metadata or basis(), tokenizer or Tokenizer()
        return build_frozen_subspace_spec(self.wall if pairs is None else pairs, tokenizer, "frozen_subspace_test",
                                          config or configuration(metadata, tokenizer=tokenizer), "freeze_test", basis_metadata=metadata)

    def test_partial_counts_complete_controls_and_two_exact_primaries(self):
        result = self.build()
        self.assertEqual(len(result["jobs"]), 28)
        self.assertEqual(result["expected_n_layouts"], 2)
        self.assertEqual(result["expected_n_pairs"], 2)
        self.assertEqual(result["layout_shortfall"], 30)
        self.assertFalse(result["cohort_complete"])
        self.assertEqual(result["protocol"]["input_alignment_rule"], ALIGNMENT)
        selected = [c for c in result["comparisons"] if c["confirmatory"]]
        self.assertEqual({c["primary_id"] for c in selected}, set(PRIMARY_IDS))
        self.assertTrue(all(c["treatment"] == "subspace_forward_0p5" for c in selected))
        for c in result["comparisons"]:
            self.assertEqual(c["expected_n_layouts"], 2)
            self.assertEqual(c["expected_n_pairs"], 2)
            if not c["confirmatory"]:
                self.assertNotIn("primary_id", c)
        for binding in result["frozen_configuration"]["primary_comparisons"].values():
            self.assertEqual(binding["expected_n_layouts"], 32)
        for pair in self.wall:
            jobs = [j for j in result["jobs"] if j["pair_id"] == pair["pair_id"]]
            self.assertEqual([j["condition"] for j in jobs], list(CONDITIONS))
            for job in jobs:
                self.assertEqual(job["metadata"]["source_pair"], pair)
                self.assertEqual(job["metadata"]["pair"]["partition"], "confirmation")
                self.assertEqual(job["target_action"], pair["counterfactual_optimal_actions"][0])
                self.assertEqual(job["job_id"], fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24])
        json.dumps(result, allow_nan=False)

    def test_inline_lineage_has_no_phantom_registered_source(self):
        result = self.build()
        inline = result["inline_input_source"]
        self.assertFalse(inline["registered"])
        self.assertEqual(inline["definition_sha256"], fingerprint(inline["definition"]))
        sources = {j["job_id"]: j for j in inline["definition"]["jobs"]}
        for job in result["jobs"]:
            meta = job["metadata"]
            self.assertNotIn("source_experiment_id", meta)
            self.assertNotIn("source_job_id", meta)
            self.assertEqual(meta["inline_input_source_definition_sha256"], inline["definition_sha256"])
            source = sources[meta["inline_input_source_job_id"]]
            self.assertEqual(meta["inline_input_source_job_sha256"], fingerprint(source))
            self.assertEqual(source["condition"], "factspan_layer0")

    def test_reverse_removal_identity_positions_and_fixed_target(self):
        jobs = {j["condition"]: j for j in self.build([self.wall[0]])["jobs"]}
        base, donor = jobs["baseline"], jobs["donor_text"]
        for condition in ("subspace_reverse_0p5", "subspace_reverse_1", "subspace_removal"):
            job = jobs[condition]
            self.assertEqual(job["recipient_token_ids"], donor["recipient_token_ids"])
            self.assertEqual(job["donor_token_ids"], base["recipient_token_ids"])
            self.assertEqual(job["metadata"]["paired_baseline_condition"], "donor_text")
            self.assertEqual(job["selection"]["recipient_positions"], job["recipient_positions"])
            self.assertEqual(job["metadata"]["donor_patch_positions"], job["donor_positions"])
            self.assertEqual(job["target_action"], base["target_action"])
        self.assertTrue(jobs["subspace_removal"]["intervention"]["zero_donor"])
        for condition in ("subspace_identity", "restoration_equivalent"):
            job = jobs[condition]
            self.assertEqual(job["recipient_token_ids"], job["donor_token_ids"])
            self.assertEqual(job["recipient_positions"], job["donor_positions"])
            self.assertFalse(job["metadata"]["independent_rescue_evidence"])

    def test_distinct_random_controls_and_same_fixed_norm_reference_policy(self):
        jobs = {j["condition"]: j for j in self.build([self.wall[0]])["jobs"]}
        for label, dose in (("0p5", 0.5), ("1", 1.0)):
            forward = jobs["subspace_forward_" + label]["intervention"]
            rank = jobs["random_subspace_" + label]["intervention"]
            matched = jobs["projected_norm_matched_random_subspace_" + label]["intervention"]
            self.assertTrue(rank["random_basis"])
            self.assertNotIn("projected_norm_matched_random_basis", rank)
            self.assertTrue(matched["projected_norm_matched_random_basis"])
            self.assertNotIn("random_basis", matched)
            self.assertEqual(rank["random_seed"], matched["random_seed"])
            self.assertEqual(matched["random_seed"], 42)
            for intervention in (forward, matched):
                self.assertTrue(intervention["audit_projected_norm"])
                self.assertEqual(intervention["projected_norm_relative_tolerance"], 0.01)
                self.assertEqual(intervention["dose"], dose)

    def test_both_selected_doses_and_ranks_keep_all_fourteen_jobs(self):
        for rank in (1, 4):
            metadata = basis(rank=rank)
            for dose in (0.5, 1.0):
                result = self.build([self.wall[0]], metadata, configuration(metadata, dose=dose))
                self.assertEqual(len(result["jobs"]), 14)
                self.assertEqual({j["metadata"]["rank"] for j in result["jobs"]}, {rank})
                label = "0p5" if dose == 0.5 else "1"
                self.assertTrue(all(c["treatment"] == "subspace_forward_" + label
                                    for c in result["comparisons"] if c["confirmatory"]))

    def test_exact_discovery_template_parity_without_relabeled_confirmation(self):
        metadata = basis()
        pairs = [p for p in self.discovery if p["family"] == "wall"]
        source = build_input_site_spec(pairs, Tokenizer(), "discovery_input", layers=(0,), natural=True, cap=4096)
        original = build_subspace_spec(source, metadata, metadata["basis_path"], metadata["basis_file_sha256"], "template_parity")
        intended = [job for job in source["jobs"] if job["condition"] == "factspan_layer0"]
        expanded = [job for row in intended for job in _expand(row, metadata, "template_parity", fingerprint(source))]
        self.assertEqual(len(original["jobs"]), len(expanded))
        fields = ("condition", "partition", "family", "pair_id", "layout_id", "seed", "target_action", "original_action",
                  "recipient_text", "donor_text", "recipient_token_ids", "donor_token_ids", "selection", "recipient_positions",
                  "donor_positions", "intervention", "natural", "temperature", "max_new_tokens")
        meta_fields = ("paired_baseline_condition", "random_control_matching", "projected_norm_audit", "random_orientation_policy",
                       "random_subspace_note", "independent_rescue_evidence", "engineering_identity_equivalence",
                       "restoration_equivalence_only", "removal_note", "restoration_note")
        for old, new in zip(original["jobs"], expanded):
            self.assertEqual({key: old[key] for key in fields}, {key: new[key] for key in fields})
            self.assertEqual({key: old["metadata"][key] for key in meta_fields}, {key: new["metadata"][key] for key in meta_fields})

    def test_partition_observed_and_family_exclusions_are_prequery_and_counted(self):
        observed = deepcopy(self.wall[0])
        observed["metadata"]["model_outputs_observed"] = True
        relabeled = deepcopy(next(p for p in self.discovery if p["family"] == "wall"))
        relabeled["partition"] = "confirmation"
        other = next(p for p in self.pairs if p["family"] != "wall")
        tokenizer = Tokenizer()
        result = self.build([observed, relabeled, other], tokenizer=tokenizer)
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(result["expected_n_layouts"], 0)
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["n_excluded_pairs"], 3)
        self.assertEqual([e["input_index"] for e in result["exclusions"]], [0, 1, 2])

    def test_training_topology_and_declared_aliases_excluded_before_tokenization(self):
        for use_alias in (False, True):
            pair = deepcopy(self.wall[0])
            metadata = basis()
            if use_alias:
                pair["metadata"]["variant_layout_ids"].append("training_extra_alias")
            else:
                metadata["training_layout_ids"][0] = pair["layout_id"]
                metadata["training_layout_aliases"].append(pair["layout_id"])
            tokenizer = Tokenizer()
            result = self.build([pair], metadata, tokenizer=tokenizer)
            self.assertEqual(tokenizer.calls, [])
            self.assertEqual(result["exclusion_counts"], {"basis_training_layout_or_alias_overlap": 1})

    def test_duplicate_layout_does_not_inflate_counts(self):
        duplicate = deepcopy(self.wall[0])
        duplicate["pair_id"] += "_repeat"
        result = self.build([self.wall[0], duplicate])
        self.assertEqual(result["expected_n_layouts"], 1)
        self.assertEqual(result["exclusion_counts"], {"duplicate_confirmation_layout_or_alias": 1})

    def test_extra_raw_source_alias_overlap_cannot_inflate_layouts(self):
        second = deepcopy(self.wall[1])
        second["layout_aliases"] = [self.wall[0]["layout_id"]]
        result = self.build([self.wall[0], second])
        self.assertEqual(result["expected_n_layouts"], 1)
        self.assertEqual(result["exclusion_counts"], {"duplicate_complete_source_layout_alias": 1})
        self.assertEqual(result["exclusions"][0]["input_index"], 1)
        self.assertFalse(result["eligibility_manifest"][1]["eligible"])

    def test_multiple_changed_tokens_allowed_but_unequal_length_excluded(self):
        class TwoToken(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                result = []
                for value in re.findall(r"False|True|[\s\S]", text):
                    result.extend([400, 401] if value == "False" else [402, 403] if value == "True" else [ord(value)])
                return result
        pair = next(p for p in self.pairs if p["family"] == "key_possession")
        metadata = basis("key_possession")
        result = self.build([pair], metadata, tokenizer=TwoToken())
        self.assertEqual(result["expected_n_layouts"], 1)
        self.assertTrue(all(len(j["recipient_positions"]) == 2 for j in result["jobs"]))
        class Unequal(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        excluded = self.build([pair], metadata, tokenizer=Unequal())
        self.assertEqual(excluded["expected_n_layouts"], 0)
        self.assertEqual(excluded["exclusion_counts"], {"unequal_full_token_lengths": 1})

    def test_missing_and_contradictory_configuration_rejected_before_encode(self):
        metadata = basis()
        mutations = [{"site": "attn_out"}, {"layer": 1}, {"rank": 2}, {"selected_dose": 0.25},
                     {"doses": [1.0]}, {"generation_seeds": [43]}, {"random_basis_seeds": [43]},
                     {"temperature": 0.0}, {"max_new_tokens": 8192}, {"n_layouts": 6},
                     {"condition_inventory": list(CONDITIONS[:-1])}, {"renderer_id": "alternate_w_dot_l_yes_no_v1"},
                     {"families": ["wall", "key_possession", "door_status"]}, {"layers": [1]},
                     {"basis_sha256": "9" * 64}, {"basis_path": "relative.npy"}, {"family_size": 2},
                     {"selection": {"scope": "explicit", "recipient_positions": [100]}},
                     {"input_alignment_rule": "nearest"}]
        for mutation in mutations:
            tokenizer, config = Tokenizer(), configuration(metadata)
            config.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.build(metadata=metadata, config=config, tokenizer=tokenizer)
            self.assertEqual(tokenizer.calls, [])
        config = configuration(metadata)
        del config["basis_metadata_fingerprint"]
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            self.build(metadata=metadata, config=config)

    def test_binding_mutations_and_basis_metadata_mutation_rejected(self):
        metadata = basis()
        for mutate in (lambda c: c["primary_comparisons"].pop(PRIMARY_IDS[1]),
                       lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(control="donor_text"),
                       lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(expected_n_layouts=64),
                       lambda c: c["primary_comparisons"][PRIMARY_IDS[0]].update(primary_endpoint="forced")):
            config = configuration(metadata)
            mutate(config)
            with self.assertRaises(ValueError):
                self.build(metadata=metadata, config=config)
        config = configuration(metadata)
        metadata["activation_representation"]["changed"] = True
        with self.assertRaisesRegex(ValueError, "semantic fingerprint"):
            self.build(metadata=metadata, config=config)

    def test_pure_deterministic_no_files_and_no_input_mutation(self):
        metadata, tokenizer = basis(), Tokenizer()
        config = configuration(metadata, tokenizer=tokenizer)
        saved = deepcopy((self.wall, metadata, config))
        with patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), patch.object(Path, "open", side_effect=AssertionError("file open")):
            first = self.build(metadata=metadata, config=config, tokenizer=tokenizer)
        rebuilt = self.build(metadata=metadata, config=first["frozen_configuration"])
        self.assertEqual(first, rebuilt)
        self.assertEqual((self.wall, metadata, config), saved)
        registry = {"schema_version": 1, "candidates": {config["candidate_id"]: list(PRIMARY_IDS)},
                    "family_size": 2, "bindings": {config["candidate_id"]: config["primary_comparisons"]}}
        self.assertEqual(validate_primary_family(registry), registry)

    def test_full_quota_and_overquota_not_fabricated(self):
        pairs = generate_pairs(33, 771990, "confirmation", families=("wall",))
        full = self.build(pairs[:32])
        self.assertTrue(full["cohort_complete"])
        self.assertEqual(len(full["jobs"]), 448)
        self.assertEqual(full["layout_shortfall"], 0)
        for comparison in full["comparisons"]:
            if comparison["confirmatory"]:
                binding = full["frozen_configuration"]["primary_comparisons"][comparison["primary_id"]]
                self.assertTrue(all(comparison[key] == value for key, value in binding.items()))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.build(pairs)


if __name__ == "__main__":
    unittest.main()
