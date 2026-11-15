"""Model-free checks of complete paired cohorts and native figure denominators."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/35_activation_causality_native_input_figure.py"
MODULE_SPEC = importlib.util.spec_from_file_location("native_input_figure", SCRIPT)
figure = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(figure)


class NativeFigureGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment_id = "synthetic_native_input"
        self.spec_path = self.root / "specs" / f"{self.experiment_id}.json"
        self.results = self.root / "results" / self.experiment_id
        self.results.mkdir(parents=True)
        self.spec_path.parent.mkdir()
        self.conditions = ["baseline", "counterfactual_text"] + [
            f"{kind}_layer0" for kind in figure.KINDS]
        self.spec = {"experiment_id": self.experiment_id, "partition": "discovery",
                     "claim_scope": figure.SCOPE, "jobs": [], "exclusions": [],
                     "n_source_pairs": 3, "n_aligned_pairs": 3, "n_excluded_pairs": 0}
        self.rows = {}
        for family in figure.FAMILIES:
            for seed in (42, 43):
                for condition in self.conditions:
                    job_id = f"{family}_{seed}_{condition}"
                    pair = {"pair_id": f"pair_{family}", "layout_id": f"layout_{family}",
                            "family": family, "partition": "discovery",
                            "recipient_optimal_actions": ["DOWN"],
                            "counterfactual_optimal_actions": ["RIGHT"],
                            "metadata": {"variant_layout_ids": [f"alias_{family}"]}}
                    job = {"experiment_id": self.experiment_id, "job_id": job_id,
                           "pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
                           "family": family, "condition": condition, "partition": "discovery",
                           "target_action": "RIGHT", "original_action": "DOWN", "seed": seed,
                           "natural": True, "semantic_test": False, "max_new_tokens": 4,
                           "temperature": 0.7, "metadata": {"pair": pair, "claim_scope": figure.SCOPE,
                                                              "representation_kind": "pre_reasoning_analysis_header_token" if condition == "lasttoken_layer0" else "input_fact_tokens",
                                                              "abstract_belief_representation_claim": False},
                           "intervention": {"layers": [] if condition in self.conditions[:2] else [0],
                                            "site": "resid_post", "operation": "replace", "dose": 1.0,
                                            "random_control": condition == "random_factspan_layer0"},
                           "recipient_token_ids": [101, 102], "donor_token_ids": [101, 103],
                           "recipient_positions": [1], "donor_positions": [1],
                           "selection": {"scope": "explicit", "recipient_positions": [1], "donor_positions": [1]}}
                    row = deepcopy(job)
                    action = "RIGHT" if condition in {"counterfactual_text", "factspan_layer0"} else "DOWN"
                    row.update({"status": "complete", "generation_status": "complete",
                                "native_action": action, "parse_status": "valid", "endpoint": "native_final",
                                "final_text": json.dumps({"action": action}), "generated_tokens": 2,
                                "generated_token_ids": [401, 402] if action == "DOWN" else [401, 403],
                                "forced_q": {"UP": 0.1, "DOWN": 0.7, "LEFT": 0.1, "RIGHT": 0.1},
                                "patch_log": [{"layer": 0, "delta_l2": [0]}] if condition == "identity_layer0" else [],
                                "engine_key": "engine", "execution_key": "execution",
                                "inference_configuration": {"head_mode": "full"}})
                    if family == "wall" and seed == 42 and condition == "factspan_layer0":
                        row.update({"generation_status": "truncated", "native_action": None,
                                    "generated_tokens": 4, "generated_token_ids": [401, 403, 404, 405]})
                    self.spec["jobs"].append(job)
                    self.rows[job_id] = row
        self.save()

    def save(self):
        encoded = json.dumps(self.spec, sort_keys=True).encode()
        self.spec_path.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        for job_id, row in self.rows.items():
            row["spec_sha256"] = digest
            (self.results / f"{job_id}.json").write_text(json.dumps(row))

    def load(self):
        return figure.load_counts(self.root, self.experiment_id)

    def mutate_job(self, job_id, key, value):
        next(j for j in self.spec["jobs"] if j["job_id"] == job_id)[key] = deepcopy(value)
        self.rows[job_id][key] = deepcopy(value)
        self.save()

    def test_trial_denominators_distinguish_layouts_and_native_unresolved(self):
        counts = self.load()
        self.assertEqual(counts["n_completed_jobs"], 36)
        self.assertEqual(counts["n_layouts"], 3)
        self.assertEqual(counts["n_paired_trials"], 6)
        self.assertEqual(counts["identity_audit"]["n_identity_jobs"], 6)
        self.assertEqual(counts["seeds"], [42, 43])
        self.assertEqual(counts["layouts_per_family"], dict.fromkeys(figure.FAMILIES, 1))
        self.assertEqual(len(counts["source_rows"]), 36)
        self.assertEqual(len(counts["raw_rows_manifest_sha256"]), 64)
        self.assertEqual(counts["native_status_counts"], {"complete": 35, "truncated": 1, "malformed": 0})
        for panel in counts["panels"]:
            for condition in panel["conditions"]:
                self.assertEqual(sum(condition[f"{kind}_count"] for kind in ("target", "other", "unresolved")), condition["n"])
                self.assertAlmostEqual(sum(condition[f"{kind}_rate"] for kind in ("target", "other", "unresolved")), 1)
        fact = next(r for r in counts["panels"][0]["conditions"] if r["condition"] == "factspan_layer0")
        self.assertEqual((fact["target_count"], fact["other_count"], fact["unresolved_count"]), (5, 0, 1))
        self.assertEqual(fact["n"], 6)
        # A parseable target fragment at truncation and forced q are never a native fallback.
        self.assertEqual(fact["native_status_counts"]["truncated"], 1)
        self.assertFalse(counts["semantic_headline_eligible"])

    def test_incomplete_inventory_refused_before_any_raw_read(self):
        job_ids = list(self.rows)
        (self.results / f"{job_ids[0]}.json").unlink()
        (self.results / f"{job_ids[1]}.json").write_text("corrupt unfinished JSON")
        with patch.object(figure, "read_hashed", wraps=figure.read_hashed) as reader:
            with self.assertRaisesRegex(ValueError, "1 missing"):
                self.load()
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(reader.call_args.args[0], self.spec_path.resolve())

    def test_extra_results_and_retirement_refused(self):
        extra = self.results / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "1 extra"):
            self.load()
        extra.unlink()
        (self.root / "reviews").mkdir()
        (self.root / "reviews" / f"{self.experiment_id}_retired.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "retired"):
            self.load()

    def test_wrong_scope_and_forced_only_refused(self):
        self.spec["claim_scope"] = "abstract_belief"
        self.save()
        with self.assertRaisesRegex(ValueError, "scope"):
            self.load()
        self.spec["claim_scope"] = figure.SCOPE
        self.mutate_job(next(iter(self.rows)), "natural", False)
        with self.assertRaisesRegex(ValueError, "native input-fact"):
            self.load()

    def test_stale_spec_or_result_override_refused(self):
        job_id = next(iter(self.rows))
        self.rows[job_id]["recipient_token_ids"] = [999]
        self.save()
        with self.assertRaisesRegex(ValueError, "frozen recipient_token_ids"):
            self.load()
        self.rows[job_id]["recipient_token_ids"] = [101, 102]
        self.save()
        row = self.rows[job_id]
        row["spec_sha256"] = "stale"
        (self.results / f"{job_id}.json").write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.load()

    def test_mixed_execution_configuration_refused(self):
        self.rows[next(iter(self.rows))]["inference_configuration"] = {"head_mode": "last_token"}
        self.save()
        with self.assertRaisesRegex(ValueError, "Mixed engine"):
            self.load()

    def test_missing_paired_control_even_with_complete_inventory_refused(self):
        job_id = "wall_42_lasttoken_layer0"
        self.spec["jobs"] = [j for j in self.spec["jobs"] if j["job_id"] != job_id]
        del self.rows[job_id]
        (self.results / f"{job_id}.json").unlink()
        self.save()
        with self.assertRaisesRegex(ValueError, "every paired trial"):
            self.load()

    def test_identity_sequence_distribution_and_norm_guard(self):
        job_id = "wall_42_identity_layer0"
        original = deepcopy(self.rows[job_id])
        for key, value, message in (
            ("generated_token_ids", [501, 502], "sequence"),
            ("forced_q", dict.fromkeys(figure.ACTIONS, 0.25), "distribution"),
            ("patch_log", [{"layer": 0, "delta_l2": [0.01]}], "zero"),
        ):
            with self.subTest(field=key):
                self.rows[job_id] = deepcopy(original)
                self.rows[job_id][key] = value
                self.save()
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_native_final_and_cap_guards(self):
        cases = [("wall_42_baseline", "final_text", '{"action":"RIGHT"}', "mismatch"),
                 ("wall_42_factspan_layer0", "native_action", "RIGHT", "Unresolved"),
                 ("wall_42_factspan_layer0", "generated_tokens", 3, "token audit"),
                 ("wall_42_baseline", "status", "technical_error", "Incomplete")]
        for job_id, key, value, message in cases:
            with self.subTest(field=key):
                original = deepcopy(self.rows[job_id])
                self.rows[job_id][key] = value
                self.save()
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
                self.rows[job_id] = original

    def test_layout_alias_collision_refused(self):
        self.mutate_job("key_possession_42_baseline", "layout_aliases", ["alias_wall"])
        with self.assertRaisesRegex(ValueError, "share a recorded alias"):
            self.load()

    def test_physical_target_and_partition_mismatch_refused(self):
        job_id = "wall_42_baseline"
        metadata = deepcopy(self.rows[job_id]["metadata"])
        metadata["pair"]["counterfactual_optimal_actions"] = ["UP"]
        self.mutate_job(job_id, "metadata", metadata)
        with self.assertRaisesRegex(ValueError, "physical pair/target"):
            self.load()

    def test_frozen_family_denominators_and_underfilled_cohort_guard(self):
        self.spec["expected_n_layouts_by_family"] = dict.fromkeys(figure.FAMILIES, 1)
        self.spec["expected_n_pairs_by_family"] = dict.fromkeys(figure.FAMILIES, 2)
        self.save()
        self.load()
        self.spec["expected_n_layouts_by_family"]["wall"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "family denominators"):
            self.load()
        self.spec["cohort_complete"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "underfilled"):
            self.load()

    def test_single_family_frozen_confirmation_has_explicit_empty_panels(self):
        self.spec["partition"] = "confirmation"
        self.spec["freeze_id"] = "fixed-before-confirmation"
        self.spec["jobs"] = [j for j in self.spec["jobs"] if j["family"] == "wall"]
        for job_id in list(self.rows):
            if self.rows[job_id]["family"] != "wall":
                del self.rows[job_id]
                (self.results / f"{job_id}.json").unlink()
        for job in self.spec["jobs"]:
            job["partition"] = "confirmation"
            job["metadata"]["pair"]["partition"] = "confirmation"
            self.rows[job["job_id"]]["partition"] = "confirmation"
            self.rows[job["job_id"]]["metadata"] = deepcopy(job["metadata"])
        self.save()
        counts = self.load()
        self.assertEqual(counts["partition"], "confirmation")
        self.assertEqual(counts["n_layouts"], 1)
        self.assertFalse(counts["semantic_headline_eligible"])
        for panel in counts["panels"][2:]:
            self.assertEqual(panel["n_layouts"], 0)
            self.assertTrue(all(row["target_rate"] is None and row["n"] == 0 for row in panel["conditions"]))
        del self.spec["freeze_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "freeze"):
            self.load()


if __name__ == "__main__":
    unittest.main()
