"""Regression checks for pre-query confirmation cohort construction."""
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.activation_causality.protocols import _confirmation_pairs


class ConfirmationCohortTests(unittest.TestCase):
    @staticmethod
    def mixed_candidate():
        return {"protocol_kind": "input_fact", "families": ["wall", "key_possession", "door_status"],
                "family_layout_counts": {"wall": 11, "key_possession": 11, "door_status": 10},
                "n_layouts": 32, "dataset_seed": 100,
                "treatment_condition": "t", "control_condition": "c"}

    @staticmethod
    def planning_spec(pairs, *args, **kwargs):
        return {"comparisons": [{"treatment": "t", "control": "c",
                "eligible_pair_ids": [pair["pair_id"] for pair in pairs if pair.get("eligible", True)]}],
                "layout_aliases": {pair["layout_id"]: pair.get("aliases", []) for pair in pairs},
                "eligibility_manifest": [{"pair_id": pair["pair_id"],
                    "exclusion_reason": None if pair.get("eligible", True) else "unequal_full_token_lengths"}
                    for pair in pairs]}

    def test_controls_cannot_reintroduce_previously_used_or_sibling_layouts(self):
        def pair(name, aliases):
            return {"pair_id": name, "layout_id": name, "aliases": aliases}

        batches = [[pair("a", ["a", "old"]), pair("b", ["b", "control_b"])],
                   [pair("c", ["c", "control_b"])], [pair("d", ["d"])]]
        candidate = {"n_layouts": 2, "dataset_seed": 100, "family": "wall", "layers": [10],
                     "site": "resid_post", "last_k": 1, "generation_seeds": [42],
                     "max_new_tokens": 4096, "treatment_condition": "t", "control_condition": "c"}

        def build(pairs, *args, **kwargs):
            return {"comparisons": [{"treatment": "t", "control": "c", "eligible_pair_ids": [p["pair_id"] for p in pairs]}],
                    "layout_aliases": {p["layout_id"]: p["aliases"] for p in pairs},
                    "eligibility_manifest": [{"pair_id": p["pair_id"]} for p in pairs]}

        observed = []

        def generate(count, seed, **kwargs):
            observed.append((count, seed, set(kwargs["excluded_layout_ids"])))
            return batches[len(observed) - 1]

        with patch("src.activation_causality.protocols.excluded_layout_ids", return_value={"old"}), \
             patch("src.activation_causality.protocols.generate_pairs", side_effect=generate), \
             patch("src.activation_causality.protocols.build_controlled_state_spec", side_effect=build):
            selected, rejected = _confirmation_pairs(Path("campaign"), Path("repo"), candidate, "test")
        self.assertEqual([p["pair_id"] for p in selected], ["b", "d"])
        self.assertEqual([r["pair"]["pair_id"] for r in rejected], ["a", "c"])
        self.assertTrue(all(r["reason"] == "layout_or_control_alias_overlap" for r in rejected))
        self.assertEqual([(count, seed) for count, seed, _ in observed], [(2, 100), (1, 101), (1, 102)])
        self.assertIn("control_b", observed[1][2])

    def test_mixed_refills_own_quota_and_excludes_aliases_across_families(self):
        observed = []

        def generate(count, seed, **kwargs):
            family = kwargs["families"][0]
            observed.append((count, seed, family, set(kwargs["excluded_layout_ids"])))
            pairs = [{"pair_id": f"{family}_{seed}_{index}", "layout_id": f"{family}_{seed}_{index}",
                      "family": family} for index in range(count)]
            if seed == 100:
                pairs[0]["aliases"] = ["old"]
                pairs[1]["eligible"] = False
                pairs[2]["aliases"] = ["shared_wall_alias"]
            if seed == 1100:
                pairs[0]["aliases"] = ["shared_wall_alias"]
            return pairs

        with patch("src.activation_causality.protocols.excluded_layout_ids", return_value={"old"}), \
             patch("src.activation_causality.protocols.generate_pairs", side_effect=generate), \
             patch("src.activation_causality.protocols._build_candidate_spec", side_effect=self.planning_spec):
            pairs, rejected = _confirmation_pairs(Path("campaign"), Path("repo"), self.mixed_candidate(), "test")
        self.assertEqual(Counter(pair["family"] for pair in pairs),
                         {"wall": 11, "key_possession": 11, "door_status": 10})
        self.assertEqual([(count, seed, family) for count, seed, family, _ in observed],
                         [(11, 100, "wall"), (2, 101, "wall"), (11, 1100, "key_possession"),
                          (1, 1101, "key_possession"), (10, 2100, "door_status")])
        self.assertEqual([row["reason"] for row in rejected],
                         ["layout_or_control_alias_overlap", "unequal_full_token_lengths", "layout_or_control_alias_overlap"])
        self.assertIn("shared_wall_alias", observed[2][3])
        self.assertTrue(all(row["selection_basis"] == "simulator_and_exact_token_alignment_before_model_query"
                            for row in rejected))

    def test_mixed_invalid_quota_contract_rejected_before_generation(self):
        changes = [
            {"family_layout_counts": {"wall": 12, "key_possession": 10, "door_status": 10}},
            {"family_layout_counts": {"wall": 16, "key_possession": 16}},
            {"family": "wall"}, {"families": ["wall", "door_status", "key_possession"]},
            {"family_weights": {"wall": 1/3, "key_possession": 1/3, "door_status": 1/3}},
            {"protocol_kind": "controlled_state"}, {"dataset_seed": True},
        ]
        for change in changes:
            with self.subTest(change=change):
                candidate = deepcopy(self.mixed_candidate())
                candidate.update(change)
                with patch("src.activation_causality.protocols.generate_pairs") as generate, \
                     self.assertRaises(ValueError):
                    _confirmation_pairs(Path("campaign"), Path("repo"), candidate, "test")
                generate.assert_not_called()

    def test_mixed_cannot_substitute_another_family(self):
        wrong = [{"pair_id": "wrong", "layout_id": "wrong", "family": "key_possession"}]
        with patch("src.activation_causality.protocols.excluded_layout_ids", return_value=set()), \
             patch("src.activation_causality.protocols.generate_pairs", return_value=wrong), \
             patch("src.activation_causality.protocols._build_candidate_spec") as build, \
             self.assertRaisesRegex(ValueError, "requested confirmation family"):
            _confirmation_pairs(Path("campaign"), Path("repo"), self.mixed_candidate(), "test")
        build.assert_not_called()

    def test_mixed_unfilled_family_stops_before_other_family_sampling(self):
        def generate(count, seed, **kwargs):
            return [{"pair_id": f"bad_{seed}", "layout_id": f"bad_{seed}", "family": "wall", "eligible": False}]

        with patch("src.activation_causality.protocols.excluded_layout_ids", return_value=set()), \
             patch("src.activation_causality.protocols.generate_pairs", side_effect=generate) as generate_mock, \
             patch("src.activation_causality.protocols._build_candidate_spec", side_effect=self.planning_spec), \
             self.assertRaisesRegex(ValueError, "wall quota"):
            _confirmation_pairs(Path("campaign"), Path("repo"), self.mixed_candidate(), "test")
        self.assertEqual(generate_mock.call_count, 100)
        self.assertTrue(all(call.kwargs["families"] == ("wall",) for call in generate_mock.call_args_list))


if __name__ == "__main__":
    unittest.main()
