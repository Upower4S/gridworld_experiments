"""Synthetic complete batteries only: no actual discovery files or models."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.activation_causality import discovery_selection as select
from src.activation_causality.registry import fingerprint


def refresh(source):
    source["source_result_hashes"] = {
        f"results/{source['experiment_id']}/{r['job_id']}.json": fingerprint(r)
        for r in source["rows"]
    }
    source["source_result_manifest_sha256"] = fingerprint(
        source["source_result_hashes"]
    )
    return source


def make_source(eid):
    context = eid == select.CONTEXTUAL_SOURCE
    family = (
        None
        if context
        else next(
            f
            for f, short in select.FAMILY_NAMES.items()
            if eid.startswith(f"subspace_{short}_")
        )
    )
    rank = None if context else int(eid.split("_r")[1].split("_")[0])
    conditions = (
        set(select.REFERENCES)
        | {
            f"layer{layer}_{role}"
            for layer in (0, 10)
            for role in select.CONTEXTUAL_ROLES
        }
        if context
        else set(select.SUBSPACE_CONDITIONS)
    )
    populations = (
        [(f, i) for f in select.FAMILIES for i in range(2)]
        if context
        else [(family, i) for i in range(4)]
    )
    jobs = []
    for fam, i in populations:
        for condition in sorted(conditions):
            dose = 0.5 if condition.endswith("0p5") else 1.0
            layer = 10 if condition.startswith("layer10_") else 0
            intervention = {
                "layers": [layer],
                "site": "resid_post",
                "operation": "replace" if context else "subspace",
                "dose": dose,
            }
            if not context:
                intervention.update(
                    audit_projected_norm=True,
                    projected_norm_relative_tolerance=0.01,
                    basis_sha256=fingerprint([family, rank]),
                    basis_metadata_sha256=fingerprint([family, rank, "metadata"]),
                )
            job = {
                "experiment_id": eid,
                "partition": "discovery",
                "layout_id": f"{fam}_layout{i}",
                "pair_id": f"{fam}_pair{i}",
                "family": fam,
                "condition": condition,
                "target_action": "RIGHT",
                "original_action": "UP",
                "natural": True,
                "seed": 42,
                "max_new_tokens": 4096,
                "temperature": 0.7,
                "selection": {
                    "scope": "explicit",
                    "recipient_positions": [1],
                    "donor_positions": [1],
                },
                "intervention": intervention,
            }
            job["job_id"] = fingerprint(job)[:24]
            jobs.append(job)
    pairs = sorted({j["pair_id"] for j in jobs})
    comparison_pairs = (
        [("counterfactual_text", "baseline")]
        + [
            pair
            for layer in (0, 10)
            for pair in (
                (f"layer{layer}_natural", "baseline"),
                (f"layer{layer}_natural", f"layer{layer}_same_fact"),
            )
        ]
        if context
        else [
            pair
            for label in ("0p5", "1")
            for pair in (
                (f"subspace_forward_{label}", "baseline"),
                (
                    f"subspace_forward_{label}",
                    f"projected_norm_matched_random_subspace_{label}",
                ),
            )
        ]
    )
    comparisons = [
        {
            "id": f"{eid}_{t}_vs_{c}",
            "treatment": t,
            "control": c,
            "predicted_direction": "increase",
            "eligible_pair_ids": pairs,
            "expected_n_pairs": len(pairs),
            "expected_n_layouts": len(pairs),
        }
        for t, c in comparison_pairs
    ]
    spec = {
        "experiment_id": eid,
        "partition": "discovery",
        "jobs": jobs,
        "comparisons": comparisons,
    }
    if not context:
        spec["basis"] = {
            "family": family,
            "rank": rank,
            "layer": 0,
            "site": "resid_post",
        }
    spec_sha = fingerprint(spec)
    rows = []
    for job in jobs:
        success = job["condition"] in {
            "counterfactual_text",
            "donor_text",
            "full_transfer",
            "layer0_natural",
            "layer10_natural",
        } or job["condition"].startswith("subspace_forward_")
        rows.append(
            {
                **deepcopy(job),
                "spec_sha256": spec_sha,
                "status": "complete",
                "generation_status": "complete",
                "native_action": "RIGHT" if success else "UP",
                "seconds": 10.0,
            }
        )
    return refresh(
        {
            "experiment_id": eid,
            "specification": spec,
            "source_spec_sha256": spec_sha,
            "rows": rows,
            "independent_audit": {
                "path": f"reviews/{eid}_independent.json",
                "sha256": fingerprint(eid),
            },
            "n_jobs": len(rows),
        }
    )


def battery():
    sources = {eid: make_source(eid) for eid in select.REQUIRED_BATTERY}
    return {
        "sources": sources,
        "complete_cohort_means_seconds": {eid: 10.0 for eid in sources},
        "minimum_budget_seconds_per_native_job": 12.5,
    }


def change(
    source, condition, *, action=None, status="complete", seconds=None, index=None
):
    rows = sorted(
        (r for r in source["rows"] if r["condition"] == condition),
        key=lambda r: r["layout_id"],
    )
    for i, row in enumerate(rows):
        if index is None or i == index:
            if seconds is None:
                row["native_action"] = action
                row["generation_status"] = status
            else:
                row["seconds"] = seconds
    refresh(source)


class DiscoverySelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = battery()

    def evaluate(self):
        return select.evaluate_verified_sources(self.data)

    def profile(self, result, name):
        return next(p for p in result["profiles"] if p["profile_id"] == name)

    def test_enumerates_all14_profiles_and_selects_two_distinct_questions(self):
        result = self.evaluate()
        self.assertEqual(len(result["profiles"]), 14)
        self.assertEqual(
            result["selected_scientific_profile_ids"],
            {
                "contextual_input": "contextual_l10",
                "fixed_family_subspace": "subspace_wall_l0_r1_dose0p5",
            },
        )
        self.assertEqual(result["n_selected_scientific_profiles"], 2)
        self.assertFalse(result["adapter_feasibility_assessed"])
        self.assertFalse(result["confirmation_opened"])

    def test_unmeasured_contextual_control_has_no_fake_outcome_or_ranking_cost(self):
        profile = self.profile(self.evaluate(), "contextual_l0")
        unknown = profile["unmeasured_components"][0]
        for name in (
            "native_itt_delta",
            "completion_sensitive_point_bounds",
            "source_comparison_id",
            "qualifies_in_discovery",
        ):
            self.assertIsNone(unknown[name])
        self.assertFalse(unknown["included_in_ranking"])
        self.assertEqual(profile["measured_inventory"]["n_rows"], 84)
        self.assertEqual(profile["measured_inventory"]["total_seconds"], 840)
        self.assertNotIn(
            unknown["control"], profile["measured_inventory"]["conditions"]
        )

    def test_subspace_efficacy_deduplicated_and_physical_reference_not_fabricated(self):
        profile = self.profile(self.evaluate(), "subspace_wall_l0_r1_dose1")
        self.assertEqual(len(profile["measured_qualification_contrasts"]), 2)
        self.assertEqual(
            profile["measured_qualification_contrasts"][0]["roles"],
            ["subspace_efficacy", "efficacy"],
        )
        physical = profile["physical_counterfactual_reference"]
        self.assertEqual(physical["treatment"], "donor_text")
        self.assertIsNone(physical["source_comparison_id"])
        self.assertEqual(physical["origin"], "policy_physical_reference_diagnostic")

    def test_all_sources_required_before_pure_evaluation(self):
        del self.data["sources"][select.REQUIRED_BATTERY[-1]]
        with self.assertRaisesRegex(ValueError, "seven"):
            self.evaluate()

    def test_production_full_battery_gate_precedes_any_policy_or_evaluation(self):
        with (
            patch.object(
                select,
                "read_closed_native_battery",
                side_effect=ValueError("not closed"),
            ) as gate,
            patch.object(select, "evaluate_verified_sources") as calculate,
            patch.object(
                Path, "read_bytes", side_effect=AssertionError("premature policy read")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not closed"):
                select.evaluate_closed_discovery(Path("absent"), Path("absent"), {})
        gate.assert_called_once()
        calculate.assert_not_called()

    def test_production_checks_policy_hash_and_carries_whole_battery_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / select.POLICY_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(b"policy")
            with patch.object(
                select, "read_closed_native_battery", return_value=self.data
            ):
                with self.assertRaisesRegex(ValueError, "policy bytes"):
                    select.evaluate_closed_discovery(root, root, {})
                with patch.object(
                    select,
                    "POLICY_SHA256",
                    select.hashlib.sha256(b"policy").hexdigest(),
                ):
                    result = select.evaluate_closed_discovery(root, root, {})
            self.assertEqual(result["minimum_budget_seconds_per_native_job"], 12.5)

    def test_unresolved_baseline_retained_and_zero_lower_bound_fails(self):
        source = self.data["sources"]["subspace_wall_l0_r1_native_v1"]
        change(source, "baseline", status="truncated")
        profile = self.profile(self.evaluate(), "subspace_wall_l0_r1_dose1")
        self.assertFalse(profile["qualified"])
        metric = profile["measured_qualification_contrasts"][0]
        self.assertEqual(metric["native_itt_delta"], 1)
        self.assertEqual(metric["completion_sensitive_point_bounds"], [0, 1])
        self.assertEqual(profile["measured_inventory"]["unresolved_count"], 4)

    def test_damaged_random_is_insufficient_without_absolute_efficacy(self):
        source = self.data["sources"]["subspace_wall_l0_r1_native_v1"]
        change(source, "baseline", action="RIGHT")
        p = self.profile(self.evaluate(), "subspace_wall_l0_r1_dose1")
        self.assertFalse(p["qualified"])
        self.assertEqual(
            p["measured_qualification_contrasts"][1]["native_itt_delta"], 1
        )
        self.assertTrue(
            any(x["gate"] == "native_itt_at_least_0.10" for x in p["failed_gates"])
        )

    def test_physical_reference_must_have_positive_censoring_lower(self):
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        change(source, "counterfactual_text", action="UP")
        result = self.evaluate()
        self.assertIsNone(result["selected_scientific_profile_ids"]["contextual_input"])
        self.assertTrue(all(p["failed_gates"] for p in result["profiles"][:2]))

    def test_whole_mixture_does_not_add_an_unadopted_each_family_gate(self):
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        for condition in ("layer0_natural", "layer10_natural"):
            for row in source["rows"]:
                if row["condition"] == condition and row["family"] == "door_status":
                    row["native_action"] = "UP"
        refresh(source)
        p = self.profile(self.evaluate(), "contextual_l10")
        self.assertTrue(p["qualified"])
        self.assertEqual(
            p["measured_qualification_contrasts"][0]["native_itt_delta"], 4 / 6
        )
        self.assertEqual(
            p["measured_qualification_contrasts"][0]["family_effects"]["door_status"][
                "native_itt_delta"
            ],
            0,
        )

    def test_family_priority_is_not_effect_size_ranking(self):
        for rank in (1, 4):
            source = self.data["sources"][f"subspace_wall_l0_r{rank}_native_v1"]
            for condition in ("subspace_forward_0p5", "subspace_forward_1"):
                change(source, condition, action="UP", index=0)
        result = self.evaluate()
        self.assertTrue(
            result["selected_scientific_profile_ids"][
                "fixed_family_subspace"
            ].startswith("subspace_wall_")
        )
        for rank in (1, 4):
            source = self.data["sources"][f"subspace_wall_l0_r{rank}_native_v1"]
            for condition in ("subspace_forward_0p5", "subspace_forward_1"):
                change(source, condition, action="UP")
        result = self.evaluate()
        self.assertTrue(
            result["selected_scientific_profile_ids"][
                "fixed_family_subspace"
            ].startswith("subspace_key_possession_")
        )

    def test_fewer_unresolved_full_inventory_precedes_contextual_depth(self):
        change(
            self.data["sources"][select.CONTEXTUAL_SOURCE],
            "layer10_irrelevant",
            status="malformed",
            index=0,
        )
        self.assertEqual(
            self.evaluate()["selected_scientific_profile_ids"]["contextual_input"],
            "contextual_l0",
        )

    def test_subspace_doses_share_all14_cost_and_unresolved_counts(self):
        source = self.data["sources"]["subspace_wall_l0_r1_native_v1"]
        change(source, "random_subspace_1", status="truncated", index=0)
        change(source, "random_subspace_1", seconds=100, index=0)
        result = self.evaluate()
        a = self.profile(result, "subspace_wall_l0_r1_dose0p5")
        b = self.profile(result, "subspace_wall_l0_r1_dose1")
        self.assertEqual(a["measured_inventory"], b["measured_inventory"])
        self.assertEqual(a["measured_inventory"]["unresolved_count"], 1)
        self.assertEqual(a["measured_inventory"]["total_seconds"], 650)

    def test_no_qualifier_leaves_slots_empty_without_population_negative(self):
        for source in self.data["sources"].values():
            for row in source["rows"]:
                row["native_action"] = "UP"
            refresh(source)
        result = self.evaluate()
        self.assertEqual(result["n_selected_scientific_profiles"], 0)
        self.assertEqual(
            set(result["question_status"].values()), {"no_qualifying_candidate"}
        )
        self.assertTrue(
            result["claims"]["no_qualifier_is_not_a_bounded_population_negative"]
        )

    def test_configuration_fingerprint_excludes_outcomes_cost_and_forced_q(self):
        before = self.evaluate()
        for source in self.data["sources"].values():
            for row in source["rows"]:
                row.update(
                    native_action="UP", seconds=987, forced_q={"invented": "ignored"}
                )
            refresh(source)
        after = self.evaluate()
        self.assertEqual(
            [p["configuration_sha256"] for p in before["profiles"]],
            [p["configuration_sha256"] for p in after["profiles"]],
        )
        self.assertNotEqual(
            before["source_audit_bindings"], after["source_audit_bindings"]
        )

    def test_ranked_native_results_ignore_forced_fields(self):
        before = self.evaluate()
        for source in self.data["sources"].values():
            for row in source["rows"]:
                row["forced_q"] = None
                row["forced_top_action"] = "garbage"
            refresh(source)
        after = self.evaluate()
        self.assertEqual(
            before["selected_scientific_profile_ids"],
            after["selected_scientific_profile_ids"],
        )
        self.assertEqual(
            [p["ranking_tuple"] for p in before["profiles"]],
            [p["ranking_tuple"] for p in after["profiles"]],
        )

    def test_missing_duplicate_error_and_non_native_rows_fail_closed(self):
        for mutation in (
            lambda s: s["rows"].pop(),
            lambda s: s["rows"].__setitem__(0, deepcopy(s["rows"][1])),
            lambda s: s["rows"][0].update(status="error"),
            lambda s: s["rows"][0].update(
                generation_status="not_requested", native_action=None
            ),
        ):
            self.data = battery()
            source = self.data["sources"][select.CONTEXTUAL_SOURCE]
            mutation(source)
            refresh(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.evaluate()

    def test_immutable_float_seed_and_restricted_comparison_rejected(self):
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        source["rows"][0]["seed"] = 42.0
        refresh(source)
        with self.assertRaisesRegex(ValueError, "immutable source field"):
            self.evaluate()
        self.data = battery()
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        source["specification"]["comparisons"][0]["eligible_pair_ids"].pop()
        with self.assertRaisesRegex(ValueError, "restricted source"):
            self.evaluate()

    def test_geometry_and_cross_rank_population_drift_rejected(self):
        source = self.data["sources"]["subspace_wall_l0_r4_native_v1"]
        source["specification"]["basis"]["rank"] = True
        with self.assertRaisesRegex(ValueError, "basis rank"):
            self.evaluate()
        self.data = battery()
        source = self.data["sources"]["subspace_wall_l0_r4_native_v1"]
        for job in source["specification"]["jobs"]:
            old = job["job_id"]
            job["layout_id"] += "changed"
            job["job_id"] = fingerprint(
                {k: v for k, v in job.items() if k != "job_id"}
            )[:24]
            row = next(r for r in source["rows"] if r["job_id"] == old)
            row.update(layout_id=job["layout_id"], job_id=job["job_id"])
        refresh(source)
        with self.assertRaisesRegex(ValueError, "shared layout"):
            self.evaluate()

    def test_pure_evaluation_has_no_file_access_or_input_mutation(self):
        before = deepcopy(self.data)
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("pure helper read a file")
        ):
            result = self.evaluate()
        self.assertEqual(self.data, before)
        result["source_audit_bindings"][select.CONTEXTUAL_SOURCE][
            "source_result_hashes"
        ].clear()
        self.assertEqual(self.data, before)

    def test_all_rank_key_priorities_are_lexicographic_and_cost_last(self):
        original = self.profile(self.evaluate(), "subspace_wall_l0_r4_dose1")
        changes = [
            ("ranking_metrics", "minimum_censoring_lower", 1.1),
            ("ranking_metrics", "minimum_itt", 1.1),
            ("ranking_metrics", "efficacy_itt", 1.1),
            ("measured_inventory", "unresolved_count", -1),
            ("geometry", "patched_layer_count", 0),
            ("geometry", "total_selected_positions", 3),
            ("geometry", "rank", 1),
            ("geometry", "dose", 0.5),
            ("measured_inventory", "total_seconds", 1),
        ]
        for group, key, value in changes:
            changed = deepcopy(original)
            (
                changed["configuration"]["geometry"]
                if group == "geometry"
                else changed[group]
            )[key] = value
            with self.subTest(group=group, key=key):
                self.assertLess(
                    select._ranking_key(changed), select._ranking_key(original)
                )
        changed = deepcopy(original)
        changed["ranking_metrics"]["efficacy_itt"] = 0.9
        changed["measured_inventory"]["total_seconds"] = 0.001
        self.assertGreater(select._ranking_key(changed), select._ranking_key(original))
        changed = deepcopy(original)
        changed["configuration_sha256"] = "0" * 64
        self.assertLess(select._ranking_key(changed), select._ranking_key(original))

    def test_missing_audit_manifest_or_comparison_origin_rejected(self):
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        source["independent_audit"].pop("sha256")
        with self.assertRaisesRegex(ValueError, "audit binding"):
            self.evaluate()
        self.data = battery()
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        source["source_result_hashes"].pop(next(iter(source["source_result_hashes"])))
        with self.assertRaisesRegex(ValueError, "manifest incomplete"):
            self.evaluate()
        self.data = battery()
        source = self.data["sources"][select.CONTEXTUAL_SOURCE]
        source["specification"]["comparisons"][1]["id"] = ""
        with self.assertRaisesRegex(ValueError, "comparison ID"):
            self.evaluate()

    def test_lexicographic_ranking_order_and_outcome_free_descriptor(self):
        p = self.profile(self.evaluate(), "contextual_l0")
        q = deepcopy(p)
        q["configuration"]["geometry"]["layer"] = 10
        self.assertLess(select._ranking_key(q), select._ranking_key(p))
        q["measured_inventory"]["unresolved_count"] += 1
        self.assertGreater(select._ranking_key(q), select._ranking_key(p))
        q = deepcopy(p)
        q["configuration"]["geometry"]["total_selected_positions"] -= 1
        self.assertLess(select._ranking_key(q), select._ranking_key(p))
        q = deepcopy(p)
        q["ranking_metrics"]["minimum_censoring_lower"] -= 0.1
        q["ranking_metrics"]["minimum_itt"] += 1
        self.assertGreater(select._ranking_key(q), select._ranking_key(p))
        self.assertEqual(
            set(p["descriptor"]),
            {
                "profile_id",
                "protocol_kind",
                "source_experiment_id",
                "source_spec_sha256",
                "layer",
                "site",
                "selected_dose",
                "rank",
                "family",
            },
        )
        self.assertIsNone(p["descriptor"]["family"])


if __name__ == "__main__":
    unittest.main()
