"""Scientific analysis regression tests without MLX, pytest, or model weights."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activation_causality.analysis import (  # noqa: E402
    SEMANTIC_CONTROL_ROLES,
    SEMANTIC_EVIDENCE,
    _stratified_draws,
    _stratified_p,
    analyze_results,
    holm_adjust,
    load_results,
    write_report,
)


def row(layout, condition, *, action="RIGHT", partition="confirmation", status="complete", **kwargs):
    return {
        "experiment_id": "wall_transfer", "layout_id": str(layout), "pair_id": f"pair-{layout}",
        "condition": condition, "family": "wall", "partition": partition, "seed": 42,
        "target_action": "RIGHT", "native_action": action if status == "complete" else None,
        "generation_status": status, "forced_q": {"UP": .25, "DOWN": .25, "LEFT": .25, "RIGHT": .25},
        **kwargs,
    }


def positive_rows(n=64, partition="confirmation"):
    return [item for i in range(n) for item in (
        row(i, "patch", partition=partition), row(i, "baseline", action="UP", partition=partition),
    )]


def with_native_controls(rows):
    rows = list(rows)
    baselines = [item for item in rows if item["condition"] == "baseline"]
    for item in baselines:
        for role in SEMANTIC_CONTROL_ROLES:
            action = "RIGHT" if role in {"counterfactual_baseline", "donor_mismatch_treatment"} else "UP"
            rows.append({**item, "condition": role, "native_action": action, "generation_status": "complete"})
    return rows


def comparison(n=64, **kwargs):
    return {"id": "wall_candidate", "experiment_id": "wall_transfer", "treatment": "patch",
            "control": "baseline", "partition": "confirmation", "expected_n_layouts": n,
            "expected_n_pairs": n,
            "evidence": {flag: True for flag in SEMANTIC_EVIDENCE}, **kwargs}


def analyze(rows, comparisons=None, **kwargs):
    return analyze_results(rows, comparisons=comparisons, n_bootstrap=500, n_permutations=500, **kwargs)


def mixed_fixture():
    counts = {"wall": 11, "key_possession": 11, "door_status": 10}
    rows = []
    for family, count in counts.items():
        for index in range(count):
            layout = f"{family}-{index}"
            rows.extend([row(layout, "patch", family=family), row(layout, "baseline", family=family, action="UP")])
    spec = comparison(32, semantic_headline_eligible=False,
                      resampling_scheme="within_family_layout_bootstrap", stratification_key="family",
                      family_layout_counts=counts, family_weights={f: n / 32 for f, n in counts.items()},
                      expected_n_layouts_by_family=counts.copy(), expected_n_pairs_by_family=counts.copy())
    return rows, spec


class AnalysisTests(unittest.TestCase):
    def test_mixed_family_positive_endpoint_has_frozen_quota_audit_and_guard(self):
        rows, spec = mixed_fixture()
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertFalse(result["headline_semantic_support"])
        self.assertTrue(result["family_stratification"]["frozen_cohort_complete"])
        self.assertEqual(result["primary"]["effect"], 1)
        self.assertGreater(result["primary"]["adjusted_ci"][0], 0)
        self.assertLess(result["primary"]["adjusted_ci"][0], 1)
        self.assertAlmostEqual(result["primary"]["p_value"], 2 * math.exp(-16))
        self.assertIn("hoeffding", result["primary"]["p_value_method"])
        self.assertIn("within_family_constant_effect_simultaneous_tail_guard", result["warnings"])
        self.assertEqual(set(result["secondary_native_by_family"]), set(spec["family_weights"]))

    def test_mixed_constant_small_or_differing_family_scores_do_not_get_tiny_p(self):
        groups = [np.arange(11), np.arange(11, 22), np.arange(22, 32)]
        weights = [11 / 32, 11 / 32, 10 / 32]
        for values in (np.full(32, .01), np.r_[np.ones(11), np.zeros(11), -np.ones(10)]):
            p_value, method, residual = _stratified_p(values, groups, weights, n_resamples=500, seed=42)
            self.assertEqual(p_value, 1)
            self.assertIn("hoeffding", method)
            self.assertAlmostEqual(residual, 0)

    def test_stratified_draws_keep_family_composition_fixed(self):
        values = np.r_[np.ones(11), np.zeros(11), -np.ones(10)]
        draws = _stratified_draws(values, [np.arange(11), np.arange(11, 22), np.arange(22, 32)],
                                  [11 / 32, 11 / 32, 10 / 32], n_resamples=500, seed=42)
        self.assertTrue(np.all(draws == 1 / 32))

    def test_mixed_layout_weighting_and_completion_bounds(self):
        rows, spec = mixed_fixture()
        for item in rows:
            if item["family"] == "wall" and item["condition"] == "patch":
                item.update(generation_status="truncated", native_action=None)
            elif item["family"] == "door_status" and item["condition"] == "baseline":
                item["native_action"] = "RIGHT"
        # Repeat one layout nine times; one family gets more trials, not weight.
        for seed in range(100, 109):
            rows.extend([{**rows[0], "seed": seed}, {**rows[1], "seed": seed}])
        spec["expected_n_pairs"] += 9
        spec["expected_n_pairs_by_family"]["wall"] += 9
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertAlmostEqual(result["primary"]["effect"], 11 / 32)
        self.assertAlmostEqual(result["treatment_completion_rate"], 21 / 32)
        self.assertAlmostEqual(result["completion_sensitive_effect_bounds"]["lower"], 11 / 32)
        self.assertAlmostEqual(result["completion_sensitive_effect_bounds"]["upper"], 22 / 32)
        self.assertEqual(result["secondary_native_by_family"]["wall"]["n_pairs"], 20)

    def test_mixed_incomplete_or_substituted_family_does_not_run_inference(self):
        rows, spec = mixed_fixture()
        for mode in ("missing_one", "missing_family", "substituted"):
            changed = [dict(item) for item in rows]
            if mode == "missing_one":
                changed = [item for item in changed if item["layout_id"] != "wall-0"]
            elif mode == "missing_family":
                changed = [item for item in changed if item["family"] != "wall"]
            else:
                for item in changed:
                    if item["layout_id"] == "wall-0":
                        item["family"] = "door_status"
            result = analyze(changed, [spec])["comparisons"][0]
            self.assertEqual(result["statistical_verdict"], "technically_invalid")
            self.assertIsNone(result["primary"]["ci"])
            self.assertIsNone(result["primary"]["p_value"])
            self.assertIn("frozen_family_counts_not_met", result["validity_blockers"])

    def test_mixed_contract_rejects_weights_count_and_layout_stratum_changes(self):
        rows, spec = mixed_fixture()
        for key, value in (("family_weights", {"wall": 1 / 3, "key_possession": 1 / 3, "door_status": 1 / 3}),
                           ("expected_n_pairs", 31), ("resampling_scheme", "pooled")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze(rows, [{**spec, key: value}])
        rows[0]["family"] = "key_possession"
        with self.assertRaisesRegex(ValueError, "multiple frozen family strata"):
            analyze(rows, [spec])

    def test_holm_adjustment_preserves_order_and_monotonicity(self):
        self.assertEqual(holm_adjust([.04, .01]), [.04, .02])
        self.assertEqual(holm_adjust([.03, .04]), [.06, .06])
        self.assertEqual(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])

    def test_positive_confirmation_needs_semantic_evidence(self):
        rows = with_native_controls(positive_rows())
        spec = comparison(semantic_controls={role: role for role in SEMANTIC_CONTROL_ROLES})
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertTrue(result["headline_semantic_support"])
        self.assertEqual(result["primary"]["effect"], 1)
        self.assertLess(result["primary"]["holm_p_value"], .05)
        self.assertIn("not_established", result["necessity_claim"])
        result = analyze(rows, [comparison(evidence={})])["comparisons"][0]
        self.assertFalse(result["headline_semantic_support"])
        self.assertEqual(set(result["missing_semantic_evidence"]), set(SEMANTIC_EVIDENCE))

    def test_true_flags_cannot_replace_missing_native_controls(self):
        result = analyze(positive_rows(), [comparison()])["comparisons"][0]
        self.assertFalse(result["headline_semantic_support"])
        self.assertFalse(result["semantic_control_audit"]["native_controls_ready"])

    def test_frozen_input_scope_cannot_become_semantic_support(self):
        rows = with_native_controls(positive_rows())
        spec = comparison(semantic_controls={role: role for role in SEMANTIC_CONTROL_ROLES},
                          semantic_headline_eligible=False)
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertFalse(result["headline_semantic_support"])
        del spec["semantic_headline_eligible"]
        for item in rows:
            item["metadata"] = {"claim_scope": "input_fact_information_routing"}
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertFalse(result["semantic_headline_eligible"])
        self.assertFalse(result["headline_semantic_support"])

    def test_completion_only_difference_cannot_support_eventual_semantic_claim(self):
        rows = with_native_controls(positive_rows())
        for item in rows:
            if item["condition"] == "baseline":
                item.update(native_action=None, generation_status="truncated")
        spec = comparison(semantic_controls={role: role for role in SEMANTIC_CONTROL_ROLES})
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertTrue(result["semantic_control_audit"]["native_controls_ready"])
        self.assertFalse(result["eventual_action_robust_to_censoring"])
        self.assertFalse(result["headline_semantic_support"])

    def test_model_donor_action_must_actually_mismatch(self):
        rows = with_native_controls(positive_rows())
        for item in rows:
            if item["condition"] == "donor_baseline":
                item["native_action"] = "RIGHT"
        result = analyze(rows, [comparison(semantic_controls={role: role for role in SEMANTIC_CONTROL_ROLES})])["comparisons"][0]
        self.assertEqual(result["semantic_control_audit"]["observed_donor_mismatch_layouts"], 0)
        self.assertFalse(result["headline_semantic_support"])

    def test_sentence_control_cannot_become_semantic_support(self):
        rows = [{**item, "family": "sentence_control"} for item in positive_rows()]
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertFalse(result["headline_semantic_support"])

    def test_inferred_or_discovery_comparisons_remain_exploratory(self):
        result = analyze(positive_rows())["comparisons"][0]
        self.assertFalse(result["explicit_confirmation"])
        self.assertFalse(result["headline_semantic_support"])
        self.assertTrue(result["claim"].startswith("Exploratory only"))
        rows = positive_rows(partition="discovery")
        result = analyze(rows, [comparison(partition="discovery")])["comparisons"][0]
        self.assertFalse(result["headline_semantic_support"])

    def test_pairing_weights_layouts_not_repeated_seeds(self):
        rows = positive_rows(n=2, partition="discovery")
        rows[2]["native_action"] = "UP"
        # Layout zero contributes +1; layout one 0. Nine extra seeds on layout
        # zero must not move the equal-layout estimate away from one half.
        for seed in range(100, 109):
            rows.extend([{**rows[0], "seed": seed}, {**rows[1], "seed": seed}])
        result = analyze(rows)["comparisons"][0]
        self.assertEqual(result["n_layouts"], 2)
        self.assertEqual(result["n_pairs"], 11)
        self.assertEqual(result["primary"]["effect"], .5)

    def test_noncompletions_are_retained_and_censoring_bounds_are_explicit(self):
        rows = [row(0, "patch", status="truncated", partition="discovery"),
                row(0, "baseline", action="RIGHT", partition="discovery"),
                row(1, "patch", partition="discovery"),
                row(1, "baseline", status="malformed", partition="discovery")]
        result = analyze(rows)["comparisons"][0]
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["primary"]["effect"], 0)
        self.assertEqual(result["treatment_completion_rate"], .5)
        self.assertEqual(result["control_completion_rate"], .5)
        self.assertEqual(result["completion_sensitive_effect_bounds"]["lower"], -.5)
        self.assertEqual(result["completion_sensitive_effect_bounds"]["upper"], .5)
        self.assertEqual(result["exclusions"]["noncompletions_excluded"], 0)

    def test_forced_only_outputs_never_substitute_for_natural_actions(self):
        rows = [{**item, "native_action": None, "generation_status": "not_requested"}
                for item in positive_rows(partition="discovery")]
        for item in rows:
            item["forced_q"] = dict(zip(("UP", "DOWN", "LEFT", "RIGHT"),
                                        (0, 0, 0, 1) if item["condition"] == "patch" else (1, 0, 0, 0)))
        result = analyze(rows)["comparisons"][0]
        self.assertEqual(result["n_pairs"], 0)
        self.assertEqual(result["n_execution_pairs"], 64)
        self.assertIsNone(result["primary"]["effect"])
        self.assertEqual(result["statistical_verdict"], "not_measured")
        self.assertEqual(result["secondary_forced"]["mean_target_probability_effect"], 1)
        self.assertFalse(result["headline_semantic_support"])

    def test_target_vs_original_log_odds_are_separate_from_target_vs_rest(self):
        rows = positive_rows(n=2, partition="discovery")
        for item in rows:
            item["original_action"] = "UP"
            if item["condition"] == "patch":
                item["forced_q"] = {"UP": .1, "DOWN": .2, "LEFT": .3, "RIGHT": .4}
        result = analyze(rows)["comparisons"][0]
        secondary = result["secondary_forced"]
        import math
        self.assertAlmostEqual(secondary["mean_target_vs_original_log_odds_effect"], math.log(4))
        self.assertAlmostEqual(secondary["mean_target_log_odds_effect"], math.log(2))
        self.assertEqual(secondary["target_vs_original_n_layouts"], 2)

    def test_technical_errors_block_inference_without_counting_as_behavior(self):
        rows = positive_rows()
        rows[0].update(native_action=None, generation_status="technical_error", status="error")
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["n_pairs"], 63)
        self.assertEqual(result["exclusions"]["technical_error_rows"], 1)
        self.assertEqual(result["statistical_verdict"], "technically_invalid")

    def test_missing_counterpart_and_wrong_frozen_count_block_confirmation(self):
        rows = positive_rows()[:-1]
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["exclusions"]["treatment_without_control"], 1)
        self.assertIn("frozen_layout_count_not_met", result["validity_blockers"])
        self.assertFalse(result["headline_semantic_support"])

    def test_heldout_layout_overlap_blocks_confirmation(self):
        rows = positive_rows() + [row(0, "unused", partition="discovery")]
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["exclusions"]["cross_partition_layout_ids"], ["0"])
        self.assertEqual(result["statistical_verdict"], "technically_invalid")

    def test_layout_alias_overlap_cannot_hide_behind_different_ids(self):
        rows = positive_rows() + [row("variant", "unused", partition="discovery", layout_aliases=["0"])]
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["exclusions"]["cross_partition_layout_ids"], ["0"])
        self.assertIn("inconsistent_layout_alias_grouping", result["validity_blockers"])

    def test_missing_whole_seed_pairs_still_block_confirmation(self):
        result = analyze(positive_rows(), [comparison(expected_n_pairs=128)])["comparisons"][0]
        self.assertEqual(result["n_layouts"], 64)
        self.assertIn("frozen_pair_count_not_met", result["validity_blockers"])

    def test_explicit_eligible_subset_prevents_false_missing_control_errors(self):
        rows = positive_rows(n=64)
        rows = [item for item in rows if item["condition"] == "baseline" or int(item["layout_id"]) < 32]
        spec = comparison(n=32, eligible_pair_ids=[f"pair-{i}" for i in range(32)])
        result = analyze(rows, [spec])["comparisons"][0]
        self.assertEqual(result["n_pairs"], 32)
        self.assertEqual(result["exclusions"]["control_without_treatment"], 0)
        self.assertEqual(result["statistical_verdict"], "supported")
        with self.assertRaisesRegex(ValueError, "eligible_pair_ids must be unique"):
            analyze(rows, [comparison(eligible_pair_ids=["pair-0", "pair-0"])])

    def test_multiple_families_do_not_change_layout_weighting(self):
        rows = positive_rows(2, partition="discovery")
        rows[2].update(family="door_status", native_action="UP")
        rows[3]["family"] = "door_status"
        rows += [{**rows[index], "pair_id": "extra-pair"} for index in (0, 1)]
        result = analyze(rows)["comparisons"][0]
        self.assertEqual(result["primary"]["effect"], .5)
        self.assertEqual(result["families"], ["door_status", "wall"])

    def test_duplicate_rows_and_inconsistent_targets_are_not_silently_dropped(self):
        rows = positive_rows()
        with self.assertRaisesRegex(ValueError, "duplicate paired"):
            analyze(rows + [rows[0]])
        rows[1]["target_action"] = "UP"
        with self.assertRaisesRegex(ValueError, "inconsistent paired target"):
            analyze(rows, [comparison()])

    def test_invalid_generation_or_probability_is_rejected(self):
        cases = [dict(generation_status="complete", native_action=None),
                 dict(generation_status="truncated", native_action="RIGHT"),
                 dict(forced_q={"UP": .1, "DOWN": .1, "LEFT": .1, "RIGHT": .1}),
                 dict(forced_q={"UP": float("nan"), "DOWN": 0, "LEFT": 0, "RIGHT": 1})]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyze([{**row(0, "baseline"), **invalid}])

    def test_two_candidate_limit_and_multiple_comparison_adjustment(self):
        rows = positive_rows()
        rows += [{**item, "condition": "patch2"} for item in rows if item["condition"] == "patch"]
        specs = [comparison(), comparison(id="second", treatment="patch2")]
        result = analyze(rows, specs)
        self.assertEqual(result["analysis_config"]["confirmation_family_size"], 2)
        for candidate in result["comparisons"]:
            self.assertGreaterEqual(candidate["primary"]["holm_p_value"], candidate["primary"]["p_value"])
            self.assertLessEqual(candidate["primary"]["adjusted_ci"][0], candidate["primary"]["ci"][0])
        with self.assertRaisesRegex(ValueError, "at most two"):
            analyze(rows, specs + [comparison(id="third", treatment="patch2")])
        result = analyze(rows, specs + [comparison(id="diagnostic", treatment="patch2", confirmatory=False)])
        self.assertEqual(result["analysis_config"]["confirmation_family_size"], 2)
        self.assertFalse(result["comparisons"][-1]["explicit_confirmation"])

    def test_successful_reverse_effect_is_not_called_a_negative_result(self):
        rows = positive_rows()
        for item in rows:
            item["native_action"] = "UP" if item["condition"] == "patch" else "RIGHT"
        result = analyze(rows, [comparison(predicted_direction="decrease")])["comparisons"][0]
        self.assertEqual(result["primary"]["effect"], -1)
        self.assertEqual(result["primary"]["effect_in_predicted_direction"], 1)
        self.assertEqual(result["statistical_verdict"], "supported")
        self.assertFalse(result["headline_semantic_support"])

    def test_separate_candidate_report_keeps_frozen_family_size(self):
        result = analyze(positive_rows(), [comparison()], confirmation_family_size=2)
        primary = result["comparisons"][0]["primary"]
        self.assertEqual(result["analysis_config"]["confirmation_family_size"], 2)
        self.assertAlmostEqual(primary["holm_p_value"], 2 * primary["p_value"])
        self.assertLess(primary["adjusted_ci"][0], primary["ci"][0])

    def test_identity_contrast_has_an_equivalence_verdict(self):
        rows = positive_rows()
        for item in rows:
            item["native_action"] = "RIGHT"
        result = analyze(rows, [comparison(predicted_direction="zero")])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "consistent_with_equivalence")
        self.assertFalse(result["headline_semantic_support"])

    def test_centered_test_records_zero_null_residual_mean(self):
        rows = positive_rows()
        for item in rows:
            if int(item["layout_id"]) % 3 == 0:
                item["native_action"] = "UP"
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertEqual(result["primary"]["p_value_method"], "centered_layout_bootstrap_plus_one")
        self.assertAlmostEqual(result["primary"]["null_residual_mean"], 0, places=14)

    def test_zero_variance_bootstrap_does_not_claim_zero_uncertainty(self):
        rows = positive_rows(n=32)
        for item in rows:
            item["native_action"] = "UP"
        result = analyze(rows, [comparison(n=32)])["comparisons"][0]
        self.assertGreater(result["primary"]["adjusted_ci"][1], 0)
        self.assertEqual(result["statistical_verdict"], "inconclusive")
        rows = positive_rows(n=64)
        for item in rows:
            item["native_action"] = "UP"
        result = analyze(rows, [comparison(n=64)])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "bounded_negative")

    def test_reviewer_rejection_overrides_a_large_effect(self):
        evidence = {flag: True for flag in SEMANTIC_EVIDENCE}
        evidence["technical_valid"] = False
        result = analyze(positive_rows(), [comparison(evidence=evidence)])["comparisons"][0]
        self.assertEqual(result["statistical_verdict"], "technically_invalid")
        self.assertFalse(result["headline_semantic_support"])

    def test_mixed_execution_versions_or_head_modes_block_confirmation(self):
        rows = [{**item, "execution_key": "version-a", "inference_configuration": {"head_mode": "full"}}
                for item in positive_rows()]
        rows[0]["execution_key"] = "version-b"
        rows[0]["inference_configuration"]["head_mode"] = "last_token"
        result = analyze(rows, [comparison()])["comparisons"][0]
        self.assertIn("mixed_execution_configuration", result["validity_blockers"])
        self.assertFalse(result["headline_semantic_support"])

    def test_reordering_rows_does_not_change_resampling(self):
        rows = positive_rows()
        for index in range(0, 40, 2):
            rows[index]["native_action"] = "UP"
        original = analyze(rows, [comparison()])
        reordered = analyze(reversed(rows), [comparison()])
        self.assertEqual(original, reordered)

    def test_jsonl_and_report_are_json_safe_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            rows = positive_rows()
            path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
            self.assertEqual(load_results(path), rows)
            output = Path(temp) / "report"
            summary = write_report(path, output, comparisons=[comparison()], make_figure=False,
                                   n_bootstrap=500, n_permutations=500)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertIn("Necessity and mediation are not established", (output / "report.md").read_text())
            json.dumps(summary, allow_nan=False)

    def test_small_pilot_report_foregrounds_exploration_and_censoring(self):
        rows = positive_rows(n=6, partition="historical")
        for item in rows:
            if int(item["layout_id"]) >= 3 and item["condition"] == "patch":
                item["native_action"] = "UP"
            if int(item["layout_id"]) < 3 and item["condition"] == "baseline":
                item.update(native_action=None, generation_status="truncated")
        with tempfile.TemporaryDirectory() as temp:
            result = write_report(rows, temp, make_figure=False, n_bootstrap=1000, n_permutations=1000)
            candidate = result["comparisons"][0]
            self.assertEqual(candidate["report_label"], "exploratory_signal")
            self.assertTrue(candidate["small_sample_descriptive_only"])
            report = (Path(temp) / "report.md").read_text()
            self.assertIn("All comparisons below are exploratory", report)
            self.assertIn("Eventual-action uncertainty remains", report)
            self.assertIn("Completion-sensitive bounds", report)
            self.assertNotIn("| supported |", report)

    def test_empty_results_are_honest(self):
        result = analyze([])
        self.assertEqual(result["n_rows"], 0)
        self.assertEqual(result["comparisons"], [])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
