"""Synthetic co-primary inference tests; no models, live outcomes or weights."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from src.activation_causality import analysis  # noqa: E402
from src.activation_causality.primary_family import (  # noqa: E402
    ENDPOINT, UNRESOLVED_POLICY, validate_primary_family,
)


def binding(treatment="patch", control="baseline", n=32):
    return {"treatment": treatment, "control": control, "predicted_direction": "increase",
            "primary_endpoint": ENDPOINT, "target_action_basis": "unique_simulator_counterfactual_optimum",
            "native_unresolved_policy": UNRESOLVED_POLICY, "expected_n_layouts": n, "expected_n_pairs": n,
            "eligibility_rule": {"identifier": "frozen_simulator_tokenizer_rule", "sha256": "a" * 64},
            "origin": {"kind": "existing_source_comparison", "provenance": {"source_spec_sha256": "b" * 64}},
            "claim_scope": "input_fact_information_routing", "semantic_headline_eligible": False}


def registry(n_candidates=2, n_primaries=2):
    candidates = {letter: [f"p{index}" for index in range(n_primaries)] for letter in "AB"[:n_candidates]}
    bindings = {name: {primary: binding("patch" + primary, "baseline") for primary in ids}
                for name, ids in candidates.items()}
    return {"schema_version": 1, "candidates": candidates,
            "family_size": n_candidates * n_primaries, "bindings": bindings}


def comparison(family, candidate, primary):
    return {**deepcopy(family["bindings"][candidate][primary]), "id": primary,
            "candidate_id": candidate, "primary_id": primary, "experiment_id": "experiment" + candidate,
            "partition": "confirmation", "confirmatory": True,
            "evidence": {name: True for name in analysis.SEMANTIC_EVIDENCE}}


def rows_for(family, candidate, *, n=32, effect=1.0):
    conditions = {value["treatment"] for value in family["bindings"][candidate].values()} | {"baseline"}
    rows = []
    for index in range(n):
        for condition in sorted(conditions):
            action = "RIGHT" if condition != "baseline" and index < n * effect else "UP"
            rows.append({"experiment_id": "experiment" + candidate, "layout_id": f"{candidate}-{index}",
                         "pair_id": f"{candidate}-pair-{index}", "family": "wall", "condition": condition,
                         "partition": "confirmation", "seed": 42, "target_action": "RIGHT",
                         "native_action": action, "generation_status": "complete"})
    return rows


def analyze(rows, comparisons, family, **kwargs):
    return analysis.analyze_results(rows, comparisons=comparisons, confirmation_primary_family=family,
                                    n_bootstrap=200, n_permutations=200, **kwargs)


class PrimaryFamilyTests(unittest.TestCase):
    def test_imports_are_stage_isolated(self):
        for name, module in sys.modules.items():
            if name == "src" or name.startswith("src."):
                if getattr(module, "__file__", None):
                    self.assertTrue(Path(module.__file__).resolve().is_relative_to(STAGE), name)
        self.assertFalse(any(name in sys.modules for name in ("mlx", "mlx_lm", "torch")))

    def test_all_supported_registry_dimensions(self):
        for candidates, primaries in ((1, 1), (1, 2), (2, 1), (2, 2)):
            family = registry(candidates, primaries)
            summary = analyze([], [], family)
            self.assertEqual(len(summary["primary_components"]), candidates * primaries)
            self.assertEqual(summary["analysis_config"]["confirmation_family_size"], candidates * primaries)
            self.assertTrue(all(c["status"] == "pending" for c in summary["candidate_conjunctions"]))

    def test_invalid_registry_inventory_and_schema(self):
        base = registry()
        variants = []
        for field in ("bindings", "family_size", "candidates"):
            item = deepcopy(base)
            del item[field]
            variants.append(item)
        item = deepcopy(base)
        item["family_size"] = 2
        variants.append(item)
        item = deepcopy(base)
        item["candidates"]["C"] = ["p0"]
        item["bindings"]["C"] = {"p0": binding()}
        variants.append(item)
        item = deepcopy(base)
        item["candidates"]["A"] = ["p0", "p1", "p2"]
        variants.append(item)
        item = deepcopy(base)
        item["candidates"]["A"] = ["p0", "p0"]
        variants.append(item)
        item = deepcopy(base)
        del item["bindings"]["A"]["p0"]
        variants.append(item)
        item = deepcopy(base)
        item["bindings"]["A"]["extra"] = binding()
        variants.append(item)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_primary_family(value)

    def test_binding_mutations_and_primary_downgrade_rejected_before_row_load(self):
        family = registry(1, 1)
        spec = comparison(family, "A", "p0")
        changes = {"control": "other", "primary_endpoint": "forced", "expected_n_layouts": 64,
                   "expected_n_pairs": 64, "predicted_direction": "decrease", "confirmatory": False,
                   "partition": "discovery", "origin": {"kind": "other", "provenance": {}},
                   "eligibility_rule": {"identifier": "different", "sha256": "a" * 64}}
        for field, value in changes.items():
            with self.subTest(field=field), patch.object(analysis, "load_results", wraps=analysis.load_results) as loader:
                with self.assertRaises(ValueError):
                    analyze([], [{**spec, field: value}], family)
                self.assertEqual(loader.call_count, 1)  # empty legacy config only
        with self.assertRaisesRegex(ValueError, "family-size override"):
            analyze([], [], family, confirmation_family_size=1)

    def test_unknown_duplicate_and_unregistered_primaries_fail(self):
        family = registry(1, 1)
        spec = comparison(family, "A", "p0")
        for specs in ([spec, spec], [{**spec, "primary_id": "unknown"}],
                      [{**spec, "candidate_id": "unknown"}], [{k: v for k, v in spec.items() if k != "primary_id"}]):
            with self.assertRaises(ValueError):
                analyze([], specs, family)

    def test_exact_hand_holm_and_bonferroni_family_with_separate_reports(self):
        family = registry()
        specs = [comparison(family, candidate, primary) for candidate, ids in family["candidates"].items() for primary in ids]
        rows = rows_for(family, "A") + rows_for(family, "B")
        values = {("A", "p0"): .02, ("A", "p1"): .03, ("B", "p0"): .001, ("B", "p1"): .005}
        original = analysis._comparison
        measured_sizes = []

        def fixed_p(records, spec, **kwargs):
            result = original(records, spec, **kwargs)
            if kwargs.get("run_inference", True):
                measured_sizes.append(kwargs["family_size"])
                result["primary"]["p_value"] = values[spec["candidate_id"], spec["primary_id"]]
            return result

        with patch.object(analysis, "_comparison", fixed_p):
            joint = analyze(rows, specs, family)
            separate = analyze(rows_for(family, "A"), specs[:2], family)
        self.assertEqual([c["primary"]["holm_p_value"] for c in joint["primary_components"]], [.04, .04, .004, .015])
        self.assertEqual([c["primary"]["holm_p_value"] for c in separate["primary_components"]], [.08, .09, 1, 1])
        self.assertEqual(set(measured_sizes), {4})
        self.assertEqual(joint["confirmation_primary_family_sha256"], separate["confirmation_primary_family_sha256"])
        self.assertTrue(separate["multiplicity_adjustment_provisional"])
        self.assertFalse(joint["multiplicity_adjustment_provisional"])
        # Real intervals use alpha/4, not an adjusted p-value as an interval level.
        vector = __import__("numpy").ones(32)
        ci, _ = analysis._bootstrap_interval(vector, alpha=.05 / 4, n_bootstrap=200, seed=42)
        self.assertEqual(joint["primary_components"][0]["primary"]["adjusted_ci"], ci)

    def test_two_supported_components_conjoin_without_semantic_upgrade(self):
        family = registry(1, 2)
        specs = [comparison(family, "A", p) for p in family["candidates"]["A"]]
        summary = analyze(rows_for(family, "A"), specs, family)
        self.assertTrue(all(c["statistical_verdict"] == "supported" for c in summary["primary_components"]))
        self.assertEqual(summary["candidate_conjunctions"][0]["status"], "supported")
        self.assertFalse(summary["candidate_conjunctions"][0]["headline_semantic_support"])
        self.assertEqual(summary["n_rows"], 96)  # shared baseline not duplicated
        self.assertEqual([c["n_layouts"] for c in summary["primary_components"]], [32, 32])

    def test_one_failed_component_never_supports_conjunction(self):
        family = registry(1, 2)
        specs = [comparison(family, "A", p) for p in family["candidates"]["A"]]
        rows = rows_for(family, "A")
        for row in rows:
            if row["condition"] == "patchp1":
                row["native_action"] = "UP"
        summary = analyze(rows, specs, family)
        self.assertEqual(summary["primary_components"][0]["statistical_verdict"], "supported")
        self.assertEqual(summary["candidate_conjunctions"][0]["status"], "not_supported")

    def test_invalid_precedes_pending_and_keeps_both_slots(self):
        family = registry(1, 2)
        rows = rows_for(family, "A")
        rows[0].update(generation_status="technical_error", status="error", native_action=None)
        summary = analyze(rows, [comparison(family, "A", "p0")], family)
        candidate = summary["candidate_conjunctions"][0]
        self.assertEqual(candidate["status"], "invalid")
        self.assertEqual(candidate["invalid_primary_ids"], ["p0"])
        self.assertEqual(candidate["pending_primary_ids"], ["p1"])
        self.assertEqual([c["primary"]["multiplicity_input_p_value"] for c in summary["primary_components"]], [1, 1])
        self.assertIsNone(summary["primary_components"][0]["primary"]["adjusted_ci"])

    def test_incomplete_component_skips_resampling_and_keeps_placeholder(self):
        family = registry(1, 1)
        with patch.object(analysis, "_centered_bootstrap_p", side_effect=AssertionError("partial inference")):
            summary = analyze(rows_for(family, "A", n=16), [comparison(family, "A", "p0")], family)
        component = summary["primary_components"][0]
        self.assertEqual(component["component_status"], "pending")
        self.assertIsNone(component["primary"]["adjusted_ci"])
        self.assertIsNone(component["primary"]["p_value"])
        self.assertIsNone(component["primary"]["effect"])

    def test_eventual_action_gate_stays_separate_from_itt(self):
        family = registry(1, 1)
        rows = rows_for(family, "A")
        for row in rows:
            if row["condition"] == "baseline":
                row.update(generation_status="truncated", native_action=None)
        result = analyze(rows, [comparison(family, "A", "p0")], family)
        candidate = result["candidate_conjunctions"][0]
        self.assertTrue(candidate["supported"])
        self.assertFalse(candidate["eventual_action_robust_to_censoring"])
        self.assertFalse(candidate["headline_semantic_support"])

    def test_legacy_output_equals_archived_original(self):
        path = STAGE / "co_primary_baseline_v1/src/activation_causality/analysis.py"
        spec = importlib.util.spec_from_file_location("original_analysis", path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        family = registry(1, 1)
        rows = rows_for(family, "A")
        old = {key: value for key, value in comparison(family, "A", "p0").items()
               if key not in {"candidate_id", "primary_id"}}
        kwargs = {"comparisons": [old], "n_bootstrap": 200, "n_permutations": 200}
        self.assertEqual(analysis.analyze_results(rows, **kwargs), baseline.analyze_results(rows, **kwargs))
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "new", Path(directory) / "old"
            analysis.write_report(rows, a, make_figure=False, **kwargs)
            baseline.write_report(rows, b, make_figure=False, **kwargs)
            self.assertEqual((a / "report.md").read_bytes(), (b / "report.md").read_bytes())

    def test_report_contains_registry_pending_and_conjunction(self):
        family = registry()
        with tempfile.TemporaryDirectory() as directory:
            result = analysis.write_report([], directory, make_figure=False, comparisons=[],
                                            confirmation_primary_family=family, n_bootstrap=200, n_permutations=200)
            text = (Path(directory) / "report.md").read_text()
            saved = json.loads((Path(directory) / "summary.json").read_text())
            self.assertIn("All 4 frozen components", text)
            self.assertIn("A/p0", text)
            self.assertIn("pending", text)
            self.assertEqual(saved, result)

    def test_mixture_reuses_existing_stratification_and_exact_weight_binding(self):
        family = registry(1, 1)
        bound = family["bindings"]["A"]["p0"]
        quotas = {"wall": 11, "key_possession": 11, "door_status": 10}
        bound.update(families=list(quotas), family_layout_counts=quotas,
                     family_weights={name: count / 32 for name, count in quotas.items()},
                     expected_n_layouts_by_family=quotas.copy(), expected_n_pairs_by_family=quotas.copy(),
                     resampling_scheme="within_family_layout_bootstrap", stratification_key="family")
        spec = comparison(family, "A", "p0")
        rows = rows_for(family, "A")
        for row in rows:
            index = int(row["layout_id"].split("-")[-1])
            row["family"] = "wall" if index < 11 else "key_possession" if index < 22 else "door_status"
        new = analyze(rows, [spec], family)["primary_components"][0]
        old = analysis.analyze_results(rows, comparisons=[spec], n_bootstrap=200, n_permutations=200)["comparisons"][0]
        for field in ("family_stratification", "completion_sensitive_effect_bounds", "secondary_native_by_family"):
            self.assertEqual(new[field], old[field])
        for field in ("effect", "adjusted_ci", "p_value", "p_value_method"):
            self.assertEqual(new["primary"][field], old["primary"][field])
        changed = deepcopy(spec)
        changed["family_weights"]["wall"] += .01
        with self.assertRaisesRegex(ValueError, "binding"):
            analyze(rows, [changed], family)
        missing = deepcopy(family)
        del missing["bindings"]["A"]["p0"]["stratification_key"]
        with self.assertRaisesRegex(ValueError, "stratification"):
            validate_primary_family(missing)

    def test_new_origin_is_explicit_and_metadata_does_not_invent_source_outcomes(self):
        family = registry(1, 1)
        bound = family["bindings"]["A"]["p0"]
        bound["origin"] = {"kind": "prospective_confirmation_control",
                           "provenance": {"reviewed_addendum_sha256": "e" * 64, "discovery_measured": False}}
        result = analyze([], [], family)
        self.assertEqual(result["primary_components"][0]["binding"]["origin"], bound["origin"])
        self.assertIsNone(result["primary_components"][0]["primary"]["effect"])
        wrong = deepcopy(family)
        wrong["bindings"]["A"]["p0"]["origin"]["kind"] = "arbitrary_new_control"
        with self.assertRaisesRegex(ValueError, "origin"):
            validate_primary_family(wrong)

    def test_registry_fingerprint_changes_when_binding_or_slot_changes(self):
        family = registry()
        original = analyze([], [], family)
        changed = deepcopy(family)
        changed["bindings"]["A"]["p0"]["control"] = "new_control"
        next_report = analyze([], [], changed)
        self.assertNotEqual(original["confirmation_primary_family_sha256"], next_report["confirmation_primary_family_sha256"])
        self.assertEqual(original["confirmation_primary_family"], family)
        smaller = registry(1, 2)
        self.assertNotEqual(original["confirmation_primary_family_sha256"], analyze([], [], smaller)["confirmation_primary_family_sha256"])

    def test_descriptive_identity_cannot_inherit_primary_status(self):
        family = registry(1, 1)
        spec = comparison(family, "A", "p0")
        diagnostic = {"id": "identity", "experiment_id": "experimentA", "treatment": "patchp0", "control": "baseline",
                      "partition": "confirmation", "confirmatory": False, "predicted_direction": "zero"}
        result = analyze(rows_for(family, "A"), [spec, diagnostic], family)
        self.assertEqual(len(result["primary_components"]), 1)
        self.assertFalse(result["comparisons"][1]["explicit_confirmation"])
        self.assertEqual(result["comparisons"][1]["predicted_direction"], "zero")
        self.assertEqual(result["analysis_config"]["confirmation_primary_family_size"], 1)


if __name__ == "__main__":
    unittest.main()
