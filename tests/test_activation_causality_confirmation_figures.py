"""Synthetic final-report fixtures only; no raw data or analysis calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/42_activation_causality_confirmation_figures.py"
SPEC = importlib.util.spec_from_file_location("confirmation_figures", SOURCE)
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)

CONTEXT = ["baseline", "counterfactual_text", "donor_text", "natural_donor_text", "same_fact_donor_text", "irrelevant_donor_text",
           "layer10_matched", "layer10_natural", "layer10_same_fact", "layer10_irrelevant", "layer10_random", "layer10_reverse",
           "layer10_identity", "layer10_restoration_equivalent", "layer10_natural_norm_random"]
SUBSPACE = ["baseline", "donor_text", "full_transfer", "subspace_identity", "subspace_removal", "restoration_equivalent",
            "subspace_forward_0p5", "subspace_reverse_0p5", "random_subspace_0p5", "projected_norm_matched_random_subspace_0p5",
            "subspace_forward_1", "subspace_reverse_1", "random_subspace_1", "projected_norm_matched_random_subspace_1"]


def fixture(*, incomplete=False):
    """Artificial values labelled at the source and in every rendered figure."""
    definitions = {"contextual_l10": [("contextual_fact_contrast", "layer10_natural", "layer10_same_fact"),
                                       ("natural_norm_direction_contrast", "layer10_natural", "layer10_natural_norm_random")],
                   "wall_subspace": [("subspace_efficacy", "subspace_forward_1", "baseline"),
                                     ("subspace_direction_contrast", "subspace_forward_1", "projected_norm_matched_random_subspace_1")]}
    candidates = {cid: [x[0] for x in values] for cid, values in definitions.items()}
    bindings = {cid: {pid: {"treatment": t, "control": c} for pid, t, c in values} for cid, values in definitions.items()}
    components, metadata, counts, inventories = [], {}, {}, {}
    points = [.28, .19, .13, .03]
    cis = [[.10, .43], [-.02, .38], [-.05, .29], [-.16, .18]]
    bounds = [[.18, .31], [.08, .27], [.09, .15], [-.02, .09]]
    for cid, values in definitions.items():
        pending = incomplete and cid == "wall_subspace"
        inventories[cid] = {"whole_cohort_status": "invalid" if pending else "complete", "expected_n_layouts": 128}
        for pid, treatment, control in values:
            i = len(components)
            components.append({"candidate_id": cid, "primary_id": pid, "component_status": "invalid" if pending else "complete",
                "binding": bindings[cid][pid], "primary": {"effect": points[i], "adjusted_ci": cis[i]},
                "completion_sensitive_effect_bounds": {"lower": bounds[i][0], "upper": bounds[i][1]},
                "statistical_verdict": "synthetic_fixture_not_evaluated"})
        for role, n in (("primary", 128), ("lexical_supplement", 32)):
            suffix = "primary" if role == "primary" else "alternate"
            eid = "synthetic_" + cid + "_" + suffix
            order = CONTEXT if cid == "contextual_l10" else SUBSPACE
            quota = {"wall": 43, "key_possession": 43, "door_status": 42} if n == 128 else {"wall": 11, "key_possession": 11, "door_status": 10}
            if cid == "wall_subspace":
                quota = {"wall": n}
            metadata[eid] = {"candidate_id": cid, "cohort_role": role, "parent_experiment_id": None if role == "primary" else "synthetic_" + cid + "_primary",
                "condition_order": list(order), "expected_n_layouts": n, "expected_n_pairs": n, "expected_jobs": n*len(order),
                "expected_condition_trials": {condition: n for condition in order}, "native_seeds": [42], "max_new_tokens": 4096,
                "temperatures": [.7], "semantic_headline_eligible": False, "family_layout_counts": quota,
                "family_weights": {f: v/n for f, v in quota.items()}, "renderer_id": "canonical_v1" if role == "primary" else "alternate_w_dot_l_yes_no_v1",
                "additional_layouts_beyond_parent": n if role == "primary" else 0,
                "claim_scope": "synthetic_fixture_input_routing", "layer": 10 if cid == "contextual_l10" else 0, "rank": None if cid == "contextual_l10" else 1}
            table = {}
            for index, condition in enumerate(order):
                index = 0 if "identity" in condition or "restoration" in condition else index
                by_family = {}
                for family, k in quota.items():
                    target = min(k-2, k//4 + (index % 5)*max(1, k//10))
                    unresolved, third = 1, max(0, k//8)
                    original = k-target-unresolved-third
                    by_family[family] = {"n":k, "n_layouts":k, "n_pairs":k, "target":target, "original":original,
                        "third_action":third, "unresolved":unresolved, "generation_status_counts":{"complete":k-1,"truncated":1}}
                table[condition] = {"n":n,"n_layouts":n,"n_pairs":n, **{key:sum(v[key] for v in by_family.values()) for key in figures.CATEGORIES},
                                    "generation_status_counts":{"complete":n-len(quota),"truncated":len(quota)},"by_family":by_family}
            counts[eid] = None if pending else table
    return {"format": figures.FORMAT, "freeze_id":"synthetic_fixture", "synthetic_fixture":True,
        "mode":"at_cutoff" if incomplete else "complete_only", "provenance":{"script_sha256":"a"*64,
            "input_file_sha256":{"/do/not/open/raw.json":"b"*64}},
        "primary":{"confirmation_primary_family":{"schema_version":1,"family_size":4,"candidates":candidates,"bindings":bindings},
            "analysis_config":{"version":4,"confirmation_primary_family_size":4,"minimum_effect":.10},
            "primary_components":components,"candidate_conjunctions":[{"candidate_id":c,"status":"synthetic_not_evaluated"} for c in candidates]},
        "whole_candidate_inventories":inventories,"cohort_display_metadata":metadata,"complete_cohort_condition_counts":counts}


class DisplayGuards(unittest.TestCase):
    def test_exact_values_copied_without_statistics(self):
        report = fixture()
        view = figures.prepare(report, synthetic=True)
        self.assertEqual(len(view["primary_effects"]), 4)
        for got, source in zip(view["primary_effects"], report["primary"]["primary_components"], strict=True):
            self.assertEqual(got["effect"], source["primary"]["effect"])
            self.assertEqual(got["adjusted_ci"], source["primary"]["adjusted_ci"])
        self.assertEqual([len(c["metadata"]["condition_order"]) for c in view["cohorts"]], [15,14,15,14])

    def test_invalid_slots_are_unestimated_despite_retained_diagnostics(self):
        view = figures.prepare(fixture(incomplete=True), synthetic=True)
        self.assertEqual([r["effect"] for r in view["primary_effects"]][2:], [None,None])
        self.assertIsNone(view["cohorts"][1]["counts"])
        self.assertEqual(len(view["cohorts"][1]["metadata"]["condition_order"]), 14)

    def test_synthetic_flag_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,"synthetic fixture flag"):
            figures.prepare(fixture())

    def test_missing_or_duplicate_slot_rejected(self):
        report=fixture()
        report["primary"]["primary_components"][-1]=deepcopy(report["primary"]["primary_components"][0])
        with self.assertRaisesRegex(ValueError,"primary slots"):
            figures.prepare(report,synthetic=True)

    def test_complete_estimate_cannot_cross_invalid_whole_cohort(self):
        report=fixture()
        report["whole_candidate_inventories"]["contextual_l10"]["whole_cohort_status"]="invalid"
        with self.assertRaisesRegex(ValueError,"whole cohort"):
            figures.prepare(report,synthetic=True)

    def test_wrong_count_and_omitted_control_rejected(self):
        report=fixture()
        table=next(iter(report["complete_cohort_condition_counts"].values()))
        table["baseline"]["target"]+=1
        with self.assertRaisesRegex(ValueError,"categories/denominator"):
            figures.prepare(report,synthetic=True)
        table["baseline"]["target"]-=1
        table.pop("baseline")
        with self.assertRaisesRegex(ValueError,"all controls"):
            figures.prepare(report,synthetic=True)

    def test_alternate_is_exact_same32_and_parent_required(self):
        report=fixture()
        alternate=next(v for v in report["cohort_display_metadata"].values() if v["cohort_role"]=="lexical_supplement")
        alternate["expected_n_layouts"]=64
        with self.assertRaisesRegex(ValueError,"same32"):
            figures.prepare(report,synthetic=True)

    def test_ci_and_sensitivity_bounds_are_separate_fields(self):
        report=fixture()
        view=figures.prepare(report,synthetic=True)
        self.assertNotEqual(view["primary_effects"][0]["adjusted_ci"],view["primary_effects"][0]["completion_bounds"])
        report["primary"]["primary_components"][0]["primary"]["adjusted_ci"]=[float("nan"),.4]
        with self.assertRaisesRegex(ValueError,"confidence interval"):
            figures.prepare(report,synthetic=True)

    def test_category_family_sums_and_statuses_checked(self):
        report=fixture()
        row=next(iter(report["complete_cohort_condition_counts"].values()))["baseline"]
        row["generation_status_counts"]["malformed"]=1
        with self.assertRaisesRegex(ValueError,"unresolved/status"):
            figures.prepare(report,synthetic=True)

    def test_binding_and_family_denominator_mutations_rejected(self):
        report = fixture()
        report["primary"]["primary_components"][0]["binding"] = {"treatment": "changed", "control": "baseline"}
        with self.assertRaisesRegex(ValueError, "binding differs"):
            figures.prepare(report, synthetic=True)
        report = fixture()
        row = next(iter(report["complete_cohort_condition_counts"].values()))["baseline"]
        row["by_family"]["wall"]["generation_status_counts"]["truncated"] = 0
        with self.assertRaisesRegex(ValueError, "per-family native status"):
            figures.prepare(report, synthetic=True)

    def test_invalid_primary_cannot_supply_condition_bars(self):
        report = fixture(incomplete=True)
        eid = "synthetic_wall_subspace_primary"
        report["complete_cohort_condition_counts"][eid] = fixture()["complete_cohort_condition_counts"][eid]
        with self.assertRaisesRegex(ValueError, "counts from incomplete/invalid"):
            figures.prepare(report, synthetic=True)

    def test_strict_json_and_output_guard_precede_render(self):
        for raw in ('{"x":1,"x":2}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                figures.strict(raw)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory).resolve()
            source = root / "report.json"
            source.write_text(json.dumps(fixture()))
            with self.assertRaisesRegex(ValueError,"protected input tree"):
                figures.render(source,root/"results/figure",synthetic=True)

    def test_render_reads_only_report_and_self_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory).resolve()
            source = root / "report.json"
            raw = json.dumps(fixture()).encode()
            source.write_bytes(raw)
            original=Path.read_bytes
            def guarded(path):
                self.assertIn(path,{source,SOURCE})
                return original(path)
            with patch.object(Path,"read_bytes",guarded):
                outputs=figures.render(source,root/"synthetic_figures",synthetic=True)
            self.assertEqual(len(outputs),5)
            for name, digest in outputs.items():
                self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)
            counts=json.loads((root/"synthetic_figures.counts.json").read_bytes())
            self.assertEqual(counts["source_report"]["sha256"],hashlib.sha256(raw).hexdigest())
            self.assertFalse(counts["statistical_recomputation"])
            with self.assertRaisesRegex(ValueError,"already exists"):
                figures.render(source,root/"synthetic_figures",synthetic=True)


if __name__ == "__main__":
    unittest.main()
