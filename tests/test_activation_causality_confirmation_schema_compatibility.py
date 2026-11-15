"""Regression checks for frozen contextual control semantics in both readers."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reader(number, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{number}_{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = reader("40", "activation_causality_confirmation_report")
technical = reader("41", "activation_causality_confirmation_technical_audit")


def contextual_fixture():
    names = ["baseline", "counterfactual_text", "layer10_identity", "layer10_restoration_equivalent",
             "layer10_natural", "layer10_natural_norm_random", "layer10_random"]
    jobs, rows = [], []
    for name in names:
        paired = "counterfactual_text" if name == "layer10_restoration_equivalent" else "baseline"
        counterfactual = name in {"counterfactual_text", "layer10_restoration_equivalent"}
        jobs.append({"pair_id": "pair", "seed": 42, "condition": name,
                     "recipient_token_ids": [20 if counterfactual else 10],
                     "metadata": {"paired_baseline_condition": paired}})
        rows.append({"pair_id": "pair", "seed": 42, "condition": name, "status": "complete",
                     "patch_log": [{"delta_l2": [0.]}], "forced_q": {"DOWN" if counterfactual else "UP": 1.},
                     "forced_metrics": {"branch": "counterfactual" if counterfactual else "recipient"}, "generated_token_ids": [2 if counterfactual else 1],
                     "generated_text": "counterfactual" if counterfactual else "recipient",
                     "generation_status": "complete", "stop_reason": "return",
                     "native_action": "DOWN" if counterfactual else "UP"})
    by_name = {r["condition"]: r for r in rows}
    reference = {"compiled_donor_float32_sha256": "a" * 64, "original_reference_donor_float32_sha256": "a" * 64,
                 "expected_recipient_float32_sha256": "c" * 64, "expected_delta_l2": [2.], "positions": [1]}
    for name in ("layer10_natural", "layer10_natural_norm_random"):
        by_name[name]["compiled_full_write_audits"] = [deepcopy(reference)]
        by_name[name]["patch_log"] = [{"delta_l2": [2.]}]
    by_name["layer10_natural_norm_random"]["compiled_full_write_audits"][0]["compiled_donor_float32_sha256"] = "b" * 64
    orientation = {"raw_gaussian_float32_sha256": "d" * 64, "recipient_float32_sha256": "c" * 64,
                   "reference_donor_float32_sha256": "a" * 64}
    for name in ("layer10_random", "layer10_natural_norm_random"):
        by_name[name]["random_orientation_audits"] = [deepcopy(orientation)]
    spec = {"protocol_kind": "contextual_input", "candidate_id": "contextual", "frozen_configuration": {"layer": 10}, "jobs": jobs}
    audit = technical.Audit(ROOT, ROOT / "outputs/activation_causality/campaign_20260912", "synthetic")
    audit.spec = spec
    audit.frozen = {"candidate_configurations": {"contextual": {"layer": 10}}}
    return spec, rows, audit


class FrozenContextualSchema(unittest.TestCase):
    def test_distinct_compiled_donors_and_counterfactual_identity_pass(self):
        spec, rows, audit = contextual_fixture()
        checks = audit.controls(rows)
        self.assertEqual({c["condition"]: c["baseline_condition"] for c in checks},
                         {"layer10_identity": "baseline", "layer10_restoration_equivalent": "counterfactual_text"})
        self.assertEqual(report._cohort_controls(spec, rows), [])

    def test_original_reference_norm_and_geometry_are_not_discarded(self):
        for key, value in (("original_reference_donor_float32_sha256", "e" * 64),
                           ("expected_recipient_float32_sha256", "e" * 64), ("expected_delta_l2", [3.]), ("positions", [2])):
            with self.subTest(key=key):
                spec, rows, audit = contextual_fixture()
                rows[5]["compiled_full_write_audits"][0][key] = value
                with self.assertRaisesRegex(ValueError, "original norm reference"):
                    audit.controls(rows)
                self.assertIn("natural_and_random_norm_references_differ", [f["reason"] for f in report._cohort_controls(spec, rows)])

    def test_recipient_replay_cannot_substitute_for_counterfactual_restoration(self):
        spec, rows, audit = contextual_fixture()
        for key in ("forced_q", "forced_metrics", "generated_token_ids", "generated_text", "native_action"):
            rows[3][key] = deepcopy(rows[0][key])
        with self.assertRaisesRegex(ValueError, "whole-continuation"):
            audit.controls(rows)
        self.assertIn("identity_is_not_exact_replay:layer10_restoration_equivalent",
                      [f["reason"] for f in report._cohort_controls(spec, rows)])

    def test_frozen_baseline_declaration_and_input_ids_are_required(self):
        for change in ("declaration", "tokens"):
            with self.subTest(change=change):
                spec, rows, audit = contextual_fixture()
                job = spec["jobs"][3]
                if change == "declaration":
                    job["metadata"]["paired_baseline_condition"] = "baseline"
                else:
                    job["recipient_token_ids"] = [10]
                for check in (lambda: audit.controls(rows), lambda: report._cohort_controls(spec, rows)):
                    with self.assertRaisesRegex(ValueError, "frozen baseline/input"):
                        check()


if __name__ == "__main__":
    unittest.main()
