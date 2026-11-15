"""Synthetic complete-control and multiplicity-boundary checks; no outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from src.activation_causality.confirmation_contract import (  # noqa: E402
    ENDPOINT, FAMILIES, PROTOCOLS, UNRESOLVED_POLICY, analysis_options, family_quotas,
    inference_cutoff, projected_jobs, remaining_inference_seconds, validate_budget,
    validate_primary_spec, validate_registry_configurations,
)
from src.activation_causality.registry import fingerprint  # noqa: E402


def fixture(protocol="input_fact_subspace", n=32):
    scope, count, primary_ids = PROTOCOLS[protocol]
    candidate_id = protocol + "_candidate"
    inventory = ["transfer", "baseline", "specificity"] + ["control_" + str(i) for i in range(count - 3)]
    config = {"candidate_id": candidate_id, "protocol_kind": protocol, "n_layouts": n,
              "generation_seeds": [42], "max_new_tokens": 4096, "temperature": 0.7,
              "claim_scope": scope, "semantic_headline_eligible": False, "condition_inventory": inventory}
    if protocol == "contextual_input":
        counts = family_quotas(n)
        config.update(families=list(FAMILIES), family_layout_counts=counts,
                      family_weights={key: value / n for key, value in counts.items()})
    else:
        config["family"] = "wall"
    bindings = {}
    for primary, control in zip(primary_ids, ("baseline", "specificity")):
        binding = {"treatment": "transfer", "control": control, "predicted_direction": "increase",
                   "primary_endpoint": ENDPOINT, "target_action_basis": "unique_simulator_counterfactual_optimum",
                   "native_unresolved_policy": UNRESOLVED_POLICY, "expected_n_layouts": n, "expected_n_pairs": n,
                   "eligibility_rule": {"identifier": "synthetic_only", "sha256": "a" * 64},
                   "origin": {"kind": "existing_source_comparison", "provenance": {"synthetic": True}},
                   "claim_scope": scope, "semantic_headline_eligible": False}
        if protocol == "contextual_input":
            binding.update({key: deepcopy(config[key]) for key in ("families", "family_layout_counts", "family_weights")})
            binding.update(expected_n_layouts_by_family=deepcopy(counts), expected_n_pairs_by_family=deepcopy(counts),
                           resampling_scheme="within_family_layout_bootstrap", stratification_key="family")
        bindings[primary] = binding
    config["primary_comparisons"] = bindings
    registry = {"schema_version": 1, "family_size": 2, "candidates": {candidate_id: list(primary_ids)},
                "bindings": {candidate_id: deepcopy(bindings)}}
    jobs = []
    layout_families = ([f for f in FAMILIES for _ in range(counts[f])]
                       if protocol == "contextual_input" else ["wall"] * n)
    for index, variable in enumerate(layout_families):
        for condition in inventory:
            job = {"experiment_id": "synthetic_primary", "partition": "confirmation", "pair_id": "pair_" + str(index),
                   "layout_id": "layout_" + str(index), "family": variable, "condition": condition,
                   "natural": True, "seed": 42, "max_new_tokens": 4096, "temperature": 0.7,
                   "claim_scope": scope, "semantic_headline_eligible": False}
            job["job_id"] = fingerprint(job)[:24]
            jobs.append(job)
    comparisons = [{**deepcopy(binding), "id": primary, "primary_id": primary, "candidate_id": candidate_id,
                    "experiment_id": "synthetic_primary", "partition": "confirmation", "confirmatory": True,
                    "eligible_pair_ids": ["pair_" + str(index) for index in range(n)]}
                   for primary, binding in bindings.items()]
    comparisons.append({key: deepcopy(value) for key, value in comparisons[0].items()
                        if key not in {"id", "primary_id"}} | {"id": "diagnostic", "confirmatory": False,
                                                              "treatment": inventory[-1]})
    spec = {"experiment_id": "synthetic_primary", "partition": "confirmation", "candidate_id": candidate_id,
            "protocol_kind": protocol, "claim_scope": scope, "semantic_headline_eligible": False,
            "jobs": jobs, "comparisons": comparisons}
    return config, registry, spec


def rekey(job):
    job["job_id"] = fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24]


class ConfirmationContractTests(unittest.TestCase):
    def test_supported_complete_populations_and_two_question_registry(self):
        configurations, combined = {}, {"schema_version": 1, "family_size": 4, "candidates": {}, "bindings": {}}
        for protocol in PROTOCOLS:
            config, registry, spec = fixture(protocol)
            self.assertEqual(validate_registry_configurations(registry, {config["candidate_id"]: config}), registry)
            validate_primary_spec(spec, config, registry)
            configurations[config["candidate_id"]] = config
            for key in ("candidates", "bindings"):
                combined[key].update(registry[key])
        self.assertEqual(validate_registry_configurations(combined, configurations), combined)

    def test_exact_balanced_quotas(self):
        self.assertEqual(family_quotas(32), dict(wall=11, key_possession=11, door_status=10))
        self.assertEqual(family_quotas(64), dict(wall=22, key_possession=21, door_status=21))
        self.assertEqual(family_quotas(128), dict(wall=43, key_possession=43, door_status=42))
        for bad in (True, 31, 33, 64.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                family_quotas(bad)

    def test_registry_binding_mutation_and_local_alpha_override_rejected(self):
        config, registry, _ = fixture()
        for change in ({"family_size": 1}, {"confirmation_family_size": 1},
                       {"generation_seeds": [True]}, {"max_new_tokens": 2048},
                       {"semantic_headline_eligible": True}, {"candidate_id": "renamed"}):
            bad = {**deepcopy(config), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_registry_configurations(registry, {config["candidate_id"]: bad})
        bad = deepcopy(config)
        bad["primary_comparisons"]["subspace_efficacy"]["control"] = "specificity"
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            validate_registry_configurations(registry, {config["candidate_id"]: bad})

    def test_no_redundant_profile_backfill(self):
        config, registry, _ = fixture()
        second = deepcopy(config)
        second["candidate_id"] = "second_rank"
        registry["candidates"]["second_rank"] = list(registry["candidates"][config["candidate_id"]])
        registry["bindings"]["second_rank"] = deepcopy(second["primary_comparisons"])
        registry["family_size"] = 4
        with self.assertRaisesRegex(ValueError, "one candidate per"):
            validate_registry_configurations(registry, {config["candidate_id"]: config, "second_rank": second})

    def test_common_size_required_across_questions(self):
        one, r1, _ = fixture(n=32)
        two, r2, _ = fixture("contextual_input", n=64)
        for key in ("candidates", "bindings"):
            r1[key].update(r2[key])
        r1["family_size"] = 4
        with self.assertRaisesRegex(ValueError, "same prospectively"):
            validate_registry_configurations(r1, {one["candidate_id"]: one, two["candidate_id"]: two})

    def test_missing_unselected_control_blocks_complete_primary(self):
        config, registry, spec = fixture()
        spec["jobs"] = [j for j in spec["jobs"] if not (j["pair_id"] == "pair_0" and j["condition"] == "control_10")]
        with self.assertRaisesRegex(ValueError, "each planned condition"):
            validate_primary_spec(spec, config, registry)

    def test_partial_layout_inventory_is_not_confirmation_ready(self):
        config, registry, spec = fixture()
        spec["jobs"] = [j for j in spec["jobs"] if j["pair_id"] != "pair_31"]
        with self.assertRaisesRegex(ValueError, "every frozen layout"):
            validate_primary_spec(spec, config, registry)

    def test_canonical_identity_and_native_parameters_are_checked(self):
        config, registry, original = fixture()
        for field, value in (("seed", True), ("natural", False), ("max_new_tokens", 2048),
                             ("temperature", 0), ("semantic_headline_eligible", True)):
            spec = deepcopy(original)
            spec["jobs"][0][field] = value
            rekey(spec["jobs"][0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_primary_spec(spec, config, registry)
        original["jobs"][0]["job_id"] = "not_the_digest"
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            validate_primary_spec(original, config, registry)

    def test_pair_layout_family_aliasing_and_wrong_mixture_rejected(self):
        config, registry, original = fixture("contextual_input")
        for field, value in (("layout_id", "layout_1"), ("pair_id", "pair_1"), ("family", "door_status")):
            spec = deepcopy(original)
            spec["jobs"][0][field] = value
            rekey(spec["jobs"][0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_primary_spec(spec, config, registry)
        spec = deepcopy(original)
        for job in spec["jobs"]:
            if job["pair_id"] == "pair_0":
                job["family"] = "door_status"
                rekey(job)
        with self.assertRaisesRegex(ValueError, "family quotas"):
            validate_primary_spec(spec, config, registry)

    def test_frozen_primary_endpoints_counts_and_origins_cannot_drift(self):
        config, registry, original = fixture()
        mutations = ({"control": "specificity"}, {"expected_n_layouts": 31},
                     {"primary_endpoint": "forced_action"}, {"primary_id": "unregistered"},
                     {"origin": {"kind": "existing_source_comparison", "provenance": {"other": True}}})
        for mutation in mutations:
            spec = deepcopy(original)
            spec["comparisons"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_primary_spec(spec, config, registry)

    def test_no_selected_subset_or_duplicate_primary(self):
        config, registry, original = fixture()
        spec = deepcopy(original)
        spec["comparisons"][0]["eligible_pair_ids"][-1] = "pair_0"
        with self.assertRaisesRegex(ValueError, "exclude frozen layouts"):
            validate_primary_spec(spec, config, registry)
        for mode in ("missing", "duplicate", "descriptive_primary", "role_missing"):
            spec = deepcopy(original)
            if mode == "missing":
                spec["comparisons"].pop(0)
            elif mode == "duplicate":
                spec["comparisons"].append({**deepcopy(spec["comparisons"][0]), "id": "duplicate"})
            elif mode == "descriptive_primary":
                spec["comparisons"][-1]["primary_id"] = "subspace_efficacy"
            else:
                del spec["comparisons"][-1]["confirmatory"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_primary_spec(spec, config, registry)

    def test_reports_use_full_family_and_keep_legacy_option(self):
        one, registry, spec = fixture()
        two, second, _ = fixture("contextual_input")
        for key in ("candidates", "bindings"):
            registry[key].update(second[key])
        registry["family_size"] = 4
        freeze = {"candidate_configurations": {one["candidate_id"]: one, two["candidate_id"]: two},
                  "confirmation_primary_family": registry}
        self.assertEqual(analysis_options(freeze, spec), {"confirmation_primary_family": registry})
        self.assertEqual(analysis_options({"candidate_configurations": {"a": {}, "b": {}}}, {}),
                         {"confirmation_family_size": 2})
        spec["cohort_role"] = "lexical_supplement"
        with self.assertRaisesRegex(ValueError, "descriptive paired"):
            analysis_options(freeze, spec)

    def test_json_type_drift_is_not_exact_binding_equality(self):
        config, registry, spec = fixture()
        changed = deepcopy(config)
        changed["primary_comparisons"]["subspace_efficacy"]["origin"]["provenance"]["synthetic"] = 1
        self.assertNotEqual(fingerprint(config["primary_comparisons"]), fingerprint(changed["primary_comparisons"]))
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            validate_registry_configurations(registry, {config["candidate_id"]: changed})
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            validate_primary_spec(spec, changed, registry)
        for key, value in (("expected_n_layouts", 32.0), ("expected_n_pairs", 32.0),
                           ("origin", {"kind": "existing_source_comparison", "provenance": {"synthetic": 1}})):
            changed_spec = deepcopy(spec)
            changed_spec["comparisons"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact frozen binding"):
                validate_primary_spec(changed_spec, config, registry)

    def test_missing_registry_cannot_downgrade_new_mode_to_legacy_multiplicity(self):
        one, _, spec = fixture()
        two, _, _ = fixture("contextual_input")
        for configs in ({one["candidate_id"]: one}, {one["candidate_id"]: one, two["candidate_id"]: two},
                        {"unknown": {"primary_comparisons": {}}}):
            with self.subTest(candidates=len(configs)), self.assertRaisesRegex(ValueError, "global registry"):
                analysis_options({"candidate_configurations": configs}, spec)

    def test_absolute_cutoff_and_resume_clock(self):
        deadline = "2026-09-15T08:24:34.513516+00:00"
        cutoff = "2026-09-15T02:24:34.513516+00:00"
        self.assertEqual(inference_cutoff(deadline), cutoff)
        self.assertEqual(remaining_inference_seconds(deadline, cutoff, now="2026-09-15T02:24:33.513516+00:00"), 1)
        self.assertEqual(remaining_inference_seconds(deadline, cutoff, now="2026-09-15T02:24:35.513516+00:00"), 0)
        self.assertEqual(remaining_inference_seconds(deadline, cutoff, now="2026-09-15T03:24:33.513516+01:00"), 1)
        for wrong in (deadline, "2026-09-15T02:24:34.513516"):
            with self.assertRaises(ValueError):
                remaining_inference_seconds(deadline, wrong, now=cutoff)
        with self.assertRaises(ValueError):
            inference_cutoff("2026-09-15T08:24:34")

    def test_budget_includes_every_control_supplement_and_overhead(self):
        one, _, _ = fixture(n=128)
        two, _, _ = fixture("contextual_input", n=128)
        configs = {one["candidate_id"]: one, two["candidate_id"]: two}
        supplements = dict.fromkeys(configs, True)
        self.assertEqual(projected_jobs(configs, supplements), (128 + 32) * 29)
        self.assertEqual(projected_jobs(configs, dict.fromkeys(configs, False)), 128 * 29)
        exact = (128 + 32) * 29 * 40 + 7200
        self.assertEqual(validate_budget(configs, supplements, 40, exact)["projected_inference_and_overhead_seconds"], exact)
        with self.assertRaisesRegex(ValueError, "whole confirmation"):
            validate_budget(configs, supplements, 40, exact - 1)
        for wrong in ({one["candidate_id"]: True}, dict.fromkeys(configs, 1)):
            with self.assertRaises(ValueError):
                projected_jobs(configs, wrong)
        for bad in (True, 0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                validate_budget(configs, supplements, bad, exact)


if __name__ == "__main__":
    unittest.main()
