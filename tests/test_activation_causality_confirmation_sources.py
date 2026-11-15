"""Source-gate checks with synthetic audit fixtures, never live outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from src.activation_causality.confirmation_sources import (  # noqa: E402
    CONTEXTUAL_SOURCE, REQUIRED_BATTERY, SUBSPACE_SOURCES, existing_comparison_origin,
    prospective_natural_norm_origin, read_closed_native_battery, read_complete_native_sources,
)
from src.activation_causality.registry import atomic_json, file_sha256, fingerprint, read_json  # noqa: E402
from src.activation_causality import confirmation_sources as sources_module  # noqa: E402


class SourceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.campaign = self.repo / "outputs/activation_causality/campaign_20260912"

    def create(self, identifier=SUBSPACE_SOURCES[0], seconds=20):
        contextual = identifier == CONTEXTUAL_SOURCE
        n_pairs, n_conditions = (6, 22) if contextual else (4, 14)
        jobs = []
        for index in range(n_pairs):
            for condition in range(n_conditions):
                job = {"experiment_id": identifier, "partition": "discovery", "pair_id": str(index),
                       "layout_id": "layout_" + str(index), "family": "wall", "condition": str(condition),
                       "natural": True, "seed": 42, "max_new_tokens": 4096, "temperature": 0.7,
                       "target_action": "RIGHT", "recipient_token_ids": [1, 2, 3]}
                job["job_id"] = fingerprint(job)[:24]
                jobs.append(job)
        spec = {"experiment_id": identifier, "partition": "discovery", "jobs": jobs,
                "comparisons": [{"id": identifier + "_contrast", "treatment": "1", "control": "0", "predicted_direction": "increase"}]}
        path = self.campaign / "specs" / f"{identifier}.json"
        atomic_json(path, spec)
        digest = file_sha256(path)
        hashes = {}
        for index, job in enumerate(jobs):
            # A cap stays in the throughput and native denominators.
            row = {**job, "spec_sha256": digest, "status": "complete", "seconds": seconds + (10 if index == 0 else 0),
                   "generation_status": "truncated" if index == 0 else "complete", "native_action": None if index == 0 else "UP",
                   "input_token_sha256": fingerprint(job["recipient_token_ids"])}
            result_path = self.campaign / "results" / identifier / f"{job['job_id']}.json"
            atomic_json(result_path, row)
            hashes[str(result_path.relative_to(self.repo))] = file_sha256(result_path)
        if contextual:
            audit = {"experiment_id": identifier, "spec_sha256": digest, "n_results": 132,
                     "technical_status_counts": {"complete": 132}, "exact_forced_replay_input_hashes_patch_logs": 132,
                     "raw_result_hashes": hashes, "preparation_checks": {"exact_pure_rebuild": True,
                         "all_alias_control_and_comparison_inventories_verified": True},
                     "identities": {f"layer{layer}_{role}": {"n": 6, "entire_generated_ids_exact": 6,
                         "forced_q_exact": 6, "max_delta_norm": 0.0}
                         for layer in (0, 10) for role in ("identity", "restoration_equivalent")}}
            name = "contextual_input_native_independent_audit_v1.json"
        else:
            audit = {"format": "subspace_complete_independent_cohort_audit_v1", "input_file_sha256": hashes,
                     "cohort": {"experiment_id": identifier, "spec_sha256": digest, "state": "complete",
                         "exact_complete_filename_inventory": True, "planned_jobs": 56, "saved_file_count": 56,
                         "exact_forced_native_replays": 56, "remaining_planned_jobs": 0,
                         "technical_status_counts": {"complete": 56}, "required_norm_writes_revalidated": 16,
                         "identity_checks": [{"pair_id": str(i), "condition": role, "exact_entire_generated_identity": True,
                             "exact_forced_identity": True} for i in range(4) for role in ("subspace_identity", "restoration_equivalent")],
                         "paired_applied_norm_checks": [{"pair_id": str(i), "dose": dose,
                             "expected_original_reference_identical": True} for i in range(4) for dose in (0.5, 1.0)]}}
            name = identifier.removesuffix("_v1") + "_independent_audit_v1.json"
        audit_path = self.campaign / "reviews" / name
        atomic_json(audit_path, audit)
        return {"path": str(audit_path.relative_to(self.repo)), "sha256": file_sha256(audit_path)}

    def read(self, binding, identifier=SUBSPACE_SOURCES[0]):
        return read_complete_native_sources(self.campaign, self.repo, [identifier], {identifier: binding})[identifier]

    def mutate_audit(self, binding, edit):
        path = self.repo / binding["path"]
        value = read_json(path)
        edit(value)
        atomic_json(path, value)
        return {**binding, "sha256": file_sha256(path)}

    def test_complete_sources_include_every_control_and_capped_duration(self):
        for identifier in (SUBSPACE_SOURCES[0], CONTEXTUAL_SOURCE):
            with self.subTest(identifier=identifier):
                source = self.read(self.create(identifier), identifier)
                self.assertEqual(len(source["source_result_hashes"]), source["n_jobs"])
                self.assertEqual(source["source_result_manifest_sha256"], fingerprint(source["source_result_hashes"]))
                self.assertEqual(source["generation_status_counts"]["truncated"], 1)
                self.assertAlmostEqual(source["mean_seconds_per_job"], 20 + 10 / source["n_jobs"])

    def test_all_inventory_gates_precede_any_result_or_audit_read(self):
        first, second = SUBSPACE_SOURCES[:2]
        bindings = {identifier: self.create(identifier) for identifier in (first, second)}
        next((self.campaign / "results" / second).glob("*.json")).unlink()
        original_text, original_bytes = Path.read_text, Path.read_bytes

        def guard(path):
            if "results" in path.parts or "reviews" in path.parts:
                raise AssertionError("Read outcome or audit before all source inventories closed")

        def no_outcomes_text(path, *args, **kwargs):
            guard(path)
            return original_text(path, *args, **kwargs)

        def no_outcomes_bytes(path):
            guard(path)
            return original_bytes(path)

        with (patch.object(Path, "read_text", no_outcomes_text), patch.object(Path, "read_bytes", no_outcomes_bytes),
              self.assertRaisesRegex(ValueError, "complete source cohort")):
            read_complete_native_sources(self.campaign, self.repo, [first, second], bindings)

    def test_extra_unknown_result_and_retirement_block_sources(self):
        binding = self.create()
        extra = self.campaign / "results" / SUBSPACE_SOURCES[0] / "unplanned.json"
        atomic_json(extra, {})
        with self.assertRaisesRegex(ValueError, "complete source cohort"):
            self.read(binding)
        extra.unlink()
        atomic_json(self.campaign / "reviews" / (SUBSPACE_SOURCES[0] + "_retired.json"), {"retired": True})
        with self.assertRaisesRegex(ValueError, "retired"):
            self.read(binding)

    def test_missing_audit_binding_and_unknown_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventories differ"):
            read_complete_native_sources(self.campaign, self.repo, [SUBSPACE_SOURCES[0]], {})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_complete_native_sources(self.campaign, self.repo, ["other_native"], {"other_native": {}})

    def test_changed_control_or_input_tokens_fail_before_origin(self):
        binding = self.create()
        path = next((self.campaign / "results" / SUBSPACE_SOURCES[0]).glob("*.json"))
        original = read_json(path)
        for key, value in (("condition", "changed"), ("input_token_sha256", "f" * 64), ("status", "error"),
                           ("generation_status", "unknown"), ("native_action", "not_an_action")):
            row = {**original, key: value}
            atomic_json(path, row)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read(binding)
        atomic_json(path, original)
        row = {**original, "seconds": 0.001}
        atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "every native source control"):
            self.read(binding)

    def test_numeric_type_drift_in_immutable_native_fields_is_rejected(self):
        binding = self.create()
        path = next((self.campaign / "results" / SUBSPACE_SOURCES[0]).glob("*.json"))
        row = read_json(path)
        row["seed"] = 42.0
        atomic_json(path, row)
        # Even an audit that honestly hashes these altered bytes cannot make
        # them equal to the integer seed in the immutable registered job.
        binding = self.mutate_audit(binding, lambda a: a["input_file_sha256"].update(
            {str(path.relative_to(self.repo)): file_sha256(path)}))
        with self.assertRaisesRegex(ValueError, "immutable job field mismatch: seed"):
            self.read(binding)

    def test_audit_changed_between_hash_guard_and_parse_is_rejected(self):
        binding = self.create()
        audit_path = self.repo / binding["path"]
        original = sources_module._bound_path
        changed = False

        def replace_after_hash(repo, relative, digest):
            nonlocal changed
            path = original(repo, relative, digest)
            if path == audit_path and not changed:
                value = read_json(path)
                value["changed_after_hash"] = True
                atomic_json(path, value)
                changed = True
            return path

        with patch.object(sources_module, "_bound_path", replace_after_hash), self.assertRaisesRegex(ValueError, "parsed source artifact fingerprint"):
            self.read(binding)
        self.assertTrue(changed)

    def test_parsed_native_row_and_its_bound_hash_cannot_diverge(self):
        binding = self.create()
        original = sources_module._read_json_bytes
        changed = False

        def replace_after_parse(path, expected=None):
            nonlocal changed
            value, digest = original(path, expected)
            if "results" in path.parts and value.get("generation_status") == "complete" and not changed:
                atomic_json(path, {**value, "native_action": "RIGHT"})
                changed = True
            return value, digest

        with patch.object(sources_module, "_read_json_bytes", replace_after_parse), self.assertRaises(ValueError):
            self.read(binding)
        self.assertTrue(changed)

    def test_final_global_bracket_rechecks_earlier_audit_after_later_cohort(self):
        first, second = SUBSPACE_SOURCES[:2]
        bindings = {identifier: self.create(identifier) for identifier in (first, second)}
        original = sources_module._audit_source

        def change_earlier_after_later(*args):
            result = original(*args)
            if args[2]["experiment_id"] == second:
                path = self.repo / bindings[first]["path"]
                atomic_json(path, {**read_json(path), "changed_after_first_source": True})
            return result

        with patch.object(sources_module, "_audit_source", change_earlier_after_later), self.assertRaisesRegex(ValueError, "artifact changed"):
            read_complete_native_sources(self.campaign, self.repo, [first, second], bindings)

    def test_named_independent_audit_cannot_be_replaced_by_automated_copy(self):
        binding = self.create()
        copied = self.campaign / "reviews" / "automated_recomputation.json"
        copied.write_bytes((self.repo / binding["path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "named independent"):
            self.read({"path": str(copied.relative_to(self.repo)), "sha256": file_sha256(copied)})
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.read({**binding, "sha256": "0" * 64})

    def test_subspace_identity_norm_and_full_manifest_are_required(self):
        binding = self.create()
        original = read_json(self.repo / binding["path"])
        edits = [lambda a: a["cohort"]["identity_checks"].pop(),
                 lambda a: a["cohort"]["identity_checks"][0].update(exact_entire_generated_identity=False),
                 lambda a: a["cohort"]["paired_applied_norm_checks"][0].update(expected_original_reference_identical=False),
                 lambda a: a["cohort"].update(required_norm_writes_revalidated=15),
                 lambda a: a["input_file_sha256"].pop(next(iter(a["input_file_sha256"])))]
        for index, edit in enumerate(edits):
            atomic_json(self.repo / binding["path"], original)
            changed = self.mutate_audit(binding, edit)
            with self.subTest(edit=index), self.assertRaises(ValueError):
                self.read(changed)

    def test_contextual_exact_rebuild_and_whole_source_identity_required(self):
        binding = self.create(CONTEXTUAL_SOURCE)
        binding = self.mutate_audit(binding, lambda a: a["preparation_checks"].update(exact_pure_rebuild=False))
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self.read(binding, CONTEXTUAL_SOURCE)

    def test_full_battery_budget_uses_slowest_complete_mean(self):
        bindings = {identifier: self.create(identifier, seconds=20 + index)
                    for index, identifier in enumerate(REQUIRED_BATTERY)}
        result = read_closed_native_battery(self.campaign, self.repo, bindings)
        self.assertEqual(set(result["sources"]), set(REQUIRED_BATTERY))
        self.assertAlmostEqual(result["minimum_budget_seconds_per_native_job"], 1.25 * (26 + 10 / 56))
        with self.assertRaisesRegex(ValueError, "all six"):
            read_closed_native_battery(self.campaign, self.repo, {key: value for key, value in bindings.items() if key != REQUIRED_BATTERY[-1]})

    def test_existing_origin_binds_real_comparison_and_all_controls(self):
        source = self.read(self.create())
        origin = existing_comparison_origin(source, "1", "0")
        self.assertTrue(origin["provenance"]["discovery_measured"])
        self.assertEqual(origin["provenance"]["source_result_manifest_sha256"], source["source_result_manifest_sha256"])
        self.assertEqual(origin["provenance"]["source_independent_audit_sha256"], source["independent_audit"]["sha256"])
        for treatment, control in (("new", "0"), ("1", "new")):
            with self.assertRaisesRegex(ValueError, "one existing"):
                existing_comparison_origin(source, treatment, control)
        duplicate = deepcopy(source)
        duplicate["specification"]["comparisons"] *= 2
        with self.assertRaisesRegex(ValueError, "one existing"):
            existing_comparison_origin(duplicate, "1", "0")

    def test_prospective_exception_has_no_fake_comparison_or_result(self):
        source = self.read(self.create(CONTEXTUAL_SOURCE), CONTEXTUAL_SOURCE)
        module = "src.activation_causality.confirmation_sources"
        with patch(module + "._bound_path") as check:
            origin = prospective_natural_norm_origin(source, self.repo, layer=10,
                adapter_path="src/activation_causality/frozen_contextual_input.py", adapter_sha256="a" * 64)
        self.assertEqual(check.call_count, 2)
        provenance = origin["provenance"]
        self.assertFalse(provenance["discovery_measured"])
        self.assertNotIn("source_comparison_id", provenance)
        self.assertEqual(provenance["control"], "layer10_natural_norm_random")
        with self.assertRaisesRegex(ValueError, "only the contextual"):
            prospective_natural_norm_origin({**source, "experiment_id": SUBSPACE_SOURCES[0]}, self.repo, layer=10,
                adapter_path="src/activation_causality/frozen_contextual_input.py", adapter_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
