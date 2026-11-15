"""CPU plan/publication guards; external model/source authenticity is stubbed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.activation_causality import confirmation_freeze as freeze
from src.activation_causality import confirmation_sampling as sampling
from src.activation_causality.confirmation_rendering import build_primary, eligibility_decisions
from src.activation_causality.environments import generate_pairs
from src.activation_causality.frozen_contextual_input import _configuration
from src.activation_causality.registry import atomic_json, file_sha256, fingerprint, immutable_json
from test_activation_causality_frozen_contextual_input import Tokenizer, configuration

STAGE = Path(__file__).resolve().parents[1]


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.campaign = Path(cls.tmp.name).resolve()
        for folder in ("specs", "datasets", "freezes", "planning", "reviews"):
            (cls.campaign / folder).mkdir()
        cls.freeze_id, cls.eid, cls.cid = "frozen_fixture", "native_fixture", "selected_contextual"
        cls.tokenizer = Tokenizer()
        cls.descriptor = {"profile_id": "contextual_l0", "protocol_kind": "contextual_input",
                          "source_experiment_id": "contextual_discovery", "source_spec_sha256": "4" * 64,
                          "layer": 0, "site": "resid_post", "selected_dose": 1.0, "rank": None, "family": None}
        config = configuration()
        config.update(dataset_seed=20261202, source_profile_descriptor=cls.descriptor)
        rule = freeze.eligibility_rule(STAGE, cls.descriptor, joint=False)
        for binding in config["primary_comparisons"].values():
            binding["eligibility_rule"] = rule
        cls.config = _configuration(config, cls.tokenizer)
        cls.configurations = {cls.cid: cls.config}
        cls.registry = {"schema_version": 1, "candidates": {cls.cid: list(cls.config["primary_comparisons"])},
                        "family_size": 2, "bindings": {cls.cid: cls.config["primary_comparisons"]}}
        deadline = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        immutable_json(cls.campaign / "campaign.json", {"deadline": deadline})
        cls.history = cls.campaign / "history.parquet"
        cls.history.write_bytes(b"CPU historical reader fixture")
        cls.history_audit = {"source": str(cls.history), "rows_checked": 0, "layout_fingerprints": 0, "status": "complete"}
        prior = freeze.prior_exclusion_snapshot(cls.campaign)
        cls.prior_path = cls.campaign / "planning/prior.json"
        immutable_json(cls.prior_path, prior)
        builder_sha = file_sha256(STAGE / "src/activation_causality/confirmation_rendering.py")
        identity = {"policy_sha256": freeze.POLICY_SHA256, "campaign_deadline_utc": deadline,
                    "inference_cutoff_utc": freeze.inference_cutoff(deadline), "dataset_seed": cls.config["dataset_seed"],
                    "generator_sha256": file_sha256(STAGE / "src/activation_causality/environments.py"),
                    "generator_parameters": {"sizes": [7, 9], "wall_density": .23, "donor_mode": "matched",
                         "require_unique_actions": True, "historical_steps": str(cls.history)},
                    "historical_exclusion": {"status": "checked", "source_path": str(cls.history),
                        "source_sha256": file_sha256(cls.history), "source_exclusion_provenance": cls.history_audit},
                    "previously_used_layout_aliases": prior["layout_aliases"],
                    "previously_used_layout_aliases_sha256": prior["layout_aliases_sha256"],
                    "previously_used_layout_aliases_provenance": {"path": str(cls.prior_path), "sha256": file_sha256(cls.prior_path)},
                    "candidates": [{"candidate_id": cls.cid, "source_experiment_id": cls.config["source_experiment_id"],
                        "source_spec_sha256": cls.config["source_spec_sha256"], "configuration": cls.config,
                        "configuration_sha256": fingerprint(cls.config), "eligibility_builder_sha256": builder_sha,
                        "eligibility_rule": rule, "families": cls.config["families"],
                        "family_layout_counts": cls.config["family_layout_counts"], "require_joint_renderer_eligibility": False}]}
        cls.state_path = cls.campaign / "planning/final_sampling_v1.json"
        state = sampling.initialize_sampling_plan(cls.state_path, identity)
        def assess(candidate, pairs):
            spec = build_primary(pairs, cls.tokenizer, cls.eid, candidate["configuration"], cls.freeze_id)
            return {"eligibility_builder_sha256": builder_sha, "decisions": eligibility_decisions(spec)}
        with patch("src.activation_causality.environments.load_historical_layout_ids", return_value=(set(), cls.history_audit)):
            while state["status"] == "sampling":
                state = sampling.advance_sampling_plan(cls.state_path, generator=generate_pairs, eligibility_builder=assess)
        if state["status"] != "complete":
            raise AssertionError(state["terminal_reason"])
        cls.state = state
        cls.pairs = freeze._sampling_pairs(state, cls.configurations)[cls.cid]
        cls.spec = build_primary(cls.pairs, cls.tokenizer, cls.eid, cls.config, cls.freeze_id)
        cls.spec_path = cls.campaign / "planning/spec.json"
        cls.dataset_path = cls.campaign / "planning/dataset.json"
        immutable_json(cls.spec_path, cls.spec)
        immutable_json(cls.dataset_path, {"experiment_id": cls.eid, "candidate_id": cls.cid, "cohort_role": "primary",
                                         "partition": "confirmation", "pairs": cls.pairs})
        cls.decision = {"selected_scientific_profile_ids": {"contextual_input": "contextual_l0", "fixed_family_subspace": None},
                        "profiles": [{"profile_id": "contextual_l0", "descriptor": cls.descriptor}]}
        cls.selection_path = cls.campaign / "planning/selection.json"
        immutable_json(cls.selection_path, {"decision": cls.decision})
        cls.plan = {"confirmation_mode": freeze.MODE, "schema_version": 4, "freeze_id": cls.freeze_id,
                    "campaign_sha256": file_sha256(cls.campaign / "campaign.json"),
                    "inference_cutoff": freeze.inference_cutoff(deadline), "analysis_configuration": freeze.analysis_configuration(),
                    "candidate_configurations": cls.configurations, "confirmation_primary_family": cls.registry,
                    "planned_experiments": {cls.eid: {"candidate_id": cls.cid, "cohort_role": "primary",
                        "spec_path": str(cls.spec_path), "spec_sha256": file_sha256(cls.spec_path),
                        "dataset_path": str(cls.dataset_path), "dataset_sha256": file_sha256(cls.dataset_path)}},
                    "prior_exclusions": {"path": str(cls.prior_path), "sha256": file_sha256(cls.prior_path)},
                    "sampling_state": {"path": str(cls.state_path), "sha256": file_sha256(cls.state_path)},
                    "scientific_selection": {"path": str(cls.selection_path), "sha256": file_sha256(cls.selection_path)},
                    "basis_manifests": {}, "qualified_unexecuted_profiles": {}, "supplements": {cls.cid: False},
                    "budget_seconds_per_native_job": 295.0}

    def context(self):
        return {"decision": deepcopy(self.decision), "battery": {"sources": {"contextual_discovery": {}},
                                                                 "minimum_budget_seconds_per_native_job": 295.0}}, self.tokenizer

    def validate(self, plan=None):
        with patch.object(freeze, "_external_context", return_value=self.context()), \
             patch.object(freeze, "_recheck_external"), \
             patch.object(freeze, "candidate_configuration", return_value=self.config), \
             patch("src.activation_causality.environments.load_historical_layout_ids", return_value=(set(), self.history_audit)):
            return freeze.validate_confirmation_plan(self.campaign, STAGE, plan or self.plan)

    def test_complete_sampled_primary_reconstructs_every_job_and_both_primaries(self):
        result = self.validate()
        self.assertEqual(len(result["specifications"][self.eid]["jobs"]), 480)
        self.assertEqual(result["budget"]["jobs"], 480)

    def test_sampling_fixture_clock_cannot_be_used_in_final_plan(self):
        state = deepcopy(self.state)
        state["clock_override_used"] = True
        state["state_sha256"] = fingerprint({k: v for k, v in state.items() if k != "state_sha256"})
        with self.assertRaisesRegex(ValueError, "production sampling"):
            freeze._sampling_pairs(state, self.configurations)

    def test_accepted_pair_denominator_cannot_be_reduced(self):
        state = deepcopy(self.state)
        state["attempted_pairs"].pop()
        state["state_sha256"] = fingerprint({k: v for k, v in state.items() if k != "state_sha256"})
        with self.assertRaises(ValueError):
            freeze._sampling_pairs(state, self.configurations)

    def test_rejected_alias_cannot_be_reclaimed_as_fresh(self):
        state = deepcopy(self.state)
        rejected = deepcopy(state["attempted_pairs"][0])
        rejected["disposition"] = "rejected"
        state["attempted_pairs"].insert(0, rejected)
        state["state_sha256"] = fingerprint({k: v for k, v in state.items() if k != "state_sha256"})
        with self.assertRaises(ValueError):
            freeze._sampling_pairs(state, self.configurations)

    def test_missing_registry_fails_before_external_reads(self):
        plan = deepcopy(self.plan)
        plan["confirmation_primary_family"]["family_size"] = 1
        with patch.object(freeze, "_external_context", side_effect=AssertionError("external read")), self.assertRaises(ValueError):
            freeze.validate_confirmation_plan(self.campaign, STAGE, plan)

    def test_prior_snapshot_never_reads_outcomes(self):
        results = self.campaign / "results/sentinel.json"
        results.parent.mkdir(exist_ok=True)
        results.write_text("invalid JSON must not be read")
        self.assertFalse(freeze.prior_exclusion_snapshot(self.campaign)["result_files_read"])

    def test_actual_source_derived_geometry_required(self):
        changed = deepcopy(self.config)
        changed["layer"] = 10
        with patch.object(freeze, "_external_context", return_value=self.context()), \
             patch.object(freeze, "candidate_configuration", return_value=changed), \
             patch("src.activation_causality.environments.load_historical_layout_ids", return_value=(set(), self.history_audit)), \
             self.assertRaisesRegex(ValueError, "source derivation"):
            freeze.validate_confirmation_plan(self.campaign, STAGE, self.plan)

    def test_supplement_decision_requires_whole_primary_joint_eligibility(self):
        plan = deepcopy(self.plan)
        plan["supplements"][self.cid] = True
        with self.assertRaisesRegex(ValueError, "supplements"):
            self.validate(plan)

    def test_historical_topology_overlap_rejected(self):
        with patch("src.activation_causality.environments.load_historical_layout_ids",
                   return_value=({self.pairs[0]["layout_id"]}, self.history_audit)), \
             patch.object(freeze, "_external_context", side_effect=AssertionError("external read")), \
             self.assertRaisesRegex(ValueError, "historical topologies"):
            freeze.validate_confirmation_plan(self.campaign, STAGE, self.plan)

    def test_exact_review_required_before_publication_or_validation(self):
        with tempfile.TemporaryDirectory(dir=self.campaign / "planning") as directory:
            plan_path, review_path = Path(directory) / "plan.json", Path(directory) / "review.json"
            immutable_json(plan_path, self.plan)
            immutable_json(review_path, {"approved": True, "freeze_id": self.freeze_id,
                                        "plan_sha256": "0" * 64, "plan_fingerprint": fingerprint(self.plan)})
            with patch.object(freeze, "validate_confirmation_plan", side_effect=AssertionError("validation preceded approval")), \
                 self.assertRaisesRegex(ValueError, "exact prepared plan"):
                freeze.freeze_confirmation_plan(self.campaign, STAGE, plan_path, review_path)

    def test_reviewed_plan_outside_planning_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, review_path = Path(directory) / "plan.json", Path(directory) / "review.json"
            immutable_json(plan_path, self.plan)
            immutable_json(review_path, {"approved": True, "freeze_id": self.freeze_id,
                                        "plan_sha256": file_sha256(plan_path), "plan_fingerprint": fingerprint(self.plan)})
            with patch.object(freeze, "validate_confirmation_plan", side_effect=AssertionError("invalid path proceeded")), \
                 self.assertRaisesRegex(ValueError, "inside campaign/planning"):
                freeze.freeze_confirmation_plan(self.campaign, STAGE, plan_path, review_path)

    def test_rehashed_sampling_accounting_mutations_cannot_pass(self):
        mutations = (
            lambda s: s["attempts"][0]["generator_call"].update(seed=-123),
            lambda s: s["attempts"][0]["generator_call"].update(max_attempts=999999999),
            lambda s: s["attempted_pairs"][0]["builder_decision"]["evidence"].update(adapter_manifest={"tampered": True}),
            lambda s: s.update(global_layout_aliases=[]),
            lambda s: s["candidates"][self.cid]["next_attempt_by_family"].update(wall=101),
        )
        for mutate in mutations:
            state = deepcopy(self.state)
            mutate(state)
            # Preserve each local fingerprint, reproducing the review finding:
            # independently self-consistent pieces still must agree globally.
            for attempt in state["attempts"]:
                attempt["generator_call_sha256"] = fingerprint(attempt["generator_call"])
            for record in state["attempted_pairs"]:
                record["builder_decision_sha256"] = fingerprint(record["builder_decision"])
            state["state_sha256"] = fingerprint({k: v for k, v in state.items() if k != "state_sha256"})
            with self.assertRaises(ValueError):
                freeze._sampling_pairs(state, self.configurations)

    def test_real_preflight_prerequisite_required_before_discovery_reconstruction(self):
        from src.activation_causality import confirmation_planning
        prerequisite = {"path": "/bound/preflight.json", "sha256": "b" * 64,
                        "model_snapshot_identity_sha256": "a" * 64}
        plan = {**deepcopy(self.plan), "implementation": {}, "audit_bindings": {},
                "worker_entrypoint_sha256": file_sha256(STAGE / "scripts/30_activation_causality_campaign.py"),
                "model_identity": {"identity_sha256": "a" * 64}, "activation_representation": {}}
        with patch("src.activation_causality.campaign.runtime_fingerprint", return_value={"sources": {}}), \
             patch("src.activation_causality.selection._model_identity", return_value=plan["model_identity"]), \
             patch("src.activation_causality.basis_runtime.activation_representation", return_value={}), \
             patch.object(confirmation_planning, "verify_preflight_prerequisite", return_value=prerequisite) as preflight, \
             patch.object(freeze, "read_closed_native_battery", side_effect=AssertionError("discovery reconstruction reached")):
            with self.assertRaisesRegex(ValueError, "preflight prerequisite"):
                freeze._external_context(self.campaign, STAGE, plan)
            preflight.assert_called_once()
            plan["technical_preflight"] = prerequisite
            with self.assertRaisesRegex(AssertionError, "discovery reconstruction reached"):
                freeze._external_context(self.campaign, STAGE, plan)

    def test_raw_artifact_hash_and_canonical_path_are_mandatory(self):
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            freeze._bound(str(self.spec_path), "0" * 64)
        alias = self.campaign / "planning/spec_alias.json"
        alias.symlink_to(self.spec_path)
        try:
            with self.assertRaisesRegex(ValueError, "canonical paths"):
                freeze._bound(str(alias), file_sha256(self.spec_path))
        finally:
            alias.unlink()

    def test_duplicate_json_and_nonfinite_payload_rejected(self):
        path = self.campaign / "planning/invalid.json"
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                freeze._read(path)

    def test_budget_includes_all_controls_both_candidates_and_optional_repeats(self):
        protocols = {"context": "contextual_input", "subspace": "input_fact_subspace"}
        choice = freeze.select_budget_allocation(protocols, 1.0, 12000)
        self.assertEqual(choice["n_layouts"], 128)
        self.assertEqual(choice["budget"]["jobs"], (128 + 32) * 29)
        reduced = freeze.select_budget_allocation(protocols, 1.0, 8500)
        self.assertEqual(reduced["n_layouts"], 32)
        self.assertEqual(reduced["supplements"], {"context": False, "subspace": False})
        with self.assertRaisesRegex(ValueError, "even standard-only"):
            freeze.select_budget_allocation(protocols, 1.0, 8000)

    def test_reviewed_publication_registration_readiness_and_registered_mutation(self):
        plan_path = self.campaign / "planning/prepared_plan.json"
        review_path = self.campaign / "reviews/exact_plan_review.json"
        immutable_json(plan_path, self.plan)
        immutable_json(review_path, {"approved": True, "freeze_id": self.freeze_id,
                                    "plan_sha256": file_sha256(plan_path), "plan_fingerprint": fingerprint(self.plan)})
        published = self.campaign / "freezes" / (self.freeze_id + ".json")
        registered_spec = self.campaign / "specs" / (self.eid + ".json")
        registered_dataset = self.campaign / "datasets" / (self.eid + ".json")
        try:
            with patch.object(freeze, "_external_context", return_value=self.context()), \
                 patch.object(freeze, "_recheck_external"), \
                 patch.object(freeze, "candidate_configuration", return_value=self.config), \
                 patch("src.activation_causality.environments.load_historical_layout_ids", return_value=(set(), self.history_audit)):
                frozen = freeze.freeze_confirmation_plan(self.campaign, STAGE, plan_path, review_path)
                self.assertEqual(frozen["confirmation_primary_family"]["family_size"], 2)
                self.assertEqual(freeze.register_frozen_plan(self.campaign, STAGE, self.freeze_id), [self.eid])
                self.assertEqual(file_sha256(registered_spec), file_sha256(self.spec_path))
                self.assertEqual(file_sha256(registered_dataset), file_sha256(self.dataset_path))
                freeze.validate_frozen_confirmation_ready(self.campaign, STAGE, self.spec)
                changed = deepcopy(self.spec)
                changed["unexpected_change"] = True
                atomic_json(registered_spec, changed)
                with self.assertRaisesRegex(ValueError, "registered artifact"):
                    freeze.validate_frozen_confirmation_ready(self.campaign, STAGE, self.spec)
        finally:
            for path in (published, registered_spec, registered_dataset, plan_path, review_path):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
