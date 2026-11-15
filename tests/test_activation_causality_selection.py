"""Freeze/holdout safety tests; all data and model metadata are temporary fakes."""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activation_causality.environments import partition_for_layout  # noqa: E402
from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS  # noqa: E402
from src.activation_causality.registry import file_sha256, fingerprint  # noqa: E402
from src.activation_causality.selection import (  # noqa: E402
    ANALYSIS_CONFIG, MIXED_FAMILIES, excluded_layout_ids, freeze_selection, validate_confirmation_ready,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class FakeTokenizer:
    def __init__(self, model_path):
        self.name_or_path = str(model_path)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("no extra tokenizer framing")
        return [300 if piece == "False" else 301 if piece == "True" else ord(piece)
                for piece in re.findall(r"False|True|[\s\S]", text)]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.campaign = self.repo / "campaign"
        self.model = self.repo / "model"
        write(self.model / "config.json", {})
        write(self.model / "tokenizer.json", {})
        write(self.model / "model.safetensors.index.json", {"weight_map": {"fake_tensor": "model-00001.safetensors"}})
        (self.model / "model-00001.safetensors").write_bytes(b"tiny fake weights; no inference")
        source_dir = self.repo / "src/activation_causality"
        source_dir.mkdir(parents=True)
        (source_dir / "analysis.py").write_text("frozen analysis\n")
        (source_dir / "engine.py").write_text("frozen engine\n")
        for relative in ENGINE_DEPENDENCY_PATHS:
            dependency = self.repo / relative
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text("# frozen dependency fixture\n")
        write(self.campaign / "campaign.json", {
            "model_path": str(self.model),
            "deadline": (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
        })
        self.source = {"experiment_id": "discovery", "partition": "discovery", "jobs": [],
                       "comparisons": [{"id": "source_contrast", "treatment": "layer10_matched", "control": "baseline"}]}
        for condition in ("baseline", "layer10_matched", "layer10_random"):
            self.source["jobs"].append({
                "job_id": condition, "experiment_id": "discovery", "partition": "discovery", "family": "wall",
                "pair_id": "development_pair", "layout_id": "development_layout", "condition": condition,
                "intervention": {"layers": [] if condition == "baseline" else [10], "site": "resid_post", "operation": "replace", "dose": 1.0},
                "selection": {"scope": "last_tokens", "last_k": 1}, "temperature": .7,
            })
        write(self.campaign / "specs/discovery.json", self.source)
        for job in self.source["jobs"]:
            write(self.campaign / "results/discovery" / f"{job['job_id']}.json",
                  {**job, "seed": 42, "status": "complete", "generation_status": "not_requested",
                   "spec_sha256": file_sha256(self.campaign / "specs/discovery.json")})
        write(self.campaign / "results/pilot/pilot1.json", {
            "job_id": "pilot1", "experiment_id": "pilot", "partition": "historical",
            "status": "complete", "natural": True, "generation_status": "complete", "native_action": "RIGHT",
            "seconds": 2.0, "max_new_tokens": 2048,
        })
        self.selection = {
            "freeze_id": "main", "review_path": str(self.repo / "review.json"),
            "selection_rationale": "Prespecified wall mechanism candidate", "pilot_job_ids": ["pilot1"],
            "pilot_seconds_per_native_job": 2.0,
            "candidates": [{"candidate_id": "wall10", "source_experiment_id": "discovery", "family": "wall",
                            "treatment_condition": "layer10_matched", "control_condition": "baseline",
                            "n_layouts": 32, "generation_seeds": [42], "max_new_tokens": 2048, "dataset_seed": 20260914}],
        }
        self.approve()

    def approve(self):
        write(self.repo / "review.json", {"approved": True, "freeze_id": self.selection["freeze_id"],
                                         "selection_sha256": fingerprint(self.selection)})

    def freeze(self):
        self.approve()
        return freeze_selection(self.campaign, self.repo, self.selection)

    def input_source(self, *, natural=True):
        """Two source positions must freeze one alignment rule, not coordinates."""
        self.source = {"experiment_id": "input", "partition": "discovery", "jobs": [],
                       "claim_scope": "input_fact_information_routing; not an abstract-belief claim",
                       "tokenizer": {"class": "FakeTokenizer", "name_or_path": str(self.model),
                                     "add_special_tokens": False, "context_renderer": "campaign.render_analysis_prompt"},
                       "comparisons": [{"id": "input_contrast", "treatment": "factspan_layer0", "control": "baseline"}]}
        for index in range(2):
            for condition in ("baseline", "counterfactual_text", "identity_layer0", "factspan_layer0", "random_factspan_layer0", "lasttoken_layer0"):
                job = {"job_id": f"{condition}_{index}", "experiment_id": "input", "partition": "discovery", "family": "wall",
                       "layout_id": f"input_layout_{index}", "pair_id": f"input_pair_{index}", "seed": 42, "condition": condition,
                       "recipient_token_ids": [10] * (index + 1) + [20, 30],
                       "donor_token_ids": [10] * (index + 1) + [21, 30],
                       "selection": {"scope": "explicit", "recipient_positions": [index + 1], "donor_positions": [index + 1]},
                       "intervention": {"layers": [] if condition in {"baseline", "counterfactual_text"} else [0],
                                        "site": "resid_post", "operation": "replace", "dose": 1.0,
                                        "random_control": condition.startswith("random")},
                       "temperature": .7, "natural": natural,
                       "metadata": {"representation_kind": "input_fact_tokens", "claim_scope": "input_fact_information_routing"}}
                self.source["jobs"].append(job)
        write(self.campaign / "specs/input.json", self.source)
        for job in self.source["jobs"]:
            write(self.campaign / "results/input" / f"{job['job_id']}.json", {
                **job, "status": "complete", "generation_status": "complete" if natural else "not_requested",
                "native_action": "RIGHT" if natural else None,
                "input_token_sha256": fingerprint(job["recipient_token_ids"]),
                "spec_sha256": file_sha256(self.campaign / "specs/input.json")})
        self.selection["candidates"][0].update(source_experiment_id="input", treatment_condition="factspan_layer0")

    def mixed_input_source(self):
        self.input_source()
        originals = self.source["jobs"]
        self.source["jobs"] = []
        for family in MIXED_FAMILIES:
            for original in originals:
                job = copy.deepcopy(original)
                job["family"] = family
                for key in ("job_id", "pair_id", "layout_id"):
                    job[key] = family + "_" + job[key]
                self.source["jobs"].append(job)
        self.write_input_source_results()
        candidate = self.selection["candidates"][0]
        candidate.pop("family")
        candidate.update(families=list(MIXED_FAMILIES),
                         family_layout_counts={"wall": 11, "key_possession": 11, "door_status": 10})

    def write_input_source_results(self):
        write(self.campaign / "specs/input.json", self.source)
        for path in (self.campaign / "results/input").glob("*.json"):
            path.unlink()
        for job in self.source["jobs"]:
            write(self.campaign / "results/input" / f"{job['job_id']}.json", {
                **job, "status": "complete", "generation_status": "complete",
                "native_action": "RIGHT",
                "input_token_sha256": fingerprint(job["recipient_token_ids"]),
                "spec_sha256": file_sha256(self.campaign / "specs/input.json")})

    def confirmation(self):
        layout_ids = []
        for index in range(1000):
            value = f"fresh_layout_{index}"
            if partition_for_layout(value) == "confirmation":
                layout_ids.append(value)
                if len(layout_ids) == 32:
                    break
        jobs = []
        for layout in layout_ids:
            pair = {"pair_id": f"pair-{layout}", "layout_id": layout, "family": "wall", "partition": "confirmation"}
            for source_job in self.source["jobs"]:
                job = {**source_job, "experiment_id": "confirmation", "partition": "confirmation",
                       "layout_id": layout, "pair_id": pair["pair_id"], "metadata": {"pair": pair},
                       "natural": True, "max_new_tokens": 2048, "seed": 42}
                job["job_id"] = fingerprint(job)
                jobs.append(job)
        comparison = {"id": "confirmation_contrast", "treatment": "layer10_matched", "control": "baseline",
                      "expected_n_layouts": 32, "expected_n_pairs": 32,
                      "eligible_pair_ids": [f"pair-{layout}" for layout in layout_ids]}
        expected = {"jobs": jobs, "comparisons": [comparison]}
        actual = {**copy.deepcopy(expected), "experiment_id": "confirmation", "partition": "confirmation",
                  "freeze_id": "main", "candidate_id": "wall10"}
        actual["comparisons"][0]["confirmatory"] = True
        return expected, actual

    def test_freeze_derives_configuration_and_freezes_analysis(self):
        frozen = self.freeze()
        config = frozen["candidate_configurations"]["wall10"]
        self.assertEqual(config["layer"], 10)
        self.assertEqual(config["conditions_per_layout_upper_bound"], 3)
        self.assertEqual(frozen["pilot"]["mean_seconds"], 2)
        self.assertEqual(frozen["analysis_configuration"], ANALYSIS_CONFIG)
        self.assertIn("src/activation_causality/analysis.py", frozen["implementation"])
        self.assertTrue(set(ENGINE_DEPENDENCY_PATHS).issubset(frozen["implementation"]))
        self.assertEqual(freeze_selection(self.campaign, self.repo, self.selection), frozen)

    def test_second_freeze_or_edit_is_rejected(self):
        self.freeze()
        self.selection["candidates"][0]["n_layouts"] = 64
        with self.assertRaisesRegex(ValueError, "cannot revise"):
            self.freeze()
        self.selection["freeze_id"] = "another"
        with self.assertRaisesRegex(ValueError, "one immutable"):
            self.freeze()

    def test_confirmation_dependency_changes_and_omissions_fail_closed(self):
        frozen = self.freeze()
        _, actual = self.confirmation()
        freeze_path = self.campaign / "freezes/main.json"
        for relative in ENGINE_DEPENDENCY_PATHS:
            with self.subTest(relative=relative):
                dependency = self.repo / relative
                original = dependency.read_bytes()
                dependency.write_bytes(original + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    validate_confirmation_ready(self.campaign, self.repo, actual)
                dependency.write_bytes(original)
                incomplete = copy.deepcopy(frozen)
                incomplete["implementation"].pop(relative)
                write(freeze_path, incomplete)
                with self.assertRaisesRegex(ValueError, "engine dependency"):
                    validate_confirmation_ready(self.campaign, self.repo, actual)
                write(freeze_path, frozen)
    def test_exact_independent_review_is_required(self):
        self.selection["selection_rationale"] = "changed after review"
        with self.assertRaisesRegex(ValueError, "exact selection"):
            freeze_selection(self.campaign, self.repo, self.selection)

    def test_throughput_cannot_be_negative_or_faster_than_pilot(self):
        for value in (-1, 0, float("nan"), 1):
            self.selection["pilot_seconds_per_native_job"] = value
            # Fingerprints reject nonfinite values independently.
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.freeze()

    def test_native_pilot_cannot_be_forced_only(self):
        path = self.campaign / "results/pilot/pilot1.json"
        value = json.loads(path.read_text())
        value.update(natural=False, generation_status="not_requested", native_action=None)
        write(path, value)
        with self.assertRaisesRegex(ValueError, "native executions"):
            self.freeze()

    def test_caller_cannot_understate_condition_count(self):
        self.selection["candidates"][0]["conditions_per_layout"] = 1
        with self.assertRaisesRegex(ValueError, "source-derived inventory"):
            self.freeze()

    def test_candidate_cannot_be_frozen_from_unexecuted_development(self):
        (self.campaign / "results/discovery/layer10_matched.json").unlink()
        with self.assertRaisesRegex(ValueError, "complete source cohort result inventory"):
            self.freeze()

    def test_input_freezes_alignment_rule_and_restricted_scope(self):
        self.input_source()
        candidate = self.freeze()["candidate_configurations"]["wall10"]
        self.assertEqual(candidate["protocol_kind"], "input_fact")
        self.assertEqual(candidate["conditions_per_layout_upper_bound"], 6)
        self.assertEqual(candidate["observed_source_pairs"], 2)
        self.assertEqual(candidate["selection"], {"scope": "explicit", "alignment": "exact_equal_length_common_prefix_and_suffix"})
        self.assertEqual(candidate["claim_scope"], "input_fact_information_routing")
        self.assertIs(candidate["semantic_headline_eligible"], False)
        self.assertNotIn("last_k", candidate)

    def test_input_forced_only_source_cannot_freeze(self):
        self.input_source(natural=False)
        with self.assertRaisesRegex(ValueError, "no completed paired"):
            self.freeze()

    def test_source_result_cannot_forge_job_fields_id_or_input_hash(self):
        self.input_source()
        path = self.campaign / "results/input/factspan_layer0_0.json"
        original = json.loads(path.read_text())
        for field, value in (("job_id", "invented"), ("family", "door_status"),
                             ("seed", 999), ("input_token_sha256", "wrong"), ("native_action", None)):
            write(path, {**original, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.freeze()
        write(path, original)
        write(path.with_name("copied_result.json"), original)
        with self.assertRaisesRegex(ValueError, "complete source cohort result inventory"):
            self.freeze()

    def test_selected_source_result_hashes_are_frozen_and_rechecked(self):
        frozen = self.freeze()
        config = frozen["candidate_configurations"]["wall10"]
        self.assertEqual(len(config["source_result_hashes"]), 3)
        self.assertEqual(frozen["source_results"], config["source_result_hashes"])
        _, actual = self.confirmation()
        path = self.campaign / "results/discovery/layer10_matched.json"
        original = path.read_text()
        path.write_text(original + "\n")
        with self.assertRaisesRegex(ValueError, "source result changed"):
            validate_confirmation_ready(self.campaign, self.repo, actual)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "complete source cohort result inventory"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_freeze_requires_unselected_source_control_to_be_present(self):
        (self.campaign / "results/discovery/layer10_random.json").unlink()
        with patch("src.activation_causality.selection._validated_source_result", side_effect=AssertionError("outcome opened")):
            with self.assertRaisesRegex(ValueError, "complete source cohort result inventory"):
                self.freeze()

    def test_freeze_rejects_technical_failure_in_unselected_control(self):
        path = self.campaign / "results/discovery/layer10_random.json"
        result = json.loads(path.read_text())
        write(path, {**result, "status": "error"})
        with self.assertRaisesRegex(ValueError, "technical completion of every source condition"):
            self.freeze()

    def test_freeze_preserves_unresolved_native_controls_in_full_manifest(self):
        self.input_source()
        for index, status in enumerate(("truncated", "malformed")):
            path = self.campaign / f"results/input/random_factspan_layer0_{index}.json"
            result = json.loads(path.read_text())
            write(path, {**result, "generation_status": status, "native_action": None})
        frozen = self.freeze()
        self.assertEqual(len(frozen["source_results"]), 12)
        for path in (self.campaign / "results/input").glob("*.json"):
            self.assertEqual(frozen["source_results"][str(path.relative_to(self.campaign))], file_sha256(path))

    def test_freeze_rejects_unrecorded_native_control_endpoint(self):
        self.input_source()
        path = self.campaign / "results/input/random_factspan_layer0_0.json"
        result = json.loads(path.read_text())
        write(path, {**result, "generation_status": "not_requested", "native_action": None})
        with self.assertRaisesRegex(ValueError, "valid completion or unresolved status"):
            self.freeze()

    def test_readiness_rechecks_unselected_control_bytes(self):
        self.freeze()
        _, actual = self.confirmation()
        path = self.campaign / "results/discovery/layer10_random.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source result changed"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_readiness_rejects_manifest_omitting_unselected_control(self):
        frozen = self.freeze()
        _, actual = self.confirmation()
        relative = "results/discovery/layer10_random.json"
        del frozen["candidate_configurations"]["wall10"]["source_result_hashes"][relative]
        del frozen["source_results"][relative]
        write(self.campaign / "freezes/main.json", frozen)
        with self.assertRaisesRegex(ValueError, "must cover every source condition"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_readiness_rejects_added_source_result(self):
        self.freeze()
        _, actual = self.confirmation()
        write(self.campaign / "results/discovery/extra.json", {})
        with self.assertRaisesRegex(ValueError, "complete source cohort result inventory"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_mixed_input_freezes_one_fixed_intervention_and_three_native_families(self):
        self.mixed_input_source()
        config = self.freeze()["candidate_configurations"]["wall10"]
        self.assertEqual(config["families"], list(MIXED_FAMILIES))
        self.assertNotIn("family", config)
        self.assertEqual(config["observed_source_pairs_by_family"], dict.fromkeys(MIXED_FAMILIES, 2))
        self.assertEqual(config["family_weights"], {"wall": 11 / 32, "key_possession": 11 / 32, "door_status": 10 / 32})
        self.assertEqual(config["conditions_per_layout_upper_bound"], 6)
        self.assertEqual(config["stratification_key"], "family")
        self.assertFalse(config["semantic_headline_eligible"])

    def test_mixed_candidate_rejects_invalid_families_quotas_or_weights(self):
        self.mixed_input_source()
        original = copy.deepcopy(self.selection["candidates"][0])
        changes = ({"family": "wall"}, {"families": list(reversed(MIXED_FAMILIES))},
                   {"families": ["wall", "door_status"]},
                   {"family_layout_counts": {"wall": 12, "key_possession": 10, "door_status": 10}},
                   {"family_layout_counts": {"wall": 11, "key_possession": 11, "door_status": 11}},
                   {"family_weights": dict.fromkeys(MIXED_FAMILIES, 1 / 3)})
        for change in changes:
            self.selection["candidates"][0] = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.freeze()

    def test_mixed_candidate_requires_native_source_pairs_in_each_family(self):
        self.mixed_input_source()
        for path in (self.campaign / "results/input").glob("*.json"):
            value = json.loads(path.read_text())
            if value["family"] == "door_status":
                value.update(generation_status="not_requested")
                write(path, value)
        with self.assertRaisesRegex(ValueError, "valid completion or unresolved status"):
            self.freeze()

    def test_mixed_source_requires_common_intervention_and_control_inventory(self):
        self.mixed_input_source()
        original = copy.deepcopy(self.source)
        changed_job = next(job for job in self.source["jobs"]
                           if job["family"] == "door_status" and job["condition"] == "factspan_layer0")
        changed_job["intervention"]["site"] = "mlp_out"
        self.write_input_source_results()
        with self.assertRaisesRegex(ValueError, "ambiguous intervention"):
            self.freeze()
        self.source = original
        self.source["jobs"] = [job for job in self.source["jobs"]
                               if not (job["family"] == "door_status" and job["condition"] == "random_factspan_layer0")]
        self.write_input_source_results()
        with self.assertRaisesRegex(ValueError, "different source control inventories"):
            self.freeze()

    def test_mixed_ready_checks_real_balanced_pairs_jobs_and_primary_counts(self):
        from src.activation_causality.confirmation_adapters import build_frozen_input_spec
        from src.activation_causality.environments import generate_pairs
        self.mixed_input_source()
        config = self.freeze()["candidate_configurations"]["wall10"]
        tokenizer = FakeTokenizer(self.model)
        pairs = generate_pairs(32, 20260918, partition="confirmation", families=MIXED_FAMILIES)
        actual = build_frozen_input_spec(pairs, tokenizer, "mixed_confirmation", config, "main")
        self.assertEqual(len(actual["jobs"]), 192)
        with patch("src.activation_causality.selection._confirmation_tokenizer", return_value=tokenizer):
            validate_confirmation_ready(self.campaign, self.repo, actual)
            changed = copy.deepcopy(actual)
            chosen = next(item for item in changed["comparisons"] if item["confirmatory"])
            chosen["expected_n_layouts_by_family"]["wall"] -= 1
            with self.assertRaisesRegex(ValueError, "eligibility or endpoint"):
                validate_confirmation_ready(self.campaign, self.repo, changed)
            changed = copy.deepcopy(actual)
            pair_id = changed["jobs"][0]["pair_id"]
            for job in changed["jobs"]:
                if job["pair_id"] == pair_id:
                    job["metadata"]["source_pair"]["family"] = "door_status"
            with self.assertRaisesRegex(ValueError, "family quotas"):
                validate_confirmation_ready(self.campaign, self.repo, changed)
            changed = copy.deepcopy(actual)
            changed["jobs"].pop()
            with self.assertRaisesRegex(ValueError, "frozen intervention/control/seed"):
                validate_confirmation_ready(self.campaign, self.repo, changed)

    def test_input_cannot_freeze_changed_source_position_or_claim(self):
        self.input_source()
        job = next(job for job in self.source["jobs"] if job["condition"] == "factspan_layer0")
        job["selection"]["recipient_positions"] = [0]
        write(self.campaign / "specs/input.json", self.source)
        with self.assertRaisesRegex(ValueError, "exact aligned"):
            self.freeze()

    def test_input_ready_rebuilds_real_fresh_layout_contract(self):
        from src.activation_causality.confirmation_adapters import build_frozen_input_spec
        from src.activation_causality.environments import generate_pairs
        self.input_source()
        config = self.freeze()["candidate_configurations"]["wall10"]
        tokenizer = FakeTokenizer(self.model)
        pairs = generate_pairs(32, 20260918, partition="confirmation", families=("wall",))
        actual = build_frozen_input_spec(pairs, tokenizer, "input_confirmation", config, "main")
        self.assertEqual(len(actual["jobs"]), 32 * 6)
        with patch("src.activation_causality.selection._confirmation_tokenizer", return_value=tokenizer):
            validate_confirmation_ready(self.campaign, self.repo, actual)
            changed = copy.deepcopy(actual)
            changed["jobs"][2]["selection"]["recipient_positions"] = [0]
            with self.assertRaisesRegex(ValueError, "frozen intervention/control/seed"):
                validate_confirmation_ready(self.campaign, self.repo, changed)
            changed = copy.deepcopy(actual)
            changed["semantic_headline_eligible"] = True
            with self.assertRaisesRegex(ValueError, "promote its frozen claim scope"):
                validate_confirmation_ready(self.campaign, self.repo, changed)
            changed = copy.deepcopy(actual)
            changed["comparisons"][0]["eligible_pair_ids"].pop()
            with self.assertRaisesRegex(ValueError, "eligibility or endpoint"):
                validate_confirmation_ready(self.campaign, self.repo, changed)

    def test_layout_aliases_are_frozen_across_datasets_and_specs(self):
        write(self.campaign / "datasets/dev.json", {"pairs": [], "layout_aliases": {"base": ["variant-a", "variant-b"]}})
        self.assertTrue({"variant-a", "variant-b", "development_layout"} <= excluded_layout_ids(self.campaign))

    def test_ready_gate_accepts_exact_rebuilt_contract(self):
        self.freeze()
        expected, actual = self.confirmation()
        with patch("src.activation_causality.experiments.build_controlled_state_spec", return_value=expected) as builder:
            validate_confirmation_ready(self.campaign, self.repo, actual)
        self.assertEqual(builder.call_args.kwargs["layers"], [10])
        self.assertEqual(builder.call_args.kwargs["seeds"], [42])
        self.assertTrue(builder.call_args.kwargs["natural"])

    def test_ready_gate_rejects_changed_source_model_or_analysis(self):
        self.freeze()
        _, actual = self.confirmation()
        for path in (self.repo / "src/activation_causality/analysis.py", self.model / "tokenizer.json",
                     self.model / "model-00001.safetensors", self.campaign / "specs/discovery.json"):
            original = path.read_text()
            path.write_text(original + " \n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_confirmation_ready(self.campaign, self.repo, actual)
            path.write_text(original)

    def test_ready_gate_rejects_missing_seed_rows_and_changed_intervention(self):
        self.freeze()
        expected, actual = self.confirmation()
        for change in ("drop", "layer", "seed", "natural"):
            changed = copy.deepcopy(actual)
            if change == "drop":
                changed["jobs"].pop()
            elif change == "layer":
                changed["jobs"][1]["intervention"]["layers"] = [15]
            elif change == "seed":
                changed["jobs"][0]["seed"] = 43
            else:
                changed["jobs"][0]["natural"] = False
            with self.subTest(change=change), patch("src.activation_causality.experiments.build_controlled_state_spec", return_value=expected):
                with self.assertRaisesRegex(ValueError, "frozen intervention/control/seed"):
                    validate_confirmation_ready(self.campaign, self.repo, changed)

    def test_ready_gate_rejects_posthoc_comparison_subset(self):
        self.freeze()
        expected, actual = self.confirmation()
        actual["comparisons"][0]["eligible_pair_ids"].pop()
        with patch("src.activation_causality.experiments.build_controlled_state_spec", return_value=expected):
            with self.assertRaisesRegex(ValueError, "eligibility or endpoint"):
                validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_confirmation_cannot_open_before_freeze(self):
        write(self.campaign / "specs/premature.json", {"partition": "confirmation", "jobs": []})
        with self.assertRaisesRegex(ValueError, "already opened"):
            self.freeze()

    def test_frozen_candidate_cannot_get_a_second_confirmation_experiment(self):
        self.freeze()
        _, actual = self.confirmation()
        write(self.campaign / "specs/other.json", {"experiment_id": "other", "partition": "confirmation",
                                                "freeze_id": "main", "candidate_id": "wall10", "jobs": []})
        with self.assertRaisesRegex(ValueError, "only one experiment"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_confirmation_rejects_sibling_alias_registered_after_freeze(self):
        self.freeze()
        _, actual = self.confirmation()
        layout = actual["jobs"][0]["layout_id"]
        write(self.campaign / "specs/sibling.json", {
            "experiment_id": "sibling", "partition": "confirmation",
            "freeze_id": "main", "candidate_id": "other_candidate",
            "jobs": [{"layout_id": "different_base", "layout_aliases": [layout]}],
        })
        with self.assertRaisesRegex(ValueError, "another registered cohort after freeze"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_confirmation_rejects_sibling_dataset_before_spec_registration(self):
        self.freeze()
        _, actual = self.confirmation()
        write(self.campaign / "datasets/sibling.json", {
            "partition": "confirmation", "pairs": [],
            "layout_aliases": {"different_base": [actual["jobs"][0]["layout_id"]]},
        })
        with self.assertRaisesRegex(ValueError, "another registered cohort after freeze"):
            validate_confirmation_ready(self.campaign, self.repo, actual)

    def test_confirmation_can_resume_its_own_registered_dataset(self):
        self.freeze()
        expected, actual = self.confirmation()
        write(self.campaign / "datasets/confirmation.json", {
            "pairs": [job["metadata"]["pair"] for job in actual["jobs"]],
        })
        write(self.campaign / "specs/confirmation.json", actual)
        with patch("src.activation_causality.experiments.build_controlled_state_spec", return_value=expected):
            validate_confirmation_ready(self.campaign, self.repo, actual)


if __name__ == "__main__":
    unittest.main()
