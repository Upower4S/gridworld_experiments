"""CPU fixtures for deterministic reservations, shared deadlines and aliases."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.activation_causality.confirmation_sampling import (
    FAMILIES, SALT, SamplingDeadlineExceeded, _bounded, _save, advance_sampling_plan, initialize_sampling_plan, select_lexical_subset,
)
from src.activation_causality.environments import canonical_layout_fingerprint, generate_pairs
from src.activation_causality.registry import file_sha256, fingerprint, read_json

START = datetime(2026, 9, 12, 15, tzinfo=timezone.utc)
HISTORICAL = None


def identity(family="wall", n=32, candidates=1, joint=False):
    definitions = []
    for index in range(candidates):
        candidate_id = f"candidate{index}"
        config = {"candidate_id": candidate_id, "source_experiment_id": "source", "source_spec_sha256": "a" * 64,
                  "family": family, "n_layouts": n, "condition_inventory": [f"condition{v}" for v in range(14)],
                  "require_joint_renderer_eligibility": joint}
        definitions.append({"candidate_id": candidate_id, "source_experiment_id": "source", "source_spec_sha256": "a" * 64,
            "configuration": config, "configuration_sha256": fingerprint(config), "eligibility_builder_sha256": "b" * 64,
            "eligibility_rule": {"identifier": "fixture_complete_controls", "sha256": "c" * 64},
            "families": [family], "family_layout_counts": {family: n}, "require_joint_renderer_eligibility": joint})
    return {"policy_sha256": "d" * 64, "campaign_deadline_utc": (START + timedelta(days=2)).isoformat(),
            "inference_cutoff_utc": (START + timedelta(days=2, hours=-6)).isoformat(), "dataset_seed": 20261101,
            "generator_sha256": "e" * 64, "generator_parameters": {"sizes": [7, 9], "wall_density": 0.23,
                "donor_mode": "matched", "require_unique_actions": True, "historical_steps": str(HISTORICAL)},
            "historical_exclusion": {"status": "checked", "source_path": str(HISTORICAL),
                "source_sha256": file_sha256(HISTORICAL), "source_exclusion_provenance": {"kind": "empty_CPU_historical_fixture"}},
            "candidates": definitions, "previously_used_layout_aliases": [],
            "previously_used_layout_aliases_sha256": fingerprint([]),
            "previously_used_layout_aliases_provenance": {"kind": "empty_CPU_fixture"}}


def decision(pair, candidate, eligible=True, extra_alias=None):
    aliases = {pair["layout_id"], *pair["metadata"]["variant_layout_ids"]}
    aliases.update(canonical_layout_fingerprint(pair[key]) for key in ("recipient", "counterfactual", "donor"))
    if extra_alias:
        aliases.add(extra_alias)
    return {"pair_id": pair["pair_id"], "layout_id": pair["layout_id"], "source_pair_sha256": fingerprint(pair),
            "eligible": eligible, "reason": None if eligible else "fixture_control_ineligible", "layout_aliases": sorted(aliases),
            "evidence": {"adapter_manifest": {"eligible": eligible,
                    "scheduled_conditions": candidate["configuration"]["condition_inventory"] if eligible else []},
                "required_condition_inventory": candidate["configuration"]["condition_inventory"],
                "joint_renderer_audit": {"eligible": True, "pair_id": pair["pair_id"]}}}


def assess(candidate, pairs):
    return {"eligibility_builder_sha256": candidate["eligibility_builder_sha256"],
            "decisions": [decision(pair, candidate) for pair in pairs]}


def returned_fixture(pairs, call):
    result = deepcopy(pairs)
    for index, pair in enumerate(result):
        pair["metadata"].update(seed=call["seed"], candidate_attempt=index)
    return result


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global HISTORICAL
        cls.history_tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.history_tmp.cleanup)
        HISTORICAL = Path(cls.history_tmp.name).resolve() / "history.json"
        HISTORICAL.write_text('[]\n')
        cls.wall = generate_pairs(66, 20261102, "confirmation", families=("wall",))
        cls.key = generate_pairs(33, 20261103, "confirmation", families=("key_possession",))
        cls.door = generate_pairs(33, 20261104, "confirmation", families=("door_status",))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "plan.json"
        self.now = START
        self.calls = []

    def clock(self):
        return self.now.isoformat()

    def initialize(self, config=None):
        return initialize_sampling_plan(self.path, config or identity(), now_utc=self.clock)

    def run_batch(self, pairs=None, builder=assess, generator=None):
        def generate(**kwargs):
            self.calls.append(deepcopy(kwargs))
            return returned_fixture(self.wall[:32] if pairs is None else pairs, kwargs)
        return advance_sampling_plan(self.path, generator=generator or generate, eligibility_builder=builder, now_utc=self.clock)

    def test_complete_and_idempotent_shared_clock(self):
        initial = self.initialize()
        done = self.run_batch()
        self.assertEqual(done["status"], "complete")
        self.assertEqual(done["candidates"]["candidate0"]["accepted_n_layouts_by_family"], {"wall": 32})
        self.assertEqual(len(done["attempted_pairs"]), 32)
        self.assertEqual(done["attempts"][0]["generator_call"]["max_attempts"], 96000)
        self.assertEqual(done["attempts"][0]["generator_call"]["seed"], 20261101)
        self.assertFalse(done["registered_dataset"])
        self.assertEqual(self.initialize()["started_utc"], initial["started_utc"])
        self.assertEqual(self.run_batch(), done)
        for record in done["attempted_pairs"]:
            self.assertEqual(record["builder_decision_sha256"], fingerprint(record["builder_decision"]))
            self.assertEqual(record["pair_sha256"], fingerprint(record["pair"]))

    def test_reservation_persisted_before_callback_and_no_restart(self):
        self.initialize()
        class Interrupt(BaseException):
            pass
        def interrupted(**kwargs):
            saved = read_json(self.path)
            self.assertEqual(saved["attempts"][0]["status"], "reserved")
            self.assertEqual(saved["candidates"]["candidate0"]["next_attempt_by_family"]["wall"], 1)
            raise Interrupt()
        with self.assertRaises(Interrupt):
            self.run_batch(generator=interrupted)
        result = self.run_batch()
        self.assertEqual(result["terminal_reason"], "interrupted_reserved_batch_not_retried")
        self.assertEqual(self.calls, [])
        self.assertEqual(result["attempts"][0]["status"], "interrupted_terminal")

    def test_returned_pairs_survive_builder_interrupt(self):
        self.initialize()
        class Interrupt(BaseException):
            pass
        def interrupted(candidate, pairs):
            self.assertEqual(len(read_json(self.path)["attempted_pairs"]), 32)
            raise Interrupt()
        with self.assertRaises(Interrupt):
            self.run_batch(builder=interrupted)
        result = self.run_batch()
        self.assertEqual(result["terminal_reason"], "interrupted_reserved_batch_not_retried")
        self.assertTrue(all(item["disposition"] == "rejected" for item in result["attempted_pairs"]))

    def test_rejections_refill_next_seed_keep_aliases_and_no_family_swap(self):
        self.initialize()
        def reject_one(candidate, pairs):
            return {"eligibility_builder_sha256": candidate["eligibility_builder_sha256"],
                    "decisions": [decision(pair, candidate, index != 0, "rejected_extra" if index == 0 else None)
                                  for index, pair in enumerate(pairs)]}
        first = self.run_batch(builder=reject_one)
        self.assertEqual(first["status"], "sampling")
        self.assertEqual(first["candidates"]["candidate0"]["accepted_n_layouts_by_family"]["wall"], 31)
        self.assertIn("rejected_extra", first["global_layout_aliases"])
        second = self.run_batch(self.wall[32:33])
        call = second["attempts"][1]["generator_call"]
        self.assertEqual(call["seed"], 20261102)
        self.assertEqual(call["n_layouts"], 1)
        self.assertEqual(call["max_attempts"], 3000)
        self.assertEqual(call["families"], ["wall"])
        self.assertIn("rejected_extra", call["excluded_layout_ids"])
        self.assertEqual(second["status"], "complete")
        self.assertEqual(len(second["attempted_pairs"]), 33)

    def test_global_aliases_cross_candidate_and_seed_schedule(self):
        config = identity(candidates=2)
        self.initialize(config)
        first = self.run_batch()
        self.assertEqual(first["status"], "sampling")
        second = self.run_batch(self.wall[32:64])
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["attempts"][1]["generator_call"]["seed"], 20261101)
        self.assertIn(self.wall[0]["layout_id"], second["attempts"][1]["generator_call"]["excluded_layout_ids"])
        self.assertEqual(second["started_utc"], first["started_utc"])

    def test_key_family_uses_canonical_seed_index(self):
        self.initialize(identity(family="key_possession"))
        result = self.run_batch(self.key[:32])
        self.assertEqual(result["attempts"][0]["generator_call"]["seed"], 20262101)
        self.assertEqual(result["status"], "complete")

    def test_exhaustion_is_terminal_and_partial_bytes_explicitly_unavailable(self):
        self.initialize()
        def exhausted(**kwargs):
            raise RuntimeError("Generated 7/32 eligible layouts within the attempt budget")
        result = self.run_batch(generator=exhausted)
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertFalse(result["attempts"][0]["partial_pair_bytes_available"])
        self.assertIn("Generated 7/32", result["attempts"][0]["error"])
        self.assertEqual(result["attempted_pairs"], [])
        self.assertEqual(self.run_batch(), result)

    def test_partial_return_preserved_but_never_accepted(self):
        self.initialize()
        result = self.run_batch(self.wall[:3])
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertEqual(len(result["attempted_pairs"]), 3)
        self.assertTrue(all(row["disposition"] == "rejected" for row in result["attempted_pairs"]))
        self.assertTrue(result["attempts"][0]["partial_pair_bytes_available"])

    def test_two_hour_shared_clock_and_absolute_cutoff(self):
        self.initialize()
        self.now += timedelta(hours=2)
        result = self.run_batch()
        self.assertEqual(result["terminal_reason"], "sampling_time_budget_exhausted")
        self.assertEqual(self.calls, [])
        other = Path(self.tmp.name) / "near_cutoff.json"
        config = identity()
        config["inference_cutoff_utc"] = (START + timedelta(minutes=5)).isoformat()
        config["campaign_deadline_utc"] = (START + timedelta(hours=6, minutes=5)).isoformat()
        result = initialize_sampling_plan(other, config, now_utc=START.isoformat())
        self.assertEqual(result["effective_sampling_cutoff_utc"], config["inference_cutoff_utc"])

    def test_backward_utc_and_monotonic_stall_rejected(self):
        self.initialize()
        self.now -= timedelta(seconds=1)
        result = self.run_batch()
        self.assertEqual(result["terminal_reason"], "clock_moved_backwards_or_stalled")
        self.assertEqual(self.calls, [])
        other = Path(self.tmp.name) / "stalled.json"
        with patch("src.activation_causality.confirmation_sampling.time.monotonic", return_value=100):
            initialize_sampling_plan(other, identity(), now_utc=START.isoformat())
        with patch("src.activation_causality.confirmation_sampling.time.monotonic", return_value=103):
            result = advance_sampling_plan(other, generator=lambda **kw: returned_fixture(self.wall[:32], kw), eligibility_builder=assess, now_utc=START.isoformat())
        self.assertEqual(result["terminal_reason"], "clock_moved_backwards_or_stalled")

    def test_deadline_after_eligibility_preserves_decisions_and_rejected_aliases(self):
        self.initialize()
        def crosses_deadline(candidate, pairs):
            self.now += timedelta(hours=2, seconds=1)
            result = assess(candidate, pairs)
            result["decisions"][0]["layout_aliases"].append("control_extra")
            return result
        result = self.run_batch(builder=crosses_deadline)
        self.assertEqual(result["terminal_reason"], "sampling_time_budget_exhausted")
        self.assertEqual(len(result["attempts"][0]["builder_assessment"]["decisions"]), 32)
        self.assertIn("control_extra", result["global_layout_aliases"])
        self.assertEqual(result["candidates"]["candidate0"]["accepted_n_layouts_by_family"]["wall"], 0)

    def test_identity_changes_tampering_and_missing_joint_evidence(self):
        self.initialize()
        config = identity()
        config["dataset_seed"] += 1
        with self.assertRaisesRegex(ValueError, "changed sampling identity"):
            self.initialize(config)
        with self.assertRaisesRegex(ValueError, "differs from caller"):
            advance_sampling_plan(self.path, generator=lambda **kw: [], eligibility_builder=assess,
                                  expected_identity_sha256="a" * 64, now_utc=self.clock)
        other = Path(self.tmp.name) / "joint.json"
        initialize_sampling_plan(other, identity(joint=True), now_utc=self.clock)
        def bad(candidate, pairs):
            result = assess(candidate, pairs)
            result["decisions"][0]["evidence"].pop("joint_renderer_audit")
            return result
        result = advance_sampling_plan(other, generator=lambda **kw: returned_fixture(self.wall[:32], kw), eligibility_builder=bad, now_utc=self.clock)
        self.assertEqual(result["status"], "terminal_incomplete")
        self.assertIn("joint renderer", result["attempts"][0]["error"])

    def test_duplicate_decision_aliases_and_omitted_controls_terminal(self):
        for mutation in (lambda d: d["layout_aliases"].append(d["layout_aliases"][0]),
                         lambda d: d.update(source_pair_sha256="f" * 64),
                         lambda d: d["evidence"]["adapter_manifest"].update(scheduled_conditions=[])):
            path = Path(self.tmp.name) / (str(id(mutation)) + ".json")
            initialize_sampling_plan(path, identity(), now_utc=self.clock)
            def bad(candidate, pairs):
                result = assess(candidate, pairs)
                mutation(result["decisions"][0])
                return result
            result = advance_sampling_plan(path, generator=lambda **kw: returned_fixture(self.wall[:32], kw), eligibility_builder=bad, now_utc=self.clock)
            self.assertEqual(result["status"], "terminal_incomplete")
            self.assertEqual(len(result["attempted_pairs"]), 32)
            self.assertEqual(result["candidates"]["candidate0"]["accepted_n_layouts_by_family"]["wall"], 0)

    def test_fixed_lexical_mixed_and_single_subsets(self):
        pairs = self.wall[:22] + self.key[:21] + self.door[:21]
        result = select_lexical_subset(pairs, families=FAMILIES)
        self.assertEqual(result["family_layout_counts"], {"wall": 11, "key_possession": 11, "door_status": 10})
        self.assertEqual(result["family_weights"], {"wall": 11 / 32, "key_possession": 11 / 32, "door_status": 10 / 32})
        expected_wall = sorted(self.wall[:22], key=lambda p: (hashlib.sha256((p["layout_id"] + "\n" + SALT).encode()).hexdigest(), p["layout_id"]))[:11]
        self.assertEqual(result["selected_pairs"][:11], expected_wall)
        self.assertEqual(select_lexical_subset(list(reversed(pairs)), families=FAMILIES)["selected_pairs"], result["selected_pairs"])
        single = select_lexical_subset(self.wall[:64], families=["wall"])
        self.assertEqual(single["family_weights"], {"wall": 1.0})
        self.assertEqual(single["n_layouts"], 32)
        self.assertEqual(single["additional_independent_layouts"], 0)
        with self.assertRaises(ValueError):
            select_lexical_subset(self.wall[:31], families=["wall"])
        with self.assertRaises(ValueError):
            select_lexical_subset(self.wall[:32] + [self.wall[0]], families=["wall"])

    def test_fixed_generator_geometry_and_historical_mutation(self):
        config = identity()
        history = Path(self.tmp.name) / "history.json"
        history.write_text('[]\n')
        config["generator_parameters"]["historical_steps"] = str(history)
        config["historical_exclusion"].update(source_path=str(history), source_sha256=file_sha256(history))
        self.initialize(config)
        result = self.run_batch()
        call = result["attempts"][0]["generator_call"]
        for key, value in config["generator_parameters"].items():
            self.assertEqual(call[key], value)
        history.write_text('[{"mutated":true}]\n')
        with self.assertRaisesRegex(ValueError, "historical source fingerprint"):
            self.initialize(config)
        changed = self.run_batch()
        self.assertEqual(changed["terminal_reason"], "historical_source_verification_failed")
        self.assertEqual(len(self.calls), 1)

    def test_generator_parameters_and_unchecked_history_rejected(self):
        for mutation in (lambda c: c["generator_parameters"].update(sizes=[7]),
                         lambda c: c["generator_parameters"].update(require_unique_actions=1),
                         lambda c: c["generator_parameters"].update(donor_mode="mismatch"),
                         lambda c: c["generator_parameters"].update(historical_steps=None),
                         lambda c: c["historical_exclusion"].update(status="not_checked"),
                         lambda c: c.update(previously_used_layout_aliases=["unbound"])):
            config = identity()
            mutation(config)
            with self.assertRaises(ValueError):
                self.initialize(config)

    def test_hundredth_attempt_is_terminal_without_seed_one_hundred(self):
        state = self.initialize()
        # Boundary fixture: production counter is advanced only by reservations.
        state["candidates"]["candidate0"]["next_attempt_by_family"]["wall"] = 99
        _save(self.path, state)
        def reject(candidate, pairs):
            return {"eligibility_builder_sha256": candidate["eligibility_builder_sha256"],
                    "decisions": [decision(pair, candidate, False) for pair in pairs]}
        result = self.run_batch(builder=reject)
        self.assertEqual(result["terminal_reason"], "outer_attempt_budget_exhausted")
        self.assertEqual(result["attempts"][0]["generator_call"]["seed"], 20261200)
        self.assertEqual(result["candidates"]["candidate0"]["next_attempt_by_family"]["wall"], 100)
        self.run_batch()
        self.assertEqual(len(self.calls), 1)

    def test_timer_interrupts_overlong_cpu_callback(self):
        import time
        with self.assertRaises(SamplingDeadlineExceeded):
            with _bounded(0.01):
                time.sleep(0.1)

    def test_mixed_three_family_route_fixed_quota_and_seed(self):
        config = identity()
        candidate = config["candidates"][0]
        quota = {"wall": 11, "key_possession": 11, "door_status": 10}
        candidate.update(families=list(FAMILIES), family_layout_counts=quota)
        candidate["configuration"].pop("family")
        candidate["configuration"].update(families=list(FAMILIES), family_layout_counts=quota)
        candidate["configuration_sha256"] = fingerprint(candidate["configuration"])
        self.initialize(config)
        for pairs in (self.wall[:11], self.key[:11], self.door[:10]):
            result = self.run_batch(pairs)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidates"]["candidate0"]["accepted_n_layouts_by_family"], quota)
        self.assertEqual([item["generator_call"]["seed"] for item in result["attempts"]], [20261101, 20262101, 20263101])

    def test_designated_contextual_adapter_complete_joint_plan(self):
        import time
        from test_activation_causality_frozen_contextual_input import Tokenizer, configuration
        from src.activation_causality.frozen_contextual_input import build_frozen_contextual_input_spec
        anchor = time.monotonic()
        self.clock = lambda: (START + timedelta(seconds=time.monotonic() - anchor)).isoformat()
        config = identity(joint=True)
        adapter_config = configuration(joint=True)
        candidate = config["candidates"][0]
        candidate.update(candidate_id=adapter_config["candidate_id"], source_experiment_id=adapter_config["source_experiment_id"],
                         source_spec_sha256=adapter_config["source_spec_sha256"], configuration=adapter_config,
                         configuration_sha256=fingerprint(adapter_config), families=list(FAMILIES),
                         family_layout_counts=deepcopy(adapter_config["family_layout_counts"]))
        self.initialize(config)
        def adapter(definition, pairs):
            built = build_frozen_contextual_input_spec(pairs, Tokenizer(), "inline_sampling_fixture",
                                                       definition["configuration"], "freeze_fixture")
            decisions = []
            for pair, manifest in zip(pairs, built["eligibility_manifest"], strict=True):
                aliases = built["layout_aliases"].get(pair["layout_id"],
                    manifest.get("control_eligibility", {}).get("layout_aliases", decision(pair, definition)["layout_aliases"]))
                decisions.append({"pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
                    "source_pair_sha256": fingerprint(pair), "eligible": manifest["eligible"], "reason": manifest["reason"],
                    "layout_aliases": aliases,
                    "evidence": {"adapter_manifest": manifest,
                        "required_condition_inventory": definition["configuration"]["condition_inventory"],
                        "joint_renderer_audit": manifest.get("control_eligibility", {}).get("joint_renderer_audit")}})
            return {"eligibility_builder_sha256": definition["eligibility_builder_sha256"], "decisions": decisions}
        def source(**call):
            pool = {"wall": self.wall, "key_possession": self.key, "door_status": self.door}[call["families"][0]]
            selected, seen = [], set(call["excluded_layout_ids"])
            for pair in pool:
                aliases = set(decision(pair, candidate)["layout_aliases"])
                if not aliases & seen:
                    selected.append(pair)
                    seen.update(aliases)
                    if len(selected) == call["n_layouts"]:
                        return returned_fixture(selected, call)
            raise RuntimeError("CPU fixture pool exhausted")
        for _ in range(6):
            result = self.run_batch(builder=adapter, generator=source)
            if result["status"] != "sampling":
                break
        self.assertEqual(result["status"], "complete", result["events"])
        self.assertEqual(sum(row["disposition"] == "accepted" for row in result["attempted_pairs"]), 32)
        self.assertEqual(result["candidates"]["selected_contextual"]["accepted_n_layouts_by_family"],
                         {"wall": 11, "key_possession": 11, "door_status": 10})
        self.assertTrue(all(row["builder_decision"]["evidence"]["joint_renderer_audit"]["eligible"]
                            for row in result["attempted_pairs"] if row["disposition"] == "accepted"))


if __name__ == "__main__":
    unittest.main()
