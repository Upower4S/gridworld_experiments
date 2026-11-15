"""Semantic ground truth and leakage boundaries for fresh counterfactuals."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from src.activation_causality.designs import fact_sentence, sentence_condition_specs, state_control_specs
from src.activation_causality.environments import (
    ACTIONS, DELTAS, apply_variable, canonical_layout_fingerprint, generate_pairs, make_state,
    optimal_actions, parse_indexed_grid, partition_for_layout, shortest_path, step,
)


class _Cell:
    """MiniGrid world-object properties read by the unchanged upstream step."""
    def __init__(self, kind):
        self.type = kind
        self.is_locked = kind == "door"
        self.is_open = False

    def can_overlap(self):
        return self.type == "goal" or (self.type == "door" and self.is_open)


class _UpstreamHarness:
    """Minimal storage/observation adapter, not a second transition algorithm."""
    def __init__(self, state):
        mapping = {"#": "wall", "D": "door", "K": "key", "G": "goal"}
        self.cells = [[_Cell(mapping[ch]) if ch in mapping else None for ch in row] for row in state["grid"]]
        self.grid = SimpleNamespace(get=lambda x, y: self.cells[y][x], set=self._set)
        self.agent_pos = tuple(state["agent"][::-1])
        self.goal_pos = next((c, r) for r, row in enumerate(state["grid"]) for c, ch in enumerate(row) if ch == "G")
        self.carrying = _Cell("key") if state["has_key"] else None
        self._action_to_direction = {action: action for action in ACTIONS}
        self.agent_dir = "UP"
        self.step_count = 0
        self.max_steps = 10000
        self.allow_quit_action = False

    def _set(self, x, y, value):
        self.cells[y][x] = value

    @property
    def front_pos(self):
        dr, dc = DELTAS[self.agent_dir]
        return self.agent_pos[0] + dc, self.agent_pos[1] + dr

    def gen_obs(self):
        return {}

    def _reward(self):
        return 1.0

    def state(self):
        symbols = {"wall": "#", "door": "D", "key": "K", "goal": "G"}
        grid = ["".join("_" if cell is None else symbols[cell.type] for cell in row) for row in self.cells]
        return make_state(grid, self.agent_pos[::-1], self.carrying is not None)


class GridSemanticsTests(unittest.TestCase):
    def test_pickup_opens_adjacent_door_and_removes_objects(self):
        state = make_state(["#######", "#AKDG_#", "#######"])
        self.assertEqual(shortest_path(state), (3, ["RIGHT", "RIGHT", "RIGHT"]))
        after = step(state, "RIGHT")
        self.assertTrue(after["has_key"])
        self.assertEqual(after["grid"][1], "#___G_#")
        self.assertEqual(after["agent"], [1, 2])
        self.assertEqual(state["grid"][1], "#_KDG_#")

    def test_blocked_attempt_can_open_door_but_not_enter_it_immediately(self):
        state = make_state(["#####", "#ADG#", "#####"], has_key=True)
        after = step(state, "RIGHT")
        self.assertEqual(after["agent"], [1, 1])
        self.assertEqual(after["grid"][1], "#__G#")
        self.assertEqual(shortest_path(state)[0], 3)

    def test_unreachable_and_equal_length_first_moves(self):
        self.assertEqual(shortest_path(make_state(["#####", "#A#G#", "#####"]))[0], None)
        state = make_state(["#####", "#A__#", "#_G_#", "#####"])
        self.assertEqual(optimal_actions(state), ["DOWN", "RIGHT"])
        self.assertEqual(shortest_path(state)[0], 2)

    def test_bounds_prevent_negative_index_wrapping(self):
        state = make_state(["A_G"])
        self.assertEqual(step(state, "LEFT"), state)
        self.assertEqual(step(state, "UP"), state)

    def test_open_door_persists_after_leaving(self):
        state = make_state(["#######", "#AKD_G#", "#_____#", "#######"])
        state = step(step(state, "RIGHT"), "DOWN")
        self.assertEqual(state["grid"][1][3], "_")
        self.assertTrue(state["has_key"])

    def test_100_reachable_states_against_unmodified_upstream_step(self):
        """400 comparisons execute the actual upstream method, without minigrid."""
        source_path = Path(__file__).resolve().parents[1] / "interp/reveng/src/reveng/environment_generator/rooms_minigrid.py"
        tree = ast.parse(source_path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RoomsMinigridEnv")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
        upstream_step = namespace["step"]
        rng = random.Random(1618033)
        initial = make_state(["#########", "#A_KD__G#", "#_#_#_#_#", "#___D___#", "#_______#", "#########"])
        pool = [initial]
        state = initial
        # Deterministically include the key/door transitions before random walks.
        for action in ["RIGHT", "RIGHT", "DOWN", "DOWN", "RIGHT"]:
            state = step(state, action)
            pool.append(state)
        for _ in range(500):
            state = step(state, rng.choice(ACTIONS))
            if state["grid"][state["agent"][0]][state["agent"][1]] == "G":
                state = deepcopy(initial)
            pool.append(state)
        snapshots = pool[:6] + rng.sample(pool[6:], 94)
        self.assertTrue(any(s["has_key"] for s in snapshots))
        self.assertTrue(any(not s["has_key"] for s in snapshots))
        self.assertTrue(any(sum(row.count("D") for row in s["grid"]) == 0 for s in snapshots))
        for index, snapshot in enumerate(snapshots):
            self.assertIsNotNone(shortest_path(snapshot)[0])
            for action in ACTIONS:
                with self.subTest(snapshot=index, action=action):
                    harness = _UpstreamHarness(snapshot)
                    upstream_step(harness, action)
                    self.assertEqual(harness.state(), step(snapshot, action))
                    self.assertEqual(harness.step_count, 1)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = generate_pairs(6, 73021, "all", donor_mode="mixed")

    def test_deterministic_and_json_serializable(self):
        self.assertEqual(self.pairs, generate_pairs(6, 73021, "all", donor_mode="mixed"))
        self.assertEqual(json.loads(json.dumps(self.pairs)), self.pairs)

    def test_symmetry_and_landmark_independent_layout_identity(self):
        grid = ["#####", "#A#G#", "#K_D#", "#####"]
        fingerprint = canonical_layout_fingerprint(grid)
        rotated = ["".join(row) for row in zip(*grid[::-1])]
        self.assertEqual(canonical_layout_fingerprint(rotated), fingerprint)
        self.assertEqual(canonical_layout_fingerprint([row[::-1] for row in grid]), fingerprint)
        plain = [row.replace("A", "_").replace("K", "_").replace("D", "_").replace("G", "_") for row in grid]
        self.assertEqual(canonical_layout_fingerprint(plain), fingerprint)

    def test_exact_counterfactual_and_ground_truth_disagreement(self):
        self.assertEqual({pair["family"] for pair in self.pairs}, {"wall", "key_possession", "door_status"})
        for pair in self.pairs:
            cell = pair["intervention"]["cell"]
            self.assertEqual(apply_variable(pair["recipient"], pair["donor"], pair["family"], cell), pair["counterfactual"])
            self.assertEqual(len(pair["recipient_optimal_actions"]), 1)
            self.assertEqual(len(pair["counterfactual_optimal_actions"]), 1)
            self.assertNotEqual(pair["recipient_optimal_actions"], pair["counterfactual_optimal_actions"])
            self.assertIsNone(pair["metadata"]["model_donor_action_mismatch"])
            self.assertEqual(pair["partition"], partition_for_layout(pair["layout_id"]))

    def test_family_partition_and_explicit_historical_exclusion(self):
        excluded = {pair["layout_id"] for pair in self.pairs}
        fresh = generate_pairs(2, 73021, "all", excluded_layout_ids=excluded)
        self.assertTrue(excluded.isdisjoint(pair["layout_id"] for pair in fresh))
        for pair in self.pairs:
            self.assertEqual(pair["metadata"]["historical_exclusion"]["status"], "not_checked")
        heldout = generate_pairs(2, 991, "confirmation")
        self.assertTrue(all(p["partition"] == "confirmation" for p in heldout))

    def test_mismatch_is_about_counterfactual_recipient_not_donor(self):
        mismatch = [p for p in self.pairs if p["metadata"]["ground_truth_donor_action_mismatch"]]
        self.assertEqual(len(mismatch), 3)
        for pair in mismatch:
            self.assertNotEqual(pair["donor"]["agent"], pair["recipient"]["agent"])
            self.assertTrue(set(pair["donor_optimal_actions"]).isdisjoint(pair["counterfactual_optimal_actions"]))

    def test_sentence_controls_are_not_claimed_as_observations(self):
        for pair in self.pairs:
            conditions = {row["condition"]: row for row in sentence_condition_specs(pair)}
            self.assertTrue(conditions["recipient_fact"]["fact_true_in_recipient"])
            self.assertFalse(conditions["donor_fact"]["fact_true_in_recipient"])
            self.assertTrue(all(row["observed_action"] is None for row in conditions.values()))
            self.assertNotEqual(fact_sentence(pair), fact_sentence(pair, source="donor"))
            self.assertNotIn("<|", pair["recipient_prompt_text"])

    def test_invalid_interchange_does_not_overwrite_other_content(self):
        pair = deepcopy(self.pairs[0])
        r, c = pair["intervention"]["cell"]
        row = list(pair["donor"]["grid"][r])
        row[c] = "G"
        pair["donor"]["grid"][r] = "".join(row)
        with self.assertRaises(ValueError):
            apply_variable(pair["recipient"], pair["donor"], "wall", [r, c])

    def test_executable_specificity_controls_preserve_claimed_variables(self):
        for pair in self.pairs:
            controls = {row["condition"]: row for row in state_control_specs(pair)}
            natural = controls["natural_donor"]
            same = controls["same_fact_donor"]
            if natural["eligible"]:
                self.assertTrue(same["eligible"])
                self.assertNotEqual(natural["state"]["agent"], pair["recipient"]["agent"])
                self.assertEqual(natural["state"]["agent"], same["state"]["agent"])
                self.assertTrue(set(natural["optimal_actions"]).isdisjoint(pair["counterfactual_optimal_actions"]))
                self.assertEqual(apply_variable(natural["state"], pair["recipient"], pair["family"], pair["intervention"]["cell"]), same["state"])
                self.assertIsNone(natural["observed_donor_action_mismatch"])
            irrelevant = controls["irrelevant_fact_donor"]
            if irrelevant["eligible"]:
                state = irrelevant["state"]
                self.assertEqual(shortest_path(state)[0], shortest_path(pair["recipient"])[0])
                self.assertEqual(optimal_actions(state), pair["recipient_optimal_actions"])
                cf = apply_variable(state, pair["donor"], pair["family"], pair["intervention"]["cell"])
                self.assertEqual(shortest_path(cf)[0], shortest_path(pair["counterfactual"])[0])
                self.assertEqual(optimal_actions(cf), pair["counterfactual_optimal_actions"])
            self.assertEqual(json.loads(json.dumps(list(controls.values()))), list(controls.values()))

    def test_historical_parser_reads_only_indexed_rows(self):
        self.assertEqual(parse_indexed_grid("  0 1 2\n0 # # #\n1 A _ G\n2 # # #"), ["###", "A_G", "###"])
        with self.assertRaises(ValueError):
            parse_indexed_grid("malformed")


if __name__ == "__main__":
    unittest.main()
