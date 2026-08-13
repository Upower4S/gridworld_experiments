"""Gridworld task concepts for conceptual belief space analyses."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ACTIONS

ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
GRID_SYMBOLS = {"#", "_", "A", "G", "K", "D"}
ACTION_DERIVED_CONCEPTS = [
    "move_toward_goal",
    "move_away_from_goal",
    "horizontal_move",
    "vertical_move",
    "move_toward_key",
    "move_toward_door",
    "wall_blocked_action",
    "shortest_path_action",
    "detour_action",
    "valid_action",
    "invalid_action",
]
STATE_CONCEPTS = [
    "door_relevant",
    "key_required",
    "goal_reachable_without_key",
    "door_requires_key",
]


@dataclass(frozen=True)
class ParsedGrid:
    grid: tuple[tuple[str, ...], ...]
    height: int
    width: int
    agent: tuple[int, int] | None
    goal: tuple[int, int] | None
    key: tuple[int, int] | None
    door: tuple[int, int] | None
    walls: frozenset[tuple[int, int]]


def action_concepts() -> list[str]:
    return list(ACTION_DERIVED_CONCEPTS)


def state_concepts() -> list[str]:
    return list(STATE_CONCEPTS)


def action_mask_columns() -> list[str]:
    return [f"env_{concept}_{action}" for concept in ACTION_DERIVED_CONCEPTS for action in ACTIONS]


def model_belief_columns() -> list[str]:
    return [f"model_belief_{concept}" for concept in ACTION_DERIVED_CONCEPTS]


def parse_grid_state_text(text: Any) -> ParsedGrid | None:
    """Parse coordinate-labeled grid text into symbols.

    The canonical grid strings include one header row with column coordinates and
    subsequent rows of the form `row_index symbol symbol ...`.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        tokens = raw_line.strip().split()
        if not tokens:
            continue
        symbols = [token for token in tokens if token in GRID_SYMBOLS]
        if symbols:
            rows.append(symbols)
    if not rows:
        return None
    width = max(len(row) for row in rows)
    if any(len(row) != width for row in rows):
        return None

    agent = goal = key = door = None
    walls: set[tuple[int, int]] = set()
    for r, row in enumerate(rows):
        for c, symbol in enumerate(row):
            if symbol == "A" and agent is None:
                agent = (r, c)
            elif symbol == "G" and goal is None:
                goal = (r, c)
            elif symbol == "K" and key is None:
                key = (r, c)
            elif symbol == "D" and door is None:
                door = (r, c)
            elif symbol == "#":
                walls.add((r, c))
    return ParsedGrid(
        grid=tuple(tuple(row) for row in rows),
        height=len(rows),
        width=width,
        agent=agent,
        goal=goal,
        key=key,
        door=door,
        walls=frozenset(walls),
    )


def in_bounds(grid: ParsedGrid, pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < grid.height and 0 <= c < grid.width


def symbol_at(grid: ParsedGrid, pos: tuple[int, int]) -> str | None:
    if not in_bounds(grid, pos):
        return None
    r, c = pos
    return grid.grid[r][c]


def manhattan(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return float("nan")
    return float(abs(left[0] - right[0]) + abs(left[1] - right[1]))


def next_position(pos: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return pos[0] + dr, pos[1] + dc


def is_valid_next_cell(symbol: str | None, carrying_key: bool) -> bool:
    if symbol is None or symbol == "#":
        return False
    if symbol == "D" and not carrying_key:
        return False
    return True


def bfs_first_actions(
    grid: ParsedGrid,
    *,
    carrying_key: bool,
    allow_key_pickup: bool = True,
    door_passable_with_key: bool = True,
) -> tuple[set[str], float]:
    """Return all first actions on shortest paths to the goal."""
    if grid.agent is None or grid.goal is None:
        return set(), float("nan")
    start_state = (grid.agent, bool(carrying_key), None)
    queue: deque[tuple[tuple[int, int], bool, str | None, int]] = deque(
        [(start_state[0], start_state[1], start_state[2], 0)]
    )
    visited = {(grid.agent, bool(carrying_key))}
    best_distance: int | None = None
    first_actions: set[str] = set()

    while queue:
        pos, has_key, first_action, distance = queue.popleft()
        if best_distance is not None and distance > best_distance:
            continue
        if pos == grid.goal:
            best_distance = distance
            if first_action is not None:
                first_actions.add(first_action)
            continue
        for action in ACTIONS:
            nxt = next_position(pos, action)
            symbol = symbol_at(grid, nxt)
            valid = is_valid_next_cell(
                symbol,
                carrying_key=bool(has_key and door_passable_with_key),
            )
            if not valid:
                continue
            next_has_key = bool(has_key or (allow_key_pickup and symbol == "K"))
            state = (nxt, next_has_key)
            if state in visited:
                continue
            visited.add(state)
            queue.append((nxt, next_has_key, first_action or action, distance + 1))
    return first_actions, float(best_distance) if best_distance is not None else float("nan")


def reachable_without_key(grid: ParsedGrid) -> bool:
    first_actions, distance = bfs_first_actions(
        grid,
        carrying_key=False,
        allow_key_pickup=False,
        door_passable_with_key=False,
    )
    return bool(first_actions or distance == 0)


def empty_action_masks() -> dict[str, float | int | bool]:
    out: dict[str, float | int | bool] = {}
    for concept in ACTION_DERIVED_CONCEPTS:
        for action in ACTIONS:
            value = np.nan
            if concept == "horizontal_move":
                value = action in {"LEFT", "RIGHT"}
            elif concept == "vertical_move":
                value = action in {"UP", "DOWN"}
            out[f"env_{concept}_{action}"] = value
    return out


def compute_action_masks(
    grid: ParsedGrid | None,
    *,
    carrying_key: bool,
) -> tuple[dict[str, float | int | bool], dict[str, Any]]:
    """Compute deterministic environment concepts per possible action."""
    masks = empty_action_masks()
    state: dict[str, Any] = {
        "has_grid_state": grid is not None,
        "parse_status": "ok" if grid is not None else "missing_or_unparseable_grid",
        "carrying_key": bool(carrying_key),
        "grid_height": np.nan,
        "grid_width": np.nan,
        "agent_row": np.nan,
        "agent_col": np.nan,
        "goal_row": np.nan,
        "goal_col": np.nan,
        "key_row": np.nan,
        "key_col": np.nan,
        "door_row": np.nan,
        "door_col": np.nan,
        "door_present": False,
        "key_present": False,
        "goal_reachable_without_key": np.nan,
        "door_requires_key": np.nan,
        "key_required": np.nan,
        "door_relevant": np.nan,
        "shortest_path_length": np.nan,
        "shortest_path_actions_json": "[]",
        "num_shortest_path_actions": 0,
    }
    if grid is None:
        return masks, state

    state.update(
        {
            "grid_height": int(grid.height),
            "grid_width": int(grid.width),
            "agent_row": grid.agent[0] if grid.agent else np.nan,
            "agent_col": grid.agent[1] if grid.agent else np.nan,
            "goal_row": grid.goal[0] if grid.goal else np.nan,
            "goal_col": grid.goal[1] if grid.goal else np.nan,
            "key_row": grid.key[0] if grid.key else np.nan,
            "key_col": grid.key[1] if grid.key else np.nan,
            "door_row": grid.door[0] if grid.door else np.nan,
            "door_col": grid.door[1] if grid.door else np.nan,
            "door_present": grid.door is not None,
            "key_present": grid.key is not None,
        }
    )

    first_actions, shortest_length = bfs_first_actions(grid, carrying_key=carrying_key)
    no_key_reachable = reachable_without_key(grid)
    door_requires_key = bool(grid.door is not None and not no_key_reachable)
    key_required = bool((not carrying_key) and grid.key is not None and door_requires_key and np.isfinite(shortest_length))
    door_relevant = bool(grid.door is not None and (door_requires_key or carrying_key))
    state.update(
        {
            "goal_reachable_without_key": bool(no_key_reachable),
            "door_requires_key": door_requires_key,
            "key_required": key_required,
            "door_relevant": door_relevant,
            "shortest_path_length": shortest_length,
            "shortest_path_actions_json": json.dumps(sorted(first_actions)),
            "num_shortest_path_actions": int(len(first_actions)),
        }
    )

    if grid.agent is None:
        return masks, state

    current_goal_dist = manhattan(grid.agent, grid.goal)
    current_key_dist = manhattan(grid.agent, grid.key)
    current_door_dist = manhattan(grid.agent, grid.door)
    for action in ACTIONS:
        nxt = next_position(grid.agent, action)
        symbol = symbol_at(grid, nxt)
        valid = is_valid_next_cell(symbol, carrying_key=carrying_key)
        wall_blocked = symbol == "#" or symbol is None
        next_goal_dist = manhattan(nxt, grid.goal)
        next_key_dist = manhattan(nxt, grid.key)
        next_door_dist = manhattan(nxt, grid.door)

        masks[f"env_valid_action_{action}"] = bool(valid)
        masks[f"env_invalid_action_{action}"] = not bool(valid)
        masks[f"env_wall_blocked_action_{action}"] = bool(wall_blocked)
        masks[f"env_shortest_path_action_{action}"] = action in first_actions
        masks[f"env_detour_action_{action}"] = bool(valid and action not in first_actions)
        masks[f"env_move_toward_goal_{action}"] = bool(valid and np.isfinite(next_goal_dist) and next_goal_dist < current_goal_dist)
        masks[f"env_move_away_from_goal_{action}"] = bool(valid and np.isfinite(next_goal_dist) and next_goal_dist > current_goal_dist)
        masks[f"env_move_toward_key_{action}"] = bool(valid and grid.key is not None and np.isfinite(next_key_dist) and next_key_dist < current_key_dist)
        masks[f"env_move_toward_door_{action}"] = bool(valid and grid.door is not None and np.isfinite(next_door_dist) and next_door_dist < current_door_dist)
    return masks, state


def concept_value_for_action(row: pd.Series, concept: str, action: str | None) -> float:
    if action not in ACTIONS:
        return float("nan")
    value = row.get(f"env_{concept}_{action}", np.nan)
    if pd.isna(value):
        return float("nan")
    return float(value)


def add_action_selected_concepts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for concept in ACTION_DERIVED_CONCEPTS:
        out[f"env_agent_action_{concept}"] = [
            concept_value_for_action(row, concept, str(row.get("agent_action")))
            for _, row in out.iterrows()
        ]
        if "top_action" in out.columns:
            out[f"env_top_action_{concept}"] = [
                concept_value_for_action(row, concept, str(row.get("top_action")))
                for _, row in out.iterrows()
            ]
    return out


def add_model_concept_beliefs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for concept in ACTION_DERIVED_CONCEPTS:
        score = np.zeros(len(out), dtype=float)
        valid_any = np.zeros(len(out), dtype=bool)
        for action in ACTIONS:
            q_col = f"q_{action}"
            mask_col = f"env_{concept}_{action}"
            if q_col not in out.columns or mask_col not in out.columns:
                continue
            q = pd.to_numeric(out[q_col], errors="coerce").to_numpy(dtype=float)
            mask = pd.to_numeric(out[mask_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(q) & np.isfinite(mask)
            score[finite] += q[finite] * mask[finite]
            valid_any |= finite
        score[~valid_any] = np.nan
        out[f"model_belief_{concept}"] = score
    return out


def build_step_concept_table(steps: pd.DataFrame) -> pd.DataFrame:
    """Build one deterministic concept row per trajectory step."""
    required = {"trajectory_id", "step_id"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"step table missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in steps.itertuples(index=False):
        row_dict = row._asdict()
        grid = parse_grid_state_text(row_dict.get("grid_state_text"))
        carrying_key = bool(row_dict.get("carrying_key", False))
        masks, state = compute_action_masks(grid, carrying_key=carrying_key)
        out = {
            "trajectory_id": str(row_dict["trajectory_id"]),
            "step_id": row_dict["step_id"],
            "canonical_agent_action": row_dict.get("agent_action"),
        }
        out.update(state)
        out.update(masks)
        rows.append(out)
    return pd.DataFrame(rows)


def join_concepts_to_beliefs(
    beliefs: pd.DataFrame,
    step_concepts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join deterministic concepts to prefix beliefs and add model concept scores."""
    out = beliefs.copy()
    if step_concepts is None or step_concepts.empty:
        static = empty_action_masks()
        for key, value in static.items():
            out[key] = value
        out["has_grid_state"] = False
        out["parse_status"] = "missing_step_concepts"
    else:
        join_cols = ["trajectory_id", "step_id"]
        missing = [col for col in join_cols if col not in out.columns or col not in step_concepts.columns]
        if missing:
            raise ValueError(f"cannot join beliefs and step concepts; missing columns: {missing}")
        out = out.merge(step_concepts, how="left", on=join_cols, validate="many_to_one")
        out["has_grid_state"] = out["has_grid_state"].fillna(False)
        out["parse_status"] = out["parse_status"].fillna("missing_step_join")
        for key, value in empty_action_masks().items():
            if key not in out.columns:
                out[key] = value
    out = add_action_selected_concepts(out)
    out = add_model_concept_beliefs(out)
    return out


def concept_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for concept in ACTION_DERIVED_CONCEPTS:
        model_col = f"model_belief_{concept}"
        agent_col = f"env_agent_action_{concept}"
        top_col = f"env_top_action_{concept}"
        rows.append(
            {
                "concept": concept,
                "concept_type": "action_derived",
                "mean_model_belief": float(pd.to_numeric(scores.get(model_col), errors="coerce").mean()) if model_col in scores else np.nan,
                "mean_agent_action_value": float(pd.to_numeric(scores.get(agent_col), errors="coerce").mean()) if agent_col in scores else np.nan,
                "mean_top_action_value": float(pd.to_numeric(scores.get(top_col), errors="coerce").mean()) if top_col in scores else np.nan,
                "mean_deterministic_state_value": np.nan,
            }
        )
    for concept in STATE_CONCEPTS:
        col = concept
        rows.append(
            {
                "concept": concept,
                "concept_type": "deterministic_state",
                "mean_model_belief": np.nan,
                "mean_agent_action_value": np.nan,
                "mean_top_action_value": np.nan,
                "mean_deterministic_state_value": float(pd.to_numeric(scores.get(col), errors="coerce").mean()) if col in scores else np.nan,
            }
        )
    return pd.DataFrame(rows)
