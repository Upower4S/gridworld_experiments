"""Fresh, reproducible grid counterfactuals; no model or upstream imports.

Coordinates are (row, column). Movement, automatic key pickup, and opening all
adjacent locked doors *after* each attempted move match ``RoomsMinigridEnv.step`` in the
read-only upstream ``rooms_minigrid.py``. An opened door disappears. A blocked
move still consumes a step and can open an adjacent door.

``grid`` contains terrain (# _ G K D), with the agent stored separately. Layout
identity conservatively ignores landmarks and inventory and canonicalizes wall
topology under all eight rotations/reflections. All variants of one pair inherit
the open-variant layout identity and its immutable partition.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
FAMILIES = ("wall", "key_possession", "door_status")
SCHEMA_VERSION = "grid_counterfactual_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(grid: Any) -> list[str]:
    return [row if isinstance(row, str) else "".join(row) for row in grid]


def make_state(grid: Iterable[Any], agent: Iterable[int] | None = None, has_key: bool = False) -> dict[str, Any]:
    """Build and validate a JSON state; accepts an optional A in input terrain."""
    rows = _rows(grid)
    positions = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "A"]
    if agent is None:
        if len(positions) != 1:
            raise ValueError("Supply agent coordinates or exactly one A")
        agent = positions[0]
    elif positions and (len(positions) != 1 or list(positions[0]) != list(agent)):
        raise ValueError("A disagrees with explicit agent coordinates")
    state = {"schema_version": SCHEMA_VERSION, "grid": [row.replace("A", "_") for row in rows],
             "agent": list(agent), "has_key": bool(has_key)}
    validate_state(state)
    return state


def validate_state(state: dict[str, Any], *, require_reachable: bool = False) -> None:
    rows = _rows(state["grid"])
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Grid must be a non-empty rectangle")
    if any(ch not in "#_GKD" for row in rows for ch in row):
        raise ValueError("Unknown terrain symbol")
    if sum(row.count("G") for row in rows) != 1:
        raise ValueError("Exactly one goal is required")
    if not isinstance(state.get("has_key"), bool):
        raise ValueError("has_key must be Boolean")
    agent = state.get("agent", [])
    if len(agent) != 2 or any(not isinstance(x, int) or isinstance(x, bool) for x in agent):
        raise ValueError("agent must contain two integer coordinates")
    r, c = agent
    if not (0 <= r < len(rows) and 0 <= c < len(rows[0])) or rows[r][c] in "#D":
        raise ValueError("Agent must occupy a traversable in-bounds cell")
    if rows[r][c] == "K":
        raise ValueError("A state after a step cannot leave a key under the agent")
    if require_reachable and shortest_path(state)[0] is None:
        raise ValueError("Goal is unreachable")


def _key(state: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(state["grid"]), tuple(state["agent"]), state["has_key"]


def step(state: dict[str, Any], action: str) -> dict[str, Any]:
    """Exact deterministic transition, including blocked-move door opening."""
    if action not in DELTAS:
        raise ValueError(f"Unknown action {action!r}")
    rows = [list(row) for row in state["grid"]]
    r, c = state["agent"]
    if rows[r][c] == "G":
        return deepcopy(state)
    dr, dc = DELTAS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < len(rows) and 0 <= nc < len(rows[0]) and rows[nr][nc] not in "#D":
        r, c = nr, nc
    has_key = state["has_key"]
    if rows[r][c] == "K":
        rows[r][c] = "_"
        has_key = True
    if has_key:
        for dr, dc in DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(rows) and 0 <= nc < len(rows[0]) and rows[nr][nc] == "D":
                rows[nr][nc] = "_"
    return {"schema_version": SCHEMA_VERSION, "grid": ["".join(row) for row in rows],
            "agent": [r, c], "has_key": has_key}


def shortest_path(state: dict[str, Any]) -> tuple[int | None, list[str]]:
    """Return shortest number of attempted moves and a deterministic witness."""
    queue = deque([(state, [])])
    seen = {_key(state)}
    while queue:
        current, path = queue.popleft()
        r, c = current["agent"]
        if current["grid"][r][c] == "G":
            return len(path), path
        for action in ACTIONS:
            nxt = step(current, action)
            key = _key(nxt)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, path + [action]))
    return None, []


def optimal_actions(state: dict[str, Any]) -> list[str]:
    """All shortest-path first actions, including ties; [] if terminal/unreachable."""
    distance, _ = shortest_path(state)
    if distance in (None, 0):
        return []
    return [action for action in ACTIONS if shortest_path(step(state, action))[0] == distance - 1]


def _rotate(rows: list[str]) -> list[str]:
    return ["".join(row) for row in zip(*rows[::-1])]


def canonical_layout_fingerprint(state_or_grid: dict[str, Any] | Iterable[Any], *,
                                 mutable_cells: Iterable[Iterable[int]] = ()) -> str:
    """D4-invariant wall topology hash, intentionally ignoring agent/goal/key/door.

    Pass the same mutable wall cell for both members to obtain a common family
    identity. Historical exclusions should also check each unmasked variant.
    """
    grid = state_or_grid["grid"] if isinstance(state_or_grid, dict) else state_or_grid
    rows = ["".join("#" if ch == "#" else "_" for ch in row) for row in _rows(grid)]
    cells = [list(row) for row in rows]
    for r, c in mutable_cells:
        cells[r][c] = "_"
    rows = ["".join(row) for row in cells]
    alternatives = []
    for _ in range(4):
        alternatives.append(tuple(rows))
        alternatives.append(tuple(row[::-1] for row in rows))
        rows = _rotate(rows)
    return _digest(min(alternatives))


def partition_for_layout(layout_id: str) -> str:
    """70/15/15 discovery/dev/confirmation, fixed independently of run seed."""
    bucket = int(hashlib.sha256(("activation-causality-split-v1:" + layout_id).encode()).hexdigest()[:16], 16) % 100
    return "discovery" if bucket < 70 else "dev" if bucket < 85 else "confirmation"


def parse_indexed_grid(text: str) -> list[str]:
    """Parse canonical_steps' indexed display, never its prompt's example grid."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[0].isdigit() and not all(x.isdigit() for x in fields[1:]):
            cells = fields[1:]
            if any(len(cell) != 1 or cell not in "#_AGKDO" for cell in cells):
                raise ValueError("Unsupported historical grid symbol")
            rows.append("".join(cells).replace("O", "_"))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Missing or malformed indexed grid")
    return rows


def load_historical_layout_ids(path: str | Path) -> tuple[set[str], dict[str, Any]]:
    """Read only grid_state_text from canonical steps; fail closed on bad rows.

    PyArrow is optional except for this explicit historical-exclusion operation.
    It never reads token, prompt, activation, or behavior columns.
    """
    import pyarrow.parquet as pq

    source = Path(path)
    fingerprints: set[str] = set()
    count = 0
    for batch in pq.ParquetFile(source).iter_batches(columns=["grid_state_text"], batch_size=512):
        for row in batch.to_pylist():
            fingerprints.add(canonical_layout_fingerprint(parse_indexed_grid(row["grid_state_text"])))
            count += 1
    return fingerprints, {"source": str(source), "rows_checked": count,
                          "layout_fingerprints": len(fingerprints), "status": "complete"}


def render_grid(state: dict[str, Any]) -> str:
    rows = [list(row) for row in state["grid"]]
    r, c = state["agent"]
    rows[r][c] = "A"
    width = max(2, len(str(max(len(rows), len(rows[0])) - 1)) + 1)
    header = " " * width + "".join(f"{c:<{width}}" for c in range(len(rows[0]))).rstrip()
    return "\n".join([header] + [f"{r:<{width}}" + "".join(f"{ch:<{width}}" for ch in row).rstrip()
                                  for r, row in enumerate(rows)])


def prompt_text(state: dict[str, Any]) -> str:
    """Human user message compatible with the original task; no chat markers."""
    return ("# Instructions\n\nYou are controlling an agent in a grid-based environment with full observability. "
            "Move from A to G using the fewest steps possible. You can move UP, DOWN, LEFT, or RIGHT.\n\n"
            "Legend: # = wall; _ = open space; G = goal; A = agent; D = locked door; K = key.\n\n"
            "Key and door mechanics:\n"
            "- Moving onto K automatically picks up the key and removes K from the grid.\n"
            "- After each attempted move, carrying the key automatically opens and removes all orthogonally adjacent locked doors.\n"
            "- Walls and locked doors prevent movement. A blocked attempt still takes one step.\n"
            "- Coordinates are (row, column); UP decreases the row and RIGHT increases the column.\n\n"
            "Your final answer must be only a JSON object of the form {\"action\": \"UP\"}, "
            "with one of UP, DOWN, LEFT, RIGHT.\n\n# Inputs\n\nCurrent grid state:\n\n"
            + render_grid(state) + "\n\nAgent status:\n- Carrying key: " + str(state["has_key"]))


def apply_variable(recipient: dict[str, Any], donor: dict[str, Any], family: str,
                   cell: Iterable[int] | None = None) -> dict[str, Any]:
    """Only the named donor variable enters the recipient counterfactual."""
    result = deepcopy(recipient)
    if family == "key_possession":
        result["has_key"] = donor["has_key"]
    elif family in ("wall", "door_status") and cell is not None:
        r, c = cell
        allowed = "#_" if family == "wall" else "D_"
        value = donor["grid"][r][c]
        if recipient["grid"][r][c] not in allowed or value not in allowed:
            raise ValueError("Intervention is not a valid binary terrain variable")
        rows = [list(row) for row in result["grid"]]
        rows[r][c] = value
        result["grid"] = ["".join(row) for row in rows]
    else:
        raise ValueError("Unknown family or missing intervention cell")
    validate_state(result)
    return result


def _candidate(rng: random.Random, family: str, size: int, density: float) -> tuple[dict[str, Any], dict[str, Any], list[int]] | None:
    rows = [["#" if r in (0, size - 1) or c in (0, size - 1) or rng.random() < density else "_"
             for c in range(size)] for r in range(size)]
    floors = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if rows[r][c] == "_"]
    if len(floors) < 6:
        return None
    agent, goal = rng.sample(floors, 2)
    if sum(abs(a - b) for a, b in zip(agent, goal)) < 3:
        return None
    rows[goal[0]][goal[1]] = "G"
    open_state = make_state(rows, agent)
    distance, path = shortest_path(open_state)
    if distance is None or distance < 3:
        return None
    # A path-derived candidate keeps rejection sampling practical without using model outcomes.
    cursor = list(agent)
    route_cells = []
    for action in path[:-1]:
        dr, dc = DELTAS[action]
        cursor = [cursor[0] + dr, cursor[1] + dc]
        if family != "key_possession" or sum(abs(a - b) for a, b in zip(cursor, agent)) > 1:
            route_cells.append(cursor[:])
    if not route_cells:
        return None
    cell = rng.choice(route_cells)
    r, c = cell
    blocked = deepcopy(open_state)
    chars = list(blocked["grid"][r])
    chars[c] = "#" if family == "wall" else "D"
    blocked["grid"][r] = "".join(chars)
    if family == "key_possession":
        counterfactual = deepcopy(blocked)
        counterfactual["has_key"] = True
    else:
        counterfactual = open_state
    return blocked, counterfactual, cell


def _mismatch_donor(counterfactual: dict[str, Any], target: list[str], rng: random.Random) -> dict[str, Any] | None:
    positions = [(r, c) for r, row in enumerate(counterfactual["grid"]) for c, ch in enumerate(row) if ch == "_"]
    rng.shuffle(positions)
    for pos in positions:
        if list(pos) == counterfactual["agent"]:
            continue
        candidate = deepcopy(counterfactual)
        candidate["agent"] = list(pos)
        if candidate["has_key"] and any(0 <= pos[0] + dr < len(candidate["grid"]) and
                                         0 <= pos[1] + dc < len(candidate["grid"][0]) and
                                         candidate["grid"][pos[0] + dr][pos[1] + dc] == "D"
                                         for dr, dc in DELTAS.values()):
            continue
        actions = optimal_actions(candidate)
        if len(actions) == 1 and set(actions).isdisjoint(target):
            return candidate
    return None


def generate_pairs(n_layouts: int, seed: int, partition: str = "discovery", *,
                   families: Iterable[str] = FAMILIES, historical_steps: str | Path | None = None,
                   excluded_layout_ids: Iterable[str] = (), max_attempts: int | None = None,
                   sizes: Iterable[int] = (7, 9), wall_density: float = 0.23,
                   donor_mode: str = "matched", require_unique_actions: bool = True) -> list[dict[str, Any]]:
    """Return one pair per new layout, cycling families; deterministic and JSON-safe.

    Historical exclusion is explicit: without ``historical_steps`` metadata says
    ``not_checked`` and callers must not describe the set as historical-data-free.
    ``mixed`` selects mismatched-context donors on alternating accepted layouts;
    their optimal action mismatch is ground truth, never an observed model result.
    """
    aliases = {"train": "discovery", "validation": "dev", "heldout": "confirmation", "holdout": "confirmation"}
    partition = aliases.get(partition, partition)
    families, sizes = tuple(families), tuple(sizes)
    if n_layouts < 0 or partition not in {"all", "discovery", "dev", "confirmation"}:
        raise ValueError("Invalid count or partition")
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError("Invalid intervention families")
    if not sizes or any(size < 5 for size in sizes) or not 0 <= wall_density < 1:
        raise ValueError("Invalid generator geometry")
    if donor_mode not in {"matched", "mismatch", "mixed"}:
        raise ValueError("Invalid donor mode")
    excluded = set(excluded_layout_ids)
    audit: dict[str, Any] = {"status": "not_checked", "source": None, "rows_checked": 0}
    if historical_steps is not None:
        historical, audit = load_historical_layout_ids(historical_steps)
        excluded.update(historical)
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    seen = set(excluded)
    for attempt in range(max_attempts if max_attempts is not None else max(1000, n_layouts * 3000)):
        if len(pairs) == n_layouts:
            break
        family = families[len(pairs) % len(families)]
        candidate = _candidate(rng, family, rng.choice(sizes), wall_density)
        if candidate is None:
            continue
        recipient, counterfactual, cell = candidate
        layout_id = canonical_layout_fingerprint(recipient, mutable_cells=[cell] if family == "wall" else [])
        layout_partition = partition_for_layout(layout_id)
        variant_ids = {canonical_layout_fingerprint(s) for s in (recipient, counterfactual)}
        if layout_id in seen or variant_ids & seen or (partition != "all" and partition != layout_partition):
            continue
        recipient_actions, target_actions = optimal_actions(recipient), optimal_actions(counterfactual)
        if not recipient_actions or not target_actions or not set(recipient_actions).isdisjoint(target_actions):
            continue
        if require_unique_actions and (len(recipient_actions) != 1 or len(target_actions) != 1):
            continue
        wants_mismatch = donor_mode == "mismatch" or (donor_mode == "mixed" and len(pairs) % 2 == 1)
        donor = _mismatch_donor(counterfactual, target_actions, rng) if wants_mismatch else deepcopy(counterfactual)
        if donor is None:
            continue
        rebuilt = apply_variable(recipient, donor, family, cell)
        if rebuilt != counterfactual:
            raise AssertionError("Interchange modified more than its named variable")
        donor_actions = optimal_actions(donor)
        pair_id = _digest({"layout_id": layout_id, "recipient": recipient, "donor": donor, "family": family})
        pairs.append({"schema_version": SCHEMA_VERSION, "pair_id": pair_id, "layout_id": layout_id,
                      "partition": layout_partition, "family": family,
                      "recipient": recipient, "donor": donor, "counterfactual": counterfactual,
                      "recipient_prompt_text": prompt_text(recipient), "donor_prompt_text": prompt_text(donor),
                      "counterfactual_prompt_text": prompt_text(counterfactual),
                      "recipient_optimal_actions": recipient_actions, "donor_optimal_actions": donor_actions,
                      "counterfactual_optimal_actions": target_actions,
                      "recipient_shortest_distance": shortest_path(recipient)[0],
                      "counterfactual_shortest_distance": shortest_path(counterfactual)[0],
                      "intervention": {"variable": family, "cell": cell if family != "key_possession" else None,
                                       "door_cell": cell if family == "key_possession" else None,
                                       "recipient_value": recipient["has_key"] if family == "key_possession" else recipient["grid"][cell[0]][cell[1]],
                                       "donor_value": donor["has_key"] if family == "key_possession" else donor["grid"][cell[0]][cell[1]]},
                      "metadata": {"seed": seed, "candidate_attempt": attempt, "generator_version": SCHEMA_VERSION,
                                   "historical_exclusion": deepcopy(audit), "explicit_exclusion_count": len(excluded),
                                   "donor_context": "different_agent_position" if wants_mismatch else "matched",
                                   "ground_truth_donor_action_mismatch": set(donor_actions).isdisjoint(target_actions),
                                   "model_donor_action_mismatch": None, "model_outputs_observed": False,
                                   "selection_basis": "simulator_only", "confirmation_status": "unqueried",
                                   "variant_layout_ids": sorted(variant_ids)}})
        seen.update(variant_ids | {layout_id})
    if len(pairs) != n_layouts:
        raise RuntimeError(f"Generated {len(pairs)}/{n_layouts} eligible layouts within the attempt budget")
    return pairs
