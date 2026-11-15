"""Fixed reversible fact encodings; staged preparation, not a query adapter.

Only canonical simulator states enter this API. There is no arbitrary prompt,
symbol-map, callback, or template override. Token eligibility is pure and does
not register experiments, load a model, or relax existing source validation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from .designs import state_control_specs
from .environments import SCHEMA_VERSION, apply_variable, optimal_actions, prompt_text, shortest_path, validate_state
from .input_sites import aligned_changed_span


CANONICAL = "canonical_v1"
ALTERNATE = "alternate_w_dot_l_yes_no_v1"
RENDERERS = (CANONICAL, ALTERNATE)
_MAPS = {
    CANONICAL: {"#": "#", "_": "_", "G": "G", "A": "A", "D": "D", "K": "K"},
    ALTERNATE: {"#": "W", "_": ".", "G": "G", "A": "A", "D": "L", "K": "K"},
}
_STATUS = {CANONICAL: {False: "False", True: "True"}, ALTERNATE: {False: "No", True: "Yes"}}
_INTRO = ("# Instructions\n\nYou are controlling an agent in a grid-based environment with full observability. "
          "Move from A to G using the fewest steps possible. You can move UP, DOWN, LEFT, or RIGHT.\n\n")
_MECHANICS = ("Key and door mechanics:\n"
              "- Moving onto K automatically picks up the key and removes K from the grid.\n"
              "- After each attempted move, carrying the key automatically opens and removes all orthogonally adjacent locked doors.\n"
              "- Walls and locked doors prevent movement. A blocked attempt still takes one step.\n"
              "- Coordinates are (row, column); UP decreases the row and RIGHT increases the column.\n\n"
              "Your final answer must be only a JSON object of the form {\"action\": \"UP\"}, "
              "with one of UP, DOWN, LEFT, RIGHT.\n\n# Inputs\n\nCurrent grid state:\n\n")
_STATUS_MARKER = "\n\nAgent status:\n- Carrying key: "


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _encoding(renderer_id: str) -> tuple[dict[str, str], dict[bool, str]]:
    if renderer_id not in RENDERERS:
        raise ValueError("unsupported_fixed_renderer")
    return _MAPS[renderer_id], _STATUS[renderer_id]


def _state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != {"schema_version", "grid", "agent", "has_key"}:
        raise ValueError("require_exact_canonical_state_fields")
    if state["schema_version"] != SCHEMA_VERSION or not isinstance(state["grid"], list) or not all(
        isinstance(row, str) for row in state["grid"]
    ) or not isinstance(state["agent"], list):
        raise ValueError("require_json_canonical_state")
    value = deepcopy(dict(state))
    validate_state(value)
    r, c = value["agent"]
    if value["grid"][r][c] != "_":
        # The existing display overlays A; a hidden goal cannot be reconstructed.
        raise ValueError("agent_overlay_would_hide_nonempty_terrain")
    return value


def _header(renderer_id: str) -> str:
    symbols, _ = _encoding(renderer_id)
    legend = (f"Legend: {symbols['#']} = wall; {symbols['_']} = open space; G = goal; A = agent; "
              f"{symbols['D']} = locked door; K = key.\n\n")
    return _INTRO + legend + _MECHANICS


def _render(state: dict[str, Any], renderer_id: str) -> str:
    symbols, values = _encoding(renderer_id)
    rows = [[symbols[cell] for cell in row] for row in state["grid"]]
    r, c = state["agent"]
    rows[r][c] = "A"
    width = max(2, len(str(max(len(rows), len(rows[0])) - 1)) + 1)
    grid = "\n".join([" " * width + "".join(f"{c:<{width}}" for c in range(len(rows[0]))).rstrip()]
                     + [f"{r:<{width}}" + "".join(f"{cell:<{width}}" for cell in row).rstrip()
                        for r, row in enumerate(rows)])
    return _header(renderer_id) + grid + _STATUS_MARKER + values[state["has_key"]]


def renderer_manifest(renderer_id: str) -> dict[str, Any]:
    symbols, values = _encoding(renderer_id)
    manifest = {"format": "fixed_grid_renderer_v1", "renderer_id": renderer_id,
                "canonical_to_display": dict(symbols), "has_key_text": {str(k): v for k, v in values.items()},
                "header_sha256": hashlib.sha256(_header(renderer_id).encode()).hexdigest(),
                "mechanics_and_action_schema_sha256": hashlib.sha256(_MECHANICS.encode()).hexdigest(),
                "grammar": "exact indexed rectangular grid; one visible A over canonical open cell; exact Boolean status",
                "custom_renderers_allowed": False}
    manifest["manifest_sha256"] = _hash(manifest)
    return manifest


def parse_prompt(text: str, renderer_id: str) -> dict[str, Any]:
    """Invert only the exact fixed prompt grammar; modified prose is rejected."""
    symbols, values = _encoding(renderer_id)
    header = _header(renderer_id)
    if not isinstance(text, str) or not text.startswith(header) or text.count(_STATUS_MARKER) != 1:
        raise ValueError("prompt_does_not_match_fixed_renderer_grammar")
    grid_text, status_text = text[len(header):].split(_STATUS_MARKER)
    status = {value: key for key, value in values.items()}
    if status_text not in status:
        raise ValueError("invalid_encoded_key_status")
    lines = grid_text.split("\n")
    columns = lines[0].split()
    if not columns or columns != [str(c) for c in range(len(columns))] or len(lines) < 2:
        raise ValueError("invalid_indexed_grid_header")
    inverse, rows, agent = {v: k for k, v in symbols.items()}, [], None
    for r, line in enumerate(lines[1:]):
        cells = line.split()
        if len(cells) != len(columns) + 1 or cells[0] != str(r):
            raise ValueError("invalid_indexed_grid_row")
        decoded = []
        for c, value in enumerate(cells[1:]):
            if value not in inverse:
                raise ValueError("unknown_display_symbol")
            cell = inverse[value]
            if cell == "A":
                if agent is not None:
                    raise ValueError("multiple_display_agents")
                agent, cell = [r, c], "_"
            decoded.append(cell)
        rows.append("".join(decoded))
    if agent is None:
        raise ValueError("missing_display_agent")
    state = _state({"schema_version": SCHEMA_VERSION, "grid": rows, "agent": agent, "has_key": status[status_text]})
    if _render(state, renderer_id) != text:
        raise ValueError("noncanonical_prompt_spacing_or_structure")
    return state


def render_with_proof(state: Mapping[str, Any], renderer_id: str = ALTERNATE) -> dict[str, Any]:
    canonical = _state(state)
    # Fail closed if the upstream canonical prompt grammar changes.
    if _render(canonical, CANONICAL) != prompt_text(canonical):
        raise ValueError("canonical_renderer_drift")
    text = _render(canonical, renderer_id)
    reconstructed = parse_prompt(text, renderer_id)
    if reconstructed != canonical:
        raise ValueError("renderer_state_round_trip_failed")
    return {"renderer_id": renderer_id, "prompt_text": text, "canonical_state": canonical,
            "proof": {"format": "exact_renderer_state_proof_v1", "state_sha256": _hash(canonical),
                      "reconstructed_state_sha256": _hash(reconstructed),
                      "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "renderer_manifest_sha256": renderer_manifest(renderer_id)["manifest_sha256"],
                      "exact_state_round_trip": True, "canonical_prompt_matches_existing_renderer": True,
                      "grid_agent_inventory_preserved": True, "mechanics_and_action_schema_preserved": True}}


def render_prompt(state: Mapping[str, Any], renderer_id: str = ALTERNATE) -> str:
    return render_with_proof(state, renderer_id)["prompt_text"]


def _toggle(state: dict[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    family, cell = pair["family"], pair["intervention"].get("cell")
    original = apply_variable(state, pair["recipient"], family, cell)
    counterfactual = apply_variable(state, pair["counterfactual"], family, cell)
    if state == original:
        return counterfactual
    if state == counterfactual:
        return original
    raise ValueError("source_fact_outside_frozen_binary_contrast")


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("invalid_tokenizer_output")
    return values


def audit_pair_renderings(pair: Mapping[str, Any], tokenizer: Any, *, profile: str) -> dict[str, Any]:
    """CPU-only joint eligibility with exact standard/alternate named-fact spans.

    Supports matched input/subspace or the complete contextual source-state
    inventory. It does not manufacture jobs or change partition/eligibility.
    """
    from .campaign import render_analysis_prompt

    if profile not in {"matched", "contextual"}:
        raise ValueError("unsupported_profile")
    pair = deepcopy(dict(pair))
    if pair["family"] not in {"wall", "key_possession", "door_status"}:
        raise ValueError("unsupported_fact_family")
    for role in ("recipient", "counterfactual"):
        _state(pair[role])
    if apply_variable(pair["recipient"], pair["counterfactual"], pair["family"],
                      pair["intervention"].get("cell")) != pair["counterfactual"]:
        raise ValueError("counterfactual_changes_more_than_named_fact")
    original, target = optimal_actions(pair["recipient"]), optimal_actions(pair["counterfactual"])
    if len(original) != len(target) or len(target) != 1 or original == target:
        raise ValueError("require_reachable_distinct_singleton_actions")
    if original != pair["recipient_optimal_actions"] or target != pair["counterfactual_optimal_actions"]:
        raise ValueError("recorded_simulator_action_mismatch")
    states = {role: pair[role] for role in ("recipient", "counterfactual")}
    edges = [("recipient", "counterfactual"), ("counterfactual", "recipient")]
    exclusions = []
    if profile == "contextual":
        states["donor"] = pair["donor"]
        for control in state_control_specs(pair):
            role = control["condition"]
            if not control["eligible"]:
                exclusions.append({"source": role, "reason": control["ineligible_reason"]})
            else:
                states[role] = control["state"]
                edges.append(("recipient", role))
    renderings = {}
    for renderer_id in RENDERERS:
        sources = {}
        for role, state in states.items():
            source = {"eligible": False, "source": role}
            try:
                rendered = render_with_proof(state, renderer_id)
                toggled = render_with_proof(_toggle(state, pair), renderer_id)
                ids = _encode(tokenizer, render_analysis_prompt(rendered["prompt_text"]))
                toggled_ids = _encode(tokenizer, render_analysis_prompt(toggled["prompt_text"]))
                span = aligned_changed_span(ids, toggled_ids)
                source.update(eligible=True, proof=rendered["proof"], toggled_proof=toggled["proof"],
                              input_token_sha256=_hash(ids), toggle_token_sha256=_hash(toggled_ids), span=span,
                              shortest_distance=shortest_path(state)[0], optimal_actions=optimal_actions(state))
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                source["reason"] = str(exc)
                exclusions.append({"renderer_id": renderer_id, "source": role, "reason": str(exc)})
            sources[role] = source
        for recipient, donor in edges:
            r, d = sources[recipient], sources[donor]
            if r["eligible"] and d["eligible"] and (
                r["span"]["n_input_tokens"] != d["span"]["n_input_tokens"] or
                r["span"]["n_changed_span_tokens"] != d["span"]["n_changed_span_tokens"]
            ):
                exclusions.append({"renderer_id": renderer_id, "recipient_source": recipient, "donor_source": donor,
                                   "reason": "cross_context_full_length_or_support_width_mismatch"})
        renderings[renderer_id] = sources
    for role in states:
        standard, alternate = (renderings[renderer][role] for renderer in RENDERERS)
        if standard["eligible"] and alternate["eligible"] and (
            standard["span"]["n_changed_span_tokens"] != alternate["span"]["n_changed_span_tokens"]
        ):
            exclusions.append({"source": role, "reason": "standard_alternate_support_width_mismatch"})
    return {"format": "joint_renderer_pair_feasibility_v1", "pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
            "family": pair["family"], "partition": pair["partition"], "profile": profile, "eligible": not exclusions,
            "eligibility_basis": "canonical_simulator_and_exact_tokenizer_only", "model_queries": 0,
            "selected_candidate": False, "registered_experiment": False, "renderings": renderings,
            "required_source_edges": [list(edge) for edge in edges], "exclusions": exclusions,
            "target_action": target[0], "original_action": original[0],
            "scope": "renderer/span feasibility on source definitions; not semantic or behavioral evidence"}
