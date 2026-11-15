"""Elicited forced factual reports on independent cached branches.

This is a new question to the model, not latent decoding, native action, or a
mediation test. Never load a model here. A caller must bind probes to the same
physical pair used by an already loaded engine's prefilled intervention state.
The native-action branch remains the original, unmodified state.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .environments import apply_variable, validate_state


CANDIDATES = {"true": " true", "false": " false"}
CLAIM_SCOPE = "elicited_forced_factual_report_diagnostic"
SUFFIX_BEFORE_QUESTION = "<|end|><|start|>user<|message|>"
SUFFIX_AFTER_QUESTION = (
    '\nReply with one JSON object containing a Boolean under the key "value".'
    '<|end|><|start|>assistant<|channel|>final<|message|>{"value":'
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_fact_probes(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Target initial-state Boolean plus a deterministic remote unchanged fact.

    The auxiliary question concerns an unchanged interior wall/floor cell at
    least two Manhattan cells from the changed coordinate (or initial agent for
    inventory changes). It is an unchanged factual control, not proven causal
    irrelevance to a route. If no such cell exists, emit an unavailable record.
    No action labels, optimal-action answers or generated reasoning are used.
    """
    pair = deepcopy(dict(pair))
    recipient, counterfactual = pair.get("recipient"), pair.get("counterfactual")
    if not isinstance(recipient, dict) or not isinstance(counterfactual, dict):
        raise ValueError("physical recipient and counterfactual states are required")
    validate_state(recipient)
    validate_state(counterfactual)
    family, intervention = pair.get("family"), pair.get("intervention", {})
    if family not in {"wall", "key_possession", "door_status"} or intervention.get("variable") != family:
        raise ValueError("unsupported or mismatched designated fact")
    cell = intervention.get("cell")
    if apply_variable(recipient, counterfactual, family, cell) != counterfactual or recipient == counterfactual:
        raise ValueError("physical counterfactual must change only the designated fact")
    if family == "key_possession":
        original, target = recipient["has_key"], counterfactual["has_key"]
        anchor = recipient["agent"]
        question = "In the initial grid state, before any move, is the agent carrying a key?"
        predicate = {"kind": "carrying_key", "cell": None}
        declared_original, declared_target = original, target
    else:
        r, c = cell
        symbol = "#" if family == "wall" else "D"
        original, target = recipient["grid"][r][c] == symbol, counterfactual["grid"][r][c] == symbol
        anchor = [r, c]
        noun = "a wall" if family == "wall" else "a locked door"
        question = f"In the initial grid state, before any move, does the cell at row {r}, column {c} contain {noun}?"
        predicate = {"kind": "wall" if family == "wall" else "locked_door", "cell": [r, c]}
        declared_original, declared_target = recipient["grid"][r][c], counterfactual["grid"][r][c]
    if original == target:
        raise ValueError("the designated Boolean fact does not change")
    if intervention.get("recipient_value", declared_original) != declared_original or intervention.get("donor_value", declared_target) != declared_target:
        raise ValueError("declared fact values disagree with physical states")
    common = {"pair_id": pair.get("pair_id"), "layout_id": pair.get("layout_id"), "family": family,
              "source_pair_sha256": _digest(pair), "truth_basis": "validated_initial_physical_states",
              "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False,
              "latent_decoding": False, "mediation_evidence": False}
    probes = [{**common, "id": "target_initial_fact", "role": "target_fact", "available": True,
               "question": question, "recipient_value": bool(original), "counterfactual_value": bool(target),
               "predicate": predicate}]
    rows = recipient["grid"]
    candidates = []
    for r in range(1, len(rows) - 1):
        for c in range(1, len(rows[0]) - 1):
            distance = abs(r - anchor[0]) + abs(c - anchor[1])
            if distance < 2 or [r, c] == recipient["agent"] or rows[r][c] not in "#_" or rows[r][c] != counterfactual["grid"][r][c]:
                continue
            candidates.append((-distance, r, c))
    if not candidates:
        probes.append({**common, "id": "unchanged_remote_grid_fact", "role": "unchanged_fact_control",
                       "available": False, "question": None, "recipient_value": None, "counterfactual_value": None,
                       "unavailable_reason": "no_unchanged_remote_interior_wall_or_floor_cell"})
    else:
        _, r, c = min(candidates)
        value = rows[r][c] == "#"
        probes.append({**common, "id": "unchanged_remote_grid_fact", "role": "unchanged_fact_control",
                       "available": True, "question": f"In the initial grid state, before any move, does the cell at row {r}, column {c} contain a wall?",
                       "recipient_value": value, "counterfactual_value": value,
                       "predicate": {"kind": "wall", "cell": [r, c]},
                       "control_note": "Remote unchanged observation; causal irrelevance to route choice is not assumed."})
    return probes


def _numpy(engine: Any, value: Any) -> np.ndarray:
    # MLX BF16 cannot expose a compatible NumPy buffer directly. Float32
    # conversion exactly represents BF16 values; preserve other dtypes.
    if "bfloat16" in str(getattr(value, "dtype", "")):
        value = value.astype(engine._mx.float32)
    if getattr(engine, "_mx", None) is not None:
        engine._mx.eval(value)
    return np.asarray(value)


def _tree_fingerprint(engine: Any, value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.ascontiguousarray(_numpy(engine, value))
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _tree_fingerprint(engine, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tree_fingerprint(engine, v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported cache metadata type: {type(value).__name__}")


def _state_proof(engine: Any, state: Any) -> dict[str, Any]:
    payload = {"token_ids": [int(t) for t in state.token_ids],
               "cache": [{"state": _tree_fingerprint(engine, c.state),
                          "meta_state": _tree_fingerprint(engine, c.meta_state)} for c in state.cache],
               "next_logits": _tree_fingerprint(engine, state.next_logits),
               "patch_log": _tree_fingerprint(engine, state.patch_log)}
    return {"state_sha256": _digest(payload), "token_ids_sha256": _digest(payload["token_ids"]),
            "cache_sha256": _digest(payload["cache"]), "next_logits_sha256": _digest(payload["next_logits"]),
            "patch_log_sha256": _digest(payload["patch_log"]), "n_tokens": len(state.token_ids),
            "n_cache_layers": len(state.cache)}


def _token_ids(values: Sequence[int], *, strictly_positive: bool = False) -> list[int]:
    ids = list(values)
    minimum = 1 if strictly_positive else 0
    if not ids or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < minimum for i in ids):
        raise ValueError("token IDs must be nonempty valid integers")
    return [int(i) for i in ids]


def score_fact_reports(engine: Any, state: Any, probes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score true/false at a verified full-context boundary on independent clones.

    Each available probe performs one suffix advance from the original state,
    not from another probe. Positive single-token IDs for ``' true'``/``' false'``
    and exact suffix/candidate token concatenation after a single-token Harmony
    separator are mandatory. The original prefix IDs remain authoritative even
    when generated tokenization is not canonical. Unsupported suffix BPE
    boundaries are explicit unavailable results, not approximations. Fingerprints
    materialize each source-cache array before and after scoring, trading some
    CPU transfer for a recorded preservation check. No RNG is touched.
    """
    if (getattr(engine, "metadata", {}).get("loaded") is not True
            or getattr(engine, "_model", None) is None or getattr(engine, "_tokenizer", None) is None):
        raise ValueError("fact scoring requires an already loaded engine; it never loads one")
    if getattr(state, "engine_id", None) != id(engine):
        raise ValueError("state belongs to a different engine")
    probes = deepcopy([dict(p) for p in probes])
    if len({p.get("id") for p in probes}) != len(probes):
        raise ValueError("probe IDs must be unique")
    for probe in probes:
        if not isinstance(probe.get("id"), str) or not probe["id"]:
            raise ValueError("each probe requires a nonempty id")
        if probe.get("available", True) is False:
            if not probe.get("unavailable_reason"):
                raise ValueError("unavailable probes require an explicit reason")
            continue
        if not isinstance(probe.get("question"), str) or not probe["question"].strip() or "<|" in probe["question"]:
            raise ValueError("probe question must be nonempty and contain no Harmony control tokens")
        if type(probe.get("recipient_value")) is not bool or type(probe.get("counterfactual_value")) is not bool:
            raise ValueError("probe truths must be Boolean")
    base_ids = _token_ids(state.token_ids)
    before = _state_proof(engine, state)
    results = []
    for probe in probes:
        report = {**probe, "probe_sha256": _digest(probe), "endpoint": "elicited_forced_fact_report",
                  "claim_scope": CLAIM_SCOPE, "semantic_headline_eligible": False,
                  "latent_decoding": False, "mediation_evidence": False,
                  "base_state": before, "model_path": str(getattr(engine, "model_path", "unspecified")),
                  "inference_configuration": {"head_mode": getattr(engine, "head_mode", "unspecified")}}
        if probe.get("available", True) is False:
            report.update(status="unavailable", query_performed=False)
            report["source_state_unchanged"] = _state_proof(engine, state) == before
            results.append(report)
            continue
        suffix = SUFFIX_BEFORE_QUESTION + probe["question"] + SUFFIX_AFTER_QUESTION
        report.update(suffix=suffix, suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest())
        try:
            separator_ids = _token_ids(engine.encode("<|end|>"), strictly_positive=True)
            if len(separator_ids) != 1 or engine.decode(separator_ids) != "<|end|>":
                raise ValueError("Harmony_separator_not_one_exact_token")
            suffix_ids = _token_ids(engine.encode(suffix), strictly_positive=True)
            if suffix_ids[0] != separator_ids[0] or engine.decode(suffix_ids) != suffix:
                raise ValueError("Harmony_separator_suffix_boundary_not_exact")
            candidate_ids = {}
            for value, text in CANDIDATES.items():
                encoded = _token_ids(engine.encode(text), strictly_positive=True)
                if len(encoded) != 1 or engine.decode(encoded) != text:
                    raise ValueError("Boolean_candidate_not_one_exact_token")
                if engine.encode(suffix + text) != suffix_ids + encoded:
                    raise ValueError("suffix_Boolean_boundary_not_exact")
                candidate_ids[value] = encoded[0]
            if len(set(candidate_ids.values())) != 2:
                raise ValueError("Boolean_candidate_token_alias")
        except ValueError as exc:
            report.update(status="unavailable", unavailable_reason=str(exc), query_performed=False,
                          source_state_unchanged=_state_proof(engine, state) == before)
            results.append(report)
            continue
        branch = engine.clone_state(state)
        if branch is state or branch.cache is state.cache or any(a is b for a, b in zip(branch.cache, state.cache)):
            raise RuntimeError("clone_state returned shared mutable cache containers")
        if _state_proof(engine, branch) != before:
            raise RuntimeError("clone_state does not preserve the original cached state")
        try:
            engine._advance(branch, suffix_ids)
            logits = _numpy(engine, branch.next_logits).astype(np.float64).reshape(-1)
            if not np.isfinite(logits).all() or max(candidate_ids.values()) >= len(logits):
                raise ValueError("invalid global logits or Boolean token outside vocabulary")
            selected = np.array([logits[candidate_ids[k]] for k in CANDIDATES])
            candidate_logsumexp = float(selected.max() + np.log(np.exp(selected - selected.max()).sum()))
            global_logsumexp = float(logits.max() + np.log(np.exp(logits - logits.max()).sum()))
            q = {k: float(np.exp(logits[token] - candidate_logsumexp)) for k, token in candidate_ids.items()}
            full_probabilities = {k: float(np.exp(logits[token] - global_logsumexp)) for k, token in candidate_ids.items()}
            report.update(status="scored", query_performed=True, boundary_verified=True,
                          boundary_verification="single_token_Harmony_separator_and_exact_suffix_candidate_BPE",
                          authoritative_prefix_token_ids_sha256=_digest(base_ids), prefix_reencoded=False,
                          suffix_token_ids=suffix_ids, suffix_token_ids_sha256=_digest(suffix_ids), candidate_token_ids=candidate_ids,
                          candidate_texts=dict(CANDIDATES), logits={k: float(logits[t]) for k, t in candidate_ids.items()},
                          q=q, candidate_global_probabilities=full_probabilities,
                          candidate_mass=float(np.exp(candidate_logsumexp - global_logsumexp)),
                          global_logsumexp=global_logsumexp, global_top_logit=float(logits.max()),
                          global_top_token_id=int(logits.argmax()), vocabulary_size=len(logits),
                          recipient_truth_q=q[str(probe["recipient_value"]).lower()],
                          counterfactual_truth_q=q[str(probe["counterfactual_value"]).lower()],
                          branch_n_tokens_after=len(branch.token_ids), independent_clone=True)
        finally:
            after = _state_proof(engine, state)
            if after != before:
                raise RuntimeError("fact scoring mutated the original native-action state")
        report.update(source_state_after=after, source_state_unchanged=True)
        results.append(report)
    return results
