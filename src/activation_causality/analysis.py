"""Independent, layout-paired analysis of activation intervention results.

The estimand is the mean over layouts of the within-layout treatment-minus-
control change in natural target-action success. Repeated seeds and related
pairs are averaged *before* layouts are resampled. Failed generations remain
in this intention-to-treat endpoint. Forced probabilities are secondary.

This module never loads a model and never upgrades exploratory action flips to
evidence of a semantic mechanism. Evidence flags must be supplied by a reviewer
after inspecting the corresponding frozen specifications and artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
NATIVE_STATUSES = {"complete", "truncated", "malformed"}
STATUSES = NATIVE_STATUSES | {"not_requested", "technical_error"}
SEMANTIC_EVIDENCE = (
    "technical_valid",
    "fresh_confirmation",
    "frozen_specification",
    "semantic_specificity",
    "donor_action_mismatch",
    "independent_replication",
)
SEMANTIC_CONTROL_ROLES = (
    "counterfactual_baseline", "donor_baseline", "same_fact",
    "irrelevant_fact", "donor_mismatch_treatment",
)
CAUTIONS = [
    "Natural target-action success counts truncated and malformed generations as failures; "
    "none are replaced by a forced answer or discarded.",
    "Completion-sensitive bounds allow either unresolved status to resolve to any action. "
    "They are sensitivity analyses, not observed natural outcomes.",
    "Layouts receive equal weight; variants, pairs, and seeds within a layout are not "
    "independent observations.",
    "Centered layout bootstrap p-values are approximate under independent, identically "
    "distributed layouts. Residual layout effects are centered at zero before resampling; "
    "this does not assume treatment/control label exchangeability.",
    "Confidence intervals are paired layout bootstrap intervals. Confirmation uses "
    "Bonferroni simultaneous intervals plus Holm-adjusted p-values; these are distinct "
    "multiplicity procedures. Bootstrap intervals are approximate finite-sample intervals.",
    "Sufficiency, forced-action redirection, and state decodability do not establish "
    "necessity or mediation. Removal and restoration need their own audited contrasts.",
    "No outcome-dependent exclusion or selection is justified by this analysis. "
    "Missing execution rows and layout leakage block confirmatory claims.",
    "Forced-only jobs are descriptive and never enter the natural-action denominator. "
    "Technical errors remain visible and block confirmation rather than count as behavioral failures.",
]


def load_results(rows: Iterable[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    """Load row mappings or a JSONL path without silently dropping bad records."""
    if isinstance(rows, (str, Path)):
        source = Path(rows)
        result = []
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            result.append(item)
        return result
    return [dict(row) for row in rows]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    if any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite probabilities")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(str(row[key]) for key in ("experiment_id", "partition", "layout_id", "pair_id", "seed"))


def _layout_aliases(row: Mapping[str, Any]) -> set[str]:
    """Include recorded wall-pair variants in the partition/identity audit."""
    aliases = {str(row["layout_id"])}
    aliases.update(str(value) for value in row.get("layout_aliases", []))
    pair = row.get("metadata", {}).get("pair", {})
    aliases.update(str(value) for value in pair.get("metadata", {}).get("variant_layout_ids", []))
    return aliases


def _validate(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[Any, ...]] = set()
    jobs: set[str] = set()
    required = (
        "experiment_id", "layout_id", "pair_id", "family", "partition",
        "condition", "target_action", "generation_status", "seed",
    )
    for index, row in enumerate(rows):
        missing = [key for key in required if key not in row or row[key] is None]
        if missing:
            raise ValueError(f"row {index}: missing {', '.join(missing)}")
        if row["target_action"] not in ACTIONS:
            raise ValueError(f"row {index}: invalid target_action {row['target_action']!r}")
        if row["generation_status"] not in STATUSES:
            raise ValueError(f"row {index}: unsupported generation_status {row['generation_status']!r}")
        action = row.get("native_action")
        if row["generation_status"] == "complete" and action not in ACTIONS:
            raise ValueError(f"row {index}: complete generation needs a valid native_action")
        if row["generation_status"] != "complete" and action is not None:
            raise ValueError(f"row {index}: unresolved generation must have native_action=null")
        if row.get("status") == "error" and row["generation_status"] != "technical_error":
            raise ValueError(f"row {index}: execution error needs generation_status=technical_error")
        key = (*_row_key(row), str(row["condition"]))
        if key in seen:
            raise ValueError(f"duplicate paired observation: {key}")
        seen.add(key)
        if row.get("job_id") is not None:
            job_id = str(row["job_id"])
            if job_id in jobs:
                raise ValueError(f"duplicate job_id: {job_id}")
            jobs.add(job_id)
        _target_probability(row)


def _target_probability(row: Mapping[str, Any]) -> float | None:
    q = row.get("forced_q")
    if q is None:
        return None
    if isinstance(q, Mapping):
        if set(q) != set(ACTIONS):
            raise ValueError("forced_q mapping must contain exactly UP/DOWN/LEFT/RIGHT")
        probabilities = [float(q[action]) for action in ACTIONS]
    elif isinstance(q, (list, tuple)):
        if len(q) != len(ACTIONS):
            raise ValueError("forced_q vector must have four probabilities in UP/DOWN/LEFT/RIGHT order")
        probabilities = [float(value) for value in q]
    else:
        raise ValueError("forced_q must be an action-probability mapping, vector, or null")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("forced_q must contain finite probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError("forced_q must sum to one")
    return probabilities[ACTIONS.index(str(row["target_action"]))]


def _target_original_log_odds(row: Mapping[str, Any]) -> float | None:
    if row.get("original_action") not in ACTIONS or row.get("forced_q") is None:
        return None
    q = row["forced_q"]
    target = _target_probability(row)
    original = float(q[row["original_action"]]) if isinstance(q, Mapping) else float(q[ACTIONS.index(row["original_action"])])
    return math.log(max(1e-7, target) / max(1e-7, original))


def _success(row: Mapping[str, Any]) -> float:
    return float(row["generation_status"] == "complete" and row.get("native_action") == row["target_action"])


def _seed(seed: int, name: str) -> int:
    return (seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)) % (2**32)


def _bootstrap_interval(
    values: np.ndarray, *, alpha: float, n_bootstrap: int, seed: int,
    support: tuple[float, float] = (-1.0, 1.0),
) -> tuple[list[float] | None, str | None]:
    """Resample already-aggregated independent layouts, retaining paired effects."""
    count = len(values)
    if count < 2:
        return None, "fewer_than_two_layouts"
    if np.all(values == values[0]):
        # A percentile bootstrap otherwise misleadingly returns [0, 0] after no
        # observed changes. Widen it by an unobserved-tail sensitivity guard.
        tail_mass = 1.0 - (alpha / 2.0) ** (1.0 / count)
        value = float(values[0])
        return [value - (value - support[0]) * tail_mass,
                value + (support[1] - value) * tail_mass], "constant_effect_unobserved_tail_guard"
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 256):
        stop = min(start + 256, n_bootstrap)
        indices = rng.integers(0, count, size=(stop - start, count))
        draws[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [alpha / 2, 1 - alpha / 2])], None


def _centered_bootstrap_p(values: np.ndarray, *, n_resamples: int, seed: int) -> tuple[float | None, str, float | None]:
    """Approximate two-sided mean-zero test, resampling centered layout effects.

    A sharp randomization null is unavailable for deterministic interventions.
    Use the iid-layout bootstrap null rather than assuming sign exchangeability.
    The plus-one estimate avoids a zero Monte Carlo p-value.
    """
    count = len(values)
    if count < 2:
        return None, "not_run_insufficient_layouts", None
    observed = abs(float(values.mean()))
    centered = values - values.mean()
    residual_mean = float(centered.mean())
    if observed < 1e-15:
        return 1.0, "centered_layout_bootstrap_zero_effect", residual_mean
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_resamples, 256):
        size = min(256, n_resamples - start)
        indices = rng.integers(0, count, size=(size, count))
        samples = centered[indices].mean(axis=1)
        extreme += int(np.sum(np.abs(samples) >= observed - 1e-12))
    return (extreme + 1) / (n_resamples + 1), "centered_layout_bootstrap_plus_one", residual_mean


def _stratified_draws(values: np.ndarray, groups: list[np.ndarray], weights: list[float],
                      *, n_resamples: int, seed: int) -> np.ndarray:
    """Keep the registered family mixture fixed while resampling paired layouts."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_resamples, dtype=float)
    for indices, weight in zip(groups, weights):
        for start in range(0, n_resamples, 256):
            stop = min(start + 256, n_resamples)
            picked = rng.integers(0, len(indices), size=(stop - start, len(indices)))
            draws[start:stop] += weight * values[indices[picked]].mean(axis=1)
    return draws


def _stratified_interval(values: np.ndarray, groups: list[np.ndarray], weights: list[float],
                         *, alpha: float, n_bootstrap: int, seed: int) -> tuple[list[float] | None, str | None]:
    if any(len(group) < 2 for group in groups):
        return None, "fewer_than_two_layouts_in_one_or_more_families"
    if any(np.all(values[group] == values[group[0]]) for group in groups):
        # A fixed-composition bootstrap cannot discover variation in a family
        # whose observed effects are constant. Combine simultaneous per-family
        # intervals, including the existing unobserved-tail guard, conservatively.
        intervals = [_bootstrap_interval(values[group], alpha=alpha / len(groups),
                                          n_bootstrap=n_bootstrap, seed=seed + i)[0]
                     for i, group in enumerate(groups)]
        return [float(sum(weight * interval[side] for weight, interval in zip(weights, intervals)))
                for side in (0, 1)], "within_family_constant_effect_simultaneous_tail_guard"
    draws = _stratified_draws(values, groups, weights, n_resamples=n_bootstrap, seed=seed)
    return [float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2])], None


def _stratified_p(values: np.ndarray, groups: list[np.ndarray], weights: list[float],
                   *, n_resamples: int, seed: int) -> tuple[float | None, str, float | None]:
    if any(len(group) < 2 for group in groups):
        return None, "not_run_insufficient_family_layouts", None
    mean = float(sum(weight * values[group].mean() for group, weight in zip(groups, weights)))
    observed = abs(mean)
    # Center the one registered mixture endpoint, preserving family differences.
    # Fixed-quota draws cancel their weighted offsets; this is not three tests.
    residuals = values - mean
    residual_mean = float(sum(weight * residuals[group].mean() for group, weight in zip(groups, weights)))
    if observed < 1e-15:
        return 1.0, "within_family_centered_layout_bootstrap_zero_effect", residual_mean
    if any(np.all(values[group] == values[group[0]]) for group in groups):
        # A residual bootstrap of constant strata is spuriously certain. For
        # independent layout scores in [-1, 1], Hoeffding bounds the two-sided
        # tail under the global weighted-mean-zero null, even with unequal
        # family means. With registered weights n_f/N this is 2 exp(-N d²/2).
        squared_weights = sum(weight * weight / len(group) for group, weight in zip(groups, weights))
        bound = min(1.0, 2.0 * math.exp(-observed * observed / (2.0 * squared_weights)))
        return bound, "within_family_constant_guard_two_sided_hoeffding", residual_mean
    draws = _stratified_draws(residuals, groups, weights, n_resamples=n_resamples, seed=seed)
    extreme = int(np.sum(np.abs(draws) >= observed - 1e-12))
    return (extreme + 1) / (n_resamples + 1), "within_family_centered_layout_bootstrap_plus_one", residual_mean


def _family_contract(spec: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    fields = {"stratification_key", "family_layout_counts", "family_weights",
              "expected_n_layouts_by_family", "expected_n_pairs_by_family"}
    scheme = spec.get("resampling_scheme")
    if scheme is None and not (fields & set(spec)):
        return None
    if scheme != "within_family_layout_bootstrap" or spec.get("stratification_key") != "family":
        raise ValueError("mixed family quotas require explicit within-family layout resampling")
    families = ("wall", "key_possession", "door_status")
    counts = spec.get("family_layout_counts")
    if (not isinstance(counts, dict) or set(counts) != set(families)
            or any(type(n) is not int or n < 1 for n in counts.values())
            or max(counts.values()) - min(counts.values()) > 1):
        raise ValueError("family_layout_counts must freeze all three positive balanced quotas")
    total = sum(counts.values())
    weights = spec.get("family_weights")
    if (not isinstance(weights, dict) or set(weights) != set(families)
            or any(type(weights[f]) not in {int, float} or not math.isfinite(weights[f])
                   or abs(weights[f] - counts[f] / total) > 1e-12 for f in families)):
        raise ValueError("family_weights must equal the frozen layout fractions")
    for field in ("expected_n_layouts_by_family", "expected_n_pairs_by_family"):
        values = spec.get(field)
        if (not isinstance(values, dict) or set(values) != set(families)
                or any(type(n) is not int or n < 0 for n in values.values())):
            raise ValueError(f"{field} must explicitly cover the three families")
    if (sum(spec["expected_n_layouts_by_family"].values()) != spec.get("expected_n_layouts")
            or sum(spec["expected_n_pairs_by_family"].values()) != spec.get("expected_n_pairs")):
        raise ValueError("expected per-family counts must sum to the registered total counts")
    layout_families: dict[str, str] = {}
    for row in rows:
        layout, family = str(row["layout_id"]), str(row["family"])
        if family not in families:
            raise ValueError("stratified comparison contains an unregistered family")
        if layout in layout_families and layout_families[layout] != family:
            raise ValueError("one layout cannot belong to multiple frozen family strata")
        layout_families[layout] = family
    return {"scheme": scheme, "stratification_key": "family", "families": list(families),
            "family_layout_counts": counts, "family_weights": weights,
            "expected_n_layouts_by_family": spec["expected_n_layouts_by_family"],
            "expected_n_pairs_by_family": spec["expected_n_pairs_by_family"],
            "layout_families": layout_families}


def _descriptive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment_id"]), str(row["partition"]), str(row["condition"]))].append(row)
    descriptions = []
    for (experiment, partition, condition), group in sorted(groups.items()):
        layout_values: dict[str, list[float]] = defaultdict(list)
        for row in group:
            if row["generation_status"] in NATIVE_STATUSES:
                layout_values[str(row["layout_id"])].append(_success(row))
        forced = [_target_probability(row) for row in group]
        forced = [value for value in forced if value is not None]
        forced_top_target = 0
        for row in group:
            q = row.get("forced_q")
            if q is not None:
                values = [q[action] for action in ACTIONS] if isinstance(q, Mapping) else q
                forced_top_target += ACTIONS[int(np.argmax(values))] == row["target_action"]
        descriptions.append({
            "experiment_id": experiment, "partition": partition, "condition": condition,
            "n_rows": len(group), "n_layouts": len({str(row["layout_id"]) for row in group}),
            "n_natural_rows": sum(len(values) for values in layout_values.values()),
            "generation_status_counts": dict(Counter(str(row["generation_status"]) for row in group)),
            "layout_weighted_native_target_rate": float(np.mean([np.mean(v) for v in layout_values.values()])) if layout_values else None,
            "descriptive_row_weighted_forced_target_probability": float(np.mean(forced)) if forced else None,
            "n_forced_rows": len(forced), "forced_target_top_count": int(forced_top_target),
        })
    return descriptions


def _infer_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["experiment_id"]), str(row["partition"]))].add(str(row["condition"]))
    comparisons = []
    for (experiment, partition), conditions in sorted(groups.items()):
        control = next((name for name in ("baseline", "identity", "recipient_text", "neutral", "control")
                        if name in conditions), None)
        if control is None:
            continue
        for treatment in sorted(conditions - {control}):
            comparisons.append({"experiment_id": experiment, "partition": partition,
                                "treatment": treatment, "control": control, "inferred": True})
    return comparisons


def _semantic_control_audit(
    rows: list[dict[str, Any]], spec: Mapping[str, Any], native_keys: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    """Require actual native control data in addition to reviewer assertions.

    ``donor_baseline`` must name the donor used by ``donor_mismatch_treatment``;
    the model's observed final action, not its simulator optimum, defines the
    mismatch subset. Statistical specificity still requires independent review.
    """
    roles = dict(spec.get("semantic_controls", {}))
    if any(not isinstance(condition, str) or not condition for condition in roles.values()):
        raise ValueError("semantic control conditions must be nonempty strings")
    if len(set(roles.values())) != len(roles):
        raise ValueError("distinct semantic control roles need distinct execution conditions")
    records = {(*_row_key(row), str(row["condition"])): row for row in rows}
    coverage = {}
    valid_by_role: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for role in SEMANTIC_CONTROL_ROLES:
        condition = roles.get(role)
        matched = {}
        for key in native_keys:
            value = records.get((*key, str(condition))) if condition else None
            if value is not None and value["generation_status"] in NATIVE_STATUSES:
                matched[key] = value
        valid_by_role[role] = matched
        coverage[role] = {"condition": condition, "n_native_pairs": len(matched),
                          "n_native_layouts": len({key[2] for key in matched})}
    missing = [role for role in SEMANTIC_CONTROL_ROLES
               if not native_keys or len(valid_by_role[role]) != len(native_keys)]
    mismatches = []
    for key, donor in valid_by_role["donor_baseline"].items():
        if donor["generation_status"] == "complete" and donor["native_action"] != donor["target_action"]:
            if key in valid_by_role["donor_mismatch_treatment"] and key in valid_by_role["same_fact"]:
                mismatches.append(key)
    mismatch_layouts = {key[2] for key in mismatches}
    per_layout: dict[str, list[float]] = defaultdict(list)
    for key in mismatches:
        per_layout[key[2]].append(_success(valid_by_role["donor_mismatch_treatment"][key])
                                  - _success(valid_by_role["same_fact"][key]))
    mismatch_effect = float(np.mean([np.mean(values) for values in per_layout.values()])) if per_layout else None
    return {"coverage": coverage, "missing_or_incomplete_roles": missing,
            "observed_donor_mismatch_pairs": len(mismatches),
            "observed_donor_mismatch_layouts": len(mismatch_layouts),
            "observed_mismatch_target_effect_vs_same_fact": mismatch_effect,
            "native_controls_ready": bool(not missing and len(mismatch_layouts) >= 2
                                           and mismatch_effect is not None and mismatch_effect > 0),
            "note": "Control presence is necessary, not sufficient: reviewer flags must audit semantic selectivity, matching, and mismatch-subset uncertainty."}


def _comparison(
    rows: list[dict[str, Any]], spec: dict[str, Any], *, n_bootstrap: int,
    n_permutations: int, alpha: float, family_size: int, seed: int,
    run_inference: bool = True,
) -> dict[str, Any]:
    experiment = str(spec["experiment_id"])
    partition = str(spec.get("partition", "confirmation"))
    treatment, control = str(spec["treatment"]), str(spec["control"])
    if treatment == control:
        raise ValueError("treatment and control must differ")
    identifier = str(spec.get("id", f"{experiment}:{partition}:{treatment}-vs-{control}"))
    direction = spec.get("predicted_direction", "increase")
    if direction not in {"increase", "decrease", "zero"}:
        raise ValueError("predicted_direction must be increase, decrease, or zero")
    frozen_pair_ids = spec.get("eligible_pair_ids")
    if frozen_pair_ids is not None:
        if not isinstance(frozen_pair_ids, list) or any(not isinstance(pair_id, str) or not pair_id for pair_id in frozen_pair_ids):
            raise ValueError("eligible_pair_ids must be a list of nonempty pair ID strings")
        if len(set(frozen_pair_ids)) != len(frozen_pair_ids):
            raise ValueError("eligible_pair_ids must be unique")
        frozen_pair_ids = set(frozen_pair_ids)
    eligible = [row for row in rows if str(row["experiment_id"]) == experiment
                and str(row["partition"]) == partition and str(row["condition"]) in {treatment, control}
                and (frozen_pair_ids is None or str(row["pair_id"]) in frozen_pair_ids)]
    stratification = _family_contract(spec, eligible)
    treated = {_row_key(row): row for row in eligible if str(row["condition"]) == treatment}
    controls = {_row_key(row): row for row in eligible if str(row["condition"]) == control}
    shared_keys = sorted(set(treated) & set(controls))
    incompatible = [key for key in shared_keys
                    if treated[key]["target_action"] != controls[key]["target_action"]
                    or treated[key]["family"] != controls[key]["family"]]
    if incompatible:
        raise ValueError(f"{identifier}: inconsistent paired target or family at {incompatible[0]}")
    native_keys = [key for key in shared_keys if treated[key]["generation_status"] in NATIVE_STATUSES
                   and controls[key]["generation_status"] in NATIVE_STATUSES]
    native_key_set = set(native_keys)
    layouts: dict[str, list[list[float]]] = defaultdict(list)
    forced_by_layout: dict[str, list[list[float]]] = defaultdict(list)
    forced_original_by_layout: dict[str, list[float]] = defaultdict(list)
    for key in shared_keys:
        t_row, c_row = treated[key], controls[key]
        if key in native_key_set:
            t, c = _success(t_row), _success(c_row)
            t_missing, c_missing = (float(row["generation_status"] != "complete") for row in (t_row, c_row))
            layouts[key[2]].append([t, c, t - c, t - c - c_missing, t - c + t_missing,
                                    1 - t_missing, 1 - c_missing])
        tq, cq = _target_probability(t_row), _target_probability(c_row)
        if tq is not None and cq is not None:
            eps = 1e-7
            logit_t = math.log(max(eps, tq) / max(eps, 1 - tq))
            logit_c = math.log(max(eps, cq) / max(eps, 1 - cq))
            forced_by_layout[key[2]].append([tq - cq, logit_t - logit_c])
            to, co = _target_original_log_odds(t_row), _target_original_log_odds(c_row)
            if to is not None and co is not None:
                if t_row["original_action"] != c_row["original_action"]:
                    raise ValueError(f"{identifier}: inconsistent paired original_action")
                forced_original_by_layout[key[2]].append(to - co)
    layout_ids = sorted(layouts)
    matrix = np.asarray([np.mean(layouts[layout], axis=0) for layout in layout_ids], dtype=float)
    groups, weights = [], []
    family_counts, family_pair_counts = {}, {}
    stratified_complete = True
    if stratification:
        groups = [np.asarray([i for i, layout in enumerate(layout_ids)
                              if stratification["layout_families"][layout] == family], dtype=int)
                  for family in stratification["families"]]
        weights = [stratification["family_weights"][family] for family in stratification["families"]]
        family_counts = {f: len(group) for f, group in zip(stratification["families"], groups)}
        family_pair_counts = {f: sum(treated[key]["family"] == f for key in native_keys)
                              for f in stratification["families"]}
        stratified_complete = (family_counts == stratification["family_layout_counts"]
                               and family_counts == stratification["expected_n_layouts_by_family"]
                               and family_pair_counts == stratification["expected_n_pairs_by_family"])

    def average(column: int) -> float | None:
        if not len(layouts) or (stratification and any(not len(group) for group in groups)):
            return None
        if stratification:
            return float(sum(weight * matrix[group, column].mean() for group, weight in zip(groups, weights)))
        return float(matrix[:, column].mean())

    def interval(column: int, level: float, suffix: str = "") -> tuple[list[float] | None, str | None]:
        if stratification:
            return _stratified_interval(matrix[:, column], groups, weights, alpha=level,
                                        n_bootstrap=n_bootstrap, seed=_seed(seed, identifier + suffix))
        return _bootstrap_interval(matrix[:, column], alpha=level, n_bootstrap=n_bootstrap,
                                    seed=_seed(seed, identifier + suffix))
    explicit_confirmation = partition == "confirmation" and not spec.get("inferred", False) and spec.get("confirmatory", True)
    adjusted_alpha = alpha / family_size if explicit_confirmation else alpha
    effect = average(2)
    estimates: dict[str, Any] = {
        "treatment_rate": average(0),
        "control_rate": average(1),
        "effect": effect, "ci": None, "adjusted_ci": None,
        "p_value": None, "holm_p_value": None,
    }
    warnings = []
    if stratification:
        estimates["interval_method"] = "within_family_layout_bootstrap_with_simultaneous_constant_family_guard"
        estimates["mixture_weights"] = dict(stratification["family_weights"])
    if len(layouts) and stratified_complete and run_inference:
        estimates["ci"], warning = interval(2, alpha)
        estimates["adjusted_ci"], _ = interval(2, adjusted_alpha)
        if warning:
            warnings.append(warning)
        if stratification:
            estimates["p_value"], estimates["p_value_method"], estimates["null_residual_mean"] = _stratified_p(
                matrix[:, 2], groups, weights, n_resamples=n_permutations, seed=_seed(seed, identifier + ":test"))
        else:
            estimates["p_value"], estimates["p_value_method"], estimates["null_residual_mean"] = _centered_bootstrap_p(
                matrix[:, 2], n_resamples=n_permutations, seed=_seed(seed, identifier + ":test"))
    elif stratification and not stratified_complete:
        warnings.append("incomplete_frozen_family_cohort_no_inference")
    bounds = {"lower": None, "upper": None, "adjusted_uncertainty_envelope": None}
    if len(layouts):
        lower_ci, lower_warning = interval(3, adjusted_alpha, ":lower") if stratified_complete and run_inference else (None, None)
        upper_ci, upper_warning = interval(4, adjusted_alpha, ":upper") if stratified_complete and run_inference else (None, None)
        bounds = {"lower": average(3), "upper": average(4),
                  "adjusted_uncertainty_envelope": [lower_ci[0], upper_ci[1]] if lower_ci and upper_ci else None}
        if stratification:
            bounds["interval_method"] = estimates["interval_method"]
            warnings.extend(warning for warning in (lower_warning, upper_warning)
                            if warning and warning not in warnings)
    all_partitions: dict[str, set[str]] = defaultdict(set)
    alias_owners: dict[str, set[str]] = defaultdict(set)
    aliases_by_layout: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        aliases = _layout_aliases(row)
        aliases_by_layout[str(row["layout_id"])].update(aliases)
        for alias in aliases:
            all_partitions[alias].add(str(row["partition"]))
            alias_owners[alias].add(str(row["layout_id"]))
    leaked = sorted(layout for layout in layout_ids
                    if any(len(all_partitions[alias]) > 1 for alias in aliases_by_layout[layout]))
    identity_conflicts = sorted(layout for layout in layout_ids
                                if any(len(alias_owners[alias]) > 1 for alias in aliases_by_layout[layout]))
    exclusions = {
        "treatment_without_control": len(set(treated) - set(controls)),
        "control_without_treatment": len(set(controls) - set(treated)),
        "unpaired_keys": [list(key) for key in sorted(set(treated) ^ set(controls))],
        "cross_partition_layout_ids": leaked,
        "layout_alias_identity_conflicts": identity_conflicts,
        "noncompletions_excluded": 0,
        "eligibility_basis": "explicit_specification_pair_ids" if frozen_pair_ids is not None else "all_pairs_in_comparison",
        "eligible_pair_ids": sorted(frozen_pair_ids) if frozen_pair_ids is not None else None,
        "technical_error_rows": sum(row["generation_status"] == "technical_error" for row in eligible),
        "forced_only_rows": sum(row["generation_status"] == "not_requested" for row in eligible),
        "endpoint_mismatch_pairs": sum(
            (treated[key]["generation_status"] in NATIVE_STATUSES) != (controls[key]["generation_status"] in NATIVE_STATUSES)
            for key in shared_keys
            if "technical_error" not in {treated[key]["generation_status"], controls[key]["generation_status"]}),
    }
    blockers = []
    if stratification and not stratified_complete:
        blockers.append("frozen_family_counts_not_met")
    if not shared_keys:
        blockers.append("no_complete_execution_pairs")
    if exclusions["technical_error_rows"]:
        blockers.append("technical_execution_errors")
    if exclusions["endpoint_mismatch_pairs"]:
        blockers.append("paired_endpoint_mismatch")
    if exclusions["unpaired_keys"]:
        blockers.append("missing_execution_rows")
    if leaked:
        blockers.append("layout_partition_leakage")
    if identity_conflicts:
        blockers.append("inconsistent_layout_alias_grouping")
    runtime_keys = sorted({str(row["execution_key"]) for row in eligible if row.get("execution_key")})
    spec_hashes = sorted({str(row["spec_sha256"]) for row in eligible if row.get("spec_sha256")})
    head_modes = sorted({str(row.get("inference_configuration", {}).get("head_mode")) for row in eligible
                         if row.get("inference_configuration", {}).get("head_mode")})
    runtime_integrity = {"execution_keys": runtime_keys, "spec_hashes": spec_hashes, "head_modes": head_modes}
    if len(runtime_keys) > 1 or len(spec_hashes) > 1 or len(head_modes) > 1:
        blockers.append("mixed_execution_configuration")
    expected = spec.get("expected_n_layouts")
    if expected is not None and (not isinstance(expected, int) or expected < 1):
        raise ValueError("expected_n_layouts must be a positive integer")
    forced_only = bool(eligible and all(row["generation_status"] == "not_requested" for row in eligible))
    completed_layout_count = len({key[2] for key in shared_keys}) if forced_only else len(layouts)
    completed_pair_count = len(shared_keys) if forced_only else len(native_keys)
    if expected is not None and completed_layout_count != expected:
        blockers.append("frozen_layout_count_not_met")
    if explicit_confirmation and expected is None:
        blockers.append("no_frozen_layout_count")
    expected_pairs = spec.get("expected_n_pairs")
    if expected_pairs is not None and (not isinstance(expected_pairs, int) or expected_pairs < 1):
        raise ValueError("expected_n_pairs must be a positive integer")
    if expected_pairs is not None and completed_pair_count != expected_pairs:
        blockers.append("frozen_pair_count_not_met")
    if explicit_confirmation and expected_pairs is None:
        blockers.append("no_frozen_pair_count")
    if explicit_confirmation and len(layouts) < 32:
        blockers.append("fewer_than_32_confirmation_layouts")
    if explicit_confirmation and forced_only:
        blockers.append("native_confirmation_endpoint_not_measured")
    evidence = dict(spec.get("evidence", {}))
    if any(type(value) is not bool for value in evidence.values()):
        raise ValueError("evidence flags must be booleans")
    for flag in ("technical_valid", "fresh_confirmation", "frozen_specification"):
        if evidence.get(flag) is False:
            blockers.append(f"reviewer_rejected_{flag}")
    families = sorted({str(row["family"]) for row in eligible})
    result = {
        "id": identifier, "experiment_id": experiment, "partition": partition,
        "treatment": treatment, "control": control, "families": families,
        "predicted_direction": direction,
        "semantic_headline_eligible": spec.get("semantic_headline_eligible", True) is True
        and not any(row.get("semantic_headline_eligible") is False
                    or row.get("metadata", {}).get("semantic_headline_eligible") is False
                    or row.get("metadata", {}).get("claim_scope") in {
                        "input_fact_information_routing", "authored_fact_state_sufficiency"}
                    for row in eligible),
        "explicit_confirmation": explicit_confirmation, "n_pairs": len(native_keys),
        "n_execution_pairs": len(shared_keys),
        "n_layouts": len(layouts), "expected_n_layouts": expected,
        "expected_n_pairs": expected_pairs,
        "layout_ids": layout_ids, "primary": estimates,
        "completion_sensitive_effect_bounds": bounds,
        "treatment_completion_rate": average(5),
        "control_completion_rate": average(6),
        "paired_generation_status_counts": {
            "treatment": dict(Counter(treated[key]["generation_status"] for key in native_keys)),
            "control": dict(Counter(controls[key]["generation_status"] for key in native_keys)),
        },
        "secondary_forced": {
            "n_paired_layouts": len(forced_by_layout),
            "n_paired_observations": sum(len(values) for values in forced_by_layout.values()),
            "mean_target_probability_effect": float(np.mean([np.mean(v, axis=0)[0] for v in forced_by_layout.values()])) if forced_by_layout else None,
            "mean_target_log_odds_effect": float(np.mean([np.mean(v, axis=0)[1] for v in forced_by_layout.values()])) if forced_by_layout else None,
            "target_log_odds_denominator": "all_other_actions",
            "mean_target_vs_original_log_odds_effect": float(np.mean([np.mean(values) for values in forced_original_by_layout.values()])) if forced_original_by_layout else None,
            "target_vs_original_n_layouts": len(forced_original_by_layout),
            "log_odds_probability_clip": 1e-7,
        },
        "exclusions": exclusions, "validity_blockers": blockers, "warnings": warnings,
        "runtime_integrity": runtime_integrity,
        "evidence": evidence,
        "missing_semantic_evidence": [flag for flag in SEMANTIC_EVIDENCE if not evidence.get(flag, False)],
        "semantic_control_audit": _semantic_control_audit(rows, spec, native_keys),
        "necessity_claim": "not_established_by_this_analysis",
    }
    if stratification:
        result["family_stratification"] = {
            key: value for key, value in stratification.items() if key != "layout_families"}
        result["family_stratification"].update(
            observed_n_layouts_by_family=family_counts, observed_n_pairs_by_family=family_pair_counts,
            frozen_cohort_complete=stratified_complete,
            interpretation="One fixed family-mixture endpoint; family summaries are descriptive and do not establish each family separately.")
        result["secondary_native_by_family"] = {
            family: {"n_layouts": len(group), "n_pairs": family_pair_counts[family],
                     **{name: float(matrix[group, column].mean()) if len(group) else None
                        for name, column in (("treatment_rate", 0), ("control_rate", 1), ("effect", 2),
                                             ("completion_bound_lower", 3), ("completion_bound_upper", 4),
                                             ("treatment_completion_rate", 5), ("control_completion_rate", 6))}}
            for family, group in zip(stratification["families"], groups)}
    return result


def _interpret_comparison(candidate: dict[str, Any], *, alpha: float, minimum_effect: float) -> None:
    """Apply the shared existing component interpretation gates in place."""
    primary = candidate["primary"]
    ci, effect = primary["adjusted_ci"], primary["effect"]
    direction = candidate["predicted_direction"]
    oriented_effect = -effect if direction == "decrease" and effect is not None else effect
    oriented_ci = [-ci[1], -ci[0]] if direction == "decrease" and ci else ci
    primary["effect_in_predicted_direction"] = oriented_effect if direction != "zero" else None
    sufficient = candidate["n_layouts"] >= 2
    statistically_positive = bool(sufficient and oriented_effect is not None and oriented_ci is not None
                                  and oriented_effect >= minimum_effect and oriented_ci[0] > 0)
    if candidate["explicit_confirmation"]:
        statistically_positive = statistically_positive and primary["holm_p_value"] <= alpha
    if candidate["validity_blockers"]:
        verdict = "technically_invalid"
    elif not candidate["n_pairs"]:
        verdict = "not_measured"
    elif direction == "zero":
        verdict = "consistent_with_equivalence" if sufficient and ci and ci[0] > -minimum_effect and ci[1] < minimum_effect else "inconclusive"
    elif statistically_positive:
        verdict = "supported"
    elif sufficient and oriented_ci and oriented_ci[1] < minimum_effect:
        verdict = "bounded_negative"
    else:
        verdict = "inconclusive"
    candidate["statistical_verdict"] = verdict
    candidate["small_sample_descriptive_only"] = bool(0 < candidate["n_layouts"] < 32)
    if candidate["small_sample_descriptive_only"]:
        candidate["warnings"].append("fewer_than_32_layouts_bootstrap_is_descriptive")
    candidate["report_label"] = (
        "exploratory_reverse_effect" if verdict == "supported" and direction == "decrease" and not candidate["explicit_confirmation"] else
        "exploratory_signal" if verdict == "supported" and not candidate["explicit_confirmation"] else
        "control_equivalence" if direction == "zero" and verdict == "consistent_with_equivalence" else
        "exploratory_small_effect" if verdict == "bounded_negative" and not candidate["explicit_confirmation"] else verdict
    )
    candidate["eventual_action_robust_to_censoring"] = bool(
        candidate["completion_sensitive_effect_bounds"]["adjusted_uncertainty_envelope"]
        and candidate["completion_sensitive_effect_bounds"]["adjusted_uncertainty_envelope"][0] > 0
    )
    candidate["headline_semantic_support"] = bool(
        candidate["explicit_confirmation"] and verdict == "supported"
        and direction == "increase"
        and candidate["semantic_headline_eligible"]
        and not candidate["missing_semantic_evidence"]
        and candidate["semantic_control_audit"]["native_controls_ready"]
        and candidate["eventual_action_robust_to_censoring"]
        and not any("sentence" in family or "positive_control" in family for family in candidate["families"])
    )
    candidate["claim"] = (
        "Held-out evidence supports semantic activation influence on natural actions under the tested intervention."
        if candidate["headline_semantic_support"] else
        "The within-budget natural-action contrast meets the effect criterion; semantic causality remains unestablished."
        if verdict == "supported" else
        "The interval bounds the tested natural-action contrast below the chosen effect threshold."
        if verdict == "bounded_negative" else
        "The control contrast is compatible with equivalence within the declared effect threshold; necessity is not established."
        if verdict == "consistent_with_equivalence" else
        "The native-action endpoint was not measured; forced-answer controls are descriptive only."
        if verdict == "not_measured" else
        "Confirmatory inference is blocked by execution or partition validity."
        if verdict == "technically_invalid" else
        "The tested contrast is inconclusive at the chosen effect threshold."
    )
    if not candidate["explicit_confirmation"]:
        candidate["claim"] = "Exploratory only. " + candidate["claim"]
    if candidate["small_sample_descriptive_only"]:
        candidate["claim"] += (f" This comparison has only {candidate['n_layouts']} independent layouts; "
                               "its bootstrap interval is descriptive and does not establish a population effect.")


def analyze_results(
    rows: Iterable[Mapping[str, Any]] | str | Path, *,
    comparisons: Sequence[Mapping[str, Any]] | None = None,
    n_bootstrap: int = 10000, n_permutations: int = 10000, seed: int = 42,
    alpha: float = 0.05, minimum_effect: float = 0.10,
    confirmation_family_size: int | None = None,
    confirmation_primary_family: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable descriptive statistics and frozen paired tests.

    Each explicit comparison supplies ``experiment_id``, ``treatment``,
    ``control``, optional ``id``, ``partition`` (default confirmation),
    ``expected_n_layouts``, ``expected_n_pairs``, and audited boolean ``evidence``. At most two explicit
    confirmation comparisons are allowed in a family. The legacy
    ``n_permutations`` argument controls centered-null bootstrap draws (default
    10,000); no permutation/exchangeability test is performed. Without comparisons,
    familiar baseline names are paired for exploratory summaries only.

    Supplying ``confirmation_primary_family`` activates an explicit frozen
    registry of at most two candidates with one or two co-primaries each. All
    slots, including absent components, retain their multiplicity adjustment.
    This optional mode rejects an independent ``confirmation_family_size``.
    """
    if confirmation_primary_family is not None:
        from .primary_family import analyze_primary_family
        return analyze_primary_family(
            rows, comparisons=comparisons, confirmation_primary_family=confirmation_primary_family,
            confirmation_family_size=confirmation_family_size, n_bootstrap=n_bootstrap,
            n_permutations=n_permutations, seed=seed, alpha=alpha, minimum_effect=minimum_effect,
        )
    if not 0 < alpha < 1 or not 0 < minimum_effect <= 1:
        raise ValueError("alpha and minimum_effect must be valid probabilities")
    if n_bootstrap < 100 or n_permutations < 100:
        raise ValueError("at least 100 interval and centered-null bootstrap draws are required")
    records = load_results(rows)
    _validate(records)
    specs = [dict(spec) for spec in comparisons] if comparisons is not None else _infer_comparisons(records)
    confirmation_count = sum(spec.get("partition", "confirmation") == "confirmation" and not spec.get("inferred", False)
                             and spec.get("confirmatory", True) for spec in specs)
    if confirmation_count > 2:
        raise ValueError("freeze at most two confirmation comparisons in one family")
    if confirmation_family_size is not None and (type(confirmation_family_size) is not int
                                                or confirmation_family_size not in {1, 2}
                                                or confirmation_family_size < confirmation_count):
        raise ValueError("confirmation_family_size must cover the one or two frozen candidates")
    family_size = confirmation_count if confirmation_family_size is None else confirmation_family_size
    candidates = [_comparison(records, spec, n_bootstrap=n_bootstrap, n_permutations=n_permutations,
                              alpha=alpha, family_size=max(1, family_size), seed=seed) for spec in specs]
    if len({candidate["id"] for candidate in candidates}) != len(candidates):
        raise ValueError("comparison IDs must be unique")
    confirm = [candidate for candidate in candidates if candidate["explicit_confirmation"]]
    if confirm:
        p_values = [candidate["primary"]["p_value"] if candidate["primary"]["p_value"] is not None else 1.0 for candidate in confirm]
        p_values.extend([1.0] * (family_size - len(confirm)))
        adjusted = holm_adjust(p_values)
        for candidate, p_value in zip(confirm, adjusted[:len(confirm)], strict=True):
            candidate["primary"]["holm_p_value"] = p_value
    for candidate in candidates:
        _interpret_comparison(candidate, alpha=alpha, minimum_effect=minimum_effect)
    return {
        "schema_version": "activation_causality_analysis_v1",
        "n_rows": len(records), "n_layouts": len({str(row["layout_id"]) for row in records}),
        "analysis_config": {"version": 3, "alpha": alpha, "minimum_effect": minimum_effect,
                            "n_bootstrap": n_bootstrap, "n_test_bootstrap": n_permutations, "seed": seed,
                            "confirmation_family_size": family_size,
                            "weighting": "equal_layout_then_equal_pair_seed",
                            "primary_endpoint": "natural_target_action_intention_to_treat",
                            "interval_adjustment": "bonferroni_simultaneous",
                            "p_value_adjustment": "holm",
                            "mixed_family_resampling": "within_family_layout_bootstrap",
                            "mixed_family_weighting": "frozen_family_layout_counts_over_total",
                            "mixed_family_constant_guard": "weighted_simultaneous_family_intervals",
                            "mixed_family_constant_p_guard": "two_sided_bounded_layout_hoeffding"},
        "conditions": _descriptive(records), "comparisons": candidates, "cautions": CAUTIONS,
    }


def write_report(
    rows: Iterable[Mapping[str, Any]] | str | Path,
    output_dir: str | Path, *, make_figure: bool = True, **analysis_kwargs: Any,
) -> dict[str, Any]:
    """Write reproducible JSON and Markdown, plus an optional standalone figure."""
    summary = analyze_results(rows, **analysis_kwargs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Activation intervention evidence", "",
             f"{summary['n_rows']} execution rows; {summary['n_layouts']} independent layout IDs.", "",
             "The primary outcome is production of a natural target action within the frozen token budget. Non-completions count as failures; "
             "forced-answer probabilities are secondary. Each layout has equal weight.", ""]
    if summary["comparisons"] and not any(item["explicit_confirmation"] for item in summary["comparisons"]):
        lines.extend(["**All comparisons below are exploratory.** A positive within-budget signal is not held-out confirmation or evidence of a semantic belief mechanism.", ""])
    if any("family_stratification" in item for item in summary["comparisons"]):
        lines.extend(["**Frozen family mixture:** mixed-family comparisons resample layouts within each family at the registered quotas and weights. "
                      "They test one mixture endpoint; descriptive family results do not establish an effect in each family. "
                      "Constant family scores trigger simultaneous family interval guards and a conservative bounded-score Hoeffding p-value.", ""])
    if any(item["small_sample_descriptive_only"] for item in summary["comparisons"]):
        lines.extend(["**Small pilot samples:** contrasts with fewer than 32 independent layouts have descriptive bootstrap intervals. "
                      "Their finite-sample calibration can be poor even when an interval excludes zero.", ""])
    native_comparisons = [item for item in summary["comparisons"] if item["n_pairs"] and item["predicted_direction"] == "increase"]
    if native_comparisons and not any(item["eventual_action_robust_to_censoring"] for item in native_comparisons):
        lines.extend(["**Eventual-action uncertainty remains:** no increasing-action contrast has a strictly positive lower uncertainty bound after completion sensitivity. "
                      "Within-budget success and eventual-action influence must remain separate claims.", ""])
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"
    if "confirmation_primary_family" in summary:
        lines.extend(["## Frozen co-primary family", "",
                      f"All {summary['analysis_config']['confirmation_primary_family_size']} frozen components retain their multiplicity slots. "
                      "Each candidate requires every registered component; technical invalidity takes precedence over pending components.", "",
                      "Registry SHA256: `" + summary["confirmation_primary_family_sha256"] + "`.", "",
                      "Frozen registry and bindings: `" + json.dumps(summary["confirmation_primary_family"], sort_keys=True) + "`.", "",
                      "| Candidate / primary | Component state | Finding | Adjustment input p | Holm p |",
                      "|---|---|---|---:|---:|"])
        for component in summary["primary_components"]:
            label = (component["candidate_id"] + "/" + component["primary_id"]).replace("|", "\\|")
            lines.append(f"| {label} | {component['component_status']} | {component['statistical_verdict']} "
                         f"| {fmt(component['primary']['multiplicity_input_p_value'])} | {fmt(component['primary']['holm_p_value'])} |")
        lines.extend(["", "| Candidate conjunction | State | Invalid components | Pending components | Eventual-action robust | Semantic support |",
                      "|---|---|---|---|---|---|"])
        for candidate in summary["candidate_conjunctions"]:
            lines.append(f"| {candidate['candidate_id']} | {candidate['status']} "
                         f"| {', '.join(candidate['invalid_primary_ids']) or 'none'} "
                         f"| {', '.join(candidate['pending_primary_ids']) or 'none'} "
                         f"| {candidate['eventual_action_robust_to_censoring']} | {candidate['headline_semantic_support']} |")
        lines.extend(["", "Holm results are provisional while registered components remain pending; final joint recomputation retains the same registry and family size."
                      if summary["multiplicity_adjustment_provisional"] else
                      "All registered components have final states; technical failures still block their candidate conjunction.", ""])
    lines.extend(["## Observed condition outcomes", "",
                  "Forced top-action counts describe conditional forced-answer readouts, not native behavior.", "",
                  "| Experiment / condition | Partition | Forced target top / trials | Mean forced target probability | Native trials | Native target rate |",
                  "|---|---|---:|---:|---:|---:|"])
    for condition in summary["conditions"]:
        lines.append(f"| {condition['experiment_id']} / {condition['condition']} | {condition['partition']} "
                     f"| {condition['forced_target_top_count']} / {condition['n_forced_rows']} "
                     f"| {fmt(condition['descriptive_row_weighted_forced_target_probability'])} "
                     f"| {condition['n_natural_rows']} | {fmt(condition['layout_weighted_native_target_rate'])} |")
    lines.extend(["", "## Paired forced-answer controls", "",
                  "| Comparison | Layouts / pairs | Mean Δ target probability | Mean Δ log(target/original) |",
                  "|---|---:|---:|---:|"])
    for candidate in summary["comparisons"]:
        secondary = candidate["secondary_forced"]
        lines.append(f"| {candidate.get('display_id', candidate['id'])} | {secondary['n_paired_layouts']} / {secondary['n_paired_observations']} "
                     f"| {fmt(secondary['mean_target_probability_effect'])} | {fmt(secondary['mean_target_vs_original_log_odds_effect'])} |")
    lines.extend(["", "## Native-action inference", "",
                  "Effects and intervals concern production within the token budget. Completion-sensitive point bounds allow unresolved outputs "
                  "to take any action and are shown before adding sampling uncertainty.", "",
                  "| Comparison | Partition | Layouts / pairs | Within-budget effect | Bootstrap interval | Completion-sensitive bounds | Holm p | Interpretation | Semantic support |",
                  "|---|---|---:|---:|---|---|---:|---|---|"])
    for candidate in summary["comparisons"]:
        primary = candidate["primary"]
        interval = primary["adjusted_ci"]
        interval_text = f"[{fmt(interval[0])}, {fmt(interval[1])}]" if interval else "n/a"
        bounds = candidate["completion_sensitive_effect_bounds"]
        bounds_text = f"[{fmt(bounds['lower'])}, {fmt(bounds['upper'])}]" if bounds["lower"] is not None else "n/a"
        label = candidate.get("display_id", candidate["id"]).replace("|", "\\|")
        lines.append(f"| {label} | {candidate['partition']} | {candidate['n_layouts']} / {candidate['n_pairs']} "
                     f"| {fmt(primary['effect'])} | {interval_text} | {bounds_text} | {fmt(primary['holm_p_value'])} "
                     f"| {candidate['report_label']} | {candidate['headline_semantic_support']} |")
    for candidate in summary["comparisons"]:
        lines.extend(["", f"## {candidate.get('display_id', candidate['id'])}", "", candidate["claim"], "",
                      "- Execution and leakage exclusions: `" + json.dumps(candidate["exclusions"], sort_keys=True) + "`.",
                      "- Completion counts: `" + json.dumps(candidate["paired_generation_status_counts"], sort_keys=True) + "`.",
                      "- Prespecified effect direction: " + candidate["predicted_direction"] + ".",
                      "- Completion-sensitive effect bounds: `" + json.dumps(candidate["completion_sensitive_effect_bounds"], sort_keys=True) + "`.",
                      "- Missing semantic evidence: " + (", ".join(candidate["missing_semantic_evidence"]) or "none") + ".",
                      "- Native semantic controls: `" + json.dumps(candidate["semantic_control_audit"], sort_keys=True) + "`.",
                      "- Execution configuration: `" + json.dumps(candidate["runtime_integrity"], sort_keys=True) + "`.",
                      "- Validity blockers: " + (", ".join(candidate["validity_blockers"]) or "none") + ".",
                      "- Interval warnings: " + (", ".join(candidate["warnings"]) or "none") + ".",
                      "- Necessity and mediation are not established by this contrast."])
        if "family_stratification" in candidate:
            lines.extend(["- Frozen family mixture and observed counts: `" + json.dumps(candidate["family_stratification"], sort_keys=True) + "`.",
                          "- Descriptive native family outcomes: `" + json.dumps(candidate["secondary_native_by_family"], sort_keys=True) + "`.",
                          "- Mixture p-value method: " + candidate["primary"].get("p_value_method", "not_run_incomplete_frozen_cohort") + "."])
    lines.extend(["", "## Interpretation limits", "", *(f"- {item}" for item in summary["cautions"]), ""])
    if make_figure and summary["comparisons"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError:
            lines.extend(["Matplotlib is unavailable; no figure was produced.", ""])
        else:
            plotted = [item for item in summary["comparisons"] if item["primary"]["adjusted_ci"]]
            if plotted:
                fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(plotted) + 1.5)))
                for index, item in enumerate(plotted):
                    low, high = item["primary"]["adjusted_ci"]
                    ax.plot([low, high], [index, index], color="#34638a", lw=2)
                    ax.scatter([item["primary"]["effect"]], [index], color="#34638a", zorder=3)
                ax.set_yticks(range(len(plotted)), [item.get("display_id", item["id"]) for item in plotted])
                ax.axvline(0, color="#555555", lw=0.8)
                ax.axvline(summary["analysis_config"]["minimum_effect"], color="#a85936", ls="--", lw=1)
                ax.set_xlabel("Change in within-budget native target-action success (treatment − control)")
                ax.set_title("Paired layout effects and bootstrap intervals" if any(item["explicit_confirmation"] for item in plotted)
                             else "Exploratory within-budget effects; pilot intervals are descriptive")
                ax.invert_yaxis()
                fig.tight_layout()
                fig.savefig(destination / "effects.png", dpi=180)
                fig.savefig(destination / "effects.pdf")
                plt.close(fig)
                lines.extend(["![Paired layout effects](effects.png)", ""])
    (destination / "report.md").write_text("\n".join(lines))
    return summary
