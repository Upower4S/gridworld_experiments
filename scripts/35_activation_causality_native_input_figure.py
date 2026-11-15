#!/usr/bin/env python3
"""Descriptive native input-fact outcomes; complete frozen cohorts only.

--campaign DIR --experiment-id ID --output STEM writes PNG, PDF and counts JSON.
No model is imported or loaded. No incomplete result cohort is read: filenames
must first match every planned job. Native target/other/unresolved rates include
all paired trials, with layout counts reported separately. No CIs or semantic
claims are generated, including for a confirmation-partition source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
FAMILIES = ("wall", "key_possession", "door_status")
KINDS = ("identity", "factspan", "random_factspan", "lasttoken")
NATIVE_STATUSES = ("complete", "truncated", "malformed")
SCOPE = "input_fact_information_routing"


def read_hashed(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def condition_label(condition: str) -> str:
    if condition == "baseline":
        return "Recipient text baseline"
    if condition == "counterfactual_text":
        return "Physical counterfactual text"
    kind, layer = condition.rsplit("_layer", 1)
    label = {"identity": "recipient identity", "factspan": "changed fact span",
             "random_factspan": "norm-matched random", "lasttoken": "last header token"}[kind]
    return f"L{layer}: {label}"


def _validate_native(row: dict) -> None:
    identifier = row.get("job_id")
    status, action = row.get("generation_status"), row.get("native_action")
    if row.get("status") != "complete" or status not in NATIVE_STATUSES:
        raise ValueError(f"Incomplete/non-native execution: {identifier}")
    tokens = row.get("generated_token_ids")
    count = row.get("generated_tokens")
    if not isinstance(tokens, list) or type(count) is not int or count < 1 or len(tokens) != count:
        raise ValueError(f"Missing/inconsistent native token audit: {identifier}")
    if status == "complete":
        if action not in ACTIONS or row.get("parse_status") != "valid" or row.get("endpoint") != "native_final":
            raise ValueError(f"Invalid native final action: {identifier}")
        try:
            payload = json.loads(row.get("final_text", ""))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid native final JSON: {identifier}") from exc
        if not isinstance(payload, dict) or str(payload.get("action", "")).strip().upper() != action:
            raise ValueError(f"Native final JSON/action mismatch: {identifier}")
    elif action is not None:
        raise ValueError(f"Unresolved native output has an action: {identifier}")
    if status == "truncated" and count != row["max_new_tokens"]:
        raise ValueError(f"Truncation does not match the frozen cap: {identifier}")
    if count > row["max_new_tokens"]:
        raise ValueError(f"Generation exceeds the frozen cap: {identifier}")


def _validate_identity(identity: dict, baseline: dict, layer: int) -> None:
    for key in ("generation_status", "native_action", "generated_token_ids"):
        if identity.get(key) != baseline.get(key):
            raise ValueError(f"Identity native sequence differs from baseline: {identity['job_id']}")
    q, base_q = identity.get("forced_q", {}), baseline.get("forced_q", {})
    if set(q) != set(ACTIONS) or set(base_q) != set(ACTIONS) or q != base_q or any(
        type(q[a]) not in (int, float) or not math.isfinite(q[a]) or not 0 <= q[a] <= 1 for a in ACTIONS
    ) or abs(sum(q.values()) - 1) > 1e-5:
        raise ValueError(f"Identity forced distribution is invalid or differs: {identity['job_id']}")
    logs = identity.get("patch_log", [])
    if len(logs) != 1 or logs[0].get("layer") != layer or not logs[0].get("delta_l2") or any(
        type(value) not in (int, float) or value != 0 for value in logs[0]["delta_l2"]
    ):
        raise ValueError(f"Identity patch is not audited as zero: {identity['job_id']}")


def load_counts(campaign: Path, experiment_id: str) -> dict:
    """Load only a complete native input-fact cohort with exact paired controls."""
    campaign = Path(campaign).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Invalid experiment ID")
    if (campaign / "reviews" / f"{experiment_id}_retired.json").exists():
        raise ValueError("Refusing retired experiment")
    spec_path = campaign / "specs" / f"{experiment_id}.json"
    spec, spec_hash = read_hashed(spec_path)
    if spec.get("experiment_id") != experiment_id or spec.get("partition") not in {"discovery", "confirmation"}:
        raise ValueError("Require matching discovery/confirmation specification")
    if str(spec.get("claim_scope", "")).split(";", 1)[0].strip() != SCOPE:
        raise ValueError("Wrong claim scope: require input_fact_information_routing")
    if spec.get("partition") == "confirmation" and not spec.get("freeze_id"):
        raise ValueError("Confirmation source is missing its freeze ID")
    if spec.get("cohort_complete") is False:
        raise ValueError("Frozen confirmation cohort is underfilled")
    if spec.get("semantic_headline_eligible") is True:
        raise ValueError("Input-fact source cannot be labeled semantic headline evidence")
    jobs = spec.get("jobs", [])
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Require nonempty unique planned jobs")
    if any(j.get("natural") is not True or j.get("partition") != spec["partition"]
           or j.get("semantic_test") is not False or j.get("family") not in FAMILIES for j in jobs):
        raise ValueError("Require native input-fact jobs in the actual source partition")
    layers = sorted({int(match.group(1)) for job in jobs
                     if (match := re.fullmatch(r"factspan_layer(\d+)", job["condition"]))})
    conditions = ["baseline", "counterfactual_text"] + [f"{kind}_layer{layer}" for layer in layers for kind in KINDS]
    if not layers or set(conditions) != {j["condition"] for j in jobs}:
        raise ValueError("Missing/extra source conditions: require all four controls at every layer")
    result_dir = campaign / "results" / experiment_id
    planned = {j["job_id"] for j in jobs}
    present = {p.stem for p in result_dir.glob("*.json")}
    # This check deliberately precedes all raw-row reads, including corrupt
    # or half-written source rows from an active, incomplete worker cohort.
    if planned != present:
        raise ValueError(f"Incomplete/extra job cohort: {len(planned - present)} missing, {len(present - planned)} extra")
    groups, sources, execution_ids, aliases = defaultdict(dict), [], set(), {}
    for job in jobs:
        pair = job.get("metadata", {}).get("pair", {})
        if (pair.get("partition") != spec["partition"] or pair.get("pair_id") != job["pair_id"]
                or pair.get("layout_id") != job["layout_id"] or pair.get("family") != job["family"]
                or pair.get("counterfactual_optimal_actions") != [job["target_action"]]
                or pair.get("recipient_optimal_actions") != [job["original_action"]]
                or job["target_action"] not in ACTIONS or job["original_action"] not in ACTIONS
                or job["target_action"] == job["original_action"]):
            raise ValueError("Frozen physical pair/target metadata is inconsistent")
        representation = "pre_reasoning_analysis_header_token" if job["condition"].startswith("lasttoken_layer") else "input_fact_tokens"
        if (job.get("metadata", {}).get("claim_scope") != SCOPE
                or job["metadata"].get("representation_kind") != representation
                or job["metadata"].get("abstract_belief_representation_claim") is not False):
            raise ValueError("Job lacks explicit input-fact representation/claim scope")
        for alias in {job["layout_id"], *job.get("layout_aliases", []),
                      *pair.get("metadata", {}).get("variant_layout_ids", [])}:
            if alias in aliases and aliases[alias] != job["layout_id"]:
                raise ValueError("Distinct layout denominators share a recorded alias")
            aliases[alias] = job["layout_id"]
        path = result_dir / f"{job['job_id']}.json"
        row, row_hash = read_hashed(path)
        if row.get("spec_sha256") != spec_hash:
            raise ValueError(f"Stale result/spec hash: {job['job_id']}")
        for key in ("experiment_id", "job_id", "condition", "layout_id", "pair_id", "family", "partition", "seed",
                    "target_action", "original_action", "natural", "max_new_tokens", "temperature", "metadata",
                    "intervention", "selection", "recipient_positions", "donor_positions",
                    "recipient_token_ids", "donor_token_ids"):
            if row.get(key) != job.get(key):
                raise ValueError(f"Result differs from frozen {key}: {job['job_id']}")
        if type(job.get("max_new_tokens")) is not int or job["max_new_tokens"] < 1:
            raise ValueError("Invalid frozen native token cap")
        _validate_native(row)
        identity = (row.get("engine_key"), row.get("execution_key"),
                    json.dumps(row.get("inference_configuration"), sort_keys=True))
        if any(not value or value == "null" for value in identity):
            raise ValueError("Missing execution identity")
        execution_ids.add(identity)
        key = (job["layout_id"], job["pair_id"], job["seed"])
        if job["condition"] in groups[key]:
            raise ValueError("Duplicate paired trial condition")
        groups[key][job["condition"]] = row
        sources.append({"job_id": job["job_id"], "path": str(path), "sha256": row_hash})
    if len(execution_ids) != 1:
        raise ValueError("Mixed engine/execution/inference configurations")
    identity_jobs = []
    layout_family = {}
    trials = []
    sites = set()
    for key, group in sorted(groups.items()):
        if set(group) != set(conditions):
            raise ValueError("Not every paired trial has every source control condition")
        base = group["baseline"]
        if any(len({json.dumps(row[field], sort_keys=True) for row in group.values()}) != 1
               for field in ("family", "target_action", "original_action", "max_new_tokens", "temperature")):
            raise ValueError("Paired target, family or generation settings differ")
        if key[0] in layout_family and layout_family[key[0]] != base["family"]:
            raise ValueError("One layout belongs to multiple families")
        layout_family[key[0]] = base["family"]
        for layer in layers:
            _validate_identity(group[f"identity_layer{layer}"], base, layer)
            identity_jobs.append(group[f"identity_layer{layer}"]["job_id"])
            for kind in KINDS:
                intervention = group[f"{kind}_layer{layer}"]["intervention"]
                if (intervention.get("layers") != [layer] or intervention.get("operation") != "replace"
                        or intervention.get("dose") != 1.0
                        or intervention.get("random_control", False) != (kind == "random_factspan")):
                    raise ValueError("Unexpected source intervention/control profile")
                sites.add(intervention["site"])
        trials.append({"layout_id": key[0], "pair_id": key[1], "seed": key[2], "family": base["family"],
                       "target_action": base["target_action"], "outcomes": [{
                           "condition": condition, "job_id": group[condition]["job_id"],
                           "generation_status": group[condition]["generation_status"],
                           "native_action": group[condition]["native_action"],
                           "generated_tokens": group[condition]["generated_tokens"],
                       } for condition in conditions]})
    if len(sites) != 1:
        raise ValueError("Mixed intervention sites")
    layout_counts = {family: sum(value == family for value in layout_family.values()) for family in FAMILIES}
    trial_counts = {family: sum(t["family"] == family for t in trials) for family in FAMILIES}
    for field, actual in (("expected_n_layouts", len(layout_family)), ("expected_n_pairs", len(trials)),
                          ("requested_n_layouts", len(layout_family))):
        if field in spec and spec[field] != actual:
            raise ValueError(f"Frozen {field} differs from the complete plotted cohort")
    for field, actual in (("expected_n_layouts_by_family", layout_counts),
                          ("requested_n_layouts_by_family", layout_counts),
                          ("expected_n_pairs_by_family", trial_counts)):
        if field in spec and spec[field] != actual:
            raise ValueError(f"Frozen {field} differs from the complete family denominators")
    panels = []
    for family in (None, *FAMILIES):
        selected = [t for t in trials if family is None or t["family"] == family]
        panel = {"id": family or "all", "title": "All layouts" if family is None else {
            "wall": "Wall", "key_possession": "Key possession", "door_status": "Door status"}[family],
                 "n_layouts": len({t["layout_id"] for t in selected}), "n_paired_trials_per_condition": len(selected),
                 "n_source_jobs": len(selected) * len(conditions), "conditions": []}
        for condition in conditions:
            rr = [next(r for r in t["outcomes"] if r["condition"] == condition) for t in selected]
            target = sum(r["generation_status"] == "complete" and r["native_action"] == t["target_action"]
                         for r, t in zip(rr, selected))
            status_counts = Counter(r["generation_status"] for r in rr)
            other = status_counts["complete"] - target
            unresolved = status_counts["truncated"] + status_counts["malformed"]
            n = len(rr)
            panel["conditions"].append({"condition": condition, "label": condition_label(condition), "n": n,
                                        "target_count": target, "other_count": other, "unresolved_count": unresolved,
                                        "target_rate": target / n if n else None,
                                        "other_rate": other / n if n else None,
                                        "unresolved_rate": unresolved / n if n else None,
                                        "native_status_counts": {s: status_counts[s] for s in NATIVE_STATUSES},
                                        "job_ids": [r["job_id"] for r in rr]})
        panels.append(panel)
    engine_key, execution_key, config = next(iter(execution_ids))
    raw_manifest = {source["job_id"]: source["sha256"] for source in sources}
    manifest_hash = hashlib.sha256(json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    statuses = Counter(row["generation_status"] for group in groups.values() for row in group.values())
    return {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign), "experiment_id": experiment_id, "partition": spec["partition"],
        "freeze_id": spec.get("freeze_id"), "claim_scope": SCOPE, "semantic_headline_eligible": False,
        "spec_path": str(spec_path), "spec_sha256": spec_hash, "source_pairs_sha256": spec.get("source_pairs_sha256"),
        "engine_key": engine_key, "execution_key": execution_key, "inference_configuration": json.loads(config),
        "n_planned_jobs": len(jobs), "n_completed_jobs": len(sources), "n_layouts": len(layout_family),
        "n_distinct_pairs": len({(t["layout_id"], t["pair_id"]) for t in trials}), "n_paired_trials": len(trials),
        "layouts_per_family": layout_counts, "trials_per_family": trial_counts,
        "native_status_counts": {status: statuses[status] for status in NATIVE_STATUSES},
        "seeds": sorted({j["seed"] for j in jobs}), "token_caps": sorted({j["max_new_tokens"] for j in jobs}),
        "temperatures": sorted({j["temperature"] for j in jobs}), "layers": layers, "site": next(iter(sites)),
        "condition_order": conditions, "panels": panels, "paired_trials": trials,
        "identity_audit": {"n_identity_jobs": len(identity_jobs), "job_ids": sorted(identity_jobs),
                           "exact_native_sequences_and_forced_q": True, "zero_patch_norms": True},
        "pre_query_exclusions": spec.get("exclusions", []), "source_pair_counts": {
            k: spec.get(k) for k in ("n_source_pairs", "n_aligned_pairs", "n_excluded_pairs")},
        "source_rows": sorted(sources, key=lambda r: r["job_id"]),
        "raw_rows_manifest_sha256": manifest_hash,
        "raw_rows_manifest_hash_definition": "SHA256 of UTF-8 JSON job_id-to-raw-file-SHA256 mapping; sorted keys and compact separators",
        "definitions": {
            "target": "Native final-channel JSON equals the simulator's unique optimal action in the paired physical counterfactual; not the observed donor model action.",
            "other": "A complete native final action differs from that physical-counterfactual target.",
            "unresolved": "Truncated or malformed native output; included in every trial denominator; never replaced by a forced answer.",
            "weighting": "Descriptive pooled paired-trial rates within each panel, not confidence intervals; distinct layouts and repeated trials are reported separately.",
            "scope": "Input-fact information routing diagnostic; no semantic-belief, necessity, mediation or population-effect claim, regardless of source partition.",
            "forced_scores": "Not plotted. Forced q is consulted only for exact identity-control validation; see the separate forced-TV diagnostic.",
        },
    }


def render(counts: dict, output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    if output.suffix in {".png", ".pdf", ".json"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(counts["condition_order"])
    height = max(7.5, 3.7 + n_rows * 0.39)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(17.6, height), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.984, top=1 - 1.85 / height, bottom=1.68 / height, wspace=0.12)
    colors = {"target": "#168371", "other": "#7893b0", "unresolved": "#d2d5da"}
    for ax, panel in zip(axes, counts["panels"]):
        for i, row in enumerate(panel["conditions"]):
            if row["condition"] in {"baseline", "counterfactual_text"}:
                ax.axhspan(i - 0.46, i + 0.46, color="#f0f3f6", zorder=0)
            if row["n"]:
                left = 0
                for category in ("target", "other", "unresolved"):
                    value = row[f"{category}_rate"]
                    ax.barh(i, value, left=left, height=0.58, color=colors[category],
                            hatch="///" if category == "unresolved" else None,
                            edgecolor="#77828f" if category == "unresolved" else "white", linewidth=0.35, zorder=2)
                    left += value
                ax.text(1.025, i, f"{row['target_count']}/{row['other_count']}/{row['unresolved_count']}",
                        va="center", fontsize=8.4, color="#30485e")
            else:
                ax.text(0.5, i, "-", ha="center", va="center", color="#8594a2")
        ax.set_xlim(0, 1.34)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(panel["title"], loc="left", fontsize=12, fontweight="bold", color="#253d51", pad=24)
        ax.text(0, 1.02, f"{panel['n_layouts']} layouts; {panel['n_paired_trials_per_condition']} trials per row",
                transform=ax.transAxes, fontsize=8.8, color="#4b6276")
        ax.tick_params(axis="both", length=0, pad=6, colors="#3e586f")
        ax.grid(axis="x", color="#e4e9ee", linewidth=0.6, zorder=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(n_rows))
    axes[0].set_yticklabels([condition_label(c) for c in counts["condition_order"]], fontsize=10)
    axes[0].set_ylim(n_rows - 0.35, -0.65)
    fig.text(0.035, 1 - 0.35 / height, "Native actions after input-fact interventions", fontsize=18,
             fontweight="bold", color="#20394d")
    caps = ", ".join(f"{c:,}" for c in counts["token_caps"])
    seeds = ", ".join(map(str, counts["seeds"]))
    fig.text(0.035, 1 - 0.68 / height,
             f"{counts['experiment_id']}  |  {counts['partition']}  |  seed(s) {seeds}  |  cap {caps} new tokens  |  "
             f"{counts['n_completed_jobs']}/{counts['n_planned_jobs']} completed executions",
             fontsize=10.5, color="#435e73")
    fig.text(0.035, 1 - 0.97 / height,
             "Target = the simulator-optimal action in the changed physical world. Counts at right: target / other / unresolved.",
             fontsize=10.5, color="#435e73")
    fig.legend(handles=[Patch(facecolor=colors["target"], label="Native target action"),
                        Patch(facecolor=colors["other"], label="Other completed native action"),
                        Patch(facecolor=colors["unresolved"], edgecolor="#77828f", hatch="///", label="Unresolved: truncated or malformed")],
               loc="lower left", bbox_to_anchor=(0.029, 1.05 / height), ncol=3, frameon=False, fontsize=10)
    fig.text(0.035, 0.92 / height,
             "Every row uses the same paired layout/pair/seed cohort. Unresolved outputs remain in denominators; no forced-answer fallback.",
             fontsize=10, color="#435e73")
    fig.text(0.035, 0.65 / height,
             "Changed fact spans are compared with same-site norm-matched random writes and the last header token. Layer indices are zero-based.",
             fontsize=10, color="#435e73")
    fig.text(0.035, 0.38 / height,
             "Descriptive native outcomes only: no confidence intervals or semantic-belief claims. Forced distributions are shown in a separate diagnostic.",
             fontsize=9.8, color="#435e73")
    fig.text(0.035, 0.12 / height,
             f"Spec SHA256 {counts['spec_sha256'][:16]}...  |  "
             f"{counts['identity_audit']['n_identity_jobs']} exact identity checks  |  Full counts and source hashes: companion .counts.json",
             fontsize=8.2, color="#6b8092")
    png, pdf, raw = (Path(str(output) + suffix) for suffix in (".png", ".pdf", ".counts.json"))
    counts["figure_script"] = {"path": str(Path(__file__).resolve()),
                               "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    counts["artifacts"] = {"png": str(png.resolve()), "pdf": str(pdf.resolve()), "counts": str(raw.resolve())}
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Native input-fact outcomes by task family"})
    plt.close(fig)
    raw.write_text(json.dumps(counts, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return png, pdf, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        counts = load_counts(args.campaign, args.experiment_id)
        outputs = render(counts, args.output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"Cannot render native-input figure: {exc}\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
