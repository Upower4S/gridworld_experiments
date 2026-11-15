#!/usr/bin/env python3
"""Plot a post-hoc input-routing diagnostic from frozen forced-only results.

Writes <output>.png, <output>.pdf and <output>.counts.json. There are no model
loads, confidence intervals or significance tests. Within each layout, average
paired cut/seed TV distances first; the bold curves weight layouts equally.
Every frozen job and identity control must be complete and valid before drawing.

Example:
  interp/.venv/bin/python scripts/32_activation_causality_routing_figure.py \
      --campaign outputs/activation_causality/campaign_20260912 \
      --experiment-id input_fact_forced_v1 \
      --output outputs/activation_causality/campaign_20260912/figures/input_fact_forced_v1_routing
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
from statistics import mean


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
SERIES = ("factspan", "random_factspan", "lasttoken")
LABELS = {"factspan": "Changed input-fact tokens", "random_factspan": "Per-position norm-matched random",
          "lasttoken": "Last analysis-header token"}
COLORS = {"factspan": "#177f73", "random_factspan": "#bd7243", "lasttoken": "#486a99"}


def read_hashed(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def probabilities(row: dict) -> dict[str, float]:
    q = row.get("forced_q", {})
    if set(q) != set(ACTIONS) or any(
        type(q[a]) not in (float, int) or not math.isfinite(q[a]) or not 0 <= q[a] <= 1 for a in ACTIONS
    ) or abs(sum(q.values()) - 1) > 1e-5:
        raise ValueError(f"Invalid four-action distribution: {row.get('job_id')}")
    top = row.get("forced_top_action")
    if top not in ACTIONS or abs(q[top] - max(q.values())) > 1e-8:
        raise ValueError(f"Forced argmax disagrees with distribution: {row.get('job_id')}")
    return {action: float(q[action]) for action in ACTIONS}


def tv(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left[action] - right[action]) for action in ACTIONS)


def load_diagnostic(campaign: Path, experiment_id: str) -> dict:
    """Require a complete paired cohort, frozen identities and exact no-op controls."""
    campaign = Path(campaign).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Invalid experiment_id")
    retired = campaign / "reviews" / f"{experiment_id}_retired.json"
    if retired.exists():
        raise ValueError(f"Refusing retired experiment: {retired}")
    spec_path = campaign / "specs" / f"{experiment_id}.json"
    spec, spec_hash = read_hashed(spec_path)
    if spec.get("experiment_id") != experiment_id or spec.get("partition") != "discovery":
        raise ValueError("Require matching frozen discovery spec")
    if not str(spec.get("claim_scope", "")).startswith("input_fact_information_routing"):
        raise ValueError("Require an input-fact information-routing specification")
    jobs = spec.get("jobs", [])
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Require nonempty unique frozen jobs")
    if any(j.get("natural") is not False or j.get("partition") != "discovery" or j.get("semantic_test") is not False for j in jobs):
        raise ValueError("Require forced-only discovery jobs with no semantic-test claim")
    condition_set = {j["condition"] for j in jobs}
    layers = sorted(int(m.group(1)) for c in condition_set if (m := re.fullmatch(r"factspan_layer(\d+)", c)))
    expected_conditions = {"baseline", "counterfactual_text"} | {
        f"{kind}_layer{layer}" for layer in layers for kind in (*SERIES, "identity")
    }
    if not layers or expected_conditions != condition_set:
        raise ValueError("Require both text baselines and fact/random/lasttoken/identity at every layer")
    result_dir = campaign / "results" / experiment_id
    planned_ids = {j["job_id"] for j in jobs}
    present_ids = {p.stem for p in result_dir.glob("*.json")}
    if present_ids != planned_ids:
        raise ValueError(f"Incomplete or unexpected cohort: {len(planned_ids - present_ids)} missing, "
                         f"{len(present_ids - planned_ids)} unexpected rows")
    groups, source_rows, identities = defaultdict(dict), [], set()
    for job in jobs:
        path = result_dir / f"{job['job_id']}.json"
        row, row_hash = read_hashed(path)
        if row.get("spec_sha256") != spec_hash:
            raise ValueError(f"Spec hash mismatch: {job['job_id']}")
        for key in ("experiment_id", "job_id", "partition", "pair_id", "layout_id", "family", "seed",
                    "condition", "target_action", "original_action", "natural", "intervention", "selection",
                    "recipient_token_ids", "donor_token_ids", "recipient_positions", "donor_positions"):
            if row.get(key) != job.get(key):
                raise ValueError(f"Result differs from frozen {key}: {job['job_id']}")
        if row.get("status") != "complete" or row.get("generation_status") != "not_requested" or row.get("native_action") is not None:
            raise ValueError(f"Require complete forced-only endpoint: {job['job_id']}")
        probabilities(row)
        identity = (row.get("engine_key"), row.get("execution_key"),
                    json.dumps(row.get("inference_configuration"), sort_keys=True))
        if any(not part or part == "null" for part in identity):
            raise ValueError("Missing engine/execution/inference identity")
        identities.add(identity)
        key = (job["layout_id"], job["pair_id"], job["seed"])
        if job["condition"] in groups[key]:
            raise ValueError(f"Duplicate paired observation: {key}")
        groups[key][job["condition"]] = row
        source_rows.append({"job_id": job["job_id"], "path": str(path), "sha256": row_hash})
    if len(identities) != 1:
        raise ValueError("Mixed execution configurations")
    if any(set(group) != expected_conditions for group in groups.values()):
        raise ValueError("Incomplete paired condition grid")
    observations, identity_reports = [], []
    layout_observations = defaultdict(list)
    sites = set()
    for key, group in sorted(groups.items()):
        if len({r["target_action"] for r in group.values()}) != 1 or len({r["family"] for r in group.values()}) != 1:
            raise ValueError(f"Paired target or family mismatch: {key}")
        baseline, cf = group["baseline"], group["counterfactual_text"]
        base_q, cf_q = probabilities(baseline), probabilities(cf)
        observation = {"layout_id": key[0], "pair_id": key[1], "seed": key[2], "family": baseline["family"],
                       "target_action": baseline["target_action"], "baseline_job_id": baseline["job_id"],
                       "physical_counterfactual_job_id": cf["job_id"], "baseline_q": base_q,
                       "physical_counterfactual_q": cf_q, "baseline_tv": tv(base_q, cf_q), "series": []}
        for layer in layers:
            identity = group[f"identity_layer{layer}"]
            q_error = max(abs(probabilities(identity)[a] - base_q[a]) for a in ACTIONS)
            logs = identity.get("patch_log", [])
            if len(logs) != 1 or logs[0].get("layer") != layer:
                raise ValueError(f"Missing identity patch audit: {identity['job_id']}")
            deltas = logs[0].get("delta_l2", [])
            if not deltas or any(type(v) not in (float, int) or not math.isfinite(v) or v != 0 for v in deltas) or q_error != 0:
                raise ValueError(f"Identity control is not exact: {identity['job_id']}")
            identity_reports.append({"job_id": identity["job_id"], "layout_id": key[0], "layer": layer,
                                     "forced_q_max_absolute_difference": q_error, "patch_delta_l2_max": max(deltas)})
            for kind in SERIES:
                row = group[f"{kind}_layer{layer}"]
                intervention = row["intervention"]
                if intervention.get("layers") != [layer] or intervention.get("operation") != "replace" or intervention.get("dose") != 1.0:
                    raise ValueError("Routing panel requires a single full-replacement layer")
                if intervention.get("random_control", False) != (kind == "random_factspan"):
                    raise ValueError("Incorrect random-control identity")
                sites.add(intervention["site"])
                q = probabilities(row)
                observation["series"].append({"kind": kind, "layer": layer, "job_id": row["job_id"],
                                              "q": q, "tv_distance": tv(q, cf_q)})
        observations.append(observation)
        layout_observations[key[0]].append(observation)
    if len(sites) != 1:
        raise ValueError("Mixed intervention sites cannot share this layer diagnostic")
    per_layout = []
    for layout, rr in sorted(layout_observations.items()):
        families = {r["family"] for r in rr}
        if len(families) != 1:
            raise ValueError("A layout appears in multiple fact families")
        record = {"layout_id": layout, "family": next(iter(families)), "n_observations": len(rr),
                  "baseline_tv": mean(r["baseline_tv"] for r in rr), "series": []}
        for kind in SERIES:
            for layer in layers:
                values = [v["tv_distance"] for r in rr for v in r["series"] if v["kind"] == kind and v["layer"] == layer]
                record["series"].append({"kind": kind, "layer": layer, "tv_distance": mean(values),
                                         "n_observations": len(values)})
        per_layout.append(record)
    summary = []
    for kind in SERIES:
        for layer in layers:
            values = [v["tv_distance"] for r in per_layout for v in r["series"] if v["kind"] == kind and v["layer"] == layer]
            summary.append({"kind": kind, "layer": layer, "mean_tv_distance": mean(values),
                            "n_layouts": len(values), "n_paired_observations": len(observations)})
    engine_key, execution_key, config = next(iter(identities))
    return {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign), "experiment_id": experiment_id, "partition": "discovery",
        "spec_path": str(spec_path), "spec_sha256": spec_hash,
        "source_pairs_sha256": spec.get("source_pairs_sha256"),
        "source_dataset_or_experiment": spec.get("source_dataset_or_experiment"), "tokenizer": spec.get("tokenizer"),
        "engine_key": engine_key, "execution_key": execution_key, "inference_configuration": json.loads(config),
        "n_planned_jobs": len(jobs), "n_completed_jobs": len(source_rows), "n_layouts": len(per_layout),
        "n_pairs": len({key[:2] for key in groups}), "n_paired_observations": len(observations),
        "seeds": sorted({key[2] for key in groups}), "layers": layers, "site": next(iter(sites)),
        "layouts_per_family": dict(sorted(Counter(r["family"] for r in per_layout).items())),
        "baseline_mean_tv": mean(r["baseline_tv"] for r in per_layout),
        "summary": summary, "per_layout": per_layout, "raw_observations": observations,
        "identity_audit": {"n_identity_jobs": len(identity_reports), "exact_forced_q_and_zero_patch": True,
                           "rows": identity_reports},
        "source_rows": sorted(source_rows, key=lambda r: r["job_id"]),
        "definitions": {
            "tv_distance": "0.5 * sum over UP,DOWN,LEFT,RIGHT of abs(q_intervention - q_physical_counterfactual). Both q vectors are renormalized four-action forced distributions.",
            "reference": "Same layout/pair/seed physical counterfactual text baseline; its distance from itself is zero. Unchanged recipient uses that same paired reference.",
            "weighting": "Average paired distances within each layout before averaging layouts equally; never compute distance between population-mean distributions.",
            "random_control": "Gaussian direction matched per position to the donor-minus-recipient L2 norm by the compiler; normal model precision/rounding still applies.",
            "primary_or_prespecified": False, "statistical_inference": "None: no confidence intervals, bootstrap, significance or confirmation claims.",
            "claim_scope": "post-hoc discovery diagnostic, forced distributions; not native action or abstract belief proof",
        },
    }


def render(diagnostic: dict, output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    if output.suffix in {".png", ".pdf", ".json"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 7.2), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.765, bottom=0.36, wspace=0.16)
    layers = diagnostic["layers"]
    largest_tv = max(v["tv_distance"] for row in diagnostic["per_layout"] for v in row["series"])
    upper = min(1.0, max(0.25, math.ceil(largest_tv / 0.25) * 0.25))
    for ax, kind in zip(axes, SERIES):
        for layout in diagnostic["per_layout"]:
            values = {v["layer"]: v["tv_distance"] for v in layout["series"] if v["kind"] == kind}
            ax.plot(layers, [values[layer] for layer in layers], color="#8995a3", linewidth=0.8,
                    alpha=0.43, marker=".", markersize=4, zorder=1)
        values = {v["layer"]: v["mean_tv_distance"] for v in diagnostic["summary"] if v["kind"] == kind}
        ax.plot(layers, [values[layer] for layer in layers], color=COLORS[kind], linewidth=2.8,
                marker="o", markersize=5.5, zorder=3)
        ax.axhline(diagnostic["baseline_mean_tv"], color="#293b4d", linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        ax.set_title(LABELS[kind], loc="left", pad=15, fontsize=12, fontweight="bold", color="#263b4e")
        ax.set_xticks(layers)
        ax.set_xlim(min(layers) - 0.8, max(layers) + 0.8)
        ax.set_ylim(-0.02, upper + 0.025)
        ax.set_yticks([v for v in (0, 0.25, 0.5, 0.75, 1) if v <= upper])
        ax.set_xlabel("Transformer block (zero-based)", labelpad=10, fontsize=10)
        ax.grid(axis="y", color="#e3e8ee", linewidth=0.6)
        ax.tick_params(axis="both", length=0, pad=7, colors="#41566a")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd4dd")
    axes[0].set_ylabel("TV distance from physical counterfactual", labelpad=10)
    fig.text(0.04, 0.94, "Routing a changed input fact into the forced action readout",
             fontsize=17, fontweight="bold", color="#203449")
    families = diagnostic["layouts_per_family"]
    cohort_text = ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in families.items())
    fig.text(0.04, 0.882, f"{diagnostic['experiment_id']}  |  {diagnostic['n_layouts']} layouts "
             f"({cohort_text})  |  {diagnostic['site']} replacement",
             fontsize=10.5, color="#41566a")
    fig.text(0.04, 0.838, "Smaller distances indicate a closer forced distribution to the paired physical counterfactual. Zero means identical four-action scores.",
             fontsize=10.2, color="#41566a")
    fig.legend(handles=[
        Line2D([], [], color="#8995a3", linewidth=0.8, marker=".", label="Individual layout"),
        Line2D([], [], color="#177f73", linewidth=2.8, marker="o", label="Mean across layouts (panel color)"),
        Line2D([], [], color="#293b4d", linestyle="--", label=f"Unchanged recipient mean TV = {diagnostic['baseline_mean_tv']:.4f}"),
    ], loc="lower left", bbox_to_anchor=(0.033, 0.221), ncol=3, frameon=False, fontsize=10)
    first = min(layers)
    first_means = {v["kind"]: v["mean_tv_distance"] for v in diagnostic["summary"] if v["layer"] == first}
    fig.text(0.04, 0.19,
             f"Block {first} mean TV: source fact {first_means['factspan']:.4f}; random {first_means['random_factspan']:.4f}; "
             f"last header {first_means['lasttoken']:.4f}. Identity controls: exact unchanged forced scores at every plotted layer.",
             fontsize=10, color="#41566a")
    balanced = len(set(families.values())) == 1
    count_caption = f"{diagnostic['n_layouts']} layouts/{next(iter(families.values()))} per family" if balanced else f"{diagnostic['n_layouts']} layouts; family counts above"
    fig.text(0.04, 0.132,
             f"Post-hoc discovery diagnostic, forced distributions, {count_caption}; not native or abstract belief proof.",
             fontsize=10.5, fontweight="bold", color="#263b4e")
    fig.text(0.04, 0.097,
             "TV = 0.5 x sum of absolute differences across the four forced action scores. Distances are paired before averaging.\n"
             "Random directions match per-position intervention norms; no confidence intervals, significance tests or held-out confirmation claims.",
             fontsize=9.2, linespacing=1.5, color="#41566a", va="top")
    fig.text(0.04, 0.017,
             f"Complete cohort: {diagnostic['n_completed_jobs']}/{diagnostic['n_planned_jobs']} jobs  |  "
             f"Spec SHA256 {diagnostic['spec_sha256'][:16]}...  |  Raw paired distributions, counts and hashes: companion .counts.json",
             fontsize=8.3, color="#687d91")
    png, pdf, raw = (Path(str(output) + suffix) for suffix in (".png", ".pdf", ".counts.json"))
    diagnostic["figure_script"] = {"path": str(Path(__file__).resolve()),
                                  "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    diagnostic["artifacts"] = {"png": str(png.resolve()), "pdf": str(pdf.resolve()), "counts": str(raw.resolve())}
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Input-fact routing: post-hoc forced-distribution diagnostic"})
    plt.close(fig)
    raw.write_text(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return png, pdf, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Filename stem for PNG, PDF and counts JSON")
    args = parser.parse_args()
    try:
        diagnostic = load_diagnostic(args.campaign, args.experiment_id)
        artifacts = render(diagnostic, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Cannot render routing figure: {exc}\n")
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
