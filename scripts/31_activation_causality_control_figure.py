#!/usr/bin/env python3
"""Descriptive, paired historical sentence controls; never load a model.

Example:
  interp/.venv/bin/python scripts/31_activation_causality_control_figure.py \
      --campaign outputs/activation_causality/campaign_20260912 \
      --experiment-id sentence_native_pilot_v1 \
      --output outputs/activation_causality/campaign_20260912/figures/sentence_native_pilot_v1

The output argument is a filename stem (or a .png/.pdf filename). Writes PNG,
PDF and .counts.json, including hashes of the frozen spec and each source row.
No confidence intervals or population effects are estimated. Partial mode
retains only cuts/seeds completed under EVERY condition, preserving pairing.
Truncated and malformed native outputs remain in all rate denominators.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ORDER = (
    "baseline", "identity", "donor_text", "donor_identity_layer10",
    "layer10", "layer15", "all_layers_full", "all_layers_last12",
    "reverse_layer10",
)
LABELS = {
    "baseline": "Neutral sentence: text baseline",
    "identity": "Neutral identity: layer 10",
    "donor_text": "Target sentence: text baseline",
    "donor_identity_layer10": "Target identity: layer 10",
    "layer10": "Target to neutral: layer 10",
    "layer15": "Target to neutral: layer 15",
    "all_layers_full": "Target to neutral: all layers, sentence",
    "all_layers_last12": "Target to neutral: all layers, last 12 tokens",
    "reverse_layer10": "Neutral to target: layer 10",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_key(job: dict) -> tuple[str, str, int]:
    return str(job["layout_id"]), str(job["pair_id"]), int(job["seed"])


def paired_control(condition: str, spec: dict) -> str:
    controls = {c["control"] for c in spec.get("comparisons", [])
                if c.get("treatment") == condition}
    if len(controls) > 1:
        raise ValueError(f"Ambiguous paired control for {condition}")
    if controls:
        return controls.pop()
    if condition in {"donor_text", "donor_identity_layer10", "reverse_layer10"}:
        return "donor_text"
    return "baseline"


def validate_row(row: dict, job: dict, spec_hash: str) -> None:
    if row.get("spec_sha256") != spec_hash:
        raise ValueError(f"Spec hash mismatch: {job['job_id']}")
    for key in ("experiment_id", "job_id", "condition", "pair_id", "layout_id",
                "seed", "target_action", "max_new_tokens", "partition", "natural"):
        if row.get(key) != job.get(key):
            raise ValueError(f"Row/spec mismatch for {key}: {job['job_id']}")
    if row.get("status") != "complete":
        return  # The cohort loader handles execution failures as missing rows.
    q = row.get("forced_q", {})
    if set(q) != set(ACTIONS) or any(
        not isinstance(q[a], (int, float)) or not math.isfinite(q[a])
        or not 0 <= q[a] <= 1 for a in ACTIONS
    ) or abs(sum(q.values()) - 1) > 1e-5:
        raise ValueError(f"Invalid four-action forced scores: {job['job_id']}")
    top = row.get("forced_top_action")
    if top not in ACTIONS or abs(q[top] - max(q.values())) > 1e-8:
        raise ValueError(f"Invalid forced top action: {job['job_id']}")
    status, action = row.get("generation_status"), row.get("native_action")
    if status not in {"complete", "truncated", "malformed"}:
        raise ValueError(f"Native endpoint unavailable: {job['job_id']} ({status})")
    if (status == "complete" and action not in ACTIONS) or (
        status != "complete" and action is not None
    ):
        raise ValueError(f"Inconsistent native action/status: {job['job_id']}")


def load_counts(campaign: Path, experiment_id: str, allow_partial: bool = False) -> dict:
    """Validate frozen provenance and return JSON-safe paired descriptive counts."""
    campaign = Path(campaign).resolve()
    if Path(experiment_id).name != experiment_id or experiment_id in {"", ".", ".."}:
        raise ValueError("experiment_id must be a single path component")
    retired = campaign / "reviews" / f"{experiment_id}_retired.json"
    if retired.exists():
        raise ValueError(f"Refusing retired experiment: {retired}")
    spec_path = campaign / "specs" / f"{experiment_id}.json"
    spec = json.loads(spec_path.read_text())
    spec_hash = sha256(spec_path)
    if spec.get("experiment_id") != experiment_id or spec.get("partition") != "historical":
        raise ValueError("This figure requires a historical experiment with a matching spec ID")
    jobs = spec.get("jobs", [])
    if not jobs or any(j.get("family") != "sentence_control" or not j.get("natural")
                       or j.get("semantic_test") is not False for j in jobs):
        raise ValueError("Require historical sentence-control jobs with native generation")
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs in spec")
    conditions = sorted({j["condition"] for j in jobs},
                        key=lambda c: (ORDER.index(c) if c in ORDER else len(ORDER), c))
    if not {"baseline", "donor_text"}.issubset(conditions):
        raise ValueError("Both neutral and target-text paired baselines are required")
    groups: dict[tuple, dict] = defaultdict(dict)
    rows, sources, missing, failed = {}, [], [], []
    result_dir = campaign / "results" / experiment_id
    planned_ids = {j["job_id"] for j in jobs}
    extra_paths = [p for p in result_dir.glob("*.json") if p.stem not in planned_ids]
    if extra_paths:
        raise ValueError(f"Unexpected result files outside frozen spec: {extra_paths}")
    for job in jobs:
        key, condition = pair_key(job), job["condition"]
        if condition in groups[key]:
            raise ValueError(f"Duplicate paired condition: {key}, {condition}")
        groups[key][condition] = job
        if job["target_action"] not in ACTIONS:
            raise ValueError(f"Invalid target: {job['job_id']}")
        path = result_dir / f"{job['job_id']}.json"
        if not path.exists():
            missing.append(job["job_id"])
            continue
        row = json.loads(path.read_text())
        validate_row(row, job, spec_hash)
        sources.append({"job_id": job["job_id"], "path": str(path), "sha256": sha256(path),
                        "status": row.get("status")})
        if row.get("status") != "complete":
            failed.append(job["job_id"])
            continue
        rows[job["job_id"]] = row
    incomplete_groups = [key for key, group in groups.items()
                         if set(group) != set(conditions)
                         or any(j["job_id"] not in rows for j in group.values())]
    if (missing or failed or incomplete_groups) and not allow_partial:
        raise ValueError(f"Incomplete job cohort: {len(missing)} missing, {len(failed)} failed, "
                         f"{len(incomplete_groups)} incomplete paired cuts/seeds; use --allow-partial")
    retained = sorted(set(groups) - set(incomplete_groups))
    if not retained:
        raise ValueError("No complete paired cuts/seeds across all conditions")
    for key in retained:
        if len({j["target_action"] for j in groups[key].values()}) != 1:
            raise ValueError(f"Target action differs across paired conditions: {key}")
        for field in ("max_new_tokens", "temperature"):
            if len({j.get(field) for j in groups[key].values()}) != 1:
                raise ValueError(f"Paired generation configuration differs ({field}): {key}")
    retained_jobs = [groups[key][condition] for key in retained for condition in conditions]
    used_rows = [rows[j["job_id"]] for j in retained_jobs]
    identities = {
        (r.get("engine_key"), r.get("execution_key"),
         json.dumps(r.get("inference_configuration"), sort_keys=True)) for r in used_rows
    }
    if len(identities) != 1 or any(not part or part == "null" for part in next(iter(identities))):
        raise ValueError("Missing or mixed engine/execution/inference identities")
    records = []
    for condition in conditions:
        control = paired_control(condition, spec)
        if control not in conditions:
            raise ValueError(f"Missing paired control {control} for {condition}")
        rr = [rows[groups[key][condition]["job_id"]] for key in retained]
        cc = [rows[groups[key][control]["job_id"]] for key in retained]
        n = len(rr)
        forced = sum(r["forced_top_action"] == r["target_action"] for r in rr)
        native = sum(r["generation_status"] == "complete" and
                     r["native_action"] == r["target_action"] for r in rr)
        statuses = Counter(r["generation_status"] for r in rr)
        unresolved = statuses["truncated"] + statuses["malformed"]
        records.append({
            "condition": condition, "label": LABELS.get(condition, condition.replace("_", " ")),
            "paired_control": control, "n": n,
            "n_layouts": len({r["layout_id"] for r in rr}),
            "n_cuts": len({(r["layout_id"], r["pair_id"]) for r in rr}),
            "forced_target_top_count": forced, "forced_target_top_rate": forced / n,
            "native_target_count": native, "native_target_rate": native / n,
            "native_other_count": statuses["complete"] - native,
            "native_unresolved_count": unresolved, "native_unresolved_rate": unresolved / n,
            "native_status_counts": {k: statuses[k] for k in ("complete", "truncated", "malformed")},
            "paired_forced_top_equal_count": sum(r["forced_top_action"] == c["forced_top_action"]
                                                  for r, c in zip(rr, cc)),
            "paired_forced_max_absolute_q_difference": max(
                abs(r["forced_q"][a] - c["forced_q"][a]) for r, c in zip(rr, cc) for a in ACTIONS),
            "paired_native_action_status_equal_count": sum(
                (r["generation_status"], r["native_action"]) ==
                (c["generation_status"], c["native_action"]) for r, c in zip(rr, cc)),
            "interventions": list({json.dumps(j["intervention"], sort_keys=True): j["intervention"]
                                   for j in retained_jobs if j["condition"] == condition}.values()),
            "selections": list({json.dumps(j["selection"], sort_keys=True): j["selection"]
                               for j in retained_jobs if j["condition"] == condition}.values()),
            "job_ids": [r["job_id"] for r in rr],
        })
    engine_key, execution_key, config_json = next(iter(identities))
    return {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign), "experiment_id": experiment_id, "partition": "historical",
        "spec_path": str(spec_path), "spec_sha256": spec_hash, "source_data": spec.get("sources", {}),
        "engine_key": engine_key, "execution_key": execution_key,
        "inference_configuration": json.loads(config_json),
        "allow_partial": allow_partial, "is_partial": bool(incomplete_groups),
        "n_planned_jobs": len(jobs), "n_source_rows": len(sources), "n_retained_jobs": len(used_rows),
        "n_paired_observations_per_condition": len(retained),
        "n_layouts": len({key[0] for key in retained}),
        "n_cuts": len({key[:2] for key in retained}),
        "seeds": sorted({key[2] for key in retained}),
        "max_new_tokens": sorted({j["max_new_tokens"] for j in retained_jobs}),
        "temperatures": sorted({j.get("temperature", 0.7) for j in retained_jobs}),
        "paired_cohort": [{"layout_id": k[0], "pair_id": k[1], "seed": k[2]} for k in retained],
        "missing_job_ids": sorted(missing), "failed_job_ids": sorted(failed),
        "excluded_paired_cohort": [list(k) for k in sorted(incomplete_groups)],
        "excluded_job_ids": sorted(j["job_id"] for k in incomplete_groups for j in groups[k].values()),
        "conditions": records, "source_rows": sorted(sources, key=lambda r: r["job_id"]),
        "definitions": {
            "forced_target_top_rate": "Fraction of paired cut/seed rows whose four-action forced readout argmax equals target; not native action or mean target q.",
            "native_target_rate": "Fraction of ALL paired cut/seed rows ending with valid native final-channel target JSON; no forced substitution.",
            "native_unresolved_rate": "Fraction truncated at cap or malformed; remains in native target denominator.",
            "paired_controls": "Neutral text baseline for forward transfer/neutral identity; target text baseline for reverse transfer/target identity.",
            "weighting": "Equal weight per retained cut/seed; layouts and cuts are reported separately. No bootstrap, confidence intervals or population-effect claims.",
            "scope": "Historical selected cuts; donor sentence explicitly names target action. These are action-injection and implementation controls, not semantic belief proof. Target identity is not an independent rescue.",
        },
    }


def render(counts: dict, output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    if output.suffix in {".png", ".pdf", ".json"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = counts["conditions"]
    by_condition = {r["condition"]: r for r in records}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 8.0), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.subplots_adjust(left=0.31, right=0.974, bottom=0.27, top=0.79, wspace=0.22)
    panels = [
        ("Forced target top", "forced_target_top_rate", "forced_target_top_count", "#346899"),
        ("Native target action", "native_target_rate", "native_target_count", "#138071"),
        ("Native unresolved", "native_unresolved_rate", "native_unresolved_count", "#abb2bb"),
    ]
    for ax, (title, rate, count, color) in zip(axes, panels):
        for i, row in enumerate(records):
            if row["condition"] in {"baseline", "donor_text"}:
                ax.axhspan(i - 0.47, i + 0.47, color="#eef2f6", zorder=0)
            ax.barh(i, row[rate], height=0.56, color=color,
                    hatch="///" if rate == "native_unresolved_rate" else None,
                    edgecolor="#66717c" if rate == "native_unresolved_rate" else color,
                    linewidth=0.4, zorder=2)
            control = by_condition[row["paired_control"]]
            ax.plot(control[rate], i, "o", markersize=6, markerfacecolor="white",
                    markeredgecolor="#263749", markeredgewidth=1.3, zorder=3, clip_on=False)
            ax.text(1.04, i, f"{row[count]}/{row['n']}", va="center", ha="left", fontsize=10,
                    color="#263749", clip_on=False)
        ax.set_xlim(0, 1.16)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=15)
        ax.grid(axis="x", color="#dee3e8", linewidth=0.6, zorder=1)
        ax.tick_params(axis="both", length=0, pad=7, colors="#344657")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(records)))
    axes[0].set_yticklabels([r["label"] for r in records], fontsize=10)
    axes[0].set_ylim(len(records) - 0.4, -0.65)
    mode = "PARTIAL PAIRED COHORT - " if counts["is_partial"] else ""
    fig.text(0.035, 0.953, "Sentence-state control: forced readout and native continuation",
             fontsize=17, fontweight="bold", color="#203449")
    caps = ", ".join(f"{c:,}" for c in counts["max_new_tokens"])
    seed_text = ", ".join(map(str, counts["seeds"]))
    fig.text(0.035, 0.909,
             mode + f"{counts['experiment_id']}  |  {counts['n_layouts']} layouts, {counts['n_cuts']} historical cuts"
             f"  |  seed(s) {seed_text}  |  native cap {caps} new tokens",
             fontsize=10.5, color="#41556a")
    fig.text(0.035, 0.873,
             f"Same {counts['n_paired_observations_per_condition']} cut/seed observations in every row; "
             "counts shown at right. Descriptive pilot; no confidence intervals.",
             fontsize=10.5, color="#41556a")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#263749", label="Paired text baseline"),
        Patch(facecolor="#eef2f6", label="Text baseline rows"),
        Patch(facecolor="#abb2bb", edgecolor="#66717c", hatch="///", label="Truncated or malformed native output"),
    ], loc="lower left", bbox_to_anchor=(0.031, 0.183), ncol=3, frameon=False, fontsize=10)
    footer = (
        "Forced: highest four-action score after an imposed final-answer prefix. Native: actual final-channel JSON from continuing the patched cache.\n"
        "Native unresolved outputs stay in the denominator; no forced fallback. Forward and neutral-identity rows pair to the neutral text baseline;\n"
        "reverse and target-identity rows pair to the target text baseline. Identity rows test self-transplant; target identity is not an independent rescue.\n"
        "Scope: the donor sentence explicitly names the target action. These historical action-injection controls do not establish a semantic belief mechanism."
    )
    fig.text(0.035, 0.157, footer, va="top", fontsize=9.2, linespacing=1.55, color="#41556a")
    fig.text(0.035, 0.023,
             f"Frozen spec SHA256 {counts['spec_sha256'][:16]}...  |  "
             f"{counts['n_retained_jobs']}/{counts['n_planned_jobs']} jobs retained  |  "
             "Full counts, definitions and source hashes: companion .counts.json",
             fontsize=8.3, color="#67798a")
    png, pdf, raw = Path(str(output) + ".png"), Path(str(output) + ".pdf"), Path(str(output) + ".counts.json")
    counts["figure_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}
    counts["artifacts"] = {"png": str(png.resolve()), "pdf": str(pdf.resolve()), "counts": str(raw.resolve())}
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Historical sentence activation controls",
                                                 "Subject": counts["definitions"]["scope"]})
    plt.close(fig)
    raw.write_text(json.dumps(counts, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return png, pdf, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename stem for PNG, PDF and counts JSON")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicitly retain only fully paired cuts/seeds from an incomplete cohort")
    args = parser.parse_args()
    try:
        counts = load_counts(args.campaign, args.experiment_id, args.allow_partial)
        artifacts = render(counts, args.output)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Cannot render control figure: {exc}\n")
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
