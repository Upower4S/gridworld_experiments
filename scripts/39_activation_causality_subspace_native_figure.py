#!/usr/bin/env python3
"""Plot only all six complete, independently audited native subspace cohorts.

No model/package imports or partial-outcome reporting. --check-ready reads only
the pinned plan/specifications and file names, never audit or result contents.
Normal execution requires six exact 56-job inventories AND six complete audits
before synthesizing any outcome. New output stems are required for new snapshots.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


PLAN = "reviews/subspace_complete_native_plan_v1.json"
AUDITOR = "reviews/subspace_complete_independent_audit_v1.py"
PLAN_SHA = "4c77834e1dc6cecd7bf10f446b5b4f5ffe8e0123463185773001b6da400a86e0"
AUDITOR_SHA = "96699e034a7eb6e6c1b18ce22d190321c2cbb5d389edebbc2e251a4c053ed6d9"
SPEC_SHAS = {
    "subspace_wall_l0_r1_native_v1": "246ccd1001c3413fce23b31537fbf5f5ecd3dc475d8cb34966ce9f0f84c09243",
    "subspace_wall_l0_r4_native_v1": "89476a7cdfc566b43c423f459244a85a6eacd4fb350eea956b77db3410512dee",
    "subspace_key_l0_r1_native_v1": "0b51408d5517717aa9d60216abc26dce9afb8a9aadb780d91ee43fe92f152a06",
    "subspace_key_l0_r4_native_v1": "5ec2158a3a9d045b0b8255b884004b809e5bb0788b78a86ef15848b56d615415",
    "subspace_door_l0_r1_native_v1": "e76566ec601d136c48b24fd108f2a6848d6bf9cf8f32bf355e482aeb3a57649a",
    "subspace_door_l0_r4_native_v1": "3d839097f1b27a4ff220f3ec685c475edb1ad3bd6fa74dc2cc312de388a6d267",
}
FAMILIES = ("wall", "key_possession", "door_status")
FAMILY_LABELS = {"wall": "Wall", "key_possession": "Key possession", "door_status": "Door status"}
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
CONDITIONS = (
    ("baseline", "Recipient text"),
    ("donor_text", "Counterfactual donor text"),
    ("full_transfer", "Full fact-span transfer"),
    ("subspace_identity", "Subspace identity"),
    ("subspace_removal", "Removal to zero (donor input)"),
    ("restoration_equivalent", "Restoration equivalent (identity)"),
    ("subspace_forward_0p5", "0.5 · Learned subspace"),
    ("subspace_reverse_0p5", "0.5 · Reverse (donor → original)"),
    ("random_subspace_0p5", "0.5 · Random: rank only"),
    ("projected_norm_matched_random_subspace_0p5", "0.5 · Random: norm matched"),
    ("subspace_forward_1", "1.0 · Learned subspace"),
    ("subspace_reverse_1", "1.0 · Reverse (donor → original)"),
    ("random_subspace_1", "1.0 · Random: rank only"),
    ("projected_norm_matched_random_subspace_1", "1.0 · Random: norm matched"),
)
LIMITATIONS = [
    "Twelve distinct held-out discovery evaluation layouts: four per family, reused across ranks; not 24 independent layouts.",
    "One registered generation seed per layout/condition; all 14 conditions retained, including unresolved native outputs.",
    "Target always denotes the physical recipient's counterfactual-optimal action. Reverse starts from donor text; lower target counts can reflect reversal toward the original action.",
    "Rank-only random subspaces do not match perturbation magnitude. Distinct norm-matched random writes and learned writes share the original task-projected reference, each audited at fixed 1% relative tolerance (exact zero when the reference is zero).",
    "Rank 1/4 fits use the same four training layouts per family; these fact-token contrasts may encode lexical substitutions. Rank 4 is not four independently established semantic dimensions.",
    "Removal to zero is a descriptive perturbation; restoration-equivalent and subspace identity are engineering checks, not independent necessity/rescue evidence.",
    "Descriptive discovery counts only: no population confidence, significance, rank superiority, abstract-belief or mediation claim.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite)


def stable_bytes(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    require(all(getattr(before, key) == getattr(after, key) for key in fields), "File changed during read: " + str(path))
    return data, hashlib.sha256(data).hexdigest()


def read_hashed(path):
    raw, digest = stable_bytes(path)
    return strict_json(raw), digest


def load_contract(campaign):
    """Read pinned design inputs only; native result/audit contents stay closed."""
    campaign = Path(campaign).resolve()
    plan, digest = read_hashed(campaign / PLAN)
    require(digest == PLAN_SHA, "Frozen native plan hash mismatch")
    require(stable_bytes(campaign / AUDITOR)[1] == AUDITOR_SHA, "Independent auditor source hash mismatch")
    require(plan["conditions"] == [name for name, _ in CONDITIONS] and plan["doses"] == [0.5, 1.0],
            "Unexpected condition/dose inventory")
    require(len(plan["fits"]) == 6 and {f["native_experiment_id"] for f in plan["fits"]} == set(SPEC_SHAS),
            "Require all six planned native cohorts")
    contracts = []
    for fit in plan["fits"]:
        eid = fit["native_experiment_id"]
        path = campaign / "specs" / (eid + ".json")
        spec, digest = read_hashed(path)
        require(digest == SPEC_SHAS[eid], "Frozen native specification hash mismatch: " + eid)
        jobs = spec["jobs"]
        expected_layouts = {row["layout_id"] for row in plan["cohorts"][fit["family"]]["evaluation"]}
        require(len(jobs) == len({j["job_id"] for j in jobs}) == 56, "Require 56 distinct planned jobs")
        require(len(expected_layouts) == 4 and {j["layout_id"] for j in jobs} == expected_layouts,
                "Evaluation layouts differ from the frozen family cohort")
        require(Counter(j["condition"] for j in jobs) == Counter({name: 4 for name, _ in CONDITIONS}),
                "Require all fourteen conditions for four layouts")
        for job in jobs:
            require(re.fullmatch(r"[0-9a-f]{24}", job["job_id"]) is not None
                    and fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24] == job["job_id"],
                    "Noncanonical frozen job identity")
            require(job["natural"] is True and job["family"] == fit["family"]
                    and job["experiment_id"] == eid and job["seed"] == 42 and job["partition"] == "discovery"
                    and job["max_new_tokens"] == 4096 and job["temperature"] == 0.7,
                    "Unexpected frozen native protocol")
        audit_name = eid.replace("_native_v1", "_native_independent_audit_v1.json")
        contracts.append({"fit": fit, "spec": spec, "spec_path": path, "spec_sha256": digest,
                          "audit_path": campaign / "reviews" / audit_name,
                          "result_dir": campaign / "results" / eid})
    require({(c["fit"]["family"], c["fit"]["rank"]) for c in contracts} ==
            {(family, rank) for family in FAMILIES for rank in (1, 4)}, "Unexpected family/rank grid")
    require(len({j["layout_id"] for c in contracts for j in c["spec"]["jobs"]}) == 12,
            "Require twelve independent evaluation layouts with rank reuse")
    return campaign, plan, contracts


def readiness(campaign, contracts):
    """Operational counts only. Never open an audit or result."""
    cohorts = []
    for item in contracts:
        eid = item["fit"]["native_experiment_id"]
        planned = {j["job_id"] + ".json" for j in item["spec"]["jobs"]}
        present = {p.name for p in item["result_dir"].glob("*.json") if p.is_file()}
        retired = any((campaign / "reviews" / (identifier + "_retired.json")).exists()
                      for identifier in (eid, item["fit"]["forced_experiment_id"]))
        cohorts.append({"experiment_id": eid, "planned": len(planned), "saved": len(present),
                        "missing": len(planned - present), "unexpected": len(present - planned),
                        "exact_filename_inventory": planned == present,
                        "independent_audit_present": item["audit_path"].is_file(), "retired": retired})
    return {"ready": all(c["exact_filename_inventory"] and c["independent_audit_present"] and not c["retired"]
                         for c in cohorts), "outcome_contents_read": False, "cohorts": cohorts}


def validate_native(row, job, spec_sha):
    require(row.get("spec_sha256") == spec_sha, "Raw result/specification hash mismatch")
    for key, value in job.items():
        if key not in {"recipient_text", "donor_text", "shared_base_text"}:
            require(key in row and row[key] == value, "Raw result changes frozen field: " + key)
    require(row.get("status") == "complete" and row.get("inference_configuration") == {"head_mode": "full"},
            "Technical failure or non-full-head result")
    status, action = row.get("generation_status"), row.get("native_action")
    require(status in {"complete", "truncated", "malformed"}, "Missing native endpoint")
    tokens = row.get("generated_token_ids")
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens)
            and 0 < len(tokens) == row.get("generated_tokens") == row.get("n_new_tokens") <= 4096,
            "Missing authoritative native tokens")
    if status == "complete":
        final = strict_json(row["final_text"])
        require(action in ACTIONS and row.get("parsed_action") == action and row.get("parse_status") == "valid"
                and row.get("stop_reason") in {"return", "eos"}
                and isinstance(final, dict) and str(final.get("action", "")).strip().upper() == action,
                "Invalid native final/action")
    else:
        require(action is None, "Unresolved native output has an assigned action")
        if status == "truncated":
            require(len(tokens) == 4096 and row.get("stop_reason") == "token_limit", "Truncation/cap mismatch")
    require(job["target_action"] in ACTIONS and job["original_action"] in ACTIONS
            and job["target_action"] != job["original_action"], "Invalid paired action references")


def count_rows(rows):
    counts = {"n": len(rows), "target": sum(r["native_action"] == r["target_action"] for r in rows),
              "other": sum(r["native_action"] in ACTIONS and r["native_action"] != r["target_action"] for r in rows),
              "unresolved": sum(r["native_action"] is None for r in rows),
              "original_action": sum(r["native_action"] == r["original_action"] for r in rows),
              "third_action": sum(r["native_action"] in ACTIONS and r["native_action"] not in
                                  {r["target_action"], r["original_action"]} for r in rows),
              "native_status_counts": dict(sorted(Counter(r["generation_status"] for r in rows).items())),
              "job_ids": sorted(r["job_id"] for r in rows), "layout_ids": sorted(r["layout_id"] for r in rows)}
    require(counts["target"] + counts["other"] + counts["unresolved"] == counts["n"] == 4,
            "Condition denominator/outcome categories differ")
    require(counts["original_action"] + counts["third_action"] == counts["other"],
            "Resolved non-target categories do not separate original and third actions")
    return counts


def load_counts(campaign, *, repo_root=None):
    """Return counts only after the entire six-cohort audit/provenance gate passes."""
    campaign, plan, contracts = load_contract(campaign)
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    require(readiness(campaign, contracts)["ready"],
            "All six exact 56-job native inventories and independent audits are required; no outcomes read")
    inputs, audits = {}, []
    # First validate ALL six audit certificates. Do not count a completed subset.
    for item in contracts:
        audit, audit_sha = read_hashed(item["audit_path"])
        cohort, fit = audit["cohort"], item["fit"]
        require(audit.get("format") == "subspace_complete_independent_cohort_audit_v1"
                and audit.get("audit_source_sha256") == AUDITOR_SHA, "Unrecognized independent audit")
        require(cohort.get("state") == "complete" and cohort.get("outcomes_read") is True
                and cohort.get("registered") is True and cohort.get("retirement") is None
                and cohort.get("exact_complete_filename_inventory") is True
                and cohort.get("planned_jobs") == cohort.get("saved_file_count") == 56
                and cohort.get("technical_status_counts") == {"complete": 56}
                and cohort.get("remaining_planned_jobs") == 0, "Incomplete/failed independent native audit")
        require(cohort.get("experiment_id") == fit["native_experiment_id"] and cohort.get("family") == fit["family"]
                and cohort.get("rank") == fit["rank"] and cohort.get("basis_id") == fit["basis_id"]
                and cohort.get("spec_sha256") == item["spec_sha256"], "Audit belongs to another source/cohort")
        require(cohort.get("exact_forced_native_replays") == 56 and cohort.get("required_norm_writes_revalidated") == 16,
                "Missing complete forced-predecessor or required norm checks")
        require(len(cohort.get("identity_checks", [])) == 8 and all(
            c.get("exact_forced_identity") is True and c.get("exact_entire_generated_identity") is True
            for c in cohort["identity_checks"]), "Missing exact identity checks")
        require(len(cohort.get("paired_applied_norm_checks", [])) == 8 and all(
            c.get("expected_original_reference_identical") is True for c in cohort["paired_applied_norm_checks"]),
            "Missing paired projected-norm reference checks")
        mapping = audit.get("input_file_sha256", {})
        require(mapping and mapping.get(str(item["spec_path"].relative_to(repo))) == item["spec_sha256"],
                "Audit does not hash its specification")
        expected_raw = {str((item["result_dir"] / (j["job_id"] + ".json")).relative_to(repo)) for j in item["spec"]["jobs"]}
        actual_raw = {name for name in mapping if (repo / name).parent.resolve() == item["result_dir"]}
        require(actual_raw == expected_raw, "Audit native raw inventory differs from the exact cohort")
        audits.append((item, audit, audit_sha))
        for name, digest in mapping.items():
            require(name not in inputs or inputs[name] == digest, "Conflicting audit source hashes")
            inputs[name] = digest
    # All certificates passed. Verify every input byte they bind, including rows.
    for name, digest in inputs.items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and path.is_file(), "Audit source escapes repository or is missing")
        if path.parent == campaign / "specs":
            require(not (campaign / "reviews" / (path.stem + "_retired.json")).exists(), "Retired audited source")
        require(stable_bytes(path)[1] == digest, "Audit input SHA mismatch: " + name)
    cohorts, execution_keys = [], set()
    for item, audit, audit_sha in audits:
        grouped = {name: [] for name, _ in CONDITIONS}
        paired = {}
        for job in item["spec"]["jobs"]:
            path = item["result_dir"] / (job["job_id"] + ".json")
            row, digest = read_hashed(path)
            require(digest == inputs[str(path.relative_to(repo))], "Result changed after audit-source validation")
            validate_native(row, job, item["spec_sha256"])
            require(row.get("execution_key") and row.get("engine_key") and row.get("model_snapshot_identity_sha256"),
                    "Missing execution/model identity")
            execution_keys.add((row["execution_key"], row["engine_key"], row["model_snapshot_identity_sha256"]))
            grouped[job["condition"]].append(row)
            key = (job["layout_id"], job["pair_id"], job["seed"])
            require(job["condition"] not in paired.setdefault(key, {}), "Duplicate layout/pair/seed condition")
            paired[key][job["condition"]] = row
        require(len(paired) == 4, "Require four paired layouts")
        for group in paired.values():
            require(set(group) == set(grouped) and len({r["target_action"] for r in group.values()}) == 1
                    and len({r["original_action"] for r in group.values()}) == 1, "Incomplete or inconsistent paired controls")
        counts = {name: count_rows(grouped[name]) for name, _ in CONDITIONS}
        require(set(audit["cohort"]["conditions"]) == set(counts), "Audit condition inventory differs")
        for name, actual in counts.items():
            expected = audit["cohort"]["conditions"][name]
            require((actual["n"], actual["target"], actual["original_action"], actual["native_status_counts"]) ==
                    (expected["n_planned_layouts"], expected["target_count"], expected["original_action_count"],
                     expected["generation_status_counts"]), "Independent audit/count disagreement: " + name)
            references = {row["metadata"]["paired_baseline_condition"] for row in grouped[name]}
            require(len(references) == 1, "Condition has inconsistent frozen interpretation references")
            actual["interpretation_reference"] = next(iter(references))
            actual["fixed_target_definition"] = "recipient counterfactual-optimal action"
            actual["reverse_aims_toward_original_action"] = name.startswith("subspace_reverse_")
        require(dict(Counter(r["generation_status"] for rows in grouped.values() for r in rows)) ==
                audit["cohort"]["generation_status_counts"], "Independent audit/status denominator disagreement")
        cohorts.append({"experiment_id": item["fit"]["native_experiment_id"], "family": item["fit"]["family"],
                        "rank": item["fit"]["rank"], "basis_id": item["fit"]["basis_id"],
                        "n_evaluation_layouts": 4, "layout_ids": sorted({key[0] for key in paired}),
                        "conditions": counts, "spec_path": str(item["spec_path"]),
                        "spec_sha256": item["spec_sha256"], "independent_audit_path": str(item["audit_path"]),
                        "independent_audit_sha256": audit_sha})
    require(len(execution_keys) == 1, "Mixed execution/model identities across the complete six-cohort campaign")
    require(readiness(campaign, contracts)["ready"], "Cohort inventory/retirement changed during validation")
    for item, _, digest in audits:
        require(stable_bytes(item["audit_path"])[1] == digest, "Independent audit changed during validation")
    return {"format": "complete_subspace_native_figure_counts_v1", "campaign": str(campaign),
            "scope": "complete descriptive discovery native counts", "all_six_cohorts_validated": True,
            "n_jobs": 336, "n_independent_evaluation_layouts": 12, "n_conditions_per_cohort": 14,
            "rank_reuses_family_layouts": True, "layer": 0, "site": "resid_post", "doses": plan["doses"],
            "generation_seed": 42, "temperature": 0.7, "max_new_tokens": 4096,
            "condition_order": [name for name, _ in CONDITIONS], "cohorts": cohorts,
            "plan_path": str(campaign / PLAN), "plan_sha256": PLAN_SHA,
            "independent_auditor_path": str(campaign / AUDITOR), "independent_auditor_sha256": AUDITOR_SHA,
            "verified_input_file_sha256": dict(sorted(inputs.items())), "execution_identity": list(next(iter(execution_keys))),
            "limitations": LIMITATIONS}


def render(counts, output):
    """Render all fourteen conditions; no comparisons or pooled outcome statistics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(19, 14), sharex=True)
    colors = {"target": "#197B78", "other": "#BFC7CE", "unresolved": "#E8AC48"}
    panels = {(c["family"], c["rank"]): c for c in counts["cohorts"]}
    for row_index, rank in enumerate((1, 4)):
        for col_index, family in enumerate(FAMILIES):
            ax = axes[row_index, col_index]
            cohort = panels[family, rank]
            for y, (condition, _) in enumerate(CONDITIONS):
                left = 0
                for kind in ("target", "other", "unresolved"):
                    value = cohort["conditions"][condition][kind]
                    ax.barh(y, value, left=left, color=colors[kind], height=0.69, edgecolor="white", linewidth=0.6)
                    if value:
                        ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=10,
                                color="white" if kind == "target" else "#17242D", fontweight="bold")
                    left += value
            ax.set_yticks(range(len(CONDITIONS)), [label for _, label in CONDITIONS], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 4)
            ax.set_xticks(range(5))
            ax.set_xlabel("Native outcomes / 4 layouts", fontsize=10)
            ax.set_title(f"{FAMILY_LABELS[family]} · rank {rank}", fontsize=14, fontweight="bold", pad=10)
            for divider in (5.5, 9.5):
                ax.axhline(divider, color="#60717D", linewidth=0.65, linestyle=":")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", alpha=0.12)
            ax.set_axisbelow(True)
    fig.suptitle("Native subspace interventions at input-fact tokens · layer 0", fontsize=19, fontweight="bold", y=0.981)
    fig.text(0.5, 0.951, "Six complete 56-job cohorts · 12 distinct discovery layouts (4 per family, reused across ranks) · seed 42 · native cap 4,096 tokens",
             ha="center", fontsize=11)
    fig.legend([Patch(facecolor=colors[key]) for key in ("target", "other", "unresolved")],
               ["Counterfactual target", "Other resolved action", "Unresolved (truncated / malformed)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False, fontsize=11)
    footnotes = (
        "Reverse and removal start from donor text. Lower target counts can reflect reversal; original-action and third-action counts are separate in JSON.",
        "Rank-only random ≠ norm matched. Separate norm-matched random and learned writes share the task-projected reference; each passes the fixed 1% applied-norm audit.",
        "Ranks reuse training and evaluation layouts. Fact-token contrasts can encode lexical substitutions; rank 4 is not four established semantic dimensions.",
        "Identity/restoration are engineering checks. Counts and removal-to-zero do not establish population effects, abstract belief, necessity, independent rescue or mediation.",
    )
    for index, line in enumerate(footnotes):
        fig.text(0.025, 0.088 - index * 0.019, line, fontsize=9, color="#34454F", va="top")
    fig.subplots_adjust(left=0.145, right=0.985, top=0.883, bottom=0.145, wspace=1.12, hspace=0.28)
    with Path(str(output) + ".png").open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180, facecolor="white")
    with Path(str(output) + ".pdf").open("xb") as stream:
        fig.savefig(stream, format="pdf", facecolor="white")
    plt.close(fig)


def validate_output_location(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.is_relative_to(campaign):
        parts = output.relative_to(campaign).parts
        require(len(parts) > 1 and parts[0] in {"figures", "reports"},
                "Campaign outputs must be under figures/ or reports/, never an input/provenance tree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path, help="New output stem for .png, .pdf and .counts.json")
    parser.add_argument("--check-ready", action="store_true", help="Only frozen specs and filenames; no audit/result contents")
    args = parser.parse_args()
    if args.check_ready:
        campaign, _, contracts = load_contract(args.campaign)
        print(json.dumps(readiness(campaign, contracts), indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --check-ready is used")
    output = args.output.resolve()
    validate_output_location(args.campaign, output)
    paths = [Path(str(output) + suffix) for suffix in (".png", ".pdf", ".counts.json")]
    require(not any(path.exists() for path in paths), "Output exists; choose a new immutable snapshot stem")
    counts = load_counts(args.campaign, repo_root=args.repo_root)
    counts["created_at"] = datetime.now(timezone.utc).isoformat()
    counts["figure_script"] = {"path": str(Path(__file__).resolve()), "sha256": stable_bytes(__file__)[1]}
    output.parent.mkdir(parents=True, exist_ok=True)
    render(counts, output)
    counts["artifacts"] = {path.suffix[1:]: {"path": str(path), "sha256": stable_bytes(path)[1]} for path in paths[:2]}
    with paths[2].open("x") as stream:
        json.dump(counts, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"outputs": [str(path) for path in paths], "all_six_cohorts_validated": True}))


if __name__ == "__main__":
    main()
