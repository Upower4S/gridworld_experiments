#!/usr/bin/env python3
"""Report a complete frozen factual-readout cohort without running a model.

Writes <output>.json and <output>.md. Explicit unavailable probes remain in
denominators; missing jobs/probes, errors, provenance drift and mixed executions
fail closed. This is descriptive forced-report analysis, with no native-action,
semantic, mediation, confidence-interval or significance claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activation_causality.fact_readout import CLAIM_SCOPE, _digest, build_fact_probes  # noqa: E402
from src.activation_causality.registry import atomic_json, fingerprint  # noqa: E402

PROBES = ("target_initial_fact", "unchanged_remote_grid_fact")
OMITTED_RESULT_FIELDS = {"recipient_text", "donor_text", "shared_base_text"}


def read_hashed(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid {label}")
    return value


def _job_id(job: dict) -> str:
    return fingerprint({key: value for key, value in job.items() if key != "job_id"})[:24]


def _active_spec(campaign: Path, experiment_id: str | None) -> tuple[dict, Path, str]:
    if experiment_id is None:
        candidates = []
        for path in sorted((campaign / "specs").glob("*.json")):
            spec, _ = read_hashed(path)
            if spec.get("claim_scope") == CLAIM_SCOPE and not (campaign / "reviews" / f"{path.stem}_retired.json").exists():
                candidates.append(path.stem)
        if len(candidates) != 1:
            raise ValueError(f"Specify --experiment-id: found {len(candidates)} active factual-readout derivatives")
        experiment_id = candidates[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Invalid experiment_id")
    if (campaign / "reviews" / f"{experiment_id}_retired.json").exists():
        raise ValueError("Refusing retired factual-readout experiment")
    path = campaign / "specs" / f"{experiment_id}.json"
    spec, digest = read_hashed(path)
    if spec.get("experiment_id") != experiment_id or spec.get("partition") != "discovery" or spec.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("Require matching frozen discovery factual-readout specification")
    return spec, path, digest


def _verify_source(campaign: Path, spec: dict) -> dict:
    source_id = spec.get("source_experiment_id")
    if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise ValueError("Missing or invalid frozen source experiment ID")
    if (campaign / "reviews" / f"{source_id}_retired.json").exists():
        raise ValueError("Refusing retired source experiment")
    source_path = campaign / "specs" / f"{source_id}.json"
    source, source_file_hash = read_hashed(source_path)
    source_hash = fingerprint(source)
    if spec.get("source_spec_sha256") != source_file_hash or spec.get("source_spec_fingerprint") != source_hash:
        raise ValueError("Frozen source file hash or specification fingerprint mismatch")
    if source.get("experiment_id") != source_id or source.get("partition") != "discovery":
        raise ValueError("Source specification identity or partition mismatch")
    if not str(source.get("claim_scope", "")).startswith("input_fact_information_routing"):
        raise ValueError("Unsupported source population/context")
    original_jobs = {job["job_id"]: job for job in source.get("jobs", [])}
    jobs = spec.get("jobs", [])
    if not original_jobs or len(original_jobs) != len(source["jobs"]) or len(jobs) != len(original_jobs):
        raise ValueError("Derivative does not preserve the complete unique source cohort")
    if spec.get("n_source_jobs") != len(original_jobs) or spec.get("n_derived_jobs") != len(jobs) or spec.get("complete_source_cohort_preserved") is not True:
        raise ValueError("Source cohort count declarations mismatch")
    if any(job["job_id"] != _job_id(job) for job in source["jobs"]):
        raise ValueError("Source job ID does not match its frozen definition")
    expected_map, physical_pairs, inventories = {}, {}, defaultdict(set)
    for job in jobs:
        if job.get("job_id") != _job_id(job) or job.get("experiment_id") != spec["experiment_id"]:
            raise ValueError("Derivative job ID or experiment mismatch")
        if job.get("natural") is not False or job.get("partition") != "discovery" or job.get("semantic_test") is not False or job.get("claim_scope") != CLAIM_SCOPE:
            raise ValueError("Require forced-only discovery diagnostic jobs")
        diagnostic = job.get("fact_readout", {})
        original_id = diagnostic.get("source_job_id")
        if original_id not in original_jobs or original_id in expected_map:
            raise ValueError("Derivative source job map is not one-to-one")
        original = original_jobs[original_id]
        original_hash = fingerprint(original)
        if diagnostic.get("source_job_fingerprint") != original_hash or diagnostic.get("source_spec_fingerprint") != source_hash or diagnostic.get("source_experiment_id") != source_id:
            raise ValueError("Derivative job source fingerprint mismatch")
        allowed_changes = {"job_id", "experiment_id", "natural", "semantic_test", "semantic_headline_eligible", "claim_scope", "metadata"}
        for key, value in original.items():
            if key not in allowed_changes and job.get(key) != value:
                raise ValueError(f"Derivative changed frozen source field {key}")
        expected_metadata = {**deepcopy(original["metadata"]), "fact_readout": diagnostic}
        if job.get("metadata") != expected_metadata:
            raise ValueError("Derivative changed original source metadata or diagnostic lineage")
        pair = original["metadata"].get("source_pair", original["metadata"].get("pair"))
        if not isinstance(pair, dict) or pair.get("partition") != "discovery":
            raise ValueError("Missing physical source pair")
        if diagnostic.get("source_pair_fingerprint") != fingerprint(pair) or job.get("fact_probes") != build_fact_probes(pair):
            raise ValueError("Fact probes or physical source-pair fingerprint mismatch")
        key = job["pair_id"]
        if key in physical_pairs and physical_pairs[key] != fingerprint(pair):
            raise ValueError("Pair facts differ across source conditions")
        physical_pairs[key] = fingerprint(pair)
        inventory = (job["condition"], job["seed"])
        if inventory in inventories[key]:
            raise ValueError("Duplicate source pair/condition/seed")
        inventories[key].add(inventory)
        expected_map[original_id] = {"derived_job_id": job["job_id"], "source_job_fingerprint": original_hash}
    if spec.get("source_job_map") != expected_map or set(expected_map) != set(original_jobs):
        raise ValueError("Frozen source_job_map differs from complete source lineage")
    if len({tuple(sorted(inventory)) for inventory in inventories.values()}) != 1:
        raise ValueError("Incomplete per-pair condition/seed cohort")
    if spec.get("source_action_comparisons") != source.get("comparisons", []) or spec.get("comparisons") != []:
        raise ValueError("Original action comparisons changed or mixed with factual endpoints")
    return {"experiment_id": source_id, "path": str(source_path), "file_sha256": source_file_hash,
            "specification_fingerprint": source_hash, "n_jobs": len(original_jobs)}


def _probability(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid probability {label}")
    return float(value)


def _validate_probe(job: dict, probe: dict, report: dict) -> None:
    if any(report.get(key) != value for key, value in probe.items()) or report.get("probe_sha256") != _digest(probe):
        raise ValueError("Factual report differs from frozen probe or probe hash")
    if report.get("source_state_unchanged") is not True:
        raise ValueError("Factual report lacks source-state preservation proof")
    base = report.get("base_state", {})
    prefix_hash = _digest(job["recipient_token_ids"])
    if base.get("token_ids_sha256") != prefix_hash or base.get("n_tokens") != len(job["recipient_token_ids"]):
        raise ValueError("Factual report base-state token provenance mismatch")
    if report.get("status") == "unavailable":
        if not isinstance(report.get("unavailable_reason"), str) or not report["unavailable_reason"] or report.get("query_performed") is not False:
            raise ValueError("Unavailable factual report needs an explicit reason and no query")
        return
    if report.get("status") != "scored" or probe.get("available") is not True:
        raise ValueError("Factual report must be scored or explicitly unavailable")
    if report.get("query_performed") is not True or report.get("boundary_verified") is not True or report.get("prefix_reencoded") is not False:
        raise ValueError("Scored factual report lacks exact query-boundary proof")
    if report.get("source_state_after") != base or report.get("authoritative_prefix_token_ids_sha256") != prefix_hash:
        raise ValueError("Scored factual report state/hash preservation mismatch")
    q = report.get("q", {})
    if set(q) != {"true", "false"} or abs(sum(_probability(q[key], key) for key in q) - 1) > 1e-6:
        raise ValueError("Invalid normalized Boolean report probabilities")
    for role in ("recipient", "counterfactual"):
        value = _probability(report.get(role + "_truth_q"), role + "_truth_q")
        if abs(value - q[str(probe[role + "_value"]).lower()]) > 1e-8:
            raise ValueError("Factual truth probability disagrees with frozen Boolean truth")
    mass = _probability(report.get("candidate_mass"), "candidate_mass")
    global_q = report.get("candidate_global_probabilities", {})
    if set(global_q) != {"true", "false"} or any(abs(_probability(global_q[key], "global_" + key) - mass * q[key]) > 1e-6 for key in q):
        raise ValueError("Boolean candidate mass disagrees with global probabilities")


def weighted_summary(records: list[dict], field: str) -> dict:
    """Average seeds within pairs, pairs within layouts, then layouts equally."""
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get(field) is not None:
            grouped[record["layout_id"]][record["pair_id"]].append(record[field])
    per_layout = [{"layout_id": layout, "mean": mean(mean(values) for values in pairs.values()),
                   "n_pairs": len(pairs), "n_pair_seed_observations": sum(len(values) for values in pairs.values())}
                  for layout, pairs in sorted(grouped.items())]
    return {"mean": mean(row["mean"] for row in per_layout) if per_layout else None,
            "n_available_layouts": len(per_layout), "per_layout": per_layout}


def load_report(campaign: Path, experiment_id: str | None = None) -> dict:
    """Audit raw files and return descriptive diagnostics; never filter on actions."""
    campaign = Path(campaign).resolve()
    spec, spec_path, spec_hash = _active_spec(campaign, experiment_id)
    source_provenance = _verify_source(campaign, spec)
    experiment_id, jobs = spec["experiment_id"], spec["jobs"]
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate derivative job IDs")
    result_dir = campaign / "results" / experiment_id
    planned = {job["job_id"] for job in jobs}
    present = {path.stem for path in result_dir.glob("*.json")}
    if present != planned:
        raise ValueError(f"Incomplete or unexpected result cohort: {len(planned - present)} missing, {len(present - planned)} unexpected")
    observations, source_rows, identities = [], [], set()
    by_condition = defaultdict(list)
    indexed = {}
    for job in jobs:
        path = result_dir / f"{job['job_id']}.json"
        row, row_hash = read_hashed(path)
        if row.get("spec_sha256") != spec_hash:
            raise ValueError("Result/derivative specification hash mismatch")
        for key, value in job.items():
            if key not in OMITTED_RESULT_FIELDS and row.get(key) != value:
                raise ValueError(f"Result differs from frozen job field {key}")
        if row.get("status") != "complete" or row.get("generation_status") != "not_requested" or row.get("native_action") is not None:
            raise ValueError("Require complete forced-only result; partial/error/native rows are not accepted")
        if row.get("input_token_sha256") != fingerprint(job["recipient_token_ids"]) or row.get("fact_probes_sha256") != fingerprint(job["fact_probes"]):
            raise ValueError("Input-token or factual-probe fingerprint mismatch")
        identity = (_sha(row.get("engine_key"), "engine key"), _sha(row.get("execution_key"), "execution key"),
                    _sha(row.get("fact_readout_implementation_sha256"), "fact helper SHA"),
                    json.dumps(row.get("inference_configuration"), sort_keys=True))
        if row.get("inference_configuration") != {"head_mode": "full"}:
            raise ValueError("Missing or unsupported full-head inference configuration")
        identities.add(identity)
        reports = row.get("fact_reports")
        if not isinstance(reports, list) or any(not isinstance(report, dict) for report in reports) or [report.get("id") for report in reports] != [probe["id"] for probe in job["fact_probes"]]:
            raise ValueError("Missing, duplicated, reordered or unexpected factual probe results")
        scored = 0
        for probe, report in zip(job["fact_probes"], reports, strict=True):
            _validate_probe(job, probe, report)
            available = report["status"] == "scored"
            scored += available
            observation = {key: job[key] for key in ("job_id", "layout_id", "pair_id", "seed", "family", "condition")}
            observation.update(probe_id=probe["id"], status=report["status"],
                recipient_value=probe["recipient_value"], counterfactual_value=probe["counterfactual_value"],
                counterfactual_truth_q=report.get("counterfactual_truth_q") if available else None,
                recipient_truth_q=report.get("recipient_truth_q") if available else None,
                candidate_mass=report.get("candidate_mass") if available else None,
                unavailable_reason=report.get("unavailable_reason"), report_fingerprint=fingerprint(report))
            observations.append(observation)
            by_condition[job["family"], job["condition"], probe["id"]].append(observation)
            indexed[job["condition"], job["pair_id"], job["seed"], probe["id"]] = observation
        expected_status = "complete" if scored == len(reports) else "partial" if scored else "unavailable"
        if (row.get("fact_readout_status") != expected_status or row.get("fact_readout_n_requested") != len(reports)
                or row.get("fact_readout_n_scored") != scored or row.get("fact_readout_n_unavailable") != len(reports) - scored):
            raise ValueError("Factual readout status/count declarations disagree with explicit probe results")
        source_rows.append({"job_id": job["job_id"], "path": str(path), "sha256": row_hash})
    if len(identities) != 1:
        raise ValueError("Mixed execution, engine, helper or inference identities")
    summaries = []
    for (family, condition, probe_id), records in sorted(by_condition.items()):
        available = [record for record in records if record["status"] == "scored"]
        summaries.append({"family": family, "condition": condition, "probe_id": probe_id,
            "n_planned": len(records), "n_available": len(available), "n_unavailable": len(records) - len(available),
            "n_planned_layouts": len({record["layout_id"] for record in records}),
            "unavailable_reasons": dict(Counter(record["unavailable_reason"] for record in records if record["status"] != "scored")),
            "truth_probability": weighted_summary(records, "counterfactual_truth_q"),
            "candidate_mass": weighted_summary(records, "candidate_mass")})
    source_comparisons = spec["source_action_comparisons"]
    source_comparison_hashes = {fingerprint(comparison) for comparison in source_comparisons}
    paired_comparisons = []
    expected_comparison_keys = {(fingerprint(comparison), probe_id) for comparison in source_comparisons for probe_id in PROBES}
    observed_comparison_keys = set()
    for comparison in spec.get("fact_readout_comparisons", []):
        if (comparison.get("experiment_id") != experiment_id or comparison.get("partition") != "discovery"
                or comparison.get("endpoint") != "elicited_forced_fact_report"
                or comparison.get("probability_key") != "counterfactual_truth_q"
                or comparison.get("semantic_headline_eligible") is not False):
            raise ValueError("Factual comparison endpoint, identity or claim scope mismatch")
        source_comparison = comparison.get("source_comparison", {})
        source_comparison_hash = fingerprint(source_comparison)
        probe_id = comparison.get("probe_id")
        if source_comparison_hash not in source_comparison_hashes or probe_id not in PROBES:
            raise ValueError("Factual comparison is not derived from a frozen source comparison")
        comparison_key = source_comparison_hash, probe_id
        if comparison_key in observed_comparison_keys:
            raise ValueError("Duplicate factual comparison")
        observed_comparison_keys.add(comparison_key)
        if any(comparison.get(key) != source_comparison.get(key) for key in ("treatment", "control")):
            raise ValueError("Factual comparison changed source treatment/control")
        treatment, control = comparison["treatment"], comparison["control"]
        expected_keys = {(job["pair_id"], job["seed"]) for job in jobs if job["condition"] == treatment} & {
            (job["pair_id"], job["seed"]) for job in jobs if job["condition"] == control}
        if "eligible_pair_ids" in source_comparison:
            expected_keys = {key for key in expected_keys if key[0] in source_comparison["eligible_pair_ids"]}
        if comparison.get("eligible_pair_ids") != sorted({key[0] for key in expected_keys}) or comparison.get("expected_n_pairs") != len(expected_keys):
            raise ValueError("Factual comparison cohort/count differs from frozen source")
        pairs = []
        for pair_id, seed in sorted(expected_keys):
            left, right = indexed[treatment, pair_id, seed, probe_id], indexed[control, pair_id, seed, probe_id]
            available = left["status"] == right["status"] == "scored"
            pairs.append({"pair_id": pair_id, "seed": seed, "layout_id": left["layout_id"], "family": left["family"],
                "treatment_job_id": left["job_id"], "control_job_id": right["job_id"], "available": available,
                "delta": left["counterfactual_truth_q"] - right["counterfactual_truth_q"] if available else None,
                "candidate_mass_delta": left["candidate_mass"] - right["candidate_mass"] if available else None,
                "treatment_unavailable_reason": left["unavailable_reason"], "control_unavailable_reason": right["unavailable_reason"]})
        if comparison.get("expected_n_layouts") != len({pair["layout_id"] for pair in pairs}):
            raise ValueError("Factual comparison layout count mismatch")
        def paired_summary(records: list[dict]) -> dict:
            return {"n_planned_pair_seed": len(records), "n_available_pair_seed": sum(record["available"] for record in records),
                    "n_unavailable_pair_seed": sum(not record["available"] for record in records),
                    "n_planned_layouts": len({record["layout_id"] for record in records}),
                    "delta": weighted_summary(records, "delta"), "candidate_mass_delta": weighted_summary(records, "candidate_mass_delta")}
        paired_comparisons.append({"id": comparison["id"], "treatment": treatment, "control": control, "probe_id": probe_id,
            "overall": paired_summary(pairs), "by_family": {family: paired_summary([pair for pair in pairs if pair["family"] == family])
                for family in sorted({pair["family"] for pair in pairs})}, "paired_observations": pairs})
    if observed_comparison_keys != expected_comparison_keys:
        raise ValueError("Missing frozen factual comparisons")
    engine_key, execution_key, helper_sha, inference_json = next(iter(identities))
    return {"schema_version": 1, "claim_scope": CLAIM_SCOPE, "experiment_id": experiment_id,
        "campaign": str(campaign), "spec_path": str(spec_path), "spec_sha256": spec_hash,
        "specification_fingerprint": fingerprint(spec), "source_specification": source_provenance,
        "engine_key": engine_key, "execution_key": execution_key, "fact_readout_implementation_sha256": helper_sha,
        "inference_configuration": json.loads(inference_json), "n_planned_jobs": len(jobs), "n_completed_jobs": len(jobs),
        "n_requested_probes": len(observations), "n_scored_probes": sum(record["status"] == "scored" for record in observations),
        "n_unavailable_probes": sum(record["status"] != "scored" for record in observations),
        "condition_summaries": summaries, "paired_comparisons": paired_comparisons, "raw_observations": observations,
        "source_result_files": source_rows, "source_exclusions": {key: deepcopy(spec.get(key)) for key in (
            "exclusions", "exclusion_counts", "eligibility_manifest", "physical_pair_exclusions", "alignment_exclusions",
            "source_ineligible_action_comparisons") if key in spec},
        "definitions": {"weighting": "Average seeds within pair, pairs within layout, then layouts equally; paired deltas are computed before aggregation.",
            "availability": "All planned jobs/probes retained. Numeric means condition on scored reports, with explicit unavailable denominators; unavailable reports are not zeros.",
            "target_truth_probability": "Forced Boolean probability assigned to the original physical pair's counterfactual target-fact truth.",
            "unchanged_truth_probability": "Forced Boolean probability of a remote fact unchanged between physical R/CF worlds; route irrelevance is not asserted.",
            "candidate_mass": "Total model probability of the two Boolean candidates before their conditional renormalization.",
            "native_action_filters": False, "native_action_claim": False, "semantic_claim": False, "mediation_claim": False,
            "statistical_inference": "None: descriptive means and paired deltas only; no confidence intervals or significance tests."}}


def write_report(report: dict, output: Path) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix in {".json", ".md"}:
        output = output.with_suffix("")
    json_path, markdown_path = Path(str(output) + ".json"), Path(str(output) + ".md")
    atomic_json(json_path, report)
    def number(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.6f}"
    lines = [f"# Factual report diagnostic: {report['experiment_id']}", "",
        f"Complete frozen cohort: {report['n_completed_jobs']}/{report['n_planned_jobs']} jobs. "
        f"Scored probes: {report['n_scored_probes']}/{report['n_requested_probes']}; explicitly unavailable: {report['n_unavailable_probes']}.", "",
        "These are elicited forced Boolean reports. They do not establish native action effects, latent belief decoding, semantic causality, or mediation. No confidence intervals or significance tests are reported.", "",
        report["definitions"]["weighting"], report["definitions"]["availability"], "",
        "| Family | Condition | Probe | Truth probability | Boolean candidate mass | Available / planned |", "|---|---|---|---:|---:|---:|"]
    for row in report["condition_summaries"]:
        label = "Target counterfactual truth" if row["probe_id"] == PROBES[0] else "Unchanged fact truth"
        lines.append(f"| {row['family']} | {row['condition']} | {label} | {number(row['truth_probability']['mean'])} | {number(row['candidate_mass']['mean'])} | {row['n_available']} / {row['n_planned']} |")
    lines += ["", "Paired deltas below use treatment minus control on the same pair and seed. Family-specific deltas, per-layout means, raw observations, unavailable reasons and file hashes are preserved in the JSON companion.", "",
        "| Treatment | Control | Probe | Paired truth-probability delta | Available / planned |", "|---|---|---|---:|---:|"]
    for row in report["paired_comparisons"]:
        summary = row["overall"]
        lines.append(f"| {row['treatment']} | {row['control']} | {row['probe_id']} | {number(summary['delta']['mean'])} | {summary['n_available_pair_seed']} / {summary['n_planned_pair_seed']} |")
    reasons = Counter(row["unavailable_reason"] for row in report["raw_observations"] if row["status"] != "scored")
    lines += ["", "Unavailable reasons: " + (", ".join(f"{reason} ({count})" for reason, count in sorted(reasons.items())) or "none") + ".", "",
        f"Derivative specification SHA-256: `{report['spec_sha256']}`.",
        f"Frozen source file SHA-256: `{report['source_specification']['file_sha256']}`.",
        f"Execution: `{report['execution_key']}`. Factual-readout helper: `{report['fact_readout_implementation_sha256']}`.", ""]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--experiment-id", help="Optional when exactly one active factual derivative exists")
    parser.add_argument("--output", type=Path, help="Output stem; defaults to campaign/reports/<experiment>_fact_report")
    args = parser.parse_args()
    try:
        report = load_report(args.campaign, args.experiment_id)
        output = args.output or args.campaign / "reports" / f"{report['experiment_id']}_fact_report"
        json_path, markdown_path = write_report(report, output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Fact report refused: {exc}\n")
    print(json.dumps({"experiment_id": report["experiment_id"], "json": str(json_path), "markdown": str(markdown_path),
                      "jobs": report["n_completed_jobs"], "scored_probes": report["n_scored_probes"], "unavailable_probes": report["n_unavailable_probes"]}))


if __name__ == "__main__":
    main()
