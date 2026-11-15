#!/usr/bin/env python3
"""Render complete, audited contextual-input native counts without model calls.

Writes PNG, PDF and .counts.json from the frozen specification, raw results,
independent native audit and exact-overlap supplement. All registered conditions
are shown; shared text references repeat across the two layer panels. No effects,
confidence intervals, forced-answer replacements or outcome exclusions are used.
Existing output files are refused; choose a new output stem for a new snapshot.
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
LAYERS = (0, 10)
REFERENCES = (
    ("baseline", "Recipient text"),
    ("counterfactual_text", "Physical counterfactual text"),
    ("donor_text", "Matched donor text (= counterfactual)"),
    ("natural_donor_text", "Natural-context donor text"),
    ("same_fact_donor_text", "Same-fact donor text"),
    ("irrelevant_donor_text", "Irrelevant-fact donor text"),
)
INTERVENTIONS = (
    ("matched", "Matched fact transfer"),
    ("natural", "Natural-context transfer"),
    ("same_fact", "Same-fact control"),
    ("irrelevant", "Irrelevant-fact control"),
    ("random", "Random update (matched norm)*"),
    ("reverse", "Reverse transfer"),
    ("identity", "Recipient identity"),
    ("restoration_equivalent", "Counterfactual identity"),
)
NATIVE_STATUSES = ("complete", "truncated", "malformed")
SCOPE = "contextual_input_fact_interchange"


def read_hashed(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_native(row: dict) -> None:
    name = row["job_id"]
    status, action = row.get("generation_status"), row.get("native_action")
    check(row.get("status") == "complete" and status in NATIVE_STATUSES,
          f"Technical error or missing native endpoint: {name}")
    tokens, count, cap = row.get("generated_token_ids"), row.get("generated_tokens"), row["max_new_tokens"]
    check(isinstance(tokens, list) and type(count) is int and 0 < count <= cap
          and len(tokens) == count and all(type(t) is int and t >= 0 for t in tokens),
          f"Missing or invalid native token audit: {name}")
    if status == "complete":
        check(action in ACTIONS and row.get("parse_status") == "valid"
              and row.get("endpoint") == "native_final", f"Invalid native final: {name}")
        final = json.loads(row["final_text"])
        check(isinstance(final, dict) and str(final.get("action", "")).strip().upper() == action,
              f"Final JSON disagrees with native action: {name}")
    else:
        check(action is None, f"Unresolved output has an assigned native action: {name}")
        if status == "truncated":
            check(count == cap, f"Truncation differs from the registered cap: {name}")


def summarize(rows: list[dict]) -> dict:
    target = sum(r["native_action"] == r["target_action"] for r in rows)
    other = sum(r["native_action"] in ACTIONS and r["native_action"] != r["target_action"] for r in rows)
    unresolved = sum(r["native_action"] is None for r in rows)
    check(target + other + unresolved == len(rows), "Outcome categories do not exhaust the cohort")
    return {"n": len(rows), "target": target, "other": other, "unresolved": unresolved,
            "native_status_counts": dict(sorted(Counter(r["generation_status"] for r in rows).items())),
            "job_ids": sorted(r["job_id"] for r in rows)}


def load_counts(campaign: Path, experiment_id: str, *, audit_path: Path | None = None,
                overlap_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """Validate the complete raw cohort and both audits, then count every condition."""
    campaign = Path(campaign).resolve()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    check(re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id) is not None, "Invalid experiment ID")
    check(not (campaign / "reviews" / f"{experiment_id}_retired.json").exists(), "Retired experiment")
    spec_path = campaign / "specs" / f"{experiment_id}.json"
    spec, spec_sha = read_hashed(spec_path)
    check(spec.get("experiment_id") == experiment_id and spec.get("partition") == "discovery",
          "Require the matching discovery specification")
    check(spec.get("claim_scope") == SCOPE and spec.get("semantic_headline_eligible") is False,
          "Require explicitly scoped contextual input-fact discovery")
    jobs = spec.get("jobs", [])
    check(bool(jobs) and len({j["job_id"] for j in jobs}) == len(jobs), "Missing or duplicate planned jobs")
    conditions = [c for c, _ in REFERENCES] + [f"layer{layer}_{kind}" for layer in LAYERS for kind, _ in INTERVENTIONS]
    check(set(conditions) == {j["condition"] for j in jobs}, "Require every reference and all eight controls at L0/L10")
    result_dir = campaign / "results" / experiment_id
    planned, present = {j["job_id"] for j in jobs}, {p.stem for p in result_dir.glob("*.json")}
    # Refuse incomplete cohorts before reading any behavioral outcome.
    check(planned == present, f"Incomplete/extra cohort: {len(planned - present)} missing, {len(present - planned)} extra")
    audit_path = Path(audit_path or campaign / "reviews/contextual_input_native_independent_audit_v1.json").resolve()
    overlap_path = Path(overlap_path or campaign / "reviews/contextual_input_native_overlap_audit_v1.json").resolve()
    audit, audit_sha = read_hashed(audit_path)
    overlap, overlap_sha = read_hashed(overlap_path)
    check(audit.get("experiment_id") == experiment_id and audit.get("spec_sha256") == spec_sha,
          "Native audit does not bind this specification")
    check(overlap.get("format") == "contextual_input_native_overlap_audit_v1", "Wrong overlap audit format")
    hashes_read = {}

    def verify_map(mapping: dict) -> None:
        check(bool(mapping), "Empty source hash inventory")
        for relative, digest in mapping.items():
            path = (repo_root / relative).resolve()
            check(path.is_relative_to(repo_root), "Source hash path escapes repository root")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            check(actual == digest, f"Source file hash mismatch: {relative}")
            hashes_read[relative] = actual

    for mapping in (audit["raw_result_hashes"], audit["forced_raw_result_hashes"],
                    overlap["raw_result_hashes"], overlap["reviewed_specifications"]):
        verify_map(mapping)
    check(set(audit["raw_result_hashes"]) == {
        str((result_dir / f"{job_id}.json").relative_to(repo_root)) for job_id in planned
    }, "Native audit raw inventory is not exactly the planned cohort")
    check(overlap["reviewed_specifications"].get(str(spec_path.relative_to(repo_root))) == spec_sha,
          "Overlap audit does not bind this specification")
    for relative in overlap['reviewed_specifications']:
        source_id = Path(relative).stem
        check(not (campaign / 'reviews' / f'{source_id}_retired.json').exists(), 'Retired overlap source')
    manifest_text = "".join(f"{path}:{digest}\n" for path, digest in sorted(audit["raw_result_hashes"].items()))
    check(hashlib.sha256(manifest_text.encode()).hexdigest() == audit["raw_results_manifest_sha256"],
          "Raw results manifest hash mismatch")
    source_dataset = campaign / "datasets" / f"{spec['source_dataset_or_experiment']}.json"
    _, dataset_sha = read_hashed(source_dataset)
    check(dataset_sha == audit["source_dataset_sha256"], "Physical source dataset hash mismatch")
    forced_path = campaign / "specs/contextual_input_forced_v1.json"
    _, forced_sha = read_hashed(forced_path)
    check(forced_sha == audit["forced_spec_sha256"], "Forced source spec hash mismatch")
    check(not (campaign / 'reviews/contextual_input_forced_v1_retired.json').exists(), 'Retired forced source')
    rows, rows_by_id, grouped, identities = [], {}, defaultdict(dict), set()
    frozen_fields = (
        "job_id", "experiment_id", "condition", "layout_id", "pair_id", "family", "partition", "seed",
        "target_action", "original_action", "natural", "max_new_tokens", "temperature", "metadata",
        "intervention", "selection", "recipient_token_ids", "donor_token_ids", "recipient_positions", "donor_positions",
    )
    for job in jobs:
        check(job.get("natural") is True and job.get("semantic_test") is False
              and job.get("partition") == "discovery" and job.get("family") in FAMILIES,
              "Unscoped or non-native planned job")
        row, digest = read_hashed(result_dir / f"{job['job_id']}.json")
        check(digest == audit["raw_result_hashes"][str((result_dir / f"{job['job_id']}.json").relative_to(repo_root))],
              "Raw row changed during validation")
        check(row.get("spec_sha256") == spec_sha, "Raw row has stale specification hash")
        for field in frozen_fields:
            check(row.get(field) == job.get(field), f"Raw row differs from frozen {field}: {job['job_id']}")
        validate_native(row)
        check(row.get("inference_configuration", {}).get("head_mode") == "full", "Require full-head inference")
        check(bool(row.get("engine_key")) and bool(row.get("execution_key")), "Missing execution identity")
        identities.add((row["engine_key"], row["execution_key"], json.dumps(row["inference_configuration"], sort_keys=True)))
        key = (row["layout_id"], row["pair_id"], row["seed"])
        check(row["condition"] not in grouped[key], "Duplicate paired condition")
        grouped[key][row["condition"]] = row
        rows.append(row)
        rows_by_id[row["job_id"]] = row
    check(len(identities) == 1, "Mixed execution identities")
    check(len({j['seed'] for j in jobs}) == 1, "Dedicated figure requires one shared seed")
    check(len(grouped) == len({r["layout_id"] for r in rows}), "Dedicated figure requires one pair/seed per layout")
    for group in grouped.values():
        check(set(group) == set(conditions), "Incomplete controls for a paired layout")
        for field in ("target_action", "original_action", "family", "max_new_tokens", "temperature"):
            check(len({r[field] for r in group.values()}) == 1, f"Paired {field} changes")
        check(group['baseline']['target_action'] in ACTIONS and group['baseline']['original_action'] in ACTIONS
              and group['baseline']['target_action'] != group['baseline']['original_action'], "Invalid paired target")
        for field in ('recipient_token_ids', 'generated_token_ids', 'forced_q'):
            check(group['donor_text'][field] == group['counterfactual_text'][field],
                  'Matched donor reference is not the physical counterfactual duplicate')
    check(len(rows) == audit['n_results'] and len(grouped) == audit['n_independent_layouts'], "Audit denominator mismatch")
    check(Counter(r['generation_status'] for r in rows) == audit['generation_status_counts'], "Audit status count mismatch")
    condition_counts = {}
    for condition in conditions:
        selected = [r for r in rows if r['condition'] == condition]
        count = summarize(selected)
        expected = audit['condition_summaries'][condition]
        check((count['n'], count['target'], count['native_status_counts']) ==
              (expected['n'], expected['target_count'], expected['status']), "Audit condition count mismatch")
        count['by_family'] = {family: summarize([r for r in selected if r['family'] == family]) for family in FAMILIES}
        for family, item in count['by_family'].items():
            expected_family = expected['by_family'][family]
            check((item['n'], item['target'], item['unresolved']) ==
                  (expected_family['n'], expected_family['target'], expected_family['incomplete']),
                  "Audit family count mismatch")
        condition_counts[condition] = count
    self_replacements = 0
    for condition, expected in audit['identities'].items():
        for group in grouped.values():
            row, baseline = group[condition], group[expected['control']]
            check(row['generated_token_ids'] == baseline['generated_token_ids']
                  and row['forced_q'] == baseline['forced_q'], "Self replacement differs from baseline")
            logs = row.get('patch_log', [])
            check(bool(logs) and all(log.get('delta_l2') and all(v == 0 for v in log['delta_l2']) for log in logs),
                  "Self replacement lacks zero applied-norm audit")
            self_replacements += 1
    check(len(overlap['checks']) == overlap['comparison_count'], "Incomplete repeated-execution audit")
    earlier_paths = {}
    for relative in overlap['raw_result_hashes']:
        path = repo_root / relative
        if path.parent != result_dir:
            earlier_paths[path.stem] = path
    overlap_job_ids = []
    for item in overlap['checks']:
        row = rows_by_id[item['job_id']]
        earlier, _ = read_hashed(earlier_paths[item['earlier_job_id']])
        for field in ('recipient_token_ids', 'input_token_sha256', 'forced_q', 'generated_token_ids', 'layout_id', 'pair_id', 'seed'):
            check(row[field] == earlier[field], f"Earlier execution differs in {field}")
        overlap_job_ids.append(row['job_id'])
    check(len(set(overlap_job_ids)) == len(overlap_job_ids), "Duplicate repeated-execution checks")
    donors = [r for r in rows if r['condition'] == 'natural_donor_text']
    donor_mismatch = sum(r['native_action'] in ACTIONS and r['native_action'] != r['target_action'] for r in donors)
    parsed_donors = sum(r['native_action'] in ACTIONS for r in donors)
    random_ratios = {}
    for layer in LAYERS:
        ratios = []
        for group in grouped.values():
            random_log, matched_log = group[f'layer{layer}_random']['patch_log'], group[f'layer{layer}_matched']['patch_log']
            check(len(random_log) == len(matched_log) == 1, 'Unexpected random/matched patch geometry')
            numer, denom = random_log[0]['delta_l2'], matched_log[0]['delta_l2']
            check(len(numer) == len(denom) and len(numer) > 0, 'Missing random norm audit')
            for num, den in zip(numer, denom):
                check(math.isfinite(num) and math.isfinite(den) and den > 0, 'Invalid matched norm reference')
                ratios.append(num / den)
        random_ratios[str(layer)] = {'reference_condition': f'layer{layer}_matched',
                                     'min': min(ratios), 'max': max(ratios), 'per_position_ratios': ratios}
        audited = [item for item in audit['random_norm_diagnostics'] if item['layer'] == layer]
        check(all(item['reference_condition'] == f'layer{layer}_matched' for item in audited),
              'Independent audit uses a different random norm reference')
        check(sorted(ratios) == sorted(value for item in audited for value in item['per_position_random_over_matched']),
              'Recomputed applied norm ratios disagree with the independent audit')
    engine_key, execution_key, inference = next(iter(identities))
    return {
        'format': 'contextual_native_figure_counts_v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'experiment_id': experiment_id, 'partition': 'discovery', 'claim_scope': SCOPE,
        'spec': {'path': str(spec_path), 'sha256': spec_sha},
        'native_audit': {'path': str(audit_path), 'sha256': audit_sha},
        'overlap_audit': {'path': str(overlap_path), 'sha256': overlap_sha},
        'physical_source_dataset': {'path': str(source_dataset), 'file_sha256': dataset_sha},
        'forced_source_spec': {'path': str(forced_path), 'sha256': forced_sha},
        'engine_key': engine_key, 'execution_key': execution_key, 'inference_configuration': json.loads(inference),
        'n_planned_jobs': len(jobs), 'n_technically_complete_jobs': len(rows), 'n_layouts': len(grouped),
        'layouts_per_family': dict(Counter(group['baseline']['family'] for group in grouped.values())),
        'layout_ids': sorted({r['layout_id'] for r in rows}), 'seeds': sorted({r['seed'] for r in rows}),
        'token_caps': sorted({r['max_new_tokens'] for r in rows}), 'temperatures': sorted({r['temperature'] for r in rows}),
        'native_status_counts': dict(sorted(Counter(r['generation_status'] for r in rows).items())),
        'conditions': condition_counts, 'reference_conditions': [c for c, _ in REFERENCES], 'layers': list(LAYERS),
        'natural_donor_observations': {'n': len(donors), 'parsed': parsed_donors, 'observed_target_mismatch': donor_mismatch,
                                       'unresolved': len(donors) - parsed_donors, 'outcome_based_exclusions': 0},
        'exact_self_replacement_count': self_replacements, 'exact_earlier_execution_count': len(overlap_job_ids),
        'exact_earlier_job_ids': sorted(overlap_job_ids), 'random_norm_reference': 'matched counterfactual; not natural-context donor',
        'random_applied_norm_ratios': random_ratios, 'pre_query_eligibility': {
            k: spec.get(k) for k in ('eligibility_manifest', 'alignment_exclusions', 'physical_pair_exclusions', 'ineligible_comparisons')
        },
        'raw_result_hashes': audit['raw_result_hashes'], 'raw_results_manifest_sha256': audit['raw_results_manifest_sha256'],
        'raw_results_manifest_definition': audit['raw_results_manifest_definition'], 'verified_source_file_hashes': hashes_read,
        'definitions': {
            'target': 'Native final action equals the recipient-relative simulator-optimal physical-counterfactual action.',
            'other': 'Resolved native action differs from that recipient target; it may be correct in a donor world.',
            'unresolved': 'Truncated or malformed native output; retained in the planned denominator, no forced fallback.',
            'reference_repetition': 'The same six text-reference conditions are displayed in both layer panels; no extra executions.',
            'scope': 'Descriptive counts on shared discovery layouts; no population intervals, abstract-belief, necessity or mediation claim.',
            'identity': 'Recipient and counterfactual self replacement; not an independent removal/rescue experiment.',
        },
    }


def render(counts: dict, output: Path) -> tuple[Path, Path, Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output = Path(output)
    if output.suffix in {'.png', '.pdf', '.json'}:
        output = output.with_suffix('')
    paths = tuple(Path(str(output) + suffix) for suffix in ('.png', '.pdf', '.counts.json'))
    check(not any(p.exists() for p in paths), 'Output exists; choose a new immutable output stem')
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 9.7), sharey=True)
    fig.subplots_adjust(left=.27, right=.968, top=.805, bottom=.23, wspace=.17)
    colors = {'target': '#197f76', 'other': '#849bac', 'unresolved': '#f0c56b'}
    labels = [label for _, label in REFERENCES] + [label for _, label in INTERVENTIONS]
    n = counts['n_layouts']
    for ax, layer in zip(axes, LAYERS):
        order = [c for c, _ in REFERENCES] + [f'layer{layer}_{kind}' for kind, _ in INTERVENTIONS]
        ax.axhspan(-.5, len(REFERENCES) - .5, color='#f3f5f7', zorder=0)
        for i, condition in enumerate(order):
            row = counts['conditions'][condition]
            left = 0
            for category in ('target', 'other', 'unresolved'):
                value = row[category]
                ax.barh(i, value, left=left, height=.64, color=colors[category],
                        hatch='///' if category == 'unresolved' else None,
                        edgecolor='#9a7f48' if category == 'unresolved' else 'white', linewidth=.35, zorder=2)
                left += value
            ax.text(n + .22, i, f"{row['target']} / {row['other']} / {row['unresolved']}",
                    va='center', fontsize=10, color='#29475b')
        ax.axhline(len(REFERENCES) - .5, color='#b9c5ce', linewidth=.8)
        ax.set_xlim(0, n + 1.38)
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_xlabel('Planned layouts', labelpad=7, color='#486073')
        ax.set_title(f'Layer {layer}', loc='left', fontsize=14, fontweight='bold', color='#253f51', pad=14)
        ax.text(.98, 1.021, 'Target / Other / ?', transform=ax.transAxes, ha='right', fontsize=8.7, color='#486073')
        ax.grid(axis='x', color='#e3e9ee', linewidth=.6, zorder=1)
        ax.tick_params(axis='both', length=0, pad=7, colors='#334f63')
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].set_ylim(len(labels) - .35, -.7)
    fig.text(.035, .952, 'Input-fact transfer across donor contexts', fontsize=20, fontweight='bold', color='#203b4e')
    families = counts['layouts_per_family']
    shared = f"{n} shared discovery layouts  |  " + ' / '.join(str(families[f]) for f in FAMILIES) + ' wall / key / door layouts'
    fig.text(.035, .917, shared, fontsize=11, color='#435e72')
    cap = ', '.join(f'{value:,}' for value in counts['token_caps'])
    statuses = counts['native_status_counts']
    fig.text(.035, .887,
             f"{counts['n_technically_complete_jobs']}/{counts['n_planned_jobs']} technically complete executions; "
             f"{statuses.get('complete', 0)} final actions, {sum(statuses.get(s, 0) for s in ('truncated', 'malformed'))} unresolved  |  "
             f"seed {', '.join(map(str, counts['seeds']))}  |  cap {cap} tokens",
             fontsize=10.5, color='#435e72')
    fig.legend(handles=[Patch(facecolor=colors['target'], label='Recipient counterfactual target'),
                        Patch(facecolor=colors['other'], label='Other resolved action'),
                        Patch(facecolor=colors['unresolved'], edgecolor='#9a7f48', hatch='///', label='Unresolved (?)')],
               loc='upper left', bbox_to_anchor=(.027, .868), ncol=3, frameon=False, fontsize=10)
    donor = counts['natural_donor_observations']
    notes = [
        "Target is the recipient's counterfactual-optimal action; an 'other' action can be correct in a donor's world.",
        f"Natural donors: {donor['observed_target_mismatch']}/{donor['parsed']} observed actions differ from this target; "
        f"{donor['unresolved']}/{donor['n']} unresolved. All planned cases remain included.",
        f"Text references (shaded) repeat across panels. {counts['exact_earlier_execution_count']} executions exactly repeat the earlier input-native cohort.",
        "* Random norms reference the matched same-context counterfactual update, not the natural-context donor update.",
        f"{counts['exact_self_replacement_count']} exact self replacements; no independent rescue. Shared layouts are not independent replications.",
        "Descriptive counts only: no population intervals, abstract-belief, necessity or mediation claim.",
    ]
    for i, note in enumerate(notes):
        fig.text(.035, .168 - i * .024, note, fontsize=9.1, color='#435e72')
    fig.text(.035, .025, f"{counts['experiment_id']}  |  Spec {counts['spec']['sha256'][:16]}...  |  "
             'All conditions, family counts and source hashes: companion .counts.json', fontsize=8.2, color='#637b8d')
    png, pdf, raw = paths
    counts['figure_script'] = {'path': str(Path(__file__).resolve()),
                               'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    counts['artifacts'] = {'png': str(png.resolve()), 'pdf': str(pdf.resolve()), 'counts': str(raw.resolve())}
    with png.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=220, facecolor='white')
    with pdf.open('xb') as stream:
        fig.savefig(stream, format='pdf', facecolor='white',
                    metadata={'Title': 'Contextual input-fact native actions: shared discovery layouts'})
    plt.close(fig)
    counts['artifact_sha256'] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (('png', png), ('pdf', pdf))}
    with raw.open('x') as stream:
        json.dump(counts, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--overlap-audit', type=Path)
    parser.add_argument('--repo-root', type=Path)
    args = parser.parse_args()
    try:
        counts = load_counts(args.campaign, args.experiment_id, audit_path=args.audit,
                             overlap_path=args.overlap_audit, repo_root=args.repo_root)
        paths = render(counts, args.output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f'Cannot render contextual native figure: {exc}\n')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
