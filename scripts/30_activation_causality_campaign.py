#!/usr/bin/env python3
"""Run the local cooperative activation-causality campaign from the repo root."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.activation_causality.registry import DEFAULT_CAMPAIGN, DEFAULT_MODEL, init_campaign, read_json, status  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--hours", type=float, default=72)
    initialize.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    sub.add_parser("status")
    sentence = sub.add_parser("prepare-sentence")
    sentence.add_argument("--experiment-id", required=True)
    sentence.add_argument("--cuts", type=int, default=12)
    sentence.add_argument("--seeds", default="42")
    sentence.add_argument("--max-new-tokens", type=int, default=2048)
    sentence.add_argument("--forced-only", action="store_true")
    states = sub.add_parser("prepare-states")
    states.add_argument("--experiment-id", required=True)
    states.add_argument("--layouts", type=int, default=24)
    states.add_argument("--seed", type=int, default=20260912)
    states.add_argument("--families", default="wall,key_possession,door_status")
    states.add_argument("--max-new-tokens", type=int, default=2048)
    states.add_argument("--forced-only", action="store_true")
    controlled = sub.add_parser("prepare-controlled")
    controlled.add_argument("--experiment-id", required=True)
    controlled.add_argument("--source-dataset")
    controlled.add_argument("--source-offset", type=int, default=0)
    controlled.add_argument("--layouts", type=int, default=12)
    controlled.add_argument("--seed", type=int, default=20260913)
    controlled.add_argument("--layers", default="3,10,15,18")
    controlled.add_argument("--site", choices=["resid_post", "attn_out", "mlp_out"], default="resid_post")
    controlled.add_argument("--last-k", type=int, default=1)
    controlled.add_argument("--seeds", default="42")
    controlled.add_argument("--max-new-tokens", type=int, default=2048)
    controlled.add_argument("--forced-only", action="store_true")
    extension = sub.add_parser("prepare-extension")
    extension.add_argument("--kind", choices=["input", "contextual-input", "fact", "reasoning", "reasoned-fact"], required=True)
    extension.add_argument("--source", required=True)
    extension.add_argument("--experiment-id", required=True)
    extension.add_argument("--layouts", type=int, default=12)
    extension.add_argument("--layers", default="10,15")
    extension.add_argument("--last-k", type=int, default=12)
    extension.add_argument("--cuts", default="32,128")
    extension.add_argument("--max-new-tokens", type=int, default=4096)
    extension.add_argument("--forced-only", action="store_true")
    factual = sub.add_parser("prepare-fact-readout")
    factual.add_argument("--source", required=True)
    factual.add_argument("--experiment-id", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection", type=Path, required=True)
    confirm = sub.add_parser("prepare-confirmation")
    confirm.add_argument("--freeze-id", required=True)
    confirm.add_argument("--candidate-id", required=True)
    confirm.add_argument("--experiment-id", required=True)
    plan_new = sub.add_parser("plan-co-primary")
    plan_new.add_argument("--decision", type=Path, required=True)
    plan_new.add_argument("--freeze-id", required=True)
    freeze_new = sub.add_parser("freeze-co-primary")
    freeze_new.add_argument("--plan", type=Path, required=True)
    freeze_new.add_argument("--review", type=Path, required=True)
    register_new = sub.add_parser("register-co-primary")
    register_new.add_argument("--freeze-id", required=True)
    supervised = sub.add_parser("run-confirmation")
    supervised.add_argument("--experiments", required=True, help="Ordered frozen primary/supplement experiment IDs")
    fit = sub.add_parser("fit-basis")
    fit.add_argument("--source", required=True)
    fit.add_argument("--condition", required=True)
    fit.add_argument("--family", choices=["wall", "key_possession", "door_status"], required=True)
    fit.add_argument("--layer", type=int, required=True)
    fit.add_argument("--site", default="resid_post", choices=["resid_post", "attn_out", "mlp_out"])
    fit.add_argument("--rank", type=int, required=True)
    fit.add_argument("--basis-id", required=True)
    subspace = sub.add_parser("prepare-subspace")
    subspace.add_argument("--source", required=True)
    subspace.add_argument("--basis-id", required=True)
    subspace.add_argument("--experiment-id", required=True)
    subspace.add_argument("--doses", default="0.5,1.0")
    subspace.add_argument("--max-new-tokens", type=int, default=4096)
    subspace.add_argument("--forced-only", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--max-jobs", type=int)
    run.add_argument("--retry-errors", action="store_true")
    queue = sub.add_parser("run-queue")
    queue.add_argument("--experiments", required=True, help="Ordered comma-separated registered experiment IDs")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.command == "init":
        result = init_campaign(campaign, REPO_ROOT, hours=args.hours, model_path=args.model_path)
    elif args.command == "status":
        result = status(campaign)
    else:
        from src.activation_causality import campaign as work
        if args.command == "prepare-sentence":
            spec = work.prepare_sentence(campaign, REPO_ROOT, experiment_id=args.experiment_id, cuts=args.cuts,
                                         seeds=tuple(int(s) for s in args.seeds.split(",")), max_new_tokens=args.max_new_tokens, natural=not args.forced_only)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "prepare-states":
            spec = work.prepare_states(campaign, REPO_ROOT, experiment_id=args.experiment_id, n_layouts=args.layouts,
                                       seed=args.seed, families=tuple(args.families.split(",")), max_new_tokens=args.max_new_tokens, natural=not args.forced_only)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "run":
            result = work.run_experiment(campaign, REPO_ROOT, args.experiment_id, max_jobs=args.max_jobs, retry_errors=args.retry_errors)
        elif args.command == "run-confirmation":
            from src.activation_causality.confirmation_supervisor import run_supervised_confirmation
            identifiers = args.experiments.split(",")
            if not identifiers or len(set(identifiers)) != len(identifiers) or any(not name for name in identifiers):
                raise ValueError("supervised queue requires unique nonempty experiment IDs")
            result = []
            for identifier in identifiers:
                outcome = run_supervised_confirmation(campaign, REPO_ROOT, identifier)
                result.append(outcome)
                if outcome.get("status") != "worker_exited" or outcome.get("incomplete"):
                    break
        elif args.command == "run-queue":
            identifiers = args.experiments.split(",")
            if not identifiers or len(set(identifiers)) != len(identifiers) or any(not name for name in identifiers):
                raise ValueError("queue requires unique nonempty experiment IDs")
            for identifier in identifiers:
                read_json(campaign / "specs" / f"{identifier}.json")
            result = []
            for identifier in identifiers:
                outcome = work.run_experiment(campaign, REPO_ROOT, identifier)
                result.append(outcome)
                if outcome.get("errors") or outcome.get("reason") == "deadline_reached":
                    break
        elif args.command == "prepare-controlled":
            from src.activation_causality.protocols import prepare_controlled
            spec = prepare_controlled(campaign, REPO_ROOT, experiment_id=args.experiment_id,
                                      source_dataset=args.source_dataset, n_layouts=args.layouts, seed=args.seed,
                                      source_offset=args.source_offset,
                                      layers=tuple(int(x) for x in args.layers.split(",")), site=args.site, last_k=args.last_k,
                                      seeds=tuple(int(x) for x in args.seeds.split(",")), natural=not args.forced_only,
                                      max_new_tokens=args.max_new_tokens)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "prepare-extension":
            from src.activation_causality.protocols import prepare_extension
            spec = prepare_extension(campaign, kind=args.kind, source=args.source, experiment_id=args.experiment_id,
                                     n_layouts=args.layouts, layers=tuple(int(x) for x in args.layers.split(",")),
                                     last_k=args.last_k, cuts=tuple(int(x) for x in args.cuts.split(",")),
                                     natural=not args.forced_only, max_new_tokens=args.max_new_tokens)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "freeze":
            from src.activation_causality.selection import freeze_selection
            result = freeze_selection(campaign, REPO_ROOT, read_json(args.selection))
        elif args.command == "plan-co-primary":
            from src.activation_causality.confirmation_planning import prepare_plan
            result = prepare_plan(campaign, REPO_ROOT, args.decision, freeze_id=args.freeze_id)
        elif args.command == "freeze-co-primary":
            from src.activation_causality.confirmation_freeze import freeze_confirmation_plan
            result = freeze_confirmation_plan(campaign, REPO_ROOT, args.plan, args.review)
        elif args.command == "register-co-primary":
            from src.activation_causality.confirmation_freeze import register_frozen_plan
            result = register_frozen_plan(campaign, REPO_ROOT, args.freeze_id)
        elif args.command == "prepare-fact-readout":
            from src.activation_causality.fact_readout_protocol import build_fact_readout_spec
            from src.activation_causality.registry import file_sha256, register_spec
            source_path = campaign / "specs" / f"{args.source}.json"
            if (campaign / "reviews" / f"{args.source}_retired.json").exists():
                raise ValueError("retired source cannot define factual-report diagnostics")
            spec = build_fact_readout_spec(read_json(source_path), args.experiment_id)
            spec["source_spec_sha256"] = file_sha256(source_path)
            register_spec(campaign, spec)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "prepare-confirmation":
            from src.activation_causality.protocols import prepare_confirmation
            spec = prepare_confirmation(campaign, REPO_ROOT, freeze_id=args.freeze_id, candidate_id=args.candidate_id, experiment_id=args.experiment_id)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "fit-basis":
            from src.activation_causality.subspace_protocol import fit_basis
            result = fit_basis(campaign, REPO_ROOT, args.source, args.condition, args.family, args.layer,
                               args.site, args.rank, args.basis_id)
        elif args.command == "prepare-subspace":
            from src.activation_causality.subspace_protocol import build_subspace_spec
            from src.activation_causality.registry import file_sha256, register_spec
            source_path = campaign / "specs" / f"{args.source}.json"
            if (campaign / "reviews" / f"{args.source}_retired.json").exists():
                raise ValueError("retired source cannot define a subspace evaluation")
            metadata_path = campaign / "bases" / f"{args.basis_id}.json"
            metadata = {**read_json(metadata_path), "basis_metadata_path": str(metadata_path),
                        "basis_metadata_sha256": file_sha256(metadata_path)}
            spec = build_subspace_spec(read_json(source_path), metadata, metadata["basis_path"], metadata["basis_file_sha256"],
                                       args.experiment_id, natural=not args.forced_only, cap=args.max_new_tokens,
                                       doses=tuple(float(x) for x in args.doses.split(",")))
            spec["source_spec_sha256"] = file_sha256(source_path)
            register_spec(campaign, spec)
            result = {"experiment_id": spec["experiment_id"], "jobs": len(spec["jobs"])}
        elif args.command == "analyze":
            from src.activation_causality.analysis import write_report
            if (campaign / "reviews" / f"{args.experiment_id}_retired.json").exists():
                raise ValueError("retired experiment is available only through its raw audit records")
            spec = read_json(campaign / "specs" / f"{args.experiment_id}.json")
            options = {}
            if spec["partition"] == "confirmation":
                from src.activation_causality.confirmation_contract import analysis_options
                frozen = read_json(campaign / "freezes" / f"{spec['freeze_id']}.json")
                options = analysis_options(frozen, spec)
            result_path = work.export_results(campaign, args.experiment_id)
            result = write_report(work.collect_results(campaign, args.experiment_id), campaign / "reports" / args.experiment_id, comparisons=spec["comparisons"], **options)
            result = {"export": str(result_path), "analysis": result}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
