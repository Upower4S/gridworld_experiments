"""Single fixed CPU-only confirmation preparation; never register or query.

Production has no tokenizer, generator, clock, population, cap, or seed override.
An immutable preparation fixes allocation before the one persisted sampling clock.
Failures remain terminal evidence; resuming cannot reallocate or reset attempts.
Outer promotion and a real independently audited forced preflight are prerequisites.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
from pathlib import Path
from typing import Any

from . import confirmation_freeze as freeze
from . import confirmation_sampling as sampling
from .confirmation_contract import remaining_inference_seconds
from .confirmation_rendering import build_primary, build_lexical_supplement, eligibility_decisions
from .environments import generate_pairs, load_historical_layout_ids
from .registry import file_sha256, fingerprint, immutable_json, now_utc

DATASET_SEED = 20260912
DECISION_REVIEW_SHA256 = "fe8830d8c2699f4bce0a2b4469f0c9a8243c2ed714dca09d34a38e2ea8fb29e3"
FORMAT = "final_confirmation_preparation_v1"
PREFLIGHT_AUDIT_SHA256 = "e0c66d2b290921761b9f8afdbce9c52b5940bbff04df0bd7d25a509711986b68"
PREFLIGHT_PROMOTION_SHA256 = "6bbcd0b0c197eeae77d456ba433be912a1fcf63053ec2bf76c576d014fb287c1"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _ref(path: Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": file_sha256(path)}


def _store(path: Path, value: Any) -> dict:
    if path.exists():
        previous, _ = freeze._read(path)
        _require(fingerprint(previous) == fingerprint(value), "immutable planning artifact differs: " + str(path))
    else:
        immutable_json(path, value)
    return _ref(path)


@contextmanager
def _planning_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "final_preparation_v1.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another final confirmation preparation owns this campaign") from exc
        yield


def _descriptors(decision: dict) -> dict[str, dict]:
    profiles = {item["profile_id"]: item["descriptor"] for item in decision["profiles"]}
    selected = decision["selected_scientific_profile_ids"]
    answer = {}
    for question in ("contextual_input", "fixed_family_subspace"):
        profile_id = selected[question]
        if profile_id is None:
            continue
        descriptor = deepcopy(profiles[profile_id])
        if question == "contextual_input":
            _require(descriptor["protocol_kind"] == "contextual_input", "selected question/protocol mismatch")
            cid = f"contextual_l{descriptor['layer']}"
        else:
            _require(descriptor["protocol_kind"] == "input_fact_subspace", "selected question/protocol mismatch")
            cid = {"wall": "wall", "key_possession": "key", "door_status": "door"}[descriptor["family"]] + "_subspace"
        _require(cid not in answer, "duplicate candidate descriptor identity")
        answer[cid] = descriptor
    _require(bool(answer), "no scientifically qualified candidate; no confirmation preparation")
    return answer


def _experiment_ids(freeze_id: str, descriptors: dict) -> dict:
    result = {}
    for cid, descriptor in descriptors.items():
        stem = "contextual" if descriptor["protocol_kind"] == "contextual_input" else cid
        result[cid] = {"primary": freeze_id + "_" + stem + "_primary",
                       "lexical_supplement": freeze_id + "_" + stem + "_alternate"}
    return result


def _preflight(campaign: Path, repo: Path, runtime: dict) -> dict:
    """Filename completeness precedes every raw read; audit binds the full cohort."""
    eid = "contextual_full_write_forced_preflight_v1"
    spec_path = campaign / "specs" / (eid + ".json")
    spec, spec_sha = freeze._read(spec_path)
    jobs = {job["job_id"]: job for job in spec["jobs"]}
    _require(spec.get("experiment_id") == eid and spec.get("technical_preflight_only") is True
             and spec.get("scientific_use") is False and len(jobs) == len(spec["jobs"]) == 72,
             "fixed complete technical preflight specification required")
    _require(not (campaign / "reviews" / (eid + "_retired.json")).exists(), "retired preflight")
    expected_paths = {campaign / "results" / eid / (job_id + ".json") for job_id in jobs}
    _require(set((campaign / "results" / eid).glob("*.json")) == expected_paths,
             "complete canonical preflight filenames required before any raw read")
    audit_path = campaign / "reviews/contextual_full_write_forced_preflight_independent_audit_v1.json"
    audit, digest = freeze._read(audit_path)
    _require(digest == PREFLIGHT_AUDIT_SHA256, "accepted independent preflight audit bytes changed")
    promotion_root = campaign / "source_snapshots/technical_preflight_promotion_20260912"
    promotion_path = promotion_root / "manifest.json"
    promotion, promotion_sha = freeze._read(promotion_path)
    _require(promotion_sha == PREFLIGHT_PROMOTION_SHA256
             and promotion.get("format") == "limited_technical_preflight_promotion_v1",
             "accepted preflight source promotion manifest changed")
    _require(promotion.get("registered_preflight_spec_sha256") == spec_sha,
             "preflight specification differs from accepted promotion")
    historical_sources = promotion["after_files"]
    _require(isinstance(historical_sources, dict) and bool(historical_sources), "preflight historical source inventory missing")
    _require(audit.get("format") == "full_write_preflight_independent_audit_v1" and audit.get("passed") is True
             and audit.get("experiment_id") == eid and audit.get("spec_sha256") == spec_sha
             and audit.get("n_results") == 72 and audit.get("technical_status_counts") == {"complete": 72}
             and audit.get("native_queries") == 0 and audit.get("full_write_audits_passed") == 48
             and audit.get("random_orientation_records") == 24 and audit.get("identity_rows_exact_zero") == 12,
             "completed independently audited full-write preflight required")
    expected = audit.get("input_files_sha256")
    _require(isinstance(expected, dict) and bool(expected), "preflight input hashes required")
    _require(all(expected.get(name) == wanted for name, wanted in historical_sources.items()),
             "preflight audit historical sources differ from accepted promotion")
    historical_resolution = {}
    for name, wanted in historical_sources.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "historical source path must be repository relative")
        archived = promotion_root / "after" / relative
        _require(archived.resolve(strict=True) == archived.absolute(), "preflight source archive must use canonical paths")
        _require(file_sha256(archived) == wanted, "accepted preflight source archive changed")
        historical_resolution[name] = {"path": str(archived), "sha256": wanted}
    hashes = {str(spec_path.relative_to(repo)): spec_sha}
    sources = audit.get("inference_source_hashes")
    _require(isinstance(sources, dict) and bool(sources)
             and all(runtime["sources"].get(name) == wanted for name, wanted in sources.items()),
             "preflight inference runtime differs from final planning runtime")
    required = {"src/activation_causality/" + name + ".py" for name in
                ("engine", "campaign", "full_write_audit", "subspace_audit", "basis_runtime", "model_identity")}
    from .engine import ENGINE_DEPENDENCY_PATHS
    _require(required | set(ENGINE_DEPENDENCY_PATHS) <= sources.keys(), "preflight omitted inference source dependencies")
    for source_name, source_sha in spec.get("sources", {}).items():
        source_path = Path(source_name).resolve(strict=True)
        _require(source_path.is_relative_to(repo), "preflight source definition outside repository")
        hashes[str(source_path.relative_to(repo))] = source_sha
    norm_count = orientation_count = identity_count = 0
    for path in sorted(expected_paths):
        row, row_sha = freeze._read(path)
        job = jobs[path.stem]
        _require(job["job_id"] == fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24], "noncanonical preflight job")
        for key, value in job.items():
            if key not in {"recipient_text", "donor_text", "shared_base_text"}:
                _require(key in row and fingerprint(row[key]) == fingerprint(value), "preflight frozen job field differs")
        _require(row.get("status") == "complete" and row.get("natural") is False
                 and row.get("native_action") is None and row.get("generation_status") == "not_requested"
                 and row.get("spec_sha256") == spec_sha and row.get("inference_source_hashes") == sources
                 and row.get("model_snapshot_identity_sha256") == audit.get("model_snapshot_identity_sha256"),
                 "preflight raw status/native/source/model evidence differs")
        if job["intervention"].get("audit_full_write_norm"):
            _require(row.get("full_write_norm_audit_status") == "passed", "preflight full-write norm gate failed")
            norm_count += 1
        orientation_count += len(row.get("random_orientation_audits", []))
        if job["condition"].endswith("_identity"):
            logs = row.get("patch_log", [])
            _require(len(logs) == 1 and all(value == 0 for value in logs[0]["delta_l2"]), "preflight identity is not exact zero")
            identity_count += 1
        load_path = Path(row["model_load_audit_path"]).resolve(strict=True)
        _require(load_path.is_relative_to(repo), "preflight load audit outside repository")
        hashes[str(load_path.relative_to(repo))] = row["model_load_audit_sha256"]
        hashes[str(path.relative_to(repo))] = row_sha
    _require((norm_count, orientation_count, identity_count) == (48, 24, 12), "preflight raw audit counts differ")
    _require(all(expected.get(name) == value for name, value in hashes.items()), "preflight audit omits a spec/raw input")
    observed = {promotion_path: promotion_sha}
    for relative, wanted in expected.items():
        _require(isinstance(relative, str) and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                 "preflight inputs must be repository relative")
        # Only the pinned source map is historical. Raw/spec/load/gate inputs
        # and archive entries themselves retain their exact direct bindings.
        if relative in historical_resolution:
            source = Path(historical_resolution[relative]["path"])
        else:
            source = (repo / relative).resolve(strict=True)
        _require(source.is_relative_to(repo) and file_sha256(source) == wanted, "preflight input changed")
        observed[source] = wanted
    for name, wanted in sources.items():
        current = repo / name
        _require(file_sha256(current) == wanted, "current preflight inference source bytes changed")
        observed[current] = wanted
    _require(set((campaign / "results" / eid).glob("*.json")) == expected_paths
             and file_sha256(audit_path) == digest
             and all(file_sha256(path) == wanted for path, wanted in observed.items()),
             "preflight inputs/inventory/audit changed during read")
    return {"path": str(audit_path.resolve()), "sha256": digest,
            "model_snapshot_identity_sha256": audit["model_snapshot_identity_sha256"],
            "historical_source_archive": {"manifest_path": str(promotion_path), "manifest_sha256": promotion_sha,
                "resolution_scope": "exact pinned after_files map only", "sources": historical_resolution},
            "current_inference_source_hashes": deepcopy(sources)}


def verify_preflight_prerequisite(campaign: Path, repo: Path, runtime: dict) -> dict:
    """Return audit/snapshot bindings plus archived and current source evidence.

    Freeze/readiness callers also compare this entire reference to the plan and
    its snapshot identity to their independently verified actual model identity.
    """
    return _preflight(Path(campaign).resolve(strict=True), Path(repo).resolve(strict=True), runtime)


def _inputs(campaign: Path, repo: Path, decision_path: Path) -> tuple[dict, dict, Any]:
    from .campaign import runtime_fingerprint
    from .basis_runtime import activation_representation
    from .confirmation_sources import read_closed_native_battery
    from .discovery_selection import evaluate_closed_discovery
    from .selection import _model_identity, _confirmation_tokenizer
    decision_record, decision_sha = freeze._read(decision_path)
    _require(decision_record.get("format") == "reviewable_discovery_decision_record_v1"
             and decision_record.get("confirmation_queries") == 0 and decision_record.get("freeze_created") is False,
             "prequery independently reviewed scientific decision required")
    review_path = campaign / "reviews/discovery_selection_record_independent_review_v1.json"
    review, review_sha = freeze._read(review_path)
    _require(review_sha == DECISION_REVIEW_SHA256 and review["decision_record_sha256"] == decision_sha
             and review["decision_fingerprint"] == fingerprint(decision_record["decision"]), "independent decision review mismatch")
    bindings = decision_record["audit_bindings"]
    battery = read_closed_native_battery(campaign, repo, bindings)
    actual = evaluate_closed_discovery(campaign, repo, bindings)
    _require(fingerprint(actual) == fingerprint(decision_record["decision"]), "decision differs from closed adopted-policy evaluation")
    runtime = runtime_fingerprint(repo)
    preflight = verify_preflight_prerequisite(campaign, repo, runtime)
    campaign_config, campaign_sha = freeze._read(campaign / "campaign.json")
    history = (repo / "outputs/canonical_steps.parquet").resolve(strict=True)
    history_sha = file_sha256(history)
    _, history_audit = load_historical_layout_ids(history)
    _require(file_sha256(history) == history_sha, "historical source changed during topology audit")
    model = _model_identity(campaign)
    _require(model["identity_sha256"] == preflight["model_snapshot_identity_sha256"], "preflight and actual model snapshot differ")
    tokenizer = _confirmation_tokenizer(campaign)
    prior = freeze.prior_exclusion_snapshot(campaign)
    fixed = {"campaign_sha256": campaign_sha, "campaign_deadline": campaign_config["deadline"],
             "inference_cutoff": freeze.inference_cutoff(campaign_config["deadline"]),
             "implementation": runtime["sources"], "activation_representation": activation_representation(runtime),
             "model_identity": model, "worker_entrypoint_sha256": file_sha256(repo / "scripts/30_activation_causality_campaign.py"),
             "scientific_selection": {"path": str(decision_path), "sha256": decision_sha},
             "scientific_selection_independent_review": {"path": str(review_path), "sha256": review_sha},
             "technical_preflight": preflight, "audit_bindings": bindings,
             "descriptors": _descriptors(actual), "prior": prior,
             "historical_exclusion": {"status": "checked", "source_path": str(history), "source_sha256": history_sha,
                                      "source_exclusion_provenance": history_audit},
             "budget_seconds_per_native_job": battery["minimum_budget_seconds_per_native_job"]}
    return fixed, battery, tokenizer


def _unchanged(repo: Path, fixed: dict) -> None:
    for name, digest in fixed["implementation"].items():
        _require(file_sha256(repo / name) == digest, "runtime changed during preparation: " + name)
    for key in ("scientific_selection", "scientific_selection_independent_review", "technical_preflight"):
        ref = fixed[key]
        _require(file_sha256(Path(ref["path"])) == ref["sha256"], "preparation prerequisite changed: " + key)
    _require(file_sha256(repo / "scripts/30_activation_causality_campaign.py") == fixed["worker_entrypoint_sha256"], "worker entrypoint changed")


def _allocation(fixed: dict, at: str) -> dict:
    return freeze.select_budget_allocation({cid: descriptor["protocol_kind"] for cid, descriptor in fixed["descriptors"].items()},
        fixed["budget_seconds_per_native_job"], remaining_inference_seconds(fixed["campaign_deadline"], fixed["inference_cutoff"], now=at))


def prepare_plan(campaign: Path, repo: Path, decision_record_path: Path, *, freeze_id: str,
                 dataset_seed: int = DATASET_SEED) -> dict:
    """Prepare one exact unregistered plan; terminal failure never resets sampling.

    Model weight bytes are hashed for authenticity, but no model is materialized.
    Production invocation is reserved for the coordinator after promoted-runtime
    review and successful independently audited real forced preflight.

    This is a one-way preparation-to-registration handoff. After any planned
    confirmation spec/dataset is registered, prior_exclusion_snapshot deliberately
    rejects another prepare call. Use the frozen registration/readiness/worker
    interfaces then; this API does not reopen a registered family or its sampler.
    """
    campaign, repo = Path(campaign).resolve(strict=True), Path(repo).resolve(strict=True)
    freeze_id = freeze._identifier(freeze_id)
    _require(type(dataset_seed) is int and dataset_seed == DATASET_SEED, "the single prospective dataset seed is fixed at 20260912")
    decision_path = Path(decision_record_path).resolve(strict=True)
    request = {"campaign": str(campaign), "repo": str(repo), "decision_record_path": str(decision_path),
               "freeze_id": freeze_id, "dataset_seed": dataset_seed}
    folder = campaign / "planning"
    with _planning_lock(folder):
        failure_path = folder / "final_preparation_failure_v1.json"
        if failure_path.exists():
            failed, _ = freeze._read(failure_path)
            _require(failed["request"] == request, "terminal preparation belongs to a different request")
            return failed
        phase = "authenticate_prerequisites"
        try:
            fixed, battery, tokenizer = _inputs(campaign, repo, decision_path)
            _unchanged(repo, fixed)
            preparation_path = folder / "final_preparation_v1.json"
            state_path = folder / "final_sampling_v1.json"
            if preparation_path.exists():
                preparation, _ = freeze._read(preparation_path)
                _require(preparation["request"] == request and fingerprint(preparation["fixed"]) == fingerprint(fixed), "refusing changed preparation identity or route")
            else:
                _require(not state_path.exists(), "sampling state exists without its original preparation; no clock reset")
                phase = "allocate_before_sampling"
                allocation_at = now_utc()
                allocation = _allocation(fixed, allocation_at)
                configs, bases, basis_manifests = {}, {}, {}
                for cid, descriptor in fixed["descriptors"].items():
                    config = freeze.candidate_configuration(repo, descriptor, battery["sources"][descriptor["source_experiment_id"]],
                        candidate_id=cid, n_layouts=allocation["n_layouts"], dataset_seed=dataset_seed, tokenizer=tokenizer,
                        joint=allocation["supplements"][cid])
                    configs[cid] = config
                    if config["protocol_kind"] == "input_fact_subspace":
                        from .confirmation_basis import validate_confirmation_basis
                        bases[cid], basis_manifests[cid] = validate_confirmation_basis(config, config["basis_metadata"],
                            model_snapshot_identity=fixed["model_identity"], activation_representation=fixed["activation_representation"])
                prior_ref = _store(folder / "prior_registered_exclusions_v1.json", fixed["prior"])
                builder_sha = file_sha256(repo / "src/activation_causality/confirmation_rendering.py")
                definitions = []
                for cid, config in configs.items():
                    families = config.get("families", [config.get("family")])
                    definitions.append({"candidate_id": cid, "source_experiment_id": config["source_experiment_id"],
                        "source_spec_sha256": config["source_spec_sha256"], "configuration": config,
                        "configuration_sha256": fingerprint(config), "eligibility_builder_sha256": builder_sha,
                        "eligibility_rule": freeze.eligibility_rule(repo, fixed["descriptors"][cid], joint=allocation["supplements"][cid]),
                        "families": families, "family_layout_counts": config.get("family_layout_counts", {config.get("family"): config["n_layouts"]}),
                        "require_joint_renderer_eligibility": allocation["supplements"][cid]})
                identity = {"policy_sha256": freeze.POLICY_SHA256, "campaign_deadline_utc": fixed["campaign_deadline"],
                    "inference_cutoff_utc": fixed["inference_cutoff"], "dataset_seed": dataset_seed,
                    "generator_sha256": file_sha256(repo / "src/activation_causality/environments.py"),
                    "generator_parameters": {"sizes": [7, 9], "wall_density": .23, "donor_mode": "matched", "require_unique_actions": True,
                                             "historical_steps": fixed["historical_exclusion"]["source_path"]},
                    "historical_exclusion": fixed["historical_exclusion"], "candidates": definitions,
                    "previously_used_layout_aliases": fixed["prior"]["layout_aliases"],
                    "previously_used_layout_aliases_sha256": fixed["prior"]["layout_aliases_sha256"],
                    "previously_used_layout_aliases_provenance": prior_ref}
                preparation = {"format": FORMAT, "request": request, "fixed": fixed, "allocation_at": allocation_at,
                    "allocation": allocation, "sampling_identity": identity, "candidate_configurations": configs,
                    "verified_basis_metadata": bases, "basis_manifests": basis_manifests,
                    "prior_exclusions": prior_ref, "experiment_ids": _experiment_ids(freeze_id, fixed["descriptors"])}
                _unchanged(repo, fixed)
                _store(preparation_path, preparation)
            phase = "initialize_or_resume_same_sampling_clock"
            _unchanged(repo, fixed)
            state = sampling.initialize_sampling_plan(state_path, preparation["sampling_identity"])
            actual_allocation = _allocation(fixed, state["started_utc"])
            _require(actual_allocation["n_layouts"] == preparation["allocation"]["n_layouts"]
                     and actual_allocation["supplements"] == preparation["allocation"]["supplements"],
                     "allocation boundary changed before actual sampling start; no reallocation permitted")
            bases = preparation["verified_basis_metadata"]
            def assess(candidate, pairs):
                _unchanged(repo, fixed)
                cid = candidate["candidate_id"]
                spec = build_primary(pairs, tokenizer, preparation["experiment_ids"][cid]["primary"],
                                     candidate["configuration"], freeze_id, basis_metadata=bases.get(cid))
                return {"eligibility_builder_sha256": candidate["eligibility_builder_sha256"], "decisions": eligibility_decisions(spec)}
            phase = "bounded_generation_and_eligibility"
            while state["status"] == "sampling":
                _unchanged(repo, fixed)
                state = sampling.advance_sampling_plan(state_path, generator=generate_pairs, eligibility_builder=assess,
                                                       expected_identity_sha256=state["identity_sha256"])
            _require(state["status"] == "complete", "sampling terminal incomplete: " + str(state.get("terminal_reason")))
            phase = "rebuild_exact_unregistered_cohorts"
            configs = preparation["candidate_configurations"]
            pairs = freeze._sampling_pairs(state, configs)
            experiments = {}
            for cid, config in configs.items():
                ids = preparation["experiment_ids"][cid]
                primary = build_primary(pairs[cid], tokenizer, ids["primary"], config, freeze_id, basis_metadata=bases.get(cid))
                cohorts = [("primary", primary, pairs[cid])]
                if preparation["allocation"]["supplements"][cid]:
                    alternate = build_lexical_supplement(primary, tokenizer, ids["lexical_supplement"], config, freeze_id, basis_metadata=bases.get(cid))
                    subset = list({job["pair_id"]: job["metadata"]["source_pair"] for job in alternate["jobs"]}.values())
                    cohorts.append(("lexical_supplement", alternate, subset))
                for role, spec, source_pairs in cohorts:
                    eid = ids[role]
                    spec_ref = _store(folder / (eid + "_spec.json"), spec)
                    dataset_ref = _store(folder / (eid + "_dataset.json"), {"experiment_id": eid, "partition": "confirmation",
                        "candidate_id": cid, "cohort_role": role, "pairs": source_pairs})
                    experiments[eid] = {"candidate_id": cid, "cohort_role": role, "spec_path": spec_ref["path"],
                        "spec_sha256": spec_ref["sha256"], "dataset_path": dataset_ref["path"], "dataset_sha256": dataset_ref["sha256"]}
            registry = {"schema_version": 1, "candidates": {cid: list(c["primary_comparisons"]) for cid, c in configs.items()},
                        "family_size": sum(len(c["primary_comparisons"]) for c in configs.values()),
                        "bindings": {cid: c["primary_comparisons"] for cid, c in configs.items()}}
            plan = {key: fixed[key] for key in ("campaign_sha256", "inference_cutoff", "implementation", "model_identity",
                    "activation_representation", "worker_entrypoint_sha256", "scientific_selection", "audit_bindings", "budget_seconds_per_native_job")}
            plan.update(confirmation_mode=freeze.MODE, schema_version=4, freeze_id=freeze_id,
                analysis_configuration=freeze.analysis_configuration(), candidate_configurations=configs,
                confirmation_primary_family=registry, planned_experiments=experiments,
                prior_exclusions=preparation["prior_exclusions"], sampling_state=_ref(state_path),
                basis_manifests=preparation["basis_manifests"], qualified_unexecuted_profiles={},
                supplements=preparation["allocation"]["supplements"], lexical_feasibility_omissions={},
                preparation=_ref(preparation_path), technical_preflight=fixed["technical_preflight"],
                scientific_selection_independent_review=fixed["scientific_selection_independent_review"])
            phase = "validate_complete_plan"
            _unchanged(repo, fixed)
            validation = freeze.validate_confirmation_plan(campaign, repo, plan)
            _unchanged(repo, fixed)
            plan_ref = _store(folder / "final_confirmation_plan_v1.json", plan)
            return {"status": "complete_unregistered", "plan": plan_ref, "sampling_state": _ref(state_path),
                    "planned_experiment_ids": list(experiments), "budget": validation["budget"],
                    "registered": False, "freeze_created": False, "model_queries": 0}
        except Exception as exc:
            evidence = {"format": "final_confirmation_preparation_failure_v1", "request": request,
                "status": "terminal_incomplete", "failed_at": now_utc(), "phase": phase,
                "error_type": type(exc).__name__, "error": str(exc), "model_queries": 0,
                "registered": False, "freeze_created": False, "reset_or_backfill_permitted": False}
            state_path = folder / "final_sampling_v1.json"
            if state_path.exists():
                evidence["sampling_state"] = _ref(state_path)
            _store(failure_path, evidence)
            return evidence
