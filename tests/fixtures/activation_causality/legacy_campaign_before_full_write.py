"""Prepare, execute and resume local activation-causality experiments."""
from __future__ import annotations

from collections import OrderedDict
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from typing import Any
import uuid

import numpy as np

from .registry import (
    SCHEMA_VERSION, append_event, atomic_json, file_sha256, fingerprint,
    immutable_json, now_utc, read_json, register_spec, seconds_remaining, worker_lock,
)

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
COUNTER = " Wait, I should double-check this. Looking at the grid again, the correct move here is {action}."
NEUTRAL = " Wait, I should double-check this. Looking at the grid again carefully before settling on a move here."


def make_job(experiment_id: str, **values: Any) -> dict:
    job = {"experiment_id": experiment_id, **values}
    return {"job_id": fingerprint(job)[:24], **job}


def prepare_sentence(campaign: Path, repo: Path, *, experiment_id: str, cuts: int = 12,
                     seeds: tuple[int, ...] = (42,), max_new_tokens: int = 2048,
                     natural: bool = True) -> dict:
    """Freeze historical cases before querying a model; reproducibility control only."""
    import pandas as pd
    injection_path = repo / "outputs/followups/commitment_gated_injection.parquet"
    prefix_path = repo / "outputs/prefixes.parquet"
    injection = pd.read_parquet(injection_path)
    injection = injection[injection["condition"] == "counter"].copy()
    # Round-robin six bins, then deterministic original-order tie break.
    injection["within_bin"] = injection.groupby("bin", observed=True).cumcount()
    injection = injection.sort_values(["within_bin", "bin", "prefix_id"]).head(cuts)
    prefixes = pd.read_parquet(prefix_path, columns=["prefix_id", "trajectory_id", "prompt_text", "analysis_prefix_text"])
    injection["prefix_id"] = injection["prefix_id"].astype(str)
    prefixes["prefix_id"] = prefixes["prefix_id"].astype(str)
    cases = injection.merge(prefixes, on="prefix_id", validate="one_to_one")
    jobs = []
    conditions = [
        ("baseline", "neutral", None, []),
        ("donor_text", "counter", None, []),
        ("identity", "neutral", "neutral", [10]),
        ("layer10", "neutral", "counter", [10]),
        ("layer15", "neutral", "counter", [15]),
        ("all_layers_full", "neutral", "counter", list(range(24))),
        ("all_layers_last12", "neutral", "counter", list(range(24))),
        ("reverse_layer10", "counter", "neutral", [10]),
        ("donor_identity_layer10", "counter", "counter", [10]),
    ]
    for row in cases.to_dict("records"):
        base = str(row["prompt_text"]).rstrip() + ANALYSIS_HEADER + ("" if pd.isna(row["analysis_prefix_text"]) else str(row["analysis_prefix_text"]))
        texts = {"neutral": base + NEUTRAL, "counter": base + COUNTER.format(action=row["target_action"])}
        for condition, recipient, donor, layers in conditions:
            for seed in seeds:
                jobs.append(make_job(
                    experiment_id, layout_id=str(row["trajectory_id"]), pair_id=str(row["prefix_id"]),
                    family="sentence_control", partition="historical", condition=condition,
                    target_action=str(row["target_action"]), original_action=str(row["top_before"]),
                    seed=int(seed), recipient_text=texts[recipient], donor_text=texts[donor] if donor else None,
                    shared_base_text=base, selection={"scope": "sentence", "last_k": 12 if condition.endswith("last12") else None},
                    intervention={"layers": layers, "site": "resid_post", "operation": "replace", "dose": 1.0},
                    natural=natural, max_new_tokens=max_new_tokens, temperature=0.7,
                    semantic_test=False, metadata={"bin": str(row["bin"]), "historical_prefix_id": str(row["prefix_id"]), "identity_not_independent_rescue": condition == "donor_identity_layer10"},
                ))
    spec = {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": "historical",
            "hypothesis": "Matched sentence-state transfer is sufficient for forced action control and may persist through native generation; explicit action suggestion is not a belief mechanism.",
            "sources": {str(injection_path): file_sha256(injection_path), str(prefix_path): file_sha256(prefix_path)},
            "jobs": jobs, "comparisons": [{"id": f"{experiment_id}_{c}", "experiment_id": experiment_id, "treatment": c, "control": "baseline", "partition": "historical"} for c in ["layer10", "layer15", "all_layers_full", "all_layers_last12"]]}
    register_spec(campaign, spec)
    return spec


def render_analysis_prompt(user_text: str) -> str:
    # Explicit, immutable Harmony template; no date-dependent template expansion.
    system = ("You are ChatGPT, a large language model trained by OpenAI.\n"
              "Reasoning: medium\n\n# Valid channels: analysis, commentary, final. "
              "Channel must be included for every message.")
    return ("<|start|>system<|message|>" + system + "<|end|><|start|>user<|message|>" + user_text
            + "<|end|><|start|>assistant" + ANALYSIS_HEADER)


def prepare_states(campaign: Path, repo: Path, *, experiment_id: str, n_layouts: int = 24,
                   seed: int = 20260912, partition: str = "discovery", families: tuple = ("wall", "key_possession", "door_status"),
                   natural: bool = True, max_new_tokens: int = 2048,
                   generation_seeds: tuple = (42,), freeze_id: str | None = None) -> dict:
    from .environments import generate_pairs
    if partition == "confirmation":
        if not freeze_id or not (campaign / "freezes" / f"{freeze_id}.json").exists():
            raise ValueError("confirmation must reference an existing frozen candidate selection")
        raise ValueError("use prepare-confirmation with a frozen candidate; unrestricted confirmation sweeps are forbidden")
    excluded_layouts = set()
    for dataset_path in sorted((campaign / "datasets").glob("*.json")):
        for old_pair in read_json(dataset_path).get("pairs", []):
            excluded_layouts.add(old_pair["layout_id"])
            excluded_layouts.update(old_pair.get("metadata", {}).get("variant_layout_ids", []))
    pairs = generate_pairs(n_layouts, seed, partition=partition, families=families,
                           historical_steps=repo / "outputs/canonical_steps.parquet",
                           excluded_layout_ids=excluded_layouts)
    dataset = {"partition": partition, "seed": seed, "pairs": pairs}
    immutable_json(campaign / "datasets" / f"{experiment_id}.json", dataset)
    jobs = []
    for pair in pairs:
        recipient = pair.get("recipient_prompt_text", pair.get("recipient", {}).get("prompt_text"))
        donor = pair.get("donor_prompt_text", pair.get("donor", {}).get("prompt_text"))
        counterfactual = pair.get("counterfactual_prompt_text", pair.get("counterfactual", {}).get("prompt_text", donor))
        if not isinstance(recipient, str) or not isinstance(donor, str):
            raise ValueError("generator must supply recipient_prompt_text and donor_prompt_text")
        targets = pair["counterfactual_optimal_actions"]
        if len(targets) != 1:
            raise ValueError("state discovery requires one predeclared counterfactual target")
        original = pair["recipient_optimal_actions"][0]
        texts = {"recipient": render_analysis_prompt(recipient), "donor": render_analysis_prompt(donor), "counterfactual": render_analysis_prompt(counterfactual)}
        conditions = [("baseline", "recipient", None, []), ("donor_text", "donor", None, []),
                      ("counterfactual_text", "counterfactual", None, []), ("identity", "recipient", "recipient", [10])]
        conditions += [(f"layer{layer}", "recipient", "donor", [layer]) for layer in (3, 10, 15, 18)]
        for condition, recipient_key, donor_key, layers in conditions:
            for generation_seed in generation_seeds:
                jobs.append(make_job(
                    experiment_id, layout_id=pair["layout_id"], pair_id=pair["pair_id"], family=pair["family"],
                    partition=partition, condition=condition, target_action=targets[0], original_action=original,
                    seed=int(generation_seed), recipient_text=texts[recipient_key], donor_text=texts[donor_key] if donor_key else None,
                    selection={"scope": "last_tokens", "last_k": 1}, intervention={"layers": layers, "site": "resid_post", "operation": "replace", "dose": 1.0},
                    natural=natural, max_new_tokens=max_new_tokens, temperature=0.7, semantic_test=True,
                    metadata={"intervention_time": "pre_reasoning", "pair": pair, "site_note": "last analysis-header token after task prompt, no generated action or forcing suffix"},
                ))
    comparisons = [{"id": f"{experiment_id}_layer{layer}", "experiment_id": experiment_id, "treatment": f"layer{layer}", "control": "baseline", "partition": partition} for layer in (3,10,15,18)]
    spec = {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "partition": partition, "freeze_id": freeze_id,
            "hypothesis": "Pre-reasoning residual state carries action-relevant wall, key or door information; donor replacement changes the recipient in the simulator-predicted direction.",
            "dataset_sha256": fingerprint(dataset), "jobs": jobs, "comparisons": comparisons}
    register_spec(campaign, spec)
    return spec


def runtime_fingerprint(repo: Path) -> dict:
    from .engine import ENGINE_DEPENDENCY_PATHS

    versions = {}
    for name in ("mlx", "mlx-lm", "numpy", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {str(p.relative_to(repo)): file_sha256(p) for p in sorted((repo / "src/activation_causality").glob("*.py"))}
    for relative in ENGINE_DEPENDENCY_PATHS:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing required engine dependency source: {relative}")
        sources[relative] = file_sha256(path)
    return {"versions": versions, "sources": sources}


def compile_patch(engine: Any, job: dict, capture_cache: OrderedDict, *,
                  loaded_model_proof: Any = None) -> tuple[list[int], list[dict]]:
    intervention = job["intervention"]
    subspace = intervention.get("operation") == "subspace"
    matched = intervention.get("projected_norm_matched_random_basis", False)
    audited = intervention.get("audit_projected_norm", False)
    for flag in ("projected_norm_matched_random_basis", "audit_projected_norm", "random_basis"):
        if flag in intervention and type(intervention[flag]) is not bool:
            raise ValueError(f"{flag} must be an explicit Boolean")
    if (matched or audited or intervention.get("random_basis")) and not subspace:
        raise ValueError("subspace control and audit flags require the subspace operation")
    if matched and not audited:
        raise ValueError("projected-norm random controls require the prospective applied-norm audit")
    if audited and (intervention.get("random_basis") or intervention.get("random_control") or intervention.get("zero_donor")):
        raise ValueError("audited task/matched-random writes cannot combine rank-only, zero or full-random controls")
    if audited and (type(intervention.get("projected_norm_relative_tolerance")) not in {int, float}
                    or intervention["projected_norm_relative_tolerance"] != 0.01):
        raise ValueError("audited subspace writes require the fixed prospective 0.01 relative tolerance")
    basis = None
    if subspace:
        from .basis_runtime import load_verified_basis
        if not isinstance(intervention.get("layers"), list) or len(intervention["layers"]) != 1:
            raise ValueError("one registered basis requires exactly one intervention layer")
        # Validate the engine instance and immutable basis before even encoding
        # inputs: encoding/capture can themselves trigger a lazy model load.
        basis, _ = load_verified_basis(intervention, job, engine, loaded_model_proof)

    def tokens(which: str) -> list[int]:
        supplied = job.get(which + "_token_ids")
        values = list(engine.encode(job[which + "_text"])) if supplied is None else supplied
        if not isinstance(values, list) or not values or any(type(i) is not int or i < 0 for i in values):
            raise ValueError("input token IDs must be nonempty nonnegative integer lists")
        return values

    def positions(values: Any, size: int) -> list[int]:
        if not isinstance(values, list) or not values or any(type(i) is not int or not 0 <= i < size for i in values):
            raise ValueError("explicit intervention positions must be nonempty integer indices in bounds")
        if len(set(values)) != len(values):
            raise ValueError("explicit intervention positions must be unique")
        return values

    recipient_ids = tokens("recipient")
    if not intervention["layers"]:
        return recipient_ids, []
    donor_ids = tokens("donor")
    selection = job["selection"]
    if selection["scope"] == "sentence":
        base_ids = engine.encode(job["shared_base_text"])
        start = len(base_ids)
        if recipient_ids[:start] != base_ids or donor_ids[:start] != base_ids or len(recipient_ids) != len(donor_ids):
            raise ValueError("donor/recipient sentence BPE boundaries do not align exactly")
        count = len(recipient_ids) - start
        if selection.get("last_k"):
            count = min(count, int(selection["last_k"]))
        recipient_pos = list(range(len(recipient_ids) - count, len(recipient_ids)))
        donor_pos = list(range(len(donor_ids) - count, len(donor_ids)))
    elif selection["scope"] == "last_tokens":
        if type(selection["last_k"]) is not int or selection["last_k"] < 1:
            raise ValueError("last_k must be a positive integer")
        count = min(int(selection["last_k"]), len(recipient_ids), len(donor_ids))
        recipient_pos = list(range(len(recipient_ids) - count, len(recipient_ids)))
        donor_pos = list(range(len(donor_ids) - count, len(donor_ids)))
    elif selection["scope"] == "explicit":
        recipient_pos = positions(selection.get("recipient_positions"), len(recipient_ids))
        donor_pos = positions(selection.get("donor_positions"), len(donor_ids))
        if len(recipient_pos) != len(donor_pos):
            raise ValueError("explicit donor/recipient spans must have equal lengths")
    else:
        raise ValueError("unsupported selection scope")
    if not recipient_pos:
        raise ValueError("empty intervention span")
    key = fingerprint({"ids": donor_ids, "positions": donor_pos, "layers": intervention["layers"], "site": intervention["site"]})
    if key not in capture_cache:
        capture_cache[key] = engine.capture(donor_ids, layers=intervention["layers"], positions=donor_pos, site=intervention["site"])
        while len(capture_cache) > 4:
            capture_cache.popitem(last=False)
    capture_cache.move_to_end(key)
    captures = capture_cache[key]
    originals = None
    if intervention.get("random_control") or intervention["operation"] == "add" or audited:
        recipient_key = fingerprint({"ids": recipient_ids, "positions": recipient_pos, "layers": intervention["layers"], "site": intervention["site"]})
        if recipient_key not in capture_cache:
            capture_cache[recipient_key] = engine.capture(recipient_ids, layers=intervention["layers"], positions=recipient_pos, site=intervention["site"])
        originals = capture_cache[recipient_key]
        capture_cache.move_to_end(recipient_key)
        while len(capture_cache) > 4:
            capture_cache.popitem(last=False)
    if intervention.get("random_control"):
        rng = np.random.default_rng(int(intervention.get("random_seed", job["seed"])))
        random_captures = {}
        for layer in intervention["layers"]:
            original = originals[layer]
            delta = captures[layer] - original
            direction = rng.standard_normal(delta.shape).astype(np.float32)
            direction *= np.linalg.norm(delta, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
            random_captures[layer] = original + direction
        captures = random_captures
    if intervention.get("zero_donor"):
        captures = {layer: np.zeros_like(value) for layer, value in captures.items()}
    if intervention["operation"] == "add":
        captures = {layer: value - originals[layer] for layer, value in captures.items()}
    patches = [{"layer": layer, "positions": recipient_pos, "donor": captures[layer],
                "site": intervention["site"], "operation": intervention["operation"], "dose": intervention["dose"]} for layer in intervention["layers"]]
    if subspace:
        task_basis = basis
        if intervention.get("random_basis"):
            rng = np.random.default_rng(int(intervention.get("random_seed", job["seed"])))
            basis, _ = np.linalg.qr(rng.standard_normal(basis.shape))
            basis = basis.astype(np.float32)
        for patch in patches:
            patch["basis"] = basis
            if audited:
                from .subspace_audit import prepare_projected_norm_audit
                original = originals[patch["layer"]]
                audit = prepare_projected_norm_audit(
                    original, captures[patch["layer"]], task_basis, patch["layer"], patch["site"],
                    patch["positions"], patch["dose"], intervention["projected_norm_relative_tolerance"],
                )
                patch["projected_norm_audit"] = audit
                patch["expected_recipient_float32_sha256"] = audit["expected_recipient_float32_sha256"]
                if matched:
                    from .subspace_controls import projected_norm_matched_random_control
                    control = projected_norm_matched_random_control(
                        original, captures[patch["layer"]], task_basis,
                        intervention.get("random_seed", job["seed"]),
                    )
                    patch.update(donor=control["donor"], basis=control["basis"],
                                 projected_norm_random_control_audit=control["audit"])
    return recipient_ids, patches


def run_experiment(campaign: Path, repo: Path, experiment_id: str, *, max_jobs: int | None = None,
                   retry_errors: bool = False, validate_first: bool = True) -> dict:
    from .engine import MLXInterventionEngine
    path = campaign / "specs" / f"{experiment_id}.json"
    if (campaign / "reviews" / f"{experiment_id}_retired.json").exists():
        raise ValueError("experiment is retired; preserve its records and register a corrected experiment")
    spec = read_json(path)
    # Cohort-level detection keeps execution identity unchanged when a resume
    # happens to contain only baseline rows or only one control condition.
    subspace_jobs = [job for job in spec["jobs"] if job.get("intervention", {}).get("operation") == "subspace"]
    requires_snapshot = bool(subspace_jobs) or spec.get("partition") == "confirmation"
    fact_jobs = [job for job in spec["jobs"] if "fact_probes" in job]
    for job in fact_jobs:
        probes = job["fact_probes"]
        if not isinstance(probes, list) or not probes or any(not isinstance(probe, dict) for probe in probes):
            raise ValueError("optional fact_probes must be a nonempty list of probe objects")
        identifiers = [probe.get("id") for probe in probes]
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("fact_probes require distinct nonempty IDs")
        for probe in probes:
            if probe.get("available", True) is False:
                if not probe.get("unavailable_reason"):
                    raise ValueError("unavailable fact_probes require an explicit reason")
            elif (not isinstance(probe.get("question"), str) or not probe["question"].strip()
                  or "<|" in probe["question"] or type(probe.get("recipient_value")) is not bool
                  or type(probe.get("counterfactual_value")) is not bool):
                raise ValueError("fact_probes require a plain question and Boolean physical truth labels")
    from .selection import validate_confirmation_ready
    validate_confirmation_ready(campaign, repo, spec)
    config = read_json(campaign / "campaign.json")
    spec_hash = file_sha256(path)
    runtime = runtime_fingerprint(repo)
    from .engine import engine_dependency_hashes
    dependency_sources = engine_dependency_hashes(runtime)
    model_path = Path(config["model_path"])
    model_files = {name: file_sha256(model_path / name) for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json") if (model_path / name).exists()}
    execution_sources = {name: digest for name, digest in runtime["sources"].items() if name.endswith(("engine.py", "campaign.py"))}
    execution_sources.update(dependency_sources)
    before_load_snapshot = None
    subspace_implementation = {}
    if requires_snapshot:
        from .model_identity import snapshot_identity
        before_load_snapshot = snapshot_identity(model_path)
        for name in ("basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit"):
            helper_path = f"src/activation_causality/{name}.py"
            digest = runtime["sources"].get(helper_path)
            if not digest:
                raise ValueError(f"snapshot-bound execution requires fingerprinted {name}.py")
            subspace_implementation[helper_path] = digest
        execution_sources.update(subspace_implementation)
    fact_readout_sha = None
    if fact_jobs:
        helper_path = "src/activation_causality/fact_readout.py"
        fact_readout_sha = runtime["sources"].get(helper_path)
        if not fact_readout_sha:
            raise ValueError("requested fact_probes require a fingerprinted fact_readout.py implementation")
        execution_sources[helper_path] = fact_readout_sha
    execution_identity = {"sources": execution_sources, "versions": runtime["versions"], "model": config["model_path"], "model_files": model_files, "head_mode": "full"}
    if requires_snapshot:
        execution_identity["model_snapshot_identity_sha256"] = before_load_snapshot["identity_sha256"]
    execution_key = fingerprint(execution_identity)
    results_dir = campaign / "results" / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for job in spec["jobs"]:
        result_path = results_dir / f"{job['job_id']}.json"
        if result_path.exists():
            previous = read_json(result_path)
            if previous["spec_sha256"] != spec_hash:
                raise ValueError("result/spec fingerprint mismatch")
            if previous.get("execution_key") != execution_key:
                raise ValueError("execution code, runtime or model changed since this experiment ran; register a new experiment ID")
            if previous["status"] == "complete" or not retry_errors:
                continue
        todo.append(job)
    if max_jobs is not None:
        todo = todo[:max_jobs]
    if not todo:
        return {"experiment_id": experiment_id, "new_results": 0, "reason": "no_pending_jobs"}
    if seconds_remaining(campaign) <= 0:
        return {"experiment_id": experiment_id, "new_results": 0, "reason": "deadline_reached"}
    engine_identity = {"engine_sha256": runtime["sources"].get("src/activation_causality/engine.py"),
                       "engine_dependency_sources": dependency_sources,
                       "versions": runtime["versions"], "model": config["model_path"],
                       "model_files": model_files, "head_mode": "full"}
    if requires_snapshot:
        engine_identity["model_snapshot_identity_sha256"] = before_load_snapshot["identity_sha256"]
    engine_key = fingerprint(engine_identity)
    gate_path = campaign / "reviews" / f"engine_validation_{engine_key[:16]}.json"
    completed = 0
    errors = 0
    with worker_lock(campaign):
        append_event(campaign, "worker_started", experiment_id=experiment_id, jobs=len(todo), engine_key=engine_key)
        engine = MLXInterventionEngine(config["model_path"])
        # Fix batching mode across validation, resumed workers and recorded jobs.
        # Computing the full head matches native prefill exactly on this BF16 model.
        engine._head_mode = "full"
        loaded_model_proof = None
        load_audit_path = None
        if requires_snapshot:
            from .basis_runtime import make_loaded_model_proof
            from .model_identity import snapshot_identity
            load_audit_path = (campaign / "reviews" / f"model_load_{experiment_id}_{uuid.uuid4().hex[:16]}.json").resolve()
            load_audit = {"format": "materialized_model_snapshot_audit_v1", "experiment_id": experiment_id,
                          "execution_key": execution_key, "before_load": before_load_snapshot,
                          "after_load": None, "passed": False, "started_at": now_utc()}
            try:
                # Construction is lazy. Hashing again before this explicit load
                # would fail to establish which bytes supplied the model arrays.
                engine._ensure_loaded()
                after_load_snapshot = snapshot_identity(model_path)
                load_audit["after_load"] = after_load_snapshot
                loaded_model_proof = make_loaded_model_proof(engine, before_load_snapshot, after_load_snapshot, runtime)
                load_audit.update(passed=True, activation_representation=loaded_model_proof.representation,
                                  materialized_engine_id=id(engine), engine_metadata=engine.metadata)
            except Exception as exc:
                load_audit.update(error_type=type(exc).__name__, error=str(exc), finished_at=now_utc())
                immutable_json(load_audit_path, load_audit)
                append_event(campaign, "model_load_snapshot_gate_failed", experiment_id=experiment_id,
                             path=str(load_audit_path), sha256=file_sha256(load_audit_path))
                raise RuntimeError(f"materialized model snapshot verification failed; see {load_audit_path}") from exc
            load_audit["finished_at"] = now_utc()
            immutable_json(load_audit_path, load_audit)
        if validate_first and not gate_path.exists():
            try:
                validation = engine.validate()
            except Exception as exc:
                validation = {"passed": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            atomic_json(gate_path, {"engine_key": engine_key, "time": now_utc(), "runtime": runtime, "validation": validation})
            if not validation.get("passed", False):
                append_event(campaign, "technical_gate_failed", experiment_id=experiment_id, path=str(gate_path))
                raise RuntimeError(f"engine correctness gate failed; see {gate_path}")
        elif gate_path.exists() and not read_json(gate_path)["validation"].get("passed", False):
            raise RuntimeError("a previous engine gate failed")
        elif not validate_first:
            raise ValueError("cannot bypass engine validation")
        if read_json(gate_path)["validation"].get("model", {}).get("head_mode") != "full":
            raise RuntimeError("validated head mode differs from the frozen full-head execution mode")
        capture_cache: OrderedDict = OrderedDict()
        for job in todo:
            if seconds_remaining(campaign) <= 0:
                break
            start = time.monotonic()
            common = {k: v for k, v in job.items() if k not in {"recipient_text", "donor_text", "shared_base_text"}}
            result = {**common, "spec_sha256": spec_hash, "engine_key": engine_key, "execution_key": execution_key, "inference_configuration": {"head_mode": "full"}, "started_at": now_utc()}
            if requires_snapshot:
                result.update(model_snapshot_identity_sha256=before_load_snapshot["identity_sha256"],
                              model_load_audit_path=str(load_audit_path), model_load_audit_sha256=file_sha256(load_audit_path),
                              subspace_implementation_sha256=subspace_implementation, inference_source_hashes=execution_sources,
                              compiled_patch_audits=[], applied_projected_norm_audits=[],
                              projected_norm_audit_status="pending" if job.get("intervention", {}).get("audit_projected_norm") else "not_requested")
            if "fact_probes" in job:
                result.update(fact_readout_status="pending", fact_reports=[],
                              fact_readout_implementation_sha256=fact_readout_sha,
                              fact_probes_sha256=fingerprint(job["fact_probes"]))
            try:
                if requires_snapshot:
                    token_ids, patches = compile_patch(engine, job, capture_cache, loaded_model_proof=loaded_model_proof)
                else:
                    token_ids, patches = compile_patch(engine, job, capture_cache)
                if any("projected_norm_audit" in patch for patch in patches):
                    result["compiled_patch_audits"] = [
                        {**{key: patch[key] for key in ("layer", "site", "positions", "dose")},
                         "projected_norm_audit": patch["projected_norm_audit"],
                         "random_control_audit": patch.get("projected_norm_random_control_audit")}
                        for patch in patches if "projected_norm_audit" in patch
                    ]
                    if any(item["random_control_audit"] is not None
                           and item["random_control_audit"].get("float32", {}).get("all_norms_match_within_tolerance") is not True
                           for item in result["compiled_patch_audits"]):
                        result["projected_norm_audit_status"] = "failed"
                        raise RuntimeError("projected-norm random control failed its recorded float32 preflight; no patched prefill executed")
                state = engine.prefill(token_ids, patches=patches)
                # Preserve actual model-dtype write measurements even when the
                # norm check fails and no forced/native endpoint is evaluated.
                result.update(patch_log=state.patch_log, n_input_tokens=len(token_ids), input_token_sha256=fingerprint(token_ids))
                if result.get("compiled_patch_audits"):
                    from .subspace_audit import validate_applied_norm_audit
                    result["applied_projected_norm_audits"] = [
                        validate_applied_norm_audit(item["projected_norm_audit"], state.patch_log)
                        for item in result["compiled_patch_audits"]
                    ]
                    passed = all(item.get("passed") is True for item in result["applied_projected_norm_audits"])
                    result["projected_norm_audit_status"] = "passed" if passed else "failed"
                    if not passed:
                        raise RuntimeError("actual subspace write failed the fixed recipient/norm audit; no endpoint scored")
                forced = engine.forced(state)
                if isinstance(forced, dict):
                    q = forced.get("q", forced.get("forced_q"))
                    if q is None and all(f"q_{a}" in forced for a in ACTIONS):
                        q = [forced[f"q_{a}"] for a in ACTIONS]
                else:
                    q = forced
                if hasattr(q, "tolist"):
                    q = q.tolist()
                if isinstance(q, dict):
                    q = [q[a] for a in ACTIONS]
                if q is None:
                    raise ValueError("engine forced result has no action distribution")
                result.update({"forced_q": dict(zip(ACTIONS, [float(x) for x in q], strict=True)),
                               "forced_metrics": forced if isinstance(forced, dict) else {},
                               "forced_top_action": ACTIONS[max(range(4), key=lambda i: q[i])],
                               "n_input_tokens": len(token_ids), "input_token_sha256": fingerprint(token_ids),
                               "patch_log": state.patch_log})
                if "fact_probes" in job:
                    try:
                        from .fact_readout import score_fact_reports
                        # The helper clones the original prefill independently
                        # per probe and proves that its cache/logits are intact.
                        reports = score_fact_reports(engine, state, job["fact_probes"])
                        if (not isinstance(reports, list) or any(not isinstance(report, dict) for report in reports)
                                or [report.get("id") for report in reports] != [probe["id"] for probe in job["fact_probes"]]
                                or any(report.get("status") not in {"scored", "unavailable"} for report in reports)):
                            raise RuntimeError("fact readout did not return one explicit result per requested probe")
                        if any(report.get("source_state_unchanged") is not True for report in reports):
                            raise RuntimeError("fact readout did not prove original-state preservation")
                        scored = sum(report["status"] == "scored" for report in reports)
                        result.update(fact_reports=reports, fact_readout_n_requested=len(reports),
                                      fact_readout_n_scored=scored, fact_readout_n_unavailable=len(reports) - scored,
                                      fact_readout_status="complete" if scored == len(reports) else "partial" if scored else "unavailable")
                    except Exception as exc:
                        result.update(fact_readout_status="error", fact_readout_error={
                            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                        raise
                if job["natural"]:
                    generated = engine.generate(state, seed=job["seed"], max_new_tokens=job["max_new_tokens"], temp=job["temperature"])
                    result.update(generated)
                else:
                    result.update({"native_action": None, "generation_status": "not_requested", "generated_tokens": 0})
                result["status"] = "complete"
                completed += 1
            except Exception as exc:
                result.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "native_action": None, "generation_status": "technical_error"})
                if isinstance(getattr(exc, "patch_audit", None), dict):
                    result["prefill_patch_failure"] = exc.patch_audit
                    result["projected_norm_audit_status"] = "failed"
                elif result.get("projected_norm_audit_status") == "pending":
                    result["projected_norm_audit_status"] = "not_run_due_to_upstream_error"
                if result.get("fact_readout_status") == "pending":
                    result.update(fact_readout_status="not_run_due_to_upstream_error", fact_readout_error={
                        "error_type": type(exc).__name__, "error": str(exc)})
                errors += 1
            result["seconds"] = time.monotonic() - start
            result["finished_at"] = now_utc()
            target = results_dir / f"{job['job_id']}.json"
            if target.exists():
                # Explicit retry preserves the previous failure as an immutable audit record.
                old = read_json(target)
                immutable_json(results_dir / "attempts" / f"{job['job_id']}_{fingerprint(old)[:12]}.json", old)
            atomic_json(target, result)
            print(json.dumps({"job": job["job_id"], "condition": job["condition"], "status": result["status"], "seconds": round(result["seconds"], 2), "forced": result.get("forced_top_action"), "native": result.get("native_action"), "generation_status": result.get("generation_status"), "error": result.get("error"),
                              **({"fact_readout_status": result["fact_readout_status"]} if "fact_readout_status" in result else {})}), flush=True)
            if errors >= 3:
                append_event(campaign, "worker_stopped_repeated_errors", experiment_id=experiment_id)
                break
        append_event(campaign, "worker_finished", experiment_id=experiment_id, completed=completed, errors=errors)
    return {"experiment_id": experiment_id, "new_results": completed, "errors": errors}


def collect_results(campaign: Path, experiment_id: str | None = None, *, include_retired: bool = False) -> list[dict]:
    root = campaign / "results"
    paths = sorted((root / experiment_id).glob("*.json")) if experiment_id else sorted(root.glob("*/*.json"))
    if not include_retired and experiment_id and (campaign / "reviews" / f"{experiment_id}_retired.json").exists():
        raise ValueError("retired experiment is excluded from scientific analysis; use include_retired only for audit")
    return [read_json(p) for p in paths if include_retired or not (campaign / "reviews" / f"{p.parent.name}_retired.json").exists()]


def export_results(campaign: Path, experiment_id: str) -> Path:
    records = collect_results(campaign, experiment_id)
    path = campaign / "reports" / f"{experiment_id}_results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    tmp.replace(path)
    return path
