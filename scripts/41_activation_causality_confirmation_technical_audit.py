#!/usr/bin/env python3
"""Complete-only saved-data technical checkpoint for one frozen confirmation cohort.

No model loading, inference, weight reads, resampling, or statistical decisions.
An incomplete filename inventory is returned without reading any result content.
Passing this automated recomputation still requires independent review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
TEXT_FIELDS = {"recipient_text", "donor_text", "shared_base_text"}
FORMAT = "complete_confirmation_technical_audit_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fp(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def same(left, right):
    return fp(left) == fp(right)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    canonical(value)  # Reject NaN, Infinity and overflowed exponents recursively.
    return value


def utc(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm", "torch"}:
            raise ImportError("model framework forbidden in technical audit: " + fullname)
        return None


class Inputs:
    def __init__(self, repo):
        self.repo, self.hashes, self.allowed_external = Path(repo).resolve(), {}, set()

    def raw(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.resolve(strict=True) == path, "noncanonical artifact path: " + str(path))
        require(path.is_relative_to(self.repo) or path in self.allowed_external, "unapproved external/model file read")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), "input changed while read")
        digest = hashlib.sha256(data).hexdigest()
        require(expected is None or expected == digest, "artifact SHA mismatch: " + str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, "input changed between reads")
        self.hashes[str(path)] = digest
        return data

    def read(self, path, expected=None):
        return strict(self.raw(path, expected))

    def finish(self):
        for name, digest in list(self.hashes.items()):
            self.raw(Path(name), digest)


def filename_inventory(campaign, spec):
    expected = {job["job_id"] + ".json" for job in spec["jobs"]}
    require(len(expected) == len(spec["jobs"]) > 0, "duplicate/empty frozen job inventory")
    actual = {path.name for path in (campaign / "results" / spec["experiment_id"]).glob("*.json")}
    return {"expected_files": sorted(expected), "saved_files": sorted(actual),
            "missing_files": sorted(expected - actual), "unexpected_files": sorted(actual - expected),
            "exact": expected == actual, "planned": len(expected), "saved": len(actual)}


def finite_norms(values, count):
    return (isinstance(values, list) and len(values) == count
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values))


def norm_interval(expected, actual):
    require(finite_norms(expected, len(expected)) and finite_norms(actual, len(expected)), "invalid norm vector")
    require(all(a == 0 if e == 0 else .99 * e <= a <= 1.01 * e
                for e, a in zip(expected, actual, strict=True)), "actual write violates fixed 1%/exact-zero norm gate")


def terminal_technical_failures(history):
    """Keep missing-only interruptions distinct; later success cannot erase failures."""
    failures = []
    for record in history:
        status = record["status"]
        require(status in {"worker_exited", "worker_failed", "supervisor_error", "cutoff_terminated", "cutoff_before_launch"},
                "unknown historical supervisor status")
        failed = status == "worker_failed" or (status == "supervisor_error"
            and record.get("error_type") not in {"SupervisorInterrupted", "KeyboardInterrupt"})
        if failed:
            failures.append({"lease_id": record["lease_id"], "status": status, "error_type": record.get("error_type")})
    return failures


class Audit:
    def __init__(self, repo, campaign, experiment_id):
        self.repo, self.campaign = Path(repo).resolve(), Path(campaign).resolve()
        require(self.campaign.is_relative_to(self.repo), "campaign must be inside repository")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id) is not None, "invalid experiment ID")
        self.eid, self.inputs = experiment_id, Inputs(self.repo)
        self.cache, self.basis_norm_hashes, self.issues = {}, {}, []

    def cached(self, path, digest=None):
        key = str(path)
        if key not in self.cache:
            self.cache[key] = self.inputs.read(path, digest)
        elif digest is not None:
            require(self.inputs.hashes[key] == digest, "same artifact bound to different hashes")
        return self.cache[key]

    def prepare(self):
        c, read = self.campaign, self.inputs.read
        self.spec_path = c / "specs" / (self.eid + ".json")
        self.spec = read(self.spec_path)
        s = self.spec
        require(s["experiment_id"] == self.eid and s["partition"] == "confirmation", "not an exact confirmation specification")
        self.freeze_path = c / "freezes" / (s["freeze_id"] + ".json")
        require(list((c / "freezes").glob("*.json")) == [self.freeze_path], "one immutable campaign-wide freeze required")
        self.frozen = read(self.freeze_path)
        f = self.frozen
        require(f["confirmation_mode"] == "co_primary_native_v4" and f["schema_version"] == 4, "explicit frozen version4 required")
        require(f["freeze_id"] == s["freeze_id"], "freeze identity mismatch")
        self.plan = read(Path(f["plan_path"]), f["plan_sha256"])
        require(Path(f["plan_path"]).is_relative_to(c / "planning"), "plan outside immutable planning directory")
        require(all(k in f and same(f[k], v) for k, v in self.plan.items()), "freeze shadows plan")
        self.review = read(Path(f["review_path"]), f["review_sha256"])
        require(self.review.get("approved") is True and self.review.get("freeze_id") == f["freeze_id"]
                and self.review.get("plan_sha256") == f["plan_sha256"]
                and self.review.get("plan_fingerprint") == fp(self.plan), "exact final plan is not approved")
        item = f["planned_experiments"][self.eid]
        self.spec_sha = self.inputs.hashes[str(self.spec_path)]
        require(self.spec_sha == item["spec_sha256"] and item["candidate_id"] == s["candidate_id"]
                and item["cohort_role"] == s["cohort_role"], "registered spec differs from frozen plan")
        require(s["protocol_kind"] in {"contextual_input", "input_fact_subspace"}
                and s["semantic_headline_eligible"] is False, "unsupported confirmation scope")
        require(not (c / "reviews" / (self.eid + "_retired.json")).exists(), "retired cohort")
        for j in s["jobs"]:
            require(fp({k: v for k, v in j.items() if k != "job_id"})[:24] == j["job_id"], "noncanonical job ID")
            require(j["experiment_id"] == self.eid and j["partition"] == "confirmation" and j["natural"] is True,
                    "non-native/unfrozen job in cohort")
            ids = j["recipient_token_ids"]
            require(isinstance(ids, list) and ids and all(type(v) is int and v >= 0 for v in ids), "invalid authoritative input IDs")
        self.inventory = filename_inventory(c, s)
        return self.inventory

    def terminal_readiness(self):
        """Supervision metadata only; a last row can precede terminal publication."""
        self.associated_lease_inventory = {}
        self.associated_launch_inventory = {}
        missing = []
        for path in sorted((self.campaign / "leases").glob("*.json")):
            lease = self.inputs.read(path)
            if lease.get("experiment_id") != self.eid:
                continue
            require(path.stem == lease.get("lease_id"), "lease filename/identity differs")
            terminal = self.campaign / "reviews/confirmation_supervision" / (path.stem + ".terminal.json")
            self.associated_lease_inventory[str(path)] = terminal.exists()
            if not terminal.exists():
                missing.append(str(terminal))
            else:
                record = self.inputs.read(terminal)
                if record.get("status") in {"worker_exited", "worker_failed"}:
                    require("launch_record_sha256" in record, "launched historical worker lacks declared launch hash")
                if "launch_record_sha256" in record:
                    launch = self.campaign / "reviews/confirmation_supervision" / (path.stem + ".launch.json")
                    require(record.get("launch_record_path") == str(launch), "historical declared launch path differs")
                    self.associated_launch_inventory[str(launch)] = launch.exists()
                    if not launch.exists():
                        missing.append(str(launch))
        require(self.associated_lease_inventory, "complete cohort has no supervisor lease records")
        return missing

    def supervision_history(self):
        """Retain even leases that ended before any row was saved."""
        records = []
        for path in self.associated_lease_inventory:
            lease_path = Path(path)
            lease = self.cached(lease_path, self.inputs.hashes[path])
            lid = lease["lease_id"]
            require(re.fullmatch(r"[0-9a-f]{32}", lid) is not None, "invalid historical lease ID")
            require(lease["freeze_sha256"] == self.inputs.hashes[str(self.freeze_path)]
                    and lease["spec_sha256"] == self.spec_sha and lease["campaign_sha256"] == self.frozen["campaign_sha256"]
                    and same(lease["source_hashes"], self.frozen["implementation"])
                    and lease["inference_cutoff"] == self.frozen["inference_cutoff"], "historical lease binding differs")
            terminal_path = self.campaign / "reviews/confirmation_supervision" / (lid + ".terminal.json")
            terminal = self.cached(terminal_path)
            require(terminal["format"] == "supervised_confirmation_terminal_v1" and terminal["lease_id"] == lid
                    and terminal["lease_path"] == path and terminal["lease_sha256"] == self.inputs.hashes[path]
                    and terminal["experiment_id"] == self.eid and terminal["terminal_record_path"] == str(terminal_path),
                    "historical terminal/lease association differs")
            if "launch_record_sha256" in terminal:
                launch_path = self.campaign / "reviews/confirmation_supervision" / (lid + ".launch.json")
                require(terminal["launch_record_path"] == str(launch_path), "historical launch path differs")
                launch = self.cached(launch_path, terminal["launch_record_sha256"])
                require(launch["format"] == "supervised_confirmation_launch_v1" and launch["lease_id"] == lid
                        and launch["lease_sha256"] == self.inputs.hashes[path]
                        and launch["supervisor_pid"] == lease["supervisor_pid"] and type(launch["worker_pid"]) is int
                        and launch["worker_pid"] > 1 and launch["worker_process_group"] == launch["worker_pid"],
                        "historical launch ownership differs")
                require(launch["command"] == [lease["interpreter_path"], "-I", "-B", lease["worker_entrypoint_path"],
                        "--campaign", str(self.campaign), "run", "--experiment-id", self.eid]
                        and utc(lease["created_at"]) <= utc(launch["launched_at"]) <= utc(terminal["finished_at"]),
                        "historical launch command/time differs")
            records.append(terminal)
        return records

    def authenticate(self):
        """Only called after complete filename gate; never materializes a model."""
        f, c, inputs = self.frozen, self.campaign, self.inputs
        self.config = inputs.read(c / "campaign.json", f["campaign_sha256"])
        require(utc(f["inference_cutoff"]) == utc(self.config["deadline"]) - timedelta(hours=6), "cutoff is not fixed deadline minus six hours")
        for name, digest in f["implementation"].items():
            inputs.raw(self.repo / name, digest)
        require(set(f["implementation"]) == {str(p.relative_to(self.repo)) for p in
                (self.repo / "src/activation_causality").glob("*.py")} |
                {"src/__init__.py", "src/conceptual_belief_space/__init__.py",
                 "src/conceptual_belief_space/config.py", "src/conceptual_belief_space/forced_suffix_probe.py"},
                "current runtime source inventory differs from freeze")
        inputs.raw(self.repo / "scripts/30_activation_causality_campaign.py", f["worker_entrypoint_sha256"])
        sys.path.insert(0, str(self.repo))
        if not any(isinstance(x, BlockModels) for x in sys.meta_path):
            sys.meta_path.insert(0, BlockModels())
        from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS, parse_native_action
        from src.activation_causality.basis_runtime import _snapshot, activation_representation
        from src.activation_causality.campaign import runtime_fingerprint
        from src.activation_causality.subspace_audit import array_fingerprint, validate_applied_norm_audit
        from src.activation_causality.full_write_audit import validate_full_write_norm_audit
        self.parser, self.array_hash = parse_native_action, array_fingerprint
        self.projected_validator, self.full_validator = validate_applied_norm_audit, validate_full_write_norm_audit
        self.runtime = runtime_fingerprint(self.repo)
        require(same(self.runtime["sources"], f["implementation"]), "current runtime differs")
        require(same(activation_representation(self.runtime), f["activation_representation"]), "capture representation differs")
        require(same(_snapshot(f["model_identity"]), f["model_identity"]), "invalid saved full snapshot identity")
        self.dependencies = {name: f["implementation"][name] for name in ENGINE_DEPENDENCY_PATHS}
        for item in f["planned_experiments"].values():
            for kind in ("spec", "dataset"):
                inputs.raw(Path(item[kind + "_path"]), item[kind + "_sha256"])
        for name in ("sampling_state", "preparation", "scientific_selection", "prior_exclusions"):
            ref = f[name]
            inputs.raw(Path(ref["path"]), ref["sha256"])
        for ref in f["audit_bindings"].values():
            inputs.raw(self.repo / ref["path"], ref["sha256"])
        for configuration in f["candidate_configurations"].values():
            eid = configuration["source_experiment_id"]
            inputs.raw(c / "specs" / (eid + ".json"), configuration["source_spec_sha256"])
            require(not (c / "reviews" / (eid + "_retired.json")).exists(), "candidate source retired")
        from src.activation_causality.confirmation_planning import verify_preflight_prerequisite
        require(same(verify_preflight_prerequisite(c, self.repo, self.runtime), f["technical_preflight"]), "real preflight prerequisite changed")
        # Its exact historical resolution map is preserved in the frozen ref;
        # the shared verifier authenticates historical sources and current inference code.
        inputs.raw(Path(f["technical_preflight"]["path"]), f["technical_preflight"]["sha256"])
        self._basis()
        self._decoder()
        require(len({j["condition"] for j in self.spec["jobs"]}) == (15 if self.spec["protocol_kind"] == "contextual_input" else 14),
                "frozen confirmation control inventory differs from reviewed protocol")
        for job in self.spec["jobs"]:
            require(job["target_action"] in ACTIONS and job["original_action"] in ACTIONS, "invalid simulator action labels")

    def _basis(self):
        import numpy as np
        from src.activation_causality.confirmation_basis import validate_confirmation_basis
        for cid, manifest in self.frozen["basis_manifests"].items():
            config = self.frozen["candidate_configurations"][cid]
            for path, digest in manifest["artifact_sha256"].items():
                self.inputs.raw(Path(path), digest)
            _, checked = validate_confirmation_basis(config, config["basis_metadata"],
                model_snapshot_identity=self.frozen["model_identity"], activation_representation=self.frozen["activation_representation"])
            require(same(checked, manifest), "fitted basis authenticity differs from freeze")
            basis = np.load(io.BytesIO(self.inputs.raw(Path(config["basis_path"]), config["basis_sha256"])), allow_pickle=False)
            self.basis_norm_hashes[cid] = self.array_hash(basis)

    def _decoder(self):
        """Direct fast-tokenizer decoding with exact saved tokenizer bytes; no HF/model load."""
        from tokenizers import Tokenizer
        snapshot = self.frozen["model_identity"]
        root = Path(snapshot["path"])
        paths = [(root / name).resolve() for name in ("tokenizer.json", "tokenizer_config.json")]
        self.inputs.allowed_external.update(paths)
        config = self.inputs.read(paths[1], snapshot["files"]["tokenizer_config.json"])
        require(config.get("clean_up_tokenization_spaces") is False
                and config.get("tokenizer_class") == "PreTrainedTokenizerFast", "unsupported tokenizer decoding contract")
        self.tokenizer = Tokenizer.from_str(self.inputs.raw(paths[0], snapshot["files"]["tokenizer.json"]).decode())
        self.terminal_ids = {}
        for marker, reason in (("<|return|>", "return"), ("<|endoftext|>", "eos"), ("<|call|>", "tool_call")):
            ids = self.tokenizer.encode(marker, add_special_tokens=False).ids
            require(len(ids) == 1, "terminal marker is not one token")
            self.terminal_ids[ids[0]] = reason

    def row_identity(self, row, job):
        require(row.get("spec_sha256") == self.spec_sha, "row/spec hash mismatch")
        for key, value in job.items():
            if key not in TEXT_FIELDS:
                require(key in row and same(row[key], value), "frozen row field differs: " + key)
        require(row.get("status") in {"complete", "error"}, "unfinished/malformed technical status")
        require(row.get("inference_configuration") == {"head_mode": "full"}, "wrong head mode")
        if row["status"] == "error":
            require(row.get("generation_status") == "technical_error" and row.get("native_action") is None,
                    "technical error claimed as native behavior")

    def runtime_evidence(self, row, job):
        f, c = self.frozen, self.campaign
        sources = {name: digest for name, digest in f["implementation"].items() if name.endswith(("engine.py", "campaign.py"))}
        sources.update(self.dependencies)
        sub = {f"src/activation_causality/{n}.py": f["implementation"][f"src/activation_causality/{n}.py"]
               for n in ("basis_runtime", "model_identity", "localization", "subspace_controls", "subspace_audit")}
        sources.update(sub)
        require(same(row.get("subspace_implementation_sha256"), sub), "snapshot helper source map differs")
        if self.spec["protocol_kind"] == "contextual_input":
            full = {f"src/activation_causality/{n}.py": f["implementation"][f"src/activation_causality/{n}.py"]
                    for n in ("full_write_audit", "subspace_audit", "basis_runtime", "model_identity")}
            sources.update(full)
            require(same(row.get("full_write_implementation_sha256"), full), "full-write helper source map differs")
        sources.update({f"src/activation_causality/{n}.py": f["implementation"][f"src/activation_causality/{n}.py"]
                        for n in ("confirmation_supervisor", "confirmation_contract")})
        require(same(row.get("inference_source_hashes"), sources), "inference source closure differs")
        snapshot = f["model_identity"]
        require(row.get("model_snapshot_identity_sha256") == snapshot["identity_sha256"], "row snapshot differs")
        common = {"versions": self.runtime["versions"], "model": self.config["model_path"], "head_mode": "full",
                  "model_files": {n: snapshot["files"][n] for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")},
                  "model_snapshot_identity_sha256": snapshot["identity_sha256"]}
        contract = {"confirmation_mode": f["confirmation_mode"], "freeze_id": f["freeze_id"],
                    "freeze_sha256": self.inputs.hashes[str(self.freeze_path)], "inference_cutoff": f["inference_cutoff"]}
        require(row["execution_key"] == fp({"sources": sources, **common, "confirmation_execution_contract": contract}), "execution key does not reconstruct")
        require(row["engine_key"] == fp({"engine_sha256": f["implementation"]["src/activation_causality/engine.py"],
                "engine_dependency_sources": self.dependencies, **common}), "engine key does not reconstruct")
        load_path = Path(row["model_load_audit_path"])
        require(load_path.parent == c / "reviews", "load audit outside reviews")
        load = self.cached(load_path, row["model_load_audit_sha256"])
        require(load.get("format") == "materialized_model_snapshot_audit_v1" and load.get("passed") is True
                and load["experiment_id"] == self.eid and load["execution_key"] == row["execution_key"]
                and same(load["before_load"], snapshot) and same(load["after_load"], snapshot)
                and same(load["activation_representation"], f["activation_representation"]), "actual model-load proof differs")
        require(type(load["materialized_engine_id"]) is int and load["engine_metadata"]["loaded"] is True
                and load["engine_metadata"]["head_mode"] == "full" and load["engine_metadata"]["hidden_dim"] == 2880,
                "missing materialized full-head evidence")
        gate = self.cached(c / "reviews" / ("engine_validation_" + row["engine_key"][:16] + ".json"))
        from src.activation_causality.basis_runtime import activation_representation
        require(gate["engine_key"] == row["engine_key"] and gate["validation"]["passed"] is True
                and gate["validation"]["model"]["head_mode"] == "full"
                and same(activation_representation(gate["runtime"]), f["activation_representation"])
                and same(gate["runtime"]["versions"], self.runtime["versions"]), "full-head engine gate differs")
        # Gate runtime may predate supervisor-only promotion: engine key and
        # conservative capture dependency map, not unrelated gate source files, bind reuse.
        self.lease_evidence(row, job, load)

    def lease_evidence(self, row, job, load):
        from src.activation_causality.confirmation_supervisor import _interpreter_binding
        c, f, lid = self.campaign, self.frozen, row["confirmation_lease_id"]
        require(re.fullmatch(r"[0-9a-f]{32}", lid) is not None, "invalid lease ID")
        path = c / "leases" / (lid + ".json")
        require(row["confirmation_lease_path"] == str(path), "noncanonical row lease path")
        lease = self.cached(path, row["confirmation_lease_sha256"])
        if not hasattr(self, "interpreter_binding"):
            self.interpreter_binding = _interpreter_binding()
            for field, digest_field in (("interpreter_resolved_path", "interpreter_binary_sha256"),
                                        ("interpreter_pyvenv_cfg_path", "interpreter_pyvenv_cfg_sha256")):
                if self.interpreter_binding[field]:
                    artifact = Path(self.interpreter_binding[field])
                    self.inputs.allowed_external.add(artifact)
                    self.inputs.raw(artifact, self.interpreter_binding[digest_field])
        expected = {"format": "supervised_confirmation_execution_lease_v1", "lease_id": lid,
            "experiment_id": self.eid, "freeze_id": f["freeze_id"], "candidate_id": self.spec["candidate_id"],
            "cohort_role": self.spec["cohort_role"], "repo_path": str(self.repo), "campaign_path": str(c),
            "campaign_sha256": f["campaign_sha256"], "spec_path": str(self.spec_path), "spec_sha256": self.spec_sha,
            "freeze_path": str(self.freeze_path), "freeze_sha256": self.inputs.hashes[str(self.freeze_path)],
            "inference_cutoff": f["inference_cutoff"], "worker_entrypoint_path": str(self.repo / "scripts/30_activation_causality_campaign.py"),
            "worker_entrypoint_sha256": f["worker_entrypoint_sha256"], "source_hashes": f["implementation"], **self.interpreter_binding}
        require(all(k in lease and same(lease[k], v) for k, v in expected.items()), "lease frozen/interpreter binding differs")
        require(row["confirmation_inference_cutoff"] == f["inference_cutoff"], "row cutoff differs")
        launch_path = c / "reviews/confirmation_supervision" / (lid + ".launch.json")
        terminal_path = c / "reviews/confirmation_supervision" / (lid + ".terminal.json")
        terminal = self.cached(terminal_path)
        launch = self.cached(launch_path, terminal["launch_record_sha256"])
        require(launch["format"] == "supervised_confirmation_launch_v1" and launch["lease_id"] == lid
                and launch["lease_sha256"] == row["confirmation_lease_sha256"]
                and launch["supervisor_pid"] == lease["supervisor_pid"] and type(launch["worker_pid"]) is int
                and launch["worker_pid"] > 1 and launch["worker_process_group"] == launch["worker_pid"], "launch ownership differs")
        require(launch["command"] == [lease["interpreter_path"], "-I", "-B", lease["worker_entrypoint_path"],
                "--campaign", str(c), "run", "--experiment-id", self.eid], "launch command differs")
        require(terminal["format"] == "supervised_confirmation_terminal_v1" and terminal["lease_id"] == lid
                and terminal["lease_path"] == str(path) and terminal["lease_sha256"] == row["confirmation_lease_sha256"]
                and terminal["experiment_id"] == self.eid and terminal["inference_cutoff"] == f["inference_cutoff"]
                and terminal["launch_record_path"] == str(launch_path) and terminal["terminal_record_path"] == str(terminal_path),
                "terminal/lease association differs")
        logfile = c / "logs" / ("confirmation_supervision_" + lid + ".log")
        require(terminal["log_path"] == str(logfile), "terminal log path differs")
        if str(logfile) not in self.inputs.hashes:
            self.inputs.raw(logfile, terminal["log_sha256"])
        require(utc(lease["created_at"]) <= utc(launch["launched_at"]) <= utc(load["started_at"])
                <= utc(load["finished_at"]) <= utc(row["started_at"]) <= utc(row["finished_at"])
                <= utc(terminal["finished_at"]), "load/job/lease time ordering differs")
        require(utc(row["finished_at"]) < utc(f["inference_cutoff"]), "saved job finishes outside frozen inference cutoff")
        self.leases[lid] = terminal

    def endpoints(self, row, job):
        require(row["input_token_sha256"] == fp(job["recipient_token_ids"])
                and row["n_input_tokens"] == len(job["recipient_token_ids"]), "authoritative input token hash differs")
        q, metrics = row["forced_q"], row["forced_metrics"]
        require(set(q) == set(ACTIONS) and finite_norms(list(q.values()), 4) and max(q.values()) <= 1
                and abs(math.fsum(q.values()) - 1) < 1e-10, "invalid forced conditional distribution")
        top = max(ACTIONS, key=lambda a: q[a])
        require(row["forced_top_action"] == metrics["top_action"] == top
                and all(q[a] == metrics["q_" + a] for a in ACTIONS), "forced diagnostic inconsistent")
        logits = [metrics["logit_" + a] for a in ACTIONS]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in logits), "invalid candidate logits")
        weights = [math.exp(v - max(logits)) for v in logits]
        require(all(abs(q[a] - w / math.fsum(weights)) <= 1e-12 for a, w in zip(ACTIONS, weights, strict=True)), "forced logits/conditional probabilities differ")
        require(type(metrics["candidate_mass"]) in (int, float) and 0 <= metrics["candidate_mass"] <= 1,
                "invalid full-vocabulary candidate mass")
        from src.conceptual_belief_space.forced_suffix_probe import FORCED_SUFFIX
        require(metrics["endpoint"] == "forced" and metrics["suffix"] == FORCED_SUFFIX, "forced suffix endpoint differs")
        ids = row["generated_token_ids"]
        require(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids)
                and 0 < len(ids) == row["generated_tokens"] == row["n_new_tokens"] <= job["max_new_tokens"], "invalid native IDs/count")
        require(self.tokenizer.decode(ids, skip_special_tokens=False) == row["generated_text"], "authoritative native tokens do not decode to saved text")
        parsed = self.parser(row["generated_text"])
        require(all(same(row[k], v) for k, v in parsed.items()), "native final parser differs")
        reason = row["stop_reason"]
        require(reason in {"return", "eos", "tool_call", "token_limit"}, "unknown native terminal reason")
        require(not any(token in self.terminal_ids for token in ids[:-1]), "native generation continued after terminal token")
        if reason == "token_limit":
            require(len(ids) == job["max_new_tokens"] and ids[-1] not in self.terminal_ids, "incorrect token-cap termination")
        else:
            require(self.terminal_ids.get(ids[-1]) == reason, "native terminal token/reason differ")
        ended = reason in {"return", "eos"}
        completed = ended and parsed["parse_status"] == "valid"
        status = "complete" if completed else ("truncated" if reason == "token_limit" else "malformed")
        require(row["generation_status"] == status and row["native_action"] == (parsed["parsed_action"] if completed else None)
                and row["action"] == row["native_action"] and row["completed"] is completed
                and row["ended_naturally"] is ended and row["truncated"] is (reason == "token_limit")
                and row["endpoint"] == "native_final", "native status/endpoint differs")

    def patch_evidence(self, row, job):
        import numpy as np
        intervention, logs = job["intervention"], row["patch_log"]
        positions = job["selection"]["recipient_positions"]
        require(len(logs) == len(intervention["layers"]), "wrong patch count")
        rank = job["metadata"]["rank"] if intervention["operation"] == "subspace" else None
        for layer, log in zip(intervention["layers"], logs, strict=True):
            for key, value in {"layer": layer, "site": intervention["site"], "positions": positions,
                               "operation": intervention["operation"], "dose": float(intervention["dose"]), "rank": rank}.items():
                require(same(log.get(key), value), "applied patch geometry differs: " + key)
            # These are the vectors emitted by the frozen engine. It does not
            # record an after-state norm; do not require or fabricate one.
            require(all(finite_norms(log.get(key), len(positions)) for key in ("before_l2", "delta_l2", "relative_delta_l2")), "invalid logged norm vectors")
        full = intervention.get("audit_full_write_norm", False)
        projected = intervention.get("audit_projected_norm", False)
        for needed, prepared_key, applied_key, status_key, validator in (
            (full, "compiled_full_write_audits", "applied_full_write_audits", "full_write_norm_audit_status", self.full_validator),
            (projected, "compiled_patch_audits", "applied_projected_norm_audits", "projected_norm_audit_status", self.projected_validator),
        ):
            require(len(row.get(prepared_key, [])) == len(row.get(applied_key, [])) == int(needed), "unexpected/missing declared norm audit")
            require(row.get(status_key, "not_requested") == ("passed" if needed else "not_requested"), "norm status differs from declaration")
            if not needed:
                continue
            container = row[prepared_key][0]
            pre = container if prepared_key == "compiled_full_write_audits" else container["projected_norm_audit"]
            require(pre["layer"] == intervention["layers"][0] and pre["site"] == intervention["site"]
                    and pre["positions"] == positions and same(pre["dose"], float(intervention["dose"]))
                    and pre["relative_tolerance"] == .01 and pre["absolute_tolerance"] == 0, "audit geometry/tolerance drift")
            applied = validator(pre, logs)
            require(same(applied, row[applied_key][0]) and applied["passed"] is True, "actual norm audit replay failed")
            norm_interval(pre["expected_delta_l2"], logs[0]["delta_l2"])
            require(pre["expected_recipient_float32_sha256"] == logs[0]["recipient_float32_sha256"], "pre-write recipient fingerprint differs")
            if prepared_key == "compiled_full_write_audits":
                from src.activation_causality.campaign import _full_write_contract
                require(_full_write_contract(job), "invalid inline source contract")
                require(same(pre["source_identity"], intervention["full_write_source_identity"])
                        and pre["donor_positions"] == job["selection"]["donor_positions"] and pre["hidden"] == 2880,
                        "full-write original reference/source differs")
                expected_donor = (row["random_orientation_audits"][0]["compiled_donor_float32_sha256"]
                                  if intervention.get("random_control") else pre["original_reference_donor_float32_sha256"])
                require(pre.get("compiled_donor_float32_sha256") == expected_donor
                        and isinstance(expected_donor, str) and re.fullmatch(r"[0-9a-f]{64}", expected_donor)
                        and pre.get("inline_definition_sha256") == fp(pre["source_identity"]),
                        "full-write compiled donor/source binding differs")
            else:
                require(pre["original_task_basis_float32_sha256"] == self.basis_norm_hashes[self.spec["candidate_id"]]
                        and pre["rank"] == rank, "projected audit uses another basis/rank")
                require(all(e == abs(intervention["dose"]) * t for e, t in zip(pre["expected_delta_l2"], pre["expected_task_projected_l2"], strict=True)), "dose differs from original projection norm")
                helper = container["random_control_audit"]
                if intervention.get("projected_norm_matched_random_basis"):
                    require(helper["float32"]["all_norms_match_within_tolerance"] is True
                            and helper["seed"] == intervention["random_seed"] and helper["rank"] == rank, "projected random preparation failed")
                else:
                    require(helper is None, "unexpected projected random helper")
        recorded = intervention.get("record_random_orientation", False)
        require(len(row.get("random_orientation_audits", [])) == int(recorded), "orientation inventory differs")
        if recorded:
            from src.activation_causality.campaign import _full_write_contract
            require(_full_write_contract(job), "invalid orientation-only source binding")
            audit = row["random_orientation_audits"][0]
            require(audit["shape"] == [len(positions), 2880] and audit["seed"] == intervention["random_seed"]
                    and audit["numpy_version"] == self.runtime["versions"]["numpy"]
                    and same(audit["source_identity"], intervention["full_write_source_identity"]), "random support/seed/source differs")
            draw = np.random.default_rng(intervention["random_seed"]).standard_normal(tuple(audit["shape"])).astype(np.float32)
            require(audit["raw_gaussian_float32_sha256"] == self.array_hash(draw), "exact seeded random orientation differs")
            if full:
                require(audit["passed"] is True, "new audited random preparation failed")
            # Old matched random is record-only; never impose a new norm gate.

    def controls(self, rows):
        groups = defaultdict(dict)
        jobs = {(j["pair_id"], j["seed"], j["condition"]): j for j in self.spec["jobs"]}
        for row in rows:
            group = groups[row["pair_id"], row["seed"]]
            require(row["condition"] not in group, "duplicate pair/seed/condition")
            group[row["condition"]] = row
        expected = {j["condition"] for j in self.spec["jobs"]}
        checks = []
        for (pair, seed), group in groups.items():
            require(set(group) == expected, "control cohort is incomplete")
            contextual = self.spec["protocol_kind"] == "contextual_input"
            if contextual:
                prefix = f"layer{self.frozen['candidate_configurations'][self.spec['candidate_id']]['layer']}_"
                identity_names = (prefix + "identity", prefix + "restoration_equivalent")
                natural, random, old = (group[prefix + x] for x in ("natural", "natural_norm_random", "random"))
                # The random write uses a synthetic donor while retaining the
                # natural contrast as its norm reference. Each compiled donor
                # is bound separately in patch_evidence.
                references = [{k: v for k, v in row["compiled_full_write_audits"][0].items()
                               if k != "compiled_donor_float32_sha256"} for row in (natural, random)]
                require(same(*references), "natural/random original norm reference differs")
                new_o, old_o = random["random_orientation_audits"][0], old["random_orientation_audits"][0]
                pre = natural["compiled_full_write_audits"][0]
                require(new_o["raw_gaussian_float32_sha256"] == old_o["raw_gaussian_float32_sha256"]
                        and new_o["recipient_float32_sha256"] == old_o["recipient_float32_sha256"] == pre["expected_recipient_float32_sha256"]
                        and new_o["reference_donor_float32_sha256"] == pre["original_reference_donor_float32_sha256"], "paired orientation/recipient/natural reference differs")
                matched_pairs = [(natural, random)]
            else:
                identity_names = ("subspace_identity", "restoration_equivalent")
                matched_pairs = [(group["subspace_forward_" + d], group["projected_norm_matched_random_subspace_" + d]) for d in ("0p5", "1")]
                for forward, random in matched_pairs:
                    require(same(forward["compiled_patch_audits"][0]["projected_norm_audit"], random["compiled_patch_audits"][0]["projected_norm_audit"]), "task/random original projected reference differs")
            for name in identity_names:
                baseline_name = "counterfactual_text" if contextual and name.endswith("restoration_equivalent") else "baseline"
                frozen_identity = jobs[pair, seed, name]
                frozen_baseline = jobs[pair, seed, baseline_name]
                require(frozen_identity["metadata"]["paired_baseline_condition"] == baseline_name
                        and same(frozen_identity["recipient_token_ids"], frozen_baseline["recipient_token_ids"]),
                        "identity frozen baseline/input binding differs")
                identity, baseline = group[name], group[baseline_name]
                require(all(v == 0 for log in identity["patch_log"] for v in log["delta_l2"]), "identity is not an exact zero write")
                for key in ("forced_q", "forced_metrics", "generated_token_ids", "generated_text", "generation_status", "stop_reason", "native_action"):
                    require(same(identity[key], baseline[key]), "identity/restoration whole-continuation mismatch: " + key)
                checks.append({"pair_id": pair, "seed": seed, "condition": name, "baseline_condition": baseline_name,
                               "exact_zero_and_full_native_replay": True})
            for forward, random in matched_pairs:
                for a, b in zip(forward["patch_log"][0]["delta_l2"], random["patch_log"][0]["delta_l2"], strict=True):
                    require(a == b == 0 if a == 0 else .99 / 1.01 <= b / a <= 1.01 / .99, "paired actual norm ratio outside fixed bounds")
        return checks

    def audit(self, check_ready=False):
        inventory = self.prepare()
        result = {"format": FORMAT, "experiment_id": self.eid, "technical_passed": False,
                  "readiness": inventory, "result_contents_read": False,
                  "scope": "automated technical recomputation; independent review and scientific analysis are separate"}
        if not inventory["exact"] or check_ready:
            result["state"] = "ready_filenames_only" if inventory["exact"] else "incomplete_filenames_only"
            return self.finish(result)
        missing_terminal = self.terminal_readiness()
        if missing_terminal:
            result.update(state="pending_supervision_evidence_no_outcomes_read", missing_supervision_records=missing_terminal)
            return self.finish(result)
        self.authenticate()
        self.leases = {}
        rows = []
        for job in self.spec["jobs"]:
            row = self.inputs.read(self.campaign / "results" / self.eid / (job["job_id"] + ".json"))
            self.row_identity(row, job)
            self.runtime_evidence(row, job)
            rows.append(row)
        result.update(result_contents_read=True, technical_status_counts=dict(Counter(r["status"] for r in rows)),
                      execution_keys=sorted({r["execution_key"] for r in rows}), engine_keys=sorted({r["engine_key"] for r in rows}))
        require(len(result["execution_keys"]) == len(result["engine_keys"]) == 1, "mixed execution identities")
        attempts = sorted((self.campaign / "results" / self.eid / "attempts").glob("*.json"))
        result["attempts"] = [{"path": str(p), "status": self.inputs.read(p).get("status")} for p in attempts]
        failures = [r for r in rows if r["status"] != "complete"]
        if failures or attempts:
            result.update(state="technical_failure", technical_failures=[{k: r.get(k) for k in ("job_id", "condition", "status", "error_type", "error")} for r in failures],
                          reason="technical error rows or prior attempts require separate adjudication; no endpoint summary")
            return self.finish(result)
        for job, row in zip(self.spec["jobs"], rows, strict=True):
            self.endpoints(row, job)
            self.patch_evidence(row, job)
        result["identity_checks"] = self.controls(rows)
        history = self.supervision_history()
        historical_failures = terminal_technical_failures(history)
        require(not historical_failures, "historical supervisor technical failures remain unresolved: " + canonical(historical_failures))
        last = max(history, key=lambda r: utc(r["finished_at"]))
        require(last["status"] == "worker_exited" and last["worker_exit_code"] == 0 and last["incomplete"] is False,
                "latest associated supervisor did not finish successfully")
        result["operational_events"] = self.operational_events(last)
        result["supervisor_records"] = [{"lease_id": v["lease_id"], "status": v["status"], "worker_exit_code": v["worker_exit_code"],
                                          "incomplete": v["incomplete"], "saved_row_lease": v["lease_id"] in self.leases}
                                         for v in sorted(history, key=lambda v: utc(v["finished_at"]))]
        result["conditions"] = []
        for name in sorted({r["condition"] for r in rows}):
            group = [r for r in rows if r["condition"] == name]
            result["conditions"].append({"condition": name, "n_trials": len(group), "n_layout_ids": len({r["layout_id"] for r in group}),
                "family_counts": dict(Counter(r["family"] for r in group)), "native_status_counts": dict(Counter(r["generation_status"] for r in group)),
                "native_unresolved": sum(r["generation_status"] != "complete" for r in group),
                "forced_diagnostics_validated": len(group), "full_write_norm_gates": sum(r["intervention"].get("audit_full_write_norm", False) for r in group),
                "projected_norm_gates": sum(r["intervention"].get("audit_projected_norm", False) for r in group),
                "record_only_orientation_failed_flags": sum(r["intervention"].get("record_random_orientation", False)
                    and not r["intervention"].get("audit_full_write_norm", False) and r["random_orientation_audits"][0]["passed"] is False for r in group)})
        result.update(state="complete_technical_pass", technical_passed=True, native_unresolved_policy=self.spec["native_unresolved_policy"]
                      if "native_unresolved_policy" in self.spec else "retain all frozen trials; no forced fallback", statistical_decisions=None)
        return self.finish(result)

    def operational_events(self, last):
        """Hash an immutable complete-line prefix; later other-cohort appends are allowed."""
        path = self.campaign / "events.jsonl"
        raw = path.read_bytes()
        end = raw.rfind(b"\n") + 1
        prefix = raw[:end]
        matching = []
        for line in prefix.splitlines():
            event = strict(line)
            if event.get("experiment_id") == self.eid:
                matching.append(event)
        failure_kinds = {"confirmation_execution_lease_rejected", "model_load_snapshot_gate_failed", "technical_gate_failed",
                         "worker_stopped_full_write_technical_failure", "worker_stopped_repeated_errors"}
        known_failures = [e for e in matching if e.get("kind") in failure_kinds]
        require(not known_failures, "historical technical failure events remain unresolved: "
                + canonical([{k: e.get(k) for k in ("time", "kind", "error_type")} for e in known_failures]))
        finished = [e for e in matching if e.get("kind") == "worker_finished"]
        require(finished and all(type(e.get("errors")) is int and e["errors"] == 0 for e in finished),
                "missing worker-finished event or previous worker technical failures")
        latest_launch = self.cached(Path(last["launch_record_path"]))
        require(any(utc(latest_launch["launched_at"]) <= utc(e["time"]) <= utc(last["finished_at"]) for e in finished),
                "final supervisor lacks associated successful worker-finished event")
        self.event_prefix = (path, prefix)
        return {"path": str(path), "complete_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "complete_prefix_bytes": len(prefix), "captured_trailing_bytes": len(raw) - end,
                "kind_counts": dict(Counter(e["kind"] for e in matching)),
                "worker_finished": [{k: e[k] for k in ("time", "completed", "errors")} for e in finished],
                "interpretation": "startup/worker events are not extra trials; later appends are outside this prefix"}

    def finish(self, result):
        require(filename_inventory(self.campaign, self.spec) == self.inventory, "result filename inventory changed during audit")
        require(not (self.campaign / "reviews" / (self.eid + "_retired.json")).exists(), "cohort retired during audit")
        self.inputs.finish()
        if hasattr(self, "associated_lease_inventory"):
            current = {}
            for path in sorted((self.campaign / "leases").glob("*.json")):
                lease = self.inputs.read(path)
                if lease.get("experiment_id") == self.eid:
                    terminal = self.campaign / "reviews/confirmation_supervision" / (path.stem + ".terminal.json")
                    current[str(path)] = terminal.exists()
            require(current == self.associated_lease_inventory, "associated lease/terminal inventory changed during audit")
            require(all(Path(path).exists() is exists for path, exists in self.associated_launch_inventory.items()),
                    "associated launch inventory changed during audit")
        if hasattr(self, "event_prefix"):
            path, prefix = self.event_prefix
            require(path.read_bytes().startswith(prefix), "captured operational-event prefix changed")
        for name, module in list(sys.modules.items()):
            if name == "src" or name.startswith("src."):
                require(Path(module.__file__).resolve().is_relative_to(self.repo / "src"), "stale implementation import")
        result.update(input_files_sha256=dict(sorted(self.inputs.hashes.items())), audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      finished_at=datetime.now(timezone.utc).isoformat(), model_queries=0, model_weight_reads=0,
                      limitations=["Original activation tensors were not saved; reference vector hashes/norms are authenticated logged proofs, not regenerated activations.",
                                   "Full-vocabulary candidate mass is retained in raw forced metrics; full logits were not saved for independent mass recomputation.",
                                   "Saved materialized-load snapshot evidence is checked; no fresh model-weight rehash or inference is performed.",
                                   "Actual basis/projection activation arrays were not saved; seeded construction is bound by code and logs, not reconstructed from model activations.",
                                   "Whole-cohort technical validation is distinct from the frozen efficacy/specificity statistical decisions."])
        return result


def output_allowed(path, campaign):
    path = path.absolute()
    require(path.parent.resolve() == path.parent, "output parent must be canonical")
    require(not path.exists(), "immutable audit output already exists")
    if path.is_relative_to(campaign):
        require(path.is_relative_to(campaign / "reports") or path.is_relative_to(campaign / "reviews/confirmation_technical"),
                "audit output must not modify an input tree")
    require(path.suffix == ".json", "audit output must be JSON")
    return path


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-ready", action="store_true", help="filename-only dry run even when complete")
    args = parser.parse_args(argv)
    campaign = args.campaign.resolve()
    output = output_allowed(args.output, campaign)
    audit = Audit(REPO, campaign, args.experiment_id)
    try:
        result = audit.audit(args.check_ready)
    except Exception as exc:
        result = {"format": FORMAT, "experiment_id": args.experiment_id, "state": "audit_failed", "technical_passed": False,
                  "error_type": type(exc).__name__, "error": str(exc), "input_files_sha256": audit.inputs.hashes,
                  "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "model_queries": 0, "model_weight_reads": 0}
        publish(output, result)
        raise
    publish(output, result)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "state": result["state"], "technical_passed": result["technical_passed"]}))
    return 0 if args.check_ready or result["technical_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
