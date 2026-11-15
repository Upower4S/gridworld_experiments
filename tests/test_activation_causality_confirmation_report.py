"""Synthetic global-family/censoring/provenance guards; no production outcomes."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.activation_causality.confirmation_contract import PROTOCOLS, ENDPOINT, UNRESOLVED_POLICY, family_quotas  # noqa: E402

_spec = importlib.util.spec_from_file_location("confirmation_report40", ROOT / "scripts/40_activation_causality_confirmation_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def fixture(n=32):
    configs, specs, cohorts = {}, {}, {}
    registry = {"schema_version": 1, "family_size": 4, "candidates": {}, "bindings": {}}
    for protocol in PROTOCOLS:
        cid = protocol
        eid = cid + "_primary"
        scope, count, primary_ids = PROTOCOLS[protocol]
        if protocol == "contextual_input":
            treatment = "layer10_natural"
            controls = ("layer10_same_fact", "layer10_natural_norm_random")
            extra = ["baseline", "counterfactual_text"]
        else:
            treatment = "subspace_forward_1"
            controls = ("baseline", "projected_norm_matched_random_subspace_1")
            extra = ["donor_text"]
        inventory = [treatment, *controls, *extra]
        inventory += ["control_" + str(i) for i in range(count - len(inventory))]
        config = {"candidate_id": cid, "protocol_kind": protocol, "n_layouts": n, "generation_seeds": [42],
                  "max_new_tokens": 4096, "temperature": .7, "claim_scope": scope, "semantic_headline_eligible": False,
                  "condition_inventory": inventory}
        quotas = family_quotas(n) if protocol == "contextual_input" else {"wall": n}
        if protocol == "contextual_input":
            config.update(families=list(report.FAMILIES), family_layout_counts=quotas,
                          family_weights={f: v/n for f, v in quotas.items()})
        else:
            config["family"] = "wall"
        bindings = {}
        for pid, control in zip(primary_ids, controls, strict=True):
            binding = {"treatment": treatment, "control": control, "predicted_direction": "increase",
                "primary_endpoint": ENDPOINT, "target_action_basis": "unique_simulator_counterfactual_optimum",
                "native_unresolved_policy": UNRESOLVED_POLICY, "expected_n_layouts": n, "expected_n_pairs": n,
                "eligibility_rule": {"identifier": "synthetic_only", "sha256": "a"*64},
                "origin": {"kind": "existing_source_comparison", "provenance": {"synthetic": True}},
                "claim_scope": scope, "semantic_headline_eligible": False}
            if protocol == "contextual_input":
                binding.update({k: deepcopy(config[k]) for k in ("families", "family_layout_counts", "family_weights")})
                binding.update(expected_n_layouts_by_family=deepcopy(quotas), expected_n_pairs_by_family=deepcopy(quotas),
                               resampling_scheme="within_family_layout_bootstrap", stratification_key="family")
            bindings[pid] = binding
        config["primary_comparisons"] = bindings
        configs[cid] = config
        registry["candidates"][cid] = list(primary_ids)
        registry["bindings"][cid] = deepcopy(bindings)
        jobs, rows, pair_ids = [], [], []
        index = 0
        for family, size in quotas.items():
            for _ in range(size):
                pair, layout = cid + "_pair_" + str(index), cid + "_layout_" + str(index)
                pair_ids.append(pair)
                for condition in inventory:
                    job = {"experiment_id": eid, "partition": "confirmation", "pair_id": pair, "layout_id": layout,
                        "family": family, "condition": condition, "natural": True, "seed": 42, "temperature": .7,
                        "max_new_tokens": 4096, "claim_scope": scope, "semantic_headline_eligible": False,
                        "target_action": "RIGHT", "original_action": "UP", "layout_aliases": [layout],
                        "metadata": {"source_pair": {"pair_id": pair, "layout_id": layout, "family": family}}}
                    job["job_id"] = report.fingerprint(job)[:24]
                    jobs.append(job)
                    rows.append({**deepcopy(job), "status": "complete", "generation_status": "complete",
                        "native_action": "RIGHT" if condition in {treatment, "counterfactual_text", "donor_text"} else "UP",
                        "forced_q": None, "spec_sha256": cid, "execution_key": cid, "inference_configuration": {"head_mode": "full"}})
                index += 1
        comparisons = [{**deepcopy(binding), "id": cid + "_" + pid, "primary_id": pid, "candidate_id": cid,
            "experiment_id": eid, "partition": "confirmation", "confirmatory": True, "eligible_pair_ids": pair_ids}
            for pid, binding in bindings.items()]
        specs[eid] = {"experiment_id": eid, "partition": "confirmation", "cohort_role": "primary", "candidate_id": cid,
            "protocol_kind": protocol, "claim_scope": scope, "semantic_headline_eligible": False,
            "jobs": jobs, "comparisons": comparisons, "frozen_configuration": config}
        cohorts[eid] = {"rows": rows, "missing_job_ids": [], "technical_failures": [], "expected_jobs": len(jobs),
                       "saved_jobs": len(rows), "technical_status_counts": {"complete": len(rows)}}
    frozen = {"freeze_id": "synthetic", "candidate_configurations": configs, "confirmation_primary_family": registry,
              "supplements": dict.fromkeys(configs, False)}
    return {"authenticated": True, "mode": "complete_only", "freeze": frozen, "specs": specs, "cohorts": cohorts,
            "provenance": {"synthetic": True, "input_file_sha256": {}}}


def remove_unselected(bundle, cid="contextual_input"):
    cohort = bundle["cohorts"][cid + "_primary"]
    index = next(i for i, row in enumerate(cohort["rows"]) if row["condition"].startswith("control_"))
    removed = cohort["rows"].pop(index)
    cohort.update(missing_job_ids=[removed["job_id"]], saved_jobs=len(cohort["rows"]), technical_status_counts={"complete": len(cohort["rows"])})


def add_lexical(bundle):
    """Synthetic exact32 subset of an N>=32 parent; never reads source results."""
    cid, eid = "contextual_input", "contextual_input_primary"
    parent = bundle["specs"][eid]
    old_rows = {r["job_id"]: r for r in bundle["cohorts"][eid]["rows"]}
    rows = []
    for job in parent["jobs"]:
        row = old_rows[job["job_id"]]
        job.update(selection={"recipient_positions": [1], "donor_positions": [1]}, intervention={"layers": []})
        job["job_id"] = report.fingerprint({k:v for k,v in job.items() if k != "job_id"})[:24]
        row.update(deepcopy(job))
        rows.append(row)
    bundle["cohorts"][eid]["rows"] = rows
    pairs = {j["pair_id"]: j["metadata"]["source_pair"] for j in parent["jobs"]}
    quotas = dict(zip(report.FAMILIES, (11,11,10)))
    selected = []
    for family, count in quotas.items():
        group = [p for p in pairs.values() if p["family"] == family]
        group.sort(key=lambda p:(report.hashlib.sha256((p["layout_id"]+"\nactivation_lexical_v1:20260912").encode()).hexdigest(),p["layout_id"]))
        selected.extend(group[:count])
    ids, weights = {p["pair_id"] for p in selected}, {f:n/32 for f,n in quotas.items()}
    jobs = [j for j in parent["jobs"] if j["pair_id"] in ids]
    config = bundle["freeze"]["candidate_configurations"][cid]
    definitions = {(b["treatment"],b["control"]) for b in config["primary_comparisons"].values()}
    definitions.update({("layer10_natural","baseline"),("counterfactual_text","baseline")})
    comparisons = [{"treatment":t,"control":c,"confirmatory":False,"eligible_pair_ids":sorted(ids),
        "expected_n_layouts":32,"expected_n_pairs":32,"family_weights":weights,"family_layout_counts":quotas} for t,c in sorted(definitions)]
    ref = {"parent_spec_fingerprint":report.fingerprint(parent),"experiment_id":eid,"family_weights":weights,"execution_jobs_added":0,
        "job_fingerprints":{j["job_id"]:report.fingerprint(j) for j in jobs},
        "job_bindings":[{k:j[k] for k in ("job_id","pair_id","layout_id","condition","seed")} for j in jobs],"comparisons":deepcopy(comparisons)}
    alt_id = cid+"_alternate"
    alt = {"experiment_id":alt_id,"candidate_id":cid,"cohort_role":"lexical_supplement","parent_experiment_id":eid,
        "parent_spec_fingerprint":report.fingerprint(parent),"lexical_subset":{"pair_ids":sorted(ids)},
        "family_weights":weights,"family_layout_counts":quotas,"standard_subset_reference":ref,"comparisons":comparisons,"jobs":[]}
    alt["lexical_subset_sha256"] = report.fingerprint(alt["lexical_subset"])
    parent_rows = {r["job_id"]:r for r in rows}
    alt_rows=[]
    for original in jobs:
        job=deepcopy(original)
        job["experiment_id"]=alt_id
        job["metadata"].update(paired_standard_job_id=original["job_id"],paired_standard_job_sha256=report.fingerprint(original))
        job["job_id"]=report.fingerprint({k:v for k,v in job.items() if k!="job_id"})[:24]
        alt["jobs"].append(job)
        alt_rows.append({**deepcopy(parent_rows[original["job_id"]]),**deepcopy(job)})
    bundle["specs"][alt_id]=alt
    bundle["cohorts"][alt_id]={"rows":alt_rows,"missing_job_ids":[],"technical_failures":[],"expected_jobs":len(alt_rows),
        "saved_jobs":len(alt_rows),"technical_status_counts":{"complete":len(alt_rows)}}
    bundle["freeze"]["supplements"][cid]=True


def full_write_fixture(audited=True, random=False):
    import numpy as np
    from types import SimpleNamespace
    from src.activation_causality.engine import MLXInterventionEngine
    from src.activation_causality.full_write_audit import prepare_full_write_norm_audit, validate_full_write_norm_audit, random_orientation_audit
    from src.activation_causality.subspace_audit import array_fingerprint
    source={k:"a"*64 for k in ("candidate_source_spec_sha256","source_pair_sha256","recipient_token_sha256","donor_token_sha256","selection_sha256")}
    source.update(candidate_source_experiment_id="synthetic",source_pair_id="pair",renderer_id="canonical_v1",
                  lineage_kind="inline_physical_source_pair",span_rule="within_context_named_variable_toggle",recipient_role="recipient",donor_role="natural")
    r=np.array([[1.,0.,0.]],dtype=np.float32)
    d=np.array([[1.,2.,0.]],dtype=np.float32)
    audit=prepare_full_write_norm_audit(r,d,layer=10,site="resid_post",recipient_positions=[1],donor_positions=[2],dose=1.,source_identity=source)
    intervention={"layers":[10],"site":"resid_post","operation":"replace","dose":1.,"full_write_source_identity":source}
    if audited:
        intervention["audit_full_write_norm"]=True
    job={"intervention":intervention,"selection":{"recipient_positions":[1],"donor_positions":[2]},"metadata":{}}
    compiled=d
    orientations=[]
    if random:
        intervention.update(record_random_orientation=True,random_seed=42,random_control=True)
        raw=np.random.default_rng(42).standard_normal(r.shape).astype(np.float32)
        direction=raw.copy()
        direction *= np.linalg.norm(d-r,axis=-1,keepdims=True)/np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-12)
        compiled=r+direction
        orientations=[random_orientation_audit(r,d,raw,direction,compiled,seed=42,source_identity=source)]
    audit.update(compiled_donor_float32_sha256=array_fingerprint(compiled),inline_definition_sha256=report.fingerprint(source))
    # Use the frozen engine's actual log emitter; no model or MLX is loaded.
    engine=SimpleNamespace(_mx=SimpleNamespace(float32=np.float32,array=np.array,eval=lambda *args: None))
    tensor=np.zeros((1,2,3),np.float32)
    tensor[0,1]=r[0]
    logs=[]
    MLXInterventionEngine._patch_tensor(engine,tensor,{"layer":10,"site":"resid_post","positions":[1],
        "operation":"replace","dose":1.,"donor":compiled,"expected_recipient_float32_sha256":array_fingerprint(r)},logs)
    row={"status":"complete","patch_log":logs,"compiled_full_write_audits":[audit] if audited else [],
         "applied_full_write_audits":[validate_full_write_norm_audit(audit,logs)] if audited else [],
         "full_write_norm_audit_status":"passed" if audited else "not_requested","random_orientation_audits":orientations}
    return job,row


def runtime_fixture(directory):
    """Synthetic files with the real frozen execution-key recipes; no model."""
    import numpy as np
    from src.activation_causality.engine import ENGINE_DEPENDENCY_PATHS
    from src.activation_causality.basis_runtime import activation_representation
    from src.activation_causality.confirmation_supervisor import _interpreter_binding
    campaign=Path(directory).resolve()
    def write(relative,value):
        path=campaign/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True))
        return str(path),report.hashlib.sha256(path.read_bytes()).hexdigest()
    sources=dict.fromkeys(ENGINE_DEPENDENCY_PATHS,"a"*64)
    names=("engine","campaign","confirmation_supervisor","confirmation_contract","basis_runtime","model_identity","localization","subspace_controls","subspace_audit","full_write_audit")
    sources.update({"src/activation_causality/"+name+".py":"a"*64 for name in names})
    versions={"mlx":"synthetic","mlx-lm":"synthetic","numpy":np.__version__,"transformers":"synthetic"}
    runtime={"sources":sources,"versions":versions}
    representation=activation_representation(runtime)
    snapshot={"files":dict.fromkeys(("config.json","tokenizer.json","tokenizer_config.json","model.safetensors.index.json"),"b"*64),"identity_sha256":"c"*64}
    cutoff="2030-01-01T00:00:00+00:00"
    frozen={"model_identity":snapshot,"implementation":sources,"activation_representation":representation,
            "confirmation_mode":"co_primary_native_v4","freeze_id":"synthetic","inference_cutoff":cutoff,"worker_entrypoint_sha256":"d"*64}
    config={"model_path":str(campaign/"never_loaded")}
    bundle={"campaign":campaign,"freeze":frozen,"freeze_sha256":"e"*64,"campaign_sha256":"f"*64,"campaign_configuration":config}
    execution={"sources":sources,"versions":versions,"model":config["model_path"],"model_files":snapshot["files"],"head_mode":"full",
        "model_snapshot_identity_sha256":snapshot["identity_sha256"],"confirmation_execution_contract":{
        "confirmation_mode":frozen["confirmation_mode"],"freeze_id":"synthetic","freeze_sha256":"e"*64,"inference_cutoff":cutoff}}
    engine={"engine_sha256":"a"*64,"engine_dependency_sources":{k:sources[k] for k in ENGINE_DEPENDENCY_PATHS},"versions":versions,
        "model":config["model_path"],"model_files":snapshot["files"],"head_mode":"full","model_snapshot_identity_sha256":snapshot["identity_sha256"]}
    execution_key,engine_key=report.fingerprint(execution),report.fingerprint(engine)
    spec={"experiment_id":"synthetic_primary","candidate_id":"contextual","cohort_role":"primary","protocol_kind":"contextual_input"}
    row={"status":"complete","patch_log":[],"execution_key":execution_key,"engine_key":engine_key,"spec_sha256":"0"*64,
        "model_snapshot_identity_sha256":snapshot["identity_sha256"],"inference_source_hashes":sources,
        "confirmation_lease_id":"1"*32,"confirmation_inference_cutoff":cutoff,"started_at":"2029-01-01T00:00:02+00:00",
        "finished_at":"2029-01-01T00:00:03+00:00"}
    for field,ns in (("subspace_implementation_sha256",("basis_runtime","model_identity","localization","subspace_controls","subspace_audit")),
                     ("full_write_implementation_sha256",("full_write_audit","subspace_audit","basis_runtime","model_identity"))):
        row[field]={"src/activation_causality/"+n+".py":"a"*64 for n in ns}
    load={"format":"materialized_model_snapshot_audit_v1","passed":True,"experiment_id":spec["experiment_id"],"execution_key":execution_key,
          "before_load":snapshot,"after_load":snapshot,"activation_representation":representation,"engine_metadata":{"loaded":True}}
    row["model_load_audit_path"],row["model_load_audit_sha256"]=write("reviews/model_load_synthetic.json",load)
    write("reviews/engine_validation_"+engine_key[:16]+".json",{"engine_key":engine_key,"runtime":runtime,"validation":{"passed":True,"model":{"head_mode":"full"}}})
    lease={"format":"supervised_confirmation_execution_lease_v1","lease_id":row["confirmation_lease_id"],
           "experiment_id":spec["experiment_id"],"candidate_id":spec["candidate_id"],"cohort_role":"primary","freeze_id":"synthetic",
           "spec_sha256":row["spec_sha256"],"freeze_sha256":"e"*64,"campaign_sha256":"f"*64,"inference_cutoff":cutoff,
           "source_hashes":sources,"worker_entrypoint_sha256":"d"*64,"supervisor_pid":123,"created_at":"2029-01-01T00:00:00+00:00",
           "repo_path":str(ROOT),"campaign_path":str(campaign),"spec_path":str(campaign/"specs"/(spec["experiment_id"]+".json")),
           "freeze_path":str(campaign/"freezes/synthetic.json"),"worker_entrypoint_path":str(ROOT/"scripts/30_activation_causality_campaign.py"),
           **_interpreter_binding()}
    row["confirmation_lease_path"],row["confirmation_lease_sha256"]=write("leases/"+row["confirmation_lease_id"]+".json",lease)
    prefix="reviews/confirmation_supervision/"+row["confirmation_lease_id"]
    launch={"format":"supervised_confirmation_launch_v1","lease_id":row["confirmation_lease_id"],"lease_sha256":row["confirmation_lease_sha256"],
            "supervisor_pid":123,"worker_pid":456,"worker_process_group":456,"launched_at":"2029-01-01T00:00:01+00:00",
            "command":[lease["interpreter_path"],"-I","-B",lease["worker_entrypoint_path"],"--campaign",str(campaign),"run","--experiment-id",spec["experiment_id"]]}
    launch_path,launch_sha=write(prefix+".launch.json",launch)
    log_path,log_sha=write("logs/confirmation_supervision_"+row["confirmation_lease_id"]+".log",{"synthetic":True})
    terminal={"format":"supervised_confirmation_terminal_v1","lease_id":row["confirmation_lease_id"],"lease_path":row["confirmation_lease_path"],
              "lease_sha256":row["confirmation_lease_sha256"],"launch_record_path":launch_path,"launch_record_sha256":launch_sha,
              "terminal_record_path":str(campaign/(prefix+".terminal.json")),"experiment_id":spec["experiment_id"],"inference_cutoff":cutoff,
              "log_path":log_path,"log_sha256":log_sha,"status":"worker_exited","worker_exit_code":0,"incomplete":False,
              "finished_at":"2029-01-01T00:00:04+00:00"}
    write(prefix+".terminal.json",terminal)
    job={"intervention":{"layers":[]},"metadata":{}}
    return bundle,spec,job,row


def registered_fixture(directory):
    from src.activation_causality.model_identity import FORMAT
    campaign=Path(directory).resolve()
    bundle=fixture()
    def write(relative,value):
        path=campaign/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True))
        return str(path),report.hashlib.sha256(path.read_bytes()).hexdigest()
    config={"model_path":str(campaign/"never_loaded"),"deadline":"2030-01-01T06:00:00+00:00"}
    _,campaign_sha=write("campaign.json",config)
    snapshot={"format":FORMAT,"path":config["model_path"],"files":{"fake.safetensors":"a"*64},
              "indexed_shards":["fake.safetensors"],"actual_weight_files":["fake.safetensors"],"index_inventory_exact":True,"stable_during_read":True}
    snapshot["identity_sha256"]=report.fingerprint(snapshot)
    plan={**deepcopy(bundle["freeze"]),"confirmation_mode":"co_primary_native_v4","schema_version":4,
          "campaign_sha256":campaign_sha,"model_identity":snapshot,"inference_cutoff":"2030-01-01T00:00:00+00:00",
          "analysis_configuration":report.ANALYSIS_CONFIGURATION,"planned_experiments":{}}
    for eid,spec in bundle["specs"].items():
        spec.update(freeze_id="synthetic",cohort_complete=True)
        pairs={j["pair_id"]:j["metadata"]["source_pair"] for j in spec["jobs"]}
        dataset={"experiment_id":eid,"partition":"confirmation","cohort_role":"primary","candidate_id":spec["candidate_id"],"pairs":list(pairs.values())}
        item={"candidate_id":spec["candidate_id"],"cohort_role":"primary"}
        for kind,folder,value in (("spec","specs",spec),("dataset","datasets",dataset)):
            path,sha=write("planning/"+eid+"."+kind+".json",value)
            write(folder+"/"+eid+".json",value)
            item.update({kind+"_path":path,kind+"_sha256":sha})
        plan["planned_experiments"][eid]=item
    plan_path,plan_sha=write("planning/final.json",plan)
    review={"approved":True,"freeze_id":"synthetic","plan_sha256":plan_sha,"plan_fingerprint":report.fingerprint(plan)}
    review_path,review_sha=write("reviews/approval.json",review)
    frozen={**plan,"plan_path":plan_path,"plan_sha256":plan_sha,"review_path":review_path,"review_sha256":review_sha}
    write("freezes/synthetic.json",frozen)
    return campaign,bundle,frozen


class ConfirmationReportTests(unittest.TestCase):
    def test_operational_history_retains_missing_completion_and_binds_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle,spec,job,row=runtime_fixture(directory)
            bundle["specs"]={spec["experiment_id"]:spec}
            bundle["freeze"]["planned_experiments"]={spec["experiment_id"]:{"spec_sha256":row["spec_sha256"]}}
            history=report._supervision_history(bundle["campaign"],bundle,report.Inputs())
            self.assertEqual(history[spec["experiment_id"]][0]["status"],"worker_exited")
            terminal=bundle["campaign"]/"reviews/confirmation_supervision"/(row["confirmation_lease_id"]+".terminal.json")
            original=terminal.read_bytes()
            terminal.unlink()
            history=report._supervision_history(bundle["campaign"],bundle,report.Inputs())
            self.assertEqual(history[spec["experiment_id"]][0]["status"],"completion_evidence_absent")
            terminal.write_bytes(original)
            bad=json.loads(original)
            bad["launch_record_sha256"]="0"*64
            terminal.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"terminal/launch"):
                report._supervision_history(bundle["campaign"],bundle,report.Inputs())

    def test_declared_launched_worker_missing_launch_is_pending_not_hash_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle,spec,job,row=runtime_fixture(directory)
            bundle["specs"]={spec["experiment_id"]:spec}
            bundle["freeze"]["planned_experiments"]={spec["experiment_id"]:{"spec_sha256":row["spec_sha256"]}}
            launch=bundle["campaign"]/"reviews/confirmation_supervision"/(row["confirmation_lease_id"]+".launch.json")
            launch.unlink()
            history=report._supervision_history(bundle["campaign"],bundle,report.Inputs())[spec["experiment_id"]]
            self.assertEqual(report._historical_pending(history),[{"reason":"historical_launched_worker_missing_launch",
                "lease":history[0]["lease_path"]}])
            self.assertEqual(report._historical_failures(history,[]),[])

    def test_basis_hash_conventions_are_distinct_and_both_bound_to_saved_bytes(self):
        import numpy as np
        from src.activation_causality.localization import _array_fingerprint
        from src.activation_causality.subspace_audit import array_fingerprint
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/"basis.npy"
            basis=np.array([[1.],[0.]],dtype=np.float32)
            np.save(path,basis)
            config={"basis_path":str(path),"basis_sha256":report.hashlib.sha256(path.read_bytes()).hexdigest(),"rank":1,
                    "basis_metadata":{"basis_array_sha256":_array_fingerprint(basis)}}
            self.assertNotEqual(_array_fingerprint(basis),array_fingerprint(basis))
            self.assertEqual(report._basis_norm_hash(config,report.Inputs()),array_fingerprint(basis))
            config["basis_metadata"]["basis_array_sha256"]=array_fingerprint(basis)
            with self.assertRaisesRegex(ValueError,"array identity"):
                report._basis_norm_hash(config,report.Inputs())

    def test_registered_and_planned_artifact_contract_without_model_or_results(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign,bundle,frozen=registered_fixture(directory)
            with patch("src.activation_causality.selection.validate_confirmation_ready",side_effect=AssertionError("readiness must not run")), \
                 patch("src.activation_causality.engine.MLXInterventionEngine._ensure_loaded",side_effect=AssertionError("model forbidden")):
                value=report._contract(campaign,"synthetic",report.Inputs())
            self.assertEqual(set(value["specs"]),set(bundle["specs"]))
            path=campaign/"specs/contextual_input_primary.json"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                report._contract(campaign,"synthetic",report.Inputs())
            planned=Path(frozen["planned_experiments"]["contextual_input_primary"]["spec_path"])
            path.write_bytes(planned.read_bytes())
            dataset=campaign/"datasets/contextual_input_primary.json"
            data=json.loads(dataset.read_text())
            data["pairs"][0]["layout_id"]="spoofed"
            dataset.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                report._contract(campaign,"synthetic",report.Inputs())

    def test_runtime_model_gate_full_source_recipe_and_lease_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle,spec,job,row=runtime_fixture(directory)
            inputs=report.Inputs()
            self.assertEqual(report._runtime_evidence(row,job,spec,bundle,inputs),[])
            self.assertEqual(len(inputs.hashes),6)
            for field,value in (("execution_key","0"*64),("model_snapshot_identity_sha256","0"*64),
                                ("confirmation_lease_id","0"*32),("subspace_implementation_sha256",{}),
                                ("started_at","2030-01-01T00:00:00+00:00")):
                bad=deepcopy(row)
                bad[field]=value
                with self.assertRaises(ValueError):
                    report._runtime_evidence(bad,job,spec,bundle,report.Inputs())
            terminal=Path(next(iter(inputs.lease_records.values()))["path"])
            terminal_bytes=terminal.read_bytes()
            terminal.unlink()
            self.assertEqual(report._runtime_evidence(row,job,spec,bundle,report.Inputs()),["operational_pending:missing_launch_or_terminal:"+row["confirmation_lease_id"]])
            terminal.write_bytes(terminal_bytes)
            value=json.loads(terminal_bytes)
            value["launch_record_sha256"]="0"*64
            terminal.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"terminal association"):
                report._runtime_evidence(row,job,spec,bundle,report.Inputs())
            terminal.write_bytes(terminal_bytes)
            path=Path(row["model_load_audit_path"])
            load=json.loads(path.read_text())
            load["engine_metadata"]["loaded"]=False
            path.write_text(json.dumps(load))
            row["model_load_audit_sha256"]=report.hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,"model-load"):
                report._runtime_evidence(row,job,spec,bundle,report.Inputs())

    def test_lexical_full_report_keeps_fixed32_weights_distinct_from_primary64(self):
        bundle=fixture(64)
        add_lexical(bundle)
        result=report.build_report(bundle)
        entry=result["lexical_supplements"][0]
        self.assertEqual(len(entry["contrasts"]),4)
        self.assertTrue(entry["descriptive_transfer_gate_met"])
        self.assertNotEqual(entry["family_weights"],entry["primary_family_weights"])
        self.assertEqual(entry["family_weights"],{"wall":11/32,"key_possession":11/32,"door_status":10/32})
        self.assertEqual(result["primary"]["analysis_config"]["confirmation_primary_family_size"],4)
        self.assertTrue(all(c["confidence_intervals"] is None for c in entry["contrasts"]))
        display=result["cohort_display_metadata"]["contextual_input_alternate"]
        self.assertEqual(display["expected_n_layouts"],32)
        self.assertEqual(display["additional_layouts_beyond_parent"],0)
        self.assertEqual(display["family_weights"],entry["family_weights"])
        counts=result["complete_cohort_condition_counts"]["contextual_input_alternate"]["baseline"]
        self.assertEqual(counts["by_family"]["wall"]["n_layouts"],11)
        for mutation in (lambda a:a["standard_subset_reference"]["job_fingerprints"].clear(),
                         lambda a:a["standard_subset_reference"]["comparisons"][0].update(family_weights=entry["primary_family_weights"]),
                         lambda a:a["jobs"][0]["selection"].update(recipient_positions=[1,2])):
            bad=deepcopy(bundle)
            mutation(bad["specs"]["contextual_input_alternate"])
            with self.assertRaises(ValueError):
                report.build_report(bad)

    def test_full_write_geometry_norm_recomputation_and_no_legacy_gate(self):
        from src.activation_causality.full_write_audit import validate_full_write_norm_audit
        job,row=full_write_fixture()
        self.assertNotIn("after_l2",row["patch_log"][0])
        self.assertEqual(report._norm_evidence(row,job),[])
        bad=deepcopy(row)
        bad["patch_log"][0]["positions"]=[2]
        with self.assertRaisesRegex(ValueError,"geometry"):
            report._norm_evidence(bad,job)
        bad=deepcopy(row)
        bad["patch_log"][0]["delta_l2"]=[1.9]
        with self.assertRaisesRegex(ValueError,"recomputed"):
            report._norm_evidence(bad,job)
        bad["applied_full_write_audits"]=[validate_full_write_norm_audit(bad["compiled_full_write_audits"][0],bad["patch_log"])]
        bad["full_write_norm_audit_status"]="failed"
        self.assertEqual(report._norm_evidence(bad,job),["failed_full_write_norm_audit_status"])
        legacy,record=full_write_fixture(audited=False,random=True)
        record["random_orientation_audits"][0].update(passed=False,errors=["synthetic record-only diagnostic"])
        self.assertEqual(report._norm_evidence(record,legacy),[])
        new,record=full_write_fixture(random=True)
        record["random_orientation_audits"][0].update(passed=False,errors=["synthetic required gate"])
        self.assertEqual(report._norm_evidence(record,new),["random_orientation_sanity_failed"])
        record["random_orientation_audits"][0]["raw_gaussian_float32_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"Gaussian"):
            report._norm_evidence(record,new)

    def test_emitted_norm_vectors_remain_required_and_finite(self):
        job,row=full_write_fixture()
        for field in ("before_l2","delta_l2","relative_delta_l2"):
            for value in (None,[],[-1.],[float("nan")],[float("inf")],[True]):
                with self.subTest(field=field,value=value):
                    bad=deepcopy(row)
                    bad["patch_log"][0][field]=value
                    with self.assertRaisesRegex(ValueError,"patch norm array"):
                        report._norm_evidence(bad,job)

    def test_natural_and_random_compiled_donor_bindings_are_required(self):
        from src.activation_causality.full_write_audit import validate_full_write_norm_audit
        for random in (False,True):
            job,row=full_write_fixture(random=random)
            self.assertEqual(report._norm_evidence(row,job),[])
            for field in ("compiled_donor_float32_sha256","inline_definition_sha256"):
                with self.subTest(random=random,field=field):
                    bad=deepcopy(row)
                    bad["compiled_full_write_audits"][0][field]="b"*64
                    bad["applied_full_write_audits"]=[validate_full_write_norm_audit(bad["compiled_full_write_audits"][0],bad["patch_log"])]
                    with self.assertRaisesRegex(ValueError,"compiled donor/source"):
                        report._norm_evidence(bad,job)

    def test_subspace_dose_rank_and_reference_recomputed(self):
        import numpy as np
        from src.activation_causality.subspace_audit import prepare_projected_norm_audit,validate_applied_norm_audit
        audit=prepare_projected_norm_audit([[1.,0.]],[[3.,2.]],np.array([[1.],[0.]],dtype=np.float32),0,"resid_post",[2],.5)
        log={"layer":0,"site":"resid_post","positions":[2],"operation":"subspace","dose":.5,"rank":1,
             "before_l2":[1.],"delta_l2":[1.],"relative_delta_l2":[1.],"recipient_float32_sha256":audit["expected_recipient_float32_sha256"]}
        job={"intervention":{"layers":[0],"operation":"subspace","site":"resid_post","dose":.5,"audit_projected_norm":True},
             "selection":{"recipient_positions":[2]},"metadata":{"rank":1}}
        row={"status":"complete","patch_log":[log],"compiled_patch_audits":[{"projected_norm_audit":audit}],
             "applied_projected_norm_audits":[validate_applied_norm_audit(audit,[log])],"projected_norm_audit_status":"passed"}
        self.assertEqual(report._norm_evidence(row,job),[])
        for field,value in (("rank",4),("dose",1.)):
            bad=deepcopy(row)
            bad["patch_log"][0][field]=value
            with self.assertRaisesRegex(ValueError,"geometry"):
                report._norm_evidence(bad,job)

    def test_global_four_slot_holm_and_both_candidate_conjunctions(self):
        bundle = fixture()
        result = report.build_report(bundle)
        summary = result["primary"]
        self.assertEqual(summary["analysis_config"]["confirmation_primary_family_size"], 4)
        self.assertEqual(len(summary["primary_components"]), 4)
        self.assertTrue(all(c["status"] == "supported" for c in summary["candidate_conjunctions"]))
        values = sorted((c["primary"]["p_value"], i) for i, c in enumerate(summary["primary_components"]))
        previous = 0
        for rank, (p, index) in enumerate(values):
            previous = max(previous, min(1, (4-rank)*p))
            self.assertEqual(summary["primary_components"][index]["primary"]["holm_p_value"], previous)
        self.assertEqual(len(result["complete_cohort_condition_counts"]["contextual_input_primary"]), 15)

    def test_missing_unselected_control_blocks_whole_candidate_without_alpha_release(self):
        bundle = fixture()
        remove_unselected(bundle)
        with self.assertRaisesRegex(ValueError, "complete-only"):
            report.build_report(bundle)
        bundle["mode"] = "at_cutoff"
        result = report.build_report(bundle, at_cutoff=True)
        components = result["primary"]["primary_components"]
        pending = [c for c in components if c["candidate_id"] == "contextual_input"]
        self.assertTrue(all(c["component_status"] == "pending" and c["primary"]["multiplicity_input_p_value"] == 1 for c in pending))
        self.assertTrue(all(c["primary"]["effect"] is None and c["expected_n_layouts"] == 32 for c in pending))
        self.assertEqual(result["primary"]["analysis_config"]["confirmation_primary_family_size"], 4)
        self.assertIsNone(result["complete_cohort_condition_counts"]["contextual_input_primary"])
        display=result["cohort_display_metadata"]["contextual_input_primary"]
        self.assertEqual(len(display["condition_order"]),15)
        self.assertEqual(display["expected_jobs"],480)
        self.assertTrue(all(n==32 for n in display["expected_condition_trials"].values()))

    def test_missing_operational_completion_is_pending_without_native_effect(self):
        bundle=fixture()
        bundle["mode"]="at_cutoff"
        bundle["cohorts"]["contextual_input_primary"]["operational_pending_reasons"]=[{"reason":"missing_terminal"}]
        result=report.build_report(bundle,at_cutoff=True)
        parts=[c for c in result["primary"]["primary_components"] if c["candidate_id"]=="contextual_input"]
        self.assertTrue(all(c["component_status"]=="pending" and c["primary"]["effect"] is None for c in parts))
        self.assertTrue(all(c["primary"]["multiplicity_input_p_value"]==1 for c in parts))

    def test_later_success_does_not_clear_historical_technical_failures(self):
        for failure in ("worker_failed", "supervisor_error", "attempt"):
            bundle=fixture()
            cohort=bundle["cohorts"]["contextual_input_primary"]
            success={"lease_path":"later", "terminal":{"status":"worker_exited","worker_exit_code":0,"incomplete":False}}
            cohort["supervision_history"]=[success]
            if failure == "attempt":
                cohort["prior_attempt_records"]=[{"path":"old_attempt","sha256":"a"*64,"job_id":cohort["rows"][0]["job_id"],
                    "status":"error","generation_status":"technical_error"}]
                cohort["prior_attempt_files_sha256"]={"old_attempt":"a"*64}
            else:
                cohort["supervision_history"].insert(0,{"lease_path":"earlier","terminal":{"status":failure,"error_type":"ValueError"}})
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError,"complete-only"):
                report.build_report(bundle)
            bundle["mode"]="at_cutoff"
            result=report.build_report(bundle,at_cutoff=True)
            parts=[c for c in result["primary"]["primary_components"] if c["candidate_id"]=="contextual_input"]
            self.assertTrue(all(c["component_status"]=="invalid" and c["primary"]["multiplicity_input_p_value"]==1 for c in parts))
            self.assertTrue(all(c["primary"]["effect"] is None for c in parts))
        interrupted={"lease_path":"earlier","terminal":{"status":"supervisor_error","error_type":"SupervisorInterrupted"}}
        self.assertEqual(report._historical_failures([interrupted,success],[]),[])

    def test_earlier_missing_completion_remains_pending_after_later_success(self):
        earlier_records=[{"lease_path":"earlier","terminal":None},
                         {"lease_path":"earlier","launch_evidence_present":False,
                          "terminal":{"status":"worker_exited","worker_exit_code":0,"incomplete":False}}]
        success={"lease_path":"later","launch_evidence_present":True,
                 "terminal":{"status":"worker_exited","worker_exit_code":0,"incomplete":False}}
        for earlier in earlier_records:
            bundle=fixture()
            cohort=bundle["cohorts"]["contextual_input_primary"]
            cohort["supervision_history"]=[earlier,success]
            with self.subTest(earlier=earlier), self.assertRaisesRegex(ValueError,"complete-only"):
                report.build_report(bundle)
            bundle["mode"]="at_cutoff"
            result=report.build_report(bundle,at_cutoff=True)
            parts=[c for c in result["primary"]["primary_components"] if c["candidate_id"]=="contextual_input"]
            self.assertTrue(all(c["component_status"]=="pending" and c["primary"]["effect"] is None
                                and c["primary"]["multiplicity_input_p_value"]==1 for c in parts))
            self.assertFalse(result["whole_candidate_inventories"]["contextual_input"]["technical_failures"])
        for status in ("cutoff_before_launch","supervisor_error"):
            prelaunch={"lease_path":"earlier","launch_evidence_present":False,
                       "terminal":{"status":status,"error_type":"SupervisorInterrupted","worker_exit_code":None}}
            self.assertEqual(report._historical_pending([prelaunch,success]),[])
            self.assertEqual(report._historical_failures([prelaunch,success],[]),[])

    def test_all_frozen_technical_event_types_block_even_after_success(self):
        kinds=sorted(report.TECHNICAL_EVENTS)+["worker_finished"]
        events=[]
        for number,kind in enumerate(kinds,1):
            event={"kind":kind,"experiment_id":"contextual_input_primary","errors":1 if kind=="worker_finished" else 0}
            events.append({"line":number,"event_sha256":report.fingerprint(event),"event":event})
        self.assertEqual(len(report._historical_failures([],[],events)),6)
        clean={"kind":"worker_finished","errors":0}
        self.assertEqual(report._historical_failures([],[],[{"line":1,"event_sha256":report.fingerprint(clean),"event":clean}]),[])
        bundle=fixture()
        bundle["cohorts"]["contextual_input_primary"]["operational_events"]=events
        with self.assertRaisesRegex(ValueError,"complete-only"):
            report.build_report(bundle)
        bundle["mode"]="at_cutoff"
        value=report.build_report(bundle,at_cutoff=True)
        parts=[c for c in value["primary"]["primary_components"] if c["candidate_id"]=="contextual_input"]
        self.assertTrue(all(c["component_status"]=="invalid" and c["primary"]["multiplicity_input_p_value"]==1 for c in parts))

    def test_event_history_exact_source_lines_and_candidate_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            events=[{"kind":"technical_gate_failed","experiment_id":"contextual_input_primary"},
                    {"kind":"worker_finished","errors":1,"experiment_id":"another_cohort"}]
            (root/"events.jsonl").write_text(''.join(json.dumps(e)+'\n' for e in events))
            inputs=report.Inputs()
            result=report._operational_events(root,fixture(),inputs)
            self.assertEqual(set(result),{"contextual_input_primary"})
            self.assertEqual(result["contextual_input_primary"][0],{"line":1,"event_sha256":report.fingerprint(events[0]),"event":events[0]})
            self.assertEqual(len(inputs.hashes),1)

    def test_technical_failure_takes_precedence_over_pending_and_never_becomes_behavior(self):
        bundle = fixture()
        remove_unselected(bundle)
        cohort = bundle["cohorts"]["contextual_input_primary"]
        row = cohort["rows"][0]
        row.update(status="error", generation_status="technical_error", native_action=None)
        cohort["technical_failures"] = [{"job_id": row["job_id"], "reason": "fixed_norm_failure"}]
        cohort["technical_status_counts"] = dict(Counter(r["status"] for r in cohort["rows"]))
        bundle["mode"] = "at_cutoff"
        result = report.build_report(bundle, at_cutoff=True)
        candidate = next(c for c in result["primary"]["candidate_conjunctions"] if c["candidate_id"] == "contextual_input")
        self.assertEqual(candidate["status"], "invalid")
        self.assertTrue(all(c["primary"]["multiplicity_input_p_value"] == 1 for c in result["primary"]["primary_components"] if c["candidate_id"] == "contextual_input"))

    def test_native_caps_are_retained_in_itt_not_treated_as_technical_errors(self):
        bundle = fixture()
        cohort = bundle["cohorts"]["input_fact_subspace_primary"]
        target_rows = [r for r in cohort["rows"] if r["condition"] == "subspace_forward_1"]
        for row in target_rows[:16]:
            row.update(generation_status="truncated", native_action=None)
        result = report.build_report(bundle)
        efficacy = next(c for c in result["primary"]["primary_components"] if c["primary_id"] == "subspace_efficacy")
        self.assertEqual(efficacy["n_pairs"], 32)
        self.assertEqual(efficacy["primary"]["effect"], .5)
        self.assertEqual(efficacy["completion_sensitive_effect_bounds"]["upper"], 1)
        self.assertEqual(efficacy["component_status"], "complete")

    def test_hidden_missing_row_and_changed_target_are_rejected(self):
        for mutation in (lambda b: b["cohorts"]["contextual_input_primary"]["rows"].pop(),
                         lambda b: b["cohorts"]["contextual_input_primary"]["rows"][0].update(target_action="DOWN")):
            bundle = fixture()
            mutation(bundle)
            with self.assertRaises(ValueError):
                report.build_report(bundle)

    def test_exact_paired_completion_interaction_bounds(self):
        rows = fixture()["cohorts"]["input_fact_subspace_primary"]["rows"]
        standard, alternate = deepcopy(rows), deepcopy(rows)
        for row in standard:
            if row["condition"] == "subspace_forward_1":
                row.update(generation_status="truncated", native_action=None)
        for row in alternate:
            if row["condition"] == "baseline":
                row.update(generation_status="malformed", native_action=None)
        ids = {r["pair_id"] for r in rows}
        value = report.descriptive_contrast(standard, alternate, "subspace_forward_1", "baseline", ids, {"wall": 1.0})
        self.assertEqual(value["standard"], {"itt_effect": 0, "completion_bounds": [0, 1]})
        self.assertEqual(value["alternate"], {"itt_effect": 1, "completion_bounds": [0, 1]})
        self.assertEqual(value["encoding_interaction"], {"itt_effect": 1, "completion_bounds": [-1, 1]})
        self.assertIsNone(value["p_values"])
        self.assertIsNone(value["confidence_intervals"])

    def test_mixed_descriptive_weights_and_no_posthoc_missing_subset(self):
        rows = fixture()["cohorts"]["contextual_input_primary"]["rows"]
        alternative = deepcopy(rows)
        for row in alternative:
            if row["condition"] == "layer10_natural" and row["family"] != "wall":
                row["native_action"] = "UP"
        weights = {"wall": 11/32, "key_possession": 11/32, "door_status": 10/32}
        value = report.descriptive_contrast(rows, alternative, "layer10_natural", "baseline", {r["pair_id"] for r in rows}, weights)
        self.assertEqual(value["alternate"]["itt_effect"], 11/32)
        self.assertEqual(value["encoding_interaction"]["itt_effect"], -21/32)
        bad = [r for r in alternative if r["pair_id"] != alternative[0]["pair_id"]]
        with self.assertRaises((ValueError, KeyError)):
            report.descriptive_contrast(rows, bad, "layer10_natural", "baseline", {r["pair_id"] for r in rows}, weights)

    def test_raw_identity_parse_cap_and_exact_json_types(self):
        job = {"job_id": "job", "seed": 42, "natural": True, "max_new_tokens": 4096,
               "recipient_token_ids": [1, 2], "target_action": "RIGHT", "original_action": "UP"}
        row = {**job, "spec_sha256": "sha", "status": "complete", "inference_configuration": {"head_mode": "full"},
            "input_token_sha256": report.fingerprint([1, 2]), "n_input_tokens": 2,
            "generated_token_ids": [3, 4], "generated_tokens": 2, "n_new_tokens": 2,
            "generated_text": '<|start|>assistant<|channel|>final<|message|>{"action":"RIGHT"}<|return|>',
            "final_text": '{"action":"RIGHT"}', "parsed_action": "RIGHT", "parse_status": "valid", "stop_reason": "return",
            "action":"RIGHT","endpoint":"native_final","completed":True,"ended_naturally":True,"truncated":False,
            "generation_status": "complete", "native_action": "RIGHT", "forced_q": dict(zip(report.ACTIONS, (.1, .2, .3, .4))), "forced_top_action": "RIGHT"}
        self.assertEqual(report._row_identity(row, job, "sha"), [])
        tool_call=deepcopy(row)
        tool_call.update(stop_reason="tool_call",generation_status="malformed",native_action=None,action=None,completed=False,ended_naturally=False)
        tool_call["generated_text"]=tool_call["generated_text"].replace("<|return|>","<|call|>")
        self.assertEqual(report._row_identity(tool_call,job,"sha"),[])
        bad = deepcopy(row)
        bad["natural"] = 1
        with self.assertRaisesRegex(ValueError, "frozen field"):
            report._row_identity(bad, job, "sha")
        row.update(stop_reason="token_limit", generation_status="truncated", native_action=None,
                   action=None,completed=False,ended_naturally=False,truncated=True,
                   generated_token_ids=[3]*4096, generated_tokens=4096, n_new_tokens=4096)
        self.assertEqual(report._row_identity(row, job, "sha"), [])
        row["native_action"] = "RIGHT"
        with self.assertRaisesRegex(ValueError, "status/action"):
            report._row_identity(row, job, "sha")

    def test_complete_only_gate_precedes_native_reads_and_cutoff_is_not_overridable(self):
        bundle = fixture()
        bundle.update(primary_ids={}, campaign_configuration={}, campaign_sha256="x", freeze_sha256="x")
        bundle["freeze"]["inference_cutoff"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        with tempfile.TemporaryDirectory() as directory, patch.object(report, "_contract", return_value=bundle), \
             patch.object(report, "_module_bindings", side_effect=AssertionError("outcome stage reached")):
            with self.assertRaisesRegex(ValueError, "no native outcomes read"):
                report.load_bundle(directory, "synthetic")
            with self.assertRaisesRegex(ValueError, "unavailable before"):
                report.load_bundle(directory, "synthetic", at_cutoff=True)

    def test_strict_hash_json_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "data.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                report.Inputs().read(path)
            path.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):
                report.Inputs().read(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                report.Inputs().read(path, "0"*64)
            stem = Path(directory)/"report"
            result = report.build_report(fixture())
            files = report.write_report(result, stem)
            self.assertEqual(len(files), 2)
            self.assertIn("One joint analysis", Path(str(stem)+".md").read_text())
            self.assertEqual(json.loads(Path(str(stem)+".json").read_text())["primary"]["confirmation_primary_family"]["family_size"], 4)
            with self.assertRaisesRegex(ValueError, "already exists"):
                report.write_report(result, stem)


if __name__ == "__main__":
    unittest.main()
