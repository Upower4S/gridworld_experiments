"""Synthetic source-only profile derivation; no outcomes/model/files needed."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from src.activation_causality import frozen_contextual_input as contextual
from src.activation_causality import frozen_subspace as subspace
from src.activation_causality.confirmation_profiles import derive_candidate_configuration
from src.activation_causality.confirmation_sources import existing_comparison_origin
from src.activation_causality.contextual_input import build_contextual_input_spec
from src.activation_causality.environments import generate_pairs
from src.activation_causality.input_sites import build_input_site_spec
from src.activation_causality.lexical_generalization import CANONICAL, renderer_manifest
from src.activation_causality.registry import fingerprint
from src.activation_causality.subspace_protocol import build_subspace_spec

from test_activation_causality_frozen_contextual_input import Tokenizer
from test_activation_causality_frozen_subspace import basis


def bundle(spec):
    source = {"experiment_id": spec["experiment_id"], "specification": spec, "source_spec_sha256": "a" * 64,
              "source_result_hashes": {f"results/{spec['experiment_id']}/{j['job_id']}.json": fingerprint(j["job_id"])
                                       for j in spec["jobs"]}, "n_jobs": len(spec["jobs"])}
    source["source_result_manifest_sha256"] = fingerprint(source["source_result_hashes"])
    source["independent_audit"] = {"path": "outputs/campaign/reviews/source_audit.json", "sha256": "b" * 64,
        "input_files_sha256": {f"outputs/campaign/specs/{spec['experiment_id']}.json": source["source_spec_sha256"],
                               **{"outputs/campaign/" + k: v for k, v in source["source_result_hashes"].items()}}}
    return source


def descriptor(source, protocol=contextual.PROTOCOL, layer=0, family=None, rank=None, dose=1.0):
    label = "0p5" if dose == 0.5 else "1"
    return {"profile_id": f"contextual_l{layer}" if protocol == contextual.PROTOCOL else f"subspace_{family}_l0_r{rank}_dose{label}",
            "protocol_kind": protocol, "source_experiment_id": source["experiment_id"],
            "source_spec_sha256": source["source_spec_sha256"], "layer": layer, "site": "resid_post",
            "family": family, "rank": rank, "selected_dose": dose}


def origins(source, profile):
    if profile["protocol_kind"] == contextual.PROTOCOL:
        treatment = f"layer{profile['layer']}_natural"
        existing = existing_comparison_origin(source, treatment, f"layer{profile['layer']}_same_fact")
        prospective = {"kind": "prospective_confirmation_control", "provenance": {
            key: value for key, value in existing["provenance"].items() if key != "source_comparison_id"}}
        prospective["provenance"].update(discovery_measured=False, treatment=treatment,
            control=f"layer{profile['layer']}_natural_norm_random", addendum_path="reviews/norm_addendum.json",
            addendum_sha256="c" * 64, adapter_path="src/activation_causality/frozen_contextual_input.py", adapter_sha256="d" * 64)
        return {contextual.PRIMARY_IDS[0]: existing, contextual.PRIMARY_IDS[1]: prospective}
    label = "0p5" if profile["selected_dose"] == 0.5 else "1"
    treatment = "subspace_forward_" + label
    return {subspace.PRIMARY_IDS[0]: existing_comparison_origin(source, treatment, "baseline"),
            subspace.PRIMARY_IDS[1]: existing_comparison_origin(source, treatment, "projected_norm_matched_random_subspace_" + label)}


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pairs = generate_pairs(6, 20261201, "discovery", families=contextual.FAMILIES)
        cls.context = bundle(build_contextual_input_spec(pairs, Tokenizer(), "contextual_input_native_pilot_v1", layers=(0, 10)))
        cls.subspaces = {}
        for family, short in (("wall", "wall"), ("key_possession", "key"), ("door_status", "door")):
            pairs = generate_pairs(4, 20261202, "discovery", families=(family,))
            source = build_input_site_spec(pairs, Tokenizer(), "input_subspace_eval_sources_v1", layers=(0,), natural=True, cap=4096)
            for rank in (1, 4):
                metadata = basis(family, rank)
                derived = build_subspace_spec(source, metadata, metadata["basis_path"], metadata["basis_file_sha256"],
                    f"subspace_{short}_l0_r{rank}_native_v1", natural=True, cap=4096)
                cls.subspaces[family, rank] = bundle(derived)

    def derive(self, source=None, profile=None, n=32, requested_origins=None, **kwargs):
        source = source or self.context
        profile = profile or descriptor(source)
        arguments = {"candidate_id": "selected_profile", "n_layouts": n, "dataset_seed": 20260912,
                     "tokenizer_metadata": contextual._tokenizer_metadata(Tokenizer()),
                     "renderer_binding": renderer_manifest(CANONICAL), "require_joint_renderer_eligibility": True,
                     "eligibility_rule": {"identifier": "complete_joint_controls_v1", "sha256": "e" * 64},
                     "primary_origins": requested_origins or origins(source, profile)}
        arguments.update(kwargs)
        return derive_candidate_configuration(profile, source, **arguments)

    def test_context_both_layers_sizes_and_adapter_exact_acceptance(self):
        for layer in (0, 10):
            for n in (32, 64, 128):
                result = self.derive(profile=descriptor(self.context, layer=layer), n=n)
                self.assertEqual(contextual._configuration(result, Tokenizer()), result)
                self.assertEqual(result["condition_inventory"], contextual.condition_inventory(layer))
                self.assertEqual(result["family_layout_counts"], contextual.QUOTAS[n])
                self.assertEqual(result["family_weights"], {f: contextual.QUOTAS[n][f] / n for f in contextual.FAMILIES})
                self.assertEqual(set(result["primary_comparisons"]), set(contextual.PRIMARY_IDS))
                for row in result["primary_comparisons"].values():
                    self.assertEqual(row["expected_n_layouts"], n)
                    self.assertEqual(row["expected_n_layouts_by_family"], contextual.QUOTAS[n])
                    self.assertEqual(row["family_weights"], result["family_weights"])
                    self.assertEqual(row["treatment"], f"layer{layer}_natural")
                self.assertFalse(result["primary_comparisons"][contextual.PRIMARY_IDS[1]]["origin"]["provenance"]["discovery_measured"])

    def test_all_six_basis_profiles_both_doses_exact_adapter_acceptance(self):
        for (family, rank), source in self.subspaces.items():
            for dose in (0.5, 1.0):
                profile = descriptor(source, subspace.PROTOCOL, family=family, rank=rank, dose=dose)
                result = self.derive(source, profile, n=64)
                normalized, metadata = subspace._configuration(result, source["specification"]["basis"])
                self.assertEqual(normalized, result)
                self.assertEqual(metadata, result["basis_metadata"])
                self.assertEqual(result["basis_metadata_fingerprint"], fingerprint(metadata))
                self.assertEqual(result["condition_inventory"], list(subspace.CONDITIONS))
                self.assertEqual(result["selected_dose"], dose)
                self.assertEqual(result["doses"], [0.5, 1.0])
                self.assertNotIn("families", result)
                self.assertNotIn("family_weights", result)
                for binding in result["primary_comparisons"].values():
                    self.assertEqual(binding["expected_n_layouts"], 64)
                    self.assertEqual(binding["origin"]["kind"], "existing_source_comparison")

    def test_source_file_manifest_audit_and_basis_provenance_preserved(self):
        result = self.derive()
        self.assertEqual(result["source_result_hashes"], self.context["source_result_hashes"])
        self.assertEqual(result["source_spec_sha256"], "a" * 64)
        self.assertEqual(result["source_spec_fingerprint"], fingerprint(self.context["specification"]))
        self.assertNotEqual(result["source_spec_sha256"], result["source_spec_fingerprint"])
        self.assertEqual(result["source_independent_audit_input_files_sha256"], self.context["independent_audit"]["input_files_sha256"])
        self.assertEqual(result["source_profile_descriptor_sha256"], fingerprint(result["source_profile_descriptor"]))

    def test_outcome_fields_never_read_copied_or_used(self):
        class Forbidden:
            def __deepcopy__(self, memo):
                raise AssertionError("outcome payload accessed")
        source = dict(self.context)
        source.update(rows=Forbidden(), mean_seconds_per_job=Forbidden(), generation_status_counts=Forbidden())
        result = self.derive(source)
        self.assertEqual(result, self.derive())
        self.assertFalse(result["source_outcome_rows_read_by_derivation"])
        self.assertNotIn("rows", result)
        self.assertNotIn("mean_seconds_per_job", result)

    def test_descriptor_unknown_fields_and_drift_rejected(self):
        for mutation in (lambda p: p.update(metrics={"effect": 1}), lambda p: p.update(layer=15),
                         lambda p: p.update(profile_id="contextual_l10"),
                         lambda p: p.update(rank=1), lambda p: p.update(family="wall"),
                         lambda p: p.update(selected_dose=0.5), lambda p: p.update(selected_dose=1),
                         lambda p: p.update(source_spec_sha256="f" * 64)):
            profile = descriptor(self.context)
            mutation(profile)
            with self.assertRaises(ValueError):
                self.derive(profile=profile)

    def test_unmeasured_origin_cannot_be_imputed_or_renamed(self):
        base = origins(self.context, descriptor(self.context))
        for mutation in (lambda o: o[contextual.PRIMARY_IDS[1]]["provenance"].update(discovery_measured=True),
                         lambda o: o[contextual.PRIMARY_IDS[1]]["provenance"].update(source_comparison_id="invented"),
                         lambda o: o[contextual.PRIMARY_IDS[1]]["provenance"].update(control="layer0_random"),
                         lambda o: o[contextual.PRIMARY_IDS[1]]["provenance"].update(adapter_path="src/other.py"),
                         lambda o: o[contextual.PRIMARY_IDS[1]].update(kind="existing_source_comparison"),
                         lambda o: o[contextual.PRIMARY_IDS[0]]["provenance"].update(source_comparison_id="invented"),
                         lambda o: o[contextual.PRIMARY_IDS[0]]["provenance"].update(source_result_manifest_sha256="f" * 64),
                         lambda o: o.update(arbitrary=deepcopy(o[contextual.PRIMARY_IDS[0]]))):
            supplied = deepcopy(base)
            mutation(supplied)
            with self.assertRaises(ValueError):
                self.derive(requested_origins=supplied)

    def test_manifest_missing_job_or_false_audit_binding_rejected(self):
        for mutation in (lambda s: s["source_result_hashes"].pop(next(iter(s["source_result_hashes"]))),
                         lambda s: s.update(n_jobs=s["n_jobs"] - 1),
                         lambda s: s["independent_audit"]["input_files_sha256"].pop("outputs/campaign/" + next(iter(s["source_result_hashes"]))),
                         lambda s: s["independent_audit"].update(path="../escape")):
            source = deepcopy(self.context)
            mutation(source)
            with self.assertRaises(ValueError):
                self.derive(source)

    def test_contextual_producer_raw_only_audit_map_is_preserved(self):
        source = deepcopy(self.context)
        source["independent_audit"]["input_files_sha256"].pop("outputs/campaign/specs/contextual_input_native_pilot_v1.json")
        result = self.derive(source)
        self.assertEqual(result["source_spec_sha256"], source["source_spec_sha256"])
        self.assertEqual(result["source_independent_audit_input_files_sha256"], source["independent_audit"]["input_files_sha256"])
        self.assertEqual(len(result["source_independent_audit_input_files_sha256"]), 132)

    def test_source_intervention_and_inventory_drift_rejected_after_rehash(self):
        for mutation in (lambda s: s["jobs"][6]["intervention"].update(dose=0.5),
                         lambda s: s["jobs"][6]["intervention"].update(layers=[1]),
                         lambda s: s["jobs"][6].update(max_new_tokens=2048),
                         lambda s: s["jobs"][6]["selection"].update(scope="last_tokens"),
                         lambda s: s["jobs"].pop(),
                         lambda s: s["jobs"][6]["metadata"].update(representation_kind="arbitrary")):
            spec = deepcopy(self.context["specification"])
            mutation(spec)
            for job in spec["jobs"]:
                job["job_id"] = fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24]
            with self.assertRaises(ValueError):
                self.derive(bundle(spec))

    def test_basis_geometry_file_binding_and_audit_flag_drift_rejected(self):
        original = self.subspaces["wall", 1]
        for mutation in (lambda s: s["basis"].update(rank=4),
                         lambda s: s["basis"].update(basis_file_sha256="f" * 64),
                         lambda s: s["jobs"][6]["intervention"].update(audit_projected_norm=False),
                         lambda s: s["jobs"][9]["intervention"].update(random_basis=True)):
            spec = deepcopy(original["specification"])
            mutation(spec)
            for job in spec["jobs"]:
                job["job_id"] = fingerprint({k: v for k, v in job.items() if k != "job_id"})[:24]
            source = bundle(spec)
            profile = descriptor(source, subspace.PROTOCOL, family="wall", rank=1, dose=0.5)
            with self.assertRaises(ValueError):
                self.derive(source, profile)

    def test_unique_existing_comparison_and_whole_source_eligibility_required(self):
        spec = deepcopy(self.context["specification"])
        target = next(c for c in spec["comparisons"] if c["treatment"] == "layer0_natural" and c["control"] == "layer0_same_fact")
        supplied = origins(self.context, descriptor(self.context))
        for mutation in (lambda s, c: s["comparisons"].append(deepcopy(c)),
                         lambda s, c: c.update(eligible_pair_ids=c["eligible_pair_ids"][:-1]),
                         lambda s, c: c.update(expected_n_pairs=5)):
            modified = deepcopy(spec)
            comparison = next(c for c in modified["comparisons"] if c["id"] == target["id"])
            mutation(modified, comparison)
            with self.assertRaises(ValueError):
                self.derive(bundle(modified), requested_origins=supplied)

    def test_exact_renderer_and_tokenizer_json_types(self):
        renderer = renderer_manifest(CANONICAL)
        renderer["custom_renderers_allowed"] = 0
        with self.assertRaises(ValueError):
            self.derive(renderer_binding=renderer)
        tokenizer = contextual._tokenizer_metadata(Tokenizer())
        tokenizer["add_special_tokens"] = 0
        with self.assertRaises(ValueError):
            self.derive(tokenizer_metadata=tokenizer)

    def test_no_files_tokenization_or_model_queries(self):
        with patch("builtins.open", side_effect=AssertionError("no IO")), patch.object(Tokenizer, "encode", side_effect=AssertionError("no encode")):
            result = self.derive()
        self.assertEqual(result["max_new_tokens"], 4096)
        self.assertEqual(result["generation_seeds"], [42])
        self.assertEqual(result["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
