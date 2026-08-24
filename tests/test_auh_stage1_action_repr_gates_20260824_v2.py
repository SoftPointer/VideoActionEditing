#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


REPO = Path(__file__).resolve().parents[1]
METHOD = REPO / "methods" / "bernini_action_editing"
LAUNCHER = METHOD / "scripts" / "auh_stage1_action_repr_gates_20260824_v2.sh"
DOC = REPO / "md" / "action_editing" / "20260824_reward"
PREREG = DOC / "stage1_v2_preregistration.json"
ADDENDUM = DOC / "stage1_v2_source_lock_addendum.json"
POSTERIOR_ADDENDUM = DOC / "stage1_v2_posterior_identity_addendum.json"
MATCHED_NOISE_ADDENDUM = DOC / "stage1_v2_matched_noise_addendum.json"
G2A_SIX_ROUTE_ADDENDUM = DOC / "stage1_v2_g2a_six_route_addendum.json"
EXPLICIT_GAUSSIAN_ADDENDUM = (
    DOC / "stage1_v2_explicit_gaussian_authority_addendum.json"
)
G1_AUTHORITY_FIXTURE_ADDENDUM = (
    DOC / "stage1_v2_g1_authority_fixture_addendum.json"
)
DETERMINISTIC_VAE_AUTHORITY_ADDENDUM = (
    DOC / "stage1_v2_deterministic_vae_authority_addendum.json"
)
QUANTIZED_ENERGY_MATCH_ADDENDUM = (
    DOC / "stage1_v2_quantized_energy_match_addendum.json"
)
MANIFEST = DOC / "experiment_manifest.json"
G2A_TEST = REPO / "tests" / "test_action_repr_g2a_adapter_v1.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage1V2StaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        cls.addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
        cls.posterior_addendum = json.loads(
            POSTERIOR_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.matched_noise_addendum = json.loads(
            MATCHED_NOISE_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.g2a_addendum = json.loads(
            G2A_SIX_ROUTE_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.explicit_gaussian_addendum = json.loads(
            EXPLICIT_GAUSSIAN_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.g1_authority_fixture_addendum = json.loads(
            G1_AUTHORITY_FIXTURE_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.deterministic_vae_authority_addendum = json.loads(
            DETERMINISTIC_VAE_AUTHORITY_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.quantized_energy_match_addendum = json.loads(
            QUANTIZED_ENERGY_MATCH_ADDENDUM.read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def test_frozen_documents_and_manifest_are_hash_bound(self) -> None:
        self.assertEqual(
            sha256(PREREG),
            "294168e596212bd61e8d555e72702ceeeb993fb18c7fa7536a43d0b00ad592b3",
        )
        self.assertEqual(
            sha256(ADDENDUM),
            "1b1b4736a0925a080c7423ebb3f5358e0be3b5719df6b403840f6c221a570985",
        )
        self.assertEqual(
            sha256(POSTERIOR_ADDENDUM),
            "62e3ed49084a85d8d969dbd68f28c1659aceacedc19f62abcd4f7324dcd05228",
        )
        self.assertEqual(
            sha256(MATCHED_NOISE_ADDENDUM),
            "5cb3ab6350f8122b84860b0a99264334d2343137921196049e71d832588fe70c",
        )
        self.assertEqual(
            sha256(G2A_SIX_ROUTE_ADDENDUM),
            "35944dbad37d38148c8305c818207c2586960e9b0692fc94f8ee8952608d11d2",
        )
        self.assertEqual(
            sha256(EXPLICIT_GAUSSIAN_ADDENDUM),
            "dc52025c8bbfe349d8917e3711dadc688f0dec5b121d239a2765edbe3f566444",
        )
        self.assertEqual(
            sha256(G1_AUTHORITY_FIXTURE_ADDENDUM),
            "fea432a543760a2771c529f1d7ff59798dea27425a6a7e617e46defb9b975266",
        )
        self.assertEqual(
            sha256(DETERMINISTIC_VAE_AUTHORITY_ADDENDUM),
            "20f9c7a138a30d742113eca4c68c7b0c3a815d40d5753bb016b6b6c30959731f",
        )
        self.assertEqual(
            sha256(QUANTIZED_ENERGY_MATCH_ADDENDUM),
            "39a2879c35bdc0fc87c67f05adc11e5766f7dae61792c75f44653450b7ee04da",
        )
        self.assertEqual(
            sha256(MANIFEST),
            "c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632",
        )
        self.assertEqual(len(self.manifest["cases"]), 8)
        self.assertEqual(self.manifest["current_experiment_optimization_steps"], 0)
        self.assertFalse(self.prereg["optimizer_creation_authorized"])
        self.assertEqual(self.prereg["optimization_steps"], 0)
        self.assertEqual(
            self.addendum["source_preregistration_sha256"], sha256(PREREG)
        )
        self.assertEqual(
            self.posterior_addendum["canonical_preregistration"]["sha256"],
            sha256(PREREG),
        )
        self.assertEqual(
            self.posterior_addendum["prior_source_lock_addendum"]["sha256"],
            sha256(ADDENDUM),
        )
        self.assertFalse(
            self.posterior_addendum["runtime_boundary"]["optimizer_created"]
        )
        self.assertEqual(
            self.posterior_addendum["runtime_boundary"]["optimization_steps"], 0
        )
        self.assertEqual(
            self.matched_noise_addendum["prior_posterior_identity_addendum"][
                "sha256"
            ],
            sha256(POSTERIOR_ADDENDUM),
        )
        self.assertEqual(
            self.g2a_addendum["prior_matched_noise_addendum"]["sha256"],
            sha256(MATCHED_NOISE_ADDENDUM),
        )
        self.assertFalse(
            self.g2a_addendum["zero_optimizer_contract"]["optimizer_created"]
        )
        self.assertEqual(
            self.g2a_addendum["zero_optimizer_contract"]["optimization_steps"],
            0,
        )
        explicit = self.explicit_gaussian_addendum
        self.assertEqual(
            explicit["schema_version"],
            "bernini-action-repr-stage1-explicit-gaussian-authority-addendum-v1",
        )
        self.assertEqual(
            explicit["canonical_preregistration"]["sha256"], sha256(PREREG)
        )
        self.assertEqual(
            explicit["prior_source_lock_addendum"]["sha256"], sha256(ADDENDUM)
        )
        self.assertEqual(
            explicit["prior_posterior_identity_addendum"]["sha256"],
            sha256(POSTERIOR_ADDENDUM),
        )
        self.assertEqual(
            explicit["prior_matched_noise_addendum"]["sha256"],
            sha256(MATCHED_NOISE_ADDENDUM),
        )
        self.assertEqual(
            explicit["prior_g2a_six_route_addendum"]["sha256"],
            sha256(G2A_SIX_ROUTE_ADDENDUM),
        )
        self.assertFalse(explicit["zero_optimizer_contract"]["optimizer_created"])
        self.assertFalse(
            explicit["zero_optimizer_contract"][
                "optimizer_creation_authorized_by_this_addendum"
            ]
        )
        self.assertEqual(explicit["zero_optimizer_contract"]["optimization_steps"], 0)
        self.assertEqual(explicit["zero_optimizer_contract"]["parameter_updates"], 0)

        fixture = self.g1_authority_fixture_addendum
        self.assertEqual(
            fixture["schema_version"],
            "bernini-action-repr-stage1-g1-authority-fixture-addendum-v1",
        )
        self.assertEqual(
            fixture["canonical_preregistration"]["sha256"], sha256(PREREG)
        )
        self.assertEqual(
            fixture["prior_explicit_gaussian_authority_addendum"]["sha256"],
            sha256(EXPLICIT_GAUSSIAN_ADDENDUM),
        )
        self.assertFalse(fixture["runtime_boundary"]["optimizer_created"])
        self.assertEqual(fixture["runtime_boundary"]["optimization_steps"], 0)
        self.assertEqual(fixture["runtime_boundary"]["parameter_updates"], 0)
        self.assertTrue(fixture["runtime_boundary"]["test_fixture_only"])
        rerun = fixture["test_rerun_contract"]
        self.assertEqual(rerun["fresh_source_root"], "source_v2_5")
        self.assertEqual(rerun["fresh_stage_root"], "stage1_v2_5")
        self.assertTrue(rerun["pythonpath_must_include_source_root"])
        self.assertTrue(rerun["pythonpath_must_include_method_root"])
        self.assertTrue(rerun["full_methods_suite_required"])
        self.assertTrue(rerun["root_G2a_and_launcher_static_suite_required"])
        self.assertTrue(rerun["launcher_preflight_required"])
        self.assertTrue(rerun["single_case_target_G0_only_after_all_tests_pass"])

        deterministic = self.deterministic_vae_authority_addendum
        self.assertEqual(
            deterministic["schema_version"],
            "bernini-action-repr-stage1-deterministic-vae-authority-addendum-v1",
        )
        self.assertEqual(
            deterministic["canonical_preregistration"]["sha256"], sha256(PREREG)
        )
        self.assertEqual(
            deterministic["prior_explicit_gaussian_authority_addendum"]["sha256"],
            sha256(EXPLICIT_GAUSSIAN_ADDENDUM),
        )
        self.assertEqual(
            deterministic["prior_g1_authority_fixture_addendum"]["sha256"],
            sha256(G1_AUTHORITY_FIXTURE_ADDENDUM),
        )
        self.assertEqual(
            deterministic["deterministic_vae_authority_contract"],
            {
                "authority_kind": "rank0_local_strict_deterministic_vae_encode",
                "policy": (
                    "rank0_two_branch_vae_encode_in_local_strict_deterministic_"
                    "scope_with_exact_flag_restoration_v1"
                ),
                "producer_rank": 0,
                "encode_call_count": 2,
                "scope": "action_and_first_frame_repeat_encode_calls_only",
                "during_flags": {
                    "deterministic_algorithms_enabled": True,
                    "deterministic_algorithms_warn_only": False,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                },
                "restore_preexisting_flags_on_success_and_exception": True,
                "raw_action_noop_posterior_phase0_bit_exact_required": True,
                "sampled_clean_phase0_bit_exact_required": True,
                "phase0_match_atol": 0.0,
                "posterior_modified_after_encode": False,
                "posterior_copy_or_splice_used": False,
                "posterior_or_clean_latent_received_by_trainer": False,
                "no_posterior_or_absolute_clean_latent_persisted": True,
                "explicit_prepack_gaussian_contract_unchanged": True,
            },
        )
        self.assertEqual(
            deterministic["zero_optimizer_contract"],
            {
                "optimizer_created": False,
                "optimizer_creation_authorized_by_this_addendum": False,
                "optimization_steps": 0,
                "parameter_updates": 0,
                "checkpoint_or_lora_created": False,
            },
        )
        self.assertEqual(
            deterministic["ordered_execution_boundary"],
            {
                "fresh_source_root": "source_v2_6",
                "fresh_stage_root": "stage1_v2_6",
                "fresh_log_root": "logs/stage1_v2_6",
                "required_order": [
                    "AUH_source_tests_and_preflight",
                    "single_case_target_G0",
                    "eight_case_G1_target",
                    "production_WORLD4_six_route_G2a",
                    "target_T0_then_TP_optimizer_experiment",
                    "independent_G1_selfgen_before_any_selfgen_optimizer_experiment",
                ],
                "pythonpath_must_include_source_root": True,
                "pythonpath_must_include_method_root": True,
                "full_methods_suite_required": True,
                "root_G2a_and_launcher_static_suite_required": True,
                "launcher_preflight_required": True,
                "single_case_target_G0_only_after_all_tests_pass": True,
                "expand_to_G1_target_only_after_G0_pass": True,
                "run_production_G2a_only_after_G1_target_pass": True,
                "create_target_optimizer_only_after_G0_G1_target_and_G2a_pass": True,
            },
        )

        quantized = self.quantized_energy_match_addendum
        self.assertEqual(
            quantized["schema_version"],
            "bernini-action-repr-stage1-quantized-energy-match-addendum-v1",
        )
        self.assertEqual(
            quantized["canonical_preregistration"]["sha256"], sha256(PREREG)
        )
        self.assertEqual(
            quantized["prior_deterministic_vae_authority_addendum"]["sha256"],
            sha256(DETERMINISTIC_VAE_AUTHORITY_ADDENDUM),
        )
        self.assertEqual(
            quantized["quantized_energy_match_contract"],
            {
                "energy_definition": (
                    "sqrt(sum(all_block_residual_squared)/"
                    "number_of_residual_scalars)"
                ),
                "analytic_scale": (
                    "correct_energy_divided_by_wrong_action_donor_energy"
                ),
                "scale_lower_bound": 0.01,
                "scale_upper_bound": 100.0,
                "bracket_lower": "max(scale_lower_bound,analytic_scale/2)",
                "bracket_upper": "min(scale_upper_bound,analytic_scale*2)",
                "quantized_bracket_must_straddle_target": True,
                "calibration_compute_dtype": "torch.float64_on_cpu",
                "candidate_publication_dtype": "exact_original_donor_tensor_dtype",
                "calibration_iterations": 32,
                "candidate_selection_order": [
                    "absolute_relative_energy_error",
                    "absolute_scale_distance_from_analytic_scale",
                    "scale",
                ],
                "final_candidate_requantization_and_exact_energy_replay_required": True,
                "energy_match_rtol": 0.00002,
                "energy_match_rtol_changed": False,
                "maximum_energy_scale_changed": False,
                "wrong_action_donor_cycle_changed": False,
                "output_dtype_promotion_authorized": False,
                "optimizer_or_trainer_access": False,
            },
        )
        self.assertEqual(
            quantized["zero_optimizer_contract"],
            {
                "optimizer_created": False,
                "optimizer_creation_authorized_by_this_addendum": False,
                "optimization_steps": 0,
                "parameter_updates": 0,
                "checkpoint_or_lora_created": False,
            },
        )
        self.assertEqual(
            quantized["ordered_execution_boundary"],
            {
                "fresh_source_root": "source_v2_7",
                "fresh_stage_root": "stage1_v2_7",
                "fresh_log_root": "logs/stage1_v2_7",
                "required_order": [
                    "AUH_source_tests_and_preflight",
                    "single_case_target_G0",
                    "eight_case_G1_target",
                    "production_WORLD4_six_route_G2a",
                    "target_T0_then_TP_optimizer_experiment",
                    "independent_G1_selfgen_before_any_selfgen_optimizer_experiment",
                ],
                "pythonpath_must_include_source_root": True,
                "pythonpath_must_include_method_root": True,
                "full_methods_suite_required": True,
                "root_G2a_and_launcher_static_suite_required": True,
                "launcher_preflight_required": True,
                "single_case_target_G0_only_after_all_tests_pass": True,
                "expand_to_G1_target_only_after_G0_pass": True,
                "run_production_G2a_only_after_G1_target_pass": True,
                "create_target_optimizer_only_after_G0_G1_target_and_G2a_pass": True,
                "reuse_source_v2_6_representation_outputs_as_canonical_v2_7_gate_evidence": False,
            },
        )

        gaussian = explicit["explicit_prepack_gaussian_contract"]
        self.assertEqual(
            gaussian["seed_binding"],
            "domain_plus_base_seed_plus_case_id_plus_instruction_sha256_not_control_video_sha256",
        )
        self.assertTrue(gaussian["same_case_correct_shuffle_reverse_authority_required"])
        self.assertFalse(gaussian["recovered_from_x_or_velocity"])

        boundary = explicit["ordered_execution_boundary"]
        self.assertEqual(boundary["fresh_source_root"], "source_v2_4")
        self.assertEqual(boundary["fresh_stage_root"], "stage1_v2_4")
        self.assertEqual(
            boundary["required_order"],
            [
                "AUH_source_tests_and_preflight",
                "single_case_target_G0",
                "eight_case_G1_target",
                "production_WORLD4_six_route_G2a",
                "target_T0_then_TP_optimizer_experiment",
                "independent_G1_selfgen_before_any_selfgen_optimizer_experiment",
            ],
        )
        self.assertTrue(boundary["expand_to_G1_target_only_after_G0_pass"])
        self.assertTrue(boundary["run_production_G2a_only_after_G1_target_pass"])
        self.assertTrue(
            boundary["create_target_optimizer_only_after_G0_G1_target_and_G2a_pass"]
        )

        vendor = explicit["pinned_vendor_authority"]
        self.assertEqual(vendor["bernini_revision"], "2d2b4591")
        self.assertEqual(vendor["relative_path"], "bernini/training/data.py")
        self.assertEqual(
            vendor["file_sha256"],
            "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
        )
        self.assertEqual(
            vendor["pack_vae_latents_source_sha256"],
            "445893fee2cca1f745265cea857740937f338a04b67e9f895fef943948c49c9f",
        )
        self.assertEqual(
            vendor["process_renderer_sample_source_sha256"],
            "9e8532898267ea167f0776a71a30233cbfada4f94132e0b546f1740115ee372e",
        )

    def test_all_runtime_sources_are_byte_pinned(self) -> None:
        pins = self.prereg["source_pins"]
        corrected = self.addendum["corrected_source_hash_pins"]
        posterior_corrected = self.posterior_addendum[
            "corrected_source_hash_pins"
        ]
        matched_corrected = self.matched_noise_addendum[
            "corrected_source_hash_pins"
        ]
        g2a_corrected = self.g2a_addendum["corrected_source_hash_pins"]
        explicit_corrected = self.explicit_gaussian_addendum[
            "corrected_source_hash_pins"
        ]
        fixture_corrected = self.g1_authority_fixture_addendum[
            "corrected_source_hash_pins"
        ]
        deterministic_corrected = self.deterministic_vae_authority_addendum[
            "corrected_source_hash_pins"
        ]
        quantized_corrected = self.quantized_energy_match_addendum[
            "corrected_source_hash_pins"
        ]
        self.assertEqual(
            set(explicit_corrected),
            {
                "materialize_decoded_middle_action_repr_v1.py",
                "test_materialize_decoded_middle_action_repr_v1.py",
                "materialize_g1_middle_control_cohort_v1.py",
                "test_materialize_g1_middle_control_cohort_v1.py",
            },
        )
        self.assertEqual(
            set(fixture_corrected),
            {"test_materialize_g1_middle_control_cohort_v1.py"},
        )
        fixture_pin = fixture_corrected[
            "test_materialize_g1_middle_control_cohort_v1.py"
        ]
        self.assertEqual(
            fixture_pin["old_sha256"],
            "d8142b1aa49a60142b49a073f5dbffde3a129ed702d02f252421c48a401c0d8b",
        )
        self.assertEqual(
            fixture_pin["new_sha256"],
            "f7787e28fe9e0b37abebbe4dcab9a928125717f700e2876636a81ea581c8f724",
        )
        self.assertEqual(
            {
                name: (pin["old_sha256"], pin["new_sha256"])
                for name, pin in deterministic_corrected.items()
            },
            {
                "materialize_decoded_middle_action_repr_v1.py": (
                    "a298342ae8a19906e651bbcfdc4f9b125bab93faeb1fa3682eff897ca3b280d2",
                    "f3fa0138ffcff997a604567c0951bf7f9aba74ae6cb66acb943eddef2aa6a1ac",
                ),
                "test_materialize_decoded_middle_action_repr_v1.py": (
                    "2cd70a81edebd9d4011a4b0e020309d2bf0cccac56f61f0db1c54f5cbbc73e36",
                    "9834d20a51d53b4608f0e57392d21bfcc6e7d08db93973cfd23cf73ea015645d",
                ),
                "materialize_g1_middle_control_cohort_v1.py": (
                    "b397079aff3bcd237dedb1d5b135f50b520a36b809d7a387e8e3239890311d0c",
                    "b7a1d87c87bfa371065b141a9535a1f72331db037940e5e9d911e4efb7eb9b9b",
                ),
                "test_materialize_g1_middle_control_cohort_v1.py": (
                    "f7787e28fe9e0b37abebbe4dcab9a928125717f700e2876636a81ea581c8f724",
                    "b9bd4f18d81701f5e2476c7e118a9208a9dd7b1af0b9604c279bf346530ae1e5",
                ),
            },
        )
        self.assertEqual(
            {
                name: (pin["old_sha256"], pin["new_sha256"])
                for name, pin in quantized_corrected.items()
            },
            {
                "materialize_g1_middle_control_cohort_v1.py": (
                    "b7a1d87c87bfa371065b141a9535a1f72331db037940e5e9d911e4efb7eb9b9b",
                    "ad7e69d058570195ab790eb7500bfedf19ac7f80637dc52913e2a91a439d9d0c",
                ),
                "test_materialize_g1_middle_control_cohort_v1.py": (
                    "b9bd4f18d81701f5e2476c7e118a9208a9dd7b1af0b9604c279bf346530ae1e5",
                    "cc14a6007d7ccb8fc3c43121cd26d6371296972254c4735d74560b7e7ea09287",
                ),
            },
        )
        self.assertEqual(
            {
                name: pin["new_sha256"]
                for name, pin in explicit_corrected.items()
            },
            {
                "materialize_decoded_middle_action_repr_v1.py": (
                    "a298342ae8a19906e651bbcfdc4f9b125bab93faeb1fa3682eff897ca3b280d2"
                ),
                "test_materialize_decoded_middle_action_repr_v1.py": (
                    "2cd70a81edebd9d4011a4b0e020309d2bf0cccac56f61f0db1c54f5cbbc73e36"
                ),
                "materialize_g1_middle_control_cohort_v1.py": (
                    "b397079aff3bcd237dedb1d5b135f50b520a36b809d7a387e8e3239890311d0c"
                ),
                "test_materialize_g1_middle_control_cohort_v1.py": (
                    "d8142b1aa49a60142b49a073f5dbffde3a129ed702d02f252421c48a401c0d8b"
                ),
            },
        )
        for name, expected in pins.items():
            if not name.startswith("test_"):
                path = METHOD / name
            elif name == G2A_TEST.name:
                path = G2A_TEST
            else:
                path = METHOD / "tests" / name
            if name in corrected:
                self.assertEqual(corrected[name]["old_sha256"], expected, name)
                expected = corrected[name]["new_sha256"]
            if name in posterior_corrected:
                self.assertEqual(
                    posterior_corrected[name]["old_sha256"], expected, name
                )
                expected = posterior_corrected[name]["new_sha256"]
            if name in matched_corrected:
                self.assertEqual(matched_corrected[name]["old_sha256"], expected, name)
                expected = matched_corrected[name]["new_sha256"]
            relative = path.relative_to(REPO).as_posix()
            if relative in g2a_corrected:
                self.assertEqual(g2a_corrected[relative]["old_sha256"], expected, name)
                expected = g2a_corrected[relative]["new_sha256"]
            if name in explicit_corrected:
                self.assertEqual(
                    explicit_corrected[name]["old_sha256"], expected, name
                )
                expected = explicit_corrected[name]["new_sha256"]
            if name in fixture_corrected:
                self.assertEqual(
                    fixture_corrected[name]["old_sha256"], expected, name
                )
                expected = fixture_corrected[name]["new_sha256"]
            if name in deterministic_corrected:
                self.assertEqual(
                    deterministic_corrected[name]["old_sha256"], expected, name
                )
                expected = deterministic_corrected[name]["new_sha256"]
            if name in quantized_corrected:
                self.assertEqual(
                    quantized_corrected[name]["old_sha256"], expected, name
                )
                expected = quantized_corrected[name]["new_sha256"]
            self.assertEqual(sha256(path), expected, name)
        for name, expected in self.prereg["middle_runtime_dependency_pins"].items():
            self.assertEqual(sha256(METHOD / name), expected, name)
        self.assertTrue(
            {
                "full30_action_learning_v1.py",
                "self_generated_action_quotient_v1.py",
                "self_generated_action_preservation_v2.py",
                "tools/build_renderer_dataset.py",
                "tools/materialize_vae.py",
            }.issubset(self.prereg["middle_runtime_dependency_pins"])
        )
        flow_extractor = METHOD / "extract_anchor_raft_flow_v1.py"
        self.assertEqual(
            sha256(flow_extractor),
            self.addendum["additional_source_hash_pins"][flow_extractor.name],
        )
        raft = self.addendum["runtime_weight_pins"]["torchvision_raft_large_C_T_SKHT_V2"]
        self.assertEqual(
            raft["sha256"],
            "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322",
        )
        for relative, expected in self.g2a_addendum[
            "additional_source_hash_pins"
        ].items():
            self.assertEqual(sha256(REPO / relative), expected, relative)

    def test_launcher_parses_and_contains_no_placeholder(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("STAGE1_V2_SOURCE_LOCK_ADDENDUM_SHA256", self.script)
        self.assertIn(f"expected_prereg_sha={sha256(PREREG)}", self.script)
        self.assertIn(f"expected_addendum_sha={sha256(ADDENDUM)}", self.script)
        self.assertIn(
            "expected_posterior_identity_addendum_sha="
            f"{sha256(POSTERIOR_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_matched_noise_addendum_sha="
            f"{sha256(MATCHED_NOISE_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_g2a_six_route_addendum_sha="
            f"{sha256(G2A_SIX_ROUTE_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_explicit_gaussian_authority_addendum_sha="
            f"{sha256(EXPLICIT_GAUSSIAN_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_g1_authority_fixture_addendum_sha="
            f"{sha256(G1_AUTHORITY_FIXTURE_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_deterministic_vae_authority_addendum_sha="
            f"{sha256(DETERMINISTIC_VAE_AUTHORITY_ADDENDUM)}",
            self.script,
        )
        self.assertIn(
            "expected_quantized_energy_match_addendum_sha="
            f"{sha256(QUANTIZED_ENERGY_MATCH_ADDENDUM)}",
            self.script,
        )
        self.assertIn('source_root="$experiment_root/source_v2_7"', self.script)
        self.assertIn('stage_root="$experiment_root/stage1_v2_7"', self.script)
        self.assertIn('log_root="$experiment_root/logs/stage1_v2_7"', self.script)
        self.assertIn('export PYTHONPATH="$source_root:$method_root"', self.script)
        self.assertNotIn('source_root="$experiment_root/source_v2_6"', self.script)
        self.assertNotIn('stage_root="$experiment_root/stage1_v2_6"', self.script)
        self.assertNotIn('source_root="$experiment_root/source_v2_5"', self.script)
        self.assertNotIn('stage_root="$experiment_root/stage1_v2_5"', self.script)
        self.assertNotIn('source_root="$experiment_root/source_v2_4"', self.script)
        self.assertNotIn('stage_root="$experiment_root/stage1_v2_4"', self.script)
        self.assertNotIn('source_root="$experiment_root/source_v2_3"', self.script)
        self.assertNotIn('stage_root="$experiment_root/stage1_v2_3"', self.script)

    def test_only_explicit_zero_update_stage1_commands_exist(self) -> None:
        labels = set(
            re.findall(
                r"^  ([a-z][a-z0-9-]*)\)\n",
                self.script,
                flags=re.MULTILINE,
            )
        )
        expected = {
            "preflight",
            "launch-target-canary",
            "target-controls",
            "launch-target-repr",
            "worker-target-repr",
            "target-g1",
            "target-eval",
            "target-admission",
            "g2a-cpu-api-audit",
            "launch-g2a-production",
            "worker-g2a-production",
            "g2a-production-status",
        }
        self.assertTrue(expected.issubset(labels))
        self.assertFalse(any("train" in label for label in labels))
        self.assertNotRegex(self.script, r"\b(?:sbatch|scancel|squeue)\b")
        self.assertNotIn("torch.optim", self.script)
        self.assertNotIn("optimizer.step", self.script)
        self.assertNotIn(".backward(", self.script)

    def test_sequence_srun_memory_and_g2a_fail_closed_contract(self) -> None:
        self.assertIn(
            "launch_target_repr 0be6494dfac3 correct \"${1:-147881}\"",
            self.script,
        )
        self.assertIn("require_canary_complete", self.script)
        self.assertNotIn("--overlap", self.script)
        self.assertIn('srun --jobid="$job" --exclusive --exact', self.script)
        self.assertIn('--nodelist="$node"', self.script)
        self.assertIn(
            'done < <(jq -r \'.middle_runtime_dependency_pins | to_entries[]',
            self.script,
        )
        self.assertIn("--gres=gpu:mi210:4", self.script)
        self.assertIn("--mem=0", self.script)
        self.assertIn("memory_guard_preflight", self.script)
        self.assertIn("SLURM_MEM_PER_NODE:-}", self.script)
        self.assertIn("slurm_mem_per_node_65536_cgroup_max_unbounded", self.script)
        self.assertIn("memory_watchdog_headroom_gib: 2", self.script)
        self.assertIn("--admission-scope target", self.script)
        self.assertIn("flow/cohort_receipt.json", self.script)
        self.assertIn("middle/cohort_receipt.json", self.script)
        self.assertIn("g1_selfgen_status", self.script)
        self.assertIn("require_target_g1_pass", self.script)
        self.assertIn("worker-g2a-production", self.script)
        self.assertIn("audit_action_repr_g2a_world4_v1.py", self.script)
        self.assertIn(
            "correct,zero,temporal_shuffle,reverse,incomplete,wrong_action",
            self.script,
        )
        self.assertIn("production WORLD4 G2a requires passed G1_target", self.script)
        self.assertIn(
            'explicit_gaussian_authority_addendum="$source_root/'
            'stage1_v2_explicit_gaussian_authority_addendum.json"',
            self.script,
        )
        self.assertIn(
            'g1_authority_fixture_addendum="$source_root/'
            'stage1_v2_g1_authority_fixture_addendum.json"',
            self.script,
        )
        self.assertIn(
            'deterministic_vae_authority_addendum="$source_root/'
            'stage1_v2_deterministic_vae_authority_addendum.json"',
            self.script,
        )
        self.assertIn(
            'quantized_energy_match_addendum="$source_root/'
            'stage1_v2_quantized_energy_match_addendum.json"',
            self.script,
        )
        self.assertIn(
            ".deterministic_vae_authority_contract.phase0_match_atol == 0",
            self.script,
        )
        self.assertIn(
            ".quantized_energy_match_contract.energy_match_rtol == 0.00002",
            self.script,
        )
        self.assertIn(
            ".source_lock.explicit_gaussian_authority_addendum_sha256",
            self.script,
        )
        self.assertIn(
            "explicit_gaussian_authority_addendum_sha256: "
            "$explicit_gaussian_authority_addendum_sha256",
            self.script,
        )
        self.assertIn(
            ".source_lock.g1_authority_fixture_addendum_sha256",
            self.script,
        )
        self.assertIn(
            "g1_authority_fixture_addendum_sha256: "
            "$g1_authority_fixture_addendum_sha256",
            self.script,
        )
        self.assertIn(
            ".source_lock.deterministic_vae_authority_addendum_sha256",
            self.script,
        )
        self.assertIn(
            "deterministic_vae_authority_addendum_sha256: "
            "$deterministic_vae_authority_addendum_sha256",
            self.script,
        )
        self.assertIn(
            ".source_lock.quantized_energy_match_addendum_sha256",
            self.script,
        )
        self.assertIn(
            "quantized_energy_match_addendum_sha256: "
            "$quantized_energy_match_addendum_sha256",
            self.script,
        )
        self.assertIn("bernini/training/data.py", self.script)
        self.assertIn(
            "expected_bernini_data_sha="
            "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
            self.script,
        )
        self.assertIn(
            "expected_pack_vae_latents_source_sha="
            "445893fee2cca1f745265cea857740937f338a04b67e9f895fef943948c49c9f",
            self.script,
        )
        self.assertIn(
            "expected_process_renderer_sample_source_sha="
            "9e8532898267ea167f0776a71a30233cbfada4f94132e0b546f1740115ee372e",
            self.script,
        )
        self.assertIn(
            ".ordered_execution_boundary.required_order",
            self.script,
        )
        self.assertIn(
            "[stage1-v2.7] PREFLIGHT_PASS cases=8 rank0_posterior=true "
            "explicit_prepack_gaussian=true deterministic_vae=true "
            "quantized_energy=true control_matched_gaussian=true",
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
