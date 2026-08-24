#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_legacy_infer_fork_rng_wrapper_v1 as wrapper
import validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1 as validator
from tools import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as package


class E00LegacyDiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = copy.deepcopy(validator.load_spec())

    def legacy_argv(self, role: str) -> list[str]:
        no_observer = role == validator.ARM_ROLES[0]
        source = "/data/source.mp4"
        anchor = "/package/source-frame0-static.mp4" if no_observer else "/data/action-anchor.mp4"
        return [
            "--source-video", source,
            "--anchor-video", anchor,
            "--output", "/out/result.mp4",
            "--transport", validator.PURE_QK_TRANSPORT,
            "--transport-steps", "0" if no_observer else "40",
            "--arm", "AQK_IID1",
            "--initial-noise-proposal-mode", "keyed_only",
            "--anchor-state-mode", "clean_noised",
        ]

    def native_receipt(self, role: str, output_sha: str) -> dict:
        no_observer = role == validator.ARM_ROLES[0]
        steps = 0 if no_observer else 40
        capture = 0 if no_observer else validator.EXPECTED_CAPTURE
        replay = 0 if no_observer else validator.EXPECTED_REPLAY
        prompts = self.spec["data_and_prompt_contract"]
        freeze = {
            "base_frozen": True,
            "trainable_parameter_tensors": 0,
            "trainable_parameter_elements": 0,
            "lora_module_count": 0,
            "adapter_modules_absent": True,
        }
        trace = {
            "candidate_counts": [1] * 40,
            "configured_early_candidate_count": 5,
            "initial_noise_proposal_mode": "keyed_only",
            "anchor_initial_gaussian_used_at_step0_candidate0": False,
            "anchor_action_reward_used_for_sga": False,
            "sga_weights_forced_to_anchor_candidate0": False,
            "anchor_model_forwards": 0 if no_observer else 80,
            "anchor_candidate_cells": steps,
            "target_raw_cfg_forwards": 80,
            "source_raw_cfg_forwards": 80,
            "target_model_forwards": 80,
            "source_model_forwards": 80,
            "anchor_action_noop_attention_observed_without_transport": False,
            "target_owned_qk_route_v14r2": not no_observer,
            "anchor_temporal_attention_kernel_contrast": not no_observer,
            "anchor_temporal_kernel_applied_to_target_value_only": not no_observer,
            "anchor_donor_cached_fields": None if no_observer else ["query", "key"],
            "anchor_donor_value_hidden_output_or_coordinate_used": None if no_observer else False,
            "anchor_to_target_appearance_correspondence_used": None if no_observer else False,
            "initial_latent_phase_clamped_after_every_update": True,
            "anchor_value_stream_copied": False,
            "source_value_stream_retained": True,
            "outer_schedule_digest": "schedule",
            "attention_cache": {
                "capture_count": capture,
                "replay_count": replay,
                "qk_only_capture_count": capture,
                "qk_only_replay_count": replay,
                "pending_entries": 0,
                "selected_block_indices": validator.BLOCKS,
            },
        }
        receipt = {
            "schema_version": validator.NATIVE_SCHEMA,
            "complete": True,
            "training_performed": False,
            "optimization_steps": 0,
            "loaded_trained_attention_checkpoint": False,
            "trained_attention_checkpoint": None,
            "source": {"sha256": prompts["source_video_sha256"]},
            "pure_t2v_anchor": {
                "sha256": prompts["pure_noobserver_placeholder"]["sha256"] if no_observer else prompts["anchor_video_sha256"],
                "active_solver_steps": steps,
                "model_forward_at_every_active_solver_step_and_candidate": not no_observer,
            },
            "anchor_generation_initial_gaussian": None,
            "prompts": copy.deepcopy(prompts["effective_native_prompt_sha256"]),
            "checkpoint_content": {
                "manifest_sha256_computed": self.spec["common_runtime_contract"]["checkpoint_manifest_sha256"],
                "manifest_sha256_expected": self.spec["common_runtime_contract"]["checkpoint_manifest_sha256"],
                "verified_file_count": validator.CHECKPOINT_CONTENT_FILE_COUNT,
                "every_file_sha256_verified": True,
                "verified_entries_digest": "e" * 64,
            },
            "mechanism": {
                "arm": "AQK_IID1",
                "transport": validator.PURE_QK_TRANSPORT,
                "transport_strength": 1.0,
                "transport_steps": steps,
                "initial_phase_clamp": True,
                "field_guidance": "raw_cfg",
                "field_model": "first_phase_caption_i2v",
                "source_cfg_scale": 4.5,
                "target_cfg_scale": 4.5,
                "early_candidate_count": 5,
                "initial_noise_proposal_mode": "keyed_only",
                "anchor_state_mode": "clean_noised",
                "anchor_cfg_scope": "shared",
                "anchor_contrast_mode": "dynamic_static_same_caption",
                "preservation_mode": "none",
                "anchor_candidate_mode": "single_shared",
                "anchor_bank_size": 1,
                "selected_blocks": validator.BLOCKS,
                "pure_t2v_anchor_values_or_pixels_copied_to_output": False,
                "trace": trace,
                "trace_digest": validator.canonical_sha256(trace),
            },
            "output": {"path": "/out/result.mp4", "sha256": output_sha, "frames": 81, "fps": 25},
            "freeze_before": freeze,
            "freeze_after": copy.deepcopy(freeze),
        }
        return receipt

    @staticmethod
    def rng_receipts(role: str, latent_sha: str) -> list[dict]:
        rows = [
            {
                "master_seed": 2027,
                "step": step,
                "candidate": 0,
                "derived_seed": validator._keyed_noise_seed(2027, step, 0),
                "raw_noise_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
            }
            for step in range(40)
        ]
        no_observer = role == validator.ARM_ROLES[0]
        route_on = role == validator.ARM_ROLES[2]
        return [
            {
                "schema_version": validator.RNG_SCHEMA,
                "complete": True,
                "arm_role": role,
                "rank": rank,
                "local_rank": rank,
                "world_size": 4,
                "fork_rng": {
                    "enabled": True,
                    "scope": "entire_legacy_inference_entrypoint_per_rank",
                    "owned_cuda_device": rank,
                    "before": {"cpu_sha256": "a" * 64, "cuda_sha256": "b" * 64},
                    "after": {"cpu_sha256": "a" * 64, "cuda_sha256": "b" * 64},
                    "cpu_state_restored": True,
                    "owned_cuda_state_restored": True,
                },
                "legacy_abi": {
                    "offline_anchor_graph_split_supported": False,
                    "target_process_reads_anchor_video_path": True,
                    "target_process_decodes_anchor_video": True,
                    "self_generated_anchor_video_read": not no_observer,
                    "required_anchor_path_is_source_placeholder": False,
                    "required_anchor_path_is_source_frame0_static_placeholder": no_observer,
                    "output_path": "/out/result.mp4",
                },
                "runtime_noise": {
                    "scheme": "sha256_keyed_cpu_torch_generator_v1",
                    "master_seed": 2027,
                    "rows": copy.deepcopy(rows),
                },
                "route_application": {
                    "enabled": route_on,
                    "exact_identity_gate": role == validator.ARM_ROLES[1],
                    "call_count": 0 if no_observer else 2 * 40 * len(validator.BLOCKS),
                },
                "predecode_latent": {
                    "raw_storage_sha256": latent_sha,
                    "dtype": "torch.float32",
                    "shape": [1, 16, 21, 88, 132],
                    "finite": True,
                },
                "native_adapter_off_proof_rank0": (
                    {
                        "native_receipt_path": "/out/result.mp4.receipt.json",
                        "native_receipt_sha256": "d" * 64,
                    }
                    if rank == 0
                    else None
                ),
            }
            for rank in range(4)
        ]

    def test_spec_is_honest_three_arm_closure(self) -> None:
        value = validator.validate_spec(self.spec)
        self.assertEqual(value["schema_version"], validator.SCHEMA)
        self.assertEqual([row["arm_role"] for row in self.spec["arms"]], list(validator.ARM_ROLES))
        self.assertTrue(self.spec["comparison_limits"]["cache_abi_identical"])
        self.assertFalse(self.spec["legacy_abi_audit"]["preferred_split_process_contract_satisfied"])

    def test_spec_rejects_anchor_free_claim_and_wrong_role(self) -> None:
        bad = copy.deepcopy(self.spec)
        bad["legacy_abi_audit"]["forbidden_claims"].remove("anchor-free K0")
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_spec(bad)
        bad = copy.deepcopy(self.spec)
        bad["data_and_prompt_contract"]["editing_instruction"] = "pour into a cup"
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_spec(bad)

    def test_wrapper_argument_closure_for_all_arms(self) -> None:
        for role in validator.ARM_ROLES:
            value = wrapper.validate_legacy_argv(self.legacy_argv(role), arm_role=role)
            self.assertEqual(value["steps"], "0" if role == validator.ARM_ROLES[0] else "40")
        bad = self.legacy_argv(validator.ARM_ROLES[0])
        bad[bad.index("--anchor-video") + 1] = "/data/source.mp4"
        with self.assertRaises(wrapper.E00LegacyWrapperError):
            wrapper.validate_legacy_argv(bad, arm_role=validator.ARM_ROLES[0])

    def test_pinned_infer_rejects_source_equal_anchor_latent(self) -> None:
        source = (METHOD_ROOT / "infer_anchor_sga_anc_event_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.equal(source_latent, item)", source)
        self.assertNotEqual(
            self.spec["data_and_prompt_contract"]["pure_noobserver_placeholder"]["sha256"],
            self.spec["data_and_prompt_contract"]["source_video_sha256"],
        )

    def test_cpu_fork_rng_restores_state(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        before = torch.random.get_rng_state().clone()
        _, proof = wrapper.run_with_rng_fork(
            torch, lambda: torch.manual_seed(991), cuda_device=None
        )
        self.assertTrue(proof["cpu_state_restored"])
        self.assertTrue(proof["owned_cuda_state_restored"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_identity_route_gate_is_exact_tensor_identity(self) -> None:
        qk = types.ModuleType("anchor_qk_transport")
        qk._qk_only_temporal_kernel_contrast_output = lambda current, *args, **kwargs: ("routed", current)
        tensor = object()
        with mock.patch.dict(sys.modules, {"anchor_qk_transport": qk}):
            with wrapper.audit_route_application(mode="identity_observer") as audit:
                result = qk._qk_only_temporal_kernel_contrast_output(tensor)
                self.assertIs(result, tensor)
            self.assertEqual(audit["call_count"], 1)
            with wrapper.audit_route_application(mode="inactive_noobserver"):
                with self.assertRaises(wrapper.E00LegacyWrapperError):
                    qk._qk_only_temporal_kernel_contrast_output(tensor)

    def test_three_arm_receipt_and_exact_observer_side_effect_gate(self) -> None:
        same_output = "1" * 64
        latent = "2" * 64
        audits = []
        for role, output_sha in zip(validator.ARM_ROLES, (same_output, same_output, "3" * 64)):
            audits.append(
                validator.build_arm_audit(
                    spec=self.spec,
                    native_receipt=self.native_receipt(role, output_sha),
                    rng_receipts=self.rng_receipts(role, latent if role != validator.ARM_ROLES[2] else "4" * 64),
                    arm_role=role,
                    native_receipt_sha256="d" * 64,
                )
            )
        pair = validator.validate_pair_audits(*audits)
        self.assertTrue(pair["pure_noobserver_vs_observer_routeoff_predecode_latent_exact"])
        self.assertTrue(pair["pure_noobserver_vs_observer_routeoff_video_sha256_exact"])
        broken = copy.deepcopy(audits[1])
        broken["rng_and_noise"]["predecode_latent_sha256"] = "9" * 64
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_pair_audits(audits[0], broken, audits[2])

    def test_rng_receipt_rejects_seed_rank_and_native_binding_tampering(self) -> None:
        role = validator.ARM_ROLES[1]
        rows = self.rng_receipts(role, "2" * 64)
        rows[0]["runtime_noise"]["rows"][0]["derived_seed"] += 1
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_rng_receipts(
                rows,
                arm_role=role,
                expected_output_path="/out/result.mp4",
                native_receipt_sha256="d" * 64,
            )
        rows = self.rng_receipts(role, "2" * 64)
        rows[0]["local_rank"] = 3
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_rng_receipts(
                rows,
                arm_role=role,
                expected_output_path="/out/result.mp4",
                native_receipt_sha256="d" * 64,
            )

    def test_native_receipt_rejects_trace_and_checkpoint_tampering(self) -> None:
        role = validator.ARM_ROLES[1]
        receipt = self.native_receipt(role, "1" * 64)
        receipt["mechanism"]["trace"]["candidate_counts"][0] = 2
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_native_receipt(receipt, spec=self.spec, arm_role=role)
        receipt = self.native_receipt(role, "1" * 64)
        receipt["checkpoint_content"]["manifest_sha256_computed"] = "0" * 64
        with self.assertRaises(validator.E00LegacyDiagnosticError):
            validator.validate_native_receipt(receipt, spec=self.spec, arm_role=role)

    def test_launcher_is_review_gated_and_preserves_slurm_gpu_visibility(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts/auh_launch_e00_three_vessel_fresh_keyed_legacy_diag_node292_v1.sh"
        ).read_text(encoding="utf-8")
        bridge = (
            METHOD_ROOT / "scripts/auh_e00_three_vessel_fresh_keyed_legacy_bridge_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            launcher.index("execution_authorized == true"),
            launcher.index("srun --jobid"),
        )
        for state in ('state:"started"', '.state="failed"', '.state="completed"'):
            self.assertIn(state, launcher)
        self.assertNotIn("export ROCR_VISIBLE_DEVICES=", bridge)
        self.assertNotIn("unset HIP_VISIBLE_DEVICES", bridge)
        self.assertIn("pure_noobserver_placeholder.package_relative_path", bridge)

    def test_immutable_package_builder_and_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dfix2 = root / "dfix2"
            overlay = root / "overlay"
            output = root / "package"
            core_relative = "methods/core.py"
            overlay_relative = "methods/new.py"
            (dfix2 / "methods").mkdir(parents=True)
            (overlay / "methods").mkdir(parents=True)
            (dfix2 / core_relative).write_text("core\n", encoding="utf-8")
            (overlay / overlay_relative).write_text("new\n", encoding="utf-8")
            expected = {core_relative: hashlib.sha256(b"core\n").hexdigest()}
            package.build_package(
                dfix2_source_tree=dfix2,
                overlay_root=overlay,
                output=output,
                expected_core=expected,
                overlay_files=(overlay_relative,),
            )
            manifest = output / package.MANIFEST_NAME
            value = package.verify_package(package_root=output, manifest_path=manifest)
            self.assertFalse(value["execution_authorized"])
            (output / overlay_relative).write_text("changed\n", encoding="utf-8")
            with self.assertRaises(package.PackageError):
                package.verify_package(package_root=output, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
