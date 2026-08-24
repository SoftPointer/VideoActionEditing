from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as original  # noqa: E402
import infer_seer_same_state_full160_lora as full160  # noqa: E402
import infer_seer_same_state_lora as four_step  # noqa: E402
import infer_seer_scoped_lora as scoped  # noqa: E402


SHA1 = "1" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _adapter_config() -> dict[str, object]:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": ["to_q", "to_out.0"],
    }


def _parameter_names() -> list[str]:
    return sorted(full160._expected_optimizer_parameter_names())


def _receipt(*, step: int = 160, max_steps: int = 160) -> dict[str, object]:
    targets = scoped.expected_lora_target_modules()
    names = _parameter_names()
    noop_sha = hashlib.sha256(
        full160.trainer.motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    value: dict[str, object] = {
        "method": full160.trainer.METHOD_NAME,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA_B,
        "bernini_commit": original.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": original.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint_tree_sha256": original.trainer.CHECKPOINT_TREE_SHA256,
        "checkpoint_path": "/base-checkpoint",
        "dataset_signature": SHA_C,
        "dataset_summary_sha256": SHA_D,
        "dataset_index_sha256": "e" * 64,
        "routing_digest": "f" * 64,
        "routing_file_sha256": "0" * 64,
        "expected_routing_jsonl_sha256": "0" * 64,
        "eligible_route_stream_count": 2,
        "eligible_route_stream_sha256": "2" * 64,
        "seed": 20260813,
        "learning_rate": 1.0e-6,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_scope": "cross_q_out",
        "target_modules": targets,
        "paired_cells": ["action", "exact_copy"],
        "posterior_statistic": "mode",
        "branch_state_mode": "shared_noisy_clean_field",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": 0.25,
        "shared_source_sigma_noise": True,
        "exact_same_noisy_query": True,
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection": None,
        "target_projection_idempotent": None,
        "motion_loss_multiplier": (
            "high_noise(sigma) / max(sigma, inverse_sigma_weight_floor)"
        ),
        "copy_boundary_loss_multiplier": (
            "1 / max(sigma, inverse_sigma_weight_floor)"
        ),
        "clean_field_loss_weight_range": [1.0, 4.0],
        "motion_objective": "causal_boundary_charbonnier",
        "motion_representation": (
            "source-relative-causal-boundary-charbonnier-v1"
        ),
        "temporal_lags": [1, 2, 4],
        "quotient_weight": 0.5,
        "motion_loss_weight": 0.5,
        "copy_loss_weight": 0.5,
        "boundary_gauge_loss_weight": 0.0,
        "boundary_gauge": "zero_first_latent_phase_of_raw_predicted_clean_delta",
        "bridge_fractions": None,
        "bridge_consistency_weight": 0.0,
        "causal_ema_decay": 0.5,
        "charbonnier_scale": 0.1,
        "inference_sigma_schedule_sha256": None,
        "inference_sigma_selector": None,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "noop_instruction_sha256": noop_sha,
        "expected_seer_manifest_sha256": SHA_A,
        "expected_seer_owner_spec_sha256": SHA_C,
        "seer_row_count": 2,
        "seer_iids_sha256": "3" * 64,
        "seer_authority": dict(full160.trainer.AUTHORITY),
        "same_generated_video_coordinate": True,
        "event_erasure_source_excludes_transition_and_terminal": True,
        "full_pair_flow_matching_weight": 1.0,
        "same_state_causal_motion_weight": 0.5,
        "same_state_noop_copy_weight": 0.5,
        "rejected_cmsg_cross_identity_gate_reused": False,
        "training_completion_is_method_success": False,
        "heldout_decoded_review_required": True,
    }
    immutable = {
        "value": value,
        "digest": original.object_sha256(value),
        "expected_seer_manifest_sha256": SHA_A,
        "expected_seer_owner_spec_sha256": SHA_C,
        "method_source_archive_sha256": SHA_B,
    }
    supervision = {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "target_video_used_as_external_condition": False,
        "projected_target_used_as_training_query": False,
        "external_mask_track_pose_trajectory": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "action_noop_forwards_per_optimizer_step": 2,
        "counterfactual_noop_forward": True,
        "branch_state_mode": "shared_noisy_clean_field",
        "exact_same_noisy_query": True,
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": 0.25,
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection": None,
        "target_projection_idempotent": None,
        "motion_loss_multiplier": (
            "high_noise(sigma) / max(sigma, inverse_sigma_weight_floor)"
        ),
        "copy_boundary_loss_multiplier": (
            "1 / max(sigma, inverse_sigma_weight_floor)"
        ),
        "only_text_condition_differs": True,
        "copy_calibration_enabled": True,
        "copy_calibration_weight": 0.5,
        "boundary_gauge_enabled": False,
        "boundary_gauge_loss_weight": 0.0,
        "boundary_gauge_field": "raw_predicted_action_minus_noop_clean_field",
        "boundary_gauge_target": "zero_first_latent_phase",
        "boundary_gauge_uses_target_appearance": False,
        "motion_loss_enabled": True,
        "motion_objective": "causal_boundary_charbonnier",
        "raw_delta_enabled": False,
        "shared_source_posterior_mode": True,
        "shared_sigma": True,
        "shared_diffusion_noise": True,
        "unreviewed_full_target_weight": 0.0,
        "motion_representation": (
            "source-relative-causal-boundary-charbonnier-v1"
        ),
        "temporal_quotient_enabled": False,
        "causal_boundary_quotient_enabled": True,
        "causal_boundary_projection_enabled": True,
        "temporal_quotient_weight": 0.5,
        "multiscale_enabled": False,
        "temporal_lags": [1, 2, 4],
        "causal_boundary_gauge_loss_weight": 0.0,
        "bridge_endpoints": None,
        "bridge_consistency_enabled": False,
        "bridge_consistency_weight": 0.0,
        "bridge_query_formula": None,
        "causal_ema_enabled": False,
        "causal_ema_decay": 0.5,
        "charbonnier_scale": 0.1,
        "inference_sigma_stratification": None,
        "inference_sigma_schedule_sha256": None,
        "self_generated_target_supervision": True,
        "event_erased_source_supervision": True,
        "same_generated_identity_background_coordinate": True,
        "full_pair_flow_matching_enabled": True,
        "full_pair_flow_matching_weight": 1.0,
        "same_state_causal_motion_weight": 0.5,
        "same_state_noop_copy_weight": 0.5,
        "training_completion_is_method_success": False,
        "heldout_decoded_review_required": True,
    }
    receipt: dict[str, object] = {
        "schema_version": full160.trainer.RECEIPT_SCHEMA,
        "method": full160.trainer.METHOD_NAME,
        "global_step": step,
        "max_steps": max_steps,
        "bernini_commit": original.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": original.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint": {
            "path": "/base-checkpoint",
            "tree_sha256": original.trainer.CHECKPOINT_TREE_SHA256,
        },
        "dataset": {
            "path": "/dataset",
            "rows": 2,
            "signature": SHA_C,
            "summary": {
                "sha256": SHA_D,
                "index_sha256": "e" * 64,
                "complete": True,
                "allow_incomplete": False,
                "expected_rows": 2,
                "materialized_rows": 2,
            },
            "routing": {
                "schema_version": full160.trainer.motion.ROUTING_SCHEMA,
                "path": "/routing.jsonl",
                "file_sha256": "0" * 64,
                "routing_digest": "f" * 64,
                "default_tier": "reject",
                "explicit_route_counts": {
                    "full_pair": 2,
                    "motion_only": 0,
                    "reject": 0,
                },
            },
        },
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": "cross_q_out",
            "target_module_count": 60,
            "target_modules": targets,
            "target_modules_sha256": original.object_sha256(targets),
            "trainable_parameter_count": 1_474_560,
            "parameter_names_sha256": original.object_sha256(names),
            "initialization_digest": SHA_A,
            "checkpoint_parameter_digest": SHA_B,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 1.0e-6,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "parameter_names": names,
            "checkpoint_state_digest": "4" * 64,
        },
        "immutable_contract": immutable,
        "supervision": supervision,
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "last_metrics": {"preclip_gradient_norm": 0.125},
        "parameter_update_evidence": {
            "initial_trainable_parameter_digest": SHA_A,
            "final_trainable_parameter_digest": SHA_B,
            "exact_parameter_bytes_changed": True,
            "engineering_execution_success": True,
            "method_success_claimed": False,
            "final_preclip_gradient_norm": 0.125,
        },
        "seer": {
            "owner_spec_sha256": SHA_C,
            "dataset_manifest_sha256": SHA_A,
            "row_count": 2,
            "self_generated_target_supervision": True,
            "event_erased_source_supervision": True,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        },
        "transformers_version": "fixture-transformers",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = original.object_sha256(receipt)
    return receipt


def _resign(receipt: dict[str, object]) -> dict[str, object]:
    receipt["receipt_digest"] = original.object_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    return receipt


class SeerSameStateFull160ContractTests(unittest.TestCase):
    def test_valid_full160_receipt_is_admitted(self) -> None:
        result = full160.validate_adapter_contract(_adapter_config(), _receipt())
        self.assertEqual(result["global_step"], 160)
        self.assertEqual(result["method_source_archive_sha256"], SHA_B)

    def test_exact_step_boundary_rejects_4_159_and_161(self) -> None:
        for step in (4, 159, 161):
            with self.subTest(step=step), self.assertRaisesRegex(
                full160.SeerSameStateFull160InferenceError, "global_step=160"
            ):
                full160.validate_adapter_contract(
                    _adapter_config(), _receipt(step=step)
                )

    def test_max_steps_and_checkpoint_directory_are_exact(self) -> None:
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "max_steps=160"
        ):
            full160.validate_adapter_contract(
                _adapter_config(), _receipt(max_steps=161)
            )
        self.assertEqual(
            full160.validate_checkpoint_save_directory(
                Path("/runs/checkpoint-00000160")
            ),
            Path("/runs/checkpoint-00000160"),
        )
        for name in ("checkpoint-00000004", "checkpoint-00000159", "adapter"):
            with self.subTest(name=name), self.assertRaisesRegex(
                full160.SeerSameStateFull160InferenceError,
                "checkpoint-00000160",
            ):
                full160.validate_checkpoint_save_directory(Path("/runs") / name)

    def test_four_step_helper_remains_closed(self) -> None:
        four = _receipt(step=4, max_steps=160)
        result = four_step.validate_adapter_contract(_adapter_config(), four)
        self.assertEqual(result["global_step"], 4)
        with self.assertRaisesRegex(
            four_step.SeerSameStateInferenceError, "four-step"
        ):
            four_step.validate_adapter_contract(_adapter_config(), _receipt())

    def test_resigned_archive_and_manifest_tampering_fail_closed(self) -> None:
        receipt = _receipt()
        immutable = copy.deepcopy(receipt["immutable_contract"])
        immutable["value"]["method_source_archive_sha256"] = "9" * 64
        immutable["digest"] = original.object_sha256(immutable["value"])
        receipt["immutable_contract"] = immutable
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "immutable"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

        receipt = _receipt()
        immutable = copy.deepcopy(receipt["immutable_contract"])
        immutable["value"]["expected_seer_manifest_sha256"] = "8" * 64
        immutable["digest"] = original.object_sha256(immutable["value"])
        immutable["expected_seer_manifest_sha256"] = "8" * 64
        receipt["immutable_contract"] = immutable
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "update evidence"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

    def test_resigned_gradient_and_objective_tampering_fail_closed(self) -> None:
        receipt = _receipt()
        update = copy.deepcopy(receipt["parameter_update_evidence"])
        update["final_preclip_gradient_norm"] = 0.5
        receipt["parameter_update_evidence"] = update
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "cross-bind"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

        receipt = _receipt()
        immutable = copy.deepcopy(receipt["immutable_contract"])
        immutable["value"]["learning_rate"] = 2.0e-6
        immutable["digest"] = original.object_sha256(immutable["value"])
        receipt["immutable_contract"] = immutable
        optimizer = copy.deepcopy(receipt["optimizer"])
        optimizer["learning_rate"] = 2.0e-6
        receipt["optimizer"] = optimizer
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "objective"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

    def test_resigned_immutable_extension_and_authority_tampering_fail_closed(self) -> None:
        receipt = _receipt()
        immutable = copy.deepcopy(receipt["immutable_contract"])
        immutable["value"]["unreviewed_extension"] = True
        immutable["digest"] = original.object_sha256(immutable["value"])
        receipt["immutable_contract"] = immutable
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "immutable objective"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

        receipt = _receipt()
        immutable = copy.deepcopy(receipt["immutable_contract"])
        immutable["value"]["seer_authority"] = dict(
            immutable["value"]["seer_authority"]
        )
        immutable["value"]["seer_authority"][
            "production_claim_authorized"
        ] = True
        immutable["digest"] = original.object_sha256(immutable["value"])
        receipt["immutable_contract"] = immutable
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "immutable"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

    def test_resigned_scope_and_provenance_tampering_fail_closed(self) -> None:
        receipt = _receipt()
        optimizer = copy.deepcopy(receipt["optimizer"])
        optimizer["parameter_names"] = optimizer["parameter_names"][:-1]
        receipt["optimizer"] = optimizer
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "120-tensor"
        ):
            full160.validate_adapter_contract(_adapter_config(), _resign(receipt))

        identity = full160.validate_adapter_contract(_adapter_config(), _receipt())
        with self.assertRaisesRegex(
            full160.SeerSameStateFull160InferenceError, "provenance"
        ):
            full160.validate_runtime_provenance(
                identity,
                method_source_revision=SHA1,
                method_source_archive_sha256="7" * 64,
            )


if __name__ == "__main__":
    unittest.main()
