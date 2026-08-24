#!/usr/bin/env python3
"""Hostile normal/-O tests for the case01 trajectory exact-five core."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from methods.bernini_action_editing import case01_object_trajectory_exact5_eval_v1 as evaluator
from methods.bernini_action_editing import infer_case01_object_trajectory_oracle_v1 as wrapper


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = REPO_ROOT / "methods/bernini_action_editing"
SOURCE = (
    REPO_ROOT
    / "artifacts/object_grounded_case01_0821_bone_interventions_r4/videos/exact_original.mp4"
)
REMOVED = (
    REPO_ROOT
    / "artifacts/object_grounded_case01_0821_bone_interventions_r4/videos/bone_removed.mp4"
)
STAGE0 = (
    REPO_ROOT / "artifacts/object_grounded_case01_0821_sam2_masklets_r2/receipt.json"
)
G0 = (
    METHOD_ROOT / "assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json"
)
SCAFFOLD = REPO_ROOT / "artifacts/case01_oracle_object_trajectory_v1/scaffold.json"
SCAFFOLD_AUDIT = (
    REPO_ROOT
    / "md/action_editing/20260821_man/evidence/case01_object_trajectory_scaffold_independent_audit_v1.json"
)


def _hold_plan(output_root: Path) -> dict:
    source = evaluator.build_file_authority(SOURCE, role="exact_original_source")
    conditions = {
        "stage0_masks": evaluator.build_file_authority(
            STAGE0, role="stage0_masks",
            payload_digest=evaluator.EXPECTED_STAGE0_RECEIPT_DIGEST,
        ),
        "g0_mouth_track": evaluator.build_file_authority(G0, role="g0_mouth_track"),
        "trajectory_scaffold": evaluator.build_file_authority(
            SCAFFOLD, role="trajectory_scaffold",
            payload_digest=evaluator.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
        ),
        "aux_bone_removed_source": evaluator.build_file_authority(
            REMOVED, role="aux_bone_removed_source"
        ),
    }
    return evaluator.build_plan(
        source_authority=source,
        condition_authorities=conditions,
        admission_authorities={
            "scaffold_independent_audit": evaluator.build_file_authority(
                SCAFFOLD_AUDIT, role="scaffold_independent_audit",
                payload_digest=evaluator.EXPECTED_SCAFFOLD_AUDIT_DIGEST,
            )
        },
        checkpoint_manifest=evaluator.incomplete_checkpoint_manifest(),
        producer=evaluator.incomplete_producer(),
        output_root=output_root,
    )


def _legacy_trace() -> dict:
    steps = []
    for index in range(evaluator.NUM_INFERENCE_STEPS):
        sigma = (evaluator.NUM_INFERENCE_STEPS - index) / evaluator.NUM_INFERENCE_STEPS
        next_sigma = (
            evaluator.NUM_INFERENCE_STEPS - index - 1
        ) / evaluator.NUM_INFERENCE_STEPS
        steps.append(
            {
                "step_index": index,
                "timestep": float(1_000 - index),
                "sigma": sigma,
                "next_sigma": next_sigma,
                "phase0_velocity": "captured_epsilon_minus_clean_source",
                "phase0_post_step": "source_noise_flow_trajectory_projection",
                "other_phases_projected": False,
                "original_scheduler_step_calls": 1,
            }
        )
    return {
        "schema_version": "bernini-source-phase0-unipc-clamp-v1",
        "policy": "hard1_every_step",
        "integrator": "original_unipc_scheduler_step",
        "prediction_type": "flow_prediction",
        "phase": 0,
        "latent_phases": 21,
        "initial_packed_noise_captured": True,
        "step_count": 40,
        "expected_steps": 40,
        "steps": steps,
        "target_video_accessed": False,
        "identity_or_background_claim": False,
    }


def _publication_identity(*, mode: int, nlink: int, size: int) -> dict:
    return {
        "device": 10,
        "inode": 11 if nlink else 12,
        "uid": 501,
        "gid": 20,
        "mode": stat.S_IFREG | mode,
        "nlink": nlink,
        "rdev": 0,
        "size": size,
        "blocks": 8,
        "mtime_ns": 13,
        "ctime_ns": 14,
    }


def _legacy_receipt(
    task: dict, producer: dict, *, output_sha: str = "a" * 64
) -> dict:
    source_physical = {
        "path": task["source_video"],
        "sha256": task["source_video_sha256"],
        "size": evaluator.EXPECTED_SOURCE_SIZE,
        "mode": 0o444,
        "device": 1,
        "inode": 2,
        "uid": 3,
        "gid": 4,
        "nlink": 1,
        "rdev": 0,
        "blocks": 1,
        "mtime_ns": 5,
        "ctime_ns": 6,
    }
    source_physical_digest = evaluator.object_sha256(source_physical)
    output_size = 123
    consumption_digest = "d" * 64
    task_digest = "e" * 64
    rank_digest = "f" * 64
    checkpoint_manifest = dict(task["adapter"]["checkpoint_manifest"])
    checkpoint_manifest.pop("pin_complete", None)
    receipt = {
        "schema_version": evaluator.LEGACY_INFERENCE_RECEIPT_SCHEMA,
        "infer_lora_source_sha256": producer["infer_lora_sha256"],
        "method_source_revision": producer["method_source_revision"],
        "method_source_archive_sha256": producer["method_source_archive_sha256"],
        "bernini_commit": evaluator.EXPECTED_BERNINI_COMMIT,
        "veomni_commit": evaluator.EXPECTED_VEOMNI_COMMIT,
        "bernini_inference_files": evaluator.EXPECTED_BERNINI_INFERENCE_FILES,
        "checkpoint_tree_sha256": evaluator.EXPECTED_CHECKPOINT_TREE_SHA256,
        "input": {
            "source_video_path": task["source_video"],
            "source_video_sha256": task["source_video_sha256"],
            "instruction_utf8_sha256": task["instruction_sha256"],
            "instruction_utf8_bytes": len(task["instruction"].encode("utf-8")),
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_image_or_video": False,
            "external_shared_i0": False,
            "source_video_physical_authority": source_physical,
            "source_video_physical_authority_digest": source_physical_digest,
            "retained_source_fd_consumed": True,
            "source_video_pre_and_post_decode_rehashed": True,
        },
        "preprocessing": {
            "frame_count": 81,
            "fps": 25.0,
            "reported_fps": 25.0,
            "source_input_hw": [736, 704],
            "source_derived_bucket_hw": [496, 480],
            "max_pixels": 245_760,
            "stride": 16,
            "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
            "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
            "resize": "torchvision_bicubic_antialias_true",
            "external_shared_i0": False,
        },
        "prompt_contract": {
            "task": "mv2v",
            "system_prompt_sha256": evaluator.EXPECTED_SYSTEM_PROMPT_SHA256,
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        },
        "sampling": {
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_mode": "v2v_apg",
            "omega_vid": 1.25,
            "omega_img": 0.0,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "seed": 2027,
            "eta": 0.5,
            "norm_threshold": [50.0, 50.0],
            "momentum": 0.0,
            "single_expert": "transformer_1",
            "ulysses_size": 4,
            "rank0_decode_and_save_only": True,
            "source_onset_policy": "hard1_every_step",
            "source_onset_solver_trace": _legacy_trace(),
        },
        "adapter": {
            "enabled": True,
            "mode": "lora_safe_merge",
            "checkpoint_root": task["adapter"]["checkpoint_root"],
            "adapter_model_path": "/retained/adapter_model.safetensors",
            "adapter_model_sha256": evaluator.EXPECTED_CHECKPOINT[
                "adapter_model_sha256"
            ],
            "training_receipt_path": "/retained/training_receipt.json",
            "training_receipt_digest": evaluator.EXPECTED_CHECKPOINT[
                "receipt_digest"
            ],
            "training_global_step": 644,
            "strictly_reloaded": True,
            "safe_merged_for_inference": True,
            "tensor_count": 480,
            "target_modules_sha256": evaluator.EXPECTED_TARGET_MODULES_SHA256,
            "profile": evaluator.FULL644_PROFILE,
            "lora_rank": 64,
            "lora_alpha": 64,
            "target_module_count": 240,
            "checkpoint_manifest": checkpoint_manifest,
        },
        "output": {
            "path": task["output"]["video_path"],
            "sha256": output_sha,
            "size": output_size,
            "frame_count": 81,
            "fps": 25.0,
            "height": 496,
            "width": 480,
            "audio_preserved": False,
            "publication_identity": _publication_identity(
                mode=0o444, nlink=1, size=output_size
            ),
            "prepublication_identity": _publication_identity(
                mode=0o600, nlink=0, size=output_size
            ),
            "anonymous_creation_method": "linux-sealed-memfd-v1",
            "anonymous_seal_mask": 15,
            "sealed_source_sha256": output_sha,
            "sealed_source_size": output_size,
            "anonymous_inode_encoded_and_decoded_before_publication": True,
            "create_only_copy_publication_after_decode": True,
            "sealed_source_and_publication_bytes_equal": True,
            "retained_inode_encoded_and_replayed": True,
            "named_output_never_replaced": True,
        },
        "model_consumption": {
            "consumption_input_digest": consumption_digest,
            "task_input_digest": task_digest,
            "model_capture_digest": "1" * 64,
            "model_view_root": "/proc/self/fd/model",
            "adapter_capture_digest": "2" * 64,
            "adapter_view_root": "/proc/self/fd/adapter",
            "fd_view_files_authorized": 10,
            "inherited_fd_binding_digest": "3" * 64,
            "inherited_fd_count": 10,
            "ptrace_authorization_used": False,
            "source_video_sha256": task["source_video_sha256"],
            "source_video_physical_authority_digest": source_physical_digest,
            "all_ranks_use_retained_source_fd": True,
            "four_rank_attestation": {
                "world_size": 4,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": rank_digest,
                "ordered_rank_evidence_digests": [rank_digest] * 4,
            },
        },
        "runtime_versions": {
            "torch": "2.4.1", "torch_hip": "6.1", "transformers": "4.57.1",
            "diffusers": "0.36.0.dev0", "peft": "0.19.1",
        },
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "consumption_input_digest": consumption_digest,
        "task_input_digest": task_digest,
    }
    receipt["receipt_digest"] = evaluator.object_sha256(receipt)
    return receipt


def _gate(stage: str) -> dict:
    success_status = {
        "stage": stage,
        "ok": True,
        "error_type": None,
        "error_text_sha256": None,
    }
    return {
        "stage": stage,
        "world_size": 4,
        "all_ranks_reported_ok": True,
        "ordered_status_digest": evaluator.object_sha256(
            [success_status] * 4
        ),
    }


def _fake_file_receipt(authority: dict) -> dict:
    identity = {
        "device": 1,
        "inode": 2,
        "mode": stat.S_IFREG | 0o644,
        "nlink": 1,
        "uid": 501,
        "gid": 20,
        "size": authority["size"],
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    payload = {
        "path": authority["path"],
        "sha256": authority["sha256"],
        "identity": identity,
    }
    return {**payload, "authority_digest": evaluator.object_sha256(payload)}


class _FakeAuthority:
    def __init__(self, authority: dict) -> None:
        self.sha256 = authority["sha256"]
        self._receipt = _fake_file_receipt(authority)

    def receipt(self) -> dict:
        return copy.deepcopy(self._receipt)

    def replay(self) -> None:
        return None


class _FakeAssets:
    def __init__(self, task: dict, producer: dict) -> None:
        external = task["external_conditions"]
        self.cli = SimpleNamespace(
            arm=task["oracle_arm"],
            scaffold_digest=evaluator.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
        )
        self.scaffold = {
            "authority": json.loads(SCAFFOLD.read_text(encoding="utf-8"))[
                "authority"
            ]
        }
        self.scaffold_file = _FakeAuthority(external["trajectory_scaffold"])
        self.aux_file = _FakeAuthority(external["aux_bone_removed_source"])
        self._producer = producer

    def producer_hashes(self) -> dict:
        return {
            "wrapper_source_sha256": self._producer[
                "inference_wrapper_sha256"
            ],
            "legacy_infer_lora_source_sha256": self._producer["infer_lora_sha256"],
            "projection_source_sha256": self._producer[
                "trajectory_projection_module_sha256"
            ],
            "scaffold_source_sha256": self._producer[
                "trajectory_scaffold_module_sha256"
            ],
        }


def _tensor_authority() -> dict:
    shapes = {
        "source_packed_full": [1, 19_530, 64],
        "aux_packed_full": [1, 19_530, 64],
        "legacy_phase0_selected_clean": [1, 930, 64],
        "source_bone_correspondence_values": [1, 377, 64],
        "source_effective_origin_values": [1, 187, 64],
        "aux_effective_origin_values": [1, 187, 64],
        "constructed_bone_selected_clean": [1, 564, 64],
        "constructed_dog_identity_clean": [1, 1_548, 64],
    }
    tensors = {
        name: {
            "label": name,
            "shape": shape,
            "dtype": "torch.float32",
            "device_type": "cuda",
            "contiguous_before_snapshot": True,
            "byte_count": shape[0] * shape[1] * shape[2] * 4,
            "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        }
        for name, shape in shapes.items()
    }
    value = {
        "tensors": tensors,
        "effective_origin_element_count": 187 * 64,
        "source_aux_effective_origin_differing_element_count": 32,
        "source_aux_effective_origin_differ": True,
        "local_device": "cuda:0",
    }
    value["content_contract_digest"] = evaluator.object_sha256(
        {key: item for key, item in value.items() if key != "local_device"}
    )
    return value


def _projection_trace(task: dict, producer: dict, assets: _FakeAssets) -> dict:
    arm = task["oracle_arm"]
    row_specs = evaluator._expected_row_specs(arm)
    row_names = [row["name"] for row in row_specs]
    projection_gates = [
        _gate(stage) for stage in evaluator.PROJECTION_COLLECTIVE_STAGES
    ]
    tensor_authority = _tensor_authority()
    contract_base = {
        "arm": arm,
        "expected_steps": 40,
        "row_names": row_names,
        "row_specs": row_specs,
        "token_plan_digest": (
            "7eaef1dbd09e91afb9df109b358f0166757df5ddc2ac59fa09831bfeec955103"
        ),
        "tensor_content_contract_digest": tensor_authority[
            "content_contract_digest"
        ],
    }
    contract_digest = evaluator.object_sha256(contract_base)
    row_construction = {
        "row_names": row_names,
        "row_specs": row_specs,
        "bone_origin_clear_token_count": 187,
        "bone_scaffold_origin_support_token_count": 198,
        "bone_target_tube_token_count": 377,
        "bone_correspondence_count": 377,
        "bone_correspondence_sha256": (
            "3ddd38d6ab846b121eaa3629f121e14cb51e26d23afa0def4b8a1012c982ea7e"
        ),
        "dog_core_token_count": 1_548,
        "responsibility_tube_token_count": 2_760,
        "overlapping_origin_target_policy": "target_source_bone_detail_wins",
        "plan_digest": contract_base["token_plan_digest"],
        "dog_row_consumed": arm == "trajectory_dog_bone",
        "origin_authority": "aux_bone_removed_source_packed",
        "target_bone_detail_authority": "same_source_bone_correspondence_scatter",
        "dog_detail_authority": "same_source_packed_dog_core",
        "single_instance_conservation_constructed": True,
        "matched_legacy_phase0_baseline": True,
        "legacy_phase0_selected_token_count": 930,
        "legacy_phase0_sigma_gate": "all_steps_all_sigma",
        "tensor_authority": tensor_authority,
        "pre_projection_build_gate": projection_gates[1],
        "pre_projection_contract_gate": projection_gates[2],
        "projector_lookup_gate": projection_gates[3],
        "lazy_bootstrap_install_gate": projection_gates[4],
        "projector_install_gate": projection_gates[5],
        "final_validation_gate": projection_gates[6],
        "projection_contract": {
            **contract_base,
            "projection_contract_digest": contract_digest,
            "four_rank_consensus": {
                "world_size": 4,
                "all_ranks_exact_projection_contract_equal": True,
                "ordered_projection_contract_digests": [contract_digest] * 4,
            },
        },
    }
    core_rows = [
        {
            "name": row["name"],
            "clean_shape": [1, 19_530, 64],
            "weight_shape": row["weight_shape"],
            "selected_token_count": row["selected_token_count"],
            "selected_element_count": row["selected_token_count"] * 64,
            "step_gates": row["step_gates"],
            "active_next_sigma_min": row["active_next_sigma_min"],
            "active_next_sigma_max": row["active_next_sigma_max"],
        }
        for row in row_specs
    ]
    steps = []
    for index in range(40):
        sigma = (40 - index) / 40
        next_sigma = (39 - index) / 40
        dog_active = arm == "trajectory_dog_bone" and next_sigma <= 0.5
        active_rows = row_names if dog_active or arm == "trajectory_bone_only" else row_names[:2]
        selected = 2_913 if dog_active else 1_477
        steps.append(
            {
                "step_index": index,
                "timestep": float(1_000 - index),
                "sigma": sigma,
                "next_sigma": next_sigma,
                "cursor_before": None if index == 0 else index,
                "cursor_after": index + 1,
                "active_rows": active_rows,
                "inactive_rows": [
                    name for name in row_names if name not in active_rows
                ],
                "projection_applied": True,
                "exact_native_delegate_no_argument_clone": False,
                "selected_token_count": selected,
                "selected_element_count": selected * 64,
                "total_element_count": 19_530 * 64,
                "selected_velocity_exact": True,
                "unselected_velocity_exact": True,
                "selected_post_step_exact": True,
                "unselected_post_step_exact": True,
                "initial_noise_snapshot_created_this_step": index == 0,
                "initial_sample_matches_registered_noise": index == 0,
                "original_scheduler_step_calls": 1,
            }
        )
    global_tokens = 1_477 if arm == "trajectory_bone_only" else 2_913
    core = {
        "schema_version": evaluator.PROJECTION_TRACE_SCHEMA,
        "contract": evaluator._expected_tensor_core_contract(),
        "zero_training_oracle": True,
        "production_runner_integration": False,
        "wrapper_installed": True,
        "wrapper_restored": True,
        "globally_enabled": True,
        "initial_noise_registration": "lazy_capture_first_native_sample",
        "initial_noise_captured_from_first_native_sample": True,
        "initial_noise_verified": True,
        "initial_noise_dtype": "torch.float32",
        "initial_noise_device": "cuda:0",
        "clean_dtype": "torch.float32",
        "clean_device": "cuda:0",
        "dimensions": {
            "source_reference": [1, 19_530, 64],
            "target_sampler": [1, 19_530, 64],
        },
        "rows": core_rows,
        "expected_steps": 40,
        "step_count": 40,
        "steps": steps,
        "globally_selected_token_count": global_tokens,
        "globally_selected_element_count": global_tokens * 64,
        "finalized": True,
    }
    scaffold_receipt = assets.scaffold_file.receipt()
    scaffold_receipt["artifact_digest"] = (
        evaluator.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST
    )
    aux_receipt = assets.aux_file.receipt()
    aux_receipt["consumed_via_retained_fd"] = True
    gates = [_gate("aux_readiness"), _gate("aux_post_broadcast")]
    value = {
        "schema_version": evaluator.OBJECT_ORACLE_RUNTIME_SCHEMA,
        "arm": arm,
        "manual_oracle": True,
        "zero_training": True,
        "renderer_abi_integration": True,
        "legacy_clamp_replaced": True,
        "projection_installation": (
            "lazy_at_first_native_step_after_runtime_schedule"
        ),
        "aux_latent_broadcast_from_rank0": True,
        "aux_latent_broadcast_calls": 1,
        "vae_encode": {
            "rank0_source_original_calls": 1,
            "rank0_aux_attempts": 1,
            "rank0_aux_original_calls": 1,
        },
        "aux_collective_gates": gates,
        "projection_collective_gates": projection_gates,
        "row_construction": row_construction,
        "typed_action_program_scope": (
            "patient_support_trajectory_and_dog_identity_exclusion_only"
        ),
        "approach_contact_dynamics_directly_enforced": False,
        "new_action_signal_for_unprojected_dynamics": (
            "legacy_edit_instruction_prompt"
        ),
        "authority": {
            "scaffold": scaffold_receipt,
            "aux_bone_removed_source": aux_receipt,
            "embedded_authorities_digest": evaluator.object_sha256(
                assets.scaffold["authority"]
            ),
            "direct_runtime_authorities": [
                "object_trajectory_scaffold", "aux_bone_removed_source"
            ],
            "derived_scaffold_authorities": [
                "stage0_object_masks", "g0_mouth_track"
            ],
            "raw_stage0_or_g0_runtime_accessed": False,
            "producer_hashes": assets.producer_hashes(),
        },
        "tensor_core": core,
        "target_video_accessed": False,
        "learned_method_claim": False,
    }
    value["trace_digest"] = evaluator.object_sha256(value)
    return value


def _custom_receipt(task: dict, producer: dict) -> dict:
    base = _legacy_receipt(task, producer)
    assets = _FakeAssets(task, producer)
    state = None
    if task["oracle_arm"] in {
        "trajectory_bone_only", "trajectory_dog_bone"
    }:
        trace = _projection_trace(task, producer, assets)
        base["sampling"]["source_onset_solver_trace"] = trace
        state = SimpleNamespace(
            projection_trace=trace,
            source_vae_encode_calls=1,
            aux_vae_encode_attempts=1,
            aux_vae_encode_calls=1,
            aux_collective_gates=[
                _gate("aux_readiness"),
                _gate("aux_post_broadcast"),
            ],
            projection_collective_gates=copy.deepcopy(
                trace["projection_collective_gates"]
            ),
            aux_broadcast_calls=1,
        )
    return wrapper._customize_receipt(base, state=state, assets=assets)


def _reseal(value: dict, digest_field: str) -> None:
    value[digest_field] = evaluator.object_sha256(
        {key: item for key, item in value.items() if key != digest_field}
    )


class ObjectTrajectoryExact5CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.temporary.name).resolve()
        self.plan = _hold_plan(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_plan_is_exact_ordered_hold(self) -> None:
        validated = evaluator.validate_plan(self.plan)
        replayed = evaluator.validate_plan(self.plan, reopen_sources=True)
        self.assertFalse(validated["launch_allowed"])
        self.assertFalse(validated["production_ready"])
        self.assertEqual(validated["arms"], list(evaluator.ARM_ORDER))
        self.assertEqual(
            replayed["admission_authorities"]["scaffold_independent_audit"][
                "sha256"
            ],
            evaluator.EXPECTED_SCAFFOLD_AUDIT_SHA256,
        )
        self.assertEqual(
            replayed["condition_authorities"]["trajectory_scaffold"][
                "payload_digest"
            ],
            evaluator.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
        )
        self.assertEqual(
            [row["source_video_sha256"] for row in validated["tasks"]],
            [evaluator.EXPECTED_SOURCE_SHA256] * 5,
        )
        self.assertIn(
            "custom_inference_producer_pins_incomplete", validated["hold_reasons"]
        )
        self.assertIn(
            "explicit_launch_release_not_granted", validated["hold_reasons"]
        )
        self.assertEqual(
            validated["producer"]["infer_lora_sha256"],
            evaluator.EXPECTED_LEGACY_INFER_LORA_SHA256,
        )
        self.assertEqual(
            validated["producer"]["infer_lora_size"],
            evaluator.EXPECTED_LEGACY_INFER_LORA_SIZE,
        )
        self.assertEqual(
            validated["producer"]["infer_lora_role"],
            "frozen_legacy_exact5_infer_lora_not_workspace_head",
        )
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "incomplete|not launchable"
        ):
            evaluator.validate_plan(self.plan, require_launchable=True)

    def test_active_and_route_off_arms_bind_all_external_authorities(self) -> None:
        tasks = {task["oracle_arm"]: task for task in self.plan["tasks"]}
        for arm in ("route_off", "trajectory_bone_only", "trajectory_dog_bone"):
            self.assertEqual(
                set(tasks[arm]["external_conditions"]),
                set(evaluator.EXTERNAL_AUTHORITY_KEYS),
            )
            self.assertEqual(
                tasks[arm]["accepted_model_conditions"],
                list(evaluator.BASE_CONDITION_NAMES + evaluator.EXTERNAL_CONDITION_NAMES),
            )
        self.assertFalse(tasks["route_off"]["routing"]["route_enabled"])
        self.assertTrue(
            tasks["route_off"]["routing"]["route_off_after_condition_validation"]
        )
        self.assertTrue(tasks["route_off"]["routing"]["oracle_assets_validated"])
        self.assertEqual(
            tasks["route_off"]["routing"]["direct_runtime_conditions"],
            [
                "source_video", "edit_instruction",
                "object_trajectory_scaffold", "aux_bone_removed_source",
            ],
        )
        self.assertEqual(
            tasks["route_off"]["routing"]["renderer_conditions_consumed"],
            ["source_video", "edit_instruction"],
        )
        self.assertEqual(
            tasks["route_off"]["routing"]["oracle_runtime_conditions_consumed"],
            [],
        )
        self.assertEqual(
            tasks["route_off"]["routing"]["derived_scaffold_authorities"],
            ["stage0_object_masks", "g0_mouth_track"],
        )
        self.assertFalse(
            tasks["route_off"]["routing"][
                "raw_stage0_masks_accessed_at_runtime"
            ]
        )
        self.assertTrue(
            tasks["trajectory_bone_only"]["routing"][
                "source_bone_trajectory_projection_enabled"
            ]
        )
        self.assertFalse(
            tasks["trajectory_bone_only"]["routing"][
                "dog_identity_projection_enabled"
            ]
        )
        self.assertTrue(
            tasks["trajectory_dog_bone"]["routing"][
                "dog_identity_projection_enabled"
            ]
        )
        self.assertEqual(
            tasks["trajectory_dog_bone"]["routing"][
                "renderer_conditions_consumed"
            ],
            [
                "source_video", "edit_instruction",
                "object_trajectory_scaffold", "aux_bone_removed_source",
            ],
        )

    def test_placeholder_cannot_be_relabelled_launchable(self) -> None:
        bad = copy.deepcopy(self.plan)
        bad["launch_allowed"] = True
        bad["status"] = "READY_FOR_EXPLICIT_LOCAL_LAUNCH"
        bad["hold_reasons"] = []
        bad["plan_digest"] = evaluator.object_sha256(
            {key: value for key, value in bad.items() if key != "plan_digest"}
        )
        with self.assertRaisesRegex(evaluator.ObjectTrajectoryEvalError, "incomplete"):
            evaluator.validate_plan(bad)

        dirty_legacy = copy.deepcopy(self.plan)
        dirty_legacy["producer"]["infer_lora_sha256"] = (
            "0c79faa8417a40a5735571db3a5ba828d6aa977d7d0507a5bfcb63368c07728d"
        )
        dirty_legacy["plan_digest"] = evaluator.object_sha256(
            {
                key: value
                for key, value in dirty_legacy.items()
                if key != "plan_digest"
            }
        )
        with self.assertRaisesRegex(evaluator.ObjectTrajectoryEvalError, "producer value"):
            evaluator.validate_plan(dirty_legacy)

    def test_reorder_and_authority_removal_fail_after_digest_reseal(self) -> None:
        reordered = copy.deepcopy(self.plan)
        reordered["tasks"][1], reordered["tasks"][2] = (
            reordered["tasks"][2], reordered["tasks"][1]
        )
        reordered["plan_digest"] = evaluator.object_sha256(
            {key: value for key, value in reordered.items() if key != "plan_digest"}
        )
        with self.assertRaisesRegex(evaluator.ObjectTrajectoryEvalError, "closure"):
            evaluator.validate_plan(reordered)

        removed = copy.deepcopy(self.plan)
        removed["tasks"][2]["external_conditions"].pop("stage0_masks")
        removed["plan_digest"] = evaluator.object_sha256(
            {key: value for key, value in removed.items() if key != "plan_digest"}
        )
        with self.assertRaisesRegex(evaluator.ObjectTrajectoryEvalError, "closure"):
            evaluator.validate_plan(removed)

        forged_audit = copy.deepcopy(self.plan)
        authority = forged_audit["admission_authorities"][
            "scaffold_independent_audit"
        ]
        authority["payload_digest"] = "0" * 64
        authority["authority_digest"] = evaluator.object_sha256(
            {
                key: value
                for key, value in authority.items()
                if key != "authority_digest"
            }
        )
        forged_audit["plan_digest"] = evaluator.object_sha256(
            {
                key: value
                for key, value in forged_audit.items()
                if key != "plan_digest"
            }
        )
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "audit authority pin"
        ):
            evaluator.validate_plan(forged_audit)

    def test_real_wrapper_receipts_cross_validate_and_reject_lies(self) -> None:
        producer = self.plan["producer"]
        route_task = self.plan["tasks"][1]
        bone_task = self.plan["tasks"][2]
        dog_task = self.plan["tasks"][3]
        for task in (route_task, bone_task, dog_task):
            receipt = _custom_receipt(task, producer)
            evaluator.validate_custom_inference_receipt(
                receipt, task, producer
            )
            self.assertEqual(
                receipt["input"]["accepted_model_conditions"],
                list(
                    evaluator.BASE_CONDITION_NAMES
                    + evaluator.EXTERNAL_CONDITION_NAMES
                ),
            )
            self.assertEqual(
                receipt["input"]["direct_runtime_conditions"],
                task["routing"]["direct_runtime_conditions"],
            )
            self.assertEqual(
                receipt["input"]["derived_scaffold_authorities"],
                task["routing"]["derived_scaffold_authorities"],
            )
            runtime = receipt["object_oracle"]["runtime"]
            self.assertEqual(
                runtime["direct_runtime_conditions_consumed"],
                task["routing"]["renderer_conditions_consumed"],
            )
            self.assertEqual(
                runtime["oracle_runtime_conditions_consumed"],
                task["routing"]["oracle_runtime_conditions_consumed"],
            )
            if task["oracle_arm"] == "route_off":
                self.assertEqual(runtime["projection_collective_gates"], [])
            else:
                trace = receipt["sampling"]["source_onset_solver_trace"]
                self.assertEqual(
                    [
                        gate["stage"]
                        for gate in trace["projection_collective_gates"]
                    ],
                    list(evaluator.PROJECTION_COLLECTIVE_STAGES),
                )
                self.assertEqual(
                    runtime["projection_collective_gates"],
                    trace["projection_collective_gates"],
                )
                self.assertEqual(
                    len(trace["aux_collective_gates"])
                    + len(trace["projection_collective_gates"])
                    + 1,
                    evaluator.EXPECTED_SUCCESSFUL_ALL_GATHER_OBJECT_CALLS,
                )

        receipt = _custom_receipt(dog_task, producer)
        lied = copy.deepcopy(receipt)
        lied["input"]["accepted_model_conditions"].remove(
            "stage0_object_masks"
        )
        _reseal(lied, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "input differs"
        ):
            evaluator.validate_custom_inference_receipt(
                lied, dog_task, producer
            )

        raw_lie = copy.deepcopy(receipt)
        raw_lie["input"]["raw_stage0_masks_accessed_at_runtime"] = True
        _reseal(raw_lie, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "input differs"
        ):
            evaluator.validate_custom_inference_receipt(
                raw_lie, dog_task, producer
            )

        wrong_source = copy.deepcopy(receipt)
        wrong_source["object_oracle"]["producer_hashes"][
            "projection_source_sha256"
        ] = "9" * 64
        _reseal(wrong_source, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "asset/producer"
        ):
            evaluator.validate_custom_inference_receipt(
                wrong_source, dog_task, producer
            )

        missing_baseline = copy.deepcopy(receipt)
        trace = missing_baseline["sampling"]["source_onset_solver_trace"]
        trace["row_construction"]["row_names"].pop(0)
        _reseal(trace, "trace_digest")
        missing_baseline["object_oracle"]["runtime"]["projection_trace"] = (
            copy.deepcopy(trace)
        )
        _reseal(missing_baseline, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "row construction"
        ):
            evaluator.validate_custom_inference_receipt(
                missing_baseline, dog_task, producer
            )

        bad_step_count = copy.deepcopy(receipt)
        for trace in (
            bad_step_count["sampling"]["source_onset_solver_trace"],
            bad_step_count["object_oracle"]["runtime"]["projection_trace"],
        ):
            trace["tensor_core"]["step_count"] = 39
            _reseal(trace, "trace_digest")
        _reseal(bad_step_count, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "tensor core"
        ):
            evaluator.validate_custom_inference_receipt(
                bad_step_count, dog_task, producer
            )

        reordered_projection = copy.deepcopy(receipt)
        reordered_trace = reordered_projection["sampling"][
            "source_onset_solver_trace"
        ]
        gates = reordered_trace["projection_collective_gates"]
        gates[3], gates[4] = gates[4], gates[3]
        _reseal(reordered_trace, "trace_digest")
        reordered_projection["object_oracle"]["runtime"][
            "projection_trace"
        ] = copy.deepcopy(reordered_trace)
        reordered_projection["object_oracle"]["runtime"][
            "projection_collective_gates"
        ] = copy.deepcopy(
            reordered_trace["projection_collective_gates"]
        )
        _reseal(reordered_projection, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "projection_projector_lookup four-rank gate",
        ):
            evaluator.validate_custom_inference_receipt(
                reordered_projection, dog_task, producer
            )

        duplicate_relabel = copy.deepcopy(receipt)
        forged_trace = duplicate_relabel["sampling"][
            "source_onset_solver_trace"
        ]
        forged_gate = copy.deepcopy(
            forged_trace["projection_collective_gates"][1]
        )
        forged_gate["stage"] = "projection_contract_build"
        forged_trace["projection_collective_gates"][2] = forged_gate
        forged_trace["row_construction"][
            "pre_projection_contract_gate"
        ] = copy.deepcopy(forged_gate)
        _reseal(forged_trace, "trace_digest")
        duplicate_relabel["object_oracle"]["runtime"][
            "projection_trace"
        ] = copy.deepcopy(forged_trace)
        duplicate_relabel["object_oracle"]["runtime"][
            "projection_collective_gates"
        ] = copy.deepcopy(forged_trace["projection_collective_gates"])
        _reseal(duplicate_relabel, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "projection_contract_build four-rank gate",
        ):
            evaluator.validate_custom_inference_receipt(
                duplicate_relabel, dog_task, producer
            )

        missing_runtime_stage = copy.deepcopy(receipt)
        missing_runtime_stage["object_oracle"]["runtime"][
            "projection_collective_gates"
        ].pop()
        _reseal(missing_runtime_stage, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "runtime consumption"
        ):
            evaluator.validate_custom_inference_receipt(
                missing_runtime_stage, dog_task, producer
            )

        runtime_trace_mismatch = copy.deepcopy(receipt)
        runtime_trace_mismatch["object_oracle"]["runtime"][
            "projection_collective_gates"
        ] = copy.deepcopy(
            runtime_trace_mismatch["object_oracle"]["runtime"][
                "projection_collective_gates"
            ]
        )
        runtime_trace_mismatch["object_oracle"]["runtime"][
            "projection_collective_gates"
        ][0]["ordered_status_digest"] = "0" * 64
        _reseal(runtime_trace_mismatch, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "projection_runtime_readiness four-rank gate",
        ):
            evaluator.validate_custom_inference_receipt(
                runtime_trace_mismatch, dog_task, producer
            )

        false_row_stage = copy.deepcopy(receipt)
        for trace in (
            false_row_stage["sampling"]["source_onset_solver_trace"],
            false_row_stage["object_oracle"]["runtime"]["projection_trace"],
        ):
            trace["row_construction"]["final_validation_gate"] = copy.deepcopy(
                trace["row_construction"]["projector_install_gate"]
            )
            _reseal(trace, "trace_digest")
        _reseal(false_row_stage, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError,
            "projection_final_validation four-rank gate",
        ):
            evaluator.validate_custom_inference_receipt(
                false_row_stage, dog_task, producer
            )

        route_lie = _custom_receipt(route_task, producer)
        route_lie["object_oracle"]["assets"]["aux_bone_removed_source"][
            "consumed_via_retained_fd"
        ] = True
        _reseal(route_lie, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "retained authority"
        ):
            evaluator.validate_custom_inference_receipt(
                route_lie, route_task, producer
            )

    def test_legacy_v5_hard1_null_receipt_is_validated_locally(self) -> None:
        task = self.plan["tasks"][0]
        receipt = _legacy_receipt(task, self.plan["producer"])
        evaluator.validate_off_inference_receipt(
            receipt, task, self.plan["producer"]
        )
        lied = copy.deepcopy(receipt)
        lied["sampling"]["source_onset_policy"] = "none"
        _reseal(lied, "receipt_digest")
        with self.assertRaisesRegex(
            evaluator.ObjectTrajectoryEvalError, "sampling coordinates"
        ):
            evaluator.validate_off_inference_receipt(
                lied, task, self.plan["producer"]
            )

    def test_null_envelope_allows_different_output_bytes(self) -> None:
        before = _legacy_receipt(
            self.plan["tasks"][0], self.plan["producer"], output_sha="4" * 64
        )
        after = _legacy_receipt(
            self.plan["tasks"][4], self.plan["producer"], output_sha="5" * 64
        )
        result = evaluator.validate_null_envelope_receipts(before, after)
        self.assertFalse(result["output_byte_equality_required"])
        self.assertFalse(result["observed_output_sha256_equal"])
        self.assertFalse(result["historical_exact_sha_gate_applied"])

    def test_verified_result_preserves_frozen_full644_replay_arm(self) -> None:
        for task, oracle_arm in zip(self.plan["tasks"], evaluator.ARM_ORDER):
            coordinates = evaluator._verified_result_coordinates(task)
            self.assertEqual(
                {
                    key: coordinates[key]
                    for key in ("task_id", "arm", "receipt_path", "output_path")
                },
                {
                    "task_id": task["task_id"],
                    "arm": "full644",
                    "receipt_path": task["output"]["receipt_path"],
                    "output_path": task["output"]["video_path"],
                },
            )
            self.assertEqual(coordinates["oracle_arm"], oracle_arm)

    def test_runner_rebinds_frozen_arguments_and_passes_external_authorities(self) -> None:
        # Import here so source-loaded module names are created only once.
        from methods.bernini_action_editing import (
            case01_object_trajectory_exact5_runner_v1 as runner,
        )

        self.assertEqual(tuple(runner.frozen.TASK_IDS), evaluator.TASK_IDS)
        self.assertIs(runner.frozen.build_inference_arguments, runner.build_inference_arguments)
        active = copy.deepcopy(self.plan["tasks"][3])
        arguments = runner.build_inference_arguments(
            plan=self.plan,
            task=active,
            bernini_root="/bernini",
            veomni_root="/veomni",
            model_view_root="/model",
            consumption_input_path="/consumption.json",
            consumption_input_sha256="6" * 64,
            consumption_input_digest="7" * 64,
            source_authority={"path": active["source_video"]},
            adapter_view_root="/adapter",
        )
        self.assertIn("--object-oracle-arm", arguments)
        self.assertEqual(
            arguments[arguments.index("--object-oracle-arm") + 1],
            "trajectory_dog_bone",
        )
        for flag in (
            "--object-oracle-scaffold",
            "--object-oracle-scaffold-sha256",
            "--object-oracle-scaffold-digest",
            "--object-oracle-bone-removed-video",
            "--object-oracle-bone-removed-video-sha256",
        ):
            self.assertIn(flag, arguments)

        null_arguments = runner.build_inference_arguments(
            plan=self.plan,
            task=self.plan["tasks"][0],
            bernini_root="/bernini",
            veomni_root="/veomni",
            model_view_root="/model",
            consumption_input_path="/consumption.json",
            consumption_input_sha256="6" * 64,
            consumption_input_digest="7" * 64,
            source_authority={"path": active["source_video"]},
            adapter_view_root="/adapter",
        )
        self.assertEqual(
            null_arguments[null_arguments.index("--object-oracle-arm") + 1], "off"
        )
        self.assertNotIn("--object-oracle-scaffold", null_arguments)
        for task in self.plan["tasks"]:
            task_arguments = runner.build_inference_arguments(
                plan=self.plan,
                task=task,
                bernini_root="/bernini",
                veomni_root="/veomni",
                model_view_root="/model",
                consumption_input_path="/consumption.json",
                consumption_input_sha256="6" * 64,
                consumption_input_digest="7" * 64,
                source_authority={"path": task["source_video"]},
                adapter_view_root="/adapter",
            )
            self.assertEqual(task_arguments.count("--source-onset-policy"), 1)
            self.assertEqual(
                task_arguments[
                    task_arguments.index("--source-onset-policy") + 1
                ],
                "hard1_every_step",
            )

    def test_runner_refuses_hold_before_allocation_arguments_are_needed(self) -> None:
        from methods.bernini_action_editing import (
            case01_object_trajectory_exact5_runner_v1 as runner,
        )

        plan_path = self.root / "held-plan.json"
        raw = evaluator.canonical_json_bytes(self.plan) + b"\n"
        plan_path.write_bytes(raw)
        args = SimpleNamespace(
            plan=str(plan_path), plan_sha256=hashlib.sha256(raw).hexdigest()
        )
        with self.assertRaisesRegex(
            runner.frozen.MatchedRunnerV2Error, "checkpoint pin is incomplete"
        ):
            runner.execute(args)

    def test_failed_runner_bootstrap_rolls_back_source_loaded_modules(self) -> None:
        runner_path = METHOD_ROOT / "case01_object_trajectory_exact5_runner_v1.py"
        program = r'''
import re, sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
s=p.read_text(encoding="utf-8")
s,n=re.subn(
 r'(OBJECT_TRAJECTORY_EVAL_SHA256\s*=\s*\(\s*")[0-9a-f]{64}("\s*\))',
 lambda m:m.group(1)+"0"*64+m.group(2),s,count=1,flags=re.S)
if n!=1: raise SystemExit("pin substitution differs")
messages=[]
for index in range(2):
 try:
  exec(compile(s,str(p),"exec"),{"__name__":f"_failed_probe_{index}","__file__":str(p)})
 except Exception as error:
  messages.append(type(error).__name__+":"+str(error))
 else:
  raise SystemExit("stale evaluator pin unexpectedly loaded")
if len(set(messages))!=1 or "source identity differs" not in messages[0]:
 raise SystemExit("failed bootstrap was not repeatable: "+repr(messages))
leaked=[name for name in sys.modules if name.startswith("_case01_exact5_") or name.startswith("_case01_object_trajectory_source_loaded") or name=="_case01_source_bone_exact5_eval_v1"]
if leaked: raise SystemExit("source-loaded modules leaked: "+repr(leaked))
print(messages[0])
'''
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", program, str(runner_path)])
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertIn("source identity differs", completed.stdout)

    def test_hostile_resealed_plan_fails_identically_normal_and_optimized(self) -> None:
        plan_path = self.root / "hold-plan.json"
        plan_path.write_bytes(evaluator.canonical_json_bytes(self.plan) + b"\n")
        program = r'''
import copy, json, sys
from pathlib import Path
from methods.bernini_action_editing import case01_object_trajectory_exact5_eval_v1 as e
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
e.validate_plan(p)
bad=copy.deepcopy(p)
bad["tasks"][2]["routing"]["dog_identity_projection_enabled"]=True
bad["plan_digest"]=e.object_sha256({k:v for k,v in bad.items() if k!="plan_digest"})
try:
    e.validate_plan(bad)
except e.ObjectTrajectoryEvalError as error:
    message=str(error)
else:
    raise SystemExit("hostile reseal unexpectedly passed")
print(e.object_sha256(p)+"|"+message)
'''
        outputs = []
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", program, str(plan_path)])
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_contract_sources_have_no_optimized_away_asserts(self) -> None:
        paths = (
            METHOD_ROOT / "case01_object_trajectory_exact5_eval_v1.py",
            METHOD_ROOT / "case01_object_trajectory_exact5_runner_v1.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
            self.assertEqual(asserts, [], f"{path.name} contains assert contracts")


if __name__ == "__main__":
    unittest.main()
