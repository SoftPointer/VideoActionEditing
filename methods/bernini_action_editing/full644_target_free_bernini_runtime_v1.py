#!/usr/bin/env python3
"""Stochastic exact40 Bernini rollout and target-free preference runtime.

The production boundary in :mod:`full644_target_free_preference_v1` quite
deliberately rejects every non-empty preference set.  This module is the
separate concrete runtime which may eventually authorize such an update.  It
owns the Gaussian sampling kernel, the exact40 trajectory artifact, sequential
UniPC replay, Bernini branch VJPs, DP2 x SP4 gradient reduction, and the
decoded-verifier adapter.  There is no callback for supplying log
probabilities or gradients.

The older calibrated decoded gate remains a diagnostic adapter because it has
no calibrated non-target-motion axis.  The production engineering path accepts
only the separately frozen Qwen exact-eight verifier authority, reopens its
candidate receipts, and still performs a true zero update whenever either
endpoint is ineligible or undetermined.  Manifest ``axis_pass`` values never
authorize an update, and every emitted claim remains engineering-only rather
than a scientific result.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import types
from typing import Any, Callable, Iterable, Mapping, NoReturn, Optional, Sequence
import weakref


SCHEMA_VERSION = "bernini-full644-stochastic-exact40-runtime-v1"
TRAJECTORY_SCHEMA = "bernini-full644-stochastic-exact40-trajectory-v1"
TRAJECTORY_ARTIFACT_SCHEMA = (
    "bernini-full644-stochastic-exact40-fp32-artifact-v1"
)
TRAJECTORY_STEP_SCHEMA = "bernini-full644-stochastic-exact40-step-v1"
VERIFIER_ADAPTER_SCHEMA = "bernini-full644-hard-axis-gate-adapter-v1"
VERIFIER_INPUT_SCHEMA = "bernini-full644-action-preservation-evidence-v1"
PREFLIGHT_SCHEMA = "bernini-full644-stochastic-one-source-preflight-v1"
UPDATE_SCHEMA = "bernini-full644-stochastic-one-update-v1"
DECODED_ROLLOUT_SCHEMA = "bernini-full644-stochastic-decoded-rollout-v1"

_ROLLOUT_STAGE_LOCAL_FIELDS = frozenset(
    {
        "world_rank",
        "dp_arm",
        "sp_rank",
        "rollout_id",
        "rollout_seed",
        "behavior_policy_sha256",
        "trajectory_receipt_path",
        "trajectory_receipt_sha256",
        "trajectory_receipt_digest",
        "trajectory_artifact_sha256",
        "terminal_state_sha256",
        "decoded_rollout_receipt_path",
        "decoded_rollout_receipt_sha256",
        "decoded_rollout_receipt_digest",
        "candidate_media_path",
        "candidate_media_sha256",
        "candidate_full_decode_tree_digest",
        "candidate_exact81_25fps",
        "peak_memory_allocated_bytes",
        "total_device_memory_bytes",
        "row_digest",
    }
)

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
TRAJECTORY_STEPS = 40
FRAME_COUNT = 81
FPS = 25.0
HARD_AXES = (
    "event",
    "participant",
    "ordered_transition",
    "terminal_hold",
    "identity",
    "camera",
    "background",
    "non_target_motion",
)
GATE_AXES = (
    "action_order",
    "onset",
    "source_identity",
    "background",
    "camera",
    "quality",
)

# One fixed, non-zero isotropic action standard deviation.  The hexadecimal
# literal is the cross-language authority; the decimal spelling is not.
ACTION_STD_FLOAT32_BE_HEX = "3d4ccccd"  # float32(0.05)
ACTION_STD = struct.unpack(">f", bytes.fromhex(ACTION_STD_FLOAT32_BE_HEX))[0]
PREFERENCE_BETA_FLOAT32_BE_HEX = "3f800000"  # float32(1.0)
PREFERENCE_BETA = struct.unpack(
    ">f", bytes.fromhex(PREFERENCE_BETA_FLOAT32_BE_HEX)
)[0]
GAUSSIAN_SCORE_REDUCTION = "mean_over_latent_elements_then_sum_exact40"
ACTIVATION_CHECKPOINT_PROFILE = "selective-nonreentrant-stride4-exact8"
ACTIVATION_CHECKPOINT_BLOCKS = tuple(range(0, 30, 4))
LORA_RANK = 256
LORA_ALPHA = 256
LORA_AFFINES = 240
LORA_TENSOR_COUNT = 480
LORA_PARAMETER_COUNT = 188_743_680
PEFT_VERSION = "0.19.1"
TORCH_VERSION = "2.7.1+rocm6.3"
TRANSFORMERS_VERSION = "5.5.4"
DIFFUSERS_VERSION = "0.38.0"
DECORD_VERSION = "0.6.0"
SAFETENSORS_VERSION = "0.8.0-rc.0"
MIOPEN_BACKEND_VERSION = 3_003_000
GIT_EXECUTABLE = Path("/usr/bin/git")
GIT_EXECUTABLE_SHA256 = (
    "fd7c9389e200d626b46551835e5233bbde49a6a2326f9ebb85c70ed235861001"
)
GIT_EXECUTABLE_SIZE = 3_710_360
GIT_EXECUTABLE_MODE = 0o755
_PEFT_CONFIG_FIELDS = frozenset(
    {
        "alora_invocation_tokens", "alpha_pattern", "arrow_config", "auto_mapping",
        "base_model_name_or_path", "bias", "corda_config", "ensure_weight_tying",
        "eva_config", "exclude_modules", "fan_in_fan_out", "inference_mode",
        "init_lora_weights", "layer_replication", "layers_pattern",
        "layers_to_transform", "loftq_config", "lora_alpha", "lora_bias",
        "lora_dropout", "lora_ga_config", "megatron_config", "megatron_core",
        "modules_to_save", "peft_type", "peft_version", "qalora_group_size", "r",
        "rank_pattern", "revision", "target_modules", "target_parameters",
        "task_type", "trainable_token_indices", "use_bdlora", "use_dora",
        "use_qalora", "use_rslora",
    }
)
_PEFT_REQUESTED_TARGET_CONTRACT = "requested_exact240_full_module_paths"
_PEFT_CANONICAL_TARGET_CONTRACT = "postinstall_exact4_unique_suffixes"
_PEFT_CANONICAL_TARGET_MODULES = frozenset(
    {"to_q", "to_k", "to_v", "to_out.0"}
)
_PEFT_CONFIG_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "peft_version",
        "target_modules_contract",
        "target_module_count",
        "target_modules",
        "target_modules_sha256",
        "config",
        "config_digest",
        "receipt_digest",
    }
)
GAUSSIAN_KERNEL = {
    "schema_version": "bernini-full644-fixed-gaussian-action-kernel-v1",
    "distribution": "isotropic_normal",
    "mean": "current_policy_guided_velocity",
    "std_float32_be_hex": ACTION_STD_FLOAT32_BE_HEX,
    "std_strictly_positive": True,
    "per_step_reduction": "mean_over_all_latent_elements",
    "trajectory_reduction": "sum_exact40",
    "recorded_action_role": "executed_unipc_model_output",
}
GAUSSIAN_KERNEL_SHA256 = hashlib.sha256(
    json.dumps(
        GAUSSIAN_KERNEL,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
APG_GUIDANCE = {
    "schema_version": "bernini-full644-current-policy-apg-guidance-v1",
    "guidance_scale_float64_hex": float(4.0).hex(),
    "momentum_float64_hex": float(0.0).hex(),
    "eta_float64_hex": float(0.5).hex(),
    "norm_threshold_float64_hex": float(50.0).hex(),
    "fresh_zero_momentum_each_step": True,
    "branch_order": ["negative", "positive"],
}
APG_GUIDANCE_SHA256 = hashlib.sha256(
    json.dumps(
        APG_GUIDANCE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()

SCHEDULE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
ACTION_GATE_SOURCE_SHA256 = (
    "399b9d31b6e830a55cb6542f9273d86809dfd5a3b705a7c7fee0dca915e74ee0"
)
ACTION_GATE_SOURCE_SIZE = 103679
FULL644_CATALOG_PATH = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/"
    "full644_source_only_catalog_707f70ab_v1/source_catalog.json"
)
FULL644_CATALOG_SHA256 = (
    "d049770159d97fd59d13c2960c521afa41a0c04139a93bca0c372388d0c8b89b"
)
FULL644_CATALOG_SIZE = 973153
FULL644_CATALOG_DIGEST = (
    "143e91321038b7eb218bbbb8c2b365cd4749258656366a32931e65030d29809d"
)
FULL644_CATALOG_RECEIPT_PATH = FULL644_CATALOG_PATH.with_name(
    "source_catalog_receipt.json"
)
FULL644_CATALOG_RECEIPT_SHA256 = (
    "f8b3cb9fa70fbc44c7f2f8e1cd8b936a86f5318a0dffebdf3e0117b435789794"
)
FULL644_CATALOG_RECEIPT_SIZE = 3024
FULL644_CATALOG_POSTFLIGHT_PATH = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/controllers/"
    "full644_source_only_707f70ab_v1/postflight_release_v3.json"
)
FULL644_CATALOG_POSTFLIGHT_SHA256 = (
    "a512a0837dbd1e5ad8803f465c78f2af4c00a180d966ea1575ff2aeb8455e230"
)
FULL644_CATALOG_POSTFLIGHT_SIZE = 5221
ONE_SOURCE_ROW_ID = "0da02d985d0e4b6f"
ONE_SOURCE_VIDEO_SHA256 = (
    "92db18f69f008c04a58ed37a3c3485d232a743a74923cef91b16c2245db61873"
)
ONE_SOURCE_INSTRUCTION_SHA256 = (
    "01e886584db31334f8933696b94dff84f4b809c719faa6c59d801d63f37ebeaf"
)
ONE_SOURCE_ROW_DIGEST = (
    "7669cd1415757f7726a08a7e36217f68469e47fb914d7a01ed207ccec0ff6d9b"
)
FULL644_CATALOG_RECEIPT_DIGEST = (
    "3ac42bf6be71460cd70bd812bea99f871e78f2d5e6f988cf0befb0fd7e2f3eac"
)
FULL644_CATALOG_POSTFLIGHT_DIGEST = (
    "acc056e5f3bcded5ca5516bbd2189c910a900902b711acdc43215b71a09b245b"
)
FULL644_CATALOG_EXTRACTOR_SHA256 = (
    "707f70aba9fdc056c03b1d9590bb4f8f03dc5d9123292177c9e91ea95c7b66c9"
)
FULL644_CATALOG_EXTRACTOR_SIZE = 42567
FULL644_CATALOG_CONTROLLER_SHA256 = (
    "8d0f8946fc5422d2b68028956aaa492c6bf3b49f55e00b15edd777cc3997f2ef"
)
FULL644_CATALOG_CONTROLLER_SIZE = 4027
FULL644_CATALOG_PRE_ADMISSION_SHA256 = (
    "270f110c96ea9398557ece5ad57d63bf5ad8f48d83b0b8d886dc084f6b4ef6ae"
)
FULL644_CATALOG_PRE_ADMISSION_SIZE = 4043
FULL644_CATALOG_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
FULL644_CATALOG_FFPROBE_SIZE = 216841
QWEN_VERIFIER_SOURCE_SHA256 = (
    "5d94a74de54498150497c29f316bece3b308075b8e9da8ba12b46f6141689f37"
)
QWEN_VERIFIER_SOURCE_SIZE = 105108
QWEN_MODEL_CLOSURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "motive/audits/qwen25_vl_7b_cc594898_model_closure.json"
)
QWEN_MODEL_CLOSURE_SHA256 = (
    "6cf8c51b8db5ff36506649ea1d9b9efa79a50ad7080ba7337a208f2ee3a8f7c6"
)
QWEN_MODEL_CLOSURE_SIZE = 3094
QWEN_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN_MODEL_SNAPSHOT_DIGEST = (
    "26c7eda811b101c0265042056aea0568858c56f8fd14ff20c2e44e130542a442"
)
QWEN_DETERMINISTIC_GENERATION = {
    "schema_version": "bernini-full644-qwen25-vl-deterministic-generation-v1",
    "model_closure_sha256": QWEN_MODEL_CLOSURE_SHA256,
    "model_closure_size_bytes": QWEN_MODEL_CLOSURE_SIZE,
    "model_revision": QWEN_MODEL_REVISION,
    "transformers_version": TRANSFORMERS_VERSION,
    "local_files_only": True,
    "trust_remote_code": False,
    "model_eval": True,
    "inference_mode": True,
    "torch_dtype": "bfloat16",
    "attention_implementation": "eager",
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 2048,
    "seed": 0,
    "source_frame_indices": [0, 7, 15, 22, 29, 36, 44, 51, 58, 65, 73, 80],
    "candidate_frame_indices": [0, 7, 15, 22, 29, 36, 44, 51, 58, 65, 73, 80],
    "prompt_sha256": "8a01e30ffe644ffac2ebf40eb8ee96fca7f13514bd56dc6d5e47e841dcaba5cb",
    "response_schema": "bernini-full644-qwen-exact8-response-v1",
}
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BASE_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
BASE_CHECKPOINT_FILE_COUNT = 23
OWNED_FACTORY_SEED = 20260817
PILOT_ROUND_INDEX = 0
PILOT_ROLLOUT_SEEDS = (2026081700, 2026081701)
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRADIENT_NORM = 1.0
MINIMUM_TRUE_GPU_MEMORY_FRACTION = 0.50
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

_HELD_LOCAL_SOURCE_BINDINGS = {
    "full644_target_free_preference_v1": (
        "full644_target_free_preference_v1.py",
        "e549fd2a4007b7be505db5237644f5fe33deceb79964ed907995e98626be2261",
        42254,
    ),
    "source_self_runtime": (
        "source_self_runtime.py",
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        36607,
    ),
    "inference_sigma_strata": (
        "inference_sigma_strata.py",
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
        17956,
    ),
    "packed_preservation_lora_v2": (
        "packed_preservation_lora_v2.py",
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
        30419,
    ),
}
_PINNED_BERNINI_SOURCE_SHA256 = {
    "bernini/models/renderer.py": "fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159",
    "bernini/models/wan_diffusion.py": "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
    "bernini/models/transformer_wan.py": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
    "bernini/models/scheduler.py": "b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97",
    "bernini/training/data.py": "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
    "bernini/attention.py": "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
    "bernini/parallel/state.py": "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
    "bernini/parallel/ops.py": "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
    "configs/bernini_renderer_wan21_1p3b/config.json": "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_LORA_NAME = re.compile(
    r"base_model\.model\.diff_dec\.transformer\.blocks\."
    r"(?P<block>[0-9]+)\.attn(?P<attention>[12])\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)\."
    r"lora_(?P<side>[AB])\.default\.weight\Z"
)
_ARTIFACT_MAGIC = b"BTF40FP32V1\x00\x00\x00\x00\x00"
_HEADER_SIZE_BYTES = 8
_MAX_HEADER_BYTES = 1024 * 1024
_OWNED_RUNTIME_TOKEN = object()

_TRAJECTORY_FIELDS = frozenset(
    {
        "schema_version",
        "runtime_schema_version",
        "rollout_id",
        "source_row_id",
        "source_video_sha256",
        "instruction_sha256",
        "behavior_policy_sha256",
        "round_index",
        "rollout_seed",
        "dp_arm",
        "sp_size",
        "step_count",
        "latent_shape",
        "latent_numel",
        "latent_dtype",
        "schedule_sha256",
        "gaussian_kernel",
        "gaussian_kernel_sha256",
        "apg_guidance_sha256",
        "initial_noise_key_sha256",
        "initial_state_sha256",
        "steps",
        "terminal_state_sha256",
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "artifact_mode_octal",
        "artifact_nlink",
        "sp4_noise_broadcast",
        "sp4_step_consensus",
        "source_only_input",
        "paired_reference_read_count",
        "external_velocity_read_count",
        "receipt_digest",
    }
)
_STEP_FIELDS = frozenset(
    {
        "schema_version",
        "schedule_index",
        "timestep",
        "sigma_float32_be_hex",
        "state_before_sha256",
        "policy_mean_sha256",
        "action_noise_key_sha256",
        "action_noise_sha256",
        "executed_action_sha256",
        "state_after_sha256",
        "scheduler_step_index_after",
        "step_digest",
    }
)
_DECODED_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version", "rollout_id", "behavior_policy_sha256", "round_index",
        "rollout_seed", "dp_arm", "source_row_id", "source_video_sha256",
        "instruction_sha256", "trajectory_receipt_path",
        "trajectory_receipt_sha256", "trajectory_receipt_digest",
        "trajectory_artifact_path", "trajectory_artifact_sha256",
        "trajectory_artifact_size_bytes", "terminal_state_sha256",
        "normalized_latent_path", "normalized_latent_sha256",
        "normalized_latent_tensor_sha256", "candidate_media_path",
        "candidate_media_sha256", "candidate_media_size_bytes",
        "candidate_frame_count", "fps_numerator", "fps_denominator", "width",
        "height", "full_decode_frame_sha256", "full_decode_tree_digest",
        "vae_authority", "source_encode_and_terminal_decode_same_vae_authority",
        "target_media_read_count", "receipt_digest",
    }
)
_VAE_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version", "base_checkpoint_tree_sha256",
        "checkpoint_content_manifest_sha256", "checkpoint_snapshot_digest",
        "vae_file_inventory_digest", "vae_config_sha256",
    }
)
_ARTIFACT_HEADER_FIELDS = frozenset(
    {
        "schema_version",
        "dtype",
        "byte_order",
        "tensor_count",
        "tensors",
        "payload_size_bytes",
        "header_digest",
    }
)
_ARTIFACT_TENSOR_FIELDS = frozenset(
    {"name", "shape", "numel", "offset", "size_bytes", "sha256"}
)
_VERIFIER_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "source_video_sha256",
        "candidate_video_sha256",
        "frame_count",
        "fps",
        "measurement_path",
        "measurement_sha256",
        "calibration_path",
        "calibration_sha256",
        "gate_source_sha256",
        "gate_source_size_bytes",
        "independent_from_student",
        "student_parameters_or_loss_read",
        "evidence_digest",
    }
)
_QWEN_VERDICT_FIELDS = frozenset(
    {
        "schema_version",
        "verifier_release_sha256",
        "model_closure_sha256",
        "model_revision",
        "rollout_id",
        "policy_sha256",
        "round_index",
        "seed",
        "dp_arm",
        "source_row_id",
        "source_video_sha256",
        "instruction_sha256",
        "decoded_rollout_receipt_path",
        "decoded_rollout_receipt_sha256",
        "decoded_rollout_receipt_digest",
        "trajectory_receipt_sha256",
        "trajectory_receipt_digest",
        "trajectory_artifact_sha256",
        "terminal_state_sha256",
        "candidate_media_path",
        "candidate_media_sha256",
        "source_media_probe",
        "candidate_media_probe",
        "visual_input_sha256",
        "visual_execution",
        "raw_response_sha256",
        "hard_axes",
        "uncertainty_codes",
        "qualification",
        "deterministic_generation",
        "independent_from_student",
        "student_parameters_or_loss_read",
        "engineering_only",
        "scientific_result_claimed",
        "receipt_digest",
    }
)
_QWEN_AXIS_FIELDS = frozenset({"state", "evidence"})
_QWEN_EVIDENCE_ROW_FIELDS = frozenset(
    {"source_frames", "candidate_frames", "observation"}
)
_QWEN_MEDIA_PROBE_FIELDS = frozenset(
    {
        "schema_version", "media_sha256", "frame_count", "fps_numerator",
        "fps_denominator", "width", "height", "fully_decoded",
        "full_decode_frame_sha256", "full_decode_tree_digest",
    }
)
_QWEN_QUALIFICATION_FIELDS = frozenset(
    {
        "eligible_for_engineering_pair_selection", "all_eight_axes_pass",
        "any_axis_fail", "any_axis_undetermined",
    }
)
_QWEN_DETERMINISTIC_FIELDS = frozenset(
    {
        "schema_version", "model_closure_sha256", "model_closure_size_bytes",
        "model_revision", "transformers_version", "local_files_only",
        "trust_remote_code", "model_eval", "inference_mode", "torch_dtype",
        "attention_implementation", "do_sample", "num_beams", "max_new_tokens",
        "seed", "source_frame_indices", "candidate_frame_indices",
        "prompt_sha256", "response_schema",
    }
)
_QWEN_VISUAL_EXECUTION_FIELDS = frozenset(
    {
        "schema_version", "model_closure_sha256", "model_snapshot_digest",
        "source_media_sha256", "candidate_media_sha256",
        "instruction_sha256", "sampled_frame_indices",
        "source_sampled_frame_sha256", "candidate_sampled_frame_sha256",
        "source_mosaic_pixel_sha256", "candidate_mosaic_pixel_sha256",
        "source_mosaic_png_sha256", "candidate_mosaic_png_sha256",
        "rendered_prompt_sha256", "input_ids_sha256", "output_ids_sha256",
        "raw_response_sha256", "visual_input_digest", "execution_digest",
    }
)


class TargetFreeBerniniRuntimeError(RuntimeError):
    """Raised before an ambiguous rollout or optimizer mutation is admitted."""


def fail(message: str) -> NoReturn:
    raise TargetFreeBerniniRuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise TargetFreeBerniniRuntimeError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    actual = set(value)
    if not all(type(key) is str for key in actual) or actual != set(fields):
        fail(f"{label} field closure differs")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be one safe identifier")
    return value


def _exact_float(value: Any, expected: float, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        fail(f"{label} differs")
    return value


def validate_world8_device_placement_v1(
    *, world_rank: Any, local_rank: Any, dp_arm: Any, sp_rank: Any, device_index: Any
) -> Mapping[str, int]:
    if (
        type(world_rank) is not int
        or world_rank not in range(WORLD_SIZE)
        or type(local_rank) is not int
        or local_rank != world_rank
        or type(dp_arm) is not int
        or dp_arm != world_rank // SP_SIZE
        or type(sp_rank) is not int
        or sp_rank != world_rank % SP_SIZE
        or type(device_index) is not int
        or device_index != local_rank
    ):
        fail("WORLD8 rank/local-rank/DP2xSP4/device placement differs")
    return {
        "world_rank": world_rank,
        "local_rank": local_rank,
        "dp_arm": dp_arm,
        "sp_rank": sp_rank,
        "device_index": device_index,
    }


def _absolute_path(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        fail(f"{label} must be one path string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or any(
        part in ("", ".", "..") for part in path.parts[1:]
    ):
        fail(f"{label} must be one absolute lexical path")
    return path


def _verify_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    declared = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        fail(f"{label} contains non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TargetFreeBerniniRuntimeError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _identity(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mode),
        int(stat_result.st_nlink),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def read_stable_file(
    path: Path, *, expected_sha256: str, label: str, expected_mode: Optional[int] = None
) -> bytes:
    expected = _sha256(expected_sha256, label=f"{label} SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"{label} cannot be opened safely: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            fail(f"{label} must be one regular nlink1 file")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            fail(f"{label} mode differs")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if _identity(before) != _identity(after) or len(raw) != int(before.st_size):
            fail(f"{label} changed while held open")
        if digest.hexdigest() != expected:
            fail(f"{label} bytes differ")
        return raw
    finally:
        os.close(descriptor)


def _stable_file_binding_v1(
    path: Path, *, label: str, expected_mode: Optional[int] = None
) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(f"{label} cannot be opened safely: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            fail(f"{label} must be regular nlink1")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            fail(f"{label} mode differs")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after) or size != int(before.st_size):
            fail(f"{label} changed while held open")
        return {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
            "nlink": int(before.st_nlink),
        }
    finally:
        os.close(descriptor)


def _load_held_local_source_module_v1(name: str) -> tuple[Any, Mapping[str, Any]]:
    """Execute one frozen local helper from authenticated source bytes only."""

    if name not in _HELD_LOCAL_SOURCE_BINDINGS:
        fail("held local source name is not registered")
    filename, expected_sha, expected_size = _HELD_LOCAL_SOURCE_BINDINGS[name]
    path = Path(__file__).resolve().with_name(filename)
    if name in sys.modules:
        fail(f"held local module cache is not empty: {name}")
    raw = read_stable_file(
        path, expected_sha256=expected_sha, label=f"held local source {name}"
    )
    if len(raw) != expected_size:
        fail(f"held local source size differs: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__full644_held_source_binding__ = (
        expected_sha,
        expected_size,
        str(path),
    )
    sys.modules[name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__, module.__dict__)
    except Exception:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    if sys.modules.get(name) is not module:
        fail(f"held local module ownership changed: {name}")
    return module, {
        "module_name": name,
        "source_path": str(path),
        "source_sha256": expected_sha,
        "source_size_bytes": expected_size,
        "executed_held_source_bytes": True,
        "python_bytecode_cache_used": False,
    }


def _require_held_local_module_v1(name: str) -> Any:
    binding = _HELD_LOCAL_SOURCE_BINDINGS.get(name)
    module = sys.modules.get(name)
    if binding is None or module is None:
        fail(f"required held local module is absent: {name}")
    expected = (binding[1], binding[2], str(Path(__file__).resolve().with_name(binding[0])))
    if getattr(module, "__full644_held_source_binding__", None) != expected:
        fail(f"required held local module binding differs: {name}")
    return module


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(descriptor, payload[offset:])
        if count <= 0:
            fail("create-only artifact write did not progress")
        offset += count


def write_create_only(path: Path, payload: bytes, *, mode: int = 0o444) -> Mapping[str, Any]:
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        fail("create-only output path differs")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"cannot create fresh output {path}: {error}"
        ) from error
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or int(after.st_nlink) != 1
            or int(after.st_size) != len(payload)
            or stat.S_IMODE(after.st_mode) != mode
        ):
            fail("create-only output identity differs")
    finally:
        os.close(descriptor)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    final = os.lstat(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(final.st_mode), "04o"),
        "nlink": int(final.st_nlink),
    }


def _mkdir_private_v1(path: Path) -> None:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        fail("private output directory must be one fresh absolute path")
    os.mkdir(path, 0o700)


def _parse_checkpoint_manifest_v1(raw: bytes) -> tuple[tuple[str, str], ...]:
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as error:
        raise TargetFreeBerniniRuntimeError(
            "checkpoint content manifest is not UTF-8"
        ) from error
    expression = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)\Z")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        match = expression.fullmatch(line)
        if match is None:
            fail("checkpoint content manifest line differs")
        digest, raw_relative = match.groups()
        parts = tuple(
            part for part in Path(raw_relative).parts if part not in ("", ".")
        )
        relative = Path(*parts).as_posix()
        if (
            not relative
            or Path(raw_relative).is_absolute()
            or ".." in Path(raw_relative).parts
            or relative in seen
        ):
            fail("checkpoint content manifest path differs")
        seen.add(relative)
        rows.append((relative, digest))
    if len(rows) != BASE_CHECKPOINT_FILE_COUNT or rows != sorted(rows):
        fail("checkpoint content manifest exact23/order differs")
    return tuple(rows)


def _snapshot_checkpoint_exact23_v1(
    source_root: Path, manifest_path: Path, destination: Path
) -> Mapping[str, Any]:
    """Copy every exact23 model byte into a fresh read-only consumer root."""

    manifest_raw = read_stable_file(
        manifest_path,
        expected_sha256=BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        label="base checkpoint content manifest",
    )
    rows = _parse_checkpoint_manifest_v1(manifest_raw)
    source = source_root.resolve(strict=True)
    if source != source_root or source.is_symlink() or not source.is_dir():
        fail("base checkpoint root differs")
    actual = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(source).parts
    )
    if actual != [relative for relative, _ in rows]:
        fail("base checkpoint physical exact23 closure differs")
    _mkdir_private_v1(destination)
    projected = []
    directories = {destination}
    for relative, expected_sha in rows:
        source_path = source / relative
        raw = read_stable_file(
            source_path,
            expected_sha256=expected_sha,
            label=f"base checkpoint member {relative}",
        )
        target = destination / relative
        missing: list[Path] = []
        parent = target.parent
        while parent != destination and not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            os.mkdir(directory, 0o700)
            directories.add(directory)
        binding = write_create_only(target, raw)
        projected.append(
            {
                "path": relative,
                "sha256": expected_sha,
                "size_bytes": len(raw),
                "mode_octal": binding["mode_octal"],
            }
        )
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o555)
    result = {
        "schema_version": "bernini-full644-private-checkpoint-snapshot-v1",
        "source_tree_sha256": BASE_CHECKPOINT_TREE_SHA256,
        "content_manifest_sha256": BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "destination": str(destination),
        "file_count": BASE_CHECKPOINT_FILE_COUNT,
        "files": projected,
        "files_digest": object_sha256(projected),
        "consumer_reads_private_snapshot_only": True,
        "all_files_mode_0444": True,
        "all_directories_mode_0555": True,
    }
    return {**result, "snapshot_digest": object_sha256(result)}


def _read_fd_bytes_v1(descriptor: int, *, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    raw = b"".join(chunks)
    if _identity(before) != _identity(after) or len(raw) != int(before.st_size):
        fail(f"{label} changed while held open")
    return raw, before


def _held_git_run_v1(
    git_descriptor: int,
    root_descriptor: int,
    arguments: Sequence[str],
    *,
    label: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    if any(type(item) is not str or "\x00" in item for item in arguments):
        fail(f"{label} git arguments differ")
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("GIT_") or key in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    command = [
        f"/proc/self/fd/{git_descriptor}",
        "--no-pager",
        "-C",
        f"/proc/self/fd/{root_descriptor}",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
            pass_fds=(git_descriptor, root_descriptor),
        )
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"{label} held git execution failed: {error}"
        ) from error
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[:1000]
        fail(f"{label} held git command failed: {detail}")
    return result


def _git_archive_tracked_projection_v1(
    *,
    git_descriptor: int,
    source_root: Path,
    expected_commit: str,
    pathspecs: Sequence[str],
    prefix: str,
) -> tuple[list[tuple[str, bytes]], Mapping[str, Any]]:
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(source_root, root_flags)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"{prefix} source root cannot be held: {error}"
        ) from error
    try:
        root_before = os.fstat(root_descriptor)
        revision = _held_git_run_v1(
            git_descriptor,
            root_descriptor,
            ("rev-parse", "HEAD"),
            label=f"{prefix} revision",
        ).stdout
        if revision != (expected_commit + "\n").encode("ascii"):
            fail(f"{prefix} source revision differs")
        for stage in ("pre", "post"):
            if stage == "post":
                # The second check is deliberately after archive construction.
                pass
            diff = _held_git_run_v1(
                git_descriptor,
                root_descriptor,
                ("diff", "--quiet", "--no-ext-diff", "--"),
                label=f"{prefix} {stage} tracked diff",
                check=False,
            )
            cached = _held_git_run_v1(
                git_descriptor,
                root_descriptor,
                ("diff", "--cached", "--quiet", "--no-ext-diff", "--"),
                label=f"{prefix} {stage} cached diff",
                check=False,
            )
            status = _held_git_run_v1(
                git_descriptor,
                root_descriptor,
                ("status", "--porcelain=v1", "--untracked-files=all"),
                label=f"{prefix} {stage} full status",
            )
            if diff.returncode != 0 or cached.returncode != 0 or status.stdout != b"":
                fail(f"{prefix} source worktree is not exact clean including untracked")
            if stage == "pre":
                break
        tracked_raw = _held_git_run_v1(
            git_descriptor,
            root_descriptor,
            ("ls-files", "-z", "--", *pathspecs),
            label=f"{prefix} tracked inventory",
        ).stdout
        tracked = [item.decode("utf-8", errors="strict") for item in tracked_raw.split(b"\x00") if item]
        if (
            not tracked
            or tracked != sorted(tracked)
            or len(tracked) != len(set(tracked))
            or any(
                not item
                or Path(item).is_absolute()
                or ".." in Path(item).parts
                or "__pycache__" in Path(item).parts
                or item.endswith((".pyc", ".pyo"))
                for item in tracked
            )
        ):
            fail(f"{prefix} exact tracked inventory differs")
        archive_raw = _held_git_run_v1(
            git_descriptor,
            root_descriptor,
            ("archive", "--format=tar", "HEAD", "--", *pathspecs),
            label=f"{prefix} committed archive",
        ).stdout
        committed: dict[str, bytes] = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
                for member in archive.getmembers():
                    if member.isdir():
                        continue
                    if not member.isfile() or member.name not in tracked:
                        fail(f"{prefix} committed archive member type differs")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        fail(f"{prefix} committed archive member is unreadable")
                    payload = extracted.read()
                    if len(payload) != member.size or member.name in committed:
                        fail(f"{prefix} committed archive member size differs")
                    committed[member.name] = payload
        except (tarfile.TarError, OSError) as error:
            raise TargetFreeBerniniRuntimeError(
                f"{prefix} committed archive is invalid: {error}"
            ) from error
        if list(sorted(committed)) != tracked:
            fail(f"{prefix} archive/tracked inventory differs")
        # Require the ordinary checkout bytes to equal the exact committed
        # bytes too.  Consumers use the private committed projection below.
        for relative in tracked:
            raw = read_stable_file(
                source_root / relative,
                expected_sha256=hashlib.sha256(committed[relative]).hexdigest(),
                label=f"{prefix} tracked checkout {relative}",
            )
            if raw != committed[relative]:
                fail(f"{prefix} tracked checkout/commit bytes differ")
        for arguments, suffix in (
            (("diff", "--quiet", "--no-ext-diff", "--"), "tracked diff"),
            (("diff", "--cached", "--quiet", "--no-ext-diff", "--"), "cached diff"),
        ):
            result = _held_git_run_v1(
                git_descriptor,
                root_descriptor,
                arguments,
                label=f"{prefix} post {suffix}",
                check=False,
            )
            if result.returncode != 0:
                fail(f"{prefix} source changed across committed projection")
        if _held_git_run_v1(
            git_descriptor,
            root_descriptor,
            ("status", "--porcelain=v1", "--untracked-files=all"),
            label=f"{prefix} post full status",
        ).stdout != b"":
            fail(f"{prefix} source status changed across committed projection")
        if _identity(root_before) != _identity(os.fstat(root_descriptor)):
            fail(f"{prefix} source root identity changed")
        rows = [(f"{prefix}/{relative}", committed[relative]) for relative in tracked]
        evidence = {
            "commit": expected_commit,
            "tracked_file_count": len(tracked),
            "tracked_path_digest": object_sha256(tracked),
            "git_diff_quiet_pre_post": True,
            "git_cached_diff_quiet_pre_post": True,
            "git_status_porcelain_untracked_all_exact_empty_pre_post": True,
            "projection_from_git_archive_head_bytes": True,
            "checkout_bytes_equal_committed_archive": True,
            "source_root_retained_across_projection": True,
        }
        return rows, evidence
    finally:
        os.close(root_descriptor)


def _frozen_no_git_source_projection_v1(
    *,
    source_root: Path,
    expected_commit: str,
    package_name: str,
    extra_relative_files: Sequence[str],
    prefix: str,
    critical_sha256: Mapping[str, str],
) -> tuple[list[tuple[str, bytes]], Mapping[str, Any]]:
    """Read a controlled 0555/0444 vendor snapshot with literal critical pins."""

    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
        or (source_root / ".git").exists()
        or stat.S_IMODE(os.lstat(source_root).st_mode) != 0o555
        or not source_root.name.endswith(expected_commit[:8])
    ):
        fail(f"{prefix} frozen no-git root authority differs")
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(source_root, root_flags)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"{prefix} frozen root cannot be held: {error}"
        ) from error
    try:
        root_before = os.fstat(root_descriptor)
        package_root = source_root / package_name
        if not package_root.is_dir() or package_root.is_symlink():
            fail(f"{prefix} frozen package root differs")
        candidates = [path for path in package_root.rglob("*") if path.is_file()]
        candidates.extend(source_root / relative for relative in extra_relative_files)
        relative_names = sorted(
            {path.relative_to(source_root).as_posix() for path in candidates}
        )
        if (
            not relative_names
            or any(
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or "__pycache__" in Path(relative).parts
                or relative.endswith((".pyc", ".pyo"))
                for relative in relative_names
            )
        ):
            fail(f"{prefix} frozen file inventory differs")
        directories = {source_root, package_root}
        for relative in relative_names:
            path = source_root / relative
            directories.update(parent for parent in path.parents if source_root in parent.parents)
        if any(
            directory.is_symlink()
            or not directory.is_dir()
            or stat.S_IMODE(os.lstat(directory).st_mode) != 0o555
            for directory in directories
        ):
            fail(f"{prefix} frozen directory mode/identity differs")
        rows: list[tuple[str, bytes]] = []
        observed_sha: dict[str, str] = {}
        for relative in relative_names:
            path = source_root / relative
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                raise TargetFreeBerniniRuntimeError(
                    f"{prefix} frozen member cannot be held: {relative}: {error}"
                ) from error
            try:
                raw, observed = _read_fd_bytes_v1(
                    descriptor, label=f"{prefix} frozen member {relative}"
                )
            finally:
                os.close(descriptor)
            if (
                int(observed.st_nlink) != 1
                or stat.S_IMODE(observed.st_mode) != 0o444
            ):
                fail(f"{prefix} frozen member mode/identity differs: {relative}")
            digest = hashlib.sha256(raw).hexdigest()
            observed_sha[relative] = digest
            rows.append((f"{prefix}/{relative}", raw))
        if any(observed_sha.get(relative) != expected for relative, expected in critical_sha256.items()):
            fail(f"{prefix} frozen critical source literal differs")
        if _identity(root_before) != _identity(os.fstat(root_descriptor)):
            fail(f"{prefix} frozen source root changed across projection")
        evidence = {
            "commit_claim": expected_commit,
            "commit_claim_bound_by_root_suffix": expected_commit[:8],
            "frozen_root_mode_octal": "0555",
            "all_projected_files_mode_0444_nlink1": True,
            "projected_file_count": len(rows),
            "projected_path_digest": object_sha256(relative_names),
            "critical_literal_sha256": dict(critical_sha256),
            "critical_literal_count": len(critical_sha256),
            "source_root_retained_across_projection": True,
            "controlled_environment_no_concurrent_mutator_assumed": True,
        }
        return rows, evidence
    finally:
        os.close(root_descriptor)


def _snapshot_python_source_tree_v1(
    *, bernini_root: Path, veomni_root: Path, destination: Path
) -> Mapping[str, Any]:
    """Project exact committed source bytes; normal checkout imports are forbidden."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        git_descriptor = os.open(GIT_EXECUTABLE, flags)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"held git executable cannot be opened: {error}"
        ) from error
    try:
        git_raw, git_before = _read_fd_bytes_v1(
            git_descriptor, label="held git executable"
        )
        if (
            len(git_raw) != GIT_EXECUTABLE_SIZE
            or hashlib.sha256(git_raw).hexdigest() != GIT_EXECUTABLE_SHA256
            or stat.S_IMODE(git_before.st_mode) != GIT_EXECUTABLE_MODE
        ):
            fail("held git executable authority differs")
        if (bernini_root / ".git").is_dir():
            bernini_projection_kind = "held_git_archive_head"
            bernini_rows, bernini_evidence = _git_archive_tracked_projection_v1(
                git_descriptor=git_descriptor,
                source_root=bernini_root,
                expected_commit=BERNINI_COMMIT,
                pathspecs=("bernini", "configs/bernini_renderer_wan21_1p3b/config.json"),
                prefix="bernini_root",
            )
        else:
            bernini_projection_kind = "held_frozen_no_git_snapshot"
            bernini_rows, bernini_evidence = _frozen_no_git_source_projection_v1(
                source_root=bernini_root,
                expected_commit=BERNINI_COMMIT,
                package_name="bernini",
                extra_relative_files=(
                    "configs/bernini_renderer_wan21_1p3b/config.json",
                ),
                prefix="bernini_root",
                critical_sha256=_PINNED_BERNINI_SOURCE_SHA256,
            )
        veomni_rows, veomni_evidence = _git_archive_tracked_projection_v1(
            git_descriptor=git_descriptor,
            source_root=veomni_root,
            expected_commit=VEOMNI_COMMIT,
            pathspecs=("veomni",),
            prefix="veomni_root",
        )
        git_after_raw, git_after = _read_fd_bytes_v1(
            git_descriptor, label="held git executable post-projection"
        )
        if (
            _identity(git_before) != _identity(git_after)
            or git_after_raw != git_raw
        ):
            fail("held git executable changed across source projection")
    finally:
        os.close(git_descriptor)
    selected = bernini_rows + veomni_rows
    if not selected or len(selected) != len({name for name, _ in selected}):
        fail("private vendor source projection inventory differs")
    committed_by_path = {name: raw for name, raw in selected}
    for relative, expected_sha in _PINNED_BERNINI_SOURCE_SHA256.items():
        key = f"bernini_root/{relative}"
        if (
            key not in committed_by_path
            or hashlib.sha256(committed_by_path[key]).hexdigest() != expected_sha
        ):
            fail(f"pinned Bernini committed source differs: {relative}")
    _mkdir_private_v1(destination)
    directories = {destination}
    rows = []
    for relative, raw in selected:
        target = destination / relative
        missing: list[Path] = []
        parent = target.parent
        while parent != destination and not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            os.mkdir(directory, 0o700)
            directories.add(directory)
        binding = write_create_only(target, raw)
        rows.append(
            {
                "path": relative,
                "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
            }
        )
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o555)
    result = {
        "schema_version": "bernini-full644-private-no-pyc-source-tree-v2",
        "bernini_commit": BERNINI_COMMIT,
        "veomni_commit": VEOMNI_COMMIT,
        "held_git_authority": {
            "path": str(GIT_EXECUTABLE),
            "sha256": GIT_EXECUTABLE_SHA256,
            "size_bytes": GIT_EXECUTABLE_SIZE,
            "mode_octal": "0755",
            "retained_across_projection": True,
        },
        "bernini_projection_kind": bernini_projection_kind,
        "bernini_source_evidence": bernini_evidence,
        "veomni_git_evidence": veomni_evidence,
        "destination": str(destination),
        "bernini_import_root": str(destination / "bernini_root"),
        "veomni_import_root": str(destination / "veomni_root"),
        "file_count": len(rows),
        "files": rows,
        "files_digest": object_sha256(rows),
        "bernini_projected_from_git_archive": (
            bernini_projection_kind == "held_git_archive_head"
        ),
        "bernini_projected_from_frozen_no_git_snapshot": (
            bernini_projection_kind == "held_frozen_no_git_snapshot"
        ),
        "veomni_projected_from_exact_git_archive_head": True,
        "python_bytecode_cache_present": False,
        "consumer_import_root_mode_0555": True,
    }
    return {**result, "snapshot_digest": object_sha256(result)}


def _canonical_fp32_tensor_bytes_v1(
    value: Any, *, label: str
) -> tuple[list[int], bytes]:
    """Return logical C-order FP32 bytes without iterating a Torch storage.

    ``bytes(UntypedStorage)`` is byte-for-byte correct but is implemented as a
    Python-level storage iteration on the pinned AUH Torch build.  A production
    latent takes seconds to project that way.  The explicit flat clone below
    gives the NumPy view canonical offset-zero C storage, and ``tobytes`` then
    copies that storage in one native operation.  Non-contiguous inputs retain
    the legacy ``contiguous()`` projection, while offset or oversized backing
    storage that survived that projection remains fail-closed.
    """

    import torch

    if type(value) is not torch.Tensor or value.layout != torch.strided:
        fail(f"{label} must be one strided torch.Tensor")
    if sys.byteorder != "little":
        fail(f"{label} requires a little-endian host")
    shape = [int(item) for item in value.shape]
    logical = (
        value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    if not bool(torch.isfinite(logical).all().item()):
        fail(f"{label} is non-finite")
    expected = int(logical.numel()) * 4
    if (
        logical.dtype != torch.float32
        or logical.device.type != "cpu"
        or not logical.is_contiguous()
        or int(logical.storage_offset()) != 0
        or int(logical.untyped_storage().nbytes()) != expected
    ):
        fail(f"{label} source FP32 storage differs")
    # ``contiguous()`` is allowed to be a no-op for singleton-stride views.
    # Flatten+clone+view forces one canonical offset-zero C-order allocation.
    cpu = logical.reshape(-1).clone().reshape(tuple(shape))
    if (
        cpu.dtype != torch.float32
        or cpu.device.type != "cpu"
        or not cpu.is_contiguous()
        or int(cpu.storage_offset()) != 0
    ):
        fail(f"{label} canonical FP32 tensor layout differs")
    array = cpu.numpy()
    if (
        array.dtype.kind != "f"
        or int(array.dtype.itemsize) != 4
        or array.dtype.byteorder not in ("=", "<")
        or not bool(array.flags.c_contiguous)
    ):
        fail(f"{label} canonical NumPy FP32 layout differs")
    raw = array.tobytes(order="C")
    if len(raw) != expected:
        fail(f"{label} canonical FP32 byte length differs")
    return shape, raw


def _tensor_sha256_from_canonical_fp32_bytes_v1(
    shape: Sequence[int], raw: bytes
) -> str:
    dimensions = [int(item) for item in shape]
    if (
        not isinstance(raw, bytes)
        or any(type(item) is not int or item < 0 for item in shape)
        or len(raw) != math.prod(dimensions) * 4
    ):
        fail("canonical FP32 tensor digest input differs")
    metadata = canonical_json_bytes(
        {"dtype": "torch.float32", "shape": dimensions}
    )
    digest = hashlib.sha256(b"full644-exact40-tensor-v1\x00")
    digest.update(struct.pack(">Q", len(metadata)))
    digest.update(metadata)
    digest.update(raw)
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    shape, raw = _canonical_fp32_tensor_bytes_v1(value, label="tensor hash input")
    return _tensor_sha256_from_canonical_fp32_bytes_v1(shape, raw)


def _source_aspect_bucket_v1(height: int, width: int) -> tuple[int, int]:
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        fail("source video dimensions differ")
    max_pixels = 245_760
    stride = 16
    scale = math.sqrt(max_pixels / float(height * width))
    bucket_h = max(stride, math.floor(height * scale / stride) * stride)
    bucket_w = max(stride, math.floor(width * scale / stride) * stride)
    while bucket_h * bucket_w > max_pixels:
        if bucket_h >= bucket_w and bucket_h > stride:
            bucket_h -= stride
        elif bucket_w > stride:
            bucket_w -= stride
        else:
            fail("source aspect bucket cannot satisfy max pixels")
    return bucket_h, bucket_w


def _decode_owned_source_row_v1(source: Any) -> tuple[Any, Mapping[str, Any]]:
    """Decode only retained catalogue source bytes through a private snapshot."""

    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    raw = read_stable_file(
        source.source_video_path,
        expected_sha256=source.source_video_sha256,
        label=f"owned full644 source {source.row_id}",
    )
    with tempfile.TemporaryDirectory(prefix="full644-owned-source-") as temporary:
        private_path = Path(temporary) / "source.mp4"
        binding = write_create_only(private_path, raw, mode=0o400)
        try:
            import decord
        except ImportError as error:
            raise TargetFreeBerniniRuntimeError(
                "owned source decode requires decord"
            ) from error
        if getattr(decord, "__version__", None) != DECORD_VERSION:
            fail("owned source decoder version differs")
        try:
            reader = decord.VideoReader(
                str(private_path), num_threads=1, ctx=decord.cpu(0)
            )
            count = len(reader)
            reported_fps = float(reader.get_avg_fps())
            frames = reader.get_batch(list(range(FRAME_COUNT))).asnumpy()
        except Exception as error:
            raise TargetFreeBerniniRuntimeError(
                f"owned source private decode failed: {error}"
            ) from error
    if (
        binding["sha256"] != source.source_video_sha256
        or count != FRAME_COUNT
        or not math.isfinite(reported_fps)
        or abs(reported_fps - FPS) > 1.0e-3
        or not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or tuple(frames.shape[:1]) != (FRAME_COUNT,)
        or int(frames.shape[-1]) != 3
    ):
        fail("owned source exact81/25fps RGB decode differs")
    source_hw = (int(frames.shape[1]), int(frames.shape[2]))
    bucket_hw = _source_aspect_bucket_v1(*source_hw)
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = tvf.resize(
        tensor,
        list(bucket_hw),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    tensor = (
        tensor.mul(2.0)
        .sub(1.0)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .contiguous()
    )
    if (
        tuple(tensor.shape) != (1, 3, FRAME_COUNT, *bucket_hw)
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or not bool(torch.isfinite(tensor).all().item())
    ):
        fail("owned source normalized tensor ABI differs")
    receipt = {
        "schema_version": "bernini-full644-owned-source-decode-v1",
        "source_row_id": source.row_id,
        "source_video_sha256": source.source_video_sha256,
        "source_bytes_size": len(raw),
        "decoded_from_private_held_bytes_snapshot": True,
        "original_path_not_reopened_by_decoder": True,
        "frame_count": FRAME_COUNT,
        "fps_float64_hex": FPS.hex(),
        "source_input_hw": list(source_hw),
        "source_derived_bucket_hw": list(bucket_hw),
        "bucket_rule": "sqrt_245760_then_floor_each_dimension_to_stride16",
        "resize": "torchvision_bicubic_antialias_true",
        "normalization": "uint8_div_255_mul_2_minus_1",
        "source_tensor_sha256": tensor_sha256(tensor),
        "target_media_read_count": 0,
    }
    return tensor, {**receipt, "decode_digest": object_sha256(receipt)}


def _validate_frozen_catalog_release_envelopes_v1(
    raw_receipt: bytes, raw_postflight: bytes
) -> Mapping[str, Any]:
    """Validate pinned catalog authorities at their actual schema locations."""

    receipt = _strict_json(raw_receipt, label="frozen catalog receipt")
    postflight = _strict_json(raw_postflight, label="frozen catalog postflight")
    ffprobe = receipt.get("ffprobe_binding")
    controller = postflight.get("controller")
    pre_admission = postflight.get("controller_pre_admission")
    producer = postflight.get("producer")
    authority = postflight.get("authority")
    trace = postflight.get("trace")
    if not all(
        isinstance(value, Mapping)
        for value in (
            ffprobe, controller, pre_admission, producer, authority, trace
        )
    ):
        fail("frozen catalog release nested schema differs")
    producer_catalog = producer.get("catalog")
    producer_receipt = producer.get("receipt")
    if not isinstance(producer_catalog, Mapping) or not isinstance(
        producer_receipt, Mapping
    ):
        fail("frozen catalog postflight producer schema differs")
    if (
        receipt.get("schema_version")
        != "bernini-full644-target-free-source-catalog-receipt-v1"
        or receipt.get("status") != "SOURCE_ONLY_EXACT644_CATALOG_COMPLETE"
        or receipt.get("receipt_digest") != FULL644_CATALOG_RECEIPT_DIGEST
        or receipt.get("catalog_sha256") != FULL644_CATALOG_SHA256
        or receipt.get("catalog_size") != FULL644_CATALOG_SIZE
        or receipt.get("catalog_digest") != FULL644_CATALOG_DIGEST
        or receipt.get("extractor_self_sha256")
        != FULL644_CATALOG_EXTRACTOR_SHA256
        or receipt.get("extractor_self_size") != FULL644_CATALOG_EXTRACTOR_SIZE
        or receipt.get("source_count") != 644
        or receipt.get("target_media_used") is not False
        or receipt.get("paired_edited_target_present") is not False
        or ffprobe.get("sha256") != FULL644_CATALOG_FFPROBE_SHA256
        or ffprobe.get("size") != FULL644_CATALOG_FFPROBE_SIZE
        or ffprobe.get("mode") != 0o555
        or ffprobe.get("nlink") != 1
        or ffprobe.get("held_fd_execution") is not True
        or receipt.get("ffprobe_held_fd_prepost_replay_verified") is not True
    ):
        fail("frozen catalog receipt authority schema differs")
    if (
        postflight.get("schema_version")
        != "bernini-full644-source-catalog-external-postflight-v1"
        or postflight.get("status")
        != "SOURCE_ONLY_EXACT644_CATALOG_POSTFLIGHT_COMPLETE"
        or postflight.get("complete") is not True
        or postflight.get("release_digest")
        != FULL644_CATALOG_POSTFLIGHT_DIGEST
        or postflight.get("downstream_source_rehash_required") is not True
        or postflight.get("external_release_sha_must_be_pinned_before_consumption")
        is not True
        or controller.get("sha256") != FULL644_CATALOG_CONTROLLER_SHA256
        or controller.get("size") != FULL644_CATALOG_CONTROLLER_SIZE
        or pre_admission.get("sha256")
        != FULL644_CATALOG_PRE_ADMISSION_SHA256
        or pre_admission.get("size") != FULL644_CATALOG_PRE_ADMISSION_SIZE
        or producer.get("source_count") != 644
        or producer.get("extractor_sha256")
        != FULL644_CATALOG_EXTRACTOR_SHA256
        or producer.get("exact_member_closure")
        != ["source_catalog.json", "source_catalog_receipt.json"]
        or producer_catalog.get("sha256") != FULL644_CATALOG_SHA256
        or producer_catalog.get("size") != FULL644_CATALOG_SIZE
        or producer_catalog.get("manifest_digest") != FULL644_CATALOG_DIGEST
        or producer_receipt.get("sha256") != FULL644_CATALOG_RECEIPT_SHA256
        or producer_receipt.get("size") != FULL644_CATALOG_RECEIPT_SIZE
        or producer_receipt.get("receipt_digest")
        != FULL644_CATALOG_RECEIPT_DIGEST
        or authority.get("catalog_integrity_release") is not True
        or authority.get("paired_edited_target_present") is not False
        or authority.get("trainer_launched") is not False
        or authority.get("training_runtime_authorized") is not False
        or authority.get("upstream_training_use_forbidden") is not True
        or trace.get("held_ffprobe_exec_success_count") != 644
        or trace.get("held_ffprobe_exit0_count") != 644
        or trace.get("source_inventory_count") != 644
        or trace.get("source_path_seen_count") != 644
        or trace.get("target_exact_path_seen_count") != 0
    ):
        fail("frozen catalog postflight authority schema differs")
    return {
        "schema_version": "bernini-full644-catalog-release-envelope-projection-v1",
        "receipt_digest": FULL644_CATALOG_RECEIPT_DIGEST,
        "postflight_digest": FULL644_CATALOG_POSTFLIGHT_DIGEST,
        "ffprobe_sha256_from_receipt_binding": FULL644_CATALOG_FFPROBE_SHA256,
        "postflight_controller_sha256": FULL644_CATALOG_CONTROLLER_SHA256,
        "postflight_pre_admission_sha256": FULL644_CATALOG_PRE_ADMISSION_SHA256,
        "postflight_catalog_and_receipt_join_verified": True,
    }


def _load_frozen_one_source_catalog_v1() -> tuple[Any, Any, Mapping[str, Any]]:
    core = _require_held_local_module_v1("full644_target_free_preference_v1")
    raw_catalog = read_stable_file(
        FULL644_CATALOG_PATH,
        expected_sha256=FULL644_CATALOG_SHA256,
        label="frozen full644 source catalog",
    )
    raw_receipt = read_stable_file(
        FULL644_CATALOG_RECEIPT_PATH,
        expected_sha256=FULL644_CATALOG_RECEIPT_SHA256,
        label="frozen full644 catalog receipt",
    )
    raw_postflight = read_stable_file(
        FULL644_CATALOG_POSTFLIGHT_PATH,
        expected_sha256=FULL644_CATALOG_POSTFLIGHT_SHA256,
        label="frozen full644 catalog postflight",
    )
    if (
        len(raw_catalog) != FULL644_CATALOG_SIZE
        or len(raw_receipt) != FULL644_CATALOG_RECEIPT_SIZE
        or len(raw_postflight) != FULL644_CATALOG_POSTFLIGHT_SIZE
    ):
        fail("frozen full644 catalog release size differs")
    catalog = core.load_source_catalog(
        FULL644_CATALOG_PATH,
        expected_sha256=FULL644_CATALOG_SHA256,
        require_source_files=False,
    )
    source = catalog.row(ONE_SOURCE_ROW_ID)
    if (
        catalog.manifest_sha256 != FULL644_CATALOG_SHA256
        or catalog.manifest_digest != FULL644_CATALOG_DIGEST
        or source.source_video_sha256 != ONE_SOURCE_VIDEO_SHA256
        or source.instruction_sha256 != ONE_SOURCE_INSTRUCTION_SHA256
        or source.row_digest != ONE_SOURCE_ROW_DIGEST
    ):
        fail("frozen one-source row/catalog semantic join differs")
    release_envelopes = _validate_frozen_catalog_release_envelopes_v1(
        raw_receipt, raw_postflight
    )
    binding = {
        "schema_version": "bernini-full644-frozen-catalog-release-binding-v1",
        "catalog_path": str(FULL644_CATALOG_PATH),
        "catalog_sha256": FULL644_CATALOG_SHA256,
        "catalog_size_bytes": FULL644_CATALOG_SIZE,
        "catalog_digest": FULL644_CATALOG_DIGEST,
        "receipt_path": str(FULL644_CATALOG_RECEIPT_PATH),
        "receipt_sha256": FULL644_CATALOG_RECEIPT_SHA256,
        "receipt_size_bytes": FULL644_CATALOG_RECEIPT_SIZE,
        "receipt_digest": FULL644_CATALOG_RECEIPT_DIGEST,
        "postflight_path": str(FULL644_CATALOG_POSTFLIGHT_PATH),
        "postflight_sha256": FULL644_CATALOG_POSTFLIGHT_SHA256,
        "postflight_size_bytes": FULL644_CATALOG_POSTFLIGHT_SIZE,
        "postflight_digest": FULL644_CATALOG_POSTFLIGHT_DIGEST,
        "extractor_sha256": FULL644_CATALOG_EXTRACTOR_SHA256,
        "extractor_size_bytes": FULL644_CATALOG_EXTRACTOR_SIZE,
        "controller_sha256": FULL644_CATALOG_CONTROLLER_SHA256,
        "controller_size_bytes": FULL644_CATALOG_CONTROLLER_SIZE,
        "pre_admission_sha256": FULL644_CATALOG_PRE_ADMISSION_SHA256,
        "pre_admission_size_bytes": FULL644_CATALOG_PRE_ADMISSION_SIZE,
        "ffprobe_sha256": FULL644_CATALOG_FFPROBE_SHA256,
        "ffprobe_size_bytes": FULL644_CATALOG_FFPROBE_SIZE,
        "release_envelope_projection": release_envelopes,
        "selected_row": source.receipt(),
    }
    return catalog, source, {**binding, "binding_digest": object_sha256(binding)}


def _tensor_fp32_bytes(value: Any) -> tuple[list[int], bytes, str]:
    shape, raw = _canonical_fp32_tensor_bytes_v1(
        value, label="trajectory artifact tensor"
    )
    if len(shape) < 1:
        fail("trajectory artifact tensor differs")
    return shape, raw, _tensor_sha256_from_canonical_fp32_bytes_v1(shape, raw)


def build_trajectory_artifact_v1(
    *, initial_state: Any, actions: Sequence[Any]
) -> tuple[bytes, Mapping[str, Any]]:
    if not isinstance(actions, (list, tuple)) or len(actions) != TRAJECTORY_STEPS:
        fail("trajectory artifact requires exact40 executed actions")
    rows: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in (
        ("initial_state", initial_state),
        *((f"action_{index:02d}", action) for index, action in enumerate(actions)),
    ):
        shape, raw, digest = _tensor_fp32_bytes(tensor)
        row = {
            "name": name,
            "shape": shape,
            "numel": math.prod(shape),
            "offset": offset,
            "size_bytes": len(raw),
            "sha256": digest,
        }
        rows.append(row)
        chunks.append(raw)
        offset += len(raw)
    header = {
        "schema_version": TRAJECTORY_ARTIFACT_SCHEMA,
        "dtype": "torch.float32",
        "byte_order": "little",
        "tensor_count": TRAJECTORY_STEPS + 1,
        "tensors": rows,
        "payload_size_bytes": offset,
    }
    sealed_header = {**header, "header_digest": object_sha256(header)}
    header_bytes = canonical_json_bytes(sealed_header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        fail("trajectory artifact header is oversized")
    payload = (
        _ARTIFACT_MAGIC
        + struct.pack(">Q", len(header_bytes))
        + header_bytes
        + b"".join(chunks)
    )
    return payload, sealed_header


def _validate_artifact_header(value: Any) -> Mapping[str, Any]:
    header = _closed(value, _ARTIFACT_HEADER_FIELDS, label="trajectory artifact header")
    if (
        header["schema_version"] != TRAJECTORY_ARTIFACT_SCHEMA
        or header["dtype"] != "torch.float32"
        or header["byte_order"] != "little"
        or type(header["tensor_count"]) is not int
        or header["tensor_count"] != TRAJECTORY_STEPS + 1
        or type(header["payload_size_bytes"]) is not int
        or header["payload_size_bytes"] <= 0
    ):
        fail("trajectory artifact header contract differs")
    _verify_digest(header, "header_digest", label="trajectory artifact header")
    tensors = header["tensors"]
    if not isinstance(tensors, list) or len(tensors) != TRAJECTORY_STEPS + 1:
        fail("trajectory artifact tensor inventory differs")
    expected_names = ["initial_state"] + [
        f"action_{index:02d}" for index in range(TRAJECTORY_STEPS)
    ]
    offset = 0
    expected_shape: Optional[list[int]] = None
    for index, raw_row in enumerate(tensors):
        row = _closed(
            raw_row, _ARTIFACT_TENSOR_FIELDS, label=f"artifact tensor {index}"
        )
        shape = row["shape"]
        if (
            row["name"] != expected_names[index]
            or not isinstance(shape, list)
            or not shape
            or any(type(item) is not int or item <= 0 for item in shape)
            or type(row["numel"]) is not int
            or row["numel"] != math.prod(shape)
            or type(row["offset"]) is not int
            or row["offset"] != offset
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] != row["numel"] * 4
        ):
            fail("trajectory artifact tensor row differs")
        _sha256(row["sha256"], label="trajectory artifact tensor SHA")
        if expected_shape is None:
            expected_shape = list(shape)
        elif list(shape) != expected_shape:
            fail("trajectory artifact action geometry differs")
        offset += row["size_bytes"]
    if offset != header["payload_size_bytes"]:
        fail("trajectory artifact payload length differs")
    return header


class TrajectoryArtifactReaderV1:
    """Retain one O_NOFOLLOW descriptor across both streaming replay passes."""

    def __init__(self, path: Path, *, expected_sha256: str) -> None:
        self.path = path
        self.expected_sha256 = _sha256(
            expected_sha256, label="trajectory artifact SHA"
        )
        self.descriptor = -1
        self.header: Mapping[str, Any] = {}
        self.payload_offset = 0
        self._identity: tuple[int, ...] = ()

    def __enter__(self) -> "TrajectoryArtifactReaderV1":
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            self.descriptor = os.open(self.path, flags)
        except OSError as error:
            raise TargetFreeBerniniRuntimeError(
                f"trajectory artifact cannot be opened safely: {error}"
            ) from error
        before = os.fstat(self.descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or stat.S_IMODE(before.st_mode) != 0o444
        ):
            self.close()
            fail("trajectory artifact identity/mode differs")
        self._identity = _identity(before)
        digest = hashlib.sha256()
        cursor = 0
        while cursor < int(before.st_size):
            block = os.pread(
                self.descriptor, min(1024 * 1024, int(before.st_size) - cursor), cursor
            )
            if not block:
                self.close()
                fail("trajectory artifact short read")
            digest.update(block)
            cursor += len(block)
        if digest.hexdigest() != self.expected_sha256:
            self.close()
            fail("trajectory artifact bytes differ")
        prefix_size = len(_ARTIFACT_MAGIC) + _HEADER_SIZE_BYTES
        prefix = os.pread(self.descriptor, prefix_size, 0)
        if len(prefix) != prefix_size or prefix[: len(_ARTIFACT_MAGIC)] != _ARTIFACT_MAGIC:
            self.close()
            fail("trajectory artifact magic differs")
        header_size = struct.unpack(">Q", prefix[-_HEADER_SIZE_BYTES:])[0]
        if not 1 <= header_size <= _MAX_HEADER_BYTES:
            self.close()
            fail("trajectory artifact header size differs")
        header_raw = os.pread(self.descriptor, header_size, prefix_size)
        if len(header_raw) != header_size:
            self.close()
            fail("trajectory artifact header short read")
        self.header = _validate_artifact_header(
            _strict_json(header_raw, label="trajectory artifact header")
        )
        self.payload_offset = prefix_size + header_size
        if (
            self.payload_offset + int(self.header["payload_size_bytes"])
            != int(before.st_size)
        ):
            self.close()
            fail("trajectory artifact total size differs")
        return self

    def tensor(self, name: str, *, device: Any = "cpu") -> Any:
        import torch

        if self.descriptor < 0 or type(name) is not str:
            fail("trajectory artifact reader is not active")
        rows = self.header["tensors"]
        matches = [row for row in rows if row["name"] == name]
        if len(matches) != 1:
            fail("trajectory artifact tensor name differs")
        row = matches[0]
        raw = os.pread(
            self.descriptor,
            int(row["size_bytes"]),
            self.payload_offset + int(row["offset"]),
        )
        if len(raw) != int(row["size_bytes"]):
            fail("trajectory artifact tensor short read")
        value = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
            tuple(row["shape"])
        )
        if tensor_sha256(value) != row["sha256"]:
            fail("trajectory artifact tensor digest differs")
        return value.to(device=device).contiguous().detach()

    def assert_stable(self) -> None:
        if self.descriptor < 0 or _identity(os.fstat(self.descriptor)) != self._identity:
            fail("trajectory artifact changed while retained")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.descriptor >= 0:
                self.assert_stable()
        finally:
            self.close()


@dataclass(frozen=True)
class TrajectoryReceiptV1:
    value: Mapping[str, Any]
    path: Path
    sha256: str

    @property
    def artifact_path(self) -> Path:
        return Path(self.value["artifact_path"])

    @property
    def artifact_sha256(self) -> str:
        return self.value["artifact_sha256"]


def validate_trajectory_receipt_value_v1(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _TRAJECTORY_FIELDS, label="trajectory receipt")
    if (
        row["schema_version"] != TRAJECTORY_SCHEMA
        or row["runtime_schema_version"] != SCHEMA_VERSION
        or type(row["round_index"]) is not int
        or row["round_index"] < 0
        or type(row["rollout_seed"]) is not int
        or not 0 <= row["rollout_seed"] < 2**63
        or type(row["dp_arm"]) is not int
        or row["dp_arm"] not in range(DP_SIZE)
        or row["sp_size"] != SP_SIZE
        or row["step_count"] != TRAJECTORY_STEPS
        or row["latent_dtype"] != "torch.float32"
        or row["schedule_sha256"] != SCHEDULE_SHA256
        or row["gaussian_kernel"] != GAUSSIAN_KERNEL
        or row["gaussian_kernel_sha256"] != GAUSSIAN_KERNEL_SHA256
        or row["apg_guidance_sha256"] != APG_GUIDANCE_SHA256
        or row["sp4_noise_broadcast"] is not True
        or row["sp4_step_consensus"] is not True
        or row["source_only_input"] is not True
        or row["paired_reference_read_count"] != 0
        or row["external_velocity_read_count"] != 0
    ):
        fail("trajectory receipt fixed closure differs")
    for key in ("rollout_id", "source_row_id"):
        _safe_id(row[key], label=f"trajectory {key}")
    for key in (
        "source_video_sha256",
        "instruction_sha256",
        "behavior_policy_sha256",
        "initial_noise_key_sha256",
        "initial_state_sha256",
        "terminal_state_sha256",
        "artifact_sha256",
    ):
        _sha256(row[key], label=f"trajectory {key}")
    shape = row["latent_shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or type(row["latent_numel"]) is not int
        or row["latent_numel"] != math.prod(shape)
    ):
        fail("trajectory latent geometry differs")
    artifact_path = _absolute_path(row["artifact_path"], label="trajectory artifact")
    if (
        type(row["artifact_size_bytes"]) is not int
        or row["artifact_size_bytes"] <= 0
        or row["artifact_mode_octal"] != "0444"
        or row["artifact_nlink"] != 1
    ):
        fail("trajectory artifact binding differs")
    steps = row["steps"]
    if not isinstance(steps, list) or len(steps) != TRAJECTORY_STEPS:
        fail("trajectory receipt requires exact40 step rows")
    previous = row["initial_state_sha256"]
    for index, raw_step in enumerate(steps):
        step = _closed(raw_step, _STEP_FIELDS, label=f"trajectory step {index}")
        if (
            step["schema_version"] != TRAJECTORY_STEP_SCHEMA
            or step["schedule_index"] != index
            or type(step["timestep"]) is not int
            or type(step["sigma_float32_be_hex"]) is not str
            or re.fullmatch(r"[0-9a-f]{8}", step["sigma_float32_be_hex"]) is None
            or step["state_before_sha256"] != previous
            or step["scheduler_step_index_after"] != index + 1
        ):
            fail("trajectory exact40 state-chain closure differs")
        for key in (
            "state_before_sha256",
            "policy_mean_sha256",
            "action_noise_key_sha256",
            "action_noise_sha256",
            "executed_action_sha256",
            "state_after_sha256",
        ):
            _sha256(step[key], label=f"trajectory step {index} {key}")
        _verify_digest(step, "step_digest", label=f"trajectory step {index}")
        previous = step["state_after_sha256"]
    if previous != row["terminal_state_sha256"]:
        fail("trajectory terminal state chain differs")
    _verify_digest(row, "receipt_digest", label="trajectory receipt")
    # Keep the lexical path validation in the returned object under ``-O`` too.
    if artifact_path != Path(row["artifact_path"]):
        fail("trajectory artifact lexical path projection differs")
    return row


def load_trajectory_receipt_v1(
    path: Path, *, expected_sha256: str
) -> TrajectoryReceiptV1:
    raw = read_stable_file(
        path,
        expected_sha256=expected_sha256,
        expected_mode=0o444,
        label="trajectory receipt",
    )
    value = validate_trajectory_receipt_value_v1(
        _strict_json(raw, label="trajectory receipt")
    )
    artifact = Path(value["artifact_path"])
    observed = os.lstat(artifact)
    if (
        int(observed.st_size) != value["artifact_size_bytes"]
        or stat.S_IMODE(observed.st_mode) != 0o444
        or int(observed.st_nlink) != 1
    ):
        fail("trajectory artifact physical binding differs")
    with TrajectoryArtifactReaderV1(
        artifact, expected_sha256=value["artifact_sha256"]
    ) as reader:
        header = reader.header
        if header["tensors"][0]["shape"] != value["latent_shape"]:
            fail("trajectory receipt/artifact geometry differs")
        if header["tensors"][0]["sha256"] != value["initial_state_sha256"]:
            fail("trajectory receipt/artifact initial state differs")
        for index in range(TRAJECTORY_STEPS):
            if (
                header["tensors"][index + 1]["sha256"]
                != value["steps"][index]["executed_action_sha256"]
            ):
                fail("trajectory receipt/artifact action inventory differs")
    return TrajectoryReceiptV1(
        value=value,
        path=path,
        sha256=_sha256(expected_sha256, label="trajectory receipt SHA"),
    )


def load_decoded_rollout_receipt_v1(
    path: Path, *, expected_sha256: str
) -> tuple[Mapping[str, Any], TrajectoryReceiptV1]:
    raw = read_stable_file(
        path,
        expected_sha256=expected_sha256,
        expected_mode=0o444,
        label="decoded rollout receipt",
    )
    row = _closed(
        _strict_json(raw, label="decoded rollout receipt"),
        _DECODED_ROLLOUT_FIELDS,
        label="decoded rollout receipt",
    )
    if (
        row["schema_version"] != DECODED_ROLLOUT_SCHEMA
        or type(row["round_index"]) is not int
        or row["round_index"] < 0
        or type(row["rollout_seed"]) is not int
        or not 0 <= row["rollout_seed"] < 2**63
        or type(row["dp_arm"]) is not int
        or row["dp_arm"] not in range(DP_SIZE)
        or row["candidate_frame_count"] != FRAME_COUNT
        or row["fps_numerator"] != 25
        or row["fps_denominator"] != 1
        or type(row["width"]) is not int
        or type(row["height"]) is not int
        or min(row["width"], row["height"]) <= 0
        or row["source_encode_and_terminal_decode_same_vae_authority"] is not True
        or row["target_media_read_count"] != 0
    ):
        fail("decoded rollout fixed contract differs")
    for key in ("rollout_id", "source_row_id"):
        _safe_id(row[key], label=f"decoded rollout {key}")
    for key in (
        "behavior_policy_sha256", "source_video_sha256", "instruction_sha256",
        "trajectory_receipt_sha256", "trajectory_receipt_digest",
        "trajectory_artifact_sha256", "terminal_state_sha256",
        "normalized_latent_sha256", "normalized_latent_tensor_sha256",
        "candidate_media_sha256", "full_decode_tree_digest",
    ):
        _sha256(row[key], label=f"decoded rollout {key}")
    for key in (
        "trajectory_receipt_path", "trajectory_artifact_path",
        "normalized_latent_path", "candidate_media_path",
    ):
        _absolute_path(row[key], label=f"decoded rollout {key}")
    frames = row["full_decode_frame_sha256"]
    if (
        not isinstance(frames, list)
        or len(frames) != FRAME_COUNT
        or any(_SHA256.fullmatch(item) is None for item in frames if type(item) is str)
        or any(type(item) is not str for item in frames)
        or object_sha256(frames) != row["full_decode_tree_digest"]
    ):
        fail("decoded rollout exact81 frame tree differs")
    vae = _closed(row["vae_authority"], _VAE_AUTHORITY_FIELDS, label="VAE authority")
    if (
        vae["schema_version"] != "bernini-full644-owned-vae-authority-v1"
        or vae["base_checkpoint_tree_sha256"] != BASE_CHECKPOINT_TREE_SHA256
        or vae["checkpoint_content_manifest_sha256"]
        != BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("decoded rollout VAE authority differs")
    for key in (
        "checkpoint_snapshot_digest", "vae_file_inventory_digest", "vae_config_sha256"
    ):
        _sha256(vae[key], label=f"decoded rollout VAE {key}")
    _verify_digest(row, "receipt_digest", label="decoded rollout receipt")
    trajectory = load_trajectory_receipt_v1(
        Path(row["trajectory_receipt_path"]),
        expected_sha256=row["trajectory_receipt_sha256"],
    )
    if (
        trajectory.value["receipt_digest"] != row["trajectory_receipt_digest"]
        or trajectory.value["artifact_path"] != row["trajectory_artifact_path"]
        or trajectory.value["artifact_sha256"] != row["trajectory_artifact_sha256"]
        or trajectory.value["artifact_size_bytes"]
        != row["trajectory_artifact_size_bytes"]
        or trajectory.value["terminal_state_sha256"] != row["terminal_state_sha256"]
        or trajectory.value["rollout_id"] != row["rollout_id"]
        or trajectory.value["behavior_policy_sha256"]
        != row["behavior_policy_sha256"]
        or trajectory.value["round_index"] != row["round_index"]
        or trajectory.value["rollout_seed"] != row["rollout_seed"]
        or trajectory.value["dp_arm"] != row["dp_arm"]
        or trajectory.value["source_row_id"] != row["source_row_id"]
        or trajectory.value["source_video_sha256"] != row["source_video_sha256"]
        or trajectory.value["instruction_sha256"] != row["instruction_sha256"]
    ):
        fail("decoded rollout/trajectory exact join differs")
    for key, size_key in (
        ("normalized_latent_path", None),
        ("candidate_media_path", "candidate_media_size_bytes"),
    ):
        candidate = Path(row[key])
        observed = os.lstat(candidate)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o444
            or int(observed.st_nlink) != 1
            or (size_key is not None and int(observed.st_size) != row[size_key])
        ):
            fail(f"decoded rollout physical {key} differs")
    read_stable_file(
        Path(row["normalized_latent_path"]),
        expected_sha256=row["normalized_latent_sha256"],
        expected_mode=0o444,
        label="decoded rollout normalized latent",
    )
    read_stable_file(
        Path(row["candidate_media_path"]),
        expected_sha256=row["candidate_media_sha256"],
        expected_mode=0o444,
        label="decoded rollout candidate media",
    )
    return row, trajectory


def _load_gate_from_held_source_v1() -> tuple[Any, Mapping[str, Any]]:
    """Compile the pinned gate bytes directly, bypassing ``__pycache__``."""

    source_path = Path(__file__).resolve().with_name("action_preservation_gate_v1.py")
    raw = read_stable_file(
        source_path,
        expected_sha256=ACTION_GATE_SOURCE_SHA256,
        label="action-preservation gate source",
    )
    if len(raw) != ACTION_GATE_SOURCE_SIZE:
        fail("action-preservation gate source size differs")
    module_name = "_full644_held_action_preservation_gate_v1"
    if module_name in sys.modules:
        fail("held action-preservation gate module cache is not empty")
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[module_name] = module
    try:
        code = compile(raw, str(source_path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__, module.__dict__)
        for name in (
            "validate_measurement",
            "validate_calibration",
            "decide",
            "validate_decision",
        ):
            function = getattr(module, name, None)
            if not callable(function) or getattr(function, "__module__", None) != module_name:
                fail("held action-preservation gate callable ownership differs")
        return module, {
            "source_path": str(source_path),
            "source_sha256": ACTION_GATE_SOURCE_SHA256,
            "source_size_bytes": ACTION_GATE_SOURCE_SIZE,
            "executed_held_source_bytes": True,
            "python_bytecode_cache_used": False,
            "callable_ownership_verified": True,
        }
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _release_held_gate_v1(module: Any) -> None:
    name = getattr(module, "__name__", None)
    if name != "_full644_held_action_preservation_gate_v1":
        fail("held action-preservation gate release identity differs")
    if sys.modules.get(name) is not module:
        fail("held action-preservation gate cache ownership changed")
    del sys.modules[name]


def _axis_state(
    *, state: str, reasons: Sequence[str], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    if state not in ("pass", "fail", "undetermined"):
        fail("adapted axis state differs")
    if (
        not isinstance(reasons, (list, tuple))
        or any(type(reason) is not str or not reason for reason in reasons)
        or len(set(reasons)) != len(reasons)
        or not isinstance(evidence, Mapping)
    ):
        fail("adapted axis evidence differs")
    return {"state": state, "reasons": list(reasons), "evidence": dict(evidence)}


def _conjunction_axis_v1(
    axes: Mapping[str, Mapping[str, Any]], names: Sequence[str], *, label: str
) -> Mapping[str, Any]:
    states = [axes[name]["state"] for name in names]
    if "fail" in states:
        state = "fail"
    elif "undetermined" in states:
        state = "undetermined"
    else:
        state = "pass"
    reasons = [
        f"{name}:{reason}"
        for name in names
        for reason in axes[name]["reasons"]
        if axes[name]["state"] != "pass"
    ]
    return _axis_state(
        state=state,
        reasons=reasons,
        evidence={"conjunction": list(names), "label": label},
    )


def _adapt_gate_axes_v1(
    *, measurement: Mapping[str, Any], calibration: Mapping[str, Any], decision: Mapping[str, Any]
) -> Mapping[str, Any]:
    gate_axes = decision["axes"]
    if set(gate_axes) != set(GATE_AXES):
        fail("action-preservation gate exact6 axis closure differs")
    thresholds = calibration["thresholds"]
    action = measurement["action_order"]

    if not action["available"]:
        ordered = _axis_state(
            state="undetermined",
            reasons=["action_order_measurement_unavailable"],
            evidence={"action_receipt_sha256": action["receipt_sha256"]},
        )
        terminal = _axis_state(
            state="undetermined",
            reasons=["terminal_hold_measurement_unavailable"],
            evidence={"action_receipt_sha256": action["receipt_sha256"]},
        )
    else:
        order_failures: list[str] = []
        if action["reverse_rejected"] is not True:
            order_failures.append("reverse_action_not_rejected")
        if action["truncation_rejected"] is not True:
            order_failures.append("truncated_action_not_rejected")
        if action["score"] < thresholds["action_order_min"]:
            order_failures.append("ordered_transition_below_calibrated_minimum")
        ordered = _axis_state(
            state="fail" if order_failures else "pass",
            reasons=order_failures,
            evidence={
                "score": action["score"],
                "minimum": thresholds["action_order_min"],
                "reverse_rejected": action["reverse_rejected"],
                "truncation_rejected": action["truncation_rejected"],
                "action_receipt_sha256": action["receipt_sha256"],
            },
        )
        terminal_failures: list[str] = []
        if action["terminal_hold_score"] < thresholds["terminal_hold_score_min"]:
            terminal_failures.append("terminal_hold_below_calibrated_minimum")
        if action["terminal_hold_frames"] < thresholds["terminal_hold_frames_min"]:
            terminal_failures.append("terminal_hold_too_short")
        terminal = _axis_state(
            state="fail" if terminal_failures else "pass",
            reasons=terminal_failures,
            evidence={
                "score": action["terminal_hold_score"],
                "minimum_score": thresholds["terminal_hold_score_min"],
                "frames": action["terminal_hold_frames"],
                "minimum_frames": thresholds["terminal_hold_frames_min"],
                "start_frame": action["terminal_hold_start_frame"],
                "end_frame": action["terminal_hold_end_frame"],
                "action_receipt_sha256": action["receipt_sha256"],
            },
        )

    projected = {
        name: _axis_state(
            state=gate_axes[source]["state"],
            reasons=gate_axes[source]["reasons"],
            evidence={
                "gate_axis": source,
                "decision_digest": decision["decision_digest"],
            },
        )
        for name, source in (
            ("identity", "source_identity"),
            ("camera", "camera"),
            ("background", "background"),
        )
    }
    participant = projected["identity"]
    if measurement["scope"]["single_subject"] is not True:
        participant = _axis_state(
            state="undetermined",
            reasons=["participant_scope_is_not_single_subject"],
            evidence={
                "single_subject": measurement["scope"]["single_subject"],
                "decision_digest": decision["decision_digest"],
            },
        )
    event = _conjunction_axis_v1(
        gate_axes, ("action_order", "onset", "quality"), label="event"
    )
    # The frozen gate has no independently calibrated foreground-excluded
    # motion measurement.  Background appearance and camera stability are not
    # substitutes for that quantity, so this must remain undetermined.
    non_target_motion = _axis_state(
        state="undetermined",
        reasons=["independent_non_target_motion_measurement_not_available"],
        evidence={
            "gate_source_sha256": ACTION_GATE_SOURCE_SHA256,
            "appearance_or_camera_proxy_accepted": False,
        },
    )
    result = {
        "event": event,
        "participant": participant,
        "ordered_transition": ordered,
        "terminal_hold": terminal,
        "identity": projected["identity"],
        "camera": projected["camera"],
        "background": projected["background"],
        "non_target_motion": non_target_motion,
    }
    if tuple(result) != HARD_AXES:
        fail("adapted hard-axis order differs")
    return result


def validate_verifier_evidence_value_v1(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _VERIFIER_INPUT_FIELDS, label="verifier evidence")
    if (
        row["schema_version"] != VERIFIER_INPUT_SCHEMA
        or row["frame_count"] != FRAME_COUNT
        or type(row["fps"]) is not float
        or row["fps"] != FPS
        or row["gate_source_sha256"] != ACTION_GATE_SOURCE_SHA256
        or row["gate_source_size_bytes"] != ACTION_GATE_SOURCE_SIZE
        or row["independent_from_student"] is not True
        or row["student_parameters_or_loss_read"] is not False
    ):
        fail("verifier evidence fixed closure differs")
    _safe_id(row["candidate_id"], label="verifier candidate ID")
    for key in (
        "source_video_sha256",
        "candidate_video_sha256",
        "measurement_sha256",
        "calibration_sha256",
    ):
        _sha256(row[key], label=f"verifier evidence {key}")
    _absolute_path(row["measurement_path"], label="verifier measurement")
    _absolute_path(row["calibration_path"], label="verifier calibration")
    _verify_digest(row, "evidence_digest", label="verifier evidence")
    return row


def adapt_verifier_evidence_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_candidate_sha256: str,
    expected_source_sha256: str,
) -> Mapping[str, Any]:
    evidence_raw = read_stable_file(
        path,
        expected_sha256=expected_sha256,
        label="verifier evidence",
    )
    evidence = validate_verifier_evidence_value_v1(
        _strict_json(evidence_raw, label="verifier evidence")
    )
    if (
        evidence["candidate_video_sha256"]
        != _sha256(expected_candidate_sha256, label="expected candidate SHA")
        or evidence["source_video_sha256"]
        != _sha256(expected_source_sha256, label="expected source SHA")
    ):
        fail("verifier evidence media binding differs")
    measurement_raw = read_stable_file(
        Path(evidence["measurement_path"]),
        expected_sha256=evidence["measurement_sha256"],
        label="verifier measurement",
    )
    calibration_raw = read_stable_file(
        Path(evidence["calibration_path"]),
        expected_sha256=evidence["calibration_sha256"],
        label="verifier calibration",
    )
    measurement_value = _strict_json(measurement_raw, label="verifier measurement")
    calibration_value = _strict_json(calibration_raw, label="verifier calibration")
    gate, source_binding = _load_gate_from_held_source_v1()
    try:
        measurement = gate.validate_measurement(measurement_value)
        calibration = gate.validate_calibration(calibration_value)
        decision = gate.decide(measurement, calibration)
        decision = gate.validate_decision(decision)
    except Exception as error:
        raise TargetFreeBerniniRuntimeError(
            f"independent action-preservation gate rejected evidence: {error}"
        ) from error
    finally:
        _release_held_gate_v1(gate)
    if (
        measurement["candidate_id"] != evidence["candidate_id"]
        or measurement["candidate_video_sha256"]
        != evidence["candidate_video_sha256"]
        or measurement["source_video_sha256"] != evidence["source_video_sha256"]
        or decision["candidate_id"] != evidence["candidate_id"]
        or decision["candidate_video_sha256"]
        != evidence["candidate_video_sha256"]
        or decision["source_video_sha256"] != evidence["source_video_sha256"]
    ):
        fail("verifier evidence/measurement/decision join differs")
    axes = _adapt_gate_axes_v1(
        measurement=measurement, calibration=calibration, decision=decision
    )
    states = [axes[axis]["state"] for axis in HARD_AXES]
    status = (
        "fail"
        if "fail" in states
        else "undetermined"
        if "undetermined" in states
        else "pass"
    )
    result = {
        "schema_version": VERIFIER_ADAPTER_SCHEMA,
        "candidate_id": evidence["candidate_id"],
        "candidate_video_sha256": evidence["candidate_video_sha256"],
        "source_video_sha256": evidence["source_video_sha256"],
        "measurement_sha256": evidence["measurement_sha256"],
        "measurement_digest": measurement["measurement_digest"],
        "calibration_sha256": evidence["calibration_sha256"],
        "calibration_digest": calibration["calibration_digest"],
        "gate_decision_digest": decision["decision_digest"],
        "gate_source_binding": source_binding,
        "hard_axes": axes,
        "hard_axis_order": list(HARD_AXES),
        "scalar_compensation_allowed": False,
        "manifest_axis_pass_used_for_admission": False,
        "status": status,
        "update_eligible": status == "pass",
        "adapter_digest": "",
    }
    result["adapter_digest"] = object_sha256(
        {key: value for key, value in result.items() if key != "adapter_digest"}
    )
    return result


def assess_pair_admission_v1(pair: Any) -> Mapping[str, Any]:
    """Recompute both endpoint verdicts; never trust manifest axis booleans."""

    chosen = pair.chosen
    rejected = pair.rejected
    chosen_adapter = adapt_verifier_evidence_v1(
        chosen.verifier_receipt_path,
        expected_sha256=chosen.verifier_receipt_sha256,
        expected_candidate_sha256=chosen.output_media_sha256,
        expected_source_sha256=pair.source.source_video_sha256,
    )
    rejected_adapter = adapt_verifier_evidence_v1(
        rejected.verifier_receipt_path,
        expected_sha256=rejected.verifier_receipt_sha256,
        expected_candidate_sha256=rejected.output_media_sha256,
        expected_source_sha256=pair.source.source_video_sha256,
    )
    undetermined = [
        f"{role}:{axis}"
        for role, adapter in (
            ("chosen", chosen_adapter),
            ("rejected", rejected_adapter),
        )
        for axis in HARD_AXES
        if adapter["hard_axes"][axis]["state"] == "undetermined"
    ]
    rejected_failures = [
        axis
        for axis in HARD_AXES
        if rejected_adapter["hard_axes"][axis]["state"] == "fail"
    ]
    eligible = (
        not undetermined
        and chosen_adapter["status"] == "pass"
        and rejected_adapter["status"] == "fail"
        and bool(rejected_failures)
    )
    result = {
        "schema_version": "bernini-full644-pair-verifier-admission-v1",
        "pair_id": pair.pair_id,
        "source_row_id": pair.source.row_id,
        "chosen_rollout_id": chosen.rollout_id,
        "rejected_rollout_id": rejected.rollout_id,
        "chosen_adapter": chosen_adapter,
        "rejected_adapter": rejected_adapter,
        "undetermined_axes": undetermined,
        "rejected_failed_axes": rejected_failures,
        "manifest_axis_pass_used_for_admission": False,
        "scalar_compensation_allowed": False,
        "eligible": eligible,
    }
    return {**result, "admission_digest": object_sha256(result)}


@dataclass(frozen=True)
class EngineeringVerifierAuthorityV1:
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    model_closure_path: Path
    model_closure_sha256: str
    model_closure_size_bytes: int
    model_revision: str


def _frozen_qwen_authority_v1() -> EngineeringVerifierAuthorityV1:
    return EngineeringVerifierAuthorityV1(
        source_path=Path(__file__).resolve().with_name(
            "full644_target_free_qwen_verifier_v1.py"
        ),
        source_sha256=QWEN_VERIFIER_SOURCE_SHA256,
        source_size_bytes=QWEN_VERIFIER_SOURCE_SIZE,
        model_closure_path=QWEN_MODEL_CLOSURE_PATH,
        model_closure_sha256=QWEN_MODEL_CLOSURE_SHA256,
        model_closure_size_bytes=QWEN_MODEL_CLOSURE_SIZE,
        model_revision=QWEN_MODEL_REVISION,
    )


def _load_qwen_verifier_from_held_source_v1(
    authority: EngineeringVerifierAuthorityV1,
) -> tuple[Any, Mapping[str, Any]]:
    if type(authority) is not EngineeringVerifierAuthorityV1:
        fail("engineering verifier authority type differs")
    if authority != _frozen_qwen_authority_v1():
        fail("engineering verifier authority is not the frozen release literal")
    raw = read_stable_file(
        authority.source_path,
        expected_sha256=authority.source_sha256,
        label="engineering Qwen verifier source",
    )
    if (
        len(raw) != authority.source_size_bytes
        or type(authority.source_size_bytes) is not int
        or authority.source_size_bytes <= 0
    ):
        fail("engineering Qwen verifier source size differs")
    model_closure_raw = read_stable_file(
        authority.model_closure_path,
        expected_sha256=authority.model_closure_sha256,
        label="engineering Qwen model closure",
    )
    if (
        len(model_closure_raw) != authority.model_closure_size_bytes
        or authority.model_closure_path != QWEN_MODEL_CLOSURE_PATH
        or authority.model_closure_sha256 != QWEN_MODEL_CLOSURE_SHA256
        or authority.model_closure_size_bytes != QWEN_MODEL_CLOSURE_SIZE
        or type(authority.model_revision) is not str
        or authority.model_revision != QWEN_MODEL_REVISION
    ):
        fail("Qwen held model closure differs")
    module_name = "_full644_held_qwen_verifier_v1"
    if module_name in sys.modules:
        fail("engineering Qwen verifier module cache is not empty")
    module = types.ModuleType(module_name)
    module.__file__ = str(authority.source_path)
    module.__package__ = ""
    module.__loader__ = None
    sys.modules[module_name] = module
    try:
        code = compile(raw, str(authority.source_path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__, module.__dict__)
        loader = getattr(module, "load_candidate_verdict_v1", None)
        if (
            not callable(loader)
            or getattr(loader, "__module__", None) != module_name
            or getattr(module, "QWEN_MODEL_CLOSURE_PATH", None)
            != authority.model_closure_path
            or getattr(module, "QWEN_MODEL_CLOSURE_SHA256", None)
            != authority.model_closure_sha256
            or getattr(module, "QWEN_MODEL_CLOSURE_SIZE", None)
            != authority.model_closure_size_bytes
            or getattr(module, "QWEN_MODEL_REVISION", None)
            != authority.model_revision
            or getattr(module, "QWEN_MODEL_SNAPSHOT_DIGEST", None)
            != QWEN_MODEL_SNAPSHOT_DIGEST
            or getattr(module, "ONE_SOURCE_ROW_ID", None) != ONE_SOURCE_ROW_ID
            or getattr(module, "ONE_SOURCE_VIDEO_SHA256", None)
            != ONE_SOURCE_VIDEO_SHA256
            or getattr(module, "ONE_SOURCE_INSTRUCTION_SHA256", None)
            != ONE_SOURCE_INSTRUCTION_SHA256
            or tuple(getattr(module, "HARD_AXES", ())) != HARD_AXES
            or getattr(module, "DETERMINISTIC_GENERATION", None)
            != QWEN_DETERMINISTIC_GENERATION
        ):
            fail("engineering Qwen verifier loader ownership differs")
        return module, {
            "source_path": str(authority.source_path),
            "source_sha256": authority.source_sha256,
            "source_size_bytes": authority.source_size_bytes,
            "model_closure_path": str(authority.model_closure_path),
            "model_closure_sha256": authority.model_closure_sha256,
            "model_closure_size_bytes": authority.model_closure_size_bytes,
            "model_revision": authority.model_revision,
            "model_closure_held_and_hashed": True,
            "executed_held_source_bytes": True,
            "python_bytecode_cache_used": False,
            "loader_ownership_verified": True,
        }
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _expected_qwen_qualification_v1(
    authority: EngineeringVerifierAuthorityV1,
) -> Mapping[str, Any]:
    if authority != _frozen_qwen_authority_v1():
        fail("Qwen qualification authority differs")
    qualification_set = {
        "schema_version": "bernini-full644-qwen-exact8-qualification-set-v1",
        "verifier_release_sha256": authority.source_sha256,
        "model_closure_sha256": authority.model_closure_sha256,
        "deterministic_generation": QWEN_DETERMINISTIC_GENERATION,
        "hard_axes": list(HARD_AXES),
    }
    return {
        "schema_version": "bernini-full644-hard-axis-verifier-qualification-v1",
        "verifier_release_sha256": authority.source_sha256,
        "verifier_model_sha256": authority.model_closure_sha256,
        "qualification_set_sha256": object_sha256(qualification_set),
        "independent_from_student": True,
        "hard_axis_conjunction": list(HARD_AXES),
        "scalar_compensation_allowed": False,
    }


def _validate_qwen_verdict_projection_v1(
    value: Any,
    *,
    authority: EngineeringVerifierAuthorityV1,
    rollout: Any,
    source: Any,
    expected_deterministic_generation: Mapping[str, Any],
) -> Mapping[str, Any]:
    row = _closed(value, _QWEN_VERDICT_FIELDS, label="engineering Qwen verdict")
    if (
        row["schema_version"] != "bernini-full644-qwen-candidate-verdict-v1"
        or row["verifier_release_sha256"] != authority.source_sha256
        or row["model_closure_sha256"] != authority.model_closure_sha256
        or row["model_revision"] != authority.model_revision
        or row["rollout_id"] != rollout.rollout_id
        or row["policy_sha256"] != rollout.policy_sha256
        or row["round_index"] != rollout.round_index
        or row["seed"] != rollout.seed
        or type(row["dp_arm"]) is not int
        or row["dp_arm"] not in range(DP_SIZE)
        or row["source_row_id"] != source.row_id
        or row["source_video_sha256"] != source.source_video_sha256
        or row["instruction_sha256"] != source.instruction_sha256
        or row["candidate_media_path"] != str(rollout.output_media_path)
        or row["candidate_media_sha256"] != rollout.output_media_sha256
        or row["independent_from_student"] is not True
        or row["student_parameters_or_loss_read"] is not False
        or row["engineering_only"] is not True
        or row["scientific_result_claimed"] is not False
    ):
        fail("engineering Qwen verdict authority/join differs")
    for key in (
        "verifier_release_sha256",
        "model_closure_sha256",
        "source_video_sha256",
        "instruction_sha256",
        "candidate_media_sha256",
        "visual_input_sha256",
        "raw_response_sha256",
        "decoded_rollout_receipt_sha256",
        "decoded_rollout_receipt_digest",
        "trajectory_receipt_sha256",
        "trajectory_receipt_digest",
        "trajectory_artifact_sha256",
        "terminal_state_sha256",
    ):
        _sha256(row[key], label=f"engineering Qwen verdict {key}")
    axes = row["hard_axes"]
    if not isinstance(axes, Mapping) or set(axes) != set(HARD_AXES):
        fail("engineering Qwen exact8 axis closure differs")
    normalized_axes = {}
    for axis in HARD_AXES:
        item = _closed(axes[axis], _QWEN_AXIS_FIELDS, label=f"Qwen axis {axis}")
        state = item["state"]
        evidence = item["evidence"]
        if state not in ("pass", "fail", "undetermined"):
            fail("engineering Qwen axis state differs")
        if not isinstance(evidence, list) or not evidence:
            fail("engineering Qwen axis evidence is empty")
        normalized_evidence = []
        for raw_evidence in evidence:
            evidence_row = _closed(
                raw_evidence,
                _QWEN_EVIDENCE_ROW_FIELDS,
                label=f"Qwen axis {axis} evidence",
            )
            for frame_key, prefix in (
                ("source_frames", "S"),
                ("candidate_frames", "C"),
            ):
                frames = evidence_row[frame_key]
                if (
                    not isinstance(frames, list)
                    or not frames
                    or any(
                        type(frame) is not str
                        or re.fullmatch(rf"{prefix}(?:[0-9]|1[01])", frame) is None
                        for frame in frames
                    )
                    or len(set(frames)) != len(frames)
                ):
                    fail("engineering Qwen evidence frame binding differs")
            observation = evidence_row["observation"]
            if (
                type(observation) is not str
                or observation != observation.strip()
                or not 1 <= len(observation) <= 2048
                or "\x00" in observation
            ):
                fail("engineering Qwen evidence observation differs")
            normalized_evidence.append(dict(evidence_row))
        normalized_axes[axis] = {"state": state, "evidence": normalized_evidence}
    uncertainty = row["uncertainty_codes"]
    if (
        not isinstance(uncertainty, list)
        or any(type(item) is not str or not item or len(item) > 128 for item in uncertainty)
        or len(set(uncertainty)) != len(uncertainty)
    ):
        fail("engineering Qwen uncertainty closure differs")
    deterministic = _closed(
        row["deterministic_generation"],
        _QWEN_DETERMINISTIC_FIELDS,
        label="Qwen deterministic generation",
    )
    if (
        not isinstance(expected_deterministic_generation, Mapping)
        or dict(deterministic) != dict(expected_deterministic_generation)
        or deterministic["schema_version"]
        != "bernini-full644-qwen25-vl-deterministic-generation-v1"
        or deterministic["model_closure_sha256"] != authority.model_closure_sha256
        or deterministic["model_revision"] != authority.model_revision
        or deterministic["transformers_version"] != "5.5.4"
        or deterministic["local_files_only"] is not True
        or deterministic["trust_remote_code"] is not False
        or deterministic["model_eval"] is not True
        or deterministic["inference_mode"] is not True
        or deterministic["torch_dtype"] != "bfloat16"
        or deterministic["attention_implementation"] != "eager"
        or deterministic["do_sample"] is not False
        or deterministic["num_beams"] != 1
        or deterministic["max_new_tokens"] != 2048
        or deterministic["seed"] != 0
        or deterministic["source_frame_indices"]
        != [0, 7, 15, 22, 29, 36, 44, 51, 58, 65, 73, 80]
        or deterministic["candidate_frame_indices"]
        != [0, 7, 15, 22, 29, 36, 44, 51, 58, 65, 73, 80]
        or deterministic["response_schema"]
        != "bernini-full644-qwen-exact8-response-v1"
    ):
        fail("engineering Qwen deterministic-generation evidence differs")
    states = [normalized_axes[axis]["state"] for axis in HARD_AXES]
    all_pass = all(state == "pass" for state in states)
    qualification = _closed(
        row["qualification"],
        _QWEN_QUALIFICATION_FIELDS,
        label="Qwen qualification",
    )
    expected_qualification = {
        "eligible_for_engineering_pair_selection": all_pass,
        "all_eight_axes_pass": all_pass,
        "any_axis_fail": any(state == "fail" for state in states),
        "any_axis_undetermined": any(state == "undetermined" for state in states),
    }
    if dict(qualification) != expected_qualification:
        fail("engineering Qwen qualification is not derived from exact8")
    decoded_path = _absolute_path(
        row["decoded_rollout_receipt_path"],
        label="Qwen decoded rollout receipt",
    )
    trajectory_suffix = ".trajectory.json"
    trajectory_path = Path(rollout.trajectory_receipt_path)
    if not trajectory_path.name.endswith(trajectory_suffix):
        fail("Qwen rollout trajectory basename cannot derive decoded receipt")
    expected_decoded_path = trajectory_path.with_name(
        trajectory_path.name[: -len(trajectory_suffix)] + ".decoded-rollout.json"
    )
    if decoded_path != expected_decoded_path:
        fail("Qwen decoded rollout path is not the fixed trajectory projection")
    decoded, decoded_trajectory = load_decoded_rollout_receipt_v1(
        decoded_path,
        expected_sha256=row["decoded_rollout_receipt_sha256"],
    )
    if (
        decoded["receipt_digest"] != row["decoded_rollout_receipt_digest"]
        or decoded_trajectory.sha256 != row["trajectory_receipt_sha256"]
        or decoded_trajectory.value["receipt_digest"]
        != row["trajectory_receipt_digest"]
        or decoded_trajectory.value["artifact_sha256"]
        != row["trajectory_artifact_sha256"]
        or decoded_trajectory.value["terminal_state_sha256"]
        != row["terminal_state_sha256"]
        or decoded["candidate_media_path"] != row["candidate_media_path"]
        or decoded["candidate_media_sha256"] != row["candidate_media_sha256"]
        or decoded["dp_arm"] != row["dp_arm"]
    ):
        fail("engineering Qwen decoded-rollout projection differs")

    def validate_media_probe(
        value: Any, *, expected_sha: str, label: str
    ) -> Mapping[str, Any]:
        probe = _closed(value, _QWEN_MEDIA_PROBE_FIELDS, label=label)
        frames = probe["full_decode_frame_sha256"]
        if (
            probe["schema_version"] != "bernini-full644-held-exact81-media-probe-v1"
            or probe["media_sha256"] != expected_sha
            or probe["frame_count"] != FRAME_COUNT
            or probe["fps_numerator"] != 25
            or probe["fps_denominator"] != 1
            or type(probe["width"]) is not int
            or type(probe["height"]) is not int
            or min(probe["width"], probe["height"]) <= 0
            or probe["fully_decoded"] is not True
            or not isinstance(frames, list)
            or len(frames) != FRAME_COUNT
            or any(type(item) is not str or _SHA256.fullmatch(item) is None for item in frames)
            or object_sha256(frames) != probe["full_decode_tree_digest"]
        ):
            fail(f"{label} exact81/25fps projection differs")
        return probe

    source_probe = validate_media_probe(
        row["source_media_probe"],
        expected_sha=source.source_video_sha256,
        label="Qwen source media probe",
    )
    candidate_probe = validate_media_probe(
        row["candidate_media_probe"],
        expected_sha=rollout.output_media_sha256,
        label="Qwen candidate media probe",
    )
    if (
        candidate_probe["width"] != decoded["width"]
        or candidate_probe["height"] != decoded["height"]
        or candidate_probe["full_decode_frame_sha256"]
        != decoded["full_decode_frame_sha256"]
        or candidate_probe["full_decode_tree_digest"]
        != decoded["full_decode_tree_digest"]
    ):
        fail("Qwen candidate media probe/decoded frame tree differs")
    visual = _closed(
        row["visual_execution"],
        _QWEN_VISUAL_EXECUTION_FIELDS,
        label="Qwen visual execution",
    )
    frame_indices = [0, 7, 15, 22, 29, 36, 44, 51, 58, 65, 73, 80]
    if (
        visual["schema_version"]
        != "bernini-full644-qwen25-vl-visual-execution-v1"
        or visual["model_closure_sha256"] != authority.model_closure_sha256
        or visual["model_snapshot_digest"] != QWEN_MODEL_SNAPSHOT_DIGEST
        or visual["source_media_sha256"] != source.source_video_sha256
        or visual["candidate_media_sha256"] != rollout.output_media_sha256
        or visual["instruction_sha256"] != source.instruction_sha256
        or visual["sampled_frame_indices"] != frame_indices
        or visual["source_sampled_frame_sha256"]
        != [source_probe["full_decode_frame_sha256"][index] for index in frame_indices]
        or visual["candidate_sampled_frame_sha256"]
        != [candidate_probe["full_decode_frame_sha256"][index] for index in frame_indices]
        or visual["raw_response_sha256"] != row["raw_response_sha256"]
        or visual["visual_input_digest"] != row["visual_input_sha256"]
    ):
        fail("Qwen visual execution input/output join differs")
    for key in (
        "model_snapshot_digest", "source_media_sha256", "candidate_media_sha256",
        "instruction_sha256", "source_mosaic_pixel_sha256",
        "candidate_mosaic_pixel_sha256", "source_mosaic_png_sha256",
        "candidate_mosaic_png_sha256", "rendered_prompt_sha256",
        "input_ids_sha256", "output_ids_sha256", "raw_response_sha256",
        "visual_input_digest", "execution_digest",
    ):
        _sha256(visual[key], label=f"Qwen visual execution {key}")
    visual_input = {
        key: visual[key]
        for key in (
            "schema_version", "model_closure_sha256", "model_snapshot_digest",
            "source_media_sha256", "candidate_media_sha256",
            "instruction_sha256", "sampled_frame_indices",
            "source_sampled_frame_sha256", "candidate_sampled_frame_sha256",
            "source_mosaic_pixel_sha256", "candidate_mosaic_pixel_sha256",
            "source_mosaic_png_sha256", "candidate_mosaic_png_sha256",
            "rendered_prompt_sha256", "input_ids_sha256",
        )
    }
    if object_sha256(visual_input) != visual["visual_input_digest"]:
        fail("Qwen visual execution input digest differs")
    _verify_digest(visual, "execution_digest", label="Qwen visual execution")
    _verify_digest(row, "receipt_digest", label="engineering Qwen verdict")
    return {**dict(row), "hard_axes": normalized_axes}


def adapt_qwen_verdict_v1(
    rollout: Any,
    *,
    source: Any,
    authority: EngineeringVerifierAuthorityV1,
) -> Mapping[str, Any]:
    """Load an independently executed exact8 engineering verdict."""

    module, source_binding = _load_qwen_verifier_from_held_source_v1(authority)
    verdict_raw = read_stable_file(
        rollout.verifier_receipt_path,
        expected_sha256=rollout.verifier_receipt_sha256,
        expected_mode=0o444,
        label="engineering Qwen verdict preprojection",
    )
    verdict_preprojection = _strict_json(
        verdict_raw, label="engineering Qwen verdict preprojection"
    )
    declared_decoded_sha = _sha256(
        verdict_preprojection.get("decoded_rollout_receipt_sha256"),
        label="Qwen declared decoded-rollout SHA",
    )
    try:
        loaded = module.load_candidate_verdict_v1(
            path=rollout.verifier_receipt_path,
            expected_sha256=rollout.verifier_receipt_sha256,
            expected_source_sha256=source.source_video_sha256,
            expected_candidate_sha256=rollout.output_media_sha256,
            expected_instruction_sha256=source.instruction_sha256,
            expected_decoded_rollout_sha256=declared_decoded_sha,
            expected_verifier_release_sha256=authority.source_sha256,
        )
        deterministic_generation = dict(module.DETERMINISTIC_GENERATION)
    except Exception as error:
        raise TargetFreeBerniniRuntimeError(
            f"engineering Qwen verifier rejected receipt: {error}"
        ) from error
    finally:
        name = getattr(module, "__name__", None)
        if sys.modules.get(name) is not module:
            fail("engineering Qwen verifier cache ownership changed")
        del sys.modules[name]
    row = _validate_qwen_verdict_projection_v1(
        loaded,
        authority=authority,
        rollout=rollout,
        source=source,
        expected_deterministic_generation=deterministic_generation,
    )
    states = [row["hard_axes"][axis]["state"] for axis in HARD_AXES]
    status = (
        "fail"
        if "fail" in states
        else "undetermined"
        if "undetermined" in states
        else "pass"
    )
    result = {
        "schema_version": "bernini-full644-qwen-exact8-adapter-v1",
        "rollout_id": rollout.rollout_id,
        "candidate_media_sha256": rollout.output_media_sha256,
        "dp_arm": row["dp_arm"],
        "decoded_rollout_receipt_path": row["decoded_rollout_receipt_path"],
        "decoded_rollout_receipt_sha256": row["decoded_rollout_receipt_sha256"],
        "decoded_rollout_receipt_digest": row["decoded_rollout_receipt_digest"],
        "trajectory_receipt_sha256": row["trajectory_receipt_sha256"],
        "trajectory_receipt_digest": row["trajectory_receipt_digest"],
        "trajectory_artifact_sha256": row["trajectory_artifact_sha256"],
        "terminal_state_sha256": row["terminal_state_sha256"],
        "source_video_sha256": source.source_video_sha256,
        "instruction_sha256": source.instruction_sha256,
        "verdict_receipt_sha256": rollout.verifier_receipt_sha256,
        "verdict_receipt_digest": row["receipt_digest"],
        "verifier_source_binding": source_binding,
        "model_closure_sha256": authority.model_closure_sha256,
        "model_snapshot_digest": QWEN_MODEL_SNAPSHOT_DIGEST,
        "visual_input_sha256": row["visual_input_sha256"],
        "visual_execution_digest": row["visual_execution"]["execution_digest"],
        "hard_axes": row["hard_axes"],
        "qualification": row["qualification"],
        "hard_axis_order": list(HARD_AXES),
        "manifest_axis_pass_used_for_admission": False,
        "engineering_only": True,
        "scientific_result_claimed": False,
        "status": status,
        "update_eligible": status == "pass",
    }
    return {**result, "adapter_digest": object_sha256(result)}


def assess_pair_engineering_admission_v1(
    pair: Any, *, authority: EngineeringVerifierAuthorityV1
) -> Mapping[str, Any]:
    chosen = adapt_qwen_verdict_v1(
        pair.chosen, source=pair.source, authority=authority
    )
    rejected = adapt_qwen_verdict_v1(
        pair.rejected, source=pair.source, authority=authority
    )
    undetermined = [
        f"{role}:{axis}"
        for role, adapter in (("chosen", chosen), ("rejected", rejected))
        for axis in HARD_AXES
        if adapter["hard_axes"][axis]["state"] == "undetermined"
    ]
    rejected_failures = [
        axis
        for axis in HARD_AXES
        if rejected["hard_axes"][axis]["state"] == "fail"
    ]
    eligible = (
        not undetermined
        and chosen["status"] == "pass"
        and rejected["status"] == "fail"
        and bool(rejected_failures)
    )
    result = {
        "schema_version": "bernini-full644-qwen-pair-engineering-admission-v1",
        "pair_id": pair.pair_id,
        "source_row_id": pair.source.row_id,
        "chosen": chosen,
        "rejected": rejected,
        "undetermined_axes": undetermined,
        "rejected_failed_axes": rejected_failures,
        "manifest_axis_pass_used_for_admission": False,
        "engineering_only": True,
        "scientific_result_claimed": False,
        "eligible": eligible,
    }
    return {**result, "admission_digest": object_sha256(result)}


@dataclass(frozen=True)
class Exact40CoordinateV1:
    index: int
    timestep: Any = field(repr=False, compare=False)
    sigma: Any = field(repr=False, compare=False)
    timestep_value: int
    sigma_float32_be_hex: str


def _load_schedule_authority_v1() -> Any:
    authority = _require_held_local_module_v1("inference_sigma_strata")
    if (
        getattr(authority, "SCHEDULE_SHA256", None) != SCHEDULE_SHA256
        or tuple(getattr(authority, "PINNED_TIMESTEPS", ())) == ()
        or len(authority.PINNED_TIMESTEPS) != TRAJECTORY_STEPS
        or len(authority.PINNED_POSITIVE_SIGMA_FLOAT32_HEX)
        != TRAJECTORY_STEPS
    ):
        fail("exact40 schedule authority differs")
    return authority


def _unpack_generated_velocity_v1(packed: Any, *, spatial_shape: Sequence[int]) -> Any:
    """Invert Bernini/Wan patch order while preserving the autograd graph."""

    import torch

    shape = tuple(int(item) for item in spatial_shape)
    if (
        len(shape) != 5
        or shape[:3] != (1, 16, 21)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % 2
        or shape[4] % 2
    ):
        fail("generated latent must be exact81 [1,16,21,H,W] with even H/W")
    batch, channels, phases, height, width = shape
    patch_h, patch_w = height // 2, width // 2
    tokens = phases * patch_h * patch_w
    if (
        type(packed) is not torch.Tensor
        or tuple(packed.shape) != (batch, tokens, 64)
        or not bool(torch.isfinite(packed).all().item())
    ):
        fail("packed Bernini generated velocity differs")
    patches = packed.reshape(batch, phases, patch_h, patch_w, 1, 2, 2, channels)
    result = (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )
    if not bool(torch.isfinite(result).all().item()):
        fail("unpacked Bernini generated velocity is non-finite")
    return result


@dataclass(frozen=True)
class _NativePackV1:
    visual: Any = field(repr=False, compare=False)
    rotary: Any = field(repr=False, compare=False)
    source_tokens: int
    generated_tokens: int


def _target_free_requested_lora_targets_v1() -> set[str]:
    return {
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    }


def _validate_target_free_peft_config_v1(
    adapter_config: Any,
    *,
    expected_targets: set[str],
    target_modules_contract: str,
) -> Mapping[str, Any]:
    """Close PEFT-0.19.1 semantics before and after its target-set rewrite.

    PEFT 0.19.1 accepts the exact240 full module paths and a null base-model
    name at construction, then canonicalizes the installed adapter config to
    the exact4 unique suffixes and the Transformers empty-string sentinel.
    These are two distinct, fixed contracts; neither is accepted at the other
    phase.
    """

    try:
        import peft
    except ImportError as error:
        raise TargetFreeBerniniRuntimeError("PEFT is unavailable") from error
    if getattr(peft, "__version__", None) != PEFT_VERSION:
        fail("target-free runtime requires exact PEFT 0.19.1")
    to_dict = getattr(adapter_config, "to_dict", None)
    if not callable(to_dict):
        fail("target-free LoRA config cannot be projected")
    value = to_dict()
    if not isinstance(value, Mapping) or set(value) != set(_PEFT_CONFIG_FIELDS):
        fail("target-free PEFT config field closure differs")
    expected = {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": None,
        "base_model_name_or_path": None,
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": LORA_ALPHA,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": PEFT_VERSION,
        "qalora_group_size": 16,
        "r": LORA_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    requested_targets = _target_free_requested_lora_targets_v1()
    canonical_targets = set(_PEFT_CANONICAL_TARGET_MODULES)
    if target_modules_contract == _PEFT_REQUESTED_TARGET_CONTRACT:
        if expected_targets != requested_targets or len(expected_targets) != LORA_AFFINES:
            fail("target-free PEFT requested exact240 authority differs")
        expected["base_model_name_or_path"] = None
    elif target_modules_contract == _PEFT_CANONICAL_TARGET_CONTRACT:
        if expected_targets != canonical_targets or len(expected_targets) != 4:
            fail("target-free PEFT canonical exact4 authority differs")
        expected["base_model_name_or_path"] = ""
    else:
        fail("target-free PEFT target-module contract differs")
    if (
        type(value.get("target_modules")) is not set
        or value["target_modules"] != expected_targets
    ):
        fail(f"target-free PEFT {target_modules_contract} target modules differ")

    def same(observed: Any, wanted: Any) -> bool:
        if wanted is None:
            return observed is None
        if type(wanted) in (bool, int, float, dict):
            return type(observed) is type(wanted) and observed == wanted
        if type(wanted) is str:
            return isinstance(observed, str) and observed == wanted
        return type(observed) is type(wanted) and observed == wanted

    for name, wanted in expected.items():
        observed = value.get(name)
        if name == "base_model_name_or_path":
            matched = (
                observed is None
                if wanted is None
                else type(observed) is str and observed == wanted
            )
        else:
            matched = same(observed, wanted)
        if not matched:
            fail(f"target-free PEFT semantic differs: {name}")
    projection = {
        key: sorted(item) if type(item) is set else item
        for key, item in value.items()
    }
    targets = sorted(expected_targets)
    receipt = {
        "schema_version": "bernini-full644-peft-config-phase-v1",
        "peft_version": PEFT_VERSION,
        "target_modules_contract": target_modules_contract,
        "target_module_count": len(targets),
        "target_modules": targets,
        "target_modules_sha256": object_sha256(targets),
        "config": projection,
        "config_digest": object_sha256(projection),
    }
    return {**receipt, "receipt_digest": object_sha256(receipt)}


def _validate_target_free_peft_config_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    expected_targets: set[str],
    target_modules_contract: str,
) -> Mapping[str, Any]:
    """Re-open one phase receipt without trusting caller-signed target sets."""

    if not isinstance(receipt, Mapping) or set(receipt) != set(
        _PEFT_CONFIG_RECEIPT_FIELDS
    ):
        fail("target-free PEFT phase receipt fields differ")
    targets = sorted(expected_targets)
    if target_modules_contract == _PEFT_REQUESTED_TARGET_CONTRACT:
        if expected_targets != _target_free_requested_lora_targets_v1():
            fail("target-free PEFT requested receipt authority differs")
    elif target_modules_contract == _PEFT_CANONICAL_TARGET_CONTRACT:
        if expected_targets != set(_PEFT_CANONICAL_TARGET_MODULES):
            fail("target-free PEFT canonical receipt authority differs")
    else:
        fail("target-free PEFT receipt contract differs")
    if (
        receipt.get("schema_version") != "bernini-full644-peft-config-phase-v1"
        or receipt.get("peft_version") != PEFT_VERSION
        or receipt.get("target_modules_contract") != target_modules_contract
        or type(receipt.get("target_module_count")) is not int
        or receipt["target_module_count"] != len(targets)
        or type(receipt.get("target_modules")) is not list
        or receipt["target_modules"] != targets
        or receipt.get("target_modules_sha256") != object_sha256(targets)
        or not isinstance(receipt.get("config"), Mapping)
        or set(receipt["config"]) != set(_PEFT_CONFIG_FIELDS)
        or receipt["config"].get("target_modules") != targets
        or receipt.get("config_digest") != object_sha256(receipt["config"])
    ):
        fail("target-free PEFT phase receipt closure differs")
    body = {key: receipt[key] for key in receipt if key != "receipt_digest"}
    if receipt.get("receipt_digest") != object_sha256(body):
        fail("target-free PEFT phase receipt digest differs")
    projected_config = dict(receipt["config"])
    projected_config["target_modules"] = set(projected_config["target_modules"])

    class _ReceiptConfigProjectionV1:
        def to_dict(self) -> Mapping[str, Any]:
            return projected_config

    reopened = _validate_target_free_peft_config_v1(
        _ReceiptConfigProjectionV1(),
        expected_targets=expected_targets,
        target_modules_contract=target_modules_contract,
    )
    if canonical_json_bytes(dict(receipt)) != canonical_json_bytes(reopened):
        fail("target-free PEFT phase receipt semantic replay differs")
    return reopened


def _bind_target_free_peft_transition_v1(
    *,
    requested_receipt: Mapping[str, Any],
    canonical_receipt: Mapping[str, Any],
    lora_installation_digest: str,
) -> Mapping[str, Any]:
    requested = _validate_target_free_peft_config_receipt_v1(
        requested_receipt,
        expected_targets=_target_free_requested_lora_targets_v1(),
        target_modules_contract=_PEFT_REQUESTED_TARGET_CONTRACT,
    )
    canonical = _validate_target_free_peft_config_receipt_v1(
        canonical_receipt,
        expected_targets=set(_PEFT_CANONICAL_TARGET_MODULES),
        target_modules_contract=_PEFT_CANONICAL_TARGET_CONTRACT,
    )
    value = {
        "schema_version": "bernini-full644-peft-exact240-to-exact4-v1",
        "peft_version": PEFT_VERSION,
        "requested_exact240": requested,
        "canonical_postinstall_exact4": canonical,
        "requested_target_modules_sha256": requested["target_modules_sha256"],
        "canonical_target_modules_sha256": canonical["target_modules_sha256"],
        "installed_exact480_lora_tensor_count": LORA_TENSOR_COUNT,
        "lora_installation_digest": _sha256(
            lora_installation_digest,
            label="PEFT transition LoRA installation digest",
        ),
        "peft_0191_unique_suffix_canonicalization_verified": True,
    }
    return {**value, "transition_digest": object_sha256(value)}


class BerniniExact40PolicyV1:
    """Concrete current-policy Bernini runner; no user log-probability seam."""

    def __init__(
        self,
        *,
        renderer: Any,
        source_state: Any,
        negative_condition: Any,
        positive_condition: Any,
        source_row_id: str,
        source_video_sha256: str,
        instruction_sha256: str,
        base_model_sha256: str,
        model_closure_sha256: str,
        peft_config_transition_receipt: Mapping[str, Any],
        parallel: Any,
        _owned_factory_token: Any,
    ) -> None:
        import torch

        if _owned_factory_token is not _OWNED_RUNTIME_TOKEN:
            fail("Bernini runtime may only be constructed by the owned source/model factory")
        if not isinstance(renderer, torch.nn.Module):
            fail("Bernini runtime renderer must be one torch module")
        diffusion = getattr(renderer, "diff_dec", None)
        if diffusion is None and hasattr(renderer, "get_base_model"):
            diffusion = getattr(renderer.get_base_model(), "diff_dec", None)
        transformer = getattr(diffusion, "transformer", None)
        scheduler = getattr(diffusion, "scheduler", None)
        if (
            diffusion is None
            or not isinstance(transformer, torch.nn.Module)
            or scheduler is None
            or getattr(diffusion, "transformer_2", None) is not None
            or not callable(getattr(diffusion, "shared_step", None))
            or not callable(getattr(transformer, "patch_vae_latent", None))
        ):
            fail("Bernini runtime requires the single official transformer_1 diffusion core")
        if (
            type(source_state) is not torch.Tensor
            or source_state.dtype != torch.float32
            or source_state.device.type != "cuda"
            or source_state.requires_grad
            or len(source_state.shape) != 5
            or tuple(source_state.shape[:3]) != (1, 16, 21)
            or int(source_state.shape[3]) % 2
            or int(source_state.shape[4]) % 2
            or not bool(torch.isfinite(source_state).all().item())
        ):
            fail("Bernini source state ABI differs")
        for label, condition in (
            ("negative", negative_condition),
            ("positive", positive_condition),
        ):
            if (
                type(condition) is not torch.Tensor
                or tuple(condition.shape) != (1, 512, 4096)
                or condition.dtype != torch.bfloat16
                or condition.device != source_state.device
                or condition.requires_grad
                or not bool(torch.isfinite(condition).all().item())
            ):
                fail(f"Bernini {label} condition ABI differs")
        if torch.equal(negative_condition, positive_condition):
            fail("Bernini positive and negative conditions alias")
        contract = getattr(parallel, "contract", None)
        topology = getattr(contract, "topology", None)
        if (
            getattr(contract, "world_size", None) != WORLD_SIZE
            or getattr(topology, "world_size", None) != WORLD_SIZE
            or getattr(topology, "dp_size", None) != DP_SIZE
            or getattr(topology, "sp_size", None) != SP_SIZE
        ):
            fail("Bernini runtime requires live WORLD8 DP2xSP4")
        validate_world8_device_placement_v1(
            world_rank=getattr(contract, "rank", None),
            local_rank=getattr(contract, "local_rank", None),
            dp_arm=getattr(contract, "arm_index", None),
            sp_rank=getattr(contract, "sp_rank", None),
            device_index=source_state.device.index,
        )
        peft_config = getattr(renderer, "peft_config", None)
        if not isinstance(peft_config, Mapping) or set(peft_config) != {"default"}:
            fail("Bernini runtime requires exactly one default LoRA adapter")
        adapter_config = peft_config["default"]
        canonical_peft_receipt = _validate_target_free_peft_config_v1(
            adapter_config,
            expected_targets=set(_PEFT_CANONICAL_TARGET_MODULES),
            target_modules_contract=_PEFT_CANONICAL_TARGET_CONTRACT,
        )
        if (
            not isinstance(peft_config_transition_receipt, Mapping)
            or set(peft_config_transition_receipt)
            != {
                "schema_version",
                "peft_version",
                "requested_exact240",
                "canonical_postinstall_exact4",
                "requested_target_modules_sha256",
                "canonical_target_modules_sha256",
                "installed_exact480_lora_tensor_count",
                "lora_installation_digest",
                "peft_0191_unique_suffix_canonicalization_verified",
                "transition_digest",
            }
            or peft_config_transition_receipt.get("schema_version")
            != "bernini-full644-peft-exact240-to-exact4-v1"
            or peft_config_transition_receipt.get("peft_version") != PEFT_VERSION
            or peft_config_transition_receipt.get("installed_exact480_lora_tensor_count")
            != LORA_TENSOR_COUNT
            or peft_config_transition_receipt.get(
                "peft_0191_unique_suffix_canonicalization_verified"
            )
            is not True
        ):
            fail("Bernini PEFT requested/canonical transition fields differ")
        requested_peft_receipt = _validate_target_free_peft_config_receipt_v1(
            peft_config_transition_receipt["requested_exact240"],
            expected_targets=_target_free_requested_lora_targets_v1(),
            target_modules_contract=_PEFT_REQUESTED_TARGET_CONTRACT,
        )
        transition_canonical_receipt = _validate_target_free_peft_config_receipt_v1(
            peft_config_transition_receipt["canonical_postinstall_exact4"],
            expected_targets=set(_PEFT_CANONICAL_TARGET_MODULES),
            target_modules_contract=_PEFT_CANONICAL_TARGET_CONTRACT,
        )
        transition_body = {
            key: peft_config_transition_receipt[key]
            for key in peft_config_transition_receipt
            if key != "transition_digest"
        }
        if (
            transition_canonical_receipt != canonical_peft_receipt
            or peft_config_transition_receipt.get("requested_target_modules_sha256")
            != requested_peft_receipt["target_modules_sha256"]
            or peft_config_transition_receipt.get("canonical_target_modules_sha256")
            != canonical_peft_receipt["target_modules_sha256"]
            or type(peft_config_transition_receipt.get("lora_installation_digest"))
            is not str
            or not _SHA256.fullmatch(
                peft_config_transition_receipt["lora_installation_digest"]
            )
            or peft_config_transition_receipt.get("transition_digest")
            != object_sha256(transition_body)
        ):
            fail("Bernini PEFT requested/canonical transition closure differs")
        named = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in renderer.named_parameters()
                    if parameter.requires_grad
                ),
                key=lambda row: row[0],
            )
        )
        expected_lora_keys = {
            (block, attention, projection, side)
            for block in range(30)
            for attention in (1, 2)
            for projection in ("to_q", "to_k", "to_v", "to_out.0")
            for side in ("A", "B")
        }
        observed_lora_keys: set[tuple[int, int, str, str]] = set()
        for name, parameter in named:
            match = _LORA_NAME.fullmatch(name)
            if match is None:
                fail("Bernini exact480 LoRA parameter name differs")
            key = (
                int(match.group("block")),
                int(match.group("attention")),
                match.group("projection"),
                match.group("side"),
            )
            if key in observed_lora_keys:
                fail("Bernini exact480 LoRA parameter key is duplicated")
            observed_lora_keys.add(key)
            expected_shape = (
                (LORA_RANK, 1536)
                if match.group("side") == "A"
                else (1536, LORA_RANK)
            )
            if (
                tuple(int(item) for item in parameter.shape) != expected_shape
                or parameter.dtype != torch.float32
                or parameter.device != source_state.device
                or parameter.requires_grad is not True
            ):
                fail("Bernini exact480 LoRA tensor ABI differs")
        if (
            len(named) != LORA_TENSOR_COUNT
            or len({id(parameter) for _, parameter in named}) != len(named)
            or any(
                ".blocks." not in name
                or not any(marker in name for marker in (".lora_A.", ".lora_B."))
                for name, _ in named
            )
            or any(
                type(parameter) is not torch.nn.Parameter
                or parameter.device != source_state.device
                or not bool(torch.isfinite(parameter.detach()).all().item())
                for _, parameter in named
            )
            or len([name for name, _ in named if ".lora_A." in name])
            != LORA_AFFINES
            or len([name for name, _ in named if ".lora_B." in name])
            != LORA_AFFINES
            or sum(int(parameter.numel()) for _, parameter in named)
            != LORA_PARAMETER_COUNT
            or observed_lora_keys != expected_lora_keys
        ):
            fail("Bernini exact480 all-attention r256 LoRA registry differs")
        try:
            import bernini.models.wan_diffusion as vendor_apg
        except ImportError as error:
            raise TargetFreeBerniniRuntimeError(
                "official Bernini APG module is unavailable"
            ) from error
        if (
            vendor_apg.__name__ != "bernini.models.wan_diffusion"
            or not callable(getattr(vendor_apg, "normalized_guidance", None))
            or not callable(getattr(vendor_apg, "MomentumBuffer", None))
        ):
            fail("official Bernini APG ownership differs")
        self.renderer = renderer
        self.diffusion = diffusion
        self.transformer = transformer
        self._pristine_scheduler = copy.deepcopy(scheduler)
        self.source_state = source_state.detach().clone(memory_format=torch.contiguous_format)
        self.negative_condition = negative_condition.detach().clone(
            memory_format=torch.contiguous_format
        )
        self.positive_condition = positive_condition.detach().clone(
            memory_format=torch.contiguous_format
        )
        self.source_row_id = _safe_id(source_row_id, label="runtime source row")
        self.source_video_sha256 = _sha256(
            source_video_sha256, label="runtime source video SHA"
        )
        self.instruction_sha256 = _sha256(
            instruction_sha256, label="runtime instruction SHA"
        )
        self.base_model_sha256 = _sha256(
            base_model_sha256, label="runtime base-model SHA"
        )
        self.model_closure_sha256 = _sha256(
            model_closure_sha256, label="runtime model-closure SHA"
        )
        self.source_state_sha256 = tensor_sha256(self.source_state)
        self.negative_condition_sha256 = tensor_sha256(self.negative_condition)
        self.positive_condition_sha256 = tensor_sha256(self.positive_condition)
        self.parallel = parallel
        self.named_trainable_parameters = named
        self.trainable_inventory = tuple(
            {
                "name": name,
                "shape": [int(item) for item in parameter.shape],
                "dtype": str(parameter.dtype),
                "device_type": parameter.device.type,
                "requires_grad": parameter.requires_grad,
                "numel": int(parameter.numel()),
            }
            for name, parameter in named
        )
        self.trainable_inventory_sha256 = object_sha256(self.trainable_inventory)
        peft_receipt = {
            "schema_version": "bernini-full644-peft-three-layer-closure-v1",
            "requested_exact240": requested_peft_receipt,
            "canonical_postinstall_exact4": canonical_peft_receipt,
            "installed_exact480_trainable_inventory_sha256": self.trainable_inventory_sha256,
            "installed_exact480_lora_tensor_count": len(named),
            "lora_installation_digest": peft_config_transition_receipt[
                "lora_installation_digest"
            ],
            "transition_digest": peft_config_transition_receipt["transition_digest"],
        }
        self.peft_config_receipt = {
            **peft_receipt,
            "receipt_digest": object_sha256(peft_receipt),
        }
        self.device = source_state.device
        self.vendor_apg = vendor_apg
        self._owned_factory_token = _OWNED_RUNTIME_TOKEN
        self._schedule_authority = _load_schedule_authority_v1()
        self._behavior_policy_sha256: Optional[str] = None
        self._update_executed = False
        self.activation_checkpoint_blocks = self._install_activation_checkpointing_v1()
        renderer.eval()
        if any(
            isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
            for module in transformer.modules()
        ):
            fail("Bernini replay requires zero-dropout transformer execution")
        # Validate one fresh copy now.  Every record/replay obtains another
        # fresh copy, so UniPC multistep history never leaks across trajectories.
        self.fresh_scheduler()

    def _install_activation_checkpointing_v1(self) -> tuple[int, ...]:
        import torch
        from torch.utils.checkpoint import checkpoint

        blocks = getattr(self.transformer, "blocks", None)
        if (
            blocks is None
            or len(blocks) != 30
            or bool(getattr(self.transformer, "gradient_checkpointing", False))
        ):
            fail("selective activation checkpointing requires exact30 blanket-off blocks")
        for index in ACTIVATION_CHECKPOINT_BLOCKS:
            block = blocks[index]
            if getattr(block, "_full644_exact40_checkpoint_v1", False):
                fail("selective activation checkpointing was already installed")
            original = block.forward

            def context_fn() -> tuple[AbstractContextManager[Any], AbstractContextManager[Any]]:
                return nullcontext(), nullcontext()

            def checkpointed_forward(
                *args: Any, _original: Any = original, **kwargs: Any
            ) -> Any:
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return checkpoint(
                    _original,
                    *args,
                    use_reentrant=False,
                    context_fn=context_fn,
                    **kwargs,
                )

            block.forward = checkpointed_forward
            block._full644_exact40_checkpoint_v1 = True
        return ACTIVATION_CHECKPOINT_BLOCKS

    @property
    def dp_arm(self) -> int:
        return int(self.parallel.contract.arm_index)

    @property
    def sp_rank(self) -> int:
        return int(self.parallel.contract.sp_rank)

    def behavior_policy_sha256(self) -> str:
        if self._behavior_policy_sha256 is not None:
            return self._behavior_policy_sha256
        trainable_sha = self.trainable_parameter_sha256()
        result = self.policy_sha256_for_trainable_digest(trainable_sha)
        self._behavior_policy_sha256 = result
        return result

    def trainable_parameter_sha256(self) -> str:
        distributed_runtime = _require_held_local_module_v1("source_self_runtime")
        return distributed_runtime.trainable_parameters_digest(
            self.named_trainable_parameters
        )

    def policy_sha256_for_trainable_digest(self, trainable_sha256: str) -> str:
        trainable_sha = _sha256(
            trainable_sha256, label="runtime trainable-parameter SHA"
        )
        return object_sha256(
            {
                "schema_version": "bernini-full644-current-policy-closure-v1",
                "base_model_sha256": self.base_model_sha256,
                "model_closure_sha256": self.model_closure_sha256,
                "trainable_parameter_sha256": trainable_sha,
                "source_row_id": self.source_row_id,
                "source_video_sha256": self.source_video_sha256,
                "source_state_sha256": self.source_state_sha256,
                "instruction_sha256": self.instruction_sha256,
                "negative_condition_sha256": self.negative_condition_sha256,
                "positive_condition_sha256": self.positive_condition_sha256,
                "gaussian_kernel_sha256": GAUSSIAN_KERNEL_SHA256,
                "apg_guidance_sha256": APG_GUIDANCE_SHA256,
                "schedule_sha256": SCHEDULE_SHA256,
            }
        )

    def commit_updated_policy_digest_v1(self, trainable_sha256: str) -> str:
        """Invalidate the pre-step cache and bind it to the actual new bytes."""

        if self._update_executed is True:
            fail("one-source runtime already executed its exact one update")
        self._behavior_policy_sha256 = None
        result = self.policy_sha256_for_trainable_digest(trainable_sha256)
        self._behavior_policy_sha256 = result
        self._update_executed = True
        return result

    def fresh_scheduler(self) -> Any:
        scheduler = copy.deepcopy(self._pristine_scheduler)
        audit = self._schedule_authority.audit_runtime_unipc_schedule(
            scheduler, initialize=True
        )
        if (
            audit.get("schedule_sha256") != SCHEDULE_SHA256
            or getattr(scheduler, "step_index", None) is not None
            or tuple(getattr(scheduler, "timesteps", ()).shape) != (TRAJECTORY_STEPS,)
            or tuple(getattr(scheduler, "sigmas", ()).shape)
            != (TRAJECTORY_STEPS + 1,)
        ):
            fail("fresh UniPC exact40 schedule differs")
        return scheduler

    def coordinate(self, scheduler: Any, index: int) -> Exact40CoordinateV1:
        import torch

        if type(index) is not int or index not in range(TRAJECTORY_STEPS):
            fail("exact40 coordinate index differs")
        timestep_value = int(self._schedule_authority.PINNED_TIMESTEPS[index])
        sigma_hex = self._schedule_authority.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
        timestep = scheduler.timesteps[index : index + 1].to(self.device).contiguous()
        sigma = scheduler.sigmas[index].detach().to(dtype=torch.float32).reshape(())
        if (
            int(timestep.item()) != timestep_value
            or struct.pack(">f", float(sigma.item())).hex() != sigma_hex
        ):
            fail("fresh UniPC coordinate differs from pinned exact40 schedule")
        return Exact40CoordinateV1(index, timestep, sigma, timestep_value, sigma_hex)

    def keyed_noise(
        self, *, shape: Sequence[int], rollout_seed: int, purpose: str, index: int
    ) -> tuple[Any, Mapping[str, Any]]:
        import torch
        import torch.distributed as dist

        dimensions = tuple(int(item) for item in shape)
        if (
            not dimensions
            or any(item <= 0 for item in dimensions)
            or type(rollout_seed) is not int
            or not 0 <= rollout_seed < 2**63
            or type(purpose) is not str
            or not purpose.isascii()
            or type(index) is not int
            or index not in range(-1, TRAJECTORY_STEPS)
        ):
            fail("keyed rollout Gaussian request differs")
        key = {
            "schema_version": "bernini-full644-rollout-keyed-gaussian-v1",
            "source_video_sha256": self.source_video_sha256,
            "instruction_sha256": self.instruction_sha256,
            "behavior_policy_sha256": self.behavior_policy_sha256(),
            "rollout_seed": rollout_seed,
            "dp_arm": self.dp_arm,
            "purpose": purpose,
            "schedule_index": index,
            "shape": list(dimensions),
            "dtype": "torch.float32",
            "generator_device": "cpu",
        }
        key_sha = object_sha256(key)
        seed = int.from_bytes(bytes.fromhex(key_sha[:16]), "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(dimensions, generator=generator, dtype=torch.float32).to(
            self.device
        )
        leader = self.dp_arm * SP_SIZE
        dist.broadcast(value, src=leader, group=self.parallel.sp_group)
        local_sha = tensor_sha256(value)
        gathered: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(gathered, local_sha, group=self.parallel.sp_group)
        if gathered != [local_sha] * SP_SIZE:
            fail("SP4 keyed rollout Gaussian differs")
        return value.detach().contiguous(), {
            "key_sha256": key_sha,
            "tensor_sha256": local_sha,
            "derived_seed": seed,
            "sp4_leader_global_rank": leader,
            "sp4_broadcast_bit_identical": True,
        }

    def _pack(self, state: Any) -> _NativePackV1:
        import torch

        if state.shape != self.source_state.shape or state.requires_grad:
            fail("Bernini generated state geometry/graph differs")
        context: AbstractContextManager[Any] = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
        patched = []
        with torch.no_grad(), context:
            for source_id, value in ((1.0, self.source_state), (0.0, state)):
                result = self.transformer.patch_vae_latent(
                    hidden_states=value.to(dtype=torch.bfloat16), source_id=source_id
                )
                if not isinstance(result, tuple) or len(result) != 2:
                    fail("Bernini patch_vae_latent return ABI differs")
                patched.append((result[0].detach(), result[1].detach()))
        source_tokens, source_rotary = patched[0]
        generated_tokens, generated_rotary = patched[1]
        token_count = 21 * (int(state.shape[3]) // 2) * (int(state.shape[4]) // 2)
        if (
            tuple(source_tokens.shape) != (1, token_count, 1536)
            or source_tokens.shape != generated_tokens.shape
            or source_tokens.dtype != torch.bfloat16
            or generated_tokens.dtype != torch.bfloat16
            or tuple(source_rotary.shape[:3]) != (1, 1, token_count)
            or source_rotary.shape != generated_rotary.shape
        ):
            fail("Bernini native source/generated pack ABI differs")
        return _NativePackV1(
            torch.cat((source_tokens, generated_tokens), dim=1).detach().contiguous(),
            torch.cat((source_rotary, generated_rotary), dim=2).detach().contiguous(),
            token_count,
            token_count,
        )

    def _raw_forward(
        self, *, pack: _NativePackV1, coordinate: Exact40CoordinateV1, condition: Any
    ) -> Any:
        import torch

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw = self.diffusion.shared_step(
                model_id="transformer_1",
                noisy_latents=pack.visual,
                timesteps=coordinate.timestep,
                cond_embeds=condition,
                rotary_embs=pack.rotary,
                batch_vae_seqlen=[pack.source_tokens + pack.generated_tokens],
                batch_text_seqlen=[int(condition.shape[1])],
            )
        if (
            type(raw) is not torch.Tensor
            or tuple(raw.shape)
            != (1, pack.source_tokens + pack.generated_tokens, 64)
            or raw.dtype != torch.bfloat16
            or not bool(torch.isfinite(raw).all().item())
        ):
            fail("Bernini shared_step output ABI differs")
        return raw

    def _guided_mean_from_spatial(
        self,
        *,
        state: Any,
        coordinate: Exact40CoordinateV1,
        negative_spatial: Any,
        positive_spatial: Any,
    ) -> Any:
        clean_negative = state - coordinate.sigma * negative_spatial.float()
        clean_positive = state - coordinate.sigma * positive_spatial.float()
        momentum = self.vendor_apg.MomentumBuffer(
            float.fromhex(APG_GUIDANCE["momentum_float64_hex"])
        )
        guided = self.vendor_apg.normalized_guidance(
            pred_cond=clean_positive,
            pred_uncond=clean_negative,
            guidance_scale=float.fromhex(
                APG_GUIDANCE["guidance_scale_float64_hex"]
            ),
            momentum_buffer=momentum,
            eta=float.fromhex(APG_GUIDANCE["eta_float64_hex"]),
            norm_threshold=float.fromhex(
                APG_GUIDANCE["norm_threshold_float64_hex"]
            ),
        )
        mean = ((state - guided) / coordinate.sigma).float().contiguous()
        if mean.shape != state.shape:
            fail("Bernini guided current-policy mean geometry differs")
        return mean

    def policy_mean_no_grad(
        self, *, state: Any, coordinate: Exact40CoordinateV1
    ) -> Any:
        import torch

        pack = self._pack(state)
        with torch.no_grad():
            negative_raw = self._raw_forward(
                pack=pack, coordinate=coordinate, condition=self.negative_condition
            )
            positive_raw = self._raw_forward(
                pack=pack, coordinate=coordinate, condition=self.positive_condition
            )
            negative_spatial = _unpack_generated_velocity_v1(
                negative_raw[:, -pack.generated_tokens :, :],
                spatial_shape=state.shape,
            )
            positive_spatial = _unpack_generated_velocity_v1(
                positive_raw[:, -pack.generated_tokens :, :],
                spatial_shape=state.shape,
            )
            mean = self._guided_mean_from_spatial(
                state=state,
                coordinate=coordinate,
                negative_spatial=negative_spatial,
                positive_spatial=positive_spatial,
            ).detach()
        del pack, negative_raw, positive_raw, negative_spatial, positive_spatial
        return mean.contiguous()

    def backward_policy_mean_v1(
        self,
        *,
        state: Any,
        coordinate: Exact40CoordinateV1,
        mean_cotangent: Any,
        expected_mean_sha256: str,
    ) -> Mapping[str, Any]:
        """APG leaf VJP followed by two serial transformer backward passes."""

        import torch

        expected = _sha256(expected_mean_sha256, label="expected policy mean SHA")
        if (
            type(mean_cotangent) is not torch.Tensor
            or mean_cotangent.shape != state.shape
            or mean_cotangent.dtype != torch.float32
            or mean_cotangent.requires_grad
            or not bool(torch.isfinite(mean_cotangent).all().item())
        ):
            fail("policy-mean cotangent differs")
        pack = self._pack(state)
        with torch.no_grad():
            raw_detached = []
            for condition in (self.negative_condition, self.positive_condition):
                raw_detached.append(
                    self._raw_forward(
                        pack=pack, coordinate=coordinate, condition=condition
                    ).detach()
                )
        leaves = [value.requires_grad_(True) for value in raw_detached]
        spatial_leaves = [
            _unpack_generated_velocity_v1(
                value[:, -pack.generated_tokens :, :], spatial_shape=state.shape
            )
            for value in leaves
        ]
        leaf_mean = self._guided_mean_from_spatial(
            state=state,
            coordinate=coordinate,
            negative_spatial=spatial_leaves[0],
            positive_spatial=spatial_leaves[1],
        )
        if tensor_sha256(leaf_mean) != expected:
            fail("APG leaf replay changed current-policy mean")
        leaf_gradients = torch.autograd.grad(
            leaf_mean,
            leaves,
            grad_outputs=mean_cotangent,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        del leaf_mean, spatial_leaves, leaves
        branch_receipts = []
        for role, condition, expected_raw, cotangent in zip(
            ("negative", "positive"),
            (self.negative_condition, self.positive_condition),
            raw_detached,
            leaf_gradients,
        ):
            raw = self._raw_forward(
                pack=pack, coordinate=coordinate, condition=condition
            )
            if tensor_sha256(raw) != tensor_sha256(expected_raw):
                fail(f"Bernini {role} serial VJP raw replay differs")
            reference = weakref.ref(raw)
            torch.autograd.backward(raw, grad_tensors=cotangent.to(raw.dtype))
            branch_receipts.append(
                {
                    "role": role,
                    "raw_sha256": tensor_sha256(expected_raw),
                    "cotangent_sha256": tensor_sha256(cotangent),
                    "one_transformer_graph_live": True,
                }
            )
            del raw, expected_raw, cotangent
            gc.collect()
            if reference() is not None:
                fail(f"Bernini {role} transformer graph was not released")
        del pack, raw_detached, leaf_gradients
        return {
            "branch_order": ["negative", "positive"],
            "branches": branch_receipts,
            "serial_graph_release_verified": True,
        }

    def scheduler_step(
        self,
        *,
        scheduler: Any,
        coordinate: Exact40CoordinateV1,
        action: Any,
        state: Any,
    ) -> Any:
        import torch

        if action.requires_grad or state.requires_grad:
            fail("UniPC replay inputs must be detached")
        with torch.no_grad():
            result = scheduler.step(
                action,
                coordinate.timestep,
                state,
                return_dict=False,
            )
        if (
            not isinstance(result, (list, tuple))
            or len(result) < 1
            or type(result[0]) is not torch.Tensor
            or result[0].shape != state.shape
            or result[0].dtype != torch.float32
            or result[0].requires_grad
            or not bool(torch.isfinite(result[0]).all().item())
        ):
            fail("UniPC scheduler step output differs")
        cursor = getattr(scheduler, "step_index", None)
        if cursor is None:
            cursor = getattr(scheduler, "_step_index", None)
        if cursor != coordinate.index + 1:
            fail("UniPC multistep cursor did not advance exactly once")
        return result[0].detach().contiguous()


@dataclass(frozen=True)
class RecordedTrajectoryV1:
    receipt: TrajectoryReceiptV1
    terminal_state: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class DecodedRolloutV1:
    value: Mapping[str, Any]
    path: Path
    sha256: str
    trajectory: TrajectoryReceiptV1


def record_exact40_trajectory_v1(
    runtime: BerniniExact40PolicyV1,
    *,
    rollout_id: str,
    round_index: int,
    rollout_seed: int,
    output_directory: Path,
) -> RecordedTrajectoryV1:
    """Sample and seal one exact40 current-policy Gaussian trajectory."""

    import torch
    import torch.distributed as dist

    if type(runtime) is not BerniniExact40PolicyV1:
        fail("trajectory recording requires concrete BerniniExact40PolicyV1")
    rollout_name = _safe_id(rollout_id, label="rollout ID")
    if type(round_index) is not int or round_index < 0:
        fail("rollout round index differs")
    if type(rollout_seed) is not int or not 0 <= rollout_seed < 2**63:
        fail("rollout seed differs")
    if (
        not output_directory.is_absolute()
        or not output_directory.is_dir()
        or output_directory.is_symlink()
    ):
        fail("trajectory output directory differs")
    behavior_sha = runtime.behavior_policy_sha256()
    scheduler = runtime.fresh_scheduler()
    state, initial_noise = runtime.keyed_noise(
        shape=runtime.source_state.shape,
        rollout_seed=rollout_seed,
        purpose="initial_unipc_state",
        index=-1,
    )
    initial_sha = tensor_sha256(state)
    leader = runtime.dp_arm * SP_SIZE
    keep_payload = runtime.parallel.contract.rank == leader
    initial_cpu = state.detach().cpu() if keep_payload else None
    actions_cpu: list[Any] = []
    steps: list[Mapping[str, Any]] = []
    for index in range(TRAJECTORY_STEPS):
        coordinate = runtime.coordinate(scheduler, index)
        before_sha = tensor_sha256(state)
        mean = runtime.policy_mean_no_grad(state=state, coordinate=coordinate)
        mean_sha = tensor_sha256(mean)
        epsilon, noise = runtime.keyed_noise(
            shape=state.shape,
            rollout_seed=rollout_seed,
            purpose="executed_action_residual",
            index=index,
        )
        action = (mean + ACTION_STD * epsilon).float().contiguous().detach()
        action_sha = tensor_sha256(action)
        next_state = runtime.scheduler_step(
            scheduler=scheduler,
            coordinate=coordinate,
            action=action,
            state=state,
        )
        after_sha = tensor_sha256(next_state)
        step = {
            "schema_version": TRAJECTORY_STEP_SCHEMA,
            "schedule_index": index,
            "timestep": coordinate.timestep_value,
            "sigma_float32_be_hex": coordinate.sigma_float32_be_hex,
            "state_before_sha256": before_sha,
            "policy_mean_sha256": mean_sha,
            "action_noise_key_sha256": noise["key_sha256"],
            "action_noise_sha256": noise["tensor_sha256"],
            "executed_action_sha256": action_sha,
            "state_after_sha256": after_sha,
            "scheduler_step_index_after": index + 1,
        }
        step = {**step, "step_digest": object_sha256(step)}
        local_step_digest = step["step_digest"]
        gathered: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(
            gathered, local_step_digest, group=runtime.parallel.sp_group
        )
        if gathered != [local_step_digest] * SP_SIZE:
            fail("SP4 exact40 step receipt differs")
        steps.append(step)
        if keep_payload:
            actions_cpu.append(action.detach().cpu())
        del mean, epsilon, action, state
        state = next_state
    if len(steps) != TRAJECTORY_STEPS:
        fail("trajectory recorder did not execute exact40 steps")
    artifact_path = output_directory / f"{rollout_name}.trajectory.fp32"
    receipt_path = output_directory / f"{rollout_name}.trajectory.json"
    artifact_binding: list[Any] = [None]
    receipt_binding: list[Any] = [None]
    if keep_payload:
        if initial_cpu is None:
            fail("SP4 leader trajectory initial-state payload is unavailable")
        artifact_payload, _ = build_trajectory_artifact_v1(
            initial_state=initial_cpu, actions=actions_cpu
        )
        artifact_binding[0] = write_create_only(artifact_path, artifact_payload)
        del artifact_payload, actions_cpu, initial_cpu
        receipt = {
            "schema_version": TRAJECTORY_SCHEMA,
            "runtime_schema_version": SCHEMA_VERSION,
            "rollout_id": rollout_name,
            "source_row_id": runtime.source_row_id,
            "source_video_sha256": runtime.source_video_sha256,
            "instruction_sha256": runtime.instruction_sha256,
            "behavior_policy_sha256": behavior_sha,
            "round_index": round_index,
            "rollout_seed": rollout_seed,
            "dp_arm": runtime.dp_arm,
            "sp_size": SP_SIZE,
            "step_count": TRAJECTORY_STEPS,
            "latent_shape": [int(item) for item in state.shape],
            "latent_numel": int(state.numel()),
            "latent_dtype": "torch.float32",
            "schedule_sha256": SCHEDULE_SHA256,
            "gaussian_kernel": GAUSSIAN_KERNEL,
            "gaussian_kernel_sha256": GAUSSIAN_KERNEL_SHA256,
            "apg_guidance_sha256": APG_GUIDANCE_SHA256,
            "initial_noise_key_sha256": initial_noise["key_sha256"],
            "initial_state_sha256": initial_sha,
            "steps": steps,
            "terminal_state_sha256": tensor_sha256(state),
            "artifact_path": artifact_binding[0]["path"],
            "artifact_sha256": artifact_binding[0]["sha256"],
            "artifact_size_bytes": artifact_binding[0]["size_bytes"],
            "artifact_mode_octal": artifact_binding[0]["mode_octal"],
            "artifact_nlink": artifact_binding[0]["nlink"],
            "sp4_noise_broadcast": True,
            "sp4_step_consensus": True,
            "source_only_input": True,
            "paired_reference_read_count": 0,
            "external_velocity_read_count": 0,
        }
        receipt = {**receipt, "receipt_digest": object_sha256(receipt)}
        receipt_payload = canonical_json_bytes(receipt)
        receipt_binding[0] = write_create_only(receipt_path, receipt_payload)
    dist.broadcast_object_list(
        artifact_binding, src=leader, group=runtime.parallel.sp_group
    )
    dist.broadcast_object_list(
        receipt_binding, src=leader, group=runtime.parallel.sp_group
    )
    dist.barrier(group=runtime.parallel.sp_group)
    if (
        not isinstance(artifact_binding[0], Mapping)
        or not isinstance(receipt_binding[0], Mapping)
        or artifact_binding[0]["path"] != str(artifact_path)
        or receipt_binding[0]["path"] != str(receipt_path)
    ):
        fail("SP4 trajectory publication binding differs")
    loaded = load_trajectory_receipt_v1(
        receipt_path, expected_sha256=receipt_binding[0]["sha256"]
    )
    if loaded.value["behavior_policy_sha256"] != behavior_sha:
        fail("recorded trajectory behavior policy changed")
    return RecordedTrajectoryV1(loaded, state.detach().contiguous())


def decode_and_seal_recorded_trajectory_v1(
    runtime: BerniniExact40PolicyV1,
    recorded: RecordedTrajectoryV1,
    *,
    checkpoint_root: Path,
    vae_authority: Mapping[str, Any],
    output_directory: Path,
) -> DecodedRolloutV1:
    """Bind one terminal latent, exact81 MP4 and original trajectory immutably."""

    import torch
    import torch.distributed as dist

    if (
        type(runtime) is not BerniniExact40PolicyV1
        or type(recorded) is not RecordedTrajectoryV1
        or not output_directory.is_absolute()
        or not output_directory.is_dir()
        or output_directory.is_symlink()
    ):
        fail("decoded rollout input boundary differs")
    authority = _closed(vae_authority, _VAE_AUTHORITY_FIELDS, label="owned VAE authority")
    if (
        authority["schema_version"] != "bernini-full644-owned-vae-authority-v1"
        or authority["base_checkpoint_tree_sha256"] != BASE_CHECKPOINT_TREE_SHA256
        or authority["checkpoint_content_manifest_sha256"]
        != BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("owned VAE authority differs")
    row = recorded.receipt.value
    if (
        tensor_sha256(recorded.terminal_state) != row["terminal_state_sha256"]
        or row["dp_arm"] != runtime.dp_arm
        or row["behavior_policy_sha256"] != runtime.behavior_policy_sha256()
    ):
        fail("terminal state/trajectory/runtime join differs")
    leader = runtime.dp_arm * SP_SIZE
    is_leader = runtime.parallel.contract.rank == leader
    publication: list[Any] = [None]
    if is_leader:
        from bernini.io_utils import save_output
        from bernini.pipeline import _vae_decode
        from diffusers.models import AutoencoderKLWan
        from safetensors.torch import load_file as load_safetensors_file
        from safetensors.torch import save as save_safetensors_bytes

        stem = row["rollout_id"]
        latent_path = output_directory / f"{stem}.terminal.safetensors"
        candidate_path = output_directory / f"{stem}.candidate.mp4"
        wrapper_path = output_directory / f"{stem}.decoded-rollout.json"
        terminal_cpu = recorded.terminal_state.detach().cpu().float().contiguous()
        latent_payload = save_safetensors_bytes(
            {"normalized_clean_latent": terminal_cpu},
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "source": "stochastic_exact40_terminal",
            },
        )
        latent_binding = write_create_only(latent_path, latent_payload)
        restored = load_safetensors_file(str(latent_path), device="cpu")
        if (
            set(restored) != {"normalized_clean_latent"}
            or tensor_sha256(restored["normalized_clean_latent"])
            != row["terminal_state_sha256"]
        ):
            fail("normalized terminal latent roundtrip differs")
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint_root),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False).to(runtime.device)
        try:
            before_terminal = tensor_sha256(recorded.terminal_state)
            with torch.no_grad():
                decoded = _vae_decode(vae, recorded.terminal_state)
            if (
                tuple(decoded.shape[:1]) != (FRAME_COUNT,)
                or decoded.ndim != 4
                or int(decoded.shape[-1]) != 3
                or tensor_sha256(recorded.terminal_state) != before_terminal
            ):
                fail("owned VAE terminal decode ABI differs")
            temporary = output_directory / f".{stem}.candidate.tmp.mp4"
            if temporary.exists() or temporary.is_symlink():
                fail("candidate temporary path already exists")
            save_output(decoded, str(temporary), fps=25)
            raw_media = read_stable_file(
                temporary,
                expected_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                label="fresh candidate temporary media",
            )
            os.unlink(temporary)
            media_binding = write_create_only(candidate_path, raw_media)
        finally:
            # Never rematerialize the large VAE on host RAM.  Two DP leaders
            # decode concurrently, so release their GPU objects in place.
            vae = None
            gc.collect()
            _trim_host_allocator_v1()
            torch.cuda.empty_cache()
        # Decode the exact held candidate bytes that Qwen will consume.  The
        # per-frame tree binds more than codec metadata or a pathname hash.
        with tempfile.TemporaryDirectory(prefix="full644-candidate-probe-") as temporary_root:
            probe_path = Path(temporary_root) / "candidate.mp4"
            write_create_only(probe_path, raw_media, mode=0o400)
            try:
                import decord
                reader = decord.VideoReader(
                    str(probe_path), num_threads=1, ctx=decord.cpu(0)
                )
                frame_count = len(reader)
                reported_fps = float(reader.get_avg_fps())
                frames = reader.get_batch(list(range(FRAME_COUNT))).asnumpy()
            except Exception as error:
                raise TargetFreeBerniniRuntimeError(
                    f"candidate exact81 probe failed: {error}"
                ) from error
        if (
            frame_count != FRAME_COUNT
            or not math.isfinite(reported_fps)
            or abs(reported_fps - FPS) > 1.0e-3
            or frames.ndim != 4
            or frames.shape[0] != FRAME_COUNT
            or frames.shape[-1] != 3
        ):
            fail("candidate exact81/25fps full decode differs")
        frame_hashes = [
            hashlib.sha256(frame.tobytes(order="C")).hexdigest() for frame in frames
        ]
        height, width = int(frames.shape[1]), int(frames.shape[2])
        wrapper = {
            "schema_version": DECODED_ROLLOUT_SCHEMA,
            "rollout_id": row["rollout_id"],
            "behavior_policy_sha256": row["behavior_policy_sha256"],
            "round_index": row["round_index"],
            "rollout_seed": row["rollout_seed"],
            "dp_arm": row["dp_arm"],
            "source_row_id": row["source_row_id"],
            "source_video_sha256": row["source_video_sha256"],
            "instruction_sha256": row["instruction_sha256"],
            "trajectory_receipt_path": str(recorded.receipt.path),
            "trajectory_receipt_sha256": recorded.receipt.sha256,
            "trajectory_receipt_digest": row["receipt_digest"],
            "trajectory_artifact_path": row["artifact_path"],
            "trajectory_artifact_sha256": row["artifact_sha256"],
            "trajectory_artifact_size_bytes": row["artifact_size_bytes"],
            "terminal_state_sha256": row["terminal_state_sha256"],
            "normalized_latent_path": str(latent_path),
            "normalized_latent_sha256": latent_binding["sha256"],
            "normalized_latent_tensor_sha256": row["terminal_state_sha256"],
            "candidate_media_path": str(candidate_path),
            "candidate_media_sha256": media_binding["sha256"],
            "candidate_media_size_bytes": media_binding["size_bytes"],
            "candidate_frame_count": FRAME_COUNT,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "width": width,
            "height": height,
            "full_decode_frame_sha256": frame_hashes,
            "full_decode_tree_digest": object_sha256(frame_hashes),
            "vae_authority": dict(authority),
            "source_encode_and_terminal_decode_same_vae_authority": True,
            "target_media_read_count": 0,
        }
        wrapper = {**wrapper, "receipt_digest": object_sha256(wrapper)}
        wrapper_binding = write_create_only(
            wrapper_path, canonical_json_bytes(wrapper)
        )
        publication[0] = {
            "path": str(wrapper_path),
            "sha256": wrapper_binding["sha256"],
            "candidate_media_path": str(candidate_path),
            "candidate_media_sha256": media_binding["sha256"],
        }
    dist.broadcast_object_list(publication, src=leader, group=runtime.parallel.sp_group)
    dist.barrier(group=runtime.parallel.sp_group)
    if not isinstance(publication[0], Mapping):
        fail("SP4 decoded rollout publication differs")
    decoded, trajectory = load_decoded_rollout_receipt_v1(
        Path(publication[0]["path"]), expected_sha256=publication[0]["sha256"]
    )
    if trajectory.sha256 != recorded.receipt.sha256:
        fail("SP4 decoded rollout changed trajectory binding")
    return DecodedRolloutV1(
        decoded,
        Path(publication[0]["path"]),
        publication[0]["sha256"],
        trajectory,
    )


def normalized_gaussian_step_logprob_v1(action: Any, mean: Any) -> Any:
    import torch

    if (
        type(action) is not torch.Tensor
        or type(mean) is not torch.Tensor
        or action.shape != mean.shape
        or action.dtype != torch.float32
        or mean.dtype != torch.float32
        or action.device != mean.device
        or action.requires_grad
        or mean.requires_grad
        or not bool(torch.isfinite(action).all().item())
        or not bool(torch.isfinite(mean).all().item())
    ):
        fail("Gaussian log-probability input ABI differs")
    residual = (action - mean) / ACTION_STD
    result = (
        -0.5 * residual.square().mean()
        - math.log(ACTION_STD)
        - 0.5 * math.log(2.0 * math.pi)
    )
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        fail("normalized Gaussian step log-probability is non-finite")
    return result.float()


def preference_coefficients_v1(
    chosen_logprob_sum: float, rejected_logprob_sum: float
) -> Mapping[str, float]:
    for label, value in (
        ("chosen log-probability", chosen_logprob_sum),
        ("rejected log-probability", rejected_logprob_sum),
    ):
        if type(value) is not float or not math.isfinite(value):
            fail(f"{label} differs")
    margin = chosen_logprob_sum - rejected_logprob_sum
    argument = -PREFERENCE_BETA * margin
    if argument >= 0.0:
        exponential = math.exp(-argument)
        sigmoid = 1.0 / (1.0 + exponential)
    else:
        exponential = math.exp(argument)
        sigmoid = exponential / (1.0 + exponential)
    loss = math.log1p(math.exp(argument)) if argument <= 40.0 else argument
    chosen = -PREFERENCE_BETA * sigmoid
    rejected = PREFERENCE_BETA * sigmoid
    return {
        "loss": float(loss),
        "margin": float(margin),
        "chosen_dloss_d_logprob_sum": float(chosen),
        "rejected_dloss_d_logprob_sum": float(rejected),
        # Each DP arm owns one endpoint, after which DP2 averages gradients.
        # Multiplying locally by two exactly cancels that collective mean.
        "chosen_local_coefficient_after_dp2_compensation": float(DP_SIZE * chosen),
        "rejected_local_coefficient_after_dp2_compensation": float(DP_SIZE * rejected),
    }


def endpoint_roles_by_dp_arm_v1(
    chosen_dp_arm: int, rejected_dp_arm: int
) -> Mapping[int, str]:
    if (
        type(chosen_dp_arm) is not int
        or type(rejected_dp_arm) is not int
        or {chosen_dp_arm, rejected_dp_arm} != set(range(DP_SIZE))
    ):
        fail("chosen/rejected trajectories must occupy distinct DP arms")
    return {chosen_dp_arm: "chosen", rejected_dp_arm: "rejected"}


def gaussian_mean_cotangent_v1(
    *, action: Any, mean: Any, trajectory_coefficient: float
) -> Any:
    import torch

    if (
        type(trajectory_coefficient) is not float
        or not math.isfinite(trajectory_coefficient)
    ):
        fail("trajectory log-probability coefficient differs")
    # d mean[(a-mu)^2] / d mu = -2(a-mu)/N; the -1/2 from
    # log Normal cancels it, yielding (a-mu)/(std^2*N).
    result = (
        trajectory_coefficient
        * (action - mean)
        / (ACTION_STD * ACTION_STD * float(action.numel()))
    ).float().contiguous().detach()
    if result.shape != mean.shape or not bool(torch.isfinite(result).all().item()):
        fail("Gaussian policy-mean cotangent differs")
    return result


def replay_trajectory_pass1_v1(
    runtime: BerniniExact40PolicyV1, receipt: TrajectoryReceiptV1
) -> Mapping[str, Any]:
    """No-grad streaming replay with one fresh stateful UniPC scheduler."""

    if type(runtime) is not BerniniExact40PolicyV1 or type(receipt) is not TrajectoryReceiptV1:
        fail("pass1 requires concrete runtime and loaded trajectory")
    row = receipt.value
    if (
        row["source_row_id"] != runtime.source_row_id
        or row["source_video_sha256"] != runtime.source_video_sha256
        or row["instruction_sha256"] != runtime.instruction_sha256
        or row["behavior_policy_sha256"] != runtime.behavior_policy_sha256()
        or row["dp_arm"] != runtime.dp_arm
    ):
        fail("trajectory is not on-policy for this source/instruction/DP arm")
    scheduler = runtime.fresh_scheduler()
    step_logprobs: list[float] = []
    mean_hashes: list[str] = []
    with TrajectoryArtifactReaderV1(
        receipt.artifact_path, expected_sha256=receipt.artifact_sha256
    ) as reader:
        state = reader.tensor("initial_state", device=runtime.device)
        regenerated_initial, initial_noise = runtime.keyed_noise(
            shape=state.shape,
            rollout_seed=row["rollout_seed"],
            purpose="initial_unipc_state",
            index=-1,
        )
        if (
            tensor_sha256(state) != row["initial_state_sha256"]
            or initial_noise["key_sha256"] != row["initial_noise_key_sha256"]
            or not bool((state == regenerated_initial).all().item())
        ):
            fail("pass1 initial state differs")
        del regenerated_initial
        for index in range(TRAJECTORY_STEPS):
            step = row["steps"][index]
            if tensor_sha256(state) != step["state_before_sha256"]:
                fail("pass1 UniPC state chain differs before step")
            coordinate = runtime.coordinate(scheduler, index)
            if (
                coordinate.timestep_value != step["timestep"]
                or coordinate.sigma_float32_be_hex != step["sigma_float32_be_hex"]
            ):
                fail("pass1 exact40 coordinate/receipt join differs")
            mean = runtime.policy_mean_no_grad(state=state, coordinate=coordinate)
            mean_sha = tensor_sha256(mean)
            if mean_sha != step["policy_mean_sha256"]:
                fail("pass1 current-policy mean differs from recorded behavior policy")
            action = reader.tensor(f"action_{index:02d}", device=runtime.device)
            epsilon, noise = runtime.keyed_noise(
                shape=state.shape,
                rollout_seed=row["rollout_seed"],
                purpose="executed_action_residual",
                index=index,
            )
            expected_action = (mean + ACTION_STD * epsilon).float().contiguous()
            if (
                noise["key_sha256"] != step["action_noise_key_sha256"]
                or noise["tensor_sha256"] != step["action_noise_sha256"]
                or tensor_sha256(action) != step["executed_action_sha256"]
                or tensor_sha256(expected_action) != step["executed_action_sha256"]
                or not bool((action == expected_action).all().item())
            ):
                fail("pass1 executed action differs")
            logprob = normalized_gaussian_step_logprob_v1(action, mean)
            step_logprobs.append(float(logprob.item()))
            mean_hashes.append(mean_sha)
            next_state = runtime.scheduler_step(
                scheduler=scheduler,
                coordinate=coordinate,
                action=action,
                state=state,
            )
            if tensor_sha256(next_state) != step["state_after_sha256"]:
                fail("pass1 fresh UniPC replay state differs")
            del state, action, mean, logprob, epsilon, expected_action
            state = next_state
        reader.assert_stable()
    if tensor_sha256(state) != row["terminal_state_sha256"]:
        fail("pass1 terminal state differs")
    return {
        "step_count": TRAJECTORY_STEPS,
        "step_logprob_float64_hex": [float(value).hex() for value in step_logprobs],
        "logprob_sum": float(sum(step_logprobs)),
        "mean_sha256": mean_hashes,
        "fresh_stateful_unipc_replay": True,
        "artifact_retained_across_full_pass": True,
        "score_semantics": "dimension_normalized_gaussian_engineering_score_not_joint_log_probability",
        "score_reduction": GAUSSIAN_SCORE_REDUCTION,
        "latent_numel": row["latent_numel"],
    }


def replay_trajectory_pass2_backward_v1(
    runtime: BerniniExact40PolicyV1,
    receipt: TrajectoryReceiptV1,
    *,
    local_trajectory_coefficient: float,
) -> Mapping[str, Any]:
    """Streaming exact40 VJP replay; one transformer graph lives at a time."""

    if type(runtime) is not BerniniExact40PolicyV1 or type(receipt) is not TrajectoryReceiptV1:
        fail("pass2 requires concrete runtime and loaded trajectory")
    row = receipt.value
    scheduler = runtime.fresh_scheduler()
    branch_digests = []
    with TrajectoryArtifactReaderV1(
        receipt.artifact_path, expected_sha256=receipt.artifact_sha256
    ) as reader:
        state = reader.tensor("initial_state", device=runtime.device)
        regenerated_initial, initial_noise = runtime.keyed_noise(
            shape=state.shape,
            rollout_seed=row["rollout_seed"],
            purpose="initial_unipc_state",
            index=-1,
        )
        if (
            tensor_sha256(state) != row["initial_state_sha256"]
            or initial_noise["key_sha256"] != row["initial_noise_key_sha256"]
            or not bool((state == regenerated_initial).all().item())
        ):
            fail("pass2 initial state differs")
        del regenerated_initial
        for index in range(TRAJECTORY_STEPS):
            step = row["steps"][index]
            if tensor_sha256(state) != step["state_before_sha256"]:
                fail("pass2 UniPC state chain differs before step")
            coordinate = runtime.coordinate(scheduler, index)
            if (
                coordinate.timestep_value != step["timestep"]
                or coordinate.sigma_float32_be_hex != step["sigma_float32_be_hex"]
            ):
                fail("pass2 exact40 coordinate/receipt join differs")
            mean = runtime.policy_mean_no_grad(state=state, coordinate=coordinate)
            mean_sha = tensor_sha256(mean)
            if mean_sha != step["policy_mean_sha256"]:
                fail("pass2 current-policy mean differs")
            action = reader.tensor(f"action_{index:02d}", device=runtime.device)
            epsilon, noise = runtime.keyed_noise(
                shape=state.shape,
                rollout_seed=row["rollout_seed"],
                purpose="executed_action_residual",
                index=index,
            )
            expected_action = (mean + ACTION_STD * epsilon).float().contiguous()
            if (
                noise["key_sha256"] != step["action_noise_key_sha256"]
                or noise["tensor_sha256"] != step["action_noise_sha256"]
                or tensor_sha256(action) != step["executed_action_sha256"]
                or tensor_sha256(expected_action) != step["executed_action_sha256"]
                or not bool((action == expected_action).all().item())
            ):
                fail("pass2 executed action/kernel reconstruction differs")
            cotangent = gaussian_mean_cotangent_v1(
                action=action,
                mean=mean,
                trajectory_coefficient=local_trajectory_coefficient,
            )
            branch = runtime.backward_policy_mean_v1(
                state=state,
                coordinate=coordinate,
                mean_cotangent=cotangent,
                expected_mean_sha256=mean_sha,
            )
            branch_digests.append(object_sha256(branch))
            next_state = runtime.scheduler_step(
                scheduler=scheduler,
                coordinate=coordinate,
                action=action,
                state=state,
            )
            if tensor_sha256(next_state) != step["state_after_sha256"]:
                fail("pass2 fresh UniPC replay state differs")
            del state, action, mean, cotangent, branch, epsilon, expected_action
            state = next_state
        reader.assert_stable()
    if tensor_sha256(state) != row["terminal_state_sha256"]:
        fail("pass2 terminal state differs")
    return {
        "step_count": TRAJECTORY_STEPS,
        "local_trajectory_coefficient_float64_hex": local_trajectory_coefficient.hex(),
        "per_step_serial_vjp_digest": branch_digests,
        "fresh_stateful_unipc_replay": True,
        "artifact_retained_across_full_pass": True,
        "one_transformer_graph_live": True,
        "per_step_graph_released": True,
    }


def _load_preference_core_v1() -> Any:
    return _require_held_local_module_v1("full644_target_free_preference_v1")


def _trajectory_for_rollout_v1(
    runtime: BerniniExact40PolicyV1,
    rollout: Any,
    *,
    source: Any,
    expected_decoded_rollout_sha256: str,
    require_local_arm: bool = True,
) -> TrajectoryReceiptV1:
    receipt = load_trajectory_receipt_v1(
        rollout.trajectory_receipt_path,
        expected_sha256=rollout.trajectory_receipt_sha256,
    )
    suffix = ".trajectory.json"
    if not rollout.trajectory_receipt_path.name.endswith(suffix):
        fail("trajectory receipt basename cannot derive decoded rollout receipt")
    decoded_path = rollout.trajectory_receipt_path.with_name(
        rollout.trajectory_receipt_path.name[: -len(suffix)] + ".decoded-rollout.json"
    )
    decoded, decoded_trajectory = load_decoded_rollout_receipt_v1(
        decoded_path,
        expected_sha256=_sha256(
            expected_decoded_rollout_sha256,
            label="expected decoded rollout receipt SHA",
        ),
    )
    row = receipt.value
    if (
        decoded["candidate_media_path"] != str(rollout.output_media_path)
        or decoded["candidate_media_sha256"] != rollout.output_media_sha256
        or decoded["rollout_id"] != rollout.rollout_id
        or decoded["behavior_policy_sha256"] != rollout.policy_sha256
        or decoded["round_index"] != rollout.round_index
        or decoded["rollout_seed"] != rollout.seed
        or decoded_trajectory.sha256 != receipt.sha256
        or row["rollout_id"] != rollout.rollout_id
        or row["source_row_id"] != source.row_id
        or row["source_video_sha256"] != source.source_video_sha256
        or row["instruction_sha256"] != source.instruction_sha256
        or row["behavior_policy_sha256"] != rollout.policy_sha256
        or row["round_index"] != rollout.round_index
        or row["rollout_seed"] != rollout.seed
        or (require_local_arm and row["dp_arm"] != runtime.dp_arm)
    ):
        fail("rollout manifest/trajectory/runtime join differs")
    return receipt


def _optimizer_projection_v1(value: Any) -> Any:
    """Canonical, type-preserving projection for AdamW save/reload equality."""

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and type(value) is torch.Tensor:
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "sha256": tensor_sha256(value),
        }
    if value is None or type(value) in (bool, int, str):
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            fail("optimizer state contains non-finite float")
        return {"kind": "float", "value_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if type(value) is tuple else "list",
            "items": [_optimizer_projection_v1(item) for item in value],
        }
    if isinstance(value, Mapping):
        rows = [
            {
                "key": _optimizer_projection_v1(key),
                "value": _optimizer_projection_v1(item),
            }
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: canonical_json_bytes(row["key"]))
        return {"kind": "mapping", "items": rows}
    fail("optimizer state contains unsupported value")


def _save_reload_exact_one_checkpoint_v1(
    runtime: BerniniExact40PolicyV1,
    optimizer: Any,
    *,
    policy_digest_before: str,
    policy_digest_after: str,
    trainable_digest_before: str,
    trainable_digest_after: str,
) -> Mapping[str, Any]:
    import torch
    import torch.distributed as dist

    parent = getattr(runtime, "_owned_output_root", None)
    if type(parent) is not Path or not parent.is_absolute() or not parent.is_dir():
        fail("owned checkpoint output root is unavailable")
    final = parent / "checkpoint-step1"
    staging = parent / ".checkpoint-step1-staging"
    publication: list[Any] = [None]
    if runtime.parallel.contract.rank == 0:
        try:
            if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
                fail("exact-one checkpoint path is not fresh")
            os.mkdir(staging, 0o700)
            adapter_temporary = staging / ".adapter.tmp.safetensors"
            adapter_path = staging / "adapter.safetensors"
            optimizer_temporary = staging / ".optimizer.tmp.pt"
            optimizer_path = staging / "optimizer.pt"
            metadata_path = staging / "metadata.json"
            from safetensors.torch import load_file as load_safetensors_file
            from safetensors.torch import save_file as save_safetensors_file

            current = {
                name: parameter.detach().to(device="cpu").contiguous()
                for name, parameter in runtime.named_trainable_parameters
            }
            save_safetensors_file(current, str(adapter_temporary))
            os.chmod(adapter_temporary, 0o444)
            os.rename(adapter_temporary, adapter_path)
            adapter_binding = _stable_file_binding_v1(
                adapter_path, label="exact-one adapter", expected_mode=0o444
            )
            loaded_adapter = load_safetensors_file(str(adapter_path), device="cpu")
            if list(sorted(loaded_adapter)) != [
                name for name, _ in runtime.named_trainable_parameters
            ]:
                fail("exact-one adapter reload name inventory differs")
            for name, parameter in runtime.named_trainable_parameters:
                loaded = loaded_adapter[name]
                if (
                    tuple(loaded.shape) != tuple(parameter.shape)
                    or loaded.dtype != parameter.dtype
                    or tensor_sha256(loaded) != tensor_sha256(parameter)
                ):
                    fail("exact-one adapter reload tensor differs")

            optimizer_state = optimizer.state_dict()
            optimizer_digest = object_sha256(_optimizer_projection_v1(optimizer_state))
            torch.save(optimizer_state, optimizer_temporary)
            os.chmod(optimizer_temporary, 0o444)
            os.rename(optimizer_temporary, optimizer_path)
            optimizer_binding = _stable_file_binding_v1(
                optimizer_path, label="exact-one optimizer", expected_mode=0o444
            )
            reloaded_optimizer = torch.load(
                optimizer_path, map_location="cpu", weights_only=True
            )
            if object_sha256(_optimizer_projection_v1(reloaded_optimizer)) != optimizer_digest:
                fail("exact-one optimizer reload differs")
            metadata = {
                "schema_version": "bernini-full644-target-free-checkpoint-v1",
                "step": 1,
                "trainable_tensor_count": LORA_TENSOR_COUNT,
                "trainable_inventory_sha256": runtime.trainable_inventory_sha256,
                "trainable_parameter_digest_before": trainable_digest_before,
                "trainable_parameter_digest_after": trainable_digest_after,
                "policy_digest_before": policy_digest_before,
                "policy_digest_after": policy_digest_after,
                "adapter_sha256": adapter_binding["sha256"],
                "optimizer_sha256": optimizer_binding["sha256"],
                "optimizer_state_digest": optimizer_digest,
                "world8_policy_consensus_completed_before_publication": True,
                "consumption_requires_one_source_update_stage_receipt": True,
                "engineering_only": True,
                "scientific_result_claimed": False,
            }
            metadata = {**metadata, "metadata_digest": object_sha256(metadata)}
            metadata_binding = write_create_only(
                metadata_path, canonical_json_bytes(metadata)
            )
            rows = [adapter_binding, optimizer_binding, metadata_binding]
            for path in (adapter_path, optimizer_path, metadata_path):
                if stat.S_IMODE(os.lstat(path).st_mode) != 0o444:
                    fail("exact-one checkpoint member mode differs")
            os.chmod(staging, 0o555)
            os.rename(staging, final)
            publication[0] = {
                "schema_version": "bernini-full644-target-free-checkpoint-binding-v1",
                "path": str(final),
                "file_count": 3,
                "files": [
                    {**dict(row), "path": Path(row["path"]).name} for row in rows
                ],
                "tree_digest": object_sha256(
                    [{**dict(row), "path": Path(row["path"]).name} for row in rows]
                ),
                "adapter_reload_exact480_verified": True,
                "optimizer_weights_only_reload_verified": True,
                "create_only_fresh_directory": True,
                "world8_policy_consensus_completed_before_publication": True,
                "consumption_requires_one_source_update_stage_receipt": True,
            }
        except Exception as error:
            publication[0] = {
                "schema_version": "bernini-full644-checkpoint-failure-v1",
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
    dist.broadcast_object_list(publication, src=0)
    row = publication[0]
    if not isinstance(row, Mapping) or row.get("ok") is False:
        fail(f"exact-one checkpoint save/reload failed: {row}")
    checkpoint_root = Path(row["path"])
    for member in row["files"]:
        observed = _stable_file_binding_v1(
            checkpoint_root / member["path"],
            label=f"published checkpoint {member['path']}",
            expected_mode=0o444,
        )
        if (
            observed["sha256"] != member["sha256"]
            or observed["size_bytes"] != member["size_bytes"]
        ):
            fail("published exact-one checkpoint member differs")
    # Every rank reloads the actual adapter artifact it would hand to inference.
    from safetensors.torch import load_file as load_safetensors_file
    loaded = load_safetensors_file(str(checkpoint_root / "adapter.safetensors"), device="cpu")
    for name, parameter in runtime.named_trainable_parameters:
        if tensor_sha256(loaded[name]) != tensor_sha256(parameter):
            fail("WORLD8 published adapter reload differs from live policy")
    return row


def _zero_update_receipt_v1(
    runtime: BerniniExact40PolicyV1,
    *,
    preference_set: Any,
    status: str,
    admission: Optional[Mapping[str, Any]],
    trainable_digest_before: str,
) -> Mapping[str, Any]:
    if status not in (
        "ZERO_UPDATE_NO_PREFERENCE_PAIR",
        "ZERO_UPDATE_VERIFIER_UNDETERMINED_OR_INELIGIBLE",
    ):
        fail("zero-update status differs")
    before = runtime.behavior_policy_sha256()
    trainable_before = _sha256(
        trainable_digest_before, label="zero-update trainable digest"
    )
    distributed_runtime = _require_held_local_module_v1("source_self_runtime")
    trainable_after = distributed_runtime.parameter_consensus(
        runtime.named_trainable_parameters,
        runtime.parallel.world_group,
        "full644 target-free zero-update postbranch parameters",
        expected_count=WORLD_SIZE,
    )
    policy_after = runtime.policy_sha256_for_trainable_digest(trainable_after)
    if trainable_after != trainable_before or policy_after != before:
        fail("zero-update branch changed current-policy bytes")
    result = {
        "schema_version": UPDATE_SCHEMA,
        "status": status,
        "training_mode": "TARGET_FREE_ON_POLICY_PREFERENCE",
        "source_row_id": runtime.source_row_id,
        "source_video_sha256": runtime.source_video_sha256,
        "instruction_sha256": runtime.instruction_sha256,
        "preference_set_sha256": preference_set.preference_set_sha256,
        "preference_set_digest": preference_set.preference_set_digest,
        "behavior_policy_sha256": before,
        "pair_count": len(preference_set.pairs),
        "engineering_verifier_admission": admission,
        "manifest_axis_pass_used_for_admission": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "trainable_parameter_digest_before": trainable_before,
        "trainable_parameter_digest_after": trainable_after,
        "policy_digest_before": before,
        "policy_digest_after": policy_after,
        "zero_update_fresh_postbranch_world8_rehash": True,
        "source_only_input": True,
        "paired_reference_read_count": 0,
        "external_velocity_read_count": 0,
        "engineering_only": True,
        "scientific_result_claimed": False,
    }
    return {**result, "receipt_digest": object_sha256(result)}


def _engineering_one_update_loaded_v1(
    runtime: BerniniExact40PolicyV1,
    *,
    preference_set: Any,
    verifier_authority: EngineeringVerifierAuthorityV1,
    learning_rate: float,
    max_gradient_norm: float,
) -> Mapping[str, Any]:
    """Execute exact one on-policy update, or a genuine verifier zero-update."""

    import torch
    import torch.distributed as dist

    core = _load_preference_core_v1()
    if (
        type(runtime) is not BerniniExact40PolicyV1
        or getattr(runtime, "_owned_factory_token", None) is not _OWNED_RUNTIME_TOKEN
    ):
        fail("one-update requires concrete BerniniExact40PolicyV1")
    if type(preference_set) is not core.PreferenceSetV1:
        fail("one-update requires one closed full644 preference-set loader result")
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or type(max_gradient_norm) is not float
        or not math.isfinite(max_gradient_norm)
        or max_gradient_norm <= 0.0
    ):
        fail("one-update optimizer constants differ")
    if runtime._update_executed is True:
        fail("one-source runtime already executed its exact one update")
    if verifier_authority != _frozen_qwen_authority_v1():
        fail("one-update verifier authority is not the frozen Qwen release")
    expected_qualification = _expected_qwen_qualification_v1(verifier_authority)
    if dict(preference_set.verifier_qualification) != expected_qualification:
        fail("preference-set Qwen qualification authority differs")
    distributed_runtime = _require_held_local_module_v1("source_self_runtime")
    trainable_digest_before = distributed_runtime.parameter_consensus(
        runtime.named_trainable_parameters,
        runtime.parallel.world_group,
        "full644 target-free initial trainable parameters",
        expected_count=WORLD_SIZE,
    )
    behavior_sha = runtime.behavior_policy_sha256()
    if (
        runtime.policy_sha256_for_trainable_digest(trainable_digest_before)
        != behavior_sha
    ):
        fail("initial trainable/current-policy digest join differs")
    if (
        behavior_sha != preference_set.behavior_policy_sha256
        or len(preference_set.pairs) not in (0, 1)
    ):
        fail("one-source preference/current-policy closure differs")
    input_closure = {
        "schema_version": "bernini-full644-world8-update-input-closure-v1",
        "source_row_id": runtime.source_row_id,
        "source_video_sha256": runtime.source_video_sha256,
        "source_state_sha256": runtime.source_state_sha256,
        "instruction_sha256": runtime.instruction_sha256,
        "negative_condition_sha256": runtime.negative_condition_sha256,
        "positive_condition_sha256": runtime.positive_condition_sha256,
        "base_model_sha256": runtime.base_model_sha256,
        "model_closure_sha256": runtime.model_closure_sha256,
        "trainable_inventory_sha256": runtime.trainable_inventory_sha256,
        "trainable_parameter_digest_before": trainable_digest_before,
        "behavior_policy_sha256": behavior_sha,
        "preference_set_sha256": preference_set.preference_set_sha256,
        "preference_set_digest": preference_set.preference_set_digest,
        "pair_count": len(preference_set.pairs),
        "learning_rate_float64_hex": learning_rate.hex(),
        "clip_max_float64_hex": max_gradient_norm.hex(),
        "verifier_source_sha256": verifier_authority.source_sha256,
        "verifier_source_size_bytes": verifier_authority.source_size_bytes,
        "verifier_model_closure_path": str(verifier_authority.model_closure_path),
        "verifier_model_closure_sha256": verifier_authority.model_closure_sha256,
        "verifier_model_closure_size_bytes": verifier_authority.model_closure_size_bytes,
        "verifier_model_revision": verifier_authority.model_revision,
        "verifier_qualification": expected_qualification,
        "verifier_qualification_digest": object_sha256(expected_qualification),
        "gaussian_kernel_sha256": GAUSSIAN_KERNEL_SHA256,
        "apg_guidance_sha256": APG_GUIDANCE_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
    }
    input_digest = object_sha256(input_closure)
    gathered_inputs: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_inputs,
        {**input_closure, "input_digest": input_digest},
        group=runtime.parallel.world_group,
    )
    if gathered_inputs != [gathered_inputs[0]] * WORLD_SIZE:
        fail("WORLD8 update input/config/runtime branch closure differs")
    if not preference_set.pairs:
        return _zero_update_receipt_v1(
            runtime,
            preference_set=preference_set,
            status="ZERO_UPDATE_NO_PREFERENCE_PAIR",
            admission=None,
            trainable_digest_before=trainable_digest_before,
        )
    pair = preference_set.pairs[0]
    if (
        pair.source.row_id != runtime.source_row_id
        or pair.source.source_video_sha256 != runtime.source_video_sha256
        or pair.source.instruction_sha256 != runtime.instruction_sha256
    ):
        fail("one-source runtime/preference source join differs")
    admission = assess_pair_engineering_admission_v1(
        pair, authority=verifier_authority
    )
    gathered_admission: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_admission, admission["admission_digest"], group=runtime.parallel.world_group
    )
    if gathered_admission != [admission["admission_digest"]] * WORLD_SIZE:
        fail("WORLD8 engineering verifier admission differs")
    if admission["eligible"] is not True:
        return _zero_update_receipt_v1(
            runtime,
            preference_set=preference_set,
            status="ZERO_UPDATE_VERIFIER_UNDETERMINED_OR_INELIGIBLE",
            admission=admission,
            trainable_digest_before=trainable_digest_before,
        )

    torch.cuda.reset_peak_memory_stats(runtime.device)

    chosen_trajectory = _trajectory_for_rollout_v1(
        runtime,
        pair.chosen,
        source=pair.source,
        expected_decoded_rollout_sha256=admission["chosen"][
            "decoded_rollout_receipt_sha256"
        ],
        require_local_arm=False,
    )
    rejected_trajectory = _trajectory_for_rollout_v1(
        runtime,
        pair.rejected,
        source=pair.source,
        expected_decoded_rollout_sha256=admission["rejected"][
            "decoded_rollout_receipt_sha256"
        ],
        require_local_arm=False,
    )
    endpoint_roles = endpoint_roles_by_dp_arm_v1(
        chosen_trajectory.value["dp_arm"], rejected_trajectory.value["dp_arm"]
    )
    arm_to_endpoint = {
        chosen_trajectory.value["dp_arm"]: (
            "chosen",
            pair.chosen,
            chosen_trajectory,
        ),
        rejected_trajectory.value["dp_arm"]: (
            "rejected",
            pair.rejected,
            rejected_trajectory,
        ),
    }
    if set(arm_to_endpoint) != set(range(DP_SIZE)):
        fail("chosen/rejected trajectories must occupy distinct DP arms")
    if {
        arm: endpoint[0] for arm, endpoint in arm_to_endpoint.items()
    } != endpoint_roles:
        fail("chosen/rejected DP arm role projection differs")
    local_role, local_rollout, local_trajectory = arm_to_endpoint[runtime.dp_arm]
    pass1 = replay_trajectory_pass1_v1(runtime, local_trajectory)
    local_pass1 = {
        "rank": runtime.parallel.contract.rank,
        "dp_arm": runtime.dp_arm,
        "sp_rank": runtime.sp_rank,
        "role": local_role,
        "rollout_id": local_rollout.rollout_id,
        "trajectory_receipt_sha256": local_trajectory.sha256,
        "logprob_sum_float64_hex": float(pass1["logprob_sum"]).hex(),
        "pass1_digest": object_sha256(pass1),
    }
    gathered_pass1: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_pass1, local_pass1, group=runtime.parallel.world_group
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("rank") != rank
        or row.get("dp_arm") != rank // SP_SIZE
        or row.get("sp_rank") != rank % SP_SIZE
        or row.get("role") != arm_to_endpoint[rank // SP_SIZE][0]
        for rank, row in enumerate(gathered_pass1)
    ):
        fail("WORLD8 pass1 rank/endpoint placement differs")
    by_role = {
        role: gathered_pass1[arm * SP_SIZE : (arm + 1) * SP_SIZE]
        for arm, (role, _, _) in arm_to_endpoint.items()
    }
    chosen_rows = by_role["chosen"]
    rejected_rows = by_role["rejected"]
    for role, rows in (("chosen", chosen_rows), ("rejected", rejected_rows)):
        projection = [
            {
                key: value
                for key, value in row.items()
                if key not in ("rank", "sp_rank")
            }
            for row in rows
        ]
        if projection != [projection[0]] * SP_SIZE or projection[0]["role"] != role:
            fail(f"SP4 {role} pass1 replay differs")
    chosen_sum = float.fromhex(chosen_rows[0]["logprob_sum_float64_hex"])
    rejected_sum = float.fromhex(rejected_rows[0]["logprob_sum_float64_hex"])
    coefficients = preference_coefficients_v1(chosen_sum, rejected_sum)
    local_coefficient = coefficients[
        f"{local_role}_local_coefficient_after_dp2_compensation"
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in runtime.named_trainable_parameters],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    pass2 = replay_trajectory_pass2_backward_v1(
        runtime,
        local_trajectory,
        local_trajectory_coefficient=float(local_coefficient),
    )
    peak_bytes = int(torch.cuda.max_memory_allocated(runtime.device))
    total_bytes = int(torch.cuda.get_device_properties(runtime.device).total_memory)
    peak_fraction = peak_bytes / float(total_bytes)
    local_memory = {
        "world_rank": runtime.parallel.contract.rank,
        "device_index": runtime.device.index,
        "peak_memory_allocated_bytes": peak_bytes,
        "total_device_memory_bytes": total_bytes,
        "peak_fraction_float64_hex": peak_fraction.hex(),
        "measurement": "torch.cuda.max_memory_allocated_after_reset_no_dummy",
        "threshold_float64_hex": MINIMUM_TRUE_GPU_MEMORY_FRACTION.hex(),
        "above_threshold": peak_fraction > MINIMUM_TRUE_GPU_MEMORY_FRACTION,
    }
    gathered_memory: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_memory, local_memory, group=runtime.parallel.world_group
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("world_rank") != rank
        or row.get("device_index") != rank
        or row.get("above_threshold") is not True
        for rank, row in enumerate(gathered_memory)
    ):
        fail("WORLD8 real training peak did not exceed 50 percent on every GPU")
    local_gradients_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in runtime.named_trainable_parameters
    )
    finite_probe = torch.tensor(
        int(local_gradients_finite), dtype=torch.int32, device=runtime.device
    )
    dist.all_reduce(finite_probe, op=dist.ReduceOp.MIN, group=runtime.parallel.world_group)
    if int(finite_probe.item()) != 1:
        fail("WORLD8 local exact40 gradient is missing/non-finite")
    preclip_norm = distributed_runtime.synchronize_gradients(
        runtime.named_trainable_parameters, runtime.parallel
    )
    clipped = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in runtime.named_trainable_parameters],
        max_norm=max_gradient_norm,
    )
    if not math.isfinite(float(clipped)) or float(clipped) <= 0.0:
        fail("WORLD8 exact40 gradient clipping differs")
    rollback_snapshot = tuple(
        (name, parameter.detach().to(device="cpu").clone())
        for name, parameter in runtime.named_trainable_parameters
    )

    def rollback_parameters() -> None:
        with torch.no_grad():
            for (expected_name, parameter), (saved_name, saved) in zip(
                runtime.named_trainable_parameters, rollback_snapshot
            ):
                if expected_name != saved_name or tuple(parameter.shape) != tuple(saved.shape):
                    fail("rollback trainable inventory changed")
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
        runtime._behavior_policy_sha256 = behavior_sha
        runtime._update_executed = False

    local_step_status: Mapping[str, Any]
    try:
        optimizer.step()
        local_step_status = {
            "rank": runtime.parallel.contract.rank,
            "ok": True,
            "error_type": None,
        }
    except Exception as error:
        local_step_status = {
            "rank": runtime.parallel.contract.rank,
            "ok": False,
            "error_type": type(error).__name__,
        }
    gathered_step_status: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_step_status, local_step_status, group=runtime.parallel.world_group
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("rank") != rank
        or row.get("ok") is not True
        for rank, row in enumerate(gathered_step_status)
    ):
        rollback_parameters()
        restored = distributed_runtime.parameter_consensus(
            runtime.named_trainable_parameters,
            runtime.parallel.world_group,
            "full644 target-free failed-step rollback",
            expected_count=WORLD_SIZE,
        )
        if restored != trainable_digest_before:
            fail("failed optimizer-step rollback bytes differ")
        fail("WORLD8 optimizer step failed and was rolled back")
    local_after = distributed_runtime.trainable_parameters_digest(
        runtime.named_trainable_parameters
    )
    gathered_after: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_after, local_after, group=runtime.parallel.world_group)
    if gathered_after != [local_after] * WORLD_SIZE:
        rollback_parameters()
        restored = distributed_runtime.parameter_consensus(
            runtime.named_trainable_parameters,
            runtime.parallel.world_group,
            "full644 target-free divergent-step rollback",
            expected_count=WORLD_SIZE,
        )
        if restored != trainable_digest_before:
            fail("divergent optimizer-step rollback bytes differ")
        fail("WORLD8 optimizer result diverged and was rolled back")
    trainable_digest_after = local_after
    if trainable_digest_after == trainable_digest_before:
        rollback_parameters()
        fail("one-update optimizer did not change current-policy bytes")
    candidate_policy_digest_after = runtime.policy_sha256_for_trainable_digest(
        trainable_digest_after
    )
    if candidate_policy_digest_after == behavior_sha:
        rollback_parameters()
        fail("one-update did not change the composite policy closure")
    try:
        distributed_runtime.digest_consensus(
            candidate_policy_digest_after,
            group=runtime.parallel.world_group,
            expected_count=WORLD_SIZE,
            label="full644 target-free prepublication updated policy",
        )
    except Exception:
        rollback_parameters()
        raise
    try:
        policy_digest_after = runtime.commit_updated_policy_digest_v1(
            trainable_digest_after
        )
    except Exception:
        rollback_parameters()
        raise
    if policy_digest_after != candidate_policy_digest_after:
        rollback_parameters()
        fail("committed policy digest differs from checkpoint candidate")
    try:
        distributed_runtime.digest_consensus(
            policy_digest_after,
            group=runtime.parallel.world_group,
            expected_count=WORLD_SIZE,
            label="full644 target-free updated policy",
        )
    except Exception:
        rollback_parameters()
        raise
    try:
        checkpoint_binding = _save_reload_exact_one_checkpoint_v1(
            runtime,
            optimizer,
            policy_digest_before=behavior_sha,
            policy_digest_after=policy_digest_after,
            trainable_digest_before=trainable_digest_before,
            trainable_digest_after=trainable_digest_after,
        )
    except Exception:
        rollback_parameters()
        raise
    result = {
        "schema_version": UPDATE_SCHEMA,
        "status": "ENGINEERING_ONE_UPDATE_COMPLETE",
        "training_mode": "TARGET_FREE_ON_POLICY_PREFERENCE",
        "source_row_id": runtime.source_row_id,
        "source_video_sha256": runtime.source_video_sha256,
        "instruction_sha256": runtime.instruction_sha256,
        "preference_set_sha256": preference_set.preference_set_sha256,
        "preference_set_digest": preference_set.preference_set_digest,
        "pair_id": pair.pair_id,
        "behavior_policy_sha256": behavior_sha,
        "world8_input_closure": input_closure,
        "world8_input_closure_digest": input_digest,
        "engineering_verifier_admission": admission,
        "local_endpoint_role": local_role,
        "pass1_world8": gathered_pass1,
        "preference_objective": {
            "beta_float32_be_hex": PREFERENCE_BETA_FLOAT32_BE_HEX,
            "score_semantics": "dimension_normalized_gaussian_engineering_score_not_joint_log_probability",
            "score_reduction": GAUSSIAN_SCORE_REDUCTION,
            **{key: float(value).hex() for key, value in coefficients.items()},
            "dp2_local_coefficient_multiplier": DP_SIZE,
        },
        "local_pass2": pass2,
        "step_count_per_endpoint": TRAJECTORY_STEPS,
        "two_pass_streaming_replay": True,
        "fresh_stateful_unipc_each_pass": True,
        "one_transformer_graph_live": True,
        "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
        "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
        "activation_checkpoint_nonreentrant": True,
        "sp4_gradient_reduction": "mean",
        "dp2_gradient_reduction": "mean_after_local_coefficient_x2",
        "all_world8_local_gradients_finite": True,
        "true_gpu_memory_world8": gathered_memory,
        "true_gpu_memory_no_dummy_all_above_50_percent": True,
        "preclip_gradient_norm_float64_hex": float(preclip_norm).hex(),
        "clip_max_float64_hex": max_gradient_norm.hex(),
        "optimizer": "torch.optim.AdamW_exact_one_step",
        "learning_rate_float64_hex": learning_rate.hex(),
        "optimizer_constructed": True,
        "optimizer_step_executed": True,
        "transaction_rollback_snapshot_exact480_cpu": True,
        "world8_optimizer_step_status": gathered_step_status,
        "failure_rolls_back_before_success_receipt": True,
        "trainable_parameter_digest_before": trainable_digest_before,
        "trainable_parameter_digest_after": trainable_digest_after,
        "policy_digest_before": behavior_sha,
        "policy_digest_after": policy_digest_after,
        "checkpoint": checkpoint_binding,
        "checkpoint_published_only_after_world8_parameter_and_policy_consensus": True,
        "checkpoint_consumption_requires_top_level_update_stage_receipt": True,
        "checkpoint_create_only_and_reload_verified": True,
        "manifest_axis_pass_used_for_admission": False,
        "source_only_input": True,
        "paired_reference_read_count": 0,
        "external_velocity_read_count": 0,
        "terminal_stdout_requires_world8_postpublication_reload_ack": True,
        "engineering_only": True,
        "scientific_result_claimed": False,
    }
    return {**result, "receipt_digest": object_sha256(result)}


def _engineering_one_update_from_paths_v1(
    runtime: BerniniExact40PolicyV1,
    *,
    preference_set_path: Path,
    expected_preference_set_sha256: str,
) -> Mapping[str, Any]:
    """Owned production entry: held-load manifests before optimizer authority."""

    core = _load_preference_core_v1()
    if (
        type(runtime) is not BerniniExact40PolicyV1
        or getattr(runtime, "_owned_factory_token", None) is not _OWNED_RUNTIME_TOKEN
    ):
        fail("path update requires one runtime from the owned factory")
    catalog = core.load_source_catalog(
        FULL644_CATALOG_PATH,
        expected_sha256=FULL644_CATALOG_SHA256,
        require_source_files=False,
    )
    source = catalog.row(runtime.source_row_id)
    if (
        source.source_video_sha256 != runtime.source_video_sha256
        or source.instruction_sha256 != runtime.instruction_sha256
    ):
        fail("held source catalogue/runtime join differs")
    # This loader retains and hashes every trajectory, candidate media and
    # independent verifier receipt named by the preference set.
    preference = core.load_preference_set(
        preference_set_path,
        expected_sha256=expected_preference_set_sha256,
        source_catalog=catalog,
        require_rollout_files=True,
    )
    return _engineering_one_update_loaded_v1(
        runtime,
        preference_set=preference,
        verifier_authority=_frozen_qwen_authority_v1(),
        learning_rate=DEFAULT_LEARNING_RATE,
        max_gradient_norm=DEFAULT_MAX_GRADIENT_NORM,
    )


@dataclass(frozen=True)
class _OwnedRuntimeBundleV1:
    runtime: BerniniExact40PolicyV1
    source: Any = field(repr=False, compare=False)
    catalog: Any = field(repr=False, compare=False)
    checkpoint_root: Path
    vae_authority: Mapping[str, Any]
    output_root: Path
    rollout_root: Path
    factory_receipt: Mapping[str, Any]


def _load_owned_local_closure_v1() -> Mapping[str, Any]:
    rows = []
    for name in (
        "full644_target_free_preference_v1",
        "source_self_runtime",
        "inference_sigma_strata",
        "packed_preservation_lora_v2",
    ):
        _module, binding = _load_held_local_source_module_v1(name)
        rows.append(binding)
    result = {
        "schema_version": "bernini-full644-held-local-execution-closure-v1",
        "source_count": len(rows),
        "sources": rows,
        "all_executed_from_held_source_bytes": True,
        "python_bytecode_cache_used": False,
    }
    return {**result, "closure_digest": object_sha256(result)}


def _verify_projected_snapshot_metadata_v1(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify one rank0-published immutable snapshot without rereading model bytes."""

    if not isinstance(value, Mapping):
        fail("private snapshot metadata root differs")
    unsigned = dict(value)
    claimed_digest = unsigned.pop("snapshot_digest", None)
    if (
        type(claimed_digest) is not str
        or not _SHA256.fullmatch(claimed_digest)
        or object_sha256(unsigned) != claimed_digest
    ):
        fail("private snapshot metadata digest differs")
    destination = Path(value["destination"])
    destination_stat = os.lstat(destination)
    if (
        not destination.is_absolute()
        or destination.is_symlink()
        or not stat.S_ISDIR(destination_stat.st_mode)
        or stat.S_IMODE(destination_stat.st_mode) != 0o555
    ):
        fail("private snapshot immutable root differs")
    rows = value.get("files")
    if not isinstance(rows, list) or value.get("file_count") != len(rows):
        fail("private snapshot file inventory differs")
    seen = set()
    metadata_rows = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row).issuperset({"path", "sha256", "size_bytes"}) is False:
            fail("private snapshot file row differs")
        relative = row["path"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            fail("private snapshot relative path differs")
        seen.add(relative)
        expected_sha = row["sha256"]
        expected_size = row["size_bytes"]
        candidate = destination / relative
        candidate_stat = os.lstat(candidate)
        if (
            type(expected_sha) is not str
            or not _SHA256.fullmatch(expected_sha)
            or type(expected_size) is not int
            or expected_size < 0
            or candidate.is_symlink()
            or not stat.S_ISREG(candidate_stat.st_mode)
            or int(candidate_stat.st_nlink) != 1
            or int(candidate_stat.st_size) != expected_size
            or stat.S_IMODE(candidate_stat.st_mode) != 0o444
            or ("mode_octal" in row and row["mode_octal"] != "0444")
        ):
            fail("private snapshot member metadata differs")
        metadata_rows.append(
            {
                "path": relative,
                "sha256": expected_sha,
                "size_bytes": expected_size,
                "mode_octal": "0444",
                "nlink": 1,
            }
        )
    expected_directories = set()
    for relative in seen:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    physical_files = []
    physical_directories = []
    for path in destination.rglob("*"):
        path_stat = os.lstat(path)
        relative = path.relative_to(destination).as_posix()
        if stat.S_ISREG(path_stat.st_mode) and not path.is_symlink():
            physical_files.append(relative)
        elif stat.S_ISDIR(path_stat.st_mode) and not path.is_symlink():
            physical_directories.append(relative)
        else:
            fail("private snapshot special filesystem entry differs")
    if (
        sorted(physical_files) != sorted(seen)
        or set(physical_directories) != expected_directories
    ):
        fail("private snapshot physical file/directory set differs")
    for directory in (destination, *(destination / item for item in physical_directories)):
        directory_stat = os.lstat(directory)
        if directory.is_symlink() or stat.S_IMODE(directory_stat.st_mode) != 0o555:
            fail("private snapshot directory metadata differs")
    result = {
        "schema_version": "bernini-full644-shared-snapshot-metadata-replay-v1",
        "destination": str(destination),
        "snapshot_digest": claimed_digest,
        "file_count": len(metadata_rows),
        "metadata_rows_digest": object_sha256(metadata_rows),
        "content_bytes_reread": False,
        "rank0_published_immutable_snapshot": True,
    }
    return {**result, "replay_digest": object_sha256(result)}


_SERIAL_OWNER_STATUS_FIELDS = frozenset(
    {
        "schema_version", "phase", "owner_rank", "reporter_rank", "role",
        "status", "evidence_digest", "error_type", "error_digest",
    }
)
_RANK0_VAE_SOURCE_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version", "producer_rank", "source_row_id",
        "source_video_sha256", "instruction_sha256", "source_row_digest",
        "catalog_sha256", "catalog_digest", "catalog_binding_digest",
        "source_decode", "source_decode_digest", "source_state_shape",
        "source_state_numel", "source_state_dtype", "source_state_layout",
        "source_state_contiguous", "source_state_requires_grad",
        "source_state_sha256", "vae_authority", "vae_authority_digest",
        "rank0_only_source_decode_and_vae_encode",
        "vae_released_without_cpu_rematerialization", "host_allocator_trim",
        "descriptor_digest",
    }
)
_RANK0_VAE_SOURCE_READY_FIELDS = frozenset(
    {
        "schema_version", "world_rank", "status", "device_index",
        "descriptor_digest", "error_type", "error_digest",
    }
)
_RANK0_VAE_SOURCE_REPLAY_FIELDS = frozenset(
    {
        "schema_version", "world_rank", "status", "device_index",
        "descriptor_digest", "source_state_shape", "source_state_dtype",
        "source_state_contiguous", "source_state_requires_grad",
        "source_state_sha256", "error_type", "error_digest",
    }
)


def _bounded_failure_row_v1(
    *, schema_version: str, world_rank: int, descriptor_digest: Any,
    error: Exception, replay: bool,
) -> Mapping[str, Any]:
    result = {
        "schema_version": schema_version,
        "world_rank": world_rank,
        "status": "FAILED",
        "device_index": world_rank,
        "descriptor_digest": descriptor_digest,
        "error_type": type(error).__name__[:128],
        "error_digest": hashlib.sha256(
            (type(error).__name__ + "\0" + str(error)[:2000]).encode("utf-8")
        ).hexdigest(),
    }
    if replay:
        result.update(
            {
                "source_state_shape": None,
                "source_state_dtype": None,
                "source_state_contiguous": None,
                "source_state_requires_grad": None,
                "source_state_sha256": None,
            }
        )
    return result


def _validate_rank0_vae_source_descriptor_v1(
    value: Any,
    *,
    expected_source_row_id: str,
    expected_source_video_sha256: str,
    expected_instruction_sha256: str,
    expected_source_row_digest: str,
    expected_catalog_binding: Mapping[str, Any],
    expected_vae_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    row = _closed(
        value, _RANK0_VAE_SOURCE_DESCRIPTOR_FIELDS,
        label="rank0 VAE source descriptor",
    )
    unsigned = dict(row)
    claimed_digest = unsigned.pop("descriptor_digest")
    decode = row["source_decode"]
    authority = row["vae_authority"]
    trim = row["host_allocator_trim"]
    if not isinstance(decode, Mapping) or not isinstance(authority, Mapping):
        fail("rank0 VAE source descriptor nested object differs")
    bucket = decode.get("source_derived_bucket_hw")
    expected_shape = None
    if (
        isinstance(bucket, list)
        and len(bucket) == 2
        and all(type(item) is int and item > 0 and item % 8 == 0 for item in bucket)
    ):
        expected_shape = [1, 16, 21, bucket[0] // 8, bucket[1] // 8]
    if (
        row["schema_version"]
        != "bernini-full644-rank0-vae-source-broadcast-v1"
        or row["producer_rank"] != 0
        or row["source_row_id"] != expected_source_row_id
        or row["source_video_sha256"] != expected_source_video_sha256
        or row["instruction_sha256"] != expected_instruction_sha256
        or row["source_row_digest"] != expected_source_row_digest
        or row["catalog_sha256"] != expected_catalog_binding.get("catalog_sha256")
        or row["catalog_digest"] != expected_catalog_binding.get("catalog_digest")
        or row["catalog_binding_digest"]
        != expected_catalog_binding.get("binding_digest")
        or decode.get("schema_version")
        != "bernini-full644-owned-source-decode-v1"
        or decode.get("source_row_id") != expected_source_row_id
        or decode.get("source_video_sha256") != expected_source_video_sha256
        or decode.get("frame_count") != FRAME_COUNT
        or decode.get("fps_float64_hex") != FPS.hex()
        or decode.get("target_media_read_count") != 0
        or type(row["source_decode_digest"]) is not str
        or not _SHA256.fullmatch(row["source_decode_digest"])
        or decode.get("decode_digest") != row["source_decode_digest"]
        or object_sha256({key: decode[key] for key in decode if key != "decode_digest"})
        != row["source_decode_digest"]
        or expected_shape is None
        or row["source_state_shape"] != expected_shape
        or type(row["source_state_numel"]) is not int
        or row["source_state_numel"] != math.prod(expected_shape)
        or row["source_state_dtype"] != "torch.float32"
        or row["source_state_layout"] != "torch.strided"
        or row["source_state_contiguous"] is not True
        or row["source_state_requires_grad"] is not False
        or type(row["source_state_sha256"]) is not str
        or not _SHA256.fullmatch(row["source_state_sha256"])
        or authority != expected_vae_authority
        or set(authority) != set(_VAE_AUTHORITY_FIELDS)
        or row["vae_authority_digest"] != object_sha256(expected_vae_authority)
        or row["rank0_only_source_decode_and_vae_encode"] is not True
        or row["vae_released_without_cpu_rematerialization"] is not True
        or not isinstance(trim, Mapping)
        or set(trim) != {"allocator", "called", "return_code"}
        or trim.get("allocator") != "glibc_malloc_trim"
        or trim.get("called") is not True
        or type(trim.get("return_code")) is not int
        or trim.get("return_code") not in (0, 1)
        or type(claimed_digest) is not str
        or not _SHA256.fullmatch(claimed_digest)
        or object_sha256(unsigned) != claimed_digest
    ):
        fail("rank0 VAE source descriptor closure differs")
    return row


def _validate_world8_rank0_vae_rows_v1(
    rows: Any, *, descriptor: Mapping[str, Any], replay: bool,
) -> list[Mapping[str, Any]]:
    fields = (
        _RANK0_VAE_SOURCE_REPLAY_FIELDS
        if replay else _RANK0_VAE_SOURCE_READY_FIELDS
    )
    expected_schema = (
        "bernini-full644-rank0-vae-source-replay-v1"
        if replay else "bernini-full644-rank0-vae-source-ready-v1"
    )
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        fail("rank0 VAE source WORLD8 envelope differs")
    validated = []
    for world_rank, raw in enumerate(rows):
        row = _closed(raw, fields, label=f"rank0 VAE source rank{world_rank}")
        if (
            row["schema_version"] != expected_schema
            or row["world_rank"] != world_rank
            or row["status"] != ("REPLAYED" if replay else "READY")
            or row["device_index"] != world_rank
            or row["descriptor_digest"] != descriptor["descriptor_digest"]
            or row["error_type"] is not None
            or row["error_digest"] is not None
        ):
            fail("rank0 VAE source WORLD8 status differs")
        if replay and (
            row["source_state_shape"] != descriptor["source_state_shape"]
            or row["source_state_dtype"] != "torch.float32"
            or row["source_state_contiguous"] is not True
            or row["source_state_requires_grad"] is not False
            or row["source_state_sha256"] != descriptor["source_state_sha256"]
        ):
            fail("rank0 VAE source WORLD8 tensor replay differs")
        validated.append(row)
    return validated


def _validate_world8_serial_owner_rows_v1(
    rows: Any, *, phase: str, owner_rank: int
) -> Mapping[str, Any]:
    if (
        type(phase) is not str
        or phase not in ("rank0_vae_source_encode", "renderer_lora_construct")
        or type(owner_rank) is not int
        or owner_rank not in range(WORLD_SIZE)
        or (phase == "rank0_vae_source_encode" and owner_rank != 0)
        or not isinstance(rows, list)
        or len(rows) != WORLD_SIZE
    ):
        fail("serialized owner status envelope differs")
    owner_row: Optional[Mapping[str, Any]] = None
    for reporter_rank, raw in enumerate(rows):
        row = _closed(
            raw, _SERIAL_OWNER_STATUS_FIELDS,
            label=f"serialized {phase} owner{owner_rank} rank{reporter_rank}",
        )
        expected_role = "owner" if reporter_rank == owner_rank else "waiter"
        if (
            row["schema_version"]
            != "bernini-full644-world8-serial-owner-status-v1"
            or row["phase"] != phase
            or row["owner_rank"] != owner_rank
            or row["reporter_rank"] != reporter_rank
            or row["role"] != expected_role
        ):
            fail("serialized owner placement/status differs")
        if reporter_rank != owner_rank:
            if (
                row["status"] != "WAITING"
                or row["evidence_digest"] is not None
                or row["error_type"] is not None
                or row["error_digest"] is not None
            ):
                fail("serialized nonowner executed construction")
        else:
            owner_row = row
    if owner_row is None:
        fail("serialized owner status is absent")
    if owner_row["status"] == "COMPLETE":
        if (
            type(owner_row["evidence_digest"]) is not str
            or not _SHA256.fullmatch(owner_row["evidence_digest"])
            or owner_row["error_type"] is not None
            or owner_row["error_digest"] is not None
        ):
            fail("serialized owner completion evidence differs")
        return {
            "owner_rank": owner_rank,
            "phase": phase,
            "status": "COMPLETE",
            "evidence_digest": owner_row["evidence_digest"],
        }
    if owner_row["status"] == "FAILED":
        if (
            owner_row["evidence_digest"] is not None
            or type(owner_row["error_type"]) is not str
            or not owner_row["error_type"]
            or len(owner_row["error_type"]) > 128
            or type(owner_row["error_digest"]) is not str
            or not _SHA256.fullmatch(owner_row["error_digest"])
        ):
            fail("serialized owner failure evidence differs")
        return {
            "owner_rank": owner_rank,
            "phase": phase,
            "status": "FAILED",
            "error_type": owner_row["error_type"],
            "error_digest": owner_row["error_digest"],
        }
    fail("serialized owner terminal status differs")


def _run_world8_serial_owner_round_v1(
    *, dist: Any, group: Any, rank: int, owner_rank: int, phase: str,
    constructor: Callable[[], tuple[Any, Mapping[str, Any]]],
    payload_validator: Callable[[Any], None],
) -> tuple[Any, Mapping[str, Any]]:
    if (
        type(rank) is not int
        or rank not in range(WORLD_SIZE)
        or not callable(constructor)
        or not callable(payload_validator)
    ):
        fail("serialized owner round arguments differ")
    payload: Any = None
    row = {
        "schema_version": "bernini-full644-world8-serial-owner-status-v1",
        "phase": phase,
        "owner_rank": owner_rank,
        "reporter_rank": rank,
        "role": "owner" if rank == owner_rank else "waiter",
        "status": "WAITING",
        "evidence_digest": None,
        "error_type": None,
        "error_digest": None,
    }
    if rank == owner_rank:
        try:
            payload, evidence = constructor()
            payload_validator(payload)
            if not isinstance(evidence, Mapping):
                fail("serialized owner construction evidence differs")
            row["status"] = "COMPLETE"
            row["evidence_digest"] = object_sha256(evidence)
        except Exception as error:
            payload = None
            row["status"] = "FAILED"
            row["error_type"] = type(error).__name__[:128]
            row["error_digest"] = hashlib.sha256(
                (type(error).__name__ + "\0" + str(error)[:2000]).encode("utf-8")
            ).hexdigest()
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, row, group=group)
    result = _validate_world8_serial_owner_rows_v1(
        gathered, phase=phase, owner_rank=owner_rank
    )
    if result["status"] != "COMPLETE":
        fail(
            f"serialized {phase} owner{owner_rank} failed: "
            f"{result['error_type']}:{result['error_digest']}"
        )
    dist.barrier(group=group)
    return payload, result


def _run_rank0_vae_source_constructor_v1(
    *, dist: Any, group: Any, rank: int,
    constructor: Callable[[], tuple[Any, Mapping[str, Any]]],
    payload_validator: Callable[[Any], None],
) -> tuple[Any, Mapping[str, Any]]:
    """Execute the frozen source decode/VAE encode exactly once on rank zero."""

    return _run_world8_serial_owner_round_v1(
        dist=dist,
        group=group,
        rank=rank,
        owner_rank=0,
        phase="rank0_vae_source_encode",
        constructor=constructor,
        payload_validator=payload_validator,
    )


def _broadcast_rank0_vae_source_state_v1(
    *,
    dist: Any,
    torch_module: Any,
    group: Any,
    rank: int,
    device: Any,
    rank0_payload: Any,
    expected_source_row_id: str,
    expected_source_video_sha256: str,
    expected_instruction_sha256: str,
    expected_source_row_digest: str,
    expected_catalog_binding: Mapping[str, Any],
    expected_vae_authority: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Broadcast one exact rank0 FP32 latent; never reduce, average or tolerate."""

    if type(rank) is not int or rank not in range(WORLD_SIZE):
        fail("rank0 VAE source broadcast rank differs")
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            if (
                not isinstance(rank0_payload, Mapping)
                or set(rank0_payload) != {"source_state", "descriptor"}
            ):
                fail("rank0 VAE source payload differs")
            descriptor = _validate_rank0_vae_source_descriptor_v1(
                rank0_payload["descriptor"],
                expected_source_row_id=expected_source_row_id,
                expected_source_video_sha256=expected_source_video_sha256,
                expected_instruction_sha256=expected_instruction_sha256,
                expected_source_row_digest=expected_source_row_digest,
                expected_catalog_binding=expected_catalog_binding,
                expected_vae_authority=expected_vae_authority,
            )
            envelope[0] = {"ok": True, "descriptor": dict(descriptor)}
        except Exception as error:
            envelope[0] = {
                "ok": False,
                "error_type": type(error).__name__[:128],
                "error_digest": hashlib.sha256(
                    (type(error).__name__ + "\0" + str(error)[:2000]).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
    dist.broadcast_object_list(envelope, src=0, group=group)
    broadcast_envelope = envelope[0]
    if (
        not isinstance(broadcast_envelope, Mapping)
        or broadcast_envelope.get("ok") is not True
        or set(broadcast_envelope) != {"ok", "descriptor"}
    ):
        fail(f"rank0 VAE source descriptor publication failed: {broadcast_envelope}")
    descriptor = broadcast_envelope["descriptor"]

    source_state: Any = None
    try:
        descriptor = _validate_rank0_vae_source_descriptor_v1(
            descriptor,
            expected_source_row_id=expected_source_row_id,
            expected_source_video_sha256=expected_source_video_sha256,
            expected_instruction_sha256=expected_instruction_sha256,
            expected_source_row_digest=expected_source_row_digest,
            expected_catalog_binding=expected_catalog_binding,
            expected_vae_authority=expected_vae_authority,
        )
        if rank == 0:
            source_state = rank0_payload["source_state"]
        else:
            if rank0_payload is not None:
                fail("nonroot supplied a private VAE source payload")
            source_state = torch_module.empty(
                tuple(descriptor["source_state_shape"]),
                dtype=torch_module.float32,
                device=device,
            )
        if (
            type(source_state) is not torch_module.Tensor
            or tuple(int(item) for item in source_state.shape)
            != tuple(descriptor["source_state_shape"])
            or source_state.dtype != torch_module.float32
            or source_state.layout != torch_module.strided
            or source_state.device != device
            or int(source_state.device.index) != rank
            or not source_state.is_contiguous()
            or source_state.requires_grad
        ):
            fail("rank0 VAE source prebroadcast tensor ABI differs")
        ready_row = {
            "schema_version": "bernini-full644-rank0-vae-source-ready-v1",
            "world_rank": rank,
            "status": "READY",
            "device_index": int(source_state.device.index),
            "descriptor_digest": descriptor["descriptor_digest"],
            "error_type": None,
            "error_digest": None,
        }
    except Exception as error:
        ready_row = _bounded_failure_row_v1(
            schema_version="bernini-full644-rank0-vae-source-ready-v1",
            world_rank=rank,
            descriptor_digest=(
                descriptor.get("descriptor_digest")
                if isinstance(descriptor, Mapping) else None
            ),
            error=error,
            replay=False,
        )
    gathered_ready: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_ready, ready_row, group=group)
    ready_rows = _validate_world8_rank0_vae_rows_v1(
        gathered_ready, descriptor=descriptor, replay=False
    )

    # This is a byte-exact tensor broadcast.  There is deliberately no
    # all_reduce, averaging, tolerance or per-rank VAE re-encode.
    dist.broadcast(source_state, src=0, group=group)
    local_state: Any = None
    try:
        local_state = source_state.detach().clone(
            memory_format=torch_module.contiguous_format
        )
        local_sha = tensor_sha256(local_state)
        if (
            tuple(int(item) for item in local_state.shape)
            != tuple(descriptor["source_state_shape"])
            or local_state.dtype != torch_module.float32
            or local_state.layout != torch_module.strided
            or local_state.device != device
            or int(local_state.device.index) != rank
            or not local_state.is_contiguous()
            or local_state.requires_grad
            or not bool(torch_module.isfinite(local_state).all().item())
            or local_sha != descriptor["source_state_sha256"]
        ):
            fail("rank0 VAE source broadcast tensor differs")
        replay_row = {
            "schema_version": "bernini-full644-rank0-vae-source-replay-v1",
            "world_rank": rank,
            "status": "REPLAYED",
            "device_index": int(local_state.device.index),
            "descriptor_digest": descriptor["descriptor_digest"],
            "source_state_shape": [int(item) for item in local_state.shape],
            "source_state_dtype": "torch.float32",
            "source_state_contiguous": True,
            "source_state_requires_grad": False,
            "source_state_sha256": local_sha,
            "error_type": None,
            "error_digest": None,
        }
    except Exception as error:
        replay_row = _bounded_failure_row_v1(
            schema_version="bernini-full644-rank0-vae-source-replay-v1",
            world_rank=rank,
            descriptor_digest=descriptor["descriptor_digest"],
            error=error,
            replay=True,
        )
    gathered_replay: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_replay, replay_row, group=group)
    replay_rows = _validate_world8_rank0_vae_rows_v1(
        gathered_replay, descriptor=descriptor, replay=True
    )
    if local_state is None:
        fail("rank0 VAE source local clone is absent")
    consensus = {
        "schema_version": "bernini-full644-rank0-vae-source-consensus-v1",
        "descriptor": dict(descriptor),
        "ready_rows": ready_rows,
        "replay_rows": replay_rows,
        "rank0_encode_call_count": 1,
        "tensor_broadcast_count": 1,
        "all_ranks_bit_exact_sha_verified": True,
        "all_ranks_device_local_contiguous_fp32_clone": True,
        "reduction_or_averaging_used": False,
        "per_rank_vae_reencode_used": False,
    }
    return local_state, descriptor["source_decode"], {
        **consensus, "consensus_digest": object_sha256(consensus)
    }


def _trim_host_allocator_v1() -> Mapping[str, Any]:
    """Return unused glibc arenas after each serialized host checkpoint load."""

    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return_code = int(malloc_trim(0))
    except Exception as error:
        raise TargetFreeBerniniRuntimeError(
            f"serialized host allocator trim failed: {type(error).__name__}"
        ) from error
    if return_code not in (0, 1):
        fail("serialized host allocator trim return code differs")
    return {
        "allocator": "glibc_malloc_trim",
        "called": True,
        "return_code": return_code,
    }


_MIOPEN_CACHE_DIRECTORY_NAMES = {
    "miopen_user_db": "miopen-user",
    "miopen_custom_cache": "miopen-custom",
    "tmp": "tmp",
    "xdg_cache": "xdg-cache",
    "pytorch_kernel_cache": "torch-kernels",
    "torch_extensions": "torch-extensions",
    "triton_cache": "triton",
    "torchinductor_cache": "inductor",
}
_MIOPEN_CACHE_ENV_ROLES = {
    "MIOPEN_USER_DB_PATH": "miopen_user_db",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen_custom_cache",
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "XDG_CACHE_HOME": "xdg_cache",
    "PYTORCH_KERNEL_CACHE_PATH": "pytorch_kernel_cache",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor_cache",
}
_MIOPEN_CACHE_BINDING_FIELDS = frozenset(
    {
        "schema_version", "world_rank", "local_rank", "world_size",
        "local_world_size", "slurm_job_id", "slurm_step_id", "hostname",
        "uid", "base_root", "output_path_sha256", "rank_root", "rank_root_device",
        "rank_root_inode", "environment", "directories", "probe_sha256",
        "sqlite_probe", "torch_absent_at_configuration",
        "all_directories_fresh_mode0700_owned", "all_probes_removed",
        "all_directories_empty_before_torch", "binding_digest",
    }
)
_MIOPEN_CACHE_DIRECTORY_FIELDS = frozenset(
    {"role", "path", "device", "inode", "uid", "mode_octal"}
)


def _required_environment_integer_v1(name: str) -> int:
    raw = os.environ.get(name)
    if type(raw) is not str or not re.fullmatch(r"0|[1-9][0-9]*", raw):
        fail(f"required {name} environment integer differs")
    return int(raw)


def _prepare_rank_local_miopen_cache_v1(
    *, output_path: Path, base_root: Path = Path("/tmp")
) -> Mapping[str, Any]:
    """Create a fresh rank-local MIOpen/SQLite closure before torch import."""

    if "torch" in sys.modules:
        fail("rank-local MIOpen cache must be configured before torch import")
    import socket
    import sqlite3

    world_rank = _required_environment_integer_v1("RANK")
    local_rank = _required_environment_integer_v1("LOCAL_RANK")
    world_size = _required_environment_integer_v1("WORLD_SIZE")
    local_world_size = _required_environment_integer_v1("LOCAL_WORLD_SIZE")
    if (
        world_size != WORLD_SIZE
        or local_world_size != WORLD_SIZE
        or world_rank != local_rank
        or world_rank not in range(WORLD_SIZE)
    ):
        fail("rank-local MIOpen cache requires one-node WORLD8 rank placement")
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    if (
        type(job_id) is not str
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", job_id) is None
        or type(step_id) is not str
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", step_id) is None
    ):
        fail("rank-local MIOpen cache Slurm identity differs")
    if not output_path.is_absolute() or output_path == Path("/"):
        fail("rank-local MIOpen cache output identity differs")
    output_path_sha = hashlib.sha256(str(output_path).encode("utf-8")).hexdigest()
    base_root = base_root.resolve(strict=True)
    base_stat = os.lstat(base_root)
    if (
        not base_root.is_absolute()
        or base_root.is_symlink()
        or not stat.S_ISDIR(base_stat.st_mode)
    ):
        fail("rank-local MIOpen cache base root differs")
    rank_root = base_root / (
        f"bernini-full644-miopen-j{job_id}-s{step_id}-o{output_path_sha[:16]}-r{world_rank}"
    )
    try:
        os.mkdir(rank_root, 0o700)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(
            f"fresh rank-local MIOpen cache root cannot be created: {error}"
        ) from error
    os.chmod(rank_root, 0o700)
    directories: dict[str, Path] = {}
    for role, basename in _MIOPEN_CACHE_DIRECTORY_NAMES.items():
        path = rank_root / basename
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
        directories[role] = path
    environment = {
        name: str(directories[role])
        for name, role in _MIOPEN_CACHE_ENV_ROLES.items()
    }
    for name, value in environment.items():
        os.environ[name] = value

    probe_payload = b"bernini-full644-rank-local-cache-write-probe-v1\n"
    probe_sha = hashlib.sha256(probe_payload).hexdigest()
    for role, path in directories.items():
        probe = path / f".{role}.write-probe"
        binding = write_create_only(probe, probe_payload, mode=0o600)
        if (
            binding["sha256"] != probe_sha
            or read_stable_file(
                probe,
                expected_sha256=probe_sha,
                expected_mode=0o600,
                label=f"rank-local {role} cache write probe",
            )
            != probe_payload
        ):
            fail("rank-local cache write/reopen probe differs")
        os.unlink(probe)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    sqlite_path = directories["miopen_custom_cache"] / ".sqlite-write-probe.db"
    connection = sqlite3.connect(str(sqlite_path))
    try:
        connection.execute("CREATE TABLE cache_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO cache_probe VALUES ('ok')")
        connection.commit()
    finally:
        connection.close()
    os.chmod(sqlite_path, 0o600)
    sqlite_binding = _stable_file_binding_v1(
        sqlite_path, label="rank-local MIOpen SQLite probe", expected_mode=0o600
    )
    readonly = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        quick_check = tuple(row[0] for row in readonly.execute("PRAGMA quick_check"))
        values = tuple(row[0] for row in readonly.execute("SELECT value FROM cache_probe"))
    finally:
        readonly.close()
    if quick_check != ("ok",) or values != ("ok",):
        fail("rank-local MIOpen SQLite create/readonly-reopen probe differs")
    os.unlink(sqlite_path)
    custom_descriptor = os.open(
        directories["miopen_custom_cache"],
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(custom_descriptor)
    finally:
        os.close(custom_descriptor)

    uid = os.getuid()
    directory_rows = []
    for role in sorted(directories):
        path = directories[role]
        path_stat = os.lstat(path)
        if (
            path.is_symlink()
            or not stat.S_ISDIR(path_stat.st_mode)
            or stat.S_IMODE(path_stat.st_mode) != 0o700
            or int(path_stat.st_uid) != uid
            or list(path.iterdir())
        ):
            fail("rank-local MIOpen cache directory closure differs")
        directory_rows.append(
            {
                "role": role,
                "path": str(path),
                "device": int(path_stat.st_dev),
                "inode": int(path_stat.st_ino),
                "uid": int(path_stat.st_uid),
                "mode_octal": "0700",
            }
        )
    root_stat = os.lstat(rank_root)
    if (
        stat.S_IMODE(root_stat.st_mode) != 0o700
        or int(root_stat.st_uid) != uid
        or {path.name for path in rank_root.iterdir()}
        != set(_MIOPEN_CACHE_DIRECTORY_NAMES.values())
    ):
        fail("rank-local MIOpen cache root closure differs")
    result = {
        "schema_version": "bernini-full644-rank-local-miopen-cache-v1",
        "world_rank": world_rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "hostname": socket.gethostname(),
        "uid": uid,
        "base_root": str(base_root),
        "output_path_sha256": output_path_sha,
        "rank_root": str(rank_root),
        "rank_root_device": int(root_stat.st_dev),
        "rank_root_inode": int(root_stat.st_ino),
        "environment": environment,
        "directories": directory_rows,
        "probe_sha256": probe_sha,
        "sqlite_probe": {
            "sqlite_version": sqlite3.sqlite_version,
            "database_sha256": sqlite_binding["sha256"],
            "database_size_bytes": sqlite_binding["size_bytes"],
            "mode_octal": "0600",
            "quick_check": "ok",
            "readonly_reopen_value": "ok",
        },
        "torch_absent_at_configuration": True,
        "all_directories_fresh_mode0700_owned": True,
        "all_probes_removed": True,
        "all_directories_empty_before_torch": True,
    }
    return {**result, "binding_digest": object_sha256(result)}


def _validate_rank_local_miopen_cache_binding_v1(
    value: Any,
    *,
    expected_rank: int,
    require_environment: bool,
    require_physical: bool,
) -> Mapping[str, Any]:
    row = _closed(
        value, _MIOPEN_CACHE_BINDING_FIELDS,
        label=f"rank{expected_rank} local MIOpen cache binding",
    )
    unsigned = dict(row)
    claimed_digest = unsigned.pop("binding_digest")
    if (
        row["schema_version"] != "bernini-full644-rank-local-miopen-cache-v1"
        or row["world_rank"] != expected_rank
        or row["local_rank"] != expected_rank
        or row["world_size"] != WORLD_SIZE
        or row["local_world_size"] != WORLD_SIZE
        or type(claimed_digest) is not str
        or not _SHA256.fullmatch(claimed_digest)
        or object_sha256(unsigned) != claimed_digest
        or row["torch_absent_at_configuration"] is not True
        or row["all_directories_fresh_mode0700_owned"] is not True
        or row["all_probes_removed"] is not True
        or row["all_directories_empty_before_torch"] is not True
    ):
        fail("rank-local MIOpen cache binding closure differs")
    environment = _closed(
        row["environment"], frozenset(_MIOPEN_CACHE_ENV_ROLES),
        label="rank-local MIOpen cache environment",
    )
    directory_rows = row["directories"]
    if not isinstance(directory_rows, list) or len(directory_rows) != len(
        _MIOPEN_CACHE_DIRECTORY_NAMES
    ):
        fail("rank-local MIOpen cache directory inventory differs")
    by_role = {}
    for raw in directory_rows:
        item = _closed(
            raw, _MIOPEN_CACHE_DIRECTORY_FIELDS,
            label="rank-local MIOpen cache directory row",
        )
        role = item["role"]
        if role not in _MIOPEN_CACHE_DIRECTORY_NAMES or role in by_role:
            fail("rank-local MIOpen cache directory role differs")
        by_role[role] = item
    rank_root = Path(row["rank_root"])
    if (
        set(by_role) != set(_MIOPEN_CACHE_DIRECTORY_NAMES)
        or type(row["output_path_sha256"]) is not str
        or not _SHA256.fullmatch(row["output_path_sha256"])
        or rank_root.parent != Path(row["base_root"])
        or rank_root.name
        != (
            f"bernini-full644-miopen-j{row['slurm_job_id']}"
            f"-s{row['slurm_step_id']}-o{row['output_path_sha256'][:16]}"
            f"-r{expected_rank}"
        )
    ):
        fail("rank-local MIOpen cache path projection differs")
    for role, basename in _MIOPEN_CACHE_DIRECTORY_NAMES.items():
        item = by_role[role]
        expected_path = rank_root / basename
        if item["path"] != str(expected_path) or item["mode_octal"] != "0700":
            fail("rank-local MIOpen cache directory path/mode differs")
        if require_physical:
            observed = os.lstat(expected_path)
            if (
                expected_path.is_symlink()
                or not stat.S_ISDIR(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o700
                or int(observed.st_dev) != item["device"]
                or int(observed.st_ino) != item["inode"]
                or int(observed.st_uid) != item["uid"]
            ):
                fail("rank-local MIOpen cache physical directory differs")
    for name, role in _MIOPEN_CACHE_ENV_ROLES.items():
        expected = str(rank_root / _MIOPEN_CACHE_DIRECTORY_NAMES[role])
        if environment[name] != expected:
            fail("rank-local MIOpen cache environment projection differs")
        if require_environment and os.environ.get(name) != expected:
            fail("rank-local MIOpen cache live environment differs")
    if require_physical:
        root_stat = os.lstat(rank_root)
        if (
            rank_root.is_symlink()
            or stat.S_IMODE(root_stat.st_mode) != 0o700
            or int(root_stat.st_dev) != row["rank_root_device"]
            or int(root_stat.st_ino) != row["rank_root_inode"]
            or int(root_stat.st_uid) != row["uid"]
            or {path.name for path in rank_root.iterdir()}
            != set(_MIOPEN_CACHE_DIRECTORY_NAMES.values())
        ):
            fail("rank-local MIOpen cache physical root differs")
    return row


def _validate_world8_miopen_cache_bindings_v1(
    rows: Any, *, local_rank: int
) -> list[Mapping[str, Any]]:
    if (
        type(local_rank) is not int
        or local_rank not in range(WORLD_SIZE)
        or not isinstance(rows, list)
        or len(rows) != WORLD_SIZE
    ):
        fail("WORLD8 rank-local MIOpen cache envelope differs")
    validated = [
        _validate_rank_local_miopen_cache_binding_v1(
            row,
            expected_rank=rank,
            require_environment=rank == local_rank,
            require_physical=True,
        )
        for rank, row in enumerate(rows)
    ]
    if (
        len({row["rank_root"] for row in validated}) != WORLD_SIZE
        or len({row["binding_digest"] for row in validated}) != WORLD_SIZE
        or len({row["hostname"] for row in validated}) != 1
        or len({row["slurm_job_id"] for row in validated}) != 1
        or len({row["slurm_step_id"] for row in validated}) != 1
        or len({row["uid"] for row in validated}) != 1
        or len({row["rank_root_device"] for row in validated}) != 1
    ):
        fail("WORLD8 rank-local MIOpen cache isolation differs")
    return validated


def _validate_private_checkpoint_v1(
    checkpoint_root: Path, checkpoint_snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    if (
        not checkpoint_root.is_absolute()
        or checkpoint_root.is_symlink()
        or not checkpoint_root.is_dir()
        or checkpoint_snapshot.get("destination") != str(checkpoint_root)
        or checkpoint_snapshot.get("source_tree_sha256")
        != BASE_CHECKPOINT_TREE_SHA256
        or checkpoint_snapshot.get("content_manifest_sha256")
        != BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or checkpoint_snapshot.get("file_count") != BASE_CHECKPOINT_FILE_COUNT
    ):
        fail("private exact23 checkpoint root differs")
    rows = checkpoint_snapshot.get("files")
    if not isinstance(rows, list):
        fail("private exact23 checkpoint inventory differs")
    by_path = {row.get("path"): row for row in rows if isinstance(row, Mapping)}
    if len(by_path) != BASE_CHECKPOINT_FILE_COUNT:
        fail("private exact23 checkpoint path inventory differs")
    for directory in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"):
        candidate = checkpoint_root / directory
        if not candidate.is_dir() or candidate.is_symlink():
            fail(f"private exact23 checkpoint lacks {directory}")
    if (checkpoint_root / "transformer_2").exists():
        fail("private exact23 checkpoint unexpectedly has transformer_2")
    config_row = by_path.get("transformer/config.json")
    if not isinstance(config_row, Mapping):
        fail("private exact23 transformer config binding is absent")
    config_raw = read_stable_file(
        checkpoint_root / "transformer/config.json",
        expected_sha256=config_row["sha256"],
        expected_mode=0o444,
        label="private exact23 transformer config",
    )
    config = _strict_json(config_raw, label="private exact23 transformer config")
    expected = {
        "num_layers": 30,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
    }
    if any(type(config.get(key)) is not int or config.get(key) != value for key, value in expected.items()):
        fail("private exact23 transformer geometry differs")
    transformer_weight_paths = [
        path
        for path in by_path
        if type(path) is str
        and path.startswith("transformer/")
        and (path.endswith(".safetensors") or path.endswith(".safetensors.index.json"))
    ]
    if not transformer_weight_paths:
        fail("private exact23 transformer has no safetensors closure")
    return dict(config)


def _renderer_config_overrides_v1(checkpoint: Path) -> Mapping[str, Any]:
    if not checkpoint.is_absolute() or not checkpoint.is_dir() or checkpoint.is_symlink():
        fail("renderer checkpoint override differs")
    return {
        "wan22_base": str(checkpoint),
        "diff_dec_config_path": str(checkpoint),
        "skip_transformer_1": False,
        "skip_transformer_2": True,
        "switch_dit_boundary": 0.0,
        "max_sequence_length": 512,
        "shift": 3.0,
        "use_src_id_rotary_emb": True,
        "scratch": False,
        "ema_decay": None,
    }


def _validate_renderer_config_mapping_v1(
    config: Mapping[str, Any], checkpoint: Path
) -> None:
    if (
        not isinstance(config, Mapping)
        or config.get("model_type") != "bernini_renderer"
        or config.get("skip_transformer_1") is not False
        or config.get("skip_transformer_2") is not True
        or config.get("wan22_base") != str(checkpoint)
        or config.get("diff_dec_config_path") != str(checkpoint)
        or config.get("use_src_id_rotary_emb") is not True
        or type(config.get("max_sequence_length")) is not int
        or config.get("max_sequence_length") != 512
        or config.get("switch_dit_boundary") != 0.0
        or config.get("shift") != 5.0
        or config.get("scratch") is not False
        or config.get("ema_decay") is not None
        or config.get("use_unipc") is not True
    ):
        fail("owned Bernini renderer config closure differs")


def _activate_private_source_trees_v1(
    bernini_root: Path, veomni_root: Path
) -> None:
    roots = (str(bernini_root), str(veomni_root))
    if any(
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or stat.S_IMODE(os.lstat(path).st_mode) != 0o555
        for path in (bernini_root, veomni_root)
    ):
        fail("private source import roots differ")
    for root in roots:
        while root in sys.path:
            sys.path.remove(root)
    sys.path[0:0] = list(roots)


def _path_independent_model_closure_v1(
    *,
    checkpoint_snapshot: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    package_versions: Mapping[str, str],
    lora_installation_digest: str,
    peft_config_transition_digest: str,
    source_state_sha256: str,
    negative_condition_sha256: str,
    positive_condition_sha256: str,
) -> Mapping[str, Any]:
    expected_versions = {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "peft": PEFT_VERSION,
        "diffusers": DIFFUSERS_VERSION,
        "decord": DECORD_VERSION,
        "safetensors": SAFETENSORS_VERSION,
    }
    if dict(package_versions) != expected_versions:
        fail("owned model-closure package versions differ")
    checkpoint_files_digest = _sha256(
        checkpoint_snapshot.get("files_digest"),
        label="model closure checkpoint files digest",
    )
    source_files_digest = _sha256(
        source_snapshot.get("files_digest"),
        label="model closure vendor files digest",
    )
    result = {
        "schema_version": "bernini-full644-owned-model-closure-v1",
        "base_checkpoint_tree_sha256": BASE_CHECKPOINT_TREE_SHA256,
        "checkpoint_exact23_files_digest": checkpoint_files_digest,
        "vendor_committed_files_digest": source_files_digest,
        "bernini_commit": BERNINI_COMMIT,
        "veomni_commit": VEOMNI_COMMIT,
        "package_versions": expected_versions,
        "lora_installation_digest": _sha256(
            lora_installation_digest, label="LoRA installation digest"
        ),
        "peft_config_transition_digest": _sha256(
            peft_config_transition_digest,
            label="PEFT requested/canonical transition digest",
        ),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "source_state_sha256": _sha256(
            source_state_sha256, label="model closure source-state SHA"
        ),
        "negative_condition_sha256": _sha256(
            negative_condition_sha256, label="model closure negative-condition SHA"
        ),
        "positive_condition_sha256": _sha256(
            positive_condition_sha256, label="model closure positive-condition SHA"
        ),
    }
    # In particular, neither snapshot destination nor output root is selected.
    return result


def _build_owned_runtime_v1(
    *,
    bernini_root: Path,
    veomni_root: Path,
    checkpoint_root: Path,
    checkpoint_content_manifest: Path,
    output_root: Path,
    miopen_cache_binding: Mapping[str, Any],
) -> _OwnedRuntimeBundleV1:
    """Only constructor authorized to turn the frozen source row into tensors."""

    preinit_rank = _required_environment_integer_v1("RANK")
    _validate_rank_local_miopen_cache_binding_v1(
        miopen_cache_binding,
        expected_rank=preinit_rank,
        require_environment=True,
        require_physical=True,
    )
    local_closure = _load_owned_local_closure_v1()
    distributed_runtime = _require_held_local_module_v1("source_self_runtime")
    packed = _require_held_local_module_v1("packed_preservation_lora_v2")
    catalog, source, catalog_binding = _load_frozen_one_source_catalog_v1()

    contract = distributed_runtime.distributed_contract()
    device = distributed_runtime.initialise_distributed(contract)
    import torch
    import torch.distributed as dist

    rank0_result: list[Any] = [None]
    if contract.rank == 0:
        try:
            if not output_root.is_absolute() or output_root == Path("/"):
                fail("owned runtime output root differs")
            _mkdir_private_v1(output_root)
            checkpoint_destination = output_root / "base-checkpoint-exact23"
            source_destination = output_root / "vendor-source-no-pyc"
            rollout_root = output_root / "rollouts"
            os.mkdir(rollout_root, 0o700)
            source_snapshot = _snapshot_python_source_tree_v1(
                bernini_root=bernini_root,
                veomni_root=veomni_root,
                destination=source_destination,
            )
            checkpoint_snapshot = _snapshot_checkpoint_exact23_v1(
                checkpoint_root,
                checkpoint_content_manifest,
                checkpoint_destination,
            )
            geometry = _validate_private_checkpoint_v1(
                checkpoint_destination, checkpoint_snapshot
            )
            source_snapshot_metadata = _verify_projected_snapshot_metadata_v1(
                source_snapshot
            )
            checkpoint_snapshot_metadata = _verify_projected_snapshot_metadata_v1(
                checkpoint_snapshot
            )
            rank0_result[0] = {
                "ok": True,
                "bernini_revision": BERNINI_COMMIT,
                "veomni_revision": VEOMNI_COMMIT,
                "transformer_geometry": geometry,
                "source_snapshot": source_snapshot,
                "checkpoint_snapshot": checkpoint_snapshot,
                "source_snapshot_metadata": source_snapshot_metadata,
                "checkpoint_snapshot_metadata": checkpoint_snapshot_metadata,
                "rollout_root": str(rollout_root),
            }
        except Exception as error:
            rank0_result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
    dist.broadcast_object_list(rank0_result, src=0)
    snapshot_result = rank0_result[0]
    if not isinstance(snapshot_result, Mapping) or snapshot_result.get("ok") is not True:
        fail(f"owned runtime private snapshot failed: {snapshot_result}")
    source_snapshot = snapshot_result["source_snapshot"]
    checkpoint_snapshot = snapshot_result["checkpoint_snapshot"]
    source_snapshot_metadata = _verify_projected_snapshot_metadata_v1(source_snapshot)
    checkpoint_snapshot_metadata = _verify_projected_snapshot_metadata_v1(
        checkpoint_snapshot
    )
    if (
        source_snapshot_metadata != snapshot_result.get("source_snapshot_metadata")
        or checkpoint_snapshot_metadata
        != snapshot_result.get("checkpoint_snapshot_metadata")
    ):
        fail("shared immutable snapshot metadata replay differs")
    private_checkpoint = Path(checkpoint_snapshot["destination"])
    private_bernini = Path(source_snapshot["bernini_import_root"])
    private_veomni = Path(source_snapshot["veomni_import_root"])
    rollout_root = Path(snapshot_result["rollout_root"])
    if stat.S_IMODE(os.lstat(rollout_root).st_mode) != 0o700:
        fail("owned rollout directory mode differs")

    # There is no bytecode in the projected trees, and their directories are
    # read-only.  Reject any previously imported vendor module before putting
    # those exact source roots first.
    if any(
        name == "bernini"
        or name.startswith("bernini.")
        or name == "veomni"
        or name.startswith("veomni.")
        for name in sys.modules
    ):
        fail("vendor module cache is not empty before held-source activation")
    sys.dont_write_bytecode = True
    _activate_private_source_trees_v1(private_bernini, private_veomni)

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    import decord
    import diffusers
    from diffusers.models import AutoencoderKLWan
    from peft import LoraConfig, get_peft_model
    import peft
    import safetensors
    import transformers
    from transformers import AutoTokenizer

    versions = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "diffusers": str(diffusers.__version__),
        "decord": str(decord.__version__),
        "safetensors": str(safetensors.__version__),
    }
    if versions != {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "peft": PEFT_VERSION,
        "diffusers": DIFFUSERS_VERSION,
        "decord": DECORD_VERSION,
        "safetensors": SAFETENSORS_VERSION,
    }:
        fail("owned target-free runtime package versions differ")

    parallel = distributed_runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    gathered_miopen_cache: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_miopen_cache, dict(miopen_cache_binding),
        group=parallel.world_group,
    )
    miopen_cache_bindings = _validate_world8_miopen_cache_bindings_v1(
        gathered_miopen_cache, local_rank=contract.rank
    )
    miopen_backend_version = torch.backends.cudnn.version()
    if (
        type(miopen_backend_version) is not int
        or miopen_backend_version != MIOPEN_BACKEND_VERSION
    ):
        fail("owned target-free MIOpen backend version differs")
    torch.manual_seed(OWNED_FACTORY_SEED)
    torch.cuda.manual_seed_all(OWNED_FACTORY_SEED)

    vae_rows = [
        row for row in checkpoint_snapshot["files"] if row["path"].startswith("vae/")
    ]
    vae_config_rows = [row for row in vae_rows if row["path"] == "vae/config.json"]
    if not vae_rows or len(vae_config_rows) != 1:
        fail("private checkpoint VAE inventory differs")
    vae_authority = {
        "schema_version": "bernini-full644-owned-vae-authority-v1",
        "base_checkpoint_tree_sha256": BASE_CHECKPOINT_TREE_SHA256,
        "checkpoint_content_manifest_sha256": BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "checkpoint_snapshot_digest": checkpoint_snapshot["snapshot_digest"],
        "vae_file_inventory_digest": object_sha256(vae_rows),
        "vae_config_sha256": vae_config_rows[0]["sha256"],
    }
    def construct_rank0_vae_state() -> tuple[Any, Mapping[str, Any]]:
        pixels: Any = None
        vae: Any = None
        local_state: Any = None
        local_decode: Any = None
        trim_receipt: Any = None
        try:
            pixels, local_decode = _decode_owned_source_row_v1(source)
            vae = AutoencoderKLWan.from_pretrained(
                str(private_checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False).to(device)
            with torch.no_grad():
                local_state = _vae_encode(
                    vae, pixels.to(device=device)
                ).float().contiguous()
            bucket_hw = local_decode["source_derived_bucket_hw"]
            if (
                tuple(local_state.shape)
                != (1, 16, 21, int(bucket_hw[0]) // 8, int(bucket_hw[1]) // 8)
                or local_state.requires_grad
                or local_state.device != device
                or not bool(torch.isfinite(local_state).all().item())
            ):
                fail("owned frozen VAE source state differs")
        finally:
            # Moving the VAE back to CPU rematerializes its full checkpoint and
            # defeats host-RAM serialization.  Rank zero drops it directly
            # before publishing the encoded source tensor.
            vae = None
            pixels = None
            gc.collect()
            trim_receipt = _trim_host_allocator_v1()
            torch.cuda.empty_cache()
        if local_state is None or local_decode is None:
            fail("rank0 VAE source construction returned no state")
        descriptor_unsigned = {
            "schema_version": "bernini-full644-rank0-vae-source-broadcast-v1",
            "producer_rank": 0,
            "source_row_id": source.row_id,
            "source_video_sha256": source.source_video_sha256,
            "instruction_sha256": source.instruction_sha256,
            "source_row_digest": source.row_digest,
            "catalog_sha256": catalog_binding["catalog_sha256"],
            "catalog_digest": catalog_binding["catalog_digest"],
            "catalog_binding_digest": catalog_binding["binding_digest"],
            "source_decode": dict(local_decode),
            "source_decode_digest": local_decode["decode_digest"],
            "source_state_shape": [int(item) for item in local_state.shape],
            "source_state_numel": int(local_state.numel()),
            "source_state_dtype": "torch.float32",
            "source_state_layout": "torch.strided",
            "source_state_contiguous": bool(local_state.is_contiguous()),
            "source_state_requires_grad": bool(local_state.requires_grad),
            "source_state_sha256": tensor_sha256(local_state),
            "vae_authority": dict(vae_authority),
            "vae_authority_digest": object_sha256(vae_authority),
            "rank0_only_source_decode_and_vae_encode": True,
            "vae_released_without_cpu_rematerialization": True,
            "host_allocator_trim": trim_receipt,
        }
        descriptor = {
            **descriptor_unsigned,
            "descriptor_digest": object_sha256(descriptor_unsigned),
        }
        _validate_rank0_vae_source_descriptor_v1(
            descriptor,
            expected_source_row_id=source.row_id,
            expected_source_video_sha256=source.source_video_sha256,
            expected_instruction_sha256=source.instruction_sha256,
            expected_source_row_digest=source.row_digest,
            expected_catalog_binding=catalog_binding,
            expected_vae_authority=vae_authority,
        )
        return {"source_state": local_state, "descriptor": descriptor}, descriptor

    def validate_rank0_vae_payload(payload: Any) -> None:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"source_state", "descriptor"}
            or type(payload["source_state"]) is not torch.Tensor
            or payload["source_state"].device != device
            or int(payload["source_state"].device.index) != 0
            or payload["source_state"].dtype != torch.float32
            or not payload["source_state"].is_contiguous()
            or payload["source_state"].requires_grad
        ):
            fail("rank0 VAE source payload differs")
        descriptor = _validate_rank0_vae_source_descriptor_v1(
            payload["descriptor"],
            expected_source_row_id=source.row_id,
            expected_source_video_sha256=source.source_video_sha256,
            expected_instruction_sha256=source.instruction_sha256,
            expected_source_row_digest=source.row_digest,
            expected_catalog_binding=catalog_binding,
            expected_vae_authority=vae_authority,
        )
        if tensor_sha256(payload["source_state"]) != descriptor["source_state_sha256"]:
            fail("rank0 VAE source payload tensor differs")

    rank0_payload, rank0_vae_status = _run_rank0_vae_source_constructor_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        constructor=construct_rank0_vae_state,
        payload_validator=validate_rank0_vae_payload,
    )
    source_state, source_decode, rank0_vae_source_broadcast = (
        _broadcast_rank0_vae_source_state_v1(
            dist=dist,
            torch_module=torch,
            group=parallel.world_group,
            rank=contract.rank,
            device=device,
            rank0_payload=rank0_payload,
            expected_source_row_id=source.row_id,
            expected_source_video_sha256=source.source_video_sha256,
            expected_instruction_sha256=source.instruction_sha256,
            expected_source_row_digest=source.row_digest,
            expected_catalog_binding=catalog_binding,
            expected_vae_authority=vae_authority,
        )
    )
    source_state_sha = rank0_vae_source_broadcast["descriptor"][
        "source_state_sha256"
    ]
    rank0_payload = None
    gc.collect()
    torch.cuda.empty_cache()

    model: Any = None
    specs: Any = None
    lora_installation: Any = None
    peft_config_transition: Any = None
    serialized_renderer_construction = []

    def construct_local_renderer() -> tuple[Any, Mapping[str, Any]]:
        renderer: Any = None
        local_model: Any = None
        local_specs: Any = None
        try:
            torch.manual_seed(OWNED_FACTORY_SEED)
            torch.cuda.manual_seed_all(OWNED_FACTORY_SEED)
            config = BerniniRendererConfig.from_pretrained(
                str(private_bernini / "configs/bernini_renderer_wan21_1p3b"),
                local_files_only=True,
                **{
                    **_renderer_config_overrides_v1(private_checkpoint),
                    "shift": 5.0,
                    "use_unipc": True,
                },
            )
            config.dtype = torch.bfloat16
            _validate_renderer_config_mapping_v1(
                config.to_dict(), private_checkpoint
            )
            renderer = BerniniRendererModel(config)
            renderer.eval().requires_grad_(False)
            local_specs = packed.select_projection_specs(renderer, "all-attention")
            target_modules = [item.name for item in local_specs]
            if (
                target_modules != sorted(target_modules)
                or len(target_modules) != LORA_AFFINES
                or set(target_modules) != _target_free_requested_lora_targets_v1()
            ):
                fail("owned factory exact240 LoRA targets differ")
            requested_lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
            )
            requested_peft_receipt = _validate_target_free_peft_config_v1(
                requested_lora_config,
                expected_targets=_target_free_requested_lora_targets_v1(),
                target_modules_contract=_PEFT_REQUESTED_TARGET_CONTRACT,
            )
            local_model = get_peft_model(
                renderer,
                requested_lora_config,
            )
            local_model.to(device).eval()
            local_peft_config = getattr(local_model, "peft_config", None)
            if (
                not isinstance(local_peft_config, Mapping)
                or set(local_peft_config) != {"default"}
            ):
                fail("owned factory postinstall default PEFT adapter differs")
            canonical_peft_receipt = _validate_target_free_peft_config_v1(
                local_peft_config["default"],
                expected_targets=set(_PEFT_CANONICAL_TARGET_MODULES),
                target_modules_contract=_PEFT_CANONICAL_TARGET_CONTRACT,
            )
            local_installation = packed.validate_lora_installation(
                local_model, local_specs
            )
            local_peft_transition = _bind_target_free_peft_transition_v1(
                requested_receipt=requested_peft_receipt,
                canonical_receipt=canonical_peft_receipt,
                lora_installation_digest=local_installation["digest"],
            )
            tensors = tuple(local_model.parameters()) + tuple(local_model.buffers())
            if not tensors or any(tensor.device != device for tensor in tensors):
                fail("serialized renderer has nonlocal parameter/buffer")
            evidence = {
                "schema_version": "bernini-full644-serialized-renderer-lora-v1",
                "owner_rank": contract.rank,
                "device_index": int(device.index),
                "lora_affine_count": len(target_modules),
                "lora_installation_digest": local_installation["digest"],
                "peft_requested_exact240_config_digest": requested_peft_receipt[
                    "config_digest"
                ],
                "peft_canonical_exact4_config_digest": canonical_peft_receipt[
                    "config_digest"
                ],
                "peft_config_transition_digest": local_peft_transition[
                    "transition_digest"
                ],
                "all_parameters_and_buffers_on_owner_device": True,
            }
            payload = {
                "model": local_model,
                "specs": local_specs,
                "lora_installation": local_installation,
                "peft_config_transition": local_peft_transition,
            }
            renderer = None
            gc.collect()
            trim_receipt = _trim_host_allocator_v1()
            torch.cuda.empty_cache()
            evidence["host_allocator_trim"] = trim_receipt
            return payload, evidence
        except Exception:
            local_model = None
            renderer = None
            local_specs = None
            gc.collect()
            _trim_host_allocator_v1()
            torch.cuda.empty_cache()
            raise

    def validate_local_renderer_payload(payload: Any) -> None:
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {"model", "specs", "lora_installation", "peft_config_transition"}
            or payload["model"] is None
            or payload["specs"] is None
            or not isinstance(payload["lora_installation"], Mapping)
            or type(payload["lora_installation"].get("digest")) is not str
            or not _SHA256.fullmatch(payload["lora_installation"]["digest"])
            or not isinstance(payload["peft_config_transition"], Mapping)
            or type(payload["peft_config_transition"].get("transition_digest"))
            is not str
            or not _SHA256.fullmatch(
                payload["peft_config_transition"]["transition_digest"]
            )
        ):
            fail("serialized renderer owner payload differs")

    for owner in range(WORLD_SIZE):
        payload, status = _run_world8_serial_owner_round_v1(
            dist=dist,
            group=parallel.world_group,
            rank=contract.rank,
            owner_rank=owner,
            phase="renderer_lora_construct",
            constructor=construct_local_renderer,
            payload_validator=validate_local_renderer_payload,
        )
        serialized_renderer_construction.append(status)
        if contract.rank == owner:
            model = payload["model"]
            specs = payload["specs"]
            lora_installation = payload["lora_installation"]
            peft_config_transition = payload["peft_config_transition"]
    if (
        model is None
        or specs is None
        or lora_installation is None
        or peft_config_transition is None
    ):
        fail("serialized owned model construction did not run locally")
    base_renderer = model.get_base_model()
    named_trainable = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    initial_trainable_sha = distributed_runtime.synchronize_initial_parameters(
        named_trainable, parallel.world_group, expected_count=WORLD_SIZE
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(private_checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    def encode_condition(text: str, label: str) -> tuple[Any, Mapping[str, Any]]:
        tokenized = distributed_runtime.tokenize_generic_instruction(
            tokenizer, text, device
        )
        with torch.inference_mode():
            output = base_renderer.get_t5_text_embeddings(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                tokenized["t5_input_lens"],
            )
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            fail(f"owned T5 {label} return ABI differs")
        text_lens, embeddings = output
        if (
            type(text_lens) is not list
            or text_lens != [512]
            or type(embeddings) is not torch.Tensor
            or tuple(embeddings.shape) != (1, 512, 4096)
            or embeddings.dtype != torch.bfloat16
            or embeddings.device != device
            or embeddings.requires_grad
            or not bool(torch.isfinite(embeddings).all().item())
        ):
            fail(f"owned T5 {label} embedding ABI differs")
        cloned = embeddings.detach().clone(memory_format=torch.contiguous_format)
        receipt = {
            "role": label,
            "utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_lens_container": "python_list",
            "text_lens_values": [512],
            "embedding_sha256": tensor_sha256(cloned),
        }
        return cloned, receipt

    negative_condition, negative_receipt = encode_condition(
        DEFAULT_NEGATIVE_PROMPT, "negative"
    )
    positive_condition, positive_receipt = encode_condition(
        source.instruction, "source_instruction"
    )
    base_renderer.t5_text_encoder = None
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("owned frozen T5 was not released")

    model_closure = _path_independent_model_closure_v1(
        checkpoint_snapshot=checkpoint_snapshot,
        source_snapshot=source_snapshot,
        package_versions=versions,
        lora_installation_digest=lora_installation["digest"],
        peft_config_transition_digest=peft_config_transition["transition_digest"],
        source_state_sha256=source_state_sha,
        negative_condition_sha256=negative_receipt["embedding_sha256"],
        positive_condition_sha256=positive_receipt["embedding_sha256"],
    )
    model_closure_sha = object_sha256(model_closure)
    runtime = BerniniExact40PolicyV1(
        renderer=model,
        source_state=source_state,
        negative_condition=negative_condition,
        positive_condition=positive_condition,
        source_row_id=source.row_id,
        source_video_sha256=source.source_video_sha256,
        instruction_sha256=source.instruction_sha256,
        base_model_sha256=BASE_CHECKPOINT_TREE_SHA256,
        model_closure_sha256=model_closure_sha,
        peft_config_transition_receipt=peft_config_transition,
        parallel=parallel,
        _owned_factory_token=_OWNED_RUNTIME_TOKEN,
    )
    if runtime.trainable_parameter_sha256() != initial_trainable_sha:
        fail("owned runtime construction changed synchronized LoRA bytes")
    factory = {
        "schema_version": "bernini-full644-owned-one-source-runtime-v1",
        "source_catalog": catalog_binding,
        "source_decode": source_decode,
        "local_execution_closure": local_closure,
        "source_snapshot_digest": source_snapshot["snapshot_digest"],
        "checkpoint_snapshot_digest": checkpoint_snapshot["snapshot_digest"],
        "shared_snapshot_metadata_replay": {
            "source": source_snapshot_metadata,
            "checkpoint": checkpoint_snapshot_metadata,
            "rank0_single_snapshot_writer": True,
            "nonowner_content_bytes_reread": False,
        },
        "rank_local_miopen_cache": {
            "backend_version": miopen_backend_version,
            "bindings": miopen_cache_bindings,
            "exact8_distinct_fresh_writable_roots": True,
            "configured_before_torch_import": True,
            "user_find_db_and_kernel_cache_isolated": True,
            "tmp_lock_root_is_rank_local": True,
        },
        "rank0_vae_source_broadcast": {
            "rank0_construction_status": rank0_vae_status,
            **rank0_vae_source_broadcast,
        },
        "serialized_renderer_construction": serialized_renderer_construction,
        "vae_authority": vae_authority,
        "negative_condition": negative_receipt,
        "positive_condition": positive_receipt,
        "model_closure": model_closure,
        "model_closure_sha256": model_closure_sha,
        "peft_config_three_layer_closure": runtime.peft_config_receipt,
        "trainable_inventory_sha256": runtime.trainable_inventory_sha256,
        "initial_trainable_parameter_sha256": initial_trainable_sha,
        "initial_policy_sha256": runtime.behavior_policy_sha256(),
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "source_only_input": True,
        "target_media_read_count": 0,
        "external_supervision_tensor_read_count": 0,
    }
    factory = {**factory, "factory_digest": object_sha256(factory)}
    gathered_factory: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_factory, factory, group=parallel.world_group)
    if gathered_factory != [factory] * WORLD_SIZE:
        fail("WORLD8 owned factory closure differs")
    factory_binding: list[Any] = [None]
    factory_path = output_root / "owned_factory_receipt.json"
    if contract.rank == 0:
        factory_binding[0] = write_create_only(
            factory_path, canonical_json_bytes(factory)
        )
    dist.broadcast_object_list(factory_binding, src=0)
    if not isinstance(factory_binding[0], Mapping):
        fail("owned factory receipt publication differs")
    factory_raw = read_stable_file(
        factory_path,
        expected_sha256=factory_binding[0]["sha256"],
        expected_mode=0o444,
        label="owned factory receipt",
    )
    if _strict_json(factory_raw, label="owned factory receipt") != factory:
        fail("owned factory receipt reload differs")
    runtime._owned_output_root = output_root
    runtime._owned_factory_digest = factory["factory_digest"]
    return _OwnedRuntimeBundleV1(
        runtime,
        source,
        catalog,
        private_checkpoint,
        vae_authority,
        output_root,
        rollout_root,
        factory,
    )


def _aggregate_rollout_stage_world8_v1(
    gathered: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Close WORLD8 semantics while truthfully aggregating rank-local peaks."""

    if type(gathered) not in (list, tuple) or len(gathered) != WORLD_SIZE:
        fail("WORLD8 rollout-stage exact8 rows differ")
    closed_rows: list[Mapping[str, Any]] = []
    for rank, value in enumerate(gathered):
        row = _closed(
            value,
            _ROLLOUT_STAGE_LOCAL_FIELDS,
            label=f"rollout stage rank {rank}",
        )
        if (
            type(row["world_rank"]) is not int
            or row["world_rank"] != rank
            or type(row["dp_arm"]) is not int
            or row["dp_arm"] != rank // SP_SIZE
            or type(row["sp_rank"]) is not int
            or row["sp_rank"] != rank % SP_SIZE
        ):
            fail("WORLD8 rollout-stage placement differs")
        _verify_digest(row, "row_digest", label=f"rollout stage rank {rank}")
        peak = row["peak_memory_allocated_bytes"]
        total = row["total_device_memory_bytes"]
        if (
            type(peak) is not int
            or type(total) is not int
            or total <= 0
            or peak <= 0
            or peak > total
        ):
            fail(f"rollout stage rank {rank} memory evidence differs")
        closed_rows.append(row)

    arm_rows: list[Mapping[str, Any]] = []
    for arm in range(DP_SIZE):
        rows = closed_rows[arm * SP_SIZE : (arm + 1) * SP_SIZE]
        semantic_projections = [
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "world_rank",
                    "sp_rank",
                    "peak_memory_allocated_bytes",
                    "row_digest",
                )
            }
            for row in rows
        ]
        semantic_bytes = [
            canonical_json_bytes(projection) for projection in semantic_projections
        ]
        if semantic_bytes != [semantic_bytes[0]] * SP_SIZE:
            fail(f"SP4 rollout-stage arm{arm} differs")
        peak = max(int(row["peak_memory_allocated_bytes"]) for row in rows)
        arm_row = {
            key: (
                peak if key == "peak_memory_allocated_bytes" else value
            )
            for key, value in rows[0].items()
            if key not in ("world_rank", "sp_rank", "row_digest")
        }
        arm_rows.append(arm_row)
    return arm_rows


def _run_one_source_rollout_stage_v1(
    bundle: _OwnedRuntimeBundleV1,
) -> Mapping[str, Any]:
    """Record and decode exact one stochastic endpoint per DP arm."""

    import torch
    import torch.distributed as dist

    runtime = bundle.runtime
    instruction_path = bundle.rollout_root / "one_source_instruction.utf8"
    instruction_binding: list[Any] = [None]
    if runtime.parallel.contract.rank == 0:
        payload = bundle.source.instruction.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != bundle.source.instruction_sha256:
            fail("owned rollout instruction bytes differ from catalogue")
        instruction_binding[0] = write_create_only(instruction_path, payload)
    dist.broadcast_object_list(
        instruction_binding, src=0, group=runtime.parallel.world_group
    )
    if not isinstance(instruction_binding[0], Mapping):
        fail("owned rollout instruction publication differs")
    instruction_raw = read_stable_file(
        instruction_path,
        expected_sha256=bundle.source.instruction_sha256,
        expected_mode=0o444,
        label="owned rollout instruction",
    )
    if instruction_raw.decode("utf-8", errors="strict") != bundle.source.instruction:
        fail("owned rollout instruction reload differs")
    torch.cuda.reset_peak_memory_stats(runtime.device)
    rollout_id = f"full644-{ONE_SOURCE_ROW_ID}-r0-arm{runtime.dp_arm}"
    recorded = record_exact40_trajectory_v1(
        runtime,
        rollout_id=rollout_id,
        round_index=PILOT_ROUND_INDEX,
        rollout_seed=PILOT_ROLLOUT_SEEDS[runtime.dp_arm],
        output_directory=bundle.rollout_root,
    )
    decoded = decode_and_seal_recorded_trajectory_v1(
        runtime,
        recorded,
        checkpoint_root=bundle.checkpoint_root,
        vae_authority=bundle.vae_authority,
        output_directory=bundle.rollout_root,
    )
    peak_bytes = int(torch.cuda.max_memory_allocated(runtime.device))
    total_bytes = int(torch.cuda.get_device_properties(runtime.device).total_memory)
    local = {
        "world_rank": runtime.parallel.contract.rank,
        "dp_arm": runtime.dp_arm,
        "sp_rank": runtime.sp_rank,
        "rollout_id": rollout_id,
        "rollout_seed": PILOT_ROLLOUT_SEEDS[runtime.dp_arm],
        "behavior_policy_sha256": runtime.behavior_policy_sha256(),
        "trajectory_receipt_path": str(recorded.receipt.path),
        "trajectory_receipt_sha256": recorded.receipt.sha256,
        "trajectory_receipt_digest": recorded.receipt.value["receipt_digest"],
        "trajectory_artifact_sha256": recorded.receipt.value["artifact_sha256"],
        "terminal_state_sha256": recorded.receipt.value["terminal_state_sha256"],
        "decoded_rollout_receipt_path": str(decoded.path),
        "decoded_rollout_receipt_sha256": decoded.sha256,
        "decoded_rollout_receipt_digest": decoded.value["receipt_digest"],
        "candidate_media_path": decoded.value["candidate_media_path"],
        "candidate_media_sha256": decoded.value["candidate_media_sha256"],
        "candidate_full_decode_tree_digest": decoded.value["full_decode_tree_digest"],
        "candidate_exact81_25fps": True,
        "peak_memory_allocated_bytes": peak_bytes,
        "total_device_memory_bytes": total_bytes,
    }
    local = {**local, "row_digest": object_sha256(local)}
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local, group=runtime.parallel.world_group)
    arm_rows = _aggregate_rollout_stage_world8_v1(gathered)
    if (
        {row["dp_arm"] for row in arm_rows} != set(range(DP_SIZE))
        or [row["rollout_seed"] for row in arm_rows] != list(PILOT_ROLLOUT_SEEDS)
        or len({row["trajectory_receipt_sha256"] for row in arm_rows}) != DP_SIZE
        or len({row["decoded_rollout_receipt_sha256"] for row in arm_rows}) != DP_SIZE
        or len({row["candidate_media_sha256"] for row in arm_rows}) != DP_SIZE
        or len({row["behavior_policy_sha256"] for row in arm_rows}) != 1
    ):
        fail("DP2 exact2 rollout-stage closure differs")
    receipt = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ONE_SOURCE_ONE_UPDATE_PREFLIGHT_ROLLOUT_COMPLETE",
        "scope": "ONE_SOURCE_ONE_UPDATE_PREFLIGHT",
        "full644_coverage_count": 1,
        "source_row_id": bundle.source.row_id,
        "source_video_path": str(bundle.source.source_video_path),
        "source_video_sha256": bundle.source.source_video_sha256,
        "instruction_path": str(instruction_path),
        "instruction_sha256": bundle.source.instruction_sha256,
        "instruction_size_bytes": instruction_binding[0]["size_bytes"],
        "instruction_mode_octal": instruction_binding[0]["mode_octal"],
        "source_catalog_sha256": FULL644_CATALOG_SHA256,
        "source_catalog_digest": FULL644_CATALOG_DIGEST,
        "owned_factory_digest": bundle.factory_receipt["factory_digest"],
        "behavior_policy_sha256": arm_rows[0]["behavior_policy_sha256"],
        "round_index": PILOT_ROUND_INDEX,
        "rollout_count": DP_SIZE,
        "rollouts": arm_rows,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "exact40_stochastic_current_policy": True,
        "qwen_verdicts_required_before_update": True,
        "qwen_verifier_release_sha256": QWEN_VERIFIER_SOURCE_SHA256,
        "qwen_model_closure_sha256": QWEN_MODEL_CLOSURE_SHA256,
        "terminal_stdout_requires_world8_postpublication_reload_ack": True,
        "source_only_input": True,
        "paired_reference_read_count": 0,
        "external_velocity_read_count": 0,
        "engineering_only": True,
        "scientific_result_claimed": False,
    }
    receipt = {**receipt, "receipt_digest": object_sha256(receipt)}
    binding: list[Any] = [None]
    path = bundle.output_root / "one_source_rollout_preflight_receipt.json"
    if runtime.parallel.contract.rank == 0:
        binding[0] = write_create_only(path, canonical_json_bytes(receipt))
    dist.broadcast_object_list(binding, src=0, group=runtime.parallel.world_group)
    if not isinstance(binding[0], Mapping):
        fail("rollout-stage receipt publication differs")
    raw = read_stable_file(
        path,
        expected_sha256=binding[0]["sha256"],
        expected_mode=0o444,
        label="rollout-stage receipt",
    )
    if _strict_json(raw, label="rollout-stage receipt") != receipt:
        fail("rollout-stage receipt reload differs")
    local_ack = {
        "world_rank": runtime.parallel.contract.rank,
        "receipt_sha256": binding[0]["sha256"],
        "receipt_digest": receipt["receipt_digest"],
        "postpublication_reload_verified": True,
    }
    gathered_ack: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_ack, local_ack, group=runtime.parallel.world_group
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("world_rank") != rank
        or row.get("receipt_sha256") != binding[0]["sha256"]
        or row.get("receipt_digest") != receipt["receipt_digest"]
        or row.get("postpublication_reload_verified") is not True
        for rank, row in enumerate(gathered_ack)
    ):
        fail("WORLD8 rollout-stage postpublication reload acknowledgement differs")
    return {
        "receipt": receipt,
        "binding": dict(binding[0]),
        "world8_postpublication_reload_ack": gathered_ack,
    }


def _publish_one_source_update_stage_v1(
    bundle: _OwnedRuntimeBundleV1,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    import torch.distributed as dist

    runtime = bundle.runtime
    _verify_digest(result, "receipt_digest", label="local update result")
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(result), group=runtime.parallel.world_group)
    statuses = []
    for rank, row in enumerate(gathered):
        if not isinstance(row, Mapping) or row.get("schema_version") != UPDATE_SCHEMA:
            fail(f"WORLD8 update result rank {rank} differs")
        _verify_digest(row, "receipt_digest", label=f"update result rank {rank}")
        statuses.append(row["status"])
    if len(set(statuses)) != 1:
        fail("WORLD8 update/zero branch diverged")
    status = statuses[0]
    if status == "ENGINEERING_ONE_UPDATE_COMPLETE":
        arm_results = []
        for arm in range(DP_SIZE):
            rows = gathered[arm * SP_SIZE : (arm + 1) * SP_SIZE]
            if rows != [rows[0]] * SP_SIZE:
                fail(f"SP4 update result arm{arm} differs")
            arm_results.append(rows[0])
        if (
            {row["local_endpoint_role"] for row in arm_results}
            != {"chosen", "rejected"}
            or len({row["trainable_parameter_digest_after"] for row in arm_results}) != 1
            or len({row["policy_digest_after"] for row in arm_results}) != 1
            or len({row["checkpoint"]["tree_digest"] for row in arm_results}) != 1
        ):
            fail("DP2 successful update result closure differs")
    elif status in (
        "ZERO_UPDATE_NO_PREFERENCE_PAIR",
        "ZERO_UPDATE_VERIFIER_UNDETERMINED_OR_INELIGIBLE",
    ):
        if gathered != [gathered[0]] * WORLD_SIZE:
            fail("WORLD8 zero-update result differs")
        arm_results = [gathered[0]]
    else:
        fail("one-source update-stage terminal status differs")
    receipt = {
        "schema_version": "bernini-full644-one-source-update-stage-v1",
        "status": status,
        "scope": "ONE_SOURCE_ONE_UPDATE_PREFLIGHT",
        "full644_coverage_count": 1,
        "source_row_id": bundle.source.row_id,
        "source_video_sha256": bundle.source.source_video_sha256,
        "instruction_sha256": bundle.source.instruction_sha256,
        "owned_factory_digest": bundle.factory_receipt["factory_digest"],
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "world8_result_digests": [row["receipt_digest"] for row in gathered],
        "arm_results": arm_results,
        "terminal_stdout_requires_world8_postpublication_reload_ack": True,
        "source_only_input": True,
        "paired_reference_read_count": 0,
        "external_velocity_read_count": 0,
        "engineering_only": True,
        "scientific_result_claimed": False,
    }
    receipt = {**receipt, "receipt_digest": object_sha256(receipt)}
    binding: list[Any] = [None]
    path = bundle.output_root / "one_source_update_receipt.json"
    if runtime.parallel.contract.rank == 0:
        binding[0] = write_create_only(path, canonical_json_bytes(receipt))
    dist.broadcast_object_list(binding, src=0, group=runtime.parallel.world_group)
    if not isinstance(binding[0], Mapping):
        fail("update-stage receipt publication differs")
    raw = read_stable_file(
        path,
        expected_sha256=binding[0]["sha256"],
        expected_mode=0o444,
        label="update-stage receipt",
    )
    if _strict_json(raw, label="update-stage receipt") != receipt:
        fail("update-stage receipt reload differs")
    local_ack = {
        "world_rank": runtime.parallel.contract.rank,
        "receipt_sha256": binding[0]["sha256"],
        "receipt_digest": receipt["receipt_digest"],
        "postpublication_reload_verified": True,
    }
    gathered_ack: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_ack, local_ack, group=runtime.parallel.world_group
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("world_rank") != rank
        or row.get("receipt_sha256") != binding[0]["sha256"]
        or row.get("receipt_digest") != receipt["receipt_digest"]
        or row.get("postpublication_reload_verified") is not True
        for rank, row in enumerate(gathered_ack)
    ):
        fail("WORLD8 update-stage postpublication reload acknowledgement differs")
    return {
        "receipt": receipt,
        "binding": dict(binding[0]),
        "world8_postpublication_reload_ack": gathered_ack,
    }


def _cli_existing_directory_v1(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        fail(f"{label} must be an absolute directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(f"{label} is unavailable: {error}") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        fail(f"{label} canonical directory differs")
    return path


def _cli_existing_file_v1(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        fail(f"{label} must be an absolute file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TargetFreeBerniniRuntimeError(f"{label} is unavailable: {error}") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        fail(f"{label} canonical file differs")
    return path


def _cli_fresh_output_v1(value: str) -> Path:
    """Validate a rank-local output candidate; rank 0 performs atomic mkdir.

    Do not test candidate existence here: torchrun ranks enter Python at
    slightly different times, so rank 0 may have created the directory before
    another rank reaches argument validation.  ``_mkdir_private_v1`` inside
    the rank-0 transaction remains the sole freshness authority.
    """

    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or ".." in path.parts
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        fail("output must be one absolute child of an existing plain directory")
    return path


def build_argument_parser_v1() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled-environment full644 target-free exact40 engineering pilot"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("rollout", "update"):
        child = subparsers.add_parser(stage)
        child.add_argument("--bernini-root", required=True)
        child.add_argument("--veomni-root", required=True)
        child.add_argument("--checkpoint", required=True)
        child.add_argument("--checkpoint-content-manifest", required=True)
        child.add_argument("--output", required=True)
        if stage == "update":
            child.add_argument("--preference-set", required=True)
            child.add_argument("--expected-preference-set-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser_v1().parse_args(argv)
    try:
        bernini_root = _cli_existing_directory_v1(
            args.bernini_root, label="Bernini root"
        )
        veomni_root = _cli_existing_directory_v1(
            args.veomni_root, label="VeOmni root"
        )
        checkpoint = _cli_existing_directory_v1(
            args.checkpoint, label="base checkpoint"
        )
        checkpoint_manifest = _cli_existing_file_v1(
            args.checkpoint_content_manifest,
            label="base checkpoint content manifest",
        )
        output = _cli_fresh_output_v1(args.output)
        miopen_cache_binding = _prepare_rank_local_miopen_cache_v1(
            output_path=output
        )
        if args.stage == "update":
            preference_path = _cli_existing_file_v1(
                args.preference_set, label="preference set"
            )
            preference_sha = _sha256(
                args.expected_preference_set_sha256,
                label="expected preference-set SHA",
            )
        bundle = _build_owned_runtime_v1(
            bernini_root=bernini_root,
            veomni_root=veomni_root,
            checkpoint_root=checkpoint,
            checkpoint_content_manifest=checkpoint_manifest,
            output_root=output,
            miopen_cache_binding=miopen_cache_binding,
        )
        if args.stage == "rollout":
            published = _run_one_source_rollout_stage_v1(bundle)
        elif args.stage == "update":
            local_result = _engineering_one_update_from_paths_v1(
                bundle.runtime,
                preference_set_path=preference_path,
                expected_preference_set_sha256=preference_sha,
            )
            published = _publish_one_source_update_stage_v1(bundle, local_result)
        else:  # pragma: no cover - argparse exact choices
            fail("unknown pilot stage")
        if bundle.runtime.parallel.contract.rank == 0:
            print(
                canonical_json_bytes(
                    {
                        "status": published["receipt"]["status"],
                        "scope": "ONE_SOURCE_ONE_UPDATE_PREFLIGHT",
                        "full644_coverage_count": 1,
                        "receipt_path": published["binding"]["path"],
                        "receipt_sha256": published["binding"]["sha256"],
                        "receipt_digest": published["receipt"]["receipt_digest"],
                        "world8_postpublication_reload_ack": (
                            len(published["world8_postpublication_reload_ack"])
                            == WORLD_SIZE
                        ),
                        "engineering_only": True,
                        "scientific_result_claimed": False,
                    }
                ).decode("ascii"),
                flush=True,
            )
        return 0
    except Exception as error:
        print(
            f"FULL644_TARGET_FREE_BERNINI_RUNTIME_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
