#!/usr/bin/env python3
"""Run a source video and a generated pure-T2V action anchor through SGA/ANC.

This is a training-free mechanism canary.  The source video supplies the clean
edit state and the source-side velocity field.  The independently generated
pure-T2V video is VAE encoded and queried online at every solver
step/candidate.  Its post-RoPE visual Q/K tensors are then hard-routed into the
target conditional forward while the target forward keeps its own V tensors.
The outer clean edit state, multi-noise candidate projection and correlated
noise chain follow the DynaEdit-style SGA/ANC controller.

The runner intentionally supports one four-GPU Ulysses arm per process group.
It performs no optimization and refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
from datetime import timedelta
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import anchor_sga_anc_controller as controller  # noqa: E402
import infer_lora as legacy  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_aligned_controller_oracle as source_audit  # noqa: E402


SCHEMA_VERSION = "bernini-pure-t2v-anchor-sga-anc-event-canary-v47"
FRAME_COUNT = 81
FPS = 25
ULYSSES_SIZE = 4
NOOP_INSTRUCTION = source_audit.NOOP_INSTRUCTION
PROMPT_EMBEDDING_SHAPE = (1, 512, 4096)
PROMPT_BANK_NAMES = (
    "action",
    "noop",
    "anchor",
    "anchor_noop",
    "source_t2v",
    "target_t2v",
    "negative",
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SUPPORTED_ATTENTION_TRAINING_OBJECTIVES = (
    "target_fm",
    "paired_delta_fm",
    "real_source_teacher_delta",
    "real_source_routed_teacher_delta",
    "real_source_target_owned_routed_teacher_delta_v14r2",
)
SUPPORTED_ATTENTION_ROUTE_OPERATORS = (
    "cross_sparse",
    "self_temporal_kernel",
    "self_target_gated_kernel25",
    "self_correspondence_kernel25",
    "self_target_owned_temporal_kernel_v14r2",
    "self_target_owned_activity_kernel10_v14r2",
    "self_target_owned_activity_kernel25_v14r2",
)
V14R2_ROUTE_TO_TRANSPORT = {
    "self_target_owned_temporal_kernel_v14r2": (
        "self_target_owned_temporal_kernel_attn_output_v14r2"
    ),
    "self_target_owned_activity_kernel10_v14r2": (
        "self_target_owned_activity_kernel10_attn_output_v14r2"
    ),
    "self_target_owned_activity_kernel25_v14r2": (
        "self_target_owned_activity_kernel25_attn_output_v14r2"
    ),
}


class AnchorEventInferenceError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise AnchorEventInferenceError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if not stat.S_ISREG(resolved.lstat().st_mode):
        raise AnchorEventInferenceError(f"{label} must be a plain file")
    return resolved


def _plain_directory(value: str, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise AnchorEventInferenceError(
            f"{label} must be an absolute non-symlink directory"
        )
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise AnchorEventInferenceError(f"{label} must be a directory")
    return resolved


def _attention_lora_checkpoint(
    value: str,
    *,
    expected_global_step: int = 0,
    expected_training_objective: str = "",
    expected_route_operator: str = "",
    expected_adapter_model_sha256: str = "",
    expected_adapter_config_sha256: str = "",
    expected_receipt_sha256: str = "",
) -> Optional[dict[str, Any]]:
    """Authenticate an optional trained all-attention LoRA.

    The historical dense-flow runs jointly trained this exact LoRA scope.  The
    flow sidecar is deliberately not loaded here: this path tests whether the
    learned generator/editor weights compose with the real online anchor
    model calls and sampling-time SGA/ANC controller.
    """

    expectations = (
        expected_global_step,
        expected_training_objective,
        expected_route_operator,
        expected_adapter_model_sha256,
        expected_adapter_config_sha256,
        expected_receipt_sha256,
    )
    if not value:
        if any(expectations):
            raise AnchorEventInferenceError(
                "trained attention expectations require a checkpoint"
            )
        return None
    if isinstance(expected_global_step, bool) or expected_global_step <= 0:
        raise AnchorEventInferenceError(
            "trained attention checkpoint requires a positive expected step"
        )
    if expected_training_objective not in SUPPORTED_ATTENTION_TRAINING_OBJECTIVES:
        raise AnchorEventInferenceError(
            "trained attention checkpoint requires a supported expected objective"
        )
    if expected_route_operator not in SUPPORTED_ATTENTION_ROUTE_OPERATORS:
        raise AnchorEventInferenceError(
            "trained attention checkpoint requires a supported expected route"
        )
    if SHA256_HEX.fullmatch(expected_adapter_model_sha256) is None:
        raise AnchorEventInferenceError(
            "trained attention checkpoint requires an expected adapter SHA-256"
        )
    if (
        expected_adapter_config_sha256
        and SHA256_HEX.fullmatch(expected_adapter_config_sha256) is None
    ):
        raise AnchorEventInferenceError(
            "trained attention checkpoint expected adapter config SHA-256 is invalid"
        )
    if SHA256_HEX.fullmatch(expected_receipt_sha256) is None:
        raise AnchorEventInferenceError(
            "trained attention checkpoint requires an expected receipt SHA-256"
        )
    root = _plain_directory(value, label="trained attention checkpoint")
    expected_root_name = f"checkpoint-{expected_global_step:08d}"
    if root.name != expected_root_name:
        raise AnchorEventInferenceError(
            "trained attention checkpoint directory differs from expected step"
        )
    receipt_path = root / "receipt.json"
    adapter_dir = root / "adapter"
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    if not all(path.is_file() and not path.is_symlink() for path in (
        receipt_path,
        config_path,
        model_path,
    )):
        raise AnchorEventInferenceError(
            "trained attention checkpoint closure is incomplete"
        )
    receipt_sha256 = file_sha256(receipt_path)
    config_sha256 = file_sha256(config_path)
    model_sha256 = file_sha256(model_path)
    if receipt_sha256 != expected_receipt_sha256:
        raise AnchorEventInferenceError("trained attention receipt SHA-256 differs")
    if model_sha256 != expected_adapter_model_sha256:
        raise AnchorEventInferenceError("trained attention adapter SHA-256 differs")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnchorEventInferenceError(
            "trained attention receipt is unreadable"
        ) from error
    if not isinstance(receipt, Mapping):
        raise AnchorEventInferenceError("trained attention receipt must be an object")
    contract = receipt.get("training_contract", {})
    if not isinstance(contract, Mapping):
        raise AnchorEventInferenceError(
            "trained attention receipt training contract must be an object"
        )
    schema_version = receipt.get("schema_version")
    supported_schema = schema_version in {
        "bernini-same-video-dense-flow-adapter-receipt-v1",
        "bernini-online-anchor-attention-training-receipt-v1",
        "bernini-online-anchor-attention-training-receipt-v2",
        "bernini-online-anchor-attention-training-receipt-v3",
    }
    v14r2_objective = "real_source_target_owned_routed_teacher_delta_v14r2"
    v14r2_schema = "bernini-online-anchor-attention-training-receipt-v3"
    v14r2_signal = (
        expected_training_objective == v14r2_objective
        or contract.get("training_objective") == v14r2_objective
        or "v14r2" in str(expected_route_operator)
        or "v14r2" in str(contract.get("route_operator", ""))
    )
    if v14r2_signal and schema_version != v14r2_schema:
        raise AnchorEventInferenceError("v14r2 checkpoint requires receipt schema v3")
    if schema_version == v14r2_schema and (
        SHA256_HEX.fullmatch(expected_adapter_config_sha256) is None
        or config_sha256 != expected_adapter_config_sha256
    ):
        raise AnchorEventInferenceError(
            "v14r2 checkpoint requires the expected adapter config SHA-256"
        )
    if schema_version == v14r2_schema and (
        expected_training_objective != v14r2_objective
        or contract.get("training_objective") != v14r2_objective
        or expected_route_operator not in V14R2_ROUTE_TO_TRANSPORT
        or contract.get("route_operator") not in V14R2_ROUTE_TO_TRANSPORT
    ):
        raise AnchorEventInferenceError("receipt schema v3 is exclusive to v14r2")
    if (
        not supported_schema
        or receipt.get("complete") is not True
        or isinstance(receipt.get("global_step"), bool)
        or receipt.get("global_step") != expected_global_step
        or contract.get("training_objective") != expected_training_objective
        or contract.get("route_operator") != expected_route_operator
        or contract.get("full_attention_lora_enabled") is not True
        or contract.get("lora_rank") != 256
        or contract.get("lora_alpha") != 256
        or contract.get("lora_scope") != "all_30_blocks_attn1_attn2_qkvo"
        or contract.get("lora_target_module_count") != 240
    ):
        raise AnchorEventInferenceError(
            "trained attention checkpoint contract differs"
        )
    required_decode_transport = None
    if schema_version == "bernini-online-anchor-attention-training-receipt-v3":
        action_probe = receipt.get("component_gradient_probes", {}).get(
            "action_objective", {}
        )
        replay_probe = receipt.get("component_gradient_probes", {}).get(
            "raw_source_caption_trajectory_replay", {}
        )
        interaction = receipt.get("component_gradient_probes", {}).get(
            "interaction", {}
        )
        objective_components = receipt.get("last_objective_components", {})
        memory_gate = receipt.get("memory_gate", {})
        gradient = receipt.get("gradient_coverage", {})
        cache = receipt.get("anchor_cache", {})
        action_sides = action_probe.get("adapter_sides", {})
        replay_sides = replay_probe.get("adapter_sides", {})
        combine_mode = contract.get("replay_combine_mode")
        memory_rows = (
            memory_gate.get("per_rank", [])
            if isinstance(memory_gate, Mapping)
            else []
        )
        minimum_reserved_fraction = (
            memory_gate.get("minimum_reserved_fraction")
            if isinstance(memory_gate, Mapping)
            else None
        )
        memory_gate_ok = (
            isinstance(memory_gate, Mapping)
            and memory_gate.get("capture_phase")
            == "after_two_real_component_backwards_before_actual_update_audit_clones"
            and memory_gate.get("actual_update_audit_allocations_excluded") is True
            and memory_gate.get("true_training_tensors_only") is True
            and memory_gate.get("dummy_or_padding_allocations") is False
            and memory_gate.get("passed") is True
            and isinstance(minimum_reserved_fraction, (int, float))
            and not isinstance(minimum_reserved_fraction, bool)
            and math.isfinite(float(minimum_reserved_fraction))
            and float(minimum_reserved_fraction) > 0.5
            and isinstance(memory_rows, list)
            and len(memory_rows) == 4
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("reserved_fraction"), (int, float))
                and not isinstance(row.get("reserved_fraction"), bool)
                and math.isfinite(float(row["reserved_fraction"]))
                and float(row["reserved_fraction"]) > 0.5
                for row in memory_rows
            )
        )

        def finite_number(name: str) -> bool:
            value = interaction.get(name)
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        def close_to(name: str, expected: float, tolerance: float) -> bool:
            return finite_number(name) and abs(
                float(interaction[name]) - float(expected)
            ) <= float(tolerance)

        geometry_fields = (
            "action_l2_norm_fp64",
            "raw_replay_l2_norm_fp64",
            "combined_l2_norm_fp64",
            "effective_replay_scale",
            "correction_ratio_q",
            "weighted_replay_gradient_fraction",
            "action_alignment_ratio",
            "action_gradient_dot_combined_gradient_fp64",
        )
        combine_geometry_ok = (
            interaction.get("replay_combine_mode") == combine_mode
            and all(finite_number(name) for name in geometry_fields)
        )
        combine_geometry_ok = combine_geometry_ok and (
            float(interaction.get("action_l2_norm_fp64", 0.0)) > 0.0
            and float(interaction.get("raw_replay_l2_norm_fp64", 0.0)) > 0.0
            and float(interaction.get("combined_l2_norm_fp64", 0.0)) > 0.0
            and float(
                interaction.get("action_gradient_dot_combined_gradient_fp64", 0.0)
            )
            > 0.0
        )
        if combine_mode == "action_only":
            combine_geometry_ok = combine_geometry_ok and all(
                close_to(name, 0.0, 1.0e-12)
                for name in (
                    "effective_replay_scale",
                    "correction_ratio_q",
                    "weighted_replay_gradient_fraction",
                )
            ) and close_to("action_alignment_ratio", 1.0, 1.0e-5) and (
                interaction.get("replay_projection_applied") is False
            )
        elif combine_mode == "norm_balanced_025":
            combine_geometry_ok = (
                combine_geometry_ok
                and close_to("correction_ratio_q", 0.25, 1.0e-5)
                and close_to("weighted_replay_gradient_fraction", 0.20, 1.0e-5)
                and float(interaction.get("action_alignment_ratio", 0.0)) >= 0.75
                and interaction.get("first_order_source_fm_preserved") is True
                and finite_number(
                    "raw_replay_gradient_dot_combined_gradient_fp64"
                )
                and float(
                    interaction[
                        "raw_replay_gradient_dot_combined_gradient_fp64"
                    ]
                )
                >= -1.0e-8
            )
        elif combine_mode == "action_priority_pcgrad_010":
            projected = interaction.get("replay_projection_applied") is True
            projection_ok = (
                finite_number("processed_replay_action_cosine")
                and abs(float(interaction["processed_replay_action_cosine"]))
                <= 1.0e-5
                and finite_number("processed_replay_retained_raw_norm_fraction")
                and float(interaction["processed_replay_retained_raw_norm_fraction"])
                >= 0.20
                if projected
                else finite_number("action_replay_cosine")
                and float(interaction["action_replay_cosine"]) >= 0.0
            )
            combine_geometry_ok = (
                combine_geometry_ok
                and close_to("correction_ratio_q", 0.10, 1.0e-5)
                and close_to("weighted_replay_gradient_fraction", 1.0 / 11.0, 1.0e-5)
                and float(interaction.get("action_alignment_ratio", 0.0)) >= 0.99
                and interaction.get(
                    "action_priority_conflict_control_not_source_preservation"
                )
                is True
                and projection_ok
            )
        elif combine_mode == "source_halfspace_001":
            combine_geometry_ok = (
                combine_geometry_ok
                and 0.01
                <= float(interaction.get("correction_ratio_q", -1.0))
                <= 1.01
                and float(interaction.get("action_alignment_ratio", 0.0)) >= 0.10
                and interaction.get("first_order_source_fm_preserved") is True
                and finite_number(
                    "raw_replay_combined_alignment_over_action_replay_norms"
                )
                and float(
                    interaction[
                        "raw_replay_combined_alignment_over_action_replay_norms"
                    ]
                )
                >= 0.009
            )
        else:
            combine_geometry_ok = False
        effective_reporting = objective_components.get(
            "effective_source_replay_scalar_for_reporting"
        )
        reporting_ok = (
            isinstance(effective_reporting, (int, float))
            and not isinstance(effective_reporting, bool)
            and math.isfinite(float(effective_reporting))
            and (
                float(effective_reporting) == 0.0
                if combine_mode == "action_only"
                else float(effective_reporting) > 0.0
            )
        )
        actual_update = receipt.get("actual_optimizer_update_probe", {})
        source_descent_required = combine_mode in (
            "norm_balanced_025",
            "source_halfspace_001",
        )
        actual_update_ok = (
            isinstance(actual_update, Mapping)
            and actual_update.get("schema_version")
            == "bernini-actual-optimizer-update-probe-v1"
            and actual_update.get("step") == expected_global_step
            and actual_update.get("replay_combine_mode") == combine_mode
            and actual_update.get("gradient_scope")
            == "separately_allreduced_global_action_and_raw_replay"
            and actual_update.get("optimizer_semantics_observed_not_modified") is True
            and actual_update.get("parameter_snapshot_native_dtype") is True
            and actual_update.get("tensor_count") == 480
            and actual_update.get("parameter_element_count") == 188743680
            and actual_update.get("changed_tensor_count") == 480
            and isinstance(actual_update.get("changed_element_count"), int)
            and actual_update.get("changed_element_count", 0) > 0
            and isinstance(
                actual_update.get("delta_theta_l2_norm_fp64"), (int, float)
            )
            and math.isfinite(
                float(actual_update.get("delta_theta_l2_norm_fp64", 0.0))
            )
            and float(actual_update.get("delta_theta_l2_norm_fp64", 0.0)) > 0.0
            and actual_update.get("action_descent_required") is True
            and actual_update.get("action_descent_passed") is True
            and isinstance(actual_update.get("action_descent_fp64"), (int, float))
            and float(actual_update.get("action_descent_fp64", 0.0)) > 0.0
            and actual_update.get("source_descent_required")
            is source_descent_required
            and (
                actual_update.get("source_descent_passed") is True
                and isinstance(
                    actual_update.get("source_descent_fp64"), (int, float)
                )
                and isinstance(
                    actual_update.get("minimum_allowed_source_descent_fp64"),
                    (int, float),
                )
                and math.isfinite(float(actual_update["source_descent_fp64"]))
                and math.isfinite(
                    float(actual_update["minimum_allowed_source_descent_fp64"])
                )
                and float(actual_update["source_descent_fp64"])
                >= float(actual_update["minimum_allowed_source_descent_fp64"])
                if source_descent_required
                else True
            )
        )
        try:
            required_decode_transport = V14R2_ROUTE_TO_TRANSPORT[
                expected_route_operator
            ]
        except KeyError as error:
            raise AnchorEventInferenceError(
                "v3 checkpoint route is not a v14r2 target-owned QK route"
            ) from error
        if (
            expected_training_objective != v14r2_objective
            or contract.get("route_transport") != required_decode_transport
            or contract.get("target_owned_qk_route_v14r2") is not True
            or contract.get("anchor_donor_cached_fields") != ["query", "key"]
            or contract.get("anchor_donor_value_cached_or_used_by_route") is not False
            or contract.get(
                "anchor_donor_hidden_or_attention_output_cached_or_used_by_route"
            )
            is not False
            or contract.get(
                "anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route"
            )
            is not False
            or contract.get("anchor_to_target_appearance_correspondence_used") is not False
            or contract.get(
                "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel"
            )
            is not True
            or contract.get("anchor_qk_phase0_only_difference_produces_zero_route")
            is not True
            or contract.get("real_source_variant_schedule") != "complete_real_source"
            or contract.get("source_variant_argument") != "not_applicable"
            or contract.get("micro_semantics")
            != "different_seed_and_cross_appearance_donor"
            or contract.get("anchor_route_replay_uses_per_capture") != 2
            or contract.get("teacher_delta_mode") != "raw"
            or contract.get("source_reconstruction_weight") is not None
            or contract.get("source_reconstruction_weight_argument") != 0.025
            or contract.get("base_replay_scale") != 0.025
            or contract.get("effective_replay_scale")
            != interaction.get("effective_replay_scale")
            or combine_mode
            not in (
                "action_only",
                "norm_balanced_025",
                "action_priority_pcgrad_010",
                "source_halfspace_001",
            )
            or contract.get("routed_teacher_mode")
            not in ("same_action_route_only", "cross_caption_two_sided")
            or (
                contract.get("routed_teacher_mode") == "same_action_route_only"
                and (
                    contract.get("student_route_off_branch_stop_gradient") is not True
                    or contract.get(
                        "action_objective_backpropagates_only_routed_student_query"
                    )
                    is not True
                    or contract.get("routed_teacher_cross_caption_source_branch")
                    is not False
                )
            )
            or (
                contract.get("routed_teacher_mode") == "cross_caption_two_sided"
                and contract.get("routed_teacher_cross_caption_source_branch")
                is not True
            )
            or contract.get("true_training_memory_fraction_strictly_above_half")
            is not True
            or contract.get("training_memory_gate_capture_phase")
            != "after_two_real_component_backwards_before_actual_update_audit_clones"
            or contract.get(
                "actual_update_audit_allocations_excluded_from_training_memory_gate"
            )
            is not True
            or not memory_gate_ok
            or receipt.get("last_loss") is not None
            or receipt.get(
                "last_reporting_scalar_is_not_a_joint_backpropagated_objective"
            )
            is not True
            or objective_components.get("base_replay_scale") != 0.025
            or objective_components.get("effective_replay_scale")
            != interaction.get("effective_replay_scale")
            or not reporting_ok
            or not actual_update_ok
            or expected_global_step < 2
            or action_probe.get("tensor_count") != 480
            or action_probe.get("nonzero_tensor_count") != 480
            or action_probe.get("epsilon_active_tensor_count") != 480
            or action_sides.get("lora_A", {}).get("nonzero_tensor_count") != 240
            or action_sides.get("lora_A", {}).get("epsilon_active_tensor_count")
            != 240
            or action_sides.get("lora_B", {}).get("nonzero_tensor_count") != 240
            or action_sides.get("lora_B", {}).get("epsilon_active_tensor_count")
            != 240
            or not isinstance(action_probe.get("l2_norm_fp64"), (int, float))
            or action_probe.get("l2_norm_fp64", 0.0) <= 0.0
            or replay_probe.get("tensor_count") != 480
            or replay_probe.get("nonzero_tensor_count") != 480
            or replay_probe.get("epsilon_active_tensor_count") != 480
            or replay_sides.get("lora_A", {}).get("nonzero_tensor_count") != 240
            or replay_sides.get("lora_A", {}).get("epsilon_active_tensor_count")
            != 240
            or replay_sides.get("lora_B", {}).get("nonzero_tensor_count") != 240
            or replay_sides.get("lora_B", {}).get("epsilon_active_tensor_count")
            != 240
            or not isinstance(replay_probe.get("l2_norm_fp64"), (int, float))
            or replay_probe.get("l2_norm_fp64", 0.0) <= 0.0
            or gradient.get("tensor_count") != 480
            or gradient.get("nonzero_tensor_count") != 480
            or not combine_geometry_ok
            or cache.get("pending_entries") != 0
            or cache.get("qk_only_cached_fields") != ["query", "key"]
            or not isinstance(cache.get("capture_count"), int)
            or not isinstance(cache.get("replay_count"), int)
            or cache.get("qk_only_capture_count") != cache.get("capture_count")
            or cache.get("qk_only_replay_count") != cache.get("replay_count")
            or cache.get("replay_count") != 2 * cache.get("capture_count", -1)
        ):
            raise AnchorEventInferenceError(
                "v3 target-owned QK checkpoint contract differs"
            )
    target_modules_sha256 = contract.get("lora_target_modules_sha256")
    if (
        not isinstance(target_modules_sha256, str)
        or SHA256_HEX.fullmatch(target_modules_sha256) is None
    ):
        raise AnchorEventInferenceError(
            "trained attention target-module registry SHA-256 is invalid"
        )
    declared_model_sha256 = receipt.get(
        "adapter_model_sha256", contract.get("adapter_model_sha256")
    )
    declared_config_sha256 = receipt.get(
        "adapter_config_sha256", contract.get("adapter_config_sha256")
    )
    if schema_version == v14r2_schema and (
        declared_model_sha256 != model_sha256
        or declared_config_sha256 != config_sha256
    ):
        raise AnchorEventInferenceError(
            "v3 receipt does not pin the complete adapter model/config closure"
        )
    if (
        declared_model_sha256 is not None
        and declared_model_sha256 != model_sha256
    ):
        raise AnchorEventInferenceError(
            "trained attention receipt declares a different adapter SHA-256"
        )
    if (
        declared_config_sha256 is not None
        and declared_config_sha256 != config_sha256
    ):
        raise AnchorEventInferenceError(
            "trained attention receipt declares a different adapter config SHA-256"
        )
    binding = {
        "receipt_sha256": receipt_sha256,
        "adapter_config_sha256": config_sha256,
        "adapter_model_sha256": model_sha256,
        "global_step": expected_global_step,
        "training_objective": expected_training_objective,
        "route_operator": expected_route_operator,
        "required_decode_transport": required_decode_transport,
    }
    binding_sha256 = hashlib.sha256(_canonical_json(binding)).hexdigest()
    return {
        "root": root,
        "adapter_dir": adapter_dir,
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "adapter_config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "target_modules_sha256": target_modules_sha256,
        "global_step": receipt.get("global_step"),
        "schema_version": receipt.get("schema_version"),
        "training_objective": contract.get("training_objective"),
        "route_operator": contract.get("route_operator"),
        "required_decode_transport": required_decode_transport,
        "binding": binding,
        "binding_sha256": binding_sha256,
    }


def _fresh_output(value: str) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.suffix.lower() != ".mp4":
        raise AnchorEventInferenceError("output must be an absolute .mp4 path")
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    receipt = output.with_name(output.name + ".receipt.json")
    if any(path.exists() or path.is_symlink() for path in (output, receipt)):
        raise AnchorEventInferenceError("refusing to overwrite an output artifact")
    return output, receipt


def _trained_route_off_control(
    *,
    trained_attention: Optional[Mapping[str, Any]],
    transport_steps: int,
    explicitly_allowed: bool,
) -> bool:
    """Validate the opt-in-only same-checkpoint no-route causal control."""

    active = (
        trained_attention is not None
        and transport_steps == 0
        and explicitly_allowed
    )
    if explicitly_allowed and not active:
        raise AnchorEventInferenceError(
            "trained route-off control requires a checkpoint and exactly zero transport steps"
        )
    if (
        trained_attention is not None
        and transport_steps == 0
        and not explicitly_allowed
    ):
        raise AnchorEventInferenceError(
            "trained attention decode with zero transport steps requires the explicit causal-control flag"
        )
    return active


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _resize_video_tensor(anchor: Any, source: Any) -> tuple[Any, dict[str, Any]]:
    """Center-crop to the source aspect ratio, then resize spatially only."""

    import torch
    import torch.nn.functional as functional

    if anchor.ndim != 5 or source.ndim != 5:
        raise AnchorEventInferenceError("decoded videos must be [B,C,T,H,W]")
    if tuple(anchor.shape[:3]) != tuple(source.shape[:3]):
        raise AnchorEventInferenceError("source and anchor batch/channel/frame axes differ")
    source_h, source_w = int(source.shape[-2]), int(source.shape[-1])
    anchor_h, anchor_w = int(anchor.shape[-2]), int(anchor.shape[-1])
    target_ratio = source_w / source_h
    anchor_ratio = anchor_w / anchor_h
    top = left = 0
    cropped_h, cropped_w = anchor_h, anchor_w
    if anchor_ratio < target_ratio:
        cropped_h = max(1, round(anchor_w / target_ratio))
        top = (anchor_h - cropped_h) // 2
    elif anchor_ratio > target_ratio:
        cropped_w = max(1, round(anchor_h * target_ratio))
        left = (anchor_w - cropped_w) // 2
    cropped = anchor[..., top : top + cropped_h, left : left + cropped_w]
    frames = cropped[0].permute(1, 0, 2, 3).contiguous()
    resized = functional.interpolate(
        frames,
        size=(source_h, source_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    result = resized.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if tuple(result.shape) != tuple(source.shape) or result.dtype != torch.float32:
        raise AnchorEventInferenceError("anchor source-bucket normalization differs")
    return result, {
        "original_hw": [anchor_h, anchor_w],
        "center_crop_tlwh": [top, left, cropped_h, cropped_w],
        "source_bucket_hw": [source_h, source_w],
        "interpolation": "per_frame_bilinear_antialias",
        "temporal_interpolation": False,
    }


def _broadcast_tensor(value: Any, *, dist: Any) -> Any:
    value = value.contiguous()
    dist.broadcast(value, src=0)
    return value


def _trim_host_allocator() -> bool:
    """Release retired checkpoint arenas to the Linux Slurm memory cgroup."""

    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
    except (AttributeError, OSError):
        # Unit tests and source review may run off-cluster on a non-glibc host.
        return False
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))


def _retire_t5_text_encoder(model: Any, *, torch_module: Any) -> bool:
    """Delete, rather than CPU-offload, this rank's frozen UMT5 encoder."""

    encoder = getattr(model, "t5_text_encoder", None)
    if encoder is None:
        raise AnchorEventInferenceError("T5 text encoder is absent before retirement")
    model.t5_text_encoder = None
    del encoder
    trimmed = _trim_host_allocator()
    torch_module.cuda.empty_cache()
    if getattr(model, "t5_text_encoder", None) is not None:
        raise AnchorEventInferenceError("T5 text encoder retirement failed")
    return trimmed


@contextmanager
def _nonzero_rank_t5_load_bypass(
    *,
    distributed_rank: int,
    t5_encoder_class: Any,
    expected_checkpoint: Path,
    expected_dtype: Any,
    placeholder_factory: Any,
) -> Any:
    """Prevent constructor-time UMT5 deserialization on nonzero ranks.

    The pinned Bernini ``BerniniRendererModel`` constructor unconditionally
    calls ``UMT5EncoderModel.from_pretrained`` and exposes no skip-text option.
    Each process has a private Python interpreter, so a rank-local, scoped
    replacement is safe as long as the exact call ABI is checked and the class
    descriptor is restored before model construction returns.
    """

    if distributed_rank < 0 or distributed_rank >= ULYSSES_SIZE:
        raise AnchorEventInferenceError("distributed rank is outside WORLD4")
    audit: dict[str, Any] = {
        "rank": distributed_rank,
        "real_t5_load": distributed_rank == 0,
        "bypassed_t5_load": distributed_rank != 0,
        "call_count": 0,
        "placeholder": None,
    }
    if distributed_rank == 0:
        yield audit
        return

    own_descriptor = vars(t5_encoder_class).get("from_pretrained")
    had_own_descriptor = "from_pretrained" in vars(t5_encoder_class)

    def bypassed_from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if cls is not t5_encoder_class:
            raise AnchorEventInferenceError("T5 bypass class identity differs")
        if (
            len(args) != 1
            or str(args[0]) != str(expected_checkpoint)
            or kwargs.get("subfolder") != "text_encoder"
            or kwargs.get("torch_dtype") != expected_dtype
            or set(kwargs) != {"subfolder", "torch_dtype"}
        ):
            raise AnchorEventInferenceError("Bernini T5 constructor call ABI differs")
        audit["call_count"] += 1
        if audit["call_count"] != 1:
            raise AnchorEventInferenceError("Bernini loaded T5 more than once")
        placeholder = placeholder_factory()
        audit["placeholder"] = placeholder
        return placeholder

    setattr(
        t5_encoder_class,
        "from_pretrained",
        classmethod(bypassed_from_pretrained),
    )
    try:
        yield audit
    finally:
        if had_own_descriptor:
            setattr(t5_encoder_class, "from_pretrained", own_descriptor)
        else:
            delattr(t5_encoder_class, "from_pretrained")
    if audit["call_count"] != 1 or audit["placeholder"] is None:
        raise AnchorEventInferenceError("nonzero-rank T5 load bypass was not exercised")


def _validate_t5_load_closure(rows: Sequence[Any]) -> None:
    """Require one real T5 load on rank zero and one bypass on every peer."""

    if len(rows) != ULYSSES_SIZE:
        raise AnchorEventInferenceError("T5 load closure does not contain WORLD4")
    for rank, row in enumerate(rows):
        expected = {
            "rank": rank,
            "real_t5_loaded": rank == 0,
            "bypassed_t5_load": rank != 0,
            "bypass_call_count": 0 if rank == 0 else 1,
            "placeholder_retained": rank != 0,
        }
        if not isinstance(row, Mapping) or dict(row) != expected:
            raise AnchorEventInferenceError(
                f"T5 load closure differs on rank {rank}: {row}"
            )


def _rank_zero_prompt_bank(
    model: Any,
    *,
    tokenized_prompts: Optional[Mapping[str, tuple[Any, Any]]],
    distributed_rank: int,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Encode seven prompts once, retire all T5 copies, then broadcast BF16 bytes.

    Nonzero ranks release their constructor-time placeholder before rank zero is
    allowed to materialize UMT5 on its GPU.  This ordering is important on a
    64-GiB node: it prevents four ranks from deserializing/moving the two UMT5
    shards concurrently.  The fixed shape is part of the audited renderer ABI,
    so receivers can allocate directly without broadcasting tensor metadata.
    """

    if distributed_rank < 0 or distributed_rank >= ULYSSES_SIZE:
        raise AnchorEventInferenceError("distributed rank is outside WORLD4")
    if distributed_rank != 0:
        if tokenized_prompts is not None:
            raise AnchorEventInferenceError("nonzero rank received prompt token tensors")
        _retire_t5_text_encoder(model, torch_module=torch_module)

    # Rank zero keeps its encoder on CPU until every nonzero copy is gone.
    dist.barrier()
    local_bank: dict[str, Any] = {}
    status: list[Any] = [None]
    if distributed_rank == 0:
        try:
            if not isinstance(tokenized_prompts, Mapping) or tuple(
                tokenized_prompts.keys()
            ) != PROMPT_BANK_NAMES:
                raise AnchorEventInferenceError("rank-zero prompt token bank differs")
            model.t5_text_encoder.to(device)
            with torch_module.inference_mode():
                for name in PROMPT_BANK_NAMES:
                    input_ids, attention_mask = tokenized_prompts[name]
                    embeddings = model.encode_prompt(
                        input_ids.to(device), attention_mask.to(device)
                    ).contiguous()
                    if (
                        embeddings.dtype != torch_module.bfloat16
                        or tuple(embeddings.shape) != PROMPT_EMBEDDING_SHAPE
                    ):
                        raise AnchorEventInferenceError(
                            f"prompt embedding contract differs for {name}"
                        )
                    local_bank[name] = embeddings
            status[0] = {"ok": True}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            _retire_t5_text_encoder(model, torch_module=torch_module)

    # A Python-side rank-zero encoding failure is published before any receiver
    # enters the seven tensor collectives, avoiding a silent collective mismatch.
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise AnchorEventInferenceError(f"rank-zero prompt encoding failed: {status[0]}")

    prompt_bank: dict[str, Any] = {}
    for name in PROMPT_BANK_NAMES:
        embeddings = (
            local_bank[name]
            if distributed_rank == 0
            else torch_module.empty(
                PROMPT_EMBEDDING_SHAPE,
                dtype=torch_module.bfloat16,
                device=device,
            )
        )
        embeddings = _broadcast_tensor(embeddings, dist=dist)
        if (
            embeddings.dtype != torch_module.bfloat16
            or tuple(embeddings.shape) != PROMPT_EMBEDDING_SHAPE
        ):
            raise AnchorEventInferenceError(
                f"broadcast prompt embedding contract differs for {name}"
            )
        prompt_bank[name] = embeddings
    return prompt_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--trained-attention-checkpoint",
        default="",
        help=(
            "Optional trained rank-256 all-attention LoRA checkpoint. The "
            "online pure-T2V anchor and real sampling-time SGA/ANC remain active."
        ),
    )
    parser.add_argument(
        "--allow-trained-route-off-control",
        action="store_true",
        help=(
            "Explicitly allow transport_steps=0 with a loaded trained adapter "
            "as a same-checkpoint causal control. The adapter remains enabled "
            "for target/source editor calls while anchor injection is disabled."
        ),
    )
    parser.add_argument("--expected-trained-attention-step", type=int, default=0)
    parser.add_argument(
        "--expected-trained-attention-objective",
        choices=SUPPORTED_ATTENTION_TRAINING_OBJECTIVES,
        default="",
    )
    parser.add_argument(
        "--expected-trained-attention-route-operator",
        choices=SUPPORTED_ATTENTION_ROUTE_OPERATORS,
        default="",
    )
    parser.add_argument(
        "--expected-trained-attention-adapter-sha256", default=""
    )
    parser.add_argument(
        "--expected-trained-attention-adapter-config-sha256", default=""
    )
    parser.add_argument(
        "--expected-trained-attention-receipt-sha256", default=""
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--anchor-video", required=True)
    parser.add_argument("--expected-anchor-sha256", required=True)
    parser.add_argument("--anchor-initial-gaussian", default="")
    parser.add_argument("--expected-anchor-initial-gaussian-sha256", default="")
    parser.add_argument("--expected-anchor-initial-gaussian-raw-sha256", default="")
    parser.add_argument(
        "--initial-noise-proposal-mode",
        choices=controller.INITIAL_NOISE_PROPOSAL_MODES,
        default="keyed_only",
    )
    parser.add_argument(
        "--anchor-state-mode",
        choices=controller.ANCHOR_STATE_MODES,
        default="clean_noised",
    )
    parser.add_argument("--extra-anchor-video", action="append", default=[])
    parser.add_argument("--expected-extra-anchor-sha256", action="append", default=[])
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--anchor-caption", required=True)
    parser.add_argument("--anchor-noop-caption", required=True)
    parser.add_argument("--arm", required=True, choices=controller.ARMS)
    parser.add_argument(
        "--transport",
        choices=controller.TRANSPORTS,
        default="hard_qk",
    )
    parser.add_argument("--transport-strength", type=float, default=1.0)
    parser.add_argument("--transport-steps", type=int, default=40)
    parser.add_argument("--no-initial-phase-clamp", action="store_true")
    parser.add_argument(
        "--field-guidance", choices=controller.FIELD_GUIDANCES, default="apg"
    )
    parser.add_argument(
        "--field-model", choices=controller.FIELD_MODELS, default="source_conditioned_rv2v"
    )
    parser.add_argument("--source-cfg-scale", type=float, default=1.0)
    parser.add_argument("--target-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--anchor-cfg-scope",
        choices=controller.ANCHOR_CFG_SCOPES,
        default="shared",
    )
    parser.add_argument(
        "--anchor-contrast-mode",
        choices=controller.ANCHOR_CONTRAST_MODES,
        default="caption_noop_same_video",
    )
    parser.add_argument(
        "--anchor-sigma-cap",
        type=float,
        choices=controller.ANCHOR_SIGMA_CAPS,
        default=1.0,
    )
    parser.add_argument(
        "--preservation-mode",
        choices=controller.PRESERVATION_MODES,
        default="none",
    )
    parser.add_argument(
        "--preservation-keep-fraction",
        type=float,
        choices=controller.PRESERVATION_KEEP_FRACTIONS,
        default=0.20,
    )
    parser.add_argument(
        "--preservation-outside-scale",
        type=float,
        choices=controller.PRESERVATION_OUTSIDE_SCALES,
        default=0.0,
    )
    parser.add_argument(
        "--preservation-dilation",
        type=int,
        choices=controller.PRESERVATION_DILATIONS,
        default=1,
    )
    parser.add_argument(
        "--preservation-residual-fraction",
        type=float,
        choices=controller.PRESERVATION_RESIDUAL_FRACTIONS,
        default=0.0,
    )
    parser.add_argument(
        "--preservation-object-identity-strength",
        type=float,
        choices=controller.PRESERVATION_OBJECT_IDENTITY_STRENGTHS,
        default=0.0,
    )
    parser.add_argument("--preservation-start-step", type=int, default=0)
    parser.add_argument("--preservation-ramp-steps", type=int, default=1)
    parser.add_argument(
        "--sga-score-mode",
        choices=controller.SGA_SCORE_MODES,
        default="global_source_cosine",
    )
    parser.add_argument(
        "--anchor-candidate-mode",
        choices=controller.ANCHOR_CANDIDATE_MODES,
        default="single_shared",
    )
    parser.add_argument(
        "--anchor-spatial-alignment",
        choices=controller.ANCHOR_SPATIAL_ALIGNMENTS,
        default="none",
    )
    parser.add_argument("--event01-forced-role-proposal-index", type=int, default=-1)
    parser.add_argument(
        "--sga-temperature",
        type=float,
        choices=controller.SUPPORTED_SGA_TEMPERATURES,
        default=controller.DYNAEDIT_SGA_TEMPERATURE,
    )
    parser.add_argument(
        "--early-candidate-count",
        type=int,
        choices=controller.SUPPORTED_EARLY_CANDIDATES,
        default=controller.EARLY_CANDIDATES,
    )
    parser.add_argument("--source-caption", default="")
    parser.add_argument("--target-caption", default="")
    parser.add_argument("--blocks", default="0-15")
    parser.add_argument("--output", required=True)
    return parser


def _parse_blocks(value: str) -> tuple[int, ...]:
    if "," in value:
        try:
            blocks = tuple(int(item) for item in value.split(","))
        except ValueError as error:
            raise AnchorEventInferenceError(
                "blocks must be comma-separated integers or inclusive START-END"
            ) from error
        if (
            not blocks
            or blocks != tuple(sorted(set(blocks)))
            or any(item < 0 or item >= 30 for item in blocks)
        ):
            raise AnchorEventInferenceError(
                "comma-separated blocks must be an increasing subset of 0..29"
            )
        return blocks
    pieces = value.split("-")
    if len(pieces) != 2:
        raise AnchorEventInferenceError(
            "blocks must be comma-separated integers or inclusive START-END"
        )
    try:
        start, end = (int(item) for item in pieces)
    except ValueError as error:
        raise AnchorEventInferenceError("blocks must be integer bounds") from error
    if start < 0 or end < start or end >= 30:
        raise AnchorEventInferenceError("blocks must be inside 0..29")
    return tuple(range(start, end + 1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    selected_blocks = _parse_blocks(args.blocks)
    trained_attention = _attention_lora_checkpoint(
        args.trained_attention_checkpoint,
        expected_global_step=args.expected_trained_attention_step,
        expected_training_objective=args.expected_trained_attention_objective,
        expected_route_operator=args.expected_trained_attention_route_operator,
        expected_adapter_model_sha256=(
            args.expected_trained_attention_adapter_sha256
        ),
        expected_adapter_config_sha256=(
            args.expected_trained_attention_adapter_config_sha256
        ),
        expected_receipt_sha256=args.expected_trained_attention_receipt_sha256,
    )
    trained_route_off_control = _trained_route_off_control(
        trained_attention=trained_attention,
        transport_steps=args.transport_steps,
        explicitly_allowed=args.allow_trained_route_off_control,
    )
    if (
        trained_attention is not None
        and trained_attention.get("required_decode_transport") is not None
        and args.transport != trained_attention["required_decode_transport"]
    ):
        raise AnchorEventInferenceError(
            "v14r2 decode transport differs from its authenticated training route"
        )
    source_path = _plain_file(args.source_video, label="source video")
    anchor_paths = [
        _plain_file(args.anchor_video, label="pure-T2V anchor video")
    ] + [
        _plain_file(value, label=f"extra pure-T2V anchor video {index}")
        for index, value in enumerate(args.extra_anchor_video, start=1)
    ]
    anchor_expected_shas = [args.expected_anchor_sha256] + list(
        args.expected_extra_anchor_sha256
    )
    anchor_initial_gaussian_path: Optional[Path] = None
    needs_anchor_initial_gaussian = (
        args.initial_noise_proposal_mode != "keyed_only"
        or args.anchor_state_mode == "native_t2v_trajectory"
    )
    if not needs_anchor_initial_gaussian:
        if any(
            (
                args.anchor_initial_gaussian,
                args.expected_anchor_initial_gaussian_sha256,
                args.expected_anchor_initial_gaussian_raw_sha256,
            )
        ):
            raise AnchorEventInferenceError(
                "inactive anchor Gaussian must not be supplied"
            )
    else:
        if not all(
            (
                args.anchor_initial_gaussian,
                args.expected_anchor_initial_gaussian_sha256,
                args.expected_anchor_initial_gaussian_raw_sha256,
            )
        ):
            raise AnchorEventInferenceError(
                "anchor seed/trajectory mode requires the Gaussian path and both hashes"
            )
        anchor_initial_gaussian_path = _plain_file(
            args.anchor_initial_gaussian,
            label="anchor native initial Gaussian",
        )
        if (
            file_sha256(anchor_initial_gaussian_path)
            != args.expected_anchor_initial_gaussian_sha256
        ):
            raise AnchorEventInferenceError(
                "anchor initial Gaussian file SHA-256 differs"
            )
    if len(anchor_paths) != len(anchor_expected_shas):
        raise AnchorEventInferenceError(
            "every extra anchor video requires one expected SHA-256"
        )
    if args.anchor_candidate_mode == "single_shared" and len(anchor_paths) != 1:
        raise AnchorEventInferenceError("single_shared accepts exactly one anchor video")
    if args.anchor_candidate_mode == "bank_per_candidate" and not 2 <= len(anchor_paths) <= 5:
        raise AnchorEventInferenceError(
            "bank_per_candidate requires two to five anchor videos"
        )
    manifest_path = _plain_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    output_path, receipt_path = _fresh_output(args.output)
    if file_sha256(source_path) != args.expected_source_sha256:
        raise AnchorEventInferenceError("source SHA-256 differs")
    for index, (path, expected_sha) in enumerate(
        zip(anchor_paths, anchor_expected_shas)
    ):
        if file_sha256(path) != expected_sha:
            raise AnchorEventInferenceError(f"anchor {index} SHA-256 differs")
    if (
        not args.instruction.strip()
        or not args.anchor_caption.strip()
        or not args.anchor_noop_caption.strip()
    ):
        raise AnchorEventInferenceError(
            "instruction and anchor action/no-op captions must be non-empty"
        )
    if args.field_model in (
        "first_phase_caption_i2v",
        "source_free_t2v",
    ) and not args.source_caption.strip():
        raise AnchorEventInferenceError(
            "caption field requires a complete source caption"
        )
    if args.field_model in (
        "first_phase_caption_i2v",
        "source_free_t2v",
    ) and not args.target_caption.strip():
        raise AnchorEventInferenceError(
            "caption field requires a source-appearance target caption"
        )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
                expected_veomni_commit=legacy.trainer.VEOMNI_TESTED_COMMIT,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise AnchorEventInferenceError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise AnchorEventInferenceError("transformer heads do not divide Ulysses=4")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)
    # These audited helpers deliberately receive their runtime modules through
    # globals so importing them never activates a second source tree.
    source_audit.legacy = legacy
    source_audit.trainer = legacy.trainer

    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file as load_safetensors_file

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise AnchorEventInferenceError("runtime mv2v system prompt differs")
    if SYSTEM_PROMPTS.get("t2v") != native.TASK_SYSTEM_PROMPTS["t2v"]:
        raise AnchorEventInferenceError("runtime t2v system prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise AnchorEventInferenceError("runtime negative prompt differs")

    distributed = legacy.inference_distributed_contract()
    if distributed.world_size != ULYSSES_SIZE:
        raise AnchorEventInferenceError("runner requires exactly four ranks")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise AnchorEventInferenceError("runner requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, manifest_path
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise AnchorEventInferenceError(
            f"rank-zero checkpoint audit failed: {checkpoint_result}"
        )

    source_tensor, source_metadata, source_sha = (
        source_audit.prepare_hashed_source_snapshot(source_path)
    )
    anchor_snapshots = [
        source_audit.prepare_hashed_source_snapshot(path) for path in anchor_paths
    ]
    anchor_tensors = [item[0] for item in anchor_snapshots]
    anchor_metadatas = [item[1] for item in anchor_snapshots]
    anchor_shas = [item[2] for item in anchor_snapshots]
    if source_sha != args.expected_source_sha256 or anchor_shas != anchor_expected_shas:
        raise AnchorEventInferenceError("post-snapshot media hash differs")
    if source_metadata.get("frame_count") != FRAME_COUNT or source_metadata.get("fps") != FPS:
        raise AnchorEventInferenceError("source must be exact81 at 25 fps")
    if any(
        metadata.get("frame_count") != FRAME_COUNT
        or metadata.get("fps") != FPS
        for metadata in anchor_metadatas
    ):
        raise AnchorEventInferenceError("every anchor must be exact81 at 25 fps")
    resized_anchor_rows = [
        _resize_video_tensor(tensor, source_tensor) for tensor in anchor_tensors
    ]
    anchor_tensors = [item[0] for item in resized_anchor_rows]
    anchor_resizes = [item[1] for item in resized_anchor_rows]
    bucket_hw = tuple(int(item) for item in source_tensor.shape[-2:])

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise AnchorEventInferenceError("renderer must use pinned UniPC shift 5")
    with _nonzero_rank_t5_load_bypass(
        distributed_rank=distributed.rank,
        t5_encoder_class=UMT5EncoderModel,
        expected_checkpoint=checkpoint,
        expected_dtype=torch.bfloat16,
        placeholder_factory=torch.nn.Identity,
    ) as t5_load_audit:
        model = BerniniRendererModel(config)
    if distributed.rank == 0:
        if not isinstance(model.t5_text_encoder, UMT5EncoderModel):
            raise AnchorEventInferenceError("rank zero did not load the real T5 encoder")
    elif model.t5_text_encoder is not t5_load_audit["placeholder"]:
        raise AnchorEventInferenceError("nonzero rank retained a non-placeholder T5")
    if trained_attention is not None:
        from peft import LoraConfig, PeftModel

        targets = legacy.trainer.select_attention_projection_names(model)
        if (
            len(targets) != 240
            or legacy.trainer.object_sha256(targets)
            != trained_attention["target_modules_sha256"]
        ):
            raise AnchorEventInferenceError(
                "runtime all-attention LoRA target registry differs"
            )
        lora_config = LoraConfig.from_pretrained(
            str(trained_attention["adapter_dir"]), local_files_only=True
        )
        lora_config.target_modules = set(targets)
        peft_model = PeftModel.from_pretrained(
            model,
            str(trained_attention["adapter_dir"]),
            is_trainable=False,
            config=lora_config,
            local_files_only=True,
        )
        if not callable(getattr(peft_model, "disable_adapter", None)):
            raise AnchorEventInferenceError(
                "trained attention PEFT wrapper cannot expose the frozen anchor teacher"
            )
        # Keep PEFT reversible.  Target/source editor calls use the adapter,
        # while the controller enters ``disable_adapter()`` only for pure-T2V
        # action/no-op anchor teacher calls.  Merging here would contaminate the
        # supposedly frozen teacher and make it drift with training step.
        model = peft_model
    model.requires_grad_(False)
    model.eval()
    t5_load_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(
        t5_load_rows,
        {
            "rank": distributed.rank,
            "real_t5_loaded": (
                distributed.rank == 0
                and isinstance(model.t5_text_encoder, UMT5EncoderModel)
            ),
            "bypassed_t5_load": bool(t5_load_audit["bypassed_t5_load"]),
            "bypass_call_count": int(t5_load_audit["call_count"]),
            "placeholder_retained": (
                distributed.rank != 0
                and model.t5_text_encoder is t5_load_audit["placeholder"]
            ),
        },
    )
    _validate_t5_load_closure(t5_load_rows)
    # This collective also prevents rank-zero's real 11-GiB UMT5
    # deserialization from overlapping another rank's VAE checkpoint load.

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    with torch.inference_mode():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        ).float().contiguous()
        anchor_latents = [
            _vae_encode(
                vae, tensor.to(device=device, dtype=torch.float32)
            ).float().contiguous()
            for tensor in anchor_tensors
        ]
        anchor_latent = (
            anchor_latents[0]
            if len(anchor_latents) == 1
            else torch.stack(anchor_latents, dim=0)
        )
    source_latent = _broadcast_tensor(source_latent, dist=dist)
    anchor_latent = _broadcast_tensor(anchor_latent, dist=dist)
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        21,
        bucket_hw[0] // 8,
        bucket_hw[1] // 8,
    )
    if (
        tuple(source_latent.shape) != expected_latent_shape
        or tuple(anchor_latent.shape[-5:]) != expected_latent_shape
        or source_latent.dtype != torch.float32
        or anchor_latent.dtype != torch.float32
        or any(
            torch.equal(source_latent, item)
            for item in (
                anchor_latent.unsqueeze(0)
                if anchor_latent.ndim == source_latent.ndim
                else anchor_latent
            ).unbind(0)
        )
    ):
        raise AnchorEventInferenceError("source/anchor latent contract differs")
    anchor_initial_gaussian = None
    anchor_initial_gaussian_identity = None
    if anchor_initial_gaussian_path is not None:
        stored = load_safetensors_file(
            str(anchor_initial_gaussian_path), device="cpu"
        )
        if tuple(stored) != ("official_initial_gaussian",):
            raise AnchorEventInferenceError(
                "anchor initial Gaussian safetensor key differs"
            )
        anchor_initial_gaussian = (
            stored["official_initial_gaussian"].float().contiguous().to(device)
        )
        del stored
        anchor_initial_gaussian_identity = source_audit.tensor_identity(
            anchor_initial_gaussian,
            label="anchor native initial Gaussian",
        )
        if (
            tuple(anchor_initial_gaussian.shape) != expected_latent_shape
            or anchor_initial_gaussian.dtype != torch.float32
            or not bool(torch.isfinite(anchor_initial_gaussian).all())
            or anchor_initial_gaussian_identity["raw_storage_sha256"]
            != args.expected_anchor_initial_gaussian_raw_sha256
        ):
            raise AnchorEventInferenceError(
                "anchor initial Gaussian tensor identity differs"
            )
    del source_tensor, anchor_tensors, anchor_latents
    vae.to("cpu")
    torch.cuda.empty_cache()

    action_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    anchor_prompt = native.build_task_prompt(
        "t2v", args.anchor_caption, prompt_cleaner=prompt_clean
    )
    anchor_noop_prompt = native.build_task_prompt(
        "t2v", args.anchor_noop_caption, prompt_cleaner=prompt_clean
    )
    source_t2v_prompt = native.build_task_prompt(
        "t2v",
        args.source_caption if args.source_caption.strip() else args.anchor_caption,
        prompt_cleaner=prompt_clean,
    )
    target_t2v_prompt = native.build_task_prompt(
        "t2v",
        args.target_caption if args.target_caption.strip() else args.anchor_caption,
        prompt_cleaner=prompt_clean,
    )
    tokenizer = None
    tokenized_prompts: Optional[Mapping[str, tuple[Any, Any]]] = None
    tokenization_status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
            )
            tokenized_prompts = {
                "action": legacy._tokenize_training_prompt(tokenizer, action_prompt),
                "noop": legacy._tokenize_training_prompt(tokenizer, noop_prompt),
                "anchor": legacy._tokenize_training_prompt(tokenizer, anchor_prompt),
                "anchor_noop": legacy._tokenize_training_prompt(
                    tokenizer, anchor_noop_prompt
                ),
                "source_t2v": legacy._tokenize_training_prompt(
                    tokenizer, source_t2v_prompt
                ),
                "target_t2v": legacy._tokenize_training_prompt(
                    tokenizer, target_t2v_prompt
                ),
                "negative": legacy._tokenize_renderer_negative(
                    tokenizer, DEFAULT_NEG_PROMPT
                ),
            }
            tokenization_status[0] = {"ok": True}
        except Exception as error:
            tokenization_status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(tokenization_status, src=0)
    if (
        not isinstance(tokenization_status[0], Mapping)
        or tokenization_status[0].get("ok") is not True
    ):
        raise AnchorEventInferenceError(
            f"rank-zero prompt tokenization failed: {tokenization_status[0]}"
        )
    prompt_bank = _rank_zero_prompt_bank(
        model,
        tokenized_prompts=tokenized_prompts,
        distributed_rank=distributed.rank,
        device=device,
        dist=dist,
        torch_module=torch,
    )
    del tokenizer, tokenized_prompts
    _trim_host_allocator()
    action_embeddings = prompt_bank["action"]
    noop_embeddings = prompt_bank["noop"]
    anchor_embeddings = prompt_bank["anchor"]
    anchor_noop_embeddings = prompt_bank["anchor_noop"]
    source_t2v_embeddings = prompt_bank["source_t2v"]
    target_t2v_embeddings = prompt_bank["target_t2v"]
    negative_embeddings = prompt_bank["negative"]

    # The before/after certificates cover the same stable denoising model:
    # UMT5 has completed its only role and is absent from both observations.
    freeze_before = source_audit.model_freeze_certificate(model)

    model.diff_dec.transformer.to(device)
    with torch.inference_mode():
        generated_latent, trace = controller.sample_anchor_sga_anc(
            model,
            source_latent=source_latent,
            anchor_latent=anchor_latent,
            anchor_initial_gaussian=anchor_initial_gaussian,
            source_rgb_frames=FRAME_COUNT,
            action_prompt_embeds=action_embeddings,
            anchor_prompt_embeds=anchor_embeddings,
            anchor_noop_prompt_embeds=anchor_noop_embeddings,
            source_t2v_prompt_embeds=source_t2v_embeddings,
            target_t2v_prompt_embeds=target_t2v_embeddings,
            noop_prompt_embeds=noop_embeddings,
            negative_prompt_embeds=negative_embeddings,
            config=controller.AnchorSGAANCConfig(
                arm=args.arm,
                transport=args.transport,
                selected_block_indices=selected_blocks,
                transport_strength=args.transport_strength,
                transport_steps=args.transport_steps,
                initial_phase_clamp=not args.no_initial_phase_clamp,
                field_guidance=args.field_guidance,
                field_model=args.field_model,
                source_cfg_scale=args.source_cfg_scale,
                target_cfg_scale=args.target_cfg_scale,
                sga_temperature=args.sga_temperature,
                early_candidate_count=args.early_candidate_count,
                initial_noise_proposal_mode=args.initial_noise_proposal_mode,
                anchor_state_mode=args.anchor_state_mode,
                anchor_cfg_scope=args.anchor_cfg_scope,
                anchor_contrast_mode=args.anchor_contrast_mode,
                anchor_sigma_cap=args.anchor_sigma_cap,
                preservation_mode=args.preservation_mode,
                preservation_keep_fraction=args.preservation_keep_fraction,
                preservation_outside_scale=args.preservation_outside_scale,
                preservation_dilation=args.preservation_dilation,
                preservation_residual_fraction=args.preservation_residual_fraction,
                preservation_object_identity_strength=(
                    args.preservation_object_identity_strength
                ),
                preservation_start_step=args.preservation_start_step,
                preservation_ramp_steps=args.preservation_ramp_steps,
                sga_score_mode=args.sga_score_mode,
                anchor_candidate_mode=args.anchor_candidate_mode,
                anchor_spatial_alignment=args.anchor_spatial_alignment,
                event01_forced_role_proposal_index=(
                    args.event01_forced_role_proposal_index
                ),
            ),
            return_trace=True,
        )
    if trained_attention is not None:
        if trace.get("anchor_teacher_uses_unadapted_base") is not True:
            raise AnchorEventInferenceError(
                "controller did not certify an unadapted pure-T2V teacher"
            )
        if trace.get("anchor_teacher_disable_adapter_context_available") is not True:
            raise AnchorEventInferenceError(
                "trained editor cannot disable its adapter for anchor teacher calls"
            )
        anchor_model_forwards = int(trace.get("anchor_model_forwards", 0))
        anchor_native_forwards = int(
            trace.get("anchor_native_trajectory_model_forwards", 0)
        )
        attention_capture_count = int(
            trace.get("attention_cache", {}).get("capture_count", 0)
        )
        if (
            trained_attention is not None
            and trained_attention.get("schema_version")
            == "bernini-online-anchor-attention-training-receipt-v3"
        ):
            early_steps = min(
                args.transport_steps, controller.EARLY_CANDIDATE_STEPS
            )
            expected_anchor_cells = (
                early_steps * args.early_candidate_count
                + max(0, args.transport_steps - early_steps)
            )
            controller._validate_target_owned_qk_route_closure(
                transport=args.transport,
                transport_steps=args.transport_steps,
                expected_anchor_cells=expected_anchor_cells,
                selected_block_count=len(selected_blocks),
                field_guidance=args.field_guidance,
                anchor_cfg_scope=args.anchor_cfg_scope,
                trace=trace,
                cache_receipt=trace.get("attention_cache", {}),
            )
        if trained_route_off_control:
            if (
                anchor_model_forwards != 0
                or anchor_native_forwards != 0
                or attention_capture_count != 0
                or trace.get("anchor_active_schedule") != []
            ):
                raise AnchorEventInferenceError(
                    "trained route-off control unexpectedly executed anchor injection"
                )
        elif anchor_model_forwards + anchor_native_forwards <= 0:
            raise AnchorEventInferenceError(
                "trained attention decode executed no pure-T2V teacher forwards"
            )
    if tuple(generated_latent.shape) != expected_latent_shape:
        raise AnchorEventInferenceError("generated latent geometry differs")
    if generated_latent.dtype != torch.float32 or not bool(torch.isfinite(generated_latent).all()):
        raise AnchorEventInferenceError("generated latent must be finite fp32")
    freeze_after = source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise AnchorEventInferenceError("frozen-model certificate changed")
    trace_digest = hashlib.sha256(_canonical_json(trace)).hexdigest()
    rank_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(
        rank_rows,
        {
            "rank": distributed.rank,
            "trace_digest": trace_digest,
            "latent": source_audit.tensor_identity(
                generated_latent, label="anchor-sga-anc generated latent"
            ),
        },
    )
    if any(row["trace_digest"] != rank_rows[0]["trace_digest"] for row in rank_rows[1:]):
        raise AnchorEventInferenceError("per-rank controller traces differ")
    if any(row["latent"] != rank_rows[0]["latent"] for row in rank_rows[1:]):
        raise AnchorEventInferenceError("per-rank generated latent bytes differ")

    model.to("cpu")
    torch.cuda.empty_cache()
    if distributed.rank == 0:
        vae.to(device)
        with torch.inference_mode():
            decoded = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        if tuple(decoded.shape) != (FRAME_COUNT, bucket_hw[0], bucket_hw[1], 3):
            raise AnchorEventInferenceError("decoded output geometry differs")
        save_output(decoded, str(output_path), fps=FPS)
        attention_cache = trace["attention_cache"]
        block_transport_enabled = bool(attention_cache["capture_count"])
        native_trajectory_enabled = (
            args.anchor_state_mode == "native_t2v_trajectory"
            and args.transport_steps > 0
        )
        velocity_transport_enabled = bool(
            trace["anchor_action_noop_velocity_contrast_transported"]
            or trace["anchor_field_velocity_residual_transported"]
        )
        if trace["native_t2v_target_velocity_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_action_velocity_hard_replaces_target_field_"
                "while_rv2v_source_field_remains"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_rolewarp_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_temporal_action_quotient_actor_object_components_"
                "rebound_to_event01_source_role_proposals"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_rolewarped_per_sga_candidate"
            )
        elif trace["native_rolewarp_sparse25_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_temporal_action_quotient_actor_object_components_"
                "rebound_to_event01_source_role_proposals_top25"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_rolewarped_per_sga_candidate"
            )
        elif trace["native_t2v_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_action_minus_noop_phase0_quotient_hard_replaces_"
                "entire_edit_delta"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_t2v_sparse25_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_action_minus_noop_phase0_quotient_top25_hard_"
                "replaces_entire_edit_delta"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_targetstate_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "source_coordinate_target_state_action_noop_temporal_quotient_"
                "with_native_generation_phase_timing"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_targetstate_sparse25_temporal_delta_hard_replacement"]:
            anchor_online_role = (
                "source_coordinate_target_state_action_noop_temporal_quotient_"
                "top25_with_native_generation_phase_timing"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_targetstate_raw_delta_hard_replacement"]:
            anchor_online_role = (
                "source_coordinate_target_state_raw_action_noop_delta_"
                "with_native_generation_phase_timing"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["native_targetstate_sparse25_raw_delta_hard_replacement"]:
            anchor_online_role = (
                "source_coordinate_target_state_raw_action_noop_delta_top25_"
                "with_native_generation_phase_timing"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["targetstate_raw_delta_hard_replacement"]:
            anchor_online_role = (
                "matched_control_source_coordinate_target_state_raw_action_noop_"
                "delta_without_native_generation_timing"
            )
            anchor_forward_scope = (
                "native_matched_compute_once_per_step_plus_target_state_every_"
                "sga_candidate"
            )
        elif trace["targetstate_sparse25_raw_delta_hard_replacement"]:
            anchor_online_role = (
                "matched_control_source_coordinate_target_state_raw_action_noop_"
                "delta_top25_without_native_generation_timing"
            )
            anchor_forward_scope = (
                "native_matched_compute_once_per_step_plus_target_state_every_"
                "sga_candidate"
            )
        elif trace["native_t2v_delta_velocity_hard_replacement"]:
            anchor_online_role = (
                "native_t2v_action_minus_noop_hard_replaces_entire_edit_delta"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace["anchor_native_phase_envelope_gated_target_state_velocity"]:
            anchor_online_role = (
                "native_trajectory_phase_gated_target_state_action_noop_"
                "velocity_route_not_anchor_pixels_or_spatial_coordinates"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif trace[
            "anchor_target_state_action_noop_velocity_contrast_transported"
        ]:
            anchor_online_role = (
                "target_state_action_noop_velocity_route_with_matched_"
                "native_trajectory_control_not_consumed"
            )
            anchor_forward_scope = (
                "native_once_per_active_step_plus_target_state_every_sga_candidate"
            )
        elif native_trajectory_enabled and velocity_transport_enabled:
            anchor_online_role = (
                "native_action_noop_trajectory_velocity_route_"
                "not_anchor_pixels_or_clean_latent"
            )
            anchor_forward_scope = "once_per_active_solver_step_shared_by_sga_candidates"
        elif trace["event01_role_graph_attention_enabled"]:
            anchor_online_role = (
                "native_action_noop_actor_object_attention_graph_applied_to_"
                "source_owned_actor_and_stone_value_regions"
            )
            anchor_forward_scope = (
                "native_trajectory_once_per_step_plus_action_noop_block_capture_"
                "and_target_replay_per_active_sga_candidate"
            )
        elif block_transport_enabled:
            anchor_online_role = "online_block_action_route_not_appearance_value_authority"
            anchor_forward_scope = "every_active_solver_step_and_candidate"
        else:
            anchor_online_role = "sga_action_reward_authority_not_model_injection"
            anchor_forward_scope = "none"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "training_performed": False,
            "optimization_steps": 0,
            "loaded_trained_attention_checkpoint": trained_attention is not None,
            "trained_attention_checkpoint": (
                {
                    "path": str(trained_attention["root"]),
                    "schema_version": trained_attention["schema_version"],
                    "global_step": trained_attention["global_step"],
                    "training_objective": trained_attention["training_objective"],
                    "route_operator": trained_attention["route_operator"],
                    "required_decode_transport": trained_attention[
                        "required_decode_transport"
                    ],
                    "receipt_sha256": trained_attention["receipt_sha256"],
                    "adapter_config_sha256": trained_attention[
                        "adapter_config_sha256"
                    ],
                    "adapter_model_sha256": trained_attention["model_sha256"],
                    "checkpoint_binding": trained_attention["binding"],
                    "checkpoint_binding_sha256": trained_attention[
                        "binding_sha256"
                    ],
                    "expectations_fail_closed": {
                        "global_step": args.expected_trained_attention_step,
                        "training_objective": (
                            args.expected_trained_attention_objective
                        ),
                        "route_operator": (
                            args.expected_trained_attention_route_operator
                        ),
                        "adapter_model_sha256": (
                            args.expected_trained_attention_adapter_sha256
                        ),
                        "adapter_config_sha256": (
                            args.expected_trained_attention_adapter_config_sha256
                        ),
                        "receipt_sha256": (
                            args.expected_trained_attention_receipt_sha256
                        ),
                        "all_validated": True,
                    },
                    "loaded_scope": "all_30_blocks_attn1_attn2_qkvo",
                    "dense_flow_sidecar_loaded": False,
                    "adapter_kept_unmerged": True,
                    "frozen_anchor_calls_use_disable_adapter": True,
                    "target_and_source_editor_calls_keep_adapter_enabled": True,
                    "adapter_enabled_for_target_source_calls": True,
                    "anchor_injection_enabled": not trained_route_off_control,
                    "same_checkpoint_route_off_causal_control": (
                        trained_route_off_control
                    ),
                }
                if trained_attention is not None
                else None
            ),
            "model": "Bernini-R-1.3B-Diffusers",
            "causal_control": {
                "enabled": trained_route_off_control,
                "kind": (
                    "same_trained_checkpoint_route_off"
                    if trained_route_off_control
                    else None
                ),
                "explicit_opt_in": args.allow_trained_route_off_control,
                "trained_adapter_loaded": trained_attention is not None,
                "adapter_enabled_for_target_source_calls": (
                    trained_attention is not None
                ),
                "anchor_injection_enabled": args.transport_steps > 0,
                "transport_steps": args.transport_steps,
            },
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "checkpoint_content": checkpoint_result["identity"],
            "source": {
                "path": str(source_path),
                "sha256": source_sha,
                "metadata": source_metadata,
                "role": "clean_edit_state_identity_appearance_scene_authority",
            },
            "pure_t2v_anchor": {
                "path": str(anchor_paths[0]),
                "sha256": anchor_shas[0],
                "metadata": anchor_metadatas[0],
                "source_bucket_normalization": anchor_resizes[0],
                "role": anchor_online_role,
                "model_forward_at_every_active_solver_step_and_candidate": (
                    "every_sga_candidate" in anchor_forward_scope
                    or anchor_forward_scope
                    == "every_active_solver_step_and_candidate"
                ),
                "online_model_forward_scope": anchor_forward_scope,
                "used_by_sga_action_reward": trace[
                    "anchor_action_reward_used_for_sga"
                ],
                "active_solver_steps": args.transport_steps,
            },
            "pure_t2v_anchor_bank": [
                {
                    "index": index,
                    "path": str(path),
                    "sha256": sha,
                    "metadata": metadata,
                    "source_bucket_normalization": resize,
                    "role": (
                        anchor_online_role
                        if len(anchor_paths) == 1
                        else "candidate_specific_online_action_route"
                        if args.transport_steps > 0
                        else "sga_action_reward_authority"
                    ),
                }
                for index, (path, sha, metadata, resize) in enumerate(
                    zip(anchor_paths, anchor_shas, anchor_metadatas, anchor_resizes)
                )
            ],
            "anchor_generation_initial_gaussian": (
                {
                    "path": str(anchor_initial_gaussian_path),
                    "file_sha256": args.expected_anchor_initial_gaussian_sha256,
                    "tensor_identity": anchor_initial_gaussian_identity,
                    "role": (
                        "native_t2v_action_noop_trajectory_initial_gaussian_"
                        "not_sga_candidate"
                        if args.anchor_state_mode == "native_t2v_trajectory"
                        and args.initial_noise_proposal_mode == "keyed_only"
                        else "dynaedit_step0_candidate0_native_generation_noise"
                    ),
                }
                if anchor_initial_gaussian_path is not None
                else None
            ),
            "prompts": {
                "action_mv2v_sha256": hashlib.sha256(action_prompt.encode()).hexdigest(),
                "source_noop_mv2v_sha256": hashlib.sha256(noop_prompt.encode()).hexdigest(),
                "anchor_t2v_sha256": hashlib.sha256(anchor_prompt.encode()).hexdigest(),
                "anchor_noop_t2v_sha256": hashlib.sha256(anchor_noop_prompt.encode()).hexdigest(),
                "source_t2v_sha256": hashlib.sha256(source_t2v_prompt.encode()).hexdigest(),
                "target_t2v_sha256": hashlib.sha256(target_t2v_prompt.encode()).hexdigest(),
                "negative_sha256": hashlib.sha256(DEFAULT_NEG_PROMPT.encode()).hexdigest(),
            },
            "mechanism": {
                "arm": args.arm,
                "transport": args.transport,
                "transport_strength": args.transport_strength,
                "transport_steps": args.transport_steps,
                "initial_phase_clamp": not args.no_initial_phase_clamp,
                "field_guidance": args.field_guidance,
                "field_model": args.field_model,
                "source_cfg_scale": args.source_cfg_scale,
                "target_cfg_scale": args.target_cfg_scale,
                "sga_temperature": args.sga_temperature,
                "early_candidate_count": args.early_candidate_count,
                "initial_noise_proposal_mode": args.initial_noise_proposal_mode,
                "anchor_state_mode": args.anchor_state_mode,
                "anchor_cfg_scope": args.anchor_cfg_scope,
                "anchor_contrast_mode": args.anchor_contrast_mode,
                "anchor_sigma_cap": args.anchor_sigma_cap,
                "preservation_mode": args.preservation_mode,
                "preservation_keep_fraction": args.preservation_keep_fraction,
                "preservation_outside_scale": args.preservation_outside_scale,
                "preservation_dilation": args.preservation_dilation,
                "preservation_residual_fraction": args.preservation_residual_fraction,
                "preservation_object_identity_strength": (
                    args.preservation_object_identity_strength
                ),
                "preservation_start_step": args.preservation_start_step,
                "preservation_ramp_steps": args.preservation_ramp_steps,
                "sga_score_mode": args.sga_score_mode,
                "anchor_candidate_mode": args.anchor_candidate_mode,
                "anchor_bank_size": len(anchor_paths),
                "anchor_spatial_alignment": args.anchor_spatial_alignment,
                "event01_forced_role_proposal_index": (
                    args.event01_forced_role_proposal_index
                ),
                "decode_audit_contract": {
                    "transport_steps": args.transport_steps,
                    "anchor_state_mode": args.anchor_state_mode,
                    "anchor_cfg_scope": args.anchor_cfg_scope,
                    "source_cfg_scale": args.source_cfg_scale,
                    "target_cfg_scale": args.target_cfg_scale,
                    "source_and_target_cfg_equal": (
                        args.source_cfg_scale == args.target_cfg_scale
                    ),
                    "pure_t2v_teacher_adapter_policy": (
                        "disable_loaded_editor_adapter"
                        if trained_attention is not None
                        else "plain_frozen_base"
                    ),
                    "target_source_editor_adapter_policy": (
                        "loaded_adapter_enabled"
                        if trained_attention is not None
                        else "plain_frozen_base"
                    ),
                    "trained_route_off_control_explicitly_allowed": (
                        args.allow_trained_route_off_control
                    ),
                    "same_checkpoint_route_off_causal_control": (
                        trained_route_off_control
                    ),
                    "anchor_injection_enabled": args.transport_steps > 0,
                },
                "selected_blocks": list(selected_blocks),
                "outer_sampler": "clean_source_state_plus_SGA_plus_ANC",
                "attention_transport": (
                    "online_anchor_post_rope_transport_bound_by_trace"
                    if block_transport_enabled
                    else "none"
                ),
                "pure_t2v_anchor_online_block_transport_enabled": block_transport_enabled,
                "pure_t2v_anchor_online_velocity_transport_enabled": (
                    velocity_transport_enabled
                ),
                "pure_t2v_anchor_used_as_sga_reward_authority": trace[
                    "anchor_action_reward_used_for_sga"
                ],
                "pure_t2v_anchor_values_or_pixels_copied_to_output": False,
                "trace": trace,
                "trace_digest": trace_digest,
            },
            "output": {
                "path": str(output_path),
                "sha256": file_sha256(output_path),
                "frames": FRAME_COUNT,
                "fps": FPS,
                "height": bucket_hw[0],
                "width": bucket_hw[1],
            },
            "rank_closure": rank_rows,
            "freeze_before": freeze_before,
            "freeze_after": freeze_after,
        }
        receipt_path.write_bytes(_canonical_json(receipt) + b"\n")
        print(_canonical_json(receipt).decode("utf-8"), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
