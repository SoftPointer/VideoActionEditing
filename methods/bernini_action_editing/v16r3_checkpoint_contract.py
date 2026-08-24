#!/usr/bin/env python3
"""Shared fail-closed contract for every saved v16r3 training checkpoint.

The terminal S644 inference wrapper predates checkpoint-sweep diagnostics and
therefore intentionally authenticates only S644.  This module is a separate
contract for the sealed save points of the same continuous run.  It keeps the
three artifact SHA-256 values external, validates the prefix-dependent receipt
semantics, and authenticates the safetensors header without importing Torch or
PEFT.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_lora as delegated  # noqa: E402


TRAINING_RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r3"
)
TRAINING_METHOD = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r3"
)
TRAINING_OBJECTIVE = "real_source_target_owned_routed_teacher_delta_v14r2"
ROUTE_OPERATOR = "self_target_owned_activity_kernel25_v14r2"
REQUIRED_DECODE_TRANSPORT = (
    "self_target_owned_activity_kernel25_attn_output_v14r2"
)
TRAINING_CLAIM_SCOPE = (
    "engineering_training_run_only_non_scientific_until_held_out_evaluation"
)
SAVE_STEPS = (1, 4, 8, 16, 28, 32, 64, 128, 256, 359, 512, 644)
MAX_STEPS = 644
S279_STEP = 279
S279_TARGET_IID = "4aeb0557a94b4db3"
S279_TARGET_FAMILY = "fall"
S279_EXPECTED_CALLS = (
    {"role": "action_micro_0", "seed": 1656484053, "timestep": 1000.0},
    {"role": "raw_replay_micro_0", "seed": 1657484056, "timestep": 580.0},
    {"role": "action_micro_1", "seed": 718898016, "timestep": 764.0},
    {"role": "raw_replay_micro_1", "seed": 719898019, "timestep": 880.0},
)
ZERO_RMS_POLICY = "exact_forward_zero_rms_zero_subgradient_v1"
ZERO_RMS_SCOPE = ("current_temporal_rms", "route_rms")
LORA_RANK = 256
LORA_ALPHA = 256
LORA_SCOPE = "all_30_blocks_attn1_attn2_qkvo"
TARGET_MODULE_COUNT = 240
ADAPTER_TENSOR_COUNT = 480
TRAINABLE_PARAMETER_COUNT = 188_743_680
PEFT_VERSION = "0.19.1"
TRANSFORMERS_VERSION = "5.5.4"
TARGET_MODULES_SHA256 = (
    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
)
S1_NONZERO_NAMES_SHA256 = (
    "a930be06afc7cefef1a485fa8ef6dd42c3e77749c942adecb68431c0496a7d95"
)
FULL_NONZERO_NAMES_SHA256 = (
    "2045270379ada5217ae5451f1f7187ccb276355b86653fe3f5b2694908d9434e"
)
FALLBACK_POLICY = (
    "unanimous_actual_action_ascent_parameter_rollback_adamw_state_reset_"
    "action_only_retry_once_v16r2"
)
FALLBACK_REASON = "actual_adamw_parameter_displacement_failed_action_descent_gate"
CHECKPOINT_BINDING_SCHEMA = "bernini-v16r3-three-sha-checkpoint-binding-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFETENSORS_HEADER_LIMIT = 4 * 1024 * 1024


class V16R3CheckpointContractError(delegated.InferenceContractError):
    """Raised before model construction when a checkpoint differs."""


@dataclass(frozen=True)
class CheckpointBundle:
    checkpoint_root: Path
    adapter_dir: Path
    adapter_config_path: Path
    adapter_model_path: Path
    training_receipt_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise V16R3CheckpointContractError(
            "v16r3 contract contains a non-canonical JSON value"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def expected_target_modules() -> list[str]:
    targets = delegated.expected_lora_target_modules()
    if len(targets) != TARGET_MODULE_COUNT:
        raise V16R3CheckpointContractError(
            "delegated target registry no longer has 240 modules"
        )
    if delegated.object_sha256(targets) != TARGET_MODULES_SHA256:
        raise V16R3CheckpointContractError(
            "delegated target registry digest differs"
        )
    return targets


def _strict_equal(observed: Any, expected: Any) -> bool:
    return canonical_json_bytes(observed) == canonical_json_bytes(expected)


def _require_field(
    mapping: Mapping[str, Any], key: str, expected: Any, *, label: str
) -> None:
    if key not in mapping or not _strict_equal(mapping[key], expected):
        raise V16R3CheckpointContractError(
            f"{label} differs for {key}: {mapping.get(key)!r}"
        )


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V16R3CheckpointContractError(
            f"{label} must be an explicit lowercase SHA-256"
        )
    return value


def require_save_step(value: Any, *, label: str = "expected global step") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SAVE_STEPS:
        raise V16R3CheckpointContractError(
            f"{label} must be one of the sealed v16r3 save steps: {SAVE_STEPS}"
        )
    return value


def validate_runtime_versions() -> None:
    for distribution, expected in (
        ("transformers", TRANSFORMERS_VERSION),
        ("peft", PEFT_VERSION),
    ):
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise V16R3CheckpointContractError(
                f"required runtime distribution is absent: {distribution}"
            ) from error
        if actual != expected:
            raise V16R3CheckpointContractError(
                f"v16r3 runtime {distribution} must be {expected}, got {actual}"
            )


def _expected_peft_config_without_targets() -> dict[str, Any]:
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "BerniniRendererModel",
            "parent_library": "bernini.models.renderer",
        },
        "base_model_name_or_path": "",
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
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


def validate_peft_config(adapter_config: Mapping[str, Any]) -> None:
    if not isinstance(adapter_config, Mapping):
        raise V16R3CheckpointContractError("adapter config must be a JSON object")
    if set(adapter_config) != delegated.FULL644_PEFT_CONFIG_FIELDS:
        raise V16R3CheckpointContractError(
            "v16r3 adapter PEFT 0.19.1 field closure differs"
        )
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not all(
        isinstance(item, str) for item in targets
    ):
        raise V16R3CheckpointContractError(
            "v16r3 adapter target_modules must be an explicit string list"
        )
    serialized = set(targets)
    if len(serialized) != len(targets) or serialized not in (
        set(expected_target_modules()),
        set(delegated.PEFT_COMPACT_TARGET_MODULES),
    ):
        raise V16R3CheckpointContractError(
            "v16r3 adapter is not exact all-30-block attn1/attn2 q/k/v/out"
        )
    observed = dict(adapter_config)
    observed.pop("target_modules")
    if not _strict_equal(observed, _expected_peft_config_without_targets()):
        raise V16R3CheckpointContractError(
            "v16r3 adapter PEFT semantic closure differs"
        )


def _validate_prefix_lists(contract: Mapping[str, Any], *, step: int) -> None:
    target_iids = contract.get("actual_distinct_target_iids")
    donor_iids = contract.get("actual_distinct_same_iid_role1_donor_iids")
    target_families = contract.get("actual_distinct_target_families")
    if (
        not isinstance(target_iids, list)
        or not all(isinstance(item, str) and item for item in target_iids)
        or len(target_iids) != step
        or len(set(target_iids)) != step
        or target_iids != sorted(target_iids)
    ):
        raise V16R3CheckpointContractError(
            "v16r3 target IID prefix inventory differs"
        )
    if donor_iids != target_iids:
        raise V16R3CheckpointContractError(
            "v16r3 same-IID donor inventory differs from target inventory"
        )
    expected_family_count = min(step, 28)
    if (
        not isinstance(target_families, list)
        or not all(isinstance(item, str) and item for item in target_families)
        or len(target_families) != expected_family_count
        or len(set(target_families)) != expected_family_count
        or target_families != sorted(target_families)
    ):
        raise V16R3CheckpointContractError(
            "v16r3 target-family prefix inventory differs"
        )


def _validate_full644_prefix(
    receipt: Mapping[str, Any], contract: Mapping[str, Any], *, step: int
) -> None:
    terminal = step == MAX_STEPS
    for key, expected in {
        "actual_distinct_target_iid_count": step,
        "actual_distinct_same_iid_role1_donor_count": step,
        "actual_distinct_target_family_count": min(step, 28),
        "all_full644_rows_targeted_exactly_once": terminal,
        "family_round_robin_first28_cover_all_families": (
            True if step >= 28 else None
        ),
    }.items():
        _require_field(contract, key, expected, label="v16r3 training contract")
    _validate_prefix_lists(contract, step=step)

    summary = receipt.get("v16_full644_summary")
    if not isinstance(summary, Mapping):
        raise V16R3CheckpointContractError("v16r3 full644 summary is absent")
    for key, expected in {
        "manifest_row_count": MAX_STEPS,
        "manifest_family_count": 28,
        "target_prefix_row_count": step,
        "target_prefix_exact_once": True,
        "family_round_robin_first28_cover_all_families": (
            True if step >= 28 else None
        ),
        "actual_target_family_count": min(step, 28),
        "all_full644_rows_targeted_exactly_once": terminal,
        "donor_selection_count": 2 * step,
        "same_iid_role1_donor_count": 2 * step,
        "distinct_donor_iid_count": step,
        "anchor_cross_appearance": False,
        "pair_decode_count": step,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
        "scientific_claim_authorized": False,
    }.items():
        _require_field(summary, key, expected, label="v16r3 full644 summary")
    for key in ("manifest_sha256", "manifest_digest", "target_prefix_iids_sha256"):
        require_sha256(summary.get(key), label=f"v16r3 full644 summary {key}")


def _validate_fallback(
    receipt: Mapping[str, Any], contract: Mapping[str, Any], *, step: int
) -> None:
    summary = receipt.get("v16r2_actual_action_descent_fallback_summary")
    if not isinstance(summary, Mapping):
        raise V16R3CheckpointContractError("v16r3 fallback summary is absent")
    count = summary.get("fallback_count")
    steps = summary.get("fallback_steps")
    iids = summary.get("fallback_target_iids")
    geometry = summary.get("fallback_geometry")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(steps, list)
        or not isinstance(iids, list)
        or not isinstance(geometry, list)
        or len(steps) != len(iids) != count
    ):
        # Keep an explicit expression below as chained inequality does not
        # prove equality among all four lengths.
        raise V16R3CheckpointContractError(
            "v16r3 fallback cumulative inventory differs"
        )
    if not (len(steps) == len(iids) == len(geometry) == count):
        raise V16R3CheckpointContractError(
            "v16r3 fallback cumulative inventory differs"
        )
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in steps)
        or steps != sorted(set(steps))
        or any(item <= 0 or item > step for item in steps)
        or not all(isinstance(item, str) and item for item in iids)
    ):
        raise V16R3CheckpointContractError(
            "v16r3 fallback step/IID history differs"
        )
    if any(
        not isinstance(row, Mapping)
        or row.get("step") != fallback_step
        or row.get("target_iid") != target_iid
        or row.get("failed_candidate_committed") is not False
        or row.get("parameter_values_exactly_restored_before_retry") is not True
        or row.get("optimizer_state_restored") is not False
        or row.get("optimizer_state_reset") is not True
        or row.get("committed_retry_reprobed_by_frozen_authority") is not True
        or not isinstance(row.get("committed_retry"), Mapping)
        or row["committed_retry"].get("action_descent_passed") is not True
        for row, fallback_step, target_iid in zip(geometry, steps, iids)
    ):
        raise V16R3CheckpointContractError(
            "v16r3 fallback event geometry differs"
        )
    uninterrupted = count == 0
    for key, expected in {
        "policy": FALLBACK_POLICY,
        "reason": FALLBACK_REASON,
        "optimizer_state_reset_count": count,
        "failed_candidates_committed": False,
        "parameter_values_exactly_restored_before_each_retry": True,
        "optimizer_state_restored": False,
        "optimizer_state_reset_before_each_retry": True,
        "committed_retry_gradient": "primary_action_only_clipped",
        "retry_limit_per_failed_candidate": 1,
        "committed_retries_reprobed_by_frozen_authority": True,
        "action_descent_gate_relaxed": False,
        "optimizer_history_matches_uninterrupted_adamw": uninterrupted,
        "continuous_parameter_trajectory_from_frozen_base": True,
        "scientific_claim_authorized": False,
    }.items():
        _require_field(summary, key, expected, label="v16r3 fallback summary")
    for key, expected in {
        "actual_action_descent_fallback_policy": FALLBACK_POLICY,
        "actual_action_descent_fallback_count": count,
        "actual_action_descent_fallback_steps": steps,
        "actual_action_descent_fallback_target_iids": iids,
        "actual_action_descent_failed_candidates_committed": False,
        "actual_action_descent_fallback_parameter_values_exactly_restored": True,
        "actual_action_descent_fallback_optimizer_state_restored": False,
        "actual_action_descent_fallback_optimizer_state_reset_count": count,
        "actual_action_descent_fallback_uses_primary_action_only": True,
        "actual_action_descent_fallback_retry_limit": 1,
        "actual_action_descent_fallback_reprobes_frozen_authority": True,
        "actual_action_descent_gate_relaxed": False,
        "optimizer_history_matches_uninterrupted_adamw": uninterrupted,
    }.items():
        _require_field(contract, key, expected, label="v16r3 training contract")


def _validate_s279(
    receipt: Mapping[str, Any], contract: Mapping[str, Any], *, step: int
) -> None:
    covered = step >= S279_STEP
    _require_field(
        contract,
        "s279_endpoint_canary_covered",
        covered,
        label="v16r3 training contract",
    )
    _require_field(
        contract,
        "s279_endpoint_canary_target_iid",
        S279_TARGET_IID,
        label="v16r3 training contract",
    )
    summary = receipt.get("v16r3_zero_rms_backward_summary")
    if not isinstance(summary, Mapping):
        raise V16R3CheckpointContractError("v16r3 zero-RMS summary is absent")
    for key, expected in {
        "policy": ZERO_RMS_POLICY,
        "scope": list(ZERO_RMS_SCOPE),
        "active_qk_route": "qk_only_target_gated_hard_temporal_kernel_contrast_output",
        "finite_nonnegative_forward_values_bit_exact": True,
        "zero_forward_value": 0.0,
        "zero_backward_subgradient": 0.0,
        "positive_backward_matches_standard_sqrt": True,
        "negative_or_nonfinite_values_masked": False,
        "loss_scale_changed": False,
        "seed_or_timestep_changed": False,
        "sample_retry_or_skip": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "policy_fixed_from_step_one": True,
        "single_continuous_fresh_from_base_exact644": True,
        "scientific_claim_authorized": False,
    }.items():
        _require_field(summary, key, expected, label="v16r3 zero-RMS summary")
    canary = summary.get("s279_endpoint_canary")
    if not isinstance(canary, Mapping):
        raise V16R3CheckpointContractError("v16r3 S279 canary is absent")
    for key, expected in {
        "step": S279_STEP,
        "target_iid": S279_TARGET_IID,
        "target_family": S279_TARGET_FAMILY,
        "expected_calls": list(S279_EXPECTED_CALLS),
        "observed_calls": list(S279_EXPECTED_CALLS) if covered else [],
        "covered_by_checkpoint": covered,
    }.items():
        _require_field(canary, key, expected, label="v16r3 S279 canary")


def _validate_memory_and_gradient(receipt: Mapping[str, Any], *, step: int) -> None:
    memory = receipt.get("memory_gate")
    if not isinstance(memory, Mapping):
        raise V16R3CheckpointContractError("v16r3 memory gate is absent")
    for key, expected in {
        "capture_phase": "after_two_real_component_backwards_before_actual_update_audit_clones",
        "actual_update_audit_allocations_excluded": True,
        "passed": True,
        "dummy_or_padding_allocations": False,
        "true_training_tensors_only": True,
    }.items():
        _require_field(memory, key, expected, label="v16r3 memory gate")
    minimum = memory.get("minimum_reserved_fraction")
    rows = memory.get("per_rank")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not 0.5 < float(minimum) <= 1.0
        or not isinstance(rows, list)
        or len(rows) != 4
        or any(
            not isinstance(row, Mapping)
            or row.get("rank") != rank
            or isinstance(row.get("reserved_fraction"), bool)
            or not isinstance(row.get("reserved_fraction"), (int, float))
            or not math.isfinite(float(row["reserved_fraction"]))
            or not 0.5 < float(row["reserved_fraction"]) <= 1.0
            for rank, row in enumerate(rows)
        )
    ):
        raise V16R3CheckpointContractError(
            "v16r3 real-memory gate is not an exact four-rank >50% observation"
        )

    coverage = receipt.get("gradient_coverage")
    expected_nonzero = 240 if step == 1 else ADAPTER_TENSOR_COUNT
    expected_digest = (
        S1_NONZERO_NAMES_SHA256 if step == 1 else FULL_NONZERO_NAMES_SHA256
    )
    if not isinstance(coverage, Mapping):
        raise V16R3CheckpointContractError("v16r3 gradient coverage is absent")
    for key, expected in {
        "tensor_count": ADAPTER_TENSOR_COUNT,
        "nonzero_tensor_count": expected_nonzero,
        "nonzero_names_sha256": expected_digest,
    }.items():
        _require_field(coverage, key, expected, label="v16r3 gradient coverage")


def validate_v16r3_checkpoint_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_global_step: int,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate one registered v16r3 prefix checkpoint without GPU imports."""

    step = require_save_step(expected_global_step)
    config_sha = require_sha256(
        expected_adapter_config_sha256, label="expected adapter config SHA-256"
    )
    model_sha = require_sha256(
        expected_adapter_model_sha256, label="expected adapter model SHA-256"
    )
    receipt_sha = require_sha256(
        expected_training_receipt_sha256,
        label="expected training receipt SHA-256",
    )
    if expected_checkpoint_tree_sha256 != delegated.trainer.CHECKPOINT_TREE_SHA256:
        raise V16R3CheckpointContractError(
            "v16r3 inference supports only the audited Bernini base checkpoint tree"
        )
    if not isinstance(training_receipt, Mapping):
        raise V16R3CheckpointContractError("training receipt must be a JSON object")
    validate_peft_config(adapter_config)

    for key, expected in {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "bernini_commit": delegated.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": delegated.trainer.VEOMNI_TESTED_COMMIT,
        "global_step": step,
        "max_steps": MAX_STEPS,
        "complete": True,
        "scientific_claim_authorized": False,
        "claim_scope": TRAINING_CLAIM_SCOPE,
        "last_reporting_scalar_is_not_a_joint_backpropagated_objective": True,
    }.items():
        _require_field(training_receipt, key, expected, label="v16r3 receipt")
    for key in ("method_source_revision", "method_source_archive_sha256"):
        require_sha256(training_receipt.get(key), label=f"v16r3 receipt {key}")

    contract = training_receipt.get("training_contract")
    if not isinstance(contract, Mapping):
        raise V16R3CheckpointContractError(
            "v16r3 receipt lacks its training_contract"
        )
    terminal = step == MAX_STEPS
    for key, expected in {
        "method": TRAINING_METHOD,
        "training_objective": TRAINING_OBJECTIVE,
        "route_operator": ROUTE_OPERATOR,
        "route_transport": REQUIRED_DECODE_TRANSPORT,
        "target_owned_qk_route_v14r2": True,
        "anchor_donor_cached_fields": ["query", "key"],
        "anchor_donor_value_cached_or_used_by_route": False,
        "anchor_donor_hidden_or_attention_output_cached_or_used_by_route": False,
        "anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route": False,
        "anchor_to_target_appearance_correspondence_used": False,
        "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel": True,
        "anchor_qk_phase0_only_difference_produces_zero_route": True,
        "dynaedit_sga_anc_reserved_for_decode_solver": True,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_scope": LORA_SCOPE,
        "lora_target_module_count": TARGET_MODULE_COUNT,
        "lora_target_modules_sha256": TARGET_MODULES_SHA256,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "full_attention_lora_enabled": True,
        "full644_optimizer_schedule": "exact644_unique_rows_once",
        "all_full644_rows_targeted_exactly_once": terminal,
        "single_continuous_fresh_from_base_exact644_run": True,
        "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
        "micro_semantics": "different_seed_same_iid_role1_action_anchor",
        "anchor_route_replay_uses_per_capture": 2,
        "teacher_delta_mode": "raw",
        "routed_teacher_mode": "same_action_route_only",
        "student_route_off_branch_stop_gradient": True,
        "action_objective_backpropagates_only_routed_student_query": True,
        "routed_teacher_cross_caption_source_branch": False,
        "source_reconstruction_weight": None,
        "source_reconstruction_weight_argument": 0.025,
        "base_replay_scale": 0.025,
        "replay_combine_mode": "action_priority_pcgrad_010",
        "qk_only_zero_rms_backward_policy": ZERO_RMS_POLICY,
        "qk_only_zero_rms_backward_scope": list(ZERO_RMS_SCOPE),
        "qk_only_zero_rms_forward_values_changed": False,
        "qk_only_zero_rms_zero_subgradient": 0.0,
        "sample_retry_or_skip_for_v16r3": False,
        "seed_or_timestep_changed_for_v16r3": False,
        "loss_scale_changed_for_v16r3": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "scientific_claim_authorized": False,
    }.items():
        _require_field(contract, key, expected, label="v16r3 training contract")

    _validate_full644_prefix(training_receipt, contract, step=step)
    _validate_fallback(training_receipt, contract, step=step)
    _validate_s279(training_receipt, contract, step=step)
    _validate_memory_and_gradient(training_receipt, step=step)

    return {
        "global_step": step,
        "max_steps": MAX_STEPS,
        "checkpoint_complete": True,
        "terminal_full644_checkpoint": terminal,
        # infer_lora's legacy field name is retained only as an internal API
        # adapter.  The published wrapper labels this value as a file SHA.
        "receipt_digest": receipt_sha,
        "training_receipt_sha256": receipt_sha,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "target_modules_sha256": TARGET_MODULES_SHA256,
        "target_modules": expected_target_modules(),
        "target_module_count": TARGET_MODULE_COUNT,
        "adapter_tensor_count": ADAPTER_TENSOR_COUNT,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_scope": LORA_SCOPE,
        "peft_version": PEFT_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "training_schema_version": TRAINING_RECEIPT_SCHEMA,
        "training_method": TRAINING_METHOD,
        "training_objective": TRAINING_OBJECTIVE,
        "route_operator": ROUTE_OPERATOR,
        "required_decode_transport": REQUIRED_DECODE_TRANSPORT,
        "claim_scope": TRAINING_CLAIM_SCOPE,
        "method_source_revision": training_receipt["method_source_revision"],
        "method_source_archive_sha256": training_receipt[
            "method_source_archive_sha256"
        ],
        "v16r3_online_anchor": True,
    }


def resolve_checkpoint(value: str | Path, *, expected_global_step: int) -> CheckpointBundle:
    step = require_save_step(expected_global_step)
    try:
        delegated_bundle = delegated.resolve_adapter_bundle(value)
    except delegated.InferenceContractError as error:
        raise V16R3CheckpointContractError(str(error)) from error
    root = delegated_bundle.checkpoint_root
    if root.name != f"checkpoint-{step:08d}":
        raise V16R3CheckpointContractError(
            "v16r3 checkpoint directory differs from expected global step"
        )
    for path, label, directory in (
        (root, "checkpoint root", True),
        (delegated_bundle.adapter_dir, "adapter directory", True),
        (delegated_bundle.adapter_config_path, "adapter config", False),
        (delegated_bundle.adapter_model_path, "adapter model", False),
        (delegated_bundle.training_receipt_path, "training receipt", False),
    ):
        if path.is_symlink():
            raise V16R3CheckpointContractError(f"v16r3 {label} must not be a symlink")
        mode = path.lstat().st_mode
        if (directory and not stat.S_ISDIR(mode)) or (
            not directory and not stat.S_ISREG(mode)
        ):
            raise V16R3CheckpointContractError(f"v16r3 {label} has the wrong file type")
    return CheckpointBundle(
        checkpoint_root=root,
        adapter_dir=delegated_bundle.adapter_dir,
        adapter_config_path=delegated_bundle.adapter_config_path,
        adapter_model_path=delegated_bundle.adapter_model_path,
        training_receipt_path=delegated_bundle.training_receipt_path,
    )


def file_sha256(path: Path, *, label: str) -> str:
    if (
        str(path) in delegated._AUTHORIZED_FD_VIEW_FILES
        and delegated._ACTIVE_INHERITED_FDS is not None
    ):
        return delegated.file_sha256(path)
    try:
        before = path.lstat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            fd_before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            fd_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as error:
        raise V16R3CheckpointContractError(
            f"cannot stably hash {label}: {path}: {error}"
        ) from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (
        identity(before)
        == identity(fd_before)
        == identity(fd_after)
        == identity(after)
    ):
        raise V16R3CheckpointContractError(f"{label} changed while hashing: {path}")
    return digest.hexdigest()


def verify_checkpoint_hashes(
    bundle: CheckpointBundle,
    *,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    include_model: bool,
) -> None:
    rows = [
        (
            bundle.adapter_config_path,
            require_sha256(
                expected_adapter_config_sha256,
                label="expected adapter config SHA-256",
            ),
            "v16r3 adapter config",
        ),
        (
            bundle.training_receipt_path,
            require_sha256(
                expected_training_receipt_sha256,
                label="expected training receipt SHA-256",
            ),
            "v16r3 training receipt",
        ),
    ]
    if include_model:
        rows.append(
            (
                bundle.adapter_model_path,
                require_sha256(
                    expected_adapter_model_sha256,
                    label="expected adapter model SHA-256",
                ),
                "v16r3 adapter model",
            )
        )
    for path, expected, label in rows:
        actual = file_sha256(path, label=label)
        if actual != expected:
            raise V16R3CheckpointContractError(
                f"{label} SHA-256 differs: expected={expected} actual={actual}"
            )


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        before = path.lstat()
        with path.open("rb") as handle:
            fd_before = os.fstat(handle.fileno())
            raw = handle.read()
            fd_after = os.fstat(handle.fileno())
        after = path.lstat()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V16R3CheckpointContractError(f"cannot read {label}: {path}") from error
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if not (
        identity(before)
        == identity(fd_before)
        == identity(fd_after)
        == identity(after)
    ):
        raise V16R3CheckpointContractError(f"{label} changed while reading: {path}")
    if not isinstance(value, Mapping):
        raise V16R3CheckpointContractError(f"{label} must be a JSON object")
    return value


def validate_adapter_safetensors_inventory(path: Path) -> dict[str, Any]:
    """Authenticate the exact 480-tensor FP32 rank-256 LoRA header.

    Parsing only the safetensors header avoids a second 755-MiB tensor
    materialization.  The externally supplied model SHA authenticates all data
    bytes; this function proves their names, shapes, dtype, and contiguous file
    closure before PEFT sees the checkpoint.
    """

    try:
        before = path.lstat()
        with path.open("rb") as handle:
            fd_before = os.fstat(handle.fileno())
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise V16R3CheckpointContractError(
                    "v16r3 adapter safetensors header length is truncated"
                )
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= _SAFETENSORS_HEADER_LIMIT:
                raise V16R3CheckpointContractError(
                    "v16r3 adapter safetensors header length differs"
                )
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise V16R3CheckpointContractError(
                    "v16r3 adapter safetensors header is truncated"
                )
            fd_after = os.fstat(handle.fileno())
        after = path.lstat()
        header = json.loads(raw_header.decode("utf-8"))
    except V16R3CheckpointContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as error:
        raise V16R3CheckpointContractError(
            f"cannot authenticate v16r3 adapter safetensors header: {path}"
        ) from error
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if not (
        identity(before)
        == identity(fd_before)
        == identity(fd_after)
        == identity(after)
    ):
        raise V16R3CheckpointContractError(
            f"adapter safetensors changed while reading its header: {path}"
        )
    if not isinstance(header, Mapping) or header.get("__metadata__") != {"format": "pt"}:
        raise V16R3CheckpointContractError(
            "v16r3 adapter safetensors metadata differs"
        )
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    expected_keys = set(delegated.expected_adapter_state_keys(expected_target_modules()))
    if set(tensors) != expected_keys or len(tensors) != ADAPTER_TENSOR_COUNT:
        raise V16R3CheckpointContractError(
            "v16r3 adapter safetensors key inventory is not the exact 480 A/B tensors"
        )
    intervals: list[tuple[int, int]] = []
    total_elements = 0
    for key, row in tensors.items():
        if not isinstance(row, Mapping) or set(row) != {"dtype", "shape", "data_offsets"}:
            raise V16R3CheckpointContractError(
                f"v16r3 adapter tensor metadata differs for {key}"
            )
        shape = row.get("shape")
        offsets = row.get("data_offsets")
        if (
            row.get("dtype") != "F32"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            raise V16R3CheckpointContractError(
                f"v16r3 adapter tensor dtype/shape/offset differs for {key}"
            )
        if key.endswith(".lora_A.weight") and shape[0] != LORA_RANK:
            raise V16R3CheckpointContractError(
                f"v16r3 adapter A rank differs for {key}"
            )
        if key.endswith(".lora_B.weight") and shape[1] != LORA_RANK:
            raise V16R3CheckpointContractError(
                f"v16r3 adapter B rank differs for {key}"
            )
        elements = math.prod(shape)
        if offsets[1] - offsets[0] != 4 * elements:
            raise V16R3CheckpointContractError(
                f"v16r3 adapter tensor byte extent differs for {key}"
            )
        total_elements += elements
        intervals.append((offsets[0], offsets[1]))
    if total_elements != TRAINABLE_PARAMETER_COUNT:
        raise V16R3CheckpointContractError(
            "v16r3 adapter tensor element count differs"
        )
    ordered = sorted(intervals)
    if ordered[0][0] != 0 or any(
        left[1] != right[0] for left, right in zip(ordered, ordered[1:])
    ):
        raise V16R3CheckpointContractError(
            "v16r3 adapter safetensors data offsets are not contiguous"
        )
    data_bytes = int(before.st_size) - 8 - header_length
    if ordered[-1][1] != data_bytes:
        raise V16R3CheckpointContractError(
            "v16r3 adapter safetensors file extent differs"
        )
    return {
        "tensor_count": len(tensors),
        "target_module_count": len(tensors) // 2,
        "parameter_element_count": total_elements,
        "dtype": "F32",
        "rank": LORA_RANK,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
    }


def authenticate_checkpoint(
    value: str | Path,
    *,
    expected_global_step: int,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
    include_model: bool = True,
) -> dict[str, Any]:
    """Bind paths, external hashes, receipt semantics, and tensor inventory."""

    step = require_save_step(expected_global_step)
    bundle = resolve_checkpoint(value, expected_global_step=step)
    verify_checkpoint_hashes(
        bundle,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        include_model=include_model,
    )
    adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
    receipt = _read_json(bundle.training_receipt_path, label="training receipt")
    identity = validate_v16r3_checkpoint_contract(
        adapter_config,
        receipt,
        expected_global_step=step,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )
    inventory = (
        validate_adapter_safetensors_inventory(bundle.adapter_model_path)
        if include_model
        else None
    )
    binding = {
        "schema_version": CHECKPOINT_BINDING_SCHEMA,
        "receipt_sha256": identity["training_receipt_sha256"],
        "adapter_config_sha256": identity["adapter_config_sha256"],
        "adapter_model_sha256": identity["adapter_model_sha256"],
        "global_step": step,
        "max_steps": MAX_STEPS,
        "training_objective": TRAINING_OBJECTIVE,
        "route_operator": ROUTE_OPERATOR,
        "required_decode_transport": REQUIRED_DECODE_TRANSPORT,
    }
    return {
        **identity,
        "bundle": bundle,
        "adapter_config": adapter_config,
        "training_receipt": receipt,
        "adapter_tensor_inventory": inventory,
        "binding": binding,
        "binding_sha256": object_sha256(binding),
    }


def assert_same_bundle(observed: Any, expected: CheckpointBundle) -> None:
    paths = (
        observed.checkpoint_root,
        observed.adapter_dir,
        observed.adapter_config_path,
        observed.adapter_model_path,
        observed.training_receipt_path,
    )
    wanted = (
        expected.checkpoint_root,
        expected.adapter_dir,
        expected.adapter_config_path,
        expected.adapter_model_path,
        expected.training_receipt_path,
    )
    if paths != wanted:
        raise V16R3CheckpointContractError(
            "v16r3 adapter bundle changed after CLI identity binding"
        )


__all__ = [
    "ADAPTER_TENSOR_COUNT",
    "CheckpointBundle",
    "LORA_ALPHA",
    "LORA_RANK",
    "LORA_SCOPE",
    "MAX_STEPS",
    "REQUIRED_DECODE_TRANSPORT",
    "ROUTE_OPERATOR",
    "SAVE_STEPS",
    "TARGET_MODULE_COUNT",
    "TARGET_MODULES_SHA256",
    "TRAINING_METHOD",
    "TRAINING_OBJECTIVE",
    "TRAINING_RECEIPT_SCHEMA",
    "V16R3CheckpointContractError",
    "assert_same_bundle",
    "authenticate_checkpoint",
    "expected_target_modules",
    "file_sha256",
    "require_save_step",
    "require_sha256",
    "validate_adapter_safetensors_inventory",
    "validate_runtime_versions",
    "validate_v16r3_checkpoint_contract",
    "verify_checkpoint_hashes",
]
