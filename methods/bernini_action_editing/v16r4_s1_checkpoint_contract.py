#!/usr/bin/env python3
"""Fail-closed identity and semantic contract for the v16r4 S1 canary.

The production trainer is intentionally an exact-644 program.  The external
canary controller may stop its Slurm *step* only after checkpoint S1 has been
fully written.  This module authenticates that prefix without pretending the
exact-644 run completed or that the checkpoint has passed its full Heldout8
promotion contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as delegated  # noqa: E402
import v16r3_checkpoint_contract as shared  # noqa: E402


TRAINING_RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r4"
)
TRAINING_METHOD = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r4"
)
TRAINING_CLAIM_SCOPE = (
    "engineering_training_run_only_non_scientific_until_automatic_held_out_evaluation"
)
CHECKPOINT_BINDING_SCHEMA = "bernini-v16r4-s1-three-sha-checkpoint-binding-v1"
AUTHENTICATION_SCHEMA = "bernini-v16r4-s1-checkpoint-authentication-v1"
GLOBAL_STEP = 1
MAX_STEPS = 644
SAVE_STEPS = (GLOBAL_STEP,)
OPTIMIZER = "global_rms_normalized_source_halfspace_sgd_v1"
REPLAY_COMBINE_MODE = "source_halfspace_001"
LEARNING_RATE = 1.0e-6
FAILURE_POLICY = (
    "fail_closed_no_retry_no_action_only_fallback_no_optimizer_state_reset_v16r4"
)
FULL644_MANIFEST_SHA256 = (
    "61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa"
)
FULL644_MANIFEST_DIGEST = (
    "96fe6188ad0f5ee72dcd89fbc018835f3f2995e45ff116f07449e863fa9b51d5"
)
HELDOUT8_MANIFEST_SHA256 = (
    "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701"
)
HELDOUT8_IIDS_SHA256 = (
    "0cb2260efdd758a0b978a93509e510211ae8555fb480c9d1bd7f9359f6c7740a"
)
S1_TARGET_IID = "0e3ba817b0ae4f28"
S1_TARGET_FAMILY = "climb"
S1_PREFIX_IIDS_SHA256 = (
    "a6f89dd61912f80f3506f1e70a89ac8c2bd1d1214364bd9d146e6abfb02f5c88"
)

LORA_RANK = shared.LORA_RANK
LORA_ALPHA = shared.LORA_ALPHA
LORA_SCOPE = shared.LORA_SCOPE
TARGET_MODULE_COUNT = shared.TARGET_MODULE_COUNT
ADAPTER_TENSOR_COUNT = shared.ADAPTER_TENSOR_COUNT
TRAINABLE_PARAMETER_COUNT = shared.TRAINABLE_PARAMETER_COUNT
TARGET_MODULES_SHA256 = shared.TARGET_MODULES_SHA256
PEFT_VERSION = shared.PEFT_VERSION
TRANSFORMERS_VERSION = shared.TRANSFORMERS_VERSION
TRAINING_OBJECTIVE = shared.TRAINING_OBJECTIVE
ROUTE_OPERATOR = shared.ROUTE_OPERATOR
REQUIRED_DECODE_TRANSPORT = shared.REQUIRED_DECODE_TRANSPORT


class V16R4S1CheckpointContractError(delegated.InferenceContractError):
    """Raised before model construction when the S1 contract differs."""


CheckpointBundle = shared.CheckpointBundle


def _fail(message: str) -> None:
    raise V16R4S1CheckpointContractError(message)


def _equal(observed: Any, expected: Any) -> bool:
    return shared.canonical_json_bytes(observed) == shared.canonical_json_bytes(
        expected
    )


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if key not in mapping or not _equal(mapping[key], expected):
        _fail(f"{label} differs for {key}: {mapping.get(key)!r}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object")
    return value


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"{label} is outside its finite contract")
    return result


def require_save_step(value: Any, *, label: str = "expected global step") -> int:
    if isinstance(value, bool) or value != GLOBAL_STEP:
        _fail(f"{label} must be the isolated v16r4 S1 checkpoint")
    return GLOBAL_STEP


def require_sha256(value: Any, *, label: str) -> str:
    try:
        return shared.require_sha256(value, label=label)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def expected_target_modules() -> list[str]:
    return shared.expected_target_modules()


def validate_runtime_versions() -> None:
    try:
        shared.validate_runtime_versions()
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def _validate_optimizer_geometry(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    interaction = _mapping(
        _mapping(receipt.get("component_gradient_probes"), label="component probes").get(
            "interaction"
        ),
        label="component-gradient interaction",
    )
    for key, expected in {
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "first_order_source_fm_preserved": True,
        "v16r4_source_descent_required": True,
        "v16r4_action_descent_required": True,
        "v16r4_action_only_fallback_allowed": False,
        "v16r4_optimizer_state_reset_allowed": False,
    }.items():
        _require(interaction, key, expected, label="v16r4 interaction")
    if (
        _finite(
            interaction.get("action_gradient_dot_combined_gradient_fp64"),
            label="formal action inner product",
        )
        <= 0.0
        or _finite(
            interaction.get("raw_replay_gradient_dot_combined_gradient_fp64"),
            label="formal source inner product",
        )
        < -1.0e-8
        or _finite(
            interaction.get("action_alignment_ratio"),
            label="formal action alignment ratio",
        )
        < 0.1
    ):
        _fail("v16r4 formal S1 dual-descent geometry differs")

    probe = _mapping(
        receipt.get("actual_optimizer_update_probe"),
        label="actual optimizer update probe",
    )
    for key, expected in {
        "schema_version": "bernini-actual-optimizer-update-probe-v1",
        "step": GLOBAL_STEP,
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "optimizer_semantics_observed_not_modified": True,
        "action_descent_required": True,
        "action_descent_passed": True,
        "source_descent_required": True,
        "source_descent_passed": True,
        "v16r4_optimizer": OPTIMIZER,
        "v16r4_probe_retry_count": 0,
        "v16r4_action_only_fallback_applied": False,
        "v16r4_optimizer_state_reset": False,
        "v16r4_failed_candidate_checkpoint_publication_allowed": False,
    }.items():
        _require(probe, key, expected, label="v16r4 actual update probe")
    if any(
        _finite(probe.get(key), label=f"v16r4 probe {key}", positive=True) <= 0.0
        for key in (
            "action_descent_fp64",
            "source_descent_fp64",
            "delta_theta_l2_norm_fp64",
        )
    ):
        _fail("v16r4 actual S1 displacement is not nontrivial dual descent")
    relative_error = _finite(
        probe.get("v16r4_actual_vs_planned_delta_l2_relative_error"),
        label="actual/planned displacement relative error",
    )
    if not 0.0 <= relative_error <= 1.0e-3:
        _fail("v16r4 stored S1 displacement differs from its plan")
    optimizer_step = _mapping(
        probe.get("v16r4_optimizer_step"), label="v16r4 optimizer step"
    )
    for key, expected in {
        "schema_version": "bernini-global-rms-projected-sgd-step-v1",
        "step": GLOBAL_STEP,
        "optimizer": OPTIMIZER,
        "learning_rate_active_coordinate_rms": LEARNING_RATE,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "coordinatewise_preconditioner": False,
        "global_gradient_direction_preserved_before_storage_rounding": True,
    }.items():
        _require(optimizer_step, key, expected, label="v16r4 optimizer step")

    source = _mapping(
        receipt.get("v16r4_source_descent_summary"),
        label="v16r4 source-descent summary",
    )
    for key, expected in {
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "optimizer": OPTIMIZER,
        "optimizer_scalar_learning_rate": LEARNING_RATE,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "coordinatewise_preconditioner": False,
        "global_positive_direction_scale_only": True,
        "global_positive_scale_preserves_both_formal_halfspaces": True,
        "formal_source_descent_required": True,
        "actual_optimizer_source_descent_required": True,
        "actual_optimizer_action_descent_required": True,
        "successful_update_count": GLOBAL_STEP,
        "optimizer_failure_policy": FAILURE_POLICY,
        "action_only_fallback_allowed": False,
        "optimizer_retry_allowed": False,
        "optimizer_state_reset_allowed": False,
        "optimizer_state_entry_count": 0,
        "optimizer_has_no_momentum_or_history_by_design": True,
        "distributed_probe_agreement_required": True,
        "failed_candidate_checkpoint_publication_allowed": False,
        "per_sample_manual_or_visual_optimizer_gate": False,
        "decoded_source_preservation_claimed": False,
        "scientific_claim_authorized": False,
    }.items():
        _require(source, key, expected, label="v16r4 source-descent summary")
    if not _equal(source.get("last_optimizer_step"), optimizer_step):
        _fail("v16r4 last optimizer-step receipts differ")

    for key, expected in {
        "optimizer": OPTIMIZER,
        "optimizer_scalar_learning_rate": LEARNING_RATE,
        "optimizer_momentum": 0.0,
        "optimizer_weight_decay": 0.0,
        "optimizer_coordinatewise_preconditioner": False,
        "optimizer_global_positive_direction_scale_only": True,
        "optimizer_global_positive_scale_preserves_both_formal_halfspaces": True,
        "coordinatewise_adaptive_preconditioner_forbidden": True,
        "source_gradient_preservation_enforced": True,
        "formal_source_descent_required": True,
        "actual_optimizer_source_descent_required": True,
        "actual_optimizer_action_descent_required": True,
        "optimizer_failure_policy": FAILURE_POLICY,
        "action_only_fallback_allowed": False,
        "optimizer_retry_allowed": False,
        "optimizer_state_reset_allowed": False,
        "optimizer_state_entry_count": 0,
        "optimizer_has_no_momentum_or_history_by_design": True,
        "last_optimizer_step_schema": "bernini-global-rms-projected-sgd-step-v1",
        "distributed_probe_agreement_required": True,
        "failed_candidate_checkpoint_publication_allowed": False,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
    }.items():
        _require(contract, key, expected, label="v16r4 training contract")


def _validate_canary_contract(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    canary = _mapping(
        receipt.get("v16r4_decoded_canary_contract"),
        label="v16r4 decoded-canary contract",
    )
    for key, expected in {
        "schema_version": "bernini-v16r4-heldout8-checkpoint-canary-contract-v1",
        "input_manifest_sha256": HELDOUT8_MANIFEST_SHA256,
        "input_manifest_schema": "action-editing-shared8-input-v1",
        "case_count": 8,
        "case_iids_sha256": HELDOUT8_IIDS_SHA256,
        "training_iid_overlap_count": 0,
        "checkpoint_save_steps": list(shared.SAVE_STEPS),
        "decoded_canary_trigger_steps": [1, 8, 32, 128, 359, 644],
        "current_checkpoint_step": GLOBAL_STEP,
        "current_checkpoint_requires_decoded_canary": True,
        "arms": ["adapter_only_route_off", "trained_editor_route_on"],
        "cases_per_arm": 8,
        "training_process_executes_decode": False,
        "external_automatic_controller_executes_decode": True,
        "checkpoint_promotion_requires_decoded_canary_sidecar": True,
        "checkpoint_promotion_eligible_from_training_receipt_alone": False,
        "decoded_canary_controls_optimizer_row_admission": False,
        "per_sample_manual_review_required": False,
        "scientific_claim_authorized": False,
    }.items():
        _require(canary, key, expected, label="v16r4 decoded-canary contract")
    required_metrics = canary.get("automatic_metrics_required")
    if required_metrics != [
        "decode_complete_81_frames_25fps",
        "high_frequency_collapse_ratio_vs_frozen_base",
        "source_structure_similarity_vs_frozen_base",
        "temporal_flicker_ratio_vs_frozen_base",
    ]:
        _fail("v16r4 decoded-canary metric contract differs")
    for key, expected in {
        "decoded_canary_manifest_sha256": HELDOUT8_MANIFEST_SHA256,
        "decoded_canary_trigger_steps": [1, 8, 32, 128, 359, 644],
        "current_checkpoint_requires_decoded_canary": True,
        "current_checkpoint_promotion_requires_decoded_canary_sidecar": True,
    }.items():
        _require(contract, key, expected, label="v16r4 training contract")


def _validate_zero_rms(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    summary = _mapping(
        receipt.get("v16r3_zero_rms_backward_summary"),
        label="v16r4 inherited zero-RMS summary",
    )
    for key, expected in {
        "policy": shared.ZERO_RMS_POLICY,
        "scope": list(shared.ZERO_RMS_SCOPE),
        "finite_nonnegative_forward_values_bit_exact": True,
        "zero_forward_value": 0.0,
        "zero_backward_subgradient": 0.0,
        "positive_backward_matches_standard_sqrt": True,
        "negative_or_nonfinite_values_masked": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "policy_fixed_from_step_one": True,
    }.items():
        _require(summary, key, expected, label="v16r4 zero-RMS summary")
    s279 = _mapping(summary.get("s279_endpoint_canary"), label="S279 canary")
    for key, expected in {
        "step": shared.S279_STEP,
        "target_iid": shared.S279_TARGET_IID,
        "target_family": shared.S279_TARGET_FAMILY,
        "expected_calls": list(shared.S279_EXPECTED_CALLS),
        "observed_calls": [],
        "covered_by_checkpoint": False,
    }.items():
        _require(s279, key, expected, label="S279 canary")
    for key, expected in {
        "qk_only_zero_rms_backward_policy": shared.ZERO_RMS_POLICY,
        "s279_endpoint_canary_covered": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
    }.items():
        _require(contract, key, expected, label="v16r4 training contract")


def validate_v16r4_s1_checkpoint_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_global_step: int,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    expected_training_method_source_revision: str,
    expected_training_method_source_archive_sha256: str,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    step = require_save_step(expected_global_step)
    config_sha = require_sha256(
        expected_adapter_config_sha256, label="expected adapter config SHA-256"
    )
    model_sha = require_sha256(
        expected_adapter_model_sha256, label="expected adapter model SHA-256"
    )
    receipt_sha = require_sha256(
        expected_training_receipt_sha256, label="expected training receipt SHA-256"
    )
    source_revision = require_sha256(
        expected_training_method_source_revision,
        label="expected training method source revision",
    )
    source_archive_sha = require_sha256(
        expected_training_method_source_archive_sha256,
        label="expected training method source archive SHA-256",
    )
    if expected_checkpoint_tree_sha256 != delegated.trainer.CHECKPOINT_TREE_SHA256:
        _fail("v16r4 S1 supports only the audited Bernini checkpoint tree")
    if not isinstance(training_receipt, Mapping):
        _fail("training receipt must be a JSON object")
    try:
        shared.validate_peft_config(adapter_config)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error

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
        "method_source_revision": source_revision,
        "method_source_archive_sha256": source_archive_sha,
    }.items():
        _require(training_receipt, key, expected, label="v16r4 receipt")

    contract = _mapping(
        training_receipt.get("training_contract"), label="training contract"
    )
    for key, expected in {
        "method": TRAINING_METHOD,
        "profile": "dynamic_static",
        "training_objective": TRAINING_OBJECTIVE,
        "route_operator": ROUTE_OPERATOR,
        "route_transport": REQUIRED_DECODE_TRANSPORT,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_scope": LORA_SCOPE,
        "lora_target_module_count": TARGET_MODULE_COUNT,
        "lora_target_modules_sha256": TARGET_MODULES_SHA256,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "full_attention_lora_enabled": True,
        "full644_manifest_sha256": FULL644_MANIFEST_SHA256,
        "full644_manifest_digest": FULL644_MANIFEST_DIGEST,
        "full644_manifest_row_count": MAX_STEPS,
        "full644_manifest_family_count": 28,
        "full644_optimizer_schedule": "exact644_unique_rows_once",
        "training_manifest_order": "family_round_robin_manifest_iid_stable_exact644_once_v16",
        "actual_distinct_target_iid_count": 1,
        "actual_distinct_target_iids": [S1_TARGET_IID],
        "actual_distinct_same_iid_role1_donor_count": 1,
        "actual_distinct_same_iid_role1_donor_iids": [S1_TARGET_IID],
        "actual_distinct_target_family_count": 1,
        "actual_distinct_target_families": [S1_TARGET_FAMILY],
        "anchor_dynamic_static_pairs_audited": 2,
        "anchor_source_and_donor_share_iid": True,
        "anchor_cross_appearance": False,
        "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
        "all_full644_rows_targeted_exactly_once": False,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "strict_selection_flag_filters_optimizer_rows": False,
        "broad_and_strict_rows_are_both_optimizer_admitted": True,
        "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        "source_preservation_claimed": False,
        "scientific_claim_authorized": False,
    }.items():
        _require(contract, key, expected, label="v16r4 training contract")
    for forbidden in (
        "actual_action_descent_fallback_uses_primary_action_only",
        "near_collinear_fallback_drops_auxiliary_replay_for_that_update",
    ):
        if forbidden in contract:
            _fail(f"v16r4 contract inherited forbidden fallback field: {forbidden}")
    for forbidden in (
        "v16r2_actual_action_descent_fallback_summary",
        "v15r2_collinear_fallback_summary",
    ):
        if forbidden in training_receipt:
            _fail(f"v16r4 receipt inherited forbidden fallback section: {forbidden}")

    try:
        shared._validate_full644_prefix(training_receipt, contract, step=step)
        shared._validate_memory_and_gradient(training_receipt, step=step)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error
    full = _mapping(training_receipt.get("v16_full644_summary"), label="full644 summary")
    for key, expected in {
        "manifest_sha256": FULL644_MANIFEST_SHA256,
        "manifest_digest": FULL644_MANIFEST_DIGEST,
        "target_prefix_iids_sha256": S1_PREFIX_IIDS_SHA256,
        "actual_strict_target_count": 0,
        "actual_broad_target_count": 1,
        "observed_latent_geometry_count": 1,
        "lazy_pair_cache_max_rows": 1,
    }.items():
        _require(full, key, expected, label="v16r4 full644 summary")

    _validate_optimizer_geometry(training_receipt, contract)
    _validate_canary_contract(training_receipt, contract)
    _validate_zero_rms(training_receipt, contract)

    return {
        "global_step": step,
        "max_steps": MAX_STEPS,
        "checkpoint_complete": True,
        "terminal_full644_checkpoint": False,
        "exact644_training_complete": False,
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
        "method_source_revision": source_revision,
        "method_source_archive_sha256": source_archive_sha,
        "decoded_canary_manifest_sha256": HELDOUT8_MANIFEST_SHA256,
        "full_heldout8_dual_arm_promotion_complete": False,
        "v16r4_online_anchor": True,
    }


def resolve_checkpoint(value: str | Path, *, expected_global_step: int) -> CheckpointBundle:
    require_save_step(expected_global_step)
    try:
        return shared.resolve_checkpoint(value, expected_global_step=GLOBAL_STEP)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def verify_checkpoint_hashes(bundle: CheckpointBundle, **kwargs: Any) -> None:
    try:
        shared.verify_checkpoint_hashes(bundle, **kwargs)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def validate_adapter_safetensors_inventory(path: Path) -> dict[str, Any]:
    try:
        return shared.validate_adapter_safetensors_inventory(path)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def authenticate_checkpoint(
    value: str | Path,
    *,
    expected_global_step: int,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    expected_training_method_source_revision: str,
    expected_training_method_source_archive_sha256: str,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
    include_model: bool = True,
) -> dict[str, Any]:
    bundle = resolve_checkpoint(value, expected_global_step=expected_global_step)
    verify_checkpoint_hashes(
        bundle,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        include_model=include_model,
    )
    try:
        adapter_config = shared._read_json(bundle.adapter_config_path, label="adapter config")
        receipt = shared._read_json(bundle.training_receipt_path, label="training receipt")
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error
    identity = validate_v16r4_s1_checkpoint_contract(
        adapter_config,
        receipt,
        expected_global_step=expected_global_step,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        expected_training_method_source_revision=expected_training_method_source_revision,
        expected_training_method_source_archive_sha256=(
            expected_training_method_source_archive_sha256
        ),
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
        "global_step": GLOBAL_STEP,
        "max_steps": MAX_STEPS,
        "training_method_source_revision": identity["method_source_revision"],
        "training_method_source_archive_sha256": identity[
            "method_source_archive_sha256"
        ],
        "decoded_canary_manifest_sha256": HELDOUT8_MANIFEST_SHA256,
    }
    return {
        **identity,
        "bundle": bundle,
        "adapter_config": adapter_config,
        "training_receipt": receipt,
        "adapter_tensor_inventory": inventory,
        "binding": binding,
        "binding_sha256": shared.object_sha256(binding),
    }


def assert_same_bundle(observed: Any, expected: CheckpointBundle) -> None:
    try:
        shared.assert_same_bundle(observed, expected)
    except delegated.InferenceContractError as error:
        raise V16R4S1CheckpointContractError(str(error)) from error


def _authentication_report(identity: Mapping[str, Any]) -> dict[str, Any]:
    inventory = identity.get("adapter_tensor_inventory")
    return {
        "schema_version": AUTHENTICATION_SCHEMA,
        "authenticated": True,
        "global_step": GLOBAL_STEP,
        "max_steps": MAX_STEPS,
        "terminal_full644_checkpoint": False,
        "exact644_training_complete": False,
        "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
        "training_method": TRAINING_METHOD,
        "adapter_config_sha256": identity["adapter_config_sha256"],
        "adapter_model_sha256": identity["adapter_model_sha256"],
        "training_receipt_sha256": identity["training_receipt_sha256"],
        "training_method_source_revision": identity["method_source_revision"],
        "training_method_source_archive_sha256": identity[
            "method_source_archive_sha256"
        ],
        "decoded_canary_manifest_sha256": HELDOUT8_MANIFEST_SHA256,
        "adapter_tensor_inventory": inventory,
        "binding": identity["binding"],
        "binding_sha256": identity["binding_sha256"],
        "full_heldout8_dual_arm_promotion_complete": False,
        "scientific_claim_authorized": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--expected-adapter-model-sha256", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--expected-training-method-source-revision", required=True)
    parser.add_argument(
        "--expected-training-method-source-archive-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-model-hash", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    identity = authenticate_checkpoint(
        args.checkpoint,
        expected_global_step=GLOBAL_STEP,
        expected_adapter_config_sha256=args.expected_adapter_config_sha256,
        expected_adapter_model_sha256=args.expected_adapter_model_sha256,
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        expected_training_method_source_revision=(
            args.expected_training_method_source_revision
        ),
        expected_training_method_source_archive_sha256=(
            args.expected_training_method_source_archive_sha256
        ),
        include_model=not args.skip_model_hash,
    )
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        _fail("authentication output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _authentication_report(identity),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    return 0


__all__ = [
    "ADAPTER_TENSOR_COUNT",
    "GLOBAL_STEP",
    "HELDOUT8_MANIFEST_SHA256",
    "LORA_ALPHA",
    "LORA_RANK",
    "LORA_SCOPE",
    "MAX_STEPS",
    "SAVE_STEPS",
    "TARGET_MODULE_COUNT",
    "TARGET_MODULES_SHA256",
    "TRAINING_METHOD",
    "TRAINING_RECEIPT_SCHEMA",
    "V16R4S1CheckpointContractError",
    "assert_same_bundle",
    "authenticate_checkpoint",
    "expected_target_modules",
    "require_save_step",
    "require_sha256",
    "validate_adapter_safetensors_inventory",
    "validate_runtime_versions",
    "validate_v16r4_s1_checkpoint_contract",
    "verify_checkpoint_hashes",
]


if __name__ == "__main__":
    raise SystemExit(main())
