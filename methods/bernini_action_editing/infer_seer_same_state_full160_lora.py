#!/usr/bin/env python3
"""Strict source-only inference for the completed 160-step SEER adapter.

This entry point deliberately does not widen ``infer_seer_same_state_lora``.
It reuses that helper's same-state FM+motion+copy receipt validation and exact
60-module/120-tensor PEFT loader, then adds the completion contract that only
the final ``checkpoint-00000160`` bundle with ``global_step=max_steps=160`` is
admissible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_seer_same_state_lora as same_state  # noqa: E402
import infer_seer_scoped_lora as scoped  # noqa: E402


base = same_state.base
trainer = same_state.trainer
REQUIRED_GLOBAL_STEP = 160
REQUIRED_MAX_STEPS = 160
REQUIRED_CHECKPOINT_DIRECTORY = "checkpoint-00000160"
EXPECTED_TRAINABLE_PARAMETER_COUNT = 1_474_560
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SeerSameStateFull160InferenceError(
    same_state.SeerSameStateInferenceError
):
    """Raised before generation when the completed SEER contract differs."""


def _expected_optimizer_parameter_names() -> set[str]:
    return {
        f"base_model.model.{module}.lora_{factor}.default.weight"
        for module in scoped.expected_lora_target_modules()
        for factor in ("A", "B")
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SeerSameStateFull160InferenceError(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _validate_full160_completion_receipt(
    receipt: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    """Add final-step, exact-objective, and internal provenance cross-binds."""

    if type(receipt.get("max_steps")) is not int or receipt["max_steps"] != REQUIRED_MAX_STEPS:
        raise SeerSameStateFull160InferenceError(
            "SEER full160 requires max_steps=160"
        )

    immutable = receipt.get("immutable_contract")
    value = immutable.get("value") if isinstance(immutable, Mapping) else None
    checkpoint = receipt.get("checkpoint")
    dataset = receipt.get("dataset")
    supervision = receipt.get("supervision")
    adapter = receipt.get("adapter")
    optimizer = receipt.get("optimizer")
    seer = receipt.get("seer")
    if not all(
        isinstance(section, Mapping)
        for section in (
            immutable,
            value,
            checkpoint,
            dataset,
            supervision,
            adapter,
            optimizer,
            seer,
        )
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 receipt section closure differs"
        )

    # These are the exact non-optional optimization choices used by the SEER
    # same-state method.  They are checked semantically even though ``value``
    # is already digest-bound, so re-signing a drifted fixture cannot admit a
    # different objective under the same method name.
    immutable_objective = {
        "learning_rate": 1.0e-6,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260813,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_scope": "cross_q_out",
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
        "boundary_gauge": (
            "zero_first_latent_phase_of_raw_predicted_clean_delta"
        ),
        "bridge_fractions": None,
        "bridge_consistency_weight": 0.0,
        "causal_ema_decay": 0.5,
        "charbonnier_scale": 0.1,
        "inference_sigma_schedule_sha256": None,
        "inference_sigma_selector": None,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "noop_instruction_sha256": hashlib.sha256(
            trainer.motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "full_pair_flow_matching_weight": 1.0,
        "same_state_causal_motion_weight": 0.5,
        "same_state_noop_copy_weight": 0.5,
    }
    immutable_dynamic_fields = {
        "method",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_commit",
        "veomni_commit",
        "checkpoint_tree_sha256",
        "checkpoint_path",
        "dataset_signature",
        "dataset_summary_sha256",
        "dataset_index_sha256",
        "routing_digest",
        "routing_file_sha256",
        "expected_routing_jsonl_sha256",
        "eligible_route_stream_count",
        "eligible_route_stream_sha256",
        "target_modules",
        "expected_seer_owner_spec_sha256",
        "expected_seer_manifest_sha256",
        "seer_row_count",
        "seer_iids_sha256",
        "seer_authority",
        "same_generated_video_coordinate",
        "event_erasure_source_excludes_transition_and_terminal",
        "rejected_cmsg_cross_identity_gate_reused",
        "training_completion_is_method_success",
        "heldout_decoded_review_required",
    }
    if (
        set(immutable) != {
            "value",
            "digest",
            "expected_seer_manifest_sha256",
            "expected_seer_owner_spec_sha256",
            "method_source_archive_sha256",
        }
        or set(value) != set(immutable_objective) | immutable_dynamic_fields
        or any(
            value.get(key) != expected
            for key, expected in immutable_objective.items()
        )
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 immutable objective differs"
        )

    supervision_objective = {
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
        "boundary_gauge_field": (
            "raw_predicted_action_minus_noop_clean_field"
        ),
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
        "causal_boundary_quotient_enabled": True,
        "causal_boundary_projection_enabled": True,
        "temporal_quotient_enabled": False,
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
    if set(supervision) != set(supervision_objective) or any(
        supervision.get(key) != expected
        for key, expected in supervision_objective.items()
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 supervision objective differs"
        )

    # Cross-bind the immutable source/dataset claims to the duplicated receipt
    # evidence consumed by inference.  The outer receipt digest alone would
    # not give these duplicated fields semantic meaning.
    summary = dataset.get("summary")
    routing = dataset.get("routing")
    if (
        value.get("method_source_revision")
        != identity.get("method_source_revision")
        or value.get("method_source_archive_sha256")
        != identity.get("method_source_archive_sha256")
        or immutable.get("method_source_archive_sha256")
        != identity.get("method_source_archive_sha256")
        or value.get("bernini_commit") != receipt.get("bernini_commit")
        or value.get("veomni_commit") != receipt.get("veomni_commit")
        or value.get("checkpoint_path") != checkpoint.get("path")
        or value.get("checkpoint_tree_sha256") != checkpoint.get("tree_sha256")
        or not isinstance(summary, Mapping)
        or not isinstance(routing, Mapping)
        or dataset.get("rows") != value.get("seer_row_count")
        or dataset.get("rows") != seer.get("row_count")
        or value.get("eligible_route_stream_count") != dataset.get("rows")
        or dataset.get("signature") != value.get("dataset_signature")
        or summary.get("sha256") != value.get("dataset_summary_sha256")
        or summary.get("index_sha256") != value.get("dataset_index_sha256")
        or routing.get("routing_digest") != value.get("routing_digest")
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or routing.get("file_sha256")
        != value.get("expected_routing_jsonl_sha256")
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": value.get("seer_row_count"), "motion_only": 0, "reject": 0}
        or summary.get("complete") is not True
        or summary.get("allow_incomplete") is not False
        or summary.get("expected_rows") != dataset.get("rows")
        or summary.get("materialized_rows") != dataset.get("rows")
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 provenance cross-bind differs"
        )
    _require_sha256(value.get("seer_iids_sha256"), label="SEER IID-set digest")
    _require_sha256(
        value.get("eligible_route_stream_sha256"),
        label="SEER eligible-route stream digest",
    )

    names = optimizer.get("parameter_names")
    expected_names = _expected_optimizer_parameter_names()
    if (
        not isinstance(names, list)
        or set(adapter) != {
            "rank",
            "alpha",
            "scope",
            "target_module_count",
            "target_modules",
            "target_modules_sha256",
            "trainable_parameter_count",
            "parameter_names_sha256",
            "initialization_digest",
            "checkpoint_parameter_digest",
        }
        or set(optimizer) != {
            "type",
            "learning_rate",
            "weight_decay",
            "max_gradient_norm",
            "parameter_names",
            "checkpoint_state_digest",
        }
        or len(names) != scoped.ADAPTER_TENSOR_COUNT
        or len(set(names)) != scoped.ADAPTER_TENSOR_COUNT
        or set(names) != expected_names
        or adapter.get("parameter_names_sha256") != base.object_sha256(names)
        or adapter.get("trainable_parameter_count")
        != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or optimizer.get("type") != "AdamW"
        or optimizer.get("learning_rate") != 1.0e-6
        or optimizer.get("weight_decay") != 0.0
        or optimizer.get("max_gradient_norm") != 1.0
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 exact 60-module/120-tensor optimizer scope differs"
        )
    _require_sha256(
        optimizer.get("checkpoint_state_digest"),
        label="SEER optimizer-state digest",
    )


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = base.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Admit only a completed, exact-objective 160-step same-state update."""

    try:
        identity = same_state._validate_adapter_contract_at_step(
            adapter_config,
            receipt,
            required_global_step=REQUIRED_GLOBAL_STEP,
            step_error=(
                "SEER full160 held-out decode requires global_step=160"
            ),
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
    except same_state.SeerSameStateInferenceError as error:
        raise SeerSameStateFull160InferenceError(str(error)) from error
    _validate_full160_completion_receipt(receipt, identity)
    return dict(identity)


def validate_checkpoint_save_directory(checkpoint_root: Path) -> Path:
    """Bind the admitted receipt to the trainer's exact final save directory."""

    if not checkpoint_root.is_absolute() or checkpoint_root.name != REQUIRED_CHECKPOINT_DIRECTORY:
        raise SeerSameStateFull160InferenceError(
            "SEER full160 adapter must be checkpoint-00000160"
        )
    return checkpoint_root


def validate_runtime_provenance(
    identity: Mapping[str, Any],
    *,
    method_source_revision: str,
    method_source_archive_sha256: str,
) -> None:
    """Cross-bind the frozen inference archive arguments to training bytes."""

    if (
        identity.get("method_source_revision") != method_source_revision
        or identity.get("method_source_archive_sha256")
        != method_source_archive_sha256
    ):
        raise SeerSameStateFull160InferenceError(
            "SEER full160 training/inference method provenance differs"
        )


def _install_specialization() -> None:
    same_state._install_specialization()
    base.validate_adapter_contract = validate_adapter_contract


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_specialization()
    preliminary = base.build_parser().parse_args(argv)
    if preliminary.adapter_checkpoint:
        bundle = base.resolve_adapter_bundle(preliminary.adapter_checkpoint)
        validate_checkpoint_save_directory(bundle.checkpoint_root)
        scoped.validate_scoped_safetensors(bundle.adapter_model_path)
        adapter_config = base._read_json(bundle.adapter_config_path, label="adapter config")
        receipt = base._read_json(bundle.training_receipt_path, label="training receipt")
        identity = validate_adapter_contract(
            adapter_config,
            receipt,
            expected_checkpoint_tree_sha256=(
                preliminary.expected_checkpoint_tree_sha256
            ),
        )
        validate_runtime_provenance(
            identity,
            method_source_revision=preliminary.method_source_revision,
            method_source_archive_sha256=(
                preliminary.method_source_archive_sha256
            ),
        )
    return base.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.InferenceContractError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
