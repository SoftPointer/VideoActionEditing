#!/usr/bin/env python3
"""Strict five-branch inference integration for Bernini RMC v7 / RS-FQT v8.

The deployed operator evaluates exactly five source-conditioned editor
branches at every one of the pinned forty UniPC states: frozen negative,
frozen no-op, frozen action, adapted no-op, and adapted action.  The paired
target and target-only generator used during training are absent.

V7 applies a bounded difference-of-differences around the frozen action field.
V8 replaces that coefficient-one appearance carrier by the frozen semantic
no-op reconstruction section and transports the complete adapted action/no-op
quotient through the same temporal geometry.  Its trained-checkpoint diagnostic
may scale both learned-prior radius terms by the audited 2.5x or 4x factors;
the absolute floor stays fixed and every such run is inference-only.  At a zero
release coefficient v7 aliases the official action tensor and v8 aliases the
frozen no-op clean tensor before the required clean-to-velocity conversion.

This module intentionally contains no end-to-end video CLI.  It exposes the
strict adapter loader, projector, reversible hook, and trace validator used by
the separate AUH runner.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import inference_sigma_strata as sigma_strata  # noqa: E402
import gauge_anchored_commutator as gauge  # noqa: E402
import infer_prior_tangent_lora as v5  # noqa: E402
import motion_commutator as commutator  # noqa: E402
import relational_commutator_objective as objective  # noqa: E402
import train_cross_mode_cmsg_lora as v6_scope  # noqa: E402
import train_relational_motion_commutator_auh as v7_train  # noqa: E402


tri = v5.tri
trainer = v5.trainer

METHOD_NAME = v7_train.METHOD_NAME
TRAINING_RECEIPT_SCHEMA = v7_train.RECEIPT_SCHEMA
INFERENCE_RECEIPT_SCHEMA = v7_train.INFERENCE_RECEIPT_SCHEMA
NUM_FRAMES = v7_train.NUM_FRAMES
LATENT_PHASES = v7_train.LATENT_PHASES
NUM_DENOISING_STEPS = commutator.NUM_DENOISING_STEPS
REQUIRED_LORA_SCOPE = v6_scope.LORA_SCOPE
REQUIRED_TARGET_MODULE_COUNT = v6_scope.EXPECTED_LORA_MODULES
REQUIRED_LORA_RANK = v6_scope.LORA_RANK
REQUIRED_LORA_ALPHA = v6_scope.LORA_ALPHA
ADAPTER_SCALE = 1.0
MAIN_COMMUTATOR_CONFIG = v7_train.MAIN_COMMUTATOR_CONFIG
LATE_EXACT_STEPS = tuple(
    step
    for step in range(NUM_DENOISING_STEPS)
    if commutator.release_rho(step) == 0.0
)
V7_RESIDUAL_ACTION_SECTION = "v7_residual_action_section"
V8_RECONSTRUCTION_SECTION_FQT = (
    "v8_reconstruction_section_feasible_quotient_transport"
)
V8_METHOD_NAME = "bernini-reconstruction-section-feasible-quotient-transport-v8"
V8_INFERENCE_RECEIPT_SCHEMA = "bernini-rs-fqt-inference-receipt-v8"
V8_TRAINING_RECEIPT_SCHEMA = "bernini-rs-fqt-auh-training-receipt-v8"
V8_TRAINING_METHOD_NAME = (
    "bernini-reconstruction-section-feasible-quotient-lora-v8-auh"
)
OPERATOR_MODES = (
    V7_RESIDUAL_ACTION_SECTION,
    V8_RECONSTRUCTION_SECTION_FQT,
)
MAIN_FEASIBLE_QUOTIENT_CONFIG = gauge.FeasibleQuotientConfig()
MAIN_V8_RADIUS_SCALE = 1.0
V8_RADIUS_SCALE_CHOICES = (1.0, 2.5, 4.0)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RelationalMotionCommutatorInferenceError(RuntimeError):
    """Raised before an unaudited adapter or scheduler value can be used."""


def feasible_quotient_config_for_radius_scale(
    radius_scale: float,
) -> gauge.FeasibleQuotientConfig:
    """Return one of the three audited V8 inference-radius configurations.

    The ablation scales both learned-prior radius terms together.  The absolute
    floor and numerical epsilon remain training-matched, so this is a radius
    diagnostic rather than a second projection implementation.
    """

    if (
        isinstance(radius_scale, bool)
        or not isinstance(radius_scale, (int, float))
        or not math.isfinite(float(radius_scale))
        or float(radius_scale) not in V8_RADIUS_SCALE_CHOICES
    ):
        raise RelationalMotionCommutatorInferenceError(
            "V8 radius scale must be one of the audited 1.0/2.5/4.0 arms"
        )
    scale = float(radius_scale)
    return gauge.FeasibleQuotientConfig(
        frozen_quotient_radius_ratio=(
            scale
            * float(
                MAIN_FEASIBLE_QUOTIENT_CONFIG.frozen_quotient_radius_ratio
            )
        ),
        noop_dynamics_radius_ratio=(
            scale
            * float(MAIN_FEASIBLE_QUOTIENT_CONFIG.noop_dynamics_radius_ratio)
        ),
        radius_floor=float(MAIN_FEASIBLE_QUOTIENT_CONFIG.radius_floor),
        epsilon=float(MAIN_FEASIBLE_QUOTIENT_CONFIG.epsilon),
    )


def validated_feasible_quotient_radius_scale(
    config: gauge.FeasibleQuotientConfig,
    *,
    operator_mode: str,
) -> float:
    """Validate a core-hook radius config and recover its audited scale."""

    if operator_mode not in OPERATOR_MODES:
        raise RelationalMotionCommutatorInferenceError(
            "unknown reconstruction-section operator mode"
        )
    if not isinstance(config, gauge.FeasibleQuotientConfig):
        raise RelationalMotionCommutatorInferenceError(
            "feasible quotient config has the wrong contract type"
        )
    try:
        config.validate()
    except gauge.GaugeAnchoredCommutatorError as error:
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    if operator_mode == V7_RESIDUAL_ACTION_SECTION:
        if config != MAIN_FEASIBLE_QUOTIENT_CONFIG:
            raise RelationalMotionCommutatorInferenceError(
                "V7 must keep the unit V8 radius placeholder"
            )
        return MAIN_V8_RADIUS_SCALE
    for radius_scale in V8_RADIUS_SCALE_CHOICES:
        if config == feasible_quotient_config_for_radius_scale(radius_scale):
            return float(radius_scale)
    raise RelationalMotionCommutatorInferenceError(
        "V8 feasible quotient config is outside the audited radius-scale set"
    )


def expected_lora_targets() -> list[str]:
    """Return the canonical exact-46 v7 LoRA target list."""

    try:
        targets = v6_scope.select_cmsg_lora_targets(
            v6_scope.canonical_attention_modules()
        )
    except v6_scope.CrossModeCMSGTrainingError as error:
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    if targets != sorted(set(targets)) or len(targets) != 46:
        raise RelationalMotionCommutatorInferenceError(
            "canonical v7 LoRA scope is not exact-46"
        )
    return targets


def expected_serialized_target_patterns() -> list[str]:
    """Return PEFT 0.19.1's audited compact serialization of exact-46.

    The thirty cross-attention targets collapse to one common suffix, while
    each selected self-attention block needs its own block-qualified suffix.
    The training receipt remains authoritative for the 46 fully-qualified
    modules; these patterns are validated by expansion against the complete
    canonical attention-module universe below.
    """

    return sorted(
        ["attn2.to_q"]
        + [f"{block}.attn1.to_q" for block in range(7, 23)]
    )


def _expand_serialized_target_patterns(patterns: Sequence[str]) -> list[str]:
    universe = v6_scope.canonical_attention_modules()
    expanded = {
        module
        for module in universe
        if any(
            module == pattern or module.endswith(f".{pattern}")
            for pattern in patterns
        )
    }
    return sorted(expanded)


def _validate_inference_commutator_config(
    config: commutator.MotionCommutatorConfig,
) -> None:
    if not isinstance(config, commutator.MotionCommutatorConfig):
        raise RelationalMotionCommutatorInferenceError(
            "commutator_config has the wrong contract type"
        )
    try:
        config.validate()
    except commutator.MotionCommutatorError as error:
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    if float(config.max_correction_increment_ratio) not in (0.25, 0.5, 1.0):
        raise RelationalMotionCommutatorInferenceError(
            "inference kappa must be one of the audited 0.25/0.5/1.0 arms"
        )
    for name in (
        "correction_increment_rms_floor",
        "temporal_smoothing",
        "epsilon",
    ):
        if getattr(config, name) != getattr(MAIN_COMMUTATOR_CONFIG, name):
            raise RelationalMotionCommutatorInferenceError(
                f"inference ablation may change only kappa, not {name}"
            )


def runtime_contract(
    commutator_config: commutator.MotionCommutatorConfig = (
        MAIN_COMMUTATOR_CONFIG
    ),
    *,
    operator_mode: str = V7_RESIDUAL_ACTION_SECTION,
    feasible_quotient_config: gauge.FeasibleQuotientConfig = (
        MAIN_FEASIBLE_QUOTIENT_CONFIG
    ),
    v8_training_matched: bool = False,
) -> dict[str, Any]:
    """Return the immutable source-only five-branch deployment contract."""

    _validate_inference_commutator_config(commutator_config)
    if operator_mode not in OPERATOR_MODES:
        raise RelationalMotionCommutatorInferenceError(
            "unknown reconstruction-section operator mode"
        )
    radius_scale = validated_feasible_quotient_radius_scale(
        feasible_quotient_config,
        operator_mode=operator_mode,
    )
    release = list(commutator.release_rho_schedule())
    operator_ablation = commutator_config != MAIN_COMMUTATOR_CONFIG
    is_v8 = operator_mode == V8_RECONSTRUCTION_SECTION_FQT
    if type(v8_training_matched) is not bool or (
        v8_training_matched and not is_v8
    ):
        raise RelationalMotionCommutatorInferenceError(
            "v8 training-match status is invalid for the selected operator"
        )
    radius_scale_ablation = bool(
        is_v8 and radius_scale != MAIN_V8_RADIUS_SCALE
    )
    if radius_scale_ablation and not v8_training_matched:
        raise RelationalMotionCommutatorInferenceError(
            "non-unit V8 radius scales require a trained V8 checkpoint"
        )
    operator_training_matched = bool(
        (v8_training_matched and not radius_scale_ablation) if is_v8 else True
    )
    runtime_training_matched = bool(
        not operator_ablation and operator_training_matched
    )
    contract = {
        "method": V8_METHOD_NAME if is_v8 else METHOD_NAME,
        "training_receipt_schema": (
            V8_TRAINING_RECEIPT_SCHEMA
            if v8_training_matched
            else TRAINING_RECEIPT_SCHEMA
        ),
        "inference_receipt_schema": (
            V8_INFERENCE_RECEIPT_SCHEMA if is_v8 else INFERENCE_RECEIPT_SCHEMA
        ),
        "external_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": (
            ["paired_target_video"]
            if v8_training_matched
            else ["paired_target_video", "t2v_generator"]
        ),
        "forbidden_inference_conditions": list(
            objective.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "per_step_editor_branches": list(v7_train.INFERENCE_FORWARD_ORDER),
        "transformer_forwards_per_step": 5,
        "frozen_branches_adapter_disabled": True,
        "adapted_branches_adapter_enabled_unmerged": True,
        "all_branch_autograd": False,
        "official_action_apg_bit_exact": True,
        "frozen_action_projection_clean_precision": (
            "local_fp32_apg_after_native_bf16_official_parity"
            if is_v8
            else "official_native_bf16_velocity_roundtrip"
        ),
        "scheduler_model_output_precision": (
            "fp32_exact_clean_transport_with_post_boundary_radius_certificate"
            if is_v8
            else "official_native_bf16"
        ),
        "apg_momentum": 0.0,
        "operator_mode": operator_mode,
        "training_correction": (
            (
                "projection_consistent_complete_action_noop_quotient"
                if v8_training_matched
                else "v7_raw_Ctheta_adapter_reused_for_operator_falsification"
            )
            if is_v8 else "raw_Ctheta"
        ),
        "deployment_correction": (
            "FIR_then_centered_full_action_quotient_projection"
            if is_v8
            else "temporal_smooth_then_hard_bound_Ctheta"
        ),
        "commutator_config": asdict(commutator_config),
        "feasible_quotient_config": (
            asdict(feasible_quotient_config) if is_v8 else None
        ),
        "v8_radius_scale": radius_scale,
        "audited_v8_radius_scales": list(V8_RADIUS_SCALE_CHOICES),
        "v8_radius_scale_training_matched": not radius_scale_ablation,
        "v8_radius_scale_inference_only_ablation": radius_scale_ablation,
        "main_operator": runtime_training_matched,
        "training_matched": runtime_training_matched,
        "inference_only_ablation": not runtime_training_matched,
        "experimental_operator_ablation": operator_ablation or (
            is_v8 and (not v8_training_matched or radius_scale_ablation)
        ),
        "operator_training_matched": operator_training_matched,
        "hard_bound_formula": (
            (
                "max("
                f"{feasible_quotient_config.frozen_quotient_radius_ratio:g}"
                "*frozen_quotient_rms,"
                f"{feasible_quotient_config.noop_dynamics_radius_ratio:g}"
                "*noop_dynamics_rms,absolute_floor)"
            )
            if is_v8
            else "max(kappa*frozen_increment_rms,absolute_floor)"
        ),
        "appearance_carrier": (
            "frozen_noop_reconstruction_section"
            if is_v8
            else "frozen_action_clean_field"
        ),
        "motion_carrier": (
            "complete_adapted_action_noop_quotient"
            if is_v8
            else "frozen_action_plus_adapter_commutator"
        ),
        "release_schedule": release,
        "release_schedule_sha256": trainer.object_sha256(release),
        "zero_release_steps": list(LATE_EXACT_STEPS),
        "zero_release_semantics": (
            "adapter_evaluated_then_noop_clean_section_selected_and_converted_to_velocity"
            if is_v8
            else "adapter_evaluated_but_exact_official_model_output_object_replayed"
        ),
        "original_unipc_calls_per_step": 1,
        "generator_forwards_per_step": 0,
        "custom_integrator": False,
        "first_frame_anchor": False,
    }
    return contract


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalMotionCommutatorInferenceError(f"receipt lacks {label}")
    return value


def _validate_sha(
    value: Any, *, label: str, pattern: re.Pattern[str]
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RelationalMotionCommutatorInferenceError(f"{label} is invalid")
    return value


def _target_only_loss_config() -> objective.RelationalCommutatorLossConfig:
    return objective.RelationalCommutatorLossConfig(
        relational_auxiliary_weight=0.0,
        commutator_config=MAIN_COMMUTATOR_CONFIG,
    )


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Accept only a completed target-only v7 exact-40/exact-46 checkpoint.

    One-step canaries, v6 receipts, relational-auxiliary checkpoints, partial
    sigma cycles, and receipts still declaring loader parity pending all fail
    closed.
    """

    if not isinstance(adapter_config, Mapping) or not isinstance(receipt, Mapping):
        raise RelationalMotionCommutatorInferenceError(
            "adapter config and receipt must be mappings"
        )
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _validate_sha(digest, label="training receipt digest", pattern=_SHA256_RE)
    if trainer.object_sha256(candidate) != digest:
        raise RelationalMotionCommutatorInferenceError(
            "training receipt digest differs"
        )
    if receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA:
        raise RelationalMotionCommutatorInferenceError(
            "training receipt is not the AUH v7 schema"
        )
    if receipt.get("method") != METHOD_NAME:
        raise RelationalMotionCommutatorInferenceError(
            "training method is not relational commutator v7"
        )
    if (
        receipt.get("global_step") != NUM_DENOISING_STEPS
        or receipt.get("max_steps") != NUM_DENOISING_STEPS
        or receipt.get("formal_40_sigma_cycle_complete") is not True
    ):
        raise RelationalMotionCommutatorInferenceError(
            "checkpoint is not one completed exact 40-sigma v7 cycle"
        )
    expected_indices = list(range(NUM_DENOISING_STEPS))
    if receipt.get("accepted_sigma_schedule_indices") != expected_indices:
        raise RelationalMotionCommutatorInferenceError(
            "accepted sigma schedule is not exactly 0..39"
        )
    step_audit = receipt.get("step_audit")
    if (
        not isinstance(step_audit, list)
        or len(step_audit) != NUM_DENOISING_STEPS
        or receipt.get("step_audit_sha256") != trainer.object_sha256(step_audit)
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training step audit is incomplete or altered"
        )
    for index, record in enumerate(step_audit):
        if (
            not isinstance(record, Mapping)
            or record.get("optimizer_step") != index + 1
            or record.get("sigma_schedule_index") != index
            or record.get("teacher_mode") != "target_only"
            or record.get("metrics_timing") != v7_train.METRICS_TIMING
            or not math.isclose(
                float(record.get("rho", math.nan)),
                commutator.release_rho(index),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RelationalMotionCommutatorInferenceError(
                f"training step audit differs at index {index}"
            )

    immutable = _require_mapping(
        receipt.get("immutable_contract"), label="immutable contract"
    )
    value = _require_mapping(immutable.get("value"), label="immutable value")
    if immutable.get("digest") != trainer.object_sha256(value):
        raise RelationalMotionCommutatorInferenceError(
            "immutable training contract digest differs"
        )
    target_config = _target_only_loss_config()
    release = list(commutator.release_rho_schedule())
    fixed_expected = {
        "method": METHOD_NAME,
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "learning_rate": v7_train.LEARNING_RATE,
        "teacher_mode": "target_only",
        "loss_config": asdict(target_config),
        "objective_contract": objective.immutable_objective_contract(
            target_config
        ),
        "training_bridge_endpoint": "source(beta=0)",
        "target_endpoint_teacher_leakage_forbidden": True,
        "forward_cell_order": list(v7_train.FORWARD_CELL_ORDER),
        "forwards_per_candidate": 7,
        "graph_forwards_per_candidate": 2,
        "graph_branch_order": list(objective.GRAPH_BRANCHES),
        "inference_forward_order": list(v7_train.INFERENCE_FORWARD_ORDER),
        "inference_forwards_per_step": 5,
        "target_motion_teacher": "Q0(target_clean-source_clean)",
        "target_used_as_model_condition": False,
        "t2v_relation_pointwise_coordinate_loss": False,
        "training_correction": "raw_Ctheta",
        "deployment_correction": "temporal_smooth_then_hard_bound_Ctheta",
        "hard_bound_formula": "max(kappa*frozen_increment_rms,absolute_floor)",
        "release_schedule": release,
        "release_schedule_sha256": trainer.object_sha256(release),
        "zero_release_steps": list(LATE_EXACT_STEPS),
        "zero_release_semantics": (
            "adapter_evaluated_but_official_model_output_object_replayed"
        ),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["paired_target_video", "t2v_generator"],
        "forbidden_inference_conditions": list(
            objective.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "inference_generator_forwards": 0,
        "metrics_timing": v7_train.METRICS_TIMING,
        "deployment_diagnostics": {
            "target_bound_mean_scale_active": True,
            "floor_dominated_fraction_active": True,
            "target_required_kappa_statistics": ["median", "p90", "max"],
            "near_zero_frozen_increment": (
                "exact float64 division when positive; finite threshold proxy "
                "and explicit unreachable fraction at exact zero"
            ),
        },
        "inference_loader_parity": {
            "verified": True,
            "verification_stage": (
                "immutable_launcher_preflight_before_model_load"
            ),
            "loader_module": v7_train.INFERENCE_LOADER_MODULE,
            "runner_module": v7_train.INFERENCE_RUNNER_MODULE,
            "finalizer_module": v7_train.INFERENCE_FINALIZER_MODULE,
            "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
            "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
            "contract_tests": list(v7_train.INFERENCE_PARITY_TESTS),
            "source_revision_and_archive_bound": True,
            "strict_loader_rejects_pending_canary_and_incomplete_cycle": True,
        },
        "resume_integrated": False,
    }
    for name, expected in fixed_expected.items():
        if value.get(name) != expected:
            raise RelationalMotionCommutatorInferenceError(
                f"immutable v7 field differs: {name}"
            )
    editor_guidance = _require_mapping(
        value.get("editor_guidance"), label="editor guidance"
    )
    if (
        editor_guidance.get("mode") != "official_momentum_zero_apg"
        or float(editor_guidance.get("momentum", math.nan)) != 0.0
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training editor APG contract differs"
        )
    for name, pattern in (
        ("method_source_revision", _SHA1_RE),
        ("method_source_archive_sha256", _SHA256_RE),
        ("dataset_summary_sha256", _SHA256_RE),
        ("dataset_index_sha256", _SHA256_RE),
        ("routing_digest", _SHA256_RE),
        ("routing_file_sha256", _SHA256_RE),
        ("eligible_route_stream_sha256", _SHA256_RE),
    ):
        _validate_sha(value.get(name), label=f"training {name}", pattern=pattern)
    if (
        receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT
        or receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT
        or value.get("bernini_commit") != receipt.get("bernini_commit")
        or value.get("veomni_commit") != receipt.get("veomni_commit")
        or value.get("checkpoint_tree_sha256")
        != expected_checkpoint_tree_sha256
        or value.get("eligible_route_count") != 359
        or value.get("routing_file_sha256") != v7_train.v5.STRICT_ROUTING_SHA256
        or type(value.get("seed")) is not int
        or not isinstance(value.get("dataset_signature"), str)
        or not value.get("dataset_signature")
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training source, data, or routing identity differs"
        )
    checkpoint = _require_mapping(
        receipt.get("checkpoint"), label="checkpoint identity"
    )
    if (
        checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
        or checkpoint.get("path") != value.get("checkpoint_path")
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training checkpoint identity differs"
        )
    dataset = _require_mapping(receipt.get("dataset"), label="dataset identity")
    summary = _require_mapping(dataset.get("summary"), label="dataset summary")
    routing = _require_mapping(dataset.get("routing"), label="routing identity")
    if (
        dataset.get("rows") != 644
        or dataset.get("signature") != value.get("dataset_signature")
        or summary.get("sha256") != value.get("dataset_summary_sha256")
        or summary.get("index_sha256") != value.get("dataset_index_sha256")
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or routing.get("routing_digest") != value.get("routing_digest")
    ):
        raise RelationalMotionCommutatorInferenceError(
            "strict-359 dataset receipt differs"
        )

    if (
        receipt.get("inference_conditions")
        != ["source_video", "action_instruction"]
        or receipt.get("training_only_generator_and_target") is not True
        or receipt.get("inference_generator_forwards") != 0
        or receipt.get("external_mask_track_flow_pose_trajectory") is not False
        or receipt.get("first_frame_anchor") is not False
        or receipt.get("experimental_training") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("resume_integrated") is not False
        or receipt.get("inference_loader_parity_pending") is not False
        or receipt.get("metrics_timing") != v7_train.METRICS_TIMING
        or receipt.get("inference_loader_parity")
        != value.get("inference_loader_parity")
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training publication or inference-boundary state differs"
        )

    artifact = _require_mapping(
        receipt.get("artifact_validation"), label="artifact validation"
    )
    artifact_value = dict(artifact)
    artifact_digest = artifact_value.pop("digest", None)
    _validate_sha(
        artifact_digest,
        label="artifact validation digest",
        pattern=_SHA256_RE,
    )
    if trainer.object_sha256(artifact_value) != artifact_digest:
        raise RelationalMotionCommutatorInferenceError(
            "artifact validation digest differs"
        )
    expected_artifact_fields = {
        "schema_version": v7_train.ARTIFACT_VALIDATION_SCHEMA,
        "verified": True,
        "status": "post_save_strict_reload_complete",
        "serialized_target_pattern_count": 17,
        "expanded_target_module_count": 46,
        "adapter_tensor_count": 92,
        "active_lora_module_count": 46,
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "validator_method_source_revision": value["method_source_revision"],
        "validator_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
        "bernini_commit": receipt["bernini_commit"],
        "veomni_commit": receipt["veomni_commit"],
        "checkpoint_tree_sha256": expected_checkpoint_tree_sha256,
    }
    for name, expected in expected_artifact_fields.items():
        if artifact_value.get(name) != expected:
            raise RelationalMotionCommutatorInferenceError(
                f"post-save artifact validation differs: {name}"
            )
    for name in ("adapter_config_sha256", "adapter_model_sha256"):
        _validate_sha(
            artifact_value.get(name),
            label=f"artifact {name}",
            pattern=_SHA256_RE,
        )
    pending_receipt_digest = _validate_sha(
        artifact_value.get("pending_receipt_digest"),
        label="pending receipt transition digest",
        pattern=_SHA256_RE,
    )
    reconstructed_pending = copy.deepcopy(dict(receipt))
    reconstructed_pending.pop("receipt_digest", None)
    reconstructed_pending["inference_loader_parity_pending"] = True
    reconstructed_pending["artifact_validation"] = {
        "schema_version": v7_train.ARTIFACT_VALIDATION_SCHEMA,
        "verified": False,
        "status": "pending_post_save_strict_reload",
    }
    if trainer.object_sha256(reconstructed_pending) != pending_receipt_digest:
        raise RelationalMotionCommutatorInferenceError(
            "ready receipt is not an auditable pending-to-ready transition"
        )

    targets = expected_lora_targets()
    canonical_target_hash = trainer.object_sha256(targets)
    immutable_lora = _require_mapping(value.get("lora"), label="immutable LoRA")
    adapter = _require_mapping(receipt.get("adapter"), label="adapter identity")
    expected_adapter = {
        "rank": 8,
        "alpha": 8,
        "scope": REQUIRED_LORA_SCOPE,
        "target_module_count": 46,
        "target_modules": targets,
        "target_modules_sha256": canonical_target_hash,
    }
    for name, expected in expected_adapter.items():
        if immutable_lora.get(name) != expected or adapter.get(name) != expected:
            raise RelationalMotionCommutatorInferenceError(
                f"training exact-46 adapter field differs: {name}"
            )
    for name in (
        "initialization_digest",
        "checkpoint_parameter_digest",
        "parameter_names_sha256",
    ):
        _validate_sha(
            adapter.get(name), label=f"adapter {name}", pattern=_SHA256_RE
        )
    if (
        type(adapter.get("trainable_parameter_count")) is not int
        or adapter["trainable_parameter_count"] <= 0
    ):
        raise RelationalMotionCommutatorInferenceError(
            "adapter trainable parameter count is invalid"
        )
    if artifact_value.get("checkpoint_parameter_digest") != adapter.get(
        "checkpoint_parameter_digest"
    ):
        raise RelationalMotionCommutatorInferenceError(
            "artifact parameter digest differs from training adapter"
        )
    optimizer = _require_mapping(
        receipt.get("optimizer"), label="optimizer identity"
    )
    parameter_names = optimizer.get("parameter_names")
    weight_decay = value.get("weight_decay")
    max_grad_norm = value.get("max_grad_norm")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
        or isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (int, float))
        or not math.isfinite(float(max_grad_norm))
        or float(max_grad_norm) <= 0.0
        or optimizer.get("type") != "AdamW"
        or float(optimizer.get("learning_rate", math.nan))
        != v7_train.LEARNING_RATE
        or float(optimizer.get("weight_decay", math.nan))
        != float(weight_decay)
        or float(optimizer.get("max_gradient_norm", math.nan))
        != float(max_grad_norm)
        or not isinstance(parameter_names, list)
        or not parameter_names
        or not all(isinstance(name, str) and name for name in parameter_names)
        or len(parameter_names) != len(set(parameter_names))
        or adapter.get("parameter_names_sha256")
        != trainer.object_sha256(parameter_names)
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training optimizer or parameter identity differs"
        )
    _validate_sha(
        optimizer.get("checkpoint_state_digest"),
        label="optimizer checkpoint digest",
        pattern=_SHA256_RE,
    )
    distributed = _require_mapping(
        receipt.get("distributed"), label="distributed identity"
    )
    if (
        distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
        or not isinstance(distributed.get("backend"), str)
        or not distributed.get("backend")
    ):
        raise RelationalMotionCommutatorInferenceError(
            "training four-rank Ulysses contract differs"
        )

    if adapter_config.get("peft_type") != "LORA":
        raise RelationalMotionCommutatorInferenceError("adapter is not LoRA")
    if adapter_config.get("r") != REQUIRED_LORA_RANK:
        raise RelationalMotionCommutatorInferenceError("adapter rank differs")
    try:
        alpha = float(adapter_config.get("lora_alpha", math.nan))
        dropout = float(adapter_config.get("lora_dropout", math.nan))
    except (TypeError, ValueError) as error:
        raise RelationalMotionCommutatorInferenceError(
            "adapter alpha or dropout is invalid"
        ) from error
    if alpha != float(REQUIRED_LORA_ALPHA) or dropout != 0.0:
        raise RelationalMotionCommutatorInferenceError(
            "adapter alpha or dropout differs"
        )
    if (
        adapter_config.get("bias") != "none"
        or adapter_config.get("modules_to_save") not in (None, [])
        or adapter_config.get("use_dora") not in (None, False)
        or adapter_config.get("use_rslora") not in (None, False)
    ):
        raise RelationalMotionCommutatorInferenceError(
            "adapter contains unsupported PEFT features"
        )
    serialized = adapter_config.get("target_modules")
    expected_patterns = expected_serialized_target_patterns()
    if (
        not isinstance(serialized, list)
        or len(serialized) != len(expected_patterns)
        or not all(isinstance(name, str) and name for name in serialized)
        or len(serialized) != len(set(serialized))
        or set(serialized) != set(expected_patterns)
        or _expand_serialized_target_patterns(serialized) != targets
    ):
        raise RelationalMotionCommutatorInferenceError(
            "serialized target_modules are not the audited 17-pattern exact46 expansion"
        )
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise RelationalMotionCommutatorInferenceError(
            "training Transformers version is missing"
        )
    return {
        "receipt_digest": digest,
        "global_step": NUM_DENOISING_STEPS,
        "scope": REQUIRED_LORA_SCOPE,
        "targets": targets,
        "serialized_target_modules": sorted(serialized),
        "target_modules_sha256": canonical_target_hash,
        "initialization_digest": adapter["initialization_digest"],
        "checkpoint_parameter_digest": adapter["checkpoint_parameter_digest"],
        "transformers_version": transformers_version,
        "training_method_source_revision": value["method_source_revision"],
        "training_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
        "artifact_validation_digest": artifact_digest,
        "adapter_config_sha256": artifact_value["adapter_config_sha256"],
        "adapter_model_sha256": artifact_value["adapter_model_sha256"],
    }


def strict_load_adapter(
    *,
    base_model: Any,
    bundle: Any,
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> tuple[Any, int, int, dict[str, Any]]:
    """Validate the v7 receipt before delegating to the audited PEFT loader."""

    identity = validate_training_adapter_contract(
        adapter_config,
        receipt,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )
    try:
        model, tensor_count, active_count = v5._strict_load_adapter(
            base_model=base_model,
            bundle=bundle,
            adapter_config=adapter_config,
            identity=identity,
        )
    except v5.PriorTangentInferenceError as error:
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    if tensor_count != 92 or active_count != 46:
        raise RelationalMotionCommutatorInferenceError(
            "reloaded adapter is not exact-46/92-tensor LoRA"
        )
    return model, tensor_count, active_count, identity


def packed_to_phase_grid(packed: Any, *, layout: Any) -> Any:
    shape = tuple(int(value) for value in getattr(packed, "shape", ()))
    if shape != layout.packed_shape:
        raise RelationalMotionCommutatorInferenceError(
            f"packed field shape {shape} differs from {layout.packed_shape}"
        )
    if layout.frames != LATENT_PHASES or layout.tokens % LATENT_PHASES:
        raise RelationalMotionCommutatorInferenceError(
            "v7 requires exactly 21 latent phases"
        )
    cells = layout.tokens // LATENT_PHASES
    return packed.reshape(
        layout.batch, LATENT_PHASES, cells, layout.packed_channels
    )


def phase_grid_to_packed(grid: Any, *, layout: Any) -> Any:
    cells = layout.tokens // LATENT_PHASES
    expected = (layout.batch, LATENT_PHASES, cells, layout.packed_channels)
    if tuple(int(value) for value in getattr(grid, "shape", ())) != expected:
        raise RelationalMotionCommutatorInferenceError(
            f"phase field shape differs from {expected}"
        )
    return grid.reshape(layout.packed_shape)


@dataclass(frozen=True)
class RawRelationalMotionCommutatorStep:
    step_index: int
    timestep: Any
    timestep_float: float
    sigma: Any
    sigma_float: float
    model_id: str
    sample_packed: Any
    official_model_output: Any
    frozen_negative_velocity_packed: Any
    frozen_noop_velocity_packed: Any
    frozen_action_velocity_packed: Any
    adapted_noop_velocity_packed: Any
    adapted_action_velocity_packed: Any
    apg: Any
    layout: Any
    commutator_config: commutator.MotionCommutatorConfig = (
        MAIN_COMMUTATOR_CONFIG
    )
    operator_mode: str = V7_RESIDUAL_ACTION_SECTION
    feasible_quotient_config: gauge.FeasibleQuotientConfig = (
        MAIN_FEASIBLE_QUOTIENT_CONFIG
    )


@dataclass(frozen=True)
class RelationalMotionCommutatorStepRecord:
    step_index: int
    timestep: float
    sigma: float
    rho: float
    model_id: str
    transformer_forwards: int
    frozen_negative_forwards: int
    frozen_noop_forwards: int
    frozen_action_forwards: int
    adapted_noop_forwards: int
    adapted_action_forwards: int
    original_scheduler_calls: int
    official_action_apg_exact: bool
    official_action_apg_rms_error: float
    official_action_apg_max_abs_error: float
    raw_commutator_correction_rms: float
    bounded_commutator_correction_rms: float
    bounded_increment_max_violation: float
    saturated_increment_fraction: float
    scheduler_boundary_correction_rms: float
    correction_phase0_max_abs: float
    exact_official_model_output_object: bool
    generator_forwards: int
    operator_mode: str = V7_RESIDUAL_ACTION_SECTION
    v8_radius_scale: float = MAIN_V8_RADIUS_SCALE
    exact_noop_phase_zero: bool = False
    rho_zero_selected_noop_clean_object: bool = False
    rho_zero_noop_velocity_exact_parity: bool = False
    rho_zero_noop_velocity_rms_error: float = 0.0
    removed_action_common_mode_rms: float = 0.0
    gauge_phase_increment_rms_error: float = 0.0
    gauge_phase_increment_tolerance: float = 0.0
    full_quotient_raw_rms: float = 0.0
    full_quotient_bounded_rms: float = 0.0
    full_quotient_radius_mean: float = 0.0
    full_quotient_saturated_fraction: float = 0.0
    frozen_action_clean_roundtrip_rms_error: float = 0.0
    frozen_action_clean_roundtrip_max_abs_error: float = 0.0
    v8_local_fp32_frozen_action_for_radius: bool = False
    v8_scheduler_model_output_fp32: bool = False
    scheduler_clean_roundtrip_rms_error: float = 0.0
    scheduler_clean_roundtrip_max_abs_error: float = 0.0
    post_boundary_increment_max_violation: float = 0.0
    post_boundary_increment_tolerance: float = 0.0


@dataclass
class RelationalMotionCommutatorTrace:
    adapter_loaded: bool
    commutator_config: commutator.MotionCommutatorConfig = (
        MAIN_COMMUTATOR_CONFIG
    )
    operator_mode: str = V7_RESIDUAL_ACTION_SECTION
    feasible_quotient_config: gauge.FeasibleQuotientConfig = (
        MAIN_FEASIBLE_QUOTIENT_CONFIG
    )
    v8_training_matched: bool = False
    records: list[RelationalMotionCommutatorStepRecord] = field(
        default_factory=list
    )
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": runtime_contract(
                self.commutator_config,
                operator_mode=self.operator_mode,
                feasible_quotient_config=self.feasible_quotient_config,
                v8_training_matched=self.v8_training_matched,
            ),
            "adapter_loaded": self.adapter_loaded,
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def _tensor_stat(value: Any, *, label: str) -> float:
    try:
        result = float(value.detach().float().cpu().item())
    except Exception as error:
        raise RelationalMotionCommutatorInferenceError(
            f"cannot serialize {label}"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise RelationalMotionCommutatorInferenceError(
            f"{label} must be finite and nonnegative"
        )
    return result


def select_frozen_action_clean_for_operator(
    local_fp32_apg_clean: Any,
    official_native_roundtrip_clean: Any,
    *,
    operator_mode: str,
) -> Any:
    """Select the frozen action section without hiding a precision mismatch.

    V8 training constructs its radius from the local fp32 APG clean field.
    Deployment therefore uses that same object after the native-BF16 official
    APG velocity has been certified bit-exact.  V7 retains its historical
    official-velocity round trip.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise RelationalMotionCommutatorInferenceError(
            "frozen-action clean selection requires PyTorch"
        ) from error
    if operator_mode not in OPERATOR_MODES:
        raise RelationalMotionCommutatorInferenceError(
            "unknown operator for frozen-action clean selection"
        )
    if (
        not isinstance(local_fp32_apg_clean, torch.Tensor)
        or not isinstance(official_native_roundtrip_clean, torch.Tensor)
        or tuple(local_fp32_apg_clean.shape)
        != tuple(official_native_roundtrip_clean.shape)
        or local_fp32_apg_clean.dtype != torch.float32
        or official_native_roundtrip_clean.dtype != torch.float32
        or local_fp32_apg_clean.device
        != official_native_roundtrip_clean.device
        or not bool(torch.isfinite(local_fp32_apg_clean).all())
        or not bool(torch.isfinite(official_native_roundtrip_clean).all())
    ):
        raise RelationalMotionCommutatorInferenceError(
            "frozen-action clean candidates differ in fp32 geometry"
        )
    return (
        local_fp32_apg_clean
        if operator_mode == V8_RECONSTRUCTION_SECTION_FQT
        else official_native_roundtrip_clean
    )


def project_relational_motion_commutator_step(
    raw: RawRelationalMotionCommutatorStep,
    *,
    frozen_action_momentum: Any,
    frozen_noop_momentum: Any,
    adapted_noop_momentum: Any,
    adapted_action_momentum: Any,
) -> tuple[Any, RelationalMotionCommutatorStepRecord]:
    """Project five same-state editor branches into one UniPC model output."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH has torch
        raise RelationalMotionCommutatorInferenceError(
            "v7 projection requires PyTorch"
        ) from error
    if not isinstance(raw, RawRelationalMotionCommutatorStep):
        raise RelationalMotionCommutatorInferenceError(
            "raw step has the wrong contract type"
        )
    _validate_inference_commutator_config(raw.commutator_config)
    radius_scale = validated_feasible_quotient_radius_scale(
        raw.feasible_quotient_config,
        operator_mode=raw.operator_mode,
    )
    if type(raw.step_index) is not int or not 0 <= raw.step_index < 40:
        raise RelationalMotionCommutatorInferenceError(
            "step_index must be an integer in [0,40)"
        )
    if raw.model_id != "transformer_1":
        raise RelationalMotionCommutatorInferenceError(
            "v7 exact-1.3B path requires transformer_1"
        )
    if not math.isfinite(float(raw.timestep_float)) or raw.apg.momentum != 0.0:
        raise RelationalMotionCommutatorInferenceError(
            "timestep or momentum-zero APG contract differs"
        )
    tensors = (
        raw.sample_packed,
        raw.official_model_output,
        raw.frozen_negative_velocity_packed,
        raw.frozen_noop_velocity_packed,
        raw.frozen_action_velocity_packed,
        raw.adapted_noop_velocity_packed,
        raw.adapted_action_velocity_packed,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise RelationalMotionCommutatorInferenceError(
            "all five branch values must be tensors"
        )
    if any(tuple(value.shape) != raw.layout.packed_shape for value in tensors):
        raise RelationalMotionCommutatorInferenceError(
            "five-branch packed shapes differ"
        )
    for velocity in tensors[2:]:
        if velocity.dtype != torch.bfloat16 or not bool(torch.isfinite(velocity).all()):
            raise RelationalMotionCommutatorInferenceError(
                "branch velocities must be finite native BF16"
            )
    if raw.sample_packed.dtype != torch.float32:
        raise RelationalMotionCommutatorInferenceError(
            "official noisy query must be float32"
        )
    sigma = raw.sigma
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma))
        or not bool(sigma > 0)
        or not math.isclose(
            float(raw.sigma_float),
            float(sigma.item()),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
    ):
        raise RelationalMotionCommutatorInferenceError(
            "sigma must be the matching positive CPU float32 UniPC scalar"
        )

    sample = tri._packed_to_spatial(raw.sample_packed, raw.layout)
    official_velocity = tri._packed_to_spatial(
        raw.official_model_output, raw.layout
    )
    negative_velocity = tri._packed_to_spatial(
        raw.frozen_negative_velocity_packed, raw.layout
    )
    frozen_noop_velocity = tri._packed_to_spatial(
        raw.frozen_noop_velocity_packed, raw.layout
    )
    frozen_action_velocity = tri._packed_to_spatial(
        raw.frozen_action_velocity_packed, raw.layout
    )
    adapted_noop_velocity = tri._packed_to_spatial(
        raw.adapted_noop_velocity_packed, raw.layout
    )
    adapted_action_velocity = tri._packed_to_spatial(
        raw.adapted_action_velocity_packed, raw.layout
    )
    negative_clean = tri.pinned_raw_condition_clean(
        sample, negative_velocity, sigma
    )
    guidance = raw.apg.guidance_scale_for(raw.model_id)

    def guided(velocity: Any, momentum: Any) -> Any:
        condition = tri.pinned_raw_condition_clean(sample, velocity, sigma)
        return tri._normalized_guidance(
            condition,
            negative_clean,
            guidance,
            momentum,
            raw.apg.eta,
            raw.apg.norm_threshold,
        )

    frozen_noop_clean = guided(frozen_noop_velocity, frozen_noop_momentum)
    local_frozen_action_clean = guided(
        frozen_action_velocity, frozen_action_momentum
    )
    adapted_noop_clean = guided(adapted_noop_velocity, adapted_noop_momentum)
    adapted_action_clean = guided(
        adapted_action_velocity, adapted_action_momentum
    )
    frozen_noop_velocity_reference = tri._spatial_to_packed(
        (sample - frozen_noop_clean) / sigma, raw.layout
    ).to(
        device=raw.official_model_output.device,
        dtype=raw.official_model_output.dtype,
    )
    rebuilt_official = tri._spatial_to_packed(
        (sample - local_frozen_action_clean) / sigma, raw.layout
    ).to(
        device=raw.official_model_output.device,
        dtype=raw.official_model_output.dtype,
    )
    parity_error = rebuilt_official.float() - raw.official_model_output.float()
    parity_rms = _tensor_stat(
        parity_error.square().mean().sqrt(), label="official APG parity RMS"
    )
    parity_max = _tensor_stat(
        parity_error.abs().max(), label="official APG parity maximum"
    )
    if not bool(torch.equal(rebuilt_official, raw.official_model_output)):
        raise RelationalMotionCommutatorInferenceError(
            "locally rebuilt frozen-action APG differs from official model_output"
        )

    # Preserve the local fp32 APG clean field for the V8 radius because that
    # is the exact numerical program used by training.  V7 keeps its legacy
    # native-BF16 official velocity round trip.  In either case the official
    # object has already passed the bit-exact certificate above.
    official_roundtrip_frozen_action_clean = sample - sigma * official_velocity
    roundtrip_error = (
        local_frozen_action_clean - official_roundtrip_frozen_action_clean
    )
    roundtrip_rms = _tensor_stat(
        roundtrip_error.square().mean().sqrt(),
        label="frozen-action clean native round-trip RMS",
    )
    roundtrip_max = _tensor_stat(
        roundtrip_error.abs().max(),
        label="frozen-action clean native round-trip maximum",
    )
    frozen_action_clean = select_frozen_action_clean_for_operator(
        local_frozen_action_clean,
        official_roundtrip_frozen_action_clean,
        operator_mode=raw.operator_mode,
    )
    clean_fields = []
    for clean in (
        frozen_action_clean,
        frozen_noop_clean,
        adapted_action_clean,
        adapted_noop_clean,
    ):
        clean_fields.append(
            packed_to_phase_grid(
                tri._spatial_to_packed(clean, raw.layout).float(),
                layout=raw.layout,
            )
        )
    frozen_action_phase, frozen_noop_phase, adapted_action_phase, adapted_noop_phase = (
        clean_fields
    )
    if raw.operator_mode not in OPERATOR_MODES:
        raise RelationalMotionCommutatorInferenceError(
            "raw step selected an unknown operator mode"
        )
    try:
        result = commutator.build_motion_commutator(
            adapted_action_phase,
            adapted_noop_phase,
            frozen_action_phase,
            frozen_noop_phase,
            config=raw.commutator_config,
        )
        if raw.operator_mode == V7_RESIDUAL_ACTION_SECTION:
            scheduled = commutator.apply_motion_commutator_to_official_tensor(
                frozen_action_phase,
                result.bounded_commutator_correction,
                step_index=raw.step_index,
            )
            executed_clean_phase = scheduled.executed_official_tensor
            feasible_projection = None
            feasible_execution = None
            gauge_anchor = None
        else:
            feasible_projection = gauge.project_complete_action_quotient(
                frozen_action_field=frozen_action_phase,
                frozen_noop_field=frozen_noop_phase,
                adapted_action_field=adapted_action_phase,
                adapted_noop_field=adapted_noop_phase,
                config=raw.feasible_quotient_config,
            )
            feasible_execution = gauge.execute_feasible_quotient_transport(
                frozen_noop_phase,
                feasible_projection.bounded_quotient,
                step_index=raw.step_index,
            )
            gauge_anchor = gauge.build_noop_gauge_anchor(
                frozen_action_phase, frozen_noop_phase
            )
            scheduled = None
            executed_clean_phase = feasible_execution.executed_clean_field
    except (
        commutator.MotionCommutatorError,
        gauge.GaugeAnchoredCommutatorError,
    ) as error:
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    rho = commutator.release_rho(raw.step_index)
    execution_rho = (
        scheduled.rho
        if scheduled is not None
        else feasible_execution.rho
    )
    if execution_rho != rho:
        raise RelationalMotionCommutatorInferenceError(
            "commutator release schedule differs"
        )
    correction_phase0 = result.bounded_commutator_correction[:, :1]
    phase0_max = _tensor_stat(
        correction_phase0.abs().max(), label="correction phase-zero maximum"
    )
    if phase0_max != 0.0 or not bool(
        torch.equal(correction_phase0, torch.zeros_like(correction_phase0))
    ):
        raise RelationalMotionCommutatorInferenceError(
            "bounded correction changed causal phase zero"
        )
    rho_zero_selected_noop = False
    rho_zero_noop_velocity_exact = False
    rho_zero_noop_velocity_rms = 0.0
    executed_clean_packed = None
    if raw.operator_mode == V7_RESIDUAL_ACTION_SECTION and rho == 0.0:
        if executed_clean_phase is not frozen_action_phase:
            raise RelationalMotionCommutatorInferenceError(
                "rho-zero clean execution lost object identity"
            )
        executed_velocity = raw.official_model_output
    else:
        if (
            raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
            and not bool(torch.equal(
                executed_clean_phase[:, 0], frozen_noop_phase[:, 0]
            ))
        ):
            raise RelationalMotionCommutatorInferenceError(
                "v8 reconstruction section changed frozen no-op phase zero"
            )
        rho_zero_selected_noop = bool(
            raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
            and rho == 0.0
            and executed_clean_phase is frozen_noop_phase
        )
        executed_clean_packed = phase_grid_to_packed(
            executed_clean_phase, layout=raw.layout
        )
        executed_velocity_fp32 = (
            (raw.sample_packed - executed_clean_packed) / sigma
        ).to(device=raw.official_model_output.device, dtype=torch.float32)
        executed_velocity = (
            executed_velocity_fp32
            if raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
            else executed_velocity_fp32.to(dtype=raw.official_model_output.dtype)
        )
        if raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT and rho == 0.0:
            bf16_compatibility = executed_velocity.to(
                dtype=raw.official_model_output.dtype
            )
            rho_zero_noop_velocity_exact = bool(
                torch.equal(
                    bf16_compatibility, frozen_noop_velocity_reference
                )
            )
            rho_zero_noop_velocity_rms = _tensor_stat(
                (
                    bf16_compatibility.float()
                    - frozen_noop_velocity_reference.float()
                ).square().mean().sqrt(),
                label="rho-zero no-op velocity reconstruction RMS",
            )
            if not rho_zero_noop_velocity_exact:
                raise RelationalMotionCommutatorInferenceError(
                    "rho-zero scheduler tensor differs from rebuilt no-op APG velocity"
                )
    if not bool(torch.isfinite(executed_velocity).all()):
        raise RelationalMotionCommutatorInferenceError(
            "scheduler model_output is non-finite"
        )
    exact_official = executed_velocity is raw.official_model_output
    expected_exact_official = bool(
        raw.operator_mode == V7_RESIDUAL_ACTION_SECTION and rho == 0.0
    )
    if exact_official is not expected_exact_official:
        raise RelationalMotionCommutatorInferenceError(
            "scheduler object identity differs from rho-zero contract"
        )

    scheduler_roundtrip_rms = 0.0
    scheduler_roundtrip_max = 0.0
    post_boundary_violation = 0.0
    post_boundary_tolerance = 0.0
    if raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT:
        if executed_velocity.dtype != torch.float32 or executed_clean_packed is None:
            raise RelationalMotionCommutatorInferenceError(
                "v8 scheduler transport must remain float32"
            )
        scheduler_effective_clean = (
            raw.sample_packed - sigma * executed_velocity
        )
        scheduler_roundtrip_error = (
            scheduler_effective_clean - executed_clean_packed.float()
        )
        scheduler_roundtrip_rms = _tensor_stat(
            scheduler_roundtrip_error.square().mean().sqrt(),
            label="scheduler clean round-trip RMS",
        )
        scheduler_roundtrip_max = _tensor_stat(
            scheduler_roundtrip_error.abs().max(),
            label="scheduler clean round-trip maximum",
        )
        frozen_noop_packed = phase_grid_to_packed(
            frozen_noop_phase, layout=raw.layout
        )
        frozen_noop_velocity_fp32 = (
            (raw.sample_packed - frozen_noop_packed) / sigma
        ).float()
        scheduler_effective_noop = (
            raw.sample_packed - sigma * frozen_noop_velocity_fp32
        )
        scheduler_correction_phase = packed_to_phase_grid(
            scheduler_effective_clean - scheduler_effective_noop,
            layout=raw.layout,
        )
        try:
            scheduler_correction_increments = commutator.phase_increments(
                commutator.causal_gauge(scheduler_correction_phase)
            )
        except commutator.MotionCommutatorError as error:
            raise RelationalMotionCommutatorInferenceError(str(error)) from error
        scheduler_increment_rms = gauge.phase_rms(
            scheduler_correction_increments
        )
        post_boundary_violation = _tensor_stat(
            (
                scheduler_increment_rms
                - float(rho) * feasible_projection.diagnostics.radius
            ).clamp_min(0.0).max(),
            label="post-scheduler-boundary increment maximum violation",
        )
        post_boundary_tolerance = max(
            float(raw.feasible_quotient_config.epsilon),
            64.0 * torch.finfo(torch.float32).eps,
        )
        if (
            scheduler_roundtrip_max > post_boundary_tolerance
            or post_boundary_violation > post_boundary_tolerance
        ):
            raise RelationalMotionCommutatorInferenceError(
                "v8 fp32 scheduler boundary changed the feasible clean transport"
            )

    diagnostics = result.diagnostics
    if feasible_projection is None:
        violation = (
            diagnostics.bounded_correction_increment_rms
            - diagnostics.correction_increment_rms_cap
        ).clamp_min(0.0)
        active_scale = diagnostics.bound_scale[:, 1:]
    else:
        feasible_diagnostics = feasible_projection.diagnostics
        violation = (
            feasible_diagnostics.bounded_increment_rms
            - feasible_diagnostics.radius
        ).clamp_min(0.0)
        active_scale = feasible_diagnostics.scale[:, 1:]
    max_violation = _tensor_stat(
        violation.max(), label="bounded increment maximum violation"
    )
    tolerance = max(
        float(raw.commutator_config.epsilon),
        8.0 * torch.finfo(torch.float32).eps,
    )
    if max_violation > tolerance:
        raise RelationalMotionCommutatorInferenceError(
            "commutator hard bound was violated"
        )
    saturated = _tensor_stat(
        (active_scale < 1.0).float().mean(),
        label="saturated increment fraction",
    )
    scheduler_correction_rms = _tensor_stat(
        (
            executed_velocity.float() - raw.official_model_output.float()
        ).square().mean().sqrt(),
        label="scheduler correction RMS",
    )
    exact_noop_phase_zero = bool(
        raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
        and torch.equal(executed_clean_phase[:, 0], frozen_noop_phase[:, 0])
    )
    if gauge_anchor is None:
        removed_common_mode_rms = 0.0
        gauge_increment_error_rms = 0.0
        gauge_increment_tolerance = 0.0
        full_raw_rms = 0.0
        full_bounded_rms = 0.0
        full_radius_mean = 0.0
        full_saturated = 0.0
    else:
        feasible_diagnostics = feasible_projection.diagnostics
        removed_common_mode_rms = _tensor_stat(
            gauge_anchor.removed_common_mode.square().mean().sqrt(),
            label="removed action common-mode RMS",
        )
        gauge_increment_error_rms = _tensor_stat(
            gauge_anchor.phase_increment_rms_error.mean(),
            label="gauge phase-increment RMS error",
        )
        gauge_increment_tolerance = float(
            gauge_anchor.phase_increment_tolerance
        )
        full_raw_rms = _tensor_stat(
            feasible_diagnostics.adapted_quotient.square().mean().sqrt(),
            label="full quotient raw RMS",
        )
        full_bounded_rms = _tensor_stat(
            feasible_projection.bounded_quotient.square().mean().sqrt(),
            label="full quotient bounded RMS",
        )
        full_radius_mean = _tensor_stat(
            feasible_diagnostics.radius[:, 1:].mean(),
            label="full quotient radius mean",
        )
        full_saturated = saturated
    record = RelationalMotionCommutatorStepRecord(
        step_index=raw.step_index,
        timestep=float(raw.timestep_float),
        sigma=float(raw.sigma_float),
        rho=float(rho),
        model_id=raw.model_id,
        transformer_forwards=5,
        frozen_negative_forwards=1,
        frozen_noop_forwards=1,
        frozen_action_forwards=1,
        adapted_noop_forwards=1,
        adapted_action_forwards=1,
        original_scheduler_calls=1,
        official_action_apg_exact=True,
        official_action_apg_rms_error=parity_rms,
        official_action_apg_max_abs_error=parity_max,
        raw_commutator_correction_rms=_tensor_stat(
            result.raw_commutator_correction.square().mean().sqrt(),
            label="raw commutator RMS",
        ),
        bounded_commutator_correction_rms=_tensor_stat(
            result.bounded_commutator_correction.square().mean().sqrt(),
            label="bounded commutator RMS",
        ),
        bounded_increment_max_violation=max_violation,
        saturated_increment_fraction=saturated,
        scheduler_boundary_correction_rms=scheduler_correction_rms,
        correction_phase0_max_abs=phase0_max,
        exact_official_model_output_object=exact_official,
        generator_forwards=0,
        operator_mode=raw.operator_mode,
        v8_radius_scale=radius_scale,
        exact_noop_phase_zero=exact_noop_phase_zero,
        rho_zero_selected_noop_clean_object=rho_zero_selected_noop,
        rho_zero_noop_velocity_exact_parity=rho_zero_noop_velocity_exact,
        rho_zero_noop_velocity_rms_error=rho_zero_noop_velocity_rms,
        removed_action_common_mode_rms=removed_common_mode_rms,
        gauge_phase_increment_rms_error=gauge_increment_error_rms,
        gauge_phase_increment_tolerance=gauge_increment_tolerance,
        full_quotient_raw_rms=full_raw_rms,
        full_quotient_bounded_rms=full_bounded_rms,
        full_quotient_radius_mean=full_radius_mean,
        full_quotient_saturated_fraction=full_saturated,
        frozen_action_clean_roundtrip_rms_error=roundtrip_rms,
        frozen_action_clean_roundtrip_max_abs_error=roundtrip_max,
        v8_local_fp32_frozen_action_for_radius=(
            raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
        ),
        v8_scheduler_model_output_fp32=(
            raw.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
        ),
        scheduler_clean_roundtrip_rms_error=scheduler_roundtrip_rms,
        scheduler_clean_roundtrip_max_abs_error=scheduler_roundtrip_max,
        post_boundary_increment_max_violation=post_boundary_violation,
        post_boundary_increment_tolerance=post_boundary_tolerance,
    )
    projected = tri.ProjectedVelocity(
        model_output=executed_velocity,
        correction_rms=scheduler_correction_rms,
        effective_guidance_scale=guidance,
        official_action_parity_rms_error=parity_rms,
        official_action_parity_max_abs_error=parity_max,
        official_action_exact_parity=True,
        sample_dtype=str(sample.dtype),
        branch_velocity_dtype=str(frozen_action_velocity.dtype),
        official_model_output_dtype=str(raw.official_model_output.dtype),
    )
    return projected, record


class _InstalledFiveBranch(v5._InstalledFourBranch):
    """Extend the audited v5 capture with an adapted semantic-noop query."""

    def __init__(
        self,
        *args: Any,
        commutator_config: commutator.MotionCommutatorConfig,
        operator_mode: str = V7_RESIDUAL_ACTION_SECTION,
        feasible_quotient_config: gauge.FeasibleQuotientConfig = (
            MAIN_FEASIBLE_QUOTIENT_CONFIG
        ),
        v8_training_matched: bool = False,
        **kwargs: Any,
    ) -> None:
        _validate_inference_commutator_config(commutator_config)
        if operator_mode not in OPERATOR_MODES:
            raise RelationalMotionCommutatorInferenceError(
                "five-branch hook received an unknown operator mode"
            )
        radius_scale = validated_feasible_quotient_radius_scale(
            feasible_quotient_config,
            operator_mode=operator_mode,
        )
        if type(v8_training_matched) is not bool or (
            v8_training_matched
            and operator_mode != V8_RECONSTRUCTION_SECTION_FQT
        ):
            raise RelationalMotionCommutatorInferenceError(
                "five-branch hook received invalid v8 training-match status"
            )
        if (
            radius_scale != MAIN_V8_RADIUS_SCALE
            and not v8_training_matched
        ):
            raise RelationalMotionCommutatorInferenceError(
                "five-branch radius ablation requires a trained V8 checkpoint"
            )
        self.commutator_config = commutator_config
        self.operator_mode = operator_mode
        self.feasible_quotient_config = feasible_quotient_config
        self.v8_training_matched = v8_training_matched
        super().__init__(*args, execution_arm="main", **kwargs)
        if not self.adapter_loaded:
            raise RelationalMotionCommutatorInferenceError(
                "v7 five-branch hook requires a loaded adapter"
            )
        self.trace = RelationalMotionCommutatorTrace(
            adapter_loaded=True,
            commutator_config=commutator_config,
            operator_mode=operator_mode,
            feasible_quotient_config=feasible_quotient_config,
            v8_training_matched=v8_training_matched,
        )
        self._last_adapted_noop_momentum: Any = None

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RelationalMotionCommutatorInferenceError(
                "shared_step ran outside a validated sample"
            )
        if not hasattr(state, "pending_adapted_noop"):
            state.pending_adapted_noop = None
            state.adapted_noop_momentum = tri._MomentumBuffer(
                state.apg.momentum, branch="adapted_noop"
            )
            self._last_adapted_noop_momentum = state.adapted_noop_momentum
        bound = tri._bind_call(self._original_shared_step, args, kwargs)
        try:
            model_id = str(bound["model_id"])
            prompt = bound["cond_embeds"]
        except KeyError as error:
            raise RelationalMotionCommutatorInferenceError(
                "shared_step lacks pinned branch arguments"
            ) from error
        if state.pending_negative is None:
            if prompt is not v5._branch_prompt(state, model_id, "uncond"):
                raise RelationalMotionCommutatorInferenceError(
                    "negative prompt object differs"
                )
            prediction = self._call_frozen(*args, **kwargs)
            state.pending_negative = v5._CapturedForward(
                tuple(args),
                dict(kwargs),
                bound,
                self._query_prediction(prediction, branch="frozen_negative"),
            )
            return prediction
        if any(
            value is not None
            for value in (
                state.pending_base_action,
                state.pending_noop,
                state.pending_adapted_noop,
                state.pending_adapted_action,
            )
        ):
            raise RelationalMotionCommutatorInferenceError(
                "more than two official shared_step calls occurred before UniPC"
            )
        negative = state.pending_negative
        if prompt is not v5._branch_prompt(state, model_id, "action"):
            raise RelationalMotionCommutatorInferenceError(
                "action prompt object differs"
            )
        if str(negative.bound.get("model_id")) != model_id:
            raise RelationalMotionCommutatorInferenceError(
                "negative and action model_id differ"
            )
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            tri._same_object(negative.bound.get(name), bound.get(name), label=name)
        tri._equal_metadata(
            negative.bound.get("batch_vae_seqlen"),
            bound.get("batch_vae_seqlen"),
            label="batch_vae_seqlen",
        )
        noop_prompt = v5._branch_prompt(state, model_id, "noop")
        noop_args, noop_kwargs = tri._replace_argument(
            self._original_shared_step,
            args,
            kwargs,
            name="cond_embeds",
            value=noop_prompt,
        )
        shape = getattr(noop_prompt, "shape", None)
        if shape is None or len(shape) < 2:
            raise RelationalMotionCommutatorInferenceError(
                "noop embedding lacks [B,L,D] geometry"
            )
        noop_args, noop_kwargs = tri._replace_argument(
            self._original_shared_step,
            noop_args,
            noop_kwargs,
            name="batch_text_seqlen",
            value=[int(shape[1])],
        )
        noop_bound = tri._bind_call(
            self._original_shared_step, noop_args, noop_kwargs
        )
        for name in ("model_id", "noisy_latents", "timesteps", "rotary_embs"):
            if name == "model_id":
                tri._equal_metadata(
                    bound.get(name), noop_bound.get(name), label="noop model_id"
                )
            else:
                tri._same_object(
                    bound.get(name),
                    noop_bound.get(name),
                    label=f"action/noop {name}",
                )
        tri._equal_metadata(
            bound.get("batch_vae_seqlen"),
            noop_bound.get("batch_vae_seqlen"),
            label="action/noop batch_vae_seqlen",
        )

        # Physical query order exactly matches the immutable v7 inference
        # branch order after the already-captured frozen negative.
        frozen_noop_prediction = self._call_frozen(*noop_args, **noop_kwargs)
        frozen_action_prediction = self._call_frozen(*args, **kwargs)
        adapted_noop_prediction = self._call_adapted(*noop_args, **noop_kwargs)
        adapted_action_prediction = self._call_adapted(*args, **kwargs)
        state.pending_noop = self._query_prediction(
            frozen_noop_prediction, branch="frozen_noop"
        )
        state.pending_base_action = v5._CapturedForward(
            tuple(args),
            dict(kwargs),
            bound,
            self._query_prediction(
                frozen_action_prediction, branch="frozen_action"
            ),
        )
        state.pending_adapted_noop = self._query_prediction(
            adapted_noop_prediction, branch="adapted_noop"
        )
        state.pending_adapted_action = self._query_prediction(
            adapted_action_prediction, branch="adapted_action"
        )
        return frozen_action_prediction

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RelationalMotionCommutatorInferenceError(
                "scheduler.step ran outside a validated sample"
            )
        pending_adapted_noop = getattr(state, "pending_adapted_noop", None)
        if any(
            value is None
            for value in (
                state.pending_negative,
                state.pending_base_action,
                state.pending_noop,
                pending_adapted_noop,
                state.pending_adapted_action,
            )
        ):
            raise RelationalMotionCommutatorInferenceError(
                "scheduler.step arrived before all five editor branches"
            )
        official = tri._extract_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = tri._extract_argument(args, kwargs, index=1, name="timestep")
        sample = tri._extract_argument(args, kwargs, index=2, name="sample")
        step_index, sigma, sigma_float = tri._resolve_sigma(
            self.scheduler, timestep
        )
        base_action = state.pending_base_action
        raw = RawRelationalMotionCommutatorStep(
            step_index=step_index,
            timestep=timestep,
            timestep_float=tri._coerce_scalar(timestep, label="timestep"),
            sigma=sigma,
            sigma_float=sigma_float,
            model_id=str(base_action.bound["model_id"]),
            sample_packed=sample,
            official_model_output=official,
            frozen_negative_velocity_packed=state.pending_negative.prediction,
            frozen_noop_velocity_packed=state.pending_noop,
            frozen_action_velocity_packed=base_action.prediction,
            adapted_noop_velocity_packed=pending_adapted_noop,
            adapted_action_velocity_packed=state.pending_adapted_action,
            apg=state.apg,
            layout=self.layout,
            commutator_config=self.commutator_config,
            operator_mode=self.operator_mode,
            feasible_quotient_config=self.feasible_quotient_config,
        )
        projected, record = project_relational_motion_commutator_step(
            raw,
            frozen_action_momentum=state.base_action_momentum,
            frozen_noop_momentum=state.base_noop_momentum,
            adapted_noop_momentum=state.adapted_noop_momentum,
            adapted_action_momentum=state.adapted_action_momentum,
        )
        call_args, call_kwargs = tri._replace_argument(
            self._original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=projected.model_output,
        )
        result = self._original_scheduler_step(*call_args, **call_kwargs)
        state.integrated_steps += 1
        self.trace.records.append(record)
        state.pending_negative = None
        state.pending_base_action = None
        state.pending_noop = None
        state.pending_adapted_noop = None
        state.pending_adapted_action = None
        return result

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        self._last_adapted_noop_momentum = None
        result = super()._wrapped_sample(*args, **kwargs)
        buffer = self._last_adapted_noop_momentum
        if buffer is None or buffer.update_count != self.expected_steps:
            raise RelationalMotionCommutatorInferenceError(
                "adapted-noop APG branch count differs"
            )
        return result


@contextmanager
def relational_motion_commutator_unipc_hook(
    renderer_or_diffusion: Any,
    *,
    adapter_model: Any,
    source_clean: Any,
    noop_prompt_embeds: Any,
    latent_shape: Sequence[int],
    bernini_commit: str,
    wan_diffusion_path: str | Path,
    expected_steps: int = NUM_DENOISING_STEPS,
    expected_flow_shift: float = 5.0,
    commutator_config: commutator.MotionCommutatorConfig = (
        MAIN_COMMUTATOR_CONFIG
    ),
    operator_mode: str = V7_RESIDUAL_ACTION_SECTION,
    feasible_quotient_config: gauge.FeasibleQuotientConfig = (
        MAIN_FEASIBLE_QUOTIENT_CONFIG
    ),
    v8_training_matched: bool = False,
) -> Iterator[RelationalMotionCommutatorTrace]:
    """Install one reversible, non-stackable five-branch v7 hook."""

    if adapter_model is None:
        raise RelationalMotionCommutatorInferenceError(
            "v7 inference requires one loaded adapter"
        )
    if expected_steps != NUM_DENOISING_STEPS:
        raise RelationalMotionCommutatorInferenceError(
            "v7 requires exactly 40 official UniPC steps"
        )
    _validate_inference_commutator_config(commutator_config)
    if operator_mode not in OPERATOR_MODES:
        raise RelationalMotionCommutatorInferenceError(
            "hook selected an unknown reconstruction-section operator"
        )
    diffusion = tri.resolve_diffusion_core(renderer_or_diffusion)
    try:
        bridge = _InstalledFiveBranch(
            diffusion,
            adapter_model=adapter_model,
            commutator_config=commutator_config,
            operator_mode=operator_mode,
            feasible_quotient_config=feasible_quotient_config,
            v8_training_matched=v8_training_matched,
            source_clean=source_clean,
            noop_prompt_embeds=noop_prompt_embeds,
            noop_prompt_embeds_t2=None,
            latent_shape=latent_shape,
            clean_field_callback=lambda fields: fields.action_guided_clean,
            expected_steps=expected_steps,
            expected_flow_shift=expected_flow_shift,
            projector=tri.project_clean_fields,
            bernini_commit=bernini_commit,
            wan_diffusion_path=wan_diffusion_path,
        )
        bridge.install()
    except (
        v5.PriorTangentInferenceError,
        tri.TriBranchHookError,
        RelationalMotionCommutatorInferenceError,
    ) as error:
        if isinstance(error, RelationalMotionCommutatorInferenceError):
            raise
        raise RelationalMotionCommutatorInferenceError(str(error)) from error
    try:
        yield bridge.trace
    finally:
        try:
            bridge.restore()
        except tri.TriBranchHookError as error:
            raise RelationalMotionCommutatorInferenceError(str(error)) from error


def validate_runtime_schedule_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(sigma_strata.PINNED_POSITIVE_SIGMAS),
        "positive_sigmas_float32_be_hex": list(
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": (
            sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        ),
    }
    if not isinstance(audit, Mapping) or dict(audit) != expected:
        raise RelationalMotionCommutatorInferenceError(
            "runtime UniPC schedule differs"
        )
    return expected


def validate_execution_trace(
    trace: RelationalMotionCommutatorTrace,
    *,
    runtime_schedule_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete 40-step/200-forward execution trace."""

    if (
        not isinstance(trace, RelationalMotionCommutatorTrace)
        or trace.adapter_loaded is not True
        or trace.operator_mode not in OPERATOR_MODES
        or trace.sample_calls != 1
        or len(trace.records) != NUM_DENOISING_STEPS
    ):
        raise RelationalMotionCommutatorInferenceError(
            "v7 trace is not one complete adapter-loaded 40-step sample"
        )
    audited = validate_runtime_schedule_audit(runtime_schedule_audit)
    is_v8 = trace.operator_mode == V8_RECONSTRUCTION_SECTION_FQT
    radius_scale = validated_feasible_quotient_radius_scale(
        trace.feasible_quotient_config,
        operator_mode=trace.operator_mode,
    )
    if radius_scale != MAIN_V8_RADIUS_SCALE and not trace.v8_training_matched:
        raise RelationalMotionCommutatorInferenceError(
            "trace radius ablation is not bound to a trained V8 checkpoint"
        )
    for index, record in enumerate(trace.records):
        selected = sigma_strata.select_sigma_stratum(index)
        expected_rho = commutator.release_rho(index)
        if (
            record.step_index != index
            or record.timestep != float(selected.timestep)
            or not math.isclose(
                record.sigma,
                selected.sigma,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            or record.rho != expected_rho
            or record.operator_mode != trace.operator_mode
            or record.v8_radius_scale != radius_scale
            or record.model_id != "transformer_1"
            or record.transformer_forwards != 5
            or record.frozen_negative_forwards != 1
            or record.frozen_noop_forwards != 1
            or record.frozen_action_forwards != 1
            or record.adapted_noop_forwards != 1
            or record.adapted_action_forwards != 1
            or record.original_scheduler_calls != 1
            or record.official_action_apg_exact is not True
            or record.frozen_action_clean_roundtrip_rms_error < 0.0
            or record.frozen_action_clean_roundtrip_max_abs_error < 0.0
            or record.v8_local_fp32_frozen_action_for_radius is not is_v8
            or record.v8_scheduler_model_output_fp32 is not is_v8
            or record.bounded_increment_max_violation
            > max(
                float(trace.commutator_config.epsilon),
                8.0 * 1.1920928955078125e-7,
            )
            or record.correction_phase0_max_abs != 0.0
            or record.exact_official_model_output_object
            is not (not is_v8 and expected_rho == 0.0)
            or record.exact_noop_phase_zero is not is_v8
            or record.rho_zero_selected_noop_clean_object
            is not (is_v8 and expected_rho == 0.0)
            or record.rho_zero_noop_velocity_exact_parity
            is not (is_v8 and expected_rho == 0.0)
            or record.rho_zero_noop_velocity_rms_error != 0.0
            or (
                is_v8
                and (
                    record.gauge_phase_increment_tolerance <= 0.0
                    or record.gauge_phase_increment_rms_error
                    > record.gauge_phase_increment_tolerance
                )
            )
            or (is_v8 and record.removed_action_common_mode_rms < 0.0)
            or (is_v8 and record.full_quotient_radius_mean <= 0.0)
            or (
                is_v8
                and (
                    record.post_boundary_increment_tolerance <= 0.0
                    or record.scheduler_clean_roundtrip_max_abs_error
                    > record.post_boundary_increment_tolerance
                    or record.post_boundary_increment_max_violation
                    > record.post_boundary_increment_tolerance
                )
            )
            or (
                is_v8
                and not 0.0
                <= record.full_quotient_saturated_fraction
                <= 1.0
            )
            or record.generator_forwards != 0
        ):
            raise RelationalMotionCommutatorInferenceError(
                f"v7 execution trace differs at step {index}"
            )
    payload = {
        "schema_version": (
            "bernini-reconstruction-section-fqt-inference-trace-v8"
            if is_v8
            else INFERENCE_RECEIPT_SCHEMA
        ),
        "contract": runtime_contract(
            trace.commutator_config,
            operator_mode=trace.operator_mode,
            feasible_quotient_config=trace.feasible_quotient_config,
            v8_training_matched=trace.v8_training_matched,
        ),
        "runtime_unipc_schedule_audit": audited,
        "trace": trace.as_dict(),
        "totals": {
            "transformer_forwards": sum(
                record.transformer_forwards for record in trace.records
            ),
            "original_scheduler_calls": sum(
                record.original_scheduler_calls for record in trace.records
            ),
            "generator_forwards": 0,
            "rho_zero_exact_official_steps": sum(
                record.exact_official_model_output_object
                for record in trace.records
            ),
            "rho_zero_noop_clean_section_steps": sum(
                record.rho_zero_selected_noop_clean_object
                for record in trace.records
            ),
        },
    }
    if payload["totals"]["transformer_forwards"] != 200:
        raise RelationalMotionCommutatorInferenceError(
            "v7 trace did not execute exactly 200 editor forwards"
        )
    if payload["totals"]["original_scheduler_calls"] != 40:
        raise RelationalMotionCommutatorInferenceError(
            "v7 trace did not execute exactly 40 original scheduler calls"
        )
    if payload["totals"]["rho_zero_exact_official_steps"] != (
        0 if is_v8 else len(LATE_EXACT_STEPS)
    ) or payload["totals"]["rho_zero_noop_clean_section_steps"] != (
        len(LATE_EXACT_STEPS) if is_v8 else 0
    ):
        raise RelationalMotionCommutatorInferenceError(
            "rho-zero reconstruction-section identity count differs"
        )
    payload["trace_digest"] = trainer.object_sha256(payload)
    return payload


__all__ = [
    "ADAPTER_SCALE",
    "INFERENCE_RECEIPT_SCHEMA",
    "LATE_EXACT_STEPS",
    "LATENT_PHASES",
    "MAIN_FEASIBLE_QUOTIENT_CONFIG",
    "MAIN_COMMUTATOR_CONFIG",
    "MAIN_V8_RADIUS_SCALE",
    "METHOD_NAME",
    "NUM_DENOISING_STEPS",
    "NUM_FRAMES",
    "OPERATOR_MODES",
    "RawRelationalMotionCommutatorStep",
    "RelationalMotionCommutatorInferenceError",
    "RelationalMotionCommutatorStepRecord",
    "RelationalMotionCommutatorTrace",
    "TRAINING_RECEIPT_SCHEMA",
    "V7_RESIDUAL_ACTION_SECTION",
    "V8_INFERENCE_RECEIPT_SCHEMA",
    "V8_METHOD_NAME",
    "V8_RECONSTRUCTION_SECTION_FQT",
    "V8_RADIUS_SCALE_CHOICES",
    "V8_TRAINING_METHOD_NAME",
    "V8_TRAINING_RECEIPT_SCHEMA",
    "expected_lora_targets",
    "expected_serialized_target_patterns",
    "feasible_quotient_config_for_radius_scale",
    "packed_to_phase_grid",
    "phase_grid_to_packed",
    "project_relational_motion_commutator_step",
    "relational_motion_commutator_unipc_hook",
    "runtime_contract",
    "select_frozen_action_clean_for_operator",
    "strict_load_adapter",
    "validate_execution_trace",
    "validate_runtime_schedule_audit",
    "validate_training_adapter_contract",
    "validated_feasible_quotient_radius_scale",
    "commutator",
    "gauge",
    "sigma_strata",
    "trainer",
]
