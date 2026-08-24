#!/usr/bin/env python3
"""Pinned four-rank AUH trainer for Bernini motion commutator LoRA v7.

The target-only arm is intentionally trainable even when the frozen T2V
relation audit is ineligible.  Every optimizer candidate nevertheless runs an
auditable seven-forward cell: five frozen model branches and the adapted
action/no-op pair, which are the only graph-bearing forwards.  The optional
relational auxiliary must be selected explicitly and fails closed.

Paired target video and the target-only generator are training-time teachers.
The deployed model receives only source video and an edit instruction.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_motion_kernel as cmkd  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_commutator as commutator  # noqa: E402
import motion_residual as motion  # noqa: E402
import relational_commutator_objective as objective  # noqa: E402
import train_cross_mode_cmsg_auh as v6_runtime  # noqa: E402
import train_cross_mode_cmsg_lora as v6_scope  # noqa: E402
import train_delta_lora as v4  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_prior_tangent_lora as v5  # noqa: E402


METHOD_NAME = "bernini-relational-motion-commutator-lora-v7-auh"
RECEIPT_SCHEMA = "bernini-r-1p3b-relational-commutator-auh-receipt-v7"
OPTIMIZER_SCHEMA = "bernini-r-1p3b-relational-commutator-auh-optimizer-v7"
ARTIFACT_VALIDATION_SCHEMA = (
    "bernini-rmc-v7-adapter-artifact-validation-v1"
)
INFERENCE_RECEIPT_SCHEMA = (
    "bernini-relational-motion-commutator-inference-receipt-v7"
)
INFERENCE_LOADER_MODULE = "infer_relational_motion_commutator.py"
INFERENCE_RUNNER_MODULE = "run_relational_motion_commutator_inference.py"
INFERENCE_FINALIZER_MODULE = (
    "finalize_relational_motion_commutator_checkpoint.py"
)
INFERENCE_PARITY_TESTS = (
    "test_infer_relational_motion_commutator.py",
    "test_run_relational_motion_commutator_inference_contract.py",
    "test_finalize_relational_motion_commutator_checkpoint.py",
    "test_infer_delta_lora_contract.py",
)
NUM_FRAMES = 81
LATENT_PHASES = 21
LEARNING_RATE = 2.0e-5
MINIMUM_TRAINING_SIGMA = 0.1
TEACHER_MODES = ("target_only", "relational_auxiliary")
METRICS_TIMING = "pre_optimizer_update"
FORWARD_CELL_ORDER = objective.FORWARD_BRANCH_ORDER
INFERENCE_FORWARD_ORDER = (
    "frozen_editor_negative_full_source",
    "frozen_editor_noop_full_source",
    "frozen_editor_action_full_source",
    "adapted_editor_noop_full_source",
    "adapted_editor_action_full_source",
)
MAIN_COMMUTATOR_CONFIG = commutator.MotionCommutatorConfig(
    max_correction_increment_ratio=0.25,
    correction_increment_rms_floor=1.0e-3,
    temporal_smoothing=True,
)


class RelationalCommutatorAUHError(RuntimeError):
    """Raised before an invalid candidate can update the v7 adapter."""


@dataclass(frozen=True)
class SevenForwardCellResult:
    weighted_loss: Any
    loss_result: Any
    inverse_sigma_weight: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pinned 81f Bernini relational commutator LoRA on four AUH GPUs"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument(
        "--expected-routing-jsonl-sha256", default=v5.STRICT_ROUTING_SHA256
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="target_only")
    parser.add_argument(
        "--relational-auxiliary-weight",
        type=float,
        default=0.0,
        help="must be zero for target_only and positive for relational_auxiliary",
    )
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument("--negative-prompt", default=v5.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument(
        "--inference-loader-parity-verified",
        action="store_true",
        help=(
            "assert that the immutable launcher ran the bundled strict loader "
            "and end-to-end runner contract tests before model construction"
        ),
    )
    return parser


def loss_config_from_args(args: argparse.Namespace) -> objective.RelationalCommutatorLossConfig:
    return objective.RelationalCommutatorLossConfig(
        relational_auxiliary_weight=float(args.relational_auxiliary_weight),
        commutator_config=MAIN_COMMUTATOR_CONFIG,
    )


def validate_cli(args: argparse.Namespace) -> None:
    if args.inference_loader_parity_verified is not True:
        raise RelationalCommutatorAUHError(
            "v7 training requires bundled inference-loader parity preflight"
        )
    if args.num_frames != NUM_FRAMES or legacy.LATENT_FRAMES != LATENT_PHASES:
        raise RelationalCommutatorAUHError("v7 requires exact 81-frame / 21-phase data")
    if type(args.max_steps) is not int or args.max_steps <= 0:
        raise RelationalCommutatorAUHError("max_steps must be a positive integer")
    if type(args.save_every) is not int or args.save_every < 0:
        raise RelationalCommutatorAUHError("save_every must be nonnegative")
    if float(args.learning_rate) != LEARNING_RATE:
        raise RelationalCommutatorAUHError(f"v7 fixes learning_rate to {LEARNING_RATE}")
    if not math.isfinite(float(args.weight_decay)) or args.weight_decay < 0.0:
        raise RelationalCommutatorAUHError("weight_decay must be finite and nonnegative")
    if not math.isfinite(float(args.max_grad_norm)) or args.max_grad_norm <= 0.0:
        raise RelationalCommutatorAUHError("max_grad_norm must be finite and positive")
    if args.teacher_mode not in TEACHER_MODES:
        raise RelationalCommutatorAUHError("unknown teacher mode")
    relational_weight = float(args.relational_auxiliary_weight)
    if not math.isfinite(relational_weight) or relational_weight < 0.0:
        raise RelationalCommutatorAUHError(
            "relational auxiliary weight must be finite and nonnegative"
        )
    if args.teacher_mode == "target_only" and relational_weight != 0.0:
        raise RelationalCommutatorAUHError(
            "target_only requires relational auxiliary weight exactly zero"
        )
    if args.teacher_mode == "relational_auxiliary" and relational_weight <= 0.0:
        raise RelationalCommutatorAUHError(
            "relational_auxiliary requires a positive explicit weight"
        )
    if args.noop_instruction != motion.DEFAULT_NOOP_INSTRUCTION:
        raise RelationalCommutatorAUHError("v7 pins the semantic no-op instruction")
    if args.negative_prompt != v5.DEFAULT_NEGATIVE_PROMPT:
        raise RelationalCommutatorAUHError("v7 pins Bernini's negative prompt")
    if args.expected_routing_jsonl_sha256 != v5.STRICT_ROUTING_SHA256:
        raise RelationalCommutatorAUHError("v7 requires the hash-bound strict359 route")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise RelationalCommutatorAUHError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_routing_jsonl_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise RelationalCommutatorAUHError(f"{name} must be lowercase SHA-256")
    if args.expected_bernini_commit.lower() != legacy.BERNINI_OFFICIAL_COMMIT:
        raise RelationalCommutatorAUHError("Bernini revision differs")
    if args.expected_veomni_commit.lower() != legacy.VEOMNI_TESTED_COMMIT:
        raise RelationalCommutatorAUHError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise RelationalCommutatorAUHError("checkpoint tree differs")
    if (
        v6_scope.LORA_RANK != 8
        or v6_scope.LORA_ALPHA != 8
        or v6_scope.EXPECTED_LORA_MODULES != 46
    ):
        raise RelationalCommutatorAUHError("v7 requires exact46 rank8/alpha8 LoRA")
    try:
        loss_config_from_args(args).validate()
    except objective.RelationalCommutatorObjectiveError as error:
        raise RelationalCommutatorAUHError(str(error)) from error


def _run_seven_forward_cell(
    *,
    renderer: Any,
    adapter_controller: Any,
    candidate: v6_runtime.MovedCandidate,
    step_index: int,
    loss_config: objective.RelationalCommutatorLossConfig,
) -> SevenForwardCellResult:
    """Execute exactly five frozen and two graph-bearing model forwards."""

    import torch

    shared_noisy = candidate.auxiliary["shared_noisy"]
    sigma = candidate.auxiliary["sigma"]
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            frozen_negative_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_negative
            )
            frozen_noop_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_noop
            )
            frozen_action_v = motion.renderer_velocity_prediction(
                renderer, candidate.editor_action
            )
    adapted_noop_v = motion.renderer_velocity_prediction(renderer, candidate.editor_noop)
    adapted_action_v = motion.renderer_velocity_prediction(
        renderer, candidate.editor_action
    )
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            generator_negative_v = motion.renderer_velocity_prediction(
                renderer, candidate.generator_negative
            )
            generator_action_v = motion.renderer_velocity_prediction(
                renderer, candidate.generator_action
            )

    velocities = (
        frozen_negative_v,
        frozen_noop_v,
        frozen_action_v,
        adapted_noop_v,
        adapted_action_v,
        generator_negative_v,
        generator_action_v,
    )
    if any(
        value.dtype != torch.bfloat16
        or tuple(value.shape) != tuple(shared_noisy.shape)
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise RelationalCommutatorAUHError(
            "all seven forwards must be finite native-BF16 fields on one query"
        )
    if any(value.requires_grad for value in velocities[:3]) or any(
        value.requires_grad for value in velocities[5:]
    ):
        raise RelationalCommutatorAUHError("one of five frozen branches retained a graph")
    if not adapted_noop_v.requires_grad or not adapted_action_v.requires_grad:
        raise RelationalCommutatorAUHError(
            "both adapted no-op and action branches must retain LoRA graphs"
        )

    try:
        frozen_negative, frozen_noop = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=frozen_negative_v,
            conditional_velocity=frozen_noop_v,
        )
        frozen_negative_action, frozen_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=frozen_negative_v,
            conditional_velocity=frozen_action_v,
        )
        frozen_negative_adapted_noop, adapted_noop = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=frozen_negative_v,
            conditional_velocity=adapted_noop_v,
        )
        frozen_negative_adapted_action, adapted_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=frozen_negative_v,
            conditional_velocity=adapted_action_v,
        )
        if not all(
            torch.equal(frozen_negative, value)
            for value in (
                frozen_negative_action,
                frozen_negative_adapted_noop,
                frozen_negative_adapted_action,
            )
        ):
            raise RelationalCommutatorAUHError(
                "editor APG branches reconstructed different negative clean fields"
            )
        generator_negative, generator_action = v6_runtime._generator_plain_cfg_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=generator_negative_v,
            action_velocity=generator_action_v,
        )
        fields = objective.SevenBranchCleanFields(
            frozen_editor_negative=frozen_negative.detach(),
            frozen_editor_noop=frozen_noop.detach(),
            frozen_editor_action=frozen_action.detach(),
            adapted_editor_noop=adapted_noop.float(),
            adapted_editor_action=adapted_action.float(),
            frozen_generator_negative=generator_negative.detach(),
            frozen_generator_action=generator_action.detach(),
            source_clean=v5._as_phase_grid(
                candidate.auxiliary["source_clean"].float()
            ).detach(),
            target_clean=v5._as_phase_grid(
                candidate.auxiliary["target_clean"].float()
            ).detach(),
        )
        result = objective.compute_relational_commutator_objective(
            fields,
            step_index=step_index,
            config=loss_config,
        )
        inverse_sigma = motion.clean_field_inverse_sigma_weight(
            sigma,
            weight_floor=sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        ).mean()
    except (
        objective.RelationalCommutatorObjectiveError,
        v5.PriorTangentTrainingError,
        motion.MotionContractError,
        v6_runtime.CMSGauhTrainingError,
    ):
        raise
    weighted = inverse_sigma * result.total
    if not bool(torch.isfinite(weighted)):
        raise RelationalCommutatorAUHError("weighted v7 loss is non-finite")
    return SevenForwardCellResult(
        weighted_loss=weighted,
        loss_result=result,
        inverse_sigma_weight=inverse_sigma,
    )


def _scalar(value: Any) -> float:
    return float(value.detach().float().mean().cpu().item())


def _loss_metrics(cell: SevenForwardCellResult) -> dict[str, float]:
    result = cell.loss_result
    metrics = {
        "loss_total": _scalar(result.total),
        "loss_weighted": _scalar(cell.weighted_loss),
        "loss_raw_target": _scalar(result.raw_target),
        "loss_noop_preservation": _scalar(result.noop_preservation),
        "loss_residual_temporal_jitter": _scalar(
            result.residual_temporal_jitter
        ),
        "loss_relational_auxiliary": _scalar(result.relational_auxiliary),
        "inverse_sigma_weight": _scalar(cell.inverse_sigma_weight),
        "rho": float(result.rho),
    }
    diagnostics = result.diagnostics
    eligibility = diagnostics.teacher_eligibility
    detached = objective.detached_receipt_diagnostics(result)
    metrics.update(
        {
            "teacher_eligible": float(bool(eligibility.eligible.all().item())),
            "teacher_kernel_alignment": _scalar(
                eligibility.centered_kernel_alignment
            ),
            "teacher_offdiag_rms": _scalar(
                eligibility.teacher.off_diagonal_relational_rms
            ),
            "target_offdiag_rms": _scalar(
                eligibility.target.off_diagonal_relational_rms
            ),
            "teacher_envelope_cosine": _scalar(eligibility.envelope_cosine),
            "teacher_envelope_error": _scalar(
                eligibility.envelope_relative_error
            ),
            "teacher_energy_ratio": _scalar(
                eligibility.teacher_target_energy_ratio
            ),
            "teacher_frequency_cosine": _scalar(
                eligibility.frequency_power_cosine
            ),
            "bound_mean_scale_active": float(
                detached["commutator_bound"]["mean_scale_active"]
            ),
            "bound_saturated_fraction_active": float(
                detached["commutator_bound"]["saturated_fraction_active"]
            ),
            "target_bound_saturated_fraction_active": float(
                detached["commutator_bound"][
                    "target_saturated_fraction_active"
                ]
            ),
            "target_bound_mean_scale_active": float(
                detached["commutator_bound"][
                    "target_bound_mean_scale_active"
                ]
            ),
            "floor_dominated_fraction_active": float(
                detached["commutator_bound"][
                    "floor_dominated_fraction_active"
                ]
            ),
            "target_floor_sufficient_fraction_active": float(
                detached["commutator_bound"][
                    "target_floor_sufficient_fraction_active"
                ]
            ),
            "target_required_kappa_median": float(
                detached["commutator_bound"][
                    "target_required_kappa_median"
                ]
            ),
            "target_required_kappa_p90": float(
                detached["commutator_bound"]["target_required_kappa_p90"]
            ),
            "target_required_kappa_max": float(
                detached["commutator_bound"]["target_required_kappa_max"]
            ),
            "target_required_kappa_near_zero_threshold": float(
                detached["commutator_bound"][
                    "target_required_kappa_near_zero_threshold"
                ]
            ),
            "frozen_increment_near_zero_fraction_active": float(
                detached["commutator_bound"][
                    "frozen_increment_near_zero_fraction_active"
                ]
            ),
            "target_required_kappa_near_zero_proxy_fraction_active": float(
                detached["commutator_bound"][
                    "target_required_kappa_near_zero_proxy_fraction_active"
                ]
            ),
            "target_required_kappa_exact_zero_unreachable_fraction_active": float(
                detached["commutator_bound"][
                    "target_required_kappa_exact_zero_unreachable_fraction_active"
                ]
            ),
        }
    )
    return metrics


def _immutable_contract(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    target_modules: Sequence[str],
    checkpoint: Path,
    loss_config: objective.RelationalCommutatorLossConfig,
) -> dict[str, Any]:
    release = list(commutator.release_rho_schedule())
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "routing_digest": router.digest,
        "routing_file_sha256": router.file_sha256,
        "eligible_route_count": len(eligible_routes),
        "eligible_route_stream_sha256": legacy.object_sha256(
            [
                {
                    "row_index": row_index,
                    "iid": route.iid,
                    "tier": route.tier,
                    "full_target_weight": route.full_target_weight,
                }
                for row_index, route in eligible_routes
            ]
        ),
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "seed": int(args.seed),
        "learning_rate": LEARNING_RATE,
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "teacher_mode": args.teacher_mode,
        "loss_config": asdict(loss_config),
        "objective_contract": objective.immutable_objective_contract(loss_config),
        "lora": {
            "scope": v6_scope.LORA_SCOPE,
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
        },
        "training_bridge_endpoint": "source(beta=0)",
        "target_endpoint_teacher_leakage_forbidden": True,
        "forward_cell_order": list(FORWARD_CELL_ORDER),
        "forwards_per_candidate": 7,
        "graph_forwards_per_candidate": 2,
        "graph_branch_order": list(objective.GRAPH_BRANCHES),
        "inference_forward_order": list(INFERENCE_FORWARD_ORDER),
        "inference_forwards_per_step": 5,
        "editor_guidance": {
            "mode": "official_momentum_zero_apg",
            "guidance_scale": v5.APG_GUIDANCE_SCALE,
            "eta": v5.APG_ETA,
            "norm_threshold": v5.APG_NORM_THRESHOLD,
            "momentum": v5.APG_MOMENTUM,
        },
        "generator_guidance": {
            "mode": "official_t2v_plain_cfg",
            "native_velocity_formula": "v_negative+4*(v_action-v_negative)",
            "scale": v6_runtime.T2V_GUIDANCE_SCALE,
            "combine_before_fp32_clean_reconstruction": True,
            "training_diagnostic_only_in_target_only_mode": True,
        },
        "target_motion_teacher": "Q0(target_clean-source_clean)",
        "target_used_as_model_condition": False,
        "t2v_relation_pointwise_coordinate_loss": False,
        "relational_teacher_semantic_validity": objective.RELATIONAL_METRIC_LIMITATION,
        "training_correction": "raw_Ctheta",
        "deployment_correction": "temporal_smooth_then_hard_bound_Ctheta",
        "hard_bound_formula": "max(kappa*frozen_increment_rms,absolute_floor)",
        "metrics_timing": METRICS_TIMING,
        "deployment_diagnostics": {
            "target_bound_mean_scale_active": True,
            "floor_dominated_fraction_active": True,
            "target_required_kappa_statistics": ["median", "p90", "max"],
            "near_zero_frozen_increment": (
                "exact float64 division when positive; finite threshold proxy "
                "and explicit unreachable fraction at exact zero"
            ),
        },
        "release_schedule": release,
        "release_schedule_sha256": legacy.object_sha256(release),
        "zero_release_steps": [
            index for index, rho in enumerate(release) if rho == 0.0
        ],
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
        "inference_loader_parity": {
            "verified": True,
            "verification_stage": "immutable_launcher_preflight_before_model_load",
            "loader_module": INFERENCE_LOADER_MODULE,
            "runner_module": INFERENCE_RUNNER_MODULE,
            "finalizer_module": INFERENCE_FINALIZER_MODULE,
            "training_receipt_schema": RECEIPT_SCHEMA,
            "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
            "contract_tests": list(INFERENCE_PARITY_TESTS),
            "source_revision_and_archive_bound": True,
            "strict_loader_rejects_pending_canary_and_incomplete_cycle": True,
        },
        "resume_integrated": False,
    }
    if len(target_modules) != 46:
        raise RelationalCommutatorAUHError(
            f"v7 LoRA scope resolved {len(target_modules)} modules, expected 46"
        )
    return {"value": value, "digest": legacy.object_sha256(value)}


def _optimizer_payload(
    *,
    optimizer: Any,
    global_step: int,
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
    step_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": OPTIMIZER_SCHEMA,
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "immutable_contract": dict(immutable),
        "parameter_names": list(parameter_names),
        "step_audit": list(step_audit),
        "step_audit_sha256": legacy.object_sha256(list(step_audit)),
        "resume_integrated": False,
    }


def _build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    metrics: Optional[Mapping[str, float]],
    step_audit: Sequence[Mapping[str, Any]],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    checkpoint: Path,
    bernini_revision: str,
    veomni_revision: str,
    distributed: Any,
    backend: str,
    target_modules: Sequence[str],
    named_trainable: Sequence[tuple[str, Any]],
    initialization_digest: str,
    transformers_version: str,
    immutable: Mapping[str, Any],
    optimizer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    names = v4._optimizer_parameter_names(named_trainable)
    accepted_indices = [int(record["sigma_schedule_index"]) for record in step_audit]
    expected_indices = list(range(global_step)) if global_step <= 40 else [
        index % 40 for index in range(global_step)
    ]
    if accepted_indices != expected_indices:
        raise RelationalCommutatorAUHError("accepted sigma schedule audit differs")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "global_step": int(global_step),
        "max_steps": int(args.max_steps),
        "formal_40_sigma_cycle_complete": global_step >= 40,
        "accepted_sigma_schedule_indices": accepted_indices,
        "step_audit": list(step_audit),
        "step_audit_sha256": legacy.object_sha256(list(step_audit)),
        "last_metrics": dict(metrics) if metrics is not None else None,
        "metrics_timing": METRICS_TIMING,
        "immutable_contract": dict(immutable),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
        },
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "routing": router.receipt(),
        },
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": v6_scope.LORA_SCOPE,
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "parameter_names_sha256": legacy.object_sha256(names),
            "initialization_digest": initialization_digest,
            "checkpoint_parameter_digest": v4._checkpoint_parameter_digest(
                named_trainable
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
            "checkpoint_state_digest": v4._stable_recursive_digest(
                optimizer_payload
            ),
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "transformers_version": transformers_version,
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_generator_and_target": True,
        "inference_generator_forwards": 0,
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "experimental_training": True,
        "dataset_post_video_acceptance": "pending",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "resume_integrated": False,
        # Source-level loader/runner preflight runs before model construction,
        # but it cannot certify files that have not yet been serialized.  A
        # separate post-save finalizer must strict-reload the exact artifact
        # before changing this release gate to false.
        "inference_loader_parity_pending": True,
        "inference_loader_parity": dict(
            immutable["value"]["inference_loader_parity"]
        ),
        "artifact_validation": {
            "schema_version": ARTIFACT_VALIDATION_SCHEMA,
            "verified": False,
            "status": "pending_post_save_strict_reload",
        },
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    loss_config = loss_config_from_args(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise RelationalCommutatorAUHError(str(error)) from error
    if transformer_config["num_attention_heads"] % 4:
        raise RelationalCommutatorAUHError(
            "1.3B attention heads must divide Ulysses=4"
        )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, SYSTEM_PROMPTS, process_renderer_sample

    if DEFAULT_NEG_PROMPT != v5.DEFAULT_NEGATIVE_PROMPT:
        raise RelationalCommutatorAUHError("runtime Bernini negative prompt differs")
    if SYSTEM_PROMPTS.get("t2v") != v6_runtime.T2V_SYSTEM_PROMPT:
        raise RelationalCommutatorAUHError("runtime Bernini T2V system prompt differs")

    distributed = legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise RelationalCommutatorAUHError(
            "AUH v7 training requires exactly four ranks"
        )
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    output = Path(args.output).expanduser().resolve()
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=False,
    )
    try:
        router = motion.ReviewRouter.load(args.routing_jsonl, default_tier="reject")
        eligible_routes = v4._build_eligible_routes(dataset, router)
        v6_runtime._strict_router(args, router, eligible_routes, dataset)
    except (motion.MotionContractError, v6_runtime.CMSGauhTrainingError) as error:
        raise RelationalCommutatorAUHError(str(error).replace("v6", "v7")) from error

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except legacy.TrainingContractError as error:
        raise RelationalCommutatorAUHError(str(error)) from error
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.t5_text_encoder.eval()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        available_modules = legacy.select_attention_projection_names(base_model)
        target_modules = v6_scope.select_cmsg_lora_targets(available_modules)
    except (
        legacy.TrainingContractError,
        v6_scope.CrossModeCMSGTrainingError,
    ) as error:
        raise RelationalCommutatorAUHError(str(error)) from error
    immutable = _immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=dataset_summary,
        router=router,
        eligible_routes=eligible_routes,
        target_modules=target_modules,
        checkpoint=checkpoint,
        loss_config=loss_config,
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.to(device)
    model.eval()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    try:
        named_trainable = legacy.trainable_lora_parameters(model)
        initialization_digest = legacy.synchronize_trainable_parameters(
            named_trainable, source_rank=0
        )
    except legacy.TrainingContractError as error:
        raise RelationalCommutatorAUHError(str(error)) from error
    parameter_names = v4._optimizer_parameter_names(named_trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    scheduler_kwargs = legacy.noise_scheduler_kwargs()
    scheduler_kwargs["noise_tmin"] = MINIMUM_TRAINING_SIGMA
    scheduler = NoiseScheduler(**scheduler_kwargs)
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=sigma_strata.FLOW_SHIFT,
    )
    sigma_strata.audit_runtime_unipc_schedule(inference_scheduler)

    global_step = 0
    last_saved = -1
    last_metrics: Optional[dict[str, float]] = None
    step_audit: list[dict[str, Any]] = []

    def save_current() -> None:
        optimizer_payload = _optimizer_payload(
            optimizer=optimizer,
            global_step=global_step,
            immutable=immutable,
            parameter_names=parameter_names,
            step_audit=step_audit,
        )
        receipt = _build_receipt(
            args=args,
            global_step=global_step,
            metrics=last_metrics,
            step_audit=step_audit,
            dataset=dataset,
            dataset_summary=dataset_summary,
            router=router,
            checkpoint=checkpoint,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            distributed=distributed,
            backend=backend,
            target_modules=target_modules,
            named_trainable=named_trainable,
            initialization_digest=initialization_digest,
            transformers_version=transformers_version,
            immutable=immutable,
            optimizer_payload=optimizer_payload,
        )
        v5._save_checkpoint(
            model=model,
            optimizer_payload=optimizer_payload,
            output=output,
            global_step=global_step,
            receipt=receipt,
            rank=distributed.rank,
        )

    while global_step < args.max_steps:
        selected_stratum = sigma_strata.select_sigma_stratum(global_step)
        row_index, raw_row, route = v4._next_routed_row(
            dataset, eligible_routes, ordinal=global_step
        )
        identity = legacy.dataset_identity(raw_row, row_index)
        legacy.assert_identical_row(identity)
        current_seed = legacy.step_seed(args.seed, global_step, row_index)
        legacy.seed_same_sample(current_seed)
        prepared = v6_runtime._prepare_candidate_cpu(
            raw_row=raw_row,
            tokenizer=tokenizer,
            prompt_cleaner=prompt_clean,
            system_prompts=SYSTEM_PROMPTS,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            noop_instruction=args.noop_instruction,
            negative_prompt=args.negative_prompt,
            process_renderer_sample=process_renderer_sample,
            selected_stratum=selected_stratum,
        )
        moved = v6_runtime._move_candidate_to_device(prepared, device=device)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            cell = _run_seven_forward_cell(
                renderer=renderer,
                adapter_controller=model,
                candidate=moved,
                step_index=selected_stratum.schedule_index,
                loss_config=loss_config,
            )
        finite = bool(torch.isfinite(cell.weighted_loss.detach()).item())
        if not legacy._distributed_boolean(finite, op="all"):
            raise RelationalCommutatorAUHError("non-finite v7 loss blocked update")
        cell.weighted_loss.backward()
        try:
            gradient_norm = legacy.all_reduce_lora_gradients(named_trainable)
        except legacy.TrainingContractError as error:
            raise RelationalCommutatorAUHError(str(error)) from error
        if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0.0:
            raise RelationalCommutatorAUHError(
                "v7 requires a finite positive preclip LoRA gradient norm"
            )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], args.max_grad_norm
        )
        optimizer.step()
        metrics = {
            **_loss_metrics(cell),
            "preclip_gradient_norm": float(gradient_norm),
            "sigma_schedule_index": float(selected_stratum.schedule_index),
            "sigma_timestep": float(selected_stratum.timestep),
        }
        record: dict[str, Any] = {
            "optimizer_step": global_step + 1,
            "row_index": row_index,
            "iid": route.iid,
            "seed": current_seed,
            "sigma_schedule_index": selected_stratum.schedule_index,
            "sigma_timestep": selected_stratum.timestep,
            "teacher_mode": args.teacher_mode,
            "metrics_timing": METRICS_TIMING,
            **metrics,
        }
        try:
            v6_runtime._assert_gate_record_equal_across_ranks(record)
        except v6_runtime.CMSGauhTrainingError as error:
            raise RelationalCommutatorAUHError(
                "complete v7 step metrics differ across ranks"
            ) from error
        step_audit.append(record)
        global_step += 1
        last_metrics = metrics
        if distributed.rank == 0:
            print(json.dumps({"event": "optimizer_step", **record}, sort_keys=True), flush=True)
        if args.save_every > 0 and global_step % args.save_every == 0:
            save_current()
            last_saved = global_step

    if last_saved != global_step:
        save_current()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


__all__ = [
    "ARTIFACT_VALIDATION_SCHEMA",
    "FORWARD_CELL_ORDER",
    "INFERENCE_FORWARD_ORDER",
    "INFERENCE_FINALIZER_MODULE",
    "INFERENCE_LOADER_MODULE",
    "INFERENCE_PARITY_TESTS",
    "INFERENCE_RECEIPT_SCHEMA",
    "INFERENCE_RUNNER_MODULE",
    "MAIN_COMMUTATOR_CONFIG",
    "METRICS_TIMING",
    "METHOD_NAME",
    "OPTIMIZER_SCHEMA",
    "RECEIPT_SCHEMA",
    "RelationalCommutatorAUHError",
    "SevenForwardCellResult",
    "build_parser",
    "loss_config_from_args",
    "main",
    "validate_cli",
    "_run_seven_forward_cell",
]


if __name__ == "__main__":
    raise SystemExit(main())
