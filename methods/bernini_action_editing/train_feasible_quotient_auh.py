#!/usr/bin/env python3
"""Exact-40 AUH pilot trainer for Bernini RS-FQT LoRA v8.

The heavy Bernini/data/distributed plumbing is the already audited v7 AUH
trainer.  This module installs a narrow strategy surface before entering that
loop: five editor forwards instead of seven, the projection-consistent
feasible-quotient objective, a distinct immutable receipt schema, and a lower
fixed pilot learning rate.  It does not load or execute the T2V generator.

This is deliberately an exact-40 falsification pilot.  A longer 320-step run
requires a separately audited warmup/cosine and resume contract; this file
does not pretend that fixed-LR pilot is the final training recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import feasible_quotient_objective as objective  # noqa: E402
import motion_commutator as commutator  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_relational_motion_commutator_auh as v7  # noqa: E402


legacy = v7.legacy
sigma_strata = v7.sigma_strata
v4 = v7.v4
v5 = v7.v5
v6_runtime = v7.v6_runtime
v6_scope = v7.v6_scope

METHOD_NAME = "bernini-reconstruction-section-feasible-quotient-lora-v8-auh"
RECEIPT_SCHEMA = "bernini-rs-fqt-auh-training-receipt-v8"
OPTIMIZER_SCHEMA = "bernini-rs-fqt-auh-optimizer-v8"
ARTIFACT_VALIDATION_SCHEMA = "bernini-rs-fqt-artifact-validation-v1"
INFERENCE_RECEIPT_SCHEMA = "bernini-rs-fqt-inference-receipt-v8"
INFERENCE_LOADER_MODULE = "infer_feasible_quotient_lora.py"
INFERENCE_RUNNER_MODULE = "run_relational_motion_commutator_inference.py"
INFERENCE_FINALIZER_MODULE = "finalize_feasible_quotient_checkpoint.py"
INFERENCE_PARITY_TESTS = (
    "test_feasible_quotient_objective.py",
    "test_train_feasible_quotient_auh.py",
    "test_feasible_quotient_sbatch_contract.py",
    "test_train_prior_tangent_lora_contract.py",
    "test_gauge_anchored_commutator.py",
    "test_infer_feasible_quotient_lora.py",
    "test_finalize_feasible_quotient_checkpoint.py",
    "test_infer_relational_motion_commutator.py",
    "test_run_relational_motion_commutator_inference_contract.py",
)
LEARNING_RATE = 1.0e-5
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23
MAX_STEPS = 40
SAVE_EVERY = 40
ZERO_RELEASE_START_INDEX = 31
METRICS_TIMING = "pre_optimizer_update"
FORWARD_CELL_ORDER = objective.FORWARD_BRANCH_ORDER
INFERENCE_FORWARD_ORDER = v7.INFERENCE_FORWARD_ORDER

_V7_BUILD_PARSER = v7.build_parser
_V7_VALIDATE_CLI = v7.validate_cli
_V7_BUILD_RECEIPT = v7._build_receipt
_V7_OPTIMIZER_PAYLOAD = v7._optimizer_payload


@dataclass(frozen=True)
class PreparedFiveBranchCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str


@dataclass(frozen=True)
class MovedFiveBranchCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str


class FeasibleQuotientAUHError(RuntimeError):
    """Raised before an invalid v8 pilot can update the adapter."""


def build_parser() -> argparse.ArgumentParser:
    parser = _V7_BUILD_PARSER()
    parser.description = (
        "Train exact-40 81f Bernini RS-FQT LoRA v8 pilot on four AUH GPUs"
    )
    actions = {action.dest: action for action in parser._actions}
    actions["learning_rate"].default = LEARNING_RATE
    actions["max_steps"].default = MAX_STEPS
    actions["save_every"].default = SAVE_EVERY
    actions["teacher_mode"].choices = ("paired_displacement_only",)
    actions["teacher_mode"].default = "paired_displacement_only"
    return parser


def loss_config_from_args(
    args: argparse.Namespace,
) -> objective.FeasibleQuotientLossConfig:
    if float(args.relational_auxiliary_weight) != 0.0:
        raise FeasibleQuotientAUHError(
            "RS-FQT has no T2V relational auxiliary"
        )
    return objective.FeasibleQuotientLossConfig()


def validate_cli(args: argparse.Namespace) -> None:
    # The v7 validator remains authoritative for model/data/source/exact46
    # geometry.  Make standalone validation deterministic as well as the
    # installed training path, then restore globals for import-side hygiene.
    if args.teacher_mode != "paired_displacement_only":
        raise FeasibleQuotientAUHError(
            "v8 supervision mode must be paired_displacement_only"
        )
    previous_learning_rate = v7.LEARNING_RATE
    previous_loss_builder = v7.loss_config_from_args
    v8_teacher_mode = args.teacher_mode
    try:
        v7.LEARNING_RATE = LEARNING_RATE
        v7.loss_config_from_args = loss_config_from_args
        # The inherited plumbing calls this legacy switch ``teacher_mode``.
        # V8 has no generator teacher; translate only while invoking the V7
        # geometry validator, then restore the scientifically accurate label
        # used by every V8 step-audit record.
        args.teacher_mode = "target_only"
        _V7_VALIDATE_CLI(args)
    finally:
        args.teacher_mode = v8_teacher_mode
        v7.LEARNING_RATE = previous_learning_rate
        v7.loss_config_from_args = previous_loss_builder
    if (
        args.teacher_mode != "paired_displacement_only"
        or float(args.relational_auxiliary_weight) != 0.0
        or int(args.max_steps) != MAX_STEPS
        or int(args.save_every) != SAVE_EVERY
        or float(args.learning_rate) != LEARNING_RATE
        or float(args.weight_decay) != 0.0
    ):
        raise FeasibleQuotientAUHError(
            "v8 pilot fixes paired displacement, exact40/save40, lr=1e-5, and zero weight decay"
        )
    try:
        loss_config_from_args(args).validate()
    except objective.FeasibleQuotientObjectiveError as error:
        raise FeasibleQuotientAUHError(str(error)) from error


def _prepare_target_state_candidate_cpu(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    prompt_cleaner: Any,
    system_prompts: Mapping[str, str],
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    negative_prompt: str,
    process_renderer_sample: Any,
    selected_stratum: Any,
) -> PreparedFiveBranchCandidate:
    """Build the standard noised-target query without a generator branch.

    The paired target constructs a training diffusion state only.  It is not
    an external model condition and remains absent from inference.
    """

    try:
        sample = legacy.sanitize_preprocessed_row(raw_row)
        spatial_hw = v6_runtime._spatial_hw_from_sample(sample, z_dim=z_dim)
        _, instruction, _ = v6_runtime._official_t2v_text_fields(
            sample,
            tokenizer=tokenizer,
            prompt_cleaner=prompt_cleaner,
            system_prompts=system_prompts,
        )
        endpoints = v5._prepare_prior_bridge_batches(
            raw_row=raw_row,
            tokenizer=tokenizer,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            noop_instruction=noop_instruction,
            negative_prompt=negative_prompt,
            minimum_training_sigma=v7.MINIMUM_TRAINING_SIGMA,
            process_renderer_sample=process_renderer_sample,
            selected_stratum=selected_stratum,
        )
    except (
        legacy.TrainingContractError,
        motion.MotionContractError,
        sigma_strata.InferenceSigmaStrataError,
        v4.DeltaTrainingError,
        v5.PriorTangentTrainingError,
        v6_runtime.CMSGauhTrainingError,
    ) as error:
        raise FeasibleQuotientAUHError(str(error)) from error
    editor_negative, editor_noop, editor_action, auxiliary = endpoints["target"]
    if float(auxiliary.get("bridge_fraction", -1.0)) != 1.0:
        raise FeasibleQuotientAUHError(
            "v8 target-state query must use exact beta=1"
        )
    negative_text = {
        field: editor_negative[field]
        for field in v6_runtime.branch_geometry.TEXT_FIELDS
    }
    editor_negative = v6_runtime._bind_text_geometry(
        editor_negative, negative_text, label="full-source negative"
    )
    return PreparedFiveBranchCandidate(
        editor_negative=editor_negative,
        editor_noop=editor_noop,
        editor_action=editor_action,
        auxiliary=auxiliary,
        spatial_hw=spatial_hw,
        instruction_sha256=hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
    )


def _move_target_state_candidate_to_device(
    candidate: PreparedFiveBranchCandidate, *, device: Any
) -> MovedFiveBranchCandidate:
    import torch

    try:
        editor_negative = legacy._move_batch(candidate.editor_negative, device)
        editor_noop = legacy._move_batch(candidate.editor_noop, device)
        editor_action = legacy._move_batch(candidate.editor_action, device)
        auxiliary = v4._move_auxiliary_to_device(
            candidate.auxiliary,
            device=device,
            branch_state_mode="source_target_bridge_clean_field",
        )
        v5._assert_same_endpoint_state(
            editor_negative, editor_noop, editor_action
        )
    except (
        legacy.TrainingContractError,
        v4.DeltaTrainingError,
        v5.PriorTangentTrainingError,
    ) as error:
        raise FeasibleQuotientAUHError(str(error)) from error
    selector = editor_action["vae_latents_mask"].squeeze(0).bool()
    target_tokens = int(selector.sum().item())
    shared_noisy = auxiliary.get("shared_noisy")
    if (
        target_tokens <= 0
        or int((~selector).sum().item()) != target_tokens
        or target_tokens != v7.LATENT_PHASES * math.prod(candidate.spatial_hw)
        or not isinstance(shared_noisy, torch.Tensor)
        or shared_noisy.dtype != torch.float32
        or tuple(shared_noisy.shape[:2]) != (1, target_tokens)
        or float(auxiliary.get("bridge_fraction", -1.0)) != 1.0
    ):
        raise FeasibleQuotientAUHError(
            "v8 target-state candidate geometry or beta differs"
        )
    packed_tail = motion.flatten_velocity_patches(
        editor_action["input_vae_latents"][selector].unsqueeze(0)
    ).float()
    if not torch.equal(packed_tail, shared_noisy):
        raise FeasibleQuotientAUHError(
            "editor target tail differs from exact noised-target query"
        )
    return MovedFiveBranchCandidate(
        editor_negative=editor_negative,
        editor_noop=editor_noop,
        editor_action=editor_action,
        auxiliary=auxiliary,
        spatial_hw=candidate.spatial_hw,
        instruction_sha256=candidate.instruction_sha256,
    )


def _run_five_forward_cell(
    *,
    renderer: Any,
    adapter_controller: Any,
    candidate: Any,
    step_index: int,
    loss_config: objective.FeasibleQuotientLossConfig,
) -> v7.SevenForwardCellResult:
    """Execute three frozen and two graph-bearing editor forwards."""

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
    adapted_noop_v = motion.renderer_velocity_prediction(
        renderer, candidate.editor_noop
    )
    adapted_action_v = motion.renderer_velocity_prediction(
        renderer, candidate.editor_action
    )
    velocities = (
        frozen_negative_v,
        frozen_noop_v,
        frozen_action_v,
        adapted_noop_v,
        adapted_action_v,
    )
    if any(
        value.dtype != torch.bfloat16
        or tuple(value.shape) != tuple(shared_noisy.shape)
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise FeasibleQuotientAUHError(
            "all five forwards must be finite native-BF16 fields on one query"
        )
    if any(value.requires_grad for value in velocities[:3]):
        raise FeasibleQuotientAUHError("a frozen editor branch retained a graph")
    if not adapted_noop_v.requires_grad or not adapted_action_v.requires_grad:
        raise FeasibleQuotientAUHError(
            "both adapted editor branches must retain LoRA graphs"
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
            raise FeasibleQuotientAUHError(
                "five APG branches reconstructed different negative fields"
            )
        fields = objective.FiveBranchCleanFields(
            frozen_editor_noop=frozen_noop.detach(),
            frozen_editor_action=frozen_action.detach(),
            adapted_editor_noop=adapted_noop.float(),
            adapted_editor_action=adapted_action.float(),
            source_clean=v5._as_phase_grid(
                candidate.auxiliary["source_clean"].float()
            ).detach(),
            target_clean=v5._as_phase_grid(
                candidate.auxiliary["target_clean"].float()
            ).detach(),
        )
        result = objective.compute_feasible_quotient_objective(
            fields,
            step_index=step_index,
            config=loss_config,
        )
        inverse_sigma = motion.clean_field_inverse_sigma_weight(
            sigma,
            weight_floor=sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        ).mean()
    except (
        objective.FeasibleQuotientObjectiveError,
        v5.PriorTangentTrainingError,
        motion.MotionContractError,
    ) as error:
        raise FeasibleQuotientAUHError(str(error)) from error
    weighted = inverse_sigma * result.total
    if not bool(torch.isfinite(weighted)):
        raise FeasibleQuotientAUHError("weighted v8 loss is non-finite")
    return v7.SevenForwardCellResult(
        weighted_loss=weighted,
        loss_result=result,
        inverse_sigma_weight=inverse_sigma,
    )


def _scalar(value: Any) -> float:
    return float(value.detach().float().mean().cpu().item())


def _loss_metrics(cell: v7.SevenForwardCellResult) -> dict[str, float]:
    result = cell.loss_result
    detached = objective.detached_receipt_diagnostics(result)
    return {
        "loss_total": _scalar(result.total),
        "loss_weighted": _scalar(cell.weighted_loss),
        "loss_canonical": _scalar(result.canonical),
        "loss_executed": _scalar(result.executed),
        "loss_noop_preservation": _scalar(result.noop_preservation),
        "loss_margin": _scalar(result.margin),
        "loss_temporal_jitter": _scalar(result.temporal_jitter),
        "inverse_sigma_weight": _scalar(cell.inverse_sigma_weight),
        "rho": float(result.rho),
        "source_only_radius_mean_active": float(
            detached["source_only_radius_mean_active"]
        ),
        "radius_floor_dominated_fraction_active": float(
            detached["radius_floor_dominated_fraction_active"]
        ),
        "target_clipped_fraction_active": float(
            detached["target_clipped_fraction_active"]
        ),
        "target_projection_scale_mean_active": float(
            detached["target_projection_scale_mean_active"]
        ),
        "target_energy_retention_mean_active": float(
            detached["target_energy_retention_mean_active"]
        ),
        "target_required_radius_multiplier_p50": float(
            detached["target_required_radius_multiplier_p50"]
        ),
        "target_required_radius_multiplier_p90": float(
            detached["target_required_radius_multiplier_p90"]
        ),
        "target_required_radius_multiplier_max": float(
            detached["target_required_radius_multiplier_max"]
        ),
        "predicted_saturated_fraction_active": float(
            detached["predicted_saturated_fraction_active"]
        ),
        "predicted_projection_scale_mean_active": float(
            detached["predicted_projection_scale_mean_active"]
        ),
        "frozen_noop_to_source_rms": float(
            detached["frozen_noop_to_source_rms"]
        ),
        "frozen_noop_to_target_rms": float(
            detached["frozen_noop_to_target_rms"]
        ),
        "frozen_noop_target_over_source_error_ratio": float(
            detached["frozen_noop_target_over_source_error_ratio"]
        ),
        "target_frozen_prior_cosine_mean_active": float(
            detached["target_frozen_prior_cosine_mean_active"]
        ),
        "target_frozen_prior_cosine_p10_active": float(
            detached["target_frozen_prior_cosine_p10_active"]
        ),
        "target_frozen_prior_positive_cosine_fraction_active": float(
            detached["target_frozen_prior_positive_cosine_fraction_active"]
        ),
        "target_high_frequency_fraction_mean_active": float(
            detached["target_high_frequency_fraction_mean_active"]
        ),
        "target_inside_deployment_radius": float(
            bool(detached["target_inside_deployment_radius"])
        ),
    }


def _immutable_contract(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    target_modules: Sequence[str],
    checkpoint: Path,
    loss_config: objective.FeasibleQuotientLossConfig,
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
        "checkpoint_content_manifest_sha256": (
            CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": CHECKPOINT_CONTENT_FILE_COUNT,
        "checkpoint_content_validation": (
            "pinned_sha256sum_manifest_before_torchrun_and_finalizer"
        ),
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
        "frames": v7.NUM_FRAMES,
        "latent_phases": v7.LATENT_PHASES,
        "seed": int(args.seed),
        "learning_rate": LEARNING_RATE,
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "teacher_mode": "paired_displacement_only",
        "loss_config": asdict(loss_config),
        "objective_contract": objective.immutable_objective_contract(
            loss_config
        ),
        "lora": {
            "scope": v6_scope.LORA_SCOPE,
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(
                list(target_modules)
            ),
        },
        "training_diffusion_query": "target(beta=1)",
        "training_diffusion_query_formula": (
            "x_sigma=(1-sigma)*paired_target+sigma*epsilon"
        ),
        "paired_target_constructs_training_diffusion_state": True,
        "paired_target_used_as_external_model_condition": False,
        "forward_cell_order": list(FORWARD_CELL_ORDER),
        "forwards_per_candidate": 5,
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
            "clean_reconstruction": (
                "fp32_noisy_minus_sigma_times_native_bf16_velocity"
            ),
            "v8_frozen_action_section": (
                "local_fp32_apg_after_bit_exact_native_bf16_official_parity"
            ),
        },
        "target_motion_teacher": (
            "Q0(target_clean-stopgrad(frozen_noop_section))"
        ),
        "target_section_reference": (
            "same_query_frozen_noop_prevents_beta1_double_count"
        ),
        "target_used_as_model_condition": False,
        "training_operator": (
            "FIR(DQ0(Atheta-Ntheta)) with feasible canonical target"
        ),
        "deployment_operator": (
            "N0+rho*Integrate(Project_r(FIR(DQ0(Atheta-Ntheta))))"
        ),
        "deployment_velocity_precision": (
            "fp32_exact_clean_transport_with_post_boundary_radius_certificate"
        ),
        "appearance_carrier": "frozen_noop_reconstruction_section",
        "source_only_radius": (
            "max(frozen_quotient_rms,0.25*noop_dynamics_rms,1e-3)"
        ),
        "metrics_timing": METRICS_TIMING,
        "release_schedule": release,
        "release_schedule_sha256": legacy.object_sha256(release),
        "zero_release_steps": [
            index for index, rho in enumerate(release) if rho == 0.0
        ],
        "zero_release_semantics": (
            "adam_moments_reset_before_suffix_then_current_noop_gradient_only"
        ),
        "zero_release_optimizer_boundary": {
            "first_zero_release_schedule_index": ZERO_RELEASE_START_INDEX,
            "reset_before_optimizer_step": ZERO_RELEASE_START_INDEX + 1,
            "reset_state": ["step", "exp_avg", "exp_avg_sq"],
            "reset_count": 1,
            "weight_decay": 0.0,
        },
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["paired_target_video"],
        "forbidden_inference_conditions": list(
            objective.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "training_generator_forwards": 0,
        "inference_generator_forwards": 0,
        "pilot_scope": "exact40_fixed_lr_falsification",
        "inference_loader_parity": {
            "verified": True,
            "verification_stage": (
                "immutable_v8_loader_finalizer_and_runner_preflight_before_model_load"
            ),
            "loader_module": INFERENCE_LOADER_MODULE,
            "runner_module": INFERENCE_RUNNER_MODULE,
            "finalizer_module": INFERENCE_FINALIZER_MODULE,
            "training_receipt_schema": RECEIPT_SCHEMA,
            "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
            "contract_tests": list(INFERENCE_PARITY_TESTS),
            "source_revision_and_archive_bound": True,
            "strict_loader_rejects_pending_canary_and_incomplete_cycle": True,
            "post_save_v8_loader_finalization_required": True,
        },
        "resume_integrated": False,
    }
    if len(target_modules) != 46:
        raise FeasibleQuotientAUHError(
            f"v8 LoRA scope resolved {len(target_modules)} modules, expected 46"
        )
    return {"value": value, "digest": legacy.object_sha256(value)}


def _build_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _V7_BUILD_RECEIPT(**kwargs)
    receipt.pop("receipt_digest", None)
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["method"] = METHOD_NAME
    receipt["training_only_generator_and_target"] = False
    receipt["training_only_paired_target"] = True
    receipt["training_generator_forwards"] = 0
    receipt["inference_generator_forwards"] = 0
    receipt["teacher_mode"] = "paired_displacement_only"
    receipt["pilot_scope"] = "exact40_fixed_lr_falsification"
    reset = kwargs["optimizer_payload"].get("zero_release_moment_reset")
    if not isinstance(reset, Mapping):
        raise FeasibleQuotientAUHError(
            "v8 optimizer payload lacks the zero-release moment reset proof"
        )
    receipt["optimizer"]["zero_release_moment_reset"] = dict(reset)
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _optimizer_payload(**kwargs: Any) -> dict[str, Any]:
    optimizer = kwargs["optimizer"]
    payload = _V7_OPTIMIZER_PAYLOAD(**kwargs)
    state_steps = sorted(
        {
            int(
                state["step"].detach().cpu().item()
                if hasattr(state.get("step"), "detach")
                else state.get("step", -1)
            )
            for state in optimizer.state.values()
        }
    )
    reset = {
        "first_zero_release_schedule_index": ZERO_RELEASE_START_INDEX,
        "reset_before_optimizer_step": ZERO_RELEASE_START_INDEX + 1,
        "completed_optimizer_steps": int(
            getattr(optimizer, "_v8_completed_optimizer_steps", -1)
        ),
        "reset_count": int(getattr(optimizer, "_v8_moment_reset_count", -1)),
        "state_step_after_reset_suffix": MAX_STEPS - ZERO_RELEASE_START_INDEX,
        "state_step_values": state_steps,
        "state_parameter_count": len(optimizer.state),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
    }
    if reset != {
        "first_zero_release_schedule_index": 31,
        "reset_before_optimizer_step": 32,
        "completed_optimizer_steps": 40,
        "reset_count": 1,
        "state_step_after_reset_suffix": 9,
        "state_step_values": [9],
        "state_parameter_count": len(kwargs["parameter_names"]),
        "weight_decay": 0.0,
    }:
        raise FeasibleQuotientAUHError(
            "v8 zero-release optimizer boundary was not executed exactly once"
        )
    payload["zero_release_moment_reset"] = reset
    return payload


def _build_zero_release_reset_adamw(base_class: Any) -> Any:
    """Return AdamW with one deterministic moment reset before step 32."""

    class ZeroReleaseResetAdamW(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if any(float(group.get("weight_decay", math.nan)) != 0.0 for group in self.param_groups):
                raise FeasibleQuotientAUHError(
                    "v8 zero-release optimizer requires zero weight decay"
                )
            self._v8_completed_optimizer_steps = 0
            self._v8_moment_reset_count = 0

        def step(self, closure: Any = None) -> Any:
            if self._v8_completed_optimizer_steps == ZERO_RELEASE_START_INDEX:
                if not self.state:
                    raise FeasibleQuotientAUHError(
                        "AdamW has no motion-bearing moments to reset"
                    )
                self.state.clear()
                self._v8_moment_reset_count += 1
            result = super().step(closure=closure)
            self._v8_completed_optimizer_steps += 1
            return result

    ZeroReleaseResetAdamW.__name__ = "ZeroReleaseResetAdamW"
    return ZeroReleaseResetAdamW


def _install_strategy() -> None:
    v7.METHOD_NAME = METHOD_NAME
    v7.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    v7.OPTIMIZER_SCHEMA = OPTIMIZER_SCHEMA
    v7.ARTIFACT_VALIDATION_SCHEMA = ARTIFACT_VALIDATION_SCHEMA
    v7.INFERENCE_RECEIPT_SCHEMA = INFERENCE_RECEIPT_SCHEMA
    v7.INFERENCE_LOADER_MODULE = INFERENCE_LOADER_MODULE
    v7.INFERENCE_RUNNER_MODULE = INFERENCE_RUNNER_MODULE
    v7.INFERENCE_FINALIZER_MODULE = INFERENCE_FINALIZER_MODULE
    v7.INFERENCE_PARITY_TESTS = INFERENCE_PARITY_TESTS
    v7.LEARNING_RATE = LEARNING_RATE
    v7.METRICS_TIMING = METRICS_TIMING
    v7.FORWARD_CELL_ORDER = FORWARD_CELL_ORDER
    v7.build_parser = build_parser
    v7.loss_config_from_args = loss_config_from_args
    v7.validate_cli = validate_cli
    v7._run_seven_forward_cell = _run_five_forward_cell
    v7._loss_metrics = _loss_metrics
    v7._immutable_contract = _immutable_contract
    v7._build_receipt = _build_receipt
    v7._optimizer_payload = _optimizer_payload
    v6_runtime._prepare_candidate_cpu = _prepare_target_state_candidate_cpu
    v6_runtime._move_candidate_to_device = (
        _move_target_state_candidate_to_device
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_strategy()
    import torch

    original_adamw = torch.optim.AdamW
    torch.optim.AdamW = _build_zero_release_reset_adamw(original_adamw)
    try:
        return v7.main(argv)
    except v7.RelationalCommutatorAUHError as error:
        raise FeasibleQuotientAUHError(str(error).replace("v7", "v8")) from error
    finally:
        torch.optim.AdamW = original_adamw


__all__ = [
    "ARTIFACT_VALIDATION_SCHEMA",
    "CHECKPOINT_CONTENT_FILE_COUNT",
    "CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "FORWARD_CELL_ORDER",
    "INFERENCE_PARITY_TESTS",
    "LEARNING_RATE",
    "MAX_STEPS",
    "METHOD_NAME",
    "OPTIMIZER_SCHEMA",
    "RECEIPT_SCHEMA",
    "SAVE_EVERY",
    "ZERO_RELEASE_START_INDEX",
    "FeasibleQuotientAUHError",
    "build_parser",
    "loss_config_from_args",
    "main",
    "validate_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())
