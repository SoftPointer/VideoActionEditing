#!/usr/bin/env python3
"""Train the PDF-v2 diagnostic LoRA without changing the audited CDF-v1 code."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import projected_differential_flow as pdf  # noqa: E402
import train_delta_lora as base  # noqa: E402


METHOD_NAME = "bernini-r-1p3b-projected-differential-flow-lora-v2-diagnostic"
RECEIPT_SCHEMA = "bernini-r-1p3b-pdf-lora-training-receipt-v2"

_base_build_parser = base.build_parser
_base_validate_cli = base.validate_cli
_base_immutable_contract = base._immutable_contract


def build_parser() -> argparse.ArgumentParser:
    parser = _base_build_parser()
    parser.description = "Train 81f Bernini Projected Differential Flow diagnostic LoRA"
    parser.set_defaults(
        # Freeze every inherited v2 diagnostic choice explicitly.  The base
        # trainer now defaults to the formal v4 projected bridge recipe; this
        # historical control requires its original two-path velocity batches
        # (including ``source_velocity``) and must never masquerade as v4.
        unreviewed_tier="motion_only",
        learning_rate=5.0e-5,
        lora_scope="cross_q_out",
        branch_state_mode="separate_clean_paths",
        inverse_sigma_weight_floor=0.25,
        copy_loss_weight=0.5,
        boundary_gauge_loss_weight=0.0,
        motion_objective="quotient_multilag",
        bridge_consistency_weight=0.0,
        high_noise_floor=0.25,
    )
    parser.add_argument("--dc-loss-weight", type=float, default=0.25)
    parser.add_argument("--integration-steps", type=int, default=40)
    parser.add_argument("--integration-flow-shift", type=float, default=5.0)
    parser.add_argument("--interval-weight-power", type=float, default=1.0)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    _base_validate_cli(args)
    if args.branch_state_mode != "separate_clean_paths":
        raise base.DeltaTrainingError(
            "PDF-v2 requires its historical separate-clean-path velocity batches"
        )
    if float(args.anchor_loss_weight) != 0.0:
        raise base.DeltaTrainingError("PDF-v2 forbids first-frame anchor loss")
    if args.motion_objective != "quotient_multilag":
        raise base.DeltaTrainingError(
            "PDF-v2 has one fixed projected objective; use quotient_multilag"
        )
    if type(args.integration_steps) is not int or args.integration_steps <= 0:
        raise base.DeltaTrainingError("integration_steps must be positive")
    for name in ("dc_loss_weight", "integration_flow_shift", "interval_weight_power"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise base.DeltaTrainingError(f"{name} must be finite and positive")


def _losses(
    *,
    renderer: Any,
    action_batch: Mapping[str, Any],
    copy_batch: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    route: motion.Route,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    """Match the field that PDF inference actually integrates.

    Target video supplies training supervision only.  It is never an inference
    condition.  Full-target framewise FM and frame anchors are deliberately
    absent from this diagnostic arm.
    """

    import torch

    action_prediction = motion.renderer_velocity_prediction(renderer, action_batch).float()
    noop_prediction = motion.renderer_velocity_prediction(renderer, copy_batch).float()
    action_target = base._flatten_target(action_batch["target_velocity"]).float()
    source_target = base._flatten_target(auxiliary["source_velocity"]).float()
    shapes = {
        tuple(x.shape)
        for x in (action_prediction, noop_prediction, action_target, source_target)
    }
    if len(shapes) != 1:
        raise base.DeltaTrainingError("PDF action/no-op/target field shapes differ")

    raw_prediction = action_prediction - noop_prediction
    raw_target = action_target - source_target
    projected_prediction = pdf.project_temporal_dc(
        raw_prediction, latent_frames=base.legacy.LATENT_FRAMES
    )
    projected_target = pdf.project_temporal_dc(
        raw_target, latent_frames=base.legacy.LATENT_FRAMES
    )
    projected_loss = torch.mean((projected_prediction - projected_target) ** 2)
    multilag_loss = motion.multiscale_temporal_difference_loss(
        projected_prediction,
        projected_target,
        latent_frames=base.legacy.LATENT_FRAMES,
        lags=tuple(args.temporal_lags),
    )
    # Explicitly learn a zero action/no-op temporal DC; inference additionally
    # enforces the same constraint as an exact hard projection.
    dc = pdf.temporal_dc(raw_prediction, latent_frames=base.legacy.LATENT_FRAMES)
    dc_loss = torch.mean(dc ** 2)
    motion_loss = args.quotient_weight * projected_loss + (
        1.0 - args.quotient_weight
    ) * multilag_loss
    interval_weight = pdf.integration_interval_weight(
        auxiliary["sigma"],
        num_steps=args.integration_steps,
        flow_shift=args.integration_flow_shift,
        power=args.interval_weight_power,
    ).mean()

    copy_target = base._flatten_target(copy_batch["target_velocity"]).float()
    copy_loss = torch.mean((noop_prediction - copy_target) ** 2)
    total = (
        args.motion_loss_weight * interval_weight * motion_loss
        + args.dc_loss_weight * dc_loss
        + args.copy_loss_weight * copy_loss
    )
    zero = torch.zeros((), device=total.device, dtype=torch.float32)
    return total, {
        "motion": motion_loss,
        "motion_projected": projected_loss,
        "motion_multiscale": multilag_loss,
        "temporal_dc": dc_loss,
        "integration_interval_weight": interval_weight,
        "copy": copy_loss,
        "full_target": zero,
        "full_target_weight": zero,
        "anchor": zero,
        "sigma": auxiliary["sigma"].float().mean(),
    }


def _immutable_contract(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    result = _base_immutable_contract(**kwargs)
    value = dict(result["value"])
    value.update(
        {
            "method": METHOD_NAME,
            "motion_objective": "hard_projected_delta_plus_multilag",
            "motion_representation": "temporal-dc-zero-action-noop-velocity-v2",
            "train_inference_projection_identical": True,
            "temporal_dc_constraint": "hard_at_inference_plus_training_penalty",
            "dc_loss_weight": float(args.dc_loss_weight),
            "sigma_weighting": "nearest_shifted_inference_interval_width_mean_one",
            "integration_steps": int(args.integration_steps),
            "integration_flow_shift": float(args.integration_flow_shift),
            "interval_weight_power": float(args.interval_weight_power),
            "full_target_framewise_loss": False,
            "first_frame_anchor": False,
            "method_status": "diagnostic_bridge_not_final_method",
        }
    )
    return {"value": value, "digest": base.legacy.object_sha256(value)}


def _supervision_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "target_used_for_training_supervision_only": True,
        "external_mask_track_pose_trajectory": False,
        "first_frame_anchor": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "counterfactual_noop_forward": True,
        "copy_calibration_enabled": float(args.copy_loss_weight) > 0.0,
        "copy_calibration_weight": float(args.copy_loss_weight),
        "shared_source_posterior_mode": True,
        "shared_sigma": True,
        "shared_diffusion_noise": True,
        "unreviewed_full_target_weight": 0.0,
        "full_target_framewise_loss_enabled": False,
        "motion_representation": "hard projected temporal-DC-zero action/noop velocity",
        "train_inference_projection_identical": True,
        "temporal_dc_training_penalty": float(args.dc_loss_weight),
        "temporal_lags": list(args.temporal_lags),
        "sigma_weighting": "shifted inference interval width",
        "optional_first_latent_anchor_weight": 0.0,
    }


def _install_v2_hooks() -> None:
    base.METHOD_NAME = METHOD_NAME
    base.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    base.build_parser = build_parser
    base.validate_cli = validate_cli
    base._losses = _losses
    base._immutable_contract = _immutable_contract
    base._supervision_receipt = _supervision_receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_v2_hooks()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
