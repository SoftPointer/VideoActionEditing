"""Reference-corrected flow DPO core for PAIR-v5.

This module owns only the mathematical preference update.  Both candidates
must already have been produced by Bernini's deployment-matched native RV2V
path and admitted by the separately sealed safe-Pareto selector.  A single
fresh Gaussian and one physical sigma are shared by the chosen/rejected pair.

There is deliberately no argument for a T2V proposal, donor, paired target,
mask, flow, pose, track, or trajectory.  Self-generated T2V videos may
calibrate the frozen action critic upstream, but cannot enter this loss as
pixels, latents, noise, or a visual condition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as functional


SCHEMA_VERSION = "bernini-pair-v5-reference-corrected-flow-dpo-v1"
FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16

FORBIDDEN_EXTERNAL_INPUT_NAMES = frozenset(
    {
        "proposal",
        "proposal_video",
        "proposal_latent",
        "proposal_noise",
        "donor",
        "donor_video",
        "donor_latent",
        "paired_target",
        "target_video",
        "target_latent",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)


class PairV5FlowDPOError(ValueError):
    """A PAIR-v5 shared-state preference packet is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5FlowDPOError("receipt value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def contract_receipt() -> Mapping[str, Any]:
    """Return the closed, digest-bound loss contract."""

    signature = set(inspect.signature(reference_corrected_flow_dpo).parameters)
    if not signature.isdisjoint(FORBIDDEN_EXTERNAL_INPUT_NAMES):
        raise PairV5FlowDPOError("public loss signature exposes a forbidden input")
    value = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "latent_channels": LATENT_CHANNELS,
        "candidate_origin": "native_rv2v_source_first_deployment_path_only",
        "shared_randomness": "one_fresh_epsilon_and_one_sigma_per_pair",
        "student_gap": "mse_student_rejected_minus_mse_student_chosen",
        "reference_gap": "mse_reference_rejected_minus_mse_reference_chosen",
        "advantage": "student_gap_minus_reference_gap",
        "loss": "weighted_mean_softplus_minus_beta_times_advantage",
        "reference_policy": "frozen_detached_bernini",
        "proposal_role": "critic_calibration_provenance_only",
        "proposal_visual_data_consumed": False,
        "paired_target_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
    }
    return {**value, "digest": _object_sha256(value)}


def _detached_fp32(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PairV5FlowDPOError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise PairV5FlowDPOError(f"{name} must be FP32")
    if value.device.type == "meta":
        raise PairV5FlowDPOError(f"{name} cannot be a meta tensor")
    if value.requires_grad or value.grad_fn is not None:
        raise PairV5FlowDPOError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all().item()):
        raise PairV5FlowDPOError(f"{name} contains NaN or infinity")
    return value


def _validate_clean_pair(
    chosen_clean: Any,
    rejected_clean: Any,
    epsilon: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = _detached_fp32("chosen_clean", chosen_clean)
    rejected = _detached_fp32("rejected_clean", rejected_clean)
    noise = _detached_fp32("epsilon", epsilon)
    if (
        chosen.ndim != 5
        or int(chosen.shape[0]) < 1
        or int(chosen.shape[1]) != LATENT_CHANNELS
        or int(chosen.shape[2]) != LATENT_PHASES
        or int(chosen.shape[3]) <= 0
        or int(chosen.shape[4]) <= 0
        or int(chosen.shape[3]) % 2
        or int(chosen.shape[4]) % 2
    ):
        raise PairV5FlowDPOError(
            "candidate latents must be exact81 [B,16,21,H,W] with even H/W"
        )
    expected = chosen.shape
    if rejected.shape != expected or noise.shape != expected:
        raise PairV5FlowDPOError(
            "chosen, rejected, and shared epsilon must have identical geometry"
        )
    if chosen.device != rejected.device or chosen.device != noise.device:
        raise PairV5FlowDPOError(
            "chosen, rejected, and shared epsilon must use one device"
        )
    if torch.equal(chosen, rejected):
        raise PairV5FlowDPOError("chosen and rejected candidates are tensor-identical")
    return chosen, rejected, noise


def _validate_sigma(sigma: Any, *, reference: torch.Tensor) -> torch.Tensor:
    value = _detached_fp32("sigma", sigma)
    batch = int(reference.shape[0])
    if value.ndim == 0:
        value = value.expand(batch)
    elif value.ndim != 1 or int(value.shape[0]) != batch:
        raise PairV5FlowDPOError("sigma must be one scalar or exact [B]")
    if value.device != reference.device:
        raise PairV5FlowDPOError("sigma and candidates must use one device")
    if bool(((value <= 0.0) | (value >= 1.0)).any().item()):
        raise PairV5FlowDPOError("PAIR-v5 DPO sigma must lie strictly in (0,1)")
    return value


def _prediction(
    name: str,
    value: Any,
    *,
    reference: torch.Tensor,
    trainable: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PairV5FlowDPOError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape or value.device != reference.device:
        raise PairV5FlowDPOError(f"{name} geometry/device differs from candidate")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise PairV5FlowDPOError(f"{name} has unsupported dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise PairV5FlowDPOError(f"{name} contains NaN or infinity")
    if trainable:
        if not value.requires_grad or value.grad_fn is None:
            raise PairV5FlowDPOError(f"{name} must remain connected to the student")
    elif value.requires_grad or value.grad_fn is not None:
        raise PairV5FlowDPOError(f"{name} must be a detached frozen-reference prediction")
    return value


def _positive_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise PairV5FlowDPOError(f"{name} must be a positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PairV5FlowDPOError(
            f"{name} must be a positive finite scalar"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise PairV5FlowDPOError(f"{name} must be a positive finite scalar")
    return result


def _weights(value: Any, *, reference: torch.Tensor) -> torch.Tensor:
    batch = int(reference.shape[0])
    if value is None:
        return torch.ones(batch, dtype=torch.float32, device=reference.device)
    result = _detached_fp32("sample_weight", value)
    if tuple(result.shape) != (batch,) or result.device != reference.device:
        raise PairV5FlowDPOError("sample_weight must be device-local exact [B]")
    if bool((result < 0.0).any().item()) or not bool((result > 0.0).any().item()):
        raise PairV5FlowDPOError("sample_weight must be nonnegative with positive sum")
    return result


@dataclass(frozen=True)
class PairV5FlowDPOResult:
    chosen_x_sigma: torch.Tensor
    rejected_x_sigma: torch.Tensor
    chosen_velocity_target: torch.Tensor
    rejected_velocity_target: torch.Tensor
    student_chosen_error: torch.Tensor
    student_rejected_error: torch.Tensor
    reference_chosen_error: torch.Tensor
    reference_rejected_error: torch.Tensor
    student_gap: torch.Tensor
    reference_gap: torch.Tensor
    advantage: torch.Tensor
    per_sample_loss: torch.Tensor
    loss: torch.Tensor


def reference_corrected_flow_dpo(
    chosen_clean: torch.Tensor,
    rejected_clean: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    student_chosen_prediction: torch.Tensor,
    student_rejected_prediction: torch.Tensor,
    reference_chosen_prediction: torch.Tensor,
    reference_rejected_prediction: torch.Tensor,
    *,
    beta: float,
    sample_weight: torch.Tensor | None = None,
) -> PairV5FlowDPOResult:
    """Compute one shared-randomness, reference-corrected flow-DPO loss.

    Predictions are exact81 spatial velocity fields.  The caller is
    responsible for producing all four predictions from the same native
    RV2V-4 condition registry; this core makes target construction and the
    preference sign explicit and auditable.
    """

    chosen, rejected, noise = _validate_clean_pair(
        chosen_clean, rejected_clean, epsilon
    )
    sigma_by_pair = _validate_sigma(sigma, reference=chosen)
    sigma_view = sigma_by_pair.reshape(
        int(chosen.shape[0]), *([1] * (chosen.ndim - 1))
    )
    chosen_x = (1.0 - sigma_view) * chosen + sigma_view * noise
    rejected_x = (1.0 - sigma_view) * rejected + sigma_view * noise
    chosen_target = noise - chosen
    rejected_target = noise - rejected
    for name, value in (
        ("chosen_x_sigma", chosen_x),
        ("rejected_x_sigma", rejected_x),
        ("chosen_velocity_target", chosen_target),
        ("rejected_velocity_target", rejected_target),
    ):
        if value.dtype != torch.float32 or value.requires_grad:
            raise PairV5FlowDPOError(f"{name} is not detached FP32")

    student_chosen = _prediction(
        "student_chosen_prediction",
        student_chosen_prediction,
        reference=chosen,
        trainable=True,
    )
    student_rejected = _prediction(
        "student_rejected_prediction",
        student_rejected_prediction,
        reference=rejected,
        trainable=True,
    )
    reference_chosen = _prediction(
        "reference_chosen_prediction",
        reference_chosen_prediction,
        reference=chosen,
        trainable=False,
    )
    reference_rejected = _prediction(
        "reference_rejected_prediction",
        reference_rejected_prediction,
        reference=rejected,
        trainable=False,
    )
    beta_value = _positive_scalar("beta", beta)
    weights = _weights(sample_weight, reference=chosen)

    def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction.float() - target).square().flatten(start_dim=1).mean(dim=1)

    student_chosen_error = mse(student_chosen, chosen_target)
    student_rejected_error = mse(student_rejected, rejected_target)
    reference_chosen_error = mse(reference_chosen, chosen_target)
    reference_rejected_error = mse(reference_rejected, rejected_target)
    student_gap = student_rejected_error - student_chosen_error
    reference_gap = reference_rejected_error - reference_chosen_error
    advantage = student_gap - reference_gap
    per_sample_loss = functional.softplus(-beta_value * advantage)
    loss = (per_sample_loss * weights).sum() / weights.sum()
    diagnostics = (
        student_chosen_error,
        student_rejected_error,
        reference_chosen_error,
        reference_rejected_error,
        student_gap,
        reference_gap,
        advantage,
        per_sample_loss,
    )
    if any(value.dtype != torch.float32 for value in diagnostics) or not bool(
        torch.isfinite(loss).item()
    ):
        raise PairV5FlowDPOError("flow-DPO diagnostics are non-finite or non-FP32")
    if not loss.requires_grad or loss.grad_fn is None:
        raise PairV5FlowDPOError("flow-DPO loss is detached from the student")
    return PairV5FlowDPOResult(
        chosen_x_sigma=chosen_x.detach(),
        rejected_x_sigma=rejected_x.detach(),
        chosen_velocity_target=chosen_target.detach(),
        rejected_velocity_target=rejected_target.detach(),
        student_chosen_error=student_chosen_error,
        student_rejected_error=student_rejected_error,
        reference_chosen_error=reference_chosen_error,
        reference_rejected_error=reference_rejected_error,
        student_gap=student_gap,
        reference_gap=reference_gap,
        advantage=advantage,
        per_sample_loss=per_sample_loss,
        loss=loss,
    )


__all__ = [
    "FORBIDDEN_EXTERNAL_INPUT_NAMES",
    "FRAME_COUNT",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "PairV5FlowDPOError",
    "PairV5FlowDPOResult",
    "SCHEMA_VERSION",
    "contract_receipt",
    "reference_corrected_flow_dpo",
]
