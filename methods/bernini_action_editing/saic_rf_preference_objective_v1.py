#!/usr/bin/env python3
"""Reference-relative rectified-flow preference objective for SAIC Stage-B.

The two endpoints share one Gaussian and one physical sigma.  The preference
term is a frozen-reference-corrected softplus with beta pinned to five.  An
additional chosen-side flow-matching term prevents a saturated pairwise loss
from merely pushing the rejected endpoint away.  Inputs are codec-reencoded,
detached exact81 endpoint latents admitted by
``saic_rollout_preference_set_v1``; pure-T2V visual data has no public route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


SCHEMA_VERSION = "bernini-saic-reference-relative-rf-preference-v1"
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
BETA = 5.0
CHOSEN_SIDE_WEIGHT = 1.0
REQUIRED_ARMS = ("dog", "human")
EXACT40_UPDATE_INDICES = (4, 12, 20, 28, 33, 34, 35, 37)
EXACT40_FORBIDDEN_UPDATE_INDICES = (38, 39)

FORBIDDEN_VISUAL_INPUT_NAMES = frozenset(
    {
        "pure_t2v_video",
        "pure_t2v_media",
        "pure_t2v_latent",
        "pure_t2v_noise",
        "proposal_video",
        "proposal_latent",
        "donor_video",
        "donor_latent",
        "target_video",
        "paired_target",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)


class SAICRFPreferenceError(ValueError):
    """A shared-state or reference-relative preference invariant failed."""


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
        raise SAICRFPreferenceError("receipt is not canonical finite JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def contract_receipt() -> Mapping[str, Any]:
    signature = set(inspect.signature(reference_relative_rf_preference).parameters)
    if not signature.isdisjoint(FORBIDDEN_VISUAL_INPUT_NAMES):
        raise SAICRFPreferenceError("objective signature exposes forbidden visual input")
    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_origin": "codec_reencoded_on_policy_source_conditioned_endpoint_only",
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "shared_state": "one_epsilon_and_one_sigma_for_chosen_and_rejected",
        "preference_advantage": (
            "student_rejected_minus_chosen_error_minus_"
            "reference_rejected_minus_chosen_error"
        ),
        "preference_loss": "softplus_minus_beta_times_reference_relative_advantage",
        "beta": BETA,
        "chosen_side_term": "student_chosen_rectified_flow_mse",
        "chosen_side_weight": CHOSEN_SIDE_WEIGHT,
        "required_arms": list(REQUIRED_ARMS),
        "exact40_update_indices": list(EXACT40_UPDATE_INDICES),
        "exact40_forbidden_update_indices": list(EXACT40_FORBIDDEN_UPDATE_INDICES),
        "pure_t2v_visual_data_consumed": False,
        "weighted_reward_compensation_used": False,
    }
    return {**body, "digest": _sha(body)}


def _fp32_detached(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICRFPreferenceError(f"{label} must be detached finite FP32")
    return value


def _endpoint(value: Any, *, label: str) -> torch.Tensor:
    result = _fp32_detached(value, label=label)
    if (
        result.ndim != 5
        or int(result.shape[0]) < 1
        or int(result.shape[1]) != LATENT_CHANNELS
        or int(result.shape[2]) != LATENT_PHASES
        or int(result.shape[3]) <= 0
        or int(result.shape[4]) <= 0
        or int(result.shape[3]) % 2
        or int(result.shape[4]) % 2
    ):
        raise SAICRFPreferenceError(
            f"{label} must be exact81 [B,16,21,H,W] with positive even H/W"
        )
    return result


def _student_prediction(
    value: Any, *, reference: torch.Tensor, label: str
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.shape != reference.shape
        or value.device != reference.device
        or value.dtype != torch.float32
        or not value.requires_grad
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise SAICRFPreferenceError(
            f"{label} must be a gradient-carrying finite FP32 output with endpoint geometry"
        )
    return value


def _reference_prediction(
    value: Any, *, reference: torch.Tensor, label: str
) -> torch.Tensor:
    result = _fp32_detached(value, label=label)
    if result.shape != reference.shape or result.device != reference.device:
        raise SAICRFPreferenceError(f"{label} geometry/device differs")
    return result


def _sigma(value: Any, *, reference: torch.Tensor) -> torch.Tensor:
    sigma = _fp32_detached(value, label="sigma")
    batch = int(reference.shape[0])
    if sigma.ndim == 0:
        sigma = sigma.expand(batch)
    elif tuple(sigma.shape) != (batch,):
        raise SAICRFPreferenceError("sigma must be scalar or exact [B]")
    if sigma.device != reference.device:
        raise SAICRFPreferenceError("sigma must share the endpoint device")
    if bool(((sigma <= 0.0) | (sigma >= 1.0)).any().item()):
        raise SAICRFPreferenceError("sigma must lie strictly in (0,1)")
    return sigma


def _arms(value: Any, *, batch: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SAICRFPreferenceError("arm_ids must be a sequence")
    result = tuple(value)
    if len(result) != batch or any(type(item) is not str for item in result):
        raise SAICRFPreferenceError("arm_ids must have one string per pair")
    if any(item not in REQUIRED_ARMS for item in result):
        raise SAICRFPreferenceError("only dog/human arms are registered")
    if any(item not in result for item in REQUIRED_ARMS):
        raise SAICRFPreferenceError(
            "dog and human must each contribute an admitted pair; zero update otherwise"
        )
    return result


def validate_update_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SAICRFPreferenceError("exact40_index must be an integer")
    if value in EXACT40_FORBIDDEN_UPDATE_INDICES:
        raise SAICRFPreferenceError("exact40 indices 38/39 are forbidden")
    if value not in EXACT40_UPDATE_INDICES:
        raise SAICRFPreferenceError("exact40_index is outside Stage-B J")
    return value


@dataclass(frozen=True)
class SAICRFPreferenceResult:
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
    reference_relative_advantage: torch.Tensor
    preference_term: torch.Tensor
    chosen_side_term: torch.Tensor
    per_sample_loss: torch.Tensor
    loss: torch.Tensor
    exact40_index: int
    arm_ids: tuple[str, ...]


def reference_relative_rf_preference(
    chosen_clean: torch.Tensor,
    rejected_clean: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    student_chosen_prediction: torch.Tensor,
    student_rejected_prediction: torch.Tensor,
    reference_chosen_prediction: torch.Tensor,
    reference_rejected_prediction: torch.Tensor,
    *,
    exact40_index: int,
    arm_ids: Sequence[str],
) -> SAICRFPreferenceResult:
    """Compute the pinned beta-5 RF preference and chosen-side objective."""

    chosen = _endpoint(chosen_clean, label="chosen_clean")
    rejected = _endpoint(rejected_clean, label="rejected_clean")
    noise = _endpoint(epsilon, label="epsilon")
    if chosen.shape != rejected.shape or chosen.shape != noise.shape:
        raise SAICRFPreferenceError("chosen/rejected/shared epsilon geometry differs")
    if chosen.device != rejected.device or chosen.device != noise.device:
        raise SAICRFPreferenceError("chosen/rejected/shared epsilon devices differ")
    if torch.equal(chosen, rejected):
        raise SAICRFPreferenceError("chosen and rejected endpoints are identical")
    arms = _arms(arm_ids, batch=int(chosen.shape[0]))
    step_index = validate_update_index(exact40_index)
    sigma_by_pair = _sigma(sigma, reference=chosen)
    sigma_view = sigma_by_pair.reshape(
        int(chosen.shape[0]), *([1] * (chosen.ndim - 1))
    )
    chosen_state = ((1.0 - sigma_view) * chosen + sigma_view * noise).detach()
    rejected_state = ((1.0 - sigma_view) * rejected + sigma_view * noise).detach()
    chosen_target = (noise - chosen).detach()
    rejected_target = (noise - rejected).detach()
    student_chosen = _student_prediction(
        student_chosen_prediction, reference=chosen, label="student_chosen_prediction"
    )
    student_rejected = _student_prediction(
        student_rejected_prediction,
        reference=rejected,
        label="student_rejected_prediction",
    )
    frozen_chosen = _reference_prediction(
        reference_chosen_prediction,
        reference=chosen,
        label="reference_chosen_prediction",
    )
    frozen_rejected = _reference_prediction(
        reference_rejected_prediction,
        reference=rejected,
        label="reference_rejected_prediction",
    )

    def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().flatten(start_dim=1).mean(dim=1)

    student_chosen_error = mse(student_chosen, chosen_target)
    student_rejected_error = mse(student_rejected, rejected_target)
    reference_chosen_error = mse(frozen_chosen, chosen_target)
    reference_rejected_error = mse(frozen_rejected, rejected_target)
    student_gap = student_rejected_error - student_chosen_error
    reference_gap = reference_rejected_error - reference_chosen_error
    advantage = student_gap - reference_gap
    preference = F.softplus(-BETA * advantage)
    chosen_side = student_chosen_error
    per_sample = preference + CHOSEN_SIDE_WEIGHT * chosen_side
    loss = per_sample.mean()
    tensors = (
        chosen_state,
        rejected_state,
        chosen_target,
        rejected_target,
        student_chosen_error,
        student_rejected_error,
        reference_chosen_error,
        reference_rejected_error,
        student_gap,
        reference_gap,
        advantage,
        preference,
        chosen_side,
        per_sample,
        loss,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors) or any(
        not bool(torch.isfinite(tensor.detach()).all().item()) for tensor in tensors
    ):
        raise SAICRFPreferenceError("objective produced non-finite or non-FP32 values")
    if not loss.requires_grad or loss.grad_fn is None:
        raise SAICRFPreferenceError("objective is detached from the student")
    return SAICRFPreferenceResult(
        chosen_x_sigma=chosen_state,
        rejected_x_sigma=rejected_state,
        chosen_velocity_target=chosen_target,
        rejected_velocity_target=rejected_target,
        student_chosen_error=student_chosen_error,
        student_rejected_error=student_rejected_error,
        reference_chosen_error=reference_chosen_error,
        reference_rejected_error=reference_rejected_error,
        student_gap=student_gap,
        reference_gap=reference_gap,
        reference_relative_advantage=advantage,
        preference_term=preference,
        chosen_side_term=chosen_side,
        per_sample_loss=per_sample,
        loss=loss,
        exact40_index=step_index,
        arm_ids=arms,
    )


__all__ = [
    "BETA",
    "CHOSEN_SIDE_WEIGHT",
    "EXACT40_FORBIDDEN_UPDATE_INDICES",
    "EXACT40_UPDATE_INDICES",
    "FORBIDDEN_VISUAL_INPUT_NAMES",
    "SAICRFPreferenceError",
    "SAICRFPreferenceResult",
    "SCHEMA_VERSION",
    "contract_receipt",
    "reference_relative_rf_preference",
    "validate_update_index",
]
