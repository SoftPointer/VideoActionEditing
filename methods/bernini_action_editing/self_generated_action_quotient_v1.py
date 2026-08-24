#!/usr/bin/env python3
"""Small, model-free contract for the self-generated action quotient screen.

The pure T2V anchor contributes only detached post-head action/phase codes.
It is never an RGB, latent, velocity, or flow-matching target for the RV2V
student.  This module keeps the eight-arm preservation factorial and the
auxiliary losses independently testable without loading Bernini.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SCHEMA = "bernini-self-generated-action-quotient-v1"
ARM_NAMES = (
    "action_only",
    "action_only_lowlr",
    "action_noop",
    "action_start",
    "action_nuisance",
    "action_start_nuisance",
    "action_start_nuisance_noop",
    "action_start_nuisance_border",
)


class ActionQuotientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    learning_rate: float
    noop_weight: float
    start_weight: float
    nuisance_weight: float
    border_weight: float


_ARMS = {
    "action_only": ArmSpec("action_only", 1.0e-4, 0.0, 0.0, 0.0, 0.0),
    "action_only_lowlr": ArmSpec("action_only_lowlr", 5.0e-5, 0.0, 0.0, 0.0, 0.0),
    "action_noop": ArmSpec("action_noop", 1.0e-4, 0.05, 0.0, 0.0, 0.0),
    "action_start": ArmSpec("action_start", 1.0e-4, 0.0, 0.25, 0.0, 0.0),
    "action_nuisance": ArmSpec("action_nuisance", 1.0e-4, 0.0, 0.0, 0.10, 0.0),
    "action_start_nuisance": ArmSpec(
        "action_start_nuisance", 1.0e-4, 0.0, 0.25, 0.10, 0.0
    ),
    "action_start_nuisance_noop": ArmSpec(
        "action_start_nuisance_noop", 1.0e-4, 0.05, 0.25, 0.10, 0.0
    ),
    "action_start_nuisance_border": ArmSpec(
        "action_start_nuisance_border", 1.0e-4, 0.0, 0.25, 0.10, 0.05
    ),
}


def arm_spec(name: str) -> ArmSpec:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise ActionQuotientError(f"unknown arm: {name}") from error


def preservation_losses(
    *, predicted_clean: Any, source_clean: Any, border_width: int = 4
) -> Mapping[str, Any]:
    """Return start-state and static-camera border losses in latent space."""

    import torch

    if (
        not isinstance(predicted_clean, torch.Tensor)
        or not isinstance(source_clean, torch.Tensor)
        or predicted_clean.shape != source_clean.shape
        or predicted_clean.ndim != 5
        or tuple(int(x) for x in predicted_clean.shape[:3]) != (1, 16, 21)
    ):
        raise ActionQuotientError("clean latents must match [1,16,21,H,W]")
    height, width = map(int, predicted_clean.shape[-2:])
    if border_width <= 0 or 2 * border_width >= min(height, width):
        raise ActionQuotientError("border width is outside latent geometry")
    delta = predicted_clean.float() - source_clean.float()
    start = delta[:, :, 0].square().mean()
    border = torch.cat(
        (
            delta[..., :border_width, :].reshape(-1),
            delta[..., -border_width:, :].reshape(-1),
            delta[..., border_width:-border_width, :border_width].reshape(-1),
            delta[..., border_width:-border_width, -border_width:].reshape(-1),
        )
    ).square().mean()
    return {"start": start, "border": border}


def nuisance_coefficient_loss(raw_code: Any, camera_unit: Any, appearance_unit: Any) -> Any:
    """Penalize the two directions removed by the detached teacher quotient.

    Projection alone would make those directions invisible to the action loss.
    Explicitly penalizing them prevents camera/appearance drift from becoming a
    free optimization channel.
    """

    import torch

    tensors = (raw_code, camera_unit, appearance_unit)
    if (
        any(not isinstance(value, torch.Tensor) for value in tensors)
        or raw_code.shape != camera_unit.shape
        or raw_code.shape != appearance_unit.shape
        or raw_code.ndim != 3
        or tuple(int(x) for x in raw_code.shape[1:]) != (21, 32)
    ):
        raise ActionQuotientError("nuisance codes must match [B,21,32]")
    flat = raw_code.float().reshape(int(raw_code.shape[0]), -1)
    camera = camera_unit.float().reshape_as(flat)
    appearance = appearance_unit.float().reshape_as(flat)
    coefficients = torch.stack(
        ((flat * camera).sum(dim=1), (flat * appearance).sum(dim=1)), dim=1
    )
    return coefficients.square().mean()


def weighted_total(
    *,
    spec: ArmSpec,
    action: Any,
    noop: Any,
    start: Any,
    nuisance: Any,
    border: Any,
) -> Any:
    values = (action, noop, start, nuisance, border)
    if any(value is None for value in values):
        raise ActionQuotientError("all scalar losses are required")
    total = (
        action
        + spec.noop_weight * noop
        + spec.start_weight * start
        + spec.nuisance_weight * nuisance
        + spec.border_weight * border
    )
    if not bool(total.ndim == 0):
        raise ActionQuotientError("weighted objective must be scalar")
    return total


def validate_arm_table() -> None:
    if tuple(_ARMS) != ARM_NAMES:
        raise ActionQuotientError("arm registration order differs")
    for name, spec in _ARMS.items():
        if spec.name != name or not math.isfinite(spec.learning_rate) or spec.learning_rate <= 0:
            raise ActionQuotientError(f"invalid arm: {name}")
        weights = (spec.noop_weight, spec.start_weight, spec.nuisance_weight, spec.border_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ActionQuotientError(f"invalid preservation weight: {name}")


validate_arm_table()
