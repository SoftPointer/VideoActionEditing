#!/usr/bin/env python3
"""Model-free contracts for frozen-editor-relative action reward v2.

The v1 screen aligns the student's total source-conditioned action code to a
single detached T2V teacher direction.  That can keep rotating a capable
frozen editor after it already performs the requested action.  V2 instead
optimizes only the adapter-induced residual relative to the frozen editor:

    delta = Psi(v_lora) - Psi(v_frozen)

It asks for a small positive gain along the detached action unit vector and
penalizes every orthogonal residual.  Thus the frozen RV2V response remains the
default behavior, while the T2V generator supplies only an action increment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SCHEMA = "bernini-self-generated-action-residual-margin-v2"
ARM_NAMES = (
    "margin_005",
    "margin_010",
    "margin_020",
    "margin_010_perp_010",
    "margin_010_perp_100",
    "margin_010_perp_100_onset_100",
    "margin_010_perp_100_onset_400",
    "margin_010_perp_100_onset_400_noop_020",
)


class ResidualMarginError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    learning_rate: float
    margin_scale: float
    perpendicular_weight: float
    onset_weight: float
    onset_frames: int
    nuisance_weight: float
    noop_weight: float


def _arm(
    name: str,
    margin_scale: float,
    *,
    perpendicular_weight: float = 0.0,
    onset_weight: float = 0.0,
    onset_frames: int = 1,
    noop_weight: float = 0.0,
) -> ArmSpec:
    return ArmSpec(
        name=name,
        learning_rate=1.0e-4,
        margin_scale=margin_scale,
        perpendicular_weight=perpendicular_weight,
        onset_weight=onset_weight,
        onset_frames=onset_frames,
        nuisance_weight=0.10,
        noop_weight=noop_weight,
    )


_ARMS = {
    "margin_005": _arm("margin_005", 0.05),
    "margin_010": _arm("margin_010", 0.10),
    "margin_020": _arm("margin_020", 0.20),
    "margin_010_perp_010": _arm(
        "margin_010_perp_010", 0.10, perpendicular_weight=0.10
    ),
    "margin_010_perp_100": _arm(
        "margin_010_perp_100", 0.10, perpendicular_weight=1.0
    ),
    "margin_010_perp_100_onset_100": _arm(
        "margin_010_perp_100_onset_100",
        0.10,
        perpendicular_weight=1.0,
        onset_weight=1.0,
        onset_frames=3,
    ),
    "margin_010_perp_100_onset_400": _arm(
        "margin_010_perp_100_onset_400",
        0.10,
        perpendicular_weight=1.0,
        onset_weight=4.0,
        onset_frames=3,
    ),
    "margin_010_perp_100_onset_400_noop_020": _arm(
        "margin_010_perp_100_onset_400_noop_020",
        0.10,
        perpendicular_weight=1.0,
        onset_weight=4.0,
        onset_frames=3,
        noop_weight=0.20,
    ),
}


def arm_spec(name: str) -> ArmSpec:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise ResidualMarginError(f"unknown arm: {name}") from error


@dataclass(frozen=True)
class ResidualMarginLoss:
    action: Any
    perpendicular: Any
    gain_mean: Any
    margin_mean: Any
    delta_norm_mean: Any


def residual_margin_loss(
    *,
    student_raw: Any,
    frozen_raw: Any,
    detached_teacher_unit: Any,
    detached_teacher_amplitude: Any,
    margin_scale: float,
) -> ResidualMarginLoss:
    """Require a bounded action gain and penalize non-action adapter change."""

    import torch

    tensors = (student_raw, frozen_raw, detached_teacher_unit)
    if (
        any(not isinstance(value, torch.Tensor) for value in tensors)
        or student_raw.shape != frozen_raw.shape
        or student_raw.shape != detached_teacher_unit.shape
        or student_raw.ndim != 3
        or tuple(int(x) for x in student_raw.shape[1:]) != (21, 32)
        or student_raw.device != frozen_raw.device
        or student_raw.device != detached_teacher_unit.device
        or detached_teacher_unit.requires_grad
        or not isinstance(detached_teacher_amplitude, torch.Tensor)
        or detached_teacher_amplitude.dtype != torch.float32
        or detached_teacher_amplitude.ndim != 1
        or int(detached_teacher_amplitude.shape[0]) != int(student_raw.shape[0])
        or detached_teacher_amplitude.device != student_raw.device
        or detached_teacher_amplitude.requires_grad
    ):
        raise ResidualMarginError("residual margin authority differs")
    if (
        not isinstance(margin_scale, (int, float))
        or isinstance(margin_scale, bool)
        or not math.isfinite(float(margin_scale))
        or float(margin_scale) <= 0
    ):
        raise ResidualMarginError("margin_scale must be finite and positive")
    if not bool(torch.isfinite(student_raw).all().item()) or not bool(
        torch.isfinite(frozen_raw).all().item()
    ):
        raise ResidualMarginError("student/frozen code is non-finite")
    if not bool(torch.isfinite(detached_teacher_amplitude).all().item()) or bool(
        (detached_teacher_amplitude <= 0).any().item()
    ):
        raise ResidualMarginError("teacher amplitude must be finite and positive")

    student = student_raw.float().reshape(int(student_raw.shape[0]), -1)
    frozen = frozen_raw.float().reshape_as(student)
    teacher = detached_teacher_unit.float().reshape_as(student)
    teacher_norm = torch.linalg.vector_norm(teacher, dim=1)
    if not bool(
        torch.allclose(
            teacher_norm,
            torch.ones_like(teacher_norm),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise ResidualMarginError("teacher direction must be unit-normalized")

    delta = student - frozen
    gain = (delta * teacher).sum(dim=1)
    margin = detached_teacher_amplitude * float(margin_scale)
    relative_deficit = torch.relu(1.0 - gain / margin)
    action = relative_deficit.square().mean()
    parallel = gain[:, None] * teacher
    perpendicular = (
        (delta - parallel).square().sum(dim=1)
        / detached_teacher_amplitude.square().clamp_min(1.0e-12)
    ).mean()
    if not bool(torch.isfinite(action).item()) or not bool(torch.isfinite(perpendicular).item()):
        raise ResidualMarginError("residual margin loss is non-finite")
    return ResidualMarginLoss(
        action=action,
        perpendicular=perpendicular,
        gain_mean=gain.mean(),
        margin_mean=margin.mean(),
        delta_norm_mean=torch.linalg.vector_norm(delta, dim=1).mean(),
    )


def onset_preservation_loss(
    *, predicted_clean: Any, source_clean: Any, onset_frames: int
) -> Any:
    """Source-clean envelope over the first 1--3 latent phases."""

    import torch

    if (
        not isinstance(predicted_clean, torch.Tensor)
        or not isinstance(source_clean, torch.Tensor)
        or predicted_clean.shape != source_clean.shape
        or predicted_clean.ndim != 5
        or int(predicted_clean.shape[2]) != 21
        or type(onset_frames) is not int
        or not 1 <= onset_frames <= 3
    ):
        raise ResidualMarginError("onset latents must match [B,C,21,H,W] and K in [1,3]")
    delta = predicted_clean[:, :, :onset_frames].float() - source_clean[
        :, :, :onset_frames
    ].float()
    weights = torch.tensor(
        (1.0, 0.5, 0.25)[:onset_frames],
        device=delta.device,
        dtype=delta.dtype,
    )
    phase_mse = delta.square().mean(dim=(0, 1, 3, 4))
    return (phase_mse * weights).sum() / weights.sum()


def weighted_total(
    *,
    spec: ArmSpec,
    action: Any,
    perpendicular: Any,
    onset: Any,
    nuisance: Any,
    noop: Any,
) -> Any:
    values = (action, perpendicular, onset, nuisance, noop)
    if any(value is None or getattr(value, "ndim", None) != 0 for value in values):
        raise ResidualMarginError("all objective terms must be scalar")
    return (
        action
        + spec.perpendicular_weight * perpendicular
        + spec.onset_weight * onset
        + spec.nuisance_weight * nuisance
        + spec.noop_weight * noop
    )


def validate_arm_table() -> None:
    if tuple(_ARMS) != ARM_NAMES:
        raise ResidualMarginError("arm registration order differs")
    for name, spec in _ARMS.items():
        if spec.name != name or not math.isfinite(spec.learning_rate) or spec.learning_rate <= 0:
            raise ResidualMarginError(f"invalid arm: {name}")
        numbers = (
            spec.margin_scale,
            spec.perpendicular_weight,
            spec.onset_weight,
            spec.nuisance_weight,
            spec.noop_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in numbers):
            raise ResidualMarginError(f"invalid objective weight: {name}")
        if not 1 <= spec.onset_frames <= 3:
            raise ResidualMarginError(f"invalid onset frame count: {name}")


validate_arm_table()
