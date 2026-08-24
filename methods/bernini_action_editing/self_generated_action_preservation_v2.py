#!/usr/bin/env python3
"""Model-free preservation objective for the action-quotient v2 canary.

This module deliberately separates *training proxies* from decoded-video
preservation evidence.  The losses below constrain the onset and the change
from the frozen renderer, but they do not claim face identity, background, or
camera preservation.  Those axes remain post-decode gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Sequence


SCHEMA = "bernini-self-generated-action-preservation-canary-v2"
ONSET_WEIGHTS = (1.0, 0.5, 0.25)
ROUTE_SCOPES = ("all_attention", "cross_attn2_qo")


class ActionPreservationV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class CanaryArm:
    name: str
    learning_rate: float
    onset_weight: float
    nuisance_weight: float
    noop_weight: float
    functional_weight: float
    route_scope: str


# A matched 20-update mechanism screen.  Every arm uses the same source rows,
# noise slots, teacher cache, initialization seed, and checkpoint schedule.
# It is not a winner table and does not authorize promotion without decoded
# full-video preservation/action review.
_ARMS = {
    "v2_onset_all": CanaryArm(
        "v2_onset_all", 1.0e-4, 0.25, 0.10, 0.0, 0.0, "all_attention"
    ),
    "v2_noop020_all": CanaryArm(
        "v2_noop020_all", 1.0e-4, 0.25, 0.10, 0.20, 0.0, "all_attention"
    ),
    "v2_func010_all": CanaryArm(
        "v2_func010_all", 1.0e-4, 0.25, 0.10, 0.0, 0.10, "all_attention"
    ),
    "v2_func025_all": CanaryArm(
        "v2_func025_all", 1.0e-4, 0.25, 0.10, 0.0, 0.25, "all_attention"
    ),
    "v2_func050_all": CanaryArm(
        "v2_func050_all", 1.0e-4, 0.25, 0.10, 0.0, 0.50, "all_attention"
    ),
    "v2_onset_cross_qo": CanaryArm(
        "v2_onset_cross_qo", 1.0e-4, 0.25, 0.10, 0.0, 0.0, "cross_attn2_qo"
    ),
    "v2_func010_cross_qo": CanaryArm(
        "v2_func010_cross_qo", 1.0e-4, 0.25, 0.10, 0.0, 0.10, "cross_attn2_qo"
    ),
    "v2_func025_cross_qo": CanaryArm(
        "v2_func025_cross_qo", 1.0e-4, 0.25, 0.10, 0.0, 0.25, "cross_attn2_qo"
    ),
}
ARM_NAMES = tuple(_ARMS)


_ATTENTION_PROJECTION = re.compile(
    r"^.+\.blocks\.(?P<block>[0-9]+)\.attn(?P<attention>[12])\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)$"
)


def arm_spec(name: str) -> CanaryArm:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise ActionPreservationV2Error(f"unknown v2 canary arm: {name}") from error


def _require_finite_tensor(value: Any, *, label: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise ActionPreservationV2Error(f"{label} is not a tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ActionPreservationV2Error(f"{label} is non-finite")
    return value


def onset_envelope_loss(
    *, predicted_clean: Any, source_clean: Any,
    weights: Sequence[float] = ONSET_WEIGHTS,
) -> Any:
    """Penalize the first three clean-latent phases with a decaying envelope.

    This is a training proxy for a continuous source-state onset.  It is not a
    decoded first-frame identity guarantee; inference must independently use
    a per-solver-step source/noise clamp before that stronger claim is made.
    """

    predicted = _require_finite_tensor(predicted_clean, label="predicted clean latent")
    source = _require_finite_tensor(source_clean, label="source clean latent")
    if (
        predicted.shape != source.shape
        or predicted.ndim != 5
        or int(predicted.shape[0]) != 1
        or int(predicted.shape[1]) != 16
        or int(predicted.shape[2]) != 21
    ):
        raise ActionPreservationV2Error(
            "clean latents must share exact [1,16,21,H,W] geometry"
        )
    normalized_weights = tuple(float(item) for item in weights)
    if (
        not normalized_weights
        or len(normalized_weights) > int(predicted.shape[2])
        or any(not math.isfinite(item) or item <= 0.0 for item in normalized_weights)
    ):
        raise ActionPreservationV2Error("onset weights are invalid")
    terms = [
        weight
        * (predicted[:, :, index].float() - source[:, :, index].float()).square().mean()
        for index, weight in enumerate(normalized_weights)
    ]
    return sum(terms)


def functional_non_regression_loss(
    *, student_action_code: Any, frozen_action_code: Any, teacher_action_unit: Any,
) -> Any:
    """Penalize LoRA drift orthogonal to the detached action direction.

    Inputs are exact post-head ``[B,21,32]`` codes for the same source, noise,
    sigma, and instruction.  Motion along the frozen detached teacher action
    unit is exempt; all orthogonal change is a functional non-regression proxy.
    The frozen code and teacher unit are authorities and therefore must not
    require gradients.
    """

    import torch

    student = _require_finite_tensor(student_action_code, label="student action code")
    frozen = _require_finite_tensor(frozen_action_code, label="frozen action code")
    teacher = _require_finite_tensor(teacher_action_unit, label="teacher action unit")
    if (
        student.shape != frozen.shape
        or student.shape != teacher.shape
        or student.ndim != 3
        or tuple(int(item) for item in student.shape[1:]) != (21, 32)
    ):
        raise ActionPreservationV2Error("functional codes must share [B,21,32]")
    if frozen.requires_grad or teacher.requires_grad:
        raise ActionPreservationV2Error("frozen functional authorities must be detached")
    flat_delta = (student.float() - frozen.float()).reshape(int(student.shape[0]), -1)
    flat_teacher = teacher.float().reshape_as(flat_delta)
    norm2 = flat_teacher.square().sum(dim=1, keepdim=True)
    if bool((norm2 <= 1.0e-12).any().item()):
        raise ActionPreservationV2Error("teacher action unit is degenerate")
    projection = (
        (flat_delta * flat_teacher).sum(dim=1, keepdim=True) / norm2
    ) * flat_teacher
    orthogonal = flat_delta - projection
    result = orthogonal.square().mean()
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        raise ActionPreservationV2Error("functional loss is invalid")
    return result


def temporal_dc_non_regression_loss(
    *, student_velocity: Any, frozen_velocity: Any, sigma: float, start_phase: int = 3,
) -> Any:
    """Penalize time-constant clean-latent drift after the onset envelope.

    Static appearance/background changes have a strong temporal-DC component.
    The loss intentionally ignores phases covered by the onset term and does
    not penalize zero-mean temporal motion.  It remains a latent diagnostic,
    not a decoded identity or background proof.
    """

    import torch

    student = _require_finite_tensor(student_velocity, label="student velocity")
    frozen = _require_finite_tensor(frozen_velocity, label="frozen velocity")
    if (
        student.shape != frozen.shape
        or student.ndim != 5
        or int(student.shape[0]) != 1
        or int(student.shape[1]) != 16
        or int(student.shape[2]) != 21
    ):
        raise ActionPreservationV2Error(
            "functional velocities must share exact [1,16,21,H,W] geometry"
        )
    if frozen.requires_grad:
        raise ActionPreservationV2Error("frozen velocity authority must be detached")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or not 0.0 <= sigma_value <= 1.0001:
        raise ActionPreservationV2Error("functional sigma is invalid")
    if type(start_phase) is not int or not 0 <= start_phase < int(student.shape[2]):
        raise ActionPreservationV2Error("functional start phase is invalid")
    clean_drift = -sigma_value * (student.float() - frozen.float())
    temporal_dc = clean_drift[:, :, start_phase:].mean(dim=2)
    result = temporal_dc.square().mean()
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        raise ActionPreservationV2Error("temporal-DC functional loss is invalid")
    return result


def select_projection_scope(names: Iterable[str], *, scope: str) -> list[str]:
    """Filter already-audited Wan attention projection names.

    ``cross_attn2_qo`` is intentionally named for the observable module
    topology.  It does not claim that attn2 contains a source-only route.
    """

    if scope not in ROUTE_SCOPES:
        raise ActionPreservationV2Error(f"unsupported route scope: {scope}")
    parsed: list[tuple[str, re.Match[str]]] = []
    seen: set[str] = set()
    for value in names:
        if not isinstance(value, str) or value in seen:
            raise ActionPreservationV2Error("projection names must be unique strings")
        seen.add(value)
        match = _ATTENTION_PROJECTION.fullmatch(value)
        if match is None:
            raise ActionPreservationV2Error(f"invalid attention projection: {value}")
        parsed.append((value, match))
    if not parsed:
        raise ActionPreservationV2Error("projection registry is empty")
    if scope == "all_attention":
        selected = [name for name, _ in parsed]
    else:
        selected = [
            name
            for name, match in parsed
            if match.group("attention") == "2"
            and match.group("projection") in {"to_q", "to_out.0"}
        ]
    selected = sorted(selected)
    if not selected:
        raise ActionPreservationV2Error(f"route scope selected no projections: {scope}")
    return selected


def weighted_total(
    *, spec: CanaryArm, action: Any, onset: Any, nuisance: Any,
    noop: Any, functional: Any,
) -> Any:
    """Compose only the preregistered canary proxy objective."""

    values = (action, onset, nuisance, noop, functional)
    if any(value is None or getattr(value, "ndim", None) != 0 for value in values):
        raise ActionPreservationV2Error("v2 loss components must be scalar tensors")
    total = (
        action
        + spec.onset_weight * onset
        + spec.nuisance_weight * nuisance
        + spec.noop_weight * noop
        + spec.functional_weight * functional
    )
    if total.ndim != 0:
        raise ActionPreservationV2Error("v2 weighted objective is not scalar")
    return total


def validate_registry() -> None:
    if tuple(_ARMS) != ARM_NAMES or len(ARM_NAMES) != 8:
        raise ActionPreservationV2Error("v2 arm registry differs")
    for name, spec in _ARMS.items():
        if spec.name != name or spec.route_scope not in ROUTE_SCOPES:
            raise ActionPreservationV2Error(f"invalid v2 arm: {name}")
        values = (
            spec.learning_rate,
            spec.onset_weight,
            spec.nuisance_weight,
            spec.noop_weight,
            spec.functional_weight,
        )
        if any(not math.isfinite(item) or item < 0.0 for item in values):
            raise ActionPreservationV2Error(f"invalid v2 arm scalar: {name}")
        if spec.learning_rate <= 0.0:
            raise ActionPreservationV2Error(f"invalid v2 learning rate: {name}")


validate_registry()


__all__ = [
    "ARM_NAMES",
    "ActionPreservationV2Error",
    "CanaryArm",
    "ONSET_WEIGHTS",
    "ROUTE_SCOPES",
    "SCHEMA",
    "arm_spec",
    "functional_non_regression_loss",
    "onset_envelope_loss",
    "select_projection_scope",
    "temporal_dc_non_regression_loss",
    "weighted_total",
]
