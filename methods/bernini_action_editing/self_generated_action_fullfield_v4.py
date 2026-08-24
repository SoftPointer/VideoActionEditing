#!/usr/bin/env python3
"""Full-field objectives for self-generated-anchor action editing V4.

The public functions in this file deliberately never compress a video field to
an endpoint, pooled vector, band statistic, or Frozen-relative trust radius.
Every action term keeps the complete ``[B,C,T,H,W]`` latent trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


SCHEMA = "bernini-self-generated-action-fullfield-v4"
LATENT_PHASES = 21
DEFAULT_LAGS = (1, 2, 4, 8)


class FullFieldObjectiveError(RuntimeError):
    pass


def _validate_field(value: Any, *, label: str) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != 16
        or int(value.shape[2]) != LATENT_PHASES
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise FullFieldObjectiveError(
            f"{label} must be finite [B,16,{LATENT_PHASES},H,W]"
        )


def causal_trajectory(value: Any) -> Any:
    """Keep the full field while making phase zero exactly zero."""

    _validate_field(value, label="causal trajectory input")
    result = value - value[:, :, :1]
    if not bool((result[:, :, 0] == 0).all().item()):
        raise FullFieldObjectiveError("causal trajectory lost its exact phase-zero gauge")
    return result


def predicted_clean(noisy: Any, velocity: Any, sigma: float) -> Any:
    _validate_field(noisy, label="noisy field")
    _validate_field(velocity, label="velocity field")
    if tuple(noisy.shape) != tuple(velocity.shape):
        raise FullFieldObjectiveError("noisy/velocity geometry differs")
    if isinstance(sigma, bool) or not math.isfinite(float(sigma)) or not 0.0 <= float(sigma) <= 1.0001:
        raise FullFieldObjectiveError("sigma must be finite in [0,1]")
    return noisy.float() - float(sigma) * velocity.float()


def anchor_action_trajectory(anchor_clean: Any) -> Any:
    """Use the generated video's entire dense motion, not an internal code."""

    return causal_trajectory(anchor_clean.float())


def student_action_trajectory(action_clean: Any, noop_clean: Any) -> Any:
    _validate_field(action_clean, label="student action clean field")
    _validate_field(noop_clean, label="student no-op clean field")
    if tuple(action_clean.shape) != tuple(noop_clean.shape):
        raise FullFieldObjectiveError("student action/no-op geometry differs")
    return causal_trajectory(action_clean.float() - noop_clean.float())


def _charbonnier_normalized(student: Any, teacher: Any, *, epsilon: float) -> Any:
    import torch

    scale = teacher.detach().square().mean().sqrt().clamp_min(float(epsilon))
    residual = (student - teacher) / scale
    return (torch.sqrt(residual.square() + 1.0e-4) - 1.0e-2).mean()


def multiscale_temporal_loss(
    student: Any,
    teacher: Any,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    epsilon: float = 1.0e-4,
) -> Any:
    _validate_field(student, label="multiscale student")
    _validate_field(teacher, label="multiscale teacher")
    if tuple(student.shape) != tuple(teacher.shape):
        raise FullFieldObjectiveError("multiscale student/teacher geometry differs")
    chosen = tuple(int(item) for item in lags)
    if not chosen or len(set(chosen)) != len(chosen) or any(item <= 0 or item >= LATENT_PHASES for item in chosen):
        raise FullFieldObjectiveError("temporal lags must be unique values in [1,20]")
    terms = []
    for lag in chosen:
        student_delta = student[:, :, lag:] - student[:, :, :-lag]
        teacher_delta = teacher[:, :, lag:] - teacher[:, :, :-lag]
        terms.append(
            _charbonnier_normalized(
                student_delta, teacher_delta, epsilon=epsilon
            )
        )
    return sum(terms) / float(len(terms))


@dataclass(frozen=True)
class FullFieldLoss:
    total: Any
    dense: Any
    multiscale: Any
    teacher_rms: Any
    student_rms: Any


def fullfield_action_loss(
    student: Any,
    teacher: Any,
    *,
    dense_weight: float = 1.0,
    multiscale_weight: float = 1.0,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> FullFieldLoss:
    import torch

    _validate_field(student, label="full-field student")
    _validate_field(teacher, label="full-field teacher")
    if tuple(student.shape) != tuple(teacher.shape):
        raise FullFieldObjectiveError("full-field student/teacher geometry differs")
    for label, value in (("dense_weight", dense_weight), ("multiscale_weight", multiscale_weight)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise FullFieldObjectiveError(f"{label} must be finite and non-negative")
    if float(dense_weight) + float(multiscale_weight) <= 0:
        raise FullFieldObjectiveError("at least one full-field loss must be active")
    dense = _charbonnier_normalized(student, teacher, epsilon=1.0e-4)
    multiscale = multiscale_temporal_loss(student, teacher, lags=lags)
    total = float(dense_weight) * dense + float(multiscale_weight) * multiscale
    if not bool(torch.isfinite(total).item()):
        raise FullFieldObjectiveError("full-field action loss is non-finite")
    return FullFieldLoss(
        total=total,
        dense=dense,
        multiscale=multiscale,
        teacher_rms=teacher.detach().square().mean().sqrt(),
        student_rms=student.detach().square().mean().sqrt(),
    )


def source_carrier_target(source_clean: Any, anchor_clean: Any) -> Any:
    """Add the generated dense trajectory to source; phase zero stays source."""

    _validate_field(source_clean, label="carrier source")
    _validate_field(anchor_clean, label="carrier anchor")
    if tuple(source_clean.shape) != tuple(anchor_clean.shape):
        raise FullFieldObjectiveError("carrier source/anchor geometry differs")
    target = source_clean.float() + anchor_action_trajectory(anchor_clean)
    if not bool((target[:, :, 0] == source_clean.float()[:, :, 0]).all().item()):
        raise FullFieldObjectiveError("source carrier does not preserve exact phase zero")
    return target.contiguous()


def project_and_cap_preservation_gradients(
    action_gradients: Sequence[Any],
    preservation_gradients: Sequence[Any],
    *,
    cap_ratio: float = 0.25,
) -> tuple[list[Any], dict[str, Any]]:
    """Action-first PCGrad: remove only conflicting preservation gradient.

    This operates on the complete optimizer gradient, not on a video
    representation.  The preservation contribution is then capped relative to
    the action gradient so it cannot turn the action objective into no-op.
    """

    import torch

    action = list(action_gradients)
    preservation = list(preservation_gradients)
    if len(action) != len(preservation) or not action:
        raise FullFieldObjectiveError("gradient lists must be equally non-empty")
    if isinstance(cap_ratio, bool) or not math.isfinite(float(cap_ratio)) or float(cap_ratio) < 0:
        raise FullFieldObjectiveError("cap_ratio must be finite and non-negative")
    if any(a is None or p is None or tuple(a.shape) != tuple(p.shape) for a, p in zip(action, preservation)):
        raise FullFieldObjectiveError("every action/preservation gradient must exist and align")
    dot = sum((a.float() * p.float()).sum() for a, p in zip(action, preservation))
    action_sq = sum(a.float().square().sum() for a in action)
    preservation_sq = sum(p.float().square().sum() for p in preservation)
    conflict = bool(dot.detach().item() < 0.0)
    coefficient = (dot / action_sq.clamp_min(1.0e-20)) if conflict else dot.new_zeros(())
    projected = [p - coefficient.to(dtype=p.dtype) * a for a, p in zip(action, preservation)]
    projected_sq = sum(value.float().square().sum() for value in projected)
    action_norm = action_sq.sqrt()
    projected_norm = projected_sq.sqrt()
    cap = float(cap_ratio) * action_norm
    scale = torch.minimum(
        torch.ones_like(projected_norm), cap / projected_norm.clamp_min(1.0e-20)
    )
    projected = [value * scale.to(dtype=value.dtype) for value in projected]
    combined = [a + p for a, p in zip(action, projected)]
    return combined, {
        "action_norm": action_norm.detach(),
        "preservation_norm": preservation_sq.sqrt().detach(),
        "action_preservation_dot": dot.detach(),
        "conflict_projected": conflict,
        "preservation_post_cap_scale": scale.detach(),
    }
