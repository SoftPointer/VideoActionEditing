#!/usr/bin/env python3
"""Endpoint-consensus action reward with frozen-editor functional trust.

V3 keeps the pure-T2V video in a detached teacher role.  It compresses each
teacher code to a net temporal displacement (late phase minus early phase),
optionally replaces the individual direction with a cross-anchor action-family
consensus, and constrains the LoRA gain to a two-sided band.  A full velocity
trust term can bound changes hidden by the compact PsiOut representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-self-generated-action-endpoint-consensus-v3"
ACTION_ROW_PAIRS = ((0, 1), (2, 3))
ARM_NAMES = (
    "endpoint_cell_band",
    "endpoint_consensus_band",
    "endpoint_consensus_trust_001",
    "endpoint_consensus_trust_010",
)


class EndpointConsensusError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    teacher_mode: str
    learning_rate: float
    lower_scale: float
    upper_scale: float
    endpoint_perpendicular_weight: float
    full_trust_weight: float
    nuisance_weight: float


def _arm(name: str, teacher_mode: str, full_trust_weight: float) -> ArmSpec:
    return ArmSpec(
        name=name,
        teacher_mode=teacher_mode,
        learning_rate=1.0e-4,
        lower_scale=0.05,
        upper_scale=0.15,
        endpoint_perpendicular_weight=0.10,
        full_trust_weight=full_trust_weight,
        nuisance_weight=0.10,
    )


_ARMS = {
    "endpoint_cell_band": _arm("endpoint_cell_band", "cell", 0.0),
    "endpoint_consensus_band": _arm("endpoint_consensus_band", "consensus", 0.0),
    "endpoint_consensus_trust_001": _arm(
        "endpoint_consensus_trust_001", "consensus", 0.01
    ),
    "endpoint_consensus_trust_010": _arm(
        "endpoint_consensus_trust_010", "consensus", 0.10
    ),
}


def arm_spec(name: str) -> ArmSpec:
    try:
        return _ARMS[name]
    except KeyError as error:
        raise EndpointConsensusError(f"unknown arm: {name}") from error


def _unit(vector: Any) -> tuple[Any, float]:
    import torch

    flat = vector.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    amplitude = float(norm.item())
    if not math.isfinite(amplitude) or amplitude <= 1.0e-12:
        raise EndpointConsensusError("endpoint teacher has zero/non-finite amplitude")
    return (flat / norm).reshape_as(vector).detach(), amplitude


def endpoint_displacement(raw: Any, *, endpoint_frames: int = 3) -> Any:
    """Return late-minus-early displacement from a ``[B,21,32]`` code."""

    import torch

    if (
        not isinstance(raw, torch.Tensor)
        or raw.ndim != 3
        or tuple(int(item) for item in raw.shape[1:]) != (21, 32)
        or type(endpoint_frames) is not int
        or not 1 <= endpoint_frames <= 5
        or not bool(torch.isfinite(raw).all().item())
    ):
        raise EndpointConsensusError("endpoint code must be finite [B,21,32]")
    return raw[:, -endpoint_frames:, :].float().mean(dim=1) - raw[
        :, :endpoint_frames, :
    ].float().mean(dim=1)


@dataclass(frozen=True)
class EndpointAuthority:
    cell_unit: Any
    consensus_unit: Any
    robust_amplitude: float
    cell_amplitude: float
    peer_consensus_cosine: float


def build_endpoint_authority(
    cells: Sequence[Mapping[str, Any]],
) -> Mapping[tuple[int, int], EndpointAuthority]:
    """Build family consensus and held-anchor admission evidence.

    The consensus used for optimization pools both available anchors.  The
    admission cosine is stricter: each cell is compared only with a centroid
    built from the other actor/scene in the same action family.
    """

    import torch

    if not isinstance(cells, Sequence) or len(cells) != 16:
        raise EndpointConsensusError("expected 16 teacher cells")
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    cell_units: dict[tuple[int, int], Any] = {}
    cell_amplitudes: dict[tuple[int, int], float] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise EndpointConsensusError("teacher cell must be a mapping")
        key = (cell.get("row_index"), cell.get("slot"))
        unit = cell.get("teacher_unit")
        amplitude = cell.get("teacher_amplitude")
        if (
            key[0] not in range(4)
            or key[1] not in range(4)
            or key in by_key
            or not isinstance(unit, torch.Tensor)
            or tuple(int(item) for item in unit.shape) != (1, 21, 32)
            or not isinstance(amplitude, (int, float))
            or isinstance(amplitude, bool)
            or not math.isfinite(float(amplitude))
            or float(amplitude) <= 0
        ):
            raise EndpointConsensusError("teacher cache cell differs")
        raw = unit.float() * float(amplitude)
        endpoint_unit, endpoint_amplitude = _unit(endpoint_displacement(raw))
        by_key[key] = cell
        cell_units[key] = endpoint_unit
        cell_amplitudes[key] = endpoint_amplitude
    expected = {(row, slot) for row in range(4) for slot in range(4)}
    if set(by_key) != expected:
        raise EndpointConsensusError("teacher cache grid is incomplete")

    result: dict[tuple[int, int], EndpointAuthority] = {}
    for left_row, right_row in ACTION_ROW_PAIRS:
        family_rows = (left_row, right_row)
        family_units = [
            cell_units[(row, slot)] for row in family_rows for slot in range(4)
        ]
        consensus, _ = _unit(torch.stack(family_units, dim=0).mean(dim=0))
        family_amplitudes = torch.tensor(
            [cell_amplitudes[(row, slot)] for row in family_rows for slot in range(4)],
            dtype=torch.float32,
        )
        robust_amplitude = float(family_amplitudes.median().item())
        if not math.isfinite(robust_amplitude) or robust_amplitude <= 0:
            raise EndpointConsensusError("family endpoint amplitude is invalid")
        for row, peer_row in ((left_row, right_row), (right_row, left_row)):
            peer, _ = _unit(
                torch.stack([cell_units[(peer_row, slot)] for slot in range(4)], dim=0).mean(dim=0)
            )
            for slot in range(4):
                key = (row, slot)
                peer_cosine = float((cell_units[key] * peer).sum().item())
                if not math.isfinite(peer_cosine) or peer_cosine <= 0:
                    raise EndpointConsensusError(
                        f"cell {key} fails positive held-anchor endpoint admission"
                    )
                result[key] = EndpointAuthority(
                    cell_unit=cell_units[key],
                    consensus_unit=consensus,
                    robust_amplitude=robust_amplitude,
                    cell_amplitude=cell_amplitudes[key],
                    peer_consensus_cosine=peer_cosine,
                )
    return result


@dataclass(frozen=True)
class EndpointBandLoss:
    action: Any
    perpendicular: Any
    gain_mean: Any
    lower_mean: Any
    upper_mean: Any
    delta_norm_mean: Any


def endpoint_band_loss(
    *,
    student_raw: Any,
    frozen_raw: Any,
    detached_teacher_unit: Any,
    detached_teacher_amplitude: Any,
    lower_scale: float,
    upper_scale: float,
) -> EndpointBandLoss:
    """Constrain frozen-relative endpoint gain to a non-zero two-sided band."""

    import torch

    if (
        not isinstance(student_raw, torch.Tensor)
        or not isinstance(frozen_raw, torch.Tensor)
        or student_raw.shape != frozen_raw.shape
        or student_raw.ndim != 3
        or tuple(int(item) for item in student_raw.shape[1:]) != (21, 32)
        or student_raw.device != frozen_raw.device
        or not isinstance(detached_teacher_unit, torch.Tensor)
        or tuple(int(item) for item in detached_teacher_unit.shape) != (1, 32)
        or detached_teacher_unit.device != student_raw.device
        or detached_teacher_unit.requires_grad
        or not isinstance(detached_teacher_amplitude, torch.Tensor)
        or detached_teacher_amplitude.dtype != torch.float32
        or detached_teacher_amplitude.ndim != 1
        or int(detached_teacher_amplitude.shape[0]) != int(student_raw.shape[0])
        or detached_teacher_amplitude.device != student_raw.device
        or detached_teacher_amplitude.requires_grad
    ):
        raise EndpointConsensusError("endpoint band authority differs")
    if (
        not isinstance(lower_scale, (int, float))
        or isinstance(lower_scale, bool)
        or not isinstance(upper_scale, (int, float))
        or isinstance(upper_scale, bool)
        or not math.isfinite(float(lower_scale))
        or not math.isfinite(float(upper_scale))
        or float(lower_scale) <= 0
        or float(upper_scale) <= float(lower_scale)
    ):
        raise EndpointConsensusError("endpoint band scales are invalid")
    teacher_norm = torch.linalg.vector_norm(detached_teacher_unit.float(), dim=1)
    if not bool(
        torch.allclose(teacher_norm, torch.ones_like(teacher_norm), atol=1.0e-5, rtol=1.0e-5)
    ):
        raise EndpointConsensusError("endpoint teacher direction must be unit-normalized")
    if not bool(torch.isfinite(detached_teacher_amplitude).all().item()) or bool(
        (detached_teacher_amplitude <= 0).any().item()
    ):
        raise EndpointConsensusError("endpoint amplitude must be finite and positive")

    delta = endpoint_displacement(student_raw) - endpoint_displacement(frozen_raw)
    teacher = detached_teacher_unit.float()
    gain = (delta * teacher).sum(dim=1)
    lower = detached_teacher_amplitude * float(lower_scale)
    upper = detached_teacher_amplitude * float(upper_scale)
    scale = lower.clamp_min(1.0e-12)
    below = torch.relu((lower - gain) / scale)
    above = torch.relu((gain - upper) / scale)
    action = (below.square() + above.square()).mean()
    parallel = gain[:, None] * teacher
    perpendicular = (
        (delta - parallel).square().sum(dim=1)
        / detached_teacher_amplitude.square().clamp_min(1.0e-12)
    ).mean()
    if not bool(torch.isfinite(action).item()) or not bool(torch.isfinite(perpendicular).item()):
        raise EndpointConsensusError("endpoint band loss is non-finite")
    return EndpointBandLoss(
        action=action,
        perpendicular=perpendicular,
        gain_mean=gain.mean(),
        lower_mean=lower.mean(),
        upper_mean=upper.mean(),
        delta_norm_mean=torch.linalg.vector_norm(delta, dim=1).mean(),
    )


def full_functional_trust(*, student_velocity: Any, frozen_velocity: Any) -> Any:
    """Normalized energy of all adapter-induced post-head velocity changes."""

    import torch

    if (
        not isinstance(student_velocity, torch.Tensor)
        or not isinstance(frozen_velocity, torch.Tensor)
        or student_velocity.shape != frozen_velocity.shape
        or student_velocity.device != frozen_velocity.device
        or frozen_velocity.requires_grad
        or not bool(torch.isfinite(student_velocity).all().item())
        or not bool(torch.isfinite(frozen_velocity).all().item())
    ):
        raise EndpointConsensusError("full functional trust authority differs")
    numerator = (student_velocity.float() - frozen_velocity.float()).square().mean()
    denominator = frozen_velocity.float().square().mean().detach().clamp_min(1.0e-12)
    result = numerator / denominator
    if not bool(torch.isfinite(result).item()):
        raise EndpointConsensusError("full functional trust is non-finite")
    return result


def weighted_total(
    *, spec: ArmSpec, action: Any, perpendicular: Any, full_trust: Any, nuisance: Any
) -> Any:
    values = (action, perpendicular, full_trust, nuisance)
    if any(value is None or getattr(value, "ndim", None) != 0 for value in values):
        raise EndpointConsensusError("all objective terms must be scalar")
    return (
        action
        + spec.endpoint_perpendicular_weight * perpendicular
        + spec.full_trust_weight * full_trust
        + spec.nuisance_weight * nuisance
    )


def validate_arm_table() -> None:
    if tuple(_ARMS) != ARM_NAMES:
        raise EndpointConsensusError("arm registration order differs")
    for name, spec in _ARMS.items():
        if (
            spec.name != name
            or spec.teacher_mode not in {"cell", "consensus"}
            or not math.isfinite(spec.learning_rate)
            or spec.learning_rate <= 0
        ):
            raise EndpointConsensusError(f"invalid arm: {name}")
