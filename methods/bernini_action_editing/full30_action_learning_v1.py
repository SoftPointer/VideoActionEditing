#!/usr/bin/env python3
"""Fail-closed numeric core for the full-30 Bernini action-learning arm.

This module deliberately contains no model loader, optimizer loop, media reader,
or launcher.  It freezes the pieces that must be identical before either formal
arm may execute:

* the post-native-head ``PsiOut_v1`` temporal action representation;
* per-record direction plus same-mode amplitude-floor loss;
* the action-first diagonal update used by the main/control pair; and
* the exact 128-row, ten-epoch DP2 x SP4 logical-record schedule.

Teacher and real-source manifests remain separate authorities.  In particular,
this module cannot authorize an optimizer from unreviewed generated media or
from the historical block-22 Phi sidecars.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Iterable, Optional


SCHEMA_VERSION = "bernini-full30-action-learning-core-v2"
PSIOUT_SCHEMA_VERSION = "bernini-post-head-action-quotient-v1"
SCHEDULE_SCHEMA_VERSION = "bernini-full30-action-schedule-v2"

LATENT_CHANNELS = 16
LATENT_PHASES = 21
QUOTIENT_WIDTH = 32
TRAIN_ROWS = 128
TRAIN_SOURCES = 64
TEACHER_SEED_CELLS = 8
EPOCHS = 10
GLOBAL_BATCH = 8
MICROBATCHES_PER_UPDATE = 4
DP_SIZE = 2
SP_SIZE = 4
WORLD_SIZE = DP_SIZE * SP_SIZE
MAX_UPDATES = 160
SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
BRANCHES = ("action", "incomplete")

ACTION_LEARNING_RATE = 1.0e-4
ACTION_BETA2 = 0.999
NUMERIC_EPSILON = 1.0e-8
MIN_ACTION_GRAD_NORM = 1.0e-12
MIN_TEACHER_NORM = 1.0e-6
MIN_NUISANCE_NORM = 1.0e-6
MIN_APPEARANCE_RESIDUAL_RATIO = 1.0e-5
GLOBAL_UPDATE_CLIP = 1.0
MAX_FP32_BUCKET_ELEMENTS = 16_777_216
PROJECTION_ROUNDING_RELATIVE_TOLERANCE = 1.0e-6

ORDER_DOMAIN = b"full30-action-source-order-v2\x00"
NOISE_DOMAIN = b"full30-action-paired-noise-v2\x00"


class Full30ActionLearningError(RuntimeError):
    """Raised when a full-30 action-learning contract is incomplete."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host dependent
        raise Full30ActionLearningError("PyTorch is required for tensor operations") from error
    return torch


def _tensor(value: Any, *, name: str, shape_tail: tuple[int, ...]) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or value.ndim != len(shape_tail) + 1
        or tuple(int(x) for x in value.shape[1:]) != shape_tail
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Full30ActionLearningError(
            f"{name} must be finite contiguous FP32 [B,{','.join(map(str, shape_tail))}]"
        )
    if int(value.shape[0]) <= 0:
        raise Full30ActionLearningError(f"{name} batch must be positive")
    return value


def _post_head_delta(value: Any, *, name: str = "post_head_delta") -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or value.ndim != 5
        or int(value.shape[1]) != LATENT_CHANNELS
        or int(value.shape[2]) != LATENT_PHASES
        or int(value.shape[3]) < 2
        or int(value.shape[4]) < 2
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Full30ActionLearningError(
            f"{name} must be finite contiguous FP32 [B,16,21,H,W] with H,W>=2"
        )
    if int(value.shape[0]) <= 0:
        raise Full30ActionLearningError(f"{name} batch must be positive")
    return value


def psiout_raw_v1(post_head_delta: Any) -> Any:
    """Map a post-native-head velocity delta to the fixed ``[B,21,32]`` code."""

    torch = _torch()
    value = _post_head_delta(post_head_delta)
    _, _, _, height, width = map(int, value.shape)
    y = (2.0 * torch.arange(height, device=value.device, dtype=torch.float32) + 1.0 - height) / float(height)
    x = (2.0 * torch.arange(width, device=value.device, dtype=torch.float32) + 1.0 - width) / float(width)
    signed = y[:, None] + x[None, :]
    signed = signed - signed.mean()
    rms = signed.square().mean().sqrt()
    if not bool(torch.isfinite(rms).item()) or float(rms.item()) <= MIN_NUISANCE_NORM:
        raise Full30ActionLearningError("PsiOut signed spatial coordinate is degenerate")
    signed = signed / rms

    spatial_mean = value.mean(dim=(-2, -1)).permute(0, 2, 1)
    signed_mean = (value * signed[None, None, None]).mean(dim=(-2, -1)).permute(0, 2, 1)
    code = torch.cat((spatial_mean, signed_mean), dim=2)
    if tuple(code.shape[1:]) != (LATENT_PHASES, QUOTIENT_WIDTH):
        raise Full30ActionLearningError("PsiOut channel concatenation differs")
    causal = code - code[:, :1]
    causal = torch.cat((torch.zeros_like(causal[:, :1]), causal[:, 1:]), dim=1)
    if not bool(torch.isfinite(causal).all().item()):
        raise Full30ActionLearningError("PsiOut raw code is non-finite")
    return causal.float().contiguous()


@dataclass(frozen=True)
class NuisancePacket:
    camera_unit: Any
    appearance_unit: Any
    camera_norm: Any
    appearance_norm: Any
    appearance_residual_ratio: Any


def build_nuisance_packet_v1(camera_raw: Any, appearance_raw: Any) -> NuisancePacket:
    """Freeze camera then Gram-Schmidt appearance nuisance directions per row."""

    torch = _torch()
    camera = _tensor(
        camera_raw,
        name="camera_raw",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    appearance = _tensor(
        appearance_raw,
        name="appearance_raw",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    if camera.shape != appearance.shape or camera.device != appearance.device:
        raise Full30ActionLearningError("camera and appearance nuisance batches differ")
    camera_flat = camera.reshape(int(camera.shape[0]), -1)
    appearance_flat = appearance.reshape(int(appearance.shape[0]), -1)
    camera_norm = torch.linalg.vector_norm(camera_flat, dim=1)
    appearance_norm = torch.linalg.vector_norm(appearance_flat, dim=1)
    if bool((camera_norm <= MIN_NUISANCE_NORM).any().item()):
        raise Full30ActionLearningError("camera nuisance is degenerate")
    if bool((appearance_norm <= MIN_NUISANCE_NORM).any().item()):
        raise Full30ActionLearningError("appearance nuisance is degenerate")
    camera_unit = camera_flat / camera_norm[:, None]
    appearance_orthogonal = appearance_flat - (
        (appearance_flat * camera_unit).sum(dim=1)[:, None] * camera_unit
    )
    appearance_orthogonal_norm = torch.linalg.vector_norm(appearance_orthogonal, dim=1)
    ratio = appearance_orthogonal_norm / appearance_norm
    if bool((ratio <= MIN_APPEARANCE_RESIDUAL_RATIO).any().item()):
        raise Full30ActionLearningError("appearance nuisance is collinear with camera")
    appearance_unit = appearance_orthogonal / appearance_orthogonal_norm[:, None]
    camera_unit = camera_unit.reshape_as(camera).float().contiguous()
    appearance_unit = appearance_unit.reshape_as(appearance).float().contiguous()
    return NuisancePacket(
        camera_unit=camera_unit,
        appearance_unit=appearance_unit,
        camera_norm=camera_norm.float().contiguous(),
        appearance_norm=appearance_norm.float().contiguous(),
        appearance_residual_ratio=ratio.float().contiguous(),
    )


def project_nuisances_v1(raw_code: Any, packet: NuisancePacket) -> Any:
    """Project one nuisance packet per batch record in the registered order."""

    torch = _torch()
    raw = _tensor(raw_code, name="raw_code", shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH))
    if not isinstance(packet, NuisancePacket):
        raise Full30ActionLearningError("nuisance packet type differs")
    camera = _tensor(
        packet.camera_unit,
        name="camera_unit",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    appearance = _tensor(
        packet.appearance_unit,
        name="appearance_unit",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    if raw.shape != camera.shape or raw.shape != appearance.shape or raw.device != camera.device or raw.device != appearance.device:
        raise Full30ActionLearningError("nuisance packet does not match raw code")
    flat = raw.reshape(int(raw.shape[0]), -1)
    camera_flat = camera.reshape_as(flat)
    appearance_flat = appearance.reshape_as(flat)
    projected = flat - (flat * camera_flat).sum(dim=1)[:, None] * camera_flat
    projected = projected - (projected * appearance_flat).sum(dim=1)[:, None] * appearance_flat
    projected = projected.reshape_as(raw)
    if not bool(torch.isfinite(projected).all().item()):
        raise Full30ActionLearningError("nuisance projection is non-finite")
    return projected.float().contiguous()


def psiout_v1(post_head_delta: Any, packet: NuisancePacket) -> Any:
    return project_nuisances_v1(psiout_raw_v1(post_head_delta), packet)


def teacher_unit_v1(projected_teacher: Any) -> Any:
    """Normalize each reviewed teacher independently; never flatten the batch."""

    torch = _torch()
    teacher = _tensor(
        projected_teacher,
        name="projected_teacher",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    flat = teacher.reshape(int(teacher.shape[0]), -1)
    norm = torch.linalg.vector_norm(flat, dim=1)
    if bool((norm <= MIN_TEACHER_NORM).any().item()):
        raise Full30ActionLearningError("teacher quotient is degenerate")
    return (flat / norm[:, None]).reshape_as(teacher).float().contiguous()


@dataclass(frozen=True)
class ActionLoss:
    total: Any
    direction_mean: Any
    amplitude_mean: Any
    per_record_direction: Any
    per_record_amplitude: Any
    student_norm: Any


def paired_action_loss_v1(
    student_code: Any,
    detached_teacher_unit: Any,
    detached_same_mode_amplitude_floor: Any,
) -> ActionLoss:
    """Direction plus one-sided same-mode amplitude loss, reduced per record."""

    torch = _torch()
    student = _tensor(
        student_code,
        name="student_code",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    teacher = _tensor(
        detached_teacher_unit,
        name="detached_teacher_unit",
        shape_tail=(LATENT_PHASES, QUOTIENT_WIDTH),
    )
    floor = detached_same_mode_amplitude_floor
    if (
        student.shape != teacher.shape
        or student.device != teacher.device
        or teacher.requires_grad
        or not isinstance(floor, torch.Tensor)
        or floor.dtype != torch.float32
        or floor.ndim != 1
        or int(floor.shape[0]) != int(student.shape[0])
        or floor.device != student.device
        or floor.requires_grad
        or not bool(torch.isfinite(floor).all().item())
        or bool((floor <= MIN_TEACHER_NORM).any().item())
    ):
        raise Full30ActionLearningError("paired action loss authority differs")
    student_flat = student.reshape(int(student.shape[0]), -1)
    teacher_flat = teacher.reshape(int(teacher.shape[0]), -1)
    teacher_norm = torch.linalg.vector_norm(teacher_flat, dim=1)
    if not bool(torch.allclose(teacher_norm, torch.ones_like(teacher_norm), atol=1.0e-5, rtol=1.0e-5)):
        raise Full30ActionLearningError("teacher quotient is not unit normalized per record")
    student_norm = torch.linalg.vector_norm(student_flat, dim=1)
    direction = 1.0 - (student_flat * teacher_flat).sum(dim=1) / (student_norm + NUMERIC_EPSILON)
    log_ratio = torch.log((floor + NUMERIC_EPSILON) / (student_norm + NUMERIC_EPSILON))
    amplitude = torch.relu(log_ratio).square()
    total = (direction + amplitude).mean()
    if not bool(torch.isfinite(total).item()):
        raise Full30ActionLearningError("paired action loss is non-finite")
    return ActionLoss(
        total=total,
        direction_mean=direction.mean(),
        amplitude_mean=amplitude.mean(),
        per_record_direction=direction,
        per_record_amplitude=amplitude,
        student_norm=student_norm,
    )


@dataclass(frozen=True)
class ActionFirstUpdate:
    new_second_moment: Any
    descent_direction: Any
    action_direction: Any
    noop_direction_after_projection: Optional[Any]
    conflict_dot_before: float
    conflict_dot_after: float
    projection_rounding_tolerance: float
    noop_cap_factor: float
    unclipped_norm: float
    clip_factor: float
    actual_action_descent_dot: float


def action_first_update_v1(
    action_gradient: Any,
    second_moment: Any,
    *,
    noop_gradient: Optional[Any] = None,
    learning_rate: float = ACTION_LEARNING_RATE,
) -> ActionFirstUpdate:
    """Return the exact no-momentum action-first descent direction.

    The caller applies ``theta -= learning_rate * descent_direction``.  Inputs
    represent one canonical globally synchronized flattened parameter vector;
    production code must compute its coefficients with the registered FP64
    bucket accumulation.
    """

    torch = _torch()
    if (
        not isinstance(action_gradient, torch.Tensor)
        or not isinstance(second_moment, torch.Tensor)
        or action_gradient.dtype != torch.float32
        or second_moment.dtype != torch.float32
        or action_gradient.ndim != 1
        or action_gradient.shape != second_moment.shape
        or action_gradient.device != second_moment.device
        or not action_gradient.is_contiguous()
        or not second_moment.is_contiguous()
        or not bool(torch.isfinite(action_gradient).all().item())
        or not bool(torch.isfinite(second_moment).all().item())
        or bool((second_moment < 0).any().item())
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise Full30ActionLearningError("action-first optimizer state differs")
    if noop_gradient is not None and (
        not isinstance(noop_gradient, torch.Tensor)
        or noop_gradient.dtype != torch.float32
        or noop_gradient.shape != action_gradient.shape
        or noop_gradient.device != action_gradient.device
        or not noop_gradient.is_contiguous()
        or not bool(torch.isfinite(noop_gradient).all().item())
    ):
        raise Full30ActionLearningError("noop gradient differs")

    with torch.no_grad():
        action_norm = torch.linalg.vector_norm(action_gradient.double())
        if float(action_norm.item()) <= MIN_ACTION_GRAD_NORM:
            raise Full30ActionLearningError("action gradient is degenerate")
        new_second_moment = (
            ACTION_BETA2 * second_moment
            + (1.0 - ACTION_BETA2) * action_gradient.square()
        ).float().contiguous()
        denominator = new_second_moment.sqrt() + NUMERIC_EPSILON
        action_direction = (action_gradient / denominator).float().contiguous()
        noop_after = None
        conflict_before = 0.0
        conflict_after = 0.0
        projection_tolerance = 0.0
        cap_factor = 0.0
        direction = action_direction.clone()
        if noop_gradient is not None:
            noop_direction = (noop_gradient / denominator).float().contiguous()
            conflict_before_tensor = (action_gradient.double() * noop_direction.double()).sum()
            conflict_before = float(conflict_before_tensor.item())
            if conflict_before < 0.0:
                action_square = (action_gradient.double().square()).sum()
                coefficient = conflict_before_tensor / action_square
                noop_direction = (
                    noop_direction.double() - coefficient * action_gradient.double()
                ).float().contiguous()
            conflict_after_tensor = (action_gradient.double() * noop_direction.double()).sum()
            conflict_after = float(conflict_after_tensor.item())
            projection_tolerance = PROJECTION_ROUNDING_RELATIVE_TOLERANCE * max(
                1.0,
                float(action_norm.item())
                * float(torch.linalg.vector_norm(noop_direction.double()).item()),
            )
            if conflict_after < -projection_tolerance:
                raise Full30ActionLearningError("noop projection still opposes action")
            action_direction_norm = torch.linalg.vector_norm(action_direction.double())
            noop_direction_norm = torch.linalg.vector_norm(noop_direction.double())
            cap_factor = min(
                1.0,
                float(action_direction_norm.item())
                / (float(noop_direction_norm.item()) + NUMERIC_EPSILON),
            )
            noop_after = (noop_direction * cap_factor).float().contiguous()
            direction = (action_direction + noop_after).float().contiguous()
        unclipped_norm_tensor = torch.linalg.vector_norm(direction.double())
        unclipped_norm = float(unclipped_norm_tensor.item())
        clip_factor = min(1.0, GLOBAL_UPDATE_CLIP / (unclipped_norm + NUMERIC_EPSILON))
        direction = (direction * clip_factor).float().contiguous()
        descent_dot = float(
            (action_gradient.double() * (float(learning_rate) * direction.double())).sum().item()
        )
        if not math.isfinite(descent_dot) or descent_dot <= 0.0:
            raise Full30ActionLearningError("actual action descent dot is not positive")
    return ActionFirstUpdate(
        new_second_moment=new_second_moment,
        descent_direction=direction,
        action_direction=action_direction,
        noop_direction_after_projection=noop_after,
        conflict_dot_before=conflict_before,
        conflict_dot_after=conflict_after,
        projection_rounding_tolerance=projection_tolerance,
        noop_cap_factor=cap_factor,
        unclipped_norm=unclipped_norm,
        clip_factor=clip_factor,
        actual_action_descent_dot=descent_dot,
    )


@dataclass(frozen=True, order=True)
class ActionPairRow:
    row_id: str
    source_id: str
    branch: str
    teacher_cell_id: str


@dataclass(frozen=True)
class ScheduledActionPair:
    global_index: int
    epoch: int
    update: int
    microbatch: int
    dp_rank: int
    sigma_index: int
    noise_seed: int
    row: ActionPairRow


def _text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise Full30ActionLearningError(f"{name} must be nonempty NUL-free text")
    return value


def _validate_rows(rows: Iterable[ActionPairRow]) -> tuple[ActionPairRow, ...]:
    values = tuple(rows)
    if len(values) != TRAIN_ROWS or any(not isinstance(row, ActionPairRow) for row in values):
        raise Full30ActionLearningError("formal schedule requires exactly 128 rows")
    for row in values:
        _text(row.row_id, name="row_id")
        _text(row.source_id, name="source_id")
        _text(row.teacher_cell_id, name="teacher_cell_id")
        if row.branch not in BRANCHES:
            raise Full30ActionLearningError("row branch differs")
    if len({row.row_id for row in values}) != TRAIN_ROWS:
        raise Full30ActionLearningError("row IDs must be unique")
    by_source: dict[str, list[ActionPairRow]] = defaultdict(list)
    for row in values:
        by_source[row.source_id].append(row)
    if len(by_source) != TRAIN_SOURCES:
        raise Full30ActionLearningError("formal schedule requires 64 unique sources")
    for source_rows in by_source.values():
        if (
            len(source_rows) != 2
            or {row.branch for row in source_rows} != set(BRANCHES)
            or len({row.teacher_cell_id for row in source_rows}) != 1
        ):
            raise Full30ActionLearningError(
                "each source must have matched action/incomplete rows and one teacher cell"
            )
    teacher_sources: dict[str, set[str]] = defaultdict(set)
    for row in values:
        teacher_sources[row.teacher_cell_id].add(row.source_id)
    if len(teacher_sources) != TEACHER_SEED_CELLS or any(
        len(sources) != 8 for sources in teacher_sources.values()
    ):
        raise Full30ActionLearningError(
            "each of eight fit teacher cells must bind exactly eight sources"
        )
    if Counter(row.branch for row in values) != Counter({"action": 64, "incomplete": 64}):
        raise Full30ActionLearningError("action/incomplete row ratio differs")
    return tuple(sorted(values, key=lambda row: row.row_id.encode("utf-8")))


def _order_key(epoch: int, source_id: str) -> tuple[bytes, bytes]:
    encoded = source_id.encode("utf-8")
    return (
        hashlib.sha256(ORDER_DOMAIN + struct.pack(">I", epoch) + b"\x00" + encoded).digest(),
        encoded,
    )


def _noise_seed(run_seed: int, epoch: int, source_id: str, sigma_index: int) -> int:
    payload = (
        NOISE_DOMAIN
        + struct.pack(">Q", run_seed)
        + struct.pack(">I", epoch)
        + b"\x00"
        + source_id.encode("utf-8")
        + struct.pack(">H", sigma_index)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def build_formal_schedule_v1(
    rows: Iterable[ActionPairRow],
    *,
    run_seed: int,
) -> tuple[ScheduledActionPair, ...]:
    """Build the exact 1280-record schedule shared by both formal arms."""

    if type(run_seed) is not int or not 0 <= run_seed < 2**64:
        raise Full30ActionLearningError("run_seed must be an unsigned 64-bit integer")
    canonical = _validate_rows(rows)
    by_source: dict[str, dict[str, ActionPairRow]] = defaultdict(dict)
    for row in canonical:
        by_source[row.source_id][row.branch] = row
    source_ids = tuple(sorted(by_source, key=lambda value: value.encode("utf-8")))
    scheduled: list[ScheduledActionPair] = []
    for epoch in range(EPOCHS):
        ordered_sources = sorted(source_ids, key=lambda value: _order_key(epoch, value))
        if len(set(ordered_sources)) != TRAIN_SOURCES:
            raise Full30ActionLearningError("epoch source permutation differs")
        for source_position, source_id in enumerate(ordered_sources):
            source_exposure_index = epoch * TRAIN_SOURCES + source_position
            update = epoch * (TRAIN_SOURCES // MICROBATCHES_PER_UPDATE) + (
                source_position // MICROBATCHES_PER_UPDATE
            )
            microbatch = source_position % MICROBATCHES_PER_UPDATE
            sigma_index = SIGMA_INDICES[source_exposure_index % len(SIGMA_INDICES)]
            noise_seed = _noise_seed(run_seed, epoch, source_id, sigma_index)
            branch_order = BRANCHES if (epoch + microbatch) % 2 == 0 else tuple(reversed(BRANCHES))
            for dp_rank, branch in enumerate(branch_order):
                global_index = update * GLOBAL_BATCH + microbatch * DP_SIZE + dp_rank
                scheduled.append(
                    ScheduledActionPair(
                        global_index=global_index,
                        epoch=epoch,
                        update=update,
                        microbatch=microbatch,
                        dp_rank=dp_rank,
                        sigma_index=sigma_index,
                        noise_seed=noise_seed,
                        row=by_source[source_id][branch],
                    )
                )
    result = tuple(scheduled)
    if len(result) != MAX_UPDATES * GLOBAL_BATCH:
        raise Full30ActionLearningError("formal schedule length differs")
    sigma_counts = Counter(item.sigma_index for item in result)
    if tuple(sigma_counts[index] for index in SIGMA_INDICES) != (214, 214, 214, 214, 212, 212):
        raise Full30ActionLearningError("formal sigma counts differ")
    for update in range(MAX_UPDATES):
        group = result[update * GLOBAL_BATCH : (update + 1) * GLOBAL_BATCH]
        if [(item.microbatch, item.dp_rank) for item in group] != [
            (microbatch, dp_rank)
            for microbatch in range(MICROBATCHES_PER_UPDATE)
            for dp_rank in range(DP_SIZE)
        ]:
            raise Full30ActionLearningError("DP2 x four-microbatch schedule differs")
        if Counter(item.row.branch for item in group) != Counter({"action": 4, "incomplete": 4}):
            raise Full30ActionLearningError("each update must be action/incomplete 1:1")
        for microbatch in range(MICROBATCHES_PER_UPDATE):
            pair = [item for item in group if item.microbatch == microbatch]
            if (
                len(pair) != 2
                or pair[0].row.source_id != pair[1].row.source_id
                or pair[0].row.teacher_cell_id != pair[1].row.teacher_cell_id
                or {item.row.branch for item in pair} != set(BRANCHES)
                or len({item.sigma_index for item in pair}) != 1
                or len({item.noise_seed for item in pair}) != 1
            ):
                raise Full30ActionLearningError(
                    "each microbatch must pair one source/action/incomplete with shared sigma and noise"
                )
    return result


def physical_branch_evaluations_per_update(arm: str) -> int:
    """Count semantic branch evaluations, independent of model batch fusion."""

    if arm == "action-only":
        return GLOBAL_BATCH * 3  # trainable branch + frozen branch + frozen noop
    if arm == "action+retain":
        return GLOBAL_BATCH * 4  # plus trainable noop
    raise Full30ActionLearningError("unknown formal arm")


__all__ = [
    "ActionFirstUpdate",
    "ActionLoss",
    "ActionPairRow",
    "Full30ActionLearningError",
    "NuisancePacket",
    "ScheduledActionPair",
    "action_first_update_v1",
    "build_formal_schedule_v1",
    "build_nuisance_packet_v1",
    "paired_action_loss_v1",
    "physical_branch_evaluations_per_update",
    "project_nuisances_v1",
    "psiout_raw_v1",
    "psiout_v1",
    "teacher_unit_v1",
]
