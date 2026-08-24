#!/usr/bin/env python3
"""One strict WORLD8 optimizer update for full-30 action learning.

This module owns orchestration only.  Model routing stays in
``full30_action_runtime_v1``; the quotient objective and formal schedule stay
in ``full30_action_learning_v1``; the sole parameter transaction stays in
``full30_action_optimizer_v1``.  Action and retain gradients are produced by
two independent forward/backward phases and are never mixed through one graph.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import struct
from typing import Any, Callable, ContextManager, Mapping, Optional, Sequence

import torch

try:  # Support both a release directory on sys.path and package imports.
    import full30_action_checkpoint_v1 as checkpoint_core
    import full30_action_learning_v1 as learning_core
    import full30_action_optimizer_v1 as optimizer_core
    import full30_action_runtime_v1 as runtime_core
except ImportError:  # pragma: no cover - package import mode
    from . import full30_action_checkpoint_v1 as checkpoint_core
    from . import full30_action_learning_v1 as learning_core
    from . import full30_action_optimizer_v1 as optimizer_core
    from . import full30_action_runtime_v1 as runtime_core


SCHEMA_VERSION = "bernini-full30-action-training-step-v1"
RECEIPT_SCHEMA_VERSION = "bernini-full30-action-training-step-receipt-v1"
OBJECTIVE_AUTHORITY_SCHEMA_VERSION = (
    "bernini-full30-action-record-objective-authority-v1"
)
GRADIENT_COLLECTIVE_SCHEMA_VERSION = (
    "bernini-full30-action-gradient-collective-v1"
)
WORLD_CONSENSUS_SCHEMA_VERSION = "bernini-full30-action-world-consensus-v1"

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
LOCAL_MICRO_RECORDS = 4
GLOBAL_BATCH = 8
MIN_GRADIENT_NORM = 1.0e-12

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LORA = re.compile(
    r"(?:^|\.)blocks\.(?P<block>\d+)\.attn(?P<attention>[12])\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)\."
    r"lora_(?P<factor>[AB])(?:\.default)?\.weight$"
)
_SOURCE_TYPED = re.compile(r"(?:^|\.)source_delta\.(?:weight|bias)$")
_TARGET_TYPED = re.compile(r"(?:^|\.)target_delta\.(?:weight|bias)$")
_ROLE_TYPED = re.compile(r"(?:^|\.)role_embedding$")


class Full30ActionTrainingStepError(RuntimeError):
    """Raised before accepting an ambiguous or partial optimizer update."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Full30ActionTrainingStepError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Full30ActionTrainingStepError(f"{label} must be lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        raise Full30ActionTrainingStepError("unsigned receipt already has a digest")
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    if not isinstance(receipt, Mapping):
        raise Full30ActionTrainingStepError("training receipt must be a mapping")
    value = dict(receipt)
    digest = value.pop("receipt_digest", None)
    if type(digest) is not str or digest != object_sha256(value):
        raise Full30ActionTrainingStepError("training receipt digest differs")
    return canonical_json_bytes(dict(receipt))


def _tensor_payload_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().contiguous().to(device="cpu")
    storage = cpu.untyped_storage()
    expected = int(cpu.numel()) * int(cpu.element_size())
    if cpu.storage_offset() != 0 or int(storage.nbytes()) != expected:
        cpu = cpu.clone(memory_format=torch.contiguous_format)
        storage = cpu.untyped_storage()
    raw = bytes(storage)
    if len(raw) != expected:
        raise Full30ActionTrainingStepError("tensor payload byte count differs")
    return raw


def tensor_sha256_v1(value: torch.Tensor, *, label: str) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Full30ActionTrainingStepError(
            f"{label} must be finite contiguous strided tensor"
        )
    metadata = canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
        }
    )
    digest = hashlib.sha256(b"full30-action-training-tensor-v1\x00")
    digest.update(struct.pack(">Q", len(metadata)))
    digest.update(metadata)
    digest.update(_tensor_payload_bytes(value))
    return digest.hexdigest()


def _nuisance_sha256(packet: learning_core.NuisancePacket) -> str:
    if not isinstance(packet, learning_core.NuisancePacket):
        raise Full30ActionTrainingStepError("nuisance packet type differs")
    return object_sha256(
        {
            "camera_unit_sha256": tensor_sha256_v1(
                packet.camera_unit, label="camera nuisance unit"
            ),
            "appearance_unit_sha256": tensor_sha256_v1(
                packet.appearance_unit, label="appearance nuisance unit"
            ),
            "camera_norm_sha256": tensor_sha256_v1(
                packet.camera_norm, label="camera nuisance raw norm"
            ),
            "appearance_norm_sha256": tensor_sha256_v1(
                packet.appearance_norm, label="appearance nuisance raw norm"
            ),
            "appearance_residual_ratio_sha256": tensor_sha256_v1(
                packet.appearance_residual_ratio,
                label="appearance nuisance residual ratio",
            ),
        }
    )


@dataclass(frozen=True)
class Full30RecordObjectiveAuthorityV1:
    row_id: str
    source_id: str
    branch: str
    teacher_cell_id: str
    sigma_index: int
    noise_seed: int
    teacher_unit: torch.Tensor = field(repr=False, compare=False)
    minimum_amplitude: torch.Tensor = field(repr=False, compare=False)
    nuisance_packet: learning_core.NuisancePacket = field(repr=False, compare=False)
    noop_target_velocity: torch.Tensor = field(repr=False, compare=False)
    teacher_unit_sha256: str
    minimum_amplitude_sha256: str
    minimum_amplitude_float32_le_sha256: str
    minimum_amplitude_bundle_digest: str
    minimum_amplitude_calibration_id: str
    nuisance_packet_sha256: str
    noop_target_sha256: str
    data_teacher_authority_manifest_sha256: str
    amplitude_authority_manifest_sha256: str
    authority_digest: str


@dataclass(frozen=True)
class Full30LocalMicroRecordV1:
    scheduled: learning_core.ScheduledActionPair
    runtime_record: runtime_core.Full30ActionRecordV1 = field(
        repr=False, compare=False
    )
    objective: Full30RecordObjectiveAuthorityV1 = field(
        repr=False, compare=False
    )


@dataclass(frozen=True)
class GradientCollectiveRequestV1:
    phase: str
    scope: str
    sequence_index: int
    rank: int
    group_ranks: tuple[int, ...]
    gradients: dict[str, torch.Tensor] = field(repr=False, compare=False)


@dataclass(frozen=True)
class WorldConsensusRequestV1:
    phase: str
    rank: int
    digest: str
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Full30ActionTrainingStepResultV1:
    receipt: Mapping[str, Any]
    optimizer_receipt: Mapping[str, Any]


GradientMeanCallback = Callable[[GradientCollectiveRequestV1], Mapping[str, Any]]
WorldConsensusCallback = Callable[[WorldConsensusRequestV1], Mapping[str, Any]]
OptimizerAllReduceSum = Callable[[torch.Tensor], Optional[torch.Tensor]]
AutocastContext = Callable[[], ContextManager[Any]]


def objective_authority_digest_v1(
    *,
    row_id: str,
    source_id: str,
    branch: str,
    teacher_cell_id: str,
    sigma_index: int,
    noise_seed: int,
    teacher_unit_sha256: str,
    minimum_amplitude_sha256: str,
    minimum_amplitude_float32_le_sha256: str,
    minimum_amplitude_bundle_digest: str,
    minimum_amplitude_calibration_id: str,
    nuisance_packet_sha256: str,
    noop_target_sha256: str,
    data_teacher_authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
) -> str:
    value = {
        "schema_version": OBJECTIVE_AUTHORITY_SCHEMA_VERSION,
        "row_id": row_id,
        "source_id": source_id,
        "branch": branch,
        "teacher_cell_id": teacher_cell_id,
        "sigma_index": sigma_index,
        "noise_seed": noise_seed,
        "teacher_unit_sha256": _sha256(
            teacher_unit_sha256, label="teacher unit SHA"
        ),
        "minimum_amplitude_sha256": _sha256(
            minimum_amplitude_sha256, label="minimum amplitude SHA"
        ),
        "minimum_amplitude_float32_le_sha256": _sha256(
            minimum_amplitude_float32_le_sha256,
            label="minimum amplitude raw float32 SHA",
        ),
        "minimum_amplitude_bundle_digest": _sha256(
            minimum_amplitude_bundle_digest,
            label="minimum amplitude calibration bundle digest",
        ),
        "minimum_amplitude_calibration_id": minimum_amplitude_calibration_id,
        "nuisance_packet_sha256": _sha256(
            nuisance_packet_sha256, label="nuisance packet SHA"
        ),
        "noop_target_sha256": _sha256(
            noop_target_sha256, label="noop target SHA"
        ),
        "data_teacher_authority_manifest_sha256": _sha256(
            data_teacher_authority_manifest_sha256,
            label="data/teacher authority manifest SHA",
        ),
        "amplitude_authority_manifest_sha256": _sha256(
            amplitude_authority_manifest_sha256,
            label="amplitude authority manifest SHA",
        ),
    }
    if (
        any(type(value[name]) is not str or not value[name] for name in (
            "row_id", "source_id", "branch", "teacher_cell_id"
        ))
        or branch not in learning_core.BRANCHES
        or type(minimum_amplitude_calibration_id) is not str
        or not minimum_amplitude_calibration_id
        or "\x00" in minimum_amplitude_calibration_id
        or type(sigma_index) is not int
        or sigma_index not in learning_core.SIGMA_INDICES
        or type(noise_seed) is not int
        or not 0 <= noise_seed < 2**64
    ):
        raise Full30ActionTrainingStepError("objective coordinate differs")
    return object_sha256(value)


def seal_record_objective_authority_v1(
    *,
    row_id: str,
    source_id: str,
    branch: str,
    teacher_cell_id: str,
    sigma_index: int,
    noise_seed: int,
    teacher_unit: torch.Tensor,
    minimum_amplitude: torch.Tensor,
    minimum_amplitude_float32_le_sha256: str,
    minimum_amplitude_bundle_digest: str,
    minimum_amplitude_calibration_id: str,
    nuisance_packet: learning_core.NuisancePacket,
    noop_target_velocity: torch.Tensor,
    data_teacher_authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
) -> Full30RecordObjectiveAuthorityV1:
    teacher_sha = tensor_sha256_v1(teacher_unit, label="teacher unit")
    minimum_sha = tensor_sha256_v1(
        minimum_amplitude, label="minimum amplitude"
    )
    if (
        minimum_amplitude.dtype != torch.float32
        or tuple(int(item) for item in minimum_amplitude.shape) != (1,)
    ):
        raise Full30ActionTrainingStepError(
            "minimum amplitude must be one exact float32 scalar"
        )
    minimum_raw_sha = hashlib.sha256(
        struct.pack("<f", float(minimum_amplitude.detach().item()))
    ).hexdigest()
    if minimum_raw_sha != _sha256(
        minimum_amplitude_float32_le_sha256,
        label="minimum amplitude raw float32 SHA",
    ):
        raise Full30ActionTrainingStepError(
            "minimum amplitude raw float32 authority differs"
        )
    nuisance_sha = _nuisance_sha256(nuisance_packet)
    noop_sha = tensor_sha256_v1(noop_target_velocity, label="noop target")
    digest = objective_authority_digest_v1(
        row_id=row_id,
        source_id=source_id,
        branch=branch,
        teacher_cell_id=teacher_cell_id,
        sigma_index=sigma_index,
        noise_seed=noise_seed,
        teacher_unit_sha256=teacher_sha,
        minimum_amplitude_sha256=minimum_sha,
        minimum_amplitude_float32_le_sha256=minimum_raw_sha,
        minimum_amplitude_bundle_digest=minimum_amplitude_bundle_digest,
        minimum_amplitude_calibration_id=minimum_amplitude_calibration_id,
        nuisance_packet_sha256=nuisance_sha,
        noop_target_sha256=noop_sha,
        data_teacher_authority_manifest_sha256=(
            data_teacher_authority_manifest_sha256
        ),
        amplitude_authority_manifest_sha256=amplitude_authority_manifest_sha256,
    )
    return Full30RecordObjectiveAuthorityV1(
        row_id=row_id,
        source_id=source_id,
        branch=branch,
        teacher_cell_id=teacher_cell_id,
        sigma_index=sigma_index,
        noise_seed=noise_seed,
        teacher_unit=teacher_unit,
        minimum_amplitude=minimum_amplitude,
        nuisance_packet=nuisance_packet,
        noop_target_velocity=noop_target_velocity,
        teacher_unit_sha256=teacher_sha,
        minimum_amplitude_sha256=minimum_sha,
        minimum_amplitude_float32_le_sha256=minimum_raw_sha,
        minimum_amplitude_bundle_digest=_sha256(
            minimum_amplitude_bundle_digest,
            label="minimum amplitude calibration bundle digest",
        ),
        minimum_amplitude_calibration_id=minimum_amplitude_calibration_id,
        nuisance_packet_sha256=nuisance_sha,
        noop_target_sha256=noop_sha,
        data_teacher_authority_manifest_sha256=(
            data_teacher_authority_manifest_sha256
        ),
        amplitude_authority_manifest_sha256=amplitude_authority_manifest_sha256,
        authority_digest=digest,
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _scheduled_row(value: learning_core.ScheduledActionPair) -> dict[str, Any]:
    if not isinstance(value, learning_core.ScheduledActionPair):
        raise Full30ActionTrainingStepError("local scheduled row type differs")
    return {
        "global_index": value.global_index,
        "epoch": value.epoch,
        "update": value.update,
        "microbatch": value.microbatch,
        "dp_rank": value.dp_rank,
        "sigma_index": value.sigma_index,
        "noise_seed": value.noise_seed,
        "row": {
            "row_id": value.row.row_id,
            "source_id": value.row.source_id,
            "branch": value.row.branch,
            "teacher_cell_id": value.row.teacher_cell_id,
        },
    }


def _optimizer_parameters(
    optimizer: optimizer_core.Full30ActionFirstOptimizerV1,
) -> tuple[tuple[str, torch.Tensor], ...]:
    names = getattr(optimizer, "canonical_parameter_names", None)
    parameters = getattr(optimizer, "_parameters", None)
    if (
        not isinstance(names, tuple)
        or not names
        or tuple(sorted(names, key=lambda item: item.encode("utf-8"))) != names
        or not isinstance(parameters, Mapping)
        or set(parameters) != set(names)
    ):
        raise Full30ActionTrainingStepError(
            "optimizer canonical live parameter inventory differs"
        )
    result: list[tuple[str, torch.Tensor]] = []
    for name in names:
        parameter = parameters[name]
        if (
            not isinstance(parameter, torch.Tensor)
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or not parameter.requires_grad
            or not parameter.is_leaf
            or not bool(torch.isfinite(parameter).all().item())
        ):
            raise Full30ActionTrainingStepError(
                f"live trainable parameter differs: {name}"
            )
        result.append((name, parameter))
    return tuple(result)


def _detached_fp32(
    value: torch.Tensor,
    *,
    label: str,
    shape: Optional[tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or (shape is not None and tuple(int(item) for item in value.shape) != shape)
        or (device is not None and value.device != device)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Full30ActionTrainingStepError(
            f"{label} must be matching finite detached contiguous FP32"
        )


def _validate_objective(
    objective: Full30RecordObjectiveAuthorityV1,
    *,
    scheduled: Mapping[str, Any],
    runtime_record: runtime_core.Full30ActionRecordV1,
    device: torch.device,
) -> None:
    if not isinstance(objective, Full30RecordObjectiveAuthorityV1):
        raise Full30ActionTrainingStepError("objective authority type differs")
    nested = scheduled["row"]
    expected_coordinates = (
        nested["row_id"],
        nested["source_id"],
        nested["branch"],
        nested["teacher_cell_id"],
        scheduled["sigma_index"],
        scheduled["noise_seed"],
    )
    actual_coordinates = (
        objective.row_id,
        objective.source_id,
        objective.branch,
        objective.teacher_cell_id,
        objective.sigma_index,
        objective.noise_seed,
    )
    if actual_coordinates != expected_coordinates:
        raise Full30ActionTrainingStepError(
            "objective authority and scheduled row coordinates differ"
        )
    _detached_fp32(
        objective.teacher_unit,
        label="teacher unit",
        shape=(1, learning_core.LATENT_PHASES, learning_core.QUOTIENT_WIDTH),
        device=device,
    )
    teacher_norm = torch.linalg.vector_norm(objective.teacher_unit.reshape(1, -1), dim=1)
    if not bool(
        torch.allclose(
            teacher_norm,
            torch.ones_like(teacher_norm),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise Full30ActionTrainingStepError("teacher unit is not normalized")
    _detached_fp32(
        objective.minimum_amplitude,
        label="sealed minimum amplitude",
        shape=(1,),
        device=device,
    )
    if not bool((objective.minimum_amplitude > 0.0).all().item()):
        raise Full30ActionTrainingStepError("sealed minimum amplitude is not positive")
    observed_minimum_raw_sha = hashlib.sha256(
        struct.pack("<f", float(objective.minimum_amplitude.item()))
    ).hexdigest()
    if (
        observed_minimum_raw_sha
        != objective.minimum_amplitude_float32_le_sha256
    ):
        raise Full30ActionTrainingStepError(
            "sealed minimum amplitude raw float32 bytes changed"
        )
    spatial_shape = tuple(int(item) for item in runtime_record.spatial_shape)
    _detached_fp32(
        objective.noop_target_velocity,
        label="epsilon-minus-source noop target",
        shape=spatial_shape,
        device=device,
    )
    if not isinstance(objective.nuisance_packet, learning_core.NuisancePacket):
        raise Full30ActionTrainingStepError("nuisance packet type differs")
    for label, value in (
        ("camera nuisance unit", objective.nuisance_packet.camera_unit),
        ("appearance nuisance unit", objective.nuisance_packet.appearance_unit),
    ):
        _detached_fp32(
            value,
            label=label,
            shape=(1, learning_core.LATENT_PHASES, learning_core.QUOTIENT_WIDTH),
            device=device,
        )
    for label, value in (
        ("camera nuisance raw norm", objective.nuisance_packet.camera_norm),
        ("appearance nuisance raw norm", objective.nuisance_packet.appearance_norm),
        (
            "appearance nuisance residual ratio",
            objective.nuisance_packet.appearance_residual_ratio,
        ),
    ):
        _detached_fp32(
            value,
            label=label,
            shape=(1,),
            device=device,
        )
    camera_flat = objective.nuisance_packet.camera_unit.reshape(1, -1)
    appearance_flat = objective.nuisance_packet.appearance_unit.reshape(1, -1)
    camera_unit_norm = torch.linalg.vector_norm(camera_flat, dim=1)
    appearance_unit_norm = torch.linalg.vector_norm(appearance_flat, dim=1)
    nuisance_dot = torch.sum(camera_flat * appearance_flat, dim=1).abs()
    if (
        not bool(
            torch.allclose(
                camera_unit_norm,
                torch.ones_like(camera_unit_norm),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )
        or not bool(
            torch.allclose(
                appearance_unit_norm,
                torch.ones_like(appearance_unit_norm),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )
        or bool((nuisance_dot > 1.0e-5).any().item())
        or bool(
            (
                objective.nuisance_packet.camera_norm
                <= learning_core.MIN_NUISANCE_NORM
            ).any().item()
        )
        or bool(
            (
                objective.nuisance_packet.appearance_norm
                <= learning_core.MIN_NUISANCE_NORM
            ).any().item()
        )
        or bool(
            (
                objective.nuisance_packet.appearance_residual_ratio
                <= learning_core.MIN_APPEARANCE_RESIDUAL_RATIO
            ).any().item()
        )
    ):
        raise Full30ActionTrainingStepError(
            "sealed nuisance packet unit/orthogonality gate differs"
        )
    observed = {
        "teacher_unit_sha256": tensor_sha256_v1(
            objective.teacher_unit, label="teacher unit"
        ),
        "minimum_amplitude_sha256": tensor_sha256_v1(
            objective.minimum_amplitude, label="minimum amplitude"
        ),
        "nuisance_packet_sha256": _nuisance_sha256(objective.nuisance_packet),
        "noop_target_sha256": tensor_sha256_v1(
            objective.noop_target_velocity, label="noop target"
        ),
    }
    declared = {
        name: getattr(objective, name) for name in observed
    }
    if observed != declared:
        raise Full30ActionTrainingStepError(
            "objective authority tensor bytes changed after sealing"
        )
    expected_digest = objective_authority_digest_v1(
        row_id=objective.row_id,
        source_id=objective.source_id,
        branch=objective.branch,
        teacher_cell_id=objective.teacher_cell_id,
        sigma_index=objective.sigma_index,
        noise_seed=objective.noise_seed,
        teacher_unit_sha256=objective.teacher_unit_sha256,
        minimum_amplitude_sha256=objective.minimum_amplitude_sha256,
        minimum_amplitude_float32_le_sha256=(
            objective.minimum_amplitude_float32_le_sha256
        ),
        minimum_amplitude_bundle_digest=(
            objective.minimum_amplitude_bundle_digest
        ),
        minimum_amplitude_calibration_id=(
            objective.minimum_amplitude_calibration_id
        ),
        nuisance_packet_sha256=objective.nuisance_packet_sha256,
        noop_target_sha256=objective.noop_target_sha256,
        data_teacher_authority_manifest_sha256=(
            objective.data_teacher_authority_manifest_sha256
        ),
        amplitude_authority_manifest_sha256=(
            objective.amplitude_authority_manifest_sha256
        ),
    )
    if objective.authority_digest != expected_digest:
        raise Full30ActionTrainingStepError("objective authority digest differs")


def _validate_update_inputs(
    *,
    arm: str,
    rank: int,
    update_index: int,
    full_schedule: Sequence[Any],
    local_records: Sequence[Full30LocalMicroRecordV1],
    optimizer: optimizer_core.Full30ActionFirstOptimizerV1,
    test_only_allow_small_capacity: bool,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Full30LocalMicroRecordV1, ...],
    tuple[tuple[str, torch.Tensor], ...],
    Mapping[str, Any],
    str,
    str,
    str,
]:
    if optimizer.__class__ is not optimizer_core.Full30ActionFirstOptimizerV1:
        raise Full30ActionTrainingStepError(
            "training step requires actual Full30ActionFirstOptimizerV1"
        )
    if arm not in runtime_core.ARMS:
        raise Full30ActionTrainingStepError("formal arm differs")
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise Full30ActionTrainingStepError("WORLD8 rank differs")
    if type(update_index) is not int or not 0 <= update_index < learning_core.MAX_UPDATES:
        raise Full30ActionTrainingStepError("formal update index differs")
    if getattr(optimizer, "update_count", None) != update_index:
        raise Full30ActionTrainingStepError(
            "optimizer update count and schedule cursor differ"
        )
    canonical_schedule = checkpoint_core.canonical_schedule_v2(full_schedule)
    schedule_full, prefix_before = checkpoint_core.schedule_digests_v2(
        canonical_schedule, update_index
    )
    _same_full, prefix_after = checkpoint_core.schedule_digests_v2(
        canonical_schedule, update_index + 1
    )
    records = tuple(local_records)
    if (
        len(records) != LOCAL_MICRO_RECORDS
        or any(not isinstance(item, Full30LocalMicroRecordV1) for item in records)
    ):
        raise Full30ActionTrainingStepError(
            "each WORLD8 rank requires exactly four local micro records"
        )
    named = _optimizer_parameters(optimizer)
    if any(parameter.grad is not None for _, parameter in named):
        raise Full30ActionTrainingStepError(
            "live gradients must be absent at update entry"
        )
    inventory = checkpoint_core.inventory_identity_v2(
        optimizer,
        test_only_allow_small_capacity=test_only_allow_small_capacity,
    )
    device = named[0][1].device
    dp_rank = rank // SP_SIZE
    for microbatch, item in enumerate(records):
        expected_index = update_index * GLOBAL_BATCH + microbatch * DP_SIZE + dp_rank
        expected = _plain_json(canonical_schedule[expected_index])
        scheduled = _scheduled_row(item.scheduled)
        if scheduled != expected:
            raise Full30ActionTrainingStepError(
                "local schedule row does not match rank DP2 ownership"
            )
        runtime_record = item.runtime_record
        nested = expected["row"]
        if (
            not isinstance(runtime_record, runtime_core.Full30ActionRecordV1)
            or runtime_record.row_id != nested["row_id"]
            or runtime_record.source_iid != nested["source_id"]
            or runtime_record.branch != nested["branch"]
        ):
            raise Full30ActionTrainingStepError(
                "runtime record identity differs from formal schedule"
            )
        _validate_objective(
            item.objective,
            scheduled=expected,
            runtime_record=runtime_record,
            device=device,
        )
    return (
        canonical_schedule,
        records,
        named,
        _plain_json(inventory),
        schedule_full,
        prefix_before,
        prefix_after,
    )


def _named_tensor_digest(
    values: Mapping[str, torch.Tensor],
    *,
    names: Sequence[str],
    domain: str,
) -> str:
    digest = hashlib.sha256(domain.encode("ascii") + b"\x00")
    if set(values) != set(names):
        raise Full30ActionTrainingStepError("named tensor digest inventory differs")
    for name in names:
        encoded = name.encode("utf-8")
        value = values[name]
        tensor_digest = tensor_sha256_v1(value, label=f"{domain} {name}")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(bytes.fromhex(tensor_digest))
    return digest.hexdigest()


def _optimizer_state_digests(
    optimizer: optimizer_core.Full30ActionFirstOptimizerV1,
) -> tuple[str, str]:
    named = _optimizer_parameters(optimizer)
    names = tuple(name for name, _ in named)
    parameters = {name: value for name, value in named}
    moments = getattr(optimizer, "_second_moments", None)
    if not isinstance(moments, Mapping) or set(moments) != set(names):
        raise Full30ActionTrainingStepError("optimizer second-moment state differs")
    for name in names:
        value = moments[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.shape != parameters[name].shape
            or value.device != parameters[name].device
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
            or bool((value < 0.0).any().item())
        ):
            raise Full30ActionTrainingStepError(
                f"optimizer second moment differs: {name}"
            )
    return (
        _named_tensor_digest(
            parameters,
            names=names,
            domain="full30-action-training-parameters-v1",
        ),
        _named_tensor_digest(
            moments,
            names=names,
            domain="full30-action-training-second-moments-v1",
        ),
    )


def _clear_live_gradients(named: Sequence[tuple[str, torch.Tensor]]) -> None:
    for _name, parameter in named:
        parameter.grad = None


def _snapshot_live_gradients(
    named: Sequence[tuple[str, torch.Tensor]], *, phase: str
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    parameter_storages = {
        (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
        for _, parameter in named
    }
    for name, parameter in named:
        gradient = parameter.grad
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.dtype != torch.float32
            or gradient.layout != torch.strided
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or not gradient.is_contiguous()
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise Full30ActionTrainingStepError(
                f"{phase} live gradient is missing or malformed: {name}"
            )
        candidate = gradient.detach().clone(memory_format=torch.contiguous_format)
        storage = (str(candidate.device), int(candidate.untyped_storage().data_ptr()))
        if storage in parameter_storages:
            raise Full30ActionTrainingStepError(
                f"{phase} gradient aliases parameter storage: {name}"
            )
        snapshot[name] = candidate
    return snapshot


def _live_gradient_mapping(
    named: Sequence[tuple[str, torch.Tensor]], *, phase: str
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter in named:
        gradient = parameter.grad
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.dtype != torch.float32
            or gradient.layout != torch.strided
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or not gradient.is_contiguous()
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise Full30ActionTrainingStepError(
                f"{phase} live gradient is missing or malformed: {name}"
            )
        result[name] = gradient
    return result


def expected_gradient_collective_receipt_v1(
    request: GradientCollectiveRequestV1,
) -> Mapping[str, Any]:
    if not isinstance(request, GradientCollectiveRequestV1):
        raise Full30ActionTrainingStepError("gradient collective request type differs")
    if request.phase not in {"action", "noop"} or request.scope not in {"SP4", "DP2"}:
        raise Full30ActionTrainingStepError("gradient collective phase/scope differs")
    participant_count = SP_SIZE if request.scope == "SP4" else DP_SIZE
    if len(request.group_ranks) != participant_count:
        raise Full30ActionTrainingStepError("gradient collective group size differs")
    value = {
        "schema_version": GRADIENT_COLLECTIVE_SCHEMA_VERSION,
        "phase": request.phase,
        "scope": request.scope,
        "sequence_index": request.sequence_index,
        "rank": request.rank,
        "group_ranks": list(request.group_ranks),
        "participant_count": participant_count,
        "reduction": "mean",
        "divisor": participant_count,
        "gradients_mutated_in_place": True,
    }
    return {**value, "collective_digest": object_sha256(value)}


def _validate_gradient_mapping(
    gradients: Mapping[str, torch.Tensor],
    *,
    named: Sequence[tuple[str, torch.Tensor]],
    phase: str,
) -> None:
    if set(gradients) != {name for name, _ in named}:
        raise Full30ActionTrainingStepError(f"{phase} gradient names differ")
    parameter_storages = {
        (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
        for _, parameter in named
    }
    gradient_storages: set[tuple[str, int]] = set()
    for name, parameter in named:
        value = gradients[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.shape != parameter.shape
            or value.device != parameter.device
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise Full30ActionTrainingStepError(
                f"{phase} synchronized gradient differs: {name}"
            )
        storage = (str(value.device), int(value.untyped_storage().data_ptr()))
        if storage in parameter_storages or storage in gradient_storages:
            raise Full30ActionTrainingStepError(
                f"{phase} synchronized gradient storage aliases: {name}"
            )
        gradient_storages.add(storage)


def _synchronize_gradient_phase(
    *,
    phase: str,
    rank: int,
    gradients: dict[str, torch.Tensor],
    named: Sequence[tuple[str, torch.Tensor]],
    gradient_mean: GradientMeanCallback,
    first_sequence_index: int,
) -> tuple[dict[str, torch.Tensor], tuple[Mapping[str, Any], ...]]:
    if not callable(gradient_mean):
        raise Full30ActionTrainingStepError("gradient mean callback is absent")
    names = tuple(name for name, _ in named)
    _validate_gradient_mapping(gradients, named=named, phase=phase)
    dp_rank = rank // SP_SIZE
    sp_rank = rank % SP_SIZE
    groups = (
        ("SP4", tuple(dp_rank * SP_SIZE + item for item in range(SP_SIZE))),
        ("DP2", tuple(sp_rank + item * SP_SIZE for item in range(DP_SIZE))),
    )
    receipts: list[Mapping[str, Any]] = []
    for offset, (scope, group_ranks) in enumerate(groups):
        identities = {
            name: (
                id(gradients[name]),
                int(gradients[name].untyped_storage().data_ptr()),
            )
            for name in names
        }
        request = GradientCollectiveRequestV1(
            phase=phase,
            scope=scope,
            sequence_index=first_sequence_index + offset,
            rank=rank,
            group_ranks=group_ranks,
            gradients=gradients,
        )
        try:
            transport_receipt = gradient_mean(request)
        except Exception as error:
            raise Full30ActionTrainingStepError(
                f"{phase} {scope} gradient collective failed"
            ) from error
        expected = expected_gradient_collective_receipt_v1(request)
        if not isinstance(transport_receipt, Mapping) or dict(transport_receipt) != dict(expected):
            raise Full30ActionTrainingStepError(
                f"{phase} {scope} gradient collective receipt differs"
            )
        if identities != {
            name: (
                id(gradients[name]),
                int(gradients[name].untyped_storage().data_ptr()),
            )
            for name in names
        }:
            raise Full30ActionTrainingStepError(
                f"{phase} {scope} collective replaced gradient storage"
            )
        _validate_gradient_mapping(gradients, named=named, phase=phase)
        receipts.append(
            _seal(
                {
                    "transport": dict(transport_receipt),
                    "gradient_object_and_storage_identity_preserved": True,
                    "post_collective_finite_contiguous_fp32": True,
                }
            )
        )
    return gradients, tuple(receipts)


def _gradient_l2_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=next(iter(gradients.values())).device)
    for value in gradients.values():
        total += torch.sum(value.double().square())
    result = float(torch.sqrt(total).item())
    if not math.isfinite(result):
        raise Full30ActionTrainingStepError("gradient L2 norm is non-finite")
    return result


def _action_gradient_gate(
    *,
    gradients: Mapping[str, torch.Tensor],
    update_index: int,
    test_only_allow_small_capacity: bool,
) -> Mapping[str, Any]:
    lora_a: list[tuple[str, float]] = []
    lora_b: list[tuple[str, float]] = []
    typed: dict[str, list[tuple[str, float]]] = {
        "source": [],
        "target": [],
        "role": [],
    }
    unclassified: list[str] = []
    for name in sorted(gradients, key=lambda item: item.encode("utf-8")):
        value = gradients[name]
        norm = float(torch.linalg.vector_norm(value.double()).item())
        if not math.isfinite(norm):
            raise Full30ActionTrainingStepError(
                f"action gradient norm is non-finite: {name}"
            )
        match = _LORA.search(name)
        if match is not None:
            (lora_a if match.group("factor") == "A" else lora_b).append(
                (name, norm)
            )
        elif _SOURCE_TYPED.search(name) is not None:
            typed["source"].append((name, norm))
        elif _TARGET_TYPED.search(name) is not None:
            typed["target"].append((name, norm))
        elif _ROLE_TYPED.search(name) is not None:
            typed["role"].append((name, norm))
        else:
            unclassified.append(name)
    if (
        unclassified
        or not lora_a
        or not lora_b
        or any(not values for values in typed.values())
    ):
        raise Full30ActionTrainingStepError(
            "action gradient affine/typed inventory is incomplete"
        )
    if not test_only_allow_small_capacity and (
        len(lora_a) != 240
        or len(lora_b) != 240
        or len(typed["source"]) != 2
        or len(typed["target"]) != 2
        or len(typed["role"]) != 1
    ):
        raise Full30ActionTrainingStepError(
            "production action gradient coverage is not full30"
        )
    required_lora = lora_b if update_index == 0 else [*lora_a, *lora_b]
    if any(norm <= MIN_GRADIENT_NORM for _name, norm in required_lora):
        stage = "u1 B" if update_index == 0 else "u2+ A/B"
        raise Full30ActionTrainingStepError(
            f"{stage} action gradient coverage gate failed"
        )
    typed_norms = {
        group: math.sqrt(sum(norm * norm for _name, norm in values))
        for group, values in typed.items()
    }
    if any(norm <= MIN_GRADIENT_NORM for norm in typed_norms.values()):
        raise Full30ActionTrainingStepError(
            "typed source/target/role action gradient gate failed"
        )
    all_norm_rows = {
        "lora_A": [[name, norm] for name, norm in lora_a],
        "lora_B": [[name, norm] for name, norm in lora_b],
        "typed": {
            group: [[name, norm] for name, norm in values]
            for group, values in typed.items()
        },
    }
    value = {
        "schema_version": "bernini-full30-action-gradient-coverage-v1",
        "update_count_before": update_index,
        "gate_stage": "u1-B-plus-typed" if update_index == 0 else "u2+-A-B-plus-typed",
        "lora_A_tensor_count": len(lora_a),
        "lora_B_tensor_count": len(lora_b),
        "lora_A_min_norm": min(norm for _name, norm in lora_a),
        "lora_B_min_norm": min(norm for _name, norm in lora_b),
        "typed_group_norms": typed_norms,
        "factor_norms_sha256": object_sha256(all_norm_rows),
        "minimum_required_norm": MIN_GRADIENT_NORM,
        "parameter_times_zero_used": False,
        "passed": True,
    }
    return {**value, "gate_digest": object_sha256(value)}


def _validate_phase_receipt(
    receipt: Mapping[str, Any],
    *,
    phase: str,
    record: runtime_core.Full30ActionRecordV1,
) -> str:
    try:
        runtime_core.canonical_receipt_bytes(receipt)
    except Exception as error:
        raise Full30ActionTrainingStepError(
            f"{phase} runtime phase receipt is invalid"
        ) from error
    expected_count = 24 if phase == "action" else 8
    expected_slots = (
        ["trainable_branch", "frozen_noop", "frozen_branch"]
        if phase == "action"
        else ["trainable_noop"]
    )
    plan = receipt.get("phase_evaluation_plan")
    if (
        receipt.get("schema_version") != runtime_core.PHASE_RECEIPT_SCHEMA_VERSION
        or receipt.get("runtime_schema_version") != runtime_core.SCHEMA_VERSION
        or receipt.get("phase") != phase
        or receipt.get("row_id") != record.row_id
        or receipt.get("source_iid") != record.source_iid
        or receipt.get("branch") != record.branch
        or not isinstance(plan, Mapping)
        or plan.get("global_batch") != GLOBAL_BATCH
        or plan.get("evaluations_per_record") != (3 if phase == "action" else 1)
        or plan.get("global_physical_evaluation_count") != expected_count
        or plan.get("slots") != expected_slots
    ):
        raise Full30ActionTrainingStepError(
            f"{phase} runtime phase receipt contract differs"
        )
    binding = receipt.get("input_binding_digest")
    if type(binding) is not str or _SHA256.fullmatch(binding) is None:
        raise Full30ActionTrainingStepError(
            f"{phase} runtime input binding digest differs"
        )
    return binding


def _phase_output_tensor(
    value: torch.Tensor,
    *,
    label: str,
    shape: tuple[int, ...],
    trainable: bool,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
        or not value.is_contiguous()
        or tuple(int(item) for item in value.shape) != shape
        or not bool(torch.isfinite(value).all().item())
        or (trainable and (not value.requires_grad or value.grad_fn is None))
        or (
            not trainable
            and (value.requires_grad or value.grad_fn is not None)
        )
    ):
        raise Full30ActionTrainingStepError(
            f"{label} post-head velocity graph/tensor contract differs"
        )
    converted = value.float().contiguous()
    if trainable and (not converted.requires_grad or converted.grad_fn is None):
        raise Full30ActionTrainingStepError(f"{label} FP32 conversion lost graph")
    return converted


def expected_world_consensus_receipt_v1(
    request: WorldConsensusRequestV1,
) -> Mapping[str, Any]:
    if (
        not isinstance(request, WorldConsensusRequestV1)
        or request.phase != "pre_optimizer"
        or type(request.rank) is not int
        or not 0 <= request.rank < WORLD_SIZE
        or _SHA256.fullmatch(request.digest) is None
        or request.digest != object_sha256(_plain_json(request.payload))
    ):
        raise Full30ActionTrainingStepError("WORLD consensus request differs")
    value = {
        "schema_version": WORLD_CONSENSUS_SCHEMA_VERSION,
        "phase": request.phase,
        "world_size": WORLD_SIZE,
        "participant_count": WORLD_SIZE,
        "all_equal": True,
        "consensus_digest": request.digest,
    }
    return {**value, "receipt_digest": object_sha256(value)}


def _world_consensus(
    *,
    rank: int,
    payload: Mapping[str, Any],
    callback: WorldConsensusCallback,
) -> Mapping[str, Any]:
    if not callable(callback):
        raise Full30ActionTrainingStepError("WORLD consensus callback is absent")
    plain = _plain_json(payload)
    request = WorldConsensusRequestV1(
        phase="pre_optimizer",
        rank=rank,
        digest=object_sha256(plain),
        payload=plain,
    )
    expected = expected_world_consensus_receipt_v1(request)
    try:
        observed = callback(request)
    except Exception as error:
        raise Full30ActionTrainingStepError(
            "pre-optimizer WORLD8 consensus failed"
        ) from error
    if not isinstance(observed, Mapping) or dict(observed) != dict(expected):
        raise Full30ActionTrainingStepError(
            "pre-optimizer WORLD8 consensus differs"
        )
    return dict(observed)


def _loss_float(value: torch.Tensor, *, label: str) -> float:
    result = float(value.detach().item())
    if not math.isfinite(result):
        raise Full30ActionTrainingStepError(f"{label} is non-finite")
    return 0.0 if result == 0.0 else result


def _disjoint_gradient_storage(
    action: Mapping[str, torch.Tensor], noop: Mapping[str, torch.Tensor]
) -> None:
    action_storage = {
        (str(value.device), int(value.untyped_storage().data_ptr()))
        for value in action.values()
    }
    noop_storage = {
        (str(value.device), int(value.untyped_storage().data_ptr()))
        for value in noop.values()
    }
    if action_storage & noop_storage:
        raise Full30ActionTrainingStepError(
            "action and noop gradient snapshots share storage"
        )


def execute_full30_action_training_step_v1(
    *,
    runtime: Any,
    optimizer: optimizer_core.Full30ActionFirstOptimizerV1,
    arm: str,
    rank: int,
    update_index: int,
    full_schedule: Sequence[Any],
    local_records: Sequence[Full30LocalMicroRecordV1],
    gradient_mean: GradientMeanCallback,
    world_consensus: WorldConsensusCallback,
    optimizer_all_reduce_sum: OptimizerAllReduceSum,
    autocast_context: AutocastContext = nullcontext,
    test_only_allow_small_capacity: bool = False,
) -> Full30ActionTrainingStepResultV1:
    """Execute exactly one formal full30 WORLD8 update on one rank.

    The four local action graphs are consumed first.  Their live gradients are
    reduced SP4 then DP2 and only then snapshotted.  Live gradients are cleared
    before the optional four-record noop replay creates its independent graph.
    Exactly one registered optimizer transaction consumes the resulting global
    snapshots.
    """

    if (
        not test_only_allow_small_capacity
        and runtime.__class__ is not runtime_core.Full30ActionBranchRuntimeV1
    ):
        raise Full30ActionTrainingStepError(
            "production update requires actual Full30ActionBranchRuntimeV1"
        )
    if not callable(getattr(runtime, "execute_action_phase", None)) or not callable(
        getattr(runtime, "execute_noop_phase", None)
    ):
        raise Full30ActionTrainingStepError("runtime phase API is incomplete")
    if not callable(autocast_context):
        raise Full30ActionTrainingStepError("autocast context factory is absent")
    (
        canonical_schedule,
        records,
        named,
        inventory,
        schedule_full_sha,
        schedule_prefix_before,
        schedule_prefix_after,
    ) = _validate_update_inputs(
        arm=arm,
        rank=rank,
        update_index=update_index,
        full_schedule=full_schedule,
        local_records=local_records,
        optimizer=optimizer,
        test_only_allow_small_capacity=test_only_allow_small_capacity,
    )
    names = tuple(name for name, _ in named)
    update_group = [
        _plain_json(item)
        for item in canonical_schedule[
            update_index * GLOBAL_BATCH : (update_index + 1) * GLOBAL_BATCH
        ]
    ]
    update_group_digest = object_sha256(
        {
            "schedule_schema_version": checkpoint_core.SCHEDULE_SCHEMA_VERSION,
            "update": update_index,
            "flat_rows": update_group,
        }
    )
    physical_plan = runtime_core.physical_evaluation_plan_receipt_v1(arm)
    expected_physical = 24 if arm == "action-only" else 32
    if physical_plan.get("physical_evaluation_count") != expected_physical:
        raise Full30ActionTrainingStepError(
            "formal physical evaluation plan count differs"
        )
    parameters_before, moments_before = _optimizer_state_digests(optimizer)
    action_rows: list[Mapping[str, Any]] = []
    action_input_bindings: list[str] = []
    noop_rows: list[Mapping[str, Any]] = []
    action_gradients: Optional[dict[str, torch.Tensor]] = None
    noop_gradients: Optional[dict[str, torch.Tensor]] = None
    optimizer_called = False
    try:
        for local_index, item in enumerate(records):
            try:
                with autocast_context():
                    output = runtime.execute_action_phase(
                        record=item.runtime_record
                    )
            except Exception as error:
                raise Full30ActionTrainingStepError(
                    f"action phase runtime failed for local record {local_index}"
                ) from error
            if not isinstance(output, runtime_core.Full30ActionPhaseOutputsV1):
                raise Full30ActionTrainingStepError(
                    "action phase output type differs"
                )
            binding = _validate_phase_receipt(
                output.receipt,
                phase="action",
                record=item.runtime_record,
            )
            shape = tuple(int(value) for value in item.runtime_record.spatial_shape)
            trainable = _phase_output_tensor(
                output.trainable_branch_velocity,
                label="trainable branch",
                shape=shape,
                trainable=True,
            )
            frozen_noop = _phase_output_tensor(
                output.frozen_noop_velocity,
                label="frozen noop",
                shape=shape,
                trainable=False,
            )
            frozen_branch = _phase_output_tensor(
                output.frozen_branch_velocity,
                label="frozen same-mode branch",
                shape=shape,
                trainable=False,
            )
            trainable_sha = tensor_sha256_v1(
                trainable, label="trainable branch output"
            )
            frozen_noop_sha = tensor_sha256_v1(
                frozen_noop, label="frozen noop output"
            )
            frozen_branch_sha = tensor_sha256_v1(
                frozen_branch, label="frozen branch output"
            )
            student = learning_core.psiout_v1(
                (trainable - frozen_noop).contiguous(),
                item.objective.nuisance_packet,
            )
            same_mode = learning_core.psiout_v1(
                (frozen_branch - frozen_noop).contiguous(),
                item.objective.nuisance_packet,
            )
            same_mode_norm = torch.linalg.vector_norm(
                same_mode.reshape(int(same_mode.shape[0]), -1), dim=1
            ).detach().float().contiguous()
            amplitude_floor = torch.maximum(
                same_mode_norm, item.objective.minimum_amplitude
            ).detach().float().contiguous()
            loss = learning_core.paired_action_loss_v1(
                student,
                item.objective.teacher_unit,
                amplitude_floor,
            )
            (loss.total / float(LOCAL_MICRO_RECORDS)).backward()
            action_input_bindings.append(binding)
            action_rows.append(
                {
                    "local_index": local_index,
                    "global_index": item.scheduled.global_index,
                    "row_id": item.objective.row_id,
                    "source_id": item.objective.source_id,
                    "branch": item.objective.branch,
                    "teacher_cell_id": item.objective.teacher_cell_id,
                    "sigma_index": item.objective.sigma_index,
                    "noise_seed": item.objective.noise_seed,
                    "objective_authority_digest": item.objective.authority_digest,
                    "teacher_unit_sha256": item.objective.teacher_unit_sha256,
                    "nuisance_packet_sha256": (
                        item.objective.nuisance_packet_sha256
                    ),
                    "minimum_amplitude_sha256": (
                        item.objective.minimum_amplitude_sha256
                    ),
                    "minimum_amplitude_float32_le_sha256": (
                        item.objective.minimum_amplitude_float32_le_sha256
                    ),
                    "minimum_amplitude_bundle_digest": (
                        item.objective.minimum_amplitude_bundle_digest
                    ),
                    "minimum_amplitude_calibration_id": (
                        item.objective.minimum_amplitude_calibration_id
                    ),
                    "noop_target_sha256": item.objective.noop_target_sha256,
                    "data_teacher_authority_manifest_sha256": (
                        item.objective.data_teacher_authority_manifest_sha256
                    ),
                    "amplitude_authority_manifest_sha256": (
                        item.objective.amplitude_authority_manifest_sha256
                    ),
                    "runtime_phase_receipt_digest": output.receipt["receipt_digest"],
                    "runtime_input_binding_digest": binding,
                    "trainable_branch_sha256": trainable_sha,
                    "frozen_noop_sha256": frozen_noop_sha,
                    "frozen_branch_sha256": frozen_branch_sha,
                    "student_code_sha256": tensor_sha256_v1(
                        student, label="student quotient code"
                    ),
                    "same_mode_code_sha256": tensor_sha256_v1(
                        same_mode, label="same-mode quotient code"
                    ),
                    "same_mode_amplitude": _loss_float(
                        same_mode_norm[0], label="same-mode amplitude"
                    ),
                    "sealed_minimum_amplitude": _loss_float(
                        item.objective.minimum_amplitude[0],
                        label="sealed minimum amplitude",
                    ),
                    "final_amplitude_floor": _loss_float(
                        amplitude_floor[0], label="final amplitude floor"
                    ),
                    "action_loss": _loss_float(loss.total, label="action loss"),
                    "direction_loss": _loss_float(
                        loss.direction_mean, label="direction loss"
                    ),
                    "amplitude_loss": _loss_float(
                        loss.amplitude_mean, label="amplitude loss"
                    ),
                    "student_norm": _loss_float(
                        loss.student_norm[0], label="student norm"
                    ),
                }
            )
            del (
                output,
                trainable,
                frozen_noop,
                frozen_branch,
                student,
                same_mode,
                same_mode_norm,
                amplitude_floor,
                loss,
            )

        live_action = _live_gradient_mapping(named, phase="action")
        live_action, action_collective_receipts = _synchronize_gradient_phase(
            phase="action",
            rank=rank,
            gradients=live_action,
            named=named,
            gradient_mean=gradient_mean,
            first_sequence_index=0,
        )
        action_gradients = _snapshot_live_gradients(named, phase="action")
        action_gradient_digest = _named_tensor_digest(
            action_gradients,
            names=names,
            domain="full30-action-global-gradient-v1",
        )
        action_gradient_norm = _gradient_l2_norm(action_gradients)
        if action_gradient_norm <= MIN_GRADIENT_NORM:
            raise Full30ActionTrainingStepError(
                "global action gradient norm is degenerate"
            )
        gradient_gate = _action_gradient_gate(
            gradients=action_gradients,
            update_index=update_index,
            test_only_allow_small_capacity=test_only_allow_small_capacity,
        )
        _clear_live_gradients(named)

        noop_collective_receipts: tuple[Mapping[str, Any], ...] = ()
        noop_gradient_digest: Optional[str] = None
        noop_gradient_norm: Optional[float] = None
        if arm == "action+retain":
            for local_index, item in enumerate(records):
                try:
                    with autocast_context():
                        output = runtime.execute_noop_phase(
                            record=item.runtime_record
                        )
                except Exception as error:
                    raise Full30ActionTrainingStepError(
                        f"noop replay runtime failed for local record {local_index}"
                    ) from error
                if not isinstance(output, runtime_core.Full30NoopPhaseOutputsV1):
                    raise Full30ActionTrainingStepError(
                        "noop phase output type differs"
                    )
                binding = _validate_phase_receipt(
                    output.receipt,
                    phase="noop",
                    record=item.runtime_record,
                )
                if binding != action_input_bindings[local_index]:
                    raise Full30ActionTrainingStepError(
                        "noop replay input authority differs from action phase"
                    )
                shape = tuple(int(value) for value in item.runtime_record.spatial_shape)
                trainable_noop = _phase_output_tensor(
                    output.trainable_noop_velocity,
                    label="trainable noop replay",
                    shape=shape,
                    trainable=True,
                )
                raw_loss = torch.nn.functional.mse_loss(
                    trainable_noop,
                    item.objective.noop_target_velocity,
                    reduction="mean",
                )
                if not bool(torch.isfinite(raw_loss).item()):
                    raise Full30ActionTrainingStepError(
                        "epsilon-minus-source noop flow MSE is non-finite"
                    )
                (raw_loss / float(LOCAL_MICRO_RECORDS)).backward()
                noop_rows.append(
                    {
                        "local_index": local_index,
                        "global_index": item.scheduled.global_index,
                        "row_id": item.objective.row_id,
                        "runtime_phase_receipt_digest": output.receipt[
                            "receipt_digest"
                        ],
                        "runtime_input_binding_digest": binding,
                        "trainable_noop_sha256": tensor_sha256_v1(
                            trainable_noop, label="trainable noop output"
                        ),
                        "epsilon_minus_source_target_sha256": (
                            item.objective.noop_target_sha256
                        ),
                        "noop_flow_mse": _loss_float(
                            raw_loss, label="noop flow MSE"
                        ),
                    }
                )
                del output, trainable_noop, raw_loss
            live_noop = _live_gradient_mapping(named, phase="noop")
            live_noop, noop_collective_receipts = _synchronize_gradient_phase(
                phase="noop",
                rank=rank,
                gradients=live_noop,
                named=named,
                gradient_mean=gradient_mean,
                first_sequence_index=2,
            )
            noop_gradients = _snapshot_live_gradients(named, phase="noop")
            noop_gradient_digest = _named_tensor_digest(
                noop_gradients,
                names=names,
                domain="full30-noop-global-gradient-v1",
            )
            noop_gradient_norm = _gradient_l2_norm(noop_gradients)
            if noop_gradient_norm <= MIN_GRADIENT_NORM:
                raise Full30ActionTrainingStepError(
                    "global noop gradient norm is degenerate"
                )
            _disjoint_gradient_storage(action_gradients, noop_gradients)
            _clear_live_gradients(named)
        elif noop_rows or noop_gradients is not None:
            raise Full30ActionTrainingStepError(
                "action-only arm created a noop gradient"
            )

        if any(parameter.grad is not None for _, parameter in named):
            raise Full30ActionTrainingStepError(
                "live gradients survived two-phase snapshotting"
            )
        consensus_payload = {
            "schema_version": "bernini-full30-action-pre-optimizer-consensus-v1",
            "arm": arm,
            "update_count_before": update_index,
            "update_count_after": update_index + 1,
            "schedule_schema_version": checkpoint_core.SCHEDULE_SCHEMA_VERSION,
            "schedule_full_sha256": schedule_full_sha,
            "schedule_prefix_before_sha256": schedule_prefix_before,
            "schedule_prefix_after_sha256": schedule_prefix_after,
            "update_group_digest": update_group_digest,
            "inventory_sha256": inventory["inventory_sha256"],
            "parameters_before_sha256": parameters_before,
            "second_moments_before_sha256": moments_before,
            "action_gradient_sha256": action_gradient_digest,
            "noop_gradient_sha256": noop_gradient_digest,
            "gradient_gate_digest": gradient_gate["gate_digest"],
            "physical_evaluation_plan_digest": physical_plan["plan_digest"],
            "physical_evaluation_count": expected_physical,
        }
        consensus_receipt = _world_consensus(
            rank=rank,
            payload=consensus_payload,
            callback=world_consensus,
        )

        if not callable(optimizer_all_reduce_sum):
            raise Full30ActionTrainingStepError(
                "optimizer WORLD8 scalar reduction callback is absent"
            )
        scalar_shapes = ((6,), (4,))
        scalar_calls = 0

        def checked_optimizer_sum(value: torch.Tensor) -> Optional[torch.Tensor]:
            nonlocal scalar_calls
            if (
                scalar_calls >= len(scalar_shapes)
                or not isinstance(value, torch.Tensor)
                or value.dtype != torch.float64
                or tuple(int(item) for item in value.shape)
                != scalar_shapes[scalar_calls]
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise Full30ActionTrainingStepError(
                    "optimizer WORLD8 scalar collective order/shape differs"
                )
            scalar_calls += 1
            return optimizer_all_reduce_sum(value)

        optimizer_called = True
        optimizer_receipt = optimizer.step(
            action_gradients,
            noop_gradients=noop_gradients,
            world_size=WORLD_SIZE,
            all_reduce_sum=checked_optimizer_sum,
        )
        # The exact optimizer class seals and validates this receipt before it
        # commits candidate moments/update_count.  Every fallible local and
        # distributed gate is therefore above ``step`` or inside its own
        # rollback transaction; finalization below only copies already-sealed
        # canonical values and cannot invoke an external callback.
        optimizer_digests = optimizer_receipt["digests"]
        optimizer_statistics = optimizer_receipt["statistics"]
        receipt = _seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "orchestrator_schema_version": SCHEMA_VERSION,
                "status": "committed",
                "arm": arm,
                "rank": rank,
                "world_size": WORLD_SIZE,
                "dp_size": DP_SIZE,
                "sp_size": SP_SIZE,
                "dp_rank": rank // SP_SIZE,
                "sp_rank": rank % SP_SIZE,
                "update_count_before": update_index,
                "update_count_after": update_index + 1,
                "local_micro_record_count": LOCAL_MICRO_RECORDS,
                "global_batch": GLOBAL_BATCH,
                "schedule": {
                    "schema_version": checkpoint_core.SCHEDULE_SCHEMA_VERSION,
                    "full_sha256": schedule_full_sha,
                    "prefix_before_sha256": schedule_prefix_before,
                    "prefix_after_sha256": schedule_prefix_after,
                    "update_group_digest": update_group_digest,
                    "local_global_indices": [
                        item.scheduled.global_index for item in records
                    ],
                },
                "inventory": inventory,
                "records": action_rows,
                "noop_replay_records": noop_rows,
                "runtime": {
                    "runtime_schema_version": runtime_core.SCHEMA_VERSION,
                    "phase_receipt_schema_version": (
                        runtime_core.PHASE_RECEIPT_SCHEMA_VERSION
                    ),
                    "action_phase_local_call_count": LOCAL_MICRO_RECORDS,
                    "noop_phase_local_call_count": (
                        LOCAL_MICRO_RECORDS if arm == "action+retain" else 0
                    ),
                    "action_route_local_evaluation_count": (
                        LOCAL_MICRO_RECORDS * 3
                    ),
                    "noop_route_local_evaluation_count": (
                        LOCAL_MICRO_RECORDS if arm == "action+retain" else 0
                    ),
                    "formal_physical_evaluation_count": expected_physical,
                    "physical_evaluation_plan": physical_plan,
                    "strict_two_phase_replay": True,
                },
                "gradients": {
                    "phase_order": [
                        "four-action-phase-forwards-and-backwards",
                        "action-SP4-mean",
                        "action-DP2-mean",
                        "snapshot-global-action-gradient",
                        "zero-live-gradients",
                        *(
                            [
                                "four-independent-noop-replay-forwards-and-backwards",
                                "noop-SP4-mean",
                                "noop-DP2-mean",
                                "snapshot-global-noop-gradient",
                                "zero-live-gradients",
                            ]
                            if arm == "action+retain"
                            else []
                        ),
                        "one-transactional-global-optimizer-step",
                    ],
                    "reduction_order": ["SP4-mean", "DP2-mean"],
                    "action_sha256": action_gradient_digest,
                    "action_l2_norm": action_gradient_norm,
                    "action_collectives": [
                        dict(item) for item in action_collective_receipts
                    ],
                    "noop_sha256": noop_gradient_digest,
                    "noop_l2_norm": noop_gradient_norm,
                    "noop_collectives": [
                        dict(item) for item in noop_collective_receipts
                    ],
                    "coverage_gate": gradient_gate,
                    "action_noop_storage_disjoint": arm == "action+retain",
                    "live_gradients_cleared_before_noop": True,
                    "live_gradients_cleared_before_optimizer": True,
                },
                "world_consensus": consensus_receipt,
                "optimizer": {
                    "step_call_count": 1,
                    "world_scalar_all_reduce_call_count": 2,
                    "receipt_digest": optimizer_receipt["receipt_digest"],
                    "actual_action_descent_dot": optimizer_statistics[
                        "actual_action_descent_dot"
                    ],
                    "orchestrator_pre_parameters_sha256": parameters_before,
                    "orchestrator_pre_second_moments_sha256": moments_before,
                    "optimizer_parameters_before_sha256": optimizer_digests[
                        "parameters_before"
                    ],
                    "optimizer_parameters_after_sha256": optimizer_digests[
                        "parameters_after"
                    ],
                    "optimizer_second_moments_before_sha256": optimizer_digests[
                        "second_moment_before"
                    ],
                    "optimizer_second_moments_after_sha256": optimizer_digests[
                        "second_moment_after"
                    ],
                },
                "objective_contract": {
                    "direction_plus_same_mode_amplitude_floor": True,
                    "amplitude_floor_is_max_frozen_same_mode_and_sealed_minimum": True,
                    "noop_target_is_epsilon_minus_real_source": True,
                    "action_branch_regresses_source_trajectory": False,
                    "parameter_times_zero_used": False,
                    "synthetic_target_index1_bytes_read": False,
                },
            }
        )
        canonical_receipt_bytes(receipt)
        return Full30ActionTrainingStepResultV1(
            receipt=receipt,
            optimizer_receipt=dict(optimizer_receipt),
        )
    except Exception as error:
        _clear_live_gradients(named)
        if optimizer_called:
            parameters_now, moments_now = _optimizer_state_digests(optimizer)
            if optimizer.update_count == update_index and (
                parameters_now != parameters_before or moments_now != moments_before
            ):
                raise Full30ActionTrainingStepError(
                    "failed optimizer transaction did not roll back exact state"
                ) from error
        if isinstance(error, Full30ActionTrainingStepError):
            raise
        raise Full30ActionTrainingStepError(
            "full30 action training update failed closed"
        ) from error


__all__ = [
    "DP_SIZE",
    "Full30ActionTrainingStepError",
    "Full30ActionTrainingStepResultV1",
    "Full30LocalMicroRecordV1",
    "Full30RecordObjectiveAuthorityV1",
    "GLOBAL_BATCH",
    "GRADIENT_COLLECTIVE_SCHEMA_VERSION",
    "GradientCollectiveRequestV1",
    "LOCAL_MICRO_RECORDS",
    "OBJECTIVE_AUTHORITY_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "WORLD_CONSENSUS_SCHEMA_VERSION",
    "WORLD_SIZE",
    "WorldConsensusRequestV1",
    "canonical_receipt_bytes",
    "execute_full30_action_training_step_v1",
    "expected_gradient_collective_receipt_v1",
    "expected_world_consensus_receipt_v1",
    "objective_authority_digest_v1",
    "seal_record_objective_authority_v1",
    "tensor_sha256_v1",
]
