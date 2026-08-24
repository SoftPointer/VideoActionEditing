#!/usr/bin/env python3
"""Bounded Case01 VAE adapter for the frozen counterfactual patient carrier.

This module closes only the tensor ABI between one 81-frame VAE latent and
``object_instance_counterfactual_carrier_v1``.  Its sole file read is a stable
byte replay of that exact sibling carrier program.  It does not read media,
load a model, run diffusion, publish media, or judge visual success.

The old Case01 scaffold is consumed only as patient *geometry*.  Its revoked
``bone_removed_auxiliary_video`` authority is deliberately outside the
geometry projection and is never read.  Phase 10's overlapping ``(-1, 0)``
mapping is replaced by the preregistered disjoint ``(-3, 0)`` mapping.  No
other target or responsibility dilation is invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable

import object_instance_counterfactual_carrier_v1 as carrier


SCHEMA_VERSION = "bernini-case01-counterfactual-patient-vae-adapter-v1"
COMPILED_PLAN_SCHEMA_VERSION = (
    "bernini-case01-counterfactual-patient-21phase-plan-v1"
)
RECEIPT_SCHEMA_VERSION = (
    "bernini-case01-counterfactual-patient-vae-only-receipt-v1"
)
OUTPUT_STATUS = "VAE_ONLY_CARRIER_FEASIBILITY_PENDING_ALL81_REVIEW"

SCAFFOLD_SCHEMA_VERSION = "case01-oracle-object-trajectory-scaffold-v1"
EXPECTED_SCAFFOLD_GEOMETRY_SHA256 = (
    "2037982a36519301f962d041f55dcad847d0ed39b9d02e4c9c4b1b45995e130c"
)
CARRIER_PROGRAM_FILENAME = "object_instance_counterfactual_carrier_v1.py"
EXPECTED_CARRIER_PROGRAM_SHA256 = (
    "a6a2536177dc12ed41c05d8298d9f61a2b728d8e9585f77b16c7bc3c7fb73f4f"
)
EXPECTED_CARRIER_PROGRAM_SIZE = 42_952

LATENT_CHANNELS = 16
LATENT_PHASES = 21
LATENT_HEIGHT = 62
LATENT_WIDTH = 60
PATCH_ROWS = LATENT_HEIGHT // 2
PATCH_COLS = LATENT_WIDTH // 2
TOKENS_PER_PHASE = PATCH_ROWS * PATCH_COLS
PACKED_TOKEN_COUNT = LATENT_PHASES * TOKENS_PER_PHASE
PACKED_CHANNELS = LATENT_CHANNELS * 2 * 2
DECODED_FRAME_COUNT = 81
RGB_HEIGHT = 496
RGB_WIDTH = 480

SOURCE_PATIENT_TOKEN_COUNT = 377
LEGACY_RESPONSIBILITY_TOKEN_COUNT = 2_760
EXPANDED_RESPONSIBILITY_TOKEN_COUNT = 2_776
PHASE10_INDEX = 10
PHASE10_REPLACEMENT_SHIFT = (-3, 0)
PHASE10_SOURCE_TOKEN_COUNT = 19
PHASE10_REPLACEMENT_RESPONSIBILITY_COUNT = 153
PHASE10_NEW_RESPONSIBILITY_TOKEN_COUNT = 16

_LAYOUT_KEYS = (
    "latent_phases",
    "patch_rows",
    "patch_cols",
    "tokens_per_phase",
    "packed_token_count",
    "scheduler_target_packed_token",
)
_PHASE_GEOMETRY_KEYS = (
    "phase_index",
    "typed_stage",
    "bone_shift_patch_xy",
    "source_bone_tokens",
    "target_bone_tokens",
    "origin_clear_tokens",
    "bone_token_correspondence",
    "target_responsibility_tokens",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Case01CounterfactualPatientVaeAdapterError(RuntimeError):
    """Raised instead of returning an ambiguous VAE-only carrier result."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production has torch
        raise Case01CounterfactualPatientVaeAdapterError(
            "Case01 VAE carrier adapter requires torch"
        ) from error
    return torch


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _verify_frozen_carrier_program() -> dict[str, Any]:
    """Replay the exact sibling source that defines the imported carrier."""

    imported_value = getattr(carrier, "__file__", None)
    if type(imported_value) is not str or not imported_value:
        raise Case01CounterfactualPatientVaeAdapterError(
            "frozen carrier program path is unavailable"
        )
    adapter_path = Path(__file__).resolve(strict=True)
    expected_path = adapter_path.with_name(CARRIER_PROGRAM_FILENAME)
    imported_path = Path(imported_value)
    try:
        imported_resolved = imported_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise Case01CounterfactualPatientVaeAdapterError(
            "frozen carrier program path is absent"
        ) from error
    if (
        imported_path.is_symlink()
        or expected_path.is_symlink()
        or imported_resolved != expected_path
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "imported carrier is not the exact sibling program"
        )
    before = expected_path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "frozen carrier program topology differs"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(expected_path), flags)
    try:
        held_before = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(held_before):
            raise Case01CounterfactualPatientVaeAdapterError(
                "frozen carrier named/held identity differs"
            )
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        held_after = os.fstat(descriptor)
        named_after = expected_path.lstat()
        if (
            _file_identity(held_before) != _file_identity(held_after)
            or _file_identity(held_after) != _file_identity(named_after)
            or size != int(held_after.st_size)
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                "frozen carrier changed while replayed"
            )
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    if (
        observed_sha256 != EXPECTED_CARRIER_PROGRAM_SHA256
        or size != EXPECTED_CARRIER_PROGRAM_SIZE
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "frozen carrier program bytes differ"
        )
    return {
        "path": str(expected_path),
        "sha256": observed_sha256,
        "size": size,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _byte_equal(left: Any, right: Any) -> bool:
    torch = _torch()
    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _allocation_identity_and_interval(
    value: Any,
) -> tuple[
    tuple[str, int | None, int],
    tuple[str, int | None, int, int],
]:
    """Return a conservative backing identity and occupied byte interval.

    Storage identity catches ordinary views (including zero-stride expanded
    videos).  The data-pointer interval also catches independently wrapped
    storages that expose overlapping bytes.  Adapter-minted values admitted
    after callbacks are contiguous, so their interval is exact; for arbitrary
    caller/decoder views the interval is deliberately conservative and may
    reject rather than silently bless uncertain backing independence.
    """

    try:
        storage = value.untyped_storage()
    except AttributeError:  # pragma: no cover - old torch fallback
        storage = value.storage()
    allocation_pointer = int(storage.data_ptr())
    first = int(value.data_ptr())
    element_size = int(value.element_size())
    minimum_element_offset = 0
    maximum_element_offset = 0
    for dimension, stride in zip(value.shape, value.stride()):
        span = (int(dimension) - 1) * int(stride)
        minimum_element_offset += min(0, span)
        maximum_element_offset += max(0, span)
    start = first + minimum_element_offset * element_size
    end = first + (maximum_element_offset + 1) * element_size
    if allocation_pointer == 0 or start == 0 or end <= start:
        raise Case01CounterfactualPatientVaeAdapterError(
            "tensor exposes a null allocation pointer"
        )
    return (
        (value.device.type, value.device.index, allocation_pointer),
        (value.device.type, value.device.index, start, end),
    )


def _require_distinct_allocations(
    values: tuple[tuple[str, Any], ...], *, label: str
) -> None:
    owners: dict[tuple[str, int | None, int], str] = {}
    intervals: list[tuple[str, int | None, int, int, str]] = []
    for name, value in values:
        key, interval = _allocation_identity_and_interval(value)
        if key in owners:
            raise Case01CounterfactualPatientVaeAdapterError(
                f"{label} allocation alias is forbidden: {owners[key]} and {name}"
            )
        device_type, device_index, start, end = interval
        for (
            previous_device_type,
            previous_device_index,
            previous_start,
            previous_end,
            previous_name,
        ) in intervals:
            if (
                device_type == previous_device_type
                and device_index == previous_device_index
                and start < previous_end
                and previous_start < end
            ):
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"{label} backing byte intervals overlap: "
                    f"{previous_name} and {name}"
                )
        owners[key] = name
        intervals.append((device_type, device_index, start, end, name))


@dataclass(frozen=True)
class _LiveTensorPin:
    """Process-local tensor ABI/backing/value pin, never serialized."""

    object_identity: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    device_type: str
    device_index: int | None
    layout: str
    contiguous: bool
    requires_grad: bool
    grad_fn_is_none: bool
    conjugated: bool
    negated: bool
    version_counter: int | None
    element_size: int
    numel: int
    allocation_identity: tuple[str, int | None, int]
    occupied_interval: tuple[str, int | None, int, int]
    raw_sha256: str


def _live_tensor_pin(
    value: Any, *, label: str, logical_video: bool = False
) -> _LiveTensorPin:
    torch = _torch()
    if type(value) is not torch.Tensor:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} is no longer an exact torch.Tensor"
        )
    allocation_identity, occupied_interval = (
        _allocation_identity_and_interval(value)
    )
    raw_sha256 = (
        _video_raw_sha256(value)
        if logical_video
        else _tensor_sha256(value)
    )
    return _LiveTensorPin(
        object_identity=id(value),
        shape=tuple(int(item) for item in value.shape),
        stride=tuple(int(item) for item in value.stride()),
        storage_offset=int(value.storage_offset()),
        dtype=str(value.dtype),
        device_type=value.device.type,
        device_index=value.device.index,
        layout=str(value.layout),
        contiguous=bool(value.is_contiguous()),
        requires_grad=bool(value.requires_grad),
        grad_fn_is_none=value.grad_fn is None,
        conjugated=bool(value.is_conj()),
        negated=bool(value.is_neg()),
        version_counter=_tensor_version_or_none(value),
        element_size=int(value.element_size()),
        numel=int(value.numel()),
        allocation_identity=allocation_identity,
        occupied_interval=occupied_interval,
        raw_sha256=raw_sha256,
    )


def _tensor_version_or_none(value: Any) -> int | None:
    """Return Torch's mutation counter when inference tensors expose one."""

    try:
        return int(value._version)
    except RuntimeError:
        return None


def _video_raw_sha256(value: Any) -> str:
    """Hash logical B,C,T,H,W bytes in bounded frame-channel chunks."""

    torch = _torch()
    digest = hashlib.sha256()
    byte_count = 0
    for batch_index in range(int(value.shape[0])):
        for channel_index in range(int(value.shape[1])):
            for frame_index in range(int(value.shape[2])):
                try:
                    block = (
                        value[batch_index, channel_index, frame_index]
                        .detach()
                        .contiguous()
                        .cpu()
                    )
                    raw = block.view(torch.uint8).numpy().tobytes(order="C")
                except Exception as error:
                    raise Case01CounterfactualPatientVaeAdapterError(
                        "cannot expose exact logical video bytes"
                    ) from error
                expected = int(block.numel()) * int(block.element_size())
                if len(raw) != expected:
                    raise Case01CounterfactualPatientVaeAdapterError(
                        "logical video byte count differs"
                    )
                digest.update(raw)
                byte_count += len(raw)
    expected_total = int(value.numel()) * int(value.element_size())
    if byte_count != expected_total:
        raise Case01CounterfactualPatientVaeAdapterError(
            "logical video total byte count differs"
        )
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    try:
        return carrier.tensor_raw_sha256(value)
    except carrier.ObjectInstanceCounterfactualCarrierError as error:
        raise Case01CounterfactualPatientVaeAdapterError(
            "cannot expose exact tensor bytes"
        ) from error


def adapter_contract() -> dict[str, Any]:
    """Return the fixed claim and geometry boundary of this adapter."""

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "pure_vae_latent_pack_compile_carrier_unpack_decode_adapter",
        "rgb_shape": ["B", 3, 81, 496, 480],
        "latent_shape": ["B", 16, 21, 62, 60],
        "packed_shape": ["B", 19_530, 64],
        "pack_order": "B,T,H2,W2,inner_y,inner_x,C",
        "phase_regimes": {
            "pre_lift": [0, 9],
            "lift": [10, 15],
            "hold": [16, 20],
        },
        "phase10_replacement_shift_xy": [-3, 0],
        "phase10_responsibility": "old_responsibility_union_new_targets",
        "selected_source_tokens": SOURCE_PATIENT_TOKEN_COUNT,
        "expanded_responsibility_tokens": EXPANDED_RESPONSIBILITY_TOKEN_COUNT,
        "carrier_program": {
            "filename": CARRIER_PROGRAM_FILENAME,
            "sha256": EXPECTED_CARRIER_PROGRAM_SHA256,
            "size": EXPECTED_CARRIER_PROGRAM_SIZE,
            "runtime_byte_replay_required": True,
        },
        "carrier_runtime_semantics_authenticated": False,
        "legacy_aux_consumed": False,
        "same_vae_object_argument_routed": True,
        "vae_model_identity_authenticated": False,
        "source_video_values_authenticated": False,
        "decoded_video_values_authenticated": False,
        "caller_backing_independence_authenticated": False,
        "caller_videos_copied_to_private_snapshots": True,
        "decoded_output_copied_to_private_snapshot": True,
        "all81_review_complete": False,
        "visual_success_claimed": False,
        "scientific_claim_authorized": False,
        "only_return_status": OUTPUT_STATUS,
    }


@dataclass(frozen=True)
class CompiledPhaseAudit:
    """Tensor-free exact geometry for one compiled packed phase."""

    phase_index: int
    regime: str
    declared_shift_xy: tuple[int, int]
    compiled_shift_xy: tuple[int, int]
    local_source_tokens: tuple[int, ...]
    local_target_tokens: tuple[int, ...]
    local_responsibility_tokens: tuple[int, ...]
    global_source_tokens: tuple[int, ...]
    global_target_tokens: tuple[int, ...]
    global_responsibility_tokens: tuple[int, ...]
    replacement_applied: bool
    source_target_disjoint: bool
    responsibility_covers_targets: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "declared_shift_xy",
            "compiled_shift_xy",
            "local_source_tokens",
            "local_target_tokens",
            "local_responsibility_tokens",
            "global_source_tokens",
            "global_target_tokens",
            "global_responsibility_tokens",
        ):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class CompiledCarrierPlan:
    """Carrier phases plus live masks and a tensor-free compiler seal."""

    phases: tuple[carrier.PhaseCorrespondence, ...]
    source_mask: Any
    target_responsibility_mask: Any
    phase_audits: tuple[CompiledPhaseAudit, ...]
    scaffold_geometry_sha256: str
    source_mask_raw_sha256: str
    target_responsibility_mask_raw_sha256: str
    plan_digest: str
    legacy_aux_consumed: bool

    def audit_receipt(self) -> dict[str, Any]:
        _validate_compiled_plan_live(self)
        return {
            "schema_version": COMPILED_PLAN_SCHEMA_VERSION,
            "scaffold_geometry_sha256": self.scaffold_geometry_sha256,
            "phase_count": len(self.phases),
            "packed_token_count": PACKED_TOKEN_COUNT,
            "source_patient_token_count": int(self.source_mask.sum().item()),
            "target_responsibility_token_count": int(
                self.target_responsibility_mask.sum().item()
            ),
            "source_mask_raw_sha256": self.source_mask_raw_sha256,
            "target_responsibility_mask_raw_sha256": (
                self.target_responsibility_mask_raw_sha256
            ),
            "phase_audits": [row.as_dict() for row in self.phase_audits],
            "legacy_aux_consumed": self.legacy_aux_consumed,
            "plan_digest": self.plan_digest,
        }


@dataclass(frozen=True)
class VaeOnlyCarrierReceipt:
    """A construction receipt that deliberately cannot express visual PASS."""

    schema_version: str
    status: str
    source_video_shape: tuple[int, ...]
    source_latent_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    decoded_shape: tuple[int, ...]
    source_video_dtype: str
    source_video_device_type: str
    dtype: str
    device_type: str
    decoded_dtype: str
    decoded_device_type: str
    carrier_program_path: str
    carrier_program_sha256: str
    carrier_program_size: int
    carrier_runtime_semantics_authenticated: bool
    scaffold_geometry_sha256: str
    compiled_plan_digest: str
    carrier_authority_digest: str
    carrier_trace_digest: str
    source_latent_raw_sha256: str
    bone_removed_v2_latent_raw_sha256: str
    source_packed_raw_sha256: str
    bone_removed_v2_packed_raw_sha256: str
    patient_residual_packed_raw_sha256: str
    counterfactual_latent_raw_sha256: str
    source_video_raw_sha256: str
    bone_removed_v2_video_raw_sha256: str
    decoded_video_raw_sha256: str
    transported_residual_raw_sha256: str
    source_pack_roundtrip_byte_exact: bool
    origin_pack_roundtrip_byte_exact: bool
    counterfactual_pack_roundtrip_byte_exact: bool
    same_vae_object_argument_routed: bool
    vae_model_identity_authenticated: bool
    source_video_values_authenticated: bool
    decoded_video_values_authenticated: bool
    caller_backing_independence_authenticated: bool
    caller_videos_copied_to_private_snapshots: bool
    decoded_output_copied_to_private_snapshot: bool
    legacy_aux_consumed: bool
    all81_review_complete: bool
    visual_success_claimed: bool
    scientific_claim_authorized: bool
    receipt_digest: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("receipt_digest")
        for key in (
            "source_video_shape",
            "source_latent_shape",
            "packed_shape",
            "decoded_shape",
        ):
            value[key] = list(value[key])
        return value

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.payload(), "receipt_digest": self.receipt_digest}

    def validate(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != RECEIPT_SCHEMA_VERSION
            or type(self.status) is not str
            or self.status != OUTPUT_STATUS
            or type(self.source_video_shape) is not tuple
            or len(self.source_video_shape) != 5
            or self.source_video_shape[0] <= 0
            or self.source_video_shape[1] != 3
            or self.source_video_shape[2] != DECODED_FRAME_COUNT
            or self.source_video_shape[3:] != (RGB_HEIGHT, RGB_WIDTH)
            or any(type(item) is not int or item <= 0 for item in self.source_video_shape)
            or type(self.source_latent_shape) is not tuple
            or self.source_latent_shape[0] != self.source_video_shape[0]
            or self.source_latent_shape[1:]
            != (
                LATENT_CHANNELS,
                LATENT_PHASES,
                LATENT_HEIGHT,
                LATENT_WIDTH,
            )
            or any(type(item) is not int or item <= 0 for item in self.source_latent_shape)
            or type(self.packed_shape) is not tuple
            or self.packed_shape
            != (
                self.source_latent_shape[0],
                PACKED_TOKEN_COUNT,
                PACKED_CHANNELS,
            )
            or type(self.decoded_shape) is not tuple
            or len(self.decoded_shape) != 5
            or self.decoded_shape[0] != self.source_latent_shape[0]
            or self.decoded_shape[1] != 3
            or self.decoded_shape[2] != DECODED_FRAME_COUNT
            or self.decoded_shape[3:] != (RGB_HEIGHT, RGB_WIDTH)
            or any(type(item) is not int or item <= 0 for item in self.decoded_shape)
            or type(self.source_video_dtype) is not str
            or self.source_video_dtype
            not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            }
            or type(self.source_video_device_type) is not str
            or self.source_video_device_type not in {"cpu", "cuda"}
            or type(self.dtype) is not str
            or self.dtype
            not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            }
            or type(self.device_type) is not str
            or self.device_type not in {"cpu", "cuda"}
            or type(self.decoded_dtype) is not str
            or self.decoded_dtype
            not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            }
            or type(self.decoded_device_type) is not str
            or self.decoded_device_type not in {"cpu", "cuda"}
            or type(self.carrier_program_path) is not str
            or self.carrier_program_path
            != str(Path(__file__).resolve().with_name(CARRIER_PROGRAM_FILENAME))
            or type(self.carrier_program_size) is not int
            or self.carrier_program_size != EXPECTED_CARRIER_PROGRAM_SIZE
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                "VAE-only receipt schema/status/tensor ABI differs"
            )
        for name in (
            "scaffold_geometry_sha256",
            "compiled_plan_digest",
            "carrier_authority_digest",
            "carrier_trace_digest",
            "carrier_program_sha256",
            "source_latent_raw_sha256",
            "bone_removed_v2_latent_raw_sha256",
            "source_packed_raw_sha256",
            "bone_removed_v2_packed_raw_sha256",
            "patient_residual_packed_raw_sha256",
            "counterfactual_latent_raw_sha256",
            "source_video_raw_sha256",
            "bone_removed_v2_video_raw_sha256",
            "decoded_video_raw_sha256",
            "transported_residual_raw_sha256",
            "receipt_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"VAE-only receipt {name} differs"
                )
        if (
            self.scaffold_geometry_sha256
            != EXPECTED_SCAFFOLD_GEOMETRY_SHA256
            or self.carrier_program_sha256
            != EXPECTED_CARRIER_PROGRAM_SHA256
            or self.carrier_runtime_semantics_authenticated is not False
            or self.source_latent_raw_sha256
            == self.bone_removed_v2_latent_raw_sha256
            or self.source_packed_raw_sha256
            == self.bone_removed_v2_packed_raw_sha256
            or self.source_video_raw_sha256
            == self.bone_removed_v2_video_raw_sha256
            or self.source_pack_roundtrip_byte_exact is not True
            or self.origin_pack_roundtrip_byte_exact is not True
            or self.counterfactual_pack_roundtrip_byte_exact is not True
            or self.same_vae_object_argument_routed is not True
            or self.vae_model_identity_authenticated is not False
            or self.source_video_values_authenticated is not False
            or self.decoded_video_values_authenticated is not False
            or self.caller_backing_independence_authenticated is not False
            or self.caller_videos_copied_to_private_snapshots is not True
            or self.decoded_output_copied_to_private_snapshot is not True
            or self.legacy_aux_consumed is not False
            or self.all81_review_complete is not False
            or self.visual_success_claimed is not False
            or self.scientific_claim_authorized is not False
            or _object_sha256(self.payload()) != self.receipt_digest
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                "VAE-only receipt claim boundary or digest differs"
            )


@dataclass(frozen=True)
class VaeOnlyCarrierResult:
    """Live VAE-only tensors and their non-visual construction receipt."""

    counterfactual_latent: Any
    decoded_video: Any
    transported_residual_packed: Any
    compiled_plan: CompiledCarrierPlan
    carrier_result: carrier.PatientCarrierResult
    receipt: VaeOnlyCarrierReceipt
    _counterfactual_latent_live_pin: _LiveTensorPin
    _decoded_video_live_pin: _LiveTensorPin
    _transported_residual_live_pin: _LiveTensorPin
    _packed_counterfactual_live_pin: _LiveTensorPin

    @property
    def status(self) -> str:
        return self.receipt.status

    def audit_receipt(self) -> dict[str, Any]:
        carrier_row = _verify_frozen_carrier_program()
        if type(self.receipt) is not VaeOnlyCarrierReceipt:
            raise Case01CounterfactualPatientVaeAdapterError(
                "live VAE-only receipt type differs"
            )
        if type(self.carrier_result) is not carrier.PatientCarrierResult:
            raise Case01CounterfactualPatientVaeAdapterError(
                "live carrier result type differs"
            )
        self.receipt.validate()
        for label, observed_pin, expected_pin in (
            (
                "counterfactual latent",
                _live_tensor_pin(
                    self.counterfactual_latent,
                    label="live counterfactual latent",
                ),
                self._counterfactual_latent_live_pin,
            ),
            (
                "decoded video",
                _live_tensor_pin(
                    self.decoded_video,
                    label="live decoded video",
                    logical_video=True,
                ),
                self._decoded_video_live_pin,
            ),
            (
                "transported residual",
                _live_tensor_pin(
                    self.transported_residual_packed,
                    label="live transported residual",
                ),
                self._transported_residual_live_pin,
            ),
            (
                "packed counterfactual",
                _live_tensor_pin(
                    self.carrier_result.counterfactual,
                    label="live packed counterfactual",
                ),
                self._packed_counterfactual_live_pin,
            ),
        ):
            if (
                type(expected_pin) is not _LiveTensorPin
                or observed_pin != expected_pin
            ):
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"live {label} ABI/backing/value pin differs"
                )
        _validate_compiled_plan_live(self.compiled_plan)
        _validate_latent(
            self.counterfactual_latent, label="live counterfactual latent"
        )
        _validate_video(self.decoded_video, label="live decoded private video")
        if not self.decoded_video.is_contiguous():
            raise Case01CounterfactualPatientVaeAdapterError(
                "live decoded video is not the adapter-minted contiguous snapshot"
            )
        _validate_packed(
            self.transported_residual_packed,
            label="live transported residual",
        )
        if type(self.carrier_result) is not carrier.PatientCarrierResult:
            raise Case01CounterfactualPatientVaeAdapterError(
                "live carrier result type differs"
            )
        if self.transported_residual_packed is not self.carrier_result.transported_residual:
            raise Case01CounterfactualPatientVaeAdapterError(
                "live transported residual is not the exact carrier output"
            )
        _validate_packed(
            self.carrier_result.counterfactual,
            label="live packed counterfactual",
        )
        _require_distinct_allocations(
            (
                ("counterfactual_latent", self.counterfactual_latent),
                ("decoded_video", self.decoded_video),
                ("transported_residual", self.transported_residual_packed),
                ("packed_counterfactual", self.carrier_result.counterfactual),
            ),
            label="live adapter outputs",
        )
        trace = self.carrier_result.trace
        if type(trace) is not carrier.CounterfactualCarrierTrace:
            raise Case01CounterfactualPatientVaeAdapterError(
                "live carrier trace type differs"
            )
        if (
            type(trace.phases) is not tuple
            or len(trace.phases) != LATENT_PHASES
            or any(
                type(row) is not carrier.PhaseCarrierTrace
                for row in trace.phases
            )
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                "live carrier phase trace topology differs"
            )
        for phase_index, (trace_phase, compiled_phase) in enumerate(
            zip(trace.phases, self.compiled_plan.phase_audits)
        ):
            correspondence_value = [
                [source, target]
                for source, target in zip(
                    compiled_phase.global_source_tokens,
                    compiled_phase.global_target_tokens,
                )
            ]
            phase_tokens = list(
                range(
                    phase_index * TOKENS_PER_PHASE,
                    (phase_index + 1) * TOKENS_PER_PHASE,
                )
            )
            target_set = set(compiled_phase.global_target_tokens)
            complement = [
                token for token in phase_tokens if token not in target_set
            ]
            expected_correspondence_sha256 = _object_sha256(
                correspondence_value
            )
            expected_partition_sha256 = _object_sha256(
                {
                    "phase_tokens": phase_tokens,
                    "targets": sorted(target_set),
                    "target_complement": complement,
                }
            )
            if (
                type(trace_phase.phase_index) is not int
                or trace_phase.phase_index != phase_index
                or type(trace_phase.phase_id) is not str
                or trace_phase.phase_id != f"phase{phase_index:02d}"
                or type(trace_phase.regime) is not str
                or trace_phase.regime != compiled_phase.regime
                or type(trace_phase.phase_token_count) is not int
                or trace_phase.phase_token_count != TOKENS_PER_PHASE
                or type(trace_phase.correspondence_count) is not int
                or trace_phase.correspondence_count
                != len(compiled_phase.global_source_tokens)
                or type(trace_phase.target_token_count) is not int
                or trace_phase.target_token_count
                != len(compiled_phase.global_target_tokens)
                or type(trace_phase.complement_token_count) is not int
                or trace_phase.complement_token_count
                != TOKENS_PER_PHASE - len(compiled_phase.global_target_tokens)
                or type(trace_phase.origin_token_count) is not int
                or trace_phase.origin_token_count
                != (0 if compiled_phase.regime == "pre_lift" else len(compiled_phase.global_source_tokens))
                or trace_phase.correspondence_sha256
                != expected_correspondence_sha256
                or trace_phase.partition_sha256
                != expected_partition_sha256
                or trace_phase.pre_lift_identity is not True
                or trace_phase.pre_lift_output_byte_equal_source
                is not (True if compiled_phase.regime == "pre_lift" else None)
                or trace_phase.target_residual_byte_exact is not True
                or trace_phase.complement_byte_exact_z0 is not True
                or trace_phase.origin_byte_exact_z0 is not True
                or trace_phase.single_target_occupancy is not True
            ):
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"live carrier phase trace {phase_index} differs"
                )
        packed_replay = pack_vae_latent(self.counterfactual_latent)
        carrier_plan_payload = {
            "schema_version": carrier.SCHEMA_VERSION,
            "packed_token_count": PACKED_TOKEN_COUNT,
            "phases": [
                {
                    "phase_index": phase.phase_index,
                    "phase_id": phase.phase_id,
                    "regime": phase.regime,
                    "phase_tokens": list(phase.phase_tokens),
                    "correspondence": [
                        [source, target]
                        for source, target in phase.correspondence
                    ],
                    "target_complement": list(phase.target_complement),
                }
                for phase in self.compiled_plan.phases
            ],
        }
        if (
            self.receipt.carrier_program_path != carrier_row["path"]
            or self.receipt.carrier_program_sha256 != carrier_row["sha256"]
            or self.receipt.carrier_program_size != carrier_row["size"]
            or self.receipt.scaffold_geometry_sha256
            != self.compiled_plan.scaffold_geometry_sha256
            or self.receipt.compiled_plan_digest != self.compiled_plan.plan_digest
            or tuple(self.receipt.source_latent_shape)
            != tuple(self.counterfactual_latent.shape)
            or tuple(self.receipt.packed_shape)
            != tuple(self.carrier_result.counterfactual.shape)
            or tuple(self.receipt.decoded_shape)
            != tuple(self.decoded_video.shape)
            or self.receipt.dtype != str(self.counterfactual_latent.dtype)
            or self.receipt.device_type != self.counterfactual_latent.device.type
            or self.receipt.decoded_dtype != str(self.decoded_video.dtype)
            or self.receipt.decoded_device_type != self.decoded_video.device.type
            or self.transported_residual_packed.dtype
            != self.counterfactual_latent.dtype
            or self.transported_residual_packed.device
            != self.counterfactual_latent.device
            or _tensor_sha256(self.counterfactual_latent)
            != self.receipt.counterfactual_latent_raw_sha256
            or _video_raw_sha256(self.decoded_video)
            != self.receipt.decoded_video_raw_sha256
            or _tensor_sha256(self.transported_residual_packed)
            != self.receipt.transported_residual_raw_sha256
            or not _byte_equal(packed_replay, self.carrier_result.counterfactual)
            or _tensor_sha256(self.carrier_result.counterfactual)
            != trace.counterfactual_raw_sha256
            or _tensor_sha256(self.transported_residual_packed)
            != trace.transported_residual_raw_sha256
            or trace.authority_digest != self.receipt.carrier_authority_digest
            or trace.trace_digest != self.receipt.carrier_trace_digest
            or trace.schema_version != carrier.TRACE_SCHEMA_VERSION
            or trace.packed_shape != self.receipt.packed_shape
            or trace.dtype != self.receipt.dtype
            or trace.device_type != self.receipt.device_type
            or trace.selected_source_token_count != SOURCE_PATIENT_TOKEN_COUNT
            or trace.source_mask_raw_sha256
            != self.compiled_plan.source_mask_raw_sha256
            or trace.source_raw_sha256
            != self.receipt.source_packed_raw_sha256
            or trace.bone_removed_origin_raw_sha256
            != self.receipt.bone_removed_v2_packed_raw_sha256
            or trace.patient_residual_raw_sha256
            != self.receipt.patient_residual_packed_raw_sha256
            or trace.phase_plan_sha256 != _object_sha256(carrier_plan_payload)
            or trace.source_is_not_aux is not True
            or trace.caller_inputs_copied_to_private_snapshots is not True
            or trace.caller_backing_independence_authenticated is not False
            or trace.working_snapshot_storages_pairwise_distinct is not True
            or trace.output_storages_fresh_and_pairwise_distinct is not True
            or trace.working_snapshots_unmutated is not True
            or trace.all_target_residuals_byte_exact is not True
            or trace.all_complements_byte_exact_z0 is not True
            or trace.all_lift_hold_origins_byte_exact_z0 is not True
            or trace.single_target_occupancy is not True
            or trace.renderer_integration is not False
            or trace.visual_success_claimed is not False
            or _object_sha256(trace.payload()) != trace.trace_digest
            or _verify_frozen_carrier_program() != carrier_row
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                "live VAE carrier output/receipt replay differs"
            )
        return self.receipt.as_dict()


def _exact_dict(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be an exact dict"
        )
    return value


def _exact_int_list(
    value: Any,
    *,
    label: str,
    upper: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if (
        type(value) is not list
        or (not value and not allow_empty)
        or any(type(item) is not int for item in value)
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be an exact integer list"
        )
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be sorted and unique"
        )
    if any(item < 0 or item >= upper for item in result):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} contains an out-of-range token"
        )
    return result


def _exact_shift(value: Any, *, label: str) -> tuple[int, int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be exact integer [dx,dy]"
        )
    return int(value[0]), int(value[1])


def _shift_local_tokens(
    source: tuple[int, ...], *, shift: tuple[int, int], label: str
) -> tuple[int, ...]:
    dx, dy = shift
    targets: list[int] = []
    for token in source:
        y, x = divmod(token, PATCH_COLS)
        target_x = x + dx
        target_y = y + dy
        if not 0 <= target_x < PATCH_COLS or not 0 <= target_y < PATCH_ROWS:
            raise Case01CounterfactualPatientVaeAdapterError(
                f"{label} shift escapes the 31x30 scaffold grid"
            )
        targets.append(target_y * PATCH_COLS + target_x)
    result = tuple(targets)
    if len(set(result)) != len(result):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} shift is not one-to-one"
        )
    return result


def _phase_regime(phase_index: int) -> str:
    if 0 <= phase_index <= 9:
        return "pre_lift"
    if 10 <= phase_index <= 15:
        return "lift"
    if 16 <= phase_index <= 20:
        return "hold"
    raise Case01CounterfactualPatientVaeAdapterError(
        "phase index escapes the exact 21-phase contract"
    )


def _plan_rows_from_audits(
    audits: tuple[CompiledPhaseAudit, ...]
) -> list[dict[str, Any]]:
    if (
        type(audits) is not tuple
        or len(audits) != LATENT_PHASES
        or any(type(row) is not CompiledPhaseAudit for row in audits)
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled phase audits differ"
        )
    rows: list[dict[str, Any]] = []
    for phase_index, row in enumerate(audits):
        if (
            type(row.phase_index) is not int
            or row.phase_index != phase_index
            or type(row.regime) is not str
            or row.regime != _phase_regime(phase_index)
            or type(row.declared_shift_xy) is not tuple
            or len(row.declared_shift_xy) != 2
            or any(type(item) is not int for item in row.declared_shift_xy)
            or type(row.compiled_shift_xy) is not tuple
            or len(row.compiled_shift_xy) != 2
            or any(type(item) is not int for item in row.compiled_shift_xy)
            or type(row.local_source_tokens) is not tuple
            or type(row.local_target_tokens) is not tuple
            or type(row.local_responsibility_tokens) is not tuple
            or type(row.global_source_tokens) is not tuple
            or type(row.global_target_tokens) is not tuple
            or type(row.global_responsibility_tokens) is not tuple
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled phase audit {phase_index} ABI differs"
            )
        offset = phase_index * TOKENS_PER_PHASE
        if (
            row.global_source_tokens
            != tuple(offset + token for token in row.local_source_tokens)
            or row.global_target_tokens
            != tuple(offset + token for token in row.local_target_tokens)
            or row.global_responsibility_tokens
            != tuple(offset + token for token in row.local_responsibility_tokens)
            or row.local_target_tokens
            != _shift_local_tokens(
                row.local_source_tokens,
                shift=row.compiled_shift_xy,
                label=f"compiled audit phase {phase_index}",
            )
            or row.replacement_applied is not (phase_index == PHASE10_INDEX)
            or row.source_target_disjoint
            is not (
                not bool(
                    set(row.local_source_tokens)
                    & set(row.local_target_tokens)
                )
            )
            or row.responsibility_covers_targets
            is not set(row.local_target_tokens).issubset(
                row.local_responsibility_tokens
            )
            or row.responsibility_covers_targets is not True
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled phase audit {phase_index} geometry differs"
            )
        rows.append(
            {
                "phase_index": phase_index,
                "regime": row.regime,
                "declared_shift_xy": list(row.declared_shift_xy),
                "compiled_shift_xy": list(row.compiled_shift_xy),
                "global_source_tokens": list(row.global_source_tokens),
                "global_target_tokens": list(row.global_target_tokens),
                "global_responsibility_tokens": list(
                    row.global_responsibility_tokens
                ),
            }
        )
    return rows


def _compiled_plan_payload(
    *,
    scaffold_geometry_sha256: str,
    source_mask_raw_sha256: str,
    target_responsibility_mask_raw_sha256: str,
    phase_audits: tuple[CompiledPhaseAudit, ...],
) -> dict[str, Any]:
    return {
        "schema_version": COMPILED_PLAN_SCHEMA_VERSION,
        "scaffold_geometry_sha256": scaffold_geometry_sha256,
        "packed_token_count": PACKED_TOKEN_COUNT,
        "source_patient_token_count": SOURCE_PATIENT_TOKEN_COUNT,
        "target_responsibility_token_count": (
            EXPANDED_RESPONSIBILITY_TOKEN_COUNT
        ),
        "phase10_new_responsibility_token_count": (
            PHASE10_NEW_RESPONSIBILITY_TOKEN_COUNT
        ),
        "source_mask_raw_sha256": source_mask_raw_sha256,
        "target_responsibility_mask_raw_sha256": (
            target_responsibility_mask_raw_sha256
        ),
        "phases": _plan_rows_from_audits(phase_audits),
        "legacy_aux_consumed": False,
    }


def _validate_compiled_plan_live(plan: Any) -> None:
    _verify_frozen_carrier_program()
    torch = _torch()
    if type(plan) is not CompiledCarrierPlan:
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled plan must be an exact CompiledCarrierPlan"
        )
    if (
        type(plan.phases) is not tuple
        or len(plan.phases) != LATENT_PHASES
        or type(plan.phase_audits) is not tuple
        or len(plan.phase_audits) != LATENT_PHASES
        or type(plan.legacy_aux_consumed) is not bool
        or plan.legacy_aux_consumed is not False
        or type(plan.scaffold_geometry_sha256) is not str
        or plan.scaffold_geometry_sha256
        != EXPECTED_SCAFFOLD_GEOMETRY_SHA256
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled plan schema or claim boundary differs"
        )
    for label, mask in (
        ("source_mask", plan.source_mask),
        ("target_responsibility_mask", plan.target_responsibility_mask),
    ):
        if (
            type(mask) is not torch.Tensor
            or mask.layout != torch.strided
            or mask.device.type not in {"cpu", "cuda"}
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (PACKED_TOKEN_COUNT,)
            or not mask.is_contiguous()
            or mask.requires_grad
            or mask.grad_fn is not None
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled plan {label} ABI differs"
            )
    if plan.source_mask.device != plan.target_responsibility_mask.device:
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled plan mask devices differ"
        )

    expected_source = tuple(
        token for row in plan.phase_audits for token in row.global_source_tokens
    )
    expected_responsibility = tuple(
        token
        for row in plan.phase_audits
        for token in row.global_responsibility_tokens
    )
    observed_source = tuple(
        int(item)
        for item in torch.nonzero(plan.source_mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    observed_responsibility = tuple(
        int(item)
        for item in torch.nonzero(
            plan.target_responsibility_mask, as_tuple=False
        )
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    if (
        len(expected_source) != SOURCE_PATIENT_TOKEN_COUNT
        or observed_source != tuple(sorted(expected_source))
        or len(expected_responsibility)
        != EXPANDED_RESPONSIBILITY_TOKEN_COUNT
        or observed_responsibility != tuple(sorted(expected_responsibility))
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled live masks differ from phase audits"
        )

    for phase_index, (phase, row) in enumerate(
        zip(plan.phases, plan.phase_audits)
    ):
        offset = phase_index * TOKENS_PER_PHASE
        expected_phase_tokens = tuple(
            range(offset, offset + TOKENS_PER_PHASE)
        )
        expected_targets = set(row.global_target_tokens)
        expected_complement = tuple(
            token for token in expected_phase_tokens if token not in expected_targets
        )
        if (
            type(phase) is not carrier.PhaseCorrespondence
            or phase.phase_index != phase_index
            or type(phase.phase_index) is not int
            or phase.phase_id != f"phase{phase_index:02d}"
            or type(phase.phase_id) is not str
            or phase.regime != row.regime
            or type(phase.regime) is not str
            or phase.phase_tokens != expected_phase_tokens
            or phase.correspondence
            != tuple(zip(row.global_source_tokens, row.global_target_tokens))
            or phase.target_complement != expected_complement
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled live carrier phase {phase_index} differs"
            )

    source_mask_sha = _tensor_sha256(plan.source_mask)
    responsibility_mask_sha = _tensor_sha256(
        plan.target_responsibility_mask
    )
    if (
        type(plan.source_mask_raw_sha256) is not str
        or source_mask_sha != plan.source_mask_raw_sha256
        or type(plan.target_responsibility_mask_raw_sha256) is not str
        or responsibility_mask_sha
        != plan.target_responsibility_mask_raw_sha256
        or type(plan.plan_digest) is not str
        or _SHA256.fullmatch(plan.plan_digest) is None
        or _object_sha256(
            _compiled_plan_payload(
                scaffold_geometry_sha256=plan.scaffold_geometry_sha256,
                source_mask_raw_sha256=source_mask_sha,
                target_responsibility_mask_raw_sha256=(
                    responsibility_mask_sha
                ),
                phase_audits=plan.phase_audits,
            )
        )
        != plan.plan_digest
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled plan byte pins or digest differ"
        )


def _geometry_projection(scaffold: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _exact_dict(scaffold, label="scaffold")
    if (
        type(root.get("schema_version")) is not str
        or root.get("schema_version") != SCAFFOLD_SCHEMA_VERSION
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "scaffold schema differs"
        )
    geometry = _exact_dict(root.get("geometry"), label="geometry")
    renderer_bucket_wh = geometry.get("renderer_bucket_wh")
    if (
        type(renderer_bucket_wh) is not list
        or len(renderer_bucket_wh) != 2
        or any(type(item) is not int for item in renderer_bucket_wh)
        or renderer_bucket_wh != [RGB_WIDTH, RGB_HEIGHT]
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "scaffold geometry.renderer_bucket_wh differs from [W,H]=[480,496]"
        )
    geometry_projection = {
        "renderer_bucket_wh": list(renderer_bucket_wh),
    }
    layout = _exact_dict(root.get("latent_layout"), label="latent_layout")
    expected_layout = {
        "latent_phases": LATENT_PHASES,
        "patch_rows": PATCH_ROWS,
        "patch_cols": PATCH_COLS,
        "tokens_per_phase": TOKENS_PER_PHASE,
        "packed_token_count": PACKED_TOKEN_COUNT,
        "scheduler_target_packed_token": "phase*930+side_local_token",
    }
    layout_projection: dict[str, Any] = {}
    for key in _LAYOUT_KEYS:
        value = layout.get(key)
        if type(value) is not type(expected_layout[key]) or value != expected_layout[key]:
            raise Case01CounterfactualPatientVaeAdapterError(
                f"latent_layout.{key} differs"
            )
        layout_projection[key] = value

    rows = root.get("latent_phases")
    if type(rows) is not list or len(rows) != LATENT_PHASES:
        raise Case01CounterfactualPatientVaeAdapterError(
            "scaffold must contain exact 21 latent phases"
        )
    row_projection: list[dict[str, Any]] = []
    for expected_index, raw_row in enumerate(rows):
        row = _exact_dict(raw_row, label=f"latent_phases[{expected_index}]")
        projected: dict[str, Any] = {}
        for key in _PHASE_GEOMETRY_KEYS:
            if key not in row:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"latent_phases[{expected_index}].{key} is missing"
                )
            projected[key] = row[key]
        if (
            type(projected["phase_index"]) is not int
            or projected["phase_index"] != expected_index
            or type(projected["typed_stage"]) is not str
            or not projected["typed_stage"]
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"latent_phases[{expected_index}] identity differs"
            )
        row_projection.append(projected)

    payload = {
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "geometry": geometry_projection,
        "latent_layout": layout_projection,
        "latent_phases": row_projection,
    }
    digest = _object_sha256(payload)
    if digest != EXPECTED_SCAFFOLD_GEOMETRY_SHA256:
        raise Case01CounterfactualPatientVaeAdapterError(
            "scaffold patient geometry differs from the frozen Case01 projection"
        )
    return payload, row_projection


def _validate_original_phase_geometry(
    row: dict[str, Any], *, phase_index: int
) -> tuple[
    tuple[int, ...],
    tuple[int, int],
    tuple[int, ...],
    tuple[int, ...],
]:
    label = f"latent_phases[{phase_index}]"
    source = _exact_int_list(
        row["source_bone_tokens"],
        label=f"{label}.source_bone_tokens",
        upper=TOKENS_PER_PHASE,
    )
    declared_shift = _exact_shift(
        row["bone_shift_patch_xy"], label=f"{label}.bone_shift_patch_xy"
    )
    declared_target = _exact_int_list(
        row["target_bone_tokens"],
        label=f"{label}.target_bone_tokens",
        upper=TOKENS_PER_PHASE,
    )
    expected_declared_target = _shift_local_tokens(
        source, shift=declared_shift, label=label
    )
    if declared_target != expected_declared_target:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} target tokens do not follow declared shift"
        )

    raw_pairs = row["bone_token_correspondence"]
    if type(raw_pairs) is not list or len(raw_pairs) != len(source):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label}.bone_token_correspondence differs"
        )
    pairs: list[tuple[int, int]] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        if (
            type(raw_pair) is not list
            or len(raw_pair) != 2
            or any(type(item) is not int for item in raw_pair)
        ):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"{label}.bone_token_correspondence[{pair_index}] differs"
            )
        pairs.append((raw_pair[0], raw_pair[1]))
    if tuple(pairs) != tuple(zip(source, declared_target)):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} correspondence differs from source plus declared shift"
        )

    origin = _exact_int_list(
        row["origin_clear_tokens"],
        label=f"{label}.origin_clear_tokens",
        upper=TOKENS_PER_PHASE,
        allow_empty=True,
    )
    expected_origin = () if phase_index <= 9 else source
    if origin != expected_origin:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} origin-clear geometry differs"
        )
    responsibility = _exact_int_list(
        row["target_responsibility_tokens"],
        label=f"{label}.target_responsibility_tokens",
        upper=TOKENS_PER_PHASE,
    )
    if not set(declared_target).issubset(responsibility):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} declared targets escape responsibility"
        )
    if phase_index <= 9 and declared_shift != (0, 0):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} pre-lift shift is not identity"
        )
    if phase_index == PHASE10_INDEX and declared_shift != (-1, 0):
        raise Case01CounterfactualPatientVaeAdapterError(
            "phase 10 no longer identifies the revoked overlapping mapping"
        )
    return source, declared_shift, declared_target, responsibility


def compile_case01_carrier_plan(
    scaffold: Any, *, device: Any = "cpu"
) -> CompiledCarrierPlan:
    """Compile the frozen scaffold geometry into the strict carrier ABI.

    Only the geometry projection is read.  In particular, the scaffold's
    legacy auxiliary authority, source hashes, artifact digest, dog masks,
    and claim/status fields are not consumed or included in the plan seal.
    """

    carrier_program_before = _verify_frozen_carrier_program()
    torch = _torch()
    geometry_payload, rows = _geometry_projection(scaffold)
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled mask device differs"
        ) from error
    if target_device.type not in {"cpu", "cuda"}:
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled masks require cpu or cuda"
        )

    phases: list[carrier.PhaseCorrespondence] = []
    audits: list[CompiledPhaseAudit] = []
    all_source: list[int] = []
    all_responsibility: list[int] = []
    legacy_responsibility_count = 0
    phase10_added_count = 0

    for phase_index, row in enumerate(rows):
        source, declared_shift, _, old_responsibility = (
            _validate_original_phase_geometry(row, phase_index=phase_index)
        )
        compiled_shift = (
            PHASE10_REPLACEMENT_SHIFT
            if phase_index == PHASE10_INDEX
            else declared_shift
        )
        target = _shift_local_tokens(
            source,
            shift=compiled_shift,
            label=f"compiled phase {phase_index}",
        )
        regime = _phase_regime(phase_index)
        source_set = set(source)
        target_set = set(target)
        if regime == "pre_lift":
            if source != target:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"compiled phase {phase_index} pre-lift mapping differs"
                )
        elif source_set & target_set:
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled phase {phase_index} source/target overlap"
            )

        responsibility = tuple(
            sorted(
                set(old_responsibility) | target_set
                if phase_index == PHASE10_INDEX
                else set(old_responsibility)
            )
        )
        if not target_set.issubset(responsibility):
            raise Case01CounterfactualPatientVaeAdapterError(
                f"compiled phase {phase_index} targets escape responsibility"
            )
        if phase_index == PHASE10_INDEX:
            if len(source) != PHASE10_SOURCE_TOKEN_COUNT:
                raise Case01CounterfactualPatientVaeAdapterError(
                    "phase 10 source token count differs"
                )
            phase10_added_count = len(set(responsibility) - set(old_responsibility))
            if (
                len(responsibility)
                != PHASE10_REPLACEMENT_RESPONSIBILITY_COUNT
                or phase10_added_count
                != PHASE10_NEW_RESPONSIBILITY_TOKEN_COUNT
            ):
                raise Case01CounterfactualPatientVaeAdapterError(
                    "phase 10 exact responsibility expansion differs"
                )

        offset = phase_index * TOKENS_PER_PHASE
        global_source = tuple(offset + token for token in source)
        global_target = tuple(offset + token for token in target)
        global_responsibility = tuple(
            offset + token for token in responsibility
        )
        phase_tokens = tuple(range(offset, offset + TOKENS_PER_PHASE))
        target_complement = tuple(
            token for token in phase_tokens if token not in set(global_target)
        )
        correspondence = tuple(zip(global_source, global_target))
        phase = carrier.PhaseCorrespondence(
            phase_index=phase_index,
            phase_id=f"phase{phase_index:02d}",
            regime=regime,
            phase_tokens=phase_tokens,
            correspondence=correspondence,
            target_complement=target_complement,
        )
        audit = CompiledPhaseAudit(
            phase_index=phase_index,
            regime=regime,
            declared_shift_xy=declared_shift,
            compiled_shift_xy=compiled_shift,
            local_source_tokens=source,
            local_target_tokens=target,
            local_responsibility_tokens=responsibility,
            global_source_tokens=global_source,
            global_target_tokens=global_target,
            global_responsibility_tokens=global_responsibility,
            replacement_applied=phase_index == PHASE10_INDEX,
            source_target_disjoint=not bool(source_set & target_set),
            responsibility_covers_targets=target_set.issubset(responsibility),
        )
        phases.append(phase)
        audits.append(audit)
        all_source.extend(global_source)
        all_responsibility.extend(global_responsibility)
        legacy_responsibility_count += len(old_responsibility)

    if (
        len(all_source) != SOURCE_PATIENT_TOKEN_COUNT
        or len(set(all_source)) != SOURCE_PATIENT_TOKEN_COUNT
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled source mask is not the exact 377-token patient mask"
        )
    if legacy_responsibility_count != LEGACY_RESPONSIBILITY_TOKEN_COUNT:
        raise Case01CounterfactualPatientVaeAdapterError(
            "legacy geometry responsibility count differs"
        )
    if (
        len(all_responsibility) != EXPANDED_RESPONSIBILITY_TOKEN_COUNT
        or len(set(all_responsibility)) != EXPANDED_RESPONSIBILITY_TOKEN_COUNT
        or phase10_added_count != PHASE10_NEW_RESPONSIBILITY_TOKEN_COUNT
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled responsibility expansion is not exact"
        )

    source_mask = torch.zeros(
        (PACKED_TOKEN_COUNT,), dtype=torch.bool, device=target_device
    )
    responsibility_mask = torch.zeros_like(source_mask)
    source_index = torch.tensor(
        tuple(all_source), dtype=torch.int64, device=target_device
    )
    responsibility_index = torch.tensor(
        tuple(all_responsibility), dtype=torch.int64, device=target_device
    )
    source_mask.index_fill_(0, source_index, True)
    responsibility_mask.index_fill_(0, responsibility_index, True)
    source_mask = source_mask.contiguous()
    responsibility_mask = responsibility_mask.contiguous()
    if (
        int(source_mask.sum().item()) != SOURCE_PATIENT_TOKEN_COUNT
        or int(responsibility_mask.sum().item())
        != EXPANDED_RESPONSIBILITY_TOKEN_COUNT
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "compiled live masks differ from exact token sets"
        )

    geometry_digest = _object_sha256(geometry_payload)
    source_mask_sha256 = _tensor_sha256(source_mask)
    responsibility_mask_sha256 = _tensor_sha256(responsibility_mask)
    frozen_audits = tuple(audits)
    plan_payload = _compiled_plan_payload(
        scaffold_geometry_sha256=geometry_digest,
        source_mask_raw_sha256=source_mask_sha256,
        target_responsibility_mask_raw_sha256=responsibility_mask_sha256,
        phase_audits=frozen_audits,
    )
    result = CompiledCarrierPlan(
        phases=tuple(phases),
        source_mask=source_mask,
        target_responsibility_mask=responsibility_mask,
        phase_audits=frozen_audits,
        scaffold_geometry_sha256=geometry_digest,
        source_mask_raw_sha256=source_mask_sha256,
        target_responsibility_mask_raw_sha256=responsibility_mask_sha256,
        plan_digest=_object_sha256(plan_payload),
        legacy_aux_consumed=False,
    )
    _validate_compiled_plan_live(result)
    if _verify_frozen_carrier_program() != carrier_program_before:
        raise Case01CounterfactualPatientVaeAdapterError(
            "frozen carrier program changed during plan compilation"
        )
    return result


def _validate_latent(value: Any, *, label: str) -> tuple[int, ...]:
    torch = _torch()
    allowed_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type not in {"cpu", "cuda"}
        or value.dtype not in allowed_dtypes
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or tuple(int(item) for item in value.shape[1:])
        != (LATENT_CHANNELS, LATENT_PHASES, LATENT_HEIGHT, LATENT_WIDTH)
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be detached contiguous finite floating "
            "[B,16,21,62,60] on cpu/cuda"
        )
    return tuple(int(item) for item in value.shape)


def _validate_packed(value: Any, *, label: str) -> tuple[int, ...]:
    torch = _torch()
    allowed_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type not in {"cpu", "cuda"}
        or value.dtype not in allowed_dtypes
        or value.ndim != 3
        or int(value.shape[0]) <= 0
        or tuple(int(item) for item in value.shape[1:])
        != (PACKED_TOKEN_COUNT, PACKED_CHANNELS)
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be detached contiguous finite floating "
            "[B,19530,64] on cpu/cuda"
        )
    return tuple(int(item) for item in value.shape)


def _pack_impl(latent: Any) -> Any:
    batch = int(latent.shape[0])
    return (
        latent.reshape(
            batch,
            LATENT_CHANNELS,
            LATENT_PHASES,
            PATCH_ROWS,
            2,
            PATCH_COLS,
            2,
        )
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, PACKED_TOKEN_COUNT, PACKED_CHANNELS)
        .contiguous()
    )


def _unpack_impl(packed: Any) -> Any:
    batch = int(packed.shape[0])
    return (
        packed.reshape(
            batch,
            LATENT_PHASES,
            PATCH_ROWS,
            PATCH_COLS,
            2,
            2,
            LATENT_CHANNELS,
        )
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(
            batch,
            LATENT_CHANNELS,
            LATENT_PHASES,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )
        .contiguous()
    )


def pack_vae_latent(latent: Any) -> Any:
    """Wan-exact 2x2 patch pack with mandatory byte-exact inverse replay."""

    _validate_latent(latent, label="VAE latent")
    packed = _pack_impl(latent)
    _validate_packed(packed, label="packed VAE latent")
    replay = _unpack_impl(packed)
    if not _byte_equal(replay, latent):
        raise Case01CounterfactualPatientVaeAdapterError(
            "VAE latent pack/unpack roundtrip differs byte-for-byte"
        )
    return packed


def unpack_vae_latent(packed: Any) -> Any:
    """Exact inverse of :func:`pack_vae_latent`, with forward replay."""

    _validate_packed(packed, label="packed carrier latent")
    latent = _unpack_impl(packed)
    _validate_latent(latent, label="unpacked carrier latent")
    replay = _pack_impl(latent)
    if not _byte_equal(replay, packed):
        raise Case01CounterfactualPatientVaeAdapterError(
            "packed carrier unpack/repack roundtrip differs byte-for-byte"
        )
    return latent


def _validate_video(value: Any, *, label: str) -> tuple[int, ...]:
    torch = _torch()
    allowed_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type not in {"cpu", "cuda"}
        or value.dtype not in allowed_dtypes
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != 3
        or int(value.shape[2]) != DECODED_FRAME_COUNT
        or int(value.shape[3]) != RGB_HEIGHT
        or int(value.shape[4]) != RGB_WIDTH
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            f"{label} must be detached floating "
            "[B,3,81,496,480] on cpu/cuda"
        )
    return tuple(int(item) for item in value.shape)


def _validate_decoded_video(
    value: Any, *, batch: int
) -> tuple[int, ...]:
    shape = _validate_video(value, label="decoded carrier video")
    if shape[0] != batch:
        raise Case01CounterfactualPatientVaeAdapterError(
            "decoded carrier batch differs from encoded batch"
        )
    return shape


def run_vae_only_carrier_feasibility(
    *,
    vae: Any,
    encode: Callable[[Any, Any], Any],
    decode: Callable[[Any, Any], Any],
    source_video: Any,
    bone_removed_v2_video: Any,
    scaffold: Any,
) -> VaeOnlyCarrierResult:
    """Encode both videos with one VAE object, carry, unpack, and decode once.

    ``encode`` and ``decode`` are explicit integration functions so this pure
    adapter does not import a renderer.  The exact same live ``vae`` object is
    passed to both encode calls and the decode call.  That routing fact does
    not authenticate the model weights or upstream video authority.
    """

    carrier_program_before = _verify_frozen_carrier_program()
    if not callable(encode) or not callable(decode):
        raise Case01CounterfactualPatientVaeAdapterError(
            "encode and decode must be callable integration functions"
        )
    source_video_shape = _validate_video(source_video, label="source_video")
    clean_video_shape = _validate_video(
        bone_removed_v2_video, label="bone_removed_v2_video"
    )
    if (
        source_video_shape != clean_video_shape
        or source_video.dtype != bone_removed_v2_video.dtype
        or source_video.device != bone_removed_v2_video.device
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "source and bone_removed_v2 video ABI differs"
        )
    _require_distinct_allocations(
        (
            ("caller_source_video", source_video),
            ("caller_bone_removed_v2_video", bone_removed_v2_video),
        ),
        label="caller videos",
    )
    source_video_pin = _live_tensor_pin(
        source_video, label="caller source video", logical_video=True
    )
    clean_video_pin = _live_tensor_pin(
        bone_removed_v2_video,
        label="caller bone_removed_v2 video",
        logical_video=True,
    )
    source_video_sha = source_video_pin.raw_sha256
    clean_video_sha = clean_video_pin.raw_sha256
    if source_video_sha == clean_video_sha:
        raise Case01CounterfactualPatientVaeAdapterError(
            "source and bone_removed_v2 video bytes are identical"
        )

    torch = _torch()
    source_video_private = source_video.detach().clone(
        memory_format=torch.contiguous_format
    )
    clean_video_private = bone_removed_v2_video.detach().clone(
        memory_format=torch.contiguous_format
    )
    _validate_video(source_video_private, label="private source video snapshot")
    _validate_video(
        clean_video_private, label="private bone_removed_v2 video snapshot"
    )
    if (
        not source_video_private.is_contiguous()
        or not clean_video_private.is_contiguous()
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "private video snapshots are not contiguous"
        )
    source_video_private_pin = _live_tensor_pin(
        source_video_private,
        label="private source video snapshot",
        logical_video=True,
    )
    clean_video_private_pin = _live_tensor_pin(
        clean_video_private,
        label="private bone_removed_v2 video snapshot",
        logical_video=True,
    )
    source_video_private_sha = source_video_private_pin.raw_sha256
    clean_video_private_sha = clean_video_private_pin.raw_sha256
    if (
        source_video_private_sha != source_video_sha
        or clean_video_private_sha != clean_video_sha
        or source_video_private_pin.shape != source_video_pin.shape
        or clean_video_private_pin.shape != clean_video_pin.shape
        or source_video_private_pin.dtype != source_video_pin.dtype
        or clean_video_private_pin.dtype != clean_video_pin.dtype
        or source_video_private_pin.device_type != source_video_pin.device_type
        or clean_video_private_pin.device_type != clean_video_pin.device_type
        or source_video_private_pin.device_index != source_video_pin.device_index
        or clean_video_private_pin.device_index != clean_video_pin.device_index
        or source_video_private_pin.layout != source_video_pin.layout
        or clean_video_private_pin.layout != clean_video_pin.layout
        or source_video_private_pin.requires_grad
        or clean_video_private_pin.requires_grad
        or not source_video_private_pin.grad_fn_is_none
        or not clean_video_private_pin.grad_fn_is_none
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "private video snapshot ABI/bytes differ from caller values"
        )
    video_values = (
        ("caller source video", source_video, source_video_pin),
        (
            "caller bone_removed_v2 video",
            bone_removed_v2_video,
            clean_video_pin,
        ),
        (
            "private source video snapshot",
            source_video_private,
            source_video_private_pin,
        ),
        (
            "private bone_removed_v2 video snapshot",
            clean_video_private,
            clean_video_private_pin,
        ),
    )
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values),
        label="caller/private videos",
    )

    def require_program_unchanged(stage: str) -> None:
        if _verify_frozen_carrier_program() != carrier_program_before:
            raise Case01CounterfactualPatientVaeAdapterError(
                f"frozen carrier program changed {stage}"
            )

    def require_video_pins(stage: str) -> None:
        for name, value, expected_pin in video_values:
            observed_pin = _live_tensor_pin(
                value, label=name, logical_video=True
            )
            if observed_pin != expected_pin:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"{stage} mutated {name} bytes/ABI/backing"
                )

    require_video_pins("private snapshot construction")

    require_program_unchanged("before source encode")
    source_latent = encode(vae, source_video_private)
    require_program_unchanged("during source encode")
    require_video_pins("source encode callback")
    source_shape = _validate_latent(source_latent, label="source VAE latent")
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values)
        + (("source_latent", source_latent),),
        label="source encode outputs",
    )
    source_latent_pin = _live_tensor_pin(
        source_latent, label="source latent"
    )
    source_latent_sha = source_latent_pin.raw_sha256

    require_program_unchanged("before bone_removed_v2 encode")
    clean_latent = encode(vae, clean_video_private)
    require_program_unchanged("during bone_removed_v2 encode")
    require_video_pins("bone_removed_v2 encode callback")
    clean_shape = _validate_latent(clean_latent, label="bone_removed_v2 VAE latent")
    if (
        source_shape != clean_shape
        or source_latent.dtype != clean_latent.dtype
        or source_latent.device != clean_latent.device
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "source and bone_removed_v2 latent ABI differs"
        )
    if _live_tensor_pin(
        source_latent, label="source latent"
    ) != source_latent_pin:
        raise Case01CounterfactualPatientVaeAdapterError(
            "bone_removed_v2 encode callback mutated the source latent bytes/ABI/backing"
        )
    clean_latent_pin = _live_tensor_pin(
        clean_latent, label="bone_removed_v2 latent"
    )
    clean_latent_sha = clean_latent_pin.raw_sha256
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values)
        + (
            ("source_latent", source_latent),
            ("bone_removed_v2_latent", clean_latent),
        ),
        label="encode inputs/outputs",
    )

    source_packed = pack_vae_latent(source_latent)
    clean_packed = pack_vae_latent(clean_latent)
    if (
        source_packed.dtype != clean_packed.dtype
        or source_packed.device != clean_packed.device
        or tuple(source_packed.shape) != tuple(clean_packed.shape)
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "packed source/origin ABI differs"
        )
    _require_distinct_allocations(
        (
            ("source_latent", source_latent),
            ("bone_removed_v2_latent", clean_latent),
            ("source_packed", source_packed),
            ("bone_removed_v2_packed", clean_packed),
        ),
        label="latent/packed tensors",
    )
    source_packed_pin = _live_tensor_pin(
        source_packed, label="source packed latent"
    )
    clean_packed_pin = _live_tensor_pin(
        clean_packed, label="bone_removed_v2 packed latent"
    )
    source_packed_sha = source_packed_pin.raw_sha256
    clean_packed_sha = clean_packed_pin.raw_sha256

    def require_latent_pins(stage: str) -> None:
        for name, value, expected_pin in (
            ("source latent", source_latent, source_latent_pin),
            ("bone_removed_v2 latent", clean_latent, clean_latent_pin),
            ("source packed latent", source_packed, source_packed_pin),
            ("bone_removed_v2 packed latent", clean_packed, clean_packed_pin),
        ):
            if _live_tensor_pin(value, label=name) != expected_pin:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"{stage} mutated {name} bytes/ABI/backing"
                )

    compiled = compile_case01_carrier_plan(
        scaffold, device=source_packed.device
    )
    require_program_unchanged("during plan compilation")
    require_video_pins("plan compilation")
    require_latent_pins("plan compilation")

    try:
        require_program_unchanged("before carrier authority seal")
        authority = carrier.seal_patient_carrier_authority(
            source_packed=source_packed,
            bone_removed_packed=clean_packed,
            source_mask=compiled.source_mask,
            phases=compiled.phases,
        )
        require_program_unchanged("during carrier authority seal")
        require_video_pins("carrier authority seal")
        require_latent_pins("carrier authority seal")
        _validate_compiled_plan_live(compiled)
        if type(authority) is not carrier.PatientCarrierAuthority:
            raise Case01CounterfactualPatientVaeAdapterError(
                "carrier authority type differs"
            )

        require_program_unchanged("before carrier construction")
        carrier_result = carrier.build_counterfactual_patient_carrier(
            source_packed=source_packed,
            bone_removed_packed=clean_packed,
            source_mask=compiled.source_mask,
            phases=compiled.phases,
            authority=authority,
        )
        require_program_unchanged("during carrier construction")
        require_video_pins("carrier construction")
        require_latent_pins("carrier construction")
        _validate_compiled_plan_live(compiled)
    except carrier.ObjectInstanceCounterfactualCarrierError as error:
        raise Case01CounterfactualPatientVaeAdapterError(
            f"frozen patient carrier rejected the compiled VAE inputs: {error}"
        ) from error
    if type(carrier_result) is not carrier.PatientCarrierResult:
        raise Case01CounterfactualPatientVaeAdapterError(
            "carrier result type differs"
        )
    if type(carrier_result.trace) is not carrier.CounterfactualCarrierTrace:
        raise Case01CounterfactualPatientVaeAdapterError(
            "carrier trace type differs"
        )
    _validate_packed(
        carrier_result.counterfactual,
        label="carrier counterfactual output",
    )
    _validate_packed(
        carrier_result.transported_residual,
        label="carrier transported residual output",
    )
    if (
        carrier_result.counterfactual.dtype != source_packed.dtype
        or carrier_result.counterfactual.device != source_packed.device
        or carrier_result.transported_residual.dtype != source_packed.dtype
        or carrier_result.transported_residual.device != source_packed.device
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "carrier output dtype/device differs from packed inputs"
        )
    _require_distinct_allocations(
        (
            ("source_packed", source_packed),
            ("bone_removed_v2_packed", clean_packed),
            ("carrier_counterfactual", carrier_result.counterfactual),
            ("carrier_transported_residual", carrier_result.transported_residual),
        ),
        label="carrier inputs/outputs",
    )
    carrier_counterfactual_pin = _live_tensor_pin(
        carrier_result.counterfactual,
        label="carrier counterfactual output",
    )
    carrier_transported_pin = _live_tensor_pin(
        carrier_result.transported_residual,
        label="carrier transported residual output",
    )
    patient_residual = (source_packed - clean_packed).contiguous()
    _validate_packed(patient_residual, label="independent patient residual replay")
    patient_residual_pin = _live_tensor_pin(
        patient_residual,
        label="independent patient residual replay",
    )
    if (
        carrier_result.trace.source_raw_sha256 != source_packed_sha
        or carrier_result.trace.bone_removed_origin_raw_sha256
        != clean_packed_sha
        or carrier_result.trace.patient_residual_raw_sha256
        != patient_residual_pin.raw_sha256
        or carrier_result.trace.counterfactual_raw_sha256
        != carrier_counterfactual_pin.raw_sha256
        or carrier_result.trace.transported_residual_raw_sha256
        != carrier_transported_pin.raw_sha256
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "carrier trace source/origin/residual/output pins differ"
        )

    def require_carrier_output_pins(stage: str) -> None:
        for name, value, expected_pin in (
            (
                "carrier counterfactual",
                carrier_result.counterfactual,
                carrier_counterfactual_pin,
            ),
            (
                "carrier transported residual",
                carrier_result.transported_residual,
                carrier_transported_pin,
            ),
        ):
            if _live_tensor_pin(value, label=name) != expected_pin:
                raise Case01CounterfactualPatientVaeAdapterError(
                    f"{stage} mutated {name} bytes/ABI/backing"
                )

    counterfactual_latent = unpack_vae_latent(
        carrier_result.counterfactual
    )
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values)
        + (
            ("source_latent", source_latent),
            ("bone_removed_v2_latent", clean_latent),
            ("source_packed", source_packed),
            ("bone_removed_v2_packed", clean_packed),
            ("carrier_counterfactual", carrier_result.counterfactual),
            ("carrier_transported_residual", carrier_result.transported_residual),
            ("counterfactual_latent", counterfactual_latent),
        ),
        label="unpacked carrier outputs",
    )
    counterfactual_latent_pin = _live_tensor_pin(
        counterfactual_latent, label="counterfactual latent"
    )
    counterfactual_sha_before_decode = (
        counterfactual_latent_pin.raw_sha256
    )
    require_program_unchanged("before decode")
    require_video_pins("before decode")
    require_latent_pins("before decode")
    require_carrier_output_pins("before decode")
    decoded_raw = decode(vae, counterfactual_latent)
    require_program_unchanged("during decode")
    decoded_shape = _validate_decoded_video(
        decoded_raw, batch=source_shape[0]
    )
    if _live_tensor_pin(
        counterfactual_latent, label="counterfactual latent"
    ) != counterfactual_latent_pin:
        raise Case01CounterfactualPatientVaeAdapterError(
            "VAE decode mutated the counterfactual latent bytes/ABI/backing"
        )
    require_video_pins("decode callback")
    require_latent_pins("decode callback")
    require_carrier_output_pins("decode callback")
    _validate_compiled_plan_live(compiled)
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values)
        + (
            ("source_latent", source_latent),
            ("bone_removed_v2_latent", clean_latent),
            ("source_packed", source_packed),
            ("bone_removed_v2_packed", clean_packed),
            ("carrier_counterfactual", carrier_result.counterfactual),
            ("carrier_transported_residual", carrier_result.transported_residual),
            ("counterfactual_latent", counterfactual_latent),
            ("decoder_raw_output", decoded_raw),
        ),
        label="decode input/output",
    )
    decoded_raw_pin = _live_tensor_pin(
        decoded_raw,
        label="decoder raw output",
        logical_video=True,
    )
    decoded_private = decoded_raw.detach().clone(
        memory_format=torch.contiguous_format
    )
    decoded_shape = _validate_decoded_video(
        decoded_private, batch=source_shape[0]
    )
    if not decoded_private.is_contiguous():
        raise Case01CounterfactualPatientVaeAdapterError(
            "decoded private snapshot is not contiguous"
        )
    _require_distinct_allocations(
        tuple((name, value) for name, value, _pin in video_values)
        + (
            ("source_latent", source_latent),
            ("bone_removed_v2_latent", clean_latent),
            ("source_packed", source_packed),
            ("bone_removed_v2_packed", clean_packed),
            ("carrier_counterfactual", carrier_result.counterfactual),
            ("carrier_transported_residual", carrier_result.transported_residual),
            ("counterfactual_latent", counterfactual_latent),
            ("decoder_raw_output", decoded_raw),
            ("decoded_private_snapshot", decoded_private),
        ),
        label="decoded private output",
    )
    decoded_private_pin = _live_tensor_pin(
        decoded_private,
        label="decoded private snapshot",
        logical_video=True,
    )
    decoded_sha = decoded_private_pin.raw_sha256
    if (
        _live_tensor_pin(
            decoded_raw,
            label="decoder raw output",
            logical_video=True,
        )
        != decoded_raw_pin
        or decoded_raw_pin.raw_sha256 != decoded_private_pin.raw_sha256
        or decoded_raw_pin.shape != decoded_private_pin.shape
        or decoded_raw_pin.dtype != decoded_private_pin.dtype
        or decoded_raw_pin.device_type != decoded_private_pin.device_type
        or decoded_raw_pin.device_index != decoded_private_pin.device_index
        or decoded_raw_pin.layout != decoded_private_pin.layout
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "decoded private snapshot ABI/bytes differ from decoder output"
        )
    require_program_unchanged("before returning result")
    require_video_pins("decoded snapshot construction")
    require_latent_pins("decoded snapshot construction")
    require_carrier_output_pins("decoded snapshot construction")
    if _live_tensor_pin(
        counterfactual_latent, label="counterfactual latent"
    ) != counterfactual_latent_pin:
        raise Case01CounterfactualPatientVaeAdapterError(
            "counterfactual latent bytes/ABI/backing changed after decode"
        )
    if (
        _tensor_sha256(carrier_result.counterfactual)
        != carrier_result.trace.counterfactual_raw_sha256
        or _tensor_sha256(carrier_result.transported_residual)
        != carrier_result.trace.transported_residual_raw_sha256
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "live carrier outputs differ from their trace"
        )

    receipt_without_digest = VaeOnlyCarrierReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=OUTPUT_STATUS,
        source_video_shape=source_video_shape,
        source_latent_shape=source_shape,
        packed_shape=tuple(int(item) for item in source_packed.shape),
        decoded_shape=decoded_shape,
        source_video_dtype=str(source_video.dtype),
        source_video_device_type=source_video.device.type,
        dtype=str(source_latent.dtype),
        device_type=source_latent.device.type,
        decoded_dtype=str(decoded_private.dtype),
        decoded_device_type=decoded_private.device.type,
        carrier_program_path=carrier_program_before["path"],
        carrier_program_sha256=carrier_program_before["sha256"],
        carrier_program_size=carrier_program_before["size"],
        carrier_runtime_semantics_authenticated=False,
        scaffold_geometry_sha256=compiled.scaffold_geometry_sha256,
        compiled_plan_digest=compiled.plan_digest,
        carrier_authority_digest=authority.authority_digest,
        carrier_trace_digest=carrier_result.trace.trace_digest,
        source_latent_raw_sha256=source_latent_sha,
        bone_removed_v2_latent_raw_sha256=clean_latent_sha,
        source_packed_raw_sha256=source_packed_sha,
        bone_removed_v2_packed_raw_sha256=clean_packed_sha,
        patient_residual_packed_raw_sha256=(
            patient_residual_pin.raw_sha256
        ),
        counterfactual_latent_raw_sha256=counterfactual_sha_before_decode,
        source_video_raw_sha256=source_video_sha,
        bone_removed_v2_video_raw_sha256=clean_video_sha,
        decoded_video_raw_sha256=decoded_sha,
        transported_residual_raw_sha256=(
            carrier_result.trace.transported_residual_raw_sha256
        ),
        source_pack_roundtrip_byte_exact=_byte_equal(
            unpack_vae_latent(source_packed), source_latent
        ),
        origin_pack_roundtrip_byte_exact=_byte_equal(
            unpack_vae_latent(clean_packed), clean_latent
        ),
        counterfactual_pack_roundtrip_byte_exact=_byte_equal(
            pack_vae_latent(counterfactual_latent),
            carrier_result.counterfactual,
        ),
        same_vae_object_argument_routed=True,
        vae_model_identity_authenticated=False,
        source_video_values_authenticated=False,
        decoded_video_values_authenticated=False,
        caller_backing_independence_authenticated=False,
        caller_videos_copied_to_private_snapshots=True,
        decoded_output_copied_to_private_snapshot=True,
        legacy_aux_consumed=False,
        all81_review_complete=False,
        visual_success_claimed=False,
        scientific_claim_authorized=False,
        receipt_digest="",
    )
    if not (
        receipt_without_digest.source_pack_roundtrip_byte_exact
        and receipt_without_digest.origin_pack_roundtrip_byte_exact
        and receipt_without_digest.counterfactual_pack_roundtrip_byte_exact
    ):
        raise Case01CounterfactualPatientVaeAdapterError(
            "one or more VAE pack/unpack witnesses differ"
        )
    receipt = VaeOnlyCarrierReceipt(
        **{
            **receipt_without_digest.__dict__,
            "receipt_digest": _object_sha256(
                receipt_without_digest.payload()
            ),
        }
    )
    receipt.validate()
    require_program_unchanged("after receipt validation")
    require_video_pins("receipt construction")
    require_latent_pins("receipt construction")
    require_carrier_output_pins("receipt construction")
    result = VaeOnlyCarrierResult(
        counterfactual_latent=counterfactual_latent,
        decoded_video=decoded_private,
        transported_residual_packed=carrier_result.transported_residual,
        compiled_plan=compiled,
        carrier_result=carrier_result,
        receipt=receipt,
        _counterfactual_latent_live_pin=counterfactual_latent_pin,
        _decoded_video_live_pin=decoded_private_pin,
        _transported_residual_live_pin=carrier_transported_pin,
        _packed_counterfactual_live_pin=carrier_counterfactual_pin,
    )
    result.audit_receipt()
    require_program_unchanged("after final live receipt audit")
    return result


__all__ = [
    "COMPILED_PLAN_SCHEMA_VERSION",
    "DECODED_FRAME_COUNT",
    "EXPANDED_RESPONSIBILITY_TOKEN_COUNT",
    "EXPECTED_SCAFFOLD_GEOMETRY_SHA256",
    "LATENT_CHANNELS",
    "LATENT_HEIGHT",
    "LATENT_PHASES",
    "LATENT_WIDTH",
    "OUTPUT_STATUS",
    "PACKED_CHANNELS",
    "PACKED_TOKEN_COUNT",
    "PATCH_COLS",
    "PATCH_ROWS",
    "PHASE10_REPLACEMENT_SHIFT",
    "RECEIPT_SCHEMA_VERSION",
    "RGB_HEIGHT",
    "RGB_WIDTH",
    "SCHEMA_VERSION",
    "SOURCE_PATIENT_TOKEN_COUNT",
    "TOKENS_PER_PHASE",
    "Case01CounterfactualPatientVaeAdapterError",
    "CompiledCarrierPlan",
    "CompiledPhaseAudit",
    "VaeOnlyCarrierReceipt",
    "VaeOnlyCarrierResult",
    "adapter_contract",
    "compile_case01_carrier_plan",
    "pack_vae_latent",
    "run_vae_only_carrier_feasibility",
    "unpack_vae_latent",
]
