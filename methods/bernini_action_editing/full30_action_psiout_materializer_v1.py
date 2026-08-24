#!/usr/bin/env python3
"""Materialize reviewed same-state post-head PsiOut and amplitude evidence.

This program is deliberately an evidence producer, never a trainer.  It
reopens sealed FP32 clean/noise tensors, constructs one noisy state per
``(record, sigma)``, and reuses that exact state for every Frozen Bernini
counterfactual.  The only persisted tensors are the strict ``[21,32]``
containers already consumed by ``full30_action_data_teacher_authority_v1``
and ``full30_action_amplitude_authority_v1``.

The production entry point is WORLD4/SP4 and uses the official Bernini
``shared_step`` post-final-norm/``proj_out`` target velocity.  Tests may inject
a CPU provider explicitly; the command line has no fake-provider switch.
Generated RGB is authenticated for its prior full81 review but is never
decoded, loaded as a tensor, or used as a regression target.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Protocol, Sequence

try:
    import full30_action_amplitude_authority_v1 as amplitude_authority
    import full30_action_data_teacher_authority_v1 as teacher_authority
except ImportError:  # pragma: no cover - package import mode
    from . import full30_action_amplitude_authority_v1 as amplitude_authority
    from . import full30_action_data_teacher_authority_v1 as teacher_authority


PLAN_SCHEMA_VERSION = "bernini-full30-action-psiout-materialization-plan-v1"
PLAN_RUNTIME_SCHEMA_VERSION = "bernini-full30-action-psiout-runtime-plan-v1"
PLAN_POPULATION_SCHEMA_VERSION = "bernini-full30-action-psiout-population-v1"
PLAN_RECORD_SCHEMA_VERSION = "bernini-full30-action-psiout-record-v1"
PLAN_ARTIFACT_SCHEMA_VERSION = "bernini-full30-action-psiout-fp32-artifact-v1"
PLAN_LATENT_AUTHORITY_SCHEMA_VERSION = "bernini-full30-action-psiout-anchor-latent-authority-v1"
PLAN_REVIEW_SCHEMA_VERSION = "bernini-full30-action-psiout-review-binding-v1"
PLAN_CONDITION_SCHEMA_VERSION = "bernini-full30-action-psiout-condition-v1"
PLAN_OUTPUT_POLICY_SCHEMA_VERSION = "bernini-full30-action-psiout-output-policy-v1"
RUN_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-psiout-materialization-run-v1"
RECORD_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-psiout-materialization-record-v1"
STATE_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-psiout-same-state-v1"
FORWARD_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-psiout-frozen-forward-v1"
NOISE_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-psiout-noise-replay-v1"
SIGMA_AUTHORITY_RECEIPT_SCHEMA_VERSION = (
    "bernini-full30-action-psiout-sigma-authority-v1"
)

_STATE_RECEIPT_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "runtime_digest",
    "record_id",
    "record_kind",
    "teacher_cell_id",
    "branch",
    "sigma_index",
    "sigma_float32_be_hex",
    "timestep",
    "clean_raw_sha256",
    "source_raw_sha256",
    "noise_raw_sha256",
    "x_sigma_raw_sha256",
    "input_hashes",
    "target_tokens",
    "spatial_shape",
    "same_x_sigma_object_for_all_counterfactuals",
    "all_rank_consensus",
    "model_parameters_updated",
    "optimizer_created",
    "state_digest",
}
_FORWARD_RECEIPT_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "record_id",
    "condition_role",
    "condition_utf8_sha256",
    "shared_state_digest",
    "runtime_digest",
    "sigma_index",
    "sigma_float32_be_hex",
    "timestep",
    "output_stage",
    "official_frozen_native_only",
    "model_eval",
    "torch_inference_mode",
    "calibrator_peft_adapter_present",
    "frozen_effective_adapter_enabled",
    "frozen_effective_typed_patch_role_enabled",
    "base_compute_dtype",
    "autocast_dtype",
    "observer_output_dtype",
    "observer_output_detached",
    "observer_output_contiguous",
    "same_state_input_objects_reused",
    "same_state_input_bytes_unchanged",
    "all_rank_consensus",
    "post_head_velocity_raw_sha256",
    "model_parameters_updated",
    "optimizer_created",
    "forward_digest",
}
_NOISE_RECEIPT_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "record_id",
    "seed",
    "generator",
    "shape",
    "artifact_raw_sha256",
    "replayed_raw_sha256",
    "byte_exact_replay",
    "noise_digest",
}
_SIGMA_AUTHORITY_ROW_FIELDS = {
    "sigma_index",
    "timestep",
    "sigma_float32_be_hex",
}
_SIGMA_AUTHORITY_RECEIPT_FIELDS = {
    "schema_version",
    "schedule_schema_version",
    "schedule_sha256",
    "scheduler_class",
    "num_train_timesteps",
    "num_inference_steps",
    "flow_shift_float64_hex",
    "timesteps_int64",
    "positive_sigmas_float32_be_hex",
    "terminal_sigma_float32_be_hex",
    "materialized_sigma_indices",
    "materialized_rows",
    "sigma_authority_digest",
}
_CONTAINER_BINDING_FIELDS = {
    "container_kind",
    "path",
    "file_sha256",
    "slice_sha256",
}
_RUN_RECORD_REFERENCE_FIELDS = {
    "record_id",
    "record_kind",
    "path",
    "file_sha256",
    "record_receipt_digest",
    "candidate_evidence_digest",
}
_REPRESENTATION_FRAGMENT_FIELDS = {
    "teacher_cell_id",
    "branch",
    "origin_record_id",
    "cross_anchor_record_id",
    "origin_evidence_digest",
    "cross_anchor_evidence_digest",
    "sigma_evidence",
}
_AMPLITUDE_CALIBRATION_FRAGMENT_FIELDS = {
    "teacher_cell_id",
    "branch",
    "calibrator_record_ids",
    "calibrator_evidence_candidates",
    "sigma_calibrations",
}
_TEACHER_ORIGIN_SIGMA_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_unit_sha256",
    "projected_raw_sha256",
    "duplicate_forward_first_sha256",
    "duplicate_forward_second_sha256",
    "duplicate_forward_bytes_identical",
    "noop_forward_first_sha256",
    "noop_forward_second_sha256",
    "same_state_noop_minus_noop_null_norm",
    "projected_teacher_raw_norm",
    "signal_to_null_snr",
    "camera_unit_sha256",
    "appearance_unit_sha256",
    "camera_residual_cosine",
    "appearance_residual_cosine",
    "wrong_controls",
}
_TEACHER_CROSS_SIGMA_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_unit_sha256",
    "camera_unit_sha256",
    "appearance_unit_sha256",
}
_AMPLITUDE_SIGMA_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_slice_sha256",
    "amplitude_norm",
    "teacher_nuisance_camera_sha256",
    "teacher_nuisance_appearance_sha256",
}
_WRONG_CONTROL_METRIC_FIELDS = {
    "control_type",
    "control_anchor_id",
    "wrong_projected_slice_sha256",
    "wrong_event_cosine",
}
_RECORD_RECEIPT_FIELDS = {
    "schema_version",
    "plan_id",
    "plan_digest",
    "runtime_digest",
    "provider_abi",
    "official_provider",
    "test_only",
    "record_ordinal",
    "record_id",
    "record_digest",
    "record_kind",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "branch",
    "record_authority",
    "record_conditions",
    "review_digest",
    "reviewed_media_sha256",
    "target_clean_latent_raw_sha256",
    "target_clean_latent_authority_digest",
    "source_clean_latent_raw_sha256",
    "source_posterior_index0_sha256",
    "noise_seed",
    "noise_raw_sha256",
    "noise_replay_receipt",
    "sigma_authority_digest",
    "state_receipts",
    "forward_receipts",
    "container_bindings",
    "sigma_metrics",
    "candidate_authority_evidence",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "record_receipt_digest",
}
_RUN_RECEIPT_FIELDS = {
    "schema_version",
    "plan_id",
    "plan_digest",
    "plan_authority",
    "population_digest",
    "record_order_sha256",
    "runtime_identity",
    "runtime_plan_digest",
    "official_helper_sources",
    "provider_abi",
    "official_provider",
    "test_only",
    "world_size",
    "dp_size",
    "sp_size",
    "sigma_indices",
    "sigma_authority",
    "record_count",
    "computation_digest",
    "record_receipts",
    "representation_sigma_evidence_candidates",
    "amplitude_sigma_calibration_candidates",
    "output_policy",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "run_digest",
}

OFFICIAL_PROVIDER_ABI = "full30-psiout-official-provider-v1"
POST_HEAD_STAGE = "post-final-norm-proj-out-target-velocity"
PINNED_PSIOUT_PROTOCOL_SHA256 = "67275ae09e7cb7b1e7e8fc43ce2928031b3fe8aabe213e8626000f37abad4ead"
PINNED_POST_HEAD_RUNTIME_SHA256 = "9179394fddfd17a2a02773b3f94c77024dabdc5076f92980c3e637c1d0dd7da1"
PINNED_SIGMA_TABLE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
WORLD_SIZE = 4
DP_SIZE = 1
SP_SIZE = 4
SP_ORDER_CONTRACT = "official-world4-sp4-rank-order-v1"

RECORD_KINDS = ("teacher_anchor", "amplitude_calibrator")
TEACHER_EVIDENCE_ROLES = ("teacher_origin", "same_event_cross_anchor")
BRANCHES = ("action", "incomplete")
TEACHER_CONDITION_ROLES = (
    "branch",
    "noop",
    "camera_only",
    "appearance_only",
    "wrong_actor",
    "wrong_object",
    "generic_wrong_motion",
)
CROSS_CONDITION_ROLES = ("branch", "noop", "camera_only", "appearance_only")
AMPLITUDE_CONDITION_ROLES = ("branch", "noop")
SIGMA_INDICES = tuple(teacher_authority.SIGMA_INDICES)
LATENT_CHANNELS = 16
LATENT_PHASES = 21
QUOTIENT_SHAPE = (21, 32)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TENSOR_FILE_BYTES = 512 * 1024 * 1024
SAFETENSORS_MAX_HEADER_BYTES = 4 * 1024 * 1024
TEACHER_NOISE_DOMAIN = b"full30-teacher-noise-v1\x00"
AMPLITUDE_NOISE_DOMAIN = b"full30-amplitude-calibrator-noise-v1\x00"
EXACT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IID = re.compile(r"^[0-9a-f]{16}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FLOAT32_HEX = re.compile(r"^[0-9a-f]{8}$")


class Full30PsiOutMaterializationError(RuntimeError):
    """Raised before ambiguous evidence can be persisted."""


def fail(message: str) -> NoReturn:
    raise Full30PsiOutMaterializationError(message)


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
        raise Full30PsiOutMaterializationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_record(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    return {**unsigned, digest_field: object_sha256(unsigned)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _closed(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        fail(
            f"{label} field closure differs: observed={observed!r}, "
            f"expected={sorted(expected)!r}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be a safe non-empty identifier")
    return value


def _revision(value: Any, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 40-hex revision")
    return value


def _iid(value: Any, label: str) -> str:
    if type(value) is not str or _IID.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 16-hex IID")
    return value


def _verify_seal(value: Mapping[str, Any], field_name: str, label: str) -> None:
    declared = _sha(value.get(field_name), f"{label}.{field_name}")
    unsigned = dict(value)
    del unsigned[field_name]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_file(value: Any, label: str) -> Path:
    if type(value) is not str:
        fail(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Full30PsiOutMaterializationError(
            f"{label} is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a plain non-symlink file")
    return path.resolve(strict=True)


def _plain_directory(value: Any, label: str) -> Path:
    if type(value) is not str:
        fail(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Full30PsiOutMaterializationError(
            f"{label} is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} must be a plain non-symlink directory")
    return path.resolve(strict=True)


def _read_stable_plain_file(
    value: Any,
    expected_sha256: Any,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    path = _plain_file(value, label)
    expected = _sha(expected_sha256, f"{label} SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Full30PsiOutMaterializationError(
            f"{label} cannot be opened safely: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum_bytes,
            f"{label} size/type differs",
        )
        remaining = before.st_size
        parts: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} was truncated while reading")
            parts.append(block)
            remaining -= len(block)
        _require(os.read(descriptor, 1) == b"", f"{label} grew while reading")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in stable),
        f"{label} metadata changed while reading",
    )
    raw = b"".join(parts)
    _require(hashlib.sha256(raw).hexdigest() == expected, f"{label} SHA-256 differs")
    return path, raw


def _load_json_file(
    path_value: Any, expected_sha256: Any, label: str
) -> tuple[Mapping[str, Any], Path, bytes]:
    path, raw = _read_stable_plain_file(
        path_value,
        expected_sha256,
        label=label,
        maximum_bytes=MAX_JSON_BYTES,
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30PsiOutMaterializationError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    return value, path, raw


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, ValueError, TypeError, struct.error) as error:
        raise Full30PsiOutMaterializationError("value is outside finite FP32") from error
    if not math.isfinite(result):
        fail("value is outside finite FP32")
    return result


@dataclass(frozen=True)
class FP32TensorV1:
    """Small dependency-free tensor used by strict I/O and CPU hostile tests."""

    shape: tuple[int, ...]
    values: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not self.shape
            or any(type(item) is not int or item <= 0 for item in self.shape)
            or math.prod(self.shape) != len(self.values)
            or any(not math.isfinite(float(item)) for item in self.values)
        ):
            fail("FP32 tensor shape/values differ")
        quantized = tuple(_f32(item) for item in self.values)
        object.__setattr__(self, "values", quantized)

    def bytes_le(self) -> bytes:
        return struct.pack(f"<{len(self.values)}f", *self.values)

    def raw_sha256(self) -> str:
        return hashlib.sha256(self.bytes_le()).hexdigest()


_ARTIFACT_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "tensor_key",
    "tensor_raw_sha256",
    "dtype",
    "shape",
}
_LATENT_AUTHORITY_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "digest_field",
    "digest",
    "media_json_pointer",
    "latent_json_pointer",
    "checkpoint_tree_sha256_json_pointer",
}


def _parse_single_f32_safetensor(
    raw: bytes,
    *,
    expected_key: str,
    expected_shape: Sequence[int],
    expected_raw_sha256: str,
    label: str,
) -> FP32TensorV1:
    _require(len(raw) >= 8, f"{label} safetensors prefix is truncated")
    header_length = struct.unpack("<Q", raw[:8])[0]
    _require(
        0 < header_length <= SAFETENSORS_MAX_HEADER_BYTES,
        f"{label} safetensors header length differs",
    )
    payload_start = 8 + header_length
    _require(payload_start <= len(raw), f"{label} safetensors header is truncated")
    try:
        header = json.loads(
            raw[8:payload_start],
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30PsiOutMaterializationError(
            f"{label} safetensors header cannot be decoded"
        ) from error
    _require(type(header) is dict, f"{label} safetensors header differs")
    tensor_keys = [key for key in header if key != "__metadata__"]
    _require(tensor_keys == [expected_key], f"{label} must contain exactly {expected_key!r}")
    if "__metadata__" in header:
        metadata = header["__metadata__"]
        _require(
            type(metadata) is dict
            and all(type(key) is str and type(value) is str for key, value in metadata.items()),
            f"{label} safetensors metadata differs",
        )
    entry = _closed(
        header[expected_key], {"dtype", "shape", "data_offsets"}, f"{label}.tensor"
    )
    shape = tuple(entry["shape"]) if type(entry["shape"]) is list else ()
    _require(entry["dtype"] == "F32", f"{label} dtype must be F32")
    _require(shape == tuple(expected_shape), f"{label} shape differs")
    expected_bytes = math.prod(shape) * 4
    _require(
        entry["data_offsets"] == [0, expected_bytes],
        f"{label} tensor offsets/extra payload differ",
    )
    payload = raw[payload_start:]
    _require(len(payload) == expected_bytes, f"{label} payload length differs")
    expected_raw = _sha(expected_raw_sha256, f"{label} tensor raw SHA-256")
    _require(
        hashlib.sha256(payload).hexdigest() == expected_raw,
        f"{label} tensor raw SHA-256 differs",
    )
    values = struct.unpack(f"<{math.prod(shape)}f", payload)
    _require(all(math.isfinite(item) for item in values), f"{label} tensor is non-finite")
    return FP32TensorV1(tuple(int(item) for item in shape), tuple(values))


def load_fp32_artifact_v1(value: Any, *, label: str) -> FP32TensorV1:
    row = _closed(value, _ARTIFACT_FIELDS, label)
    _require(row["schema_version"] == PLAN_ARTIFACT_SCHEMA_VERSION, f"{label} schema differs")
    _safe_id(row["tensor_key"], f"{label}.tensor_key")
    _require(row["dtype"] == "float32-le", f"{label}.dtype differs")
    shape = row["shape"]
    if (
        type(shape) is not list
        or len(shape) != 5
        or any(type(item) is not int or item <= 0 for item in shape)
        or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
        or shape[3] % 2
        or shape[4] % 2
    ):
        fail(f"{label}.shape must be [1,16,21,evenH,evenW]")
    _, raw = _read_stable_plain_file(
        row["path"],
        row["file_sha256"],
        label=label,
        maximum_bytes=MAX_TENSOR_FILE_BYTES,
    )
    return _parse_single_f32_safetensor(
        raw,
        expected_key=str(row["tensor_key"]),
        expected_shape=shape,
        expected_raw_sha256=str(row["tensor_raw_sha256"]),
        label=label,
    )


def teacher_noise_seed_v1(teacher_cell_id: str, branch: str) -> int:
    _safe_id(teacher_cell_id, "teacher cell id")
    if branch not in BRANCHES:
        fail("teacher noise branch differs")
    payload = TEACHER_NOISE_DOMAIN + teacher_cell_id.encode("utf-8") + b"\x00" + branch.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def amplitude_noise_seed_v1(pair_id: str) -> int:
    _safe_id(pair_id, "amplitude pair id")
    return int.from_bytes(hashlib.sha256(AMPLITUDE_NOISE_DOMAIN + pair_id.encode("utf-8")).digest()[:8], "big")


def _tensor_raw_sha256(value: Any, *, label: str) -> str:
    if isinstance(value, FP32TensorV1):
        return value.raw_sha256()
    try:
        import torch
    except ImportError as error:  # pragma: no cover - integration only
        raise Full30PsiOutMaterializationError(
            f"{label} requires PyTorch or FP32TensorV1"
        ) from error
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        fail(f"{label} must be finite contiguous FP32")
    cpu = value.detach().cpu().contiguous()
    _require(sys.byteorder == "little", f"{label} requires a little-endian runtime")
    raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _tensor_shape(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, FP32TensorV1):
        return value.shape
    shape = getattr(value, "shape", None)
    if shape is None:
        fail(f"{label} has no shape")
    return tuple(int(item) for item in shape)


class PsiOutNumericsV1(Protocol):
    def from_artifact(self, value: FP32TensorV1) -> Any: ...
    def mix(self, clean: Any, noise: Any, sigma: float) -> Any: ...
    def delta(self, left: Any, right: Any) -> Any: ...
    def psiout_raw(self, velocity_or_delta: Any) -> Any: ...
    def nuisance(self, camera_raw: Any, appearance_raw: Any) -> Any: ...
    def project(self, raw: Any, packet: Any) -> Any: ...
    def unit(self, projected: Any) -> Any: ...
    def to_fp32_tensor(self, value: Any) -> FP32TensorV1: ...
    def norm(self, value: Any) -> float: ...
    def cosine(self, left: Any, right: Any) -> float: ...


@dataclass(frozen=True)
class _ReferenceNuisancePacket:
    camera_unit: FP32TensorV1
    appearance_unit: FP32TensorV1


class ReferenceFP32NumericsV1:
    """Dependency-free reference backend, admitted only by explicit test APIs."""

    def from_artifact(self, value: FP32TensorV1) -> FP32TensorV1:
        return value

    def mix(self, clean: FP32TensorV1, noise: FP32TensorV1, sigma: float) -> FP32TensorV1:
        _require(clean.shape == noise.shape, "clean/noise shape differs")
        sigma32 = _f32(sigma)
        one_minus = _f32(1.0 - sigma32)
        return FP32TensorV1(
            clean.shape,
            tuple(
                _f32(_f32(one_minus * clean_value) + _f32(sigma32 * noise_value))
                for clean_value, noise_value in zip(clean.values, noise.values)
            ),
        )

    def delta(self, left: FP32TensorV1, right: FP32TensorV1) -> FP32TensorV1:
        _require(left.shape == right.shape, "velocity shapes differ")
        return FP32TensorV1(
            left.shape, tuple(_f32(a - b) for a, b in zip(left.values, right.values))
        )

    def psiout_raw(self, value: FP32TensorV1) -> FP32TensorV1:
        shape = value.shape
        _require(
            len(shape) == 5 and shape[:3] == (1, 16, 21),
            "post-head velocity shape differs",
        )
        _, channels, phases, height, width = shape
        signed = []
        for y_index in range(height):
            y = _f32((2.0 * y_index + 1.0 - height) / height)
            for x_index in range(width):
                x = _f32((2.0 * x_index + 1.0 - width) / width)
                signed.append(_f32(x + y))
        mean = _f32(math.fsum(signed) / len(signed))
        centered = tuple(_f32(item - mean) for item in signed)
        rms = _f32(math.sqrt(math.fsum(item * item for item in centered) / len(centered)))
        _require(rms > 1.0e-6, "PsiOut spatial coordinate is degenerate")
        weights = tuple(_f32(item / rms) for item in centered)

        def offset(channel: int, phase: int, y: int, x: int) -> int:
            return (((channel * phases + phase) * height + y) * width + x)

        code = [[0.0] * 32 for _ in range(phases)]
        spatial_count = height * width
        for phase in range(phases):
            for channel in range(channels):
                samples = [
                    value.values[offset(channel, phase, y, x)]
                    for y in range(height)
                    for x in range(width)
                ]
                code[phase][channel] = _f32(math.fsum(samples) / spatial_count)
                code[phase][16 + channel] = _f32(
                    math.fsum(_f32(sample * weight) for sample, weight in zip(samples, weights))
                    / spatial_count
                )
        base = code[0]
        values: list[float] = []
        for phase in range(phases):
            values.extend(
                0.0 if phase == 0 else _f32(code[phase][index] - base[index])
                for index in range(32)
            )
        return FP32TensorV1(QUOTIENT_SHAPE, tuple(values))

    def nuisance(
        self, camera_raw: FP32TensorV1, appearance_raw: FP32TensorV1
    ) -> _ReferenceNuisancePacket:
        _require(camera_raw.shape == appearance_raw.shape == QUOTIENT_SHAPE, "nuisance shapes differ")
        camera_norm = self.norm(camera_raw)
        appearance_norm = self.norm(appearance_raw)
        _require(camera_norm > 1.0e-6 and appearance_norm > 1.0e-6, "nuisance is degenerate")
        camera = tuple(_f32(item / camera_norm) for item in camera_raw.values)
        dot = math.fsum(a * b for a, b in zip(appearance_raw.values, camera))
        orth = tuple(_f32(a - _f32(dot * b)) for a, b in zip(appearance_raw.values, camera))
        orth_norm = math.sqrt(math.fsum(item * item for item in orth))
        _require(orth_norm / appearance_norm > 1.0e-5, "appearance nuisance is collinear")
        appearance = tuple(_f32(item / orth_norm) for item in orth)
        return _ReferenceNuisancePacket(
            FP32TensorV1(QUOTIENT_SHAPE, camera),
            FP32TensorV1(QUOTIENT_SHAPE, appearance),
        )

    def project(
        self, raw: FP32TensorV1, packet: _ReferenceNuisancePacket
    ) -> FP32TensorV1:
        _require(raw.shape == QUOTIENT_SHAPE, "projected raw shape differs")
        cam = packet.camera_unit.values
        app = packet.appearance_unit.values
        dot_cam = math.fsum(a * b for a, b in zip(raw.values, cam))
        first = tuple(_f32(a - _f32(dot_cam * b)) for a, b in zip(raw.values, cam))
        dot_app = math.fsum(a * b for a, b in zip(first, app))
        return FP32TensorV1(
            QUOTIENT_SHAPE,
            tuple(_f32(a - _f32(dot_app * b)) for a, b in zip(first, app)),
        )

    def unit(self, projected: FP32TensorV1) -> FP32TensorV1:
        norm = self.norm(projected)
        _require(norm > 1.0e-6, "projected teacher is degenerate")
        return FP32TensorV1(
            projected.shape, tuple(_f32(item / norm) for item in projected.values)
        )

    def to_fp32_tensor(self, value: FP32TensorV1) -> FP32TensorV1:
        return value

    def norm(self, value: FP32TensorV1) -> float:
        return math.sqrt(math.fsum(item * item for item in value.values))

    def cosine(self, left: FP32TensorV1, right: FP32TensorV1) -> float:
        denominator = self.norm(left) * self.norm(right)
        _require(denominator > 0.0, "cosine operand is degenerate")
        return math.fsum(a * b for a, b in zip(left.values, right.values)) / denominator


class TorchOfficialNumericsV1:
    """Production numerics: call the exact frozen ``full30_action_learning`` core."""

    def __init__(self, *, expected_protocol_path: Optional[str | Path] = None) -> None:
        try:
            import torch
            import full30_action_learning_v1 as learning
        except ImportError as error:  # pragma: no cover - AUH integration
            raise Full30PsiOutMaterializationError(
                "official PyTorch/PsiOut core is unavailable"
            ) from error
        self.torch = torch
        self.learning = learning
        observed_path = Path(learning.__file__).resolve(strict=True)
        if expected_protocol_path is not None:
            _require(
                observed_path == Path(expected_protocol_path).resolve(strict=True),
                "imported PsiOut protocol differs from the sealed physical source",
            )
        _require(
            file_sha256(observed_path) == PINNED_PSIOUT_PROTOCOL_SHA256,
            "imported PsiOut protocol SHA differs",
        )

    def from_artifact(self, value: FP32TensorV1) -> Any:
        result = self.torch.tensor(value.values, dtype=self.torch.float32).reshape(value.shape)
        return result.contiguous()

    def mix(self, clean: Any, noise: Any, sigma: float) -> Any:
        _require(clean.shape == noise.shape, "clean/noise shape differs")
        sigma_tensor = self.torch.tensor(float(sigma), dtype=self.torch.float32)
        return ((1.0 - sigma_tensor) * clean + sigma_tensor * noise).float().contiguous()

    def delta(self, left: Any, right: Any) -> Any:
        return (left.float() - right.float()).float().contiguous()

    def psiout_raw(self, value: Any) -> Any:
        return self.learning.psiout_raw_v1(value.float().contiguous())[0].contiguous()

    def nuisance(self, camera_raw: Any, appearance_raw: Any) -> Any:
        return self.learning.build_nuisance_packet_v1(
            camera_raw.unsqueeze(0).contiguous(), appearance_raw.unsqueeze(0).contiguous()
        )

    def project(self, raw: Any, packet: Any) -> Any:
        return self.learning.project_nuisances_v1(
            raw.unsqueeze(0).contiguous(), packet
        )[0].contiguous()

    def unit(self, projected: Any) -> Any:
        return self.learning.teacher_unit_v1(projected.unsqueeze(0).contiguous())[0].contiguous()

    def to_fp32_tensor(self, value: Any) -> FP32TensorV1:
        tensor = value.detach().float().cpu().contiguous()
        _require(tuple(int(item) for item in tensor.shape) == QUOTIENT_SHAPE, "quotient shape differs")
        return FP32TensorV1(QUOTIENT_SHAPE, tuple(float(item) for item in tensor.reshape(-1).tolist()))

    def norm(self, value: Any) -> float:
        result = float(self.torch.linalg.vector_norm(value.float().reshape(-1)).item())
        _require(math.isfinite(result), "tensor norm is non-finite")
        return result

    def cosine(self, left: Any, right: Any) -> float:
        left_flat = left.float().reshape(-1)
        right_flat = right.float().reshape(-1)
        denominator = self.torch.linalg.vector_norm(left_flat) * self.torch.linalg.vector_norm(right_flat)
        _require(float(denominator.item()) > 0.0, "cosine operand is degenerate")
        return float(((left_flat * right_flat).sum() / denominator).item())


def frozen_compute_contract_v1() -> Mapping[str, Any]:
    value = {
        "schema_version": amplitude_authority.COMPUTE_CONTRACT_SCHEMA_VERSION,
        "model_eval": True,
        "torch_inference_mode": True,
        "official_frozen_native_only": True,
        "calibrator_peft_adapter_present": False,
        "frozen_effective_adapter_enabled": False,
        "frozen_effective_typed_patch_role_enabled": False,
        "base_compute_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "observer_output_dtype": "torch.float32",
        "observer_output_stage": POST_HEAD_STAGE,
        "observer_output_detached": True,
        "observer_output_contiguous": True,
        "same_state_counterfactual": True,
        "branch_and_noop_share_input_state": True,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "sp_order_contract": SP_ORDER_CONTRACT,
        "all_rank_consensus": True,
    }
    _closed(
        value,
        amplitude_authority._COMPUTE_CONTRACT_FIELDS,
        "materializer frozen compute contract",
    )
    return MappingProxyType(value)


_RUNTIME_PLAN_FIELDS = {
    "schema_version",
    "frozen_runtime_identity",
    "bernini_root",
    "veomni_root",
    "checkpoint_root",
    "checkpoint_content_manifest_path",
    "checkpoint_content_manifest_sha256",
    "psiout_protocol_path",
    "official_provider_source_path",
    "official_helper_sources",
    "runtime_plan_digest",
}
_HELPER_SOURCE_FIELDS = {"module", "path", "file_sha256"}
REQUIRED_HELPER_MODULES = (
    "clean_source_visual_context_training_v1",
    "dclr_runtime_contract",
    "full30_action_runtime_v1",
    "graft_phase_a_native_training_closure_v1",
    "infer_dclr_reward_runtime_smoke",
    "inference_sigma_strata",
    "packed_preservation_lora_v2",
    "packed_preservation_release_v2",
    "source_self_runtime",
    "temporal_counterfactual_action_scorer_v1",
    "train_lora",
    "train_packed_preservation_lora_v2",
)
_POPULATION_FIELDS = {
    "schema_version",
    "population_id",
    "record_count",
    "teacher_record_count",
    "amplitude_record_count",
    "teacher_cell_ids",
    "record_order_sha256",
    "finite_closed_population",
    "block_probe",
    "population_digest",
}
_REVIEW_BINDING_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "review_digest",
}
_MEDIA_FIELDS = {"path", "file_sha256"}
_CONDITION_FIELDS = {
    "schema_version",
    "role",
    "instruction",
    "instruction_utf8_sha256",
    "authority_path",
    "authority_file_sha256",
    "authority_digest_field",
    "authority_digest",
    "json_pointer",
    "text_field",
    "sha256_field",
    "control_anchor_id",
}
_NOISE_FIELDS = {
    "artifact",
    "seed",
    "generator",
}
_RECORD_FIELDS = {
    "schema_version",
    "record_id",
    "record_kind",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "analysis_split",
    "branch",
    "event_id",
    "actor_kind",
    "q0_id",
    "actor_id",
    "scene_id",
    "anchor_id",
    "anchor_iid",
    "pair_id",
    "source_iid",
    "review",
    "reviewed_media",
    "target_clean_latent",
    "target_clean_latent_authority",
    "source_clean_latent",
    "source_posterior_index0_path",
    "source_posterior_index0_sha256",
    "source_posterior_tensor_key",
    "noise",
    "conditions",
    "record_digest",
}
_OUTPUT_POLICY_FIELDS = {
    "schema_version",
    "create_only",
    "container_mode_octal",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "persisted_tensor_role",
}
_TOP_PLAN_FIELDS = {
    "schema_version",
    "plan_id",
    "status",
    "runtime",
    "population",
    "records",
    "output_policy",
    "plan_digest",
}


def _resolve_json_pointer(root: Any, pointer: Any, label: str) -> Any:
    if type(pointer) is not str or (pointer and not pointer.startswith("/")):
        fail(f"{label} must be an empty or absolute JSON pointer")
    current = root
    if not pointer:
        return current
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            _require(token in current, f"{label} token is absent: {token!r}")
            current = current[token]
        elif isinstance(current, list):
            _require(token.isdecimal(), f"{label} list token differs")
            index = int(token)
            _require(0 <= index < len(current), f"{label} list index differs")
            current = current[index]
        else:
            fail(f"{label} traverses a scalar")
    return current


def _validate_condition(value: Any, *, label: str) -> Mapping[str, Any]:
    row = _closed(value, _CONDITION_FIELDS, label)
    _require(row["schema_version"] == PLAN_CONDITION_SCHEMA_VERSION, f"{label} schema differs")
    _safe_id(row["role"], f"{label}.role")
    instruction = row["instruction"]
    if type(instruction) is not str or not instruction or instruction.strip() != instruction:
        fail(f"{label}.instruction differs")
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    _require(row["instruction_utf8_sha256"] == instruction_sha, f"{label} instruction SHA differs")
    authority, _, _ = _load_json_file(
        row["authority_path"], row["authority_file_sha256"], f"{label}.authority"
    )
    digest_field = row["authority_digest_field"]
    if type(digest_field) is not str or digest_field not in authority:
        fail(f"{label}.authority digest field differs")
    _verify_seal(authority, digest_field, f"{label}.authority")
    _require(authority[digest_field] == row["authority_digest"], f"{label}.authority digest binding differs")
    bound = _resolve_json_pointer(authority, row["json_pointer"], f"{label}.json_pointer")
    if not isinstance(bound, Mapping):
        fail(f"{label}.json_pointer does not select an object")
    text_field = row["text_field"]
    sha_field = row["sha256_field"]
    if type(text_field) is not str or type(sha_field) is not str:
        fail(f"{label} authority field names differ")
    _require(bound.get(text_field) == instruction, f"{label} authority instruction differs")
    _require(bound.get(sha_field) == instruction_sha, f"{label} authority instruction SHA differs")
    if row["role"] in teacher_authority.WRONG_CONTROL_TYPES:
        _safe_id(row["control_anchor_id"], f"{label}.control_anchor_id")
    else:
        _require(row["control_anchor_id"] is None, f"{label}.control_anchor_id must be null")
    return row


def _validate_anchor_latent_authority_v1(
    value: Any,
    *,
    artifact: Mapping[str, Any],
    reviewed_media: Mapping[str, Any],
    expected_checkpoint_tree_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    row = _closed(value, _LATENT_AUTHORITY_FIELDS, label)
    _require(row["schema_version"] == PLAN_LATENT_AUTHORITY_SCHEMA_VERSION, f"{label} schema differs")
    authority, _, _ = _load_json_file(
        row["path"], row["file_sha256"], f"{label}.authority"
    )
    digest_field = row["digest_field"]
    if type(digest_field) is not str or digest_field not in authority:
        fail(f"{label}.digest_field differs")
    _verify_seal(authority, digest_field, f"{label}.authority")
    _require(authority[digest_field] == row["digest"], f"{label}.digest binding differs")
    media = _resolve_json_pointer(
        authority, row["media_json_pointer"], f"{label}.media_json_pointer"
    )
    latent = _resolve_json_pointer(
        authority, row["latent_json_pointer"], f"{label}.latent_json_pointer"
    )
    checkpoint_tree_sha256 = _resolve_json_pointer(
        authority,
        row["checkpoint_tree_sha256_json_pointer"],
        f"{label}.checkpoint_tree_sha256_json_pointer",
    )
    if not isinstance(media, Mapping) or not isinstance(latent, Mapping):
        fail(f"{label} pointers must select artifact objects")
    _require(
        checkpoint_tree_sha256 == expected_checkpoint_tree_sha256,
        f"{label} generation checkpoint tree differs",
    )
    media_path = _plain_file(media.get("path"), f"{label}.media.path")
    latent_path = _plain_file(latent.get("path"), f"{label}.latent.path")
    _require(
        media_path == _plain_file(reviewed_media["path"], f"{label}.reviewed_media")
        and media.get("sha256") == reviewed_media["file_sha256"],
        f"{label} reviewed media binding differs",
    )
    _require(
        latent_path == _plain_file(artifact["path"], f"{label}.artifact")
        and latent.get("sha256") == artifact["file_sha256"]
        and latent.get("tensor_key") == artifact["tensor_key"]
        and latent.get("raw_value_sha256") == artifact["tensor_raw_sha256"]
        and latent.get("shape") == artifact["shape"]
        and latent.get("stored_dtype") == "torch.float32"
        and latent.get("coordinate") == "bernini_normalized_clean_vae_latent"
        and latent.get("native_sampler_before_vae_decode") is True
        and latent.get("mp4_decode_reencode_used") is False,
        f"{label} normalized predecode latent binding differs",
    )
    return row


def _validate_review(
    value: Any,
    *,
    record: Mapping[str, Any],
    media_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    binding = _closed(value, _REVIEW_BINDING_FIELDS, label)
    _require(binding["schema_version"] == PLAN_REVIEW_SCHEMA_VERSION, f"{label} schema differs")
    review, _, _ = _load_json_file(binding["path"], binding["file_sha256"], label)
    _verify_seal(review, "review_digest", label)
    _require(review["review_digest"] == binding["review_digest"], f"{label} digest binding differs")
    if record["record_kind"] == "teacher_anchor":
        expected_fields = teacher_authority._REPRESENTATION_REVIEW_FIELDS
        review = _closed(review, expected_fields, label)
        _require(review["schema_version"] == teacher_authority.REPRESENTATION_REVIEW_SCHEMA, f"{label} teacher review schema differs")
        expected = {
            "evidence_id": record["evidence_id"],
            "anchor_id": record["anchor_id"],
            "anchor_video_sha256": media_sha256,
            "anchor_split": record["analysis_split"],
            "branch": record["branch"],
            "event_id": record["event_id"],
            "actor_kind": record["actor_kind"],
            "q0_id": record["q0_id"],
            "actor_id": record["actor_id"],
            "scene_id": record["scene_id"],
        }
        for field_name, expected_value in expected.items():
            _require(review[field_name] == expected_value, f"{label}.{field_name} differs")
        _require(review["frame_count"] == 81 and float(review["fps"]) == 25.0, f"{label} is not full81/25fps")
        for field_name in (
            "entire_full81_video_viewed",
            "independent_reviewer",
            "reviewer_blinded_to_teacher_cell",
            "reviewer_blinded_to_representation_metrics",
            "sealed_before_sidecar_extraction",
            "sealed_before_representation_admission",
            "target_event_verified",
            "actor_identity_verified",
            "scene_verified",
        ):
            _require(review[field_name] is True, f"{label}.{field_name} is not true")
    else:
        review = _closed(review, amplitude_authority._REVIEW_FIELDS, label)
        _require(review["schema_version"] == amplitude_authority.REVIEW_SCHEMA_VERSION, f"{label} amplitude review schema differs")
        expected = {
            "evidence_id": record["evidence_id"],
            "pair_id": record["pair_id"],
            "source_iid": record["source_iid"],
            "branch": record["branch"],
            "baseline_output_sha256": media_sha256,
            "frame_count": 81,
            "sampler_steps": 40,
        }
        for field_name, expected_value in expected.items():
            _require(review[field_name] == expected_value, f"{label}.{field_name} differs")
        _require(float(review["fps"]) == 25.0, f"{label}.fps differs")
        _require(review["action_result"] in ("partial", "pass"), f"{label} is not a calibrator review")
        for field_name in (
            "entire_full81_video_viewed",
            "independent_reviewer",
            "reviewer_blinded_to_amplitude_metrics",
            "sealed_before_sidecar_extraction",
            "sealed_before_optimizer_authority",
        ):
            _require(review[field_name] is True, f"{label}.{field_name} is not true")
    return review


def _validate_runtime_plan(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _RUNTIME_PLAN_FIELDS, "runtime")
    _verify_seal(row, "runtime_plan_digest", "runtime")
    _require(row["schema_version"] == PLAN_RUNTIME_SCHEMA_VERSION, "runtime plan schema differs")
    identity = row["frozen_runtime_identity"]
    try:
        amplitude_authority._validate_runtime_identity(identity)
    except Exception as error:
        raise Full30PsiOutMaterializationError("frozen runtime v2 identity differs") from error
    _require(identity["official_provider_abi"] == OFFICIAL_PROVIDER_ABI, "official provider ABI differs")
    _require(identity["compute_contract"] == dict(frozen_compute_contract_v1()), "frozen compute contract differs")
    _require(identity["compute_contract_digest"] == object_sha256(identity["compute_contract"]), "compute contract digest differs")
    bernini_root = _plain_directory(row["bernini_root"], "runtime.bernini_root")
    veomni_root = _plain_directory(row["veomni_root"], "runtime.veomni_root")
    checkpoint = _plain_directory(row["checkpoint_root"], "runtime.checkpoint_root")
    _read_stable_plain_file(
        row["checkpoint_content_manifest_path"],
        row["checkpoint_content_manifest_sha256"],
        label="runtime.checkpoint_content_manifest",
        maximum_bytes=MAX_JSON_BYTES,
    )
    transformer_config = checkpoint / "transformer" / "config.json"
    _require(
        transformer_config.is_file()
        and not transformer_config.is_symlink()
        and file_sha256(transformer_config) == identity["transformer_config_sha256"],
        "runtime transformer config identity differs",
    )
    psiout_path, _ = _read_stable_plain_file(
        row["psiout_protocol_path"],
        identity["psiout_protocol_sha256"],
        label="runtime.PsiOut protocol",
        maximum_bytes=MAX_JSON_BYTES,
    )
    provider_path, _ = _read_stable_plain_file(
        row["official_provider_source_path"],
        identity["official_provider_source_sha256"],
        label="runtime.official provider source",
        maximum_bytes=MAX_JSON_BYTES,
    )
    _require(provider_path == Path(__file__).resolve(strict=True), "official provider source is not this materializer")
    _require(psiout_path.name == "full30_action_learning_v1.py", "PsiOut protocol source differs")
    _require(
        identity["psiout_protocol_sha256"] == PINNED_PSIOUT_PROTOCOL_SHA256,
        "PsiOut protocol is not the frozen full30 core",
    )
    helpers = row["official_helper_sources"]
    if type(helpers) is not list or len(helpers) != len(REQUIRED_HELPER_MODULES):
        fail("runtime official helper-source closure differs")
    helper_root = Path(__file__).resolve(strict=True).parent
    for ordinal, module_name in enumerate(REQUIRED_HELPER_MODULES):
        helper = _closed(
            helpers[ordinal], _HELPER_SOURCE_FIELDS, f"runtime.official_helper_sources[{ordinal}]"
        )
        _require(helper["module"] == module_name, "runtime official helper module/order differs")
        helper_path, _ = _read_stable_plain_file(
            helper["path"],
            helper["file_sha256"],
            label=f"runtime helper {module_name}",
            maximum_bytes=MAX_JSON_BYTES,
        )
        _require(
            helper_path == helper_root / f"{module_name}.py",
            f"runtime helper {module_name} physical path differs",
        )
        if module_name == "full30_action_runtime_v1":
            _require(
                helper["file_sha256"] == PINNED_POST_HEAD_RUNTIME_SHA256,
                "post-head runtime helper is not the frozen v2-contract implementation",
            )
    _require(identity["bernini_revision"] == _revision(identity["bernini_revision"], "runtime Bernini revision"), "runtime Bernini revision differs")
    _require(identity["veomni_revision"] == _revision(identity["veomni_revision"], "runtime VeOmni revision"), "runtime VeOmni revision differs")
    _require(
        identity["sigma_table_sha256"]
        == _pinned_sigma_module().SCHEDULE_SHA256
        == PINNED_SIGMA_TABLE_SHA256,
        "runtime sigma table differs",
    )
    del bernini_root, veomni_root
    return row


def _pinned_sigma_module() -> Any:
    try:
        import inference_sigma_strata as sigma
    except ImportError:  # pragma: no cover - package mode
        from . import inference_sigma_strata as sigma
    _require(tuple(SIGMA_INDICES) == (4, 12, 20, 28, 35, 38), "materializer sigma indices differ")
    return sigma


def sigma_authority_receipt_v1() -> Mapping[str, Any]:
    """Return the complete pinned UniPC schedule plus the six observed rows."""

    sigma = _pinned_sigma_module()
    rows: list[Mapping[str, Any]] = []
    for sigma_index in SIGMA_INDICES:
        row = {
            "sigma_index": sigma_index,
            "timestep": int(sigma.PINNED_TIMESTEPS[sigma_index]),
            "sigma_float32_be_hex": str(
                sigma.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[sigma_index]
            ),
        }
        rows.append(_closed(row, _SIGMA_AUTHORITY_ROW_FIELDS, "sigma authority row"))
    unsigned = {
        "schema_version": SIGMA_AUTHORITY_RECEIPT_SCHEMA_VERSION,
        "schedule_schema_version": sigma.SCHEDULE_SCHEMA,
        "schedule_sha256": sigma.SCHEDULE_SHA256,
        "scheduler_class": sigma.SCHEDULER_CLASS,
        "num_train_timesteps": sigma.NUM_TRAIN_TIMESTEPS,
        "num_inference_steps": sigma.NUM_INFERENCE_STEPS,
        "flow_shift_float64_hex": sigma.FLOW_SHIFT.hex(),
        "timesteps_int64": list(sigma.PINNED_TIMESTEPS),
        "positive_sigmas_float32_be_hex": list(
            sigma.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma_float32_be_hex": sigma.TERMINAL_SIGMA_FLOAT32_HEX,
        "materialized_sigma_indices": list(SIGMA_INDICES),
        "materialized_rows": rows,
    }
    value = seal_record(unsigned, "sigma_authority_digest")
    _closed(value, _SIGMA_AUTHORITY_RECEIPT_FIELDS, "sigma authority receipt")
    _require(
        value["schedule_sha256"] == PINNED_SIGMA_TABLE_SHA256,
        "sigma authority receipt schedule differs",
    )
    return MappingProxyType(value)


def _validate_record(
    value: Any, *, ordinal: int, expected_checkpoint_tree_sha256: str
) -> Mapping[str, Any]:
    label = f"records[{ordinal}]"
    row = _closed(value, _RECORD_FIELDS, label)
    _verify_seal(row, "record_digest", label)
    _require(row["schema_version"] == PLAN_RECORD_SCHEMA_VERSION, f"{label} schema differs")
    record_id = _safe_id(row["record_id"], f"{label}.record_id")
    _safe_id(row["evidence_id"], f"{label}.evidence_id")
    _safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
    _require(row["record_kind"] in RECORD_KINDS, f"{label}.record_kind differs")
    _require(row["branch"] in BRANCHES, f"{label}.branch differs")
    _require(row["analysis_split"] in ("fit", "confirmation"), f"{label}.analysis_split differs")
    for field_name in ("event_id", "actor_kind", "q0_id", "actor_id", "scene_id"):
        _safe_id(row[field_name], f"{label}.{field_name}")
    media = _closed(row["reviewed_media"], _MEDIA_FIELDS, f"{label}.reviewed_media")
    media_path, _ = _read_stable_plain_file(
        media["path"], media["file_sha256"], label=f"{label}.reviewed_media", maximum_bytes=MAX_TENSOR_FILE_BYTES
    )
    media_sha = _sha(media["file_sha256"], f"{label}.reviewed_media SHA")
    _validate_review(row["review"], record=row, media_sha256=media_sha, label=f"{label}.review")
    target = load_fp32_artifact_v1(row["target_clean_latent"], label=f"{label}.target_clean_latent")
    noise_row = _closed(row["noise"], _NOISE_FIELDS, f"{label}.noise")
    noise = load_fp32_artifact_v1(noise_row["artifact"], label=f"{label}.noise.artifact")
    _require(target.shape == noise.shape, f"{label} clean/noise geometry differs")
    _require(noise_row["generator"] == "torch-cpu-generator-manual-seed-randn-fp32-v1", f"{label}.noise generator differs")
    if row["record_kind"] == "teacher_anchor":
        _require(row["evidence_role"] in TEACHER_EVIDENCE_ROLES, f"{label}.evidence_role differs")
        for field_name in ("anchor_id", "anchor_iid"):
            _safe_id(row[field_name], f"{label}.{field_name}")
        _iid(row["anchor_iid"], f"{label}.anchor_iid")
        _require(row["pair_id"] is None and row["source_iid"] is None, f"{label} teacher pair/source fields must be null")
        _require(row["source_clean_latent"] is None, f"{label} teacher must be source-free")
        for field_name in (
            "source_posterior_index0_path",
            "source_posterior_index0_sha256",
            "source_posterior_tensor_key",
        ):
            _require(row[field_name] is None, f"{label}.{field_name} must be null")
        _validate_anchor_latent_authority_v1(
            row["target_clean_latent_authority"],
            artifact=row["target_clean_latent"],
            reviewed_media=row["reviewed_media"],
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
            label=f"{label}.target_clean_latent_authority",
        )
        expected_seed = teacher_noise_seed_v1(str(row["teacher_cell_id"]), str(row["branch"]))
        required_roles = TEACHER_CONDITION_ROLES if row["evidence_role"] == "teacher_origin" else CROSS_CONDITION_ROLES
    else:
        _require(row["evidence_role"] == "calibrator", f"{label}.evidence_role differs")
        _require(row["anchor_id"] is None and row["anchor_iid"] is None, f"{label} amplitude anchor fields must be null")
        _safe_id(row["pair_id"], f"{label}.pair_id")
        _safe_id(row["source_iid"], f"{label}.source_iid")
        _iid(row["source_iid"], f"{label}.source_iid")
        _require(
            row["target_clean_latent_authority"] is None,
            f"{label}.target_clean_latent_authority must be null for physical source replay",
        )
        posterior_path, _ = _read_stable_plain_file(
            row["source_posterior_index0_path"],
            row["source_posterior_index0_sha256"],
            label=f"{label}.source_posterior_index0",
            maximum_bytes=MAX_TENSOR_FILE_BYTES,
        )
        _require(
            posterior_path.name
            == f"{row['source_iid']}.source-posterior-index0.pt",
            f"{label}.source_posterior_index0 filename differs",
        )
        _safe_id(row["source_posterior_tensor_key"], f"{label}.source_posterior_tensor_key")
        source = load_fp32_artifact_v1(row["source_clean_latent"], label=f"{label}.source_clean_latent")
        _require(source.shape == target.shape, f"{label} source/target geometry differs")
        _require(
            source.raw_sha256() == target.raw_sha256(),
            f"{label} same-mode source/target clean bytes differ",
        )
        _require(row["analysis_split"] == "fit", f"{label} amplitude calibrator is not fit")
        expected_seed = amplitude_noise_seed_v1(str(row["pair_id"]))
        required_roles = AMPLITUDE_CONDITION_ROLES
    _require(noise_row["seed"] == expected_seed, f"{label}.noise seed differs")
    conditions = row["conditions"]
    if type(conditions) is not list or len(conditions) != len(required_roles):
        fail(f"{label}.conditions closure differs")
    validated = [
        _validate_condition(item, label=f"{label}.conditions[{index}]")
        for index, item in enumerate(conditions)
    ]
    _require(tuple(item["role"] for item in validated) == tuple(required_roles), f"{label}.condition order differs")
    _require(len({item["instruction_utf8_sha256"] for item in validated}) == len(validated), f"{label}.condition instructions alias")
    if row["record_kind"] == "amplitude_calibrator":
        noop = next(item for item in validated if item["role"] == "noop")
        _require(noop["instruction"] == EXACT_NOOP_INSTRUCTION, f"{label} noop instruction differs from full30 trainer")
    elif row["evidence_role"] == "teacher_origin":
        control_ids = [
            item["control_anchor_id"]
            for item in validated
            if item["role"] in teacher_authority.WRONG_CONTROL_TYPES
        ]
        _require(
            len(control_ids) == len(set(control_ids)) == len(teacher_authority.WRONG_CONTROL_TYPES),
            f"{label} wrong-control anchor ids are not distinct",
        )
    _require(media_path.suffix.lower() == ".mp4", f"{label}.reviewed_media is not MP4")
    del record_id
    return row


def validate_materialization_plan_v1(value: Any) -> Mapping[str, Any]:
    plan = _closed(value, _TOP_PLAN_FIELDS, "materialization plan")
    _verify_seal(plan, "plan_digest", "materialization plan")
    _require(plan["schema_version"] == PLAN_SCHEMA_VERSION, "materialization plan schema differs")
    _safe_id(plan["plan_id"], "materialization plan id")
    _require(plan["status"] == "SEALED_REVIEWED_PRE_OPTIMIZER", "materialization plan is not review-sealed")
    runtime = _validate_runtime_plan(plan["runtime"])
    records_value = plan["records"]
    if type(records_value) is not list or not records_value:
        fail("materialization records must be one finite non-empty list")
    records = tuple(
        _validate_record(
            item,
            ordinal=index,
            expected_checkpoint_tree_sha256=runtime["frozen_runtime_identity"][
                "official_checkpoint_tree_sha256"
            ],
        )
        for index, item in enumerate(records_value)
    )
    _require(
        len({str(row["record_id"]) for row in records}) == len(records),
        "materialization record id is duplicated",
    )
    _require(
        len({str(row["evidence_id"]) for row in records}) == len(records),
        "materialization evidence id is duplicated",
    )
    teacher_rows = [row for row in records if row["record_kind"] == "teacher_anchor"]
    _require(
        len({str(row["anchor_id"]) for row in teacher_rows}) == len(teacher_rows),
        "teacher anchor id is duplicated",
    )
    _require(
        len({str(row["review"]["review_digest"]) for row in teacher_rows}) == len(teacher_rows),
        "teacher review digest is duplicated",
    )
    reused_video_bindings: dict[str, tuple[Any, ...]] = {}
    for row in teacher_rows:
        intrinsic = (
            row["anchor_iid"],
            row["analysis_split"],
            row["branch"],
            row["event_id"],
            row["actor_kind"],
            row["actor_id"],
            row["scene_id"],
        )
        previous = reused_video_bindings.setdefault(
            str(row["reviewed_media"]["file_sha256"]), intrinsic
        )
        _require(previous == intrinsic, "reused teacher video has different intrinsic binding")
    teachers_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    amplitudes_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["teacher_cell_id"]), str(row["branch"]))
        target = teachers_by_key if row["record_kind"] == "teacher_anchor" else amplitudes_by_key
        target.setdefault(key, []).append(row)
    for key, rows in teachers_by_key.items():
        roles = [str(row["evidence_role"]) for row in rows]
        _require(
            sorted(roles) == sorted(TEACHER_EVIDENCE_ROLES),
            f"teacher population {key!r} must contain one origin and one cross anchor",
        )
        origin = next(row for row in rows if row["evidence_role"] == "teacher_origin")
        cross = next(row for row in rows if row["evidence_role"] == "same_event_cross_anchor")
        for field_name in ("event_id", "actor_kind", "q0_id"):
            _require(origin[field_name] == cross[field_name], f"teacher population {key!r} {field_name} differs")
        _require(origin["anchor_iid"] != cross["anchor_iid"], f"teacher population {key!r} reuses anchor IID")
        _require(origin["actor_id"] != cross["actor_id"], f"teacher population {key!r} actor is not cross-anchor")
        _require(origin["scene_id"] != cross["scene_id"], f"teacher population {key!r} scene is not cross-anchor")
        control_ids = {
            str(condition["control_anchor_id"])
            for condition in origin["conditions"]
            if condition["role"] in teacher_authority.WRONG_CONTROL_TYPES
        }
        _require(
            control_ids.isdisjoint({str(origin["anchor_id"]), str(cross["anchor_id"])}),
            f"teacher population {key!r} wrong control reuses correct anchor",
        )
    for key, rows in amplitudes_by_key.items():
        _require(key in teachers_by_key, f"amplitude population {key!r} has no in-plan teacher nuisance authority")
        origin = next(
            row
            for row in teachers_by_key[key]
            if row["evidence_role"] == "teacher_origin"
        )
        _require(
            origin["analysis_split"] == "fit",
            f"amplitude population {key!r} does not reference a fit teacher origin",
        )
        _require(len(rows) == amplitude_authority.CALIBRATORS_PER_BUNDLE, f"amplitude population {key!r} must contain exactly two calibrators")
        _require(len({str(row["pair_id"]) for row in rows}) == len(rows), f"amplitude population {key!r} reuses pair")
        _require(len({str(row["source_iid"]) for row in rows}) == len(rows), f"amplitude population {key!r} reuses source")
        _require(len({str(row["review"]["review_digest"]) for row in rows}) == len(rows), f"amplitude population {key!r} reuses review")
        _require(len({str(row["reviewed_media"]["file_sha256"]) for row in rows}) == len(rows), f"amplitude population {key!r} reuses reviewed output")
        _require(len({str(row["noise"]["artifact"]["tensor_raw_sha256"]) for row in rows}) == len(rows), f"amplitude population {key!r} reuses calibrator noise")
    population = _closed(plan["population"], _POPULATION_FIELDS, "population")
    _verify_seal(population, "population_digest", "population")
    _require(population["schema_version"] == PLAN_POPULATION_SCHEMA_VERSION, "population schema differs")
    _safe_id(population["population_id"], "population id")
    teacher_count = sum(row["record_kind"] == "teacher_anchor" for row in records)
    amplitude_count = len(records) - teacher_count
    expected_population = {
        "record_count": len(records),
        "teacher_record_count": teacher_count,
        "amplitude_record_count": amplitude_count,
        "teacher_cell_ids": sorted({str(row["teacher_cell_id"]) for row in records}, key=lambda item: item.encode("utf-8")),
        "record_order_sha256": object_sha256([str(row["record_id"]) for row in records]),
        "finite_closed_population": True,
        "block_probe": False,
    }
    for field_name, expected in expected_population.items():
        _require(population[field_name] == expected, f"population.{field_name} differs")
    policy = _closed(plan["output_policy"], _OUTPUT_POLICY_FIELDS, "output_policy")
    expected_policy = {
        "schema_version": PLAN_OUTPUT_POLICY_SCHEMA_VERSION,
        "create_only": True,
        "container_mode_octal": "0600",
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
        "persisted_tensor_role": "detached-post-head-psiout-or-same-mode-amplitude-evidence-only",
    }
    _require(policy == expected_policy, "materialization output policy differs")
    return plan


def load_materialization_plan_file_v1(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    value, _, _ = _load_json_file(str(path), expected_sha256, "materialization plan")
    return validate_materialization_plan_v1(value)


@dataclass(frozen=True)
class PreparedSameStateV1:
    record_id: str
    sigma_index: int
    sigma_float32_be_hex: str
    timestep: int
    clean_raw_sha256: str
    source_raw_sha256: Optional[str]
    noise_raw_sha256: str
    x_sigma_raw_sha256: str
    state_digest: str
    receipt: Mapping[str, Any]
    opaque_state: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class FrozenForwardResultV1:
    velocity: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


class FrozenForwardProviderV1(Protocol):
    is_official: bool

    def verify_noise_authority_v1(
        self, *, record: Mapping[str, Any], noise: Any
    ) -> Mapping[str, Any]: ...

    def prepare_same_state_v1(
        self,
        *,
        record: Mapping[str, Any],
        clean: Any,
        source: Optional[Any],
        noise: Any,
        x_sigma: Any,
        sigma_index: int,
        sigma: float,
        timestep: int,
    ) -> PreparedSameStateV1: ...

    def forward_post_head_v1(
        self,
        *,
        state: PreparedSameStateV1,
        condition: Mapping[str, Any],
    ) -> FrozenForwardResultV1: ...

    def barrier_v1(self) -> None: ...

    def consensus_digest_v1(self, digest: str, label: str) -> None: ...

    def broadcast_rank0_v1(self, value: Any) -> Any: ...

    def close(self) -> None: ...


@dataclass
class _OfficialOpaqueStateV1:
    noisy_latents: Any
    rotary_embs: Any
    target_mask: Any
    timestep: Any
    target_tokens: int
    spatial_shape: tuple[int, int, int, int, int]
    input_objects: tuple[Any, ...]
    input_hashes: Mapping[str, str]
    record_kind: str


class OfficialWorld4SP4FrozenProviderV1:
    """Real Bernini provider shared by T2V teachers and RV2V calibrators."""

    is_official = True

    def __init__(self, plan: Mapping[str, Any]) -> None:  # pragma: no cover - AUH GPU
        runtime = plan["runtime"]
        identity = runtime["frozen_runtime_identity"]
        try:
            import torch
            import torch.distributed as dist
            from transformers import AutoTokenizer
            import clean_source_visual_context_training_v1 as source_data
            import dclr_runtime_contract
            import full30_action_runtime_v1 as branch_runtime
            import graft_phase_a_native_training_closure_v1 as graft_helper
            import infer_dclr_reward_runtime_smoke as shared_runtime
            import inference_sigma_strata as sigma_runtime
            import packed_preservation_lora_v2 as packed_core
            import packed_preservation_release_v2 as checkpoint_release
            import source_self_runtime
            import temporal_counterfactual_action_scorer_v1 as t2v_helper
            import train_lora as legacy
            import train_packed_preservation_lora_v2 as packed_trainer
        except ImportError as error:
            raise Full30PsiOutMaterializationError(
                "official Bernini materialization dependencies are unavailable"
            ) from error
        self.torch = torch
        self.dist = dist
        self.dclr_runtime_contract = dclr_runtime_contract
        self.branch_runtime = branch_runtime
        self.shared_runtime = shared_runtime
        self.packed_core = packed_core
        self.packed_trainer = packed_trainer
        self.source_self_runtime = source_self_runtime
        self.source_data = source_data
        self.identity = identity
        self._closed = False
        imported_helpers = {
            "clean_source_visual_context_training_v1": source_data,
            "dclr_runtime_contract": dclr_runtime_contract,
            "full30_action_runtime_v1": branch_runtime,
            "graft_phase_a_native_training_closure_v1": graft_helper,
            "infer_dclr_reward_runtime_smoke": shared_runtime,
            "inference_sigma_strata": sigma_runtime,
            "packed_preservation_lora_v2": packed_core,
            "packed_preservation_release_v2": checkpoint_release,
            "source_self_runtime": source_self_runtime,
            "temporal_counterfactual_action_scorer_v1": t2v_helper,
            "train_lora": legacy,
            "train_packed_preservation_lora_v2": packed_trainer,
        }
        planned_helpers = {
            str(row["module"]): row for row in runtime["official_helper_sources"]
        }
        _require(set(imported_helpers) == set(planned_helpers), "official imported helper closure differs")
        for module_name, module in imported_helpers.items():
            observed_path = Path(module.__file__).resolve(strict=True)
            _require(
                observed_path == Path(planned_helpers[module_name]["path"]).resolve(strict=True)
                and file_sha256(observed_path) == planned_helpers[module_name]["file_sha256"],
                f"imported official helper {module_name} differs from its sealed source",
            )

        if (
            not torch.cuda.is_available()
            or getattr(torch.version, "hip", None) is None
            or dist.is_initialized()
        ):
            fail("official provider requires fresh AUH ROCm WORLD4")
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world = int(os.environ.get("WORLD_SIZE", "-1"))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        _require(
            world == WORLD_SIZE
            and local_world == WORLD_SIZE
            and 0 <= rank < WORLD_SIZE
            and 0 <= local_rank < WORLD_SIZE,
            "official provider requires one-node WORLD4/SP4",
        )
        self.rank = rank
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=120),
            rank=rank,
            world_size=world,
        )
        self.device = torch.device("cuda", local_rank)

        try:
            bernini_root, veomni_root, bernini_revision, veomni_revision = (
                legacy.validate_source_trees(
                    runtime["bernini_root"],
                    runtime["veomni_root"],
                    expected_bernini_commit=identity["bernini_revision"],
                    expected_veomni_commit=identity["veomni_revision"],
                )
            )
            checkpoint, transformer_config = legacy.validate_checkpoint(
                runtime["checkpoint_root"]
            )
        except Exception as error:
            raise Full30PsiOutMaterializationError(
                f"official source/checkpoint validation failed: {error}"
            ) from error
        _require(
            bernini_revision == identity["bernini_revision"]
            and veomni_revision == identity["veomni_revision"],
            "official source revisions differ",
        )
        checkpoint_receipt: list[Any] = [None]
        if rank == 0:
            try:
                checkpoint_receipt[0] = {
                    "ok": True,
                    "value": checkpoint_release.validate_checkpoint_content(
                        checkpoint,
                        Path(runtime["checkpoint_content_manifest_path"]),
                        expected_manifest_sha256=runtime[
                            "checkpoint_content_manifest_sha256"
                        ],
                    ),
                }
            except Exception as error:
                checkpoint_receipt[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(checkpoint_receipt, src=0)
        _require(
            isinstance(checkpoint_receipt[0], Mapping)
            and checkpoint_receipt[0].get("ok") is True
            and checkpoint_receipt[0]["value"]["tree_sha256"]
            == identity["official_checkpoint_tree_sha256"],
            f"official checkpoint content differs: {checkpoint_receipt[0]!r}",
        )
        try:
            vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
        except Exception as error:
            raise Full30PsiOutMaterializationError(
                f"official VAE normalization authority failed: {error}"
            ) from error
        _require(z_dim == LATENT_CHANNELS, "official VAE latent channel authority differs")
        self.normalized_source_by_iid: dict[str, Any] = {}
        source_bindings: dict[str, Mapping[str, Any]] = {}
        for record in plan["records"]:
            if record["record_kind"] != "amplitude_calibrator":
                continue
            source_iid = str(record["source_iid"])
            binding = {
                "posterior_path": record["source_posterior_index0_path"],
                "posterior_sha256": record["source_posterior_index0_sha256"],
                "posterior_tensor_key": record["source_posterior_tensor_key"],
                "normalized_raw_sha256": record["source_clean_latent"]["tensor_raw_sha256"],
            }
            previous = source_bindings.setdefault(source_iid, binding)
            _require(previous == binding, "reused amplitude source authority differs")
            if source_iid in self.normalized_source_by_iid:
                continue
            _, posterior_raw = _read_stable_plain_file(
                record["source_posterior_index0_path"],
                record["source_posterior_index0_sha256"],
                label=f"official source posterior {source_iid}",
                maximum_bytes=MAX_TENSOR_FILE_BYTES,
            )
            try:
                parameters = source_data._decode_source_posterior_parameters(
                    posterior_raw, iid=source_iid
                )
            except Exception as error:
                raise Full30PsiOutMaterializationError(
                    f"official source posterior decode failed for {source_iid}: {error}"
                ) from error
            normalized = (
                (parameters[:, :LATENT_CHANNELS].float().contiguous() - vae_mean)
                / vae_std
            ).detach().float().contiguous()
            artifact = load_fp32_artifact_v1(
                record["source_clean_latent"],
                label=f"official normalized source {source_iid}",
            )
            artifact_tensor = torch.tensor(
                artifact.values, dtype=torch.float32
            ).reshape(artifact.shape).contiguous()
            _require(
                torch.equal(normalized, artifact_tensor)
                and _tensor_raw_sha256(normalized, label="recomputed normalized source")
                == record["source_clean_latent"]["tensor_raw_sha256"],
                f"normalized source does not replay physical posterior index0 for {source_iid}",
            )
            self.normalized_source_by_iid[source_iid] = normalized
        legacy.activate_source_trees(bernini_root, veomni_root)
        # Import no Bernini symbol until both source trees and their revisions
        # have been authenticated and placed first on ``sys.path``.
        from bernini.parallel import init_parallel_state
        from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
        from bernini.models.transformer_wan import WanRotaryPosEmbed

        init_parallel_state(ulysses_size=SP_SIZE)
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        with packed_trainer.serialized_model_load():
            renderer = BerniniRendererModel(config)
            renderer.requires_grad_(False)
            renderer.eval()
            renderer.to(self.device)
        self.renderer = renderer
        self.transformer = renderer.diff_dec.transformer
        _require(
            self.transformer is not None
            and renderer.diff_dec.transformer_2 is None
            and not renderer.training
            and not any(parameter.requires_grad for parameter in renderer.parameters()),
            "official provider model is not one frozen eval Bernini-R 1.3B",
        )
        parameter_names = tuple(name for name, _ in renderer.named_parameters())
        _require(
            not hasattr(renderer, "peft_config")
            and not any(
                token in name
                for name in parameter_names
                for token in ("lora_", "source_delta", "target_delta", "role_embedding")
            ),
            "official calibrator unexpectedly contains PEFT/typed parameters",
        )
        self.rope = WanRotaryPosEmbed(
            128, (1, 2, 2), 1024, use_src_id_rotary_emb=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            padding_side="right",
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
        )
        condition_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        for record in plan["records"]:
            for condition in record["conditions"]:
                key = (record["record_kind"], condition["instruction_utf8_sha256"])
                previous = condition_rows.setdefault(key, condition)
                _require(previous["instruction"] == condition["instruction"], "condition SHA aliases text")
        self.conditions: dict[tuple[str, str], Any] = {}
        with torch.inference_mode():
            for (record_kind, instruction_sha), condition in condition_rows.items():
                instruction = condition["instruction"]
                if record_kind == "teacher_anchor":
                    ids, mask = t2v_helper._frozen_d541801_runtime().native_generation.legacy._tokenize_training_prompt(
                        tokenizer, instruction
                    )
                    text_embs = renderer.encode_prompt(
                        ids.to(self.device), mask.to(self.device)
                    ).detach()
                    _require(
                        tuple(int(item) for item in text_embs.shape) == (1, 512, 4096),
                        "T2V condition geometry differs",
                    )
                    text_lens = [512]
                else:
                    tokenized = source_self_runtime.tokenize_generic_instruction(
                        tokenizer, instruction, self.device
                    )
                    text_lens, text_embs = renderer.get_t5_text_embeddings(
                        tokenized["input_ids"],
                        tokenized["attention_mask"],
                        tokenized["t5_input_lens"],
                    )
                self.conditions[(record_kind, instruction_sha)] = shared_runtime.TextCondition(
                    text_lens=text_lens,
                    text_embs=text_embs.detach(),
                    prompt_sha256=object_sha256(
                        {
                            "record_kind": record_kind,
                            "instruction_utf8_sha256": instruction_sha,
                        }
                    ),
                    instruction_sha256=instruction_sha,
                    task_name=f"full30-psiout-{record_kind}",
                )
        renderer.t5_text_encoder = None
        del tokenizer
        torch.cuda.empty_cache()
        _require(renderer.t5_text_encoder is None, "official provider T5 was not retired")
        self._runtime_consensus(
            object_sha256(
                {
                    "provider_abi": OFFICIAL_PROVIDER_ABI,
                    "runtime_digest": identity["runtime_digest"],
                    "condition_keys": sorted(
                        [list(key) for key in self.conditions],
                        key=lambda item: (item[0].encode("utf-8"), item[1]),
                    ),
                    "normalized_source_bindings": sorted(
                        [
                            {
                                "source_iid": source_iid,
                                **dict(binding),
                            }
                            for source_iid, binding in source_bindings.items()
                        ],
                        key=lambda item: item["source_iid"].encode("utf-8"),
                    ),
                }
            ),
            "official provider initialization",
        )

    def _runtime_consensus(self, value: str, label: str) -> None:  # pragma: no cover - AUH GPU
        gathered: list[Any] = [None] * WORLD_SIZE
        self.dist.all_gather_object(gathered, {"rank": self.rank, "digest": value})
        expected = [{"rank": rank, "digest": value} for rank in range(WORLD_SIZE)]
        _require(gathered == expected, f"{label} differs across WORLD4")

    def verify_noise_authority_v1(
        self, *, record: Mapping[str, Any], noise: Any
    ) -> Mapping[str, Any]:  # pragma: no cover - AUH GPU
        torch = self.torch
        _require(
            isinstance(noise, torch.Tensor)
            and noise.dtype == torch.float32
            and noise.device.type == "cpu"
            and noise.is_contiguous()
            and not noise.requires_grad
            and bool(torch.isfinite(noise).all().item()),
            "official noise artifact differs",
        )
        seed = record["noise"]["seed"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        replayed = torch.randn(
            tuple(int(item) for item in noise.shape),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).contiguous()
        artifact_sha = _tensor_raw_sha256(noise, label="official noise artifact")
        replayed_sha = _tensor_raw_sha256(replayed, label="official replayed noise")
        _require(artifact_sha == replayed_sha and torch.equal(noise, replayed), "official seeded CPU noise replay differs")
        unsigned = {
            "schema_version": NOISE_RECEIPT_SCHEMA_VERSION,
            "provider_abi": OFFICIAL_PROVIDER_ABI,
            "official_provider": True,
            "record_id": record["record_id"],
            "seed": seed,
            "generator": "torch-cpu-generator-manual-seed-randn-fp32-v1",
            "shape": [int(item) for item in noise.shape],
            "artifact_raw_sha256": artifact_sha,
            "replayed_raw_sha256": replayed_sha,
            "byte_exact_replay": True,
        }
        receipt = seal_record(unsigned, "noise_digest")
        self._runtime_consensus(receipt["noise_digest"], "seeded CPU noise replay")
        return MappingProxyType(receipt)

    def prepare_same_state_v1(
        self,
        *,
        record: Mapping[str, Any],
        clean: Any,
        source: Optional[Any],
        noise: Any,
        x_sigma: Any,
        sigma_index: int,
        sigma: float,
        timestep: int,
    ) -> PreparedSameStateV1:  # pragma: no cover - AUH GPU
        torch = self.torch
        for value, label in ((clean, "clean"), (noise, "noise"), (x_sigma, "x_sigma")):
            _require(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.device.type == "cpu"
                and value.is_contiguous()
                and tuple(int(item) for item in value.shape[:3]) == (1, 16, 21)
                and not value.requires_grad
                and bool(torch.isfinite(value).all().item()),
                f"official provider {label} differs",
            )
        _require(clean.shape == noise.shape == x_sigma.shape, "official provider state geometry differs")
        sigma_module = _pinned_sigma_module()
        _require(
            sigma_index in SIGMA_INDICES
            and struct.pack(">f", sigma).hex()
            == sigma_module.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[sigma_index]
            and timestep == sigma_module.PINNED_TIMESTEPS[sigma_index],
            "official provider sigma/timestep differs",
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), self.packed_core.official_frozen_native_only():
            if record["record_kind"] == "teacher_anchor":
                _require(source is None, "source-free teacher received source state")
                patched = self.transformer.patch_vae_latent(
                    x_sigma.to(device=self.device, dtype=self.transformer.dtype),
                    source_id=0,
                )
                branch = self.dclr_runtime_contract.build_t2v_target_branch(
                    patched[0], patched[1], target_source_id=0
                )
                noisy_latents = branch.noisy_latents
                rotary = branch.rotary_embs
                target_tokens = branch.target_token_count
                target_mask = torch.ones(
                    branch.total_token_count, dtype=torch.bool, device=self.device
                )
            else:
                source_iid = str(record["source_iid"])
                _require(
                    isinstance(source, torch.Tensor)
                    and source.dtype == torch.float32
                    and source.device.type == "cpu"
                    and source.is_contiguous()
                    and source.shape == clean.shape
                    and _tensor_raw_sha256(source, label="amplitude source")
                    == _tensor_raw_sha256(clean, label="amplitude target clean"),
                    "amplitude calibrator source and target are not the same real source",
                )
                _require(
                    source_iid in self.normalized_source_by_iid
                    and torch.equal(source, self.normalized_source_by_iid[source_iid]),
                    "amplitude source does not equal recomputed physical posterior index0",
                )
                coordinate = type("Coordinate", (), {"sigma": sigma, "timestep": timestep})()
                packed = self.packed_trainer.prepare_restoration_pair(
                    clean=clean,
                    corrupted_source=source,
                    epsilon=noise,
                    coordinate=coordinate,
                    rope=self.rope,
                    device=self.device,
                )
                target_tokens = int(packed["target_tokens"])
                source_tokens = int(packed["source_tokens"])
                expected_target_patches = self.packed_trainer._pack_latent_patches(
                    x_sigma.squeeze(0).contiguous()
                ).to(self.device)
                _require(
                    torch.equal(
                        packed["input_patches"][source_tokens:].float(),
                        expected_target_patches.float(),
                    ),
                    "packed amplitude target differs from independently constructed x_sigma",
                )
                embedded = self.transformer.patch_embedding(
                    packed["input_patches"]
                ).flatten(1).unsqueeze(0)
                noisy_latents = embedded
                rotary = packed["rotary"].permute(1, 0, 2).unsqueeze(0)
                target_mask = torch.zeros(
                    int(packed["total_tokens"]), dtype=torch.bool, device=self.device
                )
                target_mask[source_tokens:] = True
        timestep_tensor = torch.tensor(
            [float(timestep)], dtype=torch.float32, device=self.device
        )
        input_objects = (noisy_latents, rotary, target_mask, timestep_tensor)
        input_hashes = MappingProxyType(
            {
                "noisy_latents": _tensor_raw_sha256(
                    noisy_latents.detach().float().contiguous(), label="prepared noisy latents"
                ),
                "rotary_embs": hashlib.sha256(
                    rotary.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "target_mask": hashlib.sha256(
                    target_mask.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "timestep": _tensor_raw_sha256(timestep_tensor, label="prepared timestep"),
            }
        )
        state_unsigned = {
            "schema_version": STATE_RECEIPT_SCHEMA_VERSION,
            "provider_abi": OFFICIAL_PROVIDER_ABI,
            "official_provider": True,
            "runtime_digest": self.identity["runtime_digest"],
            "record_id": record["record_id"],
            "record_kind": record["record_kind"],
            "teacher_cell_id": record["teacher_cell_id"],
            "branch": record["branch"],
            "sigma_index": sigma_index,
            "sigma_float32_be_hex": struct.pack(">f", sigma).hex(),
            "timestep": timestep,
            "clean_raw_sha256": _tensor_raw_sha256(clean, label="prepared clean"),
            "source_raw_sha256": (
                None if source is None else _tensor_raw_sha256(source, label="prepared source")
            ),
            "noise_raw_sha256": _tensor_raw_sha256(noise, label="prepared noise"),
            "x_sigma_raw_sha256": _tensor_raw_sha256(x_sigma, label="prepared x_sigma"),
            "input_hashes": dict(input_hashes),
            "target_tokens": target_tokens,
            "spatial_shape": [int(item) for item in clean.shape],
            "same_x_sigma_object_for_all_counterfactuals": True,
            "all_rank_consensus": True,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        state_digest = object_sha256(state_unsigned)
        state_receipt = MappingProxyType(
            {**state_unsigned, "state_digest": state_digest}
        )
        opaque = _OfficialOpaqueStateV1(
            noisy_latents=noisy_latents,
            rotary_embs=rotary,
            target_mask=target_mask,
            timestep=timestep_tensor,
            target_tokens=target_tokens,
            spatial_shape=tuple(int(item) for item in clean.shape),
            input_objects=input_objects,
            input_hashes=input_hashes,
            record_kind=str(record["record_kind"]),
        )
        self._runtime_consensus(state_digest, "same-state preparation")
        return PreparedSameStateV1(
            record_id=str(record["record_id"]),
            sigma_index=sigma_index,
            sigma_float32_be_hex=struct.pack(">f", sigma).hex(),
            timestep=timestep,
            clean_raw_sha256=state_unsigned["clean_raw_sha256"],
            source_raw_sha256=state_unsigned["source_raw_sha256"],
            noise_raw_sha256=state_unsigned["noise_raw_sha256"],
            x_sigma_raw_sha256=state_unsigned["x_sigma_raw_sha256"],
            state_digest=state_digest,
            receipt=state_receipt,
            opaque_state=opaque,
        )

    def forward_post_head_v1(
        self,
        *,
        state: PreparedSameStateV1,
        condition: Mapping[str, Any],
    ) -> FrozenForwardResultV1:  # pragma: no cover - AUH GPU
        _require(isinstance(state, PreparedSameStateV1), "official forward state type differs")
        opaque = state.opaque_state
        _require(isinstance(opaque, _OfficialOpaqueStateV1), "official opaque state differs")
        key = (opaque.record_kind, condition["instruction_utf8_sha256"])
        _require(key in self.conditions, "official encoded condition is absent")
        text_condition = self.conditions[key]
        before_ids = tuple(id(item) for item in opaque.input_objects)
        before_hashes = dict(opaque.input_hashes)
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ), self.packed_core.official_frozen_native_only():
            packed = self.shared_runtime.shared_step_target_prediction(
                self.renderer,
                model_id="transformer_1",
                noisy_latents=opaque.noisy_latents,
                rotary_embs=opaque.rotary_embs,
                target_tokens=opaque.target_tokens,
                target_mask=opaque.target_mask,
                timestep=opaque.timestep,
                condition=text_condition,
            )
            velocity = self.branch_runtime.unpack_post_head_target_velocity_v1(
                packed,
                spatial_shape=opaque.spatial_shape,
            ).detach().float().contiguous()
        _require(
            not self.renderer.training
            and not velocity.requires_grad
            and velocity.grad_fn is None
            and tuple(int(item) for item in velocity.shape) == opaque.spatial_shape
            and bool(self.torch.isfinite(velocity).all().item()),
            "official Frozen post-head velocity differs",
        )
        _require(tuple(id(item) for item in opaque.input_objects) == before_ids, "same-state input objects changed")
        after_hashes = {
            "noisy_latents": _tensor_raw_sha256(
                opaque.noisy_latents.detach().float().contiguous(), label="post-forward noisy latents"
            ),
            "rotary_embs": hashlib.sha256(
                opaque.rotary_embs.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "target_mask": hashlib.sha256(
                opaque.target_mask.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "timestep": _tensor_raw_sha256(opaque.timestep, label="post-forward timestep"),
        }
        _require(after_hashes == before_hashes, "same-state input bytes changed")
        velocity_sha = _tensor_raw_sha256(velocity, label="post-head velocity")
        self._runtime_consensus(velocity_sha, "post-head velocity")
        unsigned = {
            "schema_version": FORWARD_RECEIPT_SCHEMA_VERSION,
            "provider_abi": OFFICIAL_PROVIDER_ABI,
            "official_provider": True,
            "record_id": state.record_id,
            "condition_role": condition["role"],
            "condition_utf8_sha256": condition["instruction_utf8_sha256"],
            "shared_state_digest": state.state_digest,
            "runtime_digest": self.identity["runtime_digest"],
            "sigma_index": state.sigma_index,
            "sigma_float32_be_hex": state.sigma_float32_be_hex,
            "timestep": state.timestep,
            "output_stage": POST_HEAD_STAGE,
            "official_frozen_native_only": True,
            "model_eval": True,
            "torch_inference_mode": True,
            "calibrator_peft_adapter_present": False,
            "frozen_effective_adapter_enabled": False,
            "frozen_effective_typed_patch_role_enabled": False,
            "base_compute_dtype": "torch.bfloat16",
            "autocast_dtype": "torch.bfloat16",
            "observer_output_dtype": "torch.float32",
            "observer_output_detached": True,
            "observer_output_contiguous": True,
            "same_state_input_objects_reused": True,
            "same_state_input_bytes_unchanged": True,
            "all_rank_consensus": True,
            "post_head_velocity_raw_sha256": velocity_sha,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        return FrozenForwardResultV1(
            velocity=velocity,
            receipt=MappingProxyType(seal_record(unsigned, "forward_digest")),
        )

    def close(self) -> None:  # pragma: no cover - AUH GPU
        if self._closed:
            fail("official provider was closed twice")
        self.dist.barrier()
        self.dist.destroy_process_group()
        self._closed = True

    def barrier_v1(self) -> None:  # pragma: no cover - AUH GPU
        _require(not self._closed, "official provider is closed")
        self.dist.barrier()

    def consensus_digest_v1(self, digest: str, label: str) -> None:  # pragma: no cover - AUH GPU
        _sha(digest, f"{label} digest")
        self._runtime_consensus(digest, label)

    def broadcast_rank0_v1(self, value: Any) -> Any:  # pragma: no cover - AUH GPU
        holder = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(holder, src=0)
        return holder[0]


@dataclass(frozen=True)
class MaterializationResultV1:
    output_directory: Path
    run_receipt_path: Path
    run_receipt_sha256: str
    run_digest: str
    record_count: int
    official_provider: bool
    test_only: bool


@dataclass
class _ComputedRecordV1:
    ordinal: int
    record: Mapping[str, Any]
    tensors: dict[str, FP32TensorV1]
    nuisance_tensors: dict[str, FP32TensorV1]
    noise_receipt: Mapping[str, Any]
    state_receipts: list[Mapping[str, Any]]
    forward_receipts: list[Mapping[str, Any]]
    sigma_rows: list[Mapping[str, Any]]
    packets: dict[int, Any]


def _fp32_norm(value: FP32TensorV1) -> float:
    result = math.sqrt(math.fsum(float(item) * float(item) for item in value.values))
    _require(math.isfinite(result), "persisted tensor norm is non-finite")
    return result


def _fp32_cosine(left: FP32TensorV1, right: FP32TensorV1) -> float:
    _require(left.shape == right.shape, "persisted cosine geometry differs")
    denominator = _fp32_norm(left) * _fp32_norm(right)
    _require(denominator > 0.0, "persisted cosine operand is degenerate")
    result = math.fsum(
        float(a) * float(b) for a, b in zip(left.values, right.values)
    ) / denominator
    _require(math.isfinite(result), "persisted cosine is non-finite")
    return max(-1.0, min(1.0, result))


def _fp32_difference_norm(left: FP32TensorV1, right: FP32TensorV1) -> float:
    _require(left.shape == right.shape, "persisted difference geometry differs")
    result = math.sqrt(
        math.fsum((float(a) - float(b)) ** 2 for a, b in zip(left.values, right.values))
    )
    _require(math.isfinite(result), "persisted difference norm is non-finite")
    return result


def _packet_unit_v1(packet: Any, name: str, numerics: PsiOutNumericsV1) -> FP32TensorV1:
    value = getattr(packet, name, None)
    shape = _tensor_shape(value, label=f"nuisance packet {name}")
    if shape == (1, *QUOTIENT_SHAPE):
        value = value[0]
    elif shape != QUOTIENT_SHAPE:
        fail(f"nuisance packet {name} geometry differs")
    result = numerics.to_fp32_tensor(value)
    _require(
        math.isclose(_fp32_norm(result), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6),
        f"nuisance packet {name} is not unit after FP32 persistence",
    )
    return result


def _validate_prepared_state_v1(
    state: Any,
    *,
    provider: FrozenForwardProviderV1,
    record: Mapping[str, Any],
    clean: Any,
    source: Optional[Any],
    noise: Any,
    x_sigma: Any,
    sigma_index: int,
    sigma_hex: str,
    timestep: int,
    runtime_digest: str,
    require_official: bool,
) -> PreparedSameStateV1:
    if not isinstance(state, PreparedSameStateV1):
        fail("provider did not return PreparedSameStateV1")
    receipt = _closed(state.receipt, _STATE_RECEIPT_FIELDS, "same-state receipt")
    _verify_seal(receipt, "state_digest", "same-state receipt")
    _require(receipt["schema_version"] == STATE_RECEIPT_SCHEMA_VERSION, "same-state schema differs")
    _require(receipt["provider_abi"] == OFFICIAL_PROVIDER_ABI, "same-state provider ABI differs")
    _require(receipt["official_provider"] is bool(provider.is_official), "same-state provider identity differs")
    _require(not require_official or provider.is_official, "formal materialization requires the official provider")
    expected = {
        "runtime_digest": runtime_digest,
        "record_id": record["record_id"],
        "record_kind": record["record_kind"],
        "teacher_cell_id": record["teacher_cell_id"],
        "branch": record["branch"],
        "sigma_index": sigma_index,
        "sigma_float32_be_hex": sigma_hex,
        "timestep": timestep,
        "clean_raw_sha256": _tensor_raw_sha256(clean, label="same-state clean"),
        "source_raw_sha256": (
            None if source is None else _tensor_raw_sha256(source, label="same-state source")
        ),
        "noise_raw_sha256": _tensor_raw_sha256(noise, label="same-state noise"),
        "x_sigma_raw_sha256": _tensor_raw_sha256(x_sigma, label="same-state x_sigma"),
        "spatial_shape": list(_tensor_shape(clean, label="same-state clean")),
    }
    for field_name, expected_value in expected.items():
        _require(receipt[field_name] == expected_value, f"same-state receipt {field_name} differs")
    input_hashes = _closed(
        receipt["input_hashes"],
        {"noisy_latents", "rotary_embs", "target_mask", "timestep"},
        "same-state input hashes",
    )
    for field_name, value in input_hashes.items():
        _sha(value, f"same-state input hash {field_name}")
    _require(type(receipt["target_tokens"]) is int and receipt["target_tokens"] > 0, "same-state target token count differs")
    for field_name in (
        "same_x_sigma_object_for_all_counterfactuals",
        "all_rank_consensus",
    ):
        _require(receipt[field_name] is True, f"same-state receipt {field_name} is not true")
    _require(receipt["model_parameters_updated"] is False, "same-state updated model parameters")
    _require(receipt["optimizer_created"] is False, "same-state created an optimizer")
    _require(
        state.record_id == record["record_id"]
        and state.sigma_index == sigma_index
        and state.sigma_float32_be_hex == sigma_hex
        and state.timestep == timestep
        and state.clean_raw_sha256 == expected["clean_raw_sha256"]
        and state.source_raw_sha256 == expected["source_raw_sha256"]
        and state.noise_raw_sha256 == expected["noise_raw_sha256"]
        and state.x_sigma_raw_sha256 == expected["x_sigma_raw_sha256"]
        and state.state_digest == receipt["state_digest"],
        "PreparedSameStateV1 fields differ from its sealed receipt",
    )
    return state


def _run_frozen_forward_v1(
    provider: FrozenForwardProviderV1,
    *,
    state: PreparedSameStateV1,
    condition: Mapping[str, Any],
    record: Mapping[str, Any],
    spatial_shape: Sequence[int],
    runtime_digest: str,
    require_official: bool,
) -> FrozenForwardResultV1:
    try:
        result = provider.forward_post_head_v1(state=state, condition=condition)
    except Full30PsiOutMaterializationError:
        raise
    except Exception as error:
        raise Full30PsiOutMaterializationError(
            f"Frozen forward failed for {record['record_id']}/{condition['role']}: {error}"
        ) from error
    if not isinstance(result, FrozenForwardResultV1):
        fail("provider did not return FrozenForwardResultV1")
    _require(
        _tensor_shape(result.velocity, label="post-head velocity") == tuple(spatial_shape),
        "post-head velocity geometry differs",
    )
    velocity_sha = _tensor_raw_sha256(result.velocity, label="post-head velocity")
    receipt = _closed(result.receipt, _FORWARD_RECEIPT_FIELDS, "Frozen forward receipt")
    _verify_seal(receipt, "forward_digest", "Frozen forward receipt")
    expected = {
        "schema_version": FORWARD_RECEIPT_SCHEMA_VERSION,
        "provider_abi": OFFICIAL_PROVIDER_ABI,
        "official_provider": bool(provider.is_official),
        "record_id": record["record_id"],
        "condition_role": condition["role"],
        "condition_utf8_sha256": condition["instruction_utf8_sha256"],
        "shared_state_digest": state.state_digest,
        "runtime_digest": runtime_digest,
        "sigma_index": state.sigma_index,
        "sigma_float32_be_hex": state.sigma_float32_be_hex,
        "timestep": state.timestep,
        "output_stage": POST_HEAD_STAGE,
        "official_frozen_native_only": True,
        "model_eval": True,
        "torch_inference_mode": True,
        "calibrator_peft_adapter_present": False,
        "frozen_effective_adapter_enabled": False,
        "frozen_effective_typed_patch_role_enabled": False,
        "base_compute_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "observer_output_dtype": "torch.float32",
        "observer_output_detached": True,
        "observer_output_contiguous": True,
        "same_state_input_objects_reused": True,
        "same_state_input_bytes_unchanged": True,
        "all_rank_consensus": True,
        "post_head_velocity_raw_sha256": velocity_sha,
        "model_parameters_updated": False,
        "optimizer_created": False,
    }
    for field_name, expected_value in expected.items():
        _require(receipt[field_name] == expected_value, f"Frozen forward receipt {field_name} differs")
    _require(not require_official or receipt["official_provider"] is True, "formal forward is not official")
    return result


def _create_only_file_v1(path: Path, raw: bytes, *, mode: int) -> str:
    _require(path.is_absolute(), "output file path must be absolute")
    _require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent is not a plain directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise Full30PsiOutMaterializationError(f"cannot create output file: {path}") from error
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            _require(written > 0, f"output write stalled: {path}")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == mode
        and metadata.st_size == len(raw),
        f"output file mode/size differs: {path}",
    )
    return hashlib.sha256(raw).hexdigest()


def _write_json_create_only_v1(path: Path, value: Mapping[str, Any]) -> str:
    return _create_only_file_v1(path, canonical_json_bytes(value) + b"\n", mode=0o600)


def _teacher_container_bytes_v1(
    record: Mapping[str, Any],
    *,
    container_kind: str,
    tensors: Mapping[str, FP32TensorV1],
) -> tuple[bytes, Mapping[str, str]]:
    expected_names = teacher_authority._expected_tensor_names(
        container_kind, str(record["evidence_role"])
    )
    _require(tuple(tensors) == expected_names, f"{container_kind} tensor name/order differs")
    payload_parts: list[bytes] = []
    entries: list[Mapping[str, Any]] = []
    slice_digests: dict[str, str] = {}
    for ordinal, name in enumerate(expected_names):
        tensor = tensors[name]
        _require(tensor.shape == QUOTIENT_SHAPE, f"{name} shape differs")
        raw = tensor.bytes_le()
        digest = hashlib.sha256(raw).hexdigest()
        entries.append(
            {
                "name": name,
                "dtype": teacher_authority.TENSOR_DTYPE,
                "shape": list(teacher_authority.TENSOR_SHAPE),
                "offset": ordinal * teacher_authority.TENSOR_SLICE_BYTES,
                "length": teacher_authority.TENSOR_SLICE_BYTES,
                "sha256": digest,
            }
        )
        payload_parts.append(raw)
        slice_digests[name] = digest
    payload = b"".join(payload_parts)
    header = {
        "schema_version": teacher_authority.TENSOR_CONTAINER_SCHEMA,
        "container_kind": container_kind,
        "evidence_id": record["evidence_id"],
        "evidence_role": record["evidence_role"],
        "teacher_cell_id": record["teacher_cell_id"],
        "branch": record["branch"],
        "dtype": teacher_authority.TENSOR_DTYPE,
        "shape": list(teacher_authority.TENSOR_SHAPE),
        "sigma_indices": list(teacher_authority.SIGMA_INDICES),
        "layout": teacher_authority.TENSOR_CONTAINER_LAYOUT,
        "tensor_count": len(expected_names),
        "payload_bytes": len(payload),
        "entries": entries,
    }
    header_bytes = canonical_json_bytes(header)
    _require(len(header_bytes) <= teacher_authority.TENSOR_CONTAINER_MAX_HEADER_BYTES, "teacher container header is oversized")
    raw = teacher_authority.TENSOR_CONTAINER_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + payload
    _require(len(raw) <= teacher_authority.TENSOR_CONTAINER_MAX_FILE_BYTES, "teacher container is oversized")
    return raw, MappingProxyType(slice_digests)


def _amplitude_container_bytes_v1(
    record: Mapping[str, Any], tensors: Mapping[str, FP32TensorV1]
) -> tuple[bytes, Mapping[str, str]]:
    expected_names = tuple(f"sigma_{sigma_index:02d}:projected_raw" for sigma_index in SIGMA_INDICES)
    _require(tuple(tensors) == expected_names, "amplitude tensor name/order differs")
    payload_parts: list[bytes] = []
    entries: list[Mapping[str, Any]] = []
    digests: dict[str, str] = {}
    for ordinal, (sigma_index, name) in enumerate(zip(SIGMA_INDICES, expected_names)):
        tensor = tensors[name]
        _require(tensor.shape == QUOTIENT_SHAPE, f"{name} shape differs")
        raw = tensor.bytes_le()
        digest = hashlib.sha256(raw).hexdigest()
        entries.append(
            {
                "name": name,
                "sigma_index": sigma_index,
                "dtype": amplitude_authority.TENSOR_DTYPE,
                "shape": list(amplitude_authority.TENSOR_SHAPE),
                "offset": ordinal * amplitude_authority.TENSOR_SLICE_BYTES,
                "length": amplitude_authority.TENSOR_SLICE_BYTES,
                "sha256": digest,
            }
        )
        payload_parts.append(raw)
        digests[name] = digest
    payload = b"".join(payload_parts)
    header = {
        "schema_version": amplitude_authority.CONTAINER_SCHEMA_VERSION,
        "evidence_id": record["evidence_id"],
        "pair_id": record["pair_id"],
        "source_iid": record["source_iid"],
        "teacher_cell_id": record["teacher_cell_id"],
        "branch": record["branch"],
        "dtype": amplitude_authority.TENSOR_DTYPE,
        "shape": list(amplitude_authority.TENSOR_SHAPE),
        "sigma_indices": list(amplitude_authority.SIGMA_INDICES),
        "layout": amplitude_authority.CONTAINER_LAYOUT,
        "tensor_count": len(SIGMA_INDICES),
        "payload_bytes": len(payload),
        "entries": entries,
    }
    header_bytes = canonical_json_bytes(header)
    _require(len(header_bytes) <= amplitude_authority.CONTAINER_MAX_HEADER_BYTES, "amplitude container header is oversized")
    raw = amplitude_authority.CONTAINER_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + payload
    _require(len(raw) <= amplitude_authority.CONTAINER_MAX_FILE_BYTES, "amplitude container is oversized")
    return raw, MappingProxyType(digests)


def _assert_working_tensors_unchanged_v1(
    *,
    clean: Any,
    source: Optional[Any],
    noise: Any,
    x_sigma: Any,
    expected: Mapping[str, Optional[str]],
) -> None:
    observed = {
        "clean": _tensor_raw_sha256(clean, label="post-forward clean"),
        "source": (
            None if source is None else _tensor_raw_sha256(source, label="post-forward source")
        ),
        "noise": _tensor_raw_sha256(noise, label="post-forward noise"),
        "x_sigma": _tensor_raw_sha256(x_sigma, label="post-forward x_sigma"),
    }
    _require(observed == dict(expected), "clean/source/noise/x_sigma bytes changed during Frozen observation")


def _verify_noise_replay_v1(
    provider: FrozenForwardProviderV1,
    *,
    record: Mapping[str, Any],
    noise: Any,
    require_official: bool,
) -> Mapping[str, Any]:
    try:
        receipt = provider.verify_noise_authority_v1(record=record, noise=noise)
    except Full30PsiOutMaterializationError:
        raise
    except Exception as error:
        raise Full30PsiOutMaterializationError(
            f"noise replay failed for {record['record_id']}: {error}"
        ) from error
    receipt = _closed(receipt, _NOISE_RECEIPT_FIELDS, "noise replay receipt")
    _verify_seal(receipt, "noise_digest", "noise replay receipt")
    expected_sha = _tensor_raw_sha256(noise, label="noise replay input")
    expected = {
        "schema_version": NOISE_RECEIPT_SCHEMA_VERSION,
        "provider_abi": OFFICIAL_PROVIDER_ABI,
        "official_provider": bool(provider.is_official),
        "record_id": record["record_id"],
        "seed": record["noise"]["seed"],
        "generator": record["noise"]["generator"],
        "shape": list(_tensor_shape(noise, label="noise replay input")),
        "artifact_raw_sha256": expected_sha,
        "replayed_raw_sha256": expected_sha,
        "byte_exact_replay": True,
    }
    for field_name, expected_value in expected.items():
        _require(receipt[field_name] == expected_value, f"noise replay receipt {field_name} differs")
    _require(not require_official or receipt["official_provider"] is True, "formal noise replay is not official")
    return MappingProxyType(dict(receipt))


def _prepare_state_v1(
    provider: FrozenForwardProviderV1,
    *,
    record: Mapping[str, Any],
    clean: Any,
    source: Optional[Any],
    noise: Any,
    x_sigma: Any,
    sigma_index: int,
    sigma: float,
    timestep: int,
    runtime_digest: str,
    require_official: bool,
) -> PreparedSameStateV1:
    sigma_hex = struct.pack(">f", float(sigma)).hex()
    try:
        state = provider.prepare_same_state_v1(
            record=record,
            clean=clean,
            source=source,
            noise=noise,
            x_sigma=x_sigma,
            sigma_index=sigma_index,
            sigma=sigma,
            timestep=timestep,
        )
    except Full30PsiOutMaterializationError:
        raise
    except Exception as error:
        raise Full30PsiOutMaterializationError(
            f"same-state preparation failed for {record['record_id']}/sigma {sigma_index}: {error}"
        ) from error
    return _validate_prepared_state_v1(
        state,
        provider=provider,
        record=record,
        clean=clean,
        source=source,
        noise=noise,
        x_sigma=x_sigma,
        sigma_index=sigma_index,
        sigma_hex=sigma_hex,
        timestep=timestep,
        runtime_digest=runtime_digest,
        require_official=require_official,
    )


def _condition_by_role_v1(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    result = {str(row["role"]): row for row in record["conditions"]}
    _require(len(result) == len(record["conditions"]), "condition role is duplicated")
    return MappingProxyType(result)


def _compute_teacher_record_v1(
    *,
    ordinal: int,
    record: Mapping[str, Any],
    provider: FrozenForwardProviderV1,
    numerics: PsiOutNumericsV1,
    runtime_digest: str,
    require_official: bool,
) -> _ComputedRecordV1:
    clean_artifact = load_fp32_artifact_v1(
        record["target_clean_latent"], label=f"{record['record_id']}.clean"
    )
    noise_artifact = load_fp32_artifact_v1(
        record["noise"]["artifact"], label=f"{record['record_id']}.noise"
    )
    clean = numerics.from_artifact(clean_artifact)
    noise = numerics.from_artifact(noise_artifact)
    noise_receipt = _verify_noise_replay_v1(
        provider,
        record=record,
        noise=noise,
        require_official=require_official,
    )
    _require(_tensor_shape(clean, label="teacher clean") == _tensor_shape(noise, label="teacher noise"), "teacher clean/noise geometry differs")
    conditions = _condition_by_role_v1(record)
    tensor_values: dict[str, FP32TensorV1] = {}
    nuisance_values: dict[str, FP32TensorV1] = {}
    state_receipts: list[Mapping[str, Any]] = []
    forward_receipts: list[Mapping[str, Any]] = []
    sigma_rows: list[Mapping[str, Any]] = []
    packets: dict[int, Any] = {}
    sigma_module = _pinned_sigma_module()
    runtime_shape = _tensor_shape(clean, label="teacher clean")

    for sigma_index in SIGMA_INDICES:
        sigma = float(sigma_module.PINNED_POSITIVE_SIGMAS[sigma_index])
        timestep = int(sigma_module.PINNED_TIMESTEPS[sigma_index])
        x_sigma = numerics.mix(clean, noise, sigma)
        working_hashes: Mapping[str, Optional[str]] = MappingProxyType(
            {
                "clean": _tensor_raw_sha256(clean, label="teacher clean"),
                "source": None,
                "noise": _tensor_raw_sha256(noise, label="teacher noise"),
                "x_sigma": _tensor_raw_sha256(x_sigma, label="teacher x_sigma"),
            }
        )
        state = _prepare_state_v1(
            provider,
            record=record,
            clean=clean,
            source=None,
            noise=noise,
            x_sigma=x_sigma,
            sigma_index=sigma_index,
            sigma=sigma,
            timestep=timestep,
            runtime_digest=runtime_digest,
            require_official=require_official,
        )
        state_receipts.append(dict(state.receipt))
        call_roles = ["branch", "branch", "noop", "noop", "camera_only", "appearance_only"]
        if record["evidence_role"] == "teacher_origin":
            call_roles.extend(teacher_authority.WRONG_CONTROL_TYPES)
        results: list[FrozenForwardResultV1] = []
        for role in call_roles:
            result = _run_frozen_forward_v1(
                provider,
                state=state,
                condition=conditions[role],
                record=record,
                spatial_shape=runtime_shape,
                runtime_digest=runtime_digest,
                require_official=require_official,
            )
            results.append(result)
            forward_receipts.append(dict(result.receipt))
        _assert_working_tensors_unchanged_v1(
            clean=clean,
            source=None,
            noise=noise,
            x_sigma=x_sigma,
            expected=working_hashes,
        )
        branch_first, branch_second, noop_first, noop_second, camera, appearance = (
            item.velocity for item in results[:6]
        )
        _require(
            results[0].receipt["post_head_velocity_raw_sha256"]
            == results[1].receipt["post_head_velocity_raw_sha256"],
            "duplicate branch post-head forwards are not byte deterministic",
        )
        _require(
            results[2].receipt["post_head_velocity_raw_sha256"]
            == results[3].receipt["post_head_velocity_raw_sha256"],
            "duplicate noop post-head forwards are not byte deterministic",
        )
        branch_raw_first = numerics.psiout_raw(numerics.delta(branch_first, noop_first))
        branch_raw_second = numerics.psiout_raw(numerics.delta(branch_second, noop_second))
        duplicate_first = numerics.to_fp32_tensor(branch_raw_first)
        duplicate_second = numerics.to_fp32_tensor(branch_raw_second)
        _require(
            duplicate_first.bytes_le() == duplicate_second.bytes_le(),
            "duplicate branch-minus-noop PsiOut bytes differ",
        )
        noop_code_first = numerics.to_fp32_tensor(numerics.psiout_raw(noop_first))
        noop_code_second = numerics.to_fp32_tensor(numerics.psiout_raw(noop_second))
        _require(
            noop_code_first.bytes_le() == noop_code_second.bytes_le(),
            "same-state noop-minus-noop null is not byte deterministic",
        )
        camera_raw = numerics.psiout_raw(numerics.delta(camera, noop_first))
        appearance_raw = numerics.psiout_raw(numerics.delta(appearance, noop_first))
        packet = numerics.nuisance(camera_raw, appearance_raw)
        packets[sigma_index] = packet
        camera_unit = _packet_unit_v1(packet, "camera_unit", numerics)
        appearance_unit = _packet_unit_v1(packet, "appearance_unit", numerics)
        projected_raw_backend = numerics.project(branch_raw_first, packet)
        projected_unit_backend = numerics.unit(projected_raw_backend)
        projected_raw = numerics.to_fp32_tensor(projected_raw_backend)
        projected_unit = numerics.to_fp32_tensor(projected_unit_backend)
        _require(
            math.isclose(_fp32_norm(projected_unit), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6),
            "persisted projected teacher is not unit",
        )

        prefix = f"sigma_{sigma_index:02d}:"
        if record["evidence_role"] == "teacher_origin":
            tensor_values[prefix + "projected_unit"] = projected_unit
            tensor_values[prefix + "projected_raw"] = projected_raw
            tensor_values[prefix + "duplicate_forward_first"] = duplicate_first
            tensor_values[prefix + "duplicate_forward_second"] = duplicate_second
            tensor_values[prefix + "noop_forward_first"] = noop_code_first
            tensor_values[prefix + "noop_forward_second"] = noop_code_second
            wrong_rows: list[Mapping[str, Any]] = []
            for offset, control_type in enumerate(teacher_authority.WRONG_CONTROL_TYPES, start=6):
                wrong_raw = numerics.psiout_raw(
                    numerics.delta(results[offset].velocity, noop_first)
                )
                wrong_projected = numerics.project(wrong_raw, packet)
                wrong_unit = numerics.to_fp32_tensor(numerics.unit(wrong_projected))
                tensor_values[prefix + f"{control_type}_projected_unit"] = wrong_unit
                wrong_rows.append(
                    _closed(
                        {
                        "control_type": control_type,
                        "control_anchor_id": conditions[control_type]["control_anchor_id"],
                        "wrong_projected_slice_sha256": wrong_unit.raw_sha256(),
                        "wrong_event_cosine": _fp32_cosine(projected_unit, wrong_unit),
                        },
                        _WRONG_CONTROL_METRIC_FIELDS,
                        "teacher wrong-control metric",
                    )
                )
            null_norm = _fp32_difference_norm(noop_code_first, noop_code_second)
            raw_norm = _fp32_norm(projected_raw)
            sigma_rows.append(
                _closed(
                    {
                    "sigma_index": sigma_index,
                    "state_digest": state.state_digest,
                    "projected_unit_sha256": projected_unit.raw_sha256(),
                    "projected_raw_sha256": projected_raw.raw_sha256(),
                    "duplicate_forward_first_sha256": duplicate_first.raw_sha256(),
                    "duplicate_forward_second_sha256": duplicate_second.raw_sha256(),
                    "duplicate_forward_bytes_identical": True,
                    "noop_forward_first_sha256": noop_code_first.raw_sha256(),
                    "noop_forward_second_sha256": noop_code_second.raw_sha256(),
                    "same_state_noop_minus_noop_null_norm": null_norm,
                    "projected_teacher_raw_norm": raw_norm,
                    "signal_to_null_snr": raw_norm / max(
                        null_norm, teacher_authority.DUPLICATE_SNR_DENOMINATOR_FLOOR
                    ),
                    "camera_unit_sha256": camera_unit.raw_sha256(),
                    "appearance_unit_sha256": appearance_unit.raw_sha256(),
                    "camera_residual_cosine": _fp32_cosine(projected_unit, camera_unit),
                    "appearance_residual_cosine": _fp32_cosine(projected_unit, appearance_unit),
                    "wrong_controls": wrong_rows,
                    },
                    _TEACHER_ORIGIN_SIGMA_METRIC_FIELDS,
                    "teacher origin sigma metric",
                )
            )
        else:
            tensor_values[prefix + "projected_unit"] = projected_unit
            sigma_rows.append(
                _closed(
                    {
                    "sigma_index": sigma_index,
                    "state_digest": state.state_digest,
                    "projected_unit_sha256": projected_unit.raw_sha256(),
                    "camera_unit_sha256": camera_unit.raw_sha256(),
                    "appearance_unit_sha256": appearance_unit.raw_sha256(),
                    },
                    _TEACHER_CROSS_SIGMA_METRIC_FIELDS,
                    "teacher cross-anchor sigma metric",
                )
            )
        nuisance_values[prefix + "camera_unit"] = camera_unit
        nuisance_values[prefix + "appearance_unit"] = appearance_unit
        del results, state, x_sigma

    return _ComputedRecordV1(
        ordinal=ordinal,
        record=record,
        tensors=tensor_values,
        nuisance_tensors=nuisance_values,
        noise_receipt=noise_receipt,
        state_receipts=state_receipts,
        forward_receipts=forward_receipts,
        sigma_rows=sigma_rows,
        packets=packets,
    )


def _compute_amplitude_record_v1(
    *,
    ordinal: int,
    record: Mapping[str, Any],
    provider: FrozenForwardProviderV1,
    numerics: PsiOutNumericsV1,
    runtime_digest: str,
    require_official: bool,
    nuisance_packets: Mapping[int, Any],
) -> _ComputedRecordV1:
    clean_artifact = load_fp32_artifact_v1(
        record["target_clean_latent"], label=f"{record['record_id']}.target_clean"
    )
    source_artifact = load_fp32_artifact_v1(
        record["source_clean_latent"], label=f"{record['record_id']}.source_clean"
    )
    noise_artifact = load_fp32_artifact_v1(
        record["noise"]["artifact"], label=f"{record['record_id']}.noise"
    )
    _require(
        clean_artifact.bytes_le() == source_artifact.bytes_le(),
        "same-mode amplitude target/source clean bytes differ",
    )
    clean = numerics.from_artifact(clean_artifact)
    source = numerics.from_artifact(source_artifact)
    noise = numerics.from_artifact(noise_artifact)
    noise_receipt = _verify_noise_replay_v1(
        provider,
        record=record,
        noise=noise,
        require_official=require_official,
    )
    conditions = _condition_by_role_v1(record)
    tensor_values: dict[str, FP32TensorV1] = {}
    state_receipts: list[Mapping[str, Any]] = []
    forward_receipts: list[Mapping[str, Any]] = []
    sigma_rows: list[Mapping[str, Any]] = []
    sigma_module = _pinned_sigma_module()
    runtime_shape = _tensor_shape(clean, label="amplitude clean")
    _require(runtime_shape == _tensor_shape(source, label="amplitude source") == _tensor_shape(noise, label="amplitude noise"), "amplitude latent geometry differs")
    _require(set(nuisance_packets) == set(SIGMA_INDICES), "amplitude teacher nuisance packet closure differs")

    for sigma_index in SIGMA_INDICES:
        sigma = float(sigma_module.PINNED_POSITIVE_SIGMAS[sigma_index])
        timestep = int(sigma_module.PINNED_TIMESTEPS[sigma_index])
        x_sigma = numerics.mix(clean, noise, sigma)
        working_hashes: Mapping[str, Optional[str]] = MappingProxyType(
            {
                "clean": _tensor_raw_sha256(clean, label="amplitude clean"),
                "source": _tensor_raw_sha256(source, label="amplitude source"),
                "noise": _tensor_raw_sha256(noise, label="amplitude noise"),
                "x_sigma": _tensor_raw_sha256(x_sigma, label="amplitude x_sigma"),
            }
        )
        state = _prepare_state_v1(
            provider,
            record=record,
            clean=clean,
            source=source,
            noise=noise,
            x_sigma=x_sigma,
            sigma_index=sigma_index,
            sigma=sigma,
            timestep=timestep,
            runtime_digest=runtime_digest,
            require_official=require_official,
        )
        state_receipts.append(dict(state.receipt))
        branch = _run_frozen_forward_v1(
            provider,
            state=state,
            condition=conditions["branch"],
            record=record,
            spatial_shape=runtime_shape,
            runtime_digest=runtime_digest,
            require_official=require_official,
        )
        noop = _run_frozen_forward_v1(
            provider,
            state=state,
            condition=conditions["noop"],
            record=record,
            spatial_shape=runtime_shape,
            runtime_digest=runtime_digest,
            require_official=require_official,
        )
        forward_receipts.extend((dict(branch.receipt), dict(noop.receipt)))
        _assert_working_tensors_unchanged_v1(
            clean=clean,
            source=source,
            noise=noise,
            x_sigma=x_sigma,
            expected=working_hashes,
        )
        raw = numerics.psiout_raw(numerics.delta(branch.velocity, noop.velocity))
        projected = numerics.to_fp32_tensor(
            numerics.project(raw, nuisance_packets[sigma_index])
        )
        name = f"sigma_{sigma_index:02d}:projected_raw"
        tensor_values[name] = projected
        sigma_rows.append(
            _closed(
                {
                "sigma_index": sigma_index,
                "state_digest": state.state_digest,
                "projected_slice_sha256": projected.raw_sha256(),
                "amplitude_norm": _fp32_norm(projected),
                "teacher_nuisance_camera_sha256": _packet_unit_v1(
                    nuisance_packets[sigma_index], "camera_unit", numerics
                ).raw_sha256(),
                "teacher_nuisance_appearance_sha256": _packet_unit_v1(
                    nuisance_packets[sigma_index], "appearance_unit", numerics
                ).raw_sha256(),
                },
                _AMPLITUDE_SIGMA_METRIC_FIELDS,
                "amplitude sigma metric",
            )
        )
        del branch, noop, state, x_sigma
    return _ComputedRecordV1(
        ordinal=ordinal,
        record=record,
        tensors=tensor_values,
        nuisance_tensors={},
        noise_receipt=noise_receipt,
        state_receipts=state_receipts,
        forward_receipts=forward_receipts,
        sigma_rows=sigma_rows,
        packets={},
    )


def _fresh_output_directory_v1(path_value: str | Path) -> Path:
    path = Path(path_value)
    _require(path.is_absolute(), "output directory must be absolute")
    parent = path.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise Full30PsiOutMaterializationError("output parent is unavailable") from error
    _require(
        resolved_parent == parent and parent.is_dir() and not parent.is_symlink(),
        "output parent must be one canonical plain existing directory",
    )
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise Full30PsiOutMaterializationError(f"cannot inspect output path: {path}") from error
    else:
        fail(f"create-only output directory already exists: {path}")
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700, follow_symlinks=False)
    except OSError as error:
        raise Full30PsiOutMaterializationError(f"cannot create output directory: {path}") from error
    metadata = path.lstat()
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "output directory mode/type differs",
    )
    return path.resolve(strict=True)


def _record_directory_name_v1(computed: _ComputedRecordV1) -> str:
    digest = hashlib.sha256(str(computed.record["record_id"]).encode("utf-8")).hexdigest()
    return f"record-{computed.ordinal:04d}-{digest[:16]}"


def _load_bound_review_v1(record: Mapping[str, Any]) -> Mapping[str, Any]:
    review, _, _ = _load_json_file(
        record["review"]["path"],
        record["review"]["file_sha256"],
        f"{record['record_id']}.review",
    )
    _require(review["review_digest"] == record["review"]["review_digest"], "bound review digest differs at write")
    return review


def _candidate_teacher_evidence_v1(
    record: Mapping[str, Any],
    *,
    review: Mapping[str, Any],
    psiout_path: Path,
    psiout_sha256: str,
    nuisance_path: Path,
    nuisance_sha256: str,
) -> Mapping[str, Any]:
    unsigned = {
        "schema_version": teacher_authority.REPRESENTATION_EVIDENCE_SCHEMA,
        "evidence_id": record["evidence_id"],
        "evidence_role": record["evidence_role"],
        "teacher_cell_id": record["teacher_cell_id"],
        "anchor_id": record["anchor_id"],
        "anchor_iid": record["anchor_iid"],
        "anchor_split": record["analysis_split"],
        "branch": record["branch"],
        "event_id": record["event_id"],
        "actor_kind": record["actor_kind"],
        "q0_id": record["q0_id"],
        "actor_id": record["actor_id"],
        "scene_id": record["scene_id"],
        "anchor_video_path": record["reviewed_media"]["path"],
        "anchor_video_sha256": record["reviewed_media"]["file_sha256"],
        "psiout_sidecar_path": str(psiout_path),
        "psiout_sidecar_sha256": psiout_sha256,
        "nuisance_packet_path": str(nuisance_path),
        "nuisance_packet_sha256": nuisance_sha256,
        "all_tensor_values_finite": True,
        "pre_admission_blind_review": dict(review),
    }
    value = seal_record(unsigned, "evidence_digest")
    _closed(value, teacher_authority._ANCHOR_EVIDENCE_FIELDS, "candidate teacher evidence")
    return MappingProxyType(value)


def _candidate_amplitude_evidence_v1(
    record: Mapping[str, Any],
    *,
    review: Mapping[str, Any],
    container_path: Path,
    container_sha256: str,
    official_provider: bool,
) -> Mapping[str, Any]:
    branch_condition = next(
        condition for condition in record["conditions"] if condition["role"] == "branch"
    )
    unsigned = {
        "schema_version": amplitude_authority.EVIDENCE_SCHEMA_VERSION,
        "evidence_id": record["evidence_id"],
        "evidence_role": "calibrator",
        "teacher_cell_id": record["teacher_cell_id"],
        "branch": record["branch"],
        "pair_id": record["pair_id"],
        "source_iid": record["source_iid"],
        "source_posterior_index0_sha256": record["source_posterior_index0_sha256"],
        "instruction_utf8_sha256": branch_condition["instruction_utf8_sha256"],
        "baseline_output_path": record["reviewed_media"]["path"],
        "baseline_output_sha256": record["reviewed_media"]["file_sha256"],
        "initial_gaussian_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],
        "same_source_noise_sigma_state": True,
        "official_frozen_native_only": official_provider,
        "pre_admission_review": dict(review),
        "calibrator_noise_seed": record["noise"]["seed"],
        "calibrator_noise_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],
        "amplitude_container_path": str(container_path),
        "amplitude_container_sha256": container_sha256,
    }
    value = seal_record(unsigned, "evidence_digest")
    _closed(
        value,
        amplitude_authority._COMMON_EVIDENCE_FIELDS
        | amplitude_authority._CALIBRATOR_EXTRA_FIELDS,
        "candidate amplitude evidence",
    )
    return MappingProxyType(value)


def _write_computed_records_v1(
    output_directory: Path,
    computed_records: Sequence[_ComputedRecordV1],
    *,
    plan: Mapping[str, Any],
    official_provider: bool,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    receipt_refs: list[Mapping[str, Any]] = []
    evidence_by_record: dict[str, Mapping[str, Any]] = {}
    sigma_authority_digest = sigma_authority_receipt_v1()[
        "sigma_authority_digest"
    ]
    for computed in sorted(computed_records, key=lambda item: item.ordinal):
        record = computed.record
        record_directory = output_directory / _record_directory_name_v1(computed)
        os.mkdir(record_directory, 0o700)
        os.chmod(record_directory, 0o700, follow_symlinks=False)
        container_bindings: list[Mapping[str, Any]] = []
        if record["record_kind"] == "teacher_anchor":
            psiout_raw, psiout_slices = _teacher_container_bytes_v1(
                record, container_kind="psiout", tensors=computed.tensors
            )
            nuisance_raw, nuisance_slices = _teacher_container_bytes_v1(
                record, container_kind="nuisance", tensors=computed.nuisance_tensors
            )
            psiout_path = record_directory / "psiout.tensor-container"
            nuisance_path = record_directory / "nuisance.tensor-container"
            psiout_sha = _create_only_file_v1(
                psiout_path, psiout_raw, mode=teacher_authority.TENSOR_CONTAINER_MODE
            )
            nuisance_sha = _create_only_file_v1(
                nuisance_path,
                nuisance_raw,
                mode=teacher_authority.TENSOR_CONTAINER_MODE,
            )
            parsed_psiout = teacher_authority._validate_tensor_container(
                str(psiout_path),
                psiout_sha,
                container_kind="psiout",
                evidence_id=str(record["evidence_id"]),
                evidence_role=str(record["evidence_role"]),
                teacher_cell_id=str(record["teacher_cell_id"]),
                branch=str(record["branch"]),
                label=f"{record['record_id']}.written_psiout",
            )
            parsed_nuisance = teacher_authority._validate_tensor_container(
                str(nuisance_path),
                nuisance_sha,
                container_kind="nuisance",
                evidence_id=str(record["evidence_id"]),
                evidence_role=str(record["evidence_role"]),
                teacher_cell_id=str(record["teacher_cell_id"]),
                branch=str(record["branch"]),
                label=f"{record['record_id']}.written_nuisance",
            )
            _require(
                {name: row[2] for name, row in parsed_psiout.items()}
                == dict(psiout_slices),
                "written PsiOut slice digests differ",
            )
            _require(
                {name: row[2] for name, row in parsed_nuisance.items()}
                == dict(nuisance_slices),
                "written nuisance slice digests differ",
            )
            container_bindings.extend(
                (
                    _closed(
                        {
                        "container_kind": "psiout",
                        "path": str(psiout_path),
                        "file_sha256": psiout_sha,
                        "slice_sha256": dict(psiout_slices),
                        },
                        _CONTAINER_BINDING_FIELDS,
                        "PsiOut container binding",
                    ),
                    _closed(
                        {
                        "container_kind": "nuisance",
                        "path": str(nuisance_path),
                        "file_sha256": nuisance_sha,
                        "slice_sha256": dict(nuisance_slices),
                        },
                        _CONTAINER_BINDING_FIELDS,
                        "nuisance container binding",
                    ),
                )
            )
            evidence = _candidate_teacher_evidence_v1(
                record,
                review=_load_bound_review_v1(record),
                psiout_path=psiout_path,
                psiout_sha256=psiout_sha,
                nuisance_path=nuisance_path,
                nuisance_sha256=nuisance_sha,
            )
        else:
            amplitude_raw, amplitude_slices = _amplitude_container_bytes_v1(
                record, computed.tensors
            )
            amplitude_path = record_directory / "amplitude.tensor-container"
            amplitude_sha = _create_only_file_v1(
                amplitude_path,
                amplitude_raw,
                mode=amplitude_authority.CONTAINER_MODE,
            )
            parsed = amplitude_authority._validate_container(
                str(amplitude_path),
                amplitude_sha,
                evidence_id=str(record["evidence_id"]),
                pair_id=str(record["pair_id"]),
                source_iid=str(record["source_iid"]),
                teacher_cell_id=str(record["teacher_cell_id"]),
                branch=str(record["branch"]),
                label=f"{record['record_id']}.written_amplitude",
            )
            _require(
                {
                    f"sigma_{sigma_index:02d}:projected_raw": row.sha256
                    for sigma_index, row in parsed.items()
                }
                == dict(amplitude_slices),
                "written amplitude slice digests differ",
            )
            container_bindings.append(
                _closed(
                    {
                    "container_kind": "amplitude",
                    "path": str(amplitude_path),
                    "file_sha256": amplitude_sha,
                    "slice_sha256": dict(amplitude_slices),
                    },
                    _CONTAINER_BINDING_FIELDS,
                    "amplitude container binding",
                )
            )
            evidence = _candidate_amplitude_evidence_v1(
                record,
                review=_load_bound_review_v1(record),
                container_path=amplitude_path,
                container_sha256=amplitude_sha,
                official_provider=official_provider,
            )
        evidence_by_record[str(record["record_id"])] = evidence
        receipt_unsigned = {
            "schema_version": RECORD_RECEIPT_SCHEMA_VERSION,
            "plan_id": plan["plan_id"],
            "plan_digest": plan["plan_digest"],
            "runtime_digest": plan["runtime"]["frozen_runtime_identity"]["runtime_digest"],
            "provider_abi": OFFICIAL_PROVIDER_ABI,
            "official_provider": official_provider,
            "test_only": not official_provider,
            "record_ordinal": computed.ordinal,
            "record_id": record["record_id"],
            "record_digest": record["record_digest"],
            "record_kind": record["record_kind"],
            "evidence_id": record["evidence_id"],
            "evidence_role": record["evidence_role"],
            "teacher_cell_id": record["teacher_cell_id"],
            "branch": record["branch"],
            "record_authority": dict(record),
            "record_conditions": [dict(item) for item in record["conditions"]],
            "review_digest": record["review"]["review_digest"],
            "reviewed_media_sha256": record["reviewed_media"]["file_sha256"],
            "target_clean_latent_raw_sha256": record["target_clean_latent"]["tensor_raw_sha256"],
            "target_clean_latent_authority_digest": (
                None
                if record["target_clean_latent_authority"] is None
                else record["target_clean_latent_authority"]["digest"]
            ),
            "source_clean_latent_raw_sha256": (
                None
                if record["source_clean_latent"] is None
                else record["source_clean_latent"]["tensor_raw_sha256"]
            ),
            "source_posterior_index0_sha256": record[
                "source_posterior_index0_sha256"
            ],
            "noise_seed": record["noise"]["seed"],
            "noise_raw_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],
            "noise_replay_receipt": dict(computed.noise_receipt),
            "sigma_authority_digest": sigma_authority_digest,
            "state_receipts": computed.state_receipts,
            "forward_receipts": computed.forward_receipts,
            "container_bindings": container_bindings,
            "sigma_metrics": computed.sigma_rows,
            "candidate_authority_evidence": dict(evidence),
            "generated_rgb_decoded": False,
            "generated_rgb_used_as_model_input": False,
            "generated_rgb_used_as_regression_target": False,
            "generated_latent_used_as_absolute_regression_target": False,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        record_receipt = seal_record(receipt_unsigned, "record_receipt_digest")
        _closed(
            record_receipt,
            _RECORD_RECEIPT_FIELDS,
            f"{record['record_id']}.materialization record receipt",
        )
        receipt_path = record_directory / "materialization-record.json"
        receipt_sha = _write_json_create_only_v1(receipt_path, record_receipt)
        receipt_refs.append(
            _closed(
                {
                "record_id": record["record_id"],
                "record_kind": record["record_kind"],
                "path": str(receipt_path),
                "file_sha256": receipt_sha,
                "record_receipt_digest": record_receipt["record_receipt_digest"],
                "candidate_evidence_digest": evidence["evidence_digest"],
                },
                _RUN_RECORD_REFERENCE_FIELDS,
                "run record-receipt reference",
            )
        )
    return receipt_refs, MappingProxyType(evidence_by_record)


def _representation_fragments_v1(
    computed_records: Sequence[_ComputedRecordV1],
    evidence_by_record: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, _ComputedRecordV1]] = {}
    for computed in computed_records:
        record = computed.record
        if record["record_kind"] != "teacher_anchor":
            continue
        key = (str(record["teacher_cell_id"]), str(record["branch"]))
        grouped.setdefault(key, {})[str(record["evidence_role"])] = computed
    fragments: list[Mapping[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0].encode("utf-8"), item[1])):
        roles = grouped[key]
        _require(set(roles) == set(TEACHER_EVIDENCE_ROLES), f"teacher fragment {key!r} role closure differs")
        origin = roles["teacher_origin"]
        cross = roles["same_event_cross_anchor"]
        origin_rows = {int(row["sigma_index"]): row for row in origin.sigma_rows}
        cross_rows = {int(row["sigma_index"]): row for row in cross.sigma_rows}
        sigma_evidence: list[Mapping[str, Any]] = []
        for sigma_index in SIGMA_INDICES:
            origin_row = origin_rows[sigma_index]
            cross_row = cross_rows[sigma_index]
            origin_unit = origin.tensors[f"sigma_{sigma_index:02d}:projected_unit"]
            cross_unit = cross.tensors[f"sigma_{sigma_index:02d}:projected_unit"]
            row = {
                "sigma_index": sigma_index,
                "origin_projected_slice_sha256": origin_row["projected_unit_sha256"],
                "cross_anchor_projected_slice_sha256": cross_row["projected_unit_sha256"],
                "same_event_cosine": _fp32_cosine(origin_unit, cross_unit),
                "duplicate_forward_first_sha256": origin_row["duplicate_forward_first_sha256"],
                "duplicate_forward_second_sha256": origin_row["duplicate_forward_second_sha256"],
                "duplicate_forward_bytes_identical": origin_row["duplicate_forward_bytes_identical"],
                "same_state_noop_minus_noop_null_norm": origin_row["same_state_noop_minus_noop_null_norm"],
                "projected_teacher_raw_norm": origin_row["projected_teacher_raw_norm"],
                "signal_to_null_snr": origin_row["signal_to_null_snr"],
                "camera_residual_cosine": origin_row["camera_residual_cosine"],
                "appearance_residual_cosine": origin_row["appearance_residual_cosine"],
                "wrong_controls": origin_row["wrong_controls"],
            }
            _closed(row, teacher_authority._SIGMA_EVIDENCE_FIELDS, "representation sigma fragment")
            sigma_evidence.append(row)
        fragments.append(
            _closed(
                {
                "teacher_cell_id": key[0],
                "branch": key[1],
                "origin_record_id": origin.record["record_id"],
                "cross_anchor_record_id": cross.record["record_id"],
                "origin_evidence_digest": evidence_by_record[str(origin.record["record_id"])]["evidence_digest"],
                "cross_anchor_evidence_digest": evidence_by_record[str(cross.record["record_id"])]["evidence_digest"],
                "sigma_evidence": sigma_evidence,
                },
                _REPRESENTATION_FRAGMENT_FIELDS,
                "representation candidate fragment",
            )
        )
    return fragments


def _amplitude_calibration_fragments_v1(
    computed_records: Sequence[_ComputedRecordV1],
    evidence_by_record: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, str], list[_ComputedRecordV1]] = {}
    for computed in computed_records:
        record = computed.record
        if record["record_kind"] == "amplitude_calibrator":
            grouped.setdefault(
                (str(record["teacher_cell_id"]), str(record["branch"])), []
            ).append(computed)
    result: list[Mapping[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0].encode("utf-8"), item[1])):
        calibrators = sorted(
            grouped[key], key=lambda item: str(item.record["evidence_id"]).encode("utf-8")
        )
        _require(
            len(calibrators) == amplitude_authority.CALIBRATORS_PER_BUNDLE,
            f"amplitude calibration fragment {key!r} count differs",
        )
        rows_by_record = {
            str(item.record["record_id"]): {
                int(row["sigma_index"]): row for row in item.sigma_rows
            }
            for item in calibrators
        }
        sigma_calibrations: list[Mapping[str, Any]] = []
        for sigma_index in SIGMA_INDICES:
            metrics: list[Mapping[str, Any]] = []
            norms: list[float] = []
            for computed in calibrators:
                record = computed.record
                row = rows_by_record[str(record["record_id"])][sigma_index]
                metric = {
                    "evidence_id": record["evidence_id"],
                    "pair_id": record["pair_id"],
                    "projected_slice_sha256": row["projected_slice_sha256"],
                    "amplitude_norm": row["amplitude_norm"],
                }
                _closed(metric, amplitude_authority._CALIBRATOR_METRIC_FIELDS, "amplitude metric fragment")
                metrics.append(metric)
                norms.append(float(row["amplitude_norm"]))
            median = math.fsum(sorted(norms)) / 2.0
            _, floor_hex, floor_sha = amplitude_authority._float32(
                amplitude_authority.AMPLITUDE_SCALE * median
            )
            sigma_row = {
                "sigma_index": sigma_index,
                "calibrator_metrics": metrics,
                "median_amplitude": median,
                "a_min_scale": amplitude_authority.AMPLITUDE_SCALE,
                "a_min_float32_be_hex": floor_hex,
                "a_min_float32_le_sha256": floor_sha,
            }
            _closed(sigma_row, amplitude_authority._SIGMA_CALIBRATION_FIELDS, "amplitude sigma calibration fragment")
            sigma_calibrations.append(sigma_row)
        result.append(
            _closed(
                {
                "teacher_cell_id": key[0],
                "branch": key[1],
                "calibrator_record_ids": [
                    item.record["record_id"] for item in calibrators
                ],
                "calibrator_evidence_candidates": [
                    dict(evidence_by_record[str(item.record["record_id"])])
                    for item in calibrators
                ],
                "sigma_calibrations": sigma_calibrations,
                },
                _AMPLITUDE_CALIBRATION_FRAGMENT_FIELDS,
                "amplitude calibration candidate fragment",
            )
        )
    return result


def _computed_population_digest_v1(
    computed_records: Sequence[_ComputedRecordV1],
) -> str:
    rows: list[Mapping[str, Any]] = []
    for computed in sorted(computed_records, key=lambda item: item.ordinal):
        rows.append(
            {
                "ordinal": computed.ordinal,
                "record_id": computed.record["record_id"],
                "record_digest": computed.record["record_digest"],
                "tensor_slice_sha256": {
                    name: tensor.raw_sha256() for name, tensor in computed.tensors.items()
                },
                "nuisance_slice_sha256": {
                    name: tensor.raw_sha256()
                    for name, tensor in computed.nuisance_tensors.items()
                },
                "noise_replay_digest": computed.noise_receipt["noise_digest"],
                "state_receipt_digests": [
                    row["state_digest"] for row in computed.state_receipts
                ],
                "forward_receipt_digests": [
                    row["forward_digest"] for row in computed.forward_receipts
                ],
                "sigma_rows": computed.sigma_rows,
            }
        )
    return object_sha256(rows)


def _materialize_with_provider_v1(
    plan: Mapping[str, Any],
    *,
    output_directory: str | Path,
    provider: FrozenForwardProviderV1,
    numerics: PsiOutNumericsV1,
    require_official: bool,
) -> MaterializationResultV1:
    plan = validate_materialization_plan_v1(plan)
    _require(type(provider.is_official) is bool, "provider official identity differs")
    _require(not require_official or provider.is_official, "formal materialization rejects injected providers")
    output_path = Path(output_directory)
    _require(output_path.is_absolute(), "output directory must be absolute")
    _require(not output_path.exists() and not output_path.is_symlink(), "create-only output path already exists")
    runtime_digest = str(plan["runtime"]["frozen_runtime_identity"]["runtime_digest"])
    computed_by_record: dict[str, _ComputedRecordV1] = {}
    origin_packets: dict[tuple[str, str], Mapping[int, Any]] = {}
    ordinal_by_record = {
        str(record["record_id"]): ordinal for ordinal, record in enumerate(plan["records"])
    }
    for record in plan["records"]:
        if record["record_kind"] != "teacher_anchor":
            continue
        computed = _compute_teacher_record_v1(
            ordinal=ordinal_by_record[str(record["record_id"])],
            record=record,
            provider=provider,
            numerics=numerics,
            runtime_digest=runtime_digest,
            require_official=require_official,
        )
        computed_by_record[str(record["record_id"])] = computed
        if record["evidence_role"] == "teacher_origin":
            key = (str(record["teacher_cell_id"]), str(record["branch"]))
            _require(key not in origin_packets, f"duplicate teacher nuisance packet authority: {key!r}")
            origin_packets[key] = MappingProxyType(dict(computed.packets))
    for record in plan["records"]:
        if record["record_kind"] != "amplitude_calibrator":
            continue
        key = (str(record["teacher_cell_id"]), str(record["branch"]))
        _require(key in origin_packets, f"amplitude record has no materialized origin nuisance packet: {key!r}")
        computed = _compute_amplitude_record_v1(
            ordinal=ordinal_by_record[str(record["record_id"])],
            record=record,
            provider=provider,
            numerics=numerics,
            runtime_digest=runtime_digest,
            require_official=require_official,
            nuisance_packets=origin_packets[key],
        )
        computed_by_record[str(record["record_id"])] = computed
    _require(
        len(computed_by_record) == len(plan["records"]),
        "computed record population does not close the plan",
    )
    computed_records = tuple(
        sorted(computed_by_record.values(), key=lambda item: item.ordinal)
    )
    computation_digest = _computed_population_digest_v1(computed_records)
    provider.consensus_digest_v1(
        computation_digest, "materialized same-state PsiOut population"
    )
    rank = getattr(provider, "rank", 0)
    _require(type(rank) is int and rank >= 0, "provider rank differs")
    io_payload: Any = None
    if rank == 0:
        try:
            output_root = _fresh_output_directory_v1(output_path)
            receipt_refs, evidence_by_record = _write_computed_records_v1(
                output_root,
                computed_records,
                plan=plan,
                official_provider=bool(provider.is_official),
            )
            representation_fragments = _representation_fragments_v1(
                computed_records, evidence_by_record
            )
            amplitude_fragments = _amplitude_calibration_fragments_v1(
                computed_records, evidence_by_record
            )
            sigma_authority = dict(sigma_authority_receipt_v1())
            run_unsigned = {
                "schema_version": RUN_RECEIPT_SCHEMA_VERSION,
                "plan_id": plan["plan_id"],
                "plan_digest": plan["plan_digest"],
                "plan_authority": dict(plan),
                "population_digest": plan["population"]["population_digest"],
                "record_order_sha256": plan["population"]["record_order_sha256"],
                "runtime_identity": dict(plan["runtime"]["frozen_runtime_identity"]),
                "runtime_plan_digest": plan["runtime"]["runtime_plan_digest"],
                "official_helper_sources": plan["runtime"]["official_helper_sources"],
                "provider_abi": OFFICIAL_PROVIDER_ABI,
                "official_provider": bool(provider.is_official),
                "test_only": not bool(provider.is_official),
                "world_size": WORLD_SIZE if provider.is_official else 1,
                "dp_size": DP_SIZE if provider.is_official else 1,
                "sp_size": SP_SIZE if provider.is_official else 1,
                "sigma_indices": list(SIGMA_INDICES),
                "sigma_authority": sigma_authority,
                "record_count": len(computed_records),
                "computation_digest": computation_digest,
                "record_receipts": receipt_refs,
                "representation_sigma_evidence_candidates": representation_fragments,
                "amplitude_sigma_calibration_candidates": amplitude_fragments,
                "output_policy": dict(plan["output_policy"]),
                "generated_rgb_decoded": False,
                "generated_rgb_used_as_model_input": False,
                "generated_rgb_used_as_regression_target": False,
                "generated_latent_used_as_absolute_regression_target": False,
                "model_parameters_updated": False,
                "optimizer_created": False,
            }
            run_receipt = seal_record(run_unsigned, "run_digest")
            _closed(run_receipt, _RUN_RECEIPT_FIELDS, "materialization run receipt")
            run_path = output_root / "materialization-run.json"
            run_sha = _write_json_create_only_v1(run_path, run_receipt)
            io_payload = {
                "ok": True,
                "output_directory": str(output_root),
                "run_receipt_path": str(run_path),
                "run_receipt_sha256": run_sha,
                "run_digest": run_receipt["run_digest"],
            }
        except Exception as error:
            io_payload = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    io_payload = provider.broadcast_rank0_v1(io_payload)
    if not isinstance(io_payload, Mapping) or io_payload.get("ok") is not True:
        fail(f"rank-zero materialization write failed: {io_payload!r}")
    provider.barrier_v1()
    run_path = Path(str(io_payload["run_receipt_path"]))
    _, run_raw = _read_stable_plain_file(
        str(run_path),
        io_payload["run_receipt_sha256"],
        label="written materialization run receipt",
        maximum_bytes=MAX_JSON_BYTES,
    )
    _require(run_raw.endswith(b"\n"), "written materialization run receipt lacks canonical newline")
    try:
        run_value = json.loads(
            run_raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30PsiOutMaterializationError("written run receipt cannot be decoded") from error
    _closed(run_value, _RUN_RECEIPT_FIELDS, "written run receipt")
    _verify_seal(run_value, "run_digest", "written run receipt")
    _require(run_value["run_digest"] == io_payload["run_digest"], "written run digest differs")
    return MaterializationResultV1(
        output_directory=Path(str(io_payload["output_directory"])),
        run_receipt_path=run_path,
        run_receipt_sha256=str(io_payload["run_receipt_sha256"]),
        run_digest=str(io_payload["run_digest"]),
        record_count=len(computed_records),
        official_provider=bool(provider.is_official),
        test_only=not bool(provider.is_official),
    )


def materialize_with_test_provider_v1(
    plan: Mapping[str, Any],
    *,
    output_directory: str | Path,
    provider: FrozenForwardProviderV1,
    numerics: Optional[PsiOutNumericsV1] = None,
) -> MaterializationResultV1:
    """Explicit dependency-free seam; outputs are sealed ``test_only=true``."""

    _require(provider.is_official is False, "test-provider API rejects an official provider")
    return _materialize_with_provider_v1(
        plan,
        output_directory=output_directory,
        provider=provider,
        numerics=numerics or ReferenceFP32NumericsV1(),
        require_official=False,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize official WORLD4/SP4 same-state full30 PsiOut evidence"
    )
    parser.add_argument("--plan", required=True, help="absolute sealed materialization-plan JSON")
    parser.add_argument("--plan-sha256", required=True, help="expected physical plan-file SHA-256")
    parser.add_argument("--output-dir", required=True, help="absolute fresh create-only output directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    provider: Optional[OfficialWorld4SP4FrozenProviderV1] = None
    try:
        plan = load_materialization_plan_file_v1(args.plan, args.plan_sha256)
        provider = OfficialWorld4SP4FrozenProviderV1(plan)
        result = _materialize_with_provider_v1(
            plan,
            output_directory=args.output_dir,
            provider=provider,
            numerics=TorchOfficialNumericsV1(
                expected_protocol_path=plan["runtime"]["psiout_protocol_path"]
            ),
            require_official=True,
        )
    except Exception as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        return 2
    finally:
        if provider is not None and not provider._closed:
            try:
                provider.close()
            except Exception as error:  # pragma: no cover - distributed teardown
                print(f"FAIL-CLOSED during provider teardown: {error}", file=sys.stderr)
                return 2
    print(
        canonical_json_bytes(
            {
                "status": "MATERIALIZED_OFFICIAL_NO_OPTIMIZER",
                "output_directory": str(result.output_directory),
                "run_receipt_path": str(result.run_receipt_path),
                "run_receipt_sha256": result.run_receipt_sha256,
                "run_digest": result.run_digest,
                "record_count": result.record_count,
            }
        ).decode("ascii")
    )
    return 0


__all__ = [
    "EXACT_NOOP_INSTRUCTION",
    "FP32TensorV1",
    "FORWARD_RECEIPT_SCHEMA_VERSION",
    "FrozenForwardResultV1",
    "Full30PsiOutMaterializationError",
    "MaterializationResultV1",
    "NOISE_RECEIPT_SCHEMA_VERSION",
    "OFFICIAL_PROVIDER_ABI",
    "OfficialWorld4SP4FrozenProviderV1",
    "PLAN_ARTIFACT_SCHEMA_VERSION",
    "PLAN_CONDITION_SCHEMA_VERSION",
    "PLAN_OUTPUT_POLICY_SCHEMA_VERSION",
    "PLAN_POPULATION_SCHEMA_VERSION",
    "PLAN_RECORD_SCHEMA_VERSION",
    "PLAN_REVIEW_SCHEMA_VERSION",
    "PLAN_RUNTIME_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "POST_HEAD_STAGE",
    "PreparedSameStateV1",
    "ReferenceFP32NumericsV1",
    "RECORD_RECEIPT_SCHEMA_VERSION",
    "RUN_RECEIPT_SCHEMA_VERSION",
    "SIGMA_AUTHORITY_RECEIPT_SCHEMA_VERSION",
    "STATE_RECEIPT_SCHEMA_VERSION",
    "amplitude_noise_seed_v1",
    "canonical_json_bytes",
    "frozen_compute_contract_v1",
    "load_materialization_plan_file_v1",
    "materialize_with_test_provider_v1",
    "object_sha256",
    "seal_record",
    "sigma_authority_receipt_v1",
    "teacher_noise_seed_v1",
    "validate_materialization_plan_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
