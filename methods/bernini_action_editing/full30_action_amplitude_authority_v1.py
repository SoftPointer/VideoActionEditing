#!/usr/bin/env python3
"""Strict same-mode amplitude-floor authority for full30 action learning.

The action direction teacher is source-free, but its absolute scale is not
portable to real-source RV2V.  This module therefore admits only amplitude
floors computed before training from reviewed Frozen-RV2V calibrators.  It
does not materialize evidence, infer missing values, or inspect trainable
outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

try:
    import full30_action_data_teacher_authority_v1 as parent_authority
except ImportError:  # pragma: no cover - package import mode
    from . import full30_action_data_teacher_authority_v1 as parent_authority


SCHEMA_VERSION = "bernini-full30-action-amplitude-authority-v2"
VALIDATION_SCHEMA_VERSION = "bernini-full30-action-amplitude-validation-v2"
PARENT_BINDING_SCHEMA_VERSION = "bernini-full30-action-parent-authority-binding-v1"
RUNTIME_IDENTITY_SCHEMA_VERSION = "bernini-full30-action-frozen-runtime-identity-v2"
COMPUTE_CONTRACT_SCHEMA_VERSION = "bernini-full30-action-frozen-compute-contract-v1"
BUNDLE_SCHEMA_VERSION = "bernini-full30-action-amplitude-calibration-bundle-v1"
EVIDENCE_SCHEMA_VERSION = "bernini-full30-action-amplitude-evidence-v1"
REVIEW_SCHEMA_VERSION = "bernini-full30-action-frozen-baseline-review-v1"
CONTAINER_SCHEMA_VERSION = "bernini-full30-action-amplitude-container-v1"

BRANCHES = ("action", "incomplete")
SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
EXPECTED_FIT_TEACHER_CELLS = 8
EXPECTED_BUNDLES = EXPECTED_FIT_TEACHER_CELLS * len(BRANCHES)
CALIBRATORS_PER_BUNDLE = 2
FAIL_CONTROLS_PER_BUNDLE = 2
AMPLITUDE_SCALE = 0.25
MINIMUM_ADMITTED_FLOOR = 1.0e-6

TENSOR_SHAPE = (21, 32)
TENSOR_ELEMENTS = math.prod(TENSOR_SHAPE)
TENSOR_SLICE_BYTES = TENSOR_ELEMENTS * 4
TENSOR_DTYPE = "float32-le"
CONTAINER_MODE = 0o600
CONTAINER_MAGIC = b"BERNINI-FULL30-AMIN-V1\x00"
CONTAINER_LAYOUT = "payload-relative-contiguous-v1"
CONTAINER_MAX_HEADER_BYTES = 64 * 1024
CONTAINER_MAX_FILE_BYTES = 64 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FLOAT32_HEX = re.compile(r"^[0-9a-f]{8}$")


class Full30AmplitudeAuthorityError(RuntimeError):
    """Raised before unsealed or inconsistent amplitude evidence is used."""


def fail(message: str) -> NoReturn:
    raise Full30AmplitudeAuthorityError(message)


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
        raise Full30AmplitudeAuthorityError(
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


def _verify_seal(value: Mapping[str, Any], digest_field: str, label: str) -> None:
    declared = _sha(value.get(digest_field), f"{label}.{digest_field}")
    unsigned = dict(value)
    del unsigned[digest_field]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be a non-empty safe identifier")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        fail(f"{label} must be a finite number")
    return float(value)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_file(value: Any, label: str, *, exact_mode: Optional[int] = None) -> Path:
    if type(value) is not str:
        fail(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Full30AmplitudeAuthorityError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a plain non-symlink file")
    if exact_mode is not None:
        _require(
            stat.S_IMODE(metadata.st_mode) == exact_mode,
            f"{label} mode must be exactly {exact_mode:#o}",
        )
    return path.resolve(strict=True)


def _verify_file(value: Any, expected_sha256: Any, label: str) -> Path:
    path = _plain_file(value, label)
    expected = _sha(expected_sha256, f"{label} SHA-256")
    _require(file_sha256(path) == expected, f"{label} file SHA-256 differs")
    return path


def _load_json_file(path: str | Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    source = _plain_file(str(path), label)
    expected = _sha(expected_sha256, f"{label} expected SHA-256")
    raw = source.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, f"{label} file SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30AmplitudeAuthorityError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    return value


def _float32(value: float) -> tuple[float, str, str]:
    encoded_be = struct.pack(">f", float(value))
    quantized = struct.unpack(">f", encoded_be)[0]
    encoded_le = struct.pack("<f", quantized)
    return quantized, encoded_be.hex(), hashlib.sha256(encoded_le).hexdigest()


def _tensor_name(sigma_index: int) -> str:
    return f"sigma_{sigma_index:02d}:projected_raw"


@dataclass(frozen=True)
class _TensorSlice:
    values: tuple[float, ...]
    sha256: str


_CONTAINER_FIELDS = {
    "schema_version",
    "evidence_id",
    "pair_id",
    "source_iid",
    "teacher_cell_id",
    "branch",
    "dtype",
    "shape",
    "sigma_indices",
    "layout",
    "tensor_count",
    "payload_bytes",
    "entries",
}
_ENTRY_FIELDS = {"name", "sigma_index", "dtype", "shape", "offset", "length", "sha256"}


def _read_container_bytes(path_value: Any, sha_value: Any, label: str) -> bytes:
    path = _plain_file(path_value, label, exact_mode=CONTAINER_MODE)
    expected = _sha(sha_value, f"{label} SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _require(
            0 < before.st_size <= CONTAINER_MAX_FILE_BYTES,
            f"{label} size is outside the bounded container range",
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
    _require(hashlib.sha256(raw).hexdigest() == expected, f"{label} file SHA-256 differs")
    return raw


def _validate_container(
    path_value: Any,
    sha_value: Any,
    *,
    evidence_id: str,
    pair_id: str,
    source_iid: str,
    teacher_cell_id: str,
    branch: str,
    label: str,
) -> Mapping[int, _TensorSlice]:
    raw = _read_container_bytes(path_value, sha_value, label)
    prefix = len(CONTAINER_MAGIC) + 4
    _require(len(raw) >= prefix and raw.startswith(CONTAINER_MAGIC), f"{label} magic differs")
    header_length = struct.unpack(">I", raw[len(CONTAINER_MAGIC):prefix])[0]
    _require(
        0 < header_length <= CONTAINER_MAX_HEADER_BYTES,
        f"{label} header length differs",
    )
    payload_start = prefix + header_length
    _require(payload_start <= len(raw), f"{label} header is truncated")
    header_bytes = raw[prefix:payload_start]
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30AmplitudeAuthorityError(f"{label} header cannot be decoded") from error
    header = _closed(header, _CONTAINER_FIELDS, f"{label}.header")
    _require(canonical_json_bytes(header) == header_bytes, f"{label} header is not canonical")
    expected_bindings = {
        "schema_version": CONTAINER_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "pair_id": pair_id,
        "source_iid": source_iid,
        "teacher_cell_id": teacher_cell_id,
        "branch": branch,
        "dtype": TENSOR_DTYPE,
        "shape": list(TENSOR_SHAPE),
        "sigma_indices": list(SIGMA_INDICES),
        "layout": CONTAINER_LAYOUT,
        "tensor_count": len(SIGMA_INDICES),
        "payload_bytes": len(SIGMA_INDICES) * TENSOR_SLICE_BYTES,
    }
    for field, expected in expected_bindings.items():
        _require(header[field] == expected, f"{label}.{field} differs")
    payload = raw[payload_start:]
    _require(
        len(payload) == expected_bindings["payload_bytes"],
        f"{label} payload length/extra bytes differ",
    )
    entries = header["entries"]
    if type(entries) is not list or len(entries) != len(SIGMA_INDICES):
        fail(f"{label} entries do not close six sigmas")
    result: dict[int, _TensorSlice] = {}
    for ordinal, sigma_index in enumerate(SIGMA_INDICES):
        entry_label = f"{label}.entries[{ordinal}]"
        entry = _closed(entries[ordinal], _ENTRY_FIELDS, entry_label)
        expected_offset = ordinal * TENSOR_SLICE_BYTES
        _require(entry["name"] == _tensor_name(sigma_index), f"{entry_label}.name differs")
        _require(entry["sigma_index"] == sigma_index, f"{entry_label}.sigma_index differs")
        _require(entry["dtype"] == TENSOR_DTYPE, f"{entry_label}.dtype differs")
        _require(entry["shape"] == list(TENSOR_SHAPE), f"{entry_label}.shape differs")
        _require(entry["offset"] == expected_offset, f"{entry_label}.offset differs")
        _require(entry["length"] == TENSOR_SLICE_BYTES, f"{entry_label}.length differs")
        tensor_bytes = payload[expected_offset:expected_offset + TENSOR_SLICE_BYTES]
        actual_sha = hashlib.sha256(tensor_bytes).hexdigest()
        _require(
            _sha(entry["sha256"], f"{entry_label}.sha256") == actual_sha,
            f"{entry_label} tensor SHA-256 differs",
        )
        values = struct.unpack(f"<{TENSOR_ELEMENTS}f", tensor_bytes)
        _require(all(math.isfinite(item) for item in values), f"{entry_label} is non-finite")
        result[sigma_index] = _TensorSlice(tuple(values), actual_sha)
    return MappingProxyType(result)


_REVIEW_FIELDS = {
    "schema_version",
    "review_id",
    "evidence_id",
    "pair_id",
    "source_iid",
    "branch",
    "baseline_output_sha256",
    "frame_count",
    "fps",
    "sampler_steps",
    "entire_full81_video_viewed",
    "independent_reviewer",
    "reviewer_blinded_to_amplitude_metrics",
    "sealed_before_sidecar_extraction",
    "sealed_before_optimizer_authority",
    "action_result",
    "review_digest",
}


def _validate_review(
    value: Any,
    *,
    evidence_id: str,
    pair: Mapping[str, Any],
    output_sha256: str,
    allowed_results: set[str],
    label: str,
) -> Mapping[str, Any]:
    row = _closed(value, _REVIEW_FIELDS, label)
    _verify_seal(row, "review_digest", label)
    _require(row["schema_version"] == REVIEW_SCHEMA_VERSION, f"{label}.schema differs")
    _safe_id(row["review_id"], f"{label}.review_id")
    expected = {
        "evidence_id": evidence_id,
        "pair_id": pair["pair_id"],
        "source_iid": pair["source_iid"],
        "branch": pair["branch"],
        "baseline_output_sha256": output_sha256,
        "frame_count": 81,
        "sampler_steps": 40,
    }
    for field, expected_value in expected.items():
        _require(row[field] == expected_value, f"{label}.{field} differs")
    _require(_number(row["fps"], f"{label}.fps") == 25.0, f"{label}.fps differs")
    for field in (
        "entire_full81_video_viewed",
        "independent_reviewer",
        "reviewer_blinded_to_amplitude_metrics",
        "sealed_before_sidecar_extraction",
        "sealed_before_optimizer_authority",
    ):
        _require(row[field] is True, f"{label}.{field} is not true")
    _require(row["action_result"] in allowed_results, f"{label}.action_result differs")
    return row


_COMMON_EVIDENCE_FIELDS = {
    "schema_version",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "branch",
    "pair_id",
    "source_iid",
    "source_posterior_index0_sha256",
    "instruction_utf8_sha256",
    "baseline_output_path",
    "baseline_output_sha256",
    "initial_gaussian_sha256",
    "same_source_noise_sigma_state",
    "official_frozen_native_only",
    "pre_admission_review",
    "evidence_digest",
}
_CALIBRATOR_EXTRA_FIELDS = {
    "calibrator_noise_seed",
    "calibrator_noise_sha256",
    "amplitude_container_path",
    "amplitude_container_sha256",
}


@dataclass(frozen=True)
class _Calibrator:
    row: Mapping[str, Any]
    tensors: Mapping[int, _TensorSlice]
    materialization_receipt: Mapping[str, Any]


def calibrator_noise_seed_v1(pair_id: str) -> int:
    payload = b"full30-amplitude-calibrator-noise-v1\x00" + pair_id.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_evidence(
    value: Any,
    *,
    role: str,
    pair_by_id: Mapping[str, Mapping[str, Any]],
    teacher_cell_id: str,
    branch: str,
    run_authority: parent_authority._MaterializationRunV1,
    label: str,
) -> tuple[
    Mapping[str, Any],
    Optional[Mapping[int, _TensorSlice]],
    Optional[Mapping[str, Any]],
]:
    fields = set(_COMMON_EVIDENCE_FIELDS)
    if role == "calibrator":
        fields |= _CALIBRATOR_EXTRA_FIELDS | parent_authority._MATERIALIZATION_PROVENANCE_FIELDS
    row = _closed(value, fields, label)
    _verify_seal(row, "evidence_digest", label)
    _require(row["schema_version"] == EVIDENCE_SCHEMA_VERSION, f"{label}.schema differs")
    _require(row["evidence_role"] == role, f"{label}.evidence_role differs")
    evidence_id = _safe_id(row["evidence_id"], f"{label}.evidence_id")
    _require(row["teacher_cell_id"] == teacher_cell_id, f"{label}.teacher_cell_id differs")
    _require(row["branch"] == branch, f"{label}.branch differs")
    pair_id = _safe_id(row["pair_id"], f"{label}.pair_id")
    _require(pair_id in pair_by_id, f"{label} references an unknown fit pair")
    pair = pair_by_id[pair_id]
    for field in ("source_iid", "teacher_cell_id", "branch"):
        _require(row[field] == pair[field], f"{label}.{field} differs from pair authority")
    _require(
        row["source_posterior_index0_sha256"] == pair["source_posterior_index0_sha256"],
        f"{label}.source_posterior_index0_sha256 differs",
    )
    _require(
        row["instruction_utf8_sha256"] == pair["instruction_utf8_sha256"],
        f"{label}.instruction_utf8_sha256 differs",
    )
    output_sha = _sha(row["baseline_output_sha256"], f"{label}.baseline_output_sha256")
    _verify_file(row["baseline_output_path"], output_sha, f"{label}.baseline_output")
    _sha(row["initial_gaussian_sha256"], f"{label}.initial_gaussian_sha256")
    _require(row["same_source_noise_sigma_state"] is True, f"{label} is not same-state")
    _require(row["official_frozen_native_only"] is True, f"{label} is not official Frozen")
    allowed_results = {"partial", "pass"} if role == "calibrator" else {"fail"}
    review = _validate_review(
        row["pre_admission_review"],
        evidence_id=evidence_id,
        pair=pair,
        output_sha256=output_sha,
        allowed_results=allowed_results,
        label=f"{label}.pre_admission_review",
    )
    tensors: Optional[Mapping[int, _TensorSlice]] = None
    materialization_receipt: Optional[Mapping[str, Any]] = None
    if role == "calibrator":
        expected_seed = calibrator_noise_seed_v1(pair_id)
        _require(row["calibrator_noise_seed"] == expected_seed, f"{label}.noise seed differs")
        _sha(row["calibrator_noise_sha256"], f"{label}.calibrator_noise_sha256")
        tensors = _validate_container(
            row["amplitude_container_path"],
            row["amplitude_container_sha256"],
            evidence_id=evidence_id,
            pair_id=pair_id,
            source_iid=str(pair["source_iid"]),
            teacher_cell_id=teacher_cell_id,
            branch=branch,
            label=f"{label}.amplitude_container",
        )
        try:
            materialization_receipt = (
                parent_authority._validate_materialization_record_for_evidence(
                    row,
                    run_authority=run_authority,
                    container_slices={
                        "amplitude": {
                            _tensor_name(sigma_index): tensor.sha256
                            for sigma_index, tensor in tensors.items()
                        }
                    },
                    label=label,
                )
            )
        except Exception as error:
            raise Full30AmplitudeAuthorityError(
                f"{label} materialization provenance differs: {error}"
            ) from error
        _require(
            materialization_receipt["record_kind"] == "amplitude_calibrator",
            f"{label} materialization record is not an amplitude calibrator",
        )
    _require(
        review["action_result"] in allowed_results,
        f"{label} review result is not admitted for {role}",
    )
    return row, tensors, materialization_receipt


_CALIBRATOR_METRIC_FIELDS = {
    "evidence_id",
    "pair_id",
    "projected_slice_sha256",
    "amplitude_norm",
}
_SIGMA_CALIBRATION_FIELDS = {
    "sigma_index",
    "calibrator_metrics",
    "median_amplitude",
    "a_min_scale",
    "a_min_float32_be_hex",
    "a_min_float32_le_sha256",
}
_BUNDLE_FIELDS = {
    "schema_version",
    "calibration_id",
    "teacher_cell_id",
    "branch",
    "parent_representation_admission_digest",
    "frozen_fail_evidence",
    "calibrator_evidence",
    "sigma_calibrations",
    "optimizer_admitted",
    "bundle_digest",
}


@dataclass(frozen=True)
class AmplitudeFloorV1:
    teacher_cell_id: str
    branch: str
    sigma_index: int
    value_float32: float
    float32_be_hex: str
    float32_le_sha256: str
    bundle_digest: str
    calibration_id: str


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(item) * float(item) for item in values))


def _validate_bundle(
    value: Any,
    *,
    representation_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_by_id: Mapping[str, Mapping[str, Any]],
    run_authority: parent_authority._MaterializationRunV1,
    materialization_fragment: Mapping[str, Any],
    label: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[int, AmplitudeFloorV1],
    Mapping[str, tuple[str, ...]],
]:
    row = _closed(value, _BUNDLE_FIELDS, label)
    _verify_seal(row, "bundle_digest", label)
    _require(row["schema_version"] == BUNDLE_SCHEMA_VERSION, f"{label}.schema differs")
    calibration_id = _safe_id(row["calibration_id"], f"{label}.calibration_id")
    teacher_cell_id = _safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
    branch = row["branch"]
    _require(branch in BRANCHES, f"{label}.branch differs")
    key = (teacher_cell_id, str(branch))
    _require(key in representation_by_key, f"{label} has no admitted fit representation")
    _require(
        row["parent_representation_admission_digest"]
        == representation_by_key[key]["admission_digest"],
        f"{label} parent representation digest differs",
    )
    _require(row["optimizer_admitted"] is True, f"{label} is not optimizer-admitted")

    fail_values = row["frozen_fail_evidence"]
    calibrator_values = row["calibrator_evidence"]
    if type(fail_values) is not list or len(fail_values) != FAIL_CONTROLS_PER_BUNDLE:
        fail(f"{label}.frozen_fail_evidence must contain exactly two rows")
    if type(calibrator_values) is not list or len(calibrator_values) != CALIBRATORS_PER_BUNDLE:
        fail(f"{label}.calibrator_evidence must contain exactly two rows")
    fails: list[Mapping[str, Any]] = []
    calibrators: list[_Calibrator] = []
    for ordinal, item in enumerate(fail_values):
        evidence, tensors, materialization_receipt = _validate_evidence(
            item,
            role="frozen_fail",
            pair_by_id=pair_by_id,
            teacher_cell_id=teacher_cell_id,
            branch=str(branch),
            run_authority=run_authority,
            label=f"{label}.frozen_fail_evidence[{ordinal}]",
        )
        _require(tensors is None, f"{label} fail evidence unexpectedly has tensors")
        _require(materialization_receipt is None, f"{label} fail evidence unexpectedly has materialization provenance")
        fails.append(evidence)
    for ordinal, item in enumerate(calibrator_values):
        evidence, tensors, materialization_receipt = _validate_evidence(
            item,
            role="calibrator",
            pair_by_id=pair_by_id,
            teacher_cell_id=teacher_cell_id,
            branch=str(branch),
            run_authority=run_authority,
            label=f"{label}.calibrator_evidence[{ordinal}]",
        )
        _require(tensors is not None, f"{label} calibrator tensors are absent")
        _require(materialization_receipt is not None, f"{label} calibrator materialization receipt is absent")
        calibrators.append(
            _Calibrator(evidence, tensors, materialization_receipt)  # type: ignore[arg-type]
        )
    all_evidence = [*fails, *(item.row for item in calibrators)]
    pair_ids = [str(item["pair_id"]) for item in all_evidence]
    source_iids = [str(item["source_iid"]) for item in all_evidence]
    _require(len(set(pair_ids)) == 4, f"{label} fail/calibrator pairs are not distinct")
    _require(len(set(source_iids)) == 4, f"{label} fail/calibrator sources are not distinct")
    evidence_ids = tuple(str(item["evidence_id"]) for item in all_evidence)
    _require(len(set(evidence_ids)) == 4, f"{label} evidence IDs are not distinct")
    review_ids = tuple(
        str(item["pre_admission_review"]["review_id"]) for item in all_evidence
    )
    output_shas = tuple(str(item["baseline_output_sha256"]) for item in all_evidence)
    container_shas = tuple(
        str(item.row["amplitude_container_sha256"]) for item in calibrators
    )
    calibrator_noise_shas = tuple(
        str(item.row["calibrator_noise_sha256"]) for item in calibrators
    )
    for identity_label, identities, expected_count in (
        ("review IDs", review_ids, 4),
        ("baseline output SHA-256", output_shas, 4),
        ("amplitude container SHA-256", container_shas, 2),
        ("calibrator noise SHA-256", calibrator_noise_shas, 2),
    ):
        _require(
            len(set(identities)) == expected_count,
            f"{label} {identity_label} are not distinct",
        )

    sigma_values = row["sigma_calibrations"]
    if type(sigma_values) is not list or len(sigma_values) != len(SIGMA_INDICES):
        fail(f"{label}.sigma_calibrations must contain exactly six rows")
    floor_index: dict[int, AmplitudeFloorV1] = {}
    ordered_calibrators = sorted(calibrators, key=lambda item: str(item.row["evidence_id"]))
    parent_origin_evidence = representation_by_key[key]["origin_evidence"]
    parent_nuisance = parent_authority._validate_tensor_container(
        parent_origin_evidence["nuisance_packet_path"],
        parent_origin_evidence["nuisance_packet_sha256"],
        container_kind="nuisance",
        evidence_id=str(parent_origin_evidence["evidence_id"]),
        evidence_role="teacher_origin",
        teacher_cell_id=teacher_cell_id,
        branch=str(branch),
        label=f"{label}.parent_origin_nuisance",
    )
    for ordinal, sigma_index in enumerate(SIGMA_INDICES):
        sigma_label = f"{label}.sigma_calibrations[{ordinal}]"
        sigma_row = _closed(sigma_values[ordinal], _SIGMA_CALIBRATION_FIELDS, sigma_label)
        _require(sigma_row["sigma_index"] == sigma_index, f"{sigma_label}.sigma_index differs")
        metrics = sigma_row["calibrator_metrics"]
        if type(metrics) is not list or len(metrics) != CALIBRATORS_PER_BUNDLE:
            fail(f"{sigma_label}.calibrator_metrics must contain exactly two rows")
        actual_norms: list[float] = []
        for metric_ordinal, calibrator in enumerate(ordered_calibrators):
            metric_label = f"{sigma_label}.calibrator_metrics[{metric_ordinal}]"
            metric = _closed(metrics[metric_ordinal], _CALIBRATOR_METRIC_FIELDS, metric_label)
            tensor_slice = calibrator.tensors[sigma_index]
            actual_norm = _norm(tensor_slice.values)
            expected_metric = {
                "evidence_id": calibrator.row["evidence_id"],
                "pair_id": calibrator.row["pair_id"],
                "projected_slice_sha256": tensor_slice.sha256,
            }
            for field, expected in expected_metric.items():
                _require(metric[field] == expected, f"{metric_label}.{field} differs")
            claimed_norm = _number(metric["amplitude_norm"], f"{metric_label}.amplitude_norm")
            _require(
                math.isclose(claimed_norm, actual_norm, rel_tol=1e-12, abs_tol=1e-12),
                f"{metric_label}.amplitude_norm differs from tensor bytes",
            )
            materialization_metric = calibrator.materialization_receipt["sigma_metrics"][
                ordinal
            ]
            _require(
                materialization_metric["projected_slice_sha256"]
                == tensor_slice.sha256
                and math.isclose(
                    _number(
                        materialization_metric["amplitude_norm"],
                        f"{metric_label}.materialization amplitude_norm",
                    ),
                    actual_norm,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"{metric_label} differs from materialization receipt",
            )
            _require(
                materialization_metric["teacher_nuisance_camera_sha256"]
                == parent_nuisance[
                    parent_authority._tensor_name(sigma_index, "camera_unit")
                ][2]
                and materialization_metric[
                    "teacher_nuisance_appearance_sha256"
                ]
                == parent_nuisance[
                    parent_authority._tensor_name(sigma_index, "appearance_unit")
                ][2],
                f"{metric_label} teacher nuisance binding differs",
            )
            actual_norms.append(actual_norm)
        median = math.fsum(sorted(actual_norms)) / 2.0
        claimed_median = _number(sigma_row["median_amplitude"], f"{sigma_label}.median")
        _require(
            math.isclose(claimed_median, median, rel_tol=1e-12, abs_tol=1e-12),
            f"{sigma_label}.median differs",
        )
        _require(
            _number(sigma_row["a_min_scale"], f"{sigma_label}.a_min_scale")
            == AMPLITUDE_SCALE,
            f"{sigma_label}.a_min_scale differs",
        )
        value_float32, value_hex, value_sha = _float32(AMPLITUDE_SCALE * median)
        claimed_hex = sigma_row["a_min_float32_be_hex"]
        _require(
            type(claimed_hex) is str
            and _FLOAT32_HEX.fullmatch(claimed_hex) is not None
            and claimed_hex == value_hex,
            f"{sigma_label}.a_min float32 bytes differ",
        )
        _require(
            _sha(sigma_row["a_min_float32_le_sha256"], f"{sigma_label}.a_min SHA")
            == value_sha,
            f"{sigma_label}.a_min scalar SHA differs",
        )
        _require(
            value_float32 > MINIMUM_ADMITTED_FLOOR,
            f"{sigma_label}.a_min is not greater than {MINIMUM_ADMITTED_FLOOR}",
        )
        floor_index[sigma_index] = AmplitudeFloorV1(
            teacher_cell_id=teacher_cell_id,
            branch=str(branch),
            sigma_index=sigma_index,
            value_float32=value_float32,
            float32_be_hex=value_hex,
            float32_le_sha256=value_sha,
            bundle_digest=str(row["bundle_digest"]),
            calibration_id=calibration_id,
        )
    expected_fragment = {
        "teacher_cell_id": teacher_cell_id,
        "branch": branch,
        "calibrator_record_ids": [
            item.materialization_receipt["record_id"]
            for item in ordered_calibrators
        ],
        "calibrator_evidence_candidates": [
            dict(item.materialization_receipt["candidate_authority_evidence"])
            for item in ordered_calibrators
        ],
        "sigma_calibrations": row["sigma_calibrations"],
    }
    _require(
        materialization_fragment == expected_fragment,
        f"{label} materialization amplitude fragment differs",
    )
    identities = MappingProxyType(
        {
            "evidence_ids": evidence_ids,
            "review_ids": review_ids,
            "pair_ids": tuple(pair_ids),
            "output_shas": output_shas,
            "container_shas": container_shas,
            "calibrator_noise_shas": calibrator_noise_shas,
            "materialization_record_ids": tuple(
                item.materialization_receipt["record_id"]
                for item in calibrators
            ),
        }
    )
    return row, MappingProxyType(floor_index), identities


_PARENT_FIELDS = {
    "schema_version",
    "manifest_file_sha256",
    "manifest_digest",
    "validation_digest",
    "binding_digest",
}
_RUNTIME_FIELDS = {
    "schema_version",
    "bernini_revision",
    "veomni_revision",
    "official_checkpoint_tree_sha256",
    "transformer_config_sha256",
    "sigma_table_sha256",
    "psiout_protocol_sha256",
    "official_provider_source_sha256",
    "official_provider_abi",
    "compute_contract",
    "compute_contract_digest",
    "frame_count",
    "fps",
    "sampler_steps",
    "runtime_digest",
}
_COMPUTE_CONTRACT_FIELDS = {
    "schema_version",
    "model_eval",
    "torch_inference_mode",
    "official_frozen_native_only",
    "calibrator_peft_adapter_present",
    "frozen_effective_adapter_enabled",
    "frozen_effective_typed_patch_role_enabled",
    "base_compute_dtype",
    "autocast_dtype",
    "observer_output_dtype",
    "observer_output_stage",
    "observer_output_detached",
    "observer_output_contiguous",
    "same_state_counterfactual",
    "branch_and_noop_share_input_state",
    "world_size",
    "dp_size",
    "sp_size",
    "sp_order_contract",
    "all_rank_consensus",
}
_AUTHORITY_FIELDS = {
    "status",
    "calibration_complete",
    "current_optimizer_bundles",
    "current_calibrator_evidence",
    "current_frozen_fail_evidence",
    "optimizer_authorized",
}
_TOP_FIELDS = {
    "schema_version",
    "parent_authority",
    "materialization_run_receipt",
    "frozen_runtime_identity",
    "calibration_bundles",
    "authority_counts",
    "authority",
    "manifest_digest",
}


@dataclass(frozen=True)
class ValidatedAmplitudeAuthorityV1:
    manifest_file_sha256: str
    manifest_digest: str
    parent_manifest_file_sha256: str
    parent_manifest_digest: str
    frozen_runtime_digest: str
    floors: Mapping[tuple[str, str, int], AmplitudeFloorV1]
    validation_receipt: Mapping[str, Any]

    def resolve(self, teacher_cell_id: str, branch: str, sigma_index: int) -> AmplitudeFloorV1:
        key = (teacher_cell_id, branch, sigma_index)
        if key not in self.floors:
            fail("requested amplitude floor is not optimizer-admitted")
        return self.floors[key]


def _parent_runtime_indexes(
    parent_manifest: Mapping[str, Any],
) -> tuple[Mapping[tuple[str, str], Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    representations = {
        (str(row["teacher_cell_id"]), str(row["branch"])): row
        for row in parent_manifest["representation_admissions"]
        if row["analysis_split"] == "fit" and row["optimizer_admitted"] is True
    }
    source_by_iid = {str(row["source_iid"]): row for row in parent_manifest["sources"]}
    pairs: dict[str, Mapping[str, Any]] = {}
    for row in parent_manifest["pairs"]:
        if row["analysis_split"] != "fit" or row["optimizer_admitted"] is not True:
            continue
        augmented = dict(row)
        augmented["source_posterior_index0_sha256"] = source_by_iid[
            str(row["source_iid"])
        ]["source_posterior_index0_sha256"]
        pairs[str(row["pair_id"])] = MappingProxyType(augmented)
    _require(len(representations) == EXPECTED_BUNDLES, "parent fit representation closure differs")
    _require(len(pairs) == 128, "parent fit pair closure differs")
    return MappingProxyType(representations), MappingProxyType(pairs)


def _validate_parent_binding(
    value: Any,
    *,
    parent_manifest_file_sha256: str,
    parent_manifest: Mapping[str, Any],
    parent_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    row = _closed(value, _PARENT_FIELDS, "parent_authority")
    _verify_seal(row, "binding_digest", "parent_authority")
    _require(
        row["schema_version"] == PARENT_BINDING_SCHEMA_VERSION,
        "parent authority binding schema differs",
    )
    expected = {
        "manifest_file_sha256": parent_manifest_file_sha256,
        "manifest_digest": parent_manifest["manifest_digest"],
        "validation_digest": parent_receipt["validation_digest"],
    }
    for field, expected_value in expected.items():
        _require(row[field] == expected_value, f"parent_authority.{field} differs")
    return row


def _validate_runtime_identity(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _RUNTIME_FIELDS, "frozen_runtime_identity")
    _verify_seal(row, "runtime_digest", "frozen_runtime_identity")
    _require(
        row["schema_version"] == RUNTIME_IDENTITY_SCHEMA_VERSION,
        "frozen runtime identity schema differs",
    )
    _require(
        type(row["bernini_revision"]) is str
        and _REVISION.fullmatch(row["bernini_revision"]) is not None,
        "frozen runtime Bernini revision differs",
    )
    _require(
        type(row["veomni_revision"]) is str
        and _REVISION.fullmatch(row["veomni_revision"]) is not None,
        "frozen runtime VeOmni revision differs",
    )
    for field in (
        "official_checkpoint_tree_sha256",
        "transformer_config_sha256",
        "sigma_table_sha256",
        "psiout_protocol_sha256",
        "official_provider_source_sha256",
    ):
        _sha(row[field], f"frozen_runtime_identity.{field}")
    _safe_id(
        row["official_provider_abi"],
        "frozen_runtime_identity.official_provider_abi",
    )
    compute = _closed(
        row["compute_contract"],
        _COMPUTE_CONTRACT_FIELDS,
        "frozen_runtime_identity.compute_contract",
    )
    _require(
        compute["schema_version"] == COMPUTE_CONTRACT_SCHEMA_VERSION,
        "frozen compute contract schema differs",
    )
    expected_compute = {
        "model_eval": True,
        "torch_inference_mode": True,
        "official_frozen_native_only": True,
        "calibrator_peft_adapter_present": False,
        "frozen_effective_adapter_enabled": False,
        "frozen_effective_typed_patch_role_enabled": False,
        "base_compute_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "observer_output_dtype": "torch.float32",
        "observer_output_stage": "post-final-norm-proj-out-target-velocity",
        "observer_output_detached": True,
        "observer_output_contiguous": True,
        "same_state_counterfactual": True,
        "branch_and_noop_share_input_state": True,
        "world_size": 4,
        "dp_size": 1,
        "sp_size": 4,
        "sp_order_contract": "official-world4-sp4-rank-order-v1",
        "all_rank_consensus": True,
    }
    for field, expected in expected_compute.items():
        _require(
            compute[field] == expected,
            f"frozen compute contract {field} differs",
        )
    compute_digest = _sha(
        row["compute_contract_digest"],
        "frozen_runtime_identity.compute_contract_digest",
    )
    _require(
        compute_digest == object_sha256(compute),
        "frozen compute contract digest differs",
    )
    _require(row["frame_count"] == 81, "frozen runtime frame count differs")
    _require(_number(row["fps"], "frozen runtime fps") == 25.0, "frozen runtime fps differs")
    _require(row["sampler_steps"] == 40, "frozen runtime sampler steps differ")
    return row


def _load_validated(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    parent_manifest_path: str | Path,
    expected_parent_manifest_sha256: str,
) -> ValidatedAmplitudeAuthorityV1:
    parent_sha = _sha(
        expected_parent_manifest_sha256, label="parent manifest expected SHA-256"
    )
    parent_manifest = _load_json_file(
        parent_manifest_path, parent_sha, "parent authority manifest"
    )
    try:
        parent_receipt = parent_authority.validate_full30_action_authority_v1(parent_manifest)
    except Exception as error:
        raise Full30AmplitudeAuthorityError("parent action authority is not admitted") from error
    manifest_sha = _sha(expected_manifest_sha256, label="amplitude manifest expected SHA-256")
    manifest = _load_json_file(manifest_path, manifest_sha, "amplitude authority manifest")
    manifest = _closed(manifest, _TOP_FIELDS, "amplitude authority manifest")
    _verify_seal(manifest, "manifest_digest", "amplitude authority manifest")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "amplitude authority schema differs")
    try:
        materialization_run = parent_authority._load_materialization_run_v1(
            manifest["materialization_run_receipt"]
        )
    except Exception as error:
        raise Full30AmplitudeAuthorityError(
            "amplitude materialization run is not admitted"
        ) from error
    parent_binding = _validate_parent_binding(
        manifest["parent_authority"],
        parent_manifest_file_sha256=parent_sha,
        parent_manifest=parent_manifest,
        parent_receipt=parent_receipt,
    )
    runtime = _validate_runtime_identity(manifest["frozen_runtime_identity"])
    _require(
        runtime == materialization_run.receipt["runtime_identity"],
        "amplitude runtime differs from materialization run",
    )
    representations, pair_by_id = _parent_runtime_indexes(parent_manifest)
    bundles = manifest["calibration_bundles"]
    if type(bundles) is not list or len(bundles) != EXPECTED_BUNDLES:
        fail(f"calibration_bundles must contain exactly {EXPECTED_BUNDLES} rows")
    seen_keys: set[tuple[str, str]] = set()
    seen_calibration_ids: set[str] = set()
    seen_identities: dict[str, set[str]] = {
        "evidence_ids": set(),
        "review_ids": set(),
        "pair_ids": set(),
        "output_shas": set(),
        "container_shas": set(),
        "calibrator_noise_shas": set(),
        "materialization_record_ids": set(),
    }
    fragment_values = materialization_run.receipt[
        "amplitude_sigma_calibration_candidates"
    ]
    if type(fragment_values) is not list:
        fail("amplitude materialization fragment closure differs")
    materialization_fragments: dict[tuple[str, str], Mapping[str, Any]] = {}
    for fragment_ordinal, item in enumerate(fragment_values):
        fragment = _closed(
            item,
            parent_authority._MATERIALIZATION_AMPLITUDE_FRAGMENT_FIELDS,
            f"amplitude materialization fragments[{fragment_ordinal}]",
        )
        fragment_key = (str(fragment["teacher_cell_id"]), str(fragment["branch"]))
        _require(fragment_key not in materialization_fragments, "amplitude materialization fragment key is reused")
        materialization_fragments[fragment_key] = fragment
    floors: dict[tuple[str, str, int], AmplitudeFloorV1] = {}
    for ordinal, item in enumerate(bundles):
        candidate_key = (str(item.get("teacher_cell_id")), str(item.get("branch"))) if isinstance(item, Mapping) else ("", "")
        _require(candidate_key in materialization_fragments, f"calibration_bundles[{ordinal}] has no materialization fragment")
        bundle, bundle_floors, identities = _validate_bundle(
            item,
            representation_by_key=representations,
            pair_by_id=pair_by_id,
            run_authority=materialization_run,
            materialization_fragment=materialization_fragments[candidate_key],
            label=f"calibration_bundles[{ordinal}]",
        )
        key = (str(bundle["teacher_cell_id"]), str(bundle["branch"]))
        _require(key not in seen_keys, "amplitude cell/branch is duplicated")
        seen_keys.add(key)
        calibration_id = str(bundle["calibration_id"])
        _require(calibration_id not in seen_calibration_ids, "calibration id is duplicated")
        seen_calibration_ids.add(calibration_id)
        for identity_kind, values in identities.items():
            observed = set(values)
            _require(
                not (observed & seen_identities[identity_kind]),
                f"amplitude {identity_kind} are reused across bundles",
            )
            seen_identities[identity_kind].update(observed)
        for sigma_index, floor in bundle_floors.items():
            floor_key = (*key, sigma_index)
            _require(floor_key not in floors, "amplitude floor key is duplicated")
            floors[floor_key] = floor
    _require(seen_keys == set(representations), "amplitude cell/branch closure differs")
    _require(set(materialization_fragments) == set(representations), "amplitude materialization fragment key closure differs")
    run_calibrator_records = {
        record_id
        for record_id, receipt in materialization_run.record_receipts.items()
        if receipt["record_kind"] == "amplitude_calibrator"
    }
    _require(
        seen_identities["materialization_record_ids"] == run_calibrator_records,
        "amplitude materialization record closure has extra/missing records",
    )
    _require(len(floors) == EXPECTED_BUNDLES * len(SIGMA_INDICES), "amplitude floor closure differs")

    expected_counts = {
        "optimizer_bundles": EXPECTED_BUNDLES,
        "calibrator_evidence": EXPECTED_BUNDLES * CALIBRATORS_PER_BUNDLE,
        "frozen_fail_evidence": EXPECTED_BUNDLES * FAIL_CONTROLS_PER_BUNDLE,
        "sigma_floor_rows": EXPECTED_BUNDLES * len(SIGMA_INDICES),
    }
    _require(manifest["authority_counts"] == expected_counts, "amplitude authority counts differ")
    authority_row = _closed(manifest["authority"], _AUTHORITY_FIELDS, "authority")
    expected_authority = {
        "status": "optimizer_admitted",
        "calibration_complete": True,
        "current_optimizer_bundles": EXPECTED_BUNDLES,
        "current_calibrator_evidence": expected_counts["calibrator_evidence"],
        "current_frozen_fail_evidence": expected_counts["frozen_fail_evidence"],
        "optimizer_authorized": True,
    }
    _require(authority_row == expected_authority, "amplitude authority is incomplete")
    receipt_unsigned = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "manifest_file_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "parent_manifest_file_sha256": parent_sha,
        "parent_manifest_digest": parent_binding["manifest_digest"],
        "parent_validation_digest": parent_binding["validation_digest"],
        "frozen_runtime_digest": runtime["runtime_digest"],
        "materialization_run_digest": materialization_run.receipt["run_digest"],
        "materialization_run_receipt_sha256": materialization_run.binding[
            "file_sha256"
        ],
        "materialization_record_receipts": len(
            materialization_run.record_receipts
        ),
        "optimizer_bundles": EXPECTED_BUNDLES,
        "calibrator_evidence": expected_counts["calibrator_evidence"],
        "frozen_fail_evidence": expected_counts["frozen_fail_evidence"],
        "sigma_floor_rows": len(floors),
        "all_floors_greater_than_1e-6": True,
        "optimizer_authorized": True,
    }
    receipt = {
        **receipt_unsigned,
        "validation_digest": object_sha256(receipt_unsigned),
    }
    return ValidatedAmplitudeAuthorityV1(
        manifest_file_sha256=manifest_sha,
        manifest_digest=str(manifest["manifest_digest"]),
        parent_manifest_file_sha256=parent_sha,
        parent_manifest_digest=str(parent_binding["manifest_digest"]),
        frozen_runtime_digest=str(runtime["runtime_digest"]),
        floors=MappingProxyType(floors),
        validation_receipt=MappingProxyType(receipt),
    )


def load_amplitude_authority_v1(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    parent_manifest_path: str | Path,
    expected_parent_manifest_sha256: str,
) -> ValidatedAmplitudeAuthorityV1:
    return _load_validated(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        parent_manifest_path=parent_manifest_path,
        expected_parent_manifest_sha256=expected_parent_manifest_sha256,
    )


def validate_amplitude_manifest_file_v1(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    parent_manifest_path: str | Path,
    expected_parent_manifest_sha256: str,
) -> Mapping[str, Any]:
    return load_amplitude_authority_v1(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        parent_manifest_path=parent_manifest_path,
        expected_parent_manifest_sha256=expected_parent_manifest_sha256,
    ).validation_receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        receipt = validate_amplitude_manifest_file_v1(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_sha256,
            parent_manifest_path=args.parent_manifest,
            expected_parent_manifest_sha256=args.expected_parent_sha256,
        )
    except (Full30AmplitudeAuthorityError, OSError) as error:
        print(f"full30 amplitude authority rejected: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(dict(receipt)) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMPLITUDE_SCALE",
    "AmplitudeFloorV1",
    "BRANCHES",
    "CALIBRATORS_PER_BUNDLE",
    "CONTAINER_LAYOUT",
    "CONTAINER_MAGIC",
    "CONTAINER_MODE",
    "CONTAINER_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "FAIL_CONTROLS_PER_BUNDLE",
    "Full30AmplitudeAuthorityError",
    "MINIMUM_ADMITTED_FLOOR",
    "REVIEW_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SIGMA_INDICES",
    "TENSOR_DTYPE",
    "TENSOR_ELEMENTS",
    "TENSOR_SHAPE",
    "TENSOR_SLICE_BYTES",
    "ValidatedAmplitudeAuthorityV1",
    "calibrator_noise_seed_v1",
    "canonical_json_bytes",
    "file_sha256",
    "load_amplitude_authority_v1",
    "object_sha256",
    "seal_record",
    "validate_amplitude_manifest_file_v1",
]
