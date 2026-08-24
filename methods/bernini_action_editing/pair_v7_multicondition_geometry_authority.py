#!/usr/bin/env python3
"""Prospective, no-update authority for PAIR-v7 multicondition Phase-A.

This module deliberately stops before model execution.  It seals the four
fixed core4 action events and the two objectively selected active-schedule
medoids (16 and 35), while recording the already-observed schedule-33 result
as an excluded pilot.  It validates the bytes and tensor contents named by an
authored draft, but it does *not* validate CAST score receipts and therefore
cannot authorize a gradient measurement, optimizer, parameter mutation, or
action-editing success claim.

The implementation has no Torch dependency.  The narrow SafeTensors reader
checks a single detached finite F32 tensor and computes the same tensor digest
used by the existing PAIR-v7 fit-only authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Optional, Sequence


METHOD_NAME = "bernini-pair-v7-multicondition-preregistration-authority"
DRAFT_SCHEMA = "bernini-pair-v7-multicondition-authored-draft-v1"
EVENT_SCHEMA = "bernini-pair-v7-multicondition-bound-event-v1"
CELL_SCHEMA = "bernini-pair-v7-multicondition-primary-cell-v1"
PLAN_SCHEMA = "bernini-pair-v7-multicondition-preregistration-v1"
EVIDENCE_SKELETON_SCHEMA = (
    "bernini-pair-v7-multicondition-incomplete-evidence-skeleton-v1"
)

FRAME_COUNT = 81
FPS = 25.0
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PRIMARY_SCHEDULE_INDICES = (16, 35)
PILOT_SCHEDULE_INDEX = 33
SOURCE_NOISE_MASTER_SEED = 20260808
IDENTITY_FAMILIES = ("deploy_camera_delta", "deploy_noop_identity")
IDENTITY_SKETCHES_PER_FAMILY = 4
SCHEDULE_SELECTION_RULE = (
    "active_high_and_mid_stratum_exact_medians_fixed_before_primary_run"
)

EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256 = (
    "23509df54df4c64f5353cc405fc783793fee9789aab8aee8e181f678453a7215"
)
EXPECTED_PILOT_FILE_SHA256 = (
    "31ad34d69422772c9dee5524f93dc95bc7fd323dcabc199402098ac69b4c60ce"
)
EXPECTED_PILOT_RECEIPT_DIGEST = (
    "3712e90ce0f01fe458331b112c56a998074ac502dbf900872c0e21f607ef1672"
)

# These bindings describe the only four currently complete core4 events.  A
# caller supplies paths and repeats every digest in its draft; authoring then
# checks both the immutable constants below and the bytes at those paths.
FIXED_EVENTS: tuple[Mapping[str, Any], ...] = (
    {
        "event_id": "pair5-t2v-core4-v2-7b88a1ca1f804f41-action",
        "source_sample_id": "7b88a1ca1f804f41",
        "action_family": "dog-sit-facing-camera",
        "analysis_split": "fit",
        "pair_wave": "fit",
        "dp_arm": 0,
        "generation_seed": 2026080825,
        "latent_shape": [1, 16, 21, 60, 62],
        "source_video_file_sha256": (
            "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
        ),
        "clean_latent_file_sha256": (
            "50db7bbe40bf10fce6a3057064afe901a9bf15c49f5e336a4c0b63b8d53e47a6"
        ),
        "clean_latent_tensor_sha256": (
            "1b042adc261d2f153569cf04e54143a44245347d5c1319eff0ec23d913a198f2"
        ),
        "official_gaussian_file_sha256": (
            "d5e4af003ad23ca3e211c386a84933a5c4ddd3dcccce000356a094f626ba4b26"
        ),
        "official_gaussian_tensor_sha256": (
            "822c683b3500aa30e3a031f374fe9cb83fd42da3c7f229a65a61fbd8e63386b7"
        ),
    },
    {
        "event_id": "pair5-t2v-core4-v2-a35b590961d24694-action",
        "source_sample_id": "a35b590961d24694",
        "action_family": "human-rise-to-stand",
        "analysis_split": "fit",
        "pair_wave": "fit",
        "dp_arm": 1,
        "generation_seed": 2026080827,
        "latent_shape": [1, 16, 21, 64, 58],
        "source_video_file_sha256": (
            "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed"
        ),
        "clean_latent_file_sha256": (
            "a7c1c29098a7eb7e2d766764db99738ac3f4dcf2dc3d325727c92a0e5e98106d"
        ),
        "clean_latent_tensor_sha256": (
            "bf1cb7251f51ce349369d35b672a9d73c19f26e62818046079a572b670e14c0f"
        ),
        "official_gaussian_file_sha256": (
            "a2bd84db7d450c4a9bd7fe259805edd4ae3b3759f5e7e3d49989b08ab2a36bd0"
        ),
        "official_gaussian_tensor_sha256": (
            "0344813a946483e97c4d36d246ec0966f0efce87f077311086a3a2f66134a3a4"
        ),
    },
    {
        "event_id": "pair5-t2v-core4-v2-841b5e0080a1441d-action",
        "source_sample_id": "841b5e0080a1441d",
        "action_family": "dog-sit-facing-camera",
        "analysis_split": "confirmation",
        "pair_wave": "confirmation",
        "dp_arm": 0,
        "generation_seed": 2026080826,
        "latent_shape": [1, 16, 21, 60, 62],
        "source_video_file_sha256": (
            "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a"
        ),
        "clean_latent_file_sha256": (
            "e757e9b2d12ddc58f93d6d2570e0dabd321d9835e22d54187c27ba5c14fefaab"
        ),
        "clean_latent_tensor_sha256": (
            "0815f44b6b8faea304ad31f5085ed059fcc9d24b788d0841ced1621015600f05"
        ),
        "official_gaussian_file_sha256": (
            "2c49cdc438c234c48786aeb98fcc88ad0ff9c718b812039a0a171d0fcd8a8ec6"
        ),
        "official_gaussian_tensor_sha256": (
            "5d0e97de7e796b15ae1c9e7690aa598be406fd37026e8c0855f62604832c65f9"
        ),
    },
    {
        "event_id": "pair5-t2v-core4-v2-a66e6818e4144928-action",
        "source_sample_id": "a66e6818e4144928",
        "action_family": "human-rise-to-stand",
        "analysis_split": "confirmation",
        "pair_wave": "confirmation",
        "dp_arm": 1,
        "generation_seed": 2026080828,
        "latent_shape": [1, 16, 21, 68, 54],
        "source_video_file_sha256": (
            "0fdc54d89250f355d2170a4d6f6aac0867abf592afb849668a8e2879a6617147"
        ),
        "clean_latent_file_sha256": (
            "2ebb2b2b288f5d46e004a2451fe844cd0918cfcd9d313d92aca429864f609dcb"
        ),
        "clean_latent_tensor_sha256": (
            "801fd7622eab96d4b001b97f649abd84eca0e5270ea89cf08f355fd9ad54d64f"
        ),
        "official_gaussian_file_sha256": (
            "40e8d6c047eb81d20b8d18206f2f3a4a99883e70695d2b8afdd6ec1a2e80117c"
        ),
        "official_gaussian_tensor_sha256": (
            "9de3b5e86942cb65edc4b2c5d23566b40f0f00251837be7dd06d81bda3d412c0"
        ),
    },
)

SCHEDULES: Mapping[int, Mapping[str, Any]] = {
    16: {
        "schedule_index": 16,
        "timestep": 882,
        "sigma_float32_be_hex": "3f61ed37",
        "sigma": 0.8825258612632751,
        "gate_name": "high",
        "gate_weight": 1.0,
        "selection_rule": "exact_median_of_active_high_indices_0_through_32",
    },
    35: {
        "schedule_index": 35,
        "timestep": 418,
        "sigma_float32_be_hex": "3ed6539a",
        "sigma": 0.41860657930374146,
        "gate_name": "mid",
        "gate_weight": 0.5,
        "selection_rule": "exact_median_of_active_mid_indices_33_through_37",
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = 512 * 1024 * 1024

_NO_AUTHORITY = {
    "global_population_go": False,
    "optimizer_authorized": False,
    "parameter_update_authorized": False,
    "action_success_claimed": False,
    "scientific_action_editing_success_claim": False,
}

_DRAFT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_tree_sha256",
        "action_adapter_schema_sha256",
        "primary_schedule_indices",
        "source_noise_master_seed",
        "pilot_receipt_path",
        "pilot_receipt_file_sha256",
        "pilot_receipt_digest",
        "events",
        "cast_validation_performed",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "scientific_action_editing_success_claim",
    }
)
_DRAFT_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "source_sample_id",
        "action_family",
        "analysis_split",
        "pair_wave",
        "dp_arm",
        "generation_seed",
        "source_video_path",
        "source_video_file_sha256",
        "clean_latent_path",
        "clean_latent_file_sha256",
        "clean_latent_tensor_key",
        "clean_latent_tensor_sha256",
        "official_gaussian_path",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_key",
        "official_gaussian_tensor_sha256",
        "latent_shape",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "source_sample_id",
        "action_family",
        "analysis_split",
        "pair_wave",
        "dp_arm",
        "generation_seed",
        "source_video_path",
        "source_video_file_sha256",
        "clean_latent_path",
        "clean_latent_file_sha256",
        "clean_latent_tensor_key",
        "clean_latent_tensor_sha256",
        "official_gaussian_path",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_key",
        "official_gaussian_tensor_sha256",
        "latent_shape",
        "frame_count",
        "fps",
        "source_media_geometry_runtime_revalidation_required",
        "artifact_file_bytes_validated",
        "artifact_tensor_bytes_validated",
        "cast_validation_performed",
        "source_noise_key_sha256",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "scientific_action_editing_success_claim",
        "event_digest",
    }
)
_CELL_FIELDS = frozenset(
    {
        "schema_version",
        "condition_id",
        "pair_wave",
        "analysis_split",
        "schedule",
        "event_ids",
        "source_sample_ids",
        "action_families",
        "primary_gate_member",
        "prospective_after_observed_s33_pilot",
        "pilot_schedule_index_used",
        "exact81",
        "frame_count",
        "fps",
        "cell_digest",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "method_name",
        "authority_scope",
        "checkpoint_tree_sha256",
        "action_adapter_schema_sha256",
        "primary_schedule_indices",
        "schedule_selection_rule",
        "source_noise_contract",
        "pilot_exclusion",
        "event_count",
        "events",
        "primary_condition_count",
        "primary_cells",
        "global_common_direction_spec",
        "primary_gate_definition",
        "artifact_validation",
        "cast_validation_performed",
        "cast_score_receipts_consumed",
        "geometry_measurement_authorized",
        "create_only_authoring",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "scientific_action_editing_success_claim",
        "preregistration_digest",
    }
)


class PairV7MulticonditionAuthorityError(RuntimeError):
    """Raised when prospective multicondition closure is not exact."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare both JSON value and JSON type (for example, 1 != true)."""

    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError, UnicodeError):
        return False


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in value:
        raise PairV7MulticonditionAuthorityError(f"{field} is already present")
    return {**dict(value), field: object_sha256(value)}


def _check_seal(value: Mapping[str, Any], *, field: str, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        raise PairV7MulticonditionAuthorityError(f"{label} seal differs")
    return declared


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV7MulticonditionAuthorityError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairV7MulticonditionAuthorityError(f"{label} is unsafe")
    return value


def _closed(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PairV7MulticonditionAuthorityError(f"{label} field closure differs")
    return dict(value)


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii", errors="strict")
        value = json.loads(text)
    except Exception as error:
        raise PairV7MulticonditionAuthorityError(
            f"{label} must be strict ASCII JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise PairV7MulticonditionAuthorityError(f"{label} must be a JSON object")
    return value


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise PairV7MulticonditionAuthorityError(f"{label} path differs")
    raw = Path(value)
    if not raw.is_absolute():
        raise PairV7MulticonditionAuthorityError(f"{label} must be absolute")
    try:
        path = raw.resolve(strict=True)
    except OSError as error:
        raise PairV7MulticonditionAuthorityError(f"{label} is absent") from error
    if path != raw or not path.is_file() or path.is_symlink():
        raise PairV7MulticonditionAuthorityError(
            f"{label} must resolve to a canonical plain file"
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only_json(path_value: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise PairV7MulticonditionAuthorityError("output path must be absolute")
    if path.parent.resolve(strict=True) != path.parent or path.parent.is_symlink():
        raise PairV7MulticonditionAuthorityError(
            "output parent must be a canonical plain directory"
        )
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as error:
        raise PairV7MulticonditionAuthorityError("output must be create-only") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


class TensorInspection:
    def __init__(self, tensor_sha256: str, shape: Sequence[int]):
        self.tensor_sha256 = _sha256(tensor_sha256, label="tensor digest")
        self.shape = tuple(int(item) for item in shape)


def _tensor_digest(raw: bytes, shape: Sequence[int]) -> str:
    metadata = {
        "shape": [int(item) for item in shape],
        "dtype": "torch.float32",
        "layout": "torch.strided",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(bytes((0,)))
    digest.update(raw)
    return digest.hexdigest()


def _inspect_tensor_artifact(path: Path, key: str, *, label: str) -> TensorInspection:
    """Inspect one finite F32 SafeTensor without importing Torch."""

    if not isinstance(key, str) or not key or "\x00" in key:
        raise PairV7MulticonditionAuthorityError(f"{label} tensor key differs")
    size = path.stat().st_size
    if size < 10:
        raise PairV7MulticonditionAuthorityError(f"{label} SafeTensor is truncated")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        header_length = int.from_bytes(prefix, byteorder="little", signed=False)
        if not 2 <= header_length <= _MAX_HEADER_BYTES:
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor header length differs"
            )
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor header is truncated"
            )
        try:
            header = json.loads(header_raw.decode("utf-8", errors="strict"))
        except Exception as error:
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor header JSON differs"
            ) from error
        if not isinstance(header, Mapping):
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor header must be an object"
            )
        metadata = header.get("__metadata__")
        if metadata is not None and (
            not isinstance(metadata, Mapping)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
        ):
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor metadata differs"
            )
        tensor_keys = set(header) - {"__metadata__"}
        if tensor_keys != {key}:
            raise PairV7MulticonditionAuthorityError(
                f"{label} tensor-key closure differs"
            )
        descriptor = header[key]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise PairV7MulticonditionAuthorityError(
                f"{label} tensor descriptor differs"
            )
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            descriptor.get("dtype") != "F32"
            or not isinstance(shape, list)
            or len(shape) != 5
            or any(type(item) is not int or item <= 0 for item in shape)
            or shape[:3] != [1, LATENT_CHANNELS, LATENT_PHASES]
            or shape[3] % 2
            or shape[4] % 2
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
            or offsets[0] != 0
        ):
            raise PairV7MulticonditionAuthorityError(
                f"{label} must be exact81 finite F32 [1,16,21,H,W]"
            )
        element_count = math.prod(shape)
        tensor_bytes = element_count * 4
        if (
            tensor_bytes > _MAX_TENSOR_BYTES
            or offsets[1] != tensor_bytes
            or size != 8 + header_length + tensor_bytes
        ):
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor data closure differs"
            )
        raw = handle.read(tensor_bytes)
        if len(raw) != tensor_bytes or handle.read(1):
            raise PairV7MulticonditionAuthorityError(
                f"{label} SafeTensor payload differs"
            )
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", raw)):
        raise PairV7MulticonditionAuthorityError(f"{label} tensor is non-finite")
    return TensorInspection(_tensor_digest(raw, shape), shape)


def _require_no_authority(value: Mapping[str, Any], *, label: str) -> None:
    for field, expected in _NO_AUTHORITY.items():
        if value.get(field) is not expected:
            raise PairV7MulticonditionAuthorityError(
                f"{label}.{field} must remain false"
            )


def _pilot_receipt(
    path_value: Any,
    *,
    expected_file_sha256: Any,
    expected_receipt_digest: Any,
) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="s33 pilot receipt")
    file_digest = _sha256(expected_file_sha256, label="pilot file")
    receipt_digest = _sha256(expected_receipt_digest, label="pilot receipt")
    if (
        file_digest != EXPECTED_PILOT_FILE_SHA256
        or receipt_digest != EXPECTED_PILOT_RECEIPT_DIGEST
        or _file_sha256(path) != file_digest
    ):
        raise PairV7MulticonditionAuthorityError("fixed s33 pilot bytes differ")
    receipt = dict(_strict_json(path, label="s33 pilot receipt"))
    if _check_seal(receipt, field="receipt_digest", label="s33 pilot receipt") != receipt_digest:
        raise PairV7MulticonditionAuthorityError("fixed s33 pilot digest differs")
    schedule = receipt.get("schedule_policy")
    manifest = receipt.get("action_manifest")
    fit_ids = [
        str(row["event_id"])
        for row in FIXED_EVENTS
        if row["analysis_split"] == "fit"
    ]
    if (
        receipt.get("schema_version")
        != "bernini-pair-v7-phase-a-geometry-audit-v3"
        or receipt.get("audit_complete") is not True
        or receipt.get("geometry_audit_passed") is not True
        or not isinstance(schedule, Mapping)
        or schedule.get("schedule_index") != PILOT_SCHEDULE_INDEX
        or not isinstance(manifest, Mapping)
        or manifest.get("candidate_ids") != fit_ids
        or receipt.get("parameter_mutation_performed") is not False
        or receipt.get("optimizer_constructed") is not False
        or receipt.get("optimizer_step_called") is not False
        or receipt.get("parameter_add_called") is not False
    ):
        raise PairV7MulticonditionAuthorityError("s33 pilot semantic closure differs")
    _require_no_authority(receipt, label="s33 pilot receipt")
    return {
        "path": str(path),
        "file_sha256": file_digest,
        "receipt_digest": receipt_digest,
        "schema_version": receipt["schema_version"],
        "schedule_index": PILOT_SCHEDULE_INDEX,
        "geometry_audit_passed": True,
    }


def _source_noise_key(sample_id: str) -> str:
    material = (
        f"{SOURCE_NOISE_MASTER_SEED}\x00{sample_id}\x00"
        "pair-v7-multicondition-source-native"
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _bound_event(
    draft_value: Any, expected: Mapping[str, Any], *, ordinal: int
) -> Mapping[str, Any]:
    draft = _closed(
        draft_value, _DRAFT_EVENT_FIELDS, label=f"event draft[{ordinal}]"
    )
    scalar_fields = (
        "event_id",
        "source_sample_id",
        "action_family",
        "analysis_split",
        "pair_wave",
        "dp_arm",
        "generation_seed",
        "latent_shape",
        "source_video_file_sha256",
        "clean_latent_file_sha256",
        "clean_latent_tensor_sha256",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_sha256",
    )
    if any(
        not _same_json_value(draft.get(field), expected.get(field))
        for field in scalar_fields
    ):
        raise PairV7MulticonditionAuthorityError(
            f"event draft[{ordinal}] fixed core4 binding differs"
        )
    _safe_id(draft["event_id"], label="event ID")
    _safe_id(draft["source_sample_id"], label="source sample ID")
    _safe_id(draft["action_family"], label="action family")
    if draft["analysis_split"] not in {"fit", "confirmation"}:
        raise PairV7MulticonditionAuthorityError("analysis split differs")
    if draft["pair_wave"] != draft["analysis_split"] or draft["dp_arm"] not in {0, 1}:
        raise PairV7MulticonditionAuthorityError("DP2 pair-wave closure differs")

    source = _plain_absolute_file(draft["source_video_path"], label="source video")
    clean = _plain_absolute_file(draft["clean_latent_path"], label="clean latent")
    gaussian = _plain_absolute_file(
        draft["official_gaussian_path"], label="official Gaussian"
    )
    for path, field, label in (
        (source, "source_video_file_sha256", "source video"),
        (clean, "clean_latent_file_sha256", "clean latent"),
        (gaussian, "official_gaussian_file_sha256", "official Gaussian"),
    ):
        declared = _sha256(draft[field], label=f"{label} file")
        if _file_sha256(path) != declared:
            raise PairV7MulticonditionAuthorityError(f"{label} file bytes differ")
    clean_inspection = _inspect_tensor_artifact(
        clean, draft["clean_latent_tensor_key"], label="clean latent"
    )
    gaussian_inspection = _inspect_tensor_artifact(
        gaussian,
        draft["official_gaussian_tensor_key"],
        label="official Gaussian",
    )
    if (
        list(clean_inspection.shape) != draft["latent_shape"]
        or gaussian_inspection.shape != clean_inspection.shape
        or clean_inspection.tensor_sha256
        != draft["clean_latent_tensor_sha256"]
        or gaussian_inspection.tensor_sha256
        != draft["official_gaussian_tensor_sha256"]
    ):
        raise PairV7MulticonditionAuthorityError(
            f"event draft[{ordinal}] tensor content binding differs"
        )
    unsigned = {
        "schema_version": EVENT_SCHEMA,
        "event_id": draft["event_id"],
        "source_sample_id": draft["source_sample_id"],
        "action_family": draft["action_family"],
        "analysis_split": draft["analysis_split"],
        "pair_wave": draft["pair_wave"],
        "dp_arm": draft["dp_arm"],
        "generation_seed": draft["generation_seed"],
        "source_video_path": str(source),
        "source_video_file_sha256": draft["source_video_file_sha256"],
        "clean_latent_path": str(clean),
        "clean_latent_file_sha256": draft["clean_latent_file_sha256"],
        "clean_latent_tensor_key": draft["clean_latent_tensor_key"],
        "clean_latent_tensor_sha256": draft["clean_latent_tensor_sha256"],
        "official_gaussian_path": str(gaussian),
        "official_gaussian_file_sha256": draft[
            "official_gaussian_file_sha256"
        ],
        "official_gaussian_tensor_key": draft["official_gaussian_tensor_key"],
        "official_gaussian_tensor_sha256": draft[
            "official_gaussian_tensor_sha256"
        ],
        "latent_shape": list(draft["latent_shape"]),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "source_media_geometry_runtime_revalidation_required": True,
        "artifact_file_bytes_validated": True,
        "artifact_tensor_bytes_validated": True,
        "cast_validation_performed": False,
        "source_noise_key_sha256": _source_noise_key(draft["source_sample_id"]),
        **_NO_AUTHORITY,
    }
    return _seal(unsigned, field="event_digest")


def _schedule(index: int) -> Mapping[str, Any]:
    if index not in SCHEDULES:
        raise PairV7MulticonditionAuthorityError("primary schedule cell differs")
    return dict(SCHEDULES[index])


def _cells(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for pair_wave in ("fit", "confirmation"):
        pair = [row for row in events if row["pair_wave"] == pair_wave]
        pair.sort(key=lambda row: int(row["dp_arm"]))
        if len(pair) != 2 or [row["dp_arm"] for row in pair] != [0, 1]:
            raise PairV7MulticonditionAuthorityError(
                f"{pair_wave} exact DP2 event closure differs"
            )
        if len({row["action_family"] for row in pair}) != 2:
            raise PairV7MulticonditionAuthorityError(
                f"{pair_wave} requires two action families"
            )
        for schedule_index in PRIMARY_SCHEDULE_INDICES:
            unsigned = {
                "schema_version": CELL_SCHEMA,
                "condition_id": f"{pair_wave}-s{schedule_index}",
                "pair_wave": pair_wave,
                "analysis_split": pair_wave,
                "schedule": _schedule(schedule_index),
                "event_ids": [row["event_id"] for row in pair],
                "source_sample_ids": [row["source_sample_id"] for row in pair],
                "action_families": [row["action_family"] for row in pair],
                "primary_gate_member": True,
                "prospective_after_observed_s33_pilot": True,
                "pilot_schedule_index_used": False,
                "exact81": True,
                "frame_count": FRAME_COUNT,
                "fps": FPS,
            }
            rows.append(_seal(unsigned, field="cell_digest"))
    return rows


def _pilot_exclusion(pilot: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        **dict(pilot),
        "observed_before_this_preregistration": True,
        "included_in_primary_gate": False,
        "included_in_primary_condition_ids": [],
        "raw_gradient_or_safe_direction_reused": False,
        "role": "excluded_prior_pilot_boundary_only",
    }


def _source_noise_contract(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "master_seed": SOURCE_NOISE_MASTER_SEED,
        "derivation": (
            "sha256(master_seed_NUL_source_sample_id_NUL_"
            "pair-v7-multicondition-source-native)"
        ),
        "arm_index_only_derivation_allowed": False,
        "same_source_epsilon_reused_across_primary_schedules": True,
        "different_source_sample_ids_have_distinct_keys": True,
        "source_key_sha256_by_sample": {
            str(row["source_sample_id"]): str(row["source_noise_key_sha256"])
            for row in events
        },
    }


def _global_common_direction_spec() -> Mapping[str, Any]:
    return {
        "primary_cell_count": 4,
        "action_component_count": 8,
        "identity_probe_count": 64,
        "source_count": 4,
        "schedule_count": 2,
        "identity_family_count": len(IDENTITY_FAMILIES),
        "identity_sketches_per_source_schedule_family": (
            IDENTITY_SKETCHES_PER_FAMILY
        ),
        "action_aggregation": (
            "arithmetic_mean_of_all_8_preregistered_action_components_"
            "after_each_SP4_average"
        ),
        "identity_aggregation": "union_all_64_unprojected_identity_rows_once",
        "root_solver": "single_world_rank0_cpu_fp64_geometry",
        "local_project_then_average": False,
        "every_action_component_requires_positive_descent": True,
        "raw_gradient_persistence_authorized": False,
    }


def _primary_gate_definition() -> Mapping[str, Any]:
    return {
        "all_four_primary_cells_must_pass": True,
        "global_common_direction_union_must_pass": True,
        "candidate_or_seed_selection_allowed": False,
        "failed_cell_substitution_allowed": False,
        "minimum_per_source_schedule_family_effective_rank": 3,
        "minimum_per_cell_identity_global_effective_rank": 8,
        "minimum_multicondition_identity_global_effective_rank": 16,
        "minimum_action_gradient_survival": 0.05,
        "minimum_each_action_component_descent_cosine": 0.05,
        "maximum_effective_condition_number": 100000.0,
        "maximum_identity_cosine": 0.00002,
        "maximum_identity_relative_dot": 0.00002,
        "maximum_probe_reconstruction_residual": 0.000001,
        "parameter_update_authorized_even_if_passed": False,
    }


def _artifact_validation(draft_path: Path) -> Mapping[str, Any]:
    return {
        "draft_path": str(draft_path),
        "draft_file_sha256": _file_sha256(draft_path),
        "all_source_file_hashes_verified": True,
        "all_latent_file_hashes_verified": True,
        "all_gaussian_file_hashes_verified": True,
        "all_latent_tensor_hashes_verified": True,
        "all_gaussian_tensor_hashes_verified": True,
        "source_media_exact81_runtime_revalidation_required": True,
    }


def author_preregistration(
    *, draft_path: str | Path, output_path: str | Path
) -> Mapping[str, Any]:
    """Validate one authored draft and create the prospective plan once."""

    draft_file = _plain_absolute_file(draft_path, label="authored draft")
    draft = _closed(
        _strict_json(draft_file, label="authored draft"),
        _DRAFT_FIELDS,
        label="authored draft",
    )
    if (
        draft.get("schema_version") != DRAFT_SCHEMA
        or draft.get("checkpoint_tree_sha256")
        != EXPECTED_CHECKPOINT_TREE_SHA256
        or draft.get("action_adapter_schema_sha256")
        != EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256
        or not _same_json_value(
            draft.get("primary_schedule_indices"),
            list(PRIMARY_SCHEDULE_INDICES),
        )
        or not _same_json_value(
            draft.get("source_noise_master_seed"), SOURCE_NOISE_MASTER_SEED
        )
        or draft.get("cast_validation_performed") is not False
    ):
        raise PairV7MulticonditionAuthorityError(
            "authored draft prospective constants differ"
        )
    _require_no_authority(draft, label="authored draft")
    event_drafts = draft.get("events")
    if not isinstance(event_drafts, list) or len(event_drafts) != len(FIXED_EVENTS):
        raise PairV7MulticonditionAuthorityError(
            "authored draft requires exactly four fixed events"
        )
    events = [
        _bound_event(raw, expected, ordinal=ordinal)
        for ordinal, (raw, expected) in enumerate(zip(event_drafts, FIXED_EVENTS))
    ]
    if (
        len({row["event_id"] for row in events}) != 4
        or len({row["source_sample_id"] for row in events}) != 4
        or [row["analysis_split"] for row in events]
        != ["fit", "fit", "confirmation", "confirmation"]
    ):
        raise PairV7MulticonditionAuthorityError("fixed four-event closure differs")
    pilot = _pilot_receipt(
        draft["pilot_receipt_path"],
        expected_file_sha256=draft["pilot_receipt_file_sha256"],
        expected_receipt_digest=draft["pilot_receipt_digest"],
    )
    cells = _cells(events)
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "method_name": METHOD_NAME,
        "authority_scope": "prospective_preregistration_only_no_runtime_authority",
        "checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREE_SHA256,
        "action_adapter_schema_sha256": EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256,
        "primary_schedule_indices": list(PRIMARY_SCHEDULE_INDICES),
        "schedule_selection_rule": SCHEDULE_SELECTION_RULE,
        "source_noise_contract": _source_noise_contract(events),
        "pilot_exclusion": _pilot_exclusion(pilot),
        "event_count": 4,
        "events": events,
        "primary_condition_count": 4,
        "primary_cells": cells,
        "global_common_direction_spec": _global_common_direction_spec(),
        "primary_gate_definition": _primary_gate_definition(),
        "artifact_validation": _artifact_validation(draft_file),
        "cast_validation_performed": False,
        "cast_score_receipts_consumed": False,
        "geometry_measurement_authorized": False,
        "create_only_authoring": True,
        **_NO_AUTHORITY,
    }
    plan = _seal(unsigned, field="preregistration_digest")
    _write_create_only_json(output_path, plan)
    return plan


def _validate_event_row(raw: Any, expected: Mapping[str, Any], *, ordinal: int) -> None:
    row = _closed(raw, _EVENT_FIELDS, label=f"bound event[{ordinal}]")
    _check_seal(row, field="event_digest", label=f"bound event[{ordinal}]")
    _require_no_authority(row, label=f"bound event[{ordinal}]")
    for field in (
        "event_id",
        "source_sample_id",
        "action_family",
        "analysis_split",
        "pair_wave",
        "dp_arm",
        "generation_seed",
        "latent_shape",
        "source_video_file_sha256",
        "clean_latent_file_sha256",
        "clean_latent_tensor_sha256",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_sha256",
    ):
        if not _same_json_value(row.get(field), expected.get(field)):
            raise PairV7MulticonditionAuthorityError(
                f"bound event[{ordinal}] fixed core4 field differs: {field}"
            )
    if (
        row.get("schema_version") != EVENT_SCHEMA
        or not _same_json_value(row.get("frame_count"), FRAME_COUNT)
        or not _same_json_value(row.get("fps"), FPS)
        or row.get("source_media_geometry_runtime_revalidation_required") is not True
        or row.get("artifact_file_bytes_validated") is not True
        or row.get("artifact_tensor_bytes_validated") is not True
        or row.get("cast_validation_performed") is not False
        or row.get("source_noise_key_sha256")
        != _source_noise_key(str(row["source_sample_id"]))
    ):
        raise PairV7MulticonditionAuthorityError(
            f"bound event[{ordinal}] preregistration semantics differ"
        )
    source = _plain_absolute_file(row["source_video_path"], label="source video")
    clean = _plain_absolute_file(row["clean_latent_path"], label="clean latent")
    gaussian = _plain_absolute_file(
        row["official_gaussian_path"], label="official Gaussian"
    )
    for path, field, label in (
        (source, "source_video_file_sha256", "source video"),
        (clean, "clean_latent_file_sha256", "clean latent"),
        (gaussian, "official_gaussian_file_sha256", "official Gaussian"),
    ):
        if _file_sha256(path) != row[field]:
            raise PairV7MulticonditionAuthorityError(f"bound {label} changed")
    clean_tensor = _inspect_tensor_artifact(
        clean, row["clean_latent_tensor_key"], label="clean latent"
    )
    gaussian_tensor = _inspect_tensor_artifact(
        gaussian,
        row["official_gaussian_tensor_key"],
        label="official Gaussian",
    )
    if (
        clean_tensor.tensor_sha256 != row["clean_latent_tensor_sha256"]
        or gaussian_tensor.tensor_sha256
        != row["official_gaussian_tensor_sha256"]
        or list(clean_tensor.shape) != row["latent_shape"]
        or gaussian_tensor.shape != clean_tensor.shape
    ):
        raise PairV7MulticonditionAuthorityError("bound tensor changed")


def validate_preregistration(
    *, plan_path: str | Path, expected_plan_file_sha256: str
) -> Mapping[str, Any]:
    """Revalidate seals, semantics, and every bound external artifact."""

    path = _plain_absolute_file(plan_path, label="preregistration plan")
    expected_file = _sha256(expected_plan_file_sha256, label="plan file")
    if _file_sha256(path) != expected_file:
        raise PairV7MulticonditionAuthorityError("preregistration file bytes differ")
    plan = _closed(
        _strict_json(path, label="preregistration plan"),
        _PLAN_FIELDS,
        label="preregistration plan",
    )
    _check_seal(plan, field="preregistration_digest", label="preregistration plan")
    _require_no_authority(plan, label="preregistration plan")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("method_name") != METHOD_NAME
        or plan.get("authority_scope")
        != "prospective_preregistration_only_no_runtime_authority"
        or plan.get("checkpoint_tree_sha256")
        != EXPECTED_CHECKPOINT_TREE_SHA256
        or plan.get("action_adapter_schema_sha256")
        != EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256
        or not _same_json_value(
            plan.get("primary_schedule_indices"),
            list(PRIMARY_SCHEDULE_INDICES),
        )
        or plan.get("schedule_selection_rule") != SCHEDULE_SELECTION_RULE
        or not _same_json_value(plan.get("event_count"), 4)
        or not _same_json_value(plan.get("primary_condition_count"), 4)
        or plan.get("cast_validation_performed") is not False
        or plan.get("cast_score_receipts_consumed") is not False
        or plan.get("geometry_measurement_authorized") is not False
        or plan.get("create_only_authoring") is not True
    ):
        raise PairV7MulticonditionAuthorityError(
            "preregistration authority boundary differs"
        )
    events = plan.get("events")
    if not isinstance(events, list) or len(events) != len(FIXED_EVENTS):
        raise PairV7MulticonditionAuthorityError("preregistered event closure differs")
    for ordinal, (row, expected) in enumerate(zip(events, FIXED_EVENTS)):
        _validate_event_row(row, expected, ordinal=ordinal)
    expected_noise = _source_noise_contract(events)
    if not _same_json_value(plan.get("source_noise_contract"), expected_noise):
        raise PairV7MulticonditionAuthorityError("source-noise seed/derivation differs")
    expected_cells = _cells(events)
    cells = plan.get("primary_cells")
    if not _same_json_value(cells, expected_cells):
        raise PairV7MulticonditionAuthorityError("primary cell closure differs")
    for ordinal, cell in enumerate(cells):
        row = _closed(cell, _CELL_FIELDS, label=f"primary cell[{ordinal}]")
        _check_seal(row, field="cell_digest", label=f"primary cell[{ordinal}]")
    if (
        not _same_json_value(
            plan.get("global_common_direction_spec"),
            _global_common_direction_spec(),
        )
        or not _same_json_value(
            plan.get("primary_gate_definition"), _primary_gate_definition()
        )
    ):
        raise PairV7MulticonditionAuthorityError("primary multicondition gate differs")
    artifact = plan.get("artifact_validation")
    if not isinstance(artifact, Mapping):
        raise PairV7MulticonditionAuthorityError("artifact validation is absent")
    draft_file = _plain_absolute_file(
        artifact.get("draft_path"), label="bound authored draft"
    )
    if not _same_json_value(artifact, _artifact_validation(draft_file)):
        raise PairV7MulticonditionAuthorityError(
            "artifact validation closure differs"
        )
    pilot = plan.get("pilot_exclusion")
    if not isinstance(pilot, Mapping):
        raise PairV7MulticonditionAuthorityError("pilot exclusion is absent")
    if (
        pilot.get("included_in_primary_gate") is not False
        or pilot.get("included_in_primary_condition_ids") != []
        or pilot.get("raw_gradient_or_safe_direction_reused") is not False
        or pilot.get("schedule_index") != PILOT_SCHEDULE_INDEX
        or pilot.get("role") != "excluded_prior_pilot_boundary_only"
    ):
        raise PairV7MulticonditionAuthorityError("pilot leakage into primary gate")
    revalidated_pilot = _pilot_receipt(
        pilot.get("path"),
        expected_file_sha256=pilot.get("file_sha256"),
        expected_receipt_digest=pilot.get("receipt_digest"),
    )
    if any(pilot.get(key) != value for key, value in revalidated_pilot.items()):
        raise PairV7MulticonditionAuthorityError("pilot binding changed")
    if not _same_json_value(pilot, _pilot_exclusion(revalidated_pilot)):
        raise PairV7MulticonditionAuthorityError("pilot exclusion closure differs")
    return plan


def author_evidence_skeleton(
    *,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Create an explicitly incomplete evidence skeleton, never runtime GO."""

    plan = validate_preregistration(
        plan_path=plan_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
    )
    path = _plain_absolute_file(plan_path, label="preregistration plan")
    unsigned = {
        "schema_version": EVIDENCE_SKELETON_SCHEMA,
        "method_name": METHOD_NAME,
        "status": "incomplete_missing_cast_and_runtime_validation",
        "preregistration_path": str(path),
        "preregistration_file_sha256": _file_sha256(path),
        "preregistration_digest": plan["preregistration_digest"],
        "missing_required_steps": [
            "CAST-v4 root/group/40-child receipt validation for all four events",
            "immutable runtime source archive validation",
            "WORLD8 DP2xSP4 execution and consensus",
            "postflight external-artifact rehash",
        ],
        "cast_validation_performed": False,
        "cast_score_receipts_consumed": False,
        "geometry_measurement_authorized": False,
        "runtime_launch_authorized": False,
        "create_only_authoring": True,
        **_NO_AUTHORITY,
    }
    skeleton = _seal(unsigned, field="evidence_skeleton_digest")
    _write_create_only_json(output_path, skeleton)
    return skeleton


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    author = subparsers.add_parser("author-preregistration")
    author.add_argument("--draft", required=True)
    author.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-preregistration")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-file-sha256", required=True)
    skeleton = subparsers.add_parser("author-evidence-skeleton")
    skeleton.add_argument("--plan", required=True)
    skeleton.add_argument("--expected-plan-file-sha256", required=True)
    skeleton.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "author-preregistration":
        value = author_preregistration(
            draft_path=args.draft, output_path=args.output
        )
        output = Path(args.output)
        result = {
            "preregistration": str(output),
            "preregistration_file_sha256": _file_sha256(output),
            "preregistration_digest": value["preregistration_digest"],
            "geometry_measurement_authorized": False,
            **_NO_AUTHORITY,
        }
    elif args.command == "validate-preregistration":
        value = validate_preregistration(
            plan_path=args.plan,
            expected_plan_file_sha256=args.expected_plan_file_sha256,
        )
        result = {
            "preregistration_validated": True,
            "preregistration_digest": value["preregistration_digest"],
            "geometry_measurement_authorized": False,
            **_NO_AUTHORITY,
        }
    else:
        value = author_evidence_skeleton(
            plan_path=args.plan,
            expected_plan_file_sha256=args.expected_plan_file_sha256,
            output_path=args.output,
        )
        output = Path(args.output)
        result = {
            "evidence_skeleton": str(output),
            "evidence_skeleton_file_sha256": _file_sha256(output),
            "evidence_skeleton_digest": value["evidence_skeleton_digest"],
            "geometry_measurement_authorized": False,
            **_NO_AUTHORITY,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DRAFT_SCHEMA",
    "EVIDENCE_SKELETON_SCHEMA",
    "EXPECTED_ACTION_ADAPTER_SCHEMA_SHA256",
    "EXPECTED_CHECKPOINT_TREE_SHA256",
    "FIXED_EVENTS",
    "METHOD_NAME",
    "PAIR_V7_MULTICONDITION_AUTHORITY_ERROR",
    "PILOT_SCHEDULE_INDEX",
    "PLAN_SCHEMA",
    "PRIMARY_SCHEDULE_INDICES",
    "SOURCE_NOISE_MASTER_SEED",
    "PairV7MulticonditionAuthorityError",
    "TensorInspection",
    "author_evidence_skeleton",
    "author_preregistration",
    "canonical_json_bytes",
    "object_sha256",
    "validate_preregistration",
]

# Backward-free symbolic export for static consumers; it is intentionally an
# alias, not a second exception type.
PAIR_V7_MULTICONDITION_AUTHORITY_ERROR = PairV7MulticonditionAuthorityError
