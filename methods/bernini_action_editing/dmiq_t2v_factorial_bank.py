#!/usr/bin/env python3
"""Preregister, render, and independently audit the DMIQ pure-T2V bank.

The scientific design is a four-level L16 orthogonal array over actor, scene,
camera, and wording.  Every OA row is crossed with two fixed seed replicates,
giving 32 matched proposal cells.  A cell contains one positive and the nine
FITQ semantic negatives, all rendered with the same native Gaussian and on one
SP4 group.  Actor level four is a frozen confirmation holdout (24 discovery
and 8 confirmation cells); no split is inferred from rendered quality.

Renderer completion is provenance evidence only.  The legacy v2 event-audit
validator can check rubric rows and report counts, but its plain JSON digest
cannot prove assessor identity, split-release chronology, or same-state causal
queries.  It is therefore permanently FITQ-ineligible.  Scientific use needs
an external discovery-signature / confirmation-release-signature /
confirmation-signature chain plus a state-owner by prompt cross-query receipt.
Failed attempts are retained and individual winners are never selected.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from string import Formatter
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


SPEC_SCHEMA = "bernini-dmiq-pure-t2v-factorial-spec-v2"
MANIFEST_SCHEMA = "bernini-dmiq-pure-t2v-factorial-manifest-v2"
BANK_RECEIPT_SCHEMA = "bernini-dmiq-pure-t2v-factorial-bank-receipt-v2"
EVENT_AUDIT_SCHEMA = "bernini-dmiq-pure-t2v-event-audit-v2"
EVENT_AUDIT_RECEIPT_SCHEMA = "bernini-dmiq-pure-t2v-event-audit-receipt-v2"
NATIVE_RECEIPT_SCHEMA = "bernini-native-identity-generation-canary-v1"

BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)

FRAME_COUNT = 81
LATENT_SHAPE = (1, 16, 21, 62, 60)
VIDEO_HEIGHT = 496
VIDEO_WIDTH = 480
FPS = 25
NUM_INFERENCE_STEPS = 40
ULYSSES_SIZE = 4
GROUPS = ("sp4-a", "sp4-b")
GROUP_VISIBLE_DEVICE_SLOTS = {
    "sp4-a": (0, 1, 2, 3),
    "sp4-b": (4, 5, 6, 7),
}
PROFILES = ("engineering_micro", "scientific")
AXIS_SPLITS = ("discovery", "confirmation")
MICRO_DESIGN = "engineering_cartesian_tiny_v2"
SCIENTIFIC_DESIGN = "oa_l16_4level_actor_scene_camera_wording_2rep_v2"
BRANCH_ORDER = (
    "full_action",
    "noop",
    "incomplete_action",
    "reverse_action",
    "shuffled_action",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
NEGATIVE_BRANCH_ORDER = BRANCH_ORDER[1:]

MIN_VERIFIED_DISCOVERY_POSITIVES = 8
MIN_VERIFIED_CONFIRMATION_POSITIVES = 4
GENERIC_WRONG_MOTION_ATOMIC_EVENT_COUNT = 4
BLINDED_FIELDS = (
    "analysis_split",
    "seed",
    "seed_replicate_id",
    "attempt_rung",
    "execution_group",
    "fitq_features",
    "fitq_scores",
    "training_outputs",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")

_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "bank_id",
        "profile",
        "design",
        "action_family_id",
        "action_program",
        "source_geometry_video",
        "object_pair",
        "actor_levels",
        "scene_levels",
        "camera_levels",
        "wording_levels",
        "seed_replicates",
        "spatial_sketch",
    }
)
_SOURCE_FIELDS = frozenset({"sha256", "frame_count", "fps", "height", "width"})
_ACTION_PROGRAM_FIELDS = frozenset(
    {
        "actor_role",
        "patient_role",
        "preconditions",
        "ordered_milestones",
        "terminal_hold_required",
        "terminal_hold_video_frames",
        "reverse_action_definition",
    }
)
_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "object_phrase",
        "object_reference",
        "distractor_object_phrase",
        "distractor_object_reference",
    }
)
_ACTOR_FIELDS = frozenset(
    {
        "actor_id",
        "split",
        "actor_phrase",
        "actor_reference",
        "distractor_actor_phrase",
        "distractor_actor_reference",
    }
)
_SCENE_FIELDS = frozenset({"scene_id", "scene_phrase"})
_CAMERA_FIELDS = frozenset({"camera_id", "camera_phrase"})
_WORDING_FIELDS = frozenset({"wording_id", "templates"})
_SEED_REPLICATE_FIELDS = frozenset(
    {"replicate_id", "initial_seed", "topup_seeds"}
)
_SPATIAL_SKETCH_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "sketch_id",
        "construction_id",
        "construction_digest",
        "seed",
        "matrix_shape",
        "patch_height",
        "patch_width",
        "flatten_order",
        "value_dtype",
        "value_encoding",
        "normalization",
        "signed",
        "data_dependent",
        "matrix_value_sha256",
        "matrix_raw_bytes_sha256",
        "matrix_value_digest_scheme",
        "verified_exact_row_rank",
    }
)
_PROMPT_PLACEHOLDERS = frozenset(
    {
        "actor",
        "actor_ref",
        "distractor_actor",
        "distractor_actor_ref",
        "object",
        "object_ref",
        "distractor_object",
        "distractor_object_ref",
        "scene",
        "camera",
    }
)
_EVENT_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "bank_manifest_digest",
        "bank_receipt_digest",
        "assessor_contract",
        "split_isolation",
        "rows",
        "audit_digest",
    }
)
_ASSESSOR_FIELDS = frozenset(
    {
        "assessor_id",
        "organization_id",
        "independent_of_renderer_and_method",
        "no_fitq_or_training_outputs_seen",
        "proposal_videos_are_only_model_outputs_seen",
        "blinded_fields",
        "attestation",
    }
)
_SPLIT_ISOLATION_FIELDS = frozenset(
    {
        "discovery_decisions_frozen_before_confirmation_opened",
        "discovery_decisions_sha256",
        "confirmation_opened_after_freeze",
        "confirmation_rows_never_used_for_prompt_seed_or_design_selection",
        "row_order_matches_manifest",
    }
)
_EVENT_ROW_FIELDS = frozenset(
    {
        "entry_id",
        "video_sha256",
        "semantic_branch",
        "analysis_split",
        "initial_preconditions_realized",
        "ordered_milestone_realized",
        "ordered_milestone_first_frame",
        "terminal_hold_start_frame",
        "terminal_hold_through_final_frame",
        "correct_actor_performs_target",
        "distractor_actor_performs_target",
        "correct_object_is_target",
        "distractor_object_is_target",
        "registered_branch_realized",
        "gross_motion_energy_class",
        "atomic_event_count",
        "assessor_confidence",
        "reverse_opposite_state_transition_realized",
        "object_remained_ground_supported_throughout",
        "acting_dog_empty_mouthed_at_final_frame",
    }
)

SPATIAL_SKETCH_ALGORITHM_DESCRIPTOR = {
    "algorithm": "sha256-counter-rademacher-f32le-v1",
    "counter_encoding": "ascii-decimal-seed:row:column",
    "bit_rule": "sha256(counter)[0]&1;0=>-1,1=>+1",
    "scale": "float32(1/sqrt(930))",
    "layout": "c-row-major-[16,930]",
    "endianness": "little",
    "registered_value_digest": "sha256(fitq-canonical-fp32-little-endian-v1|shape=16,930|+raw-bytes)",
    "rank_check": "exact-rational-rank-of-integer-sign-gram",
}


class T2VFactorialBankError(RuntimeError):
    """Raised before an ambiguous, selected, or privileged artifact is accepted."""


def _reject_json_constant(value: str) -> None:
    raise T2VFactorialBankError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise T2VFactorialBankError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise T2VFactorialBankError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_initial_gaussian_safetensors(path: Path) -> dict[str, Any]:
    """Independently parse and hash the one-tensor native Gaussian artifact.

    This deliberately does not import ``torch`` or trust hashes copied into the
    renderer receipt.  Safetensors stores a little-endian unsigned 64-bit JSON
    header length followed by tensor bytes.  The pinned native writer emits one
    contiguous FP32 tensor and fixed observer metadata; any extra tensor,
    offset, byte, dtype, shape, or metadata change is rejected.
    """

    resolved = _require_plain_file(path, label="native initial Gaussian")
    expected_numel = math.prod(LATENT_SHAPE)
    expected_nbytes = expected_numel * 4
    file_size = resolved.stat().st_size
    try:
        with resolved.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors header is truncated"
                )
            header_length = struct.unpack("<Q", prefix)[0]
            if (
                header_length <= 0
                or header_length > 1 << 20
                or header_length > file_size - 8
            ):
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors header length differs"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors header is truncated"
                )
            try:
                header = json.loads(
                    header_bytes.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors header is invalid"
                ) from error
            if not isinstance(header, dict):
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors header root differs"
                )
            if set(header) != {"__metadata__", "official_initial_gaussian"}:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors tensor/key closure differs"
                )
            metadata = header["__metadata__"]
            expected_metadata = {
                "coordinate": "bernini_native_target_latent_before_rearrange",
                "source": "observed_return_of_official_module_global_randn_tensor",
                "observer_only": "true",
                "external_initial_noise_injection": "false",
            }
            if metadata != expected_metadata:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors metadata differs"
                )
            tensor = header["official_initial_gaussian"]
            if (
                not isinstance(tensor, dict)
                or set(tensor) != {"dtype", "shape", "data_offsets"}
                or tensor.get("dtype") != "F32"
                or tensor.get("shape") != list(LATENT_SHAPE)
                or tensor.get("data_offsets") != [0, expected_nbytes]
            ):
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors tensor contract differs"
                )
            data_start = 8 + header_length
            if file_size != data_start + expected_nbytes:
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors data extent differs"
                )
            artifact_digest = hashlib.sha256(prefix + header_bytes)
            raw_digest = hashlib.sha256()
            remaining = expected_nbytes
            while remaining:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    raise T2VFactorialBankError(
                        "initial Gaussian safetensors data is truncated"
                    )
                raw_digest.update(chunk)
                artifact_digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise T2VFactorialBankError(
                    "initial Gaussian safetensors has trailing data"
                )
    except OSError as error:
        raise T2VFactorialBankError(
            "initial Gaussian safetensors could not be read"
        ) from error
    return {
        "tensor_key": "official_initial_gaussian",
        "dtype": "F32",
        "shape": list(LATENT_SHAPE),
        "numel": expected_numel,
        "byte_count": expected_nbytes,
        "artifact_file_sha256": artifact_digest.hexdigest(),
        "tensor_value_sha256": raw_digest.hexdigest(),
        "independently_parsed_without_renderer_receipt": True,
    }


def _require_plain_file(path: str | Path, *, label: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise T2VFactorialBankError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise T2VFactorialBankError(f"{label} must be a plain file")
    return resolved


def load_json_file(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _require_plain_file(path, label=label)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise T2VFactorialBankError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise T2VFactorialBankError(f"{label} root must be an object")
    return value, resolved


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise T2VFactorialBankError(
            f"{label} fields differ; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _require_slug(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise T2VFactorialBankError(f"{label} must be a lowercase path-safe slug")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise T2VFactorialBankError(f"{label} must be a lowercase SHA-256")
    return value


def _require_prompt(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r", "\t"))
        or len(value.encode("utf-8")) > 4096
    ):
        raise T2VFactorialBankError(f"{label} must be trimmed, single-line text")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise T2VFactorialBankError(f"{label} must be boolean")
    return value


def spatial_sketch_construction_digest() -> str:
    """Digest the complete, versioned reconstruction algorithm descriptor."""

    return object_sha256(SPATIAL_SKETCH_ALGORITHM_DESCRIPTOR)


def reconstruct_spatial_sketch(metadata: Mapping[str, Any]) -> bytes:
    """Reconstruct the fixed [16,930] signed FP32 patch-sketch matrix.

    Each element is generated independently from
    ``SHA256(ascii("seed:row:column"))[0] & 1`` and encoded as little-endian
    FP32 ``+/- float32(1/sqrt(930))`` in C row-major order.  Returning bytes
    keeps this verifier independent of NumPy and PyTorch.
    """

    if not isinstance(metadata, Mapping):
        raise T2VFactorialBankError("spatial sketch metadata must be an object")
    if metadata.get("construction_id") != SPATIAL_SKETCH_ALGORITHM_DESCRIPTOR["algorithm"]:
        raise T2VFactorialBankError("spatial sketch construction id differs")
    _require_slug(metadata.get("sketch_id"), label="spatial sketch id")
    if metadata.get("construction_digest") != spatial_sketch_construction_digest():
        raise T2VFactorialBankError("spatial sketch construction digest differs")
    seed = metadata.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise T2VFactorialBankError("spatial sketch seed differs")
    if (
        metadata.get("matrix_shape") != [16, 930]
        or metadata.get("patch_height") != 31
        or metadata.get("patch_width") != 30
        or metadata.get("flatten_order") != "patch-y-x"
        or metadata.get("value_dtype") != "float32"
        or metadata.get("value_encoding") != "little-endian-c-row-major"
        or metadata.get("normalization") != "per-row-rademacher-1-over-sqrt-930"
        or metadata.get("matrix_value_digest_scheme")
        != "fitq-canonical-fp32-little-endian-v1"
        or metadata.get("signed") is not True
        or metadata.get("data_dependent") is not False
        or metadata.get("verified_exact_row_rank") != 16
    ):
        raise T2VFactorialBankError("spatial sketch matrix contract differs")
    scale = struct.unpack("<f", struct.pack("<f", 1.0 / math.sqrt(930.0)))[0]
    output = bytearray(16 * 930 * 4)
    offset = 0
    for row in range(16):
        for column in range(930):
            counter = f"{seed}:{row}:{column}".encode("ascii")
            sign = 1.0 if hashlib.sha256(counter).digest()[0] & 1 else -1.0
            struct.pack_into("<f", output, offset, sign * scale)
            offset += 4
    return bytes(output)


def validate_spatial_sketch(metadata: Mapping[str, Any]) -> str:
    """Reconstruct a sketch and return the FITQ canonical value digest."""

    if metadata.get("status") != "preregistered":
        raise T2VFactorialBankError("spatial sketch is not preregistered")
    expected_value = _require_sha256(
        metadata.get("matrix_value_sha256"), label="spatial sketch matrix SHA-256"
    )
    expected_raw = _require_sha256(
        metadata.get("matrix_raw_bytes_sha256"),
        label="spatial sketch raw-byte SHA-256",
    )
    raw = reconstruct_spatial_sketch(metadata)
    actual_raw = hashlib.sha256(raw).hexdigest()
    header = b"fitq-canonical-fp32-little-endian-v1|shape=16,930|"
    actual_value = hashlib.sha256(header + raw).hexdigest()
    if (
        actual_raw != expected_raw
        or actual_value != expected_value
        or spatial_sketch_exact_row_rank(metadata) != 16
    ):
        raise T2VFactorialBankError("spatial sketch reconstructed matrix digest differs")
    return actual_value


def spatial_sketch_exact_row_rank(metadata: Mapping[str, Any]) -> int:
    """Return exact row rank via the integer sign Gram matrix."""

    seed = metadata.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise T2VFactorialBankError("spatial sketch seed differs")
    signs = [
        [
            1
            if hashlib.sha256(f"{seed}:{row}:{column}".encode("ascii")).digest()[0]
            & 1
            else -1
            for column in range(930)
        ]
        for row in range(16)
    ]
    matrix = [
        [
            Fraction(sum(left * right for left, right in zip(signs[row], signs[column])))
            for column in range(16)
        ]
        for row in range(16)
    ]
    rank = 0
    for column in range(16):
        pivot = next((row for row in range(rank, 16) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(16):
            if row == rank or not matrix[row][column]:
                continue
            scale = matrix[row][column]
            matrix[row] = [
                left - scale * right
                for left, right in zip(matrix[row], matrix[rank])
            ]
        rank += 1
    return rank


def _require_template(value: Any, *, label: str) -> str:
    template = _require_prompt(value, label=label)
    observed: set[str] = set()
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as error:
        raise T2VFactorialBankError(f"{label} has invalid format syntax") from error
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _PROMPT_PLACEHOLDERS:
            raise T2VFactorialBankError(
                f"{label} contains unsupported placeholder {field_name!r}"
            )
        if format_spec or conversion:
            raise T2VFactorialBankError(f"{label} may not use formatting operators")
        observed.add(field_name)
    if observed != set(_PROMPT_PLACEHOLDERS):
        raise T2VFactorialBankError(
            f"{label} must name primary and distractor actors and objects plus scene/camera"
        )
    return template


def _normalize_level_rows(
    raw_rows: Any,
    *,
    count: Optional[int],
    fields: frozenset[str],
    id_field: str,
    text_fields: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list) or not raw_rows:
        raise T2VFactorialBankError(f"{label} must be a nonempty list")
    if count is not None and len(raw_rows) != count:
        raise T2VFactorialBankError(f"{label} requires exactly {count} levels")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    phrases: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise T2VFactorialBankError(f"{label}[{index}] must be an object")
        _require_exact_fields(raw, fields, label=f"{label}[{index}]")
        row = dict(raw)
        row[id_field] = _require_slug(raw.get(id_field), label=f"{label}[{index}].{id_field}")
        if row[id_field] in ids:
            raise T2VFactorialBankError(f"{label} ids must be unique")
        ids.add(row[id_field])
        for field in text_fields:
            row[field] = _require_prompt(raw.get(field), label=f"{label}[{index}].{field}")
            if row[field] in phrases:
                raise T2VFactorialBankError(f"{label} phrase fields must be distinct")
            phrases.add(row[field])
        result.append(row)
    return result


def _gf4_multiply_alpha(value: int) -> int:
    # Encoding 0, 1, alpha, alpha+1 with alpha^2=alpha+1.
    return (0, 2, 3, 1)[value]


def scientific_oa_rows() -> list[dict[str, int]]:
    """Return the fixed L16(4^4) table; every pair of axes occurs once."""

    return [
        {
            "oa_row_index": 4 * x + y,
            "actor_index": x,
            "scene_index": y,
            "camera_index": x ^ y,
            "wording_index": x ^ _gf4_multiply_alpha(y),
        }
        for x in range(4)
        for y in range(4)
    ]


def _design_rank(
    rows: Sequence[Mapping[str, Any]], factors: Sequence[str]
) -> tuple[int, int, dict[str, list[Any]]]:
    levels = {factor: sorted({row[factor] for row in rows}) for factor in factors}
    matrix = [
        [Fraction(1)]
        + [
            Fraction(row[factor] == level)
            for factor in factors
            for level in levels[factor][1:]
        ]
        for row in rows
    ]
    columns = 1 + sum(len(levels[factor]) - 1 for factor in factors)
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column]), None
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            scale = matrix[row][column]
            matrix[row] = [
                left - scale * right
                for left, right in zip(matrix[row], matrix[rank])
            ]
        rank += 1
    return rank, columns, levels


def _compile_prompt(
    template: str,
    *,
    actor: Mapping[str, Any],
    object_pair: Mapping[str, Any],
    scene: Mapping[str, Any],
    camera: Mapping[str, Any],
    label: str,
) -> str:
    prompt = template.format(
        actor=actor["actor_phrase"],
        actor_ref=actor["actor_reference"],
        distractor_actor=actor["distractor_actor_phrase"],
        distractor_actor_ref=actor["distractor_actor_reference"],
        object=object_pair["object_phrase"],
        object_ref=object_pair["object_reference"],
        distractor_object=object_pair["distractor_object_phrase"],
        distractor_object_ref=object_pair["distractor_object_reference"],
        scene=scene["scene_phrase"],
        camera=camera["camera_phrase"],
    )
    return _require_prompt(prompt, label=label)


def validate_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T2VFactorialBankError("spec must be an object")
    _require_exact_fields(value, _SPEC_FIELDS, label="spec")
    if value.get("schema_version") != SPEC_SCHEMA:
        raise T2VFactorialBankError("spec schema_version differs")
    profile = value.get("profile")
    if profile not in PROFILES:
        raise T2VFactorialBankError(f"profile must be one of {PROFILES}")
    expected_design = MICRO_DESIGN if profile == "engineering_micro" else SCIENTIFIC_DESIGN
    if value.get("design") != expected_design:
        raise T2VFactorialBankError(f"{profile} requires design={expected_design}")

    action = value.get("action_program")
    if not isinstance(action, Mapping):
        raise T2VFactorialBankError("action_program must be an object")
    _require_exact_fields(action, _ACTION_PROGRAM_FIELDS, label="action_program")
    preconditions = action.get("preconditions")
    milestones = action.get("ordered_milestones")
    hold = action.get("terminal_hold_video_frames")
    if not isinstance(preconditions, list) or len(preconditions) < 2:
        raise T2VFactorialBankError("action_program requires at least two preconditions")
    if not isinstance(milestones, list) or len(milestones) != 4:
        raise T2VFactorialBankError("action_program requires exactly four milestones")
    if hold != [65, 80] or action.get("terminal_hold_required") is not True:
        raise T2VFactorialBankError("terminal hold must be preregistered at video frames 65..80")
    normalized_action = {
        "actor_role": _require_prompt(action.get("actor_role"), label="action actor role"),
        "patient_role": _require_prompt(action.get("patient_role"), label="action patient role"),
        "preconditions": [
            _require_prompt(item, label=f"precondition[{index}]")
            for index, item in enumerate(preconditions)
        ],
        "ordered_milestones": [
            _require_prompt(item, label=f"milestone[{index}]")
            for index, item in enumerate(milestones)
        ],
        "terminal_hold_required": True,
        "terminal_hold_video_frames": [65, 80],
        "reverse_action_definition": _require_prompt(
            action.get("reverse_action_definition"),
            label="action reverse-action definition",
        ),
    }
    if len(set(normalized_action["ordered_milestones"])) != 4:
        raise T2VFactorialBankError("action milestones must be unique")

    source = value.get("source_geometry_video")
    if not isinstance(source, Mapping):
        raise T2VFactorialBankError("source_geometry_video must be an object")
    _require_exact_fields(source, _SOURCE_FIELDS, label="source_geometry_video")
    if (
        source.get("frame_count") != FRAME_COUNT
        or source.get("fps") != FPS
        or source.get("height") != VIDEO_HEIGHT
        or source.get("width") != VIDEO_WIDTH
    ):
        raise T2VFactorialBankError("source geometry must be 81x496x480 at 25 fps")
    normalized_source = {
        "sha256": _require_sha256(source.get("sha256"), label="source sha256"),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "height": VIDEO_HEIGHT,
        "width": VIDEO_WIDTH,
    }

    object_pair = value.get("object_pair")
    if not isinstance(object_pair, Mapping):
        raise T2VFactorialBankError("object_pair must be an object")
    _require_exact_fields(object_pair, _OBJECT_FIELDS, label="object_pair")
    normalized_object = {
        "object_id": _require_slug(object_pair.get("object_id"), label="object id")
    }
    for field in _OBJECT_FIELDS - {"object_id"}:
        normalized_object[field] = _require_prompt(
            object_pair.get(field), label=f"object_pair.{field}"
        )
    if len(set(normalized_object.values())) != len(normalized_object):
        raise T2VFactorialBankError("primary and distractor object terms must be distinct")

    exact = 4 if profile == "scientific" else None
    actors = _normalize_level_rows(
        value.get("actor_levels"), count=exact, fields=_ACTOR_FIELDS,
        id_field="actor_id",
        text_fields=("actor_phrase", "actor_reference", "distractor_actor_phrase", "distractor_actor_reference"),
        label="actor_levels",
    )
    scenes = _normalize_level_rows(
        value.get("scene_levels"), count=exact, fields=_SCENE_FIELDS,
        id_field="scene_id", text_fields=("scene_phrase",), label="scene_levels",
    )
    cameras = _normalize_level_rows(
        value.get("camera_levels"), count=exact, fields=_CAMERA_FIELDS,
        id_field="camera_id", text_fields=("camera_phrase",), label="camera_levels",
    )
    for index, actor in enumerate(actors):
        if actor.get("split") not in AXIS_SPLITS:
            raise T2VFactorialBankError(f"actor_levels[{index}].split differs")

    raw_wordings = value.get("wording_levels")
    if not isinstance(raw_wordings, list) or not raw_wordings:
        raise T2VFactorialBankError("wording_levels must be nonempty")
    if exact is not None and len(raw_wordings) != exact:
        raise T2VFactorialBankError("scientific wording_levels requires exactly 4 levels")
    wordings: list[dict[str, Any]] = []
    wording_ids: set[str] = set()
    template_hashes: set[str] = set()
    for index, raw in enumerate(raw_wordings):
        if not isinstance(raw, Mapping):
            raise T2VFactorialBankError(f"wording_levels[{index}] must be an object")
        _require_exact_fields(raw, _WORDING_FIELDS, label=f"wording_levels[{index}]")
        wording_id = _require_slug(raw.get("wording_id"), label=f"wording_levels[{index}].wording_id")
        if wording_id in wording_ids:
            raise T2VFactorialBankError("wording ids must be unique")
        wording_ids.add(wording_id)
        templates = raw.get("templates")
        if not isinstance(templates, Mapping):
            raise T2VFactorialBankError("wording templates must be an object")
        _require_exact_fields(templates, frozenset(BRANCH_ORDER), label=f"wording_levels[{index}].templates")
        normalized_templates: dict[str, str] = {}
        for branch in BRANCH_ORDER:
            template = _require_template(templates.get(branch), label=f"wording[{index}].{branch}")
            digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
            if digest in template_hashes:
                raise T2VFactorialBankError("every wording/branch template must be textually distinct")
            template_hashes.add(digest)
            normalized_templates[branch] = template
        wordings.append({"wording_id": wording_id, "templates": normalized_templates})

    raw_replicates = value.get("seed_replicates")
    if not isinstance(raw_replicates, list) or not raw_replicates:
        raise T2VFactorialBankError("seed_replicates must be nonempty")
    if profile == "scientific" and len(raw_replicates) != 2:
        raise T2VFactorialBankError("scientific design requires exactly two seed replicates")
    replicates: list[dict[str, Any]] = []
    all_seeds: set[int] = set()
    replicate_ids: set[str] = set()
    ladder_length: Optional[int] = None
    for index, raw in enumerate(raw_replicates):
        if not isinstance(raw, Mapping):
            raise T2VFactorialBankError(f"seed_replicates[{index}] must be an object")
        _require_exact_fields(raw, _SEED_REPLICATE_FIELDS, label=f"seed_replicates[{index}]")
        replicate_id = _require_slug(raw.get("replicate_id"), label="replicate id")
        initial = raw.get("initial_seed")
        topups = raw.get("topup_seeds")
        if replicate_id in replicate_ids:
            raise T2VFactorialBankError("replicate ids must be unique")
        if type(initial) is not int or not 0 <= initial < 2**63:
            raise T2VFactorialBankError("initial seed is out of range")
        if not isinstance(topups, list) or (profile == "scientific" and len(topups) < 2):
            raise T2VFactorialBankError("scientific seed ladder requires at least two top-up seeds")
        ladder = [initial, *topups]
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in ladder):
            raise T2VFactorialBankError("top-up seed is out of range")
        if any(seed in all_seeds for seed in ladder) or len(set(ladder)) != len(ladder):
            raise T2VFactorialBankError("all seed-ladder values must be globally unique")
        if ladder_length is not None and len(ladder) != ladder_length:
            raise T2VFactorialBankError("replicate seed ladders must have equal length")
        ladder_length = len(ladder)
        all_seeds.update(ladder)
        replicate_ids.add(replicate_id)
        replicates.append({"replicate_id": replicate_id, "initial_seed": initial, "topup_seeds": list(topups), "seed_ladder": ladder})

    sketch = value.get("spatial_sketch")
    if not isinstance(sketch, Mapping):
        raise T2VFactorialBankError("spatial_sketch must be an object")
    _require_exact_fields(sketch, _SPATIAL_SKETCH_FIELDS, label="spatial_sketch")
    normalized_sketch = dict(sketch)
    if profile == "scientific":
        expected = {
            "schema_version": "fitq-fixed-signed-spatial-sketch-v1",
            "status": "preregistered",
            "sketch_id": "dmiq-fitq-patch31x30-rademacher-s20260808017-v1",
            "construction_id": "sha256-counter-rademacher-f32le-v1",
            "construction_digest": spatial_sketch_construction_digest(),
            "matrix_shape": [16, 930],
            "patch_height": 31,
            "patch_width": 30,
            "flatten_order": "patch-y-x",
            "value_dtype": "float32",
            "value_encoding": "little-endian-c-row-major",
            "normalization": "per-row-rademacher-1-over-sqrt-930",
            "verified_exact_row_rank": 16,
            "signed": True,
            "data_dependent": False,
        }
        if any(normalized_sketch.get(key) != expected_value for key, expected_value in expected.items()):
            raise T2VFactorialBankError("scientific spatial sketch is not fixed, signed, and preregistered")
        validate_spatial_sketch(normalized_sketch)
    else:
        if normalized_sketch.get("status") not in {"pending", "preregistered"}:
            raise T2VFactorialBankError("micro spatial sketch status differs")
        if normalized_sketch.get("status") == "preregistered":
            validate_spatial_sketch(normalized_sketch)

    if profile == "scientific":
        if [row["split"] for row in actors].count("discovery") != 3 or actors[-1]["split"] != "confirmation":
            raise T2VFactorialBankError("scientific actor levels require first 3 discovery and final confirmation")
        # The exact table and factor-level ordering are part of preregistration.
        rows = scientific_oa_rows()
        for left_index, left in enumerate(("actor_index", "scene_index", "camera_index", "wording_index")):
            for right in ("actor_index", "scene_index", "camera_index", "wording_index")[left_index + 1 :]:
                pairs = [(row[left], row[right]) for row in rows]
                if len(set(pairs)) != 16:
                    raise T2VFactorialBankError("internal L16 table lost strength-two balance")

    normalized = {
        "schema_version": SPEC_SCHEMA,
        "bank_id": _require_slug(value.get("bank_id"), label="bank_id"),
        "profile": profile,
        "design": expected_design,
        "action_family_id": _require_slug(value.get("action_family_id"), label="action_family_id"),
        "action_program": normalized_action,
        "source_geometry_video": normalized_source,
        "object_pair": normalized_object,
        "actor_levels": actors,
        "scene_levels": scenes,
        "camera_levels": cameras,
        "wording_levels": wordings,
        "seed_replicates": replicates,
        "spatial_sketch": normalized_sketch,
    }
    # Detect accidental prompt collisions only after the entire factor grid exists.
    prompt_hashes: set[str] = set()
    for actor in actors:
        for scene in scenes:
            for camera in cameras:
                for wording in wordings:
                    for branch in BRANCH_ORDER:
                        prompt = _compile_prompt(wording["templates"][branch], actor=actor, object_pair=normalized_object, scene=scene, camera=camera, label="compiled prompt")
                        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        if digest in prompt_hashes:
                            raise T2VFactorialBankError("compiled prompts must be globally unique")
                        prompt_hashes.add(digest)
    return normalized


def build_manifest(
    spec: Mapping[str, Any],
    *,
    method_source_revision: str,
    method_source_archive_sha256: str,
    attempt_rung: int = 0,
) -> dict[str, Any]:
    normalized = validate_spec(spec)
    if (
        type(method_source_revision) is not str
        or _SHA1_RE.fullmatch(method_source_revision) is None
    ):
        raise T2VFactorialBankError(
            "manifest method source revision must be a full lowercase SHA-1"
        )
    method_archive_digest = _require_sha256(
        method_source_archive_sha256,
        label="manifest method source archive SHA-256",
    )
    ladder_length = len(normalized["seed_replicates"][0]["seed_ladder"])
    if type(attempt_rung) is not int or not 0 <= attempt_rung < ladder_length:
        raise T2VFactorialBankError("attempt_rung is outside the preregistered seed ladder")

    if normalized["profile"] == "scientific":
        design_rows = scientific_oa_rows()
    else:
        design_rows = [
            {
                "oa_row_index": index,
                "actor_index": actor_index,
                "scene_index": scene_index,
                "camera_index": camera_index,
                "wording_index": wording_index,
            }
            for index, (actor_index, scene_index, camera_index, wording_index) in enumerate(
                (a, s, c, w)
                for a in range(len(normalized["actor_levels"]))
                for s in range(len(normalized["scene_levels"]))
                for c in range(len(normalized["camera_levels"]))
                for w in range(len(normalized["wording_levels"]))
            )
        ]

    cells: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    group_counts = {group: 0 for group in GROUPS}
    # A top-up manifest is cumulative: rung N contains every cell and entry
    # from rungs 0..N, in rung-major order.  Thus failed attempts remain in the
    # analyzed bank and cannot be silently replaced by a later winner.
    for rendered_rung in range(attempt_rung + 1):
        for row in design_rows:
            actor = normalized["actor_levels"][row["actor_index"]]
            scene = normalized["scene_levels"][row["scene_index"]]
            camera = normalized["camera_levels"][row["camera_index"]]
            wording = normalized["wording_levels"][row["wording_index"]]
            for replicate_index, replicate in enumerate(normalized["seed_replicates"]):
                slot_id = f"oa{row['oa_row_index']:02d}--{replicate['replicate_id']}"
                cell_id = f"{slot_id}--r{rendered_rung}"
                group = GROUPS[(row["oa_row_index"] + replicate_index) % 2]
                seed = replicate["seed_ladder"][rendered_rung]
                cell = {
                    "design_slot_id": slot_id,
                    "proposal_cell_id": cell_id,
                    "proposal_cell_index": len(cells),
                    "oa_row_index": row["oa_row_index"],
                    "attempt_rung": rendered_rung,
                    "analysis_split": actor["split"],
                    "actor_id": actor["actor_id"],
                    "scene_id": scene["scene_id"],
                    "camera_id": camera["camera_id"],
                    "wording_id": wording["wording_id"],
                    "seed_replicate_id": replicate["replicate_id"],
                    "seed": seed,
                    "seed_ladder": list(replicate["seed_ladder"]),
                    "execution_group": group,
                    "matched_semantic_branch_count": len(BRANCH_ORDER),
                }
                cells.append(cell)
                for branch_index, branch in enumerate(BRANCH_ORDER):
                    prompt = _compile_prompt(
                        wording["templates"][branch], actor=actor,
                        object_pair=normalized["object_pair"], scene=scene, camera=camera,
                        label=f"proposal {cell_id}/{branch}",
                    )
                    entry_id = f"{cell_id}--{branch.replace('_', '-')}"
                    _require_slug(entry_id, label="entry_id")
                    entries.append(
                        {
                            "entry_id": entry_id,
                            "ordinal": len(entries),
                            "design_slot_id": slot_id,
                            "proposal_cell_id": cell_id,
                            "proposal_cell_index": cell["proposal_cell_index"],
                            "oa_row_index": row["oa_row_index"],
                            "attempt_rung": rendered_rung,
                            "analysis_split": actor["split"],
                            "semantic_branch": branch,
                            "branch_index": branch_index,
                            "action_family_id": normalized["action_family_id"],
                            "actor_id": actor["actor_id"],
                            "scene_id": scene["scene_id"],
                            "camera_id": camera["camera_id"],
                            "wording_id": wording["wording_id"],
                            "object_id": normalized["object_pair"]["object_id"],
                            "seed_replicate_id": replicate["replicate_id"],
                            "seed": seed,
                            "prompt": prompt,
                            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                            "execution_group": group,
                            "group_local_order": group_counts[group],
                            "output_subdir": f"entries/{entry_id}",
                        }
                    )
                    group_counts[group] += 1

    factors = ("actor_index", "scene_index", "camera_index", "wording_index")
    full_rank, full_columns, full_levels = _design_rank(design_rows, factors)
    discovery_rows = [
        row for row in design_rows
        if normalized["actor_levels"][row["actor_index"]]["split"] == "discovery"
    ]
    confirmation_rows = [
        row for row in design_rows
        if normalized["actor_levels"][row["actor_index"]]["split"] == "confirmation"
    ]
    discovery_rank, discovery_columns, discovery_levels = _design_rank(discovery_rows, factors)
    requested_discovery = sum(cell["analysis_split"] == "discovery" for cell in cells)
    requested_confirmation = sum(cell["analysis_split"] == "confirmation" for cell in cells)
    scientific_shape_met = (
        normalized["profile"] == "scientific"
        and len(design_rows) == 16
        and len(cells) == 32 * (attempt_rung + 1)
        and requested_discovery == 24 * (attempt_rung + 1)
        and requested_confirmation == 8 * (attempt_rung + 1)
        and full_rank == full_columns == 13
        and discovery_rank == discovery_columns == 12
    )
    if normalized["profile"] == "scientific" and not scientific_shape_met:
        raise T2VFactorialBankError("scientific OA or discovery rank contract differs")

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "bank_id": normalized["bank_id"],
        "profile": normalized["profile"],
        "design": normalized["design"],
        "attempt_rung": attempt_rung,
        "action_family_id": normalized["action_family_id"],
        "action_program": normalized["action_program"],
        "action_program_digest": object_sha256(normalized["action_program"]),
        "source_geometry_video": {
            **normalized["source_geometry_video"],
            "renderer_use": "exact81_bucket_selection_and_hash_verification_only",
            "source_pixels_forwarded_to_t2v": False,
            "source_latent_constructed_for_t2v": False,
        },
        "renderer_contract": {
            "implementation": "infer_native_identity_generation_canary.py",
            "implementation_arm": "t2v",
            "method_source_revision": method_source_revision,
            "method_source_archive_sha256": method_archive_digest,
            "method_source_preregistered_before_render": True,
            "bernini_commit": BERNINI_COMMIT,
            "veomni_commit": VEOMNI_COMMIT,
            "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "video_height": VIDEO_HEIGHT,
            "video_width": VIDEO_WIDTH,
            "latent_shape": list(LATENT_SHAPE),
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_mode": "t2v_apg",
            "omega_vid": 1.25,
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "eta": 0.5,
            "norm_threshold": [50.0, 50.0],
            "momentum": 0.0,
            "target_initialization": "official_gen_wanx22_fresh_gaussian",
            "single_expert": "transformer_1",
            "ulysses_size": ULYSSES_SIZE,
            "initial_noise_artifact_required": True,
            "initial_noise_raw_value_sha256_required": True,
            "full_source_video_count": 0,
            "source_derived_reference_count": 0,
            "target_mixed_with_source_latent": False,
            "source_or_reference_latent_forbidden": True,
            "external_mask_flow_pose_track_trajectory": False,
            "training_performed": False,
        },
        "factorial_contract": {
            "semantic_branches": list(BRANCH_ORDER),
            "negative_branches": list(NEGATIVE_BRANCH_ORDER),
            "oa_row_count": len(design_rows),
            "seed_replicate_count": len(normalized["seed_replicates"]),
            "base_design_slot_count": len(design_rows)
            * len(normalized["seed_replicates"]),
            "included_attempt_rungs": list(range(attempt_rung + 1)),
            "manifest_is_cumulative_through_attempt_rung": True,
            "requested_matched_cell_count": len(cells),
            "requested_entry_count": len(entries),
            "requested_discovery_full_action_count": requested_discovery,
            "requested_confirmation_full_action_count": requested_confirmation,
            "independently_verified_discovery_full_action_count": None,
            "independently_verified_confirmation_full_action_count": None,
            "verified_counts_source": "pending_blind_event_audit",
            "minimum_verified_discovery_full_actions": MIN_VERIFIED_DISCOVERY_POSITIVES,
            "minimum_verified_confirmation_full_actions": MIN_VERIFIED_CONFIRMATION_POSITIVES,
            "scientific_shape_and_rank_preregistered": scientific_shape_met,
            "fitq_eligible_after_render": False,
            "same_cell_all_branches_same_sp4_group": True,
            "same_cell_all_branches_same_initial_noise_required": True,
            "replicate_group_crossover": normalized["profile"] == "scientific",
            "rgb_or_latent_targets": False,
            "action_marginal_only": True,
        },
        "design_diagnostics": {
            "factors": list(factors),
            "oa_strength": 2 if normalized["profile"] == "scientific" else None,
            "full_design": {"row_count": len(design_rows), "rank": full_rank, "columns": full_columns, "observed_levels": full_levels, "full_column_rank": full_rank == full_columns},
            "discovery_only": {"row_count": len(discovery_rows), "rank": discovery_rank, "columns": discovery_columns, "observed_levels": discovery_levels, "full_column_rank": discovery_rank == discovery_columns},
            "confirmation_only": {"row_count": len(confirmation_rows), "main_effect_rank_not_claimed": True},
            "replicate_rows_do_not_inflate_design_rank": True,
            "provenance_note": "rank is exact rational rank of preregistered rows; confirmation is a frozen actor-level generalization test, not a second fitted factorial model",
        },
        "axis_levels": {
            "object_pair": normalized["object_pair"],
            "actor_levels": normalized["actor_levels"],
            "scene_levels": normalized["scene_levels"],
            "camera_levels": normalized["camera_levels"],
            "wording_levels": normalized["wording_levels"],
            "seed_replicates": normalized["seed_replicates"],
        },
        "spatial_sketch": normalized["spatial_sketch"],
        "spatial_sketch_contract": {
            "coordinate": "bernini-hidden-patch-grid-31x30",
            "sketch_id": normalized["spatial_sketch"].get("sketch_id"),
            "matrix_shape": normalized["spatial_sketch"].get("matrix_shape"),
            "matrix_value_sha256": normalized["spatial_sketch"].get(
                "matrix_value_sha256"
            ),
            "matrix_raw_bytes_sha256": normalized["spatial_sketch"].get(
                "matrix_raw_bytes_sha256"
            ),
            "matrix_value_digest_scheme": normalized["spatial_sketch"].get(
                "matrix_value_digest_scheme"
            ),
            "verified_exact_row_rank": normalized["spatial_sketch"].get(
                "verified_exact_row_rank"
            ),
            "construction_digest": normalized["spatial_sketch"].get(
                "construction_digest"
            ),
            "reconstructed_before_manifest_acceptance": normalized["spatial_sketch"].get(
                "status"
            )
            == "preregistered",
            "fitq_core_must_bind_exact_matrix_value_sha256": True,
            "latent-grid-62x60-sketch-forbidden": True,
        },
        "topup_contract": {
            "policy": "whole_32_cell_cohort_next_rung_only",
            "current_attempt_rung": attempt_rung,
            "maximum_attempt_rung": ladder_length - 1,
            "next_attempt_rung": attempt_rung + 1 if attempt_rung + 1 < ladder_length else None,
            "trigger": "blind_verified_minimum_not_met",
            "individual_cell_or_winner_topup_forbidden": True,
            "previous_failed_and_successful_attempts_retained": True,
            "previous_attempt_entries_included_in_this_manifest": True,
            "manifest_is_cumulative_through_current_rung": True,
            "cumulative_prior_artifact_reuse_worker_implemented": False,
            "attempt_rung_greater_than_zero_launch_authorized": False,
            "seed_ladders_frozen_in_spec_before_rung0": True,
        },
        "registered_design_cells": cells,
        "execution_topology": {
            "nodes": 1,
            "gpu_type": "mi210",
            "gpu_count": 8,
            "parallel_groups": [
                {"group_id": group, "visible_device_slots": list(GROUP_VISIBLE_DEVICE_SLOTS[group]), "world_size": ULYSSES_SIZE, "ulysses_size": ULYSSES_SIZE, "cell_count": sum(cell["execution_group"] == group for cell in cells), "entry_count": group_counts[group]}
                for group in GROUPS
            ],
            "groups_are_isolated": True,
            "ulysses8_forbidden": True,
            "launcher_readiness": {
                "launcher": "auh_dmiq_t2v_factorial_bank_dual4.sbatch",
                "engineering_micro_timing_authorized": True,
                "scientific_scale_ready": False,
                "scientific_launch_authorized": False,
                "persistent_model_worker_implemented": False,
                "cumulative_topup_artifact_reuse_implemented": False,
                "current_wrapper_model_loads_per_sp4_group": group_counts[
                    "sp4-a"
                ],
                "blocker": "current wrapper starts one torchrun and reloads the frozen model per entry; implement a persistent per-SP4 worker before the 320-render scientific bank",
            },
        },
        "entries": entries,
        "interpretation": {
            "optimizer_update_authorized": False,
            "model_training_authorized": False,
            "best_proposal_selection_authorized": False,
            "scientific_claim_authorized": False,
            "fitq_confirmation_eligible": False,
            "semantic_event_status": "pending_independent_split_isolated_event_audit",
            "renderer_self_receipt_proves_semantics": False,
            "current_launcher_authorizes_scientific_render": False,
        },
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    return manifest


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise T2VFactorialBankError("manifest schema differs")
    declared = _require_sha256(value.get("manifest_digest"), label="manifest digest")
    unsigned = dict(value)
    unsigned.pop("manifest_digest", None)
    if object_sha256(unsigned) != declared:
        raise T2VFactorialBankError("manifest embedded digest differs")
    axes = value.get("axis_levels")
    source = value.get("source_geometry_video")
    renderer = value.get("renderer_contract")
    if (
        not isinstance(axes, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(renderer, Mapping)
    ):
        raise T2VFactorialBankError("manifest axes/source/renderer differ")
    spec = {
        "schema_version": SPEC_SCHEMA,
        "bank_id": value.get("bank_id"),
        "profile": value.get("profile"),
        "design": value.get("design"),
        "action_family_id": value.get("action_family_id"),
        "action_program": value.get("action_program"),
        "source_geometry_video": {key: source.get(key) for key in _SOURCE_FIELDS},
        "object_pair": axes.get("object_pair"),
        "actor_levels": axes.get("actor_levels"),
        "scene_levels": axes.get("scene_levels"),
        "camera_levels": axes.get("camera_levels"),
        "wording_levels": axes.get("wording_levels"),
        "seed_replicates": [
            {key: row.get(key) for key in _SEED_REPLICATE_FIELDS}
            for row in axes.get("seed_replicates", [])
            if isinstance(row, Mapping)
        ],
        "spatial_sketch": value.get("spatial_sketch"),
    }
    rebuilt = build_manifest(
        spec,
        method_source_revision=renderer.get("method_source_revision"),
        method_source_archive_sha256=renderer.get(
            "method_source_archive_sha256"
        ),
        attempt_rung=value.get("attempt_rung"),
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(value):
        raise T2VFactorialBankError("manifest differs from deterministic reconstruction")
    return rebuilt


def _fresh_absolute_output(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise T2VFactorialBankError(f"{label} must be absolute and non-root")
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise T2VFactorialBankError(f"refusing to overwrite {label}")
    return output


def write_json_atomically(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise T2VFactorialBankError("refusing to overwrite JSON artifact")
    payload = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def load_validated_manifest(path: str | Path, *, expected_file_sha256: Optional[str] = None) -> tuple[dict[str, Any], Path, str]:
    value, resolved = load_json_file(path, label="factorial manifest")
    actual_sha = file_sha256(resolved)
    if expected_file_sha256 is not None and actual_sha != _require_sha256(expected_file_sha256, label="manifest file sha256"):
        raise T2VFactorialBankError("manifest file SHA-256 differs")
    return validate_manifest(value), resolved, actual_sha


def entry_by_id(manifest: Mapping[str, Any], entry_id: str) -> dict[str, Any]:
    _require_slug(entry_id, label="entry_id")
    rows = [entry for entry in manifest["entries"] if entry["entry_id"] == entry_id]
    if len(rows) != 1:
        raise T2VFactorialBankError("entry_id is absent or non-unique")
    return dict(rows[0])


def render_entry(
    *, manifest_path: str, manifest_file_sha256: str, entry_id: str,
    output_root: str, bernini_root: str, veomni_root: str, checkpoint: str,
    checkpoint_content_manifest: str, source_video: str,
    method_source_revision: str, method_source_archive_sha256: str,
) -> int:
    manifest, _, _ = load_validated_manifest(manifest_path, expected_file_sha256=manifest_file_sha256)
    entry = entry_by_id(manifest, entry_id)
    if _SHA1_RE.fullmatch(method_source_revision) is None:
        raise T2VFactorialBankError("method source revision must be full SHA-1")
    _require_sha256(method_source_archive_sha256, label="method archive SHA-256")
    renderer = manifest["renderer_contract"]
    if (
        method_source_revision != renderer["method_source_revision"]
        or method_source_archive_sha256
        != renderer["method_source_archive_sha256"]
    ):
        raise T2VFactorialBankError(
            "runtime method source differs from preregistered manifest"
        )
    source = _require_plain_file(source_video, label="source geometry video")
    if file_sha256(source) != manifest["source_geometry_video"]["sha256"]:
        raise T2VFactorialBankError("source geometry video SHA-256 differs")
    root = Path(output_root).expanduser()
    if not root.is_absolute() or root != root.resolve(strict=True) or not root.is_dir() or root.is_symlink():
        raise T2VFactorialBankError("output root must be an existing canonical directory")
    output = root / entry["output_subdir"]
    if output.parent.resolve(strict=True) != (root / "entries").resolve(strict=True):
        raise T2VFactorialBankError("entry output escaped output root")
    import infer_native_identity_generation_canary as native
    return native.main(
        [
            "--bernini-root", bernini_root, "--veomni-root", veomni_root,
            "--checkpoint", checkpoint, "--checkpoint-content-manifest", checkpoint_content_manifest,
            "--source-video", str(source), "--expected-source-sha256", manifest["source_geometry_video"]["sha256"],
            "--action-prompt", entry["prompt"], "--expected-action-prompt-sha256", entry["prompt_utf8_sha256"],
            "--output-dir", str(output), "--arms", "t2v", "--num-inference-steps", str(NUM_INFERENCE_STEPS),
            "--seed", str(entry["seed"]), "--expected-bernini-commit", renderer["bernini_commit"],
            "--expected-veomni-commit", renderer["veomni_commit"],
            "--expected-checkpoint-tree-sha256", renderer["checkpoint_tree_sha256"],
            "--method-source-revision", method_source_revision,
            "--method-source-archive-sha256", method_source_archive_sha256,
        ]
    )


def _load_sealed_json(path: Path, *, label: str, digest_field: str) -> dict[str, Any]:
    value, _ = load_json_file(path, label=label)
    declared = _require_sha256(value.get(digest_field), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    if object_sha256(unsigned) != declared:
        raise T2VFactorialBankError(f"{label} embedded digest differs")
    return value


def _audit_native_entry(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    output_root: Path,
    gaussian_inspection_cache: Optional[dict[tuple[Any, ...], dict[str, Any]]] = None,
) -> dict[str, Any]:
    entry_root = output_root / entry["output_subdir"]
    receipt_path = _require_plain_file(entry_root / "receipt.json", label=f"{entry['entry_id']} receipt")
    receipt = _load_sealed_json(receipt_path, label="native receipt", digest_field="receipt_digest")
    if receipt.get("schema_version") != NATIVE_RECEIPT_SCHEMA or receipt.get("arms") != ["t2v"]:
        raise T2VFactorialBankError("native receipt schema/arm differs")
    inputs = receipt.get("input")
    sampling = receipt.get("sampling", {}).get("t2v")
    conditioning = receipt.get("conditioning", {}).get("t2v")
    renderer = manifest["renderer_contract"]
    if not all(isinstance(value, Mapping) for value in (inputs, sampling, conditioning)):
        raise T2VFactorialBankError("native input/sampling/conditioning receipt differs")
    if inputs.get("source_video_sha256") != manifest["source_geometry_video"]["sha256"] or inputs.get("action_prompt_utf8_sha256") != entry["prompt_utf8_sha256"]:
        raise T2VFactorialBankError("native input digest differs")
    if inputs.get("accepted_external_conditions") != ["source_video", "action_prompt"]:
        raise T2VFactorialBankError("native external condition closure differs")
    for name in ("target_video", "external_reference_image_or_video", "external_mask_flow_pose_track_trajectory", "external_first_frame_anchor"):
        if inputs.get(name) is not False:
            raise T2VFactorialBankError("privileged input appeared in native entry")
    source_ids = conditioning.get("source_ids")
    if (
        conditioning.get("full_source_video_count") != 0
        or conditioning.get("source_derived_reference_count") != 0
        or conditioning.get("source_frame_indices") != []
        or receipt.get("source_condition_artifact") is not None
        or not isinstance(source_ids, Mapping)
        or source_ids.get("target_source_id") != 0
        or source_ids.get("conditioning_source_count") != 0
        or source_ids.get("video_source_ids") != []
        or source_ids.get("reference_source_ids") != []
    ):
        raise T2VFactorialBankError("source/reference latent appeared in T2V entry")
    expected_sampling = {
        "num_frames": FRAME_COUNT, "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_mode": renderer["guidance_mode"], "omega_vid": renderer["omega_vid"],
        "omega_img": renderer["omega_img"], "omega_txt": renderer["omega_txt"],
        "omega_scale": renderer["omega_scale"], "flow_shift": renderer["flow_shift"],
        "seed": entry["seed"], "eta": renderer["eta"],
        "norm_threshold": renderer["norm_threshold"], "momentum": renderer["momentum"],
        "target_initialization": renderer["target_initialization"],
        "target_mixed_with_source_latent": False, "custom_sampler_or_scheduler": False,
        "single_expert": renderer["single_expert"], "ulysses_size": ULYSSES_SIZE,
    }
    if any(sampling.get(key) != expected for key, expected in expected_sampling.items()):
        raise T2VFactorialBankError("native T2V exact sampling contract differs")
    geometry = receipt.get("latent_geometry")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("video_latent_shape") != list(LATENT_SHAPE)
        or geometry.get("target_patch_tokens") != 19_530
        or geometry.get("one_reference_patch_tokens") != 930
    ):
        raise T2VFactorialBankError("native latent geometry differs from [1,16,21,62,60]")
    if (
        receipt.get("bernini_commit") != renderer["bernini_commit"]
        or receipt.get("veomni_commit") != renderer["veomni_commit"]
        or receipt.get("checkpoint", {}).get("tree_sha256") != renderer["checkpoint_tree_sha256"]
        or receipt.get("freeze_certificate", {}).get("base_frozen") is not True
        or receipt.get("interpretation", {}).get("training_performed") is not False
        or receipt.get("interpretation", {}).get("best_arm_selected") is not False
    ):
        raise T2VFactorialBankError("native frozen model provenance differs")
    revision = receipt.get("method_source_revision")
    archive_sha = receipt.get("method_source_archive_sha256")
    if (
        revision != renderer["method_source_revision"]
        or archive_sha != renderer["method_source_archive_sha256"]
    ):
        raise T2VFactorialBankError(
            "native method source differs from preregistered manifest"
        )

    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"t2v"} or not isinstance(outputs["t2v"], Mapping):
        raise T2VFactorialBankError("native output set differs")
    output = outputs["t2v"]
    video = _require_plain_file(output.get("path", ""), label="T2V video")
    clean = output.get("normalized_clean_latent")
    if not isinstance(clean, Mapping):
        raise T2VFactorialBankError("native clean latent is absent")
    clean_path = _require_plain_file(clean.get("path", ""), label="T2V clean latent")
    if video.parent != entry_root.resolve(strict=True) or clean_path.parent != entry_root.resolve(strict=True):
        raise T2VFactorialBankError("native output escaped entry directory")
    if (
        file_sha256(video) != output.get("sha256")
        or output.get("frame_count") != FRAME_COUNT
        or output.get("fps") != FPS
        or output.get("height") != VIDEO_HEIGHT
        or output.get("width") != VIDEO_WIDTH
    ):
        raise T2VFactorialBankError("native video artifact differs")
    if (
        file_sha256(clean_path) != clean.get("sha256")
        or clean.get("shape") != list(LATENT_SHAPE)
        or clean.get("stored_dtype") != "torch.float32"
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("source_video_vae_encode_before_any_decode") is not False
        or clean.get("mp4_decode_reencode_used") is not False
    ):
        raise T2VFactorialBankError("native clean latent contract differs")

    gaussian_map = receipt.get("initial_noise_artifacts")
    if not isinstance(gaussian_map, Mapping) or set(gaussian_map) != {"t2v"} or not isinstance(gaussian_map["t2v"], Mapping):
        raise T2VFactorialBankError("native initial_noise_artifacts.t2v is absent")
    gaussian = gaussian_map["t2v"]
    gaussian_path = _require_plain_file(gaussian.get("path", ""), label="native initial Gaussian")
    if gaussian_path.parent != entry_root.resolve(strict=True):
        raise T2VFactorialBankError("native initial Gaussian escaped entry directory")
    gaussian_sha = _require_sha256(gaussian.get("sha256"), label="initial Gaussian artifact SHA-256")
    raw_sha = _require_sha256(gaussian.get("tensor_value_sha256"), label="initial Gaussian raw-value SHA-256")
    gaussian_stat = gaussian_path.stat()
    cache_key = (
        gaussian_stat.st_dev,
        gaussian_stat.st_ino,
        gaussian_stat.st_size,
        gaussian_stat.st_mtime_ns,
        gaussian_sha,
    )
    independent = None
    if gaussian_inspection_cache is not None:
        independent = gaussian_inspection_cache.get(cache_key)
    if independent is None:
        independent = _inspect_initial_gaussian_safetensors(gaussian_path)
        if gaussian_inspection_cache is not None:
            gaussian_inspection_cache[cache_key] = independent
    if (
        independent["artifact_file_sha256"] != gaussian_sha
        or independent["tensor_value_sha256"] != raw_sha
        or gaussian.get("raw_value_sha256") != raw_sha
        or gaussian.get("tensor_key") != independent["tensor_key"]
        or gaussian.get("shape") != list(LATENT_SHAPE)
        or gaussian.get("stored_dtype") != "torch.float32"
        or gaussian.get("numel") != independent["numel"]
        or gaussian.get("byte_count") != independent["byte_count"]
        or gaussian.get("official_randn_tensor_call_count") != 1
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("source_or_target_derived") is not False
    ):
        raise T2VFactorialBankError("native initial Gaussian provenance differs")
    return {
        "entry_id": entry["entry_id"], "semantic_branch": entry["semantic_branch"],
        "proposal_cell_id": entry["proposal_cell_id"], "design_slot_id": entry["design_slot_id"],
        "analysis_split": entry["analysis_split"], "execution_group": entry["execution_group"],
        "seed_replicate_id": entry["seed_replicate_id"], "seed": entry["seed"],
        "attempt_rung": entry["attempt_rung"],
        "native_receipt_path": str(receipt_path), "native_receipt_file_sha256": file_sha256(receipt_path),
        "native_receipt_digest": receipt["receipt_digest"],
        "video_path": str(video), "video_sha256": output["sha256"],
        "clean_latent_path": str(clean_path), "clean_latent_sha256": clean["sha256"],
        "initial_noise_path": str(gaussian_path), "initial_noise_file_sha256": gaussian_sha,
        "initial_noise_tensor_value_sha256": raw_sha,
        "initial_noise_value_digest_independently_recomputed": True,
        "method_source_revision": revision, "method_source_archive_sha256": archive_sha,
        "pure_t2v_condition_audit_pass": True,
    }


def finalize_bank(*, manifest_path: str, manifest_file_sha256: str, output_root: str, output_receipt: str) -> dict[str, Any]:
    manifest, manifest_resolved, manifest_sha = load_validated_manifest(manifest_path, expected_file_sha256=manifest_file_sha256)
    requested_root = Path(output_root).expanduser()
    if not requested_root.is_absolute() or requested_root.is_symlink():
        raise T2VFactorialBankError("bank output root must be canonical")
    root = requested_root.resolve(strict=True)
    if root != requested_root or not root.is_dir() or root.is_symlink():
        raise T2VFactorialBankError("bank output root must be a plain directory")
    gaussian_inspection_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    audited = [
        _audit_native_entry(
            manifest,
            entry,
            root,
            gaussian_inspection_cache=gaussian_inspection_cache,
        )
        for entry in manifest["entries"]
    ]
    revisions = {row["method_source_revision"] for row in audited}
    archives = {row["method_source_archive_sha256"] for row in audited}
    if len(revisions) != 1 or len(archives) != 1:
        raise T2VFactorialBankError("entries do not share one immutable method source")
    for cell in manifest["registered_design_cells"]:
        rows = [row for row in audited if row["proposal_cell_id"] == cell["proposal_cell_id"]]
        if [row["semantic_branch"] for row in rows] != list(BRANCH_ORDER):
            raise T2VFactorialBankError("cell semantic branch order/closure differs")
        if len({row["execution_group"] for row in rows}) != 1:
            raise T2VFactorialBankError("cell was split across SP4 groups")
        if len({row["initial_noise_tensor_value_sha256"] for row in rows}) != 1:
            raise T2VFactorialBankError("same-cell branches did not use byte-identical initial Gaussian values")
    receipt: dict[str, Any] = {
        "schema_version": BANK_RECEIPT_SCHEMA,
        "bank_id": manifest["bank_id"], "profile": manifest["profile"],
        "manifest_path": str(manifest_resolved), "manifest_file_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"], "attempt_rung": manifest["attempt_rung"],
        "entry_count": len(audited), "proposal_cell_count": len(manifest["registered_design_cells"]),
        "entries": audited,
        "native_method_provenance": {
            "method_source_revision": next(iter(revisions)),
            "method_source_archive_sha256": next(iter(archives)),
            "preregistered_in_manifest_before_render": True,
            "all_entries_exact": True,
        },
        "condition_closure": {
            "renderer_arm": "t2v", "source_video_role": "exact81_bucket_selection_and_hash_verification_only",
            "source_latent_or_reference_consumed": False, "target_video_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "all_native_entry_audits_pass": True,
            "all_cells_share_exact_initial_noise_across_ten_branches": True,
            "all_initial_noise_value_digests_independently_recomputed": True,
        },
        "requested_counts": {
            "discovery_full_action": manifest["factorial_contract"]["requested_discovery_full_action_count"],
            "confirmation_full_action": manifest["factorial_contract"]["requested_confirmation_full_action_count"],
        },
        "independently_event_verified_counts": {"discovery_full_action": None, "confirmation_full_action": None, "fully_realized_discovery_cells": None, "fully_realized_confirmation_cells": None},
        "interpretation": {
            "factorial_render_complete": True, "action_marginal_only": True,
            "rgb_or_latent_supervision_target": False, "optimizer_update_authorized": False,
            "optimizer_update": "null", "training_performed": False,
            "best_proposal_selected": False, "scientific_claim_authorized": False,
            "fitq_confirmation_eligible": False,
            "fitq_eligibility_status": "pending_independent_split_isolated_event_audit",
            "renderer_self_receipt_proves_semantics": False,
            "failed_attempts_must_be_retained": True,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    output = _fresh_absolute_output(output_receipt, label="bank receipt")
    write_json_atomically(output, receipt)
    return receipt


def _branch_audit_pass(row: Mapping[str, Any], *, milestone_count: int) -> bool:
    branch = row["semantic_branch"]
    realized = row["ordered_milestone_realized"]
    frames = row["ordered_milestone_first_frame"]
    present_frames = [frame for frame in frames if frame is not None]
    ordered = len(present_frames) == milestone_count and all(left < right for left, right in zip(present_frames, present_frames[1:]))
    all_realized = realized == [True] * milestone_count
    none_realized = realized == [False] * milestone_count
    terminal = row["terminal_hold_start_frame"] is not None and row["terminal_hold_start_frame"] <= 65 and row["terminal_hold_through_final_frame"] is True
    correct_actor = row["correct_actor_performs_target"]
    distractor_actor = row["distractor_actor_performs_target"]
    correct_object = row["correct_object_is_target"]
    distractor_object = row["distractor_object_is_target"]
    registered = row["registered_branch_realized"] is True
    preconditions = all(row["initial_preconditions_realized"])
    if not registered or not preconditions or row["assessor_confidence"] != "high":
        return False
    if branch == "full_action":
        return all_realized and ordered and terminal and correct_actor and not distractor_actor and correct_object and not distractor_object
    if branch == "noop":
        return none_realized and not correct_actor and not distractor_actor and not correct_object and not distractor_object and row["atomic_event_count"] == 0
    if branch == "incomplete_action":
        prefix = (
            realized[:2] == [True, True]
            and realized[2:] == [False, False]
            and len(present_frames) == 2
            and present_frames[0] < present_frames[1]
        )
        return (
            prefix
            and not terminal
            and correct_actor
            and not distractor_actor
            and correct_object
            and not distractor_object
        )
    if branch == "reverse_action":
        return (
            realized == [True, True, False, False]
            and len(present_frames) == 2
            and present_frames[0] < present_frames[1]
            and not terminal
            and correct_actor
            and not distractor_actor
            and correct_object
            and not distractor_object
            and row["reverse_opposite_state_transition_realized"] is True
            and row["object_remained_ground_supported_throughout"] is True
            and row["acting_dog_empty_mouthed_at_final_frame"] is True
            and row["gross_motion_energy_class"] == "matched-to-full-action"
            and row["atomic_event_count"] == 4
        )
    if branch == "shuffled_action":
        return any(realized) and len(present_frames) >= 2 and not ordered and not terminal and correct_actor and not distractor_actor
    if branch in {"camera_only", "appearance_only"}:
        return none_realized and not correct_actor and not distractor_actor and not correct_object and not distractor_object
    if branch == "generic_wrong_motion":
        return (
            not all_realized and not terminal and not correct_actor and not distractor_actor
            and row["gross_motion_energy_class"] == "matched-to-full-action"
            and row["atomic_event_count"] == GENERIC_WRONG_MOTION_ATOMIC_EVENT_COUNT
        )
    if branch == "wrong_actor":
        return all_realized and ordered and terminal and not correct_actor and distractor_actor and correct_object and not distractor_object
    if branch == "wrong_object":
        return all_realized and ordered and terminal and correct_actor and not distractor_actor and not correct_object and distractor_object
    raise T2VFactorialBankError("unknown semantic branch in audit")


def validate_event_audit(manifest: Mapping[str, Any], bank_receipt: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_manifest(manifest)
    if bank_receipt.get("schema_version") != BANK_RECEIPT_SCHEMA:
        raise T2VFactorialBankError("bank receipt schema differs")
    bank_declared = _require_sha256(bank_receipt.get("receipt_digest"), label="bank receipt digest")
    bank_unsigned = dict(bank_receipt)
    bank_unsigned.pop("receipt_digest", None)
    if object_sha256(bank_unsigned) != bank_declared or bank_receipt.get("manifest_digest") != manifest["manifest_digest"]:
        raise T2VFactorialBankError("bank receipt digest/manifest binding differs")
    bank_entries = bank_receipt.get("entries")
    bank_condition = bank_receipt.get("condition_closure")
    bank_interpretation = bank_receipt.get("interpretation")
    if (
        bank_receipt.get("bank_id") != manifest["bank_id"]
        or bank_receipt.get("profile") != manifest["profile"]
        or bank_receipt.get("attempt_rung") != manifest["attempt_rung"]
        or bank_receipt.get("entry_count") != len(manifest["entries"])
        or bank_receipt.get("proposal_cell_count")
        != len(manifest["registered_design_cells"])
        or not isinstance(bank_entries, list)
        or not isinstance(bank_condition, Mapping)
        or not isinstance(bank_interpretation, Mapping)
        or [row.get("entry_id") for row in bank_entries if isinstance(row, Mapping)]
        != [entry["entry_id"] for entry in manifest["entries"]]
        or bank_condition.get("all_native_entry_audits_pass")
        is not True
        or bank_condition.get(
            "all_cells_share_exact_initial_noise_across_ten_branches"
        )
        is not True
        or bank_interpretation.get("factorial_render_complete")
        is not True
        or bank_interpretation.get("fitq_confirmation_eligible")
        is not False
    ):
        raise T2VFactorialBankError("bank receipt render/provenance closure differs")
    if not isinstance(audit, Mapping):
        raise T2VFactorialBankError("event audit must be an object")
    _require_exact_fields(audit, _EVENT_AUDIT_FIELDS, label="event audit")
    if audit.get("schema_version") != EVENT_AUDIT_SCHEMA:
        raise T2VFactorialBankError("event audit schema differs")
    declared = _require_sha256(audit.get("audit_digest"), label="event audit digest")
    unsigned = dict(audit)
    unsigned.pop("audit_digest", None)
    if object_sha256(unsigned) != declared:
        raise T2VFactorialBankError("event audit embedded digest differs")
    if audit.get("bank_manifest_digest") != manifest["manifest_digest"] or audit.get("bank_receipt_digest") != bank_declared:
        raise T2VFactorialBankError("event audit bank binding differs")

    assessor = audit.get("assessor_contract")
    isolation = audit.get("split_isolation")
    rows = audit.get("rows")
    if not isinstance(assessor, Mapping) or not isinstance(isolation, Mapping) or not isinstance(rows, list):
        raise T2VFactorialBankError("event audit assessor/split/rows differ")
    _require_exact_fields(assessor, _ASSESSOR_FIELDS, label="assessor contract")
    _require_exact_fields(isolation, _SPLIT_ISOLATION_FIELDS, label="split isolation")
    _require_slug(assessor.get("assessor_id"), label="assessor id")
    _require_slug(assessor.get("organization_id"), label="organization id")
    _require_prompt(assessor.get("attestation"), label="assessor attestation")
    if (
        assessor.get("independent_of_renderer_and_method") is not True
        or assessor.get("no_fitq_or_training_outputs_seen") is not True
        or assessor.get("proposal_videos_are_only_model_outputs_seen") is not True
        or assessor.get("blinded_fields") != list(BLINDED_FIELDS)
    ):
        raise T2VFactorialBankError("assessor independence/blinding contract differs")
    if any(
        isolation.get(field) is not True
        for field in (
            "discovery_decisions_frozen_before_confirmation_opened",
            "confirmation_opened_after_freeze",
            "confirmation_rows_never_used_for_prompt_seed_or_design_selection",
            "row_order_matches_manifest",
        )
    ):
        raise T2VFactorialBankError("discovery/confirmation split isolation failed")
    if len(rows) != len(manifest["entries"]):
        raise T2VFactorialBankError("event audit must contain exactly every requested entry")
    receipt_rows = {row["entry_id"]: row for row in bank_entries if isinstance(row, Mapping)}
    row_pass: dict[str, bool] = {}
    normalized_rows: list[dict[str, Any]] = []
    for index, (entry, raw) in enumerate(zip(manifest["entries"], rows)):
        if not isinstance(raw, Mapping):
            raise T2VFactorialBankError(f"event audit row {index} is not an object")
        _require_exact_fields(raw, _EVENT_ROW_FIELDS, label=f"event audit rows[{index}]")
        if raw.get("entry_id") != entry["entry_id"] or raw.get("semantic_branch") != entry["semantic_branch"] or raw.get("analysis_split") != entry["analysis_split"]:
            raise T2VFactorialBankError("event audit row order or manifest label differs")
        rendered = receipt_rows.get(entry["entry_id"])
        if rendered is None or raw.get("video_sha256") != rendered.get("video_sha256"):
            raise T2VFactorialBankError("event audit video binding differs")
        preconditions = raw.get("initial_preconditions_realized")
        realized = raw.get("ordered_milestone_realized")
        frames = raw.get("ordered_milestone_first_frame")
        milestone_count = len(manifest["action_program"]["ordered_milestones"])
        if not isinstance(preconditions, list) or len(preconditions) != len(manifest["action_program"]["preconditions"]) or any(type(flag) is not bool for flag in preconditions):
            raise T2VFactorialBankError("event audit precondition vector differs")
        if not isinstance(realized, list) or len(realized) != milestone_count or any(type(flag) is not bool for flag in realized):
            raise T2VFactorialBankError("event audit milestone vector differs")
        if not isinstance(frames, list) or len(frames) != milestone_count or any(frame is not None and (type(frame) is not int or not 0 <= frame < FRAME_COUNT) for frame in frames):
            raise T2VFactorialBankError("event audit milestone frames differ")
        if any((flag and frame is None) or (not flag and frame is not None) for flag, frame in zip(realized, frames)):
            raise T2VFactorialBankError("milestone realization/frame presence differs")
        hold_start = raw.get("terminal_hold_start_frame")
        if hold_start is not None and (type(hold_start) is not int or not 0 <= hold_start < FRAME_COUNT):
            raise T2VFactorialBankError("terminal hold start differs")
        for field in (
            "terminal_hold_through_final_frame", "correct_actor_performs_target",
            "distractor_actor_performs_target", "correct_object_is_target",
            "distractor_object_is_target", "registered_branch_realized",
            "reverse_opposite_state_transition_realized",
            "object_remained_ground_supported_throughout",
            "acting_dog_empty_mouthed_at_final_frame",
        ):
            _require_bool(raw.get(field), label=f"event row {field}")
        if raw.get("gross_motion_energy_class") not in {"none", "below-full-action", "matched-to-full-action", "above-full-action", "not-applicable"}:
            raise T2VFactorialBankError("gross motion energy class differs")
        if type(raw.get("atomic_event_count")) is not int or not 0 <= raw["atomic_event_count"] <= 16:
            raise T2VFactorialBankError("atomic event count differs")
        if raw.get("assessor_confidence") not in {"high", "medium", "low"}:
            raise T2VFactorialBankError("assessor confidence differs")
        normalized = dict(raw)
        normalized_rows.append(normalized)
        row_pass[entry["entry_id"]] = _branch_audit_pass(normalized, milestone_count=milestone_count)

    discovery_rows = [row for row in normalized_rows if row["analysis_split"] == "discovery"]
    if object_sha256(discovery_rows) != isolation.get("discovery_decisions_sha256"):
        raise T2VFactorialBankError("frozen discovery decision digest differs")
    verified_positive = {"discovery": 0, "confirmation": 0}
    verified_cells = {"discovery": 0, "confirmation": 0}
    for cell in manifest["registered_design_cells"]:
        cell_entries = [entry for entry in manifest["entries"] if entry["proposal_cell_id"] == cell["proposal_cell_id"]]
        positive = next(entry for entry in cell_entries if entry["semantic_branch"] == "full_action")
        split = cell["analysis_split"]
        verified_positive[split] += int(row_pass[positive["entry_id"]])
        verified_cells[split] += int(all(row_pass[entry["entry_id"]] for entry in cell_entries))
    try:
        sketch_digest = validate_spatial_sketch(manifest["spatial_sketch"])
        sketch_ready = (
            sketch_digest
            == manifest["spatial_sketch_contract"]["matrix_value_sha256"]
        )
    except T2VFactorialBankError:
        sketch_digest = None
        sketch_ready = False
    numerical_minimum_met = (
        manifest["profile"] == "scientific"
        and sketch_ready
        and verified_positive["discovery"] >= MIN_VERIFIED_DISCOVERY_POSITIVES
        and verified_positive["confirmation"] >= MIN_VERIFIED_CONFIRMATION_POSITIVES
        and verified_cells["discovery"] >= MIN_VERIFIED_DISCOVERY_POSITIVES
        and verified_cells["confirmation"] >= MIN_VERIFIED_CONFIRMATION_POSITIVES
    )
    # This v2 object is a useful rubric/count audit, but it is one mutable JSON
    # containing self-asserted chronology booleans and manifest-revealing entry
    # IDs.  A digest can detect accidental mutation; it cannot prove assessor
    # identity, that discovery was sealed first, or that confirmation was not
    # opened.  It therefore must never authorize scientific FITQ.  The sealed
    # workflow is a separate preregistration + discovery signature + release
    # signature + confirmation signature chain.  Likewise, confirmation
    # failure must not trigger this legacy whole-cohort top-up path.
    eligible = False
    next_rung = None
    receipt: dict[str, Any] = {
        "schema_version": EVENT_AUDIT_RECEIPT_SCHEMA,
        "manifest_digest": manifest["manifest_digest"],
        "bank_receipt_digest": bank_declared,
        "event_audit_digest": declared,
        "assessor_id": assessor["assessor_id"],
        "entry_count": len(rows),
        "row_pass_by_entry_id": row_pass,
        "requested_counts": {
            "discovery_full_action": manifest["factorial_contract"]["requested_discovery_full_action_count"],
            "confirmation_full_action": manifest["factorial_contract"]["requested_confirmation_full_action_count"],
        },
        "independently_event_verified_counts": {
            "discovery_full_action": verified_positive["discovery"],
            "confirmation_full_action": verified_positive["confirmation"],
            "fully_realized_discovery_cells": verified_cells["discovery"],
            "fully_realized_confirmation_cells": verified_cells["confirmation"],
        },
        "fitq_bank_eligible": eligible,
        "event_count_minimum_met": numerical_minimum_met,
        "legacy_plain_json_audit_only": True,
        "external_discovery_assessor_signature_verified": False,
        "external_confirmation_release_signature_verified": False,
        "external_confirmation_assessor_signature_verified": False,
        "same_state_owner_by_prompt_cross_query_verified": False,
        "scientific_method_claim_authorized": False,
        "spatial_sketch_matrix_value_sha256": sketch_digest,
        "spatial_sketch_id": manifest["spatial_sketch_contract"]["sketch_id"],
        "fitq_core_exact_spatial_sketch_binding_required": True,
        "optimizer_update_authorized": False,
        "optimizer_update": "null",
        "next_preregistered_whole_cohort_topup_rung": next_rung,
        "individual_winner_selection_authorized": False,
        "all_failed_attempts_retained_required": True,
        "status": "ineligible-unsealed-audit-and-same-state-cross-query-missing",
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--spec", required=True)
    build.add_argument("--method-source-revision", required=True)
    build.add_argument("--method-source-archive-sha256", required=True)
    build.add_argument("--attempt-rung", type=int, default=0)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--expected-file-sha256")
    listing = sub.add_parser("list-entry-ids")
    listing.add_argument("--manifest", required=True)
    listing.add_argument("--expected-file-sha256", required=True)
    listing.add_argument("--group", required=True, choices=GROUPS)
    render = sub.add_parser("render-entry")
    render.add_argument("--manifest", required=True)
    render.add_argument("--expected-file-sha256", required=True)
    render.add_argument("--entry-id", required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--bernini-root", required=True)
    render.add_argument("--veomni-root", required=True)
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--checkpoint-content-manifest", required=True)
    render.add_argument("--source-video", required=True)
    render.add_argument("--method-source-revision", required=True)
    render.add_argument("--method-source-archive-sha256", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--expected-file-sha256", required=True)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--output-receipt", required=True)
    event = sub.add_parser("validate-event-audit")
    event.add_argument("--manifest", required=True)
    event.add_argument("--expected-manifest-file-sha256", required=True)
    event.add_argument("--bank-receipt", required=True)
    event.add_argument("--event-audit", required=True)
    event.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        spec, _ = load_json_file(args.spec, label="factorial spec")
        manifest = build_manifest(
            spec,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
            attempt_rung=args.attempt_rung,
        )
        output = _fresh_absolute_output(args.output, label="manifest output")
        write_json_atomically(output, manifest)
        print(file_sha256(output), flush=True)
        return 0
    if args.command == "validate":
        manifest, path, file_sha = load_validated_manifest(args.manifest, expected_file_sha256=args.expected_file_sha256)
        print(json.dumps({"manifest": str(path), "file_sha256": file_sha, "manifest_digest": manifest["manifest_digest"], "valid": True}, sort_keys=True), flush=True)
        return 0
    if args.command == "list-entry-ids":
        manifest, _, _ = load_validated_manifest(args.manifest, expected_file_sha256=args.expected_file_sha256)
        for entry in manifest["entries"]:
            if entry["execution_group"] == args.group:
                print(entry["entry_id"])
        return 0
    if args.command == "render-entry":
        return render_entry(
            manifest_path=args.manifest, manifest_file_sha256=args.expected_file_sha256,
            entry_id=args.entry_id, output_root=args.output_root,
            bernini_root=args.bernini_root, veomni_root=args.veomni_root,
            checkpoint=args.checkpoint, checkpoint_content_manifest=args.checkpoint_content_manifest,
            source_video=args.source_video, method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
    if args.command == "finalize":
        finalize_bank(manifest_path=args.manifest, manifest_file_sha256=args.expected_file_sha256, output_root=args.output_root, output_receipt=args.output_receipt)
        return 0
    if args.command == "validate-event-audit":
        manifest, _, _ = load_validated_manifest(args.manifest, expected_file_sha256=args.expected_manifest_file_sha256)
        bank_receipt, _ = load_json_file(args.bank_receipt, label="bank receipt")
        event_audit, _ = load_json_file(args.event_audit, label="event audit")
        result = validate_event_audit(manifest, bank_receipt, event_audit)
        output = _fresh_absolute_output(args.output_receipt, label="event audit receipt")
        write_json_atomically(output, result)
        print(file_sha256(output), flush=True)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except T2VFactorialBankError as error:
        print(f"DMIQ_T2V_FACTORIAL_BANK_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


__all__ = [
    "BANK_RECEIPT_SCHEMA", "BERNINI_COMMIT", "BLINDED_FIELDS", "BRANCH_ORDER",
    "CHECKPOINT_TREE_SHA256", "EVENT_AUDIT_RECEIPT_SCHEMA", "EVENT_AUDIT_SCHEMA",
    "LATENT_SHAPE", "MANIFEST_SCHEMA", "MICRO_DESIGN", "NATIVE_RECEIPT_SCHEMA",
    "SCIENTIFIC_DESIGN", "SPATIAL_SKETCH_ALGORITHM_DESCRIPTOR", "SPEC_SCHEMA",
    "T2VFactorialBankError", "VEOMNI_COMMIT",
    "build_manifest", "canonical_json_bytes", "file_sha256", "finalize_bank",
    "load_json_file", "object_sha256", "render_entry", "scientific_oa_rows",
    "reconstruct_spatial_sketch", "spatial_sketch_construction_digest",
    "spatial_sketch_exact_row_rank",
    "validate_event_audit", "validate_manifest", "validate_spatial_sketch",
    "validate_spec", "write_json_atomically",
]
