#!/usr/bin/env python3
"""Build the case01 hand-authored object-trajectory scaffold.

This is an oracle routing artifact, not a learned representation and not
ground-truth target motion.  It compiles the reviewed source dog/bone masks
and sparse mouth boxes into the exact 21 x 31 x 30 Wan latent-patch layout.
The patient object remains at its source support through grip, moves once
during lift, and is held at the source-dog mouth anchor.  Each target bone
token has a source-bone token correspondence, and the original support is
cleared whenever the patient has moved.

Only Python's standard library is used so the authority can be replayed in a
minimal, isolated runner environment.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence
import zlib


SCHEMA_VERSION = "case01-oracle-object-trajectory-scaffold-v1"
CASE_ID = "case01"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."

SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
SOURCE_SIZE = 10_887_043
BONE_REMOVED_SHA256 = (
    "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9"
)
BONE_REMOVED_SIZE = 5_424_975
STAGE0_RECEIPT_SHA256 = (
    "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
)
STAGE0_RECEIPT_SIZE = 22_160
STAGE0_RECEIPT_DIGEST = (
    "36d9b072febab782647f4cda4e63df9d78656b392d33ce4e02777af11697b8fa"
)
G0_SPARSE_SHA256 = (
    "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020"
)
G0_SPARSE_SIZE = 6_882

SOURCE_WIDTH = 704
SOURCE_HEIGHT = 736
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 496
LATENT_PHASES = 21
PATCH_ROWS = 31
PATCH_COLS = 30
TOKENS_PER_PHASE = PATCH_ROWS * PATCH_COLS
PACKED_TOKEN_COUNT = LATENT_PHASES * TOKENS_PER_PHASE
MOUTH_DILATION_SOURCE_PX = 24
HEAD_DILATION_SOURCE_PX = 12
HOLD_MIN_FRAMES = 10
SHA256_RE = re.compile(r"[0-9a-f]{64}")

STAGE_RANGES = (
    ("preserve", 0, 8),
    ("approach", 9, 24),
    ("contact", 25, 28),
    ("grip", 29, 36),
    ("lift", 37, 60),
    ("hold", 61, 80),
)


class Case01TrajectoryError(RuntimeError):
    """Raised instead of emitting an ambiguous or lossy oracle artifact."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Case01TrajectoryError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _stable_file(
    path: Path, *, expected_sha256: str, expected_size: int, label: str
) -> bytes:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Case01TrajectoryError(f"{label} path differs")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_uid,
        row.st_gid,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or digest.hexdigest() != expected_sha256
        or len(payload) != expected_size
    ):
        raise Case01TrajectoryError(f"{label} identity differs")
    return bytes(payload)


def _strict_json(raw: bytes, *, label: str, canonical_lf: bool) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Case01TrajectoryError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise Case01TrajectoryError(f"{label} must be an object")
    if canonical_lf and raw != canonical_json_bytes(value) + b"\n":
        raise Case01TrajectoryError(f"{label} is not canonical JSON plus LF")
    return value


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_binary_png(raw: bytes, *, label: str) -> tuple[int, int, list[bytes]]:
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise Case01TrajectoryError(f"{label} PNG signature differs")
    pos = 8
    idat = bytearray()
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    saw_iend = False
    while pos < len(raw):
        if pos + 12 > len(raw):
            raise Case01TrajectoryError(f"{label} PNG is truncated")
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        kind = raw[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(raw):
            raise Case01TrajectoryError(f"{label} PNG chunk is truncated")
        data = raw[pos + 8 : pos + 8 + length]
        crc = struct.unpack(">I", raw[pos + 8 + length : end])[0]
        if (binascii.crc32(kind + data) & 0xFFFFFFFF) != crc:
            raise Case01TrajectoryError(f"{label} PNG CRC differs")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            saw_iend = True
        pos = end
    if ihdr is None or not saw_iend:
        raise Case01TrajectoryError(f"{label} PNG closure differs")
    width, height, depth, color, compression, filter_method, interlace = ihdr
    if (depth, color, compression, filter_method, interlace) != (8, 0, 0, 0, 0):
        raise Case01TrajectoryError(f"{label} must be 8-bit grayscale noninterlaced")
    try:
        decoded = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise Case01TrajectoryError(f"{label} PNG zlib differs") from error
    if len(decoded) != height * (width + 1):
        raise Case01TrajectoryError(f"{label} PNG payload length differs")
    rows: list[bytes] = []
    previous = bytearray(width)
    offset = 0
    for _ in range(height):
        filter_type = decoded[offset]
        offset += 1
        current = bytearray(decoded[offset : offset + width])
        offset += width
        for x in range(width):
            left = current[x - 1] if x else 0
            above = previous[x]
            upper_left = previous[x - 1] if x else 0
            if filter_type == 1:
                current[x] = (current[x] + left) & 0xFF
            elif filter_type == 2:
                current[x] = (current[x] + above) & 0xFF
            elif filter_type == 3:
                current[x] = (current[x] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                current[x] = (current[x] + _paeth(left, above, upper_left)) & 0xFF
            elif filter_type != 0:
                raise Case01TrajectoryError(f"{label} PNG filter differs")
        if set(current) - {0, 255}:
            raise Case01TrajectoryError(f"{label} PNG mask is not binary")
        rows.append(bytes(current))
        previous = current
    return width, height, rows


def _interpolate_box(
    frame: int, sparse: Sequence[Mapping[str, Any]], key: str
) -> tuple[float, float, float, float]:
    left = next((row for row in reversed(sparse) if row["frame_index"] <= frame), None)
    right = next((row for row in sparse if row["frame_index"] >= frame), None)
    if left is None or right is None:
        raise Case01TrajectoryError("sparse box interpolation is uncovered")
    a = tuple(float(value) for value in left[key])
    b = tuple(float(value) for value in right[key])
    if left["frame_index"] == right["frame_index"]:
        return a
    weight = (frame - left["frame_index"]) / (
        right["frame_index"] - left["frame_index"]
    )
    return tuple((1.0 - weight) * x + weight * y for x, y in zip(a, b))


def _expanded_box(
    box: Sequence[float], amount: int
) -> tuple[float, float, float, float]:
    return (
        max(0.0, box[0] - amount),
        max(0.0, box[1] - amount),
        min(float(SOURCE_WIDTH), box[2] + amount),
        min(float(SOURCE_HEIGHT), box[3] + amount),
    )


def _patches_from_mask(
    rows: Sequence[bytes], *, exclude_box: Sequence[float] | None = None
) -> set[int]:
    patches: set[int] = set()
    for y, row in enumerate(rows):
        output_y = min(OUTPUT_HEIGHT - 1, (y * OUTPUT_HEIGHT) // SOURCE_HEIGHT)
        patch_y = output_y // 16
        for x, value in enumerate(row):
            if not value:
                continue
            if exclude_box is not None and (
                exclude_box[0] <= x < exclude_box[2]
                and exclude_box[1] <= y < exclude_box[3]
            ):
                continue
            output_x = min(OUTPUT_WIDTH - 1, (x * OUTPUT_WIDTH) // SOURCE_WIDTH)
            patch_x = output_x // 16
            patches.add(patch_y * PATCH_COLS + patch_x)
    return patches


def _patches_from_box(box: Sequence[float]) -> set[int]:
    left = max(0, min(OUTPUT_WIDTH - 1, math.floor(box[0] * OUTPUT_WIDTH / SOURCE_WIDTH)))
    top = max(0, min(OUTPUT_HEIGHT - 1, math.floor(box[1] * OUTPUT_HEIGHT / SOURCE_HEIGHT)))
    right = max(left, min(OUTPUT_WIDTH - 1, math.ceil(box[2] * OUTPUT_WIDTH / SOURCE_WIDTH) - 1))
    bottom = max(top, min(OUTPUT_HEIGHT - 1, math.ceil(box[3] * OUTPUT_HEIGHT / SOURCE_HEIGHT) - 1))
    return {
        y * PATCH_COLS + x
        for y in range(top // 16, bottom // 16 + 1)
        for x in range(left // 16, right // 16 + 1)
    }


def _stage(frame: int) -> str:
    matches = [name for name, start, stop in STAGE_RANGES if start <= frame <= stop]
    if len(matches) != 1:
        raise Case01TrajectoryError(f"frame {frame} stage coverage differs")
    return matches[0]


def _phase_window(phase: int) -> tuple[int, ...]:
    if phase == 0:
        return (0,)
    return tuple(range(4 * phase - 3, 4 * phase + 1))


def _translate_patch_set(tokens: set[int], dx: int, dy: int) -> set[int]:
    translated: set[int] = set()
    for token in tokens:
        y, x = divmod(token, PATCH_COLS)
        tx, ty = x + dx, y + dy
        if not (0 <= tx < PATCH_COLS and 0 <= ty < PATCH_ROWS):
            raise Case01TrajectoryError("bone trajectory leaves the patch grid")
        translated.add(ty * PATCH_COLS + tx)
    if len(translated) != len(tokens):
        raise Case01TrajectoryError("bone trajectory loses token identity")
    return translated


def _load_mask(
    stage0_root: Path,
    output_rows: Mapping[str, Mapping[str, Any]],
    kind: str,
    frame: int,
) -> list[bytes]:
    relative = f"masks/{kind}/{frame:05d}.png"
    row = output_rows.get(relative)
    if (
        not isinstance(row, Mapping)
        or set(row) != {"path", "sha256", "size"}
        or row.get("path") != relative
        or not isinstance(row.get("sha256"), str)
        or SHA256_RE.fullmatch(row["sha256"]) is None
        or type(row.get("size")) is not int
        or row["size"] <= 0
    ):
        raise Case01TrajectoryError(f"Stage0 lacks exact {relative}")
    raw = _stable_file(
        (stage0_root / relative).resolve(strict=True),
        expected_sha256=row["sha256"],
        expected_size=row["size"],
        label=relative,
    )
    width, height, rows = _decode_binary_png(raw, label=relative)
    if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
        raise Case01TrajectoryError(f"{relative} geometry differs")
    return rows


def _validate_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(value)
    claimed = artifact.pop("artifact_digest", None)
    if not isinstance(claimed, str) or claimed != object_sha256(artifact):
        raise Case01TrajectoryError("artifact digest differs")
    if set(artifact) != {
        "schema_version",
        "status",
        "case_id",
        "iid",
        "instruction",
        "authority",
        "geometry",
        "latent_layout",
        "typed_action_program",
        "dog_identity_policy",
        "frames",
        "latent_phases",
        "invariants",
        "claim_limits",
    }:
        raise Case01TrajectoryError("artifact root schema differs")
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("status") != "ORACLE_SCAFFOLD_READY_NOT_RENDERER_RESULT"
        or artifact.get("case_id") != CASE_ID
        or artifact.get("iid") != IID
        or artifact.get("instruction") != INSTRUCTION
        or artifact.get("latent_layout", {}).get("packed_token_count")
        != PACKED_TOKEN_COUNT
    ):
        raise Case01TrajectoryError("artifact identity differs")
    if artifact.get("authority") != {
        "source_video": {"sha256": SOURCE_SHA256, "size": SOURCE_SIZE},
        "bone_removed_auxiliary_video": {
            "sha256": BONE_REMOVED_SHA256,
            "size": BONE_REMOVED_SIZE,
        },
        "stage0_receipt": {
            "sha256": STAGE0_RECEIPT_SHA256,
            "size": STAGE0_RECEIPT_SIZE,
            "receipt_digest": STAGE0_RECEIPT_DIGEST,
            "mask_count": 162,
        },
        "g0_sparse_annotations": {
            "sha256": G0_SPARSE_SHA256,
            "size": G0_SPARSE_SIZE,
        },
    }:
        raise Case01TrajectoryError("artifact authority differs")
    if artifact.get("latent_layout") != {
        "causal_phase_policy": "phase0=f0;phase_p=union_frames_4p-3_through_4p",
        "latent_phases": LATENT_PHASES,
        "patch_rows": PATCH_ROWS,
        "patch_cols": PATCH_COLS,
        "tokens_per_phase": TOKENS_PER_PHASE,
        "packed_token_count": PACKED_TOKEN_COUNT,
        "side_local_spatial_token": "patch_y*30+patch_x",
        "scheduler_target_packed_token": "phase*930+side_local_token",
        "attention_source_half_offset": 0,
        "attention_target_half_offset": PACKED_TOKEN_COUNT,
    }:
        raise Case01TrajectoryError("artifact latent layout differs")
    frames = artifact.get("frames")
    phases = artifact.get("latent_phases")
    if not isinstance(frames, list) or len(frames) != 81:
        raise Case01TrajectoryError("artifact frame closure differs")
    if not isinstance(phases, list) or len(phases) != LATENT_PHASES:
        raise Case01TrajectoryError("artifact phase closure differs")
    frame_fields = {
        "frame_index",
        "typed_stage",
        "mouth_center_source_xy",
        "mouth_center_patch_xy",
        "bone_shift_patch_xy",
        "source_bone_token_count",
        "target_bone_token_count",
    }
    for frame_index, row in enumerate(frames):
        if not isinstance(row, Mapping) or set(row) != frame_fields:
            raise Case01TrajectoryError("artifact frame schema differs")
        shift = row.get("bone_shift_patch_xy")
        if (
            row.get("frame_index") != frame_index
            or row.get("typed_stage") != _stage(frame_index)
            or not isinstance(shift, list)
            or len(shift) != 2
            or any(type(item) is not int for item in shift)
            or (frame_index < 37 and shift != [0, 0])
            or (frame_index >= 61 and shift == [0, 0])
            or row.get("source_bone_token_count")
            != row.get("target_bone_token_count")
            or type(row.get("source_bone_token_count")) is not int
            or row["source_bone_token_count"] <= 0
        ):
            raise Case01TrajectoryError("artifact frame semantics differ")
    phase_fields = {
        "phase_index",
        "frame_window",
        "representative_frame",
        "typed_stage",
        "bone_shift_patch_xy",
        "dog_body_core_tokens",
        "dog_identity_core_tokens",
        "source_bone_tokens",
        "target_bone_tokens",
        "origin_clear_tokens",
        "target_responsibility_tokens",
        "bone_token_correspondence",
    }
    for phase_index, row in enumerate(phases):
        if not isinstance(row, Mapping) or set(row) != phase_fields:
            raise Case01TrajectoryError("artifact phase schema differs")
        window = list(_phase_window(phase_index))
        representative = window[-1]
        if (
            row.get("phase_index") != phase_index
            or row.get("frame_window") != window
            or row.get("representative_frame") != representative
            or row.get("typed_stage") != _stage(representative)
            or row.get("bone_shift_patch_xy")
            != frames[representative]["bone_shift_patch_xy"]
        ):
            raise Case01TrajectoryError("artifact phase order differs")
        source = row.get("source_bone_tokens")
        target = row.get("target_bone_tokens")
        pairs = row.get("bone_token_correspondence")
        if (
            not isinstance(source, list)
            or not isinstance(target, list)
            or not isinstance(pairs, list)
            or source != sorted(set(source))
            or target != sorted(set(target))
            or len(source) != len(target)
            or len(pairs) != len(source)
            or {pair[0] for pair in pairs} != set(source)
            or {pair[1] for pair in pairs} != set(target)
        ):
            raise Case01TrajectoryError("bone token conservation differs")
        for key in (
            "dog_body_core_tokens",
            "dog_identity_core_tokens",
            "source_bone_tokens",
            "target_bone_tokens",
            "origin_clear_tokens",
            "target_responsibility_tokens",
        ):
            values = row.get(key)
            if (
                not isinstance(values, list)
                or values != sorted(set(values))
                or any(type(item) is not int or item not in range(TOKENS_PER_PHASE) for item in values)
            ):
                raise Case01TrajectoryError(f"{key} closure differs")
        if not row["dog_identity_core_tokens"] or not row["source_bone_tokens"]:
            raise Case01TrajectoryError("identity/patient support is empty")
        dx, dy = row["bone_shift_patch_xy"]
        expected_pairs = sorted(
            [
                token,
                (token // PATCH_COLS + dy) * PATCH_COLS
                + (token % PATCH_COLS + dx),
            ]
            for token in source
        )
        expected_target = sorted(pair[1] for pair in expected_pairs)
        expected_origin = [] if (dx, dy) == (0, 0) else source
        patient_projection = set(expected_origin) | set(expected_target)
        if (
            pairs != expected_pairs
            or target != expected_target
            or row["origin_clear_tokens"] != expected_origin
            or not set(target).issubset(row["target_responsibility_tokens"])
            or set(row["dog_identity_core_tokens"]) & patient_projection
            or set(row["dog_body_core_tokens"]) & patient_projection
        ):
            raise Case01TrajectoryError("artifact phase projection differs")
    if sum(row["typed_stage"] == "hold" for row in frames) != 20:
        raise Case01TrajectoryError("artifact terminal hold differs")
    if artifact.get("claim_limits") != {
        "hand_authored_oracle": True,
        "learned_representation": False,
        "target_motion_ground_truth": False,
        "renderer_execution": False,
        "scientific_claim_authorized": False,
        "purpose": "frozen-renderer-feasibility-and-condition-consumption-canary",
    }:
        raise Case01TrajectoryError("artifact claim limits differ")
    artifact["artifact_digest"] = claimed
    return artifact


def build_artifact(
    *,
    stage0_root: Path,
    g0_sparse_path: Path,
    source_video: Path,
    bone_removed_video: Path,
) -> dict[str, Any]:
    stage0_root = stage0_root.resolve(strict=True)
    receipt_path = (stage0_root / "receipt.json").resolve(strict=True)
    receipt_raw = _stable_file(
        receipt_path,
        expected_sha256=STAGE0_RECEIPT_SHA256,
        expected_size=STAGE0_RECEIPT_SIZE,
        label="Stage0 receipt",
    )
    receipt = _strict_json(receipt_raw, label="Stage0 receipt", canonical_lf=True)
    if (
        receipt.get("status") != "COMPLETE_STAGE0_MASKLET_DIAGNOSTIC"
        or receipt.get("receipt_digest") != STAGE0_RECEIPT_DIGEST
        or receipt.get("source", {}).get("sha256") != SOURCE_SHA256
        or receipt.get("visible_frame_counts") != {"bone": 81, "dog": 81}
    ):
        raise Case01TrajectoryError("Stage0 receipt semantics differ")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 165:
        raise Case01TrajectoryError("Stage0 output closure differs")
    output_rows = {row.get("path"): row for row in outputs if isinstance(row, dict)}
    if len(output_rows) != 165:
        raise Case01TrajectoryError("Stage0 output paths are not unique")

    g0_raw = _stable_file(
        g0_sparse_path.resolve(strict=True),
        expected_sha256=G0_SPARSE_SHA256,
        expected_size=G0_SPARSE_SIZE,
        label="G0 sparse annotations",
    )
    g0 = _strict_json(g0_raw, label="G0 sparse annotations", canonical_lf=False)
    sparse = g0.get("sparse_annotations")
    if (
        g0.get("iid") != IID
        or g0.get("source_sha256") != SOURCE_SHA256
        or g0.get("image_size_wh") != [SOURCE_WIDTH, SOURCE_HEIGHT]
        or not isinstance(sparse, list)
        or [row.get("frame_index") for row in sparse] != list(range(0, 81, 10))
    ):
        raise Case01TrajectoryError("G0 sparse authority differs")

    _stable_file(
        source_video.resolve(strict=True),
        expected_sha256=SOURCE_SHA256,
        expected_size=SOURCE_SIZE,
        label="exact source video",
    )
    _stable_file(
        bone_removed_video.resolve(strict=True),
        expected_sha256=BONE_REMOVED_SHA256,
        expected_size=BONE_REMOVED_SIZE,
        label="bone-removed auxiliary video",
    )

    frame_data: list[dict[str, Any]] = []
    per_frame_masks: list[dict[str, set[int]]] = []
    for frame in range(81):
        dog_rows = _load_mask(stage0_root, output_rows, "dog", frame)
        bone_rows = _load_mask(stage0_root, output_rows, "bone", frame)
        head_box = _interpolate_box(frame, sparse, "head_bbox_xyxy")
        mouth_box = _interpolate_box(frame, sparse, "mouth_bbox_xyxy")
        head_corridor = _expanded_box(head_box, HEAD_DILATION_SOURCE_PX)
        mouth_corridor = _expanded_box(mouth_box, MOUTH_DILATION_SOURCE_PX)
        dog_all = _patches_from_mask(dog_rows)
        head_corridor_tokens = _patches_from_box(head_corridor)
        mouth_corridor_tokens = _patches_from_box(mouth_corridor)
        dog_body = dog_all - head_corridor_tokens
        dog_identity = dog_all - mouth_corridor_tokens
        source_bone = _patches_from_mask(bone_rows)
        if not dog_body or not dog_identity or not source_bone:
            raise Case01TrajectoryError(f"frame {frame} support is empty")
        mouth_center_source = (
            0.5 * (mouth_box[0] + mouth_box[2]),
            0.5 * (mouth_box[1] + mouth_box[3]),
        )
        mouth_center_patch = (
            int(round(mouth_center_source[0] * OUTPUT_WIDTH / SOURCE_WIDTH / 16.0)),
            int(round(mouth_center_source[1] * OUTPUT_HEIGHT / SOURCE_HEIGHT / 16.0)),
        )
        bone_xy = [(token % PATCH_COLS, token // PATCH_COLS) for token in source_bone]
        bone_center_patch = (
            sum(x for x, _ in bone_xy) / len(bone_xy),
            sum(y for _, y in bone_xy) / len(bone_xy),
        )
        final_dx = int(round(mouth_center_patch[0] - bone_center_patch[0]))
        final_dy = int(round(mouth_center_patch[1] - bone_center_patch[1]))
        stage = _stage(frame)
        if stage == "lift":
            fraction = (frame - 37) / (60 - 37)
        elif stage == "hold":
            fraction = 1.0
        else:
            fraction = 0.0
        shift = (int(round(fraction * final_dx)), int(round(fraction * final_dy)))
        target = _translate_patch_set(source_bone, *shift)
        responsibility = target | _patches_from_box(mouth_corridor)
        per_frame_masks.append(
            {
                "dog_body": dog_body,
                "dog_identity": dog_identity,
                "dog_all": dog_all,
                "head_corridor": head_corridor_tokens,
                "mouth_corridor": mouth_corridor_tokens,
                "source_bone": source_bone,
                "target_bone": target,
                "responsibility": responsibility,
            }
        )
        frame_data.append(
            {
                "frame_index": frame,
                "typed_stage": stage,
                "mouth_center_source_xy": [round(value, 6) for value in mouth_center_source],
                "mouth_center_patch_xy": list(mouth_center_patch),
                "bone_shift_patch_xy": list(shift),
                "source_bone_token_count": len(source_bone),
                "target_bone_token_count": len(target),
            }
        )

    latent_rows: list[dict[str, Any]] = []
    for phase in range(LATENT_PHASES):
        window = _phase_window(phase)
        representative = window[-1]
        source_bone = set().union(
            *(per_frame_masks[frame]["source_bone"] for frame in window)
        )
        dog_all = set().union(
            *(per_frame_masks[frame]["dog_all"] for frame in window)
        )
        head_corridor = set().union(
            *(per_frame_masks[frame]["head_corridor"] for frame in window)
        )
        mouth_corridor = set().union(
            *(per_frame_masks[frame]["mouth_corridor"] for frame in window)
        )
        dx, dy = frame_data[representative]["bone_shift_patch_xy"]
        target_bone = _translate_patch_set(source_bone, dx, dy)
        pairs = sorted(
            [source, target]
            for source, target in zip(
                sorted(source_bone),
                sorted(
                    (token // PATCH_COLS + dy) * PATCH_COLS
                    + (token % PATCH_COLS + dx)
                    for token in source_bone
                ),
            )
        )
        # The sorted zip above is valid for a pure integer translation, but
        # assert it instead of relying on ordering folklore.
        expected_pairs = sorted(
            [
                token,
                (token // PATCH_COLS + dy) * PATCH_COLS
                + (token % PATCH_COLS + dx),
            ]
            for token in source_bone
        )
        if pairs != expected_pairs:
            raise Case01TrajectoryError("bone correspondence ordering differs")
        origin_clear = source_bone if (dx, dy) != (0, 0) else set()
        patient_projection = origin_clear | target_bone
        dog_body = dog_all - head_corridor - patient_projection
        dog_identity = dog_all - mouth_corridor - patient_projection
        responsibility = set().union(
            *(per_frame_masks[frame]["responsibility"] for frame in window)
        ) | target_bone
        if dog_identity & (origin_clear | target_bone):
            raise Case01TrajectoryError(
                f"phase {phase} dog identity core overlaps patient projection"
            )
        latent_rows.append(
            {
                "phase_index": phase,
                "frame_window": list(window),
                "representative_frame": representative,
                "typed_stage": _stage(representative),
                "bone_shift_patch_xy": [dx, dy],
                "dog_body_core_tokens": sorted(dog_body),
                "dog_identity_core_tokens": sorted(dog_identity),
                "source_bone_tokens": sorted(source_bone),
                "target_bone_tokens": sorted(target_bone),
                "origin_clear_tokens": sorted(origin_clear),
                "target_responsibility_tokens": sorted(responsibility),
                "bone_token_correspondence": expected_pairs,
            }
        )

    hold_frames = sum(row["typed_stage"] == "hold" for row in frame_data)
    if hold_frames < HOLD_MIN_FRAMES:
        raise Case01TrajectoryError("terminal hold duration differs")
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ORACLE_SCAFFOLD_READY_NOT_RENDERER_RESULT",
        "case_id": CASE_ID,
        "iid": IID,
        "instruction": INSTRUCTION,
        "authority": {
            "source_video": {"sha256": SOURCE_SHA256, "size": SOURCE_SIZE},
            "bone_removed_auxiliary_video": {
                "sha256": BONE_REMOVED_SHA256,
                "size": BONE_REMOVED_SIZE,
            },
            "stage0_receipt": {
                "sha256": STAGE0_RECEIPT_SHA256,
                "size": STAGE0_RECEIPT_SIZE,
                "receipt_digest": STAGE0_RECEIPT_DIGEST,
                "mask_count": 162,
            },
            "g0_sparse_annotations": {
                "sha256": G0_SPARSE_SHA256,
                "size": G0_SPARSE_SIZE,
            },
        },
        "geometry": {
            "source_wh": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "renderer_bucket_wh": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "mask_downsample": (
                "source-positive-forward-floor-union-to-renderer-then-16px-patch"
            ),
            "small_object_erasure_by_mean_pooling_forbidden": True,
        },
        "latent_layout": {
            "causal_phase_policy": "phase0=f0;phase_p=union_frames_4p-3_through_4p",
            "latent_phases": LATENT_PHASES,
            "patch_rows": PATCH_ROWS,
            "patch_cols": PATCH_COLS,
            "tokens_per_phase": TOKENS_PER_PHASE,
            "packed_token_count": PACKED_TOKEN_COUNT,
            "side_local_spatial_token": "patch_y*30+patch_x",
            "scheduler_target_packed_token": "phase*930+side_local_token",
            "attention_source_half_offset": 0,
            "attention_target_half_offset": PACKED_TOKEN_COUNT,
        },
        "typed_action_program": {
            "actor": "dog#1",
            "effector": "dog#1.mouth",
            "patient": "bone#1",
            "stages": [name for name, _, _ in STAGE_RANGES],
            "stage_ranges_inclusive": [
                {"stage": name, "start": start, "stop": stop}
                for name, start, stop in STAGE_RANGES
            ],
            "pre_lift_patient_stationary": True,
            "lift_is_integer_patch_translation": True,
            "original_support_cleared_after_nonzero_translation": True,
            "terminal_hold_frame_count": hold_frames,
        },
        "dog_identity_policy": {
            "body_core": (
                "phase_union_dog_mask_minus_phase_union_dilated_head_box"
                "_minus_patient_projection"
            ),
            "identity_core": (
                "phase_union_dog_mask_minus_phase_union_dilated_mouth_corridor"
                "_minus_patient_projection"
            ),
            "mouth_corridor_left_free_for_action": True,
            "head_dilation_source_px": HEAD_DILATION_SOURCE_PX,
            "mouth_dilation_source_px": MOUTH_DILATION_SOURCE_PX,
        },
        "frames": frame_data,
        "latent_phases": latent_rows,
        "invariants": {
            "all_81_frames_bound": True,
            "all_21_latent_phases_bound": True,
            "source_target_bone_token_count_equal_every_phase": True,
            "bone_correspondence_bijective_every_phase": True,
            "bone_trajectory_in_bounds": True,
            "dog_identity_core_nonempty_every_phase": True,
            "dog_patient_projection_disjoint_every_phase": True,
            "hold_at_least_10_frames": True,
        },
        "claim_limits": {
            "hand_authored_oracle": True,
            "learned_representation": False,
            "target_motion_ground_truth": False,
            "renderer_execution": False,
            "scientific_claim_authorized": False,
            "purpose": "frozen-renderer-feasibility-and-condition-consumption-canary",
        },
    }
    artifact["artifact_digest"] = object_sha256(artifact)
    return _validate_artifact(artifact)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise Case01TrajectoryError("output path must be canonical absolute")
    if path.exists() or path.is_symlink():
        raise Case01TrajectoryError("refusing to overwrite scaffold output")
    parent = path.parent.resolve(strict=True)
    if parent != path.parent or parent.is_symlink():
        raise Case01TrajectoryError("output parent differs")
    raw = canonical_json_bytes(value) + b"\n"
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Case01TrajectoryError("scaffold write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        if os.pread(descriptor, len(raw), 0) != raw:
            raise Case01TrajectoryError("scaffold bytes differ after write")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", required=True)
    parser.add_argument("--g0-sparse", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--bone-removed-video", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_artifact(
        stage0_root=Path(args.stage0_root).expanduser(),
        g0_sparse_path=Path(args.g0_sparse).expanduser(),
        source_video=Path(args.source_video).expanduser(),
        bone_removed_video=Path(args.bone_removed_video).expanduser(),
    )
    _write_create_only(Path(args.output).expanduser(), artifact)
    print(canonical_json_bytes({
        "status": artifact["status"],
        "artifact_digest": artifact["artifact_digest"],
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
