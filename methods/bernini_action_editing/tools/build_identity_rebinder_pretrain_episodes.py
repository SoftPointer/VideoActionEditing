#!/usr/bin/env python3
"""Build raw-video-only pretraining episodes for IdentityRebinder v1.

Every episode selects target RGB frames from one authority-bound clip and
identity memory from a different authority-bound clip of the same entity.
Exact RGB collisions and perceptual near-duplicates across that clip boundary
are rejected.  A different authority identity is bound as the negative, and
deterministic shuffle/drop/resample views are materialized.  Frame indices
remain audit-only provenance; the model contract consumes decoded pixels and
the orderless atlas encoder only.  No edited video, instruction, caption,
action, mask, track, flow or pose field is accepted.

The programmatic builder is side-effect free.  The CLI publishes canonical
JSONL plus a receipt only when ``--publish`` is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence


RAW_VIDEO_ROW_SCHEMA = "identity-rebinder-raw-video-row-v2"
EPISODE_SCHEMA = "bernini-identity-rebinder-pretrain-episode-v2"
RECEIPT_SCHEMA = "bernini-identity-rebinder-pretrain-receipt-v2"
AUTHORITY_SCHEMA = "bernini-identity-authority-release-v1"
AUTHORITY_ASSERTION_SCHEMA = "bernini-identity-authority-assertion-v1"
DEFAULT_SEED = 20260809
DEFAULT_TARGET_FRAMES = 2
DEFAULT_DROP_FRACTION = 0.25
MINIMUM_FRAMES = 8
MAXIMUM_DECODED_FRAMES = 1024
MAXIMUM_FRAME_EDGE = 4096
MINIMUM_CLIP_PER_IDENTITY = 2
MINIMUM_NEAR_DUPLICATE_HAMMING = 12
MEDIA_DECODE_TIMEOUT_SECONDS = 180

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "video_id",
        "authority_assertion_id",
        "split",
        "source_video_path",
        "source_video_sha256",
        "frames",
    }
)
_FRAME_FIELDS = frozenset(
    {
        "frame_index",
        "frame_path",
        "frame_sha256",
        "decoded_rgb_sha256",
        "perceptual_rgbq4_8x8",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "authority_id",
        "release_id",
        "evidence_policy",
        "assertions",
    }
)
_ASSERTION_FIELDS = frozenset(
    {
        "schema_version",
        "assertion_id",
        "video_id",
        "clip_id",
        "identity_id",
        "split",
        "source_video_sha256",
        "evidence_digest",
    }
)
_ALLOWED_EVIDENCE_POLICIES = frozenset(
    {
        "human-audited-stable-entity-v1",
        "licensed-stable-entity-registry-v1",
    }
)
_FORBIDDEN_FIELD_FRAGMENTS = (
    "action",
    "caption",
    "instruction",
    "target_video",
    "edited",
    "mask",
    "track",
    "flow",
    "pose",
    "motion",
    "timestamp",
)


class IdentityEpisodeBuildError(RuntimeError):
    """Raised before an unsafe or semantically ambiguous episode is emitted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IdentityEpisodeBuildError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IdentityEpisodeBuildError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise IdentityEpisodeBuildError(f"non-finite JSON constant: {value}")


def _decode_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except IdentityEpisodeBuildError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IdentityEpisodeBuildError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise IdentityEpisodeBuildError(f"{context} must be one JSON object")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise IdentityEpisodeBuildError(f"{label} is not a safe identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityEpisodeBuildError(f"{label} must be lowercase SHA-256")
    return value


def _plain_absolute_file(path_value: Any, *, expected_sha256: str, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value or "\x00" in path_value:
        raise IdentityEpisodeBuildError(f"{label} must be a non-empty path")
    path = Path(path_value)
    if not path.is_absolute():
        raise IdentityEpisodeBuildError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IdentityEpisodeBuildError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise IdentityEpisodeBuildError(f"{label} must be a plain non-symlink file")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise IdentityEpisodeBuildError(f"{label} SHA-256 differs")
    return path.resolve()


def _run_media_tool(arguments: Sequence[str], *, label: str) -> bytes:
    executable = shutil.which(arguments[0])
    if executable is None:
        raise IdentityEpisodeBuildError(f"{arguments[0]} is required for {label}")
    command = [executable, *arguments[1:]]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=MEDIA_DECODE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise IdentityEpisodeBuildError(f"cannot run {label}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise IdentityEpisodeBuildError(f"{label} failed: {detail}")
    return completed.stdout


def _probe_video_dimensions(path: Path, *, label: str) -> tuple[int, int]:
    payload = _run_media_tool(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ),
        label=f"{label} ffprobe",
    )
    value = _decode_object(payload, context=f"{label} ffprobe JSON")
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise IdentityEpisodeBuildError(f"{label} must expose exactly one first video stream")
    width = streams[0].get("width")
    height = streams[0].get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width > MAXIMUM_FRAME_EDGE
        or height > MAXIMUM_FRAME_EDGE
    ):
        raise IdentityEpisodeBuildError(f"{label} dimensions are outside the closed contract")
    return width, height


def _perceptual_rgbq4_8x8(rgb: bytes, *, width: int, height: int) -> str:
    if len(rgb) != width * height * 3:
        raise IdentityEpisodeBuildError("RGB frame byte length differs from dimensions")
    fingerprint = bytearray()
    for grid_y in range(8):
        y = min(height - 1, ((2 * grid_y + 1) * height) // 16)
        for grid_x in range(8):
            x = min(width - 1, ((2 * grid_x + 1) * width) // 16)
            offset = (y * width + x) * 3
            fingerprint.extend(
                (rgb[offset] >> 4, rgb[offset + 1] >> 4, rgb[offset + 2] >> 4)
            )
    return bytes(fingerprint).hex()


def _perceptual_hamming(left: str, right: str) -> int:
    try:
        first = bytes.fromhex(left)
        second = bytes.fromhex(right)
    except ValueError as error:
        raise IdentityEpisodeBuildError("perceptual fingerprint is not hexadecimal") from error
    if len(first) != 8 * 8 * 3 or len(second) != len(first):
        raise IdentityEpisodeBuildError("perceptual fingerprint length differs")
    # ``int.bit_count`` is unavailable in the Python 3.7 environment used by
    # one of the repository's CPU preflight runners.
    return sum(bin(a ^ b).count("1") for a, b in zip(first, second))


def _decode_binary_ppm(path: Path) -> Optional[tuple[int, int, bytes]]:
    payload = path.read_bytes()
    if not payload.startswith(b"P6"):
        return None
    index = 0

    def token() -> bytes:
        nonlocal index
        while index < len(payload):
            if payload[index] == ord("#"):
                end = payload.find(b"\n", index)
                if end < 0:
                    raise IdentityEpisodeBuildError("unterminated PPM comment")
                index = end + 1
            elif payload[index] in b" \t\r\n":
                index += 1
            else:
                break
        start = index
        while index < len(payload) and payload[index] not in b" \t\r\n#":
            index += 1
        if start == index:
            raise IdentityEpisodeBuildError("invalid binary PPM header")
        return payload[start:index]

    magic = token()
    try:
        width = int(token())
        height = int(token())
        maximum = int(token())
    except ValueError as error:
        raise IdentityEpisodeBuildError("invalid binary PPM dimensions") from error
    if magic != b"P6" or maximum != 255 or width <= 0 or height <= 0:
        raise IdentityEpisodeBuildError("binary PPM contract differs")
    if index >= len(payload) or payload[index] not in b" \t\r\n":
        raise IdentityEpisodeBuildError("binary PPM lacks pixel separator")
    if payload[index] == ord("\r") and index + 1 < len(payload) and payload[index + 1] == ord("\n"):
        index += 2
    else:
        index += 1
    rgb = payload[index:]
    if len(rgb) != width * height * 3:
        raise IdentityEpisodeBuildError("binary PPM pixel payload length differs")
    return width, height, rgb


def _decode_static_rgb(path: Path, *, width: int, height: int, label: str) -> bytes:
    ppm = _decode_binary_ppm(path)
    if ppm is not None:
        observed_width, observed_height, rgb = ppm
        if (observed_width, observed_height) != (width, height):
            raise IdentityEpisodeBuildError(f"{label} dimensions differ from source video")
        return rgb
    observed = _probe_video_dimensions(path, label=label)
    if observed != (width, height):
        raise IdentityEpisodeBuildError(f"{label} dimensions differ from source video")
    rgb = _run_media_tool(
        (
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "2",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ),
        label=f"{label} RGB decode",
    )
    if len(rgb) != width * height * 3:
        raise IdentityEpisodeBuildError(
            f"{label} must decode to exactly one source-sized RGB frame"
        )
    return rgb


def _decode_source_video(path: Path, *, label: str) -> tuple[int, int, tuple[bytes, ...]]:
    width, height = _probe_video_dimensions(path, label=label)
    payload = _run_media_tool(
        (
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(MAXIMUM_DECODED_FRAMES + 1),
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ),
        label=f"{label} RGB decode",
    )
    frame_bytes = width * height * 3
    if (
        len(payload) == 0
        or len(payload) % frame_bytes != 0
        or len(payload) // frame_bytes > MAXIMUM_DECODED_FRAMES
    ):
        raise IdentityEpisodeBuildError(f"{label} decoded frame stream differs")
    frames = tuple(
        payload[offset : offset + frame_bytes]
        for offset in range(0, len(payload), frame_bytes)
    )
    return width, height, frames


def _media_decoder_receipt() -> Mapping[str, Any]:
    ffmpeg = _run_media_tool(("ffmpeg", "-version"), label="ffmpeg version")
    ffprobe = _run_media_tool(("ffprobe", "-version"), label="ffprobe version")
    value = {
        "ffmpeg_first_line": ffmpeg.decode("utf-8", errors="replace").splitlines()[0],
        "ffprobe_first_line": ffprobe.decode("utf-8", errors="replace").splitlines()[0],
        "decode": "first_video_stream_rgb24_vsync0",
        "perceptual_fingerprint": "rgbq4_nearest_8x8_v1",
        "minimum_near_duplicate_hamming": MINIMUM_NEAR_DUPLICATE_HAMMING,
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    frame_path: str
    frame_sha256: str
    decoded_rgb_sha256: str
    perceptual_rgbq4_8x8: str

    def model_ref(self) -> Mapping[str, Any]:
        # Index is retained for disjointness audits but is forbidden as a model
        # tensor or embedding.  The loader decodes only frame_path.
        return {
            "frame_index_audit_only": self.frame_index,
            "frame_path": self.frame_path,
            "frame_sha256": self.frame_sha256,
            "decoded_rgb_sha256": self.decoded_rgb_sha256,
            "perceptual_rgbq4_8x8": self.perceptual_rgbq4_8x8,
            "model_consumed_fields": ["decoded_rgb_pixels"],
        }


@dataclass(frozen=True)
class RawVideoRecord:
    video_id: str
    identity_id: str
    clip_id: str
    authority_assertion_id: str
    authority_assertion_digest: str
    split: str
    source_video_path: str
    source_video_sha256: str
    frame_width: int
    frame_height: int
    decoded_sequence_digest: str
    frames: tuple[FrameRecord, ...]
    row_digest: str


@dataclass(frozen=True)
class IdentityAuthorityAssertion:
    assertion_id: str
    video_id: str
    clip_id: str
    identity_id: str
    split: str
    source_video_sha256: str
    evidence_digest: str


@dataclass(frozen=True)
class IdentityAuthorityRelease:
    authority_id: str
    release_id: str
    evidence_policy: str
    file_sha256: str
    assertions: Mapping[str, IdentityAuthorityAssertion]


@dataclass(frozen=True)
class EpisodeBuildPayload:
    jsonl_bytes: bytes
    receipt_bytes: bytes
    receipt: Mapping[str, Any]
    output_jsonl: Path
    output_receipt: Path


def authority_evidence_digest(
    *,
    authority_id: str,
    release_id: str,
    evidence_policy: str,
    assertion_id: str,
    video_id: str,
    clip_id: str,
    identity_id: str,
    split: str,
    source_video_sha256: str,
) -> str:
    return object_sha256(
        {
            "schema_version": AUTHORITY_ASSERTION_SCHEMA,
            "authority_id": authority_id,
            "release_id": release_id,
            "evidence_policy": evidence_policy,
            "assertion_id": assertion_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "identity_id": identity_id,
            "split": split,
            "source_video_sha256": source_video_sha256,
        }
    )


def load_identity_authority(
    path: Path, *, expected_sha256: str
) -> IdentityAuthorityRelease:
    expected = _sha(expected_sha256, label="authority manifest expected SHA-256")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IdentityEpisodeBuildError(f"cannot stat authority manifest: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise IdentityEpisodeBuildError("authority manifest must be a plain non-symlink file")
    actual = file_sha256(path)
    if actual != expected:
        raise IdentityEpisodeBuildError("authority manifest SHA-256 differs from trust pin")
    raw = _decode_object(path.read_bytes(), context="identity authority manifest")
    if set(raw) != _AUTHORITY_FIELDS or raw.get("schema_version") != AUTHORITY_SCHEMA:
        raise IdentityEpisodeBuildError("identity authority field/schema closure differs")
    authority_id = _safe_id(raw.get("authority_id"), label="authority_id")
    release_id = _safe_id(raw.get("release_id"), label="release_id")
    evidence_policy = raw.get("evidence_policy")
    if evidence_policy not in _ALLOWED_EVIDENCE_POLICIES:
        raise IdentityEpisodeBuildError("identity authority evidence policy is not trusted")
    raw_assertions = raw.get("assertions")
    if not isinstance(raw_assertions, list) or not raw_assertions:
        raise IdentityEpisodeBuildError("identity authority requires assertions")
    assertions: dict[str, IdentityAuthorityAssertion] = {}
    video_ids: set[str] = set()
    clip_ids: set[str] = set()
    source_hashes: set[str] = set()
    for offset, item in enumerate(raw_assertions):
        if not isinstance(item, Mapping) or set(item) != _ASSERTION_FIELDS:
            raise IdentityEpisodeBuildError(
                f"identity authority assertion {offset} field closure differs"
            )
        if item.get("schema_version") != AUTHORITY_ASSERTION_SCHEMA:
            raise IdentityEpisodeBuildError("identity authority assertion schema differs")
        assertion_id = _safe_id(item.get("assertion_id"), label="assertion_id")
        video_id = _safe_id(item.get("video_id"), label="authority video_id")
        clip_id = _safe_id(item.get("clip_id"), label="authority clip_id")
        identity_id = _safe_id(item.get("identity_id"), label="authority identity_id")
        split = item.get("split")
        if split not in {"train", "validation", "test"}:
            raise IdentityEpisodeBuildError("authority split differs")
        source_sha = _sha(
            item.get("source_video_sha256"), label="authority source video SHA-256"
        )
        evidence_digest = _sha(
            item.get("evidence_digest"), label="authority evidence digest"
        )
        expected_evidence = authority_evidence_digest(
            authority_id=authority_id,
            release_id=release_id,
            evidence_policy=evidence_policy,
            assertion_id=assertion_id,
            video_id=video_id,
            clip_id=clip_id,
            identity_id=identity_id,
            split=split,
            source_video_sha256=source_sha,
        )
        if evidence_digest != expected_evidence:
            raise IdentityEpisodeBuildError("authority evidence digest differs")
        if (
            assertion_id in assertions
            or video_id in video_ids
            or clip_id in clip_ids
            or source_sha in source_hashes
        ):
            raise IdentityEpisodeBuildError(
                "authority assertion/video/clip/source coordinates must be unique"
            )
        assertions[assertion_id] = IdentityAuthorityAssertion(
            assertion_id=assertion_id,
            video_id=video_id,
            clip_id=clip_id,
            identity_id=identity_id,
            split=split,
            source_video_sha256=source_sha,
            evidence_digest=evidence_digest,
        )
        video_ids.add(video_id)
        clip_ids.add(clip_id)
        source_hashes.add(source_sha)
    return IdentityAuthorityRelease(
        authority_id=authority_id,
        release_id=release_id,
        evidence_policy=evidence_policy,
        file_sha256=actual,
        assertions=assertions,
    )


def _validate_no_forbidden_keys(value: Any, *, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
                raise IdentityEpisodeBuildError(f"forbidden semantic field at {path}.{key}")
            _validate_no_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_keys(child, path=f"{path}[{index}]")


def _parse_record(
    raw: Mapping[str, Any],
    *,
    row_number: int,
    authority: IdentityAuthorityRelease,
) -> RawVideoRecord:
    if set(raw) != _ROW_FIELDS:
        raise IdentityEpisodeBuildError(
            f"manifest row {row_number} field closure differs: {sorted(set(raw) ^ _ROW_FIELDS)}"
        )
    if raw.get("schema_version") != RAW_VIDEO_ROW_SCHEMA:
        raise IdentityEpisodeBuildError(f"manifest row {row_number} schema differs")
    # Validate unknown nested additions before consuming any paths.
    _validate_no_forbidden_keys(raw)
    video_id = _safe_id(raw.get("video_id"), label="video_id")
    assertion_id = _safe_id(
        raw.get("authority_assertion_id"), label="authority_assertion_id"
    )
    assertion = authority.assertions.get(assertion_id)
    if assertion is None:
        raise IdentityEpisodeBuildError(
            f"manifest row {row_number} lacks a trusted authority assertion"
        )
    split = raw.get("split")
    if split not in {"train", "validation", "test"}:
        raise IdentityEpisodeBuildError("split must be train/validation/test")
    source_sha = _sha(raw.get("source_video_sha256"), label="source video SHA-256")
    if (
        assertion.video_id != video_id
        or assertion.split != split
        or assertion.source_video_sha256 != source_sha
    ):
        raise IdentityEpisodeBuildError(
            f"manifest row {row_number} differs from its authority assertion"
        )
    source_path = _plain_absolute_file(
        raw.get("source_video_path"),
        expected_sha256=source_sha,
        label=f"{video_id} source video",
    )
    width, height, decoded_source_frames = _decode_source_video(
        source_path, label=f"{video_id} source video"
    )
    raw_frames = raw.get("frames")
    if (
        not isinstance(raw_frames, list)
        or len(raw_frames) < MINIMUM_FRAMES
        or len(raw_frames) != len(decoded_source_frames)
    ):
        raise IdentityEpisodeBuildError(
            f"{video_id} frame records must cover every decoded source frame"
        )
    frames: list[FrameRecord] = []
    for offset, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping) or set(raw_frame) != _FRAME_FIELDS:
            raise IdentityEpisodeBuildError(f"{video_id} frame {offset} field closure differs")
        index = raw_frame.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index != offset:
            raise IdentityEpisodeBuildError(f"{video_id} frame indices must be contiguous from zero")
        frame_sha = _sha(raw_frame.get("frame_sha256"), label="frame SHA-256")
        expected_rgb_sha = _sha(
            raw_frame.get("decoded_rgb_sha256"), label="decoded RGB SHA-256"
        )
        expected_perceptual = raw_frame.get("perceptual_rgbq4_8x8")
        if (
            not isinstance(expected_perceptual, str)
            or len(expected_perceptual) != 8 * 8 * 3 * 2
            or any(character not in "0123456789abcdef" for character in expected_perceptual)
        ):
            raise IdentityEpisodeBuildError(
                f"{video_id} frame {index} perceptual fingerprint differs"
            )
        frame_path = _plain_absolute_file(
            raw_frame.get("frame_path"),
            expected_sha256=frame_sha,
            label=f"{video_id} frame {index}",
        )
        frame_rgb = _decode_static_rgb(
            frame_path,
            width=width,
            height=height,
            label=f"{video_id} frame {index}",
        )
        actual_rgb_sha = bytes_sha256(frame_rgb)
        source_rgb_sha = bytes_sha256(decoded_source_frames[index])
        actual_perceptual = _perceptual_rgbq4_8x8(
            frame_rgb, width=width, height=height
        )
        if (
            actual_rgb_sha != expected_rgb_sha
            or actual_rgb_sha != source_rgb_sha
            or frame_rgb != decoded_source_frames[index]
            or actual_perceptual != expected_perceptual
        ):
            raise IdentityEpisodeBuildError(
                f"{video_id} frame {index} is not a source-derived RGB extraction"
            )
        frames.append(
            FrameRecord(
                index,
                str(frame_path),
                frame_sha,
                actual_rgb_sha,
                actual_perceptual,
            )
        )
    if (
        len({frame.frame_sha256 for frame in frames}) != len(frames)
        or len({frame.decoded_rgb_sha256 for frame in frames}) != len(frames)
    ):
        raise IdentityEpisodeBuildError(f"{video_id} contains duplicate decoded frames")
    decoded_sequence_digest = object_sha256(
        {
            "source_video_sha256": source_sha,
            "width": width,
            "height": height,
            "decoded_rgb_sha256": [frame.decoded_rgb_sha256 for frame in frames],
        }
    )
    unsigned = {
        "schema_version": RAW_VIDEO_ROW_SCHEMA,
        "video_id": video_id,
        "authority_assertion_id": assertion.assertion_id,
        "authority_assertion_digest": assertion.evidence_digest,
        "identity_id_from_authority": assertion.identity_id,
        "clip_id_from_authority": assertion.clip_id,
        "split": split,
        "source_video_path": str(source_path),
        "source_video_sha256": source_sha,
        "frame_width": width,
        "frame_height": height,
        "decoded_sequence_digest": decoded_sequence_digest,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "frame_path": frame.frame_path,
                "frame_sha256": frame.frame_sha256,
                "decoded_rgb_sha256": frame.decoded_rgb_sha256,
                "perceptual_rgbq4_8x8": frame.perceptual_rgbq4_8x8,
            }
            for frame in frames
        ],
    }
    return RawVideoRecord(
        video_id=video_id,
        identity_id=assertion.identity_id,
        clip_id=assertion.clip_id,
        authority_assertion_id=assertion.assertion_id,
        authority_assertion_digest=assertion.evidence_digest,
        split=split,
        source_video_path=str(source_path),
        source_video_sha256=source_sha,
        frame_width=width,
        frame_height=height,
        decoded_sequence_digest=decoded_sequence_digest,
        frames=tuple(frames),
        row_digest=object_sha256(unsigned),
    )


def load_manifest(
    path: Path, *, authority: IdentityAuthorityRelease
) -> tuple[bytes, tuple[RawVideoRecord, ...]]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IdentityEpisodeBuildError(f"cannot stat manifest: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise IdentityEpisodeBuildError("manifest must be a plain non-symlink file")
    payload = path.read_bytes()
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise IdentityEpisodeBuildError("manifest must be non-empty JSONL without blank rows")
    records = tuple(
        _parse_record(
            _decode_object(line, context=f"manifest row {index}"),
            row_number=index,
            authority=authority,
        )
        for index, line in enumerate(lines, start=1)
    )
    if len({record.video_id for record in records}) != len(records):
        raise IdentityEpisodeBuildError("video_id values must be unique")
    if len({record.source_video_sha256 for record in records}) != len(records):
        raise IdentityEpisodeBuildError("source videos must be content-distinct")
    if len({record.clip_id for record in records}) != len(records):
        raise IdentityEpisodeBuildError("authority clip_id values must be unique")
    if len({record.authority_assertion_id for record in records}) != len(records):
        raise IdentityEpisodeBuildError("authority assertions must bind one row each")
    if {record.authority_assertion_id for record in records} != set(authority.assertions):
        raise IdentityEpisodeBuildError(
            "manifest must consume the complete pinned authority release"
        )
    identity_splits: dict[str, str] = {}
    identity_counts: dict[str, int] = {}
    pixel_owners: dict[str, str] = {}
    for record in records:
        prior = identity_splits.setdefault(record.identity_id, record.split)
        if prior != record.split:
            raise IdentityEpisodeBuildError("one identity crosses dataset splits")
        identity_counts[record.identity_id] = identity_counts.get(record.identity_id, 0) + 1
        for frame in record.frames:
            owner = pixel_owners.setdefault(frame.decoded_rgb_sha256, record.identity_id)
            if owner != record.identity_id:
                raise IdentityEpisodeBuildError(
                    "decoded RGB frame collides across authority identities"
                )
    if any(count < MINIMUM_CLIP_PER_IDENTITY for count in identity_counts.values()):
        raise IdentityEpisodeBuildError(
            "every authority identity requires at least two distinct clips"
        )
    return payload, records


def _seeded_rng(*, seed: int, label: str) -> random.Random:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise IdentityEpisodeBuildError("seed must lie in [0,2^63)")
    digest = hashlib.sha256(f"{seed}\0{label}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _wrong_record(source: RawVideoRecord, records: Sequence[RawVideoRecord]) -> RawVideoRecord:
    candidates = sorted(
        (
            record
            for record in records
            if record.split == source.split
            and record.identity_id != source.identity_id
        ),
        key=lambda record: (record.identity_id, record.video_id, record.row_digest),
    )
    if not candidates:
        raise IdentityEpisodeBuildError(
            f"{source.video_id} has no same-split wrong identity"
        )
    index = int(source.row_digest[:16], 16) % len(candidates)
    return candidates[index]


def _correct_memory_record(
    source: RawVideoRecord,
    *,
    targets: Sequence[FrameRecord],
    records: Sequence[RawVideoRecord],
) -> tuple[RawVideoRecord, int]:
    """Choose an independently captured same-identity clip, fail closed on leakage."""

    target_rgb = {frame.decoded_rgb_sha256 for frame in targets}
    candidates: list[tuple[RawVideoRecord, int]] = []
    for record in sorted(
        records,
        key=lambda item: (item.identity_id, item.clip_id, item.video_id, item.row_digest),
    ):
        if (
            record.identity_id != source.identity_id
            or record.split != source.split
            or record.clip_id == source.clip_id
            or record.video_id == source.video_id
            or record.source_video_sha256 == source.source_video_sha256
        ):
            continue
        memory_rgb = {frame.decoded_rgb_sha256 for frame in record.frames}
        if target_rgb & memory_rgb:
            continue
        minimum_hamming = min(
            _perceptual_hamming(target.perceptual_rgbq4_8x8, memory.perceptual_rgbq4_8x8)
            for target in targets
            for memory in record.frames
        )
        if minimum_hamming < MINIMUM_NEAR_DUPLICATE_HAMMING:
            continue
        candidates.append((record, minimum_hamming))
    if not candidates:
        raise IdentityEpisodeBuildError(
            f"{source.video_id} has no clip-disjoint, non-near-duplicate same-identity memory"
        )
    index = int(source.row_digest[16:32], 16) % len(candidates)
    return candidates[index]


def _frame_refs(frames: Sequence[FrameRecord]) -> list[Mapping[str, Any]]:
    return [frame.model_ref() for frame in frames]


def _build_episode(
    source: RawVideoRecord,
    *,
    records: Sequence[RawVideoRecord],
    seed: int,
    target_frame_count: int,
    drop_fraction: float,
) -> Mapping[str, Any]:
    rng = _seeded_rng(seed=seed, label=source.row_digest)
    target_indices = set(rng.sample(range(len(source.frames)), target_frame_count))
    targets = tuple(source.frames[index] for index in sorted(target_indices))
    memory_record, minimum_target_memory_hamming = _correct_memory_record(
        source, targets=targets, records=records
    )
    memory = sorted(
        memory_record.frames,
        key=lambda frame: (frame.frame_sha256, frame.frame_path),
    )
    if len(memory) < 2:
        raise IdentityEpisodeBuildError("memory must retain at least two frames")
    target_rgb_hashes = {frame.decoded_rgb_sha256 for frame in targets}
    if target_rgb_hashes & {frame.decoded_rgb_sha256 for frame in memory}:
        raise IdentityEpisodeBuildError("heldout target aliases correct memory")

    shuffled = list(memory)
    rng.shuffle(shuffled)
    if shuffled == memory:
        shuffled = shuffled[1:] + shuffled[:1]
    drop_count = max(1, int(math.ceil(len(memory) * drop_fraction)))
    if len(memory) - drop_count < 2:
        raise IdentityEpisodeBuildError("drop view would contain fewer than two frames")
    dropped_indices = set(rng.sample(range(len(memory)), drop_count))
    dropped = [frame for index, frame in enumerate(memory) if index not in dropped_indices]
    resampled = [memory[rng.randrange(len(memory))] for _ in range(len(memory))]

    wrong = _wrong_record(source, records)
    wrong_memory = sorted(wrong.frames, key=lambda frame: (frame.frame_sha256, frame.frame_path))
    # Equal memory cardinality prevents identity classification by token count.
    wrong_rng = _seeded_rng(seed=seed, label=f"wrong\0{source.row_digest}\0{wrong.row_digest}")
    wrong_view = [wrong_memory[wrong_rng.randrange(len(wrong_memory))] for _ in range(len(memory))]
    if target_rgb_hashes & {frame.decoded_rgb_sha256 for frame in wrong_view}:
        raise IdentityEpisodeBuildError("wrong identity contains a heldout target duplicate")

    unsigned = {
        "schema_version": EPISODE_SCHEMA,
        "episode_id": f"irb-{source.row_digest[:20]}",
        "split": source.split,
        "source": {
            "video_id": source.video_id,
            "identity_id": source.identity_id,
            "clip_id": source.clip_id,
            "authority_assertion_id": source.authority_assertion_id,
            "authority_assertion_digest": source.authority_assertion_digest,
            "source_video_path": source.source_video_path,
            "source_video_sha256": source.source_video_sha256,
            "raw_manifest_row_digest": source.row_digest,
            "full_frame_count": len(source.frames),
        },
        "heldout_targets": {
            "frames": _frame_refs(targets),
            "disjoint_from_every_correct_memory_view": True,
            "training_role": "raw_rgb_target_frame_recovery_only",
        },
        "correct_identity_memory": {
            "video_id": memory_record.video_id,
            "identity_id": memory_record.identity_id,
            "clip_id": memory_record.clip_id,
            "authority_assertion_id": memory_record.authority_assertion_id,
            "authority_assertion_digest": memory_record.authority_assertion_digest,
            "source_video_sha256": memory_record.source_video_sha256,
            "raw_manifest_row_digest": memory_record.row_digest,
            "canonical_orderless_set": _frame_refs(memory),
            "shuffle_view": _frame_refs(shuffled),
            "drop_view": _frame_refs(dropped),
            "resample_with_replacement_view": _frame_refs(resampled),
            "frame_indices_are_model_inputs": False,
            "frame_paths_are_decoded_to_rgb_then_discarded": True,
            "set_order_is_semantically_void": True,
            "target_and_memory_clip_ids_are_distinct": True,
            "target_and_memory_source_video_hashes_are_distinct": True,
            "minimum_target_memory_perceptual_hamming": minimum_target_memory_hamming,
            "minimum_allowed_perceptual_hamming": MINIMUM_NEAR_DUPLICATE_HAMMING,
        },
        "wrong_identity_memory": {
            "video_id": wrong.video_id,
            "identity_id": wrong.identity_id,
            "clip_id": wrong.clip_id,
            "authority_assertion_id": wrong.authority_assertion_id,
            "authority_assertion_digest": wrong.authority_assertion_digest,
            "raw_manifest_row_digest": wrong.row_digest,
            "matched_memory_count": len(wrong_view),
            "frames": _frame_refs(wrong_view),
        },
        "pretraining_contract": {
            "edited_target_present": False,
            "instruction_present": False,
            "action_label_present": False,
            "source_caption_present": False,
            "mask_track_flow_pose_present": False,
            "temporal_order_consumed_by_model": False,
            "target_memory_clip_disjoint": True,
            "target_memory_exact_rgb_collision": False,
            "target_memory_near_duplicate_below_threshold": False,
            "losses": [
                "heldout_target_recovery",
                "correct_vs_wrong_identity_contrast",
                "shuffle_drop_resample_consistency",
            ],
        },
    }
    return {**unsigned, "episode_digest": object_sha256(unsigned)}


def build_payload(
    *,
    manifest: Path,
    authority_manifest: Path,
    authority_manifest_sha256: str,
    output_jsonl: Path,
    output_receipt: Path,
    seed: int = DEFAULT_SEED,
    target_frame_count: int = DEFAULT_TARGET_FRAMES,
    drop_fraction: float = DEFAULT_DROP_FRACTION,
) -> EpisodeBuildPayload:
    if (
        isinstance(target_frame_count, bool)
        or not isinstance(target_frame_count, int)
        or target_frame_count <= 0
    ):
        raise IdentityEpisodeBuildError("target_frame_count must be positive")
    if (
        isinstance(drop_fraction, bool)
        or not isinstance(drop_fraction, (int, float))
        or not math.isfinite(float(drop_fraction))
        or not 0.0 < float(drop_fraction) < 0.5
    ):
        raise IdentityEpisodeBuildError("drop_fraction must be finite in (0,0.5)")
    authority = load_identity_authority(
        authority_manifest, expected_sha256=authority_manifest_sha256
    )
    manifest_payload, records = load_manifest(manifest, authority=authority)
    if any(len(record.frames) < target_frame_count for record in records):
        raise IdentityEpisodeBuildError("target frame count exceeds a source clip")
    episodes = tuple(
        _build_episode(
            record,
            records=records,
            seed=seed,
            target_frame_count=target_frame_count,
            drop_fraction=float(drop_fraction),
        )
        for record in sorted(records, key=lambda item: (item.split, item.video_id))
    )
    jsonl = b"".join(canonical_json_bytes(episode) + b"\n" for episode in episodes)
    unsigned_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": bytes_sha256(manifest_payload),
        "authority_manifest_path": str(authority_manifest.resolve()),
        "authority_manifest_sha256": authority.file_sha256,
        "authority_id": authority.authority_id,
        "authority_release_id": authority.release_id,
        "authority_evidence_policy": authority.evidence_policy,
        "authority_assertion_count": len(authority.assertions),
        "episode_jsonl_path": str(output_jsonl.resolve()),
        "episode_jsonl_sha256": bytes_sha256(jsonl),
        "episode_count": len(episodes),
        "episode_digests": [episode["episode_digest"] for episode in episodes],
        "seed": seed,
        "target_frame_count": target_frame_count,
        "drop_fraction_hex": float(drop_fraction).hex(),
        "raw_video_only": True,
        "memory_target_clip_disjoint": True,
        "memory_target_exact_rgb_collision_rejected": True,
        "memory_target_near_duplicate_hamming_threshold": MINIMUM_NEAR_DUPLICATE_HAMMING,
        "cross_identity_exact_rgb_collision_rejected": True,
        "wrong_identity_same_split_and_matched_memory_count": True,
        "media_decoder": _media_decoder_receipt(),
        "model_receives_frame_indices": False,
        "model_receives_temporal_order": False,
        "edited_targets_or_action_annotations": False,
        "gpu_validated": False,
        "training_authorized": False,
    }
    receipt = {**unsigned_receipt, "receipt_digest": object_sha256(unsigned_receipt)}
    return EpisodeBuildPayload(
        jsonl_bytes=jsonl,
        receipt_bytes=canonical_json_bytes(receipt) + b"\n",
        receipt=receipt,
        output_jsonl=output_jsonl,
        output_receipt=output_receipt,
    )


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise IdentityEpisodeBuildError(f"refusing to replace existing output: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority-manifest", type=Path, required=True)
    parser.add_argument("--authority-manifest-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-frame-count", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--drop-fraction", type=float, default=DEFAULT_DROP_FRACTION)
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(
        manifest=args.manifest,
        authority_manifest=args.authority_manifest,
        authority_manifest_sha256=args.authority_manifest_sha256,
        output_jsonl=args.output_jsonl,
        output_receipt=args.output_receipt,
        seed=args.seed,
        target_frame_count=args.target_frame_count,
        drop_fraction=args.drop_fraction,
    )
    if args.publish:
        _publish_create_only(payload.output_jsonl, payload.jsonl_bytes)
        # Receipt is the ready marker and is always published last.
        _publish_create_only(payload.output_receipt, payload.receipt_bytes)
    print(payload.receipt["receipt_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
