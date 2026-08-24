#!/usr/bin/env python3
"""Fail-closed acceptance gate for a fresh case01 bone-removed-v2 asset.

This program does not remove an object, run a renderer, train a model, or use
a GPU.  It audits a separately produced candidate and admits it only as an
*input asset*.  The old dilate-3 bidirectional-interpolation asset is
deliberately impossible to promote through this gate.

The producer must publish two versions of the same candidate:

* a lossless FFV1 canonical RGB authority, used for exact pixel accounting;
* an H.264/yuv420p convenience video, used only for human playback.

All canonical pixels outside a fresh, all-frame bone-plus-cast-shadow support
must be byte-identical to the decoded source; every pixel inside that support
must be byte-identical to a separately hash-bound raw generator donor.  The
source dog plus an eight-pixel guard must also be byte-identical.  Automatic
texture and seam tests are diagnostics against the known smooth, bone-shaped
interpolation scar; they do not establish semantic absence.  Semantic bone
absence, scar absence, dog identity and background preservation additionally
require two independent, blinded, all-81-frame reviewers.  The H.264 delivery
is explicitly lossy human-viewing convenience, never carrier/context identity
authority.  This contract requires every identity-sensitive downstream
consumer to read canonical FFV1, not the H.264 convenience file; it does not
claim that any not-yet-audited downstream consumer actually complied.
"""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib


RECEIPT_SCHEMA = "bernini-case01-bone-removed-v2-producer-receipt-v1"
OBSERVATION_SCHEMA = "bernini-case01-bone-removed-v2-observations-v1"
REPORT_SCHEMA = "bernini-case01-bone-removed-v2-acceptance-report-v1"
ATTEMPT_SCHEMA = "bernini-case01-bone-removed-v2-create-only-attempt-v1"
PUBLICATION_SCHEMA = "bernini-case01-bone-removed-v2-create-only-publication-v1"
TREE_MANIFEST_SCHEMA = "bernini-case01-bone-removed-v2-authority-tree-manifest-v1"
SUPPORT_REVIEW_SCHEMA = "bernini-case01-bone-shadow-support-dual-review-v2"
SUPPORT_REVIEW_STATUS = (
    "PASS_ALL_81_NATIVE_SUPPORT_REVIEW_TWO_EXTERNAL_REVIEWERS"
)
SUPPORT_PACKET_SCHEMA = "bernini-case01-bone-contact-support-review-packet-v1"
SUPPORT_PACKET_STATUS = "UNSIGNED_CANDIDATE_HOLD_PENDING_TWO_EXTERNAL_REVIEWS"
EXTERNAL_REVIEW_SCHEMA = (
    "bernini-case01-bone-contact-support-external-review-template-v1"
)
EXTERNAL_SIGNATURE_KIND = "OPAQUE_EXTERNAL_TRUST_ROOT_EVIDENCE_V1"
SUPPORT_PACKET_MANIFEST_SHA256 = (
    "91c2a3bb101621edc6b93b96cbb9af75369fc4c5474c5d61c5395620046b4435"
)
SUPPORT_PACKET_MANIFEST_SIZE = 260_175
SUPPORT_PACKET_PREMANIFEST_DIGEST = (
    "6374275b26be8c9e0f6f86cbcde4bca1ca6ad46cd0db9d7a7cdaee76f1cbf36e"
)
EXTERNAL_REVIEW_INSTRUCTIONS = (
    "Copy this template outside the immutable candidate directory before filling it.",
    "Inspect the native 704x736 overlay and unscaled source/overlay crop for every frame.",
    "Reject or request edits if the original bone, contact shadow, halo, or adjacent ground is outside support.",
    "Reject if support touches the dog or would edit dog identity pixels.",
    "Do not infer PASS from geometry or from the other reviewer's decision.",
)
EXTERNAL_INDEPENDENCE_KEYS = (
    "human_visual_review_performed",
    "independent_from_packet_producer",
    "independent_from_generator",
    "independent_from_other_reviewer",
    "other_reviewer_ballot_not_seen_before_finalization",
)
SUPPORT_REVIEW_CLAIM_LIMITS = {
    "input_support_gate_only": True,
    "external_receipt_bytes_and_ballots_structurally_replayed": True,
    "reviewer_identity_verified_by_generator": False,
    "reviewer_affiliation_verified_by_generator": False,
    "reviewer_independence_verified_by_generator": False,
    "reviewer_authorship_cryptographically_proven": False,
    "signature_evidence_cryptographically_verified": False,
    "visual_review_reperformed_by_generator": False,
    "cleanplate_generated": False,
    "renderer_or_vace_run_authorized": False,
    "gpu_execution_performed": False,
    "training_performed": False,
    "scientific_claim_authorized": False,
}
CASE_ID = "case01"
IID = "288545b9c031491a"
ROLE = "aux_bone_removed_source_v2"
MODEL_AUTHORITY_ROLES = (
    "python_runtime_tree",
    "vace_checkpoint_tree",
    "vace_source_tree",
)

WIDTH = 704
HEIGHT = 736
RAW_DONOR_WIDTH = 624
RAW_DONOR_HEIGHT = 640
FPS = 25
FRAME_COUNT = 81
FRAME_PIXELS = WIDTH * HEIGHT
RGB_FRAME_BYTES = FRAME_PIXELS * 3
PRECANVAS_PIXELS = RAW_DONOR_WIDTH * RAW_DONOR_HEIGHT
PRECANVAS_RGB_FRAME_BYTES = PRECANVAS_PIXELS * 3

SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
SOURCE_SIZE = 10_887_043
SAM2_RECEIPT_SHA256 = "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
SAM2_RECEIPT_SIZE = 22_160
FFMPEG_SHA256 = "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
FFMPEG_SIZE = 79_826_272
FFMPEG_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
FFPROBE_SHA256 = "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
FFPROBE_SIZE = 216_841
FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
GENERATOR_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/methods/bernini_action_editing/"
    "generate_case01_bone_removed_v2_vace_v1.py"
)
GENERATOR_SHA256 = "f6dc4edb5ea3da03e14dd00399a800c3af545379bd0030aeab0fc8e2a205ce86"
GENERATOR_SIZE = 85_957

# These are rejected lineage, never eligible v2 authority.
OLD_BONE_REMOVED_VIDEO_SHA256 = (
    "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9"
)
OLD_REMOVAL_SUPPORT_SHA256 = (
    "83818847c61b506370edc9a4a6cae8b9fe2bb06681f5425e40d9fb6e850fd554"
)
FORBIDDEN_ALGORITHM_IDS = {
    "per_frame_SAM2_bone_mask_dilate3_bidirectional_boundary_interpolation",
    "deterministic_ffmpeg_removelogo_spatial_interpolation_r4",
}

MIN_SUPPORT_DILATION = 8
DOG_GUARD_DILATION = 8
CONTEXT_RING_DILATION = 12
MAX_SUPPORT_TO_BONE_AREA_RATIO = 4.0
MIN_BONE_CHANGED_FRACTION_PER_FRAME = 0.98
MIN_BONE_SOURCE_RESIDUAL_P10 = 18.0
MIN_TEXTURE_RATIO_P10 = 0.55
MIN_TEXTURE_RATIO_MEDIAN = 0.75
MAX_TEXTURE_RATIO_MEDIAN = 1.80
MAX_LOW_TEXTURE_FRAMES = 4
LOW_TEXTURE_RATIO = 0.50
MAX_SEAM_RATIO_MEDIAN = 2.00
MAX_SEAM_RATIO = 3.00
MAX_DELIVERY_RGB_MAD_MEAN = 6.0
MAX_DELIVERY_RGB_MAD_FRAME = 10.0

SHA256_HEX = set("0123456789abcdef")
BINARY_THRESHOLD_TABLE = bytes(0 if value < 128 else 255 for value in range(256))


def _float32(value: float) -> float:
    """Round one scalar exactly as an IEEE-754 little-endian float32."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _prepared_source_float32(value: int) -> float:
    # Mirrors torch uint8 -> float().div_(127.5).sub_(1.) with a float32
    # rounding boundary after each tensor operation.
    converted = _float32(float(value))
    divided = _float32(converted / _float32(127.5))
    return _float32(divided - _float32(1.0))


def _prepared_mask_float32(value: int) -> float:
    # VACE normalizes the decoded mask as video first, then takes channel 0
    # and applies clamp((x + 1) / 2, 0, 1), all in float32.
    normalized = _prepared_source_float32(value)
    shifted = _float32(normalized + _float32(1.0))
    divided = _float32(shifted / _float32(2.0))
    return min(_float32(1.0), max(_float32(0.0), divided))


PREPARED_SOURCE_FLOAT32_CHARMAP = {
    value: struct.pack("<f", _prepared_source_float32(value))
    for value in range(256)
}
PREPARED_MASK_FLOAT32_CHARMAP = {
    value: struct.pack("<f", _prepared_mask_float32(value))
    for value in range(256)
}


class BoneRemovedV2Error(RuntimeError):
    """A v2 authority or acceptance invariant differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoneRemovedV2Error(message)


def _update_prepared_float32_digest(
    digest: Any,
    values: bytes | bytearray,
    charmap: Mapping[int, bytes],
    label: str,
) -> None:
    """Expand uint8 values to canonical little-endian float32 bytes."""

    chunk_size = 1024 * 1024
    view = memoryview(values)
    for offset in range(0, len(view), chunk_size):
        chunk = bytes(view[offset : offset + chunk_size])
        try:
            encoded, consumed = codecs.charmap_encode(
                chunk.decode("latin1"),
                "strict",
                charmap,
            )
        except (UnicodeError, TypeError) as error:
            raise BoneRemovedV2Error(
                f"cannot replay VACE prepared {label} float32 bytes"
            ) from error
        _require(consumed == len(chunk), f"short VACE prepared {label} replay")
        _require(
            len(encoded) == len(chunk) * 4,
            f"VACE prepared {label} byte width differs",
        )
        digest.update(encoded)


def _prepared_tensor_digests(
    source_planes_cthw: Sequence[bytes | bytearray],
    mask_plane_cthw: bytes | bytearray,
) -> Mapping[str, str]:
    """Replay the exact VACE float32 source/mask tensor byte digests.

    Each source plane is already laid out frame -> height -> width, so
    concatenating R, G, B planes yields the C,T,H,W contiguous byte order
    produced by ``permute(0,3,1,2).transpose(0,1).contiguous()``.  The mask
    is VACE's first decoded channel in T,H,W order.
    """

    _require(len(source_planes_cthw) == 3, "VACE prepared source needs three planes")
    plane_sizes = {len(plane) for plane in source_planes_cthw}
    _require(len(plane_sizes) == 1, "VACE prepared source plane sizes differ")
    source_plane_size = next(iter(plane_sizes))
    _require(
        source_plane_size == len(mask_plane_cthw),
        "VACE prepared source/mask element counts differ",
    )
    source_digest = hashlib.sha256()
    for plane in source_planes_cthw:
        _update_prepared_float32_digest(
            source_digest,
            plane,
            PREPARED_SOURCE_FLOAT32_CHARMAP,
            "source tensor",
        )
    mask_digest = hashlib.sha256()
    _update_prepared_float32_digest(
        mask_digest,
        mask_plane_cthw,
        PREPARED_MASK_FLOAT32_CHARMAP,
        "mask tensor",
    )
    return {
        "source_tensor_sha256": source_digest.hexdigest(),
        "mask_tensor_sha256": mask_digest.hexdigest(),
    }


def _verify_prepared_tensor_replay(
    trace: Mapping[str, Any],
    source_planes_cthw: Sequence[bytes | bytearray],
    mask_plane_cthw: bytes | bytearray,
) -> Mapping[str, Any]:
    replay = _prepared_tensor_digests(source_planes_cthw, mask_plane_cthw)
    for name, replay_key in (
        ("source_tensor", "source_tensor_sha256"),
        ("mask_tensor", "mask_tensor_sha256"),
    ):
        expected = replay[replay_key]
        tensor = trace[name]
        _require(
            tensor["pre_generate_sha256"] == expected,
            f"VACE prepared {name} digest differs from lossless precanvas replay",
        )
        _require(
            tensor["post_generate_sha256"] == expected,
            f"VACE prepared {name} post-generation digest differs from replay",
        )
    return {
        "digest_definition": trace["digest_definition"],
        "source_tensor_sha256": replay["source_tensor_sha256"],
        "mask_tensor_sha256": replay["mask_tensor_sha256"],
        "source_shape": list(trace["source_tensor"]["shape"]),
        "mask_shape": list(trace["mask_tensor"]["shape"]),
        "independently_replayed_from_lossless_precanvas": True,
        "matches_pre_and_post_generation_trace": True,
    }


def _stat_identity(row: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _digest_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


class HeldFile:
    """A verified inode kept open through every consumer and rejoined at close."""

    def __init__(
        self,
        *,
        path: Path,
        descriptor: int,
        identity: tuple[int, int, int, int, int, int, int],
        sha256: str,
        size: int,
        label: str,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.sha256 = sha256
        self.size = size
        self.label = label
        self.closed = False

    @property
    def proc_path(self) -> str:
        _require(sys.platform.startswith("linux"), "held-FD execution requires Linux")
        return f"/proc/self/fd/{self.descriptor}"

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self.descriptor,)

    def close_verified(self) -> None:
        if self.closed:
            return
        try:
            after = os.fstat(self.descriptor)
            named_after = self.path.lstat()
            digest, size = _digest_descriptor(self.descriptor)
            _require(
                _stat_identity(after) == self.identity == _stat_identity(named_after),
                f"held file or named join changed during use: {self.label}",
            )
            _require(
                digest == self.sha256 and size == self.size,
                f"held file bytes changed during use: {self.label}",
            )
        finally:
            os.close(self.descriptor)
            self.closed = True


def _close_held_files(files: Iterable[HeldFile]) -> None:
    first_error: BaseException | None = None
    for held in reversed(tuple(files)):
        try:
            held.close_verified()
        except BaseException as error:  # close every authority even when one differs
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


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
        raise BoneRemovedV2Error("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_tuple = tuple(expected)
    wanted = set(expected_tuple)
    _require(
        len(expected_tuple) == len(wanted),
        f"{label} contract contains duplicate expected keys",
    )
    observed = set(value)
    _require(
        observed == wanted,
        f"{label} key closure differs: missing={sorted(wanted-observed)} "
        f"extra={sorted(observed-wanted)}",
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_HEX


def _regular_file(path_value: str | Path) -> Path:
    path = Path(path_value)
    _require(path.is_absolute(), f"path is not absolute: {path}")
    _require(os.path.normpath(str(path)) == str(path), f"path is not canonical: {path}")
    _require(not path.is_symlink(), f"path is a symlink: {path}")
    resolved = path.resolve(strict=True)
    _require(resolved == path, f"path traverses a symlink: {path}")
    return resolved


def stable_file(
    path_value: str | Path,
    *,
    require_nlink1: bool = False,
) -> tuple[str, int]:
    path = _regular_file(path_value)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        _require(
            not require_nlink1 or before.st_nlink == 1,
            f"file is not single-link create-only authority: {path}",
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        identity = lambda row: (  # noqa: E731
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        _require(
            identity(before) == identity(after) == identity(named_after),
            f"file or named join changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _stable_bytes(path_value: str | Path) -> bytes:
    """Read one regular file through the descriptor whose identity is checked."""

    path = _regular_file(path_value)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        identity = lambda row: (  # noqa: E731
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        _require(
            identity(before) == identity(after) == identity(named_after),
            f"file or named join changed while reading: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_row(value: Mapping[str, Any], label: str) -> None:
    _exact_keys(value, ("path", "sha256", "size"), label)
    _require(isinstance(value["path"], str) and value["path"], f"{label} path differs")
    _require(_is_sha256(value["sha256"]), f"{label} SHA-256 differs")
    _require(type(value["size"]) is int and value["size"] > 0, f"{label} size differs")


def _verify_file_row(
    value: Mapping[str, Any],
    label: str,
    *,
    require_nlink1: bool = False,
) -> Path:
    _file_row(value, label)
    path = _regular_file(value["path"])
    digest, size = stable_file(path, require_nlink1=require_nlink1)
    _require(digest == value["sha256"], f"{label} SHA-256 does not match file")
    _require(size == value["size"], f"{label} size does not match file")
    return path


def _open_held_file_row(
    value: Mapping[str, Any],
    label: str,
    *,
    require_nlink1: bool = False,
    require_executable: bool = False,
) -> HeldFile:
    _file_row(value, label)
    path = _regular_file(value["path"])
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        _require(
            not require_nlink1 or before.st_nlink == 1,
            f"file is not single-link create-only authority: {path}",
        )
        _require(
            not require_executable or bool(before.st_mode & 0o111),
            f"pinned executable mode differs: {label}",
        )
        digest, size = _digest_descriptor(descriptor)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        _require(
            _stat_identity(before)
            == _stat_identity(after)
            == _stat_identity(named_after),
            f"file or named join changed while opening held authority: {path}",
        )
        _require(digest == value["sha256"], f"{label} SHA-256 does not match file")
        _require(size == value["size"], f"{label} size does not match file")
        return HeldFile(
            path=path,
            descriptor=descriptor,
            identity=_stat_identity(after),
            sha256=digest,
            size=size,
            label=label,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_model_authorities(generator: Mapping[str, Any]) -> tuple[Path, ...]:
    """Replay every declared model/config/checkpoint byte authority."""

    verified: list[Path] = []
    for index, model in enumerate(generator["model_authorities"]):
        file_fields = {key: model[key] for key in ("path", "sha256", "size")}
        verified.append(
            _verify_file_row(
                file_fields,
                f"generator model authority {index} ({model['role']})",
            )
        )
    return tuple(verified)


def _validate_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    _require(_is_sha256(value.get(field)), f"{label} digest differs")
    payload = dict(value)
    observed = payload.pop(field)
    _require(observed == object_sha256(payload), f"{label} digest mismatch")


def validate_producer_receipt(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "role",
            "source",
            "mask_authority",
            "media_tools",
            "acceptance_contract",
            "generator",
            "support",
            "canonical_candidate",
            "delivery_candidate",
            "construction_audit",
            "create_only_authority",
            "claim_limits",
            "receipt_digest",
        ),
        "producer receipt",
    )
    _require(value["schema_version"] == RECEIPT_SCHEMA, "producer receipt schema differs")
    _require(value["status"] == "COMPLETE_CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE", "producer status differs")
    _require(value["case_id"] == CASE_ID and value["iid"] == IID, "producer case differs")
    _require(value["role"] == ROLE, "producer role differs")

    source = value["source"]
    _require(isinstance(source, dict), "source row missing")
    _file_row(source, "source")
    _require(source["sha256"] == SOURCE_SHA256 and source["size"] == SOURCE_SIZE, "source authority differs")

    masks = value["mask_authority"]
    _exact_keys(
        masks,
        ("receipt", "bone_mask_count", "dog_mask_count", "all_81_masks_hash_bound"),
        "mask authority",
    )
    _require(isinstance(masks["receipt"], dict), "mask receipt row missing")
    _file_row(masks["receipt"], "mask receipt")
    _require(
        masks["receipt"]["sha256"] == SAM2_RECEIPT_SHA256
        and masks["receipt"]["size"] == SAM2_RECEIPT_SIZE,
        "SAM2 receipt authority differs",
    )
    _require(masks["bone_mask_count"] == 81 and masks["dog_mask_count"] == 81, "mask count differs")
    _require(masks["all_81_masks_hash_bound"] is True, "all mask hashes are not bound")

    media_tools = value["media_tools"]
    _exact_keys(media_tools, ("ffmpeg", "ffprobe"), "media tools")
    _require(isinstance(media_tools["ffmpeg"], dict), "ffmpeg row missing")
    _require(isinstance(media_tools["ffprobe"], dict), "ffprobe row missing")
    _file_row(media_tools["ffmpeg"], "ffmpeg")
    _file_row(media_tools["ffprobe"], "ffprobe")
    _require(
        media_tools["ffmpeg"]["path"] == FFMPEG_PATH
        and
        media_tools["ffmpeg"]["sha256"] == FFMPEG_SHA256
        and media_tools["ffmpeg"]["size"] == FFMPEG_SIZE,
        "ffmpeg authority differs",
    )
    _require(
        media_tools["ffprobe"]["path"] == FFPROBE_PATH
        and
        media_tools["ffprobe"]["sha256"] == FFPROBE_SHA256
        and media_tools["ffprobe"]["size"] == FFPROBE_SIZE,
        "ffprobe authority differs",
    )
    _require(isinstance(value["acceptance_contract"], dict), "acceptance contract row missing")
    _file_row(value["acceptance_contract"], "acceptance contract")

    generator = value["generator"]
    _exact_keys(
        generator,
        (
            "program",
            "model_authorities",
            "authority_replay",
            "raw_support_donor",
            "algorithm_id",
            "deterministic",
            "seed",
            "generative_inpainting_inside_support",
            "whole_frame_generation",
            "outside_support_hard_composite_source_rgb",
            "uses_bidirectional_boundary_interpolation",
            "uses_ffmpeg_removelogo",
            "training_performed",
            "optimizer_updates",
        ),
        "generator",
    )
    _require(isinstance(generator["program"], dict), "generator program row missing")
    _file_row(generator["program"], "generator program")
    _require(
        generator["program"]
        == {
            "path": GENERATOR_PATH,
            "sha256": GENERATOR_SHA256,
            "size": GENERATOR_SIZE,
        },
        "generator program authority differs from frozen producer",
    )
    model_authorities = generator["model_authorities"]
    _require(isinstance(model_authorities, list), "generator model authorities differ")
    seen_model_roles: set[str] = set()
    seen_model_paths: set[str] = set()
    for index, model in enumerate(model_authorities):
        _require(isinstance(model, dict), f"generator model authority {index} differs")
        _exact_keys(
            model,
            ("role", "path", "sha256", "size"),
            f"generator model authority {index}",
        )
        role = model["role"]
        _require(
            isinstance(role, str)
            and role in MODEL_AUTHORITY_ROLES
            and role not in seen_model_roles,
            f"generator model authority role {index} differs",
        )
        seen_model_roles.add(role)
        file_fields = {key: model[key] for key in ("path", "sha256", "size")}
        _file_row(file_fields, f"generator model authority {index}")
        _require(
            model["path"] not in seen_model_paths,
            "generator model authority path repeats",
        )
        seen_model_paths.add(model["path"])
    _require(
        isinstance(generator["algorithm_id"], str)
        and generator["algorithm_id"]
        and generator["algorithm_id"] not in FORBIDDEN_ALGORITHM_IDS,
        "generator algorithm is missing or forbidden old interpolation lineage",
    )
    _require(generator["deterministic"] is True, "generation is not deterministic")
    _require(type(generator["seed"]) is int, "generation seed differs")
    _require(type(generator["generative_inpainting_inside_support"]) is bool, "inpainting declaration differs")
    if generator["generative_inpainting_inside_support"]:
        _require(
            seen_model_roles == set(MODEL_AUTHORITY_ROLES),
            "generative inpainting lacks the exact VACE/runtime tree authorities",
        )
    else:
        _require(
            not model_authorities,
            "non-generative algorithm declares model/checkpoint authority",
        )
    authority_replay = generator["authority_replay"]
    _exact_keys(
        authority_replay,
        ("before_generation_digest", "after_generation_digest", "unchanged"),
        "generator authority replay",
    )
    _require(
        _is_sha256(authority_replay["before_generation_digest"])
        and authority_replay["before_generation_digest"]
        == authority_replay["after_generation_digest"],
        "generator before/after authority replay differs",
    )
    _require(authority_replay["unchanged"] is True, "generator authority tree changed")
    raw_donor = generator["raw_support_donor"]
    _exact_keys(
        raw_donor,
        (
            "video",
            "frame_count",
            "index_mapping",
            "normalization",
            "used_only_inside_support",
            "source_or_identity_authority",
        ),
        "raw support donor",
    )
    _require(isinstance(raw_donor["video"], dict), "raw support donor video row missing")
    _file_row(raw_donor["video"], "raw support donor video")
    _require(
        raw_donor["video"]["sha256"] != OLD_BONE_REMOVED_VIDEO_SHA256,
        "old bone-removed video is forbidden as raw donor",
    )
    _require(raw_donor["frame_count"] == FRAME_COUNT, "raw donor frame count differs")
    _require(
        raw_donor["index_mapping"]
        == "exact_frame_index_0_through_80_ignore_container_fps_timestamps",
        "raw donor frame-index mapping differs",
    )
    normalization = raw_donor["normalization"]
    _exact_keys(
        normalization,
        (
            "algorithm",
            "source_width",
            "source_height",
            "precanvas_width",
            "precanvas_height",
            "fit_width",
            "fit_height",
            "pad_left",
            "pad_right",
            "pad_top",
            "pad_bottom",
            "source_fit_kernel",
            "support_fit_kernel",
            "pad_value",
            "inverse_crop_xyxy",
            "inverse_resize_kernel",
            "python_hash_seed",
            "frame_indices",
            "prepare_source_trace",
            "precanvas_authority_scope",
            "processed_cache_authority_scope",
            "precanvas_source_video",
            "precanvas_mask_video",
            "processed_source_video",
            "processed_mask_video",
        ),
        "raw donor normalization",
    )
    _require(
        normalization["algorithm"]
        == "vace_precanvas_fitpad624x640_inverse_crop_lanczos_v1"
        and normalization["source_width"] == WIDTH
        and normalization["source_height"] == HEIGHT
        and normalization["precanvas_width"] == RAW_DONOR_WIDTH
        and normalization["precanvas_height"] == RAW_DONOR_HEIGHT
        and normalization["fit_width"] == 612
        and normalization["fit_height"] == 640
        and normalization["pad_left"] == 6
        and normalization["pad_right"] == 6
        and normalization["pad_top"] == 0
        and normalization["pad_bottom"] == 0
        and normalization["source_fit_kernel"] == "lanczos"
        and normalization["support_fit_kernel"] == "nearest"
        and normalization["pad_value"] == 0
        and normalization["inverse_crop_xyxy"] == [6, 0, 618, 640]
        and normalization["inverse_resize_kernel"] == "lanczos"
        and normalization["python_hash_seed"] == 20260822
        and normalization["frame_indices"] == list(range(FRAME_COUNT)),
        "raw donor normalization differs",
    )
    _require(
        normalization["precanvas_authority_scope"]
        == "lossless_vace_input_authority"
        and normalization["processed_cache_authority_scope"]
        == "nonauthoritative_codec_diagnostic_only",
        "raw donor preprocessing authority scopes differ",
    )
    trace = normalization["prepare_source_trace"]
    _exact_keys(
        trace,
        (
            "frame_indices",
            "resize_crop_applied",
            "digest_definition",
            "source_tensor",
            "mask_tensor",
        ),
        "VACE prepare_source trace",
    )
    _require(trace["frame_indices"] == list(range(FRAME_COUNT)), "VACE frame IDs differ")
    _require(trace["resize_crop_applied"] is False, "VACE processor resized or cropped precanvas")
    _require(
        trace["digest_definition"]
        == "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
        "VACE prepared tensor digest definition differs",
    )
    for name, shape in (
        ("source_tensor", [3, FRAME_COUNT, RAW_DONOR_HEIGHT, RAW_DONOR_WIDTH]),
        ("mask_tensor", [1, FRAME_COUNT, RAW_DONOR_HEIGHT, RAW_DONOR_WIDTH]),
    ):
        tensor = trace[name]
        _exact_keys(
            tensor,
            (
                "shape",
                "dtype",
                "pre_generate_sha256",
                "post_generate_sha256",
                "unchanged",
            ),
            f"VACE prepared {name}",
        )
        _require(tensor["shape"] == shape, f"VACE prepared {name} shape differs")
        _require(tensor["dtype"] == "float32", f"VACE prepared {name} dtype differs")
        _require(
            _is_sha256(tensor["pre_generate_sha256"])
            and tensor["pre_generate_sha256"]
            == tensor["post_generate_sha256"],
            f"VACE prepared {name} before/after digest differs",
        )
        _require(tensor["unchanged"] is True, f"VACE prepared {name} changed during generation")
    normalization_paths: set[str] = set()
    for name in (
        "precanvas_source_video",
        "precanvas_mask_video",
        "processed_source_video",
        "processed_mask_video",
    ):
        row = normalization[name]
        _require(isinstance(row, dict), f"raw donor normalization {name} row missing")
        _file_row(row, f"raw donor normalization {name}")
        _require(row["path"] not in normalization_paths, "raw donor normalization path repeats")
        normalization_paths.add(row["path"])
    _require(
        raw_donor["used_only_inside_support"] is True,
        "raw donor is not confined to support",
    )
    _require(
        raw_donor["source_or_identity_authority"] is False,
        "raw donor improperly claims source/identity authority",
    )
    _require(generator["whole_frame_generation"] is False, "whole-frame generation is forbidden")
    _require(generator["outside_support_hard_composite_source_rgb"] is True, "source RGB hard composite is absent")
    _require(generator["uses_bidirectional_boundary_interpolation"] is False, "old boundary interpolation is forbidden")
    _require(generator["uses_ffmpeg_removelogo"] is False, "ffmpeg removelogo lineage is forbidden")
    _require(generator["training_performed"] is False and generator["optimizer_updates"] == 0, "candidate generation training differs")

    support = value["support"]
    _exact_keys(
        support,
        (
            "tube",
            "definition",
            "frame_count",
            "contains_bone_and_cast_shadow_all_frames",
            "all_81_frames_manually_reviewed",
            "old_dilate3_tube_reused",
            "review_receipt",
            "frame_masks",
        ),
        "support",
    )
    _require(isinstance(support["tube"], dict), "support tube row missing")
    _file_row(support["tube"], "support tube")
    _require(support["tube"]["sha256"] != OLD_REMOVAL_SUPPORT_SHA256, "old dilate3 support is forbidden")
    _require(support["definition"] == "per_frame_bone_plus_cast_shadow_support_v2", "support definition differs")
    _require(support["frame_count"] == 81, "support frame count differs")
    _require(support["contains_bone_and_cast_shadow_all_frames"] is True, "support does not cover bone and cast shadow")
    _require(support["all_81_frames_manually_reviewed"] is True, "support lacks all-frame review")
    _require(support["old_dilate3_tube_reused"] is False, "old support reuse is forbidden")
    _require(isinstance(support["review_receipt"], dict), "support review receipt row missing")
    _file_row(support["review_receipt"], "support review receipt")
    frame_masks = support["frame_masks"]
    _require(
        isinstance(frame_masks, list) and len(frame_masks) == FRAME_COUNT,
        "support frame-mask row count differs",
    )
    seen_support_mask_paths: set[str] = set()
    for frame_index, row in enumerate(frame_masks):
        _require(isinstance(row, dict), f"support frame-mask row {frame_index} differs")
        _exact_keys(
            row,
            ("frame_index", "path", "sha256", "size"),
            f"support frame-mask row {frame_index}",
        )
        _require(
            type(row["frame_index"]) is int
            and row["frame_index"] == frame_index,
            "support frame-mask order differs",
        )
        file_fields = {key: row[key] for key in ("path", "sha256", "size")}
        _file_row(file_fields, f"support frame-mask row {frame_index}")
        _require(
            row["path"] not in seen_support_mask_paths,
            "support frame-mask path repeats",
        )
        seen_support_mask_paths.add(row["path"])

    canonical = value["canonical_candidate"]
    _exact_keys(
        canonical,
        (
            "video",
            "codec",
            "lossless",
            "stored_pixel_format",
            "decoded_pixel_format",
            "frame_count",
        ),
        "canonical candidate",
    )
    _require(isinstance(canonical["video"], dict), "canonical candidate video row missing")
    _file_row(canonical["video"], "canonical candidate video")
    _require(canonical["video"]["sha256"] != OLD_BONE_REMOVED_VIDEO_SHA256, "old bone-removed video is forbidden")
    _require(canonical["codec"] == "ffv1" and canonical["lossless"] is True, "canonical candidate is not FFV1 lossless")
    _require(
        canonical["stored_pixel_format"] == "bgr0",
        "canonical stored pixel format is not FFV1-compatible 8-bit RGB bgr0",
    )
    _require(canonical["decoded_pixel_format"] == "rgb24" and canonical["frame_count"] == 81, "canonical decode contract differs")

    delivery = value["delivery_candidate"]
    _exact_keys(
        delivery,
        (
            "video",
            "codec",
            "pixel_format",
            "frame_count",
            "derived_only_from_canonical",
            "authority_scope",
            "identity_authority",
            "canonical_is_identity_authority",
        ),
        "delivery candidate",
    )
    _require(isinstance(delivery["video"], dict), "delivery candidate video row missing")
    _file_row(delivery["video"], "delivery candidate video")
    _require(delivery["video"]["sha256"] != OLD_BONE_REMOVED_VIDEO_SHA256, "old delivery video is forbidden")
    _require(delivery["codec"] == "h264" and delivery["pixel_format"] == "yuv420p", "delivery codec differs")
    _require(delivery["frame_count"] == 81 and delivery["derived_only_from_canonical"] is True, "delivery derivation differs")
    _require(
        delivery["authority_scope"] == "human_playback_convenience_lossy_transport_only"
        and delivery["identity_authority"] is False
        and delivery["canonical_is_identity_authority"] is True,
        "delivery authority scope differs",
    )

    audit = value["construction_audit"]
    _exact_keys(
        audit,
        (
            "frame_count",
            "outside_support_changed_pixels",
            "dog_guard_changed_pixels",
            "support_pixels_not_equal_raw_donor",
            "full_frame_pixel_scan",
            "source_bone_changed_fraction_minimum",
        ),
        "construction audit",
    )
    _require(audit["frame_count"] == 81 and audit["full_frame_pixel_scan"] is True, "construction scan differs")
    _require(audit["outside_support_changed_pixels"] == 0, "producer reports out-of-support changes")
    _require(audit["dog_guard_changed_pixels"] == 0, "producer reports dog-guard changes")
    _require(
        audit["support_pixels_not_equal_raw_donor"] == 0,
        "producer reports canonical/support pixels unequal to raw donor",
    )
    _require(
        type(audit["source_bone_changed_fraction_minimum"]) in (int, float)
        and float(audit["source_bone_changed_fraction_minimum"]) >= MIN_BONE_CHANGED_FRACTION_PER_FRAME,
        "producer source-bone change fraction differs",
    )

    create = value["create_only_authority"]
    _exact_keys(
        create,
        (
            "controller_program",
            "attempt_receipt",
            "publication_receipt",
            "controller_distinct_from_generator",
            "fresh_root",
            "existing_path_reused",
            "overwrite_performed",
            "atomic_publish",
            "staging_removed_after_publish",
        ),
        "create-only authority",
    )
    for name in ("controller_program", "attempt_receipt", "publication_receipt"):
        _require(isinstance(create[name], dict), f"create-only {name} row missing")
        _file_row(create[name], f"create-only {name}")
    _require(
        create["controller_program"]["sha256"]
        != generator["program"]["sha256"],
        "create-only controller is not independent from generator",
    )
    _require(
        create["controller_distinct_from_generator"] is True,
        "create-only controller independence is not declared",
    )
    _require(create["fresh_root"] is True, "candidate root was not fresh")
    _require(create["existing_path_reused"] is False, "existing path was reused")
    _require(create["overwrite_performed"] is False, "candidate path was overwritten")
    _require(create["atomic_publish"] is True, "candidate was not atomically published")
    _require(create["staging_removed_after_publish"] is True, "staging remains after publish")

    limits = value["claim_limits"]
    _exact_keys(
        limits,
        (
            "input_asset_authority_only",
            "renderer_inference_performed",
            "renderer_result_claim_authorized",
            "scientific_claim_authorized",
            "semantic_absence_requires_human_review",
            "downstream_identity_sensitive_consumption_requires_canonical",
            "actual_downstream_consumer_verified",
            "generation_execution_lineage_verified",
        ),
        "producer claim limits",
    )
    _require(limits["input_asset_authority_only"] is True, "producer scope differs")
    _require(limits["renderer_inference_performed"] is False, "renderer inference claim differs")
    _require(limits["renderer_result_claim_authorized"] is False, "renderer result claim differs")
    _require(limits["scientific_claim_authorized"] is False, "scientific claim differs")
    _require(limits["semantic_absence_requires_human_review"] is True, "human semantic review is not required")
    _require(
        limits["downstream_identity_sensitive_consumption_requires_canonical"]
        is True,
        "producer contract does not require canonical downstream consumption",
    )
    _require(
        limits["actual_downstream_consumer_verified"] is False,
        "producer improperly claims an actual downstream consumer was verified",
    )
    _require(
        limits["generation_execution_lineage_verified"] is False,
        "self-reported generation execution lineage cannot be promoted",
    )
    _validate_digest(value, "receipt_digest", "producer receipt")


def validate_observations(
    value: Mapping[str, Any],
    *,
    candidate_sha256: str,
    delivery_sha256: str,
    support_sha256: str,
) -> None:
    _exact_keys(
        value,
        (
            "schema_version",
            "case_id",
            "iid",
            "candidate_sha256",
            "support_sha256",
            "blinding",
            "review_protocol",
            "reviewers",
            "claim_limits",
            "observation_digest",
        ),
        "observations",
    )
    _require(value["schema_version"] == OBSERVATION_SCHEMA, "observation schema differs")
    _require(value["case_id"] == CASE_ID and value["iid"] == IID, "observation case differs")
    _require(value["candidate_sha256"] == candidate_sha256, "observation candidate binding differs")
    _require(value["support_sha256"] == support_sha256, "observation support binding differs")
    blinding = value["blinding"]
    _exact_keys(blinding, ("candidate_id_randomized", "arm_name_hidden", "reviewers_independent"), "blinding")
    _require(all(blinding[key] is True for key in blinding), "review was not blinded and independent")

    protocol = value["review_protocol"]
    _exact_keys(
        protocol,
        (
            "evidence_source",
            "canonical_candidate_sha256",
            "delivery_candidate_sha256",
            "support_sha256",
            "source_sha256",
            "decoded_frame_indices",
            "direct_canonical_decode_reviewed",
            "direct_delivery_playback_reviewed",
            "native_resolution_support_crop_reviewed_all_frames",
            "mask_outline_hidden_during_scar_ballot",
            "convenience_surfaces_used_as_authority",
        ),
        "review protocol",
    )
    _require(
        protocol["evidence_source"]
        == "direct_hash_bound_candidate_decode_not_pre_rendered_surfaces",
        "review evidence source differs",
    )
    _require(
        protocol["canonical_candidate_sha256"] == candidate_sha256
        and protocol["delivery_candidate_sha256"] == delivery_sha256
        and protocol["support_sha256"] == support_sha256
        and protocol["source_sha256"] == SOURCE_SHA256,
        "review protocol hash binding differs",
    )
    _require(
        protocol["decoded_frame_indices"] == list(range(FRAME_COUNT)),
        "review protocol is not exact ordered all-81",
    )
    _require(protocol["direct_canonical_decode_reviewed"] is True, "canonical candidate was not directly reviewed")
    _require(protocol["direct_delivery_playback_reviewed"] is True, "delivery candidate was not directly reviewed")
    _require(
        protocol["native_resolution_support_crop_reviewed_all_frames"] is True,
        "native-resolution support crops were not reviewed",
    )
    _require(
        protocol["mask_outline_hidden_during_scar_ballot"] is True,
        "scar ballot was visually cued by the mask outline",
    )
    _require(
        protocol["convenience_surfaces_used_as_authority"] is False,
        "pre-rendered convenience surfaces cannot be review authority",
    )

    reviewers = value["reviewers"]
    _require(isinstance(reviewers, list) and len(reviewers) >= 2, "at least two reviewers are required")
    reviewer_ids: set[str] = set()
    for index, reviewer in enumerate(reviewers):
        _require(isinstance(reviewer, dict), f"reviewer {index} differs")
        _exact_keys(
            reviewer,
            (
                "reviewer_id",
                "all_81_frames_reviewed",
                "support_bone_shadow_coverage",
                "bone_absence",
                "bone_shaped_scar_absence",
                "seam_absence",
                "texture_collapse_absence",
                "temporal_flicker_absence",
                "cast_shadow_absence",
                "dog_identity_preservation",
                "background_identity_preservation",
            ),
            f"reviewer {index}",
        )
        reviewer_id = reviewer["reviewer_id"]
        _require(isinstance(reviewer_id, str) and reviewer_id and reviewer_id not in reviewer_ids, "reviewer identities differ")
        reviewer_ids.add(reviewer_id)
        _require(reviewer["all_81_frames_reviewed"] is True, "review is not all-frame")
        for gate_name in (
            "support_bone_shadow_coverage",
            "bone_absence",
            "bone_shaped_scar_absence",
            "seam_absence",
            "texture_collapse_absence",
            "temporal_flicker_absence",
            "cast_shadow_absence",
            "dog_identity_preservation",
            "background_identity_preservation",
        ):
            gate = reviewer[gate_name]
            _exact_keys(gate, ("status", "failure_frames", "note"), f"reviewer {index} {gate_name}")
            _require(gate["status"] == "PASS", f"reviewer {index} did not pass {gate_name}")
            _require(gate["failure_frames"] == [], f"reviewer {index} reports failure frames for {gate_name}")
            _require(isinstance(gate["note"], str) and gate["note"], f"reviewer {index} note missing for {gate_name}")

    limits = value["claim_limits"]
    _exact_keys(limits, ("input_asset_review_only", "renderer_result_reviewed", "scientific_claim_authorized"), "observation claim limits")
    _require(limits["input_asset_review_only"] is True, "observation scope differs")
    _require(limits["renderer_result_reviewed"] is False, "observation reviewed renderer output")
    _require(limits["scientific_claim_authorized"] is False, "observation authorizes science claim")
    _validate_digest(value, "observation_digest", "observations")


def _canonical_absolute_unresolved(path_value: Any, label: str) -> Path:
    _require(isinstance(path_value, str) and path_value, f"{label} path differs")
    path = Path(path_value)
    _require(path.is_absolute(), f"{label} path is not absolute")
    _require(os.path.normpath(str(path)) == str(path), f"{label} path is not canonical")
    return path


def _verify_exact_published_tree(
    final_root_value: str | Path,
    asset_paths: Mapping[str, Path],
) -> tuple[str, ...]:
    """Require the actual final tree to contain exactly the three published files."""

    _require(
        set(asset_paths) == {"support", "canonical_candidate", "delivery_candidate"},
        "published tree asset labels differ",
    )
    if isinstance(final_root_value, Path):
        final_root_value = str(final_root_value)
    final_root = _canonical_absolute_unresolved(final_root_value, "published final root")
    _require(not final_root.is_symlink(), "published final root is a symlink")
    resolved_root = final_root.resolve(strict=True)
    _require(resolved_root == final_root and final_root.is_dir(), "published final root differs")

    expected_files: set[Path] = set()
    expected_directories: set[Path] = set()
    for label, raw_path in asset_paths.items():
        asset_path = _regular_file(raw_path)
        try:
            relative = asset_path.relative_to(final_root)
        except ValueError as error:
            raise BoneRemovedV2Error(f"published asset escapes final root: {label}") from error
        _require(relative.parts and relative != Path("."), f"published asset relative path differs: {label}")
        _require(relative not in expected_files, "published assets share one path")
        expected_files.add(relative)
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent)
            parent = parent.parent
    _require(
        not (expected_files & expected_directories),
        "published file/directory authority collides",
    )

    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()

    def visit(directory: Path, relative_parent: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            relative = relative_parent / entry.name
            observed = entry.stat(follow_symlinks=False)
            _require(not entry.is_symlink(), f"published tree contains symlink: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                _require(
                    relative in expected_directories,
                    f"published tree contains extra directory: {relative}",
                )
                observed_directories.add(relative)
                visit(Path(entry.path), relative)
            elif stat.S_ISREG(observed.st_mode):
                _require(
                    relative in expected_files,
                    f"published tree contains extra file: {relative}",
                )
                _require(
                    observed.st_nlink == 1,
                    f"published asset is not single-link create-only authority: {relative}",
                )
                observed_files.add(relative)
            else:
                raise BoneRemovedV2Error(
                    f"published tree contains non-regular entry: {relative}"
                )

    visit(final_root, Path("."))
    _require(observed_files == expected_files, "published tree file inventory differs")
    _require(
        observed_directories == expected_directories,
        "published tree directory inventory differs",
    )
    return tuple(sorted(path.as_posix() for path in observed_files))


def validate_create_only_receipts(
    attempt: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    producer: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replay two independently written controller receipts and cross-bind them."""

    _exact_keys(
        attempt,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "attempt_token",
            "controller_program_sha256",
            "generator_program_sha256",
            "model_authorities_digest",
            "final_root",
            "staging_root",
            "preflight",
            "attempt_digest",
        ),
        "create-only attempt receipt",
    )
    _require(attempt["schema_version"] == ATTEMPT_SCHEMA, "attempt receipt schema differs")
    _require(attempt["status"] == "RESERVED_FRESH_BEFORE_GENERATION", "attempt receipt status differs")
    _require(attempt["case_id"] == CASE_ID and attempt["iid"] == IID, "attempt receipt case differs")
    _require(_is_sha256(attempt["attempt_token"]), "attempt token differs")
    _require(
        attempt["controller_program_sha256"]
        == producer["create_only_authority"]["controller_program"]["sha256"],
        "attempt controller binding differs",
    )
    _require(
        attempt["generator_program_sha256"]
        == producer["generator"]["program"]["sha256"],
        "attempt generator binding differs",
    )
    _require(
        attempt["model_authorities_digest"]
        == object_sha256(producer["generator"]["model_authorities"]),
        "attempt model authority binding differs",
    )
    final_root = _canonical_absolute_unresolved(attempt["final_root"], "attempt final root")
    staging_root = _canonical_absolute_unresolved(attempt["staging_root"], "attempt staging root")
    _require(final_root != staging_root, "attempt final and staging roots coincide")
    preflight = attempt["preflight"]
    _exact_keys(
        preflight,
        (
            "performed_before_generation",
            "final_root_absent",
            "staging_root_absent",
            "all_target_paths_absent",
            "reservation_create_only",
        ),
        "create-only preflight",
    )
    _require(all(preflight[key] is True for key in preflight), "create-only preflight did not pass")
    _validate_digest(attempt, "attempt_digest", "create-only attempt receipt")

    _exact_keys(
        publication,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "attempt_token",
            "controller_program_sha256",
            "final_root",
            "staging_root",
            "published_assets",
            "publication",
            "publication_digest",
        ),
        "create-only publication receipt",
    )
    _require(publication["schema_version"] == PUBLICATION_SCHEMA, "publication receipt schema differs")
    _require(publication["status"] == "PUBLISHED_FRESH_NO_REPLACE", "publication receipt status differs")
    _require(publication["case_id"] == CASE_ID and publication["iid"] == IID, "publication receipt case differs")
    _require(publication["attempt_token"] == attempt["attempt_token"], "publication attempt token differs")
    _require(
        publication["controller_program_sha256"]
        == producer["create_only_authority"]["controller_program"]["sha256"],
        "publication controller binding differs",
    )
    _require(publication["final_root"] == str(final_root), "publication final root differs")
    _require(publication["staging_root"] == str(staging_root), "publication staging root differs")
    assets = publication["published_assets"]
    _exact_keys(
        assets,
        ("support", "canonical_candidate", "delivery_candidate"),
        "published assets",
    )
    expected_assets = {
        "support": producer["support"]["tube"],
        "canonical_candidate": producer["canonical_candidate"]["video"],
        "delivery_candidate": producer["delivery_candidate"]["video"],
    }
    for name, expected in expected_assets.items():
        row = assets[name]
        _require(isinstance(row, dict), f"published asset row missing: {name}")
        _file_row(row, f"published asset {name}")
        _require(row == expected, f"published asset binding differs: {name}")
        asset_path = _canonical_absolute_unresolved(row["path"], f"published asset {name}")
        try:
            asset_path.relative_to(final_root)
        except ValueError as error:
            raise BoneRemovedV2Error(f"published asset escapes final root: {name}") from error
    publication_facts = publication["publication"]
    _exact_keys(
        publication_facts,
        (
            "atomic_rename_noreplace",
            "final_root_absent_before_publish",
            "overwrite_performed",
            "staging_removed_after_publish",
            "published_tree_regular_nonsymlink_nlink1",
            "directory_fsync_performed",
        ),
        "publication facts",
    )
    _require(publication_facts["atomic_rename_noreplace"] is True, "rename-noreplace was not used")
    _require(publication_facts["final_root_absent_before_publish"] is True, "final root was not fresh")
    _require(publication_facts["overwrite_performed"] is False, "publication overwrote a path")
    _require(publication_facts["staging_removed_after_publish"] is True, "publication staging remains")
    _require(
        not os.path.lexists(staging_root),
        "actual publication staging path remains",
    )
    _require(
        publication_facts["published_tree_regular_nonsymlink_nlink1"] is True,
        "published tree topology differs",
    )
    _require(publication_facts["directory_fsync_performed"] is True, "publication directory was not fsynced")
    _validate_digest(publication, "publication_digest", "create-only publication receipt")
    return {
        "attempt_token": attempt["attempt_token"],
        "final_root": str(final_root),
        "staging_root": str(staging_root),
    }


def _parse_canonical_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BoneRemovedV2Error(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise BoneRemovedV2Error(f"non-finite JSON constant in {label}: {value}")

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BoneRemovedV2Error(f"invalid JSON: {label}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {label}")
    _require(
        payload == canonical_json_bytes(value) + b"\n",
        f"JSON bytes are not canonical with one trailing LF: {label}",
    )
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    return _parse_canonical_json(_stable_bytes(path), label)


def _load_json_file_row(
    value: Mapping[str, Any],
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    _file_row(value, label)
    path = _regular_file(value["path"])
    payload = _stable_bytes(path)
    _require(
        hashlib.sha256(payload).hexdigest() == value["sha256"],
        f"{label} SHA-256 does not match file",
    )
    _require(len(payload) == value["size"], f"{label} size does not match file")
    return path, _parse_canonical_json(payload, label)


def _manifest_relative_path(value: Any, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value and "\\" not in value, f"{label} differs")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and str(relative) == value
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} is not canonical relative POSIX",
    )
    return relative


def _replay_authority_tree_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    expected_role: str,
) -> Mapping[str, Any]:
    _exact_keys(
        manifest,
        (
            "schema_version",
            "authority_role",
            "inventory_policy",
            "tree_root",
            "entries",
            "file_count",
            "total_bytes",
            "tree_digest",
            "manifest_digest",
        ),
        f"authority tree manifest {expected_role}",
    )
    _require(
        manifest["schema_version"] == TREE_MANIFEST_SCHEMA,
        f"authority tree manifest schema differs: {expected_role}",
    )
    _require(
        manifest["authority_role"] == expected_role
        and expected_role in MODEL_AUTHORITY_ROLES,
        f"authority tree manifest role differs: {expected_role}",
    )
    _require(
        manifest["inventory_policy"]
        == "exact_recursive_regular_nonsymlink_nlink1",
        f"authority tree inventory policy differs: {expected_role}",
    )
    tree_root = _canonical_absolute_unresolved(
        manifest["tree_root"],
        f"authority tree root {expected_role}",
    )
    _require(not tree_root.is_symlink(), f"authority tree root is symlink: {expected_role}")
    _require(
        tree_root.resolve(strict=True) == tree_root and tree_root.is_dir(),
        f"authority tree root differs: {expected_role}",
    )
    try:
        manifest_path.relative_to(tree_root)
    except ValueError:
        pass
    else:
        raise BoneRemovedV2Error(
            f"authority manifest must be outside its tree root: {expected_role}"
        )

    entries = manifest["entries"]
    _require(isinstance(entries, list) and entries, f"authority entries missing: {expected_role}")
    expected_files: dict[str, Mapping[str, Any]] = {}
    ordered_paths: list[str] = []
    for index, row in enumerate(entries):
        _require(isinstance(row, dict), f"authority entry differs: {expected_role}:{index}")
        _exact_keys(
            row,
            ("relative_path", "sha256", "size"),
            f"authority entry {expected_role}:{index}",
        )
        relative = _manifest_relative_path(
            row["relative_path"],
            f"authority relative path {expected_role}:{index}",
        )
        _require(_is_sha256(row["sha256"]), f"authority SHA-256 differs: {expected_role}:{index}")
        _require(
            type(row["size"]) is int and row["size"] >= 0,
            f"authority size differs: {expected_role}:{index}",
        )
        relative_text = relative.as_posix()
        _require(relative_text not in expected_files, f"authority path repeats: {expected_role}")
        expected_files[relative_text] = row
        ordered_paths.append(relative_text)
    _require(
        ordered_paths == sorted(ordered_paths),
        f"authority entries are not path-sorted: {expected_role}",
    )
    _require(
        manifest["file_count"] == len(entries),
        f"authority file count differs: {expected_role}",
    )
    _require(
        manifest["total_bytes"] == sum(row["size"] for row in entries),
        f"authority total bytes differs: {expected_role}",
    )
    _require(
        manifest["tree_digest"] == object_sha256(entries),
        f"authority tree digest differs: {expected_role}",
    )
    _validate_digest(
        manifest,
        "manifest_digest",
        f"authority tree manifest {expected_role}",
    )

    observed_files: set[str] = set()

    def visit(directory: Path, relative_parent: PurePosixPath | None) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
        for child in children:
            child_stat = child.stat(follow_symlinks=False)
            _require(not child.is_symlink(), f"authority tree contains symlink: {expected_role}")
            relative = (
                PurePosixPath(child.name)
                if relative_parent is None
                else relative_parent / child.name
            )
            relative_text = relative.as_posix()
            if stat.S_ISDIR(child_stat.st_mode):
                visit(Path(child.path), relative)
            elif stat.S_ISREG(child_stat.st_mode):
                _require(
                    child_stat.st_nlink == 1,
                    f"authority tree file is not nlink1: {expected_role}:{relative_text}",
                )
                _require(
                    relative_text in expected_files,
                    f"authority tree has extra file: {expected_role}:{relative_text}",
                )
                row = expected_files[relative_text]
                digest, size = stable_file(Path(child.path), require_nlink1=True)
                _require(
                    digest == row["sha256"] and size == row["size"],
                    f"authority tree file differs: {expected_role}:{relative_text}",
                )
                observed_files.add(relative_text)
            else:
                raise BoneRemovedV2Error(
                    f"authority tree contains special entry: {expected_role}:{relative_text}"
                )

    visit(tree_root, None)
    _require(
        observed_files == set(expected_files),
        f"authority tree file inventory differs: {expected_role}",
    )
    return {
        "role": expected_role,
        "tree_digest": manifest["tree_digest"],
        "file_count": len(entries),
        "total_bytes": manifest["total_bytes"],
    }


def _replay_model_authority_manifests(generator: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    replay_rows: list[Mapping[str, Any]] = []
    report_rows: list[Mapping[str, Any]] = []
    for model in generator["model_authorities"]:
        file_fields = {key: model[key] for key in ("path", "sha256", "size")}
        manifest_path, manifest = _load_json_file_row(
            file_fields,
            f"model authority manifest {model['role']}",
        )
        replay = _replay_authority_tree_manifest(
            manifest,
            manifest_path=manifest_path,
            expected_role=model["role"],
        )
        replay_rows.append(
            {
                "role": model["role"],
                "manifest_sha256": model["sha256"],
                "tree_digest": replay["tree_digest"],
            }
        )
        report_rows.append(replay)
    replay_rows.sort(key=lambda row: row["role"])
    replay_digest = object_sha256(replay_rows)
    declared = generator["authority_replay"]
    _require(
        replay_digest == declared["before_generation_digest"]
        == declared["after_generation_digest"],
        "actual authority tree replay differs from before/after generation binding",
    )
    return sorted(report_rows, key=lambda row: row["role"])


def _same_file_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("sha256") == right.get("sha256")
        and left.get("size") == right.get("size")
    )


def _replay_support_packet_manifest(
    candidate_packet: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    _require(type(candidate_packet) is dict, "candidate packet differs")
    _exact_keys(
        candidate_packet,
        ("manifest", "sha256sums", "premanifest_output_tree_digest"),
        "candidate packet",
    )
    manifest_row = candidate_packet["manifest"]
    manifest_path = _verify_file_row(
        manifest_row,
        "candidate packet manifest",
        require_nlink1=True,
    )
    _require(
        (manifest_row["sha256"], manifest_row["size"])
        == (SUPPORT_PACKET_MANIFEST_SHA256, SUPPORT_PACKET_MANIFEST_SIZE),
        "candidate packet manifest authority differs",
    )
    loaded_manifest_path, manifest = _load_json_file_row(
        manifest_row,
        "candidate packet manifest",
    )
    _require(
        loaded_manifest_path == manifest_path,
        "candidate packet manifest path changed during replay",
    )
    _verify_file_row(
        manifest_row,
        "candidate packet manifest",
        require_nlink1=True,
    )
    _exact_keys(
        manifest,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "fps",
            "frame_count",
            "image_size_wh",
            "candidate_is_review_passed",
            "contact_shadow_visual_coverage",
            "derivation",
            "negative_evidence",
            "authority",
            "review_gate",
            "claim_limits",
            "frames",
            "premanifest_output_tree",
            "premanifest_output_tree_digest",
        ),
        "candidate packet manifest",
    )
    _require(
        manifest["schema_version"] == SUPPORT_PACKET_SCHEMA
        and manifest["status"] == SUPPORT_PACKET_STATUS
        and (manifest["case_id"], manifest["iid"]) == (CASE_ID, IID),
        "candidate packet manifest identity differs",
    )
    _require(
        type(manifest["fps"]) is int
        and manifest["fps"] == FPS
        and type(manifest["frame_count"]) is int
        and manifest["frame_count"] == FRAME_COUNT
        and manifest["image_size_wh"] == [WIDTH, HEIGHT]
        and manifest["candidate_is_review_passed"] is False
        and manifest["contact_shadow_visual_coverage"]
        == "PENDING_TWO_EXTERNAL_REVIEWS",
        "candidate packet manifest geometry/status differs",
    )
    _require(
        candidate_packet["premanifest_output_tree_digest"]
        == manifest["premanifest_output_tree_digest"]
        == SUPPORT_PACKET_PREMANIFEST_DIGEST,
        "candidate packet premanifest digest differs",
    )
    records = manifest["premanifest_output_tree"]
    _require(type(records) is list and bool(records), "candidate packet tree is empty")
    paths: set[str] = set()
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        _require(type(record) is dict, "candidate packet tree row differs")
        _exact_keys(record, ("path", "sha256", "size"), "candidate packet tree row")
        relative = _manifest_relative_path(record["path"], "candidate packet tree path")
        _require(
            relative.as_posix() not in paths,
            "candidate packet tree path repeats",
        )
        _require(_is_sha256(record["sha256"]), "candidate packet tree SHA-256 differs")
        _require(
            type(record["size"]) is int and record["size"] > 0,
            "candidate packet tree size differs",
        )
        paths.add(relative.as_posix())
        normalized_records.append(dict(record))
    _require(
        [row["path"] for row in normalized_records]
        == sorted(row["path"] for row in normalized_records),
        "candidate packet tree is not canonical path order",
    )
    _require(
        hashlib.sha256(canonical_json_bytes(normalized_records) + b"\n").hexdigest()
        == manifest["premanifest_output_tree_digest"],
        "candidate packet tree digest differs",
    )
    records_by_path = {row["path"]: row for row in normalized_records}

    sums_row = candidate_packet["sha256sums"]
    sums_path = _verify_file_row(
        sums_row,
        "candidate packet SHA256SUMS",
        require_nlink1=True,
    )
    expected_sums = {row["path"]: row["sha256"] for row in normalized_records}
    expected_sums["manifest.json"] = manifest_row["sha256"]
    expected_payload = "".join(
        f"{expected_sums[name]}  {name}\n" for name in sorted(expected_sums)
    ).encode("utf-8")
    _require(
        _stable_bytes(sums_path) == expected_payload,
        "candidate packet SHA256SUMS inventory differs",
    )
    _verify_file_row(
        sums_row,
        "candidate packet SHA256SUMS",
        require_nlink1=True,
    )

    authority = manifest["authority"]
    _require(type(authority) is dict, "candidate packet authority differs")
    for name in ("source_video", "masklet_receipt"):
        _require(
            type(authority.get(name)) is dict,
            f"candidate packet {name} differs",
        )
        _exact_keys(
            authority[name],
            ("path", "sha256", "size"),
            f"candidate packet {name}",
        )

    frames = manifest["frames"]
    _require(
        type(frames) is list and len(frames) == FRAME_COUNT,
        "candidate packet frame count differs",
    )
    expected_support: list[Mapping[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        _require(
            type(frame) is dict
            and type(frame.get("frame_index")) is int
            and frame.get("frame_index") == frame_index,
            "candidate packet frame order differs",
        )
        outputs = frame.get("outputs")
        _require(type(outputs) is dict, "candidate packet frame outputs differ")
        support = outputs.get("candidate_support")
        _require(type(support) is dict, "candidate packet support row differs")
        _exact_keys(
            support,
            ("path", "sha256", "size"),
            "candidate packet support row",
        )
        _require(
            support["path"] == f"masks/candidate_support/{frame_index:05d}.png"
            and _is_sha256(support["sha256"])
            and type(support["size"]) is int
            and support["size"] > 0,
            "candidate packet support binding differs",
        )
        _require(
            records_by_path.get(support["path"]) == support,
            "candidate packet support is not bound into exact inventory",
        )
        expected_support.append(support)
    return manifest, expected_support


def _replay_external_support_review(
    formal: Any,
    *,
    expected_slot: int,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    _require(type(formal) is dict, "external review differs")
    _exact_keys(
        formal,
        (
            "reviewer_slot",
            "receipt",
            "reviewer_identity",
            "reviewer_affiliation_or_role",
            "reviewed_at_utc",
            "independence_attestation",
            "signature",
            "evidence",
        ),
        "external review",
    )
    _require(
        type(formal["reviewer_slot"]) is int
        and formal["reviewer_slot"] == expected_slot,
        "external reviewer slot differs",
    )
    receipt_path = _verify_file_row(
        formal["receipt"],
        "external review receipt",
        require_nlink1=True,
    )
    loaded_receipt_path, receipt = _load_json_file_row(
        formal["receipt"],
        "external review receipt",
    )
    _require(
        loaded_receipt_path == receipt_path,
        "external review receipt path changed during replay",
    )
    _verify_file_row(
        formal["receipt"],
        "external review receipt",
        require_nlink1=True,
    )
    _exact_keys(
        receipt,
        (
            "schema_version",
            "reviewer_slot",
            "reviewer_identity",
            "reviewer_affiliation_or_role",
            "candidate_manifest_sha256",
            "reviewed_at_utc",
            "independence_attestation",
            "all_81_native_frames_reviewed",
            "instructions",
            "frames",
            "overall_decision",
            "signature_or_external_receipt",
            "claim_limits_acknowledged",
        ),
        "external review receipt",
    )
    _require(
        receipt["schema_version"] == EXTERNAL_REVIEW_SCHEMA
        and type(receipt["reviewer_slot"]) is int
        and receipt["reviewer_slot"] == expected_slot
        and receipt["candidate_manifest_sha256"] == manifest_sha256,
        "external review receipt identity/manifest differs",
    )
    for name in ("reviewer_identity", "reviewer_affiliation_or_role"):
        _require(
            type(receipt[name]) is str
            and bool(receipt[name])
            and receipt[name] == receipt[name].strip(),
            f"external review {name} differs",
        )
        _require(
            receipt[name] == formal[name],
            f"formal external review {name} differs",
        )
    reviewed_at = receipt["reviewed_at_utc"]
    _require(
        type(reviewed_at) is str
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            reviewed_at,
        )
        is not None
        and reviewed_at == formal["reviewed_at_utc"],
        "external review timestamp differs",
    )
    try:
        parsed_reviewed_at = datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise BoneRemovedV2Error("external review timestamp differs") from error
    _require(
        parsed_reviewed_at.strftime("%Y-%m-%dT%H:%M:%SZ") == reviewed_at,
        "external review timestamp differs",
    )
    independence = receipt["independence_attestation"]
    _require(type(independence) is dict, "external review independence differs")
    _exact_keys(
        independence,
        EXTERNAL_INDEPENDENCE_KEYS,
        "external review independence",
    )
    _require(
        all(independence[name] is True for name in EXTERNAL_INDEPENDENCE_KEYS)
        and independence == formal["independence_attestation"],
        "external review independence declaration differs",
    )
    _require(
        receipt["all_81_native_frames_reviewed"] is True
        and receipt["claim_limits_acknowledged"] is True
        and receipt["overall_decision"] == "PASS"
        and receipt["instructions"] == list(EXTERNAL_REVIEW_INSTRUCTIONS),
        "external review all-frame decision/limits differs",
    )
    frames = receipt["frames"]
    _require(
        type(frames) is list and len(frames) == FRAME_COUNT,
        "external review frame count differs",
    )
    for frame_index, frame in enumerate(frames):
        _require(type(frame) is dict, "external review frame differs")
        _exact_keys(
            frame,
            (
                "frame_index",
                "bone_coverage",
                "contact_shadow_coverage",
                "halo_and_adjacent_ground_coverage",
                "dog_and_guard_protection",
                "boundary_edit_requested",
                "notes",
                "decision",
            ),
            "external review frame",
        )
        _require(
            type(frame["frame_index"]) is int
            and frame["frame_index"] == frame_index,
            "external review frame order differs",
        )
        _require(
            all(
                frame[name] == "PASS"
                for name in (
                    "bone_coverage",
                    "contact_shadow_coverage",
                    "halo_and_adjacent_ground_coverage",
                    "dog_and_guard_protection",
                    "decision",
                )
            )
            and frame["boundary_edit_requested"] is False
            and type(frame["notes"]) is str
            and bool(frame["notes"].strip()),
            f"external review frame ballot differs: {frame_index}",
        )

    signature = receipt["signature_or_external_receipt"]
    _require(type(signature) is dict, "external review signature differs")
    _exact_keys(
        signature,
        ("kind", "review_projection_sha256", "evidence_sha256", "evidence_size"),
        "external review signature",
    )
    _require(
        signature["kind"] == EXTERNAL_SIGNATURE_KIND
        and _is_sha256(signature["review_projection_sha256"])
        and _is_sha256(signature["evidence_sha256"])
        and type(signature["evidence_size"]) is int
        and signature["evidence_size"] > 0
        and signature == formal["signature"],
        "external review opaque signature declaration differs",
    )
    projection = dict(receipt)
    projection["signature_or_external_receipt"] = None
    _require(
        object_sha256(projection) == signature["review_projection_sha256"],
        "external review projection binding differs",
    )
    evidence_path = _verify_file_row(
        formal["evidence"],
        "external review evidence",
        require_nlink1=True,
    )
    _require(
        formal["evidence"]["sha256"] == signature["evidence_sha256"]
        and formal["evidence"]["size"] == signature["evidence_size"],
        "external review evidence binding differs",
    )
    _verify_file_row(
        formal["evidence"],
        "external review evidence",
        require_nlink1=True,
    )
    return {
        "identity": receipt["reviewer_identity"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": formal["receipt"]["sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": formal["evidence"]["sha256"],
    }


def _replay_support_review(
    review: Mapping[str, Any],
    *,
    producer: Mapping[str, Any],
) -> tuple[tuple[bytes, ...], Mapping[str, Any]]:
    _exact_keys(
        review,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "candidate_packet",
            "source",
            "sam2_receipt",
            "external_reviews",
            "protocol",
            "frame_masks",
            "claim_limits",
            "review_digest",
        ),
        "support review receipt",
    )
    _require(review["schema_version"] == SUPPORT_REVIEW_SCHEMA, "support review schema differs")
    _require(review["status"] == SUPPORT_REVIEW_STATUS, "support review status differs")
    _require(review["case_id"] == CASE_ID and review["iid"] == IID, "support review case differs")
    _require(review["source"] == producer["source"], "support review source binding differs")
    _require(
        review["sam2_receipt"] == producer["mask_authority"]["receipt"],
        "support review SAM2 binding differs",
    )

    packet_manifest, packet_support_rows = _replay_support_packet_manifest(
        review["candidate_packet"]
    )
    packet_authority = packet_manifest["authority"]
    _require(
        _same_file_bytes(review["source"], packet_authority["source_video"]),
        "support review source bytes differ from candidate packet",
    )
    _require(
        _same_file_bytes(
            review["sam2_receipt"],
            packet_authority["masklet_receipt"],
        ),
        "support review SAM2 bytes differ from candidate packet",
    )

    external_reviews = review["external_reviews"]
    _require(
        type(external_reviews) is list and len(external_reviews) == 2,
        "exactly two external reviewers are required",
    )
    external_rows = [
        _replay_external_support_review(
            external_reviews[index],
            expected_slot=index + 1,
            manifest_sha256=review["candidate_packet"]["manifest"]["sha256"],
        )
        for index in range(2)
    ]
    for field in (
        "identity",
        "receipt_path",
        "receipt_sha256",
        "evidence_path",
        "evidence_sha256",
    ):
        _require(
            external_rows[0][field] != external_rows[1][field],
            f"external reviewer {field} repeats",
        )

    protocol = review["protocol"]
    _require(type(protocol) is dict, "support review protocol differs")
    _exact_keys(
        protocol,
        (
            "native_resolution_704x736",
            "all_81_frames_reviewed_by_each",
            "required_external_reviewers",
            "bone_covered_all_frames_by_each",
            "cast_shadow_and_halo_covered_all_frames_by_each",
            "minimum_bone_dilation_pixels",
            "old_dilate3_reused",
            "dog_guard_excluded_all_frames",
        ),
        "support review protocol",
    )
    _require(protocol["native_resolution_704x736"] is True, "support review was not native resolution")
    _require(
        protocol["all_81_frames_reviewed_by_each"] is True,
        "support review is not all-81 per reviewer",
    )
    _require(
        type(protocol["required_external_reviewers"]) is int
        and protocol["required_external_reviewers"] == 2,
        "support reviewer count differs",
    )
    _require(
        protocol["bone_covered_all_frames_by_each"] is True,
        "support review reports uncovered bone",
    )
    _require(
        protocol["cast_shadow_and_halo_covered_all_frames_by_each"] is True,
        "support review reports uncovered cast shadow/halo",
    )
    _require(
        type(protocol["minimum_bone_dilation_pixels"]) is int
        and protocol["minimum_bone_dilation_pixels"] >= MIN_SUPPORT_DILATION,
        "support review minimum bone dilation differs",
    )
    _require(protocol["old_dilate3_reused"] is False, "support review reused old dilate3")
    _require(
        protocol["dog_guard_excluded_all_frames"] is True,
        "support review reports dog-guard intersection",
    )

    claim_limits = review["claim_limits"]
    _require(type(claim_limits) is dict, "support review claim limits differ")
    _exact_keys(
        claim_limits,
        SUPPORT_REVIEW_CLAIM_LIMITS,
        "support review claim limits",
    )
    _require(
        claim_limits == SUPPORT_REVIEW_CLAIM_LIMITS,
        "support review overclaims external facts",
    )

    reviewed_rows = review["frame_masks"]
    producer_rows = producer["support"]["frame_masks"]
    _require(
        isinstance(reviewed_rows, list) and len(reviewed_rows) == FRAME_COUNT,
        "support review frame-mask count differs",
    )
    _require(
        isinstance(producer_rows, list) and len(producer_rows) == FRAME_COUNT,
        "producer support frame-mask count differs",
    )
    payloads: list[bytes] = []
    parents: set[Path] = set()
    observed_paths: set[str] = set()
    for frame_index, (reviewed, declared, packet_row) in enumerate(
        zip(reviewed_rows, producer_rows, packet_support_rows)
    ):
        _require(isinstance(reviewed, dict), f"support reviewed mask {frame_index} differs")
        _exact_keys(
            reviewed,
            (
                "frame_index",
                "path",
                "sha256",
                "size",
                "bone_and_cast_shadow_covered",
                "native_resolution_reviewed",
            ),
            f"support reviewed mask {frame_index}",
        )
        _require(
            type(reviewed["frame_index"]) is int
            and reviewed["frame_index"] == frame_index,
            "support reviewed-mask order differs",
        )
        _require(
            reviewed["bone_and_cast_shadow_covered"] is True
            and reviewed["native_resolution_reviewed"] is True,
            f"support reviewed-mask ballot fails: {frame_index}",
        )
        reviewed_file = {
            key: reviewed[key]
            for key in ("frame_index", "path", "sha256", "size")
        }
        _require(reviewed_file == declared, f"support reviewed-mask binding differs: {frame_index}")
        _require(
            reviewed["sha256"] == packet_row["sha256"]
            and reviewed["size"] == packet_row["size"],
            f"support mask differs from candidate packet: {frame_index}",
        )
        file_fields = {key: reviewed[key] for key in ("path", "sha256", "size")}
        path = _verify_file_row(
            file_fields,
            f"support reviewed-mask file {frame_index}",
            require_nlink1=True,
        )
        _require(path.name == f"{frame_index:05d}.png", "support mask filename differs")
        _require(str(path) not in observed_paths, "support mask path repeats")
        observed_paths.add(str(path))
        parents.add(path.parent)
        payload = _stable_bytes(path)
        _require(
            hashlib.sha256(payload).hexdigest() == file_fields["sha256"]
            and len(payload) == file_fields["size"],
            f"support reviewed-mask file differs: {frame_index}",
        )
        _validate_single_png_frame(
            payload,
            f"support reviewed mask {frame_index}",
            require_grayscale_8bit=True,
        )
        payloads.append(payload)
    _require(len(parents) == 1, "support masks do not share one directory")
    parent = next(iter(parents))
    _require(
        {entry.name for entry in parent.iterdir()}
        == {f"{frame_index:05d}.png" for frame_index in range(FRAME_COUNT)},
        "support mask directory has extra/missing entries",
    )
    _validate_digest(review, "review_digest", "support review receipt")
    return tuple(payloads), {
        "reviewer_identities": [row["identity"] for row in external_rows],
        "external_review_receipt_sha256s": [
            row["receipt_sha256"] for row in external_rows
        ],
        "review_digest": review["review_digest"],
        "frame_count": len(payloads),
    }


def _split_ffmpeg_stream_fields(description: str) -> tuple[str, ...]:
    fields: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(description):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            _require(depth >= 0, "static ffmpeg stream parentheses differ")
        elif character == "," and depth == 0:
            fields.append(description[start:index].strip())
            start = index + 1
    _require(depth == 0, "static ffmpeg stream parentheses are unclosed")
    fields.append(description[start:].strip())
    return tuple(fields)


def _parse_static_ffmpeg_probe(
    stderr: str,
    *,
    label: str,
    enforce_fps: bool,
    expected_width: int,
    expected_height: int,
) -> Mapping[str, Any]:
    _require("Stream mapping:" in stderr, f"static ffmpeg stream mapping missing: {label}")
    input_section = stderr.split("Stream mapping:", 1)[0]
    stream_pattern = re.compile(
        r"^\s*Stream #0:\d+(?:\[[^\]]+\])?(?:\([^)]*\))?:\s*"
        r"([A-Za-z]+):\s*(.+)$"
    )
    input_streams: list[tuple[str, str]] = []
    for line in input_section.splitlines():
        matched = stream_pattern.match(line)
        if matched is not None:
            input_streams.append((matched.group(1), matched.group(2)))
    _require(
        len(input_streams) == 1 and input_streams[0][0] == "Video",
        f"static ffmpeg input stream closure differs: {label}",
    )
    description = input_streams[0][1]
    fields = _split_ffmpeg_stream_fields(description)
    _require(len(fields) >= 3, f"static ffmpeg video description differs: {label}")
    codec_match = re.match(r"^([A-Za-z0-9_]+)(?:\s|$)", fields[0])
    pixel_format_match = re.match(r"^([A-Za-z0-9_]+)(?:\(|\s|$)", fields[1])
    _require(
        codec_match is not None and pixel_format_match is not None,
        f"static ffmpeg codec fields differ: {label}",
    )

    config_matches = re.findall(
        r"\bconfig in time_base:\s*(-?\d+)/(-?\d+),\s*frame_rate:\s*(-?\d+)/(-?\d+)",
        stderr,
    )
    _require(len(config_matches) == 1, f"static ffmpeg showinfo config differs: {label}")
    try:
        time_base = Fraction(int(config_matches[0][0]), int(config_matches[0][1]))
        frame_rate = Fraction(int(config_matches[0][2]), int(config_matches[0][3]))
    except (ValueError, ZeroDivisionError) as error:
        raise BoneRemovedV2Error(f"static ffmpeg timing schema differs: {label}") from error
    _require(time_base > 0 and frame_rate > 0, f"static ffmpeg timing differs: {label}")

    frame_rows: list[tuple[int, int, str, int, int]] = []
    for line in stderr.splitlines():
        if "Parsed_showinfo_" not in line or re.search(r"\bn:\s*\d+", line) is None:
            continue
        number = re.search(r"\bn:\s*(\d+)", line)
        pts = re.search(r"\bpts:\s*(-?\d+)", line)
        pixel_format = re.search(r"\bfmt:([A-Za-z0-9_]+)", line)
        geometry = re.search(r"\bs:(\d+)x(\d+)", line)
        _require(
            number is not None
            and pts is not None
            and pixel_format is not None
            and geometry is not None,
            f"static ffmpeg showinfo frame differs: {label}",
        )
        frame_rows.append(
            (
                int(number.group(1)),
                int(pts.group(1)),
                pixel_format.group(1),
                int(geometry.group(1)),
                int(geometry.group(2)),
            )
        )
    _require(
        len(frame_rows) == FRAME_COUNT
        and [row[0] for row in frame_rows] == list(range(FRAME_COUNT)),
        f"static ffmpeg exact frame closure differs: {label}",
    )
    _require(
        all(
            row[2] == pixel_format_match.group(1)
            and (row[3], row[4]) == (expected_width, expected_height)
            for row in frame_rows
        ),
        f"static ffmpeg decoded geometry/pixel format differs: {label}",
    )
    pts_values = [row[1] for row in frame_rows]
    _require(pts_values[0] == 0, f"static ffmpeg first PTS is not zero: {label}")
    steps = [
        (pts_values[index + 1] - pts_values[index]) * time_base
        for index in range(FRAME_COUNT - 1)
    ]
    _require(
        steps and all(step > 0 and step == steps[0] for step in steps),
        f"static ffmpeg frame PTS are not strictly uniform: {label}",
    )
    decoded_rate = 1 / steps[0]
    _require(
        decoded_rate == frame_rate,
        f"static ffmpeg PTS/frame-rate binding differs: {label}",
    )
    _require(
        not enforce_fps or frame_rate == Fraction(FPS, 1),
        f"media frame rate differs: {label}",
    )
    timing = {
        "time_base_num": time_base.numerator,
        "time_base_den": time_base.denominator,
        "frame_pts": pts_values,
        "uniform_step_num": steps[0].numerator,
        "uniform_step_den": steps[0].denominator,
    }
    return {
        "inspection_tool": "pinned_static_ffmpeg_showinfo_held_fd",
        "codec_type": "video",
        "codec_name": codec_match.group(1),
        "width": expected_width,
        "height": expected_height,
        "pixel_format": pixel_format_match.group(1),
        "fps_num": frame_rate.numerator,
        "fps_den": frame_rate.denominator,
        "frame_count": len(frame_rows),
        "frame_timing": timing,
        "frame_timing_digest": object_sha256(timing),
    }


def _probe_video(
    ffmpeg: HeldFile,
    media: HeldFile,
    *,
    enforce_fps: bool = True,
    expected_width: int = WIDTH,
    expected_height: int = HEIGHT,
) -> Mapping[str, Any]:
    completed = subprocess.run(
        (
            ffmpeg.proc_path,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            media.proc_path,
            "-map",
            "0:v:0",
            "-vf",
            "showinfo",
            "-vsync",
            "0",
            "-frames:v",
            str(FRAME_COUNT + 1),
            "-f",
            "null",
            "pipe:1",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=tuple(sorted({ffmpeg.descriptor, media.descriptor})),
    )
    _require(
        completed.returncode == 0,
        f"static ffmpeg inspection failed: {media.label}: {completed.stderr}",
    )
    return _parse_static_ffmpeg_probe(
        completed.stderr,
        label=media.label,
        enforce_fps=enforce_fps,
        expected_width=expected_width,
        expected_height=expected_height,
    )


def _verify_vace_frame_and_geometry_consistency(
    source_probe: Mapping[str, Any],
    mask_probe: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Check frozen keep-last math against static-ffmpeg CFR observations.

    This is deliberately not described as an independent Decord frame-ID
    replay.  The independently replayed prepared CTHW byte digest is the
    binding for the actual ordered model input.
    """

    _require(
        source_probe["frame_timing"] == mask_probe["frame_timing"],
        "lossless precanvas source/mask timing differs",
    )
    _require(
        (source_probe["width"], source_probe["height"])
        == (mask_probe["width"], mask_probe["height"])
        == (RAW_DONOR_WIDTH, RAW_DONOR_HEIGHT),
        "lossless precanvas geometry differs before VACE replay",
    )
    timing = source_probe["frame_timing"]
    time_base = Fraction(timing["time_base_num"], timing["time_base_den"])
    pts = timing["frame_pts"]
    step = Fraction(timing["uniform_step_num"], timing["uniform_step_den"])
    _require(
        len(pts) == FRAME_COUNT
        and [value * time_base for value in pts]
        == [Fraction(index, FPS) for index in range(FRAME_COUNT)]
        and step == Fraction(1, FPS),
        "precanvas timestamps cannot replay exact VACE frame IDs",
    )

    downsample = (4, 16, 16)
    sequence_length = 32760
    model_area = 480 * 832
    latent_area = min(
        sequence_length,
        model_area // (downsample[1] * downsample[2]),
        (RAW_DONOR_HEIGHT // downsample[1])
        * (RAW_DONOR_WIDTH // downsample[2]),
    )
    latent_frames = min(
        (FRAME_COUNT - 1) // downsample[0] + 1,
        sequence_length // latent_area,
    )
    target_latent_area = min(latent_area, sequence_length // latent_frames)
    latent_height = round(
        math.sqrt(
            float(
                Fraction(target_latent_area * RAW_DONOR_HEIGHT, RAW_DONOR_WIDTH)
            )
        )
    )
    latent_width = target_latent_area // latent_height
    selected_frame_count = (latent_frames - 1) * downsample[0] + 1
    processed_height = latent_height * downsample[1]
    processed_width = latent_width * downsample[2]
    _require(
        (latent_area, selected_frame_count, processed_width, processed_height)
        == (1560, FRAME_COUNT, RAW_DONOR_WIDTH, RAW_DONOR_HEIGHT),
        "frozen VACE keep-last area/downsample replay differs",
    )

    starts = [value * time_base for value in pts]
    ends = [start + step for start in starts]
    duration = (starts[-1] + ends[-1]) / 2
    target_timestamps = [
        duration * index / (selected_frame_count - 1)
        for index in range(selected_frame_count)
    ]
    frame_indices: list[int] = []
    for target in target_timestamps:
        matches = [
            index
            for index, (start, end) in enumerate(zip(starts, ends))
            if target >= start and target <= end
        ]
        _require(matches, "frozen VACE target timestamp has no input frame")
        frame_indices.append(matches[0])
    _require(
        frame_indices == list(range(FRAME_COUNT)) == trace["frame_indices"],
        "VACE frame-ID trace is inconsistent with static CFR keep-last math",
    )
    resize_crop_applied = (
        processed_width != RAW_DONOR_WIDTH
        or processed_height != RAW_DONOR_HEIGHT
    )
    _require(
        resize_crop_applied is False and trace["resize_crop_applied"] is False,
        "VACE resize/crop trace is inconsistent with frozen geometry math",
    )
    return {
        "implementation_contract": (
            "frozen_VaceVideoProcessor_keep_last_downsample_4_16_16_"
            "seq_len_32760_area_480x832"
        ),
        "input_frame_timing_digest": source_probe["frame_timing_digest"],
        "frame_indices": frame_indices,
        "latent_area": latent_area,
        "latent_frame_count": latent_frames,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "resize_crop_applied": resize_crop_applied,
        "static_ffmpeg_timing_and_frozen_formula_consistent": True,
        "frame_ids_trace_independently_replayed_with_decord": False,
    }


def _decoder(
    ffmpeg: HeldFile,
    media: HeldFile,
    pixel_format: str,
    *,
    video_filter: str | None = None,
) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg.proc_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        media.proc_path,
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-frames:v",
        str(FRAME_COUNT),
    ]
    if video_filter is not None:
        command.extend(("-vf", video_filter))
    command.extend(("-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"))
    return subprocess.Popen(
        tuple(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=tuple(sorted({ffmpeg.descriptor, media.descriptor})),
    )


def _validate_single_png_frame(
    payload: bytes,
    label: str,
    *,
    require_grayscale_8bit: bool = False,
) -> None:
    """Validate one independently framed PNG, including every chunk CRC."""

    _require(payload.startswith(b"\x89PNG\r\n\x1a\n"), f"PNG signature differs: {label}")
    offset = 8
    chunk_index = 0
    saw_ihdr = False
    saw_idat = False
    saw_iend = False
    while offset < len(payload):
        _require(len(payload) - offset >= 12, f"truncated PNG chunk: {label}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        _require(
            all(
                (ord("A") <= value <= ord("Z"))
                or (ord("a") <= value <= ord("z"))
                for value in chunk_type
            ),
            f"PNG chunk type differs: {label}",
        )
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        _require(crc_end <= len(payload), f"truncated PNG payload: {label}")
        chunk_data = payload[data_start:data_end]
        declared_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        _require(declared_crc == actual_crc, f"PNG chunk CRC differs: {label}")
        if chunk_index == 0:
            _require(chunk_type == b"IHDR", f"PNG IHDR is not first: {label}")
        if chunk_type == b"IHDR":
            _require(not saw_ihdr and length == 13, f"PNG IHDR closure differs: {label}")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
            _require(
                (width, height) == (WIDTH, HEIGHT),
                f"PNG geometry differs: {label}",
            )
            _require(
                compression == 0 and filtering == 0 and interlace in (0, 1),
                f"PNG coding fields differ: {label}",
            )
            if require_grayscale_8bit:
                _require(
                    bit_depth == 8 and color_type == 0 and interlace == 0,
                    f"support PNG is not noninterlaced 8-bit grayscale: {label}",
                )
            saw_ihdr = True
        elif chunk_type == b"IDAT":
            _require(saw_ihdr and not saw_iend, f"PNG IDAT ordering differs: {label}")
            saw_idat = True
        elif chunk_type == b"IEND":
            _require(
                saw_ihdr and saw_idat and not saw_iend and length == 0,
                f"PNG IEND closure differs: {label}",
            )
            saw_iend = True
            _require(crc_end == len(payload), f"PNG has bytes after IEND: {label}")
        else:
            _require(not saw_iend, f"PNG chunk follows IEND: {label}")
        offset = crc_end
        chunk_index += 1
    _require(
        offset == len(payload) and saw_ihdr and saw_idat and saw_iend,
        f"PNG framing differs: {label}",
    )


def _decode_mask_bytes(
    ffmpeg: HeldFile,
    png_frames: Sequence[bytes],
    object_name: str,
) -> bytes:
    _require(len(png_frames) == FRAME_COUNT, f"mask byte count differs: {object_name}")
    for frame_index, payload in enumerate(png_frames):
        _validate_single_png_frame(payload, f"{object_name} frame {frame_index}")
    completed = subprocess.run(
        (
            ffmpeg.proc_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            str(FPS),
            "-vcodec",
            "png",
            "-i",
            "pipe:0",
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        input=b"".join(png_frames),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        pass_fds=ffmpeg.pass_fds,
    )
    _require(
        completed.returncode == 0,
        f"mask decoder failed: {object_name}: "
        f"{completed.stderr.decode('utf-8', errors='replace')}",
    )
    _require(
        len(completed.stdout) == FRAME_PIXELS * FRAME_COUNT,
        f"mask decoder byte count differs: {object_name}",
    )
    return completed.stdout


def _read_exact(process: subprocess.Popen[bytes], byte_count: int, label: str) -> bytes:
    _require(process.stdout is not None, f"decoder stdout unavailable: {label}")
    payload = bytearray()
    while len(payload) < byte_count:
        block = process.stdout.read(byte_count - len(payload))
        if not block:
            break
        payload.extend(block)
    _require(len(payload) == byte_count, f"decoder ended early: {label}")
    return bytes(payload)


def _finish_decoder(process: subprocess.Popen[bytes], label: str) -> None:
    if process.stdout is not None:
        extra = process.stdout.read(1)
        process.stdout.close()
        _require(extra == b"", f"decoder produced extra frames: {label}")
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.stderr is not None:
        process.stderr.close()
    return_code = process.wait()
    _require(return_code == 0, f"decoder failed: {label}: {stderr.decode('utf-8', errors='replace')}")


def _active(mask: bytes, label: str) -> set[int]:
    values = set(mask)
    _require(values <= {0, 255} and 255 in values, f"mask is not nonempty binary: {label}")
    return {index for index, value in enumerate(mask) if value}


def dilate(indices: set[int], passes: int) -> set[int]:
    current = set(indices)
    for _ in range(passes):
        expanded = set(current)
        for index in current:
            y, x = divmod(index, WIDTH)
            for new_y in range(max(0, y - 1), min(HEIGHT, y + 2)):
                row = new_y * WIDTH
                for new_x in range(max(0, x - 1), min(WIDTH, x + 2)):
                    expanded.add(row + new_x)
        current = expanded
    return current


def _luma(frame: bytes, index: int) -> int:
    offset = index * 3
    return (
        77 * frame[offset]
        + 150 * frame[offset + 1]
        + 29 * frame[offset + 2]
    ) >> 8


def _laplacian_energy(frame: bytes, region: set[int]) -> float:
    values: list[int] = []
    for index in region:
        y, x = divmod(index, WIDTH)
        if x == 0 or x == WIDTH - 1 or y == 0 or y == HEIGHT - 1:
            continue
        values.append(
            abs(
                4 * _luma(frame, index)
                - _luma(frame, index - 1)
                - _luma(frame, index + 1)
                - _luma(frame, index - WIDTH)
                - _luma(frame, index + WIDTH)
            )
        )
    _require(values, "texture region is empty")
    return sum(values) / len(values)


def _context_gradient(frame: bytes, region: set[int]) -> float:
    total = 0
    count = 0
    for index in region:
        y, x = divmod(index, WIDTH)
        for neighbor in (index + 1 if x + 1 < WIDTH else None, index + WIDTH if y + 1 < HEIGHT else None):
            if neighbor is not None and neighbor in region:
                total += abs(_luma(frame, index) - _luma(frame, neighbor))
                count += 1
    _require(count > 0, "context gradient region is empty")
    return total / count


def _boundary_gradient(frame: bytes, support: set[int]) -> float:
    total = 0
    count = 0
    for index in support:
        y, x = divmod(index, WIDTH)
        for neighbor in (
            index - 1 if x else None,
            index + 1 if x + 1 < WIDTH else None,
            index - WIDTH if y else None,
            index + WIDTH if y + 1 < HEIGHT else None,
        ):
            if neighbor is not None and neighbor not in support:
                total += abs(_luma(frame, index) - _luma(frame, neighbor))
                count += 1
    _require(count > 0, "support boundary is empty")
    return total / count


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(values, "percentile input is empty")
    ordered = sorted(values)
    position = int((len(ordered) - 1) * fraction)
    return float(ordered[position])


def evaluate_metric_summary(summary: Mapping[str, Any]) -> None:
    _exact_keys(
        summary,
        (
            "frame_count",
            "outside_support_changed_pixels",
            "dog_guard_changed_pixels",
            "support_pixels_not_equal_raw_donor",
            "precanvas_source_mismatch_pixels",
            "precanvas_mask_mismatch_pixels",
            "precanvas_support_pad_active_pixels",
            "processed_source_rgb_mad_mean",
            "processed_source_rgb_mad_frame_maximum",
            "processed_mask_threshold_mismatch_pixels",
            "bone_changed_fraction_minimum",
            "bone_source_residual_p10",
            "texture_ratio_p10",
            "texture_ratio_median",
            "texture_ratio_maximum",
            "low_texture_frame_count",
            "seam_ratio_median",
            "seam_ratio_maximum",
            "delivery_rgb_mad_mean",
            "delivery_rgb_mad_frame_maximum",
            "delivery_outside_support_rgb_mad_mean",
            "delivery_outside_support_rgb_mad_frame_maximum",
            "delivery_dog_guard_rgb_mad_mean",
            "delivery_dog_guard_rgb_mad_frame_maximum",
        ),
        "metric summary",
    )
    _require(summary["frame_count"] == 81, "metric frame count differs")
    _require(summary["outside_support_changed_pixels"] == 0, "canonical pixels changed outside support")
    _require(summary["dog_guard_changed_pixels"] == 0, "canonical dog-guard pixels changed")
    _require(
        summary["support_pixels_not_equal_raw_donor"] == 0,
        "canonical support pixels differ from raw donor",
    )
    _require(
        summary["precanvas_source_mismatch_pixels"] == 0,
        "lossless precanvas source differs from source fit/pad replay",
    )
    _require(
        summary["precanvas_mask_mismatch_pixels"] == 0,
        "lossless precanvas mask differs from support fit/pad replay",
    )
    _require(
        summary["precanvas_support_pad_active_pixels"] == 0,
        "support enters the donor-free precanvas padding",
    )
    _require(
        float(summary["processed_source_rgb_mad_mean"])
        <= MAX_DELIVERY_RGB_MAD_MEAN
        and float(summary["processed_source_rgb_mad_frame_maximum"])
        <= MAX_DELIVERY_RGB_MAD_FRAME,
        "nonauthoritative processed-source cache diagnostic differs",
    )
    _require(
        summary["processed_mask_threshold_mismatch_pixels"] == 0,
        "nonauthoritative processed-mask cache diagnostic differs",
    )
    _require(float(summary["bone_changed_fraction_minimum"]) >= MIN_BONE_CHANGED_FRACTION_PER_FRAME, "source bone pixels were retained")
    _require(float(summary["bone_source_residual_p10"]) >= MIN_BONE_SOURCE_RESIDUAL_P10, "source-bone residual is too small")
    _require(float(summary["texture_ratio_p10"]) >= MIN_TEXTURE_RATIO_P10, "candidate core has a low-texture scar tail")
    _require(float(summary["texture_ratio_median"]) >= MIN_TEXTURE_RATIO_MEDIAN, "candidate core remains interpolation-smooth")
    _require(float(summary["texture_ratio_median"]) <= MAX_TEXTURE_RATIO_MEDIAN, "candidate core has excessive texture energy")
    _require(int(summary["low_texture_frame_count"]) <= MAX_LOW_TEXTURE_FRAMES, "too many low-texture scar frames")
    _require(float(summary["seam_ratio_median"]) <= MAX_SEAM_RATIO_MEDIAN, "median support seam is excessive")
    _require(float(summary["seam_ratio_maximum"]) <= MAX_SEAM_RATIO, "one or more support seams are excessive")
    _require(float(summary["delivery_rgb_mad_mean"]) <= MAX_DELIVERY_RGB_MAD_MEAN, "delivery video is too far from canonical")
    _require(float(summary["delivery_rgb_mad_frame_maximum"]) <= MAX_DELIVERY_RGB_MAD_FRAME, "one delivery frame is too far from canonical")
    _require(
        float(summary["delivery_outside_support_rgb_mad_mean"])
        <= MAX_DELIVERY_RGB_MAD_MEAN,
        "delivery background is too far from canonical",
    )
    _require(
        float(summary["delivery_outside_support_rgb_mad_frame_maximum"])
        <= MAX_DELIVERY_RGB_MAD_FRAME,
        "one delivery background frame is too far from canonical",
    )
    _require(
        float(summary["delivery_dog_guard_rgb_mad_mean"])
        <= MAX_DELIVERY_RGB_MAD_MEAN,
        "delivery dog guard is too far from canonical",
    )
    _require(
        float(summary["delivery_dog_guard_rgb_mad_frame_maximum"])
        <= MAX_DELIVERY_RGB_MAD_FRAME,
        "one delivery dog-guard frame is too far from canonical",
    )


def _replay_mask_receipt(
    mask_root: Path,
    receipt: Mapping[str, Any],
) -> Mapping[str, tuple[bytes, ...]]:
    _require(receipt.get("schema_version") == "bernini-case01-oracle-sam2-masklets-receipt-v1", "SAM2 receipt schema differs")
    _require(receipt.get("case_id") == CASE_ID and receipt.get("iid") == IID, "SAM2 receipt case differs")
    _require(_is_sha256(receipt.get("receipt_digest")), "SAM2 receipt digest differs")
    receipt_payload = dict(receipt)
    receipt_digest = receipt_payload.pop("receipt_digest")
    _require(
        receipt_digest
        == hashlib.sha256(canonical_json_bytes(receipt_payload) + b"\n").hexdigest(),
        "SAM2 receipt canonical digest mismatch",
    )
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, list), "SAM2 outputs missing")
    rows = {row.get("path"): row for row in outputs if isinstance(row, dict)}
    required = {
        f"masks/{object_name}/{frame_index:05d}.png"
        for object_name in ("bone", "dog")
        for frame_index in range(FRAME_COUNT)
    }
    _require(required <= set(rows), "SAM2 receipt lacks one or more mask frames")
    payloads: dict[str, list[bytes]] = {"bone": [], "dog": []}
    for object_name in ("bone", "dog"):
        for frame_index in range(FRAME_COUNT):
            relative = f"masks/{object_name}/{frame_index:05d}.png"
            row = rows[relative]
            _exact_keys(row, ("path", "sha256", "size"), f"SAM2 output {relative}")
            path = _regular_file(mask_root / relative)
            payload = _stable_bytes(path)
            _require(
                hashlib.sha256(payload).hexdigest() == row["sha256"]
                and len(payload) == row["size"],
                f"SAM2 mask differs: {relative}",
            )
            payloads[object_name].append(payload)
    return {name: tuple(frames) for name, frames in payloads.items()}


def audit_media(
    *,
    receipt: Mapping[str, Any],
    observations: Mapping[str, Any],
    mask_root: Path,
    ffmpeg: str,
    ffprobe: str,
) -> Mapping[str, Any]:
    held_files: list[HeldFile] = []
    try:
        return _audit_media_impl(
            receipt=receipt,
            observations=observations,
            mask_root=mask_root,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            held_files=held_files,
        )
    finally:
        _close_held_files(held_files)


def _audit_media_impl(
    *,
    receipt: Mapping[str, Any],
    observations: Mapping[str, Any],
    mask_root: Path,
    ffmpeg: str,
    ffprobe: str,
    held_files: list[HeldFile],
) -> Mapping[str, Any]:
    validate_producer_receipt(receipt)
    source_held = _open_held_file_row(receipt["source"], "source")
    held_files.append(source_held)
    mask_receipt_path, mask_receipt = _load_json_file_row(
        receipt["mask_authority"]["receipt"],
        "mask receipt",
    )
    ffmpeg_held = _open_held_file_row(
        receipt["media_tools"]["ffmpeg"],
        "ffmpeg",
        require_executable=True,
    )
    held_files.append(ffmpeg_held)
    ffprobe_held = _open_held_file_row(
        receipt["media_tools"]["ffprobe"],
        "ffprobe",
        require_executable=True,
    )
    held_files.append(ffprobe_held)
    _require(ffmpeg == FFMPEG_PATH, "CLI ffmpeg differs from pinned authority")
    _require(ffprobe == FFPROBE_PATH, "CLI ffprobe differs from pinned authority")
    acceptance_held = _open_held_file_row(
        receipt["acceptance_contract"],
        "acceptance contract",
    )
    held_files.append(acceptance_held)
    _require(
        acceptance_held.path == Path(__file__).resolve(strict=True),
        "executing acceptance contract differs from producer binding",
    )
    generator_path = _verify_file_row(receipt["generator"]["program"], "generator program")
    del generator_path
    model_authority_replay = _replay_model_authority_manifests(receipt["generator"])
    create_authority = receipt["create_only_authority"]
    _verify_file_row(create_authority["controller_program"], "create-only controller program")
    _, attempt_receipt = _load_json_file_row(
        create_authority["attempt_receipt"],
        "create-only attempt receipt",
    )
    _, publication_receipt = _load_json_file_row(
        create_authority["publication_receipt"],
        "create-only publication receipt",
    )
    _, support_review = _load_json_file_row(
        receipt["support"]["review_receipt"],
        "support review receipt",
    )
    support_png_bytes, support_review_replay = _replay_support_review(
        support_review,
        producer=receipt,
    )
    raw_donor_held = _open_held_file_row(
        receipt["generator"]["raw_support_donor"]["video"],
        "raw support donor video",
        require_nlink1=True,
    )
    held_files.append(raw_donor_held)
    normalization = receipt["generator"]["raw_support_donor"]["normalization"]
    normalization_media: dict[str, HeldFile] = {}
    for name in (
        "precanvas_source_video",
        "precanvas_mask_video",
        "processed_source_video",
        "processed_mask_video",
    ):
        held = _open_held_file_row(
            normalization[name],
            f"raw donor normalization {name}",
            require_nlink1=True,
        )
        held_files.append(held)
        normalization_media[name] = held
    support_held = _open_held_file_row(
        receipt["support"]["tube"],
        "support tube",
        require_nlink1=True,
    )
    held_files.append(support_held)
    canonical_held = _open_held_file_row(
        receipt["canonical_candidate"]["video"],
        "canonical candidate video",
        require_nlink1=True,
    )
    held_files.append(canonical_held)
    delivery_held = _open_held_file_row(
        receipt["delivery_candidate"]["video"],
        "delivery candidate video",
        require_nlink1=True,
    )
    held_files.append(delivery_held)
    create_only_replay = validate_create_only_receipts(
        attempt_receipt,
        publication_receipt,
        producer=receipt,
    )
    create_only_replay = dict(create_only_replay)
    create_only_replay["published_tree_relative_files"] = list(
        _verify_exact_published_tree(
            create_only_replay["final_root"],
            {
                "support": support_held.path,
                "canonical_candidate": canonical_held.path,
                "delivery_candidate": delivery_held.path,
            },
        )
    )
    _require(mask_receipt_path.parent == mask_root, "mask root differs from receipt parent")
    mask_png_bytes = _replay_mask_receipt(mask_root, mask_receipt)
    validate_observations(
        observations,
        candidate_sha256=receipt["canonical_candidate"]["video"]["sha256"],
        delivery_sha256=receipt["delivery_candidate"]["video"]["sha256"],
        support_sha256=receipt["support"]["tube"]["sha256"],
    )

    probes = {
        "source": _probe_video(ffmpeg_held, source_held),
        "support": _probe_video(ffmpeg_held, support_held),
        "canonical": _probe_video(ffmpeg_held, canonical_held),
        "delivery": _probe_video(ffmpeg_held, delivery_held),
        "raw_support_donor": _probe_video(
            ffmpeg_held,
            raw_donor_held,
            enforce_fps=False,
            expected_width=RAW_DONOR_WIDTH,
            expected_height=RAW_DONOR_HEIGHT,
        ),
        **{
            name: _probe_video(
                ffmpeg_held,
                held,
                enforce_fps=name.startswith("precanvas_"),
                expected_width=RAW_DONOR_WIDTH,
                expected_height=RAW_DONOR_HEIGHT,
            )
            for name, held in normalization_media.items()
        },
    }
    vace_frame_geometry_consistency = _verify_vace_frame_and_geometry_consistency(
        probes["precanvas_source_video"],
        probes["precanvas_mask_video"],
        normalization["prepare_source_trace"],
    )
    _require(probes["support"]["codec_name"] == "ffv1", "support tube is not FFV1")
    _require(probes["support"]["pixel_format"] == "gray", "support tube is not lossless gray")
    _require(probes["canonical"]["codec_name"] == "ffv1", "canonical candidate is not FFV1")
    _require(
        probes["canonical"]["pixel_format"] == "bgr0",
        "canonical candidate stored pixel format is not FFV1-compatible bgr0",
    )
    _require(probes["delivery"]["codec_name"] == "h264", "delivery candidate is not H.264")
    _require(probes["delivery"]["pixel_format"] == "yuv420p", "delivery candidate is not yuv420p")
    _require(
        probes["raw_support_donor"]["codec_name"] == "h264"
        and probes["raw_support_donor"]["pixel_format"] == "yuv420p",
        "raw VACE support donor is not the H.264 out_video",
    )
    _require(
        probes["precanvas_source_video"]["codec_name"] == "ffv1"
        and probes["precanvas_source_video"]["pixel_format"] == "bgr0",
        "precanvas source is not lossless FFV1/bgr0 authority",
    )
    _require(
        probes["precanvas_mask_video"]["codec_name"] == "ffv1"
        and probes["precanvas_mask_video"]["pixel_format"] == "gray",
        "precanvas mask is not lossless FFV1/gray authority",
    )
    for name in ("processed_source_video", "processed_mask_video"):
        _require(
            probes[name]["codec_name"] == "h264"
            and probes[name]["pixel_format"] == "yuv420p",
            f"nonauthoritative VACE cache media differs: {name}",
        )

    decoded_masks = {
        object_name: _decode_mask_bytes(
            ffmpeg_held,
            mask_png_bytes[object_name],
            object_name,
        )
        for object_name in ("bone", "dog")
    }
    decoded_support_masks = _decode_mask_bytes(
        ffmpeg_held,
        support_png_bytes,
        "reviewed support",
    )
    processes = {
        "source": _decoder(ffmpeg_held, source_held, "rgb24"),
        "canonical": _decoder(ffmpeg_held, canonical_held, "rgb24"),
        "delivery": _decoder(ffmpeg_held, delivery_held, "rgb24"),
        "support": _decoder(ffmpeg_held, support_held, "gray"),
        "raw_support_donor": _decoder(
            ffmpeg_held,
            raw_donor_held,
            "rgb24",
            video_filter=(
                "crop=612:640:6:0,"
                "scale=704:736:flags=lanczos,format=rgb24"
            ),
        ),
        "expected_precanvas_source": _decoder(
            ffmpeg_held,
            source_held,
            "rgb24",
            video_filter=(
                "scale=612:640:flags=lanczos,"
                "pad=624:640:6:0:color=black,format=rgb24"
            ),
        ),
        "precanvas_source": _decoder(
            ffmpeg_held,
            normalization_media["precanvas_source_video"],
            "rgb24",
        ),
        "expected_precanvas_mask": _decoder(
            ffmpeg_held,
            support_held,
            "gray",
            video_filter=(
                "scale=612:640:flags=neighbor,"
                "pad=624:640:6:0:color=black,format=gray"
            ),
        ),
        "precanvas_mask": _decoder(
            ffmpeg_held,
            normalization_media["precanvas_mask_video"],
            "gray",
        ),
        "processed_source": _decoder(
            ffmpeg_held,
            normalization_media["processed_source_video"],
            "rgb24",
        ),
        "processed_mask": _decoder(
            ffmpeg_held,
            normalization_media["processed_mask_video"],
            "gray",
        ),
    }
    outside_changed = 0
    dog_guard_changed = 0
    support_not_equal_raw_donor = 0
    bone_changed_fractions: list[float] = []
    bone_residuals: list[float] = []
    texture_ratios: list[float] = []
    seam_ratios: list[float] = []
    delivery_mads: list[float] = []
    delivery_outside_support_mads: list[float] = []
    delivery_dog_guard_mads: list[float] = []
    precanvas_source_mismatch_pixels = 0
    precanvas_mask_mismatch_pixels = 0
    precanvas_support_pad_active_pixels = 0
    processed_source_mads: list[float] = []
    processed_mask_threshold_mismatch_pixels = 0
    support_areas: list[int] = []
    prepared_source_planes = (bytearray(), bytearray(), bytearray())
    prepared_mask_plane = bytearray()
    try:
        for frame_index in range(FRAME_COUNT):
            source = _read_exact(processes["source"], RGB_FRAME_BYTES, "source")
            candidate = _read_exact(processes["canonical"], RGB_FRAME_BYTES, "canonical")
            delivery = _read_exact(processes["delivery"], RGB_FRAME_BYTES, "delivery")
            raw_donor = _read_exact(
                processes["raw_support_donor"],
                RGB_FRAME_BYTES,
                "raw support donor",
            )
            expected_precanvas_source = _read_exact(
                processes["expected_precanvas_source"],
                PRECANVAS_RGB_FRAME_BYTES,
                "expected precanvas source",
            )
            precanvas_source = _read_exact(
                processes["precanvas_source"],
                PRECANVAS_RGB_FRAME_BYTES,
                "precanvas source",
            )
            expected_precanvas_mask = _read_exact(
                processes["expected_precanvas_mask"],
                PRECANVAS_PIXELS,
                "expected precanvas mask",
            )
            precanvas_mask = _read_exact(
                processes["precanvas_mask"],
                PRECANVAS_PIXELS,
                "precanvas mask",
            )
            processed_source = _read_exact(
                processes["processed_source"],
                PRECANVAS_RGB_FRAME_BYTES,
                "nonauthoritative processed source cache",
            )
            processed_mask = _read_exact(
                processes["processed_mask"],
                PRECANVAS_PIXELS,
                "nonauthoritative processed mask cache",
            )
            _active(expected_precanvas_mask, f"expected precanvas mask {frame_index}")
            _active(precanvas_mask, f"precanvas mask {frame_index}")
            _require(
                expected_precanvas_source == precanvas_source,
                f"lossless precanvas source differs from fit/pad replay in frame {frame_index}",
            )
            _require(
                expected_precanvas_mask == precanvas_mask,
                f"lossless precanvas mask differs from fit/pad replay in frame {frame_index}",
            )
            # ffmpeg's rgb24 decode is T,H,W,C.  Split it now, retaining each
            # channel as T,H,W so the final channel concatenation is exactly
            # the contiguous C,T,H,W order produced by VACE/PyTorch.
            prepared_source_planes[0].extend(precanvas_source[0::3])
            prepared_source_planes[1].extend(precanvas_source[1::3])
            prepared_source_planes[2].extend(precanvas_source[2::3])
            prepared_mask_plane.extend(precanvas_mask)
            for y in range(RAW_DONOR_HEIGHT):
                row_start = y * RAW_DONOR_WIDTH
                precanvas_support_pad_active_pixels += sum(
                    value != 0
                    for value in (
                        precanvas_mask[row_start : row_start + 6]
                        + precanvas_mask[row_start + 618 : row_start + 624]
                    )
                )
            thresholded_processed_mask = processed_mask.translate(
                BINARY_THRESHOLD_TABLE
            )
            _require(
                thresholded_processed_mask == precanvas_mask,
                f"nonauthoritative processed mask diagnostic differs in frame {frame_index}",
            )
            processed_source_mads.append(
                sum(
                    abs(processed - authoritative)
                    for processed, authoritative in zip(
                        processed_source,
                        precanvas_source,
                    )
                )
                / PRECANVAS_RGB_FRAME_BYTES
            )
            support_raw = _read_exact(processes["support"], FRAME_PIXELS, "support")
            mask_start = frame_index * FRAME_PIXELS
            mask_end = mask_start + FRAME_PIXELS
            _require(
                support_raw == decoded_support_masks[mask_start:mask_end],
                f"published support tube differs from reviewed mask in frame {frame_index}",
            )
            support = _active(support_raw, f"support frame {frame_index}")
            bone = _active(
                decoded_masks["bone"][mask_start:mask_end],
                f"bone frame {frame_index}",
            )
            dog = _active(
                decoded_masks["dog"][mask_start:mask_end],
                f"dog frame {frame_index}",
            )
            required_support = dilate(bone, MIN_SUPPORT_DILATION)
            _require(required_support <= support, f"support does not contain dilate-{MIN_SUPPORT_DILATION} bone in frame {frame_index}")
            _require(len(support) / len(bone) <= MAX_SUPPORT_TO_BONE_AREA_RATIO, f"support is too broad in frame {frame_index}")
            dog_guard = dilate(dog, DOG_GUARD_DILATION)
            _require(not (support & dog_guard), f"support intersects dog guard in frame {frame_index}")
            support_areas.append(len(support))

            frame_outside_changed = 0
            frame_dog_guard_changed = 0
            bone_changed = 0
            bone_residual = 0
            delivery_residual = 0
            delivery_outside_support_residual = 0
            delivery_dog_guard_residual = 0
            for pixel in range(FRAME_PIXELS):
                offset = pixel * 3
                same = candidate[offset : offset + 3] == source[offset : offset + 3]
                if pixel not in support and not same:
                    frame_outside_changed += 1
                if pixel in dog_guard and not same:
                    frame_dog_guard_changed += 1
                if (
                    pixel in support
                    and candidate[offset : offset + 3]
                    != raw_donor[offset : offset + 3]
                ):
                    support_not_equal_raw_donor += 1
                if pixel in bone:
                    if not same:
                        bone_changed += 1
                    bone_residual += sum(
                        abs(candidate[offset + channel] - source[offset + channel])
                        for channel in range(3)
                    )
                delivery_pixel_residual = sum(
                    abs(delivery[offset + channel] - candidate[offset + channel])
                    for channel in range(3)
                )
                delivery_residual += delivery_pixel_residual
                if pixel not in support:
                    delivery_outside_support_residual += delivery_pixel_residual
                if pixel in dog_guard:
                    delivery_dog_guard_residual += delivery_pixel_residual
            outside_changed += frame_outside_changed
            dog_guard_changed += frame_dog_guard_changed
            bone_changed_fractions.append(bone_changed / len(bone))
            bone_residuals.append(bone_residual / (len(bone) * 3))
            delivery_mads.append(delivery_residual / RGB_FRAME_BYTES)
            outside_support_pixels = FRAME_PIXELS - len(support)
            _require(outside_support_pixels > 0, "support covers the entire frame")
            delivery_outside_support_mads.append(
                delivery_outside_support_residual / (outside_support_pixels * 3)
            )
            delivery_dog_guard_mads.append(
                delivery_dog_guard_residual / (len(dog_guard) * 3)
            )

            context = dilate(support, CONTEXT_RING_DILATION) - support - dog_guard
            _require(context, f"context ring is empty in frame {frame_index}")
            context_energy = _laplacian_energy(candidate, context)
            _require(context_energy > 0.0, f"context has zero texture in frame {frame_index}")
            texture_ratios.append(_laplacian_energy(candidate, bone) / context_energy)
            context_gradient = _context_gradient(candidate, context)
            _require(context_gradient > 0.0, f"context has zero gradient in frame {frame_index}")
            seam_ratios.append(_boundary_gradient(candidate, support) / context_gradient)
    finally:
        for label, process in processes.items():
            _finish_decoder(process, label)

    expected_prepared_elements = FRAME_COUNT * PRECANVAS_PIXELS
    _require(
        all(len(plane) == expected_prepared_elements for plane in prepared_source_planes)
        and len(prepared_mask_plane) == expected_prepared_elements,
        "VACE prepared tensor replay element count differs",
    )
    prepared_tensor_replay = _verify_prepared_tensor_replay(
        normalization["prepare_source_trace"],
        prepared_source_planes,
        prepared_mask_plane,
    )
    del prepared_source_planes
    del prepared_mask_plane
    post_media_published_files = list(
        _verify_exact_published_tree(
            create_only_replay["final_root"],
            {
                "support": support_held.path,
                "canonical_candidate": canonical_held.path,
                "delivery_candidate": delivery_held.path,
            },
        )
    )
    _require(
        post_media_published_files
        == create_only_replay["published_tree_relative_files"],
        "published tree changed across media audit",
    )
    _require(
        not os.path.lexists(create_only_replay["staging_root"]),
        "staging root reappeared during media audit",
    )
    create_only_replay["post_media_tree_rescan_relative_files"] = (
        post_media_published_files
    )
    create_only_replay["post_media_staging_absent"] = True
    create_only_replay["held_leaf_named_join_replayed_on_close"] = True

    summary = {
        "frame_count": FRAME_COUNT,
        "outside_support_changed_pixels": outside_changed,
        "dog_guard_changed_pixels": dog_guard_changed,
        "support_pixels_not_equal_raw_donor": support_not_equal_raw_donor,
        "precanvas_source_mismatch_pixels": precanvas_source_mismatch_pixels,
        "precanvas_mask_mismatch_pixels": precanvas_mask_mismatch_pixels,
        "precanvas_support_pad_active_pixels": precanvas_support_pad_active_pixels,
        "processed_source_rgb_mad_mean": sum(processed_source_mads)
        / len(processed_source_mads),
        "processed_source_rgb_mad_frame_maximum": max(processed_source_mads),
        "processed_mask_threshold_mismatch_pixels": (
            processed_mask_threshold_mismatch_pixels
        ),
        "bone_changed_fraction_minimum": min(bone_changed_fractions),
        "bone_source_residual_p10": _percentile(bone_residuals, 0.10),
        "texture_ratio_p10": _percentile(texture_ratios, 0.10),
        "texture_ratio_median": _percentile(texture_ratios, 0.50),
        "texture_ratio_maximum": max(texture_ratios),
        "low_texture_frame_count": sum(value < LOW_TEXTURE_RATIO for value in texture_ratios),
        "seam_ratio_median": _percentile(seam_ratios, 0.50),
        "seam_ratio_maximum": max(seam_ratios),
        "delivery_rgb_mad_mean": sum(delivery_mads) / len(delivery_mads),
        "delivery_rgb_mad_frame_maximum": max(delivery_mads),
        "delivery_outside_support_rgb_mad_mean": sum(delivery_outside_support_mads)
        / len(delivery_outside_support_mads),
        "delivery_outside_support_rgb_mad_frame_maximum": max(
            delivery_outside_support_mads
        ),
        "delivery_dog_guard_rgb_mad_mean": sum(delivery_dog_guard_mads)
        / len(delivery_dog_guard_mads),
        "delivery_dog_guard_rgb_mad_frame_maximum": max(delivery_dog_guard_mads),
    }
    evaluate_metric_summary(summary)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_CURRENT_BYTES_CONSTRAINTS_INPUT_ASSET_ONLY",
        "case_id": CASE_ID,
        "iid": IID,
        "role": ROLE,
        "producer_receipt_digest": receipt["receipt_digest"],
        "observation_digest": observations["observation_digest"],
        "acceptance_contract": dict(receipt["acceptance_contract"]),
        "create_only_replay": create_only_replay,
        "model_authority_replay": model_authority_replay,
        "support_review_replay": support_review_replay,
        "prepared_tensor_replay": prepared_tensor_replay,
        "vace_frame_geometry_consistency": vace_frame_geometry_consistency,
        "media_inspection_authority": {
            "static_ffmpeg_held_fd_used_for_every_probe_and_decode": True,
            "dynamic_ffprobe_executed": False,
            "ffprobe_row_hash_bound_for_receipt_compatibility_only": True,
        },
        "media_probes": probes,
        "metric_summary": summary,
        "support_area": {
            "minimum": min(support_areas),
            "maximum": max(support_areas),
            "space_time_pixels": sum(support_areas),
        },
        "gates": {
            "fresh_v2_not_old_interpolation_lineage": "PASS",
            "source_and_mask_authority": "PASS",
            "declared_model_and_runtime_current_tree_replay_only": "PASS",
            "reviewed_support_png_tube_exactness": "PASS",
            "lossless_precanvas_fit_pad_and_inverse_geometry": "PASS",
            "vace_frame_ids_static_cfr_formula_consistency": "PASS",
            "vace_prepared_tensor_independent_replay_and_before_after_binding": "PASS",
            "lossless_outside_support_identity": "PASS",
            "dog_plus_guard_identity": "PASS",
            "bone_source_pixel_removal": "PASS",
            "texture_and_seam_diagnostics": "PASS",
            "self_declared_two_reviewer_all81_semantic_ballots": "PASS",
            "create_only_current_tree_plus_controller_attestation": "PASS",
        },
        "claim_limits": {
            "input_asset_authority_only": True,
            "renderer_inference_performed": False,
            "renderer_result_claim_authorized": False,
            "scientific_claim_authorized": False,
            "automatic_metrics_alone_prove_semantic_absence": False,
            "canonical_ffv1_is_identity_authority": True,
            "h264_delivery_is_identity_authority": False,
            "downstream_identity_sensitive_consumption_requires_canonical": True,
            "actual_downstream_identity_sensitive_consumer_verified": False,
            "generation_execution_lineage_verified": False,
            "model_checkpoint_preapproval_verified": False,
            "lossless_precanvas_is_vace_input_authority": True,
            "prepared_tensor_digests_independently_replayed_from_lossless_precanvas": True,
            "frame_ids_trace_independently_replayed_with_decord": False,
            "saved_processed_h264_cache_is_identity_authority": False,
            "reviewer_authorship_cryptographically_proven": False,
            "semantic_ballots_authorize_scientific_claim": False,
            "create_history_cryptographically_proven": False,
        },
    }
    report["report_digest"] = object_sha256(report)
    return report


def write_create_only(path_value: str | Path, value: Mapping[str, Any]) -> None:
    """Create one immutable report inode without replacing an existing name.

    On failure this function only unlinks the exact inode it created.  Under a
    same-UID hostile concurrent rename, an attacker's replacement name may
    remain; the call still fails closed and no report is returned.  Therefore
    this helper proves no-replace publication of a successful report, not
    failure-path name absence against a same-UID adversary.
    """

    path = Path(path_value)
    _require(path.is_absolute(), "report path must be absolute")
    _require(os.path.normpath(str(path)) == str(path), "report path must be canonical")
    _require(path.parent.resolve(strict=True) == path.parent, "report parent traverses a symlink")
    _require(path.name not in ("", ".", ".."), "report basename differs")
    payload = canonical_json_bytes(value) + b"\n"
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(path.parent, directory_flags)
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
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
        except FileExistsError as error:
            raise BoneRemovedV2Error("fresh report path required") from error
        created = os.fstat(descriptor)
        _require(stat.S_ISREG(created.st_mode) and created.st_nlink == 1, "report reservation differs")
        created_identity = (created.st_dev, created.st_ino)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            _require(count > 0, "zero-byte report write")
            written += count
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        replay = bytearray()
        while len(replay) < len(payload):
            block = os.read(descriptor, len(payload) - len(replay))
            _require(block != b"", "report read-back ended early")
            replay.extend(block)
        _require(bytes(replay) == payload and os.read(descriptor, 1) == b"", "report read-back differs")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require(
            stat.S_ISREG(observed.st_mode)
            and stat.S_IMODE(observed.st_mode) == 0o444
            and observed.st_nlink == 1
            and observed.st_size == len(payload)
            and (observed.st_dev, observed.st_ino) == (named.st_dev, named.st_ino),
            "report publication differs",
        )
        os.close(descriptor)
        descriptor = None
        replay_descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            reopened_before = os.fstat(replay_descriptor)
            _require(
                stat.S_ISREG(reopened_before.st_mode)
                and stat.S_IMODE(reopened_before.st_mode) == 0o444
                and reopened_before.st_nlink == 1
                and reopened_before.st_size == len(payload)
                and (reopened_before.st_dev, reopened_before.st_ino)
                == created_identity,
                "reopened report identity differs",
            )
            replay_after_close = bytearray()
            while len(replay_after_close) < len(payload):
                block = os.read(replay_descriptor, len(payload) - len(replay_after_close))
                _require(block != b"", "named report read-back ended early")
                replay_after_close.extend(block)
            _require(
                bytes(replay_after_close) == payload
                and os.read(replay_descriptor, 1) == b"",
                "named report read-back differs",
            )
            os.fsync(parent_descriptor)
            reopened_after = os.fstat(replay_descriptor)
            named_after = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require(
                _stat_identity(reopened_before)
                == _stat_identity(reopened_after)
                == _stat_identity(named_after),
                "reopened report or named join changed",
            )
        finally:
            os.close(replay_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created_identity is not None:
            try:
                named = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                if (named.st_dev, named.st_ino) == created_identity:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-receipt", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args(argv)

    receipt_path = _regular_file(args.producer_receipt)
    observations_path = _regular_file(args.observations)
    mask_root = args.mask_root.resolve(strict=True)
    _require(mask_root.is_dir() and not mask_root.is_symlink(), "mask root differs")
    receipt = _load_json(receipt_path, "producer receipt")
    observations = _load_json(observations_path, "observations")
    report = audit_media(
        receipt=receipt,
        observations=observations,
        mask_root=mask_root,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    if args.output_report is None:
        sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    else:
        write_create_only(args.output_report, report)
        print(json.dumps({"status": report["status"], "report": str(args.output_report), "report_digest": report["report_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BoneRemovedV2Error as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
