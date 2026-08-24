#!/usr/bin/env python3
"""Seal, validate, or promote external case01 support reviews.

The unsigned packet remains immutable and PENDING.  This tool replays that
packet in full.  ``verify-packet`` checks the packet alone before review;
``seal-review`` mechanically binds one already-completed human ballot to
detached opaque evidence, including non-promotable FAIL ballots;
``verify-submission`` replays one complete sealed submission.  ``validate-only``
twice replays two separately authored all-81 PASS receipts without filesystem
output.  ``promote`` creates the v1 support bundle consumable by the frozen
bone-removed-v2 generator; its receipts bind the chosen absolute final path, so
the published directory must not be moved and reused as authority.  No command
performs a visual review, verifies a human identity, runs VACE, or touches a
GPU.

Publication is create-only.  A private same-parent staging directory is
made complete and fsynced before Darwin ``renameatx_np(RENAME_EXCL)`` or Linux
``renameat2(RENAME_NOREPLACE)`` publishes it.  There is no ordinary-rename
fallback and no write is ever made through the requested final path before
that atomic operation.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import types
import unicodedata
from typing import Any, Iterable, Mapping, NoReturn, Sequence


CASE_ID = "case01"
IID = "288545b9c031491a"
FRAME_COUNT = 81

PACKET_MANIFEST_SHA256 = (
    "91c2a3bb101621edc6b93b96cbb9af75369fc4c5474c5d61c5395620046b4435"
)
PACKET_MANIFEST_SIZE = 260_175
PACKET_PREMANIFEST_DIGEST = (
    "6374275b26be8c9e0f6f86cbcde4bca1ca6ad46cd0db9d7a7cdaee76f1cbf36e"
)
PACKET_SCHEMA = "bernini-case01-bone-contact-support-review-packet-v1"
PACKET_STATUS = "UNSIGNED_CANDIDATE_HOLD_PENDING_TWO_EXTERNAL_REVIEWS"

METHOD_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = METHOD_ROOT / "generate_case01_bone_removed_v2_vace_v1.py"
GENERATOR_SHA256 = (
    "f6dc4edb5ea3da03e14dd00399a800c3af545379bd0030aeab0fc8e2a205ce86"
)
GENERATOR_SIZE = 85_957

PROMOTION_SCHEMA = (
    "bernini-case01-bone-contact-support-dual-review-path-bound-promotion-v1"
)
PROMOTION_STATUS = (
    "PASS_PATH_BOUND_SUPPORT_AUTHORITY_CREATED_FROM_TWO_EXTERNAL_REVIEWS"
)
PREFLIGHT_SCHEMA = "bernini-case01-bone-contact-support-input-preflight-v1"
PREFLIGHT_STATUS = "VALIDATED_INPUT_BYTES_HOLD_NO_AUTHORITY_CREATED"
PACKET_PREFLIGHT_SCHEMA = "bernini-case01-bone-contact-support-packet-preflight-v1"
PACKET_PREFLIGHT_STATUS = "VERIFIED_PACKET_BYTES_HOLD_NO_EXTERNAL_REVIEW"
REVIEW_SUBMISSION_SCHEMA = (
    "bernini-case01-bone-contact-support-external-review-submission-v1"
)
REVIEW_SUBMISSION_PASS_STATUS = (
    "SEALED_STRUCTURAL_PASS_HOLD_PENDING_DISTINCT_PAIR_PREFLIGHT"
)
REVIEW_SUBMISSION_FAIL_STATUS = "SEALED_NONPROMOTABLE_FAIL_REVIEW"
REVIEW_SUBMISSION_COMPLETE_BYTES = (
    b"BERNINI_CASE01_EXTERNAL_REVIEW_SUBMISSION_COMPLETE_V1\n"
)
COMPLETE_BYTES = b"BERNINI_CASE01_SUPPORT_DUAL_REVIEW_PROMOTION_COMPLETE_V1\n"
PROMOTION_CLAIM_LIMITS = {
    "human_identity_or_affiliation_verified": False,
    "human_independence_verified": False,
    "external_evidence_cryptographically_verified": False,
    "visual_review_reperformed": False,
    "input_support_gate_only": True,
    "cleanplate_generated": False,
    "renderer_or_vace_run_authorized_by_promotion_alone": False,
    "gpu_execution_performed": False,
    "training_performed": False,
    "scientific_claim_authorized": False,
}
PREFLIGHT_CLAIM_LIMITS = {
    "point_in_time_input_validation_only": True,
    "human_identity_or_affiliation_verified": False,
    "human_independence_verified": False,
    "opaque_evidence_authenticity_verified": False,
    "visual_review_reperformed": False,
    "formal_support_authority_created": False,
    "promotion_bundle_published": False,
    "renderer_or_vace_run_authorized": False,
    "gpu_execution_performed": False,
    "valid_as_future_promotion_token": False,
    "scientific_claim_authorized": False,
}
PACKET_PREFLIGHT_CLAIM_LIMITS = {
    "point_in_time_packet_validation_only": True,
    "visual_review_performed": False,
    "external_review_receipt_created": False,
    "formal_support_authority_created": False,
    "renderer_or_vace_run_authorized": False,
    "gpu_execution_performed": False,
    "valid_as_future_promotion_token": False,
    "scientific_claim_authorized": False,
}
REVIEW_SUBMISSION_CLAIM_LIMITS = {
    "human_authored_fields_modified_by_sealer": False,
    "candidate_manifest_binding_added_by_sealer": True,
    "canonicalization_and_evidence_binding_added_by_sealer": True,
    "human_identity_or_affiliation_verified": False,
    "human_independence_verified": False,
    "opaque_evidence_authenticity_verified": False,
    "visual_review_reperformed": False,
    "second_distinct_review_validated": False,
    "formal_support_authority_created": False,
    "promotion_bundle_published": False,
    "renderer_or_vace_run_authorized": False,
    "gpu_execution_performed": False,
    "scientific_claim_authorized": False,
}
MAX_PACKET_BYTES = 512 * 1024 * 1024
MAX_SINGLE_PACKET_FILE_BYTES = 64 * 1024 * 1024
MAX_EXTERNAL_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_EXTERNAL_EVIDENCE_BYTES = 16 * 1024 * 1024
SHA256_HEX = frozenset("0123456789abcdef")


class SupportPromotionHold(RuntimeError):
    """No path-bound v1 support authority may be published."""


def fail(message: str) -> NoReturn:
    raise SupportPromotionHold(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


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
        raise SupportPromotionHold("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= SHA256_HEX


def _exact_keys(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(keys)
    require(
        type(value) is dict and set(value) == expected,
        "%s key closure differs" % label,
    )
    return value


def _canonical_absolute(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    require(path.is_absolute(), "%s is not absolute" % label)
    require(os.path.normpath(str(path)) == str(path), "%s is not canonical" % label)
    return path


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory(path_value: str | Path, label: str) -> Path:
    path = _canonical_absolute(path_value, label)
    try:
        named = path.lstat()
    except OSError as error:
        raise SupportPromotionHold("%s is unavailable" % label) from error
    require(
        stat.S_ISDIR(named.st_mode)
        and not path.is_symlink()
        and path.resolve(strict=True) == path,
        "%s must be a canonical plain directory" % label,
    )
    return path


def _stable_file(
    path_value: str | Path,
    label: str,
    *,
    require_nlink1: bool,
    maximum_bytes: int,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    path = _canonical_absolute(path_value, label)
    try:
        named_before = path.lstat()
        require(
            stat.S_ISREG(named_before.st_mode)
            and not path.is_symlink(),
            "%s is not a plain regular file" % label,
        )
        require(path.resolve(strict=True) == path, "%s traverses a symlink" % label)
    except OSError as error:
        raise SupportPromotionHold("%s is unavailable" % label) from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SupportPromotionHold("%s could not be opened" % label) from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "%s is not regular" % label)
        require(
            not require_nlink1 or int(before.st_nlink) == 1,
            "%s is not nlink1" % label,
        )
        require(
            0 < int(before.st_size) <= maximum_bytes,
            "%s size is outside the allowed bound" % label,
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
        named = path.lstat()
        require(
            _identity(before) == _identity(after) == _identity(named),
            "%s changed while hashing" % label,
        )
        require(size == int(after.st_size), "%s byte count differs" % label)
        return (
            {"path": str(path), "sha256": digest.hexdigest(), "size": size},
            _identity(after),
        )
    finally:
        os.close(descriptor)


def _stable_bytes(
    path_value: str | Path,
    label: str,
    *,
    require_nlink1: bool,
    maximum_bytes: int,
) -> tuple[bytes, dict[str, Any], tuple[int, ...]]:
    row, first_identity = _stable_file(
        path_value,
        label,
        require_nlink1=require_nlink1,
        maximum_bytes=maximum_bytes,
    )
    path = Path(row["path"])
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(_identity(before) == first_identity, "%s identity changed before read" % label)
        chunks: list[bytes] = []
        remaining = int(row["size"])
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            require(bool(block), "%s ended early" % label)
            chunks.append(block)
            remaining -= len(block)
        require(os.read(descriptor, 1) == b"", "%s grew while reading" % label)
        after = os.fstat(descriptor)
        named = path.lstat()
        payload = b"".join(chunks)
        require(
            _identity(before) == _identity(after) == _identity(named) == first_identity,
            "%s changed while reading" % label,
        )
        require(
            len(payload) == row["size"]
            and hashlib.sha256(payload).hexdigest() == row["sha256"],
            "%s bytes differ from stable row" % label,
        )
        return payload, row, first_identity
    finally:
        os.close(descriptor)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        require(key not in value, "duplicate JSON key: %s" % key)
        value[key] = child
    return value


def _decode_json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: fail("non-finite JSON constant: %s" % token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportPromotionHold("%s is invalid JSON" % label) from error
    require(type(value) is dict, "%s root is not an object" % label)
    return value


def _load_canonical_json(
    path_value: str | Path,
    label: str,
    *,
    require_nlink1: bool,
    maximum_bytes: int,
) -> tuple[Mapping[str, Any], dict[str, Any], tuple[int, ...]]:
    payload, row, identity = _stable_bytes(
        path_value,
        label,
        require_nlink1=require_nlink1,
        maximum_bytes=maximum_bytes,
    )
    value = _decode_json_object(payload, label)
    require(
        payload == canonical_json_bytes(value) + b"\n",
        "%s is not canonical one-LF JSON" % label,
    )
    return value, row, identity


def _load_json_draft(
    path_value: str | Path,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[Mapping[str, Any], dict[str, Any], tuple[int, ...]]:
    payload, row, identity = _stable_bytes(
        path_value,
        label,
        require_nlink1=True,
        maximum_bytes=maximum_bytes,
    )
    return _decode_json_object(payload, label), row, identity


def _relative_path(value: Any, label: str) -> PurePosixPath:
    require(type(value) is str and bool(value) and "\\" not in value, "%s differs" % label)
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(part not in ("", ".", "..") for part in relative.parts),
        "%s is not a canonical relative POSIX path" % label,
    )
    return relative


def _load_generator() -> Any:
    payload, row, _ = _stable_bytes(
        GENERATOR_PATH,
        "frozen generator",
        require_nlink1=False,
        maximum_bytes=2 * 1024 * 1024,
    )
    require(
        (row["sha256"], row["size"]) == (GENERATOR_SHA256, GENERATOR_SIZE),
        "frozen generator tuple differs",
    )
    name = "case01_bone_removed_v2_generator_for_support_promotion"
    module = types.ModuleType(name)
    module.__file__ = str(GENERATOR_PATH)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    try:
        code = compile(
            payload,
            str(GENERATOR_PATH),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module.__dict__)
    except Exception as error:
        raise SupportPromotionHold("frozen generator source execution failed") from error
    payload_after, row_after, _ = _stable_bytes(
        GENERATOR_PATH,
        "frozen generator",
        require_nlink1=False,
        maximum_bytes=2 * 1024 * 1024,
    )
    require(
        row_after == row and payload_after == payload,
        "frozen generator changed during source execution",
    )
    return module


def _walk_exact_packet(root: Path, expected_files: set[str]) -> None:
    expected_directories = {""}
    for relative_text in expected_files:
        relative = PurePosixPath(relative_text)
        for parent in relative.parents:
            if parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
    observed_files: set[str] = set()
    observed_directories: set[str] = {""}

    def onerror(error: OSError) -> None:
        raise SupportPromotionHold("packet walk failed") from error

    for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=onerror):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            child = directory_path / dirname
            named = child.lstat()
            require(
                stat.S_ISDIR(named.st_mode) and not child.is_symlink(),
                "packet contains a symlink or non-directory",
            )
            relative = child.relative_to(root).as_posix()
            observed_directories.add(relative)
        for filename in filenames:
            child = directory_path / filename
            relative = child.relative_to(root).as_posix()
            named = child.lstat()
            require(
                stat.S_ISREG(named.st_mode)
                and not child.is_symlink()
                and int(named.st_nlink) == 1,
                "packet contains a symlink, special, or non-nlink1 file: %s" % relative,
            )
            observed_files.add(relative)
    require(observed_files == expected_files, "packet file inventory differs")
    require(observed_directories == expected_directories, "packet directory inventory differs")


def replay_packet(packet_root_value: str | Path) -> Mapping[str, Any]:
    root = _directory(packet_root_value, "packet root")
    manifest, manifest_row, _ = _load_canonical_json(
        root / "manifest.json",
        "packet manifest",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    require(
        (manifest_row["sha256"], manifest_row["size"])
        == (PACKET_MANIFEST_SHA256, PACKET_MANIFEST_SIZE),
        "packet manifest authority differs",
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
        "packet manifest",
    )
    require(
        manifest["schema_version"] == PACKET_SCHEMA
        and manifest["status"] == PACKET_STATUS
        and (manifest["case_id"], manifest["iid"]) == (CASE_ID, IID)
        and type(manifest["frame_count"]) is int
        and manifest["frame_count"] == FRAME_COUNT
        and manifest["candidate_is_review_passed"] is False,
        "packet manifest identity/status differs",
    )
    records = manifest["premanifest_output_tree"]
    require(type(records) is list and bool(records), "packet premanifest tree is empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    for row in records:
        _exact_keys(row, ("path", "sha256", "size"), "packet tree row")
        relative = _relative_path(row["path"], "packet tree path").as_posix()
        require(relative not in seen, "packet tree path repeats")
        require(_is_sha256(row["sha256"]), "packet tree SHA-256 differs")
        require(type(row["size"]) is int and row["size"] > 0, "packet tree size differs")
        seen.add(relative)
        total_bytes += row["size"]
        normalized.append(dict(row))
    require(
        [row["path"] for row in normalized] == sorted(row["path"] for row in normalized),
        "packet tree order differs",
    )
    require(total_bytes <= MAX_PACKET_BYTES, "packet exceeds byte bound")
    digest = hashlib.sha256(canonical_json_bytes(normalized) + b"\n").hexdigest()
    require(
        digest == manifest["premanifest_output_tree_digest"] == PACKET_PREMANIFEST_DIGEST,
        "packet premanifest digest differs",
    )
    rows_by_path = {row["path"]: row for row in normalized}
    expected_files = set(rows_by_path) | {"manifest.json", "SHA256SUMS"}
    _walk_exact_packet(root, expected_files)
    for relative, expected in rows_by_path.items():
        row, _ = _stable_file(
            root / relative,
            "packet file %s" % relative,
            require_nlink1=True,
            maximum_bytes=MAX_SINGLE_PACKET_FILE_BYTES,
        )
        require(
            (row["sha256"], row["size"]) == (expected["sha256"], expected["size"]),
            "packet file bytes differ: %s" % relative,
        )
    sums_payload, sums_row, _ = _stable_bytes(
        root / "SHA256SUMS",
        "packet SHA256SUMS",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    sums = {row["path"]: row["sha256"] for row in normalized}
    sums["manifest.json"] = manifest_row["sha256"]
    expected_sums = "".join(
        "%s  %s\n" % (sums[name], name) for name in sorted(sums)
    ).encode("utf-8")
    require(sums_payload == expected_sums, "packet SHA256SUMS bytes differ")
    frames = manifest["frames"]
    require(type(frames) is list and len(frames) == FRAME_COUNT, "packet frames differ")
    support_rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        require(
            type(frame) is dict
            and type(frame.get("frame_index")) is int
            and frame.get("frame_index") == index,
            "packet frame order differs",
        )
        outputs = frame.get("outputs")
        require(type(outputs) is dict, "packet frame outputs differ")
        support = outputs.get("candidate_support")
        _exact_keys(support, ("path", "sha256", "size"), "packet support row")
        expected_path = "masks/candidate_support/%05d.png" % index
        require(
            support["path"] == expected_path
            and rows_by_path.get(expected_path) == support,
            "packet support is not inventory-bound: %d" % index,
        )
        support_rows.append(dict(support))
    authority = manifest["authority"]
    require(type(authority) is dict, "packet authority differs")
    for name in ("source_video", "masklet_receipt"):
        _exact_keys(authority.get(name), ("path", "sha256", "size"), "packet %s" % name)
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_row": manifest_row,
        "sha256sums_row": sums_row,
        "support_rows": support_rows,
        "file_count": len(expected_files),
        "premanifest_output_tree_digest": digest,
    }


def _outside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        fail("%s must be outside the immutable packet" % label)

    # Lexical comparison is insufficient on case-insensitive or
    # normalization-insensitive filesystems: e.g. ``packet`` and ``PACKET``
    # can name the same APFS directory while Path.relative_to() says they are
    # unrelated.  Walk every existing candidate ancestor and compare the
    # directory identity to the immutable packet root before any staging can
    # be created.
    try:
        root_named = root.lstat()
    except OSError as error:
        raise SupportPromotionHold("immutable packet root is unavailable") from error
    require(
        stat.S_ISDIR(root_named.st_mode) and not root.is_symlink(),
        "immutable packet root differs",
    )
    root_identity = (int(root_named.st_dev), int(root_named.st_ino))
    probe = path
    while True:
        try:
            named = probe.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise SupportPromotionHold(
                "%s ancestor could not be inspected" % label
            ) from error
        else:
            if stat.S_ISDIR(named.st_mode) and (
                int(named.st_dev),
                int(named.st_ino),
            ) == root_identity:
                fail("%s must be outside the immutable packet" % label)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent


def _completed_review_projection(
    generator: Any,
    value: Any,
    *,
    expected_slot: int,
    manifest_sha256: str,
) -> str:
    review = _exact_keys(
        value,
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
        "completed external review",
    )
    require(
        review["schema_version"] == generator.EXTERNAL_REVIEW_SCHEMA
        and type(review["reviewer_slot"]) is int
        and review["reviewer_slot"] == expected_slot
        and review["candidate_manifest_sha256"] == manifest_sha256
        and review["signature_or_external_receipt"] is None,
        "completed external review identity/manifest/signature differs",
    )
    identity = review["reviewer_identity"]
    affiliation = review["reviewer_affiliation_or_role"]
    require(
        type(identity) is str
        and bool(identity)
        and identity == identity.strip()
        and identity == unicodedata.normalize("NFC", identity),
        "completed external reviewer identity differs",
    )
    require(
        type(affiliation) is str
        and bool(affiliation)
        and affiliation == affiliation.strip(),
        "completed external reviewer affiliation/role differs",
    )
    reviewed_at = review["reviewed_at_utc"]
    require(
        type(reviewed_at) is str
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            reviewed_at,
        )
        is not None,
        "completed external review timestamp differs",
    )
    try:
        parsed_reviewed_at = datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise SupportPromotionHold("completed external review timestamp differs") from error
    require(
        parsed_reviewed_at.strftime("%Y-%m-%dT%H:%M:%SZ") == reviewed_at,
        "completed external review timestamp differs",
    )
    independence = review["independence_attestation"]
    _exact_keys(
        independence,
        generator.EXTERNAL_INDEPENDENCE_KEYS,
        "completed external review independence",
    )
    require(
        all(independence[name] is True for name in generator.EXTERNAL_INDEPENDENCE_KEYS),
        "completed external review independence declaration differs",
    )
    require(
        review["all_81_native_frames_reviewed"] is True
        and review["claim_limits_acknowledged"] is True
        and review["instructions"] == list(generator.EXTERNAL_REVIEW_INSTRUCTIONS),
        "completed external review all-frame decision/limits differs",
    )
    frames = review["frames"]
    require(
        type(frames) is list and len(frames) == FRAME_COUNT,
        "completed external review frame count differs",
    )
    all_pass = True
    coverage_fields = (
        "bone_coverage",
        "contact_shadow_coverage",
        "halo_and_adjacent_ground_coverage",
        "dog_and_guard_protection",
    )
    for frame_index, frame in enumerate(frames):
        _exact_keys(
            frame,
            (
                "frame_index",
                *coverage_fields,
                "boundary_edit_requested",
                "notes",
                "decision",
            ),
            "completed external review frame",
        )
        require(
            type(frame["frame_index"]) is int
            and frame["frame_index"] == frame_index,
            "completed external review frame order differs",
        )
        require(
            all(frame[name] in ("PASS", "FAIL") for name in coverage_fields)
            and type(frame["boundary_edit_requested"]) is bool
            and type(frame["notes"]) is str
            and bool(frame["notes"].strip()),
            "completed external review frame ballot differs: %d" % frame_index,
        )
        frame_pass = (
            all(frame[name] == "PASS" for name in coverage_fields)
            and frame["boundary_edit_requested"] is False
        )
        require(
            frame["decision"] == ("PASS" if frame_pass else "FAIL"),
            "completed external review frame decision is inconsistent: %d"
            % frame_index,
        )
        all_pass = all_pass and frame_pass
    overall = "PASS" if all_pass else "FAIL"
    require(
        review["overall_decision"] == overall,
        "completed external review overall decision is inconsistent",
    )
    return overall


def _review_draft_input(
    generator: Any,
    packet: Mapping[str, Any],
    *,
    expected_slot: int,
    draft_path_value: str | Path,
    evidence_path_value: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    require(
        type(expected_slot) is int and expected_slot in (1, 2),
        "reviewer slot differs",
    )
    packet_root = Path(packet["root"])
    draft_path = _canonical_absolute(draft_path_value, "completed review draft")
    evidence_path = _canonical_absolute(evidence_path_value, "review evidence")
    _outside(draft_path, packet_root, "completed review draft")
    _outside(evidence_path, packet_root, "review evidence")
    require(draft_path != evidence_path, "review draft/evidence path repeats")
    draft, draft_row, draft_identity = _load_json_draft(
        draft_path,
        "completed review draft",
        maximum_bytes=MAX_EXTERNAL_RECEIPT_BYTES,
    )
    _, evidence_row, evidence_identity = _stable_bytes(
        evidence_path,
        "review evidence",
        require_nlink1=True,
        maximum_bytes=MAX_EXTERNAL_EVIDENCE_BYTES,
    )
    require(
        (draft_identity[0], draft_identity[1])
        != (evidence_identity[0], evidence_identity[1]),
        "review draft/evidence inode repeats",
    )
    receipt = dict(draft)
    require(
        receipt.get("candidate_manifest_sha256") is None,
        "completed review draft candidate manifest must remain null for sealing",
    )
    require(
        receipt.get("signature_or_external_receipt") is None,
        "completed review draft is already sealed",
    )
    receipt["candidate_manifest_sha256"] = packet["manifest_row"]["sha256"]
    overall = _completed_review_projection(
        generator,
        receipt,
        expected_slot=expected_slot,
        manifest_sha256=packet["manifest_row"]["sha256"],
    )
    projection_sha256 = object_sha256(receipt)
    receipt["signature_or_external_receipt"] = {
        "kind": generator.EXTERNAL_SIGNATURE_KIND,
        "review_projection_sha256": projection_sha256,
        "evidence_sha256": evidence_row["sha256"],
        "evidence_size": evidence_row["size"],
    }
    return receipt, draft_row, evidence_row, overall


def _review_inputs(
    generator: Any,
    packet: Mapping[str, Any],
    receipt_values: Sequence[str | Path],
    evidence_values: Sequence[str | Path],
) -> list[Mapping[str, Any]]:
    require(len(receipt_values) == len(evidence_values) == 2, "exactly two reviews are required")
    packet_root = Path(packet["root"])
    external_rows: list[Mapping[str, Any]] = []
    identities: list[str] = []
    all_paths: list[Path] = []
    all_inodes: list[tuple[int, int]] = []
    for offset in range(2):
        slot = offset + 1
        receipt_path = _canonical_absolute(receipt_values[offset], "review receipt")
        evidence_path = _canonical_absolute(evidence_values[offset], "review evidence")
        _outside(receipt_path, packet_root, "review receipt")
        _outside(evidence_path, packet_root, "review evidence")
        receipt, receipt_row, receipt_identity = _load_canonical_json(
            receipt_path,
            "external review receipt %d" % slot,
            require_nlink1=True,
            maximum_bytes=MAX_EXTERNAL_RECEIPT_BYTES,
        )
        _, evidence_row, evidence_identity = _stable_bytes(
            evidence_path,
            "external review evidence %d" % slot,
            require_nlink1=True,
            maximum_bytes=MAX_EXTERNAL_EVIDENCE_BYTES,
        )
        formal = {
            "reviewer_slot": slot,
            "receipt": receipt_row,
            "reviewer_identity": receipt.get("reviewer_identity"),
            "reviewer_affiliation_or_role": receipt.get("reviewer_affiliation_or_role"),
            "reviewed_at_utc": receipt.get("reviewed_at_utc"),
            "independence_attestation": receipt.get("independence_attestation"),
            "signature": receipt.get("signature_or_external_receipt"),
            "evidence": evidence_row,
        }
        generator._validate_external_review(
            formal,
            expected_slot=slot,
            manifest_sha256=packet["manifest_row"]["sha256"],
        )
        identity = receipt["reviewer_identity"]
        require(
            identity == unicodedata.normalize("NFC", identity),
            "reviewer identity is not NFC-normalized",
        )
        identities.append(identity)
        all_paths.extend((receipt_path, evidence_path))
        all_inodes.extend(
            (
                (receipt_identity[0], receipt_identity[1]),
                (evidence_identity[0], evidence_identity[1]),
            )
        )
        external_rows.append(formal)
    require(
        identities[0].casefold() != identities[1].casefold(),
        "external reviewer identities are not distinct after casefold",
    )
    require(len({str(path) for path in all_paths}) == 4, "review input path repeats")
    require(len(set(all_inodes)) == 4, "review input inode repeats")
    for field in ("receipt", "evidence"):
        require(
            external_rows[0][field]["path"] != external_rows[1][field]["path"],
            "external reviewer %s path repeats" % field,
        )
        require(
            external_rows[0][field]["sha256"]
            != external_rows[1][field]["sha256"],
            "external reviewer %s SHA256 repeats" % field,
        )
    return external_rows


def _deployment_authorities(
    generator: Any,
    packet: Mapping[str, Any],
    source_path_value: str | Path | None,
    sam2_path_value: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = packet["manifest"]["authority"]
    source_path = (
        Path(authority["source_video"]["path"])
        if source_path_value is None
        else _canonical_absolute(source_path_value, "deployment source")
    )
    sam2_path = (
        Path(authority["masklet_receipt"]["path"])
        if sam2_path_value is None
        else _canonical_absolute(sam2_path_value, "deployment SAM2 receipt")
    )
    source_row, _ = _stable_file(
        source_path,
        "deployment source",
        require_nlink1=False,
        maximum_bytes=64 * 1024 * 1024,
    )
    sam2_row, _ = _stable_file(
        sam2_path,
        "deployment SAM2 receipt",
        require_nlink1=False,
        maximum_bytes=2 * 1024 * 1024,
    )
    require(
        (source_row["sha256"], source_row["size"])
        == (generator.SOURCE_SHA256, generator.SOURCE_SIZE)
        == (authority["source_video"]["sha256"], authority["source_video"]["size"]),
        "deployment source authority differs",
    )
    require(
        (sam2_row["sha256"], sam2_row["size"])
        == (generator.SAM2_RECEIPT_SHA256, generator.SAM2_RECEIPT_SIZE)
        == (authority["masklet_receipt"]["sha256"], authority["masklet_receipt"]["size"]),
        "deployment SAM2 authority differs",
    )
    return source_row, sam2_row


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        require(type(written) is int and written > 0, "output write made no progress")
        offset += written


def _write_stage_file(path: Path, payload: bytes) -> dict[str, Any]:
    require(path.is_absolute() and path.parent.is_dir(), "stage output parent differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        named = path.lstat()
        require(
            _identity(held) == _identity(named)
            and stat.S_ISREG(held.st_mode)
            and int(held.st_nlink) == 1
            and stat.S_IMODE(held.st_mode) == 0o400,
            "stage output identity differs",
        )
    finally:
        os.close(descriptor)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _copy_stage_file(source: Path, target: Path, label: str, maximum_bytes: int) -> dict[str, Any]:
    payload, before, _ = _stable_bytes(
        source,
        label,
        require_nlink1=True,
        maximum_bytes=maximum_bytes,
    )
    written = _write_stage_file(target, payload)
    after, _ = _stable_file(
        source,
        label,
        require_nlink1=True,
        maximum_bytes=maximum_bytes,
    )
    require(after == before, "%s changed while copying" % label)
    require(
        (written["sha256"], written["size"]) == (before["sha256"], before["size"]),
        "%s copied bytes differ" % label,
    )
    return written


def _final_row(stage_row: Mapping[str, Any], stage: Path, final: Path) -> dict[str, Any]:
    stage_path = Path(stage_row["path"])
    relative = stage_path.relative_to(stage)
    return {
        "path": str(final / relative),
        "sha256": stage_row["sha256"],
        "size": stage_row["size"],
    }


def _retarget(value: Any, source_root: Path, destination_root: Path) -> Any:
    if type(value) is dict:
        result = {key: _retarget(child, source_root, destination_root) for key, child in value.items()}
        # A bare file row has exactly these three keys, while a frame-mask row
        # carries the same file fields alongside its frame index and review
        # booleans.  Both must be retargeted for the pre-publication stage
        # replay; source/SAM2 rows are outside source_root and remain unchanged.
        if {"path", "sha256", "size"} <= set(result) and type(result["path"]) is str:
            path = Path(result["path"])
            try:
                relative = path.relative_to(source_root)
            except ValueError:
                pass
            else:
                result["path"] = str(destination_root / relative)
        return result
    if type(value) is list:
        return [_retarget(child, source_root, destination_root) for child in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _write_stage_file(path, canonical_json_bytes(value) + b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _rename_noreplace(
    source: Path,
    destination: Path,
    publication_state: dict[str, bool] | None = None,
) -> None:
    if publication_state is not None:
        require(
            publication_state == {"committed": False},
            "publication state differs before rename",
        )
    require(
        source.is_absolute()
        and destination.is_absolute()
        and source.parent == destination.parent
        and source != destination,
        "atomic publication requires one same-parent pair",
    )
    source_before = source.lstat()
    parent_before = source.parent.lstat()
    require(
        stat.S_ISDIR(source_before.st_mode)
        and not source.is_symlink()
        and stat.S_ISDIR(parent_before.st_mode)
        and not source.parent.is_symlink(),
        "publication source/parent differs",
    )
    parent_descriptor = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_parent = os.fstat(parent_descriptor)
        parent_identity = (int(opened_parent.st_dev), int(opened_parent.st_ino))
        require(
            parent_identity == (int(parent_before.st_dev), int(parent_before.st_ino)),
            "publication parent changed while opening",
        )
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            function = getattr(library, "renameat2", None)
            require(function is not None, "renameat2(RENAME_NOREPLACE) is unavailable")
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            result = function(
                parent_descriptor,
                os.fsencode(source.name),
                parent_descriptor,
                os.fsencode(destination.name),
                1,
            )
        elif sys.platform == "darwin":
            function = getattr(library, "renameatx_np", None)
            require(function is not None, "renameatx_np(RENAME_EXCL) is unavailable")
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            result = function(
                parent_descriptor,
                os.fsencode(source.name),
                parent_descriptor,
                os.fsencode(destination.name),
                0x00000004,
            )
        else:
            fail("atomic no-replace directory publication is unsupported")
        if result != 0:
            number = ctypes.get_errno()
            if number in (errno.EEXIST, errno.ENOTEMPTY):
                fail("promotion output already exists")
            fail("atomic no-replace publication failed: errno=%d" % number)
        if publication_state is not None:
            publication_state["committed"] = True
        os.fsync(parent_descriptor)
        try:
            os.stat(source.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            fail("publication staging name remains")
        published = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        require(
            stat.S_ISDIR(published.st_mode)
            and (int(published.st_dev), int(published.st_ino))
            == (int(source_before.st_dev), int(source_before.st_ino)),
            "published directory identity differs",
        )
    finally:
        os.close(parent_descriptor)


def _sealed_review_files(
    generator: Any,
    packet: Mapping[str, Any],
    *,
    expected_slot: int,
    receipt_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    receipt, observed_receipt, receipt_identity = _load_canonical_json(
        receipt_row["path"],
        "sealed external review receipt",
        require_nlink1=True,
        maximum_bytes=MAX_EXTERNAL_RECEIPT_BYTES,
    )
    _, observed_evidence, evidence_identity = _stable_bytes(
        evidence_row["path"],
        "sealed external review evidence",
        require_nlink1=True,
        maximum_bytes=MAX_EXTERNAL_EVIDENCE_BYTES,
    )
    require(
        observed_receipt == receipt_row and observed_evidence == evidence_row,
        "sealed external review file row differs",
    )
    require(
        (receipt_identity[0], receipt_identity[1])
        != (evidence_identity[0], evidence_identity[1]),
        "sealed external review receipt/evidence inode repeats",
    )
    signature = receipt.get("signature_or_external_receipt")
    _exact_keys(
        signature,
        ("kind", "review_projection_sha256", "evidence_sha256", "evidence_size"),
        "sealed external review signature",
    )
    require(
        signature["kind"] == generator.EXTERNAL_SIGNATURE_KIND
        and _is_sha256(signature["review_projection_sha256"])
        and signature["evidence_sha256"] == evidence_row["sha256"]
        and signature["evidence_size"] == evidence_row["size"],
        "sealed external review evidence declaration differs",
    )
    projection = dict(receipt)
    projection["signature_or_external_receipt"] = None
    overall = _completed_review_projection(
        generator,
        projection,
        expected_slot=expected_slot,
        manifest_sha256=packet["manifest_row"]["sha256"],
    )
    require(
        object_sha256(projection) == signature["review_projection_sha256"],
        "sealed external review projection binding differs",
    )
    if overall == "PASS":
        formal = {
            "reviewer_slot": expected_slot,
            "receipt": dict(receipt_row),
            "reviewer_identity": receipt["reviewer_identity"],
            "reviewer_affiliation_or_role": receipt[
                "reviewer_affiliation_or_role"
            ],
            "reviewed_at_utc": receipt["reviewed_at_utc"],
            "independence_attestation": receipt["independence_attestation"],
            "signature": signature,
            "evidence": dict(evidence_row),
        }
        generator._validate_external_review(
            formal,
            expected_slot=expected_slot,
            manifest_sha256=packet["manifest_row"]["sha256"],
        )
    return receipt, overall


def _review_submission_document(
    *,
    program: Mapping[str, Any],
    packet: Mapping[str, Any],
    receipt: Mapping[str, Any],
    overall: str,
    completed_draft_row: Mapping[str, Any],
    receipt_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
) -> dict[str, Any]:
    document = {
        "schema_version": REVIEW_SUBMISSION_SCHEMA,
        "status": (
            REVIEW_SUBMISSION_PASS_STATUS
            if overall == "PASS"
            else REVIEW_SUBMISSION_FAIL_STATUS
        ),
        "case_id": CASE_ID,
        "iid": IID,
        "sealer": dict(program),
        "generator": _generator_identity(),
        "candidate_manifest": {
            "sha256": packet["manifest_row"]["sha256"],
            "size": packet["manifest_row"]["size"],
        },
        "reviewer_slot": receipt["reviewer_slot"],
        "declared_reviewer_identity": receipt["reviewer_identity"],
        "declared_reviewer_affiliation_or_role": receipt[
            "reviewer_affiliation_or_role"
        ],
        "reviewed_at_utc": receipt["reviewed_at_utc"],
        "overall_decision": overall,
        "promotion_eligible_by_this_ballot_alone": False,
        "completed_draft_input": {
            "sha256": completed_draft_row["sha256"],
            "size": completed_draft_row["size"],
        },
        "receipt": {
            "sha256": receipt_row["sha256"],
            "size": receipt_row["size"],
        },
        "evidence": {
            "sha256": evidence_row["sha256"],
            "size": evidence_row["size"],
        },
        "claim_limits": dict(REVIEW_SUBMISSION_CLAIM_LIMITS),
    }
    document["submission_digest"] = object_sha256(document)
    return document


def _review_submission_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for name, maximum in (
        ("COMPLETE", 1024),
        ("completed_draft.input.json", MAX_EXTERNAL_RECEIPT_BYTES),
        ("evidence.bin", MAX_EXTERNAL_EVIDENCE_BYTES),
        ("receipt.json", MAX_EXTERNAL_RECEIPT_BYTES),
        ("submission.json", 2 * 1024 * 1024),
    ):
        row, _ = _stable_file(
            root / name,
            "review submission %s" % name,
            require_nlink1=True,
            maximum_bytes=maximum,
        )
        records.append({"path": name, "sha256": row["sha256"], "size": row["size"]})
    return sorted(records, key=lambda row: row["path"])


def _replay_review_submission(
    root_value: str | Path,
    generator: Any,
    packet: Mapping[str, Any],
    *,
    expected_slot: int,
    expected_program: Mapping[str, Any],
) -> Mapping[str, Any]:
    require(
        _program_identity() == expected_program,
        "review submission sealer program differs",
    )
    root = _directory(root_value, "review submission")
    named_root = root.lstat()
    require(
        stat.S_IMODE(named_root.st_mode) == 0o700,
        "review submission root mode differs",
    )
    expected_names = {
        "COMPLETE",
        "completed_draft.input.json",
        "evidence.bin",
        "receipt.json",
        "submission.json",
        "SHA256SUMS",
    }
    observed_names = set()
    for child in root.iterdir():
        named = child.lstat()
        require(
            stat.S_ISREG(named.st_mode)
            and not child.is_symlink()
            and int(named.st_nlink) == 1
            and stat.S_IMODE(named.st_mode) == 0o400,
            "review submission file identity/mode differs",
        )
        observed_names.add(child.name)
    require(observed_names == expected_names, "review submission inventory differs")
    complete, _, _ = _stable_bytes(
        root / "COMPLETE",
        "review submission COMPLETE",
        require_nlink1=True,
        maximum_bytes=1024,
    )
    require(
        complete == REVIEW_SUBMISSION_COMPLETE_BYTES,
        "review submission COMPLETE differs",
    )
    records = _review_submission_records(root)
    expected_sums = "".join(
        "%s  %s\n" % (row["sha256"], row["path"]) for row in records
    ).encode("utf-8")
    sums, _, _ = _stable_bytes(
        root / "SHA256SUMS",
        "review submission SHA256SUMS",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    require(sums == expected_sums, "review submission checksums differ")
    by_name = {row["path"]: row for row in records}
    receipt_row = {
        "path": str(root / "receipt.json"),
        "sha256": by_name["receipt.json"]["sha256"],
        "size": by_name["receipt.json"]["size"],
    }
    evidence_row = {
        "path": str(root / "evidence.bin"),
        "sha256": by_name["evidence.bin"]["sha256"],
        "size": by_name["evidence.bin"]["size"],
    }
    expected_receipt, draft_row, replayed_evidence, expected_overall = _review_draft_input(
        generator,
        packet,
        expected_slot=expected_slot,
        draft_path_value=root / "completed_draft.input.json",
        evidence_path_value=root / "evidence.bin",
    )
    require(
        replayed_evidence == evidence_row,
        "review submission copied evidence differs from completed draft binding",
    )
    receipt, overall = _sealed_review_files(
        generator,
        packet,
        expected_slot=expected_slot,
        receipt_row=receipt_row,
        evidence_row=evidence_row,
    )
    require(
        receipt == expected_receipt and overall == expected_overall,
        "review submission receipt modifies human-authored values",
    )
    submission, _, _ = _load_canonical_json(
        root / "submission.json",
        "review submission receipt",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    expected_submission = _review_submission_document(
        program=expected_program,
        packet=packet,
        receipt=receipt,
        overall=overall,
        completed_draft_row=draft_row,
        receipt_row=receipt_row,
        evidence_row=evidence_row,
    )
    require(submission == expected_submission, "review submission receipt differs")
    return {
        "root": str(root),
        "submission": submission,
        "receipt": receipt_row,
        "evidence": evidence_row,
        "file_count": len(expected_names),
    }


def _bundle_records(root: Path, *, exclude_sums: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        named = path.lstat()
        require(not path.is_symlink(), "bundle contains a symlink")
        if stat.S_ISDIR(named.st_mode):
            continue
        require(
            stat.S_ISREG(named.st_mode)
            and int(named.st_nlink) == 1
            and stat.S_IMODE(named.st_mode) == 0o400,
            "bundle contains a special or non-nlink1 file",
        )
        if exclude_sums and relative == "SHA256SUMS":
            continue
        row, _ = _stable_file(
            path,
            "bundle file %s" % relative,
            require_nlink1=True,
            maximum_bytes=MAX_SINGLE_PACKET_FILE_BYTES,
        )
        records.append({"path": relative, "sha256": row["sha256"], "size": row["size"]})
    return records


def _expected_bundle_files() -> set[str]:
    return {
        "candidate_packet/manifest.json",
        "candidate_packet/SHA256SUMS",
        *{
            "masks/candidate_support/%05d.png" % index
            for index in range(FRAME_COUNT)
        },
        "external_reviews/reviewer_1_receipt.json",
        "external_reviews/reviewer_1_evidence.bin",
        "external_reviews/reviewer_2_receipt.json",
        "external_reviews/reviewer_2_evidence.bin",
        "support_review_receipt.json",
        "promotion_receipt.json",
        "SHA256SUMS",
        "COMPLETE",
    }


def _program_identity() -> dict[str, Any]:
    _, row, _ = _stable_bytes(
        Path(__file__).resolve(strict=True),
        "promotion program",
        require_nlink1=False,
        maximum_bytes=2 * 1024 * 1024,
    )
    return {"sha256": row["sha256"], "size": row["size"]}


def _generator_identity() -> dict[str, Any]:
    return {
        "name": GENERATOR_PATH.name,
        "sha256": GENERATOR_SHA256,
        "size": GENERATOR_SIZE,
    }


def _expected_prepublication_records(
    formal: Mapping[str, Any],
    final: Path,
    formal_row: Mapping[str, Any],
    promotion_row: Mapping[str, Any],
    complete_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[Mapping[str, Any]] = [
        formal["candidate_packet"]["manifest"],
        formal["candidate_packet"]["sha256sums"],
        *formal["frame_masks"],
        *[
            child
            for review in formal["external_reviews"]
            for child in (review["receipt"], review["evidence"])
        ],
        formal_row,
        promotion_row,
        complete_row,
    ]
    records: list[dict[str, Any]] = []
    for row in rows:
        path = _canonical_absolute(row["path"], "expected promotion file")
        try:
            relative = path.relative_to(final).as_posix()
        except ValueError as error:
            raise SupportPromotionHold("expected promotion file escapes final root") from error
        records.append(
            {
                "path": relative,
                "sha256": row["sha256"],
                "size": row["size"],
            }
        )
    records.sort(key=lambda row: row["path"])
    require(
        {row["path"] for row in records}
        == _expected_bundle_files() - {"SHA256SUMS"}
        and len(records) == len({row["path"] for row in records}),
        "expected prepublication record inventory differs",
    )
    return records


def _verify_prepublication_seal(
    stage: Path,
    expected_records: Sequence[Mapping[str, Any]],
    sums_payload: bytes,
    formal_payload: bytes,
    promotion_payload: bytes,
) -> None:
    _replay_bundle_inventory(stage)
    require(
        _bundle_records(stage, exclude_sums=True) == list(expected_records),
        "promotion staging records differ",
    )
    for relative, expected, maximum in (
        ("SHA256SUMS", sums_payload, 2 * 1024 * 1024),
        ("support_review_receipt.json", formal_payload, 2 * 1024 * 1024),
        ("promotion_receipt.json", promotion_payload, 2 * 1024 * 1024),
        ("COMPLETE", COMPLETE_BYTES, 1024),
    ):
        observed, _, _ = _stable_bytes(
            stage / relative,
            "staged %s" % relative,
            require_nlink1=True,
            maximum_bytes=maximum,
        )
        require(observed == expected, "staged %s bytes differ" % relative)


def _replay_bundle_inventory(root: Path) -> None:
    root_named = root.lstat()
    require(
        stat.S_ISDIR(root_named.st_mode)
        and not root.is_symlink()
        and stat.S_IMODE(root_named.st_mode) == 0o700,
        "promotion bundle root directory differs",
    )
    expected_files = _expected_bundle_files()
    expected_directories = {
        "",
        "candidate_packet",
        "masks",
        "masks/candidate_support",
        "external_reviews",
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = {""}

    def onerror(error: OSError) -> None:
        raise SupportPromotionHold("promotion bundle walk failed") from error

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=onerror
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            child = directory_path / dirname
            named = child.lstat()
            require(
                stat.S_ISDIR(named.st_mode)
                and not child.is_symlink()
                and stat.S_IMODE(named.st_mode) == 0o700,
                "promotion bundle directory differs",
            )
            observed_directories.add(child.relative_to(root).as_posix())
        for filename in filenames:
            child = directory_path / filename
            named = child.lstat()
            require(
                stat.S_ISREG(named.st_mode)
                and not child.is_symlink()
                and int(named.st_nlink) == 1
                and stat.S_IMODE(named.st_mode) == 0o400,
                "promotion bundle file identity/mode differs",
            )
            observed_files.add(child.relative_to(root).as_posix())
    require(observed_files == expected_files, "promotion bundle file inventory differs")
    require(
        observed_directories == expected_directories,
        "promotion bundle directory inventory differs",
    )


def _replay_bundle(root_value: str | Path, generator: Any, source: Mapping[str, Any], sam2: Mapping[str, Any]) -> Mapping[str, Any]:
    root = _directory(root_value, "promotion bundle")
    _replay_bundle_inventory(root)
    sums_payload, _, _ = _stable_bytes(
        root / "SHA256SUMS",
        "promotion bundle SHA256SUMS",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    records = _bundle_records(root, exclude_sums=True)
    expected = "".join(
        "%s  %s\n" % (row["sha256"], row["path"]) for row in records
    ).encode("utf-8")
    require(sums_payload == expected, "promotion bundle checksum inventory differs")
    complete, _, _ = _stable_bytes(
        root / "COMPLETE",
        "promotion COMPLETE marker",
        require_nlink1=True,
        maximum_bytes=1024,
    )
    require(complete == COMPLETE_BYTES, "promotion COMPLETE marker differs")
    formal_path = root / "support_review_receipt.json"
    formal_row, compact = generator.validate_support_review(
        formal_path,
        source_row=source,
        sam2_row=sam2,
    )
    require(len(compact) == FRAME_COUNT, "promotion formal support count differs")
    formal, _, _ = _load_canonical_json(
        formal_path,
        "formal support review receipt",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    promotion, _, _ = _load_canonical_json(
        root / "promotion_receipt.json",
        "promotion receipt",
        require_nlink1=True,
        maximum_bytes=2 * 1024 * 1024,
    )
    _exact_keys(
        promotion,
        (
            "schema_version",
            "status",
            "case_id",
            "iid",
            "program",
            "generator",
            "candidate_packet",
            "source",
            "sam2_receipt",
            "external_reviews",
            "support_review_receipt",
            "frame_count",
            "claim_limits",
            "promotion_digest",
        ),
        "promotion receipt",
    )
    _exact_keys(promotion["program"], ("sha256", "size"), "promotion program")
    _exact_keys(
        promotion["generator"],
        ("name", "sha256", "size"),
        "promotion generator",
    )
    require(
        promotion["schema_version"] == PROMOTION_SCHEMA
        and promotion["status"] == PROMOTION_STATUS
        and (promotion["case_id"], promotion["iid"]) == (CASE_ID, IID)
        and promotion["source"] == source
        and promotion["sam2_receipt"] == sam2
        and promotion["candidate_packet"] == formal["candidate_packet"]
        and promotion["external_reviews"] == formal["external_reviews"]
        and promotion["support_review_receipt"] == formal_row
        and promotion["program"] == _program_identity()
        and promotion["generator"] == _generator_identity()
        and type(promotion["frame_count"]) is int
        and promotion["frame_count"] == FRAME_COUNT
        and promotion["claim_limits"] == PROMOTION_CLAIM_LIMITS,
        "promotion receipt binding differs",
    )
    unsigned = dict(promotion)
    observed = unsigned.pop("promotion_digest")
    require(observed == object_sha256(unsigned), "promotion receipt digest differs")
    return {
        "root": str(root),
        "support_review_receipt": formal_row,
        "promotion_digest": observed,
        "file_count": len(records) + 1,
    }


def _promote_once(
    *,
    packet_root: str | Path,
    reviewer_receipts: Sequence[str | Path],
    reviewer_evidence: Sequence[str | Path],
    output_root: str | Path,
    source_path: str | Path | None = None,
    sam2_receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    program = _program_identity()
    generator = _load_generator()
    final = _canonical_absolute(output_root, "promotion output")
    parent = _directory(final.parent, "promotion output parent")
    require(final.parent == parent, "promotion output parent differs")
    require(not os.path.lexists(final), "promotion output already exists")
    packet = replay_packet(packet_root)
    _outside(final, Path(packet["root"]), "promotion output")
    external_inputs = _review_inputs(
        generator,
        packet,
        reviewer_receipts,
        reviewer_evidence,
    )
    source_row, sam2_row = _deployment_authorities(
        generator,
        packet,
        source_path,
        sam2_receipt_path,
    )
    stage = Path(tempfile.mkdtemp(prefix=".%s.partial." % final.name, dir=str(parent)))
    publication_state = {"committed": False}
    try:
        require(stage.parent == parent and stage != final, "promotion staging path differs")
        for relative in (
            "candidate_packet",
            "masks",
            "masks/candidate_support",
            "external_reviews",
        ):
            (stage / relative).mkdir(mode=0o700)

        manifest_stage = _copy_stage_file(
            Path(packet["root"]) / "manifest.json",
            stage / "candidate_packet/manifest.json",
            "packet manifest",
            2 * 1024 * 1024,
        )
        packet_sums_stage = _copy_stage_file(
            Path(packet["root"]) / "SHA256SUMS",
            stage / "candidate_packet/SHA256SUMS",
            "packet SHA256SUMS",
            2 * 1024 * 1024,
        )
        mask_rows: list[dict[str, Any]] = []
        for index, expected in enumerate(packet["support_rows"]):
            source_mask = Path(packet["root"]) / expected["path"]
            target_mask = stage / ("masks/candidate_support/%05d.png" % index)
            copied = _copy_stage_file(
                source_mask,
                target_mask,
                "candidate support mask %d" % index,
                2 * 1024 * 1024,
            )
            require(
                (copied["sha256"], copied["size"])
                == (expected["sha256"], expected["size"]),
                "candidate support copy differs: %d" % index,
            )
            mask_rows.append(
                {
                    "frame_index": index,
                    **_final_row(copied, stage, final),
                    "bone_and_cast_shadow_covered": True,
                    "native_resolution_reviewed": True,
                }
            )

        formal_external: list[Mapping[str, Any]] = []
        for index, input_row in enumerate(external_inputs):
            slot = index + 1
            receipt_copy = _copy_stage_file(
                Path(input_row["receipt"]["path"]),
                stage / ("external_reviews/reviewer_%d_receipt.json" % slot),
                "external review receipt %d" % slot,
                MAX_EXTERNAL_RECEIPT_BYTES,
            )
            evidence_copy = _copy_stage_file(
                Path(input_row["evidence"]["path"]),
                stage / ("external_reviews/reviewer_%d_evidence.bin" % slot),
                "external review evidence %d" % slot,
                MAX_EXTERNAL_EVIDENCE_BYTES,
            )
            formal_external.append(
                {
                    "reviewer_slot": slot,
                    "receipt": _final_row(receipt_copy, stage, final),
                    "reviewer_identity": input_row["reviewer_identity"],
                    "reviewer_affiliation_or_role": input_row["reviewer_affiliation_or_role"],
                    "reviewed_at_utc": input_row["reviewed_at_utc"],
                    "independence_attestation": input_row["independence_attestation"],
                    "signature": input_row["signature"],
                    "evidence": _final_row(evidence_copy, stage, final),
                }
            )

        formal: dict[str, Any] = {
            "schema_version": generator.SUPPORT_REVIEW_SCHEMA,
            "status": generator.SUPPORT_REVIEW_STATUS,
            "case_id": CASE_ID,
            "iid": IID,
            "candidate_packet": {
                "manifest": _final_row(manifest_stage, stage, final),
                "sha256sums": _final_row(packet_sums_stage, stage, final),
                "premanifest_output_tree_digest": PACKET_PREMANIFEST_DIGEST,
            },
            "source": source_row,
            "sam2_receipt": sam2_row,
            "external_reviews": formal_external,
            "protocol": {
                "native_resolution_704x736": True,
                "all_81_frames_reviewed_by_each": True,
                "required_external_reviewers": 2,
                "bone_covered_all_frames_by_each": True,
                "cast_shadow_and_halo_covered_all_frames_by_each": True,
                "minimum_bone_dilation_pixels": 8,
                "old_dilate3_reused": False,
                "dog_guard_excluded_all_frames": True,
            },
            "frame_masks": mask_rows,
            "claim_limits": dict(generator.SUPPORT_REVIEW_CLAIM_LIMITS),
        }
        formal["review_digest"] = generator.object_sha256(formal)
        formal_payload = generator.canonical_json_bytes(formal) + b"\n"
        formal_stage_row = _write_stage_file(
            stage / "support_review_receipt.json",
            formal_payload,
        )
        formal_final_row = _final_row(formal_stage_row, stage, final)

        # Validate an exact stage-path isomorph before publication.  The final
        # receipt itself is validated again after the atomic rename.
        stage_formal = _retarget(formal, final, stage)
        stage_formal.pop("review_digest")
        stage_formal["review_digest"] = generator.object_sha256(stage_formal)
        audit_path = stage / ".support_review_receipt.stage-audit.json"
        _write_stage_file(
            audit_path,
            generator.canonical_json_bytes(stage_formal) + b"\n",
        )
        generator.validate_support_review(
            audit_path,
            source_row=source_row,
            sam2_row=sam2_row,
        )
        audit_path.unlink()
        _fsync_directory(stage)

        promotion: dict[str, Any] = {
            "schema_version": PROMOTION_SCHEMA,
            "status": PROMOTION_STATUS,
            "case_id": CASE_ID,
            "iid": IID,
            "program": program,
            "generator": _generator_identity(),
            "candidate_packet": formal["candidate_packet"],
            "source": source_row,
            "sam2_receipt": sam2_row,
            "external_reviews": formal_external,
            "support_review_receipt": formal_final_row,
            "frame_count": FRAME_COUNT,
            "claim_limits": dict(PROMOTION_CLAIM_LIMITS),
        }
        promotion["promotion_digest"] = object_sha256(promotion)
        promotion_payload = canonical_json_bytes(promotion) + b"\n"
        promotion_stage_row = _write_stage_file(
            stage / "promotion_receipt.json",
            promotion_payload,
        )
        complete_stage_row = _write_stage_file(stage / "COMPLETE", COMPLETE_BYTES)
        expected_records = _expected_prepublication_records(
            formal,
            final,
            formal_final_row,
            _final_row(promotion_stage_row, stage, final),
            _final_row(complete_stage_row, stage, final),
        )
        sums_payload = "".join(
            "%s  %s\n" % (row["sha256"], row["path"])
            for row in expected_records
        ).encode("utf-8")
        _write_stage_file(stage / "SHA256SUMS", sums_payload)
        _fsync_tree(stage)

        # Recheck every external input and the full packet immediately before
        # the sole publication operation.  Any drift leaves only quarantined
        # staging and never creates the requested final path.
        program_after = _program_identity()
        generator_after = _load_generator()
        require(program_after == program, "promotion program changed before publication")
        packet_after = replay_packet(packet_root)
        require(
            packet_after["manifest_row"] == packet["manifest_row"]
            and packet_after["sha256sums_row"] == packet["sha256sums_row"],
            "packet changed before publication",
        )
        external_after = _review_inputs(
            generator_after,
            packet_after,
            reviewer_receipts,
            reviewer_evidence,
        )
        require(external_after == external_inputs, "external review inputs changed")
        source_after, sam2_after = _deployment_authorities(
            generator_after,
            packet_after,
            source_path,
            sam2_receipt_path,
        )
        require(source_after == source_row and sam2_after == sam2_row, "deployment authority changed")
        # Stage replay uses a transient retargeted receipt because the sealed
        # final receipt deliberately contains the not-yet-created final paths.
        _verify_prepublication_seal(
            stage,
            expected_records,
            sums_payload,
            formal_payload,
            promotion_payload,
        )
        _rename_noreplace(stage, final, publication_state)
        generator_published = _load_generator()
        program_published = _program_identity()
        require(
            program_published == program,
            "promotion program changed at publication",
        )
        result = _replay_bundle(
            final,
            generator_published,
            source_row,
            sam2_row,
        )
        _load_generator()
        program_return = _program_identity()
        require(
            program_return == program,
            "promotion program changed before successful return",
        )
        require(not os.path.lexists(stage), "promotion staging path reappeared")
        return result
    except BaseException as error:
        # Never remove a stage after an unexpected failure: another process or
        # a late fault may have changed its identity.  Its dot-prefixed name is
        # an explicit quarantine and the requested final path remains absent
        # unless the one atomic publication already succeeded.
        if not isinstance(error, Exception):
            raise
        if publication_state["committed"]:
            message = (
                "published promotion failed post-publication replay; "
                "treat bundle as quarantined at %s" % final
            )
            raise SupportPromotionHold(message) from error
        if isinstance(error, SupportPromotionHold):
            raise
        message = "support promotion failed; staging quarantined at %s" % stage
        raise SupportPromotionHold(message) from error


def _seal_review_once(
    *,
    packet_root: str | Path,
    reviewer_slot: int,
    completed_draft: str | Path,
    detached_evidence: str | Path,
    output_root: str | Path,
) -> Mapping[str, Any]:
    program = _program_identity()
    generator = _load_generator()
    final = _canonical_absolute(output_root, "review submission output")
    parent = _directory(final.parent, "review submission output parent")
    require(final.parent == parent, "review submission output parent differs")
    require(not os.path.lexists(final), "review submission output already exists")
    packet = replay_packet(packet_root)
    _outside(final, Path(packet["root"]), "review submission output")
    receipt, draft_row, evidence_row, overall = _review_draft_input(
        generator,
        packet,
        expected_slot=reviewer_slot,
        draft_path_value=completed_draft,
        evidence_path_value=detached_evidence,
    )
    stage = Path(tempfile.mkdtemp(prefix=".%s.partial." % final.name, dir=str(parent)))
    publication_state = {"committed": False}
    try:
        draft_stage = _copy_stage_file(
            Path(draft_row["path"]),
            stage / "completed_draft.input.json",
            "completed review draft",
            MAX_EXTERNAL_RECEIPT_BYTES,
        )
        receipt_stage = _write_stage_file(
            stage / "receipt.json",
            canonical_json_bytes(receipt) + b"\n",
        )
        evidence_stage = _copy_stage_file(
            Path(evidence_row["path"]),
            stage / "evidence.bin",
            "review evidence",
            MAX_EXTERNAL_EVIDENCE_BYTES,
        )
        receipt_stage_row = {
            "path": str(stage / "receipt.json"),
            "sha256": receipt_stage["sha256"],
            "size": receipt_stage["size"],
        }
        evidence_stage_row = {
            "path": str(stage / "evidence.bin"),
            "sha256": evidence_stage["sha256"],
            "size": evidence_stage["size"],
        }
        sealed_receipt, sealed_overall = _sealed_review_files(
            generator,
            packet,
            expected_slot=reviewer_slot,
            receipt_row=receipt_stage_row,
            evidence_row=evidence_stage_row,
        )
        require(
            sealed_receipt == receipt and sealed_overall == overall,
            "sealed review bytes differ from completed draft projection",
        )
        submission = _review_submission_document(
            program=program,
            packet=packet,
            receipt=receipt,
            overall=overall,
            completed_draft_row=draft_stage,
            receipt_row=receipt_stage_row,
            evidence_row=evidence_stage_row,
        )
        _write_stage_file(
            stage / "submission.json",
            canonical_json_bytes(submission) + b"\n",
        )
        _write_stage_file(stage / "COMPLETE", REVIEW_SUBMISSION_COMPLETE_BYTES)
        records = _review_submission_records(stage)
        sums_payload = "".join(
            "%s  %s\n" % (row["sha256"], row["path"]) for row in records
        ).encode("utf-8")
        _write_stage_file(stage / "SHA256SUMS", sums_payload)
        _fsync_tree(stage)

        program_after = _program_identity()
        generator_after = _load_generator()
        packet_after = replay_packet(packet_root)
        receipt_after, draft_after, evidence_after, overall_after = _review_draft_input(
            generator_after,
            packet_after,
            expected_slot=reviewer_slot,
            draft_path_value=completed_draft,
            evidence_path_value=detached_evidence,
        )
        require(program_after == program, "sealer program changed before publication")
        require(packet_after == packet, "packet changed before review sealing")
        require(
            receipt_after == receipt
            and draft_after == draft_row
            and evidence_after == evidence_row
            and overall_after == overall,
            "review draft/evidence changed before sealing",
        )
        _replay_review_submission(
            stage,
            generator_after,
            packet_after,
            expected_slot=reviewer_slot,
            expected_program=program,
        )
        _rename_noreplace(stage, final, publication_state)
        generator_published = _load_generator()
        program_published = _program_identity()
        require(
            program_published == program,
            "sealer program changed at publication",
        )
        result = _replay_review_submission(
            final,
            generator_published,
            packet_after,
            expected_slot=reviewer_slot,
            expected_program=program,
        )
        _load_generator()
        program_return = _program_identity()
        require(
            program_return == program,
            "sealer program changed before successful return",
        )
        require(not os.path.lexists(stage), "review submission staging path reappeared")
        return result
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        if publication_state["committed"]:
            message = (
                "published review submission failed post-publication replay; "
                "treat bundle as quarantined at %s" % final
            )
            raise SupportPromotionHold(message) from error
        if isinstance(error, SupportPromotionHold):
            raise
        message = "review sealing failed; staging quarantined at %s" % stage
        raise SupportPromotionHold(message) from error


def seal_review(
    *,
    packet_root: str | Path,
    reviewer_slot: int,
    completed_draft: str | Path,
    detached_evidence: str | Path,
    output_root: str | Path,
) -> Mapping[str, Any]:
    """Create one canonical PASS-or-FAIL review submission without reviewing it."""

    try:
        return _seal_review_once(
            packet_root=packet_root,
            reviewer_slot=reviewer_slot,
            completed_draft=completed_draft,
            detached_evidence=detached_evidence,
            output_root=output_root,
        )
    except SupportPromotionHold:
        raise
    except Exception as error:
        raise SupportPromotionHold("review sealing failed closed before publication") from error


def promote(
    *,
    packet_root: str | Path,
    reviewer_receipts: Sequence[str | Path],
    reviewer_evidence: Sequence[str | Path],
    output_root: str | Path,
    source_path: str | Path | None = None,
    sam2_receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Fail closed under this module's public exception at every boundary."""

    try:
        return _promote_once(
            packet_root=packet_root,
            reviewer_receipts=reviewer_receipts,
            reviewer_evidence=reviewer_evidence,
            output_root=output_root,
            source_path=source_path,
            sam2_receipt_path=sam2_receipt_path,
        )
    except SupportPromotionHold:
        raise
    except Exception as error:
        raise SupportPromotionHold("support promotion failed closed before publication") from error


def _preflight_summary(
    *,
    program: Mapping[str, Any],
    packet: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    sam2: Mapping[str, Any],
) -> Mapping[str, Any]:
    external_reviews = []
    for review in reviews:
        signature = review["signature"]
        external_reviews.append(
            {
                "reviewer_slot": review["reviewer_slot"],
                "declared_reviewer_identity": review["reviewer_identity"],
                "declared_reviewer_affiliation_or_role": review[
                    "reviewer_affiliation_or_role"
                ],
                "reviewed_at_utc": review["reviewed_at_utc"],
                "receipt": dict(review["receipt"]),
                "evidence": dict(review["evidence"]),
                "review_projection_sha256": signature[
                    "review_projection_sha256"
                ],
                "frame_count": FRAME_COUNT,
                "overall_decision": "PASS",
            }
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "case_id": CASE_ID,
        "iid": IID,
        "program": dict(program),
        "generator": _generator_identity(),
        "packet": {
            "manifest": dict(packet["manifest_row"]),
            "sha256sums": dict(packet["sha256sums_row"]),
            "premanifest_output_tree_digest": packet[
                "premanifest_output_tree_digest"
            ],
            "file_count": packet["file_count"],
            "support_frame_count": len(packet["support_rows"]),
        },
        "external_reviews": external_reviews,
        "source": dict(source),
        "sam2_receipt": dict(sam2),
        "validation_scope": {
            "full_packet_bytes_and_inventory_replayed": True,
            "two_raw_review_abis_validated": True,
            "pair_distinctness_validated": True,
            "source_and_sam2_exact_byte_authorities_bound": True,
            "formal_authority_created": False,
            "publication_attempted": False,
            "requires_fresh_validation_during_promotion": True,
        },
        "claim_limits": dict(PREFLIGHT_CLAIM_LIMITS),
    }


def _validate_only_once(
    *,
    packet_root: str | Path,
    reviewer_receipts: Sequence[str | Path],
    reviewer_evidence: Sequence[str | Path],
    source_path: str | Path | None = None,
    sam2_receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Replay all promotion inputs twice without creating filesystem output."""

    program = _program_identity()
    generator = _load_generator()
    packet = replay_packet(packet_root)
    reviews = _review_inputs(
        generator,
        packet,
        reviewer_receipts,
        reviewer_evidence,
    )
    source, sam2 = _deployment_authorities(
        generator,
        packet,
        source_path,
        sam2_receipt_path,
    )

    # This is deliberately a second complete read, not a reusable token.  It
    # narrows the validation-time drift window; promote still repeats its own
    # validation immediately before the sole create-only publication step.
    generator_after = _load_generator()
    packet_after = replay_packet(packet_root)
    reviews_after = _review_inputs(
        generator_after,
        packet_after,
        reviewer_receipts,
        reviewer_evidence,
    )
    source_after, sam2_after = _deployment_authorities(
        generator_after,
        packet_after,
        source_path,
        sam2_receipt_path,
    )
    program_after = _program_identity()
    require(program_after == program, "promotion program changed during preflight")
    require(packet_after == packet, "packet changed during preflight")
    require(reviews_after == reviews, "external review inputs changed during preflight")
    require(
        source_after == source and sam2_after == sam2,
        "deployment authority changed during preflight",
    )
    _load_generator()
    program_return = _program_identity()
    require(
        program_return == program,
        "promotion program changed before preflight return",
    )
    return _preflight_summary(
        program=program,
        packet=packet,
        reviews=reviews,
        source=source,
        sam2=sam2,
    )


def validate_only(
    *,
    packet_root: str | Path,
    reviewer_receipts: Sequence[str | Path],
    reviewer_evidence: Sequence[str | Path],
    source_path: str | Path | None = None,
    sam2_receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate point-in-time input bytes; never create or publish authority."""

    try:
        return _validate_only_once(
            packet_root=packet_root,
            reviewer_receipts=reviewer_receipts,
            reviewer_evidence=reviewer_evidence,
            source_path=source_path,
            sam2_receipt_path=sam2_receipt_path,
        )
    except SupportPromotionHold:
        raise
    except Exception as error:
        raise SupportPromotionHold("support preflight failed closed") from error


def _packet_preflight_summary(
    *,
    program: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema_version": PACKET_PREFLIGHT_SCHEMA,
        "status": PACKET_PREFLIGHT_STATUS,
        "case_id": CASE_ID,
        "iid": IID,
        "program": dict(program),
        "packet": {
            "root": packet["root"],
            "manifest": dict(packet["manifest_row"]),
            "sha256sums": dict(packet["sha256sums_row"]),
            "premanifest_output_tree_digest": packet[
                "premanifest_output_tree_digest"
            ],
            "file_count": packet["file_count"],
            "support_frame_count": len(packet["support_rows"]),
        },
        "validation_scope": {
            "full_packet_bytes_and_inventory_replayed_twice": True,
            "external_review_inputs_read": False,
            "filesystem_output_created": False,
            "requires_fresh_packet_replay_for_later_commands": True,
        },
        "claim_limits": dict(PACKET_PREFLIGHT_CLAIM_LIMITS),
    }


def _verify_packet_only_once(
    *,
    packet_root: str | Path,
) -> Mapping[str, Any]:
    """Replay the immutable packet twice without reading review inputs."""

    program = _program_identity()
    packet = replay_packet(packet_root)
    packet_after = replay_packet(packet_root)
    program_after = _program_identity()
    require(program_after == program, "promotion program changed during packet verification")
    require(packet_after == packet, "packet changed during packet verification")
    return _packet_preflight_summary(program=program, packet=packet)


def verify_packet_only(
    *,
    packet_root: str | Path,
) -> Mapping[str, Any]:
    """Validate point-in-time packet bytes; never create a review or authority."""

    try:
        return _verify_packet_only_once(packet_root=packet_root)
    except SupportPromotionHold:
        raise
    except Exception as error:
        raise SupportPromotionHold("packet verification failed closed") from error


def _verify_submission_only_once(
    *,
    packet_root: str | Path,
    reviewer_slot: int,
    submission_root: str | Path,
) -> Mapping[str, Any]:
    """Replay one complete PASS-or-FAIL sealed submission twice without writes."""

    require(
        type(reviewer_slot) is int and reviewer_slot in (1, 2),
        "reviewer slot differs",
    )
    program = _program_identity()
    generator = _load_generator()
    packet = replay_packet(packet_root)
    submission = _replay_review_submission(
        submission_root,
        generator,
        packet,
        expected_slot=reviewer_slot,
        expected_program=program,
    )

    generator_after = _load_generator()
    packet_after = replay_packet(packet_root)
    program_after = _program_identity()
    require(
        program_after == program,
        "promotion program changed during review submission verification",
    )
    require(
        packet_after == packet,
        "packet changed during review submission verification",
    )
    submission_after = _replay_review_submission(
        submission_root,
        generator_after,
        packet_after,
        expected_slot=reviewer_slot,
        expected_program=program,
    )
    require(
        submission_after == submission,
        "review submission changed during verification",
    )
    _load_generator()
    program_return = _program_identity()
    require(
        program_return == program,
        "promotion program changed before review submission verification return",
    )
    return submission


def verify_submission_only(
    *,
    packet_root: str | Path,
    reviewer_slot: int,
    submission_root: str | Path,
) -> Mapping[str, Any]:
    """Validate one complete sealed submission; never create pair authority."""

    try:
        return _verify_submission_only_once(
            packet_root=packet_root,
            reviewer_slot=reviewer_slot,
            submission_root=submission_root,
        )
    except SupportPromotionHold:
        raise
    except Exception as error:
        raise SupportPromotionHold(
            "review submission verification failed closed"
        ) from error


def _add_review_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--packet-root", required=True)
    command.add_argument("--reviewer-1-receipt", required=True)
    command.add_argument("--reviewer-1-evidence", required=True)
    command.add_argument("--reviewer-2-receipt", required=True)
    command.add_argument("--reviewer-2-evidence", required=True)
    command.add_argument("--source")
    command.add_argument("--sam2-receipt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("promote")
    _add_review_arguments(command)
    command.add_argument("--output-root", required=True)
    validate = subparsers.add_parser("validate-only")
    _add_review_arguments(validate)
    verify_packet = subparsers.add_parser("verify-packet")
    verify_packet.add_argument("--packet-root", required=True)
    verify_submission = subparsers.add_parser("verify-submission")
    verify_submission.add_argument("--packet-root", required=True)
    verify_submission.add_argument(
        "--reviewer-slot", required=True, type=int, choices=(1, 2)
    )
    verify_submission.add_argument("--submission-root", required=True)
    seal = subparsers.add_parser("seal-review")
    seal.add_argument("--packet-root", required=True)
    seal.add_argument("--reviewer-slot", required=True, type=int, choices=(1, 2))
    seal.add_argument("--completed-draft", required=True)
    seal.add_argument("--detached-evidence", required=True)
    seal.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.command == "verify-packet":
        result = verify_packet_only(packet_root=args.packet_root)
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    if args.command == "verify-submission":
        result = verify_submission_only(
            packet_root=args.packet_root,
            reviewer_slot=args.reviewer_slot,
            submission_root=args.submission_root,
        )
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    if args.command == "seal-review":
        result = seal_review(
            packet_root=args.packet_root,
            reviewer_slot=args.reviewer_slot,
            completed_draft=args.completed_draft,
            detached_evidence=args.detached_evidence,
            output_root=args.output_root,
        )
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    common = {
        "packet_root": args.packet_root,
        "reviewer_receipts": (args.reviewer_1_receipt, args.reviewer_2_receipt),
        "reviewer_evidence": (args.reviewer_1_evidence, args.reviewer_2_evidence),
        "source_path": args.source,
        "sam2_receipt_path": args.sam2_receipt,
    }
    if args.command == "validate-only":
        result = validate_only(**common)
    else:
        result = promote(output_root=args.output_root, **common)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupportPromotionHold as error:
        print("HOLD: %s" % error, file=sys.stderr)
        raise SystemExit(96)
