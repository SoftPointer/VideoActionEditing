#!/usr/bin/env python3
"""Generate a fresh case01 bone-removed-v2 candidate with VACE.

This is the generator half of a create-only producer.  It deliberately does
not publish a candidate or claim semantic object removal.  A controller must
reserve a fresh bundle, call :func:`preflight` before creating that bundle,
run this program, seal the receipts, and atomically publish the whole bundle.

The only pixels admitted from VACE are inside a separately reviewed
bone-plus-cast-shadow support.  Every canonical frame starts as the decoded
source RGB frame and receives a byte-for-byte overwrite from the inverse
mapped VACE donor only at support pixels.  There is no alpha blend, feather,
scan-line interpolation, removelogo operation, or whole-frame VACE authority.

No GPU is touched by ``preflight``.  ``run`` is fail-closed and requires three
canonical exact-tree manifests, a canonical support-review receipt, the exact
case source/SAM2 authority, and pinned media tools before it writes anything.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


CASE_ID = "case01"
IID = "288545b9c031491a"
WIDTH = 704
HEIGHT = 736
FPS = 25
FRAME_COUNT = 81
FRAME_PIXELS = WIDTH * HEIGHT
RGB_FRAME_BYTES = FRAME_PIXELS * 3

RAW_WIDTH = 624
RAW_HEIGHT = 640
FIT_WIDTH = 612
FIT_HEIGHT = 640
PAD_LEFT = 6
PAD_RIGHT = 6
PYTHON_HASH_SEED = 20260822

SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
SOURCE_SIZE = 10_887_043
SAM2_RECEIPT_SHA256 = "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
SAM2_RECEIPT_SIZE = 22_160

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
GENERATION_EVIDENCE_SCHEMA = "bernini-case01-bone-removed-v2-vace-generation-evidence-v1"
MODEL_AUTHORITY_ROLES = (
    "python_runtime_tree",
    "vace_checkpoint_tree",
    "vace_source_tree",
)
ALGORITHM_ID = "vace_1p3b_reviewed_support_hard_composite_fitpad_v1"
NORMALIZATION_ALGORITHM = "vace_precanvas_fitpad624x640_inverse_crop_lanczos_v1"
PROMPT = (
    "Remove the existing bone and its contact shadow. Fill only the masked area "
    "with temporally coherent concrete floor texture matching the surrounding "
    "surface. Do not add another bone or object. Preserve the dog, camera, "
    "lighting, background, and every unmasked region."
)

REQUIRED_VACE_FILES = {
    "vace/vace_wan_inference.py",
    "vace/models/wan/wan_vace.py",
    "vace/models/wan/configs/__init__.py",
    "vace/models/wan/configs/shared_config.py",
    "vace/models/wan/configs/wan_t2v_1_3B.py",
}
REQUIRED_CHECKPOINT_FILES = {
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
}

SHA256_CHARS = set("0123456789abcdef")


class ProducerHold(RuntimeError):
    """The candidate is not authorized to generate or publish."""


class HeldFile:
    """An authenticated regular file retained across every byte consumer.

    The logical path remains receipt-facing, while subprocesses consume the
    already-authenticated inode through ``/proc/self/fd``.  This closes the
    verify-by-name/use-by-name window for source, masks, and generated media.
    It is still not a substitute for a controller-sealed tree namespace for
    VACE's recursively opened source/checkpoint/runtime trees.
    """

    def __init__(
        self,
        row: Mapping[str, Any],
        label: str,
        *,
        require_nlink1: bool = False,
    ) -> None:
        _exact_keys(row, ("path", "sha256", "size"), label)
        require(_is_sha256(row["sha256"]), "%s SHA-256 differs" % label)
        require(type(row["size"]) is int and row["size"] > 0, "%s size differs" % label)
        self.logical_path = canonical_absolute(row["path"], label)
        self.expected_sha256 = row["sha256"]
        self.expected_size = row["size"]
        self.label = label
        self.require_nlink1 = require_nlink1
        self.fd = -1
        self.identity: tuple[int, ...] | None = None

    def __enter__(self) -> "HeldFile":
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.fd = os.open(self.logical_path, flags)
            observed = os.fstat(self.fd)
            require(stat.S_ISREG(observed.st_mode), "%s is not regular" % self.label)
            require(
                not self.require_nlink1 or observed.st_nlink == 1,
                "%s is not nlink1" % self.label,
            )
            self.identity = _stat_identity(observed)
            digest, size = self._digest()
            require(
                (digest, size) == (self.expected_sha256, self.expected_size),
                "%s held bytes differ" % self.label,
            )
            named = self.logical_path.lstat()
            require(
                _stat_identity(named) == self.identity,
                "%s named identity changed before hold" % self.label,
            )
            return self
        except Exception:
            self.close()
            raise

    def _digest(self) -> tuple[str, int]:
        require(self.fd >= 0, "%s is not held" % self.label)
        digest = hashlib.sha256()
        size = 0
        offset = 0
        while True:
            block = os.pread(self.fd, 1024 * 1024, offset)
            if not block:
                break
            digest.update(block)
            size += len(block)
            offset += len(block)
        return digest.hexdigest(), size

    @property
    def fd_path(self) -> Path:
        require(self.fd >= 0, "%s is not held" % self.label)
        linux = Path("/proc/self/fd/%d" % self.fd)
        if linux.parent.is_dir():
            return linux
        portable = Path("/dev/fd/%d" % self.fd)
        require(portable.parent.is_dir(), "retained-FD media transport is unavailable")
        return portable

    def read_bytes(self) -> bytes:
        require(self.fd >= 0, "%s is not held" % self.label)
        chunks: list[bytes] = []
        offset = 0
        while offset < self.expected_size:
            block = os.pread(self.fd, min(1024 * 1024, self.expected_size - offset), offset)
            require(block, "%s held file ended early" % self.label)
            chunks.append(block)
            offset += len(block)
        return b"".join(chunks)

    def verify_unchanged(self, *, require_named_identity: bool = True) -> None:
        require(self.fd >= 0 and self.identity is not None, "%s is not held" % self.label)
        observed = os.fstat(self.fd)
        # A hostile rename legitimately changes inode ctime while the retained
        # descriptor still names the same immutable bytes.  Bind dev/inode,
        # mode/nlink, size, and mtime here; compare full identity to the named
        # path separately when the caller requires namespace continuity.
        require(
            _stat_identity(observed)[:6] == self.identity[:6],
            "%s held identity changed" % self.label,
        )
        digest, size = self._digest()
        require(
            (digest, size) == (self.expected_sha256, self.expected_size),
            "%s held bytes changed" % self.label,
        )
        if require_named_identity:
            named = self.logical_path.lstat()
            require(
                _stat_identity(named) == self.identity,
                "%s named identity changed while held" % self.label,
            )

    def evidence_identity(self) -> dict[str, int]:
        require(self.identity is not None, "%s is not held" % self.label)
        dev, ino, mode, nlink, size, mtime_ns, ctime_ns = self.identity
        return {
            "device": dev,
            "inode": ino,
            "mode": mode,
            "nlink": nlink,
            "size": size,
            "mtime_ns": mtime_ns,
            "ctime_ns": ctime_ns,
        }

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProducerHold(message)


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
        raise ProducerHold("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_CHARS


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    observed = set(value)
    require(
        observed == expected,
        "%s key closure differs: missing=%s extra=%s"
        % (label, sorted(expected - observed), sorted(observed - expected)),
    )


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


def canonical_absolute(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    require(path.is_absolute(), "%s is not absolute" % label)
    require(os.path.normpath(str(path)) == str(path), "%s is not canonical" % label)
    return path


def stable_file(path_value: str | Path, *, nlink1: bool = False) -> tuple[str, int, tuple[int, ...]]:
    path = canonical_absolute(path_value, "file")
    require(not path.is_symlink(), "file is a symlink: %s" % path)
    require(path.resolve(strict=True) == path, "file traverses a symlink: %s" % path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "not a regular file: %s" % path)
        require(not nlink1 or before.st_nlink == 1, "file is not nlink1: %s" % path)
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
            _stat_identity(before) == _stat_identity(after) == _stat_identity(named),
            "file changed while hashing: %s" % path,
        )
        return digest.hexdigest(), size, _stat_identity(after)
    finally:
        os.close(descriptor)


def file_row(path_value: str | Path, *, nlink1: bool = False) -> dict[str, Any]:
    path = canonical_absolute(path_value, "file row")
    digest, size, _ = stable_file(path, nlink1=nlink1)
    return {"path": str(path), "sha256": digest, "size": size}


def verify_file_row(value: Mapping[str, Any], label: str, *, nlink1: bool = False) -> Path:
    _exact_keys(value, ("path", "sha256", "size"), label)
    require(_is_sha256(value["sha256"]), "%s SHA-256 differs" % label)
    require(type(value["size"]) is int and value["size"] > 0, "%s size differs" % label)
    path = canonical_absolute(value["path"], label)
    digest, size, _ = stable_file(path, nlink1=nlink1)
    require((digest, size) == (value["sha256"], value["size"]), "%s file differs" % label)
    return path


def load_canonical_json(path_value: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    path = canonical_absolute(path_value, label)
    payload = path.read_bytes()

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key in %s: %s" % (label, key))
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProducerHold("non-finite JSON constant in %s: %s" % (label, value))

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProducerHold("invalid JSON: %s" % label) from error
    require(isinstance(value, dict), "%s JSON root is not an object" % label)
    require(payload == canonical_json_bytes(value) + b"\n", "%s is not canonical one-LF JSON" % label)
    stable_file(path)
    return path, value


def _validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    require(_is_sha256(value.get(field)), "%s digest differs" % label)
    payload = dict(value)
    observed = payload.pop(field)
    require(observed == object_sha256(payload), "%s digest mismatch" % label)


def _relative_path(value: Any, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value and "\\" not in value, "%s differs" % label)
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(part not in ("", ".", "..") for part in relative.parts),
        "%s is not canonical relative POSIX" % label,
    )
    return relative


def replay_tree_manifest(path_value: str | Path, expected_role: str) -> dict[str, Any]:
    manifest_path, manifest = load_canonical_json(path_value, "tree manifest %s" % expected_role)
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
        "tree manifest %s" % expected_role,
    )
    require(manifest["schema_version"] == TREE_MANIFEST_SCHEMA, "tree manifest schema differs")
    require(manifest["authority_role"] == expected_role, "tree manifest role differs")
    require(expected_role in MODEL_AUTHORITY_ROLES, "unknown tree manifest role")
    require(
        manifest["inventory_policy"] == "exact_recursive_regular_nonsymlink_nlink1",
        "tree inventory policy differs",
    )
    root = canonical_absolute(manifest["tree_root"], "tree root")
    require(not root.is_symlink() and root.resolve(strict=True) == root and root.is_dir(), "tree root differs")
    try:
        manifest_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ProducerHold("tree manifest must be outside its tree")

    entries = manifest["entries"]
    require(isinstance(entries, list) and entries, "tree entries missing")
    rows: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for index, row in enumerate(entries):
        require(isinstance(row, dict), "tree entry is not an object")
        _exact_keys(row, ("relative_path", "sha256", "size"), "tree entry")
        relative = _relative_path(row["relative_path"], "tree relative path")
        text = relative.as_posix()
        require(text not in rows, "tree relative path repeats")
        require(_is_sha256(row["sha256"]), "tree entry SHA differs")
        require(type(row["size"]) is int and row["size"] >= 0, "tree entry size differs")
        rows[text] = row
        order.append(text)
    require(order == sorted(order), "tree entries are not sorted")
    require(manifest["file_count"] == len(entries), "tree file count differs")
    require(manifest["total_bytes"] == sum(row["size"] for row in entries), "tree bytes differ")
    require(manifest["tree_digest"] == object_sha256(entries), "tree digest differs")
    _validate_self_digest(manifest, "manifest_digest", "tree manifest")

    observed: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        for dirname in list(dirnames):
            child = directory_path / dirname
            row = child.lstat()
            require(stat.S_ISDIR(row.st_mode) and not child.is_symlink(), "tree contains non-directory/symlink")
        for filename in filenames:
            child = directory_path / filename
            relative = child.relative_to(root).as_posix()
            require(relative in rows, "tree contains extra file: %s" % relative)
            digest, size, _ = stable_file(child, nlink1=True)
            expected = rows[relative]
            require((digest, size) == (expected["sha256"], expected["size"]), "tree file differs: %s" % relative)
            observed.add(relative)
    require(observed == set(rows), "tree inventory differs")
    return {
        "role": expected_role,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_row(manifest_path)["sha256"],
        "manifest_size": file_row(manifest_path)["size"],
        "tree_root": str(root),
        "tree_digest": manifest["tree_digest"],
        "entries": rows,
    }


def replay_digest(authorities: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "role": row["role"],
            "manifest_sha256": row["manifest_sha256"],
            "tree_digest": row["tree_digest"],
        }
        for row in authorities
    ]
    rows.sort(key=lambda row: row["role"])
    return object_sha256(rows)


def _manifest_file_set(authority: Mapping[str, Any]) -> set[str]:
    return set(authority["entries"])


def _same_file_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("sha256") == right.get("sha256")
        and left.get("size") == right.get("size")
    )


def _validate_packet_manifest(
    candidate_packet: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    require(type(candidate_packet) is dict, "candidate packet differs")
    _exact_keys(
        candidate_packet,
        ("manifest", "sha256sums", "premanifest_output_tree_digest"),
        "candidate packet",
    )
    manifest_row = candidate_packet["manifest"]
    manifest_path = verify_file_row(
        manifest_row, "candidate packet manifest", nlink1=True
    )
    require(
        (manifest_row["sha256"], manifest_row["size"])
        == (SUPPORT_PACKET_MANIFEST_SHA256, SUPPORT_PACKET_MANIFEST_SIZE),
        "candidate packet manifest authority differs",
    )
    _, manifest = load_canonical_json(manifest_path, "candidate packet manifest")
    verify_file_row(manifest_row, "candidate packet manifest", nlink1=True)
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
    require(
        manifest["schema_version"] == SUPPORT_PACKET_SCHEMA
        and manifest["status"] == SUPPORT_PACKET_STATUS
        and (manifest["case_id"], manifest["iid"]) == (CASE_ID, IID),
        "candidate packet manifest identity differs",
    )
    require(
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
    require(
        candidate_packet["premanifest_output_tree_digest"]
        == manifest["premanifest_output_tree_digest"]
        == SUPPORT_PACKET_PREMANIFEST_DIGEST,
        "candidate packet premanifest digest differs",
    )
    records = manifest["premanifest_output_tree"]
    require(type(records) is list and bool(records), "candidate packet tree is empty")
    paths: set[str] = set()
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        require(type(record) is dict, "candidate packet tree row differs")
        _exact_keys(record, ("path", "sha256", "size"), "candidate packet tree row")
        relative = _relative_path(record["path"], "candidate packet tree path")
        require(relative.as_posix() not in paths, "candidate packet tree path repeats")
        require(_is_sha256(record["sha256"]), "candidate packet tree SHA-256 differs")
        require(type(record["size"]) is int and record["size"] > 0, "candidate packet tree size differs")
        paths.add(relative.as_posix())
        normalized_records.append(dict(record))
        require(index == len(normalized_records) - 1, "candidate packet tree order differs")
    require(
        [row["path"] for row in normalized_records]
        == sorted(row["path"] for row in normalized_records),
        "candidate packet tree is not canonical path order",
    )
    require(
        hashlib.sha256(canonical_json_bytes(normalized_records) + b"\n").hexdigest()
        == manifest["premanifest_output_tree_digest"],
        "candidate packet tree digest differs",
    )
    records_by_path = {row["path"]: row for row in normalized_records}

    sums_row = candidate_packet["sha256sums"]
    sums_path = verify_file_row(sums_row, "candidate packet SHA256SUMS", nlink1=True)
    expected_sums = {
        row["path"]: row["sha256"] for row in normalized_records
    }
    expected_sums["manifest.json"] = manifest_row["sha256"]
    expected_payload = "".join(
        "%s  %s\n" % (expected_sums[name], name)
        for name in sorted(expected_sums)
    ).encode("utf-8")
    require(
        sums_path.read_bytes() == expected_payload,
        "candidate packet SHA256SUMS inventory differs",
    )
    verify_file_row(sums_row, "candidate packet SHA256SUMS", nlink1=True)

    authority = manifest["authority"]
    require(type(authority) is dict, "candidate packet authority differs")
    for name in ("source_video", "masklet_receipt"):
        require(type(authority.get(name)) is dict, "candidate packet %s differs" % name)
        _exact_keys(authority[name], ("path", "sha256", "size"), "candidate packet %s" % name)

    frames = manifest["frames"]
    require(type(frames) is list and len(frames) == FRAME_COUNT, "candidate packet frame count differs")
    expected_support: list[Mapping[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        require(
            type(frame) is dict
            and type(frame.get("frame_index")) is int
            and frame.get("frame_index") == frame_index,
            "candidate packet frame order differs",
        )
        outputs = frame.get("outputs")
        require(type(outputs) is dict, "candidate packet frame outputs differ")
        support = outputs.get("candidate_support")
        require(type(support) is dict, "candidate packet support row differs")
        _exact_keys(support, ("path", "sha256", "size"), "candidate packet support row")
        require(
            support["path"] == "masks/candidate_support/%05d.png" % frame_index
            and _is_sha256(support["sha256"])
            and type(support["size"]) is int
            and support["size"] > 0,
            "candidate packet support binding differs",
        )
        require(
            records_by_path.get(support["path"]) == support,
            "candidate packet support is not bound into exact inventory",
        )
        expected_support.append(support)
    return manifest, expected_support


def _validate_external_review(
    formal: Any,
    *,
    expected_slot: int,
    manifest_sha256: str,
) -> dict[str, Any]:
    require(type(formal) is dict, "external review differs")
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
    require(
        type(formal["reviewer_slot"]) is int
        and formal["reviewer_slot"] == expected_slot,
        "external reviewer slot differs",
    )
    receipt_path = verify_file_row(
        formal["receipt"], "external review receipt", nlink1=True
    )
    _, receipt = load_canonical_json(receipt_path, "external review receipt")
    verify_file_row(formal["receipt"], "external review receipt", nlink1=True)
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
    require(
        receipt["schema_version"] == EXTERNAL_REVIEW_SCHEMA
        and type(receipt["reviewer_slot"]) is int
        and receipt["reviewer_slot"] == expected_slot
        and receipt["candidate_manifest_sha256"] == manifest_sha256,
        "external review receipt identity/manifest differs",
    )
    for name in ("reviewer_identity", "reviewer_affiliation_or_role"):
        require(
            type(receipt[name]) is str and receipt[name] and receipt[name] == receipt[name].strip(),
            "external review %s differs" % name,
        )
        require(receipt[name] == formal[name], "formal external review %s differs" % name)
    reviewed_at = receipt["reviewed_at_utc"]
    require(
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
        raise ProducerHold("external review timestamp differs") from error
    require(
        parsed_reviewed_at.strftime("%Y-%m-%dT%H:%M:%SZ") == reviewed_at,
        "external review timestamp differs",
    )
    independence = receipt["independence_attestation"]
    require(type(independence) is dict, "external review independence differs")
    _exact_keys(independence, EXTERNAL_INDEPENDENCE_KEYS, "external review independence")
    require(
        all(independence[name] is True for name in EXTERNAL_INDEPENDENCE_KEYS)
        and independence == formal["independence_attestation"],
        "external review independence declaration differs",
    )
    require(
        receipt["all_81_native_frames_reviewed"] is True
        and receipt["claim_limits_acknowledged"] is True
        and receipt["overall_decision"] == "PASS"
        and receipt["instructions"] == list(EXTERNAL_REVIEW_INSTRUCTIONS),
        "external review all-frame decision/limits differs",
    )
    frames = receipt["frames"]
    require(type(frames) is list and len(frames) == FRAME_COUNT, "external review frame count differs")
    for frame_index, frame in enumerate(frames):
        require(type(frame) is dict, "external review frame differs")
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
        require(
            type(frame["frame_index"]) is int
            and frame["frame_index"] == frame_index,
            "external review frame order differs",
        )
        require(
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
            "external review frame ballot differs: %d" % frame_index,
        )

    signature = receipt["signature_or_external_receipt"]
    require(type(signature) is dict, "external review signature differs")
    _exact_keys(
        signature,
        ("kind", "review_projection_sha256", "evidence_sha256", "evidence_size"),
        "external review signature",
    )
    require(
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
    require(
        object_sha256(projection) == signature["review_projection_sha256"],
        "external review projection binding differs",
    )
    evidence_path = verify_file_row(
        formal["evidence"], "external review evidence", nlink1=True
    )
    require(
        formal["evidence"]["sha256"] == signature["evidence_sha256"]
        and formal["evidence"]["size"] == signature["evidence_size"],
        "external review evidence binding differs",
    )
    verify_file_row(formal["evidence"], "external review evidence", nlink1=True)
    return {
        "identity": receipt["reviewer_identity"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": formal["receipt"]["sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": formal["evidence"]["sha256"],
    }


def validate_support_review(
    review_path: str | Path,
    *,
    source_row: Mapping[str, Any],
    sam2_row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path, review = load_canonical_json(review_path, "support review")
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
        "support review",
    )
    require(review["schema_version"] == SUPPORT_REVIEW_SCHEMA, "support review schema differs")
    require(review["status"] == SUPPORT_REVIEW_STATUS, "support review did not pass")
    require((review["case_id"], review["iid"]) == (CASE_ID, IID), "support review case differs")
    require(review["source"] == source_row, "support review source binding differs")
    require(review["sam2_receipt"] == sam2_row, "support review SAM2 binding differs")

    packet_manifest, expected_support = _validate_packet_manifest(
        review["candidate_packet"]
    )
    authority = packet_manifest["authority"]
    require(
        _same_file_bytes(review["source"], authority["source_video"]),
        "support review source bytes differ from candidate packet",
    )
    require(
        _same_file_bytes(review["sam2_receipt"], authority["masklet_receipt"]),
        "support review SAM2 bytes differ from candidate packet",
    )

    external_reviews = review["external_reviews"]
    require(
        type(external_reviews) is list and len(external_reviews) == 2,
        "exactly two external reviewers are required",
    )
    external_rows = [
        _validate_external_review(
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
        require(
            external_rows[0][field] != external_rows[1][field],
            "external reviewer %s repeats" % field,
        )

    protocol = review["protocol"]
    require(type(protocol) is dict, "support protocol differs")
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
        "support protocol",
    )
    require(protocol["native_resolution_704x736"] is True, "support review resolution differs")
    require(protocol["all_81_frames_reviewed_by_each"] is True, "support review is not all-frame per reviewer")
    require(type(protocol["required_external_reviewers"]) is int and protocol["required_external_reviewers"] == 2, "support reviewer count differs")
    require(protocol["bone_covered_all_frames_by_each"] is True, "support does not cover bone for both reviewers")
    require(protocol["cast_shadow_and_halo_covered_all_frames_by_each"] is True, "support misses shadow/halo for a reviewer")
    require(type(protocol["minimum_bone_dilation_pixels"]) is int and protocol["minimum_bone_dilation_pixels"] >= 8, "support dilation differs")
    require(protocol["old_dilate3_reused"] is False, "old support was reused")
    require(protocol["dog_guard_excluded_all_frames"] is True, "support intersects dog guard")

    claim_limits = review["claim_limits"]
    require(type(claim_limits) is dict, "support review claim limits differ")
    _exact_keys(claim_limits, SUPPORT_REVIEW_CLAIM_LIMITS, "support review claim limits")
    require(claim_limits == SUPPORT_REVIEW_CLAIM_LIMITS, "support review overclaims external facts")

    masks = review["frame_masks"]
    require(isinstance(masks, list) and len(masks) == FRAME_COUNT, "support frame-mask count differs")
    compact: list[dict[str, Any]] = []
    parents: set[Path] = set()
    observed_paths: set[str] = set()
    for index, (row, expected) in enumerate(zip(masks, expected_support)):
        require(isinstance(row, dict), "support frame-mask row differs")
        _exact_keys(
            row,
            (
                "frame_index", "path", "sha256", "size",
                "bone_and_cast_shadow_covered", "native_resolution_reviewed",
            ),
            "support frame-mask",
        )
        require(
            type(row["frame_index"]) is int and row["frame_index"] == index,
            "support frame-mask order differs",
        )
        require(row["bone_and_cast_shadow_covered"] is True and row["native_resolution_reviewed"] is True, "support frame ballot fails")
        require(
            row["sha256"] == expected["sha256"]
            and row["size"] == expected["size"],
            "support mask differs from candidate packet: %d" % index,
        )
        file_fields = {key: row[key] for key in ("path", "sha256", "size")}
        mask_path = verify_file_row(file_fields, "support frame mask", nlink1=True)
        require(mask_path.name == "%05d.png" % index, "support mask filename differs")
        require(str(mask_path) not in observed_paths, "support mask path repeats")
        observed_paths.add(str(mask_path))
        parents.add(mask_path.parent)
        compact.append({"frame_index": index, **file_fields})
    require(len(parents) == 1, "support masks do not share one directory")
    parent = next(iter(parents))
    actual = {entry.name for entry in parent.iterdir()}
    require(actual == {"%05d.png" % i for i in range(FRAME_COUNT)}, "support mask directory has extra/missing entries")
    _validate_self_digest(review, "review_digest", "support review")
    return file_row(path), compact


def _sam2_mask_rows(receipt_path: str | Path) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]]]:
    path, receipt = load_canonical_json(receipt_path, "SAM2 receipt")
    row = file_row(path)
    require((row["sha256"], row["size"]) == (SAM2_RECEIPT_SHA256, SAM2_RECEIPT_SIZE), "SAM2 receipt authority differs")
    require(receipt.get("schema_version") == "bernini-case01-oracle-sam2-masklets-receipt-v1", "SAM2 schema differs")
    require((receipt.get("case_id"), receipt.get("iid")) == (CASE_ID, IID), "SAM2 case differs")
    payload = dict(receipt)
    observed_digest = payload.pop("receipt_digest", None)
    require(observed_digest == hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest(), "SAM2 receipt digest differs")
    outputs = receipt.get("outputs")
    require(isinstance(outputs, list), "SAM2 outputs missing")
    rows = {entry.get("path"): entry for entry in outputs if isinstance(entry, dict)}
    required = {
        "masks/%s/%05d.png" % (name, index)
        for name in ("bone", "dog")
        for index in range(FRAME_COUNT)
    }
    require(required <= set(rows), "SAM2 receipt lacks masks")
    absolute_rows: dict[str, dict[str, Any]] = {}
    for relative in sorted(required):
        entry = rows[relative]
        _exact_keys(entry, ("path", "sha256", "size"), "SAM2 output")
        actual_path = (path.parent / relative).resolve(strict=True)
        expected = {"path": str(actual_path), "sha256": entry["sha256"], "size": entry["size"]}
        verify_file_row(expected, "SAM2 mask")
        absolute_rows[relative] = expected
    return row, path.parent, absolute_rows


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    input_bytes: bytes | None = None,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(command),
        env=None if env is None else dict(env),
        check=False,
        input=input_bytes,
        stdin=subprocess.DEVNULL if input_bytes is None else None,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=tuple(sorted(set(pass_fds))),
    )
    require(completed.returncode == 0, "command failed (%d): %s\n%s" % (completed.returncode, command, completed.stderr.decode("utf-8", errors="replace")))
    return completed


def _media_reference(value: Path | HeldFile) -> tuple[Path, tuple[int, ...]]:
    if isinstance(value, HeldFile):
        return value.fd_path, (value.fd,)
    return Path(value), ()


def probe_video(ffprobe: Path, path: Path | HeldFile) -> dict[str, Any]:
    media_path, pass_fds = _media_reference(path)
    result = _run(
        (
            str(ffprobe), "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
            "-of", "json", str(media_path),
        ),
        capture=True,
        pass_fds=pass_fds,
    )
    try:
        streams = json.loads(result.stdout)["streams"]
        require(isinstance(streams, list) and len(streams) == 1, "media stream closure differs")
        stream = streams[0]
        return {
            "codec_type": stream["codec_type"],
            "codec_name": stream["codec_name"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "pixel_format": stream["pix_fmt"],
            "average_frame_rate": stream["avg_frame_rate"],
            "frame_count": int(stream["nb_read_frames"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProducerHold("ffprobe schema differs: %s" % media_path) from error


def _decode_png_gray(ffmpeg: Path, path: Path | HeldFile) -> bytes:
    media_path, pass_fds = _media_reference(path)
    result = _run(
        (
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(media_path),
            "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ),
        capture=True,
        pass_fds=pass_fds,
    )
    require(len(result.stdout) == FRAME_PIXELS, "PNG geometry differs: %s" % media_path)
    require(set(result.stdout) <= {0, 255} and 255 in result.stdout, "PNG is not nonempty binary")
    return result.stdout


def _dilate(mask: set[int], passes: int) -> set[int]:
    current = set(mask)
    for _ in range(passes):
        expanded = set(current)
        for index in current:
            y, x = divmod(index, WIDTH)
            for yy in range(max(0, y - 1), min(HEIGHT, y + 2)):
                base = yy * WIDTH
                for xx in range(max(0, x - 1), min(WIDTH, x + 2)):
                    expanded.add(base + xx)
        current = expanded
    return current


def validate_support_geometry(
    ffmpeg: Path,
    support_rows: Sequence[Mapping[str, Any]],
    sam2_root: Path,
) -> None:
    support_frames = [_decode_png_gray(ffmpeg, Path(row["path"])) for row in support_rows]
    bone_frames = [
        _decode_png_gray(ffmpeg, sam2_root / "masks" / "bone" / ("%05d.png" % index))
        for index in range(FRAME_COUNT)
    ]
    dog_frames = [
        _decode_png_gray(ffmpeg, sam2_root / "masks" / "dog" / ("%05d.png" % index))
        for index in range(FRAME_COUNT)
    ]
    validate_support_geometry_frames(support_frames, bone_frames, dog_frames)


def validate_support_geometry_frames(
    support_frames: Sequence[bytes],
    bone_frames: Sequence[bytes],
    dog_frames: Sequence[bytes],
) -> None:
    require(
        len(support_frames) == len(bone_frames) == len(dog_frames) == FRAME_COUNT,
        "support geometry frame count differs",
    )
    for index, (support_bytes, bone_bytes, dog_bytes) in enumerate(
        zip(support_frames, bone_frames, dog_frames)
    ):
        require(
            len(support_bytes) == len(bone_bytes) == len(dog_bytes) == FRAME_PIXELS,
            "support geometry frame bytes differ",
        )
        support = {i for i, value in enumerate(support_bytes) if value}
        bone = {i for i, value in enumerate(bone_bytes) if value}
        dog = {i for i, value in enumerate(dog_bytes) if value}
        require(_dilate(bone, 8) <= support, "support misses dilate-8 bone in frame %d" % index)
        require(len(support) <= 4 * len(bone), "support is too broad in frame %d" % index)
        require(not (support & _dilate(dog, 8)), "support intersects dog guard in frame %d" % index)


def transform_contract(*, trace: Mapping[str, Any], media_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "algorithm": NORMALIZATION_ALGORITHM,
        "source_width": WIDTH,
        "source_height": HEIGHT,
        "precanvas_width": RAW_WIDTH,
        "precanvas_height": RAW_HEIGHT,
        "fit_width": FIT_WIDTH,
        "fit_height": FIT_HEIGHT,
        "pad_left": PAD_LEFT,
        "pad_right": PAD_RIGHT,
        "pad_top": 0,
        "pad_bottom": 0,
        "source_fit_kernel": "lanczos",
        "support_fit_kernel": "nearest",
        "pad_value": 0,
        "inverse_crop_xyxy": [6, 0, 618, 640],
        "inverse_resize_kernel": "lanczos",
        "python_hash_seed": PYTHON_HASH_SEED,
        "frame_indices": list(range(FRAME_COUNT)),
        "prepare_source_trace": dict(trace),
        "precanvas_authority_scope": "lossless_vace_input_authority",
        "processed_cache_authority_scope": "nonauthoritative_codec_diagnostic_only",
        "precanvas_source_video": dict(media_rows["precanvas_source_video"]),
        "precanvas_mask_video": dict(media_rows["precanvas_mask_video"]),
        "processed_source_video": dict(media_rows["processed_source_video"]),
        "processed_mask_video": dict(media_rows["processed_mask_video"]),
    }


def hard_composite_frame(source: bytes, donor: bytes, support: bytes) -> tuple[bytes, int, int]:
    """Return source with exact donor RGB copied only where support is nonzero."""

    require(len(source) == RGB_FRAME_BYTES, "source RGB frame length differs")
    require(len(donor) == RGB_FRAME_BYTES, "donor RGB frame length differs")
    require(len(support) == FRAME_PIXELS, "support frame length differs")
    require(set(support) <= {0, 255}, "support frame is not binary")
    result = bytearray(source)
    changed_outside = 0
    donor_mismatch = 0
    for index, active in enumerate(support):
        if active:
            offset = index * 3
            result[offset : offset + 3] = donor[offset : offset + 3]
    # These explicit full-frame scans make the receipt facts independently
    # checkable and protect future refactors of the loop above.
    for index, active in enumerate(support):
        offset = index * 3
        pixel = bytes(result[offset : offset + 3])
        if active:
            donor_mismatch += int(pixel != donor[offset : offset + 3])
        else:
            changed_outside += int(pixel != source[offset : offset + 3])
    return bytes(result), changed_outside, donor_mismatch


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Replay every immutable authority without creating files or using a GPU."""

    source = file_row(args.source_video)
    require((source["sha256"], source["size"]) == (SOURCE_SHA256, SOURCE_SIZE), "source authority differs")
    sam2, sam2_root, sam2_masks = _sam2_mask_rows(args.sam2_receipt)
    review, support_rows = validate_support_review(
        args.support_review_receipt,
        source_row=source,
        sam2_row=sam2,
    )
    ffmpeg = file_row(args.ffmpeg)
    ffprobe = file_row(args.ffprobe)
    acceptance = file_row(args.acceptance_contract)

    authority_inputs = {
        "python_runtime_tree": args.python_runtime_manifest,
        "vace_checkpoint_tree": args.vace_checkpoint_manifest,
        "vace_source_tree": args.vace_source_manifest,
    }
    authorities = [replay_tree_manifest(authority_inputs[role], role) for role in MODEL_AUTHORITY_ROLES]
    by_role = {row["role"]: row for row in authorities}
    require(REQUIRED_VACE_FILES <= _manifest_file_set(by_role["vace_source_tree"]), "VACE tree lacks required sources")
    require(REQUIRED_CHECKPOINT_FILES <= _manifest_file_set(by_role["vace_checkpoint_tree"]), "checkpoint tree lacks required files")
    python_bin = canonical_absolute(args.python_bin, "Python executable")
    require(bool(python_bin.stat().st_mode & 0o111), "Python is not executable")
    runtime_root = Path(by_role["python_runtime_tree"]["tree_root"])
    try:
        python_relative = python_bin.relative_to(runtime_root).as_posix()
    except ValueError as error:
        raise ProducerHold("Python executable is outside runtime tree") from error
    require(python_relative in _manifest_file_set(by_role["python_runtime_tree"]), "Python executable is not tree-bound")
    python_executable = file_row(python_bin, nlink1=True)
    require(Path(by_role["vace_checkpoint_tree"]["tree_root"]) == canonical_absolute(args.vace_checkpoint_root, "checkpoint root"), "checkpoint root differs from manifest")
    require(Path(by_role["vace_source_tree"]["tree_root"]) == canonical_absolute(args.vace_root, "VACE root"), "VACE root differs from manifest")

    validate_support_geometry(Path(ffmpeg["path"]), support_rows, sam2_root)
    model_rows = [
        {
            "role": row["role"],
            "path": row["manifest_path"],
            "sha256": row["manifest_sha256"],
            "size": row["manifest_size"],
        }
        for row in authorities
    ]
    return {
        "status": "PASS_AUTHORITY_PREFLIGHT_NO_OUTPUT_CREATED",
        "source": source,
        "sam2_receipt": sam2,
        "support_review_receipt": review,
        "support_frame_masks": support_rows,
        "sam2_frame_masks": {
            name: [
                {"frame_index": index, **sam2_masks["masks/%s/%05d.png" % (name, index)]}
                for index in range(FRAME_COUNT)
            ]
            for name in ("bone", "dog")
        },
        "media_tools": {"ffmpeg": ffmpeg, "ffprobe": ffprobe},
        "acceptance_contract": acceptance,
        "model_authorities": model_rows,
        "authority_replay_digest": replay_digest(authorities),
        "vace_required_files": sorted(REQUIRED_VACE_FILES),
        "checkpoint_required_files": sorted(REQUIRED_CHECKPOINT_FILES),
        "python_executable_relative_path": python_relative,
        "python_executable": python_executable,
        "generation_execution_lineage_verified": False,
        "hold_reason": "preflight alone is not external execution attestation or publication authority",
    }


def deterministic_environment(
    args: argparse.Namespace,
    runtime_root: Path,
    *,
    ffmpeg_entry: Path | None = None,
) -> dict[str, str]:
    """Construct the complete child environment; inherit no ambient knobs."""

    runtime_root = canonical_absolute(runtime_root, "runtime environment root")
    device = args.gpu_visible_device
    require(
        isinstance(device, str)
        and device
        and "," not in device
        and all(character.isalnum() or character in "-_:" for character in device),
        "exactly one explicit GPU-visible device is required",
    )
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(runtime_root),
        "TMPDIR": str(runtime_root),
        "XDG_CACHE_HOME": str(runtime_root / "xdg-cache"),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": str(PYTHON_HASH_SEED),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(Path(args.vace_root)),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HIPBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
        "IMAGEIO_FFMPEG_EXE": str(args.ffmpeg if ffmpeg_entry is None else ffmpeg_entry),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "CUDA_VISIBLE_DEVICES": device,
        "HIP_VISIBLE_DEVICES": device,
        "ROCR_VISIBLE_DEVICES": device,
    }


def vace_argv(
    args: argparse.Namespace,
    precanvas_source: Path,
    precanvas_mask: Path,
    raw_output: Path,
    *,
    python_entry: Path | None = None,
    generator_entry: Path | None = None,
) -> list[str]:
    return [
        str(args.python_bin if python_entry is None else python_entry),
        str(Path(__file__).resolve() if generator_entry is None else generator_entry),
        "_vace_child",
        "--vace-root", str(args.vace_root),
        "--checkpoint-root", str(args.vace_checkpoint_root),
        "--source", str(precanvas_source),
        "--mask", str(precanvas_mask),
        "--save-dir", str(raw_output.parent),
        "--save-file", str(raw_output),
        "--trace-out", str(raw_output.parent / "prepare_source_trace.json"),
        "--seed", str(args.seed),
    ]


def _write_mask_video(
    ffmpeg: Path,
    frames: Sequence[bytes],
    output: Path,
    *,
    filter_graph: str | None = None,
) -> None:
    require(len(frames) == FRAME_COUNT, "mask frame count differs")
    require(all(len(frame) == FRAME_PIXELS for frame in frames), "mask frame geometry differs")
    require(all(set(frame) <= {0, 255} for frame in frames), "mask frames are not binary")
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "gray", "-s:v", "704x736", "-r", str(FPS), "-i", "pipe:0",
        "-frames:v", str(FRAME_COUNT),
    ]
    if filter_graph is not None:
        command.extend(("-vf", filter_graph))
    command.extend(("-r", str(FPS), "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", str(output)))
    _run(command, input_bytes=b"".join(frames))


def _write_precanvas(
    ffmpeg: Path,
    source: Path | HeldFile,
    support_frames: Sequence[bytes],
    source_out: Path,
    mask_out: Path,
) -> None:
    source_path, source_fds = _media_reference(source)
    _run(
        (
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source_path),
            "-map", "0:v:0", "-frames:v", str(FRAME_COUNT),
            "-vf", "scale=612:640:flags=lanczos,pad=624:640:6:0:color=black,format=bgr0",
            "-r", str(FPS), "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
            str(source_out),
        ),
        pass_fds=source_fds,
    )
    _write_mask_video(
        ffmpeg,
        support_frames,
        mask_out,
        filter_graph="scale=612:640:flags=neighbor,pad=624:640:6:0:color=black,format=gray",
    )


def _decoder(
    ffmpeg: Path,
    video: Path | HeldFile,
    pixel_format: str,
    filter_graph: str | None = None,
) -> subprocess.Popen[bytes]:
    video_path, pass_fds = _media_reference(video)
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-map", "0:v:0"]
    if filter_graph:
        command.extend(("-vf", filter_graph))
    command.extend(("-frames:v", str(FRAME_COUNT), "-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"))
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=pass_fds,
    )


def _read_exact(stream: Any, count: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        require(block, "%s ended early" % label)
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _encode_canonical(
    ffmpeg: Path,
    source: Path | HeldFile,
    raw: Path | HeldFile,
    support_frames: Sequence[bytes],
    bone_frames: Sequence[bytes],
    output: Path,
) -> dict[str, Any]:
    require(len(support_frames) == len(bone_frames) == FRAME_COUNT, "composite masks differ")
    source_proc = _decoder(ffmpeg, source, "rgb24")
    donor_proc = _decoder(
        ffmpeg,
        raw,
        "rgb24",
        "crop=612:640:6:0,scale=704:736:flags=lanczos,format=rgb24",
    )
    encoder = subprocess.Popen(
        (
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s:v", "704x736", "-r", str(FPS), "-i", "pipe:0",
            "-frames:v", str(FRAME_COUNT), "-an", "-c:v", "ffv1", "-level", "3",
            "-pix_fmt", "bgr0", str(output),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    minimum_bone_fraction = 1.0
    try:
        require(source_proc.stdout is not None and donor_proc.stdout is not None and encoder.stdin is not None, "media pipe missing")
        for index, (support_frame, bone_frame) in enumerate(zip(support_frames, bone_frames)):
            source_frame = _read_exact(source_proc.stdout, RGB_FRAME_BYTES, "source")
            donor_frame = _read_exact(donor_proc.stdout, RGB_FRAME_BYTES, "donor")
            composite, outside, mismatch = hard_composite_frame(source_frame, donor_frame, support_frame)
            require(outside == 0 and mismatch == 0, "hard composite invariant differs")
            bone_pixels = [pixel for pixel, value in enumerate(bone_frame) if value]
            require(bone_pixels, "source bone mask is empty")
            changed = 0
            for pixel in bone_pixels:
                offset = pixel * 3
                changed += int(
                    composite[offset : offset + 3]
                    != source_frame[offset : offset + 3]
                )
            fraction = changed / len(bone_pixels)
            minimum_bone_fraction = min(minimum_bone_fraction, fraction)
            encoder.stdin.write(composite)
        encoder.stdin.close()
        encoder_rc = encoder.wait()
        source_rc = source_proc.wait()
        donor_rc = donor_proc.wait()
        require(encoder_rc == source_rc == donor_rc == 0, "media pipeline failed")
    finally:
        for process in (source_proc, donor_proc, encoder):
            if process.poll() is None:
                process.kill()
                process.wait()
    require(
        minimum_bone_fraction >= 0.98,
        "VACE donor retained too many source-bone pixels: %.9f" % minimum_bone_fraction,
    )
    return {
        "frame_count": FRAME_COUNT,
        "outside_support_changed_pixels": 0,
        "dog_guard_changed_pixels": 0,
        "support_pixels_not_equal_raw_donor": 0,
        "full_frame_pixel_scan": True,
        "source_bone_changed_fraction_minimum": minimum_bone_fraction,
    }


def run_generator(args: argparse.Namespace) -> Mapping[str, Any]:
    """Run VACE and materialize staging artifacts; never publish them."""

    before = preflight(args)
    asset_root = canonical_absolute(args.asset_staging_root, "asset staging root")
    evidence_root = canonical_absolute(args.evidence_staging_root, "evidence staging root")
    require(asset_root.is_dir() and evidence_root.is_dir(), "controller did not reserve staging roots")
    require(not any(asset_root.iterdir()) and not any(evidence_root.iterdir()), "staging root is not empty")
    for root, label in ((asset_root, "asset staging root"), (evidence_root, "evidence staging root")):
        info = root.stat()
        require(stat.S_IMODE(info.st_mode) == 0o700, "%s is not controller-private mode 0700" % label)

    def row3(value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value[key] for key in ("path", "sha256", "size")}

    def held_receipt_row(value: HeldFile, path: Path | None = None) -> dict[str, Any]:
        return {
            "path": str(value.logical_path if path is None else path),
            "sha256": value.expected_sha256,
            "size": value.expected_size,
        }

    with ExitStack() as stack:
        retained: list[HeldFile] = []

        def hold(row: Mapping[str, Any], label: str, *, nlink1: bool = False) -> HeldFile:
            value = stack.enter_context(HeldFile(row3(row), label, require_nlink1=nlink1))
            retained.append(value)
            return value

        program = hold(file_row(Path(__file__).resolve(), nlink1=True), "generator program", nlink1=True)
        python_bin = hold(before["python_executable"], "Python executable", nlink1=True)
        ffmpeg_program = hold(before["media_tools"]["ffmpeg"], "ffmpeg program")
        ffprobe_program = hold(before["media_tools"]["ffprobe"], "ffprobe program")
        source = hold(before["source"], "source video")
        support_files = [
            hold(row, "support PNG %05d" % index, nlink1=True)
            for index, row in enumerate(before["support_frame_masks"])
        ]
        bone_files = [
            hold(row, "SAM2 bone PNG %05d" % index, nlink1=True)
            for index, row in enumerate(before["sam2_frame_masks"]["bone"])
        ]
        dog_files = [
            hold(row, "SAM2 dog PNG %05d" % index, nlink1=True)
            for index, row in enumerate(before["sam2_frame_masks"]["dog"])
        ]
        ffmpeg = Path(before["media_tools"]["ffmpeg"]["path"])
        ffprobe = Path(before["media_tools"]["ffprobe"]["path"])
        support_frames = [_decode_png_gray(ffmpeg, value) for value in support_files]
        bone_frames = [_decode_png_gray(ffmpeg, value) for value in bone_files]
        dog_frames = [_decode_png_gray(ffmpeg, value) for value in dog_files]
        validate_support_geometry_frames(support_frames, bone_frames, dog_frames)

        precanvas_source_path = evidence_root / "precanvas_source_ffv1.mkv"
        precanvas_mask_path = evidence_root / "precanvas_support_ffv1.mkv"
        raw_dir = evidence_root / "vace_run"
        raw_dir.mkdir(mode=0o700)
        runtime_env_root = raw_dir / "runtime_env"
        runtime_env_root.mkdir(mode=0o700)
        raw_output_path = raw_dir / "out_video.mp4"
        support_tube_path = asset_root / "support_ffv1.mkv"
        canonical_path = asset_root / "bone_removed_v2_canonical_ffv1.mkv"
        delivery_path = asset_root / "bone_removed_v2_delivery_h264.mp4"

        _write_precanvas(
            ffmpeg,
            source,
            support_frames,
            precanvas_source_path,
            precanvas_mask_path,
        )
        _write_mask_video(ffmpeg, support_frames, support_tube_path)
        precanvas_source = hold(file_row(precanvas_source_path, nlink1=True), "precanvas source", nlink1=True)
        precanvas_mask = hold(file_row(precanvas_mask_path, nlink1=True), "precanvas mask", nlink1=True)
        support_tube = hold(file_row(support_tube_path, nlink1=True), "support tube", nlink1=True)
        source_probe = probe_video(ffprobe, precanvas_source)
        mask_probe = probe_video(ffprobe, precanvas_mask)
        require(
            (
                source_probe["codec_name"], source_probe["width"], source_probe["height"],
                source_probe["pixel_format"], source_probe["average_frame_rate"], source_probe["frame_count"],
            ) == ("ffv1", RAW_WIDTH, RAW_HEIGHT, "bgr0", "25/1", FRAME_COUNT),
            "precanvas source media contract differs",
        )
        require(
            (
                mask_probe["codec_name"], mask_probe["width"], mask_probe["height"],
                mask_probe["pixel_format"], mask_probe["average_frame_rate"], mask_probe["frame_count"],
            ) == ("ffv1", RAW_WIDTH, RAW_HEIGHT, "gray", "25/1", FRAME_COUNT),
            "precanvas mask media contract differs",
        )

        command = vace_argv(
            args,
            precanvas_source.fd_path,
            precanvas_mask.fd_path,
            raw_output_path,
            python_entry=python_bin.fd_path,
            generator_entry=program.fd_path,
        )
        env = deterministic_environment(
            args,
            runtime_env_root,
        )
        child_fds = (
            python_bin.fd,
            program.fd,
            precanvas_source.fd,
            precanvas_mask.fd,
        )
        _run(command, env=env, pass_fds=child_fds)
        require(raw_output_path.is_file(), "VACE did not produce raw output")
        raw_output = hold(file_row(raw_output_path, nlink1=True), "actual VACE raw output", nlink1=True)
        raw_probe = probe_video(ffprobe, raw_output)
        require(
            (
                raw_probe["codec_name"], raw_probe["width"], raw_probe["height"],
                raw_probe["pixel_format"], raw_probe["average_frame_rate"], raw_probe["frame_count"],
            ) == ("h264", RAW_WIDTH, RAW_HEIGHT, "yuv420p", "16/1", FRAME_COUNT),
            "actual VACE raw output media contract differs",
        )
        trace_path = raw_dir / "prepare_source_trace.json"
        _, trace = load_canonical_json(trace_path, "prepare_source trace")
        validate_prepare_source_trace(trace)

        media_paths = {
            "precanvas_source_video": precanvas_source_path,
            "precanvas_mask_video": precanvas_mask_path,
            "processed_source_video": raw_dir / "src_video.mp4",
            "processed_mask_video": raw_dir / "src_mask.mp4",
        }
        require(all(path.is_file() for path in media_paths.values()), "VACE processed-cache evidence missing")
        media_held = {
            "precanvas_source_video": precanvas_source,
            "precanvas_mask_video": precanvas_mask,
            "processed_source_video": hold(file_row(media_paths["processed_source_video"], nlink1=True), "VACE processed source", nlink1=True),
            "processed_mask_video": hold(file_row(media_paths["processed_mask_video"], nlink1=True), "VACE processed mask", nlink1=True),
        }
        for name, value in media_held.items():
            observed = probe_video(ffprobe, value)
            if name.startswith("processed_"):
                require(
                    (
                        observed["codec_name"], observed["width"], observed["height"],
                        observed["pixel_format"], observed["average_frame_rate"], observed["frame_count"],
                    ) == ("h264", RAW_WIDTH, RAW_HEIGHT, "yuv420p", "16/1", FRAME_COUNT),
                    "%s diagnostic-cache media contract differs" % name,
                )

        audit = _encode_canonical(
            ffmpeg,
            source,
            raw_output,
            support_frames,
            bone_frames,
            canonical_path,
        )
        canonical = hold(file_row(canonical_path, nlink1=True), "canonical candidate", nlink1=True)
        _run(
            (
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(canonical.fd_path),
                "-map", "0:v:0", "-frames:v", str(FRAME_COUNT), "-r", str(FPS), "-an",
                "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(delivery_path),
            ),
            pass_fds=(canonical.fd,),
        )
        delivery = hold(file_row(delivery_path, nlink1=True), "delivery candidate", nlink1=True)
        after = preflight(args)
        require(before["authority_replay_digest"] == after["authority_replay_digest"], "authority changed during generation")

        support_probe = probe_video(ffprobe, support_tube)
        canonical_probe = probe_video(ffprobe, canonical)
        delivery_probe = probe_video(ffprobe, delivery)
        require(
            (
                support_probe["codec_name"], support_probe["width"], support_probe["height"],
                support_probe["pixel_format"], support_probe["average_frame_rate"], support_probe["frame_count"],
            ) == ("ffv1", WIDTH, HEIGHT, "gray", "25/1", FRAME_COUNT),
            "support tube media contract differs",
        )
        require(
            (
                canonical_probe["codec_name"], canonical_probe["width"], canonical_probe["height"],
                canonical_probe["pixel_format"], canonical_probe["average_frame_rate"], canonical_probe["frame_count"],
            ) == ("ffv1", WIDTH, HEIGHT, "bgr0", "25/1", FRAME_COUNT),
            "canonical media contract differs",
        )
        require(
            (
                delivery_probe["codec_name"], delivery_probe["width"], delivery_probe["height"],
                delivery_probe["pixel_format"], delivery_probe["average_frame_rate"], delivery_probe["frame_count"],
            ) == ("h264", WIDTH, HEIGHT, "yuv420p", "25/1", FRAME_COUNT),
            "delivery media contract differs",
        )
        for value in retained:
            value.verify_unchanged()

        final_evidence_root = canonical_absolute(args.evidence_final_root, "final evidence root")
        media_rows = {
            name: held_receipt_row(value, final_evidence_root / media_paths[name].relative_to(evidence_root))
            for name, value in media_held.items()
        }
        raw_row = held_receipt_row(
            raw_output,
            final_evidence_root / raw_output_path.relative_to(evidence_root),
        )
        program_row = held_receipt_row(program)
        fragment = {
            "program": program_row,
            "model_authorities": before["model_authorities"],
            "authority_replay": {
                "before_generation_digest": before["authority_replay_digest"],
                "after_generation_digest": after["authority_replay_digest"],
                "unchanged": True,
            },
            "raw_support_donor": {
                "video": raw_row,
                "frame_count": FRAME_COUNT,
                "index_mapping": "exact_frame_index_0_through_80_ignore_container_fps_timestamps",
                "normalization": transform_contract(trace=trace, media_rows=media_rows),
                "used_only_inside_support": True,
                "source_or_identity_authority": False,
            },
            "algorithm_id": ALGORITHM_ID,
            "deterministic": True,
            "seed": args.seed,
            "generative_inpainting_inside_support": True,
            "whole_frame_generation": False,
            "outside_support_hard_composite_source_rgb": True,
            "uses_bidirectional_boundary_interpolation": False,
            "uses_ffmpeg_removelogo": False,
            "training_performed": False,
            "optimizer_updates": 0,
        }
        execution = {
            "schema_version": GENERATION_EVIDENCE_SCHEMA,
            "status": "COMPLETE_GENERATION_UNPUBLISHED_PENDING_CONTROLLER_ATTESTATION",
            "case_id": CASE_ID,
            "iid": IID,
            "exact_argv": command,
            "exact_environment": env,
            "generator_program": program_row,
            "generator_program_identity": program.evidence_identity(),
            "python_executable_identity": python_bin.evidence_identity(),
            "ffmpeg_executable_identity": ffmpeg_program.evidence_identity(),
            "ffprobe_executable_identity": ffprobe_program.evidence_identity(),
            "raw_output": raw_row,
            "raw_output_identity": raw_output.evidence_identity(),
            "authority_replay": fragment["authority_replay"],
            "prepare_source_trace": trace,
            "authenticated_inputs_consumed_via_retained_fds": True,
            "generation_execution_lineage_verified": False,
            "hold_reason": "controller/external immutable execution attestation and hard-pinned manifest paths are absent",
        }
        execution["attempt_binding_digest"] = object_sha256(
            {
                "exact_argv": command,
                "exact_environment": env,
                "generator_program": program_row,
                "raw_output": raw_row,
                "authority_replay": fragment["authority_replay"],
            }
        )
        execution["evidence_digest"] = object_sha256(execution)
        return {
            "generator_fragment": fragment,
            "support_review_receipt": before["support_review_receipt"],
            "support_frame_masks": before["support_frame_masks"],
            "support_tube_stage": held_receipt_row(support_tube),
            "canonical_stage": held_receipt_row(canonical),
            "delivery_stage": held_receipt_row(delivery),
            "delivery_contract": {
                "authority_scope": "human_playback_convenience_lossy_transport_only",
                "identity_authority": False,
                "canonical_is_identity_authority": True,
            },
            "construction_audit": audit,
            "claim_limits": {
                "input_asset_authority_only": True,
                "renderer_inference_performed": False,
                "renderer_result_claim_authorized": False,
                "scientific_claim_authorized": False,
                "semantic_absence_requires_human_review": True,
                "downstream_identity_sensitive_consumption_requires_canonical": True,
                "actual_downstream_consumer_verified": False,
                "generation_execution_lineage_verified": False,
            },
            "execution_evidence": execution,
        }


def validate_prepare_source_trace(trace: Mapping[str, Any]) -> None:
    _exact_keys(
        trace,
        ("frame_indices", "resize_crop_applied", "digest_definition", "source_tensor", "mask_tensor"),
        "prepare_source trace",
    )
    require(trace["frame_indices"] == list(range(FRAME_COUNT)), "VACE frame indices differ")
    require(trace["resize_crop_applied"] is False, "VACE unexpectedly resized/cropped precanvas")
    require(trace["digest_definition"] == "sha256(torch.float32 contiguous CPU little-endian C-order bytes)", "tensor digest definition differs")
    for name, shape in (("source_tensor", [3, 81, 640, 624]), ("mask_tensor", [1, 81, 640, 624])):
        value = trace[name]
        _exact_keys(value, ("shape", "dtype", "pre_generate_sha256", "post_generate_sha256", "unchanged"), name)
        require(value["shape"] == shape and value["dtype"] == "float32", "%s tensor contract differs" % name)
        require(_is_sha256(value["pre_generate_sha256"]), "%s digest differs" % name)
        require(value["pre_generate_sha256"] == value["post_generate_sha256"] and value["unchanged"] is True, "%s mutated" % name)


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    data = value.numpy().tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def run_traced_vace_entry(
    module: Any,
    call: Mapping[str, Any],
    *,
    tensor_digest: Any = _tensor_sha256,
) -> dict[str, Any]:
    """Patch and invoke the exact ``WanVace`` class imported by official main."""

    require(hasattr(module, "WanVace") and callable(module.main), "official VACE entry ABI differs")
    target = module.WanVace
    original_prepare = target.prepare_source
    original_generate = target.generate
    captured: dict[str, Any] = {"prepare_calls": 0, "generate_calls": 0}

    def wrapped_prepare(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        captured["prepare_calls"] += 1
        result = original_prepare(self, *call_args, **call_kwargs)
        source_tensor = result[0][0]
        mask_tensor = result[1][0]
        captured["source"] = source_tensor
        captured["mask"] = mask_tensor
        captured["source_pre"] = tensor_digest(source_tensor)
        captured["mask_pre"] = tensor_digest(mask_tensor)
        return result

    def wrapped_generate(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        captured["generate_calls"] += 1
        require("source" in captured and "mask" in captured, "VACE generate ran before traced prepare_source")
        result = original_generate(self, *call_args, **call_kwargs)
        captured["source_post"] = tensor_digest(captured["source"])
        captured["mask_post"] = tensor_digest(captured["mask"])
        return result

    target.prepare_source = wrapped_prepare
    target.generate = wrapped_generate
    try:
        module.main(dict(call))
    finally:
        target.prepare_source = original_prepare
        target.generate = original_generate
    require(
        captured["prepare_calls"] == 1
        and captured["generate_calls"] == 1
        and all(
            key in captured
            for key in ("source", "mask", "source_pre", "mask_pre", "source_post", "mask_post")
        ),
        "official VACE main did not traverse the traced WanVace instance exactly once",
    )
    return captured


def _vace_child(args: argparse.Namespace) -> int:
    """Run the frozen official entrypoint while tracing its exact inputs."""

    import torch

    require(sys.byteorder == "little", "tensor digest requires little-endian host")
    require(
        os.environ.get("PYTHONHASHSEED") == str(PYTHON_HASH_SEED),
        "VACE child PYTHONHASHSEED differs",
    )
    require(
        (os.environ.get("RANK"), os.environ.get("WORLD_SIZE"), os.environ.get("LOCAL_RANK"))
        == ("0", "1", "0"),
        "VACE child is not exact single-process rank zero",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "VACE child does not observe exactly one GPU",
    )
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    vace_root = canonical_absolute(args.vace_root, "VACE root")
    sys.path.insert(0, str(vace_root / "vace"))
    entry = vace_root / "vace" / "vace_wan_inference.py"
    spec = importlib.util.spec_from_file_location("frozen_vace_wan_inference", entry)
    require(spec is not None and spec.loader is not None, "cannot load VACE entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Replay the exact frozen processor before loading the model.  This is an
    # observed trace, not a calculation copied into the receipt.  The same
    # input paths and process-level hash seed are then passed to WanVace.
    from models.utils.preprocessor import VaceVideoProcessor
    import decord

    source_reader = decord.VideoReader(str(args.source))
    mask_reader = decord.VideoReader(str(args.mask))
    require(len(source_reader) == FRAME_COUNT and len(mask_reader) == FRAME_COUNT, "precanvas frame count differs")
    require(tuple(source_reader[0].shape[:2]) == (RAW_HEIGHT, RAW_WIDTH), "precanvas source geometry differs")
    require(tuple(mask_reader[0].shape[:2]) == (RAW_HEIGHT, RAW_WIDTH), "precanvas mask geometry differs")
    processor = VaceVideoProcessor(
        downsample=(4, 16, 16),
        min_area=480 * 832,
        max_area=480 * 832,
        min_fps=16,
        max_fps=16,
        zero_start=True,
        seq_len=32760,
        keep_last=True,
    )
    traced_source, traced_mask, frame_ids, traced_hw, _ = processor.load_video_pair(
        str(args.source),
        str(args.mask),
    )
    require(frame_ids == list(range(FRAME_COUNT)), "observed VACE frame IDs differ")
    require(tuple(traced_hw) == (RAW_HEIGHT, RAW_WIDTH), "observed VACE processor geometry differs")
    traced_mask = torch.clamp((traced_mask[:1, :, :, :] + 1) / 2, min=0, max=1)
    traced_source_sha = _tensor_sha256(traced_source)
    traced_mask_sha = _tensor_sha256(traced_mask)

    call = {
        "model_name": "vace-1.3B",
        "size": "480p",
        "frame_num": FRAME_COUNT,
        "ckpt_dir": str(args.checkpoint_root),
        "offload_model": True,
        "ulysses_size": 1,
        "ring_size": 1,
        "t5_fsdp": False,
        "t5_cpu": False,
        "dit_fsdp": False,
        "save_dir": str(args.save_dir),
        "save_file": str(args.save_file),
        "src_video": str(args.source),
        "src_mask": str(args.mask),
        "src_ref_images": None,
        "prompt": PROMPT,
        "use_prompt_extend": "plain",
        "base_seed": args.seed,
        "sample_solver": "unipc",
        "sample_steps": 50,
        "sample_shift": 16.0,
        "sample_guide_scale": 5.0,
    }
    captured = run_traced_vace_entry(module, call)
    source = captured["source"]
    mask = captured["mask"]
    require(_tensor_sha256(source) == traced_source_sha, "actual prepared source differs from traced processor")
    require(_tensor_sha256(mask) == traced_mask_sha, "actual prepared mask differs from traced processor")
    trace = {
        "frame_indices": frame_ids,
        "resize_crop_applied": False,
        "digest_definition": "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
        "source_tensor": {
            "shape": list(source.shape),
            "dtype": "float32",
            "pre_generate_sha256": captured["source_pre"],
            "post_generate_sha256": captured["source_post"],
            "unchanged": captured["source_pre"] == captured["source_post"],
        },
        "mask_tensor": {
            "shape": list(mask.shape),
            "dtype": "float32",
            "pre_generate_sha256": captured["mask_pre"],
            "post_generate_sha256": captured["mask_post"],
            "unchanged": captured["mask_pre"] == captured["mask_post"],
        },
    }
    validate_prepare_source_trace(trace)
    output = canonical_json_bytes(trace) + b"\n"
    descriptor = os.open(args.trace_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        written = os.write(descriptor, output)
        require(written == len(output), "short trace write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--sam2-receipt", type=Path, required=True)
    parser.add_argument("--support-review-receipt", type=Path, required=True)
    parser.add_argument("--vace-source-manifest", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--python-runtime-manifest", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--vace-root", type=Path, required=True)
    parser.add_argument("--vace-checkpoint-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--acceptance-contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026082201)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _common_arguments(preflight_parser)
    run_parser = subparsers.add_parser("run")
    _common_arguments(run_parser)
    run_parser.add_argument("--asset-staging-root", type=Path, required=True)
    run_parser.add_argument("--evidence-staging-root", type=Path, required=True)
    run_parser.add_argument("--asset-final-root", type=Path, required=True)
    run_parser.add_argument("--evidence-final-root", type=Path, required=True)
    run_parser.add_argument("--gpu-visible-device", required=True)

    child = subparsers.add_parser("_vace_child")
    child.add_argument("--vace-root", type=Path, required=True)
    child.add_argument("--checkpoint-root", type=Path, required=True)
    child.add_argument("--source", type=Path, required=True)
    child.add_argument("--mask", type=Path, required=True)
    child.add_argument("--save-dir", type=Path, required=True)
    child.add_argument("--save-file", type=Path, required=True)
    child.add_argument("--trace-out", type=Path, required=True)
    child.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args)
            sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
            return 0
        if args.command == "run":
            result = run_generator(args)
            sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
            return 0
        if args.command == "_vace_child":
            return _vace_child(args)
        raise ProducerHold("unknown command")
    except (OSError, ProducerHold, subprocess.SubprocessError) as error:
        print("HOLD: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
