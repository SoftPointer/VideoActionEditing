#!/usr/bin/env python3
"""Detached trust root for the preservation-v2 decoded-eval release.

This file is deliberately outside the decoded-eval release archive.  A trusted operator
invokes it through :data:`ROOT_CONTROLLER_BOOTSTRAP_SOURCE` while supplying a
literal, independently reviewed controller SHA and deployment-request SHA.
The bootstrap captures this controller from one held descriptor before any of
its code runs.  The controller then captures the pinned verified runtime in
the same way and publishes a create-only runtime authority.  After that
authority exists, a second create-only receipt binds the complete scientific
source/runtime spec by an independently supplied literal file SHA; this
two-stage publication avoids a hash cycle through the controller authority.

No command here submits a scheduler job, retries work, reads training loss, or
authorizes scientific promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import sys
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-request-v3"
RECEIPT_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-receipt-v3"
WORK_ROOT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-work-root-authority-v1"
)
SOURCE_SPEC_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-spec-authority-v2"
)
SOURCE_RUNTIME_SCHEMA = "bernini-action-preservation-source-runtime-spec-v2"
WORK_ROOT_BINDING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-inherited-work-root-v2"
)
WORK_ROOT_BINDING_ENV = "APV2_EVAL_WORK_ROOT_AUTHORITY"
COMPLETION_ANCHOR_CHANNEL_ENV = (
    "APV2_EVAL_COMPLETION_ANCHOR_CHANNEL"
)
COMPLETION_ANCHOR_SENT_ENV = (
    "APV2_EVAL_COMPLETION_ANCHOR_SENT_DIGEST"
)
COMPLETION_ANCHOR_CHANNEL_SCHEMA = (
    "bernini-action-preservation-holder-completion-anchor-channel-v1"
)
HOLDER_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-holder-completion-anchor-v1"
)
AGGREGATE_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-aggregate-completion-anchor-v1"
)
EXECUTOR_TARGET = "action_preservation_decoded_eval_executor_v2.py"
AGGREGATE_TARGET = "action_preservation_decoded_eval_aggregate_v2.py"
DYNAMIC_ANCHOR_TARGETS = frozenset({EXECUTOR_TARGET, AGGREGATE_TARGET})
HOLDER_JOB_IDS = ("136719", "136141", "136309", "136140")
HOLDER_COMPLETION_SUFFIX = ".holder-directory-completion.json"
EXECUTION_SHARD_DIRECTORY = "execution_shards"
SHARD_SUMMARY_FILENAME = "shard_summary.json"
MAX_COMPLETION_ANCHOR_PACKET_SIZE = 16384
VERIFIED_RUNTIME_RELATIVE_PATH = (
    "methods/bernini_action_editing/"
    "action_preservation_decoded_eval_verified_release_v1.py"
)
ROOT_PYTHON_PATH = Path("/usr/bin/python3.10")
ROOT_PYTHON_UID = 0
ROOT_PYTHON_GID = 0
ROOT_PYTHON_MODE = 0o755
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")


class DecodedEvalDeploymentControllerError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise DecodedEvalDeploymentControllerError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise DecodedEvalDeploymentControllerError(
            "controller JSON is not canonical finite UTF-8"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode,
        value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _identity_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "rdev": value.st_rdev,
        "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _immutable_directory_identity(value: os.stat_result) -> dict[str, int]:
    row = _identity_row(value)
    return {
        key: row[key]
        for key in ("device", "inode", "uid", "gid", "mode", "rdev")
    }


def _validate_identity_row(value: Any, *, label: str) -> dict[str, int]:
    fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(value[field]) is not int for field in fields)
    ):
        fail(f"{label} identity closure differs")
    return dict(value)


def _validate_immutable_directory_identity(
    value: Any, *, label: str
) -> dict[str, int]:
    fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(value[field]) is not int for field in fields)
        or not stat.S_ISDIR(value["mode"])
    ):
        fail(f"{label} immutable identity closure differs")
    return dict(value)


def validate_work_root_authority(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "path", "parent_path", "creation_identity",
        "immutable_identity", "parent_immutable_identity", "initial_entries",
        "retained_parent_fd_through_request_publication",
        "retained_root_fd_through_request_publication", "authority_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("work root authority field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    path = _absolute(row["path"], label="work root authority path")
    parent = _absolute(
        row["parent_path"], label="work root authority parent"
    )
    creation = _validate_identity_row(
        row["creation_identity"], label="work root creation"
    )
    immutable = _validate_immutable_directory_identity(
        row["immutable_identity"], label="work root"
    )
    parent_immutable = _validate_immutable_directory_identity(
        row["parent_immutable_identity"], label="work root parent"
    )
    if (
        row["schema_version"] != WORK_ROOT_AUTHORITY_SCHEMA
        or path.parent != parent
        or not stat.S_ISDIR(creation["mode"])
        or stat.S_IMODE(creation["mode"]) != 0o700
        or immutable
        != {key: creation[key] for key in immutable}
        or row["initial_entries"] != []
        or row["retained_parent_fd_through_request_publication"] is not True
        or row["retained_root_fd_through_request_publication"] is not True
        or not isinstance(claimed, str)
        or SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        fail("work root authority differs")
    row["immutable_identity"] = immutable
    row["parent_immutable_identity"] = parent_immutable
    return row


def _validate_work_root_capture(
    value: Any,
    *,
    authority: Mapping[str, Any],
    expected_entries: Sequence[str],
) -> dict[str, Any]:
    fields = {
        "path", "identity", "parent_identity", "entries",
        "retained_parent_fd", "retained_root_fd",
    }
    if type(value) is not dict or set(value) != fields:
        fail("work root capture field closure differs")
    row = dict(value)
    identity = _validate_identity_row(
        row["identity"], label="work root capture"
    )
    parent_identity = _validate_identity_row(
        row["parent_identity"], label="work root parent capture"
    )
    if (
        row["path"] != authority["path"]
        or row["entries"] != sorted(expected_entries)
        or len(row["entries"]) != len(set(expected_entries))
        or any(type(item) is not str for item in row["entries"])
        or row["retained_parent_fd"] is not True
        or row["retained_root_fd"] is not True
        or {key: identity[key] for key in authority["immutable_identity"]}
        != authority["immutable_identity"]
        or {
            key: parent_identity[key]
            for key in authority["parent_immutable_identity"]
        }
        != authority["parent_immutable_identity"]
    ):
        fail("work root capture authority differs")
    row["identity"] = identity
    row["parent_identity"] = parent_identity
    return row


class _HeldWorkRoot:
    def __init__(
        self,
        *,
        authority: Mapping[str, Any],
        parent_fd: int,
        root_fd: int,
        parent_anchor: os.stat_result,
    ) -> None:
        self.authority = dict(authority)
        self.parent_fd = parent_fd
        self.root_fd = root_fd
        self.parent_anchor = parent_anchor
        self.closed = False

    @classmethod
    def open(cls, authority: Mapping[str, Any]) -> "_HeldWorkRoot":
        parent_path = Path(authority["parent_path"])
        root_path = Path(authority["path"])
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            fail("safe held work root is unavailable")
        parent_fd = os.open(
            parent_path, flags | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        root_fd: int | None = None
        try:
            root_fd = os.open(
                root_path.name,
                flags | os.O_NOFOLLOW | os.O_DIRECTORY,
                dir_fd=parent_fd,
            )
            os.set_inheritable(parent_fd, False)
            os.set_inheritable(root_fd, False)
            parent_anchor = os.fstat(parent_fd)
            observed = cls(
                authority=authority,
                parent_fd=parent_fd,
                root_fd=root_fd,
                parent_anchor=parent_anchor,
            )
            observed.capture(expected_entries=None)
            return observed
        except Exception:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            raise

    def capture(
        self, *, expected_entries: set[str] | None
    ) -> dict[str, Any]:
        if self.closed:
            fail("held work root is closed")
        root_path = Path(self.authority["path"])
        parent_path = Path(self.authority["parent_path"])
        try:
            parent_before = os.fstat(self.parent_fd)
            root_before = os.fstat(self.root_fd)
            first = os.listdir(self.root_fd)
            root_middle = os.fstat(self.root_fd)
            second = os.listdir(self.root_fd)
            root_after = os.fstat(self.root_fd)
            named_root = os.stat(
                root_path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            parent_after = os.fstat(self.parent_fd)
            named_parent = parent_path.lstat()
        except OSError as error:
            raise DecodedEvalDeploymentControllerError(
                "held work root replay is unavailable"
            ) from error
        if (
            _identity(parent_before) != _identity(self.parent_anchor)
            or _identity(parent_before) != _identity(parent_after)
            or _identity(parent_before) != _identity(named_parent)
            or _identity(root_before) != _identity(root_middle)
            or _identity(root_before) != _identity(root_after)
            or _identity(root_before) != _identity(named_root)
            or _immutable_directory_identity(root_before)
            != self.authority["immutable_identity"]
            or _immutable_directory_identity(parent_before)
            != self.authority["parent_immutable_identity"]
            or sorted(first) != sorted(second)
            or (
                expected_entries is not None
                and (
                    sorted(first) != sorted(expected_entries)
                    or len(first) != len(expected_entries)
                )
            )
            or os.get_inheritable(self.parent_fd)
            or os.get_inheritable(self.root_fd)
        ):
            fail("held work root identity or entry closure differs")
        return {
            "path": str(root_path),
            "identity": _identity_row(root_after),
            "parent_identity": _identity_row(parent_after),
            "entries": sorted(first),
            "retained_parent_fd": True,
            "retained_root_fd": True,
        }

    def stable_member(
        self,
        name: str,
        *,
        expected_sha256: str,
        expected_mode: int,
        expected_entries: set[str] | None,
        label: str,
    ) -> tuple[bytes, dict[str, Any]]:
        if (
            type(name) is not str
            or name in ("", ".", "..")
            or os.path.sep in name
            or (os.path.altsep is not None and os.path.altsep in name)
        ):
            fail(f"{label} held member name differs")
        if type(expected_sha256) is not str or SHA256_RE.fullmatch(
            expected_sha256
        ) is None:
            fail(f"{label} expected SHA differs")
        self.capture(expected_entries=expected_entries)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if not hasattr(os, "O_NOFOLLOW"):
            fail(f"{label} no-follow replay is unavailable")
        try:
            descriptor = os.open(
                name, flags | os.O_NOFOLLOW, dir_fd=self.root_fd
            )
        except OSError as error:
            raise DecodedEvalDeploymentControllerError(
                f"cannot open held {label}"
            ) from error
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.root_fd, follow_symlinks=False
            )
        finally:
            os.close(descriptor)
        digest = hashlib.sha256(first).hexdigest()
        self.capture(expected_entries=expected_entries)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != second
            or digest != expected_sha256
            or os.get_inheritable(self.parent_fd)
            or os.get_inheritable(self.root_fd)
        ):
            fail(f"{label} held same-FD replay differs")
        root_path = Path(self.authority["path"])
        return first, {
            "path": str(root_path / name),
            "sha256": digest,
            "size": len(first),
            "mode": stat.S_IMODE(before.st_mode),
            "device": before.st_dev,
            "inode": before.st_ino,
            "uid": before.st_uid,
            "gid": before.st_gid,
            "nlink": before.st_nlink,
            "rdev": before.st_rdev,
            "blocks": getattr(before, "st_blocks", 0),
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }

    def publish_member(
        self,
        name: str,
        raw: bytes,
        *,
        expected_entries_before: set[str],
        expected_entries_after: set[str],
        label: str,
    ) -> dict[str, Any]:
        if (
            type(name) is not str
            or name in ("", ".", "..")
            or os.path.sep in name
            or (os.path.altsep is not None and os.path.altsep in name)
            or expected_entries_after
            != expected_entries_before | {name}
        ):
            fail(f"{label} held publication topology differs")
        self.capture(expected_entries=expected_entries_before)
        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
        )
        if not hasattr(os, "O_NOFOLLOW"):
            fail(f"{label} no-follow publication is unavailable")
        try:
            descriptor = os.open(
                name, flags | os.O_NOFOLLOW, 0o444, dir_fd=self.root_fd
            )
        except OSError as error:
            raise DecodedEvalDeploymentControllerError(
                f"cannot create held {label}"
            ) from error
        try:
            os.set_inheritable(descriptor, False)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    fail(f"{label} held write made no progress")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.fsync(self.root_fd)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.root_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o444
                or _identity(before) != _identity(middle)
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(named)
                or first != raw
                or second != raw
                or os.get_inheritable(descriptor)
            ):
                fail(f"{label} held same-FD publication differs")
        finally:
            os.close(descriptor)
        self.capture(expected_entries=expected_entries_after)
        return {
            "path": str(Path(self.authority["path"]) / name),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "mode": 0o444,
            "device": before.st_dev,
            "inode": before.st_ino,
            "uid": before.st_uid,
            "gid": before.st_gid,
            "nlink": before.st_nlink,
            "rdev": before.st_rdev,
            "blocks": getattr(before, "st_blocks", 0),
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }

    def inherited_binding(
        self,
        *,
        deployment_receipt: Mapping[str, str],
        source_spec_authority: Mapping[str, str],
        deployment_receipt_digest: str,
        source_spec_authority_digest: str,
        target: str,
        capture_receipt_path: Path,
    ) -> dict[str, Any]:
        current = self.capture(expected_entries=None)
        root_path = Path(self.authority["path"])
        file_pairs = {
            "deployment receipt": deployment_receipt,
            "source spec authority": source_spec_authority,
        }
        validated_pairs: dict[str, dict[str, str]] = {}
        for label, item in file_pairs.items():
            if (
                type(item) is not dict
                or set(item) != {"path", "sha256"}
                or type(item.get("path")) is not str
                or Path(item["path"]).parent != root_path
                or Path(item["path"]).name in ("", ".", "..")
                or type(item.get("sha256")) is not str
                or SHA256_RE.fullmatch(item["sha256"]) is None
            ):
                fail(f"inherited {label} file binding differs")
            validated_pairs[label] = dict(item)
        if (
            validated_pairs["deployment receipt"]["path"]
            == validated_pairs["source spec authority"]["path"]
        ):
            fail("inherited authority receipt paths collide")
        if (
            type(deployment_receipt_digest) is not str
            or SHA256_RE.fullmatch(deployment_receipt_digest) is None
            or type(source_spec_authority_digest) is not str
            or SHA256_RE.fullmatch(source_spec_authority_digest) is None
            or type(target) is not str
            or not target
        ):
            fail("inherited work root authority digest or target differs")
        if (
            not capture_receipt_path.is_absolute()
            or capture_receipt_path.parent != root_path
            or capture_receipt_path.name in ("", ".", "..")
            or capture_receipt_path.name in current["entries"]
        ):
            fail("runtime capture receipt must be a fresh direct held-root member")
        if (
            self.parent_fd == self.root_fd
            or self.parent_fd < 3
            or self.root_fd < 3
            or os.get_inheritable(self.parent_fd)
            or os.get_inheritable(self.root_fd)
        ):
            fail("inherited work root FD state differs")
        value: dict[str, Any] = {
            "schema_version": WORK_ROOT_BINDING_SCHEMA,
            "path": self.authority["path"],
            "parent_path": self.authority["parent_path"],
            "parent_fd": self.parent_fd,
            "root_fd": self.root_fd,
            "parent_identity": current["parent_identity"],
            "root_identity": current["identity"],
            "parent_immutable_identity": self.authority[
                "parent_immutable_identity"
            ],
            "root_immutable_identity": self.authority["immutable_identity"],
            "entries": current["entries"],
            "work_root_authority": self.authority,
            "deployment_receipt": validated_pairs["deployment receipt"],
            "source_spec_authority": validated_pairs[
                "source spec authority"
            ],
            "work_root_authority_digest": self.authority[
                "authority_digest"
            ],
            "deployment_receipt_digest": deployment_receipt_digest,
            "source_spec_authority_digest": source_spec_authority_digest,
            "target": target,
            "capture_receipt_path": str(capture_receipt_path),
            "exact_two_directory_fds": True,
            "fds_inheritable_only_across_verified_exec": True,
        }
        value["binding_digest"] = object_sha256(value)
        return value

    def close(self) -> None:
        if not self.closed:
            os.close(self.root_fd)
            os.close(self.parent_fd)
            self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        fail(f"{label} is not a path")
    path = Path(value)
    if (
        not path.is_absolute() or value == os.path.sep
        or os.path.normpath(value) != value
    ):
        fail(f"{label} must be a normalized absolute non-root path")
    return path


def stable_file(
    value: str | Path, *, label: str, expected_sha256: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    path = _absolute(str(value), label=label)
    if path.resolve(strict=True) != path or not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} canonical no-follow capture is unavailable")
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(
        expected_sha256
    ) is None:
        fail(f"{label} expected SHA differs")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(first).hexdigest()
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second or len(first) != before.st_size
        or digest != expected_sha256
        or (
            expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode
        )
    ):
        fail(f"{label} stable physical identity or bytes differ")
    return first, {
        "path": str(path), "sha256": digest, "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode), "device": before.st_dev,
        "inode": before.st_ino, "uid": before.st_uid, "gid": before.st_gid,
        "nlink": before.st_nlink, "rdev": before.st_rdev,
        "blocks": getattr(before, "st_blocks", 0),
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
    }


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                fail(f"{label} contains a duplicate key")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvalDeploymentControllerError(
            f"cannot decode {label}"
        ) from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical newline JSON")
    return value


def _validate_holder_completion_anchor(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "holder_job_id", "completion_path",
        "initial_inode_identity", "completion_sha256", "completion_size",
        "completion_mode", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    }
    identity_fields = {"device", "inode", "uid", "gid", "rdev"}
    if type(value) is not dict or set(value) != fields:
        fail("holder completion anchor field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest", None)
    holder = row.get("holder_job_id")
    identity = row.get("initial_inode_identity")
    path = (
        Path(row["completion_path"])
        if type(row.get("completion_path")) is str else Path()
    )
    expected_suffix = (
        f"{EXECUTION_SHARD_DIRECTORY}/{holder}{HOLDER_COMPLETION_SUFFIX}"
    )
    if (
        row.get("schema_version") != HOLDER_COMPLETION_ANCHOR_SCHEMA
        or holder not in HOLDER_JOB_IDS
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not str(path).endswith("/" + expected_suffix)
        or type(identity) is not dict
        or set(identity) != identity_fields
        or any(type(identity[field]) is not int or identity[field] < 0
               for field in identity_fields)
        or type(row.get("completion_size")) is not int
        or row["completion_size"] <= 0
        or row.get("completion_mode") != 0o444
        or type(row.get("completion_mode")) is not int
        or any(
            type(row.get(field)) is not str
            or SHA256_RE.fullmatch(row[field]) is None
            for field in (
                "completion_sha256", "completion_digest",
                "holder_summary_digest",
            )
        )
        or type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or object_sha256(unsigned) != claimed
    ):
        fail("holder completion anchor binding differs")
    row["initial_inode_identity"] = dict(identity)
    return row


def _safe_component(value: str, *, label: str) -> str:
    if (
        type(value) is not str
        or value in ("", ".", "..")
        or os.path.sep in value
        or (os.path.altsep is not None and os.path.altsep in value)
    ):
        fail(f"{label} path component differs")
    return value


def _open_stable_directory_at(
    parent_fd: int, name: str, *, label: str,
) -> tuple[int, os.stat_result]:
    name = _safe_component(name, label=label)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        fail(f"{label} held directory replay is unavailable")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = sorted(os.listdir(descriptor))
        middle = os.fstat(descriptor)
        second = sorted(os.listdir(descriptor))
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise DecodedEvalDeploymentControllerError(
            f"cannot replay held {label}"
        ) from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or os.get_inheritable(descriptor)
    ):
        os.close(descriptor)
        fail(f"{label} held directory identity differs")
    return descriptor, before


def _stable_file_at(
    parent_fd: int, name: str, *, label: str,
) -> tuple[bytes, dict[str, Any]]:
    name = _safe_component(name, label=label)
    if not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} held file replay is unavailable")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise DecodedEvalDeploymentControllerError(
            f"cannot replay held {label}"
        ) from error
    finally:
        if "descriptor" in locals():
            try:
                os.close(descriptor)
            except OSError:
                pass
    digest = hashlib.sha256(first).hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or len(first) != before.st_size
    ):
        fail(f"{label} held file identity differs")
    return first, {
        "sha256": digest, "size": len(first),
        "mode": stat.S_IMODE(before.st_mode), "device": before.st_dev,
        "inode": before.st_ino, "uid": before.st_uid,
        "gid": before.st_gid, "nlink": before.st_nlink,
        "rdev": before.st_rdev,
        "blocks": getattr(before, "st_blocks", 0),
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
        "identity": _identity_row(before),
    }


def _exact_option(arguments: Sequence[str], option: str) -> str:
    positions = [
        index for index, value in enumerate(arguments) if value == option
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(arguments)
        or not isinstance(arguments[positions[0] + 1], str)
        or arguments[positions[0] + 1].startswith("--")
    ):
        fail(f"executor {option} differs")
    return arguments[positions[0] + 1]


def _verify_completion_anchor_from_work_root(
    anchor: Mapping[str, Any],
    *,
    target_arguments: Sequence[str],
    work_root: _HeldWorkRoot,
) -> dict[str, Any]:
    row = _validate_holder_completion_anchor(anchor)
    evaluation_root = _absolute(
        _exact_option(target_arguments, "--evaluation-root"),
        label="executor evaluation root",
    )
    holder = _exact_option(target_arguments, "--holder-job-id")
    work_path = Path(work_root.authority["path"])
    if (
        holder != row["holder_job_id"]
        or holder not in HOLDER_JOB_IDS
        or evaluation_root.parent != work_path
        or evaluation_root.name in ("", ".", "..")
    ):
        fail("holder completion anchor executor ownership differs")
    completion_path = (
        evaluation_root / EXECUTION_SHARD_DIRECTORY
        / f"{holder}{HOLDER_COMPLETION_SUFFIX}"
    )
    if str(completion_path) != row["completion_path"]:
        fail("holder completion anchor path differs")
    work_root.capture(expected_entries=None)
    evaluation_fd, _ = _open_stable_directory_at(
        work_root.root_fd, evaluation_root.name, label="evaluation root",
    )
    execution_fd: int | None = None
    holder_fd: int | None = None
    try:
        execution_fd, _ = _open_stable_directory_at(
            evaluation_fd, EXECUTION_SHARD_DIRECTORY,
            label="execution shard root",
        )
        completion_raw, completion_file = _stable_file_at(
            execution_fd, f"{holder}{HOLDER_COMPLETION_SUFFIX}",
            label="holder completion",
        )
        immutable = {
            field: completion_file[field]
            for field in ("device", "inode", "uid", "gid", "rdev")
        }
        if (
            immutable != row["initial_inode_identity"]
            or completion_file["sha256"] != row["completion_sha256"]
            or completion_file["size"] != row["completion_size"]
            or completion_file["mode"] != row["completion_mode"]
        ):
            fail("holder completion anchor physical file differs")
        completion = _strict_json(
            completion_raw, label="anchored holder completion"
        )
        completion_unsigned = dict(completion)
        completion_digest = completion_unsigned.pop("completion_digest", None)
        if (
            completion.get("holder_job_id") != holder
            or completion.get("evaluation_root") != str(evaluation_root)
            or completion_digest != row["completion_digest"]
            or object_sha256(completion_unsigned) != completion_digest
            or completion.get("holder_summary_digest")
            != row["holder_summary_digest"]
        ):
            fail("holder completion anchor document digest differs")
        holder_fd, _ = _open_stable_directory_at(
            execution_fd, holder, label="holder execution root",
        )
        summary_raw, _ = _stable_file_at(
            holder_fd, SHARD_SUMMARY_FILENAME, label="holder summary",
        )
        summary = _strict_json(summary_raw, label="anchored holder summary")
        summary_unsigned = dict(summary)
        summary_digest = summary_unsigned.pop("summary_digest", None)
        if (
            summary_digest != row["holder_summary_digest"]
            or object_sha256(summary_unsigned) != summary_digest
        ):
            fail("holder completion anchor summary digest differs")
        holder_replay_fd, _ = _open_stable_directory_at(
            execution_fd, holder, label="holder execution final replay",
        )
        os.close(holder_replay_fd)
        final_raw, final_file = _stable_file_at(
            execution_fd, f"{holder}{HOLDER_COMPLETION_SUFFIX}",
            label="holder completion final replay",
        )
        if final_raw != completion_raw or any(
            final_file[field] != completion_file[field]
            for field in (
                "sha256", "size", "mode", "device", "inode", "uid",
                "gid", "nlink", "rdev",
            )
        ):
            fail("holder completion anchor changed during controller replay")
    finally:
        if holder_fd is not None:
            os.close(holder_fd)
        if execution_fd is not None:
            os.close(execution_fd)
        os.close(evaluation_fd)
    work_root.capture(expected_entries=None)
    return row


def _validate_aggregate_anchor_identity(
    value: Any, *, label: str, directory: bool,
) -> dict[str, int]:
    fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(value[field]) is not int or value[field] < 0
               for field in fields)
        or (directory and not stat.S_ISDIR(value["mode"]))
        or (not directory and not stat.S_ISREG(value["mode"]))
    ):
        fail(f"{label} identity differs")
    return dict(value)


def _validate_aggregate_anchor_file(
    value: Any, *, relative_path: str, mode: int, label: str,
) -> dict[str, Any]:
    fields = {
        "relative_path", "sha256", "size", "mode", "identity",
        "object_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} field closure differs")
    row = dict(value)
    identity = _validate_aggregate_anchor_identity(
        row.get("identity"), label=label, directory=False,
    )
    if (
        row.get("relative_path") != relative_path
        or type(row.get("sha256")) is not str
        or SHA256_RE.fullmatch(row["sha256"]) is None
        or type(row.get("size")) is not int
        or row["size"] <= 0
        or row.get("mode") != mode
        or type(row.get("mode")) is not int
        or stat.S_IMODE(identity["mode"]) != mode
        or identity["size"] != row["size"]
        or identity["nlink"] != 1
        or type(row.get("object_digest")) is not str
        or SHA256_RE.fullmatch(row["object_digest"]) is None
    ):
        fail(f"{label} binding differs")
    row["identity"] = identity
    return row


def _validate_aggregate_completion_anchor(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "aggregate_root",
        "aggregate_root_identity", "aggregate_file", "private_file",
        "public_file", "media_directory_identity", "media_file_count",
        "media_rows_digest", "media_tree_digest", "anchor_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("aggregate completion anchor field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest", None)
    root = (
        Path(row["aggregate_root"])
        if type(row.get("aggregate_root")) is str else Path()
    )
    root_identity = _validate_aggregate_anchor_identity(
        row.get("aggregate_root_identity"),
        label="aggregate root", directory=True,
    )
    media_identity = _validate_aggregate_anchor_identity(
        row.get("media_directory_identity"),
        label="aggregate media directory", directory=True,
    )
    aggregate_file = _validate_aggregate_anchor_file(
        row.get("aggregate_file"), relative_path="evaluation_complete.json",
        mode=0o444, label="aggregate file",
    )
    private_file = _validate_aggregate_anchor_file(
        row.get("private_file"), relative_path="private_blind_mapping.json",
        mode=0o400, label="private mapping file",
    )
    public_file = _validate_aggregate_anchor_file(
        row.get("public_file"), relative_path="blind_review_packet.json",
        mode=0o444, label="public packet file",
    )
    if (
        row.get("schema_version") != AGGREGATE_COMPLETION_ANCHOR_SCHEMA
        or type(row.get("evaluation_id")) is not str
        or not row["evaluation_id"]
        or not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or root.name in ("", ".", "..")
        or stat.S_IMODE(root_identity["mode"]) != 0o555
        or stat.S_IMODE(media_identity["mode"]) != 0o555
        or type(row.get("media_file_count")) is not int
        or row["media_file_count"] <= 0
        or any(
            type(row.get(field)) is not str
            or SHA256_RE.fullmatch(row[field]) is None
            for field in ("media_rows_digest", "media_tree_digest")
        )
        or type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or object_sha256(unsigned) != claimed
    ):
        fail("aggregate completion anchor binding differs")
    row.update(
        aggregate_root_identity=root_identity,
        media_directory_identity=media_identity,
        aggregate_file=aggregate_file,
        private_file=private_file,
        public_file=public_file,
    )
    return row


def _verify_aggregate_anchor_from_work_root(
    anchor: Mapping[str, Any],
    *,
    target_arguments: Sequence[str],
    work_root: _HeldWorkRoot,
) -> dict[str, Any]:
    row = _validate_aggregate_completion_anchor(anchor)
    aggregate_root = _absolute(
        _exact_option(target_arguments, "--aggregate-root"),
        label="aggregate output root",
    )
    if (
        aggregate_root != Path(row["aggregate_root"])
        or aggregate_root.parent != Path(work_root.authority["path"])
    ):
        fail("aggregate completion anchor root differs")
    work_root.capture(expected_entries=None)
    root_fd, root_stat = _open_stable_directory_at(
        work_root.root_fd, aggregate_root.name,
        label="aggregate publication root",
    )
    media_fd: int | None = None
    try:
        if _identity_row(root_stat) != row["aggregate_root_identity"]:
            fail("aggregate completion anchor root identity differs")
        expected_root_entries = {
            "media", "evaluation_complete.json",
            "private_blind_mapping.json", "blind_review_packet.json",
        }
        if set(os.listdir(root_fd)) != expected_root_entries:
            fail("aggregate completion anchor root closure differs")
        documents: dict[str, dict[str, Any]] = {}
        file_specs = (
            (
                "aggregate_file", "evaluation_complete.json",
                "aggregate_digest",
            ),
            (
                "private_file", "private_blind_mapping.json",
                "private_mapping_digest",
            ),
            (
                "public_file", "blind_review_packet.json",
                "public_packet_digest",
            ),
        )
        for binding_key, basename, digest_field in file_specs:
            raw, observed = _stable_file_at(
                root_fd, basename, label=f"aggregate {binding_key}"
            )
            binding = row[binding_key]
            if (
                observed["sha256"] != binding["sha256"]
                or observed["size"] != binding["size"]
                or observed["mode"] != binding["mode"]
                or observed["identity"] != binding["identity"]
            ):
                fail(f"aggregate completion {binding_key} file differs")
            document = _strict_json(raw, label=f"aggregate {binding_key}")
            unsigned_document = dict(document)
            declared = unsigned_document.pop(digest_field, None)
            if (
                declared != binding["object_digest"]
                or object_sha256(unsigned_document) != declared
            ):
                fail(f"aggregate completion {binding_key} digest differs")
            documents[binding_key] = document
        aggregate_document = documents["aggregate_file"]
        private_document = documents["private_file"]
        public_document = documents["public_file"]
        if (
            aggregate_document.get("evaluation_id") != row["evaluation_id"]
            or private_document.get("evaluation_id") != row["evaluation_id"]
            or aggregate_document.get("private_mapping_digest")
            != row["private_file"]["object_digest"]
            or aggregate_document.get("public_packet_digest")
            != row["public_file"]["object_digest"]
            or public_document.get("private_mapping_digest")
            != row["private_file"]["object_digest"]
        ):
            fail("aggregate completion document cross-binding differs")
        media_fd, media_stat = _open_stable_directory_at(
            root_fd, "media", label="aggregate media root",
        )
        if _identity_row(media_stat) != row["media_directory_identity"]:
            fail("aggregate completion media identity differs")
        media_names = sorted(os.listdir(media_fd))
        media_rows: list[dict[str, Any]] = []
        for basename in media_names:
            if (
                re.fullmatch(r"[0-9a-f]{64}\.mp4", basename) is None
            ):
                fail("aggregate completion media basename differs")
            raw, observed = _stable_file_at(
                media_fd, basename, label="aggregate media file"
            )
            expected_sha = basename[:-4]
            if (
                observed["sha256"] != expected_sha
                or observed["mode"] != 0o444
                or len(raw) != observed["size"]
            ):
                fail("aggregate completion media file differs")
            media_rows.append(
                {
                    "relative_path": f"media/{basename}",
                    "sha256": expected_sha,
                    "size": observed["size"],
                    "mode": 0o444,
                    "identity": observed["identity"],
                }
            )
        rows_digest = object_sha256(media_rows)
        tree_digest = object_sha256(
            {
                "media_directory_identity": row[
                    "media_directory_identity"
                ],
                "media_file_count": len(media_rows),
                "media_rows_digest": rows_digest,
            }
        )
        if (
            len(media_rows) != row["media_file_count"]
            or rows_digest != row["media_rows_digest"]
            or tree_digest != row["media_tree_digest"]
        ):
            fail("aggregate completion media tree digest differs")
        if (
            _identity_row(os.fstat(media_fd))
            != row["media_directory_identity"]
            or set(os.listdir(media_fd)) != set(media_names)
            or _identity_row(os.fstat(root_fd))
            != row["aggregate_root_identity"]
            or set(os.listdir(root_fd)) != expected_root_entries
        ):
            fail("aggregate completion tree changed during replay")
    finally:
        if media_fd is not None:
            os.close(media_fd)
        os.close(root_fd)
    work_root.capture(expected_entries=None)
    return row


def _completion_anchor_channel_binding(
    *, descriptor: int, controller_pid: int, target_pid: int,
    expected_target: str,
) -> dict[str, Any]:
    if expected_target not in DYNAMIC_ANCHOR_TARGETS:
        fail("completion anchor channel target differs")
    value: dict[str, Any] = {
        "schema_version": COMPLETION_ANCHOR_CHANNEL_SCHEMA,
        "descriptor": descriptor,
        "controller_pid": controller_pid,
        "target_pid": target_pid,
        "expected_target": expected_target,
    }
    value["binding_digest"] = object_sha256(value)
    return value


def _recv_completion_anchor_packet(
    channel: socket.socket, *, expected_pid: int,
) -> bytes:
    if (
        not hasattr(socket, "SO_PASSCRED")
        or not hasattr(socket, "SCM_CREDENTIALS")
    ):
        fail("kernel credentialed completion anchor channel is unavailable")
    credentials_size = struct.calcsize("3i")
    try:
        raw, ancillary, flags, _ = channel.recvmsg(
            MAX_COMPLETION_ANCHOR_PACKET_SIZE + 1,
            socket.CMSG_SPACE(credentials_size),
        )
    except OSError as error:
        raise DecodedEvalDeploymentControllerError(
            "cannot receive completion anchor packet"
        ) from error
    credentials = []
    for level, kind, value in ancillary:
        if (
            level == socket.SOL_SOCKET
            and kind == socket.SCM_CREDENTIALS
            and len(value) >= credentials_size
        ):
            credentials.append(struct.unpack("3i", value[:credentials_size]))
    if (
        not raw
        or len(raw) > MAX_COMPLETION_ANCHOR_PACKET_SIZE
        or flags & getattr(socket, "MSG_TRUNC", 0)
        or credentials != [(expected_pid, os.getuid(), os.getgid())]
    ):
        fail("completion anchor packet sender or framing differs")
    return raw


def _wait_for_exact_child_exit(child_pid: int) -> None:
    try:
        observed_pid, status = os.waitpid(child_pid, 0)
    except OSError as error:
        raise DecodedEvalDeploymentControllerError(
            "cannot wait for verified executor child"
        ) from error
    if (
        observed_pid != child_pid
        or not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
    ):
        fail("verified executor child did not exit successfully")


def run_target_with_completion_anchor(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    work_root: _HeldWorkRoot,
    target: str,
    target_arguments: Sequence[str],
) -> dict[str, Any]:
    """Run one verified target while retaining its dynamic anchor in memory."""

    if (
        not command
        or type(command[0]) is not str
        or target not in DYNAMIC_ANCHOR_TARGETS
    ):
        fail("verified dynamic-anchor target command differs")
    if not hasattr(os, "fork") or not hasattr(socket, "SOCK_SEQPACKET"):
        fail("credentialed executor continuation is unavailable")
    if (
        not hasattr(socket, "SO_PASSCRED")
        or not hasattr(socket, "SCM_CREDENTIALS")
    ):
        fail("credentialed executor continuation is unavailable")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent.set_inheritable(False)
        child.set_inheritable(False)
        controller_pid = os.getpid()
        child_pid = os.fork()
        if child_pid == 0:
            try:
                parent.close()
                child_pid_value = os.getpid()
                channel_binding = _completion_anchor_channel_binding(
                    descriptor=child.fileno(),
                    controller_pid=controller_pid,
                    target_pid=child_pid_value,
                    expected_target=target,
                )
                child_environment = dict(environment)
                child_environment[COMPLETION_ANCHOR_CHANNEL_ENV] = (
                    canonical_json_bytes(channel_binding).decode("utf-8")
                )
                child_environment.pop(COMPLETION_ANCHOR_SENT_ENV, None)
                os.set_inheritable(work_root.parent_fd, True)
                os.set_inheritable(work_root.root_fd, True)
                child.set_inheritable(True)
                os.dup2(2, 1)
                os.execve(command[0], list(command), child_environment)
            except BaseException as error:
                try:
                    os.write(
                        2,
                        (
                            "decoded-eval-controller child exec failed: "
                            + str(error) + "\n"
                        ).encode("utf-8", "replace"),
                    )
                finally:
                    os._exit(70)
        child.close()
        raw: bytes | None = None
        receive_error: BaseException | None = None
        try:
            raw = _recv_completion_anchor_packet(
                parent, expected_pid=child_pid
            )
        except BaseException as error:
            receive_error = error
        _wait_for_exact_child_exit(child_pid)
        if receive_error is not None:
            raise receive_error
        assert raw is not None
        parent.setblocking(False)
        try:
            trailing = parent.recv(MAX_COMPLETION_ANCHOR_PACKET_SIZE + 1)
        except BlockingIOError:
            fail("completion anchor channel remained open after child exit")
        if trailing != b"":
            fail("completion anchor channel emitted more than one packet")
        value = _strict_json(raw, label="holder completion anchor packet")
        if target == EXECUTOR_TARGET:
            return _verify_completion_anchor_from_work_root(
                value,
                target_arguments=target_arguments,
                work_root=work_root,
            )
        return _verify_aggregate_anchor_from_work_root(
            value,
            target_arguments=target_arguments,
            work_root=work_root,
        )
    finally:
        for endpoint in (parent, child):
            try:
                endpoint.close()
            except OSError:
                pass


def _load_runtime(raw: bytes, *, origin: Path) -> Mapping[str, Any]:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DecodedEvalDeploymentControllerError(
            "verified runtime source is not UTF-8"
        ) from error
    namespace: dict[str, Any] = {
        "__name__": "_apv2_detached_verified_runtime",
        "__file__": str(origin), "__package__": None, "__spec__": None,
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(origin), "exec", dont_inherit=True), namespace)
    required = {
        "capture_executable_binding", "capture_file_binding",
        "capture_torchrun_binding", "capture_release_binding",
        "extract_verified_release",
        "publish_controller_authority_receipt",
        "validate_controller_authority_binding", "verified_target_argv",
        "RELEASE_GENERATION", "EVAL_RELEASE_MEMBERS",
    }
    if not required.issubset(namespace):
        fail("captured verified runtime API closure differs")
    return namespace


def validate_request(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "release_generation", "work_root_authority", "controller",
        "root_python", "frozen_python", "site_packages_path", "torchrun",
        "release_root", "archive", "manifest", "manifest_digest",
        "content_revision", "envelope", "envelope_digest",
        "verified_runtime_source", "source_runtime_spec_path",
        "source_spec_authority_receipt_path",
        "controller_authority_receipt_path",
        "deployment_receipt_path", "automatic_retry", "network_allowed",
        "scientific_promotion_authorized", "request_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("deployment request field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("request_digest", None)
    work_root = validate_work_root_authority(row["work_root_authority"])
    row["work_root_authority"] = work_root
    for field in (
        "controller", "root_python", "frozen_python", "torchrun",
        "verified_runtime_source",
    ):
        item = row[field]
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            fail(f"deployment request {field} closure differs")
        _absolute(item["path"], label=f"deployment request {field}")
        if SHA256_RE.fullmatch(str(item["sha256"])) is None:
            fail(f"deployment request {field} SHA differs")
    for field in ("archive", "manifest", "envelope"):
        item = row[field]
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            fail(f"deployment request {field} closure differs")
        _absolute(item["path"], label=f"deployment request {field}")
        if SHA256_RE.fullmatch(str(item["sha256"])) is None:
            fail(f"deployment request {field} SHA differs")
    for field in (
        "site_packages_path", "release_root", "source_runtime_spec_path",
        "source_spec_authority_receipt_path",
        "controller_authority_receipt_path", "deployment_receipt_path",
    ):
        _absolute(row[field], label=f"deployment request {field}")
    output_fields = (
        "release_root", "source_runtime_spec_path",
        "source_spec_authority_receipt_path",
        "controller_authority_receipt_path", "deployment_receipt_path",
    )
    outputs = [Path(row[field]) for field in output_fields]
    if len(set(outputs)) != len(outputs) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(outputs)
        for right in outputs[index + 1:]
    ):
        fail("deployment request output path topology differs")
    work_root_path = Path(work_root["path"])
    if any(path.parent != work_root_path for path in outputs):
        fail("deployment request outputs escape held work root")
    for field in (
        "manifest_digest", "envelope_digest", "request_digest",
    ):
        if not isinstance(row[field], str) or SHA256_RE.fullmatch(row[field]) is None:
            fail(f"deployment request {field} differs")
    if not isinstance(row["content_revision"], str) or SHA1_RE.fullmatch(
        row["content_revision"]
    ) is None:
        fail("deployment request content revision differs")
    if (
        row["schema_version"] != REQUEST_SCHEMA
        or row["automatic_retry"] is not False
        or row["network_allowed"] is not False
        or row["scientific_promotion_authorized"] is not False
        or claimed != object_sha256(unsigned)
    ):
        fail("deployment request authority differs")
    return row


def _controller_binding(runtime: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    observed = runtime["capture_file_binding"](
        Path(request["controller"]["path"]), label="detached controller",
        expected_sha256=request["controller"]["sha256"], expected_mode=0o444,
    )
    running = Path(__file__).resolve(strict=True)
    if observed["path"] != str(running):
        fail("running controller differs from deployment request")
    return observed


def capture_authority(
    *, request_path: Path, expected_request_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_raw, request_file = stable_file(
        request_path, label="deployment request",
        expected_sha256=expected_request_sha256, expected_mode=0o444,
    )
    request = validate_request(_strict_json(request_raw, label="deployment request"))
    work_root_path = Path(request["work_root_authority"]["path"])
    if request_path.parent != work_root_path:
        fail("deployment request is outside held work root")
    work_root = _HeldWorkRoot.open(request["work_root_authority"])
    work_root.capture(expected_entries={request_path.name})
    for field in (
        "source_runtime_spec_path", "source_spec_authority_receipt_path"
    ):
        if os.path.lexists(request[field]):
            fail(f"deployment request {field} is not fresh")
    runtime_source_path = Path(request["verified_runtime_source"]["path"])
    runtime_raw, runtime_source_file = stable_file(
        runtime_source_path, label="detached verified runtime source",
        expected_sha256=request["verified_runtime_source"]["sha256"],
        expected_mode=0o444,
    )
    runtime = _load_runtime(runtime_raw, origin=runtime_source_path)
    if request["release_generation"] != runtime["RELEASE_GENERATION"]:
        fail("deployment request release generation differs")
    release_root = Path(request["release_root"])
    if os.path.lexists(release_root):
        fail("materialized release root is not fresh")
    runtime["extract_verified_release"](
        archive=Path(request["archive"]["path"]),
        expected_archive_sha256=request["archive"]["sha256"],
        manifest=Path(request["manifest"]["path"]),
        expected_manifest_sha256=request["manifest"]["sha256"],
        expected_content_revision=request["content_revision"],
        envelope=Path(request["envelope"]["path"]),
        expected_envelope_sha256=request["envelope"]["sha256"],
        output_root=release_root,
        retained_parent_fd=work_root.root_fd,
    )
    work_root.capture(
        expected_entries={request_path.name, release_root.name}
    )
    runtime_path = release_root / VERIFIED_RUNTIME_RELATIVE_PATH
    materialized_runtime_raw, runtime_file = stable_file(
        runtime_path, label="materialized verified runtime",
        expected_sha256=request["verified_runtime_source"]["sha256"],
        expected_mode=0o444,
    )
    if materialized_runtime_raw != runtime_raw:
        fail("materialized verified runtime differs from detached source")
    controller = _controller_binding(runtime, request)
    work_root.capture(
        expected_entries={request_path.name, release_root.name}
    )
    root_python = runtime["capture_executable_binding"](
        Path(request["root_python"]["path"]), label="root Python"
    )
    frozen_python = runtime["capture_executable_binding"](
        Path(request["frozen_python"]["path"]), label="frozen Python"
    )
    if (
        root_python["sha256"] != request["root_python"]["sha256"]
        or frozen_python["sha256"] != request["frozen_python"]["sha256"]
        or root_python["path"] != str(ROOT_PYTHON_PATH)
        or root_python["uid"] != ROOT_PYTHON_UID
        or root_python["gid"] != ROOT_PYTHON_GID
        or root_python["mode"] != ROOT_PYTHON_MODE
    ):
        fail("deployment interpreter pin differs")
    torchrun = runtime["capture_torchrun_binding"](
        Path(request["site_packages_path"]), label="deployment torchrun"
    )
    if (
        torchrun["source"]["path"] != request["torchrun"]["path"]
        or torchrun["source"]["sha256"] != request["torchrun"]["sha256"]
    ):
        fail("deployment torchrun pin differs")
    work_root.capture(
        expected_entries={request_path.name, release_root.name}
    )
    release = runtime["capture_release_binding"](
        release_root=release_root,
        archive=Path(request["archive"]["path"]),
        expected_archive_sha256=request["archive"]["sha256"],
        manifest=Path(request["manifest"]["path"]),
        expected_manifest_sha256=request["manifest"]["sha256"],
        expected_content_revision=request["content_revision"],
        envelope=Path(request["envelope"]["path"]),
        expected_envelope_sha256=request["envelope"]["sha256"],
        retained_parent_fd=work_root.root_fd,
    )
    if (
        release["manifest_digest"] != request["manifest_digest"]
        or release["envelope_digest"] != request["envelope_digest"]
    ):
        fail("deployment release digest pin differs")
    authority = runtime["publish_controller_authority_receipt"](
        Path(request["controller_authority_receipt_path"]),
        controller_binding=controller, root_python_binding=root_python,
        frozen_python_binding=frozen_python,
        site_packages_binding=torchrun["site_packages"],
        release_binding=release, torchrun_binding=torchrun,
        retained_parent_fd=work_root.root_fd,
    )
    controller_authority_path = Path(
        request["controller_authority_receipt_path"]
    )
    phase_a_entries_before_receipt = {
        request_path.name,
        release_root.name,
        controller_authority_path.name,
    }
    work_root_before_receipt = work_root.capture(
        expected_entries=phase_a_entries_before_receipt
    )
    expected_phase_a_entries = sorted(
        phase_a_entries_before_receipt
        | {Path(request["deployment_receipt_path"]).name}
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "release_generation": runtime["RELEASE_GENERATION"],
        "work_root_authority": request["work_root_authority"],
        "work_root_capture_before_receipt": work_root_before_receipt,
        "work_root_expected_phase_a_entries": expected_phase_a_entries,
        "work_root_held_fd_through_controller_publication": True,
        "deployment_request": request_file,
        "deployment_request_digest": request["request_digest"],
        "controller": controller, "root_python": root_python,
        "frozen_python": frozen_python,
        "site_packages": torchrun["site_packages"], "torchrun": torchrun,
        "release": release, "verified_runtime_source": runtime_source_file,
        "verified_runtime": runtime_file,
        "source_runtime_spec_path": request["source_runtime_spec_path"],
        "source_spec_authority_receipt_path": request[
            "source_spec_authority_receipt_path"
        ],
        "controller_authority": authority,
        "literal_request_sha_required": True,
        "controller_executed_from_same_fd_captured_bytes": True,
        "verified_runtime_executed_from_same_fd_captured_bytes": True,
        "automatic_retry": False, "network_used": False,
        "scientific_promotion_authorized": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    raw = canonical_json_bytes(receipt) + b"\n"
    deployment_receipt_file = work_root.publish_member(
        Path(request["deployment_receipt_path"]).name,
        raw,
        expected_entries_before=phase_a_entries_before_receipt,
        expected_entries_after=set(expected_phase_a_entries),
        label="deployment receipt",
    )
    if (
        deployment_receipt_file["path"] != request["deployment_receipt_path"]
        or deployment_receipt_file["sha256"]
        != hashlib.sha256(raw).hexdigest()
    ):
        fail("deployment receipt held publication differs")
    work_root.capture(expected_entries=set(expected_phase_a_entries))
    work_root.close()
    return receipt, runtime


def load_deployment_receipt(
    path: Path, *, expected_sha256: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    raw, _ = stable_file(
        path, label="deployment receipt", expected_sha256=expected_sha256,
        expected_mode=0o444,
    )
    value = _strict_json(raw, label="deployment receipt")
    fields = {
        "schema_version", "release_generation", "work_root_authority",
        "work_root_capture_before_receipt",
        "work_root_expected_phase_a_entries",
        "work_root_held_fd_through_controller_publication",
        "deployment_request",
        "deployment_request_digest", "controller", "root_python",
        "frozen_python", "site_packages", "torchrun", "release",
        "verified_runtime_source", "verified_runtime",
        "source_runtime_spec_path", "source_spec_authority_receipt_path",
        "controller_authority",
        "literal_request_sha_required",
        "controller_executed_from_same_fd_captured_bytes",
        "verified_runtime_executed_from_same_fd_captured_bytes",
        "automatic_retry", "network_used", "scientific_promotion_authorized",
        "receipt_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("deployment receipt field closure differs")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        value["schema_version"] != RECEIPT_SCHEMA
        or value["work_root_held_fd_through_controller_publication"] is not True
        or value["literal_request_sha_required"] is not True
        or value["controller_executed_from_same_fd_captured_bytes"] is not True
        or value["verified_runtime_executed_from_same_fd_captured_bytes"] is not True
        or value["automatic_retry"] is not False
        or value["network_used"] is not False
        or value["scientific_promotion_authorized"] is not False
        or claimed != object_sha256(unsigned)
    ):
        fail("deployment receipt authority differs")
    work_root_authority = validate_work_root_authority(
        value["work_root_authority"]
    )
    expected_phase_a_entries = value["work_root_expected_phase_a_entries"]
    if (
        type(expected_phase_a_entries) is not list
        or len(expected_phase_a_entries) != 4
        or any(type(item) is not str for item in expected_phase_a_entries)
        or expected_phase_a_entries != sorted(set(expected_phase_a_entries))
    ):
        fail("deployment receipt phase-A entry closure differs")
    before_receipt = _validate_work_root_capture(
        value["work_root_capture_before_receipt"],
        authority=work_root_authority,
        expected_entries=[
            item
            for item in expected_phase_a_entries
            if item != path.name
        ],
    )
    request_file = value["deployment_request"]
    request_raw, observed_request = stable_file(
        request_file["path"],
        label="receipt deployment request",
        expected_sha256=request_file["sha256"],
        expected_mode=0o444,
    )
    request = validate_request(
        _strict_json(request_raw, label="receipt deployment request")
    )
    if (
        observed_request != request_file
        or request["work_root_authority"] != work_root_authority
        or request["request_digest"] != value["deployment_request_digest"]
        or Path(request["deployment_receipt_path"]) != path
        or sorted(
            {
                Path(request["deployment_request_path"]).name
                if "deployment_request_path" in request
                else Path(request_file["path"]).name,
                Path(request["release_root"]).name,
                Path(request["controller_authority_receipt_path"]).name,
                path.name,
            }
        )
        != expected_phase_a_entries
        or before_receipt["entries"]
        != [item for item in expected_phase_a_entries if item != path.name]
    ):
        fail("deployment receipt work root continuity differs")
    held_work_root = _HeldWorkRoot.open(work_root_authority)
    current_work_root = held_work_root.capture(expected_entries=None)
    held_work_root.close()
    if not set(expected_phase_a_entries).issubset(current_work_root["entries"]):
        fail("deployment receipt work root entries are incomplete")
    runtime_source_file = value["verified_runtime_source"]
    runtime_source_raw, observed_runtime_source = stable_file(
        runtime_source_file["path"], label="receipt detached verified runtime",
        expected_sha256=runtime_source_file["sha256"], expected_mode=0o444,
    )
    if observed_runtime_source != runtime_source_file:
        fail("deployment receipt detached runtime identity differs")
    runtime_file = value["verified_runtime"]
    runtime_raw, observed_runtime = stable_file(
        runtime_file["path"], label="receipt verified runtime",
        expected_sha256=runtime_file["sha256"], expected_mode=0o444,
    )
    if observed_runtime != runtime_file or runtime_raw != runtime_source_raw:
        fail("deployment receipt runtime identity differs")
    for field in (
        "source_runtime_spec_path", "source_spec_authority_receipt_path"
    ):
        _absolute(value[field], label=f"deployment receipt {field}")
    runtime = _load_runtime(runtime_raw, origin=Path(runtime_file["path"]))
    runtime["validate_controller_authority_binding"](
        value["controller_authority"], controller_binding=value["controller"],
        root_python_binding=value["root_python"],
        frozen_python_binding=value["frozen_python"],
        site_packages_binding=value["site_packages"],
        release_binding=value["release"], torchrun_binding=value["torchrun"],
        require_torchrun_continuity=True, verify_file=True,
    )
    if value["release_generation"] != runtime["RELEASE_GENERATION"]:
        fail("deployment receipt release generation differs")
    return value, runtime


def _pair(binding: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(binding["path"]), "sha256": str(binding["sha256"])}


def _validate_source_spec_authority_continuity(
    value: Any, deployment: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version", "pins", "pin_files", "sources", "runtime",
        "spec_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("source/runtime spec field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("spec_digest", None)
    if (
        row["schema_version"] != SOURCE_RUNTIME_SCHEMA
        or not isinstance(claimed, str)
        or SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        fail("source/runtime spec digest differs")
    runtime = row["runtime"]
    pin_files = row["pin_files"]
    if type(runtime) is not dict or type(pin_files) is not dict:
        fail("source/runtime authority shape differs")
    release = deployment["release"]
    authority = deployment["controller_authority"]
    expected = {
        "root_python": _pair(deployment["root_python"]),
        "python": _pair(deployment["frozen_python"]),
        "site_packages": deployment["site_packages"]["path"],
        "torchrun": _pair(deployment["torchrun"]["source"]),
        "deployment_controller": _pair(deployment["controller"]),
        "controller_authority": {
            "receipt": _pair(authority["receipt"]),
            "authority_digest": authority["authority_digest"],
        },
        "eval_release_root": release["release_root"]["path"],
        "eval_release_archive": _pair(release["archive"]),
        "eval_release_envelope": _pair(release["envelope"]),
        "eval_release_manifest_digest": release["manifest_digest"],
        "eval_release_content_revision": release["content_revision"],
        "eval_release_envelope_digest": release["envelope_digest"],
    }
    if any(runtime.get(field) != item for field, item in expected.items()):
        fail("source/runtime spec differs from detached runtime authority")
    if pin_files.get("inference_release_manifest") != _pair(release["manifest"]):
        fail("source/runtime spec release manifest differs from authority")
    return row


def _held_work_member_name(
    path: Path,
    *,
    authority: Mapping[str, Any],
    label: str,
) -> str:
    root = Path(authority["path"])
    if (
        not path.is_absolute()
        or path.parent != root
        or path.name in ("", ".", "..")
        or os.path.sep in path.name
        or (os.path.altsep is not None and os.path.altsep in path.name)
    ):
        fail(f"{label} escapes held work root")
    return path.name


def _source_spec_work_entries(
    deployment: Mapping[str, Any],
    *,
    deployment_receipt_path: Path,
    source_runtime_spec_path: Path,
    source_spec_authority_path: Path,
) -> tuple[set[str], set[str], set[str]]:
    authority = validate_work_root_authority(
        deployment["work_root_authority"]
    )
    phase_a = {
        _held_work_member_name(
            Path(deployment["deployment_request"]["path"]),
            authority=authority,
            label="deployment request",
        ),
        _held_work_member_name(
            Path(deployment["release"]["release_root"]["path"]),
            authority=authority,
            label="materialized release root",
        ),
        _held_work_member_name(
            Path(deployment["controller_authority"]["receipt"]["path"]),
            authority=authority,
            label="controller authority receipt",
        ),
        _held_work_member_name(
            deployment_receipt_path,
            authority=authority,
            label="deployment receipt",
        ),
    }
    spec_name = _held_work_member_name(
        source_runtime_spec_path,
        authority=authority,
        label="source/runtime spec",
    )
    authority_name = _held_work_member_name(
        source_spec_authority_path,
        authority=authority,
        label="source spec authority receipt",
    )
    phase_b_before = phase_a | {spec_name}
    phase_b_after = phase_b_before | {authority_name}
    if (
        len(phase_a) != 4
        or len(phase_b_before) != 5
        or len(phase_b_after) != 6
        or sorted(phase_a)
        != deployment["work_root_expected_phase_a_entries"]
        or str(source_runtime_spec_path)
        != deployment["source_runtime_spec_path"]
        or str(source_spec_authority_path)
        != deployment["source_spec_authority_receipt_path"]
    ):
        fail("source spec work root entry topology differs")
    return phase_a, phase_b_before, phase_b_after


def publish_source_spec_authority(
    *, deployment_receipt_path: Path,
    expected_deployment_receipt_sha256: str,
    source_runtime_spec_path: Path,
    expected_source_runtime_spec_sha256: str,
) -> dict[str, Any]:
    _, deployment_file = stable_file(
        deployment_receipt_path, label="deployment receipt",
        expected_sha256=expected_deployment_receipt_sha256,
        expected_mode=0o444,
    )
    deployment, _runtime = load_deployment_receipt(
        deployment_receipt_path,
        expected_sha256=expected_deployment_receipt_sha256,
    )
    output = Path(deployment["source_spec_authority_receipt_path"])
    _phase_a, phase_b_before, phase_b_after = _source_spec_work_entries(
        deployment,
        deployment_receipt_path=deployment_receipt_path,
        source_runtime_spec_path=source_runtime_spec_path,
        source_spec_authority_path=output,
    )
    work_root = _HeldWorkRoot.open(deployment["work_root_authority"])
    try:
        work_root.capture(expected_entries=phase_b_before)
        _, held_deployment_file = work_root.stable_member(
            deployment_receipt_path.name,
            expected_sha256=expected_deployment_receipt_sha256,
            expected_mode=0o444,
            expected_entries=phase_b_before,
            label="deployment receipt",
        )
        if held_deployment_file != deployment_file:
            fail("deployment receipt changed while authorizing source spec")
        raw, source_runtime_spec = work_root.stable_member(
            source_runtime_spec_path.name,
            expected_sha256=expected_source_runtime_spec_sha256,
            expected_mode=0o444,
            expected_entries=phase_b_before,
            label="source/runtime spec",
        )
        source_value = _validate_source_spec_authority_continuity(
            _strict_json(raw, label="source/runtime spec"), deployment
        )
        work_root_before_receipt = work_root.capture(
            expected_entries=phase_b_before
        )
        receipt: dict[str, Any] = {
            "schema_version": SOURCE_SPEC_AUTHORITY_SCHEMA,
            "release_generation": deployment["release_generation"],
            "work_root_authority": deployment["work_root_authority"],
            "work_root_capture_before_receipt": work_root_before_receipt,
            "work_root_expected_source_spec_entries": sorted(phase_b_after),
            "work_root_held_fd_through_source_spec_publication": True,
            "deployment_receipt": deployment_file,
            "deployment_receipt_digest": deployment["receipt_digest"],
            "controller_authority": deployment["controller_authority"],
            "source_runtime_spec": source_runtime_spec,
            "source_runtime_spec_digest": source_value["spec_digest"],
            "receipt_path": str(output),
            "literal_source_runtime_spec_sha_required": True,
            "runtime_authority_continuity_verified": True,
            "automatic_retry": False,
            "network_used": False,
            "scientific_promotion_authorized": False,
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        work_root.publish_member(
            output.name,
            canonical_json_bytes(receipt) + b"\n",
            expected_entries_before=phase_b_before,
            expected_entries_after=phase_b_after,
            label="source spec authority receipt",
        )
        work_root.capture(expected_entries=phase_b_after)
        return receipt
    finally:
        work_root.close()


def load_source_spec_authority(
    path: Path, *, expected_sha256: str,
    deployment: Mapping[str, Any],
    deployment_receipt_path: Path,
    expected_deployment_receipt_sha256: str,
) -> dict[str, Any]:
    raw, authority_receipt_file = stable_file(
        path, label="source spec authority receipt",
        expected_sha256=expected_sha256, expected_mode=0o444,
    )
    value = _strict_json(raw, label="source spec authority receipt")
    fields = {
        "schema_version", "release_generation", "deployment_receipt",
        "work_root_authority", "work_root_capture_before_receipt",
        "work_root_expected_source_spec_entries",
        "work_root_held_fd_through_source_spec_publication",
        "deployment_receipt_digest", "controller_authority",
        "source_runtime_spec", "source_runtime_spec_digest", "receipt_path",
        "literal_source_runtime_spec_sha_required",
        "runtime_authority_continuity_verified", "automatic_retry",
        "network_used", "scientific_promotion_authorized", "receipt_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("source spec authority receipt field closure differs")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        value["schema_version"] != SOURCE_SPEC_AUTHORITY_SCHEMA
        or value["release_generation"] != deployment["release_generation"]
        or value["work_root_authority"] != deployment["work_root_authority"]
        or value[
            "work_root_held_fd_through_source_spec_publication"
        ] is not True
        or value["deployment_receipt_digest"] != deployment["receipt_digest"]
        or value["controller_authority"] != deployment["controller_authority"]
        or value["receipt_path"] != str(path)
        or value["literal_source_runtime_spec_sha_required"] is not True
        or value["runtime_authority_continuity_verified"] is not True
        or value["automatic_retry"] is not False
        or value["network_used"] is not False
        or value["scientific_promotion_authorized"] is not False
        or claimed != object_sha256(unsigned)
    ):
        fail("source spec authority receipt differs")
    deployment_file = value["deployment_receipt"]
    if deployment_file["path"] != str(deployment_receipt_path):
        fail("source authority deployment receipt path differs")
    source_runtime_spec = value["source_runtime_spec"]
    phase_a, phase_b_before, phase_b_after = _source_spec_work_entries(
        deployment,
        deployment_receipt_path=deployment_receipt_path,
        source_runtime_spec_path=Path(source_runtime_spec["path"]),
        source_spec_authority_path=path,
    )
    expected_source_spec_entries = value[
        "work_root_expected_source_spec_entries"
    ]
    if (
        expected_source_spec_entries != sorted(phase_b_after)
        or _validate_work_root_capture(
            value["work_root_capture_before_receipt"],
            authority=value["work_root_authority"],
            expected_entries=sorted(phase_b_before),
        )["entries"]
        != sorted(phase_b_before)
        or not phase_a.issubset(phase_b_before)
    ):
        fail("source spec authority work root continuity differs")
    work_root = _HeldWorkRoot.open(value["work_root_authority"])
    try:
        current = work_root.capture(expected_entries=None)
        if not phase_b_after.issubset(current["entries"]):
            fail("source spec authority work root entries are incomplete")
        authority_raw, observed_authority_receipt = work_root.stable_member(
            path.name,
            expected_sha256=expected_sha256,
            expected_mode=0o444,
            expected_entries=None,
            label="source spec authority receipt",
        )
        if (
            authority_raw != raw
            or observed_authority_receipt != authority_receipt_file
        ):
            fail("source spec authority receipt held replay differs")
        _, observed_deployment_file = work_root.stable_member(
            deployment_receipt_path.name,
            expected_sha256=expected_deployment_receipt_sha256,
            expected_mode=0o444,
            expected_entries=None,
            label="source authority deployment receipt",
        )
        source_raw, observed_source_runtime_spec = work_root.stable_member(
            Path(source_runtime_spec["path"]).name,
            expected_sha256=source_runtime_spec["sha256"],
            expected_mode=0o444,
            expected_entries=None,
            label="authorized source/runtime spec",
        )
        source_value = _validate_source_spec_authority_continuity(
            _strict_json(source_raw, label="authorized source/runtime spec"),
            deployment,
        )
        work_root.capture(expected_entries=None)
    finally:
        work_root.close()
    if observed_deployment_file != deployment_file:
        fail("source authority deployment receipt identity differs")
    if (
        observed_source_runtime_spec != source_runtime_spec
        or source_value["spec_digest"] != value["source_runtime_spec_digest"]
    ):
        fail("authorized source/runtime spec identity differs")
    return value


def build_target_argv(
    receipt: Mapping[str, Any], runtime: Mapping[str, Any], *, target: str,
    arguments: Sequence[str], capture_receipt_path: Path,
    source_spec_authority: Mapping[str, Any] | None = None,
) -> list[str]:
    argument_list = list(arguments)
    if source_spec_authority is None:
        fail("target source spec authority is absent")
    if (
        source_spec_authority.get("work_root_authority")
        != receipt["work_root_authority"]
        or source_spec_authority.get(
            "work_root_held_fd_through_source_spec_publication"
        ) is not True
    ):
        fail("target work root authority differs")
    if target == "action_preservation_decoded_eval_bridge_v1.py":
        source_runtime_spec = source_spec_authority["source_runtime_spec"]
        required = {
            "--source-runtime-spec": source_runtime_spec["path"],
            "--source-runtime-spec-sha256": source_runtime_spec["sha256"],
        }
        for option, expected in required.items():
            positions = [
                index for index, value in enumerate(argument_list)
                if value == option
            ]
            if (
                len(positions) != 1
                or positions[0] + 1 >= len(argument_list)
                or argument_list[positions[0] + 1] != expected
            ):
                fail(f"bridge {option} differs from detached authority")
    return runtime["verified_target_argv"](
        receipt["root_python"], receipt["frozen_python"],
        receipt["site_packages"], receipt["release"], receipt["controller"],
        receipt["controller_authority"], target, argument_list,
        str(capture_receipt_path),
    )


ROOT_CONTROLLER_BOOTSTRAP_SOURCE = r'''import hashlib,os,stat,sys
path,expected=sys.argv[1:3]
if not os.path.isabs(path) or os.path.normpath(path)!=path or len(expected)!=64 or os.path.islink(path) or os.path.realpath(path)!=path or not hasattr(os,"O_NOFOLLOW"): raise SystemExit(70)
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
def ident(v): return (v.st_dev,v.st_ino,v.st_uid,v.st_gid,v.st_mode,v.st_nlink,v.st_rdev,v.st_size,getattr(v,"st_blocks",0),v.st_mtime_ns,v.st_ctime_ns)
def read():
 os.lseek(fd,0,os.SEEK_SET); out=[]
 while True:
  block=os.read(fd,1024*1024)
  if not block: return b"".join(out)
  out.append(block)
before=os.fstat(fd); first=read(); middle=os.fstat(fd); second=read(); after=os.fstat(fd); named=os.lstat(path); os.close(fd)
if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o444 or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=expected: raise SystemExit(70)
namespace={"__name__":"__main__","__file__":path,"__package__":None,"__spec__":None,"__builtins__":__builtins__}
sys.argv=[path,*sys.argv[3:]]
exec(compile(first,path,"exec",dont_inherit=True),namespace)'''


def controller_bootstrap_argv(
    *, controller_path: Path, expected_controller_sha256: str,
    arguments: Sequence[str], root_python_path: Path = ROOT_PYTHON_PATH,
) -> list[str]:
    return [
        str(root_python_path), "-I", "-S", "-B", "-c",
        ROOT_CONTROLLER_BOOTSTRAP_SOURCE, str(controller_path),
        expected_controller_sha256, *list(arguments),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture-authority")
    capture.add_argument("--deployment-request", required=True)
    capture.add_argument("--deployment-request-sha256", required=True)
    source = commands.add_parser("capture-source-spec-authority")
    source.add_argument("--deployment-receipt", required=True)
    source.add_argument("--deployment-receipt-sha256", required=True)
    source.add_argument("--source-runtime-spec", required=True)
    source.add_argument("--source-runtime-spec-sha256", required=True)
    run = commands.add_parser("run-target")
    run.add_argument("--deployment-receipt", required=True)
    run.add_argument("--deployment-receipt-sha256", required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--capture-receipt", required=True)
    run.add_argument("--source-spec-authority")
    run.add_argument("--source-spec-authority-sha256")
    run.add_argument("target_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "capture-authority":
        receipt, _ = capture_authority(
            request_path=Path(args.deployment_request),
            expected_request_sha256=args.deployment_request_sha256,
        )
        print(receipt["controller_authority"]["authority_digest"], flush=True)
        return 0
    if args.command == "capture-source-spec-authority":
        receipt = publish_source_spec_authority(
            deployment_receipt_path=Path(args.deployment_receipt),
            expected_deployment_receipt_sha256=args.deployment_receipt_sha256,
            source_runtime_spec_path=Path(args.source_runtime_spec),
            expected_source_runtime_spec_sha256=args.source_runtime_spec_sha256,
        )
        print(receipt["receipt_digest"], flush=True)
        return 0
    receipt, runtime = load_deployment_receipt(
        Path(args.deployment_receipt),
        expected_sha256=args.deployment_receipt_sha256,
    )
    target_arguments = list(args.target_arguments)
    if target_arguments[:1] == ["--"]:
        target_arguments = target_arguments[1:]
    if not args.source_spec_authority or not args.source_spec_authority_sha256:
        fail("target source spec authority arguments are absent")
    source_spec_authority = load_source_spec_authority(
        Path(args.source_spec_authority),
        expected_sha256=args.source_spec_authority_sha256,
        deployment=receipt,
        deployment_receipt_path=Path(args.deployment_receipt),
        expected_deployment_receipt_sha256=args.deployment_receipt_sha256,
    )
    command = build_target_argv(
        receipt, runtime, target=args.target, arguments=target_arguments,
        capture_receipt_path=Path(args.capture_receipt),
        source_spec_authority=source_spec_authority,
    )
    work_root = _HeldWorkRoot.open(receipt["work_root_authority"])
    completion_anchor: dict[str, Any] | None = None
    try:
        _, deployment_receipt_file = work_root.stable_member(
            Path(args.deployment_receipt).name,
            expected_sha256=args.deployment_receipt_sha256,
            expected_mode=0o444,
            expected_entries=None,
            label="run-target deployment receipt",
        )
        _, source_spec_authority_file = work_root.stable_member(
            Path(args.source_spec_authority).name,
            expected_sha256=args.source_spec_authority_sha256,
            expected_mode=0o444,
            expected_entries=None,
            label="run-target source spec authority",
        )
        inherited_work_root = work_root.inherited_binding(
            deployment_receipt={
                "path": deployment_receipt_file["path"],
                "sha256": deployment_receipt_file["sha256"],
            },
            source_spec_authority={
                "path": source_spec_authority_file["path"],
                "sha256": source_spec_authority_file["sha256"],
            },
            deployment_receipt_digest=receipt["receipt_digest"],
            source_spec_authority_digest=source_spec_authority[
                "receipt_digest"
            ],
            target=args.target,
            capture_receipt_path=Path(args.capture_receipt),
        )
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {
                "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT",
                COMPLETION_ANCHOR_CHANNEL_ENV, COMPLETION_ANCHOR_SENT_ENV,
            }
            and not key.startswith(("LD_", "DYLD_"))
        }
        environment[WORK_ROOT_BINDING_ENV] = canonical_json_bytes(
            inherited_work_root
        ).decode("utf-8")
        work_root.capture(expected_entries=None)
        if args.target in DYNAMIC_ANCHOR_TARGETS:
            completion_anchor = run_target_with_completion_anchor(
                command,
                environment=environment,
                work_root=work_root,
                target=args.target,
                target_arguments=target_arguments,
            )
        else:
            os.set_inheritable(work_root.parent_fd, True)
            os.set_inheritable(work_root.root_fd, True)
            os.execve(command[0], command, environment)
    finally:
        for descriptor in (work_root.parent_fd, work_root.root_fd):
            try:
                os.set_inheritable(descriptor, False)
            except OSError:
                pass
        work_root.close()
    if completion_anchor is not None:
        print(
            canonical_json_bytes(completion_anchor).decode("utf-8"),
            flush=True,
        )
        return 0
    fail("verified target exec unexpectedly returned")
    return 70


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DecodedEvalDeploymentControllerError as error:
        print(f"decoded-eval-controller: {error}", file=sys.stderr)
        raise SystemExit(70)
