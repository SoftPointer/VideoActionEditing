#!/usr/bin/env python3
"""Retained-FD, create-only executor for one decoded-evaluation holder shard.

The executor has no scheduler, SSH, network, upload, retry, or loss-ranking
code.  A caller must explicitly invoke one local shard.  Every attempted task
is claimed once and retains its request, process logs, staging media, and a
terminal success/failure receipt.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import action_preservation_decoded_eval_plan_v1 as plan
import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_decoder_adapter_v1 as decoder_adapter
import action_preservation_decoded_eval_model_authority_v2 as model_authority


TASK_INPUT_SCHEMA = "bernini-action-preservation-decode-task-input-v3"
PROCESS_SCHEMA = "bernini-action-preservation-decode-process-v2"
TASK_OUTPUT_SCHEMA = "bernini-action-preservation-decode-task-output-v2"
TASK_FAILURE_SCHEMA = "bernini-action-preservation-decode-task-failure-v2"
SHARD_SUMMARY_SCHEMA = "bernini-action-preservation-decode-shard-summary-v2"
HOLDER_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-holder-completion-anchor-v1"
)
FD_INHERITANCE_SCHEMA = (
    "bernini-action-preservation-decoder-fd-inheritance-v2"
)

EXECUTION_DIRECTORY = "execution_shards"
INPUT_RECEIPT_FILENAME = "input_receipt.json"
STDOUT_FILENAME = "decoder.stdout"
STDERR_FILENAME = "decoder.stderr"
PROCESS_RECEIPT_FILENAME = "process_receipt.json"
STAGING_VIDEO_FILENAME = "candidate.staging.mp4"
OUTPUT_RECEIPT_FILENAME = "output_receipt.json"
FAILURE_RECEIPT_FILENAME = "failure_receipt.json"
SUMMARY_FILENAME = "shard_summary.json"
EXECUTION_CLAIM_FILENAME = "execution_claim.json"
DECODER_RUNTIME_CAPTURE_FILENAME = "decoder_verified_runtime_capture.json"
MODEL_CAPTURE_FILENAME = "model_consumption_capture.json"
MODEL_FINAL_FILENAME = "model_consumption_final.json"
CONSUMPTION_INPUT_FILENAME = "consumption_input.json"
ADAPTER_CAPTURE_FILENAME = "adapter_consumption_capture.json"
MODEL_PRE_USE_FILENAME = "model_consumption_pre_use.json"
MODEL_POST_USE_FILENAME = "model_consumption_post_use.json"
ADAPTER_PRE_USE_FILENAME = "adapter_consumption_pre_use.json"
ADAPTER_POST_USE_FILENAME = "adapter_consumption_post_use.json"
ADAPTER_FINAL_FILENAME = "adapter_consumption_final.json"
CONSUMPTION_CHAIN_FILENAME = "consumption_chain.json"
PUBLICATION_GATE_FILENAME = "consumption_publication_gate.json"
CONSUMPTION_AUTHORITY_DIRECTORY = "consumption_authority"
MODEL_FILE_UID = 2012
MODEL_FILE_GID = 2000
MODEL_FILE_DEVICE = 48
MODEL_FILE_MODE = 0o644
SUBPROCESS_ENV_DENYLIST = (
    "BASH_ENV",
    "ENV",
    "ZDOTDIR",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    # The holder-completion authority channel belongs exclusively to the
    # executor/controller continuation.  Decoder/torchrun children receive a
    # separate exact task-FD authority and must never learn or inherit either
    # the channel descriptor binding or its one-shot send state.
    "APV2_EVAL_COMPLETION_ANCHOR_CHANNEL",
    "APV2_EVAL_COMPLETION_ANCHOR_SENT_DIGEST",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class DecodedEvaluationExecutorError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return plan.canonical_json_bytes(value)
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    return plan.file_sha256(path)


def _hash_fd(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)
        offset += len(chunk)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_rdev,
        value.st_size,
        getattr(value, "st_blocks", 0),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


_PUBLISHED_INODE_IDENTITY_FIELDS = frozenset(
    {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
)


def _stat_identity_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "rdev": int(value.st_rdev),
        "size": int(value.st_size),
        "blocks": int(getattr(value, "st_blocks", 0)),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _published_inode_identity(value: Any) -> dict[str, int]:
    row = dict(
        _closed(
            value,
            _PUBLISHED_INODE_IDENTITY_FIELDS,
            label="published inode identity",
        )
    )
    if any(type(item) is not int or item < 0 for item in row.values()):
        raise DecodedEvaluationExecutorError(
            "published inode identity values differ"
        )
    if (
        not stat.S_ISREG(row["mode"])
        or stat.S_IMODE(row["mode"]) != 0o444
        or row["nlink"] != 2
        or row["size"] <= 0
    ):
        raise DecodedEvaluationExecutorError(
            "published inode identity topology differs"
        )
    return row


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise DecodedEvaluationExecutorError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationExecutorError(f"{label} is not a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DecodedEvaluationExecutorError(f"{label} is invalid")
    return value


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    payload = dict(value)
    payload.pop(field)
    if object_sha256(payload) != digest:
        raise DecodedEvaluationExecutorError(f"{label} digest differs")
    return digest


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationExecutorError(f"{label} does not exist") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationExecutorError(f"{label} is not a plain file")
    return path


def _plain_directory(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationExecutorError(f"{label} does not exist") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationExecutorError(f"{label} is not a plain directory")
    return path


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DecodedEvaluationExecutorError(
            "safe retained directory descriptors are unavailable"
        )
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )


def _directory_identity(value: os.stat_result) -> dict[str, int]:
    return _stat_identity_row(value)


def _directory_inode(value: os.stat_result | Mapping[str, int]) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        return tuple(
            int(value[field])
            for field in ("device", "inode", "uid", "gid", "rdev")
        )
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_uid),
        int(value.st_gid), int(value.st_rdev),
    )


def _directory_immutable_identity(
    value: os.stat_result | Mapping[str, int],
) -> tuple[int, ...]:
    """Return the directory fields that cannot change through child writes.

    A work-root parent legitimately changes size, link count, mtime and ctime as
    sibling phase roots are published.  Pinning those mutable fields would make
    an otherwise valid retained child unusable.  Device/inode/owner/full mode
    and rdev still prove that the named parent is the originally captured
    directory and reject rename-out/same-name replacement.
    """

    if isinstance(value, Mapping):
        return tuple(
            int(value[field])
            for field in ("device", "inode", "uid", "gid", "mode", "rdev")
        )
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_uid),
        int(value.st_gid), int(value.st_mode), int(value.st_rdev),
    )


class _HeldDirectory:
    """Retain one directory inode and perform all namespace I/O via openat.

    ``identity`` is advanced only after an exact entry replay around an
    authorized mutation.  Ordinary lookups never replace the retained FD with
    a freshly resolved pathname.
    """

    def __init__(
        self,
        *,
        path: Path,
        name: str,
        descriptor: int,
        parent_descriptor: int,
        identity: Mapping[str, int],
        entries: set[str],
        parent: "_HeldDirectory | None" = None,
        parent_path: Path | None = None,
        parent_identity: Mapping[str, int] | None = None,
        owns_parent_descriptor: bool = False,
    ) -> None:
        self.path = path
        self.name = name
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.identity = dict(identity)
        self.entries = set(entries)
        self.parent = parent
        self.parent_path = parent_path
        self.parent_identity = (
            None if parent_identity is None else dict(parent_identity)
        )
        self.owns_parent_descriptor = owns_parent_descriptor
        self.closed = False
        self.children: list[_HeldDirectory] = []
        if parent is not None:
            parent.children.append(self)

    @classmethod
    def open_root(
        cls,
        path: Path,
        *,
        expected_identity: Mapping[str, int] | None = None,
        expected_entries: set[str] | None = None,
        label: str,
    ) -> "_HeldDirectory":
        if (
            not path.is_absolute()
            or str(path) == os.path.sep
            or os.path.normpath(str(path)) != str(path)
            or path.name in {"", ".", ".."}
        ):
            raise DecodedEvaluationExecutorError(f"{label} path differs")
        parent_path = _plain_directory(path.parent, label=f"{label} parent")
        parent_descriptor = os.open(parent_path, _directory_flags())
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path.name, _directory_flags(), dir_fd=parent_descriptor
            )
            os.set_inheritable(parent_descriptor, False)
            os.set_inheritable(descriptor, False)
            parent = os.fstat(parent_descriptor)
            parent_named = parent_path.lstat()
            current = os.fstat(descriptor)
            named = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            entries = os.listdir(descriptor)
            identity = _directory_identity(current)
            if (
                not stat.S_ISDIR(current.st_mode)
                or _directory_identity(parent) != _directory_identity(parent_named)
                or identity != _directory_identity(named)
                or (
                    expected_identity is not None
                    and identity != dict(expected_identity)
                )
                or (
                    expected_entries is not None
                    and (
                        sorted(entries) != sorted(expected_entries)
                        or len(entries) != len(expected_entries)
                    )
                )
            ):
                raise DecodedEvaluationExecutorError(
                    f"{label} retained identity differs"
                )
            return cls(
                path=path,
                name=path.name,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                identity=identity,
                entries=set(entries),
                parent_path=parent_path,
                parent_identity=_directory_identity(parent),
                owns_parent_descriptor=True,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
            raise

    @classmethod
    def open_root_from_work_binding(
        cls,
        path: Path,
        *,
        work_root_binding: Mapping[str, Any],
        label: str,
    ) -> "_HeldDirectory":
        try:
            binding = bridge.verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        work_root_path = Path(binding["path"])
        if (
            not path.is_absolute()
            or path.parent != work_root_path
            or path.name in {"", ".", ".."}
        ):
            raise DecodedEvaluationExecutorError(
                f"{label} is outside the inherited work root"
            )
        parent_descriptor = os.dup(binding["root_fd"])
        descriptor: int | None = None
        try:
            os.set_inheritable(parent_descriptor, False)
            descriptor = os.open(
                path.name, _directory_flags(), dir_fd=parent_descriptor
            )
            os.set_inheritable(descriptor, False)
            parent = os.fstat(parent_descriptor)
            parent_named = os.stat(
                work_root_path.name,
                dir_fd=binding["parent_fd"],
                follow_symlinks=False,
            )
            current = os.fstat(descriptor)
            named = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            identity = _directory_identity(current)
            if (
                _directory_immutable_identity(parent)
                != _directory_immutable_identity(binding["root_identity"])
                or _directory_immutable_identity(parent_named)
                != _directory_immutable_identity(binding["root_identity"])
                or not stat.S_ISDIR(current.st_mode)
                or identity != _directory_identity(named)
            ):
                raise DecodedEvaluationExecutorError(
                    f"{label} inherited identity differs"
                )
            return cls(
                path=path,
                name=path.name,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                identity=identity,
                entries=set(os.listdir(descriptor)),
                parent_path=work_root_path,
                parent_identity=_directory_identity(parent),
                owns_parent_descriptor=True,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
            raise

    def _ensure_open(self) -> None:
        if self.closed:
            raise DecodedEvaluationExecutorError(
                "retained directory is closed"
            )

    def _named(self) -> os.stat_result:
        return os.stat(
            self.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )

    def replay(self, *, expected_entries: set[str] | None = None) -> None:
        self._ensure_open()
        expected = self.entries if expected_entries is None else expected_entries
        try:
            before = os.fstat(self.descriptor)
            first = os.listdir(self.descriptor)
            middle = os.fstat(self.descriptor)
            second = os.listdir(self.descriptor)
            after = os.fstat(self.descriptor)
            named = self._named()
            if self.parent is None:
                assert self.parent_path is not None
                assert self.parent_identity is not None
                parent = os.fstat(self.parent_descriptor)
                parent_named = self.parent_path.lstat()
                parent_expected = self.parent_identity
            else:
                parent = os.fstat(self.parent.descriptor)
                parent_named = self.parent._named()
                parent_expected = self.parent.identity
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                "retained directory replay failed"
            ) from error
        if (
            _directory_identity(before) != self.identity
            or _directory_identity(middle) != self.identity
            or _directory_identity(after) != self.identity
            or _directory_identity(named) != self.identity
            or (
                self.parent is None
                and (
                    _directory_immutable_identity(parent)
                    != _directory_immutable_identity(parent_expected)
                    or _directory_immutable_identity(parent_named)
                    != _directory_immutable_identity(parent_expected)
                )
            )
            or (
                self.parent is not None
                and (
                    _directory_identity(parent) != parent_expected
                    or _directory_identity(parent_named) != parent_expected
                )
            )
            or sorted(first) != sorted(second)
            or sorted(first) != sorted(expected)
            or len(first) != len(expected)
            or os.get_inheritable(self.descriptor)
        ):
            raise DecodedEvaluationExecutorError(
                "retained directory identity or entry closure differs"
            )

    def _refresh(self, entries: set[str]) -> None:
        try:
            before = os.fstat(self.descriptor)
            first = os.listdir(self.descriptor)
            middle = os.fstat(self.descriptor)
            second = os.listdir(self.descriptor)
            after = os.fstat(self.descriptor)
            named = self._named()
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                "retained directory mutation replay failed"
            ) from error
        identities = [
            _directory_identity(item)
            for item in (before, middle, after, named)
        ]
        if (
            any(
                _directory_inode(item) != _directory_inode(self.identity)
                for item in identities
            )
            or len({tuple(item.values()) for item in identities}) != 1
            or sorted(first) != sorted(second)
            or sorted(first) != sorted(entries)
            or len(first) != len(entries)
        ):
            raise DecodedEvaluationExecutorError(
                "retained directory mutation escaped its inode"
            )
        self.identity = identities[0]
        self.entries = set(entries)
        if self.parent is not None:
            self.parent._refresh(self.parent.entries)

    def create_child(self, name: str, *, label: str) -> "_HeldDirectory":
        _identifier(name, label=f"{label} basename")
        self.replay()
        try:
            os.mkdir(name, 0o700, dir_fd=self.descriptor)
        except FileExistsError as error:
            raise DecodedEvaluationExecutorError(
                f"{label} already exists; retries are forbidden: {self.path / name}"
            ) from error
        descriptor = os.open(name, _directory_flags(), dir_fd=self.descriptor)
        try:
            os.set_inheritable(descriptor, False)
            current = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or stat.S_IMODE(current.st_mode) != 0o700
                or _directory_identity(current) != _directory_identity(named)
                or os.listdir(descriptor)
            ):
                raise DecodedEvaluationExecutorError(
                    f"{label} fresh identity differs"
                )
            os.fsync(self.descriptor)
            self._refresh(self.entries | {name})
            return _HeldDirectory(
                path=self.path / name,
                name=name,
                descriptor=descriptor,
                parent_descriptor=self.descriptor,
                identity=_directory_identity(current),
                entries=set(),
                parent=self,
            )
        except Exception:
            os.close(descriptor)
            raise

    def open_child(
        self,
        name: str,
        *,
        label: str,
        expected_identity: Mapping[str, int] | None = None,
    ) -> "_HeldDirectory":
        _identifier(name, label=f"{label} basename")
        self.replay()
        descriptor = os.open(name, _directory_flags(), dir_fd=self.descriptor)
        try:
            os.set_inheritable(descriptor, False)
            current = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
            identity = _directory_identity(current)
            if (
                not stat.S_ISDIR(current.st_mode)
                or identity != _directory_identity(named)
                or (
                    expected_identity is not None
                    and identity != dict(expected_identity)
                )
            ):
                raise DecodedEvaluationExecutorError(
                    f"{label} retained identity differs"
                )
            return _HeldDirectory(
                path=self.path / name,
                name=name,
                descriptor=descriptor,
                parent_descriptor=self.descriptor,
                identity=identity,
                entries=set(os.listdir(descriptor)),
                parent=self,
            )
        except Exception:
            os.close(descriptor)
            raise

    def write(self, name: str, payload: bytes, *, mode: int = 0o400) -> Path:
        _identifier(name, label="create-only artifact basename")
        self.replay()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, mode, dir_fd=self.descriptor)
        except FileExistsError as error:
            raise DecodedEvaluationExecutorError(
                f"refusing to overwrite create-only artifact: {self.path / name}"
            ) from error
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise DecodedEvaluationExecutorError(
                        "create-only write made no progress"
                    )
                offset += written
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            before = os.fstat(descriptor)
            digest, size = _hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != mode
                or _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(named)
                or digest != bytes_sha256(payload)
                or size != len(payload)
            ):
                raise DecodedEvaluationExecutorError(
                    "create-only artifact same-FD replay differs"
                )
        finally:
            os.close(descriptor)
        os.fsync(self.descriptor)
        self._refresh(self.entries | {name})
        return self.path / name

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        return self.write(name, canonical_json_bytes(value) + b"\n")

    def adopt_entries(self, entries: set[str]) -> None:
        """Authorize an exact externally-created entry transition.

        The only intended callers are retained-FD consumers (the decoder and
        model-authority implementation) that were explicitly given this
        directory descriptor.  ``_refresh`` still proves the directory inode,
        named binding and exact resulting closure before advancing the anchor.
        """

        self._ensure_open()
        if any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
            for name in entries
        ):
            raise DecodedEvaluationExecutorError(
                "retained directory adopted entry closure differs"
            )
        self._refresh(set(entries))

    def exists(self, name: str) -> bool:
        _identifier(name, label="retained entry basename")
        self.replay()
        try:
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def stat_file(
        self,
        name: str,
        *,
        label: str,
        expected_nlink: int | None = None,
        expected_mode: int | None = None,
    ) -> os.stat_result:
        _identifier(name, label=f"{label} basename")
        self.replay()
        try:
            observed = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                f"cannot stat {label}: {error}"
            ) from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or (
                expected_nlink is not None
                and observed.st_nlink != expected_nlink
            )
            or (
                expected_mode is not None
                and stat.S_IMODE(observed.st_mode) != expected_mode
            )
        ):
            raise DecodedEvaluationExecutorError(
                f"{label} physical topology differs"
            )
        return observed

    def stable_file(
        self,
        name: str,
        *,
        label: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
        expected_identity: Mapping[str, Any] | None = None,
        expected_nlink: int | None = None,
        expected_mode: int | None = None,
    ) -> tuple[dict[str, int], str, int]:
        """Double-hash one named regular file through a retained descriptor."""

        _identifier(name, label=f"{label} basename")
        self.replay()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=self.descriptor)
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                f"cannot open {label}: {error}"
            ) from error
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = _hash_fd(descriptor)
            middle = os.fstat(descriptor)
            second_sha, second_size = _hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
        finally:
            os.close(descriptor)
        identity = _stat_identity_row(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first_sha != second_sha
            or first_size != second_size
            or (
                expected_sha256 is not None
                and first_sha != expected_sha256
            )
            or (
                expected_size is not None
                and first_size != expected_size
            )
            or (
                expected_identity is not None
                and identity != dict(expected_identity)
            )
            or (
                expected_nlink is not None
                and before.st_nlink != expected_nlink
            )
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            raise DecodedEvaluationExecutorError(
                f"{label} same-FD replay differs"
            )
        self.replay()
        return identity, first_sha, first_size

    def consumer_path(self, name: str, *, production_mode: bool) -> Path:
        """Return a path that resolves through this retained FD in production."""

        _identifier(name, label="retained consumer basename")
        if production_mode:
            if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
                raise DecodedEvaluationExecutorError(
                    "retained directory consumer paths require Linux procfs"
                )
            return Path(f"/proc/self/fd/{self.descriptor}/{name}")
        return self.path / name

    def read(
        self,
        name: str,
        *,
        label: str,
        expected_sha256: str | None = None,
    ) -> tuple[bytes, dict[str, int]]:
        _identifier(name, label=f"{label} basename")
        self.replay()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=self.descriptor)
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                f"cannot open {label}: {error}"
            ) from error
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = _hash_fd(descriptor)
            first = os.pread(descriptor, first_size, 0)
            middle = os.fstat(descriptor)
            second_sha, second_size = _hash_fd(descriptor)
            second = os.pread(descriptor, second_size, 0)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first != second
            or first_sha != second_sha
            or first_size != second_size
            or len(first) != first_size
            or (expected_sha256 is not None and first_sha != expected_sha256)
        ):
            raise DecodedEvaluationExecutorError(
                f"{label} same-FD replay differs"
            )
        self.replay()
        return first, _stat_identity_row(before)

    def authority_row(self) -> dict[str, Any]:
        self.replay()
        row: dict[str, Any] = {
            "path": str(self.path),
            "fd": self.descriptor,
            "identity": dict(self.identity),
            "role": "publication_root",
        }
        row["authority_digest"] = object_sha256(row)
        return row

    def close(self) -> None:
        if self.closed:
            return
        for child in reversed(self.children):
            child.close()
        self.closed = True
        os.close(self.descriptor)
        if self.owns_parent_descriptor:
            os.close(self.parent_descriptor)


class _HeldCompletionReservation:
    """Retain and fill this holder's pre-authorized completion inode."""

    def __init__(
        self,
        *,
        holder_job_id: str,
        relative_path: str,
        descriptor: int,
        parent: _HeldDirectory,
        initial_identity: Mapping[str, int],
    ) -> None:
        self.holder_job_id = holder_job_id
        self.relative_path = relative_path
        self.descriptor = descriptor
        self.parent = parent
        self.initial_identity = dict(initial_identity)
        self.closed = False
        self.filled = False
        self.final_binding: dict[str, Any] | None = None

    @classmethod
    def capture(
        cls,
        *,
        bundle: Mapping[str, Any],
        holder_job_id: str,
        execution_parent: _HeldDirectory,
    ) -> "_HeldCompletionReservation":
        try:
            rows = plan.validate_holder_completion_reservations(
                bundle["publication_receipt"][
                    "holder_completion_reservations"
                ],
                evaluation_root=bundle["manifest"]["evaluation_root"],
                materialized_required=True,
                directory_authority=bundle["directory_authority"],
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        matches = [
            row for row in rows if row["holder_job_id"] == holder_job_id
        ]
        if len(matches) != 1:
            raise DecodedEvaluationExecutorError(
                "holder completion reservation differs"
            )
        row = matches[0]
        relative = plan.holder_completion_reservation_relative(holder_job_id)
        relative_path = Path(relative)
        if (
            row["relative_path"] != relative
            or relative_path.parent.as_posix()
            != plan.EXECUTION_SHARD_DIRECTORY
            or row["path"]
            != str(Path(bundle["manifest"]["evaluation_root"]) / relative)
            or _stat_identity_row(os.fstat(execution_parent.descriptor))
            != row["parent_identity"]
        ):
            raise DecodedEvaluationExecutorError(
                "holder completion reservation parent differs"
            )
        execution_parent.replay()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                relative_path.name,
                os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=execution_parent.descriptor,
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = _hash_fd(descriptor)
            middle = os.fstat(descriptor)
            second_sha, second_size = _hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                relative_path.name,
                dir_fd=execution_parent.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode)
                != plan.HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
                or _stat_identity_row(before) != row["identity"]
                or _stat_identity(before) != _stat_identity(middle)
                or _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(named)
                or first_sha != row["sha256"]
                or second_sha != row["sha256"]
                or first_size != 0
                or second_size != 0
            ):
                raise DecodedEvaluationExecutorError(
                    "holder completion reservation same-FD replay differs"
                )
            result = cls(
                holder_job_id=holder_job_id,
                relative_path=relative,
                descriptor=descriptor,
                parent=execution_parent,
                initial_identity=row["identity"],
            )
            descriptor = None
            return result
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def fill(
        self,
        completion: Mapping[str, Any],
        *,
        topology: Sequence[Mapping[str, Any]],
        base_directory_authority: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.closed or self.filled:
            raise DecodedEvaluationExecutorError(
                "holder completion reservation is not fresh"
            )
        try:
            validated = plan.validate_holder_directory_completion(
                completion,
                topology=topology,
                base_directory_authority=base_directory_authority,
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if validated["holder_job_id"] != self.holder_job_id:
            raise DecodedEvaluationExecutorError(
                "holder completion ownership differs"
            )
        payload = canonical_json_bytes(validated) + b"\n"
        before_empty = os.fstat(self.descriptor)
        if (
            _stat_identity_row(before_empty) != self.initial_identity
            or before_empty.st_size != 0
            or before_empty.st_nlink != 1
        ):
            raise DecodedEvaluationExecutorError(
                "holder completion reservation changed before fill"
            )
        offset = 0
        while offset < len(payload):
            count = os.write(self.descriptor, payload[offset:])
            if count <= 0:
                raise DecodedEvaluationExecutorError(
                    "holder completion fill made no progress"
                )
            offset += count
        os.fchmod(
            self.descriptor,
            plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE,
        )
        os.fsync(self.descriptor)
        before = os.fstat(self.descriptor)
        first_sha, first_size = _hash_fd(self.descriptor)
        middle = os.fstat(self.descriptor)
        second_sha, second_size = _hash_fd(self.descriptor)
        after = os.fstat(self.descriptor)
        basename = Path(self.relative_path).name
        named = os.stat(
            basename,
            dir_fd=self.parent.descriptor,
            follow_symlinks=False,
        )
        self.parent.replay()
        expected_sha = bytes_sha256(payload)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode)
            != plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE
            or _directory_inode(before) != _directory_inode(
                self.initial_identity
            )
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first_sha != expected_sha
            or second_sha != expected_sha
            or first_size != len(payload)
            or second_size != len(payload)
        ):
            raise DecodedEvaluationExecutorError(
                "holder completion filled inode replay differs"
            )
        self.filled = True
        self.final_binding = {
            "path": str(self.parent.path / basename),
            "sha256": expected_sha,
            "size": len(payload),
            **_stat_identity_row(before),
        }
        return dict(self.final_binding)

    def replay_final(self) -> dict[str, Any]:
        if self.closed or not self.filled or self.final_binding is None:
            raise DecodedEvaluationExecutorError(
                "holder completion final authority is unavailable"
            )
        before = os.fstat(self.descriptor)
        first_sha, first_size = _hash_fd(self.descriptor)
        middle = os.fstat(self.descriptor)
        second_sha, second_size = _hash_fd(self.descriptor)
        after = os.fstat(self.descriptor)
        named = os.stat(
            Path(self.relative_path).name,
            dir_fd=self.parent.descriptor,
            follow_symlinks=False,
        )
        observed = {
            "path": self.final_binding["path"],
            "sha256": first_sha,
            "size": first_size,
            **_stat_identity_row(before),
        }
        if (
            observed != self.final_binding
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first_sha != second_sha
            or first_size != second_size
        ):
            raise DecodedEvaluationExecutorError(
                "holder completion final same-FD replay differs"
            )
        return dict(observed)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.descriptor)


class _RetainedTaskMedia:
    """Retain the exact decoder output inode through probe and publication."""

    def __init__(
        self,
        *,
        directory: _HeldDirectory,
        name: str,
        descriptor: int,
        identity: Mapping[str, int],
        sha256: str,
        size: int,
    ) -> None:
        self.directory = directory
        self.name = name
        self.descriptor = descriptor
        self.identity = dict(identity)
        self.sha256 = sha256
        self.size = size
        self.closed = False

    @classmethod
    def capture(
        cls, directory: _HeldDirectory, *, name: str, label: str,
    ) -> "_RetainedTaskMedia":
        _identifier(name, label=f"{label} basename")
        directory.replay()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory.descriptor,
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = _hash_fd(descriptor)
            middle = os.fstat(descriptor)
            second_sha, second_size = _hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=directory.descriptor, follow_symlinks=False
            )
            identity = _stat_identity_row(before)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or first_size <= 0
                or identity != _stat_identity_row(middle)
                or identity != _stat_identity_row(after)
                or identity != _stat_identity_row(named)
                or first_sha != second_sha
                or first_size != second_size
            ):
                raise DecodedEvaluationExecutorError(
                    f"{label} retained-FD capture differs"
                )
            retained = cls(
                directory=directory,
                name=name,
                descriptor=descriptor,
                identity=identity,
                sha256=first_sha,
                size=first_size,
            )
            descriptor = None
            return retained
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def replay(self, *, rehash: bool) -> None:
        if self.closed:
            raise DecodedEvaluationExecutorError(
                "retained task media is closed"
            )
        before = os.fstat(self.descriptor)
        named = os.stat(
            self.name,
            dir_fd=self.directory.descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_identity_row(before) != self.identity
            or _stat_identity_row(named) != self.identity
            or os.get_inheritable(self.descriptor)
        ):
            raise DecodedEvaluationExecutorError(
                "retained task media inode/name replay differs"
            )
        if rehash:
            first_sha, first_size = _hash_fd(self.descriptor)
            middle = os.fstat(self.descriptor)
            second_sha, second_size = _hash_fd(self.descriptor)
            after = os.fstat(self.descriptor)
            named_after = os.stat(
                self.name,
                dir_fd=self.directory.descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_identity_row(middle) != self.identity
                or _stat_identity_row(after) != self.identity
                or _stat_identity_row(named_after) != self.identity
                or first_sha != self.sha256
                or second_sha != self.sha256
                or first_size != self.size
                or second_size != self.size
            ):
                raise DecodedEvaluationExecutorError(
                    "retained task media byte replay differs"
                )

    def consumer_path(self, *, production_mode: bool) -> Path:
        if not production_mode:
            return self.directory.path / self.name
        if (
            self.closed
            or sys.platform != "linux"
            or not Path("/proc/self/fd").is_dir()
        ):
            raise DecodedEvaluationExecutorError(
                "retained task media proc-FD is unavailable"
            )
        return Path(f"/proc/self/fd/{self.descriptor}")

    def replay_published(
        self,
        *,
        published_identity: Mapping[str, Any],
        final_directory: _HeldDirectory,
        final_name: str,
    ) -> None:
        expected = _published_inode_identity(published_identity)
        first_sha, first_size = _hash_fd(self.descriptor)
        middle = os.fstat(self.descriptor)
        second_sha, second_size = _hash_fd(self.descriptor)
        after = os.fstat(self.descriptor)
        named_staging = os.stat(
            self.name,
            dir_fd=self.directory.descriptor,
            follow_symlinks=False,
        )
        named_final = os.stat(
            final_name,
            dir_fd=final_directory.descriptor,
            follow_symlinks=False,
        )
        immutable = ("device", "inode", "uid", "gid", "rdev")
        if (
            any(
                _stat_identity_row(item) != expected
                for item in (middle, after, named_staging, named_final)
            )
            or any(expected[field] != self.identity[field] for field in immutable)
            or expected["nlink"] != 2
            or stat.S_IMODE(expected["mode"]) != 0o444
            or first_sha != self.sha256
            or second_sha != self.sha256
            or first_size != self.size
            or second_size != self.size
        ):
            raise DecodedEvaluationExecutorError(
                "retained task media publication replay differs"
            )
        self.identity = dict(expected)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.descriptor)


def validate_holder_completion_anchor(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "holder_job_id", "completion_path",
        "initial_inode_identity", "completion_sha256", "completion_size",
        "completion_mode", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    }
    row = dict(_closed(value, fields, label="holder completion anchor"))
    initial_fields = {"device", "inode", "uid", "gid", "rdev"}
    initial = dict(
        _closed(
            row["initial_inode_identity"],
            initial_fields,
            label="holder completion initial inode",
        )
    )
    expected_relative = plan.holder_completion_reservation_relative(
        row["holder_job_id"]
    )
    if (
        row["schema_version"] != HOLDER_COMPLETION_ANCHOR_SCHEMA
        or row["holder_job_id"]
        not in {holder["job_id"] for holder in plan.HOLDER_ROWS}
        or not isinstance(row["completion_path"], str)
        or not Path(row["completion_path"]).is_absolute()
        or not row["completion_path"].endswith("/" + expected_relative)
        or any(type(initial[field]) is not int or initial[field] < 0
               for field in initial_fields)
        or type(row["completion_size"]) is not int
        or row["completion_size"] <= 0
        or row["completion_mode"]
        != plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE
    ):
        raise DecodedEvaluationExecutorError(
            "holder completion anchor binding differs"
        )
    for field in (
        "completion_sha256", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    ):
        _sha(row[field], label=f"holder completion anchor {field}")
    _verify_digest(
        row, field="anchor_digest", label="holder completion anchor"
    )
    return row


def build_holder_completion_anchor(
    *,
    holder_job_id: str,
    reservation: _HeldCompletionReservation,
    completion: Mapping[str, Any],
    holder_summary_digest: str,
) -> dict[str, Any]:
    final = reservation.replay_final()
    initial = reservation.initial_identity
    value: dict[str, Any] = {
        "schema_version": HOLDER_COMPLETION_ANCHOR_SCHEMA,
        "holder_job_id": holder_job_id,
        "completion_path": final["path"],
        "initial_inode_identity": {
            field: initial[field]
            for field in ("device", "inode", "uid", "gid", "rdev")
        },
        "completion_sha256": final["sha256"],
        "completion_size": final["size"],
        "completion_mode": stat.S_IMODE(final["mode"]),
        "completion_digest": completion["completion_digest"],
        "holder_summary_digest": holder_summary_digest,
    }
    value["anchor_digest"] = object_sha256(value)
    return validate_holder_completion_anchor(value)


def _publish_gated_staging_inode(
    *,
    staging_path: Path,
    final_path: Path,
    publication_gate: Mapping[str, Any],
    production_mode: bool,
    staging_directory: _HeldDirectory | None = None,
    final_directory: _HeldDirectory | None = None,
) -> dict[str, int]:
    """Publish only the exact inode/bytes authenticated by the post-use gate.

    On Linux production hosts the hard link is made through the retained
    descriptor, so a rename/swap of the staging pathname cannot redirect the
    publication.  The injected test backend retains the same before/after
    identity checks while using the portable named-link fallback.
    """

    if (staging_directory is None) != (final_directory is None):
        raise DecodedEvaluationExecutorError(
            "publication retained-directory closure differs"
        )
    retained_directories = staging_directory is not None
    if production_mode and not retained_directories:
        raise DecodedEvaluationExecutorError(
            "production publication lacks retained directory authority"
        )
    source_name = staging_path.name
    destination_name = final_path.name
    _identifier(source_name, label="staging output basename")
    _identifier(destination_name, label="published output basename")
    if staging_directory is not None:
        staging_directory.replay()
        assert final_directory is not None
        final_directory.replay()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(
        source_name if staging_directory is not None else staging_path,
        flags,
        dir_fd=(
            staging_directory.descriptor
            if staging_directory is not None
            else None
        ),
    )
    parent_descriptor: int | None = None
    close_parent_descriptor = False
    linked = False
    try:
        if final_directory is not None:
            parent_descriptor = final_directory.descriptor
        else:
            parent_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                parent_flags |= os.O_NOFOLLOW
            parent_descriptor = os.open(final_path.parent, parent_flags)
            close_parent_descriptor = True
            os.set_inheritable(parent_descriptor, False)
        parent_before = os.fstat(parent_descriptor)
        parent_named_before = (
            final_directory._named()
            if final_directory is not None
            else final_path.parent.lstat()
        )
        parent_anchor = (
            int(parent_before.st_dev), int(parent_before.st_ino),
            int(parent_before.st_uid), int(parent_before.st_gid),
            int(parent_before.st_mode), int(parent_before.st_rdev),
        )
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or stat.S_ISLNK(parent_named_before.st_mode)
            or parent_anchor
            != (
                int(parent_named_before.st_dev), int(parent_named_before.st_ino),
                int(parent_named_before.st_uid), int(parent_named_before.st_gid),
                int(parent_named_before.st_mode), int(parent_named_before.st_rdev),
            )
        ):
            raise DecodedEvaluationExecutorError(
                "published output parent directory identity differs"
            )
        before = os.fstat(descriptor)
        first_sha, first_size = _hash_fd(descriptor)
        middle = os.fstat(descriptor)
        second_sha, second_size = _hash_fd(descriptor)
        after = os.fstat(descriptor)
        named = (
            os.stat(
                source_name,
                dir_fd=staging_directory.descriptor,
                follow_symlinks=False,
            )
            if staging_directory is not None
            else staging_path.lstat()
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first_sha != publication_gate.get("staging_sha256")
            or second_sha != publication_gate.get("staging_sha256")
            or first_size != publication_gate.get("staging_size")
            or second_size != publication_gate.get("staging_size")
        ):
            raise DecodedEvaluationExecutorError(
                "staging inode/bytes changed after the publication gate"
            )
        os.fchmod(descriptor, 0o444)
        if production_mode:
            if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
                raise DecodedEvaluationExecutorError(
                    "held-FD publication is unavailable on the production host"
                )
            os.link(
                f"/proc/self/fd/{descriptor}",
                destination_name,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=True,
            )
        else:
            if staging_directory is not None:
                os.link(
                    source_name,
                    destination_name,
                    src_dir_fd=staging_directory.descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            else:
                os.link(
                    staging_path,
                    destination_name,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
        linked = True
        if final_directory is not None:
            final_directory.adopt_entries(
                final_directory.entries | {destination_name}
            )
        retained = os.fstat(descriptor)
        published = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published_named = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        final_sha, final_size = _hash_fd(descriptor)
        retained_after_hash = os.fstat(descriptor)
        parent_after = os.fstat(parent_descriptor)
        parent_named_after = (
            final_directory._named()
            if final_directory is not None
            else final_path.parent.lstat()
        )
        if (
            (retained.st_dev, retained.st_ino)
            != (published.st_dev, published.st_ino)
            or _stat_identity(retained) != _stat_identity(published)
            or _stat_identity(retained) != _stat_identity(published_named)
            or _stat_identity(retained) != _stat_identity(retained_after_hash)
            or retained.st_nlink != 2
            or published.st_nlink != 2
            or stat.S_IMODE(retained.st_mode) != 0o444
            or stat.S_IMODE(published.st_mode) != 0o444
            or parent_anchor
            != (
                int(parent_after.st_dev), int(parent_after.st_ino),
                int(parent_after.st_uid), int(parent_after.st_gid),
                int(parent_after.st_mode), int(parent_after.st_rdev),
            )
            or parent_anchor
            != (
                int(parent_named_after.st_dev), int(parent_named_after.st_ino),
                int(parent_named_after.st_uid), int(parent_named_after.st_gid),
                int(parent_named_after.st_mode), int(parent_named_after.st_rdev),
            )
            or final_sha != publication_gate.get("staging_sha256")
            or final_size != publication_gate.get("staging_size")
        ):
            raise DecodedEvaluationExecutorError(
                "published output differs from its held staging inode"
            )
        return _published_inode_identity(_stat_identity_row(retained))
    except Exception:
        if linked:
            try:
                retained = os.fstat(descriptor)
                if parent_descriptor is None:
                    raise FileNotFoundError
                named_final = os.stat(
                    destination_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (retained.st_dev, retained.st_ino) == (
                    named_final.st_dev,
                    named_final.st_ino,
                ):
                    os.unlink(destination_name, dir_fd=parent_descriptor)
                    if final_directory is not None:
                        final_directory.adopt_entries(
                            final_directory.entries - {destination_name}
                        )
            except FileNotFoundError:
                pass
        raise
    finally:
        if parent_descriptor is not None and close_parent_descriptor:
            os.close(parent_descriptor)
        os.close(descriptor)


def _stable_published_file(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_sha256: str,
    expected_size: int,
    label: str,
    directory: _HeldDirectory | None = None,
) -> tuple[dict[str, int], str, int]:
    """Replay a published file's identity and bytes through one held FD.

    The named path is checked only as a binding to the already-open descriptor;
    the hashes are never computed by reopening the name.  This prevents a
    path replacement between ``lstat`` and hashing from combining identity
    evidence from one inode with bytes from another.
    """

    publication_identity = _published_inode_identity(expected_identity)
    expected_sha256 = _sha(expected_sha256, label=f"{label} expected SHA")
    if type(expected_size) is not int or expected_size <= 0:
        raise DecodedEvaluationExecutorError(
            f"{label} expected size differs"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            path.name if directory is not None else path,
            flags,
            dir_fd=directory.descriptor if directory is not None else None,
        )
    except OSError as error:
        raise DecodedEvaluationExecutorError(
            f"cannot open {label}: {error}"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first_sha, first_size = _hash_fd(descriptor)
        middle = os.fstat(descriptor)
        second_sha, second_size = _hash_fd(descriptor)
        after = os.fstat(descriptor)
        try:
            named = (
                os.stat(
                    path.name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                if directory is not None
                else path.lstat()
            )
        except OSError as error:
            raise DecodedEvaluationExecutorError(
                f"cannot replay named {label}: {error}"
            ) from error
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _stat_identity_row(before) != publication_identity
        or _stat_identity_row(middle) != publication_identity
        or _stat_identity_row(after) != publication_identity
        or _stat_identity_row(named) != publication_identity
        or first_sha != expected_sha256
        or second_sha != expected_sha256
        or first_size != expected_size
        or second_size != expected_size
    ):
        raise DecodedEvaluationExecutorError(
            f"{label} bytes/inode differ from the post-use publication gate"
        )
    if directory is not None:
        directory.replay()
    return publication_identity, first_sha, first_size


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw, _ = bridge._stable_file(path, label=label)
        value = json.loads(raw.decode("utf-8"))
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationExecutorError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise DecodedEvaluationExecutorError(f"{label} root is not an object")
    return dict(value)


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationExecutorError(
            f"cannot load {label}: {error}"
        ) from error
    if (
        not isinstance(value, Mapping)
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise DecodedEvaluationExecutorError(
            f"{label} is not canonical JSON"
        )
    return dict(value)


def _write_create_only(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    _plain_directory(path.parent, label="create-only parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise DecodedEvaluationExecutorError(
            f"refusing to overwrite create-only artifact: {path}"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise DecodedEvaluationExecutorError(
                    "create-only write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise DecodedEvaluationExecutorError("create-only reread differs")


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_create_only(path, canonical_json_bytes(value) + b"\n")


def _ensure_directory_tree(root: Path, relative_parent: Path) -> Path:
    if relative_parent.is_absolute() or ".." in relative_parent.parts:
        raise DecodedEvaluationExecutorError("relative output parent escapes root")
    current = _plain_directory(root, label="evaluation root")
    for component in relative_parent.parts:
        if component in ("", "."):
            continue
        current = current / component
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            pass
        _plain_directory(current, label="output directory")
    return current


def _claim_directory(path: Path, *, label: str) -> Path:
    _plain_directory(path.parent, label=f"{label} parent")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as error:
        raise DecodedEvaluationExecutorError(
            f"{label} already exists; retries are forbidden: {path}"
        ) from error
    return _plain_directory(path, label=label)


def load_published_bundle(
    evaluation_root: str | Path,
    *,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(evaluation_root)
    if not root.is_absolute() or str(root) == os.path.sep:
        raise DecodedEvaluationExecutorError("evaluation root must be absolute and non-root")
    held_root = (
        _HeldDirectory.open_root(root, label="evaluation root")
        if work_root_binding is None
        else _HeldDirectory.open_root_from_work_binding(
            root,
            work_root_binding=work_root_binding,
            label="evaluation root",
        )
    )
    shard_directory: _HeldDirectory | None = None
    try:
        direct_names = {
            plan.INPUT_FILENAME: "evaluation input",
            plan.REVIEW_CONTRACT_FILENAME: "review packet contract",
            plan.MANIFEST_FILENAME: "evaluation manifest",
            plan.PUBLICATION_FILENAME: "publication receipt",
            plan.DIRECTORY_AUTHORITY_FILENAME: "directory authority",
        }
        direct_raw = {
            name: held_root.read(name, label=label)[0]
            for name, label in direct_names.items()
        }
        input_spec = _decode_canonical_json(
            direct_raw[plan.INPUT_FILENAME], label="evaluation input"
        )
        review = _decode_canonical_json(
            direct_raw[plan.REVIEW_CONTRACT_FILENAME],
            label="review packet contract",
        )
        manifest = _decode_canonical_json(
            direct_raw[plan.MANIFEST_FILENAME], label="evaluation manifest"
        )
        receipt = _decode_canonical_json(
            direct_raw[plan.PUBLICATION_FILENAME], label="publication receipt"
        )
        directory_authority = _decode_canonical_json(
            direct_raw[plan.DIRECTORY_AUTHORITY_FILENAME],
            label="directory authority",
        )
        topology = plan.build_directory_topology(
            manifest, input_spec=input_spec
        )
        try:
            directory_authority = plan.validate_directory_authority(
                directory_authority,
                topology=topology,
                materialized_required=True,
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        authority_rows = {
            row["relative_path"]: dict(row)
            for row in directory_authority["rows"]
        }
        root_row = authority_rows.get(".")
        if (
            root_row is None
            or held_root.identity != root_row["identity"]
            or held_root.entries != set(root_row["expected_entries"])
            or _directory_inode(held_root.parent_identity)
            != _directory_inode(root_row["parent_identity"])
        ):
            raise DecodedEvaluationExecutorError(
                "evaluation root differs from its materialized authority"
            )
        held_root.replay(expected_entries=set(root_row["expected_entries"]))
        shard_row = authority_rows.get(plan.SHARD_DIRECTORY)
        if shard_row is None:
            raise DecodedEvaluationExecutorError(
                "shard directory authority is absent"
            )
        shard_directory = held_root.open_child(
            plan.SHARD_DIRECTORY,
            label="published shard directory",
            expected_identity=shard_row["identity"],
        )
        shard_directory.replay(
            expected_entries=set(shard_row["expected_entries"])
        )
        shards: dict[str, dict[str, Any]] = {}
        shard_raw: dict[str, bytes] = {}
        for holder in plan.HOLDER_ROWS:
            filename = f"{holder['job_id']}.json"
            raw, _identity = shard_directory.read(
                filename, label=f"holder {holder['job_id']} shard"
            )
            shard_raw[f"{plan.SHARD_DIRECTORY}/{filename}"] = raw
            shards[holder["job_id"]] = _decode_canonical_json(
                raw, label=f"holder {holder['job_id']} shard"
            )
        bundle = {
            "input_spec": input_spec,
            "review_contract": review,
            "manifest": manifest,
            "shards": shards,
        }
        try:
            validated_receipt = plan.validate_publication_receipt(
                receipt,
                bundle=bundle,
                directory_authority=directory_authority,
                verify_directory_authority=True,
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        observed_raw = {
            plan.INPUT_FILENAME: direct_raw[plan.INPUT_FILENAME],
            plan.MANIFEST_FILENAME: direct_raw[plan.MANIFEST_FILENAME],
            plan.REVIEW_CONTRACT_FILENAME: direct_raw[
                plan.REVIEW_CONTRACT_FILENAME
            ],
            plan.DIRECTORY_AUTHORITY_FILENAME: direct_raw[
                plan.DIRECTORY_AUTHORITY_FILENAME
            ],
            **shard_raw,
        }
        for item in validated_receipt["files"]:
            raw = observed_raw.get(item["relpath"])
            if raw is None or bytes_sha256(raw) != item["sha256"]:
                raise DecodedEvaluationExecutorError(
                    f"published bundle file hash differs: {item['relpath']}"
                )
        if manifest["evaluation_root"] != str(root):
            raise DecodedEvaluationExecutorError(
                "manifest evaluation root binding differs"
            )
        shard_directory.close()
        shard_directory = None
        return {
            **bundle,
            "publication_receipt": validated_receipt,
            "directory_authority": directory_authority,
            "_evaluation_root_handle": held_root,
            "_work_root_binding": (
                None if work_root_binding is None else dict(work_root_binding)
            ),
        }
    except Exception:
        if shard_directory is not None:
            shard_directory.close()
        held_root.close()
        raise


def _holder_directory_handles(
    bundle: Mapping[str, Any], *, holder_job_id: str
) -> dict[str, _HeldDirectory]:
    root = bundle.get("_evaluation_root_handle")
    authority = bundle.get("directory_authority")
    if not isinstance(root, _HeldDirectory) or not isinstance(authority, Mapping):
        raise DecodedEvaluationExecutorError(
            "materialized evaluation directory authority is absent"
        )
    rows = {
        item["relative_path"]: dict(item)
        for item in authority.get("rows", [])
        if isinstance(item, Mapping)
    }
    root_row = rows.get(".")
    if root_row is None:
        raise DecodedEvaluationExecutorError(
            "evaluation root directory authority is absent"
        )
    root.replay(expected_entries=set(root_row["expected_entries"]))
    required = {
        plan.EXECUTION_SHARD_DIRECTORY,
        f"{plan.EXECUTION_SHARD_DIRECTORY}/{holder_job_id}",
    }
    shard = bundle["shards"][holder_job_id]
    holder_root = f"{plan.EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
    task_parent_relative = f"{holder_root}/tasks"
    required.update(
        {
            f"{holder_root}/{CONSUMPTION_AUTHORITY_DIRECTORY}",
            task_parent_relative,
        }
    )
    for task in shard["tasks"]:
        required.add(f"{task_parent_relative}/{_task_id(task)}")
        parent = Path(task["record"]["output_relpath"]).parent
        current = Path()
        for component in parent.parts:
            current /= component
            required.add(current.as_posix())
    expanded = set(required)
    for relative in list(required):
        current = Path(relative)
        while current.parent != Path("."):
            current = current.parent
            expanded.add(current.as_posix())
    handles: dict[str, _HeldDirectory] = {".": root}
    try:
        for relative in sorted(expanded, key=lambda item: (item.count("/"), item)):
            row = rows.get(relative)
            if row is None:
                raise DecodedEvaluationExecutorError(
                    f"directory authority row is absent: {relative}"
                )
            owner = row["owner_holder_job_id"]
            if owner not in {None, holder_job_id}:
                raise DecodedEvaluationExecutorError(
                    f"holder attempted to open a foreign directory: {relative}"
                )
            relative_path = Path(relative)
            parent_key = (
                "."
                if relative_path.parent == Path(".")
                else relative_path.parent.as_posix()
            )
            parent = handles.get(parent_key)
            if parent is None:
                raise DecodedEvaluationExecutorError(
                    f"retained directory parent is absent: {relative}"
                )
            if _directory_inode(parent.identity) != _directory_inode(
                row["parent_identity"]
            ):
                raise DecodedEvaluationExecutorError(
                    f"directory authority parent differs: {relative}"
                )
            handle = parent.open_child(
                relative_path.name,
                label=f"evaluation directory {relative}",
                expected_identity=row["identity"],
            )
            if stat.S_IMODE(handle.identity["mode"]) != row["expected_mode"]:
                handle.close()
                raise DecodedEvaluationExecutorError(
                    f"directory authority mode differs: {relative}"
                )
            handle.replay(expected_entries=set(row["expected_entries"]))
            handles[relative] = handle
        return handles
    except Exception:
        for relative, handle in sorted(
            handles.items(), key=lambda item: item[0].count("/"), reverse=True
        ):
            if relative != ".":
                handle.close()
        raise


def _build_holder_directory_completion(
    *,
    bundle: Mapping[str, Any],
    holder_job_id: str,
    directory_handles: Mapping[str, _HeldDirectory],
    holder_summary_digest: str,
) -> dict[str, Any]:
    """Project the holder's retained mutable directories after summary seal."""

    topology = plan.build_directory_topology(
        bundle["manifest"], input_spec=bundle["input_spec"]
    )
    mutable = plan._holder_mutable_topology_rows(topology, holder_job_id)
    rows: list[dict[str, Any]] = []
    for expected in mutable:
        relative = expected["relative_path"]
        handle = directory_handles.get(relative)
        if handle is None or handle.parent is None:
            raise DecodedEvaluationExecutorError(
                f"holder mutable directory is not retained: {relative}"
            )
        handle.replay()
        handle.parent.replay()
        observed = os.fstat(handle.descriptor)
        parent = os.fstat(handle.parent.descriptor)
        if (
            _directory_identity(observed) != handle.identity
            or _directory_identity(parent) != handle.parent.identity
        ):
            raise DecodedEvaluationExecutorError(
                f"holder mutable directory identity differs: {relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "path": str(
                    Path(bundle["manifest"]["evaluation_root"]) / relative
                ),
                "owner_holder_job_id": holder_job_id,
                "expected_mode": stat.S_IMODE(observed.st_mode),
                "expected_entries": sorted(handle.entries),
                "identity": _stat_identity_row(observed),
                "parent_identity": _stat_identity_row(parent),
            }
        )
    value: dict[str, Any] = {
        "schema_version": plan.HOLDER_DIRECTORY_COMPLETION_SCHEMA,
        "evaluation_root": bundle["manifest"]["evaluation_root"],
        "base_authority_digest": bundle["directory_authority"][
            "authority_digest"
        ],
        "base_topology_digest": bundle["directory_authority"][
            "topology_digest"
        ],
        "holder_job_id": holder_job_id,
        "holder_summary_digest": holder_summary_digest,
        "rows": rows,
        "row_count": len(rows),
    }
    value["completion_digest"] = object_sha256(value)
    try:
        return plan.validate_holder_directory_completion(
            value,
            topology=topology,
            base_directory_authority=bundle["directory_authority"],
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error


def _execution_claim(
    *, bundle: Mapping[str, Any], holder_job_id: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "bernini-action-preservation-execution-claim-v1",
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "directory_authority_digest": bundle["directory_authority"][
            "authority_digest"
        ],
        "holder_job_id": holder_job_id,
        "claim_pid": os.getpid(),
        "create_only_o_excl": True,
        "automatic_retry_allowed": False,
    }
    value["claim_digest"] = object_sha256(value)
    return value


def _task_id(task: Mapping[str, Any]) -> str:
    if task["task_kind"] == "adapter_candidate":
        return _identifier(task["record"]["candidate_id"], label="candidate id")
    if task["task_kind"] == "frozen_base_control":
        return _identifier(task["record"]["control_id"], label="control id")
    raise DecodedEvaluationExecutorError("task kind differs")


def _tool_identity(value: Any, *, label: str, verify_file: bool) -> dict[str, str]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    path = row["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path must be absolute")
    _sha(row["sha256"], label=f"{label} SHA")
    if verify_file:
        file_path = _plain_file(Path(path), label=label)
        if file_sha256(file_path) != row["sha256"]:
            raise DecodedEvaluationExecutorError(f"{label} file hash differs")
        if not os.access(file_path, os.X_OK):
            raise DecodedEvaluationExecutorError(f"{label} is not executable")
    return row


def _artifact_identity(value: Any, *, label: str, verify_file: bool) -> dict[str, str]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    path = row["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path must be absolute")
    _sha(row["sha256"], label=f"{label} SHA")
    if verify_file:
        file_path = _plain_file(Path(path), label=label)
        if file_sha256(file_path) != row["sha256"]:
            raise DecodedEvaluationExecutorError(f"{label} file hash differs")
    return row


def _consumption_input_identity(
    value: Any, *, verify_file: bool
) -> dict[str, str]:
    fields = {"path", "sha256", "consumption_input_digest"}
    row = dict(_closed(value, fields, label="consumption input identity"))
    path = row["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedEvaluationExecutorError("consumption input path differs")
    _sha(row["sha256"], label="consumption input file")
    _sha(row["consumption_input_digest"], label="consumption input")
    if verify_file:
        value_on_disk = _load_json(Path(path), label="consumption input")
        try:
            validated = model_authority.validate_consumption_input(value_on_disk)
        except model_authority.ModelConsumptionAuthorityError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if (
            file_sha256(path) != row["sha256"]
            or validated["consumption_input_digest"]
            != row["consumption_input_digest"]
        ):
            raise DecodedEvaluationExecutorError(
                "consumption input file binding differs"
            )
    return row


def _capture_evidence(
    value: Any, *, label: str, allow_none: bool = False
) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    fields = {
        "receipt_path", "receipt_sha256", "capture_digest", "target",
        "target_arguments_sha256",
    }
    row = dict(_closed(value, fields, label=label))
    if not isinstance(row["receipt_path"], str) or not Path(
        row["receipt_path"]
    ).is_absolute():
        raise DecodedEvaluationExecutorError(f"{label} path differs")
    for field in (
        "receipt_sha256", "capture_digest", "target_arguments_sha256"
    ):
        _sha(row[field], label=f"{label} {field}")
    if row["target"] not in bridge.verified_release.ALLOWED_PYTHON_TARGETS:
        raise DecodedEvaluationExecutorError(f"{label} target differs")
    return row


def build_task_input_receipt(
    *,
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    task: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    consumption_input_identity: Mapping[str, Any],
    verify_tools: bool,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = _task_id(task)
    decoder = _tool_identity(decoder_identity, label="decoder adapter", verify_file=verify_tools)
    ffprobe = _tool_identity(ffprobe_identity, label="ffprobe", verify_file=verify_tools)
    physical_bindings = _artifact_identity(
        physical_bindings_identity,
        label="physical bindings",
        verify_file=verify_tools,
    )
    consumption_input = _consumption_input_identity(
        consumption_input_identity, verify_file=verify_tools
    )
    executor_capture = _capture_evidence(
        executor_verified_release_capture,
        label="executor verified release capture",
        allow_none=not verify_tools,
    )
    if verify_tools and (
        executor_capture is None
        or executor_capture["target"]
        != "action_preservation_decoded_eval_executor_v2.py"
    ):
        raise DecodedEvaluationExecutorError(
            "production task lacks executor verified release capture"
        )
    if not verify_tools and executor_capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected task may not claim an executor verified capture"
        )
    value = {
        "schema_version": TASK_INPUT_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": dict(shard["holder"]),
        "task_id": task_id,
        "task_kind": task["task_kind"],
        "task_record": task["record"],
        "task_record_digest": task["record"]["record_digest"],
        "decoder_adapter": decoder,
        "ffprobe": ffprobe,
        "physical_bindings": physical_bindings,
        "model_consumption_input": consumption_input,
        "executor_source_sha256": file_sha256(__file__),
        "executor_verified_release_capture": executor_capture,
        "execution_backend": "pinned_local_subprocess"
        if verify_tools
        else "injected_stub",
        "tool_files_verified": verify_tools,
        "attempt_number": 1,
        "retry_allowed": False,
        "training_loss_read_or_used": False,
        "network_allowed": False,
        "remote_launch_performed": False,
        "direct_exec_shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
    }
    value["input_digest"] = object_sha256(value)
    return validate_task_input_receipt(value, task=task, bundle=bundle, shard=shard)


def validate_task_input_receipt(
    value: Any,
    *,
    task: Mapping[str, Any],
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_manifest_digest",
        "publication_digest",
        "shard_digest",
        "holder",
        "task_id",
        "task_kind",
        "task_record",
        "task_record_digest",
        "decoder_adapter",
        "ffprobe",
        "physical_bindings",
        "model_consumption_input",
        "executor_source_sha256",
        "executor_verified_release_capture",
        "execution_backend",
        "tool_files_verified",
        "attempt_number",
        "retry_allowed",
        "training_loss_read_or_used",
        "network_allowed",
        "remote_launch_performed",
        "direct_exec_shell",
        "subprocess_environment_denylist",
        "input_digest",
    }
    row = dict(_closed(value, fields, label="task input receipt"))
    if row["schema_version"] != TASK_INPUT_SCHEMA:
        raise DecodedEvaluationExecutorError("task input schema differs")
    expected = {
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": shard["holder"],
        "task_id": _task_id(task),
        "task_kind": task["task_kind"],
        "task_record": task["record"],
        "task_record_digest": task["record"]["record_digest"],
        "physical_bindings": row["physical_bindings"],
        "attempt_number": 1,
        "retry_allowed": False,
        "training_loss_read_or_used": False,
        "network_allowed": False,
        "remote_launch_performed": False,
        "direct_exec_shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise DecodedEvaluationExecutorError(f"task input binding differs: {key}")
    _tool_identity(row["decoder_adapter"], label="decoder adapter", verify_file=False)
    _tool_identity(row["ffprobe"], label="ffprobe", verify_file=False)
    _artifact_identity(row["physical_bindings"], label="physical bindings", verify_file=False)
    _consumption_input_identity(
        row["model_consumption_input"], verify_file=False
    )
    _sha(row["executor_source_sha256"], label="executor source")
    if row["execution_backend"] not in {"pinned_local_subprocess", "injected_stub"}:
        raise DecodedEvaluationExecutorError("execution backend differs")
    if type(row["tool_files_verified"]) is not bool:
        raise DecodedEvaluationExecutorError("tool verification flag differs")
    if (row["execution_backend"] == "pinned_local_subprocess") is not row[
        "tool_files_verified"
    ]:
        raise DecodedEvaluationExecutorError("execution backend/tool verification closure differs")
    capture = _capture_evidence(
        row["executor_verified_release_capture"],
        label="executor verified release capture",
        allow_none=not row["tool_files_verified"],
    )
    if row["tool_files_verified"] and (
        capture is None
        or capture["target"]
        != "action_preservation_decoded_eval_executor_v2.py"
    ):
        raise DecodedEvaluationExecutorError(
            "task executor verified release capture differs"
        )
    if not row["tool_files_verified"] and capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected task claims a verified release capture"
        )
    _verify_digest(row, field="input_digest", label="task input receipt")
    return row


def _fd_inheritance_evidence(
    binding: Mapping[str, Any], *, production_mode: bool,
    spawn_performed: bool,
) -> dict[str, Any]:
    inherited = model_authority.validate_inherited_fd_binding(
        binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    value: dict[str, Any] = {
        "schema_version": FD_INHERITANCE_SCHEMA,
        "fd_binding": inherited,
        "fd_binding_digest": inherited["fd_binding_digest"],
        "fd_count": inherited["fd_count"],
        "production_mode": production_mode,
        "decoder_spawn_performed": spawn_performed,
        "decoder_spawn_close_fds_true": spawn_performed,
        "exact_pass_fds_only": spawn_performed,
        "executor_parent_fds_cloexec_before_spawn": True,
        "executor_parent_fds_cloexec_after_wait": True,
        "unrelated_child_inherits_authority_fds": False,
        "ptrace_authorization_used": False,
        "injected_fixture": not production_mode,
    }
    value["inheritance_digest"] = object_sha256(value)
    return value


def _validate_fd_inheritance_evidence(
    value: Any, *, production_required: bool, return_code: int
) -> dict[str, Any]:
    fields = {
        "schema_version", "fd_binding", "fd_binding_digest", "fd_count",
        "production_mode",
        "decoder_spawn_performed", "decoder_spawn_close_fds_true",
        "exact_pass_fds_only", "executor_parent_fds_cloexec_before_spawn",
        "executor_parent_fds_cloexec_after_wait",
        "unrelated_child_inherits_authority_fds",
        "ptrace_authorization_used", "injected_fixture", "inheritance_digest",
    }
    row = dict(_closed(value, fields, label="decoder FD inheritance"))
    binding = model_authority.validate_inherited_fd_binding(
        row["fd_binding"], verify_open_fds=False
    )
    unsigned = dict(row)
    claimed = unsigned.pop("inheritance_digest")
    if (
        row["schema_version"] != FD_INHERITANCE_SCHEMA
        or row["fd_binding_digest"] != binding["fd_binding_digest"]
        or row["fd_count"] != binding["fd_count"]
        or row["executor_parent_fds_cloexec_before_spawn"] is not True
        or row["executor_parent_fds_cloexec_after_wait"] is not True
        or row["unrelated_child_inherits_authority_fds"] is not False
        or row["ptrace_authorization_used"] is not False
        or claimed != object_sha256(unsigned)
    ):
        raise DecodedEvaluationExecutorError(
            "decoder FD inheritance evidence differs"
        )
    if production_required:
        spawn = row["decoder_spawn_performed"]
        if (
            row["production_mode"] is not True
            or row["injected_fixture"] is not False
            or type(spawn) is not bool
            or row["decoder_spawn_close_fds_true"] is not spawn
            or row["exact_pass_fds_only"] is not spawn
            or (not spawn and return_code != 253)
        ):
            raise DecodedEvaluationExecutorError(
                "production decoder FD inheritance differs"
            )
    elif (
        row["production_mode"] is not False
        or row["injected_fixture"] is not True
        or type(row["decoder_spawn_performed"]) is not bool
        or row["decoder_spawn_close_fds_true"]
        is not row["decoder_spawn_performed"]
        or row["exact_pass_fds_only"]
        is not row["decoder_spawn_performed"]
    ):
        raise DecodedEvaluationExecutorError(
            "injected decoder FD inheritance differs"
        )
    return row


def _validate_process_observation(
    value: Any, *, production_required: bool
) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {"return_code", "stdout", "stderr", "fd_inheritance"},
            label="decoder process observation",
        )
    )
    if type(row["return_code"]) is not int:
        raise DecodedEvaluationExecutorError("decoder return code is not an integer")
    for key in ("stdout", "stderr"):
        if not isinstance(row[key], bytes):
            raise DecodedEvaluationExecutorError(f"decoder {key} is not bytes")
    row["fd_inheritance"] = _validate_fd_inheritance_evidence(
        row["fd_inheritance"], production_required=production_required,
        return_code=row["return_code"],
    )
    return row


def build_process_receipt(
    *,
    input_receipt: Mapping[str, Any],
    observation: Mapping[str, Any],
    request_path: Path,
    staging_path: Path,
    consumption_digest: str,
) -> dict[str, Any]:
    observed = _validate_process_observation(
        observation,
        production_required=input_receipt["tool_files_verified"],
    )
    _sha(consumption_digest, label="process consumption")
    value = {
        "schema_version": PROCESS_SCHEMA,
        "task_id": input_receipt["task_id"],
        "input_digest": input_receipt["input_digest"],
        "consumption_digest": consumption_digest,
        "decoder_adapter_sha256": input_receipt["decoder_adapter"]["sha256"],
        "protocol_target": (
            "action_preservation_decoded_eval_decoder_adapter_v1.py"
        ),
        "protocol_arguments": [
            "--request",
            str(request_path),
            "--output",
            str(staging_path),
        ],
        "verified_runtime_required": input_receipt["tool_files_verified"],
        "root_bootstrap_source_sha256": (
            bytes_sha256(
                bridge.verified_release.ROOT_BOOTSTRAP_SOURCE.encode("utf-8")
            )
            if input_receipt["tool_files_verified"]
            else None
        ),
        "decoder_runtime_capture_receipt_path": (
            str(request_path.parent / DECODER_RUNTIME_CAPTURE_FILENAME)
            if input_receipt["tool_files_verified"]
            else None
        ),
        "release_member_path_executed_directly": False,
        "return_code": observed["return_code"],
        "stdout_sha256": bytes_sha256(observed["stdout"]),
        "stdout_size": len(observed["stdout"]),
        "stderr_sha256": bytes_sha256(observed["stderr"]),
        "stderr_size": len(observed["stderr"]),
        "retry_attempted": False,
        "shell": False,
        "subprocess_environment_denylist": list(SUBPROCESS_ENV_DENYLIST),
        "fd_inheritance": observed["fd_inheritance"],
    }
    value["process_digest"] = object_sha256(value)
    return value


def _fraction(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str):
        raise DecodedEvaluationExecutorError(f"{label} is not a fraction string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise DecodedEvaluationExecutorError(f"{label} is invalid") from error
    if result <= 0:
        raise DecodedEvaluationExecutorError(f"{label} is non-positive")
    return result


def parse_ffprobe_json(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("streams"), list)
        or not isinstance(value.get("frames"), list)
    ):
        raise DecodedEvaluationExecutorError("ffprobe JSON stream closure differs")
    video_streams = [
        stream
        for stream in value["streams"]
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise DecodedEvaluationExecutorError("decoded output must have exactly one video stream")
    stream = video_streams[0]
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        frame_count = int(frame_value)
    except (TypeError, ValueError) as error:
        raise DecodedEvaluationExecutorError("decoded frame count is unavailable") from error
    rate = _fraction(
        stream.get("avg_frame_rate", stream.get("r_frame_rate")),
        label="decoded average frame rate",
    )
    format_value = value.get("format", {})
    format_name = format_value.get("format_name") if isinstance(format_value, Mapping) else None
    if not isinstance(format_name, str) or not format_name:
        raise DecodedEvaluationExecutorError("decoded container format is unavailable")
    video_frames = [
        item
        for item in value["frames"]
        if isinstance(item, Mapping) and item.get("media_type") == "video"
    ]
    timestamps: list[str] = []
    for item in video_frames:
        timestamp = item.get("best_effort_timestamp_time", item.get("pts_time"))
        if not isinstance(timestamp, str):
            raise DecodedEvaluationExecutorError(
                "decoded frame timestamp is unavailable"
            )
        try:
            Fraction(timestamp)
        except (ValueError, ZeroDivisionError) as error:
            raise DecodedEvaluationExecutorError(
                "decoded frame timestamp is invalid"
            ) from error
        timestamps.append(timestamp)
    return {
        "video_stream_count": len(video_streams),
        "frame_count": frame_count,
        "fps_num": rate.numerator,
        "fps_den": rate.denominator,
        "format_name": format_name,
        "frame_timestamp_times": timestamps,
    }


def validate_probe_result(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {
                "video_stream_count", "frame_count", "fps_num", "fps_den",
                "format_name", "frame_timestamp_times",
            },
            label="probe result",
        )
    )
    if (
        row["video_stream_count"] != 1
        or row["frame_count"] != 81
        or row["fps_num"] != 25
        or row["fps_den"] != 1
    ):
        raise DecodedEvaluationExecutorError("decoded output is not exact full81@25fps")
    if not isinstance(row["format_name"], str) or not row["format_name"]:
        raise DecodedEvaluationExecutorError("decoded format name differs")
    if "mp4" not in row["format_name"].lower() and "mov" not in row["format_name"].lower():
        raise DecodedEvaluationExecutorError("decoded output is not an MP4-family container")
    timestamps = row["frame_timestamp_times"]
    if not isinstance(timestamps, list) or len(timestamps) != 81:
        raise DecodedEvaluationExecutorError(
            "decoded output lacks exact 81-frame PTS evidence"
        )
    try:
        parsed = [Fraction(item) for item in timestamps]
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise DecodedEvaluationExecutorError(
            "decoded frame PTS evidence is invalid"
        ) from error
    if any(
        current - previous != Fraction(1, 25)
        for previous, current in zip(parsed, parsed[1:])
    ):
        raise DecodedEvaluationExecutorError(
            "decoded output is variable-rate or not exact 25fps PTS cadence"
        )
    return row


def _staging_observation(
    path: Path, *, directory: _HeldDirectory | None = None
) -> dict[str, Any]:
    if directory is None:
        if not os.path.lexists(path):
            return {"exists": False, "sha256": None, "size": None}
        try:
            info = path.lstat()
        except OSError:
            return {"exists": True, "sha256": None, "size": None}
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return {"exists": True, "sha256": None, "size": None}
        return {
            "exists": True,
            "sha256": file_sha256(path),
            "size": info.st_size,
        }
    try:
        if not directory.exists(path.name):
            return {"exists": False, "sha256": None, "size": None}
        _identity, digest, size = directory.stable_file(
            path.name, label="failure artifact"
        )
    except DecodedEvaluationExecutorError:
        return {"exists": True, "sha256": None, "size": None}
    return {"exists": True, "sha256": digest, "size": size}


def build_failure_receipt(
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any] | None,
    staging_path: Path,
    final_path: Path,
    failure_kind: str,
    failure_detail: str,
    consumption_digest: str,
    staging_directory: _HeldDirectory | None = None,
    final_directory: _HeldDirectory | None = None,
) -> dict[str, Any]:
    _identifier(failure_kind, label="failure kind")
    _sha(consumption_digest, label="failure consumption")
    if not isinstance(failure_detail, str) or not failure_detail:
        raise DecodedEvaluationExecutorError("failure detail is empty")
    value = {
        "schema_version": TASK_FAILURE_SCHEMA,
        "task_id": input_receipt["task_id"],
        "input_digest": input_receipt["input_digest"],
        "consumption_digest": consumption_digest,
        "process_digest": process_receipt["process_digest"]
        if process_receipt is not None
        else None,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "staging_artifact": _staging_observation(
            staging_path, directory=staging_directory
        ),
        "final_artifact": _staging_observation(
            final_path, directory=final_directory
        ),
        "attempt_number": 1,
        "retry_attempted": False,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "training_loss_read_or_used": False,
    }
    value["failure_digest"] = object_sha256(value)
    return value


def validate_failure_receipt(
    value: Any, *, input_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "input_digest",
        "consumption_digest",
        "process_digest",
        "failure_kind",
        "failure_detail",
        "staging_artifact",
        "final_artifact",
        "attempt_number",
        "retry_attempted",
        "retry_allowed",
        "failure_artifacts_retained",
        "training_loss_read_or_used",
        "failure_digest",
    }
    row = dict(_closed(value, fields, label="task failure receipt"))
    if row["schema_version"] != TASK_FAILURE_SCHEMA:
        raise DecodedEvaluationExecutorError("task failure schema differs")
    if (
        row["task_id"] != input_receipt["task_id"]
        or row["input_digest"] != input_receipt["input_digest"]
    ):
        raise DecodedEvaluationExecutorError("task failure input binding differs")
    _sha(row["consumption_digest"], label="failure consumption")
    if row["process_digest"] is not None:
        _sha(row["process_digest"], label="failure process digest")
    _identifier(row["failure_kind"], label="failure kind")
    if not isinstance(row["failure_detail"], str) or not row["failure_detail"]:
        raise DecodedEvaluationExecutorError("failure detail differs")
    artifact_fields = {"exists", "sha256", "size"}
    for label in ("staging_artifact", "final_artifact"):
        artifact = _closed(row[label], artifact_fields, label=label)
        if type(artifact["exists"]) is not bool:
            raise DecodedEvaluationExecutorError(f"{label} existence differs")
        if artifact["sha256"] is not None:
            _sha(artifact["sha256"], label=f"{label} SHA")
        if artifact["size"] is not None and (
            type(artifact["size"]) is not int or artifact["size"] < 0
        ):
            raise DecodedEvaluationExecutorError(f"{label} size differs")
    expected = {
        "attempt_number": 1,
        "retry_attempted": False,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "training_loss_read_or_used": False,
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise DecodedEvaluationExecutorError(f"failure policy differs: {key}")
    _verify_digest(row, field="failure_digest", label="task failure receipt")
    return row


def build_output_receipt(
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    output_path: Path,
    probe_result: Mapping[str, Any],
    native_inference_evidence: Mapping[str, Any] | None = None,
    consumption_chain: Mapping[str, Any] | None = None,
    publication_gate: Mapping[str, Any] | None = None,
    published_inode_identity: Mapping[str, Any] | None = None,
    output_directory: _HeldDirectory | None = None,
    published_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe = validate_probe_result(probe_result)
    try:
        consumption = model_authority.validate_consumption_chain(
            consumption_chain
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    if (
        not isinstance(publication_gate, Mapping)
        or publication_gate.get("schema_version")
        != model_authority.PUBLICATION_GATE_SCHEMA
        or publication_gate.get("task_id") != input_receipt["task_id"]
        or publication_gate.get("consumption_digest")
        != consumption["consumption_digest"]
        or publication_gate.get("publication_authorized") is not True
        or publication_gate.get("publication_has_occurred") is not False
    ):
        raise DecodedEvaluationExecutorError("consumption publication gate differs")
    unsigned_gate = dict(publication_gate)
    claimed_gate = unsigned_gate.pop("publication_gate_digest", None)
    if claimed_gate != object_sha256(unsigned_gate):
        raise DecodedEvaluationExecutorError("publication gate digest differs")
    if (
        consumption["task_id"] != input_receipt["task_id"]
        or consumption["consumption_input_digest"]
        != input_receipt["model_consumption_input"][
            "consumption_input_digest"
        ]
        or process_receipt.get("input_digest") != input_receipt["input_digest"]
        or process_receipt.get("consumption_digest")
        != consumption["consumption_digest"]
    ):
        raise DecodedEvaluationExecutorError(
            "output receipt input/process/consumption closure differs"
        )
    if published_observation is None:
        publication_identity, output_sha256, output_size = _stable_published_file(
            output_path,
            expected_identity=published_inode_identity,
            expected_sha256=publication_gate.get("staging_sha256"),
            expected_size=publication_gate.get("staging_size"),
            label="published output",
            directory=output_directory,
        )
    else:
        observed = dict(
            _closed(
                published_observation,
                {"identity", "sha256", "size"},
                label="published output observation",
            )
        )
        publication_identity = _published_inode_identity(
            observed["identity"]
        )
        output_sha256 = _sha(
            observed["sha256"], label="published output observation"
        )
        output_size = observed["size"]
        if (
            type(output_size) is not int
            or output_size <= 0
            or publication_identity != _published_inode_identity(
                published_inode_identity
            )
            or publication_identity["size"] != output_size
            or output_sha256 != publication_gate.get("staging_sha256")
            or output_size != publication_gate.get("staging_size")
        ):
            raise DecodedEvaluationExecutorError(
                "published output observation differs"
            )
    if input_receipt["tool_files_verified"]:
        if not isinstance(native_inference_evidence, Mapping) or set(
            native_inference_evidence
        ) != {
            "receipt_path",
            "receipt_sha256",
            "receipt_digest",
            "decoder_verified_release_capture",
            "inference_verified_release_capture",
        }:
            raise DecodedEvaluationExecutorError(
                "verified execution lacks native inference receipt evidence"
            )
        native = dict(native_inference_evidence)
        if not isinstance(native["receipt_path"], str) or not Path(
            native["receipt_path"]
        ).is_absolute():
            raise DecodedEvaluationExecutorError("native inference receipt path differs")
        _sha(native["receipt_sha256"], label="native inference receipt file")
        _sha(native["receipt_digest"], label="native inference receipt")
        decoder_capture = _capture_evidence(
            native["decoder_verified_release_capture"],
            label="decoder verified release capture",
        )
        inference_capture = _capture_evidence(
            native["inference_verified_release_capture"],
            label="inference verified release capture",
        )
        if (
            decoder_capture is None
            or decoder_capture["target"]
            != "action_preservation_decoded_eval_decoder_adapter_v1.py"
            or inference_capture is None
            or inference_capture["target"] != "infer_lora.py"
        ):
            raise DecodedEvaluationExecutorError(
                "native verified release capture target differs"
            )
        if native["receipt_digest"] != consumption[
            "native_inference_receipt_digest"
        ]:
            raise DecodedEvaluationExecutorError(
                "native inference receipt differs from consumption chain"
            )
    elif native_inference_evidence is not None:
        raise DecodedEvaluationExecutorError(
            "injected execution may not claim native inference evidence"
        )
    else:
        native = None
    value = {
        "schema_version": TASK_OUTPUT_SCHEMA,
        "task_id": input_receipt["task_id"],
        "task_kind": input_receipt["task_kind"],
        "input_digest": input_receipt["input_digest"],
        "process_digest": process_receipt["process_digest"],
        "consumption_chain": consumption,
        "consumption_digest": consumption["consumption_digest"],
        "publication_gate": dict(publication_gate),
        "publication_gate_digest": claimed_gate,
        "task_record_digest": input_receipt["task_record_digest"],
        "source_video_sha256": input_receipt["task_record"]["source_video_sha256"],
        "instruction_sha256": input_receipt["task_record"]["instruction_sha256"],
        "seed": input_receipt["task_record"]["seed"],
        "onset_policy": input_receipt["task_record"]["onset_policy"]["name"],
        "execution_backend": input_receipt["execution_backend"],
        "tool_files_verified": input_receipt["tool_files_verified"],
        "output_relpath": input_receipt["task_record"]["output_relpath"],
        "output_video_sha256": output_sha256,
        "output_byte_size": output_size,
        "published_inode_identity": publication_identity,
        "probe": probe,
        "media_contract_satisfied": True,
        "native_inference_receipt": native,
        "exact_input_binding_satisfied": native is not None,
        "training_loss_read_or_used": False,
        "retry_attempted": False,
        "retry_allowed": False,
        "remote_launch_performed": False,
    }
    value["output_digest"] = object_sha256(value)
    return value


def validate_output_receipt(
    value: Any,
    *,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    output_path: Path,
    output_directory: _HeldDirectory | None = None,
    published_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "task_kind",
        "input_digest",
        "process_digest",
        "consumption_chain",
        "consumption_digest",
        "publication_gate",
        "publication_gate_digest",
        "task_record_digest",
        "source_video_sha256",
        "instruction_sha256",
        "seed",
        "onset_policy",
        "execution_backend",
        "tool_files_verified",
        "output_relpath",
        "output_video_sha256",
        "output_byte_size",
        "published_inode_identity",
        "probe",
        "media_contract_satisfied",
        "native_inference_receipt",
        "exact_input_binding_satisfied",
        "training_loss_read_or_used",
        "retry_attempted",
        "retry_allowed",
        "remote_launch_performed",
        "output_digest",
    }
    row = dict(_closed(value, fields, label="task output receipt"))
    if row["schema_version"] != TASK_OUTPUT_SCHEMA:
        raise DecodedEvaluationExecutorError("task output schema differs")
    expected = build_output_receipt(
        input_receipt=input_receipt,
        process_receipt=process_receipt,
        output_path=output_path,
        probe_result=row["probe"],
        native_inference_evidence=row["native_inference_receipt"],
        consumption_chain=row["consumption_chain"],
        publication_gate=row["publication_gate"],
        published_inode_identity=row["published_inode_identity"],
        output_directory=output_directory,
        published_observation=published_observation,
    )
    if row != expected:
        raise DecodedEvaluationExecutorError("task output receipt binding differs")
    _verify_digest(row, field="output_digest", label="task output receipt")
    return row


DecoderRunner = Callable[
    [Path, Path, Mapping[str, Any]], Mapping[str, Any]
]
VideoProber = Callable[[Path], Mapping[str, Any]]


def _verify_native_inference_receipt(
    *,
    input_receipt: Mapping[str, Any],
    input_receipt_path: Path,
    staging_path: Path,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> dict[str, Any]:
    try:
        normalized_request, bindings, source, checkpoint = decoder_adapter.resolve_request(
            input_receipt, verify_files=True, verify_inherited_fds=False
        )
        receipt_path = staging_path.with_name(staging_path.name + ".receipt.json")
        validated = decoder_adapter.validate_inference_receipt(
            receipt,
            request=normalized_request,
            bindings=bindings,
            source=source,
            checkpoint=checkpoint,
            output_path=staging_path,
        )
        decoder_capture = bridge.validate_verified_capture_receipt(
            bindings,
            receipt_path=input_receipt_path.parent
            / DECODER_RUNTIME_CAPTURE_FILENAME,
            target="action_preservation_decoded_eval_decoder_adapter_v1.py",
            expected_arguments=[
                "--request", str(input_receipt_path),
                "--output", str(staging_path),
            ],
            verify_file=True,
        )
        inference_arguments = decoder_adapter.inference_target_arguments(
            request=normalized_request, bindings=bindings, source=source,
            checkpoint=checkpoint, output_path=staging_path,
        )
        inference_capture = bridge.validate_verified_capture_receipt(
            bindings,
            receipt_path=decoder_adapter.inference_runtime_capture_path(
                staging_path
            ),
            target="infer_lora.py",
            expected_arguments=inference_arguments,
            verify_file=True,
        )
    except (
        decoder_adapter.DecodedEvaluationDecoderError,
        bridge.DecodedEvaluationBridgeError,
    ) as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha(
            receipt_sha256, label="native inference receipt file"
        ),
        "receipt_digest": validated["receipt_digest"],
        "decoder_verified_release_capture": decoder_capture,
        "inference_verified_release_capture": inference_capture,
    }


def _validate_decoder_stdout_authority(
    *,
    stdout: bytes,
    native_receipt_raw: bytes,
    native_receipt: Mapping[str, Any],
    decoder_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind child-produced files to the two unavoidable canonical lines.

    A same-UID writer may append to the pipe, but it cannot delete the genuine
    infer and decoder lines.  Exact-two parsing therefore turns injection into
    a fail-closed denial of service rather than a forgeable file authority.
    """

    if not isinstance(stdout, bytes):
        raise DecodedEvaluationExecutorError(
            "decoder stdout authority is not bytes"
        )
    lines = stdout.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") for line in lines):
        raise DecodedEvaluationExecutorError(
            "decoder stdout authority is not exact-two canonical lines"
        )
    first = _decode_canonical_json(lines[0], label="decoder native stdout")
    second = _decode_canonical_json(lines[1], label="decoder result stdout")
    if (
        lines[0] != native_receipt_raw
        or first != dict(native_receipt)
        or second != dict(decoder_result)
    ):
        raise DecodedEvaluationExecutorError(
            "decoder stdout/file authority differs"
        )
    output = first.get("output")
    if (
        not isinstance(output, Mapping)
        or not isinstance(output.get("sha256"), str)
    ):
        raise DecodedEvaluationExecutorError(
            "decoder stdout output authority differs"
        )
    _sha(output["sha256"], label="decoder stdout output")
    return first


def _validate_retained_staging_against_native(
    *, native_receipt: Mapping[str, Any], staging_sha256: str,
    staging_size: int, staging_identity: Mapping[str, Any],
) -> None:
    output = native_receipt.get("output")
    if (
        not isinstance(output, Mapping)
        or staging_sha256 != output.get("sha256")
        or staging_size != output.get("size")
        or dict(staging_identity) != output.get("publication_identity")
    ):
        raise DecodedEvaluationExecutorError(
            "retained staging bytes/inode differ from decoder stdout authority"
        )


def _publish_failure(
    *,
    task_root: _HeldDirectory,
    output_parent: _HeldDirectory,
    input_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any] | None,
    staging_path: Path,
    final_path: Path,
    failure_kind: str,
    failure_detail: str,
    consumption_chain: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        chain = model_authority.validate_consumption_chain(consumption_chain)
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    receipt = build_failure_receipt(
        input_receipt=input_receipt,
        process_receipt=process_receipt,
        staging_path=staging_path,
        final_path=final_path,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        consumption_digest=chain["consumption_digest"],
        staging_directory=task_root,
        final_directory=output_parent,
    )
    receipt = validate_failure_receipt(receipt, input_receipt=input_receipt)
    task_root.write_json(FAILURE_RECEIPT_FILENAME, receipt)
    result = {
        "task_id": input_receipt["task_id"],
        "status": "failure",
        "terminal_receipt_digest": receipt["failure_digest"],
        "consumption_digest": chain["consumption_digest"],
        "publication_gate_digest": None,
        "output_relpath": input_receipt["task_record"]["output_relpath"],
    }
    result["result_digest"] = object_sha256(result)
    task_root.replay()
    return result


def _execute_task_impl(
    *,
    evaluation_root: Path,
    task_parent: _HeldDirectory,
    directory_handles: Mapping[str, _HeldDirectory],
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    task: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    physical_bindings: Mapping[str, Any],
    model_consumption: model_authority.ModelAuthority,
    model_capture_identity: Mapping[str, Any],
    adapter_proc_fd_prefix: str | None,
    run_decoder: DecoderRunner,
    probe_video: VideoProber,
    verify_tools: bool,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
    active_adapter_authorities: list[model_authority.AdapterAuthority],
) -> dict[str, Any]:
    task_id = _task_id(task)
    holder_job_id = shard["holder"]["job_id"]
    task_relative = (
        f"{plan.EXECUTION_SHARD_DIRECTORY}/{holder_job_id}/tasks/{task_id}"
    )
    task_root = directory_handles.get(task_relative)
    if task_root is None or task_root.parent is not task_parent:
        raise DecodedEvaluationExecutorError(
            "pre-authorized retained task directory is absent"
        )
    task_root.replay(expected_entries=set())
    record = task["record"]
    relative_output = Path(record["output_relpath"])
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise DecodedEvaluationExecutorError("task output path escapes evaluation root")
    output_parent_key = relative_output.parent.as_posix()
    output_parent = directory_handles.get(output_parent_key)
    if output_parent is None:
        raise DecodedEvaluationExecutorError(
            "task output parent lacks retained directory authority"
        )
    output_parent.replay()
    final_path = output_parent.consumer_path(
        relative_output.name, production_mode=verify_tools
    )
    staging_path = task_root.consumer_path(
        STAGING_VIDEO_FILENAME, production_mode=verify_tools
    )
    model_pre = model_consumption.begin_task(task_id)
    task_root.write_json(MODEL_PRE_USE_FILENAME, model_pre)
    task_root.write_json(
        MODEL_CAPTURE_FILENAME, model_consumption.capture_receipt
    )
    task_model_capture_path = task_root.consumer_path(
        MODEL_CAPTURE_FILENAME, production_mode=verify_tools
    )
    task_model_capture_identity = {
        "path": str(task_model_capture_path),
        "sha256": bytes_sha256(
            canonical_json_bytes(model_consumption.capture_receipt) + b"\n"
        ),
    }
    adapter_consumption: model_authority.AdapterAuthority | None = None
    adapter_pre: Mapping[str, Any] | None = None
    adapter_capture_path: Path | None = None
    adapter_capture_identity: dict[str, Any] | None = None
    if task["task_kind"] == "adapter_candidate":
        matches = [
            item
            for item in physical_bindings["checkpoints"]
            if (item["arm"], item["checkpoint_step"])
            == (record["arm"], record["checkpoint_step"])
        ]
        if len(matches) != 1:
            raise DecodedEvaluationExecutorError(
                "adapter authority checkpoint binding differs"
            )
        checkpoint = matches[0]
        adapter_model_identity = checkpoint["adapter_model"]
        try:
            adapter_consumption = model_authority.AdapterAuthority.capture(
                task_id=task_id,
                checkpoint_root=checkpoint["checkpoint_root"],
                expected_sha256={
                    "receipt.json": checkpoint["checkpoint_receipt"]["sha256"],
                    "adapter/adapter_config.json": checkpoint["adapter_config"]["sha256"],
                    "adapter/adapter_model.safetensors": adapter_model_identity["sha256"],
                },
                private_parent=task_root.path,
                private_parent_fd=task_root.descriptor,
                view_name="adapter_fd_view",
                expected_uid=adapter_model_identity["uid"],
                expected_gid=adapter_model_identity["gid"],
                expected_file_mode=0o444,
                proc_fd_prefix=adapter_proc_fd_prefix,
            )
            active_adapter_authorities.append(adapter_consumption)
        except model_authority.ModelConsumptionAuthorityError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        task_root.adopt_entries(
            task_root.entries
            | {adapter_consumption.capture_receipt["private_root_name"]}
        )
        task_root.write_json(
            ADAPTER_CAPTURE_FILENAME, adapter_consumption.capture_receipt
        )
        adapter_capture_path = task_root.consumer_path(
            ADAPTER_CAPTURE_FILENAME, production_mode=verify_tools
        )
        adapter_capture_identity = {
            "path": str(adapter_capture_path),
            "sha256": bytes_sha256(
                canonical_json_bytes(adapter_consumption.capture_receipt)
                + b"\n"
            ),
        }
        adapter_pre = adapter_consumption.begin_use()
        task_root.write_json(ADAPTER_PRE_USE_FILENAME, adapter_pre)
    try:
        task_publication_root = (
            model_authority.task_publication_root_binding(
                descriptor=task_root.descriptor,
                path=task_root.path,
            )
        )
        inherited_fd_binding = model_authority.build_inherited_fd_binding(
            task_id=task_id,
            model_capture=model_consumption.capture_receipt,
            adapter_capture=(
                None
                if adapter_consumption is None
                else adapter_consumption.capture_receipt
            ),
            task_publication_root=task_publication_root,
        )
        model_authority.validate_inherited_fd_binding(
            inherited_fd_binding,
            model_capture=model_consumption.capture_receipt,
            adapter_capture=(
                None
                if adapter_consumption is None
                else adapter_consumption.capture_receipt
            ),
            task_publication_root=task_publication_root,
            verify_open_fds=True,
            expected_inheritable=False,
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    try:
        consumption_input = model_authority.build_consumption_input(
            task_id=task_id,
            physical_bindings_digest=physical_bindings["physical_bindings_digest"],
            model_capture=model_consumption.capture_receipt,
            model_pre_use=model_pre,
            model_capture_receipt_path=task_model_capture_identity["path"],
            model_capture_receipt_sha256=task_model_capture_identity["sha256"],
            adapter_capture=(
                adapter_consumption.capture_receipt
                if adapter_consumption is not None
                else None
            ),
            adapter_pre_use=adapter_pre,
            adapter_capture_receipt_path=adapter_capture_path,
            adapter_capture_receipt_sha256=(
                adapter_capture_identity["sha256"]
                if adapter_capture_identity is not None
                else None
            ),
            inherited_fd_binding=inherited_fd_binding,
            task_publication_root=task_publication_root,
            production_mode=verify_tools,
            task_member_path_prefix=(
                None if verify_tools else task_root.path
            ),
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    task_root.write_json(
        CONSUMPTION_INPUT_FILENAME, consumption_input
    )
    consumption_input_path = task_root.consumer_path(
        CONSUMPTION_INPUT_FILENAME, production_mode=verify_tools
    )
    consumption_input_identity = {
        "path": str(consumption_input_path),
        "sha256": bytes_sha256(
            canonical_json_bytes(consumption_input) + b"\n"
        ),
        "consumption_input_digest": consumption_input["consumption_input_digest"],
    }
    input_receipt = build_task_input_receipt(
        bundle=bundle,
        shard=shard,
        task=task,
        decoder_identity=decoder_identity,
        ffprobe_identity=ffprobe_identity,
        physical_bindings_identity=physical_bindings_identity,
        consumption_input_identity=consumption_input_identity,
        verify_tools=verify_tools,
        executor_verified_release_capture=executor_verified_release_capture,
    )
    input_receipt_path = task_root.consumer_path(
        INPUT_RECEIPT_FILENAME, production_mode=verify_tools
    )
    expected_input_bytes = canonical_json_bytes(input_receipt) + b"\n"
    task_root.write_json(INPUT_RECEIPT_FILENAME, input_receipt)
    try:
        os.stat(
            relative_output.name,
            dir_fd=output_parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        output_preexists = False
    else:
        output_preexists = True
    if output_preexists:
        observation = {
            "return_code": 253,
            "stdout": b"",
            "stderr": b"create-only final output path already exists",
            "fd_inheritance": _fd_inheritance_evidence(
                inherited_fd_binding,
                production_mode=verify_tools,
                spawn_performed=False,
            ),
        }
    else:
        try:
            observation = dict(
                run_decoder(
                    input_receipt_path,
                    staging_path,
                    inherited_fd_binding,
                )
            )
            if "fd_inheritance" not in observation and not verify_tools:
                observation["fd_inheritance"] = _fd_inheritance_evidence(
                    inherited_fd_binding,
                    production_mode=False,
                    spawn_performed=False,
                )
            observation = _validate_process_observation(
                observation, production_required=verify_tools
            )
            if (
                observation["fd_inheritance"]["fd_binding"]
                != inherited_fd_binding
            ):
                raise DecodedEvaluationExecutorError(
                    "decoder observation inherited FD binding differs"
                )
        except Exception as error:  # backend failures become retained terminal evidence
            if verify_tools:
                raise DecodedEvaluationExecutorError(
                    f"production decoder backend failed: {type(error).__name__}: {error}"
                ) from error
            try:
                failed_fd_inheritance = _fd_inheritance_evidence(
                    inherited_fd_binding,
                    production_mode=False,
                    spawn_performed=False,
                )
            except model_authority.ModelConsumptionAuthorityError as authority_error:
                raise DecodedEvaluationExecutorError(
                    f"decoder consumption authority failed: {authority_error}"
                ) from authority_error
            observation = {
                "return_code": 255,
                "stdout": b"",
                "stderr": (
                    f"decoder backend exception: {type(error).__name__}: {error}"
                ).encode("utf-8", errors="replace"),
                "fd_inheritance": failed_fd_inheritance,
            }

    allowed_decoder_entries = {
        STAGING_VIDEO_FILENAME,
        STAGING_VIDEO_FILENAME + ".receipt.json",
    }
    if verify_tools:
        allowed_decoder_entries.update(
            {
                DECODER_RUNTIME_CAPTURE_FILENAME,
                STAGING_VIDEO_FILENAME
                + decoder_adapter.INFERENCE_RUNTIME_CAPTURE_SUFFIX,
            }
        )
    observed_task_entries = set(os.listdir(task_root.descriptor))
    added_by_decoder = observed_task_entries - task_root.entries
    if (
        not task_root.entries.issubset(observed_task_entries)
        or not added_by_decoder.issubset(allowed_decoder_entries)
    ):
        raise DecodedEvaluationExecutorError(
            "decoder changed the retained task-root entry closure"
        )
    task_root.adopt_entries(observed_task_entries)
    for added_name in added_by_decoder:
        task_root.stat_file(
            added_name,
            label=f"decoder-created task artifact {added_name}",
            expected_nlink=1,
        )

    # The decoder process is now gone.  Validate its native receipt while the
    # exact adapter FD-view is still alive; then replay both authorities and
    # fully rehash/close the adapter before any staging inode can be published.
    native_receipt_path = staging_path.with_name(staging_path.name + ".receipt.json")
    native_inference_evidence: Mapping[str, Any] | None = None
    native_receipt_name = STAGING_VIDEO_FILENAME + ".receipt.json"
    native_receipt_exists = task_root.exists(native_receipt_name)
    if observation["return_code"] == 0 and native_receipt_exists:
        native_raw, _native_identity = task_root.read(
            native_receipt_name,
            label="native inference receipt pre-chain",
        )
        native_value = _decode_canonical_json(
            native_raw, label="native inference receipt pre-chain"
        )
        native_digest = _sha(
            native_value.get("receipt_digest"), label="native inference receipt"
        )
        if native_value.get("consumption_input_digest") != consumption_input[
            "consumption_input_digest"
        ]:
            raise DecodedEvaluationExecutorError(
                "native inference receipt lacks the exact consumption input digest"
            )
        native_inference_evidence = (
            _verify_native_inference_receipt(
                input_receipt=input_receipt,
                input_receipt_path=input_receipt_path,
                staging_path=staging_path,
                receipt=native_value,
                receipt_sha256=bytes_sha256(native_raw),
            )
            if verify_tools
            else None
        )
        if verify_tools:
            _validate_decoder_stdout_authority(
                stdout=observation["stdout"],
                native_receipt_raw=native_raw,
                native_receipt=native_value,
                decoder_result=native_inference_evidence,
            )
    elif verify_tools and observation["return_code"] == 0:
        raise DecodedEvaluationExecutorError(
            "successful decoder lacks native inference receipt"
        )
    else:
        native_digest = object_sha256(
            {
                "task_id": task_id,
                "native_inference_not_completed": True,
                "return_code": observation["return_code"],
                "consumption_input_digest": consumption_input[
                    "consumption_input_digest"
                ],
            }
        )
    try:
        adapter_post = (
            adapter_consumption.end_use()
            if adapter_consumption is not None
            else None
        )
        adapter_final = (
            adapter_consumption.finalize_and_close()
            if adapter_consumption is not None
            else None
        )
        if adapter_consumption is not None:
            task_root.adopt_entries(
                task_root.entries
                - {adapter_consumption.capture_receipt["private_root_name"]}
            )
        model_post = model_consumption.end_task(task_id)
        if adapter_post is not None:
            task_root.write_json(ADAPTER_POST_USE_FILENAME, adapter_post)
        if adapter_final is not None:
            task_root.write_json(ADAPTER_FINAL_FILENAME, adapter_final)
        task_root.write_json(MODEL_POST_USE_FILENAME, model_post)
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(
            f"post-use consumption authority failed: {error}"
        ) from error

    try:
        consumption_chain = model_authority.build_consumption_chain(
            task_id=task_id,
            model_capture_digest=model_consumption.capture_digest,
            model_pre_use_digest=model_pre["use_digest"],
            model_post_use_digest=model_post["use_digest"],
            adapter_capture_digest=(
                adapter_consumption.capture_digest
                if adapter_consumption is not None
                else None
            ),
            adapter_pre_use_digest=(
                adapter_pre["use_digest"] if adapter_pre is not None else None
            ),
            adapter_post_use_digest=(
                adapter_post["use_digest"] if adapter_post is not None else None
            ),
            adapter_final_digest=(
                adapter_final["adapter_final_digest"]
                if adapter_final is not None
                else None
            ),
            native_inference_receipt_digest=native_digest,
            consumption_input_digest=consumption_input[
                "consumption_input_digest"
            ],
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    task_root.write_json(CONSUMPTION_CHAIN_FILENAME, consumption_chain)
    model_consumption.record_task_consumption(
        consumption_chain["consumption_digest"]
    )
    task_root.write(STDOUT_FILENAME, observation["stdout"])
    task_root.write(STDERR_FILENAME, observation["stderr"])
    process_receipt = build_process_receipt(
        input_receipt=input_receipt,
        observation=observation,
        request_path=input_receipt_path,
        staging_path=staging_path,
        consumption_digest=consumption_chain["consumption_digest"],
    )
    task_root.write_json(PROCESS_RECEIPT_FILENAME, process_receipt)
    try:
        replayed_input, _input_identity = task_root.read(
            INPUT_RECEIPT_FILENAME,
            label="sealed task input receipt",
            expected_sha256=bytes_sha256(expected_input_bytes),
        )
        input_receipt_unchanged = replayed_input == expected_input_bytes
    except DecodedEvaluationExecutorError:
        input_receipt_unchanged = False
    if not input_receipt_unchanged:
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="input_receipt_mutated",
            failure_detail="decoder changed or removed its sealed input receipt",
            consumption_chain=consumption_chain,
        )
    if observation["return_code"] != 0:
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind=(
                "output_already_exists" if output_preexists else "decoder_nonzero"
            ),
            failure_detail=(
                "create-only final output path already exists"
                if output_preexists
                else f"decoder returned {observation['return_code']}"
            ),
            consumption_chain=consumption_chain,
        )
    staging_authority: _RetainedTaskMedia | None = None
    try:
        staging_authority = _RetainedTaskMedia.capture(
            task_root,
            name=STAGING_VIDEO_FILENAME,
            label="decoder staging output",
        )
        staging_identity = dict(staging_authority.identity)
        staging_sha256 = staging_authority.sha256
        staging_size = staging_authority.size
        if verify_tools:
            _validate_retained_staging_against_native(
                native_receipt=native_value,
                staging_sha256=staging_sha256,
                staging_size=staging_size,
                staging_identity=staging_identity,
            )
        probe_result = validate_probe_result(
            probe_video(
                staging_authority.consumer_path(production_mode=verify_tools)
            )
        )
        if verify_tools and native_inference_evidence["receipt_digest"] != native_digest:
            raise DecodedEvaluationExecutorError(
                "native inference receipt changed across the post-use close"
            )
    except Exception as error:
        if staging_authority is not None:
            staging_authority.close()
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="media_validation_failed",
            failure_detail=f"{type(error).__name__}: {error}",
            consumption_chain=consumption_chain,
        )
    try:
        publication_gate = model_authority.build_publication_gate(
            consumption_chain=consumption_chain,
            staging_path=staging_path,
            staging_sha256=staging_sha256,
            staging_size=staging_size,
        )
        task_root.write_json(PUBLICATION_GATE_FILENAME, publication_gate)
    except Exception as error:
        staging_authority.close()
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="consumption_publication_gate_failed",
            failure_detail=str(error),
            consumption_chain=consumption_chain,
        )
    try:
        staging_authority.replay(rehash=True)
        published_inode_identity = _publish_gated_staging_inode(
            staging_path=staging_path,
            final_path=final_path,
            publication_gate=publication_gate,
            production_mode=verify_tools,
            staging_directory=task_root,
            final_directory=output_parent,
        )
        staging_published, _staging_sha, _staging_size = (
            task_root.stable_file(
                STAGING_VIDEO_FILENAME,
                label="sealed staging output",
                expected_sha256=staging_sha256,
                expected_size=staging_size,
                expected_identity=published_inode_identity,
                expected_nlink=2,
                expected_mode=0o444,
            )
        )
        final_published, _final_sha, _final_size = output_parent.stable_file(
            relative_output.name,
            label="sealed final output",
            expected_sha256=staging_sha256,
            expected_size=staging_size,
            expected_identity=published_inode_identity,
            expected_nlink=2,
            expected_mode=0o444,
        )
        if (
            staging_published != published_inode_identity
            or final_published != published_inode_identity
        ):
            raise DecodedEvaluationExecutorError(
                "published output hard-link/sealing closure differs"
            )
        staging_authority.replay_published(
            published_identity=published_inode_identity,
            final_directory=output_parent,
            final_name=relative_output.name,
        )
    except Exception as error:
        staging_authority.close()
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="output_publication_failed",
            failure_detail=f"{type(error).__name__}: {error}",
            consumption_chain=consumption_chain,
        )
    try:
        output_receipt = build_output_receipt(
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            output_path=final_path,
            probe_result=probe_result,
            native_inference_evidence=native_inference_evidence,
            consumption_chain=consumption_chain,
            publication_gate=publication_gate,
            published_inode_identity=published_inode_identity,
            output_directory=output_parent,
        )
        output_receipt = validate_output_receipt(
            output_receipt,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            output_path=final_path,
            output_directory=output_parent,
        )
        task_root.write_json(OUTPUT_RECEIPT_FILENAME, output_receipt)
    except Exception as error:
        staging_authority.close()
        return _publish_failure(
            task_root=task_root,
            output_parent=output_parent,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            staging_path=staging_path,
            final_path=final_path,
            failure_kind="output_receipt_failed",
            failure_detail=f"{type(error).__name__}: {error}",
            consumption_chain=consumption_chain,
        )
    result = {
        "task_id": task_id,
        "status": "success",
        "terminal_receipt_digest": output_receipt["output_digest"],
        "consumption_digest": consumption_chain["consumption_digest"],
        "publication_gate_digest": publication_gate["publication_gate_digest"],
        "output_relpath": record["output_relpath"],
        "result_digest": object_sha256(
            {
                "task_id": task_id,
                "status": "success",
                "terminal_receipt_digest": output_receipt["output_digest"],
                "consumption_digest": consumption_chain["consumption_digest"],
                "publication_gate_digest": publication_gate[
                    "publication_gate_digest"
                ],
                "output_relpath": record["output_relpath"],
            }
        ),
    }
    try:
        staging_authority.replay_published(
            published_identity=published_inode_identity,
            final_directory=output_parent,
            final_name=relative_output.name,
        )
        task_root.replay()
        return result
    finally:
        staging_authority.close()


def execute_task(
    *,
    evaluation_root: Path,
    task_parent: _HeldDirectory,
    directory_handles: Mapping[str, _HeldDirectory],
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    task: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    physical_bindings: Mapping[str, Any],
    model_consumption: model_authority.ModelAuthority,
    model_capture_identity: Mapping[str, Any],
    adapter_proc_fd_prefix: str | None,
    run_decoder: DecoderRunner,
    probe_video: VideoProber,
    verify_tools: bool,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one task and deterministically close any adapter authority.

    Adapter capture precedes several receipt and child-process gates.  A
    single owner wrapper ensures every exceptional exit aborts the retained
    view and closes all of its descriptors instead of relying on each gate to
    remember local cleanup.
    """

    active_adapters: list[model_authority.AdapterAuthority] = []
    try:
        return _execute_task_impl(
            evaluation_root=evaluation_root,
            task_parent=task_parent,
            directory_handles=directory_handles,
            bundle=bundle,
            shard=shard,
            task=task,
            decoder_identity=decoder_identity,
            ffprobe_identity=ffprobe_identity,
            physical_bindings_identity=physical_bindings_identity,
            physical_bindings=physical_bindings,
            model_consumption=model_consumption,
            model_capture_identity=model_capture_identity,
            adapter_proc_fd_prefix=adapter_proc_fd_prefix,
            run_decoder=run_decoder,
            probe_video=probe_video,
            verify_tools=verify_tools,
            executor_verified_release_capture=executor_verified_release_capture,
            active_adapter_authorities=active_adapters,
        )
    finally:
        for adapter in reversed(active_adapters):
            if getattr(adapter, "_closed", False):
                continue
            try:
                adapter.abort(reason="executor task failed before final replay")
            except Exception:
                pass


def build_shard_summary(
    *,
    bundle: Mapping[str, Any],
    shard: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    verify_tools: bool,
    holder_execution_authority: Mapping[str, Any] | None = None,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
    model_capture_receipt: Mapping[str, Any],
    model_final_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = [_task_id(task) for task in shard["tasks"]]
    observed_ids = [row.get("task_id") for row in results]
    if observed_ids != expected_ids:
        raise DecodedEvaluationExecutorError("shard execution result order differs")
    if any(row.get("status") not in {"success", "failure"} for row in results):
        raise DecodedEvaluationExecutorError("shard execution status differs")
    result_fields = {
        "task_id", "status", "terminal_receipt_digest", "consumption_digest",
        "publication_gate_digest", "output_relpath", "result_digest",
    }
    for row in results:
        if set(row) != result_fields:
            raise DecodedEvaluationExecutorError(
                "shard execution result field closure differs"
            )
        unsigned_result = dict(row)
        claimed_result = unsigned_result.pop("result_digest")
        if claimed_result != object_sha256(unsigned_result):
            raise DecodedEvaluationExecutorError("shard result digest differs")
        _sha(row["consumption_digest"], label="result consumption")
        if row["status"] == "success":
            _sha(row["publication_gate_digest"], label="result publication gate")
        elif row["publication_gate_digest"] is not None:
            raise DecodedEvaluationExecutorError(
                "failed result claims a publication gate"
            )
    if (
        model_capture_receipt.get("schema_version")
        != model_authority.MODEL_CAPTURE_SCHEMA
        or model_final_receipt.get("schema_version")
        != model_authority.MODEL_FINAL_SCHEMA
        or model_final_receipt.get("model_capture_digest")
        != model_capture_receipt.get("capture_digest")
        or model_final_receipt.get("task_consumption_digests")
        != [row["consumption_digest"] for row in results]
    ):
        raise DecodedEvaluationExecutorError("shard model authority differs")
    expected_holder = shard["holder"]
    if holder_execution_authority is None:
        if verify_tools:
            raise DecodedEvaluationExecutorError(
                "production shard lacks physical holder execution authority"
            )
        holder_execution = {
            "expected_job_id": expected_holder["job_id"],
            "expected_node": expected_holder["node"],
            "observed_slurm_job_id": None,
            "observed_hostname": None,
            "exact_holder_match": False,
        }
        holder_execution["holder_execution_digest"] = object_sha256(
            holder_execution
        )
    else:
        holder_execution = dict(holder_execution_authority)
        fields = {
            "expected_job_id", "expected_node", "observed_slurm_job_id",
            "observed_hostname", "exact_holder_match",
            "holder_execution_digest",
        }
        if set(holder_execution) != fields:
            raise DecodedEvaluationExecutorError(
                "holder execution authority field closure differs"
            )
        unsigned_holder = dict(holder_execution)
        claimed_holder = unsigned_holder.pop("holder_execution_digest")
        if (
            holder_execution["expected_job_id"] != expected_holder["job_id"]
            or holder_execution["expected_node"] != expected_holder["node"]
            or holder_execution["observed_slurm_job_id"]
            != expected_holder["job_id"]
            or holder_execution["observed_hostname"] != expected_holder["node"]
            or holder_execution["exact_holder_match"] is not True
            or claimed_holder != object_sha256(unsigned_holder)
        ):
            raise DecodedEvaluationExecutorError(
                "holder execution authority differs from planned job/node"
            )
    executor_capture = _capture_evidence(
        executor_verified_release_capture,
        label="shard executor verified release capture",
        allow_none=not verify_tools,
    )
    if verify_tools and (
        executor_capture is None
        or executor_capture["target"]
        != "action_preservation_decoded_eval_executor_v2.py"
    ):
        raise DecodedEvaluationExecutorError(
            "production shard lacks executor verified release capture"
        )
    if not verify_tools and executor_capture is not None:
        raise DecodedEvaluationExecutorError(
            "injected shard may not claim an executor verified capture"
        )
    value = {
        "schema_version": SHARD_SUMMARY_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "publication_digest": bundle["publication_receipt"]["publication_digest"],
        "shard_digest": shard["shard_digest"],
        "holder": shard["holder"],
        "model_capture_digest": model_capture_receipt["capture_digest"],
        "model_final_digest": model_final_receipt["model_final_digest"],
        "task_consumption_set_digest": model_final_receipt[
            "task_consumption_set_digest"
        ],
        "holder_execution_authority": holder_execution,
        "executor_verified_release_capture": executor_capture,
        "planned_task_count": shard["total_task_count"],
        "attempted_task_count": len(results),
        "success_count": sum(row["status"] == "success" for row in results),
        "failure_count": sum(row["status"] == "failure" for row in results),
        "results": [dict(row) for row in results],
        "all_tasks_attempted_exactly_once": len(results) == shard["total_task_count"],
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "failure_artifacts_retained": True,
        "execution_backend": "pinned_local_subprocess"
        if verify_tools
        else "injected_stub",
        "tool_files_verified": verify_tools,
        "training_loss_read_or_used": False,
        "network_used": False,
        "remote_launch_performed": False,
        "scientific_promotion_authorized": False,
    }
    value["summary_digest"] = object_sha256(value)
    return value


def execute_shard(
    *,
    bundle: Mapping[str, Any],
    holder_job_id: str,
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    run_decoder: DecoderRunner,
    probe_video: VideoProber,
    verify_tools: bool = True,
    holder_execution_authority: Mapping[str, Any] | None = None,
    executor_verified_release_capture: Mapping[str, Any] | None = None,
    injected_physical_bindings: Mapping[str, Any] | None = None,
    injected_model_consumption: model_authority.ModelAuthority | None = None,
    injected_proc_fd_prefix: str | None = None,
    completion_anchor_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if holder_job_id not in bundle["shards"]:
        raise DecodedEvaluationExecutorError("holder job is outside the exact four-shard plan")
    shard = bundle["shards"][holder_job_id]
    try:
        shard = plan.validate_shard(
            shard,
            manifest=bundle["manifest"],
            input_spec=bundle["input_spec"],
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    if verify_tools:
        inherited_work_root = bundle.get("_work_root_binding")
        if not isinstance(inherited_work_root, Mapping):
            raise DecodedEvaluationExecutorError(
                "production executor lacks inherited work-root authority"
            )
        try:
            bridge.verified_release.validate_inherited_work_root_binding(
                inherited_work_root,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
    _tool_identity(decoder_identity, label="decoder adapter", verify_file=verify_tools)
    _tool_identity(ffprobe_identity, label="ffprobe", verify_file=verify_tools)
    physical_identity = _artifact_identity(
        physical_bindings_identity,
        label="physical bindings",
        verify_file=verify_tools,
    )
    if verify_tools:
        try:
            bindings = bridge.load_physical_bindings(
                physical_identity["path"],
                expected_sha256=physical_identity["sha256"],
                verify_files=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        try:
            for relative_path, module_path in (
                (
                    "action_preservation_decoded_eval_executor_v2.py",
                    __file__,
                ),
                (
                    "action_preservation_decoded_eval_decoder_adapter_v1.py",
                    decoder_adapter.__file__,
                ),
                (
                    "action_preservation_decoded_eval_bridge_v1.py",
                    bridge.__file__,
                ),
                (
                    "action_preservation_decoded_eval_plan_v1.py",
                    plan.__file__,
                ),
                ("action_preservation_gate_v1.py", plan.gate.__file__),
            ):
                bridge.require_running_eval_release_member(
                    bindings["eval_release"],
                    relative_path=relative_path,
                    running_path=module_path,
                )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if (
            bindings["evaluation_id"] != bundle["manifest"]["evaluation_id"]
            or bindings["input_digest"] != bundle["input_spec"]["input_digest"]
            or bindings["manifest_digest"] != bundle["manifest"]["manifest_digest"]
        ):
            raise DecodedEvaluationExecutorError(
                "physical bindings differ from published evaluation bundle"
            )
        if any(
            bindings["runtime"][runtime_key][field] != identity[field]
            for runtime_key, identity in (
                ("decoder_adapter", decoder_identity),
                ("ffprobe", ffprobe_identity),
            )
            for field in ("path", "sha256")
        ):
            raise DecodedEvaluationExecutorError(
                "decoder/ffprobe tools differ from physical runtime authority"
            )
        pin_file_map = {
            "source_manifest_sha256": "source_manifest",
            "adapter_release_manifest_sha256": "adapter_release_manifest",
            "model_release_manifest_sha256": "model_release_manifest",
            "inference_release_manifest_sha256": "inference_release_manifest",
            "inference_config_sha256": "inference_config",
            "source_preprocessing_sha256": "source_preprocessing",
        }
        if any(
            bindings["pin_files"][file_key]["sha256"]
            != bundle["input_spec"]["pins"][pin_key]
            for pin_key, file_key in pin_file_map.items()
        ) or (
            bindings["runtime"]["infer_lora"]["sha256"]
            != bundle["input_spec"]["pins"]["inference_source_sha256"]
        ) or bindings["calibration_digest"] != bundle["input_spec"]["pins"][
            "calibration_digest"
        ]:
            raise DecodedEvaluationExecutorError(
                "physical pin files differ from evaluation input authority"
            )
        capture = _capture_evidence(
            executor_verified_release_capture,
            label="executor verified release capture",
        )
        if capture is None or capture["target"] != (
            "action_preservation_decoded_eval_executor_v2.py"
        ):
            raise DecodedEvaluationExecutorError(
                "production executor is outside the verified runtime"
            )
        expected_executor_arguments = [
            "--evaluation-root", bundle["manifest"]["evaluation_root"],
            "--holder-job-id", holder_job_id,
            "--decoder-adapter", decoder_identity["path"],
            "--decoder-adapter-sha256", decoder_identity["sha256"],
            "--ffprobe", ffprobe_identity["path"],
            "--ffprobe-sha256", ffprobe_identity["sha256"],
            "--physical-bindings", physical_identity["path"],
            "--physical-bindings-sha256", physical_identity["sha256"],
            "--confirmation",
            f"execute-local-decoded-eval-shard-v2-{holder_job_id}",
        ]
        try:
            replayed_capture = bridge.validate_verified_capture_receipt(
                bindings,
                receipt_path=capture["receipt_path"],
                target="action_preservation_decoded_eval_executor_v2.py",
                expected_arguments=expected_executor_arguments,
                expected_capture_digest=capture["capture_digest"],
                verify_file=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationExecutorError(str(error)) from error
        if replayed_capture != capture:
            raise DecodedEvaluationExecutorError(
                "executor verified capture replay differs"
            )
        release_executor_sha = next(
            item["sha256"] for item in bindings["eval_release"]["members"]
            if item["relative_path"]
            == "action_preservation_decoded_eval_executor_v2.py"
        )
        if file_sha256(__file__) != release_executor_sha:
            raise DecodedEvaluationExecutorError(
                "executor source differs from exact eval release"
            )
        if getattr(run_decoder, "verified_release_bootstrap", False) is not True:
            raise DecodedEvaluationExecutorError(
                "production decoder backend is not the verified runtime adapter"
            )
    else:
        if executor_verified_release_capture is not None:
            raise DecodedEvaluationExecutorError(
                "injected executor may not claim a verified runtime"
            )
        if (
            not isinstance(injected_physical_bindings, Mapping)
            or injected_model_consumption is None
        ):
            raise DecodedEvaluationExecutorError(
                "injected executor lacks explicit model-consumption fixtures"
            )
        bindings = dict(injected_physical_bindings)
    if verify_tools and (
        injected_physical_bindings is not None
        or injected_model_consumption is not None
        or injected_proc_fd_prefix is not None
    ):
        raise DecodedEvaluationExecutorError(
            "production executor rejects injected consumption authority"
        )
    evaluation_root = Path(bundle["manifest"]["evaluation_root"])
    directory_handles = _holder_directory_handles(
        bundle, holder_job_id=holder_job_id
    )
    execution_parent = directory_handles[plan.EXECUTION_SHARD_DIRECTORY]
    shard_root = directory_handles[
        f"{plan.EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
    ]
    try:
        completion_reservation = _HeldCompletionReservation.capture(
            bundle=bundle,
            holder_job_id=holder_job_id,
            execution_parent=execution_parent,
        )
    except Exception:
        for _relative, handle in sorted(
            directory_handles.items(),
            key=lambda item: item[0].count("/"),
            reverse=True,
        ):
            handle.close()
        raise
    model_consumption: model_authority.ModelAuthority | None = None
    resources_closed = False

    def close_owned_resources() -> None:
        nonlocal resources_closed
        if resources_closed:
            return
        resources_closed = True
        if model_consumption is not None:
            try:
                model_consumption.close()
            except Exception:
                try:
                    model_consumption.abort(
                        reason="executor resource cleanup after failure"
                    )
                except Exception:
                    pass
        try:
            completion_reservation.close()
        except Exception:
            pass
        for _relative, handle in sorted(
            directory_handles.items(),
            key=lambda item: item[0].count("/"),
            reverse=True,
        ):
            try:
                handle.close()
            except Exception:
                pass

    try:
        shard_root.write_json(
            EXECUTION_CLAIM_FILENAME,
            _execution_claim(bundle=bundle, holder_job_id=holder_job_id),
        )
    except Exception:
        close_owned_resources()
        raise
    holder_root_relative = (
        f"{plan.EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
    )
    authority_root = directory_handles[
        f"{holder_root_relative}/{CONSUMPTION_AUTHORITY_DIRECTORY}"
    ]
    task_parent = directory_handles[f"{holder_root_relative}/tasks"]
    if verify_tools:
        if (
            bindings["pin_files"]["model_release_manifest"]["sha256"]
            != model_authority.MODEL_MANIFEST_SHA256
        ):
            close_owned_resources()
            raise DecodedEvaluationExecutorError(
                "base-model content manifest is not the pinned exact-23 authority"
            )
        try:
            model_consumption = model_authority.ModelAuthority.capture(
                model_root=bindings["runtime"]["model_checkpoint_root"],
                manifest_path=bindings["pin_files"]["model_release_manifest"]["path"],
                private_parent=authority_root.path,
                private_parent_fd=authority_root.descriptor,
                view_name="model_fd_view",
                expected_uid=MODEL_FILE_UID,
                expected_gid=MODEL_FILE_GID,
                expected_device=MODEL_FILE_DEVICE,
                expected_manifest_sha256=model_authority.MODEL_MANIFEST_SHA256,
                expected_file_mode=MODEL_FILE_MODE,
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            close_owned_resources()
            raise DecodedEvaluationExecutorError(str(error)) from error
    else:
        assert injected_model_consumption is not None
        model_consumption = injected_model_consumption
    try:
        if verify_tools:
            authority_root.adopt_entries(
                authority_root.entries
                | {model_consumption.capture_receipt["private_root_name"]}
            )
        authority_root.write_json(
            MODEL_CAPTURE_FILENAME, model_consumption.capture_receipt
        )
        model_capture_path = authority_root.consumer_path(
            MODEL_CAPTURE_FILENAME, production_mode=verify_tools
        )
        model_capture_identity = {
            "path": str(model_capture_path),
            "sha256": bytes_sha256(
                canonical_json_bytes(model_consumption.capture_receipt) + b"\n"
            ),
        }
    except Exception:
        close_owned_resources()
        raise
    results: list[dict[str, Any]] = []
    try:
        for task in shard["tasks"]:
            results.append(
                execute_task(
                    evaluation_root=evaluation_root,
                    task_parent=task_parent,
                    directory_handles=directory_handles,
                    bundle=bundle,
                    shard=shard,
                    task=task,
                    decoder_identity=decoder_identity,
                    ffprobe_identity=ffprobe_identity,
                    physical_bindings_identity=physical_identity,
                    physical_bindings=bindings,
                    model_consumption=model_consumption,
                    model_capture_identity=model_capture_identity,
                    adapter_proc_fd_prefix=injected_proc_fd_prefix,
                    run_decoder=run_decoder,
                    probe_video=probe_video,
                    verify_tools=verify_tools,
                    executor_verified_release_capture=executor_verified_release_capture,
                )
            )
        model_final = model_consumption.finalize(
            expected_task_count=shard["total_task_count"]
        )
        authority_root.write_json(MODEL_FINAL_FILENAME, model_final)
    except Exception:
        try:
            aborted = model_consumption.abort(reason="shard execution failed closed")
            if verify_tools:
                authority_root.adopt_entries(
                    authority_root.entries
                    - {model_consumption.capture_receipt["private_root_name"]}
                )
            authority_root.write_json("model_consumption_abort.json", aborted)
        except Exception:
            pass
        close_owned_resources()
        raise
    try:
        summary = build_shard_summary(
            bundle=bundle,
            shard=shard,
            results=results,
            verify_tools=verify_tools,
            holder_execution_authority=holder_execution_authority,
            executor_verified_release_capture=executor_verified_release_capture,
            model_capture_receipt=model_consumption.capture_receipt,
            model_final_receipt=model_final,
        )
        shard_root.write_json(SUMMARY_FILENAME, summary)
        model_consumption.close()
        if verify_tools:
            authority_root.adopt_entries(
                authority_root.entries
                - {model_consumption.capture_receipt["private_root_name"]}
            )
        shard_root.replay(
            expected_entries={
                EXECUTION_CLAIM_FILENAME,
                "tasks",
                CONSUMPTION_AUTHORITY_DIRECTORY,
                SUMMARY_FILENAME,
            }
        )
        authority_root.replay(
            expected_entries={MODEL_CAPTURE_FILENAME, MODEL_FINAL_FILENAME}
        )
        task_parent.replay(
            expected_entries=set(_task_id(task) for task in shard["tasks"])
        )
        if summary["failure_count"] == 0:
            holder_completion = _build_holder_directory_completion(
                bundle=bundle,
                holder_job_id=holder_job_id,
                directory_handles=directory_handles,
                holder_summary_digest=summary["summary_digest"],
            )
            completion_reservation.fill(
                holder_completion,
                topology=plan.build_directory_topology(
                    bundle["manifest"], input_spec=bundle["input_spec"]
                ),
                base_directory_authority=bundle["directory_authority"],
            )
            completion_anchor = build_holder_completion_anchor(
                holder_job_id=holder_job_id,
                reservation=completion_reservation,
                completion=holder_completion,
                holder_summary_digest=summary["summary_digest"],
            )
            if completion_anchor_sink is None:
                if verify_tools:
                    raise DecodedEvaluationExecutorError(
                        "production executor lacks completion anchor channel"
                    )
            else:
                completion_anchor_sink(completion_anchor)
                completion_reservation.replay_final()
        return summary
    finally:
        close_owned_resources()


def sanitized_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in SUBPROCESS_ENV_DENYLIST:
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def subprocess_decoder_runner(
    authority: Mapping[str, Any] | str | Path,
) -> DecoderRunner:
    """Return the decoder backend.

    A mapping is the production form and is a validated physical binding.  It
    launches the decoder only through the root bootstrap/captured exact15
    runtime.  The pathname form exists solely for injected unit fixtures and
    must never be paired with ``verify_tools=True``.
    """

    bindings = dict(authority) if isinstance(authority, Mapping) else None
    executable = None if bindings is not None else str(authority)

    def run(
        request_path: Path,
        output_path: Path,
        inherited_fd_binding: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        arguments = [
            "--request", str(request_path), "--output", str(output_path)
        ]
        if bindings is not None:
            try:
                argv = bridge.verified_target_argv(
                    bindings,
                    target=(
                        "action_preservation_decoded_eval_decoder_adapter_v1.py"
                    ),
                    arguments=arguments,
                    capture_receipt_path=(
                        request_path.parent / DECODER_RUNTIME_CAPTURE_FILENAME
                    ),
                )
            except bridge.DecodedEvaluationBridgeError as error:
                raise DecodedEvaluationExecutorError(str(error)) from error
        else:
            assert executable is not None
            argv = [executable, *arguments]
        inherited = model_authority.validate_inherited_fd_binding(
            inherited_fd_binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        pass_fds = model_authority.inherited_fd_numbers(inherited)
        environment = sanitized_subprocess_environment()
        environment[model_authority.INHERITED_FD_BINDING_ENV] = (
            model_authority.inherited_fd_environment_value(inherited)
        )
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=environment,
            close_fds=True,
            pass_fds=pass_fds,
        )
        stdout, stderr = process.communicate()
        fd_inheritance = _fd_inheritance_evidence(
            inherited,
            production_mode=bindings is not None,
            spawn_performed=True,
        )
        return {
            "return_code": int(process.returncode),
            "stdout": bytes(stdout),
            "stderr": bytes(stderr),
            "fd_inheritance": fd_inheritance,
        }

    setattr(run, "verified_release_bootstrap", bindings is not None)
    return run


def ffprobe_video_prober(ffprobe_path: str | Path) -> VideoProber:
    executable = str(ffprobe_path)

    def probe(video_path: Path) -> Mapping[str, Any]:
        pass_fds: tuple[int, ...] = ()
        match = re.fullmatch(r"/proc/self/fd/([0-9]+)", str(video_path))
        if match is not None:
            descriptor = int(match.group(1))
            try:
                observed = os.fstat(descriptor)
            except OSError as error:
                raise DecodedEvaluationExecutorError(
                    "ffprobe retained media FD is unavailable"
                ) from error
            if (
                descriptor < 3
                or not stat.S_ISREG(observed.st_mode)
                or os.get_inheritable(descriptor)
            ):
                raise DecodedEvaluationExecutorError(
                    "ffprobe retained media FD differs"
                )
            pass_fds = (descriptor,)
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-show_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            shell=False,
            env=sanitized_subprocess_environment(),
            close_fds=True,
            pass_fds=pass_fds,
        )
        if completed.returncode != 0:
            raise DecodedEvaluationExecutorError(
                f"ffprobe returned {completed.returncode}: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DecodedEvaluationExecutorError("ffprobe output is invalid JSON") from error
        return parse_ffprobe_json(value)

    return probe


def local_holder_execution_authority(holder_job_id: str) -> dict[str, Any]:
    matches = [
        item for item in plan.HOLDER_ROWS if item["job_id"] == holder_job_id
    ]
    if len(matches) != 1:
        raise DecodedEvaluationExecutorError("holder job is outside the exact plan")
    expected = matches[0]
    observed_job = os.environ.get("SLURM_JOB_ID")
    observed_hostname = socket.gethostname().split(".", 1)[0]
    exact = (
        observed_job == expected["job_id"]
        and observed_hostname == expected["node"]
    )
    if not exact:
        raise DecodedEvaluationExecutorError(
            "local executor is not running on the planned holder job/node"
        )
    value: dict[str, Any] = {
        "expected_job_id": expected["job_id"],
        "expected_node": expected["node"],
        "observed_slurm_job_id": observed_job,
        "observed_hostname": observed_hostname,
        "exact_holder_match": True,
    }
    value["holder_execution_digest"] = object_sha256(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--holder-job-id", required=True)
    parser.add_argument("--decoder-adapter", required=True)
    parser.add_argument("--decoder-adapter-sha256", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffprobe-sha256", required=True)
    parser.add_argument("--physical-bindings", required=True)
    parser.add_argument("--physical-bindings-sha256", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    expected_confirmation = (
        f"execute-local-decoded-eval-shard-v2-{args.holder_job_id}"
    )
    if args.confirmation != expected_confirmation:
        raise DecodedEvaluationExecutorError("local execution confirmation differs")
    holder_execution = local_holder_execution_authority(args.holder_job_id)
    try:
        inherited_work_root = (
            bridge.verified_release.load_inherited_work_root_environment(
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        )
    except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    if inherited_work_root["target"] != (
        "action_preservation_decoded_eval_executor_v2.py"
    ):
        raise DecodedEvaluationExecutorError(
            "executor inherited work-root target differs"
        )
    bundle = load_published_bundle(
        args.evaluation_root, work_root_binding=inherited_work_root
    )
    decoder_identity = {
        "path": str(Path(args.decoder_adapter).resolve(strict=True)),
        "sha256": args.decoder_adapter_sha256,
    }
    ffprobe_identity = {
        "path": str(Path(args.ffprobe).resolve(strict=True)),
        "sha256": args.ffprobe_sha256,
    }
    physical_bindings_identity = {
        "path": str(Path(args.physical_bindings).resolve(strict=True)),
        "sha256": args.physical_bindings_sha256,
    }
    if decoder_identity["sha256"] != file_sha256(decoder_adapter.__file__):
        raise DecodedEvaluationExecutorError(
            "decoder adapter differs from the audited implementation"
        )
    try:
        bindings = bridge.load_physical_bindings(
            physical_bindings_identity["path"],
            expected_sha256=physical_bindings_identity["sha256"],
            verify_files=True,
        )
        executor_capture = bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_executor_v2.py",
            expected_arguments=list(sys.argv[1:] if argv is None else argv),
            verify_file=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationExecutorError(str(error)) from error
    summary = execute_shard(
        bundle=bundle,
        holder_job_id=args.holder_job_id,
        decoder_identity=decoder_identity,
        ffprobe_identity=ffprobe_identity,
        physical_bindings_identity=physical_bindings_identity,
        run_decoder=subprocess_decoder_runner(bindings),
        probe_video=ffprobe_video_prober(ffprobe_identity["path"]),
        verify_tools=True,
        holder_execution_authority=holder_execution,
        executor_verified_release_capture=executor_capture,
        completion_anchor_sink=(
            bridge.verified_release.publish_holder_completion_anchor
        ),
    )
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
