#!/usr/bin/env python3
"""Offline exact-264 aggregation for retained-FD decoded evaluation.

The holder's file descriptors and private FD views are intentionally gone by
the time this command runs.  Consequently this verifier never resolves a
``/proc/*/fd`` leaf and never asks the decoder to resolve a request.  It
instead closes the canonical receipts that were written while those
authorities were alive:

    holder model capture -> D0 consumption input -> task input -> native
    inference -> C (post-use/final consumption chain) -> process -> output ->
    result -> shard summary -> holder model final

Only after all four holders and all 264 tasks close does the command build the
opaque blind-review packet.  It has no scheduler, retry, upload, training-loss,
ranking, or scientific-promotion path.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_decoder_adapter_v1 as decoder_adapter
import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as model_authority
import action_preservation_decoded_eval_plan_v1 as plan


AGGREGATE_SCHEMA = "bernini-action-preservation-decoded-eval-aggregate-v3"
PRIVATE_PACKET_SCHEMA = "bernini-action-preservation-blind-private-map-v1"
PUBLIC_PACKET_SCHEMA = "blind-full-video-review-packet-v1"
AGGREGATE_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-aggregate-completion-anchor-v1"
)

AGGREGATE_FILENAME = "evaluation_complete.json"
PRIVATE_FILENAME = "private_blind_mapping.json"
PUBLIC_FILENAME = "blind_review_packet.json"
MEDIA_DIRECTORY = "media"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = {
    "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
    "blocks", "mtime_ns", "ctime_ns",
}
_MODEL_CAPTURE_FIELDS = {
    "schema_version", "model_root", "model_view_root", "executor_pid",
    "manifest", "expected_uid", "expected_gid", "expected_device",
    "expected_file_mode", "file_count", "source_directory_count",
    "view_directory_count", "files", "source_directories",
    "view_directories", "view_links", "files_digest",
    "private_parent", "private_root_name",
    "view_created_only_via_held_parent_fd",
    "source_directories_digest", "view_directories_digest",
    "view_links_digest", "initial_replay_digest",
    "same_fd_double_hash_complete", "full_identity_captured",
    "file_and_directory_fds_retained", "fd_view_leaf_target_kind",
    "capture_digest",
}
_ADAPTER_CAPTURE_FIELDS = {
    "schema_version", "task_id", "checkpoint_root", "adapter_view_root",
    "executor_pid", "file_count", "source_directory_count",
    "view_directory_count", "files", "source_directories",
    "view_directories", "view_links", "files_digest",
    "private_parent", "private_root_name",
    "view_created_only_via_held_parent_fd",
    "source_directories_digest", "view_directories_digest",
    "view_links_digest", "initial_replay_digest",
    "same_fd_double_hash_complete", "full_identity_captured",
    "file_and_directory_fds_retained", "fd_view_leaf_target_kind",
    "safetensors_consumption_path",
    "safetensors_consumption_is_explicit_executor_proc_fd_view",
    "capture_digest",
}


class DecodedEvaluationAggregateError(RuntimeError):
    """Stored evidence is incomplete, mixed, non-canonical, or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return plan.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationAggregateError(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _closed(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DecodedEvaluationAggregateError(f"{label} field closure differs")
    return dict(value)


def _digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    claimed = _sha(value.get(field), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if object_sha256(unsigned) != claimed:
        raise DecodedEvaluationAggregateError(f"{label} digest differs")
    return claimed


def _absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise DecodedEvaluationAggregateError(f"{label} path differs")
    path = Path(value)
    if (
        not path.is_absolute()
        or value == os.path.sep
        or os.path.normpath(value) != value
    ):
        raise DecodedEvaluationAggregateError(f"{label} path differs")
    return path


def _plain(path: Path, *, directory: bool, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationAggregateError(f"{label} does not exist") from error
    wanted = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not wanted or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationAggregateError(f"{label} is not a plain artifact")
    return path


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DecodedEvaluationAggregateError(
            "safe retained directory descriptors are unavailable"
        )
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
    )


def _directory_identity(value: os.stat_result) -> dict[str, int]:
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


def _directory_inode(value: os.stat_result | Mapping[str, int]) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        return tuple(
            int(value[field])
            for field in ("device", "inode", "uid", "gid", "rdev")
        )
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_rdev),
    )


class _HeldPublicationDirectory:
    """A directory namespace that is never re-resolved after capture.

    Mutations are performed relative to ``descriptor``.  Before every
    mutation the previous full identity and exact entry set are replayed; the
    resulting identity is then captured together with the named binding.  A
    rename-out/same-name replacement therefore cannot redirect any write.
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
        parent_path: Path | None = None,
        parent_identity: Mapping[str, int] | None = None,
        parent: "_HeldPublicationDirectory | None" = None,
        owns_parent_descriptor: bool = False,
    ) -> None:
        self.path = path
        self.name = name
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.identity = dict(identity)
        self.entries = set(entries)
        self.parent_path = parent_path
        self.parent_identity = (
            None if parent_identity is None else dict(parent_identity)
        )
        self.parent = parent
        self.owns_parent_descriptor = owns_parent_descriptor
        self.closed = False

    @classmethod
    def create_root(cls, path: Path) -> "_HeldPublicationDirectory":
        if (
            not path.is_absolute()
            or str(path) == os.path.sep
            or os.path.normpath(str(path)) != str(path)
            or path.name in {"", ".", ".."}
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate root must be normalized and absolute"
            )
        parent_path = _plain(
            path.parent, directory=True, label="aggregate parent"
        )
        parent_descriptor = os.open(parent_path, _directory_flags())
        descriptor: int | None = None
        try:
            os.set_inheritable(parent_descriptor, False)
            parent_before = os.fstat(parent_descriptor)
            parent_named = parent_path.lstat()
            if (
                not stat.S_ISDIR(parent_before.st_mode)
                or _directory_identity(parent_before)
                != _directory_identity(parent_named)
            ):
                raise DecodedEvaluationAggregateError(
                    "aggregate parent identity differs"
                )
            try:
                os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise DecodedEvaluationAggregateError(
                    "aggregate root is not fresh"
                )
            os.mkdir(path.name, 0o700, dir_fd=parent_descriptor)
            descriptor = os.open(
                path.name, _directory_flags(), dir_fd=parent_descriptor
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            named = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o700
                or _directory_identity(before) != _directory_identity(named)
                or os.listdir(descriptor)
            ):
                raise DecodedEvaluationAggregateError(
                    "fresh aggregate root identity differs"
                )
            os.fsync(parent_descriptor)
            parent_after = os.fstat(parent_descriptor)
            parent_named_after = parent_path.lstat()
            if (
                _directory_inode(parent_after) != _directory_inode(parent_before)
                or _directory_identity(parent_after)
                != _directory_identity(parent_named_after)
            ):
                raise DecodedEvaluationAggregateError(
                    "aggregate parent changed during root creation"
                )
            return cls(
                path=path,
                name=path.name,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                identity=_directory_identity(before),
                entries=set(),
                parent_path=parent_path,
                parent_identity=_directory_identity(parent_after),
                owns_parent_descriptor=True,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
            raise

    @classmethod
    def create_root_from_work_binding(
        cls,
        path: Path,
        *,
        work_root_binding: Mapping[str, Any],
    ) -> "_HeldPublicationDirectory":
        """Create the aggregate root only below the inherited work-root FD."""

        try:
            binding = bridge.verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        parent_path = Path(binding["path"])
        if (
            not path.is_absolute()
            or os.path.normpath(str(path)) != str(path)
            or path.parent != parent_path
            or path.name in {"", ".", ".."}
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate root is outside the inherited work root"
            )
        parent_descriptor = os.dup(binding["root_fd"])
        descriptor: int | None = None
        try:
            os.set_inheritable(parent_descriptor, False)
            parent_before = os.fstat(parent_descriptor)
            named_parent = parent_path.lstat()
            if (
                _directory_identity(parent_before)
                != _directory_identity(named_parent)
                or not stat.S_ISDIR(parent_before.st_mode)
            ):
                raise DecodedEvaluationAggregateError(
                    "inherited aggregate parent identity differs"
                )
            try:
                os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise DecodedEvaluationAggregateError(
                    "aggregate root is not fresh"
                )
            os.mkdir(path.name, 0o700, dir_fd=parent_descriptor)
            descriptor = os.open(
                path.name, _directory_flags(), dir_fd=parent_descriptor
            )
            os.set_inheritable(descriptor, False)
            root_now = os.fstat(descriptor)
            named_root = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.fsync(parent_descriptor)
            parent_after = os.fstat(parent_descriptor)
            named_parent_after = parent_path.lstat()
            if (
                not stat.S_ISDIR(root_now.st_mode)
                or stat.S_IMODE(root_now.st_mode) != 0o700
                or _directory_identity(root_now)
                != _directory_identity(named_root)
                or os.listdir(descriptor)
                or _directory_inode(parent_after)
                != _directory_inode(parent_before)
                or _directory_identity(parent_after)
                != _directory_identity(named_parent_after)
            ):
                raise DecodedEvaluationAggregateError(
                    "inherited aggregate root creation replay differs"
                )
            return cls(
                path=path,
                name=path.name,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                identity=_directory_identity(root_now),
                entries=set(),
                parent_path=parent_path,
                parent_identity=_directory_identity(parent_after),
                owns_parent_descriptor=True,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
            raise

    def _ensure_open(self) -> None:
        if self.closed:
            raise DecodedEvaluationAggregateError(
                "retained publication directory is closed"
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
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if self.parent is not None:
                parent_before = os.fstat(self.parent.descriptor)
                parent_named = os.stat(
                    self.parent.name,
                    dir_fd=self.parent.parent_descriptor,
                    follow_symlinks=False,
                )
                parent_expected = self.parent.identity
            else:
                assert self.parent_path is not None
                assert self.parent_identity is not None
                parent_before = os.fstat(self.parent_descriptor)
                parent_named = self.parent_path.lstat()
                parent_expected = self.parent_identity
        except OSError as error:
            raise DecodedEvaluationAggregateError(
                "retained publication directory replay failed"
            ) from error
        observed = _directory_identity(before)
        if (
            observed != self.identity
            or _directory_identity(middle) != self.identity
            or _directory_identity(after) != self.identity
            or _directory_identity(named) != self.identity
            or _directory_identity(parent_before) != parent_expected
            or _directory_identity(parent_named) != parent_expected
            or sorted(first) != sorted(second)
            or sorted(first) != sorted(expected)
            or len(first) != len(expected)
            or os.get_inheritable(self.descriptor)
        ):
            raise DecodedEvaluationAggregateError(
                "retained publication directory identity or closure differs"
            )

    def _refresh_after_mutation(self, expected_entries: set[str]) -> None:
        try:
            before = os.fstat(self.descriptor)
            first = os.listdir(self.descriptor)
            middle = os.fstat(self.descriptor)
            second = os.listdir(self.descriptor)
            after = os.fstat(self.descriptor)
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise DecodedEvaluationAggregateError(
                "publication directory mutation replay failed"
            ) from error
        identities = [
            _directory_identity(item)
            for item in (before, middle, after, named)
        ]
        if (
            any(_directory_inode(item) != _directory_inode(self.identity) for item in identities)
            or len({tuple(item.values()) for item in identities}) != 1
            or sorted(first) != sorted(second)
            or sorted(first) != sorted(expected_entries)
            or len(first) != len(expected_entries)
        ):
            raise DecodedEvaluationAggregateError(
                "publication directory mutation escaped its retained inode"
            )
        self.identity = identities[0]
        self.entries = set(expected_entries)
        if self.parent is not None:
            self.parent._refresh_after_mutation(self.parent.entries)

    def mkdir(self, name: str, *, mode: int = 0o700) -> "_HeldPublicationDirectory":
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise DecodedEvaluationAggregateError(
                "publication directory basename differs"
            )
        self.replay()
        try:
            os.mkdir(name, mode, dir_fd=self.descriptor)
        except FileExistsError as error:
            raise DecodedEvaluationAggregateError(
                "publication child directory is not fresh"
            ) from error
        child_descriptor = os.open(
            name, _directory_flags(), dir_fd=self.descriptor
        )
        try:
            os.set_inheritable(child_descriptor, False)
            child = os.fstat(child_descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(child.st_mode)
                or stat.S_IMODE(child.st_mode) != mode
                or _directory_identity(child) != _directory_identity(named)
                or os.listdir(child_descriptor)
            ):
                raise DecodedEvaluationAggregateError(
                    "publication child directory identity differs"
                )
            os.fsync(self.descriptor)
            self._refresh_after_mutation(self.entries | {name})
            return _HeldPublicationDirectory(
                path=self.path / name,
                name=name,
                descriptor=child_descriptor,
                parent_descriptor=self.descriptor,
                identity=_directory_identity(child),
                entries=set(),
                parent=self,
            )
        except Exception:
            os.close(child_descriptor)
            raise

    def write(self, name: str, payload: bytes, *, mode: int) -> Path:
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise DecodedEvaluationAggregateError(
                "publication artifact basename differs"
            )
        self.replay()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, mode, dir_fd=self.descriptor)
        except FileExistsError as error:
            raise DecodedEvaluationAggregateError(
                f"refusing to overwrite: {self.path / name}"
            ) from error
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise DecodedEvaluationAggregateError(
                        "create-only write made no progress"
                    )
                offset += written
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            before = os.fstat(descriptor)
            digest, size = executor._hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
            expected_sha = hashlib.sha256(payload).hexdigest()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != mode
                or executor._stat_identity_row(before)
                != executor._stat_identity_row(after)
                or executor._stat_identity_row(before)
                != executor._stat_identity_row(named)
                or digest != expected_sha
                or size != len(payload)
            ):
                raise DecodedEvaluationAggregateError(
                    "create-only artifact replay differs"
                )
        finally:
            os.close(descriptor)
        os.fsync(self.descriptor)
        self._refresh_after_mutation(self.entries | {name})
        return self.path / name

    def seal(self, *, mode: int, expected_entries: set[str]) -> None:
        self.replay(expected_entries=expected_entries)
        os.fchmod(self.descriptor, mode)
        os.fsync(self.descriptor)
        current = os.fstat(self.descriptor)
        named = os.stat(
            self.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _directory_inode(current) != _directory_inode(self.identity)
            or _directory_identity(current) != _directory_identity(named)
            or stat.S_IMODE(current.st_mode) != mode
        ):
            raise DecodedEvaluationAggregateError(
                "sealed publication directory identity differs"
            )
        self.identity = _directory_identity(current)
        self.entries = set(expected_entries)
        self.replay(expected_entries=expected_entries)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.descriptor)
        if self.owns_parent_descriptor:
            os.close(self.parent_descriptor)


def _validate_sealed_publication_pair(
    *,
    staging_path: Path,
    final_path: Path,
    expected_identity: Mapping[str, Any],
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> dict[str, int]:
    """Authenticate both hard-link names and bytes through retained FDs.

    Both names are opened before hashing.  All byte reads use the retained
    staging descriptor, and both descriptors plus both names are replayed
    before and after the double hash.  A pathname replacement can therefore
    never splice a cached identity from one inode to bytes read from another.
    """

    try:
        identity = executor._published_inode_identity(expected_identity)
    except executor.DecodedEvaluationExecutorError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    expected_sha256 = _sha(expected_sha256, label=f"{label} expected SHA")
    if type(expected_size) is not int or expected_size <= 0:
        raise DecodedEvaluationAggregateError(f"{label} expected size differs")
    if staging_path == final_path:
        raise DecodedEvaluationAggregateError(f"{label} names are not distinct")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    staging_descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        staging_descriptor = os.open(staging_path, flags)
        final_descriptor = os.open(final_path, flags)
        os.set_inheritable(staging_descriptor, False)
        os.set_inheritable(final_descriptor, False)
        staging_before = os.fstat(staging_descriptor)
        final_before = os.fstat(final_descriptor)
        named_staging_before = staging_path.lstat()
        named_final_before = final_path.lstat()
        first_sha, first_size = executor._hash_fd(staging_descriptor)
        staging_middle = os.fstat(staging_descriptor)
        final_middle = os.fstat(final_descriptor)
        second_sha, second_size = executor._hash_fd(staging_descriptor)
        staging_after = os.fstat(staging_descriptor)
        final_after = os.fstat(final_descriptor)
        named_staging_after = staging_path.lstat()
        named_final_after = final_path.lstat()
    except OSError as error:
        raise DecodedEvaluationAggregateError(
            f"cannot replay {label}: {error}"
        ) from error
    finally:
        if final_descriptor is not None:
            os.close(final_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)

    observed = (
        staging_before,
        final_before,
        named_staging_before,
        named_final_before,
        staging_middle,
        final_middle,
        staging_after,
        final_after,
        named_staging_after,
        named_final_after,
    )
    if (
        any(not stat.S_ISREG(item.st_mode) for item in observed)
        or any(executor._stat_identity_row(item) != identity for item in observed)
        or first_sha != expected_sha256
        or second_sha != expected_sha256
        or first_size != expected_size
        or second_size != expected_size
    ):
        raise DecodedEvaluationAggregateError(
            f"{label} held-FD bytes/inode or named replay differs"
        )
    return identity


class _RetainedMediaFile:
    """One authenticated media inode retained through reprobe and copying."""

    def __init__(
        self,
        *,
        path: Path,
        descriptor: int,
        parent_descriptor: int,
        identity: Mapping[str, Any],
        sha256: str,
        size: int,
        owns_parent_descriptor: bool,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.identity = dict(identity)
        self.sha256 = sha256
        self.size = size
        self.owns_parent_descriptor = owns_parent_descriptor
        self.closed = False

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int | None = None,
        expected_identity: Mapping[str, Any] | None = None,
        expected_nlink: set[int] = {1},
        parent_descriptor: int | None = None,
    ) -> "_RetainedMediaFile":
        expected_sha256 = _sha(
            expected_sha256, label="retained media expected SHA"
        )
        if (
            not path.is_absolute()
            or os.path.normpath(str(path)) != str(path)
            or path.name in {"", ".", ".."}
        ):
            raise DecodedEvaluationAggregateError(
                "retained media path differs"
            )
        owns_parent = parent_descriptor is None
        parent_fd = (
            os.open(path.parent, _directory_flags())
            if parent_descriptor is None
            else parent_descriptor
        )
        descriptor: int | None = None
        try:
            os.set_inheritable(parent_fd, False)
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = executor._hash_fd(descriptor)
            middle = os.fstat(descriptor)
            second_sha, second_size = executor._hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False
            )
            observed = executor._stat_identity_row(before)
            expected = None
            if expected_identity is not None:
                expected = dict(
                    _closed(
                        expected_identity,
                        _IDENTITY_FIELDS,
                        label="retained media expected identity",
                    )
                )
                if any(
                    type(expected[field]) is not int
                    or expected[field] < 0
                    for field in _IDENTITY_FIELDS
                ):
                    raise DecodedEvaluationAggregateError(
                        "retained media expected identity differs"
                    )
                if not stat.S_IFMT(expected["mode"]):
                    expected["mode"] = stat.S_IFREG | expected["mode"]
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink not in expected_nlink
                or executor._stat_identity_row(middle) != observed
                or executor._stat_identity_row(after) != observed
                or executor._stat_identity_row(named) != observed
                or (expected is not None and observed != expected)
                or first_sha != expected_sha256
                or second_sha != expected_sha256
                or first_size != second_size
                or (
                    expected_size is not None
                    and first_size != expected_size
                )
            ):
                raise DecodedEvaluationAggregateError(
                    "retained media same-FD capture differs"
                )
            result = cls(
                path=path,
                descriptor=descriptor,
                parent_descriptor=parent_fd,
                identity=observed,
                sha256=expected_sha256,
                size=first_size,
                owns_parent_descriptor=owns_parent,
            )
            descriptor = None
            if owns_parent:
                parent_fd = -1
            return result
        except executor.DecodedEvaluationExecutorError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if owns_parent and parent_fd >= 0:
                os.close(parent_fd)

    def replay(self, *, rehash: bool) -> None:
        if self.closed:
            raise DecodedEvaluationAggregateError(
                "retained media authority is closed"
            )
        before = os.fstat(self.descriptor)
        named = os.stat(
            self.path.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            executor._stat_identity_row(before) != self.identity
            or executor._stat_identity_row(named) != self.identity
            or os.get_inheritable(self.descriptor)
        ):
            raise DecodedEvaluationAggregateError(
                "retained media inode/name replay differs"
            )
        if rehash:
            first_sha, first_size = executor._hash_fd(self.descriptor)
            middle = os.fstat(self.descriptor)
            second_sha, second_size = executor._hash_fd(self.descriptor)
            after = os.fstat(self.descriptor)
            named_after = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                executor._stat_identity_row(middle) != self.identity
                or executor._stat_identity_row(after) != self.identity
                or executor._stat_identity_row(named_after) != self.identity
                or first_sha != self.sha256
                or second_sha != self.sha256
                or first_size != self.size
                or second_size != self.size
            ):
                raise DecodedEvaluationAggregateError(
                    "retained media byte replay differs"
                )

    def consumer_path(self) -> Path:
        if (
            self.closed
            or sys.platform != "linux"
            or not Path("/proc/self/fd").is_dir()
        ):
            raise DecodedEvaluationAggregateError(
                "retained media proc-FD consumption is unavailable"
            )
        return Path(f"/proc/self/fd/{self.descriptor}")

    def copy_to(
        self,
        *,
        destination_directory: _HeldPublicationDirectory,
        basename: str,
    ) -> None:
        self.replay(rehash=False)
        destination_directory.replay()
        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            output = os.open(
                basename,
                flags,
                0o444,
                dir_fd=destination_directory.descriptor,
            )
        except OSError as error:
            raise DecodedEvaluationAggregateError(
                "retained media destination is not fresh"
            ) from error
        digest = hashlib.sha256()
        try:
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            while True:
                block = os.read(self.descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                offset = 0
                while offset < len(block):
                    count = os.write(output, block[offset:])
                    if count <= 0:
                        raise DecodedEvaluationAggregateError(
                            "retained media copy made no progress"
                        )
                    offset += count
            os.fchmod(output, 0o444)
            os.fsync(output)
            output_before = os.fstat(output)
            output_sha, output_size = executor._hash_fd(output)
            output_after = os.fstat(output)
            output_named = os.stat(
                basename,
                dir_fd=destination_directory.descriptor,
                follow_symlinks=False,
            )
        finally:
            os.close(output)
        self.replay(rehash=True)
        if (
            digest.hexdigest() != self.sha256
            or output_sha != self.sha256
            or output_size != self.size
            or output_before.st_nlink != 1
            or stat.S_IMODE(output_before.st_mode) != 0o444
            or executor._stat_identity_row(output_before)
            != executor._stat_identity_row(output_after)
            or executor._stat_identity_row(output_before)
            != executor._stat_identity_row(output_named)
        ):
            raise DecodedEvaluationAggregateError(
                "retained media destination replay differs"
            )
        os.fsync(destination_directory.descriptor)
        destination_directory._refresh_after_mutation(
            destination_directory.entries | {basename}
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.descriptor)
        if self.owns_parent_descriptor:
            os.close(self.parent_descriptor)


def _bytes(path: Path, *, label: str) -> bytes:
    try:
        raw, _ = bridge._stable_file(path, label=label)
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    return raw


def _json_with_sha(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    raw = _bytes(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationAggregateError(
            f"cannot decode {label}: {error}"
        ) from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        raise DecodedEvaluationAggregateError(f"{label} is not canonical JSON")
    return dict(value), hashlib.sha256(raw).hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    return _json_with_sha(path, label=label)[0]


def _capture_holder_completion_documents(
    bundle: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read the four filled reserved inodes before deriving final topology."""

    root = bundle.get("_evaluation_root_handle")
    if not isinstance(root, executor._HeldDirectory):
        raise DecodedEvaluationAggregateError(
            "evaluation root retained authority is absent"
        )
    rows = {
        row["relative_path"]: dict(row)
        for row in bundle["directory_authority"]["rows"]
    }
    parent_row = rows.get(plan.EXECUTION_SHARD_DIRECTORY)
    if parent_row is None:
        raise DecodedEvaluationAggregateError(
            "holder completion parent authority is absent"
        )
    try:
        reservations = plan.validate_holder_completion_reservations(
            bundle["publication_receipt"][
                "holder_completion_reservations"
            ],
            evaluation_root=bundle["manifest"]["evaluation_root"],
            materialized_required=True,
            directory_authority=bundle["directory_authority"],
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    root.replay()
    parent = root.open_child(
        plan.EXECUTION_SHARD_DIRECTORY,
        label="holder completion reservation directory",
        expected_identity=parent_row["identity"],
    )
    result: dict[str, dict[str, Any]] = {}
    try:
        parent.replay(expected_entries=set(parent_row["expected_entries"]))
        for reservation in reservations:
            relative = Path(reservation["relative_path"])
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    relative.name,
                    os.O_RDONLY | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent.descriptor,
                )
                os.set_inheritable(descriptor, False)
                before = os.fstat(descriptor)
                first_sha, first_size = executor._hash_fd(descriptor)
                first = os.pread(descriptor, first_size, 0)
                middle = os.fstat(descriptor)
                second_sha, second_size = executor._hash_fd(descriptor)
                second = os.pread(descriptor, second_size, 0)
                after = os.fstat(descriptor)
                named = os.stat(
                    relative.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                immutable = ("device", "inode", "uid", "gid", "rdev")
                observed = executor._stat_identity_row(before)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode)
                    != plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE
                    or any(
                        observed[field]
                        != reservation["identity"][field]
                        for field in immutable
                    )
                    or executor._stat_identity(before)
                    != executor._stat_identity(middle)
                    or executor._stat_identity(before)
                    != executor._stat_identity(after)
                    or executor._stat_identity(before)
                    != executor._stat_identity(named)
                    or first_sha != second_sha
                    or first_size != second_size
                    or first != second
                ):
                    raise DecodedEvaluationAggregateError(
                        "filled holder completion inode differs"
                    )
                try:
                    value = json.loads(first.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DecodedEvaluationAggregateError(
                        "filled holder completion JSON differs"
                    ) from error
                if (
                    not isinstance(value, Mapping)
                    or canonical_json_bytes(value) + b"\n" != first
                ):
                    raise DecodedEvaluationAggregateError(
                        "filled holder completion serialization differs"
                    )
                try:
                    completion = plan.validate_holder_directory_completion(
                        value,
                        topology=plan.build_directory_topology(
                            bundle["manifest"],
                            input_spec=bundle["input_spec"],
                        ),
                        base_directory_authority=bundle[
                            "directory_authority"
                        ],
                    )
                except plan.DecodedEvaluationPlanError as error:
                    raise DecodedEvaluationAggregateError(str(error)) from error
                holder = reservation["holder_job_id"]
                if completion["holder_job_id"] != holder:
                    raise DecodedEvaluationAggregateError(
                        "filled holder completion ownership differs"
                    )
                result[holder] = {
                    "completion": completion,
                    "file": {
                        "path": reservation["path"],
                        "sha256": first_sha,
                        "size": first_size,
                        **observed,
                    },
                }
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        parent.replay(expected_entries=set(parent_row["expected_entries"]))
    finally:
        parent.close()
    if set(result) != {row["job_id"] for row in plan.HOLDER_ROWS}:
        raise DecodedEvaluationAggregateError(
            "filled holder completion closure differs"
        )
    return result


def validate_holder_completion_anchors(
    value: Any,
    *,
    bundle: Mapping[str, Any],
    completion_documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DecodedEvaluationAggregateError(
            "holder completion anchor set differs"
        )
    holder_ids = [row["job_id"] for row in plan.HOLDER_ROWS]
    if len(value) != len(holder_ids):
        raise DecodedEvaluationAggregateError(
            "holder completion anchor count differs"
        )
    try:
        reservations = plan.validate_holder_completion_reservations(
            bundle["publication_receipt"][
                "holder_completion_reservations"
            ],
            evaluation_root=bundle["manifest"]["evaluation_root"],
            materialized_required=True,
            directory_authority=bundle["directory_authority"],
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    reservation_by_holder = {
        row["holder_job_id"]: row for row in reservations
    }
    result: dict[str, dict[str, Any]] = {}
    for expected_holder, raw in zip(holder_ids, value):
        try:
            anchor = executor.validate_holder_completion_anchor(raw)
        except executor.DecodedEvaluationExecutorError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        reservation = reservation_by_holder[expected_holder]
        document = completion_documents[expected_holder]
        initial = {
            field: reservation["identity"][field]
            for field in ("device", "inode", "uid", "gid", "rdev")
        }
        if (
            anchor["holder_job_id"] != expected_holder
            or anchor["completion_path"] != reservation["path"]
            or anchor["initial_inode_identity"] != initial
            or anchor["completion_sha256"] != document["file"]["sha256"]
            or anchor["completion_size"] != document["file"]["size"]
            or anchor["completion_mode"]
            != stat.S_IMODE(document["file"]["mode"])
            or anchor["completion_digest"]
            != document["completion"]["completion_digest"]
            or anchor["holder_summary_digest"]
            != document["completion"]["holder_summary_digest"]
        ):
            raise DecodedEvaluationAggregateError(
                "holder completion dynamic anchor differs"
            )
        result[expected_holder] = anchor
    return result


def _open_final_publication_authority(
    *,
    bundle: Mapping[str, Any],
    bindings: Mapping[str, Any],
    work_root_binding: Mapping[str, Any],
    completion_documents: Mapping[str, Mapping[str, Any]],
) -> tuple[plan.RetainedPublicationRoot, dict[str, Any]]:
    initial_topology = plan.build_directory_topology(
        bundle["manifest"], input_spec=bundle["input_spec"]
    )
    authority: plan.RetainedPublicationRoot | None = None
    try:
        merge = plan.merge_holder_directory_completions(
            topology=initial_topology,
            base_directory_authority=bundle["directory_authority"],
            completions={
                holder: row["completion"]
                for holder, row in completion_documents.items()
            },
        )
        authority = plan.RetainedPublicationRoot.open_materialized(
            bundle["manifest"]["evaluation_root"],
            directory_authority=merge["directory_authority"],
            topology=merge["topology"],
            holder_job_id=None,
            label="aggregate final evaluation tree",
            error_type=DecodedEvaluationAggregateError,
            retained_parent_fd=work_root_binding["root_fd"],
            retained_parent_parent_fd=work_root_binding["parent_fd"],
            expected_parent_immutable_identity=work_root_binding[
                "root_immutable_identity"
            ],
            expected_parent_parent_immutable_identity=work_root_binding[
                "parent_immutable_identity"
            ],
            expected_root_authority=bindings["evaluation_publication"][
                "root_authority"
            ],
        )
        replayed = authority.capture_filled_holder_completions(
            bundle["publication_receipt"],
            topology=initial_topology,
            base_directory_authority=bundle["directory_authority"],
        )
    except (
        plan.DecodedEvaluationPlanError,
        DecodedEvaluationAggregateError,
        KeyError,
    ) as error:
        if authority is not None:
            authority.close()
        if isinstance(error, DecodedEvaluationAggregateError):
            raise
        raise DecodedEvaluationAggregateError(str(error)) from error
    if any(
        replayed[holder]["completion"] != row["completion"]
        or replayed[holder]["file"]["sha256"]
        != row["file"]["sha256"]
        for holder, row in completion_documents.items()
    ):
        authority.close()
        raise DecodedEvaluationAggregateError(
            "filled holder completion changed before final tree capture"
        )
    return authority, merge


def _read_final_tree_member(
    authority: plan.RetainedPublicationRoot,
    *,
    evaluation_root: Path,
    path: Path,
    label: str,
    expected_mode: int = 0o400,
) -> tuple[bytes, dict[str, Any]]:
    """Consume one sealed evidence file from an already retained directory."""

    try:
        relative = path.relative_to(evaluation_root)
    except ValueError as error:
        raise DecodedEvaluationAggregateError(
            f"{label} escapes the final evaluation tree"
        ) from error
    if (
        relative == Path(".")
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise DecodedEvaluationAggregateError(f"{label} path differs")
    parent_relative = (
        "." if relative.parent == Path(".") else relative.parent.as_posix()
    )
    parent_fd = authority.directory_fd(parent_relative)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first_sha, first_size = executor._hash_fd(descriptor)
        first = os.pread(descriptor, first_size, 0)
        middle = os.fstat(descriptor)
        second_sha, second_size = executor._hash_fd(descriptor)
        second = os.pread(descriptor, second_size, 0)
        after = os.fstat(descriptor)
        named = os.stat(
            relative.name, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        raise DecodedEvaluationAggregateError(
            f"cannot read retained {label}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
        or executor._stat_identity(before) != executor._stat_identity(middle)
        or executor._stat_identity(before) != executor._stat_identity(after)
        or executor._stat_identity(before) != executor._stat_identity(named)
        or first != second
        or first_sha != second_sha
        or first_size != second_size
        or len(first) != first_size
    ):
        raise DecodedEvaluationAggregateError(
            f"retained {label} same-FD replay differs"
        )
    return first, {
        "path": str(path),
        "sha256": first_sha,
        "size": first_size,
        **executor._stat_identity_row(before),
    }


def _read_final_tree_json(
    authority: plan.RetainedPublicationRoot,
    *,
    evaluation_root: Path,
    path: Path,
    label: str,
    expected_mode: int = 0o400,
) -> tuple[dict[str, Any], str]:
    raw, binding = _read_final_tree_member(
        authority,
        evaluation_root=evaluation_root,
        path=path,
        label=label,
        expected_mode=expected_mode,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationAggregateError(
            f"cannot decode retained {label}: {error}"
        ) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) + b"\n" != raw:
        raise DecodedEvaluationAggregateError(
            f"retained {label} is not canonical JSON"
        )
    return dict(value), binding["sha256"]


def _identity(value: Any, *, label: str) -> dict[str, int]:
    row = _closed(value, _IDENTITY_FIELDS, label=label)
    if any(type(item) is not int for item in row.values()):
        raise DecodedEvaluationAggregateError(f"{label} values differ")
    return row


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise DecodedEvaluationAggregateError(f"{label} differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise DecodedEvaluationAggregateError(f"{label} differs")
    return value


def _capture_rows_by_relative(capture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["relative_path"]: dict(row) for row in capture["files"]}


def _expected_replay_digest(
    capture: Mapping[str, Any], *, stage: str, adapter: bool,
    private_parent_current_identity: Mapping[str, Any] | None = None,
) -> str:
    """Reconstruct one live retained-FD replay from its persisted capture.

    The holder's descriptors are intentionally closed at aggregate time, but
    every replay preimage is a deterministic projection of the capture rows.
    Recomputing it here prevents a self-consistent receipt from substituting an
    arbitrary replay SHA and then propagating that forgery through C.
    """

    root_key = "checkpoint_root" if adapter else "model_root"
    view_key = "adapter_view_root" if adapter else "model_view_root"
    file_rows = [
        {
            "relative_path": item["relative_path"],
            "sha256": item["sha256"],
            "identity": item["identity"],
            "view_target": capture["view_links"][item["relative_path"]],
        }
        for item in capture["files"]
    ]
    parent_identity = (
        capture["private_parent"]["identity"]
        if private_parent_current_identity is None
        else _identity(
            private_parent_current_identity,
            label="replay private-parent current identity",
        )
    )
    immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    if any(
        parent_identity[field]
        != capture["private_parent"]["identity"][field]
        for field in immutable_fields
    ):
        raise DecodedEvaluationAggregateError(
            "replay private-parent immutable identity differs"
        )
    directory_rows = [
        {
            "scope": scope,
            "relative_path": item["relative_path"],
            "identity": item["identity"],
        }
        for scope, rows in (
            ("source", capture["source_directories"]),
            (
                "view_parent",
                [{**capture["private_parent"], "identity": parent_identity}],
            ),
            ("view", capture["view_directories"]),
        )
        for item in rows
    ]
    replay = {
        "schema_version": model_authority.MODEL_REPLAY_SCHEMA,
        "stage": stage,
        "source_root": capture[root_key],
        "view_root": capture[view_key],
        "file_count": len(file_rows),
        "directory_count": len(directory_rows),
        "files_digest": object_sha256(file_rows),
        "directories_digest": object_sha256(directory_rows),
        "private_parent_current_identity": parent_identity,
        "all_retained_fds_still_open": True,
        "named_paths_replayed": True,
        "fd_view_replayed": True,
    }
    return object_sha256(replay)


def _expected_final_rehash_digest(
    capture: Mapping[str, Any], *, stage: str, adapter: bool,
    private_parent_current_identity: Mapping[str, Any] | None = None,
) -> str:
    rows = [
        {
            "relative_path": item["relative_path"],
            "sha256": item["sha256"],
            "size": item["identity"]["size"],
            "identity": item["identity"],
        }
        for item in capture["files"]
    ]
    final = {
        "schema_version": (
            model_authority.ADAPTER_FINAL_SCHEMA
            if adapter
            else model_authority.MODEL_FINAL_SCHEMA
        ),
        "stage": stage,
        "replay_digest": _expected_replay_digest(
            capture, stage=stage, adapter=adapter,
            private_parent_current_identity=private_parent_current_identity,
        ),
        "private_parent_current_identity": (
            capture["private_parent"]["identity"]
            if private_parent_current_identity is None
            else _identity(
                private_parent_current_identity,
                label="final private-parent current identity",
            )
        ),
        "file_count": len(rows),
        "fully_rehashed_rows_digest": object_sha256(rows),
        "all_expected_sha256_matched": True,
        "all_full_identities_matched": True,
    }
    return object_sha256(final)


def _validate_capture_offline(
    value: Any,
    *,
    adapter: bool,
    task_id: str | None = None,
    production_required: bool = True,
    expected_model_files: Mapping[str, str] | None = None,
    expected_model_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a capture receipt without dereferencing its now-dead FD view."""

    fields = _ADAPTER_CAPTURE_FIELDS if adapter else _MODEL_CAPTURE_FIELDS
    row = _closed(value, fields, label="adapter capture" if adapter else "model capture")
    schema = (
        model_authority.ADAPTER_CAPTURE_SCHEMA
        if adapter
        else model_authority.MODEL_CAPTURE_SCHEMA
    )
    if row["schema_version"] != schema:
        raise DecodedEvaluationAggregateError("capture schema differs")
    _digest(row, field="capture_digest", label="capture receipt")
    if type(row["executor_pid"]) is not int or row["executor_pid"] <= 1:
        raise DecodedEvaluationAggregateError("capture executor PID differs")
    root_key = "checkpoint_root" if adapter else "model_root"
    view_key = "adapter_view_root" if adapter else "model_view_root"
    source_root = _absolute(row[root_key], label="capture source root")
    view_root = _absolute(row[view_key], label="capture view root")
    private_parent = _closed(
        row["private_parent"],
        {"relative_path", "path", "authority_fd", "identity"},
        label="capture private parent",
    )
    private_parent_identity = _identity(
        private_parent["identity"], label="capture private-parent identity"
    )
    if (
        private_parent["relative_path"] != "."
        or _absolute(
            private_parent["path"], label="capture private-parent path"
        ) != view_root.parent
        or type(private_parent["authority_fd"]) is not int
        or private_parent["authority_fd"] < 3
        or not stat.S_ISDIR(private_parent_identity["mode"])
        or type(row["private_root_name"]) is not str
        or row["private_root_name"] in {"", ".", ".."}
        or "/" in row["private_root_name"]
        or "\x00" in row["private_root_name"]
        or row["private_root_name"] != view_root.name
        or row["view_created_only_via_held_parent_fd"] is not True
    ):
        raise DecodedEvaluationAggregateError(
            "capture private-parent authority differs"
        )
    expected_files = (
        tuple(model_authority.ADAPTER_RELATIVE_FILES)
        if adapter
        else tuple(model_authority.MODEL_RELATIVE_FILES)
    )
    expected_directories = (
        tuple(model_authority.ADAPTER_RELATIVE_DIRECTORIES)
        if adapter
        else tuple(model_authority.MODEL_RELATIVE_DIRECTORIES)
    )
    if (
        row["file_count"] != len(expected_files)
        or row["source_directory_count"] != len(expected_directories)
        or row["view_directory_count"] != len(expected_directories)
        or row["same_fd_double_hash_complete"] is not True
        or row["full_identity_captured"] is not True
        or row["file_and_directory_fds_retained"] is not True
        or not isinstance(row["files"], list)
        or not isinstance(row["source_directories"], list)
        or not isinstance(row["view_directories"], list)
        or not isinstance(row["view_links"], Mapping)
        or [item.get("relative_path") for item in row["files"]]
        != list(expected_files)
        or row["files_digest"] != object_sha256(row["files"])
        or row["source_directories_digest"]
        != object_sha256(row["source_directories"])
        or row["view_directories_digest"]
        != object_sha256(row["view_directories"])
        or row["view_links_digest"] != object_sha256(row["view_links"])
    ):
        raise DecodedEvaluationAggregateError("capture exact tree closure differs")
    if set(row["view_links"]) != set(expected_files):
        raise DecodedEvaluationAggregateError("capture view-link closure differs")

    file_fds: set[int] = set()
    expected_uid = row.get("expected_uid") if not adapter else None
    expected_gid = row.get("expected_gid") if not adapter else None
    expected_device = row.get("expected_device") if not adapter else None
    expected_mode = row.get("expected_file_mode") if not adapter else 0o444
    if not adapter:
        if (
            type(expected_uid) is not int
            or type(expected_gid) is not int
            or (expected_device is not None and type(expected_device) is not int)
            or type(expected_mode) is not int
        ):
            raise DecodedEvaluationAggregateError("model capture metadata differs")
        if production_required and (
            expected_uid != executor.MODEL_FILE_UID
            or expected_gid != executor.MODEL_FILE_GID
            or expected_device != executor.MODEL_FILE_DEVICE
            or expected_mode != executor.MODEL_FILE_MODE
        ):
            raise DecodedEvaluationAggregateError("model production metadata differs")
    adapter_owner: tuple[int, int, int] | None = None
    for item, relative in zip(row["files"], expected_files):
        item = _closed(
            item,
            {"relative_path", "path", "sha256", "identity", "authority_fd", "proc_fd_path"},
            label="capture file row",
        )
        if item["relative_path"] != relative:
            raise DecodedEvaluationAggregateError("capture file order differs")
        _relative_path(relative, label="captured relative path")
        expected_path = source_root.joinpath(*PurePosixPath(relative).parts)
        if _absolute(item["path"], label="captured file") != expected_path:
            raise DecodedEvaluationAggregateError("captured named file path differs")
        _sha(item["sha256"], label="captured file")
        identity = _identity(item["identity"], label="captured file identity")
        fd = item["authority_fd"]
        if type(fd) is not int or fd < 3 or fd in file_fds:
            raise DecodedEvaluationAggregateError("captured file FD differs")
        file_fds.add(fd)
        target = item["proc_fd_path"]
        if (
            not isinstance(target, str)
            or target != row["view_links"][relative]
            or not target.endswith(f"/fd/{fd}")
            or not stat.S_ISREG(identity["mode"])
            or identity["nlink"] != 1
            or identity["size"] < 0
            or stat.S_IMODE(identity["mode"]) != expected_mode
        ):
            raise DecodedEvaluationAggregateError("captured file identity differs")
        if adapter:
            owner = (identity["uid"], identity["gid"], identity["device"])
            if adapter_owner is None:
                adapter_owner = owner
            elif owner != adapter_owner:
                raise DecodedEvaluationAggregateError("adapter captured owner/device differs")
        elif (
            identity["uid"] != expected_uid
            or identity["gid"] != expected_gid
            or (expected_device is not None and identity["device"] != expected_device)
        ):
            raise DecodedEvaluationAggregateError("model captured owner/device differs")

    all_fds = {private_parent["authority_fd"], *file_fds}
    for scope, directories, base in (
        ("source", row["source_directories"], source_root),
        ("view", row["view_directories"], view_root),
    ):
        if (
            not isinstance(directories, list)
            or [item.get("relative_path") for item in directories]
            != list(expected_directories)
        ):
            raise DecodedEvaluationAggregateError(
                f"capture {scope} directory order differs"
            )
        for item, relative in zip(directories, expected_directories):
            item = _closed(
                item,
                {"relative_path", "path", "authority_fd", "identity"},
                label=f"capture {scope} directory row",
            )
            expected_path = base if relative == "." else base / relative
            if _absolute(item["path"], label=f"capture {scope} directory") != expected_path:
                raise DecodedEvaluationAggregateError(
                    f"capture {scope} directory path differs"
                )
            fd = item["authority_fd"]
            identity = _identity(
                item["identity"], label=f"capture {scope} directory identity"
            )
            if (
                type(fd) is not int
                or fd < 3
                or fd in all_fds
                or not stat.S_ISDIR(identity["mode"])
            ):
                raise DecodedEvaluationAggregateError(
                    f"capture {scope} directory FD differs"
                )
            all_fds.add(fd)

    kind = row["fd_view_leaf_target_kind"]
    if production_required:
        if kind != "inherited_proc_self_fd" or any(
            item["proc_fd_path"] != f"/proc/self/fd/{item['authority_fd']}"
            for item in row["files"]
        ):
            raise DecodedEvaluationAggregateError(
                "production capture FD-view target differs"
            )
    elif kind not in {"inherited_proc_self_fd", "injected_test_fd_prefix"}:
        raise DecodedEvaluationAggregateError("capture FD-view target kind differs")

    if adapter:
        if row["task_id"] != task_id:
            raise DecodedEvaluationAggregateError("adapter capture task differs")
        expected_safetensors = str(
            view_root / "adapter/adapter_model.safetensors"
        )
        if (
            row["safetensors_consumption_path"] != expected_safetensors
            or row[
                "safetensors_consumption_is_explicit_executor_proc_fd_view"
            ] is not (kind == "inherited_proc_self_fd")
        ):
            raise DecodedEvaluationAggregateError(
                "adapter safetensors consumption path differs"
            )
    else:
        manifest = _closed(
            row["manifest"],
            {"path", "sha256", "identity", "row_count", "ordered_rows_digest"},
            label="model content manifest binding",
        )
        _absolute(manifest["path"], label="model content manifest")
        _identity(manifest["identity"], label="model content manifest identity")
        ordered = [
            {"relative_path": item["relative_path"], "sha256": item["sha256"]}
            for item in row["files"]
        ]
        if (
            manifest["sha256"] != model_authority.MODEL_MANIFEST_SHA256
            or manifest["row_count"] != model_authority.MODEL_FILE_COUNT
            or manifest["ordered_rows_digest"] != object_sha256(ordered)
        ):
            raise DecodedEvaluationAggregateError(
                "model exact-23 manifest binding differs"
            )
        if (expected_model_files is None) is not (
            expected_model_manifest is None
        ):
            raise DecodedEvaluationAggregateError(
                "model expected-manifest closure differs"
            )
        if production_required and expected_model_files is None:
            raise DecodedEvaluationAggregateError(
                "production model expected-manifest authority is absent"
            )
        if expected_model_files is not None:
            expected_mapping = dict(expected_model_files)
            if (
                tuple(expected_mapping) != expected_files
                or manifest != dict(expected_model_manifest)
                or any(
                    item["sha256"] != expected_mapping[item["relative_path"]]
                    for item in row["files"]
                )
            ):
                raise DecodedEvaluationAggregateError(
                    "model manifest/full expected file binding differs"
                )
    initial_stage = f"adapter_capture:{task_id}" if adapter else "holder_capture"
    if row["initial_replay_digest"] != _expected_replay_digest(
        row,
        stage=initial_stage,
        adapter=adapter,
        private_parent_current_identity=private_parent_identity,
    ):
        raise DecodedEvaluationAggregateError(
            "capture initial replay digest differs"
        )
    return row


def _expected_fd_binding(
    *, task_id: str, model_capture: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
    task_publication_root: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "fd": item["authority_fd"], "scope": "model", "role": "file",
            "relative_path": item["relative_path"],
            "source_path": item["path"], "identity": item["identity"],
        }
        for item in model_capture["files"]
    ]
    model_namespace = next(
        item for item in model_capture["view_directories"]
        if item["relative_path"] == "."
    )
    rows.append(
        {
            "fd": model_namespace["authority_fd"],
            "scope": "model", "role": "namespace_root",
            "relative_path": ".", "source_path": model_namespace["path"],
            "identity": model_namespace["identity"],
        }
    )
    if adapter_capture is not None:
        rows.extend(
            {
                "fd": item["authority_fd"], "scope": "adapter", "role": "file",
                "relative_path": item["relative_path"],
                "source_path": item["path"], "identity": item["identity"],
            }
            for item in adapter_capture["files"]
        )
        adapter_namespace = next(
            item for item in adapter_capture["view_directories"]
            if item["relative_path"] == "."
        )
        rows.append(
            {
                "fd": adapter_namespace["authority_fd"],
                "scope": "adapter", "role": "namespace_root",
                "relative_path": ".",
                "source_path": adapter_namespace["path"],
                "identity": adapter_namespace["identity"],
            }
        )
    task_root = model_authority.validate_task_publication_root(
        task_publication_root, verify_open_fd=False
    )
    rows.append(
        {
            "fd": task_root["fd"], "scope": "task",
            "role": "publication_root", "relative_path": ".",
            "source_path": task_root["path"],
            "identity": task_root["identity"],
        }
    )
    rows.sort(key=lambda item: item["fd"])
    value: dict[str, Any] = {
        "schema_version": model_authority.INHERITED_FD_BINDING_SCHEMA,
        "task_id": task_id,
        "model_capture_digest": model_capture["capture_digest"],
        "adapter_capture_digest": (
            None if adapter_capture is None else adapter_capture["capture_digest"]
        ),
        "fd_count": len(rows),
        "fd_rows": rows,
        "fd_rows_digest": object_sha256(rows),
        "namespace_root_count": 1 if adapter_capture is None else 2,
        "publication_root_count": 1,
        "exact_allowlist_only": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
    }
    value["fd_binding_digest"] = object_sha256(value)
    return value


def _validate_use_receipt(
    value: Any, *, task_id: str, phase: str, capture: Mapping[str, Any],
    pre_use_digest: str | None = None,
) -> dict[str, Any]:
    adapter = phase.startswith("adapter_")
    capture_digest = capture["capture_digest"]
    fields = {
        "schema_version", "task_id", "phase",
        "adapter_capture_digest" if adapter else "model_capture_digest",
        "replay_digest", "private_parent_current_identity", "use_digest",
    }
    if pre_use_digest is not None:
        fields.add("pre_use_digest")
    row = _closed(value, fields, label=f"{phase} receipt")
    capture_key = "adapter_capture_digest" if adapter else "model_capture_digest"
    if (
        row["schema_version"] != model_authority.MODEL_REPLAY_SCHEMA
        or row["task_id"] != task_id
        or row["phase"] != phase
        or row[capture_key] != capture_digest
        or (pre_use_digest is not None and row["pre_use_digest"] != pre_use_digest)
    ):
        raise DecodedEvaluationAggregateError(f"{phase} binding differs")
    stage_by_phase = {
        "pre_use": f"task_pre:{task_id}",
        "post_use": f"task_post:{task_id}",
        "adapter_pre_use": f"adapter_pre:{task_id}",
        "adapter_post_use": f"adapter_post:{task_id}",
    }
    if row["replay_digest"] != _expected_replay_digest(
        capture,
        stage=stage_by_phase[phase],
        adapter=adapter,
        private_parent_current_identity=row[
            "private_parent_current_identity"
        ],
    ):
        raise DecodedEvaluationAggregateError(f"{phase} replay digest differs")
    _digest(row, field="use_digest", label=f"{phase} receipt")
    return row


def _validate_adapter_final(
    value: Any, *, task_id: str, capture: Mapping[str, Any],
    post_use_digest: str,
) -> dict[str, Any]:
    capture_digest = capture["capture_digest"]
    fields = {
        "schema_version", "task_id", "adapter_capture_digest",
        "post_use_digest", "final_rehash_digest",
        "private_parent_current_identity",
        "all_adapter_bytes_rehashed_after_decoder_exit",
        "all_adapter_file_and_directory_fds_retained_through_rehash",
        "adapter_final_digest",
    }
    row = _closed(value, fields, label="adapter final receipt")
    if (
        row["schema_version"] != model_authority.ADAPTER_FINAL_SCHEMA
        or row["task_id"] != task_id
        or row["adapter_capture_digest"] != capture_digest
        or row["post_use_digest"] != post_use_digest
        or row["all_adapter_bytes_rehashed_after_decoder_exit"] is not True
        or row[
            "all_adapter_file_and_directory_fds_retained_through_rehash"
        ] is not True
    ):
        raise DecodedEvaluationAggregateError("adapter final binding differs")
    if row["final_rehash_digest"] != _expected_final_rehash_digest(
        capture,
        stage=f"adapter_final:{task_id}",
        adapter=True,
        private_parent_current_identity=row[
            "private_parent_current_identity"
        ],
    ):
        raise DecodedEvaluationAggregateError(
            "adapter final rehash digest differs"
        )
    _digest(row, field="adapter_final_digest", label="adapter final receipt")
    return row


def _validate_model_final(
    value: Any, *, model_capture: Mapping[str, Any],
    ordered_consumption_digests: Sequence[str],
) -> dict[str, Any]:
    model_capture_digest = model_capture["capture_digest"]
    fields = {
        "schema_version", "model_capture_digest", "task_count",
        "task_consumption_digests", "task_consumption_set_digest",
        "final_rehash_digest", "all_model_bytes_rehashed_after_last_task",
        "private_parent_current_identity",
        "all_model_file_and_directory_fds_retained_through_final_rehash",
        "model_final_digest",
    }
    row = _closed(value, fields, label="model final receipt")
    expected = list(ordered_consumption_digests)
    if (
        row["schema_version"] != model_authority.MODEL_FINAL_SCHEMA
        or row["model_capture_digest"] != model_capture_digest
        or row["task_count"] != len(expected)
        or row["task_consumption_digests"] != expected
        or row["task_consumption_set_digest"] != object_sha256(expected)
        or row["all_model_bytes_rehashed_after_last_task"] is not True
        or row[
            "all_model_file_and_directory_fds_retained_through_final_rehash"
        ] is not True
    ):
        raise DecodedEvaluationAggregateError("model final ordered task set differs")
    if row["final_rehash_digest"] != _expected_final_rehash_digest(
        model_capture,
        stage="holder_final",
        adapter=False,
        private_parent_current_identity=row[
            "private_parent_current_identity"
        ],
    ):
        raise DecodedEvaluationAggregateError("model final rehash digest differs")
    _digest(row, field="model_final_digest", label="model final receipt")
    return row


def _validate_consumption_input_offline(
    value: Any,
    *,
    task_id: str,
    physical_bindings_digest: str,
    model_capture: Mapping[str, Any],
    model_pre: Mapping[str, Any],
    model_capture_path: Path,
    model_capture_sha256: str,
    adapter_capture: Mapping[str, Any] | None,
    adapter_pre: Mapping[str, Any] | None,
    adapter_capture_path: Path | None,
    adapter_capture_sha256: str | None,
    production_required: bool,
) -> dict[str, Any]:
    try:
        row = model_authority.validate_consumption_input(value)
        inherited = model_authority.validate_inherited_fd_binding(
            row["inherited_fds"], verify_open_fds=False
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if production_required and (
        row["production_mode"] is not True
        or row["task_member_path_kind"] != "inherited_proc_self_fd"
    ):
        raise DecodedEvaluationAggregateError(
            "D0 production FD-view authority differs"
        )
    task_rows = [
        item for item in inherited["fd_rows"]
        if item["scope"] == "task" and item["role"] == "publication_root"
    ]
    if len(task_rows) != 1:
        raise DecodedEvaluationAggregateError(
            "D0 task publication-root binding differs"
        )
    task_publication_root = {
        "fd": task_rows[0]["fd"],
        "path": task_rows[0]["source_path"],
        "identity": task_rows[0]["identity"],
    }
    try:
        expected_inherited = _expected_fd_binding(
            task_id=task_id,
            model_capture=model_capture,
            adapter_capture=adapter_capture,
            task_publication_root=task_publication_root,
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if row["production_mode"]:
        expected_task_root = Path(
            model_authority.inherited_proc_root(
                inherited, scope="task", role="publication_root"
            )
        )
        expected_model_view = model_authority.inherited_proc_root(
            inherited, scope="model", role="namespace_root"
        )
        expected_adapter_view = (
            None
            if adapter_capture is None
            else model_authority.inherited_proc_root(
                inherited, scope="adapter", role="namespace_root"
            )
        )
    else:
        expected_task_root = Path(task_publication_root["path"])
        expected_model_view = model_capture["model_view_root"]
        expected_adapter_view = (
            None
            if adapter_capture is None
            else adapter_capture["adapter_view_root"]
        )
    model_binding = row["model"]
    if (
        row["task_id"] != task_id
        or row["physical_bindings_digest"] != physical_bindings_digest
        or model_binding["capture_receipt_path"] != str(model_capture_path)
        or model_binding["capture_receipt_sha256"] != model_capture_sha256
        or model_binding["capture_digest"] != model_capture["capture_digest"]
        or model_binding["pre_use_digest"] != model_pre["use_digest"]
        or Path(model_capture_path).parent != expected_task_root
        or model_binding["view_root"] != expected_model_view
        or inherited != expected_inherited
    ):
        raise DecodedEvaluationAggregateError("D0 model/FD binding differs")
    adapter_binding = row["adapter"]
    if adapter_capture is None:
        if (
            adapter_pre is not None
            or adapter_capture_path is not None
            or adapter_capture_sha256 is not None
            or adapter_binding is not None
        ):
            raise DecodedEvaluationAggregateError("D0 base-control adapter closure differs")
    else:
        if (
            adapter_pre is None
            or adapter_capture_path is None
            or adapter_capture_sha256 is None
            or not isinstance(adapter_binding, Mapping)
            or adapter_binding["capture_receipt_path"]
            != str(adapter_capture_path)
            or adapter_binding["capture_receipt_sha256"]
            != adapter_capture_sha256
            or adapter_binding["capture_digest"]
            != adapter_capture["capture_digest"]
            or adapter_binding["pre_use_digest"] != adapter_pre["use_digest"]
            or Path(adapter_capture_path).parent != expected_task_root
            or adapter_binding["view_root"] != expected_adapter_view
        ):
            raise DecodedEvaluationAggregateError("D0 adapter binding differs")
    return row


def _validate_native_authority_offline(
    value: Any,
    *,
    task_id: str,
    task_input_digest: str,
    consumption_input: Mapping[str, Any],
    model_capture: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DecodedEvaluationAggregateError("native receipt differs")
    row = dict(value)
    _digest(row, field="receipt_digest", label="native inference receipt")
    if (
        row.get("schema_version") != decoder_adapter.INFERENCE_RECEIPT_SCHEMA
        or row.get("consumption_input_digest")
        != consumption_input["consumption_input_digest"]
        or row.get("task_input_digest") != task_input_digest
    ):
        raise DecodedEvaluationAggregateError("native D0/input binding differs")
    evidence = row.get("model_consumption")
    required = {
        "consumption_input_digest", "task_input_digest",
        "model_capture_digest", "model_view_root", "adapter_capture_digest",
        "adapter_view_root", "fd_view_files_authorized",
        "inherited_fd_binding_digest", "inherited_fd_count",
        "ptrace_authorization_used", "four_rank_attestation",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        raise DecodedEvaluationAggregateError(
            "native model-consumption evidence closure differs"
        )
    evidence = dict(evidence)
    attestation = _closed(
        evidence["four_rank_attestation"],
        {
            "world_size", "all_ranks_replayed_exact_fd_views",
            "rank_evidence_digest", "ordered_rank_evidence_digests",
        },
        label="native four-rank attestation",
    )
    rank_evidence = dict(evidence)
    rank_evidence.pop("four_rank_attestation")
    expected_rank_evidence = {
        "consumption_input_digest": consumption_input[
            "consumption_input_digest"
        ],
        "task_input_digest": task_input_digest,
        "model_capture_digest": model_capture["capture_digest"],
        "model_view_root": consumption_input["model"]["view_root"],
        "adapter_capture_digest": (
            None if adapter_capture is None else adapter_capture["capture_digest"]
        ),
        "adapter_view_root": (
            None
            if adapter_capture is None
            else consumption_input["adapter"]["view_root"]
        ),
        "fd_view_files_authorized": (
            model_capture["file_count"]
            + (0 if adapter_capture is None else adapter_capture["file_count"])
        ),
        "inherited_fd_binding_digest": consumption_input["inherited_fds"][
            "fd_binding_digest"
        ],
        "inherited_fd_count": consumption_input["inherited_fds"]["fd_count"],
        "ptrace_authorization_used": False,
    }
    rank_digest = object_sha256(expected_rank_evidence)
    if (
        rank_evidence != expected_rank_evidence
        or attestation["world_size"] != 4
        or attestation["all_ranks_replayed_exact_fd_views"] is not True
        or attestation["rank_evidence_digest"] != rank_digest
        or attestation["ordered_rank_evidence_digests"] != [rank_digest] * 4
    ):
        raise DecodedEvaluationAggregateError(
            "native four-rank FD-view attestation differs"
        )
    return row


def _validate_publication_gate_offline(
    value: Any,
    *,
    task_id: str,
    consumption_digest: str,
    staging_path: str,
    staging_sha256: str,
    staging_size: int,
) -> dict[str, Any]:
    fields = {
        "schema_version", "task_id", "consumption_digest", "staging_path",
        "staging_sha256", "staging_size", "model_post_use_verified",
        "adapter_post_use_verified_or_base_control",
        "adapter_fds_closed_or_base_control", "publication_authorized",
        "publication_has_occurred", "publication_gate_digest",
    }
    row = _closed(value, fields, label="consumption publication gate")
    if (
        row["schema_version"] != model_authority.PUBLICATION_GATE_SCHEMA
        or row["task_id"] != task_id
        or row["consumption_digest"] != consumption_digest
        or row["staging_path"] != staging_path
        or row["staging_sha256"] != staging_sha256
        or row["staging_size"] != staging_size
        or row["model_post_use_verified"] is not True
        or row["adapter_post_use_verified_or_base_control"] is not True
        or row["adapter_fds_closed_or_base_control"] is not True
        or row["publication_authorized"] is not True
        or row["publication_has_occurred"] is not False
    ):
        raise DecodedEvaluationAggregateError(
            "consumption publication gate/staging binding differs"
        )
    _digest(row, field="publication_gate_digest", label="publication gate")
    return row


def validate_offline_authority_chain(
    *,
    task_id: str,
    task_kind: str,
    physical_bindings_digest: str,
    model_capture: Mapping[str, Any],
    model_capture_path: Path,
    model_capture_sha256: str,
    model_pre: Mapping[str, Any],
    model_post: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
    adapter_capture_path: Path | None,
    adapter_capture_sha256: str | None,
    adapter_pre: Mapping[str, Any] | None,
    adapter_post: Mapping[str, Any] | None,
    adapter_final: Mapping[str, Any] | None,
    consumption_input: Mapping[str, Any],
    consumption_input_path: Path,
    consumption_input_sha256: str,
    task_input: Mapping[str, Any],
    native_receipt: Mapping[str, Any],
    consumption_chain: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    publication_gate: Mapping[str, Any],
    output_receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    shard_summary_digest: str,
    staging_path: Path,
    production_required: bool = False,
) -> dict[str, Any]:
    """Close one task's authority DAG using stored values only.

    This deliberately performs no filesystem operation.  The caller first
    loads canonical files and, separately, re-hashes the published media.
    Tests use nonexistent FD-view roots to guarantee this boundary.
    """

    _sha(physical_bindings_digest, label="physical bindings")
    _sha(model_capture_sha256, label="model capture file")
    _sha(consumption_input_sha256, label="D0 file")
    _sha(shard_summary_digest, label="shard summary")
    model = _validate_capture_offline(
        model_capture, adapter=False, production_required=False
    )
    model_pre_row = _validate_use_receipt(
        model_pre,
        task_id=task_id,
        phase="pre_use",
        capture=model,
    )
    has_adapter = task_kind == "adapter_candidate"
    if task_kind not in {"adapter_candidate", "frozen_base_control"}:
        raise DecodedEvaluationAggregateError("task kind differs")
    adapter_values = (
        adapter_capture, adapter_capture_path, adapter_capture_sha256,
        adapter_pre, adapter_post, adapter_final,
    )
    if has_adapter is not all(value is not None for value in adapter_values):
        raise DecodedEvaluationAggregateError("task adapter evidence closure differs")
    adapter: dict[str, Any] | None = None
    adapter_pre_row: dict[str, Any] | None = None
    if has_adapter:
        assert adapter_capture is not None
        assert adapter_pre is not None
        adapter = _validate_capture_offline(
            adapter_capture,
            adapter=True,
            task_id=task_id,
            production_required=False,
        )
        adapter_pre_row = _validate_use_receipt(
            adapter_pre,
            task_id=task_id,
            phase="adapter_pre_use",
            capture=adapter,
        )
    d0 = _validate_consumption_input_offline(
        consumption_input,
        task_id=task_id,
        physical_bindings_digest=physical_bindings_digest,
        model_capture=model,
        model_pre=model_pre_row,
        model_capture_path=model_capture_path,
        model_capture_sha256=model_capture_sha256,
        adapter_capture=adapter,
        adapter_pre=adapter_pre_row,
        adapter_capture_path=adapter_capture_path,
        adapter_capture_sha256=adapter_capture_sha256,
        production_required=production_required,
    )

    if not isinstance(task_input, Mapping):
        raise DecodedEvaluationAggregateError("task input differs")
    task_input = dict(task_input)
    input_digest = _digest(task_input, field="input_digest", label="task input")
    input_identity = task_input.get("model_consumption_input")
    if (
        task_input.get("schema_version") != executor.TASK_INPUT_SCHEMA
        or task_input.get("task_id") != task_id
        or task_input.get("task_kind") != task_kind
        or not isinstance(input_identity, Mapping)
        or dict(input_identity)
        != {
            "path": str(consumption_input_path),
            "sha256": consumption_input_sha256,
            "consumption_input_digest": d0["consumption_input_digest"],
        }
    ):
        raise DecodedEvaluationAggregateError("D0 to task-input binding differs")

    native = _validate_native_authority_offline(
        native_receipt,
        task_id=task_id,
        task_input_digest=input_digest,
        consumption_input=d0,
        model_capture=model,
        adapter_capture=adapter,
    )
    native_digest = native["receipt_digest"]

    model_post_row = _validate_use_receipt(
        model_post,
        task_id=task_id,
        phase="post_use",
        capture=model,
        pre_use_digest=model_pre_row["use_digest"],
    )
    adapter_post_row: dict[str, Any] | None = None
    adapter_final_row: dict[str, Any] | None = None
    if has_adapter:
        assert adapter is not None
        assert adapter_pre_row is not None
        assert adapter_post is not None
        assert adapter_final is not None
        adapter_post_row = _validate_use_receipt(
            adapter_post,
            task_id=task_id,
            phase="adapter_post_use",
            capture=adapter,
            pre_use_digest=adapter_pre_row["use_digest"],
        )
        adapter_final_row = _validate_adapter_final(
            adapter_final,
            task_id=task_id,
            capture=adapter,
            post_use_digest=adapter_post_row["use_digest"],
        )
    try:
        chain = model_authority.validate_consumption_chain(consumption_chain)
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if (
        chain["task_id"] != task_id
        or chain["consumption_input_digest"] != d0["consumption_input_digest"]
        or chain["model_capture_digest"] != model["capture_digest"]
        or chain["model_pre_use_digest"] != model_pre_row["use_digest"]
        or chain["model_post_use_digest"] != model_post_row["use_digest"]
        or chain["adapter_capture_digest"]
        != (None if adapter is None else adapter["capture_digest"])
        or chain["adapter_pre_use_digest"]
        != (None if adapter_pre_row is None else adapter_pre_row["use_digest"])
        or chain["adapter_post_use_digest"]
        != (None if adapter_post_row is None else adapter_post_row["use_digest"])
        or chain["adapter_final_digest"]
        != (
            None
            if adapter_final_row is None
            else adapter_final_row["adapter_final_digest"]
        )
        or chain["native_inference_receipt_digest"] != native_digest
    ):
        raise DecodedEvaluationAggregateError("native to C authority binding differs")
    consumption_digest = chain["consumption_digest"]

    if not isinstance(process_receipt, Mapping):
        raise DecodedEvaluationAggregateError("process receipt differs")
    process = dict(process_receipt)
    process_digest = _digest(process, field="process_digest", label="process receipt")
    try:
        validated_inheritance = executor._validate_fd_inheritance_evidence(
            process.get("fd_inheritance"),
            production_required=True,
            return_code=process.get("return_code"),
        )
    except executor.DecodedEvaluationExecutorError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if (
        process.get("schema_version") != executor.PROCESS_SCHEMA
        or process.get("task_id") != task_id
        or process.get("input_digest") != input_digest
        or process.get("consumption_digest") != consumption_digest
        or process.get("return_code") != 0
        or validated_inheritance.get("fd_binding") != d0["inherited_fds"]
        or validated_inheritance.get("fd_binding_digest")
        != d0["inherited_fds"]["fd_binding_digest"]
        or validated_inheritance.get("production_mode") is not True
        or validated_inheritance.get("decoder_spawn_performed") is not True
        or validated_inheritance.get("decoder_spawn_close_fds_true") is not True
        or validated_inheritance.get("exact_pass_fds_only") is not True
        or validated_inheritance.get(
            "unrelated_child_inherits_authority_fds"
        ) is not False
        or validated_inheritance.get("ptrace_authorization_used") is not False
    ):
        raise DecodedEvaluationAggregateError("C to process/FD binding differs")

    if not isinstance(output_receipt, Mapping):
        raise DecodedEvaluationAggregateError("output receipt differs")
    output = dict(output_receipt)
    output_digest = _digest(output, field="output_digest", label="output receipt")
    staging_sha = _sha(output.get("output_video_sha256"), label="output video")
    staging_size = output.get("output_byte_size")
    if type(staging_size) is not int or staging_size <= 0:
        raise DecodedEvaluationAggregateError("output byte size differs")
    try:
        published_inode_identity = executor._published_inode_identity(
            output.get("published_inode_identity")
        )
    except executor.DecodedEvaluationExecutorError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if published_inode_identity["size"] != staging_size:
        raise DecodedEvaluationAggregateError(
            "published inode/output size binding differs"
        )
    gate = _validate_publication_gate_offline(
        publication_gate,
        task_id=task_id,
        consumption_digest=consumption_digest,
        staging_path=str(staging_path),
        staging_sha256=staging_sha,
        staging_size=staging_size,
    )
    if (
        output.get("schema_version") != executor.TASK_OUTPUT_SCHEMA
        or output.get("task_id") != task_id
        or output.get("task_kind") != task_kind
        or output.get("input_digest") != input_digest
        or output.get("process_digest") != process_digest
        or output.get("consumption_chain") != chain
        or output.get("consumption_digest") != consumption_digest
        or output.get("publication_gate") != gate
        or output.get("publication_gate_digest")
        != gate["publication_gate_digest"]
    ):
        raise DecodedEvaluationAggregateError("process to output binding differs")
    native_evidence = output.get("native_inference_receipt")
    if (
        not isinstance(native_evidence, Mapping)
        or native_evidence.get("receipt_digest") != native_digest
    ):
        raise DecodedEvaluationAggregateError(
            "native to output evidence binding differs"
        )

    result_fields = {
        "task_id", "status", "terminal_receipt_digest", "consumption_digest",
        "publication_gate_digest", "output_relpath", "result_digest",
    }
    result_row = _closed(result, result_fields, label="shard task result")
    result_digest = _digest(result_row, field="result_digest", label="shard task result")
    if (
        result_row["task_id"] != task_id
        or result_row["status"] != "success"
        or result_row["terminal_receipt_digest"] != output_digest
        or result_row["consumption_digest"] != consumption_digest
        or result_row["publication_gate_digest"]
        != gate["publication_gate_digest"]
        or result_row["output_relpath"] != output.get("output_relpath")
    ):
        raise DecodedEvaluationAggregateError("output to result binding differs")
    projection = {
        "task_id": task_id,
        "holder_model_capture_digest": model["capture_digest"],
        "d0_consumption_input_digest": d0["consumption_input_digest"],
        "task_input_digest": input_digest,
        "native_inference_receipt_digest": native_digest,
        "consumption_digest": consumption_digest,
        "process_digest": process_digest,
        "publication_gate_digest": gate["publication_gate_digest"],
        "output_digest": output_digest,
        "result_digest": result_digest,
        "shard_summary_digest": shard_summary_digest,
    }
    projection["authority_chain_digest"] = object_sha256(projection)
    return projection


def _validate_summary(
    value: Any, *, bundle: Mapping[str, Any], shard: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_manifest_digest",
        "publication_digest", "shard_digest", "holder",
        "model_capture_digest", "model_final_digest",
        "task_consumption_set_digest", "holder_execution_authority",
        "executor_verified_release_capture", "planned_task_count",
        "attempted_task_count", "success_count", "failure_count", "results",
        "all_tasks_attempted_exactly_once", "automatic_retry_count",
        "retry_allowed", "failure_artifacts_retained", "execution_backend",
        "tool_files_verified", "training_loss_read_or_used", "network_used",
        "remote_launch_performed", "scientific_promotion_authorized",
        "summary_digest",
    }
    row = _closed(value, fields, label="shard summary")
    holder_fields = {
        "expected_job_id", "expected_node", "observed_slurm_job_id",
        "observed_hostname", "exact_holder_match", "holder_execution_digest",
    }
    holder = _closed(
        row["holder_execution_authority"],
        holder_fields,
        label="holder execution authority",
    )
    holder_digest = _digest(
        holder, field="holder_execution_digest", label="holder execution authority"
    )
    expected_holder = shard["holder"]
    if (
        holder["expected_job_id"] != expected_holder["job_id"]
        or holder["expected_node"] != expected_holder["node"]
        or holder["observed_slurm_job_id"] != expected_holder["job_id"]
        or holder["observed_hostname"] != expected_holder["node"]
        or holder["exact_holder_match"] is not True
    ):
        raise DecodedEvaluationAggregateError("holder execution authority differs")
    capture = executor._capture_evidence(
        row["executor_verified_release_capture"],
        label="shard executor verified release capture",
    )
    expected_ids = [executor._task_id(task) for task in shard["tasks"]]
    expected_relpaths = [task["record"]["output_relpath"] for task in shard["tasks"]]
    results = row["results"]
    if not isinstance(results, list):
        raise DecodedEvaluationAggregateError("shard results differ")
    for result in results:
        _digest(
            _closed(
                result,
                {
                    "task_id", "status", "terminal_receipt_digest",
                    "consumption_digest", "publication_gate_digest",
                    "output_relpath", "result_digest",
                },
                label="shard result",
            ),
            field="result_digest",
            label="shard result",
        )
    if (
        row["schema_version"] != executor.SHARD_SUMMARY_SCHEMA
        or row["evaluation_id"] != bundle["manifest"]["evaluation_id"]
        or row["evaluation_manifest_digest"]
        != bundle["manifest"]["manifest_digest"]
        or row["publication_digest"]
        != bundle["publication_receipt"]["publication_digest"]
        or row["shard_digest"] != shard["shard_digest"]
        or row["holder"] != expected_holder
        or row["planned_task_count"] != shard["total_task_count"]
        or row["attempted_task_count"] != shard["total_task_count"]
        or row["success_count"] != shard["total_task_count"]
        or row["failure_count"] != 0
        or [item.get("task_id") for item in results] != expected_ids
        or [item.get("output_relpath") for item in results] != expected_relpaths
        or any(item.get("status") != "success" for item in results)
        or row["all_tasks_attempted_exactly_once"] is not True
        or row["automatic_retry_count"] != 0
        or row["retry_allowed"] is not False
        or row["failure_artifacts_retained"] is not True
        or row["execution_backend"] != "pinned_local_subprocess"
        or row["tool_files_verified"] is not True
        or capture is None
        or capture["target"]
        != "action_preservation_decoded_eval_executor_v2.py"
        or row["training_loss_read_or_used"] is not False
        or row["network_used"] is not False
        or row["remote_launch_performed"] is not False
        or row["scientific_promotion_authorized"] is not False
    ):
        raise DecodedEvaluationAggregateError(
            "shard is not an exact successful production execution"
        )
    for field in (
        "model_capture_digest", "model_final_digest",
        "task_consumption_set_digest",
    ):
        _sha(row[field], label=f"shard {field}")
    _digest(row, field="summary_digest", label="shard summary")
    # Kept local so a caller cannot replace the holder object after its digest
    # was checked while preserving the enclosing summary shape.
    if holder_digest != row["holder_execution_authority"]["holder_execution_digest"]:
        raise DecodedEvaluationAggregateError("holder digest binding differs")
    return row


def _task_physical_bindings(
    *, task: Mapping[str, Any], bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = task["record"]
    sources = [item for item in bindings["sources"] if item["iid"] == record["iid"]]
    if len(sources) != 1:
        raise DecodedEvaluationAggregateError("task physical source differs")
    source = dict(sources[0])
    if (
        source["source_video"]["sha256"] != record["source_video_sha256"]
        or source["source_receipt"]["sha256"] != record["source_receipt_sha256"]
        or source["instruction_sha256"] != record["instruction_sha256"]
        or source["action_review_contract_digest"]
        != record["action_review_contract"]["contract_digest"]
        or source["seed"] != record["seed"]
    ):
        raise DecodedEvaluationAggregateError("task source bytes differ")
    checkpoint: dict[str, Any] | None = None
    if task["task_kind"] == "adapter_candidate":
        matches = [
            item
            for item in bindings["checkpoints"]
            if (item["arm"], item["checkpoint_step"])
            == (record["arm"], record["checkpoint_step"])
        ]
        if len(matches) != 1:
            raise DecodedEvaluationAggregateError("task physical checkpoint differs")
        checkpoint = dict(matches[0])
        if (
            checkpoint["checkpoint_receipt"]["sha256"]
            != record["checkpoint_receipt_sha256"]
            or checkpoint["adapter_model"]["sha256"] != record["adapter_sha256"]
        ):
            raise DecodedEvaluationAggregateError("task adapter bytes differ")
    elif task["task_kind"] != "frozen_base_control":
        raise DecodedEvaluationAggregateError("task kind differs")
    return source, checkpoint


def _validate_native_semantics_offline(
    row: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    bindings: Mapping[str, Any],
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    model_capture: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
    staging_path: Path,
) -> None:
    runtime = bindings["runtime"]
    record = request["task_record"]
    if (
        row.get("method_source_revision") != runtime["method_source_revision"]
        or row.get("method_source_archive_sha256")
        != runtime["method_source_archive_sha256"]
        or row.get("bernini_commit") != runtime["expected_bernini_commit"]
        or row.get("veomni_commit") != runtime["expected_veomni_commit"]
        or row.get("checkpoint_tree_sha256")
        != runtime["expected_checkpoint_tree_sha256"]
        or row.get("experimental_inference") is not True
        or row.get("production_claim_forbidden") is not True
        or row.get("scientific_claim_authorized") is not False
    ):
        raise DecodedEvaluationAggregateError("native runtime authority differs")
    input_row = row.get("input")
    if not isinstance(input_row, Mapping) or (
        input_row.get("source_video_path") != source["source_video"]["path"]
        or input_row.get("source_video_sha256") != record["source_video_sha256"]
        or input_row.get("instruction_utf8_sha256") != record["instruction_sha256"]
        or input_row.get("accepted_model_conditions")
        != ["source_video", "edit_instruction"]
        or any(
            input_row.get(key) is not False
            for key in (
                "target_video_argument", "target_accessed_by_inference",
                "external_mask_or_swept_tube",
                "external_tracking_pose_or_trajectory",
                "reference_image_or_video", "external_shared_i0",
            )
        )
    ):
        raise DecodedEvaluationAggregateError("native input semantics differ")
    sampling = row.get("sampling")
    if not isinstance(sampling, Mapping) or (
        sampling.get("seed") != record["seed"]
        or sampling.get("num_inference_steps") != 40
        or sampling.get("num_frames") != 81
        or sampling.get("source_onset_policy") != record["onset_policy"]["name"]
        or sampling.get("ulysses_size") != 4
        or sampling.get("rank0_decode_and_save_only") is not True
    ):
        raise DecodedEvaluationAggregateError("native sampling semantics differ")
    trace = sampling.get("source_onset_solver_trace")
    try:
        if record["onset_policy"]["name"] == "hard1_every_step":
            decoder_adapter._validate_source_onset_solver_trace(trace)
        elif trace is not None:
            raise DecodedEvaluationAggregateError(
                "native non-every-step solver trace differs"
            )
    except decoder_adapter.DecodedEvaluationDecoderError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    output = row.get("output")
    staging_sha = plan.file_sha256(staging_path)
    if not isinstance(output, Mapping) or (
        output.get("path") != str(staging_path)
        or output.get("sha256") != staging_sha
        or output.get("frame_count") != 81
        or float(output.get("fps", -1)) != 25.0
    ):
        raise DecodedEvaluationAggregateError("native staging output differs")
    adaptation = row.get("adapter")
    if not isinstance(adaptation, Mapping):
        raise DecodedEvaluationAggregateError("native adapter receipt differs")
    if checkpoint is None:
        if (
            adapter_capture is not None
            or adaptation.get("enabled") is not False
            or adaptation.get("mode") != "frozen_base_no_adapter"
            or adaptation.get("tensor_count") != 0
        ):
            raise DecodedEvaluationAggregateError("native base control differs")
    elif (
        adapter_capture is None
        or adaptation.get("enabled") is not True
        or adaptation.get("mode") != "lora_safe_merge"
        or adaptation.get("checkpoint_root")
        != adapter_capture["adapter_view_root"]
        or adaptation.get("adapter_model_path")
        != str(
            Path(adapter_capture["adapter_view_root"])
            / "adapter/adapter_model.safetensors"
        )
        or adaptation.get("adapter_model_sha256")
        != checkpoint["adapter_model"]["sha256"]
        or adaptation.get("training_receipt_path")
        != str(Path(adapter_capture["adapter_view_root"]) / "receipt.json")
        or adaptation.get("training_receipt_digest")
        != checkpoint["checkpoint_receipt_digest"]
        or adaptation.get("training_global_step")
        != checkpoint["checkpoint_step"]
        or adaptation.get("strictly_reloaded") is not True
        or adaptation.get("safe_merged_for_inference") is not True
    ):
        raise DecodedEvaluationAggregateError("native adapter authority differs")


def _inference_arguments_offline(
    *,
    request: Mapping[str, Any],
    bindings: Mapping[str, Any],
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    model_capture: Mapping[str, Any],
    adapter_capture: Mapping[str, Any] | None,
    consumption_input: Mapping[str, Any],
    output_path: Path,
) -> list[str]:
    runtime = bindings["runtime"]
    record = request["task_record"]
    arguments = [
        "--bernini-root", runtime["bernini_root"],
        "--veomni-root", runtime["veomni_root"],
        "--checkpoint", consumption_input["model"]["view_root"],
        "--source-video", source["source_video"]["path"],
        "--instruction", record["instruction"],
        "--output", str(output_path),
        "--num-inference-steps", str(runtime["num_inference_steps"]),
        "--seed", str(record["seed"]),
        "--source-onset-policy", record["onset_policy"]["name"],
        "--expected-bernini-commit", runtime["expected_bernini_commit"],
        "--expected-veomni-commit", runtime["expected_veomni_commit"],
        "--expected-checkpoint-tree-sha256",
        runtime["expected_checkpoint_tree_sha256"],
        "--method-source-revision", runtime["method_source_revision"],
        "--method-source-archive-sha256",
        runtime["method_source_archive_sha256"],
        "--model-consumption-input", request["model_consumption_input"]["path"],
        "--model-consumption-input-sha256",
        request["model_consumption_input"]["sha256"],
        "--model-consumption-input-digest",
        request["model_consumption_input"]["consumption_input_digest"],
        "--task-input-digest", request["input_digest"],
    ]
    if checkpoint is None:
        arguments.append("--base-only")
    else:
        if adapter_capture is None:
            raise DecodedEvaluationAggregateError("adapter FD-view binding is absent")
        arguments.extend(
            ["--adapter-checkpoint", consumption_input["adapter"]["view_root"]]
        )
    return arguments


def _collect_verified_outputs_impl(
    *,
    evaluation_root: str | Path,
    physical_bindings_path: str | Path,
    physical_bindings_sha256: str,
    work_root_binding: Mapping[str, Any] | None = None,
    holder_completion_anchors: Sequence[Mapping[str, Any]] | None = None,
    retained_cleanup: list[Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(evaluation_root)
    try:
        bundle = executor.load_published_bundle(
            root, work_root_binding=work_root_binding
        )
        initial_handle = bundle.get("_evaluation_root_handle")
        if initial_handle is not None:
            retained_cleanup.append(initial_handle)
        bindings = bridge.load_physical_bindings(
            physical_bindings_path,
            expected_sha256=physical_bindings_sha256,
            verify_files=True,
        )
        for relative_path, module_path in (
            ("action_preservation_decoded_eval_aggregate_v2.py", __file__),
            ("action_preservation_decoded_eval_executor_v2.py", executor.__file__),
            (
                "action_preservation_decoded_eval_decoder_adapter_v1.py",
                decoder_adapter.__file__,
            ),
            ("action_preservation_decoded_eval_bridge_v1.py", bridge.__file__),
            ("action_preservation_decoded_eval_plan_v1.py", plan.__file__),
            (
                "action_preservation_decoded_eval_model_authority_v2.py",
                model_authority.__file__,
            ),
            ("action_preservation_gate_v1.py", plan.gate.__file__),
        ):
            bridge.require_running_eval_release_member(
                bindings["eval_release"],
                relative_path=relative_path,
                running_path=module_path,
            )
    except (
        executor.DecodedEvaluationExecutorError,
        bridge.DecodedEvaluationBridgeError,
    ) as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if (
        bindings["evaluation_root"] != str(root)
        or bindings["evaluation_id"] != bundle["manifest"]["evaluation_id"]
        or bindings["input_digest"] != bundle["input_spec"]["input_digest"]
        or bindings["manifest_digest"] != bundle["manifest"]["manifest_digest"]
    ):
        raise DecodedEvaluationAggregateError("aggregate physical binding differs")
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
        raise DecodedEvaluationAggregateError(
            "aggregate physical pins differ from evaluation input"
        )
    if (
        bindings["pin_files"]["model_release_manifest"]["sha256"]
        != model_authority.MODEL_MANIFEST_SHA256
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate model pin is not exact-23"
        )
    model_manifest_binding = bindings["pin_files"]["model_release_manifest"]
    try:
        expected_model_files, expected_model_manifest = (
            model_authority.parse_exact23_manifest(
                model_manifest_binding["path"],
                expected_manifest_sha256=model_authority.MODEL_MANIFEST_SHA256,
            )
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    manifest_identity = expected_model_manifest["identity"]
    if (
        expected_model_manifest["path"] != model_manifest_binding["path"]
        or expected_model_manifest["sha256"] != model_manifest_binding["sha256"]
        or stat.S_IMODE(manifest_identity["mode"])
        != model_manifest_binding["mode"]
        or any(
            manifest_identity[key] != model_manifest_binding[key]
            for key in (
                "size", "device", "inode", "uid", "gid", "nlink",
                "rdev", "blocks", "mtime_ns", "ctime_ns",
            )
        )
    ):
        raise DecodedEvaluationAggregateError(
            "exact-23 manifest/physical identity binding differs"
        )
    if not isinstance(work_root_binding, Mapping):
        raise DecodedEvaluationAggregateError(
            "production aggregate lacks inherited work-root authority"
        )
    completion_documents = _capture_holder_completion_documents(bundle)
    dynamic_completion_anchors = validate_holder_completion_anchors(
        holder_completion_anchors,
        bundle=bundle,
        completion_documents=completion_documents,
    )
    final_publication_authority, final_directory_merge = (
        _open_final_publication_authority(
            bundle=bundle,
            bindings=bindings,
            work_root_binding=work_root_binding,
            completion_documents=completion_documents,
        )
    )
    bundle["_final_publication_authority"] = final_publication_authority
    bundle["_final_directory_merge"] = final_directory_merge
    retained_cleanup.append(final_publication_authority)
    initial_root_handle = bundle.pop("_evaluation_root_handle", None)
    if isinstance(initial_root_handle, executor._HeldDirectory):
        initial_root_handle.close()

    def tree_json_with_sha(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
        return _read_final_tree_json(
            final_publication_authority,
            evaluation_root=root,
            path=path,
            label=label,
        )

    def tree_json(path: Path, *, label: str) -> dict[str, Any]:
        return tree_json_with_sha(path, label=label)[0]

    def tree_bytes(path: Path, *, label: str) -> bytes:
        return _read_final_tree_member(
            final_publication_authority,
            evaluation_root=root,
            path=path,
            label=label,
        )[0]

    physical_identity = {
        "path": str(Path(physical_bindings_path).resolve(strict=True)),
        "sha256": physical_bindings_sha256,
    }
    release_members = {
        item["relative_path"]: item
        for item in bindings["eval_release"]["members"]
    }
    expected_executor_sha = release_members[
        "action_preservation_decoded_eval_executor_v2.py"
    ]["sha256"]
    expected_decoder_sha = release_members[
        "action_preservation_decoded_eval_decoder_adapter_v1.py"
    ]["sha256"]
    reprobe_video = executor.ffprobe_video_prober(
        bindings["runtime"]["ffprobe"]["path"]
    )
    outputs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for holder in plan.HOLDER_ROWS:
        job_id = holder["job_id"]
        shard = bundle["shards"][job_id]
        shard_root = root / executor.EXECUTION_DIRECTORY / job_id
        summary_path = shard_root / executor.SUMMARY_FILENAME
        summary_value, summary_sha = tree_json_with_sha(
            summary_path, label=f"holder {job_id} summary"
        )
        summary = _validate_summary(summary_value, bundle=bundle, shard=shard)
        completion_document = completion_documents[job_id]
        holder_completion = completion_document["completion"]
        completion_anchor = dynamic_completion_anchors[job_id]
        if holder_completion["holder_summary_digest"] != summary[
            "summary_digest"
        ]:
            raise DecodedEvaluationAggregateError(
                "holder completion/summary binding differs"
            )
        expected_executor_arguments = [
            "--evaluation-root", str(root),
            "--holder-job-id", job_id,
            "--decoder-adapter", bindings["runtime"]["decoder_adapter"]["path"],
            "--decoder-adapter-sha256",
            bindings["runtime"]["decoder_adapter"]["sha256"],
            "--ffprobe", bindings["runtime"]["ffprobe"]["path"],
            "--ffprobe-sha256", bindings["runtime"]["ffprobe"]["sha256"],
            "--physical-bindings", physical_identity["path"],
            "--physical-bindings-sha256", physical_identity["sha256"],
            "--confirmation", f"execute-local-decoded-eval-shard-v2-{job_id}",
        ]
        try:
            replayed_executor_capture = bridge.validate_verified_capture_receipt(
                bindings,
                receipt_path=summary["executor_verified_release_capture"][
                    "receipt_path"
                ],
                target="action_preservation_decoded_eval_executor_v2.py",
                expected_arguments=expected_executor_arguments,
                expected_capture_digest=summary[
                    "executor_verified_release_capture"
                ]["capture_digest"],
                verify_file=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        if replayed_executor_capture != summary[
            "executor_verified_release_capture"
        ]:
            raise DecodedEvaluationAggregateError(
                "holder executor capture replay differs"
            )

        authority_root = shard_root / executor.CONSUMPTION_AUTHORITY_DIRECTORY
        model_capture_path = authority_root / executor.MODEL_CAPTURE_FILENAME
        model_final_path = authority_root / executor.MODEL_FINAL_FILENAME
        model_capture_value, model_capture_sha = tree_json_with_sha(
            model_capture_path, label=f"holder {job_id} model capture"
        )
        model_capture = _validate_capture_offline(
            model_capture_value,
            adapter=False,
            production_required=True,
            expected_model_files=expected_model_files,
            expected_model_manifest=expected_model_manifest,
        )
        if (
            model_capture["model_root"]
            != bindings["runtime"]["model_checkpoint_root"]
            or model_capture["manifest"]["path"]
            != bindings["pin_files"]["model_release_manifest"]["path"]
            or model_capture["manifest"]["sha256"]
            != bindings["pin_files"]["model_release_manifest"]["sha256"]
            or summary["model_capture_digest"]
            != model_capture["capture_digest"]
        ):
            raise DecodedEvaluationAggregateError(
                "holder model capture/physical binding differs"
            )
        ordered_consumptions: list[str] = []
        ordered_chain_digests: list[str] = []

        for task, result in zip(shard["tasks"], summary["results"]):
            task_id = executor._task_id(task)
            task_root = shard_root / "tasks" / task_id
            request_path = task_root / executor.INPUT_RECEIPT_FILENAME
            consumption_input_path = task_root / executor.CONSUMPTION_INPUT_FILENAME
            process_path = task_root / executor.PROCESS_RECEIPT_FILENAME
            output_receipt_path = task_root / executor.OUTPUT_RECEIPT_FILENAME
            staging_path = task_root / executor.STAGING_VIDEO_FILENAME
            final_path = root / task["record"]["output_relpath"]

            model_pre = tree_json(
                task_root / executor.MODEL_PRE_USE_FILENAME,
                label=f"{task_id} model pre-use",
            )
            adapter_capture: dict[str, Any] | None = None
            adapter_capture_path: Path | None = None
            adapter_capture_sha: str | None = None
            adapter_pre: dict[str, Any] | None = None
            adapter_post: dict[str, Any] | None = None
            adapter_final: dict[str, Any] | None = None
            source, checkpoint = _task_physical_bindings(
                task=task, bindings=bindings
            )
            if checkpoint is not None:
                adapter_capture_path = task_root / executor.ADAPTER_CAPTURE_FILENAME
                adapter_capture_value, adapter_capture_sha = tree_json_with_sha(
                    adapter_capture_path, label=f"{task_id} adapter capture"
                )
                adapter_capture = _validate_capture_offline(
                    adapter_capture_value,
                    adapter=True,
                    task_id=task_id,
                    production_required=True,
                )
                captured_files = _capture_rows_by_relative(adapter_capture)
                expected_files = {
                    "receipt.json": checkpoint["checkpoint_receipt"],
                    "adapter/adapter_config.json": checkpoint["adapter_config"],
                    "adapter/adapter_model.safetensors": checkpoint["adapter_model"],
                }
                if (
                    adapter_capture["checkpoint_root"]
                    != checkpoint["checkpoint_root"]
                    or any(
                        captured_files[relative]["path"] != bound["path"]
                        or captured_files[relative]["sha256"] != bound["sha256"]
                        or captured_files[relative]["identity"]["size"]
                        != bound["size"]
                        or stat.S_IMODE(
                            captured_files[relative]["identity"]["mode"]
                        )
                        != bound["mode"]
                        or any(
                            captured_files[relative]["identity"][key]
                            != bound[key]
                            for key in (
                                "device", "inode", "uid", "gid", "nlink",
                                "rdev", "blocks", "mtime_ns", "ctime_ns",
                            )
                        )
                        for relative, bound in expected_files.items()
                    )
                ):
                    raise DecodedEvaluationAggregateError(
                        f"{task_id} adapter captured bytes differ"
                    )
                adapter_pre = tree_json(
                    task_root / executor.ADAPTER_PRE_USE_FILENAME,
                    label=f"{task_id} adapter pre-use",
                )

            consumption_input, consumption_input_sha = tree_json_with_sha(
                consumption_input_path, label=f"{task_id} D0 consumption input"
            )
            request_value = tree_json(
                request_path, label=f"{task_id} task input"
            )
            try:
                request = executor.validate_task_input_receipt(
                    request_value, task=task, bundle=bundle, shard=shard
                )
            except executor.DecodedEvaluationExecutorError as error:
                raise DecodedEvaluationAggregateError(str(error)) from error
            if (
                request["physical_bindings"] != physical_identity
                or request["executor_verified_release_capture"]
                != replayed_executor_capture
                or request["decoder_adapter"]["sha256"] != expected_decoder_sha
                or request["executor_source_sha256"] != expected_executor_sha
                or request["tool_files_verified"] is not True
            ):
                raise DecodedEvaluationAggregateError(
                    f"{task_id} production task input differs"
                )

            native_receipt_path = staging_path.with_name(
                staging_path.name + ".receipt.json"
            )
            native_receipt, native_receipt_sha = tree_json_with_sha(
                native_receipt_path, label=f"{task_id} native receipt"
            )
            # This semantic verification is explicitly offline: it only uses
            # stored strings/hashes and the published staging inode, never the
            # dead model/adapter views.
            _validate_native_semantics_offline(
                native_receipt,
                request=request,
                bindings=bindings,
                source=source,
                checkpoint=checkpoint,
                model_capture=model_capture,
                adapter_capture=adapter_capture,
                staging_path=staging_path,
            )

            model_post = tree_json(
                task_root / executor.MODEL_POST_USE_FILENAME,
                label=f"{task_id} model post-use",
            )
            if checkpoint is not None:
                adapter_post = tree_json(
                    task_root / executor.ADAPTER_POST_USE_FILENAME,
                    label=f"{task_id} adapter post-use",
                )
                adapter_final = tree_json(
                    task_root / executor.ADAPTER_FINAL_FILENAME,
                    label=f"{task_id} adapter final",
                )
            consumption_chain = tree_json(
                task_root / executor.CONSUMPTION_CHAIN_FILENAME,
                label=f"{task_id} consumption chain C",
            )
            stdout = tree_bytes(
                task_root / executor.STDOUT_FILENAME, label=f"{task_id} stdout"
            )
            stderr = tree_bytes(
                task_root / executor.STDERR_FILENAME, label=f"{task_id} stderr"
            )
            process_receipt = tree_json(
                process_path, label=f"{task_id} process"
            )
            try:
                expected_process = executor.build_process_receipt(
                    input_receipt=request,
                    observation={
                        "return_code": process_receipt.get("return_code"),
                        "stdout": stdout,
                        "stderr": stderr,
                        "fd_inheritance": process_receipt.get("fd_inheritance"),
                    },
                    request_path=request_path,
                    staging_path=staging_path,
                    consumption_digest=consumption_chain.get("consumption_digest"),
                )
            except executor.DecodedEvaluationExecutorError as error:
                raise DecodedEvaluationAggregateError(str(error)) from error
            if process_receipt != expected_process or process_receipt["return_code"] != 0:
                raise DecodedEvaluationAggregateError(
                    f"{task_id} process receipt differs"
                )

            publication_gate = tree_json(
                task_root / executor.PUBLICATION_GATE_FILENAME,
                label=f"{task_id} publication gate",
            )
            output_receipt, output_receipt_sha = tree_json_with_sha(
                output_receipt_path, label=f"{task_id} output receipt"
            )
            projection = validate_offline_authority_chain(
                task_id=task_id,
                task_kind=task["task_kind"],
                physical_bindings_digest=bindings["physical_bindings_digest"],
                model_capture=model_capture,
                model_capture_path=model_capture_path,
                model_capture_sha256=model_capture_sha,
                model_pre=model_pre,
                model_post=model_post,
                adapter_capture=adapter_capture,
                adapter_capture_path=adapter_capture_path,
                adapter_capture_sha256=adapter_capture_sha,
                adapter_pre=adapter_pre,
                adapter_post=adapter_post,
                adapter_final=adapter_final,
                consumption_input=consumption_input,
                consumption_input_path=consumption_input_path,
                consumption_input_sha256=consumption_input_sha,
                task_input=request,
                native_receipt=native_receipt,
                consumption_chain=consumption_chain,
                process_receipt=process_receipt,
                publication_gate=publication_gate,
                output_receipt=output_receipt,
                result=result,
                shard_summary_digest=summary["summary_digest"],
                staging_path=staging_path,
                production_required=True,
            )

            task_relative = (
                Path(executor.EXECUTION_DIRECTORY) / job_id / "tasks" / task_id
            )
            final_relative = Path(task["record"]["output_relpath"])
            staging_media = _RetainedMediaFile.capture(
                staging_path,
                expected_sha256=output_receipt["output_video_sha256"],
                expected_size=output_receipt["output_byte_size"],
                expected_identity=output_receipt["published_inode_identity"],
                expected_nlink={2},
                parent_descriptor=final_publication_authority.directory_fd(
                    task_relative.as_posix()
                ),
            )
            retained_cleanup.append(staging_media)
            final_media = _RetainedMediaFile.capture(
                final_path,
                expected_sha256=output_receipt["output_video_sha256"],
                expected_size=output_receipt["output_byte_size"],
                expected_identity=output_receipt["published_inode_identity"],
                expected_nlink={2},
                parent_descriptor=final_publication_authority.directory_fd(
                    final_relative.parent.as_posix()
                ),
            )
            retained_cleanup.append(final_media)
            if staging_media.identity != final_media.identity:
                staging_media.close()
                final_media.close()
                raise DecodedEvaluationAggregateError(
                    f"{task_id} staging/publication inode differs"
                )
            try:
                output_receipt = executor.validate_output_receipt(
                    output_receipt,
                    input_receipt=request,
                    process_receipt=process_receipt,
                    output_path=final_path,
                    published_observation={
                        "identity": final_media.identity,
                        "sha256": final_media.sha256,
                        "size": final_media.size,
                    },
                )
                replayed_probe = executor.validate_probe_result(
                    reprobe_video(final_media.consumer_path())
                )
            except executor.DecodedEvaluationExecutorError as error:
                staging_media.close()
                final_media.close()
                raise DecodedEvaluationAggregateError(str(error)) from error
            if output_receipt["probe"] != replayed_probe:
                raise DecodedEvaluationAggregateError(
                    f"{task_id} current media probe differs"
                )

            try:
                decoder_capture = bridge.validate_verified_capture_receipt(
                    bindings,
                    receipt_path=task_root
                    / executor.DECODER_RUNTIME_CAPTURE_FILENAME,
                    target="action_preservation_decoded_eval_decoder_adapter_v1.py",
                    expected_arguments=[
                        "--request", str(request_path),
                        "--output", str(staging_path),
                    ],
                    verify_file=True,
                )
                inference_capture = bridge.validate_verified_capture_receipt(
                    bindings,
                    receipt_path=staging_path.with_name(
                        staging_path.name
                        + decoder_adapter.INFERENCE_RUNTIME_CAPTURE_SUFFIX
                    ),
                    target="infer_lora.py",
                    expected_arguments=_inference_arguments_offline(
                        request=request,
                        bindings=bindings,
                        source=source,
                        checkpoint=checkpoint,
                        model_capture=model_capture,
                        adapter_capture=adapter_capture,
                        consumption_input=consumption_input,
                        output_path=staging_path,
                    ),
                    verify_file=True,
                )
            except bridge.DecodedEvaluationBridgeError as error:
                raise DecodedEvaluationAggregateError(str(error)) from error
            native_evidence = {
                "receipt_path": str(native_receipt_path),
                "receipt_sha256": native_receipt_sha,
                "receipt_digest": native_receipt["receipt_digest"],
                "decoder_verified_release_capture": decoder_capture,
                "inference_verified_release_capture": inference_capture,
            }
            if output_receipt["native_inference_receipt"] != native_evidence:
                raise DecodedEvaluationAggregateError(
                    f"{task_id} native receipt evidence differs"
                )

            staging_media.replay(rehash=True)
            final_media.replay(rehash=True)
            staging_media.close()
            if (
                publication_gate["staging_sha256"]
                != output_receipt["output_video_sha256"]
                or publication_gate["staging_size"]
                != output_receipt["output_byte_size"]
            ):
                raise DecodedEvaluationAggregateError(
                    f"{task_id} sealed staging/publication evidence differs"
                )
            ordered_consumptions.append(projection["consumption_digest"])
            ordered_chain_digests.append(projection["authority_chain_digest"])
            outputs.append(
                {
                    "task_kind": task["task_kind"],
                    "task_id": task_id,
                    "record": task["record"],
                    "output_path": str(final_path),
                    "output_video_sha256": output_receipt["output_video_sha256"],
                    "output_receipt_path": str(output_receipt_path),
                    "output_receipt_sha256": output_receipt_sha,
                    "output_digest": output_receipt["output_digest"],
                    "consumption_digest": projection["consumption_digest"],
                    "publication_gate_digest": projection[
                        "publication_gate_digest"
                    ],
                    "result_digest": projection["result_digest"],
                    "authority_chain_digest": projection[
                        "authority_chain_digest"
                    ],
                    "_retained_media": final_media,
                }
            )

        model_final_value, model_final_sha = tree_json_with_sha(
            model_final_path, label=f"holder {job_id} model final"
        )
        model_final = _validate_model_final(
            model_final_value,
            model_capture=model_capture,
            ordered_consumption_digests=ordered_consumptions,
        )
        ordered_chain_digests_digest = object_sha256(ordered_chain_digests)
        if (
            summary["model_final_digest"] != model_final["model_final_digest"]
            or summary["task_consumption_set_digest"]
            != model_final["task_consumption_set_digest"]
        ):
            raise DecodedEvaluationAggregateError(
                "holder summary/model-final authority differs"
            )
        holder_authority = {
            "job_id": job_id,
            "holder_completion_anchor_digest": completion_anchor[
                "anchor_digest"
            ],
            "holder_directory_completion_digest": holder_completion[
                "completion_digest"
            ],
            "model_capture_digest": model_capture["capture_digest"],
            "model_final_digest": model_final["model_final_digest"],
            "task_consumption_set_digest": model_final[
                "task_consumption_set_digest"
            ],
            "ordered_chain_digests_digest": ordered_chain_digests_digest,
        }
        holder_authority["holder_authority_digest"] = object_sha256(
            holder_authority
        )
        summaries.append(
            {
                "job_id": job_id,
                "node": holder["node"],
                "summary_path": str(summary_path),
                "summary_sha256": summary_sha,
                "summary_digest": summary["summary_digest"],
                "holder_execution_digest": summary[
                    "holder_execution_authority"
                ]["holder_execution_digest"],
                "holder_directory_completion_path": completion_document[
                    "file"
                ]["path"],
                "holder_directory_completion_sha256": completion_document[
                    "file"
                ]["sha256"],
                "holder_directory_completion_digest": holder_completion[
                    "completion_digest"
                ],
                "holder_completion_anchor": completion_anchor,
                "executor_verified_release_capture": replayed_executor_capture,
                "model_capture_path": str(model_capture_path),
                "model_capture_sha256": model_capture_sha,
                "model_capture_digest": model_capture["capture_digest"],
                "model_final_path": str(model_final_path),
                "model_final_sha256": model_final_sha,
                "model_final_digest": model_final["model_final_digest"],
                "task_consumption_set_digest": model_final[
                    "task_consumption_set_digest"
                ],
                "ordered_chain_digests_digest": ordered_chain_digests_digest,
                "holder_authority_digest": holder_authority[
                    "holder_authority_digest"
                ],
                "all_task_fd_inheritance_evidence_verified": True,
            }
        )

    if (
        len(outputs) != 264
        or sum(item["task_kind"] == "adapter_candidate" for item in outputs) != 256
        or sum(item["task_kind"] == "frozen_base_control" for item in outputs) != 8
    ):
        raise DecodedEvaluationAggregateError("exact 264 output closure differs")
    try:
        replayed_bindings = bridge.load_physical_bindings(
            physical_identity["path"],
            expected_sha256=physical_identity["sha256"],
            verify_files=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if replayed_bindings != bindings:
        raise DecodedEvaluationAggregateError(
            "physical bindings changed during aggregate verification"
        )
    return bundle, bindings, outputs, summaries


def collect_verified_outputs(
    *,
    evaluation_root: str | Path,
    physical_bindings_path: str | Path,
    physical_bindings_sha256: str,
    work_root_binding: Mapping[str, Any] | None = None,
    holder_completion_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect exact264 evidence, closing every retained FD on failure.

    On success ownership of the final directory authority and final media FDs
    transfers to the returned bundle/outputs until ``publish`` completes.
    """

    retained_cleanup: list[Any] = []
    try:
        return _collect_verified_outputs_impl(
            evaluation_root=evaluation_root,
            physical_bindings_path=physical_bindings_path,
            physical_bindings_sha256=physical_bindings_sha256,
            work_root_binding=work_root_binding,
            holder_completion_anchors=holder_completion_anchors,
            retained_cleanup=retained_cleanup,
        )
    except BaseException:
        for retained in reversed(retained_cleanup):
            close = getattr(retained, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        raise


def _validate_holder_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if len(summaries) != len(plan.HOLDER_ROWS):
        raise DecodedEvaluationAggregateError(
            "blind packet four-holder closure differs"
        )
    holder_authority_fields = {
        "job_id", "node", "summary_path", "summary_sha256", "summary_digest",
        "holder_execution_digest", "executor_verified_release_capture",
        "holder_directory_completion_path",
        "holder_directory_completion_sha256",
        "holder_directory_completion_digest",
        "holder_completion_anchor",
        "model_capture_path", "model_capture_sha256", "model_capture_digest",
        "model_final_path", "model_final_sha256", "model_final_digest",
        "task_consumption_set_digest", "ordered_chain_digests_digest",
        "holder_authority_digest",
        "all_task_fd_inheritance_evidence_verified",
    }
    normalized_summaries: list[dict[str, Any]] = []
    holder_authority_rows: list[dict[str, str]] = []
    for expected_holder, item in zip(plan.HOLDER_ROWS, summaries):
        row = _closed(item, holder_authority_fields, label="aggregate holder row")
        capture = executor._capture_evidence(
            row["executor_verified_release_capture"],
            label="aggregate holder executor capture",
        )
        authority = {
            "job_id": row["job_id"],
            "holder_completion_anchor_digest": row[
                "holder_completion_anchor"
            ]["anchor_digest"],
            "holder_directory_completion_digest": row[
                "holder_directory_completion_digest"
            ],
            "model_capture_digest": row["model_capture_digest"],
            "model_final_digest": row["model_final_digest"],
            "task_consumption_set_digest": row["task_consumption_set_digest"],
            "ordered_chain_digests_digest": row[
                "ordered_chain_digests_digest"
            ],
        }
        if (
            row["job_id"] != expected_holder["job_id"]
            or row["node"] != expected_holder["node"]
            or row["all_task_fd_inheritance_evidence_verified"] is not True
            or capture is None
            or capture["target"]
            != "action_preservation_decoded_eval_executor_v2.py"
            or row["holder_authority_digest"] != object_sha256(authority)
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate holder authority differs"
            )
        try:
            completion_anchor = executor.validate_holder_completion_anchor(
                row["holder_completion_anchor"]
            )
        except executor.DecodedEvaluationExecutorError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        if (
            completion_anchor["holder_job_id"] != row["job_id"]
            or completion_anchor["completion_path"]
            != row["holder_directory_completion_path"]
            or completion_anchor["completion_sha256"]
            != row["holder_directory_completion_sha256"]
            or completion_anchor["completion_digest"]
            != row["holder_directory_completion_digest"]
            or completion_anchor["holder_summary_digest"]
            != row["summary_digest"]
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate holder completion anchor differs"
            )
        for key in (
            "summary_sha256", "summary_digest", "holder_execution_digest",
            "holder_directory_completion_sha256",
            "holder_directory_completion_digest",
            "model_capture_sha256", "model_capture_digest", "model_final_sha256",
            "model_final_digest", "task_consumption_set_digest",
            "ordered_chain_digests_digest", "holder_authority_digest",
        ):
            _sha(row[key], label=f"aggregate holder {key}")
        for key in (
            "summary_path", "holder_directory_completion_path",
            "model_capture_path", "model_final_path"
        ):
            _absolute(row[key], label=f"aggregate holder {key}")
        holder_authority_rows.append(authority)
        normalized_summaries.append(row)
    return normalized_summaries, holder_authority_rows


def build_blind_packet(
    *,
    bundle: Mapping[str, Any],
    bindings: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    blinding_key: bytes,
    aggregate_verified_release_capture: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(blinding_key) < 32:
        raise DecodedEvaluationAggregateError(
            "blinding key must contain at least 32 bytes"
        )
    candidates = [
        dict(item) for item in outputs if item["task_kind"] == "adapter_candidate"
    ]
    controls = {
        item["task_id"]: dict(item)
        for item in outputs
        if item["task_kind"] == "frozen_base_control"
    }
    if len(candidates) != 256 or len(controls) != 8 or len(outputs) != 264:
        raise DecodedEvaluationAggregateError(
            "blind packet exact-264 closure differs"
        )
    aggregate_capture = executor._capture_evidence(
        aggregate_verified_release_capture,
        label="aggregate verified release capture",
    )
    if (
        aggregate_capture is None
        or aggregate_capture["target"]
        != "action_preservation_decoded_eval_aggregate_v2.py"
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate verified release capture differs"
        )
    normalized_summaries, holder_authority_rows = _validate_holder_rows(
        summaries
    )

    sources = {item["iid"]: item for item in bindings["sources"]}
    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record = candidate["record"]
        control = controls.get(record["matched_frozen_base_control_id"])
        if control is None:
            raise DecodedEvaluationAggregateError(
                "matched frozen-base output is missing"
            )
        base = control["record"]
        if any(
            record[key] != base[key]
            for key in (
                "iid", "seed", "onset_policy", "source_video_sha256",
                "source_receipt_sha256", "instruction", "instruction_sha256",
                "action_review_contract", "model_release_manifest_sha256",
                "inference_source_sha256", "inference_release_manifest_sha256",
                "inference_config_sha256", "source_preprocessing_sha256",
                "calibration_digest",
            )
        ):
            raise DecodedEvaluationAggregateError(
                "matched frozen-base pairing differs"
            )
        blind_digest = hmac.new(
            blinding_key,
            ("id\0" + record["candidate_id"]).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        order_digest = hmac.new(
            blinding_key,
            ("order\0" + record["candidate_id"]).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        blind_id = "blind-" + blind_digest[:32]
        source = sources[record["iid"]]["source_video"]
        public_row: dict[str, Any] = {
            "blind_candidate_id": blind_id,
            "source_media_sha256": source["sha256"],
            "source_receipt_sha256": record["source_receipt_sha256"],
            "source_media_relpath": f"{MEDIA_DIRECTORY}/{source['sha256']}.mp4",
            "review_media_sha256": candidate["output_video_sha256"],
            "review_media_relpath": (
                f"{MEDIA_DIRECTORY}/{candidate['output_video_sha256']}.mp4"
            ),
            "review_output_digest": candidate["output_digest"],
            "full_video_receipt_sha256": candidate["output_receipt_sha256"],
            "matched_base_media_sha256": control["output_video_sha256"],
            "matched_base_media_relpath": (
                f"{MEDIA_DIRECTORY}/{control['output_video_sha256']}.mp4"
            ),
            "matched_base_output_digest": control["output_digest"],
            "matched_base_full_video_receipt_sha256": control[
                "output_receipt_sha256"
            ],
            "instruction": record["instruction"],
            "instruction_sha256": record["instruction_sha256"],
            "action_review_contract": record["action_review_contract"],
            "action_review_contract_digest": record["action_review_contract"][
                "contract_digest"
            ],
            "required_axes": list(plan.REVIEW_AXES),
            "minimum_independent_reviewer_count": 2,
            "full_81_frame_video_required": True,
        }
        public_row["blind_row_digest"] = object_sha256(public_row)
        public_rows.append(public_row)
        private_row: dict[str, Any] = {
            "blind_candidate_id": blind_id,
            "blind_row_digest": public_row["blind_row_digest"],
            "order_digest": order_digest,
            "candidate_id": record["candidate_id"],
            "arm": record["arm"],
            "checkpoint_step": record["checkpoint_step"],
            "iid": record["iid"],
            "onset_policy": record["onset_policy"]["name"],
            "matched_control_id": control["task_id"],
            "candidate_output_path": candidate["output_path"],
            "candidate_output_receipt_path": candidate["output_receipt_path"],
            "candidate_output_receipt_sha256": candidate[
                "output_receipt_sha256"
            ],
            "candidate_output_digest": candidate["output_digest"],
            "matched_base_output_receipt_path": control["output_receipt_path"],
            "matched_base_output_receipt_sha256": control[
                "output_receipt_sha256"
            ],
            "matched_base_output_digest": control["output_digest"],
            "instruction_sha256": record["instruction_sha256"],
            "action_review_contract_digest": record["action_review_contract"][
                "contract_digest"
            ],
        }
        private_row["private_row_digest"] = object_sha256(private_row)
        private_rows.append(private_row)
    ordering = {
        row["blind_candidate_id"]: row["order_digest"] for row in private_rows
    }
    private_rows.sort(key=lambda item: item["order_digest"])
    public_rows.sort(key=lambda item: ordering[item["blind_candidate_id"]])
    if len({item["blind_candidate_id"] for item in public_rows}) != 256:
        raise DecodedEvaluationAggregateError("blind identifiers collide")
    key_sha = hashlib.sha256(blinding_key).hexdigest()
    private: dict[str, Any] = {
        "schema_version": PRIVATE_PACKET_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "blinding_key_sha256": key_sha,
        "rows": private_rows,
        "row_count": 256,
        "method_arm_checkpoint_policy_private": True,
    }
    private["private_mapping_digest"] = object_sha256(private)
    public: dict[str, Any] = {
        "schema_version": PUBLIC_PACKET_SCHEMA,
        "packet_id": "packet-"
        + hmac.new(blinding_key, b"packet", hashlib.sha256).hexdigest()[:32],
        "review_contract_digest": bundle["review_contract"]["contract_digest"],
        "private_mapping_digest": private["private_mapping_digest"],
        "rows": public_rows,
        "row_count": 256,
        "method_hidden": True,
        "arm_hidden": True,
        "checkpoint_hidden": True,
        "onset_policy_hidden": True,
        "private_key_in_public_packet": False,
        "training_loss_present": False,
    }
    public["public_packet_digest"] = object_sha256(public)
    holder_authority_set_digest = object_sha256(holder_authority_rows)
    ordered_task_chain_set_digest = object_sha256(
        [item["authority_chain_digest"] for item in outputs]
    )
    calibration = bindings["calibration_digest"]
    aggregate: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "physical_bindings_digest": bindings["physical_bindings_digest"],
        "holder_summaries": normalized_summaries,
        "holder_count": 4,
        "holder_authority_set_digest": holder_authority_set_digest,
        "ordered_task_authority_chain_set_digest": ordered_task_chain_set_digest,
        "candidate_output_count": 256,
        "matched_base_output_count": 8,
        "total_output_count": 264,
        "exact_full81_at_25fps_pts_verified": True,
        "all_native_inference_receipts_verified": True,
        "all_model_and_adapter_consumption_authority_verified_offline": True,
        "all_fd_inheritance_evidence_verified": True,
        "all_consumption_publication_gates_verified": True,
        "all_outputs_create_only_and_sealed": True,
        "aggregate_verified_release_capture": aggregate_capture,
        "automatic_retry_count": 0,
        "training_loss_read_or_used": False,
        "checkpoint_loss_ranking": False,
        "private_mapping_digest": private["private_mapping_digest"],
        "public_packet_digest": public["public_packet_digest"],
        "blinding_key_sha256": key_sha,
        "machine_calibration_digest": calibration,
        "machine_status": (
            "WAIT_FOR_MACHINE_MEASUREMENT"
            if calibration is not None
            else "ABSTAIN_CALIBRATION_MISSING"
        ),
        "blind_review_status": "WAIT_FOR_BLIND_REVIEW",
        "next_action": "WAIT_FOR_BLIND_REVIEW",
        "scientific_promotion_authorized": False,
    }
    aggregate["aggregate_digest"] = object_sha256(aggregate)
    return aggregate, private, public


def _write(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise DecodedEvaluationAggregateError(
            f"refusing to overwrite: {path}"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise DecodedEvaluationAggregateError(
                    "create-only write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_media(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    destination_directory: _HeldPublicationDirectory | None = None,
) -> None:
    source = _plain(source, directory=False, label="blind packet source media")
    if destination_directory is not None:
        if destination.parent != destination_directory.path:
            raise DecodedEvaluationAggregateError(
                "blind media destination parent differs"
            )
        destination_directory.replay()
    source_flags = os.O_RDONLY
    output_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        output_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    try:
        before = os.fstat(source_descriptor)
        try:
            output_descriptor = os.open(
                destination.name if destination_directory is not None else destination,
                output_flags,
                0o444,
                **(
                    {"dir_fd": destination_directory.descriptor}
                    if destination_directory is not None
                    else {}
                ),
            )
        except FileExistsError as error:
            raise DecodedEvaluationAggregateError(
                f"refusing to overwrite: {destination}"
            ) from error
        first_digest = hashlib.sha256()
        try:
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                first_digest.update(block)
                offset = 0
                while offset < len(block):
                    written = os.write(output_descriptor, block[offset:])
                    if written <= 0:
                        raise DecodedEvaluationAggregateError(
                            "blind media copy made no progress"
                        )
                    offset += written
            os.fchmod(output_descriptor, 0o444)
            os.fsync(output_descriptor)
            middle = os.fstat(source_descriptor)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            second_digest = hashlib.sha256()
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                second_digest.update(block)
            destination_before = os.fstat(output_descriptor)
            os.lseek(output_descriptor, 0, os.SEEK_SET)
            destination_digest = hashlib.sha256()
            while True:
                block = os.read(output_descriptor, 1024 * 1024)
                if not block:
                    break
                destination_digest.update(block)
            destination_after = os.fstat(output_descriptor)
            destination_named = (
                os.stat(
                    destination.name,
                    dir_fd=destination_directory.descriptor,
                    follow_symlinks=False,
                )
                if destination_directory is not None
                else destination.lstat()
            )
        finally:
            os.close(output_descriptor)
        after = os.fstat(source_descriptor)
        named = source.lstat()
    finally:
        os.close(source_descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev, item.st_ino, item.st_uid, item.st_gid, item.st_mode,
            item.st_nlink, item.st_rdev, item.st_size,
            getattr(item, "st_blocks", 0), item.st_mtime_ns, item.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink not in {1, 2}
        or identity(before) != identity(middle)
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or first_digest.hexdigest() != expected_sha256
        or second_digest.hexdigest() != expected_sha256
        or destination_digest.hexdigest() != expected_sha256
        or identity(destination_before) != identity(destination_after)
        or identity(destination_before) != identity(destination_named)
        or destination_before.st_nlink != 1
        or stat.S_IMODE(destination_before.st_mode) != 0o444
    ):
        raise DecodedEvaluationAggregateError(
            "blind packet source media changed or differs"
        )
    if destination_directory is not None:
        os.fsync(destination_directory.descriptor)
        destination_directory._refresh_after_mutation(
            destination_directory.entries | {destination.name}
        )


def _capture_aggregate_json_anchor(
    directory: _HeldPublicationDirectory,
    *,
    name: str,
    value: Mapping[str, Any],
    mode: int,
    digest_field: str,
) -> dict[str, Any]:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor: int | None = None
    try:
        directory.replay()
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory.descriptor,
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first_sha, first_size = executor._hash_fd(descriptor)
        first = os.pread(descriptor, first_size, 0)
        middle = os.fstat(descriptor)
        second_sha, second_size = executor._hash_fd(descriptor)
        second = os.pread(descriptor, second_size, 0)
        after = os.fstat(descriptor)
        named = os.stat(
            name, dir_fd=directory.descriptor, follow_symlinks=False
        )
    except OSError as error:
        raise DecodedEvaluationAggregateError(
            f"aggregate completion file replay failed: {name}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = executor._stat_identity_row(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != mode
        or identity != executor._stat_identity_row(middle)
        or identity != executor._stat_identity_row(after)
        or identity != executor._stat_identity_row(named)
        or first_sha != second_sha
        or first_size != second_size
        or first != second
        or first != payload
        or value.get(digest_field) != object_sha256(
            {key: item for key, item in value.items() if key != digest_field}
        )
    ):
        raise DecodedEvaluationAggregateError(
            f"aggregate completion file differs: {name}"
        )
    return {
        "relative_path": name,
        "sha256": first_sha,
        "size": first_size,
        "mode": mode,
        "identity": identity,
        "object_digest": value[digest_field],
    }


def _capture_aggregate_media_rows(
    media_directory: _HeldPublicationDirectory,
) -> list[dict[str, Any]]:
    media_directory.replay()
    rows: list[dict[str, Any]] = []
    for name in sorted(media_directory.entries):
        match = re.fullmatch(r"([0-9a-f]{64})\.mp4", name)
        if match is None:
            raise DecodedEvaluationAggregateError(
                "aggregate completion media basename differs"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=media_directory.descriptor,
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first_sha, first_size = executor._hash_fd(descriptor)
            middle = os.fstat(descriptor)
            second_sha, second_size = executor._hash_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name,
                dir_fd=media_directory.descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise DecodedEvaluationAggregateError(
                "aggregate completion media replay failed"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identity = executor._stat_identity_row(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or identity != executor._stat_identity_row(middle)
            or identity != executor._stat_identity_row(after)
            or identity != executor._stat_identity_row(named)
            or first_sha != second_sha
            or first_size != second_size
            or first_sha != match.group(1)
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate completion media bytes differ"
            )
        rows.append(
            {
                "relative_path": f"{MEDIA_DIRECTORY}/{name}",
                "sha256": first_sha,
                "size": first_size,
                "mode": 0o444,
                "identity": identity,
            }
        )
    media_directory.replay()
    return rows


def validate_aggregate_completion_anchor(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "aggregate_root",
        "aggregate_root_identity", "aggregate_file", "private_file",
        "public_file", "media_directory_identity", "media_file_count",
        "media_rows_digest", "media_tree_digest", "anchor_digest",
    }
    row = dict(_closed(value, fields, label="aggregate completion anchor"))
    root_identity = _identity(
        row["aggregate_root_identity"], label="aggregate root identity"
    )
    media_identity = _identity(
        row["media_directory_identity"], label="aggregate media identity"
    )
    expected_files = (
        ("aggregate_file", AGGREGATE_FILENAME, 0o444),
        ("private_file", PRIVATE_FILENAME, 0o400),
        ("public_file", PUBLIC_FILENAME, 0o444),
    )
    normalized_files: dict[str, dict[str, Any]] = {}
    file_fields = {
        "relative_path", "sha256", "size", "mode", "identity",
        "object_digest",
    }
    for field, relative, mode in expected_files:
        item = dict(_closed(row[field], file_fields, label=field))
        identity = _identity(item["identity"], label=f"{field} identity")
        if (
            item["relative_path"] != relative
            or item["mode"] != mode
            or type(item["size"]) is not int
            or item["size"] <= 0
            or stat.S_IMODE(identity["mode"]) != mode
            or identity["nlink"] != 1
            or identity["size"] != item["size"]
        ):
            raise DecodedEvaluationAggregateError(
                "aggregate completion file binding differs"
            )
        _sha(item["sha256"], label=f"{field} SHA")
        _sha(item["object_digest"], label=f"{field} object digest")
        item["identity"] = identity
        normalized_files[field] = item
    if (
        row["schema_version"] != AGGREGATE_COMPLETION_ANCHOR_SCHEMA
        or not isinstance(row["evaluation_id"], str)
        or not row["evaluation_id"]
        or not isinstance(row["aggregate_root"], str)
        or not Path(row["aggregate_root"]).is_absolute()
        or os.path.normpath(row["aggregate_root"]) != row["aggregate_root"]
        or not stat.S_ISDIR(root_identity["mode"])
        or stat.S_IMODE(root_identity["mode"]) != 0o555
        or not stat.S_ISDIR(media_identity["mode"])
        or stat.S_IMODE(media_identity["mode"]) != 0o555
        or type(row["media_file_count"]) is not int
        or row["media_file_count"] <= 0
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate completion anchor binding differs"
        )
    for field in ("media_rows_digest", "media_tree_digest", "anchor_digest"):
        _sha(row[field], label=f"aggregate completion {field}")
    expected_tree = {
        "media_directory_identity": media_identity,
        "media_file_count": row["media_file_count"],
        "media_rows_digest": row["media_rows_digest"],
    }
    if row["media_tree_digest"] != object_sha256(expected_tree):
        raise DecodedEvaluationAggregateError(
            "aggregate completion media tree digest differs"
        )
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationAggregateError(
            "aggregate completion anchor digest differs"
        )
    row.update(normalized_files)
    row["aggregate_root_identity"] = root_identity
    row["media_directory_identity"] = media_identity
    return row


def build_aggregate_completion_anchor(
    *,
    held_root: _HeldPublicationDirectory,
    media_root: _HeldPublicationDirectory,
    aggregate: Mapping[str, Any],
    private: Mapping[str, Any],
    public: Mapping[str, Any],
) -> dict[str, Any]:
    held_root.replay(
        expected_entries={
            MEDIA_DIRECTORY,
            PRIVATE_FILENAME,
            PUBLIC_FILENAME,
            AGGREGATE_FILENAME,
        }
    )
    media_rows = _capture_aggregate_media_rows(media_root)
    value: dict[str, Any] = {
        "schema_version": AGGREGATE_COMPLETION_ANCHOR_SCHEMA,
        "evaluation_id": aggregate["evaluation_id"],
        "aggregate_root": str(held_root.path),
        "aggregate_root_identity": _directory_identity(
            os.fstat(held_root.descriptor)
        ),
        "aggregate_file": _capture_aggregate_json_anchor(
            held_root,
            name=AGGREGATE_FILENAME,
            value=aggregate,
            mode=0o444,
            digest_field="aggregate_digest",
        ),
        "private_file": _capture_aggregate_json_anchor(
            held_root,
            name=PRIVATE_FILENAME,
            value=private,
            mode=0o400,
            digest_field="private_mapping_digest",
        ),
        "public_file": _capture_aggregate_json_anchor(
            held_root,
            name=PUBLIC_FILENAME,
            value=public,
            mode=0o444,
            digest_field="public_packet_digest",
        ),
        "media_directory_identity": _directory_identity(
            os.fstat(media_root.descriptor)
        ),
        "media_file_count": len(media_rows),
        "media_rows_digest": object_sha256(media_rows),
    }
    value["media_tree_digest"] = object_sha256(
        {
            "media_directory_identity": value["media_directory_identity"],
            "media_file_count": value["media_file_count"],
            "media_rows_digest": value["media_rows_digest"],
        }
    )
    value["anchor_digest"] = object_sha256(value)
    anchor = validate_aggregate_completion_anchor(value)
    held_root.replay()
    media_root.replay()
    return anchor


def publish(
    *,
    aggregate_root: str | Path,
    aggregate: Mapping[str, Any],
    private: Mapping[str, Any],
    public: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    work_root_binding: Mapping[str, Any] | None = None,
    completion_anchor_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Path:
    root = Path(aggregate_root)
    held_root = (
        _HeldPublicationDirectory.create_root(root)
        if work_root_binding is None
        else _HeldPublicationDirectory.create_root_from_work_binding(
            root, work_root_binding=work_root_binding
        )
    )
    media_root: _HeldPublicationDirectory | None = None
    retained_media: dict[str, _RetainedMediaFile] = {}
    all_retained_media: list[_RetainedMediaFile] = []
    try:
        media_root = held_root.mkdir(MEDIA_DIRECTORY)
        for source in bindings["sources"]:
            binding = source["source_video"]
            retained = _RetainedMediaFile.capture(
                Path(binding["path"]),
                expected_sha256=binding["sha256"],
                expected_size=binding["size"],
                expected_identity={
                    field: binding[field] for field in _IDENTITY_FIELDS
                },
                expected_nlink={1},
            )
            all_retained_media.append(retained)
            existing = retained_media.get(retained.sha256)
            if existing is None:
                retained_media[retained.sha256] = retained
            else:
                existing.replay(rehash=True)
                retained.replay(rehash=True)
                if existing.size != retained.size:
                    raise DecodedEvaluationAggregateError(
                        "aggregate duplicate media size differs"
                    )
        for item in outputs:
            retained = item.get("_retained_media")
            if (
                not isinstance(retained, _RetainedMediaFile)
                or retained.closed
                or retained.sha256 != item["output_video_sha256"]
            ):
                raise DecodedEvaluationAggregateError(
                    "aggregate output lacks retained media authority"
                )
            all_retained_media.append(retained)
            existing = retained_media.get(retained.sha256)
            if existing is None:
                retained_media[retained.sha256] = retained
            else:
                existing.replay(rehash=True)
                retained.replay(rehash=True)
                if existing.size != retained.size:
                    raise DecodedEvaluationAggregateError(
                        "aggregate duplicate media size differs"
                    )
        expected_media_names: set[str] = set()
        for digest, retained in sorted(retained_media.items()):
            basename = f"{digest}.mp4"
            retained.copy_to(
                destination_directory=media_root,
                basename=basename,
            )
            expected_media_names.add(basename)
        media_root.seal(mode=0o555, expected_entries=expected_media_names)
        held_root.write(
            PRIVATE_FILENAME,
            canonical_json_bytes(private) + b"\n",
            mode=0o400,
        )
        held_root.write(
            PUBLIC_FILENAME,
            canonical_json_bytes(public) + b"\n",
            mode=0o444,
        )
        output = held_root.write(
            AGGREGATE_FILENAME,
            canonical_json_bytes(aggregate) + b"\n",
            mode=0o444,
        )
        held_root.seal(
            mode=0o555,
            expected_entries={
                MEDIA_DIRECTORY,
                PRIVATE_FILENAME,
                PUBLIC_FILENAME,
                AGGREGATE_FILENAME,
            },
        )
        if completion_anchor_sink is None and work_root_binding is not None:
            raise DecodedEvaluationAggregateError(
                "production aggregate lacks completion anchor channel"
            )
        if completion_anchor_sink is not None:
            anchor = build_aggregate_completion_anchor(
                held_root=held_root,
                media_root=media_root,
                aggregate=aggregate,
                private=private,
                public=public,
            )
            completion_anchor_sink(anchor)
            if build_aggregate_completion_anchor(
                held_root=held_root,
                media_root=media_root,
                aggregate=aggregate,
                private=private,
                public=public,
            ) != anchor:
                raise DecodedEvaluationAggregateError(
                    "aggregate completion changed after anchor publication"
                )
        return output
    finally:
        closed: set[int] = set()
        for retained in all_retained_media:
            if id(retained) not in closed:
                retained.close()
                closed.add(id(retained))
        if media_root is not None:
            media_root.close()
        held_root.close()


def _read_inherited_work_root_member(
    work_root_binding: Mapping[str, Any],
    *,
    path: str | Path,
    expected_sha256: str,
    expected_mode: int,
    label: str,
) -> bytes:
    """Read one direct WORK_ROOT member through the inherited root FD.

    The literal digest comes from the phase-two launch authority.  Never
    reopen this input through its absolute pathname: the evaluation UID can
    rename and replace that pathname while the aggregate process is alive.
    """

    expected_sha = _sha(expected_sha256, label=f"{label} file")
    try:
        live = bridge.verified_release.validate_inherited_work_root_binding(
            work_root_binding,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    member = Path(path)
    root = Path(live["path"])
    if (
        not member.is_absolute()
        or os.path.normpath(str(member)) != str(member)
        or member.parent != root
        or member.name in ("", ".", "..")
    ):
        raise DecodedEvaluationAggregateError(
            f"{label} is not a direct inherited WORK_ROOT member"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            member.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=live["root_fd"],
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first_sha, first_size = executor._hash_fd(descriptor)
        first = os.pread(descriptor, first_size, 0)
        middle = os.fstat(descriptor)
        second_sha, second_size = executor._hash_fd(descriptor)
        second = os.pread(descriptor, second_size, 0)
        after = os.fstat(descriptor)
        named = os.stat(
            member.name,
            dir_fd=live["root_fd"],
            follow_symlinks=False,
        )
    except OSError as error:
        raise DecodedEvaluationAggregateError(
            f"cannot read retained {label}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
        or executor._stat_identity(before) != executor._stat_identity(middle)
        or executor._stat_identity(before) != executor._stat_identity(after)
        or executor._stat_identity(before) != executor._stat_identity(named)
        or first_sha != second_sha
        or first_size != second_size
        or first != second
        or len(first) != first_size
        or first_sha != expected_sha
    ):
        raise DecodedEvaluationAggregateError(
            f"retained {label} same-FD replay differs"
        )
    try:
        bridge.verified_release.validate_inherited_work_root_binding(
            live,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    return first


def _close_collected_authorities(
    bundle: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]],
) -> None:
    for item in outputs:
        retained = item.get("_retained_media")
        if isinstance(retained, _RetainedMediaFile):
            retained.close()
    final_authority = bundle.get("_final_publication_authority")
    if isinstance(final_authority, plan.RetainedPublicationRoot):
        final_authority.close()


def _finish_aggregate_main(
    *,
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    inherited_work_root: Mapping[str, Any],
    blinding_key: bytes,
    bundle: Mapping[str, Any],
    bindings: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> Path:
    expected_arguments = list(sys.argv[1:] if argv is None else argv)
    expected_capture_path = Path(args.aggregate_runtime_capture_receipt)
    if (
        not expected_capture_path.is_absolute()
        or os.path.normpath(str(expected_capture_path))
        != str(expected_capture_path)
        or expected_capture_path.parent != Path(inherited_work_root["path"])
        or expected_capture_path.name in ("", ".", "..")
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate runtime capture path differs"
        )
    if inherited_work_root["capture_receipt_path"] != str(
        expected_capture_path
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate inherited capture path differs"
        )
    try:
        aggregate_capture = bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_aggregate_v2.py",
            expected_arguments=expected_arguments,
            verify_file=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if aggregate_capture["receipt_path"] != str(expected_capture_path):
        raise DecodedEvaluationAggregateError(
            "aggregate runtime capture path differs"
        )
    aggregate, private, public = build_blind_packet(
        bundle=bundle,
        bindings=bindings,
        outputs=outputs,
        summaries=summaries,
        blinding_key=blinding_key,
        aggregate_verified_release_capture=aggregate_capture,
    )
    return publish(
        aggregate_root=args.aggregate_root,
        aggregate=aggregate,
        private=private,
        public=public,
        outputs=outputs,
        bindings=bindings,
        work_root_binding=inherited_work_root,
        completion_anchor_sink=(
            bridge.verified_release.publish_aggregate_completion_anchor
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--physical-bindings", required=True)
    parser.add_argument("--physical-bindings-sha256", required=True)
    parser.add_argument("--blinding-key-file", required=True)
    parser.add_argument("--blinding-key-sha256", required=True)
    parser.add_argument("--aggregate-root", required=True)
    parser.add_argument("--aggregate-runtime-capture-receipt", required=True)
    parser.add_argument(
        "--holder-completion-anchor", action="append", required=True
    )
    args = parser.parse_args(argv)
    holder_completion_anchors: list[dict[str, Any]] = []
    for raw in args.holder_completion_anchor:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise DecodedEvaluationAggregateError(
                "holder completion anchor argument is not JSON"
            ) from error
        if (
            not isinstance(value, Mapping)
            or canonical_json_bytes(value).decode("utf-8") != raw
        ):
            raise DecodedEvaluationAggregateError(
                "holder completion anchor argument is not canonical"
            )
        holder_completion_anchors.append(dict(value))
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
        raise DecodedEvaluationAggregateError(str(error)) from error
    if (
        inherited_work_root["target"]
        != "action_preservation_decoded_eval_aggregate_v2.py"
        or Path(args.aggregate_root).parent
        != Path(inherited_work_root["path"])
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate inherited work-root authority differs"
        )
    key_path = Path(args.blinding_key_file)
    blinding_key = _read_inherited_work_root_member(
        inherited_work_root,
        path=key_path,
        expected_sha256=args.blinding_key_sha256,
        expected_mode=0o400,
        label="blinding key",
    )
    bundle, bindings, outputs, summaries = collect_verified_outputs(
        evaluation_root=args.evaluation_root,
        physical_bindings_path=args.physical_bindings,
        physical_bindings_sha256=args.physical_bindings_sha256,
        work_root_binding=inherited_work_root,
        holder_completion_anchors=holder_completion_anchors,
    )
    try:
        output = _finish_aggregate_main(
            args=args,
            argv=argv,
            inherited_work_root=inherited_work_root,
            blinding_key=blinding_key,
            bundle=bundle,
            bindings=bindings,
            outputs=outputs,
            summaries=summaries,
        )
    finally:
        _close_collected_authorities(bundle, outputs)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
