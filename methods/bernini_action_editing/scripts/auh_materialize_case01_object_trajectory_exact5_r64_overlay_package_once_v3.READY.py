#!/usr/bin/env python3
"""One-shot receipt-gated exact35 + exact6 overlay -> fresh r64 HOLD package.

This source is checked in HOLD until the exact35 manifest, its final sibling
0400 publication receipt, and the published root identity exist and are
pinned.  A future reviewed state-only READY copy opens the publication receipt
before the exact35 root, retains the receipt/root/manifest/materializer,
Python, and controller authorities, performs the pinned materializer preflight,
and durably claims one attempt before one isolated materializer child.

The materializer owns the NFS-truthful receipt-reserved shadow publication.
This controller independently consumes the package sibling receipt first,
then replays the internal materialization receipt and exact package tree.  It
contains no SSH, Slurm, renderer, launch, or retry path.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-r64-overlay-package-controller-v3"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
EVIDENCE_SCHEMA = SCHEMA + "-evidence"
CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_R64_HOLD_PACKAGE"
READY_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_R64_HOLD_PACKAGE"

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
SNAPSHOT_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1"
)
SNAPSHOT_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "receipt_v2.json"
)
SNAPSHOT_MANIFEST_NAME = (
    "case01_object_trajectory_exact5_source_snapshot_manifest_v2.json"
)
SNAPSHOT_MANIFEST_PATH = SNAPSHOT_ROOT / SNAPSHOT_MANIFEST_NAME
PACKAGE_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3"
)
PACKAGE_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "publication_receipt_v4.json"
)
ATTEMPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "materialize_attempt_v3.json"
)
EVIDENCE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "materialize_controller_evidence_v3.json"
)
INTERNAL_RECEIPT_RELATIVE = "authority/package_materialization_receipt_v4.json"
INTERNAL_RECEIPT_PATH = PACKAGE_ROOT / INTERNAL_RECEIPT_RELATIVE
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-"
    "rank-cache"
)

OVERLAY_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r5f_v4_source_overlay_6_20260822_r1"
)
OVERLAY_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r5f_v4_source_overlay_6_20260822_r1."
    "receipt_v1.json"
)

MATERIALIZER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_overlay_package_v3.py"
)
MATERIALIZER_PATH = OVERLAY_ROOT / MATERIALIZER_RELATIVE
MATERIALIZER_SHA256 = (
    "63196562f8f036cae8c54ba7799dd3695ee6805f01a3dd9756b1167eaa7f3d13"
)
MATERIALIZER_SIZE = 119_970
BASE_MATERIALIZER_SHA256 = (
    "31c0184c8187fe0224c92bcb425dd0ec27731e7197898bd552aef82f83fa49f9"
)
OVERLAY_RECEIPT_SHA256 = (
    "5827df34b30496b3b768f26e4be91de71b4a54dda0da000f0b00373549146be4"
)
OVERLAY_RECEIPT_SIZE = 3_915
OVERLAY_RECEIPT_DIGEST = (
    "463bf54d848730ac8b13a625c8a66e436c5523b7595dcda29095fe194986baa2"
)
OVERLAY_ROOT_IDENTITY: list[int] = [
    48, 5704346356003806783, 2012, 2000, 16749, 2, 0, 4096, 0,
    1787377341057464968, 1787377456971845705,
]
VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
VACE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
VACE_PYTHON_SIZE = 31_490_256
REMOTE_UID = 2012
REMOTE_GID = 2000
JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"

SNAPSHOT_MANIFEST_SCHEMA = "case01-object-trajectory-exact5-source-snapshot-v2"
SNAPSHOT_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-source-snapshot-publication-v2-receipt"
)
PACKAGE_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v4-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v4"
)
OVERLAY_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-r5f-v4-source-overlay-v1-receipt"
)
OVERLAY_PUBLICATION_PROTOCOL = (
    "fresh_precreated_exact_tree_then_held_files_sealed_and_sibling_O_EXCL_receipt"
)
PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
PRODUCTION_IDENTITY_ROLES = (
    "runner", "legacy_exact5_runner", "object_eval", "legacy_exact5_eval",
    "frozen_runner", "bridge", "adapter", "object_wrapper_inner",
    "legacy_infer_alias", "trajectory_projection",
    "trajectory_scaffold_module", "base_adapter", "eval_v1", "eval_v2",
    "model_authority", "torchrun_source", "torchrun_handler_source",
    "torch_local_agent_source", "torch_dynamic_rendezvous_source",
    "torch_multiprocessing_api_source", "base_model_manifest",
    "r64_checkpoint_manifest", "python", "ffmpeg", "ffprobe", "plan",
)
if len(PRODUCTION_IDENTITY_ROLES) != 26:
    raise RuntimeError("controller exact26 role closure differs")

# Exact immutable authorities produced by the reviewed builder READY run.
# This HOLD is audited with those coordinates before its state-only READY copy.
SNAPSHOT_MANIFEST_SHA256 = (
    "da9c070e012ff11ebf5c61115d9949a573d37a99701fb7e6b8d7b2a6d5eee8f9"
)
SNAPSHOT_MANIFEST_SIZE = 10_889
SNAPSHOT_MANIFEST_DIGEST = (
    "9114102687bf58291d6b96eddebc15557c669faa51e51bfde9b26a1aa7040968"
)
SNAPSHOT_RECEIPT_SHA256 = (
    "ea7089857a8593734603d544aa6f3c238ea06abfd2da6775de7fea2adf0ce2a4"
)
SNAPSHOT_RECEIPT_SIZE = 2_072
SNAPSHOT_RECEIPT_DIGEST = (
    "49ea7163e41382dff45ba59b0eb4b9e35480dead8426c7399d74da80b2f110a3"
)
SNAPSHOT_ROOT_IDENTITY = [
    48, 6200596844122101067, 2012, 2000, 16749, 2, 0, 4096, 0,
    1787356256218061495, 1787356256279241444,
]

FILE_MODE = 0o444
RECEIPT_MODE = 0o400
DIRECTORY_MODE = 0o555
CONTROLLER_MODE = 0o444
MAX_JSON_SIZE = 16 * 1024 * 1024
MAX_SOURCE_SIZE = 2 * 1024 * 1024
MAX_PACKAGE_FILE_SIZE = 32 * 1024 * 1024
MATERIALIZER_TIMEOUT_SECONDS = 300
PROCESS_POLL_SECONDS = 0.02
PROCESS_TERM_GRACE_SECONDS = 2.0
PROCESS_KILL_GRACE_SECONDS = 2.0
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class PackageControllerError(RuntimeError):
    """The reviewed one-shot materialization contract differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PackageControllerError("value is not canonical JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(info.st_mode), int(info.st_nlink), int(info.st_rdev),
        int(info.st_size), int(getattr(info, "st_blocks", 0)),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def inode_anchor(info: os.stat_result) -> list[int]:
    return [
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)),
    ]


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PackageControllerError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise PackageControllerError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise PackageControllerError(f"noncanonical JSON authority: {label}")
    return value


def strict_child_stdout(raw: bytes) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PackageControllerError("duplicate child stdout key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise PackageControllerError("materializer stdout is invalid JSON") from error
    expected = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    if type(value) is not dict or raw != expected:
        raise PackageControllerError("materializer stdout is not one exact JSON row")
    return value


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise PackageControllerError("held read size differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        chunks.append(block); offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise PackageControllerError("held read is incomplete")
    return raw


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...], raw: bytes,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.held_identity = held_identity
        self.raw = raw

    def row(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw), "identity": list(self.held_identity),
        }

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or read_fd(self.descriptor, opened.st_size) != self.raw
        ):
            raise PackageControllerError(f"held authority changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


class HeldDirectory:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...],
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.held_identity = held_identity

    def row(self) -> dict[str, Any]:
        return {"path": str(self.path), "identity": list(self.held_identity)}

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
        ):
            raise PackageControllerError(f"held directory changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def open_authority(
    path: Path, *, expected_sha256: str | None, expected_size: int | None,
    expected_mode: int, expected_uid: int, expected_gid: int,
    executable: bool = False, maximum_size: int = MAX_JSON_SIZE,
) -> HeldAuthority:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise PackageControllerError(f"noncanonical authority path: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise PackageControllerError(f"missing authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != expected_mode
        or named.st_uid != expected_uid or named.st_gid != expected_gid
        or named.st_size <= 0 or named.st_size > maximum_size
        or (expected_size is not None and named.st_size != expected_size)
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise PackageControllerError(f"named authority differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            identity(before) != identity(named)
            or identity(before) != identity(middle)
            or identity(before) != identity(after)
            or identity(before) != identity(named_after)
            or first != second
            or (expected_sha256 is not None
                and hashlib.sha256(first).hexdigest() != expected_sha256)
            or (expected_size is not None and len(first) != expected_size)
        ):
            raise PackageControllerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), first)
    except BaseException:
        os.close(descriptor)
        raise


def open_directory(
    path: Path, *, expected_identity: Sequence[int], expected_mode: int,
    expected_uid: int, expected_gid: int,
) -> HeldDirectory:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or type(expected_identity) not in (list, tuple)
        or len(expected_identity) != 11
        or any(type(value) is not int for value in expected_identity)
    ):
        raise PackageControllerError("held directory coordinate differs")
    named = os.lstat(path)
    if (
        not stat.S_ISDIR(named.st_mode)
        or stat.S_IMODE(named.st_mode) != expected_mode
        or named.st_uid != expected_uid or named.st_gid != expected_gid
        or identity(named) != tuple(expected_identity)
        or path.resolve(strict=True) != path
    ):
        raise PackageControllerError(f"named directory differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        named_after = os.lstat(path)
        if identity(opened) != identity(named) or identity(named_after) != identity(named):
            raise PackageControllerError(f"directory replay differs: {path}")
        return HeldDirectory(path, descriptor, identity(opened))
    except BaseException:
        os.close(descriptor)
        raise


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
        "snapshot_manifest_size": SNAPSHOT_MANIFEST_SIZE,
        "snapshot_manifest_digest": SNAPSHOT_MANIFEST_DIGEST,
        "snapshot_receipt_sha256": SNAPSHOT_RECEIPT_SHA256,
        "snapshot_receipt_size": SNAPSHOT_RECEIPT_SIZE,
        "snapshot_receipt_digest": SNAPSHOT_RECEIPT_DIGEST,
        "snapshot_root_identity": SNAPSHOT_ROOT_IDENTITY,
        "overlay_receipt_sha256": OVERLAY_RECEIPT_SHA256,
        "overlay_receipt_size": OVERLAY_RECEIPT_SIZE,
        "overlay_receipt_digest": OVERLAY_RECEIPT_DIGEST,
        "overlay_root_identity": OVERLAY_ROOT_IDENTITY,
        "overlay_materializer_sha256": MATERIALIZER_SHA256,
        "overlay_materializer_size": MATERIALIZER_SIZE,
    }


def blocked_dynamic_pins() -> tuple[str, ...]:
    blocked: list[str] = []
    for name, value in dynamic_pin_values().items():
        if name.endswith("_size"):
            if type(value) is not int or value <= 0:
                blocked.append(name)
        elif name in {"snapshot_root_identity", "overlay_root_identity"}:
            if (
                type(value) is not list or len(value) != 11
                or any(type(item) is not int for item in value)
            ):
                blocked.append(name)
        elif type(value) is not str or SHA_RE.fullmatch(value) is None:
            blocked.append(name)
    return tuple(blocked)


def authorization_token() -> str:
    value = {
        "schema_version": SCHEMA + "-authorization-v1",
        "state": READY_STATE,
        "snapshot_root": str(SNAPSHOT_ROOT),
        "snapshot_manifest_path": str(SNAPSHOT_MANIFEST_PATH),
        "snapshot_receipt_path": str(SNAPSHOT_RECEIPT_PATH),
        "overlay_root": str(OVERLAY_ROOT),
        "overlay_receipt_path": str(OVERLAY_RECEIPT_PATH),
        "package_root": str(PACKAGE_ROOT),
        "package_receipt_path": str(PACKAGE_RECEIPT_PATH),
        "attempt_path": str(ATTEMPT_PATH), "evidence_path": str(EVIDENCE_PATH),
        "materializer": {
            "path": str(MATERIALIZER_PATH), "sha256": MATERIALIZER_SHA256,
            "size": MATERIALIZER_SIZE,
        },
        "python": {
            "path": str(VACE_PYTHON), "sha256": VACE_PYTHON_SHA256,
            "size": VACE_PYTHON_SIZE,
        },
        "job_id": JOB_ID, "node": NODE,
        "dynamic_pins": dynamic_pin_values(),
        "single_attempt": True, "retry_allowed": False,
        "launch_allowed": False,
    }
    return object_digest(value)


SNAPSHOT_MANIFEST_FIELDS = {
    "schema_version", "status", "launch_allowed", "old_snapshot_root",
    "staging_root", "staging_receipt_path",
    "snapshot_publication_receipt_path", "target_root", "content_leaf_count",
    "physical_file_count_including_manifest", "release_file_count",
    "legacy_alias_is_distinct_regular_inode", "builder_authority",
    "staging_receipt_authority", "publication_protocol", "rename_noreplace",
    "cooperative_writer_exclusion", "target_absent_rechecked",
    "whole_tree_atomically_visible", "uncooperative_same_uid_race_out_of_scope",
    "retry_allowed", "formal_review_test", "files", "manifest_digest",
}
SNAPSHOT_RECEIPT_FIELDS = {
    "schema_version", "status", "target_root", "receipt_path",
    "manifest_path", "manifest_sha256", "manifest_digest",
    "staging_receipt_sha256", "staging_receipt_digest", "content_leaf_count",
    "physical_file_count_including_manifest", "publication_protocol",
    "rename_noreplace", "cooperative_writer_exclusion",
    "target_absent_rechecked_before_rename", "ordinary_posix_rename_performed",
    "publication_observation", "whole_tree_atomically_visible",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
    "receipt_is_admission", "launch_allowed", "receipt_inode_anchor",
    "receipt_digest",
}


def validate_snapshot_receipt(held: HeldAuthority) -> dict[str, Any]:
    value = strict_json(held.raw, label="exact35 publication receipt")
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    if (
        set(value) != SNAPSHOT_RECEIPT_FIELDS
        or hashlib.sha256(held.raw).hexdigest() != SNAPSHOT_RECEIPT_SHA256
        or len(held.raw) != SNAPSHOT_RECEIPT_SIZE
        or claimed != SNAPSHOT_RECEIPT_DIGEST
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != SNAPSHOT_RECEIPT_SCHEMA
        or value.get("status") != "PUBLISHED_RECEIPT_GATED"
        or value.get("target_root") != str(SNAPSHOT_ROOT)
        or value.get("receipt_path") != str(SNAPSHOT_RECEIPT_PATH)
        or value.get("manifest_path") != str(SNAPSHOT_MANIFEST_PATH)
        or value.get("manifest_sha256") != SNAPSHOT_MANIFEST_SHA256
        or value.get("manifest_digest") != SNAPSHOT_MANIFEST_DIGEST
        or value.get("content_leaf_count") != 34
        or value.get("physical_file_count_including_manifest") != 35
        or value.get("publication_protocol") != PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("target_absent_rechecked_before_rename") is not True
        or value.get("ordinary_posix_rename_performed") is not True
        or value.get("publication_observation") != {
            "namespace_state": "target_same_inode_source_absent",
            "rename_returned_zero": True, "rename_error_errno": None,
            "parent_fsync_returned_zero": True, "parent_fsync_error_errno": None,
        }
        or value.get("whole_tree_atomically_visible") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("retry_allowed") is not False
        or value.get("target_root_identity") != SNAPSHOT_ROOT_IDENTITY
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("launch_allowed") is not False
        or value.get("receipt_inode_anchor")
        != inode_anchor(os.fstat(held.descriptor))
    ):
        raise PackageControllerError("exact35 publication receipt differs")
    return value


def validate_snapshot_manifest(
    held: HeldAuthority, receipt: Mapping[str, Any], root: HeldDirectory,
) -> dict[str, Any]:
    value = strict_json(held.raw, label="exact35 manifest")
    unsigned = dict(value); claimed = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    paths = (
        [row.get("path") for row in rows]
        if type(rows) is list and all(type(row) is dict for row in rows)
        else []
    )
    if (
        set(value) != SNAPSHOT_MANIFEST_FIELDS
        or hashlib.sha256(held.raw).hexdigest() != SNAPSHOT_MANIFEST_SHA256
        or len(held.raw) != SNAPSHOT_MANIFEST_SIZE
        or claimed != SNAPSHOT_MANIFEST_DIGEST
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA
        or value.get("status") != "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE"
        or value.get("launch_allowed") is not False
        or value.get("target_root") != str(SNAPSHOT_ROOT)
        or value.get("snapshot_publication_receipt_path")
        != str(SNAPSHOT_RECEIPT_PATH)
        or value.get("content_leaf_count") != 34
        or value.get("physical_file_count_including_manifest") != 35
        or value.get("release_file_count") != 25
        or value.get("publication_protocol") != PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("target_absent_rechecked") is not True
        or value.get("whole_tree_atomically_visible") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("retry_allowed") is not False
        or type(rows) is not list or len(rows) != 34
        or not all(type(path) is str for path in paths)
        or paths != sorted(paths) or len(set(paths)) != 34
        or receipt.get("manifest_sha256") != hashlib.sha256(held.raw).hexdigest()
        or receipt.get("manifest_digest") != claimed
        or receipt.get("target_root_identity") != list(root.held_identity)
    ):
        raise PackageControllerError("exact35 manifest differs")
    return value


OVERLAY_RECEIPT_FIELDS = {
    "schema_version", "status", "launch_allowed", "target_root",
    "receipt_path", "source_file_count", "file_mode", "directory_mode",
    "receipt_mode", "files", "directories", "root_identity",
    "materializer_relative", "publication_protocol", "precreated_by_scp",
    "exact_tree", "held_source_descriptors_through_receipt_reservation",
    "receipt_is_consumption_gate", "retry_allowed", "receipt_inode_anchor",
    "receipt_digest",
}
OVERLAY_RELATIVES = (
    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v4.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v4.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v4.py",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v3.py",
    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_auh_r5f_v4.py",
    MATERIALIZER_RELATIVE,
)


def validate_overlay_receipt(
    receipt: HeldAuthority, root: HeldDirectory,
) -> tuple[dict[str, Any], list[HeldAuthority]]:
    value = strict_json(receipt.raw, label="overlay publication receipt")
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    rows = value.get("files"); directory_rows = value.get("directories")
    if (
        set(value) != OVERLAY_RECEIPT_FIELDS
        or value.get("schema_version") != OVERLAY_RECEIPT_SCHEMA
        or value.get("status") != "PUBLISHED_RECEIPT_GATED"
        or value.get("launch_allowed") is not False
        or value.get("target_root") != str(OVERLAY_ROOT)
        or value.get("receipt_path") != str(OVERLAY_RECEIPT_PATH)
        or value.get("source_file_count") != 6
        or value.get("file_mode") != FILE_MODE
        or value.get("directory_mode") != DIRECTORY_MODE
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("materializer_relative") != MATERIALIZER_RELATIVE
        or value.get("publication_protocol") != OVERLAY_PUBLICATION_PROTOCOL
        or value.get("precreated_by_scp") is not True
        or value.get("exact_tree") is not True
        or value.get("held_source_descriptors_through_receipt_reservation") is not True
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("retry_allowed") is not False
        or claimed != object_digest(unsigned)
        or claimed != OVERLAY_RECEIPT_DIGEST
        or value.get("root_identity") != OVERLAY_ROOT_IDENTITY
        or value.get("root_identity") != list(root.held_identity)
        or value.get("receipt_inode_anchor")
        != inode_anchor(os.fstat(receipt.descriptor))
        or type(rows) is not list or len(rows) != 6
        or [row.get("path") if type(row) is dict else None for row in rows]
        != sorted(OVERLAY_RELATIVES)
        or type(directory_rows) is not list
    ):
        raise PackageControllerError("overlay publication receipt differs")
    expected_directories = {"."}
    for relative in OVERLAY_RELATIVES:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent)); parent = parent.parent
    if [row.get("path") if type(row) is dict else None for row in directory_rows] != sorted(expected_directories):
        raise PackageControllerError("overlay directory receipt ordering differs")
    actual_files: set[str] = set(); actual_directories = {"."}
    pending = [(OVERLAY_ROOT, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != DIRECTORY_MODE
            or info.st_uid != REMOTE_UID or info.st_gid != REMOTE_GID
        ):
            raise PackageControllerError(f"overlay directory differs: {prefix}")
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = entry.name if prefix == "." else f"{prefix}/{entry.name}"
                child = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(child.st_mode):
                    actual_directories.add(relative); pending.append((Path(entry.path), relative))
                elif stat.S_ISREG(child.st_mode) and child.st_nlink == 1:
                    actual_files.add(relative)
                else:
                    raise PackageControllerError(f"overlay special/link differs: {relative}")
    if actual_files != set(OVERLAY_RELATIVES) or actual_directories != expected_directories:
        raise PackageControllerError("overlay exact6 physical tree differs")
    for row in directory_rows:
        relative = row["path"]
        path = OVERLAY_ROOT if relative == "." else OVERLAY_ROOT / relative
        if set(row) != {"path", "identity"} or row.get("identity") != list(identity(os.lstat(path))):
            raise PackageControllerError(f"overlay directory row differs: {relative}")
    held: list[HeldAuthority] = []
    try:
        for row in rows:
            relative = row["path"]
            if (
                set(row) != {"path", "sha256", "size", "mode", "identity", "provenance"}
                or SHA_RE.fullmatch(str(row.get("sha256"))) is None
                or type(row.get("size")) is not int or row["size"] <= 0
                or row.get("mode") != FILE_MODE
                or row.get("provenance") != "fresh_scp_then_held_exact6_seal"
            ):
                raise PackageControllerError(f"overlay source row differs: {relative}")
            authority = open_authority(
                OVERLAY_ROOT / relative, expected_sha256=row["sha256"],
                expected_size=row["size"], expected_mode=FILE_MODE,
                expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
                maximum_size=MAX_SOURCE_SIZE,
            )
            if row.get("identity") != list(authority.held_identity):
                authority.close()
                raise PackageControllerError(f"overlay source identity differs: {relative}")
            if relative == MATERIALIZER_RELATIVE and (
                row["sha256"] != MATERIALIZER_SHA256
                or row["size"] != MATERIALIZER_SIZE
            ):
                authority.close()
                raise PackageControllerError("overlay materializer row differs")
            held.append(authority)
        return value, held
    except BaseException:
        for authority in reversed(held):
            authority.close()
        raise


def load_materializer(raw: bytes) -> types.ModuleType:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise PackageControllerError("materializer is not UTF-8") from error
    module = types.ModuleType("_held_case01_r64_materializer")
    module.__file__ = str(MATERIALIZER_PATH)
    module.__package__ = None
    exec(
        compile(source, str(MATERIALIZER_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    if (
        getattr(module, "SOURCE_SNAPSHOT_ROOT", None) != SNAPSHOT_ROOT
        or getattr(module, "SNAPSHOT_PUBLICATION_RECEIPT_PATH", None)
        != SNAPSHOT_RECEIPT_PATH
        or getattr(module, "OVERLAY_ROOT", None) != OVERLAY_ROOT
        or getattr(module, "OVERLAY_RECEIPT_PATH", None) != OVERLAY_RECEIPT_PATH
        or getattr(module, "TARGET_ROOT", None) != PACKAGE_ROOT
        or getattr(module, "PACKAGE_PUBLICATION_RECEIPT_PATH", None)
        != PACKAGE_RECEIPT_PATH
        or getattr(module, "OVERLAY_MATERIALIZER_RELATIVE", None)
        != MATERIALIZER_RELATIVE
        or getattr(module, "BASE_MATERIALIZER_SHA256", None)
        != BASE_MATERIALIZER_SHA256
        or getattr(module, "RANK_CACHE_ROOT", None) != RANK_CACHE_ROOT
        or not callable(getattr(module, "preflight_snapshot", None))
        or not callable(getattr(module, "preflight_overlay", None))
        or not callable(getattr(module, "materialize", None))
    ):
        raise PackageControllerError("pinned materializer configuration differs")
    return module


def open_self_authority() -> HeldAuthority:
    return open_authority(
        Path(__file__), expected_sha256=None, expected_size=None,
        expected_mode=CONTROLLER_MODE, expected_uid=REMOTE_UID,
        expected_gid=REMOTE_GID, maximum_size=MAX_SOURCE_SIZE,
    )


def open_runtime_authority() -> HeldAuthority:
    held = open_authority(
        VACE_PYTHON, expected_sha256=VACE_PYTHON_SHA256,
        expected_size=VACE_PYTHON_SIZE, expected_mode=0o755,
        expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
        executable=True, maximum_size=64 * 1024 * 1024,
    )
    try:
        if identity(os.stat("/proc/self/exe")) != held.held_identity:
            raise PackageControllerError("executing Python differs from held runtime")
        return held
    except BaseException:
        held.close(); raise


def require_fresh_outputs() -> None:
    for path in (
        PACKAGE_ROOT, PACKAGE_RECEIPT_PATH, ATTEMPT_PATH, EVIDENCE_PATH,
        RANK_CACHE_ROOT,
    ):
        if os.path.lexists(path):
            raise PackageControllerError(f"single-attempt target is not fresh: {path}")


def create_immutable_json(
    path: Path, value: Mapping[str, Any],
) -> tuple[bytes, list[int]]:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.parent != EXPERIMENTS or path.name in {"", ".", ".."}
    ):
        raise PackageControllerError("controller JSON target path differs")
    parent_info = os.lstat(EXPERIMENTS)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or EXPERIMENTS.resolve(strict=True) != EXPERIMENTS
        or parent_info.st_uid != REMOTE_UID or parent_info.st_gid != REMOTE_GID
        or stat.S_IMODE(parent_info.st_mode) & 0o002
    ):
        raise PackageControllerError("controller JSON parent differs")
    parent_fd = os.open(
        EXPERIMENTS,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    raw = canonical(value) + b"\n"
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600, dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise PackageControllerError("controller JSON write made no progress")
            offset += count
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            identity(before) != identity(named) or before.st_nlink != 1
            or before.st_uid != REMOTE_UID or before.st_gid != REMOTE_GID
            or stat.S_IMODE(before.st_mode) != 0o600
            or read_fd(descriptor, before.st_size) != raw
        ):
            raise PackageControllerError("controller JSON staging differs")
        # Successful 0400 application is immutable.  No later path demotes,
        # truncates, unlinks, or retries this inode after any terminal error.
        os.fchmod(descriptor, RECEIPT_MODE)
        os.fsync(descriptor); os.fsync(parent_fd)
        after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            identity(after) != identity(named_after)
            or stat.S_IMODE(after.st_mode) != RECEIPT_MODE
            or read_fd(descriptor, after.st_size) != raw
        ):
            raise PackageControllerError("controller JSON seal differs")
        return raw, inode_anchor(after)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def replay_all(
    authorities: Sequence[HeldAuthority], directories: Sequence[HeldDirectory],
) -> None:
    for authority in authorities:
        authority.replay()
    for directory in directories:
        directory.replay()


def _process_group_present(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise PackageControllerError(
            "materializer process-group probe differs"
        ) from error


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return
    except PermissionError:
        # EPERM means the group may still exist.  The bounded ESRCH poll is
        # authoritative; a permission failure is never interpreted as zero.
        return
    except OSError as error:
        if error.errno in (errno.ESRCH, errno.EPERM):
            return
        raise PackageControllerError(
            "materializer process-group signal differs"
        ) from error


def _poll_group_absent(
    process: subprocess.Popen[bytes], process_group: int, deadline: float,
) -> bool:
    while True:
        # Reap a terminal direct child while continuing to reason about the
        # saved PGID.  Descendants may outlive their leader and retain pipes.
        process.poll()
        if not _process_group_present(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_POLL_SECONDS)


def _process_group_absent(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not _process_group_present(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_POLL_SECONDS)


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    errors: list[BaseException] = []
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise PackageControllerError(
            "materializer terminal pipe close differs"
        ) from errors[0]


def _seal_process_group(
    process: subprocess.Popen[bytes], process_group: int,
) -> None:
    # Cleanup is keyed by the PGID saved immediately after Popen.  It never
    # returns merely because the leader has already exited: a descendant can
    # still hold stdout/stderr or ignore SIGTERM in the same session.
    pipe_error: BaseException | None = None
    try:
        _close_process_pipes(process)
    except BaseException as error:
        pipe_error = error

    _signal_process_group(process_group, signal.SIGTERM)
    term_deadline = time.monotonic() + PROCESS_TERM_GRACE_SECONDS
    _poll_group_absent(process, process_group, term_deadline)
    if _process_group_present(process_group):
        _signal_process_group(process_group, signal.SIGKILL)

    try:
        process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # The process object is still the owned direct child.  This fallback
        # does not replace the independent saved-PGID zero gate below.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise PackageControllerError(
                "materializer direct child did not reap"
            ) from error

    kill_deadline = time.monotonic() + PROCESS_KILL_GRACE_SECONDS
    if not _poll_group_absent(process, process_group, kill_deadline):
        raise PackageControllerError(
            "materializer process group did not reach ESRCH"
        )
    if process.poll() is None:
        raise PackageControllerError("materializer direct child remains unreaped")
    if pipe_error is not None:
        raise PackageControllerError(
            "materializer terminal pipe seal differs"
        ) from pipe_error


def materializer_argv(runtime_fd: int, materializer_fd: int) -> list[str]:
    return [
        f"/proc/self/fd/{runtime_fd}", "-I", "-S", "-B",
        f"/proc/self/fd/{materializer_fd}",
        "materialize",
        "--root", str(PACKAGE_ROOT), "--source-root", str(SNAPSHOT_ROOT),
        "--overlay-root", str(OVERLAY_ROOT),
        "--overlay-receipt", str(OVERLAY_RECEIPT_PATH),
        "--job-id", JOB_ID, "--node", NODE,
        "--snapshot-manifest-sha256", str(SNAPSHOT_MANIFEST_SHA256),
        "--snapshot-materializer-sha256", BASE_MATERIALIZER_SHA256,
        "--overlay-materializer-sha256", MATERIALIZER_SHA256,
        "--overlay-receipt-sha256", OVERLAY_RECEIPT_SHA256,
        "--overlay-receipt-size", str(OVERLAY_RECEIPT_SIZE),
        "--overlay-receipt-digest", OVERLAY_RECEIPT_DIGEST,
        "--overlay-root-identity-json", canonical(OVERLAY_ROOT_IDENTITY).decode("utf-8"),
    ]


def run_one_materializer(
    runtime: HeldAuthority, materializer: HeldAuthority,
) -> tuple[bytes, bytes, int, list[str]]:
    argv = materializer_argv(runtime.descriptor, materializer.descriptor)
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/vast/users/guangyi.chen",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        argv, executable=argv[0], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, close_fds=True,
        pass_fds=(runtime.descriptor, materializer.descriptor),
        start_new_session=True,
    )
    # start_new_session=True establishes PGID==PID before exec.  Preserve the
    # deterministic group identity before communicate or any other operation
    # can fail, including an early leader exit with live descendants.
    process_group = process.pid
    stdout = b""
    stderr = b""
    primary_error: BaseException | None = None
    try:
        stdout, stderr = process.communicate(timeout=MATERIALIZER_TIMEOUT_SECONDS)
    except BaseException as error:
        primary_error = error

    group_present_after_communicate = False
    try:
        group_present_after_communicate = _process_group_present(process_group)
    except BaseException as error:
        if primary_error is None:
            primary_error = error
    try:
        _seal_process_group(process, process_group)
    except BaseException as cleanup_error:
        raise PackageControllerError(
            "materializer process/pipe zero gate differs"
        ) from (primary_error if primary_error is not None else cleanup_error)

    if isinstance(primary_error, subprocess.TimeoutExpired):
        raise PackageControllerError("single materializer attempt timed out") from primary_error
    if primary_error is not None:
        raise primary_error
    if process.returncode is None:
        raise PackageControllerError("materializer lacks terminal return code")
    if group_present_after_communicate:
        raise PackageControllerError(
            "terminal materializer process group required cleanup"
        )
    return stdout, stderr, int(process.returncode), argv


PACKAGE_RECEIPT_FIELDS = {
    "schema_version", "status", "target_root", "receipt_path",
    "materialization_receipt_path", "materialization_receipt_sha256",
    "materialization_receipt_digest", "source_snapshot_manifest_sha256",
    "source_snapshot_manifest_digest", "source_staging_receipt_sha256",
    "source_staging_receipt_digest", "source_overlay_receipt_sha256",
    "source_overlay_receipt_digest", "source_overlay_root_identity",
    "publication_protocol",
    "rename_noreplace", "cooperative_writer_exclusion",
    "target_absent_rechecked_before_rename", "ordinary_posix_rename_performed",
    "publication_observation", "whole_tree_atomically_visible",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
    "receipt_is_admission", "launch_allowed", "receipt_inode_anchor",
    "receipt_digest",
}
REPORT_FIELDS = {
    "schema_version", "status", "launch_allowed", "root",
    "source_snapshot_root", "source_snapshot",
    "source_overlay_root", "source_overlay", "source_provenance",
    "source_staging_receipt_authority", "package_publication_receipt_path",
    "publication_protocol", "rename_noreplace", "cooperative_writer_exclusion",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "release_file_count", "production_identity_count",
    "release", "production",
    "condition_and_admission_authority_count", "plan", "launch", "admission",
    "slurm_step_launched", "gpu_attempt_claimed", "artifacts", "receipt_digest",
}
LAUNCH_FIELDS = {
    "schema_version", "status", "launch_allowed", "slurm_step_launched",
    "gpu_attempt_claimed", "input", "release", "payload_path",
    "payload_sha256", "payload_size", "receipt_digest",
}
LAUNCH_RELEASE_FIELDS = {
    "schema_version", "status", "launch_allowed", "campaign_mode",
    "selected_task_ids", "identity_roles", "identities", "input_sha256",
    "ready_overlay_required", "named_payload_execution_forbidden",
    "release_digest",
}


def _expected_directories(files: set[str]) -> set[str]:
    result = {"."}
    for relative in files:
        parent = Path(relative).parent
        while str(parent) != ".":
            result.add(str(parent)); parent = parent.parent
    return result


def validate_package_tree(
    expected: Mapping[str, tuple[str, int, int]],
) -> dict[str, Any]:
    actual_files: set[str] = set()
    actual_directories = {"."}
    pending = [(PACKAGE_ROOT, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o022
            or info.st_uid != REMOTE_UID or info.st_gid != REMOTE_GID
        ):
            raise PackageControllerError(f"package directory differs: {prefix}")
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = entry.name if prefix == "." else f"{prefix}/{entry.name}"
                child = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(child.st_mode):
                    actual_directories.add(relative)
                    pending.append((Path(entry.path), relative))
                elif stat.S_ISREG(child.st_mode) and child.st_nlink == 1:
                    actual_files.add(relative)
                else:
                    raise PackageControllerError(
                        f"package special/link entry differs: {relative}"
                    )
    extra = {"evidence", "outputs", "outputs/media_v3", "final", "logs", "runtime"}
    if (
        actual_files != set(expected) or len(actual_files) != 39
        or actual_directories != _expected_directories(set(expected)) | extra
    ):
        raise PackageControllerError("package exact39 closure differs")
    for relative, (digest, size, mode) in expected.items():
        held = open_authority(
            PACKAGE_ROOT / relative, expected_sha256=digest,
            expected_size=size, expected_mode=mode,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_PACKAGE_FILE_SIZE,
        )
        held.close()
    return {"file_count": 39, "directory_count": len(actual_directories)}


def validate_package(
    child_report: Mapping[str, Any], snapshot_bytes: Mapping[str, bytes],
    snapshot_evidence: Mapping[str, Any], overlay_bytes: Mapping[str, bytes],
    overlay_evidence: Mapping[str, Any], materializer_module: types.ModuleType,
) -> dict[str, Any]:
    # Receipt-first is literal here too: no package target/root/internal leaf is
    # named before the final sibling 0400 publication receipt is held.
    publication_authority = open_authority(
        PACKAGE_RECEIPT_PATH, expected_sha256=None, expected_size=None,
        expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
        expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
    )
    package_root: HeldDirectory | None = None
    internal_authority: HeldAuthority | None = None
    plan_authority: HeldAuthority | None = None
    try:
        publication = strict_json(
            publication_authority.raw, label="package publication receipt",
        )
        unsigned_publication = dict(publication)
        publication_digest = unsigned_publication.pop("receipt_digest", None)
        if (
            set(publication) != PACKAGE_RECEIPT_FIELDS
            or publication_digest != object_digest(unsigned_publication)
            or publication.get("schema_version") != PACKAGE_RECEIPT_SCHEMA
            or publication.get("status") != "PUBLISHED_RECEIPT_GATED"
            or publication.get("target_root") != str(PACKAGE_ROOT)
            or publication.get("receipt_path") != str(PACKAGE_RECEIPT_PATH)
            or publication.get("materialization_receipt_path")
            != str(INTERNAL_RECEIPT_PATH)
            or publication.get("source_snapshot_manifest_sha256")
            != SNAPSHOT_MANIFEST_SHA256
            or publication.get("source_snapshot_manifest_digest")
            != SNAPSHOT_MANIFEST_DIGEST
            or publication.get("source_overlay_receipt_sha256")
            != OVERLAY_RECEIPT_SHA256
            or publication.get("source_overlay_receipt_digest")
            != OVERLAY_RECEIPT_DIGEST
            or publication.get("source_overlay_root_identity")
            != OVERLAY_ROOT_IDENTITY
            or publication.get("publication_protocol") != PUBLICATION_PROTOCOL
            or publication.get("rename_noreplace") is not False
            or publication.get("cooperative_writer_exclusion") is not True
            or publication.get("target_absent_rechecked_before_rename") is not True
            or publication.get("ordinary_posix_rename_performed") is not True
            or publication.get("publication_observation") != {
                "namespace_state": "target_same_inode_source_absent",
                "rename_returned_zero": True, "rename_error_errno": None,
                "parent_fsync_returned_zero": True,
                "parent_fsync_error_errno": None,
            }
            or publication.get("whole_tree_atomically_visible") is not True
            or publication.get("uncooperative_same_uid_race_out_of_scope") is not True
            or publication.get("retry_allowed") is not False
            or publication.get("receipt_mode") != RECEIPT_MODE
            or publication.get("receipt_is_consumption_gate") is not True
            or publication.get("receipt_is_admission") is not True
            or publication.get("launch_allowed") is not False
            or publication.get("receipt_inode_anchor")
            != inode_anchor(os.fstat(publication_authority.descriptor))
        ):
            raise PackageControllerError("package publication receipt differs")
        package_root = open_directory(
            PACKAGE_ROOT, expected_identity=publication.get("target_root_identity"),
            expected_mode=0o700, expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
        )
        internal_authority = open_authority(
            INTERNAL_RECEIPT_PATH,
            expected_sha256=publication.get("materialization_receipt_sha256"),
            expected_size=None, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        report = strict_json(
            internal_authority.raw, label="internal materialization receipt",
        )
        unsigned_report = dict(report); report_digest = unsigned_report.pop(
            "receipt_digest", None,
        )
        if (
            set(report) != REPORT_FIELDS or report != child_report
            or report_digest != object_digest(unsigned_report)
            or report_digest != publication.get("materialization_receipt_digest")
            or hashlib.sha256(internal_authority.raw).hexdigest()
            != publication.get("materialization_receipt_sha256")
            or report.get("schema_version") != MATERIALIZATION_SCHEMA
            or report.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
            or report.get("launch_allowed") is not False
            or report.get("root") != str(PACKAGE_ROOT)
            or report.get("source_snapshot_root") != str(SNAPSHOT_ROOT)
            or report.get("source_snapshot") != snapshot_evidence
            or report.get("source_overlay_root") != str(OVERLAY_ROOT)
            or report.get("source_overlay") != overlay_evidence
            or report.get("source_provenance") != {
                "base_release_leaf_count": 20,
                "overlay_release_leaf_count": 5,
                "overlay_nonrelease_materializer_leaf_count": 1,
                "burned_canary_v1_consumed": False,
                "failed_canary_v2_consumed": False,
            }
            or report.get("source_staging_receipt_authority")
            != snapshot_evidence.get("staging_receipt_authority")
            or report.get("package_publication_receipt_path")
            != str(PACKAGE_RECEIPT_PATH)
            or report.get("publication_protocol") != PUBLICATION_PROTOCOL
            or report.get("rename_noreplace") is not False
            or report.get("cooperative_writer_exclusion") is not True
            or report.get("uncooperative_same_uid_race_out_of_scope") is not True
            or report.get("retry_allowed") is not False
            or report.get("release_file_count") != 25
            or report.get("production_identity_count") != 26
            or report.get("condition_and_admission_authority_count") != 6
            or report.get("admission") != {
                "static_executed": False, "root_fake_executed": False,
                "world4_executed": False,
            }
            or report.get("slurm_step_launched") is not False
            or report.get("gpu_attempt_claimed") is not False
        ):
            raise PackageControllerError("internal materialization receipt differs")
        staging = snapshot_evidence.get("staging_receipt_authority")
        if (
            type(staging) is not dict
            or publication.get("source_staging_receipt_sha256")
            != staging.get("sha256")
            or publication.get("source_staging_receipt_digest")
            != staging.get("receipt_digest")
        ):
            raise PackageControllerError("package staging authority differs")

        release_block = report.get("release")
        production = report.get("production")
        release_rows = release_block.get("files") if type(release_block) is dict else None
        expected_release_rows: list[dict[str, Any]] = []
        for relative, digest in sorted(materializer_module.RELEASE_FILES.items()):
            raw = (
                overlay_bytes[relative]
                if relative in materializer_module.OVERLAY_RELEASE_FILES
                else snapshot_bytes[relative]
            )
            expected_release_rows.append({
                "path": "release/" + relative, "sha256": digest,
                "size": len(raw),
                "provenance": (
                    "receipt_gated_exact6_overlay"
                    if relative in materializer_module.OVERLAY_RELEASE_FILES
                    else "receipt_gated_exact35_snapshot"
                ),
            })
        identities = production.get("identities") if type(production) is dict else None
        crosslink = (
            production.get("inner_outer_crosslink")
            if type(production) is dict else None
        )
        if (
            type(release_block) is not dict
            or set(release_block) != {"files", "manifest_digest"}
            or release_rows != expected_release_rows
            or release_block.get("manifest_digest") != object_digest(expected_release_rows)
            or type(production) is not dict
            or set(production) != {
                "identity_roles", "identities", "identity_set_digest",
                "inner_outer_crosslink",
            }
            or production.get("identity_roles") != list(PRODUCTION_IDENTITY_ROLES)
            or type(identities) is not dict
            or set(identities) != set(PRODUCTION_IDENTITY_ROLES)
            or len(identities) != 26
            or production.get("identity_set_digest") != object_digest(identities)
            or type(crosslink) is not dict
            or set(crosslink) != {
                "adapter", "object_wrapper_inner", "producer_adapter",
                "producer_object_wrapper_inner", "distinct_paths",
                "outer_calls_pinned_inner_contract",
            }
            or crosslink.get("adapter") != identities.get("adapter")
            or crosslink.get("object_wrapper_inner")
            != identities.get("object_wrapper_inner")
            or crosslink.get("producer_adapter") != identities.get("adapter")
            or crosslink.get("producer_object_wrapper_inner")
            != identities.get("object_wrapper_inner")
            or crosslink.get("distinct_paths") is not True
            or crosslink.get("outer_calls_pinned_inner_contract") is not True
        ):
            raise PackageControllerError("release25/exact26 crosslink differs")

        expected: dict[str, tuple[str, int, int]] = {}
        expected_artifacts: dict[str, dict[str, Any]] = {}
        for relative, digest in materializer_module.RELEASE_FILES.items():
            package_relative = "release/" + relative
            raw = (
                overlay_bytes[relative]
                if relative in materializer_module.OVERLAY_RELEASE_FILES
                else snapshot_bytes[relative]
            )
            expected[package_relative] = (digest, len(raw), FILE_MODE)
            expected_artifacts[package_relative] = {
                "sha256": digest, "size": len(raw),
            }
        for relative, digest in materializer_module.DIAGNOSTIC_FILES.items():
            package_relative = "diagnostics/" + Path(relative).name
            raw = snapshot_bytes[relative]
            expected[package_relative] = (digest, len(raw), FILE_MODE)
            expected_artifacts[package_relative] = {
                "sha256": digest, "size": len(raw),
            }
        if report.get("artifacts") != expected_artifacts:
            raise PackageControllerError("package artifact map differs")

        source_relative, source_digest, source_size = materializer_module.SOURCE_VIDEO
        aux_relative, aux_digest, aux_size = materializer_module.AUX_VIDEO
        expected["authority/conditions/" + Path(source_relative).name] = (
            source_digest, source_size, FILE_MODE,
        )
        expected["authority/conditions/" + Path(aux_relative).name] = (
            aux_digest, aux_size, FILE_MODE,
        )
        authority_names = (
            "stage0_receipt.json", "g0_sparse_annotations.json",
            "trajectory_scaffold.json", "scaffold_independent_audit.json",
        )
        for (relative, digest), name in zip(
            materializer_module.SNAPSHOT_AUTHORITY_FILES.items(), authority_names,
        ):
            expected["authority/conditions/" + name] = (
                digest, len(snapshot_bytes[relative]), FILE_MODE,
            )

        plan = report.get("plan")
        if (
            type(plan) is not dict
            or set(plan) != {"path", "sha256", "plan_digest"}
            or plan.get("path") != str(
                PACKAGE_ROOT / "plan/"
                "case01_object_trajectory_exact5_r64_HOLD_plan_v3.json"
            )
            or SHA_RE.fullmatch(str(plan.get("sha256"))) is None
            or SHA_RE.fullmatch(str(plan.get("plan_digest"))) is None
        ):
            raise PackageControllerError("package plan row differs")
        plan_authority = open_authority(
            Path(plan["path"]), expected_sha256=plan["sha256"],
            expected_size=None, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        plan_value = strict_json(plan_authority.raw, label="package HOLD plan")
        unsigned_plan = dict(plan_value); plan_digest = unsigned_plan.pop(
            "plan_digest", None,
        )
        producer = plan_value.get("producer")
        if (
            plan_digest != plan["plan_digest"]
            or plan_digest != object_digest(unsigned_plan)
            or identities.get("plan") != {
                "path": plan["path"], "sha256": plan["sha256"],
                "size": len(plan_authority.raw),
            }
            or type(producer) is not dict
            or identities.get("adapter") != {
                "path": producer.get("inference_wrapper_path"),
                "sha256": producer.get("inference_wrapper_sha256"),
                "size": producer.get("inference_wrapper_size"),
            }
            or identities.get("object_wrapper_inner") != {
                "path": producer.get("object_wrapper_inner_path"),
                "sha256": producer.get("object_wrapper_inner_sha256"),
                "size": producer.get("object_wrapper_inner_size"),
            }
        ):
            raise PackageControllerError("package HOLD plan digest differs")
        expected[str(Path(plan["path"]).relative_to(PACKAGE_ROOT))] = (
            plan["sha256"], len(plan_authority.raw), FILE_MODE,
        )

        launch = report.get("launch")
        if (
            type(launch) is not dict or set(launch) != LAUNCH_FIELDS
            or launch.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
            or launch.get("launch_allowed") is not False
            or launch.get("slurm_step_launched") is not False
            or launch.get("gpu_attempt_claimed") is not False
            or launch.get("payload_path")
            != str(PACKAGE_ROOT / "launch/root_launch_payload_HOLD_v3.sh")
            or SHA_RE.fullmatch(str(launch.get("payload_sha256"))) is None
            or type(launch.get("payload_size")) is not int
            or launch["payload_size"] <= 0
        ):
            raise PackageControllerError("package launch receipt differs")
        unsigned_launch = dict(launch); launch_digest = unsigned_launch.pop(
            "receipt_digest", None,
        )
        input_row = launch.get("input")
        release = launch.get("release")
        unsigned_release = dict(release) if type(release) is dict else {}
        release_digest = unsigned_release.pop("release_digest", None)
        if (
            launch.get("schema_version")
            != "case01-object-trajectory-exact5-hold-launch-receipt-auh-v3"
            or launch_digest != object_digest(unsigned_launch)
            or type(input_row) is not dict
            or set(input_row) != {"path", "sha256", "size", "mode", "nlink"}
            or input_row.get("path")
            != str(PACKAGE_ROOT / "launch/root_launch_input_HOLD_v3.json")
            or SHA_RE.fullmatch(str(input_row.get("sha256"))) is None
            or type(input_row.get("size")) is not int or input_row["size"] <= 0
            or input_row.get("mode") != FILE_MODE or input_row.get("nlink") != 1
            or type(release) is not dict or set(release) != LAUNCH_RELEASE_FIELDS
            or release.get("schema_version")
            != "case01-object-trajectory-exact5-hold-launch-release-auh-v3"
            or release.get("status") != "HOLD_NOT_LAUNCHABLE"
            or release.get("launch_allowed") is not False
            or release.get("identity_roles") != list(PRODUCTION_IDENTITY_ROLES)
            or release.get("identities") != identities
            or release.get("input_sha256") != input_row.get("sha256")
            or release.get("ready_overlay_required") is not True
            or release.get("named_payload_execution_forbidden") is not True
            or release_digest != object_digest(unsigned_release)
        ):
            raise PackageControllerError("package launch input differs")
        expected["launch/root_launch_input_HOLD_v3.json"] = (
            input_row["sha256"], input_row["size"], FILE_MODE,
        )
        expected["launch/root_launch_payload_HOLD_v3.sh"] = (
            launch["payload_sha256"], launch["payload_size"], FILE_MODE,
        )
        launch_raw = canonical(launch) + b"\n"
        expected["launch/root_launch_receipt_HOLD_v3.json"] = (
            hashlib.sha256(launch_raw).hexdigest(), len(launch_raw), RECEIPT_MODE,
        )
        expected[INTERNAL_RECEIPT_RELATIVE] = (
            hashlib.sha256(internal_authority.raw).hexdigest(),
            len(internal_authority.raw), RECEIPT_MODE,
        )
        tree = validate_package_tree(expected)
        publication_authority.replay(); package_root.replay()
        internal_authority.replay(); plan_authority.replay()
        return {
            "publication_receipt": publication_authority.row(),
            "publication_receipt_digest": publication_digest,
            "materialization_receipt": internal_authority.row(),
            "materialization_receipt_digest": report_digest,
            "package_root": package_root.row(),
            "release": {
                "file_count": 25,
                "manifest_digest": release_block["manifest_digest"],
            },
            "production": {
                "identity_count": 26,
                "identity_roles": list(PRODUCTION_IDENTITY_ROLES),
                "identity_set_digest": production["identity_set_digest"],
                "inner_outer_crosslink": crosslink,
            },
            "gpu_attempt_claimed": False,
            "srun_performed": False,
            **tree,
        }
    finally:
        if plan_authority is not None:
            plan_authority.close()
        if internal_authority is not None:
            internal_authority.close()
        if package_root is not None:
            package_root.close()
        publication_authority.close()


def controller() -> dict[str, Any]:
    authorities: list[HeldAuthority] = []
    directories: list[HeldDirectory] = []
    try:
        # No overlay root/source is named until its pinned 0400 sibling receipt
        # is held.  All six source descriptors remain held through the child.
        overlay_receipt = open_authority(
            OVERLAY_RECEIPT_PATH, expected_sha256=OVERLAY_RECEIPT_SHA256,
            expected_size=OVERLAY_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(overlay_receipt)
        overlay_root = open_directory(
            OVERLAY_ROOT, expected_identity=OVERLAY_ROOT_IDENTITY,
            expected_mode=DIRECTORY_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID,
        )
        directories.append(overlay_root)
        overlay_receipt_value, overlay_sources = validate_overlay_receipt(
            overlay_receipt, overlay_root,
        )
        authorities.extend(overlay_sources)
        materializer_authority = next(
            authority for authority in overlay_sources
            if authority.path == MATERIALIZER_PATH
        )
        materializer = load_materializer(materializer_authority.raw)
        # No SNAPSHOT_ROOT or exact35 leaf is named before this final 0400
        # sibling receipt has been opened and pinned.
        snapshot_receipt = open_authority(
            SNAPSHOT_RECEIPT_PATH, expected_sha256=SNAPSHOT_RECEIPT_SHA256,
            expected_size=SNAPSHOT_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(snapshot_receipt)
        snapshot_receipt_value = validate_snapshot_receipt(snapshot_receipt)
        snapshot_root = open_directory(
            SNAPSHOT_ROOT, expected_identity=SNAPSHOT_ROOT_IDENTITY,
            expected_mode=DIRECTORY_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID,
        )
        directories.append(snapshot_root)
        snapshot_manifest = open_authority(
            SNAPSHOT_MANIFEST_PATH, expected_sha256=SNAPSHOT_MANIFEST_SHA256,
            expected_size=SNAPSHOT_MANIFEST_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(snapshot_manifest)
        validate_snapshot_manifest(
            snapshot_manifest, snapshot_receipt_value, snapshot_root,
        )
        try:
            snapshot_bytes, snapshot_evidence = materializer.preflight_snapshot(
                SNAPSHOT_ROOT, manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
                materializer_sha256=BASE_MATERIALIZER_SHA256,
            )
            overlay_bytes, overlay_evidence = materializer.preflight_overlay(
                OVERLAY_ROOT, OVERLAY_RECEIPT_PATH,
                materializer_sha256=MATERIALIZER_SHA256,
                receipt_sha256=OVERLAY_RECEIPT_SHA256,
                receipt_size=OVERLAY_RECEIPT_SIZE,
                receipt_digest=OVERLAY_RECEIPT_DIGEST,
                root_identity=OVERLAY_ROOT_IDENTITY,
            )
        except Exception as error:
            raise PackageControllerError("dual-source materializer preflight refused") from error
        if (
            snapshot_evidence.get("sha256") != SNAPSHOT_MANIFEST_SHA256
            or snapshot_evidence.get("size") != SNAPSHOT_MANIFEST_SIZE
            or snapshot_evidence.get("manifest_digest") != SNAPSHOT_MANIFEST_DIGEST
            or snapshot_evidence.get("snapshot_publication_receipt", {}).get("sha256")
            != SNAPSHOT_RECEIPT_SHA256
            or snapshot_evidence.get("snapshot_publication_receipt", {}).get("size")
            != SNAPSHOT_RECEIPT_SIZE
            or snapshot_evidence.get("snapshot_publication_receipt", {}).get(
                "receipt_digest"
            ) != SNAPSHOT_RECEIPT_DIGEST
            or overlay_bytes.get(MATERIALIZER_RELATIVE) != materializer_authority.raw
            or overlay_evidence.get("receipt", {}).get("sha256")
            != OVERLAY_RECEIPT_SHA256
            or overlay_evidence.get("receipt", {}).get("size")
            != OVERLAY_RECEIPT_SIZE
            or overlay_evidence.get("receipt", {}).get("receipt_digest")
            != OVERLAY_RECEIPT_DIGEST
            or overlay_evidence.get("root_identity") != OVERLAY_ROOT_IDENTITY
            or overlay_receipt_value.get("receipt_digest") != OVERLAY_RECEIPT_DIGEST
        ):
            raise PackageControllerError("materializer dual-source evidence differs")

        runtime = open_runtime_authority(); authorities.append(runtime)
        self_authority = open_self_authority(); authorities.append(self_authority)
        replay_all(authorities, directories)
        require_fresh_outputs()
        argv = materializer_argv(runtime.descriptor, materializer_authority.descriptor)
        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_MATERIALIZER_CHILD",
            "single_attempt": True, "retry_allowed": False,
            "launch_allowed": False,
            "snapshot": {
                "root": snapshot_root.row(),
                "manifest": snapshot_manifest.row(),
                "manifest_digest": SNAPSHOT_MANIFEST_DIGEST,
                "publication_receipt": snapshot_receipt.row(),
                "publication_receipt_digest": SNAPSHOT_RECEIPT_DIGEST,
            },
            "overlay": {
                "root": overlay_root.row(),
                "publication_receipt": overlay_receipt.row(),
                "publication_receipt_digest": OVERLAY_RECEIPT_DIGEST,
                "source_file_count": 6,
            },
            "controller": self_authority.row(), "python": runtime.row(),
            "materializer": materializer_authority.row(),
            "package_root": str(PACKAGE_ROOT),
            "package_receipt_path": str(PACKAGE_RECEIPT_PATH),
            "job_id": JOB_ID, "node": NODE,
            "exact_argv": argv, "exact_argv_digest": object_digest(argv),
            "authorization_token": authorization_token(),
        }
        attempt["attempt_digest"] = object_digest(attempt)
        attempt_raw, attempt_anchor = create_immutable_json(ATTEMPT_PATH, attempt)
        replay_all(authorities, directories)
        try:
            stdout, stderr, returncode, observed_argv = run_one_materializer(
                runtime, materializer_authority,
            )
        except Exception as error:
            # The immutable 0400 attempt already exists.  Every child/spawn/
            # timeout failure is a permanent manual HOLD, never a retry.
            raise PackageControllerError(
                "materializer failed after the durable attempt claim"
            ) from error
        if observed_argv != argv:
            raise PackageControllerError("materializer argv changed")
        if returncode != 0 or stderr != b"":
            raise PackageControllerError("single materializer child refused")
        child_report = strict_child_stdout(stdout)
        replay_all(authorities, directories)
        try:
            replayed_bytes, replayed_evidence = materializer.preflight_snapshot(
                SNAPSHOT_ROOT, manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
                materializer_sha256=BASE_MATERIALIZER_SHA256,
            )
            replayed_overlay_bytes, replayed_overlay_evidence = (
                materializer.preflight_overlay(
                    OVERLAY_ROOT, OVERLAY_RECEIPT_PATH,
                    materializer_sha256=MATERIALIZER_SHA256,
                    receipt_sha256=OVERLAY_RECEIPT_SHA256,
                    receipt_size=OVERLAY_RECEIPT_SIZE,
                    receipt_digest=OVERLAY_RECEIPT_DIGEST,
                    root_identity=OVERLAY_ROOT_IDENTITY,
                )
            )
        except Exception as error:
            raise PackageControllerError("post-child snapshot replay refused") from error
        if replayed_bytes != snapshot_bytes or replayed_evidence != snapshot_evidence:
            raise PackageControllerError("snapshot changed across materialization")
        if (
            replayed_overlay_bytes != overlay_bytes
            or replayed_overlay_evidence != overlay_evidence
        ):
            raise PackageControllerError("overlay changed across materialization")
        publication = validate_package(
            child_report, snapshot_bytes, snapshot_evidence,
            overlay_bytes, overlay_evidence, materializer,
        )
        replay_all(authorities, directories)
        evidence: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "PASS_R64_HOLD_PACKAGE_RECEIPT_GATED",
            "single_attempt": True, "retry_allowed": False,
            "launch_allowed": False,
            "attempt": {
                "path": str(ATTEMPT_PATH),
                "sha256": hashlib.sha256(attempt_raw).hexdigest(),
                "size": len(attempt_raw), "attempt_digest": attempt["attempt_digest"],
                "receipt_inode_anchor": attempt_anchor,
            },
            "snapshot": attempt["snapshot"],
            "overlay": attempt["overlay"],
            "controller": self_authority.row(), "python": runtime.row(),
            "materializer": materializer_authority.row(),
            "child": {
                "returncode": returncode,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stdout_size": len(stdout),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "stderr_size": len(stderr), "stderr_empty": True,
                "single_process_attempt": True,
                "process_group_terminal": True,
            },
            "publication": publication,
            "ssh_performed": False, "slurm_performed": False,
            "srun_performed": False, "gpu_attempt_claimed": False,
            "renderer_invoked": False,
        }
        evidence["evidence_digest"] = object_digest(evidence)
        create_immutable_json(EVIDENCE_PATH, evidence)
        return evidence
    finally:
        for authority in reversed(authorities):
            try:
                authority.close()
            except OSError:
                pass
        for directory in reversed(directories):
            try:
                directory.close()
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    # This state gate precedes argv parsing and every explicit stat/open/read,
    # process, target, receipt, cache, directory, or network action.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: r64 package controller awaits frozen core-v4 source pins, "
            "fresh exact6 overlay receipt/root pins, and a reviewed state-only "
            "READY copy",
            file=sys.stderr,
        )
        return 88
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        blocked = blocked_dynamic_pins()
        if blocked:
            raise PackageControllerError(
                "HOLD: dynamic exact35 pins are blocked: " + ",".join(blocked)
            )
        if values != ["--execute", authorization_token()]:
            raise PackageControllerError("controller argv/token differs")
        result = controller()
        print(canonical(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, KeyError, PackageControllerError) as error:
        print(f"r64 package controller refused: {error}", file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96


if __name__ == "__main__":
    raise SystemExit(main())
