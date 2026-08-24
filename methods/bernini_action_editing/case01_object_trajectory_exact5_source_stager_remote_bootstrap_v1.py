#!/usr/bin/env python3
"""Frozen remote-only bootstrap for exact15 receipt-gated stage/recovery.

This source is an independent held-stdin authority.  It validates the pinned
physical15 payload, performs either the exclusive-receipt-reserved stage
transaction or the replay-only receipt recovery transaction, and emits one
canonical result.  A sealed shadow is made visible with one same-parent POSIX
rename only while this process holds the O_EXCL-created sibling receipt
reservation.  This preserves whole-tree atomic visibility, but is not a
kernel no-replace guarantee: an uncooperative same-UID writer in the final
absence-check/rename window is explicitly outside the threat model.
It contains no SSH transport, local controller, state transition, or launch.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-source-stager-auh-v2"
MANIFEST_SCHEMA = SCHEMA + "-manifest"
PAYLOAD_SCHEMA = SCHEMA + "-payload"
RECEIPT_SCHEMA = SCHEMA + "-receipt"
RECEIPT_RESERVATION_SCHEMA = SCHEMA + "-receipt-reservation"
TERMINAL_SCHEMA = SCHEMA + "-terminal"
STAGE_OPERATION = "stage"
RECOVER_RECEIPT_OPERATION = "recover-receipt"
SHA_RE = re.compile(r"[0-9a-f]{64}")
FILE_MODE = 0o444
DIRECTORY_MODE = 0o555
RECEIPT_MODE = 0o400
RECEIPT_RESERVATION_MODE = 0o600
MAX_RECEIPT_SIZE = 1_048_576
EXPECTED_FILE_COUNT = 15
REMOTE_PARENT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
REMOTE_TARGET_ROOT = (
    REMOTE_PARENT / "bernini_case01_object_trajectory_exact5_source_staging_v1"
)
REMOTE_RECEIPT_PATH = REMOTE_PARENT / (
    "bernini_case01_object_trajectory_exact5_source_staging_v1.receipt_v1.json"
)
REMOTE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
REMOTE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
REMOTE_PYTHON_SIZE = 31_490_256
REMOTE_UID = 2012
REMOTE_GID = 2000

# Stable structural allowlist only.  Exact SHA-256/size rows come from the
# canonical controller-authenticated manifest and are rebound to every held
# payload byte below.  Keeping mutable downstream hashes out of this bootstrap
# makes the authority graph one-way: downstream source pins -> controller ->
# this generic frozen verifier.
SOURCE_RELATIVE_ALLOWLIST: tuple[str, ...] = (
    "artifacts/case01_oracle_object_trajectory_v1/scaffold.json",
    "artifacts/object_grounded_case01_0821_sam2_masklets_r2/receipt.json",
    "md/action_editing/20260821_man/evidence/case01_object_trajectory_scaffold_independent_audit_v1.json",
    "methods/bernini_action_editing/assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v1.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_root_fake_runner_v1.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v1.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v1.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_static_probe_v1.py",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_world4_probe_v1.py",
    "methods/bernini_action_editing/case01_oracle_object_trajectory_v1.py",
    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_v1.py",
    "methods/bernini_action_editing/object_trajectory_projection_v1.py",
    "methods/bernini_action_editing/tools/build_case01_object_trajectory_exact5_source_snapshot_v1.py",
    "methods/bernini_action_editing/tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py",
)


class SourceStageError(RuntimeError):
    """The receipt-reserved physical15 staging contract differs."""


class CommitRecoveryRequired(SourceStageError):
    """The target commit happened but its sibling receipt is not authoritative."""

    def __init__(self, terminal: Mapping[str, Any]) -> None:
        super().__init__("target committed; recover-receipt is required")
        self.terminal = dict(terminal)


class RenameOutcomeError(SourceStageError):
    """One rename returned an error after a held-inode namespace audit."""

    def __init__(
        self, *, committed: bool | None, classification: str, number: int,
    ):
        super().__init__(
            "ordinary POSIX publication rename returned error: "
            f"errno={number}; classification={classification}"
        )
        self.committed = committed
        self.classification = classification
        self.number = number


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceStageError("value is not canonical JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceStageError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SourceStageError(f"invalid JSON: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise SourceStageError(f"noncanonical JSON: {label}")
    return value


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(info.st_mode), int(info.st_nlink), int(info.st_rdev),
        int(info.st_size), int(getattr(info, "st_blocks", 0)),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _directory_anchor(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)), int(stat.S_IMODE(info.st_mode)),
    )


def _inode_anchor(info: os.stat_result) -> tuple[int, ...]:
    """Identity fields that remain stable while an owned inode is populated."""
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)),
    )


def _inode_anchor_from_identity(identity: Sequence[int]) -> tuple[int, ...]:
    if len(identity) != 11:
        raise SourceStageError("full inode identity shape differs")
    return (
        int(identity[0]), int(identity[1]), int(identity[2]), int(identity[3]),
        int(stat.S_IFMT(identity[4])),
    )


def _read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise SourceStageError("held read size differs")
    blocks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        blocks.append(block)
        offset += len(block)
    raw = b"".join(blocks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise SourceStageError("held read is incomplete")
    return raw


def _valid_relative(value: Any) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise SourceStageError("source relative path differs")
    path = Path(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise SourceStageError("source relative path is not canonical")
    return value


def _validate_specs(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) not in (list, tuple):
        raise SourceStageError("physical15 source rows differ")
    if any(type(row) is not dict for row in value):
        raise SourceStageError("physical15 source row type differs")
    rows = tuple(dict(row) for row in value)
    relatives = [_valid_relative(row.get("relative")) for row in rows]
    if (
        len(rows) != EXPECTED_FILE_COUNT
        or len(set(relatives)) != EXPECTED_FILE_COUNT
        or tuple(relatives) != SOURCE_RELATIVE_ALLOWLIST
    ):
        raise SourceStageError("physical15 source ordering differs")
    for row in rows:
        if (
            set(row) != {"relative", "sha256", "size"}
            or type(row["sha256"]) is not str
            or SHA_RE.fullmatch(row["sha256"]) is None
            or type(row["size"]) is not int
            or row["size"] <= 0
        ):
            raise SourceStageError("source authority fields differ")
    return rows


def blocked_sources(value: Any) -> tuple[str, ...]:
    return tuple(
        row["relative"]
        for row in _validate_specs(value)
        if (
            type(row["sha256"]) is not str
            or SHA_RE.fullmatch(row["sha256"]) is None
            or type(row["size"]) is not int
            or row["size"] <= 0
        )
    )


def expected_directories(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    directories = {"."}
    for row in rows:
        parent = Path(_valid_relative(row["relative"])).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return tuple(sorted(directories, key=lambda value: (value.count("/"), value)))


class HeldAuthority:
    def __init__(
        self,
        path: Path,
        descriptor: int,
        identity: tuple[int, ...],
        raw: bytes,
        sha256: str,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.raw = raw
        self.sha256 = sha256

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        replay = _read_fd(self.descriptor, opened.st_size)
        if (
            _identity(opened) != self.identity
            or _identity(named) != self.identity
            or hashlib.sha256(replay).hexdigest() != self.sha256
            or replay != self.raw
        ):
            raise SourceStageError(f"held authority changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _open_local_authority(
    path: Path, *, sha256: str | None, size: int | None,
) -> HeldAuthority:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise SourceStageError(f"local authority path differs: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) not in (0o444, 0o600, 0o644, 0o755)
    ):
        raise SourceStageError(f"local authority identity differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        observed = hashlib.sha256(first).hexdigest()
        if (
            _identity(before) != _identity(named)
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
            or first != second
            or (sha256 is not None and observed != sha256)
            or (size is not None and before.st_size != size)
        ):
            raise SourceStageError(f"local authority replay differs: {path}")
        return HeldAuthority(path, descriptor, _identity(before), first, observed)
    except BaseException:
        os.close(descriptor)
        raise


def _manifest_value(specifications: Any) -> dict[str, Any]:
    rows = _validate_specs(specifications)
    if blocked_sources(rows):
        raise SourceStageError(
            "physical15 pins remain BLOCKED: "
            + ",".join(blocked_sources(rows))
        )
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "target_root": str(REMOTE_TARGET_ROOT),
        "receipt_path": str(REMOTE_RECEIPT_PATH),
        "remote_python": {
            "path": str(REMOTE_PYTHON),
            "sha256": REMOTE_PYTHON_SHA256,
            "size": REMOTE_PYTHON_SIZE,
        },
        "remote_uid": REMOTE_UID,
        "remote_gid": REMOTE_GID,
        "file_count": EXPECTED_FILE_COUNT,
        "file_mode": FILE_MODE,
        "directory_mode": DIRECTORY_MODE,
        "receipt_mode": RECEIPT_MODE,
        "directories": list(expected_directories(rows)),
        "files": [dict(row) for row in rows],
        "publication_protocol": (
            "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
        ),
        "whole_tree_atomically_visible": True,
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "target_absent_rechecked": True,
        "receipt_is_consumption_gate": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "launch_allowed": False,
    }
    value["manifest_digest"] = object_digest(value)
    return value


def _payload_value(
    manifest: Mapping[str, Any], bootstrap_sha256: str,
    sources: Sequence[HeldAuthority], *, operation: str = STAGE_OPERATION,
    commit_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_files = manifest.get("files")
    rows = _validate_specs(manifest_files)
    if dict(manifest) != _manifest_value(rows):
        raise SourceStageError("held source manifest differs")
    if (
        len(sources) != len(rows)
        or operation not in (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION)
        or (operation == STAGE_OPERATION and commit_terminal is not None)
        or (operation == RECOVER_RECEIPT_OPERATION
            and type(commit_terminal) is not dict)
    ):
        raise SourceStageError("held source count differs")
    value: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA,
        "operation": operation,
        "commit_terminal": (
            None if commit_terminal is None else dict(commit_terminal)
        ),
        "manifest": dict(manifest),
        "bootstrap_source_sha256": bootstrap_sha256,
        "files": [
            {
                "relative": row["relative"],
                "sha256": row["sha256"],
                "size": row["size"],
                "content_b64": base64.b64encode(authority.raw).decode("ascii"),
            }
            for row, authority in zip(rows, sources)
        ],
    }
    value["authority_digest"] = object_digest(value)
    return value


def _parse_payload(
    raw: bytes, *, claimed_sha256: str, held_bootstrap_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes], tuple[dict[str, Any], ...]]:
    if (
        SHA_RE.fullmatch(claimed_sha256) is None
        or SHA_RE.fullmatch(held_bootstrap_sha256) is None
        or hashlib.sha256(raw).hexdigest() != claimed_sha256
    ):
        raise SourceStageError("held payload SHA differs")
    payload = _strict_json(raw, label="held source payload")
    if set(payload) != {
        "schema_version", "operation", "commit_terminal", "manifest",
        "bootstrap_source_sha256", "files", "authority_digest",
    }:
        raise SourceStageError("held payload field closure differs")
    unsigned = dict(payload)
    claimed_authority = unsigned.pop("authority_digest")
    manifest = payload.get("manifest")
    if type(manifest) is not dict:
        raise SourceStageError("held manifest type differs")
    rows = _validate_specs(manifest.get("files"))
    if (
        claimed_authority != object_digest(unsigned)
        or payload["schema_version"] != PAYLOAD_SCHEMA
        or payload["operation"]
        not in (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION)
        or payload["bootstrap_source_sha256"] != held_bootstrap_sha256
        or manifest != _manifest_value(rows)
    ):
        raise SourceStageError("held payload authority differs")
    files = payload["files"]
    if type(files) is not list or len(files) != EXPECTED_FILE_COUNT:
        raise SourceStageError("held payload file count differs")
    captured: dict[str, bytes] = {}
    for value, expected in zip(files, rows):
        if (
            type(value) is not dict
            or set(value) != {"relative", "sha256", "size", "content_b64"}
            or value["relative"] != expected["relative"]
            or value["sha256"] != expected["sha256"]
            or value["size"] != expected["size"]
        ):
            raise SourceStageError("held payload source row differs")
        try:
            content = base64.b64decode(value["content_b64"], validate=True)
        except Exception as error:
            raise SourceStageError("held payload base64 differs") from error
        if (
            len(content) != expected["size"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
            or base64.b64encode(content).decode("ascii") != value["content_b64"]
        ):
            raise SourceStageError("held payload source bytes differ")
        captured[expected["relative"]] = content
    _validate_commit_terminal_for_payload(payload)
    return payload, captured, rows


def _payload_raw_for_operation(
    payload: Mapping[str, Any], operation: str,
) -> bytes:
    if operation not in (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION):
        raise SourceStageError("payload operation differs")
    value = dict(payload)
    value.pop("authority_digest", None)
    value["operation"] = operation
    if operation == STAGE_OPERATION:
        value["commit_terminal"] = None
    elif type(value.get("commit_terminal")) is not dict:
        raise SourceStageError("receipt-recovery commit terminal is absent")
    value["authority_digest"] = object_digest(value)
    return canonical(value) + b"\n"


def _receipt_value(
    *,
    manifest: Mapping[str, Any],
    operation: str,
    request_payload_sha256: str,
    stage_payload_sha256: str,
    bootstrap_sha256: str,
    tree: Mapping[str, Any],
    target_identity: tuple[int, ...],
    receipt_inode_anchor: tuple[int, ...] | None = None,
    commit_terminal_digest: str | None = None,
) -> dict[str, Any]:
    if operation == STAGE_OPERATION:
        status = "STAGED_RECEIPT_GATED"
        observation = {
            "kind": "live_posix_rename_under_held_receipt_reservation",
            "root_identity": list(target_identity),
            "held_inode_continuity": True,
            "ordinary_posix_rename_performed_this_operation": True,
            "rename_noreplace_performed_this_operation": False,
            "target_absent_rechecked_before_rename": True,
            "whole_tree_atomically_visible": True,
            "historical_replacement_claim": "not_made",
        }
    elif operation == RECOVER_RECEIPT_OPERATION:
        status = "RECOVERED_RECEIPT_ONLY"
        observation = {
            "kind": "recovered_existing_exact15_current_inode",
            "root_identity": list(target_identity),
            "held_inode_continuity": True,
            "ordinary_posix_rename_performed_this_operation": False,
            "rename_noreplace_performed_this_operation": False,
            "target_absent_rechecked_before_rename": False,
            "whole_tree_atomically_visible": True,
            "historical_replacement_claim": "not_made",
        }
    else:
        raise SourceStageError("receipt operation differs")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "operation": operation,
        "target_root": manifest["target_root"],
        "receipt_path": manifest["receipt_path"],
        "manifest_digest": manifest["manifest_digest"],
        "request_payload_sha256": request_payload_sha256,
        "stage_payload_sha256": stage_payload_sha256,
        "bootstrap_source_sha256": bootstrap_sha256,
        "file_count": EXPECTED_FILE_COUNT,
        "files": tree["files"],
        "directories": tree["directories"],
        "file_mode": FILE_MODE,
        "directory_mode": DIRECTORY_MODE,
        "receipt_mode": RECEIPT_MODE,
        "held_parent_identity_replayed": True,
        "ancestor_chain_nofollow": True,
        "publication_protocol": (
            "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
        ),
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "receipt_is_consumption_gate": True,
        "receipt_is_admission": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "target_observation": observation,
        "commit_terminal_digest": commit_terminal_digest,
        "receipt_inode_anchor": (
            None if receipt_inode_anchor is None
            else list(receipt_inode_anchor)
        ),
        "launch_allowed": False,
    }
    receipt["receipt_digest"] = object_digest(receipt)
    return receipt


def _receipt_reservation_value(
    *,
    manifest: Mapping[str, Any],
    request_payload_sha256: str,
    bootstrap_sha256: str,
    receipt_inode_anchor: tuple[int, ...],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RECEIPT_RESERVATION_SCHEMA,
        "status": "RESERVED_NOT_ADMISSION",
        "operation": STAGE_OPERATION,
        "target_root": manifest["target_root"],
        "receipt_path": manifest["receipt_path"],
        "manifest_digest": manifest["manifest_digest"],
        "request_payload_sha256": request_payload_sha256,
        "stage_payload_sha256": request_payload_sha256,
        "bootstrap_source_sha256": bootstrap_sha256,
        "receipt_inode_anchor": list(receipt_inode_anchor),
        "receipt_mode": RECEIPT_RESERVATION_MODE,
        "receipt_is_admission": False,
        "cooperative_writer_exclusion": True,
        "launch_allowed": False,
    }
    value["reservation_digest"] = object_digest(value)
    return value


def _commit_terminal(
    manifest: Mapping[str, Any],
    *,
    request_payload_sha256: str,
    bootstrap_sha256: str,
    target_identity: tuple[int, ...],
    receipt_reservation_state: Mapping[str, Any],
    rename_result: str,
    rename_classification: str,
    receipt_phase: str,
    receipt_authoritative: bool,
    named_target_same_held_inode: bool,
    recovery_admissible: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
        "operation": STAGE_OPERATION,
        "target_root": manifest["target_root"],
        "receipt_path": manifest["receipt_path"],
        "manifest_digest": manifest["manifest_digest"],
        "request_payload_sha256": request_payload_sha256,
        "stage_payload_sha256": request_payload_sha256,
        "bootstrap_source_sha256": bootstrap_sha256,
        "target_root_identity": list(target_identity),
        "target_rename_commit_point_crossed": True,
        "rename_attempted": True,
        "rename_attempt_count": 1,
        "rename_result": rename_result,
        "rename_classification": rename_classification,
        "whole_tree_atomically_visible": True,
        "rename_noreplace": False,
        "target_absent_rechecked": True,
        "receipt_is_consumption_gate": True,
        "receipt_reservation_state": dict(receipt_reservation_state),
        "receipt_phase": receipt_phase,
        "receipt_authoritative": receipt_authoritative,
        "recovery_admissible": recovery_admissible,
        "named_target_same_held_inode": named_target_same_held_inode,
        "zero_publication_claim": False,
        "uncooperative_same_uid_race_out_of_scope": True,
        "launch_allowed": False,
    }
    value["terminal_digest"] = object_digest(value)
    return value


def _validated_target_full_identity(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != 11
        or any(type(item) is not int for item in value)
    ):
        raise SourceStageError("commit terminal target identity shape differs")
    identity = tuple(value)
    if (
        identity[0] <= 0
        or identity[1] <= 0
        or identity[2] != REMOTE_UID
        or identity[3] != REMOTE_GID
        or not stat.S_ISDIR(identity[4])
        or stat.S_IMODE(identity[4]) != DIRECTORY_MODE
        or identity[5] < 2
        or identity[6] != 0
        or any(item < 0 for item in identity[7:])
    ):
        raise SourceStageError("commit terminal target identity value differs")
    return identity


def _validate_commit_terminal_for_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    operation = payload.get("operation")
    terminal = payload.get("commit_terminal")
    if operation == STAGE_OPERATION:
        if terminal is not None:
            raise SourceStageError("stage payload commit terminal differs")
        return None
    if operation != RECOVER_RECEIPT_OPERATION or type(terminal) is not dict:
        raise SourceStageError("receipt-recovery commit terminal differs")
    target_identity = _validated_target_full_identity(
        terminal.get("target_root_identity")
    )
    reservation_state = _validated_receipt_reservation_state(
        terminal.get("receipt_reservation_state")
    )
    rename_result = terminal.get("rename_result")
    rename_classification = terminal.get("rename_classification")
    receipt_phase = terminal.get("receipt_phase")
    receipt_authoritative = terminal.get("receipt_authoritative")
    named_target_same = terminal.get("named_target_same_held_inode")
    recovery_admissible = terminal.get("recovery_admissible")
    if (
        (rename_result, rename_classification) not in (
            ("returned_success", "target_is_held_shadow"),
            ("raised", "applied_then_error_target_is_held_shadow"),
        )
        or receipt_phase not in (
            "reserved_0600", "partial_0600", "sealed_0400_exact",
            "sealed_0400_unverified", "unavailable",
        )
        or type(receipt_authoritative) is not bool
        or type(named_target_same) is not bool
        or type(recovery_admissible) is not bool
        or recovery_admissible is not True
        or named_target_same is not True
        or reservation_state["available"] is not True
        or receipt_authoritative != (receipt_phase == "sealed_0400_exact")
        or (
            receipt_phase in ("reserved_0600", "partial_0600")
            and reservation_state["mode"] != RECEIPT_RESERVATION_MODE
        )
        or (
            receipt_phase == "sealed_0400_exact"
            and reservation_state["mode"] != RECEIPT_MODE
        )
    ):
        raise SourceStageError("receipt-recovery terminal phase differs")
    stage_payload_sha256 = hashlib.sha256(
        _payload_raw_for_operation(payload, STAGE_OPERATION)
    ).hexdigest()
    expected = _commit_terminal(
        payload["manifest"],
        request_payload_sha256=stage_payload_sha256,
        bootstrap_sha256=payload["bootstrap_source_sha256"],
        target_identity=target_identity,
        receipt_reservation_state=reservation_state,
        rename_result=rename_result,
        rename_classification=rename_classification,
        receipt_phase=receipt_phase,
        receipt_authoritative=receipt_authoritative,
        named_target_same_held_inode=named_target_same,
        recovery_admissible=recovery_admissible,
    )
    if terminal != expected:
        raise SourceStageError("receipt-recovery commit terminal authority differs")
    return dict(terminal)


def _frame_payload(bootstrap_raw: bytes, payload_raw: bytes) -> bytes:
    if not bootstrap_raw or not payload_raw.endswith(b"\n"):
        raise SourceStageError("held stdin framing input differs")
    return str(len(bootstrap_raw)).encode("ascii") + b"\n" + bootstrap_raw + payload_raw


def _unframe_payload(raw: bytes) -> tuple[bytes, bytes]:
    line, separator, remainder = raw.partition(b"\n")
    if (
        separator != b"\n" or not line.isdigit()
        or line.startswith(b"0") or len(line) > 12
    ):
        raise SourceStageError("held stdin length framing differs")
    size = int(line)
    bootstrap = remainder[:size]
    payload = remainder[size:]
    if len(bootstrap) != size or not payload:
        raise SourceStageError("held stdin byte framing differs")
    return bootstrap, payload


def _hold_directory_chain(
    path: Path,
) -> tuple[list[int], int, tuple[tuple[int, ...], ...]]:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise SourceStageError(f"remote parent path differs: {path}")
    descriptors: list[int] = []
    anchors: list[tuple[int, ...]] = []
    current = os.open(
        "/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptors.append(current)
    root_info = os.fstat(current)
    if not stat.S_ISDIR(root_info.st_mode):
        os.close(current)
        raise SourceStageError("remote root directory identity differs")
    anchors.append(_directory_anchor(root_info))
    try:
        for component in path.parts[1:]:
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            child = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current,
            )
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(named.st_mode)
                or _directory_anchor(named) != _directory_anchor(opened)
            ):
                os.close(child)
                raise SourceStageError("remote ancestor identity differs")
            descriptors.append(child)
            anchors.append(_directory_anchor(opened))
            current = child
        parent_info = os.fstat(current)
        if parent_info.st_uid != REMOTE_UID or parent_info.st_gid != REMOTE_GID:
            raise SourceStageError("remote parent ownership differs")
        return descriptors, current, tuple(anchors)
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _replay_directory_chain(
    path: Path,
    descriptors: Sequence[int],
    anchors: Sequence[tuple[int, ...]],
) -> None:
    if (
        len(descriptors) != len(path.parts)
        or len(anchors) != len(descriptors)
        or path.resolve(strict=True) != path
    ):
        raise SourceStageError("held remote ancestor closure differs")
    for index, (descriptor, anchor) in enumerate(zip(descriptors, anchors)):
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_anchor(opened) != anchor
        ):
            raise SourceStageError("held remote ancestor changed")
        if index:
            named = os.stat(
                path.parts[index],
                dir_fd=descriptors[index - 1],
                follow_symlinks=False,
            )
            if _directory_anchor(named) != anchor:
                raise SourceStageError("named remote ancestor changed")
    parent = os.fstat(descriptors[-1])
    if parent.st_uid != REMOTE_UID or parent.st_gid != REMOTE_GID:
        raise SourceStageError("held remote parent ownership changed")


def _absent_at(parent_fd: int, name: str, *, label: str) -> None:
    if not name or "/" in name or name in (".", ".."):
        raise SourceStageError(f"{label} basename differs")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise SourceStageError(f"create-only {label} already exists")


def _plain_basename(value: str, *, label: str) -> str:
    if (
        type(value) is not str or not value or value in (".", "..")
        or "/" in value or "\x00" in value
    ):
        raise SourceStageError(f"{label} basename differs")
    return value


def _open_child_directory(
    parent_fd: int, name: str, *, expected_mode: int | None,
) -> int:
    name = _plain_basename(name, label="child directory")
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            _identity(named) != _identity(opened)
            or not stat.S_ISDIR(opened.st_mode)
            or (expected_mode is not None
                and stat.S_IMODE(opened.st_mode) != expected_mode)
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
        ):
            raise SourceStageError(f"child directory identity differs: {name}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_created_directory(
    parent_fd: int, name: str, *, label: str,
) -> tuple[int, os.stat_result]:
    """Hold a just-created directory before any named identity gate."""
    name = _plain_basename(name, label=label)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        return descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _open_created_child_directory(
    parent_fd: int, name: str,
) -> tuple[int, os.stat_result]:
    return _open_created_directory(
        parent_fd, name, label="created shadow child directory",
    )


def _reserve_creation_descriptor(parent_fd: int, *, label: str) -> int:
    """Reserve one descriptor slot before mkdir creates an unowned name."""
    try:
        descriptor = os.dup(parent_fd)
    except OSError as error:
        raise SourceStageError(
            f"{label} descriptor reserve unavailable before mkdir"
        ) from error
    try:
        if _identity(os.fstat(descriptor)) != _identity(os.fstat(parent_fd)):
            raise SourceStageError(f"{label} descriptor reserve differs")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_file_at(
    parent_fd: int,
    name: str,
    raw: bytes,
    sha256: str,
    *,
    relative: str,
    creation_anchors: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    name = _plain_basename(name, label="shadow source")
    relative = _valid_relative(relative)
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0,
        dir_fd=parent_fd,
    )
    try:
        created = os.fstat(descriptor)
        creation_anchors[relative] = _inode_anchor(created)
        named_created = os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _identity(created) != _identity(named_created)
            or not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or stat.S_IMODE(created.st_mode) != 0
            or created.st_uid != REMOTE_UID
            or created.st_gid != REMOTE_GID
        ):
            raise SourceStageError("shadow source creation identity differs")
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise SourceStageError("shadow source write made no progress")
            offset += count
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        replay = _read_fd(descriptor, before.st_size)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0
            or replay != raw or hashlib.sha256(replay).hexdigest() != sha256
        ):
            raise SourceStageError("shadow source preseal replay differs")
        os.fchmod(descriptor, FILE_MODE)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _identity(after) != _identity(named)
            or after.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != FILE_MODE
            or after.st_uid != REMOTE_UID
            or after.st_gid != REMOTE_GID
        ):
            raise SourceStageError("shadow source seal differs")
        return _identity(after)
    finally:
        os.close(descriptor)


def _open_shadow_root(
    parent_fd: int, name: str,
) -> tuple[int, os.stat_result]:
    return _open_created_directory(
        parent_fd, name, label="created random shadow root",
    )


def _assert_created_directory_named(
    parent_fd: int,
    name: str,
    descriptor: int,
    creation_anchor: tuple[int, ...],
    *,
    label: str,
) -> tuple[int, ...]:
    """Validate a held creation only after its ownership anchor is recorded."""
    name = _plain_basename(name, label=label)
    opened = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        _inode_anchor(opened) != creation_anchor
        or _inode_anchor(named) != creation_anchor
        or _identity(opened) != _identity(named)
        or not stat.S_ISDIR(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o700
        or opened.st_uid != REMOTE_UID
        or opened.st_gid != REMOTE_GID
    ):
        raise SourceStageError(f"{label} identity differs")
    return _identity(opened)


def _assert_named_inode(
    parent_fd: int,
    name: str,
    descriptor: int,
    anchor: tuple[int, ...],
    *,
    expected_identity: tuple[int, ...] | None,
    label: str,
) -> tuple[int, ...]:
    name = _plain_basename(name, label=label)
    opened = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        _inode_anchor(opened) != anchor
        or _inode_anchor(named) != anchor
        or _identity(opened) != _identity(named)
        or (expected_identity is not None
            and _identity(opened) != expected_identity)
    ):
        raise SourceStageError(f"held {label} inode differs")
    return _identity(opened)


def _build_shadow_at(
    shadow_fd: int,
    captured: Mapping[str, bytes],
    creation_anchors: dict[str, tuple[int, ...]],
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    rows = _validate_specs(specifications)
    directories = expected_directories(rows)
    directory_fds: dict[str, int] = {".": shadow_fd}
    owned_directory_fds: list[int] = []
    sealed_identities: dict[str, tuple[int, ...]] = {}
    try:
        for relative in directories:
            if relative == ".":
                continue
            relative_path = Path(relative)
            parent_relative = str(relative_path.parent)
            parent_directory_fd = directory_fds[parent_relative]
            reserve_fd = _reserve_creation_descriptor(
                parent_directory_fd, label="shadow child creation",
            )
            try:
                os.mkdir(
                    relative_path.name, 0o700,
                    dir_fd=parent_directory_fd,
                )
                os.close(reserve_fd)
                reserve_fd = -1
                try:
                    child_fd, opened_child = _open_created_child_directory(
                        parent_directory_fd, relative_path.name,
                    )
                except BaseException as error:
                    # Without a held fd there is no creation authority.  A
                    # post-failure named stat could describe a competitor, so
                    # the unknown entry is preserved and exact cleanup refuses.
                    raise SourceStageError(
                        "shadow child held-open unavailable after mkdir; "
                        "unknown inode preserved for permanent HOLD"
                    ) from error
            finally:
                if reserve_fd >= 0:
                    os.close(reserve_fd)
            creation_anchors[relative] = _inode_anchor(opened_child)
            directory_fds[relative] = child_fd
            owned_directory_fds.append(child_fd)
            _assert_created_directory_named(
                parent_directory_fd,
                relative_path.name,
                child_fd,
                creation_anchors[relative],
                label="shadow child directory creation",
            )
        for row in rows:
            relative_path = Path(row["relative"])
            sealed_identities[row["relative"]] = _write_file_at(
                directory_fds[str(relative_path.parent)],
                relative_path.name,
                captured[row["relative"]],
                row["sha256"],
                relative=row["relative"],
                creation_anchors=creation_anchors,
            )
        for relative in reversed(directories):
            descriptor = directory_fds[relative]
            os.fsync(descriptor)
            os.fchmod(descriptor, DIRECTORY_MODE)
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(sealed.st_mode)
                or stat.S_IMODE(sealed.st_mode) != DIRECTORY_MODE
                or sealed.st_uid != REMOTE_UID
                or sealed.st_gid != REMOTE_GID
            ):
                raise SourceStageError(f"shadow directory seal differs: {relative}")
            sealed_identities[relative] = _identity(sealed)
        expected_nodes = set(directories) | {row["relative"] for row in rows}
        if (
            set(creation_anchors) != expected_nodes
            or set(sealed_identities) != expected_nodes
        ):
            raise SourceStageError("shadow inode authority closure differs")
        return sealed_identities
    finally:
        for descriptor in reversed(owned_directory_fds):
            os.close(descriptor)


def _read_tree_held(
    root_fd: int,
    sealed_identities: Mapping[str, tuple[int, ...]],
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _validate_specs(specifications)
    expected_files = {row["relative"]: row for row in rows}
    expected_dirs = set(expected_directories(rows))
    actual_files: dict[str, dict[str, Any]] = {}
    actual_dirs = {"."}
    if set(sealed_identities) != expected_dirs | set(expected_files):
        raise SourceStageError("sealed inode plan closure differs")
    pending: list[tuple[int, str]] = [(os.dup(root_fd), ".")]
    try:
        while pending:
            directory_fd, prefix = pending.pop()
            before = os.fstat(directory_fd)
            try:
                if (
                    _identity(before) != sealed_identities.get(prefix)
                    or
                    not stat.S_ISDIR(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != DIRECTORY_MODE
                    or before.st_uid != REMOTE_UID
                    or before.st_gid != REMOTE_GID
                ):
                    raise SourceStageError(
                        f"sealed directory differs: {prefix}"
                    )
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        relative = (
                            entry.name if prefix == "."
                            else f"{prefix}/{entry.name}"
                        )
                        child = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(child.st_mode):
                            opened_child = _open_child_directory(
                                directory_fd,
                                entry.name,
                                expected_mode=DIRECTORY_MODE,
                            )
                            actual_dirs.add(relative)
                            pending.append((opened_child, relative))
                            if (
                                _identity(child) != sealed_identities.get(relative)
                                or _identity(os.fstat(opened_child))
                                != sealed_identities.get(relative)
                            ):
                                raise SourceStageError(
                                    "sealed child directory inode differs: "
                                    + relative
                                )
                        elif stat.S_ISREG(child.st_mode):
                            if (
                                _identity(child) != sealed_identities.get(relative)
                                or
                                child.st_nlink != 1
                                or stat.S_IMODE(child.st_mode) != FILE_MODE
                                or child.st_uid != REMOTE_UID
                                or child.st_gid != REMOTE_GID
                            ):
                                raise SourceStageError(
                                    "sealed file identity differs: " + relative
                                )
                            expected = expected_files.get(relative)
                            if expected is None:
                                raise SourceStageError(
                                    "sealed tree has extra file: " + relative
                                )
                            descriptor = os.open(
                                entry.name,
                                os.O_RDONLY
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0)
                                | getattr(os, "O_NONBLOCK", 0),
                                dir_fd=directory_fd,
                            )
                            try:
                                opened = os.fstat(descriptor)
                                raw = _read_fd(descriptor, opened.st_size)
                                after = os.fstat(descriptor)
                                named = os.stat(
                                    entry.name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                            finally:
                                os.close(descriptor)
                            if (
                                _identity(opened) != _identity(after)
                                or _identity(opened) != _identity(named)
                                or _identity(opened)
                                != sealed_identities.get(relative)
                                or opened.st_size != expected["size"]
                                or hashlib.sha256(raw).hexdigest()
                                != expected["sha256"]
                            ):
                                raise SourceStageError(
                                    "sealed file bytes differ: " + relative
                                )
                            actual_files[relative] = {
                                "relative": relative,
                                "sha256": expected["sha256"],
                                "size": expected["size"],
                                "mode": FILE_MODE,
                                "nlink": 1,
                            }
                        else:
                            raise SourceStageError(
                                "sealed tree special entry: " + relative
                            )
                if _identity(os.fstat(directory_fd)) != _identity(before):
                    raise SourceStageError(
                        f"sealed directory changed: {prefix}"
                    )
            finally:
                os.close(directory_fd)
    except BaseException:
        for descriptor, _prefix in pending:
            os.close(descriptor)
        raise
    if set(actual_files) != set(expected_files) or actual_dirs != expected_dirs:
        raise SourceStageError("sealed exact-tree closure differs")
    return {
        "files": [actual_files[row["relative"]] for row in rows],
        "directories": sorted(actual_dirs, key=lambda value: (value.count("/"), value)),
    }


def _rename_under_receipt_reservation(
    parent_fd: int,
    shadow_name: str,
    target_name: str,
    shadow_fd: int,
    shadow_anchor: tuple[int, ...],
    reservation: "HeldReceiptReservation",
) -> None:
    """Publish once with ordinary POSIX rename under a cooperative lock.

    The immediately preceding target absence check is intentionally not
    described as a kernel no-replace guarantee.  The held O_EXCL receipt
    reservation excludes another conforming stager; an uncooperative same-UID
    writer in this final window is outside the declared threat model.
    """
    shadow_name = _plain_basename(shadow_name, label="random shadow")
    target_name = _plain_basename(target_name, label="target root")
    reservation.require_reserved(parent_fd)
    _absent_at(parent_fd, target_name, label="target root recheck")
    try:
        os.rename(
            shadow_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except OSError as error:
        opened_anchor = _inode_anchor(os.fstat(shadow_fd))
        if opened_anchor != shadow_anchor:
            raise RenameOutcomeError(
                committed=None,
                classification="held_shadow_inode_changed",
                number=error.errno or 0,
            ) from error
        observed: dict[str, tuple[int, ...] | None] = {}
        for label, name in (("shadow", shadow_name), ("target", target_name)):
            try:
                observed[label] = _inode_anchor(os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False,
                ))
            except FileNotFoundError:
                observed[label] = None
        if observed == {"shadow": None, "target": shadow_anchor}:
            raise RenameOutcomeError(
                committed=True,
                classification="applied_then_error_target_is_held_shadow",
                number=error.errno or 0,
            ) from error
        if observed == {"shadow": shadow_anchor, "target": None}:
            raise RenameOutcomeError(
                committed=False,
                classification="not_applied_shadow_remains_target_absent",
                number=error.errno or 0,
            ) from error
        raise RenameOutcomeError(
            committed=None,
            classification="ambiguous_namespace_preserved_permanent_hold",
            number=error.errno or 0,
        ) from error


class HeldReceiptReservation:
    def __init__(
        self,
        name: str,
        descriptor: int,
        anchor: tuple[int, ...],
        reserved_raw: bytes | None,
    ) -> None:
        self.name = name
        self.descriptor = descriptor
        self.anchor = anchor
        self.reserved_raw = reserved_raw

    def snapshot(self, parent_fd: int) -> tuple[bytes, tuple[int, ...]]:
        if self.descriptor < 0:
            raise SourceStageError("held receipt reservation is closed")
        opened = os.fstat(self.descriptor)
        if opened.st_size > MAX_RECEIPT_SIZE:
            raise SourceStageError("receipt reservation size exceeds bound")
        first = _read_fd(self.descriptor, opened.st_size)
        middle = os.fstat(self.descriptor)
        second = _read_fd(self.descriptor, opened.st_size)
        after = os.fstat(self.descriptor)
        named = os.stat(
            self.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _inode_anchor(opened) != self.anchor
            or _inode_anchor(middle) != self.anchor
            or _inode_anchor(after) != self.anchor
            or _inode_anchor(named) != self.anchor
            or _identity(opened) != _identity(middle)
            or _identity(opened) != _identity(after)
            or _identity(opened) != _identity(named)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode)
            not in (RECEIPT_RESERVATION_MODE, RECEIPT_MODE)
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
            or first != second
        ):
            raise SourceStageError("held receipt reservation changed")
        return first, _identity(opened)

    def require_reserved(self, parent_fd: int) -> tuple[int, ...]:
        raw, identity = self.snapshot(parent_fd)
        if (
            self.reserved_raw is None
            or raw != self.reserved_raw
            or identity[7] != len(self.reserved_raw)
            or stat.S_IMODE(identity[4]) != RECEIPT_RESERVATION_MODE
        ):
            raise SourceStageError("receipt reservation record differs")
        return identity

    def detach(self) -> int:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor < 0:
            raise SourceStageError("receipt reservation detach differs")
        return descriptor

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _receipt_reservation_state_value(
    raw: bytes, identity: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "available": True,
        "inode_anchor": list(_inode_anchor_from_identity(identity)),
        "identity": list(identity),
        "mode": stat.S_IMODE(identity[4]),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _unavailable_receipt_reservation_state() -> dict[str, Any]:
    return {"available": False}


def _validated_receipt_reservation_state(
    value: Any,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("available")) is not bool:
        raise SourceStageError("receipt reservation terminal state differs")
    if value["available"] is False:
        if set(value) != {"available"}:
            raise SourceStageError("unavailable receipt state closure differs")
        return {"available": False}
    if set(value) != {
        "available", "inode_anchor", "identity", "mode", "size", "sha256",
    }:
        raise SourceStageError("receipt reservation state closure differs")
    identity = _validated_receipt_full_identity(value["identity"])
    anchor = value["inode_anchor"]
    if (
        type(anchor) is not list
        or len(anchor) != 5
        or any(type(item) is not int for item in anchor)
        or tuple(anchor) != _inode_anchor_from_identity(identity)
        or value["mode"] != stat.S_IMODE(identity[4])
        or value["mode"]
        not in (RECEIPT_RESERVATION_MODE, RECEIPT_MODE)
        or value["size"] != identity[7]
        or type(value["size"]) is not int
        or not 0 <= value["size"] <= MAX_RECEIPT_SIZE
        or type(value["sha256"]) is not str
        or SHA_RE.fullmatch(value["sha256"]) is None
    ):
        raise SourceStageError("receipt reservation state value differs")
    return dict(value)


def _validated_receipt_full_identity(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != 11
        or any(type(item) is not int for item in value)
    ):
        raise SourceStageError("receipt reservation identity shape differs")
    identity = tuple(value)
    if (
        identity[0] <= 0
        or identity[1] <= 0
        or identity[2] != REMOTE_UID
        or identity[3] != REMOTE_GID
        or not stat.S_ISREG(identity[4])
        or stat.S_IMODE(identity[4])
        not in (RECEIPT_RESERVATION_MODE, RECEIPT_MODE)
        or identity[5] != 1
        or identity[6] != 0
        or identity[7] < 0
        or any(item < 0 for item in identity[8:])
    ):
        raise SourceStageError("receipt reservation identity value differs")
    return identity


def _reserve_receipt(
    parent_fd: int,
    name: str,
    *,
    manifest: Mapping[str, Any],
    request_payload_sha256: str,
    bootstrap_sha256: str,
) -> HeldReceiptReservation:
    name = _plain_basename(name, label="receipt reservation")
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0,
        dir_fd=parent_fd,
    )
    anchor: tuple[int, ...] | None = None
    try:
        created = os.fstat(descriptor)
        anchor = _inode_anchor(created)
        os.fchmod(descriptor, RECEIPT_RESERVATION_MODE)
        value = _receipt_reservation_value(
            manifest=manifest,
            request_payload_sha256=request_payload_sha256,
            bootstrap_sha256=bootstrap_sha256,
            receipt_inode_anchor=anchor,
        )
        raw = canonical(value) + b"\n"
        if len(raw) > MAX_RECEIPT_SIZE:
            raise SourceStageError("receipt reservation exceeds bound")
        offset = 0
        while offset < len(raw):
            count = os.pwrite(descriptor, raw[offset:], offset)
            if count <= 0:
                raise SourceStageError(
                    "receipt reservation write made no progress"
                )
            offset += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _inode_anchor(opened) != anchor
            or _inode_anchor(named) != anchor
            or _identity(opened) != _identity(named)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
            or stat.S_IMODE(opened.st_mode) != RECEIPT_RESERVATION_MODE
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
        ):
            raise SourceStageError("receipt reservation creation differs")
        os.fsync(parent_fd)
        reservation = HeldReceiptReservation(name, descriptor, anchor, raw)
        reservation.require_reserved(parent_fd)
        return reservation
    except BaseException as error:
        cleanup_error: BaseException | None = None
        if anchor is not None:
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if _inode_anchor(named) != anchor:
                    raise SourceStageError(
                        "receipt reservation replacement preserved"
                    )
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                if _inode_anchor(os.fstat(descriptor)) != anchor:
                    raise SourceStageError(
                        "discarded receipt reservation inode changed"
                    )
            except FileNotFoundError:
                cleanup_error = SourceStageError(
                    "receipt reservation disappeared during creation"
                )
            except BaseException as discard_error:
                cleanup_error = discard_error
        os.close(descriptor)
        if cleanup_error is not None:
            raise cleanup_error from error
        raise


def _discard_receipt_reservation(
    parent_fd: int,
    reservation: HeldReceiptReservation,
    *,
    validate_only: bool = False,
) -> None:
    identity = reservation.require_reserved(parent_fd)
    if validate_only:
        return
    os.unlink(reservation.name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    try:
        os.stat(
            reservation.name, dir_fd=parent_fd, follow_symlinks=False,
        )
    except FileNotFoundError:
        after = os.fstat(reservation.descriptor)
        if (
            _inode_anchor(after) != reservation.anchor
            or after.st_nlink != 0
            or _inode_anchor_from_identity(identity) != reservation.anchor
        ):
            raise SourceStageError("discarded receipt reservation differs")
        return
    raise SourceStageError("receipt reservation removal was not exact")


def _open_receipt_reservation_for_recovery(
    parent_fd: int, name: str, expected_state: Mapping[str, Any],
) -> HeldReceiptReservation:
    state = _validated_receipt_reservation_state(expected_state)
    if state["available"] is not True:
        raise SourceStageError("receipt recovery state is unavailable")
    name = _plain_basename(name, label="recovery receipt reservation")
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    named_mode = stat.S_IMODE(named.st_mode)
    if not (
        named_mode == state["mode"]
        or (
            state["mode"] == RECEIPT_RESERVATION_MODE
            and named_mode == RECEIPT_MODE
        )
    ):
        raise SourceStageError("receipt recovery mode transition differs")
    access = (
        os.O_RDWR
        if named_mode == RECEIPT_RESERVATION_MODE
        else os.O_RDONLY
    )
    descriptor = os.open(
        name,
        access | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        anchor = tuple(state["inode_anchor"])
        reservation = HeldReceiptReservation(name, descriptor, anchor, None)
        raw, identity = reservation.snapshot(parent_fd)
        if (
            _identity(named) != identity
            or (
                named_mode == state["mode"]
                and _receipt_reservation_state_value(raw, identity) != state
            )
        ):
            raise SourceStageError(
                "receipt recovery reservation differs from terminal"
            )
        return reservation
    except BaseException:
        os.close(descriptor)
        raise


class HeldReceipt:
    def __init__(
        self,
        name: str,
        descriptor: int,
        identity: tuple[int, ...],
        raw: bytes,
    ) -> None:
        self.name = name
        self.descriptor = descriptor
        self.identity = identity
        self.raw = raw

    def replay(
        self, parent_fd: int,
    ) -> tuple[bytes, dict[str, Any]]:
        opened = os.fstat(self.descriptor)
        first = _read_fd(self.descriptor, opened.st_size)
        middle = os.fstat(self.descriptor)
        second = _read_fd(self.descriptor, opened.st_size)
        after = os.fstat(self.descriptor)
        named = os.stat(
            self.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        value = _strict_json(first, label="source staging receipt")
        if (
            _identity(opened) != self.identity
            or _identity(middle) != self.identity
            or _identity(after) != self.identity
            or _identity(named) != self.identity
            or first != self.raw
            or second != self.raw
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != RECEIPT_MODE
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
            or value.get("receipt_inode_anchor")
            != list(_inode_anchor(opened))
        ):
            raise SourceStageError("held receipt inode or bytes changed")
        return first, value

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _seal_reserved_receipt(
    parent_fd: int,
    reservation: HeldReceiptReservation,
    value: Mapping[str, Any],
    *,
    expected_prior_state: Mapping[str, Any] | None = None,
) -> HeldReceipt:
    prior_raw, prior_identity = reservation.snapshot(parent_fd)
    prior_state = _receipt_reservation_state_value(prior_raw, prior_identity)
    if expected_prior_state is None:
        if (
            reservation.reserved_raw is None
            or prior_raw != reservation.reserved_raw
            or stat.S_IMODE(prior_identity[4]) != RECEIPT_RESERVATION_MODE
        ):
            raise SourceStageError("receipt reservation preseal state differs")
    elif prior_state != _validated_receipt_reservation_state(
        expected_prior_state
    ):
        raise SourceStageError("receipt recovery prior state differs")
    materialized = dict(value)
    if materialized.get("receipt_inode_anchor") is not None:
        raise SourceStageError("receipt inode anchor was preclaimed")
    materialized["receipt_inode_anchor"] = list(reservation.anchor)
    materialized.pop("receipt_digest", None)
    materialized["receipt_digest"] = object_digest(materialized)
    raw = canonical(materialized) + b"\n"
    if len(raw) > MAX_RECEIPT_SIZE:
        raise SourceStageError("materialized receipt exceeds bound")
    os.fchmod(reservation.descriptor, RECEIPT_RESERVATION_MODE)
    os.ftruncate(reservation.descriptor, 0)
    offset = 0
    while offset < len(raw):
        count = os.pwrite(reservation.descriptor, raw[offset:], offset)
        if count <= 0:
            raise SourceStageError("receipt seal write made no progress")
        offset += count
    os.fsync(reservation.descriptor)
    if _read_fd(reservation.descriptor, len(raw)) != raw:
        raise SourceStageError("receipt preseal replay differs")
    os.fchmod(reservation.descriptor, RECEIPT_MODE)
    os.fsync(reservation.descriptor)
    opened = os.fstat(reservation.descriptor)
    named = os.stat(
        reservation.name, dir_fd=parent_fd, follow_symlinks=False,
    )
    if (
        _inode_anchor(opened) != reservation.anchor
        or _inode_anchor(named) != reservation.anchor
        or _identity(opened) != _identity(named)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != RECEIPT_MODE
        or opened.st_uid != REMOTE_UID
        or opened.st_gid != REMOTE_GID
        or opened.st_size != len(raw)
    ):
        raise SourceStageError("receipt seal differs")
    os.fsync(parent_fd)
    held = HeldReceipt(
        reservation.name,
        reservation.detach(),
        _identity(opened),
        raw,
    )
    return held


def _discard_shadow_held(
    parent_fd: int,
    name: str,
    root_fd: int,
    root_anchor: tuple[int, ...],
    creation_anchors: Mapping[str, tuple[int, ...]],
    *,
    validate_only: bool = False,
) -> None:
    """Remove only the exact inodes created by this attempt.

    Validation is deliberately two-pass.  A pre-existing replacement is
    preserved and causes refusal before any owned child is removed.  The
    deletion pass repeats each openat/statat identity gate immediately before
    unlinkat/rmdirat so a stale name is never accepted as our inode.
    """
    name = _plain_basename(name, label="random shadow cleanup")
    if creation_anchors.get(".") != root_anchor:
        raise SourceStageError("shadow cleanup root authority differs")
    _assert_named_inode(
        parent_fd,
        name,
        root_fd,
        root_anchor,
        expected_identity=None,
        label="shadow cleanup root",
    )

    observed: set[str] = set()

    def validate_tree(directory_fd: int, prefix: str) -> None:
        expected = creation_anchors.get(prefix)
        if expected is None or _inode_anchor(os.fstat(directory_fd)) != expected:
            raise SourceStageError("shadow cleanup directory replacement")
        observed.add(prefix)
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                relative = (
                    entry.name if prefix == "."
                    else f"{prefix}/{entry.name}"
                )
                expected_child = creation_anchors.get(relative)
                child = entry.stat(follow_symlinks=False)
                if (
                    expected_child is None
                    or _inode_anchor(child) != expected_child
                ):
                    raise SourceStageError(
                        "shadow cleanup named child replacement: " + relative
                    )
                observed.add(relative)
                if stat.S_ISDIR(child.st_mode):
                    child_fd = _open_child_directory(
                        directory_fd, entry.name, expected_mode=None,
                    )
                    try:
                        if _inode_anchor(os.fstat(child_fd)) != expected_child:
                            raise SourceStageError(
                                "shadow cleanup opened child replacement: "
                                + relative
                            )
                        validate_tree(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif not stat.S_ISREG(child.st_mode):
                    raise SourceStageError(
                        "shadow cleanup special child differs: " + relative
                    )
        if _inode_anchor(os.fstat(directory_fd)) != expected:
            raise SourceStageError("shadow cleanup directory changed")

    validate_tree(root_fd, ".")
    if observed != set(creation_anchors):
        raise SourceStageError("shadow cleanup exact inode closure differs")
    if validate_only:
        return

    def require_absent(directory_fd: int, child_name: str) -> None:
        try:
            os.stat(child_name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise SourceStageError("shadow cleanup removal was not exact")

    def discard_contents(directory_fd: int, prefix: str) -> None:
        expected_directory = creation_anchors[prefix]
        if _inode_anchor(os.fstat(directory_fd)) != expected_directory:
            raise SourceStageError("shadow cleanup held directory changed")
        os.fchmod(directory_fd, 0o700)
        with os.scandir(directory_fd) as iterator:
            names = sorted(entry.name for entry in iterator)
        for child_name in names:
            relative = (
                child_name if prefix == "."
                else f"{prefix}/{child_name}"
            )
            expected_child = creation_anchors[relative]
            named = os.stat(
                child_name, dir_fd=directory_fd, follow_symlinks=False,
            )
            if _inode_anchor(named) != expected_child:
                raise SourceStageError(
                    "shadow cleanup child changed before removal: " + relative
                )
            if stat.S_ISDIR(named.st_mode):
                child_fd = _open_child_directory(
                    directory_fd, child_name, expected_mode=None,
                )
                try:
                    if _inode_anchor(os.fstat(child_fd)) != expected_child:
                        raise SourceStageError(
                            "shadow cleanup directory changed before removal: "
                            + relative
                        )
                    discard_contents(child_fd, relative)
                    named_again = os.stat(
                        child_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        _inode_anchor(named_again) != expected_child
                        or _inode_anchor(os.fstat(child_fd)) != expected_child
                    ):
                        raise SourceStageError(
                            "shadow cleanup directory replacement before rmdir: "
                            + relative
                        )
                    os.rmdir(child_name, dir_fd=directory_fd)
                    require_absent(directory_fd, child_name)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(named.st_mode):
                child_fd = os.open(
                    child_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    named_again = os.stat(
                        child_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        _inode_anchor(opened) != expected_child
                        or _inode_anchor(named_again) != expected_child
                        or _identity(opened) != _identity(named_again)
                    ):
                        raise SourceStageError(
                            "shadow cleanup file replacement before unlink: "
                            + relative
                        )
                    os.unlink(child_name, dir_fd=directory_fd)
                    require_absent(directory_fd, child_name)
                    if _inode_anchor(os.fstat(child_fd)) != expected_child:
                        raise SourceStageError(
                            "shadow cleanup unlinked file identity differs: "
                            + relative
                        )
                finally:
                    os.close(child_fd)
            else:
                raise SourceStageError(
                    "shadow cleanup special entry before removal: " + relative
                )
        os.fsync(directory_fd)
        if _inode_anchor(os.fstat(directory_fd)) != expected_directory:
            raise SourceStageError("shadow cleanup held directory inode changed")

    discard_contents(root_fd, ".")
    _assert_named_inode(
        parent_fd,
        name,
        root_fd,
        root_anchor,
        expected_identity=None,
        label="shadow cleanup root",
    )
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if _inode_anchor(os.fstat(root_fd)) != root_anchor:
            raise SourceStageError("removed shadow root identity differs")
        return
    raise SourceStageError("shadow cleanup root removal differs")


def _open_existing_target(
    parent_fd: int, name: str,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Hold an already-published target without following a named link."""
    name = _plain_basename(name, label="existing target root")
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor = _open_child_directory(
        parent_fd, name, expected_mode=DIRECTORY_MODE,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            _identity(named) != _identity(opened)
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
        ):
            raise SourceStageError("existing target root identity differs")
        return descriptor, _inode_anchor(opened), _identity(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _capture_existing_tree_held(
    root_fd: int,
    captured: Mapping[str, bytes],
    specifications: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    """Capture and replay the exact existing physical15 tree via held dirfds."""
    rows = _validate_specs(specifications)
    expected_files = {row["relative"]: row for row in rows}
    expected_dirs = set(expected_directories(rows))
    if set(captured) != set(expected_files):
        raise SourceStageError("existing target source-byte closure differs")
    identities: dict[str, tuple[int, ...]] = {}
    visited_dirs: set[str] = set()
    visited_files: set[str] = set()
    pending: list[tuple[int, str]] = [(os.dup(root_fd), ".")]
    try:
        while pending:
            directory_fd, prefix = pending.pop()
            try:
                before = os.fstat(directory_fd)
                if (
                    prefix not in expected_dirs
                    or not stat.S_ISDIR(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != DIRECTORY_MODE
                    or before.st_uid != REMOTE_UID
                    or before.st_gid != REMOTE_GID
                ):
                    raise SourceStageError(
                        "existing target directory differs: " + prefix
                    )
                identity = _identity(before)
                prior = identities.setdefault(prefix, identity)
                if prior != identity or prefix in visited_dirs:
                    raise SourceStageError(
                        "existing target directory identity closure differs"
                    )
                visited_dirs.add(prefix)
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        relative = (
                            entry.name if prefix == "."
                            else f"{prefix}/{entry.name}"
                        )
                        child = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(child.st_mode):
                            if relative not in expected_dirs:
                                raise SourceStageError(
                                    "existing target has extra directory: "
                                    + relative
                                )
                            child_fd = _open_child_directory(
                                directory_fd,
                                entry.name,
                                expected_mode=DIRECTORY_MODE,
                            )
                            opened_child = os.fstat(child_fd)
                            if _identity(child) != _identity(opened_child):
                                os.close(child_fd)
                                raise SourceStageError(
                                    "existing target child directory changed: "
                                    + relative
                                )
                            identities[relative] = _identity(opened_child)
                            pending.append((child_fd, relative))
                        elif stat.S_ISREG(child.st_mode):
                            expected = expected_files.get(relative)
                            if expected is None or relative in visited_files:
                                raise SourceStageError(
                                    "existing target file closure differs: "
                                    + relative
                                )
                            descriptor = os.open(
                                entry.name,
                                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0)
                                | getattr(os, "O_NONBLOCK", 0),
                                dir_fd=directory_fd,
                            )
                            try:
                                opened = os.fstat(descriptor)
                                first = _read_fd(descriptor, opened.st_size)
                                middle = os.fstat(descriptor)
                                second = _read_fd(descriptor, opened.st_size)
                                after = os.fstat(descriptor)
                                named = os.stat(
                                    entry.name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                                if (
                                    _identity(child) != _identity(opened)
                                    or _identity(opened) != _identity(middle)
                                    or _identity(opened) != _identity(after)
                                    or _identity(opened) != _identity(named)
                                    or opened.st_nlink != 1
                                    or stat.S_IMODE(opened.st_mode) != FILE_MODE
                                    or opened.st_uid != REMOTE_UID
                                    or opened.st_gid != REMOTE_GID
                                    or opened.st_size != expected["size"]
                                    or first != second
                                    or first != captured[relative]
                                    or hashlib.sha256(first).hexdigest()
                                    != expected["sha256"]
                                ):
                                    raise SourceStageError(
                                        "existing target held file differs: "
                                        + relative
                                    )
                                identities[relative] = _identity(opened)
                                visited_files.add(relative)
                            finally:
                                os.close(descriptor)
                        else:
                            raise SourceStageError(
                                "existing target special entry: " + relative
                            )
                if _identity(os.fstat(directory_fd)) != identity:
                    raise SourceStageError(
                        "existing target held directory changed: " + prefix
                    )
            finally:
                os.close(directory_fd)
    except BaseException:
        for descriptor, _prefix in pending:
            os.close(descriptor)
        raise
    if visited_dirs != expected_dirs or visited_files != set(expected_files):
        raise SourceStageError("existing target exact-tree closure differs")
    tree = _read_tree_held(root_fd, identities, rows)
    return tree, identities


def _open_existing_receipt(parent_fd: int, name: str) -> HeldReceipt:
    """Open an existing receipt and retain the exact named inode."""
    name = _plain_basename(name, label="existing receipt")
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        first = _read_fd(descriptor, opened.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, opened.st_size)
        after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _identity(named) != _identity(opened)
            or _identity(opened) != _identity(middle)
            or _identity(opened) != _identity(after)
            or _identity(opened) != _identity(named_after)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != RECEIPT_MODE
            or opened.st_uid != REMOTE_UID
            or opened.st_gid != REMOTE_GID
            or first != second
        ):
            raise SourceStageError("existing receipt held identity differs")
        _strict_json(first, label="existing source staging receipt")
        return HeldReceipt(name, descriptor, _identity(opened), first)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_remote_runtime(manifest: Mapping[str, Any]) -> None:
    if not sys.platform.startswith("linux"):
        raise SourceStageError("remote bootstrap requires Linux")
    if os.geteuid() != REMOTE_UID or os.getegid() != REMOTE_GID:
        raise SourceStageError("remote bootstrap ownership differs")
    expected = manifest["remote_python"]
    descriptor = os.open(
        "/proc/self/exe", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        opened = os.fstat(descriptor)
        first = _read_fd(descriptor, opened.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, opened.st_size)
        after = os.fstat(descriptor)
        named = os.lstat(expected["path"])
    finally:
        os.close(descriptor)
    if (
        _identity(opened) != _identity(middle)
        or _identity(opened) != _identity(after)
        or _identity(opened) != _identity(named)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_size != expected["size"]
        or first != second
        or hashlib.sha256(first).hexdigest() != expected["sha256"]
        or Path("/proc/self/exe").resolve(strict=True) != Path(expected["path"])
    ):
        raise SourceStageError("remote Python authority differs")


def _remote_coordinates(
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    target = Path(manifest["target_root"])
    receipt_path = Path(manifest["receipt_path"])
    if (
        target.parent != receipt_path.parent
        or target.parent != REMOTE_PARENT
        or target != REMOTE_TARGET_ROOT
        or receipt_path != REMOTE_RECEIPT_PATH
    ):
        raise SourceStageError("remote target coordinates differ")
    return target, receipt_path


def _remote_stage(
    payload: Mapping[str, Any],
    captured: Mapping[str, bytes],
    specifications: Sequence[Mapping[str, Any]],
    *,
    payload_raw: bytes,
    claimed_payload_sha256: str,
    held_bootstrap_sha256: str,
) -> dict[str, Any]:
    if (
        payload["operation"] != STAGE_OPERATION
        or _payload_raw_for_operation(payload, STAGE_OPERATION) != payload_raw
    ):
        raise SourceStageError("stage payload operation differs")
    manifest = payload["manifest"]
    target, receipt_path = _remote_coordinates(manifest)
    descriptors, parent_fd, ancestor_anchors = _hold_directory_chain(
        target.parent,
    )
    shadow_name: str | None = None
    shadow_fd = -1
    root_anchor: tuple[int, ...] | None = None
    creation_anchors: dict[str, tuple[int, ...]] = {}
    sealed_identities: dict[str, tuple[int, ...]] = {}
    reservation: HeldReceiptReservation | None = None
    held_receipt: HeldReceipt | None = None
    shadow_owned = False
    target_committed = False
    cleanup_allowed = True
    rename_result = "not_attempted"
    rename_classification = "not_attempted"
    prepublish: dict[str, Any] | None = None
    expected_receipt_raw: bytes | None = None
    try:
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        _absent_at(parent_fd, target.name, label="target root")
        _absent_at(parent_fd, receipt_path.name, label="receipt")
        reservation = _reserve_receipt(
            parent_fd,
            receipt_path.name,
            manifest=manifest,
            request_payload_sha256=claimed_payload_sha256,
            bootstrap_sha256=held_bootstrap_sha256,
        )
        reservation.require_reserved(parent_fd)
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        shadow_name = "." + target.name + ".shadow-" + secrets.token_hex(16)
        _absent_at(parent_fd, shadow_name, label="random shadow")
        reserve_fd = _reserve_creation_descriptor(
            parent_fd, label="random shadow root creation",
        )
        try:
            os.mkdir(shadow_name, 0o700, dir_fd=parent_fd)
            os.close(reserve_fd)
            reserve_fd = -1
            try:
                shadow_fd, opened_root = _open_shadow_root(
                    parent_fd, shadow_name,
                )
            except BaseException as error:
                cleanup_allowed = False
                raise SourceStageError(
                    "random shadow held-open unavailable after mkdir; "
                    "unknown inode preserved for permanent HOLD"
                ) from error
        finally:
            if reserve_fd >= 0:
                os.close(reserve_fd)
        root_anchor = _inode_anchor(opened_root)
        creation_anchors["."] = root_anchor
        shadow_owned = True
        _assert_created_directory_named(
            parent_fd,
            shadow_name,
            shadow_fd,
            root_anchor,
            label="random shadow root creation",
        )
        sealed_identities = _build_shadow_at(
            shadow_fd, captured, creation_anchors, specifications,
        )
        _assert_named_inode(
            parent_fd,
            shadow_name,
            shadow_fd,
            root_anchor,
            expected_identity=sealed_identities["."],
            label="shadow root",
        )
        prepublish = _read_tree_held(
            shadow_fd, sealed_identities, specifications,
        )
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        reservation.require_reserved(parent_fd)
        try:
            _rename_under_receipt_reservation(
                parent_fd,
                shadow_name,
                target.name,
                shadow_fd,
                root_anchor,
                reservation,
            )
        except RenameOutcomeError as error:
            rename_result = "raised"
            rename_classification = error.classification
            if error.committed is True:
                target_committed = True
                shadow_owned = False
            elif error.committed is None:
                cleanup_allowed = False
            raise
        else:
            rename_result = "returned_success"
            rename_classification = "target_is_held_shadow"
            target_committed = True
            shadow_owned = False
        # Rename is the irreversible target commit point.  Record that state
        # before durability synchronization so an fsync error cannot enter
        # pre-commit cleanup or make recovery unreachable.
        os.fsync(parent_fd)
        _absent_at(parent_fd, shadow_name, label="published shadow")
        sealed_identities["."] = _assert_named_inode(
            parent_fd,
            target.name,
            shadow_fd,
            root_anchor,
            expected_identity=None,
            label="published target root",
        )
        published_tree = _read_tree_held(
            shadow_fd, sealed_identities, specifications,
        )
        if published_tree != prepublish:
            raise SourceStageError("published tree replay differs")
        receipt = _receipt_value(
            manifest=manifest,
            operation=STAGE_OPERATION,
            request_payload_sha256=claimed_payload_sha256,
            stage_payload_sha256=claimed_payload_sha256,
            bootstrap_sha256=held_bootstrap_sha256,
            tree=published_tree,
            target_identity=sealed_identities["."],
        )
        expected_receipt = _receipt_value(
            manifest=manifest,
            operation=STAGE_OPERATION,
            request_payload_sha256=claimed_payload_sha256,
            stage_payload_sha256=claimed_payload_sha256,
            bootstrap_sha256=held_bootstrap_sha256,
            tree=published_tree,
            target_identity=sealed_identities["."],
            receipt_inode_anchor=reservation.anchor,
        )
        expected_receipt_raw = canonical(expected_receipt) + b"\n"
        held_receipt = _seal_reserved_receipt(
            parent_fd, reservation, receipt,
        )
        receipt = _receipt_value(
            manifest=manifest,
            operation=STAGE_OPERATION,
            request_payload_sha256=claimed_payload_sha256,
            stage_payload_sha256=claimed_payload_sha256,
            bootstrap_sha256=held_bootstrap_sha256,
            tree=published_tree,
            target_identity=sealed_identities["."],
            receipt_inode_anchor=_inode_anchor_from_identity(
                held_receipt.identity
            ),
        )
        expected_raw = canonical(receipt) + b"\n"
        if (
            expected_raw != expected_receipt_raw
            or held_receipt.raw != expected_raw
        ):
            raise SourceStageError("receipt write bytes differ")
        replay_raw, replay = held_receipt.replay(parent_fd)
        if (
            replay_raw != expected_raw
            or replay != receipt
            or _assert_named_inode(
                parent_fd,
                target.name,
                shadow_fd,
                root_anchor,
                expected_identity=sealed_identities["."],
                label="published target root",
            ) != sealed_identities["."]
            or _read_tree_held(
                shadow_fd, sealed_identities, specifications,
            ) != published_tree
        ):
            raise SourceStageError("post-publication readback differs")
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        final_raw, final_receipt = held_receipt.replay(parent_fd)
        if final_raw != expected_raw or final_receipt != receipt:
            raise SourceStageError("final held receipt replay differs")
        return receipt
    except BaseException:
        if target_committed:
            committed_identity = _identity(os.fstat(shadow_fd))
            named_target_same_held_inode = False
            recovery_admissible = False
            reservation_state = _unavailable_receipt_reservation_state()
            receipt_phase = "unavailable"
            receipt_authoritative = False
            try:
                current_identities = dict(sealed_identities)
                current_identities["."] = committed_identity
                named_target_same_held_inode = (
                    prepublish is not None
                    and _assert_named_inode(
                        parent_fd,
                        target.name,
                        shadow_fd,
                        root_anchor,
                        expected_identity=committed_identity,
                        label="committed recovery target root",
                    ) == committed_identity
                    and _read_tree_held(
                        shadow_fd, current_identities, specifications,
                    )
                    == prepublish
                )
            except BaseException:
                named_target_same_held_inode = False
            try:
                if reservation is not None and reservation.descriptor >= 0:
                    receipt_raw, receipt_identity = reservation.snapshot(
                        parent_fd,
                    )
                elif held_receipt is not None and held_receipt.descriptor >= 0:
                    snapshot = HeldReceiptReservation(
                        held_receipt.name,
                        held_receipt.descriptor,
                        _inode_anchor_from_identity(held_receipt.identity),
                        None,
                    )
                    receipt_raw, receipt_identity = snapshot.snapshot(parent_fd)
                else:
                    raise SourceStageError(
                        "committed target lost held receipt reservation"
                    )
                reservation_state = _receipt_reservation_state_value(
                    receipt_raw, receipt_identity,
                )
                receipt_mode = stat.S_IMODE(receipt_identity[4])
                receipt_authoritative = (
                    receipt_mode == RECEIPT_MODE
                    and expected_receipt_raw is not None
                    and receipt_raw == expected_receipt_raw
                )
                if receipt_mode == RECEIPT_RESERVATION_MODE:
                    receipt_phase = (
                        "reserved_0600"
                        if reservation is not None
                        and reservation.reserved_raw is not None
                        and receipt_raw == reservation.reserved_raw
                        else "partial_0600"
                    )
                elif receipt_authoritative:
                    receipt_phase = "sealed_0400_exact"
                else:
                    receipt_phase = "sealed_0400_unverified"
            except BaseException:
                reservation_state = _unavailable_receipt_reservation_state()
                receipt_phase = "unavailable"
                receipt_authoritative = False
            recovery_admissible = (
                named_target_same_held_inode
                and reservation_state["available"] is True
                and (
                    receipt_phase in ("reserved_0600", "partial_0600")
                    or receipt_authoritative
                )
            )
            return _commit_terminal(
                manifest,
                request_payload_sha256=claimed_payload_sha256,
                bootstrap_sha256=held_bootstrap_sha256,
                target_identity=committed_identity,
                receipt_reservation_state=reservation_state,
                rename_result=rename_result,
                rename_classification=rename_classification,
                receipt_phase=receipt_phase,
                receipt_authoritative=receipt_authoritative,
                named_target_same_held_inode=named_target_same_held_inode,
                recovery_admissible=recovery_admissible,
            )
        raise
    finally:
        try:
            if held_receipt is not None:
                held_receipt.close()
            if (
                shadow_name is not None
                and shadow_owned
                and root_anchor is not None
            ):
                if shadow_fd < 0:
                    raise SourceStageError(
                        "owned shadow lacks held root; preserved for permanent HOLD"
                    )
                if not cleanup_allowed:
                    raise SourceStageError(
                        "precommit namespace is ambiguous; owned names preserved "
                        "for permanent HOLD"
                    )
                _absent_at(
                    parent_fd, target.name,
                    label="precommit cleanup target root",
                )
                _discard_shadow_held(
                    parent_fd,
                    shadow_name,
                    shadow_fd,
                    root_anchor,
                    creation_anchors,
                    validate_only=True,
                )
                if reservation is None or reservation.descriptor < 0:
                    raise SourceStageError(
                        "precommit cleanup receipt reservation is unavailable"
                    )
                _discard_receipt_reservation(
                    parent_fd, reservation, validate_only=True,
                )
                _absent_at(
                    parent_fd, target.name,
                    label="precommit cleanup target root replay",
                )
                _discard_shadow_held(
                    parent_fd,
                    shadow_name,
                    shadow_fd,
                    root_anchor,
                    creation_anchors,
                )
                _discard_receipt_reservation(parent_fd, reservation)
            elif (
                not target_committed
                and reservation is not None
                and reservation.descriptor >= 0
            ):
                if not cleanup_allowed:
                    raise SourceStageError(
                        "unknown precommit inode preserved for permanent HOLD"
                    )
                _absent_at(
                    parent_fd, target.name,
                    label="receipt-only cleanup target root",
                )
                _discard_receipt_reservation(
                    parent_fd, reservation, validate_only=True,
                )
                _discard_receipt_reservation(parent_fd, reservation)
        finally:
            try:
                if reservation is not None:
                    reservation.close()
                if shadow_fd >= 0:
                    os.close(shadow_fd)
            finally:
                for descriptor in reversed(descriptors):
                    os.close(descriptor)


def _remote_recover_receipt(
    payload: Mapping[str, Any],
    captured: Mapping[str, bytes],
    specifications: Sequence[Mapping[str, Any]],
    *,
    payload_raw: bytes,
    claimed_payload_sha256: str,
    held_bootstrap_sha256: str,
) -> dict[str, Any]:
    if (
        payload["operation"] != RECOVER_RECEIPT_OPERATION
        or _payload_raw_for_operation(payload, RECOVER_RECEIPT_OPERATION)
        != payload_raw
    ):
        raise SourceStageError("receipt-recovery payload operation differs")
    manifest = payload["manifest"]
    commit_terminal = _validate_commit_terminal_for_payload(payload)
    if commit_terminal is None:
        raise SourceStageError("receipt-recovery commit terminal is absent")
    expected_target_identity = _validated_target_full_identity(
        commit_terminal["target_root_identity"]
    )
    target, receipt_path = _remote_coordinates(manifest)
    stage_payload_raw = _payload_raw_for_operation(payload, STAGE_OPERATION)
    stage_payload_sha256 = hashlib.sha256(stage_payload_raw).hexdigest()
    descriptors, parent_fd, ancestor_anchors = _hold_directory_chain(target.parent)
    target_fd = -1
    target_anchor: tuple[int, ...] | None = None
    target_identity: tuple[int, ...] | None = None
    reservation: HeldReceiptReservation | None = None
    held_receipt: HeldReceipt | None = None
    rewrite_started = False
    tree: dict[str, Any] | None = None
    sealed_identities: dict[str, tuple[int, ...]] = {}
    expected_recovered_raw: bytes | None = None
    try:
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        target_fd, target_anchor, target_identity = _open_existing_target(
            parent_fd, target.name,
        )
        if target_identity != expected_target_identity:
            raise SourceStageError(
                "receipt-recovery target differs from commit terminal"
            )
        tree, sealed_identities = _capture_existing_tree_held(
            target_fd, captured, specifications,
        )
        if (
            sealed_identities["."] != target_identity
            or _assert_named_inode(
                parent_fd,
                target.name,
                target_fd,
                target_anchor,
                expected_identity=target_identity,
                label="recovery target root",
            ) != target_identity
        ):
            raise SourceStageError("receipt-recovery target root replay differs")
        def expected_receipt(
            receipt_operation: str,
            receipt_anchor: tuple[int, ...] | None,
        ) -> dict[str, Any]:
            return _receipt_value(
                manifest=manifest,
                operation=receipt_operation,
                request_payload_sha256=(
                    stage_payload_sha256
                    if receipt_operation == STAGE_OPERATION
                    else claimed_payload_sha256
                ),
                stage_payload_sha256=stage_payload_sha256,
                bootstrap_sha256=held_bootstrap_sha256,
                tree=tree,
                target_identity=target_identity,
                receipt_inode_anchor=receipt_anchor,
                commit_terminal_digest=(
                    None if receipt_operation == STAGE_OPERATION
                    else commit_terminal["terminal_digest"]
                ),
            )
        terminal_receipt_state = _validated_receipt_reservation_state(
            commit_terminal["receipt_reservation_state"]
        )
        reservation = _open_receipt_reservation_for_recovery(
            parent_fd, receipt_path.name, terminal_receipt_state,
        )
        receipt_raw, receipt_identity = reservation.snapshot(parent_fd)
        receipt_anchor = _inode_anchor_from_identity(receipt_identity)
        live_receipt = expected_receipt(STAGE_OPERATION, receipt_anchor)
        recovered_receipt = expected_receipt(
            RECOVER_RECEIPT_OPERATION, receipt_anchor,
        )
        expected_live_raw = canonical(live_receipt) + b"\n"
        expected_recovered_raw = canonical(recovered_receipt) + b"\n"
        receipt_mode = stat.S_IMODE(receipt_identity[4])
        if receipt_mode == RECEIPT_MODE:
            # 0400 is the admission commit.  It is strictly read-only here:
            # malformed or unbound bytes are never truncated or rewritten.
            if receipt_raw == expected_live_raw:
                receipt = live_receipt
            elif receipt_raw == expected_recovered_raw:
                receipt = recovered_receipt
            else:
                raise SourceStageError(
                    "sealed receipt is not exact for held target"
                )
            held_receipt = HeldReceipt(
                reservation.name,
                reservation.detach(),
                receipt_identity,
                receipt_raw,
            )
            replay_raw, replay = held_receipt.replay(parent_fd)
            if replay_raw != receipt_raw or replay != receipt:
                raise SourceStageError("sealed receipt verify-only replay differs")
        elif receipt_mode == RECEIPT_RESERVATION_MODE:
            recovered_template = expected_receipt(
                RECOVER_RECEIPT_OPERATION, None,
            )
            # From this point the held 0600 inode may be mutated.  Any error
            # must publish an updated terminal that binds its exact new state;
            # the immutable prior terminal can no longer authorize a retry.
            rewrite_started = True
            held_receipt = _seal_reserved_receipt(
                parent_fd,
                reservation,
                recovered_template,
                expected_prior_state=terminal_receipt_state,
            )
            receipt = recovered_receipt
            receipt_raw, replay = held_receipt.replay(parent_fd)
            if receipt_raw != expected_recovered_raw or replay != receipt:
                raise SourceStageError("recovered receipt write replay differs")
        else:
            raise SourceStageError("receipt recovery mode is not admitted")
        if (
            _assert_named_inode(
                parent_fd,
                target.name,
                target_fd,
                target_anchor,
                expected_identity=target_identity,
                label="recovery target root",
            ) != target_identity
            or _read_tree_held(
                target_fd, sealed_identities, specifications,
            ) != tree
        ):
            raise SourceStageError("receipt-recovery post-write target differs")
        _replay_directory_chain(target.parent, descriptors, ancestor_anchors)
        final_raw, final_receipt = held_receipt.replay(parent_fd)
        if final_raw != receipt_raw or final_receipt != receipt:
            raise SourceStageError("receipt-recovery final held receipt differs")
        if (
            _assert_named_inode(
                parent_fd,
                target.name,
                target_fd,
                target_anchor,
                expected_identity=target_identity,
                label="recovery target root",
            ) != target_identity
            or _read_tree_held(
                target_fd, sealed_identities, specifications,
            ) != tree
        ):
            raise SourceStageError("receipt-recovery final target differs")
        return receipt
    except BaseException:
        if not rewrite_started:
            raise
        named_target_same_held_inode = False
        reservation_state = _unavailable_receipt_reservation_state()
        receipt_phase = "unavailable"
        receipt_authoritative = False
        try:
            if (
                target_fd < 0
                or target_anchor is None
                or target_identity is None
                or tree is None
            ):
                raise SourceStageError(
                    "receipt-recovery mutation lost target authority"
                )
            named_target_same_held_inode = (
                _assert_named_inode(
                    parent_fd,
                    target.name,
                    target_fd,
                    target_anchor,
                    expected_identity=target_identity,
                    label="mutated-recovery target root",
                ) == target_identity
                and _read_tree_held(
                    target_fd, sealed_identities, specifications,
                ) == tree
            )
        except BaseException:
            named_target_same_held_inode = False
        try:
            if reservation is not None and reservation.descriptor >= 0:
                receipt_raw, receipt_identity = reservation.snapshot(parent_fd)
            elif held_receipt is not None and held_receipt.descriptor >= 0:
                snapshot = HeldReceiptReservation(
                    held_receipt.name,
                    held_receipt.descriptor,
                    _inode_anchor_from_identity(held_receipt.identity),
                    None,
                )
                receipt_raw, receipt_identity = snapshot.snapshot(parent_fd)
            else:
                raise SourceStageError(
                    "receipt-recovery mutation lost held receipt inode"
                )
            reservation_state = _receipt_reservation_state_value(
                receipt_raw, receipt_identity,
            )
            receipt_mode = stat.S_IMODE(receipt_identity[4])
            receipt_authoritative = (
                receipt_mode == RECEIPT_MODE
                and expected_recovered_raw is not None
                and receipt_raw == expected_recovered_raw
            )
            if receipt_mode == RECEIPT_RESERVATION_MODE:
                receipt_phase = (
                    "reserved_0600"
                    if reservation is not None
                    and reservation.reserved_raw is not None
                    and receipt_raw == reservation.reserved_raw
                    else "partial_0600"
                )
            elif receipt_authoritative:
                receipt_phase = "sealed_0400_exact"
            else:
                receipt_phase = "sealed_0400_unverified"
        except BaseException:
            reservation_state = _unavailable_receipt_reservation_state()
            receipt_phase = "unavailable"
            receipt_authoritative = False
        recovery_admissible = (
            named_target_same_held_inode
            and reservation_state["available"] is True
            and (
                receipt_phase in ("reserved_0600", "partial_0600")
                or receipt_authoritative
            )
        )
        if receipt_authoritative and named_target_same_held_inode:
            # The 0400 inode is already the admission commit and its bytes are
            # bound to the input terminal.  A later replay/check error cannot
            # be represented by a new terminal digest (that would change the
            # receipt binding).  Independently replay the held authorities and
            # return the exact recovered receipt as success when they agree.
            try:
                _replay_directory_chain(
                    target.parent, descriptors, ancestor_anchors,
                )
                if reservation is not None and reservation.descriptor >= 0:
                    final_raw, final_identity = reservation.snapshot(parent_fd)
                elif held_receipt is not None and held_receipt.descriptor >= 0:
                    final_snapshot = HeldReceiptReservation(
                        held_receipt.name,
                        held_receipt.descriptor,
                        _inode_anchor_from_identity(held_receipt.identity),
                        None,
                    )
                    final_raw, final_identity = final_snapshot.snapshot(parent_fd)
                else:
                    raise SourceStageError(
                        "sealed recovery receipt lost held inode"
                    )
                if (
                    final_raw != expected_recovered_raw
                    or final_identity != receipt_identity
                    or target_identity is None
                    or tree is None
                    or _assert_named_inode(
                        parent_fd,
                        target.name,
                        target_fd,
                        target_anchor,
                        expected_identity=target_identity,
                        label="sealed-recovery target root",
                    ) != target_identity
                    or _read_tree_held(
                        target_fd, sealed_identities, specifications,
                    ) != tree
                ):
                    raise SourceStageError(
                        "sealed recovery independent replay differs"
                    )
                return recovered_receipt
            except BaseException:
                # 0400 can never be rewritten.  If the independent audit also
                # fails, emit a non-admissible terminal for manual HOLD only.
                receipt_authoritative = False
                receipt_phase = "sealed_0400_unverified"
                recovery_admissible = False
        return _commit_terminal(
            manifest,
            request_payload_sha256=stage_payload_sha256,
            bootstrap_sha256=held_bootstrap_sha256,
            target_identity=(
                expected_target_identity
                if target_identity is None else target_identity
            ),
            receipt_reservation_state=reservation_state,
            rename_result=commit_terminal["rename_result"],
            rename_classification=commit_terminal["rename_classification"],
            receipt_phase=receipt_phase,
            receipt_authoritative=receipt_authoritative,
            named_target_same_held_inode=named_target_same_held_inode,
            recovery_admissible=recovery_admissible,
        )
    finally:
        try:
            if held_receipt is not None:
                held_receipt.close()
            if reservation is not None:
                reservation.close()
        finally:
            try:
                if target_fd >= 0:
                    os.close(target_fd)
            finally:
                for descriptor in reversed(descriptors):
                    os.close(descriptor)


def _remote_dispatch(
    payload_raw: bytes,
    claimed_payload_sha256: str,
    held_bootstrap_sha256: str,
) -> dict[str, Any]:
    payload, captured, specifications = _parse_payload(
        payload_raw,
        claimed_sha256=claimed_payload_sha256,
        held_bootstrap_sha256=held_bootstrap_sha256,
    )
    _validate_remote_runtime(payload["manifest"])
    if payload["operation"] == STAGE_OPERATION:
        return _remote_stage(
            payload,
            captured,
            specifications,
            payload_raw=payload_raw,
            claimed_payload_sha256=claimed_payload_sha256,
            held_bootstrap_sha256=held_bootstrap_sha256,
        )
    if payload["operation"] == RECOVER_RECEIPT_OPERATION:
        return _remote_recover_receipt(
            payload,
            captured,
            specifications,
            payload_raw=payload_raw,
            claimed_payload_sha256=claimed_payload_sha256,
            held_bootstrap_sha256=held_bootstrap_sha256,
        )
    raise SourceStageError("remote operation differs")


def _remote_bootstrap(
    payload_raw: bytes,
    claimed_payload_sha256: str,
    held_bootstrap_sha256: str,
) -> dict[str, Any]:
    """Compatibility name for tests; the frozen loader calls dispatch."""
    return _remote_dispatch(
        payload_raw, claimed_payload_sha256, held_bootstrap_sha256,
    )



# The held loader calls _remote_dispatch directly; this authority has no CLI.
