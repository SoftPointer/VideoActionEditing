#!/usr/bin/env python3
"""Publish four exact local executor commands without executing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_plan_v1 as plan


SCHEMA = "bernini-action-preservation-decoded-eval-local-launch-manifest-v7"
FILENAME = "local_launch_manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
DEPLOYMENT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-deployment-authority-binding-v1"
)
WORK_ROOT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-work-root-authority-v1"
)
EXECUTOR_CAPTURE_BASENAME_PREFIX = "executor-"
EXECUTOR_CAPTURE_BASENAME_SUFFIX = "-verified-runtime-capture.json"
LAUNCHER_WORK_ROOT_PROJECTION_SCHEMA = (
    "bernini-action-preservation-launcher-work-root-projection-v1"
)
HOLDER_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-holder-completion-anchor-v1"
)
HOLDER_COMPLETION_ANCHOR_CHANNEL_SCHEMA = (
    "bernini-action-preservation-holder-completion-anchor-channel-v1"
)
HOLDER_COMPLETION_DYNAMIC_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-holder-completion-dynamic-authority-v1"
)
AGGREGATE_COMMAND_PLAN_SCHEMA = (
    "bernini-action-preservation-decoded-eval-aggregate-command-plan-v1"
)
AGGREGATE_COMPLETION_ANCHOR_SCHEMA = (
    "bernini-action-preservation-aggregate-completion-anchor-v1"
)
AGGREGATE_COMPLETION_ANCHOR_CHANNEL_SCHEMA = (
    "bernini-action-preservation-aggregate-completion-anchor-channel-v1"
)
AGGREGATE_COMPLETION_ANCHOR_FIELDS = (
    "schema_version", "evaluation_id", "aggregate_root",
    "aggregate_root_identity", "aggregate_file", "private_file",
    "public_file", "media_directory_identity", "media_file_count",
    "media_rows_digest", "media_tree_digest", "anchor_digest",
)
AGGREGATE_TARGET = "action_preservation_decoded_eval_aggregate_v2.py"
AGGREGATE_CAPTURE_BASENAME = "aggregate-verified-runtime-capture.json"
LAUNCH_MANIFEST_ANCHOR_SCHEMA = (
    "bernini-action-preservation-launch-manifest-anchor-v1"
)
LAUNCH_MANIFEST_ANCHOR_CHANNEL_SCHEMA = (
    "bernini-action-preservation-launch-manifest-anchor-channel-v1"
)
LAUNCH_MANIFEST_ANCHOR_FIELDS = (
    "schema_version", "path", "sha256", "size", "mode", "identity",
    "launch_manifest_digest", "anchor_digest",
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


class DecodedEvaluationLauncherError(RuntimeError):
    pass


def _publication_barrier(_stage: str) -> None:
    """Mark a publication transaction boundary.

    The production implementation deliberately does nothing.  Keeping the
    boundary explicit lets the hostile tests exchange names at deterministic
    points instead of relying on scheduler timing.
    """


def _identity(value: os.stat_result) -> tuple[int, ...]:
    """Return the stable, non-atime physical identity of a filesystem node."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_rdev),
        int(value.st_size),
        int(getattr(value, "st_blocks", 0)),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
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


def _open_directory(path: Path, *, label: str) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise DecodedEvaluationLauncherError(
            "safe launch publication requires O_DIRECTORY and O_NOFOLLOW"
        )
    try:
        return os.open(
            path,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"cannot open {label}: {error}"
        ) from error


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise DecodedEvaluationLauncherError(
            "safe launch publication requires O_DIRECTORY and O_NOFOLLOW"
        )
    try:
        return os.open(
            name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"cannot open {label}: {error}"
        ) from error


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _duplicate_noninheritable_pair(
    first: int, second: int, *, label: str,
) -> tuple[int, int]:
    first_copy: int | None = None
    second_copy: int | None = None
    try:
        first_copy = os.dup(first)
        second_copy = os.dup(second)
        os.set_inheritable(first_copy, False)
        os.set_inheritable(second_copy, False)
        return first_copy, second_copy
    except OSError as error:
        if second_copy is not None:
            os.close(second_copy)
        if first_copy is not None:
            os.close(first_copy)
        raise DecodedEvaluationLauncherError(
            f"cannot retain {label} descriptors: {error}"
        ) from error


def _require_canonical_parent(path: Path, descriptor: int) -> tuple[int, int]:
    try:
        resolved = path.resolve(strict=True)
        named = path.lstat()
        held = os.fstat(descriptor)
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"launch root parent identity cannot be replayed: {error}"
        ) from error
    if (
        resolved != path
        or not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _identity(named) != _identity(held)
    ):
        raise DecodedEvaluationLauncherError(
            "launch root parent canonical named identity differs"
        )
    return int(held.st_dev), int(held.st_ino)


def _replay_canonical_parent(
    path: Path, descriptor: int, expected: tuple[int, int]
) -> None:
    observed = _require_canonical_parent(path, descriptor)
    if observed != expected:
        raise DecodedEvaluationLauncherError(
            "launch root parent held identity differs"
        )


def _immutable_directory_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": value.st_mode, "rdev": value.st_rdev,
    }


def _retained_work_root_snapshot(
    *, work_root: Path, root_descriptor: int, parent_descriptor: int,
    expected_root_immutable: Mapping[str, int],
    expected_parent_immutable: Mapping[str, int],
    expected_identity: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    try:
        held = os.fstat(root_descriptor)
        named = os.stat(
            work_root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        held_parent = os.fstat(parent_descriptor)
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"signed work-root identity cannot be replayed: {error}"
        ) from error
    observed = _identity(held)
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or observed != _identity(named)
        or (expected_identity is not None and observed != expected_identity)
        or _immutable_directory_row(held) != dict(expected_root_immutable)
        or _immutable_directory_row(held_parent)
        != dict(expected_parent_immutable)
    ):
        raise DecodedEvaluationLauncherError(
            "signed work-root held/named authority differs"
        )
    return observed


def _directory_snapshot(
    descriptor: int, *, label: str, expected_mode: int
) -> tuple[int, ...]:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_nlink < 1
        or stat.S_IMODE(info.st_mode) != expected_mode
    ):
        raise DecodedEvaluationLauncherError(
            f"{label} held identity or mode differs"
        )
    return _identity(info)


def _replay_directory_at(
    *,
    parent_descriptor: int,
    name: str,
    descriptor: int,
    expected: tuple[int, ...],
    label: str,
) -> None:
    try:
        held = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"{label} named identity cannot be replayed: {error}"
        ) from error
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _identity(held) != expected
        or _identity(named) != expected
    ):
        raise DecodedEvaluationLauncherError(
            f"{label} held/named identity differs"
        )


def _file_snapshot(descriptor: int, *, payload_size: int) -> tuple[int, ...]:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_size != payload_size
    ):
        raise DecodedEvaluationLauncherError(
            "launch manifest held identity, mode, size, or link count differs"
        )
    return _identity(info)


def _replay_file_at(
    *,
    directory_descriptor: int,
    descriptor: int,
    expected: tuple[int, ...],
    payload_size: int,
) -> None:
    try:
        held = os.fstat(descriptor)
        named = os.stat(
            FILENAME, dir_fd=directory_descriptor, follow_symlinks=False
        )
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"launch manifest named identity cannot be replayed: {error}"
        ) from error
    if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or held.st_nlink != 1
        or named.st_nlink != 1
        or stat.S_IMODE(held.st_mode) != 0o400
        or held.st_size != payload_size
        or _identity(held) != expected
        or _identity(named) != expected
    ):
        raise DecodedEvaluationLauncherError(
            "launch manifest held/named identity or hard-link closure differs"
        )


def _relative_entry_absent(
    directory_descriptor: int, name: str, *, label: str
) -> None:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"cannot establish fresh {label}: {error}"
        ) from error
    raise DecodedEvaluationLauncherError(f"{label} is not fresh")


def _exact_entries(
    descriptor: int, expected: set[str], *, label: str
) -> None:
    try:
        first = tuple(sorted(os.listdir(descriptor)))
        second = tuple(sorted(os.listdir(descriptor)))
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"cannot list {label}: {error}"
        ) from error
    if first != second or set(first) != expected:
        raise DecodedEvaluationLauncherError(f"{label} exact entry closure differs")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return plan.canonical_json_bytes(value)
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise DecodedEvaluationLauncherError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationLauncherError(f"{label} is not a lowercase SHA-256")
    return value


def _canonical_absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise DecodedEvaluationLauncherError(f"{label} path differs")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) == os.path.sep
        or os.path.normpath(str(path)) != str(path)
    ):
        raise DecodedEvaluationLauncherError(f"{label} path differs")
    return path


def _identity_row(
    value: Any, *, fields: set[str], label: str,
) -> dict[str, int]:
    row = dict(_closed(value, fields, label=label))
    if any(type(row[field]) is not int or row[field] < 0 for field in fields):
        raise DecodedEvaluationLauncherError(f"{label} value differs")
    return row


def _unique_json_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_canonical_json_literal(value: Any, *, label: str) -> dict[str, Any]:
    if (
        type(value) is not str
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise DecodedEvaluationLauncherError(f"{label} literal differs")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvaluationLauncherError(
            f"{label} is not strict JSON"
        ) from error
    if (
        type(decoded) is not dict
        or canonical_json_bytes(decoded).decode("utf-8") != value
    ):
        raise DecodedEvaluationLauncherError(f"{label} is not canonical")
    return decoded


def _materialized_publication_receipt(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        raise DecodedEvaluationLauncherError("evaluation bundle differs")
    try:
        receipt = plan.validate_publication_receipt(
            bundle["publication_receipt"],
            bundle=bundle,
            directory_authority=bundle["directory_authority"],
            verify_directory_authority=True,
        )
    except (KeyError, plan.DecodedEvaluationPlanError) as error:
        raise DecodedEvaluationLauncherError(
            "materialized evaluation publication differs"
        ) from error
    if receipt["directory_authority_materialized"] is not True:
        raise DecodedEvaluationLauncherError(
            "materialized evaluation publication is required"
        )
    return receipt


def validate_holder_completion_anchor(
    value: Any,
    *,
    bundle: Mapping[str, Any],
    expected_holder_job_id: str | None = None,
) -> dict[str, Any]:
    """Validate one online completion anchor against its reserved inode.

    The materialized publication receipt is the immutable source for the
    initial inode identity and exact completion pathname.  The anchor itself
    authenticates the final bytes observed by the controller after the holder
    process has exited.
    """

    fields = {
        "schema_version", "holder_job_id", "completion_path",
        "initial_inode_identity", "completion_sha256", "completion_size",
        "completion_mode", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    }
    row = dict(_closed(value, fields, label="holder completion anchor"))
    identity_fields = {"device", "inode", "uid", "gid", "rdev"}
    initial = _identity_row(
        row["initial_inode_identity"],
        fields=identity_fields,
        label="holder completion initial inode",
    )
    holder_ids = [item["job_id"] for item in plan.HOLDER_ROWS]
    holder_job_id = row["holder_job_id"]
    if (
        row["schema_version"] != HOLDER_COMPLETION_ANCHOR_SCHEMA
        or holder_job_id not in holder_ids
        or (
            expected_holder_job_id is not None
            and holder_job_id != expected_holder_job_id
        )
        or type(row["completion_size"]) is not int
        or row["completion_size"] <= 0
        or row["completion_mode"]
        != plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE
    ):
        raise DecodedEvaluationLauncherError(
            "holder completion anchor binding differs"
        )
    completion_path = _canonical_absolute(
        row["completion_path"], label="holder completion"
    )
    for field in (
        "completion_sha256", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    ):
        _sha(row[field], label=f"holder completion anchor {field}")
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationLauncherError(
            "holder completion anchor digest differs"
        )

    receipt = _materialized_publication_receipt(bundle)
    reservations = receipt["holder_completion_reservations"]
    matches = [
        item for item in reservations
        if item["holder_job_id"] == holder_job_id
    ]
    if len(matches) != 1:
        raise DecodedEvaluationLauncherError(
            "holder completion reservation differs"
        )
    reservation = matches[0]
    expected_initial = {
        field: reservation["identity"][field]
        for field in identity_fields
    }
    if (
        str(completion_path) != reservation["path"]
        or initial != expected_initial
    ):
        raise DecodedEvaluationLauncherError(
            "holder completion anchor reservation binding differs"
        )
    return row


def parse_holder_completion_anchor_literal(
    value: Any,
    *,
    bundle: Mapping[str, Any],
    expected_holder_job_id: str | None = None,
) -> dict[str, Any]:
    decoded = _decode_canonical_json_literal(
        value, label="holder completion anchor"
    )
    return validate_holder_completion_anchor(
        decoded,
        bundle=bundle,
        expected_holder_job_id=expected_holder_job_id,
    )


def parse_holder_completion_anchor_stdout(
    stdout: Any,
    *,
    return_code: Any,
    bundle: Mapping[str, Any],
    expected_holder_job_id: str,
) -> dict[str, Any]:
    """Accept only controller success plus one complete canonical line.

    A same-UID process may append to the controller's stdout pipe, but it
    cannot remove the controller's mandatory success line or forge its exit
    status.  Requiring exact-one line, EOF (represented by the complete bytes
    argument), and exit zero turns injection into fail-closed denial of
    service rather than forged authority.
    """

    if type(return_code) is not int or return_code != 0 or type(stdout) is not bytes:
        raise DecodedEvaluationLauncherError(
            "holder controller did not exit successfully with bytes stdout"
        )
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise DecodedEvaluationLauncherError(
            "holder controller stdout is not exactly one complete line"
        )
    try:
        literal = stdout[:-1].decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DecodedEvaluationLauncherError(
            "holder controller stdout is not UTF-8"
        ) from error
    return parse_holder_completion_anchor_literal(
        literal,
        bundle=bundle,
        expected_holder_job_id=expected_holder_job_id,
    )


def build_holder_completion_dynamic_authority(
    anchor_literals: Sequence[str],
    *,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        isinstance(anchor_literals, (str, bytes, bytearray))
        or not isinstance(anchor_literals, Sequence)
        or len(anchor_literals) != len(plan.HOLDER_ROWS)
    ):
        raise DecodedEvaluationLauncherError(
            "holder completion anchor literal closure differs"
        )
    parsed: dict[str, dict[str, Any]] = {}
    for literal in anchor_literals:
        anchor = parse_holder_completion_anchor_literal(
            literal, bundle=bundle
        )
        holder_job_id = anchor["holder_job_id"]
        if holder_job_id in parsed:
            raise DecodedEvaluationLauncherError(
                "holder completion anchor holder is duplicated"
            )
        parsed[holder_job_id] = anchor
    holder_ids = [item["job_id"] for item in plan.HOLDER_ROWS]
    if set(parsed) != set(holder_ids):
        raise DecodedEvaluationLauncherError(
            "holder completion anchor holder closure differs"
        )
    receipt = _materialized_publication_receipt(bundle)
    value = {
        "schema_version": HOLDER_COMPLETION_DYNAMIC_AUTHORITY_SCHEMA,
        "evaluation_root": bundle["manifest"]["evaluation_root"],
        "publication_digest": receipt["publication_digest"],
        "anchors": [parsed[holder] for holder in holder_ids],
        "anchor_count": len(holder_ids),
    }
    value["authority_digest"] = object_sha256(value)
    return validate_holder_completion_dynamic_authority(value, bundle=bundle)


def validate_holder_completion_dynamic_authority(
    value: Any, *, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_root", "publication_digest",
        "anchors", "anchor_count", "authority_digest",
    }
    row = dict(
        _closed(value, fields, label="holder completion dynamic authority")
    )
    receipt = _materialized_publication_receipt(bundle)
    holder_ids = [item["job_id"] for item in plan.HOLDER_ROWS]
    if (
        row["schema_version"] != HOLDER_COMPLETION_DYNAMIC_AUTHORITY_SCHEMA
        or row["evaluation_root"] != bundle["manifest"]["evaluation_root"]
        or row["publication_digest"] != receipt["publication_digest"]
        or row["anchor_count"] != len(holder_ids)
        or not isinstance(row["anchors"], list)
        or len(row["anchors"]) != len(holder_ids)
    ):
        raise DecodedEvaluationLauncherError(
            "holder completion dynamic authority binding differs"
        )
    anchors = [
        validate_holder_completion_anchor(
            raw, bundle=bundle, expected_holder_job_id=holder_job_id
        )
        for raw, holder_job_id in zip(row["anchors"], holder_ids)
    ]
    if row["anchors"] != anchors:
        raise DecodedEvaluationLauncherError(
            "holder completion dynamic authority anchor differs"
        )
    claimed = _sha(
        row["authority_digest"],
        label="holder completion dynamic authority digest",
    )
    unsigned = dict(row)
    unsigned.pop("authority_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationLauncherError(
            "holder completion dynamic authority digest differs"
        )
    return row


def holder_completion_anchor_arguments(
    authority: Mapping[str, Any], *, bundle: Mapping[str, Any]
) -> list[str]:
    row = validate_holder_completion_dynamic_authority(
        authority, bundle=bundle
    )
    arguments: list[str] = []
    for anchor in row["anchors"]:
        arguments.extend(
            [
                "--holder-completion-anchor",
                canonical_json_bytes(anchor).decode("utf-8"),
            ]
        )
    return arguments


def _aggregate_completion_anchor_channel() -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_COMPLETION_ANCHOR_CHANNEL_SCHEMA,
        "payload_schema_version": AGGREGATE_COMPLETION_ANCHOR_SCHEMA,
        "payload_exact_fields": list(AGGREGATE_COMPLETION_ANCHOR_FIELDS),
        "required_on_success": True,
        "success_exit_code": 0,
        "stdout_exact_line_count": 1,
        "stdout_serialization": "canonical-json-newline-v1",
        "complete_stdout_and_eof_required": True,
        "extra_duplicate_or_partial_stdout_rejected": True,
        "external_trusted_orchestrator_required": True,
        "retain_in_memory_for_gate_authority": True,
        "disk_authority_forbidden": True,
    }


def validate_aggregate_completion_anchor(
    value: Any,
    *,
    launch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            set(AGGREGATE_COMPLETION_ANCHOR_FIELDS),
            label="aggregate completion anchor",
        )
    )
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }

    def identity(raw: Any, *, label: str, directory: bool) -> dict[str, int]:
        observed = _identity_row(raw, fields=identity_fields, label=label)
        if (
            (directory and not stat.S_ISDIR(observed["mode"]))
            or (not directory and not stat.S_ISREG(observed["mode"]))
            or observed["nlink"] < 1
        ):
            raise DecodedEvaluationLauncherError(f"{label} type differs")
        return observed

    launch = dict(launch_manifest)
    aggregate_root = _canonical_absolute(
        row["aggregate_root"], label="aggregate completion root"
    )
    root_identity = identity(
        row["aggregate_root_identity"],
        label="aggregate completion root identity",
        directory=True,
    )
    media_identity = identity(
        row["media_directory_identity"],
        label="aggregate completion media identity",
        directory=True,
    )
    source_ids = getattr(plan, "SOURCE_IDS", plan.FITTED_IIDS)
    maximum_media_file_count = plan.TOTAL_DECODE_COUNT + len(source_ids)
    if (
        row["schema_version"] != AGGREGATE_COMPLETION_ANCHOR_SCHEMA
        or row["evaluation_id"] != launch.get("evaluation_id")
        or str(aggregate_root) != launch.get("aggregate_root")
        or stat.S_IMODE(root_identity["mode"]) != 0o555
        or stat.S_IMODE(media_identity["mode"]) != 0o555
        or type(row["media_file_count"]) is not int
        or row["media_file_count"] <= 0
        or row["media_file_count"] > maximum_media_file_count
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate completion anchor binding differs"
        )
    expected_files = (
        ("aggregate_file", "evaluation_complete.json", 0o444, "aggregate_digest"),
        ("private_file", "private_blind_mapping.json", 0o400, "private_mapping_digest"),
        ("public_file", "blind_review_packet.json", 0o444, "public_packet_digest"),
    )
    for key, relative, mode, _object_field in expected_files:
        file_row = dict(
            _closed(
                row[key],
                {"relative_path", "sha256", "size", "mode", "identity", "object_digest"},
                label=f"aggregate completion {key}",
            )
        )
        file_identity = identity(
            file_row["identity"],
            label=f"aggregate completion {key} identity",
            directory=False,
        )
        _sha(file_row["sha256"], label=f"aggregate completion {key} SHA")
        _sha(
            file_row["object_digest"],
            label=f"aggregate completion {key} object digest",
        )
        if (
            file_row["relative_path"] != relative
            or type(file_row["size"]) is not int
            or file_row["size"] <= 0
            or file_row["mode"] != mode
            or stat.S_IMODE(file_identity["mode"]) != mode
            or file_identity["nlink"] != 1
            or file_identity["size"] != file_row["size"]
        ):
            raise DecodedEvaluationLauncherError(
                f"aggregate completion {key} binding differs"
            )
    for field in ("media_rows_digest", "media_tree_digest", "anchor_digest"):
        _sha(row[field], label=f"aggregate completion {field}")
    expected_media_tree = object_sha256(
        {
            "media_directory_identity": row["media_directory_identity"],
            "media_file_count": row["media_file_count"],
            "media_rows_digest": row["media_rows_digest"],
        }
    )
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest")
    if (
        row["media_tree_digest"] != expected_media_tree
        or claimed != object_sha256(unsigned)
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate completion anchor digest differs"
        )
    return row


def parse_aggregate_completion_anchor_stdout(
    stdout: Any,
    *,
    return_code: Any,
    launch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if type(return_code) is not int or return_code != 0 or type(stdout) is not bytes:
        raise DecodedEvaluationLauncherError(
            "aggregate controller did not exit successfully with bytes stdout"
        )
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise DecodedEvaluationLauncherError(
            "aggregate controller stdout is not exactly one complete line"
        )
    try:
        literal = stdout[:-1].decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DecodedEvaluationLauncherError(
            "aggregate controller stdout is not UTF-8"
        ) from error
    decoded = _decode_canonical_json_literal(
        literal, label="aggregate completion anchor"
    )
    return validate_aggregate_completion_anchor(
        decoded, launch_manifest=launch_manifest
    )


def _deployment_authority(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "work_root_authority", "deployment_receipt",
        "source_spec_authority", "deployment_receipt_digest",
        "source_spec_authority_digest", "authority_digest",
    }
    row = dict(_closed(value, fields, label="deployment authority"))
    work_fields = {
        "schema_version", "path", "parent_path", "creation_identity",
        "immutable_identity", "parent_immutable_identity", "initial_entries",
        "retained_parent_fd_through_request_publication",
        "retained_root_fd_through_request_publication", "authority_digest",
    }
    work = dict(
        _closed(
            row["work_root_authority"], work_fields,
            label="deployment work root authority",
        )
    )
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    creation = _identity_row(
        work["creation_identity"], fields=identity_fields,
        label="deployment work root creation",
    )
    immutable = _identity_row(
        work["immutable_identity"], fields=immutable_fields,
        label="deployment work root immutable identity",
    )
    parent_immutable = _identity_row(
        work["parent_immutable_identity"], fields=immutable_fields,
        label="deployment work root parent immutable identity",
    )
    work_path = _canonical_absolute(
        work["path"], label="deployment work root"
    )
    parent_path = _canonical_absolute(
        work["parent_path"], label="deployment work root parent"
    )
    work_unsigned = dict(work)
    work_digest = work_unsigned.pop("authority_digest", None)
    if (
        work["schema_version"] != WORK_ROOT_AUTHORITY_SCHEMA
        or work_path.parent != parent_path
        or not stat.S_ISDIR(creation["mode"])
        or stat.S_IMODE(creation["mode"]) != 0o700
        or immutable != {key: creation[key] for key in immutable_fields}
        or not stat.S_ISDIR(parent_immutable["mode"])
        or work["initial_entries"] != []
        or work["retained_parent_fd_through_request_publication"] is not True
        or work["retained_root_fd_through_request_publication"] is not True
        or not isinstance(work_digest, str)
        or _SHA256.fullmatch(work_digest) is None
        or work_digest != object_sha256(work_unsigned)
    ):
        raise DecodedEvaluationLauncherError(
            "deployment work root authority differs"
        )
    pairs: dict[str, dict[str, str]] = {}
    for key in ("deployment_receipt", "source_spec_authority"):
        item = dict(_closed(row[key], {"path", "sha256"}, label=key))
        item_path = _canonical_absolute(item["path"], label=key)
        _sha(item["sha256"], label=f"{key} SHA")
        if item_path.parent != work_path or item_path.name in ("", ".", ".."):
            raise DecodedEvaluationLauncherError(
                f"{key} is not a direct work-root member"
            )
        pairs[key] = item
    for field in (
        "deployment_receipt_digest", "source_spec_authority_digest",
        "authority_digest",
    ):
        _sha(row[field], label=f"deployment authority {field}")
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest")
    if (
        row["schema_version"] != DEPLOYMENT_AUTHORITY_SCHEMA
        or pairs["deployment_receipt"]["path"]
        == pairs["source_spec_authority"]["path"]
        or claimed != object_sha256(unsigned)
    ):
        raise DecodedEvaluationLauncherError("deployment authority differs")
    row["work_root_authority"] = work
    row.update(pairs)
    return row


def _executor_capture_rows(
    deployment_authority: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    work_root = (
        None
        if deployment_authority is None
        else Path(deployment_authority["work_root_authority"]["path"])
    )
    rows = []
    for holder in plan.HOLDER_ROWS:
        basename = (
            EXECUTOR_CAPTURE_BASENAME_PREFIX
            + holder["job_id"]
            + EXECUTOR_CAPTURE_BASENAME_SUFFIX
        )
        rows.append(
            {
                "holder_job_id": holder["job_id"],
                "basename": basename,
                "path": None if work_root is None else str(work_root / basename),
            }
        )
    return rows


def _holder_anchor_channel(
    holder_job_id: str, *, required: bool,
) -> dict[str, Any]:
    if holder_job_id not in {item["job_id"] for item in plan.HOLDER_ROWS}:
        raise DecodedEvaluationLauncherError(
            "holder anchor channel job differs"
        )
    return {
        "schema_version": HOLDER_COMPLETION_ANCHOR_CHANNEL_SCHEMA,
        "holder_job_id": holder_job_id,
        "required_on_success": required,
        "success_exit_code": 0,
        "stdout_exact_line_count": 1 if required else 0,
        "stdout_serialization": (
            "canonical-json-newline-v1" if required else None
        ),
        "payload_schema_version": (
            HOLDER_COMPLETION_ANCHOR_SCHEMA if required else None
        ),
        "complete_stdout_and_eof_required": required,
        "extra_duplicate_or_partial_stdout_rejected": required,
        "external_trusted_orchestrator_required": required,
        "disk_authority_forbidden": True,
    }


def _evaluation_publication_projection(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            {
                "schema_version", "evaluation_root", "root_authority",
                "publication_receipt", "directory_authority",
                "directory_topology_digest", "materialized",
            },
            label="evaluation publication",
        )
    )
    publication = dict(
        _closed(
            row["publication_receipt"],
            {"file", "publication_digest"},
            label="evaluation publication receipt",
        )
    )
    directory = dict(
        _closed(
            row["directory_authority"],
            {"file", "authority_digest"},
            label="evaluation directory authority",
        )
    )
    for label, item, digest_field in (
        ("publication receipt", publication, "publication_digest"),
        ("directory authority", directory, "authority_digest"),
    ):
        file_row = item["file"]
        if not isinstance(file_row, Mapping):
            raise DecodedEvaluationLauncherError(
                f"evaluation {label} file binding differs"
            )
        _canonical_absolute(file_row.get("path"), label=f"evaluation {label}")
        _sha(file_row.get("sha256"), label=f"evaluation {label} file SHA")
        _sha(item[digest_field], label=f"evaluation {label} object digest")
    _sha(
        row["directory_topology_digest"],
        label="evaluation directory topology digest",
    )
    if row["materialized"] is not True:
        raise DecodedEvaluationLauncherError(
            "evaluation publication is not materialized"
        )
    return {
        "evaluation_root": row["evaluation_root"],
        "publication_receipt": {
            "path": publication["file"]["path"],
            "sha256": publication["file"]["sha256"],
            "publication_digest": publication["publication_digest"],
        },
        "directory_authority": {
            "path": directory["file"]["path"],
            "sha256": directory["file"]["sha256"],
            "authority_digest": directory["authority_digest"],
        },
        "directory_topology_digest": row["directory_topology_digest"],
    }


def _controller_target_argv(
    *, root_python: Mapping[str, Any], controller: Mapping[str, Any],
    deployment_authority: Mapping[str, Any], target: str,
    target_arguments: Sequence[str], capture_receipt_path: str,
) -> list[str]:
    root_path = _canonical_absolute(
        root_python.get("path"), label="controller root Python"
    )
    controller_path = _canonical_absolute(
        controller.get("path"), label="detached controller"
    )
    _sha(root_python.get("sha256"), label="controller root Python SHA")
    _sha(controller.get("sha256"), label="detached controller SHA")
    if root_path != executor.bridge.ROOT_PYTHON_PATH:
        raise DecodedEvaluationLauncherError(
            "controller root Python literal path differs"
        )
    capture = _canonical_absolute(
        capture_receipt_path, label="executor runtime capture receipt"
    )
    work_root = Path(
        deployment_authority["work_root_authority"]["path"]
    )
    if capture.parent != work_root or capture.name in ("", ".", ".."):
        raise DecodedEvaluationLauncherError(
            "executor runtime capture is not a direct work-root member"
        )
    return [
        str(root_path), "-I", "-S", "-B", "-c",
        ROOT_CONTROLLER_BOOTSTRAP_SOURCE,
        str(controller_path), controller["sha256"],
        "run-target",
        "--deployment-receipt",
        deployment_authority["deployment_receipt"]["path"],
        "--deployment-receipt-sha256",
        deployment_authority["deployment_receipt"]["sha256"],
        "--source-spec-authority",
        deployment_authority["source_spec_authority"]["path"],
        "--source-spec-authority-sha256",
        deployment_authority["source_spec_authority"]["sha256"],
        "--target", target,
        "--capture-receipt", str(capture),
        "--", *list(target_arguments),
    ]


def _tool(value: Any, *, label: str, verify_file: bool) -> dict[str, str]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    if not isinstance(row["path"], str) or not Path(row["path"]).is_absolute():
        raise DecodedEvaluationLauncherError(f"{label} path must be absolute")
    _sha(row["sha256"], label=f"{label} SHA")
    if verify_file:
        path = Path(row["path"])
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise DecodedEvaluationLauncherError(f"{label} does not exist") from error
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise DecodedEvaluationLauncherError(f"{label} is not a plain file")
        if plan.file_sha256(path) != row["sha256"]:
            raise DecodedEvaluationLauncherError(f"{label} hash differs")
    return row


def _capture_blinding_key(
    path_value: Any, *, expected_sha256: Any,
) -> dict[str, Any]:
    path = _canonical_absolute(path_value, label="blinding key")
    expected = _sha(expected_sha256, label="blinding key SHA")
    try:
        if path.resolve(strict=True) != path or path.is_symlink():
            raise DecodedEvaluationLauncherError(
                "blinding key path is not canonical"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise DecodedEvaluationLauncherError(
            f"cannot open blinding key: {error}"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = _read_descriptor(descriptor)
        middle = os.fstat(descriptor)
        second = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    observed = hashlib.sha256(first).hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_size < 32
        or first != second
        or len(first) != before.st_size
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or observed != expected
    ):
        raise DecodedEvaluationLauncherError(
            "blinding key same-FD identity, mode, size, or SHA differs"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "size": len(first),
        "mode": stat.S_IMODE(before.st_mode),
    }


def _blinding_key(
    value: Any, *, verify_file: bool,
) -> dict[str, Any]:
    if verify_file:
        pair = dict(
            _closed(value, {"path", "sha256"}, label="blinding key input")
        )
        return _capture_blinding_key(
            pair["path"], expected_sha256=pair["sha256"]
        )
    row = dict(
        _closed(
            value,
            {"path", "sha256", "size", "mode"},
            label="blinding key",
        )
    )
    _canonical_absolute(row["path"], label="blinding key")
    _sha(row["sha256"], label="blinding key SHA")
    if (
        type(row["size"]) is not int
        or row["size"] < 32
        or row["mode"] != 0o400
    ):
        raise DecodedEvaluationLauncherError(
            "blinding key size or mode differs"
        )
    return row


def _validate_launcher_work_root_continuity(
    *, launcher_capture: Mapping[str, Any],
    deployment_authority: Mapping[str, Any],
    inherited_work_root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        if inherited_work_root is not None:
            work_root = (
                executor.bridge.verified_release
                .validate_inherited_work_root_binding(
                    inherited_work_root,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            )
            receipt_authority_kind = "work_root"
            task_fd_binding = None
        else:
            raw, _ = executor.bridge._stable_file(
                launcher_capture["receipt_path"],
                label="launcher verified release capture",
                expected_sha256=launcher_capture["receipt_sha256"],
            )
            receipt = executor.bridge._json(
                raw, label="launcher verified release capture", canonical=True
            )
            receipt = dict(
                executor.bridge.verified_release.validate_capture_receipt(
                    receipt, verify_file=False
                )
            )
            work_root = receipt["work_root"]
            if work_root is not None:
                work_root = (
                    executor.bridge.verified_release
                    .validate_inherited_work_root_binding(
                        work_root,
                        verify_open_fds=True,
                        expected_inheritable=False,
                        verify_entries=False,
                        allow_root_metadata_change=True,
                    )
                )
            receipt_authority_kind = receipt["publication_authority_kind"]
            task_fd_binding = receipt["task_fd_binding"]
    except (
        executor.bridge.DecodedEvaluationBridgeError,
        executor.bridge.verified_release.DecodedEvalVerifiedReleaseError,
    ) as error:
        raise DecodedEvaluationLauncherError(str(error)) from error
    expected_work = deployment_authority["work_root_authority"]
    if (
        receipt_authority_kind != "work_root"
        or work_root is None
        or task_fd_binding is not None
        or work_root["path"] != expected_work["path"]
        or work_root["work_root_authority_digest"]
        != expected_work["authority_digest"]
        or work_root["deployment_receipt_digest"]
        != deployment_authority["deployment_receipt_digest"]
        or work_root["source_spec_authority_digest"]
        != deployment_authority["source_spec_authority_digest"]
        or work_root["target"]
        != "action_preservation_decoded_eval_launcher_v1.py"
        or work_root["capture_receipt_path"]
        != launcher_capture["receipt_path"]
    ):
        raise DecodedEvaluationLauncherError(
            "launcher signed work-root continuity differs"
        )
    return dict(work_root)


def _launcher_work_root_projection(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "path", "parent_path", "root_immutable_identity",
        "parent_immutable_identity", "entries_before_current_capture",
        "work_root_authority_digest", "deployment_receipt_digest",
        "source_spec_authority_digest", "target", "capture_receipt_path",
        "projection_digest",
    }
    if isinstance(value, Mapping) and value.get("schema_version") == (
        LAUNCHER_WORK_ROOT_PROJECTION_SCHEMA
    ):
        row = dict(_closed(value, fields, label="launcher work-root projection"))
    else:
        if not isinstance(value, Mapping):
            raise DecodedEvaluationLauncherError(
                "launcher work-root binding differs"
            )
        row = {
            "schema_version": LAUNCHER_WORK_ROOT_PROJECTION_SCHEMA,
            "path": value.get("path"),
            "parent_path": value.get("parent_path"),
            "root_immutable_identity": value.get("root_immutable_identity"),
            "parent_immutable_identity": value.get(
                "parent_immutable_identity"
            ),
            "entries_before_current_capture": value.get("entries"),
            "work_root_authority_digest": value.get(
                "work_root_authority_digest"
            ),
            "deployment_receipt_digest": value.get(
                "deployment_receipt_digest"
            ),
            "source_spec_authority_digest": value.get(
                "source_spec_authority_digest"
            ),
            "target": value.get("target"),
            "capture_receipt_path": value.get("capture_receipt_path"),
        }
        row["projection_digest"] = object_sha256(row)
    path = _canonical_absolute(row["path"], label="launcher work root")
    parent = _canonical_absolute(
        row["parent_path"], label="launcher work-root parent"
    )
    capture = _canonical_absolute(
        row["capture_receipt_path"], label="launcher capture receipt"
    )
    immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    for label, identity in (
        ("root", row["root_immutable_identity"]),
        ("parent", row["parent_immutable_identity"]),
    ):
        if (
            not isinstance(identity, Mapping)
            or set(identity) != immutable_fields
            or any(
                type(identity[field]) is not int or identity[field] < 0
                for field in immutable_fields
            )
            or not stat.S_ISDIR(identity["mode"])
        ):
            raise DecodedEvaluationLauncherError(
                f"launcher work-root {label} identity differs"
            )
    entries = row["entries_before_current_capture"]
    if (
        path.parent != parent
        or capture.parent != path
        or capture.name in ("", ".", "..")
        or not isinstance(entries, list)
        or entries != sorted(set(entries))
        or any(
            type(name) is not str or name in ("", ".", "..")
            or os.path.sep in name
            for name in entries
        )
        or capture.name in entries
        or row["target"]
        != "action_preservation_decoded_eval_launcher_v1.py"
    ):
        raise DecodedEvaluationLauncherError(
            "launcher work-root projection differs"
        )
    for field in (
        "work_root_authority_digest", "deployment_receipt_digest",
        "source_spec_authority_digest", "projection_digest",
    ):
        _sha(row[field], label=f"launcher work-root {field}")
    unsigned = dict(row)
    claimed = unsigned.pop("projection_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationLauncherError(
            "launcher work-root projection digest differs"
        )
    return row


def build_launch_manifest(
    *,
    bundle: Mapping[str, Any],
    launch_root: str | Path,
    python_identity: Mapping[str, Any],
    executor_identity: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    ffprobe_identity: Mapping[str, Any],
    physical_bindings_identity: Mapping[str, Any],
    verify_tools: bool,
    blinding_key_identity: Mapping[str, Any] | None = None,
    aggregate_root: str | Path | None = None,
    launcher_verified_release_capture: Mapping[str, Any] | None = None,
    launcher_work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(launch_root)
    if not root.is_absolute() or str(root) == os.path.sep or os.path.normpath(str(root)) != str(root):
        raise DecodedEvaluationLauncherError("launch root must be normalized, absolute, and non-root")
    python = _tool(python_identity, label="python", verify_file=verify_tools)
    executor_tool = _tool(executor_identity, label="executor", verify_file=verify_tools)
    decoder = _tool(decoder_identity, label="decoder adapter", verify_file=verify_tools)
    ffprobe = _tool(ffprobe_identity, label="ffprobe", verify_file=verify_tools)
    physical_bindings = _tool(
        physical_bindings_identity, label="physical bindings", verify_file=verify_tools
    )
    if blinding_key_identity is None:
        if verify_tools:
            raise DecodedEvaluationLauncherError(
                "production launch planning requires a pinned blinding key"
            )
        blinding_key_identity = {
            "path": "/injected-stub/blinding-key-not-executed",
            "sha256": hashlib.sha256(
                b"injected-stub-blinding-key-not-executed"
            ).hexdigest(),
            "size": 32,
            "mode": 0o400,
        }
    blinding_key = _blinding_key(
        blinding_key_identity, verify_file=verify_tools
    )
    if aggregate_root is None:
        if verify_tools:
            raise DecodedEvaluationLauncherError(
                "production launch planning requires an aggregate root"
            )
        aggregate_root = root.parent / "aggregate"
    aggregate = _canonical_absolute(str(aggregate_root), label="aggregate root")
    if aggregate.parent != root.parent or aggregate == root:
        raise DecodedEvaluationLauncherError(
            "aggregate root must be a distinct direct launch-root sibling"
        )
    aggregate_capture = aggregate.parent / AGGREGATE_CAPTURE_BASENAME
    if verify_tools and executor_tool["sha256"] != plan.file_sha256(executor.__file__):
        raise DecodedEvaluationLauncherError("executor module hash differs from loaded implementation")
    if verify_tools and decoder["sha256"] != plan.file_sha256(
        executor.decoder_adapter.__file__
    ):
        raise DecodedEvaluationLauncherError(
            "decoder adapter differs from loaded audited implementation"
        )
    bindings = None
    verified_runtime = None
    launcher_capture = None
    deployment_authority = None
    launcher_work_root = None
    evaluation_publication = None
    if verify_tools:
        if launcher_work_root_binding is None:
            raise DecodedEvaluationLauncherError(
                "production launch planning requires inherited work-root FDs"
            )
        try:
            bindings = executor.bridge.load_physical_bindings(
                physical_bindings["path"],
                expected_sha256=physical_bindings["sha256"],
                verify_files=True,
            )
        except executor.bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationLauncherError(str(error)) from error
        deployment_authority = _deployment_authority(
            bindings.get("deployment_authority")
        )
        evaluation_publication = _evaluation_publication_projection(
            bindings.get("evaluation_publication")
        )
        launcher_capture = executor._capture_evidence(
            launcher_verified_release_capture,
            label="launcher verified release capture",
        )
        if (
            launcher_capture is None
            or launcher_capture["target"]
            != "action_preservation_decoded_eval_launcher_v1.py"
        ):
            raise DecodedEvaluationLauncherError(
                "production launcher is outside the verified runtime"
            )
        expected_launcher_arguments = [
            "--evaluation-root", bundle["manifest"]["evaluation_root"],
            "--launch-root", str(root),
            "--python", python["path"],
            "--python-sha256", python["sha256"],
            "--executor", executor_tool["path"],
            "--executor-sha256", executor_tool["sha256"],
            "--decoder-adapter", decoder["path"],
            "--decoder-adapter-sha256", decoder["sha256"],
            "--ffprobe", ffprobe["path"],
            "--ffprobe-sha256", ffprobe["sha256"],
            "--physical-bindings", physical_bindings["path"],
            "--physical-bindings-sha256", physical_bindings["sha256"],
            "--blinding-key-file", blinding_key["path"],
            "--blinding-key-sha256", blinding_key["sha256"],
            "--aggregate-root", str(aggregate),
        ]
        try:
            replayed_launcher_capture = (
                executor.bridge.validate_verified_capture_receipt(
                    bindings,
                    receipt_path=launcher_capture["receipt_path"],
                    target="action_preservation_decoded_eval_launcher_v1.py",
                    expected_arguments=expected_launcher_arguments,
                    expected_capture_digest=launcher_capture["capture_digest"],
                    verify_file=True,
                )
            )
        except executor.bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationLauncherError(str(error)) from error
        if replayed_launcher_capture != launcher_capture:
            raise DecodedEvaluationLauncherError(
                "launcher verified capture replay differs"
            )
        launcher_work_root_binding = _validate_launcher_work_root_continuity(
            launcher_capture=launcher_capture,
            deployment_authority=deployment_authority,
            inherited_work_root=launcher_work_root_binding,
        )
        launcher_work_root = _launcher_work_root_projection(
            launcher_work_root_binding
        )
        try:
            for relative_path, module_path in (
                (
                    "action_preservation_decoded_eval_launcher_v1.py",
                    __file__,
                ),
                (
                    "action_preservation_decoded_eval_executor_v2.py",
                    executor.__file__,
                ),
                (
                    "action_preservation_decoded_eval_decoder_adapter_v1.py",
                    executor.decoder_adapter.__file__,
                ),
                (
                    "action_preservation_decoded_eval_bridge_v1.py",
                    executor.bridge.__file__,
                ),
                (
                    "action_preservation_decoded_eval_plan_v1.py",
                    plan.__file__,
                ),
                ("action_preservation_gate_v1.py", plan.gate.__file__),
            ):
                executor.bridge.require_running_eval_release_member(
                    bindings["eval_release"],
                    relative_path=relative_path,
                    running_path=module_path,
                )
        except executor.bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationLauncherError(str(error)) from error
        if (
            bindings["evaluation_id"] != bundle["manifest"]["evaluation_id"]
            or bindings["input_digest"] != bundle["input_spec"]["input_digest"]
            or bindings["manifest_digest"] != bundle["manifest"]["manifest_digest"]
        ):
            raise DecodedEvaluationLauncherError(
                "physical bindings differ from evaluation bundle"
            )
        if any(
            bindings["runtime"][runtime_key][field] != identity[field]
            for runtime_key, identity in (
                ("decoder_adapter", decoder), ("ffprobe", ffprobe)
            )
            for field in ("path", "sha256")
        ):
            raise DecodedEvaluationLauncherError(
                "decoder/ffprobe tools differ from physical runtime authority"
            )
        if any(
            bindings["runtime"]["python"][field] != python[field]
            for field in ("path", "sha256")
        ):
            raise DecodedEvaluationLauncherError(
                "launcher Python differs from physical runtime authority"
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
            raise DecodedEvaluationLauncherError(
                "physical pin files differ from evaluation input authority"
            )
        verified_runtime = {
            "root_python": executor.bridge.executable_runtime_binding(
                bindings["runtime"]["root_python"]
            ),
            "frozen_python": executor.bridge.executable_runtime_binding(
                bindings["runtime"]["python"]
            ),
            "site_packages": executor.bridge.directory_runtime_binding(
                bindings["runtime"]["site_packages"]
            ),
            "torchrun": executor.bridge.torchrun_runtime_binding(
                bindings["runtime"]["torchrun"]
            ),
            "release": executor.bridge.eval_release_runtime_binding(
                bindings["eval_release"]
            ),
            "controller": executor.bridge.executable_runtime_binding(
                bindings["runtime"]["deployment_controller"]
            ),
            "controller_authority": (
                executor.bridge.controller_authority_runtime_binding(
                    bindings["runtime"]["controller_authority"]
                )
            ),
        }
    elif (
        launcher_verified_release_capture is not None
        or launcher_work_root_binding is not None
    ):
        raise DecodedEvaluationLauncherError(
            "injected launcher may not claim a verified runtime"
        )
    manifest = bundle["manifest"]
    executor_capture_receipts = _executor_capture_rows(deployment_authority)
    capture_by_holder = {
        item["holder_job_id"]: item for item in executor_capture_receipts
    }
    if verify_tools and any(
        item["basename"]
        in launcher_work_root["entries_before_current_capture"]
        or item["path"] == launcher_capture["receipt_path"]
        for item in executor_capture_receipts
    ):
        raise DecodedEvaluationLauncherError(
            "executor runtime capture receipt is not fresh"
        )
    if verify_tools and (
        aggregate.parent
        != Path(deployment_authority["work_root_authority"]["path"])
        or aggregate.name in launcher_work_root["entries_before_current_capture"]
        or aggregate_capture.name
        in launcher_work_root["entries_before_current_capture"]
        or str(aggregate_capture) == launcher_capture["receipt_path"]
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate root or runtime capture is not fresh in signed work root"
        )
    if verify_tools and (
        Path(blinding_key["path"]).parent
        != Path(deployment_authority["work_root_authority"]["path"])
        or Path(blinding_key["path"]).name
        not in launcher_work_root["entries_before_current_capture"]
        or blinding_key["path"]
        in {
            deployment_authority["deployment_receipt"]["path"],
            deployment_authority["source_spec_authority"]["path"],
            launcher_capture["receipt_path"],
            str(aggregate_capture),
        }
    ):
        raise DecodedEvaluationLauncherError(
            "blinding key is not a distinct signed work-root member"
        )
    commands = []
    for holder in plan.HOLDER_ROWS:
        job_id = holder["job_id"]
        shard = bundle["shards"][job_id]
        target_arguments = [
            "--evaluation-root",
            manifest["evaluation_root"],
            "--holder-job-id",
            job_id,
            "--decoder-adapter",
            decoder["path"],
            "--decoder-adapter-sha256",
            decoder["sha256"],
            "--ffprobe",
            ffprobe["path"],
            "--ffprobe-sha256",
            ffprobe["sha256"],
            "--physical-bindings",
            physical_bindings["path"],
            "--physical-bindings-sha256",
            physical_bindings["sha256"],
            "--confirmation",
            f"execute-local-decoded-eval-shard-v2-{job_id}",
        ]
        capture_path = capture_by_holder[job_id]["path"]
        if verify_tools:
            assert bindings is not None
            assert verified_runtime is not None
            assert deployment_authority is not None
            assert capture_path is not None
            argv = _controller_target_argv(
                root_python=verified_runtime["root_python"],
                controller=verified_runtime["controller"],
                deployment_authority=deployment_authority,
                target="action_preservation_decoded_eval_executor_v2.py",
                target_arguments=target_arguments,
                capture_receipt_path=capture_path,
            )
        else:
            argv = [python["path"], executor_tool["path"], *target_arguments]
        commands.append(
            {
                "holder": {"job_id": job_id, "node": holder["node"]},
                "shard_digest": shard["shard_digest"],
                "candidate_task_count": 64,
                "base_control_task_count": 2,
                "argv": argv,
                "shell": False,
                "operator_must_invoke_manually": True,
                "execution_transport": (
                    "external_trusted_orchestrator_inside_existing_holder_allocation"
                ),
                "holder_completion_anchor_channel": _holder_anchor_channel(
                    job_id, required=verify_tools
                ),
                "verified_runtime_capture_receipt_path": (
                    capture_path if verify_tools else None
                ),
                "detached_controller_run_target": verify_tools,
                "release_member_path_executed_directly": False,
            }
        )
    value = {
        "schema_version": SCHEMA,
        "launch_root": str(root),
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_root": manifest["evaluation_root"],
        "evaluation_manifest_digest": manifest["manifest_digest"],
        "publication_digest": (
            bundle["publication_receipt"]["publication_digest"]
            if evaluation_publication is None
            else evaluation_publication["publication_receipt"][
                "publication_digest"
            ]
        ),
        "evaluation_publication": evaluation_publication,
        "tools": {
            "python": python,
            "executor": executor_tool,
            "decoder_adapter": decoder,
            "ffprobe": ffprobe,
            "physical_bindings": physical_bindings,
        },
        "blinding_key": blinding_key,
        "aggregate_root": str(aggregate),
        "aggregate_runtime_capture_receipt_path": str(aggregate_capture),
        "verified_runtime": verified_runtime,
        "deployment_authority": deployment_authority,
        "launcher_work_root": launcher_work_root,
        "executor_runtime_capture_receipts": executor_capture_receipts,
        "launcher_verified_release_capture": launcher_capture,
        "launch_manifest_anchor_channel": (
            _launch_manifest_anchor_channel(required=verify_tools)
        ),
        "controller_bootstrap_source_sha256": (
            hashlib.sha256(
                ROOT_CONTROLLER_BOOTSTRAP_SOURCE.encode("utf-8")
            ).hexdigest()
            if verify_tools
            else None
        ),
        "commands": commands,
        "exact_holder_command_count": 4,
        "two_stage_execution_required": verify_tools,
        "external_trusted_orchestrator_required": verify_tools,
        "online_anchor_disk_authority_forbidden": True,
        "holder_anchor_collection_performed": False,
        "aggregate_command_plan_emitted": False,
        "tool_files_verified": verify_tools,
        "execution_backend": "detached_controller_run_target"
        if verify_tools
        else "injected_stub_plan",
        "command_execution_performed": False,
        "subprocess_spawned": False,
        "network_used": False,
        "remote_launch_performed": False,
        "scheduler_command_present": False,
        "subprocess_environment_denylist": list(executor.SUBPROCESS_ENV_DENYLIST),
        "automatic_retry": False,
        "training_loss_read_or_used": False,
        "create_only_publication": True,
    }
    value["launch_manifest_digest"] = object_sha256(value)
    return validate_launch_manifest(value, bundle=bundle)


def validate_launch_manifest(value: Any, *, bundle: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "launch_root",
        "evaluation_id",
        "evaluation_root",
        "evaluation_manifest_digest",
        "publication_digest",
        "evaluation_publication",
        "tools",
        "blinding_key",
        "aggregate_root",
        "aggregate_runtime_capture_receipt_path",
        "verified_runtime",
        "deployment_authority",
        "launcher_work_root",
        "executor_runtime_capture_receipts",
        "launcher_verified_release_capture",
        "launch_manifest_anchor_channel",
        "controller_bootstrap_source_sha256",
        "commands",
        "exact_holder_command_count",
        "two_stage_execution_required",
        "external_trusted_orchestrator_required",
        "online_anchor_disk_authority_forbidden",
        "holder_anchor_collection_performed",
        "aggregate_command_plan_emitted",
        "tool_files_verified",
        "execution_backend",
        "command_execution_performed",
        "subprocess_spawned",
        "network_used",
        "remote_launch_performed",
        "scheduler_command_present",
        "subprocess_environment_denylist",
        "automatic_retry",
        "training_loss_read_or_used",
        "create_only_publication",
        "launch_manifest_digest",
    }
    row = dict(_closed(value, fields, label="local launch manifest"))
    if row["schema_version"] != SCHEMA:
        raise DecodedEvaluationLauncherError("local launch schema differs")
    root = Path(row["launch_root"])
    if not root.is_absolute() or str(root) == os.path.sep or os.path.normpath(str(root)) != str(root):
        raise DecodedEvaluationLauncherError("launch root differs")
    manifest = bundle["manifest"]
    expected_bindings = {
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_root": manifest["evaluation_root"],
        "evaluation_manifest_digest": manifest["manifest_digest"],
    }
    for key, expected in expected_bindings.items():
        if row[key] != expected:
            raise DecodedEvaluationLauncherError(f"launch binding differs: {key}")
    tools = dict(
        _closed(
            row["tools"],
            {"python", "executor", "decoder_adapter", "ffprobe", "physical_bindings"},
            label="launch tools",
        )
    )
    for key, label in (
        ("python", "python"),
        ("executor", "executor"),
        ("decoder_adapter", "decoder adapter"),
        ("ffprobe", "ffprobe"),
        ("physical_bindings", "physical bindings"),
    ):
        tools[key] = _tool(tools[key], label=label, verify_file=False)
    blinding_key = _blinding_key(row["blinding_key"], verify_file=False)
    aggregate_root = _canonical_absolute(
        row["aggregate_root"], label="aggregate root"
    )
    aggregate_capture = _canonical_absolute(
        row["aggregate_runtime_capture_receipt_path"],
        label="aggregate runtime capture receipt",
    )
    if (
        aggregate_root.parent != root.parent
        or aggregate_root == root
        or aggregate_capture.parent != aggregate_root.parent
        or aggregate_capture.name != AGGREGATE_CAPTURE_BASENAME
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate root or runtime capture binding differs"
        )
    if row["tool_files_verified"]:
        runtime_fields = {
            "root_python", "frozen_python", "site_packages", "torchrun",
            "release", "controller", "controller_authority",
        }
        runtime = dict(
            _closed(
                row["verified_runtime"], runtime_fields,
                label="launch verified runtime",
            )
        )
        root_python = runtime["root_python"]
        frozen_python = runtime["frozen_python"]
        site_packages = runtime["site_packages"]
        torchrun = runtime["torchrun"]
        release = runtime["release"]
        controller = runtime["controller"]
        controller_authority = runtime["controller_authority"]
        deployment_authority = _deployment_authority(
            row["deployment_authority"]
        )
        evaluation_publication = _evaluation_publication_projection(
            row["evaluation_publication"]
        )
        launcher_work_root = _launcher_work_root_projection(
            row["launcher_work_root"]
        )
        # The runtime API performs the closed-field validation while commands
        # are reconstructed below.
        launcher_capture = executor._capture_evidence(
            row["launcher_verified_release_capture"],
            label="launcher verified release capture",
        )
        if (
            launcher_capture is None
            or launcher_capture["target"]
            != "action_preservation_decoded_eval_launcher_v1.py"
            or row["controller_bootstrap_source_sha256"]
            != hashlib.sha256(
                ROOT_CONTROLLER_BOOTSTRAP_SOURCE.encode("utf-8")
            ).hexdigest()
            or tools["python"]["path"] != frozen_python.get("path")
            or tools["python"]["sha256"] != frozen_python.get("sha256")
            or launcher_work_root["path"]
            != deployment_authority["work_root_authority"]["path"]
            or launcher_work_root["work_root_authority_digest"]
            != deployment_authority["work_root_authority"]["authority_digest"]
            or launcher_work_root["deployment_receipt_digest"]
            != deployment_authority["deployment_receipt_digest"]
            or launcher_work_root["source_spec_authority_digest"]
            != deployment_authority["source_spec_authority_digest"]
            or evaluation_publication["evaluation_root"]
            != manifest["evaluation_root"]
            or row["publication_digest"]
            != evaluation_publication["publication_receipt"][
                "publication_digest"
            ]
        ):
            raise DecodedEvaluationLauncherError(
                "launch verified runtime authority differs"
            )
    else:
        if (
            row["verified_runtime"] is not None
            or row["deployment_authority"] is not None
            or row["launcher_work_root"] is not None
            or row["evaluation_publication"] is not None
            or row["launcher_verified_release_capture"] is not None
            or row["controller_bootstrap_source_sha256"] is not None
        ):
            raise DecodedEvaluationLauncherError(
                "injected launch plan claims a verified runtime"
            )
        root_python = frozen_python = site_packages = torchrun = release = None
        controller = controller_authority = None
        deployment_authority = launcher_work_root = None
        evaluation_publication = None
        if row["publication_digest"] != bundle["publication_receipt"][
            "publication_digest"
        ]:
            raise DecodedEvaluationLauncherError(
                "synthetic publication digest differs"
            )
    expected_capture_receipts = _executor_capture_rows(deployment_authority)
    if row["launch_manifest_anchor_channel"] != (
        _launch_manifest_anchor_channel(required=row["tool_files_verified"])
    ):
        raise DecodedEvaluationLauncherError(
            "launch manifest anchor channel differs"
        )
    if row["executor_runtime_capture_receipts"] != expected_capture_receipts:
        raise DecodedEvaluationLauncherError(
            "executor runtime capture receipt closure differs"
        )
    capture_by_holder = {
        item["holder_job_id"]: item for item in expected_capture_receipts
    }
    if row["tool_files_verified"] and any(
        item["basename"]
        in launcher_work_root["entries_before_current_capture"]
        or item["path"] == launcher_capture["receipt_path"]
        for item in expected_capture_receipts
    ):
        raise DecodedEvaluationLauncherError(
            "executor runtime capture receipt is not fresh"
        )
    if row["tool_files_verified"] and (
        aggregate_root.parent != Path(launcher_work_root["path"])
        or aggregate_root.name
        in launcher_work_root["entries_before_current_capture"]
        or aggregate_capture.name
        in launcher_work_root["entries_before_current_capture"]
        or str(aggregate_capture) == launcher_capture["receipt_path"]
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate root or runtime capture is not fresh in signed work root"
        )
    if row["tool_files_verified"] and (
        Path(blinding_key["path"]).parent != Path(launcher_work_root["path"])
        or Path(blinding_key["path"]).name
        not in launcher_work_root["entries_before_current_capture"]
        or blinding_key["path"]
        in {
            deployment_authority["deployment_receipt"]["path"],
            deployment_authority["source_spec_authority"]["path"],
            launcher_capture["receipt_path"],
            str(aggregate_capture),
        }
    ):
        raise DecodedEvaluationLauncherError(
            "blinding key is not a distinct signed work-root member"
        )
    if not isinstance(row["commands"], list) or len(row["commands"]) != 4:
        raise DecodedEvaluationLauncherError("local holder command count differs")
    forbidden_tokens = {"ssh", "srun", "sbatch", "scancel", "curl", "wget"}
    for holder, command in zip(plan.HOLDER_ROWS, row["commands"]):
        target_arguments = [
            "--evaluation-root",
            manifest["evaluation_root"],
            "--holder-job-id",
            holder["job_id"],
            "--decoder-adapter",
            tools["decoder_adapter"]["path"],
            "--decoder-adapter-sha256",
            tools["decoder_adapter"]["sha256"],
            "--ffprobe",
            tools["ffprobe"]["path"],
            "--ffprobe-sha256",
            tools["ffprobe"]["sha256"],
            "--physical-bindings",
            tools["physical_bindings"]["path"],
            "--physical-bindings-sha256",
            tools["physical_bindings"]["sha256"],
            "--confirmation",
            f"execute-local-decoded-eval-shard-v2-{holder['job_id']}",
        ]
        capture_path = capture_by_holder[holder["job_id"]]["path"]
        if row["tool_files_verified"]:
            assert root_python is not None
            assert frozen_python is not None
            assert release is not None
            assert controller is not None
            assert deployment_authority is not None
            assert capture_path is not None
            expected_argv = _controller_target_argv(
                root_python=root_python,
                controller=controller,
                deployment_authority=deployment_authority,
                target="action_preservation_decoded_eval_executor_v2.py",
                target_arguments=target_arguments,
                capture_receipt_path=capture_path,
            )
        else:
            expected_argv = [
                tools["python"]["path"], tools["executor"]["path"],
                *target_arguments,
            ]
        expected = {
            "holder": {"job_id": holder["job_id"], "node": holder["node"]},
            "shard_digest": bundle["shards"][holder["job_id"]]["shard_digest"],
            "candidate_task_count": 64,
            "base_control_task_count": 2,
            "argv": expected_argv,
            "shell": False,
            "operator_must_invoke_manually": True,
            "execution_transport": (
                "external_trusted_orchestrator_inside_existing_holder_allocation"
            ),
            "holder_completion_anchor_channel": _holder_anchor_channel(
                holder["job_id"], required=row["tool_files_verified"]
            ),
            "verified_runtime_capture_receipt_path": (
                capture_path if row["tool_files_verified"] else None
            ),
            "detached_controller_run_target": row["tool_files_verified"],
            "release_member_path_executed_directly": False,
        }
        if command != expected:
            raise DecodedEvaluationLauncherError("local holder command differs")
        if any(Path(token).name in forbidden_tokens for token in expected_argv):
            raise DecodedEvaluationLauncherError("network or scheduler command is forbidden")
    scalar = {
        "exact_holder_command_count": 4,
        "two_stage_execution_required": row["tool_files_verified"],
        "external_trusted_orchestrator_required": row[
            "tool_files_verified"
        ],
        "online_anchor_disk_authority_forbidden": True,
        "holder_anchor_collection_performed": False,
        "aggregate_command_plan_emitted": False,
        "command_execution_performed": False,
        "subprocess_spawned": False,
        "network_used": False,
        "remote_launch_performed": False,
        "scheduler_command_present": False,
        "subprocess_environment_denylist": list(executor.SUBPROCESS_ENV_DENYLIST),
        "automatic_retry": False,
        "training_loss_read_or_used": False,
        "create_only_publication": True,
    }
    for key, expected in scalar.items():
        if row[key] != expected:
            raise DecodedEvaluationLauncherError(f"local launch policy differs: {key}")
    if type(row["tool_files_verified"]) is not bool:
        raise DecodedEvaluationLauncherError("launch tool verification flag differs")
    if row["execution_backend"] not in {
        "detached_controller_run_target",
        "injected_stub_plan",
    }:
        raise DecodedEvaluationLauncherError("launch execution backend differs")
    if (row["execution_backend"] == "detached_controller_run_target") is not row[
        "tool_files_verified"
    ]:
        raise DecodedEvaluationLauncherError("launch backend/tool verification closure differs")
    digest = _sha(row["launch_manifest_digest"], label="launch manifest digest")
    payload = dict(row)
    payload.pop("launch_manifest_digest")
    if object_sha256(payload) != digest:
        raise DecodedEvaluationLauncherError("launch manifest digest differs")
    return row


def _launch_manifest_anchor_channel(*, required: bool) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_MANIFEST_ANCHOR_CHANNEL_SCHEMA,
        "payload_schema_version": (
            LAUNCH_MANIFEST_ANCHOR_SCHEMA if required else None
        ),
        "payload_exact_fields": (
            list(LAUNCH_MANIFEST_ANCHOR_FIELDS) if required else []
        ),
        "required_on_success": required,
        "success_exit_code": 0,
        "stdout_exact_line_count": 1 if required else 0,
        "stdout_serialization": (
            "canonical-json-newline-v1" if required else None
        ),
        "complete_stdout_and_eof_required": required,
        "extra_duplicate_or_partial_stdout_rejected": required,
        "external_trusted_orchestrator_required": required,
        "retain_literal_for_aggregate_launch": required,
        "post_exit_path_hashing_forbidden": True,
    }


def build_launch_manifest_anchor(
    *,
    launch_manifest: Mapping[str, Any],
    path: str | Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    launch_path = _canonical_absolute(
        str(path), label="launch manifest anchor"
    )
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    captured_identity = _identity_row(
        identity,
        fields=identity_fields,
        label="launch manifest anchor identity",
    )
    payload = canonical_json_bytes(launch_manifest) + b"\n"
    value = {
        "schema_version": LAUNCH_MANIFEST_ANCHOR_SCHEMA,
        "path": str(launch_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "mode": 0o400,
        "identity": captured_identity,
        "launch_manifest_digest": launch_manifest["launch_manifest_digest"],
    }
    value["anchor_digest"] = object_sha256(value)
    return validate_launch_manifest_anchor(value, launch_manifest=launch_manifest)


def validate_launch_manifest_anchor(
    value: Any, *, launch_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(
        _closed(
            value,
            set(LAUNCH_MANIFEST_ANCHOR_FIELDS),
            label="launch manifest anchor",
        )
    )
    path = _canonical_absolute(row["path"], label="launch manifest anchor")
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    identity = _identity_row(
        row["identity"],
        fields=identity_fields,
        label="launch manifest anchor identity",
    )
    payload = canonical_json_bytes(launch_manifest) + b"\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    if (
        row["schema_version"] != LAUNCH_MANIFEST_ANCHOR_SCHEMA
        or path != Path(launch_manifest["launch_root"]) / FILENAME
        or row["sha256"] != expected_sha
        or row["size"] != len(payload)
        or row["mode"] != 0o400
        or not stat.S_ISREG(identity["mode"])
        or stat.S_IMODE(identity["mode"]) != 0o400
        or identity["nlink"] != 1
        or identity["size"] != len(payload)
        or row["launch_manifest_digest"]
        != launch_manifest["launch_manifest_digest"]
    ):
        raise DecodedEvaluationLauncherError(
            "launch manifest anchor binding differs"
        )
    _sha(row["sha256"], label="launch manifest anchor SHA")
    _sha(
        row["launch_manifest_digest"],
        label="launch manifest anchor manifest digest",
    )
    claimed = _sha(row["anchor_digest"], label="launch manifest anchor digest")
    unsigned = dict(row)
    unsigned.pop("anchor_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationLauncherError(
            "launch manifest anchor digest differs"
        )
    return row


def parse_launch_manifest_anchor_stdout(
    stdout: Any,
    *,
    return_code: Any,
    launch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if type(return_code) is not int or return_code != 0 or type(stdout) is not bytes:
        raise DecodedEvaluationLauncherError(
            "launcher did not exit successfully with bytes stdout"
        )
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise DecodedEvaluationLauncherError(
            "launcher stdout is not exactly one complete line"
        )
    try:
        literal = stdout[:-1].decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DecodedEvaluationLauncherError(
            "launcher stdout is not UTF-8"
        ) from error
    decoded = _decode_canonical_json_literal(
        literal, label="launch manifest anchor"
    )
    return validate_launch_manifest_anchor(
        decoded, launch_manifest=launch_manifest
    )


def load_pinned_launch_manifest(
    path_value: str | Path,
    *,
    expected_sha256: str,
    work_root_binding: Mapping[str, Any],
    expected_anchor: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture a pinned launch manifest beneath the retained signed work root."""

    manifest_path = _canonical_absolute(
        str(path_value), label="pinned launch manifest"
    )
    expected_sha = _sha(
        expected_sha256, label="pinned launch manifest SHA"
    )
    try:
        work = (
            executor.bridge.verified_release
            .validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        )
    except executor.bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error
    work_root = Path(work["path"])
    launch_root = manifest_path.parent
    if (
        manifest_path.name != FILENAME
        or launch_root.parent != work_root
        or launch_root.name in ("", ".", "..")
    ):
        raise DecodedEvaluationLauncherError(
            "pinned launch manifest is not in one direct work-root child"
        )

    work_descriptor: int | None = None
    work_parent_descriptor: int | None = None
    launch_descriptor: int | None = None
    manifest_descriptor: int | None = None
    evaluation_root_handle: Any = None
    try:
        work_descriptor, work_parent_descriptor = (
            _duplicate_noninheritable_pair(
                work["root_fd"], work["parent_fd"],
                label="pinned launch work-root",
            )
        )
        work_identity = _retained_work_root_snapshot(
            work_root=work_root,
            root_descriptor=work_descriptor,
            parent_descriptor=work_parent_descriptor,
            expected_root_immutable=work["root_immutable_identity"],
            expected_parent_immutable=work["parent_immutable_identity"],
        )
        launch_descriptor = _open_directory_at(
            work_descriptor,
            launch_root.name,
            label="pinned launch root",
        )
        os.set_inheritable(launch_descriptor, False)
        launch_identity = _directory_snapshot(
            launch_descriptor,
            label="pinned launch root",
            expected_mode=0o555,
        )
        _replay_directory_at(
            parent_descriptor=work_descriptor,
            name=launch_root.name,
            descriptor=launch_descriptor,
            expected=launch_identity,
            label="pinned launch root",
        )
        _exact_entries(
            launch_descriptor, {FILENAME}, label="pinned launch root"
        )
        manifest_descriptor = os.open(
            FILENAME,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=launch_descriptor,
        )
        os.set_inheritable(manifest_descriptor, False)
        before = os.fstat(manifest_descriptor)
        if expected_anchor is not None:
            anchor_identity = expected_anchor.get("identity")
            if (
                not isinstance(anchor_identity, Mapping)
                or _stat_identity_row(before) != dict(anchor_identity)
                or expected_anchor.get("path") != str(manifest_path)
                or expected_anchor.get("sha256") != expected_sha
            ):
                raise DecodedEvaluationLauncherError(
                    "pinned launch manifest anchor identity differs"
                )
        manifest_identity = _file_snapshot(
            manifest_descriptor, payload_size=before.st_size
        )
        first = _read_descriptor(manifest_descriptor)
        middle = os.fstat(manifest_descriptor)
        _publication_barrier("aggregate_launch_after_first_manifest_read")
        second = _read_descriptor(manifest_descriptor)
        after = os.fstat(manifest_descriptor)
        if (
            first != second
            or not first.endswith(b"\n")
            or first.count(b"\n") != 1
            or len(first) != before.st_size
            or _identity(middle) != manifest_identity
            or _identity(after) != manifest_identity
            or hashlib.sha256(first).hexdigest() != expected_sha
        ):
            raise DecodedEvaluationLauncherError(
                "pinned launch manifest same-FD bytes or SHA differs"
            )
        try:
            literal = first[:-1].decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise DecodedEvaluationLauncherError(
                "pinned launch manifest is not UTF-8"
            ) from error
        decoded = _decode_canonical_json_literal(
            literal, label="pinned launch manifest"
        )
        try:
            loaded_bundle = executor.load_published_bundle(
                decoded.get("evaluation_root"),
                work_root_binding=work,
            )
        except (TypeError, executor.DecodedEvaluationExecutorError) as error:
            raise DecodedEvaluationLauncherError(
                "pinned launch manifest evaluation bundle differs"
            ) from error
        evaluation_root_handle = loaded_bundle.get("_evaluation_root_handle")
        bundle = {
            key: item for key, item in loaded_bundle.items()
            if not key.startswith("_")
        }
        row = validate_launch_manifest(decoded, bundle=bundle)
        if expected_anchor is not None:
            validated_anchor = validate_launch_manifest_anchor(
                expected_anchor, launch_manifest=row
            )
            if (
                validated_anchor["path"] != str(manifest_path)
                or validated_anchor["sha256"] != expected_sha
            ):
                raise DecodedEvaluationLauncherError(
                    "pinned launch manifest anchor literal differs"
                )
        if row["launch_root"] != str(launch_root):
            raise DecodedEvaluationLauncherError(
                "pinned launch manifest root binding differs"
            )
        _publication_barrier("aggregate_launch_before_final_manifest_replay")
        _retained_work_root_snapshot(
            work_root=work_root,
            root_descriptor=work_descriptor,
            parent_descriptor=work_parent_descriptor,
            expected_root_immutable=work["root_immutable_identity"],
            expected_parent_immutable=work["parent_immutable_identity"],
            expected_identity=work_identity,
        )
        _replay_directory_at(
            parent_descriptor=work_descriptor,
            name=launch_root.name,
            descriptor=launch_descriptor,
            expected=launch_identity,
            label="pinned launch root",
        )
        _replay_file_at(
            directory_descriptor=launch_descriptor,
            descriptor=manifest_descriptor,
            expected=manifest_identity,
            payload_size=len(first),
        )
        _exact_entries(
            launch_descriptor, {FILENAME}, label="pinned launch root"
        )
        if _read_descriptor(manifest_descriptor) != first:
            raise DecodedEvaluationLauncherError(
                "pinned launch manifest final same-FD reread differs"
            )
        return row, bundle
    finally:
        if evaluation_root_handle is not None:
            evaluation_root_handle.close()
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        if launch_descriptor is not None:
            os.close(launch_descriptor)
        if work_descriptor is not None:
            os.close(work_descriptor)
        if work_parent_descriptor is not None:
            os.close(work_parent_descriptor)


def _aggregate_target_arguments(
    launch_manifest: Mapping[str, Any],
    *,
    dynamic_authority: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[str]:
    return [
        "--evaluation-root", launch_manifest["evaluation_root"],
        "--physical-bindings",
        launch_manifest["tools"]["physical_bindings"]["path"],
        "--physical-bindings-sha256",
        launch_manifest["tools"]["physical_bindings"]["sha256"],
        "--blinding-key-file", launch_manifest["blinding_key"]["path"],
        "--blinding-key-sha256", launch_manifest["blinding_key"]["sha256"],
        "--aggregate-runtime-capture-receipt",
        launch_manifest["aggregate_runtime_capture_receipt_path"],
        "--aggregate-root", launch_manifest["aggregate_root"],
        *holder_completion_anchor_arguments(
            dynamic_authority, bundle=bundle
        ),
    ]


def build_aggregate_command_plan(
    *,
    launch_manifest: Mapping[str, Any],
    launch_manifest_path: str | Path,
    launch_manifest_sha256: str,
    launch_manifest_anchor: Mapping[str, Any],
    holder_completion_anchor_literals: Sequence[str],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    launch = validate_launch_manifest(launch_manifest, bundle=bundle)
    if launch["tool_files_verified"] is not True:
        raise DecodedEvaluationLauncherError(
            "aggregate launch requires a production launch manifest"
        )
    manifest_path = _canonical_absolute(
        str(launch_manifest_path), label="aggregate launch manifest"
    )
    manifest_sha = _sha(
        launch_manifest_sha256, label="aggregate launch manifest SHA"
    )
    if (
        manifest_path != Path(launch["launch_root"]) / FILENAME
        or hashlib.sha256(
            canonical_json_bytes(launch) + b"\n"
        ).hexdigest() != manifest_sha
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate launch manifest literal binding differs"
        )
    manifest_anchor = validate_launch_manifest_anchor(
        launch_manifest_anchor, launch_manifest=launch
    )
    if (
        manifest_anchor["path"] != str(manifest_path)
        or manifest_anchor["sha256"] != manifest_sha
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate launch manifest anchor differs"
        )
    manifest_anchor_literal = canonical_json_bytes(manifest_anchor).decode(
        "utf-8"
    )
    dynamic = build_holder_completion_dynamic_authority(
        holder_completion_anchor_literals, bundle=bundle
    )
    canonical_literals = [
        canonical_json_bytes(anchor).decode("utf-8")
        for anchor in dynamic["anchors"]
    ]
    target_arguments = _aggregate_target_arguments(
        launch, dynamic_authority=dynamic, bundle=bundle
    )
    argv = _controller_target_argv(
        root_python=launch["verified_runtime"]["root_python"],
        controller=launch["verified_runtime"]["controller"],
        deployment_authority=launch["deployment_authority"],
        target=AGGREGATE_TARGET,
        target_arguments=target_arguments,
        capture_receipt_path=launch[
            "aggregate_runtime_capture_receipt_path"
        ],
    )
    value = {
        "schema_version": AGGREGATE_COMMAND_PLAN_SCHEMA,
        "launch_manifest_anchor": manifest_anchor,
        "launch_manifest_anchor_literal": manifest_anchor_literal,
        "holder_completion_dynamic_authority": dynamic,
        "holder_completion_anchor_literals": canonical_literals,
        "aggregate_completion_anchor_channel": (
            _aggregate_completion_anchor_channel()
        ),
        "argv": argv,
        "shell": False,
        "execution_transport": "external_trusted_orchestrator",
        "external_trusted_orchestrator_required": True,
        "stdout_exact_one_canonical_plan_line_required": True,
        "command_execution_performed": False,
        "subprocess_spawned": False,
    }
    value["plan_digest"] = object_sha256(value)
    return validate_aggregate_command_plan(
        value, bundle=bundle, launch_manifest=launch
    )


def validate_aggregate_command_plan(
    value: Any, *, bundle: Mapping[str, Any],
    launch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version", "launch_manifest_anchor",
        "launch_manifest_anchor_literal",
        "holder_completion_dynamic_authority",
        "holder_completion_anchor_literals",
        "aggregate_completion_anchor_channel", "argv", "shell",
        "execution_transport", "external_trusted_orchestrator_required",
        "stdout_exact_one_canonical_plan_line_required",
        "command_execution_performed", "subprocess_spawned", "plan_digest",
    }
    row = dict(_closed(value, fields, label="aggregate command plan"))
    launch = validate_launch_manifest(launch_manifest, bundle=bundle)
    launch_binding = validate_launch_manifest_anchor(
        row["launch_manifest_anchor"], launch_manifest=launch
    )
    if (
        launch_binding["path"] != str(Path(launch["launch_root"]) / FILENAME)
        or launch_binding["sha256"]
        != hashlib.sha256(canonical_json_bytes(launch) + b"\n").hexdigest()
        or launch_binding["launch_manifest_digest"]
        != launch["launch_manifest_digest"]
        or row["launch_manifest_anchor_literal"]
        != canonical_json_bytes(launch_binding).decode("utf-8")
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate command launch manifest binding differs"
        )
    dynamic = validate_holder_completion_dynamic_authority(
        row["holder_completion_dynamic_authority"], bundle=bundle
    )
    expected_literals = [
        canonical_json_bytes(anchor).decode("utf-8")
        for anchor in dynamic["anchors"]
    ]
    if row["holder_completion_anchor_literals"] != expected_literals:
        raise DecodedEvaluationLauncherError(
            "aggregate command anchor literals differ"
        )
    if row["aggregate_completion_anchor_channel"] != (
        _aggregate_completion_anchor_channel()
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate completion anchor channel differs"
        )
    expected_scalars = {
        "schema_version": AGGREGATE_COMMAND_PLAN_SCHEMA,
        "shell": False,
        "execution_transport": "external_trusted_orchestrator",
        "external_trusted_orchestrator_required": True,
        "stdout_exact_one_canonical_plan_line_required": True,
        "command_execution_performed": False,
        "subprocess_spawned": False,
    }
    if any(row[key] != expected for key, expected in expected_scalars.items()):
        raise DecodedEvaluationLauncherError(
            "aggregate command plan policy differs"
        )
    expected_argv = _controller_target_argv(
        root_python=launch["verified_runtime"]["root_python"],
        controller=launch["verified_runtime"]["controller"],
        deployment_authority=launch["deployment_authority"],
        target=AGGREGATE_TARGET,
        target_arguments=_aggregate_target_arguments(
            launch, dynamic_authority=dynamic, bundle=bundle
        ),
        capture_receipt_path=launch[
            "aggregate_runtime_capture_receipt_path"
        ],
    )
    if (
        not isinstance(row["argv"], list)
        or any(type(item) is not str for item in row["argv"])
        or row["argv"] != expected_argv
        or row["argv"].count("--holder-completion-anchor")
        != len(plan.HOLDER_ROWS)
        or any(
            literal not in row["argv"]
            for literal in expected_literals
        )
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate command argv anchor closure differs"
        )
    claimed = _sha(row["plan_digest"], label="aggregate command plan digest")
    unsigned = dict(row)
    unsigned.pop("plan_digest")
    if claimed != object_sha256(unsigned):
        raise DecodedEvaluationLauncherError(
            "aggregate command plan digest differs"
        )
    return row


def parse_aggregate_command_plan_stdout(
    stdout: Any,
    *,
    return_code: Any,
    bundle: Mapping[str, Any],
    launch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if type(return_code) is not int or return_code != 0 or type(stdout) is not bytes:
        raise DecodedEvaluationLauncherError(
            "aggregate launcher did not exit successfully with bytes stdout"
        )
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise DecodedEvaluationLauncherError(
            "aggregate launcher stdout is not exactly one complete line"
        )
    try:
        literal = stdout[:-1].decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise DecodedEvaluationLauncherError(
            "aggregate launcher stdout is not UTF-8"
        ) from error
    decoded = _decode_canonical_json_literal(
        literal, label="aggregate command plan"
    )
    return validate_aggregate_command_plan(
        decoded, bundle=bundle, launch_manifest=launch_manifest
    )


def publish_launch_manifest_authorized(
    value: Mapping[str, Any], *, bundle: Mapping[str, Any],
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = validate_launch_manifest(value, bundle=bundle)
    root = Path(row["launch_root"])
    payload = canonical_json_bytes(row) + b"\n"
    output = root / FILENAME
    retained_parent_descriptor: int | None = None
    if row["tool_files_verified"]:
        if work_root_binding is None:
            raise DecodedEvaluationLauncherError(
                "production launch publication requires signed work-root FDs"
            )
        try:
            work = (
                executor.bridge.verified_release
                .validate_inherited_work_root_binding(
                    work_root_binding,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            )
        except executor.bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationLauncherError(str(error)) from error
        if (
            _launcher_work_root_projection(work) != row["launcher_work_root"]
            or root.parent != Path(work["path"])
            or root.name in ("", ".", "..")
        ):
            raise DecodedEvaluationLauncherError(
                "launch root is not a direct signed work-root member"
            )
        parent_descriptor, retained_parent_descriptor = (
            _duplicate_noninheritable_pair(
                work["root_fd"], work["parent_fd"],
                label="launch publication work-root",
            )
        )
        try:
            expected_root_immutable = work["root_immutable_identity"]
            expected_parent_immutable = work["parent_immutable_identity"]
            parent_identity = _retained_work_root_snapshot(
                work_root=Path(work["path"]),
                root_descriptor=parent_descriptor,
                parent_descriptor=retained_parent_descriptor,
                expected_root_immutable=expected_root_immutable,
                expected_parent_immutable=expected_parent_immutable,
            )
        except Exception:
            os.close(parent_descriptor)
            os.close(retained_parent_descriptor)
            raise

        def replay_parent(expected: tuple[int, ...]) -> None:
            _retained_work_root_snapshot(
                work_root=Path(work["path"]),
                root_descriptor=parent_descriptor,
                parent_descriptor=retained_parent_descriptor,
                expected_root_immutable=expected_root_immutable,
                expected_parent_immutable=expected_parent_immutable,
                expected_identity=expected,
            )
    else:
        if work_root_binding is not None:
            raise DecodedEvaluationLauncherError(
                "synthetic launch publication received signed work-root FDs"
            )
        parent_descriptor = _open_directory(
            root.parent, label="launch root parent"
        )
        try:
            parent_identity = _require_canonical_parent(
                root.parent, parent_descriptor
            )
        except Exception:
            os.close(parent_descriptor)
            raise

        def replay_parent(expected: tuple[int, ...]) -> None:
            _replay_canonical_parent(
                root.parent, parent_descriptor, expected
            )

    root_descriptor: int | None = None
    manifest_descriptor: int | None = None
    published_anchor: dict[str, Any] | None = None
    try:
        _relative_entry_absent(parent_descriptor, root.name, label="launch root")
        _publication_barrier("before_root_mkdir")
        replay_parent(parent_identity)
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise DecodedEvaluationLauncherError(
                f"launch root is not fresh: {root}"
            ) from error
        except OSError as error:
            raise DecodedEvaluationLauncherError(
                f"cannot create launch root: {error}"
            ) from error
        root_descriptor = _open_directory_at(
            parent_descriptor, root.name, label="launch root"
        )
        os.fchmod(root_descriptor, 0o700)
        os.fsync(root_descriptor)
        root_identity = _directory_snapshot(
            root_descriptor, label="launch root", expected_mode=0o700
        )
        if retained_parent_descriptor is not None:
            parent_identity = _retained_work_root_snapshot(
                work_root=root.parent,
                root_descriptor=parent_descriptor,
                parent_descriptor=retained_parent_descriptor,
                expected_root_immutable=expected_root_immutable,
                expected_parent_immutable=expected_parent_immutable,
            )
        _publication_barrier("after_root_open")
        replay_parent(parent_identity)
        _replay_directory_at(
            parent_descriptor=parent_descriptor,
            name=root.name,
            descriptor=root_descriptor,
            expected=root_identity,
            label="launch root",
        )

        _relative_entry_absent(root_descriptor, FILENAME, label="launch manifest")
        _publication_barrier("before_manifest_create")
        _replay_directory_at(
            parent_descriptor=parent_descriptor,
            name=root.name,
            descriptor=root_descriptor,
            expected=root_identity,
            label="launch root",
        )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            manifest_descriptor = os.open(
                FILENAME, flags, 0o400, dir_fd=root_descriptor
            )
        except FileExistsError as error:
            raise DecodedEvaluationLauncherError(
                "launch manifest is not fresh"
            ) from error
        except OSError as error:
            raise DecodedEvaluationLauncherError(
                f"cannot create launch manifest: {error}"
            ) from error
        offset = 0
        while offset < len(payload):
            written = os.write(manifest_descriptor, payload[offset:])
            if written <= 0:
                raise DecodedEvaluationLauncherError("launch manifest write made no progress")
            offset += written
        os.fchmod(manifest_descriptor, 0o400)
        os.fsync(manifest_descriptor)
        manifest_identity = _file_snapshot(
            manifest_descriptor, payload_size=len(payload)
        )
        root_identity = _directory_snapshot(
            root_descriptor, label="launch root", expected_mode=0o700
        )
        _publication_barrier("after_manifest_sync")
        _replay_file_at(
            directory_descriptor=root_descriptor,
            descriptor=manifest_descriptor,
            expected=manifest_identity,
            payload_size=len(payload),
        )
        first = _read_descriptor(manifest_descriptor)
        middle_identity = _identity(os.fstat(manifest_descriptor))
        second = _read_descriptor(manifest_descriptor)
        if (
            first != payload
            or second != payload
            or middle_identity != manifest_identity
            or _identity(os.fstat(manifest_descriptor)) != manifest_identity
        ):
            raise DecodedEvaluationLauncherError(
                "launch manifest same-FD double reread differs"
            )
        _replay_file_at(
            directory_descriptor=root_descriptor,
            descriptor=manifest_descriptor,
            expected=manifest_identity,
            payload_size=len(payload),
        )
        _exact_entries(root_descriptor, {FILENAME}, label="launch root")
        _replay_directory_at(
            parent_descriptor=parent_descriptor,
            name=root.name,
            descriptor=root_descriptor,
            expected=root_identity,
            label="launch root",
        )

        os.fsync(root_descriptor)
        os.fsync(parent_descriptor)
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
        os.fsync(parent_descriptor)
        root_identity = _directory_snapshot(
            root_descriptor, label="launch root", expected_mode=0o555
        )
        _publication_barrier("after_seal_before_final_replay")

        replay_parent(parent_identity)
        _replay_directory_at(
            parent_descriptor=parent_descriptor,
            name=root.name,
            descriptor=root_descriptor,
            expected=root_identity,
            label="launch root",
        )
        _replay_file_at(
            directory_descriptor=root_descriptor,
            descriptor=manifest_descriptor,
            expected=manifest_identity,
            payload_size=len(payload),
        )
        _exact_entries(root_descriptor, {FILENAME}, label="launch root")
        if _read_descriptor(manifest_descriptor) != payload:
            raise DecodedEvaluationLauncherError(
                "launch manifest final same-FD reread differs"
            )
        replay_parent(parent_identity)
        _replay_directory_at(
            parent_descriptor=parent_descriptor,
            name=root.name,
            descriptor=root_descriptor,
            expected=root_identity,
            label="launch root",
        )
        _replay_file_at(
            directory_descriptor=root_descriptor,
            descriptor=manifest_descriptor,
            expected=manifest_identity,
            payload_size=len(payload),
        )
        final_manifest_info = os.fstat(manifest_descriptor)
        if _identity(final_manifest_info) != manifest_identity:
            raise DecodedEvaluationLauncherError(
                "launch manifest final held identity differs"
            )
        published_anchor = build_launch_manifest_anchor(
            launch_manifest=row,
            path=output,
            identity=_stat_identity_row(final_manifest_info),
        )
    finally:
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
        if retained_parent_descriptor is not None:
            os.close(retained_parent_descriptor)
    if published_anchor is None:
        raise DecodedEvaluationLauncherError(
            "launch manifest anchor was not captured"
        )
    return {
        "output": str(output),
        "launch_manifest_anchor": published_anchor,
    }


def publish_launch_manifest(
    value: Mapping[str, Any], *, bundle: Mapping[str, Any],
    work_root_binding: Mapping[str, Any] | None = None,
) -> Path:
    result = publish_launch_manifest_authorized(
        value, bundle=bundle, work_root_binding=work_root_binding
    )
    return Path(result["output"])


def _phase_one_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--launch-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--executor-sha256", required=True)
    parser.add_argument("--decoder-adapter", required=True)
    parser.add_argument("--decoder-adapter-sha256", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffprobe-sha256", required=True)
    parser.add_argument("--physical-bindings", required=True)
    parser.add_argument("--physical-bindings-sha256", required=True)
    parser.add_argument("--blinding-key-file", required=True)
    parser.add_argument("--blinding-key-sha256", required=True)
    parser.add_argument("--aggregate-root", required=True)
    args = parser.parse_args(argv)
    try:
        inherited_work_root = (
            executor.bridge.verified_release
            .load_inherited_work_root_environment(
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        )
    except executor.bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error
    if inherited_work_root["target"] != (
        "action_preservation_decoded_eval_launcher_v1.py"
    ):
        raise DecodedEvaluationLauncherError(
            "launcher inherited work-root target differs"
        )
    bundle = executor.load_published_bundle(
        args.evaluation_root, work_root_binding=inherited_work_root
    )

    def identity(path_value: str, sha_value: str) -> dict[str, str]:
        return {
            "path": str(Path(path_value).resolve(strict=True)),
            "sha256": sha_value,
        }

    physical_identity = identity(
        args.physical_bindings, args.physical_bindings_sha256
    )
    try:
        bindings = executor.bridge.load_physical_bindings(
            physical_identity["path"],
            expected_sha256=physical_identity["sha256"],
            verify_files=True,
        )
        launcher_capture = executor.bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_launcher_v1.py",
            expected_arguments=list(sys.argv[1:] if argv is None else argv),
            verify_file=True,
            work_root_binding=inherited_work_root,
        )
    except executor.bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error

    value = build_launch_manifest(
        bundle=bundle,
        launch_root=args.launch_root,
        python_identity=identity(args.python, args.python_sha256),
        executor_identity=identity(args.executor, args.executor_sha256),
        decoder_identity=identity(args.decoder_adapter, args.decoder_adapter_sha256),
        ffprobe_identity=identity(args.ffprobe, args.ffprobe_sha256),
        physical_bindings_identity=physical_identity,
        blinding_key_identity={
            "path": args.blinding_key_file,
            "sha256": args.blinding_key_sha256,
        },
        aggregate_root=args.aggregate_root,
        verify_tools=True,
        launcher_verified_release_capture=launcher_capture,
        launcher_work_root_binding=inherited_work_root,
    )
    published = publish_launch_manifest_authorized(
        value, bundle=bundle, work_root_binding=inherited_work_root
    )
    sys.stdout.buffer.write(
        canonical_json_bytes(published["launch_manifest_anchor"]) + b"\n"
    )
    sys.stdout.buffer.flush()
    return 0


def aggregate_launch_main(
    argv: Sequence[str] | None = None,
    *,
    invocation_arguments: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate four online holder anchors and emit one aggregate "
            "detached-controller command plan without executing it."
        )
    )
    parser.add_argument("--launch-manifest", required=True)
    parser.add_argument("--launch-manifest-sha256", required=True)
    parser.add_argument("--launch-manifest-anchor", required=True)
    parser.add_argument(
        "--holder-completion-anchor", action="append", required=True
    )
    args = parser.parse_args(argv)
    if len(args.holder_completion_anchor) != len(plan.HOLDER_ROWS):
        raise DecodedEvaluationLauncherError(
            "aggregate launch requires exactly four holder anchors"
        )
    manifest_anchor = _decode_canonical_json_literal(
        args.launch_manifest_anchor, label="launch manifest anchor"
    )
    if (
        manifest_anchor.get("path") != args.launch_manifest
        or manifest_anchor.get("sha256") != args.launch_manifest_sha256
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate launch manifest anchor arguments differ"
        )
    try:
        inherited_work_root = (
            executor.bridge.verified_release
            .load_inherited_work_root_environment(
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        )
    except executor.bridge.verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error
    if inherited_work_root["target"] != (
        "action_preservation_decoded_eval_launcher_v1.py"
    ):
        raise DecodedEvaluationLauncherError(
            "aggregate launcher inherited work-root target differs"
        )
    launch, bundle = load_pinned_launch_manifest(
        args.launch_manifest,
        expected_sha256=args.launch_manifest_sha256,
        work_root_binding=inherited_work_root,
        expected_anchor=manifest_anchor,
    )
    expected_arguments = list(
        invocation_arguments
        if invocation_arguments is not None
        else ["aggregate-launch", *(argv or ())]
    )
    try:
        bindings = executor.bridge.load_physical_bindings(
            launch["tools"]["physical_bindings"]["path"],
            expected_sha256=launch["tools"]["physical_bindings"]["sha256"],
            verify_files=True,
        )
        current_capture = executor.bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_launcher_v1.py",
            expected_arguments=expected_arguments,
            verify_file=True,
            work_root_binding=inherited_work_root,
        )
    except executor.bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationLauncherError(str(error)) from error
    if current_capture["receipt_path"] in {
        launch["launcher_verified_release_capture"]["receipt_path"],
        launch["aggregate_runtime_capture_receipt_path"],
    }:
        raise DecodedEvaluationLauncherError(
            "aggregate launcher verified capture path is not distinct"
        )
    command_plan = build_aggregate_command_plan(
        launch_manifest=launch,
        launch_manifest_path=args.launch_manifest,
        launch_manifest_sha256=args.launch_manifest_sha256,
        launch_manifest_anchor=manifest_anchor,
        holder_completion_anchor_literals=args.holder_completion_anchor,
        bundle=bundle,
    )
    sys.stdout.buffer.write(canonical_json_bytes(command_plan) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["aggregate-launch"]:
        return aggregate_launch_main(
            arguments[1:], invocation_arguments=arguments
        )
    return _phase_one_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
