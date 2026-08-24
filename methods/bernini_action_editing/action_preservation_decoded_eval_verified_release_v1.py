#!/usr/bin/env python3
"""Stdlib-only verified runtime for the preservation-v2 decoded evaluation.

The deployment envelope, manifest, archive, materialized tree, and interpreter
are captured through retained descriptors.  Release Python imports execute
only from the captured byte closure, including the ``tools`` namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import tarfile
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = "bernini-action-preservation-decoded-eval-source-release-v2"
ENVELOPE_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-v1"
CAPTURE_RECEIPT_SCHEMA = (
    "bernini-action-preservation-decoded-eval-runtime-capture-v4"
)
CONTROLLER_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-controller-authority-v1"
)
RELEASE_GENERATION = "preservation-v2-decoded-eval-exact15-r3"
ARCHIVE_FORMAT = "fixed-ustar-ascii-zero-dev-sorted-owner0-mtime0-record10240-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
FIXED_USTAR_BLOCK_SIZE = 512
FIXED_USTAR_RECORD_SIZE = 10240
CAPTURE_DIGEST_ENV = "APV2_EVAL_RELEASE_CAPTURE_DIGEST"
CAPTURE_RECEIPT_ENV = "APV2_EVAL_RELEASE_CAPTURE_RECEIPT"
BOOTSTRAP_IDENTITY_ENV = "APV2_EVAL_FROZEN_PYTHON_IDENTITY"
CONTROLLER_AUTHORITY_DIGEST_ENV = "APV2_EVAL_CONTROLLER_AUTHORITY_DIGEST"
TORCHRUN_BINDING_ENV = "APV2_EVAL_CAPTURED_TORCHRUN_BINDING"
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
WORK_ROOT_BINDING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-inherited-work-root-v2"
)
WORK_ROOT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-work-root-authority-v1"
)
TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH = (
    "torch/distributed/elastic/multiprocessing/subprocess_handler/"
    "subprocess_handler.py"
)
TORCHRUN_SUBPROCESS_HANDLER_SHA256 = (
    "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87"
)
TORCHRUN_SUBPROCESS_HANDLER_SIZE = 2436
TORCHRUN_SOURCE_SHA256 = (
    "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
)
TORCHRUN_SOURCE_SIZE = 31587

EVAL_RELEASE_MEMBERS = tuple(sorted({
    "action_preservation_decoded_eval_aggregate_v2.py",
    "action_preservation_decoded_eval_bridge_v1.py",
    "action_preservation_decoded_eval_decoder_adapter_v1.py",
    "action_preservation_decoded_eval_executor_v2.py",
    "action_preservation_decoded_eval_launcher_v1.py",
    "action_preservation_decoded_eval_model_authority_v2.py",
    "action_preservation_decoded_eval_plan_v1.py",
    "action_preservation_decoded_eval_verified_release_v1.py",
    "action_preservation_gate_v1.py",
    "action_preservation_loop_controller_v1.py",
    "infer_lora.py",
    "self_generated_action_preservation_v2.py",
    "tools/build_renderer_dataset.py",
    "tools/materialize_vae.py",
    "train_lora.py",
}))
EXECUTABLE_MEMBER = "action_preservation_decoded_eval_decoder_adapter_v1.py"
MEMBER_MODES = {
    relative: (0o555 if relative == EXECUTABLE_MEMBER else 0o444)
    for relative in EVAL_RELEASE_MEMBERS
}
ALLOWED_PYTHON_TARGETS = frozenset(
    {
        "infer_lora.py",
        "action_preservation_gate_v1.py",
        "action_preservation_decoded_eval_plan_v1.py",
        "action_preservation_decoded_eval_bridge_v1.py",
        "action_preservation_decoded_eval_decoder_adapter_v1.py",
        "action_preservation_decoded_eval_executor_v2.py",
        "action_preservation_decoded_eval_launcher_v1.py",
        "action_preservation_decoded_eval_aggregate_v2.py",
        "action_preservation_loop_controller_v1.py",
        "tools/materialize_vae.py",
        "tools/build_renderer_dataset.py",
    }
)

# The independently reviewed exact15 non-runtime closure is a second trust
# anchor.  The runtime's own row remains anchored by the externally pinned
# manifest because a source file cannot contain its ordinary SHA-256 as a
# fixed-point literal.  The successor packager pins all fifteen rows,
# including this runtime.
TRUSTED_EXACT15: Mapping[str, Tuple[str, int, int]] = {
    "infer_lora.py": (
        "dde5e3293e4fc833618c970eb51ba61fef4c66ef38dd1e67ab0e12b142f05e48", 95828, 0o444
    ),
    "train_lora.py": (
        "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e", 84216, 0o444
    ),
    "self_generated_action_preservation_v2.py": (
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c", 11334, 0o444
    ),
    "action_preservation_gate_v1.py": (
        "2c5e6d2a2e64b59c620b581aab38f243e9d7d0a53e764787fe013f0eede4f844", 78097, 0o444
    ),
    "action_preservation_decoded_eval_plan_v1.py": (
        "287efb71142c91bd0ad78354f6f72948a7aebc5b746c96fb32f701aa7158072b", 49347, 0o444
    ),
    "action_preservation_decoded_eval_bridge_v1.py": (
        "91248c78cb03b290b6f12ad4d39bcd0942a21ae3efe8143b3c14ba8e9834cfe4", 101949, 0o444
    ),
    "action_preservation_decoded_eval_decoder_adapter_v1.py": (
        "0b30ff6d2e4d17b20844abbeea5c26e51d376740cab092f905854279ad713fd1", 38381, 0o555
    ),
    "action_preservation_decoded_eval_executor_v2.py": (
        "8915693b5816d7309e9f66f5a2b08975e579286c6df9e8ea410791e0ad3cce29", 105577, 0o444
    ),
    "action_preservation_decoded_eval_launcher_v1.py": (
        "3646bd09a6f1054d4d5664f8f1ea818a8c1254873acfd97f689afef0aa0c2280", 27965, 0o444
    ),
    "action_preservation_decoded_eval_aggregate_v2.py": (
        "88d909f188372c588a5eac7ddd9d2edae278ba264b550c14656fbe48fc40b963", 109226, 0o444
    ),
    "action_preservation_loop_controller_v1.py": (
        "b070cd82c11251b9b638ff1f39a3c346e8347a0137b8b1e17f8aa2a67661db6c", 49068, 0o444
    ),
    "tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0", 32195, 0o444
    ),
    "tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5", 31012, 0o444
    ),
    "action_preservation_decoded_eval_model_authority_v2.py": (
        "6ba965cfc81e073025a918a060a5aebceee836bf9d748d180924c98391b68f16", 78592, 0o444
    ),
}

AUTHORITY = {
    "evaluation_kind": "preservation-v2-full-video-decoded-exact264",
    "candidate_count": 264,
    "full_video_frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "source_identity_background_camera_are_conjunctive": True,
    "training_loss_is_not_evaluation_evidence": True,
    "missing_calibration_requires_abstain": True,
    "distinct_blind_reviewers_required": 2,
    "automatic_scientific_promotion_authorized": False,
}
MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "release_generation", "archive_format", "member_root",
        "exact_member_closure", "file_count", "files", "content_revision",
        "allowed_entrypoints", "authority", "component_sha256", "manifest_digest",
    }
)
ENVELOPE_FIELDS = frozenset(
    {
        "schema_version", "release_generation", "remote_release_exact_entries",
        "source_archive", "source_manifest", "create_only_deployment_required",
        "fresh_materialized_root_required", "verified_runtime_required",
        "detached_controller_authority_receipt_required",
        "automatic_scientific_promotion_authorized", "envelope_digest",
    }
)
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class DecodedEvalVerifiedReleaseError(RuntimeError):
    """Raised before unverified release bytes can be executed."""


def fail(message: str) -> NoReturn:
    raise DecodedEvalVerifiedReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "release JSON is not canonical finite UTF-8"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_revision(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha1(canonical_json_bytes(list(rows))).hexdigest()


def _identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode,
        value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _identity_value(value: os.stat_result) -> Dict[str, int]:
    return {
        "device": int(value.st_dev), "inode": int(value.st_ino),
        "uid": int(value.st_uid), "gid": int(value.st_gid),
        "mode": int(stat.S_IMODE(value.st_mode)), "nlink": int(value.st_nlink),
        "rdev": int(value.st_rdev), "size": int(value.st_size),
        "blocks": int(getattr(value, "st_blocks", 0)),
        "mtime_ns": int(value.st_mtime_ns), "ctime_ns": int(value.st_ctime_ns),
    }


_WORK_ROOT_IDENTITY_FIELDS = frozenset(
    {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
)
_WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS = frozenset(
    {"device", "inode", "uid", "gid", "mode", "rdev"}
)
_WORK_ROOT_BINDING_FIELDS = frozenset(
    {
        "schema_version", "path", "parent_path", "parent_fd", "root_fd",
        "parent_identity", "root_identity", "parent_immutable_identity",
        "root_immutable_identity", "entries", "work_root_authority_digest",
        "work_root_authority", "deployment_receipt",
        "source_spec_authority",
        "deployment_receipt_digest", "source_spec_authority_digest", "target",
        "capture_receipt_path", "exact_two_directory_fds",
        "fds_inheritable_only_across_verified_exec", "binding_digest",
    }
)
_COMPLETION_ANCHOR_CHANNEL_FIELDS = frozenset(
    {
        "schema_version", "descriptor", "controller_pid", "target_pid",
        "expected_target", "binding_digest",
    }
)
_HOLDER_COMPLETION_ANCHOR_FIELDS = frozenset(
    {
        "schema_version", "holder_job_id", "completion_path",
        "initial_inode_identity", "completion_sha256", "completion_size",
        "completion_mode", "completion_digest", "holder_summary_digest",
        "anchor_digest",
    }
)
_HOLDER_COMPLETION_INODE_FIELDS = frozenset(
    {"device", "inode", "uid", "gid", "rdev"}
)
_EXECUTOR_TARGET = "action_preservation_decoded_eval_executor_v2.py"
_AGGREGATE_TARGET = "action_preservation_decoded_eval_aggregate_v2.py"
_DYNAMIC_ANCHOR_TARGETS = frozenset({_EXECUTOR_TARGET, _AGGREGATE_TARGET})
_HOLDER_JOB_IDS = frozenset({"136719", "136141", "136309", "136140"})
_AGGREGATE_COMPLETION_ANCHOR_FIELDS = frozenset(
    {
        "schema_version", "evaluation_id", "aggregate_root",
        "aggregate_root_identity", "aggregate_file", "private_file",
        "public_file", "media_directory_identity", "media_file_count",
        "media_rows_digest", "media_tree_digest", "anchor_digest",
    }
)
_AGGREGATE_ANCHOR_FILE_FIELDS = frozenset(
    {"relative_path", "sha256", "size", "mode", "identity", "object_digest"}
)


def _validate_work_root_authority_shape(value: Any) -> Dict[str, Any]:
    fields = {
        "schema_version", "path", "parent_path", "creation_identity",
        "immutable_identity", "parent_immutable_identity", "initial_entries",
        "retained_parent_fd_through_request_publication",
        "retained_root_fd_through_request_publication", "authority_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("inherited work root authority closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    path = Path(row["path"]) if type(row.get("path")) is str else Path()
    parent = (
        Path(row["parent_path"])
        if type(row.get("parent_path")) is str else Path()
    )
    creation = _validate_work_root_identity(
        row.get("creation_identity"), label="work root authority creation"
    )
    immutable = _validate_work_root_identity(
        row.get("immutable_identity"),
        label="work root authority immutable root", immutable=True,
    )
    parent_immutable = _validate_work_root_identity(
        row.get("parent_immutable_identity"),
        label="work root authority immutable parent", immutable=True,
    )
    if (
        row.get("schema_version") != WORK_ROOT_AUTHORITY_SCHEMA
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not parent.is_absolute()
        or os.path.normpath(str(parent)) != str(parent)
        or path.parent != parent
        or stat.S_IMODE(creation["mode"]) != 0o700
        or immutable
        != {field: creation[field] for field in immutable}
        or row.get("initial_entries") != []
        or row.get("retained_parent_fd_through_request_publication") is not True
        or row.get("retained_root_fd_through_request_publication") is not True
        or type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        fail("inherited work root authority differs")
    row["creation_identity"] = creation
    row["immutable_identity"] = immutable
    row["parent_immutable_identity"] = parent_immutable
    return row


def _validate_work_root_file_pair(
    value: Any, *, root: Path, label: str,
) -> Dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != {"path", "sha256"}
        or type(value.get("path")) is not str
        or Path(value["path"]).parent != root
        or Path(value["path"]).name in ("", ".", "..")
        or type(value.get("sha256")) is not str
        or SHA256_RE.fullmatch(value["sha256"]) is None
    ):
        fail(f"inherited {label} file binding differs")
    return dict(value)


def _stable_work_root_file_pair(
    root_fd: int, value: Mapping[str, str], *, label: str,
) -> Tuple[bytes, os.stat_result]:
    name = Path(value["path"]).name
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            f"inherited {label} held replay is unavailable"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or hashlib.sha256(first).hexdigest() != value["sha256"]
    ):
        fail(f"inherited {label} held replay differs")
    return first, before


def _work_root_identity_value(value: os.stat_result) -> Dict[str, int]:
    return {
        "device": int(value.st_dev), "inode": int(value.st_ino),
        "uid": int(value.st_uid), "gid": int(value.st_gid),
        "mode": int(value.st_mode), "nlink": int(value.st_nlink),
        "rdev": int(value.st_rdev), "size": int(value.st_size),
        "blocks": int(getattr(value, "st_blocks", 0)),
        "mtime_ns": int(value.st_mtime_ns), "ctime_ns": int(value.st_ctime_ns),
    }


def _validate_work_root_identity(
    value: Any, *, label: str, immutable: bool = False,
) -> Dict[str, int]:
    fields = (
        _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
        if immutable else _WORK_ROOT_IDENTITY_FIELDS
    )
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(value[field]) is not int or value[field] < 0 for field in fields)
        or not stat.S_ISDIR(value["mode"])
    ):
        fail(f"{label} identity closure differs")
    return dict(value)


def validate_inherited_work_root_binding(
    value: Any,
    *,
    verify_open_fds: bool,
    expected_inheritable: Optional[bool] = None,
    verify_entries: bool = True,
    allow_root_metadata_change: bool = False,
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _WORK_ROOT_BINDING_FIELDS:
        fail("inherited work root binding field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("binding_digest", None)
    path = Path(row["path"]) if type(row.get("path")) is str else Path()
    parent_path = (
        Path(row["parent_path"])
        if type(row.get("parent_path")) is str else Path()
    )
    capture_path = (
        Path(row["capture_receipt_path"])
        if type(row.get("capture_receipt_path")) is str else Path()
    )
    work_root_authority = _validate_work_root_authority_shape(
        row.get("work_root_authority")
    )
    deployment_receipt = _validate_work_root_file_pair(
        row.get("deployment_receipt"), root=path,
        label="deployment receipt",
    )
    source_spec_authority = _validate_work_root_file_pair(
        row.get("source_spec_authority"), root=path,
        label="source spec authority",
    )
    parent_identity = _validate_work_root_identity(
        row.get("parent_identity"), label="inherited work root parent"
    )
    root_identity = _validate_work_root_identity(
        row.get("root_identity"), label="inherited work root"
    )
    parent_immutable = _validate_work_root_identity(
        row.get("parent_immutable_identity"),
        label="inherited work root immutable parent", immutable=True,
    )
    root_immutable = _validate_work_root_identity(
        row.get("root_immutable_identity"),
        label="inherited work root immutable root", immutable=True,
    )
    entries = row.get("entries")
    digest_fields = (
        "work_root_authority_digest", "deployment_receipt_digest",
        "source_spec_authority_digest", "binding_digest",
    )
    if (
        row.get("schema_version") != WORK_ROOT_BINDING_SCHEMA
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not parent_path.is_absolute()
        or os.path.normpath(str(parent_path)) != str(parent_path)
        or path.parent != parent_path
        or work_root_authority["path"] != str(path)
        or work_root_authority["parent_path"] != str(parent_path)
        or work_root_authority["authority_digest"]
        != row.get("work_root_authority_digest")
        or work_root_authority["immutable_identity"] != root_immutable
        or work_root_authority["parent_immutable_identity"] != parent_immutable
        or deployment_receipt["path"] == source_spec_authority["path"]
        or not capture_path.is_absolute()
        or capture_path.parent != path
        or capture_path.name in ("", ".", "..")
        or type(row.get("parent_fd")) is not int
        or type(row.get("root_fd")) is not int
        or row["parent_fd"] < 3
        or row["root_fd"] < 3
        or row["parent_fd"] == row["root_fd"]
        or type(entries) is not list
        or any(type(item) is not str or item in ("", ".", "..") for item in entries)
        or entries != sorted(set(entries))
        or Path(deployment_receipt["path"]).name not in entries
        or Path(source_spec_authority["path"]).name not in entries
        or capture_path.name in entries
        or row.get("target") not in ALLOWED_PYTHON_TARGETS
        or row.get("exact_two_directory_fds") is not True
        or row.get("fds_inheritable_only_across_verified_exec") is not True
        or any(
            type(row.get(field)) is not str
            or SHA256_RE.fullmatch(row[field]) is None
            for field in digest_fields
        )
        or claimed != object_sha256(unsigned)
        or {
            field: parent_identity[field]
            for field in _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
        } != parent_immutable
        or {
            field: root_identity[field]
            for field in _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
        } != root_immutable
    ):
        fail("inherited work root binding value or digest differs")
    row["work_root_authority"] = work_root_authority
    row["deployment_receipt"] = deployment_receipt
    row["source_spec_authority"] = source_spec_authority
    if verify_open_fds:
        try:
            parent_before = os.fstat(row["parent_fd"])
            root_before = os.fstat(row["root_fd"])
            first_entries = os.listdir(row["root_fd"])
            root_middle = os.fstat(row["root_fd"])
            second_entries = os.listdir(row["root_fd"])
            root_after = os.fstat(row["root_fd"])
            named_root = os.stat(
                path.name,
                dir_fd=row["parent_fd"],
                follow_symlinks=False,
            )
            parent_after = os.fstat(row["parent_fd"])
            named_parent = parent_path.lstat()
            parent_inheritable = os.get_inheritable(row["parent_fd"])
            root_inheritable = os.get_inheritable(row["root_fd"])
        except OSError as error:
            raise DecodedEvalVerifiedReleaseError(
                "inherited work root FD replay is unavailable"
            ) from error
        deployment_raw, _ = _stable_work_root_file_pair(
            row["root_fd"], deployment_receipt,
            label="deployment receipt",
        )
        source_authority_raw, _ = _stable_work_root_file_pair(
            row["root_fd"], source_spec_authority,
            label="source spec authority",
        )
        deployment_value = _decode_json(
            deployment_raw, label="inherited deployment receipt"
        )
        source_authority_value = _decode_json(
            source_authority_raw,
            label="inherited source spec authority",
        )
        if (
            deployment_value.get("receipt_digest")
            != row["deployment_receipt_digest"]
            or deployment_value.get("work_root_authority")
            != work_root_authority
            or source_authority_value.get("receipt_digest")
            != row["source_spec_authority_digest"]
            or source_authority_value.get("work_root_authority")
            != work_root_authority
            or source_authority_value.get("deployment_receipt_digest")
            != row["deployment_receipt_digest"]
        ):
            fail("inherited deployment/source authority continuity differs")
        observed_root_before = _work_root_identity_value(root_before)
        observed_root_middle = _work_root_identity_value(root_middle)
        observed_root_after = _work_root_identity_value(root_after)
        observed_named_root = _work_root_identity_value(named_root)
        root_replay_differs = (
            (
                not allow_root_metadata_change
                and (
                    observed_root_before != root_identity
                    or observed_root_middle != root_identity
                    or observed_root_after != root_identity
                    or observed_named_root != root_identity
                )
            )
            or (
                allow_root_metadata_change
                and (
                    observed_root_before != observed_root_middle
                    or observed_root_before != observed_root_after
                    or observed_root_before != observed_named_root
                    or {
                        field: observed_root_before[field]
                        for field in _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
                    } != root_immutable
                )
            )
        )
        if (
            _work_root_identity_value(parent_before) != parent_identity
            or _work_root_identity_value(parent_after) != parent_identity
            or _work_root_identity_value(named_parent) != parent_identity
            or root_replay_differs
            or sorted(first_entries) != sorted(second_entries)
            or (verify_entries and sorted(first_entries) != entries)
            or (
                expected_inheritable is not None
                and (
                    parent_inheritable is not expected_inheritable
                    or root_inheritable is not expected_inheritable
                )
            )
        ):
            fail("inherited work root FD identity or entries differ")
    return row


def load_inherited_work_root_environment(
    *,
    verify_open_fds: bool,
    expected_inheritable: Optional[bool] = None,
    verify_entries: bool = True,
    allow_root_metadata_change: bool = False,
) -> Dict[str, Any]:
    raw = os.environ.get(WORK_ROOT_BINDING_ENV)
    if raw is None:
        fail("inherited work root environment is absent")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "inherited work root environment is not JSON"
        ) from error
    if type(value) is not dict or canonical_json_bytes(value).decode("utf-8") != raw:
        fail("inherited work root environment is not canonical")
    return validate_inherited_work_root_binding(
        value,
        verify_open_fds=verify_open_fds,
        expected_inheritable=expected_inheritable,
        verify_entries=verify_entries,
        allow_root_metadata_change=allow_root_metadata_change,
    )


def seal_inherited_work_root_fds(value: Mapping[str, Any]) -> Dict[str, Any]:
    row = validate_inherited_work_root_binding(
        value, verify_open_fds=True, expected_inheritable=True,
    )
    os.set_inheritable(row["parent_fd"], False)
    os.set_inheritable(row["root_fd"], False)
    return validate_inherited_work_root_binding(
        row, verify_open_fds=True, expected_inheritable=False,
    )


def _validate_completion_anchor_channel_shape(
    value: Any,
    *,
    expected_target: str,
    expected_inheritable: Optional[bool],
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _COMPLETION_ANCHOR_CHANNEL_FIELDS:
        fail("completion anchor channel field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("binding_digest", None)
    descriptor = row.get("descriptor")
    if (
        row.get("schema_version") != COMPLETION_ANCHOR_CHANNEL_SCHEMA
        or type(descriptor) is not int
        or descriptor < 3
        or type(row.get("controller_pid")) is not int
        or row["controller_pid"] <= 1
        or type(row.get("target_pid")) is not int
        or row["target_pid"] <= 1
        or row.get("expected_target") != expected_target
        or expected_target not in _DYNAMIC_ANCHOR_TARGETS
        or type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or object_sha256(unsigned) != claimed
        or row["target_pid"] != os.getpid()
        or row["controller_pid"] != os.getppid()
    ):
        fail("completion anchor channel binding differs")
    try:
        observed = os.fstat(descriptor)
        inheritable = os.get_inheritable(descriptor)
        channel = socket.socket(fileno=descriptor)
        try:
            socket_type = channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            channel.getpeername()
        finally:
            channel.detach()
    except (OSError, ValueError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "completion anchor channel descriptor differs"
        ) from error
    if (
        not stat.S_ISSOCK(observed.st_mode)
        or socket_type != socket.SOCK_SEQPACKET
        or (
            expected_inheritable is not None
            and inheritable is not expected_inheritable
        )
    ):
        fail("completion anchor channel physical binding differs")
    return row


def load_completion_anchor_channel(
    *,
    expected_target: str,
    expected_inheritable: Optional[bool],
) -> Dict[str, Any]:
    raw = os.environ.get(COMPLETION_ANCHOR_CHANNEL_ENV)
    if raw is None:
        fail("completion anchor channel environment is absent")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "completion anchor channel environment is not JSON"
        ) from error
    if (
        type(value) is not dict
        or canonical_json_bytes(value).decode("utf-8") != raw
    ):
        fail("completion anchor channel environment is not canonical")
    return _validate_completion_anchor_channel_shape(
        value,
        expected_target=expected_target,
        expected_inheritable=expected_inheritable,
    )


def seal_completion_anchor_channel(*, expected_target: str) -> Dict[str, Any]:
    row = load_completion_anchor_channel(
        expected_target=expected_target, expected_inheritable=True,
    )
    os.set_inheritable(row["descriptor"], False)
    return load_completion_anchor_channel(
        expected_target=expected_target, expected_inheritable=False,
    )


def validate_holder_completion_anchor(value: Any) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _HOLDER_COMPLETION_ANCHOR_FIELDS:
        fail("holder completion anchor field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("anchor_digest", None)
    identity = row.get("initial_inode_identity")
    holder = row.get("holder_job_id")
    path = (
        Path(row["completion_path"])
        if type(row.get("completion_path")) is str else Path()
    )
    expected_suffix = (
        f"execution_shards/{holder}.holder-directory-completion.json"
    )
    if (
        row.get("schema_version") != HOLDER_COMPLETION_ANCHOR_SCHEMA
        or holder not in _HOLDER_JOB_IDS
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not str(path).endswith("/" + expected_suffix)
        or type(identity) is not dict
        or set(identity) != _HOLDER_COMPLETION_INODE_FIELDS
        or any(type(identity[field]) is not int or identity[field] < 0
               for field in _HOLDER_COMPLETION_INODE_FIELDS)
        or type(row.get("completion_size")) is not int
        or row["completion_size"] <= 0
        or type(row.get("completion_mode")) is not int
        or row["completion_mode"] != 0o444
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


def _validate_aggregate_anchor_identity(
    value: Any, *, label: str, directory: bool,
) -> Dict[str, int]:
    if (
        type(value) is not dict
        or set(value) != _WORK_ROOT_IDENTITY_FIELDS
        or any(
            type(value[field]) is not int or value[field] < 0
            for field in _WORK_ROOT_IDENTITY_FIELDS
        )
        or (directory and not stat.S_ISDIR(value["mode"]))
        or (not directory and not stat.S_ISREG(value["mode"]))
    ):
        fail(f"{label} identity differs")
    return dict(value)


def _validate_aggregate_anchor_file(
    value: Any,
    *,
    relative_path: str,
    mode: int,
    label: str,
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _AGGREGATE_ANCHOR_FILE_FIELDS:
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
        or type(row.get("mode")) is not int
        or row["mode"] != mode
        or stat.S_IMODE(identity["mode"]) != mode
        or identity["size"] != row["size"]
        or identity["nlink"] != 1
        or type(row.get("object_digest")) is not str
        or SHA256_RE.fullmatch(row["object_digest"]) is None
    ):
        fail(f"{label} binding differs")
    row["identity"] = identity
    return row


def validate_aggregate_completion_anchor(value: Any) -> Dict[str, Any]:
    if (
        type(value) is not dict
        or set(value) != _AGGREGATE_COMPLETION_ANCHOR_FIELDS
    ):
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


def _publish_completion_anchor_packet(
    row: Mapping[str, Any], *, expected_target: str,
) -> Dict[str, Any]:
    """Send the one trusted holder completion anchor to the controller.

    The descriptor is inherited only through the verified controller exec
    chain, sealed CLOEXEC before target code runs, and never passed to decoder
    or torchrun descendants.  ``SOCK_SEQPACKET`` preserves the one-object
    message boundary; the controller authenticates its direct child using
    kernel supplied credentials.
    """

    if os.environ.get(COMPLETION_ANCHOR_SENT_ENV) is not None:
        fail("completion anchor was already sent")
    channel_binding = load_completion_anchor_channel(
        expected_target=expected_target, expected_inheritable=False,
    )
    payload = canonical_json_bytes(row) + b"\n"
    channel = socket.socket(fileno=channel_binding["descriptor"])
    try:
        flags = getattr(socket, "MSG_NOSIGNAL", 0)
        sent = channel.send(payload, flags)
        if sent != len(payload):
            fail("holder completion anchor packet was truncated")
        channel.shutdown(socket.SHUT_WR)
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            "holder completion anchor packet publication failed"
        ) from error
    finally:
        channel.detach()
    os.environ[COMPLETION_ANCHOR_SENT_ENV] = row["anchor_digest"]
    return dict(row)


def publish_holder_completion_anchor(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and send the executor's one holder completion anchor."""

    row = validate_holder_completion_anchor(value)
    return _publish_completion_anchor_packet(
        row, expected_target=_EXECUTOR_TARGET
    )


def publish_aggregate_completion_anchor(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and send the aggregate's one sealed-tree anchor."""

    row = validate_aggregate_completion_anchor(value)
    return _publish_completion_anchor_packet(
        row, expected_target=_AGGREGATE_TARGET
    )


def _close_completion_anchor_channel(binding: Mapping[str, Any]) -> None:
    descriptor = binding.get("descriptor")
    if type(descriptor) is int:
        try:
            os.close(descriptor)
        except OSError:
            pass


_TASK_FD_BINDING_FIELDS = frozenset(
    {
        "schema_version", "task_id", "model_capture_digest",
        "adapter_capture_digest", "fd_count", "fd_rows", "fd_rows_digest",
        "namespace_root_count", "publication_root_count",
        "exact_allowlist_only", "proc_self_fd_consumption_required",
        "cross_process_proc_fd_access_forbidden", "ptrace_authorization_used",
        "fd_binding_digest",
    }
)

_TASK_MODEL_RELATIVE_FILES = (
    ".gitattributes",
    "README.md",
    "assets/arena.png",
    "assets/bernini-icon.png",
    "config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_TASK_ADAPTER_RELATIVE_FILES = (
    "receipt.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)
_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


def _validate_task_fd_publication_binding(
    value: Any,
    *,
    verify_open_fds: bool,
    expected_inheritable: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if type(value) is not dict or set(value) != _TASK_FD_BINDING_FIELDS:
        fail("task FD publication binding closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("fd_binding_digest", None)
    fd_rows = row.get("fd_rows")
    adapter_digest = row.get("adapter_capture_digest")
    if (
        row.get("schema_version")
        != "bernini-action-preservation-inherited-fd-binding-v3"
        or type(row.get("task_id")) is not str
        or _SAFE_TASK_ID.fullmatch(row["task_id"]) is None
        or type(row.get("model_capture_digest")) is not str
        or SHA256_RE.fullmatch(row["model_capture_digest"]) is None
        or (
            adapter_digest is not None
            and (
                type(adapter_digest) is not str
                or SHA256_RE.fullmatch(adapter_digest) is None
            )
        )
        or row.get("fd_count") not in {25, 29}
        or type(fd_rows) is not list
        or len(fd_rows) != row["fd_count"]
        or row.get("fd_rows_digest") != object_sha256(fd_rows)
        or row.get("namespace_root_count") != (1 if len(fd_rows) == 25 else 2)
        or row.get("publication_root_count") != 1
        or row.get("exact_allowlist_only") is not True
        or row.get("proc_self_fd_consumption_required") is not True
        or row.get("cross_process_proc_fd_access_forbidden") is not True
        or row.get("ptrace_authorization_used") is not False
        or type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        fail("task FD publication binding policy differs")
    descriptors: list[int] = []
    task_rows: list[Dict[str, Any]] = []
    scope_role_paths: list[Tuple[str, str, str]] = []
    for item in fd_rows:
        if (
            type(item) is not dict
            or set(item) != {
                "fd", "scope", "role", "relative_path", "source_path",
                "identity",
            }
            or type(item.get("fd")) is not int
            or item["fd"] < 3
            or type(item.get("source_path")) is not str
            or not Path(item["source_path"]).is_absolute()
            or os.path.normpath(item["source_path"]) != item["source_path"]
            or type(item.get("identity")) is not dict
            or set(item["identity"]) != _WORK_ROOT_IDENTITY_FIELDS
            or any(type(value) is not int for value in item["identity"].values())
            or item.get("scope") not in {"model", "adapter", "task"}
            or item.get("role") not in {
                "file", "namespace_root", "publication_root",
            }
            or type(item.get("relative_path")) is not str
        ):
            fail("task FD publication row differs")
        descriptors.append(item["fd"])
        scope_role_paths.append(
            (item["scope"], item["role"], item["relative_path"])
        )
        if item.get("scope") == "task" and item.get("role") == "publication_root":
            task_rows.append(dict(item))
        if verify_open_fds:
            try:
                observed = _work_root_identity_value(os.fstat(item["fd"]))
                inheritable = os.get_inheritable(item["fd"])
                named = _work_root_identity_value(
                    os.lstat(item["source_path"])
                )
            except OSError as error:
                raise DecodedEvalVerifiedReleaseError(
                    "task FD publication descriptor is unavailable"
                ) from error
            task_root_mutable = (
                item["scope"] == "task"
                and item["role"] == "publication_root"
            )
            identity_differs = (
                (
                    not task_root_mutable
                    and (observed != item["identity"] or named != item["identity"])
                )
                or (
                    task_root_mutable
                    and (
                        observed != named
                        or {
                            field: observed[field]
                            for field in _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
                        } != {
                            field: item["identity"][field]
                            for field in _WORK_ROOT_IMMUTABLE_IDENTITY_FIELDS
                        }
                    )
                )
            )
            if identity_differs or (
                expected_inheritable is not None
                and inheritable is not expected_inheritable
            ):
                fail("task FD publication descriptor identity differs")
    expected_scope_role_paths = {
        *(("model", "file", relative) for relative in _TASK_MODEL_RELATIVE_FILES),
        ("model", "namespace_root", "."),
        ("task", "publication_root", "."),
        *(
            () if adapter_digest is None else tuple(
                ("adapter", "file", relative)
                for relative in _TASK_ADAPTER_RELATIVE_FILES
            )
        ),
        *(
            () if adapter_digest is None
            else (("adapter", "namespace_root", "."),)
        ),
    }
    if (
        descriptors != sorted(descriptors)
        or len(descriptors) != len(set(descriptors))
        or len(scope_role_paths) != len(expected_scope_role_paths)
        or set(scope_role_paths) != expected_scope_role_paths
        or (adapter_digest is None) is not (len(descriptors) == 25)
        or len(task_rows) != 1
        or task_rows[0].get("relative_path") != "."
        or not stat.S_ISDIR(task_rows[0]["identity"]["mode"])
    ):
        fail("task FD publication exact descriptor closure differs")
    return row, task_rows[0]


def _load_and_seal_task_fd_publication_environment(
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
    if raw is None:
        fail("task FD publication environment is absent")
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "task FD publication environment is not JSON"
        ) from error
    if type(value) is not dict or canonical_json_bytes(value).decode("utf-8") != raw:
        fail("task FD publication environment is not canonical")
    row, task = _validate_task_fd_publication_binding(
        value, verify_open_fds=True, expected_inheritable=True
    )
    for item in row["fd_rows"]:
        os.set_inheritable(item["fd"], False)
    row, task = _validate_task_fd_publication_binding(
        row, verify_open_fds=True, expected_inheritable=False
    )
    return row, task


def _task_publication_member(
    task_row: Mapping[str, Any], path_value: Path, *, label: str,
) -> Tuple[int, str]:
    path = Path(path_value)
    root = Path(f"/proc/self/fd/{task_row['fd']}")
    if (
        not path.is_absolute()
        or path.parent != root
        or path.name in ("", ".", "..")
        or "/" in path.name
        or "\x00" in path.name
    ):
        fail(f"{label} path differs from inherited task publication root")
    return int(task_row["fd"]), path.name


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _canonical_file_path(path_value: Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink path")
    try:
        if path.resolve(strict=True) != path:
            fail(f"{label} must be a canonical path")
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(f"{label} is unavailable") from error
    return path


def _stable_capture(
    path_value: Path, *, label: str, expected_sha256: Optional[str] = None,
    expected_mode: Optional[int] = None,
) -> Tuple[bytes, os.stat_result]:
    if expected_sha256 is not None and SHA256_RE.fullmatch(expected_sha256) is None:
        fail(f"{label} expected SHA-256 differs")
    path = _canonical_file_path(Path(path_value), label=label)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
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
    if (
        not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second or len(first) != before.st_size
        or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        fail(f"{label} physical identity changed or differs")
    digest = hashlib.sha256(first).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        fail(f"{label} SHA-256 differs")
    return first, before


FILE_BINDING_FIELDS = frozenset(
    {
        "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
        "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
    }
)
DIRECTORY_BINDING_FIELDS = frozenset(FILE_BINDING_FIELDS - {"sha256"})


def _file_binding_value(path: Path, raw: bytes, metadata: os.stat_result) -> Dict[str, Any]:
    return {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        **_identity_value(metadata),
    }


def capture_file_binding(
    path_value: Path, *, label: str, expected_sha256: Optional[str] = None,
    expected_mode: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path_value)
    raw, metadata = _stable_capture(
        path, label=label, expected_sha256=expected_sha256,
        expected_mode=expected_mode,
    )
    return _file_binding_value(path, raw, metadata)


def capture_directory_binding(path_value: Path, *, label: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        if path.resolve(strict=True) != path:
            fail(f"{label} must be canonical")
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(f"{label} is unavailable") from error
    descriptor = os.open(path, _directory_flags())
    try:
        before = os.fstat(descriptor)
        middle = os.fstat(descriptor)
        named = path.lstat()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(named.st_mode)
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
    ):
        fail(f"{label} physical identity changed or differs")
    return {"path": str(path), **_identity_value(before)}


def _capture_child_directory_binding(
    path_value: Path, *, parent_fd: int, label: str,
) -> Dict[str, Any]:
    path = Path(path_value)
    if (
        type(parent_fd) is not int
        or parent_fd < 3
        or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.name in ("", ".", "..")
    ):
        fail(f"{label} retained child path differs")
    parent_before = os.fstat(parent_fd)
    named_parent_before = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(named_parent_before.st_mode)
        or _identity(parent_before) != _identity(named_parent_before)
    ):
        fail(f"{label} retained parent identity differs")
    descriptor = os.open(path.name, _directory_flags(), dir_fd=parent_fd)
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        first = sorted(os.listdir(descriptor))
        middle = os.fstat(descriptor)
        second = sorted(os.listdir(descriptor))
        after = os.fstat(descriptor)
        named = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
    finally:
        os.close(descriptor)
    parent_after = os.fstat(parent_fd)
    named_parent_after = path.parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or _identity(parent_after) != _identity(named_parent_after)
        or (
            parent_before.st_dev,
            parent_before.st_ino,
            parent_before.st_uid,
            parent_before.st_gid,
            stat.S_IFMT(parent_before.st_mode),
            parent_before.st_rdev,
        )
        != (
            parent_after.st_dev,
            parent_after.st_ino,
            parent_after.st_uid,
            parent_after.st_gid,
            stat.S_IFMT(parent_after.st_mode),
            parent_after.st_rdev,
        )
    ):
        fail(f"{label} retained physical identity changed or differs")
    return {"path": str(path), **_identity_value(before)}


def _validate_file_binding_shape(value: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != FILE_BINDING_FIELDS:
        fail(f"{label} binding field closure differs")
    row = dict(value)
    if (
        not isinstance(row.get("path"), str) or not Path(row["path"]).is_absolute()
        or os.path.normpath(row["path"]) != row["path"]
        or not isinstance(row.get("sha256"), str)
        or SHA256_RE.fullmatch(row["sha256"]) is None
        or any(
            type(row.get(field)) is not int or row[field] < 0
            for field in FILE_BINDING_FIELDS - {"path", "sha256"}
        )
        or row["size"] <= 0 or row["nlink"] != 1 or row["mode"] & ~0o7777
    ):
        fail(f"{label} binding value differs")
    return row


def _validate_directory_binding_shape(
    value: Mapping[str, Any], *, label: str
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != DIRECTORY_BINDING_FIELDS:
        fail(f"{label} binding field closure differs")
    row = dict(value)
    if (
        not isinstance(row.get("path"), str) or not Path(row["path"]).is_absolute()
        or os.path.normpath(row["path"]) != row["path"]
        or any(
            type(row.get(field)) is not int or row[field] < 0
            for field in DIRECTORY_BINDING_FIELDS - {"path"}
        )
        or row["mode"] & ~0o7777
    ):
        fail(f"{label} binding value differs")
    return row


def replay_file_binding(value: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    row = _validate_file_binding_shape(value, label=label)
    observed = capture_file_binding(
        Path(row["path"]), label=label, expected_sha256=row["sha256"],
        expected_mode=row["mode"],
    )
    if observed != row:
        fail(f"{label} full physical identity differs")
    return observed


def replay_directory_binding(value: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    row = _validate_directory_binding_shape(value, label=label)
    observed = capture_directory_binding(Path(row["path"]), label=label)
    if observed != row:
        fail(f"{label} full physical identity differs")
    return observed


def _unique_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate release JSON key: {key!r}")
        value[key] = item
    return value


def _decode_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DecodedEvalVerifiedReleaseError(f"{label} is not strict JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        fail(f"{label} is not one canonical JSON object")
    return value


def _validate_member_path(relative: Any) -> str:
    if not isinstance(relative, str):
        fail("release member path is not text")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts
        or pure.as_posix() != relative or relative.startswith("._") or "/._" in relative
    ):
        fail("release member path is unsafe or non-canonical")
    return relative


def validate_manifest(
    value: Mapping[str, Any], *, expected_manifest_sha256: str,
    expected_content_revision: str,
) -> Mapping[str, Any]:
    if (
        SHA256_RE.fullmatch(expected_manifest_sha256) is None
        or SHA1_RE.fullmatch(expected_content_revision) is None
    ):
        fail("expected release identities differ")
    unsigned = dict(value)
    declared_digest = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    if (
        type(value) is not dict or set(value) != MANIFEST_FIELDS
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("exact_member_closure") is not True
        or value.get("file_count") != len(EVAL_RELEASE_MEMBERS)
        or type(rows) is not list or len(rows) != len(EVAL_RELEASE_MEMBERS)
        or value.get("content_revision") != expected_content_revision
        or value.get("allowed_entrypoints") != sorted(ALLOWED_PYTHON_TARGETS)
        or value.get("authority") != AUTHORITY
        or not isinstance(declared_digest, str)
        or SHA256_RE.fullmatch(declared_digest) is None
        or object_sha256(unsigned) != declared_digest
    ):
        fail("release manifest schema, authority, or digest differs")
    components = value.get("component_sha256")
    if type(components) is not dict or set(components) != set(EVAL_RELEASE_MEMBERS):
        fail("release component closure differs")
    for expected_path, row in zip(EVAL_RELEASE_MEMBERS, rows):
        if type(row) is not dict or set(row) != {"path", "mode", "size", "sha256"}:
            fail("release manifest member row schema differs")
        relative = _validate_member_path(row.get("path"))
        if (
            relative != expected_path or row.get("mode") != MEMBER_MODES[relative]
            or type(row.get("size")) is not int or row["size"] <= 0
            or not isinstance(row.get("sha256"), str)
            or SHA256_RE.fullmatch(row["sha256"]) is None
            or components.get(relative) != row["sha256"]
        ):
            fail(f"release manifest member row differs: {relative}")
        trusted = TRUSTED_EXACT15.get(relative)
        if trusted is not None and (
            row["sha256"], row["size"], row["mode"]
        ) != trusted:
            fail(f"release member differs from independently reviewed pin: {relative}")
    if content_revision(rows) != expected_content_revision:
        fail("release content revision differs from exact member rows")
    return value


def validate_envelope(
    value: Mapping[str, Any], *, archive_sha256: str, manifest_sha256: str,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = dict(value)
    declared = unsigned.pop("envelope_digest", None)
    expected_archive = {"basename": "source.tar", "sha256": archive_sha256, "mode": 0o444}
    expected_manifest = {
        "basename": "source.manifest.json", "sha256": manifest_sha256,
        "manifest_digest": manifest["manifest_digest"],
        "content_revision": manifest["content_revision"],
        "file_count": len(EVAL_RELEASE_MEMBERS), "mode": 0o444,
    }
    if (
        type(value) is not dict or set(value) != ENVELOPE_FIELDS
        or value.get("schema_version") != ENVELOPE_SCHEMA
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("remote_release_exact_entries")
        != ["deployment-envelope.json", "source.manifest.json", "source.tar"]
        or value.get("source_archive") != expected_archive
        or value.get("source_manifest") != expected_manifest
        or value.get("create_only_deployment_required") is not True
        or value.get("fresh_materialized_root_required") is not True
        or value.get("verified_runtime_required") is not True
        or value.get("detached_controller_authority_receipt_required") is not True
        or value.get("automatic_scientific_promotion_authorized") is not False
        or not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        fail("deployment envelope schema, binding, or authority differs")
    return value


def _ustar_text(value: str, width: int, label: str) -> bytes:
    if type(value) is not str or "\0" in value:
        fail(f"{label} differs")
    try:
        raw = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise DecodedEvalVerifiedReleaseError(f"{label} is not USTAR ASCII") from error
    if len(raw) > width:
        fail(f"{label} exceeds canonical USTAR width")
    return raw + b"\0" * (width - len(raw))


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    if type(value) is not int or value < 0 or value >= 8 ** (width - 1):
        fail(f"{label} differs")
    return f"{value:0{width - 1}o}".encode("ascii") + b"\0"


def _ustar_name_fields(value: str) -> Tuple[bytes, bytes]:
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise DecodedEvalVerifiedReleaseError("USTAR member name is not ASCII") from error
    if len(encoded) <= 100:
        return _ustar_text(value, 100, "USTAR name"), b"\0" * 155
    for index in range(len(value) - 1, -1, -1):
        if value[index] != "/":
            continue
        prefix, basename = value[:index], value[index + 1:]
        try:
            prefix_raw, basename_raw = prefix.encode("ascii"), basename.encode("ascii")
        except UnicodeEncodeError:
            continue
        if prefix and basename and len(prefix_raw) <= 155 and len(basename_raw) <= 100:
            return _ustar_text(basename, 100, "USTAR name"), _ustar_text(prefix, 155, "USTAR prefix")
    fail("USTAR member name cannot be represented without extensions")


def fixed_ustar_header(name: str, *, size: int, mode: int) -> bytes:
    name_field, prefix_field = _ustar_name_fields(name)
    header = bytearray(FIXED_USTAR_BLOCK_SIZE)
    header[0:100] = name_field
    header[100:108] = _ustar_octal(mode, 8, "USTAR mode")
    header[108:116] = _ustar_octal(0, 8, "USTAR uid")
    header[116:124] = _ustar_octal(0, 8, "USTAR gid")
    header[124:136] = _ustar_octal(size, 12, "USTAR size")
    header[136:148] = _ustar_octal(0, 12, "USTAR mtime")
    header[148:156] = b" " * 8
    header[156:157] = b"0"
    header[157:257] = b"\0" * 100
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[265:329] = b"\0" * 64
    header[329:337] = _ustar_octal(0, 8, "USTAR devmajor")
    header[337:345] = _ustar_octal(0, 8, "USTAR devminor")
    header[345:500] = prefix_field
    header[500:512] = b"\0" * 12
    checksum = sum(header)
    if checksum >= 8 ** 6:
        fail("USTAR checksum exceeds field width")
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    return bytes(header)


def fixed_ustar_archive(
    rows: Sequence[Mapping[str, Any]], payloads: Mapping[str, bytes]
) -> bytes:
    if [row.get("path") for row in rows] != list(EVAL_RELEASE_MEMBERS):
        fail("fixed USTAR row closure differs")
    if set(payloads) != set(EVAL_RELEASE_MEMBERS):
        fail("fixed USTAR payload closure differs")
    output = bytearray()
    for row in rows:
        relative = row["path"]
        raw = payloads[relative]
        if (
            type(raw) is not bytes or len(raw) != row["size"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            fail(f"fixed USTAR payload differs: {relative}")
        output.extend(fixed_ustar_header(
            f"{MEMBER_ROOT}/{relative}", size=len(raw), mode=row["mode"]
        ))
        output.extend(raw)
        output.extend(b"\0" * (-len(raw) % FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (2 * FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (-len(output) % FIXED_USTAR_RECORD_SIZE))
    return bytes(output)


def verify_archive_snapshot(
    raw: bytes, manifest: Mapping[str, Any]
) -> Mapping[str, bytes]:
    rows = manifest["files"]
    expected_names = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    payloads: Dict[str, bytes] = {}
    expected_offset = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
            members = bundle.getmembers()
            if [member.name for member in members] != expected_names:
                fail("archive exact regular member closure/order differs")
            for member, row, expected_name in zip(members, rows, expected_names):
                if (
                    member.type != tarfile.REGTYPE or not member.isfile()
                    or member.issym() or member.islnk() or member.linkname != ""
                    or member.pax_headers or member.mode != row["mode"]
                    or member.uid != 0 or member.gid != 0
                    or member.uname != "" or member.gname != "" or member.mtime != 0
                    or member.size != row["size"] or member.offset != expected_offset
                    or member.offset_data != expected_offset + FIXED_USTAR_BLOCK_SIZE
                ):
                    fail(f"archive member metadata differs: {expected_name}")
                header = raw[member.offset:member.offset + FIXED_USTAR_BLOCK_SIZE]
                if header != fixed_ustar_header(
                    expected_name, size=row["size"], mode=row["mode"]
                ):
                    fail(f"archive member is not canonical USTAR: {expected_name}")
                handle = bundle.extractfile(member)
                payload = b"" if handle is None else handle.read()
                if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    fail(f"archive member bytes differ: {expected_name}")
                payloads[row["path"]] = payload
                blocks = (member.size + FIXED_USTAR_BLOCK_SIZE - 1) // FIXED_USTAR_BLOCK_SIZE
                expected_offset = member.offset_data + blocks * FIXED_USTAR_BLOCK_SIZE
    except (tarfile.TarError, OSError) as error:
        raise DecodedEvalVerifiedReleaseError("release archive is not readable USTAR") from error
    trailer = raw[expected_offset:]
    if (
        len(raw) % FIXED_USTAR_RECORD_SIZE != 0
        or len(trailer) < 2 * FIXED_USTAR_BLOCK_SIZE or any(trailer)
        or raw != fixed_ustar_archive(rows, payloads)
    ):
        fail("archive canonical zero trailer or record boundary differs")
    return payloads


def capture_release_artifacts(
    *, archive: Path, expected_archive_sha256: str, manifest: Path,
    expected_manifest_sha256: str, expected_content_revision: str,
    envelope: Path, expected_envelope_sha256: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, bytes], Dict[str, Any]]:
    """Capture and verify the detached three-file release authority."""

    manifest_path = Path(manifest)
    manifest_raw, manifest_metadata = _stable_capture(
        manifest_path, label="release manifest",
        expected_sha256=expected_manifest_sha256, expected_mode=0o444,
    )
    manifest_value = _decode_json(manifest_raw, label="release manifest")
    validate_manifest(
        manifest_value, expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
    )
    archive_path = Path(archive)
    archive_raw, archive_metadata = _stable_capture(
        archive_path, label="release archive",
        expected_sha256=expected_archive_sha256, expected_mode=0o444,
    )
    payloads = verify_archive_snapshot(archive_raw, manifest_value)
    envelope_path = Path(envelope)
    envelope_raw, envelope_metadata = _stable_capture(
        envelope_path, label="deployment envelope",
        expected_sha256=expected_envelope_sha256, expected_mode=0o444,
    )
    envelope_value = _decode_json(envelope_raw, label="deployment envelope")
    validate_envelope(
        envelope_value, archive_sha256=expected_archive_sha256,
        manifest_sha256=expected_manifest_sha256, manifest=manifest_value,
    )
    captured_rows = [
        {
            "path": row["path"], "mode": row["mode"],
            "size": len(payloads[row["path"]]),
            "sha256": hashlib.sha256(payloads[row["path"]]).hexdigest(),
        }
        for row in manifest_value["files"]
    ]
    if captured_rows != manifest_value["files"]:
        fail("captured archive member rows differ from manifest")
    authority = {
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_digest": manifest_value["manifest_digest"],
        "content_revision": expected_content_revision,
        "envelope_sha256": expected_envelope_sha256,
        "envelope_digest": envelope_value["envelope_digest"],
        "all_members_capture_digest": object_sha256(captured_rows),
        "member_count": len(captured_rows),
        "release_artifacts": {
            "archive": _file_binding_value(
                archive_path, archive_raw, archive_metadata
            ),
            "manifest": _file_binding_value(
                manifest_path, manifest_raw, manifest_metadata
            ),
            "envelope": _file_binding_value(
                envelope_path, envelope_raw, envelope_metadata
            ),
        },
    }
    return manifest_value, envelope_value, payloads, authority


def _canonical_fresh_output(path_value: Path) -> Tuple[Path, Path]:
    path = Path(path_value)
    if (
        not path.is_absolute() or path.exists() or path.is_symlink()
        or path.name in ("", ".", "..")
    ):
        fail("release extraction root must be one fresh absolute path")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = path.parent.lstat()
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            "release extraction parent is unavailable"
        ) from error
    if parent != path.parent or path.parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        fail("release extraction parent must be canonical")
    return path, parent


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        fail("safe extraction requires O_DIRECTORY and O_NOFOLLOW")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _expected_tree() -> Tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = {"."}
    root = PurePosixPath(MEMBER_ROOT)
    for relative in EVAL_RELEASE_MEMBERS:
        path = root / relative
        files.add(path.as_posix())
        parent = path.parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _scan_tree(root: Path) -> Tuple[Dict[str, os.stat_result], Dict[str, os.stat_result]]:
    files: Dict[str, os.stat_result] = {}
    directories: Dict[str, os.stat_result] = {}
    stack = [root]
    while stack:
        current = stack.pop()
        relative = current.relative_to(root)
        key = "." if relative == Path(".") else relative.as_posix()
        before = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(before.st_mode):
            fail(f"materialized release directory differs: {key}")
        if stat.S_IMODE(before.st_mode) != 0o555:
            fail(f"materialized release directory mode differs: {key}")
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError as error:
            raise DecodedEvalVerifiedReleaseError(
                f"cannot scan materialized release directory: {key}"
            ) from error
        after = current.lstat()
        if _identity(before) != _identity(after):
            fail(f"materialized release directory changed while scanning: {key}")
        directories[key] = after
        children = []
        for entry in entries:
            child = current / entry.name
            child_relative = child.relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                fail(f"materialized release symlink is forbidden: {child_relative}")
            if stat.S_ISDIR(metadata.st_mode):
                children.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                files[child_relative] = metadata
            else:
                fail(f"materialized release special entry is forbidden: {child_relative}")
        stack.extend(reversed(children))
    return files, directories


def capture_materialized_release(
    release_root: Path, manifest: Mapping[str, Any]
) -> Mapping[str, bytes]:
    root = Path(release_root)
    if not root.is_absolute() or root.is_symlink():
        fail("materialized release root must be absolute and non-symlink")
    try:
        if root.resolve(strict=True) != root:
            fail("materialized release root must be canonical")
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            "materialized release root is unavailable"
        ) from error
    expected_files, expected_directories = _expected_tree()
    files_before, directories_before = _scan_tree(root)
    if set(files_before) != expected_files or set(directories_before) != expected_directories:
        fail("materialized release has links, extras, or missing entries")
    rows = {f"{MEMBER_ROOT}/{row['path']}": row for row in manifest["files"]}
    payloads: Dict[str, bytes] = {}
    identities: Dict[str, Tuple[int, ...]] = {}
    for full_path in sorted(expected_files):
        row = rows[full_path]
        raw, metadata = _stable_capture(
            root / full_path, label=f"materialized release member {full_path}",
            expected_sha256=row["sha256"], expected_mode=row["mode"],
        )
        if len(raw) != row["size"]:
            fail(f"materialized release member size differs: {full_path}")
        payloads[row["path"]] = raw
        identities[full_path] = _identity(metadata)
    files_after, directories_after = _scan_tree(root)
    if set(files_after) != expected_files or set(directories_after) != expected_directories:
        fail("materialized release tree changed during capture")
    for path in expected_files:
        if identities[path] != _identity(files_after[path]):
            fail(f"materialized release member changed after capture: {path}")
    for path in expected_directories:
        if _identity(directories_before[path]) != _identity(directories_after[path]):
            fail(f"materialized release directory changed during capture: {path}")
    return payloads


def _capture_materialized_release_from_open_fds(
    *, destination: Path, parent_fd: int,
    directory_fds: Mapping[str, int], manifest: Mapping[str, Any],
) -> Mapping[str, bytes]:
    expected_files, expected_directories = _expected_tree()
    if set(directory_fds) != expected_directories:
        fail("retained materialized release directory closure differs")
    expected_entries: Dict[str, set[str]] = {
        relative: set() for relative in expected_directories
    }
    for relative in expected_directories - {"."}:
        pure = PurePosixPath(relative)
        parent_key = (
            "." if pure.parent == PurePosixPath(".")
            else pure.parent.as_posix()
        )
        expected_entries[parent_key].add(pure.name)
    for relative in expected_files:
        pure = PurePosixPath(relative)
        parent_key = (
            "." if pure.parent == PurePosixPath(".")
            else pure.parent.as_posix()
        )
        expected_entries[parent_key].add(pure.name)
    directory_identities: Dict[str, Tuple[int, ...]] = {}
    for relative in sorted(expected_directories):
        descriptor = directory_fds[relative]
        before = os.fstat(descriptor)
        first = sorted(os.listdir(descriptor))
        middle = os.fstat(descriptor)
        second = sorted(os.listdir(descriptor))
        after = os.fstat(descriptor)
        if relative == ".":
            named = os.stat(
                destination.name, dir_fd=parent_fd, follow_symlinks=False
            )
        else:
            pure = PurePosixPath(relative)
            parent_key = (
                "." if pure.parent == PurePosixPath(".")
                else pure.parent.as_posix()
            )
            named = os.stat(
                pure.name,
                dir_fd=directory_fds[parent_key],
                follow_symlinks=False,
            )
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o555
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != second
            or first != sorted(expected_entries[relative])
        ):
            fail(
                f"retained materialized release directory differs: {relative}"
            )
        directory_identities[relative] = _identity(before)
    rows = {f"{MEMBER_ROOT}/{row['path']}": row for row in manifest["files"]}
    payloads: Dict[str, bytes] = {}
    file_identities: Dict[str, Tuple[int, ...]] = {}
    for full_path in sorted(expected_files):
        pure = PurePosixPath(full_path)
        parent_key = (
            "." if pure.parent == PurePosixPath(".")
            else pure.parent.as_posix()
        )
        row = rows[full_path]
        descriptor = os.open(
            pure.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fds[parent_key],
        )
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                pure.name,
                dir_fd=directory_fds[parent_key],
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != row["mode"]
            or before.st_size != row["size"]
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != second
            or hashlib.sha256(first).hexdigest() != row["sha256"]
        ):
            fail(f"retained materialized release member differs: {full_path}")
        payloads[row["path"]] = first
        file_identities[full_path] = _identity(before)
    for relative, expected_identity in directory_identities.items():
        if _identity(os.fstat(directory_fds[relative])) != expected_identity:
            fail(
                f"retained materialized release directory changed: {relative}"
            )
    for full_path, expected_identity in file_identities.items():
        pure = PurePosixPath(full_path)
        parent_key = (
            "." if pure.parent == PurePosixPath(".")
            else pure.parent.as_posix()
        )
        named = os.stat(
            pure.name,
            dir_fd=directory_fds[parent_key],
            follow_symlinks=False,
        )
        if _identity(named) != expected_identity:
            fail(f"retained materialized release member changed: {full_path}")
    return payloads


def extract_verified_release(
    *, archive: Path, expected_archive_sha256: str, manifest: Path,
    expected_manifest_sha256: str, expected_content_revision: str,
    envelope: Path, expected_envelope_sha256: str, output_root: Path,
    retained_parent_fd: Optional[int] = None,
) -> Mapping[str, Any]:
    manifest_value, envelope_value, payloads, authority = capture_release_artifacts(
        archive=archive, expected_archive_sha256=expected_archive_sha256,
        manifest=manifest, expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision, envelope=envelope,
        expected_envelope_sha256=expected_envelope_sha256,
    )
    if retained_parent_fd is None:
        destination, parent = _canonical_fresh_output(Path(output_root))
        parent_fd = os.open(parent, _directory_flags())
    else:
        destination = Path(output_root)
        parent = destination.parent
        if (
            type(retained_parent_fd) is not int
            or retained_parent_fd < 3
            or not destination.is_absolute()
            or os.path.normpath(str(destination)) != str(destination)
            or destination.name in ("", ".", "..")
        ):
            fail("retained release output path differs")
        try:
            parent_fd = os.dup(retained_parent_fd)
        except OSError as error:
            raise DecodedEvalVerifiedReleaseError(
                "retained release parent FD is unavailable"
            ) from error
        os.set_inheritable(parent_fd, False)
    directory_fds: Dict[str, int] = {}
    captured: Mapping[str, bytes] | None = None
    try:
        parent_before = os.fstat(parent_fd)
        named_parent_before = parent.lstat()
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or stat.S_ISLNK(named_parent_before.st_mode)
            or _identity(parent_before) != _identity(named_parent_before)
        ):
            fail("retained release parent identity differs")
        try:
            os.stat(
                destination.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            fail("materialized release root is not fresh")
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        root_fd = os.open(destination.name, _directory_flags(), dir_fd=parent_fd)
        directory_fds["."] = root_fd
        os.fchmod(root_fd, 0o700)
        os.fsync(root_fd)
        _, expected_directories = _expected_tree()
        for directory in sorted(
            expected_directories - {"."}, key=lambda item: (item.count("/"), item)
        ):
            pure = PurePosixPath(directory)
            parent_key = "." if pure.parent == PurePosixPath(".") else pure.parent.as_posix()
            parent_directory_fd = directory_fds[parent_key]
            os.mkdir(pure.name, 0o700, dir_fd=parent_directory_fd)
            child_fd = os.open(pure.name, _directory_flags(), dir_fd=parent_directory_fd)
            os.fchmod(child_fd, 0o700)
            os.fsync(child_fd)
            os.fsync(parent_directory_fd)
            directory_fds[directory] = child_fd
        for row in manifest_value["files"]:
            relative = PurePosixPath(row["path"])
            parent_key = (PurePosixPath(MEMBER_ROOT) / relative.parent).as_posix()
            descriptor = os.open(
                relative.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                row["mode"], dir_fd=directory_fds[parent_key],
            )
            try:
                raw = payloads[row["path"]]
                offset = 0
                while offset < len(raw):
                    count = os.write(descriptor, raw[offset:])
                    if count <= 0:
                        fail("release member write made no progress")
                    offset += count
                os.fchmod(descriptor, row["mode"])
                os.fsync(descriptor)
                opened = os.fstat(descriptor)
                named = os.stat(
                    relative.name, dir_fd=directory_fds[parent_key],
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_size != row["size"]
                    or stat.S_IMODE(opened.st_mode) != row["mode"]
                    or _identity(opened) != _identity(named)
                ):
                    fail(f"created release member identity differs: {row['path']}")
            finally:
                os.close(descriptor)
            os.fsync(directory_fds[parent_key])
        for directory in sorted(
            directory_fds, key=lambda item: (item.count("/"), item), reverse=True
        ):
            descriptor = directory_fds[directory]
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        os.fsync(parent_fd)
        captured = _capture_materialized_release_from_open_fds(
            destination=destination,
            parent_fd=parent_fd,
            directory_fds=directory_fds,
            manifest=manifest_value,
        )
        parent_after = os.fstat(parent_fd)
        named_parent_after = parent.lstat()
        if (
            _identity(parent_after) != _identity(named_parent_after)
            or (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_uid,
                parent_before.st_gid,
                stat.S_IFMT(parent_before.st_mode),
                parent_before.st_rdev,
            )
            != (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_uid,
                parent_after.st_gid,
                stat.S_IFMT(parent_after.st_mode),
                parent_after.st_rdev,
            )
        ):
            fail("retained release parent changed during extraction")
    except FileExistsError as error:
        raise DecodedEvalVerifiedReleaseError(
            "create-only release extraction collided with an existing entry"
        ) from error
    finally:
        for descriptor in directory_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent_fd)
    if captured is None:
        fail("retained materialized release capture is absent")
    if captured != payloads:
        fail("post-extraction captured bytes differ from archive")
    result = {
        "release_root": str(destination), "member_root": str(destination / MEMBER_ROOT),
        **authority, "envelope_digest": envelope_value["envelope_digest"],
        "exact_tree_verified": True, "directories_sealed_mode": "0555",
    }
    result["receipt_digest"] = object_sha256(result)
    return result


def _path_is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _strip_forbidden_sys_path(forbidden_roots: Sequence[Path]) -> None:
    kept = []
    cwd = Path.cwd().resolve()
    roots = tuple(root.resolve() for root in forbidden_roots)
    for entry in sys.path:
        if entry == "":
            continue
        try:
            candidate = Path(entry).expanduser()
            if not candidate.is_absolute():
                candidate = cwd / candidate
            candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            kept.append(entry)
            continue
        if candidate == cwd or any(_path_is_within(candidate, root) for root in roots):
            continue
        kept.append(entry)
    sys.path[:] = kept


class _CapturedModuleLoader(importlib.abc.Loader):
    def __init__(
        self, fullname: str, source: Path, raw: bytes, *, is_package: bool = False
    ) -> None:
        self.fullname = fullname
        self.source = source
        self.raw = raw
        self.is_package = is_package

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.source)
        module.__package__ = self.fullname if self.is_package else self.fullname.rpartition(".")[0]
        module.__cached__ = None
        if self.is_package:
            # An empty namespace search path prevents PathFinder from reopening
            # a post-capture ``tools.*`` path.  Our finder ignores this path and
            # still serves every allowed tools module from retained bytes.
            module.__path__ = []
        if self.raw:
            exec(compile(self.raw, str(self.source), "exec", dont_inherit=True), module.__dict__)


class _CapturedReleaseFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self, *, modules: Mapping[str, Tuple[Path, bytes]],
        namespace_roots: Mapping[str, Path], forbidden_roots: Sequence[Path],
    ) -> None:
        self.modules = dict(modules)
        self.namespace_roots = dict(namespace_roots)
        self.forbidden_roots = tuple(forbidden_roots)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        _strip_forbidden_sys_path(self.forbidden_roots)
        namespace = self.namespace_roots.get(fullname)
        if namespace is not None:
            loader = _CapturedModuleLoader(
                fullname, namespace / "__init__.py", b"", is_package=True
            )
            spec = importlib.machinery.ModuleSpec(
                fullname, loader, origin=str(namespace), is_package=True
            )
            spec.submodule_search_locations = []
            return spec
        item = self.modules.get(fullname)
        if item is None:
            if fullname.startswith("tools."):
                raise ImportError(
                    f"module {fullname!r} is outside the captured tools closure"
                )
            return None
        source, raw = item
        loader = _CapturedModuleLoader(fullname, source, raw)
        return importlib.machinery.ModuleSpec(
            fullname, loader, origin=str(source), is_package=False
        )


_RELEASE_BINDING_FIELDS = frozenset(
    {
        "release_root", "archive", "manifest", "manifest_digest",
        "content_revision", "envelope", "envelope_digest",
        "all_members_capture_digest",
    }
)
_CONTROLLER_AUTHORITY_BINDING_FIELDS = frozenset(
    {"receipt", "authority_digest"}
)
_TORCHRUN_BINDING_FIELDS = frozenset(
    {"source", "subprocess_handler_source", "site_packages"}
)


def _validate_torchrun_binding(
    value: Mapping[str, Any], *, label: str,
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _TORCHRUN_BINDING_FIELDS:
        fail(f"{label} binding field closure differs")
    row = {
        "source": _validate_file_binding_shape(
            value["source"], label=f"{label} source"
        ),
        "subprocess_handler_source": _validate_file_binding_shape(
            value["subprocess_handler_source"],
            label=f"{label} subprocess handler source",
        ),
        "site_packages": _validate_directory_binding_shape(
            value["site_packages"], label=f"{label} site-packages"
        ),
    }
    expected = Path(row["site_packages"]["path"]) / "torch/distributed/run.py"
    if Path(row["source"]["path"]) != expected:
        fail(f"{label} source is not the bound torch/distributed/run.py")
    if (
        row["source"]["sha256"] != TORCHRUN_SOURCE_SHA256
        or row["source"]["size"] != TORCHRUN_SOURCE_SIZE
    ):
        fail(f"{label} torchrun source bytes differ")
    expected_handler = (
        Path(row["site_packages"]["path"])
        / TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
    )
    if (
        Path(row["subprocess_handler_source"]["path"]) != expected_handler
        or row["subprocess_handler_source"]["sha256"]
        != TORCHRUN_SUBPROCESS_HANDLER_SHA256
        or row["subprocess_handler_source"]["size"]
        != TORCHRUN_SUBPROCESS_HANDLER_SIZE
    ):
        fail(f"{label} subprocess handler source differs")
    return row


def _validate_executable_binding(value: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    row = _validate_file_binding_shape(value, label=label)
    if row["mode"] & 0o111 == 0:
        fail(f"{label} is not executable")
    return row


def capture_executable_binding(path_value: Path, *, label: str) -> Dict[str, Any]:
    row = capture_file_binding(Path(path_value), label=label)
    return _validate_executable_binding(row, label=label)


def _validate_release_binding(value: Mapping[str, Any]) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _RELEASE_BINDING_FIELDS:
        fail("release binding field closure differs")
    row = dict(value)
    row["release_root"] = _validate_directory_binding_shape(
        row["release_root"], label="release root"
    )
    for field in ("archive", "manifest", "envelope"):
        row[field] = _validate_file_binding_shape(
            row[field], label=f"release {field}"
        )
    for field in ("manifest_digest", "envelope_digest", "all_members_capture_digest"):
        if not isinstance(row.get(field), str) or SHA256_RE.fullmatch(row[field]) is None:
            fail(f"release binding {field} differs")
    if not isinstance(row.get("content_revision"), str) or SHA1_RE.fullmatch(row["content_revision"]) is None:
        fail("release binding content revision differs")
    return row


def _capture_interpreter(
    binding: Mapping[str, Any], *, label: str
) -> Dict[str, Any]:
    row = _validate_executable_binding(binding, label=label)
    return replay_file_binding(row, label=label)


def capture_release_binding(
    *, release_root: Path, archive: Path, expected_archive_sha256: str,
    manifest: Path, expected_manifest_sha256: str,
    expected_content_revision: str, envelope: Path,
    expected_envelope_sha256: str,
    retained_parent_fd: Optional[int] = None,
) -> Dict[str, Any]:
    """Capture a materialized root plus the complete detached release authority."""

    root_before = (
        capture_directory_binding(
            Path(release_root), label="materialized release root"
        )
        if retained_parent_fd is None
        else _capture_child_directory_binding(
            Path(release_root), parent_fd=retained_parent_fd,
            label="materialized release root",
        )
    )
    _, _, _, authority = capture_release_artifacts(
        archive=Path(archive), expected_archive_sha256=expected_archive_sha256,
        manifest=Path(manifest),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_revision=expected_content_revision,
        envelope=Path(envelope),
        expected_envelope_sha256=expected_envelope_sha256,
    )
    root_after = (
        capture_directory_binding(
            Path(release_root), label="materialized release root"
        )
        if retained_parent_fd is None
        else _capture_child_directory_binding(
            Path(release_root), parent_fd=retained_parent_fd,
            label="materialized release root",
        )
    )
    if root_before != root_after:
        fail("materialized release root changed while binding release artifacts")
    return _validate_release_binding(
        {
            "release_root": root_before,
            **authority["release_artifacts"],
            "manifest_digest": authority["manifest_digest"],
            "content_revision": authority["content_revision"],
            "envelope_digest": authority["envelope_digest"],
            "all_members_capture_digest": authority["all_members_capture_digest"],
        }
    )


def capture_torchrun_binding(
    site_packages_path: Path, *, label: str = "captured torchrun"
) -> Dict[str, Any]:
    """Capture only run.py bytes plus the containing site path topology.

    The directory binding deliberately does not claim a digest of the complete
    dependency tree.  ``torch/distributed/run.py`` itself is same-fd captured.
    """

    site_before = capture_directory_binding(
        Path(site_packages_path), label=f"{label} site-packages"
    )
    source = capture_file_binding(
        Path(site_before["path"]) / "torch/distributed/run.py",
        label=f"{label} source",
    )
    subprocess_handler_source = capture_file_binding(
        Path(site_before["path"])
        / TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH,
        label=f"{label} subprocess handler source",
    )
    if (
        subprocess_handler_source["sha256"]
        != TORCHRUN_SUBPROCESS_HANDLER_SHA256
    ):
        fail(f"{label} subprocess handler SHA differs")
    site_after = capture_directory_binding(
        Path(site_packages_path), label=f"{label} site-packages"
    )
    if site_before != site_after:
        fail(f"{label} site-packages changed while capturing run.py")
    return _validate_torchrun_binding(
        {
            "source": source,
            "subprocess_handler_source": subprocess_handler_source,
            "site_packages": site_before,
        },
        label=label,
    )


def replay_torchrun_binding(
    value: Mapping[str, Any], *, label: str = "captured torchrun"
) -> Dict[str, Any]:
    row = _validate_torchrun_binding(value, label=label)
    site_before = replay_directory_binding(
        row["site_packages"], label=f"{label} site-packages"
    )
    source = replay_file_binding(row["source"], label=f"{label} source")
    subprocess_handler_source = replay_file_binding(
        row["subprocess_handler_source"],
        label=f"{label} subprocess handler source",
    )
    site_after = replay_directory_binding(
        row["site_packages"], label=f"{label} site-packages"
    )
    if site_before != site_after:
        fail(f"{label} site-packages changed while replaying run.py")
    return {
        "source": source,
        "subprocess_handler_source": subprocess_handler_source,
        "site_packages": site_before,
    }


def replay_release_binding(
    value: Mapping[str, Any], *, verify_materialized_root: bool = True,
) -> Tuple[Dict[str, Any], Mapping[str, Any], Mapping[str, bytes], Dict[str, Any]]:
    row = _validate_release_binding(value)
    if verify_materialized_root:
        replay_directory_binding(row["release_root"], label="materialized release root")
    manifest, envelope, payloads, authority = capture_release_artifacts(
        archive=Path(row["archive"]["path"]),
        expected_archive_sha256=row["archive"]["sha256"],
        manifest=Path(row["manifest"]["path"]),
        expected_manifest_sha256=row["manifest"]["sha256"],
        expected_content_revision=row["content_revision"],
        envelope=Path(row["envelope"]["path"]),
        expected_envelope_sha256=row["envelope"]["sha256"],
    )
    if (
        authority["release_artifacts"] != {
            "archive": row["archive"], "manifest": row["manifest"],
            "envelope": row["envelope"],
        }
        or authority["manifest_digest"] != row["manifest_digest"]
        or authority["envelope_digest"] != row["envelope_digest"]
        or authority["all_members_capture_digest"]
        != row["all_members_capture_digest"]
    ):
        fail("release binding full physical authority differs")
    return row, manifest, payloads, authority


def _rank_is_zero() -> bool:
    raw = os.environ.get("RANK")
    if raw is None:
        return True
    if not re.fullmatch(r"0|[1-9][0-9]*", raw):
        fail("RANK is not a canonical non-negative integer")
    return raw == "0"


def _write_create_only(
    path_value: Path, raw: bytes, *, mode: int = 0o444,
    label: str = "capture receipt",
    retained_parent_fd: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.name in ("", ".", "..")
    ):
        fail(f"{label} path must be one fresh absolute path")
    if retained_parent_fd is None:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent or path.parent.is_symlink():
            fail(f"{label} parent must be canonical")
        parent_fd = os.open(parent, _directory_flags())
    else:
        if type(retained_parent_fd) is not int or retained_parent_fd < 3:
            fail(f"{label} retained parent FD differs")
        try:
            parent_fd = os.dup(retained_parent_fd)
        except OSError as error:
            raise DecodedEvalVerifiedReleaseError(
                f"{label} retained parent FD is unavailable"
            ) from error
        os.set_inheritable(parent_fd, False)
    try:
        parent_before = os.fstat(parent_fd)
        named_parent_before = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or stat.S_ISLNK(named_parent_before.st_mode)
            or _identity(parent_before) != _identity(named_parent_before)
        ):
            fail(f"{label} retained parent identity differs")
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            fail(f"{label} path must be one fresh absolute path")
        descriptor = os.open(
            path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0), mode, dir_fd=parent_fd,
        )
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(descriptor, raw[offset:])
                if count <= 0:
                    fail(f"{label} write made no progress")
                offset += count
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != mode
                or _identity(before) != _identity(middle)
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(named)
                or first != raw or second != raw
            ):
                fail(f"{label} same-fd write replay differs")
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        replay = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            os.set_inheritable(replay, False)
            replay_before = os.fstat(replay)
            replay_first = _read_fd(replay)
            replay_middle = os.fstat(replay)
            replay_second = _read_fd(replay)
            replay_after = os.fstat(replay)
            replay_named = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False
            )
        finally:
            os.close(replay)
        parent_after = os.fstat(parent_fd)
        named_parent_after = path.parent.lstat()
        if (
            _identity(replay_before) != _identity(before)
            or _identity(replay_middle) != _identity(before)
            or _identity(replay_after) != _identity(before)
            or _identity(replay_named) != _identity(before)
            or replay_first != raw
            or replay_second != raw
            or _identity(parent_after) != _identity(named_parent_after)
            or (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_uid,
                parent_before.st_gid,
                stat.S_IFMT(parent_before.st_mode),
                parent_before.st_rdev,
            )
            != (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_uid,
                parent_after.st_gid,
                stat.S_IFMT(parent_after.st_mode),
                parent_after.st_rdev,
            )
        ):
            fail(f"{label} retained post-close replay differs")
    finally:
        os.close(parent_fd)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **_identity_value(before),
    }


def _write_create_only_in_work_root(
    work_root_binding: Mapping[str, Any],
    path_value: Path,
    raw: bytes,
    *,
    mode: int = 0o444,
    label: str = "capture receipt",
) -> Dict[str, Any]:
    row = validate_inherited_work_root_binding(
        work_root_binding,
        verify_open_fds=True,
        expected_inheritable=False,
        verify_entries=True,
    )
    path = Path(path_value)
    if (
        not path.is_absolute()
        or path.parent != Path(row["path"])
        or str(path) != row["capture_receipt_path"]
        or path.name in row["entries"]
    ):
        fail(f"{label} path differs from inherited work root")
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    )
    if not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} safe relative publication is unavailable")
    try:
        descriptor = os.open(
            path.name,
            flags | os.O_NOFOLLOW,
            mode,
            dir_fd=row["root_fd"],
        )
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            f"cannot create held-root {label}"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                fail(f"{label} held-root write made no progress")
            offset += count
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.fsync(row["root_fd"])
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            path.name, dir_fd=row["root_fd"], follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != mode
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != raw
            or second != raw
            or os.get_inheritable(descriptor)
        ):
            fail(f"{label} held-root same-FD write replay differs")
    finally:
        os.close(descriptor)
    replay = os.open(
        path.name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=row["root_fd"],
    )
    try:
        replay_before = os.fstat(replay)
        replay_first = _read_fd(replay)
        replay_middle = os.fstat(replay)
        replay_second = _read_fd(replay)
        replay_after = os.fstat(replay)
        replay_named = os.stat(
            path.name, dir_fd=row["root_fd"], follow_symlinks=False
        )
    finally:
        os.close(replay)
    if (
        _identity(replay_before) != _identity(before)
        or _identity(replay_middle) != _identity(before)
        or _identity(replay_after) != _identity(before)
        or _identity(replay_named) != _identity(before)
        or replay_first != raw
        or replay_second != raw
    ):
        fail(f"{label} held-root post-close replay differs")
    validate_inherited_work_root_binding(
        row,
        verify_open_fds=True,
        expected_inheritable=False,
        verify_entries=False,
        allow_root_metadata_change=True,
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **_identity_value(before),
    }


def _write_create_only_in_task_root(
    task_fd_binding: Mapping[str, Any],
    task_row: Mapping[str, Any],
    path_value: Path,
    raw: bytes,
    *,
    mode: int = 0o444,
    label: str = "capture receipt",
) -> Dict[str, Any]:
    binding, observed_task = _validate_task_fd_publication_binding(
        task_fd_binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    if observed_task != dict(task_row):
        fail(f"{label} task publication authority differs")
    root_fd, basename = _task_publication_member(
        observed_task, Path(path_value), label=label
    )
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    )
    if not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} safe relative publication is unavailable")
    try:
        descriptor = os.open(
            basename,
            flags | os.O_NOFOLLOW,
            mode,
            dir_fd=root_fd,
        )
    except OSError as error:
        raise DecodedEvalVerifiedReleaseError(
            f"cannot create inherited-task-root {label}"
        ) from error
    try:
        os.set_inheritable(descriptor, False)
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                fail(f"{label} inherited-task-root write made no progress")
            offset += count
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.fsync(root_fd)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != mode
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != raw
            or second != raw
            or os.get_inheritable(descriptor)
        ):
            fail(f"{label} inherited-task-root same-FD write replay differs")
    finally:
        os.close(descriptor)
    replay = os.open(
        basename,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        replay_before = os.fstat(replay)
        replay_first = _read_fd(replay)
        replay_middle = os.fstat(replay)
        replay_second = _read_fd(replay)
        replay_after = os.fstat(replay)
        replay_named = os.stat(
            basename, dir_fd=root_fd, follow_symlinks=False
        )
    finally:
        os.close(replay)
    if (
        _identity(replay_before) != _identity(before)
        or _identity(replay_middle) != _identity(before)
        or _identity(replay_after) != _identity(before)
        or _identity(replay_named) != _identity(before)
        or replay_first != raw
        or replay_second != raw
    ):
        fail(f"{label} inherited-task-root post-close replay differs")
    _validate_task_fd_publication_binding(
        binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    return {
        "path": str(Path(path_value)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **_identity_value(before),
    }


def _controller_authority_core(
    *, controller: Mapping[str, Any], root_python: Mapping[str, Any],
    frozen_python: Mapping[str, Any], site_packages: Mapping[str, Any],
    release: Mapping[str, Any],
    torchrun: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": CONTROLLER_AUTHORITY_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "controller": dict(controller), "root_python": dict(root_python),
        "frozen_python": dict(frozen_python), "site_packages": dict(site_packages),
        "release": dict(release),
        "torchrun": None if torchrun is None else dict(torchrun),
        "create_only_o_excl": True,
        "same_fd_double_read_after_fsync": True,
        "named_identity_replay_after_write": True,
        "automatic_scientific_promotion_authorized": False,
    }


def publish_controller_authority_receipt(
    output_path: Path, *, controller_binding: Mapping[str, Any],
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any],
    release_binding: Mapping[str, Any],
    torchrun_binding: Optional[Mapping[str, Any]] = None,
    retained_parent_fd: Optional[int] = None,
) -> Dict[str, Any]:
    controller = replay_file_binding(controller_binding, label="detached controller")
    root_python = _capture_interpreter(root_python_binding, label="root Python")
    frozen_python = _capture_interpreter(
        frozen_python_binding, label="frozen Python"
    )
    site_packages = replay_directory_binding(
        site_packages_binding, label="frozen site-packages"
    )
    release, _, _, _ = replay_release_binding(release_binding)
    torchrun = (
        None if torchrun_binding is None
        else replay_torchrun_binding(
            torchrun_binding, label="preauthorized captured torchrun"
        )
    )
    if torchrun is not None and torchrun["site_packages"] != site_packages:
        fail("preauthorized torchrun site-packages continuity differs")
    receipt = _controller_authority_core(
        controller=controller, root_python=root_python,
        frozen_python=frozen_python, site_packages=site_packages, release=release,
        torchrun=torchrun,
    )
    receipt["authority_digest"] = object_sha256(receipt)
    raw = canonical_json_bytes(receipt) + b"\n"
    receipt_file = _write_create_only(
        Path(output_path), raw, mode=0o444,
        label="controller authority receipt",
        retained_parent_fd=retained_parent_fd,
    )
    validate_controller_authority_receipt(receipt)
    if retained_parent_fd is None:
        replay_raw, _ = _stable_capture(
            Path(output_path), label="controller authority receipt",
            expected_sha256=receipt_file["sha256"], expected_mode=0o444,
        )
        if replay_raw != raw:
            fail("controller authority receipt post-publication replay differs")
    return {"receipt": receipt_file, "authority_digest": receipt["authority_digest"]}


def validate_controller_authority_receipt(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version", "release_generation", "controller", "root_python",
        "frozen_python", "site_packages", "release", "torchrun",
        "create_only_o_excl",
        "same_fd_double_read_after_fsync", "named_identity_replay_after_write",
        "automatic_scientific_promotion_authorized", "authority_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("controller authority receipt field closure differs")
    row = dict(value)
    controller = _validate_file_binding_shape(row["controller"], label="controller")
    root_python = _validate_executable_binding(row["root_python"], label="root Python")
    frozen_python = _validate_executable_binding(row["frozen_python"], label="frozen Python")
    site_packages = _validate_directory_binding_shape(
        row["site_packages"], label="site-packages"
    )
    release = _validate_release_binding(row["release"])
    torchrun = (
        None if row["torchrun"] is None
        else _validate_torchrun_binding(
            row["torchrun"], label="preauthorized captured torchrun"
        )
    )
    unsigned = dict(row)
    declared = unsigned.pop("authority_digest", None)
    if (
        row["schema_version"] != CONTROLLER_AUTHORITY_SCHEMA
        or row["release_generation"] != RELEASE_GENERATION
        or row["create_only_o_excl"] is not True
        or row["same_fd_double_read_after_fsync"] is not True
        or row["named_identity_replay_after_write"] is not True
        or row["automatic_scientific_promotion_authorized"] is not False
        or (torchrun is not None and torchrun["site_packages"] != site_packages)
        or not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        fail("controller authority receipt value or digest differs")
    row.update(
        controller=controller, root_python=root_python,
        frozen_python=frozen_python, site_packages=site_packages, release=release,
        torchrun=torchrun,
    )
    return row


def validate_controller_authority_binding(
    value: Mapping[str, Any], *, controller_binding: Mapping[str, Any],
    root_python_binding: Mapping[str, Any], frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any], release_binding: Mapping[str, Any],
    torchrun_binding: Optional[Mapping[str, Any]] = None,
    require_torchrun_continuity: bool = False, verify_file: bool = True,
    replay_torchrun_source: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if type(value) is not dict or set(value) != _CONTROLLER_AUTHORITY_BINDING_FIELDS:
        fail("controller authority binding field closure differs")
    binding = {
        "receipt": _validate_file_binding_shape(
            value["receipt"], label="controller authority receipt"
        ),
        "authority_digest": value.get("authority_digest"),
    }
    if (
        not isinstance(binding["authority_digest"], str)
        or SHA256_RE.fullmatch(binding["authority_digest"]) is None
    ):
        fail("controller authority digest differs")
    raw, metadata = _stable_capture(
        Path(binding["receipt"]["path"]), label="controller authority receipt",
        expected_sha256=binding["receipt"]["sha256"], expected_mode=0o444,
    )
    observed_receipt = _file_binding_value(
        Path(binding["receipt"]["path"]), raw, metadata
    )
    if verify_file and observed_receipt != binding["receipt"]:
        fail("controller authority receipt full identity differs")
    receipt = validate_controller_authority_receipt(
        _decode_json(raw, label="controller authority receipt")
    )
    expected = _controller_authority_core(
        controller=_validate_file_binding_shape(
            controller_binding, label="detached controller"
        ),
        root_python=_validate_executable_binding(
            root_python_binding, label="root Python"
        ),
        frozen_python=_validate_executable_binding(
            frozen_python_binding, label="frozen Python"
        ),
        site_packages=_validate_directory_binding_shape(
            site_packages_binding, label="site-packages"
        ),
        release=_validate_release_binding(release_binding),
        torchrun=(
            receipt["torchrun"]
            if not require_torchrun_continuity
            else (
                None if torchrun_binding is None
                else _validate_torchrun_binding(
                    torchrun_binding, label="preauthorized captured torchrun"
                )
            )
        ),
    )
    if (
        receipt["authority_digest"] != binding["authority_digest"]
        or any(receipt[field] != expected[field] for field in expected)
    ):
        fail("controller authority cross-process continuity differs")
    if verify_file:
        replay_file_binding(receipt["controller"], label="detached controller")
        _capture_interpreter(receipt["root_python"], label="root Python")
        _capture_interpreter(receipt["frozen_python"], label="frozen Python")
        replay_directory_binding(
            receipt["site_packages"], label="frozen site-packages"
        )
        replay_release_binding(receipt["release"])
        if receipt["torchrun"] is not None and replay_torchrun_source:
            replay_torchrun_binding(
                receipt["torchrun"], label="preauthorized captured torchrun"
            )
    return binding, receipt


def validate_capture_receipt(
    value: Mapping[str, Any], *, verify_file: bool = False,
    receipt_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    fields = {
        "schema_version", "release_generation", "archive_sha256", "manifest_sha256",
        "manifest_digest", "content_revision", "envelope_sha256", "envelope_digest",
        "all_members_capture_digest", "member_count", "target", "target_sha256",
        "target_size", "target_mode", "target_arguments_sha256", "root_python",
        "frozen_python", "site_packages", "release_artifacts",
        "controller_authority", "captured_torchrun", "work_root",
        "task_fd_binding", "publication_authority_kind", "receipt_path",
        "publication_policy", "capture_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("runtime capture receipt field closure differs")
    row = dict(value)
    unsigned = dict(row)
    declared = unsigned.pop("capture_digest", None)
    work_root = row.get("work_root")
    task_fd_binding = row.get("task_fd_binding")
    validated_work_root: Dict[str, Any] | None = None
    validated_task_fds: Dict[str, Any] | None = None
    task_root: Dict[str, Any] | None = None
    if work_root is not None:
        validated_work_root = validate_inherited_work_root_binding(
            work_root, verify_open_fds=False
        )
    if task_fd_binding is not None:
        validated_task_fds, task_root = _validate_task_fd_publication_binding(
            task_fd_binding, verify_open_fds=False
        )
    receipt_path_value = Path(row["receipt_path"])
    publication_authority_differs = (
        (validated_work_root is None) == (validated_task_fds is None)
        or (
            validated_work_root is not None
            and (
                row.get("publication_authority_kind") != "work_root"
                or validated_work_root != work_root
                or validated_work_root["target"] != row["target"]
                or validated_work_root["capture_receipt_path"]
                != row["receipt_path"]
            )
        )
        or (
            validated_task_fds is not None
            and (
                row.get("publication_authority_kind") != "task_root"
                or validated_task_fds != task_fd_binding
                or row["target"] not in {
                    "action_preservation_decoded_eval_decoder_adapter_v1.py",
                    "infer_lora.py",
                }
                or task_root is None
                or receipt_path_value.parent
                != Path(f"/proc/self/fd/{task_root['fd']}")
                or receipt_path_value.name in ("", ".", "..")
            )
        )
    )
    if (
        row["schema_version"] != CAPTURE_RECEIPT_SCHEMA
        or row["release_generation"] != RELEASE_GENERATION
        or row["target"] not in ALLOWED_PYTHON_TARGETS
        or row["target_mode"] != MEMBER_MODES[row["target"]]
        or type(row["target_size"]) is not int or row["target_size"] <= 0
        or type(row["member_count"]) is not int or row["member_count"] != len(EVAL_RELEASE_MEMBERS)
        or _validate_executable_binding(row["root_python"], label="receipt root Python")
        != row["root_python"]
        or _validate_executable_binding(row["frozen_python"], label="receipt frozen Python")
        != row["frozen_python"]
        or _validate_directory_binding_shape(
            row["site_packages"], label="receipt site-packages"
        ) != row["site_packages"]
        or type(row["release_artifacts"]) is not dict
        or set(row["release_artifacts"]) != {"archive", "manifest", "envelope"}
        or any(
            _validate_file_binding_shape(
                row["release_artifacts"][name], label=f"receipt release {name}"
            ) != row["release_artifacts"][name]
            for name in ("archive", "manifest", "envelope")
        )
        or row["release_artifacts"]["archive"]["sha256"]
        != row["archive_sha256"]
        or row["release_artifacts"]["manifest"]["sha256"]
        != row["manifest_sha256"]
        or row["release_artifacts"]["envelope"]["sha256"]
        != row["envelope_sha256"]
        or type(row["controller_authority"]) is not dict
        or set(row["controller_authority"])
        != _CONTROLLER_AUTHORITY_BINDING_FIELDS
        or _validate_file_binding_shape(
            row["controller_authority"]["receipt"],
            label="receipt controller authority",
        ) != row["controller_authority"]["receipt"]
        or not isinstance(row["controller_authority"].get("authority_digest"), str)
        or SHA256_RE.fullmatch(
            row["controller_authority"]["authority_digest"]
        ) is None
        or (
            row["captured_torchrun"] is not None
            and (
                _validate_torchrun_binding(
                    row["captured_torchrun"], label="receipt captured torchrun"
                ) != row["captured_torchrun"]
                or row["captured_torchrun"]["site_packages"]
                != row["site_packages"]
            )
        )
        or publication_authority_differs
        or not isinstance(row["receipt_path"], str)
        or not Path(row["receipt_path"]).is_absolute()
        or os.path.normpath(row["receipt_path"]) != row["receipt_path"]
        or row["publication_policy"] != {
            "create_only_o_excl": True, "mode": 0o444,
            "rank_zero_only": True, "same_fd_double_read_after_fsync": True,
            "named_identity_replay_after_write": True,
            "post_close_stable_double_read": True, "parent_directory_fsync": True,
        }
        or any(
            not isinstance(row.get(field), str) or SHA256_RE.fullmatch(row[field]) is None
            for field in (
                "archive_sha256", "manifest_sha256", "manifest_digest", "envelope_sha256",
                "envelope_digest", "all_members_capture_digest", "target_sha256",
                "target_arguments_sha256", "capture_digest",
            )
        )
        or not isinstance(row.get("content_revision"), str)
        or SHA1_RE.fullmatch(row["content_revision"]) is None
        or object_sha256(unsigned) != declared
    ):
        fail("runtime capture receipt value or digest differs")
    if verify_file:
        if receipt_path is None:
            fail("capture receipt verification path is absent")
        if str(receipt_path) != row["receipt_path"]:
            fail("capture receipt path differs from receipt authority")
        expected_raw = canonical_json_bytes(row) + b"\n"
        if validated_work_root is not None:
            live_work_root = validate_inherited_work_root_binding(
                validated_work_root,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
            if Path(receipt_path).parent != Path(live_work_root["path"]):
                fail("runtime capture receipt escapes inherited work root")
            raw, metadata = _stable_work_root_file_pair(
                live_work_root["root_fd"],
                {
                    "path": str(receipt_path),
                    "sha256": hashlib.sha256(expected_raw).hexdigest(),
                },
                label="runtime capture receipt",
            )
        else:
            assert validated_task_fds is not None and task_root is not None
            live_binding, live_task = _validate_task_fd_publication_binding(
                validated_task_fds,
                verify_open_fds=True,
                expected_inheritable=False,
            )
            if live_binding != validated_task_fds or live_task != task_root:
                fail("runtime capture receipt task-root replay differs")
            root_fd, basename = _task_publication_member(
                live_task, Path(receipt_path), label="runtime capture receipt"
            )
            descriptor = os.open(
                basename,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
            try:
                before = os.fstat(descriptor)
                first = _read_fd(descriptor)
                middle = os.fstat(descriptor)
                second = _read_fd(descriptor)
                after = os.fstat(descriptor)
                named = os.stat(
                    basename, dir_fd=root_fd, follow_symlinks=False
                )
            finally:
                os.close(descriptor)
            if (
                _identity(before) != _identity(middle)
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(named)
                or first != second
                or hashlib.sha256(first).hexdigest()
                != hashlib.sha256(expected_raw).hexdigest()
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1
            ):
                fail("runtime capture receipt task-root bytes differ")
            raw, metadata = first, before
        if raw != expected_raw:
            fail("runtime capture receipt bytes differ")
        if _file_binding_value(Path(receipt_path), raw, metadata)["nlink"] != 1:
            fail("runtime capture receipt link closure differs")
        authority_binding = _controller_authority_binding_shape(
            row["controller_authority"]
        )
        authority_raw, authority_metadata = _stable_capture(
            Path(authority_binding["receipt"]["path"]),
            label="receipt controller authority",
            expected_sha256=authority_binding["receipt"]["sha256"],
            expected_mode=0o444,
        )
        if (
            _file_binding_value(
                Path(authority_binding["receipt"]["path"]),
                authority_raw, authority_metadata,
            ) != authority_binding["receipt"]
        ):
            fail("receipt controller authority full identity differs")
        authority_receipt = validate_controller_authority_receipt(
            _decode_json(authority_raw, label="receipt controller authority")
        )
        if (
            authority_receipt["authority_digest"]
            != authority_binding["authority_digest"]
            or (
                row["captured_torchrun"] is not None
                and authority_receipt["torchrun"] != row["captured_torchrun"]
            )
        ):
            fail("receipt controller authority digest continuity differs")
    return row


def _bootstrap_identity() -> Optional[Mapping[str, Any]]:
    raw = os.environ.get(BOOTSTRAP_IDENTITY_ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DecodedEvalVerifiedReleaseError("bootstrap identity is not JSON") from error
    if type(value) is not dict or canonical_json_bytes(value).decode("utf-8") != raw:
        fail("bootstrap identity is not canonical")
    return _validate_executable_binding(value, label="bootstrap frozen Python")


def _captured_torchrun_from_environment() -> Optional[Mapping[str, Any]]:
    raw = os.environ.get(TORCHRUN_BINDING_ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DecodedEvalVerifiedReleaseError(
            "captured torchrun binding is not JSON"
        ) from error
    if type(value) is not dict or canonical_json_bytes(value).decode("utf-8") != raw:
        fail("captured torchrun binding is not canonical")
    return _validate_torchrun_binding(value, label="captured torchrun")


def _cleanup_darwin_captured_exec() -> None:
    """Remove only the private captured interpreter copy made by our bootstrap."""

    raw_path = os.environ.pop("APV2_EVAL_DARWIN_CAPTURED_EXEC", None)
    if raw_path is None:
        return
    identity = _bootstrap_identity()
    if sys.platform != "darwin" or identity is None:
        fail("Darwin captured interpreter cleanup authority differs")
    path = Path(raw_path)
    parent = path.parent
    if (
        path.name != "python" or parent.parent != Path("/private/tmp")
        or not parent.name.startswith("apv2-eval-held-python-")
        or path.is_symlink() or parent.is_symlink()
        or path.resolve(strict=True) != path or parent.resolve(strict=True) != parent
    ):
        fail("Darwin captured interpreter cleanup path differs")
    copied, details = _stable_capture(
        path, label="Darwin captured interpreter copy",
        expected_sha256=identity.get("sha256"), expected_mode=0o500,
    )
    if len(copied) != identity.get("size") or details.st_uid != os.getuid():
        fail("Darwin captured interpreter cleanup identity differs")
    parent.chmod(0o700)
    path.unlink()
    parent.rmdir()


def _require_exact15_consumption_arguments(
    target: str, arguments: Sequence[str]
) -> None:
    """Close the direct ``infer_lora`` legacy-path authority bypass.

    ``infer_lora.py`` remains shared with non-exact15 workflows, so its parser
    keeps the four authority options all-or-none.  Inside this exact15 runtime,
    however, every allowed inference invocation must carry all four literal
    bindings.  The target then replays them against the inherited 23/26-FD D0
    before resolving either checkpoint path.
    """

    if target != "infer_lora.py":
        return
    values: Dict[str, str] = {}
    required = (
        "--model-consumption-input",
        "--model-consumption-input-sha256",
        "--model-consumption-input-digest",
        "--task-input-digest",
    )
    for option in required:
        if any(argument.startswith(option + "=") for argument in arguments):
            fail(f"exact15 inference authority option must be a separate token: {option}")
        indices = [
            index for index, argument in enumerate(arguments)
            if argument == option
        ]
        if len(indices) != 1 or indices[0] + 1 >= len(arguments):
            fail(f"exact15 inference authority option differs: {option}")
        value = arguments[indices[0] + 1]
        if not isinstance(value, str) or not value or value.startswith("--"):
            fail(f"exact15 inference authority value differs: {option}")
        values[option] = value
    input_path = values["--model-consumption-input"]
    if not os.path.isabs(input_path) or os.path.normpath(input_path) != input_path:
        fail("exact15 inference consumption-input path differs")
    for option in required[1:]:
        if SHA256_RE.fullmatch(values[option]) is None:
            fail(f"exact15 inference authority digest differs: {option}")


def verified_python_run(
    *, release_binding: Mapping[str, Any],
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any], target: str,
    target_arguments: Sequence[str], capture_receipt_path: Path,
) -> int:
    """Execute an allowed target and its release imports from captured bytes."""

    if target not in ALLOWED_PYTHON_TARGETS:
        fail("verified Python target is not allowed")
    fd_targets = {
        "action_preservation_decoded_eval_decoder_adapter_v1.py",
        "infer_lora.py",
    }
    work_environment_present = WORK_ROOT_BINDING_ENV in os.environ
    task_environment_present = (
        "APV2_EVAL_INHERITED_AUTHORITY_FDS" in os.environ
    )
    if work_environment_present == task_environment_present:
        fail("verified target publication authority must be exactly one root")
    work_root: Dict[str, Any] | None = None
    task_fd_binding: Dict[str, Any] | None = None
    task_publication_root: Dict[str, Any] | None = None
    completion_anchor_channel: Dict[str, Any] | None = None
    if work_environment_present:
        if target in fd_targets:
            fail("task FD target received global work-root authority")
        work_root = seal_inherited_work_root_fds(
            load_inherited_work_root_environment(
                verify_open_fds=True,
                expected_inheritable=True,
                verify_entries=True,
            )
        )
        if (
            work_root["target"] != target
            or work_root["capture_receipt_path"] != str(capture_receipt_path)
        ):
            fail("verified target inherited work root continuity differs")
    else:
        if target not in fd_targets:
            fail("model authority FDs reached an unauthorized release target")
        task_fd_binding, task_publication_root = (
            _load_and_seal_task_fd_publication_environment()
        )
        _task_publication_member(
            task_publication_root,
            Path(capture_receipt_path),
            label="runtime capture receipt",
        )
    anchor_environment_present = COMPLETION_ANCHOR_CHANNEL_ENV in os.environ
    if target in _DYNAMIC_ANCHOR_TARGETS:
        if (
            work_root is None
            or not anchor_environment_present
            or os.environ.get(COMPLETION_ANCHOR_SENT_ENV) is not None
        ):
            fail("dynamic completion anchor channel authority differs")
        completion_anchor_channel = seal_completion_anchor_channel(
            expected_target=target
        )
    elif (
        anchor_environment_present
        or os.environ.get(COMPLETION_ANCHOR_SENT_ENV) is not None
    ):
        fail("completion anchor channel reached an unauthorized target")
    root_identity = _capture_interpreter(
        root_python_binding, label="root Python"
    )
    frozen_identity = _capture_interpreter(
        frozen_python_binding, label="frozen Python"
    )
    site_packages = replay_directory_binding(
        site_packages_binding, label="frozen site-packages"
    )
    release, manifest_value, archive_payloads, authority = replay_release_binding(
        release_binding
    )
    controller = replay_file_binding(
        controller_binding, label="detached controller"
    )
    captured_torchrun = _captured_torchrun_from_environment()
    controller_authority, _ = validate_controller_authority_binding(
        controller_authority_binding, controller_binding=controller,
        root_python_binding=root_identity, frozen_python_binding=frozen_identity,
        site_packages_binding=site_packages, release_binding=release,
        torchrun_binding=captured_torchrun,
        require_torchrun_continuity=captured_torchrun is not None,
        verify_file=True, replay_torchrun_source=False,
    )
    controller_digest_environment = os.environ.get(
        CONTROLLER_AUTHORITY_DIGEST_ENV
    )
    if (
        controller_digest_environment is not None
        and controller_digest_environment != controller_authority["authority_digest"]
    ):
        fail("bootstrap controller authority continuity differs")
    root = Path(release["release_root"]["path"])
    payloads = capture_materialized_release(root, manifest_value)
    if payloads != archive_payloads:
        fail("materialized release captured bytes differ from archive")
    bootstrap_identity = _bootstrap_identity()
    if bootstrap_identity is not None and dict(bootstrap_identity) != frozen_identity:
        fail("held-fd bootstrap interpreter identity differs from runtime replay")

    row_by_path = {row["path"]: row for row in manifest_value["files"]}
    target_row = row_by_path[target]
    arguments = list(target_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    _require_exact15_consumption_arguments(target, arguments)
    receipt_path = Path(capture_receipt_path)
    if not receipt_path.is_absolute():
        fail("capture receipt path must be absolute")
    receipt: Dict[str, Any] = {
        "schema_version": CAPTURE_RECEIPT_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        **authority,
        "target": target,
        "target_sha256": target_row["sha256"],
        "target_size": target_row["size"],
        "target_mode": target_row["mode"],
        "target_arguments_sha256": object_sha256(arguments),
        "root_python": root_identity, "frozen_python": frozen_identity,
        "site_packages": site_packages,
        "release_artifacts": authority["release_artifacts"],
        "controller_authority": controller_authority,
        "captured_torchrun": captured_torchrun,
        "work_root": work_root,
        "task_fd_binding": task_fd_binding,
        "publication_authority_kind": (
            "work_root" if work_root is not None else "task_root"
        ),
        "receipt_path": str(receipt_path),
        "publication_policy": {
            "create_only_o_excl": True, "mode": 0o444,
            "rank_zero_only": True, "same_fd_double_read_after_fsync": True,
            "named_identity_replay_after_write": True,
            "post_close_stable_double_read": True,
            "parent_directory_fsync": True,
        },
    }
    receipt["capture_digest"] = object_sha256(receipt)
    validate_capture_receipt(receipt)
    receipt_path_text = str(receipt_path)
    if _rank_is_zero():
        if work_root is not None:
            receipt_file = _write_create_only_in_work_root(
                work_root,
                receipt_path,
                canonical_json_bytes(receipt) + b"\n",
                mode=0o444,
                label="runtime capture receipt",
            )
        else:
            assert task_fd_binding is not None
            assert task_publication_root is not None
            receipt_file = _write_create_only_in_task_root(
                task_fd_binding,
                task_publication_root,
                receipt_path,
                canonical_json_bytes(receipt) + b"\n",
                mode=0o444,
                label="runtime capture receipt",
            )
        if receipt_file["sha256"] != hashlib.sha256(
            canonical_json_bytes(receipt) + b"\n"
        ).hexdigest():
            fail("runtime capture receipt held publication differs")

    method_root = root / MEMBER_ROOT
    modules: Dict[str, Tuple[Path, bytes]] = {}
    namespace_roots = {"tools": method_root / "tools"}
    module_directories = {method_root, method_root / "tools"}
    for relative, raw in payloads.items():
        pure = PurePosixPath(relative)
        if pure.suffix != ".py":
            continue
        if len(pure.parts) == 1:
            module_name = pure.stem
        elif len(pure.parts) == 2 and pure.parts[0] == "tools":
            module_name = f"tools.{pure.stem}"
        else:
            fail("release Python namespace depth differs")
        if not all(part.isidentifier() for part in module_name.split(".")) or module_name in modules:
            fail("release Python module name is invalid or ambiguous")
        source = method_root / relative
        modules[module_name] = (source, raw)
        module_directories.add(source.parent)
    guarded_names = set(modules) | set(namespace_roots)
    if any(name in sys.modules for name in guarded_names):
        fail("release-local Python module was imported before verified capture")
    for name, module in tuple(sys.modules.items()):
        source_value = getattr(module, "__file__", None)
        if not isinstance(source_value, str):
            continue
        try:
            source_path = Path(source_value).resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if _path_is_within(source_path, root):
            fail(f"unknown release-root module was imported before capture: {name}")
    forbidden_roots = tuple(sorted({root, *module_directories}, key=str))
    old_path = list(sys.path)
    old_meta_path = list(sys.meta_path)
    old_argv = list(sys.argv)
    old_dont_write = sys.dont_write_bytecode
    old_capture_digest = os.environ.get(CAPTURE_DIGEST_ENV)
    old_capture_receipt = os.environ.get(CAPTURE_RECEIPT_ENV)
    loaded_before = set(sys.modules)
    finder = _CapturedReleaseFinder(
        modules=modules, namespace_roots=namespace_roots,
        forbidden_roots=forbidden_roots,
    )
    target_path = method_root / target
    target_package = "tools" if target.startswith("tools/") else None
    try:
        _strip_forbidden_sys_path(forbidden_roots)
        if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
            fail("automatic site customization was loaded before verified target")
        if site_packages["path"] not in sys.path:
            sys.path.append(site_packages["path"])
        sys.meta_path.insert(0, finder)
        fd_environment_present = (
            "APV2_EVAL_INHERITED_AUTHORITY_FDS" in os.environ
        )
        if target in fd_targets:
            if not fd_environment_present or task_fd_binding is None:
                fail("model authority FD environment is absent")
            fd_authority = __import__(
                "action_preservation_decoded_eval_model_authority_v2"
            )
            inherited_fds = fd_authority.load_inherited_fd_environment(
                verify_open_fds=True, expected_inheritable=False
            )
            if inherited_fds != task_fd_binding:
                fail("model authority FD/runtime publication binding differs")
        elif fd_environment_present:
            fail("model authority FDs reached an unauthorized release target")
        sys.dont_write_bytecode = True
        sys.argv = [str(target_path), *arguments]
        os.environ[CAPTURE_DIGEST_ENV] = receipt["capture_digest"]
        os.environ[CAPTURE_RECEIPT_ENV] = receipt_path_text
        globals_value = {
            "__name__": "__main__", "__file__": str(target_path),
            "__package__": target_package,
            "__loader__": _CapturedModuleLoader("__main__", target_path, payloads[target]),
            "__spec__": None, "__cached__": None, "__builtins__": __builtins__,
        }
        exec(
            compile(payloads[target], str(target_path), "exec", dont_inherit=True),
            globals_value,
        )
    finally:
        missing_completion_anchor = False
        if completion_anchor_channel is not None:
            sent_digest = os.environ.pop(COMPLETION_ANCHOR_SENT_ENV, None)
            missing_completion_anchor = (
                type(sent_digest) is not str
                or SHA256_RE.fullmatch(sent_digest) is None
            )
            _close_completion_anchor_channel(completion_anchor_channel)
        if work_root is not None:
            validate_inherited_work_root_binding(
                work_root,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        else:
            assert task_fd_binding is not None
            _validate_task_fd_publication_binding(
                task_fd_binding,
                verify_open_fds=True,
                expected_inheritable=False,
            )
        sys.path[:] = old_path
        sys.meta_path[:] = old_meta_path
        sys.argv[:] = old_argv
        sys.dont_write_bytecode = old_dont_write
        if old_capture_digest is None:
            os.environ.pop(CAPTURE_DIGEST_ENV, None)
        else:
            os.environ[CAPTURE_DIGEST_ENV] = old_capture_digest
        if old_capture_receipt is None:
            os.environ.pop(CAPTURE_RECEIPT_ENV, None)
        else:
            os.environ[CAPTURE_RECEIPT_ENV] = old_capture_receipt
        for name in set(sys.modules) - loaded_before:
            module = sys.modules.get(name)
            loader = getattr(module, "__loader__", None)
            if isinstance(loader, _CapturedModuleLoader):
                sys.modules.pop(name, None)
        if missing_completion_anchor:
            fail("verified target returned without a completion anchor")
    return 0


def _canonical_binding_argument(value: Mapping[str, Any]) -> str:
    return canonical_json_bytes(dict(value)).decode("utf-8")


def _controller_authority_binding_shape(value: Mapping[str, Any]) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != _CONTROLLER_AUTHORITY_BINDING_FIELDS:
        fail("controller authority binding field closure differs")
    row = {
        "receipt": _validate_file_binding_shape(
            value["receipt"], label="controller authority receipt"
        ),
        "authority_digest": value.get("authority_digest"),
    }
    if (
        not isinstance(row["authority_digest"], str)
        or SHA256_RE.fullmatch(row["authority_digest"]) is None
    ):
        fail("controller authority digest differs")
    return row


def verified_runtime_arguments(
    *, release_binding: Mapping[str, Any],
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any], target: str,
    args: Sequence[str], capture_receipt_path: str,
) -> list[str]:
    """Return the canonical JSON CLI for a captured runtime invocation."""

    if target not in ALLOWED_PYTHON_TARGETS:
        fail("verified target differs")
    _require_exact15_consumption_arguments(target, list(args))
    receipt = Path(capture_receipt_path)
    if not receipt.is_absolute() or os.path.normpath(str(receipt)) != str(receipt):
        fail("capture receipt path must be canonical and absolute")
    release = _validate_release_binding(release_binding)
    root = _validate_executable_binding(root_python_binding, label="root Python")
    frozen = _validate_executable_binding(
        frozen_python_binding, label="frozen Python"
    )
    site_packages = _validate_directory_binding_shape(
        site_packages_binding, label="frozen site-packages"
    )
    controller = _validate_file_binding_shape(
        controller_binding, label="detached controller"
    )
    controller_authority = _controller_authority_binding_shape(
        controller_authority_binding
    )
    return [
        "verified-run",
        "--release-binding-json", _canonical_binding_argument(release),
        "--root-python-binding-json", _canonical_binding_argument(root),
        "--frozen-python-binding-json", _canonical_binding_argument(frozen),
        "--site-packages-binding-json", _canonical_binding_argument(site_packages),
        "--controller-binding-json", _canonical_binding_argument(controller),
        "--controller-authority-binding-json",
        _canonical_binding_argument(controller_authority),
        "--target", target, "--capture-receipt", str(receipt),
        "--", *list(args),
    ]


ISOLATED_TORCHRUN_BOOTSTRAP = r'''import hashlib,importlib,json,os,stat,sys,types
source,origin,declared,handler_source,handler_origin,handler_declared,site_raw=sys.argv[1:8]
arguments=sys.argv[8:]
def pairs(items):
 value={}
 for key,item in items:
  if key in value: raise RuntimeError("duplicate torchrun binding key")
  value[key]=item
 return value
def canonical(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def ident(value):
 return {"path":site["path"],"size":value.st_size,"mode":stat.S_IMODE(value.st_mode),"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"nlink":value.st_nlink,"rdev":value.st_rdev,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
site=json.loads(site_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
fields={"path","size","mode","device","inode","uid","gid","nlink","rdev","blocks","mtime_ns","ctime_ns"}
if type(site) is not dict or set(site)!=fields or canonical(site)!=site_raw: raise RuntimeError("torchrun site-packages binding differs")
if not os.path.isabs(site["path"]) or os.path.normpath(site["path"])!=site["path"] or os.path.islink(site["path"]) or os.path.realpath(site["path"])!=site["path"]: raise RuntimeError("torchrun site-packages path differs")
flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)
if not hasattr(os,"O_DIRECTORY") or not hasattr(os,"O_NOFOLLOW"): raise RuntimeError("torchrun safe directory replay unavailable")
fd=os.open(site["path"],flags|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 before=os.fstat(fd); middle=os.fstat(fd); named=os.lstat(site["path"]); after=os.fstat(fd)
finally: os.close(fd)
if not stat.S_ISDIR(before.st_mode) or ident(before)!=site or ident(middle)!=site or ident(named)!=site or ident(after)!=site: raise RuntimeError("torchrun site-packages full identity differs")
source_raw=source.encode("utf-8","strict")
if hashlib.sha256(source_raw).hexdigest()!=declared or declared!="1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c" or len(source_raw)!=31587: raise RuntimeError("captured torchrun SHA/size differs")
if not os.path.isabs(origin) or os.path.normpath(origin)!=origin: raise RuntimeError("captured torchrun origin differs")
handler_raw=handler_source.encode("utf-8","strict")
if hashlib.sha256(handler_raw).hexdigest()!=handler_declared or handler_declared!="9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87" or len(handler_raw)!=2436: raise RuntimeError("captured torchrun subprocess handler SHA/size differs")
expected_handler=site["path"]+"/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py"
if handler_origin!=expected_handler or not os.path.isabs(handler_origin) or os.path.normpath(handler_origin)!=handler_origin: raise RuntimeError("captured torchrun subprocess handler origin differs")
if "sitecustomize" in sys.modules or "usercustomize" in sys.modules: raise RuntimeError("automatic site customization was loaded")
if any(name=="torch" or name.startswith("torch.") for name in sys.modules): raise RuntimeError("torch was imported before captured subprocess handler preload")
if site["path"] not in sys.path: sys.path.append(site["path"])
fd_raw=os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
if fd_raw is None: raise RuntimeError("torchrun inherited authority FD binding is absent")
fd_binding=json.loads(fd_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
fd_fields={"schema_version","task_id","model_capture_digest","adapter_capture_digest","fd_count","fd_rows","fd_rows_digest","namespace_root_count","publication_root_count","exact_allowlist_only","proc_self_fd_consumption_required","cross_process_proc_fd_access_forbidden","ptrace_authorization_used","fd_binding_digest"}
if type(fd_binding) is not dict or set(fd_binding)!=fd_fields or canonical(fd_binding)!=fd_raw: raise RuntimeError("torchrun inherited authority FD binding differs")
unsigned=dict(fd_binding); claimed=unsigned.pop("fd_binding_digest",None)
if fd_binding.get("schema_version")!="bernini-action-preservation-inherited-fd-binding-v3" or fd_binding.get("exact_allowlist_only") is not True or fd_binding.get("proc_self_fd_consumption_required") is not True or fd_binding.get("cross_process_proc_fd_access_forbidden") is not True or fd_binding.get("ptrace_authorization_used") is not False or claimed!=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest(): raise RuntimeError("torchrun inherited authority FD policy differs")
fd_rows=fd_binding.get("fd_rows")
if type(fd_rows) is not list or fd_binding.get("fd_count")!=len(fd_rows) or fd_binding.get("fd_rows_digest")!=hashlib.sha256(canonical(fd_rows).encode("utf-8")).hexdigest(): raise RuntimeError("torchrun inherited authority FD rows differ")
def fd_ident(value):
 return {"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"nlink":value.st_nlink,"rdev":value.st_rdev,"size":value.st_size,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
authority_fds=[]
scope_roles_relatives=[]
identity_fields={"device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"}
immutable_identity_fields={"device","inode","uid","gid","mode","rdev"}
model_relatives={".gitattributes","README.md","assets/arena.png","assets/bernini-icon.png","config.json","scheduler/scheduler_config.json","text_encoder/config.json","text_encoder/model-00001-of-00005.safetensors","text_encoder/model-00002-of-00005.safetensors","text_encoder/model-00003-of-00005.safetensors","text_encoder/model-00004-of-00005.safetensors","text_encoder/model-00005-of-00005.safetensors","text_encoder/model.safetensors.index.json","tokenizer/special_tokens_map.json","tokenizer/spiece.model","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","transformer/config.json","transformer/diffusion_pytorch_model-00001-of-00002.safetensors","transformer/diffusion_pytorch_model-00002-of-00002.safetensors","transformer/diffusion_pytorch_model.safetensors.index.json","vae/config.json","vae/diffusion_pytorch_model.safetensors"}
adapter_relatives={"receipt.json","adapter/adapter_config.json","adapter/adapter_model.safetensors"}
def fd_row_matches(row,expected_inheritable):
 observed=fd_ident(os.fstat(row["fd"])); named=fd_ident(os.lstat(row["source_path"])); mutable=row["scope"]=="task" and row["role"]=="publication_root"
 return ((not mutable and observed==row["identity"] and named==row["identity"]) or (mutable and observed==named and {field:observed[field] for field in immutable_identity_fields}=={field:row["identity"][field] for field in immutable_identity_fields})) and os.get_inheritable(row["fd"]) is expected_inheritable
for row in fd_rows:
 if type(row) is not dict or set(row)!={"fd","scope","role","relative_path","source_path","identity"} or type(row.get("fd")) is not int or row["fd"]<3 or row.get("scope") not in {"model","adapter","task"} or row.get("role") not in {"file","namespace_root","publication_root"} or type(row.get("relative_path")) is not str or type(row.get("source_path")) is not str or not os.path.isabs(row["source_path"]) or os.path.normpath(row["source_path"])!=row["source_path"] or type(row.get("identity")) is not dict or set(row["identity"])!=identity_fields or any(type(value) is not int for value in row["identity"].values()) or (row["role"]=="file" and not stat.S_ISREG(row["identity"]["mode"])) or (row["role"]!="file" and not stat.S_ISDIR(row["identity"]["mode"])) or not fd_row_matches(row,True): raise RuntimeError("torchrun inherited authority FD identity differs")
 authority_fds.append(row["fd"])
 scope_roles_relatives.append((row["scope"],row["role"],row["relative_path"]))
expected_scope_roles_relatives={("model","file",relative) for relative in model_relatives}|{("model","namespace_root","."),("task","publication_root",".")}
if fd_binding.get("adapter_capture_digest") is not None: expected_scope_roles_relatives|={("adapter","file",relative) for relative in adapter_relatives}|{("adapter","namespace_root",".")}
if authority_fds!=sorted(authority_fds) or len(authority_fds)!=len(set(authority_fds)) or set(scope_roles_relatives)!=expected_scope_roles_relatives or len(scope_roles_relatives)!=len(expected_scope_roles_relatives) or (fd_binding.get("adapter_capture_digest") is None)!=(len(authority_fds)==25) or fd_binding.get("namespace_root_count")!=(1 if len(authority_fds)==25 else 2) or fd_binding.get("publication_root_count")!=1: raise RuntimeError("torchrun inherited authority FD allowlist differs")
for authority_fd in authority_fds: os.set_inheritable(authority_fd,False)
if any(os.get_inheritable(authority_fd) for authority_fd in authority_fds): raise RuntimeError("torchrun authority FDs remain inheritable")
handler_name="torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler"
handler_module=types.ModuleType(handler_name)
handler_module.__file__=handler_origin
handler_module.__package__="torch.distributed.elastic.multiprocessing.subprocess_handler"
handler_module.__loader__=None
handler_module.__spec__=None
handler_module.__cached__=None
handler_module.__builtins__=__builtins__
sys.modules[handler_name]=handler_module
exec(compile(handler_source,handler_origin,"exec",dont_inherit=True),handler_module.__dict__)
if handler_module.SubprocessHandler.__module__!=handler_name or handler_module.SubprocessHandler.__qualname__!="SubprocessHandler": raise RuntimeError("captured subprocess handler class differs")
no_python=[index for index,value in enumerate(arguments) if value=="--no-python"]
if len(no_python)!=1 or no_python[0]+1>=len(arguments): raise RuntimeError("captured torchrun rank target differs")
expected_rank=tuple(arguments[no_python[0]+1:])
def authority_popen(self,args,env):
 if tuple(args)!=expected_rank or type(env) is not dict or env.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")!=fd_raw or env.get("APV2_EVAL_WORK_ROOT_AUTHORITY") is not None: raise RuntimeError("torchrun rank subprocess target differs")
 for row in fd_rows:
  if not fd_row_matches(row,False): raise RuntimeError("pre-rank authority FD identity differs")
 process=handler_module.subprocess.Popen(args=args,env=env,stdout=self._stdout,stderr=self._stderr,start_new_session=True,close_fds=True,pass_fds=tuple(authority_fds))
 if any(os.get_inheritable(authority_fd) for authority_fd in authority_fds): raise RuntimeError("torchrun parent authority FD leaked inheritable")
 return process
handler_module.SubprocessHandler._popen=authority_popen
handler_package=importlib.import_module("torch.distributed.elastic.multiprocessing.subprocess_handler")
handlers_module=importlib.import_module("torch.distributed.elastic.multiprocessing.subprocess_handler.handlers")
api_module=importlib.import_module("torch.distributed.elastic.multiprocessing.api")
if handler_package.SubprocessHandler is not handler_module.SubprocessHandler or handlers_module.SubprocessHandler is not handler_module.SubprocessHandler or api_module.SubprocessHandler is not handler_module.SubprocessHandler: raise RuntimeError("captured subprocess handler aliases differ")
sys.argv=[origin,*arguments]
module=types.ModuleType("__main__")
module.__file__=origin
module.__package__="torch.distributed"
module.__loader__=None
module.__spec__=None
module.__cached__=None
module.__builtins__=__builtins__
sys.modules["__main__"]=module
exec(compile(source,origin,"exec",dont_inherit=True),module.__dict__)'''


_ROOT_BOOTSTRAP_TEMPLATE = r'''import hashlib,io,json,os,re,stat,sys,tarfile,tempfile
FILE_FIELDS={"path","sha256","size","mode","device","inode","uid","gid","nlink","rdev","blocks","mtime_ns","ctime_ns"}
DIR_FIELDS=FILE_FIELDS-{"sha256"}
WORK_FIELDS={"schema_version","path","parent_path","parent_fd","root_fd","parent_identity","root_identity","parent_immutable_identity","root_immutable_identity","entries","work_root_authority","deployment_receipt","source_spec_authority","work_root_authority_digest","deployment_receipt_digest","source_spec_authority_digest","target","capture_receipt_path","exact_two_directory_fds","fds_inheritable_only_across_verified_exec","binding_digest"}
WORK_AUTH_FIELDS={"schema_version","path","parent_path","creation_identity","immutable_identity","parent_immutable_identity","initial_entries","retained_parent_fd_through_request_publication","retained_root_fd_through_request_publication","authority_digest"}
WORK_IDENT={"device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"}
WORK_IMM={"device","inode","uid","gid","mode","rdev"}
TASK_FIELDS={"schema_version","task_id","model_capture_digest","adapter_capture_digest","fd_count","fd_rows","fd_rows_digest","namespace_root_count","publication_root_count","exact_allowlist_only","proc_self_fd_consumption_required","cross_process_proc_fd_access_forbidden","ptrace_authorization_used","fd_binding_digest"}
TASK_ROW_FIELDS={"fd","scope","role","relative_path","source_path","identity"}
MODEL_RELATIVES=[".gitattributes","README.md","assets/arena.png","assets/bernini-icon.png","config.json","scheduler/scheduler_config.json","text_encoder/config.json","text_encoder/model-00001-of-00005.safetensors","text_encoder/model-00002-of-00005.safetensors","text_encoder/model-00003-of-00005.safetensors","text_encoder/model-00004-of-00005.safetensors","text_encoder/model-00005-of-00005.safetensors","text_encoder/model.safetensors.index.json","tokenizer/special_tokens_map.json","tokenizer/spiece.model","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","transformer/config.json","transformer/diffusion_pytorch_model-00001-of-00002.safetensors","transformer/diffusion_pytorch_model-00002-of-00002.safetensors","transformer/diffusion_pytorch_model.safetensors.index.json","vae/config.json","vae/diffusion_pytorch_model.safetensors"]
ADAPTER_RELATIVES=["receipt.json","adapter/adapter_config.json","adapter/adapter_model.safetensors"]
RELEASE_FIELDS={"release_root","archive","manifest","manifest_digest","content_revision","envelope","envelope_digest","all_members_capture_digest"}
EXPECTED_NAMES=["action_preservation_decoded_eval_aggregate_v2.py","action_preservation_decoded_eval_bridge_v1.py","action_preservation_decoded_eval_decoder_adapter_v1.py","action_preservation_decoded_eval_executor_v2.py","action_preservation_decoded_eval_launcher_v1.py","action_preservation_decoded_eval_model_authority_v2.py","action_preservation_decoded_eval_plan_v1.py","action_preservation_decoded_eval_verified_release_v1.py","action_preservation_gate_v1.py","action_preservation_loop_controller_v1.py","infer_lora.py","self_generated_action_preservation_v2.py","tools/build_renderer_dataset.py","tools/materialize_vae.py","train_lora.py"]
ISOLATED=@@ISOLATED_TORCHRUN_REPR@@
def pairs(items):
 value={}
 for key,item in items:
  if key in value: raise RuntimeError("duplicate bootstrap JSON key")
  value[key]=item
 return value
def canonical(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def decode_argument(raw,label):
 try: value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 except (ValueError,TypeError,UnicodeError) as error: raise RuntimeError(label+" is not JSON") from error
 if type(value) is not dict or canonical(value)!=raw: raise RuntimeError(label+" is not canonical")
 return value
def decode_file(raw,label):
 try: text=raw.decode("utf-8","strict"); value=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 except (ValueError,TypeError,UnicodeError) as error: raise RuntimeError(label+" is not JSON") from error
 if type(value) is not dict or canonical(value)+"\n"!=text: raise RuntimeError(label+" is not canonical")
 return value
def digest(value,field):
 unsigned=dict(value); declared=unsigned.pop(field,None)
 observed=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest()
 if declared!=observed: raise RuntimeError(field+" differs")
 return declared
def shape(value,fields,label):
 if type(value) is not dict or set(value)!=fields: raise RuntimeError(label+" field closure differs")
 if type(value.get("path")) is not str or not os.path.isabs(value["path"]) or os.path.normpath(value["path"])!=value["path"]: raise RuntimeError(label+" path differs")
 for field in fields-{"path","sha256"}:
  if type(value.get(field)) is not int or value[field]<0: raise RuntimeError(label+" identity differs")
 if "sha256" in fields and (type(value.get("sha256")) is not str or len(value["sha256"])!=64 or any(x not in "0123456789abcdef" for x in value["sha256"])): raise RuntimeError(label+" SHA differs")
 if value["mode"]&~0o7777: raise RuntimeError(label+" mode differs")
 return value
def ident(value,path):
 return {"path":path,"size":value.st_size,"mode":stat.S_IMODE(value.st_mode),"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"nlink":value.st_nlink,"rdev":value.st_rdev,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
def work_ident(value):
 return {"size":value.st_size,"mode":value.st_mode,"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"nlink":value.st_nlink,"rdev":value.st_rdev,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
def readfd(fd):
 os.lseek(fd,0,os.SEEK_SET); out=[]
 while True:
  block=os.read(fd,1024*1024)
  if not block: return b"".join(out)
  out.append(block)
def stable_file(binding,label,keep=False,executable=False):
 value=shape(binding,FILE_FIELDS,label); path=value["path"]
 if os.path.islink(path) or os.path.realpath(path)!=path: raise RuntimeError(label+" canonical path differs")
 if not hasattr(os,"O_NOFOLLOW"): raise RuntimeError("safe file capture unavailable")
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0)); retained=False
 try:
  before=os.fstat(fd); first=readfd(fd); middle=os.fstat(fd); second=readfd(fd); after=os.fstat(fd); named=os.lstat(path)
  observed={"sha256":hashlib.sha256(first).hexdigest(),**ident(before,path)}
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or observed!=value or ident(middle,path)!=ident(before,path) or ident(after,path)!=ident(before,path) or ident(named,path)!=ident(before,path) or first!=second or len(first)!=before.st_size: raise RuntimeError(label+" full physical identity differs")
  if executable and not before.st_mode&0o111: raise RuntimeError(label+" is not executable")
  if keep: retained=True; return fd,first
  return first
 finally:
  if not retained: os.close(fd)
def stable_directory(binding,label):
 value=shape(binding,DIR_FIELDS,label); path=value["path"]
 if os.path.islink(path) or os.path.realpath(path)!=path or not hasattr(os,"O_DIRECTORY") or not hasattr(os,"O_NOFOLLOW"): raise RuntimeError(label+" canonical directory differs")
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
 try: before=os.fstat(fd); middle=os.fstat(fd); named=os.lstat(path); after=os.fstat(fd)
 finally: os.close(fd)
 if not stat.S_ISDIR(before.st_mode) or ident(before,path)!=value or ident(middle,path)!=value or ident(named,path)!=value or ident(after,path)!=value: raise RuntimeError(label+" full physical identity differs")
 return value
def work_identity(value,fields,label):
 if type(value) is not dict or set(value)!=fields or any(type(value.get(field)) is not int or value[field]<0 for field in fields) or not stat.S_ISDIR(value["mode"]): raise RuntimeError(label+" identity differs")
 return value
def work_pair(value,root,entries,root_fd,label):
 if type(value) is not dict or set(value)!={"path","sha256"} or type(value.get("path")) is not str or os.path.dirname(value["path"])!=root or os.path.basename(value["path"]) in {"",".",".."} or os.path.basename(value["path"]) not in entries or type(value.get("sha256")) is not str or len(value["sha256"])!=64 or any(char not in "0123456789abcdef" for char in value["sha256"]): raise RuntimeError("inherited "+label+" binding differs")
 fd=os.open(os.path.basename(value["path"]),os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=root_fd)
 try:
  before=os.fstat(fd); first=readfd(fd); middle=os.fstat(fd); second=readfd(fd); after=os.fstat(fd); named=os.stat(os.path.basename(value["path"]),dir_fd=root_fd,follow_symlinks=False)
 finally: os.close(fd)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o444 or work_ident(before)!=work_ident(middle) or work_ident(before)!=work_ident(after) or work_ident(before)!=work_ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=value["sha256"]: raise RuntimeError("inherited "+label+" physical replay differs")
 return value,first
def inherited_work(expected_inheritable):
 raw=os.environ.get("APV2_EVAL_WORK_ROOT_AUTHORITY")
 if raw is None: raise RuntimeError("inherited work root environment is absent")
 value=decode_argument(raw,"inherited work root binding")
 if set(value)!=WORK_FIELDS or value.get("schema_version")!="bernini-action-preservation-decoded-eval-inherited-work-root-v2" or digest(value,"binding_digest")!=value["binding_digest"]: raise RuntimeError("inherited work root binding closure differs")
 path=value.get("path"); parent=value.get("parent_path"); capture=value.get("capture_receipt_path"); parent_fd=value.get("parent_fd"); root_fd=value.get("root_fd"); entries=value.get("entries")
 if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path)!=path or type(parent) is not str or not os.path.isabs(parent) or os.path.normpath(parent)!=parent or os.path.dirname(path)!=parent or type(capture) is not str or not os.path.isabs(capture) or os.path.dirname(capture)!=path or os.path.basename(capture) in {"",".",".."}: raise RuntimeError("inherited work root path differs")
 if type(parent_fd) is not int or type(root_fd) is not int or parent_fd<3 or root_fd<3 or parent_fd==root_fd or type(entries) is not list or entries!=sorted(set(entries)) or any(type(item) is not str or item in {"",".",".."} for item in entries) or os.path.basename(capture) in entries: raise RuntimeError("inherited work root FD/entry closure differs")
 parent_identity=work_identity(value.get("parent_identity"),WORK_IDENT,"work parent"); root_identity=work_identity(value.get("root_identity"),WORK_IDENT,"work root"); parent_immutable=work_identity(value.get("parent_immutable_identity"),WORK_IMM,"immutable work parent"); root_immutable=work_identity(value.get("root_immutable_identity"),WORK_IMM,"immutable work root")
 authority=value.get("work_root_authority")
 if type(authority) is not dict or set(authority)!=WORK_AUTH_FIELDS or authority.get("schema_version")!="bernini-action-preservation-decoded-eval-work-root-authority-v1" or digest(authority,"authority_digest")!=authority["authority_digest"]: raise RuntimeError("inherited work root authority differs")
 creation=work_identity(authority.get("creation_identity"),WORK_IDENT,"work authority creation"); authority_root_immutable=work_identity(authority.get("immutable_identity"),WORK_IMM,"work authority root"); authority_parent_immutable=work_identity(authority.get("parent_immutable_identity"),WORK_IMM,"work authority parent")
 if authority.get("path")!=path or authority.get("parent_path")!=parent or stat.S_IMODE(creation["mode"])!=0o700 or authority_root_immutable!={field:creation[field] for field in WORK_IMM} or authority_root_immutable!=root_immutable or authority_parent_immutable!=parent_immutable or authority.get("initial_entries")!=[] or authority.get("retained_parent_fd_through_request_publication") is not True or authority.get("retained_root_fd_through_request_publication") is not True or authority.get("authority_digest")!=value.get("work_root_authority_digest"): raise RuntimeError("inherited work root authority continuity differs")
 deployment_pair,deployment_raw=work_pair(value.get("deployment_receipt"),path,entries,root_fd,"deployment receipt"); source_pair,source_raw=work_pair(value.get("source_spec_authority"),path,entries,root_fd,"source spec authority")
 deployment_value=decode_file(deployment_raw,"inherited deployment receipt"); source_value=decode_file(source_raw,"inherited source spec authority")
 if deployment_pair["path"]==source_pair["path"] or deployment_value.get("receipt_digest")!=value.get("deployment_receipt_digest") or deployment_value.get("work_root_authority")!=authority or source_value.get("receipt_digest")!=value.get("source_spec_authority_digest") or source_value.get("work_root_authority")!=authority or source_value.get("deployment_receipt_digest")!=value.get("deployment_receipt_digest") or {field:parent_identity[field] for field in WORK_IMM}!=parent_immutable or {field:root_identity[field] for field in WORK_IMM}!=root_immutable or value.get("exact_two_directory_fds") is not True or value.get("fds_inheritable_only_across_verified_exec") is not True or value.get("target") not in EXPECTED_NAMES: raise RuntimeError("inherited work root policy differs")
 for field in ("work_root_authority_digest","deployment_receipt_digest","source_spec_authority_digest","binding_digest"):
  item=value.get(field)
  if type(item) is not str or len(item)!=64 or any(char not in "0123456789abcdef" for char in item): raise RuntimeError("inherited work root digest differs")
 parent_before=os.fstat(parent_fd); root_before=os.fstat(root_fd); first=os.listdir(root_fd); root_middle=os.fstat(root_fd); second=os.listdir(root_fd); root_after=os.fstat(root_fd); named_root=os.stat(os.path.basename(path),dir_fd=parent_fd,follow_symlinks=False); parent_after=os.fstat(parent_fd); named_parent=os.lstat(parent)
 if work_ident(parent_before)!=parent_identity or work_ident(parent_after)!=parent_identity or work_ident(named_parent)!=parent_identity or work_ident(root_before)!=root_identity or work_ident(root_middle)!=root_identity or work_ident(root_after)!=root_identity or work_ident(named_root)!=root_identity or sorted(first)!=entries or sorted(second)!=entries or os.get_inheritable(parent_fd) is not expected_inheritable or os.get_inheritable(root_fd) is not expected_inheritable: raise RuntimeError("inherited work root physical replay differs")
 return value
def inherited_task(expected_inheritable):
 raw=os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
 if raw is None: raise RuntimeError("inherited task FD environment is absent")
 value=decode_argument(raw,"inherited task FD binding")
 if set(value)!=TASK_FIELDS or value.get("schema_version")!="bernini-action-preservation-inherited-fd-binding-v3" or digest(value,"fd_binding_digest")!=value["fd_binding_digest"]: raise RuntimeError("inherited task FD binding closure differs")
 rows=value.get("fd_rows"); count=value.get("fd_count"); adapter=value.get("adapter_capture_digest")
 if type(value.get("task_id")) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",value["task_id"]) is None or type(value.get("model_capture_digest")) is not str or len(value["model_capture_digest"])!=64 or any(char not in "0123456789abcdef" for char in value["model_capture_digest"]) or (adapter is not None and (type(adapter) is not str or len(adapter)!=64 or any(char not in "0123456789abcdef" for char in adapter))): raise RuntimeError("inherited task FD digest/task differs")
 if type(rows) is not list or type(count) is not int or count not in {25,29} or len(rows)!=count or value.get("fd_rows_digest")!=hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest() or value.get("namespace_root_count")!=(1 if count==25 else 2) or value.get("publication_root_count")!=1 or value.get("exact_allowlist_only") is not True or value.get("proc_self_fd_consumption_required") is not True or value.get("cross_process_proc_fd_access_forbidden") is not True or value.get("ptrace_authorization_used") is not False or (adapter is None)!=(count==25): raise RuntimeError("inherited task FD policy differs")
 expected={("model","file",relative) for relative in MODEL_RELATIVES}|{("model","namespace_root","."),("task","publication_root",".")}
 if adapter is not None: expected|={("adapter","file",relative) for relative in ADAPTER_RELATIVES}|{("adapter","namespace_root",".")}
 fds=[]; observed_roles=[]
 for row in rows:
  if type(row) is not dict or set(row)!=TASK_ROW_FIELDS or type(row.get("fd")) is not int or row["fd"]<3 or row.get("scope") not in {"model","adapter","task"} or row.get("role") not in {"file","namespace_root","publication_root"} or type(row.get("relative_path")) is not str or type(row.get("source_path")) is not str or not os.path.isabs(row["source_path"]) or os.path.normpath(row["source_path"])!=row["source_path"]: raise RuntimeError("inherited task FD row differs")
  identity=work_identity(row.get("identity"),WORK_IDENT,"inherited task FD"); before=work_ident(os.fstat(row["fd"])); named=work_ident(os.lstat(row["source_path"])); mutable=row["scope"]=="task" and row["role"]=="publication_root"
  if (row["role"]=="file" and not stat.S_ISREG(identity["mode"])) or (row["role"]!="file" and not stat.S_ISDIR(identity["mode"])) or (not mutable and (before!=identity or named!=identity)) or (mutable and (before!=named or {field:before[field] for field in WORK_IMM}!={field:identity[field] for field in WORK_IMM})) or os.get_inheritable(row["fd"]) is not expected_inheritable: raise RuntimeError("inherited task FD physical replay differs")
  fds.append(row["fd"]); observed_roles.append((row["scope"],row["role"],row["relative_path"]))
 if fds!=sorted(fds) or len(fds)!=len(set(fds)) or len(observed_roles)!=len(expected) or set(observed_roles)!=expected: raise RuntimeError("inherited task FD exact allowlist differs")
 return value
action=sys.argv[1]
if action not in {"runtime","frozen-exec","captured-torchrun"}: raise RuntimeError("bootstrap action differs")
work_root=None
task_fds=None
if action=="runtime":
 has_work=os.environ.get("APV2_EVAL_WORK_ROOT_AUTHORITY") is not None; has_task=os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS") is not None
 if has_work==has_task: raise RuntimeError("runtime bootstrap requires exactly one publication authority")
 if has_work:
  work_root=inherited_work(True)
  os.set_inheritable(work_root["parent_fd"],False); os.set_inheritable(work_root["root_fd"],False)
  inherited_work(False)
 else:
  task_fds=inherited_task(True)
  for item in task_fds["fd_rows"]: os.set_inheritable(item["fd"],False)
  inherited_task(False)
elif action=="captured-torchrun":
 if os.environ.get("APV2_EVAL_WORK_ROOT_AUTHORITY") is not None: raise RuntimeError("work root authority reached captured torchrun")
 task_fds=inherited_task(True)
 for item in task_fds["fd_rows"]: os.set_inheritable(item["fd"],False)
 inherited_task(False)
elif os.environ.get("APV2_EVAL_WORK_ROOT_AUTHORITY") is not None: raise RuntimeError("work root authority reached an unauthorized child bootstrap")
elif os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS") is not None: raise RuntimeError("task FD authority reached an unauthorized child bootstrap")
root=decode_argument(sys.argv[2],"root Python binding")
frozen=decode_argument(sys.argv[3],"frozen Python binding")
site=decode_argument(sys.argv[4],"site-packages binding")
release=decode_argument(sys.argv[5],"release binding")
controller=decode_argument(sys.argv[6],"controller binding")
authority_binding=decode_argument(sys.argv[7],"controller authority binding")
torch_binding=None
if action=="captured-torchrun":
 torch_binding=decode_argument(sys.argv[8],"captured torchrun binding"); payload=sys.argv[9:]
else: payload=sys.argv[8:]
if set(release)!=RELEASE_FIELDS: raise RuntimeError("release binding closure differs")
for field in ("manifest_digest","envelope_digest","all_members_capture_digest"):
 if type(release.get(field)) is not str or len(release[field])!=64 or any(x not in "0123456789abcdef" for x in release[field]): raise RuntimeError("release digest differs")
if type(release.get("content_revision")) is not str or len(release["content_revision"])!=40 or any(x not in "0123456789abcdef" for x in release["content_revision"]): raise RuntimeError("content revision differs")
if os.path.realpath(sys.executable)!=root.get("path"): raise RuntimeError("running root Python differs")
stable_file(root,"root Python",executable=True)
stable_directory(site,"site-packages")
stable_directory(release["release_root"],"materialized release root")
stable_file(controller,"detached controller")
manifest_raw=stable_file(release["manifest"],"release manifest")
manifest=decode_file(manifest_raw,"release manifest")
if manifest.get("schema_version")!="bernini-action-preservation-decoded-eval-source-release-v2" or manifest.get("release_generation")!="preservation-v2-decoded-eval-exact15-r3" or manifest.get("member_root")!="methods/bernini_action_editing" or manifest.get("content_revision")!=release["content_revision"] or digest(manifest,"manifest_digest")!=release["manifest_digest"]: raise RuntimeError("manifest authority differs")
rows=manifest.get("files")
if EXPECTED_NAMES!=sorted(EXPECTED_NAMES) or type(rows) is not list or len(rows)!=15 or [row.get("path") if type(row) is dict else None for row in rows]!=EXPECTED_NAMES or hashlib.sha1(canonical(rows).encode("utf-8")).hexdigest()!=release["content_revision"]: raise RuntimeError("manifest exact15 closure differs")
for row in rows:
 if set(row)!={"path","mode","size","sha256"} or type(row["size"]) is not int or row["size"]<=0 or row["mode"] not in {0o444,0o555} or type(row["sha256"]) is not str or len(row["sha256"])!=64: raise RuntimeError("manifest row differs")
archive_raw=stable_file(release["archive"],"release archive")
captured={}; captured_rows=[]
with tarfile.open(fileobj=io.BytesIO(archive_raw),mode="r:") as handle:
 members=handle.getmembers()
 if [member.name for member in members] != ["methods/bernini_action_editing/"+name for name in EXPECTED_NAMES]: raise RuntimeError("archive exact15 closure differs")
 for member,row in zip(members,rows):
  if not member.isfile() or member.issym() or member.islnk() or member.linkname or member.pax_headers or member.mode!=row["mode"] or member.uid!=0 or member.gid!=0 or member.uname or member.gname or member.mtime!=0 or member.size!=row["size"]: raise RuntimeError("archive member metadata differs")
  stream=handle.extractfile(member); raw=b"" if stream is None else stream.read()
  if len(raw)!=row["size"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]: raise RuntimeError("archive member payload differs")
  captured[row["path"]]=raw; captured_rows.append({"path":row["path"],"mode":row["mode"],"size":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
if hashlib.sha256(canonical(captured_rows).encode("utf-8")).hexdigest()!=release["all_members_capture_digest"]: raise RuntimeError("all-members capture digest differs")
envelope_raw=stable_file(release["envelope"],"deployment envelope")
envelope=decode_file(envelope_raw,"deployment envelope")
if envelope.get("schema_version")!="bernini-action-preservation-decoded-eval-deployment-v1" or envelope.get("release_generation")!="preservation-v2-decoded-eval-exact15-r3" or envelope.get("detached_controller_authority_receipt_required") is not True or digest(envelope,"envelope_digest")!=release["envelope_digest"]: raise RuntimeError("deployment envelope authority differs")
if envelope.get("source_archive")!={"basename":"source.tar","mode":0o444,"sha256":release["archive"]["sha256"]}: raise RuntimeError("archive envelope binding differs")
source_manifest=envelope.get("source_manifest")
if type(source_manifest) is not dict or source_manifest!={"basename":"source.manifest.json","mode":0o444,"sha256":release["manifest"]["sha256"],"manifest_digest":release["manifest_digest"],"content_revision":release["content_revision"],"file_count":15}: raise RuntimeError("manifest envelope binding differs")
if type(authority_binding) is not dict or set(authority_binding)!={"receipt","authority_digest"}: raise RuntimeError("controller authority binding closure differs")
authority_raw=stable_file(authority_binding["receipt"],"controller authority receipt")
authority=decode_file(authority_raw,"controller authority receipt")
authority_fields={"schema_version","release_generation","controller","root_python","frozen_python","site_packages","release","torchrun","create_only_o_excl","same_fd_double_read_after_fsync","named_identity_replay_after_write","automatic_scientific_promotion_authorized","authority_digest"}
authorized_torch=authority.get("torchrun")
if authorized_torch is not None:
 if type(authorized_torch) is not dict or set(authorized_torch)!={"source","subprocess_handler_source","site_packages"} or authorized_torch.get("site_packages")!=site: raise RuntimeError("preauthorized torchrun binding differs")
 shape(authorized_torch["source"],FILE_FIELDS,"preauthorized torchrun source"); shape(authorized_torch["subprocess_handler_source"],FILE_FIELDS,"preauthorized torchrun subprocess handler source"); shape(authorized_torch["site_packages"],DIR_FIELDS,"preauthorized torchrun site-packages")
 if authorized_torch["source"]["path"]!=site["path"]+"/torch/distributed/run.py" or authorized_torch["source"]["sha256"]!="1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c" or authorized_torch["source"]["size"]!=31587 or authorized_torch["subprocess_handler_source"]["path"]!=site["path"]+"/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py" or authorized_torch["subprocess_handler_source"]["sha256"]!="9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87" or authorized_torch["subprocess_handler_source"]["size"]!=2436: raise RuntimeError("preauthorized torchrun path/bytes differ")
if set(authority)!=authority_fields or authority.get("schema_version")!="bernini-action-preservation-decoded-eval-controller-authority-v1" or authority.get("release_generation")!="preservation-v2-decoded-eval-exact15-r3" or authority.get("controller")!=controller or authority.get("root_python")!=root or authority.get("frozen_python")!=frozen or authority.get("site_packages")!=site or authority.get("release")!=release or (action=="captured-torchrun" and authorized_torch!=torch_binding) or authority.get("create_only_o_excl") is not True or authority.get("same_fd_double_read_after_fsync") is not True or authority.get("named_identity_replay_after_write") is not True or authority.get("automatic_scientific_promotion_authorized") is not False or digest(authority,"authority_digest")!=authority_binding.get("authority_digest"): raise RuntimeError("controller authority continuity differs")
inherited_torch=None
inherited_torch_raw=os.environ.get("APV2_EVAL_CAPTURED_TORCHRUN_BINDING")
if inherited_torch_raw is not None:
 inherited_torch=decode_argument(inherited_torch_raw,"inherited captured torchrun binding")
 if inherited_torch!=authorized_torch: raise RuntimeError("inherited captured torchrun authority differs")
frozen_fd,frozen_raw=stable_file(frozen,"frozen Python",keep=True,executable=True)
environment={key:value for key,value in os.environ.items() if key not in {"PYTHONHOME","PYTHONPATH","PYTHONSTARTUP","PYTHONINSPECT","PYTHONUSERBASE","GCONV_PATH","LOCPATH","NLSPATH","HOSTALIASES","RES_OPTIONS","LOCALDOMAIN"} and not key.startswith(("LD_","DYLD_","PYTHON"))}
environment["APV2_EVAL_FROZEN_PYTHON_IDENTITY"]=canonical(frozen)
environment["APV2_EVAL_CONTROLLER_AUTHORITY_DIGEST"]=authority_binding["authority_digest"]
environment.pop("APV2_EVAL_CAPTURED_TORCHRUN_BINDING",None)
if inherited_torch is not None: environment["APV2_EVAL_CAPTURED_TORCHRUN_BINDING"]=canonical(inherited_torch)
if hasattr(os,"supports_fd") and os.execve in os.supports_fd:
 exec_target=frozen_fd
elif sys.platform=="darwin" and os.path.exists("/dev/fd"):
 directory=tempfile.mkdtemp(prefix="apv2-eval-held-python-",dir="/private/tmp"); exec_target=directory+"/python"
 copied=os.open(exec_target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o500)
 offset=0
 while offset<len(frozen_raw):
  count=os.write(copied,frozen_raw[offset:])
  if count<=0: raise RuntimeError("captured interpreter copy made no progress")
  offset+=count
 os.fchmod(copied,0o500); os.fsync(copied); os.close(copied); os.chmod(directory,0o500)
 environment["APV2_EVAL_DARWIN_CAPTURED_EXEC"]=exec_target
else: raise RuntimeError("held-fd exec is unavailable")
if action=="runtime":
 try: runtime=captured["action_preservation_decoded_eval_verified_release_v1.py"].decode("utf-8","strict")
 except UnicodeDecodeError as error: raise RuntimeError("verified runtime is not UTF-8") from error
 argv=[frozen["path"],"-I","-S","-B","-c",runtime,*payload]
elif action=="captured-torchrun":
 if type(torch_binding) is not dict or set(torch_binding)!={"source","subprocess_handler_source","site_packages"} or torch_binding.get("site_packages")!=site or torch_binding.get("source",{}).get("path")!=site["path"]+"/torch/distributed/run.py" or torch_binding.get("source",{}).get("sha256")!="1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c" or torch_binding.get("source",{}).get("size")!=31587 or torch_binding.get("subprocess_handler_source",{}).get("path")!=site["path"]+"/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py" or torch_binding.get("subprocess_handler_source",{}).get("sha256")!="9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87" or torch_binding.get("subprocess_handler_source",{}).get("size")!=2436: raise RuntimeError("captured torchrun binding differs")
 run_raw=stable_file(torch_binding["source"],"captured torchrun source")
 handler_raw=stable_file(torch_binding["subprocess_handler_source"],"captured torchrun subprocess handler source")
 try: run_source=run_raw.decode("utf-8","strict")
 except UnicodeDecodeError as error: raise RuntimeError("captured torchrun is not UTF-8") from error
 try: handler_source=handler_raw.decode("utf-8","strict")
 except UnicodeDecodeError as error: raise RuntimeError("captured torchrun subprocess handler is not UTF-8") from error
 environment["APV2_EVAL_CAPTURED_TORCHRUN_BINDING"]=canonical(torch_binding)
 argv=[frozen["path"],"-I","-S","-B","-c",ISOLATED,run_source,torch_binding["source"]["path"],torch_binding["source"]["sha256"],handler_source,torch_binding["subprocess_handler_source"]["path"],torch_binding["subprocess_handler_source"]["sha256"],canonical(site),*payload]
else:
 if payload[:3]!=["-I","-S","-B"] or any(payload[index:index+2]==["-m","torch.distributed.run"] for index in range(len(payload)-1)): raise RuntimeError("frozen argv isolation differs")
 argv=[frozen["path"],*payload]
if work_root is not None:
 inherited_work(False)
 os.set_inheritable(work_root["parent_fd"],True); os.set_inheritable(work_root["root_fd"],True)
if task_fds is not None:
 inherited_task(False)
 for item in task_fds["fd_rows"]: os.set_inheritable(item["fd"],True)
try: os.execve(exec_target,argv,environment)
finally:
 if work_root is not None:
  os.set_inheritable(work_root["parent_fd"],False); os.set_inheritable(work_root["root_fd"],False)
 if task_fds is not None:
  for item in task_fds["fd_rows"]: os.set_inheritable(item["fd"],False)
raise RuntimeError("held-fd exec unexpectedly returned")'''


ROOT_BOOTSTRAP_SOURCE = _ROOT_BOOTSTRAP_TEMPLATE.replace(
    "@@ISOLATED_TORCHRUN_REPR@@", repr(ISOLATED_TORCHRUN_BOOTSTRAP)
)


def _bootstrap_prefix(
    *, action: str, root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any],
    release_binding: Mapping[str, Any], controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any],
    torchrun_binding: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    if action not in {"runtime", "frozen-exec", "captured-torchrun"}:
        fail("bootstrap action differs")
    root = _validate_executable_binding(root_python_binding, label="root Python")
    frozen = _validate_executable_binding(frozen_python_binding, label="frozen Python")
    site_packages = _validate_directory_binding_shape(
        site_packages_binding, label="frozen site-packages"
    )
    release = _validate_release_binding(release_binding)
    controller = _validate_file_binding_shape(
        controller_binding, label="detached controller"
    )
    controller_authority = _controller_authority_binding_shape(
        controller_authority_binding
    )
    result = [
        root["path"], "-I", "-S", "-B", "-c", ROOT_BOOTSTRAP_SOURCE,
        action, _canonical_binding_argument(root),
        _canonical_binding_argument(frozen),
        _canonical_binding_argument(site_packages),
        _canonical_binding_argument(release),
        _canonical_binding_argument(controller),
        _canonical_binding_argument(controller_authority),
    ]
    if action == "captured-torchrun":
        if torchrun_binding is None:
            fail("captured torchrun binding is absent")
        torchrun = _validate_torchrun_binding(
            torchrun_binding, label="captured torchrun"
        )
        if torchrun["site_packages"] != site_packages:
            fail("captured torchrun site-packages continuity differs")
        result.append(_canonical_binding_argument(torchrun))
    elif torchrun_binding is not None:
        fail("unexpected captured torchrun binding")
    return result


def verified_target_argv(
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any], release_binding: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any], target: str,
    args: Sequence[str], capture_receipt_path: str,
) -> list[str]:
    """Return a root ``-I -S -B`` argv ending in captured ``verified-run``."""

    runtime_args = verified_runtime_arguments(
        release_binding=release_binding,
        root_python_binding=root_python_binding,
        frozen_python_binding=frozen_python_binding,
        site_packages_binding=site_packages_binding,
        controller_binding=controller_binding,
        controller_authority_binding=controller_authority_binding,
        target=target, args=args, capture_receipt_path=capture_receipt_path,
    )
    return [
        *_bootstrap_prefix(
            action="runtime", root_python_binding=root_python_binding,
            frozen_python_binding=frozen_python_binding,
            site_packages_binding=site_packages_binding,
            release_binding=release_binding, controller_binding=controller_binding,
            controller_authority_binding=controller_authority_binding,
        ),
        *runtime_args,
    ]


def frozen_exec_argv(
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any], release_binding: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any], frozen_args: Sequence[str],
) -> list[str]:
    """Held-fd exec a non-torch frozen command under the complete authority."""

    arguments = list(frozen_args)
    if arguments[:3] != ["-I", "-S", "-B"]:
        fail("frozen arguments must begin with -I -S -B")
    if any(
        arguments[index:index + 2] == ["-m", "torch.distributed.run"]
        for index in range(len(arguments) - 1)
    ):
        fail("torch.distributed.run must use captured_torchrun_argv")
    return [
        *_bootstrap_prefix(
            action="frozen-exec", root_python_binding=root_python_binding,
            frozen_python_binding=frozen_python_binding,
            site_packages_binding=site_packages_binding,
            release_binding=release_binding, controller_binding=controller_binding,
            controller_authority_binding=controller_authority_binding,
        ),
        *arguments,
    ]


def captured_torchrun_argv(
    root_python_binding: Mapping[str, Any],
    frozen_python_binding: Mapping[str, Any],
    site_packages_binding: Mapping[str, Any],
    torchrun_binding: Mapping[str, Any], release_binding: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    controller_authority_binding: Mapping[str, Any],
    torchrun_arguments: Sequence[str], rank_target_argv: Sequence[str],
) -> list[str]:
    """Run captured ``torch/distributed/run.py`` without Python path reopening."""

    arguments = list(torchrun_arguments)
    forbidden = ("--no-python", "--no_python", "--max-restarts", "--max_restarts")
    if any(
        argument in {"-m", "torch.distributed.run"}
        or any(argument == flag or argument.startswith(flag + "=") for flag in forbidden)
        for argument in arguments
    ):
        fail("torchrun arguments contain a bootstrap-controlled option")
    torchrun = _validate_torchrun_binding(
        torchrun_binding, label="captured torchrun"
    )
    validate_controller_authority_binding(
        controller_authority_binding,
        controller_binding=controller_binding,
        root_python_binding=root_python_binding,
        frozen_python_binding=frozen_python_binding,
        site_packages_binding=site_packages_binding,
        release_binding=release_binding, torchrun_binding=torchrun,
        require_torchrun_continuity=True, verify_file=True,
        replay_torchrun_source=True,
    )
    rank = list(rank_target_argv)
    expected_prefix = _bootstrap_prefix(
        action="runtime", root_python_binding=root_python_binding,
        frozen_python_binding=frozen_python_binding,
        site_packages_binding=site_packages_binding,
        release_binding=release_binding, controller_binding=controller_binding,
        controller_authority_binding=controller_authority_binding,
    )
    if rank[:len(expected_prefix)] != expected_prefix:
        fail("torchrun rank target is not the matching verified root bootstrap")
    return [
        *_bootstrap_prefix(
            action="captured-torchrun", root_python_binding=root_python_binding,
            frozen_python_binding=frozen_python_binding,
            site_packages_binding=site_packages_binding,
            release_binding=release_binding, controller_binding=controller_binding,
            controller_authority_binding=controller_authority_binding,
            torchrun_binding=torchrun,
        ),
        "--max-restarts=0", *arguments, "--no-python", *rank,
    ]


def _add_release_options(value: argparse.ArgumentParser, *, include_root: bool) -> None:
    if include_root:
        value.add_argument("--release-root", required=True)
    value.add_argument("--archive", required=True)
    value.add_argument("--expected-archive-sha256", required=True)
    value.add_argument("--manifest", required=True)
    value.add_argument("--expected-manifest-sha256", required=True)
    value.add_argument("--expected-manifest-digest", required=True)
    value.add_argument(
        "--expected-content-revision", "--method-revision",
        dest="revision", required=True,
    )
    value.add_argument("--envelope", required=True)
    value.add_argument("--expected-envelope-sha256", required=True)
    value.add_argument("--expected-envelope-digest", required=True)


def _binding_from_json_argument(raw: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DecodedEvalVerifiedReleaseError(
            f"{label} argument is not strict JSON"
        ) from error
    if type(value) is not dict or canonical_json_bytes(value).decode("utf-8") != raw:
        fail(f"{label} argument is not one canonical JSON object")
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    _add_release_options(extract, include_root=False)
    extract.add_argument("--output-root", required=True)
    run = commands.add_parser("verified-run")
    for option in (
        "release-binding", "root-python-binding", "frozen-python-binding",
        "site-packages-binding", "controller-binding",
        "controller-authority-binding",
    ):
        run.add_argument(f"--{option}-json", required=True)
    run.add_argument("--target", required=True, choices=sorted(ALLOWED_PYTHON_TARGETS))
    run.add_argument("--capture-receipt", required=True)
    run.add_argument("target_arguments", nargs=argparse.REMAINDER)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    _cleanup_darwin_captured_exec()
    args = parser().parse_args(argv)
    if args.command == "extract":
        result = extract_verified_release(
            archive=Path(args.archive),
            expected_archive_sha256=args.expected_archive_sha256,
            manifest=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_content_revision=args.revision,
            envelope=Path(args.envelope),
            expected_envelope_sha256=args.expected_envelope_sha256,
            output_root=Path(args.output_root),
        )
        if (
            result["manifest_digest"] != args.expected_manifest_digest
            or result["envelope_digest"] != args.expected_envelope_digest
        ):
            fail("extraction digest pins differ")
        print(canonical_json_bytes(result).decode("utf-8"), flush=True)
        return 0
    return verified_python_run(
        release_binding=_binding_from_json_argument(
            args.release_binding_json, label="release binding"
        ),
        root_python_binding=_binding_from_json_argument(
            args.root_python_binding_json, label="root Python binding"
        ),
        frozen_python_binding=_binding_from_json_argument(
            args.frozen_python_binding_json, label="frozen Python binding"
        ),
        site_packages_binding=_binding_from_json_argument(
            args.site_packages_binding_json, label="site-packages binding"
        ),
        controller_binding=_binding_from_json_argument(
            args.controller_binding_json, label="controller binding"
        ),
        controller_authority_binding=_binding_from_json_argument(
            args.controller_authority_binding_json,
            label="controller authority binding",
        ),
        target=args.target, target_arguments=args.target_arguments,
        capture_receipt_path=Path(args.capture_receipt),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_COMPLETION_ANCHOR_SCHEMA", "ALLOWED_PYTHON_TARGETS",
    "AUTHORITY", "CAPTURE_DIGEST_ENV",
    "CAPTURE_RECEIPT_ENV", "CAPTURE_RECEIPT_SCHEMA",
    "COMPLETION_ANCHOR_CHANNEL_ENV", "COMPLETION_ANCHOR_CHANNEL_SCHEMA",
    "COMPLETION_ANCHOR_SENT_ENV", "HOLDER_COMPLETION_ANCHOR_SCHEMA",
    "CONTROLLER_AUTHORITY_DIGEST_ENV", "CONTROLLER_AUTHORITY_SCHEMA",
    "DecodedEvalVerifiedReleaseError", "DIRECTORY_BINDING_FIELDS",
    "ENVELOPE_SCHEMA", "EVAL_RELEASE_MEMBERS", "FILE_BINDING_FIELDS",
    "ISOLATED_TORCHRUN_BOOTSTRAP", "MEMBER_MODES", "RELEASE_GENERATION",
    "ROOT_BOOTSTRAP_SOURCE", "SCHEMA_VERSION", "TORCHRUN_BINDING_ENV",
    "WORK_ROOT_BINDING_ENV", "WORK_ROOT_BINDING_SCHEMA",
    "TORCHRUN_SOURCE_SHA256", "TORCHRUN_SOURCE_SIZE",
    "TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH",
    "TORCHRUN_SUBPROCESS_HANDLER_SHA256",
    "TORCHRUN_SUBPROCESS_HANDLER_SIZE", "TRUSTED_EXACT15",
    "capture_directory_binding", "capture_executable_binding",
    "capture_file_binding", "capture_materialized_release",
    "capture_release_artifacts", "capture_release_binding",
    "capture_torchrun_binding", "captured_torchrun_argv", "content_revision",
    "extract_verified_release",
    "fixed_ustar_archive", "fixed_ustar_header", "frozen_exec_argv",
    "object_sha256", "publish_controller_authority_receipt",
    "publish_aggregate_completion_anchor",
    "publish_holder_completion_anchor",
    "replay_directory_binding", "replay_file_binding", "replay_release_binding",
    "replay_torchrun_binding",
    "validate_capture_receipt", "validate_controller_authority_binding",
    "validate_aggregate_completion_anchor",
    "validate_controller_authority_receipt", "validate_envelope",
    "validate_holder_completion_anchor",
    "validate_inherited_work_root_binding",
    "validate_manifest", "verified_python_run", "verified_runtime_arguments",
    "load_inherited_work_root_environment", "seal_inherited_work_root_fds",
    "verified_target_argv", "verify_archive_snapshot",
]
