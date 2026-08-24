#!/usr/bin/env python3
"""Bind a completed preservation-v2 training tree to one decoded-eval plan.

This is a local, create-only evidence bridge.  It does not launch decoding,
use or copy training loss as evaluation input, or authorize a scientific
transition.  It verifies the
published ``TRAINING_COMPLETE.json`` and ``logs/training-audit.json`` against
the physical 32-checkpoint tree, verifies the four source videos and the
pinned inference runtime, publishes the hash-only decoded-evaluation bundle,
and emits a separate physical-bindings authority consumed by the real decoder
adapter.
"""

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

import action_preservation_decoded_eval_plan_v1 as plan
import action_preservation_decoded_eval_verified_release_v1 as verified_release
import action_preservation_decoded_eval_model_authority_v2 as model_authority


SOURCE_RUNTIME_SCHEMA = "bernini-action-preservation-source-runtime-spec-v2"
PHYSICAL_BINDINGS_SCHEMA = "bernini-action-preservation-physical-bindings-v6"
BRIDGE_RECEIPT_SCHEMA = "bernini-action-preservation-eval-bridge-receipt-v7"
DEPLOYMENT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-deployment-authority-binding-v1"
)
SOURCE_SPEC_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-spec-authority-v2"
)
TRAINING_AUDIT_SERIALIZATION = (
    "python-json-sort-keys-indent2-ensure-ascii-false-finite-newline-v1"
)
SOURCE_PREPROCESSING_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-preprocessing-authority-v1"
)
SOURCE_PREPROCESSING_AUTHORITY_SHA256 = (
    "f0ee7196c00fb0dd0b4345707ec8a069ee2ba20a6f304b1982ef8d7945be15dd"
)
SOURCE_PREPROCESSING_AUTHORITY_DIGEST = (
    "6e19837719e052a735ff7c0258e16f0a9b72f4e9d3fe3a3751bad1847c8b8265"
)
SOURCE_PREPROCESSING_SERIALIZATION = "canonical-json-newline-v1"
EVAL_RELEASE_SCHEMA = verified_release.SCHEMA_VERSION
EVAL_RELEASE_BINDING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-release-binding-v3"
)

PHYSICAL_BINDINGS_FILENAME = "physical_bindings.json"
BRIDGE_RECEIPT_FILENAME = "bridge_receipt.json"
EVAL_RELEASE_MANIFEST_FILENAME = "source.manifest.json"
EVAL_RELEASE_MEMBER_ROOT = Path(verified_release.MEMBER_ROOT)
EVAL_RELEASE_MEMBERS = verified_release.EVAL_RELEASE_MEMBERS

ROOT_PYTHON_PATH = Path("/usr/bin/python3.10")
ROOT_PYTHON_UID = 0
ROOT_PYTHON_GID = 0
ROOT_PYTHON_MODE = 0o755
FFPROBE_PATH = Path("/usr/bin/ffprobe")
FFPROBE_UID = 0
FFPROBE_GID = 0
FFPROBE_MODE = 0o755

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class DecodedEvaluationBridgeError(RuntimeError):
    """A physical training/source/runtime binding is incomplete or hostile."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return plan.canonical_json_bytes(value)
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise DecodedEvaluationBridgeError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationBridgeError(f"{label} is not a lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise DecodedEvaluationBridgeError(f"{label} is not a lowercase SHA-1")
    return value


def _deployment_authority_from_work_root_binding(
    value: Mapping[str, Any], *, verify_open_fds: bool,
) -> dict[str, Any]:
    try:
        binding = verified_release.validate_inherited_work_root_binding(
            value,
            verify_open_fds=verify_open_fds,
            expected_inheritable=False if verify_open_fds else None,
            verify_entries=False,
            allow_root_metadata_change=verify_open_fds,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    result: dict[str, Any] = {
        "schema_version": DEPLOYMENT_AUTHORITY_SCHEMA,
        "work_root_authority": binding["work_root_authority"],
        "deployment_receipt": binding["deployment_receipt"],
        "source_spec_authority": binding["source_spec_authority"],
        "deployment_receipt_digest": binding[
            "deployment_receipt_digest"
        ],
        "source_spec_authority_digest": binding[
            "source_spec_authority_digest"
        ],
    }
    result["authority_digest"] = object_sha256(result)
    return result


_DEPLOYMENT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version", "release_generation", "work_root_authority",
        "work_root_capture_before_receipt",
        "work_root_expected_phase_a_entries",
        "work_root_held_fd_through_controller_publication",
        "deployment_request", "deployment_request_digest", "controller",
        "root_python", "frozen_python", "site_packages", "torchrun",
        "release", "verified_runtime_source", "verified_runtime",
        "source_runtime_spec_path", "source_spec_authority_receipt_path",
        "controller_authority", "literal_request_sha_required",
        "controller_executed_from_same_fd_captured_bytes",
        "verified_runtime_executed_from_same_fd_captured_bytes",
        "automatic_retry", "network_used", "scientific_promotion_authorized",
        "receipt_digest",
    }
)
_SOURCE_SPEC_AUTHORITY_FIELDS = frozenset(
    {
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
)
_AUTHORITY_FILE_FIELDS = frozenset(
    {
        "path", "sha256", "size", "mode", "device", "inode", "uid",
        "gid", "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
    }
)


def _authority_file_shape(
    value: Any, *, root: Path, label: str,
) -> dict[str, Any]:
    row = dict(_closed(value, _AUTHORITY_FILE_FIELDS, label=label))
    path = _absolute(row.get("path"), label=f"{label} path")
    if (
        path.parent != root
        or path.name in ("", ".", "..")
        or _sha(row.get("sha256"), label=f"{label} SHA") != row["sha256"]
        or any(
            type(row.get(field)) is not int or row[field] < 0
            for field in _AUTHORITY_FILE_FIELDS - {"path", "sha256"}
        )
        or row["mode"] != 0o444
        or row["nlink"] != 1
    ):
        raise DecodedEvaluationBridgeError(f"{label} physical binding differs")
    return row


def _strict_deployment_receipt(
    value: Any, *, work_root: Mapping[str, Any], expected_digest: str,
) -> dict[str, Any]:
    row = dict(_closed(value, _DEPLOYMENT_RECEIPT_FIELDS, label="deployment receipt"))
    claimed = _sha(row.get("receipt_digest"), label="deployment receipt digest")
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    if (
        claimed != expected_digest
        or claimed != object_sha256(unsigned)
        or row["work_root_authority"] != work_root
        or row["work_root_held_fd_through_controller_publication"] is not True
        or row["literal_request_sha_required"] is not True
        or row["controller_executed_from_same_fd_captured_bytes"] is not True
        or row["verified_runtime_executed_from_same_fd_captured_bytes"] is not True
        or row["automatic_retry"] is not False
        or row["network_used"] is not False
        or row["scientific_promotion_authorized"] is not False
    ):
        raise DecodedEvaluationBridgeError("deployment receipt authority differs")
    return row


def _strict_source_spec_authority(
    value: Any, *, root: Path, work_root: Mapping[str, Any],
    deployment_pair: Mapping[str, str], deployment: Mapping[str, Any],
    expected_digest: str, source_authority_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = dict(
        _closed(
            value, _SOURCE_SPEC_AUTHORITY_FIELDS,
            label="source spec authority receipt",
        )
    )
    claimed = _sha(
        row.get("receipt_digest"), label="source spec authority receipt digest"
    )
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    deployment_file = _authority_file_shape(
        row.get("deployment_receipt"), root=root,
        label="source authority deployment receipt",
    )
    source_spec_file = _authority_file_shape(
        row.get("source_runtime_spec"), root=root,
        label="authorized source/runtime spec",
    )
    expected_entries = row.get("work_root_expected_source_spec_entries")
    if (
        row.get("schema_version") != SOURCE_SPEC_AUTHORITY_SCHEMA
        or claimed != expected_digest
        or claimed != object_sha256(unsigned)
        or row.get("work_root_authority") != work_root
        or row.get("deployment_receipt_digest") != deployment["receipt_digest"]
        or deployment_file["path"] != deployment_pair["path"]
        or deployment_file["sha256"] != deployment_pair["sha256"]
        or row.get("release_generation") != deployment["release_generation"]
        or row.get("controller_authority") != deployment["controller_authority"]
        or row.get("receipt_path") != str(source_authority_path)
        or row.get("source_runtime_spec_digest") is None
        or _sha(
            row.get("source_runtime_spec_digest"),
            label="authorized source/runtime spec digest",
        ) != row["source_runtime_spec_digest"]
        or type(expected_entries) is not list
        or expected_entries != sorted(set(expected_entries))
        or source_spec_file["path"] != deployment["source_runtime_spec_path"]
        or Path(source_spec_file["path"]).name not in expected_entries
        or source_authority_path.name not in expected_entries
        or row.get("work_root_held_fd_through_source_spec_publication") is not True
        or row.get("literal_source_runtime_spec_sha_required") is not True
        or row.get("runtime_authority_continuity_verified") is not True
        or row.get("automatic_retry") is not False
        or row.get("network_used") is not False
        or row.get("scientific_promotion_authorized") is not False
    ):
        raise DecodedEvaluationBridgeError(
            "source spec authority receipt continuity differs"
        )
    return row, source_spec_file


def _authority_stat_binding(path: Path, value: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path), "sha256": "",
        "size": value.st_size, "mode": stat.S_IMODE(value.st_mode),
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "nlink": value.st_nlink, "rdev": value.st_rdev,
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
    }


def _stable_deployment_member(
    pair: Mapping[str, str], *, label: str,
    work_root_binding: Mapping[str, Any] | None,
) -> tuple[bytes, dict[str, Any]]:
    if work_root_binding is None:
        return _stable_file(
            pair["path"], label=label, expected_sha256=pair["sha256"]
        )
    try:
        live = verified_release.validate_inherited_work_root_binding(
            work_root_binding,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
        if Path(pair["path"]).parent != Path(live["path"]):
            raise DecodedEvaluationBridgeError(
                f"{label} escapes held deployment work root"
            )
        raw, info = verified_release._stable_work_root_file_pair(
            live["root_fd"], pair, label=label
        )
        verified_release.validate_inherited_work_root_binding(
            live,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    evidence = _authority_stat_binding(Path(pair["path"]), info)
    evidence["sha256"] = hashlib.sha256(raw).hexdigest()
    return raw, evidence


def _validate_deployment_authority(
    value: Any, *, verify_files: bool,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version", "work_root_authority", "deployment_receipt",
        "source_spec_authority", "deployment_receipt_digest",
        "source_spec_authority_digest", "authority_digest",
    }
    row = dict(_closed(value, fields, label="deployment authority"))
    try:
        work_root = verified_release._validate_work_root_authority_shape(
            row["work_root_authority"]
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    root = Path(work_root["path"])
    pairs: dict[str, dict[str, str]] = {}
    for field, label in (
        ("deployment_receipt", "deployment receipt"),
        ("source_spec_authority", "source spec authority"),
    ):
        item = row[field]
        if (
            type(item) is not dict
            or set(item) != {"path", "sha256"}
            or type(item.get("path")) is not str
            or Path(item["path"]).parent != root
            or Path(item["path"]).name in ("", ".", "..")
        ):
            raise DecodedEvaluationBridgeError(
                f"{label} deployment binding differs"
            )
        pairs[field] = {
            "path": item["path"],
            "sha256": _sha(item.get("sha256"), label=f"{label} file SHA"),
        }
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    if (
        row["schema_version"] != DEPLOYMENT_AUTHORITY_SCHEMA
        or pairs["deployment_receipt"]["path"]
        == pairs["source_spec_authority"]["path"]
        or _sha(
            row["deployment_receipt_digest"],
            label="deployment receipt object digest",
        ) != row["deployment_receipt_digest"]
        or _sha(
            row["source_spec_authority_digest"],
            label="source spec authority object digest",
        ) != row["source_spec_authority_digest"]
        or claimed != object_sha256(unsigned)
    ):
        raise DecodedEvaluationBridgeError(
            "deployment authority policy or digest differs"
        )
    row["work_root_authority"] = work_root
    row.update(pairs)
    if verify_files:
        if work_root_binding is None:
            try:
                root_info = root.lstat()
                parent_info = root.parent.lstat()
            except OSError as error:
                raise DecodedEvaluationBridgeError(
                    "deployment work root named replay is unavailable"
                ) from error
        else:
            try:
                live = verified_release.validate_inherited_work_root_binding(
                    work_root_binding,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
            if (
                live["work_root_authority"] != work_root
                or live["deployment_receipt"] != pairs["deployment_receipt"]
                or live["source_spec_authority"]
                != pairs["source_spec_authority"]
                or live["deployment_receipt_digest"]
                != row["deployment_receipt_digest"]
                or live["source_spec_authority_digest"]
                != row["source_spec_authority_digest"]
            ):
                raise DecodedEvaluationBridgeError(
                    "held deployment authority projection differs"
                )
            root_info = os.fstat(live["root_fd"])
            parent_info = os.fstat(live["parent_fd"])
        immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
        root_identity = {
            "device": root_info.st_dev, "inode": root_info.st_ino,
            "uid": root_info.st_uid, "gid": root_info.st_gid,
            "mode": root_info.st_mode, "rdev": root_info.st_rdev,
        }
        parent_identity = {
            "device": parent_info.st_dev, "inode": parent_info.st_ino,
            "uid": parent_info.st_uid, "gid": parent_info.st_gid,
            "mode": parent_info.st_mode, "rdev": parent_info.st_rdev,
        }
        if (
            set(root_identity) != immutable_fields
            or root_identity != work_root["immutable_identity"]
            or parent_identity != work_root["parent_immutable_identity"]
            or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_ISLNK(root_info.st_mode)
        ):
            raise DecodedEvaluationBridgeError(
                "deployment work root physical identity differs"
            )
        deployment_raw, deployment_file = _stable_deployment_member(
            pairs["deployment_receipt"],
            label="bound deployment receipt",
            work_root_binding=work_root_binding,
        )
        source_raw, source_file = _stable_deployment_member(
            pairs["source_spec_authority"],
            label="bound source spec authority",
            work_root_binding=work_root_binding,
        )
        deployment = _strict_deployment_receipt(
            _json(
            deployment_raw, label="bound deployment receipt", canonical=True
            ),
            work_root=work_root,
            expected_digest=row["deployment_receipt_digest"],
        )
        source = _json(
            source_raw, label="bound source spec authority", canonical=True
        )
        source, authorized_source_spec = _strict_source_spec_authority(
            source,
            root=root,
            work_root=work_root,
            deployment_pair=pairs["deployment_receipt"],
            deployment=deployment,
            expected_digest=row["source_spec_authority_digest"],
            source_authority_path=Path(pairs["source_spec_authority"]["path"]),
        )
        source_spec_raw, source_spec_file = _stable_deployment_member(
            {
                "path": authorized_source_spec["path"],
                "sha256": authorized_source_spec["sha256"],
            },
            label="authorized source/runtime spec",
            work_root_binding=work_root_binding,
        )
        source_spec_value = _json(
            source_spec_raw, label="authorized source/runtime spec", canonical=True
        )
        if (
            deployment_file["mode"] != 0o444
            or deployment_file["nlink"] != 1
            or source_file["mode"] != 0o444
            or source_file["nlink"] != 1
            or source_spec_file != authorized_source_spec
            or source_spec_value.get("spec_digest")
            != source["source_runtime_spec_digest"]
            or object_sha256(
                {
                    key: item for key, item in source_spec_value.items()
                    if key != "spec_digest"
                }
            ) != source["source_runtime_spec_digest"]
        ):
            raise DecodedEvaluationBridgeError(
                "deployment authority receipt continuity differs"
            )
        if work_root_binding is not None:
            try:
                verified_release.validate_inherited_work_root_binding(
                    work_root_binding,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
    return row


def _authorized_source_runtime_spec(
    deployment_authority: Mapping[str, Any], *,
    work_root_binding: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority = _validate_deployment_authority(
        deployment_authority,
        verify_files=True,
        work_root_binding=work_root_binding,
    )
    root = Path(authority["work_root_authority"]["path"])
    deployment_raw, _ = _stable_deployment_member(
        authority["deployment_receipt"],
        label="authorized deployment receipt replay",
        work_root_binding=work_root_binding,
    )
    deployment = _strict_deployment_receipt(
        _json(
            deployment_raw,
            label="authorized deployment receipt replay",
            canonical=True,
        ),
        work_root=authority["work_root_authority"],
        expected_digest=authority["deployment_receipt_digest"],
    )
    source_raw, _ = _stable_deployment_member(
        authority["source_spec_authority"],
        label="authorized source spec authority replay",
        work_root_binding=work_root_binding,
    )
    source_authority, source_spec_file = _strict_source_spec_authority(
        _json(
            source_raw,
            label="authorized source spec authority replay",
            canonical=True,
        ),
        root=root,
        work_root=authority["work_root_authority"],
        deployment_pair=authority["deployment_receipt"],
        deployment=deployment,
        expected_digest=authority["source_spec_authority_digest"],
        source_authority_path=Path(authority["source_spec_authority"]["path"]),
    )
    spec_raw, observed_spec_file = _stable_deployment_member(
        {
            "path": source_spec_file["path"],
            "sha256": source_spec_file["sha256"],
        },
        label="authorized source/runtime spec replay",
        work_root_binding=work_root_binding,
    )
    spec = _json(
        spec_raw, label="authorized source/runtime spec replay", canonical=True
    )
    if (
        observed_spec_file != source_spec_file
        or spec.get("spec_digest") != source_authority["source_runtime_spec_digest"]
        or object_sha256(
            {key: item for key, item in spec.items() if key != "spec_digest"}
        ) != source_authority["source_runtime_spec_digest"]
    ):
        raise DecodedEvaluationBridgeError(
            "authorized source/runtime spec continuity differs"
        )
    return authority, spec, source_spec_file


def _absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DecodedEvaluationBridgeError(f"{label} is not a path")
    path = Path(value)
    if not path.is_absolute() or value == os.path.sep or os.path.normpath(value) != value:
        raise DecodedEvaluationBridgeError(
            f"{label} must be a normalized absolute non-root path"
        )
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = _absolute(str(value), label=label)
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationBridgeError(f"{label} does not exist") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationBridgeError(f"{label} is not a plain directory")
    if path.resolve(strict=True) != path:
        raise DecodedEvaluationBridgeError(f"{label} is not canonical")
    return path


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return every stable inode field except atime.

    Eval authorities are consumed by same-UID processes.  A path/hash check is
    therefore not sufficient: a named file can be exchanged after hashing, or
    an external writable hard link can be added.  The identity is deliberately
    the same closure used by the verified runtime.
    """

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


def _read_file_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _stable_file(value: str | Path, *, label: str, expected_sha256: str | None = None) -> tuple[bytes, dict[str, Any]]:
    path = _absolute(str(value), label=label)
    if path.resolve(strict=True) != path:
        raise DecodedEvaluationBridgeError(f"{label} is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DecodedEvaluationBridgeError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        first = _read_file_descriptor(descriptor)
        middle = os.fstat(descriptor)
        second = _read_file_descriptor(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_ISLNK(named.st_mode)
        or _file_identity(before) != _file_identity(middle)
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(named)
        or first != second
        or len(first) != before.st_size
    ):
        raise DecodedEvaluationBridgeError(
            f"{label} changed during stable double read or has a hard link"
        )
    digest = hashlib.sha256(first).hexdigest()
    if expected_sha256 is not None and digest != _sha(expected_sha256, label=f"{label} expected SHA"):
        raise DecodedEvaluationBridgeError(f"{label} SHA differs")
    return first, {
        "path": str(path),
        "sha256": digest,
        "size": len(first),
        "mode": stat.S_IMODE(named.st_mode),
        "device": int(named.st_dev),
        "inode": int(named.st_ino),
        "uid": int(named.st_uid),
        "gid": int(named.st_gid),
        "nlink": int(named.st_nlink),
        "rdev": int(named.st_rdev),
        "blocks": int(getattr(named, "st_blocks", 0)),
        "mtime_ns": int(named.st_mtime_ns),
        "ctime_ns": int(named.st_ctime_ns),
    }


def _json(raw: bytes, *, label: str, canonical: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationBridgeError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise DecodedEvaluationBridgeError(f"{label} root is not an object")
    row = dict(value)
    if canonical and raw not in (canonical_json_bytes(row), canonical_json_bytes(row) + b"\n"):
        raise DecodedEvaluationBridgeError(f"{label} is not canonical JSON")
    return row


def _producer_training_audit_json(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode only the training producer's exact pretty-JSON serialization.

    ``TRAINING_COMPLETE.json`` independently pins the file SHA before this
    function is called.  This check additionally prevents a re-signed
    whitespace variant, duplicate-key object, or non-finite numeric token from
    becoming a second accepted representation of the same audit authority.
    Other decoded-eval JSON continues to use ``_json(..., canonical=True)``.
    """

    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise DecodedEvaluationBridgeError(
                    f"{label} contains a duplicate key"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise DecodedEvaluationBridgeError(
            f"cannot decode {label}: {error}"
        ) from error
    if type(value) is not dict:
        raise DecodedEvaluationBridgeError(f"{label} root is not an object")
    try:
        expected = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeError) as error:
        raise DecodedEvaluationBridgeError(
            f"{label} is not finite producer JSON"
        ) from error
    if raw != expected:
        raise DecodedEvaluationBridgeError(
            f"{label} producer-exact serialization differs"
        )
    return value


def _verify_object_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != digest:
        raise DecodedEvaluationBridgeError(f"{label} digest differs")
    return digest


def _file_binding(value: Any, *, label: str) -> dict[str, Any]:
    row = dict(_closed(value, {"path", "sha256"}, label=label))
    raw, evidence = _stable_file(
        row["path"], label=label, expected_sha256=_sha(row["sha256"], label=f"{label} SHA")
    )
    if not raw:
        raise DecodedEvaluationBridgeError(f"{label} is empty")
    return evidence


def _sealed_directory(path: Path, *, label: str) -> Path:
    directory = _plain_directory(path, label=label)
    if stat.S_IMODE(directory.lstat().st_mode) != 0o555:
        raise DecodedEvaluationBridgeError(f"{label} is not sealed mode 0555")
    return directory


def _exact_directory_entries(
    path: Path, expected: set[str], *, label: str
) -> None:
    try:
        names = {item.name for item in os.scandir(path)}
    except OSError as error:
        raise DecodedEvaluationBridgeError(f"cannot list {label}: {error}") from error
    if names != expected:
        raise DecodedEvaluationBridgeError(f"{label} exact entry closure differs")


def _stable_sealed_directory(
    path: Path, *, expected: set[str], label: str,
    expected_uid: int | None = None, expected_gid: int | None = None,
) -> os.stat_result:
    directory = _plain_directory(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    try:
        before = os.fstat(descriptor)
        first = tuple(sorted(os.listdir(descriptor)))
        middle = os.fstat(descriptor)
        second = tuple(sorted(os.listdir(descriptor)))
        after = os.fstat(descriptor)
        named = directory.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o555
        # Directory link counts are filesystem-specific (APFS counts regular
        # children here, while ext4 does not).  Exact double-scanned entries
        # and the stable full identity are the portable topology authority.
        or before.st_nlink < 1
        or _file_identity(before) != _file_identity(middle)
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(named)
        or first != second
        or set(first) != expected
        or (expected_uid is not None and before.st_uid != expected_uid)
        or (expected_gid is not None and before.st_gid != expected_gid)
    ):
        raise DecodedEvaluationBridgeError(
            f"{label} sealed identity or exact entry closure differs"
        )
    return before


def _captured_directory(path: Path, metadata: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": int(metadata.st_size),
        "mode": int(stat.S_IMODE(metadata.st_mode)),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
        "nlink": int(metadata.st_nlink),
        "rdev": int(metadata.st_rdev),
        "blocks": int(getattr(metadata, "st_blocks", 0)),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def load_eval_release_manifest(
    path: str | Path, *, expected_sha256: str, release_root: str | Path,
    archive_path: str | Path, expected_archive_sha256: str,
    envelope_path: str | Path, expected_envelope_sha256: str,
    expected_content_revision: str, expected_manifest_digest: str,
    expected_envelope_digest: str, verify_files: bool = True,
) -> dict[str, Any]:
    manifest_raw, manifest_file = _stable_file(
        path, label="eval release manifest", expected_sha256=expected_sha256
    )
    archive_raw, archive_file = _stable_file(
        archive_path, label="eval release archive",
        expected_sha256=expected_archive_sha256,
    )
    envelope_raw, envelope_file = _stable_file(
        envelope_path, label="eval release deployment envelope",
        expected_sha256=expected_envelope_sha256,
    )
    if any(
        item["mode"] != 0o444 or item["nlink"] != 1
        for item in (manifest_file, archive_file, envelope_file)
    ):
        raise DecodedEvaluationBridgeError(
            "eval release detached artifact topology differs"
        )
    try:
        manifest, envelope, archive_payloads, authority = (
            verified_release.capture_release_artifacts(
                archive=Path(archive_file["path"]),
                expected_archive_sha256=archive_file["sha256"],
                manifest=Path(manifest_file["path"]),
                expected_manifest_sha256=manifest_file["sha256"],
                expected_content_revision=expected_content_revision,
                envelope=Path(envelope_file["path"]),
                expected_envelope_sha256=envelope_file["sha256"],
            )
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    if (
        manifest_raw != verified_release.canonical_json_bytes(manifest) + b"\n"
        or envelope_raw != verified_release.canonical_json_bytes(envelope) + b"\n"
        or hashlib.sha256(archive_raw).hexdigest() != authority["archive_sha256"]
        or authority["manifest_digest"] != expected_manifest_digest
        or authority["envelope_digest"] != expected_envelope_digest
        or authority["content_revision"] != expected_content_revision
        or authority["member_count"] != len(EVAL_RELEASE_MEMBERS)
    ):
        raise DecodedEvaluationBridgeError(
            "eval release detached authority differs"
        )
    root = _absolute(str(release_root), label="eval release root")
    member_root = root / EVAL_RELEASE_MEMBER_ROOT
    members: list[dict[str, Any]] = []
    if verify_files:
        try:
            materialized_payloads = verified_release.capture_materialized_release(
                root, manifest
            )
        except verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if materialized_payloads != archive_payloads:
            raise DecodedEvaluationBridgeError(
                "materialized eval release differs from captured archive"
            )
        root_info = _stable_sealed_directory(
            root, expected={"methods"},
            label="eval release root",
        )
        _stable_sealed_directory(
            root / "methods", expected={"bernini_action_editing"},
            label="eval release methods root",
            expected_uid=root_info.st_uid, expected_gid=root_info.st_gid,
        )
        root_files = {
            relative for relative in EVAL_RELEASE_MEMBERS if "/" not in relative
        }
        _stable_sealed_directory(
            member_root, expected=root_files | {"tools"},
            label="eval release member root", expected_uid=root_info.st_uid,
            expected_gid=root_info.st_gid,
        )
        _stable_sealed_directory(
            member_root / "tools",
            expected={
                Path(relative).name for relative in EVAL_RELEASE_MEMBERS
                if "/" in relative
            },
            label="eval release tools root",
            expected_uid=root_info.st_uid, expected_gid=root_info.st_gid,
        )
        for item in manifest["files"]:
            _, evidence = _stable_file(
                member_root / item["path"],
                label=f"eval release {item['path']}",
                expected_sha256=item["sha256"],
            )
            if (
                evidence["size"] != item["size"]
                or evidence["mode"] != item["mode"]
                or evidence["uid"] != root_info.st_uid
                or evidence["gid"] != root_info.st_gid
            ):
                raise DecodedEvaluationBridgeError(
                    "eval release member physical metadata differs"
                )
            members.append({"relative_path": item["path"], **evidence})
    else:
        raise DecodedEvaluationBridgeError(
            "new eval release bindings require one physical capture"
        )
    binding: dict[str, Any] = {
        "schema_version": EVAL_RELEASE_BINDING_SCHEMA,
        "release_root": str(root),
        "release_root_directory": _captured_directory(root, root_info),
        "member_root": str(member_root),
        "archive_file": archive_file,
        "manifest_file": manifest_file,
        "manifest_digest": authority["manifest_digest"],
        "content_revision": authority["content_revision"],
        "envelope_file": envelope_file,
        "envelope_digest": authority["envelope_digest"],
        "all_members_capture_digest": authority["all_members_capture_digest"],
        "members": members,
        "member_count": len(EVAL_RELEASE_MEMBERS),
        "exact_member_closure": True,
    }
    binding["release_binding_digest"] = object_sha256(binding)
    return binding


def validate_eval_release_binding(
    value: Any, *, verify_files: bool
) -> dict[str, Any]:
    fields = {
        "schema_version", "release_root", "release_root_directory",
        "member_root", "archive_file",
        "manifest_file", "manifest_digest", "content_revision", "envelope_file",
        "envelope_digest", "all_members_capture_digest", "members",
        "member_count", "exact_member_closure", "release_binding_digest",
    }
    row = dict(_closed(value, fields, label="eval release binding"))
    if row["schema_version"] != EVAL_RELEASE_BINDING_SCHEMA:
        raise DecodedEvaluationBridgeError("eval release binding schema differs")
    release_root = _absolute(row["release_root"], label="bound eval release root")
    root_directory = _validate_captured_directory(
        row["release_root_directory"], label="bound eval release root",
        verify_directory=verify_files,
    )
    if root_directory["path"] != str(release_root):
        raise DecodedEvaluationBridgeError(
            "bound eval release root directory differs"
        )
    if row["member_root"] != str(release_root / EVAL_RELEASE_MEMBER_ROOT):
        raise DecodedEvaluationBridgeError("bound eval release member root differs")
    archive = _validate_captured_file(
        row["archive_file"], label="bound eval release archive",
        verify_file=verify_files,
    )
    manifest = _validate_captured_file(
        row["manifest_file"], label="bound eval release manifest",
        verify_file=verify_files,
    )
    envelope = _validate_captured_file(
        row["envelope_file"], label="bound eval release deployment envelope",
        verify_file=verify_files,
    )
    _sha(row["manifest_digest"], label="bound eval release manifest digest")
    _sha(row["envelope_digest"], label="bound eval release envelope digest")
    _sha(
        row["all_members_capture_digest"],
        label="bound eval release member capture digest",
    )
    _sha1(row["content_revision"], label="bound eval release content revision")
    if (
        row["member_count"] != len(EVAL_RELEASE_MEMBERS)
        or row["exact_member_closure"] is not True
        or not isinstance(row["members"], list)
        or len(row["members"]) != len(EVAL_RELEASE_MEMBERS)
    ):
        raise DecodedEvaluationBridgeError("bound eval release count differs")
    normalized_members: list[dict[str, Any]] = []
    for expected_relative, member_value in zip(EVAL_RELEASE_MEMBERS, row["members"]):
        member = dict(
            _closed(
                member_value, set(_CAPTURED_FILE_FIELDS) | {"relative_path"},
                label="bound eval release member",
            )
        )
        expected_mode = (
            0o555
            if expected_relative
            == "action_preservation_decoded_eval_decoder_adapter_v1.py"
            else 0o444
        )
        if (
            member["relative_path"] != expected_relative
            or member["path"] != str(release_root / EVAL_RELEASE_MEMBER_ROOT / expected_relative)
            or member["mode"] != expected_mode
        ):
            raise DecodedEvaluationBridgeError("bound eval release member differs")
        _validate_captured_file(
            {key: member[key] for key in _CAPTURED_FILE_FIELDS},
            label=f"bound eval release {expected_relative}",
            verify_file=verify_files,
        )
        normalized_members.append(member)
    _verify_object_digest(
        row, field="release_binding_digest", label="eval release binding"
    )
    if verify_files:
        replayed = load_eval_release_manifest(
            manifest["path"], expected_sha256=manifest["sha256"],
            release_root=release_root, archive_path=archive["path"],
            expected_archive_sha256=archive["sha256"],
            envelope_path=envelope["path"],
            expected_envelope_sha256=envelope["sha256"],
            expected_content_revision=row["content_revision"],
            expected_manifest_digest=row["manifest_digest"],
            expected_envelope_digest=row["envelope_digest"], verify_files=True,
        )
        if replayed != row:
            raise DecodedEvaluationBridgeError("eval release binding replay differs")
    row["archive_file"] = archive
    row["manifest_file"] = manifest
    row["envelope_file"] = envelope
    row["members"] = normalized_members
    row["release_root_directory"] = root_directory
    return row


def eval_release_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a physical release binding into the verified-runtime argv schema."""

    row = validate_eval_release_binding(value, verify_files=False)
    return {
        "release_root": row["release_root_directory"],
        "archive": row["archive_file"],
        "manifest": row["manifest_file"],
        "manifest_digest": row["manifest_digest"],
        "content_revision": row["content_revision"],
        "envelope": row["envelope_file"],
        "envelope_digest": row["envelope_digest"],
        "all_members_capture_digest": row["all_members_capture_digest"],
    }


def executable_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one captured executable into the runtime's closed binding."""

    row = _validate_captured_file(value, label="runtime executable", verify_file=False)
    return {key: row[key] for key in _CAPTURED_FILE_FIELDS}


def directory_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one captured directory into the runtime's closed binding."""

    row = _validate_captured_directory(
        value, label="runtime directory", verify_directory=False
    )
    return {key: row[key] for key in _CAPTURED_DIRECTORY_FIELDS}


def controller_authority_runtime_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the detached controller receipt into the runtime argv schema."""

    row = dict(
        _closed(
            value,
            {"receipt", "authority_digest"},
            label="controller authority binding",
        )
    )
    row["receipt"] = _validate_captured_file(
        row["receipt"], label="controller authority receipt", verify_file=False
    )
    _sha(row["authority_digest"], label="controller authority digest")
    return row


def torchrun_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the captured torchrun source and site directory binding."""

    row = dict(
        _closed(
            value,
            {"source", "subprocess_handler_source", "site_packages"},
            label="torchrun binding",
        )
    )
    row["source"] = _validate_captured_file(
        row["source"], label="torchrun source", verify_file=False
    )
    row["site_packages"] = _validate_captured_directory(
        row["site_packages"], label="torchrun site-packages",
        verify_directory=False,
    )
    row["subprocess_handler_source"] = _validate_captured_file(
        row["subprocess_handler_source"],
        label="torchrun subprocess handler source",
        verify_file=False,
    )
    if row["source"]["path"] != str(
        Path(row["site_packages"]["path"]) / "torch/distributed/run.py"
    ) or (
        row["source"]["sha256"]
        != verified_release.TORCHRUN_SOURCE_SHA256
        or row["source"]["size"] != verified_release.TORCHRUN_SOURCE_SIZE
    ):
        raise DecodedEvaluationBridgeError(
            "torchrun source path/bytes differ"
        )
    expected_handler = (
        Path(row["site_packages"]["path"])
        / verified_release.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
    )
    if (
        row["subprocess_handler_source"]["path"]
        != str(expected_handler)
        or row["subprocess_handler_source"]["sha256"]
        != verified_release.TORCHRUN_SUBPROCESS_HANDLER_SHA256
        or row["subprocess_handler_source"]["size"]
        != verified_release.TORCHRUN_SUBPROCESS_HANDLER_SIZE
    ):
        raise DecodedEvaluationBridgeError(
            "torchrun subprocess handler source differs"
        )
    return row


def verified_target_argv(
    bindings: Mapping[str, Any], *, target: str, arguments: Sequence[str],
    capture_receipt_path: str | Path,
) -> list[str]:
    """Build one held-FD/captured-bytes argv from a validated binding.

    Callers must first load ``bindings`` through :func:`load_physical_bindings`.
    The returned command never executes a release member by its pathname.
    """

    receipt_path = _absolute(
        str(capture_receipt_path), label="verified runtime capture receipt"
    )
    runtime = bindings["runtime"]
    try:
        return verified_release.verified_target_argv(
            executable_runtime_binding(runtime["root_python"]),
            executable_runtime_binding(runtime["python"]),
            directory_runtime_binding(runtime["site_packages"]),
            eval_release_runtime_binding(bindings["eval_release"]),
            executable_runtime_binding(runtime["deployment_controller"]),
            controller_authority_runtime_binding(
                runtime["controller_authority"]
            ),
            target,
            list(arguments),
            str(receipt_path),
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error


def frozen_exec_argv(
    bindings: Mapping[str, Any], *, frozen_arguments: Sequence[str]
) -> list[str]:
    """Build one held-FD frozen-interpreter command from physical bindings."""

    runtime = bindings["runtime"]
    try:
        return verified_release.frozen_exec_argv(
            executable_runtime_binding(runtime["root_python"]),
            executable_runtime_binding(runtime["python"]),
            directory_runtime_binding(runtime["site_packages"]),
            eval_release_runtime_binding(bindings["eval_release"]),
            executable_runtime_binding(runtime["deployment_controller"]),
            controller_authority_runtime_binding(
                runtime["controller_authority"]
            ),
            list(frozen_arguments),
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error


def captured_torchrun_argv(
    bindings: Mapping[str, Any], *, torchrun_arguments: Sequence[str],
    rank_target_argv: Sequence[str],
) -> list[str]:
    """Run captured ``torch.distributed.run`` under detached authority."""

    runtime = bindings["runtime"]
    try:
        return verified_release.captured_torchrun_argv(
            executable_runtime_binding(runtime["root_python"]),
            executable_runtime_binding(runtime["python"]),
            directory_runtime_binding(runtime["site_packages"]),
            torchrun_runtime_binding(runtime["torchrun"]),
            eval_release_runtime_binding(bindings["eval_release"]),
            executable_runtime_binding(runtime["deployment_controller"]),
            controller_authority_runtime_binding(
                runtime["controller_authority"]
            ),
            list(torchrun_arguments),
            list(rank_target_argv),
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error


def validate_verified_capture_receipt(
    bindings: Mapping[str, Any], *, receipt_path: str | Path, target: str,
    expected_arguments: Sequence[str], expected_capture_digest: str | None = None,
    verify_file: bool = True,
    inherited_fd_binding: Mapping[str, Any] | None = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one create-only runtime capture receipt against exact15."""

    if expected_capture_digest is not None:
        _sha(expected_capture_digest, label="verified release capture digest")
    receipt_path = _absolute(
        str(receipt_path), label="verified release capture receipt"
    )
    if inherited_fd_binding is not None and work_root_binding is not None:
        raise DecodedEvaluationBridgeError(
            "verified capture received mixed publication roots"
        )
    if (
        inherited_fd_binding is None
        and work_root_binding is None
        and verified_release.WORK_ROOT_BINDING_ENV in os.environ
    ):
        try:
            work_root_binding = (
                verified_release.load_inherited_work_root_environment(
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            )
        except verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
    if inherited_fd_binding is None and work_root_binding is None:
        raw, receipt_file = _stable_file(
            receipt_path, label="verified release capture receipt"
        )
    elif work_root_binding is not None:
        try:
            live_work_root = verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
            if (
                receipt_path.parent != Path(live_work_root["path"])
                or str(receipt_path)
                != live_work_root["capture_receipt_path"]
            ):
                raise DecodedEvaluationBridgeError(
                    "verified capture path differs from held work root"
                )
            descriptor = os.open(
                receipt_path.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=live_work_root["root_fd"],
            )
            try:
                os.set_inheritable(descriptor, False)
                identity = os.fstat(descriptor)
                raw = verified_release._read_fd(descriptor)
                middle = os.fstat(descriptor)
                replay_raw = verified_release._read_fd(descriptor)
                after = os.fstat(descriptor)
                named = os.stat(
                    receipt_path.name,
                    dir_fd=live_work_root["root_fd"],
                    follow_symlinks=False,
                )
            finally:
                os.close(descriptor)
            if (
                not stat.S_ISREG(identity.st_mode)
                or identity.st_nlink != 1
                or stat.S_IMODE(identity.st_mode) != 0o444
                or verified_release._identity(identity)
                != verified_release._identity(middle)
                or verified_release._identity(identity)
                != verified_release._identity(after)
                or verified_release._identity(identity)
                != verified_release._identity(named)
                or raw != replay_raw
            ):
                raise DecodedEvaluationBridgeError(
                    "verified release held work-root capture differs"
                )
        except (verified_release.DecodedEvalVerifiedReleaseError, OSError) as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        receipt_file = {
            "path": str(receipt_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": identity.st_size,
            "mode": stat.S_IMODE(identity.st_mode),
            "device": identity.st_dev, "inode": identity.st_ino,
            "uid": identity.st_uid, "gid": identity.st_gid,
            "nlink": identity.st_nlink, "rdev": identity.st_rdev,
            "blocks": getattr(identity, "st_blocks", 0),
            "mtime_ns": identity.st_mtime_ns,
            "ctime_ns": identity.st_ctime_ns,
        }
    else:
        try:
            raw, identity = model_authority.stable_inherited_task_file(
                receipt_path,
                inherited_fd_binding=inherited_fd_binding,
                label="verified release capture receipt",
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        receipt_file = {
            "path": str(receipt_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": identity["size"],
            "mode": stat.S_IMODE(identity["mode"]),
            **{
                field: identity[field]
                for field in (
                    "device", "inode", "uid", "gid", "nlink", "rdev",
                    "blocks", "mtime_ns", "ctime_ns",
                )
            },
        }
    if receipt_file["mode"] != 0o444 or receipt_file["nlink"] != 1:
        raise DecodedEvaluationBridgeError(
            "verified release capture receipt topology differs"
        )
    receipt = _json(raw, label="verified release capture receipt", canonical=True)
    try:
        receipt = dict(
            verified_release.validate_capture_receipt(
                receipt, verify_file=False
            )
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    release = bindings["eval_release"]
    runtime = bindings["runtime"]
    members = {
        item["relative_path"]: item for item in release["members"]
    }
    if target not in members:
        raise DecodedEvaluationBridgeError(
            "verified release capture target is outside exact15"
        )
    root_python = receipt["root_python"]
    frozen = receipt["frozen_python"]
    site_packages = receipt["site_packages"]
    release_artifacts = receipt["release_artifacts"]
    controller_authority = receipt["controller_authority"]
    expected_torchrun = (
        torchrun_runtime_binding(runtime["torchrun"])
        if target == "infer_lora.py"
        else None
    )
    if (
        (
            expected_capture_digest is not None
            and receipt["capture_digest"] != expected_capture_digest
        )
        or receipt["target"] != target
        or receipt["target_arguments_sha256"]
        != object_sha256(list(expected_arguments))
        or receipt["archive_sha256"] != release["archive_file"]["sha256"]
        or receipt["manifest_sha256"] != release["manifest_file"]["sha256"]
        or receipt["manifest_digest"] != release["manifest_digest"]
        or receipt["content_revision"] != release["content_revision"]
        or receipt["envelope_sha256"] != release["envelope_file"]["sha256"]
        or receipt["envelope_digest"] != release["envelope_digest"]
        or receipt["all_members_capture_digest"]
        != release["all_members_capture_digest"]
        or receipt["member_count"] != len(EVAL_RELEASE_MEMBERS)
        or receipt["target_sha256"] != members[target]["sha256"]
        or receipt["target_size"] != members[target]["size"]
        or receipt["target_mode"] != members[target]["mode"]
        or receipt["receipt_path"] != str(receipt_path)
        or any(
            root_python[field] != runtime["root_python"][field]
            for field in _CAPTURED_FILE_FIELDS
        )
        or any(
            frozen[field] != runtime["python"][field]
            for field in _CAPTURED_FILE_FIELDS
        )
        or any(
            site_packages[field] != runtime["site_packages"][field]
            for field in _CAPTURED_DIRECTORY_FIELDS
        )
        or any(
            release_artifacts[name][field]
            != release[f"{name}_file"][field]
            for name in ("archive", "manifest", "envelope")
            for field in _CAPTURED_FILE_FIELDS
        )
        or controller_authority
        != controller_authority_runtime_binding(runtime["controller_authority"])
        or receipt["captured_torchrun"] != expected_torchrun
        or (
            work_root_binding is not None
            and receipt["work_root"] != work_root_binding
        )
    ):
        raise DecodedEvaluationBridgeError(
            "verified release runtime capture differs from physical authority"
        )
    if verify_file:
        if inherited_fd_binding is None and work_root_binding is None:
            _validate_captured_file(
                receipt_file, label="verified release capture receipt",
                verify_file=True,
            )
        elif inherited_fd_binding is not None:
            try:
                replay_raw, replay_identity = (
                    model_authority.stable_inherited_task_file(
                        receipt_path,
                        inherited_fd_binding=inherited_fd_binding,
                        label="verified release capture receipt replay",
                        expected_sha256=receipt_file["sha256"],
                    )
                )
            except model_authority.ModelConsumptionAuthorityError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
            if (
                replay_raw != raw
                or replay_identity["inode"] != receipt_file["inode"]
                or replay_identity["device"] != receipt_file["device"]
            ):
                raise DecodedEvaluationBridgeError(
                    "verified release task-root capture replay differs"
                )
        else:
            assert work_root_binding is not None
            try:
                replay_raw, replay_info = (
                    verified_release._stable_work_root_file_pair(
                        work_root_binding["root_fd"],
                        {
                            "path": str(receipt_path),
                            "sha256": receipt_file["sha256"],
                        },
                        label="verified release capture receipt replay",
                    )
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
            if (
                replay_raw != raw
                or replay_info.st_ino != receipt_file["inode"]
                or replay_info.st_dev != receipt_file["device"]
            ):
                raise DecodedEvaluationBridgeError(
                    "verified release work-root capture replay differs"
                )
    if work_root_binding is not None:
        try:
            final_work_root = verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if final_work_root != work_root_binding:
            raise DecodedEvaluationBridgeError(
                "verified release work-root authority changed during replay"
            )
    return {
        "receipt_path": receipt_file["path"],
        "receipt_sha256": receipt_file["sha256"],
        "capture_digest": receipt["capture_digest"],
        "target": receipt["target"],
        "target_arguments_sha256": receipt["target_arguments_sha256"],
    }


def validate_running_verified_capture(
    bindings: Mapping[str, Any], *, target: str,
    expected_arguments: Sequence[str], verify_file: bool = True,
    inherited_fd_binding: Mapping[str, Any] | None = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate this process' verified-runtime capture against exact15.

    The bootstrap publishes both the digest and create-only receipt path in the
    environment before it executes a target from captured bytes.
    """

    digest = os.environ.get(verified_release.CAPTURE_DIGEST_ENV)
    receipt_text = os.environ.get(verified_release.CAPTURE_RECEIPT_ENV)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise DecodedEvaluationBridgeError(
            "verified release capture digest is absent or invalid"
        )
    if not isinstance(receipt_text, str) or not receipt_text:
        raise DecodedEvaluationBridgeError(
            "verified release capture receipt path is absent"
        )
    return validate_verified_capture_receipt(
        bindings, receipt_path=receipt_text, target=target,
        expected_arguments=expected_arguments,
        expected_capture_digest=digest, verify_file=verify_file,
        inherited_fd_binding=inherited_fd_binding,
        work_root_binding=work_root_binding,
    )


def require_running_eval_release_member(
    eval_release: Mapping[str, Any], *, relative_path: str,
    running_path: str | Path,
) -> dict[str, Any]:
    """Bind a loaded production module to one exact sealed release member."""

    if relative_path not in EVAL_RELEASE_MEMBERS:
        raise DecodedEvaluationBridgeError("unknown eval release member")
    matches = [
        item for item in eval_release.get("members", [])
        if item.get("relative_path") == relative_path
    ]
    if len(matches) != 1:
        raise DecodedEvaluationBridgeError("eval release member binding differs")
    member = matches[0]
    resolved = Path(running_path).resolve(strict=True)
    if str(resolved) != member["path"]:
        raise DecodedEvaluationBridgeError(
            f"running {relative_path} is outside the exact eval release"
        )
    _, observed = _stable_file(
        resolved, label=f"running eval release {relative_path}",
        expected_sha256=member["sha256"],
    )
    if any(observed[field] != member[field] for field in _CAPTURED_FILE_FIELDS):
        raise DecodedEvaluationBridgeError(
            f"running {relative_path} identity differs from exact eval release"
        )
    return dict(member)


def _validate_source_preprocessing_authority(
    value: Any, *, source_manifest_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema_version", "serialization", "source_manifest_sha256",
        "source_manifest_digest", "source_order", "sources",
        "source_video_bytes_consumed_directly",
        "precomputed_transformed_source_artifact_used",
        "runtime_decode_bound_by_inference_release",
        "target_video_available_to_inference",
        "training_loss_read_or_used_for_selection", "remote_launch_performed",
        "scientific_promotion_authorized", "authority_digest",
    }
    row = dict(_closed(value, fields, label="source preprocessing authority"))
    source_fields = {
        "iid", "source_video_path", "source_video_sha256",
        "source_receipt_path", "source_receipt_sha256", "instruction",
        "instruction_sha256", "action_review_contract", "seed",
    }
    if (
        row["schema_version"] != SOURCE_PREPROCESSING_AUTHORITY_SCHEMA
        or row["serialization"] != SOURCE_PREPROCESSING_SERIALIZATION
        or row["source_manifest_sha256"] != source_manifest_sha256
        or _sha(
            row["source_manifest_digest"],
            label="source preprocessing manifest digest",
        )
        != "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
        or row["source_order"] != list(plan.FITTED_IIDS)
        or row["source_video_bytes_consumed_directly"] is not True
        or row["precomputed_transformed_source_artifact_used"] is not False
        or row["runtime_decode_bound_by_inference_release"] is not True
        or row["target_video_available_to_inference"] is not False
        or row["training_loss_read_or_used_for_selection"] is not False
        or row["remote_launch_performed"] is not False
        or row["scientific_promotion_authorized"] is not False
        or row["authority_digest"] != SOURCE_PREPROCESSING_AUTHORITY_DIGEST
        or _verify_object_digest(
            row, field="authority_digest", label="source preprocessing authority"
        )
        != SOURCE_PREPROCESSING_AUTHORITY_DIGEST
    ):
        raise DecodedEvaluationBridgeError(
            "source preprocessing authority policy differs"
        )
    if type(row["sources"]) is not list or len(row["sources"]) != len(
        plan.FITTED_IIDS
    ):
        raise DecodedEvaluationBridgeError(
            "source preprocessing source count differs"
        )
    sources: list[dict[str, Any]] = []
    for expected_iid, source_value in zip(plan.FITTED_IIDS, row["sources"]):
        source = dict(
            _closed(
                source_value,
                source_fields,
                label="source preprocessing source",
            )
        )
        if (
            source["iid"] != expected_iid
            or str(_absolute(source["source_video_path"], label="source video path"))
            != source["source_video_path"]
            or str(
                _absolute(source["source_receipt_path"], label="source receipt path")
            )
            != source["source_receipt_path"]
            or _sha(source["source_video_sha256"], label="source video SHA")
            != source["source_video_sha256"]
            or _sha(source["source_receipt_sha256"], label="source receipt SHA")
            != source["source_receipt_sha256"]
            or not isinstance(source["instruction"], str)
            or not source["instruction"].strip()
            or source["instruction"] != source["instruction"].strip()
            or "\x00" in source["instruction"]
            or plan.text_sha256(source["instruction"])
            != _sha(source["instruction_sha256"], label="instruction SHA")
            or type(source["seed"]) is not int
            or not 0 <= source["seed"] < 2**63
        ):
            raise DecodedEvaluationBridgeError(
                "source preprocessing source authority differs"
            )
        try:
            source["action_review_contract"] = plan.validate_action_review_contract(
                source["action_review_contract"]
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        sources.append(source)
    row["sources"] = sources
    return row


def validate_source_runtime_spec(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "pins",
        "pin_files",
        "sources",
        "runtime",
        "spec_digest",
    }
    row = dict(_closed(value, fields, label="source/runtime spec"))
    if row["schema_version"] != SOURCE_RUNTIME_SCHEMA:
        raise DecodedEvaluationBridgeError("source/runtime schema differs")
    pins = dict(_closed(row["pins"], plan.PIN_FIELDS, label="source/runtime pins"))
    for key in sorted(plan.PIN_FIELDS - {"calibration_digest"}):
        pins[key] = _sha(pins[key], label=key)
    calibration_digest = pins["calibration_digest"]
    if calibration_digest is not None:
        pins["calibration_digest"] = _sha(calibration_digest, label="calibration digest")

    pin_file_fields = {
        "source_manifest", "adapter_release_manifest", "model_release_manifest",
        "inference_release_manifest", "inference_config",
        "source_preprocessing", "calibration",
    }
    pin_files_value = dict(
        _closed(row["pin_files"], pin_file_fields, label="pin files")
    )
    pin_to_file = {
        "source_manifest_sha256": "source_manifest",
        "adapter_release_manifest_sha256": "adapter_release_manifest",
        "model_release_manifest_sha256": "model_release_manifest",
        "inference_release_manifest_sha256": "inference_release_manifest",
        "inference_config_sha256": "inference_config",
        "source_preprocessing_sha256": "source_preprocessing",
    }
    pin_files: dict[str, Any] = {}
    for pin_key, file_key in pin_to_file.items():
        binding = _file_binding(pin_files_value[file_key], label=file_key)
        if binding["sha256"] != pins[pin_key]:
            raise DecodedEvaluationBridgeError(f"{file_key} physical pin differs")
        pin_files[file_key] = binding
    if (
        pins["source_preprocessing_sha256"]
        != SOURCE_PREPROCESSING_AUTHORITY_SHA256
        or pin_files["source_preprocessing"]["sha256"]
        != SOURCE_PREPROCESSING_AUTHORITY_SHA256
    ):
        raise DecodedEvaluationBridgeError(
            "source preprocessing differs from the exact r7 authority"
        )
    preprocessing_raw, preprocessing_evidence = _stable_file(
        pin_files["source_preprocessing"]["path"],
        label="source preprocessing authority replay",
        expected_sha256=SOURCE_PREPROCESSING_AUTHORITY_SHA256,
    )
    if preprocessing_evidence != pin_files["source_preprocessing"]:
        raise DecodedEvaluationBridgeError(
            "source preprocessing physical identity replay differs"
        )
    source_preprocessing = _validate_source_preprocessing_authority(
        _json(
            preprocessing_raw,
            label="source preprocessing authority",
            canonical=True,
        ),
        source_manifest_sha256=pins["source_manifest_sha256"],
    )
    if pins["calibration_digest"] is None:
        if pin_files_value["calibration"] is not None:
            raise DecodedEvaluationBridgeError(
                "uncalibrated evaluation unexpectedly binds a calibration file"
            )
        pin_files["calibration"] = None
    else:
        calibration_file = _file_binding(
            pin_files_value["calibration"], label="calibration"
        )
        calibration_raw, _ = _stable_file(
            calibration_file["path"], label="calibration",
            expected_sha256=calibration_file["sha256"],
        )
        calibration_value = _json(
            calibration_raw, label="calibration", canonical=True
        )
        try:
            calibration_authority = plan.gate.validate_calibration(calibration_value)
        except plan.gate.ActionPreservationGateError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if calibration_authority["calibration_digest"] != pins["calibration_digest"]:
            raise DecodedEvaluationBridgeError("calibration physical pin differs")
        pin_files["calibration"] = calibration_file

    source_fields = {
        "iid",
        "source_video_path",
        "source_video_sha256",
        "source_receipt_path",
        "source_receipt_sha256",
        "instruction",
        "instruction_sha256",
        "action_review_contract",
        "seed",
    }
    if not isinstance(row["sources"], list) or len(row["sources"]) != len(plan.FITTED_IIDS):
        raise DecodedEvaluationBridgeError("source/runtime source count differs")
    sources: list[dict[str, Any]] = []
    for expected_iid, item_value in zip(plan.FITTED_IIDS, row["sources"]):
        item = dict(_closed(item_value, source_fields, label="source/runtime source"))
        if item["iid"] != expected_iid:
            raise DecodedEvaluationBridgeError("source/runtime IID order differs")
        _stable_file(
            item["source_video_path"],
            label=f"source video {expected_iid}",
            expected_sha256=_sha(item["source_video_sha256"], label="source video SHA"),
        )
        _stable_file(
            item["source_receipt_path"],
            label=f"source receipt {expected_iid}",
            expected_sha256=_sha(item["source_receipt_sha256"], label="source receipt SHA"),
        )
        if (
            not isinstance(item["instruction"], str)
            or not item["instruction"].strip()
            or item["instruction"] != item["instruction"].strip()
            or "\x00" in item["instruction"]
            or plan.text_sha256(item["instruction"])
            != _sha(item["instruction_sha256"], label="instruction SHA")
        ):
            raise DecodedEvaluationBridgeError("source instruction differs")
        try:
            item["action_review_contract"] = plan.validate_action_review_contract(
                item["action_review_contract"]
            )
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if type(item["seed"]) is not int or not 0 <= item["seed"] < 2**63:
            raise DecodedEvaluationBridgeError("source seed differs")
        sources.append(item)
    if sources != source_preprocessing["sources"]:
        raise DecodedEvaluationBridgeError(
            "source/runtime sources differ from source preprocessing authority"
        )

    runtime_fields = {
        "root_python",
        "python",
        "site_packages",
        "torchrun",
        "deployment_controller",
        "controller_authority",
        "infer_lora",
        "decoder_adapter",
        "ffprobe",
        "eval_release_root",
        "eval_release_archive",
        "eval_release_envelope",
        "eval_release_manifest_digest",
        "eval_release_content_revision",
        "eval_release_envelope_digest",
        "bernini_root",
        "veomni_root",
        "model_checkpoint_root",
        "expected_bernini_commit",
        "expected_veomni_commit",
        "expected_checkpoint_tree_sha256",
        "method_source_revision",
        "method_source_archive_sha256",
        "num_inference_steps",
    }
    runtime = dict(_closed(row["runtime"], runtime_fields, label="inference runtime"))
    runtime["root_python"] = _file_binding(
        runtime["root_python"], label="root bootstrap Python"
    )
    runtime["python"] = _file_binding(runtime["python"], label="runtime Python")
    site_packages_path = _plain_directory(
        runtime["site_packages"], label="runtime site-packages"
    )
    try:
        runtime["site_packages"] = verified_release.capture_directory_binding(
            site_packages_path, label="runtime site-packages"
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    requested_torchrun = _file_binding(
        runtime["torchrun"], label="captured torchrun source"
    )
    try:
        runtime["torchrun"] = verified_release.capture_torchrun_binding(
            site_packages_path, label="captured torchrun"
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    if requested_torchrun != runtime["torchrun"]["source"]:
        raise DecodedEvaluationBridgeError(
            "captured torchrun differs from its detached source pin"
        )
    runtime["deployment_controller"] = _file_binding(
        runtime["deployment_controller"], label="detached eval controller"
    )
    controller_authority_value = dict(
        _closed(
            runtime["controller_authority"],
            {"receipt", "authority_digest"},
            label="controller authority",
        )
    )
    controller_authority_value["receipt"] = _file_binding(
        controller_authority_value["receipt"],
        label="controller authority receipt",
    )
    _sha(
        controller_authority_value["authority_digest"],
        label="controller authority digest",
    )
    runtime["controller_authority"] = controller_authority_value
    runtime["infer_lora"] = _file_binding(runtime["infer_lora"], label="infer_lora")
    runtime["decoder_adapter"] = _file_binding(
        runtime["decoder_adapter"], label="decoder adapter"
    )
    runtime["ffprobe"] = _file_binding(runtime["ffprobe"], label="ffprobe")
    runtime["eval_release_archive"] = _file_binding(
        runtime["eval_release_archive"], label="eval release archive"
    )
    runtime["eval_release_envelope"] = _file_binding(
        runtime["eval_release_envelope"], label="eval release envelope"
    )
    runtime["eval_release_root"] = str(
        _plain_directory(runtime["eval_release_root"], label="eval release root")
    )
    for key in ("root_python", "python", "decoder_adapter", "ffprobe"):
        if not os.access(runtime[key]["path"], os.X_OK):
            raise DecodedEvaluationBridgeError(f"runtime {key} is not executable")
    if (
        runtime["root_python"]["path"] != str(ROOT_PYTHON_PATH)
        or runtime["root_python"]["uid"] != ROOT_PYTHON_UID
        or runtime["root_python"]["gid"] != ROOT_PYTHON_GID
        or runtime["root_python"]["mode"] != ROOT_PYTHON_MODE
        or runtime["root_python"]["nlink"] != 1
    ):
        raise DecodedEvaluationBridgeError(
            "root bootstrap Python is not the fixed root-owned executable"
        )
    if (
        runtime["deployment_controller"]["mode"] != 0o444
        or runtime["controller_authority"]["receipt"]["mode"] != 0o444
    ):
        raise DecodedEvaluationBridgeError(
            "detached controller authority is not sealed mode 0444"
        )
    if (
        runtime["ffprobe"]["path"] != str(FFPROBE_PATH)
        or runtime["ffprobe"]["uid"] != FFPROBE_UID
        or runtime["ffprobe"]["gid"] != FFPROBE_GID
        or runtime["ffprobe"]["mode"] != FFPROBE_MODE
        or runtime["ffprobe"]["nlink"] != 1
    ):
        raise DecodedEvaluationBridgeError(
            "ffprobe is not the fixed root-owned executable"
        )
    for key in ("bernini_root", "veomni_root", "model_checkpoint_root"):
        runtime[key] = str(_plain_directory(runtime[key], label=key))
    _sha1(runtime["expected_bernini_commit"], label="Bernini commit")
    _sha1(runtime["expected_veomni_commit"], label="VeOmni commit")
    _sha(runtime["expected_checkpoint_tree_sha256"], label="checkpoint tree")
    _sha1(runtime["method_source_revision"], label="method source revision")
    _sha(runtime["method_source_archive_sha256"], label="method source archive")
    _sha(
        runtime["eval_release_manifest_digest"],
        label="eval release manifest digest",
    )
    _sha1(
        runtime["eval_release_content_revision"],
        label="eval release content revision",
    )
    _sha(
        runtime["eval_release_envelope_digest"],
        label="eval release envelope digest",
    )
    if runtime["num_inference_steps"] != 40:
        raise DecodedEvaluationBridgeError("inference step count must be exactly 40")
    if pins["inference_source_sha256"] != runtime["infer_lora"]["sha256"]:
        raise DecodedEvaluationBridgeError("inference source pin differs from physical infer_lora")
    eval_release = load_eval_release_manifest(
        pin_files["inference_release_manifest"]["path"],
        expected_sha256=pin_files["inference_release_manifest"]["sha256"],
        release_root=runtime["eval_release_root"],
        archive_path=runtime["eval_release_archive"]["path"],
        expected_archive_sha256=runtime["eval_release_archive"]["sha256"],
        envelope_path=runtime["eval_release_envelope"]["path"],
        expected_envelope_sha256=runtime["eval_release_envelope"]["sha256"],
        expected_content_revision=runtime["eval_release_content_revision"],
        expected_manifest_digest=runtime["eval_release_manifest_digest"],
        expected_envelope_digest=runtime["eval_release_envelope_digest"],
        verify_files=True,
    )
    try:
        verified_release.validate_controller_authority_binding(
            runtime["controller_authority"],
            controller_binding=runtime["deployment_controller"],
            root_python_binding=runtime["root_python"],
            frozen_python_binding=runtime["python"],
            site_packages_binding=runtime["site_packages"],
            torchrun_binding=runtime["torchrun"],
            require_torchrun_continuity=True,
            release_binding=eval_release_runtime_binding(eval_release),
            verify_file=True,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    release_members = {
        item["relative_path"]: item for item in eval_release["members"]
    }
    for runtime_key, relative_path in (
        ("infer_lora", "infer_lora.py"),
        (
            "decoder_adapter",
            "action_preservation_decoded_eval_decoder_adapter_v1.py",
        ),
    ):
        if any(
            runtime[runtime_key][field] != release_members[relative_path][field]
            for field in _CAPTURED_FILE_FIELDS
        ):
            raise DecodedEvaluationBridgeError(
                f"runtime {runtime_key} differs from exact eval release"
            )
    verified_member = release_members[
        "action_preservation_decoded_eval_verified_release_v1.py"
    ]
    runtime["verified_release"] = {
        key: verified_member[key] for key in _CAPTURED_FILE_FIELDS
    }
    _verify_object_digest(row, field="spec_digest", label="source/runtime spec")
    row.update(
        pins=pins, pin_files=pin_files, sources=sources, runtime=runtime,
        eval_release=eval_release,
        source_preprocessing_authority=source_preprocessing,
    )
    return row


def _validated_audit_rows(audit: Any) -> dict[tuple[str, int], Mapping[str, Any]]:
    audit_fields = {
        "training_audit_go", "arm_count", "checkpoint_count", "checkpoint_steps",
        "route_scopes", "initialization_digest_by_scope",
        "checkpoint_zero_adapter_sha256_by_scope",
        "adapter_config_sha256_by_scope", "receipt_rows",
        "decoded_evaluation_complete", "scientific_promotion_authorized",
    }
    authority = _closed(audit, audit_fields, label="training audit")
    if (
        authority["training_audit_go"] is not True
        or authority["arm_count"] != len(plan.ARMS)
        or authority["checkpoint_count"] != 32
        or authority["checkpoint_steps"] != list(plan.CHECKPOINT_STEPS)
        or authority["decoded_evaluation_complete"] is not False
        or authority["scientific_promotion_authorized"] is not False
    ):
        raise DecodedEvaluationBridgeError("training audit authority differs")
    rows = authority["receipt_rows"]
    row_fields = {
        "arm", "step", "receipt_sha256", "adapter_sha256",
        "adapter_config_sha256", "optimizer_sha256", "loss",
        "preclip_gradient_norm",
    }
    if not isinstance(rows, list) or len(rows) != 32:
        raise DecodedEvaluationBridgeError("training audit row count differs")
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row_value in rows:
        row = _closed(row_value, row_fields, label="training audit row")
        if row["arm"] not in plan.ARMS or row["step"] not in plan.CHECKPOINT_STEPS:
            raise DecodedEvaluationBridgeError("training audit arm/step differs")
        for field in (
            "receipt_sha256", "adapter_sha256", "adapter_config_sha256",
            "optimizer_sha256",
        ):
            _sha(row[field], label=f"training audit {field}")
        key = (row["arm"], row["step"])
        if key in by_key:
            raise DecodedEvaluationBridgeError("duplicate training audit row")
        by_key[key] = row
    expected = {(arm, step) for arm in plan.ARMS for step in plan.CHECKPOINT_STEPS}
    if set(by_key) != expected:
        raise DecodedEvaluationBridgeError("training audit checkpoint closure differs")
    return by_key


def _validated_checkpoint_receipt_digest(
    receipt: Any, *, arm: str, step: int
) -> str:
    if not isinstance(receipt, Mapping):
        raise DecodedEvaluationBridgeError("checkpoint receipt root differs")
    unsigned = dict(receipt)
    receipt_digest = unsigned.pop("receipt_digest", None)
    contract = receipt.get("training_contract")
    if (
        _sha(receipt_digest, label="checkpoint receipt digest")
        != object_sha256(unsigned)
        or not isinstance(contract, Mapping)
        or contract.get("arm") != arm
        or receipt.get("global_step") != step
    ):
        raise DecodedEvaluationBridgeError("checkpoint receipt arm/step differs")
    return receipt_digest


def _training_authority(
    *, experiment_root: Path, expected_completion_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    completion_path = experiment_root / "TRAINING_COMPLETE.json"
    completion_raw, completion_file = _stable_file(
        completion_path,
        label="training completion marker",
        expected_sha256=expected_completion_sha256,
    )
    completion = _json(completion_raw, label="training completion marker", canonical=True)
    completion_fields = {
        "schema_version", "seed", "cache_sha256", "source_archive_sha256",
        "source_revision", "source_data_manifest_sha256",
        "source_data_manifest_digest", "release_manifest_sha256",
        "controller_sha256", "deployment_envelope_sha256", "cache_audit_sha256",
        "training_audit_sha256", "cache_receipt_sha256", "retained_tree_digest",
        "retained_tree_file_count", "retained_tree_stable_double_read_before_commit",
        "retained_tree_held_fd_identity_replay", "optimizer_updates_per_arm",
        "arm_count", "decoded_evaluation_complete", "scientific_promotion_authorized",
        "parent_allocations_cancelled", "automatic_retry", "completion_digest",
    }
    _closed(completion, completion_fields, label="training completion marker")
    if (
        completion["schema_version"] != "bernini-action-preservation-v2-training-complete-v3"
        or completion["seed"] != 20260818
        or completion["optimizer_updates_per_arm"] != 20
        or completion["arm_count"] != len(plan.ARMS)
        or completion["decoded_evaluation_complete"] is not False
        or completion["scientific_promotion_authorized"] is not False
        or completion["parent_allocations_cancelled"] is not False
        or completion["automatic_retry"] is not False
        or completion["retained_tree_stable_double_read_before_commit"] is not True
        or completion["retained_tree_held_fd_identity_replay"] is not True
    ):
        raise DecodedEvaluationBridgeError("training completion authority differs")
    _verify_object_digest(completion, field="completion_digest", label="training completion")

    audit_path = experiment_root / "logs" / "training-audit.json"
    audit_raw, audit_file = _stable_file(
        audit_path,
        label="training audit",
        expected_sha256=_sha(completion["training_audit_sha256"], label="training audit SHA"),
    )
    audit = _producer_training_audit_json(
        audit_raw, label="training audit"
    )
    by_key = _validated_audit_rows(audit)

    checkpoints: list[dict[str, Any]] = []
    for arm in plan.ARMS:
        for step in plan.CHECKPOINT_STEPS:
            row = by_key[(arm, step)]
            checkpoint_root = _plain_directory(
                experiment_root / "runs" / arm / f"checkpoint-{step:08d}",
                label=f"checkpoint root {arm}@{step}",
            )
            receipt_raw, receipt_file = _stable_file(
                checkpoint_root / "receipt.json",
                label=f"checkpoint receipt {arm}@{step}",
                expected_sha256=_sha(row["receipt_sha256"], label="checkpoint receipt SHA"),
            )
            receipt = _json(receipt_raw, label=f"checkpoint receipt {arm}@{step}", canonical=True)
            receipt_digest = _validated_checkpoint_receipt_digest(
                receipt, arm=arm, step=step
            )
            adapter_raw, adapter_file = _stable_file(
                checkpoint_root / "adapter" / "adapter_model.safetensors",
                label=f"adapter model {arm}@{step}",
                expected_sha256=_sha(row["adapter_sha256"], label="adapter SHA"),
            )
            if not adapter_raw:
                raise DecodedEvaluationBridgeError("adapter model is empty")
            config_raw, config_file = _stable_file(
                checkpoint_root / "adapter" / "adapter_config.json",
                label=f"adapter config {arm}@{step}",
                expected_sha256=_sha(row["adapter_config_sha256"], label="adapter config SHA"),
            )
            _json(config_raw, label=f"adapter config {arm}@{step}")
            optimizer_raw, optimizer_file = _stable_file(
                checkpoint_root / "optimizer.pt",
                label=f"optimizer {arm}@{step}",
                expected_sha256=_sha(
                    row["optimizer_sha256"], label="optimizer SHA"
                ),
            )
            if not optimizer_raw:
                raise DecodedEvaluationBridgeError("optimizer state is empty")
            checkpoints.append(
                {
                    "arm": arm,
                    "checkpoint_step": step,
                    "checkpoint_root": str(checkpoint_root),
                    "checkpoint_receipt": receipt_file,
                    "checkpoint_receipt_digest": _sha(
                        receipt_digest, label="checkpoint receipt digest"
                    ),
                    "adapter_model": adapter_file,
                    "adapter_config": config_file,
                    "optimizer": optimizer_file,
                }
            )
    return completion, audit, checkpoints, completion_file, audit_file


def build_bridge(
    *,
    experiment_root: str | Path,
    completion_sha256: str,
    source_runtime_spec: Mapping[str, Any],
    evaluation_id: str,
    evaluation_root: str | Path,
    bridge_root: str | Path,
    deployment_authority: Mapping[str, Any] | None = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = _plain_directory(experiment_root, label="training experiment root")
    if deployment_authority is None:
        if work_root_binding is None:
            try:
                work_root_binding = (
                    verified_release.load_inherited_work_root_environment(
                        verify_open_fds=True,
                        expected_inheritable=False,
                        verify_entries=False,
                        allow_root_metadata_change=True,
                    )
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(
                    "bridge deployment authority is absent or invalid"
                ) from error
        deployment_authority = _deployment_authority_from_work_root_binding(
            work_root_binding, verify_open_fds=True
        )
    (
        deployment_authority_row,
        authorized_source_runtime_spec,
        _authorized_source_runtime_spec_file,
    ) = (
        _authorized_source_runtime_spec(
            deployment_authority,
            work_root_binding=work_root_binding,
        )
    )
    if source_runtime_spec != authorized_source_runtime_spec:
        raise DecodedEvaluationBridgeError(
            "source/runtime spec differs from held source spec authority"
        )
    source_runtime = validate_source_runtime_spec(source_runtime_spec)
    completion, audit, checkpoints, completion_file, audit_file = _training_authority(
        experiment_root=experiment,
        expected_completion_sha256=_sha(completion_sha256, label="completion file SHA"),
    )
    if source_runtime["pins"]["source_manifest_sha256"] != completion["source_data_manifest_sha256"]:
        raise DecodedEvaluationBridgeError("source manifest pin differs from completion")
    if source_runtime["pins"]["adapter_release_manifest_sha256"] != completion["release_manifest_sha256"]:
        raise DecodedEvaluationBridgeError("adapter release pin differs from completion")
    runtime = source_runtime["runtime"]
    if runtime["method_source_revision"] != completion["source_revision"]:
        raise DecodedEvaluationBridgeError("runtime source revision differs from completion")
    if runtime["method_source_archive_sha256"] != completion["source_archive_sha256"]:
        raise DecodedEvaluationBridgeError("runtime source archive differs from completion")

    logical_sources = [
        {
            "iid": item["iid"],
            "source_video_sha256": item["source_video_sha256"],
            "source_receipt_sha256": item["source_receipt_sha256"],
            "instruction": item["instruction"],
            "instruction_sha256": item["instruction_sha256"],
            "action_review_contract": item["action_review_contract"],
            "seed": item["seed"],
        }
        for item in source_runtime["sources"]
    ]
    logical_checkpoints = [
        {
            "arm": item["arm"],
            "checkpoint_step": item["checkpoint_step"],
            "checkpoint_receipt_sha256": item["checkpoint_receipt"]["sha256"],
            "adapter_sha256": item["adapter_model"]["sha256"],
        }
        for item in checkpoints
    ]
    input_spec = plan.build_input_spec(
        evaluation_id=evaluation_id,
        evaluation_root=str(_absolute(str(evaluation_root), label="evaluation root")),
        pins=source_runtime["pins"],
        sources=logical_sources,
        checkpoints=logical_checkpoints,
    )
    bundle = plan.build_bundle(input_spec)
    training_audit_digest = object_sha256(audit)
    bindings: dict[str, Any] = {
        "schema_version": PHYSICAL_BINDINGS_SCHEMA,
        "evaluation_id": evaluation_id,
        "evaluation_root": input_spec["evaluation_root"],
        "input_digest": input_spec["input_digest"],
        "manifest_digest": bundle["manifest"]["manifest_digest"],
        "training_experiment_root": str(experiment),
        "training_complete": {
            **completion_file,
            "completion_digest": completion["completion_digest"],
            "retained_tree_digest": completion["retained_tree_digest"],
        },
        "training_audit": audit_file,
        "training_audit_digest": training_audit_digest,
        "training_audit_serialization": TRAINING_AUDIT_SERIALIZATION,
        "source_preprocessing_authority_sha256": (
            SOURCE_PREPROCESSING_AUTHORITY_SHA256
        ),
        "source_preprocessing_authority_digest": (
            source_runtime["source_preprocessing_authority"]["authority_digest"]
        ),
        "source_preprocessing_source_order": list(plan.FITTED_IIDS),
        "pin_files": source_runtime["pin_files"],
        "eval_release": source_runtime["eval_release"],
        "sources": [],
        "checkpoints": checkpoints,
        "runtime": runtime,
        "deployment_authority": deployment_authority_row,
        "calibration_digest": source_runtime["pins"]["calibration_digest"],
        "evaluation_publication": None,
        "training_loss_read_or_copied": False,
        "decoded_evaluation_complete": False,
        "scientific_promotion_authorized": False,
    }
    for item in source_runtime["sources"]:
        _, video = _stable_file(
            item["source_video_path"], label=f"source video {item['iid']}",
            expected_sha256=item["source_video_sha256"],
        )
        _, receipt = _stable_file(
            item["source_receipt_path"], label=f"source receipt {item['iid']}",
            expected_sha256=item["source_receipt_sha256"],
        )
        bindings["sources"].append(
            {
                "iid": item["iid"],
                "source_video": video,
                "source_receipt": receipt,
                "instruction_sha256": item["instruction_sha256"],
                "action_review_contract_digest": item[
                    "action_review_contract"
                ]["contract_digest"],
                "seed": item["seed"],
            }
        )
    bindings["physical_bindings_digest"] = object_sha256(bindings)

    receipt: dict[str, Any] = {
        "schema_version": BRIDGE_RECEIPT_SCHEMA,
        "evaluation_id": evaluation_id,
        "evaluation_root": input_spec["evaluation_root"],
        "bridge_root": str(_absolute(str(bridge_root), label="bridge root")),
        "training_complete_file_sha256": completion_file["sha256"],
        "training_completion_digest": completion["completion_digest"],
        "training_audit_file_sha256": audit_file["sha256"],
        "training_audit_digest": training_audit_digest,
        "training_audit_serialization": TRAINING_AUDIT_SERIALIZATION,
        "source_preprocessing_authority_sha256": (
            SOURCE_PREPROCESSING_AUTHORITY_SHA256
        ),
        "source_preprocessing_authority_digest": (
            source_runtime["source_preprocessing_authority"]["authority_digest"]
        ),
        "source_preprocessing_source_order": list(plan.FITTED_IIDS),
        "physical_bindings_digest": bindings["physical_bindings_digest"],
        "physical_bindings_file_sha256": None,
        "input_digest": input_spec["input_digest"],
        "manifest_digest": bundle["manifest"]["manifest_digest"],
        "evaluation_publication": None,
        "checkpoint_count": 32,
        "source_count": 4,
        "planned_decode_count": plan.TOTAL_DECODE_COUNT,
        "training_loss_read_or_copied": False,
        "remote_launch_performed": False,
        "automatic_retry": False,
        "scientific_promotion_authorized": False,
        "create_only_publication": True,
    }
    receipt["bridge_receipt_digest"] = object_sha256(receipt)
    return bundle, bindings, validate_bridge_receipt(
        receipt,
        bundle=bundle,
        bindings=bindings,
        materialized_required=False,
    )


_CAPTURED_FILE_FIELDS = frozenset(
    {
        "path", "sha256", "size", "mode", "device", "inode", "uid",
        "gid", "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
    }
)
_CAPTURED_DIRECTORY_FIELDS = frozenset(
    _CAPTURED_FILE_FIELDS - {"sha256"}
)
_COMMON_ROOT_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version", "path", "identity", "parent_identity", "entries",
        "retained_parent_fd", "retained_root_fd",
    }
)


def _validate_captured_directory(
    value: Any, *, label: str, verify_directory: bool,
) -> dict[str, Any]:
    row = dict(_closed(value, _CAPTURED_DIRECTORY_FIELDS, label=label))
    path = _absolute(row["path"], label=f"{label} path")
    for key in _CAPTURED_DIRECTORY_FIELDS - {"path"}:
        if type(row[key]) is not int or row[key] < 0:
            raise DecodedEvaluationBridgeError(f"{label} {key} differs")
    if row["mode"] & ~0o7777:
        raise DecodedEvaluationBridgeError(f"{label} mode differs")
    if verify_directory:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            middle = os.fstat(descriptor)
            after = os.fstat(descriptor)
            named = path.lstat()
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _file_identity(before) != _file_identity(middle)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(named)
            or _captured_directory(path, before) != row
        ):
            raise DecodedEvaluationBridgeError(
                f"{label} full physical identity differs"
            )
    return row


def _validate_captured_file(
    value: Any, *, label: str, verify_file: bool
) -> dict[str, Any]:
    row = dict(_closed(value, _CAPTURED_FILE_FIELDS, label=label))
    _absolute(row["path"], label=f"{label} path")
    _sha(row["sha256"], label=f"{label} SHA")
    for key in (
        "size", "mode", "device", "inode", "uid", "gid", "nlink",
        "rdev", "blocks", "mtime_ns", "ctime_ns",
    ):
        if type(row[key]) is not int or row[key] < 0:
            raise DecodedEvaluationBridgeError(f"{label} {key} differs")
    if row["nlink"] != 1:
        raise DecodedEvaluationBridgeError(f"{label} hard-link closure differs")
    if verify_file:
        _, observed = _stable_file(
            row["path"], label=label, expected_sha256=row["sha256"]
        )
        if observed != row:
            raise DecodedEvaluationBridgeError(f"{label} physical identity differs")
    return row


def _validate_evaluation_publication(
    value: Any, *, evaluation_root: Path, verify_files: bool,
    required: bool, expected_evaluation_id: str,
    expected_input_digest: str, expected_manifest_digest: str,
    validation_barrier: Any = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise DecodedEvaluationBridgeError(
                "materialized evaluation publication is required"
            )
        return None
    fields = {
        "schema_version", "evaluation_root", "root_authority",
        "publication_receipt", "directory_authority",
        "directory_topology_digest", "materialized",
    }
    row = dict(_closed(value, fields, label="evaluation publication"))
    if (
        row["schema_version"]
        != "bernini-action-preservation-evaluation-publication-binding-v1"
        or row["evaluation_root"] != str(evaluation_root)
        or row["materialized"] is not True
    ):
        raise DecodedEvaluationBridgeError(
            "evaluation publication header differs"
        )
    topology_digest = _sha(
        row["directory_topology_digest"],
        label="evaluation directory topology digest",
    )
    _sha(expected_input_digest, label="expected evaluation input digest")
    _sha(expected_manifest_digest, label="expected evaluation manifest digest")
    if not isinstance(expected_evaluation_id, str) or not expected_evaluation_id:
        raise DecodedEvaluationBridgeError(
            "expected evaluation ID differs"
        )
    root_authority = dict(
        _closed(
            row["root_authority"],
            _COMMON_ROOT_AUTHORITY_FIELDS,
            label="evaluation root authority",
        )
    )
    if (
        root_authority["schema_version"]
        != "bernini-retained-directory-authority-v1"
        or root_authority["path"] != str(evaluation_root)
        or root_authority["retained_parent_fd"] is not True
        or root_authority["retained_root_fd"] is not True
        or not isinstance(root_authority["entries"], list)
        or len(root_authority["entries"]) != len(set(root_authority["entries"]))
    ):
        raise DecodedEvaluationBridgeError(
            "evaluation root authority differs"
        )
    for label, identity in (
        ("evaluation root", root_authority["identity"]),
        ("evaluation root parent", root_authority["parent_identity"]),
    ):
        if (
            not isinstance(identity, Mapping)
            or set(identity) != plan._IDENTITY_FIELDS
            or any(type(identity[key]) is not int or identity[key] < 0
                   for key in plan._IDENTITY_FIELDS)
        ):
            raise DecodedEvaluationBridgeError(f"{label} identity differs")
    live_work_root: dict[str, Any] | None = None
    if work_root_binding is not None:
        try:
            live_work_root = verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if (
            evaluation_root.parent != Path(live_work_root["path"])
            or any(
                root_authority["parent_identity"][field]
                != live_work_root["root_immutable_identity"][field]
                for field in ("device", "inode", "uid", "gid", "mode", "rdev")
            )
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation publication parent differs from held work root"
            )
    if verify_files:
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            evaluation_root.name if live_work_root is not None else evaluation_root,
            flags,
            **(
                {"dir_fd": live_work_root["root_fd"]}
                if live_work_root is not None else {}
            ),
        )
        try:
            before = os.fstat(descriptor)
            first = os.listdir(descriptor)
            middle = os.fstat(descriptor)
            second = os.listdir(descriptor)
            after = os.fstat(descriptor)
            named = (
                os.stat(
                    evaluation_root.name,
                    dir_fd=live_work_root["root_fd"],
                    follow_symlinks=False,
                )
                if live_work_root is not None
                else evaluation_root.lstat()
            )
        finally:
            os.close(descriptor)
        expected_identity = dict(root_authority["identity"])
        expected_identity["mode"] = stat.S_IMODE(expected_identity["mode"])
        if (
            _file_identity(before) != _file_identity(middle)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(named)
            or _captured_directory(evaluation_root, before)
            != {"path": str(evaluation_root), **expected_identity}
            or sorted(first) != sorted(second)
            or sorted(first) != sorted(root_authority["entries"])
            or len(first) != len(root_authority["entries"])
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation root physical authority differs"
            )
        parent = (
            os.fstat(live_work_root["root_fd"])
            if live_work_root is not None
            else evaluation_root.parent.lstat()
        )
        observed_parent = plan._identity_row(parent)
        expected_parent = root_authority["parent_identity"]
        if any(
            observed_parent[key] != expected_parent[key]
            for key in ("device", "inode", "uid", "gid", "mode", "rdev")
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation root parent immutable authority differs"
            )
    publications: dict[str, dict[str, Any]] = {}
    expected = {
        "publication_receipt": (
            plan.PUBLICATION_FILENAME, "publication_digest"
        ),
        "directory_authority": (
            plan.DIRECTORY_AUTHORITY_FILENAME, "authority_digest"
        ),
    }
    decoded: dict[str, dict[str, Any]] = {}
    for key, (filename, digest_field) in expected.items():
        item = dict(
            _closed(
                row[key], {"file", digest_field},
                label=f"evaluation {key}",
            )
        )
        file_row = _validate_captured_file(
            item["file"], label=f"evaluation {key} file",
            verify_file=verify_files,
        )
        if file_row["path"] != str(evaluation_root / filename):
            raise DecodedEvaluationBridgeError(
                f"evaluation {key} path differs"
            )
        raw, observed = _stable_file(
            file_row["path"], label=f"evaluation {key}",
            expected_sha256=file_row["sha256"],
        ) if verify_files else (b"", file_row)
        if verify_files and observed != file_row:
            raise DecodedEvaluationBridgeError(
                f"evaluation {key} identity differs"
            )
        _sha(item[digest_field], label=f"evaluation {key} object digest")
        if verify_files:
            value_row = _json(raw, label=f"evaluation {key}", canonical=True)
            _verify_object_digest(
                value_row, field=digest_field, label=f"evaluation {key}"
            )
            if value_row[digest_field] != item[digest_field]:
                raise DecodedEvaluationBridgeError(
                    f"evaluation {key} digest binding differs"
                )
            decoded[key] = value_row
        publications[key] = {"file": file_row, digest_field: item[digest_field]}
    if verify_files:
        publication = decoded["publication_receipt"]
        directory = decoded["directory_authority"]
        if (
            publication.get("schema_version") != plan.PUBLICATION_SCHEMA
            or publication.get("directory_authority_materialized") is not True
            or publication.get("directory_topology_digest") != topology_digest
            or publication.get("directory_authority_file_sha256")
            != publications["directory_authority"]["file"]["sha256"]
            or publication.get("directory_authority_digest")
            != directory.get("authority_digest")
            or directory.get("topology_digest") != topology_digest
            or directory.get("materialized") is not True
            or directory.get("evaluation_root") != str(evaluation_root)
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation publication cross-binding differs"
            )
        topology = publication.get("directory_topology")
        if not isinstance(topology, list):
            raise DecodedEvaluationBridgeError(
                "evaluation publication topology closure differs"
            )
        publication_files = publication.get("files")
        expected_relpaths = {
            plan.INPUT_FILENAME,
            plan.MANIFEST_FILENAME,
            plan.REVIEW_CONTRACT_FILENAME,
            plan.DIRECTORY_AUTHORITY_FILENAME,
            *{
                f"{plan.SHARD_DIRECTORY}/{holder['job_id']}.json"
                for holder in plan.HOLDER_ROWS
            },
        }
        if (
            not isinstance(publication_files, list)
            or len(publication_files) != len(expected_relpaths)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"relpath", "sha256"}
                or item.get("relpath") not in expected_relpaths
                or not isinstance(item.get("sha256"), str)
                or _SHA256.fullmatch(item["sha256"]) is None
                for item in publication_files
            )
            or len({item["relpath"] for item in publication_files})
            != len(expected_relpaths)
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation publication payload closure differs"
            )
        file_sha_by_relpath = {
            item["relpath"]: item["sha256"] for item in publication_files
        }
        retained: plan.RetainedPublicationRoot | None = None
        try:
            retained = plan.RetainedPublicationRoot.open_materialized(
                evaluation_root,
                directory_authority=directory,
                topology=topology,
                holder_job_id=None,
                label="bridge evaluation publication root",
                error_type=DecodedEvaluationBridgeError,
                retained_parent_fd=(
                    None if live_work_root is None
                    else live_work_root["root_fd"]
                ),
                retained_parent_parent_fd=(
                    None if live_work_root is None
                    else live_work_root["parent_fd"]
                ),
                expected_parent_immutable_identity=(
                    None if live_work_root is None
                    else live_work_root["root_immutable_identity"]
                ),
                expected_parent_parent_immutable_identity=(
                    None if live_work_root is None
                    else live_work_root["parent_immutable_identity"]
                ),
                expected_root_authority=root_authority,
            )
            captured_values: dict[str, dict[str, Any]] = {}
            for relative in sorted(expected_relpaths):
                raw, captured = retained.read_bytes(
                    relative,
                    expected_sha256=file_sha_by_relpath[relative],
                )
                value_row = _json(
                    raw, label=f"evaluation publication {relative}",
                    canonical=True,
                )
                captured_values[relative] = value_row
                if relative == plan.PUBLICATION_FILENAME:
                    raise AssertionError("publication receipt is not a payload")
                if relative == plan.DIRECTORY_AUTHORITY_FILENAME and (
                    captured != publications["directory_authority"]["file"]
                    or value_row != directory
                ):
                    raise DecodedEvaluationBridgeError(
                        "retained directory authority capture differs"
                    )
            receipt_raw, receipt_capture = retained.read_bytes(
                plan.PUBLICATION_FILENAME,
                expected_sha256=publications["publication_receipt"]["file"][
                    "sha256"
                ],
            )
            if (
                receipt_capture != publications["publication_receipt"]["file"]
                or _json(
                    receipt_raw, label="retained publication receipt",
                    canonical=True,
                ) != publication
            ):
                raise DecodedEvaluationBridgeError(
                    "retained publication receipt capture differs"
                )
            published_bundle = {
                "input_spec": captured_values[plan.INPUT_FILENAME],
                "manifest": captured_values[plan.MANIFEST_FILENAME],
                "review_contract": captured_values[
                    plan.REVIEW_CONTRACT_FILENAME
                ],
                "shards": {
                    holder["job_id"]: captured_values[
                        f"{plan.SHARD_DIRECTORY}/{holder['job_id']}.json"
                    ]
                    for holder in plan.HOLDER_ROWS
                },
            }
            validated_publication = plan.validate_publication_receipt(
                publication,
                bundle=published_bundle,
                directory_authority=directory,
                verify_directory_authority=True,
            )
            for reservation in validated_publication[
                "holder_completion_reservations"
            ]:
                raw, captured = retained.read_bytes(
                    reservation["relative_path"],
                    expected_sha256=reservation["sha256"],
                    expected_mode=reservation["mode"],
                )
                expected_capture = {
                    "path": reservation["path"],
                    "sha256": reservation["sha256"],
                    "size": reservation["size"],
                    **reservation["identity"],
                }
                expected_capture["mode"] = reservation["mode"]
                if raw != b"" or captured != expected_capture:
                    raise DecodedEvaluationBridgeError(
                        "holder completion reservation capture differs"
                    )
            if (
                validated_publication["evaluation_id"]
                != expected_evaluation_id
                or validated_publication["input_digest"]
                != expected_input_digest
                or validated_publication["manifest_digest"]
                != expected_manifest_digest
            ):
                raise DecodedEvaluationBridgeError(
                    "physical/publication evaluation binding differs"
                )
            if validation_barrier is not None:
                validation_barrier(
                    "before-final-retained-replay", evaluation_root
                )
            retained.seal(topology=topology)
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        finally:
            if retained is not None:
                retained.close()
        root_rows = [
            item for item in directory.get("rows", [])
            if item.get("relative_path") == "."
        ]
        if (
            len(root_rows) != 1
            or root_rows[0].get("identity") != root_authority["identity"]
            or root_rows[0].get("expected_entries")
            != root_authority["entries"]
        ):
            raise DecodedEvaluationBridgeError(
                "evaluation root/directory authority differs"
            )
        if live_work_root is not None:
            try:
                verified_release.validate_inherited_work_root_binding(
                    live_work_root,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
    row["root_authority"] = root_authority
    row.update(publications)
    return row


def validate_physical_bindings(
    value: Any, *, verify_files: bool = True,
    require_evaluation_publication: bool = False,
    evaluation_validation_barrier: Any = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_root", "input_digest",
        "manifest_digest", "training_experiment_root", "training_complete",
        "training_audit", "training_audit_digest",
        "training_audit_serialization",
        "source_preprocessing_authority_sha256",
        "source_preprocessing_authority_digest",
        "source_preprocessing_source_order",
        "pin_files", "eval_release", "sources",
        "checkpoints", "runtime", "deployment_authority",
        "calibration_digest", "training_loss_read_or_copied",
        "evaluation_publication",
        "decoded_evaluation_complete", "scientific_promotion_authorized",
        "physical_bindings_digest",
    }
    row = dict(_closed(value, fields, label="physical bindings"))
    if row["schema_version"] != PHYSICAL_BINDINGS_SCHEMA:
        raise DecodedEvaluationBridgeError("physical bindings schema differs")
    deployment_authority = _validate_deployment_authority(
        row["deployment_authority"], verify_files=verify_files
    )
    _sha(row["input_digest"], label="evaluation input digest")
    _sha(row["manifest_digest"], label="evaluation manifest digest")
    evaluation_root = _absolute(row["evaluation_root"], label="evaluation root")
    evaluation_publication = _validate_evaluation_publication(
        row["evaluation_publication"], evaluation_root=evaluation_root,
        verify_files=verify_files,
        required=require_evaluation_publication,
        expected_evaluation_id=row["evaluation_id"],
        expected_input_digest=row["input_digest"],
        expected_manifest_digest=row["manifest_digest"],
        validation_barrier=evaluation_validation_barrier,
        work_root_binding=work_root_binding,
    )
    experiment = _absolute(row["training_experiment_root"], label="training experiment root")
    if verify_files:
        _plain_directory(experiment, label="training experiment root")
    completion = dict(
        _closed(
            row["training_complete"],
            set(_CAPTURED_FILE_FIELDS) | {"completion_digest", "retained_tree_digest"},
            label="bound training completion",
        )
    )
    _sha(completion["completion_digest"], label="training completion digest")
    _sha(completion["retained_tree_digest"], label="retained tree digest")
    if completion["path"] != str(experiment / "TRAINING_COMPLETE.json"):
        raise DecodedEvaluationBridgeError("bound training completion path differs")
    _validate_captured_file(
        {key: completion[key] for key in _CAPTURED_FILE_FIELDS},
        label="bound training completion file",
        verify_file=verify_files,
    )
    audit = _validate_captured_file(
        row["training_audit"], label="bound training audit", verify_file=verify_files
    )
    if audit["path"] != str(experiment / "logs" / "training-audit.json"):
        raise DecodedEvaluationBridgeError("bound training audit path differs")
    _sha(row["training_audit_digest"], label="bound training audit digest")
    if row["training_audit_serialization"] != TRAINING_AUDIT_SERIALIZATION:
        raise DecodedEvaluationBridgeError(
            "bound training audit serialization differs"
        )
    if (
        row["source_preprocessing_authority_sha256"]
        != SOURCE_PREPROCESSING_AUTHORITY_SHA256
        or row["source_preprocessing_authority_digest"]
        != SOURCE_PREPROCESSING_AUTHORITY_DIGEST
        or row["source_preprocessing_source_order"] != list(plan.FITTED_IIDS)
    ):
        raise DecodedEvaluationBridgeError(
            "bound source preprocessing authority differs"
        )
    pin_files_value = dict(
        _closed(
            row["pin_files"],
            {
                "source_manifest", "adapter_release_manifest",
                "model_release_manifest", "inference_release_manifest",
                "inference_config", "source_preprocessing", "calibration",
            },
            label="bound pin files",
        )
    )
    pin_files: dict[str, Any] = {}
    for key in (
        "source_manifest", "adapter_release_manifest", "model_release_manifest",
        "inference_release_manifest", "inference_config", "source_preprocessing",
    ):
        pin_files[key] = _validate_captured_file(
            pin_files_value[key], label=f"bound {key}", verify_file=verify_files
        )
    if (
        pin_files["source_preprocessing"]["sha256"]
        != SOURCE_PREPROCESSING_AUTHORITY_SHA256
    ):
        raise DecodedEvaluationBridgeError(
            "bound source preprocessing file differs"
        )
    source_preprocessing: dict[str, Any] | None = None
    if verify_files:
        preprocessing_raw, _ = _stable_file(
            pin_files["source_preprocessing"]["path"],
            label="bound source preprocessing authority",
            expected_sha256=SOURCE_PREPROCESSING_AUTHORITY_SHA256,
        )
        source_preprocessing = _validate_source_preprocessing_authority(
            _json(
                preprocessing_raw,
                label="bound source preprocessing authority",
                canonical=True,
            ),
            source_manifest_sha256=pin_files["source_manifest"]["sha256"],
        )
    if row["calibration_digest"] is None:
        if pin_files_value["calibration"] is not None:
            raise DecodedEvaluationBridgeError("bound calibration closure differs")
        pin_files["calibration"] = None
    else:
        pin_files["calibration"] = _validate_captured_file(
            pin_files_value["calibration"], label="bound calibration",
            verify_file=verify_files,
        )
        calibration_raw, _ = _stable_file(
            pin_files["calibration"]["path"], label="bound calibration",
            expected_sha256=pin_files["calibration"]["sha256"],
        )
        calibration_value = _json(
            calibration_raw, label="bound calibration", canonical=True
        )
        try:
            calibration_authority = plan.gate.validate_calibration(calibration_value)
        except plan.gate.ActionPreservationGateError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        if calibration_authority["calibration_digest"] != row["calibration_digest"]:
            raise DecodedEvaluationBridgeError("bound calibration digest differs")
    eval_release = validate_eval_release_binding(
        row["eval_release"], verify_files=verify_files
    )
    if any(
        eval_release["manifest_file"][field]
        != pin_files["inference_release_manifest"][field]
        for field in _CAPTURED_FILE_FIELDS
    ):
        raise DecodedEvaluationBridgeError(
            "bound eval release manifest differs from inference release pin"
        )
    completion_value: dict[str, Any] | None = None
    audit_rows: dict[tuple[str, int], Mapping[str, Any]] | None = None
    if verify_files:
        completion_raw, _ = _stable_file(
            completion["path"], label="replayed training completion",
            expected_sha256=completion["sha256"],
        )
        completion_value = _json(
            completion_raw, label="replayed training completion", canonical=True
        )
        completion_unsigned = dict(completion_value)
        completion_digest = completion_unsigned.pop("completion_digest", None)
        if (
            completion_digest != completion["completion_digest"]
            or object_sha256(completion_unsigned) != completion_digest
            or completion_value.get("training_audit_sha256") != audit["sha256"]
            or completion_value.get("source_data_manifest_sha256")
            != pin_files["source_manifest"]["sha256"]
            or completion_value.get("release_manifest_sha256")
            != pin_files["adapter_release_manifest"]["sha256"]
        ):
            raise DecodedEvaluationBridgeError(
                "replayed completion/pin-file authority differs"
            )
        audit_raw, _ = _stable_file(
            audit["path"], label="replayed training audit",
            expected_sha256=audit["sha256"],
        )
        audit_value = _producer_training_audit_json(
            audit_raw, label="replayed training audit"
        )
        if object_sha256(audit_value) != row["training_audit_digest"]:
            raise DecodedEvaluationBridgeError(
                "replayed training audit digest differs"
            )
        audit_rows = _validated_audit_rows(audit_value)

    source_fields = {
        "iid", "source_video", "source_receipt", "instruction_sha256",
        "action_review_contract_digest", "seed"
    }
    if not isinstance(row["sources"], list) or len(row["sources"]) != len(plan.FITTED_IIDS):
        raise DecodedEvaluationBridgeError("bound source count differs")
    sources: list[dict[str, Any]] = []
    for expected_iid, item_value in zip(plan.FITTED_IIDS, row["sources"]):
        item = dict(_closed(item_value, source_fields, label="bound source"))
        if item["iid"] != expected_iid:
            raise DecodedEvaluationBridgeError("bound source IID order differs")
        item["source_video"] = _validate_captured_file(
            item["source_video"], label=f"bound source video {expected_iid}",
            verify_file=verify_files,
        )
        item["source_receipt"] = _validate_captured_file(
            item["source_receipt"], label=f"bound source receipt {expected_iid}",
            verify_file=verify_files,
        )
        _sha(item["instruction_sha256"], label="bound instruction")
        _sha(
            item["action_review_contract_digest"],
            label="bound action review contract",
        )
        if type(item["seed"]) is not int or not 0 <= item["seed"] < 2**63:
            raise DecodedEvaluationBridgeError("bound source seed differs")
        if source_preprocessing is not None:
            authority_source = source_preprocessing["sources"][len(sources)]
            if (
                item["source_video"]["path"]
                != authority_source["source_video_path"]
                or item["source_video"]["sha256"]
                != authority_source["source_video_sha256"]
                or item["source_receipt"]["path"]
                != authority_source["source_receipt_path"]
                or item["source_receipt"]["sha256"]
                != authority_source["source_receipt_sha256"]
                or item["instruction_sha256"]
                != authority_source["instruction_sha256"]
                or item["action_review_contract_digest"]
                != authority_source["action_review_contract"]["contract_digest"]
                or item["seed"] != authority_source["seed"]
            ):
                raise DecodedEvaluationBridgeError(
                    "bound source differs from source preprocessing authority"
                )
        sources.append(item)

    checkpoint_fields = {
        "arm", "checkpoint_step", "checkpoint_root", "checkpoint_receipt",
        "checkpoint_receipt_digest", "adapter_model", "adapter_config", "optimizer",
    }
    if not isinstance(row["checkpoints"], list) or len(row["checkpoints"]) != 32:
        raise DecodedEvaluationBridgeError("bound checkpoint count differs")
    checkpoints: list[dict[str, Any]] = []
    for expected_arm in plan.ARMS:
        for expected_step in plan.CHECKPOINT_STEPS:
            index = len(checkpoints)
            item = dict(
                _closed(row["checkpoints"][index], checkpoint_fields, label="bound checkpoint")
            )
            if (item["arm"], item["checkpoint_step"]) != (expected_arm, expected_step):
                raise DecodedEvaluationBridgeError("bound checkpoint order differs")
            expected_root = experiment / "runs" / expected_arm / f"checkpoint-{expected_step:08d}"
            if item["checkpoint_root"] != str(expected_root):
                raise DecodedEvaluationBridgeError("bound checkpoint path differs")
            if verify_files:
                _plain_directory(expected_root, label=f"bound checkpoint {expected_arm}@{expected_step}")
            item["checkpoint_receipt"] = _validate_captured_file(
                item["checkpoint_receipt"], label="bound checkpoint receipt",
                verify_file=verify_files,
            )
            item["adapter_model"] = _validate_captured_file(
                item["adapter_model"], label="bound adapter model",
                verify_file=verify_files,
            )
            item["adapter_config"] = _validate_captured_file(
                item["adapter_config"], label="bound adapter config",
                verify_file=verify_files,
            )
            item["optimizer"] = _validate_captured_file(
                item["optimizer"], label="bound optimizer", verify_file=verify_files
            )
            if item["checkpoint_receipt"]["path"] != str(expected_root / "receipt.json"):
                raise DecodedEvaluationBridgeError("bound checkpoint receipt path differs")
            if item["adapter_model"]["path"] != str(expected_root / "adapter" / "adapter_model.safetensors"):
                raise DecodedEvaluationBridgeError("bound adapter path differs")
            if item["adapter_config"]["path"] != str(expected_root / "adapter" / "adapter_config.json"):
                raise DecodedEvaluationBridgeError("bound adapter config path differs")
            if item["optimizer"]["path"] != str(expected_root / "optimizer.pt"):
                raise DecodedEvaluationBridgeError("bound optimizer path differs")
            _sha(item["checkpoint_receipt_digest"], label="bound checkpoint receipt digest")
            if verify_files:
                assert audit_rows is not None
                audit_row = audit_rows[(expected_arm, expected_step)]
                expected_hashes = {
                    "checkpoint_receipt": audit_row["receipt_sha256"],
                    "adapter_model": audit_row["adapter_sha256"],
                    "adapter_config": audit_row["adapter_config_sha256"],
                    "optimizer": audit_row["optimizer_sha256"],
                }
                if any(
                    item[file_key]["sha256"] != expected_sha
                    for file_key, expected_sha in expected_hashes.items()
                ):
                    raise DecodedEvaluationBridgeError(
                        "bound checkpoint differs from replayed training audit"
                    )
                receipt_raw, _ = _stable_file(
                    item["checkpoint_receipt"]["path"],
                    label="replayed checkpoint receipt",
                    expected_sha256=item["checkpoint_receipt"]["sha256"],
                )
                receipt_value = _json(
                    receipt_raw, label="replayed checkpoint receipt", canonical=True
                )
                if _validated_checkpoint_receipt_digest(
                    receipt_value, arm=expected_arm, step=expected_step
                ) != item["checkpoint_receipt_digest"]:
                    raise DecodedEvaluationBridgeError(
                        "bound checkpoint receipt digest differs"
                    )
            checkpoints.append(item)

    runtime_fields = {
        "root_python", "python", "site_packages", "torchrun",
        "deployment_controller", "controller_authority", "infer_lora",
        "decoder_adapter", "ffprobe",
        "verified_release", "eval_release_root", "eval_release_archive",
        "eval_release_envelope", "eval_release_manifest_digest",
        "eval_release_content_revision", "eval_release_envelope_digest",
        "bernini_root", "veomni_root",
        "model_checkpoint_root", "expected_bernini_commit", "expected_veomni_commit",
        "expected_checkpoint_tree_sha256", "method_source_revision",
        "method_source_archive_sha256", "num_inference_steps",
    }
    runtime = dict(_closed(row["runtime"], runtime_fields, label="bound runtime"))
    runtime["root_python"] = _validate_captured_file(
        runtime["root_python"], label="bound root bootstrap Python",
        verify_file=verify_files,
    )
    runtime["python"] = _validate_captured_file(
        runtime["python"], label="bound runtime Python", verify_file=verify_files
    )
    runtime["site_packages"] = _validate_captured_directory(
        runtime["site_packages"], label="bound runtime site-packages",
        verify_directory=verify_files,
    )
    runtime["torchrun"] = torchrun_runtime_binding(runtime["torchrun"])
    if verify_files:
        runtime["torchrun"]["source"] = _validate_captured_file(
            runtime["torchrun"]["source"], label="bound torchrun source",
            verify_file=True,
        )
        runtime["torchrun"]["subprocess_handler_source"] = (
            _validate_captured_file(
                runtime["torchrun"]["subprocess_handler_source"],
                label="bound torchrun subprocess handler source",
                verify_file=True,
            )
        )
        runtime["torchrun"]["site_packages"] = _validate_captured_directory(
            runtime["torchrun"]["site_packages"],
            label="bound torchrun site-packages", verify_directory=True,
        )
    if runtime["torchrun"]["site_packages"] != runtime["site_packages"]:
        raise DecodedEvaluationBridgeError(
            "bound torchrun site-packages continuity differs"
        )
    runtime["deployment_controller"] = _validate_captured_file(
        runtime["deployment_controller"], label="bound detached eval controller",
        verify_file=verify_files,
    )
    runtime["controller_authority"] = controller_authority_runtime_binding(
        runtime["controller_authority"]
    )
    if verify_files:
        runtime["controller_authority"]["receipt"] = _validate_captured_file(
            runtime["controller_authority"]["receipt"],
            label="bound controller authority receipt", verify_file=True,
        )
    runtime["infer_lora"] = _validate_captured_file(
        runtime["infer_lora"], label="bound infer_lora", verify_file=verify_files
    )
    runtime["decoder_adapter"] = _validate_captured_file(
        runtime["decoder_adapter"], label="bound decoder adapter",
        verify_file=verify_files,
    )
    runtime["ffprobe"] = _validate_captured_file(
        runtime["ffprobe"], label="bound ffprobe", verify_file=verify_files
    )
    runtime["verified_release"] = _validate_captured_file(
        runtime["verified_release"], label="bound verified eval runtime",
        verify_file=verify_files,
    )
    runtime["eval_release_archive"] = _validate_captured_file(
        runtime["eval_release_archive"], label="bound eval release archive",
        verify_file=verify_files,
    )
    runtime["eval_release_envelope"] = _validate_captured_file(
        runtime["eval_release_envelope"], label="bound eval release envelope",
        verify_file=verify_files,
    )
    release_members = {
        item["relative_path"]: item for item in eval_release["members"]
    }
    for runtime_key, relative_path in (
        ("infer_lora", "infer_lora.py"),
        (
            "decoder_adapter",
            "action_preservation_decoded_eval_decoder_adapter_v1.py",
        ),
        (
            "verified_release",
            "action_preservation_decoded_eval_verified_release_v1.py",
        ),
    ):
        if any(
            runtime[runtime_key][field] != release_members[relative_path][field]
            for field in _CAPTURED_FILE_FIELDS
        ):
            raise DecodedEvaluationBridgeError(
                f"bound runtime {runtime_key} differs from exact eval release"
            )
    if (
        runtime["eval_release_root"] != eval_release["release_root"]
        or any(
            runtime["eval_release_archive"][field]
            != eval_release["archive_file"][field]
            for field in _CAPTURED_FILE_FIELDS
        )
        or any(
            runtime["eval_release_envelope"][field]
            != eval_release["envelope_file"][field]
            for field in _CAPTURED_FILE_FIELDS
        )
        or runtime["eval_release_manifest_digest"]
        != eval_release["manifest_digest"]
        or runtime["eval_release_content_revision"]
        != eval_release["content_revision"]
        or runtime["eval_release_envelope_digest"]
        != eval_release["envelope_digest"]
    ):
        raise DecodedEvaluationBridgeError(
            "bound runtime detached eval release authority differs"
        )
    try:
        verified_release.validate_controller_authority_binding(
            runtime["controller_authority"],
            controller_binding=runtime["deployment_controller"],
            root_python_binding=runtime["root_python"],
            frozen_python_binding=runtime["python"],
            site_packages_binding=runtime["site_packages"],
            torchrun_binding=runtime["torchrun"],
            require_torchrun_continuity=True,
            release_binding=eval_release_runtime_binding(eval_release),
            verify_file=verify_files,
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    if (
        runtime["root_python"]["path"] != str(ROOT_PYTHON_PATH)
        or runtime["root_python"]["uid"] != ROOT_PYTHON_UID
        or runtime["root_python"]["gid"] != ROOT_PYTHON_GID
        or runtime["root_python"]["mode"] != ROOT_PYTHON_MODE
        or runtime["deployment_controller"]["mode"] != 0o444
        or runtime["controller_authority"]["receipt"]["mode"] != 0o444
        or runtime["ffprobe"]["path"] != str(FFPROBE_PATH)
        or runtime["ffprobe"]["uid"] != FFPROBE_UID
        or runtime["ffprobe"]["gid"] != FFPROBE_GID
        or runtime["ffprobe"]["mode"] != FFPROBE_MODE
    ):
        raise DecodedEvaluationBridgeError(
            "bound root-owned executable authority differs"
        )
    for key in ("bernini_root", "veomni_root", "model_checkpoint_root"):
        runtime[key] = str(_absolute(runtime[key], label=key))
        if verify_files:
            _plain_directory(runtime[key], label=key)
    _sha1(runtime["expected_bernini_commit"], label="bound Bernini commit")
    _sha1(runtime["expected_veomni_commit"], label="bound VeOmni commit")
    _sha(runtime["expected_checkpoint_tree_sha256"], label="bound checkpoint tree")
    _sha1(runtime["method_source_revision"], label="bound source revision")
    _sha(runtime["method_source_archive_sha256"], label="bound source archive")
    _sha(
        runtime["eval_release_manifest_digest"],
        label="bound eval release manifest digest",
    )
    _sha1(
        runtime["eval_release_content_revision"],
        label="bound eval release content revision",
    )
    _sha(
        runtime["eval_release_envelope_digest"],
        label="bound eval release envelope digest",
    )
    if runtime["num_inference_steps"] != 40:
        raise DecodedEvaluationBridgeError("bound inference step count differs")
    if verify_files:
        assert completion_value is not None
        if (
            runtime["method_source_revision"]
            != completion_value.get("source_revision")
            or runtime["method_source_archive_sha256"]
            != completion_value.get("source_archive_sha256")
        ):
            raise DecodedEvaluationBridgeError(
                "bound runtime differs from replayed training source"
            )
    if row["calibration_digest"] is not None:
        _sha(row["calibration_digest"], label="bound calibration digest")
    if (
        row["training_loss_read_or_copied"] is not False
        or row["decoded_evaluation_complete"] is not False
        or row["scientific_promotion_authorized"] is not False
    ):
        raise DecodedEvaluationBridgeError("physical binding authority overclaims")
    _verify_object_digest(row, field="physical_bindings_digest", label="physical bindings")
    row.update(
        training_complete=completion,
        training_audit=audit,
        pin_files=pin_files,
        eval_release=eval_release,
        sources=sources,
        checkpoints=checkpoints,
        runtime=runtime,
        deployment_authority=deployment_authority,
        evaluation_publication=evaluation_publication,
    )
    return row


def validate_bridge_receipt(
    value: Any, *, bundle: Mapping[str, Any], bindings: Mapping[str, Any],
    materialized_required: bool,
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_root", "bridge_root",
        "training_complete_file_sha256", "training_completion_digest",
        "training_audit_file_sha256", "training_audit_digest",
        "training_audit_serialization",
        "source_preprocessing_authority_sha256",
        "source_preprocessing_authority_digest",
        "source_preprocessing_source_order", "physical_bindings_digest",
        "physical_bindings_file_sha256", "input_digest", "manifest_digest",
        "evaluation_publication", "checkpoint_count", "source_count",
        "planned_decode_count", "training_loss_read_or_copied",
        "remote_launch_performed", "automatic_retry",
        "scientific_promotion_authorized", "create_only_publication",
        "bridge_receipt_digest",
    }
    row = dict(_closed(value, fields, label="bridge receipt"))
    bridge_root = _absolute(row["bridge_root"], label="bridge receipt root")
    materialized = row["evaluation_publication"] is not None
    if materialized_required is not materialized:
        raise DecodedEvaluationBridgeError(
            "bridge receipt materialization differs"
        )
    if materialized:
        _sha(
            row["physical_bindings_file_sha256"],
            label="physical bindings file SHA",
        )
    elif row["physical_bindings_file_sha256"] is not None:
        raise DecodedEvaluationBridgeError(
            "unmaterialized bridge receipt carries file SHA"
        )
    input_spec = bundle.get("input_spec")
    manifest = bundle.get("manifest")
    if not isinstance(input_spec, Mapping) or not isinstance(manifest, Mapping):
        raise DecodedEvaluationBridgeError(
            "bridge receipt bundle closure differs"
        )
    completion = bindings.get("training_complete")
    training_audit = bindings.get("training_audit")
    if not isinstance(completion, Mapping) or not isinstance(training_audit, Mapping):
        raise DecodedEvaluationBridgeError(
            "bridge receipt physical evidence differs"
        )
    expected = {
        "schema_version": BRIDGE_RECEIPT_SCHEMA,
        "evaluation_id": bindings.get("evaluation_id"),
        "evaluation_root": bindings.get("evaluation_root"),
        "training_complete_file_sha256": completion.get("sha256"),
        "training_completion_digest": completion.get("completion_digest"),
        "training_audit_file_sha256": training_audit.get("sha256"),
        "training_audit_digest": bindings.get("training_audit_digest"),
        "training_audit_serialization": TRAINING_AUDIT_SERIALIZATION,
        "source_preprocessing_authority_sha256": (
            SOURCE_PREPROCESSING_AUTHORITY_SHA256
        ),
        "source_preprocessing_authority_digest": bindings.get(
            "source_preprocessing_authority_digest"
        ),
        "source_preprocessing_source_order": list(plan.FITTED_IIDS),
        "physical_bindings_digest": bindings.get("physical_bindings_digest"),
        "input_digest": input_spec.get("input_digest"),
        "manifest_digest": manifest.get("manifest_digest"),
        "evaluation_publication": bindings.get("evaluation_publication"),
        "checkpoint_count": 32,
        "source_count": 4,
        "planned_decode_count": plan.TOTAL_DECODE_COUNT,
        "training_loss_read_or_copied": False,
        "remote_launch_performed": False,
        "automatic_retry": False,
        "scientific_promotion_authorized": False,
        "create_only_publication": True,
    }
    if bridge_root == Path(os.path.sep) or any(
        row[key] != expected_value for key, expected_value in expected.items()
    ):
        raise DecodedEvaluationBridgeError("bridge receipt binding differs")
    _verify_object_digest(
        row, field="bridge_receipt_digest", label="bridge receipt"
    )
    return row


def load_physical_bindings(
    path: str | Path, *, expected_sha256: str, verify_files: bool = True
) -> dict[str, Any]:
    raw, _ = _stable_file(
        path, label="physical bindings file", expected_sha256=expected_sha256
    )
    value = _json(raw, label="physical bindings file", canonical=True)
    return validate_physical_bindings(
        value, verify_files=verify_files,
        require_evaluation_publication=True,
    )


def _evaluation_publication_binding(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = result["publication_receipt"]
    authority = result["directory_authority"]
    return {
        "schema_version": (
            "bernini-action-preservation-evaluation-publication-binding-v1"
        ),
        "evaluation_root": authority["evaluation_root"],
        "root_authority": result["root_authority"],
        "publication_receipt": {
            "file": result["publication_receipt_file"],
            "publication_digest": receipt["publication_digest"],
        },
        "directory_authority": {
            "file": result["directory_authority_file"],
            "authority_digest": authority["authority_digest"],
        },
        "directory_topology_digest": authority["topology_digest"],
        "materialized": True,
    }


def publish_bridge(
    *, bundle: Mapping[str, Any], bindings: Mapping[str, Any],
    receipt: Mapping[str, Any], bridge_barrier: Any = None,
    evaluation_barrier: Any = None,
    work_root_binding: Mapping[str, Any] | None = None,
) -> Path:
    receipt = validate_bridge_receipt(
        receipt,
        bundle=bundle,
        bindings=bindings,
        materialized_required=False,
    )
    bridge_root = _absolute(receipt["bridge_root"], label="bridge root")
    retained_parent_fd: int | None = None
    retained_parent_parent_fd: int | None = None
    expected_parent_immutable_identity: Mapping[str, int] | None = None
    expected_parent_parent_immutable_identity: Mapping[str, int] | None = None
    live_work_root: dict[str, Any] | None = None
    if work_root_binding is not None:
        try:
            live_work_root = verified_release.validate_inherited_work_root_binding(
                work_root_binding,
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        except verified_release.DecodedEvalVerifiedReleaseError as error:
            raise DecodedEvaluationBridgeError(str(error)) from error
        work_path = Path(live_work_root["path"])
        evaluation_root = Path(bundle["manifest"]["evaluation_root"])
        if (
            bridge_root.parent != work_path
            or evaluation_root.parent != work_path
            or bridge_root == evaluation_root
        ):
            raise DecodedEvaluationBridgeError(
                "bridge/evaluation roots escape inherited work root"
            )
        retained_parent_fd = live_work_root["root_fd"]
        retained_parent_parent_fd = live_work_root["parent_fd"]
        expected_parent_immutable_identity = live_work_root[
            "root_immutable_identity"
        ]
        expected_parent_parent_immutable_identity = live_work_root[
            "parent_immutable_identity"
        ]
    if os.path.lexists(bridge_root):
        raise DecodedEvaluationBridgeError("bridge root is not fresh")
    try:
        publication_result = plan.publish_bundle_authorized(
            bundle,
            publication_barrier=evaluation_barrier,
            retained_parent_fd=retained_parent_fd,
            retained_parent_parent_fd=retained_parent_parent_fd,
            expected_parent_immutable_identity=(
                expected_parent_immutable_identity
            ),
            expected_parent_parent_immutable_identity=(
                expected_parent_parent_immutable_identity
            ),
        )
    except plan.DecodedEvaluationPlanError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    publication = _evaluation_publication_binding(publication_result)
    _validate_evaluation_publication(
        publication,
        evaluation_root=_absolute(
            publication["evaluation_root"], label="published evaluation root"
        ),
        verify_files=True, required=True,
        expected_evaluation_id=bindings["evaluation_id"],
        expected_input_digest=bindings["input_digest"],
        expected_manifest_digest=bindings["manifest_digest"],
        work_root_binding=live_work_root,
    )
    bindings_value = dict(bindings)
    bindings_value["evaluation_publication"] = publication
    bindings_value.pop("physical_bindings_digest", None)
    bindings_value["physical_bindings_digest"] = object_sha256(bindings_value)
    bindings_value = validate_physical_bindings(
        bindings_value,
        verify_files=True,
        require_evaluation_publication=True,
        work_root_binding=live_work_root,
    )
    receipt_value = dict(receipt)
    receipt_value["evaluation_publication"] = publication
    receipt_value["physical_bindings_file_sha256"] = hashlib.sha256(
        canonical_json_bytes(bindings_value) + b"\n"
    ).hexdigest()
    receipt_value["physical_bindings_digest"] = bindings_value[
        "physical_bindings_digest"
    ]
    # Adding the physical file hash changes the receipt preimage deliberately.
    receipt_value.pop("bridge_receipt_digest", None)
    receipt_value["bridge_receipt_digest"] = object_sha256(receipt_value)
    receipt_value = validate_bridge_receipt(
        receipt_value,
        bundle=bundle,
        bindings=bindings_value,
        materialized_required=True,
    )
    root_authority = plan.RetainedPublicationRoot.create(
        bridge_root, label="bridge publication root",
        error_type=DecodedEvaluationBridgeError, barrier=bridge_barrier,
        retained_parent_fd=retained_parent_fd,
        retained_parent_parent_fd=retained_parent_parent_fd,
        expected_parent_immutable_identity=expected_parent_immutable_identity,
        expected_parent_parent_immutable_identity=(
            expected_parent_parent_immutable_identity
        ),
    )
    try:
        root_authority.write_bytes(
            PHYSICAL_BINDINGS_FILENAME,
            canonical_json_bytes(bindings_value) + b"\n",
        )
        root_authority.write_bytes(
            BRIDGE_RECEIPT_FILENAME,
            canonical_json_bytes(receipt_value) + b"\n",
        )
        if root_authority.authority_row()["entries"] != sorted(
            {PHYSICAL_BINDINGS_FILENAME, BRIDGE_RECEIPT_FILENAME}
        ):
            raise DecodedEvaluationBridgeError(
                "bridge publication root closure differs"
            )
        root_authority.set_directory_mode(".", 0o555)
        root_authority.seal()
        if live_work_root is not None:
            try:
                verified_release.validate_inherited_work_root_binding(
                    live_work_root,
                    verify_open_fds=True,
                    expected_inheritable=False,
                    verify_entries=False,
                    allow_root_metadata_change=True,
                )
            except verified_release.DecodedEvalVerifiedReleaseError as error:
                raise DecodedEvaluationBridgeError(str(error)) from error
        return bridge_root / BRIDGE_RECEIPT_FILENAME
    finally:
        root_authority.close()


def _load(
    path: str | Path, *, label: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    raw, _ = _stable_file(
        path, label=label, expected_sha256=expected_sha256
    )
    return _json(raw, label=label, canonical=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--training-complete-sha256", required=True)
    parser.add_argument("--source-runtime-spec", required=True)
    parser.add_argument("--source-runtime-spec-sha256", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--bridge-root", required=True)
    args = parser.parse_args(argv)
    try:
        work_root_binding = (
            verified_release.load_inherited_work_root_environment(
                verify_open_fds=True,
                expected_inheritable=False,
                verify_entries=False,
                allow_root_metadata_change=True,
            )
        )
    except verified_release.DecodedEvalVerifiedReleaseError as error:
        raise DecodedEvaluationBridgeError(str(error)) from error
    deployment_authority = _deployment_authority_from_work_root_binding(
        work_root_binding, verify_open_fds=True
    )
    (
        _validated_deployment_authority,
        authorized_source_runtime_spec,
        authorized_source_runtime_spec_file,
    ) = _authorized_source_runtime_spec(
        deployment_authority,
        work_root_binding=work_root_binding,
    )
    if (
        str(
            _absolute(
                args.source_runtime_spec, label="source/runtime spec CLI path"
            )
        ) != authorized_source_runtime_spec_file["path"]
        or _sha(
            args.source_runtime_spec_sha256,
            label="source/runtime spec CLI SHA",
        ) != authorized_source_runtime_spec_file["sha256"]
    ):
        raise DecodedEvaluationBridgeError(
            "source/runtime spec CLI binding differs from held authority"
        )
    bundle, bindings, receipt = build_bridge(
        experiment_root=args.experiment_root,
        completion_sha256=args.training_complete_sha256,
        source_runtime_spec=authorized_source_runtime_spec,
        evaluation_id=args.evaluation_id,
        evaluation_root=args.evaluation_root,
        bridge_root=args.bridge_root,
        deployment_authority=deployment_authority,
        work_root_binding=work_root_binding,
    )
    for relative_path, module_path in (
        ("action_preservation_decoded_eval_bridge_v1.py", __file__),
        ("action_preservation_decoded_eval_plan_v1.py", plan.__file__),
        ("action_preservation_gate_v1.py", plan.gate.__file__),
    ):
        require_running_eval_release_member(
            bindings["eval_release"],
            relative_path=relative_path,
            running_path=module_path,
        )
    validate_running_verified_capture(
        bindings,
        target="action_preservation_decoded_eval_bridge_v1.py",
        expected_arguments=list(sys.argv[1:] if argv is None else argv),
        verify_file=True,
        work_root_binding=work_root_binding,
    )
    output = publish_bridge(
        bundle=bundle, bindings=bindings, receipt=receipt,
        work_root_binding=work_root_binding,
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
