#!/usr/bin/env python3
"""Receipt-first one-shot static admission controller (checked-in HOLD).

The future READY copy consumes the final sibling package-publication receipt
before naming the package root.  It then pins and replays the immutable
materialization report, launch input, launcher, and static-probe bytes, invokes
the pure-stdlib probe exactly once in-process, and publishes only its canonical
self-digested 0400 receipt.  It never invokes SSH, Slurm, a renderer, or a
subprocess.  A probe refusal produces no output and is never retried by this
controller; an already-sealed output is a terminal one-shot state.

This checked-in HOLD source performs no path, argv, directory, or process I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-static-controller-v1"
CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_STATIC_PROBE"
READY_STATE = "READY_EXPLICIT_SINGLE_STATIC_PROBE"

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
PACKAGE_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_r64_canary_v1"
)
PACKAGE_PUBLICATION_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v1."
    "publication_receipt_v2.json"
)
MATERIALIZATION_REPORT_PATH = (
    PACKAGE_ROOT / "authority/package_materialization_receipt_v1.json"
)
SOURCE_SNAPSHOT_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1"
)
STATIC_PROBE_PATH = (
    PACKAGE_ROOT / "diagnostics/case01_object_trajectory_exact5_static_probe_v1.py"
)
LAUNCHER_PATH = PACKAGE_ROOT / (
    "release/methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
)
LAUNCH_INPUT_PATH = PACKAGE_ROOT / "launch/root_launch_input_HOLD_v1.json"
LAUNCH_PAYLOAD_PATH = PACKAGE_ROOT / "launch/root_launch_payload_HOLD_v1.sh"
PLAN_PATH = (
    PACKAGE_ROOT / "plan/case01_object_trajectory_exact5_r64_HOLD_plan_v1.json"
)
STATIC_OUTPUT_PATH = PACKAGE_ROOT / "evidence/exact5_static_probe_receipt_v1.json"
OUTPUT_REPORT_PATH = (
    PACKAGE_ROOT / "final/object_trajectory_exact5_report_v1.json"
)
RUNNER_ATTESTATION_PATH = (
    PACKAGE_ROOT / "final/object_trajectory_exact5_runner_attestation_v1.json"
)
AUTHORITY_ROOT = PACKAGE_ROOT / "runtime/model-authority"
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r1-"
    "rank-cache"
)

VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
VACE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
VACE_PYTHON_SIZE = 31_490_256
STATIC_PROBE_SHA256 = (
    "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9"
)
STATIC_PROBE_SIZE = 5_887
LAUNCHER_SHA256 = (
    "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f"
)
LAUNCHER_SIZE = 27_492

REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
CONTROLLER_MODE = 0o444
MAX_JSON_SIZE = 32 * 1024 * 1024
MAX_SOURCE_SIZE = 2 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

PACKAGE_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v2-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v1"
)
STATIC_SCHEMA = "case01-object-trajectory-exact5-static-admission-v1"
LAUNCH_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-receipt-auh-v1"
)
LAUNCH_RELEASE_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-release-auh-v1"
)
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"
JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
ARM_ORDER = (
    "null_before", "route_off", "trajectory_bone_only",
    "trajectory_dog_bone", "null_after",
)
TASK_IDS = tuple(
    f"case01-object-trajectory-{arm}-full644" for arm in ARM_ORDER
)

# Final immutable AUH package authorities injected only after the independently
# audited generic HOLD was frozen.  The sibling receipt authenticates both the
# internal report and exact package-root inode identity before root observation.
PACKAGE_PUBLICATION_RECEIPT_SHA256 = (
    "b3766694f24ead6d7da04e5a1da077de69a9dbbf06df8f06ff0c9db77d84c533"
)
PACKAGE_PUBLICATION_RECEIPT_SIZE: int | str = (
    2_209
)
PACKAGE_PUBLICATION_RECEIPT_DIGEST = (
    "5cab7d2db0079d4b6960273e681c20b60941b892c3a42bfdbd70be819d991cb9"
)
MATERIALIZATION_REPORT_SHA256 = (
    "e1e4d7ae266828f27f77f39528672cd7ccae9aa067fdee291d4e5e32f9a9bf2f"
)
MATERIALIZATION_REPORT_SIZE: int | str = (
    21_743
)
MATERIALIZATION_REPORT_DIGEST = (
    "99ba2595bde82371257a46b08ef55f77f54cb5b86877aa791daf6976237868c4"
)
PACKAGE_ROOT_IDENTITY: list[int] | str = (
    [
        48, 12_038_280_342_419_913_116, 2012, 2000, 16_832, 2, 0, 4096,
        0, 1_787_357_728_317_453_482, 1_787_357_728_652_385_810,
    ]
)


class StaticControllerError(RuntimeError):
    """The reviewed one-shot static-admission contract differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise StaticControllerError("value is not canonical JSON") from error


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
                raise StaticControllerError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise StaticControllerError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise StaticControllerError(f"noncanonical JSON authority: {label}")
    return value


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise StaticControllerError("held read size differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise StaticControllerError("held read is incomplete")
    return raw


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...], raw: bytes,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.held_identity = held_identity
        self.raw = raw

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        replay = read_fd(self.descriptor, opened.st_size)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or replay != self.raw
        ):
            raise StaticControllerError(f"held authority changed: {self.path}")

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

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
        ):
            raise StaticControllerError("held package root changed")

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
        raise StaticControllerError(f"noncanonical authority path: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise StaticControllerError(f"missing authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != expected_mode
        or named.st_uid != expected_uid or named.st_gid != expected_gid
        or named.st_size <= 0 or named.st_size > maximum_size
        or (expected_size is not None and named.st_size != expected_size)
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise StaticControllerError(f"named authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
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
            raise StaticControllerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), first)
    except BaseException:
        os.close(descriptor)
        raise


def open_package_root(expected_identity: Sequence[int]) -> HeldDirectory:
    if (
        type(expected_identity) is not list or len(expected_identity) != 11
        or any(type(value) is not int for value in expected_identity)
    ):
        raise StaticControllerError("package root identity pin differs")
    named = os.lstat(PACKAGE_ROOT)
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_uid != REMOTE_UID or named.st_gid != REMOTE_GID
        or stat.S_IMODE(named.st_mode) & 0o022
        or identity(named) != tuple(expected_identity)
        or PACKAGE_ROOT.resolve(strict=True) != PACKAGE_ROOT
    ):
        raise StaticControllerError("named package root differs")
    descriptor = os.open(
        PACKAGE_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    opened = os.fstat(descriptor)
    named_after = os.lstat(PACKAGE_ROOT)
    if identity(opened) != identity(named) or identity(opened) != identity(named_after):
        os.close(descriptor)
        raise StaticControllerError("opened package root differs")
    return HeldDirectory(PACKAGE_ROOT, descriptor, identity(opened))


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "package_publication_receipt_sha256":
            PACKAGE_PUBLICATION_RECEIPT_SHA256,
        "package_publication_receipt_size": PACKAGE_PUBLICATION_RECEIPT_SIZE,
        "package_publication_receipt_digest":
            PACKAGE_PUBLICATION_RECEIPT_DIGEST,
        "materialization_report_sha256": MATERIALIZATION_REPORT_SHA256,
        "materialization_report_size": MATERIALIZATION_REPORT_SIZE,
        "materialization_report_digest": MATERIALIZATION_REPORT_DIGEST,
        "package_root_identity": PACKAGE_ROOT_IDENTITY,
    }


def blocked_dynamic_pins() -> tuple[str, ...]:
    blocked: list[str] = []
    for name, value in dynamic_pin_values().items():
        if name == "package_root_identity":
            if (
                type(value) is not list or len(value) != 11
                or any(type(item) is not int for item in value)
            ):
                blocked.append(name)
        elif name.endswith("_size"):
            if type(value) is not int or value <= 0:
                blocked.append(name)
        elif type(value) is not str or SHA_RE.fullmatch(value) is None:
            blocked.append(name)
    return tuple(blocked)


def authorization_token() -> str:
    return object_digest({
        "schema_version": SCHEMA + "-authorization-v1",
        "state": READY_STATE,
        "package_root": str(PACKAGE_ROOT),
        "package_publication_receipt": str(PACKAGE_PUBLICATION_RECEIPT_PATH),
        "materialization_report": str(MATERIALIZATION_REPORT_PATH),
        "static_probe": {
            "path": str(STATIC_PROBE_PATH), "sha256": STATIC_PROBE_SHA256,
            "size": STATIC_PROBE_SIZE,
        },
        "launcher": {
            "path": str(LAUNCHER_PATH), "sha256": LAUNCHER_SHA256,
            "size": LAUNCHER_SIZE,
        },
        "launch_input": str(LAUNCH_INPUT_PATH),
        "output": str(STATIC_OUTPUT_PATH),
        "dynamic_pins": dynamic_pin_values(),
        "probe_calls": 1,
        "retry_allowed": False,
        "ssh_allowed": False,
        "slurm_allowed": False,
    })


PUBLICATION_FIELDS = {
    "schema_version", "status", "target_root", "receipt_path",
    "materialization_receipt_path", "materialization_receipt_sha256",
    "materialization_receipt_digest", "source_snapshot_manifest_sha256",
    "source_snapshot_manifest_digest", "source_staging_receipt_sha256",
    "source_staging_receipt_digest", "publication_protocol",
    "rename_noreplace", "cooperative_writer_exclusion",
    "target_absent_rechecked_before_rename", "ordinary_posix_rename_performed",
    "publication_observation", "whole_tree_atomically_visible",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
    "receipt_is_admission", "launch_allowed", "receipt_inode_anchor",
    "receipt_digest",
}


def validate_publication_receipt(held: HeldAuthority) -> dict[str, Any]:
    value = strict_json(held.raw, label="package publication receipt")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    success = {
        "namespace_state": "target_same_inode_source_absent",
        "rename_returned_zero": True,
        "rename_error_errno": None,
        "parent_fsync_returned_zero": True,
        "parent_fsync_error_errno": None,
    }
    if (
        set(value) != PUBLICATION_FIELDS
        or hashlib.sha256(held.raw).hexdigest()
        != PACKAGE_PUBLICATION_RECEIPT_SHA256
        or len(held.raw) != PACKAGE_PUBLICATION_RECEIPT_SIZE
        or claimed != PACKAGE_PUBLICATION_RECEIPT_DIGEST
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != PACKAGE_PUBLICATION_SCHEMA
        or value.get("status") != "PUBLISHED_RECEIPT_GATED"
        or value.get("target_root") != str(PACKAGE_ROOT)
        or value.get("receipt_path") != str(PACKAGE_PUBLICATION_RECEIPT_PATH)
        or value.get("materialization_receipt_path")
        != str(MATERIALIZATION_REPORT_PATH)
        or value.get("materialization_receipt_sha256")
        != MATERIALIZATION_REPORT_SHA256
        or value.get("materialization_receipt_digest")
        != MATERIALIZATION_REPORT_DIGEST
        or value.get("publication_protocol") != PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("target_absent_rechecked_before_rename") is not True
        or value.get("ordinary_posix_rename_performed") is not True
        or value.get("publication_observation") != success
        or value.get("whole_tree_atomically_visible") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("retry_allowed") is not False
        or value.get("target_root_identity") != PACKAGE_ROOT_IDENTITY
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("launch_allowed") is not False
        or value.get("receipt_inode_anchor")
        != inode_anchor(os.fstat(held.descriptor))
        or SHA_RE.fullmatch(str(value.get("source_snapshot_manifest_sha256")))
        is None
        or SHA_RE.fullmatch(str(value.get("source_snapshot_manifest_digest")))
        is None
        or SHA_RE.fullmatch(str(value.get("source_staging_receipt_sha256")))
        is None
        or SHA_RE.fullmatch(str(value.get("source_staging_receipt_digest")))
        is None
    ):
        raise StaticControllerError("package publication receipt differs")
    return value


MATERIALIZATION_FIELDS = {
    "schema_version", "status", "launch_allowed", "root",
    "source_snapshot_root", "source_snapshot",
    "source_staging_receipt_authority", "package_publication_receipt_path",
    "publication_protocol", "rename_noreplace", "cooperative_writer_exclusion",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "release_file_count", "production_identity_count",
    "condition_and_admission_authority_count", "plan", "launch", "admission",
    "slurm_step_launched", "gpu_attempt_claimed", "artifacts", "receipt_digest",
}
LAUNCH_FIELDS = {
    "schema_version", "status", "launch_allowed", "slurm_step_launched",
    "gpu_attempt_claimed", "input", "release", "payload_path",
    "payload_sha256", "payload_size", "receipt_digest",
}
RELEASE_FIELDS = {
    "schema_version", "status", "launch_allowed", "campaign_mode",
    "selected_task_ids", "identity_roles", "identities", "input_sha256",
    "ready_overlay_required", "named_payload_execution_forbidden",
    "release_digest",
}


def _artifact_row(value: Mapping[str, Any], relative: str) -> dict[str, Any]:
    artifacts = value.get("artifacts")
    row = artifacts.get(relative) if type(artifacts) is dict else None
    if (
        type(row) is not dict or set(row) != {"sha256", "size"}
        or SHA_RE.fullmatch(str(row.get("sha256"))) is None
        or type(row.get("size")) is not int or row["size"] <= 0
    ):
        raise StaticControllerError(f"materialization artifact row differs: {relative}")
    return row


def validate_materialization_report(
    held: HeldAuthority, publication: Mapping[str, Any],
) -> dict[str, Any]:
    value = strict_json(held.raw, label="package materialization report")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    launch = value.get("launch")
    launch_unsigned = dict(launch) if type(launch) is dict else {}
    launch_claimed = launch_unsigned.pop("receipt_digest", None)
    release = launch.get("release") if type(launch) is dict else None
    release_unsigned = dict(release) if type(release) is dict else {}
    release_claimed = release_unsigned.pop("release_digest", None)
    launch_input = launch.get("input") if type(launch) is dict else None
    plan = value.get("plan")
    identity_roles = release.get("identity_roles") if type(release) is dict else None
    identities = release.get("identities") if type(release) is dict else None
    source_snapshot = value.get("source_snapshot")
    source_staging = value.get("source_staging_receipt_authority")
    if (
        set(value) != MATERIALIZATION_FIELDS
        or hashlib.sha256(held.raw).hexdigest() != MATERIALIZATION_REPORT_SHA256
        or len(held.raw) != MATERIALIZATION_REPORT_SIZE
        or claimed != MATERIALIZATION_REPORT_DIGEST
        or claimed != object_digest(unsigned)
        or publication.get("materialization_receipt_sha256")
        != hashlib.sha256(held.raw).hexdigest()
        or publication.get("materialization_receipt_digest") != claimed
        or value.get("schema_version") != MATERIALIZATION_SCHEMA
        or value.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or value.get("launch_allowed") is not False
        or value.get("root") != str(PACKAGE_ROOT)
        or value.get("source_snapshot_root") != str(SOURCE_SNAPSHOT_ROOT)
        or type(source_snapshot) is not dict
        or type(source_staging) is not dict
        or source_snapshot.get("staging_receipt_authority") != source_staging
        or source_snapshot.get("sha256")
        != publication.get("source_snapshot_manifest_sha256")
        or source_snapshot.get("manifest_digest")
        != publication.get("source_snapshot_manifest_digest")
        or source_staging.get("sha256")
        != publication.get("source_staging_receipt_sha256")
        or source_staging.get("receipt_digest")
        != publication.get("source_staging_receipt_digest")
        or value.get("package_publication_receipt_path")
        != str(PACKAGE_PUBLICATION_RECEIPT_PATH)
        or value.get("publication_protocol") != PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("retry_allowed") is not False
        or value.get("release_file_count") != 25
        or value.get("production_identity_count") != 25
        or value.get("condition_and_admission_authority_count") != 6
        or value.get("admission") != {
            "static_executed": False, "root_fake_executed": False,
            "world4_executed": False,
        }
        or value.get("slurm_step_launched") is not False
        or value.get("gpu_attempt_claimed") is not False
        or type(value.get("artifacts")) is not dict
        or len(value["artifacts"]) != 28
        or type(plan) is not dict
        or set(plan) != {"path", "sha256", "plan_digest"}
        or plan.get("path") != str(PLAN_PATH)
        or SHA_RE.fullmatch(str(plan.get("sha256"))) is None
        or SHA_RE.fullmatch(str(plan.get("plan_digest"))) is None
        or type(launch) is not dict or set(launch) != LAUNCH_FIELDS
        or launch.get("schema_version") != LAUNCH_RECEIPT_SCHEMA
        or launch.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or launch.get("launch_allowed") is not False
        or launch.get("slurm_step_launched") is not False
        or launch.get("gpu_attempt_claimed") is not False
        or launch.get("payload_path") != str(LAUNCH_PAYLOAD_PATH)
        or SHA_RE.fullmatch(str(launch.get("payload_sha256"))) is None
        or type(launch.get("payload_size")) is not int
        or launch["payload_size"] <= 0
        or launch_claimed != object_digest(launch_unsigned)
        or type(release) is not dict or set(release) != RELEASE_FIELDS
        or release.get("schema_version") != LAUNCH_RELEASE_SCHEMA
        or release.get("status") != "HOLD_NOT_LAUNCHABLE"
        or release.get("launch_allowed") is not False
        or release.get("campaign_mode") != CAMPAIGN
        or release.get("selected_task_ids") != list(TASK_IDS)
        or release.get("ready_overlay_required") is not True
        or release.get("named_payload_execution_forbidden") is not True
        or type(identity_roles) is not list or len(identity_roles) != 25
        or not all(type(role) is str and role for role in identity_roles)
        or len(set(identity_roles)) != 25
        or type(identities) is not dict
        or set(identities) != set(identity_roles)
        or any(
            type(row) is not dict
            or set(row) != {"path", "sha256", "size"}
            or type(row.get("path")) is not str
            or not os.path.isabs(row["path"])
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            for row in identities.values()
        )
        or identities.get("plan") != {
            "path": plan.get("path"), "sha256": plan.get("sha256"),
            "size": identities.get("plan", {}).get("size"),
        }
        or release_claimed != object_digest(release_unsigned)
        or type(launch_input) is not dict
        or set(launch_input) != {"path", "sha256", "size", "mode", "nlink"}
        or launch_input.get("path") != str(LAUNCH_INPUT_PATH)
        or SHA_RE.fullmatch(str(launch_input.get("sha256"))) is None
        or type(launch_input.get("size")) is not int
        or launch_input["size"] <= 0
        or launch_input.get("mode") != FILE_MODE
        or launch_input.get("nlink") != 1
        or release.get("input_sha256") != launch_input.get("sha256")
    ):
        raise StaticControllerError("package materialization report differs")
    static_row = _artifact_row(
        value, "diagnostics/case01_object_trajectory_exact5_static_probe_v1.py",
    )
    launcher_row = _artifact_row(
        value,
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py",
    )
    if static_row != {"sha256": STATIC_PROBE_SHA256, "size": STATIC_PROBE_SIZE}:
        raise StaticControllerError("static-probe materialization row differs")
    if launcher_row != {"sha256": LAUNCHER_SHA256, "size": LAUNCHER_SIZE}:
        raise StaticControllerError("launcher materialization row differs")
    return value


def load_static_probe(raw: bytes) -> types.ModuleType:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise StaticControllerError("static probe is not UTF-8") from error
    module = types.ModuleType("_held_case01_object_trajectory_static_probe")
    module.__file__ = str(STATIC_PROBE_PATH)
    module.__package__ = None
    exec(
        compile(source, str(STATIC_PROBE_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    if (
        getattr(module, "SCHEMA", None) != STATIC_SCHEMA
        or not callable(getattr(module, "probe", None))
        or not callable(getattr(module, "_load_launcher", None))
    ):
        raise StaticControllerError("pinned static-probe API differs")
    return module


def load_launcher(raw: bytes) -> types.ModuleType:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise StaticControllerError("launcher is not UTF-8") from error
    module = types.ModuleType("_held_case01_object_trajectory_launcher")
    module.__file__ = str(LAUNCHER_PATH)
    module.__package__ = None
    exec(
        compile(source, str(LAUNCHER_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    roles = getattr(module, "IDENTITY_ROLES", None)
    basenames = getattr(module, "METHOD_ROLE_BASENAMES", None)
    expected = getattr(module, "EXPECTED_STATIC_SHA256", None)
    if (
        getattr(module, "SCHEMA", None) != LAUNCH_RELEASE_SCHEMA
        or getattr(module, "INPUT_SCHEMA", None)
        != "case01-object-trajectory-exact5-hold-launch-input-auh-v1"
        or getattr(module, "RECEIPT_SCHEMA", None) != LAUNCH_RECEIPT_SCHEMA
        or getattr(module, "CAMPAIGN", None) != CAMPAIGN
        or tuple(getattr(module, "ARM_ORDER", ())) != ARM_ORDER
        or tuple(getattr(module, "TASK_IDS", ())) != TASK_IDS
        or type(roles) is not tuple or len(roles) != 25
        or len(set(roles)) != 25
        or type(basenames) is not dict or len(basenames) != 14
        or not set(basenames).issubset(set(roles))
        or type(expected) is not dict
        or set(expected) != set(roles) - {"python", "ffmpeg", "ffprobe", "plan"}
    ):
        raise StaticControllerError("pinned launcher API differs")
    return module


LAUNCH_INPUT_FIELDS = {
    "schema_version", "entry_mode", "campaign_mode", "holder_job_id",
    "expected_node", "expected_allocation_gpu_count", "identities",
    "output_report", "runner_attestation", "model_root", "bernini_root",
    "veomni_root", "authority_root", "rank_cache_root",
}


def validate_launch_identity_closure(
    report: Mapping[str, Any], launch_input: Mapping[str, Any],
    launcher: types.ModuleType,
) -> None:
    launch = report.get("launch")
    release = launch.get("release") if type(launch) is dict else None
    report_roles = release.get("identity_roles") if type(release) is dict else None
    report_identities = release.get("identities") if type(release) is dict else None
    input_identities = launch_input.get("identities")
    pinned_roles = getattr(launcher, "IDENTITY_ROLES", None)
    basenames = getattr(launcher, "METHOD_ROLE_BASENAMES", None)
    expected_static = getattr(launcher, "EXPECTED_STATIC_SHA256", None)
    if (
        type(pinned_roles) is not tuple or len(pinned_roles) != 25
        or type(basenames) is not dict or type(expected_static) is not dict
        or type(release) is not dict
        or report_roles != list(pinned_roles)
        or type(report_identities) is not dict
        or type(input_identities) is not dict
        or report_identities != input_identities
        or set(report_identities) != set(pinned_roles)
        or set(launch_input) != LAUNCH_INPUT_FIELDS
        or launch_input.get("schema_version")
        != "case01-object-trajectory-exact5-hold-launch-input-auh-v1"
        or launch_input.get("entry_mode") != "trusted_stdin"
        or launch_input.get("campaign_mode") != CAMPAIGN
        or launch_input.get("holder_job_id") != JOB_ID
        or launch_input.get("expected_node") != NODE
        or launch_input.get("expected_allocation_gpu_count") != 8
        or launch_input.get("output_report") != str(OUTPUT_REPORT_PATH)
        or launch_input.get("runner_attestation")
        != str(RUNNER_ATTESTATION_PATH)
        or launch_input.get("authority_root") != str(AUTHORITY_ROOT)
        or launch_input.get("rank_cache_root") != str(RANK_CACHE_ROOT)
    ):
        raise StaticControllerError("held launch input/report role closure differs")

    method_root = (
        PACKAGE_ROOT / "release/methods/bernini_action_editing"
    )
    artifacts = report.get("artifacts")
    for role in pinned_roles:
        row = input_identities.get(role)
        if (
            type(row) is not dict or set(row) != {"path", "sha256", "size"}
            or type(row.get("path")) is not str
            or type(row.get("sha256")) is not str
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise StaticControllerError(f"held launch identity differs: {role}")
        expected_sha256 = expected_static.get(role)
        if expected_sha256 is not None and row["sha256"] != expected_sha256:
            raise StaticControllerError(
                f"held launch identity SHA differs: {role}"
            )

    for role, basename in basenames.items():
        expected_path = method_root / basename
        row = input_identities[role]
        relative = str(expected_path.relative_to(PACKAGE_ROOT))
        artifact = artifacts.get(relative) if type(artifacts) is dict else None
        if (
            row["path"] != str(expected_path)
            or artifact != {"sha256": row["sha256"], "size": row["size"]}
        ):
            raise StaticControllerError(
                f"package-internal method identity escaped closure: {role}"
            )

    plan_row = input_identities.get("plan")
    plan = report.get("plan")
    if (
        type(plan_row) is not dict or type(plan) is not dict
        or plan_row.get("path") != str(PLAN_PATH)
        or plan_row.get("path") != plan.get("path")
        or plan_row.get("sha256") != plan.get("sha256")
        or launch.get("input", {}).get("path") != str(LAUNCH_INPUT_PATH)
        or _artifact_row(
            report,
            "release/methods/bernini_action_editing/"
            "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py",
        ) != {"sha256": LAUNCHER_SHA256, "size": LAUNCHER_SIZE}
    ):
        raise StaticControllerError(
            "package-internal plan/launcher/input closure differs"
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
            raise StaticControllerError("executing Python differs from held runtime")
        return held
    except BaseException:
        held.close()
        raise


def open_self_authority() -> HeldAuthority:
    return open_authority(
        Path(__file__), expected_sha256=None, expected_size=None,
        expected_mode=CONTROLLER_MODE, expected_uid=REMOTE_UID,
        expected_gid=REMOTE_GID, maximum_size=MAX_SOURCE_SIZE,
    )


STATIC_RESULT_FIELDS = {
    "schema_version", "status", "launch_allowed", "blocked_roles",
    "final_source_pins_complete", "exact_identity_count", "task_ids",
    "arm_order", "all_tasks_hard1_every_step",
    "null_arms_have_no_external_conditions",
    "route_and_active_arms_have_external_conditions", "torch_imported",
    "renderer_imported", "publication_performed", "input_sha256",
    "launcher_sha256", "receipt_digest",
}


def validate_static_result(
    value: Mapping[str, Any], *, input_sha256: str,
) -> dict[str, Any]:
    result = dict(value)
    unsigned = dict(result)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        set(result) != STATIC_RESULT_FIELDS
        or result.get("schema_version") != STATIC_SCHEMA
        or result.get("status") != "ADMITTED_STATIC_HOLD_ONLY"
        or result.get("launch_allowed") is not False
        or result.get("blocked_roles") != []
        or result.get("final_source_pins_complete") is not True
        or result.get("exact_identity_count") != 25
        or result.get("task_ids") != list(TASK_IDS)
        or result.get("arm_order") != list(ARM_ORDER)
        or result.get("all_tasks_hard1_every_step") is not True
        or result.get("null_arms_have_no_external_conditions") is not True
        or result.get("route_and_active_arms_have_external_conditions") is not True
        or result.get("torch_imported") is not False
        or result.get("renderer_imported") is not False
        or result.get("publication_performed") is not False
        or result.get("input_sha256") != input_sha256
        or result.get("launcher_sha256") != LAUNCHER_SHA256
        or claimed != object_digest(unsigned)
    ):
        raise StaticControllerError("static probe result differs")
    return result


def require_fresh_output() -> None:
    if os.path.lexists(STATIC_OUTPUT_PATH):
        raise StaticControllerError("single static output is not fresh")
    for path in (
        PACKAGE_ROOT / "outputs/media", PACKAGE_ROOT / "final",
        PACKAGE_ROOT / "runtime",
    ):
        if os.listdir(path):
            raise StaticControllerError(f"static production path is not fresh: {path}")


def create_immutable_receipt(path: Path, value: Mapping[str, Any]) -> bytes:
    if path != STATIC_OUTPUT_PATH or path.parent != PACKAGE_ROOT / "evidence":
        raise StaticControllerError("static output target path differs")
    raw = canonical(value) + b"\n"
    # The self-digest is checked before the first output-namespace mutation.
    validate_static_result(
        strict_json(raw, label="prospective static output"),
        input_sha256=str(value.get("input_sha256")),
    )
    parent_info = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != REMOTE_UID or parent_info.st_gid != REMOTE_GID
        or stat.S_IMODE(parent_info.st_mode) & 0o022
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise StaticControllerError("static output parent differs")
    parent_fd = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    sealed = False
    try:
        if identity(os.fstat(parent_fd)) != identity(parent_info):
            raise StaticControllerError("held static output parent differs")
        descriptor = os.open(
            path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600, dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise StaticControllerError("static output write made no progress")
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
            raise StaticControllerError("static output staging differs")
        os.fchmod(descriptor, RECEIPT_MODE)
        sealed = True
        os.fsync(descriptor)
        os.fsync(parent_fd)
        after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            identity(after) != identity(named_after)
            or stat.S_IMODE(after.st_mode) != RECEIPT_MODE
            or read_fd(descriptor, after.st_size) != raw
        ):
            raise StaticControllerError("static output seal differs")
        return raw
    except BaseException:
        # A pre-seal partial file is not evidence and cannot be mistaken for a
        # receipt.  Once 0400 is reached it is an immutable terminal requiring
        # audit; it is never demoted, unlinked, or retried.
        if descriptor >= 0 and not sealed:
            try:
                opened = os.fstat(descriptor)
                named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                if identity(opened) == identity(named):
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def postflight_output(expected: Mapping[str, Any], expected_raw: bytes) -> HeldAuthority:
    held = open_authority(
        STATIC_OUTPUT_PATH,
        expected_sha256=hashlib.sha256(expected_raw).hexdigest(),
        expected_size=len(expected_raw), expected_mode=RECEIPT_MODE,
        expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
        maximum_size=MAX_JSON_SIZE,
    )
    try:
        value = strict_json(held.raw, label="sealed static output")
        if value != dict(expected):
            raise StaticControllerError("sealed static output bytes differ")
        validate_static_result(
            value, input_sha256=str(expected.get("input_sha256")),
        )
        held.replay()
        return held
    except BaseException:
        held.close()
        raise


def controller() -> dict[str, Any]:
    authorities: list[HeldAuthority] = []
    package_root: HeldDirectory | None = None
    output: HeldAuthority | None = None
    try:
        # Receipt-first is literal: this is the first named package authority.
        publication_authority = open_authority(
            PACKAGE_PUBLICATION_RECEIPT_PATH,
            expected_sha256=PACKAGE_PUBLICATION_RECEIPT_SHA256,
            expected_size=PACKAGE_PUBLICATION_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(publication_authority)
        publication = validate_publication_receipt(publication_authority)

        # Only a final admission receipt may authorize the first package-root
        # observation, and the observed directory must match its literal pin.
        package_root = open_package_root(PACKAGE_ROOT_IDENTITY)
        report_authority = open_authority(
            MATERIALIZATION_REPORT_PATH,
            expected_sha256=MATERIALIZATION_REPORT_SHA256,
            expected_size=MATERIALIZATION_REPORT_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(report_authority)
        report = validate_materialization_report(report_authority, publication)

        runtime = open_runtime_authority(); authorities.append(runtime)
        self_authority = open_self_authority(); authorities.append(self_authority)
        probe_authority = open_authority(
            STATIC_PROBE_PATH, expected_sha256=STATIC_PROBE_SHA256,
            expected_size=STATIC_PROBE_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_SOURCE_SIZE,
        )
        authorities.append(probe_authority)
        launcher_authority = open_authority(
            LAUNCHER_PATH, expected_sha256=LAUNCHER_SHA256,
            expected_size=LAUNCHER_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_SOURCE_SIZE,
        )
        authorities.append(launcher_authority)
        launch_input_row = report["launch"]["input"]
        input_authority = open_authority(
            LAUNCH_INPUT_PATH, expected_sha256=launch_input_row["sha256"],
            expected_size=launch_input_row["size"], expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(input_authority)
        launcher_module = load_launcher(launcher_authority.raw)
        launch_input_value = strict_json(
            input_authority.raw, label="held package launch input",
        )
        validate_launch_identity_closure(
            report, launch_input_value, launcher_module,
        )
        probe_module = load_static_probe(probe_authority.raw)

        package_root.replay()
        for authority in authorities:
            authority.replay()
        require_fresh_output()

        # There is exactly one call site and no retry loop or subprocess.
        try:
            probed = probe_module.probe(
                str(LAUNCHER_PATH), LAUNCHER_SHA256, str(LAUNCH_INPUT_PATH),
            )
        except Exception as error:
            raise StaticControllerError(
                "pinned static probe refused; zero output and no retry"
            ) from error

        package_root.replay()
        for authority in authorities:
            authority.replay()
        result = validate_static_result(
            probed, input_sha256=hashlib.sha256(input_authority.raw).hexdigest(),
        )
        require_fresh_output()
        output_raw = create_immutable_receipt(STATIC_OUTPUT_PATH, result)
        output = postflight_output(result, output_raw)

        package_root.replay()
        for authority in authorities:
            authority.replay()
        output.replay()
        return result
    finally:
        if output is not None:
            output.close()
        for authority in reversed(authorities):
            try:
                authority.close()
            except OSError:
                pass
        if package_root is not None:
            try:
                package_root.close()
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    # The state gate precedes argv iteration and every explicit path/open/stat,
    # directory, output, probe, process, network, or subprocess action.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: static admission awaits final package publication/report/"
            "root pins and a reviewed state-only READY copy",
            file=sys.stderr,
        )
        return 88
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        blocked = blocked_dynamic_pins()
        if blocked:
            raise StaticControllerError(
                "HOLD: dynamic package pins are blocked: " + ",".join(blocked)
            )
        if values != ["--execute", authorization_token()]:
            raise StaticControllerError("static controller argv/token differs")
        result = controller()
        print(canonical(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, KeyError, StaticControllerError) as error:
        print(f"static admission controller refused: {error}", file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96


if __name__ == "__main__":
    raise SystemExit(main())
