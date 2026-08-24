#!/usr/bin/env python3
"""Receipt-first one-shot captured-root admission controller (HOLD).

The future state-only READY copy consumes the final package-publication,
materialization, and static-admission receipts before it holds the package
root.  It then replays the exact25 production identities and the independent
captured runner, performs exactly one isolated in-process captured-root entry,
and publishes one canonical self-digested 0400 receipt.

This checked-in HOLD source performs no argv, path, directory, process, or
network I/O.  The execution body contains no subprocess transport: the pinned
captured runner is compiled with a refusing ``subprocess`` shim and its real
``captured_main`` ABI is invoked once with output creation captured in memory.
Only a fully validated result is admitted to the final create-only namespace.
"""

from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-root-fake-controller-v1"
CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_IN_PROCESS_CAPTURED_ROOT"
READY_STATE = "READY_EXPLICIT_SINGLE_IN_PROCESS_CAPTURED_ROOT"

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
STATIC_RECEIPT_PATH = (
    PACKAGE_ROOT / "evidence/exact5_static_probe_receipt_v1.json"
)
ROOT_FAKE_OUTPUT_PATH = (
    PACKAGE_ROOT / "evidence/exact5_root_fake_runner_probe_receipt_v1.json"
)
ROOT_FAKE_RUNNER_PATH = PACKAGE_ROOT / (
    "diagnostics/case01_object_trajectory_exact5_root_fake_runner_v1.py"
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

REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
CONTROLLER_MODE = 0o444
MAX_JSON_SIZE = 32 * 1024 * 1024
MAX_SOURCE_SIZE = 2 * 1024 * 1024
MAX_IDENTITY_SIZE = 512 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

LAUNCHER_SHA256 = (
    "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f"
)
LAUNCHER_SIZE = 27_492
ROOT_FAKE_RUNNER_SHA256 = (
    "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872"
)
ROOT_FAKE_RUNNER_SIZE = 21_596
PRODUCTION_RUNNER_SHA256 = (
    "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c"
)
PRODUCTION_RUNNER_SIZE = 21_188

PACKAGE_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v2-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v1"
)
STATIC_SCHEMA = "case01-object-trajectory-exact5-static-admission-v1"
ROOT_FAKE_SCHEMA = "case01-object-trajectory-exact5-root-fake-admission-v4"
ROOT_SPEC_SCHEMA = (
    "case01-object-trajectory-exact5-root-bootstrap-diagnostic-v3"
)
ROOT_ENTRY_SCHEMA = "case01-object-trajectory-exact5-captured-root-entry-v3"
LAUNCH_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-receipt-auh-v1"
)
LAUNCH_RELEASE_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-release-auh-v1"
)
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"
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
IDENTITY_ROLES = (
    "runner", "legacy_exact5_runner", "object_eval", "legacy_exact5_eval",
    "frozen_runner", "bridge", "adapter", "legacy_infer_alias",
    "trajectory_projection", "trajectory_scaffold_module", "base_adapter",
    "eval_v1", "eval_v2", "model_authority", "torchrun_source",
    "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
    "base_model_manifest", "r64_checkpoint_manifest", "python", "ffmpeg",
    "ffprobe", "plan",
)
EXTERNAL_KEYS = {
    "stage0_masks", "g0_mouth_track", "trajectory_scaffold",
    "aux_bone_removed_source",
}
ROOT_MARKER = "CASE01_OBJECT_TRAJECTORY_ROOT_FAKE_PASS "

# Final immutable AUH authorities injected only after the independently
# audited generic HOLD was frozen.  The READY copy differs from this pinned
# HOLD by the controller-state line only.
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
STATIC_RECEIPT_SHA256 = (
    "3e65f4342f33a0d4264fa7f09759bad3aa2f4c4622a6965db675f2c551fb07b8"
)
STATIC_RECEIPT_SIZE: int | str = 1_035
STATIC_RECEIPT_DIGEST = (
    "7ed16825624ca99dc7f2cbbea3c9a5a991122108aff4867796a3ac01456ab6be"
)
PACKAGE_ROOT_IDENTITY: list[int] | str = (
    [
        48, 12_038_280_342_419_913_116, 2012, 2000, 16_832, 2, 0, 4096,
        0, 1_787_357_728_317_453_482, 1_787_357_728_652_385_810,
    ]
)


class RootFakeControllerError(RuntimeError):
    """The reviewed one-shot captured-root contract differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RootFakeControllerError("value is not canonical JSON") from error


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
                raise RootFakeControllerError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RootFakeControllerError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise RootFakeControllerError(f"noncanonical JSON authority: {label}")
    return value


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise RootFakeControllerError("held read size differs")
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
        raise RootFakeControllerError("held read is incomplete")
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
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or read_fd(self.descriptor, opened.st_size) != self.raw
        ):
            raise RootFakeControllerError(f"held authority changed: {self.path}")

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
        if (
            identity(os.fstat(self.descriptor)) != self.held_identity
            or identity(os.lstat(self.path)) != self.held_identity
        ):
            raise RootFakeControllerError("held package root changed")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def open_authority(
    path: Path, *, expected_sha256: str | None, expected_size: int | None,
    expected_mode: int | None, expected_uid: int | None,
    expected_gid: int | None, executable: bool = False,
    maximum_size: int = MAX_JSON_SIZE,
) -> HeldAuthority:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise RootFakeControllerError(f"noncanonical authority path: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise RootFakeControllerError(f"missing authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or (expected_mode is not None
            and stat.S_IMODE(named.st_mode) != expected_mode)
        or (expected_uid is not None and named.st_uid != expected_uid)
        or (expected_gid is not None and named.st_gid != expected_gid)
        or named.st_size <= 0 or named.st_size > maximum_size
        or (expected_size is not None and named.st_size != expected_size)
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise RootFakeControllerError(f"named authority differs: {path}")
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
            raise RootFakeControllerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), first)
    except BaseException:
        os.close(descriptor)
        raise


def open_package_root(expected_identity: Sequence[int]) -> HeldDirectory:
    if (
        type(expected_identity) is not list or len(expected_identity) != 11
        or any(type(value) is not int for value in expected_identity)
    ):
        raise RootFakeControllerError("package root identity pin differs")
    named = os.lstat(PACKAGE_ROOT)
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_uid != REMOTE_UID or named.st_gid != REMOTE_GID
        or stat.S_IMODE(named.st_mode) & 0o022
        or identity(named) != tuple(expected_identity)
        or PACKAGE_ROOT.resolve(strict=True) != PACKAGE_ROOT
    ):
        raise RootFakeControllerError("named package root differs")
    descriptor = os.open(
        PACKAGE_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    if (
        identity(os.fstat(descriptor)) != identity(named)
        or identity(os.lstat(PACKAGE_ROOT)) != identity(named)
    ):
        os.close(descriptor)
        raise RootFakeControllerError("opened package root differs")
    return HeldDirectory(PACKAGE_ROOT, descriptor, identity(named))


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
        "static_receipt_sha256": STATIC_RECEIPT_SHA256,
        "static_receipt_size": STATIC_RECEIPT_SIZE,
        "static_receipt_digest": STATIC_RECEIPT_DIGEST,
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
        "publication_receipt": str(PACKAGE_PUBLICATION_RECEIPT_PATH),
        "materialization_report": str(MATERIALIZATION_REPORT_PATH),
        "static_receipt": str(STATIC_RECEIPT_PATH),
        "launcher": {
            "path": str(LAUNCHER_PATH), "sha256": LAUNCHER_SHA256,
            "size": LAUNCHER_SIZE,
        },
        "captured_runner": {
            "path": str(ROOT_FAKE_RUNNER_PATH),
            "sha256": ROOT_FAKE_RUNNER_SHA256,
            "size": ROOT_FAKE_RUNNER_SIZE,
        },
        "launch_input": str(LAUNCH_INPUT_PATH),
        "plan": str(PLAN_PATH),
        "output": str(ROOT_FAKE_OUTPUT_PATH),
        "dynamic_pins": dynamic_pin_values(),
        "captured_main_calls": 1,
        "retry_allowed": False,
        "subprocess_allowed": False,
        "network_allowed": False,
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
        raise RootFakeControllerError("package publication receipt differs")
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


def artifact_row(value: Mapping[str, Any], relative: str) -> dict[str, Any]:
    artifacts = value.get("artifacts")
    row = artifacts.get(relative) if type(artifacts) is dict else None
    if (
        type(row) is not dict or set(row) != {"sha256", "size"}
        or SHA_RE.fullmatch(str(row.get("sha256"))) is None
        or type(row.get("size")) is not int or row["size"] <= 0
    ):
        raise RootFakeControllerError(
            f"materialization artifact row differs: {relative}"
        )
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
        or identity_roles != list(IDENTITY_ROLES)
        or type(identities) is not dict
        or set(identities) != set(IDENTITY_ROLES)
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
        or len({row["path"] for row in identities.values()}) != 25
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
        or identities.get("plan", {}).get("path") != plan.get("path")
        or identities.get("plan", {}).get("sha256") != plan.get("sha256")
    ):
        raise RootFakeControllerError("package materialization report differs")
    expected_artifacts = {
        "diagnostics/case01_object_trajectory_exact5_root_fake_runner_v1.py": {
            "sha256": ROOT_FAKE_RUNNER_SHA256, "size": ROOT_FAKE_RUNNER_SIZE,
        },
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py": {
            "sha256": LAUNCHER_SHA256, "size": LAUNCHER_SIZE,
        },
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_runner_v1.py": {
            "sha256": PRODUCTION_RUNNER_SHA256, "size": PRODUCTION_RUNNER_SIZE,
        },
    }
    for relative, expected in expected_artifacts.items():
        if artifact_row(value, relative) != expected:
            raise RootFakeControllerError(
                f"pinned package artifact differs: {relative}"
            )
    if identities["runner"] != {
        "path": str(
            PACKAGE_ROOT / "release/methods/bernini_action_editing/"
            "case01_object_trajectory_exact5_runner_v1.py"
        ),
        "sha256": PRODUCTION_RUNNER_SHA256,
        "size": PRODUCTION_RUNNER_SIZE,
    }:
        raise RootFakeControllerError("production runner identity differs")
    return value


STATIC_RESULT_FIELDS = {
    "schema_version", "status", "launch_allowed", "blocked_roles",
    "final_source_pins_complete", "exact_identity_count", "task_ids",
    "arm_order", "all_tasks_hard1_every_step",
    "null_arms_have_no_external_conditions",
    "route_and_active_arms_have_external_conditions", "torch_imported",
    "renderer_imported", "publication_performed", "input_sha256",
    "launcher_sha256", "receipt_digest",
}


def validate_static_receipt(
    held: HeldAuthority, report: Mapping[str, Any],
) -> dict[str, Any]:
    value = strict_json(held.raw, label="static admission receipt")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    input_row = report.get("launch", {}).get("input", {})
    if (
        set(value) != STATIC_RESULT_FIELDS
        or hashlib.sha256(held.raw).hexdigest() != STATIC_RECEIPT_SHA256
        or len(held.raw) != STATIC_RECEIPT_SIZE
        or claimed != STATIC_RECEIPT_DIGEST
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != STATIC_SCHEMA
        or value.get("status") != "ADMITTED_STATIC_HOLD_ONLY"
        or value.get("launch_allowed") is not False
        or value.get("blocked_roles") != []
        or value.get("final_source_pins_complete") is not True
        or value.get("exact_identity_count") != 25
        or value.get("task_ids") != list(TASK_IDS)
        or value.get("arm_order") != list(ARM_ORDER)
        or value.get("all_tasks_hard1_every_step") is not True
        or value.get("null_arms_have_no_external_conditions") is not True
        or value.get("route_and_active_arms_have_external_conditions") is not True
        or value.get("torch_imported") is not False
        or value.get("renderer_imported") is not False
        or value.get("publication_performed") is not False
        or value.get("input_sha256") != input_row.get("sha256")
        or value.get("launcher_sha256") != LAUNCHER_SHA256
    ):
        raise RootFakeControllerError("static admission receipt differs")
    return value


def validate_plan_and_crosslinks(
    plan: Mapping[str, Any], identities: Mapping[str, Mapping[str, Any]],
) -> None:
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if (
        plan.get("status") != "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY"
        or plan.get("production_ready") is not False
        or plan.get("launch_allowed") is not False
        or type(plan.get("hold_reasons")) is not list
        or not plan["hold_reasons"]
        or type(tasks) is not list or len(tasks) != 5
        or [row.get("task_id") for row in tasks] != list(TASK_IDS)
        or [row.get("oracle_arm") for row in tasks] != list(ARM_ORDER)
        or any(row.get("source_onset_policy") != "hard1_every_step"
               for row in tasks)
    ):
        raise RootFakeControllerError("five-arm HOLD plan differs")
    for row in tasks:
        external = row.get("external_conditions")
        if row["oracle_arm"] in {"null_before", "null_after"}:
            if external != {}:
                raise RootFakeControllerError(
                    "null arm carries external authority"
                )
        elif type(external) is not dict or set(external) != EXTERNAL_KEYS:
            raise RootFakeControllerError("non-null external closure differs")

    producer = plan.get("producer")
    checkpoint = plan.get("checkpoint_manifest")
    expected = {
        "legacy_infer_alias": (
            "infer_lora_path", "infer_lora_sha256", "infer_lora_size",
        ),
        "adapter": (
            "inference_wrapper_path", "inference_wrapper_sha256",
            "inference_wrapper_size",
        ),
        "trajectory_projection": (
            "trajectory_projection_module_path",
            "trajectory_projection_module_sha256",
            "trajectory_projection_module_size",
        ),
        "trajectory_scaffold_module": (
            "trajectory_scaffold_module_path",
            "trajectory_scaffold_module_sha256",
            "trajectory_scaffold_module_size",
        ),
        "ffprobe": ("ffprobe_path", "ffprobe_sha256", "ffprobe_size"),
    }
    if type(producer) is not dict or type(checkpoint) is not dict:
        raise RootFakeControllerError("plan producer/checkpoint closure differs")
    for role, keys in expected.items():
        if identities.get(role) != {
            "path": producer.get(keys[0]), "sha256": producer.get(keys[1]),
            "size": producer.get(keys[2]),
        }:
            raise RootFakeControllerError(
                f"plan producer identity differs: {role}"
            )
    checkpoint_row = identities.get("r64_checkpoint_manifest")
    if (
        type(checkpoint_row) is not dict
        or checkpoint.get("path") != checkpoint_row.get("path")
        or checkpoint.get("sha256") != checkpoint_row.get("sha256")
        or any(
            type(row.get("adapter")) is not dict
            or row["adapter"].get("checkpoint_manifest") != checkpoint
            for row in tasks
        )
    ):
        raise RootFakeControllerError("plan checkpoint identity differs")


def load_launcher(raw: bytes) -> types.ModuleType:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise RootFakeControllerError("launcher is not UTF-8") from error
    module = types.ModuleType("_held_case01_object_trajectory_root_launcher")
    module.__file__ = str(LAUNCHER_PATH)
    module.__package__ = None
    exec(
        compile(source, str(LAUNCHER_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    if (
        getattr(module, "CAMPAIGN", None) != CAMPAIGN
        or tuple(getattr(module, "IDENTITY_ROLES", ())) != IDENTITY_ROLES
        or tuple(getattr(module, "TASK_IDS", ())) != TASK_IDS
        or tuple(getattr(module, "ARM_ORDER", ())) != ARM_ORDER
        or getattr(module, "EXPECTED_CAPTURED_ROOT_FAKE_SHA256", None)
        != ROOT_FAKE_RUNNER_SHA256
        or getattr(module, "EXPECTED_CAPTURED_ROOT_FAKE_SIZE", None)
        != ROOT_FAKE_RUNNER_SIZE
        or not isinstance(getattr(module, "ROOT_BOOTSTRAP", None), str)
        or not callable(getattr(module, "validate_input", None))
    ):
        raise RootFakeControllerError("pinned launcher root ABI differs")
    try:
        compile(
            module.ROOT_BOOTSTRAP, "<held-root-bootstrap>", "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError, TypeError) as error:
        raise RootFakeControllerError("held root bootstrap does not compile") from error
    return module


def load_root_fake_runner(raw: bytes) -> types.ModuleType:
    """Compile the real captured runner while making process launch impossible."""
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise RootFakeControllerError("captured runner is not UTF-8") from error

    def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        raise RootFakeControllerError("subprocess is forbidden in captured mode")

    refusing_subprocess = types.SimpleNamespace(
        run=forbidden_process,
        PIPE=object(),
        DEVNULL=object(),
        SubprocessError=RuntimeError,
    )
    real_import = builtins.__import__

    def isolated_import(
        name: str, globals_value: Any = None, locals_value: Any = None,
        fromlist: Sequence[str] = (), level: int = 0,
    ) -> Any:
        if name == "subprocess":
            return refusing_subprocess
        return real_import(name, globals_value, locals_value, fromlist, level)

    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = isolated_import
    module = types.ModuleType("_held_case01_object_trajectory_root_fake")
    module.__file__ = str(ROOT_FAKE_RUNNER_PATH)
    module.__package__ = None
    module.__dict__["__builtins__"] = safe_builtins
    exec(
        compile(source, str(ROOT_FAKE_RUNNER_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    if (
        getattr(module, "SCHEMA", None) != ROOT_FAKE_SCHEMA
        or getattr(module, "SPEC_SCHEMA", None) != ROOT_SPEC_SCHEMA
        or getattr(module, "ENTRY_SCHEMA", None) != ROOT_ENTRY_SCHEMA
        or tuple(getattr(module, "IDENTITY_ROLES", ())) != IDENTITY_ROLES
        or tuple(getattr(module, "TASKS", ())) != TASK_IDS
        or tuple(getattr(module, "ARMS", ())) != ARM_ORDER
        or not callable(getattr(module, "captured_main", None))
        or module.subprocess is not refusing_subprocess
    ):
        raise RootFakeControllerError("pinned captured-root ABI differs")
    return module


def open_exact25(
    identities: Mapping[str, Mapping[str, Any]],
) -> list[HeldAuthority]:
    if (
        type(identities) is not dict
        or set(identities) != set(IDENTITY_ROLES)
        or len(identities) != 25
        or len({row.get("path") for row in identities.values()
                if type(row) is dict}) != 25
    ):
        raise RootFakeControllerError("exact25 identity closure differs")
    held: list[HeldAuthority] = []
    try:
        for role in IDENTITY_ROLES:
            row = identities[role]
            if (
                type(row) is not dict
                or set(row) != {"path", "sha256", "size"}
                or type(row.get("path")) is not str
                or not os.path.isabs(row["path"])
                or os.path.normpath(row["path"]) != row["path"]
                or SHA_RE.fullmatch(str(row.get("sha256"))) is None
                or type(row.get("size")) is not int or row["size"] <= 0
            ):
                raise RootFakeControllerError(f"identity row differs: {role}")
            held.append(open_authority(
                Path(row["path"]), expected_sha256=row["sha256"],
                expected_size=row["size"], expected_mode=None,
                expected_uid=None, expected_gid=None,
                executable=role in {"python", "ffmpeg", "ffprobe"},
                maximum_size=MAX_IDENTITY_SIZE,
            ))
        return held
    except BaseException:
        for authority in reversed(held):
            authority.close()
        raise


def validate_running_python(exact25: Sequence[HeldAuthority]) -> None:
    if len(exact25) != len(IDENTITY_ROLES):
        raise RootFakeControllerError("held exact25 count differs")
    python_authority = exact25[IDENTITY_ROLES.index("python")]
    try:
        running = os.stat("/proc/self/exe")
    except OSError as error:
        raise RootFakeControllerError(
            "executing Python identity is unavailable"
        ) from error
    if identity(running) != python_authority.held_identity:
        raise RootFakeControllerError(
            "executing Python differs from exact25 Python"
        )


def build_root_spec_and_entry(
    *, report: Mapping[str, Any], launcher: types.ModuleType,
    launch_input_raw: bytes, plan: Mapping[str, Any],
    captured_authority: HeldAuthority,
) -> tuple[dict[str, Any], dict[str, Any]]:
    launch_input = strict_json(launch_input_raw, label="HOLD launch input")
    release = report["launch"]["release"]
    identities = release["identities"]
    input_row = report["launch"]["input"]
    plan_row = identities["plan"]
    plan_raw = canonical(plan) + b"\n"
    if (
        launch_input.get("identities") != identities
        or hashlib.sha256(launch_input_raw).hexdigest() != input_row["sha256"]
        or len(launch_input_raw) != input_row["size"]
        or hashlib.sha256(plan_raw).hexdigest() != plan_row["sha256"]
        or len(plan_raw) != plan_row["size"]
        or report["plan"].get("sha256") != plan_row["sha256"]
    ):
        raise RootFakeControllerError("launch input/plan crosslink differs")
    validate_plan_and_crosslinks(plan, identities)
    try:
        validated = launcher.validate_input(
            launch_input, reopen=False, plan_override=plan,
        )
    except Exception as error:
        raise RootFakeControllerError(
            "launcher rejected exact25 HOLD input"
        ) from error
    if validated.get("identities") != identities:
        raise RootFakeControllerError("launcher exact25 identities differ")

    captured_runner = {
        "path": str(ROOT_FAKE_RUNNER_PATH),
        "sha256": hashlib.sha256(captured_authority.raw).hexdigest(),
        "size": len(captured_authority.raw),
    }
    if (
        captured_runner != {
            "path": str(ROOT_FAKE_RUNNER_PATH),
            "sha256": ROOT_FAKE_RUNNER_SHA256,
            "size": ROOT_FAKE_RUNNER_SIZE,
        }
        or captured_runner["path"] in {
            row["path"] for row in identities.values()
        }
        or captured_runner == identities["runner"]
        or identities["runner"].get("sha256") != PRODUCTION_RUNNER_SHA256
        or identities["runner"].get("size") != PRODUCTION_RUNNER_SIZE
    ):
        raise RootFakeControllerError(
            "captured and production runner authorities differ"
        )
    spec: dict[str, Any] = {
        "schema_version": ROOT_SPEC_SCHEMA,
        "campaign_mode": CAMPAIGN,
        "launch_allowed": False,
        "identities": identities,
        "captured_runner": captured_runner,
        "launch_input": {
            "path": str(LAUNCH_INPUT_PATH),
            "sha256": input_row["sha256"], "size": input_row["size"],
        },
        "result_path": str(ROOT_FAKE_OUTPUT_PATH),
    }
    entry: dict[str, Any] = {
        "schema_version": ROOT_ENTRY_SCHEMA,
        "release_digest": object_digest(spec),
        "identity_roles": list(IDENTITY_ROLES),
        "identity_set_digest": object_digest(identities),
        "launch_input_sha256": input_row["sha256"],
        "production_runner": identities["runner"],
        "captured_runner": captured_runner,
        "captured_runner_identity": list(captured_authority.held_identity),
        "plan_sha256": plan_row["sha256"],
        "task_ids": list(TASK_IDS),
        "arm_order": list(ARM_ORDER),
        "all_exact25_named_identities_replayed": True,
        "captured_runner_outside_exact25": True,
        "captured_runner_bytes_compiled": True,
        "publication_performed": False,
    }
    entry["authority_digest"] = object_digest(entry)
    return spec, entry


ROOT_RESULT_FIELDS = {
    "schema_version", "status", "campaign_mode", "launch_allowed",
    "exact_identity_count", "identity_roles", "task_ids", "arm_order",
    "release_digest", "identity_set_digest", "launch_input_sha256",
    "entry_authority_digest", "plan_sha256", "production_runner_sha256",
    "captured_runner_sha256", "all_exact25_named_identities_replayed",
    "captured_runner_outside_exact25", "captured_runner_bytes_compiled",
    "torch_imported", "renderer_imported", "publication_performed",
    "receipt_digest",
}


def validate_root_result_intrinsic(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    unsigned = dict(result)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        set(result) != ROOT_RESULT_FIELDS
        or result.get("schema_version") != ROOT_FAKE_SCHEMA
        or result.get("status") != "PASS_CAPTURED_ROOT_FAKE_HOLD"
        or result.get("campaign_mode") != CAMPAIGN
        or result.get("launch_allowed") is not False
        or result.get("exact_identity_count") != 25
        or result.get("identity_roles") != list(IDENTITY_ROLES)
        or result.get("task_ids") != list(TASK_IDS)
        or result.get("arm_order") != list(ARM_ORDER)
        or SHA_RE.fullmatch(str(result.get("release_digest"))) is None
        or SHA_RE.fullmatch(str(result.get("identity_set_digest"))) is None
        or SHA_RE.fullmatch(str(result.get("launch_input_sha256"))) is None
        or SHA_RE.fullmatch(str(result.get("entry_authority_digest"))) is None
        or SHA_RE.fullmatch(str(result.get("plan_sha256"))) is None
        or result.get("production_runner_sha256")
        != PRODUCTION_RUNNER_SHA256
        or result.get("captured_runner_sha256") != ROOT_FAKE_RUNNER_SHA256
        or result.get("all_exact25_named_identities_replayed") is not True
        or result.get("captured_runner_outside_exact25") is not True
        or result.get("captured_runner_bytes_compiled") is not True
        or result.get("torch_imported") is not False
        or result.get("renderer_imported") is not False
        or result.get("publication_performed") is not False
        or claimed != object_digest(unsigned)
    ):
        raise RootFakeControllerError("captured-root result differs")
    return result


def validate_root_result(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
    entry: Mapping[str, Any], marker: str,
) -> dict[str, Any]:
    result = validate_root_result_intrinsic(value)
    claimed = result["receipt_digest"]
    if (
        result.get("release_digest") != object_digest(spec)
        or result.get("identity_set_digest")
        != object_digest(spec["identities"])
        or result.get("launch_input_sha256")
        != spec["launch_input"]["sha256"]
        or result.get("entry_authority_digest")
        != entry.get("authority_digest")
        or result.get("plan_sha256")
        != spec["identities"]["plan"]["sha256"]
        or marker != ROOT_MARKER + str(claimed) + "\n"
    ):
        raise RootFakeControllerError("captured-root result differs")
    return result


def run_isolated_root_fake(
    module: types.ModuleType, *, spec: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Invoke the real captured ``captured_main`` once without final I/O."""
    captured: list[dict[str, Any]] = []

    def capture_create(path: Path, value: Mapping[str, Any]) -> None:
        if path != ROOT_FAKE_OUTPUT_PATH or captured:
            raise RootFakeControllerError("captured create contract differs")
        captured.append(dict(value))

    original_create = module.create
    original_environment = dict(os.environ)
    original_argv = sys.argv
    sentinel = object()
    original_main = sys.modules.get("__main__", sentinel)
    output = io.StringIO()
    try:
        module.create = capture_create
        os.environ.clear()
        os.environ["CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY"] = (
            canonical(entry).decode("utf-8")
        )
        sys.argv = [
            str(ROOT_FAKE_RUNNER_PATH), "--captured-result",
            str(ROOT_FAKE_OUTPUT_PATH),
        ]
        sys.modules["__main__"] = module
        with redirect_stdout(output):
            # Exactly one real-ABI call site; there is no retry loop.
            return_code = module.captured_main(
                ["--captured-result", str(ROOT_FAKE_OUTPUT_PATH)]
            )
    except Exception as error:
        raise RootFakeControllerError(
            "pinned captured-root entry refused; zero output and no retry"
        ) from error
    finally:
        module.create = original_create
        os.environ.clear()
        os.environ.update(original_environment)
        sys.argv = original_argv
        if original_main is sentinel:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original_main
    if return_code != 0 or len(captured) != 1:
        raise RootFakeControllerError(
            "captured-root call count/result differs; zero output and no retry"
        )
    return validate_root_result(
        captured[0], spec=spec, entry=entry, marker=output.getvalue(),
    )


def require_fresh_output() -> None:
    if os.path.lexists(ROOT_FAKE_OUTPUT_PATH):
        raise RootFakeControllerError("single root-fake output is not fresh")
    for path in (
        PACKAGE_ROOT / "outputs/media", PACKAGE_ROOT / "final",
        PACKAGE_ROOT / "runtime",
    ):
        if os.listdir(path):
            raise RootFakeControllerError(
                f"root-fake production path is not fresh: {path}"
            )


def create_immutable_receipt(path: Path, value: Mapping[str, Any]) -> bytes:
    if path != ROOT_FAKE_OUTPUT_PATH or path.parent != PACKAGE_ROOT / "evidence":
        raise RootFakeControllerError("root-fake output target path differs")
    raw = canonical(value) + b"\n"
    # Prospective self-digest validation precedes the first namespace mutation.
    if (
        strict_json(raw, label="prospective root-fake output") != dict(value)
    ):
        raise RootFakeControllerError("prospective root-fake receipt differs")
    try:
        validate_root_result_intrinsic(value)
    except RootFakeControllerError as error:
        raise RootFakeControllerError(
            "prospective root-fake receipt differs"
        ) from error
    parent_info = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != REMOTE_UID or parent_info.st_gid != REMOTE_GID
        or stat.S_IMODE(parent_info.st_mode) & 0o022
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise RootFakeControllerError("root-fake output parent differs")
    parent_fd = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    sealed = False
    try:
        if identity(os.fstat(parent_fd)) != identity(parent_info):
            raise RootFakeControllerError("held root-fake output parent differs")
        descriptor = os.open(
            path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600, dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise RootFakeControllerError(
                    "root-fake output write made no progress"
                )
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
            raise RootFakeControllerError("root-fake output staging differs")
        os.fchmod(descriptor, RECEIPT_MODE)
        sealed = True
        os.fsync(descriptor)
        os.fsync(parent_fd)
        after = os.fstat(descriptor)
        named_after = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            identity(after) != identity(named_after)
            or stat.S_IMODE(after.st_mode) != RECEIPT_MODE
            or read_fd(descriptor, after.st_size) != raw
        ):
            raise RootFakeControllerError("root-fake output seal differs")
        return raw
    except BaseException:
        # A pre-seal partial is not evidence.  A sealed 0400 inode is terminal
        # and is never demoted, unlinked, or retried.
        if descriptor >= 0 and not sealed:
            try:
                opened = os.fstat(descriptor)
                named = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False,
                )
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


def postflight_output(
    expected: Mapping[str, Any], expected_raw: bytes,
) -> HeldAuthority:
    held = open_authority(
        ROOT_FAKE_OUTPUT_PATH,
        expected_sha256=hashlib.sha256(expected_raw).hexdigest(),
        expected_size=len(expected_raw), expected_mode=RECEIPT_MODE,
        expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
        maximum_size=MAX_JSON_SIZE,
    )
    try:
        value = strict_json(held.raw, label="sealed root-fake output")
        if value != dict(expected):
            raise RootFakeControllerError("sealed root-fake output differs")
        validate_root_result_intrinsic(value)
        held.replay()
        return held
    except BaseException:
        held.close()
        raise


def replay_full_chain(
    authorities: Sequence[HeldAuthority], package_root: HeldDirectory,
    exact25: Sequence[HeldAuthority],
) -> None:
    # The first three slots are the publication, internal materialization, and
    # static receipts.  Their replay remains ahead of every package-root replay.
    if len(authorities) != 7 or len(exact25) != 25:
        raise RootFakeControllerError("held root-fake authority count differs")
    for authority in authorities[:3]:
        authority.replay()
    package_root.replay()
    for authority in tuple(authorities[3:]) + tuple(exact25):
        authority.replay()


def controller() -> dict[str, Any]:
    authorities: list[HeldAuthority] = []
    exact25: list[HeldAuthority] = []
    package_root: HeldDirectory | None = None
    output: HeldAuthority | None = None
    try:
        # Receipt-first order is literal and intentionally precedes the root FD.
        publication_authority = open_authority(
            PACKAGE_PUBLICATION_RECEIPT_PATH,
            expected_sha256=PACKAGE_PUBLICATION_RECEIPT_SHA256,
            expected_size=PACKAGE_PUBLICATION_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(publication_authority)
        publication = validate_publication_receipt(publication_authority)

        report_authority = open_authority(
            MATERIALIZATION_REPORT_PATH,
            expected_sha256=MATERIALIZATION_REPORT_SHA256,
            expected_size=MATERIALIZATION_REPORT_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(report_authority)
        report = validate_materialization_report(report_authority, publication)

        static_authority = open_authority(
            STATIC_RECEIPT_PATH, expected_sha256=STATIC_RECEIPT_SHA256,
            expected_size=STATIC_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(static_authority)
        validate_static_receipt(static_authority, report)

        package_root = open_package_root(PACKAGE_ROOT_IDENTITY)
        launcher_authority = open_authority(
            LAUNCHER_PATH, expected_sha256=LAUNCHER_SHA256,
            expected_size=LAUNCHER_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_SOURCE_SIZE,
        )
        authorities.append(launcher_authority)
        input_row = report["launch"]["input"]
        input_authority = open_authority(
            LAUNCH_INPUT_PATH, expected_sha256=input_row["sha256"],
            expected_size=input_row["size"], expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(input_authority)
        captured_authority = open_authority(
            ROOT_FAKE_RUNNER_PATH, expected_sha256=ROOT_FAKE_RUNNER_SHA256,
            expected_size=ROOT_FAKE_RUNNER_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_SOURCE_SIZE,
        )
        authorities.append(captured_authority)
        plan_row = report["launch"]["release"]["identities"]["plan"]
        plan_authority = open_authority(
            PLAN_PATH, expected_sha256=plan_row["sha256"],
            expected_size=plan_row["size"], expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(plan_authority)

        launcher = load_launcher(launcher_authority.raw)
        root_fake = load_root_fake_runner(captured_authority.raw)
        plan = strict_json(plan_authority.raw, label="HOLD five-arm plan")
        spec, entry = build_root_spec_and_entry(
            report=report, launcher=launcher,
            launch_input_raw=input_authority.raw, plan=plan,
            captured_authority=captured_authority,
        )
        exact25 = open_exact25(spec["identities"])
        validate_running_python(exact25)

        replay_full_chain(authorities, package_root, exact25)
        require_fresh_output()

        try:
            result = run_isolated_root_fake(
                root_fake, spec=spec, entry=entry,
            )
        except Exception as error:
            raise RootFakeControllerError(
                "pinned captured-root refused; zero output and no retry"
            ) from error

        replay_full_chain(authorities, package_root, exact25)
        require_fresh_output()
        output_raw = create_immutable_receipt(ROOT_FAKE_OUTPUT_PATH, result)
        output = postflight_output(result, output_raw)

        replay_full_chain(authorities, package_root, exact25)
        output.replay()
        return result
    finally:
        if output is not None:
            output.close()
        for authority in reversed(exact25):
            try:
                authority.close()
            except OSError:
                pass
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
    # This state comparison is intentionally the first executable statement.
    # It precedes argv iteration and every explicit path/open/stat/directory,
    # output, process, network, controller, and dynamic-pin action.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: root-fake admission awaits final package publication/"
            "materialization/static/root pins and a reviewed state-only READY",
            file=sys.stderr,
        )
        return 88
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        blocked = blocked_dynamic_pins()
        if blocked:
            raise RootFakeControllerError(
                "HOLD: dynamic root-fake pins are blocked: "
                + ",".join(blocked)
            )
        if values != ["--execute", authorization_token()]:
            raise RootFakeControllerError("root-fake controller argv/token differs")
        result = controller()
        print(canonical(result).decode("utf-8"))
        return 0
    except Exception as error:
        print(f"root-fake admission controller refused: {error}", file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96


if __name__ == "__main__":
    raise SystemExit(main())
