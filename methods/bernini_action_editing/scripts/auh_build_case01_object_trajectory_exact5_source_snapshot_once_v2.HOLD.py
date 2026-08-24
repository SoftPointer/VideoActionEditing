#!/usr/bin/env python3
"""One-shot receipt-gated exact35 snapshot controller (checked-in HOLD).

The future READY copy is allowed to run only after the physical15 AUHv2
receipt and the CPU world4 admission receipts have been produced and pinned.
It opens those immutable receipts before any physical15 named source, retains
the exact Python/controller/builder authorities, durably claims one attempt,
and calls the pinned builder exactly once in-process.  The builder owns the
NFS-truthful receipt-reserved shadow publication; this controller independently
replays the final physical35 tree and its sibling publication receipt.

This checked-in source performs no I/O while HOLD.  It contains no SSH, Slurm,
renderer, or launch path.
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


SCHEMA = "case01-object-trajectory-exact5-source-snapshot-controller-v1"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
EVIDENCE_SCHEMA = SCHEMA + "-evidence"
CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"
READY_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_EXACT35_SNAPSHOT"

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
SOURCE_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_source_staging_v1"
)
SOURCE_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_staging_v1.receipt_v1.json"
)
OLD_ROOT = EXPERIMENTS / (
    "bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
)
TARGET_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1"
)
TARGET_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "receipt_v2.json"
)
ATTEMPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "build_attempt_v1.json"
)
EVIDENCE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "build_controller_evidence_v1.json"
)
MANIFEST_NAME = "case01_object_trajectory_exact5_source_snapshot_manifest_v2.json"
BUILDER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "build_case01_object_trajectory_exact5_source_snapshot_v1.py"
)
BUILDER_PATH = SOURCE_ROOT / BUILDER_RELATIVE
BUILDER_SHA256 = (
    "8ece3b3310b4065ceb8b7b8331f61d0ab6897f35e25febabd0f705f202a31432"
)
BUILDER_SIZE = 66_981

VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
VACE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
VACE_PYTHON_SIZE = 31_490_256
REMOTE_UID = 2012
REMOTE_GID = 2000

CPU_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_world4_cpu_admission_v1"
)
CPU_WORLD4_RECEIPT_PATH = CPU_ROOT / "evidence/world4_receipt_v1.json"
CPU_CONTROLLER_EVIDENCE_PATH = CPU_ROOT / "evidence/controller_evidence_v1.json"
CPU_WORLD4_SCHEMA = "case01-object-trajectory-exact5-world4-admission-v5"
CPU_CONTROLLER_SCHEMA = "case01-object-trajectory-exact5-world4-cpu-auh-controller-v1"
CPU_JOB_ID = "143808"
CPU_NODE = "auh7-1b-gpu-292"

SOURCE_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-source-stager-auh-v2-receipt"
)
SOURCE_PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
SOURCE_BOOTSTRAP_SHA256 = (
    "33c63bb114d6008bd32c67819cd86fb4acce7b796696c7ed34f41a431836e08a"
)
SNAPSHOT_MANIFEST_SCHEMA = "case01-object-trajectory-exact5-source-snapshot-v2"
SNAPSHOT_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-source-snapshot-publication-v2-receipt"
)
STAGING_RECEIPT_COPY_RELATIVE = "authority/source_staging_receipt_auh_v2.json"

# Exact final remote authorities produced by the admitted CPU run.  This HOLD
# is reviewed with those immutable coordinates before its state-only READY copy.
SOURCE_RECEIPT_SHA256 = (
    "d91b18336ab56c72f95891da842e8ae57261f68c9a340b0bafbf9f0beeca8c5f"
)
SOURCE_RECEIPT_SIZE = 5_347
SOURCE_RECEIPT_DIGEST = (
    "b13fc3ba5e9f61bfd244492da66570bf91db8d4fba373ccab8522ab256429091"
)
CPU_WORLD4_RECEIPT_SHA256 = (
    "61d72a7e37fc197fdab24f7173e74b289ee53e92379f5089ab89d5cdfb348083"
)
CPU_WORLD4_RECEIPT_SIZE = 49_335
CPU_WORLD4_RECEIPT_DIGEST = (
    "bcf618ad9eeafeebf6dcbc794a9d4bf5fbd27fa13274ec5264d4672a9944ad28"
)
CPU_CONTROLLER_EVIDENCE_SHA256 = (
    "0e138b349688028ad7bed82602e01e1b441e190857e87937d2d96cbb556879a2"
)
CPU_CONTROLLER_EVIDENCE_SIZE = 12_395
CPU_CONTROLLER_EVIDENCE_DIGEST = (
    "b7195233777db70fa5ad068f0e88de7828cf62826bb80f0c950b9b01366209ee"
)

SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
DIRECTORY_MODE = 0o555
CONTROLLER_MODE = 0o444
MAX_JSON_SIZE = 16 * 1024 * 1024
MAX_SOURCE_SIZE = 2 * 1024 * 1024


class SnapshotControllerError(RuntimeError):
    """The reviewed one-shot snapshot controller contract differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SnapshotControllerError("value is not canonical JSON") from error


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
                raise SnapshotControllerError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SnapshotControllerError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise SnapshotControllerError(f"noncanonical JSON authority: {label}")
    return value


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise SnapshotControllerError("held read size differs")
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
        raise SnapshotControllerError("held read is incomplete")
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
            "path": str(self.path),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw),
            "identity": list(self.held_identity),
        }

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        replay = read_fd(self.descriptor, opened.st_size)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or replay != self.raw
        ):
            raise SnapshotControllerError(f"held authority changed: {self.path}")

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
        raise SnapshotControllerError(f"noncanonical authority path: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise SnapshotControllerError(f"missing authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != expected_mode
        or named.st_uid != expected_uid or named.st_gid != expected_gid
        or named.st_size <= 0 or named.st_size > maximum_size
        or (expected_size is not None and named.st_size != expected_size)
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise SnapshotControllerError(f"named authority differs: {path}")
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
        observed_sha256 = hashlib.sha256(first).hexdigest()
        if (
            identity(before) != identity(named)
            or identity(before) != identity(middle)
            or identity(before) != identity(after)
            or identity(before) != identity(named_after)
            or first != second
            or (expected_sha256 is not None
                and observed_sha256 != expected_sha256)
            or (expected_size is not None and len(first) != expected_size)
        ):
            raise SnapshotControllerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), first)
    except BaseException:
        os.close(descriptor)
        raise


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_receipt_size": SOURCE_RECEIPT_SIZE,
        "source_receipt_digest": SOURCE_RECEIPT_DIGEST,
        "cpu_world4_receipt_sha256": CPU_WORLD4_RECEIPT_SHA256,
        "cpu_world4_receipt_size": CPU_WORLD4_RECEIPT_SIZE,
        "cpu_world4_receipt_digest": CPU_WORLD4_RECEIPT_DIGEST,
        "cpu_controller_evidence_sha256": CPU_CONTROLLER_EVIDENCE_SHA256,
        "cpu_controller_evidence_size": CPU_CONTROLLER_EVIDENCE_SIZE,
        "cpu_controller_evidence_digest": CPU_CONTROLLER_EVIDENCE_DIGEST,
    }


def blocked_dynamic_pins() -> tuple[str, ...]:
    blocked: list[str] = []
    for name, value in dynamic_pin_values().items():
        if name.endswith("_size"):
            if type(value) is not int or value <= 0:
                blocked.append(name)
        elif type(value) is not str or SHA_RE.fullmatch(value) is None:
            blocked.append(name)
    return tuple(blocked)


def authorization_token() -> str:
    value = {
        "schema_version": SCHEMA + "-authorization-v1",
        "state": READY_STATE,
        "source_root": str(SOURCE_ROOT),
        "source_receipt_path": str(SOURCE_RECEIPT_PATH),
        "old_root": str(OLD_ROOT),
        "target_root": str(TARGET_ROOT),
        "target_receipt_path": str(TARGET_RECEIPT_PATH),
        "attempt_path": str(ATTEMPT_PATH),
        "evidence_path": str(EVIDENCE_PATH),
        "builder": {
            "path": str(BUILDER_PATH), "sha256": BUILDER_SHA256,
            "size": BUILDER_SIZE,
        },
        "python": {
            "path": str(VACE_PYTHON), "sha256": VACE_PYTHON_SHA256,
            "size": VACE_PYTHON_SIZE,
        },
        "cpu_world4_receipt_path": str(CPU_WORLD4_RECEIPT_PATH),
        "cpu_controller_evidence_path": str(CPU_CONTROLLER_EVIDENCE_PATH),
        "dynamic_pins": dynamic_pin_values(),
        "single_attempt": True,
        "retry_allowed": False,
        "launch_allowed": False,
    }
    return object_digest(value)


SOURCE_RECEIPT_FIELDS = {
    "schema_version", "status", "operation", "target_root", "receipt_path",
    "manifest_digest", "request_payload_sha256", "stage_payload_sha256",
    "bootstrap_source_sha256", "file_count", "files", "directories",
    "file_mode", "directory_mode", "receipt_mode",
    "held_parent_identity_replayed", "ancestor_chain_nofollow",
    "publication_protocol", "rename_noreplace", "cooperative_writer_exclusion",
    "receipt_is_consumption_gate", "receipt_is_admission",
    "uncooperative_same_uid_race_out_of_scope", "target_observation",
    "commit_terminal_digest", "receipt_inode_anchor", "launch_allowed",
    "receipt_digest",
}


def validate_source_receipt_prefix(
    held: HeldAuthority,
) -> dict[str, Any]:
    value = strict_json(held.raw, label="physical15 AUHv2 receipt")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    operation = value.get("operation")
    expected_status = {
        "stage": "STAGED_RECEIPT_GATED",
        "recover-receipt": "RECOVERED_RECEIPT_ONLY",
    }.get(operation)
    rows = value.get("files")
    row_paths = (
        [row.get("relative") for row in rows]
        if type(rows) is list and all(type(row) is dict for row in rows)
        else []
    )
    builder_rows = (
        [row for row in rows if row.get("relative") == BUILDER_RELATIVE]
        if type(rows) is list else []
    )
    if (
        set(value) != SOURCE_RECEIPT_FIELDS
        or value.get("schema_version") != SOURCE_RECEIPT_SCHEMA
        or expected_status is None or value.get("status") != expected_status
        or claimed != SOURCE_RECEIPT_DIGEST
        or claimed != object_digest(unsigned)
        or hashlib.sha256(held.raw).hexdigest() != SOURCE_RECEIPT_SHA256
        or len(held.raw) != SOURCE_RECEIPT_SIZE
        or value.get("target_root") != str(SOURCE_ROOT)
        or value.get("receipt_path") != str(SOURCE_RECEIPT_PATH)
        or value.get("bootstrap_source_sha256") != SOURCE_BOOTSTRAP_SHA256
        or value.get("file_count") != 15
        or type(rows) is not list or len(rows) != 15
        or row_paths != sorted(row_paths) or len(set(row_paths)) != 15
        or any(
            set(row) != {"relative", "sha256", "size", "mode", "nlink"}
            or type(row.get("relative")) is not str
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            or row.get("mode") != FILE_MODE or row.get("nlink") != 1
            for row in rows
        )
        or builder_rows != [{
            "relative": BUILDER_RELATIVE, "sha256": BUILDER_SHA256,
            "size": BUILDER_SIZE, "mode": FILE_MODE, "nlink": 1,
        }]
        or value.get("file_mode") != FILE_MODE
        or value.get("directory_mode") != DIRECTORY_MODE
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("held_parent_identity_replayed") is not True
        or value.get("ancestor_chain_nofollow") is not True
        or value.get("publication_protocol") != SOURCE_PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("launch_allowed") is not False
        or value.get("receipt_inode_anchor")
        != inode_anchor(os.fstat(held.descriptor))
    ):
        raise SnapshotControllerError("physical15 AUHv2 receipt prefix differs")
    return value


CPU_WORLD4_FIELDS = {
    "schema_version", "status", "launch_allowed", "scenario_order",
    "scenarios", "runtime_identities", "runtime_identity_digest",
    "expected_runtime_versions", "expected_gpu_contract",
    "cpu_thread_contract", "active_row_counts_admitted",
    "happy_scheduler_steps", "real_torchrun_process_count_per_scenario",
    "controller_python_optimize_level", "timeout_seconds_per_scenario",
    "timeout_cleanup_policy", "publication_performed",
    "renderer_or_vae_loaded", "scope", "receipt_digest",
}
CPU_EVIDENCE_FIELDS = {
    "schema_version", "status", "holder_job_id", "node",
    "single_srun_attempt", "retry_allowed", "srun_returncode",
    "attempt_sha256", "attempt_digest", "compute_digest",
    "world4_receipt_sha256", "world4_receipt_digest", "stdout_sha256",
    "stderr_sha256", "stderr_empty", "process_group_zero",
    "publication_empty", "stage_cache_absent", "torch_visible_gpu_count",
    "torch_num_threads", "torch_num_interop_threads", "cpu_thread_environment",
    "requested_srun_gpu_export", "gpu_visibility_environment",
    "environment_source", "source_staging_receipt",
    "login_held_project_authorities", "compute_reopened_project_authorities",
    "shared_source_rows_equal", "login_held_runtime_authorities",
    "compute_reopened_runtime_authorities", "runtime_rows_equal",
    "per_scenario_timeout_seconds", "launch_allowed", "renderer_or_vae_loaded",
    "evidence_digest",
}


def validate_cpu_authorities(
    world4: HeldAuthority, evidence: HeldAuthority,
) -> tuple[dict[str, Any], dict[str, Any]]:
    world4_value = strict_json(world4.raw, label="CPU world4 receipt")
    world4_unsigned = dict(world4_value)
    world4_claimed = world4_unsigned.pop("receipt_digest", None)
    if (
        set(world4_value) != CPU_WORLD4_FIELDS
        or hashlib.sha256(world4.raw).hexdigest() != CPU_WORLD4_RECEIPT_SHA256
        or len(world4.raw) != CPU_WORLD4_RECEIPT_SIZE
        or world4_claimed != CPU_WORLD4_RECEIPT_DIGEST
        or world4_claimed != object_digest(world4_unsigned)
        or world4_value.get("schema_version") != CPU_WORLD4_SCHEMA
        or world4_value.get("status")
        != "ADMITTED_WORLD4_TENSOR_ABI_HOLD_ONLY"
        or world4_value.get("launch_allowed") is not False
        or world4_value.get("publication_performed") is not False
        or world4_value.get("renderer_or_vae_loaded") is not False
        or world4_value.get("scope")
        != "distributed_tensor_projection_abi_not_renderer_integration"
        or world4_value.get("timeout_seconds_per_scenario") != 30
    ):
        raise SnapshotControllerError("CPU world4 receipt differs")

    evidence_value = strict_json(evidence.raw, label="CPU controller evidence")
    evidence_unsigned = dict(evidence_value)
    evidence_claimed = evidence_unsigned.pop("evidence_digest", None)
    if (
        set(evidence_value) != CPU_EVIDENCE_FIELDS
        or hashlib.sha256(evidence.raw).hexdigest()
        != CPU_CONTROLLER_EVIDENCE_SHA256
        or len(evidence.raw) != CPU_CONTROLLER_EVIDENCE_SIZE
        or evidence_claimed != CPU_CONTROLLER_EVIDENCE_DIGEST
        or evidence_claimed != object_digest(evidence_unsigned)
        or evidence_value.get("schema_version") != CPU_CONTROLLER_SCHEMA
        or evidence_value.get("status") != "PASS"
        or evidence_value.get("holder_job_id") != CPU_JOB_ID
        or evidence_value.get("node") != CPU_NODE
        or evidence_value.get("single_srun_attempt") is not True
        or evidence_value.get("retry_allowed") is not False
        or evidence_value.get("srun_returncode") != 0
        or evidence_value.get("world4_receipt_sha256")
        != CPU_WORLD4_RECEIPT_SHA256
        or evidence_value.get("world4_receipt_digest")
        != CPU_WORLD4_RECEIPT_DIGEST
        or evidence_value.get("stderr_empty") is not True
        or evidence_value.get("process_group_zero") is not True
        or evidence_value.get("publication_empty") is not True
        or evidence_value.get("stage_cache_absent") is not True
        or evidence_value.get("shared_source_rows_equal") is not True
        or evidence_value.get("runtime_rows_equal") is not True
        or evidence_value.get("launch_allowed") is not False
        or evidence_value.get("renderer_or_vae_loaded") is not False
    ):
        raise SnapshotControllerError("CPU controller evidence differs")
    return world4_value, evidence_value


def validate_cpu_source_crosslink(
    evidence: Mapping[str, Any], source_receipt: Mapping[str, Any],
    held_source_receipt: HeldAuthority,
) -> None:
    row = evidence.get("source_staging_receipt")
    if (
        type(row) is not dict
        or row.get("path") != str(SOURCE_RECEIPT_PATH)
        or row.get("sha256") != SOURCE_RECEIPT_SHA256
        or row.get("size") != SOURCE_RECEIPT_SIZE
        or row.get("receipt_digest") != SOURCE_RECEIPT_DIGEST
        or row.get("manifest_digest") != source_receipt.get("manifest_digest")
        or row.get("file_count") != 15
        or row.get("mode") != RECEIPT_MODE
        or row.get("schema_version") != SOURCE_RECEIPT_SCHEMA
        or hashlib.sha256(held_source_receipt.raw).hexdigest()
        != row.get("sha256")
    ):
        raise SnapshotControllerError("CPU/physical15 receipt crosslink differs")


def load_builder(raw: bytes) -> types.ModuleType:
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise SnapshotControllerError("builder is not UTF-8") from error
    module = types.ModuleType("_held_case01_exact35_builder")
    module.__file__ = str(BUILDER_PATH)
    module.__package__ = None
    exec(
        compile(source, str(BUILDER_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    if (
        getattr(module, "STAGING_ROOT", None) != SOURCE_ROOT
        or getattr(module, "STAGING_RECEIPT_PATH", None) != SOURCE_RECEIPT_PATH
        or getattr(module, "OLD_EXACT5_SNAPSHOT", None) != OLD_ROOT
        or getattr(module, "TARGET_ROOT", None) != TARGET_ROOT
        or getattr(module, "SNAPSHOT_PUBLICATION_RECEIPT_PATH", None)
        != TARGET_RECEIPT_PATH
        or getattr(module, "BUILDER_RELATIVE", None) != BUILDER_RELATIVE
        or not callable(getattr(module, "open_staging_gate", None))
        or not callable(getattr(module, "build", None))
    ):
        raise SnapshotControllerError("pinned builder configuration differs")
    return module


def open_self_authority() -> HeldAuthority:
    path = Path(__file__)
    return open_authority(
        path, expected_sha256=None, expected_size=None,
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
        process = os.stat("/proc/self/exe")
        if identity(process) != held.held_identity:
            raise SnapshotControllerError("executing Python differs from held runtime")
        return held
    except BaseException:
        held.close()
        raise


def require_fresh_outputs() -> None:
    for path in (TARGET_ROOT, TARGET_RECEIPT_PATH, ATTEMPT_PATH, EVIDENCE_PATH):
        if os.path.lexists(path):
            raise SnapshotControllerError(f"single-attempt target is not fresh: {path}")


def create_immutable_json(path: Path, value: Mapping[str, Any]) -> tuple[bytes, list[int]]:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.parent != EXPERIMENTS or path.name in {"", ".", ".."}
    ):
        raise SnapshotControllerError("controller JSON target path differs")
    parent_info = os.lstat(EXPERIMENTS)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or EXPERIMENTS.resolve(strict=True) != EXPERIMENTS
        or parent_info.st_uid != REMOTE_UID or parent_info.st_gid != REMOTE_GID
        or stat.S_IMODE(parent_info.st_mode) & 0o002
    ):
        raise SnapshotControllerError("controller JSON parent differs")
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
                raise SnapshotControllerError("controller JSON write made no progress")
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
            raise SnapshotControllerError("controller JSON staging differs")
        # Once this succeeds the attempt/evidence is immutable.  Every later
        # error remains a terminal HOLD; this function never demotes/unlinks it.
        os.fchmod(descriptor, RECEIPT_MODE)
        os.fsync(descriptor)
        os.fsync(parent_fd)
        after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            identity(after) != identity(named_after)
            or stat.S_IMODE(after.st_mode) != RECEIPT_MODE
            or read_fd(descriptor, after.st_size) != raw
        ):
            raise SnapshotControllerError("controller JSON seal differs")
        return raw, inode_anchor(after)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


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


def validate_snapshot_tree(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = manifest.get("files")
    row_paths = (
        [row.get("path") for row in rows]
        if type(rows) is list and all(type(row) is dict for row in rows)
        else []
    )
    if (
        type(rows) is not list or len(rows) != 34
        or not all(type(path) is str for path in row_paths)
        or row_paths != sorted(row_paths) or len(set(row_paths)) != 34
    ):
        raise SnapshotControllerError("snapshot manifest rows differ")
    expected_files = {row["path"] for row in rows} | {MANIFEST_NAME}
    actual_files: set[str] = set()
    actual_directories = {"."}
    pending = [(TARGET_ROOT, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != DIRECTORY_MODE
            or info.st_uid != REMOTE_UID or info.st_gid != REMOTE_GID
        ):
            raise SnapshotControllerError(f"snapshot directory differs: {prefix}")
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
                    raise SnapshotControllerError(
                        f"snapshot special/link entry differs: {relative}"
                    )
    expected_directories = {"."}
    for relative in expected_files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent)); parent = parent.parent
    if (
        actual_files != expected_files or len(actual_files) != 35
        or actual_directories != expected_directories
    ):
        raise SnapshotControllerError("snapshot physical35 closure differs")

    for row in rows:
        if (
            set(row) != {"path", "sha256", "size", "mode", "provenance"}
            or type(row.get("size")) is not int or row["size"] <= 0
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or row.get("mode")
            != (RECEIPT_MODE if row["path"] == STAGING_RECEIPT_COPY_RELATIVE
                else FILE_MODE)
        ):
            raise SnapshotControllerError(f"snapshot row differs: {row}")
        held = open_authority(
            TARGET_ROOT / row["path"], expected_sha256=row["sha256"],
            expected_size=row["size"], expected_mode=row["mode"],
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        held.close()
    return {"file_count": 35, "directory_count": len(actual_directories)}


def validate_publication(
    returned_manifest: Mapping[str, Any], source_receipt: HeldAuthority,
) -> dict[str, Any]:
    manifest_authority = open_authority(
        TARGET_ROOT / MANIFEST_NAME, expected_sha256=None, expected_size=None,
        expected_mode=FILE_MODE, expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
        maximum_size=MAX_JSON_SIZE,
    )
    receipt_authority: HeldAuthority | None = None
    try:
        manifest = strict_json(manifest_authority.raw, label="exact35 manifest")
        manifest_unsigned = dict(manifest)
        manifest_claimed = manifest_unsigned.pop("manifest_digest", None)
        if (
            set(manifest) != SNAPSHOT_MANIFEST_FIELDS
            or manifest != returned_manifest
            or manifest.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA
            or manifest.get("status") != "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE"
            or manifest_claimed != object_digest(manifest_unsigned)
            or manifest.get("launch_allowed") is not False
            or manifest.get("old_snapshot_root") != str(OLD_ROOT)
            or manifest.get("staging_root") != str(SOURCE_ROOT)
            or manifest.get("staging_receipt_path") != str(SOURCE_RECEIPT_PATH)
            or manifest.get("snapshot_publication_receipt_path")
            != str(TARGET_RECEIPT_PATH)
            or manifest.get("target_root") != str(TARGET_ROOT)
            or manifest.get("content_leaf_count") != 34
            or manifest.get("physical_file_count_including_manifest") != 35
            or manifest.get("release_file_count") != 25
            or manifest.get("legacy_alias_is_distinct_regular_inode") is not True
            or manifest.get("builder_authority") != {
                "path": str(BUILDER_PATH), "sha256": BUILDER_SHA256,
                "size": BUILDER_SIZE, "sealed_bytes_in_snapshot": False,
            }
            or manifest.get("publication_protocol") != SOURCE_PUBLICATION_PROTOCOL
            or manifest.get("rename_noreplace") is not False
            or manifest.get("cooperative_writer_exclusion") is not True
            or manifest.get("target_absent_rechecked") is not True
            or manifest.get("whole_tree_atomically_visible") is not True
            or manifest.get("uncooperative_same_uid_race_out_of_scope") is not True
            or manifest.get("retry_allowed") is not False
        ):
            raise SnapshotControllerError("exact35 manifest closure differs")
        staging = manifest.get("staging_receipt_authority")
        if (
            type(staging) is not dict
            or staging.get("source_path") != str(SOURCE_RECEIPT_PATH)
            or staging.get("snapshot_relative") != STAGING_RECEIPT_COPY_RELATIVE
            or staging.get("sha256") != SOURCE_RECEIPT_SHA256
            or staging.get("size") != SOURCE_RECEIPT_SIZE
            or staging.get("mode") != RECEIPT_MODE
            or staging.get("schema_version") != SOURCE_RECEIPT_SCHEMA
            or staging.get("receipt_digest") != SOURCE_RECEIPT_DIGEST
            or staging.get("staging_file_count") != 15
            or staging.get("copied_as_snapshot_leaf") is not True
            or staging.get("replayed_before_and_after_snapshot_build") is not True
        ):
            raise SnapshotControllerError("snapshot staging authority differs")
        tree = validate_snapshot_tree(manifest)
        copied = open_authority(
            TARGET_ROOT / STAGING_RECEIPT_COPY_RELATIVE,
            expected_sha256=SOURCE_RECEIPT_SHA256,
            expected_size=SOURCE_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        try:
            if copied.raw != source_receipt.raw:
                raise SnapshotControllerError("copied staging receipt bytes differ")
        finally:
            copied.close()

        receipt_authority = open_authority(
            TARGET_RECEIPT_PATH, expected_sha256=None, expected_size=None,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        receipt = strict_json(receipt_authority.raw, label="snapshot publication receipt")
        receipt_unsigned = dict(receipt)
        receipt_claimed = receipt_unsigned.pop("receipt_digest", None)
        target_info = os.lstat(TARGET_ROOT)
        expected_observation = {
            "namespace_state": "target_same_inode_source_absent",
            "rename_returned_zero": True,
            "rename_error_errno": None,
            "parent_fsync_returned_zero": True,
            "parent_fsync_error_errno": None,
        }
        if (
            set(receipt) != SNAPSHOT_RECEIPT_FIELDS
            or receipt_claimed != object_digest(receipt_unsigned)
            or receipt.get("schema_version") != SNAPSHOT_RECEIPT_SCHEMA
            or receipt.get("status") != "PUBLISHED_RECEIPT_GATED"
            or receipt.get("target_root") != str(TARGET_ROOT)
            or receipt.get("receipt_path") != str(TARGET_RECEIPT_PATH)
            or receipt.get("manifest_path") != str(TARGET_ROOT / MANIFEST_NAME)
            or receipt.get("manifest_sha256")
            != hashlib.sha256(manifest_authority.raw).hexdigest()
            or receipt.get("manifest_digest") != manifest_claimed
            or receipt.get("staging_receipt_sha256") != SOURCE_RECEIPT_SHA256
            or receipt.get("staging_receipt_digest") != SOURCE_RECEIPT_DIGEST
            or receipt.get("content_leaf_count") != 34
            or receipt.get("physical_file_count_including_manifest") != 35
            or receipt.get("publication_protocol") != SOURCE_PUBLICATION_PROTOCOL
            or receipt.get("rename_noreplace") is not False
            or receipt.get("cooperative_writer_exclusion") is not True
            or receipt.get("target_absent_rechecked_before_rename") is not True
            or receipt.get("ordinary_posix_rename_performed") is not True
            or receipt.get("publication_observation") != expected_observation
            or receipt.get("whole_tree_atomically_visible") is not True
            or receipt.get("uncooperative_same_uid_race_out_of_scope") is not True
            or receipt.get("retry_allowed") is not False
            or receipt.get("target_root_identity") != list(identity(target_info))
            or receipt.get("receipt_mode") != RECEIPT_MODE
            or receipt.get("receipt_is_consumption_gate") is not True
            or receipt.get("receipt_is_admission") is not True
            or receipt.get("launch_allowed") is not False
            or receipt.get("receipt_inode_anchor")
            != inode_anchor(os.fstat(receipt_authority.descriptor))
        ):
            raise SnapshotControllerError("snapshot publication receipt differs")
        return {
            "manifest": {
                "path": str(TARGET_ROOT / MANIFEST_NAME),
                "sha256": hashlib.sha256(manifest_authority.raw).hexdigest(),
                "size": len(manifest_authority.raw),
                "manifest_digest": manifest_claimed,
            },
            "publication_receipt": {
                "path": str(TARGET_RECEIPT_PATH),
                "sha256": hashlib.sha256(receipt_authority.raw).hexdigest(),
                "size": len(receipt_authority.raw),
                "receipt_digest": receipt_claimed,
            },
            "target_root_identity": list(identity(target_info)),
            **tree,
        }
    finally:
        if receipt_authority is not None:
            receipt_authority.close()
        manifest_authority.close()


def controller() -> dict[str, Any]:
    authorities: list[HeldAuthority] = []
    source_gate: Any | None = None
    try:
        # CPU admission is the operational prerequisite.  Both of its final
        # receipts are opened and pinned before the physical15 gate.
        cpu_world4 = open_authority(
            CPU_WORLD4_RECEIPT_PATH,
            expected_sha256=CPU_WORLD4_RECEIPT_SHA256,
            expected_size=CPU_WORLD4_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(cpu_world4)
        cpu_evidence = open_authority(
            CPU_CONTROLLER_EVIDENCE_PATH,
            expected_sha256=CPU_CONTROLLER_EVIDENCE_SHA256,
            expected_size=CPU_CONTROLLER_EVIDENCE_SIZE,
            expected_mode=RECEIPT_MODE, expected_uid=REMOTE_UID,
            expected_gid=REMOTE_GID, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(cpu_evidence)
        _cpu_world4_value, cpu_evidence_value = validate_cpu_authorities(
            cpu_world4, cpu_evidence,
        )

        # Receipt-first is literal: no SOURCE_ROOT or physical15 leaf has been
        # named/opened before this final canonical 0400 AUHv2 receipt.
        source_receipt = open_authority(
            SOURCE_RECEIPT_PATH, expected_sha256=SOURCE_RECEIPT_SHA256,
            expected_size=SOURCE_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(source_receipt)
        source_receipt_value = validate_source_receipt_prefix(source_receipt)
        validate_cpu_source_crosslink(
            cpu_evidence_value, source_receipt_value, source_receipt,
        )

        runtime = open_runtime_authority(); authorities.append(runtime)
        self_authority = open_self_authority(); authorities.append(self_authority)
        builder_authority = open_authority(
            BUILDER_PATH, expected_sha256=BUILDER_SHA256,
            expected_size=BUILDER_SIZE, expected_mode=FILE_MODE,
            expected_uid=REMOTE_UID, expected_gid=REMOTE_GID,
            maximum_size=MAX_SOURCE_SIZE,
        )
        authorities.append(builder_authority)
        builder = load_builder(builder_authority.raw)

        # The pinned builder performs the full held live exact15 validation;
        # its second receipt open must reproduce the already-held first bytes.
        source_gate = builder.open_staging_gate(BUILDER_SHA256)
        if (
            source_gate.receipt.raw != source_receipt.raw
            or source_gate.receipt_value != source_receipt_value
        ):
            raise SnapshotControllerError("builder/controller staging gates differ")
        for authority in authorities:
            authority.replay()
        source_gate.replay()
        require_fresh_outputs()

        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_BUILDER",
            "single_attempt": True,
            "retry_allowed": False,
            "launch_allowed": False,
            "source_staging_receipt": {
                "path": str(SOURCE_RECEIPT_PATH),
                "sha256": SOURCE_RECEIPT_SHA256,
                "size": SOURCE_RECEIPT_SIZE,
                "receipt_digest": SOURCE_RECEIPT_DIGEST,
            },
            "cpu_world4_receipt": {
                "path": str(CPU_WORLD4_RECEIPT_PATH),
                "sha256": CPU_WORLD4_RECEIPT_SHA256,
                "size": CPU_WORLD4_RECEIPT_SIZE,
                "receipt_digest": CPU_WORLD4_RECEIPT_DIGEST,
            },
            "cpu_controller_evidence": {
                "path": str(CPU_CONTROLLER_EVIDENCE_PATH),
                "sha256": CPU_CONTROLLER_EVIDENCE_SHA256,
                "size": CPU_CONTROLLER_EVIDENCE_SIZE,
                "evidence_digest": CPU_CONTROLLER_EVIDENCE_DIGEST,
            },
            "controller": self_authority.row(),
            "python": runtime.row(),
            "builder": builder_authority.row(),
            "old_root": str(OLD_ROOT),
            "target_root": str(TARGET_ROOT),
            "target_receipt_path": str(TARGET_RECEIPT_PATH),
            "exact_argv": [
                str(BUILDER_PATH), "--old-root", str(OLD_ROOT),
                "--staging-root", str(SOURCE_ROOT), "--target-root",
                str(TARGET_ROOT), "--builder-sha256", BUILDER_SHA256,
            ],
            "authorization_token": authorization_token(),
        }
        attempt["attempt_digest"] = object_digest(attempt)
        attempt_raw, attempt_anchor = create_immutable_json(ATTEMPT_PATH, attempt)

        for authority in authorities:
            authority.replay()
        source_gate.replay()
        try:
            returned_manifest = builder.build(
                OLD_ROOT, SOURCE_ROOT, TARGET_ROOT,
                builder_sha256=BUILDER_SHA256,
            )
        except Exception as error:
            # The 0400 attempt claim already exists.  A builder refusal is a
            # permanent single-attempt HOLD and is never retried or erased.
            raise SnapshotControllerError(
                "pinned builder failed after the durable attempt claim"
            ) from error
        for authority in authorities:
            authority.replay()
        source_gate.replay()
        publication = validate_publication(returned_manifest, source_receipt)

        evidence: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "PASS_EXACT35_PUBLISHED_RECEIPT_GATED",
            "single_attempt": True,
            "retry_allowed": False,
            "launch_allowed": False,
            "attempt": {
                "path": str(ATTEMPT_PATH),
                "sha256": hashlib.sha256(attempt_raw).hexdigest(),
                "size": len(attempt_raw),
                "attempt_digest": attempt["attempt_digest"],
                "receipt_inode_anchor": attempt_anchor,
            },
            "source_staging_receipt": attempt["source_staging_receipt"],
            "cpu_world4_receipt": attempt["cpu_world4_receipt"],
            "cpu_controller_evidence": attempt["cpu_controller_evidence"],
            "controller": self_authority.row(),
            "python": runtime.row(),
            "builder": builder_authority.row(),
            "publication": publication,
            "builder_called_once": True,
            "ssh_performed": False,
            "slurm_performed": False,
        }
        evidence["evidence_digest"] = object_digest(evidence)
        create_immutable_json(EVIDENCE_PATH, evidence)
        return evidence
    finally:
        if source_gate is not None:
            source_gate.close()
        for authority in reversed(authorities):
            try:
                authority.close()
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    # This gate precedes argv parsing and every explicit stat/open/read/write,
    # process, network, temporary-file, target, receipt, or directory action.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: exact35 controller awaits final physical15/CPU receipt pins "
            "and a reviewed state-only READY copy",
            file=sys.stderr,
        )
        return 88
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        blocked = blocked_dynamic_pins()
        if blocked:
            raise SnapshotControllerError(
                "HOLD: dynamic receipt pins are blocked: " + ",".join(blocked)
            )
        if values != ["--execute", authorization_token()]:
            raise SnapshotControllerError("controller argv/token differs")
        result = controller()
        print(canonical(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, KeyError, SnapshotControllerError) as error:
        print(f"exact35 snapshot controller refused: {error}", file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96


if __name__ == "__main__":
    raise SystemExit(main())
