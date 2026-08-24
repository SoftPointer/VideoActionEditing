#!/usr/bin/env python3
"""One-shot four-process CPU admission for the fresh r5f-v4 composite.

This checked-in source is deliberately HOLD.  It consumes only the fresh
package publication/materialization/controller receipts, runs one CPU-only
Slurm step on node292, and publishes fresh sibling evidence.  It never writes
inside the package and never names the production GPU rank cache from compute
except to prove that path remains absent before and after the probe.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-r5f-v4-composite-cpu-controller-v2"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
EVIDENCE_SCHEMA = SCHEMA + "-evidence"
RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-r5f-v4-composite-cpu-admission-v2"
)
CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_CPU_SRUN_NO_RETRY"
READY_STATE = "READY_EXPLICIT_SINGLE_CPU_SRUN_NO_RETRY"

HOLDER_JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
CPUS_PER_TASK = 8
MEMORY = "16G"
SRUN_TIMEOUT_SECONDS = 1_200
CHILD_TIMEOUT_SECONDS = 180
PROCESS_POLL_SECONDS = 0.05
PROCESS_TERM_GRACE_SECONDS = 10.0
PROCESS_KILL_GRACE_SECONDS = 10.0
REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
DIRECTORY_MODE = 0o700
MAX_JSON_SIZE = 64 * 1024 * 1024
MAX_SOURCE_SIZE = 4 * 1024 * 1024
MAX_EXECUTABLE_SIZE = 128 * 1024 * 1024
MAX_STDIN_SIZE = 256 * 1024
MAX_ARGV_BYTES = 32 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SHARED_OFD_PAYLOAD_SHA256 = (
    "08e33aedf25337c87eb15e08c32a58f6f4caa21fe073d00b53014c57f8d148e0"
)

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
PACKAGE_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_r64_canary_v3"
)
PUBLICATION_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "publication_receipt_v4.json"
)
MATERIALIZATION_PATH = (
    PACKAGE_ROOT / "authority/package_materialization_receipt_v4.json"
)
PACKAGE_CONTROLLER_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "materialize_controller_evidence_v3.json"
)
ATTEMPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_attempt_v2.json"
)
RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_receipt_v2.json"
)
EVIDENCE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_controller_evidence_v2.json"
)
STDOUT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_srun_v2.stdout.log"
)
STDERR_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_srun_v2.stderr.log"
)
PRODUCTION_RANK_CACHE = Path(
    "/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-"
    "rank-cache"
)

PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v4-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v4"
)
PACKAGE_CONTROLLER_SCHEMA = (
    "case01-object-trajectory-exact5-r64-overlay-package-controller-v3-evidence"
)
PUBLICATION_FIELDS = frozenset({
    "schema_version", "status", "target_root", "receipt_path",
    "materialization_receipt_path", "materialization_receipt_sha256",
    "materialization_receipt_digest", "source_snapshot_manifest_sha256",
    "source_snapshot_manifest_digest", "source_staging_receipt_sha256",
    "source_staging_receipt_digest", "source_overlay_receipt_sha256",
    "source_overlay_receipt_digest", "source_overlay_root_identity",
    "publication_protocol", "rename_noreplace", "cooperative_writer_exclusion",
    "target_absent_rechecked_before_rename", "ordinary_posix_rename_performed",
    "publication_observation", "whole_tree_atomically_visible",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
    "receipt_is_admission", "launch_allowed", "receipt_inode_anchor",
    "receipt_digest",
})
MATERIALIZATION_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "root",
    "source_snapshot_root", "source_snapshot", "source_overlay_root",
    "source_overlay", "source_provenance", "source_staging_receipt_authority",
    "package_publication_receipt_path", "publication_protocol",
    "rename_noreplace", "cooperative_writer_exclusion",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "release_file_count", "production_identity_count", "release", "production",
    "condition_and_admission_authority_count", "plan", "launch", "admission",
    "slurm_step_launched", "gpu_attempt_claimed", "artifacts", "receipt_digest",
})
LAUNCH_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "slurm_step_launched",
    "gpu_attempt_claimed", "input", "release", "payload_path",
    "payload_sha256", "payload_size", "receipt_digest",
})
LAUNCH_RELEASE_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "campaign_mode",
    "selected_task_ids", "identity_roles", "identities", "input_sha256",
    "ready_overlay_required", "named_payload_execution_forbidden",
    "release_digest",
})
REPORT_RELEASE_FIELDS = frozenset({"files", "manifest_digest"})
REPORT_RELEASE_ROW_FIELDS = frozenset({
    "path", "sha256", "size", "provenance",
})
PRODUCTION_FIELDS = frozenset({
    "identity_roles", "identities", "identity_set_digest",
    "inner_outer_crosslink",
})
CROSSLINK_FIELDS = frozenset({
    "adapter", "object_wrapper_inner", "producer_adapter",
    "producer_object_wrapper_inner", "distinct_paths",
    "outer_calls_pinned_inner_contract",
})
PLAN_ROW_FIELDS = frozenset({"path", "sha256", "plan_digest"})
PACKAGE_PUBLICATION_FIELDS = frozenset({
    "publication_receipt", "publication_receipt_digest",
    "materialization_receipt", "materialization_receipt_digest",
    "package_root", "release", "production", "gpu_attempt_claimed",
    "srun_performed", "file_count", "directory_count",
})
AUTHORITY_ROW_FIELDS = frozenset({"path", "sha256", "size", "identity"})
DIRECTORY_ROW_FIELDS = frozenset({"path", "identity"})
TASK_IDS = tuple(
    "case01-object-trajectory-" + arm + "-full644"
    for arm in (
        "null_before", "route_off", "trajectory_bone_only",
        "trajectory_dog_bone", "null_after",
    )
)
DIAGNOSTIC_ARTIFACT_SHA256 = {
    "diagnostics/case01_object_trajectory_exact5_static_probe_v1.py":
        "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9",
    "diagnostics/case01_object_trajectory_exact5_root_fake_runner_v1.py":
        "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872",
    "diagnostics/case01_object_trajectory_exact5_world4_probe_v1.py":
        "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
}
PACKAGE_CONTROLLER_FIELDS = frozenset({
    "schema_version", "status", "single_attempt", "retry_allowed",
    "launch_allowed", "attempt", "snapshot", "overlay", "controller",
    "python", "materializer", "child", "publication", "ssh_performed",
    "slurm_performed", "srun_performed", "gpu_attempt_claimed",
    "renderer_invoked", "evidence_digest",
})
CPU_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "package", "world_size", "rank_count", "rank_rows",
    "isolated_runtime", "private_parent_fd", "shared_ofd_pread",
    "module_binding", "activation_import", "side_effects", "cache_lifecycle",
    "process_cleanup", "launch_allowed", "receipt_digest",
})
CPU_RANK_ROW_FIELDS = frozenset({
    "rank", "pid", "private_parent_fd_number",
    "private_parent_replacement_inode", "pread_bytes_sha256",
    "pread_offset_before", "pread_offset_after",
    "activation_callback_import_module",
    "activation_import_before_callback_return",
    "captured_vendor_finder_preinstalled",
    "captured_vendor_finder_count", "captured_vendor_loader_type",
    "captured_vendor_spec_loader_type",
    "captured_vendor_loader_is_spec_loader",
    "captured_vendor_cached_is_none", "rank_digest",
})
CPU_EVIDENCE_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "single_srun_attempt", "retry_allowed", "srun_count", "srun_ntasks",
    "real_rank_process_count", "cpus_per_task", "gpu_count",
    "srun_returncode", "receipt", "receipt_digest", "stdout", "stderr",
    "stderr_empty", "process_group_zero", "launch_allowed",
    "renderer_or_vae_loaded", "publication_performed", "evidence_digest",
})
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle-v3"
IDENTITY_ROLES = (
    "runner", "legacy_exact5_runner", "object_eval", "legacy_exact5_eval",
    "frozen_runner", "bridge", "adapter", "object_wrapper_inner",
    "legacy_infer_alias", "trajectory_projection",
    "trajectory_scaffold_module", "base_adapter", "eval_v1", "eval_v2",
    "model_authority", "torchrun_source", "torchrun_handler_source",
    "torch_local_agent_source", "torch_dynamic_rendezvous_source",
    "torch_multiprocessing_api_source", "base_model_manifest",
    "r64_checkpoint_manifest", "python", "ffmpeg", "ffprobe", "plan",
)
if len(IDENTITY_ROLES) != 26 or len(set(IDENTITY_ROLES)) != 26:
    raise RuntimeError("composite CPU exact26 role closure differs")

PUBLICATION_SHA256 = (
    "ffb74c4cf70ced6491cde23a37d9389b3f8c65431e354194d96842dd6a494871"
)
PUBLICATION_SIZE: int | str = 2_528
PUBLICATION_DIGEST = (
    "82533716cd0286182fc731e3ffdf46cf8b95ed1ae0bb0a421d26aa77685bf720"
)
MATERIALIZATION_SHA256 = (
    "c60c28ab1418914fd61480507c7c2e284ea58a1132fb265a40e3a5aa2ec56c95"
)
MATERIALIZATION_SIZE: int | str = 41_726
MATERIALIZATION_DIGEST = (
    "d0790a3618539d918d7deaa07a066961b08a19e4973de5e11f8abca9cd52d7be"
)
PACKAGE_CONTROLLER_SHA256 = (
    "bed59791557b9cdebd8280edbd3a68976c4588984815045eeea1f45b864ea0c7"
)
PACKAGE_CONTROLLER_SIZE: int | str = 8_099
PACKAGE_CONTROLLER_DIGEST = (
    "39dbc033a2845cb7a73d759334df76411521cebd20a455620d37f5db5339236e"
)
PACKAGE_ROOT_IDENTITY: list[int] | str = [
    48, 3113453814725663979, 2012, 2000, 16832, 2, 0, 4096, 0,
    1787378196307021665, 1787378196629068696,
]

BASE_ADAPTER_SHA256 = (
    "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120"
)
CORE4_RELEASE_AUTHORITIES = {
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v3.py":
        (BASE_ADAPTER_SHA256, 124_612),
    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_auh_r5f_v4.py":
        ("797c5d1e7cb8bbfda1f2e4cc3825702c248d3ce64770ddc1520155f5635c3557", 42_184),
    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v4.py":
        ("381ba375147bec7580b451226b07b3d1cab9125866978602de05fbba4f16aaa3", 116_371),
    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v4.py":
        ("326ccfff1a09d6db8c93d02cfe6018e465e127263547f325cc7f18e7d16a7148", 21_712),
    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v4.py":
        ("0315a8630f77e816c3fc5fc9139b8fb72323db59d5d155f85b039ba132cc9b5a", 27_878),
}
PLAN_SHA256 = "d9dadcd5a293e2313e4e5381bd095380f2da730add4c29afba2dd38f9b2e7483"
PLAN_SIZE = 32_050
PLAN_DIGEST = "e4485a73c1988a3560378000cdee2182266c81aff5f0318b83ac21d9ee787d24"
LAUNCH_RECEIPT_SHA256 = (
    "a100ab8200ff0eedfc2fd065559256b74c00e12b10dcbd716013b74a336e4e01"
)
LAUNCH_RECEIPT_SIZE = 10_292
LAUNCH_RECEIPT_DIGEST = (
    "73c1ec7d6a355edc515ea2a3221d06f6bce38e8e60b755c6f71ccefaf660ee7b"
)
LAUNCH_INPUT_SHA256 = (
    "7973c9311e5a539a106a938bf452d5405f492fdd8f7f41a9038b5da09279d347"
)
LAUNCH_INPUT_SIZE = 9_788
LAUNCH_PAYLOAD_SHA256 = (
    "f5eb7add48c521d01893e64e4d12963401b0aa2986e25ed06f66afb4fdaa1ccf"
)
LAUNCH_PAYLOAD_SIZE = 12_783

SRUN_AUTHORITY = {
    "path": "/usr/bin/srun",
    "sha256": "2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e",
    "size": 164_720,
}


class CompositeCPUError(RuntimeError):
    """The reviewed fresh-package CPU admission differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CompositeCPUError("value is not canonical JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(info.st_mode), int(info.st_nlink), int(info.st_rdev),
        int(info.st_size), int(getattr(info, "st_blocks", 0)),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise CompositeCPUError("held size differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        chunks.append(block); offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise CompositeCPUError("held read is incomplete")
    return raw


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CompositeCPUError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise CompositeCPUError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise CompositeCPUError(f"noncanonical JSON authority: {label}")
    return value


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...],
        raw: bytes,
    ) -> None:
        self.path = path; self.descriptor = descriptor
        self.held_identity = held_identity; self.raw = raw

    def replay(self) -> None:
        opened = os.fstat(self.descriptor); named = os.lstat(self.path)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or read_fd(self.descriptor, opened.st_size) != self.raw
        ):
            raise CompositeCPUError(f"held authority changed: {self.path}")

    def row(self) -> dict[str, Any]:
        info = os.fstat(self.descriptor)
        return {
            "path": str(self.path),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw), "identity": list(identity(info)),
            "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
        }

    def bare_row(self) -> dict[str, Any]:
        row = self.row()
        return {key: row[key] for key in ("path", "sha256", "size", "identity")}

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


class HeldDirectory:
    def __init__(self, path: Path, descriptor: int, held_identity: tuple[int, ...]):
        self.path = path; self.descriptor = descriptor
        self.held_identity = held_identity

    def replay(self) -> None:
        if (
            identity(os.fstat(self.descriptor)) != self.held_identity
            or identity(os.lstat(self.path)) != self.held_identity
        ):
            raise CompositeCPUError(f"held directory changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def open_authority(
    path: Path, *, expected_sha256: str | None,
    expected_size: int | None, expected_mode: int | None,
    maximum_size: int, executable: bool = False,
    expected_uid: int | None = REMOTE_UID,
    expected_gid: int | None = REMOTE_GID,
) -> HeldAuthority:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise CompositeCPUError(f"authority path differs: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or path.resolve(strict=True) != path
        or named.st_size < 0 or named.st_size > maximum_size
        or (expected_uid is not None and named.st_uid != expected_uid)
        or (expected_gid is not None and named.st_gid != expected_gid)
        or (expected_mode is not None
            and stat.S_IMODE(named.st_mode) != expected_mode)
        or (expected_size is not None and named.st_size != expected_size)
        or (executable and not named.st_mode & 0o111)
    ):
        raise CompositeCPUError(f"named authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor); first = read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor); second = read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor); named_after = os.lstat(path)
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
            raise CompositeCPUError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), first)
    except BaseException:
        os.close(descriptor); raise


def open_directory(path: Path, expected_identity: Sequence[int]) -> HeldDirectory:
    if (
        type(expected_identity) not in {list, tuple}
        or len(expected_identity) != 11
        or any(type(value) is not int for value in expected_identity)
    ):
        raise CompositeCPUError("package root identity pin differs")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor); named = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != REMOTE_UID or opened.st_gid != REMOTE_GID
            or stat.S_IMODE(opened.st_mode) != DIRECTORY_MODE
            or identity(opened) != identity(named)
            or identity(opened) != tuple(expected_identity)
            or path.resolve(strict=True) != path
        ):
            raise CompositeCPUError("held package root differs")
        return HeldDirectory(path, descriptor, identity(opened))
    except BaseException:
        os.close(descriptor); raise


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "publication_sha256": PUBLICATION_SHA256,
        "publication_size": PUBLICATION_SIZE,
        "publication_digest": PUBLICATION_DIGEST,
        "materialization_sha256": MATERIALIZATION_SHA256,
        "materialization_size": MATERIALIZATION_SIZE,
        "materialization_digest": MATERIALIZATION_DIGEST,
        "package_controller_sha256": PACKAGE_CONTROLLER_SHA256,
        "package_controller_size": PACKAGE_CONTROLLER_SIZE,
        "package_controller_digest": PACKAGE_CONTROLLER_DIGEST,
        "package_root_identity": PACKAGE_ROOT_IDENTITY,
    }


def blocked_dynamic_pins() -> tuple[str, ...]:
    blocked: list[str] = []
    for key, value in dynamic_pin_values().items():
        if key == "package_root_identity":
            valid = (
                type(value) is list and len(value) == 11
                and all(type(item) is int for item in value)
            )
        elif key.endswith("_size"):
            valid = type(value) is int and 0 < value <= MAX_JSON_SIZE
        else:
            valid = type(value) is str and SHA_RE.fullmatch(value) is not None
        if not valid:
            blocked.append(key)
    return tuple(blocked)


def authorization_token() -> str:
    return object_digest({
        "schema_version": SCHEMA + "-authorization",
        "state": CONTROLLER_STATE, "job_id": HOLDER_JOB_ID, "node": NODE,
        "dynamic_pins": dynamic_pin_values(),
        "targets": [
            str(ATTEMPT_PATH), str(RECEIPT_PATH), str(EVIDENCE_PATH),
            str(STDOUT_PATH), str(STDERR_PATH),
        ],
        "single_srun": True, "retry_allowed": False, "gpu_count": 0,
    })


def _self_digest(value: Mapping[str, Any], field: str, expected: str) -> None:
    unsigned = dict(value); claimed = unsigned.pop(field, None)
    if claimed != expected or claimed != object_digest(unsigned):
        raise CompositeCPUError(f"{field} closure differs")


def validate_package_receipts(
    publication_held: HeldAuthority, materialization_held: HeldAuthority,
    controller_held: HeldAuthority,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    publication = strict_json(publication_held.raw, label="package publication")
    _self_digest(publication, "receipt_digest", PUBLICATION_DIGEST)
    if (
        set(publication) != PUBLICATION_FIELDS
        or publication.get("schema_version") != PUBLICATION_SCHEMA
        or publication.get("status") != "PUBLISHED_RECEIPT_GATED"
        or publication.get("target_root") != str(PACKAGE_ROOT)
        or publication.get("receipt_path") != str(PUBLICATION_PATH)
        or publication.get("materialization_receipt_path")
        != str(MATERIALIZATION_PATH)
        or publication.get("materialization_receipt_sha256")
        != MATERIALIZATION_SHA256
        or publication.get("materialization_receipt_digest")
        != MATERIALIZATION_DIGEST
        or publication.get("target_root_identity") != PACKAGE_ROOT_IDENTITY
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
    ):
        raise CompositeCPUError("package publication semantics differ")

    report = strict_json(materialization_held.raw, label="materialization")
    _self_digest(report, "receipt_digest", MATERIALIZATION_DIGEST)
    report_release = report.get("release")
    release_rows = (
        report_release.get("files") if type(report_release) is dict else None
    )
    production = report.get("production")
    identities = production.get("identities") if type(production) is dict else None
    crosslink = (
        production.get("inner_outer_crosslink")
        if type(production) is dict else None
    )
    plan = report.get("plan")
    launch = report.get("launch")
    release = launch.get("release") if type(launch) is dict else None
    input_row = launch.get("input") if type(launch) is dict else None
    if (
        set(report) != MATERIALIZATION_FIELDS
        or report.get("schema_version") != MATERIALIZATION_SCHEMA
        or report.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or report.get("root") != str(PACKAGE_ROOT)
        or report.get("launch_allowed") is not False
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
        or type(report_release) is not dict
        or set(report_release) != REPORT_RELEASE_FIELDS
        or type(release_rows) is not list or len(release_rows) != 25
        or report_release.get("manifest_digest") != object_digest(release_rows)
        or type(production) is not dict
        or set(production) != PRODUCTION_FIELDS
        or production.get("identity_roles") != list(IDENTITY_ROLES)
        or type(identities) is not dict
        or set(identities) != set(IDENTITY_ROLES)
        or len(identities) != 26
        or production.get("identity_set_digest") != object_digest(identities)
        or type(crosslink) is not dict
        or set(crosslink) != CROSSLINK_FIELDS
        or crosslink.get("adapter") != identities.get("adapter")
        or crosslink.get("object_wrapper_inner")
        != identities.get("object_wrapper_inner")
        or crosslink.get("producer_adapter") != identities.get("adapter")
        or crosslink.get("producer_object_wrapper_inner")
        != identities.get("object_wrapper_inner")
        or crosslink.get("distinct_paths") is not True
        or crosslink.get("outer_calls_pinned_inner_contract") is not True
        or type(release) is not dict
        or set(release) != LAUNCH_RELEASE_FIELDS
        or type(launch) is not dict or set(launch) != LAUNCH_FIELDS
        or type(plan) is not dict or set(plan) != PLAN_ROW_FIELDS
        or type(input_row) is not dict
        or set(input_row) != {"path", "sha256", "size", "mode", "nlink"}
        or release.get("campaign_mode") != CAMPAIGN
        or release.get("schema_version")
        != "case01-object-trajectory-exact5-hold-launch-release-auh-v3"
        or release.get("status") != "HOLD_NOT_LAUNCHABLE"
        or release.get("selected_task_ids") != list(TASK_IDS)
        or release.get("identity_roles") != list(IDENTITY_ROLES)
        or release.get("identities") != identities
        or release.get("launch_allowed") is not False
        or release.get("named_payload_execution_forbidden") is not True
        or release.get("ready_overlay_required") is not True
        or release.get("input_sha256") != input_row.get("sha256")
        or launch.get("schema_version")
        != "case01-object-trajectory-exact5-hold-launch-receipt-auh-v3"
        or launch.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or launch.get("launch_allowed") is not False
        or launch.get("slurm_step_launched") is not False
        or launch.get("gpu_attempt_claimed") is not False
        or launch.get("payload_path") != str(
            PACKAGE_ROOT / "launch/root_launch_payload_HOLD_v3.sh"
        )
        or launch.get("payload_size") != LAUNCH_PAYLOAD_SIZE
        or launch.get("payload_sha256") != LAUNCH_PAYLOAD_SHA256
        or launch.get("receipt_digest") != LAUNCH_RECEIPT_DIGEST
        or hashlib.sha256(canonical(launch) + b"\n").hexdigest()
        != LAUNCH_RECEIPT_SHA256
        or len(canonical(launch) + b"\n") != LAUNCH_RECEIPT_SIZE
        or input_row.get("path") != str(
            PACKAGE_ROOT / "launch/root_launch_input_HOLD_v3.json"
        )
        or input_row.get("size") != LAUNCH_INPUT_SIZE
        or input_row.get("sha256") != LAUNCH_INPUT_SHA256
        or input_row.get("mode") != FILE_MODE or input_row.get("nlink") != 1
        or plan.get("path") != str(
            PACKAGE_ROOT / "plan/"
            "case01_object_trajectory_exact5_r64_HOLD_plan_v3.json"
        )
        or plan.get("sha256") != PLAN_SHA256
        or plan.get("plan_digest") != PLAN_DIGEST
        or identities.get("plan") != {
            "path": plan.get("path"), "sha256": plan.get("sha256"),
            "size": identities.get("plan", {}).get("size"),
        }
        or identities.get("plan", {}).get("size") != PLAN_SIZE
    ):
        raise CompositeCPUError("fresh materialization semantics differ")
    unsigned_launch = dict(launch); launch_digest = unsigned_launch.pop(
        "receipt_digest", None,
    )
    unsigned_release = dict(release); release_digest = unsigned_release.pop(
        "release_digest", None,
    )
    if (
        launch_digest != object_digest(unsigned_launch)
        or release_digest != object_digest(unsigned_release)
    ):
        raise CompositeCPUError("fresh launch digest closure differs")
    release_paths: list[str] = []
    for row in release_rows:
        if (
            type(row) is not dict or set(row) != REPORT_RELEASE_ROW_FIELDS
            or type(row.get("path")) is not str
            or not row["path"].startswith("release/")
            or Path(row["path"]).is_absolute()
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            or row.get("provenance") not in {
                "receipt_gated_exact6_overlay",
                "receipt_gated_exact35_snapshot",
            }
        ):
            raise CompositeCPUError("fresh release25 row differs")
        release_paths.append(row["path"])
    if release_paths != sorted(release_paths) or len(set(release_paths)) != 25:
        raise CompositeCPUError("fresh release25 path closure differs")
    release_by_relative = {
        row["path"][len("release/"):]: (row["sha256"], row["size"])
        for row in release_rows
    }
    if any(
        release_by_relative.get(relative) != authority
        for relative, authority in CORE4_RELEASE_AUTHORITIES.items()
    ):
        raise CompositeCPUError("fresh core4/base release authority differs")
    artifacts = report.get("artifacts")
    if type(artifacts) is not dict or len(artifacts) != 28:
        raise CompositeCPUError("fresh package artifact map differs")
    expected_release_artifacts = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in release_rows
    }
    if not set(expected_release_artifacts) <= set(artifacts):
        raise CompositeCPUError("fresh release artifact map differs")
    for path, expected in expected_release_artifacts.items():
        if artifacts.get(path) != expected:
            raise CompositeCPUError("fresh release artifact authority differs")
    if set(artifacts) - set(expected_release_artifacts) != set(
        DIAGNOSTIC_ARTIFACT_SHA256
    ):
        raise CompositeCPUError("fresh diagnostic artifact paths differ")
    for path, digest in DIAGNOSTIC_ARTIFACT_SHA256.items():
        row = artifacts.get(path)
        if (
            type(row) is not dict or set(row) != {"sha256", "size"}
            or row.get("sha256") != digest
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise CompositeCPUError("fresh diagnostic artifact differs")
    for role, row in identities.items():
        if (
            type(row) is not dict or set(row) != {"path", "sha256", "size"}
            or type(row.get("path")) is not str
            or not Path(row["path"]).is_absolute()
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise CompositeCPUError(f"production identity differs: {role}")
    expected_core_roles = {
        "runner": CORE4_RELEASE_AUTHORITIES[
            "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v4.py"
        ],
        "object_eval": CORE4_RELEASE_AUTHORITIES[
            "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v4.py"
        ],
        "adapter": CORE4_RELEASE_AUTHORITIES[
            "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_auh_r5f_v4.py"
        ],
        "base_adapter": CORE4_RELEASE_AUTHORITIES[
            "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v3.py"
        ],
    }
    for role, (expected_sha256, expected_size) in expected_core_roles.items():
        row = identities[role]
        if (
            row.get("sha256") != expected_sha256
            or row.get("size") != expected_size
            or Path(row["path"]).name not in {
                "case01_object_trajectory_exact5_runner_v4.py",
                "case01_object_trajectory_exact5_eval_v4.py",
                "infer_case01_object_trajectory_oracle_auh_r5f_v4.py",
                "full644_exploratory_matched_infer_adapter_v3.py",
            }
        ):
            raise CompositeCPUError(f"fresh core production identity differs: {role}")

    evidence = strict_json(controller_held.raw, label="package controller")
    _self_digest(evidence, "evidence_digest", PACKAGE_CONTROLLER_DIGEST)
    package = evidence.get("publication")
    if (
        set(evidence) != PACKAGE_CONTROLLER_FIELDS
        or evidence.get("schema_version") != PACKAGE_CONTROLLER_SCHEMA
        or evidence.get("status") != "PASS_R64_HOLD_PACKAGE_RECEIPT_GATED"
        or evidence.get("single_attempt") is not True
        or evidence.get("retry_allowed") is not False
        or evidence.get("launch_allowed") is not False
        or evidence.get("ssh_performed") is not False
        or evidence.get("slurm_performed") is not False
        or evidence.get("srun_performed") is not False
        or evidence.get("gpu_attempt_claimed") is not False
        or evidence.get("renderer_invoked") is not False
        or type(package) is not dict
        or set(package) != PACKAGE_PUBLICATION_FIELDS
        or type(package.get("publication_receipt")) is not dict
        or set(package["publication_receipt"]) != AUTHORITY_ROW_FIELDS
        or package["publication_receipt"].get("path") != str(PUBLICATION_PATH)
        or package["publication_receipt"].get("size") != len(publication_held.raw)
        or package["publication_receipt"].get("identity")
        != list(publication_held.held_identity)
        or package.get("publication_receipt", {}).get("sha256")
        != hashlib.sha256(publication_held.raw).hexdigest()
        or package.get("publication_receipt_digest") != PUBLICATION_DIGEST
        or type(package.get("materialization_receipt")) is not dict
        or set(package["materialization_receipt"]) != AUTHORITY_ROW_FIELDS
        or package["materialization_receipt"].get("path")
        != str(MATERIALIZATION_PATH)
        or package["materialization_receipt"].get("size")
        != len(materialization_held.raw)
        or package["materialization_receipt"].get("identity")
        != list(materialization_held.held_identity)
        or package.get("materialization_receipt", {}).get("sha256")
        != hashlib.sha256(materialization_held.raw).hexdigest()
        or package.get("materialization_receipt_digest") != MATERIALIZATION_DIGEST
        or type(package.get("package_root")) is not dict
        or set(package["package_root"]) != DIRECTORY_ROW_FIELDS
        or package.get("package_root") != {
            "path": str(PACKAGE_ROOT), "identity": PACKAGE_ROOT_IDENTITY,
        }
        or package.get("release") != {
            "file_count": 25,
            "manifest_digest": report["release"]["manifest_digest"],
        }
        or package.get("production") != {
            "identity_count": 26,
            "identity_roles": list(IDENTITY_ROLES),
            "identity_set_digest": production["identity_set_digest"],
            "inner_outer_crosslink": crosslink,
        }
        or package.get("gpu_attempt_claimed") is not False
        or package.get("srun_performed") is not False
        or package.get("file_count") != 39
        or type(package.get("directory_count")) is not int
        or package["directory_count"] <= 0
    ):
        raise CompositeCPUError("package controller crosslink differs")
    return publication, report, evidence


CHILD_BOOTSTRAP = r'''import hashlib,json,os,stat,sys,types
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def read_fd(fd,size):
 chunks=[];offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  chunks.append(block);offset+=len(block)
 raw=b"".join(chunks)
 if len(raw)!=size or os.pread(fd,1,size)!=b"": raise RuntimeError("child held read differs")
 return raw
if len(sys.argv)!=11: raise RuntimeError("composite child argv differs")
composite_fd=int(sys.argv[1]);composite_path=sys.argv[2];composite_sha=sys.argv[3];composite_size=int(sys.argv[4]);shared_fd=int(sys.argv[5]);shared_sha=sys.argv[6];rank=int(sys.argv[7]);admission_cache_root=sys.argv[8];base_adapter_path=sys.argv[9];base_adapter_sha=sys.argv[10]
if sys.flags.isolated!=1 or sys.flags.no_site!=1 or not sys.dont_write_bytecode or rank not in range(4): raise RuntimeError("child isolated flags differ")
capture=json.loads(os.environ.pop("R5F_CAPTURE"));binding=json.loads(os.environ.pop("R5F_BINDING"))
for row in binding["fd_rows"]: os.set_inheritable(row["fd"],False)
for fd in (composite_fd,shared_fd): os.set_inheritable(fd,False)
composite_raw=read_fd(composite_fd,composite_size)
if hashlib.sha256(composite_raw).hexdigest()!=composite_sha: raise RuntimeError("composite child source differs")
module=types.ModuleType("_case01_r5f_v4_composite_cpu_rank_%d"%rank);module.__file__=composite_path;module.__package__=None;module.__loader__=None;module.__cached__=None;module.__builtins__=__builtins__;sys.modules[module.__name__]=module
exec(compile(composite_raw,composite_path,"exec",dont_inherit=True),module.__dict__)
if base_adapter_sha!="7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120" or module._BASE_ADAPTER_SHA256!=base_adapter_sha or module.base.__file__!=base_adapter_path or module.base.__cached__ is not None: raise RuntimeError("base adapter v3 authority differs")
private_fd=capture["private_parent"]["authority_fd"]
owned={row["fd"] for row in binding["fd_rows"]}|{composite_fd,shared_fd}
if private_fd in owned: raise RuntimeError("private-parent FD unexpectedly inherited")
devnull=os.open("/dev/null",os.O_RDONLY)
if devnull!=private_fd:
 os.dup2(devnull,private_fd,inheritable=False);os.close(devnull)
else: os.set_inheritable(private_fd,False)
replacement=os.fstat(private_fd)
old_error=None
try: module._FROZEN_VALIDATE_INHERITED(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)
except module.model_authority.ModelConsumptionAuthorityError as error: old_error=str(error)
if old_error!="capture private-parent FD replay differs": raise RuntimeError("frozen private-parent validator did not reject")
observed=module.validate_inherited_fd_binding_r5f(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)
if observed!=binding: raise RuntimeError("r5f private-parent validator differs")
before=os.lseek(shared_fd,0,os.SEEK_CUR);payloads=[module.read_fd_with_pread_r5f(shared_fd) for _ in range(32)];after=os.lseek(shared_fd,0,os.SEEK_CUR)
if before!=after or any(hashlib.sha256(raw).hexdigest()!=shared_sha for raw in payloads): raise RuntimeError("shared OFD pread differs")
p0_root=module.Path(admission_cache_root)/("activation-import-p0-rank-%d"%rank);p0_root.mkdir(mode=0o700);site=p0_root/"site";site.mkdir(mode=0o700);bernini_live=p0_root/"bernini-live";bernini_live.mkdir(mode=0o700);veomni_live=p0_root/"veomni-live";veomni_live.mkdir(mode=0o700)
def p0_write(path,raw):
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 try:
  offset=0
  while offset<len(raw):
   written=os.write(fd,raw[offset:])
   if written<=0: raise RuntimeError("P0 fixture write made no progress")
   offset+=written
  os.fchmod(fd,0o400)
 finally: os.close(fd)
for dependency in ("torch","diffusers","peft","transformers"): p0_write(site/(dependency+".py"),("ORIGIN="+repr(dependency)+"\n").encode("utf-8"))
bernini_files={"bernini/__init__.py":b"P0_PACKAGE='captured'\n","bernini/pipeline.py":b"P0_MARKER='captured-before-activation-return'\n","configs/bernini_renderer_wan21_1p3b/config.json":b"{\"p0\":true}\n"};veomni_files={"veomni/__init__.py":b"P0_PACKAGE='captured'\n"}
def p0_capture(root_value,expected_commit,scopes,label):
 root=module.Path(root_value).resolve(strict=True)
 if label=="Bernini": files=bernini_files;directories=("bernini","configs","configs/bernini_renderer_wan21_1p3b");expected_root=bernini_live
 elif label=="VeOmni": files=veomni_files;directories=("veomni",);expected_root=veomni_live
 else: raise RuntimeError("P0 capture label differs")
 if root!=expected_root or tuple(scopes)!=(module.base._BERNINI_TREE_SCOPES if label=="Bernini" else module.base._VEOMNI_TREE_SCOPES): raise RuntimeError("P0 capture request differs")
 return module.base._CapturedVendorTree(label=label,live_root=root,expected_commit=expected_commit,scopes=tuple(scopes),directories=directories,file_modes={key:0o644 for key in files},file_git_blobs={key:"0"*40 for key in files},file_sha256={key:hashlib.sha256(raw).hexdigest() for key,raw in files.items()},file_bytes=dict(files),closure_digest=hashlib.sha256(("p0:"+label+":"+expected_commit).encode("utf-8")).hexdigest())
class P0Redirect:
 def __init__(self): self.closed=False
 def install(self,value): raise RuntimeError("P0 renderer redirect unexpectedly installed")
 def finalize_authority(self):
  if self.closed: raise RuntimeError("P0 renderer redirect closed early")
  return {"schema_version":"case01-r5f-v4-cpu-p0-redirect","authority_digest":"b"*64}
 def restore_and_close(self):
  if self.closed: raise RuntimeError("P0 renderer redirect closed twice")
  self.closed=True
p0_redirects=[]
def p0_make_redirect(raw,logical_directory):
 if raw!=bernini_files["configs/bernini_renderer_wan21_1p3b/config.json"] or logical_directory.name!="bernini_renderer_wan21_1p3b": raise RuntimeError("P0 renderer redirect request differs")
 value=P0Redirect();p0_redirects.append(value);return value
p0_third_party={}
def p0_preload(site_root):
 if module.Path(site_root)!=site or sys.path.count(str(site))!=1 or sys.path[-1]!=str(site): raise RuntimeError("P0 preload path differs")
 for dependency in ("torch","diffusers","peft","transformers"):
  value=types.ModuleType(dependency);value.__file__=str(site/(dependency+".py"));value.__package__="";value.__cached__=None;sys.modules[dependency]=value;p0_third_party[dependency]=value
 return {"schema_version":"case01-r5f-v4-cpu-p0-preload","authority_digest":"a"*64}
p0_base=module.base;p0_trainer=p0_base.infer_lora.trainer;p0_validate=p0_trainer.validate_source_trees;p0_activate=p0_trainer.activate_source_trees;p0_preload_original=p0_base._preload_pinned_dependencies;p0_capture_original=p0_base._capture_git_vendor_tree;p0_redirect_original=p0_base._create_sealed_renderer_config_redirect;p0_rank_cache_original=p0_base._EARLY_RANK_CACHE;p0_path_before=list(sys.path);p0_importer_before=dict(sys.path_importer_cache);p0_meta_owner=sys.meta_path;p0_meta_before=list(sys.meta_path);p0_observation={};p0_authority=None
def p0_unscoped_validate(*arguments,**keywords): raise RuntimeError("unscoped P0 validator executed")
def p0_active_callback(bernini_value,veomni_value):
 roots=[str(bernini_value),str(veomni_value)]
 for value in roots:
  while value in sys.path: sys.path.remove(value)
 sys.path[0:0]=roots
 imported=p0_base.importlib.import_module("bernini.pipeline");spec=getattr(imported,"__spec__",None);loader=getattr(imported,"__loader__",None);finder_count=sum(isinstance(value,p0_base._CapturedVendorFinder) for value in sys.meta_path)
 if not sys.meta_path or not isinstance(sys.meta_path[0],p0_base._CapturedVendorFinder) or not isinstance(loader,p0_base._CapturedVendorLoader) or spec is None or getattr(spec,"loader",None) is not loader or getattr(imported,"__cached__","non-none") is not None or finder_count!=1 or getattr(imported,"P0_MARKER",None)!="captured-before-activation-return": raise RuntimeError("activation-time captured import differs")
 p0_observation.update({"activation_callback_import_module":"bernini.pipeline","activation_import_before_callback_return":True,"captured_vendor_finder_preinstalled":True,"captured_vendor_finder_count":finder_count,"captured_vendor_loader_type":type(loader).__name__,"captured_vendor_spec_loader_type":type(spec.loader).__name__,"captured_vendor_loader_is_spec_loader":spec.loader is loader,"captured_vendor_cached_is_none":imported.__cached__ is None})
try:
 p0_rank_cache=p0_root/"rank-cache";p0_rank_cache.mkdir(mode=0o700);p0_base._EARLY_RANK_CACHE=p0_rank_cache;p0_base._preload_pinned_dependencies=p0_preload;p0_base._capture_git_vendor_tree=p0_capture;p0_base._create_sealed_renderer_config_redirect=p0_make_redirect;p0_trainer.validate_source_trees=p0_unscoped_validate;p0_trainer.activate_source_trees=p0_active_callback;sys.path.append(str(site))
 with p0_base.pinned_dependency_import_paths(site) as p0_authority:
  values=p0_trainer.validate_source_trees(bernini_live,veomni_live,expected_bernini_commit=p0_trainer.BERNINI_OFFICIAL_COMMIT,expected_veomni_commit=p0_trainer.VEOMNI_TESTED_COMMIT);p0_trainer.activate_source_trees(values[0],values[1]);p0_base.importlib.import_module("veomni")
 if set(p0_observation)!={"activation_callback_import_module","activation_import_before_callback_return","captured_vendor_finder_preinstalled","captured_vendor_finder_count","captured_vendor_loader_type","captured_vendor_spec_loader_type","captured_vendor_loader_is_spec_loader","captured_vendor_cached_is_none"} or p0_authority.get("activation_call_count")!=1 or [row.get("module") for row in p0_authority.get("loaded_vendor_modules",[])]!=["bernini","bernini.pipeline","veomni"] or len(p0_redirects)!=1 or not p0_redirects[0].closed: raise RuntimeError("P0 activation authority closure differs")
finally:
 p0_trainer.validate_source_trees=p0_validate;p0_trainer.activate_source_trees=p0_activate;p0_base._preload_pinned_dependencies=p0_preload_original;p0_base._capture_git_vendor_tree=p0_capture_original;p0_base._create_sealed_renderer_config_redirect=p0_redirect_original;p0_base._EARLY_RANK_CACHE=p0_rank_cache_original;sys.path[:]=p0_path_before;sys.path_importer_cache.clear();sys.path_importer_cache.update(p0_importer_before)
 for dependency,value in p0_third_party.items():
  if sys.modules.get(dependency) is value: sys.modules.pop(dependency,None)
 for name in tuple(sys.modules):
  if name in ("bernini","veomni") or name.startswith(("bernini.","veomni.")): sys.modules.pop(name,None)
 if sys.meta_path is not p0_meta_owner or sys.meta_path!=p0_meta_before or p0_trainer.validate_source_trees is not p0_validate or p0_trainer.activate_source_trees is not p0_activate: raise RuntimeError("P0 activation state restoration differs")
with module.held_object_sources() as held:
 inner,legacy_source,outer_source,inner_source=held
 cli,legacy_argv=inner.peel_object_oracle_cli(["--object-oracle-arm=route_off","--object-oracle-scaffold=/not-opened","--output","/logical/not-published.mp4"])
 if cli.arm!="route_off" or legacy_argv!=["--output","/logical/not-published.mp4"]: raise RuntimeError("object CLI peel differs")
 class Assets:
  def __init__(self): self.cli=cli
  def producer_hashes(self): return {"wrapper_source_sha256":module.OBJECT_WRAPPER_INNER_SHA256,"legacy_infer_lora_source_sha256":module.base.INFER_LORA_SHA256,"projection_source_sha256":"a"*64,"scaffold_source_sha256":"b"*64}
 assets=Assets();module._bind_composite_producer_hashes(assets,inner_authority=inner_source,composite_authority=outer_source)
 legacy=module.base.infer_lora
 if legacy is not sys.modules.get("infer_lora") or "_bernini_full644_r5_infer_lora_acc46" in sys.modules: raise RuntimeError("legacy module origin differs")
 original_receipt=legacy.build_inference_receipt;original_encoded=legacy._create_retained_encoded_output;original_atomic=legacy._atomic_write_json
 paths=types.SimpleNamespace(logical_output=module.Path("/logical/not-published.mp4"),logical_receipt=module.Path("/logical/not-published.mp4.receipt.json"),runtime_output=module.Path("/proc/self/fd/999/not-published.mp4"),runtime_receipt=module.Path("/proc/self/fd/999/not-published.mp4.receipt.json"),task_fd=999)
 with inner._patched_legacy(legacy,assets):
  if legacy.build_inference_receipt is original_receipt: raise RuntimeError("object patch missed base module")
  with module.base.translated_publication(paths,inference_module=legacy):
   if legacy._create_retained_encoded_output is original_encoded or legacy._atomic_write_json is original_atomic: raise RuntimeError("publication patch missed base module")
 if legacy.build_inference_receipt is not original_receipt or legacy._create_retained_encoded_output is not original_encoded or legacy._atomic_write_json is not original_atomic: raise RuntimeError("composite patch restoration differs")
 hashes=assets.producer_hashes()
 if hashes.get("wrapper_source_sha256")!=composite_sha or hashes.get("object_wrapper_inner_source_sha256")!=module.OBJECT_WRAPPER_INNER_SHA256 or hashes.get("legacy_infer_lora_source_sha256")!=module.base.INFER_LORA_SHA256: raise RuntimeError("composite producer hashes differ")
legacy_instances=sum(1 for value in sys.modules.values() if value is module.base.infer_lora)
if legacy_instances!=1 or "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules) or any(name in sys.modules for name in ("imageio","imageio_ffmpeg","diffusers")): raise RuntimeError("CPU no-renderer module closure differs")
row={"rank":rank,"pid":os.getpid(),"private_parent_fd_number":private_fd,"private_parent_replacement_inode":replacement.st_ino,"pread_bytes_sha256":shared_sha,"pread_offset_before":before,"pread_offset_after":after,**p0_observation}
row["rank_digest"]=digest(row)
print(canonical(row).decode("utf-8"))
'''


_ROOT_TEMPLATE = r'''import base64,hashlib,json,os,shutil,stat,subprocess,sys,types
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate compute JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def ident(info): return [info.st_dev,info.st_ino,info.st_uid,info.st_gid,info.st_mode,info.st_nlink,info.st_rdev,info.st_size,getattr(info,"st_blocks",0),info.st_mtime_ns,info.st_ctime_ns]
def read_fd(fd,size):
 chunks=[];offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  chunks.append(block);offset+=len(block)
 raw=b"".join(chunks)
 if len(raw)!=size or os.pread(fd,1,size)!=b"": raise RuntimeError("compute held read differs")
 return raw
def write_all(fd,payload):
 offset=0
 while offset<len(payload):
  written=os.write(fd,payload[offset:])
  if written<=0: raise RuntimeError("compute write made no progress")
  offset+=written
def replay(role,row,executable=False):
 if type(row) is not dict or set(row)!={"path","sha256","size","identity","mode","nlink"}: raise RuntimeError("compute identity row differs: "+role)
 path=row["path"];fd=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
 try:
  before=os.fstat(fd);raw=read_fd(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 except BaseException: os.close(fd);raise
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named) or ident(before)!=row["identity"] or stat.S_IMODE(before.st_mode)!=row["mode"] or before.st_nlink!=row["nlink"] or len(raw)!=row["size"] or hashlib.sha256(raw).hexdigest()!=row["sha256"] or (executable and not before.st_mode&0o111): os.close(fd);raise RuntimeError("compute identity replay differs: "+role)
 return fd,raw
if len(sys.argv)!=8: raise RuntimeError("compute root argv differs")
python_fd=int(sys.argv[1]);release_b64=sys.argv[2];release_digest=sys.argv[3];bootstrap_sha=sys.argv[4];job_id=sys.argv[5];step_id=sys.argv[6];node=sys.argv[7]
if sys.flags.isolated!=1 or sys.flags.no_site!=1 or not sys.dont_write_bytecode or job_id!="143808" or node!="auh7-1b-gpu-292" or not step_id.isdecimal() or str(int(step_id))!=step_id or int(step_id)<=0: raise RuntimeError("compute root identity differs")
release_raw=base64.b64decode(release_b64.encode("ascii"),validate=True);release=json.loads(release_raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if release_raw!=canonical(release) or digest(release)!=release_digest or release.get("schema_version")!="case01-object-trajectory-exact5-r5f-v4-composite-cpu-release-v2" or release.get("production_rank_cache")!="/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-rank-cache" or release.get("root_bootstrap_sha256")!=bootstrap_sha or release.get("child_bootstrap_sha256")!=hashlib.sha256(base64.b64decode("__CHILD_B64__".encode("ascii"),validate=True)).hexdigest(): raise RuntimeError("compute release differs")
if os.path.lexists(release["production_rank_cache"]): raise RuntimeError("production rank cache touched before admission")
roles=("runner","legacy_exact5_runner","object_eval","legacy_exact5_eval","frozen_runner","bridge","adapter","object_wrapper_inner","legacy_infer_alias","trajectory_projection","trajectory_scaffold_module","base_adapter","eval_v1","eval_v2","model_authority","torchrun_source","torchrun_handler_source","torch_local_agent_source","torch_dynamic_rendezvous_source","torch_multiprocessing_api_source","base_model_manifest","r64_checkpoint_manifest","python","ffmpeg","ffprobe","plan")
rows=release.get("identities")
if type(rows) is not dict or set(rows)!=set(roles) or len(rows)!=26: raise RuntimeError("compute exact26 closure differs")
held={};raw={};children=[];model=None;view_fd=-1;shared_fd=-1;cache_root=None
try:
 for role in roles: held[role],raw[role]=replay(role,rows[role],role in {"python","ffmpeg","ffprobe"})
 if ident(os.fstat(python_fd))!=rows["python"]["identity"] or ident(os.stat("/proc/self/exe"))!=rows["python"]["identity"]: raise RuntimeError("compute held Python differs")
 cache_root="/tmp/bernini-case01-object-trajectory-r5f-v4-composite-cpu-job%s-step%s-cache"%(job_id,step_id)
 if os.path.lexists(cache_root): raise RuntimeError("admission cache is not fresh")
 os.mkdir(cache_root,0o700);model_root=os.path.join(cache_root,"model");view_parent=os.path.join(cache_root,"views");os.mkdir(model_root,0o700);os.mkdir(view_parent,0o700)
 authority_module=types.ModuleType("_composite_cpu_model_authority");authority_module.__file__=rows["model_authority"]["path"];authority_module.__package__=None;authority_module.__loader__=None;authority_module.__cached__=None;authority_module.__builtins__=__builtins__;sys.modules[authority_module.__name__]=authority_module;exec(compile(raw["model_authority"],rows["model_authority"]["path"],"exec",dont_inherit=True),authority_module.__dict__)
 manifest_rows=[]
 for index,line in enumerate(raw["base_model_manifest"].decode("utf-8","strict").splitlines()):
  if "  ./" not in line: raise RuntimeError("base manifest syntax differs")
  relative=line.split("  ./",1)[1];path=os.path.join(model_root,relative);parent=os.path.dirname(path);os.makedirs(parent,mode=0o755,exist_ok=True);payload=("composite-cpu-synthetic:%d:%s\n"%(index,relative)).encode("utf-8");fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);write_all(fd,payload);os.fchmod(fd,0o644);os.close(fd);manifest_rows.append(hashlib.sha256(payload).hexdigest()+"  ./"+relative)
 for relative in authority_module.MODEL_RELATIVE_DIRECTORIES:
  path=model_root if relative=="." else os.path.join(model_root,relative);os.chmod(path,0o755)
 synthetic_manifest=os.path.join(cache_root,"synthetic-model.sha256");manifest_raw=("\n".join(manifest_rows)+"\n").encode("utf-8");fd=os.open(synthetic_manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);write_all(fd,manifest_raw);os.fchmod(fd,0o644);os.close(fd)
 view_fd=os.open(view_parent,os.O_RDONLY|os.O_DIRECTORY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0));os.set_inheritable(view_fd,False)
 model=authority_module.ModelAuthority.capture(model_root=authority_module.Path(model_root),manifest_path=authority_module.Path(synthetic_manifest),private_parent=authority_module.Path(view_parent),private_parent_fd=view_fd,view_name="model-fd-view",expected_uid=os.getuid(),expected_gid=os.getgid(),expected_device=None,expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
 binding=authority_module.build_inherited_fd_binding(task_id="composite-cpu-rank",model_capture=model.capture_receipt,adapter_capture=None,task_publication_root=authority_module.task_publication_root_binding(descriptor=view_fd,path=view_parent))
 inherited=authority_module.inherited_fd_numbers(binding);private_fd=model.capture_receipt["private_parent"]["authority_fd"]
 if private_fd in inherited: raise RuntimeError("private-parent entered inherited allowlist")
 shared_path=os.path.join(cache_root,"shared-ofd.bin");shared_payload=bytes(range(251))*4096;fd=os.open(shared_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);write_all(fd,shared_payload);os.close(fd);shared_fd=os.open(shared_path,os.O_RDONLY);os.lseek(shared_fd,13,os.SEEK_SET);shared_sha=hashlib.sha256(shared_payload).hexdigest()
 child=base64.b64decode("__CHILD_B64__".encode("ascii"),validate=True).decode("utf-8","strict")
 composite_fd=held["adapter"]
 environment={"R5F_CAPTURE":canonical(model.capture_receipt).decode("utf-8"),"R5F_BINDING":canonical(binding).decode("utf-8"),"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C","CUDA_VISIBLE_DEVICES":"","HIP_VISIBLE_DEVICES":"","ROCR_VISIBLE_DEVICES":"-1"}
 pass_fds=tuple(sorted(set((python_fd,composite_fd,shared_fd,*inherited))))
 for rank in range(4):
  children.append(subprocess.Popen(["/proc/self/fd/%d"%python_fd,"-I","-S","-B","-c",child,str(composite_fd),rows["adapter"]["path"],rows["adapter"]["sha256"],str(rows["adapter"]["size"]),str(shared_fd),shared_sha,str(rank),cache_root,rows["base_adapter"]["path"],rows["base_adapter"]["sha256"]],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,pass_fds=pass_fds,env=environment,start_new_session=False))
 rank_rows=[]
 for child_process in children:
  stdout,stderr=child_process.communicate(timeout=180)
  if child_process.returncode!=0 or stderr!=b"" or stdout.count(b"\n")!=1: raise RuntimeError("composite CPU child failed: "+stderr.decode("utf-8","replace"))
  rank_rows.append(json.loads(stdout.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token))))
 if [row.get("rank") for row in rank_rows]!=[0,1,2,3] or len({row.get("pid") for row in rank_rows})!=4 or os.lseek(shared_fd,0,os.SEEK_CUR)!=13: raise RuntimeError("four-process shared OFD closure differs")
 model.abort(reason="composite CPU admission complete");model=None
 os.close(view_fd);view_fd=-1;os.close(shared_fd);shared_fd=-1
 completed_cache_root=cache_root;shutil.rmtree(completed_cache_root);cache_root=None
 if os.path.lexists(completed_cache_root) or os.path.lexists(release["production_rank_cache"]): raise RuntimeError("CPU cache terminal closure differs")
 package=release["package"]
 activation_import={"module":"bernini.pipeline","callback_phase":"inside_original_activate_before_return","finder_installed_before_callback":True,"finder_count_per_rank":[row["captured_vendor_finder_count"] for row in rank_rows],"loader_type":"_CapturedVendorLoader","spec_loader_type":"_CapturedVendorLoader","loader_is_spec_loader":True,"cached_is_none":True,"base_adapter_role":"base_adapter","base_adapter_path":rows["base_adapter"]["path"],"base_adapter_sha256":rows["base_adapter"]["sha256"],"rank_count":4}
 receipt={"schema_version":"case01-object-trajectory-exact5-r5f-v4-composite-cpu-admission-v2","status":"PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_HOLD","holder_job_id":job_id,"node":node,"slurm_step_id":step_id,"package":package,"world_size":4,"rank_count":4,"rank_rows":rank_rows,"isolated_runtime":{"python_flags":["-I","-S","-B"],"isolated":1,"no_site":1,"dont_write_bytecode":True,"entry_via_proc_self_fd":True},"private_parent_fd":{"synthetic_model_capture":True,"captured_parent_omitted":True,"captured_parent_closed_or_reused":True,"frozen_validator_rejected":True,"r5f_validator_accepted":True,"r5f_pread_path_exercised":True},"shared_ofd_pread":{"rank_count":4,"all_reads_exact":True,"offsets_unchanged":True},"module_binding":{"module_name":"infer_lora","base_infer_lora_same_object":True,"object_cli_applied_to_base_module":True,"translated_publication_applied_to_base_module":True,"legacy_module_instance_count":1,"duplicate_legacy_module_loaded":False},"activation_import":activation_import,"side_effects":{"gpu_requested":False,"torch_imported":False,"renderer_or_vae_loaded":False,"publication_performed":False},"cache_lifecycle":{"admission_cache_root":completed_cache_root,"admission_cache_fresh":True,"admission_cache_cleanup_performed":True,"admission_cache_absent_terminal":True,"production_rank_cache":release["production_rank_cache"],"production_rank_cache_untouched":True,"production_rank_cache_absent_before_and_after":True},"process_cleanup":{"all_rank_returncodes_zero":True,"rank_processes_zero":True,"torchrun_processes_zero":True,"child_processes_terminal":True},"launch_allowed":False}
 receipt["receipt_digest"]=digest(receipt)
 print(canonical(receipt).decode("utf-8"))
finally:
 cleanup_errors=[]
 for child_process in children:
  try:
   if child_process.poll() is None: child_process.terminate()
   try: child_process.communicate(timeout=5)
   except subprocess.TimeoutExpired:
    child_process.kill();child_process.communicate(timeout=5)
   if child_process.poll() is None: cleanup_errors.append("child-not-terminal")
  except BaseException as error: cleanup_errors.append("child:"+type(error).__name__)
 if model is not None:
  try: model.abort(reason="composite CPU admission failure cleanup")
  except BaseException as error: cleanup_errors.append("model:"+type(error).__name__)
 for fd in (view_fd,shared_fd):
  if fd>=0:
   try: os.close(fd)
   except OSError as error: cleanup_errors.append("fd:"+str(error.errno))
 if cache_root is not None:
  try:
   if os.path.lexists(cache_root): shutil.rmtree(cache_root)
   if os.path.lexists(cache_root): cleanup_errors.append("cache-present")
  except BaseException as error: cleanup_errors.append("cache:"+type(error).__name__)
 for fd in held.values():
  try: os.close(fd)
  except OSError as error: cleanup_errors.append("held:"+str(error.errno))
 if cleanup_errors: raise RuntimeError("compute cleanup differs: "+",".join(cleanup_errors))
'''

ROOT_BOOTSTRAP = _ROOT_TEMPLATE.replace(
    "__CHILD_B64__",
    base64.b64encode(CHILD_BOOTSTRAP.encode("utf-8")).decode("ascii"),
)


def _shell_quote(value: str) -> str:
    if "\x00" in value:
        raise CompositeCPUError("shell value contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_payload(release: Mapping[str, Any]) -> bytes:
    release_b64 = base64.b64encode(canonical(release)).decode("ascii")
    release_digest = object_digest(release)
    bootstrap_sha = hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest()
    python_path = release["identities"]["python"]["path"]
    return f'''#!/bin/bash -p
set -Eeuo pipefail
umask 077
[[ "$-" == *p* ]] || exit 91
[[ "${{SLURM_JOB_ID-}}" == {HOLDER_JOB_ID} ]] || exit 92
[[ "${{SLURM_STEP_ID-}}" =~ ^[1-9][0-9]*$ ]] || exit 93
[[ "${{SLURM_JOB_NODELIST-}}" == {NODE} && "${{SLURM_STEP_NODELIST-}}" == {NODE} ]] || exit 94
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly PINNED_PYTHON={_shell_quote(python_path)}
exec {{PINNED_PYTHON_FD}}<"$PINNED_PYTHON"
[[ "$PINNED_PYTHON_FD" =~ ^[0-9]+$ && -r "/proc/self/fd/$PINNED_PYTHON_FD" ]] || exit 95
exec -c "/proc/self/fd/$PINNED_PYTHON_FD" -I -S -B -c {_shell_quote(ROOT_BOOTSTRAP)} "$PINNED_PYTHON_FD" {_shell_quote(release_b64)} {_shell_quote(release_digest)} {_shell_quote(bootstrap_sha)} "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_STEP_NODELIST"
'''.encode("utf-8")


def build_srun_argv() -> list[str]:
    return [
        SRUN_AUTHORITY["path"], "--jobid=" + HOLDER_JOB_ID,
        "--job-name=case01-object-r5f-v4-composite-cpu",
        "--nodes=1", "--ntasks=1", "--nodelist=" + NODE,
        "--cpus-per-task=" + str(CPUS_PER_TASK), "--mem=" + MEMORY,
        "--gres=none", "--overlap", "--exact", "--kill-on-bad-exit=1",
        "--immediate=10", "--export=NONE", "--time=00:20:00",
        "/bin/bash", "-p", "-s",
    ]


def transport_preflight(command: Sequence[str], payload: bytes) -> dict[str, Any]:
    if type(command) not in {list, tuple} or any(
        type(item) is not str or not item or "\x00" in item for item in command
    ):
        raise CompositeCPUError("exact CPU srun argv differs")
    argv_bytes = sum(len(item.encode("utf-8")) + 1 for item in command)
    nested = len(ROOT_BOOTSTRAP.encode("utf-8"))
    if (
        argv_bytes > MAX_ARGV_BYTES or len(payload) > MAX_STDIN_SIZE
        or nested > 128 * 1024
    ):
        raise CompositeCPUError("CPU admission transport exceeds bound")
    return {
        "argv_bytes": argv_bytes, "argv_bound": MAX_ARGV_BYTES,
        "stdin_bytes": len(payload), "stdin_bound": MAX_STDIN_SIZE,
        "stdin_sha256": hashlib.sha256(payload).hexdigest(),
        "root_bootstrap_bytes": nested,
        "child_bootstrap_bytes": len(CHILD_BOOTSTRAP.encode("utf-8")),
        "production_rank_cache": str(PRODUCTION_RANK_CACHE),
    }


def create_immutable(path: Path, raw: bytes, mode: int) -> None:
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise CompositeCPUError("create-only write made no progress")
            offset += written
        os.fchmod(descriptor, mode); os.fsync(descriptor)
        if read_fd(descriptor, len(raw)) != raw:
            raise CompositeCPUError("create-only replay differs")
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def create_json(path: Path, value: Mapping[str, Any], mode: int) -> bytes:
    raw = canonical(value) + b"\n"; create_immutable(path, raw, mode); return raw


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
        raise CompositeCPUError("CPU admission process-group probe differs") from error


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    except OSError as error:
        if error.errno in (errno.ESRCH, errno.EPERM):
            return
        raise CompositeCPUError("CPU admission process-group signal differs") from error


def _poll_group_absent(
    process: subprocess.Popen[bytes], process_group: int, deadline: float,
) -> bool:
    while True:
        # Reap the owned leader while continuing to probe the PGID saved at
        # spawn.  A descendant may outlive that leader and retain a pipe.
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
        raise CompositeCPUError("CPU admission terminal pipe close differs") from errors[0]


def _seal_process_group(
    process: subprocess.Popen[bytes], process_group: int,
) -> None:
    # Cleanup is keyed by the saved PGID, never by the current leader state.
    # communicate drains all pipes on its normal path; this terminal seal
    # explicitly closes every remaining pipe on error paths before enforcing
    # TERM -> KILL -> direct-child reap -> independently probed ESRCH.
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
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise CompositeCPUError(
                "CPU admission direct child did not reap"
            ) from error

    kill_deadline = time.monotonic() + PROCESS_KILL_GRACE_SECONDS
    if not _poll_group_absent(process, process_group, kill_deadline):
        raise CompositeCPUError("CPU admission process group did not reach ESRCH")
    if process.poll() is None:
        raise CompositeCPUError("CPU admission direct child remains unreaped")
    if pipe_error is not None:
        raise CompositeCPUError("CPU admission terminal pipe seal differs") from pipe_error


def run_srun(command: Sequence[str], payload: bytes) -> tuple[int, bytes, bytes, int]:
    transport_preflight(command, payload)
    process = subprocess.Popen(
        list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "HOME": "/vast/users/guangyi.chen",
             "LANG": "C", "LC_ALL": "C", "BASH_ENV": "/dev/null"},
        close_fds=True, start_new_session=True,
    )
    # start_new_session=True establishes PGID==PID before exec.  Save it
    # immediately so an early-exited leader cannot hide live descendants.
    process_group = process.pid
    stdout = b""
    stderr = b""
    primary_error: BaseException | None = None
    try:
        stdout, stderr = process.communicate(
            input=payload, timeout=SRUN_TIMEOUT_SECONDS,
        )
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
        raise CompositeCPUError(
            "CPU admission process/pipe zero gate differs"
        ) from (primary_error if primary_error is not None else cleanup_error)

    if isinstance(primary_error, subprocess.TimeoutExpired):
        raise CompositeCPUError("single CPU admission srun timed out") from primary_error
    if primary_error is not None:
        raise primary_error
    if process.returncode is None:
        raise CompositeCPUError("CPU admission srun lacks terminal return code")
    if group_present_after_communicate:
        raise CompositeCPUError(
            "terminal CPU admission process group required cleanup"
        )
    return int(process.returncode), stdout, stderr, process_group


def validate_receipt(
    raw: bytes, package: Mapping[str, Any],
) -> dict[str, Any]:
    value = strict_json(raw, label="composite CPU receipt")
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    rows = value.get("rank_rows")
    if (
        set(value) != CPU_RECEIPT_FIELDS
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("status")
        != "PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_HOLD"
        or value.get("holder_job_id") != HOLDER_JOB_ID
        or value.get("node") != NODE
        or type(value.get("slurm_step_id")) is not str
        or not value["slurm_step_id"].isdecimal()
        or str(int(value["slurm_step_id"])) != value["slurm_step_id"]
        or int(value["slurm_step_id"]) <= 0
        or value.get("package") != package
        or value.get("world_size") != 4 or value.get("rank_count") != 4
        or type(rows) is not list or len(rows) != 4
        or value.get("isolated_runtime") != {
            "python_flags": ["-I", "-S", "-B"], "isolated": 1,
            "no_site": 1, "dont_write_bytecode": True,
            "entry_via_proc_self_fd": True,
        }
        or value.get("private_parent_fd") != {
            "synthetic_model_capture": True,
            "captured_parent_omitted": True,
            "captured_parent_closed_or_reused": True,
            "frozen_validator_rejected": True,
            "r5f_validator_accepted": True,
            "r5f_pread_path_exercised": True,
        }
        or value.get("shared_ofd_pread") != {
            "rank_count": 4, "all_reads_exact": True,
            "offsets_unchanged": True,
        }
        or value.get("module_binding") != {
            "module_name": "infer_lora",
            "base_infer_lora_same_object": True,
            "object_cli_applied_to_base_module": True,
            "translated_publication_applied_to_base_module": True,
            "legacy_module_instance_count": 1,
            "duplicate_legacy_module_loaded": False,
        }
        or value.get("activation_import") != {
            "module": "bernini.pipeline",
            "callback_phase": "inside_original_activate_before_return",
            "finder_installed_before_callback": True,
            "finder_count_per_rank": [1, 1, 1, 1],
            "loader_type": "_CapturedVendorLoader",
            "spec_loader_type": "_CapturedVendorLoader",
            "loader_is_spec_loader": True,
            "cached_is_none": True,
            "base_adapter_role": "base_adapter",
            "base_adapter_path": str(
                PACKAGE_ROOT / "release/methods/bernini_action_editing/"
                "full644_exploratory_matched_infer_adapter_v3.py"
            ),
            "base_adapter_sha256": BASE_ADAPTER_SHA256,
            "rank_count": 4,
        }
        or value.get("cache_lifecycle") != {
            "admission_cache_root": (
                "/tmp/bernini-case01-object-trajectory-r5f-v4-composite-cpu-"
                f"job{HOLDER_JOB_ID}-step{value.get('slurm_step_id')}-cache"
            ),
            "admission_cache_fresh": True,
            "admission_cache_cleanup_performed": True,
            "admission_cache_absent_terminal": True,
            "production_rank_cache": str(PRODUCTION_RANK_CACHE),
            "production_rank_cache_untouched": True,
            "production_rank_cache_absent_before_and_after": True,
        }
        or value.get("process_cleanup") != {
            "all_rank_returncodes_zero": True,
            "rank_processes_zero": True,
            "torchrun_processes_zero": True,
            "child_processes_terminal": True,
        }
        or value.get("side_effects") != {
            "gpu_requested": False, "torch_imported": False,
            "renderer_or_vae_loaded": False, "publication_performed": False,
        }
        or value.get("launch_allowed") is not False
    ):
        raise CompositeCPUError("composite CPU receipt semantics differ")
    if [row.get("rank") for row in rows if type(row) is dict] != [0, 1, 2, 3]:
        raise CompositeCPUError("composite CPU rank order differs")
    if len({row.get("pid") for row in rows if type(row) is dict}) != 4:
        raise CompositeCPUError("composite CPU rank PID closure differs")
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != CPU_RANK_ROW_FIELDS:
            raise CompositeCPUError("composite CPU rank schema differs")
        unsigned_row = dict(row); rank_digest = unsigned_row.pop("rank_digest", None)
        if (
            rank_digest != object_digest(unsigned_row)
            or row.get("rank") != index
            or type(row.get("pid")) is not int or row["pid"] <= 1
            or type(row.get("private_parent_fd_number")) is not int
            or row["private_parent_fd_number"] < 3
            or type(row.get("private_parent_replacement_inode")) is not int
            or row["private_parent_replacement_inode"] <= 0
            or row.get("pread_bytes_sha256") != SHARED_OFD_PAYLOAD_SHA256
            or row.get("pread_offset_before") != 13
            or row.get("pread_offset_after") != 13
            or row.get("activation_callback_import_module")
            != "bernini.pipeline"
            or row.get("activation_import_before_callback_return") is not True
            or row.get("captured_vendor_finder_preinstalled") is not True
            or row.get("captured_vendor_finder_count") != 1
            or row.get("captured_vendor_loader_type")
            != "_CapturedVendorLoader"
            or row.get("captured_vendor_spec_loader_type")
            != "_CapturedVendorLoader"
            or row.get("captured_vendor_loader_is_spec_loader") is not True
            or row.get("captured_vendor_cached_is_none") is not True
        ):
            raise CompositeCPUError("composite CPU rank proof differs")
    return value


def controller() -> dict[str, Any]:
    authorities: list[HeldAuthority] = []
    root: HeldDirectory | None = None
    try:
        # Literal receipt-first boundary: no package root or target is named yet.
        publication_held = open_authority(
            PUBLICATION_PATH, expected_sha256=PUBLICATION_SHA256,
            expected_size=PUBLICATION_SIZE, expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(publication_held)
        materialization_held = open_authority(
            MATERIALIZATION_PATH, expected_sha256=MATERIALIZATION_SHA256,
            expected_size=MATERIALIZATION_SIZE, expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(materialization_held)
        package_controller_held = open_authority(
            PACKAGE_CONTROLLER_PATH, expected_sha256=PACKAGE_CONTROLLER_SHA256,
            expected_size=PACKAGE_CONTROLLER_SIZE, expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(package_controller_held)
        publication, report, package_controller = validate_package_receipts(
            publication_held, materialization_held, package_controller_held,
        )
        root = open_directory(PACKAGE_ROOT, PACKAGE_ROOT_IDENTITY)
        identities: dict[str, HeldAuthority] = {}
        for role in IDENTITY_ROLES:
            row = report["production"]["identities"][role]
            held = open_authority(
                Path(row["path"]), expected_sha256=row["sha256"],
                expected_size=row["size"], expected_mode=None,
                maximum_size=(
                    MAX_EXECUTABLE_SIZE if role in {"python", "ffmpeg", "ffprobe"}
                    else MAX_SOURCE_SIZE
                ), executable=role in {"python", "ffmpeg", "ffprobe"},
                expected_uid=None, expected_gid=None,
            )
            identities[role] = held; authorities.append(held)
        srun = open_authority(
            Path(SRUN_AUTHORITY["path"]),
            expected_sha256=SRUN_AUTHORITY["sha256"],
            expected_size=SRUN_AUTHORITY["size"], expected_mode=0o755,
            maximum_size=MAX_SOURCE_SIZE, executable=True,
            expected_uid=0, expected_gid=0,
        ); authorities.append(srun)
        if any(os.path.lexists(path) for path in (
            ATTEMPT_PATH, RECEIPT_PATH, EVIDENCE_PATH, STDOUT_PATH, STDERR_PATH,
        )):
            raise CompositeCPUError("fresh composite CPU target differs")
        package_block = {
            "root": str(PACKAGE_ROOT), "root_identity": PACKAGE_ROOT_IDENTITY,
            "publication_receipt_sha256": hashlib.sha256(
                publication_held.raw
            ).hexdigest(),
            "publication_receipt_digest": PUBLICATION_DIGEST,
            "materialization_receipt_sha256": hashlib.sha256(
                materialization_held.raw
            ).hexdigest(),
            "materialization_receipt_digest": MATERIALIZATION_DIGEST,
            "package_controller_evidence_sha256": hashlib.sha256(
                package_controller_held.raw
            ).hexdigest(),
            "package_controller_evidence_digest": PACKAGE_CONTROLLER_DIGEST,
            "release_file_count": 25,
            "release_manifest_digest": report["release"]["manifest_digest"],
            "production_identity_count": 26,
            "identity_roles": list(IDENTITY_ROLES),
            "identity_set_digest": report["production"]["identity_set_digest"],
            "inner_outer_crosslink": report["production"][
                "inner_outer_crosslink"
            ],
        }
        release = {
            "schema_version": (
                "case01-object-trajectory-exact5-r5f-v4-composite-cpu-release-v2"
            ),
            "package": package_block,
            "identities": {role: identities[role].row() for role in IDENTITY_ROLES},
            "production_rank_cache": str(PRODUCTION_RANK_CACHE),
            "world_size": 4, "gpu_count": 0,
            "root_bootstrap_sha256": hashlib.sha256(
                ROOT_BOOTSTRAP.encode("utf-8")
            ).hexdigest(),
            "child_bootstrap_sha256": hashlib.sha256(
                CHILD_BOOTSTRAP.encode("utf-8")
            ).hexdigest(),
        }
        payload = build_payload(release); command = build_srun_argv()
        transport = transport_preflight(command, payload)
        attempt = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_SINGLE_CPU_SRUN",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "single_srun_attempt": True, "retry_allowed": False,
            "gpu_count": 0, "world_size": 4,
            "package_publication_sha256": PUBLICATION_SHA256,
            "materialization_sha256": MATERIALIZATION_SHA256,
            "package_controller_sha256": PACKAGE_CONTROLLER_SHA256,
            "production_rank_cache": str(PRODUCTION_RANK_CACHE),
            "production_rank_cache_must_remain_absent": True,
            "exact_srun_argv": command,
            "exact_srun_argv_digest": object_digest(command),
            "transport": transport,
        }
        attempt["attempt_digest"] = object_digest(attempt)
        attempt_raw = create_json(ATTEMPT_PATH, attempt, RECEIPT_MODE)
        attempt_held = open_authority(
            ATTEMPT_PATH, expected_sha256=hashlib.sha256(attempt_raw).hexdigest(),
            expected_size=len(attempt_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(attempt_held)
        for authority in authorities:
            authority.replay()
        root.replay()
        returncode, stdout, stderr, process_group = run_srun(command, payload)
        if returncode != 0 or stderr != b"" or stdout.count(b"\n") != 1:
            raise CompositeCPUError(
                "single composite CPU srun failed: "
                + stderr.decode("utf-8", "replace")
            )
        receipt = validate_receipt(stdout, package_block)
        create_immutable(STDOUT_PATH, stdout, RECEIPT_MODE)
        create_immutable(STDERR_PATH, stderr, RECEIPT_MODE)
        create_immutable(RECEIPT_PATH, stdout, RECEIPT_MODE)
        stdout_held = open_authority(
            STDOUT_PATH, expected_sha256=hashlib.sha256(stdout).hexdigest(),
            expected_size=len(stdout), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(stdout_held)
        stderr_held = open_authority(
            STDERR_PATH, expected_sha256=hashlib.sha256(stderr).hexdigest(),
            expected_size=len(stderr), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(stderr_held)
        receipt_held = open_authority(
            RECEIPT_PATH, expected_sha256=hashlib.sha256(stdout).hexdigest(),
            expected_size=len(stdout), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(receipt_held)
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "PASS_FRESH_CANARY_V3_COMPOSITE_CPU_CONTROLLER",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "slurm_step_id": receipt["slurm_step_id"],
            "single_srun_attempt": True, "retry_allowed": False,
            "srun_count": 1, "srun_ntasks": 1,
            "real_rank_process_count": 4,
            "cpus_per_task": CPUS_PER_TASK, "gpu_count": 0,
            "srun_returncode": 0,
            "receipt": receipt_held.bare_row(),
            "receipt_digest": receipt["receipt_digest"],
            "stdout": stdout_held.bare_row(), "stderr": stderr_held.bare_row(),
            "stderr_empty": True, "process_group_zero": True,
            "launch_allowed": False, "renderer_or_vae_loaded": False,
            "publication_performed": False,
        }
        evidence["evidence_digest"] = object_digest(evidence)
        if set(evidence) != CPU_EVIDENCE_FIELDS:
            raise CompositeCPUError("composite CPU evidence schema differs")
        evidence_raw = create_json(EVIDENCE_PATH, evidence, RECEIPT_MODE)
        evidence_held = open_authority(
            EVIDENCE_PATH, expected_sha256=hashlib.sha256(evidence_raw).hexdigest(),
            expected_size=len(evidence_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        ); authorities.append(evidence_held)
        for authority in authorities:
            authority.replay()
        root.replay()
        return evidence
    finally:
        for authority in reversed(authorities):
            try:
                authority.close()
            except OSError:
                pass
        if root is not None:
            root.close()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # State and placeholder checks precede every filesystem/process operation.
    if CONTROLLER_STATE != READY_STATE:
        print("HOLD: composite CPU controller state is not READY", file=sys.stderr)
        return 88
    blocked = blocked_dynamic_pins()
    if blocked:
        print("HOLD: blocked dynamic pins: " + ",".join(blocked), file=sys.stderr)
        return 88
    if values != ["--execute", authorization_token()]:
        print("HOLD: exact composite CPU authorization required", file=sys.stderr)
        return 88
    try:
        value = controller()
    except (OSError, ValueError, CompositeCPUError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 96
    print(canonical(value).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
