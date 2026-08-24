#!/usr/bin/env python3
"""Receipt-first package-root world4 admission controller v3 (HOLD).

The future state-and-pin-only READY copy consumes the final package
publication receipt, immutable materialization report, static receipt, and
captured-root receipt before it admits the package root.  Both login and
compute then reopen the package's wrapper, projection, scaffold module,
scaffold artifact, and world4 probe, plus the exact production Python/Torch
runtime.  Exactly one CPU-only ``srun`` is allowed after a 0400 attempt claim.

The separately deployed CPU-v2 controller is used only as pinned source-code
machinery for staging, seven-scenario world4 validation, bounded process-group
cleanup, and the single ``subprocess.Popen`` call.  This controller never
consumes or trusts the old CPU admission receipt.  It substitutes a live
package receipt gate on both sides of the Slurm boundary and supplies its own
package paths and fresh output namespace.

Production v2 refused before its attempt claim because the real canonical
package plan expands to an 11,553-byte joined ``srun`` argv, beyond the pinned
CPU engine's legacy 8,192-byte local guard.  Read-only postflight proved every
v2 attempt/world4/evidence/log/stage path absent and no Slurm step started.
Those absences are production evidence, not a runtime dependency on the old
namespace.  V3 locally constructs the otherwise exact CPU argv under a
fail-closed 32 KiB transport/argv bound, conservatively below AUH's observed
2,097,152-byte ``ARG_MAX``.

This checked-in HOLD source performs no argv, path, dynamic-pin, directory,
file, process, Slurm, or network I/O.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-package-world4-controller-v3"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
COMPUTE_SCHEMA = SCHEMA + "-compute"
CONTROLLER_STATE = "HOLD_PENDING_V3_WIDE_PLAN_ARGV_REVIEW"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_PACKAGE_WORLD4_ADMISSION_V3"

HOLDER_JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
CPUS_PER_TASK = 16
GPU_COUNT = 0
PER_SCENARIO_TIMEOUT_SECONDS = 30
CONTROLLER_TIMEOUT_SECONDS = 270
REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
MAX_JSON_SIZE = 32 * 1024 * 1024
MAX_SOURCE_SIZE = 2 * 1024 * 1024
MAX_SRUN_TRANSPORT_BYTES = 32_768
OBSERVED_AUH_ARG_MAX = 2_097_152
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
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
PLAN_PATH = (
    PACKAGE_ROOT / "plan/case01_object_trajectory_exact5_r64_HOLD_plan_v1.json"
)
LAUNCH_INPUT_PATH = PACKAGE_ROOT / "launch/root_launch_input_HOLD_v1.json"
LAUNCH_PAYLOAD_PATH = PACKAGE_ROOT / "launch/root_launch_payload_HOLD_v1.sh"
STATIC_RECEIPT_PATH = (
    PACKAGE_ROOT / "evidence/exact5_static_probe_receipt_v1.json"
)
ROOT_FAKE_RECEIPT_PATH = (
    PACKAGE_ROOT / "evidence/exact5_root_fake_runner_probe_receipt_v1.json"
)
WORLD4_RECEIPT_PATH = (
    PACKAGE_ROOT / "evidence/exact5_world4_receipt_v3.json"
)
ATTEMPT_PATH = PACKAGE_ROOT / "evidence/exact5_world4_attempt_v3.json"
EVIDENCE_PATH = (
    PACKAGE_ROOT / "evidence/exact5_world4_controller_receipt_v3.json"
)
STDOUT_PATH = PACKAGE_ROOT / "logs/exact5_world4_srun_v3.stdout.log"
STDERR_PATH = PACKAGE_ROOT / "logs/exact5_world4_srun_v3.stderr.log"
STAGE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-package-world4-"
    "job143808-node292-v3"
)
PUBLICATION_ROOT = STAGE_ROOT / "publication"

# Production v2 was a pre-attempt local-width refusal.  These paths were
# independently observed absent after rc96.  They are documentary evidence
# only: v3 never reads, retries, repairs, or makes admission depend on v2.
V2_REFUSAL_ABSENCE_EVIDENCE_PATHS = (
    PACKAGE_ROOT / "evidence/exact5_world4_attempt_v2.json",
    PACKAGE_ROOT / "evidence/exact5_world4_receipt_v2.json",
    PACKAGE_ROOT / "evidence/exact5_world4_controller_receipt_v2.json",
    PACKAGE_ROOT / "logs/exact5_world4_srun_v2.stdout.log",
    PACKAGE_ROOT / "logs/exact5_world4_srun_v2.stderr.log",
    Path(
        "/tmp/bernini-case01-object-trajectory-package-world4-"
        "job143808-node292-v2"
    ),
)

ENGINE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_world4_cpu_controller_v2/"
    "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)
ENGINE_SHA256 = (
    "9d5aebcdf4b7938848e0763b839010fbd58df196f8a0155515b05a032cc99cbd"
)
ENGINE_SIZE = 86_998
ENGINE_STATE = "READY_EXPLICIT_SINGLE_SRUN_CPU_ADMISSION"

PACKAGE_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v2-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v1"
)
STATIC_SCHEMA = "case01-object-trajectory-exact5-static-admission-v1"
ROOT_FAKE_SCHEMA = "case01-object-trajectory-exact5-root-fake-admission-v4"
LAUNCH_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-receipt-auh-v1"
)
LAUNCH_RELEASE_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-release-auh-v1"
)
PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"

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

PACKAGE_PROJECT_AUTHORITIES = {
    "wrapper": {
        "relative": "release/methods/bernini_action_editing/"
        "infer_case01_object_trajectory_oracle_v1.py",
        "sha256": "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        "size": 74_281,
    },
    "projection": {
        "relative": "release/methods/bernini_action_editing/"
        "object_trajectory_projection_v1.py",
        "sha256": "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e",
        "size": 47_588,
    },
    "scaffold_module": {
        "relative": "release/methods/bernini_action_editing/"
        "case01_oracle_object_trajectory_v1.py",
        "sha256": "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a",
        "size": 35_803,
    },
    "scaffold": {
        "relative": "authority/conditions/trajectory_scaffold.json",
        "sha256": "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a",
        "size": 54_801,
    },
    "world4": {
        "relative": "diagnostics/"
        "case01_object_trajectory_exact5_world4_probe_v1.py",
        "sha256": "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
        "size": 54_489,
    },
}

VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
TORCH_ROOT = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch"
)
RUNTIME_AUTHORITIES = {
    "python": {
        "path": str(VACE_PYTHON),
        "sha256": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
        "size": 31_490_256, "executable": True,
    },
    "torchrun_source": {
        "path": str(TORCH_ROOT / "distributed/run.py"),
        "sha256": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
        "size": 31_587, "executable": False,
    },
    "torchrun_handler_source": {
        "path": str(TORCH_ROOT / "distributed/elastic/multiprocessing/"
                    "subprocess_handler/subprocess_handler.py"),
        "sha256": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
        "size": 2_436, "executable": False,
    },
    "torch_local_agent_source": {
        "path": str(TORCH_ROOT / "distributed/elastic/agent/server/"
                    "local_elastic_agent.py"),
        "sha256": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
        "size": 16_741, "executable": False,
    },
    "torch_dynamic_rendezvous_source": {
        "path": str(TORCH_ROOT / "distributed/elastic/rendezvous/"
                    "dynamic_rendezvous.py"),
        "sha256": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
        "size": 49_422, "executable": False,
    },
    "torch_multiprocessing_api_source": {
        "path": str(TORCH_ROOT / "distributed/elastic/multiprocessing/api.py"),
        "sha256": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
        "size": 33_740, "executable": False,
    },
}

SRUN_AUTHORITY = {
    "path": "/usr/bin/srun",
    "sha256": "2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e",
    "size": 164_720,
}
CPU_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUESTED_SRUN_GPU_EXPORT = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "-1",
}
EXPECTED_COMPUTE_GPU_VISIBILITY = {
    "CUDA_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": None,
}

# Final immutable AUH package/static/root-fake authority pins.  The checked-in
# source remains HOLD until the reviewed state-only READY copy is created.
PACKAGE_PUBLICATION_RECEIPT_SHA256 = (
    "b3766694f24ead6d7da04e5a1da077de69a9dbbf06df8f06ff0c9db77d84c533"
)
PACKAGE_PUBLICATION_RECEIPT_SIZE: int | str = 2_209
PACKAGE_PUBLICATION_RECEIPT_DIGEST = (
    "5cab7d2db0079d4b6960273e681c20b60941b892c3a42bfdbd70be819d991cb9"
)
MATERIALIZATION_REPORT_SHA256 = (
    "e1e4d7ae266828f27f77f39528672cd7ccae9aa067fdee291d4e5e32f9a9bf2f"
)
MATERIALIZATION_REPORT_SIZE: int | str = 21_743
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
ROOT_FAKE_RECEIPT_SHA256 = (
    "af4cb28c23bc9e7a8355133f2068d02af5f97eda16083fa8b591e5131062f619"
)
ROOT_FAKE_RECEIPT_SIZE: int | str = 1_975
ROOT_FAKE_RECEIPT_DIGEST = (
    "4a65b5dab48904fced093fd0bff0c16a50c13b5caa30b6a70dc4e4ae9c6b170a"
)
PACKAGE_ROOT_IDENTITY: list[int] | str = (
    [48, 12038280342419913116, 2012, 2000, 16832, 2, 0, 4096, 0,
     1787357728317453482, 1787357728652385810]
)


class PackageWorld4Error(RuntimeError):
    """The one-shot package world4 authority closure differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PackageWorld4Error("value is not canonical JSON") from error


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


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise PackageWorld4Error("held read size differs")
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
        raise PackageWorld4Error("held read is incomplete")
    return raw


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PackageWorld4Error(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise PackageWorld4Error(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise PackageWorld4Error(f"noncanonical JSON authority: {label}")
    return value


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...],
        raw: bytes,
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
            raise PackageWorld4Error(f"held authority changed: {self.path}")

    def row(self) -> dict[str, Any]:
        info = os.fstat(self.descriptor)
        return {
            "path": str(self.path),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw), "device": info.st_dev,
            "inode": info.st_ino, "uid": info.st_uid, "gid": info.st_gid,
            "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
            "rdev": info.st_rdev, "blocks": getattr(info, "st_blocks", 0),
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        }

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class HeldDirectory:
    def __init__(self, path: Path, descriptor: int, held: tuple[int, ...]) -> None:
        self.path = path
        self.descriptor = descriptor
        self.held_identity = held

    def replay(self) -> None:
        if (
            identity(os.fstat(self.descriptor)) != self.held_identity
            or identity(os.lstat(self.path)) != self.held_identity
        ):
            raise PackageWorld4Error("held package root changed")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def open_authority(
    path: Path, *, expected_sha256: str, expected_size: int,
    expected_mode: int, executable: bool = False,
    maximum_size: int = MAX_SOURCE_SIZE,
) -> HeldAuthority:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or SHA_RE.fullmatch(str(expected_sha256)) is None
        or type(expected_size) is not int or not (0 < expected_size <= maximum_size)
    ):
        raise PackageWorld4Error(f"noncanonical authority pin: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or named.st_uid != REMOTE_UID or named.st_gid != REMOTE_GID
        or stat.S_IMODE(named.st_mode) != expected_mode
        or named.st_size != expected_size
        or path.resolve(strict=True) != path
        or (executable and not named.st_mode & 0o111)
    ):
        raise PackageWorld4Error(f"named authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = read_fd(descriptor, before.st_size)
        replay = read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            identity(before) != identity(named)
            or identity(before) != identity(after)
            or identity(before) != identity(named_after)
            or raw != replay
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise PackageWorld4Error(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), raw)
    except BaseException:
        os.close(descriptor)
        raise


def open_observed_authority(
    path: Path, *, expected_mode: int, maximum_size: int,
) -> HeldAuthority:
    """Hold a named controller whose digest is recorded, not predeclared."""
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or type(maximum_size) is not int or maximum_size <= 0
    ):
        raise PackageWorld4Error("noncanonical observed authority")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or named.st_uid != REMOTE_UID or named.st_gid != REMOTE_GID
        or stat.S_IMODE(named.st_mode) != expected_mode
        or not (0 < named.st_size <= maximum_size)
        or path.resolve(strict=True) != path
    ):
        raise PackageWorld4Error(f"named observed authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = read_fd(descriptor, before.st_size)
        replay = read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            identity(before) != identity(named)
            or identity(before) != identity(after)
            or identity(before) != identity(named_after)
            or raw != replay
        ):
            raise PackageWorld4Error(f"observed authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), raw)
    except BaseException:
        os.close(descriptor)
        raise


def open_package_root(expected_identity: Sequence[int]) -> HeldDirectory:
    if (
        type(expected_identity) is not list or len(expected_identity) != 11
        or any(type(value) is not int for value in expected_identity)
    ):
        raise PackageWorld4Error("package root identity pin differs")
    descriptor = os.open(
        PACKAGE_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(PACKAGE_ROOT)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != REMOTE_UID or opened.st_gid != REMOTE_GID
            or identity(opened) != tuple(expected_identity)
            or identity(named) != tuple(expected_identity)
            or PACKAGE_ROOT.resolve(strict=True) != PACKAGE_ROOT
        ):
            raise PackageWorld4Error("package root identity differs")
        return HeldDirectory(PACKAGE_ROOT, descriptor, identity(opened))
    except BaseException:
        os.close(descriptor)
        raise


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "package_publication_receipt_sha256": PACKAGE_PUBLICATION_RECEIPT_SHA256,
        "package_publication_receipt_size": PACKAGE_PUBLICATION_RECEIPT_SIZE,
        "package_publication_receipt_digest": PACKAGE_PUBLICATION_RECEIPT_DIGEST,
        "materialization_report_sha256": MATERIALIZATION_REPORT_SHA256,
        "materialization_report_size": MATERIALIZATION_REPORT_SIZE,
        "materialization_report_digest": MATERIALIZATION_REPORT_DIGEST,
        "static_receipt_sha256": STATIC_RECEIPT_SHA256,
        "static_receipt_size": STATIC_RECEIPT_SIZE,
        "static_receipt_digest": STATIC_RECEIPT_DIGEST,
        "root_fake_receipt_sha256": ROOT_FAKE_RECEIPT_SHA256,
        "root_fake_receipt_size": ROOT_FAKE_RECEIPT_SIZE,
        "root_fake_receipt_digest": ROOT_FAKE_RECEIPT_DIGEST,
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
        "engine": {"path": str(ENGINE_PATH), "sha256": ENGINE_SHA256,
                   "size": ENGINE_SIZE},
        "outputs": {
            "attempt": str(ATTEMPT_PATH), "world4": str(WORLD4_RECEIPT_PATH),
            "evidence": str(EVIDENCE_PATH),
        },
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
        "rename_returned_zero": True, "rename_error_errno": None,
        "parent_fsync_returned_zero": True, "parent_fsync_error_errno": None,
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
        or any(
            SHA_RE.fullmatch(str(value.get(key))) is None
            for key in (
                "source_snapshot_manifest_sha256",
                "source_snapshot_manifest_digest",
                "source_staging_receipt_sha256",
                "source_staging_receipt_digest",
            )
        )
    ):
        raise PackageWorld4Error("package publication receipt differs")
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


def _simple_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("path", "sha256", "size")}


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
    identities = release.get("identities") if type(release) is dict else None
    identity_roles = (
        release.get("identity_roles") if type(release) is dict else None
    )
    artifacts = value.get("artifacts")
    source_snapshot = value.get("source_snapshot")
    source_staging = value.get("source_staging_receipt_authority")
    plan = value.get("plan")
    launch_input = launch.get("input") if type(launch) is dict else None
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
        or type(artifacts) is not dict or len(artifacts) != 28
        or any(
            type(row) is not dict or set(row) != {"sha256", "size"}
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            for row in artifacts.values()
        )
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
        or release.get("campaign_mode") != CAMPAIGN
        or release.get("selected_task_ids") != list(TASK_IDS)
        or release.get("launch_allowed") is not False
        or release.get("ready_overlay_required") is not True
        or release.get("named_payload_execution_forbidden") is not True
        or release_claimed != object_digest(release_unsigned)
        or identity_roles != list(IDENTITY_ROLES)
        or type(identities) is not dict or set(identities) != set(IDENTITY_ROLES)
        or any(
            type(row) is not dict or set(row) != {"path", "sha256", "size"}
            or type(row.get("path")) is not str
            or not os.path.isabs(row["path"])
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            for row in identities.values()
        )
        or type(launch_input) is not dict
        or set(launch_input) != {"path", "sha256", "size", "mode", "nlink"}
        or launch_input.get("path") != str(LAUNCH_INPUT_PATH)
        or SHA_RE.fullmatch(str(launch_input.get("sha256"))) is None
        or type(launch_input.get("size")) is not int
        or launch_input["size"] <= 0
        or launch_input.get("mode") != FILE_MODE
        or launch_input.get("nlink") != 1
        or release.get("input_sha256") != launch_input.get("sha256")
        or identities.get("plan") != {
            "path": plan.get("path"), "sha256": plan.get("sha256"),
            "size": identities.get("plan", {}).get("size"),
        }
    ):
        raise PackageWorld4Error("package materialization report differs")
    for role, runtime in RUNTIME_AUTHORITIES.items():
        row = identities.get(role)
        if type(row) is not dict or row != {
            "path": runtime["path"], "sha256": runtime["sha256"],
            "size": runtime["size"],
        }:
            raise PackageWorld4Error(
                f"materialized runtime authority differs: {role}"
            )
    for role in ("wrapper", "projection", "scaffold_module"):
        expected = PACKAGE_PROJECT_AUTHORITIES[role]
        row = artifacts.get(expected["relative"])
        if row != {"sha256": expected["sha256"], "size": expected["size"]}:
            raise PackageWorld4Error(
                f"materialized project authority differs: {role}"
            )
    world4 = PACKAGE_PROJECT_AUTHORITIES["world4"]
    if artifacts.get(world4["relative"]) != {
        "sha256": world4["sha256"], "size": world4["size"],
    }:
        raise PackageWorld4Error("materialized world4 authority differs")
    return value


STATIC_FIELDS = {
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
        set(value) != STATIC_FIELDS
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
        or value.get("launcher_sha256")
        != "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f"
    ):
        raise PackageWorld4Error("static admission receipt differs")
    return value


ROOT_FAKE_FIELDS = {
    "schema_version", "status", "campaign_mode", "launch_allowed",
    "exact_identity_count", "identity_roles", "task_ids", "arm_order",
    "release_digest", "identity_set_digest", "launch_input_sha256",
    "entry_authority_digest", "plan_sha256", "production_runner_sha256",
    "captured_runner_sha256", "all_exact25_named_identities_replayed",
    "captured_runner_outside_exact25", "captured_runner_bytes_compiled",
    "torch_imported", "renderer_imported", "publication_performed",
    "receipt_digest",
}


def validate_root_fake_receipt(
    held: HeldAuthority, report: Mapping[str, Any],
) -> dict[str, Any]:
    value = strict_json(held.raw, label="captured-root admission receipt")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    release = report.get("launch", {}).get("release", {})
    identities = release.get("identities", {})
    input_row = report.get("launch", {}).get("input", {})
    plan = report.get("plan", {})
    if (
        set(value) != ROOT_FAKE_FIELDS
        or hashlib.sha256(held.raw).hexdigest() != ROOT_FAKE_RECEIPT_SHA256
        or len(held.raw) != ROOT_FAKE_RECEIPT_SIZE
        or claimed != ROOT_FAKE_RECEIPT_DIGEST
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != ROOT_FAKE_SCHEMA
        or value.get("status") != "PASS_CAPTURED_ROOT_FAKE_HOLD"
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("launch_allowed") is not False
        or value.get("exact_identity_count") != 25
        or value.get("identity_roles") != list(IDENTITY_ROLES)
        or value.get("task_ids") != list(TASK_IDS)
        or value.get("arm_order") != list(ARM_ORDER)
        or value.get("identity_set_digest") != object_digest(identities)
        or value.get("launch_input_sha256") != input_row.get("sha256")
        or value.get("plan_sha256") != plan.get("sha256")
        or value.get("production_runner_sha256")
        != "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c"
        or value.get("captured_runner_sha256")
        != "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872"
        or value.get("all_exact25_named_identities_replayed") is not True
        or value.get("captured_runner_outside_exact25") is not True
        or value.get("captured_runner_bytes_compiled") is not True
        or value.get("torch_imported") is not False
        or value.get("renderer_imported") is not False
        or value.get("publication_performed") is not False
        or any(
            SHA_RE.fullmatch(str(value.get(key))) is None
            for key in ("release_digest", "entry_authority_digest")
        )
    ):
        raise PackageWorld4Error("captured-root admission receipt differs")
    return value


class HeldPackageGate:
    def __init__(
        self, authorities: Sequence[HeldAuthority], root: HeldDirectory,
        values: Mapping[str, Mapping[str, Any]],
        projects: Mapping[str, HeldAuthority],
    ) -> None:
        self.authorities = list(authorities)
        self.root = root
        self.values = {key: dict(value) for key, value in values.items()}
        self.projects = dict(projects)
        self.closed = False

    def replay(self) -> None:
        if self.closed or len(self.authorities) != 4:
            raise PackageWorld4Error("package gate is not live")
        # Upstream receipts always replay ahead of root/project authorities.
        for authority in self.authorities:
            authority.replay()
        self.root.replay()
        for authority in self.projects.values():
            authority.replay()
        publication = validate_publication_receipt(self.authorities[0])
        report = validate_materialization_report(self.authorities[1], publication)
        static_value = validate_static_receipt(self.authorities[2], report)
        root_fake = validate_root_fake_receipt(self.authorities[3], report)
        if self.values != {
            "publication": publication, "materialization": report,
            "static": static_value, "root_fake": root_fake,
        }:
            raise PackageWorld4Error("held package receipt values changed")

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA + "-package-gate",
            "package_root": str(PACKAGE_ROOT),
            "package_root_identity": list(self.root.held_identity),
            "publication": {
                "path": str(PACKAGE_PUBLICATION_RECEIPT_PATH),
                "sha256": hashlib.sha256(self.authorities[0].raw).hexdigest(),
                "size": len(self.authorities[0].raw),
                "receipt_digest": self.values["publication"]["receipt_digest"],
            },
            "materialization": {
                "path": str(MATERIALIZATION_REPORT_PATH),
                "sha256": hashlib.sha256(self.authorities[1].raw).hexdigest(),
                "size": len(self.authorities[1].raw),
                "receipt_digest": self.values["materialization"]["receipt_digest"],
            },
            "static": {
                "path": str(STATIC_RECEIPT_PATH),
                "sha256": hashlib.sha256(self.authorities[2].raw).hexdigest(),
                "size": len(self.authorities[2].raw),
                "receipt_digest": self.values["static"]["receipt_digest"],
            },
            "root_fake": {
                "path": str(ROOT_FAKE_RECEIPT_PATH),
                "sha256": hashlib.sha256(self.authorities[3].raw).hexdigest(),
                "size": len(self.authorities[3].raw),
                "receipt_digest": self.values["root_fake"]["receipt_digest"],
            },
            "project_authorities": {
                role: _simple_row(authority.row())
                for role, authority in self.projects.items()
            },
            "old_cpu_admission_receipt_consumed": False,
            "launch_allowed": False,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for authority in reversed(tuple(self.projects.values())):
            authority.close()
        for authority in reversed(self.authorities):
            authority.close()
        self.root.close()


def open_package_gate() -> HeldPackageGate:
    """Open publication receipt first, then the complete package chain."""
    authorities: list[HeldAuthority] = []
    projects: dict[str, HeldAuthority] = {}
    root: HeldDirectory | None = None
    try:
        publication_authority = open_authority(
            PACKAGE_PUBLICATION_RECEIPT_PATH,
            expected_sha256=PACKAGE_PUBLICATION_RECEIPT_SHA256,
            expected_size=PACKAGE_PUBLICATION_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(publication_authority)
        publication = validate_publication_receipt(publication_authority)

        report_authority = open_authority(
            MATERIALIZATION_REPORT_PATH,
            expected_sha256=MATERIALIZATION_REPORT_SHA256,
            expected_size=MATERIALIZATION_REPORT_SIZE,
            expected_mode=RECEIPT_MODE, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(report_authority)
        report = validate_materialization_report(report_authority, publication)

        static_authority = open_authority(
            STATIC_RECEIPT_PATH, expected_sha256=STATIC_RECEIPT_SHA256,
            expected_size=STATIC_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(static_authority)
        static_value = validate_static_receipt(static_authority, report)

        root_fake_authority = open_authority(
            ROOT_FAKE_RECEIPT_PATH, expected_sha256=ROOT_FAKE_RECEIPT_SHA256,
            expected_size=ROOT_FAKE_RECEIPT_SIZE, expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(root_fake_authority)
        root_fake = validate_root_fake_receipt(root_fake_authority, report)

        root = open_package_root(PACKAGE_ROOT_IDENTITY)
        for role, expected in PACKAGE_PROJECT_AUTHORITIES.items():
            projects[role] = open_authority(
                PACKAGE_ROOT / expected["relative"],
                expected_sha256=expected["sha256"],
                expected_size=expected["size"], expected_mode=FILE_MODE,
                maximum_size=MAX_SOURCE_SIZE,
            )
        held = HeldPackageGate(
            authorities, root,
            {"publication": publication, "materialization": report,
             "static": static_value, "root_fake": root_fake},
            projects,
        )
        held.replay()
        return held
    except BaseException:
        for authority in reversed(tuple(projects.values())):
            authority.close()
        for authority in reversed(authorities):
            authority.close()
        if root is not None:
            root.close()
        raise


def load_engine() -> tuple[types.ModuleType, HeldAuthority]:
    held = open_authority(
        ENGINE_PATH, expected_sha256=ENGINE_SHA256, expected_size=ENGINE_SIZE,
        expected_mode=FILE_MODE, maximum_size=MAX_SOURCE_SIZE,
    )
    try:
        source = held.raw.decode("utf-8", "strict")
        module = types.ModuleType("_held_case01_package_world4_cpu_engine_v2")
        module.__file__ = str(ENGINE_PATH)
        module.__package__ = None
        exec(compile(source, str(ENGINE_PATH), "exec"), module.__dict__)
        if (
            module.CONTROLLER_STATE != ENGINE_STATE
            or module.READY_STATE != ENGINE_STATE
            or module.HOLDER_JOB_ID != HOLDER_JOB_ID
            or module.NODE != NODE
            or module.CPUS_PER_TASK != CPUS_PER_TASK
            or module.GPU_COUNT != GPU_COUNT
            or module.PER_SCENARIO_TIMEOUT_SECONDS
            != PER_SCENARIO_TIMEOUT_SECONDS
            or module.CONTROLLER_TIMEOUT_SECONDS != CONTROLLER_TIMEOUT_SECONDS
            or module.RUNTIME_AUTHORITIES != RUNTIME_AUTHORITIES
            or module.SRUN_AUTHORITY["path"] != SRUN_AUTHORITY["path"]
            or module.SRUN_AUTHORITY["sha256"] != SRUN_AUTHORITY["sha256"]
            or module.SRUN_AUTHORITY["size"] != SRUN_AUTHORITY["size"]
            or module.CPU_THREAD_ENVIRONMENT != CPU_THREAD_ENVIRONMENT
            or module.REQUESTED_SRUN_GPU_EXPORT != REQUESTED_SRUN_GPU_EXPORT
            or module.EXPECTED_COMPUTE_GPU_VISIBILITY
            != EXPECTED_COMPUTE_GPU_VISIBILITY
        ):
            raise PackageWorld4Error("pinned CPU engine contract differs")
        return module, held
    except BaseException:
        held.close()
        raise


def configure_engine(
    engine: types.ModuleType, gate_factory: Any,
) -> None:
    engine.SCHEMA = SCHEMA
    engine.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    engine.COMPUTE_SCHEMA = COMPUTE_SCHEMA
    engine.SOURCE_ROOT = PACKAGE_ROOT
    engine.SOURCE_RECEIPT_PATH = PACKAGE_PUBLICATION_RECEIPT_PATH
    engine.PROJECT_AUTHORITIES = {
        role: dict(row) for role, row in PACKAGE_PROJECT_AUTHORITIES.items()
    }
    engine.RUNTIME_AUTHORITIES = {
        role: dict(row) for role, row in RUNTIME_AUTHORITIES.items()
    }
    engine.TORCH_ROLES = tuple(
        role for role in RUNTIME_AUTHORITIES if role != "python"
    )
    engine.VACE_PYTHON = VACE_PYTHON
    engine.TORCH_ROOT = TORCH_ROOT
    engine.TARGET_ROOT = PACKAGE_ROOT
    engine.EVIDENCE_DIR = PACKAGE_ROOT / "evidence"
    engine.LOGS_DIR = PACKAGE_ROOT / "logs"
    engine.ATTEMPT_PATH = ATTEMPT_PATH
    engine.WORLD4_RECEIPT_PATH = WORLD4_RECEIPT_PATH
    engine.EVIDENCE_PATH = EVIDENCE_PATH
    engine.STDOUT_PATH = STDOUT_PATH
    engine.STDERR_PATH = STDERR_PATH
    engine.STAGE_ROOT = STAGE_ROOT
    engine.PUBLICATION_ROOT = PUBLICATION_ROOT
    engine.ENVIRONMENT_SOURCE = (
        "package_receipt_gated_slurm_normalized_step143808_node292"
    )
    engine.open_source_stage_gate = gate_factory


def build_package_srun_argv(plan_b64: str) -> list[str]:
    """Build the exact CPU-engine argv under v3's reviewed 32 KiB bound."""
    if not (0 < MAX_SRUN_TRANSPORT_BYTES < OBSERVED_AUH_ARG_MAX):
        raise PackageWorld4Error("v3/AUH argv bounds differ")
    if type(plan_b64) is not str or not plan_b64:
        raise PackageWorld4Error("compute plan transport differs")
    try:
        plan_width = len(plan_b64.encode("ascii", "strict"))
    except UnicodeError as error:
        raise PackageWorld4Error("compute plan transport is not ASCII") from error
    if plan_width >= MAX_SRUN_TRANSPORT_BYTES:
        raise PackageWorld4Error("compute plan transport exceeds v3 bound")
    exported = {**CPU_THREAD_ENVIRONMENT, **REQUESTED_SRUN_GPU_EXPORT}
    export_argument = "--export=" + ",".join(
        key + "=" + value for key, value in exported.items()
    )
    argv = [
        SRUN_AUTHORITY["path"], "--jobid=" + HOLDER_JOB_ID,
        "--job-name=case01-object-world4-cpu-v1", "--nodes=1", "--ntasks=1",
        "--nodelist=" + NODE, "--cpus-per-task=16", "--mem=32G",
        "--gres=none", "--overlap", "--exact", "--kill-on-bad-exit=1",
        "--immediate=10", export_argument, "--time=00:10:00",
        str(VACE_PYTHON), "-I", "-B", "-", "compute", plan_b64,
    ]
    try:
        joined_width = len(" ".join(argv).encode("ascii", "strict"))
    except UnicodeError as error:
        raise PackageWorld4Error("exact srun argv is not ASCII") from error
    if joined_width >= MAX_SRUN_TRANSPORT_BYTES:
        raise PackageWorld4Error("exact srun argv exceeds v3 bound")
    return argv


def _fresh_outputs(engine: types.ModuleType) -> None:
    for directory in (PACKAGE_ROOT / "evidence", PACKAGE_ROOT / "logs"):
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != REMOTE_UID or info.st_gid != REMOTE_GID
            or stat.S_IMODE(info.st_mode) != 0o700
            or directory.resolve(strict=True) != directory
        ):
            raise PackageWorld4Error(f"package output directory differs: {directory}")
    for path in (
        ATTEMPT_PATH, WORLD4_RECEIPT_PATH, EVIDENCE_PATH,
        STDOUT_PATH, STDERR_PATH, STAGE_ROOT,
    ):
        engine._fresh(path)


def compute(plan_b64: str) -> int:
    gate = open_package_gate()
    engine: types.ModuleType | None = None
    engine_authority: HeldAuthority | None = None
    try:
        engine, engine_authority = load_engine()
        configure_engine(engine, lambda: gate)
        return int(engine.compute(plan_b64))
    finally:
        # engine.compute closes the shared gate on success; failure is closed
        # here.  HeldPackageGate.close is deliberately idempotent.
        gate.close()
        if engine_authority is not None:
            engine_authority.close()


def controller() -> dict[str, Any]:
    package_gate: HeldPackageGate | None = None
    engine_authority: HeldAuthority | None = None
    self_authority: HeldAuthority | None = None
    world4_authority: HeldAuthority | None = None
    evidence_authority: HeldAuthority | None = None
    pinned: list[int] = []
    try:
        # Literal receipt-first package authority gate: no TARGET/STAGE/output
        # observation, engine load, runtime open, attempt, or srun precedes it.
        package_gate = open_package_gate()
        engine, engine_authority = load_engine()
        configure_engine(engine, open_package_gate)

        self_path = Path(__file__)
        self_authority = open_observed_authority(
            self_path, expected_mode=FILE_MODE, maximum_size=MAX_SOURCE_SIZE,
        )
        login_project_rows: dict[str, dict[str, Any]] = {}
        for role, authority in package_gate.projects.items():
            login_project_rows[role] = authority.row()
        login_runtime_rows: dict[str, dict[str, Any]] = {}
        for role, expected in RUNTIME_AUTHORITIES.items():
            descriptor, _raw, row = engine._open_pinned(
                Path(expected["path"]), expected["sha256"], expected["size"],
                executable=bool(expected["executable"]),
            )
            pinned.append(descriptor)
            login_runtime_rows[role] = row
        srun_fd, _srun_raw, srun_row = engine._open_pinned(
            Path(SRUN_AUTHORITY["path"]), SRUN_AUTHORITY["sha256"],
            SRUN_AUTHORITY["size"], executable=True,
        )
        pinned.append(srun_fd)

        package_gate.replay()
        _fresh_outputs(engine)
        plan = engine.build_compute_plan(package_gate.evidence())
        plan_b64 = base64.b64encode(engine.canonical(plan)).decode("ascii")
        command = build_package_srun_argv(plan_b64)
        if (
            command.count(SRUN_AUTHORITY["path"]) != 1
            or "--gres=none" not in command
            or "--ntasks=1" not in command
            or "--nodes=1" not in command
            or "--nodelist=" + NODE not in command
        ):
            raise PackageWorld4Error("single CPU-only srun argv differs")

        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_SINGLE_SRUN",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "cpus_per_task": CPUS_PER_TASK, "gpu_count": GPU_COUNT,
            "single_srun_attempt": True, "retry_allowed": False,
            "package_gate": package_gate.evidence(),
            "controller": self_authority.row(),
            "held_stdin_sha256": hashlib.sha256(self_authority.raw).hexdigest(),
            "engine": engine_authority.row(),
            "project_authorities": login_project_rows,
            "runtime_authorities": login_runtime_rows,
            "srun_authority": srun_row,
            "compute_plan_digest": plan["plan_digest"],
            "exact_srun_argv": command,
            "exact_srun_argv_digest": engine.digest(command),
            "world4_receipt_path": str(WORLD4_RECEIPT_PATH),
            "evidence_path": str(EVIDENCE_PATH),
        }
        attempt["attempt_digest"] = object_digest(attempt)
        # This immutable claim is the first mutation and makes every refusal
        # terminal.  There is no retry branch anywhere in this controller.
        attempt_raw = engine._create_json(ATTEMPT_PATH, attempt)

        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/vast/users/guangyi.chen",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1", **CPU_THREAD_ENVIRONMENT,
            **REQUESTED_SRUN_GPU_EXPORT,
        }
        package_gate.replay()
        returncode, stdout_raw, stderr_raw = engine._run_single_srun(
            command, self_authority.raw, environment,
        )
        if returncode != 0 or stderr_raw or stdout_raw.count(b"\n") != 1:
            raise PackageWorld4Error("single srun terminal streams differ")
        compute_result = strict_json(stdout_raw, label="single srun stdout")
        receipt_raw, receipt = engine._wait_canonical_json(WORLD4_RECEIPT_PATH)
        engine._validate_compute_result(
            compute_result, plan=plan, receipt_raw=receipt_raw,
            receipt=receipt, login_project_rows=login_project_rows,
            login_runtime_rows=login_runtime_rows,
        )
        world4_authority = open_authority(
            WORLD4_RECEIPT_PATH,
            expected_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            expected_size=len(receipt_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        if world4_authority.raw != receipt_raw:
            raise PackageWorld4Error("held world4 receipt bytes differ")
        if os.path.lexists(STAGE_ROOT):
            raise PackageWorld4Error("compute stage cache remains after srun")
        package_gate.replay()

        evidence: dict[str, Any] = {
            "schema_version": SCHEMA, "status": "PASS_PACKAGE_WORLD4",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "single_srun_attempt": True, "retry_allowed": False,
            "gpu_count": GPU_COUNT, "srun_returncode": returncode,
            "attempt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_digest": attempt["attempt_digest"],
            "compute_digest": compute_result["compute_digest"],
            "world4_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "world4_receipt_digest": receipt["receipt_digest"],
            "stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
            "stderr_empty": True, "process_group_zero": True,
            "publication_empty": True, "stage_cache_absent": True,
            "scenario_count": 7,
            "per_scenario_timeout_seconds": PER_SCENARIO_TIMEOUT_SECONDS,
            "package_gate": package_gate.evidence(),
            "login_held_project_authorities": login_project_rows,
            "compute_reopened_project_authorities": compute_result[
                "compute_reopened_project_authorities"
            ],
            "project_rows_equal": True,
            "login_held_runtime_authorities": login_runtime_rows,
            "compute_reopened_runtime_authorities": compute_result[
                "runtime_authorities"
            ],
            "runtime_rows_equal": True,
            "old_cpu_admission_receipt_consumed": False,
            "launch_allowed": False, "renderer_or_vae_loaded": False,
        }
        evidence["evidence_digest"] = object_digest(evidence)
        evidence_raw = engine._create_json(EVIDENCE_PATH, evidence)
        evidence_authority = open_authority(
            EVIDENCE_PATH,
            expected_sha256=hashlib.sha256(evidence_raw).hexdigest(),
            expected_size=len(evidence_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        package_gate.replay()
        world4_authority.replay()
        evidence_authority.replay()
        return evidence
    finally:
        for descriptor in pinned:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if self_authority is not None:
            self_authority.close()
        if evidence_authority is not None:
            evidence_authority.close()
        if world4_authority is not None:
            world4_authority.close()
        if engine_authority is not None:
            engine_authority.close()
        if package_gate is not None:
            package_gate.close()


def main(argv: Sequence[str] | None = None) -> int:
    # State is the first executable gate: HOLD cannot even inspect argv or pins.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: package world4 awaits final publication/materialization/"
            "static/root-fake/root pins and state-only activation",
            file=sys.stderr,
        )
        return 88
    blocked = blocked_dynamic_pins()
    if blocked:
        print(
            "HOLD: dynamic package world4 pins are blocked: "
            + ",".join(blocked), file=sys.stderr,
        )
        return 88
    try:
        values = list(sys.argv[1:] if argv is None else argv)
        if values and values[0] == "compute":
            if len(values) != 2:
                raise PackageWorld4Error("compute argv differs")
            return compute(values[1])
        if values != ["--execute", authorization_token()]:
            raise PackageWorld4Error("controller authorization argv differs")
        controller()
        return 0
    except Exception as error:
        print(f"AUH package world4 controller refused: {error}", file=sys.stderr)
        return 96


if __name__ == "__main__":
    raise SystemExit(main())
