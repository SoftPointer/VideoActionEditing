#!/usr/bin/env python3
"""Create-only AUH CPU world4 admission controller (initially sealed HOLD).

This source is intentionally self-contained: after an explicit reviewed state
change, the login-side instance supplies these exact bytes to one ``srun`` over
an anonymous held stdin file.  The compute-side instance stages only the five
object-trajectory sources, reopens the pinned production Python/Torch runtime,
and runs the seven-scenario world4 probe.  The checked-in state cannot create a
directory, file, subprocess, Slurm step, or receipt.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-world4-cpu-auh-controller-v1"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
COMPUTE_SCHEMA = SCHEMA + "-compute"
CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_ACTIVATION"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_CPU_ADMISSION"
HOLDER_JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
CPUS_PER_TASK = 16
GPU_COUNT = 0
PER_SCENARIO_TIMEOUT_SECONDS = 30
CONTROLLER_TIMEOUT_SECONDS = 270
EXPECTED_TORCH_VERSION = "2.7.1+rocm6.3"
EXPECTED_HIP_VERSION = "6.3.42131-fa1d09cbd"
EXPECTED_SCAFFOLD_ARTIFACT_DIGEST = (
    "5e6156909d8261a23c3add3134059bec20505b682ca0eb13dc88fa8512eeace1"
)
CPU_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REQUESTED_SRUN_GPU_EXPORT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "-1",
}
EXPECTED_COMPUTE_GPU_VISIBILITY = {
    "CUDA_VISIBLE_DEVICES": None,
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": None,
}
ENVIRONMENT_SOURCE = "slurm_normalized_explicit_export_step143808_475"
SHA_RE = re.compile(r"[0-9a-f]{64}")

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
SOURCE_ROOT = (
    EXPERIMENTS
    / "bernini_case01_object_trajectory_exact5_source_staging_v1"
)
TARGET_ROOT = (
    EXPERIMENTS
    / "bernini_case01_object_trajectory_exact5_world4_cpu_admission_v1"
)
EVIDENCE_DIR = TARGET_ROOT / "evidence"
LOGS_DIR = TARGET_ROOT / "logs"
ATTEMPT_PATH = EVIDENCE_DIR / "attempt_v1.json"
WORLD4_RECEIPT_PATH = EVIDENCE_DIR / "world4_receipt_v1.json"
EVIDENCE_PATH = EVIDENCE_DIR / "controller_evidence_v1.json"
STDOUT_PATH = LOGS_DIR / "srun.stdout.log"
STDERR_PATH = LOGS_DIR / "srun.stderr.log"
STAGE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-world4-cpu-"
    "job143808-node292-v1"
)
PUBLICATION_ROOT = STAGE_ROOT / "publication"

PROJECT_AUTHORITIES = {
    "wrapper": {
        "relative": "methods/bernini_action_editing/"
        "infer_case01_object_trajectory_oracle_v1.py",
        "sha256": "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        "size": 74281,
    },
    "projection": {
        "relative": "methods/bernini_action_editing/"
        "object_trajectory_projection_v1.py",
        "sha256": "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e",
        "size": 47588,
    },
    "scaffold_module": {
        "relative": "methods/bernini_action_editing/"
        "case01_oracle_object_trajectory_v1.py",
        "sha256": "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a",
        "size": 35803,
    },
    "scaffold": {
        "relative": "artifacts/case01_oracle_object_trajectory_v1/scaffold.json",
        "sha256": "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a",
        "size": 54801,
    },
    "world4": {
        "relative": "methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_world4_probe_v1.py",
        "sha256": "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
        "size": 54489,
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
        "size": 31490256,
        "executable": True,
    },
    "torchrun_source": {
        "path": str(TORCH_ROOT / "distributed/run.py"),
        "sha256": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
        "size": 31587,
        "executable": False,
    },
    "torchrun_handler_source": {
        "path": str(
            TORCH_ROOT
            / "distributed/elastic/multiprocessing/subprocess_handler/"
            "subprocess_handler.py"
        ),
        "sha256": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
        "size": 2436,
        "executable": False,
    },
    "torch_local_agent_source": {
        "path": str(
            TORCH_ROOT / "distributed/elastic/agent/server/local_elastic_agent.py"
        ),
        "sha256": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
        "size": 16741,
        "executable": False,
    },
    "torch_dynamic_rendezvous_source": {
        "path": str(
            TORCH_ROOT / "distributed/elastic/rendezvous/dynamic_rendezvous.py"
        ),
        "sha256": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
        "size": 49422,
        "executable": False,
    },
    "torch_multiprocessing_api_source": {
        "path": str(TORCH_ROOT / "distributed/elastic/multiprocessing/api.py"),
        "sha256": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
        "size": 33740,
        "executable": False,
    },
}
SRUN_AUTHORITY = {
    "path": "/usr/bin/srun",
    "sha256": "2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e",
    "size": 164720,
    "executable": True,
}
TORCH_ROLES = tuple(role for role in RUNTIME_AUTHORITIES if role != "python")
IDENTITY_ROW_KEYS = {
    "path", "sha256", "size", "device", "inode", "uid", "gid", "mode",
    "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
}


class CpuAdmissionError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def _identity_row(path: Path, info: os.stat_result, sha256: str) -> dict[str, Any]:
    return {
        "path": str(path), "sha256": sha256, "size": info.st_size,
        "device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink, "rdev": info.st_rdev,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def _open_pinned(
    path: Path, sha256: str, size: int, *, executable: bool = False,
) -> tuple[int, bytes, dict[str, Any]]:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or SHA_RE.fullmatch(sha256) is None or type(size) is not int or size <= 0
    ):
        raise CpuAdmissionError(f"noncanonical authority: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise CpuAdmissionError(f"missing authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or named.st_size != size or path.resolve(strict=True) != path
        or (executable and not named.st_mode & 0o111)
    ):
        raise CpuAdmissionError(f"named authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != size or _identity(before) != _identity(named)
            or (executable and not before.st_mode & 0o111)
        ):
            raise CpuAdmissionError(f"opened authority differs: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < size:
            block = os.pread(descriptor, min(1_048_576, size - offset), offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        raw = b"".join(chunks)
        replay = b"".join(
            os.pread(descriptor, min(1_048_576, size - at), at)
            for at in range(0, size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
    except BaseException:
        os.close(descriptor)
        raise
    if (
        len(raw) != size or raw != replay or eof != b""
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or hashlib.sha256(raw).hexdigest() != sha256
    ):
        os.close(descriptor)
        raise CpuAdmissionError(f"authority replay differs: {path}")
    return descriptor, raw, _identity_row(path, before, sha256)


def _open_observed(
    path: Path, *, maximum_size: int, expected_mode: int | None = None,
) -> tuple[int, bytes, dict[str, Any]]:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or type(maximum_size) is not int or maximum_size <= 0
    ):
        raise CpuAdmissionError(f"noncanonical observed authority: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or not (0 < named.st_size <= maximum_size)
        or path.resolve(strict=True) != path
        or (expected_mode is not None
            and stat.S_IMODE(named.st_mode) != expected_mode)
    ):
        raise CpuAdmissionError(f"named observed authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != named.st_size
            or _identity(before) != _identity(named)
            or (expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode)
        ):
            raise CpuAdmissionError(f"opened observed authority differs: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1_048_576, before.st_size - offset), offset,
            )
            if not block:
                break
            chunks.append(block); offset += len(block)
        raw = b"".join(chunks)
        replay = b"".join(
            os.pread(descriptor, min(1_048_576, before.st_size - at), at)
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor); named_after = os.lstat(path)
    except BaseException:
        os.close(descriptor); raise
    if (
        len(raw) != before.st_size or raw != replay or eof != b""
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        os.close(descriptor)
        raise CpuAdmissionError(f"observed authority replay differs: {path}")
    sha256 = hashlib.sha256(raw).hexdigest()
    return descriptor, raw, _identity_row(path, before, sha256)


def _wait_canonical_json(path: Path, timeout_seconds: float = 30.0) -> tuple[bytes, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() <= deadline:
        descriptor: int | None = None
        try:
            descriptor, raw, _row = _open_observed(
                path, maximum_size=4_194_304, expected_mode=0o400,
            )
            value = json.loads(raw)
            if type(value) is not dict or raw != canonical(value) + b"\n":
                raise CpuAdmissionError("canonical receipt bytes differ")
            return raw, value
        except (OSError, ValueError, CpuAdmissionError) as error:
            last_error = error
            time.sleep(0.05)
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise CpuAdmissionError(f"canonical receipt visibility timed out: {path}") from last_error


def _fresh(path: Path) -> None:
    if os.path.lexists(path):
        raise CpuAdmissionError(f"create-only target is not fresh: {path}")


def _process_group_state(process_group: int) -> str:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "inaccessible"
    return "present"


def _signal_live_process_group(
    process: subprocess.Popen[Any], process_group: int, signal_number: int,
) -> str:
    if process.poll() is not None:
        return "leader_reaped"
    try:
        observed = os.getpgid(process.pid)
    except ProcessLookupError:
        return "leader_missing"
    except PermissionError:
        return "leader_inaccessible"
    if observed != process_group or process_group != process.pid:
        return "identity_changed"
    if _process_group_state(process_group) != "present":
        return _process_group_state(process_group)
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "inaccessible"
    return "signaled"


def _process_group_absent(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if _process_group_state(process_group) == "absent":
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _terminate_and_reap_process_group(
    process: subprocess.Popen[Any], process_group: int,
) -> tuple[bool, bool]:
    _signal_live_process_group(process, process_group, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _signal_live_process_group(process, process_group, signal.SIGKILL)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    direct_child_reaped = process.poll() is not None
    # Once the leader is reaped, never signal its naked numeric PGID again.
    # Only ESRCH from bounded polling proves absence; EPERM remains failure.
    group_absent = _process_group_absent(process_group, 2)
    return direct_child_reaped, group_absent


def _create_json(path: Path, value: Mapping[str, Any]) -> bytes:
    raw = canonical(value) + b"\n"
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise CpuAdmissionError("create-only JSON write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        named = os.lstat(path)
        replay = os.pread(descriptor, len(raw), 0)
    finally:
        os.close(descriptor)
    if (
        replay != raw or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o400
        or _identity(before) != _identity(named)
    ):
        raise CpuAdmissionError(f"create-only JSON replay differs: {path}")
    return raw


def _project_rows() -> dict[str, dict[str, Any]]:
    return {
        role: {
            "path": str(SOURCE_ROOT / row["relative"]),
            "sha256": row["sha256"], "size": row["size"],
        }
        for role, row in PROJECT_AUTHORITIES.items()
    }


def build_srun_argv(plan_b64: str) -> list[str]:
    if not plan_b64 or len(plan_b64) > 131072:
        raise CpuAdmissionError("compute plan transport differs")
    exported = {
        **CPU_THREAD_ENVIRONMENT, **REQUESTED_SRUN_GPU_EXPORT,
    }
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
    if len(" ".join(argv).encode("ascii")) >= 8192:
        raise CpuAdmissionError("exact srun argv exceeds admitted width")
    return argv


def build_compute_plan() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA + "-plan",
        "holder_job_id": HOLDER_JOB_ID, "node": NODE,
        "cpus_per_task": CPUS_PER_TASK, "gpu_count": GPU_COUNT,
        "single_srun_attempt": True, "retry_allowed": False,
        "per_scenario_timeout_seconds": PER_SCENARIO_TIMEOUT_SECONDS,
        "controller_timeout_seconds": CONTROLLER_TIMEOUT_SECONDS,
        "expected_torch_version": EXPECTED_TORCH_VERSION,
        "expected_hip_version": EXPECTED_HIP_VERSION,
        "cpu_thread_environment": dict(CPU_THREAD_ENVIRONMENT),
        "requested_srun_gpu_export": dict(REQUESTED_SRUN_GPU_EXPORT),
        "expected_compute_gpu_visibility": dict(
            EXPECTED_COMPUTE_GPU_VISIBILITY
        ),
        "environment_source": ENVIRONMENT_SOURCE,
        "project_authorities": _project_rows(),
        "runtime_authorities": {
            role: {key: row[key] for key in ("path", "sha256", "size")}
            for role, row in RUNTIME_AUTHORITIES.items()
        },
        "stage_root": str(STAGE_ROOT),
        "publication_root": str(PUBLICATION_ROOT),
        "world4_receipt_path": str(WORLD4_RECEIPT_PATH),
        "world4_scenarios": [
            "happy", "hostile_rank0_tensor", "hostile_rank2_tensor",
            "hostile_rank0_aux", "hostile_rank2_abi",
            "hostile_rank1_row_build", "hostile_rank3_final_scheduler",
        ],
        "publication_allowed": False,
    }
    value["plan_digest"] = digest(value)
    return value


def validate_compute_plan(value: Mapping[str, Any]) -> None:
    expected = build_compute_plan()
    if type(value) is not dict or value != expected:
        raise CpuAdmissionError("compute plan closure differs")


def _stage_project(
    plan: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int, int]:
    # Reopen and retain every shared-source authority before the first stage
    # mutation.  This makes a missing, swapped, linked, or special source fail
    # without leaving a partial /tmp tree.
    held: dict[str, tuple[int, bytes, dict[str, Any]]] = {}
    try:
        for role, expected in plan["project_authorities"].items():
            held[role] = _open_pinned(
                Path(expected["path"]), expected["sha256"], expected["size"],
            )
    except BaseException:
        for descriptor, _raw, _row in held.values():
            os.close(descriptor)
        raise
    rows: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    parent_descriptor: int | None = None
    stage_descriptor: int | None = None
    created_stage_identity: tuple[int, ...] | None = None
    try:
        if STAGE_ROOT.name in {"", ".", ".."}:
            raise CpuAdmissionError("stage root basename differs")
        parent = STAGE_ROOT.parent
        parent_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        held_parent = os.fstat(parent_descriptor)
        named_parent = os.lstat(parent)
        if (
            not stat.S_ISDIR(held_parent.st_mode)
            or _directory_identity(held_parent)
            != _directory_identity(named_parent)
            or parent.resolve(strict=True) != parent
        ):
            raise CpuAdmissionError("stage parent identity differs")
        try:
            os.stat(
                STAGE_ROOT.name, dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise CpuAdmissionError(
                f"create-only target is not fresh: {STAGE_ROOT}"
            )
        os.mkdir(STAGE_ROOT.name, 0o700, dir_fd=parent_descriptor)
        created_stage = os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        created_stage_identity = _directory_identity(created_stage)
        stage_descriptor = os.open(
            STAGE_ROOT.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        held_stage = os.fstat(stage_descriptor)
        named_stage = os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(held_stage.st_mode)
            or _directory_identity(held_stage) != created_stage_identity
            or _directory_identity(held_stage) != _directory_identity(named_stage)
            or STAGE_ROOT.resolve(strict=True) != STAGE_ROOT
        ):
            raise CpuAdmissionError("new stage root identity differs")
        os.mkdir("publication", 0o700, dir_fd=stage_descriptor)
        for role, expected in plan["project_authorities"].items():
            descriptor, raw, source_row = held[role]
            source_rows[role] = source_row
            suffix = ".json" if role == "scaffold" else ".py"
            target_name = role + suffix
            target = STAGE_ROOT / target_name
            output = os.open(
                target_name, os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0), 0,
                dir_fd=stage_descriptor,
            )
            try:
                offset = 0
                while offset < len(raw):
                    count = os.write(output, raw[offset:])
                    if count <= 0:
                        raise CpuAdmissionError("stage write made no progress")
                    offset += count
                os.fsync(output); os.fchmod(output, 0o400); os.fsync(output)
                info = os.fstat(output); replay = os.pread(output, len(raw), 0)
            finally:
                os.close(output)
            if (
                replay != raw or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o400
            ):
                raise CpuAdmissionError(f"staged authority differs: {role}")
            rows[role] = _identity_row(target, info, expected["sha256"])
        if set(os.listdir(stage_descriptor)) != {
            "publication", *(
                role + (".json" if role == "scaffold" else ".py")
                for role in plan["project_authorities"]
            ),
        }:
            raise CpuAdmissionError("staged exact tree differs")
        return rows, source_rows, parent_descriptor, stage_descriptor
    except BaseException as error:
        try:
            if stage_descriptor is not None and parent_descriptor is not None:
                _remove_owned_stage_tree(parent_descriptor, stage_descriptor)
            elif (
                created_stage_identity is not None
                and parent_descriptor is not None
            ):
                _remove_created_empty_stage_root(
                    parent_descriptor, created_stage_identity,
                )
        except BaseException as cleanup_error:
            raise CpuAdmissionError(
                f"partial stage cleanup differs: {cleanup_error}"
            ) from error
        finally:
            try:
                if stage_descriptor is not None:
                    os.close(stage_descriptor)
            finally:
                if parent_descriptor is not None:
                    os.close(parent_descriptor)
        raise
    finally:
        for descriptor, _raw, _row in held.values():
            os.close(descriptor)


def _simple_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("path", "sha256", "size")}


def _validate_identity_row(
    row: Mapping[str, Any], *, path: str, sha256: str, size: int,
    mode: int | None = None,
) -> None:
    if (
        type(row) is not dict or set(row) != IDENTITY_ROW_KEYS
        or row.get("path") != path or row.get("sha256") != sha256
        or row.get("size") != size or row.get("nlink") != 1
        or any(type(row.get(key)) is not int for key in IDENTITY_ROW_KEYS - {
            "path", "sha256",
        })
        or (mode is not None and row.get("mode") != mode)
    ):
        raise CpuAdmissionError(f"identity row closure differs: {path}")


def _expected_world4_runtime_identities(
    staged_rows: Mapping[str, Mapping[str, Any]],
    runtime_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = {
        role: _simple_identity(runtime_rows[role])
        for role in ("python", *TORCH_ROLES)
    }
    expected.update({
        role: _simple_identity(staged_rows[role])
        for role in ("wrapper", "projection", "scaffold_module", "scaffold")
    })
    return expected


def _validate_world4(
    result: Mapping[str, Any], plan: Mapping[str, Any], *,
    staged_rows: Mapping[str, Mapping[str, Any]],
    runtime_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    result_keys = {
        "schema_version", "status", "launch_allowed", "scenario_order",
        "scenarios", "runtime_identities", "runtime_identity_digest",
        "expected_runtime_versions", "expected_gpu_contract",
        "cpu_thread_contract", "active_row_counts_admitted",
        "happy_scheduler_steps", "real_torchrun_process_count_per_scenario",
        "controller_python_optimize_level", "timeout_seconds_per_scenario",
        "timeout_cleanup_policy", "publication_performed",
        "renderer_or_vae_loaded", "scope", "receipt_digest",
    }
    expected_runtime = _expected_world4_runtime_identities(
        staged_rows, runtime_rows,
    )
    if (
        type(result) is not dict or set(result) != result_keys
        or result.get("schema_version")
        != "case01-object-trajectory-exact5-world4-admission-v5"
        or result.get("status") != "ADMITTED_WORLD4_TENSOR_ABI_HOLD_ONLY"
        or result.get("launch_allowed") is not False
        or result.get("publication_performed") is not False
        or result.get("renderer_or_vae_loaded") is not False
        or result.get("scope")
        != "distributed_tensor_projection_abi_not_renderer_integration"
        or result.get("timeout_seconds_per_scenario") != 30
        or result.get("timeout_cleanup_policy")
        != "new_session_sigterm_then_sigkill_bounded_reap"
        or result.get("scenario_order") != plan["world4_scenarios"]
        or result.get("runtime_identities") != expected_runtime
        or result.get("runtime_identity_digest") != digest(expected_runtime)
        or result.get("expected_runtime_versions") != {
            "torch": EXPECTED_TORCH_VERSION, "hip": EXPECTED_HIP_VERSION,
        }
        or result.get("cpu_thread_contract") != {
            "environment": CPU_THREAD_ENVIRONMENT,
            "torch_num_threads": 1, "torch_num_interop_threads": 1,
        }
        or result.get("expected_gpu_contract") != {
            "device_count": 0,
            "visibility_environment": EXPECTED_COMPUTE_GPU_VISIBILITY,
        }
        or result.get("active_row_counts_admitted") != [2, 3]
        or result.get("happy_scheduler_steps") != 40
        or result.get("real_torchrun_process_count_per_scenario") != 4
        or result.get("controller_python_optimize_level") != 0
        or type(result.get("scenarios")) is not list
        or len(result["scenarios"]) != 7
    ):
        raise CpuAdmissionError("world4 target receipt closure differs")
    unsigned = dict(result); claimed = unsigned.pop("receipt_digest", None)
    if claimed != digest(unsigned):
        raise CpuAdmissionError("world4 target receipt digest differs")

    scenario_keys = {
        "scenario", "worker_result", "elapsed_milliseconds",
        "timeout_seconds", "process_group_id", "process_group_reaped",
        "publication_empty_after_scenario", "worker_optimize_level",
        "stdout_sha256", "stderr_sha256",
    }
    worker_keys = {
        "schema_version", "scenario", "status", "world_size",
        "cpu_thread_contract", "expected_runtime_versions", "rank_rows",
        "expected_gpu_contract", "publication_performed", "result_digest",
    }
    rank_keys = {
        "rank", "local_rank", "scenario", "world_size",
        "python_optimize_level", "torch_version", "distributed_backend",
        "torch_hip_version", "expected_torch_version",
        "expected_hip_version", "gpu_visibility_environment",
        "expected_gpu_count", "torch_visible_gpu_count",
        "cpu_thread_environment", "torch_num_threads",
        "torch_num_interop_threads", "source_broadcast_calls",
        "aux_broadcast_calls", "active_arm", "row_count",
        "consensus_failed", "stage_gate_failed", "failure_stage",
        "trace_steps", "scheduler_calls", "scheduler_token_count",
        "operational_path", "operational_aux_gate_count",
        "operational_projection_gate_count",
        "operational_wrapper_trace_steps", "publication_empty",
        "scaffold_digest", "runtime_identity_digest", "row_digest",
    }
    failure_stages = {
        "happy": None,
        "hostile_rank0_tensor": "projection_contract_consensus",
        "hostile_rank2_tensor": "projection_contract_consensus",
        "hostile_rank0_aux": "aux_readiness",
        "hostile_rank2_abi": "aux_readiness",
        "hostile_rank1_row_build": "projection_row_build",
        "hostile_rank3_final_scheduler": "projection_final_validation",
    }
    runtime_rank_digest = digest({
        role: expected_runtime[role] for role in ("python", *TORCH_ROLES)
    })
    observed_process_groups: set[int] = set()
    for scenario_index, scenario in enumerate(result["scenarios"]):
        name = plan["world4_scenarios"][scenario_index]
        if (
            type(scenario) is not dict or set(scenario) != scenario_keys
            or scenario.get("scenario") != name
            or scenario.get("process_group_reaped") is not True
            or scenario.get("publication_empty_after_scenario") is not True
            or scenario.get("timeout_seconds") != 30
            or scenario.get("worker_optimize_level") != 0
            or type(scenario.get("process_group_id")) is not int
            or scenario["process_group_id"] <= 1
            or type(scenario.get("elapsed_milliseconds")) is not int
            or not (0 <= scenario["elapsed_milliseconds"] < 30000)
            or SHA_RE.fullmatch(str(scenario.get("stdout_sha256"))) is None
            or SHA_RE.fullmatch(str(scenario.get("stderr_sha256"))) is None
        ):
            raise CpuAdmissionError("world4 process/publication evidence differs")
        observed_process_groups.add(scenario["process_group_id"])
        worker = scenario.get("worker_result")
        if type(worker) is not dict:
            raise CpuAdmissionError("world4 worker result is not a mapping")
        unsigned_worker = dict(worker)
        claimed_worker = unsigned_worker.pop("result_digest", None)
        if (
            set(worker) != worker_keys
            or worker.get("schema_version")
            != "case01-object-trajectory-exact5-world4-worker-v5"
            or worker.get("scenario") != name
            or worker.get("status")
            != ("PASS_HAPPY" if name == "happy" else "PASS_EXPECTED_HOSTILE")
            or worker.get("world_size") != 4
            or worker.get("publication_performed") is not False
            or worker.get("cpu_thread_contract")
            != result["cpu_thread_contract"]
            or worker.get("expected_runtime_versions")
            != result["expected_runtime_versions"]
            or worker.get("expected_gpu_contract")
            != result["expected_gpu_contract"]
            or claimed_worker != digest(unsigned_worker)
            or type(worker.get("rank_rows")) is not list
            or len(worker["rank_rows"]) != 4
        ):
            raise CpuAdmissionError("world4 worker result closure differs")
        expected_arm = (
            "trajectory_bone_only"
            if name in {"hostile_rank0_tensor", "hostile_rank0_aux"}
            else "trajectory_dog_bone"
        )
        for rank_index, row in enumerate(worker["rank_rows"]):
            if type(row) is not dict:
                raise CpuAdmissionError("world4 rank row is not a mapping")
            unsigned_row = dict(row); claimed_row = unsigned_row.pop("row_digest", None)
            expected_steps = (
                39 if name == "hostile_rank3_final_scheduler" and rank_index == 3
                else 40 if name in {"happy", "hostile_rank3_final_scheduler"}
                else 0
            )
            expected_operational = {
                "happy": (
                    "oracle_execution_state.clamp_full_path", 2, 7, 1,
                    expected_steps, 19530,
                ),
                "hostile_rank0_tensor": (
                    "projection_contract_consensus", 0, 0, 0, 0, 0,
                ),
                "hostile_rank2_tensor": (
                    "projection_contract_consensus", 0, 0, 0, 0, 0,
                ),
                "hostile_rank0_aux": (
                    "oracle_execution_state.distributed_aux", 0, 0, 0, 0, 0,
                ),
                "hostile_rank2_abi": (
                    "oracle_execution_state.distributed_aux", 0, 0, 0, 0, 0,
                ),
                "hostile_rank1_row_build": (
                    "oracle_execution_state.clamp_row_build", 2, 1, 1, 0, 0,
                ),
                "hostile_rank3_final_scheduler": (
                    "oracle_execution_state.clamp_full_path", 2, 6, 1,
                    expected_steps, 19530,
                ),
            }[name]
            expected_consensus_failed = name in {
                "hostile_rank0_tensor", "hostile_rank2_tensor",
            }
            expected_stage_gate_failed = name in {
                "hostile_rank0_aux", "hostile_rank2_abi",
                "hostile_rank1_row_build", "hostile_rank3_final_scheduler",
            }
            if (
                set(row) != rank_keys
                or row.get("rank") != rank_index
                or row.get("local_rank") != rank_index
                or row.get("scenario") != name or row.get("world_size") != 4
                or row.get("python_optimize_level") != 0
                or row.get("distributed_backend") != "gloo"
                or row.get("torch_version") != EXPECTED_TORCH_VERSION
                or row.get("torch_hip_version") != EXPECTED_HIP_VERSION
                or row.get("expected_torch_version") != EXPECTED_TORCH_VERSION
                or row.get("expected_hip_version") != EXPECTED_HIP_VERSION
                or row.get("torch_num_threads") != 1
                or row.get("torch_num_interop_threads") != 1
                or row.get("cpu_thread_environment") != CPU_THREAD_ENVIRONMENT
                or row.get("gpu_visibility_environment")
                != EXPECTED_COMPUTE_GPU_VISIBILITY
                or row.get("expected_gpu_count") != 0
                or row.get("torch_visible_gpu_count") != 0
                or row.get("source_broadcast_calls") != 1
                or row.get("active_arm") != expected_arm
                or row.get("row_count") != (2 if expected_arm.endswith("bone_only") else 3)
                or row.get("failure_stage") != failure_stages[name]
                or row.get("consensus_failed") is not expected_consensus_failed
                or row.get("stage_gate_failed") is not expected_stage_gate_failed
                or row.get("trace_steps") != expected_steps
                or row.get("scheduler_calls") != expected_steps
                or row.get("scheduler_token_count") != expected_operational[5]
                or row.get("operational_path") != expected_operational[0]
                or row.get("operational_aux_gate_count")
                != expected_operational[1]
                or row.get("operational_projection_gate_count")
                != expected_operational[2]
                or row.get("aux_broadcast_calls") != expected_operational[3]
                or row.get("operational_wrapper_trace_steps")
                != expected_operational[4]
                or row.get("publication_empty") is not True
                or row.get("scaffold_digest")
                != EXPECTED_SCAFFOLD_ARTIFACT_DIGEST
                or row.get("runtime_identity_digest") != runtime_rank_digest
                or claimed_row != digest(unsigned_row)
            ):
                raise CpuAdmissionError(
                    f"world4 rank runtime closure differs: {name}/r{rank_index}"
                )
    if len(observed_process_groups) != len(plan["world4_scenarios"]):
        raise CpuAdmissionError("world4 scenario process groups are not distinct")


def _residual_stage_processes(token: str) -> list[int]:
    residual: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise CpuAdmissionError("AUH /proc is unavailable")
    for child in proc.iterdir():
        if not child.name.isdecimal() or int(child.name) == os.getpid():
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except OSError:
            continue
        if token.encode() in raw:
            residual.append(int(child.name))
    return sorted(residual)


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid,
        stat.S_IMODE(info.st_mode), stat.S_IFMT(info.st_mode), info.st_rdev,
    )


def _remove_created_empty_stage_root(
    parent_descriptor: int, created_identity: tuple[int, ...],
) -> None:
    try:
        named = os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise CpuAdmissionError(
            "new stage root disappeared before open-failure cleanup"
        ) from error
    if (
        not stat.S_ISDIR(named.st_mode)
        or _directory_identity(named) != created_identity
    ):
        raise CpuAdmissionError(
            "new stage root was replaced before open-failure cleanup"
        )
    os.rmdir(STAGE_ROOT.name, dir_fd=parent_descriptor)
    try:
        os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise CpuAdmissionError("new empty stage root remains after cleanup")


def _clear_owned_directory(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        if name in {"", ".", ".."} or "/" in name:
            raise CpuAdmissionError("owned stage child name differs")
        try:
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as error:
            raise CpuAdmissionError(
                "owned stage child disappeared during cleanup"
            ) from error
        if stat.S_ISDIR(named.st_mode):
            child = os.open(
                name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                held = os.fstat(child)
                if _directory_identity(held) != _directory_identity(named):
                    raise CpuAdmissionError(
                        "owned stage child identity changed before cleanup"
                    )
                _clear_owned_directory(child)
                named_after = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False,
                )
                if (
                    _directory_identity(os.fstat(child))
                    != _directory_identity(named_after)
                ):
                    raise CpuAdmissionError(
                        "owned stage child identity changed after cleanup"
                    )
                os.rmdir(name, dir_fd=descriptor)
            finally:
                os.close(child)
        else:
            os.unlink(name, dir_fd=descriptor)


def _remove_owned_stage_tree(
    parent_descriptor: int, stage_descriptor: int,
) -> None:
    held = os.fstat(stage_descriptor)
    try:
        named = os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise CpuAdmissionError(
            "owned stage root disappeared before cleanup"
        ) from error
    if (
        not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _directory_identity(held) != _directory_identity(named)
        or STAGE_ROOT.resolve(strict=True) != STAGE_ROOT
    ):
        raise CpuAdmissionError("world4 owned stage cleanup target differs")
    _clear_owned_directory(stage_descriptor)
    named_after_clear = os.stat(
        STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
    )
    if _directory_identity(held) != _directory_identity(named_after_clear):
        raise CpuAdmissionError("owned stage root changed during cleanup")
    os.rmdir(STAGE_ROOT.name, dir_fd=parent_descriptor)
    try:
        os.stat(
            STAGE_ROOT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise CpuAdmissionError("world4 local stage cleanup differs")


def _execute_world4(
    plan: Mapping[str, Any], runtime_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bytes, dict[str, Any]]:
    staged, compute_source_rows, parent_descriptor, stage_descriptor = (
        _stage_project(plan)
    )
    try:
        world4_path = STAGE_ROOT / "world4.py"
        world4_pin = PROJECT_AUTHORITIES["world4"]
        descriptor, source, _row = _open_pinned(
            world4_path, world4_pin["sha256"], world4_pin["size"],
        )
        os.close(descriptor)
        spec = importlib.util.spec_from_loader(
            "_case01_target_world4", loader=None, origin=str(world4_path),
        )
        if spec is None:
            raise CpuAdmissionError("cannot create staged world4 module")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(world4_path)
        exec(
            compile(source.decode("utf-8", "strict"), str(world4_path), "exec"),
            module.__dict__,
        )
        argv = [
            "run", "--python", str(VACE_PYTHON),
            "--python-sha256", RUNTIME_AUTHORITIES["python"]["sha256"],
            "--expected-torch-version", EXPECTED_TORCH_VERSION,
            "--expected-hip-version", EXPECTED_HIP_VERSION,
            "--expected-gpu-count", "0",
            "--expected-cuda-visible-devices", module.UNSET_ENV_SENTINEL,
            "--expected-hip-visible-devices", EXPECTED_COMPUTE_GPU_VISIBILITY[
                "HIP_VISIBLE_DEVICES"
            ],
            "--expected-rocr-visible-devices", module.UNSET_ENV_SENTINEL,
            "--wrapper", str(STAGE_ROOT / "wrapper.py"),
            "--projection", str(STAGE_ROOT / "projection.py"),
            "--scaffold-module", str(STAGE_ROOT / "scaffold_module.py"),
            "--scaffold", str(STAGE_ROOT / "scaffold.json"),
            "--publication-root", str(PUBLICATION_ROOT),
            "--output", str(WORLD4_RECEIPT_PATH),
        ]
        for role in TORCH_ROLES:
            row = RUNTIME_AUTHORITIES[role]; option = role.replace("_", "-")
            argv.extend([
                "--" + option, row["path"],
                "--" + option + "-sha256", row["sha256"],
            ])
        result = module.controller(module.build_parser().parse_args(argv))
        _validate_world4(
            result, plan, staged_rows=staged, runtime_rows=runtime_rows,
        )
        receipt_raw, receipt_value = _wait_canonical_json(WORLD4_RECEIPT_PATH)
        if receipt_value != result or any(PUBLICATION_ROOT.iterdir()):
            raise CpuAdmissionError(
                "world4 target receipt/publication replay differs"
            )
        residual = _residual_stage_processes(str(STAGE_ROOT))
        if residual:
            raise CpuAdmissionError(
                f"world4 residual process ids differ: {residual}"
            )
        return staged, compute_source_rows, receipt_raw, result
    finally:
        try:
            _remove_owned_stage_tree(parent_descriptor, stage_descriptor)
        finally:
            try:
                os.close(stage_descriptor)
            finally:
                os.close(parent_descriptor)


def compute(plan_b64: str) -> int:
    try:
        raw = base64.b64decode(plan_b64, validate=True)
        plan = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as error:
        raise CpuAdmissionError("compute plan transport is invalid") from error
    if raw != canonical(plan):
        raise CpuAdmissionError("compute plan transport is not canonical")
    validate_compute_plan(plan)
    if (
        os.environ.get("SLURM_JOB_ID") != HOLDER_JOB_ID
        or os.environ.get("SLURMD_NODENAME") != NODE
        or os.environ.get("SLURM_CPUS_PER_TASK") != str(CPUS_PER_TASK)
        or socket.gethostname().split(".", 1)[0] != NODE
        or {key: os.environ.get(key) for key in CPU_THREAD_ENVIRONMENT}
        != CPU_THREAD_ENVIRONMENT
        or {key: os.environ.get(key) for key in EXPECTED_COMPUTE_GPU_VISIBILITY}
        != EXPECTED_COMPUTE_GPU_VISIBILITY
    ):
        raise CpuAdmissionError("compute Slurm/CPU environment differs")
    runtime_rows: dict[str, dict[str, Any]] = {}
    for role, expected in plan["runtime_authorities"].items():
        descriptor, _raw, row = _open_pinned(
            Path(expected["path"]), expected["sha256"], expected["size"],
            executable=role == "python",
        )
        os.close(descriptor); runtime_rows[role] = row
    proc_exe = Path("/proc/self/exe").resolve(strict=True)
    if proc_exe != VACE_PYTHON:
        raise CpuAdmissionError("compute held Python entry differs")
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        compute_num_threads = int(torch.get_num_threads())
        compute_num_interop_threads = int(torch.get_num_interop_threads())
    except BaseException as error:
        raise CpuAdmissionError(
            "compute Torch CPU thread configuration failed"
        ) from error
    if (
        str(torch.__version__) != EXPECTED_TORCH_VERSION
        or getattr(torch.version, "hip", None) != EXPECTED_HIP_VERSION
        or int(torch.cuda.device_count()) != 0
        or compute_num_threads != 1 or compute_num_interop_threads != 1
    ):
        raise CpuAdmissionError("compute Torch/HIP/GPU-zero contract differs")
    staged, compute_source_rows, receipt_raw, result = _execute_world4(
        plan, runtime_rows,
    )
    value: dict[str, Any] = {
        "schema_version": COMPUTE_SCHEMA, "status": "PASS",
        "holder_job_id": HOLDER_JOB_ID, "node": NODE,
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cpus_per_task": CPUS_PER_TASK, "gpu_count": GPU_COUNT,
        "single_srun_attempt": True, "retry_allowed": False,
        "expected_torch_version": EXPECTED_TORCH_VERSION,
        "expected_hip_version": EXPECTED_HIP_VERSION,
        "torch_visible_gpu_count": 0,
        "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "cpu_thread_environment": dict(CPU_THREAD_ENVIRONMENT),
        "requested_srun_gpu_export": dict(REQUESTED_SRUN_GPU_EXPORT),
        "gpu_visibility_environment": dict(EXPECTED_COMPUTE_GPU_VISIBILITY),
        "environment_source": ENVIRONMENT_SOURCE,
        "project_authorities": staged, "runtime_authorities": runtime_rows,
        "compute_reopened_project_authorities": compute_source_rows,
        "world4_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "world4_receipt_digest": result["receipt_digest"],
        "scenario_count": 7, "process_group_zero": True,
        "publication_empty": True, "stage_cache_absent": True,
        "renderer_or_vae_loaded": False, "launch_allowed": False,
    }
    value["compute_digest"] = digest(value)
    print(canonical(value).decode("utf-8"), flush=True)
    return 0


def _seal_log(descriptor: int, path: Path) -> bytes:
    os.fsync(descriptor); os.fchmod(descriptor, 0o400); os.fsync(descriptor)
    info = os.fstat(descriptor); named = os.lstat(path)
    raw = b""; offset = 0
    while offset < info.st_size:
        block = os.pread(descriptor, min(1_048_576, info.st_size - offset), offset)
        if not block: break
        raw += block; offset += len(block)
    if (
        len(raw) != info.st_size or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or _identity(info) != _identity(named)
    ):
        raise CpuAdmissionError(f"held log replay differs: {path}")
    return raw


def _run_single_srun(
    command: Sequence[str], self_raw: bytes, environment: Mapping[str, str],
) -> tuple[int, bytes, bytes]:
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    payload: Any | None = None
    process: subprocess.Popen[Any] | None = None
    process_group: int | None = None
    candidate_group: int | None = None
    returncode: int | None = None
    stdout_raw = b""
    stderr_raw = b""
    body_error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        stdout_fd = os.open(
            STDOUT_PATH, os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        stderr_fd = os.open(
            STDERR_PATH, os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        payload = tempfile.TemporaryFile()
        payload.write(self_raw); payload.flush(); payload.seek(0)
        process = subprocess.Popen(
            list(command), stdin=payload, stdout=stdout_fd, stderr=stderr_fd,
            env=dict(environment), start_new_session=True, close_fds=True,
        )
        candidate_group = process.pid
        try:
            process_group = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError) as error:
            raise CpuAdmissionError(
                "cannot establish held srun process-group identity"
            ) from error
        if process_group != process.pid:
            raise CpuAdmissionError("srun process-group identity differs")
        try:
            returncode = process.wait(timeout=CONTROLLER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            direct_reaped, group_absent = _terminate_and_reap_process_group(
                process, process_group,
            )
            if not direct_reaped or not group_absent:
                raise CpuAdmissionError(
                    "single srun timeout cleanup lacks reaped/ESRCH proof"
                ) from error
            raise CpuAdmissionError("single srun controller timeout") from error
        if process.poll() is None:
            raise CpuAdmissionError("single srun direct child was not reaped")
        if not _process_group_absent(process_group, 2):
            raise CpuAdmissionError(
                "single srun terminal process group lacks ESRCH proof"
            )
    except BaseException as error:
        body_error = error
    finally:
        if process is not None and process.poll() is None:
            if process_group is None:
                # No group identity was proven, so only the still-held direct
                # child may be signalled.  This path is always a refusal.
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                        process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            else:
                direct_reaped, group_absent = _terminate_and_reap_process_group(
                    process, process_group,
                )
                if not direct_reaped or not group_absent:
                    cleanup_errors.append(
                        "direct child or process group remains unproven"
                    )
        if (
            process is not None and process_group is None
            and candidate_group is not None
            and not _process_group_absent(candidate_group, 2)
        ):
            cleanup_errors.append("unproven process-group identity")
        if payload is not None:
            try:
                payload.close()
            except OSError:
                cleanup_errors.append("held stdin close failed")
        for label, descriptor, path in (
            ("stdout", stdout_fd, STDOUT_PATH),
            ("stderr", stderr_fd, STDERR_PATH),
        ):
            if descriptor is None:
                continue
            try:
                raw = _seal_log(descriptor, path)
                if label == "stdout":
                    stdout_raw = raw
                else:
                    stderr_raw = raw
            except BaseException as error:
                cleanup_errors.append(f"{label} seal failed: {error}")
            finally:
                try:
                    os.close(descriptor)
                except OSError:
                    cleanup_errors.append(f"{label} close failed")
    if cleanup_errors:
        raise CpuAdmissionError(
            "single srun cleanup differs: " + "; ".join(cleanup_errors)
        ) from body_error
    if body_error is not None:
        raise body_error
    if returncode is None:
        raise CpuAdmissionError("single srun lacks a terminal return code")
    return returncode, stdout_raw, stderr_raw


def _validate_compute_result(
    value: Mapping[str, Any], *, plan: Mapping[str, Any],
    receipt_raw: bytes, receipt: Mapping[str, Any],
    login_project_rows: Mapping[str, Mapping[str, Any]],
    login_runtime_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    keys = {
        "schema_version", "status", "holder_job_id", "node",
        "slurm_step_id", "cpus_per_task", "gpu_count",
        "single_srun_attempt", "retry_allowed", "expected_torch_version",
        "expected_hip_version", "torch_visible_gpu_count",
        "torch_num_threads", "torch_num_interop_threads",
        "cpu_thread_environment", "requested_srun_gpu_export",
        "gpu_visibility_environment", "environment_source",
        "project_authorities", "runtime_authorities",
        "compute_reopened_project_authorities", "world4_receipt_sha256",
        "world4_receipt_digest", "scenario_count", "process_group_zero",
        "publication_empty", "stage_cache_absent", "renderer_or_vae_loaded",
        "launch_allowed", "compute_digest",
    }
    if type(value) is not dict:
        raise CpuAdmissionError("compute result is not a mapping")
    unsigned = dict(value); claimed = unsigned.pop("compute_digest", None)
    if (
        set(value) != keys or claimed != digest(unsigned)
        or value.get("schema_version") != COMPUTE_SCHEMA
        or value.get("status") != "PASS"
        or value.get("holder_job_id") != HOLDER_JOB_ID
        or value.get("node") != NODE
        or type(value.get("slurm_step_id")) is not str
        or re.fullmatch(r"[0-9]+", value["slurm_step_id"]) is None
        or value.get("cpus_per_task") != CPUS_PER_TASK
        or value.get("gpu_count") != GPU_COUNT
        or value.get("single_srun_attempt") is not True
        or value.get("retry_allowed") is not False
        or value.get("expected_torch_version") != EXPECTED_TORCH_VERSION
        or value.get("expected_hip_version") != EXPECTED_HIP_VERSION
        or value.get("torch_visible_gpu_count") != 0
        or value.get("torch_num_threads") != 1
        or value.get("torch_num_interop_threads") != 1
        or value.get("cpu_thread_environment") != CPU_THREAD_ENVIRONMENT
        or value.get("requested_srun_gpu_export") != REQUESTED_SRUN_GPU_EXPORT
        or value.get("gpu_visibility_environment")
        != EXPECTED_COMPUTE_GPU_VISIBILITY
        or value.get("environment_source") != ENVIRONMENT_SOURCE
        or value.get("compute_reopened_project_authorities")
        != login_project_rows
        or value.get("runtime_authorities") != login_runtime_rows
        or value.get("world4_receipt_sha256")
        != hashlib.sha256(receipt_raw).hexdigest()
        or value.get("world4_receipt_digest") != receipt.get("receipt_digest")
        or value.get("scenario_count") != 7
        or value.get("process_group_zero") is not True
        or value.get("publication_empty") is not True
        or value.get("stage_cache_absent") is not True
        or value.get("renderer_or_vae_loaded") is not False
        or value.get("launch_allowed") is not False
    ):
        raise CpuAdmissionError("compute terminal evidence differs")
    staged = value.get("project_authorities")
    if type(staged) is not dict or set(staged) != set(PROJECT_AUTHORITIES):
        raise CpuAdmissionError("compute staged authority roles differ")
    for role, authority in PROJECT_AUTHORITIES.items():
        suffix = ".json" if role == "scaffold" else ".py"
        _validate_identity_row(
            staged[role], path=str(STAGE_ROOT / (role + suffix)),
            sha256=authority["sha256"], size=authority["size"], mode=0o400,
        )
    _validate_world4(
        receipt, plan, staged_rows=staged,
        runtime_rows=value["runtime_authorities"],
    )


def controller() -> int:
    descriptors: list[int] = []
    project_rows: dict[str, dict[str, Any]] = {}
    runtime_rows: dict[str, dict[str, Any]] = {}
    try:
        # The executing source is the first authority.  Do not resolve it
        # before lstat/open: a symlinked entry must be rejected, not normalized.
        self_path = Path(__file__)
        self_fd, self_raw, self_row = _open_observed(
            self_path, maximum_size=1_048_576,
        )
        descriptors.append(self_fd)
        for role, expected in _project_rows().items():
            descriptor, _raw, row = _open_pinned(
                Path(expected["path"]), expected["sha256"], expected["size"],
            )
            descriptors.append(descriptor); project_rows[role] = row
        for role, expected in RUNTIME_AUTHORITIES.items():
            descriptor, _raw, row = _open_pinned(
                Path(expected["path"]), expected["sha256"], expected["size"],
                executable=role == "python",
            )
            descriptors.append(descriptor); runtime_rows[role] = row
        srun_fd, _srun_raw, srun_row = _open_pinned(
            Path(SRUN_AUTHORITY["path"]), SRUN_AUTHORITY["sha256"],
            SRUN_AUTHORITY["size"], executable=True,
        )
        descriptors.append(srun_fd)
        for path in (TARGET_ROOT, STAGE_ROOT): _fresh(path)
        parent = TARGET_ROOT.parent
        if not parent.is_dir() or parent.resolve(strict=True) != parent:
            raise CpuAdmissionError("diagnostic target parent differs")
        plan = build_compute_plan(); plan_raw = canonical(plan)
        plan_b64 = base64.b64encode(plan_raw).decode("ascii")
        command = build_srun_argv(plan_b64)
        os.mkdir(TARGET_ROOT, 0o700); os.mkdir(EVIDENCE_DIR, 0o700); os.mkdir(LOGS_DIR, 0o700)
        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_SRUN",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "cpus_per_task": CPUS_PER_TASK, "gpu_count": GPU_COUNT,
            "single_srun_attempt": True, "retry_allowed": False,
            "controller": self_row, "held_stdin_sha256": hashlib.sha256(self_raw).hexdigest(),
            "project_authorities": project_rows,
            "runtime_authorities": runtime_rows, "srun_authority": srun_row,
            "compute_plan_digest": plan["plan_digest"],
            "exact_srun_argv": command,
            "exact_srun_argv_digest": digest(command),
            "world4_receipt_path": str(WORLD4_RECEIPT_PATH),
            "evidence_path": str(EVIDENCE_PATH),
        }
        attempt["attempt_digest"] = digest(attempt)
        attempt_raw = _create_json(ATTEMPT_PATH, attempt)
        environment = {
            "PATH": "/usr/bin:/bin", "HOME": "/vast/users/guangyi.chen",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1", **CPU_THREAD_ENVIRONMENT,
            **REQUESTED_SRUN_GPU_EXPORT,
        }
        returncode, stdout_raw, stderr_raw = _run_single_srun(
            command, self_raw, environment,
        )
        if returncode != 0 or stderr_raw or stdout_raw.count(b"\n") != 1:
            raise CpuAdmissionError("single srun terminal streams differ")
        compute_result = json.loads(stdout_raw)
        if stdout_raw != canonical(compute_result) + b"\n":
            raise CpuAdmissionError("single srun stdout is not canonical JSON")
        receipt_raw, receipt = _wait_canonical_json(WORLD4_RECEIPT_PATH)
        _validate_compute_result(
            compute_result, plan=plan, receipt_raw=receipt_raw,
            receipt=receipt, login_project_rows=project_rows,
            login_runtime_rows=runtime_rows,
        )
        if os.path.lexists(STAGE_ROOT):
            raise CpuAdmissionError("compute stage cache remains after srun")
        evidence: dict[str, Any] = {
            "schema_version": SCHEMA, "status": "PASS",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "single_srun_attempt": True, "retry_allowed": False,
            "srun_returncode": returncode,
            "attempt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_digest": attempt["attempt_digest"],
            "compute_digest": compute_result["compute_digest"],
            "world4_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "world4_receipt_digest": receipt["receipt_digest"],
            "stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
            "stderr_empty": True, "process_group_zero": True,
            "publication_empty": True, "stage_cache_absent": True,
            "torch_visible_gpu_count": 0,
            "torch_num_threads": 1, "torch_num_interop_threads": 1,
            "cpu_thread_environment": dict(CPU_THREAD_ENVIRONMENT),
            "requested_srun_gpu_export": dict(REQUESTED_SRUN_GPU_EXPORT),
            "gpu_visibility_environment": dict(EXPECTED_COMPUTE_GPU_VISIBILITY),
            "environment_source": ENVIRONMENT_SOURCE,
            "login_held_project_authorities": project_rows,
            "compute_reopened_project_authorities": compute_result[
                "compute_reopened_project_authorities"
            ],
            "shared_source_rows_equal": True,
            "login_held_runtime_authorities": runtime_rows,
            "compute_reopened_runtime_authorities": compute_result[
                "runtime_authorities"
            ],
            "runtime_rows_equal": True,
            "per_scenario_timeout_seconds": 30,
            "launch_allowed": False, "renderer_or_vae_loaded": False,
        }
        evidence["evidence_digest"] = digest(evidence)
        _create_json(EVIDENCE_PATH, evidence)
        return 0
    finally:
        for descriptor in descriptors:
            try: os.close(descriptor)
            except OSError: pass


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: AUH CPU world4 admission awaits independent review and activation",
            file=sys.stderr,
        )
        return 88
    try:
        if values and values[0] == "compute":
            if len(values) != 2:
                raise CpuAdmissionError("compute argv differs")
            return compute(values[1])
        if values:
            raise CpuAdmissionError("controller argv differs")
        return controller()
    except (OSError, ValueError, KeyError, ImportError, CpuAdmissionError) as error:
        print(f"AUH CPU world4 controller refused: {error}", file=sys.stderr)
        return 96


if __name__ == "__main__":
    raise SystemExit(main())
