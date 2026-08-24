#!/usr/bin/env python3
"""Fail-closed BOX-EXP-013 exact2 controller.

Completion reopens the two new incomplete receipts and every external action
artifact (MP4, native receipt, calibration receipt, and official Gaussian),
then independently recomputes both cross-run same-seed Gaussian proofs.  The
output is media-pair authority for the R4 preoptimizer path; it does not create
Q, a_min, training state, or an optimizer.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import full30_action_arms_incomplete_repair_exact2_plan_v1 as plan_contract  # noqa: E402
import full30_action_arms_incomplete_repair_exact2_generator_v1 as generator  # noqa: E402
import build_full30_action_arms_incomplete_repair_exact2_release_v1 as release  # noqa: E402


CONTROLLER_PLAN_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-controller-plan-v3"
)
REVIEW_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-review-admission-v3"
)
BLIND_REVIEW_MANIFEST_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-blind-review-manifest-v2"
)
BLIND_REVIEW_KEY_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-blind-review-key-v2"
)
COMPUTE_PREFLIGHT_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-compute-preflight-v2"
)
CHILD_SCRATCH_PREPARE_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-scratch-prepare-v1"
)
CHILD_SCRATCH_RETAINED_TERMINAL_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-scratch-retained-terminal-v1"
)
CHILD_TERMINAL_PHYSICAL_ATTESTATION_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-terminal-physical-attestation-v2"
)
CHILD_SCRATCH_FAILURE_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-scratch-failure-v1"
)
CHILD_TASK_SCRATCH_BIND_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-task-scratch-bind-v1"
)
TERMINAL_HOST_GATE_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-terminal-host-memory-gate-v3"
)
COMPLETION_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-completion-v6"
)
CHILD_TERMINAL_READY_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-terminal-ready-v1"
)
PARENT_GENERATION_PRECOMMIT_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-parent-generation-precommit-v1"
)
PARENT_GENERATION_STATUS_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-parent-generation-status-v2"
)
CHILD_TERMINAL_READY_BASENAME = "child-terminal-ready.status"
PARENT_GENERATION_PRECOMMIT_BASENAME = "parent-generation.precommit.json"
PARENT_GENERATION_STATUS_BASENAME = "parent-generation.status"
PARENT_GENERATION_PUBLISH_READY_PREFIX = (
    "BOX-EXP-013-r6-PARENT-PUBLISH-READY "
)
PARENT_GENERATION_PUBLISH_ACK_PREFIX = (
    "BOX-EXP-013-r6-PARENT-PUBLISH-ACK "
)
PARENT_GENERATION_PUBLISH_COMMIT_TOKEN = (
    b"BOX-EXP-013-r6-PARENT-PUBLISH-COMMIT\n"
)
PARENT_GENERATION_PUBLISH_TIMEOUT_SECONDS = 30
PARENT_GENERATION_ROLLBACK_TIMEOUT_SECONDS = 5
HOLDER_JOB = "136140"
HOLDER_NODE = "auh7-1b-gpu-215"
PORTABLE_FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
PORTABLE_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
TERMINAL_RESOURCE_CONTRACT_SHA256 = (
    "c0749bdb7694128fcf5deffb7503c46e3dbe2967adabec4aca4b5c800a5ed01b"
)
TERMINAL_RESOURCE_MODULE_NAME = (
    "_bernini_full30_fit_repair_resource_r6_c0749bdb"
)
_RESOURCE_MODULE_CACHE: Optional[ModuleType] = None
_RESOURCE_MODULE_PRIMITIVES: Optional[tuple[Any, ...]] = None
ALLOWED_NODE_LOCAL_FILESYSTEM_TYPES = frozenset({"ext2/ext3", "xfs", "tmpfs"})
CHILD_SCRATCH_AUTHORITY = "launcher_created_compute_child_tmp"
CHILD_SCRATCH_PARENT = Path("/tmp")
CHILD_SCRATCH_LEAF_PREFIX = "BOX-EXP-013-r6-"
CHILD_SCRATCH_PARENT_UID = 0
CHILD_SCRATCH_PARENT_GID = 0
CHILD_SCRATCH_PARENT_MODE = 0o1777
CHILD_SCRATCH_OWNER_UID = 2012
CHILD_SCRATCH_OWNER_GID = 2000
CHILD_SCRATCH_MODE = 0o700
CHILD_SCRATCH_STATFS_MAGIC = "ef53"
CHILD_SCRATCH_STATFS_TYPE = "ext2/ext3"
CHILD_SCRATCH_MOUNT_POINT = "/"
CHILD_SCRATCH_MOUNT_FILESYSTEM = "ext4"
CHILD_SCRATCH_MOUNT_SOURCE = "/dev/mapper/vgroot-lvroot"
CHILD_SCRATCH_MOUNT_MAJOR_MINOR = "253:0"
CHILD_SCRATCH_PROBE_BASENAME = ".box-exp-013-r6-oexcl-fsync-probe"
CHILD_SCRATCH_PROBE_BYTES = b"BOX-EXP-013-r6-node-local-probe-v1\n"
CHILD_RENDERER_LOAD_LOCK_BASENAME = "renderer-load.lock"
EXTERNAL_ACTION_PREFLIGHT = (
    {
        "seed": 2026080821,
        "basename": "formal_00.mp4",
        "file_sha256": (
            "6f07c69ba2a8ff613ce2c74accfc7578d63602a7e0bf86b486a0b1af9554330b"
        ),
    },
    {
        "seed": 2026080921,
        "basename": "formal_02.mp4",
        "file_sha256": (
            "6c1451ca0c85d151346a2efbacd740e88a113cbfafc5e2373ff97fba9dea6fbc"
        ),
    },
)
REVOKED_LIVE_RELEASE_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/"
    "releases/full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146"
)
REVOKED_LIVE_MATERIALIZATION_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/"
    "release_materializations/full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146"
)
REVOKED_LIVE_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/"
    "data_prep/full30-action-arms-incomplete-exact2-r5-3b59480b-j136140-r1"
)
SHARED_DATA_PREP_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/data_prep"
)
R6_RUN_LEAF_RE = re.compile(
    r"full30-action-arms-incomplete-exact2-r6-[0-9a-f]{8}-j136140-r[1-9][0-9]*"
)
REVOKED_LIVE_LOG_SHA256 = (
    "6df1462415b72bbf966f8b41c125932e7fcd39353c58a8172121577c41f9285a"
)
REVOKED_LIVE_ARCHIVE_SHA256 = "3db741464fa8e5bd258d4d0b7a3f90c5ee9c9eeb78695cab094dcea919fb94b5"
REVOKED_LIVE_MANIFEST_SHA256 = "b0129020fd4134e50e6840a2fdf8e61d0cd32f267bc2dadda8ba17e241d92208"
REVOKED_LIVE_LAUNCHER_SHA256 = "018ddaf9f4ab8c423dd8d081fd7303139d3256eb32c815f8c9d4a8494b4f2e6d"
REVOKED_LIVE_ENVELOPE_SHA256 = "517e31a381fb7e8fe626a1cc71829e59aea4cbe62febb4a68aeb8f39b49c3154"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FULL81_INDEX_SHA256 = hashlib.sha256(
    json.dumps(list(range(81)), separators=(",", ":")).encode("ascii")
).hexdigest()


class ArmsIncompleteExact2ControllerError(RuntimeError):
    """Raised before unbound or over-authorized exact2 state can pass."""


def fail(message: str) -> NoReturn:
    raise ArmsIncompleteExact2ControllerError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_byte_equal(left: Path, right: Path) -> bool:
    left_stat, right_stat = left.stat(), right.stat()
    if left_stat.st_size != right_stat.st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def _validate_cached_resource_module(
    module: ModuleType, resource_path: Path
) -> ModuleType:
    specification = getattr(module, "__spec__", None)
    primitives = tuple(
        getattr(module, name, None)
        for name in (
            "load_compile_smoke_receipt",
            "_load_compile_smoke_receipt_postretention_attested",
            "load_host_cgroup_memory_monitor_start",
            "_journal_prefix",
            "_sample_row",
            "_process_identity_is_live",
        )
    )
    require(
        type(module) is ModuleType
        and module.__name__ == TERMINAL_RESOURCE_MODULE_NAME
        and getattr(module, "__file__", None) == str(resource_path)
        and specification is not None
        and specification.name == TERMINAL_RESOURCE_MODULE_NAME
        and specification.origin == str(resource_path)
        and sys.modules.get(TERMINAL_RESOURCE_MODULE_NAME) is module
        and file_sha256(resource_path) == TERMINAL_RESOURCE_CONTRACT_SHA256
        and getattr(module, "HOST_MEMORY_LIMIT_GIB", None) == 60
        and getattr(module, "HOST_MEMORY_SAFE_CEILING_GIB", None) == 56
        and getattr(module, "HOST_MEMORY_SAMPLE_INTERVAL_NS", None) == 10_000_000
        and getattr(module, "T2V_GPU_MEMORY_LIMIT_GIB", None) == 52
        and all(callable(value) for value in primitives),
        "cached terminal resource module identity differs",
    )
    if _RESOURCE_MODULE_PRIMITIVES is not None:
        require(
            all(
                observed is expected
                for observed, expected in zip(
                    primitives, _RESOURCE_MODULE_PRIMITIVES
                )
            ),
            "cached terminal resource module primitives drifted",
        )
    return module


def _load_or_reuse_resource_contract(
    value: str | Path,
    expected_sha256: str,
    resource_module: Optional[ModuleType] = None,
) -> ModuleType:
    """Load once, then reuse only the exact module object loaded here."""

    global _RESOURCE_MODULE_CACHE, _RESOURCE_MODULE_PRIMITIVES
    resource_path = plain_file(value, "terminal resource contract")
    fixed_resource_path = plain_file(
        generator.METHOD_ROOT
        / "tools"
        / generator.RESOURCE_SPECIALIZED_BASENAME,
        "fixed terminal resource contract",
    )
    require(
        resource_path == fixed_resource_path
        and resource_path.name == generator.RESOURCE_SPECIALIZED_BASENAME
        and expected_sha256 == TERMINAL_RESOURCE_CONTRACT_SHA256
        and file_sha256(resource_path) == TERMINAL_RESOURCE_CONTRACT_SHA256,
        "terminal resource contract path/SHA-256 differs",
    )
    present = sys.modules.get(TERMINAL_RESOURCE_MODULE_NAME)
    if resource_module is not None:
        require(
            _RESOURCE_MODULE_CACHE is resource_module
            and present is resource_module,
            "supplied terminal resource module was not loaded by this controller",
        )
        return _validate_cached_resource_module(resource_module, resource_path)
    if _RESOURCE_MODULE_CACHE is not None:
        require(
            present is _RESOURCE_MODULE_CACHE,
            "cached terminal resource module registration drifted",
        )
        return _validate_cached_resource_module(
            _RESOURCE_MODULE_CACHE, resource_path
        )
    require(
        present is None,
        "untrusted terminal resource specialization is preloaded",
    )
    try:
        loaded = generator.load_resource_contract(resource_path)
        _RESOURCE_MODULE_CACHE = loaded
        _RESOURCE_MODULE_PRIMITIVES = tuple(
            getattr(loaded, name)
            for name in (
                "load_compile_smoke_receipt",
                "_load_compile_smoke_receipt_postretention_attested",
                "load_host_cgroup_memory_monitor_start",
                "_journal_prefix",
                "_sample_row",
                "_process_identity_is_live",
            )
        )
        return _validate_cached_resource_module(loaded, resource_path)
    except Exception as error:
        if sys.modules.get(TERMINAL_RESOURCE_MODULE_NAME) is _RESOURCE_MODULE_CACHE:
            sys.modules.pop(TERMINAL_RESOURCE_MODULE_NAME, None)
        _RESOURCE_MODULE_CACHE = None
        _RESOURCE_MODULE_PRIMITIVES = None
        if isinstance(error, ArmsIncompleteExact2ControllerError):
            raise
        raise ArmsIncompleteExact2ControllerError(str(error)) from error


def plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain directory",
    )
    return resolved


def validate_ffprobe(value: str | Path, expected_sha256: str) -> Path:
    path = plain_file(value, "ffprobe executable")
    require(
        str(path) == PORTABLE_FFPROBE_PATH
        and expected_sha256 == PORTABLE_FFPROBE_SHA256
        and file_sha256(path) == PORTABLE_FFPROBE_SHA256
        and os.access(path, os.X_OK),
        "portable ffprobe canonical path/SHA/executable identity differs",
    )
    return path


def filesystem_type(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["stat", "-f", "-c", "%T", "--", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "cannot identify node-local scratch filesystem"
        ) from error
    observed = completed.stdout.strip()
    require(
        observed in ALLOWED_NODE_LOCAL_FILESYSTEM_TYPES,
        "node-local scratch filesystem type is not allowed",
    )
    return observed


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            f"cannot fsync directory: {path}"
        ) from error


def _decode_mountinfo_path(value: str) -> str:
    replacements = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    result = value
    for encoded, decoded in replacements.items():
        result = result.replace(encoded, decoded)
    return result


def _mountinfo_identity(path: Path) -> Mapping[str, Any]:
    try:
        raw = Path("/proc/self/mountinfo").read_text(encoding="ascii")
        namespace = os.readlink("/proc/self/ns/mnt")
    except (OSError, UnicodeError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "compute-child mount identity is unavailable"
        ) from error
    candidates: list[dict[str, str]] = []
    for line in raw.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) < separator + 4:
            continue
        mount_point = _decode_mountinfo_path(fields[4])
        normalized = mount_point.rstrip("/") or "/"
        path_text = str(path)
        if path_text != normalized and not path_text.startswith(
            normalized.rstrip("/") + "/"
        ):
            continue
        candidates.append(
            {
                "mount_id": fields[0],
                "parent_mount_id": fields[1],
                "major_minor": fields[2],
                "mount_root": _decode_mountinfo_path(fields[3]),
                "mount_point": normalized,
                "mount_options": fields[5],
                "filesystem_type": fields[separator + 1],
                "mount_source": _decode_mountinfo_path(fields[separator + 2]),
                "super_options": fields[separator + 3],
                "mount_namespace": namespace,
            }
        )
    require(bool(candidates), "no mountinfo row owns compute-child /tmp")
    selected = max(candidates, key=lambda row: len(row["mount_point"]))
    require(
        selected["mount_point"] == CHILD_SCRATCH_MOUNT_POINT
        and selected["filesystem_type"] == CHILD_SCRATCH_MOUNT_FILESYSTEM
        and selected["mount_source"] == CHILD_SCRATCH_MOUNT_SOURCE
        and selected["major_minor"] == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and "rw" in selected["mount_options"].split(",")
        and re.fullmatch(r"mnt:\[[0-9]+\]", selected["mount_namespace"])
        is not None,
        "compute-child /tmp is not the pinned local ext4 mount",
    )
    return selected


def _retention_mount_snapshot(outer: Path) -> Mapping[str, Any]:
    """Prove that no bind/submount redirects any scratch descendant."""

    try:
        raw_bytes = Path("/proc/self/mountinfo").read_bytes()
        raw = raw_bytes.decode("ascii")
        namespace = os.readlink("/proc/self/ns/mnt")
    except (OSError, UnicodeError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "scratch retention mount snapshot is unavailable"
        ) from error
    outer_text = str(outer)
    descendant_rows: list[Mapping[str, str]] = []
    for line in raw.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) < separator + 4:
            continue
        mount_point = _decode_mountinfo_path(fields[4]).rstrip("/") or "/"
        if mount_point != outer_text and not mount_point.startswith(
            outer_text + "/"
        ):
            continue
        descendant_rows.append(
            {
                "mount_id": fields[0],
                "parent_mount_id": fields[1],
                "major_minor": fields[2],
                "mount_point": mount_point,
                "filesystem_type": fields[separator + 1],
                "mount_source": _decode_mountinfo_path(fields[separator + 2]),
            }
        )
    require(
        not descendant_rows,
        "scratch root/descendant is a mount point; retention seal is forbidden",
    )
    owning = _mountinfo_identity(outer)
    require(
        owning["mount_point"] == CHILD_SCRATCH_MOUNT_POINT,
        "scratch retention owning mount differs",
    )
    return {
        "mount_namespace": namespace,
        "mountinfo_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "owning_mount": owning,
        "scratch_or_descendant_mount_points": [],
    }


def _statfs_identity(path: Path) -> Mapping[str, str]:
    try:
        completed = subprocess.run(
            ["stat", "-f", "-c", "%t|%T", "--", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "compute-child scratch statfs identity is unavailable"
        ) from error
    fields = completed.stdout.strip().split("|")
    require(
        fields == [CHILD_SCRATCH_STATFS_MAGIC, CHILD_SCRATCH_STATFS_TYPE],
        "compute-child /tmp statfs identity is not pinned ext4",
    )
    return {"magic_hex": fields[0], "raw_filesystem_type": fields[1]}


def _device_major_minor(metadata: os.stat_result) -> str:
    return f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"


def _metadata_value(metadata: os.stat_result) -> Mapping[str, Any]:
    return {
        "device": metadata.st_dev,
        "device_major_minor": _device_major_minor(metadata),
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "link_count": metadata.st_nlink,
    }


def _expected_child_scratch_path(job_id: str, step_id: str) -> Path:
    require(
        job_id == HOLDER_JOB and re.fullmatch(r"[0-9]+", step_id) is not None,
        "child scratch Slurm job/step differs",
    )
    return CHILD_SCRATCH_PARENT / f"{CHILD_SCRATCH_LEAF_PREFIX}{job_id}-{step_id}"


def _controller_plan_reference_without_source_replay(
    value: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str]:
    plan, path, observed = load_json(value, "scratch-bound controller plan", expected_sha256)
    unsigned = dict(plan)
    declared = unsigned.pop("plan_digest", None)
    release_ref = plan.get("release")
    require(
        plan.get("schema_version") == CONTROLLER_PLAN_SCHEMA
        and SHA256_RE.fullmatch(str(declared)) is not None
        and object_sha256(unsigned) == declared
        and type(release_ref) is dict
        and Path(str(release_ref.get("manifest_path"))).is_absolute()
        and SHA256_RE.fullmatch(str(release_ref.get("manifest_file_sha256")))
        is not None
        and SHA256_RE.fullmatch(str(release_ref.get("manifest_digest"))) is not None,
        "scratch-bound controller-plan/release reference differs",
    )
    return plan, path, observed


def prepare_child_scratch(args: argparse.Namespace) -> Mapping[str, Any]:
    """Create and seal the one fixed gpu215-local outer scratch root."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    hostname = socket.gethostname().split(".", 1)[0]
    scratch = _expected_child_scratch_path(job_id, step_id)
    require(hostname == HOLDER_NODE, "child scratch hostname differs")
    require(
        "SLURM_TMPDIR" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH_FSTYPE" not in os.environ
        and "TMPDIR" not in os.environ,
        "caller-provided scratch authority is forbidden, including empty values",
    )
    controller_plan, controller_plan_path, controller_plan_sha = (
        _controller_plan_reference_without_source_replay(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    )
    parent = CHILD_SCRATCH_PARENT
    try:
        parent_metadata = parent.lstat()
        parent_resolved = parent.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "compute-child /tmp parent is unavailable"
        ) from error
    require(
        parent_resolved == parent
        and stat.S_ISDIR(parent_metadata.st_mode)
        and not stat.S_ISLNK(parent_metadata.st_mode)
        and parent_metadata.st_uid == CHILD_SCRATCH_PARENT_UID
        and parent_metadata.st_gid == CHILD_SCRATCH_PARENT_GID
        and stat.S_IMODE(parent_metadata.st_mode) == CHILD_SCRATCH_PARENT_MODE
        and _device_major_minor(parent_metadata) == CHILD_SCRATCH_MOUNT_MAJOR_MINOR,
        "compute-child /tmp parent physical identity differs",
    )
    mount = _mountinfo_identity(parent)
    statfs_value = _statfs_identity(parent)
    require(not os.path.lexists(scratch), "fixed child scratch root already exists")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_descriptor = os.open(parent, parent_flags)
    scratch_descriptor = -1
    try:
        descriptor_parent_metadata = os.fstat(parent_descriptor)
        require(
            (descriptor_parent_metadata.st_dev, descriptor_parent_metadata.st_ino)
            == (parent_metadata.st_dev, parent_metadata.st_ino),
            "compute-child /tmp changed before mkdirat",
        )
        os.mkdir(scratch.name, mode=CHILD_SCRATCH_MODE, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        scratch_descriptor = os.open(scratch.name, parent_flags, dir_fd=parent_descriptor)
        scratch_metadata = os.fstat(scratch_descriptor)
        require(
            stat.S_ISDIR(scratch_metadata.st_mode)
            and scratch_metadata.st_dev == parent_metadata.st_dev
            and scratch_metadata.st_uid == os.geteuid() == CHILD_SCRATCH_OWNER_UID
            and scratch_metadata.st_gid == os.getegid() == CHILD_SCRATCH_OWNER_GID
            and stat.S_IMODE(scratch_metadata.st_mode) == CHILD_SCRATCH_MODE
            and scratch_metadata.st_nlink == 2
            and scratch.resolve(strict=True) == scratch,
            "fresh child scratch root physical identity differs",
        )
        probe_descriptor = os.open(
            CHILD_SCRATCH_PROBE_BASENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=scratch_descriptor,
        )
        try:
            written = os.write(probe_descriptor, CHILD_SCRATCH_PROBE_BYTES)
            require(written == len(CHILD_SCRATCH_PROBE_BYTES), "scratch probe short write")
            os.fsync(probe_descriptor)
            probe_metadata = os.fstat(probe_descriptor)
            require(
                stat.S_ISREG(probe_metadata.st_mode)
                and probe_metadata.st_dev == scratch_metadata.st_dev
                and probe_metadata.st_uid == CHILD_SCRATCH_OWNER_UID
                and probe_metadata.st_gid == CHILD_SCRATCH_OWNER_GID
                and stat.S_IMODE(probe_metadata.st_mode) == 0o600
                and probe_metadata.st_nlink == 1
                and probe_metadata.st_size == len(CHILD_SCRATCH_PROBE_BYTES),
                "scratch O_EXCL/fsync probe identity differs",
            )
        finally:
            os.close(probe_descriptor)
        os.fsync(scratch_descriptor)
        final_scratch_metadata = os.fstat(scratch_descriptor)
        require(
            (final_scratch_metadata.st_dev, final_scratch_metadata.st_ino)
            == (scratch_metadata.st_dev, scratch_metadata.st_ino)
            and final_scratch_metadata.st_nlink == 2
            and os.path.lexists(scratch / CHILD_SCRATCH_PROBE_BASENAME),
            "scratch root changed across O_EXCL/fsync probe",
        )
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "atomic child scratch creation/probe failed; any created root is retained"
        ) from error
    finally:
        if scratch_descriptor >= 0:
            os.close(scratch_descriptor)
        os.close(parent_descriptor)
    release_ref = controller_plan["release"]
    unsigned = {
        "schema_version": CHILD_SCRATCH_PREPARE_SCHEMA,
        "authority": CHILD_SCRATCH_AUTHORITY,
        "controller_plan": {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        },
        "release_manifest": {
            "path": release_ref["manifest_path"],
            "file_sha256": release_ref["manifest_file_sha256"],
            "manifest_digest": release_ref["manifest_digest"],
        },
        "runtime": {
            "slurm_job_id": job_id,
            "slurm_step_id": step_id,
            "hostname": hostname,
            "sole_numbered_compute_child_required": True,
        },
        "pre_environment": {
            "SLURM_TMPDIR": {"present": False, "used_as_authority": False},
            "GADP_NODE_LOCAL_SCRATCH": {"present": False},
            "GADP_NODE_LOCAL_SCRATCH_FSTYPE": {"present": False},
            "TMPDIR": {
                "present": False,
                "value": None,
                "used_as_authority": False,
            },
        },
        "delivery_boundary": {
            "parent_env_u_scrub_required": [
                "SLURM_TMPDIR",
                "TMPDIR",
                "GADP_NODE_LOCAL_SCRATCH",
                "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            ],
            "delivered_to_child_after_parent_scrub_absent": True,
            "preexisting_caller_value_not_used_as_authority": True,
            "slurm_provided_tmpdir_claimed": False,
        },
        "scratch_parent": {
            "path": str(parent),
            "canonical_non_symlink": True,
            **_metadata_value(parent_metadata),
        },
        "filesystem": {
            **statfs_value,
            "mountinfo": mount,
            "scratch_and_parent_same_device": True,
            "local_block_device_required": True,
        },
        "scratch_root": {
            "path": str(scratch),
            "basename": scratch.name,
            "canonical_non_symlink": True,
            **_metadata_value(final_scratch_metadata),
        },
        "retained_probe_file": {
            "path": str(scratch / CHILD_SCRATCH_PROBE_BASENAME),
            "basename": CHILD_SCRATCH_PROBE_BASENAME,
            **_metadata_value(probe_metadata),
            "size_bytes": probe_metadata.st_size,
            "file_sha256": hashlib.sha256(CHILD_SCRATCH_PROBE_BYTES).hexdigest(),
        },
        "creation": {
            "fresh_absent_before_mkdirat": True,
            "mkdirat_create_only": True,
            "parent_directory_fsync_after_mkdir": True,
            "o_excl_no_follow_probe": True,
            "probe_file_fsync": True,
            "probe_retained_as_authorized_forensic_member": True,
            "scratch_directory_fsync_after_probe_retention": True,
            "probe_bytes_sha256": hashlib.sha256(
                CHILD_SCRATCH_PROBE_BYTES
            ).hexdigest(),
        },
        "formal_candidate_count_at_gate": 0,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def _validate_child_scratch_prepare_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, CHILD_SCRATCH_PREPARE_SCHEMA, "child scratch prepare")
    reference_fields = {"path", "file_sha256", "plan_digest"}
    release_fields = {"path", "file_sha256", "manifest_digest"}
    metadata_fields = {
        "path",
        "canonical_non_symlink",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    scratch_fields = metadata_fields | {"basename"}
    probe_fields = {
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
        "size_bytes",
        "file_sha256",
    }
    runtime = value.get("runtime")
    pre_environment = value.get("pre_environment")
    parent = value.get("scratch_parent")
    filesystem = value.get("filesystem")
    scratch = value.get("scratch_root")
    probe = value.get("retained_probe_file")
    creation = value.get("creation")
    plan_ref = value.get("controller_plan")
    release_ref = value.get("release_manifest")
    expected_path = _expected_child_scratch_path(
        str(runtime.get("slurm_job_id", "")) if type(runtime) is dict else "",
        str(runtime.get("slurm_step_id", "")) if type(runtime) is dict else "",
    )
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "authority",
            "controller_plan",
            "release_manifest",
            "runtime",
            "pre_environment",
            "delivery_boundary",
            "scratch_parent",
            "filesystem",
            "scratch_root",
            "retained_probe_file",
            "creation",
            "formal_candidate_count_at_gate",
            "diagnostic_task_count",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(plan_ref) is dict
        and set(plan_ref) == reference_fields
        and Path(str(plan_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(plan_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(plan_ref.get("plan_digest"))) is not None
        and type(release_ref) is dict
        and set(release_ref) == release_fields
        and Path(str(release_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(release_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(release_ref.get("manifest_digest"))) is not None
        and type(runtime) is dict
        and set(runtime)
        == {
            "slurm_job_id",
            "slurm_step_id",
            "hostname",
            "sole_numbered_compute_child_required",
        }
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and re.fullmatch(r"[0-9]+", str(runtime.get("slurm_step_id"))) is not None
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and pre_environment
        == {
            "SLURM_TMPDIR": {"present": False, "used_as_authority": False},
            "GADP_NODE_LOCAL_SCRATCH": {"present": False},
            "GADP_NODE_LOCAL_SCRATCH_FSTYPE": {"present": False},
            "TMPDIR": pre_environment.get("TMPDIR")
            if type(pre_environment) is dict
            else None,
        }
        and type(pre_environment.get("TMPDIR")) is dict
        and set(pre_environment["TMPDIR"])
        == {"present", "value", "used_as_authority"}
        and pre_environment["TMPDIR"]
        == {"present": False, "value": None, "used_as_authority": False}
        and pre_environment["TMPDIR"]["used_as_authority"] is False
        and value.get("delivery_boundary")
        == {
            "parent_env_u_scrub_required": [
                "SLURM_TMPDIR",
                "TMPDIR",
                "GADP_NODE_LOCAL_SCRATCH",
                "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            ],
            "delivered_to_child_after_parent_scrub_absent": True,
            "preexisting_caller_value_not_used_as_authority": True,
            "slurm_provided_tmpdir_claimed": False,
        }
        and type(parent) is dict
        and set(parent) == metadata_fields
        and parent.get("path") == str(CHILD_SCRATCH_PARENT)
        and parent.get("canonical_non_symlink") is True
        and parent.get("device") == os.makedev(253, 0)
        and parent.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and parent.get("uid") == CHILD_SCRATCH_PARENT_UID
        and parent.get("gid") == CHILD_SCRATCH_PARENT_GID
        and parent.get("mode_octal") == "1777"
        and all(type(parent.get(name)) is int and parent[name] >= 0 for name in ("device", "inode", "link_count"))
        and type(filesystem) is dict
        and set(filesystem)
        == {
            "magic_hex",
            "raw_filesystem_type",
            "mountinfo",
            "scratch_and_parent_same_device",
            "local_block_device_required",
        }
        and filesystem.get("magic_hex") == CHILD_SCRATCH_STATFS_MAGIC
        and filesystem.get("raw_filesystem_type") == CHILD_SCRATCH_STATFS_TYPE
        and filesystem.get("scratch_and_parent_same_device") is True
        and filesystem.get("local_block_device_required") is True
        and type(filesystem.get("mountinfo")) is dict
        and set(filesystem["mountinfo"])
        == {
            "mount_id",
            "parent_mount_id",
            "major_minor",
            "mount_root",
            "mount_point",
            "mount_options",
            "filesystem_type",
            "mount_source",
            "super_options",
            "mount_namespace",
        }
        and str(filesystem["mountinfo"].get("mount_id", "")).isdecimal()
        and str(filesystem["mountinfo"].get("parent_mount_id", "")).isdecimal()
        and filesystem["mountinfo"].get("mount_root") == "/"
        and "rw" in str(filesystem["mountinfo"].get("mount_options", "")).split(",")
        and re.fullmatch(
            r"mnt:\[[0-9]+\]",
            str(filesystem["mountinfo"].get("mount_namespace", "")),
        )
        is not None
        and filesystem["mountinfo"].get("mount_point") == CHILD_SCRATCH_MOUNT_POINT
        and filesystem["mountinfo"].get("filesystem_type")
        == CHILD_SCRATCH_MOUNT_FILESYSTEM
        and filesystem["mountinfo"].get("mount_source")
        == CHILD_SCRATCH_MOUNT_SOURCE
        and filesystem["mountinfo"].get("major_minor")
        == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(scratch) is dict
        and set(scratch) == scratch_fields
        and scratch.get("path") == str(expected_path)
        and scratch.get("basename") == expected_path.name
        and scratch.get("canonical_non_symlink") is True
        and scratch.get("device") == parent.get("device") == os.makedev(253, 0)
        and scratch.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and scratch.get("uid") == CHILD_SCRATCH_OWNER_UID
        and scratch.get("gid") == CHILD_SCRATCH_OWNER_GID
        and scratch.get("mode_octal") == "0700"
        and scratch.get("link_count") == 2
        and type(scratch.get("inode")) is int
        and scratch["inode"] > 0
        and type(probe) is dict
        and set(probe) == probe_fields
        and probe.get("path") == str(expected_path / CHILD_SCRATCH_PROBE_BASENAME)
        and probe.get("basename") == CHILD_SCRATCH_PROBE_BASENAME
        and probe.get("device") == scratch.get("device")
        and probe.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(probe.get("inode")) is int
        and probe["inode"] > 0
        and probe.get("uid") == CHILD_SCRATCH_OWNER_UID
        and probe.get("gid") == CHILD_SCRATCH_OWNER_GID
        and probe.get("mode_octal") == "0600"
        and probe.get("link_count") == 1
        and probe.get("size_bytes") == len(CHILD_SCRATCH_PROBE_BYTES)
        and probe.get("file_sha256")
        == hashlib.sha256(CHILD_SCRATCH_PROBE_BYTES).hexdigest()
        and creation
        == {
            "fresh_absent_before_mkdirat": True,
            "mkdirat_create_only": True,
            "parent_directory_fsync_after_mkdir": True,
            "o_excl_no_follow_probe": True,
            "probe_file_fsync": True,
            "probe_retained_as_authorized_forensic_member": True,
            "scratch_directory_fsync_after_probe_retention": True,
            "probe_bytes_sha256": hashlib.sha256(
                CHILD_SCRATCH_PROBE_BYTES
            ).hexdigest(),
        }
        and value.get("formal_candidate_count_at_gate") == 0
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "child scratch prepare field/authority closure differs",
    )
    return value


def _replay_child_scratch_prepare_physical(
    value: Mapping[str, Any], *, require_initial_link_count: bool
) -> None:
    _validate_child_scratch_prepare_shape(value)
    parent = CHILD_SCRATCH_PARENT
    scratch = Path(value["scratch_root"]["path"])
    try:
        parent_metadata = parent.lstat()
        scratch_metadata = scratch.lstat()
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "prepared child scratch physical replay is unavailable"
        ) from error
    expected_parent = value["scratch_parent"]
    expected_scratch = value["scratch_root"]
    expected_probe = value["retained_probe_file"]
    probe = scratch / CHILD_SCRATCH_PROBE_BASENAME
    try:
        probe_metadata = probe.lstat()
        probe_resolved = probe.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "retained scratch probe physical replay is unavailable"
        ) from error
    require(
        parent.resolve(strict=True) == parent
        and scratch.resolve(strict=True) == scratch
        and not parent.is_symlink()
        and not scratch.is_symlink()
        and stat.S_ISDIR(parent_metadata.st_mode)
        and stat.S_ISDIR(scratch_metadata.st_mode)
        and (parent_metadata.st_dev, parent_metadata.st_ino)
        == (expected_parent["device"], expected_parent["inode"])
        and parent_metadata.st_uid == CHILD_SCRATCH_PARENT_UID
        and parent_metadata.st_gid == CHILD_SCRATCH_PARENT_GID
        and stat.S_IMODE(parent_metadata.st_mode) == CHILD_SCRATCH_PARENT_MODE
        and (scratch_metadata.st_dev, scratch_metadata.st_ino)
        == (expected_scratch["device"], expected_scratch["inode"])
        and scratch_metadata.st_uid == CHILD_SCRATCH_OWNER_UID
        and scratch_metadata.st_gid == CHILD_SCRATCH_OWNER_GID
        and stat.S_IMODE(scratch_metadata.st_mode) == CHILD_SCRATCH_MODE
        and (
            (
                require_initial_link_count
                and scratch_metadata.st_nlink == expected_scratch["link_count"] == 2
            )
            or (not require_initial_link_count and scratch_metadata.st_nlink >= 2)
        ),
        "prepared child scratch parent/root identity drifted",
    )
    entries = sorted(path.name for path in scratch.iterdir())
    require(
        probe_resolved == probe
        and not probe.is_symlink()
        and stat.S_ISREG(probe_metadata.st_mode)
        and {
            "path": str(probe),
            "basename": probe.name,
            **_metadata_value(probe_metadata),
            "size_bytes": probe_metadata.st_size,
            "file_sha256": file_sha256(probe),
        }
        == expected_probe
        and (
            (require_initial_link_count and entries == [CHILD_SCRATCH_PROBE_BASENAME])
            or (
                not require_initial_link_count
                and CHILD_SCRATCH_PROBE_BASENAME in entries
                and len(entries) in {1, 2}
            )
        ),
        "retained scratch probe/inventory drifted",
    )
    require(
        _statfs_identity(parent)
        == {
            "magic_hex": value["filesystem"]["magic_hex"],
            "raw_filesystem_type": value["filesystem"]["raw_filesystem_type"],
        }
        and _mountinfo_identity(parent) == value["filesystem"]["mountinfo"],
        "prepared child scratch mount/statfs identity drifted",
    )


def validate_child_scratch_prepare(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Child-only physical validator; parent receipt-only code must not call this."""

    _replay_child_scratch_prepare_physical(value, require_initial_link_count=True)
    return value


def _validate_child_scratch_prepare_postretention(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Receipt-only validator for login-parent/post-review use."""

    return _validate_child_scratch_prepare_shape(value)


def probe_exact81_25fps(path: str | Path, ffprobe_bin: Path) -> Mapping[str, Any]:
    """Physically reopen one MP4 and count every frame with pinned ffprobe."""

    video = plain_file(path, "generated/review MP4")
    command = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ArmsIncompleteExact2ControllerError(
            f"ffprobe failed for generated/review MP4: {video}"
        ) from error
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    require(
        type(streams) is list and len(streams) == 1 and type(streams[0]) is dict,
        f"generated/review MP4 video-stream closure differs: {video}",
    )
    stream = streams[0]
    require(
        stream.get("nb_read_frames") == "81"
        and stream.get("avg_frame_rate") == "25/1"
        and type(stream.get("width")) is int
        and type(stream.get("height")) is int
        and stream["width"] > 0
        and stream["height"] > 0,
        f"generated/review MP4 is not exact81/25fps: {video}",
    )
    return {
        "frame_count": 81,
        "fps": 25,
        "width": stream["width"],
        "height": stream["height"],
        "ffprobe_count_frames": True,
    }


def load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            f"{label} cannot be opened without following links"
        ) from error

    def stable_fields(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_blocks,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        opened = os.fstat(descriptor)
        first_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            first_chunks.append(chunk)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        closed = os.fstat(descriptor)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            f"{label} stable read failed"
        ) from error
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            f"{label} named identity is unavailable"
        ) from error
    raw = b"".join(first_chunks)
    second = b"".join(second_chunks)
    observed = hashlib.sha256(raw).hexdigest()
    require(
        resolved == path
        and stat.S_ISREG(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and opened.st_nlink == 1
        and stable_fields(opened)
        == stable_fields(middle)
        == stable_fields(closed)
        == stable_fields(named)
        and raw == second
        and len(raw) == opened.st_size,
        f"{label} changed during its single-fd stable read",
    )
    if expected_sha256 is not None:
        require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256,
            f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ArmsIncompleteExact2ControllerError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArmsIncompleteExact2ControllerError(f"{label} is not valid JSON") from error
    require(type(result) is dict, f"{label} must be an object")
    require(raw == canonical_json_bytes(result) + b"\n", f"{label} is not canonical JSON")
    return result, path, observed


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    parent = plain_dir(path.parent, "controller output parent")
    require(
        path.is_absolute()
        and parent == path.parent
        and not path.exists()
        and not path.is_symlink(),
        "controller output must be a fresh canonical absolute path",
    )
    raw = canonical_json_bytes(value) + b"\n"
    descriptor: Optional[int] = None
    created_identity: Optional[tuple[int, int]] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        os.fchmod(descriptor, 0o400)
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "controller output write made no progress")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        require(
            stat.S_ISREG(sealed.st_mode)
            and (sealed.st_dev, sealed.st_ino) == created_identity
            and sealed.st_uid == os.geteuid()
            and sealed.st_gid == os.getegid()
            and stat.S_IMODE(sealed.st_mode) == 0o400
            and sealed.st_nlink == 1
            and sealed.st_size == len(raw),
            "controller output opened-file identity differs",
        )
        os.close(descriptor)
        descriptor = None
        _fsync_directory(parent)
        replay_descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            replay_opened = os.fstat(replay_descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(replay_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            replay_closed = os.fstat(replay_descriptor)
        finally:
            os.close(replay_descriptor)
        replayed = b"".join(chunks)
        observed = hashlib.sha256(raw).hexdigest()
        require(
            created_identity is not None
            and (replay_opened.st_dev, replay_opened.st_ino) == created_identity
            and (replay_closed.st_dev, replay_closed.st_ino) == created_identity
            and (
                replay_opened.st_dev,
                replay_opened.st_ino,
                replay_opened.st_uid,
                replay_opened.st_gid,
                replay_opened.st_mode,
                replay_opened.st_nlink,
                replay_opened.st_size,
                replay_opened.st_blocks,
                replay_opened.st_mtime_ns,
                replay_opened.st_ctime_ns,
            )
            == (
                replay_closed.st_dev,
                replay_closed.st_ino,
                replay_closed.st_uid,
                replay_closed.st_gid,
                replay_closed.st_mode,
                replay_closed.st_nlink,
                replay_closed.st_size,
                replay_closed.st_blocks,
                replay_closed.st_mtime_ns,
                replay_closed.st_ctime_ns,
            )
            and stat.S_ISREG(replay_opened.st_mode)
            and replay_opened.st_nlink == 1
            and stat.S_IMODE(replay_opened.st_mode) == 0o400
            and replayed == raw
            and hashlib.sha256(replayed).hexdigest() == observed
            and path.resolve(strict=True) == path,
            "controller output durable replay differs",
        )
    except Exception:
        # A failed create-only receipt is retained as invalid evidence even
        # when a public caller selected a bound-scratch descendant.
        if descriptor is not None:
            os.close(descriptor)
        raise
    return observed


class _LinuxStatfs(ctypes.Structure):
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong),
        ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong),
        ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    ]


def _require_shared_nfs_fd(descriptor: int) -> None:
    """Check NFS through one already-open directory without fork/exec."""

    filesystem = _LinuxStatfs()
    libc = ctypes.CDLL(None, use_errno=True)
    fstatfs = libc.fstatfs
    fstatfs.argtypes = [ctypes.c_int, ctypes.POINTER(_LinuxStatfs)]
    fstatfs.restype = ctypes.c_int
    result = fstatfs(descriptor, ctypes.byref(filesystem))
    require(
        result == 0 and filesystem.f_type == 0x6969,
        "terminal marker parent is not shared NFS",
    )


def _write_shared_terminal_marker(path: Path, value: Mapping[str, Any]) -> str:
    """Durably publish one marker only beneath the fresh r6 shared logs root."""

    parent = path.parent
    run_root = parent.parent
    require(
        path.is_absolute()
        and parent.name == "logs"
        and run_root.parent == SHARED_DATA_PREP_ROOT
        and R6_RUN_LEAF_RE.fullmatch(run_root.name) is not None
        and not str(path).startswith("/tmp/")
        and path.name not in {"", ".", ".."},
        "terminal marker must be a fresh canonical r6 shared logs path",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    shared_descriptor: Optional[int] = None
    run_descriptor: Optional[int] = None
    logs_descriptor: Optional[int] = None
    descriptor: Optional[int] = None
    created_identity: Optional[tuple[int, int]] = None

    def directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
        )

    try:
        shared_descriptor = os.open(SHARED_DATA_PREP_ROOT, directory_flags)
        run_descriptor = os.open(
            run_root.name, directory_flags, dir_fd=shared_descriptor
        )
        logs_descriptor = os.open("logs", directory_flags, dir_fd=run_descriptor)
    except OSError as error:
        for opened_descriptor in (
            logs_descriptor,
            run_descriptor,
            shared_descriptor,
        ):
            if opened_descriptor is not None:
                os.close(opened_descriptor)
        raise ArmsIncompleteExact2ControllerError(
            "terminal marker run/log directory authority cannot be opened"
        ) from error

    shared_opened = os.fstat(shared_descriptor)
    run_opened = os.fstat(run_descriptor)
    logs_opened = os.fstat(logs_descriptor)

    def replay_directory_anchors() -> None:
        try:
            shared_now = os.fstat(shared_descriptor)
            run_now = os.fstat(run_descriptor)
            logs_now = os.fstat(logs_descriptor)
            shared_named = SHARED_DATA_PREP_ROOT.lstat()
            run_named = os.stat(
                run_root.name, dir_fd=shared_descriptor, follow_symlinks=False
            )
            logs_named = os.stat(
                "logs", dir_fd=run_descriptor, follow_symlinks=False
            )
            lexical_run_named = run_root.lstat()
            lexical_logs_named = parent.lstat()
            shared_resolved = SHARED_DATA_PREP_ROOT.resolve(strict=True)
            run_resolved = run_root.resolve(strict=True)
            logs_resolved = parent.resolve(strict=True)
        except OSError as error:
            raise ArmsIncompleteExact2ControllerError(
                "terminal marker run/log directory authority drifted"
            ) from error
        require(
            all(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_nlink >= 2
                for metadata in (shared_now, run_now, logs_now)
            )
            and directory_identity(shared_opened)
            == directory_identity(shared_now)
            == directory_identity(shared_named)
            and directory_identity(run_opened)
            == directory_identity(run_now)
            == directory_identity(run_named)
            == directory_identity(lexical_run_named)
            and directory_identity(logs_opened)
            == directory_identity(logs_now)
            == directory_identity(logs_named)
            == directory_identity(lexical_logs_named)
            and run_now.st_dev == logs_now.st_dev == shared_now.st_dev
            and run_now.st_uid == logs_now.st_uid == os.geteuid()
            and run_now.st_gid == logs_now.st_gid == os.getegid()
            and stat.S_IMODE(run_now.st_mode)
            == stat.S_IMODE(logs_now.st_mode)
            == 0o700
            and shared_resolved == SHARED_DATA_PREP_ROOT
            and run_resolved == run_root
            and logs_resolved == parent,
            "terminal marker run/log fd-to-name topology differs",
        )

    try:
        replay_directory_anchors()
        try:
            os.stat(path.name, dir_fd=logs_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ArmsIncompleteExact2ControllerError(
                "terminal marker fresh-name check failed"
            ) from error
        else:
            raise ArmsIncompleteExact2ControllerError(
                "terminal marker output name already exists"
            )
        # Publication runs inside the signal-masked commit window.  Never
        # fork/exec a path-resolved ``stat`` process here: fstatfs is anchored
        # to the already-open logs directory and completes as one syscall.
        _require_shared_nfs_fd(logs_descriptor)
        replay_directory_anchors()
    except Exception:
        for opened_descriptor in (
            logs_descriptor,
            run_descriptor,
            shared_descriptor,
        ):
            if opened_descriptor is not None:
                os.close(opened_descriptor)
        raise
    raw = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
            dir_fd=logs_descriptor,
        )
        os.fchmod(descriptor, 0o400)
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "terminal marker write made no progress")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        require(
            stat.S_ISREG(sealed.st_mode)
            and (sealed.st_dev, sealed.st_ino) == created_identity
            and sealed.st_uid == os.geteuid()
            and sealed.st_gid == os.getegid()
            and stat.S_IMODE(sealed.st_mode) == 0o400
            and sealed.st_nlink == 1
            and sealed.st_size == len(raw),
            "terminal marker opened-file identity differs",
        )
        os.close(descriptor)
        descriptor = None
        os.fsync(logs_descriptor)
        replay_directory_anchors()
        replay = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=logs_descriptor,
        )
        try:
            replay_stat = os.fstat(replay)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(replay, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            replay_after = os.fstat(replay)
        finally:
            os.close(replay)
        observed = hashlib.sha256(raw).hexdigest()
        replayed = b"".join(chunks)
        marker_named = os.stat(
            path.name, dir_fd=logs_descriptor, follow_symlinks=False
        )
        marker_lexical = path.lstat()
        replay_directory_anchors()
        require(
            created_identity is not None
            and (replay_stat.st_dev, replay_stat.st_ino) == created_identity
            and (replay_after.st_dev, replay_after.st_ino) == created_identity
            and (marker_named.st_dev, marker_named.st_ino) == created_identity
            and (marker_lexical.st_dev, marker_lexical.st_ino) == created_identity
            and replay_stat.st_size == replay_after.st_size == len(raw)
            and replay_stat.st_mtime_ns == replay_after.st_mtime_ns
            and replay_stat.st_nlink == replay_after.st_nlink == 1
            and stat.S_IMODE(replay_stat.st_mode) == 0o400
            and replayed == raw
            and hashlib.sha256(replayed).hexdigest() == observed
            and path.resolve(strict=True) == path,
            "terminal marker durable replay differs",
        )
        return observed
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        # This rollback is restricted to the shared marker inode created by
        # this call.  Any identity uncertainty leaves a poisoned run artifact.
        if created_identity is not None:
            try:
                metadata = os.stat(
                    path.name, dir_fd=logs_descriptor, follow_symlinks=False
                )
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and (metadata.st_dev, metadata.st_ino) == created_identity
                ):
                    os.unlink(path.name, dir_fd=logs_descriptor)
                    os.fsync(logs_descriptor)
            except (OSError, ArmsIncompleteExact2ControllerError):
                pass
        raise
    finally:
        for opened_descriptor in (
            logs_descriptor,
            run_descriptor,
            shared_descriptor,
        ):
            if opened_descriptor is not None:
                os.close(opened_descriptor)


class _PreparedResidentSharedMarker:
    """A fully checked marker publication held at the O_EXCL boundary."""

    def __init__(self, path: Path, value: Mapping[str, Any]) -> None:
        self.path = path
        self.parent = path.parent
        self.run_root = self.parent.parent
        require(
            path.is_absolute()
            and self.parent.name == "logs"
            and self.run_root.parent == SHARED_DATA_PREP_ROOT
            and R6_RUN_LEAF_RE.fullmatch(self.run_root.name) is not None
            and not str(path).startswith("/tmp/")
            and path.name not in {"", ".", ".."},
            "resident terminal marker path differs",
        )
        self.raw = canonical_json_bytes(value) + b"\n"
        self.file_sha256 = hashlib.sha256(self.raw).hexdigest()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        self.shared_descriptor = -1
        self.run_descriptor = -1
        self.logs_descriptor = -1
        self.published = False
        self.published_identity: Optional[tuple[int, ...]] = None
        try:
            self.shared_descriptor = os.open(SHARED_DATA_PREP_ROOT, flags)
            self.run_descriptor = os.open(
                self.run_root.name,
                flags,
                dir_fd=self.shared_descriptor,
            )
            self.logs_descriptor = os.open(
                "logs", flags, dir_fd=self.run_descriptor
            )
            self.shared_identity = self._directory_identity(
                os.fstat(self.shared_descriptor)
            )
            self.run_identity = self._directory_identity(
                os.fstat(self.run_descriptor)
            )
            self.logs_identity = self._directory_identity(
                os.fstat(self.logs_descriptor)
            )
            self._replay_precommit_directory_authority()
            _require_shared_nfs_fd(self.logs_descriptor)
            try:
                os.stat(
                    self.path.name,
                    dir_fd=self.logs_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ArmsIncompleteExact2ControllerError(
                    "resident terminal marker fresh-name check failed"
                ) from error
            else:
                raise ArmsIncompleteExact2ControllerError(
                    "resident terminal marker output name already exists"
                )
            self._replay_held_directory_authority()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
        )

    def _replay_precommit_directory_authority(self) -> None:
        try:
            shared_named = SHARED_DATA_PREP_ROOT.lstat()
            run_named = os.stat(
                self.run_root.name,
                dir_fd=self.shared_descriptor,
                follow_symlinks=False,
            )
            logs_named = os.stat(
                "logs", dir_fd=self.run_descriptor, follow_symlinks=False
            )
            lexical_run = self.run_root.lstat()
            lexical_logs = self.parent.lstat()
            shared_resolved = SHARED_DATA_PREP_ROOT.resolve(strict=True)
            run_resolved = self.run_root.resolve(strict=True)
            logs_resolved = self.parent.resolve(strict=True)
        except OSError as error:
            raise ArmsIncompleteExact2ControllerError(
                "resident marker directory authority cannot be replayed"
            ) from error
        require(
            self._directory_identity(shared_named) == self.shared_identity
            and self._directory_identity(run_named)
            == self._directory_identity(lexical_run)
            == self.run_identity
            and self._directory_identity(logs_named)
            == self._directory_identity(lexical_logs)
            == self.logs_identity
            and shared_resolved == SHARED_DATA_PREP_ROOT
            and run_resolved == self.run_root
            and logs_resolved == self.parent,
            "resident marker absolute/fd directory topology differs",
        )
        self._replay_held_directory_authority()

    def _replay_held_directory_authority(self) -> None:
        shared_now = os.fstat(self.shared_descriptor)
        run_now = os.fstat(self.run_descriptor)
        logs_now = os.fstat(self.logs_descriptor)
        run_named = os.stat(
            self.run_root.name,
            dir_fd=self.shared_descriptor,
            follow_symlinks=False,
        )
        logs_named = os.stat(
            "logs", dir_fd=self.run_descriptor, follow_symlinks=False
        )
        require(
            self._directory_identity(shared_now) == self.shared_identity
            and self._directory_identity(run_now)
            == self._directory_identity(run_named)
            == self.run_identity
            and self._directory_identity(logs_now)
            == self._directory_identity(logs_named)
            == self.logs_identity
            and all(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_nlink >= 2
                for metadata in (shared_now, run_now, logs_now)
            )
            and run_now.st_dev == logs_now.st_dev == shared_now.st_dev
            and run_now.st_uid == logs_now.st_uid == os.geteuid()
            and run_now.st_gid == logs_now.st_gid == os.getegid()
            and stat.S_IMODE(run_now.st_mode)
            == stat.S_IMODE(logs_now.st_mode)
            == 0o700,
            "resident marker held directory authority drifted",
        )

    def close(self) -> None:
        for name in (
            "logs_descriptor",
            "run_descriptor",
            "shared_descriptor",
        ):
            descriptor = getattr(self, name, -1)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, name, -1)

    def publish(self) -> str:
        """Cross only the create/write/fsync boundary using already-held fds."""

        require(not self.published, "resident marker publication was reused")
        self.published = True
        descriptor: Optional[int] = None
        created_identity: Optional[tuple[int, int]] = None
        try:
            descriptor = os.open(
                self.path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=self.logs_descriptor,
            )
            os.fchmod(descriptor, 0o400)
            opened = os.fstat(descriptor)
            created_identity = (opened.st_dev, opened.st_ino)
            offset = 0
            while offset < len(self.raw):
                written = os.write(descriptor, self.raw[offset:])
                require(written > 0, "resident marker write made no progress")
                offset += written
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
            require(
                stat.S_ISREG(sealed.st_mode)
                and (sealed.st_dev, sealed.st_ino) == created_identity
                and sealed.st_uid == os.geteuid()
                and sealed.st_gid == os.getegid()
                and stat.S_IMODE(sealed.st_mode) == 0o400
                and sealed.st_nlink == 1
                and sealed.st_size == len(self.raw),
                "resident marker opened-file identity differs",
            )
            sealed_identity = (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_uid,
                sealed.st_gid,
                sealed.st_mode,
                sealed.st_nlink,
                sealed.st_size,
                sealed.st_blocks,
                sealed.st_mtime_ns,
                sealed.st_ctime_ns,
            )
            os.close(descriptor)
            descriptor = None
            os.fsync(self.logs_descriptor)
            replay = os.open(
                self.path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.logs_descriptor,
            )
            try:
                replay_opened = os.fstat(replay)
                first_chunks: list[bytes] = []
                while True:
                    chunk = os.read(replay, 1024 * 1024)
                    if not chunk:
                        break
                    first_chunks.append(chunk)
                replay_middle = os.fstat(replay)
                os.lseek(replay, 0, os.SEEK_SET)
                second_chunks: list[bytes] = []
                while True:
                    chunk = os.read(replay, 1024 * 1024)
                    if not chunk:
                        break
                    second_chunks.append(chunk)
                replay_second = os.fstat(replay)
                marker_named = os.stat(
                    self.path.name,
                    dir_fd=self.logs_descriptor,
                    follow_symlinks=False,
                )
                self._replay_held_directory_authority()
                replay_final = os.fstat(replay)
                first_raw = b"".join(first_chunks)
                second_raw = b"".join(second_chunks)
                identity = lambda metadata: (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_blocks,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
                require(
                    created_identity is not None
                    and sealed_identity
                    == identity(replay_opened)
                    == identity(replay_middle)
                    == identity(replay_second)
                    == identity(marker_named)
                    == identity(replay_final)
                    and stat.S_ISREG(replay_opened.st_mode)
                    and replay_opened.st_nlink == 1
                    and stat.S_IMODE(replay_opened.st_mode) == 0o400
                    and first_raw == second_raw == self.raw
                    and hashlib.sha256(first_raw).hexdigest()
                    == hashlib.sha256(second_raw).hexdigest()
                    == self.file_sha256,
                    "resident marker durable held-fd replay differs",
                )
            finally:
                os.close(replay)
            self.published_identity = sealed_identity
            return self.file_sha256
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if created_identity is not None:
                try:
                    metadata = os.stat(
                        self.path.name,
                        dir_fd=self.logs_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISREG(metadata.st_mode)
                        and (metadata.st_dev, metadata.st_ino)
                        == created_identity
                    ):
                        os.unlink(self.path.name, dir_fd=self.logs_descriptor)
                        os.fsync(self.logs_descriptor)
                except OSError:
                    pass
            raise

    def rollback_published_marker(self) -> None:
        """Best-effort same-inode rollback for a post-publish chain failure."""

        if self.published_identity is None:
            return
        try:
            metadata = os.stat(
                self.path.name,
                dir_fd=self.logs_descriptor,
                follow_symlinks=False,
            )
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_blocks,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if identity == self.published_identity:
                os.unlink(self.path.name, dir_fd=self.logs_descriptor)
                os.fsync(self.logs_descriptor)
                self.published_identity = None
        except OSError:
            pass

    def replay_published_marker(self) -> None:
        """Reopen the committed inode and repeat the full stable-byte replay."""

        require(
            self.published_identity is not None,
            "resident marker was not durably published",
        )
        descriptor = os.open(
            self.path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=self.logs_descriptor,
        )
        try:
            identity = lambda metadata: (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_blocks,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            opened = os.fstat(descriptor)
            first_chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                first_chunks.append(chunk)
            middle = os.fstat(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            second_chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                second_chunks.append(chunk)
            second = os.fstat(descriptor)
            named = os.stat(
                self.path.name,
                dir_fd=self.logs_descriptor,
                follow_symlinks=False,
            )
            self._replay_held_directory_authority()
            final = os.fstat(descriptor)
            first_raw = b"".join(first_chunks)
            second_raw = b"".join(second_chunks)
            require(
                identity(opened)
                == identity(middle)
                == identity(second)
                == identity(named)
                == identity(final)
                == self.published_identity
                and first_raw == second_raw == self.raw
                and hashlib.sha256(first_raw).hexdigest()
                == hashlib.sha256(second_raw).hexdigest()
                == self.file_sha256,
                "resident marker final stable replay differs",
            )
        finally:
            os.close(descriptor)


class _HeldResidentReceipt:
    """Pin one small signed receipt fd across READY and publication."""

    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        expected_raw: bytes,
        logs_descriptor: int,
        logs_parent: Path,
    ) -> None:
        require(
            path.is_absolute()
            and path.parent == logs_parent
            and SHA256_RE.fullmatch(expected_sha256) is not None,
            "resident held receipt path/SHA topology differs",
        )
        self.path = path
        self.expected_sha256 = expected_sha256
        self.expected_raw = expected_raw
        self.logs_descriptor = logs_descriptor
        self.descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=logs_descriptor,
        )
        self.identity: Optional[tuple[int, ...]] = None
        try:
            self.replay()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_blocks,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def replay(self) -> None:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        opened = os.fstat(self.descriptor)
        first_chunks: list[bytes] = []
        while True:
            chunk = os.read(self.descriptor, 1024 * 1024)
            if not chunk:
                break
            first_chunks.append(chunk)
        middle = os.fstat(self.descriptor)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while True:
            chunk = os.read(self.descriptor, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        second = os.fstat(self.descriptor)
        named = os.stat(
            self.path.name,
            dir_fd=self.logs_descriptor,
            follow_symlinks=False,
        )
        final = os.fstat(self.descriptor)
        first_raw = b"".join(first_chunks)
        second_raw = b"".join(second_chunks)
        identity = self._identity(opened)
        require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_nlink == 1
            and opened.st_uid == os.geteuid()
            and opened.st_gid == os.getegid()
            and stat.S_IMODE(opened.st_mode) == 0o400
            and identity
            == self._identity(middle)
            == self._identity(second)
            == self._identity(named)
            == self._identity(final)
            and (self.identity is None or identity == self.identity)
            and first_raw == second_raw == self.expected_raw
            and hashlib.sha256(first_raw).hexdigest()
            == hashlib.sha256(second_raw).hexdigest()
            == self.expected_sha256,
            "resident held receipt bytes/identity drifted",
        )
        if self.identity is None:
            self.identity = identity

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _shared_terminal_marker_run_root(
    path: Path, expected_basename: str, label: str
) -> Path:
    """Return the exact r6 run root for one canonical logs marker path."""

    require(
        path.is_absolute()
        and path.name == expected_basename
        and path.parent.name == "logs"
        and path.parent.parent.parent == SHARED_DATA_PREP_ROOT
        and R6_RUN_LEAF_RE.fullmatch(path.parent.parent.name) is not None,
        f"{label} path/topology differs",
    )
    logs_root = plain_dir(path.parent, f"{label} logs parent")
    run_root = logs_root.parent
    require(
        run_root.resolve(strict=True) == run_root
        and run_root.parent == SHARED_DATA_PREP_ROOT,
        f"{label} run-root topology differs",
    )
    return run_root


def _fresh_directory(path: Path, label: str) -> Path:
    require(
        path.is_absolute()
        and path != Path("/")
        and not path.exists()
        and not path.is_symlink()
        and path.parent.is_dir()
        and not path.parent.is_symlink(),
        f"{label} must be a fresh absolute directory",
    )
    path.mkdir(mode=0o700)
    return path.resolve(strict=True)


def _copy_create_only(source: Path, output: Path) -> str:
    source = plain_file(source, "blind-review source MP4")
    require(
        output.is_absolute()
        and output.parent.is_dir()
        and not output.parent.is_symlink()
        and not output.exists()
        and not output.is_symlink(),
        "blind-review MP4 output must be fresh and absolute",
    )
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        output_descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
    except Exception:
        os.close(source_descriptor)
        raise
    try:
        with os.fdopen(source_descriptor, "rb") as reader, os.fdopen(
            output_descriptor, "wb"
        ) as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        # Retain an invalid partial create-only output, including beneath a
        # caller-selected bound scratch tree.
        raise
    _fsync_directory(output.parent)
    observed = file_sha256(output)
    source_stat, output_stat = source.stat(), output.stat()
    require(
        observed == file_sha256(source)
        and files_byte_equal(source, output)
        and source != output
        and (source_stat.st_dev, source_stat.st_ino)
        != (output_stat.st_dev, output_stat.st_ino)
        and output_stat.st_nlink == 1,
        "blind-review MP4 copy bytes/inode topology differs",
    )
    return observed


def _signed(unsigned: Mapping[str, Any]) -> Mapping[str, Any]:
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _validate_signed(value: Mapping[str, Any], schema: str, label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        value.get("schema_version") == schema
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        f"{label} schema/digest differs",
    )


def validate_release_tree(
    *, method_root: str | Path, manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    root = Path(method_root)
    require(root.is_absolute(), "release method root must be absolute")
    try:
        metadata, resolved = root.lstat(), root.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "release method root is unavailable"
        ) from error
    require(
        resolved == root
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        "release method root must be one canonical directory",
    )
    manifest, manifest_path_resolved, manifest_sha = load_json(
        manifest_path, "release manifest", expected_manifest_sha256
    )
    try:
        release.validate_manifest(manifest)
    except release.ArmsIncompleteExact2ReleaseError as error:
        raise ArmsIncompleteExact2ControllerError(str(error)) from error
    require(
        manifest_sha == expected_manifest_sha256
        and manifest_path_resolved.parent != root,
        "release manifest must be separately pinned outside its member root",
    )
    expected_paths = {row["path"] for row in manifest["files"]}
    observed_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), "release tree has non-plain member")
        observed_paths.add(path.relative_to(root).as_posix())
    require(observed_paths == expected_paths, "release tree exact member closure differs")
    for row in manifest["files"]:
        member = plain_file(root / row["path"], f"release member {row['path']}")
        member_metadata = member.stat()
        require(
            member_metadata.st_size == row["size"]
            and stat.S_IMODE(member_metadata.st_mode) == row["mode"]
            and file_sha256(member) == row["sha256"],
            f"release member identity differs: {row['path']}",
        )
    return manifest


def _controller_plan_value(
    *, method_root: Path, manifest_path: Path, manifest_sha256: str,
    manifest: Mapping[str, Any], exact2_plan_path: Path,
    exact2_plan_sha256: str, exact2_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema_version": CONTROLLER_PLAN_SCHEMA,
        "experiment_id": "BOX-EXP-013",
        "purpose": "repair only the two missing arms incomplete clips for R4",
        "scientific_target": "two complete same-seed action/before-terminal arms cells",
        "learning_target": "N/A; optimizer-free frozen data authoring",
        "numeric_target": {
            "new_incomplete_full81_pass": [2, 2],
            "external_action_full81_pass": [2, 2],
            "cross_run_same_gaussian_pass": [2, 2],
            "optimizer_updates": 0,
        },
        "dataset": plan_contract.DATASET,
        "training_steps": 0,
        "baseline": "BOX-EXP-011 arms actions 2/2 PASS and incompletes 0/2",
        "revoked_live_attempt": {
            "holder_step": "136140.15",
            "release_root": REVOKED_LIVE_RELEASE_ROOT,
            "materialization_root": REVOKED_LIVE_MATERIALIZATION_ROOT,
            "run_root": REVOKED_LIVE_RUN_ROOT,
            "archive_file_sha256": REVOKED_LIVE_ARCHIVE_SHA256,
            "manifest_file_sha256": REVOKED_LIVE_MANIFEST_SHA256,
            "launcher_file_sha256": REVOKED_LIVE_LAUNCHER_SHA256,
            "deployment_envelope_file_sha256": REVOKED_LIVE_ENVELOPE_SHA256,
            "terminal_log_file_sha256": REVOKED_LIVE_LOG_SHA256,
            "terminal_state": "FAILED",
            "terminal_exit_code": "1:0",
            "elapsed_seconds": 1,
            "permanent_no_go": True,
            "exact_roots_descendants_and_renamed_identity_copies_forbidden": True,
            "generated_candidate_count": 0,
            "gpu_or_model_invocation_count": 0,
        },
        "core_validation": (
            "full81 review of each new incomplete plus completion-time physical "
            "reopen and exact recomputation of both cross-run Gaussian proofs"
        ),
        "release": {
            "method_root": str(method_root),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "frozen_base_archive_sha256": (
                plan_contract.FROZEN_FIT_REPAIR_ARCHIVE_SHA256
            ),
            "frozen_base_manifest_sha256": (
                plan_contract.FROZEN_FIT_REPAIR_MANIFEST_SHA256
            ),
            "prompt_plan_member": release.PLAN_MEMBER,
            "prompt_plan_sha256": manifest["component_pins"]["prompt_plan_sha256"],
        },
        "exact2_plan": {
            "path": str(exact2_plan_path),
            "file_sha256": exact2_plan_sha256,
            "plan_digest": exact2_plan["plan_digest"],
            "formal_candidate_count": 2,
            "comparator_cell_count": 2,
            "new_branch_order": ["incomplete", "incomplete"],
            "diagnostic_task_count": 0,
            "prompt_utf8_sha256": plan_contract.INCOMPLETE_PROMPT_SHA256,
            "forbidden_prompt_token_count": 0,
        },
        "holder": {
            "job_id": 136140,
            "node": HOLDER_NODE,
            "static_plan_binds_live_numbered_child": False,
        },
        "formal_shard_order": [row["shard_id"] for row in exact2_plan["shards"]],
        "authority": {
            "external_action_full81_blind_pass_already_locked": True,
            "independent_full81_review_required_for_each_new_incomplete": True,
            "completion_physically_reopens_external_action_artifacts": True,
            "completion_physically_reopens_new_incomplete_artifacts": True,
            "generation_audit_binds_generated_mp4_path_sha_exact81_25fps": True,
            "post_generation_blind_review_input_create_only": True,
            "controller_never_creates_review_decision": True,
            "blind_copy_canonical_opaque_filename_and_reviewer_dir_required": True,
            "blind_copy_path_and_inode_distinct_from_generated_required": True,
            "blind_copy_bytes_sha_equal_to_generated_required": True,
            "manifest_key_review_topology_chain_replayed_at_completion": True,
            "portable_ffprobe": {
                "path": PORTABLE_FFPROBE_PATH,
                "file_sha256": PORTABLE_FFPROBE_SHA256,
                "caller_override_allowed": False,
                "compute_child_external_action_exact81_25fps_preflight_required": True,
            },
            "scratch_filesystem_type": {
                "source": (
                    "signed child-scratch-prepare receipt for launcher-created "
                    "exact /tmp outer root; statfs and mountinfo are physically replayed"
                ),
                "authority": CHILD_SCRATCH_AUTHORITY,
                "allowed_raw_values": [CHILD_SCRATCH_STATFS_TYPE],
                "required_mount_filesystem": CHILD_SCRATCH_MOUNT_FILESYSTEM,
                "required_mount_source": CHILD_SCRATCH_MOUNT_SOURCE,
                "required_mount_major_minor": CHILD_SCRATCH_MOUNT_MAJOR_MINOR,
                "delivered_child_slurm_tmpdir_required_absent_after_parent_env_u_scrub": True,
                "slurm_tmpdir_used_as_authority": False,
                "caller_declaration_allowed": False,
                "prepare_compute_attestation_retention_receipt_binding_required": True,
                "rank_resource_receipt_chain_replay_required": True,
            },
            "completion_reopens_hashes_and_probes_reviewed_generated_mp4": True,
            "completion_recomputes_cross_run_same_gaussian_per_seed": True,
            "diagnostic_generation_allowed": False,
            "action_generation_allowed": False,
            "q_input_authorized": False,
            "a_min_input_authorized": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "training_authorized": False,
            "prompts_frozen_before_any_new_media": True,
        },
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def build_controller_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.method_root).resolve(strict=True)
    manifest_path = plain_file(args.release_manifest, "release manifest")
    manifest = validate_release_tree(
        method_root=root,
        manifest_path=manifest_path,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
    )
    exact2, exact2_path, exact2_sha = plan_contract.load_plan(
        args.exact2_plan, args.expected_exact2_plan_sha256
    )
    value = _controller_plan_value(
        method_root=root,
        manifest_path=manifest_path,
        manifest_sha256=args.expected_release_manifest_sha256,
        manifest=manifest,
        exact2_plan_path=exact2_path,
        exact2_plan_sha256=exact2_sha,
        exact2_plan=exact2,
    )
    write_create_only(Path(args.output), value)
    return value


def load_controller_plan(
    path: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str, Mapping[str, Any]]:
    value, resolved, observed = load_json(path, "controller plan", expected_sha256)
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    require(
        value.get("schema_version") == CONTROLLER_PLAN_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "controller plan schema/digest differs",
    )
    release_ref = value.get("release", {})
    exact2_ref = value.get("exact2_plan", {})
    manifest = validate_release_tree(
        method_root=release_ref["method_root"],
        manifest_path=release_ref["manifest_path"],
        expected_manifest_sha256=release_ref["manifest_file_sha256"],
    )
    exact2, exact2_path, exact2_sha = plan_contract.load_plan(
        exact2_ref["path"], exact2_ref["file_sha256"]
    )
    expected = _controller_plan_value(
        method_root=Path(release_ref["method_root"]),
        manifest_path=Path(release_ref["manifest_path"]),
        manifest_sha256=release_ref["manifest_file_sha256"],
        manifest=manifest,
        exact2_plan_path=exact2_path,
        exact2_plan_sha256=exact2_sha,
        exact2_plan=exact2,
    )
    require(value == expected, "controller plan replay differs")
    return value, resolved, observed, exact2


def validate_runtime_environment(
    controller_plan: str | Path, expected_plan_sha256: str
) -> Mapping[str, Any]:
    value, _, observed, _ = load_controller_plan(
        controller_plan, expected_plan_sha256
    )
    hostname = socket.gethostname().split(".", 1)[0]
    require(
        os.environ.get("SLURM_JOB_ID") == HOLDER_JOB
        and str(os.environ.get("SLURM_STEP_ID", "")).isdecimal()
        and hostname == HOLDER_NODE,
        "runtime is not the retained 136140/gpu215 numbered child",
    )
    forbidden = {
        "OPTIMIZER_STATE",
        "TRAINING_STEP",
        "FULL30_ACTION_OPTIMIZER",
        "ALLOW_DIAGNOSTIC_GENERATION",
        "ALLOW_ACTION_GENERATION",
    }
    require(
        not any(os.environ.get(name) for name in forbidden),
        "optimizer/action/diagnostic authority environment is forbidden",
    )
    return {
        "controller_plan_digest": value["plan_digest"],
        "controller_plan_file_sha256": observed,
        "slurm_job_id": HOLDER_JOB,
        "slurm_step_id": os.environ["SLURM_STEP_ID"],
        "hostname": hostname,
        "runtime_authorized": True,
        "formal_candidate_count": 2,
        "optimizer_authorized": False,
    }


def seal_compute_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    """Measure portable media probing and scratch provenance inside the child."""

    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "child scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    validate_child_scratch_prepare(prepare)
    prepare_plan, prepare_plan_path, prepare_plan_sha = (
        _controller_plan_reference_without_source_replay(
            prepare["controller_plan"]["path"],
            prepare["controller_plan"]["file_sha256"],
        )
    )
    require(
        str(prepare_plan_path)
        == prepare["controller_plan"]["path"]
        == args.controller_plan
        and prepare_plan_sha
        == prepare["controller_plan"]["file_sha256"]
        and prepare["controller_plan"]["file_sha256"]
        == args.expected_controller_plan_sha256
        and prepare_plan["plan_digest"]
        == prepare["controller_plan"]["plan_digest"]
        and prepare["release_manifest"]
        == {
            "path": prepare_plan["release"]["manifest_path"],
            "file_sha256": prepare_plan["release"]["manifest_file_sha256"],
            "manifest_digest": prepare_plan["release"]["manifest_digest"],
        }
        and "SLURM_TMPDIR" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH_FSTYPE" not in os.environ
        and "TMPDIR" not in os.environ,
        "compute preflight scratch-prepare/environment binding differs",
    )
    # This identity check intentionally precedes plan/source replay.
    ffprobe_bin = validate_ffprobe(
        args.ffprobe_bin, args.expected_ffprobe_sha256
    )
    media_paths = [args.external_action_mp4_seed1, args.external_action_mp4_seed2]
    media_rows: list[Mapping[str, Any]] = []
    for specification, path_value in zip(EXTERNAL_ACTION_PREFLIGHT, media_paths):
        media = plain_file(
            path_value,
            f"compute-preflight external action MP4 seed {specification['seed']}",
        )
        require(
            media.name == specification["basename"]
            and media.parent.name == "source_media"
            and file_sha256(media) == specification["file_sha256"],
            f"compute-preflight external action MP4 binding differs for seed {specification['seed']}",
        )
        probe = probe_exact81_25fps(media, ffprobe_bin)
        require(
            probe.get("frame_count") == 81 and probe.get("fps") == 25,
            f"compute-preflight external action MP4 probe differs for seed {specification['seed']}",
        )
        media_rows.append(
            {
                "seed": specification["seed"],
                "path": str(media),
                "file_sha256": specification["file_sha256"],
                "frame_count": 81,
                "fps": 25,
                "width": probe["width"],
                "height": probe["height"],
                "ffprobe_count_frames": True,
            }
        )
    scratch = Path(prepare["scratch_root"]["path"])
    scratch_fstype = prepare["filesystem"]["raw_filesystem_type"]
    controller_plan, controller_plan_path, controller_plan_sha, exact2 = (
        load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    )
    require(
        str(controller_plan_path) == str(prepare_plan_path)
        and controller_plan_sha == prepare_plan_sha
        and controller_plan["plan_digest"] == prepare_plan["plan_digest"]
        and controller_plan["release"] == prepare_plan["release"],
        "full controller-plan replay differs from scratch prepare",
    )
    expected_action_paths = {
        row["seed"]: row["mp4"]["runtime_path"]
        for row in exact2["external_action_cells"]
    }
    require(
        all(
            row["path"] == expected_action_paths[row["seed"]]
            for row in media_rows
        ),
        "compute-preflight action MP4 paths differ from controller plan",
    )
    runtime = validate_runtime_environment(
        args.controller_plan, args.expected_controller_plan_sha256
    )
    unsigned = {
        "schema_version": COMPUTE_PREFLIGHT_SCHEMA,
        "controller_plan": {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        },
        "runtime": {
            "slurm_job_id": runtime["slurm_job_id"],
            "slurm_step_id": runtime["slurm_step_id"],
            "hostname": runtime["hostname"],
            "sole_numbered_compute_child_required": True,
        },
        "portable_ffprobe": {
            "path": str(ffprobe_bin),
            "file_sha256": PORTABLE_FFPROBE_SHA256,
            "executable": True,
            "caller_override_allowed": False,
        },
        "external_action_media_probes": media_rows,
        "scratch_prepare": {
            "path": str(prepare_path),
            "file_sha256": prepare_sha,
            "receipt_digest": prepare["receipt_digest"],
        },
        "scratch_parent": {
            "path": str(scratch),
            "filesystem_type": scratch_fstype,
            "measurement_command": ["stat", "-f", "-c", "%T", "--", str(scratch)],
            "authority": CHILD_SCRATCH_AUTHORITY,
            "device": prepare["scratch_root"]["device"],
            "inode": prepare["scratch_root"]["inode"],
            "mount_filesystem_type": CHILD_SCRATCH_MOUNT_FILESYSTEM,
            "mount_source": CHILD_SCRATCH_MOUNT_SOURCE,
            "mount_major_minor": CHILD_SCRATCH_MOUNT_MAJOR_MINOR,
            "environment_variable_after_receipt": "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            "slurm_tmpdir_present_before_prepare": False,
        },
        "preflight_sequence": [
            "child_scratch_prepare_physical_replay",
            "portable_ffprobe_canonical_path_sha_executable",
            "external_action_seed2026080821_exact81_25fps",
            "external_action_seed2026080921_exact81_25fps",
            "launcher_created_scratch_statfs_mount_identity",
            "controller_plan_and_runtime_replay",
        ],
        "completed_before_monitor_smoke_model_or_generation": True,
        "source_media_access_before_portable_ffprobe_validation": False,
        "external_action_mp4_paths_match_controller_plan": True,
        "formal_candidate_count_at_gate": 0,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def validate_compute_preflight(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Child-only physical replay of prepare, media, and scratch."""

    return _validate_compute_preflight_core(value, postretention_attested=False)


def _validate_compute_preflight_postretention(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Parent replay: shared refs/media remain physical; child /tmp is receipt-only."""

    return _validate_compute_preflight_core(value, postretention_attested=True)


def _validate_compute_preflight_core(
    value: Mapping[str, Any], *, postretention_attested: bool
) -> Mapping[str, Any]:
    _validate_signed(value, COMPUTE_PREFLIGHT_SCHEMA, "compute preflight")
    expected_fields = {
        "schema_version",
        "controller_plan",
        "runtime",
        "portable_ffprobe",
        "external_action_media_probes",
        "scratch_prepare",
        "scratch_parent",
        "preflight_sequence",
        "completed_before_monitor_smoke_model_or_generation",
        "source_media_access_before_portable_ffprobe_validation",
        "external_action_mp4_paths_match_controller_plan",
        "formal_candidate_count_at_gate",
        "diagnostic_task_count",
        "optimizer_authorized",
        "receipt_digest",
    }
    reference_fields = {"path", "file_sha256", "plan_digest"}
    runtime_fields = {
        "slurm_job_id",
        "slurm_step_id",
        "hostname",
        "sole_numbered_compute_child_required",
    }
    probe_fields = {
        "seed",
        "path",
        "file_sha256",
        "frame_count",
        "fps",
        "width",
        "height",
        "ffprobe_count_frames",
    }
    scratch_fields = {
        "path",
        "filesystem_type",
        "measurement_command",
        "authority",
        "device",
        "inode",
        "mount_filesystem_type",
        "mount_source",
        "mount_major_minor",
        "environment_variable_after_receipt",
        "slurm_tmpdir_present_before_prepare",
    }
    portable = value.get("portable_ffprobe")
    rows = value.get("external_action_media_probes")
    scratch = value.get("scratch_parent")
    controller_ref = value.get("controller_plan")
    runtime = value.get("runtime")
    prepare_ref = value.get("scratch_prepare")
    reference_receipt_fields = {"path", "file_sha256", "receipt_digest"}
    require(
        set(value) == expected_fields
        and type(controller_ref) is dict
        and set(controller_ref) == reference_fields
        and type(runtime) is dict
        and set(runtime) == runtime_fields
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and str(runtime.get("slurm_step_id", "")).isdecimal()
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and portable
        == {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
            "executable": True,
            "caller_override_allowed": False,
        }
        and type(rows) is list
        and len(rows) == 2
        and type(prepare_ref) is dict
        and set(prepare_ref) == reference_receipt_fields
        and Path(str(prepare_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(prepare_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(prepare_ref.get("receipt_digest"))) is not None
        and type(scratch) is dict
        and set(scratch) == scratch_fields
        and scratch.get("filesystem_type") == CHILD_SCRATCH_STATFS_TYPE
        and scratch.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(scratch.get("device")) is int
        and scratch["device"] >= 0
        and type(scratch.get("inode")) is int
        and scratch["inode"] > 0
        and scratch.get("mount_filesystem_type") == CHILD_SCRATCH_MOUNT_FILESYSTEM
        and scratch.get("mount_source") == CHILD_SCRATCH_MOUNT_SOURCE
        and scratch.get("mount_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and scratch.get("measurement_command")
        == ["stat", "-f", "-c", "%T", "--", scratch.get("path")]
        and scratch.get("environment_variable_after_receipt")
        == "GADP_NODE_LOCAL_SCRATCH_FSTYPE"
        and scratch.get("slurm_tmpdir_present_before_prepare") is False
        and value.get("preflight_sequence")
        == [
            "child_scratch_prepare_physical_replay",
            "portable_ffprobe_canonical_path_sha_executable",
            "external_action_seed2026080821_exact81_25fps",
            "external_action_seed2026080921_exact81_25fps",
            "launcher_created_scratch_statfs_mount_identity",
            "controller_plan_and_runtime_replay",
        ]
        and value.get("completed_before_monitor_smoke_model_or_generation") is True
        and value.get("source_media_access_before_portable_ffprobe_validation") is False
        and value.get("external_action_mp4_paths_match_controller_plan") is True
        and value.get("formal_candidate_count_at_gate") == 0
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "compute preflight field/authority closure differs",
    )
    prepare, prepare_path, prepare_sha = load_json(
        prepare_ref["path"],
        "compute-preflight scratch prepare",
        prepare_ref["file_sha256"],
    )
    if postretention_attested:
        _validate_child_scratch_prepare_postretention(prepare)
    else:
        _replay_child_scratch_prepare_physical(
            prepare, require_initial_link_count=False
        )
    prepared_plan, prepared_plan_path, prepared_plan_sha = (
        _controller_plan_reference_without_source_replay(
            prepare["controller_plan"]["path"],
            prepare["controller_plan"]["file_sha256"],
        )
    )
    require(
        str(prepare_path) == prepare_ref["path"]
        and prepare_sha == prepare_ref["file_sha256"]
        and prepare["receipt_digest"] == prepare_ref["receipt_digest"]
        and str(prepared_plan_path) == prepare["controller_plan"]["path"]
        and prepared_plan_sha == prepare["controller_plan"]["file_sha256"]
        and prepared_plan["plan_digest"]
        == prepare["controller_plan"]["plan_digest"]
        and prepare["release_manifest"]
        == {
            "path": prepared_plan["release"]["manifest_path"],
            "file_sha256": prepared_plan["release"]["manifest_file_sha256"],
            "manifest_digest": prepared_plan["release"]["manifest_digest"],
        }
        and prepare["scratch_root"]["path"] == scratch["path"]
        and prepare["scratch_root"]["device"] == scratch["device"]
        and prepare["scratch_root"]["inode"] == scratch["inode"]
        and prepare["filesystem"]["raw_filesystem_type"]
        == scratch["filesystem_type"],
        "compute-preflight scratch prepare reference differs",
    )
    controller_plan, controller_plan_path, controller_plan_sha, exact2 = (
        load_controller_plan(
            controller_ref["path"], controller_ref["file_sha256"]
        )
    )
    expected_action_paths = {
        row["seed"]: row["mp4"]["runtime_path"]
        for row in exact2["external_action_cells"]
    }
    require(
        str(controller_plan_path) == controller_ref["path"]
        and controller_plan_sha == controller_ref["file_sha256"]
        and controller_plan["plan_digest"] == controller_ref["plan_digest"],
        "compute preflight controller-plan reference differs",
    )
    require(
        str(controller_plan_path) == str(prepared_plan_path)
        and controller_plan_sha == prepared_plan_sha
        and controller_plan["plan_digest"] == prepared_plan["plan_digest"]
        and controller_plan["release"] == prepared_plan["release"],
        "compute preflight full plan differs from scratch prepare",
    )
    for row, specification in zip(rows, EXTERNAL_ACTION_PREFLIGHT):
        require(
            type(row) is dict
            and set(row) == probe_fields
            and row.get("seed") == specification["seed"]
            and Path(str(row.get("path"))).name == specification["basename"]
            and row.get("path") == expected_action_paths[specification["seed"]]
            and row.get("file_sha256") == specification["file_sha256"]
            and row.get("frame_count") == 81
            and row.get("fps") == 25
            and type(row.get("width")) is int
            and row["width"] > 0
            and type(row.get("height")) is int
            and row["height"] > 0
            and row.get("ffprobe_count_frames") is True,
            f"compute preflight external action row differs for seed {specification['seed']}",
        )
    ffprobe_bin = validate_ffprobe(
        PORTABLE_FFPROBE_PATH, PORTABLE_FFPROBE_SHA256
    )
    for row in rows:
        media = plain_file(row["path"], f"preflight replay seed {row['seed']}")
        require(
            file_sha256(media) == row["file_sha256"],
            f"preflight external action hash replay differs for seed {row['seed']}",
        )
        probe = probe_exact81_25fps(media, ffprobe_bin)
        require(
            probe.get("frame_count") == 81
            and probe.get("fps") == 25
            and probe.get("width") == row["width"]
            and probe.get("height") == row["height"],
            f"preflight external action probe replay differs for seed {row['seed']}",
        )
    return value


def _validate_blind_packet(
    *, review: Mapping[str, Any], exact2_plan: Mapping[str, Any],
    generation_audit: Mapping[str, Any], ffprobe_bin: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest_ref = review.get("blind_review_manifest")
    key_ref = review.get("sealed_key")
    generation_ref = review.get("generation_audit")
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    require(
        type(manifest_ref) is dict
        and set(manifest_ref) == reference_fields
        and type(key_ref) is dict
        and set(key_ref) == reference_fields
        and type(generation_ref) is dict
        and set(generation_ref) == reference_fields,
        "review packet reference field closure differs",
    )
    generation_reopened, _, generation_sha = load_json(
        generation_ref["path"],
        "review-bound generation audit",
        generation_ref["file_sha256"],
    )
    require(
        generation_reopened == generation_audit
        and generation_sha == generation_ref["file_sha256"]
        and generation_audit.get("receipt_digest")
        == generation_ref["receipt_digest"],
        "review-bound generation audit differs",
    )
    manifest, manifest_path, manifest_sha = load_json(
        manifest_ref["path"],
        "blind review manifest",
        manifest_ref["file_sha256"],
    )
    key, key_path, key_sha = load_json(
        key_ref["path"], "blind review sealed key", key_ref["file_sha256"]
    )
    _validate_signed(manifest, BLIND_REVIEW_MANIFEST_SCHEMA, "blind review manifest")
    _validate_signed(key, BLIND_REVIEW_KEY_SCHEMA, "blind review sealed key")
    manifest_fields = {
        "schema_version",
        "packet_id",
        "plan_digest",
        "generation_audit_digest",
        "required_ffprobe",
        "candidate_count",
        "blinded",
        "review_decision_present",
        "sample_order",
        "samples",
        "receipt_digest",
    }
    key_fields = {
        "schema_version",
        "packet_id",
        "plan_digest",
        "generation_audit",
        "review_manifest",
        "required_ffprobe",
        "candidate_count",
        "mappings",
        "review_decision_present",
        "receipt_digest",
    }
    sample_fields = {
        "sample_id",
        "opaque_filename",
        "reviewer_mp4_path",
        "reviewer_mp4_file_sha256",
        "frame_count",
        "fps",
    }
    mapping_fields = {
        "sample_id",
        "opaque_filename",
        "candidate_id",
        "candidate_receipt_path",
        "candidate_receipt_file_sha256",
        "candidate_receipt_digest",
        "generated_mp4_path",
        "generated_mp4_file_sha256",
        "reviewer_mp4_path",
        "reviewer_mp4_file_sha256",
    }
    samples, mappings = manifest.get("samples"), key.get("mappings")
    require(
        set(manifest) == manifest_fields
        and set(key) == key_fields
        and manifest.get("packet_id") == key.get("packet_id") == review.get("packet_id")
        and re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("packet_id")))
        is not None
        and manifest.get("plan_digest")
        == key.get("plan_digest")
        == exact2_plan["plan_digest"]
        and manifest.get("generation_audit_digest")
        == generation_audit["receipt_digest"]
        and manifest.get("required_ffprobe")
        == key.get("required_ffprobe")
        == review.get("required_ffprobe")
        == {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
        }
        and manifest.get("candidate_count") == key.get("candidate_count") == 2
        and manifest.get("blinded") is True
        and manifest.get("review_decision_present") is False
        and key.get("review_decision_present") is False
        and key.get("generation_audit") == generation_ref
        and key.get("review_manifest")
        == {
            "path": str(manifest_path),
            "file_sha256": manifest_sha,
            "receipt_digest": manifest["receipt_digest"],
        }
        and str(key_path) == key_ref["path"]
        and key_path.name == "sealed-key.json"
        and manifest_path.name == "review-manifest.json"
        and manifest_path.parent.name == "reviewer"
        and manifest_path.parent == key_path.parent / "reviewer"
        and key_sha == key_ref["file_sha256"]
        and key.get("receipt_digest") == key_ref["receipt_digest"]
        and manifest.get("receipt_digest") == manifest_ref["receipt_digest"]
        and type(samples) is list
        and type(mappings) is list
        and len(samples) == len(mappings) == 2
        and manifest.get("sample_order")
        == [sample.get("sample_id") for sample in samples],
        "blind review packet top-level closure differs",
    )
    audit_by_candidate = {
        row["candidate_id"]: row for row in generation_audit["candidate_receipts"]
    }
    mapping_by_sample: dict[str, Mapping[str, Any]] = {}
    for sample, mapping in zip(samples, mappings):
        require(
            type(sample) is dict
            and set(sample) == sample_fields
            and type(mapping) is dict
            and set(mapping) == mapping_fields
            and sample.get("sample_id") == mapping.get("sample_id")
            and re.fullmatch(r"[0-9a-f]{24}", str(sample.get("sample_id")))
            is not None
            and sample.get("opaque_filename")
            == mapping.get("opaque_filename")
            == f"sample_{sample.get('sample_id')}.mp4"
            and sample.get("frame_count") == 81
            and sample.get("fps") == 25
            and sample.get("reviewer_mp4_path")
            == mapping.get("reviewer_mp4_path")
            and sample.get("reviewer_mp4_file_sha256")
            == mapping.get("reviewer_mp4_file_sha256"),
            "blind review sample/mapping field closure differs",
        )
        candidate_id = mapping["candidate_id"]
        require(
            candidate_id in audit_by_candidate,
            "blind review mapping names an unknown candidate",
        )
        audit_row = audit_by_candidate[candidate_id]
        require(
            mapping["candidate_receipt_path"]
            == audit_row["candidate_receipt_path"]
            and mapping["candidate_receipt_file_sha256"]
            == audit_row["candidate_receipt_file_sha256"]
            and mapping["candidate_receipt_digest"]
            == audit_row["candidate_receipt_digest"]
            and mapping["generated_mp4_path"]
            == audit_row["generated_mp4_path"]
            and mapping["generated_mp4_file_sha256"]
            == audit_row["generated_mp4_file_sha256"]
            and sample["reviewer_mp4_file_sha256"]
            == audit_row["generated_mp4_file_sha256"],
            "blind review mapping differs from generation audit/candidate receipt",
        )
        generated_mp4 = plain_file(
            mapping["generated_mp4_path"],
            f"blind review generated source {candidate_id}",
        )
        review_mp4 = plain_file(
            sample["reviewer_mp4_path"], f"blind review sample {sample['sample_id']}"
        )
        generated_stat, review_stat = generated_mp4.stat(), review_mp4.stat()
        expected_review_mp4 = manifest_path.parent / sample["opaque_filename"]
        require(
            str(generated_mp4) == mapping["generated_mp4_path"]
            and str(review_mp4)
            == sample["reviewer_mp4_path"]
            == mapping["reviewer_mp4_path"]
            == str(expected_review_mp4)
            and review_mp4.parent == manifest_path.parent
            and review_mp4.name == sample["opaque_filename"]
            and generated_mp4 != review_mp4
            and (generated_stat.st_dev, generated_stat.st_ino)
            != (review_stat.st_dev, review_stat.st_ino)
            and review_stat.st_nlink == 1
            and file_sha256(generated_mp4)
            == mapping["generated_mp4_file_sha256"]
            == file_sha256(review_mp4)
            == sample["reviewer_mp4_file_sha256"]
            and files_byte_equal(generated_mp4, review_mp4),
            "blind review opaque-copy path/bytes/inode topology differs",
        )
        probe_exact81_25fps(review_mp4, ffprobe_bin)
        require(
            sample["sample_id"] not in mapping_by_sample,
            "duplicate blind review sample id",
        )
        mapping_by_sample[sample["sample_id"]] = mapping
    require(
        set(audit_by_candidate)
        == {mapping["candidate_id"] for mapping in mappings},
        "blind review packet candidate closure differs",
    )
    return manifest, key


def validate_review_admission(
    value: Mapping[str, Any], exact2_plan: Mapping[str, Any],
    generation_audit: Mapping[str, Any], ffprobe_bin: Path,
) -> Mapping[str, Any]:
    ffprobe_bin = validate_ffprobe(ffprobe_bin, PORTABLE_FFPROBE_SHA256)
    _validate_signed(value, REVIEW_SCHEMA, "review admission")
    expected_fields = {
        "schema_version",
        "packet_id",
        "reviewer_receipt_id",
        "plan_digest",
        "generation_audit",
        "blind_review_manifest",
        "sealed_key",
        "required_ffprobe",
        "review_population",
        "reviewer_independent_of_generator",
        "reviewer_independent_of_materializer",
        "reviewer_did_not_read_sealed_key_before_decisions",
        "candidate_count",
        "external_action_count",
        "external_actions_reused_from_locked_blind_review",
        "diagnostic_candidate_count",
        "candidate_reviews",
        "receipt_digest",
    }
    row_fields = {
        "sample_id",
        "opaque_filename",
        "candidate_id",
        "semantic_branch",
        "generated_mp4_path",
        "generated_mp4_file_sha256",
        "reviewer_mp4_path",
        "reviewer_mp4_file_sha256",
        "frame_count",
        "fps",
        "reviewed_frame_count",
        "reviewed_frame_indices_sha256",
        "all_81_frames_reviewed",
        "classification",
        "verdict",
        "terminal_action_state_absent",
    }
    rows = value.get("candidate_reviews")
    require(
        set(value) == expected_fields
        and value.get("plan_digest") == exact2_plan["plan_digest"]
        and re.fullmatch(r"[0-9a-f]{32}", str(value.get("reviewer_receipt_id")))
        is not None
        and value.get("review_population") == plan_contract.DATASET
        and value.get("required_ffprobe")
        == {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
        }
        and value.get("reviewer_independent_of_generator") is True
        and value.get("reviewer_independent_of_materializer") is True
        and value.get("reviewer_did_not_read_sealed_key_before_decisions") is True
        and value.get("candidate_count") == 2
        and value.get("external_action_count") == 2
        and value.get("external_actions_reused_from_locked_blind_review") is True
        and value.get("diagnostic_candidate_count") == 0
        and type(rows) is list
        and len(rows) == 2,
        "review admission top-level/population closure differs",
    )
    manifest, key = _validate_blind_packet(
        review=value,
        exact2_plan=exact2_plan,
        generation_audit=generation_audit,
        ffprobe_bin=ffprobe_bin,
    )
    mappings = {row["sample_id"]: row for row in key["mappings"]}
    require(
        [row.get("sample_id") for row in rows] == manifest["sample_order"],
        "review row/sample order differs",
    )
    observed_candidates: set[str] = set()
    for row in rows:
        require(
            type(row) is dict and set(row) == row_fields,
            "review row field closure differs",
        )
        sample_id = row["sample_id"]
        require(sample_id in mappings, "review row sample id is unknown")
        mapping = mappings[sample_id]
        candidate_id = mapping["candidate_id"]
        require(
            row.get("candidate_id") == candidate_id
            and row.get("opaque_filename") == mapping["opaque_filename"]
            and row.get("semantic_branch") == "incomplete"
            and row.get("generated_mp4_path")
            == mapping["generated_mp4_path"]
            and row.get("generated_mp4_file_sha256")
            == mapping["generated_mp4_file_sha256"]
            and row.get("reviewer_mp4_path") == mapping["reviewer_mp4_path"]
            and row.get("reviewer_mp4_file_sha256")
            == mapping["reviewer_mp4_file_sha256"]
            and row.get("generated_mp4_file_sha256")
            == row.get("reviewer_mp4_file_sha256")
            and row.get("frame_count") == 81
            and row.get("fps") == 25
            and row.get("reviewed_frame_count") == 81
            and row.get("reviewed_frame_indices_sha256") == FULL81_INDEX_SHA256
            and row.get("all_81_frames_reviewed") is True
            and row.get("classification") == "correct_before_terminal_and_hold"
            and row.get("verdict") == "pass"
            and row.get("terminal_action_state_absent") is True,
            f"full81 generated-MP4 review binding failed: {candidate_id}",
        )
        generated_mp4 = plain_file(
            row["generated_mp4_path"], f"review generated MP4 {candidate_id}"
        )
        reviewer_mp4 = plain_file(
            row["reviewer_mp4_path"], f"review copy MP4 {candidate_id}"
        )
        require(
            str(generated_mp4) == row["generated_mp4_path"]
            and str(reviewer_mp4) == row["reviewer_mp4_path"]
            and reviewer_mp4.parent
            == Path(value["blind_review_manifest"]["path"]).parent
            and reviewer_mp4.name == row["opaque_filename"]
            and generated_mp4 != reviewer_mp4
            and (generated_mp4.stat().st_dev, generated_mp4.stat().st_ino)
            != (reviewer_mp4.stat().st_dev, reviewer_mp4.stat().st_ino)
            and reviewer_mp4.stat().st_nlink == 1
            and file_sha256(generated_mp4)
            == row["generated_mp4_file_sha256"]
            and file_sha256(reviewer_mp4)
            == row["reviewer_mp4_file_sha256"]
            and files_byte_equal(generated_mp4, reviewer_mp4),
            f"review MP4 physical hash replay differs: {candidate_id}",
        )
        generated_probe = probe_exact81_25fps(generated_mp4, ffprobe_bin)
        reviewer_probe = probe_exact81_25fps(reviewer_mp4, ffprobe_bin)
        require(
            generated_probe.get("frame_count")
            == reviewer_probe.get("frame_count")
            == 81
            and generated_probe.get("fps") == reviewer_probe.get("fps") == 25,
            f"review MP4 exact81/25fps replay differs: {candidate_id}",
        )
        require(candidate_id not in observed_candidates, "duplicate review candidate")
        observed_candidates.add(candidate_id)
    require(
        observed_candidates
        == {task["candidate_id"] for task in exact2_plan["admission_tasks"]},
        "review candidate closure differs",
    )
    return value


def validate_exact2_audit(
    value: Mapping[str, Any],
    exact2_plan: Mapping[str, Any],
    ffprobe_bin: Path,
    resource_module: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    """Child validation with physical scratch replay."""

    return _validate_exact2_audit_core(
        value,
        exact2_plan,
        ffprobe_bin,
        resource_module,
        postretention_attested=False,
    )


def _validate_exact2_audit_postretention_attested(
    value: Mapping[str, Any],
    exact2_plan: Mapping[str, Any],
    ffprobe_bin: Path,
    resource_module: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    return _validate_exact2_audit_core(
        value,
        exact2_plan,
        ffprobe_bin,
        resource_module,
        postretention_attested=True,
    )


def _validate_exact2_audit_core(
    value: Mapping[str, Any],
    exact2_plan: Mapping[str, Any],
    ffprobe_bin: Path,
    resource_module: Optional[ModuleType],
    *,
    postretention_attested: bool,
) -> Mapping[str, Any]:
    ffprobe_bin = validate_ffprobe(ffprobe_bin, PORTABLE_FFPROBE_SHA256)
    _validate_signed(value, generator.AUDIT_SCHEMA, "exact2 generation audit")
    expected_fields = {
        "schema_version",
        "plan_path",
        "plan_file_sha256",
        "plan_digest",
        "dataset",
        "candidate_count",
        "comparator_cell_count",
        "new_branch_order",
        "compute_preflight",
        "task_scratch_bind",
        "rank_resource_scratch_binding",
        "shard_receipt",
        "candidate_receipts",
        "cross_run_same_gaussian_pair_proofs",
        "all_candidates_exact81",
        "independent_full81_review_performed",
        "review_admission_authorized",
        "q_input_authorized",
        "a_min_input_authorized",
        "training_performed",
        "optimizer_created",
        "optimizer_authorized",
        "diagnostic_task_count",
        "diagnostic_generation_observed_or_required",
        "action_generation_observed_or_required",
        "receipt_digest",
    }
    require(set(value) == expected_fields, "exact2 generation audit field closure differs")
    try:
        audit_plan, audit_plan_path, audit_plan_sha = plan_contract.load_plan(
            value["plan_path"], value["plan_file_sha256"]
        )
    except plan_contract.ArmsIncompleteExact2PlanError as error:
        raise ArmsIncompleteExact2ControllerError(str(error)) from error
    require(
        audit_plan == exact2_plan
        and str(audit_plan_path) == value["plan_path"]
        and audit_plan_sha == value["plan_file_sha256"],
        "exact2 generation audit plan replay differs",
    )
    compute_ref = value.get("compute_preflight")
    task_bind_ref = value.get("task_scratch_bind")
    rank_binding = value.get("rank_resource_scratch_binding")
    shard_ref = value.get("shard_receipt")
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    rank_fields = {
        "preflight_scratch_parent_path",
        "rank_task_scratch_path",
        "filesystem_type",
        "source_environment_variable",
        "preflight_stat_f_matches_rank_resource_receipt",
        "compile_smoke_runtime_matches_rank_resource_receipt",
    }
    require(
        isinstance(compute_ref, Mapping)
        and set(compute_ref) == reference_fields
        and isinstance(task_bind_ref, Mapping)
        and set(task_bind_ref) == reference_fields
        and isinstance(shard_ref, Mapping)
        and set(shard_ref) == reference_fields
        and isinstance(rank_binding, Mapping)
        and set(rank_binding) == rank_fields
        and Path(str(compute_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(compute_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(compute_ref.get("receipt_digest"))) is not None
        and Path(str(task_bind_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(task_bind_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(task_bind_ref.get("receipt_digest"))) is not None
        and Path(str(shard_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(shard_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(shard_ref.get("receipt_digest"))) is not None
        and Path(str(rank_binding.get("preflight_scratch_parent_path"))).is_absolute()
        and Path(str(rank_binding.get("rank_task_scratch_path"))).parent
        == Path(str(rank_binding.get("preflight_scratch_parent_path")))
        and rank_binding.get("filesystem_type")
        in ALLOWED_NODE_LOCAL_FILESYSTEM_TYPES
        and rank_binding.get("source_environment_variable")
        == "GADP_NODE_LOCAL_SCRATCH_FSTYPE"
        and rank_binding.get("preflight_stat_f_matches_rank_resource_receipt")
        is True
        and rank_binding.get("compile_smoke_runtime_matches_rank_resource_receipt")
        is True,
        "generation stat-f/rank resource reference closure differs",
    )
    preflight, preflight_path, preflight_sha = load_json(
        compute_ref["path"],
        "generation-bound compute preflight",
        compute_ref["file_sha256"],
    )
    if postretention_attested:
        _validate_compute_preflight_postretention(preflight)
    else:
        validate_compute_preflight(preflight)
    require(
        str(preflight_path) == compute_ref["path"]
        and preflight_sha == compute_ref["file_sha256"]
        and preflight.get("receipt_digest") == compute_ref["receipt_digest"]
        and preflight["scratch_parent"]["path"]
        == rank_binding["preflight_scratch_parent_path"]
        and preflight["scratch_parent"]["filesystem_type"]
        == rank_binding["filesystem_type"],
        "generation compute-preflight reference replay differs",
    )
    task_bind, task_bind_path, task_bind_sha = generator.load_task_scratch_bind(
        task_bind_ref["path"], task_bind_ref["file_sha256"]
    )
    require(
        str(task_bind_path) == task_bind_ref["path"]
        and task_bind_sha == task_bind_ref["file_sha256"]
        and task_bind["receipt_digest"] == task_bind_ref["receipt_digest"]
        and task_bind["compute_preflight"] == compute_ref
        and task_bind["scratch_inner"]["path"]
        == rank_binding["rank_task_scratch_path"],
        "generation task-scratch bind reference replay differs",
    )
    if not postretention_attested:
        generator.replay_task_scratch_bind_physical(task_bind)
    resource = _load_or_reuse_resource_contract(
        generator.METHOD_ROOT
        / "tools"
        / generator.RESOURCE_SPECIALIZED_BASENAME,
        TERMINAL_RESOURCE_CONTRACT_SHA256,
        resource_module,
    )
    shard_validator = (
        generator._validate_shard_scratch_binding_postretention_attested
        if postretention_attested
        else generator.validate_shard_scratch_binding
    )
    replayed_compute, replayed_rank, replayed_shard = (
        shard_validator(
            root=Path(shard_ref["path"]).parent,
            plan={
                **exact2_plan,
                "_path": str(audit_plan_path),
                "_file_sha256": audit_plan_sha,
            },
            resource=resource,
            compute_preflight=compute_ref["path"],
            expected_compute_preflight_sha256=compute_ref["file_sha256"],
        )
    )
    require(
        replayed_compute == compute_ref
        and replayed_rank == rank_binding
        and replayed_shard == shard_ref,
        "generation shard/rank resource scratch replay differs",
    )
    tasks = exact2_plan["admission_tasks"]
    rows = value.get("candidate_receipts")
    require(
        value.get("plan_digest") == exact2_plan["plan_digest"]
        and value.get("dataset") == plan_contract.DATASET
        and value.get("candidate_count") == 2
        and value.get("comparator_cell_count") == 2
        and value.get("new_branch_order") == ["incomplete", "incomplete"]
        and value.get("all_candidates_exact81") is True
        and value.get("independent_full81_review_performed") is False
        and value.get("review_admission_authorized") is False
        and value.get("q_input_authorized") is False
        and value.get("a_min_input_authorized") is False
        and value.get("training_performed") is False
        and value.get("optimizer_created") is False
        and value.get("optimizer_authorized") is False
        and value.get("diagnostic_task_count") == 0
        and value.get("diagnostic_generation_observed_or_required") is False
        and value.get("action_generation_observed_or_required") is False
        and isinstance(rows, list)
        and len(rows) == 2,
        "exact2 generation audit authority differs",
    )
    receipt_fields = {
        "candidate_id",
        "calibration_group_id",
        "semantic_branch",
        "candidate_receipt_path",
        "candidate_receipt_file_sha256",
        "candidate_receipt_digest",
        "generated_mp4_path",
        "generated_mp4_file_sha256",
        "generated_mp4_frame_count",
        "generated_mp4_fps",
    }
    new_receipts: dict[str, Mapping[str, Any]] = {}
    for row, task in zip(rows, tasks):
        require(
            isinstance(row, Mapping)
            and set(row) == receipt_fields
            and row.get("candidate_id") == task["candidate_id"]
            and row.get("calibration_group_id") == task["calibration_group_id"]
            and row.get("semantic_branch") == "incomplete"
            and row.get("generated_mp4_frame_count") == 81
            and row.get("generated_mp4_fps") == 25
            and SHA256_RE.fullmatch(
                str(row.get("candidate_receipt_file_sha256"))
            )
            is not None
            and SHA256_RE.fullmatch(
                str(row.get("candidate_receipt_digest"))
            )
            is not None
            and SHA256_RE.fullmatch(
                str(row.get("generated_mp4_file_sha256"))
            )
            is not None,
            f"new candidate receipt reference differs: {task['candidate_id']}",
        )
        receipt_path = plain_file(
            row["candidate_receipt_path"],
            f"new incomplete receipt {task['candidate_id']}",
        )
        require(
            str(receipt_path) == row["candidate_receipt_path"]
            and receipt_path.name == "pair-v5-t2v-calibration-receipt.json"
            and receipt_path.parent.name == task["candidate_id"]
            and file_sha256(receipt_path)
            == row["candidate_receipt_file_sha256"],
            f"new candidate receipt path/SHA differs: {task['candidate_id']}",
        )
        try:
            receipt = generator._validate_candidate_receipt(
                resource, task, receipt_path
            )
        except generator.ArmsIncompleteExact2GenerationError as error:
            raise ArmsIncompleteExact2ControllerError(str(error)) from error
        mp4 = receipt.get("artifacts", {}).get("mp4", {})
        mp4_path = plain_file(
            row["generated_mp4_path"],
            f"generated incomplete MP4 {task['candidate_id']}",
        )
        require(
            receipt.get("receipt_digest") == row["candidate_receipt_digest"]
            and file_sha256(receipt_path)
            == row["candidate_receipt_file_sha256"]
            and str(mp4_path) == row["generated_mp4_path"] == mp4.get("path")
            and mp4_path.parent == receipt_path.parent
            and mp4_path.name == "t2v.mp4"
            and file_sha256(mp4_path)
            == row["generated_mp4_file_sha256"]
            == mp4.get("sha256")
            and mp4.get("frame_count")
            == row["generated_mp4_frame_count"]
            == 81
            and mp4.get("fps") == row["generated_mp4_fps"] == 25,
            f"new candidate receipt digest differs: {task['candidate_id']}",
        )
        probe = probe_exact81_25fps(mp4_path, ffprobe_bin)
        require(
            probe.get("frame_count") == 81 and probe.get("fps") == 25,
            f"generated incomplete MP4 replay differs: {task['candidate_id']}",
        )
        new_receipts[task["candidate_id"]] = receipt
    action_by_seed = {
        action["seed"]: action for action in exact2_plan["external_action_cells"]
    }
    recomputed: list[Mapping[str, Any]] = []
    for task in tasks:
        action = action_by_seed[task["seed"]]
        try:
            action_receipt, _ = generator._validate_external_action_artifacts(
                action, resource
            )
            recomputed.append(
                generator.cross_run_same_gaussian_proof(
                    task=task,
                    action=action,
                    incomplete_receipt=new_receipts[task["candidate_id"]],
                    action_receipt=action_receipt,
                    resource=resource,
                )
            )
        except generator.ArmsIncompleteExact2GenerationError as error:
            raise ArmsIncompleteExact2ControllerError(str(error)) from error
    require(
        len(recomputed) == 2
        and value.get("cross_run_same_gaussian_pair_proofs") == recomputed,
        "exact2 cross-run same-Gaussian proofs do not replay",
    )
    return value


def seal_blind_review_input(args: argparse.Namespace) -> Mapping[str, Any]:
    """Create only a blinded post-generation input; never create a verdict."""

    _, _, _, exact2 = load_controller_plan(
        args.controller_plan, args.expected_controller_plan_sha256
    )
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "exact2 generation audit",
        args.expected_generation_audit_sha256,
    )
    ffprobe_bin = validate_ffprobe(args.ffprobe_bin, args.expected_ffprobe_sha256)
    validate_exact2_audit(generation, exact2, ffprobe_bin)
    output = _fresh_directory(Path(args.output_dir), "blind review packet root")
    reviewer_root = output / "reviewer"
    reviewer_root.mkdir(mode=0o700)
    packet_id = secrets.token_hex(16)
    generation_rows = list(generation["candidate_receipts"])
    secrets.SystemRandom().shuffle(generation_rows)
    samples: list[Mapping[str, Any]] = []
    mappings: list[Mapping[str, Any]] = []
    sample_ids: set[str] = set()
    for row in generation_rows:
        sample_id = secrets.token_hex(12)
        require(sample_id not in sample_ids, "blind sample id collision")
        sample_ids.add(sample_id)
        opaque_filename = f"sample_{sample_id}.mp4"
        reviewer_mp4 = reviewer_root / opaque_filename
        copied_sha = _copy_create_only(
            Path(row["generated_mp4_path"]), reviewer_mp4
        )
        probe = probe_exact81_25fps(reviewer_mp4, ffprobe_bin)
        require(
            copied_sha == row["generated_mp4_file_sha256"]
            and probe.get("frame_count") == 81
            and probe.get("fps") == 25,
            "blind review copy physical replay differs",
        )
        samples.append(
            {
                "sample_id": sample_id,
                "opaque_filename": opaque_filename,
                "reviewer_mp4_path": str(reviewer_mp4),
                "reviewer_mp4_file_sha256": copied_sha,
                "frame_count": 81,
                "fps": 25,
            }
        )
        mappings.append(
            {
                "sample_id": sample_id,
                "opaque_filename": opaque_filename,
                "candidate_id": row["candidate_id"],
                "candidate_receipt_path": row["candidate_receipt_path"],
                "candidate_receipt_file_sha256": row[
                    "candidate_receipt_file_sha256"
                ],
                "candidate_receipt_digest": row["candidate_receipt_digest"],
                "generated_mp4_path": row["generated_mp4_path"],
                "generated_mp4_file_sha256": row[
                    "generated_mp4_file_sha256"
                ],
                "reviewer_mp4_path": str(reviewer_mp4),
                "reviewer_mp4_file_sha256": copied_sha,
            }
        )
    manifest_unsigned = {
        "schema_version": BLIND_REVIEW_MANIFEST_SCHEMA,
        "packet_id": packet_id,
        "plan_digest": exact2["plan_digest"],
        "generation_audit_digest": generation["receipt_digest"],
        "required_ffprobe": {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
        },
        "candidate_count": 2,
        "blinded": True,
        "review_decision_present": False,
        "sample_order": [row["sample_id"] for row in samples],
        "samples": samples,
    }
    manifest = _signed(manifest_unsigned)
    manifest_path = reviewer_root / "review-manifest.json"
    manifest_sha = write_create_only(manifest_path, manifest)
    key_unsigned = {
        "schema_version": BLIND_REVIEW_KEY_SCHEMA,
        "packet_id": packet_id,
        "plan_digest": exact2["plan_digest"],
        "generation_audit": {
            "path": str(generation_path),
            "file_sha256": generation_sha,
            "receipt_digest": generation["receipt_digest"],
        },
        "review_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_sha,
            "receipt_digest": manifest["receipt_digest"],
        },
        "required_ffprobe": {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
        },
        "candidate_count": 2,
        "mappings": mappings,
        "review_decision_present": False,
    }
    key = _signed(key_unsigned)
    key_path = output / "sealed-key.json"
    key_sha = write_create_only(key_path, key)
    return {
        "schema_version": (
            "bernini-full30-action-arms-incomplete-repair-exact2-"
            "blind-review-input-created-v1"
        ),
        "packet_id": packet_id,
        "review_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_sha,
            "receipt_digest": manifest["receipt_digest"],
        },
        "sealed_key": {
            "path": str(key_path),
            "file_sha256": key_sha,
            "receipt_digest": key["receipt_digest"],
        },
        "candidate_count": 2,
        "review_decision_present": False,
    }


def _derive_terminal_host_gate_from_physical(
    *,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
    resource_contract: str | Path,
    expected_resource_contract_sha256: Optional[str],
    monitor_start_receipt: str | Path,
    expected_monitor_start_receipt_sha256: str,
    monitor_exit_status: int,
    resource_module: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    """Child derivation with physical scratch and PID replay."""

    return _derive_terminal_host_gate_core(
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
        resource_contract=resource_contract,
        expected_resource_contract_sha256=expected_resource_contract_sha256,
        monitor_start_receipt=monitor_start_receipt,
        expected_monitor_start_receipt_sha256=expected_monitor_start_receipt_sha256,
        monitor_exit_status=monitor_exit_status,
        resource_module=resource_module,
        postretention_attested=False,
    )


def _derive_terminal_host_gate_postretention_attested(
    *,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
    resource_contract: str | Path,
    expected_resource_contract_sha256: Optional[str],
    monitor_start_receipt: str | Path,
    expected_monitor_start_receipt_sha256: str,
    monitor_exit_status: int,
    resource_module: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    return _derive_terminal_host_gate_core(
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
        resource_contract=resource_contract,
        expected_resource_contract_sha256=expected_resource_contract_sha256,
        monitor_start_receipt=monitor_start_receipt,
        expected_monitor_start_receipt_sha256=expected_monitor_start_receipt_sha256,
        monitor_exit_status=monitor_exit_status,
        resource_module=resource_module,
        postretention_attested=True,
    )


def _derive_terminal_host_gate_core(
    *,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
    resource_contract: str | Path,
    expected_resource_contract_sha256: Optional[str],
    monitor_start_receipt: str | Path,
    expected_monitor_start_receipt_sha256: str,
    monitor_exit_status: int,
    resource_module: Optional[ModuleType],
    postretention_attested: bool,
) -> Mapping[str, Any]:
    """Derive the exact2 terminal gate only from physically reopened bytes."""

    require(
        type(monitor_exit_status) is int and monitor_exit_status == 0,
        "bound monitor wait did not return exact status zero",
    )
    preflight, preflight_path, preflight_sha = load_json(
        compute_preflight,
        "terminal compute preflight",
        expected_compute_preflight_sha256,
    )
    if postretention_attested:
        _validate_compute_preflight_postretention(preflight)
    else:
        validate_compute_preflight(preflight)
    resource_path = plain_file(resource_contract, "terminal resource contract")
    resource_sha = file_sha256(resource_path)
    resource = _load_or_reuse_resource_contract(
        resource_path,
        str(expected_resource_contract_sha256),
        resource_module,
    )
    try:
        start, start_path, start_sha = resource.load_host_cgroup_memory_monitor_start(
            monitor_start_receipt,
            expected_monitor_start_receipt_sha256,
        )
        raw, packed_rows, metadata, _ = resource._journal_prefix(
            start, exact_terminal_size=True
        )
        rows = [resource._sample_row(row) for row in packed_rows]
        monitor_dead = (
            True
            if postretention_attested
            else not resource._process_identity_is_live(
                int(start["monitor_pid"]), int(start["monitor_proc_start_ticks"])
            )
        )
    except Exception as error:
        raise ArmsIncompleteExact2ControllerError(str(error)) from error
    require(
        len(rows) >= 2
        and rows[0] == start["initial_sample"]
        and [row["sequence"] for row in rows] == list(range(len(rows)))
        and all(row["sample_kind"] == "periodic" for row in rows[:-1])
        and rows[-1]["sample_kind"] == "stop_final"
        and monitor_dead,
        "terminal sample sequence/kind or monitor-dead closure differs",
    )
    preflight_runtime = preflight["runtime"]
    require(
        start["slurm_job_id"] == preflight_runtime["slurm_job_id"]
        and start["slurm_step_id"] == preflight_runtime["slurm_step_id"],
        "terminal monitor/preflight Slurm job-step binding differs",
    )
    wall = [row["wall_time_ns"] for row in rows]
    monotonic = [row["monotonic_time_ns"] for row in rows]
    currents = [row["memory_current_bytes"] for row in rows]
    require(
        all(type(value) is int and value > 0 for value in wall)
        and all(type(value) is int and value > 0 for value in monotonic)
        and all(type(value) is int and 0 <= value < 56 * 1024**3 for value in currents)
        and all(row["memory_max_bytes"] == 60 * 1024**3 for row in rows)
        and all(row["memory_events"] == {"oom": 0, "oom_kill": 0} for row in rows),
        "terminal sampled memory/max/OOM closure differs",
    )
    gaps = [right - left for left, right in zip(monotonic, monotonic[1:])]
    require(
        all(0 < gap <= 100_000_000 for gap in gaps),
        "terminal sample gap is not in (0, 100ms]",
    )
    journal = start["sample_journal"]
    journal_mode = stat.S_IMODE(metadata.st_mode)
    require(
        metadata.st_dev == journal["device"]
        and metadata.st_ino == journal["inode"]
        and metadata.st_nlink == 1
        and journal_mode == 0o400
        and len(raw) == len(rows) * journal["record_size"],
        "terminal journal physical identity/size differs",
    )
    unsigned = {
        "schema_version": TERMINAL_HOST_GATE_SCHEMA,
        "measurement_phase": "terminal_after_arms_incomplete_repair_exact2",
        "formal_candidate_count_at_gate": 2,
        "compute_preflight": {
            "path": str(preflight_path),
            "file_sha256": preflight_sha,
            "receipt_digest": preflight["receipt_digest"],
        },
        "resource_contract": {
            "path": str(resource_path),
            "file_sha256": resource_sha,
        },
        "monitor_start_receipt": {
            "path": str(start_path),
            "file_sha256": start_sha,
            "receipt_digest": start["receipt_digest"],
        },
        "slurm": {
            "job_id": preflight_runtime["slurm_job_id"],
            "step_id": preflight_runtime["slurm_step_id"],
        },
        "sample_journal": {
            "path": journal["path"],
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "nlink": metadata.st_nlink,
            "mode_octal": "0400",
            "record_size": journal["record_size"],
            "record_encoding": journal["record_encoding"],
            "byte_count": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "sample_count": len(rows),
        },
        "sampling": {
            "requested_interval_ns": 10_000_000,
            "maximum_allowed_gap_ns": 100_000_000,
            "sample_count": len(rows),
            "first_sequence": 0,
            "last_sequence": len(rows) - 1,
            "start_wall_time_ns": wall[0],
            "end_wall_time_ns": wall[-1],
            "start_monotonic_time_ns": monotonic[0],
            "end_monotonic_time_ns": monotonic[-1],
            "duration_ns": monotonic[-1] - monotonic[0],
            "maximum_observed_gap_ns": max(gaps),
            "all_gaps_positive_and_within_100_ms": True,
            "periodic_then_single_stop_final": True,
        },
        "memory": {
            "strict_limit_gib": 60,
            "strict_limit_bytes": 60 * 1024**3,
            "safe_ceiling_gib": 56,
            "safe_ceiling_bytes": 56 * 1024**3,
            "sampled_peak_memory_current_bytes": max(currents),
            "sampled_peak_strictly_below_safe_ceiling": True,
            "all_samples_memory_max_exactly_60_gib": True,
            "all_samples_zero_oom_and_oom_kill": True,
        },
        "monitor": {
            "bound_supervisor_wait_exit_status": 0,
            "bound_supervisor_wait_completed": True,
            "monitor_identity_dead_at_gate": True,
        },
        "diagnostic_task_count": 0,
        "action_generation_authorized": False,
        "optimizer_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _validate_terminal_host_gate_shape(value: Mapping[str, Any]) -> None:
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    expected_fields = {
        "schema_version",
        "measurement_phase",
        "formal_candidate_count_at_gate",
        "compute_preflight",
        "resource_contract",
        "monitor_start_receipt",
        "slurm",
        "sample_journal",
        "sampling",
        "memory",
        "monitor",
        "diagnostic_task_count",
        "action_generation_authorized",
        "optimizer_authorized",
        "receipt_digest",
    }
    journal_fields = {
        "path",
        "device",
        "inode",
        "nlink",
        "mode_octal",
        "record_size",
        "record_encoding",
        "byte_count",
        "file_sha256",
        "sample_count",
    }
    sampling_fields = {
        "requested_interval_ns",
        "maximum_allowed_gap_ns",
        "sample_count",
        "first_sequence",
        "last_sequence",
        "start_wall_time_ns",
        "end_wall_time_ns",
        "start_monotonic_time_ns",
        "end_monotonic_time_ns",
        "duration_ns",
        "maximum_observed_gap_ns",
        "all_gaps_positive_and_within_100_ms",
        "periodic_then_single_stop_final",
    }
    memory_fields = {
        "strict_limit_gib",
        "strict_limit_bytes",
        "safe_ceiling_gib",
        "safe_ceiling_bytes",
        "sampled_peak_memory_current_bytes",
        "sampled_peak_strictly_below_safe_ceiling",
        "all_samples_memory_max_exactly_60_gib",
        "all_samples_zero_oom_and_oom_kill",
    }
    journal = value.get("sample_journal")
    sampling = value.get("sampling")
    memory = value.get("memory")
    monitor = value.get("monitor")
    require(
        type(value) is dict
        and set(value) == expected_fields
        and value.get("schema_version") == TERMINAL_HOST_GATE_SCHEMA
        and type(value.get("compute_preflight")) is dict
        and set(value["compute_preflight"]) == reference_fields
        and type(value.get("resource_contract")) is dict
        and set(value["resource_contract"]) == {"path", "file_sha256"}
        and type(value.get("monitor_start_receipt")) is dict
        and set(value["monitor_start_receipt"]) == reference_fields
        and type(value.get("slurm")) is dict
        and set(value["slurm"]) == {"job_id", "step_id"}
        and type(journal) is dict
        and set(journal) == journal_fields
        and type(sampling) is dict
        and set(sampling) == sampling_fields
        and type(memory) is dict
        and set(memory) == memory_fields
        and type(monitor) is dict
        and set(monitor)
        == {
            "bound_supervisor_wait_exit_status",
            "bound_supervisor_wait_completed",
            "monitor_identity_dead_at_gate",
        }
        and all(
            type(journal.get(field)) is int and journal[field] > 0
            for field in ("inode", "nlink", "record_size", "byte_count", "sample_count")
        )
        and type(journal.get("device")) is int
        and journal["device"] >= 0
        and journal.get("mode_octal") == "0400"
        and SHA256_RE.fullmatch(str(journal.get("file_sha256"))) is not None
        and all(
            type(sampling.get(field)) is int and sampling[field] >= 0
            for field in (
                "sample_count",
                "first_sequence",
                "last_sequence",
                "start_wall_time_ns",
                "end_wall_time_ns",
                "start_monotonic_time_ns",
                "end_monotonic_time_ns",
                "duration_ns",
                "maximum_observed_gap_ns",
            )
        )
        and type(memory.get("sampled_peak_memory_current_bytes")) is int
        and memory["sampled_peak_memory_current_bytes"] >= 0,
        "terminal host gate top/nested field closure differs",
    )
    _validate_signed(value, TERMINAL_HOST_GATE_SCHEMA, "terminal host gate")


def seal_terminal_host_gate(args: argparse.Namespace) -> Mapping[str, Any]:
    value = _derive_terminal_host_gate_from_physical(
        compute_preflight=args.compute_preflight,
        expected_compute_preflight_sha256=args.expected_compute_preflight_sha256,
        resource_contract=args.resource_contract,
        expected_resource_contract_sha256=args.expected_resource_contract_sha256,
        monitor_start_receipt=args.monitor_start_receipt,
        expected_monitor_start_receipt_sha256=(
            args.expected_monitor_start_receipt_sha256
        ),
        monitor_exit_status=args.monitor_exit_status,
    )
    write_create_only(Path(args.output), value)
    return value


def validate_terminal_host_gate(
    value: Mapping[str, Any], resource_module: Optional[ModuleType] = None
) -> Mapping[str, Any]:
    """Child validation with physical scratch and PID replay."""

    return _validate_terminal_host_gate_core(
        value, resource_module, postretention_attested=False
    )


def _validate_terminal_host_gate_postretention_attested(
    value: Mapping[str, Any], resource_module: Optional[ModuleType] = None
) -> Mapping[str, Any]:
    return _validate_terminal_host_gate_core(
        value, resource_module, postretention_attested=True
    )


def _validate_terminal_host_gate_core(
    value: Mapping[str, Any],
    resource_module: Optional[ModuleType],
    *,
    postretention_attested: bool,
) -> Mapping[str, Any]:
    _validate_terminal_host_gate_shape(value)
    derive = (
        _derive_terminal_host_gate_postretention_attested
        if postretention_attested
        else _derive_terminal_host_gate_from_physical
    )
    derived = derive(
        compute_preflight=value["compute_preflight"]["path"],
        expected_compute_preflight_sha256=value["compute_preflight"][
            "file_sha256"
        ],
        resource_contract=value["resource_contract"]["path"],
        expected_resource_contract_sha256=value["resource_contract"][
            "file_sha256"
        ],
        monitor_start_receipt=value["monitor_start_receipt"]["path"],
        expected_monitor_start_receipt_sha256=value["monitor_start_receipt"][
            "file_sha256"
        ],
        monitor_exit_status=value["monitor"][
            "bound_supervisor_wait_exit_status"
        ],
        resource_module=resource_module,
    )
    require(
        canonical_json_bytes(value) == canonical_json_bytes(derived),
        "terminal host gate differs from physical derivation",
    )
    return value


def _receipt_reference(
    value: Mapping[str, Any], path: Path, file_sha256_value: str
) -> Mapping[str, str]:
    return {
        "path": str(path),
        "file_sha256": file_sha256_value,
        "receipt_digest": str(value["receipt_digest"]),
    }


def _task_scratch_physical_value(
    task_scratch: str | Path, prepare: Mapping[str, Any]
) -> Mapping[str, Any]:
    task = plain_dir(task_scratch, "child inner task scratch")
    outer = Path(prepare["scratch_root"]["path"])
    metadata = task.lstat()
    require(
        task.parent == outer
        and re.fullmatch(
            rf"arms-incomplete-exact2-{HOLDER_JOB}-[0-9]+\.[A-Za-z0-9]{{8}}",
            task.name,
        )
        is not None
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == prepare["scratch_root"]["device"]
        and metadata.st_uid == CHILD_SCRATCH_OWNER_UID
        and metadata.st_gid == CHILD_SCRATCH_OWNER_GID
        and stat.S_IMODE(metadata.st_mode) == CHILD_SCRATCH_MODE
        and metadata.st_nlink >= 2,
        "child inner task scratch physical identity differs",
    )
    outer_entries = sorted(outer.iterdir(), key=lambda path: path.name)
    require(
        outer_entries
        == sorted(
            [outer / CHILD_SCRATCH_PROBE_BASENAME, task],
            key=lambda path: path.name,
        )
        and all(not path.is_symlink() for path in outer_entries),
        "prepared outer scratch does not contain exact retained probe plus inner root",
    )
    return {"path": str(task), "basename": task.name, **_metadata_value(metadata)}


def _renderer_load_lock_physical_value(
    inner: Path, *, expected_device: int
) -> Mapping[str, Any]:
    lock = inner / CHILD_RENDERER_LOAD_LOCK_BASENAME
    try:
        metadata = lock.lstat()
        resolved = lock.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "controller-created renderer load lock is unavailable"
        ) from error
    require(
        resolved == lock
        and lock.parent == inner
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == expected_device
        and metadata.st_uid == CHILD_SCRATCH_OWNER_UID
        and metadata.st_gid == CHILD_SCRATCH_OWNER_GID
        and stat.S_IMODE(metadata.st_mode) == 0o400
        and metadata.st_nlink == 1
        and metadata.st_size == 0
        and file_sha256(lock) == hashlib.sha256(b"").hexdigest(),
        "controller-created renderer load lock physical identity differs",
    )
    return {
        "path": str(lock),
        "basename": lock.name,
        **_metadata_value(metadata),
        "size_bytes": metadata.st_size,
        "empty_file_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _validate_renderer_load_lock_shape(
    value: Any, inner: Any
) -> Mapping[str, Any]:
    fields = {
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
        "size_bytes",
        "empty_file_sha256",
    }
    require(
        type(value) is dict
        and type(inner) is dict
        and set(value) == fields
        and value.get("path")
        == str(Path(str(inner.get("path"))) / CHILD_RENDERER_LOAD_LOCK_BASENAME)
        and value.get("basename") == CHILD_RENDERER_LOAD_LOCK_BASENAME
        and value.get("device") == inner.get("device")
        and value.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(value.get("inode")) is int
        and value["inode"] > 0
        and value.get("uid") == CHILD_SCRATCH_OWNER_UID
        and value.get("gid") == CHILD_SCRATCH_OWNER_GID
        and value.get("mode_octal") == "0400"
        and value.get("link_count") == 1
        and value.get("size_bytes") == 0
        and value.get("empty_file_sha256") == hashlib.sha256(b"").hexdigest(),
        "renderer load lock field/authority closure differs",
    )
    return value


def _validate_child_task_scratch_bind_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, CHILD_TASK_SCRATCH_BIND_SCHEMA, "child task scratch bind")
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    inner_fields = {
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    lock_fields = inner_fields | {"size_bytes", "empty_file_sha256"}
    probe_fields = inner_fields | {"size_bytes", "file_sha256"}
    outer_fields = inner_fields | {"canonical_non_symlink"}
    runtime_fields = {
        "slurm_job_id",
        "slurm_step_id",
        "hostname",
        "sole_numbered_compute_child_required",
    }
    runtime = value.get("runtime")
    outer = value.get("scratch_outer")
    inner = value.get("scratch_inner")
    load_lock = value.get("renderer_load_lock")
    retained_probe = value.get("retained_probe_file")
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "authority",
            "runtime",
            "scratch_prepare",
            "compute_preflight",
            "scratch_outer",
            "scratch_inner",
            "renderer_load_lock",
            "retained_probe_file",
            "creation",
            "formal_candidate_count_at_gate",
            "diagnostic_task_count",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(runtime) is dict
        and set(runtime) == runtime_fields
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and runtime.get("hostname") == HOLDER_NODE
        and re.fullmatch(r"[0-9]+", str(runtime.get("slurm_step_id"))) is not None
        and runtime.get("sole_numbered_compute_child_required") is True
        and type(value.get("scratch_prepare")) is dict
        and set(value["scratch_prepare"]) == reference_fields
        and type(value.get("compute_preflight")) is dict
        and set(value["compute_preflight"]) == reference_fields
        and type(outer) is dict
        and set(outer) == outer_fields
        and outer.get("path")
        == str(
            _expected_child_scratch_path(
                HOLDER_JOB, str(runtime.get("slurm_step_id"))
            )
        )
        and outer.get("basename") == Path(str(outer.get("path"))).name
        and outer.get("canonical_non_symlink") is True
        and outer.get("device") == os.makedev(253, 0)
        and outer.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(outer.get("inode")) is int
        and outer["inode"] > 0
        and outer.get("uid") == CHILD_SCRATCH_OWNER_UID
        and outer.get("gid") == CHILD_SCRATCH_OWNER_GID
        and outer.get("mode_octal") == "0700"
        and outer.get("link_count") == 2
        and type(inner) is dict
        and set(inner) == inner_fields
        and Path(str(inner.get("path"))).parent
        == Path(value["scratch_outer"]["path"])
        and inner.get("basename") == Path(str(inner.get("path"))).name
        and re.fullmatch(
            rf"arms-incomplete-exact2-{HOLDER_JOB}-{runtime.get('slurm_step_id')}\.[0-9a-f]{{8}}",
            str(inner.get("basename")),
        )
        is not None
        and inner.get("device") == outer.get("device")
        and inner.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and inner.get("uid") == CHILD_SCRATCH_OWNER_UID
        and inner.get("gid") == CHILD_SCRATCH_OWNER_GID
        and inner.get("mode_octal") == "0700"
        and inner.get("link_count") == 2
        and type(inner.get("inode")) is int
        and inner["inode"] > 0
        and type(load_lock) is dict
        and set(load_lock) == lock_fields
        and load_lock.get("path")
        == str(Path(str(inner.get("path"))) / CHILD_RENDERER_LOAD_LOCK_BASENAME)
        and load_lock.get("basename") == CHILD_RENDERER_LOAD_LOCK_BASENAME
        and load_lock.get("device") == inner.get("device")
        and load_lock.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(load_lock.get("inode")) is int
        and load_lock["inode"] > 0
        and load_lock.get("uid") == CHILD_SCRATCH_OWNER_UID
        and load_lock.get("gid") == CHILD_SCRATCH_OWNER_GID
        and load_lock.get("mode_octal") == "0400"
        and load_lock.get("link_count") == 1
        and load_lock.get("size_bytes") == 0
        and load_lock.get("empty_file_sha256") == hashlib.sha256(b"").hexdigest()
        and type(retained_probe) is dict
        and set(retained_probe) == probe_fields
        and retained_probe.get("path")
        == str(Path(str(outer.get("path"))) / CHILD_SCRATCH_PROBE_BASENAME)
        and retained_probe.get("basename") == CHILD_SCRATCH_PROBE_BASENAME
        and retained_probe.get("device") == outer.get("device")
        and retained_probe.get("device_major_minor")
        == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(retained_probe.get("inode")) is int
        and retained_probe["inode"] > 0
        and retained_probe.get("uid") == CHILD_SCRATCH_OWNER_UID
        and retained_probe.get("gid") == CHILD_SCRATCH_OWNER_GID
        and retained_probe.get("mode_octal") == "0600"
        and retained_probe.get("link_count") == 1
        and retained_probe.get("size_bytes") == len(CHILD_SCRATCH_PROBE_BYTES)
        and retained_probe.get("file_sha256")
        == hashlib.sha256(CHILD_SCRATCH_PROBE_BYTES).hexdigest()
        and type(value.get("creation")) is dict
        and set(value["creation"])
        == {
            "nonce_hex",
            "controller_generated_nonce",
            "caller_path_or_nonce_allowed",
            "outer_inventory_exact_retained_probe_before_mkdirat",
            "mkdirat_create_only",
            "outer_directory_fsync_after_mkdir",
            "outer_inventory_exact_retained_probe_and_inner_after_mkdir",
            "renderer_lock_openat_o_excl_no_follow",
            "renderer_lock_fchmod_0400",
            "renderer_lock_file_fsync",
            "inner_directory_fsync_after_renderer_lock",
        }
        and value["creation"].get("nonce_hex")
        == str(inner.get("basename", "")).rsplit(".", 1)[-1]
        and re.fullmatch(r"[0-9a-f]{8}", str(value["creation"].get("nonce_hex")))
        is not None
        and {key: item for key, item in value["creation"].items() if key != "nonce_hex"}
        == {
            "controller_generated_nonce": True,
            "caller_path_or_nonce_allowed": False,
            "outer_inventory_exact_retained_probe_before_mkdirat": True,
            "mkdirat_create_only": True,
            "outer_directory_fsync_after_mkdir": True,
            "outer_inventory_exact_retained_probe_and_inner_after_mkdir": True,
            "renderer_lock_openat_o_excl_no_follow": True,
            "renderer_lock_fchmod_0400": True,
            "renderer_lock_file_fsync": True,
            "inner_directory_fsync_after_renderer_lock": True,
        }
        and value.get("formal_candidate_count_at_gate") == 0
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "child task scratch bind field/authority closure differs",
    )
    return value


def _replay_child_task_scratch_bind_physical(
    value: Mapping[str, Any], prepare: Mapping[str, Any]
) -> None:
    _validate_child_task_scratch_bind_shape(value)
    observed = _task_scratch_physical_value(
        value["scratch_inner"]["path"], prepare
    )
    observed_lock = _renderer_load_lock_physical_value(
        Path(observed["path"]), expected_device=observed["device"]
    )
    expected = value["scratch_inner"]
    require(
        value["scratch_outer"] == prepare["scratch_root"]
        and value["retained_probe_file"] == prepare["retained_probe_file"]
        and observed["path"] == expected["path"]
        and observed["basename"] == expected["basename"]
        and observed["device"] == expected["device"]
        and observed["device_major_minor"] == expected["device_major_minor"]
        and observed["inode"] == expected["inode"]
        and observed["uid"] == expected["uid"]
        and observed["gid"] == expected["gid"]
        and observed["mode_octal"] == expected["mode_octal"]
        and observed["link_count"] >= expected["link_count"] == 2,
        "child task scratch was renamed/recreated or physically drifted",
    )
    require(
        observed_lock == value["renderer_load_lock"],
        "controller-created renderer load lock was renamed/recreated or drifted",
    )


def create_and_bind_child_task_scratch(args: argparse.Namespace) -> Mapping[str, Any]:
    """Create the inner runtime root and immediately seal its original inode."""

    require(
        "SLURM_TMPDIR" not in os.environ
        and "TMPDIR" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH" not in os.environ
        and "GADP_NODE_LOCAL_SCRATCH_FSTYPE" not in os.environ,
        "task scratch must be bound before any scratch environment export",
    )
    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "task-bind scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    _replay_child_scratch_prepare_physical(
        prepare, require_initial_link_count=True
    )
    preflight, preflight_path, preflight_sha = load_json(
        args.compute_preflight,
        "task-bind compute preflight",
        args.expected_compute_preflight_sha256,
    )
    validate_compute_preflight(preflight)
    require(
        preflight["scratch_prepare"]
        == _receipt_reference(prepare, prepare_path, prepare_sha),
        "task scratch bind prepare/compute reference differs",
    )
    outer = Path(prepare["scratch_root"]["path"])
    require(
        sorted(path.name for path in outer.iterdir())
        == [CHILD_SCRATCH_PROBE_BASENAME],
        "outer scratch does not contain exactly the retained probe before inner create",
    )
    nonce = secrets.token_hex(4)
    inner_name = (
        f"arms-incomplete-exact2-{HOLDER_JOB}-"
        f"{prepare['runtime']['slurm_step_id']}.{nonce}"
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    outer_descriptor = os.open(outer, flags)
    inner_descriptor = -1
    lock_descriptor = -1
    lock_metadata: Optional[os.stat_result] = None
    try:
        outer_metadata = os.fstat(outer_descriptor)
        require(
            (outer_metadata.st_dev, outer_metadata.st_ino)
            == (
                prepare["scratch_root"]["device"],
                prepare["scratch_root"]["inode"],
            ),
            "outer scratch changed before inner mkdirat",
        )
        os.mkdir(inner_name, mode=CHILD_SCRATCH_MODE, dir_fd=outer_descriptor)
        os.fsync(outer_descriptor)
        inner_descriptor = os.open(inner_name, flags, dir_fd=outer_descriptor)
        inner_metadata = os.fstat(inner_descriptor)
        require(
            stat.S_ISDIR(inner_metadata.st_mode)
            and inner_metadata.st_dev == outer_metadata.st_dev
            and inner_metadata.st_uid == CHILD_SCRATCH_OWNER_UID
            and inner_metadata.st_gid == CHILD_SCRATCH_OWNER_GID
            and stat.S_IMODE(inner_metadata.st_mode) == CHILD_SCRATCH_MODE
            and inner_metadata.st_nlink == 2,
            "new inner task scratch physical identity differs",
        )
        lock_descriptor = os.open(
            CHILD_RENDERER_LOAD_LOCK_BASENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=inner_descriptor,
        )
        os.fchmod(lock_descriptor, 0o400)
        os.fsync(lock_descriptor)
        lock_metadata = os.fstat(lock_descriptor)
        require(
            stat.S_ISREG(lock_metadata.st_mode)
            and lock_metadata.st_dev == inner_metadata.st_dev
            and lock_metadata.st_uid == CHILD_SCRATCH_OWNER_UID
            and lock_metadata.st_gid == CHILD_SCRATCH_OWNER_GID
            and stat.S_IMODE(lock_metadata.st_mode) == 0o400
            and lock_metadata.st_nlink == 1
            and lock_metadata.st_size == 0,
            "controller-created renderer load lock identity differs",
        )
        os.fsync(inner_descriptor)
    except OSError as error:
        raise ArmsIncompleteExact2ControllerError(
            "inner task scratch create-only binding failed; roots are retained"
        ) from error
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        if inner_descriptor >= 0:
            os.close(inner_descriptor)
        os.close(outer_descriptor)
    inner_path = outer / inner_name
    observed = _task_scratch_physical_value(inner_path, prepare)
    observed_lock = _renderer_load_lock_physical_value(
        inner_path, expected_device=observed["device"]
    )
    require(observed["link_count"] == 2, "new inner task scratch link count differs")
    unsigned = {
        "schema_version": CHILD_TASK_SCRATCH_BIND_SCHEMA,
        "authority": CHILD_SCRATCH_AUTHORITY,
        "runtime": prepare["runtime"],
        "scratch_prepare": _receipt_reference(prepare, prepare_path, prepare_sha),
        "compute_preflight": _receipt_reference(
            preflight, preflight_path, preflight_sha
        ),
        "scratch_outer": prepare["scratch_root"],
        "scratch_inner": observed,
        "renderer_load_lock": observed_lock,
        "retained_probe_file": prepare["retained_probe_file"],
        "creation": {
            "nonce_hex": nonce,
            "controller_generated_nonce": True,
            "caller_path_or_nonce_allowed": False,
            "outer_inventory_exact_retained_probe_before_mkdirat": True,
            "mkdirat_create_only": True,
            "outer_directory_fsync_after_mkdir": True,
            "outer_inventory_exact_retained_probe_and_inner_after_mkdir": True,
            "renderer_lock_openat_o_excl_no_follow": True,
            "renderer_lock_fchmod_0400": True,
            "renderer_lock_file_fsync": True,
            "inner_directory_fsync_after_renderer_lock": True,
        },
        "formal_candidate_count_at_gate": 0,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def validate_child_task_scratch_bind(args: argparse.Namespace) -> Mapping[str, Any]:
    prepare, _, _ = load_json(
        args.scratch_prepare,
        "task-bind validation prepare",
        args.expected_scratch_prepare_sha256,
    )
    bind, bind_path, bind_sha = load_json(
        args.task_scratch_bind,
        "child task scratch bind",
        args.expected_task_scratch_bind_sha256,
    )
    _replay_child_scratch_prepare_physical(
        prepare, require_initial_link_count=False
    )
    _replay_child_task_scratch_bind_physical(bind, prepare)
    require(
        bind["scratch_prepare"]["path"] == args.scratch_prepare
        and bind["scratch_prepare"]["file_sha256"]
        == args.expected_scratch_prepare_sha256,
        "task-bind validation prepare reference differs",
    )
    return {**bind, "_path": str(bind_path), "_file_sha256": bind_sha}


def _proc_identity(pid: int) -> Mapping[str, Any]:
    require(type(pid) is int and pid > 1, "process id differs")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        cgroup_raw = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ArmsIncompleteExact2ControllerError(
            f"process identity unavailable: {pid}"
        ) from error
    right = raw.rfind(")")
    require(right > 0, "process stat format differs")
    fields = raw[right + 2 :].split()
    require(len(fields) >= 20, "process stat field closure differs")
    v2_rows = [line for line in cgroup_raw.splitlines() if line.startswith("0::")]
    require(len(v2_rows) == 1, "process cgroup-v2 binding differs")
    return {
        "pid": pid,
        "state": fields[0],
        "parent_pid": int(fields[1]),
        "start_ticks": int(fields[19]),
        "cgroup_v2_path": v2_rows[0][3:],
    }


def _stable_cgroup_membership(
    cgroup_v2_path: str,
) -> tuple[Path, tuple[int, ...], dict[int, Mapping[str, Any]]]:
    require(
        isinstance(cgroup_v2_path, str)
        and cgroup_v2_path.startswith("/")
        and ".." not in Path(cgroup_v2_path).parts,
        "process cgroup-v2 path differs",
    )
    membership_path = Path("/sys/fs/cgroup") / cgroup_v2_path.lstrip("/") / "cgroup.procs"

    def read_members() -> tuple[int, ...]:
        try:
            raw = membership_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise ArmsIncompleteExact2ControllerError(
                "exact Slurm cgroup membership is unavailable"
            ) from error
        rows = raw.splitlines()
        require(
            all(row.isdecimal() and int(row) > 1 for row in rows)
            and len(rows) == len(set(rows)),
            "exact Slurm cgroup membership format differs",
        )
        return tuple(sorted(int(row) for row in rows))

    before = read_members()
    identities = {pid: _proc_identity(pid) for pid in before}
    middle = read_members()
    replayed = {pid: _proc_identity(pid) for pid in middle}
    after = read_members()
    require(
        before == middle == after
        and identities == replayed
        and all(row["cgroup_v2_path"] == cgroup_v2_path for row in identities.values())
        and all(row["state"] != "Z" for row in identities.values()),
        "exact Slurm cgroup membership/start-ticks did not replay stably",
    )
    return membership_path, before, identities


def _slurm_child_cgroup_census(supervisor_pid: int) -> Mapping[str, Any]:
    current_pid = os.getpid()
    current = _proc_identity(current_pid)
    supervisor = _proc_identity(supervisor_pid)
    require(
        current["parent_pid"] == supervisor_pid
        and current["cgroup_v2_path"] == supervisor["cgroup_v2_path"],
        "attestation process is not the supervisor's exact cgroup child",
    )
    allowed: dict[int, Mapping[str, Any]] = {
        current_pid: current,
        supervisor_pid: supervisor,
    }
    ancestor_pid = int(supervisor["parent_pid"])
    while ancestor_pid > 1:
        ancestor = _proc_identity(ancestor_pid)
        if ancestor["cgroup_v2_path"] != current["cgroup_v2_path"]:
            break
        allowed[ancestor_pid] = ancestor
        ancestor_pid = int(ancestor["parent_pid"])
    membership_path, membership, observed = _stable_cgroup_membership(
        str(current["cgroup_v2_path"])
    )
    require(
        set(observed) == set(allowed),
        "unexpected process remains in the exact Slurm child cgroup",
    )
    try:
        completed = subprocess.run(
            ["squeue", "-s", "-j", HOLDER_JOB, "-h", "-o", "%i"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "cannot census numbered Slurm child"
        ) from error
    numbered = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if re.fullmatch(rf"{HOLDER_JOB}\.[0-9]+", line.strip())
    )
    expected_step = f"{HOLDER_JOB}.{os.environ.get('SLURM_STEP_ID', '')}"
    require(numbered == [expected_step], "attestation is not the sole numbered step")
    return {
        "supervisor": supervisor,
        "attestation_process": current,
        "same_cgroup_process_count": len(observed),
        "same_cgroup_processes": [observed[pid] for pid in sorted(observed)],
        "cgroup_procs_path": str(membership_path),
        "stable_membership_before": list(membership),
        "stable_membership_after": list(membership),
        "identities_and_start_ticks_replayed_stably": True,
        "unexpected_same_cgroup_process_count": 0,
        "numbered_steps": numbered,
        "sole_numbered_step": expected_step,
        "cgroup_census_before_terminal_retention": True,
    }


def _generation_resource_smoke_receipt(
    generation: Mapping[str, Any], resource: ModuleType
) -> Mapping[str, Any]:
    shard_ref = generation["shard_receipt"]
    shard, _, _ = load_json(
        shard_ref["path"], "attested exact2 shard receipt", shard_ref["file_sha256"]
    )
    require(
        shard.get("receipt_digest") == shard_ref["receipt_digest"],
        "attested shard receipt digest differs",
    )
    smoke_ref = shard.get("resource_compile_smoke")
    require(
        type(smoke_ref) is dict
        and set(smoke_ref)
        == {
            "path",
            "file_sha256",
            "receipt_digest",
            "formal_candidate_count_at_gate",
        },
        "attested resource smoke reference differs",
    )
    try:
        smoke, smoke_path, smoke_sha = resource.load_compile_smoke_receipt(
            smoke_ref["path"], smoke_ref["file_sha256"]
        )
    except Exception as error:
        raise ArmsIncompleteExact2ControllerError(str(error)) from error
    require(
        str(smoke_path) == smoke_ref["path"]
        and smoke_sha == smoke_ref["file_sha256"]
        and smoke["receipt_digest"] == smoke_ref["receipt_digest"],
        "attested resource smoke receipt chain differs",
    )
    return smoke


def _scratch_inventory_metadata_tuple(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_blocks,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_regular_file_inventory_row_at(
    descriptor: int, name: str, *, expected_device: int
) -> Mapping[str, Any]:
    """Double-hash one no-follow fd while its immutable metadata stays exact."""

    metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == expected_device
        and metadata.st_uid == CHILD_SCRATCH_OWNER_UID
        and metadata.st_gid == CHILD_SCRATCH_OWNER_GID
        and metadata.st_nlink == 1,
        "scratch inventory found non-regular, cross-device, foreign, or linked file",
    )
    file_descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=descriptor,
    )
    try:
        opened = os.fstat(file_descriptor)

        def digest_pass() -> str:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()

        first_digest = digest_pass()
        middle = os.fstat(file_descriptor)
        second_digest = digest_pass()
        closed = os.fstat(file_descriptor)
        named_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        expected_metadata = _scratch_inventory_metadata_tuple(metadata)
        require(
            _scratch_inventory_metadata_tuple(opened)
            == _scratch_inventory_metadata_tuple(middle)
            == _scratch_inventory_metadata_tuple(closed)
            == _scratch_inventory_metadata_tuple(named_after)
            == expected_metadata
            and first_digest == second_digest,
            "scratch regular file changed during double stable SHA read",
        )
    finally:
        os.close(file_descriptor)
    return {
        "kind": "regular_file",
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "link_count": metadata.st_nlink,
        "size_bytes": metadata.st_size,
        "allocated_512_blocks": metadata.st_blocks,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "file_sha256": first_digest,
    }


def _scan_directory_contents_no_follow(
    descriptor: int, *, expected_device: int
) -> tuple[int, int, Mapping[str, Mapping[str, Any]]]:
    """Two-enumeration recursive inventory with double-read file hashes."""

    file_count = 0
    directory_count = 0
    initial_names = sorted(os.listdir(descriptor))
    inventory: dict[str, Mapping[str, Any]] = {}
    for name in initial_names:
        require(
            name not in {"", ".", ".."} and "/" not in name,
            "scratch descendant basename differs",
        )
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        require(
            metadata.st_dev == expected_device
            and metadata.st_uid == CHILD_SCRATCH_OWNER_UID
            and metadata.st_gid == CHILD_SCRATCH_OWNER_GID
            and not stat.S_ISLNK(metadata.st_mode),
            "scratch inventory found cross-device, foreign, or symlink state",
        )
        common = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "link_count": metadata.st_nlink,
            "size_bytes": metadata.st_size,
            "allocated_512_blocks": metadata.st_blocks,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child_descriptor)
                require(
                    (opened.st_dev, opened.st_ino)
                    == (metadata.st_dev, metadata.st_ino),
                    "scratch directory changed during inventory open",
                )
                nested_files, nested_directories, children = (
                    _scan_directory_contents_no_follow(
                        child_descriptor, expected_device=expected_device
                    )
                )
                closed = os.fstat(child_descriptor)
                require(
                    (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_uid,
                        opened.st_gid,
                        opened.st_mode,
                        opened.st_nlink,
                        opened.st_size,
                        opened.st_blocks,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    )
                    == (
                        closed.st_dev,
                        closed.st_ino,
                        closed.st_uid,
                        closed.st_gid,
                        closed.st_mode,
                        closed.st_nlink,
                        closed.st_size,
                        closed.st_blocks,
                        closed.st_mtime_ns,
                        closed.st_ctime_ns,
                    )
                    == (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_uid,
                        metadata.st_gid,
                        metadata.st_mode,
                        metadata.st_nlink,
                        metadata.st_size,
                        metadata.st_blocks,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ),
                    "scratch directory changed during inventory scan",
                )
            finally:
                os.close(child_descriptor)
            file_count += nested_files
            directory_count += nested_directories + 1
            inventory[name] = {"kind": "directory", **common, "children": children}
        else:
            file_count += 1
            row = _stable_regular_file_inventory_row_at(
                descriptor, name, expected_device=expected_device
            )
            require(
                all(row[field] == common[field] for field in common),
                "scratch regular file lstat changed before stable SHA read",
            )
            inventory[name] = row
    require(
        sorted(os.listdir(descriptor)) == initial_names,
        "scratch inventory changed during stable recursive read",
    )
    return file_count, directory_count, inventory


def _attest_authorized_scratch_inventory(
    task_bind: Mapping[str, Any], smoke: Mapping[str, Any]
) -> Mapping[str, Any]:
    outer = Path(task_bind["scratch_outer"]["path"])
    inner = Path(task_bind["scratch_inner"]["path"])
    expected_device = task_bind["scratch_inner"]["device"]
    smoke_identity = smoke["smoke_root_retention"]
    smoke_root = Path(smoke_identity["path"])
    rank_pattern = re.compile(
        rf"gadp-{HOLDER_JOB}-{task_bind['runtime']['slurm_step_id']}-"
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}-r([0-3])\.[A-Za-z0-9]{8}"
    )
    require(
        inner.parent == outer
        and inner.name == task_bind["scratch_inner"]["basename"]
        and outer.name == task_bind["scratch_outer"]["basename"]
        and smoke_root.parent == inner
        and smoke_root.name.startswith("generic-action-compile-smoke."),
        "scratch outer/inner/smoke lexical topology differs",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    outer_descriptor = os.open(outer, flags)
    inner_descriptor = -1
    try:
        outer_opened = os.fstat(outer_descriptor)
        outer_names_before = sorted(os.listdir(outer_descriptor))
        require(
            outer_names_before
            == sorted([inner.name, CHILD_SCRATCH_PROBE_BASENAME]),
            "terminal scratch outer inventory differs before anchored scan",
        )
        inner_descriptor = os.open(
            inner.name, flags, dir_fd=outer_descriptor
        )
        inner_opened = os.fstat(inner_descriptor)
        inner_named_opened = os.stat(
            inner.name, dir_fd=outer_descriptor, follow_symlinks=False
        )
        require(
            _scratch_inventory_metadata_tuple(inner_opened)
            == _scratch_inventory_metadata_tuple(inner_named_opened)
            and (
                inner_opened.st_dev,
                inner_opened.st_ino,
                inner_opened.st_uid,
                inner_opened.st_gid,
                stat.S_IMODE(inner_opened.st_mode),
            )
            == (
                task_bind["scratch_inner"]["device"],
                task_bind["scratch_inner"]["inode"],
                task_bind["scratch_inner"]["uid"],
                task_bind["scratch_inner"]["gid"],
                0o700,
            ),
            "bound scratch inner openat identity differs before inventory",
        )
        direct_names_before = sorted(os.listdir(inner_descriptor))
        rank_names = [
            name for name in direct_names_before if rank_pattern.fullmatch(name)
        ]
        expected_names = {
            CHILD_RENDERER_LOAD_LOCK_BASENAME,
            smoke_root.name,
            *rank_names,
        }
        require(
            len(direct_names_before) == 14
            and len(rank_names) == 12
            and len(expected_names) == 14
            and set(direct_names_before) == expected_names,
            "scratch direct inventory contains an unexpected or missing root",
        )
        ranks = [int(rank_pattern.fullmatch(name).group(1)) for name in rank_names]
        require(
            {rank: ranks.count(rank) for rank in range(4)}
            == {0: 3, 1: 3, 2: 3, 3: 3},
            "scratch rank-root cardinality differs",
        )
        probe_row = _stable_regular_file_inventory_row_at(
            outer_descriptor,
            CHILD_SCRATCH_PROBE_BASENAME,
            expected_device=expected_device,
        )
        regular_count, directory_count, inventory = (
            _scan_directory_contents_no_follow(
                inner_descriptor, expected_device=expected_device
            )
        )
        replay_regular_count, replay_directory_count, replay_inventory = (
            _scan_directory_contents_no_follow(
                inner_descriptor, expected_device=expected_device
            )
        )
        require(
            (regular_count, directory_count, inventory)
            == (replay_regular_count, replay_directory_count, replay_inventory),
            "scratch recursive inventory differs across two full-tree replays",
        )
        probe_replay_row = _stable_regular_file_inventory_row_at(
            outer_descriptor,
            CHILD_SCRATCH_PROBE_BASENAME,
            expected_device=expected_device,
        )
        require(
            probe_replay_row == probe_row,
            "scratch retained probe differs across the full inner-tree replay",
        )
        direct_names_after = sorted(os.listdir(inner_descriptor))
        inner_closed = os.fstat(inner_descriptor)
        inner_named_closed = os.stat(
            inner.name, dir_fd=outer_descriptor, follow_symlinks=False
        )
        outer_names_after = sorted(os.listdir(outer_descriptor))
        outer_closed = os.fstat(outer_descriptor)
        outer_lexical = outer.lstat()
        inner_lexical = inner.lstat()
        probe_lexical = (outer / CHILD_SCRATCH_PROBE_BASENAME).lstat()
        require(
            direct_names_before == direct_names_after
            and outer_names_before == outer_names_after
            and _scratch_inventory_metadata_tuple(inner_opened)
            == _scratch_inventory_metadata_tuple(inner_named_opened)
            == _scratch_inventory_metadata_tuple(inner_closed)
            == _scratch_inventory_metadata_tuple(inner_named_closed)
            == _scratch_inventory_metadata_tuple(inner_lexical)
            and _scratch_inventory_metadata_tuple(outer_opened)
            == _scratch_inventory_metadata_tuple(outer_closed)
            == _scratch_inventory_metadata_tuple(outer_lexical)
            and (probe_lexical.st_dev, probe_lexical.st_ino)
            == (probe_replay_row["device"], probe_replay_row["inode"])
            and outer.resolve(strict=True) == outer
            and inner.resolve(strict=True) == inner
            and (outer / CHILD_SCRATCH_PROBE_BASENAME).resolve(strict=True)
            == outer / CHILD_SCRATCH_PROBE_BASENAME,
            "scratch outer/inner/probe changed during fd-anchored inventory",
        )
    finally:
        if inner_descriptor >= 0:
            os.close(inner_descriptor)
        os.close(outer_descriptor)
    outer_metadata = outer_closed
    inner_metadata = inner_closed
    probe_path = outer / CHILD_SCRATCH_PROBE_BASENAME
    require(
        outer_metadata.st_nlink == 3
        and inner_metadata.st_nlink == 15
        and (
            outer_metadata.st_dev,
            outer_metadata.st_ino,
            outer_metadata.st_uid,
            outer_metadata.st_gid,
            stat.S_IMODE(outer_metadata.st_mode),
        )
        == (
            task_bind["scratch_outer"]["device"],
            task_bind["scratch_outer"]["inode"],
            task_bind["scratch_outer"]["uid"],
            task_bind["scratch_outer"]["gid"],
            0o700,
        )
        and (
            inner_metadata.st_dev,
            inner_metadata.st_ino,
            inner_metadata.st_uid,
            inner_metadata.st_gid,
            stat.S_IMODE(inner_metadata.st_mode),
        )
        == (
            task_bind["scratch_inner"]["device"],
            task_bind["scratch_inner"]["inode"],
            task_bind["scratch_inner"]["uid"],
            task_bind["scratch_inner"]["gid"],
            0o700,
        )
        and (probe_row["device"], probe_row["inode"])
        == (
            task_bind["retained_probe_file"]["device"],
            task_bind["retained_probe_file"]["inode"],
        )
        and probe_row["uid"] == task_bind["retained_probe_file"]["uid"]
        and probe_row["gid"] == task_bind["retained_probe_file"]["gid"]
        and probe_row["mode_octal"]
        == task_bind["retained_probe_file"]["mode_octal"]
        and probe_row["link_count"]
        == task_bind["retained_probe_file"]["link_count"]
        and probe_row["size_bytes"]
        == task_bind["retained_probe_file"]["size_bytes"]
        and probe_row["file_sha256"]
        == task_bind["retained_probe_file"]["file_sha256"],
        "terminal outer/inner/probe link-count or inode identity differs",
    )
    directory_rows: list[Mapping[str, Any]] = []
    for name in [smoke_root.name, *rank_names]:
        row = inventory.get(name)
        require(
            type(row) is dict
            and row.get("kind") == "directory"
            and row.get("device") == expected_device
            and row.get("uid") == CHILD_SCRATCH_OWNER_UID
            and row.get("gid") == CHILD_SCRATCH_OWNER_GID
            and row.get("mode_octal") == "0700"
            and type(row.get("link_count")) is int
            and row["link_count"] >= 2,
            "authorized scratch direct directory inventory row differs",
        )
        directory_rows.append(
            {
                "path": str(inner / name),
                "basename": name,
                "device": row["device"],
                "device_major_minor": CHILD_SCRATCH_MOUNT_MAJOR_MINOR,
                "inode": row["inode"],
                "uid": row["uid"],
                "gid": row["gid"],
                "mode_octal": row["mode_octal"],
                "link_count": row["link_count"],
            }
        )
    smoke_row = next(row for row in directory_rows if row["path"] == str(smoke_root))
    require(
        all(
            smoke_row[field] == smoke_identity[field]
            for field in (
                "path",
                "basename",
                "device",
                "inode",
                "uid",
                "gid",
                "mode_octal",
            )
        ),
        "retained smoke root differs from its signed resource receipt",
    )
    lock_row = inventory.get(CHILD_RENDERER_LOAD_LOCK_BASENAME)
    require(
        type(lock_row) is dict
        and lock_row.get("kind") == "regular_file"
        and lock_row.get("device") == task_bind["renderer_load_lock"]["device"]
        and lock_row.get("inode") == task_bind["renderer_load_lock"]["inode"]
        and lock_row.get("uid") == task_bind["renderer_load_lock"]["uid"]
        and lock_row.get("gid") == task_bind["renderer_load_lock"]["gid"]
        and lock_row.get("mode_octal")
        == task_bind["renderer_load_lock"]["mode_octal"]
        and lock_row.get("link_count")
        == task_bind["renderer_load_lock"]["link_count"]
        and lock_row.get("size_bytes")
        == task_bind["renderer_load_lock"]["size_bytes"]
        and lock_row.get("file_sha256")
        == task_bind["renderer_load_lock"]["empty_file_sha256"],
        "renderer lock direct/recursive inventory binding differs",
    )
    rank_roots = [inner / name for name in rank_names]
    tree_inventory = {
        probe_path.name: probe_row,
        inner.name: {
            "kind": "directory",
            "device": inner_metadata.st_dev,
            "inode": inner_metadata.st_ino,
            "uid": inner_metadata.st_uid,
            "gid": inner_metadata.st_gid,
            "mode_octal": f"{stat.S_IMODE(inner_metadata.st_mode):04o}",
            "link_count": inner_metadata.st_nlink,
            "size_bytes": inner_metadata.st_size,
            "allocated_512_blocks": inner_metadata.st_blocks,
            "mtime_ns": inner_metadata.st_mtime_ns,
            "ctime_ns": inner_metadata.st_ctime_ns,
            "children": inventory,
        },
    }
    regular_rows: list[Mapping[str, Any]] = []
    directory_rows_all: list[Mapping[str, Any]] = []

    def collect(rows: Mapping[str, Mapping[str, Any]]) -> None:
        for row in rows.values():
            if row["kind"] == "directory":
                directory_rows_all.append(row)
                collect(row["children"])
            else:
                regular_rows.append(row)

    collect(tree_inventory)
    terminal_outer = {
        "path": str(outer),
        "basename": outer.name,
        **_metadata_value(outer_metadata),
        "size_bytes": outer_metadata.st_size,
        "allocated_512_blocks": outer_metadata.st_blocks,
        "mtime_ns": outer_metadata.st_mtime_ns,
        "ctime_ns": outer_metadata.st_ctime_ns,
    }
    terminal_inner = {
        "path": str(inner),
        "basename": inner.name,
        **_metadata_value(inner_metadata),
        "size_bytes": inner_metadata.st_size,
        "allocated_512_blocks": inner_metadata.st_blocks,
        "mtime_ns": inner_metadata.st_mtime_ns,
        "ctime_ns": inner_metadata.st_ctime_ns,
    }
    return {
        "outer_entry_count": 2,
        "outer_probe": {
            "path": str(probe_path),
            "basename": probe_path.name,
            **{key: item for key, item in probe_row.items() if key != "kind"},
        },
        "terminal_outer_identity": terminal_outer,
        "terminal_inner_identity": terminal_inner,
        "direct_entry_count": 14,
        "direct_entry_identities": inventory,
        "rank_root_count": 12,
        "rank_root_count_by_rank": {str(rank): 3 for rank in range(4)},
        "rank_root_basenames": [path.name for path in rank_roots],
        "retained_smoke_root": smoke_row,
        "renderer_load_lock": task_bind["renderer_load_lock"],
        "recursive_regular_file_count": regular_count,
        "recursive_directory_count": directory_count,
        "tree_inventory": tree_inventory,
        "tree_inventory_sha256": object_sha256(tree_inventory),
        "tree_regular_file_count": len(regular_rows),
        "tree_directory_count_below_outer": len(directory_rows_all),
        "tree_logical_regular_file_bytes": sum(row["size_bytes"] for row in regular_rows),
        "tree_allocated_bytes_from_st_blocks_512": sum(
            row["allocated_512_blocks"] * 512
            for row in [*regular_rows, *directory_rows_all]
        )
        + outer_metadata.st_blocks * 512,
        "tree_maximum_mtime_ns": max(
            [
                outer_metadata.st_mtime_ns,
                *(row["mtime_ns"] for row in regular_rows),
                *(row["mtime_ns"] for row in directory_rows_all),
            ]
        ),
        "regular_file_double_sha256_same_fd_recomputed": True,
        "regular_file_ctime_stable_across_both_hash_passes": True,
        "recursive_tree_full_second_replay_equal": True,
        "same_device_no_symlink_single_link_regular_enforced": True,
        "no_unexpected_direct_entries": True,
    }


def seal_child_terminal_physical_attestation(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Seal the last all-physical child proof before terminal retention."""

    controller_plan, controller_plan_path, controller_plan_sha, exact2 = (
        load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    )
    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "attestation scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    _replay_child_scratch_prepare_physical(
        prepare, require_initial_link_count=False
    )
    preflight, preflight_path, preflight_sha = load_json(
        args.compute_preflight,
        "attestation compute preflight",
        args.expected_compute_preflight_sha256,
    )
    validate_compute_preflight(preflight)
    task_bind, task_bind_path, task_bind_sha = load_json(
        args.task_scratch_bind,
        "attestation child task scratch bind",
        args.expected_task_scratch_bind_sha256,
    )
    _replay_child_task_scratch_bind_physical(task_bind, prepare)
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "attestation exact2 generation audit",
        args.expected_generation_audit_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "attestation terminal host gate",
        args.expected_terminal_host_gate_sha256,
    )
    ffprobe_bin = validate_ffprobe(args.ffprobe_bin, args.expected_ffprobe_sha256)
    resource_module = _load_or_reuse_resource_contract(
        generator.METHOD_ROOT
        / "tools"
        / generator.RESOURCE_SPECIALIZED_BASENAME,
        TERMINAL_RESOURCE_CONTRACT_SHA256,
    )
    validate_exact2_audit(generation, exact2, ffprobe_bin, resource_module)
    validate_terminal_host_gate(terminal, resource_module)
    manifest, manifest_path, manifest_sha = load_json(
        args.blind_review_manifest,
        "attestation blind-review manifest",
        args.expected_blind_review_manifest_sha256,
    )
    key, key_path, key_sha = load_json(
        args.blind_review_key,
        "attestation blind-review key",
        args.expected_blind_review_key_sha256,
    )
    pseudo_review = {
        "packet_id": manifest.get("packet_id"),
        "required_ffprobe": manifest.get("required_ffprobe"),
        "generation_audit": _receipt_reference(
            generation, generation_path, generation_sha
        ),
        "blind_review_manifest": _receipt_reference(
            manifest, manifest_path, manifest_sha
        ),
        "sealed_key": _receipt_reference(key, key_path, key_sha),
    }
    reopened_manifest, reopened_key = _validate_blind_packet(
        review=pseudo_review,
        exact2_plan=exact2,
        generation_audit=generation,
        ffprobe_bin=ffprobe_bin,
    )
    smoke = _generation_resource_smoke_receipt(generation, resource_module)
    scratch_inventory = _attest_authorized_scratch_inventory(task_bind, smoke)
    inner = _task_scratch_physical_value(
        task_bind["scratch_inner"]["path"], prepare
    )
    census = _slurm_child_cgroup_census(args.supervisor_pid)
    require(
        prepare["controller_plan"]
        == {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }
        and task_bind["runtime"]
        == preflight["runtime"]
        == prepare["runtime"]
        and preflight["scratch_prepare"]
        == _receipt_reference(prepare, prepare_path, prepare_sha)
        and task_bind["scratch_prepare"]
        == _receipt_reference(prepare, prepare_path, prepare_sha)
        and task_bind["compute_preflight"]
        == _receipt_reference(preflight, preflight_path, preflight_sha)
        and generation["task_scratch_bind"]
        == _receipt_reference(task_bind, task_bind_path, task_bind_sha)
        and inner["inode"] == task_bind["scratch_inner"]["inode"]
        and generation["compute_preflight"]
        == terminal["compute_preflight"]
        == _receipt_reference(preflight, preflight_path, preflight_sha)
        and reopened_manifest == manifest
        and reopened_key == key,
        "child terminal attestation reference chain differs",
    )
    unsigned = {
        "schema_version": CHILD_TERMINAL_PHYSICAL_ATTESTATION_SCHEMA,
        "authority": CHILD_SCRATCH_AUTHORITY,
        "runtime": prepare["runtime"],
        "controller_plan": prepare["controller_plan"],
        "scratch_prepare": _receipt_reference(prepare, prepare_path, prepare_sha),
        "compute_preflight": _receipt_reference(
            preflight, preflight_path, preflight_sha
        ),
        "task_scratch_bind": _receipt_reference(
            task_bind, task_bind_path, task_bind_sha
        ),
        "generation_audit": _receipt_reference(
            generation, generation_path, generation_sha
        ),
        "terminal_host_gate": _receipt_reference(
            terminal, terminal_path, terminal_sha
        ),
        "blind_review_manifest": _receipt_reference(
            manifest, manifest_path, manifest_sha
        ),
        "blind_review_key": _receipt_reference(key, key_path, key_sha),
        "scratch_outer": prepare["scratch_root"],
        "scratch_inner": task_bind["scratch_inner"],
        "renderer_load_lock": task_bind["renderer_load_lock"],
        "scratch_inventory": scratch_inventory,
        "slurm_cgroup_census": census,
        "physical_validation": {
            "prepare_outer_mount_statfs_identity_replayed": True,
            "compute_preflight_media_and_scratch_replayed": True,
            "generation_audit_and_rank_scratch_replayed": True,
            "terminal_journal_monitor_pid_and_resource_replayed": True,
            "blind_packet_media_hash_probe_topology_replayed": True,
            "outer_contains_exact_retained_probe_and_authorized_inner": True,
            "no_unexpected_same_cgroup_processes": True,
        },
        "formal_candidate_count": 2,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
        "ready_for_terminal_retention_seal": True,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def _validate_tree_inventory_manifest(
    tree: Any, *, expected_device: int
) -> Mapping[str, int]:
    require(type(tree) is dict and bool(tree), "scratch tree inventory is absent")
    seen_inodes: set[tuple[int, int]] = set()
    regular_count = 0
    directory_count = 0
    logical_bytes = 0
    allocated_bytes = 0
    maximum_mtime_ns = 0
    common_fields = {
        "kind",
        "device",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
        "size_bytes",
        "allocated_512_blocks",
        "mtime_ns",
        "ctime_ns",
    }

    def visit(rows: Mapping[str, Any]) -> None:
        nonlocal regular_count, directory_count, logical_bytes
        nonlocal allocated_bytes, maximum_mtime_ns
        require(
            type(rows) is dict and list(rows) == sorted(rows),
            "scratch tree inventory mapping is not canonical basename order",
        )
        for basename, row in rows.items():
            require(
                isinstance(basename, str)
                and basename not in {"", ".", ".."}
                and "/" not in basename
                and "\x00" not in basename
                and type(row) is dict,
                "scratch tree inventory relative basename differs",
            )
            kind = row.get("kind")
            expected_fields = (
                common_fields | {"children"}
                if kind == "directory"
                else common_fields | {"file_sha256"}
            )
            require(
                set(row) == expected_fields
                and kind in {"directory", "regular_file"}
                and type(row.get("device")) is int
                and row["device"] == expected_device
                and type(row.get("inode")) is int
                and row["inode"] > 0
                and row.get("uid") == CHILD_SCRATCH_OWNER_UID
                and row.get("gid") == CHILD_SCRATCH_OWNER_GID
                and re.fullmatch(r"[0-7]{4}", str(row.get("mode_octal")))
                is not None
                and type(row.get("link_count")) is int
                and row["link_count"] > 0
                and type(row.get("size_bytes")) is int
                and row["size_bytes"] >= 0
                and type(row.get("allocated_512_blocks")) is int
                and row["allocated_512_blocks"] >= 0
                and type(row.get("mtime_ns")) is int
                and row["mtime_ns"] > 0
                and type(row.get("ctime_ns")) is int
                and row["ctime_ns"] > 0
                and (row["device"], row["inode"]) not in seen_inodes,
                "scratch tree inventory entry closure differs",
            )
            seen_inodes.add((row["device"], row["inode"]))
            allocated_bytes += row["allocated_512_blocks"] * 512
            maximum_mtime_ns = max(maximum_mtime_ns, row["mtime_ns"])
            if kind == "directory":
                require(
                    row["link_count"] >= 2,
                    "scratch tree directory link count differs",
                )
                directory_count += 1
                visit(row["children"])
            else:
                require(
                    row["link_count"] == 1
                    and SHA256_RE.fullmatch(str(row.get("file_sha256")))
                    is not None,
                    "scratch tree regular-file closure differs",
                )
                regular_count += 1
                logical_bytes += row["size_bytes"]

    visit(tree)
    return {
        "regular_count": regular_count,
        "directory_count": directory_count,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "maximum_mtime_ns": maximum_mtime_ns,
    }


def _scratch_inventory_creation_identity_closes(
    inventory: Any, outer: Any, inner: Any, load_lock: Any
) -> bool:
    """Bind terminal inventory rows to the signed creation identities."""

    if not all(type(item) is dict for item in (inventory, outer, inner, load_lock)):
        return False
    terminal_outer = inventory.get("terminal_outer_identity")
    terminal_inner = inventory.get("terminal_inner_identity")
    outer_probe = inventory.get("outer_probe")
    tree = inventory.get("tree_inventory")
    if not all(
        type(item) is dict
        for item in (terminal_outer, terminal_inner, outer_probe, tree)
    ):
        return False
    inner_basename = inner.get("basename")
    tree_probe = tree.get(CHILD_SCRATCH_PROBE_BASENAME)
    tree_inner = tree.get(inner_basename)
    if type(tree_probe) is not dict or type(tree_inner) is not dict:
        return False
    root_identity_fields = (
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
    )
    tree_metadata_fields = (
        "device",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
        "size_bytes",
        "allocated_512_blocks",
        "mtime_ns",
        "ctime_ns",
    )
    tree_children = tree_inner.get("children")
    tree_lock = (
        tree_children.get(CHILD_RENDERER_LOAD_LOCK_BASENAME)
        if type(tree_children) is dict
        else None
    )
    return bool(
        set(tree) == {CHILD_SCRATCH_PROBE_BASENAME, inner_basename}
        and inventory.get("direct_entry_identities") == tree_children
        and type(tree_children) is dict
        and len(tree_children) == 14
        and all(terminal_outer.get(field) == outer.get(field) for field in root_identity_fields)
        and terminal_outer.get("link_count") == 3
        and all(terminal_inner.get(field) == inner.get(field) for field in root_identity_fields)
        and terminal_inner.get("link_count") == 15
        and inventory.get("renderer_load_lock") == load_lock
        and outer_probe.get("path")
        == str(Path(str(outer.get("path"))) / CHILD_SCRATCH_PROBE_BASENAME)
        and outer_probe.get("basename") == CHILD_SCRATCH_PROBE_BASENAME
        and outer_probe.get("device") == outer.get("device")
        and outer_probe.get("device_major_minor") == outer.get("device_major_minor")
        and tree_probe.get("kind") == "regular_file"
        and all(
            tree_probe.get(field) == outer_probe.get(field)
            for field in tree_metadata_fields
        )
        and tree_probe.get("file_sha256") == outer_probe.get("file_sha256")
        and tree_inner.get("kind") == "directory"
        and all(
            tree_inner.get(field) == terminal_inner.get(field)
            for field in tree_metadata_fields
        )
        and type(tree_lock) is dict
        and tree_lock.get("kind") == "regular_file"
        and all(
            tree_lock.get(field) == load_lock.get(field)
            for field in (
                "device",
                "inode",
                "uid",
                "gid",
                "mode_octal",
                "link_count",
                "size_bytes",
            )
        )
        and tree_lock.get("file_sha256") == load_lock.get("empty_file_sha256")
    )


def _validate_child_terminal_attestation_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(
        value,
        CHILD_TERMINAL_PHYSICAL_ATTESTATION_SCHEMA,
        "child terminal physical attestation",
    )
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    reference_names = {
        "scratch_prepare",
        "compute_preflight",
        "task_scratch_bind",
        "generation_audit",
        "terminal_host_gate",
        "blind_review_manifest",
        "blind_review_key",
    }
    controller_plan_fields = {"path", "file_sha256", "plan_digest"}
    runtime_fields = {
        "slurm_job_id",
        "slurm_step_id",
        "hostname",
        "sole_numbered_compute_child_required",
    }
    outer_fields = {
        "path",
        "basename",
        "canonical_non_symlink",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    inner_fields = outer_fields - {"canonical_non_symlink"}
    terminal_identity_fields = inner_fields | {
        "size_bytes",
        "allocated_512_blocks",
        "mtime_ns",
        "ctime_ns",
    }
    census = value.get("slurm_cgroup_census")
    identities = census.get("same_cgroup_processes") if type(census) is dict else None
    runtime = value.get("runtime")
    controller_plan = value.get("controller_plan")
    outer = value.get("scratch_outer")
    inner = value.get("scratch_inner")
    load_lock = value.get("renderer_load_lock")
    scratch_inventory = value.get("scratch_inventory")
    tree_manifest = (
        scratch_inventory.get("tree_inventory")
        if type(scratch_inventory) is dict
        else None
    )
    tree_summary = _validate_tree_inventory_manifest(
        tree_manifest, expected_device=os.makedev(253, 0)
    )
    step_id = str(runtime.get("slurm_step_id", "")) if type(runtime) is dict else ""
    _validate_renderer_load_lock_shape(load_lock, inner)
    inventory_row_fields = {
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    inventory_rows = (
        [scratch_inventory.get("retained_smoke_root")]
        if type(scratch_inventory) is dict
        else []
    )
    rank_basenames = (
        scratch_inventory.get("rank_root_basenames")
        if type(scratch_inventory) is dict
        else None
    )
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "authority",
            "runtime",
            "controller_plan",
            *reference_names,
            "scratch_outer",
            "scratch_inner",
            "renderer_load_lock",
            "scratch_inventory",
            "slurm_cgroup_census",
            "physical_validation",
            "formal_candidate_count",
            "diagnostic_task_count",
            "optimizer_authorized",
            "ready_for_terminal_retention_seal",
            "receipt_digest",
        }
        and value.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(controller_plan) is dict
        and set(controller_plan) == controller_plan_fields
        and Path(str(controller_plan.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(controller_plan.get("file_sha256")))
        is not None
        and SHA256_RE.fullmatch(str(controller_plan.get("plan_digest")))
        is not None
        and all(
            type(value.get(name)) is dict
            and set(value[name]) == reference_fields
            and Path(str(value[name].get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(value[name].get("file_sha256")))
            is not None
            and SHA256_RE.fullmatch(str(value[name].get("receipt_digest")))
            is not None
            for name in reference_names
        )
        and type(runtime) is dict
        and set(runtime) == runtime_fields
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and runtime.get("hostname") == HOLDER_NODE
        and step_id.isdecimal()
        and runtime.get("sole_numbered_compute_child_required") is True
        and type(outer) is dict
        and set(outer) == outer_fields
        and outer.get("path") == f"/tmp/{CHILD_SCRATCH_LEAF_PREFIX}{HOLDER_JOB}-{step_id}"
        and outer.get("basename") == Path(str(outer.get("path"))).name
        and outer.get("canonical_non_symlink") is True
        and outer.get("device") == os.makedev(253, 0)
        and outer.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(outer.get("inode")) is int
        and outer["inode"] > 0
        and outer.get("uid") == CHILD_SCRATCH_OWNER_UID
        and outer.get("gid") == CHILD_SCRATCH_OWNER_GID
        and outer.get("mode_octal") == "0700"
        and outer.get("link_count") == 2
        and type(inner) is dict
        and set(inner) == inner_fields
        and Path(str(inner.get("path"))).parent == Path(str(outer.get("path")))
        and inner.get("basename") == Path(str(inner.get("path"))).name
        and re.fullmatch(
            rf"arms-incomplete-exact2-{HOLDER_JOB}-{step_id}\.[0-9a-f]{{8}}",
            str(inner.get("basename")),
        )
        is not None
        and inner.get("device") == outer.get("device")
        and inner.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(inner.get("inode")) is int
        and inner["inode"] > 0
        and inner.get("uid") == CHILD_SCRATCH_OWNER_UID
        and inner.get("gid") == CHILD_SCRATCH_OWNER_GID
        and inner.get("mode_octal") == "0700"
        and inner.get("link_count") == 2
        and type(scratch_inventory) is dict
        and set(scratch_inventory)
        == {
            "direct_entry_count",
            "direct_entry_identities",
            "outer_entry_count",
            "outer_probe",
            "terminal_outer_identity",
            "terminal_inner_identity",
            "rank_root_count",
            "rank_root_count_by_rank",
            "rank_root_basenames",
            "retained_smoke_root",
            "renderer_load_lock",
            "recursive_regular_file_count",
            "recursive_directory_count",
            "tree_inventory",
            "tree_inventory_sha256",
            "tree_regular_file_count",
            "tree_directory_count_below_outer",
            "tree_logical_regular_file_bytes",
            "tree_allocated_bytes_from_st_blocks_512",
            "tree_maximum_mtime_ns",
            "regular_file_double_sha256_same_fd_recomputed",
            "regular_file_ctime_stable_across_both_hash_passes",
            "recursive_tree_full_second_replay_equal",
            "same_device_no_symlink_single_link_regular_enforced",
            "no_unexpected_direct_entries",
        }
        and scratch_inventory.get("direct_entry_count") == 14
        and type(scratch_inventory.get("direct_entry_identities")) is dict
        and scratch_inventory.get("outer_entry_count") == 2
        and type(scratch_inventory.get("outer_probe")) is dict
        and set(scratch_inventory["outer_probe"])
        == {
            "path",
            "basename",
            "device",
            "device_major_minor",
            "inode",
            "uid",
            "gid",
            "mode_octal",
            "link_count",
            "size_bytes",
            "file_sha256",
            "allocated_512_blocks",
            "mtime_ns",
            "ctime_ns",
        }
        and scratch_inventory["outer_probe"].get("path")
        == str(Path(str(outer.get("path"))) / CHILD_SCRATCH_PROBE_BASENAME)
        and scratch_inventory["outer_probe"].get("basename")
        == CHILD_SCRATCH_PROBE_BASENAME
        and scratch_inventory["outer_probe"].get("device") == outer.get("device")
        and scratch_inventory["outer_probe"].get("mode_octal") == "0600"
        and scratch_inventory["outer_probe"].get("link_count") == 1
        and scratch_inventory["outer_probe"].get("size_bytes")
        == len(CHILD_SCRATCH_PROBE_BYTES)
        and scratch_inventory["outer_probe"].get("file_sha256")
        == hashlib.sha256(CHILD_SCRATCH_PROBE_BYTES).hexdigest()
        and type(scratch_inventory.get("terminal_outer_identity")) is dict
        and set(scratch_inventory["terminal_outer_identity"])
        == terminal_identity_fields
        and scratch_inventory["terminal_outer_identity"].get("path")
        == outer.get("path")
        and scratch_inventory["terminal_outer_identity"].get("device")
        == outer.get("device")
        and scratch_inventory["terminal_outer_identity"].get("inode")
        == outer.get("inode")
        and scratch_inventory["terminal_outer_identity"].get("link_count") == 3
        and type(scratch_inventory.get("terminal_inner_identity")) is dict
        and set(scratch_inventory["terminal_inner_identity"])
        == terminal_identity_fields
        and scratch_inventory["terminal_inner_identity"].get("path")
        == inner.get("path")
        and scratch_inventory["terminal_inner_identity"].get("device")
        == inner.get("device")
        and scratch_inventory["terminal_inner_identity"].get("inode")
        == inner.get("inode")
        and scratch_inventory["terminal_inner_identity"].get("link_count") == 15
        and scratch_inventory.get("rank_root_count") == 12
        and scratch_inventory.get("rank_root_count_by_rank")
        == {"0": 3, "1": 3, "2": 3, "3": 3}
        and type(rank_basenames) is list
        and len(rank_basenames) == len(set(rank_basenames)) == 12
        and all(
            re.fullmatch(
                rf"gadp-{HOLDER_JOB}-{step_id}-[A-Za-z0-9][A-Za-z0-9._-]{{0,159}}-r[0-3]\.[A-Za-z0-9]{{8}}",
                str(name),
            )
            is not None
            for name in rank_basenames
        )
        and len(inventory_rows) == 1
        and all(
            type(row) is dict
            and set(row) == inventory_row_fields
            and Path(str(row.get("path"))).parent == Path(str(inner.get("path")))
            and row.get("basename") == Path(str(row.get("path"))).name
            and row.get("device") == inner.get("device")
            and row.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
            and type(row.get("inode")) is int
            and row["inode"] > 0
            and row.get("uid") == CHILD_SCRATCH_OWNER_UID
            and row.get("gid") == CHILD_SCRATCH_OWNER_GID
            and row.get("mode_octal") == "0700"
            and type(row.get("link_count")) is int
            and row["link_count"] >= 2
            for row in inventory_rows
        )
        and scratch_inventory["retained_smoke_root"]["basename"].startswith(
            "generic-action-compile-smoke."
        )
        and scratch_inventory.get("renderer_load_lock") == load_lock
        and _scratch_inventory_creation_identity_closes(
            scratch_inventory, outer, inner, load_lock
        )
        and type(scratch_inventory.get("recursive_regular_file_count")) is int
        and scratch_inventory["recursive_regular_file_count"] >= 1
        and type(scratch_inventory.get("recursive_directory_count")) is int
        and scratch_inventory["recursive_directory_count"] >= 13
        and type(tree_manifest) is dict
        and set(tree_manifest)
        == {CHILD_SCRATCH_PROBE_BASENAME, inner.get("basename")}
        and object_sha256(tree_manifest)
        == scratch_inventory.get("tree_inventory_sha256")
        and SHA256_RE.fullmatch(str(scratch_inventory.get("tree_inventory_sha256")))
        is not None
        and scratch_inventory.get("tree_regular_file_count")
        == scratch_inventory["recursive_regular_file_count"] + 1
        == tree_summary["regular_count"]
        and scratch_inventory.get("tree_directory_count_below_outer")
        == scratch_inventory["recursive_directory_count"] + 1
        == tree_summary["directory_count"]
        and type(scratch_inventory.get("tree_logical_regular_file_bytes")) is int
        and scratch_inventory["tree_logical_regular_file_bytes"] >= 0
        and scratch_inventory["tree_logical_regular_file_bytes"]
        == tree_summary["logical_bytes"]
        and type(
            scratch_inventory.get("tree_allocated_bytes_from_st_blocks_512")
        )
        is int
        and scratch_inventory["tree_allocated_bytes_from_st_blocks_512"] >= 0
        and scratch_inventory["tree_allocated_bytes_from_st_blocks_512"]
        == tree_summary["allocated_bytes"]
        + scratch_inventory["terminal_outer_identity"]["allocated_512_blocks"] * 512
        and type(scratch_inventory.get("tree_maximum_mtime_ns")) is int
        and scratch_inventory["tree_maximum_mtime_ns"] > 0
        and scratch_inventory["tree_maximum_mtime_ns"]
        == max(
            tree_summary["maximum_mtime_ns"],
            scratch_inventory["terminal_outer_identity"]["mtime_ns"],
        )
        and scratch_inventory.get("regular_file_double_sha256_same_fd_recomputed")
        is True
        and scratch_inventory.get(
            "regular_file_ctime_stable_across_both_hash_passes"
        )
        is True
        and scratch_inventory.get("recursive_tree_full_second_replay_equal")
        is True
        and scratch_inventory.get(
            "same_device_no_symlink_single_link_regular_enforced"
        )
        is True
        and scratch_inventory.get("no_unexpected_direct_entries") is True
        and type(census) is dict
        and set(census)
        == {
            "supervisor",
            "attestation_process",
            "same_cgroup_process_count",
            "same_cgroup_processes",
            "cgroup_procs_path",
            "stable_membership_before",
            "stable_membership_after",
            "identities_and_start_ticks_replayed_stably",
            "unexpected_same_cgroup_process_count",
            "numbered_steps",
            "sole_numbered_step",
            "cgroup_census_before_terminal_retention",
        }
        and type(identities) is list
        and len(identities) == census.get("same_cgroup_process_count")
        and len(identities) >= 2
        and all(
            type(row) is dict
            and set(row) == {"pid", "state", "parent_pid", "start_ticks", "cgroup_v2_path"}
            and all(type(row[field]) is int and row[field] >= 0 for field in ("pid", "parent_pid", "start_ticks"))
            and isinstance(row["state"], str)
            and re.fullmatch(r"[A-Z]", row["state"]) is not None
            and row["state"] != "Z"
            and str(row["cgroup_v2_path"]).startswith("/")
            for row in identities
        )
        and census.get("cgroup_procs_path")
        == str(
            Path("/sys/fs/cgroup")
            / str(identities[0]["cgroup_v2_path"]).lstrip("/")
            / "cgroup.procs"
        )
        and type(census.get("stable_membership_before")) is list
        and census.get("stable_membership_before")
        == census.get("stable_membership_after")
        == sorted(row["pid"] for row in identities)
        and census.get("identities_and_start_ticks_replayed_stably") is True
        and census.get("unexpected_same_cgroup_process_count") == 0
        and census.get("numbered_steps") == [census.get("sole_numbered_step")]
        and census.get("sole_numbered_step")
        == f"{HOLDER_JOB}.{step_id}"
        and census.get("cgroup_census_before_terminal_retention") is True
        and value.get("physical_validation")
        == {
            "prepare_outer_mount_statfs_identity_replayed": True,
            "compute_preflight_media_and_scratch_replayed": True,
            "generation_audit_and_rank_scratch_replayed": True,
            "terminal_journal_monitor_pid_and_resource_replayed": True,
            "blind_packet_media_hash_probe_topology_replayed": True,
            "outer_contains_exact_retained_probe_and_authorized_inner": True,
            "no_unexpected_same_cgroup_processes": True,
        }
        and value.get("formal_candidate_count") == 2
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False
        and value.get("ready_for_terminal_retention_seal") is True,
        "child terminal physical attestation field/authority closure differs",
    )
    return value



def seal_child_scratch_retained_terminal(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Seal point-in-time success retention; never remove any scratch byte."""

    controller_plan, controller_plan_path, controller_plan_sha, exact2 = (
        load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    )
    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "terminal-retained scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    _replay_child_scratch_prepare_physical(
        prepare, require_initial_link_count=False
    )
    preflight, preflight_path, preflight_sha = load_json(
        args.compute_preflight,
        "terminal-retained compute preflight",
        args.expected_compute_preflight_sha256,
    )
    validate_compute_preflight(preflight)
    task_bind, task_bind_path, task_bind_sha = load_json(
        args.task_scratch_bind,
        "terminal-retained task scratch bind",
        args.expected_task_scratch_bind_sha256,
    )
    _replay_child_task_scratch_bind_physical(task_bind, prepare)
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "terminal-retained generation audit",
        args.expected_generation_audit_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "terminal-retained terminal host gate",
        args.expected_terminal_host_gate_sha256,
    )
    attestation, attestation_path, attestation_sha = load_json(
        args.physical_attestation,
        "terminal-retained physical attestation",
        args.expected_physical_attestation_sha256,
    )
    _validate_child_terminal_attestation_shape(attestation)
    ffprobe_bin = validate_ffprobe(PORTABLE_FFPROBE_PATH, PORTABLE_FFPROBE_SHA256)
    resource_module = _load_or_reuse_resource_contract(
        generator.METHOD_ROOT / "tools" / generator.RESOURCE_SPECIALIZED_BASENAME,
        TERMINAL_RESOURCE_CONTRACT_SHA256,
    )
    validate_exact2_audit(generation, exact2, ffprobe_bin, resource_module)
    validate_terminal_host_gate(terminal, resource_module)
    smoke = _generation_resource_smoke_receipt(generation, resource_module)
    mount_before = _retention_mount_snapshot(Path(prepare["scratch_root"]["path"]))
    retained_inventory = _attest_authorized_scratch_inventory(task_bind, smoke)
    mount_after = _retention_mount_snapshot(Path(prepare["scratch_root"]["path"]))
    second_census = _slurm_child_cgroup_census(args.supervisor_pid)
    prepare_ref = _receipt_reference(prepare, prepare_path, prepare_sha)
    preflight_ref = _receipt_reference(preflight, preflight_path, preflight_sha)
    task_bind_ref = _receipt_reference(task_bind, task_bind_path, task_bind_sha)
    generation_ref = _receipt_reference(generation, generation_path, generation_sha)
    terminal_ref = _receipt_reference(terminal, terminal_path, terminal_sha)
    attestation_ref = _receipt_reference(
        attestation, attestation_path, attestation_sha
    )
    require(
        mount_before == mount_after
        and prepare["controller_plan"]
        == {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }
        and attestation["runtime"]
        == task_bind["runtime"]
        == preflight["runtime"]
        == prepare["runtime"]
        and preflight["scratch_prepare"] == prepare_ref
        and task_bind["scratch_prepare"] == prepare_ref
        and task_bind["compute_preflight"] == preflight_ref
        and generation["compute_preflight"] == preflight_ref
        and generation["task_scratch_bind"] == task_bind_ref
        and terminal["compute_preflight"] == preflight_ref
        and attestation["scratch_prepare"] == prepare_ref
        and attestation["compute_preflight"] == preflight_ref
        and attestation["task_scratch_bind"] == task_bind_ref
        and attestation["generation_audit"] == generation_ref
        and attestation["terminal_host_gate"] == terminal_ref
        and attestation["scratch_outer"] == prepare["scratch_root"]
        and attestation["scratch_inner"] == task_bind["scratch_inner"]
        and attestation["renderer_load_lock"] == task_bind["renderer_load_lock"]
        and all(
            attestation["scratch_inventory"]["outer_probe"].get(field)
            == task_bind["retained_probe_file"].get(field)
            for field in task_bind["retained_probe_file"]
        )
        and retained_inventory == attestation["scratch_inventory"],
        "terminal retention physical/reference replay differs",
    )
    statvfs = os.statvfs(CHILD_SCRATCH_PARENT)
    host_capacity = {
        "observation_node": HOLDER_NODE,
        "observation_scope": "host_wide_ext4_statvfs_at_child_terminal_seal",
        "filesystem_total_bytes": statvfs.f_blocks * statvfs.f_frsize,
        "filesystem_available_bytes_at_terminal_seal": statvfs.f_bavail
        * statvfs.f_frsize,
        "filesystem_total_inodes": statvfs.f_files,
        "filesystem_used_inodes_at_terminal_seal": statvfs.f_files
        - statvfs.f_ffree,
        "pre_r6_read_only_observation_available_bytes": 297_197_441_024,
        "pre_r6_read_only_observation_used_inodes": 1_389_837,
        "pre_r6_read_only_observation_node": HOLDER_NODE,
        "pre_r6_read_only_observation_source": (
            "gpu215 statvfs read-only audit before r6 candidate"
        ),
        "host_wide_values_are_not_tree_usage": True,
        "future_capacity_guaranteed": False,
    }
    require(
        host_capacity["filesystem_total_bytes"] == 470_343_073_792
        and host_capacity["filesystem_total_inodes"] == 29_237_248
        and host_capacity["filesystem_available_bytes_at_terminal_seal"] > 0,
        "terminal retention host capacity identity differs",
    )
    unsigned = {
        "schema_version": CHILD_SCRATCH_RETAINED_TERMINAL_SCHEMA,
        "authority": CHILD_SCRATCH_AUTHORITY,
        "runtime": prepare["runtime"],
        "controller_plan": prepare["controller_plan"],
        "scratch_prepare": prepare_ref,
        "compute_preflight": preflight_ref,
        "task_scratch_bind": task_bind_ref,
        "generation_audit": generation_ref,
        "terminal_host_gate": terminal_ref,
        "physical_attestation": attestation_ref,
        "scratch_outer_creation_identity": prepare["scratch_root"],
        "scratch_inner_creation_identity": task_bind["scratch_inner"],
        "renderer_load_lock_creation_identity": task_bind["renderer_load_lock"],
        "retained_inventory": retained_inventory,
        "mount_snapshot": mount_after,
        "second_terminal_cgroup_census": second_census,
        "host_capacity_observation": host_capacity,
        "retention_semantics": {
            "retained_at_child_terminal_seal": True,
            "deletion_attempted": False,
            "cleanup_authorized": False,
            "manual_cleanup_authorized_by_release": False,
            "retained_nonreusable": True,
            "parent_physical_replay_allowed": False,
            "future_availability_guaranteed": False,
            "future_content_immutability_guaranteed": False,
            "persistence_after_step_or_reboot_guaranteed": False,
            "cluster_or_admin_cleanup_controlled_by_release": False,
            "second_point_in_time_inventory_equal_to_attestation_observation": True,
            "continuous_immutability_between_observations_guaranteed": False,
            "atomic_filesystem_snapshot_performed": False,
        },
        "formal_candidate_count": 2,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def _validate_child_scratch_retained_terminal_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(
        value,
        CHILD_SCRATCH_RETAINED_TERMINAL_SCHEMA,
        "child scratch retained terminal",
    )
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    references = {
        "scratch_prepare",
        "compute_preflight",
        "task_scratch_bind",
        "generation_audit",
        "terminal_host_gate",
        "physical_attestation",
    }
    runtime = value.get("runtime")
    controller_plan = value.get("controller_plan")
    outer = value.get("scratch_outer_creation_identity")
    inner = value.get("scratch_inner_creation_identity")
    load_lock = value.get("renderer_load_lock_creation_identity")
    inventory = value.get("retained_inventory")
    mount = value.get("mount_snapshot")
    census = value.get("second_terminal_cgroup_census")
    capacity = value.get("host_capacity_observation")
    runtime_fields = {
        "slurm_job_id",
        "slurm_step_id",
        "hostname",
        "sole_numbered_compute_child_required",
    }
    outer_fields = {
        "path",
        "basename",
        "canonical_non_symlink",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    inner_fields = outer_fields - {"canonical_non_symlink"}
    inventory_fields = {
        "direct_entry_count",
        "direct_entry_identities",
        "outer_entry_count",
        "outer_probe",
        "terminal_outer_identity",
        "terminal_inner_identity",
        "rank_root_count",
        "rank_root_count_by_rank",
        "rank_root_basenames",
        "retained_smoke_root",
        "renderer_load_lock",
        "recursive_regular_file_count",
        "recursive_directory_count",
        "tree_inventory",
        "tree_inventory_sha256",
        "tree_regular_file_count",
        "tree_directory_count_below_outer",
        "tree_logical_regular_file_bytes",
        "tree_allocated_bytes_from_st_blocks_512",
        "tree_maximum_mtime_ns",
        "regular_file_double_sha256_same_fd_recomputed",
        "regular_file_ctime_stable_across_both_hash_passes",
        "recursive_tree_full_second_replay_equal",
        "same_device_no_symlink_single_link_regular_enforced",
        "no_unexpected_direct_entries",
    }
    tree_summary = _validate_tree_inventory_manifest(
        inventory.get("tree_inventory") if type(inventory) is dict else None,
        expected_device=os.makedev(253, 0),
    )
    require(
        type(inner) is dict and type(load_lock) is dict,
        "retained scratch inner/renderer-lock shape differs",
    )
    _validate_renderer_load_lock_shape(load_lock, inner)
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "authority",
            "runtime",
            "controller_plan",
            *references,
            "scratch_outer_creation_identity",
            "scratch_inner_creation_identity",
            "renderer_load_lock_creation_identity",
            "retained_inventory",
            "mount_snapshot",
            "second_terminal_cgroup_census",
            "host_capacity_observation",
            "retention_semantics",
            "formal_candidate_count",
            "diagnostic_task_count",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("authority") == CHILD_SCRATCH_AUTHORITY
        and all(
            type(value.get(name)) is dict
            and set(value[name]) == reference_fields
            and Path(str(value[name].get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(value[name].get("file_sha256"))) is not None
            and SHA256_RE.fullmatch(str(value[name].get("receipt_digest"))) is not None
            for name in references
        )
        and type(runtime) is dict
        and set(runtime) == runtime_fields
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and str(runtime.get("slurm_step_id", "")).isdecimal()
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and type(controller_plan) is dict
        and set(controller_plan) == {"path", "file_sha256", "plan_digest"}
        and Path(str(controller_plan.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(controller_plan.get("file_sha256")))
        is not None
        and SHA256_RE.fullmatch(str(controller_plan.get("plan_digest")))
        is not None
        and type(outer) is dict
        and set(outer) == outer_fields
        and outer.get("path")
        == f"/tmp/{CHILD_SCRATCH_LEAF_PREFIX}{HOLDER_JOB}-{runtime['slurm_step_id']}"
        and outer.get("basename") == Path(str(outer.get("path"))).name
        and outer.get("canonical_non_symlink") is True
        and outer.get("device") == os.makedev(253, 0)
        and outer.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(outer.get("inode")) is int
        and outer["inode"] > 0
        and outer.get("uid") == CHILD_SCRATCH_OWNER_UID
        and outer.get("gid") == CHILD_SCRATCH_OWNER_GID
        and outer.get("mode_octal") == "0700"
        and outer.get("link_count") == 2
        and type(inner) is dict
        and set(inner) == inner_fields
        and Path(str(inner.get("path"))).parent == Path(outer["path"])
        and inner.get("device") == outer.get("device")
        and inner.get("device_major_minor") == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and type(inner.get("inode")) is int
        and inner["inode"] > 0
        and inner.get("uid") == CHILD_SCRATCH_OWNER_UID
        and inner.get("gid") == CHILD_SCRATCH_OWNER_GID
        and inner.get("mode_octal") == "0700"
        and inner.get("link_count") == 2
        and type(inventory) is dict
        and set(inventory) == inventory_fields
        and inventory.get("renderer_load_lock") == load_lock
        and _scratch_inventory_creation_identity_closes(
            inventory, outer, inner, load_lock
        )
        and type(inventory.get("tree_inventory")) is dict
        and object_sha256(inventory["tree_inventory"])
        == inventory.get("tree_inventory_sha256")
        and inventory.get("tree_regular_file_count")
        == tree_summary["regular_count"]
        and inventory.get("tree_directory_count_below_outer")
        == tree_summary["directory_count"]
        and inventory.get("tree_logical_regular_file_bytes")
        == tree_summary["logical_bytes"]
        and inventory.get("regular_file_double_sha256_same_fd_recomputed")
        is True
        and inventory.get("regular_file_ctime_stable_across_both_hash_passes")
        is True
        and inventory.get("recursive_tree_full_second_replay_equal") is True
        and inventory.get("same_device_no_symlink_single_link_regular_enforced")
        is True
        and inventory.get("no_unexpected_direct_entries") is True
        and type(mount) is dict
        and set(mount)
        == {
            "mount_namespace",
            "mountinfo_file_sha256",
            "owning_mount",
            "scratch_or_descendant_mount_points",
        }
        and re.fullmatch(r"mnt:\[[0-9]+\]", str(mount.get("mount_namespace")))
        is not None
        and SHA256_RE.fullmatch(str(mount.get("mountinfo_file_sha256")))
        is not None
        and mount.get("scratch_or_descendant_mount_points") == []
        and type(mount.get("owning_mount")) is dict
        and set(mount["owning_mount"])
        == {
            "mount_id",
            "parent_mount_id",
            "major_minor",
            "mount_root",
            "mount_point",
            "mount_options",
            "filesystem_type",
            "mount_source",
            "super_options",
            "mount_namespace",
        }
        and mount["owning_mount"].get("major_minor")
        == CHILD_SCRATCH_MOUNT_MAJOR_MINOR
        and mount["owning_mount"].get("mount_root") == "/"
        and mount["owning_mount"].get("mount_point") == CHILD_SCRATCH_MOUNT_POINT
        and mount["owning_mount"].get("filesystem_type")
        == CHILD_SCRATCH_MOUNT_FILESYSTEM
        and mount["owning_mount"].get("mount_source") == CHILD_SCRATCH_MOUNT_SOURCE
        and mount["owning_mount"].get("mount_namespace")
        == mount.get("mount_namespace")
        and "rw" in str(mount["owning_mount"].get("mount_options", "")).split(",")
        and type(census) is dict
        and set(census)
        == {
            "supervisor",
            "attestation_process",
            "same_cgroup_process_count",
            "same_cgroup_processes",
            "cgroup_procs_path",
            "stable_membership_before",
            "stable_membership_after",
            "identities_and_start_ticks_replayed_stably",
            "unexpected_same_cgroup_process_count",
            "numbered_steps",
            "sole_numbered_step",
            "cgroup_census_before_terminal_retention",
        }
        and type(census.get("same_cgroup_processes")) is list
        and len(census["same_cgroup_processes"])
        == census.get("same_cgroup_process_count")
        and len(census["same_cgroup_processes"]) >= 2
        and all(
            type(row) is dict
            and set(row)
            == {"pid", "state", "parent_pid", "start_ticks", "cgroup_v2_path"}
            and type(row.get("pid")) is int
            and row["pid"] > 1
            and type(row.get("parent_pid")) is int
            and row["parent_pid"] >= 0
            and type(row.get("start_ticks")) is int
            and row["start_ticks"] > 0
            and re.fullmatch(r"[A-Z]", str(row.get("state"))) is not None
            and row.get("state") != "Z"
            and str(row.get("cgroup_v2_path", "")).startswith("/")
            for row in census["same_cgroup_processes"]
        )
        and census.get("stable_membership_before")
        == census.get("stable_membership_after")
        == sorted(row["pid"] for row in census["same_cgroup_processes"])
        and census.get("identities_and_start_ticks_replayed_stably") is True
        and census.get("unexpected_same_cgroup_process_count") == 0
        and census.get("numbered_steps") == [census.get("sole_numbered_step")]
        and census.get("sole_numbered_step")
        == f"{HOLDER_JOB}.{runtime['slurm_step_id']}"
        and census.get("cgroup_census_before_terminal_retention") is True
        and type(capacity) is dict
        and set(capacity)
        == {
            "observation_node",
            "observation_scope",
            "filesystem_total_bytes",
            "filesystem_available_bytes_at_terminal_seal",
            "filesystem_total_inodes",
            "filesystem_used_inodes_at_terminal_seal",
            "pre_r6_read_only_observation_available_bytes",
            "pre_r6_read_only_observation_used_inodes",
            "pre_r6_read_only_observation_node",
            "pre_r6_read_only_observation_source",
            "host_wide_values_are_not_tree_usage",
            "future_capacity_guaranteed",
        }
        and capacity.get("observation_node") == HOLDER_NODE
        and capacity.get("observation_scope")
        == "host_wide_ext4_statvfs_at_child_terminal_seal"
        and capacity.get("filesystem_total_bytes") == 470_343_073_792
        and type(capacity.get("filesystem_available_bytes_at_terminal_seal"))
        is int
        and 0 < capacity["filesystem_available_bytes_at_terminal_seal"]
        <= capacity["filesystem_total_bytes"]
        and capacity.get("filesystem_total_inodes") == 29_237_248
        and type(capacity.get("filesystem_used_inodes_at_terminal_seal")) is int
        and 0 <= capacity["filesystem_used_inodes_at_terminal_seal"]
        <= capacity["filesystem_total_inodes"]
        and capacity.get("pre_r6_read_only_observation_available_bytes")
        == 297_197_441_024
        and capacity.get("pre_r6_read_only_observation_used_inodes") == 1_389_837
        and capacity.get("pre_r6_read_only_observation_node") == HOLDER_NODE
        and capacity.get("pre_r6_read_only_observation_source")
        == "gpu215 statvfs read-only audit before r6 candidate"
        and capacity.get("host_wide_values_are_not_tree_usage") is True
        and capacity.get("future_capacity_guaranteed") is False
        and value.get("retention_semantics")
        == {
            "retained_at_child_terminal_seal": True,
            "deletion_attempted": False,
            "cleanup_authorized": False,
            "manual_cleanup_authorized_by_release": False,
            "retained_nonreusable": True,
            "parent_physical_replay_allowed": False,
            "future_availability_guaranteed": False,
            "future_content_immutability_guaranteed": False,
            "persistence_after_step_or_reboot_guaranteed": False,
            "cluster_or_admin_cleanup_controlled_by_release": False,
            "second_point_in_time_inventory_equal_to_attestation_observation": True,
            "continuous_immutability_between_observations_guaranteed": False,
            "atomic_filesystem_snapshot_performed": False,
        }
        and type(value.get("retained_inventory")) is dict
        and SHA256_RE.fullmatch(
            str(value["retained_inventory"].get("tree_inventory_sha256"))
        )
        is not None
        and type(value.get("mount_snapshot")) is dict
        and value["mount_snapshot"].get("scratch_or_descendant_mount_points") == []
        and type(value.get("second_terminal_cgroup_census")) is dict
        and value["second_terminal_cgroup_census"].get(
            "unexpected_same_cgroup_process_count"
        )
        == 0
        and type(value.get("host_capacity_observation")) is dict
        and value["host_capacity_observation"].get("filesystem_total_bytes")
        == 470_343_073_792
        and value["host_capacity_observation"].get("filesystem_total_inodes")
        == 29_237_248
        and value["host_capacity_observation"].get(
            "host_wide_values_are_not_tree_usage"
        )
        is True
        and value["host_capacity_observation"].get("future_capacity_guaranteed")
        is False
        and value.get("formal_candidate_count") == 2
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "child scratch retained-terminal field/authority closure differs",
    )
    return value


def validate_child_scratch_retained_terminal(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Parent receipt/shared-media replay without child /tmp or /proc access."""

    controller_plan, controller_plan_path, controller_plan_sha, exact2 = (
        load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    )
    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "retained parent scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    _validate_child_scratch_prepare_postretention(prepare)
    preflight, preflight_path, preflight_sha = load_json(
        args.compute_preflight,
        "retained parent compute preflight",
        args.expected_compute_preflight_sha256,
    )
    _validate_compute_preflight_postretention(preflight)
    task_bind, task_bind_path, task_bind_sha = load_json(
        args.task_scratch_bind,
        "retained parent task bind",
        args.expected_task_scratch_bind_sha256,
    )
    _validate_child_task_scratch_bind_shape(task_bind)
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "retained parent generation audit",
        args.expected_generation_audit_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "retained parent terminal gate",
        args.expected_terminal_host_gate_sha256,
    )
    _validate_terminal_host_gate_postretention_attested(terminal)
    attestation, attestation_path, attestation_sha = load_json(
        args.physical_attestation,
        "retained parent physical attestation",
        args.expected_physical_attestation_sha256,
    )
    _validate_child_terminal_attestation_shape(attestation)
    retained, retained_path, retained_sha = load_json(
        args.scratch_retained_terminal,
        "child scratch retained-terminal receipt",
        args.expected_scratch_retained_terminal_sha256,
    )
    _validate_child_scratch_retained_terminal_shape(retained)
    ffprobe_bin = validate_ffprobe(PORTABLE_FFPROBE_PATH, PORTABLE_FFPROBE_SHA256)
    resource_module = _load_or_reuse_resource_contract(
        generator.METHOD_ROOT / "tools" / generator.RESOURCE_SPECIALIZED_BASENAME,
        TERMINAL_RESOURCE_CONTRACT_SHA256,
    )
    _validate_exact2_audit_postretention_attested(
        generation, exact2, ffprobe_bin, resource_module
    )
    manifest, manifest_path, manifest_sha = load_json(
        args.blind_review_manifest,
        "retained parent blind-review manifest",
        args.expected_blind_review_manifest_sha256,
    )
    key, key_path, key_sha = load_json(
        args.blind_review_key,
        "retained parent blind-review key",
        args.expected_blind_review_key_sha256,
    )
    pseudo_review = {
        "packet_id": manifest.get("packet_id"),
        "required_ffprobe": manifest.get("required_ffprobe"),
        "generation_audit": _receipt_reference(
            generation, generation_path, generation_sha
        ),
        "blind_review_manifest": _receipt_reference(
            manifest, manifest_path, manifest_sha
        ),
        "sealed_key": _receipt_reference(key, key_path, key_sha),
    }
    reopened_manifest, reopened_key = _validate_blind_packet(
        review=pseudo_review,
        exact2_plan=exact2,
        generation_audit=generation,
        ffprobe_bin=ffprobe_bin,
    )
    prepare_ref = _receipt_reference(prepare, prepare_path, prepare_sha)
    preflight_ref = _receipt_reference(preflight, preflight_path, preflight_sha)
    task_bind_ref = _receipt_reference(task_bind, task_bind_path, task_bind_sha)
    generation_ref = _receipt_reference(generation, generation_path, generation_sha)
    terminal_ref = _receipt_reference(terminal, terminal_path, terminal_sha)
    attestation_ref = _receipt_reference(
        attestation, attestation_path, attestation_sha
    )
    require(
        prepare["controller_plan"]
        == retained["controller_plan"]
        == attestation["controller_plan"]
        == {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }
        and retained["runtime"]
        == attestation["runtime"]
        == task_bind["runtime"]
        == preflight["runtime"]
        == prepare["runtime"]
        and preflight["scratch_prepare"] == prepare_ref
        and task_bind["scratch_prepare"] == prepare_ref
        and task_bind["compute_preflight"] == preflight_ref
        and generation["compute_preflight"] == preflight_ref
        and generation["task_scratch_bind"] == task_bind_ref
        and terminal["compute_preflight"] == preflight_ref
        and attestation["scratch_prepare"] == prepare_ref
        and attestation["compute_preflight"] == preflight_ref
        and attestation["task_scratch_bind"] == task_bind_ref
        and attestation["generation_audit"] == generation_ref
        and attestation["terminal_host_gate"] == terminal_ref
        and attestation["scratch_outer"] == prepare["scratch_root"]
        and attestation["scratch_inner"] == task_bind["scratch_inner"]
        and attestation["renderer_load_lock"] == task_bind["renderer_load_lock"]
        and all(
            attestation["scratch_inventory"]["outer_probe"].get(field)
            == task_bind["retained_probe_file"].get(field)
            for field in task_bind["retained_probe_file"]
        )
        and attestation["blind_review_manifest"]
        == _receipt_reference(manifest, manifest_path, manifest_sha)
        and attestation["blind_review_key"]
        == _receipt_reference(key, key_path, key_sha)
        and reopened_manifest == manifest
        and reopened_key == key
        and retained["scratch_prepare"] == prepare_ref
        and retained["compute_preflight"] == preflight_ref
        and retained["task_scratch_bind"] == task_bind_ref
        and retained["generation_audit"] == generation_ref
        and retained["terminal_host_gate"] == terminal_ref
        and retained["physical_attestation"] == attestation_ref
        and retained["retained_inventory"] == attestation["scratch_inventory"]
        and retained["scratch_outer_creation_identity"] == prepare["scratch_root"]
        and retained["scratch_inner_creation_identity"] == task_bind["scratch_inner"]
        and retained["renderer_load_lock_creation_identity"]
        == task_bind["renderer_load_lock"],
        "retained parent prepare/compute/task/generation/terminal chain differs",
    )
    return {**retained, "_path": str(retained_path), "_file_sha256": retained_sha}


def _terminal_marker_chain(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Replay trusted terminal refs without child scratch or PID access."""

    controller_plan, controller_path, controller_sha, _ = load_controller_plan(
        args.controller_plan, args.expected_controller_plan_sha256
    )
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "terminal marker generation audit",
        args.expected_generation_audit_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "terminal marker host gate",
        args.expected_terminal_host_gate_sha256,
    )
    attestation, attestation_path, attestation_sha = load_json(
        args.physical_attestation,
        "terminal marker physical attestation",
        args.expected_physical_attestation_sha256,
    )
    _validate_child_terminal_attestation_shape(attestation)
    retained, retained_path, retained_sha = load_json(
        args.scratch_retained_terminal,
        "terminal marker scratch-retained receipt",
        args.expected_scratch_retained_terminal_sha256,
    )
    _validate_child_scratch_retained_terminal_shape(retained)
    manifest, manifest_path, manifest_sha = load_json(
        args.blind_review_manifest,
        "terminal marker blind manifest",
        args.expected_blind_review_manifest_sha256,
    )
    key, key_path, key_sha = load_json(
        args.blind_review_key,
        "terminal marker blind key",
        args.expected_blind_review_key_sha256,
    )
    validate_child_scratch_retained_terminal(
        argparse.Namespace(
            controller_plan=str(controller_path),
            expected_controller_plan_sha256=controller_sha,
            scratch_prepare=attestation["scratch_prepare"]["path"],
            expected_scratch_prepare_sha256=attestation["scratch_prepare"][
                "file_sha256"
            ],
            compute_preflight=attestation["compute_preflight"]["path"],
            expected_compute_preflight_sha256=attestation[
                "compute_preflight"
            ]["file_sha256"],
            task_scratch_bind=attestation["task_scratch_bind"]["path"],
            expected_task_scratch_bind_sha256=attestation[
                "task_scratch_bind"
            ]["file_sha256"],
            generation_audit=str(generation_path),
            expected_generation_audit_sha256=generation_sha,
            terminal_host_gate=str(terminal_path),
            expected_terminal_host_gate_sha256=terminal_sha,
            physical_attestation=str(attestation_path),
            expected_physical_attestation_sha256=attestation_sha,
            scratch_retained_terminal=str(retained_path),
            expected_scratch_retained_terminal_sha256=retained_sha,
            blind_review_manifest=str(manifest_path),
            expected_blind_review_manifest_sha256=manifest_sha,
            blind_review_key=str(key_path),
            expected_blind_review_key_sha256=key_sha,
        )
    )
    controller_ref = {
        "path": str(controller_path),
        "file_sha256": controller_sha,
        "plan_digest": controller_plan["plan_digest"],
    }
    refs = {
        "generation_audit": _receipt_reference(
            generation, generation_path, generation_sha
        ),
        "terminal_host_gate": _receipt_reference(
            terminal, terminal_path, terminal_sha
        ),
        "physical_attestation": _receipt_reference(
            attestation, attestation_path, attestation_sha
        ),
        "scratch_retained_terminal": _receipt_reference(
            retained, retained_path, retained_sha
        ),
        "blind_review_manifest": _receipt_reference(
            manifest, manifest_path, manifest_sha
        ),
        "blind_review_key": _receipt_reference(key, key_path, key_sha),
    }
    require(
        attestation["controller_plan"] == retained["controller_plan"] == controller_ref
        and attestation["generation_audit"] == refs["generation_audit"]
        and attestation["terminal_host_gate"] == refs["terminal_host_gate"]
        and attestation["blind_review_manifest"] == refs["blind_review_manifest"]
        and attestation["blind_review_key"] == refs["blind_review_key"]
        and retained["physical_attestation"] == refs["physical_attestation"],
        "terminal marker trusted reference chain differs",
    )
    return {
        "controller_plan": controller_ref,
        **refs,
        "runtime": attestation["runtime"],
        "packet_id": manifest["packet_id"],
    }


def _validate_child_terminal_ready_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, CHILD_TERMINAL_READY_SCHEMA, "child terminal ready")
    receipt_ref_fields = {"path", "file_sha256", "receipt_digest"}
    reference_names = {
        "generation_audit",
        "terminal_host_gate",
        "physical_attestation",
        "scratch_retained_terminal",
        "blind_review_manifest",
        "blind_review_key",
    }
    require(
        set(value)
        == {
            "schema_version",
            "controller_plan",
            *reference_names,
            "runtime",
            "packet_id",
            "retained_at_child_terminal_seal",
            "srun_exit_observed_by_child",
            "parent_step_gone_observed_by_child",
            "formal_candidate_count",
            "diagnostic_task_count",
            "optimizer_authorized",
            "receipt_digest",
        }
        and type(value.get("controller_plan")) is dict
        and set(value["controller_plan"])
        == {"path", "file_sha256", "plan_digest"}
        and all(
            type(value.get(name)) is dict
            and set(value[name]) == receipt_ref_fields
            for name in reference_names
        )
        and type(value.get("runtime")) is dict
        and re.fullmatch(r"[0-9a-f]{32}", str(value.get("packet_id"))) is not None
        and value.get("retained_at_child_terminal_seal") is True
        and value.get("srun_exit_observed_by_child") is False
        and value.get("parent_step_gone_observed_by_child") is False
        and value.get("formal_candidate_count") == 2
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "child terminal-ready field closure differs",
    )
    return value


def seal_child_terminal_ready(args: argparse.Namespace) -> Mapping[str, Any]:
    chain = _terminal_marker_chain(args)
    output = Path(args.output)
    _shared_terminal_marker_run_root(
        output, CHILD_TERMINAL_READY_BASENAME, "child terminal-ready"
    )
    unsigned = {
        "schema_version": CHILD_TERMINAL_READY_SCHEMA,
        **chain,
        "retained_at_child_terminal_seal": True,
        "srun_exit_observed_by_child": False,
        "parent_step_gone_observed_by_child": False,
        "formal_candidate_count": 2,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    _validate_child_terminal_ready_shape(value)
    _write_shared_terminal_marker(output, value)
    return value


def _parent_holder_and_step_observation() -> Mapping[str, Any]:
    try:
        holder = subprocess.run(
            ["scontrol", "show", "job", "-o", HOLDER_JOB],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        steps = subprocess.run(
            ["squeue", "-s", "-j", HOLDER_JOB, "-h", "-o", "%i"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmsIncompleteExact2ControllerError(
            "parent cannot replay holder/numbered-step state"
        ) from error
    holder_line = holder.stdout.strip()
    numbered = sorted(
        line.strip()
        for line in steps.stdout.splitlines()
        if re.fullmatch(rf"{HOLDER_JOB}\.[0-9]+", line.strip())
    )
    require(
        len(holder_line.splitlines()) == 1
        and re.search(r"(?:^| )JobId=136140(?: |$)", holder_line) is not None
        and re.search(r"(?:^| )JobState=RUNNING(?: |$)", holder_line) is not None
        and re.search(r"(?:^| )UserId=[^ ]+\(2012\)(?: |$)", holder_line)
        is not None
        and re.search(
            rf"(?:^| )NodeList={re.escape(HOLDER_NODE)}(?: |$)", holder_line
        )
        is not None
        and numbered == [],
        "parent holder changed or numbered child still exists",
    )
    return {
        "job_id": HOLDER_JOB,
        "job_state": "RUNNING",
        "node": HOLDER_NODE,
        "owner_uid": 2012,
        "numbered_steps": [],
        "parent_job_untouched": True,
    }


_PARENT_GENERATION_CHAIN_NAMES = (
    "generation_audit",
    "terminal_host_gate",
    "physical_attestation",
    "scratch_retained_terminal",
    "blind_review_manifest",
    "blind_review_key",
)
_PARENT_GENERATION_COPY_FIELDS = (
    "controller_plan",
    *_PARENT_GENERATION_CHAIN_NAMES,
    "runtime",
    "packet_id",
    "child_terminal_ready",
    "srun_exit_status",
    "holder_after_srun",
    "generation_success",
    "review_pending",
    "experiment_completion",
    "parent_child_tmp_or_proc_physical_replay_performed",
    "formal_candidate_count",
    "diagnostic_task_count",
    "optimizer_authorized",
)


def _validate_parent_generation_common_shape(
    value: Mapping[str, Any], *, precommit: bool
) -> Mapping[str, Any]:
    schema = (
        PARENT_GENERATION_PRECOMMIT_SCHEMA
        if precommit
        else PARENT_GENERATION_STATUS_SCHEMA
    )
    _validate_signed(value, schema, "parent generation status")
    extras = (
        {
            "full_shared_chain_replay_completed",
            "holder_and_numbered_step_replayed",
            "publication_pending",
        }
        if precommit
        else {
            "parent_generation_precommit",
            "publication_from_precommit_only",
            "external_or_large_file_replay_during_publication",
        }
    )
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    runtime = value.get("runtime")
    holder = value.get("holder_after_srun")
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            *_PARENT_GENERATION_COPY_FIELDS,
            *extras,
            "receipt_digest",
        }
        and type(value.get("controller_plan")) is dict
        and set(value["controller_plan"])
        == {"path", "file_sha256", "plan_digest"}
        and Path(str(value["controller_plan"].get("path"))).is_absolute()
        and SHA256_RE.fullmatch(
            str(value["controller_plan"].get("file_sha256"))
        )
        is not None
        and SHA256_RE.fullmatch(str(value["controller_plan"].get("plan_digest")))
        is not None
        and all(
            type(value.get(name)) is dict
            and set(value[name]) == reference_fields
            and Path(str(value[name].get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(value[name].get("file_sha256")))
            is not None
            and SHA256_RE.fullmatch(str(value[name].get("receipt_digest")))
            is not None
            for name in (*_PARENT_GENERATION_CHAIN_NAMES, "child_terminal_ready")
        )
        and type(runtime) is dict
        and set(runtime)
        == {
            "slurm_job_id",
            "slurm_step_id",
            "hostname",
            "sole_numbered_compute_child_required",
        }
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and str(runtime.get("slurm_step_id", "")).isdecimal()
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and re.fullmatch(r"[0-9a-f]{32}", str(value.get("packet_id"))) is not None
        and value.get("srun_exit_status") == 0
        and holder
        == {
            "job_id": HOLDER_JOB,
            "job_state": "RUNNING",
            "node": HOLDER_NODE,
            "owner_uid": 2012,
            "numbered_steps": [],
            "parent_job_untouched": True,
        }
        and value.get("generation_success") is True
        and value.get("review_pending") is True
        and value.get("experiment_completion") is False
        and value.get("parent_child_tmp_or_proc_physical_replay_performed") is False
        and value.get("formal_candidate_count") == 2
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False
        and (
            (
                value.get("full_shared_chain_replay_completed") is True
                and value.get("holder_and_numbered_step_replayed") is True
                and value.get("publication_pending") is True
            )
            if precommit
            else (
                type(value.get("parent_generation_precommit")) is dict
                and set(value["parent_generation_precommit"]) == reference_fields
                and Path(
                    str(value["parent_generation_precommit"].get("path"))
                ).is_absolute()
                and SHA256_RE.fullmatch(
                    str(value["parent_generation_precommit"].get("file_sha256"))
                )
                is not None
                and SHA256_RE.fullmatch(
                    str(value["parent_generation_precommit"].get("receipt_digest"))
                )
                is not None
                and value.get("publication_from_precommit_only") is True
                and value.get("external_or_large_file_replay_during_publication")
                is False
            )
        ),
        "parent generation status field/authority closure differs",
    )
    return value


def _parent_generation_status_unsigned(
    precommit: Mapping[str, Any],
    precommit_reference: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "schema_version": PARENT_GENERATION_STATUS_SCHEMA,
        **{field: precommit[field] for field in _PARENT_GENERATION_COPY_FIELDS},
        "parent_generation_precommit": precommit_reference,
        "publication_from_precommit_only": True,
        "external_or_large_file_replay_during_publication": False,
    }


def prepare_parent_generation_status(args: argparse.Namespace) -> Mapping[str, Any]:
    require(args.srun_exit_status == 0, "parent srun exit status is not zero")
    chain = _terminal_marker_chain(args)
    child_ready, child_ready_path, child_ready_sha = load_json(
        args.child_terminal_ready,
        "parent child-terminal-ready receipt",
        args.expected_child_terminal_ready_sha256,
    )
    _validate_child_terminal_ready_shape(child_ready)
    child_run_root = _shared_terminal_marker_run_root(
        child_ready_path,
        CHILD_TERMINAL_READY_BASENAME,
        "parent child-terminal-ready",
    )
    output = Path(args.output)
    require(
        _shared_terminal_marker_run_root(
            output,
            PARENT_GENERATION_PRECOMMIT_BASENAME,
            "parent generation precommit",
        )
        == child_run_root,
        "parent generation precommit and child-ready run roots differ",
    )
    expected_ready = {
        "schema_version": CHILD_TERMINAL_READY_SCHEMA,
        **chain,
        "retained_at_child_terminal_seal": True,
        "srun_exit_observed_by_child": False,
        "parent_step_gone_observed_by_child": False,
        "formal_candidate_count": 2,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    require(
        child_ready == _signed(expected_ready),
        "parent child-terminal-ready chain differs",
    )
    holder = _parent_holder_and_step_observation()
    unsigned = {
        "schema_version": PARENT_GENERATION_PRECOMMIT_SCHEMA,
        **chain,
        "child_terminal_ready": _receipt_reference(
            child_ready, child_ready_path, child_ready_sha
        ),
        "srun_exit_status": 0,
        "holder_after_srun": holder,
        "generation_success": True,
        "review_pending": True,
        "experiment_completion": False,
        "parent_child_tmp_or_proc_physical_replay_performed": False,
        "formal_candidate_count": 2,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
        "full_shared_chain_replay_completed": True,
        "holder_and_numbered_step_replayed": True,
        "publication_pending": True,
    }
    value = _signed(unsigned)
    _validate_parent_generation_common_shape(value, precommit=True)
    _write_shared_terminal_marker(output, value)
    return value


def _prepare_parent_generation_status_value(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], Path, Mapping[str, Any], Path, str]:
    precommit, precommit_path, precommit_sha = load_json(
        args.parent_generation_precommit,
        "parent generation precommit",
        args.expected_parent_generation_precommit_sha256,
    )
    _validate_parent_generation_common_shape(precommit, precommit=True)
    precommit_run_root = _shared_terminal_marker_run_root(
        precommit_path,
        PARENT_GENERATION_PRECOMMIT_BASENAME,
        "parent generation precommit",
    )
    child_ready_path = Path(str(precommit["child_terminal_ready"]["path"]))
    require(
        _shared_terminal_marker_run_root(
            child_ready_path,
            CHILD_TERMINAL_READY_BASENAME,
            "precommit child-terminal-ready",
        )
        == precommit_run_root,
        "parent generation precommit and child-ready run roots differ",
    )
    output = Path(args.output)
    require(
        _shared_terminal_marker_run_root(
            output,
            PARENT_GENERATION_STATUS_BASENAME,
            "parent generation status",
        )
        == precommit_run_root,
        "parent generation status and precommit run roots differ",
    )
    value = _signed(
        _parent_generation_status_unsigned(
            precommit, _receipt_reference(precommit, precommit_path, precommit_sha)
        )
    )
    _validate_parent_generation_common_shape(value, precommit=False)
    return value, output, precommit, precommit_path, precommit_sha


def resident_publish_parent_generation_status(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Prepare completely, announce READY, then cross one held-fd commit edge."""

    value, output, precommit, precommit_path, precommit_sha = (
        _prepare_parent_generation_status_value(args)
    )
    prepared = _PreparedResidentSharedMarker(output, value)
    held_precommit: Optional[_HeldResidentReceipt] = None
    previous_alarm_handler: Any = None

    def publish_timeout(_signal_number: int, _frame: Any) -> NoReturn:
        # Keep a short second alarm armed while the publisher performs its
        # same-inode best-effort shared-marker rollback.
        signal.alarm(PARENT_GENERATION_ROLLBACK_TIMEOUT_SECONDS)
        raise ArmsIncompleteExact2ControllerError(
            "resident parent-status held-fd publication timed out"
        )

    try:
        held_precommit = _HeldResidentReceipt(
            precommit_path,
            precommit_sha,
            canonical_json_bytes(precommit) + b"\n",
            prepared.logs_descriptor,
            prepared.parent,
        )
        previous_alarm_handler = signal.signal(signal.SIGALRM, publish_timeout)
        # Expensive/fallible preparation is complete.  Block terminal signals
        # before READY so the commit token cannot race process death; parent
        # abort after READY is represented by closing stdin and delivering EOF.
        signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {signal.SIGINT, signal.SIGTERM, signal.SIGHUP},
        )
        print(
            PARENT_GENERATION_PUBLISH_READY_PREFIX + prepared.file_sha256,
            flush=True,
        )
        token = sys.stdin.buffer.readline(
            len(PARENT_GENERATION_PUBLISH_COMMIT_TOKEN) + 1
        )
        require(
            token == PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
            "resident parent-status commit token differs",
        )
        signal.alarm(PARENT_GENERATION_PUBLISH_TIMEOUT_SECONDS)
        try:
            held_precommit.replay()
            observed = prepared.publish()
            held_precommit.replay()
            prepared.replay_published_marker()
        finally:
            signal.alarm(0)
        print(PARENT_GENERATION_PUBLISH_ACK_PREFIX + observed, flush=True)
        return value
    except Exception:
        if previous_alarm_handler is not None:
            signal.alarm(PARENT_GENERATION_ROLLBACK_TIMEOUT_SECONDS)
        try:
            prepared.rollback_published_marker()
        finally:
            signal.alarm(0)
        raise
    finally:
        signal.alarm(0)
        if previous_alarm_handler is not None:
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        if held_precommit is not None:
            held_precommit.close()
        prepared.close()


def _load_parent_generation_status(
    path: str, expected_sha256: str
) -> tuple[
    Mapping[str, Any],
    Path,
    str,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    status, status_path, status_sha = load_json(
        path, "parent generation status", expected_sha256
    )
    _validate_parent_generation_common_shape(status, precommit=False)
    status_run_root = _shared_terminal_marker_run_root(
        status_path,
        PARENT_GENERATION_STATUS_BASENAME,
        "parent generation status",
    )
    precommit_ref = status["parent_generation_precommit"]
    precommit, precommit_path, precommit_sha = load_json(
        precommit_ref["path"],
        "parent generation precommit reference",
        precommit_ref["file_sha256"],
    )
    _validate_parent_generation_common_shape(precommit, precommit=True)
    require(
        _shared_terminal_marker_run_root(
            precommit_path,
            PARENT_GENERATION_PRECOMMIT_BASENAME,
            "parent generation precommit reference",
        )
        == status_run_root,
        "parent generation status and precommit run roots differ",
    )
    child_ref = status["child_terminal_ready"]
    child_ready, child_ready_path, child_ready_sha = load_json(
        child_ref["path"],
        "parent status child-terminal-ready reference",
        child_ref["file_sha256"],
    )
    _validate_child_terminal_ready_shape(child_ready)
    require(
        _shared_terminal_marker_run_root(
            child_ready_path,
            CHILD_TERMINAL_READY_BASENAME,
            "parent status child-terminal-ready reference",
        )
        == status_run_root,
        "parent generation status and child-ready run roots differ",
    )
    require(
        precommit_ref
        == _receipt_reference(precommit, precommit_path, precommit_sha)
        and child_ref
        == _receipt_reference(child_ready, child_ready_path, child_ready_sha)
        and status
        == _signed(_parent_generation_status_unsigned(precommit, precommit_ref))
        and all(
            status[field] == child_ready[field]
            for field in (
                "controller_plan",
                *_PARENT_GENERATION_CHAIN_NAMES,
                "runtime",
                "packet_id",
                "formal_candidate_count",
                "diagnostic_task_count",
                "optimizer_authorized",
            )
        ),
        "parent generation status precommit/child-ready chain differs",
    )
    return status, status_path, status_sha, precommit, child_ready


def validate_parent_generation_status(args: argparse.Namespace) -> Mapping[str, Any]:
    status, _, _, _, _ = _load_parent_generation_status(
        args.parent_generation_status,
        args.expected_parent_generation_status_sha256,
    )
    return status


CHILD_SCRATCH_FAILURE_PHASES = frozenset(
    {
        "prepare",
        "compute",
        "inner",
        "monitor",
        "smoke",
        "generation",
        "audit",
        "blind",
        "terminal",
        "attestation",
        "retained-terminal",
        "signal-int",
        "signal-term",
        "signal-hup",
    }
)


def seal_child_scratch_failure(args: argparse.Namespace) -> Mapping[str, Any]:
    """Best-effort truthful failure evidence; this function never deletes."""

    require(
        args.failure_phase in CHILD_SCRATCH_FAILURE_PHASES
        and type(args.exit_status) is int
        and 1 <= args.exit_status <= 255,
        "child scratch failure phase/status differs",
    )
    prepare, prepare_path, prepare_sha = load_json(
        args.scratch_prepare,
        "failure scratch prepare",
        args.expected_scratch_prepare_sha256,
    )
    _validate_child_scratch_prepare_shape(prepare)
    outer_identity_present = True
    try:
        _replay_child_scratch_prepare_physical(
            prepare, require_initial_link_count=False
        )
    except ArmsIncompleteExact2ControllerError:
        outer_identity_present = False
    compute_ref: Optional[Mapping[str, str]] = None
    if args.compute_preflight is not None:
        require(
            args.expected_compute_preflight_sha256 is not None,
            "failure compute-preflight expected SHA is absent",
        )
        preflight, preflight_path, preflight_sha = load_json(
            args.compute_preflight,
            "failure compute preflight",
            args.expected_compute_preflight_sha256,
        )
        _validate_signed(preflight, COMPUTE_PREFLIGHT_SCHEMA, "failure compute preflight")
        compute_ref = _receipt_reference(preflight, preflight_path, preflight_sha)
    else:
        require(
            args.expected_compute_preflight_sha256 is None,
            "failure compute-preflight SHA without path is forbidden",
        )
    task_bind_ref: Optional[Mapping[str, str]] = None
    inner_identity_present: Optional[bool] = None
    inner_identity: Optional[Mapping[str, Any]] = None
    if args.task_scratch_bind is not None:
        require(
            args.expected_task_scratch_bind_sha256 is not None,
            "failure task-bind expected SHA is absent",
        )
        task_bind, task_bind_path, task_bind_sha = load_json(
            args.task_scratch_bind,
            "failure task scratch bind",
            args.expected_task_scratch_bind_sha256,
        )
        _validate_child_task_scratch_bind_shape(task_bind)
        task_bind_ref = _receipt_reference(task_bind, task_bind_path, task_bind_sha)
        inner_identity = task_bind["scratch_inner"]
        inner_identity_present = True
        try:
            _replay_child_task_scratch_bind_physical(task_bind, prepare)
        except ArmsIncompleteExact2ControllerError:
            inner_identity_present = False
    else:
        require(
            args.expected_task_scratch_bind_sha256 is None,
            "failure task-bind SHA without path is forbidden",
        )
    unsigned = {
        "schema_version": CHILD_SCRATCH_FAILURE_SCHEMA,
        "authority": CHILD_SCRATCH_AUTHORITY,
        "runtime": prepare["runtime"],
        "scratch_prepare": _receipt_reference(prepare, prepare_path, prepare_sha),
        "compute_preflight": compute_ref,
        "task_scratch_bind": task_bind_ref,
        "failure": {
            "phase": args.failure_phase,
            "exit_status": args.exit_status,
            "signal_phase": args.failure_phase.startswith("signal-"),
        },
        "retention_observation": {
            "outer_original_identity_present": outer_identity_present,
            "inner_original_identity": inner_identity,
            "inner_original_identity_present": inner_identity_present,
            "substitute_path_never_deleted": True,
            "deletion_attempted": False,
            "terminal_retained_success_claimed": False,
            "observation_host": socket.gethostname().split(".", 1)[0],
        },
        "retention_semantics": {
            "point_in_time_observation_only": True,
            "future_availability_guaranteed": False,
            "future_content_immutability_guaranteed": False,
            "manual_cleanup_authorized_by_release": False,
            "scratch_reusable": False,
        },
        "failure_receipt_is_best_effort": True,
        "parent_may_not_physically_replay_child_tmp": True,
        "optimizer_authorized": False,
    }
    value = _signed(unsigned)
    write_create_only(Path(args.output), value)
    return value


def _validate_child_scratch_failure_shape(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, CHILD_SCRATCH_FAILURE_SCHEMA, "child scratch failure")
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    retention = value.get("retention_observation")
    failure = value.get("failure")
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "authority",
            "runtime",
            "scratch_prepare",
            "compute_preflight",
            "task_scratch_bind",
            "failure",
            "retention_observation",
            "retention_semantics",
            "failure_receipt_is_best_effort",
            "parent_may_not_physically_replay_child_tmp",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(value.get("scratch_prepare")) is dict
        and set(value["scratch_prepare"]) == reference_fields
        and all(
            reference is None
            or (
                type(reference) is dict
                and set(reference) == reference_fields
                and Path(str(reference.get("path"))).is_absolute()
                and SHA256_RE.fullmatch(str(reference.get("file_sha256"))) is not None
                and SHA256_RE.fullmatch(str(reference.get("receipt_digest"))) is not None
            )
            for reference in (
                value.get("compute_preflight"),
                value.get("task_scratch_bind"),
            )
        )
        and type(failure) is dict
        and set(failure) == {"phase", "exit_status", "signal_phase"}
        and failure.get("phase") in CHILD_SCRATCH_FAILURE_PHASES
        and type(failure.get("exit_status")) is int
        and 1 <= failure["exit_status"] <= 255
        and failure.get("signal_phase")
        is str(failure.get("phase", "")).startswith("signal-")
        and type(retention) is dict
        and set(retention)
        == {
            "outer_original_identity_present",
            "inner_original_identity",
            "inner_original_identity_present",
            "substitute_path_never_deleted",
            "deletion_attempted",
            "terminal_retained_success_claimed",
            "observation_host",
        }
        and type(retention.get("outer_original_identity_present")) is bool
        and (
            retention.get("inner_original_identity_present") is None
            or type(retention.get("inner_original_identity_present")) is bool
        )
        and retention.get("substitute_path_never_deleted") is True
        and retention.get("deletion_attempted") is False
        and retention.get("terminal_retained_success_claimed") is False
        and value.get("retention_semantics")
        == {
            "point_in_time_observation_only": True,
            "future_availability_guaranteed": False,
            "future_content_immutability_guaranteed": False,
            "manual_cleanup_authorized_by_release": False,
            "scratch_reusable": False,
        }
        and value.get("failure_receipt_is_best_effort") is True
        and value.get("parent_may_not_physically_replay_child_tmp") is True
        and value.get("optimizer_authorized") is False,
        "child scratch failure field/authority closure differs",
    )
    return value


def validate_child_scratch_failure(args: argparse.Namespace) -> Mapping[str, Any]:
    value, path, observed = load_json(
        args.scratch_failure,
        "child scratch failure receipt",
        args.expected_scratch_failure_sha256,
    )
    _validate_child_scratch_failure_shape(value)
    return {**value, "_path": str(path), "_file_sha256": observed}


def seal_completion(args: argparse.Namespace) -> Mapping[str, Any]:
    controller_plan, plan_path, plan_sha, exact2 = load_controller_plan(
        args.controller_plan, args.expected_controller_plan_sha256
    )
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "exact2 generation audit",
        args.expected_generation_audit_sha256,
    )
    review, review_path, review_sha = load_json(
        args.review_admission,
        "independent full81 review admission",
        args.expected_review_admission_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "terminal exact2 host gate",
        args.expected_terminal_host_gate_sha256,
    )
    attestation, attestation_path, attestation_sha = load_json(
        args.child_terminal_physical_attestation,
        "completion child terminal physical attestation",
        args.expected_child_terminal_physical_attestation_sha256,
    )
    retained, retained_path, retained_sha = load_json(
        args.child_scratch_retained_terminal,
        "completion child scratch retained-terminal",
        args.expected_child_scratch_retained_terminal_sha256,
    )
    parent_status, parent_status_path, parent_status_sha, _, _ = (
        _load_parent_generation_status(
            args.parent_generation_status,
            args.expected_parent_generation_status_sha256,
        )
    )
    _validate_child_terminal_attestation_shape(attestation)
    _validate_child_scratch_retained_terminal_shape(retained)
    validate_child_scratch_retained_terminal(
        argparse.Namespace(
            controller_plan=str(plan_path),
            expected_controller_plan_sha256=plan_sha,
            scratch_prepare=attestation["scratch_prepare"]["path"],
            expected_scratch_prepare_sha256=attestation["scratch_prepare"][
                "file_sha256"
            ],
            compute_preflight=attestation["compute_preflight"]["path"],
            expected_compute_preflight_sha256=attestation[
                "compute_preflight"
            ]["file_sha256"],
            task_scratch_bind=attestation["task_scratch_bind"]["path"],
            expected_task_scratch_bind_sha256=attestation[
                "task_scratch_bind"
            ]["file_sha256"],
            generation_audit=str(generation_path),
            expected_generation_audit_sha256=generation_sha,
            terminal_host_gate=str(terminal_path),
            expected_terminal_host_gate_sha256=terminal_sha,
            physical_attestation=str(attestation_path),
            expected_physical_attestation_sha256=attestation_sha,
            scratch_retained_terminal=str(retained_path),
            expected_scratch_retained_terminal_sha256=retained_sha,
            blind_review_manifest=review["blind_review_manifest"]["path"],
            expected_blind_review_manifest_sha256=review[
                "blind_review_manifest"
            ]["file_sha256"],
            blind_review_key=review["sealed_key"]["path"],
            expected_blind_review_key_sha256=review["sealed_key"][
                "file_sha256"
            ],
        )
    )
    ffprobe_bin = validate_ffprobe(args.ffprobe_bin, args.expected_ffprobe_sha256)
    resource_module = _load_or_reuse_resource_contract(
        generator.METHOD_ROOT
        / "tools"
        / generator.RESOURCE_SPECIALIZED_BASENAME,
        TERMINAL_RESOURCE_CONTRACT_SHA256,
    )
    _validate_exact2_audit_postretention_attested(
        generation, exact2, ffprobe_bin, resource_module
    )
    validate_review_admission(review, exact2, generation, ffprobe_bin)
    _validate_terminal_host_gate_postretention_attested(
        terminal, resource_module
    )
    require(
        generation["compute_preflight"] == terminal["compute_preflight"],
        "completion compute-preflight chain differs across generation and terminal",
    )
    require(
        attestation["generation_audit"]
        == _receipt_reference(generation, generation_path, generation_sha)
        and attestation["terminal_host_gate"]
        == _receipt_reference(terminal, terminal_path, terminal_sha)
        and retained["physical_attestation"]
        == _receipt_reference(attestation, attestation_path, attestation_sha)
        and parent_status["controller_plan"]
        == {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }
        and parent_status["generation_audit"]
        == _receipt_reference(generation, generation_path, generation_sha)
        and parent_status["terminal_host_gate"]
        == _receipt_reference(terminal, terminal_path, terminal_sha)
        and parent_status["physical_attestation"]
        == _receipt_reference(attestation, attestation_path, attestation_sha)
        and parent_status["scratch_retained_terminal"]
        == _receipt_reference(retained, retained_path, retained_sha)
        and parent_status["blind_review_manifest"]
        == review["blind_review_manifest"]
        and parent_status["blind_review_key"] == review["sealed_key"]
        and parent_status["runtime"] == attestation["runtime"],
        "completion attestation/terminal-retention chain differs",
    )
    proof_rows = generation["cross_run_same_gaussian_pair_proofs"]
    require(
        len(proof_rows) == 2
        and [row["seed"] for row in proof_rows] == [2026080821, 2026080921]
        and all(
            row["action_incomplete_official_gaussian_tensor_values_byte_equal"]
            is True
            and row["physical_artifacts_reopened"] is True
            and row["physical_safetensors_safe_open_recomputed"] is True
            for row in proof_rows
        ),
        "completion Gaussian proof closure differs",
    )
    unsigned = {
        "schema_version": COMPLETION_SCHEMA,
        "controller_plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        },
        "generation_audit": {
            "path": str(generation_path),
            "file_sha256": generation_sha,
            "receipt_digest": generation["receipt_digest"],
        },
        "independent_full81_review_admission": {
            "path": str(review_path),
            "file_sha256": review_sha,
            "receipt_digest": review["receipt_digest"],
            "reviewer_receipt_id": review["reviewer_receipt_id"],
            "packet_id": review["packet_id"],
            "blind_review_manifest": review["blind_review_manifest"],
            "sealed_key": review["sealed_key"],
            "new_incomplete_pass": [2, 2],
        },
        "terminal_host_gate": {
            "path": str(terminal_path),
            "file_sha256": terminal_sha,
            "receipt_digest": terminal["receipt_digest"],
        },
        "child_terminal_physical_attestation": _receipt_reference(
            attestation, attestation_path, attestation_sha
        ),
        "child_scratch_retained_terminal": _receipt_reference(
            retained, retained_path, retained_sha
        ),
        "parent_generation_status": _receipt_reference(
            parent_status, parent_status_path, parent_status_sha
        ),
        "post_child_exit_scratch_validation_mode": "attested_retention_receipt_only",
        "child_tmp_or_proc_physical_replay_performed": False,
        "scratch_current_retention_claimed_by_parent": False,
        "child_point_in_time_retention_receipt_only": True,
        "compute_preflight": terminal["compute_preflight"],
        "rank_resource_scratch_binding": generation[
            "rank_resource_scratch_binding"
        ],
        "portable_ffprobe_revalidated": {
            "path": str(ffprobe_bin),
            "file_sha256": PORTABLE_FFPROBE_SHA256,
            "caller_override_allowed": False,
        },
        "dataset": plan_contract.DATASET,
        "formal_candidate_count": 2,
        "external_action_count": 2,
        "comparator_cell_count": 2,
        "diagnostic_task_count": 0,
        "cross_run_same_gaussian_pair_proofs": proof_rows,
        "completion_physically_recomputed_proof_count": 2,
        "r4_teacher_pair_media_authority_ready": True,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "materializer_pending": True,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "training_performed": False,
    }
    value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(Path(args.output), value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_terminal_marker_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--controller-plan", required=True)
        command.add_argument("--expected-controller-plan-sha256", required=True)
        command.add_argument("--generation-audit", required=True)
        command.add_argument("--expected-generation-audit-sha256", required=True)
        command.add_argument("--terminal-host-gate", required=True)
        command.add_argument("--expected-terminal-host-gate-sha256", required=True)
        command.add_argument("--physical-attestation", required=True)
        command.add_argument("--expected-physical-attestation-sha256", required=True)
        command.add_argument("--scratch-retained-terminal", required=True)
        command.add_argument(
            "--expected-scratch-retained-terminal-sha256", required=True
        )
        command.add_argument("--blind-review-manifest", required=True)
        command.add_argument(
            "--expected-blind-review-manifest-sha256", required=True
        )
        command.add_argument("--blind-review-key", required=True)
        command.add_argument("--expected-blind-review-key-sha256", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--method-root", required=True)
    plan.add_argument("--release-manifest", required=True)
    plan.add_argument("--expected-release-manifest-sha256", required=True)
    plan.add_argument("--exact2-plan", required=True)
    plan.add_argument("--expected-exact2-plan-sha256", required=True)
    plan.add_argument("--output", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--controller-plan", required=True)
    validate.add_argument("--expected-controller-plan-sha256", required=True)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--controller-plan", required=True)
    runtime.add_argument("--expected-controller-plan-sha256", required=True)
    scratch_prepare = commands.add_parser("prepare-child-scratch")
    scratch_prepare.add_argument("--controller-plan", required=True)
    scratch_prepare.add_argument(
        "--expected-controller-plan-sha256", required=True
    )
    scratch_prepare.add_argument("--output", required=True)
    preflight = commands.add_parser("seal-compute-preflight")
    preflight.add_argument("--controller-plan", required=True)
    preflight.add_argument("--expected-controller-plan-sha256", required=True)
    preflight.add_argument("--ffprobe-bin", required=True)
    preflight.add_argument("--expected-ffprobe-sha256", required=True)
    preflight.add_argument("--external-action-mp4-seed1", required=True)
    preflight.add_argument("--external-action-mp4-seed2", required=True)
    preflight.add_argument("--scratch-prepare", required=True)
    preflight.add_argument("--expected-scratch-prepare-sha256", required=True)
    preflight.add_argument("--output", required=True)
    task_bind = commands.add_parser("create-and-bind-child-task-scratch")
    task_bind.add_argument("--scratch-prepare", required=True)
    task_bind.add_argument("--expected-scratch-prepare-sha256", required=True)
    task_bind.add_argument("--compute-preflight", required=True)
    task_bind.add_argument("--expected-compute-preflight-sha256", required=True)
    task_bind.add_argument("--output", required=True)
    task_validate = commands.add_parser("validate-child-task-scratch-bind")
    task_validate.add_argument("--scratch-prepare", required=True)
    task_validate.add_argument(
        "--expected-scratch-prepare-sha256", required=True
    )
    task_validate.add_argument("--task-scratch-bind", required=True)
    task_validate.add_argument(
        "--expected-task-scratch-bind-sha256", required=True
    )
    terminal = commands.add_parser("seal-terminal-host-gate")
    terminal.add_argument("--compute-preflight", required=True)
    terminal.add_argument("--expected-compute-preflight-sha256", required=True)
    terminal.add_argument("--resource-contract", required=True)
    terminal.add_argument("--expected-resource-contract-sha256", required=True)
    terminal.add_argument("--monitor-start-receipt", required=True)
    terminal.add_argument("--expected-monitor-start-receipt-sha256", required=True)
    terminal.add_argument("--monitor-exit-status", type=int, required=True)
    terminal.add_argument("--output", required=True)
    attestation = commands.add_parser(
        "seal-child-terminal-physical-attestation"
    )
    attestation.add_argument("--controller-plan", required=True)
    attestation.add_argument(
        "--expected-controller-plan-sha256", required=True
    )
    attestation.add_argument("--generation-audit", required=True)
    attestation.add_argument("--expected-generation-audit-sha256", required=True)
    attestation.add_argument("--terminal-host-gate", required=True)
    attestation.add_argument(
        "--expected-terminal-host-gate-sha256", required=True
    )
    attestation.add_argument("--blind-review-manifest", required=True)
    attestation.add_argument(
        "--expected-blind-review-manifest-sha256", required=True
    )
    attestation.add_argument("--blind-review-key", required=True)
    attestation.add_argument(
        "--expected-blind-review-key-sha256", required=True
    )
    attestation.add_argument("--scratch-prepare", required=True)
    attestation.add_argument(
        "--expected-scratch-prepare-sha256", required=True
    )
    attestation.add_argument("--compute-preflight", required=True)
    attestation.add_argument(
        "--expected-compute-preflight-sha256", required=True
    )
    attestation.add_argument("--task-scratch-bind", required=True)
    attestation.add_argument(
        "--expected-task-scratch-bind-sha256", required=True
    )
    attestation.add_argument("--ffprobe-bin", required=True)
    attestation.add_argument("--expected-ffprobe-sha256", required=True)
    attestation.add_argument("--supervisor-pid", type=int, required=True)
    attestation.add_argument("--output", required=True)
    retained = commands.add_parser("seal-child-scratch-retained-terminal")
    retained.add_argument("--controller-plan", required=True)
    retained.add_argument("--expected-controller-plan-sha256", required=True)
    retained.add_argument("--scratch-prepare", required=True)
    retained.add_argument("--expected-scratch-prepare-sha256", required=True)
    retained.add_argument("--compute-preflight", required=True)
    retained.add_argument("--expected-compute-preflight-sha256", required=True)
    retained.add_argument("--task-scratch-bind", required=True)
    retained.add_argument("--expected-task-scratch-bind-sha256", required=True)
    retained.add_argument("--generation-audit", required=True)
    retained.add_argument("--expected-generation-audit-sha256", required=True)
    retained.add_argument("--terminal-host-gate", required=True)
    retained.add_argument("--expected-terminal-host-gate-sha256", required=True)
    retained.add_argument("--physical-attestation", required=True)
    retained.add_argument(
        "--expected-physical-attestation-sha256", required=True
    )
    retained.add_argument("--supervisor-pid", type=int, required=True)
    retained.add_argument("--output", required=True)
    failure = commands.add_parser("seal-child-scratch-failure")
    failure.add_argument("--scratch-prepare", required=True)
    failure.add_argument("--expected-scratch-prepare-sha256", required=True)
    failure.add_argument("--compute-preflight")
    failure.add_argument("--expected-compute-preflight-sha256")
    failure.add_argument("--task-scratch-bind")
    failure.add_argument("--expected-task-scratch-bind-sha256")
    failure.add_argument(
        "--failure-phase", choices=sorted(CHILD_SCRATCH_FAILURE_PHASES), required=True
    )
    failure.add_argument("--exit-status", type=int, required=True)
    failure.add_argument("--output", required=True)
    failure_validate = commands.add_parser("validate-child-scratch-failure")
    failure_validate.add_argument("--scratch-failure", required=True)
    failure_validate.add_argument(
        "--expected-scratch-failure-sha256", required=True
    )
    retained_validate = commands.add_parser(
        "validate-child-scratch-retained-terminal"
    )
    retained_validate.add_argument("--controller-plan", required=True)
    retained_validate.add_argument(
        "--expected-controller-plan-sha256", required=True
    )
    retained_validate.add_argument("--scratch-prepare", required=True)
    retained_validate.add_argument(
        "--expected-scratch-prepare-sha256", required=True
    )
    retained_validate.add_argument("--compute-preflight", required=True)
    retained_validate.add_argument(
        "--expected-compute-preflight-sha256", required=True
    )
    retained_validate.add_argument("--task-scratch-bind", required=True)
    retained_validate.add_argument(
        "--expected-task-scratch-bind-sha256", required=True
    )
    retained_validate.add_argument("--generation-audit", required=True)
    retained_validate.add_argument(
        "--expected-generation-audit-sha256", required=True
    )
    retained_validate.add_argument("--terminal-host-gate", required=True)
    retained_validate.add_argument(
        "--expected-terminal-host-gate-sha256", required=True
    )
    retained_validate.add_argument("--physical-attestation", required=True)
    retained_validate.add_argument(
        "--expected-physical-attestation-sha256", required=True
    )
    retained_validate.add_argument("--scratch-retained-terminal", required=True)
    retained_validate.add_argument(
        "--expected-scratch-retained-terminal-sha256", required=True
    )
    retained_validate.add_argument("--blind-review-manifest", required=True)
    retained_validate.add_argument(
        "--expected-blind-review-manifest-sha256", required=True
    )
    retained_validate.add_argument("--blind-review-key", required=True)
    retained_validate.add_argument(
        "--expected-blind-review-key-sha256", required=True
    )
    child_ready = commands.add_parser("seal-child-terminal-ready")
    add_terminal_marker_arguments(child_ready)
    child_ready.add_argument("--output", required=True)
    parent_precommit = commands.add_parser("prepare-parent-generation-status")
    add_terminal_marker_arguments(parent_precommit)
    parent_precommit.add_argument("--child-terminal-ready", required=True)
    parent_precommit.add_argument(
        "--expected-child-terminal-ready-sha256", required=True
    )
    parent_precommit.add_argument("--srun-exit-status", type=int, required=True)
    parent_precommit.add_argument("--output", required=True)
    parent_publish = commands.add_parser(
        "resident-publish-parent-generation-status"
    )
    parent_publish.add_argument("--parent-generation-precommit", required=True)
    parent_publish.add_argument(
        "--expected-parent-generation-precommit-sha256", required=True
    )
    parent_publish.add_argument("--output", required=True)
    parent_status_validate = commands.add_parser(
        "validate-parent-generation-status"
    )
    parent_status_validate.add_argument("--parent-generation-status", required=True)
    parent_status_validate.add_argument(
        "--expected-parent-generation-status-sha256", required=True
    )
    blind = commands.add_parser("seal-blind-review-input")
    blind.add_argument("--controller-plan", required=True)
    blind.add_argument("--expected-controller-plan-sha256", required=True)
    blind.add_argument("--generation-audit", required=True)
    blind.add_argument("--expected-generation-audit-sha256", required=True)
    blind.add_argument("--ffprobe-bin", required=True)
    blind.add_argument("--expected-ffprobe-sha256", required=True)
    blind.add_argument("--output-dir", required=True)
    complete = commands.add_parser("seal-completion")
    complete.add_argument("--controller-plan", required=True)
    complete.add_argument("--expected-controller-plan-sha256", required=True)
    complete.add_argument("--generation-audit", required=True)
    complete.add_argument("--expected-generation-audit-sha256", required=True)
    complete.add_argument("--review-admission", required=True)
    complete.add_argument("--expected-review-admission-sha256", required=True)
    complete.add_argument("--terminal-host-gate", required=True)
    complete.add_argument("--expected-terminal-host-gate-sha256", required=True)
    complete.add_argument("--child-terminal-physical-attestation", required=True)
    complete.add_argument(
        "--expected-child-terminal-physical-attestation-sha256", required=True
    )
    complete.add_argument("--child-scratch-retained-terminal", required=True)
    complete.add_argument(
        "--expected-child-scratch-retained-terminal-sha256", required=True
    )
    complete.add_argument("--parent-generation-status", required=True)
    complete.add_argument(
        "--expected-parent-generation-status-sha256", required=True
    )
    complete.add_argument("--ffprobe-bin", required=True)
    complete.add_argument("--expected-ffprobe-sha256", required=True)
    complete.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        value = build_controller_plan(args)
    elif args.command == "validate-plan":
        value, _, _, _ = load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    elif args.command == "validate-runtime":
        value = validate_runtime_environment(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    elif args.command == "prepare-child-scratch":
        value = prepare_child_scratch(args)
    elif args.command == "seal-compute-preflight":
        value = seal_compute_preflight(args)
    elif args.command == "create-and-bind-child-task-scratch":
        value = create_and_bind_child_task_scratch(args)
    elif args.command == "validate-child-task-scratch-bind":
        value = validate_child_task_scratch_bind(args)
    elif args.command == "seal-terminal-host-gate":
        value = seal_terminal_host_gate(args)
    elif args.command == "seal-child-terminal-physical-attestation":
        value = seal_child_terminal_physical_attestation(args)
    elif args.command == "seal-child-scratch-retained-terminal":
        value = seal_child_scratch_retained_terminal(args)
    elif args.command == "seal-child-scratch-failure":
        value = seal_child_scratch_failure(args)
    elif args.command == "validate-child-scratch-failure":
        value = validate_child_scratch_failure(args)
    elif args.command == "validate-child-scratch-retained-terminal":
        value = validate_child_scratch_retained_terminal(args)
    elif args.command == "seal-child-terminal-ready":
        value = seal_child_terminal_ready(args)
    elif args.command == "prepare-parent-generation-status":
        value = prepare_parent_generation_status(args)
    elif args.command == "resident-publish-parent-generation-status":
        value = resident_publish_parent_generation_status(args)
        return 0
    elif args.command == "validate-parent-generation-status":
        value = validate_parent_generation_status(args)
    elif args.command == "seal-blind-review-input":
        value = seal_blind_review_input(args)
    else:
        value = seal_completion(args)
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmsIncompleteExact2ControllerError",
    "BLIND_REVIEW_KEY_SCHEMA",
    "BLIND_REVIEW_MANIFEST_SCHEMA",
    "COMPLETION_SCHEMA",
    "COMPUTE_PREFLIGHT_SCHEMA",
    "CONTROLLER_PLAN_SCHEMA",
    "FULL81_INDEX_SHA256",
    "PORTABLE_FFPROBE_PATH",
    "PORTABLE_FFPROBE_SHA256",
    "PARENT_GENERATION_PRECOMMIT_SCHEMA",
    "PARENT_GENERATION_STATUS_SCHEMA",
    "REVIEW_SCHEMA",
    "TERMINAL_RESOURCE_CONTRACT_SHA256",
    "TERMINAL_RESOURCE_MODULE_NAME",
    "TERMINAL_HOST_GATE_SCHEMA",
    "_derive_terminal_host_gate_from_physical",
    "_load_or_reuse_resource_contract",
    "load_controller_plan",
    "object_sha256",
    "probe_exact81_25fps",
    "seal_blind_review_input",
    "seal_compute_preflight",
    "seal_completion",
    "validate_compute_preflight",
    "validate_exact2_audit",
    "validate_review_admission",
    "validate_terminal_host_gate",
]
